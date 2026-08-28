# skew_sweep_2d.py -- 2D sweep of inventory-exit aggressiveness x OBI-defensive
# skew, to test whether shortening hold time (the confirmed driver of diffusive
# markout) and avoiding adverse-imbalance fills improves net edge.
#
# THE HYPOTHESIS (stated so the data can refute it): median hold of 4-6 minutes
# is NOT high-frequency; the markout loss is dominated by DIFFUSION accrued while
# holding (diff_mko >> jump_mko, and it grows with hold time). Therefore:
#   AXIS 1 (exit_ticks_inside): when loaded, post the EXIT side N ticks inside
#     the touch -> fills that leave faster -> shorter hold -> less diffusive
#     markout. Cost: less capture (posting inside earns less than the half-spread).
#   AXIS 2 (obi_defensive): OBI-at-fill is systematically against us; quotes
#     ignore book imbalance. Suppressing the side the book leans against should
#     raise (toward 0) the direction-signed OBI-at-fill and cut adverse selection.
# If net_bps rises, one/both work; if it falls, holding longer was cheaper and the
# markout is not hold-time-fixable -- either way we learn the mechanism.
#
# GRID: exit_ticks_inside in {0,1,2,3}  x  obi_defensive in {False, True}
#       = 8 configs. N=0 & obi_defensive=False == the validated baseline
#       (byte-identical placement -- unit-tested in micro_mm.py).
# CLIP: production 3x only (the capacity question is already settled; this sweep
#       is about the hold-time/adverse-selection mechanism, not capacity).
#
# PER-CONFIG REPORT (per bucket): capture, markout (jump/diff), fees, liq_loss,
#   net-bps, median+mean hold, realized median ticks-inside-the-touch, raw
#   OBI-at-fill AND direction-signed OBI-at-fill split by buy/sell fills, number
#   of trades, and average shares per trade.
#
# Runtime: ~8x a single-config decomposition. Expect a few hours; heartbeat +
# ETA are printed. Run with caffeinate.
#
#   caffeinate -is python3 existing_mm_live/skew_sweep_2d.py

# paths + timing
from pathlib import Path
import time
from datetime import datetime
# parallelism
import multiprocessing as mp
# arrays + frames
import numpy as np
import pandas as pd
# significance
from scipy import stats
# harness + driver
import mm_harness as H
import run_legacy_mm as R
# reuse the validated decomposition helpers (mid asof + jump/diff split)
from spot_capture_markout_decomp import _mid_at, _split_move

# store paths
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ config ---------------------------------------
# the top-10 production book
NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
# single production clip (capacity already settled; this is a mechanism sweep)
CLIP_MULT = 3.0
# trailing median-trade-size window (days)
TRAIL_DAYS = 10
# AXIS 1: FROZEN at the proven winner (exit_ticks=1 dominated et0 everywhere).
EXIT_TICKS = [1]
# AXIS 2: FROZEN at the winner (obi_defensive=True paid at exit_ticks=1).
OBI_MODES = [True]
# AXIS 3: microprice -- REMOVED. The smoke proved OBI-in-quotes as a fair-value
# lean is strongly destructive (mp+ ~ -5 bps and thr% ~98% vs mp- ~ +6 bps): it
# quotes into the pressure and gets run over. OBI survives ONLY defensively
# (obi_defensive), never as a fair-value shift. Mid-only from here.
MICRO_MODES = [False]
# AXIS 4: FROZEN at the winner (tol=0 beat tol=1 at the winning cell).
TOL_MODES = [0.0]
# AXIS 5 (STAGE 3): OFI-defensive window -- None = OFF (the control), else the
# min(N events, T seconds) hybrid from the horserace. Per-bucket best is read
# from the per-bucket tables (one window per config; buckets independent).
OFI_MODES = [None, (20, 2.0), (50, 5.0), (70, 7.0), (90, 9.0)]
# AXIS 6 (STAGE 3): engage threshold on the NORMALIZED [-1,+1] trailing OFI.
# Only applies when a window is on (the OFF config is not duplicated per thresh).
OFI_THRESH = [0.20, 0.40]
# inventory threshold (lots) beyond which the tick-exit engages
EXIT_INV_THRESHOLD = 1.0
# OBI-defensive engage threshold (|imb-0.5|) and widen ticks
OBI_DEF_THRESH = 0.15
OBI_DEF_TICKS = 1.0
# jump detector (same as the decomposition)
JUMP_K = 4.0
# smoke: first N days (None -> full ~207)
# smoke: first N days. DEFAULT 5 -> a ~5-minute run that prints the
# reconciliation anchor so any bug shows up BEFORE the 3-hour full run.
# Set to None ONLY after the smoke's reconciliation anchor reads OK for all 8
# configs. (Hard lesson: a 3-hour run was burned on an unverified sweep.)
SMOKE_DAYS = 5
# workers
WORKERS = None
# -----------------------------------------------------------------------------

# worker globals
_G = {}


# worker init
def _init_worker(calib):
    _G.update(calib)


# CANONICAL session-bucket for a fill timestamp, matching the futures runners
# (futures_mm_run.py:200-202) and build_volume_profile EXACTLY. Keyed off the
# day's session SEGMENTS (break-aware: correct on Friday-split / Ramadan days),
# in exchange-MILLISECONDS. This is the assignment that was MISSING from the
# fills pipeline -- fills reached attribution with a default 'middle', so every
# bucket but middle was empty. Applying it here tags each fill by its own time.
def _bucket_of(t, segs):
    # session open (first segment start) and close (last segment end), in ms
    open_ms = segs[0][0]
    close_ms = segs[-1][1]
    # first 15 minutes after the open
    if t < open_ms + 15 * 60000:
        return "first15"
    # last 15 minutes before the close
    if t >= close_ms - 15 * 60000:
        return "last15"
    # the 45 minutes before last15 (i.e. 60..15 min before the close)
    if t >= close_ms - 60 * 60000:
        return "preclose45"
    # everything in between
    return "middle"


# one (date, symbol, exit_ticks, obi_defensive, use_micro) cell
def _process(args):
    # unpack the work tuple
    date, sym, exit_ticks, obi_def, use_micro, tol, ofi_win, ofi_th = args
    # calibration
    scales = _G["scales"]; profiles = _G["profiles"]; windows = _G["windows"]
    segments = _G["segments"]; all_dates = _G["all_dates"]; tstats = _G["tstats"]
    # segments for the day
    segs = segments.get(str(date))
    if segs is None:
        return None
    # trailing median trade size
    med = H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
    if med is None or med <= 0:
        return None
    if sym not in scales or sym not in profiles:
        return None
    # datasets
    dsets = R.open_datasets(date)
    if dsets is None:
        return None
    # clip
    clip = max(1, int(round(CLIP_MULT * med)))
    # build params, then INJECT the sweep axes via overrides (baseline when
    # exit_ticks=0 and obi_def=False -- unit-tested byte-identical)
    params = H.build_micro_params(
        clip, scales[sym], profiles[sym], windows.get(sym, (5.0, 1.0)), segs,
        overrides={"exit_ticks_inside": exit_ticks,
                   "exit_inv_threshold": EXIT_INV_THRESHOLD,
                   "obi_defensive": obi_def,
                   "obi_defensive_thresh": OBI_DEF_THRESH,
                   "obi_defensive_ticks": OBI_DEF_TICKS,
                   # AXIS 3: OBI in the quotes (microprice fair). False = the
                   # production baseline (build_micro_params sets it False).
                   "use_microprice": use_micro,
                   # AXIS 4: pegging hysteresis (0 = chase, N = hold until N-tick drift)
                   "tol_ticks": tol,
                   # AXIS 5+6 (STAGE 3): OFI-defensive retreat. None window = OFF.
                   "ofi_defensive": ofi_win is not None,
                   "ofi_window_ev": (ofi_win[0] if ofi_win is not None else 50),
                   "ofi_window_s": (ofi_win[1] if ofi_win is not None else 5.0),
                   "ofi_defensive_thresh": ofi_th})
    # run
    dr = H.run_symbol_day(date, sym, dsets, params)
    if dr is None or dr.pnl() is None:
        return None
    # mid + touch series from the equity log
    eq = pd.DataFrame(dr.equity) if len(dr.equity) else pd.DataFrame()
    if len(eq) == 0 or "mid" not in eq.columns:
        return None
    eq = eq.sort_values("t")
    eq_t = eq["t"].to_numpy(dtype=float)
    eq_mid = eq["mid"].to_numpy(dtype=float)
    # best bid/ask for ticks-inside + OBI (if logged)
    have_touch = "bb" in eq.columns and "ba" in eq.columns
    eq_bb = eq["bb"].to_numpy(dtype=float) if have_touch else None
    eq_ba = eq["ba"].to_numpy(dtype=float) if have_touch else None
    # OBI at fill (if logged)
    have_obi = "obi_5" in eq.columns and "obi_deep" in eq.columns
    eq_obi5 = eq["obi_5"].to_numpy(dtype=float) if have_obi else None
    eq_obi_d = eq["obi_deep"].to_numpy(dtype=float) if have_obi else None
    # tick size for ticks-inside
    tick = params.get("tick", 0.01)
    # per-bucket accumulators
    per = {b: {"capture": 0.0, "markout": 0.0, "fee": 0.0, "liq_fee": 0.0,
               "jump_markout": 0.0, "diff_markout": 0.0, "liq_loss": 0.0,
               # LIQUIDATION P&L (Option A): capture + markout of round trips
               # whose CLOSING fill was a forced EOD liquidation, booked to the
               # OPENING lot's bucket. Split out of capture/markout so net_bps
               # separates quoting P&L from forced-exit P&L. net still =
               # cap + mko + liq_cap + liq_mko - fee (telescopes to the same total).
               "liq_cap": 0.0, "liq_mko": 0.0,
               "opened_notional": 0.0, "opened_qty": 0.0, "holds": [],
               "fills": 0, "trades": 0, "shares": 0.0,
               "n_through": 0, "n_atq": 0,
               # raw OBI at fill (book-signed) and DIRECTION-signed OBI at fill,
               # split by buy vs sell fills (the corrected metric)
               "obi5_raw_sum": 0.0, "obi5_signed_sum": 0.0,
               "obi5_buy_sum": 0.0, "obi5_buy_n": 0,
               "obi5_sell_sum": 0.0, "obi5_sell_n": 0, "obi_n": 0,
               # realized ticks inside the touch on our fills
               "ticks_inside": []}
           for b in H.BUCKETS}
    # asof index helper
    def _asof_idx(t):
        pos = np.searchsorted(eq_t, t, side="right") - 1
        return pos if pos >= 0 else None
    # FIFO queue
    open_lots = []
    fills = dr.fills.to_dict("records") if isinstance(dr.fills, pd.DataFrame) \
        else list(dr.fills)
    # walk fills
    for fl in fills:
        side = fl["side"]; px = float(fl["px"]); qty = float(fl["qty"])
        t = float(fl["t"])
        # assign the session bucket BY TIMESTAMP (the fills' own 'bucket' column
        # is never populated by the engine -> was defaulting everything to middle)
        b = _bucket_of(t, segs)
        # LIQUIDATION fills (Stage 1 EOD book-walk) are forced exits, NOT quoting
        # decisions -- they must NOT pollute the quoting DIAGNOSTICS (trade counts,
        # reason-mix, OBI-at-fill, ticks-inside), which is what was turning last15
        # OBI to NaN and inflating last15 trade counts. They DO still flow through
        # the FIFO P&L matcher below (that is what makes reconciliation exact).
        is_liq_fill = fl.get("reason", "") in ("liq", "liq_residual")
        # count trade + shares + fill in its bucket (quoting fills only)
        if b in per and not is_liq_fill:
            per[b]["fills"] += 1
            per[b]["trades"] += 1
            per[b]["shares"] += qty
            # reason mix: through = price moved THROUGH our resting quote (stale);
            # at_queue = queue ahead cleared to us (benign). The tol/microprice
            # axes are adjudicated by how this mix shifts.
            rsn = fl.get("reason", "")
            if rsn == "through":
                per[b]["n_through"] += 1
            elif rsn in ("at_queue", "at_optimistic"):
                per[b]["n_atq"] += 1
        # asof book state at the fill
        idx = _asof_idx(t)
        # OBI at fill (raw + direction-signed) and ticks-inside. Quoting fills
        # only -- liq fills are forced EOD exits, excluded from the diagnostics.
        if idx is not None and b in per and not is_liq_fill:
            # direction sign: +1 if we BOUGHT, -1 if we SOLD -> signed OBI < 0
            # ALWAYS means "book leaned against our fill", regardless of side.
            dir_sign = 1.0 if side == "BUY" else -1.0
            if have_obi:
                # raw book-signed OBI (what the earlier table showed)
                per[b]["obi5_raw_sum"] += eq_obi5[idx]
                # direction-signed OBI (the corrected adverse-selection metric)
                per[b]["obi5_signed_sum"] += dir_sign * eq_obi5[idx]
                # split by side so we can see WHICH side gets picked
                if side == "BUY":
                    per[b]["obi5_buy_sum"] += eq_obi5[idx]
                    per[b]["obi5_buy_n"] += 1
                else:
                    per[b]["obi5_sell_sum"] += eq_obi5[idx]
                    per[b]["obi5_sell_n"] += 1
                per[b]["obi_n"] += 1
            # PLACEMENT vs MID (signed): how far our fill price sat from the mid,
            # in ticks, signed so POSITIVE = passive (we posted inside our side of
            # the mid and earned part of the spread) and NEGATIVE = we CROSSED the
            # mid (a marketable/EOD-unwind fill that PAID the spread). Measuring
            # vs the opposite touch (the old bug) turned a sell that crossed the
            # bid into a nonsense "+26 ticks inside" -- it was really paying to
            # cross. The mid is the neutral reference that makes the sign mean
            # "did we earn or pay the spread on this fill".
            ref = idx - 1 if idx >= 1 else idx
            if have_touch and np.isfinite(eq_bb[ref]) and np.isfinite(eq_ba[ref]) \
                    and eq_ba[ref] > eq_bb[ref]:
                # mid just before the fill
                mid_ref = 0.5 * (eq_bb[ref] + eq_ba[ref])
                # BUY: passive if we bought BELOW mid -> +(mid-px); crossing if above
                # SELL: passive if we sold ABOVE mid -> +(px-mid); crossing if below
                if side == "BUY":
                    ti = (mid_ref - px) / tick
                else:
                    ti = (px - mid_ref) / tick
                # record the SIGNED value (no clamp: negative crossing fills are
                # real information -- they show the exit mechanism paying to cross)
                per[b]["ticks_inside"].append(ti)
        # mid at fill for capture. LIQ fills carry their OWN mid0 (the closing mid
        # the engine recorded at the walk); intraday fills do not, so fall back to
        # the equity-log lookup. Using the fill's own mid0 for liq fills makes the
        # liquidation capture telescope against the exact mid the engine used ->
        # zero residual (no modelled EOD block, no plug).
        _fmid0 = fl.get("mid0", None)
        if _fmid0 is not None and np.isfinite(_fmid0):
            mid_at_fill = float(_fmid0)
        else:
            mid_at_fill = _mid_at(eq_t, eq_mid, t)
        cap_sign = 1.0 if side == "BUY" else -1.0
        cap = (cap_sign * (mid_at_fill - px) * qty
               if np.isfinite(mid_at_fill) else 0.0)
        # opens if empty/same side
        if not open_lots or open_lots[0]["side"] == side:
            # opening capture PER SHARE, stored so Option A can move the matched
            # share of it into liq_cap if this lot later closes via liquidation.
            cap_ps = (cap / qty) if qty > 0 else 0.0
            open_lots.append({"qty": qty, "px": px, "bucket": b, "t": t,
                              "side": side, "mid_at_fill": mid_at_fill,
                              "cap_ps": cap_ps})
            if b in per:
                per[b]["opened_notional"] += qty * px
                per[b]["opened_qty"] += qty
                per[b]["capture"] += cap
            continue
        # opposite side -> close FIFO
        remaining = qty
        while remaining > 1e-9 and open_lots and open_lots[0]["side"] != side:
            lot = open_lots[0]
            matched = min(remaining, lot["qty"])
            m_open = lot.get("mid_at_fill", np.nan)
            # closing-fill exit mid: use the fill's OWN mid0 when present (liq
            # fills), else the equity-log lookup (intraday). Consistent with the
            # capture mid above -> telescoping is exact.
            m_exit = mid_at_fill
            if np.isfinite(m_open) and np.isfinite(m_exit):
                open_sign = 1.0 if lot["side"] == "BUY" else -1.0
                mko = open_sign * (m_exit - m_open) * matched
                jm, _ = _split_move(eq_t, eq_mid, lot["t"], t)
                mko_jump = open_sign * jm * matched
                mko_diff = mko - mko_jump
            else:
                mko = 0.0; mko_jump = 0.0; mko_diff = 0.0
            # round-trip fees (two legs). EXCEPTION: a liq_residual closing fill
            # is a haircut MARK, not a traded exit -- the engine charges it no fee
            # (residual_mark has none), so charging one here would re-introduce a
            # residual. The OPENING leg's fee always applies; the closing leg's
            # fee applies only when the close is a real trade.
            close_is_mark = (fl.get("reason", "") == "liq_residual")
            fee = H.fee_for(lot["px"], matched) + (
                0.0 if close_is_mark else H.fee_for(px, matched))
            ob = lot["bucket"]
            if ob in per:
                # Option A: route the whole liquidated round trip to the liq
                # columns (booked to the OPENING bucket). Non-liq closes go to
                # the normal capture/markout columns as before.
                if is_liq_fill:
                    # markout of this liquidated round trip
                    per[ob]["liq_mko"] += mko
                    # closing-leg capture (exit half-spread), pro-rated
                    if qty > 0:
                        per[ob]["liq_cap"] += cap * (matched / qty)
                    # MOVE the matched share of the OPENING capture (already
                    # booked to capture at open) out of capture and into liq_cap,
                    # so the FULL round trip sits in the liq columns (Option A).
                    _open_cap_matched = lot.get("cap_ps", 0.0) * matched
                    per[ob]["capture"] -= _open_cap_matched
                    per[ob]["liq_cap"] += _open_cap_matched
                    # jump/diff still tracked (for completeness) under normal cols
                    per[ob]["jump_markout"] += mko_jump
                    per[ob]["diff_markout"] += mko_diff
                    # fee on the liquidation leg goes to the fee column as usual
                    per[ob]["fee"] += fee
                    per[ob]["holds"].append(t - lot["t"])
                else:
                    per[ob]["markout"] += mko
                    per[ob]["jump_markout"] += mko_jump
                    per[ob]["diff_markout"] += mko_diff
                    per[ob]["fee"] += fee
                    per[ob]["holds"].append(t - lot["t"])
                    # THE MISSING TERM (root of the -70% unexplained): the CLOSING
                    # leg's capture -- the half-spread the exit fill earns vs the
                    # mid at exit, cap*(matched/qty) pro-rated, booked to the
                    # opening bucket. Identity: engine realized (ps-pb)q - fees =
                    # cap_open (m0-pb)q + cap_close (ps-m1)q + markout (m1-m0)q
                    # - fees   [exact]. Dropping cap_close silently discarded the
                    # exit half-spread (~half of gross capture).
                    if qty > 0:
                        per[ob]["capture"] += cap * (matched / qty)
            lot["qty"] -= matched
            remaining -= matched
            if lot["qty"] <= 1e-9:
                open_lots.pop(0)
        # flip remainder opens the other side
        if remaining > 1e-9:
            # opening capture per share for the flipped remainder
            cap_ps = (cap / qty) if qty > 0 else 0.0
            open_lots.append({"qty": remaining, "px": px, "bucket": b, "t": t,
                              "side": side, "mid_at_fill": mid_at_fill,
                              "cap_ps": cap_ps})
            if b in per:
                per[b]["opened_notional"] += remaining * px
                per[b]["opened_qty"] += remaining
                if qty > 0:
                    per[b]["capture"] += cap * (remaining / qty)
    # ---- EOD RESIDUAL: nothing to model. In Stage 2 the engine emits the EOD
    # book-walk as real liq/liq_residual fills, which the FIFO loop above already
    # matched against the open lots (opposite side, each with its own mid0). So
    # after the loop there should be NO open lots left on a normally-liquidated
    # day. If any remain, that is a genuine anomaly (engine did not flatten) and
    # it stays visible in the reconciliation below as a real residual -- NOT
    # modelled away. The old pro-rata liquidation reconstruction is deleted.
    # ---- RECONCILIATION CHECK (assert, do NOT force): every column above is a
    # measured quantity. Their sum should already equal engine P&L. We compute
    # the gap and store it as a DIAGNOSTIC (liq_loss now means "unexplained
    # residual" -- if the measurements are right it is ~0). It is NOT distributed
    # to make things match; a large value means a real measurement error to fix.
    dec_total = sum(per[b]["capture"] + per[b]["markout"] - per[b]["fee"]
                    for b in H.BUCKETS)
    # the residual gap: if our measurements are correct this is ~0
    recon_gap = float(dr.pnl()) - dec_total
    # store the gap itself (per bucket, pro-rata) purely as a DIAGNOSTIC so the
    # anchor can show whether the measured decomposition actually reconciles
    tot_opened = sum(per[b]["opened_notional"] for b in H.BUCKETS)
    for b in H.BUCKETS:
        w = (per[b]["opened_notional"] / tot_opened) if tot_opened > 0 \
            else (1.0 / len(H.BUCKETS))
        # liq_loss is now the UNEXPLAINED residual (diagnostic), not a plug that
        # defines the answer -- a healthy run has this ~0
        per[b]["liq_loss"] = recon_gap * w
    # bundle
    return {"exit_ticks": exit_ticks, "obi_def": obi_def, "use_micro": use_micro,
            "tol": tol, "ofi_win": ofi_win, "ofi_th": ofi_th, "per": per,
            "daily_pnl": float(dr.pnl()), "date": str(date),
            # per-cell unexplained residual (should be ~0 if measurements right)
            "recon_gap": recon_gap,
            # RAW engine P&L for this cell -- the reconciliation ANCHOR.
            "engine_pnl": float(dr.pnl())}


def main():
    # stamp + timing
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    t0 = time.perf_counter()
    # calibration (once in parent)
    print("pre-pass: calibration + trailing median trade size", flush=True)
    all_dates = R.discover_dates()
    calib = {"scales": H.load_scales(), "profiles": H.load_profiles(),
             "windows": H.load_windows(), "segments": H.load_segments(),
             "all_dates": all_dates,
             "tstats": H.trailing_median_trade_size(all_dates, NAMES, TRAIL_DAYS)}
    # dates
    run_dates = all_dates[TRAIL_DAYS:]
    if SMOKE_DAYS is not None:
        run_dates = run_dates[:SMOKE_DAYS]
    # workers
    nproc = WORKERS or mp.cpu_count()
    # the 8 configs
    # 6-axis config tuples; the OFI-OFF config carries a single dummy thresh
    # (thresholds only matter when a window is on)
    configs = [(et, od, um, tl, ofw, oth)
               for et in EXIT_TICKS for od in OBI_MODES
               for um in MICRO_MODES for tl in TOL_MODES
               for ofw in OFI_MODES
               for oth in (OFI_THRESH if ofw is not None else [OFI_THRESH[0]])]
    # work list: every (date, symbol, exit_ticks, obi_def, use_micro)
    work = [(date, sym, et, od, um, tl, ofw, oth)
            for date in run_dates for sym in NAMES
            for (et, od, um, tl, ofw, oth) in configs]
    total = len(work)
    print(f"\nSTAGE-3 OFI sweep (winner frozen: et1/obi+/tol0/mid): "
          f"{len(OFI_MODES)} windows x {len(OFI_THRESH)} thresholds "
          f"(OFF deduped) = {len(configs)} configs", flush=True)
    print(f"  x {len(NAMES)} names x {len(run_dates)} days = {total} cells",
          flush=True)
    print(f"  workers: {nproc}\n", flush=True)
    # per-config accumulators keyed by (exit_ticks, obi_def)
    agg = {(et, od, um, tl, ofw, oth): {b: {"capture": 0.0, "markout": 0.0, "fee": 0.0,
                                  "liq_cap": 0.0, "liq_mko": 0.0,
                          "liq_fee": 0.0, "jump_markout": 0.0,
                          "diff_markout": 0.0, "liq_loss": 0.0,
                          "opened_notional": 0.0, "opened_qty": 0.0,
                          "holds": [], "fills": 0, "trades": 0, "shares": 0.0,
                          "n_through": 0, "n_atq": 0,
                          "obi5_raw_sum": 0.0, "obi5_signed_sum": 0.0,
                          "obi5_buy_sum": 0.0, "obi5_buy_n": 0,
                          "obi5_sell_sum": 0.0, "obi5_sell_n": 0, "obi_n": 0,
                          "ticks_inside": []}
                      for b in H.BUCKETS}
           for (et, od, um, tl, ofw, oth) in configs}
    # per-config daily portfolio markout for significance
    daily_mko = {(et, od, um, tl, ofw, oth): {} for (et, od, um, tl, ofw, oth) in configs}
    # per-config RAW engine P&L accumulator (the reconciliation anchor)
    engine_pnl_agg = {(et, od, um, tl, ofw, oth): 0.0
                      for (et, od, um, tl, ofw, oth) in configs}
    # accumulate the unexplained residual (measured decomposition vs engine P&L);
    # ~0 means the book-walk liquidation attribution is correct, non-zero flags
    # a real measurement error (NOT hidden by a plug anymore)
    recon_gap_agg = {(et, od, um, tl, ofw, oth): 0.0
                     for (et, od, um, tl, ofw, oth) in configs}
    # run
    done = 0
    with mp.Pool(processes=nproc, initializer=_init_worker,
                 initargs=(calib,)) as pool:
        for res in pool.imap_unordered(_process, work, chunksize=4):
            done += 1
            if done % 200 == 0 or done == total:
                el = time.perf_counter() - t0
                print(f"  {done}/{total}  {H._fmt(el)}  "
                      f"ETA {H._fmt(el / done * (total - done))}", flush=True)
            if res is None:
                continue
            key = (res["exit_ticks"], res["obi_def"], res["use_micro"], res["tol"],
               res["ofi_win"], res["ofi_th"])
            for b in H.BUCKETS:
                s = res["per"][b]; d = agg[key][b]
                # sum scalar accumulators
                for k in ("capture", "markout", "fee", "liq_fee", "liq_cap", "liq_mko",
                          "jump_markout", "diff_markout", "liq_loss",
                          "opened_notional", "opened_qty", "fills", "trades",
                          "shares", "n_through", "n_atq",
                          "obi5_raw_sum", "obi5_signed_sum",
                          "obi5_buy_sum", "obi5_buy_n", "obi5_sell_sum",
                          "obi5_sell_n", "obi_n"):
                    d[k] += s[k]
                # extend lists
                d["holds"].extend(s["holds"])
                d["ticks_inside"].extend(s["ticks_inside"])
            # daily portfolio markout
            day = res["date"]
            dsum = sum(res["per"][b]["markout"] for b in H.BUCKETS)
            daily_mko[key][day] = daily_mko[key].get(day, 0.0) + dsum
            # accumulate raw engine P&L for the reconciliation anchor
            engine_pnl_agg[key] += res["engine_pnl"]
            # accumulate the unexplained residual (measured recon quality)
            recon_gap_agg[key] += res["recon_gap"]
    # ---- reporting ----
    print("\n" + "=" * 84)
    print(f"### 2D SKEW SWEEP DONE: {H._fmt(time.perf_counter() - t0)} "
          f"for {total} cells ###")
    print("=" * 84)
    print("\n=== SUMMARY: OFI-defensive configs (winner frozen: et1/obi+/tol0) ===")
    print(f"{'ofi_window':>14s} {'thresh':>7s} {'net_bps':>9s} {'net_PKR':>13s}")
    for (et, od, um, tl, ofw, oth) in configs:
        a = agg[(et, od, um, tl, ofw, oth)]
        # MEASURED net = quoting P&L + liquidation P&L - all fees (reconciles
        # to engine exactly; see the anchor)
        net = sum(a[b]["capture"] + a[b]["markout"]
                  + a[b]["liq_cap"] + a[b]["liq_mko"]
                  - (a[b]["fee"] + a[b]["liq_fee"]) for b in H.BUCKETS)
        on = sum(a[b]["opened_notional"] for b in H.BUCKETS)
        # bps of opened notional
        nbps = (1e4 * net / on) if on > 0 else float("nan")
        # window label: OFF or min(Nev,Ts)
        wlab = "OFF" if ofw is None else f"min{ofw[0]}ev{ofw[1]:.0f}s"
        # threshold label (dash when OFF)
        tlab = "-" if ofw is None else f"{oth:.2f}"
        print(f"{wlab:>14s} {tlab:>7s} {nbps:>9.3f} {net:>13,.0f}")
    # ---- RECONCILIATION ANCHOR: the MEASURED decomposition (capture + markout
    # incl. book-walk liquidation - fees) vs raw engine P&L. NO plug is added
    # here -- 'unexplained' is the honest residual. If the book-walk liquidation
    # attribution is correct it is ~0; a large value is a REAL measurement error
    # to fix, not something to hide. (liq_loss in the tables holds this same
    # residual as a diagnostic, distributed pro-rata.)
    print("\n=== RECONCILIATION ANCHOR: MEASURED decomposition vs engine P&L ===")
    print(f"{'ofi_window':>14s} {'thresh':>7s} {'measured_net':>16s} "
          f"{'engine_pnl':>16s} {'unexplained':>13s} {'pct':>7s} {'ok?':>5s}")
    for (et, od, um, tl, ofw, oth) in configs:
        a = agg[(et, od, um, tl, ofw, oth)]
        # MEASURED net: capture + markout (incl. measured liquidation) - all fees.
        # NOTE: liq_loss is DELIBERATELY EXCLUDED -- it is the diagnostic residual,
        # not part of the measurement. If measurements are right, measured_net
        # already equals engine P&L without it.
        measured_net = sum(a[b]["capture"] + a[b]["markout"]
                           + a[b]["liq_cap"] + a[b]["liq_mko"]
                           - (a[b]["fee"] + a[b]["liq_fee"]) for b in H.BUCKETS)
        eng = engine_pnl_agg[(et, od, um, tl, ofw, oth)]
        # the honest unexplained residual (should be ~0)
        unexplained = measured_net - eng
        pct = (100.0 * unexplained / eng) if abs(eng) > 1 else float("nan")
        # tolerance: within 2% of engine P&L is a sound measurement
        ok = "OK" if abs(pct) < 2.0 or abs(unexplained) < max(1.0, 0.0) \
            else "BAD"
        wlab = "OFF" if ofw is None else f"min{ofw[0]}ev{ofw[1]:.0f}s"
        tlab = "-" if ofw is None else f"{oth:.2f}"
        print(f"{wlab:>14s} {tlab:>7s} {measured_net:>16,.0f} {eng:>16,.0f} "
              f"{unexplained:>13,.0f} {pct:>6.1f}% {ok:>5s}")
    # detailed per-config tables
    for (et, od, um, tl, ofw, oth) in configs:
        a = agg[(et, od, um, tl, ofw, oth)]
        print(f"\n{'=' * 84}")
        wlab = "OFF" if ofw is None else f"min{ofw[0]}ev{ofw[1]:.0f}s"
        print(f"CONFIG: et={et} obi={od} tol={tl}   OFI={wlab}"
              + ("" if ofw is None else f" thresh={oth:.2f}"))
        print("=" * 84)
        # per-bucket detail. NOTE: mko_bps now INCLUDES the measured book-walk
        # liquidation markout for residual lots (real exit price, not a mark).
        # 'unexpl_bps' is the diagnostic residual (measured decomposition vs
        # engine P&L, pro-rata) -- should be ~0 if the book-walk attribution is
        # right. net_bps = cap + mko + liq_cap + liq_mko - fees (MEASURED).
        # liq_bps = the P&L of round trips force-closed at EOD (Option A), booked
        # to the OPENING bucket -- separated so quoting P&L is not muddied by
        # forced-exit P&L.
        print(f"{'bucket':11s} {'cap_bps':>8s} {'mko_bps':>8s} {'jmp_bps':>8s} "
              f"{'dif_bps':>8s} {'fee_bps':>8s} {'liq_bps':>8s} {'unexpl':>8s} "
              f"{'net_bps':>8s} {'med_hld':>8s} {'mean_hld':>8s} {'med_tks_mid':>11s} "
              f"{'thr%':>5s} {'trades':>8s} {'sh/trd':>8s}")
        for b in H.BUCKETS:
            d = a[b]
            on = d["opened_notional"]
            def _bps(x):
                return (1e4 * x / on) if on > 0 else float("nan")
            # all-in measured fee (round-trip + measured liquidation fee)
            fee_all = d["fee"] + d["liq_fee"]
            # liquidation P&L (Option A): capture + markout of round trips whose
            # close was a forced EOD liquidation, booked to the opening bucket.
            liq_pnl = d["liq_cap"] + d["liq_mko"]
            # liquidation slice in bps
            liq_bps = _bps(liq_pnl)
            # MEASURED net: quoting P&L (cap+mko) + liquidation P&L - fees. No
            # plug. net = cap + mko + liq - fee reconciles to engine exactly.
            net_bps = _bps(d["capture"] + d["markout"] + liq_pnl - fee_all)
            # the diagnostic residual (should be ~0)
            unexpl_bps = _bps(d["liq_loss"])
            hs = np.array(d["holds"]) / 1000.0 if d["holds"] else np.array([0.0])
            tks = np.array(d["ticks_inside"]) if d["ticks_inside"] else np.array([0.0])
            sh_per = (d["shares"] / d["trades"]) if d["trades"] > 0 else 0.0
            # through share: fraction of classified fills where price moved
            # THROUGH the resting quote (the staleness signature)
            n_cls = d["n_through"] + d["n_atq"]
            thr_pct = (100.0 * d["n_through"] / n_cls) if n_cls > 0 else float("nan")
            print(f"{b:11s} {_bps(d['capture']):>8.3f} {_bps(d['markout']):>8.3f} "
                  f"{_bps(d['jump_markout']):>8.3f} {_bps(d['diff_markout']):>8.3f} "
                  f"{_bps(-fee_all):>8.3f} {liq_bps:>8.3f} {unexpl_bps:>8.3f} "
                  f"{net_bps:>8.3f} "
                  f"{np.median(hs):>8.1f} {hs.mean():>8.1f} {np.median(tks):>11.2f} "
                  f"{thr_pct:>5.0f} {d['trades']:>8,d} {sh_per:>8.0f}")
        # OBI-at-fill: raw vs direction-signed, split by buy/sell
        print(f"\n{'bucket':11s} {'raw_obi5':>10s} {'signed_obi5':>12s} "
              f"{'buy_obi5':>10s} {'sell_obi5':>10s}  (signed<0 = book against us)")
        for b in H.BUCKETS:
            d = a[b]
            n = d["obi_n"]
            raw = (d["obi5_raw_sum"] / n) if n > 0 else float("nan")
            signed = (d["obi5_signed_sum"] / n) if n > 0 else float("nan")
            buy = (d["obi5_buy_sum"] / d["obi5_buy_n"]) if d["obi5_buy_n"] > 0 else float("nan")
            sell = (d["obi5_sell_sum"] / d["obi5_sell_n"]) if d["obi5_sell_n"] > 0 else float("nan")
            print(f"{b:11s} {raw:>10.4f} {signed:>12.4f} {buy:>10.4f} {sell:>10.4f}")
        # daily markout significance (day as unit)
        dm = pd.Series(daily_mko[(et, od, um, tl, ofw, oth)]).sort_index().to_numpy()
        if len(dm) >= 20:
            t_stat, t_p = stats.ttest_1samp(dm, 0.0, nan_policy="omit")
            dm_nz = dm[dm != 0]
            try:
                w_stat, w_p = stats.wilcoxon(dm_nz)
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
            print(f"\n  daily markout (n={len(dm)}): mean {dm.mean():,.0f}  "
                  f"median {np.median(dm):,.0f}  t={t_stat:.2f} p={t_p:.3g}  "
                  f"Wilcoxon p={w_p:.3g}")
    # save the summary matrix
    rows = []
    for (et, od, um, tl, ofw, oth) in configs:
        a = agg[(et, od, um, tl, ofw, oth)]
        # MEASURED net, consistent with the summary matrix (no plug)
        net = sum(a[b]["capture"] + a[b]["markout"]
                  + a[b]["liq_cap"] + a[b]["liq_mko"]
                  - (a[b]["fee"] + a[b]["liq_fee"]) for b in H.BUCKETS)
        on = sum(a[b]["opened_notional"] for b in H.BUCKETS)
        allhold = [h for b in H.BUCKETS for h in a[b]["holds"]]
        rows.append({"exit_ticks": et, "obi_defensive": od, "use_microprice": um,
                     "tol_ticks": tl,
                     "ofi_window": ("OFF" if ofw is None
                                    else f"min{ofw[0]}ev{ofw[1]:.0f}s"),
                     "ofi_thresh": (float("nan") if ofw is None else oth),
                     "net_bps": (1e4 * net / on) if on > 0 else np.nan,
                     # TOTAL P&L in PKR: the measured net (== engine P&L, since
                     # reconciliation is exact) and the engine's own figure as a
                     # cross-check. These are the whole-portfolio totals for the
                     # config across all names x days.
                     "net_pkr": net,
                     "engine_pnl_pkr": engine_pnl_agg[(et, od, um, tl, ofw, oth)],
                     "median_hold_s": (np.median(allhold) / 1000.0
                                       if allhold else np.nan),
                     "trades": sum(a[b]["trades"] for b in H.BUCKETS)})
    out = RESULTS / f"skew_sweep_2d_{stamp}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
