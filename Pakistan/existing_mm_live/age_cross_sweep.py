# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
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
import json
import os
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
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS = Path(str(_hft_paths.RESULTS_ROOT))

# ------------------------------ config ---------------------------------------
# the top-10 production book
NAMES = ['AKBL', 'ATRL', 'BAFL', 'BOP', 'DGKC', 'ENGROH', 'FFC', 'FNEL',
         'HASCOL', 'HBL', 'HUBC', 'KEL', 'LUCK', 'MARI', 'MEBL', 'MLCF',
         'NBP', 'NCPL', 'NML', 'NPL', 'NRL', 'OGDC', 'PACE', 'PAEL',
         'PIAHCLA', 'PIBTL', 'PIOC', 'PPL', 'PSO', 'PTC', 'SAZEW', 'SEARL',
         'SYS', 'THCCL', 'TOMCL', 'TPL', 'TRG', 'UBL']
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
# RUN A (lambda sweep): OFI restricted to OFF so only the defensive-lean sweep
# runs. The OFI verdict is settled (n=197: ties/loses); re-running OFI configs
# here would waste ~12h. To re-enable the full OFI grid, restore the 5-window list.
OFI_MODES = [None]
# AXIS 6 (STAGE 3): engage threshold on the NORMALIZED [-1,+1] trailing OFI.
# Only applies when a window is on (the OFF config is not duplicated per thresh).
OFI_THRESH = [0.20, 0.40]
# AXIS 7 (microprice defensive lean): continuous lambda on OFI=OFF ONLY. The
# classic microprice (lambda=+1) was destructive (leans INTO flow -> 'through'
# pick-offs). These NEGATIVE values lean the OTHER way (away from imbalance) to
# convert 'through' fills into 'at_queue'. Swept only on the OFF config (no
# crossing with OFI windows). None = the mid baseline (unchanged behavior).
MICRO_LAMBDA = [None]
# ---- POV ACQUISITION-CAP SWEEP (does capping late inventory by unwind capacity help?) ----
# Base = current production: OBI throttle (0.5x/300ms), OFI OFF. When the cap is
# on, max acquirable |pos| = min(max_inv, unwind_capacity x mult); mult scales how
# many unwind-capacities of inventory you allow to build. NONE = no-throttle anchor.
# ---- AGE-CROSS P&L SWEEP ----
# Once the current net position has been held > X minutes, CROSS THE SPREAD to
# fully flatten (a tagged taker order; engine allow_taker turned on by the same
# switch). Pays spread + fee for a guaranteed exit. Judged on RISK (drawdown,
# Sortino) as much as net bps -- the mean P&L is expected to fall.
#   OBI (thr=0) = current production = control (post-only engine, byte-identical).
#   AC_5 / 7.5 / 10 / 15 / 20 = OBI + age-cross at that minutes-held cutoff.
_OBI = dict(obi_throttle=True, ofi_throttle=False, obi_throttle_thresh=0.15,
            throttle_frac=0.5, throttle_hold_ms=300.0, qdr_throttle=False,
            enable_pov_cap=False, flow_throttle=False, enable_run_reprice=False,
            enable_aggr_lean=False, enable_age_cross=False)
# minutes-held cutoffs to sweep
_AC_MIN = [5.0, 7.5, 10.0, 15.0, 20.0]
# thr=0 OBI control; then OBI + age-cross at each cutoff
THROTTLE_MODES = [dict(_OBI)]
THROTTLE_LABELS = ["OBI"]
for _m in _AC_MIN:
    THROTTLE_MODES.append(dict(_OBI, enable_age_cross=True, age_cross_ms=_m * 60000.0))
    THROTTLE_LABELS.append(f"OBI+AC_{_m:g}m")
def _cfg_lab(thr):
    return THROTTLE_LABELS[thr]
def _cfg_thr(thr):
    m = THROTTLE_MODES[thr]
    return f"{m['age_cross_ms']/60000:g}m" if m.get("enable_age_cross") else "-"

EXIT_INV_THRESHOLD = 1.0
# OBI-defensive engage threshold (|imb-0.5|) and widen ticks
OBI_DEF_THRESH = 0.15
OBI_DEF_TICKS = 1.0
# jump detector (same as the decomposition)
JUMP_K = 4.0
# CANARY vs FULL: 2 = a fast 38-name x 9-config canary that exercises the real
# path and prints the anchor, so config/calibration errors surface in minutes,
# not 4 hours. Set to None for the FULL ~207-day run ONLY after the canary's
# anchor reads 0 on all 9 configs and preflight_coverage.py shows all names OK.
# (Hard lesson: a multi-hour run was burned on an unverified sweep.)
SMOKE_DAYS = 30
# workers
WORKERS = 9
# -----------------------------------------------------------------------------

# worker globals
_G = {}


# ---- CANONICAL NET (defined ONCE; every net computation routes through here) --
# The measured decomposition identity, per bucket:
#   net = capture + markout + liq_cap + liq_mko - fee - liq_fee
# capture/markout = intraday quoting P&L; liq_cap/liq_mko = liquidated round
# trips' P&L (Option A, booked to the opening bucket); fee = round-trip fees;
# liq_fee = a legacy liquidation-fee column (0 since Stage 2 routes liq fees
# through 'fee', kept for back-compat). This telescopes to engine realized cash
# EXACTLY -- so the anchor's 'unexplained' is ~0. Defining it in one place means
# a future column change touches ONE function, never four copies that can drift.
def bucket_net(d):
    # net P&L (in the dict's native units, PKR) of a single bucket dict `d`.
    return (d["capture"] + d["markout"] + d["liq_cap"] + d["liq_mko"]
            - d["fee"] - d["liq_fee"])


def net_of(per):
    # net P&L summed over all buckets of a per-bucket mapping `per`.
    return sum(bucket_net(per[b]) for b in H.BUCKETS)


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
    date, sym, exit_ticks, obi_def, use_micro, tol, ofi_win, ofi_th, micro_lam, thr = args
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
                   "ofi_defensive_thresh": ofi_th,
                   # AXIS 7: continuous microprice lean. None -> use_microprice
                   # governs (baseline); negative -> defensive lean (OFF config).
                   "micro_lambda": micro_lam})
    # STAGE 4: apply the throttle overrides for this config (thr indexes
    # THROTTLE_MODES). OFF (thr=0) sets both flags False -> byte-identical to the
    # frozen winner; ON (thr=1) enables the 0.5x time-boxed size throttle. Merged
    # AFTER build so it cleanly overrides the defaults without touching H.
    params.update(THROTTLE_MODES[thr])
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
    dec_total = net_of(per)
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
            "tol": tol, "ofi_win": ofi_win, "ofi_th": ofi_th,
            # micro_lambda (defensive lean) -- part of the config identity
            "mlam": micro_lam,
            # throttle index (0=OFF, 1=ON) -- the config identity for this sweep
            "thr": thr, "per": per,
            # SYMBOL: retained so the reduction can accumulate per-NAME P&L (the
            # axis a 38-name study must keep -- lets top-N / per-name t-tests be
            # sliced after the fact without re-running).
            "sym": str(sym),
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
    # 7-axis config tuples. micro_lambda (mlam) varies ONLY on the OFI-OFF config
    # (ofw is None); every OFI-on config carries mlam=None (its own baseline).
    # This runs the defensive-lean sweep alongside the OFI sweep WITHOUT crossing
    # the two axes (per spec: lambda on OFF only).
    # Config = the frozen winner (et=1, obi_def=True, mid, tol=0, OFI off,
    # mlam=None) crossed with the THROTTLE axis. An 8th tuple element `thr` (an
    # index into THROTTLE_MODES) is the only thing that varies -> exactly 2
    # configs. Everything else is a singleton list, so this is a clean 2-config
    # sweep with per-name daily output inherited from the base sweep machinery.
    configs = []
    for et in EXIT_TICKS:
        for od in OBI_MODES:
            for um in MICRO_MODES:
                for tl in TOL_MODES:
                    for ofw in OFI_MODES:
                        for mlam in MICRO_LAMBDA:
                            # throttle index 0 (OFF) and 1 (ON)
                            for thr in range(len(THROTTLE_MODES)):
                                configs.append((et, od, um, tl, ofw,
                                                OFI_THRESH[0], mlam, thr))
    # work list: every (date, symbol) x each config (now an 8-tuple)
    work = [(date, sym, et, od, um, tl, ofw, oth, mlam, thr)
            for date in run_dates for sym in NAMES
            for (et, od, um, tl, ofw, oth, mlam, thr) in configs]
    total_all = len(work)
    # ---- CHECKPOINTING (crash/kill/quit-proof): a JOURNAL that appends one raw
    # line per completed (config, date, sym) as it finishes, flushed every time.
    # If the run dies (OOM, PyCharm quit, power loss), completed work is on disk.
    # On restart, we load the journal and SKIP work already done -> resume, not
    # restart. The journal is the finest grain; the polished CSVs are still built
    # from the in-memory aggregation at the end (unchanged). File is fixed-named
    # (no stamp) so a resume finds the SAME journal.
    ckpt_path = RESULTS / "age_cross_sweep_CKPT.jsonl"
    # a work item's identity for resume = (date, sym, thr) -- the only varying
    # axes here (everything else is a singleton). Load any already-done keys.
    done_keys = set()
    if ckpt_path.exists():
        # read the journal; each line is a JSON dict of one completed cell
        with open(ckpt_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done_keys.add((r["date"], r["sym"], r["thr"]))
                except (ValueError, KeyError):
                    # a torn last line from a hard kill -> skip it, harmless
                    continue
        print(f"  RESUME: journal has {len(done_keys)} completed cells; "
              f"skipping those.", flush=True)
    # filter the work list to only NOT-yet-done cells (resume)
    if done_keys:
        work = [w for w in work
                if (w[0], w[1], w[9]) not in done_keys]
    total = len(work)
    print(f"  {total} cells to run this session "
          f"({total_all} total, {total_all - total} already journaled).",
          flush=True)
    # open the journal in APPEND mode for this session (line-buffered so every
    # write hits disk promptly; we also flush explicitly per cell)
    ckpt_fh = open(ckpt_path, "a", buffering=1)
    print(f"\nAGE-CROSS -- cross the spread to flatten aged inventory (winner frozen "
          f"et1/obi+/tol0/mid, OFI-defensive off; framing #2 = throttle STACKS "
          f"on obi_defensive): OFF vs ON = {len(configs)} configs",
          flush=True)
    print(f"  x {len(NAMES)} names x {len(run_dates)} days = {total} cells",
          flush=True)
    print(f"  workers: {nproc}\n", flush=True)
    # per-config accumulators keyed by (exit_ticks, obi_def)
    agg = {(et, od, um, tl, ofw, oth, mlam, thr): {b: {"capture": 0.0, "markout": 0.0, "fee": 0.0,
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
           for (et, od, um, tl, ofw, oth, mlam, thr) in configs}
    # per-config daily portfolio markout for significance
    daily_mko = {(et, od, um, tl, ofw, oth, mlam, thr): {} for (et, od, um, tl, ofw, oth, mlam, thr) in configs}
    # per-config DAY-AS-UNIT net: {day -> {"net": pkr, "on": opened_notional}}.
    # Each trading day is ONE observation (net_bps for that day's portfolio);
    # the SE across these days is the day-as-unit standard error -- the correct
    # dispersion, NOT a pooled row-count SE that would overstate significance.
    daily_net = {(et, od, um, tl, ofw, oth, mlam, thr): {} for (et, od, um, tl, ofw, oth, mlam, thr) in configs}
    # GRANULAR per-(config, day, bucket) accumulator for the daily-trend CSV.
    # {key -> {day -> {bucket -> [net_pkr, opened_notional, fills]}}}. This keeps
    # every day's per-bucket net so the trend over the 197 days is visible (is
    # OFI's edge stable / decaying / strengthening, and in WHICH bucket?), rather
    # than only a single collapsed average.
    daily_bkt = {(et, od, um, tl, ofw, oth, mlam, thr): {} for (et, od, um, tl, ofw, oth, mlam, thr) in configs}
    # FINEST-GRAIN per-(config, day, NAME, bucket) accumulator. This is the axis
    # that was missing: it lets top-N-name analysis, per-name OFF-vs-OFI tests,
    # and the portfolio rollup ALL be derived after the fact from ONE run. Keyed
    # {config -> {(day, sym, bucket) -> [net_pkr, opened_notional, fills]}}.
    name_bkt = {(et, od, um, tl, ofw, oth, mlam, thr): {} for (et, od, um, tl, ofw, oth, mlam, thr) in configs}
    # per-config RAW engine P&L accumulator (the reconciliation anchor)
    engine_pnl_agg = {(et, od, um, tl, ofw, oth, mlam, thr): 0.0
                      for (et, od, um, tl, ofw, oth, mlam, thr) in configs}
    # accumulate the unexplained residual (measured decomposition vs engine P&L);
    # ~0 means the book-walk liquidation attribution is correct, non-zero flags
    # a real measurement error (NOT hidden by a plug anymore)
    recon_gap_agg = {(et, od, um, tl, ofw, oth, mlam, thr): 0.0
                     for (et, od, um, tl, ofw, oth, mlam, thr) in configs}
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
               res["ofi_win"], res["ofi_th"], res["mlam"], res["thr"])
            # ---- CHECKPOINT: journal this completed cell IMMEDIATELY (before any
            # aggregation) so a crash right now still keeps it. One JSON line with
            # the full per-bucket decomposition -> the PERNAME CSV can be rebuilt
            # from the journal alone if the final write never happens. Flushed +
            # fsync'd so it survives a hard kill.
            _jrow = {"date": res["date"], "sym": res["sym"], "thr": res["thr"],
                     "per": {b: {k: res["per"][b][k]
                                 for k in ("capture", "markout", "fee", "liq_fee",
                                           "liq_cap", "liq_mko", "opened_notional",
                                           "fills")}
                             for b in H.BUCKETS}}
            ckpt_fh.write(json.dumps(_jrow) + "\n")
            ckpt_fh.flush()
            os.fsync(ckpt_fh.fileno())
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
            # DAY-AS-UNIT net: accumulate this symbol-day's net PKR and opened
            # notional into the day's bucket (summed across symbols -> one
            # portfolio observation per day). net = cap+mko+liq-fee (same
            # identity as the anchor; reconciles to engine exactly).
            dnet = net_of(res["per"])
            don = sum(res["per"][b]["opened_notional"] for b in H.BUCKETS)
            # this day's running (net_pkr, opened_notional) for the config
            slot = daily_net[key].get(day, (0.0, 0.0))
            daily_net[key][day] = (slot[0] + dnet, slot[1] + don)
            # GRANULAR: accumulate per-bucket net + notional + fills for this
            # (config, day), summed across symbols. One row per bucket per day.
            dayb = daily_bkt[key].setdefault(day, {})
            for b in H.BUCKETS:
                sb = res["per"][b]
                # per-bucket net PKR (same identity as the portfolio net)
                bnet = bucket_net(sb)
                # running [net_pkr, opened_notional, fills] for this bucket-day
                acc = dayb.setdefault(b, [0.0, 0.0, 0])
                acc[0] += bnet
                acc[1] += sb["opened_notional"]
                acc[2] += sb["fills"]
            # FINEST-GRAIN: per-(day, name, bucket) FULL decomposition + notional
            # + fills. Each symbol-day cell contributes one row per bucket. No
            # summing across names -> the name axis is preserved. Carrying the
            # full decomposition (not just net) means a per-name OFI effect can be
            # EXPLAINED (markout improvement vs capture shuffle vs liquidation
            # luck) from this one file, never needing another run.
            sym = res["sym"]
            for b in H.BUCKETS:
                sb = res["per"][b]
                nk = (day, sym, b)
                # [net, capture, markout, liq, fee, opened_notional, fills]
                nacc = name_bkt[key].setdefault(nk, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
                nacc[0] += bucket_net(sb)
                nacc[1] += sb["capture"]
                nacc[2] += sb["markout"]
                nacc[3] += sb["liq_cap"] + sb["liq_mko"]
                nacc[4] += sb["fee"] + sb["liq_fee"]
                nacc[5] += sb["opened_notional"]
                nacc[6] += sb["fills"]
            # accumulate raw engine P&L for the reconciliation anchor
            engine_pnl_agg[key] += res["engine_pnl"]
            # accumulate the unexplained residual (measured recon quality)
            recon_gap_agg[key] += res["recon_gap"]
    # ---- CHECKPOINT: the pool finished cleanly. Close the journal and rename it
    # to .done so a future run starts FRESH instead of wrongly "resuming" a
    # completed sweep. (If the run had died, this line is never reached and the
    # journal stays as .jsonl -> the next launch resumes from it.)
    ckpt_fh.close()
    done_marker = ckpt_path.with_suffix(".jsonl.done")
    try:
        # replace any stale marker, then rename
        if done_marker.exists():
            done_marker.unlink()
        ckpt_path.rename(done_marker)
        print(f"\n  checkpoint complete -> {done_marker.name}", flush=True)
    except OSError:
        # non-fatal: the polished CSVs are the real output
        pass
    # ---- reporting ----
    print("\n" + "=" * 84)
    print(f"### 2D SKEW SWEEP DONE: {H._fmt(time.perf_counter() - t0)} "
          f"for {total} cells ###")
    print("=" * 84)
    print("\n=== SUMMARY: OFI-defensive configs (winner frozen: et1/obi+/tol0) ===")
    print(f"{'ofi_window':>14s} {'thresh':>7s} {'net_bps':>9s} {'net_PKR':>13s}")
    for (et, od, um, tl, ofw, oth, mlam, thr) in configs:
        a = agg[(et, od, um, tl, ofw, oth, mlam, thr)]
        # MEASURED net = quoting P&L + liquidation P&L - all fees (reconciles
        # to engine exactly; see the anchor)
        net = net_of(a)
        on = sum(a[b]["opened_notional"] for b in H.BUCKETS)
        # bps of opened notional
        nbps = (1e4 * net / on) if on > 0 else float("nan")
        # window label: OFF or min(Nev,Ts)
        wlab = _cfg_lab(thr)
        tlab = _cfg_thr(thr)
        print(f"{wlab:>14s} {tlab:>7s} {nbps:>9.3f} {net:>13,.0f}")

    # ---- DAY-AS-UNIT ERROR BARS -------------------------------------------
    # Each trading day = ONE observation of portfolio net_bps for a config.
    # The SE across days is the honest dispersion (NOT a pooled row-count SE).
    # We report per-config mean +/- SE, then the OFF-vs-each-OFI PAIRED
    # difference (paired by day: same days under both configs), which is the
    # correct test of "does OFF beat this OFI variant beyond day-to-day noise".
    def _day_bps_series(key):
        # {day -> net_bps} for a config: day net PKR / day opened notional * 1e4
        out = {}
        for day, (net_pkr, on) in daily_net[key].items():
            if on > 0:
                out[day] = 1e4 * net_pkr / on
        return out
    # locate the OFF key (window is None); it is the control
    off_key = next(((et, od, um, tl, ofw, oth, mlam, thr)
                    for (et, od, um, tl, ofw, oth, mlam, thr) in configs if thr == 0), None)
    # per-config mean +/- day-as-unit SE
    print("\n=== DAY-AS-UNIT net_bps: mean +/- SE (each day = one observation) ===")
    print(f"{'ofi_window':>14s} {'thresh':>7s} {'n_days':>7s} "
          f"{'mean_bps':>9s} {'se':>7s}")
    for (et, od, um, tl, ofw, oth, mlam, thr) in configs:
        ser = _day_bps_series((et, od, um, tl, ofw, oth, mlam, thr))
        vals = list(ser.values())
        n = len(vals)
        # mean and day-as-unit SE (sample sd / sqrt(n))
        if n >= 2:
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            se = (var ** 0.5) / (n ** 0.5)
        else:
            mean = (vals[0] if n == 1 else float("nan"))
            se = float("nan")
        wlab = _cfg_lab(thr)
        tlab = _cfg_thr(thr)
        print(f"{wlab:>14s} {tlab:>7s} {n:>7d} {mean:>9.3f} {se:>7.3f}")
    # OFF-vs-OFI paired differences (paired by day)
    if off_key is not None:
        off_ser = _day_bps_series(off_key)
        print("\n=== control(OBI) - config (PAIRED by day): positive => current OBI wins ===")
        print(f"{'config':>14s} {'thresh':>7s} {'n_pair':>7s} "
              f"{'d_bps':>8s} {'se':>7s} {'t':>7s}")
        for (et, od, um, tl, ofw, oth, mlam, thr) in configs:
            # skip the control (NONE, thr==0) itself
            if thr == 0:
                continue
            ser = _day_bps_series((et, od, um, tl, ofw, oth, mlam, thr))
            # paired differences on the days present in BOTH configs
            days = sorted(set(off_ser) & set(ser))
            diffs = [off_ser[d] - ser[d] for d in days]
            m = len(diffs)
            # paired mean diff, SE, and t-stat (day-as-unit)
            if m >= 2:
                dm = sum(diffs) / m
                dv = sum((x - dm) ** 2 for x in diffs) / (m - 1)
                dse = (dv ** 0.5) / (m ** 0.5)
                tstat = dm / dse if dse > 0 else float("nan")
            else:
                dm = (diffs[0] if m == 1 else float("nan"))
                dse = float("nan"); tstat = float("nan")
            wlab = _cfg_lab(thr)
            tlab = _cfg_thr(thr)
            print(f"{wlab:>14s} {tlab:>7s} {m:>7d} {dm:>8.3f} {dse:>7.3f} "
                  f"{tstat:>7.2f}")
        # reading guide
        print("  (|t| >~ 2 => the gap vs NONE is beyond "
              "day-to-day noise)")

    # ---- RECONCILIATION ANCHOR: the MEASURED decomposition (capture + markout
    # incl. book-walk liquidation - fees) vs raw engine P&L. NO plug is added
    # here -- 'unexplained' is the honest residual. If the book-walk liquidation
    # attribution is correct it is ~0; a large value is a REAL measurement error
    # to fix, not something to hide. (liq_loss in the tables holds this same
    # residual as a diagnostic, distributed pro-rata.)
    print("\n=== RECONCILIATION ANCHOR: MEASURED decomposition vs engine P&L ===")
    print(f"{'ofi_window':>14s} {'thresh':>7s} {'measured_net':>16s} "
          f"{'engine_pnl':>16s} {'unexplained':>13s} {'pct':>7s} {'ok?':>5s}")
    for (et, od, um, tl, ofw, oth, mlam, thr) in configs:
        a = agg[(et, od, um, tl, ofw, oth, mlam, thr)]
        # MEASURED net: capture + markout (incl. measured liquidation) - all fees.
        # NOTE: liq_loss is DELIBERATELY EXCLUDED -- it is the diagnostic residual,
        # not part of the measurement. If measurements are right, measured_net
        # already equals engine P&L without it.
        measured_net = net_of(a)
        eng = engine_pnl_agg[(et, od, um, tl, ofw, oth, mlam, thr)]
        # the honest unexplained residual (should be ~0)
        unexplained = measured_net - eng
        pct = (100.0 * unexplained / eng) if abs(eng) > 1 else float("nan")
        # tolerance: within 2% of engine P&L is a sound measurement
        ok = "OK" if abs(pct) < 2.0 or abs(unexplained) < max(1.0, 0.0) \
            else "BAD"
        wlab = _cfg_lab(thr)
        tlab = _cfg_thr(thr)
        print(f"{wlab:>14s} {tlab:>7s} {measured_net:>16,.0f} {eng:>16,.0f} "
              f"{unexplained:>13,.0f} {pct:>6.1f}% {ok:>5s}")
    # detailed per-config tables
    for (et, od, um, tl, ofw, oth, mlam, thr) in configs:
        a = agg[(et, od, um, tl, ofw, oth, mlam, thr)]
        print(f"\n{'=' * 84}")
        wlab = _cfg_lab(thr)
        tlab = _cfg_thr(thr)
        print(f"CONFIG: et={et} obi={od} tol={tl}   throttle={wlab} thresh={tlab}")
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
              f"{'net_bps':>8s} {'net_pkr':>11s} {'med_hld':>8s} {'mean_hld':>8s} "
              f"{'med_tks_mid':>11s} {'thr%':>5s} {'trades':>8s} {'sh/trd':>8s}")
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
            net_bps = _bps(bucket_net(d))
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
                  f"{net_bps:>8.3f} {bucket_net(d):>11,.0f} "
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
        dm = pd.Series(daily_mko[(et, od, um, tl, ofw, oth, mlam, thr)]).sort_index().to_numpy()
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
    for (et, od, um, tl, ofw, oth, mlam, thr) in configs:
        a = agg[(et, od, um, tl, ofw, oth, mlam, thr)]
        # MEASURED net, consistent with the summary matrix (no plug)
        net = net_of(a)
        on = sum(a[b]["opened_notional"] for b in H.BUCKETS)
        allhold = [h for b in H.BUCKETS for h in a[b]["holds"]]
        rows.append({"exit_ticks": et, "obi_defensive": od, "use_microprice": um,
                     "tol_ticks": tl,
                     # STAGE 4: the config identity for THIS sweep
                     "throttle": THROTTLE_LABELS[thr],
                     "ofi_window": _cfg_lab(thr),
                     "ofi_thresh": _cfg_thr(thr),
                     "net_bps": (1e4 * net / on) if on > 0 else np.nan,
                     # TOTAL P&L in PKR: the measured net (== engine P&L, since
                     # reconciliation is exact) and the engine's own figure as a
                     # cross-check. These are the whole-portfolio totals for the
                     # config across all names x days.
                     "net_pkr": net,
                     "engine_pnl_pkr": engine_pnl_agg[(et, od, um, tl, ofw, oth, mlam, thr)],
                     "median_hold_s": (np.median(allhold) / 1000.0
                                       if allhold else np.nan),
                     "trades": sum(a[b]["trades"] for b in H.BUCKETS)})
    out = RESULTS / f"age_cross_sweep_{stamp}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")

    # ---- GRANULAR DAILY CSV (the time-trend artifact) --------------------
    # One row per (config, day, bucket): net_bps + net_pkr + opened_notional +
    # fills, PLUS a portfolio-level (bucket="ALL") row per config-day, and the
    # OFF-minus-this-config daily paired difference on the ALL row. This is what
    # lets you see whether OFI's edge is stable / decaying / strengthening across
    # the 197 days -- and in which bucket -- rather than a single collapsed mean.
    # first, portfolio net_bps per (config, day) from daily_net (for the diff)
    def _port_day_bps(key):
        # {day -> portfolio net_bps}
        return {d: (1e4 * npkr / on if on > 0 else float("nan"))
                for d, (npkr, on) in daily_net[key].items()}
    # OFF control's per-day portfolio bps (for the paired daily difference)
    off_key = next(((et, od, um, tl, ofw, oth, mlam, thr)
                    for (et, od, um, tl, ofw, oth, mlam, thr) in configs if thr == 0), None)
    off_day_bps = _port_day_bps(off_key) if off_key is not None else {}
    # build the long-format rows
    drows = []
    for (et, od, um, tl, ofw, oth, mlam, thr) in configs:
        key = (et, od, um, tl, ofw, oth, mlam, thr)
        # config labels
        wlab = _cfg_lab(thr)
        tlab = _cfg_thr(thr)
        # portfolio per-day bps for this config (for the ALL row + diff)
        port = _port_day_bps(key)
        # every day this config produced
        for day in sorted(daily_bkt[key].keys()):
            # ---- per-bucket rows ----
            for b in H.BUCKETS:
                acc = daily_bkt[key][day].get(b)
                if acc is None:
                    continue
                bnet, bon, bfills = acc
                drows.append({
                    "date": day, "throttle": THROTTLE_LABELS[thr], "ofi_window": wlab, "ofi_thresh": tlab,
                    "bucket": b,
                    "net_bps": (1e4 * bnet / bon) if bon > 0 else float("nan"),
                    "net_pkr": bnet, "opened_notional": bon, "fills": bfills,
                    # OFF diff only meaningful on the ALL row -> NaN per bucket
                    "off_minus_this_bps": float("nan")})
            # ---- portfolio ALL row (with the OFF paired daily diff) ----
            pbps = port.get(day, float("nan"))
            # OFF - this config, same day (positive => OFF wins that day). NaN on
            # the OFF row itself and on days OFF did not run.
            if ofw is None:
                diff = float("nan")
            else:
                ob = off_day_bps.get(day)
                diff = (ob - pbps) if (ob is not None and pbps == pbps) else float("nan")
            # portfolio net_pkr + notional for the ALL row
            npkr, on = daily_net[key].get(day, (float("nan"), 0.0))
            drows.append({
                "date": day, "throttle": THROTTLE_LABELS[thr], "ofi_window": wlab, "ofi_thresh": tlab,
                "bucket": "ALL",
                "net_bps": pbps, "net_pkr": npkr, "opened_notional": on,
                "fills": sum(daily_bkt[key][day][b][2]
                             for b in H.BUCKETS if b in daily_bkt[key][day]),
                "off_minus_this_bps": diff})
    # write the granular daily CSV
    dout = RESULTS / f"age_cross_sweep_DAILY_{stamp}.csv"
    pd.DataFrame(drows).to_csv(dout, index=False)
    print(f"wrote {dout}  ({len(drows)} rows: per config x day x bucket + ALL)")

    # ---- PER-NAME DAILY CSV (finest grain; the axis that must never be dropped)
    # One row per (config, day, name, bucket): net_pkr + opened_notional + fills.
    # net_bps is per-row net_pkr / opened_notional * 1e4. From THIS file you can
    # derive: top-N-name subsets, per-name OFF-vs-OFI paired t-tests, per-name
    # trend over time, and the portfolio rollup -- all WITHOUT re-running. It is
    # the single source the aggregated CSVs summarize.
    nrows = []
    for (et, od, um, tl, ofw, oth, mlam, thr) in configs:
        key = (et, od, um, tl, ofw, oth, mlam, thr)
        wlab = _cfg_lab(thr)
        tlab = _cfg_thr(thr)
        for (day, sym, b), acc in name_bkt[key].items():
            net_pkr, cap_pkr, mko_pkr, liq_pkr, fee_pkr, on, fills = acc
            # bps helper for this row
            _b = (lambda x: (1e4 * x / on) if on > 0 else float("nan"))
            nrows.append({
                "date": day, "symbol": sym,
                "throttle": THROTTLE_LABELS[thr],
                "ofi_window": wlab, "ofi_thresh": tlab, "bucket": b,
                # net (the headline) in bps + PKR
                "net_bps": _b(net_pkr), "net_pkr": net_pkr,
                # full decomposition in PKR so any per-name effect is explainable
                "capture_pkr": cap_pkr, "markout_pkr": mko_pkr,
                "liq_pkr": liq_pkr, "fee_pkr": fee_pkr,
                # and markout in bps (the adverse-selection read, per name)
                "markout_bps": _b(mko_pkr),
                "opened_notional": on, "fills": fills})
    nout = RESULTS / f"age_cross_sweep_PERNAME_{stamp}.csv"
    pd.DataFrame(nrows).to_csv(nout, index=False)
    print(f"wrote {nout}  ({len(nrows)} rows: per config x day x name x bucket)")


if __name__ == "__main__":
    main()
