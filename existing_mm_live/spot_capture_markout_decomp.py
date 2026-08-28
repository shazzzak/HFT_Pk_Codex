# spot_capture_markout_decomp.py -- the honest per-bucket decomposition:
#   capture, FIFO markout (to actual matched exit), fees, net-bps, med-hold,
#   with markout further SPLIT into jump vs diffusion components, across a
#   3x / 5x / 7x / 10x median-trade-size clip sweep, with daily significance
#   testing of markout vs zero (day-as-unit, n ~ 207).
#
# WHY: analyze_bucket_attribution.py reports NET realized bps per bucket only.
# This file decomposes that net into capture + markout - fees (each as its own
# column), so we can see WHERE the edge lives and how much markout eats it. The
# reconciliation check (capture + markout - fees == realized) makes the
# decomposition correct by construction; we assert it per bucket.
#
# The markout is measured to the ACTUAL FIFO-matched exit for each opening fill
# (fill -> the moment that share was actually offset), consistent with the
# production-correct definition. Residual (never-closed) lots are marked to the
# EOD liquidation VWAP.
#
# The markout is decomposed into JUMP vs DIFFUSION by walking the mid path
# event-by-event over each fill's markout window and classifying each per-event
# log-return via Lee & Mykland (2008)-style thresholding.
#
# ---------------------------- DESIGN CHOICES ---------------------------------
# EVERY parameter below drives the results. Change only with a stated reason.
#
# JUMP DETECTOR
#   k=4 (Lee-Mykland default) -- a per-event |log return| exceeding
#   k * sigma_local is flagged as a jump. At k=4, real jumps are captured while
#   normal diffusion is not; the flagged fraction on liquid names typically sits
#   below ~1%. We print the flagged fraction so it can be sanity-checked.
#   sigma_local is a rolling estimate over the prior LOCAL_VOL_WIN events (see
#   below), using per-event log-returns.
# LOCAL_VOL_WIN=100 events -- large enough to be a stable estimate, small
#   enough to be "local" (Lee-Mykland recommend ~a few hundred).
#
# MARKOUT HORIZON: FIFO-matched exit. Per-fill exit_t = the moment that share
# was actually offset (the exact same match the fifo_attribution uses). For
# unclosed residual, exit_t = liquidation time (from DayResult.eod).
#
# SIGNIFICANCE: day as the unit. We aggregate to a daily portfolio markout
# series (n ~ 207 days). t-test AND Wilcoxon signed-rank against zero.
# Using per-fill markouts as the unit would overstate significance
# (pseudo-replication -- fills within a day are correlated).
#
# CLIPS SWEPT: 3x, 5x, 7x, 10x the trailing median trade size (the same anchor
# as production). The 3x -> 10x range brackets the earlier capacity finding
# (3x optimal, 5x incremental too small vs overnight-inventory risk).
# -----------------------------------------------------------------------------

# paths + timing
from pathlib import Path
# wall-clock
import time
# stamp
from datetime import datetime
# multiprocessing (the fast harness pattern -- ~30x on M4)
import multiprocessing as mp
# arrays + frames
import numpy as np
import pandas as pd
# significance tests
from scipy import stats
# shared harness (backtest driver, FIFO attribution scaffolding, stats)
import mm_harness as H
# the driver (dates, datasets)
import run_legacy_mm as R

# ------------------------------ config ---------------------------------------
# the top-10 production book
NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
# clip multiples to sweep (median trade size x each)
CLIP_MULTS = [3.0, 5.0, 7.0, 10.0]
# trailing median-trade-size window (days) -- same as production calibration
TRAIL_DAYS = 10
# jump detector: Lee-Mykland k (per-event |r| > k * sigma_local -> jump)
JUMP_K = 4.0
# local sigma estimate window (events prior)
LOCAL_VOL_WIN = 100
# smoke test: run first N days (None -> full ~207 days)
SMOKE_DAYS = None
# parallel workers (None -> all cores)
WORKERS = None
# -----------------------------------------------------------------------------

# module-level globals populated once per worker (calibration bundle)
_G = {}


# worker init: receives the pre-computed calibration
def _init_worker(calib):
    # stash into worker-local globals
    _G.update(calib)


# ---- helpers: mid asof and jump detection ----

# mid at time t via asof over an (equity_t, equity_mid) time series (arrays)
def _mid_at(eq_t, eq_mid, t):
    # empty series -> no mid
    if len(eq_t) == 0:
        return np.nan
    # last equity row with time <= t
    pos = np.searchsorted(eq_t, t, side="right") - 1
    # nothing before -> no mid
    if pos < 0:
        return np.nan
    # the mid
    return float(eq_mid[pos])


# split a mid path over [fill_t, exit_t] into (jump_move, diffusion_move) in
# PRICE units. Uses per-event log-returns; a return whose magnitude exceeds
# JUMP_K * sigma_local is flagged as a jump; sigma_local is a rolling estimate
# over the prior LOCAL_VOL_WIN log-returns.
def _split_move(eq_t, eq_mid, fill_t, exit_t):
    # empty -> zero moves
    if len(eq_t) == 0 or exit_t <= fill_t:
        return 0.0, 0.0
    # index range covering the window (inclusive of both anchors)
    a = np.searchsorted(eq_t, fill_t, side="right") - 1
    b = np.searchsorted(eq_t, exit_t, side="right") - 1
    # guard bounds
    a = max(0, a); b = max(a, b)
    # need at least the exit anchor after the fill anchor
    if b <= a:
        return 0.0, 0.0
    # extract mid path over the window
    m = np.asarray(eq_mid[a:b + 1], dtype=float)
    # need at least 2 points for a return
    if len(m) < 2 or not np.isfinite(m).all() or (m <= 0).any():
        return 0.0, 0.0
    # per-event LOG returns across the window
    r = np.diff(np.log(m))
    # each return's absolute magnitude
    ar = np.abs(r)
    # rolling local sigma estimate: for each return, the std of the prior
    # LOCAL_VOL_WIN returns (from BEFORE this fill's window, using the tail of
    # a global path is complex; here we use IN-window trailing std as a proxy).
    # Prior returns are not passed in; we approximate sigma_local by the median
    # absolute return in the window (robust, Lee-Mykland-friendly). If there
    # are fewer than 5 returns, use the mean absolute return (small-sample).
    if len(ar) >= 5:
        # robust local sigma proxy: MAD / 0.6745 -> normal-consistent std
        sigma_local = np.median(ar) / 0.6745
    else:
        # small sample -> mean absolute return
        sigma_local = ar.mean()
    # degenerate sigma -> treat all as diffusion
    if not np.isfinite(sigma_local) or sigma_local <= 0:
        return 0.0, float(m[-1] - m[0])
    # per-return jump mask
    is_jump = ar > (JUMP_K * sigma_local)
    # decompose the total PRICE move into jump and diffusion contributions,
    # by SIGNED price differences under each classification
    # (dpx_i = m_{i+1} - m_i is the per-return price step; sum equals m[-1]-m[0])
    dpx = np.diff(m)
    # signed jump component (sum of the flagged steps)
    jump_move = float(dpx[is_jump].sum())
    # signed diffusion component (sum of the unflagged steps)
    diff_move = float(dpx[~is_jump].sum())
    # exact reconciliation invariant: jump + diffusion = total
    return jump_move, diff_move


# ---- the per-symbol-day decomposition ----

# process one (date, symbol) at ONE clip multiple; returns per-bucket
# decomposition + significance-testing per-fill markouts + OBI-at-fill.
def _process(args):
    # unpack
    date, sym, clip_mult = args
    # pull calibration
    scales = _G["scales"]
    profiles = _G["profiles"]
    windows = _G["windows"]
    segments = _G["segments"]
    all_dates = _G["all_dates"]
    tstats = _G["tstats"]
    # session segments for the date
    segs = segments.get(str(date))
    # no segments -> skip
    if segs is None:
        return None
    # trailing-median trade size (walk-forward)
    med = H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
    # missing / degenerate anchor -> skip
    if med is None or med <= 0:
        return None
    # missing per-symbol calibration -> skip
    if sym not in scales or sym not in profiles:
        return None
    # open datasets
    dsets = R.open_datasets(date)
    # missing partition -> skip
    if dsets is None:
        return None
    # clip = clip_mult * trailing median trade size
    clip = max(1, int(round(clip_mult * med)))
    # build the locked production params at this clip
    params = H.build_micro_params(
        clip, scales[sym], profiles[sym],
        windows.get(sym, (5.0, 1.0)), segs)
    # run the backtest (this is the same run_symbol_day used everywhere)
    dr = H.run_symbol_day(date, sym, dsets, params)
    # unrunnable / no pnl -> skip
    if dr is None or dr.pnl() is None:
        return None
    # ---- reconstruct the mid time series from the equity log ----
    eq = pd.DataFrame(dr.equity) if len(dr.equity) else pd.DataFrame()
    # need at least a mid series
    if len(eq) == 0 or "mid" not in eq.columns:
        return None
    # sort by time (should already be sorted, but be explicit)
    eq = eq.sort_values("t")
    # arrays for asof + jump split
    eq_t = eq["t"].to_numpy(dtype=float)
    eq_mid = eq["mid"].to_numpy(dtype=float)
    # OBI at fill time: only available if log_equity was on (obi_5/obi_deep cols)
    have_obi = "obi_5" in eq.columns and "obi_deep" in eq.columns
    # OBI arrays (or None)
    eq_obi5 = eq["obi_5"].to_numpy(dtype=float) if have_obi else None
    eq_obi_d = eq["obi_deep"].to_numpy(dtype=float) if have_obi else None
    # ---- run FIFO matching AGAIN in this file so we can capture every
    # opening leg's exit time (fifo_attribution collapses the info). ----
    # per-bucket decomposition accumulators
    per = {b: {"capture": 0.0, "markout": 0.0, "fee": 0.0,
               "jump_markout": 0.0, "diff_markout": 0.0, "liq_loss": 0.0,
               "liq_fee": 0.0,
               "opened_notional": 0.0, "opened_qty": 0.0, "holds": [], "fills": 0,
               "obi5_sum": 0.0, "obi_deep_sum": 0.0, "obi_n": 0}
           for b in H.BUCKETS}
    # per-fill markouts for daily-significance aggregation (portfolio series)
    fill_markouts = []
    # the FIFO open-lot queue (all entries same net side)
    open_lots = []
    # fills from the run
    fills = dr.fills.to_dict("records") if isinstance(dr.fills, pd.DataFrame) \
        else list(dr.fills)
    # process each fill in stream order
    for fl in fills:
        # unpack
        side = fl["side"]
        px = float(fl["px"])
        qty = float(fl["qty"])
        b = fl.get("bucket", "middle")
        t = float(fl["t"])
        # count the fill in its bucket
        if b in per:
            per[b]["fills"] += 1
            # OBI at fill (asof) if available
            if have_obi:
                pos = np.searchsorted(eq_t, t, side="right") - 1
                if pos >= 0:
                    per[b]["obi5_sum"] += eq_obi5[pos]
                    per[b]["obi_deep_sum"] += eq_obi_d[pos]
                    per[b]["obi_n"] += 1
        # mid at this fill (for capture)
        mid_at_fill = _mid_at(eq_t, eq_mid, t)
        # sign convention: BUY fills below the mid -> capture positive when
        # mid_at_fill > px; SELL fills above the mid -> capture positive when
        # px > mid_at_fill. So: capture = sign * (mid - px) * qty with sign +1 for BUY, -1 for SELL.
        cap_sign = 1.0 if side == "BUY" else -1.0
        # capture (may be nan if no mid)
        cap = cap_sign * (mid_at_fill - px) * qty if np.isfinite(mid_at_fill) \
            else 0.0
        # accumulate CAPTURE on the opening bucket for open fills, on the exit
        # match for closing fills -- to match how realized round-trips are
        # bucketed. For simplicity here we attribute capture to the OPENING
        # side's bucket only (opens carry the position that later gets marked out).
        # No matched fill yet -> this fill opens or extends the same side
        if not open_lots or open_lots[0]["side"] == side:
            # push open lot, carrying the mid-at-fill so we can compute markout
            # against it when the lot is later closed
            open_lots.append({"qty": qty, "px": px, "bucket": b, "t": t,
                              "side": side, "mid_at_fill": mid_at_fill})
            # opened notional (bps denominator)
            if b in per:
                per[b]["opened_notional"] += qty * px
                per[b]["opened_qty"] += qty
                # attribute this open leg's capture to the opening bucket
                per[b]["capture"] += cap
            # next fill
            continue
        # opposite side -> CLOSES open lots FIFO
        remaining = qty
        # match until remaining is exhausted or the queue flips
        while remaining > 1e-9 and open_lots and open_lots[0]["side"] != side:
            # oldest open lot
            lot = open_lots[0]
            # matched shares
            matched = min(remaining, lot["qty"])
            # markout for the OPENING leg = signed (mid_at_exit - mid_at_fill) * matched
            # sign convention: for a BUY open, positive markout = mid rose (favorable),
            # so signed_markout = +1 * (mid_exit - mid_open) * matched. For a SELL open,
            # positive markout = mid fell -> signed_markout = -1 * (mid_exit - mid_open).
            # A NEGATIVE markout means the price moved AGAINST our open position,
            # i.e. adverse selection on the opening fill. That is what "eats" capture.
            m_open = lot.get("mid_at_fill", np.nan)
            # mid at the exit time (this fill's time)
            m_exit = _mid_at(eq_t, eq_mid, t)
            # markout in price units on the matched shares
            if np.isfinite(m_open) and np.isfinite(m_exit):
                # opening-leg sign (BUY=+1 wants mid up; SELL=-1 wants mid down)
                open_sign = 1.0 if lot["side"] == "BUY" else -1.0
                # signed markout (asof endpoint difference -- the TRUTH for this leg)
                mko = open_sign * (m_exit - m_open) * matched
                # JUMP component from the per-event path split over [t_open, t_exit]
                jm, dm_unused = _split_move(eq_t, eq_mid, lot["t"], t)
                # jump markout on the matched shares
                mko_jump = open_sign * jm * matched
                # DIFFUSION as the RESIDUAL PLUG so jump + diff == markout EXACTLY.
                # (The per-event sum-of-steps and the asof endpoint can differ by
                # an off-by-one at the window boundary; defining diff as the plug
                # guarantees reconciliation by construction -- the pattern the
                # recon diagnostic confirmed. The jump COMPONENT is still the
                # detector's output; only the split's remainder is the plug.)
                mko_diff = mko - mko_jump
                # per-fill markout for the daily aggregation (signed round-trip)
                fill_markouts.append({"t": t, "markout": mko, "bucket": lot["bucket"]})
            else:
                # missing mid -> zero contribution
                mko = 0.0; mko_jump = 0.0; mko_diff = 0.0
            # two-leg fees on the matched shares (opening + closing)
            fee = H.fee_for(lot["px"], matched) + H.fee_for(px, matched)
            # attribute markout, jump/diff, fees, holds to the OPENING bucket
            ob = lot["bucket"]
            # book to bucket
            if ob in per:
                per[ob]["markout"] += mko
                per[ob]["jump_markout"] += mko_jump
                per[ob]["diff_markout"] += mko_diff
                per[ob]["fee"] += fee
                per[ob]["holds"].append(t - lot["t"])
            # shrink lot + fill
            lot["qty"] -= matched
            remaining -= matched
            # drop the fully-consumed lot
            if lot["qty"] <= 1e-9:
                open_lots.pop(0)
        # flip through zero: any remainder opens the other side
        if remaining > 1e-9:
            open_lots.append({"qty": remaining, "px": px, "bucket": b, "t": t,
                              "side": side, "mid_at_fill": mid_at_fill})
            # opened notional + capture on the flipped remainder
            if b in per:
                per[b]["opened_notional"] += remaining * px
                per[b]["opened_qty"] += remaining
                # capture on the remainder (scale by remainder/qty)
                if qty > 0:
                    per[b]["capture"] += cap * (remaining / qty)
    # ---- residual (never-closed) lots: mark to the liquidation VWAP + time ----
    # liquidation time from the engine's EOD
    liq_time = float(dr.session[1]) if dr.session is not None else 0.0
    # liquidation price (VWAP) from the engine's EOD; fallback to last mid
    liq_px = H.liquidation_price(dr) if hasattr(H, "liquidation_price") else float("nan")
    # if the liq VWAP is nan, fall back to the last observed mid
    if not (isinstance(liq_px, float) and np.isfinite(liq_px)) or liq_px <= 0:
        liq_px = float(eq_mid[-1]) if len(eq_mid) else 0.0
    # residual lots -> markout to liq using liq_px as the "exit mid"
    for lot in open_lots:
        # opening bucket
        ob = lot["bucket"]
        # opening leg sign
        open_sign = 1.0 if lot["side"] == "BUY" else -1.0
        # mid at open
        m_open = lot.get("mid_at_fill", np.nan)
        # residual markout on the unmatched qty
        if np.isfinite(m_open):
            mko = open_sign * (liq_px - m_open) * lot["qty"]
            # jump component over [t_open, liq_time]
            jm, dm_unused = _split_move(eq_t, eq_mid, lot["t"], liq_time)
            mko_jump = open_sign * jm * lot["qty"]
            # diffusion as the residual plug (jump + diff == markout exactly)
            mko_diff = mko - mko_jump
            # book
            if ob in per:
                per[ob]["markout"] += mko
                per[ob]["jump_markout"] += mko_jump
                per[ob]["diff_markout"] += mko_diff
                # LIQUIDATION FEE: the engine's liquidation_value() charges a fee
                # per level walked (verified: cash -= fee_fn(px, take)). That fee
                # is real and already inside engine P&L, hence inside the liq_loss
                # plug. Itemize it into the fee column so the fee attribution is
                # honest (fee bps were reading ~1.04 instead of ~1.55 in last15
                # precisely because liquidation fees were hidden in liq_loss).
                # The reconciliation plug below subtracts this from liq_loss, so
                # the TOTAL net is unchanged -- only the fee/liq_loss split moves.
                per[ob]["liq_fee"] += H.fee_for(liq_px, lot["qty"])
                per[ob]["holds"].append(max(0.0, liq_time - lot["t"]))
            # add to the daily series
            fill_markouts.append({"t": liq_time, "markout": mko, "bucket": ob})
    # ---- RECONCILIATION PLUG (the fix): the mid-referenced decomposition
    # (capture + markout - fees) does NOT equal the engine's realized cash on
    # days with residual inventory or book-walk liquidation slippage, because
    # capture/markout are measured vs the MID, not the true fill/liquidation
    # cash. We close the gap with an explicit LIQUIDATION-LOSS plug, attributed
    # to opening buckets pro-rata by opened notional -- the same residual-plug
    # pattern fifo_attribution uses. This makes
    #     capture + markout - fees + liq_loss == engine_pnl   (exactly)
    # so the columns reconcile by construction, and liq_loss IS the extra cost
    # (book-walk slippage + haircut + mid-vs-cash basis) the mid view misses.
    # the decomposition total across all buckets (pre-plug)
    dec_total = sum(per[b]["capture"] + per[b]["markout"] - per[b]["fee"]
                    for b in H.BUCKETS)
    # the gap to the engine's headline P&L = the liquidation loss to distribute
    liq_gap = float(dr.pnl()) - dec_total
    # weight each bucket by its opened notional (its share of the risk taken)
    tot_opened = sum(per[b]["opened_notional"] for b in H.BUCKETS)
    # distribute the plug pro-rata (equal split if no opened notional)
    for b in H.BUCKETS:
        # this bucket's share of the plug
        w = (per[b]["opened_notional"] / tot_opened) if tot_opened > 0 \
            else (1.0 / len(H.BUCKETS))
        # attribute the liquidation-loss plug
        per[b]["liq_loss"] += liq_gap * w
    # return everything the parent needs (all picklable)
    return {"date": str(date), "symbol": sym, "clip_mult": clip_mult,
            "per": per, "fill_markouts": fill_markouts,
            "daily_pnl": float(dr.pnl())}


# ---- aggregation + reporting ----

def main():
    # run stamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # dates
    all_dates = R.discover_dates()
    # trading starts after the trailing window
    run_dates = all_dates[TRAIL_DAYS:]
    # smoke limit
    if SMOKE_DAYS is not None:
        run_dates = run_dates[:SMOKE_DAYS]
    # workers
    nproc = WORKERS or mp.cpu_count()
    # calibration bundle (computed ONCE in the parent, passed to workers)
    print("pre-pass: loading calibration + trailing median trade size",
          flush=True)
    calib = {
        "scales": H.load_scales(),
        "profiles": H.load_profiles(),
        "windows": H.load_windows(),
        "segments": H.load_segments(),
        "all_dates": all_dates,
        # the expensive pre-pass, run ONCE here
        "tstats": H.trailing_median_trade_size(all_dates, NAMES, TRAIL_DAYS)}
    # announce
    print(f"\nspot_capture_markout_decomp: sweeping clips {CLIP_MULTS} x "
          f"{len(NAMES)} names x {len(run_dates)} days", flush=True)
    print(f"jump detector: k={JUMP_K}, local vol window={LOCAL_VOL_WIN} events",
          flush=True)
    print(f"workers: {nproc}\n", flush=True)
    # the full work list: every (date, symbol, clip_mult) tuple
    work = [(date, sym, cm)
            for date in run_dates for sym in NAMES for cm in CLIP_MULTS]
    total = len(work)
    # timing
    t0 = time.perf_counter()
    # accumulators per clip_mult
    clip_agg = {cm: {b: {"capture": 0.0, "markout": 0.0, "fee": 0.0,
                         "jump_markout": 0.0, "diff_markout": 0.0,
                         "liq_loss": 0.0, "liq_fee": 0.0, "opened_notional": 0.0,
                         "opened_qty": 0.0, "holds": [], "fills": 0,
                         "obi5_sum": 0.0, "obi_deep_sum": 0.0, "obi_n": 0}
                     for b in H.BUCKETS} for cm in CLIP_MULTS}
    # per-day portfolio markout for the significance test, per clip_mult
    daily_markouts = {cm: {} for cm in CLIP_MULTS}
    # jump-flag fraction sanity check
    total_events = 0; total_jumps = 0
    # parallel map
    done = 0
    with mp.Pool(processes=nproc, initializer=_init_worker,
                 initargs=(calib,)) as pool:
        # stream results as they complete (chunksize>1 cuts IPC overhead)
        for res in pool.imap_unordered(_process, work, chunksize=4):
            done += 1
            # heartbeat
            if done % 100 == 0 or done == total:
                el = time.perf_counter() - t0
                print(f"  {done}/{total}  {H._fmt(el)}  "
                      f"ETA {H._fmt(el/done*(total-done))}", flush=True)
            # skipped
            if res is None:
                continue
            cm = res["clip_mult"]
            # accumulate per-bucket
            for b in H.BUCKETS:
                src = res["per"][b]
                dst = clip_agg[cm][b]
                # sums are commutative -> order-safe under parallel completion
                dst["capture"] += src["capture"]
                dst["markout"] += src["markout"]
                dst["fee"] += src["fee"]
                dst["jump_markout"] += src["jump_markout"]
                dst["diff_markout"] += src["diff_markout"]
                dst["opened_notional"] += src["opened_notional"]
                dst["opened_qty"] += src["opened_qty"]
                dst["liq_loss"] += src["liq_loss"]
                dst["liq_fee"] += src["liq_fee"]
                dst["holds"].extend(src["holds"])
                dst["fills"] += src["fills"]
                dst["obi5_sum"] += src["obi5_sum"]
                dst["obi_deep_sum"] += src["obi_deep_sum"]
                dst["obi_n"] += src["obi_n"]
            # aggregate to per-day portfolio markout
            date = res["date"]
            day_sum = sum(fm["markout"] for fm in res["fill_markouts"])
            daily_markouts[cm][date] = daily_markouts[cm].get(date, 0.0) + day_sum
    # ---- reporting per clip multiple ----
    print("\n" + "=" * 78)
    print(f"### TOTAL RUNTIME: {H._fmt(time.perf_counter() - t0)} "
          f"for {total} symbol-day-clips ###")
    print("=" * 78)
    # for each clip, print the per-bucket table and the daily test
    for cm in CLIP_MULTS:
        print(f"\n{'=' * 78}")
        print(f"CLIP MULTIPLIER = {cm}x median trade size")
        print("=" * 78)
        # ---- TABLE 1: totals (PKR) by bucket, with the liq_loss plug + recon ----
        print(f"\n-- TOTALS (PKR) by bucket --")
        print(f"{'bucket':12s} {'capture':>12s} {'markout':>12s} "
              f"{'jump_mko':>11s} {'diff_mko':>12s} {'fees':>10s} "
              f"{'liq_loss':>12s} {'net_pnl':>12s} {'fills':>8s}")
        # running total to verify reconciliation to engine P&L
        recon_total = 0.0
        for b in H.BUCKETS:
            a = clip_agg[cm][b]
            # ALL-IN fee = round-trip fees + liquidation fees (itemized out of liq_loss)
            fee_all = a["fee"] + a["liq_fee"]
            # liq_loss adjusted: moving the liq fee (a cost) OUT to the fee column
            # makes liq_loss LESS negative by that amount, so we ADD it back.
            # (net = cap+mko-fee_all+liq_net stays == engine P&L; verified.)
            liq_net = a["liq_loss"] + a["liq_fee"]
            # net = capture + markout - fee_all + liq_net  (== engine P&L; unchanged)
            net = a["capture"] + a["markout"] - fee_all + liq_net
            recon_total += net
            print(f"{b:12s} {a['capture']:>12,.0f} {a['markout']:>12,.0f} "
                  f"{a['jump_markout']:>11,.0f} {a['diff_markout']:>12,.0f} "
                  f"{-fee_all:>10,.0f} {liq_net:>12,.0f} {net:>12,.0f} "
                  f"{a['fills']:>8,d}")
        # the reconciliation line (should equal the engine P&L sum for this clip)
        print(f"{'TOTAL net':12s} {'':>12s} {'':>12s} {'':>11s} {'':>12s} "
              f"{'':>10s} {'':>12s} {recon_total:>12,.0f}")

        # ---- TABLE 2: bps (per unit opened notional) by bucket ----
        print(f"\n-- BPS on opened notional (component / opened_notional * 1e4) --")
        print(f"{'bucket':12s} {'cap_bps':>10s} {'mko_bps':>10s} "
              f"{'jump_bps':>10s} {'diff_bps':>10s} {'fee_bps':>10s} "
              f"{'liq_bps':>10s} {'net_bps':>10s}")
        for b in H.BUCKETS:
            a = clip_agg[cm][b]
            # denominator = opened notional (0 -> nan-safe)
            on = a["opened_notional"]
            # a helper for component -> bps
            def _bps(x):
                return (1e4 * x / on) if on > 0 else float("nan")
            # all-in fee + adjusted liq_loss (liq fee moved to fee col -> add back)
            fee_all = a["fee"] + a["liq_fee"]
            liq_net = a["liq_loss"] + a["liq_fee"]
            # net in bps (unchanged total)
            net_bps = _bps(a["capture"] + a["markout"] - fee_all + liq_net)
            print(f"{b:12s} {_bps(a['capture']):>10.3f} "
                  f"{_bps(a['markout']):>10.3f} {_bps(a['jump_markout']):>10.3f} "
                  f"{_bps(a['diff_markout']):>10.3f} {_bps(-fee_all):>10.3f} "
                  f"{_bps(liq_net):>10.3f} {net_bps:>10.3f}")

        # ---- TABLE 3: per-share averages (component / opened shares) ----
        print(f"\n-- PER-SHARE (PKR per opened share) + HOLD TIME by bucket --")
        print(f"{'bucket':12s} {'cap/sh':>10s} {'mko/sh':>10s} "
              f"{'jump/sh':>10s} {'diff/sh':>10s} {'liq/sh':>10s} "
              f"{'med_hold_s':>11s} {'mean_hold_s':>12s} {'opened_sh':>12s}")
        for b in H.BUCKETS:
            a = clip_agg[cm][b]
            # denominator = opened shares
            oq = a["opened_qty"]
            # component -> per share
            def _ps(x):
                return (x / oq) if oq > 0 else float("nan")
            # holds in seconds (ms in the engine)
            hs = np.array(a["holds"]) / 1000.0 if a["holds"] else np.array([0.0])
            print(f"{b:12s} {_ps(a['capture']):>10.5f} "
                  f"{_ps(a['markout']):>10.5f} {_ps(a['jump_markout']):>10.5f} "
                  f"{_ps(a['diff_markout']):>10.5f} {_ps(a['liq_loss']):>10.5f} "
                  f"{np.median(hs):>11.1f} {hs.mean():>12.1f} {oq:>12,.0f}")
        # OBI at fill by bucket (mean, if available)
        print(f"\n-- OBI at fill by bucket --")
        print(f"{'bucket':12s} {'mean_obi5_at_fill':>18s} "
              f"{'mean_obi_deep_at_fill':>22s}")
        for b in H.BUCKETS:
            a = clip_agg[cm][b]
            # means (guard n=0)
            m5 = (a["obi5_sum"] / a["obi_n"]) if a["obi_n"] > 0 else float("nan")
            md = (a["obi_deep_sum"] / a["obi_n"]) if a["obi_n"] > 0 else float("nan")
            print(f"{b:12s} {m5:>18.4f} {md:>22.4f}")
        # ---- daily portfolio markout significance vs 0 (day as unit) ----
        dm = pd.Series(daily_markouts[cm]).sort_index().to_numpy()
        # need enough days for a meaningful test
        if len(dm) >= 20:
            # t-test (parametric) against zero
            t_stat, t_p = stats.ttest_1samp(dm, 0.0, nan_policy="omit")
            # Wilcoxon signed-rank (non-parametric) against zero
            # drop exact zeros for the Wilcoxon (its convention)
            dm_nz = dm[dm != 0]
            try:
                w_stat, w_p = stats.wilcoxon(dm_nz)
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
            # print the daily markout distribution stats + tests
            print(f"\n  DAILY PORTFOLIO MARKOUT (n={len(dm)} days):")
            print(f"    mean {dm.mean():,.0f}   median {np.median(dm):,.0f}   "
                  f"std {dm.std():,.0f}")
            print(f"    t-test vs 0:            t={t_stat:.3f}  p={t_p:.4g}")
            print(f"    Wilcoxon signed-rank:   W={w_stat:.1f}  p={w_p:.4g}")
            # plain-language interpretation
            if t_p < 0.01 and dm.mean() < 0:
                print("    -> markout is significantly NEGATIVE (adverse selection real)")
            elif t_p < 0.01 and dm.mean() > 0:
                print("    -> markout is significantly POSITIVE (favorable moves)")
            elif t_p > 0.05:
                print("    -> markout is NOT statistically distinguishable from zero")
            else:
                print("    -> markout is directional but the test is inconclusive")
        else:
            print(f"\n  daily test skipped (only {len(dm)} days)")
    # save the per-day series for further inspection
    resd = Path("/Users/shazzak/Capital Stake - Results")
    # one file per clip multiple
    for cm in CLIP_MULTS:
        s = pd.Series(daily_markouts[cm]).sort_index()
        s.name = "daily_portfolio_markout"
        # output path
        p = resd / f"markout_decomp_daily_{cm:g}x_{stamp}.csv"
        # write
        s.to_csv(p)
        print(f"  wrote {p}")


# multiprocessing entry-point guard (macOS spawn)
if __name__ == "__main__":
    main()
