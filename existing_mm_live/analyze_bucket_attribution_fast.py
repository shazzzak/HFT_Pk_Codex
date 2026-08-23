# analyze_bucket_attribution_fast.py -- PARALLEL twin of the attribution consumer.
#
# WHY: the serial version took ~152 min for the full run. The engine's snapshot
# bottleneck is ALREADY fixed (snapshot_prep/PreparsedSnapshot), so the remaining
# cost is the per-symbol-day backtest itself -- and those symbol-days are
# INDEPENDENT (run_symbol_day reads its own data, uses a seeded latency model, and
# shares no mutable state). So the clean, certain win is to run them in parallel
# across CPU cores. On an M4 (10+ cores) this should cut wall-clock ~8-10x.
#
# It also carries a cProfile toggle: rather than ASSUME where the time goes (the
# snapshot fix means it is NOT Book.snapshot anymore), PROFILE=True on the 5-day
# smoke tells you the real hot spot from the data.
#
# IDENTICAL RESULTS: same params, same run_symbol_day, same fifo_attribution, same
# aggregation and output columns as analyze_bucket_attribution.py -- only the
# iteration is parallel and order-independent (the accumulators are commutative
# sums / list-extends, so worker order does not change the totals).
#
# Run from existing_mm_live/:  caffeinate -is python3 analyze_bucket_attribution_fast.py

# filesystem paths
from pathlib import Path
# wall-clock timing
import time
# run stamp
from datetime import datetime
# multiprocessing for the parallel symbol-day map
import multiprocessing as mp
# profiler (optional, toggled)
import cProfile
import pstats
import io
# numeric arrays
import numpy as np
# dataframes
import pandas as pd
# shared scaffolding (backtest driver, FIFO attribution, stats)
import mm_harness as H
# the driver module (dates + datasets)
import run_legacy_mm as R

# ------------------------------ config ---------------------------------------
# the production book (top-10 winners)
NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
# the locked production clip multiple
CLIP_MULT = 3.0
# trailing window (days) for the median-trade-size clip anchor
TRAIL_DAYS = 10
# number of worker processes (None -> all cores; set lower if memory-bound)
WORKERS = None
# SMOKE_DAYS: run only the first N trading days (after the trailing window) so
# you can match the serial 5-day result; None -> full run
SMOKE_DAYS = 5
# PROFILE: wrap a SINGLE worker call in cProfile to reveal the true hot spot
# (prints the top cumulative-time functions). Turn off for the real run.
PROFILE = False
# -----------------------------------------------------------------------------

# module-level globals populated ONCE per worker process (via the initializer).
# CRITICAL: the calibration (scales/profiles/windows/segments/tstats) is computed
# ONCE IN THE PARENT and passed into the initializer -- NOT recomputed per worker.
# (An earlier version recomputed the full-history trailing-median pre-pass inside
# every worker, i.e. N times; that made the pre-pass the bottleneck.)
_G = {}


# worker initializer: receives the pre-computed calibration bundle and stashes it
def _init_worker(calib):
    # unpack the shared, read-only calibration into this worker's globals
    _G.update(calib)


# process ONE (date, symbol) -> the attribution + panel row, or None to skip.
# self-contained: opens its own datasets, runs the backtest, returns plain dicts
# (picklable) so the parent can aggregate. Identical logic to the serial loop.
def _process_symbol_day(args):
    # unpack the work item
    date, sym = args
    # calibration from the process globals
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
    # the walk-forward clip anchor (trailing median, prior days only)
    med = H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
    # not enough history -> skip
    if med is None or med <= 0:
        return None
    # missing calibration -> skip
    if sym not in scales or sym not in profiles:
        return None
    # open the date's datasets (each worker opens independently)
    dsets = R.open_datasets(date)
    # missing partition -> skip
    if dsets is None:
        return None
    # the production clip = CLIP_MULT x trailing median trade size
    clip = max(1, int(round(CLIP_MULT * med)))
    # assemble the locked production params
    params = H.build_micro_params(
        clip, scales[sym], profiles[sym],
        windows.get(sym, (5.0, 1.0)), segs)
    # THE single backtest path
    dr = H.run_symbol_day(date, sym, dsets, params)
    # unrunnable day -> skip
    if dr is None or dr.pnl() is None:
        return None
    # FIFO open-bucket attribution (reconciles by construction)
    per = H.fifo_attribution(dr.fills.to_dict("records"),
                             float(dr.pnl()), dr.session[1])
    # attributed total across buckets
    attr_total = sum(per[b]["realized"] for b in H.BUCKETS)
    # execution + inventory + intraday panels
    es = H.execution_stats(dr)
    inv = H.inventory_stats(dr)
    _, cum = H.intraday_pnl_path(dr)
    # the panel row (identical columns to the serial version)
    row = {"date": str(date), "symbol": sym,
           "pnl": round(float(dr.pnl()), 2),
           "otr": round(es["otr"], 2),
           "cancel_to_trade": round(es["cancel_to_trade"], 2),
           "fill_ratio": (round(es["fill_ratio"], 4)
                          if np.isfinite(es["fill_ratio"]) else np.nan),
           "intraday_maxdd": round(H.max_drawdown(cum), 2),
           "max_abs_inv": inv["max_abs_inv"],
           "twa_abs_inv": round(inv["twa_abs_inv"], 1),
           "eod_inv": inv["eod_inv"]}
    # return everything the parent needs to aggregate (all picklable)
    return {"per": {b: {"realized": per[b]["realized"],
                        "opened_notional": per[b]["opened_notional"],
                        "holds": per[b]["holds"], "fills": per[b]["fills"]}
                    for b in H.BUCKETS},
            "recon_err": attr_total - float(dr.pnl()),
            "daily_pnl": float(dr.pnl()),
            "row": row}


def main():
    # run stamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # all trading dates
    all_dates = R.discover_dates()
    # ---- compute the shared calibration ONCE in the parent (not per worker) ----
    print("pre-pass: loading calibration + trailing median trade size", flush=True)
    # the read-only calibration bundle every worker needs
    calib = {
        # per-symbol session scales
        "scales": H.load_scales(),
        # per-symbol volume profiles
        "profiles": H.load_profiles(),
        # per-symbol EOD windows
        "windows": H.load_windows(),
        # per-date session segments
        "segments": H.load_segments(),
        # all trading dates
        "all_dates": all_dates,
        # per (symbol, day) trailing median trade size -- the expensive pre-pass,
        # now run exactly ONCE here instead of once per worker
        "tstats": H.trailing_median_trade_size(all_dates, NAMES, TRAIL_DAYS)}
    # trading starts after the trailing window
    run_dates = all_dates[TRAIL_DAYS:]
    # SMOKE limit to match the serial 5-day check
    if SMOKE_DAYS is not None:
        run_dates = run_dates[:SMOKE_DAYS]
    # the full work list: every (date, symbol) pair
    work = [(date, sym) for date in run_dates for sym in NAMES]
    # total work items
    total = len(work)
    # announce
    print(f"analyze_bucket_attribution_fast: base {CLIP_MULT:g}x, "
          f"{len(NAMES)} names x {len(run_dates)} days = {total} symbol-days",
          flush=True)
    # worker count
    nproc = WORKERS or mp.cpu_count()
    # announce parallelism
    print(f"parallel workers: {nproc}   profile: {PROFILE}\n", flush=True)
    # run timer
    t0 = time.perf_counter()

    # ---- PROFILE path: run ONE symbol-day under cProfile to find the hot spot --
    if PROFILE:
        # stash the already-computed calibration in THIS process for profiling
        _init_worker(calib)
        # profiler
        pr = cProfile.Profile()
        # profile the first runnable work item
        pr.enable()
        # process a handful so the profile is representative
        for w in work[:min(10, len(work))]:
            _process_symbol_day(w)
        pr.disable()
        # format the top-20 by cumulative time
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        # show the profile
        print("=== cPROFILE (top 20 cumulative) -- the REAL hot spot ===")
        print(s.getvalue())
        # stop here; profiling is diagnostic only
        return

    # ---- parallel map over symbol-days ----
    # per-bucket accumulators
    agg = {b: {"realized": 0.0, "opened_notional": 0.0, "holds": [], "fills": 0}
           for b in H.BUCKETS}
    # reconciliation errors + daily P&Ls + panel rows
    recon_err = []
    daily_pnls = []
    exec_rows = []
    # completed counter for the heartbeat
    done = 0
    # the process pool (each worker inits calibration once)
    with mp.Pool(processes=nproc, initializer=_init_worker,
                 initargs=(calib,)) as pool:
        # stream results as they complete (chunksize>1 cuts IPC overhead)
        for res in pool.imap_unordered(_process_symbol_day, work, chunksize=4):
            # tick the counter
            done += 1
            # heartbeat + ETA every 20 completed and at the end
            if done % 20 == 0 or done == total:
                # elapsed
                el = time.perf_counter() - t0
                # progress + ETA
                print(f"  {done}/{total}  {H._fmt(el)}  "
                      f"ETA {H._fmt(el/done*(total-done))}", flush=True)
            # skipped symbol-day
            if res is None:
                continue
            # accumulate the per-bucket attribution (commutative -> order-safe)
            for b in H.BUCKETS:
                # realized round-trip P&L
                agg[b]["realized"] += res["per"][b]["realized"]
                # opened notional (bps denominator)
                agg[b]["opened_notional"] += res["per"][b]["opened_notional"]
                # holding times
                agg[b]["holds"].extend(res["per"][b]["holds"])
                # fills
                agg[b]["fills"] += res["per"][b]["fills"]
            # reconciliation error
            recon_err.append(res["recon_err"])
            # daily P&L
            daily_pnls.append(res["daily_pnl"])
            # panel row
            exec_rows.append(res["row"])

    # ---- outputs (identical to the serial version) ----
    # results directory
    resd = Path("/Users/shazzak/Capital Stake - Results")
    # panel frame
    ex = pd.DataFrame(exec_rows)
    # panel path
    panel_path = resd / f"attribution_panel_fast_{stamp}.csv"
    # write the panel
    ex.to_csv(panel_path, index=False)

    # per-bucket attribution table
    print("\n=== FIFO OPEN-BUCKET ATTRIBUTION (base 3x, no fixed horizon) ===")
    # header
    print(f"{'bucket':12s} {'realized_pnl':>13s} {'bps_opened':>11s} "
          f"{'med_hold_s':>11s} {'mean_hold_s':>11s} {'fills':>9s}")
    # one row per bucket
    for b in H.BUCKETS:
        # this bucket's accumulator
        a = agg[b]
        # bps on opened notional
        bps = (1e4 * a["realized"] / a["opened_notional"]
               if a["opened_notional"] > 0 else 0.0)
        # holds as an array (seconds)
        hs = np.array(a["holds"]) / 1000.0 if a["holds"] else np.array([0.0])
        # print the bucket row
        print(f"{b:12s} {a['realized']:>13,.0f} {bps:>11.3f} "
              f"{np.median(hs):>11.1f} {hs.mean():>11.1f} {a['fills']:>9,d}")

    # reconciliation summary
    re = np.array(recon_err) if recon_err else np.array([0.0])
    # print recon
    print(f"\nRECONCILIATION (attributed - engine daily P&L), {len(re)} symbol-days:")
    print(f"  mean abs {np.abs(re).mean():.4f} PKR   max abs {np.abs(re).max():.4f} PKR")
    # exact?
    print("  -> EXACT" if np.abs(re).max() < 1e-6 else "  -> residual remains")

    # run-level risk panel
    dp = np.array(daily_pnls) if daily_pnls else np.array([0.0])
    # annualisation factor (252 trading days)
    ann = np.sqrt(252)
    # Sharpe / Sortino
    sharpe = (dp.mean() / dp.std() * ann) if dp.std() > 0 else np.nan
    # downside deviation
    downside = dp[dp < 0]
    sortino = (dp.mean() / downside.std() * ann
               if len(downside) > 1 and downside.std() > 0 else np.nan)
    # print the risk panel
    print("\n=== RUN-LEVEL RISK PANEL ===")
    print(f"  mean daily P&L:   {dp.mean():,.0f}")
    print(f"  Sharpe (ann.):    {sharpe:.2f}")
    print(f"  Sortino (ann.):   {sortino:.2f}")

    # total wall-clock
    print(f"\n### TOTAL RUNTIME: {H._fmt(time.perf_counter()-t0)} "
          f"for {total} symbol-days ({nproc} workers) ###")
    # wrote
    print(f"wrote {panel_path}")


# multiprocessing entry-point guard (required on macOS spawn start method)
if __name__ == "__main__":
    # macOS uses spawn; guard ensures workers re-import cleanly
    main()
