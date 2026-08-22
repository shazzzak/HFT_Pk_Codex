# analyze_bucket_attribution.py -- THIN CONSUMER of mm_harness.
#
# Replaces the standalone attribute_fills_fifo.py + decompose_buckets.py: all the
# scaffolding (loaders, backtest driver, FIFO attribution, stats) lives in
# mm_harness and is imported, not copied. This script is ONLY the config + the
# per-bucket report. Reconciliation asserted: attributed realized == engine P&L.
#
# Run from existing_mm_live/:  caffeinate -is python3 analyze_bucket_attribution.py

# filesystem paths
from pathlib import Path
# wall-clock timing for the heartbeat
import time
# run stamp for output filenames
from datetime import datetime
# numeric arrays
import numpy as np
# dataframes
import pandas as pd
# EVERYTHING shared comes from the harness -- no copied scaffolding
import mm_harness as H
# the driver module (dates + datasets come from here)
import run_legacy_mm as R

# ------------------------------ config ---------------------------------------
# the top-10 winners (the production book)
NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
# the production clip multiple (the locked knee)
CLIP_MULT = 3.0
# trailing window (days) for the median-trade-size clip anchor
TRAIL_DAYS = 10
# -----------------------------------------------------------------------------


def main():
    # timestamp for the output files
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # per-symbol back-solved session scales (harness loader)
    scales = H.load_scales()
    # per-symbol 4-bucket volume profiles (harness loader)
    profiles = H.load_profiles()
    # per-symbol EOD ramp/cliff windows (harness loader)
    windows = H.load_windows()
    # per-date session segments (harness loader)
    segments = H.load_segments()
    # every trading date in the parsed store
    all_dates = R.discover_dates()

    # announce the pre-pass
    print("pre-pass: trailing median trade size", flush=True)
    # per (symbol, day) median trade size (harness owns the loop)
    tstats = H.trailing_median_trade_size(all_dates, NAMES, TRAIL_DAYS)

    # trading starts after the trailing window has history
    run_dates = all_dates[TRAIL_DAYS:]
    # total symbol-days for the ETA
    total = len(run_dates) * len(NAMES)
    # per-bucket accumulators: realized, opened notional, holds, fills
    agg = {b: {"realized": 0.0, "opened_notional": 0.0, "holds": [], "fills": 0}
           for b in H.BUCKETS}
    # per-day reconciliation errors (attributed - engine P&L)
    recon_err = []
    # daily P&Ls for the run-level Sharpe / drawdown
    daily_pnls = []
    # per symbol-day execution + risk panel rows
    exec_rows = []
    # announce the run
    print(f"\nanalyze_bucket_attribution: base {CLIP_MULT:g}x, {len(NAMES)} names "
          f"x {len(run_dates)} days (FIFO open-bucket, no fixed horizon)\n",
          flush=True)
    # run timer
    t0 = time.perf_counter()
    # symbol-day counter for the heartbeat
    sd = 0
    # outer loop: dates
    for date in run_dates:
        # open the date's datasets once (shared across symbols)
        dsets = R.open_datasets(date)
        # missing partition -> skip the date
        if dsets is None:
            continue
        # the day's session segments
        segs = segments.get(str(date))
        # no segments -> skip the date (unwind model needs them)
        if segs is None:
            continue
        # inner loop: symbols
        for sym in NAMES:
            # tick the heartbeat counter
            sd += 1
            # the walk-forward clip anchor (median trade size, prior days only)
            med = H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
            # not enough history -> skip the symbol-day
            if med is None or med <= 0:
                continue
            # missing calibration -> skip rather than guess
            if sym not in scales or sym not in profiles:
                continue
            # the production clip: CLIP_MULT x the trailing median trade size
            clip = max(1, int(round(CLIP_MULT * med)))
            # assemble the locked production params via the harness
            params = H.build_micro_params(
                clip, scales[sym], profiles[sym],
                windows.get(sym, (5.0, 1.0)), segs)
            # THE single backtest path (shared driver)
            dr = H.run_symbol_day(date, sym, dsets, params)
            # unrunnable day or missing headline P&L -> skip
            if dr is None or dr.pnl() is None:
                continue
            # the true residual-liquidation price for the FIFO close-out
            liq_px = H.liquidation_price(dr)
            # FIFO open-bucket attribution to each fill's actual offset
            per = H.fifo_attribution(dr.fills.to_dict("records"), liq_px,
                                     dr.session[1])
            # attributed total across buckets
            attr_total = sum(per[b]["realized"] for b in H.BUCKETS)
            # reconciliation error vs the engine's daily P&L
            recon_err.append(attr_total - float(dr.pnl()))
            # collect the daily P&L for the run-level risk panel
            daily_pnls.append(float(dr.pnl()))
            # accumulate the per-bucket attribution
            for b in H.BUCKETS:
                # realized round-trip P&L attributed to this bucket
                agg[b]["realized"] += per[b]["realized"]
                # opened notional (the bps denominator)
                agg[b]["opened_notional"] += per[b]["opened_notional"]
                # open->close holding times
                agg[b]["holds"].extend(per[b]["holds"])
                # fills counted in this bucket
                agg[b]["fills"] += per[b]["fills"]
            # the execution/compliance panel for this symbol-day (shared)
            es = H.execution_stats(dr)
            # the inventory-risk panel for this symbol-day (shared)
            inv = H.inventory_stats(dr)
            # the intraday cumulative P&L path (shared)
            _, cum = H.intraday_pnl_path(dr)
            # one panel row per symbol-day
            exec_rows.append({"date": str(date), "symbol": sym,
                              "pnl": round(float(dr.pnl()), 2),
                              "otr": round(es["otr"], 2),
                              "cancel_to_trade": round(es["cancel_to_trade"], 2),
                              "fill_ratio": round(es["fill_ratio"], 4)
                              if np.isfinite(es["fill_ratio"]) else np.nan,
                              "intraday_maxdd": round(H.max_drawdown(cum), 2),
                              "max_abs_inv": inv["max_abs_inv"],
                              "twa_abs_inv": round(inv["twa_abs_inv"], 1),
                              "eod_inv": inv["eod_inv"]})
            # heartbeat + ETA every 10 symbol-days and at the end
            if sd % 10 == 0 or sd == total:
                # elapsed run time
                el = time.perf_counter() - t0
                # progress + projection
                print(f"  {sd}/{total}  {H._fmt(el)}  "
                      f"ETA {H._fmt(el/sd*(total-sd))}", flush=True)

    # the results directory
    res = Path("/Users/shazzak/Capital Stake - Results")
    # the per-symbol-day panel as a frame
    ex = pd.DataFrame(exec_rows)
    # timestamped panel output path
    panel_path = res / f"attribution_panel_{stamp}.csv"
    # write the panel
    ex.to_csv(panel_path, index=False)

    # ---- per-bucket attribution table ----
    print("\n=== FIFO OPEN-BUCKET ATTRIBUTION (base 3x, no fixed horizon) ===")
    # table header
    print(f"{'bucket':12s} {'realized_pnl':>13s} {'bps_opened':>11s} "
          f"{'med_hold_s':>11s} {'mean_hold_s':>11s} {'fills':>9s}")
    # one row per bucket
    for b in H.BUCKETS:
        # this bucket's accumulator
        a = agg[b]
        # realized P&L per opened notional, in bps
        bps = (1e4 * a["realized"] / a["opened_notional"]
               if a["opened_notional"] > 0 else float("nan"))
        # median open->close holding time in seconds
        med_h = np.median(a["holds"]) / 1000.0 if a["holds"] else float("nan")
        # mean open->close holding time in seconds
        mean_h = np.mean(a["holds"]) / 1000.0 if a["holds"] else float("nan")
        # the formatted bucket row
        print(f"{b:12s} {a['realized']:>13,.0f} {bps:>11.3f} "
              f"{med_h:>11.1f} {mean_h:>11.1f} {a['fills']:>9,}")

    # ---- reconciliation ----
    # the reconciliation errors as an array
    re = np.array(recon_err)
    # report the reconciliation quality
    print(f"\nRECONCILIATION (attributed - engine daily P&L), {len(re)} symbol-days:")
    # mean and max absolute error
    print(f"  mean abs {np.mean(np.abs(re)):.4f} PKR   max abs {np.max(np.abs(re)):.4f} PKR")
    # verdict: exact (trustworthy) or investigate
    print("  -> EXACT" if np.max(np.abs(re)) < 1.0
          else "  -> WARNING: does not reconcile; investigate")

    # ---- run-level risk panel ----
    # Sharpe/Sortino over the daily P&Ls
    ss = H.sharpe_sortino(daily_pnls)
    # panel header
    print(f"\n=== RUN-LEVEL RISK PANEL ===")
    # mean daily P&L
    print(f"  mean daily P&L:   {ss['mean_daily']:,.0f}")
    # annualised Sharpe
    print(f"  Sharpe (ann.):    {ss['sharpe']:.2f}")
    # annualised Sortino
    print(f"  Sortino (ann.):   {ss['sortino']:.2f}")
    # run-level (multi-day) max drawdown
    print(f"  run-level max DD: {H.run_level_drawdown(daily_pnls):,.0f}")
    # execution panel medians when we have rows
    if len(ex):
        # panel header
        print(f"\n=== EXECUTION PANEL (medians across symbol-days) ===")
        # median order-to-trade ratio
        print(f"  OTR:              {ex['otr'].median():.1f}")
        # median cancel-to-trade ratio
        print(f"  cancel-to-trade:  {ex['cancel_to_trade'].median():.1f}")
        # median fill ratio
        print(f"  fill ratio:       {ex['fill_ratio'].median():.4f}")
        # median intraday max drawdown
        print(f"  intraday max DD:  {ex['intraday_maxdd'].median():,.0f} (median day)")
        # worst single-day intraday drawdown
        print(f"  worst intraday DD:{ex['intraday_maxdd'].min():,.0f}")
    # the panel output location
    print(f"\nwrote {panel_path}")


# entry point
if __name__ == "__main__":
    main()
