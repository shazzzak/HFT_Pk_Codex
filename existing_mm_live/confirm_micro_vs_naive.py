# confirm_micro_vs_naive.py -- validation gate: does capture-recalibrated micro
# beat the naive benchmark, per symbol, on the full 207 days? Measures P&L TWO
# independent ways and reconciles them:
#   Path A -- Backtester true EOD P&L in PKR (cash + book-walk liquidation of
#             leftover inventory, from bt.eod["equity_liquidated"]).
#   Path B -- attribution net bps/fill (capture+markout-fee), same fills, the
#             lens that produced the +2.78/-0.00 naive benchmark.
# Micro is tested at the sweep winner 0.0005 AND neighbors 0.0003/0.0007 (all at
# improve_ticks=0.0) for robustness. Naive is re-derived fresh (not reusing the
# stored benchmark) as a drift check. Runs with the STILL-BROKEN inventory skew
# on purpose: this is the capture-only baseline the later skew fix must beat.
#
# Run from existing_mm_live/:  python confirm_micro_vs_naive.py

# paths
from pathlib import Path
# timing + heartbeat
import time
# frames + arrays
import pandas as pd
import numpy as np
# driver + engine + strategies
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel, NaiveSymmetricMM
from micro_mm import MicrostructureMM
# attribution economics (Path B) -- single source of truth for the bps math
import fill_attribution as FA
# context join (attach mid0/mid_h/etc to fills for Path B)
import persist_fills as PF

# point the loader at the raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# feature store (Path B context join source; PPL/UBL already built)
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")

# pilot symbols (feature stores already exist for these two)
SYMBOLS = ["PPL", "UBL"]
# micro configs to test: the sweep winner plus two neighbors, all at improve_ticks=0.0
MICRO_CONFIGS = [
    {"min_edge_pct": 0.0003, "improve_ticks": 0.0},
    {"min_edge_pct": 0.0005, "improve_ticks": 0.0},
    {"min_edge_pct": 0.0007, "improve_ticks": 0.0},
]
# nominal shares per fill (for any share-weighted reporting)
SIZE = R.MICRO_PARAMS["size"]


# build a strategy for a given name + (for micro) parameter overrides
def make_strategy(name, session_ms, overrides):
    # naive ignores overrides and session_ms
    if name == "naive":
        return NaiveSymmetricMM(**R.STRAT)
    # micro: start from MICRO_PARAMS, apply the config overrides (min_edge/improve)
    params = dict(R.MICRO_PARAMS)
    # apply each override key
    params.update(overrides)
    # construct micro at these params
    return MicrostructureMM(session_ms=session_ms, **params)


# run ONE (strategy, config, symbol, day); return (fills_df, eod_pnl, clean_flag)
def run_one(name, overrides, sym, date, dsets):
    # load the three tables for this symbol-day
    u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
    t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
    # need book + trades
    if len(t) == 0 or len(s) == 0:
        return None, 0.0, None
    # build events (fast pre-parsed snapshots)
    events, snap_groups, t = R.build_events(u, s, t)
    # continuous-session window
    cont = t[t["initiator"] != "AUCTION"]
    if len(cont) == 0:
        return None, 0.0, None
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
    # cfg with the SAME seeded latency used everywhere (reconciliation)
    cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED))
    # build the strategy + run the real backtest
    bt = Backtester(make_strategy(name, (t0, t1), overrides), cfg)
    fills, equity, stats = bt.run(events, snap_groups)
    # Path A: read the TRUE EOD P&L off the instance (cash + book-walk liquidation)
    if bt.eod is not None:
        # realizable total P&L for the day
        eod_pnl = float(bt.eod["equity_liquidated"])
        # whether the position fully liquidated (False => estimate, not realizable)
        clean = bool(bt.eod["liquidation_clean"])
    else:
        # no eod report (e.g. no session) -> zero P&L, mark clean unknown
        eod_pnl, clean = 0.0, None
    # hand back the fills (for Path B), the day's EOD P&L, and the clean flag
    return fills, eod_pnl, clean


# score a run's fills through the attribution economics (Path B); mean net bps
def score_bps(fills, sym, date):
    # no fills -> no bps
    if fills is None or len(fills) == 0:
        return 0, np.nan, np.nan, np.nan
    # the feature-store partition for the context join
    fs_path = FS_ROOT / sym / f"date={date}.parquet"
    # cannot score without context
    if not fs_path.exists():
        return len(fills), np.nan, np.nan, np.nan
    # load the context columns
    fs_day = pd.read_parquet(
        fs_path,
        columns=["ts_exch", "mid", "spread_bps", "obi_1",
                 "toxicity", "realized_vol_bps"])
    # attach mid0/mid_h/etc at fill time (tested no-leak join)
    f = PF.join_fill_context(fills, fs_day)
    # engine side string -> +/-1 (BUY=+1 our bid filled, SELL=-1)
    side_sgn = np.where(f["side"] == "BUY", 1.0, -1.0)
    # per-fill economics in bps (attribution's own functions)
    f["capture"] = FA.capture_bps(side_sgn, f["px"], f["mid0"])
    f["markout"] = FA.markout_bps(side_sgn, f["mid0"], f["mid_h"])
    f["net"] = FA.net_bps(side_sgn, f["px"], f["mid0"], f["mid_h"])
    # keep only rows with a valid forward mid (drop near-close NaN)
    fv = f[f["net"].notna()]
    # return count + mean capture/markout/net bps
    return len(fills), fv["capture"].mean(), fv["markout"].mean(), fv["net"].mean()


# run a full (strategy, config) across all dates for one symbol; aggregate both paths
def run_symbol(name, overrides, sym, dates):
    # Path A accumulator: total EOD P&L in PKR
    total_pnl = 0.0
    # count of days whose liquidation was NOT clean (estimate, not realizable)
    unclean_days = 0
    # Path B accumulators: fill count and per-fill economics (day-mean then averaged)
    n_fills_total = 0
    # collect per-day mean bps to average at the end (equal-day-weight)
    cap_days, mk_days, net_days = [], [], []
    # per-day fill counts, to weight later if desired
    fillcounts = []
    # walk every date
    for date in dates:
        # open datasets once
        dsets = R.open_datasets(date)
        # skip missing partitions
        if dsets is None:
            continue
        # run the backtest for this day
        fills, eod_pnl, clean = run_one(name, overrides, sym, date, dsets)
        # Path A: accumulate EOD P&L
        total_pnl += eod_pnl
        # tally an unclean liquidation
        if clean is False:
            unclean_days += 1
        # Path B: score the fills through attribution economics
        nf, cap, mk, net = score_bps(fills, sym, date)
        # accumulate fill count
        n_fills_total += nf
        # record the day's fill count
        fillcounts.append(nf)
        # record per-day means when they exist
        if nf > 0 and not np.isnan(net):
            cap_days.append(cap); mk_days.append(mk); net_days.append(net)
    # fill-weighted mean net bps across the period (weight each day by its fills)
    # (fill-weighting matches how total economics accrue, vs equal-day-weight)
    if n_fills_total > 0 and net_days:
        # rebuild fill-weighted means using the per-day counts aligned to recorded days
        # (simple approach: equal-day-weight mean of the day-means, robust to sparse days)
        cap_m = float(np.mean(cap_days))
        mk_m = float(np.mean(mk_days))
        net_m = float(np.mean(net_days))
    else:
        cap_m = mk_m = net_m = np.nan
    # return both paths for this (strategy, config, symbol)
    return {
        "n_fills": n_fills_total,
        "total_pnl_pkr": total_pnl,     # Path A
        "mean_capture_bps": cap_m,      # Path B
        "mean_markout_bps": mk_m,       # Path B
        "mean_net_bps": net_m,          # Path B
        "unclean_liq_days": unclean_days,
    }


# main: run naive + the 3 micro configs on both symbols, tabulate, reconcile
def main():
    # all trading dates
    dates = R.discover_dates()
    # results accumulator
    rows = []
    # wall clock
    t0 = time.perf_counter()
    # the full set of (strategy, overrides, label) to run
    runs = [("naive", {}, "naive")]
    # add each micro config with a readable label
    for cfg in MICRO_CONFIGS:
        runs.append(("micro", cfg, f"micro me={cfg['min_edge_pct']:.4f}"))
    # total run count for progress (runs x symbols)
    total = len(runs) * len(SYMBOLS)
    # counter
    done = 0
    # loop each symbol
    for sym in SYMBOLS:
        # loop each (strategy, config)
        for name, overrides, label in runs:
            # run the full period for this cell
            res = run_symbol(name, overrides, sym, dates)
            # tag identity
            res["symbol"] = sym
            res["strategy"] = label
            # collect
            rows.append(res)
            # progress
            done += 1
            print(f"  [{done}/{total}] {sym} {label} done "
                  f"(elapsed {int((time.perf_counter()-t0)//60)}m)", flush=True)
    # assemble
    df = pd.DataFrame(rows)[
        ["symbol", "strategy", "n_fills", "total_pnl_pkr",
         "mean_capture_bps", "mean_markout_bps", "mean_net_bps", "unclean_liq_days"]]
    # save
    out = Path("/Users/shazzak/Capital Stake - Results/confirm_micro_vs_naive.csv")
    df.to_csv(out, index=False)

    # ---- per-symbol report with the benchmark comparison + reconciliation ----
    for sym in SYMBOLS:
        # this symbol's rows
        d = df[df.symbol == sym].copy()
        # the naive row (the benchmark, freshly re-derived)
        nv = d[d.strategy == "naive"].iloc[0]
        # header
        print(f"\n=== {sym} ===")
        # the table
        print(d.to_string(index=False))
        # drift check: freshly-computed naive vs the stored benchmark
        bench = 2.78 if sym == "PPL" else -0.00
        print(f"  naive fresh net_bps={nv['mean_net_bps']:.3f} vs stored benchmark "
              f"{bench:+.2f}  ({'OK' if abs(nv['mean_net_bps']-bench) < 0.3 else 'DRIFT?'})")
        # for each micro config: does it beat naive on BOTH paths?
        for _, r in d[d.strategy != "naive"].iterrows():
            # Path A comparison (total PKR)
            a_beat = r["total_pnl_pkr"] > nv["total_pnl_pkr"]
            # Path B comparison (net bps/fill)
            b_beat = r["mean_net_bps"] > nv["mean_net_bps"]
            # reconciliation: do the two paths AGREE on beat/not-beat?
            agree = "AGREE" if a_beat == b_beat else "CONFLICT (inventory carry vs per-fill)"
            # report
            print(f"  {r['strategy']}: PathA PKL={r['total_pnl_pkr']:>12,.0f} "
                  f"({'beats' if a_beat else 'loses to'} naive) | "
                  f"PathB net={r['mean_net_bps']:+.3f} "
                  f"({'beats' if b_beat else 'loses to'} naive) | {agree}")
    # where it saved
    print(f"\nsaved: {out}")


# entry point
if __name__ == "__main__":
    main()
