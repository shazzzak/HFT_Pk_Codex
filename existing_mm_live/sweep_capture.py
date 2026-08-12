# ============================================================================
# sweep_capture.py -- 2D capture calibration for micro on REAL queue fills,
# full 207 days, per symbol. Sweeps min_edge_pct x improve_ticks.
#
# PERFORMANCE FIX vs the first version: the event stream does NOT depend on the
# swept parameters (they only change the STRATEGY, not the market events). The
# old loop rebuilt parse+events once PER CELL per symbol-day = 15x redundant
# reloading. This version builds each symbol-day's events ONCE, then loops all
# parameter cells INSIDE, reusing the stream. Expected ~10x+ speedup.
#
# Loop order: for date -> for symbol -> BUILD ONCE -> for (edge,improve) -> run.
#
# Run FRESH from existing_mm_live/:  python sweep_capture.py
# ============================================================================

# filesystem paths
from pathlib import Path
# timers
import time
# arrays
import numpy as np
# frames
import pandas as pd

# engine + latency
from mm_backtest import Backtester, LatencyModel
# micro strategy
from micro_mm import MicrostructureMM
# driver: loader, events, CFG, MICRO_PARAMS, dates
import run_legacy_mm as R
# reuse persist_fills' tested context join
import persist_fills as PF
# reuse attribution fee + economics
import fill_attribution as FA

# raw store (moved location)
PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# override loader root in-process
R.PARSED_ROOT = PARSED_ROOT
# feature store (context join source)
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")

# pilot symbols
SYMBOLS = ["PPL", "UBL"]

# ---- THE GRID (edit to thin/widen) ----
# edge-floor candidates (fractions): higher = wider rest = more capture, fewer fills
EDGE_GRID = [0.0000, 0.0003, 0.0005, 0.0007, 0.0010]
# improve-ticks candidates: 0 = join touch (max capture), 1 = one tick inside (less)
IMPROVE_GRID = [0.0, 0.5, 1.0]

# round-trip + per-side fee in bps (the hurdle capture must clear)
FEE_RT_BPS = FA.fee_bps_roundtrip()
FEE_SIDE_BPS = FA.fee_bps_per_side()
# nominal size per fill for the P&L proxy
SIZE = R.MICRO_PARAMS["size"]


# score a fills DataFrame in place: capture/markout/net bps. Returns valid subset.
def score_fills(fills, fs_day):
    # join fill-time context + forward mid (tested join from persist_fills)
    fills = PF.join_fill_context(fills, fs_day)
    # engine side string -> +/-1
    side_sgn = np.where(fills["side"] == "BUY", 1.0, -1.0)
    # per-fill economics in bps
    fills["capture"] = FA.capture_bps(side_sgn, fills["px"], fills["mid0"])
    fills["markout"] = FA.markout_bps(side_sgn, fills["mid0"], fills["mid_h"])
    fills["net_bps"] = FA.net_bps(side_sgn, fills["px"], fills["mid0"], fills["mid_h"])
    # keep price for the P&L proxy
    fills["price"] = fills["px"]
    # drop near-close rows with no forward mid
    return fills[fills["net_bps"].notna()]


# compact mm:ss
def _fmt(sec):
    return f"{int(sec//60)}m{int(sec%60):02d}s"


# main: build events ONCE per symbol-day, loop all cells inside
def main():
    # all trading dates
    dates = R.discover_dates()
    # count for ETA
    n_dates = len(dates)
    # per-(cell,symbol,date) records
    records = []
    # all parameter cells
    cells = [(e, i) for e in EDGE_GRID for i in IMPROVE_GRID]
    # scope announcement
    print(f"GRID: {len(EDGE_GRID)} edge x {len(IMPROVE_GRID)} improve = {len(cells)} cells")
    print(f"NEW loop order: build events ONCE/symbol-day, run {len(cells)} cells on each")
    print(f"fee hurdle: round-trip={FEE_RT_BPS:.3f} bps\n")
    # whole-sweep timer
    t0_all = time.perf_counter()
    # count symbol-days processed (for progress + ETA)
    sd_done = 0
    # total symbol-days to process (for ETA; approximate, some may be skipped)
    sd_total = n_dates * len(SYMBOLS)
    # OUTER loop: dates (build events once here)
    for date in dates:
        # open datasets once per date
        dsets = R.open_datasets(date)
        # skip missing
        if dsets is None:
            continue
        # MIDDLE loop: symbols
        for sym in SYMBOLS:
            # feature store partition (context source)
            fs_path = FS_ROOT / sym / f"date={date}.parquet"
            # need the store to score
            if not fs_path.exists():
                continue
            # ---- BUILD ONCE: parse + events, reused across ALL cells ----
            # time the one-time build
            tb = time.perf_counter()
            # load the three tables
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # need book + trades
            if len(t) == 0 or len(s) == 0:
                continue
            # build the event stream ONCE (parameter-independent)
            events, snap_groups, t = R.build_events(u, s, t)
            # continuous session window
            cont = t[t["initiator"] != "AUCTION"]
            if len(cont) == 0:
                continue
            t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
            # load context columns once (reused across cells)
            fs_day = pd.read_parquet(
                fs_path,
                columns=["ts_exch", "mid", "spread_bps", "obi_1",
                         "toxicity", "realized_vol_bps"])
            # one-time build cost for this symbol-day
            build_s = time.perf_counter() - tb
            # ---- INNER loop: all parameter cells reuse the SAME events ----
            # time the cell sweep for this symbol-day
            tc = time.perf_counter()
            # loop every (edge, improve) cell
            for (min_edge, improve) in cells:
                # cfg with seeded latency; log_equity=False skips the per-event OBI
                # scans that dominate runtime -- fills are identical, ~50x faster.
                cfg = dict(R.CFG, session=(t0, t1),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED),
                           log_equity=False)

                # micro params for this cell
                params = dict(R.MICRO_PARAMS)
                params["min_edge_pct"] = min_edge
                params["improve_ticks"] = improve
                # build micro at this cell
                strat = MicrostructureMM(session_ms=(t0, t1), **params)
                # run the backtest on the REUSED event stream
                bt = Backtester(strat, cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                # skip empty
                if fills is None or len(fills) == 0:
                    continue
                # score (context join + economics)
                fv = score_fills(fills, fs_day)
                # nothing valid
                if len(fv) == 0:
                    continue
                # record aggregates for this (cell, symbol, date)
                records.append({
                    "min_edge_pct": min_edge, "improve_ticks": improve,
                    "symbol": sym, "date": date,
                    "n_fills": int(len(fv)),
                    "mean_capture": float(fv["capture"].mean()),
                    "mean_markout": float(fv["markout"].mean()),
                    "mean_net_bps": float(fv["net_bps"].mean()),
                    "pnl_proxy": float((fv["net_bps"] / 1e4 * fv["price"] * SIZE).sum()),
                })
            # cell-sweep cost for this symbol-day
            cells_s = time.perf_counter() - tc
            # tick symbol-day counter
            sd_done += 1
            # progress + ETA every 25 symbol-days
            if sd_done % 25 == 0:
                # elapsed overall
                el = time.perf_counter() - t0_all
                # projected total from rate so far
                proj = el / sd_done * sd_total
                # heartbeat: last symbol-day's build vs cell-sweep split + ETA
                print(f"  {sd_done}/{sd_total} symbol-days  "
                      f"(last: build {build_s:.1f}s, {len(cells)} cells {cells_s:.1f}s)  "
                      f"elapsed {_fmt(el)}  ETA {_fmt(proj - el)}", flush=True)
    # assemble
    df = pd.DataFrame(records)
    # save raw
    out = Path("/Users/shazzak/Capital Stake - Results/capture_sweep.csv")
    df.to_csv(out, index=False)
    # summary per (symbol, edge, improve)
    summary = (df.groupby(["symbol", "min_edge_pct", "improve_ticks"])
                 .agg(total_fills=("n_fills", "sum"),
                      total_pnl_proxy=("pnl_proxy", "sum"),
                      mean_capture=("mean_capture", "mean"),
                      mean_markout=("mean_markout", "mean"),
                      mean_net_bps=("mean_net_bps", "mean"))
                 .reset_index())
    # fee-hurdle flag
    summary["capture_ge_fee"] = summary["mean_capture"] >= FEE_RT_BPS
    # save summary
    summary.to_csv(Path("/Users/shazzak/Capital Stake - Results/capture_sweep_summary.csv"), index=False)
    # per-symbol report
    for sym in SYMBOLS:
        # this symbol's grid
        ss = summary[summary.symbol == sym].sort_values(["min_edge_pct", "improve_ticks"])
        # full grid
        print(f"\n=== {sym}: full grid (objective=total_pnl_proxy) ===")
        print(ss.to_string(index=False))
        # P&L argmax
        best = ss.loc[ss["total_pnl_proxy"].idxmax()]
        print(f"  -> MAX total P&L: min_edge={best['min_edge_pct']:.4f} "
              f"improve_ticks={best['improve_ticks']} "
              f"(pnl_proxy={best['total_pnl_proxy']:,.0f}, fills={int(best['total_fills']):,}, "
              f"capture={best['mean_capture']:.2f} bps, net={best['mean_net_bps']:.3f} bps)")
        # fee frontier
        fe = ss[ss["capture_ge_fee"]]
        if len(fe):
            pt = fe.loc[fe["mean_capture"].idxmin()]
            print(f"  -> fee frontier (capture>={FEE_RT_BPS:.2f} bps): min_edge={pt['min_edge_pct']:.4f} "
                  f"improve_ticks={pt['improve_ticks']} (capture={pt['mean_capture']:.2f} bps)")
        else:
            print(f"  -> fee frontier: NO cell reaches capture>={FEE_RT_BPS:.2f} bps -- widen grid")
    print(f"\nsaved: {out} and capture_sweep_summary.csv")


# entry point
if __name__ == "__main__":
    main()
