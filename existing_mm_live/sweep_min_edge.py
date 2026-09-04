# ============================================================================
# sweep_min_edge.py -- find the net-P&L-maximizing min_edge_pct per symbol,
# on REAL queue fills across all 207 days (not single-day, not per-fill-bps).
# ============================================================================
# For each candidate min_edge_pct: rebuild micro's real queue fills via the
# engine, attribute them per symbol, and record TOTAL net P&L (not just
# net-bps-per-fill -- widening raises per-fill capture but cuts fill count,
# so the objective must be total, which has an interior optimum).
#
# Reuses persist_fills.py's machinery (same Backtester, seeded latency, fees,
# feature-context join) so every sweep point is queue-realistic and reconciles.
#
# Run FRESH from existing_mm_live/:  python sweep_min_edge.py
# WARNING: each grid point re-runs the backtest for both pilot symbols x 207
# days (~minutes per point). A 6-point grid is ~tens of minutes. Overnight-safe.
# ============================================================================

# filesystem paths
from pathlib import Path
# timing per grid point
import time
# arrays
import numpy as np
# frames + the fee/economics helpers
import pandas as pd

# the engine + strategies
from mm_backtest import Backtester, LatencyModel
# micro strategy
from micro_mm import MicrostructureMM
# driver: loader, events, CFG, MICRO_PARAMS, dates
import run_legacy_mm as R
# reuse persist_fills' context-join + fee math so economics match attribution exactly
import persist_fills as PF
# reuse the attribution fee constant (round-trip bps) for net
import fill_attribution as FA

# raw store (moved location)
PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# override loader root in-process
R.PARSED_ROOT = PARSED_ROOT
# feature store (for the fill-time context join)
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")

# pilot symbols (the two with feature stores + validated fills)
SYMBOLS = ["PPL", "UBL"]
# the grid of edge floors to test, in fractions (0 = current untuned; 0.0010 = 10bps)
# spans the range the single-day sweep explored, now judged on FULL-PERIOD TOTAL P&L
EDGE_GRID = [0.0000, 0.0002, 0.0003, 0.0005, 0.0007, 0.0010]
# round-trip fee in bps (from the attribution module -- single source of truth)
FEE_RT_BPS = FA.fee_bps_roundtrip()
# nominal shares per fill (size) -- for turning bps into a comparable P&L proxy
# NOTE: uses MICRO_PARAMS['size']; total "pnl proxy" = sum(net_bps/1e4 * price * size)
SIZE = R.MICRO_PARAMS["size"]


# run ONE symbol-day at a given min_edge_pct; return the fill frame with net_bps
def run_symbol_day(date, sym, dsets, fs_day, min_edge):
    # load the three tables (same loader as run_one / persist_fills)
    u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
    t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
    # not runnable without book + trades
    if len(t) == 0 or len(s) == 0:
        return None
    # build the event stream
    events, snap_groups, t = R.build_events(u, s, t)
    # continuous session window
    cont = t[t["initiator"] != "AUCTION"]
    if len(cont) == 0:
        return None
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
    # cfg with seeded latency (same as persist_fills for reconciliation)
    cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED))
    # micro params with THIS grid point's min_edge_pct overriding the default
    params = dict(R.MICRO_PARAMS)
    # override just the edge floor being swept
    params["min_edge_pct"] = min_edge
    # build micro at this edge floor
    strat = MicrostructureMM(session_ms=(t0, t1), **params)
    # run the real backtest (real queue fills)
    bt = Backtester(strat, cfg)
    fills, equity, stats = bt.run(events, snap_groups)
    # no fills -> nothing
    if fills is None or len(fills) == 0:
        return None
    # join fill-time context + forward mid (reuse persist_fills' tested join)
    fills = PF.join_fill_context(fills, fs_day)
    # map engine side string to +/-1 (BUY=+1 our bid filled, SELL=-1)
    side_sgn = np.where(fills["side"] == "BUY", 1.0, -1.0)
    # per-fill net bps: gross (fill price -> forward mid) minus round-trip fee
    fills["net_bps"] = FA.net_bps(side_sgn, fills["px"], fills["mid0"], fills["mid_h"])
    # keep price for the P&L-proxy weighting
    fills["price"] = fills["px"]
    # return the scored fills
    return fills


# sweep all grid points across all dates, aggregate per (symbol, min_edge)
def main():
    # all trading dates
    dates = R.discover_dates()
    # accumulator: one record per (min_edge, symbol, date) with totals
    records = []
    # loop each edge-floor candidate
    for min_edge in EDGE_GRID:
        # announce the grid point
        print(f"\n=== min_edge_pct = {min_edge:.4f} ===")
        # time it
        g0 = time.perf_counter()
        # loop dates
        for date in dates:
            # open datasets once per date
            dsets = R.open_datasets(date)
            # skip missing partitions
            if dsets is None:
                continue
            # each symbol
            for sym in SYMBOLS:
                # the feature store partition for the context join
                fs_path = FS_ROOT / sym / f"date={date}.parquet"
                # need the store to score fills; skip if absent
                if not fs_path.exists():
                    continue
                # load the day's feature rows (context source)
                fs_day = pd.read_parquet(
                    fs_path,
                    columns=["ts_exch", "mid", "spread_bps", "obi_1",
                             "toxicity", "realized_vol_bps"])
                # run + score this symbol-day at this edge floor
                try:
                    f = run_symbol_day(date, sym, dsets, fs_day, min_edge)
                except Exception as e:
                    # note the failure, continue
                    print(f"   {sym} {date} ERROR {e!r}")
                    continue
                # skip empty
                if f is None:
                    continue
                # drop NaN-net rows (near-close fills with no forward mid)
                valid = f["net_bps"].notna()
                # scored subset
                fv = f[valid]
                # nothing valid -> skip
                if len(fv) == 0:
                    continue
                # record totals for this (min_edge, symbol, date)
                records.append({
                    "min_edge_pct": min_edge,
                    "symbol": sym,
                    "date": date,
                    # fill count (the volume side of the tradeoff)
                    "n_fills": int(len(fv)),
                    # mean net bps per fill (the price side)
                    "mean_net_bps": float(fv["net_bps"].mean()),
                    # P&L proxy: sum over fills of net_bps/1e4 * price * size
                    "pnl_proxy": float((fv["net_bps"] / 1e4 * fv["price"] * SIZE).sum()),
                })
        # grid-point timing
        print(f"   done in {time.perf_counter()-g0:.1f}s")

    # assemble
    df = pd.DataFrame(records)
    # save the raw per-day records for later inspection
    out = Path("/Users/shazzak/Capital Stake - Results/min_edge_sweep.csv")
    df.to_csv(out, index=False)

    # ---- summary: aggregate over dates, per (symbol, min_edge) ----
    # group and sum/mean the key objective columns
    summary = (df.groupby(["symbol", "min_edge_pct"])
                 .agg(total_fills=("n_fills", "sum"),
                      total_pnl_proxy=("pnl_proxy", "sum"),
                      mean_net_bps=("mean_net_bps", "mean"))
                 .reset_index())
    # print the frontier per symbol
    print("\n=== SWEEP SUMMARY (objective = total_pnl_proxy, per symbol) ===")
    # show each symbol's curve
    for sym in SYMBOLS:
        # this symbol's rows sorted by edge floor
        s = summary[summary.symbol == sym].sort_values("min_edge_pct")
        # header
        print(f"\n{sym}:")
        # the table
        print(s.to_string(index=False))
        # the argmax on total P&L proxy -- the recommended edge floor
        best = s.loc[s["total_pnl_proxy"].idxmax()]
        # call it out
        print(f"  -> max total P&L at min_edge_pct = {best['min_edge_pct']:.4f} "
              f"(total_pnl_proxy={best['total_pnl_proxy']:,.0f}, "
              f"fills={int(best['total_fills']):,}, "
              f"mean_net={best['mean_net_bps']:.3f} bps)")
    # where it saved
    print(f"\nsaved per-day records: {out}")


# entry point
if __name__ == "__main__":
    main()
