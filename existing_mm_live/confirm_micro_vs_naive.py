# confirm_micro_vs_naive.py -- validation gate: does capture-recalibrated micro
# beat naive, per symbol, on 207 days? Measures P&L two ways and reconciles:
#   Path A -- Backtester true EOD P&L in PKR (bt.eod["equity_liquidated"]).
#   Path B -- attribution net bps/fill (the +2.78/-0.00 benchmark lens).
# Micro tested at 0.0003/0.0005/0.0007 (improve_ticks=0.0); naive re-derived fresh.
#
# PERFORMANCE: events (esp. the 9s snapshot pre-parse) are built ONCE per
# symbol-day and reused across all 4 configs -- the old per-config structure
# rebuilt them 4x per symbol-day (~37s wasted/day). Also caches the feature-store
# day once per symbol-day for Path B.
#
# Run from existing_mm_live/:  python confirm_micro_vs_naive.py

# paths
from pathlib import Path
# timing
import time
# frames + arrays
import pandas as pd
import numpy as np
# driver + engine + strategies
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel, NaiveSymmetricMM
from micro_mm import MicrostructureMM
# attribution economics (Path B)
import fill_attribution as FA
# context join (Path B)
import persist_fills as PF

# raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# feature store (Path B context)
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")

# pilot symbols
SYMBOLS = ["PPL", "UBL"]
# the run set: naive + 3 micro configs. Each entry: (name, overrides, label).
RUNSET = [
    ("naive", {}, "naive"),
    ("micro", {"min_edge_pct": 0.0003, "improve_ticks": 0.0}, "micro me=0.0003"),
    ("micro", {"min_edge_pct": 0.0005, "improve_ticks": 0.0}, "micro me=0.0005"),
    ("micro", {"min_edge_pct": 0.0007, "improve_ticks": 0.0}, "micro me=0.0007"),
]


# build a strategy for a given name + overrides (session_ms only used by micro)
def make_strategy(name, session_ms, overrides):
    # naive ignores overrides + session_ms
    if name == "naive":
        return NaiveSymmetricMM(**R.STRAT)
    # micro: MICRO_PARAMS with the config overrides applied
    params = dict(R.MICRO_PARAMS)
    params.update(overrides)
    return MicrostructureMM(session_ms=session_ms, **params)


# score a run's fills through attribution economics (Path B); returns per-fill means.
# fs_day is passed IN (cached once per symbol-day, not re-read per config).
def score_bps(fills, fs_day):
    # no fills -> nothing
    if fills is None or len(fills) == 0:
        return 0, np.nan, np.nan, np.nan
    # attach context + forward mid (tested no-leak join)
    f = PF.join_fill_context(fills, fs_day)
    # engine side string -> +/-1
    side_sgn = np.where(f["side"] == "BUY", 1.0, -1.0)
    # per-fill economics
    f["capture"] = FA.capture_bps(side_sgn, f["px"], f["mid0"])
    f["markout"] = FA.markout_bps(side_sgn, f["mid0"], f["mid_h"])
    f["net"] = FA.net_bps(side_sgn, f["px"], f["mid0"], f["mid_h"])
    # drop near-close NaN rows
    fv = f[f["net"].notna()]
    # count + mean capture/markout/net
    return len(fills), fv["capture"].mean(), fv["markout"].mean(), fv["net"].mean()


# compact mm:ss
def _fmt(sec):
    return f"{int(sec//60)}m{int(sec % 60):02d}s"


# main: build events ONCE per symbol-day, run all configs on the reused stream
def main():
    # all dates
    dates = R.discover_dates()
    # per-(symbol, config) accumulators, keyed by (sym, label)
    acc = {}
    # init accumulators for every (symbol, config)
    for sym in SYMBOLS:
        for _, _, label in RUNSET:
            # totals: Path A PKR, fills, per-day Path B means, unclean-liq days
            acc[(sym, label)] = {"pnl": 0.0, "fills": 0, "unclean": 0,
                                 "cap": [], "mk": [], "net": []}
    # whole-run timer
    t0_all = time.perf_counter()
    # symbol-day counter for the heartbeat
    sd = 0
    # total symbol-days (for ETA)
    sd_total = len(dates) * len(SYMBOLS)
    # announce
    print(f"confirm: {len(RUNSET)} configs x {len(SYMBOLS)} symbols x {len(dates)} days; "
          f"build-once per symbol-day\n", flush=True)
    # OUTER: dates
    for date in dates:
        # open datasets once
        dsets = R.open_datasets(date)
        # skip missing
        if dsets is None:
            continue
        # MIDDLE: symbols
        for sym in SYMBOLS:
            # feature-store day for Path B (read ONCE, reused across configs)
            fs_path = FS_ROOT / sym / f"date={date}.parquet"
            # need it for Path B; skip symbol-day if absent
            if not fs_path.exists():
                continue
            # load context columns once
            fs_day = pd.read_parquet(
                fs_path,
                columns=["ts_exch", "mid", "spread_bps", "obi_1",
                         "toxicity", "realized_vol_bps"])
            # ---- BUILD EVENTS ONCE (the 9s snapshot pre-parse, paid once) ----
            # load the three tables
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # need book + trades
            if len(t) == 0 or len(s) == 0:
                continue
            # build the event stream once
            events, snap_groups, t = R.build_events(u, s, t)
            # continuous-session window
            cont = t[t["initiator"] != "AUCTION"]
            if len(cont) == 0:
                continue
            t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
            # ---- INNER: every config reuses the SAME events + fs_day ----
            for name, overrides, label in RUNSET:
                # fresh seeded latency per run (reproducible, order-independent)
                cfg = dict(R.CFG, session=(t0, t1),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                # build + run
                bt = Backtester(make_strategy(name, (t0, t1), overrides), cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                # Path A: true EOD P&L
                if bt.eod is not None:
                    acc[(sym, label)]["pnl"] += float(bt.eod["equity_liquidated"])
                    if bt.eod["liquidation_clean"] is False:
                        acc[(sym, label)]["unclean"] += 1
                # Path B: score fills (reusing cached fs_day)
                nf, cap, mk, net = score_bps(fills, fs_day)
                acc[(sym, label)]["fills"] += nf
                # record per-day means when valid
                if nf > 0 and not np.isnan(net):
                    acc[(sym, label)]["cap"].append(cap)
                    acc[(sym, label)]["mk"].append(mk)
                    acc[(sym, label)]["net"].append(net)
            # tick symbol-day + heartbeat every 25
            sd += 1
            if sd % 25 == 0:
                el = time.perf_counter() - t0_all
                proj = el / sd * sd_total
                print(f"  {sd}/{sd_total} symbol-days  elapsed {_fmt(el)}  "
                      f"ETA {_fmt(proj - el)}", flush=True)
    # ---- assemble results ----
    rows = []
    for (sym, label), a in acc.items():
        rows.append({
            "symbol": sym, "strategy": label,
            "n_fills": a["fills"],
            "total_pnl_pkr": a["pnl"],
            "mean_capture_bps": float(np.mean(a["cap"])) if a["cap"] else np.nan,
            "mean_markout_bps": float(np.mean(a["mk"])) if a["mk"] else np.nan,
            "mean_net_bps": float(np.mean(a["net"])) if a["net"] else np.nan,
            "unclean_liq_days": a["unclean"],
        })
    # frame + save
    df = pd.DataFrame(rows)
    out = Path("/Users/shazzak/Capital Stake - Results/confirm_micro_vs_naive.csv")
    df.to_csv(out, index=False)
    # ---- per-symbol report with benchmark + reconciliation ----
    for sym in SYMBOLS:
        d = df[df.symbol == sym].copy()
        # order: naive first, then micro configs
        d["_ord"] = d["strategy"].map({"naive": 0, "micro me=0.0003": 1,
                                       "micro me=0.0005": 2, "micro me=0.0007": 3})
        d = d.sort_values("_ord").drop(columns="_ord")
        # naive benchmark row
        nv = d[d.strategy == "naive"].iloc[0]
        print(f"\n=== {sym} ===")
        print(d.to_string(index=False))
        # drift check vs stored benchmark
        bench = 2.78 if sym == "PPL" else -0.00
        print(f"  naive fresh net_bps={nv['mean_net_bps']:.3f} vs stored {bench:+.2f}  "
              f"({'OK' if abs(nv['mean_net_bps'] - bench) < 0.4 else 'DRIFT?'})")
        # per micro config: beats naive on both paths? do they agree?
        for _, r in d[d.strategy != "naive"].iterrows():
            a_beat = r["total_pnl_pkr"] > nv["total_pnl_pkr"]
            b_beat = r["mean_net_bps"] > nv["mean_net_bps"]
            agree = "AGREE" if a_beat == b_beat else "CONFLICT (carry vs per-fill)"
            print(f"  {r['strategy']}: PathA={r['total_pnl_pkr']:>12,.0f} "
                  f"({'beats' if a_beat else 'loses'}) | "
                  f"PathB={r['mean_net_bps']:+.3f} "
                  f"({'beats' if b_beat else 'loses'}) | {agree}")
    print(f"\nsaved: {out}")


# entry point
if __name__ == "__main__":
    main()
