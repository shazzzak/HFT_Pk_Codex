# confirm_micro_vs_naive.py -- validation gate: does capture-recalibrated micro
# beat naive, per symbol, on 207 days? Measures P&L two ways and reconciles:
#   Path A -- Backtester true EOD P&L in PKR (bt.eod["equity_liquidated"]).
#   Path B -- attribution net bps/fill (the +2.78/-0.00 benchmark lens).
# Micro now sweeps session_scale per symbol (the inventory-skew fix); naive fresh.
#
# PERFORMANCE: events (esp. the 9s snapshot pre-parse) are built ONCE per
# symbol-day and reused across all configs -- the old per-config structure
# rebuilt them per config per symbol-day. Also caches the feature-store day once
# per symbol-day for Path B.
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

# Per-symbol session_scale (still injected for micro; skew proven inert but the
# arg is required). Held at the 1x point -- session_scale is NO LONGER the lever.
SESSION_SCALE_BASE = {"PPL": 7.6, "UBL": 3.9}
# min_edge held at the diagnosed region for this experiment.
_ME = 0.0005
# 2x2 experiment + naive, to isolate the two confirmed drivers of the short drift:
#   base       = microprice ON,  no band   -> reproduces the current losing config
#   MID        = microprice OFF (plain mid), no band   -> isolates the microprice lean
#   band150    = microprice ON,  soft band 150   -> isolates the inventory brake
#   MID+band150= microprice OFF, soft band 150   -> both together
# Read EOD position per config: does MID and/or the band pull mean_pos toward 0?
# Run set as 4-tuples: (name, overrides, ss_mult, label). naive carries no scale.
RUNSET = [
    ("naive", {}, None, "naive"),
    ("micro", {"min_edge_pct": _ME, "improve_ticks": 0.0,
               "use_microprice": True}, 1.0, "micro base"),
    ("micro", {"min_edge_pct": _ME, "improve_ticks": 0.0,
               "use_microprice": False}, 1.0, "micro MID"),
    ("micro", {"min_edge_pct": _ME, "improve_ticks": 0.0,
               "use_microprice": True, "soft_inv": 150}, 1.0, "micro band150"),
    ("micro", {"min_edge_pct": _ME, "improve_ticks": 0.0,
               "use_microprice": False, "soft_inv": 150}, 1.0, "micro MID+band150"),
]


# build a strategy for name + overrides. sym + ss_mult added so micro gets its
# per-symbol, swept session_scale injected (naive ignores all of sym/ss_mult).
def make_strategy(name, sym, session_ms, overrides, ss_mult):
    # naive ignores overrides, session_ms, sym, ss_mult.
    if name == "naive":
        return NaiveSymmetricMM(**R.STRAT)
    # micro: start from MICRO_PARAMS, apply the config overrides.
    params = dict(R.MICRO_PARAMS)
    params.update(overrides)
    # Inject the required keyword-only session_scale: per-symbol base x multiplier.
    # [] not .get() -> an uncalibrated symbol raises KeyError, never runs on a guess.
    params["session_scale"] = SESSION_SCALE_BASE[sym] * ss_mult
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
    # per-day EOD position rows for plot_skew_validation.py (one row per run/day).
    _eod_rows = []
    # init accumulators for every (symbol, config)
    for sym in SYMBOLS:
        # 4-tuple now; only the label is needed here.
        for _, _, _, label in RUNSET:
            # totals: Path A PKR, summed mid-mark PKR, fills, per-day Path B means,
            # unclean-liq days, day-counts, and liq_none_days so a None in the
            # headline liquidated P&L is surfaced, never silently dropped.
            # liqslip = per-day liquidation slippage per share (direct unwind cost,
            # independent of the None-mid problem); unfilled = shares left unfilled.
            acc[(sym, label)] = {"pnl": 0.0, "mid": 0.0, "fills": 0, "unclean": 0,
                                 "pnl_days": 0, "mid_days": 0, "liq_none_days": 0,
                                 "liqslip": [], "unfilled": 0.0,
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
            # RUNSET entries are now 4-tuples: (name, overrides, ss_mult, label).
            for name, overrides, ss_mult, label in RUNSET:
                # fresh seeded latency per run (reproducible, order-independent)
                cfg = dict(R.CFG, session=(t0, t1),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                # build + run
                # pass sym + ss_mult so make_strategy injects the per-symbol,
                # swept session_scale (7.6 PPL / 3.9 UBL x the multiplier).
                bt = Backtester(
                    make_strategy(name, sym, (t0, t1), overrides, ss_mult), cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                # Path A: true EOD P&L
                if bt.eod is not None:
                    # Path A total: true post-liquidation EOD P&L (summed over days).
                    # equity_liquidated should never be None, but if it is we skip
                    # the day and COUNT it (liq_none_days) rather than crash the whole
                    # run or -- worse -- silently drop it and corrupt the headline
                    # basis. The printed count keeps the total's day-basis honest
                    # when comparing to the naive benchmark (measured over 207 days).
                    _liq = bt.eod["equity_liquidated"]
                    if _liq is not None:
                        acc[(sym, label)]["pnl"] += float(_liq)
                        acc[(sym, label)]["pnl_days"] += 1
                    else:
                        acc[(sym, label)]["liq_none_days"] += 1
                    # Mid-mark P&L can be None on a day with no valid closing mid.
                    # Accumulate only when present (it is a diagnostic, not the
                    # headline). Days skipped here are counted below for honesty.
                    _mid = bt.eod["equity_mid_mark"]
                    if _mid is not None:
                        acc[(sym, label)]["mid"] += float(_mid)
                        acc[(sym, label)]["mid_days"] += 1
                    # Per-day EOD signed position for the histogram; skip if absent.
                    _pos = bt.eod["pos_at_close"]
                    if _pos is not None:
                        _eod_rows.append({"symbol": sym, "variant": label,
                                          "eod_pos": float(_pos)})
                    # count clean vs unclean liquidation days.
                    if bt.eod["liquidation_clean"] is False:
                        acc[(sym, label)]["unclean"] += 1
                    # direct unwind cost: liquidation slippage per share (PKR/sh),
                    # accumulated when present -> the real "how bad is the close" metric.
                    _ls = bt.eod.get("liq_slippage_per_sh")
                    if _ls is not None:
                        acc[(sym, label)]["liqslip"].append(float(_ls))
                    # shares the liquidation couldn't fill (residual / thin book).
                    _uf = bt.eod.get("unfilled_sh")
                    if _uf is not None:
                        acc[(sym, label)]["unfilled"] += float(_uf)
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
            # direct unwind cost + residual: how expensive/incomplete the close was.
            "mean_liq_slip_per_sh": float(np.mean(a["liqslip"])) if a["liqslip"] else np.nan,
            "total_unfilled_sh": a["unfilled"],
            "unclean_liq_days": a["unclean"],
        })
    # frame + save
    df = pd.DataFrame(rows)
    out = Path("/Users/shazzak/Capital Stake - Results/confirm_micro_vs_naive.csv")
    df.to_csv(out, index=False)
    # ---- LOUD: any day where the headline liquidated P&L was None ----------
    # These days are excluded from total_pnl_pkr, so if the count is nonzero the
    # total is NOT on the same 207-day basis as the naive benchmark. Surfaced
    # here (not buried) precisely because it distorts the headline comparison.
    _liq_bad = [(sym, lbl, a["liq_none_days"])
                for (sym, lbl), a in acc.items() if a["liq_none_days"] > 0]
    if _liq_bad:
        print("\n!! LIQUIDATED P&L WAS None ON SOME DAYS (excluded from totals):")
        for sym, lbl, n in _liq_bad:
            print(f"     {sym} {lbl}: {n} day(s) -> total is over fewer than all days")
    else:
        print("\nliquidated P&L present on every day (headline basis intact).")
    # ---- per-symbol report with benchmark + reconciliation ----
    for sym in SYMBOLS:
        d = df[df.symbol == sym].copy()
        # order: naive first, then micro configs
        # rank by position in RUNSET (robust to any labels).
        _order = {lbl_: i for i, (_, _, _, lbl_) in enumerate(RUNSET)}
        # naive is RUNSET index 0, so it still sorts first.
        d["_ord"] = d["strategy"].map(_order)
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
    # ---- validation CSVs for plot_skew_validation.py -----------------------
    # per-day EOD signed positions: one row per (symbol, variant, day).
    eod_df = pd.DataFrame(_eod_rows)
    # per-(symbol, variant) P&L: summed mid-mark vs summed liquidated over all days.
    # The gap between the two IS the inventory carry cost the skew fix targets.
    pnl_df = pd.DataFrame([
        {"symbol": s_, "variant": lbl_,
         "midmark_pnl": a_["mid"], "liquidated_pnl": a_["pnl"],
         "mid_days": a_["mid_days"], "pnl_days": a_["pnl_days"]}
        for (s_, lbl_), a_ in acc.items()
    ])
    # honesty check: mid-mark is summed over mid_days, liquidated over pnl_days.
    # If they differ the carry-gap bar compares slightly different day-sets.
    _mism = pnl_df[pnl_df["mid_days"] != pnl_df["pnl_days"]]
    if len(_mism):
        print("  NOTE: mid/liquidated day-set mismatch (mid None on some days):")
        print(_mism[["symbol", "variant", "mid_days", "pnl_days"]].to_string(index=False))
    # write both next to the confirm CSV, where the plotter looks by default.
    res = Path("/Users/shazzak/Capital Stake - Results")
    # per-day positions -> the drift-toward-flat histogram.
    eod_df.to_csv(res / "eod_positions.csv", index=False)
    # P&L summary -> the mid-mark vs liquidated carry-gap bars.
    pnl_df.to_csv(res / "pnl_summary.csv", index=False)
    # confirm on stdout.
    print(f"wrote {res / 'eod_positions.csv'} and {res / 'pnl_summary.csv'}")


# entry point
if __name__ == "__main__":
    main()
