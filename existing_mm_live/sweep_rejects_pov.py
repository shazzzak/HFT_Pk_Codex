# sweep_rejects_pov.py -- THE REJECT/BORDERLINE EXPERIMENT: does the NEW unwind (SZ's POV model) beat
# the OLD fixed-window unwind, per clip size? The hypothesis: at 1x the new model
# is dormant (recovers the ~4% quoting give-back); at 3-5x, inventory is real and
# the POV unwind's early, liquidity-aware exit should WIN. The comparison is
# new-vs-old AT THE SAME CLIP, so the fill-model's large-clip blind spots cancel
# in the difference.
#
# 2 strategies x 5 clips x top-10 names x 197 days. Both strategies are the
# locked MID config (microprice off, both triggers); the ONLY difference is the
# unwind engagement rule (POV profile passed vs not).
#
# Reads: session_scales_*.csv, volume_profile_*.csv, session_segments_*.csv,
# time_windows_*.csv (newest of each). Writes ranked + daily CSVs (timestamped),
# daily rows carry the full instrumentation (fills, window fills, liq split).
#
# Run from existing_mm_live/:  python sweep_top10_pov.py

# paths
from pathlib import Path
# timing + stamp
import time
from datetime import datetime
# frames + arrays
import pandas as pd
import numpy as np
# driver + engine + strategy
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel
from micro_mm import MicrostructureMM
# reuse confirm's scorer + formatter
import confirm_micro_vs_naive as C

# raw store + results
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ experiment knobs ------------------------------
# the reject + borderline names: does the window-based POV unwind flip them?
#   HARD REJECTS (loss-shrink test; probably stay negative -- their jump-day
#   toxicity is a vol-gate problem, not an unwind problem):
#     FNEL, PACE, AKBL, TPL, HASCOL
#   BORDERLINE (graduation test; already profitable, dirty exits -- the unwind
#   may clean them into the 24-name portfolio, WIDENING the tradeable universe):
#     THCCL, TOMCL, TRG, BOP, NCPL
TOP10 = ["FNEL", "PACE", "AKBL", "TPL", "HASCOL",
         "THCCL", "TOMCL", "TRG", "BOP", "NCPL"]
# clip multiples of the trailing median trade size
MULTS = [1.0, 2.0, 3.0, 4.0, 5.0]
# inventory limits in clips (unchanged ratios)
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
# participation cap for the POV unwind
MAX_POV = 0.10
# trailing window for the median clip (walk-forward)
TRAIL_DAYS = 10
# locked MID base
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_eod_trigger=True, enable_lock_trigger=True)
# ------------------------------------------------------------------------------


# newest file matching a pattern (timestamped names sort chronologically)
def newest(pattern):
    cands = sorted(RESULTS.glob(pattern))
    if not cands:
        raise SystemExit(f"missing {pattern} -- run its calibration script first")
    return cands[-1]


# {symbol: value-dict} from a calibration CSV, ok rows only
def load_table(pattern, cols):
    df = pd.read_csv(newest(pattern))
    ok = df[df["note"] == "ok"] if "note" in df.columns else df
    return {r["symbol"]: tuple(r[c] for c in cols) for _, r in ok.iterrows()}


# per-date session segments: {date_str: [(s,e), ...]}
def load_segments():
    df = pd.read_csv(newest("session_segments_*.csv"))
    out = {}
    for _, r in df.iterrows():
        segs = [tuple(int(x) for x in p.split(":")) for p in r["segments"].split(";")]
        out[r["date"]] = segs
    return out


# trailing median trade size + median notional strictly before a date
def trailing(stats_sym, all_dates, date):
    prior = [stats_sym[str(d)] for d in all_dates
             if str(d) < str(date) and str(d) in stats_sym]
    if len(prior) < TRAIL_DAYS:
        return None, None
    win = prior[-TRAIL_DAYS:]
    return float(np.median([w[0] for w in win])), float(np.median([w[1] for w in win]))


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # calibration tables (newest of each)
    scales = {k: v[0] for k, v in load_table("session_scales_*.csv",
                                             ["session_scale"]).items()}
    profiles = load_table("volume_profile_*.csv",
                          ["vol_first15", "vol_middle", "vol_preclose45",
                           "vol_last15"])
    windows = load_table("time_windows_*.csv",
                         ["eod_ramp_start_min", "eod_cliff_min"])
    segments = load_segments()
    # every symbol must be fully calibrated (fail loud)
    for sym in TOP10:
        if sym not in scales or sym not in profiles:
            raise SystemExit(f"{sym} missing scale or volume profile -- calibrate first")

    # dates + trades pre-pass (median clip anchor)
    all_dates = R.discover_dates()
    print(f"pre-pass: trailing median trade size, {len(all_dates)} days "
          f"x {len(TOP10)} names", flush=True)
    tstats = {s: {} for s in TOP10}
    t0 = time.perf_counter()
    for i, date in enumerate(all_dates, 1):
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        for sym in TOP10:
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0:
                continue
            tstats[sym][str(date)] = (float(t["qty"].median()),
                                      float((t["price"] * t["qty"]).sum()))
        if i % 50 == 0 or i == len(all_dates):
            print(f"  pre-pass {i}/{len(all_dates)}  "
                  f"elapsed {C._fmt(time.perf_counter() - t0)}", flush=True)

    # run days (after the trailing warm-up)
    run_dates = all_dates[TRAIL_DAYS:]
    # the strategy variants: OLD = fixed-window unwind, NEW = POV unwind
    VARIANTS = ["OLD", "NEW"]
    total = len(run_dates) * len(TOP10)
    print(f"\nsweep_top10_pov: {len(VARIANTS)} strategies x {len(MULTS)} clips x "
          f"{len(TOP10)} names x {len(run_dates)} days\n", flush=True)

    # accumulators keyed (sym, variant, mult)
    acc = {}
    for sym in TOP10:
        for v in VARIANTS:
            for mlt in MULTS:
                acc[(sym, v, mlt)] = {"pnl": 0.0, "fills": 0, "days": 0,
                                      "net": [], "unclean": 0}
    rows = []
    t0_all = time.perf_counter()
    sd = 0

    # OUTER: dates
    for date in run_dates:
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # this day's segments (skip a date with no segment record)
        segs = segments.get(str(date))
        if segs is None:
            continue
        # MIDDLE: symbols
        for sym in TOP10:
            sd += 1
            med_qty, adv = trailing(tstats[sym], all_dates, date)
            if med_qty is None or med_qty <= 0:
                continue
            fs_path = FS_ROOT / sym / f"date={date}.parquet"
            if not fs_path.exists():
                continue
            fs_day = pd.read_parquet(
                fs_path, columns=["ts_exch", "mid", "spread_bps", "obi_1",
                                  "toxicity", "realized_vol_bps"])
            # build events ONCE per symbol-day (shared by all 10 runs)
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0 or len(s) == 0:
                continue
            events, snap_groups, t = R.build_events(u, s, t)
            cont = s[s["phase"] == "CONTINUOUS_AUCTION"]
            if len(cont) == 0:
                continue
            t0_, t1_ = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
            # INNER: strategy x clip on the same event stream
            for v in VARIANTS:
                for mlt in MULTS:
                    clip = max(1, int(round(mlt * med_qty)))
                    params = dict(MID_BASE)
                    params["size"] = clip
                    params["max_inv"] = int(round(MAXINV_CLIPS * clip))
                    params["soft_inv"] = int(round(SOFTINV_CLIPS * clip))
                    params["session_scale"] = scales[sym]
                    # per-name POV time windows (both variants keep the flat-side
                    # ramp/cliff widening; only the HOLDING engagement differs)
                    rmp, clf = windows.get(sym, (5.0, 1.0))
                    params["eod_ramp_start_min"] = rmp
                    params["eod_cliff_min"] = clf
                    # NEW: the POV unwind model gets the profile + segments
                    if v == "NEW":
                        params["unwind_profile"] = profiles[sym]
                        params["unwind_pov"] = MAX_POV
                        params["session_segments"] = segs
                    cfg = dict(R.CFG, session=(t0_, t1_),
                               latency_model=LatencyModel(seed=R.LATENCY_SEED))
                    strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
                    bt = Backtester(strat, cfg)
                    fills, equity, stats = bt.run(events, snap_groups)
                    nf, cap_b, mk_b, net_b = C.score_bps(fills, fs_day)
                    f = fills if isinstance(fills, pd.DataFrame) else pd.DataFrame(fills)
                    # window-tagged fills
                    wc = {"time_ramp": 0, "time_cliff": 0, "lock_ramp": 0,
                          "lock_cliff": 0}
                    if len(f) and "window" in f.columns:
                        for w, n in f["window"].value_counts().items():
                            if w in wc:
                                wc[w] = int(n)
                    liq = bt.eod["equity_liquidated"] if bt.eod is not None else None
                    eq_mid = bt.eod.get("equity_mid_mark") if bt.eod is not None else None
                    resid = bt.eod.get("residual_marked", 0.0) if bt.eod is not None else None
                    a = acc[(sym, v, mlt)]
                    if liq is not None:
                        a["pnl"] += float(liq)
                        a["days"] += 1
                    a["fills"] += nf
                    if isinstance(net_b, (float, np.floating)) and not np.isnan(net_b):
                        a["net"].append(float(net_b))
                    if bt.eod is not None and bt.eod["liquidation_clean"] is False:
                        a["unclean"] += 1
                    rows.append({"date": str(date), "symbol": sym, "strategy": v,
                                 "clip_mult": mlt, "clip_sh": clip, "fills": nf,
                                 "pnl_pkr": (round(float(liq), 2) if liq is not None else np.nan),
                                 "pnl_mid_mark": (round(float(eq_mid), 2) if eq_mid is not None else np.nan),
                                 "residual_marked": (round(float(resid), 2) if resid is not None else np.nan),
                                 "fills_time_ramp": wc["time_ramp"],
                                 "fills_time_cliff": wc["time_cliff"],
                                 "fills_lock_ramp": wc["lock_ramp"],
                                 "fills_lock_cliff": wc["lock_cliff"],
                                 "net_bps": (round(net_b, 3) if isinstance(net_b, float) else np.nan)})
            # heartbeat with ETA
            if sd % 10 == 0 or sd == total:
                el = time.perf_counter() - t0_all
                eta = el / sd * (total - sd)
                print(f"  {sd}/{total} symbol-days  elapsed {C._fmt(el)}  "
                      f"ETA {C._fmt(eta)}", flush=True)

    # ---- outputs ----
    daily_csv = RESULTS / f"sweep_rejects_daily_{stamp}.csv"
    pd.DataFrame(rows).to_csv(daily_csv, index=False)

    # ---- the verdict table: NEW minus OLD per (symbol, clip) ----
    print(f"\n=== NEW (POV unwind) vs OLD (fixed window), per clip ===")
    print(f"{'sym':8s} {'mult':>5s} {'old_pnl':>10s} {'new_pnl':>10s} {'delta':>9s} "
          f"{'old_uncl':>8s} {'new_uncl':>8s} {'old_bps':>8s} {'new_bps':>8s}")
    summ = []
    for sym in TOP10:
        for mlt in MULTS:
            o = acc[(sym, "OLD", mlt)]
            n = acc[(sym, "NEW", mlt)]
            if o["days"] == 0:
                continue
            d = n["pnl"] - o["pnl"]
            print(f"{sym:8s} {mlt:>5.1f} {o['pnl']:>10,.0f} {n['pnl']:>10,.0f} "
                  f"{d:>+9,.0f} {o['unclean']:>8d} {n['unclean']:>8d} "
                  f"{np.nanmean(o['net']) if o['net'] else float('nan'):>8.3f} "
                  f"{np.nanmean(n['net']) if n['net'] else float('nan'):>8.3f}")
            summ.append({"symbol": sym, "clip_mult": mlt, "old_pnl": o["pnl"],
                         "new_pnl": n["pnl"], "delta": d,
                         "old_unclean": o["unclean"], "new_unclean": n["unclean"]})
    S = pd.DataFrame(summ)
    rank_csv = RESULTS / f"sweep_rejects_summary_{stamp}.csv"
    S.to_csv(rank_csv, index=False)
    # the headline: total delta per clip across the 10 names
    print("\nTOTAL new-minus-old per clip (the hypothesis test):")
    for mlt in MULTS:
        d = S[S.clip_mult == mlt]["delta"].sum()
        print(f"  {mlt:.0f}x: {d:>+12,.0f}")
    print("\nHYPOTHESIS: ~0 or slightly + at 1x (dormancy recovers the give-back);")
    print("increasingly positive at 3-5x (the unwind earns its keep at size).")
    print(f"\nwrote {rank_csv}\nwrote {daily_csv}")


if __name__ == "__main__":
    main()
