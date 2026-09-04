# sweep_bucket_size.py -- JOB 1 of the volume-sizing experiment: can we extend
# profitability by quoting LARGER clips in the high-volume end-of-day buckets?
#
# The measured question (not assumed): if we scale the clip up in preclose45 and
# last15, does net_bps per bucket HOLD (real capacity -> more P&L) or ERODE
# (adverse selection eats the extra spread -> buying volume with edge)? The close
# buckets are the safe place to try this: short risk window, deep closing
# liquidation. The OPEN is deferred to job 2 (needs the discovery gate).
#
# Baseline: middle=preclose=last15 = the 3x production clip (flat, no scaling).
# Treatment: middle held at 3x; sweep preclose45 and last15 multipliers UP.
# To isolate each bucket, we sweep them INDEPENDENTLY (preclose alone, then
# last15 alone) around the baseline -- not a joint grid.
#
# The bucket_mult tuple is (first15, middle, preclose45, last15) as MULTIPLES OF
# THE MEDIAN TRADE SIZE (same unit as the clip). middle=3.0 anchors the baseline;
# first15 held at 3.0 (job 2 handles the open). Instrumentation per bucket:
# net_bps, fills, and mean|inventory| (holding-time proxy).
#
# Decision metric: net_bps PER BUCKET (P&L reported alongside). Read RELATIVELY
# across multipliers -- the fill model under-prices large-clip adverse selection,
# so a bucket whose net_bps holds as size rises has real capacity; one that
# erodes does not. Run from existing_mm_live/:  python sweep_bucket_size.py

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
# scorer + formatter
import confirm_micro_vs_naive as C

R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ experiment knobs ------------------------------
# the top-10 winners (where the P&L is; extending THESE is the prize)
NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
# baseline clip multiple for middle + first15 (the production knee)
BASE_MULT = 3.0
# preclose45 / last15 multipliers to sweep (as multiples of median trade size)
BUCKET_LEVELS = [3.0, 5.0, 8.0, 12.0]
# inventory limits ride at these ratios to the LIVE clip (max_inv scales with it)
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
MAX_POV = 0.10
TRAIL_DAYS = 10
# locked production MID base (NEW POV unwind on)
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_eod_trigger=True, enable_lock_trigger=True)
# ------------------------------------------------------------------------------


def newest(pattern):
    cands = sorted(RESULTS.glob(pattern))
    if not cands:
        raise SystemExit(f"missing {pattern}")
    return cands[-1]


def load_table(pattern, cols):
    df = pd.read_csv(newest(pattern))
    ok = df[df["note"] == "ok"] if "note" in df.columns else df
    return {r["symbol"]: tuple(r[c] for c in cols) for _, r in ok.iterrows()}


def load_segments():
    df = pd.read_csv(newest("session_segments_*.csv"))
    out = {}
    for _, r in df.iterrows():
        out[r["date"]] = [tuple(int(x) for x in p.split(":"))
                          for p in r["segments"].split(";")]
    return out


def trailing(stats_sym, all_dates, date):
    prior = [stats_sym[str(d)] for d in all_dates
             if str(d) < str(date) and str(d) in stats_sym]
    if len(prior) < TRAIL_DAYS:
        return None
    return float(np.median(prior[-TRAIL_DAYS:]))


# score net_bps PER BUCKET from the fills frame + the feature-store mid/markouts.
# reuses C.score_bps' logic but grouped by the fill's bucket tag.
def per_bucket_bps(fills, fs_day):
    f = fills if isinstance(fills, pd.DataFrame) else pd.DataFrame(fills)
    if len(f) == 0 or "bucket" not in f.columns:
        return {}
    out = {}
    for b, g in f.groupby("bucket"):
        nf, cap_b, mk_b, net_b = C.score_bps(g, fs_day)
        out[b] = {"fills": nf, "net_bps": net_b}
    return out


# the config grid: baseline (flat 3x) + preclose sweep + last15 sweep (independent)
def build_configs():
    cfgs = [("base", BASE_MULT, BASE_MULT)]  # (label, preclose_mult, last15_mult)
    # preclose swept, last15 at base
    for lv in BUCKET_LEVELS:
        if lv != BASE_MULT:
            cfgs.append((f"preclose{lv:g}", lv, BASE_MULT))
    # last15 swept, preclose at base
    for lv in BUCKET_LEVELS:
        if lv != BASE_MULT:
            cfgs.append((f"last15_{lv:g}", BASE_MULT, lv))
    return cfgs


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    scales = {k: v[0] for k, v in load_table("session_scales_*.csv",
                                             ["session_scale"]).items()}
    profiles = load_table("volume_profile_*.csv",
                          ["vol_first15", "vol_middle", "vol_preclose45",
                           "vol_last15"])
    windows = load_table("time_windows_*.csv",
                         ["eod_ramp_start_min", "eod_cliff_min"])
    segments = load_segments()
    for sym in NAMES:
        if sym not in scales or sym not in profiles:
            raise SystemExit(f"{sym} missing calibration")

    CONFIGS = build_configs()
    all_dates = R.discover_dates()
    print(f"pre-pass: trailing median trade size, {len(all_dates)} days "
          f"x {len(NAMES)} names", flush=True)
    tstats = {s: {} for s in NAMES}
    t0 = time.perf_counter()
    for i, date in enumerate(all_dates, 1):
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        for sym in NAMES:
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t):
                tstats[sym][str(date)] = float(t["qty"].median())
        if i % 50 == 0 or i == len(all_dates):
            print(f"  pre-pass {i}/{len(all_dates)}  "
                  f"elapsed {C._fmt(time.perf_counter() - t0)}", flush=True)

    run_dates = all_dates[TRAIL_DAYS:]
    total = len(run_dates) * len(NAMES)
    print(f"\nsweep_bucket_size: {len(CONFIGS)} configs x {len(NAMES)} names x "
          f"{len(run_dates)} days (base {BASE_MULT:g}x, sweep {BUCKET_LEVELS})\n",
          flush=True)

    # accumulators keyed (sym, config); per-bucket net_bps lists + fills + holding
    acc = {}
    for sym in NAMES:
        for label, _, _ in CONFIGS:
            acc[(sym, label)] = {"pnl": 0.0, "days": 0,
                                 "bucket_bps": {b: [] for b in
                                                ("first15", "middle",
                                                 "preclose45", "last15")},
                                 "bucket_fills": {b: 0 for b in
                                                  ("first15", "middle",
                                                   "preclose45", "last15")},
                                 "bucket_meaninv": {b: [] for b in
                                                    ("first15", "middle",
                                                     "preclose45", "last15")}}
    rows = []
    t0_all = time.perf_counter()
    sd = 0
    for date in run_dates:
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        segs = segments.get(str(date))
        if segs is None:
            continue
        for sym in NAMES:
            sd += 1
            med_qty = trailing(tstats[sym], all_dates, date)
            if med_qty is None or med_qty <= 0:
                continue
            fs_path = FS_ROOT / sym / f"date={date}.parquet"
            if not fs_path.exists():
                continue
            fs_day = pd.read_parquet(
                fs_path, columns=["ts_exch", "mid", "spread_bps", "obi_1",
                                  "toxicity", "realized_vol_bps"])
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
            for label, pre_m, last_m in CONFIGS:
                # base clip = median trade size (multipliers are in these units)
                clip1 = max(1, int(round(med_qty)))
                params = dict(MID_BASE)
                params["size"] = clip1
                # max_inv/soft_inv ride at ratios to the LARGEST bucket clip so
                # the ceiling never caps a sized-up bucket (max mult in play)
                top_mult = max(BASE_MULT, pre_m, last_m)
                params["max_inv"] = int(round(MAXINV_CLIPS * top_mult * clip1))
                params["soft_inv"] = int(round(SOFTINV_CLIPS * top_mult * clip1))
                params["session_scale"] = scales[sym]
                rmp, clf = windows.get(sym, (5.0, 1.0))
                params["eod_ramp_start_min"] = rmp
                params["eod_cliff_min"] = clf
                params["unwind_profile"] = profiles[sym]
                params["unwind_pov"] = MAX_POV
                params["session_segments"] = segs
                # per-bucket multipliers: (first15, middle, preclose45, last15)
                # first15 at base (job 2 handles the open); middle at base
                params["bucket_mult"] = (BASE_MULT, BASE_MULT, pre_m, last_m)
                cfg = dict(R.CFG, session=(t0_, t1_),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
                bt = Backtester(strat, cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                liq = bt.eod["equity_liquidated"] if bt.eod is not None else None
                pb = per_bucket_bps(fills, fs_day)
                a = acc[(sym, label)]
                if liq is not None:
                    a["pnl"] += float(liq)
                    a["days"] += 1
                for b, d in pb.items():
                    if b in a["bucket_bps"]:
                        if isinstance(d["net_bps"], float) and not np.isnan(d["net_bps"]):
                            a["bucket_bps"][b].append(d["net_bps"])
                        a["bucket_fills"][b] += d["fills"]
                # per-bucket mean|inventory| (holding-time proxy)
                for b in a["bucket_meaninv"]:
                    it = strat._inv_time.get(b, 0.0)
                    if it > 0:
                        a["bucket_meaninv"][b].append(strat._pos_area[b] / it)
                rows.append({"date": str(date), "symbol": sym, "config": label,
                             "pnl_pkr": (round(float(liq), 2) if liq is not None else np.nan),
                             **{f"bps_{b}": (round(pb[b]["net_bps"], 3)
                                            if b in pb and isinstance(pb[b]["net_bps"], float) else np.nan)
                                for b in ("first15", "middle", "preclose45", "last15")},
                             **{f"fills_{b}": (pb[b]["fills"] if b in pb else 0)
                                for b in ("first15", "middle", "preclose45", "last15")},
                             **{f"meaninv_{b}": (round(strat._pos_area[b] / strat._inv_time[b], 1)
                                                if strat._inv_time.get(b, 0) > 0 else np.nan)
                                for b in ("first15", "middle", "preclose45", "last15")}})
            if sd % 10 == 0 or sd == total:
                el = time.perf_counter() - t0_all
                print(f"  {sd}/{total} symbol-days  elapsed {C._fmt(el)}  "
                      f"ETA {C._fmt(el / sd * (total - sd))}", flush=True)

    daily = RESULTS / f"bucket_size_daily_{stamp}.csv"
    pd.DataFrame(rows).to_csv(daily, index=False)

    # ---- verdict: for each config, total P&L + net_bps in the swept bucket ----
    print("\n=== BUCKET SIZING: does net_bps HOLD as the clip scales up? ===")
    print("(net_bps read RELATIVELY vs base; holding = real capacity, eroding =")
    print(" adverse selection eating the spread. P&L reported alongside.)\n")
    summ = []
    for label, pre_m, last_m in CONFIGS:
        tot_pnl = sum(acc[(s, label)]["pnl"] for s in NAMES)
        # the bucket being swept in this config
        swept = "preclose45" if label.startswith("preclose") else (
            "last15" if label.startswith("last15") else "both")
        # portfolio-mean net_bps in each end bucket
        def mean_bps(b):
            vals = [v for s in NAMES for v in acc[(s, label)]["bucket_bps"][b]]
            return np.mean(vals) if vals else float("nan")
        def mean_inv(b):
            vals = [v for s in NAMES for v in acc[(s, label)]["bucket_meaninv"][b]]
            return np.mean(vals) if vals else float("nan")
        summ.append({"config": label, "swept_bucket": swept,
                     "total_pnl": tot_pnl,
                     "bps_preclose45": mean_bps("preclose45"),
                     "bps_last15": mean_bps("last15"),
                     "meaninv_preclose45": mean_inv("preclose45"),
                     "meaninv_last15": mean_inv("last15")})
    S = pd.DataFrame(summ)
    base_pnl = S[S.config == "base"]["total_pnl"].iloc[0]
    print(f"{'config':14s} {'total_pnl':>12s} {'dPnL vs base':>13s} "
          f"{'bps_pre':>8s} {'bps_last':>8s} {'inv_pre':>8s} {'inv_last':>8s}")
    for _, r in S.iterrows():
        print(f"{r['config']:14s} {r['total_pnl']:>12,.0f} "
              f"{r['total_pnl']-base_pnl:>+13,.0f} "
              f"{r['bps_preclose45']:>8.3f} {r['bps_last15']:>8.3f} "
              f"{r['meaninv_preclose45']:>8.0f} {r['meaninv_last15']:>8.0f}")
    print("\nREAD: follow the swept bucket's own net_bps column down the levels.")
    print("If bps_last15 holds ~flat as last15_* rises AND dPnL climbs -> real")
    print("capacity, size that bucket up. If bps erodes as size rises -> the extra")
    print("volume is toxic; the optimal multiplier is where bps starts to break.")
    S.to_csv(RESULTS / f"bucket_size_summary_{stamp}.csv", index=False)
    print(f"\nwrote {RESULTS / f'bucket_size_summary_{stamp}.csv'}\nwrote {daily}")


if __name__ == "__main__":
    main()
