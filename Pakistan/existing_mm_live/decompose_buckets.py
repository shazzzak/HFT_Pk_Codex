# decompose_buckets.py -- CAPTURE / MARKOUT / FEE breakdown per session bucket.
#
# The bucket-size sweep persisted only net_bps per bucket. This recomputes the
# full decomposition (net = capture + markout - fee) per bucket on the BASE 3x
# config, so we can see WHY the open loses and the mid/preclose win:
#   capture  = half-spread earned at fill (fill inside the mid) -- the reward
#   markout  = adverse mid move over the horizon vs our position -- the cost
#   fee      = constant spot TREC round-trip (1.554 bps) -- same every bucket
#
# Uses the SAME formulas as fill_attribution.py (capture/markout/net), scored
# against the feature-store mids (the validated price basis). RECONCILIATION
# CHECK: per-bucket net here must match the net_bps already in the bucket-size
# CSV -- if it does, the split is trustworthy; if not, we've found a bug.
#
# One config (base 3x), top-10, ~187 days -> ~1/7th of the bucket-size sweep.
# Run:  caffeinate -is python3 decompose_buckets.py

from pathlib import Path
import time
from datetime import datetime
import pandas as pd
import numpy as np
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel, FEE_TOTAL_PCT
from micro_mm import MicrostructureMM
import confirm_micro_vs_naive as C

R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
CLIP_MULT = 3.0
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
MAX_POV = 0.10
TRAIL_DAYS = 10
# NOTE: markout horizon is NOT set here -- we call the EXISTING C.score_bps,
# which joins the forward mid via PF.join_fill_context (5s primary, the tested
# no-leak join). This GUARANTEES the decomposition reconciles to the sweep's
# net_bps, because it is the identical code path. No reimplementation.
BUCKETS = ("first15", "middle", "preclose45", "last15")
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_eod_trigger=True, enable_lock_trigger=True)
# round-trip fee in bps (constant across buckets)
FEE_RT_BPS = FEE_TOTAL_PCT * 2 * 1e4


def newest(p):
    c = sorted(RESULTS.glob(p))
    if not c:
        raise SystemExit(f"missing {p}")
    return c[-1]


def load_table(p, cols):
    df = pd.read_csv(newest(p))
    ok = df[df["note"] == "ok"] if "note" in df.columns else df
    return {r["symbol"]: tuple(r[c] for c in cols) for _, r in ok.iterrows()}


def load_segments():
    df = pd.read_csv(newest("session_segments_*.csv"))
    return {r["date"]: [tuple(int(x) for x in q.split(":"))
                        for q in r["segments"].split(";")]
            for _, r in df.iterrows()}


def trailing(stats_sym, all_dates, date):
    prior = [stats_sym[str(d)] for d in all_dates
             if str(d) < str(date) and str(d) in stats_sym]
    if len(prior) < TRAIL_DAYS:
        return None
    return float(np.median(prior[-TRAIL_DAYS:]))


# group fills by bucket and score each group with the VALIDATED C.score_bps,
# which returns (n, capture, markout, net) -- the sweep only kept net. Same join,
# same fee, same horizon as every other run -> reconciles by construction.
def decompose(fills, fs_day):
    f = fills if isinstance(fills, pd.DataFrame) else pd.DataFrame(fills)
    if len(f) == 0 or "bucket" not in f.columns:
        return {}
    out = {}
    for b, g in f.groupby("bucket"):
        n, cap, mko, net = C.score_bps(g, fs_day)
        if n > 0 and np.isfinite(net):
            out[b] = {"fills": n, "capture": cap, "markout": mko, "net": net}
    return out


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"fee (round-trip): {FEE_RT_BPS:.4f} bps  (constant across buckets)")
    scales = {k: v[0] for k, v in load_table("session_scales_*.csv",
                                             ["session_scale"]).items()}
    profiles = load_table("volume_profile_*.csv",
                          ["vol_first15", "vol_middle", "vol_preclose45",
                           "vol_last15"])
    windows = load_table("time_windows_*.csv",
                         ["eod_ramp_start_min", "eod_cliff_min"])
    segments = load_segments()

    all_dates = R.discover_dates()
    print(f"pre-pass: trailing median trade size", flush=True)
    tstats = {s: {} for s in NAMES}
    t0 = time.perf_counter()
    for i, date in enumerate(all_dates, 1):
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        for s in NAMES:
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, s)
            if len(t):
                tstats[s][str(date)] = float(t["qty"].median())
        if i % 50 == 0 or i == len(all_dates):
            print(f"  {i}/{len(all_dates)}  {C._fmt(time.perf_counter()-t0)}",
                  flush=True)

    run_dates = all_dates[TRAIL_DAYS:]
    total = len(run_dates) * len(NAMES)
    # accumulators: per bucket, concatenated component arrays (fills-level)
    agg = {b: {"capture": [], "markout": [], "net": []} for b in BUCKETS}
    rows = []
    print(f"\ndecompose_buckets: base 3x, {len(NAMES)} names x {len(run_dates)} days\n",
          flush=True)
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
            fs_day = pd.read_parquet(fs_path, columns=[
                "ts_exch", "mid", "spread_bps", "obi_1", "toxicity",
                "realized_vol_bps", "markout_5000ms_bps"])
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s_ = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0 or len(s_) == 0:
                continue
            events, snap_groups, t = R.build_events(u, s_, t)
            cont = s_[s_["phase"] == "CONTINUOUS_AUCTION"]
            if len(cont) == 0:
                continue
            t0_, t1_ = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
            clip = max(1, int(round(CLIP_MULT * med_qty)))
            params = dict(MID_BASE)
            params["size"] = clip
            params["max_inv"] = int(round(MAXINV_CLIPS * clip))
            params["soft_inv"] = int(round(SOFTINV_CLIPS * clip))
            params["session_scale"] = scales[sym]
            rmp, clf = windows.get(sym, (5.0, 1.0))
            params["eod_ramp_start_min"] = rmp
            params["eod_cliff_min"] = clf
            params["unwind_profile"] = profiles[sym]
            params["unwind_pov"] = MAX_POV
            params["session_segments"] = segs
            cfg = dict(R.CFG, session=(t0_, t1_),
                       latency_model=LatencyModel(seed=R.LATENCY_SEED))
            strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
            bt = Backtester(strat, cfg)
            fills, equity, stats = bt.run(events, snap_groups)
            dec = decompose(fills, fs_day)
            for b, d in dec.items():
                if b in agg:
                    # store (value, weight) for fills-weighted portfolio means
                    agg[b]["capture"].append((d["capture"], d["fills"]))
                    agg[b]["markout"].append((d["markout"], d["fills"]))
                    agg[b]["net"].append((d["net"], d["fills"]))
                    rows.append({"date": str(date), "symbol": sym, "bucket": b,
                                 "fills": d["fills"],
                                 "capture": round(d["capture"], 3),
                                 "markout": round(d["markout"], 3),
                                 "net": round(d["net"], 3)})
            if sd % 10 == 0 or sd == total:
                el = time.perf_counter() - t0_all
                print(f"  {sd}/{total}  {C._fmt(el)}  "
                      f"ETA {C._fmt(el/sd*(total-sd))}", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / f"bucket_decomp_daily_{stamp}.csv",
                              index=False)

    # ---- the decomposition table (fills-weighted, P&L-relevant) ----
    print("\n=== CAPTURE / MARKOUT / FEE by session bucket (base 3x) ===")
    print("net = capture + markout - fee   (fee constant "
          f"{FEE_RT_BPS:.3f} bps)\n")
    print(f"{'bucket':12s} {'capture':>9s} {'markout':>9s} {'fee':>7s} "
          f"{'net':>8s} {'fills':>9s}")
    summ = []
    def wmean(pairs):
        v = np.array([p[0] for p in pairs]); w = np.array([p[1] for p in pairs], float)
        return float(np.average(v, weights=w)) if w.sum() > 0 else float("nan")
    for b in BUCKETS:
        if not agg[b]["net"]:
            print(f"{b:12s}  (no fills)")
            continue
        cap = wmean(agg[b]["capture"]); mko = wmean(agg[b]["markout"])
        net = wmean(agg[b]["net"]); tot = int(sum(p[1] for p in agg[b]["net"]))
        # capture + markout - fee should equal net (reconciliation within rounding)
        print(f"{b:12s} {cap:>9.3f} {mko:>9.3f} {-FEE_RT_BPS:>7.3f} "
              f"{net:>8.3f} {tot:>9,}")
        summ.append({"bucket": b, "capture": cap, "markout": mko,
                     "fee": -FEE_RT_BPS, "net": net, "fills": tot,
                     "recon_check": round(cap + mko - FEE_RT_BPS - net, 4)})
    pd.DataFrame(summ).to_csv(RESULTS / f"bucket_decomp_summary_{stamp}.csv",
                              index=False)
    print("\nRECONCILE: 'net' above should match the bps_<bucket> means in the")
    print("bucket-size CSV (base config). If they agree, the split is trustworthy.")
    print("READ: capture = spread earned; markout = adverse selection (negative =")
    print("picked off). The open's story is whether capture is LOW or markout is HIGH.")
    print(f"\nwrote {RESULTS / f'bucket_decomp_summary_{stamp}.csv'}")


if __name__ == "__main__":
    main()
