# sweep_middle_capacity.py -- THE INVERSE of the failed close-sizing test.
#
# The bucket-sizing result showed: the CLOSE has high volume but TOXIC flow
# (last15 fills ~1.0 bps, eroded to negative when sized up). The MIDDLE has the
# real edge (~4.7 bps/fill). SZ's hypothesis: size up in the MIDDLE (clean,
# high-edge flow), carry the extra inventory, and LIQUIDATE it into the deep
# closing volume. Earn in the middle; exit at the close.
#
# This is NOT the refuted "size up where volume is high" thesis -- it is "size up
# where the EDGE is, exit where the LIQUIDITY is." Different, and supported by the
# 4.7-vs-1.0 bps split. But two traps must be measured, not assumed:
#   (A) CAPACITY: does middle net_bps HOLD as the middle clip scales, or does
#       adverse selection erode it (bigger mid clips attract informed flow too)?
#   (B) COUPLING/EXIT COST: bigger mid inventory must be unwound into the close.
#       The bucket test already showed close fill-quality COLLAPSES as carried
#       inventory rises (last15 bps 1.0 -> -0.005 when preclose inventory grew).
#       Exiting into a TOXIC close may cost more than the mid capture gained.
#
# Design: first15/preclose45/last15 fixed at BASE_MULT (3x); sweep MIDDLE up.
# Decision: (1) middle net_bps must HOLD as it scales (capacity), AND (2) TOTAL
# net P&L must climb AFTER the close-exit drag. Instrumentation isolates both:
# per-bucket net_bps, per-bucket mean|inventory|, last15 unclean contribution.
#
# Top-10 winners (where the edge is). Fixed 3x baseline on the other buckets.
# Run from existing_mm_live/:  caffeinate -is python3 sweep_middle_capacity.py

from pathlib import Path
import time
from datetime import datetime
import pandas as pd
import numpy as np
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel
from micro_mm import MicrostructureMM
import confirm_micro_vs_naive as C

R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ knobs ----------------------------------------
NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
# base multiplier for first15/preclose45/last15 (the locked production knee)
BASE_MULT = 3.0
# MIDDLE multipliers to sweep (base included as the control)
MIDDLE_LEVELS = [3.0, 5.0, 8.0, 12.0]
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
MAX_POV = 0.10
TRAIL_DAYS = 10
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_eod_trigger=True, enable_lock_trigger=True)
BUCKETS = ("first15", "middle", "preclose45", "last15")
# -----------------------------------------------------------------------------


def newest(pattern):
    c = sorted(RESULTS.glob(pattern))
    if not c:
        raise SystemExit(f"missing {pattern}")
    return c[-1]


def load_table(pattern, cols):
    df = pd.read_csv(newest(pattern))
    ok = df[df["note"] == "ok"] if "note" in df.columns else df
    return {r["symbol"]: tuple(r[c] for c in cols) for _, r in ok.iterrows()}


def load_segments():
    df = pd.read_csv(newest("session_segments_*.csv"))
    return {r["date"]: [tuple(int(x) for x in p.split(":"))
                        for p in r["segments"].split(";")]
            for _, r in df.iterrows()}


def trailing(stats_sym, all_dates, date):
    prior = [stats_sym[str(d)] for d in all_dates
             if str(d) < str(date) and str(d) in stats_sym]
    if len(prior) < TRAIL_DAYS:
        return None
    return float(np.median(prior[-TRAIL_DAYS:]))


# per-bucket net_bps from the fills frame (tagged with 'bucket') + feature store
def per_bucket_bps(fills, fs_day):
    f = fills if isinstance(fills, pd.DataFrame) else pd.DataFrame(fills)
    out = {}
    if len(f) == 0 or "bucket" not in f.columns:
        return out
    for b, g in f.groupby("bucket"):
        nf, cap_b, mk_b, net_b = C.score_bps(g, fs_day)
        # count unclean-liq attribution is per-day, not per-bucket; we track
        # last15 fills as the exit-quality proxy
        out[b] = {"fills": nf, "net_bps": net_b}
    return out


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
    for s in NAMES:
        if s not in scales or s not in profiles:
            raise SystemExit(f"{s} missing calibration")

    all_dates = R.discover_dates()
    print(f"pre-pass: trailing median trade size, {len(all_dates)} days", flush=True)
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
            print(f"  pre-pass {i}/{len(all_dates)}  "
                  f"elapsed {C._fmt(time.perf_counter() - t0)}", flush=True)

    run_dates = all_dates[TRAIL_DAYS:]
    total = len(run_dates) * len(NAMES)
    print(f"\nsweep_middle_capacity: {len(MIDDLE_LEVELS)} middle levels x "
          f"{len(NAMES)} names x {len(run_dates)} days "
          f"(others fixed at {BASE_MULT:g}x)\n", flush=True)

    acc = {}
    for s in NAMES:
        for mv in MIDDLE_LEVELS:
            acc[(s, mv)] = {"pnl": 0.0, "days": 0, "unclean": 0,
                            "bps": {b: [] for b in BUCKETS},
                            "fills": {b: 0 for b in BUCKETS},
                            "inv": {b: [] for b in BUCKETS}}
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
            s_ = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0 or len(s_) == 0:
                continue
            events, snap_groups, t = R.build_events(u, s_, t)
            cont = s_[s_["phase"] == "CONTINUOUS_AUCTION"]
            if len(cont) == 0:
                continue
            t0_, t1_ = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
            clip1 = max(1, int(round(med_qty)))
            for mv in MIDDLE_LEVELS:
                params = dict(MID_BASE)
                params["size"] = clip1
                # max_inv rides at the LARGEST bucket clip so the sized-up middle
                # is never capped by the inventory ceiling
                top_mult = max(BASE_MULT, mv)
                params["max_inv"] = int(round(MAXINV_CLIPS * top_mult * clip1))
                params["soft_inv"] = int(round(SOFTINV_CLIPS * top_mult * clip1))
                params["session_scale"] = scales[sym]
                rmp, clf = windows.get(sym, (5.0, 1.0))
                params["eod_ramp_start_min"] = rmp
                params["eod_cliff_min"] = clf
                params["unwind_profile"] = profiles[sym]
                params["unwind_pov"] = MAX_POV
                params["session_segments"] = segs
                # (first15, MIDDLE swept, preclose45, last15)
                params["bucket_mult"] = (BASE_MULT, mv, BASE_MULT, BASE_MULT)
                cfg = dict(R.CFG, session=(t0_, t1_),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
                bt = Backtester(strat, cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                liq = bt.eod["equity_liquidated"] if bt.eod is not None else None
                pb = per_bucket_bps(fills, fs_day)
                a = acc[(sym, mv)]
                if liq is not None:
                    a["pnl"] += float(liq)
                    a["days"] += 1
                if bt.eod is not None and bt.eod["liquidation_clean"] is False:
                    a["unclean"] += 1
                for b in BUCKETS:
                    if b in pb:
                        if isinstance(pb[b]["net_bps"], float) and not np.isnan(pb[b]["net_bps"]):
                            a["bps"][b].append(pb[b]["net_bps"])
                        a["fills"][b] += pb[b]["fills"]
                    it = strat._inv_time.get(b, 0.0)
                    if it > 0:
                        a["inv"][b].append(strat._pos_area[b] / it)
                rows.append({"date": str(date), "symbol": sym, "middle_mult": mv,
                             "pnl_pkr": (round(float(liq), 2) if liq is not None else np.nan),
                             "unclean": int(bt.eod is not None and bt.eod["liquidation_clean"] is False),
                             **{f"bps_{b}": (round(pb[b]["net_bps"], 3)
                                            if b in pb and isinstance(pb[b]["net_bps"], float) else np.nan)
                                for b in BUCKETS},
                             **{f"inv_{b}": (round(strat._pos_area[b] / strat._inv_time[b], 1)
                                            if strat._inv_time.get(b, 0) > 0 else np.nan)
                                for b in BUCKETS}})
            if sd % 10 == 0 or sd == total:
                el = time.perf_counter() - t0_all
                print(f"  {sd}/{total} symbol-days  elapsed {C._fmt(el)}  "
                      f"ETA {C._fmt(el / sd * (total - sd))}", flush=True)

    daily = RESULTS / f"middle_capacity_daily_{stamp}.csv"
    pd.DataFrame(rows).to_csv(daily, index=False)

    # ---- verdict: the two traps, side by side ----
    print("\n=== MIDDLE-BUCKET CAPACITY: earn in the middle, exit at the close ===")
    print("(A) middle net_bps must HOLD as middle_mult rises (real capacity).")
    print("(B) close-exit cost: last15 net_bps + unclean must NOT blow out as the")
    print("    carried inventory (inv_last) rises. Net P&L is the final judge.\n")
    print(f"{'mid_x':>6s} {'total_pnl':>12s} {'dPnL':>10s} {'bps_mid':>8s} "
          f"{'bps_last':>8s} {'inv_mid':>8s} {'inv_last':>8s} {'unclean':>7s}")

    def agg(mv, key, bucket):
        vals = [v for s in NAMES for v in acc[(s, mv)][key][bucket]]
        return np.mean(vals) if vals else float("nan")

    base_pnl = sum(acc[(s, BASE_MULT)]["pnl"] for s in NAMES)
    summ = []
    for mv in MIDDLE_LEVELS:
        tot = sum(acc[(s, mv)]["pnl"] for s in NAMES)
        unc = sum(acc[(s, mv)]["unclean"] for s in NAMES)
        print(f"{mv:>6.1f} {tot:>12,.0f} {tot-base_pnl:>+10,.0f} "
              f"{agg(mv,'bps','middle'):>8.3f} {agg(mv,'bps','last15'):>8.3f} "
              f"{agg(mv,'inv','middle'):>8.0f} {agg(mv,'inv','last15'):>8.0f} "
              f"{unc:>7d}")
        summ.append({"middle_mult": mv, "total_pnl": tot, "dpnl": tot-base_pnl,
                     "bps_middle": agg(mv, "bps", "middle"),
                     "bps_last15": agg(mv, "bps", "last15"),
                     "inv_middle": agg(mv, "inv", "middle"),
                     "inv_last15": agg(mv, "inv", "last15"), "unclean": unc})
    pd.DataFrame(summ).to_csv(RESULTS / f"middle_capacity_summary_{stamp}.csv",
                              index=False)
    print("\nREAD: if bps_mid HOLDS near base AND dPnL climbs AND bps_last/unclean")
    print("stay controlled -> the middle has real capacity, size it up (the win).")
    print("If bps_mid erodes as mid_x rises -> even clean flow saturates; optimal is")
    print("the knee. If bps_last collapses / unclean spikes -> the close-exit cost")
    print("eats the mid gain (the coupling trap) -> carrying doesn't pay.")
    print(f"\nwrote {RESULTS / f'middle_capacity_summary_{stamp}.csv'}\nwrote {daily}")


if __name__ == "__main__":
    main()
