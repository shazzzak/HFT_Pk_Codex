# sweep_open_suppress.py -- does SITTING OUT the volatile open pay?
#
# Thesis: the open impounds overnight news through a wide, chaotic spread. PSX
# jumps have no book precursor, but the OPEN is a scheduled window of concentrated
# gap risk. Fully suppressing quotes until discovery settles avoids some gap
# losses -- but FORGOES real spread capture on calm mornings (the open spread is
# WIDE = profitable when there's no gap). Net sign is unknown -> measure it.
#
# Gate (SZ's design): resume quoting when MAX/AND -- time elapsed >= wait AND
# current spread <= trailing_median_open_spread + k*trailing_sd. The reference is
# WALK-FORWARD per name (median/sd of the first-15-min spread over PRIOR days),
# never the current day.
#
# Baseline: no gate. Treatment: the gate across (wait x k). Decision: total P&L
# vs baseline, DECOMPOSED into (a) forgone open-window fills [the cost] and (b)
# open-window P&L avoided on gap mornings [the benefit] -- so a positive result
# is shown to be real gap-avoidance, not accidental (the reactive-gate lesson).
#
# Top-10 winners. Run:  caffeinate -is python3 sweep_open_suppress.py

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
CLIP_MULT = 3.0
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
MAX_POV = 0.10
TRAIL_DAYS = 10
# gate grid
WAIT_S = [60.0, 120.0, 300.0]          # min seconds before resuming
K_SD = [1.0, 2.0, 3.0]                 # spread must be within k*sd of trailing med
# trailing window (days) for the opening-spread reference
REF_TRAIL = 10
# open window = first 15 tradeable minutes
OPEN_MIN = 15.0
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_eod_trigger=True, enable_lock_trigger=True)
# -----------------------------------------------------------------------------


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


def trailing_med(stats_sym, all_dates, date, ndays):
    prior = [stats_sym[str(d)] for d in all_dates
             if str(d) < str(date) and str(d) in stats_sym]
    return prior[-ndays:] if len(prior) >= ndays else None


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
    # ---- pre-pass: per-name-day (median trade size, opening-spread median) ----
    # opening spread = median (ba-bb) over the first OPEN_MIN of continuous phase
    print(f"pre-pass: median trade size + opening spread, {len(all_dates)} days",
          flush=True)
    tsize = {s: {} for s in NAMES}
    ospr = {s: {} for s in NAMES}
    t0 = time.perf_counter()
    for i, date in enumerate(all_dates, 1):
        dsets = R.open_datasets(date)
        if dsets is None or str(date) not in segments:
            continue
        seg0 = segments[str(date)][0][0]
        f_end = seg0 + OPEN_MIN * 60000
        for s in NAMES:
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, s)
            if len(t):
                tsize[s][str(date)] = float(t["qty"].median())
            # opening spread from snapshots in the first 15 min, continuous phase
            snp = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, s)
            if len(snp):
                c = snp[snp["phase"] == "CONTINUOUS_AUCTION"].copy()
                if len(c):
                    c["ts"] = R.to_ms(c["orig_time"])
                    c = c[c["ts"] <= f_end]
                    bids = c[c["entry_type"] == "BID"]
                    asks = c[c["entry_type"] == "OFFER"]
                    if len(bids) and len(asks):
                        bb = bids.groupby("msg_seq")["px"].max()
                        ba = asks.groupby("msg_seq")["px"].min()
                        j = pd.concat([bb.rename("bb"), ba.rename("ba")], axis=1).dropna()
                        j = j[j["ba"] >= j["bb"]]
                        if len(j):
                            ospr[s][str(date)] = float((j["ba"] - j["bb"]).median())
        if i % 50 == 0 or i == len(all_dates):
            print(f"  pre-pass {i}/{len(all_dates)}  "
                  f"elapsed {C._fmt(time.perf_counter() - t0)}", flush=True)

    run_dates = all_dates[max(TRAIL_DAYS, REF_TRAIL):]
    # configs: baseline (off) + grid
    CONFIGS = [("OFF", None)]
    for w in WAIT_S:
        for k in K_SD:
            CONFIGS.append((f"w{w:g}|k{k:g}", (w, k)))
    total = len(run_dates) * len(NAMES)
    print(f"\nsweep_open_suppress: {len(CONFIGS)} configs x {len(NAMES)} names x "
          f"{len(run_dates)} days @ {CLIP_MULT:g}x\n", flush=True)

    acc = {}
    for s in NAMES:
        for lbl, _ in CONFIGS:
            acc[(s, lbl)] = {"pnl": 0.0, "days": 0, "fills": 0, "suppressed": 0}
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
            tw = trailing_med(tsize[sym], all_dates, date, TRAIL_DAYS)
            if tw is None:
                continue
            med_qty = float(np.median(tw))
            if med_qty <= 0:
                continue
            # trailing opening-spread reference (median + sd over prior days)
            ow = trailing_med(ospr[sym], all_dates, date, REF_TRAIL)
            ref_med = float(np.median(ow)) if ow else 0.0
            ref_sd = float(np.std(ow)) if ow else 0.0
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
            open_end = t0_ + OPEN_MIN * 60000
            clip = max(1, int(round(CLIP_MULT * med_qty)))
            for lbl, gate in CONFIGS:
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
                if gate is not None:
                    w, k = gate
                    params["open_supp_on"] = True
                    params["open_supp_wait_s"] = w
                    params["open_supp_ref_med"] = ref_med
                    params["open_supp_ref_sd"] = ref_sd
                    params["open_supp_k"] = k
                cfg = dict(R.CFG, session=(t0_, t1_),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
                bt = Backtester(strat, cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                nf, cap_b, mk_b, net_b = C.score_bps(fills, fs_day)
                liq = bt.eod["equity_liquidated"] if bt.eod is not None else None
                # decomposition: fills in the open window (cost proxy) -- count and
                # their net P&L contribution via score on the open subset
                f = fills if isinstance(fills, pd.DataFrame) else pd.DataFrame(fills)
                open_fills = 0
                if len(f) and "t" in f.columns:
                    open_fills = int((f["t"] <= open_end).sum())
                a = acc[(sym, lbl)]
                if liq is not None:
                    a["pnl"] += float(liq)
                    a["days"] += 1
                a["fills"] += nf
                a["suppressed"] += int(strat.stats.get("open_suppressed", 0))
                rows.append({"date": str(date), "symbol": sym, "config": lbl,
                             "pnl_pkr": (round(float(liq), 2) if liq is not None else np.nan),
                             "fills": nf, "open_fills": open_fills,
                             "ref_med": round(ref_med, 3),
                             "net_bps": (round(net_b, 3) if isinstance(net_b, float) else np.nan)})
            if sd % 10 == 0 or sd == total:
                el = time.perf_counter() - t0_all
                print(f"  {sd}/{total} symbol-days  elapsed {C._fmt(el)}  "
                      f"ETA {C._fmt(el / sd * (total - sd))}", flush=True)

    daily = RESULTS / f"open_suppress_daily_{stamp}.csv"
    dfall = pd.DataFrame(rows)
    dfall.to_csv(daily, index=False)

    # ---- verdict: each config vs OFF baseline, with the open-window decomposition
    base = {s: acc[(s, "OFF")]["pnl"] for s in NAMES}
    base_tot = sum(base.values())
    print(f"\n=== OPEN SUPPRESSION vs BASELINE (gate off) ===")
    print(f"baseline total P&L: {base_tot:,.0f}\n")
    print(f"{'config':12s} {'total_pnl':>12s} {'dPnL':>10s} {'suppress_ev':>11s} "
          f"{'open_fills':>10s}")
    # baseline open-window fills (the exposure the gate removes)
    base_openf = dfall[dfall.config == "OFF"]["open_fills"].sum()
    summ = []
    for lbl, _ in CONFIGS:
        if lbl == "OFF":
            continue
        tot = sum(acc[(s, lbl)]["pnl"] for s in NAMES)
        supp = sum(acc[(s, lbl)]["suppressed"] for s in NAMES)
        of = dfall[dfall.config == lbl]["open_fills"].sum()
        print(f"{lbl:12s} {tot:>12,.0f} {tot-base_tot:>+10,.0f} {supp:>11,} "
              f"{of:>10,}")
        summ.append({"config": lbl, "total_pnl": tot, "dpnl": tot - base_tot,
                     "suppress_events": supp, "open_fills": of,
                     "baseline_open_fills": base_openf})
    S = pd.DataFrame(summ).sort_values("dpnl", ascending=False)
    S.to_csv(RESULTS / f"open_suppress_summary_{stamp}.csv", index=False)
    best = S.iloc[0]
    print(f"\nBEST: {best['config']}  dPnL {best['dpnl']:+,.0f}")
    if best["dpnl"] <= 0:
        print(">>> NO config beats baseline: sitting out the open FORGOES more calm-")
        print(">>> morning spread than it saves in gap losses (in-backtest). Note the")
        print(">>> clean-EOD backtest may understate live gap protection.")
    else:
        print(">>> A config helps. CHECK the daily CSV: is the gain from a few gap")
        print(">>> mornings (real) or broad across all days (suspect, like the")
        print(">>> reactive gate)? Concentration in high-vol days = real protection.")
    print(f"\nwrote {RESULTS / f'open_suppress_summary_{stamp}.csv'}\nwrote {daily}")


if __name__ == "__main__":
    main()
