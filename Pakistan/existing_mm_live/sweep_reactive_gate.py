# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# sweep_reactive_gate.py -- DOES THE REACTIVE JUMP GATE ADD VALUE?
#
# PSX jumps have NO book precursor (validated), so we cannot predict them. This
# tests the REACTIVE alternative: after a large move over a short lookback, go
# dark for a cooldown to avoid the compounding 2nd/3rd toxic fill. The honest
# trap (measured, not assumed): post-jump mean-reversion is PROFITABLE for a MM,
# so a gate can HELP (skips toxic trend-continuation fills) or HURT (skips the
# reversion fills that make money). Only the with-vs-without P&L tells us which.
#
# Baseline  : the locked PRODUCTION strategy (NEW 4-bucket POV unwind), gate OFF.
# Treatment : same strategy + reactive gate, over a 2D grid:
#               mode     in {symmetric, inventory}
#               k        in {2, 3, 4}      (trigger = k * trailing-sigma move)
#               cooldown in {30,60,120,300}s
#             = 1 (off) + 2*3*4 = 25 configs per symbol-day.
#
# SPLIT RUN (SZ, 2026-08-21): REJECTS FIRST (5 names, ~15h) -- where jumps do the
# damage and the gate SHOULD help. If it adds nothing even here, it is dead and
# we skip the 30h winners run. Fixed clip = 3x median (the production knee), so
# the gate effect is isolated from sizing.
#
# Reads the newest calibration CSVs. Writes timestamped daily + summary.
# Run from existing_mm_live/:  python sweep_reactive_gate.py

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
# reuse the scorer + formatter
import confirm_micro_vs_naive as C

# raw store + results
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# Resolve this filesystem path through the canonical checkout/data configuration.
FS_ROOT = Path(str(_hft_paths.RESULTS_ROOT / 'feature_store'))
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS = Path(str(_hft_paths.RESULTS_ROOT))

# ------------------------------ experiment knobs ------------------------------
# SPLIT RUN part 1: the reject/blowup names (jumps hurt these most)
NAMES = ["FNEL", "PACE", "AKBL", "TPL", "HASCOL",
         "THCCL", "TOMCL", "TRG", "BOP", "NCPL"]
# fixed clip = the production knee (isolate the gate from sizing)
CLIP_MULT = 3.0
# inventory limits in clips (production ratios)
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
# POV unwind participation cap
MAX_POV = 0.10
# trailing window for the median clip (walk-forward)
TRAIL_DAYS = 10
# ---- the 2D reactive grid ----
# trigger modes
GATE_MODES = ["symmetric", "inventory"]
# trigger size (multiples of trailing-sigma move over the lookback)
GATE_K = [2.0, 3.0, 4.0]
# stay-dark durations (seconds)
GATE_COOLDOWN = [30.0, 60.0, 120.0, 300.0]
# the move lookback (seconds) -- fixed (a jump is fast); could be swept later
GATE_LOOKBACK = 10.0
# locked production MID base (NEW unwind added per-name below)
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_eod_trigger=True, enable_lock_trigger=True)
# ------------------------------------------------------------------------------


# newest file matching a pattern
def newest(pattern):
    cands = sorted(RESULTS.glob(pattern))
    if not cands:
        raise SystemExit(f"missing {pattern} -- run its calibration script first")
    return cands[-1]


# {symbol: value-tuple} from a calibration CSV, ok rows only
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


# trailing median trade size strictly before a date
def trailing(stats_sym, all_dates, date):
    prior = [stats_sym[str(d)] for d in all_dates
             if str(d) < str(date) and str(d) in stats_sym]
    if len(prior) < TRAIL_DAYS:
        return None
    return float(np.median(prior[-TRAIL_DAYS:]))


# build the list of (label, gate-kwargs) configs: baseline + the 2D grid
def build_configs():
    # baseline: gate off
    cfgs = [("OFF|k0|c0", dict(reactive_mode="off"))]
    # the grid
    for mode in GATE_MODES:
        for k in GATE_K:
            for cd in GATE_COOLDOWN:
                label = f"{mode}|k{k:g}|c{cd:g}"
                cfgs.append((label, dict(reactive_mode=mode, reactive_k=k,
                                         reactive_cooldown_s=cd,
                                         reactive_lookback_s=GATE_LOOKBACK)))
    return cfgs


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # calibration tables
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
            raise SystemExit(f"{sym} missing scale or volume profile -- calibrate first")

    # the config grid
    CONFIGS = build_configs()
    # dates + trades pre-pass (median clip anchor)
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
            if len(t) == 0:
                continue
            tstats[sym][str(date)] = float(t["qty"].median())
        if i % 50 == 0 or i == len(all_dates):
            print(f"  pre-pass {i}/{len(all_dates)}  "
                  f"elapsed {C._fmt(time.perf_counter() - t0)}", flush=True)

    run_dates = all_dates[TRAIL_DAYS:]
    total = len(run_dates) * len(NAMES)
    print(f"\nsweep_reactive_gate: {len(CONFIGS)} configs (1 off + "
          f"{len(GATE_MODES)}x{len(GATE_K)}x{len(GATE_COOLDOWN)} grid) x "
          f"{len(NAMES)} names x {len(run_dates)} days @ {CLIP_MULT:g}x clip\n",
          flush=True)

    # accumulators keyed (sym, config_label)
    acc = {}
    for sym in NAMES:
        for label, _ in CONFIGS:
            acc[(sym, label)] = {"pnl": 0.0, "fills": 0, "days": 0,
                                 "net": [], "unclean": 0, "darkened": 0}
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
            # events ONCE per symbol-day (shared by all configs)
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
            # fixed clip for this name-day
            clip = max(1, int(round(CLIP_MULT * med_qty)))
            # INNER: each gate config on the SAME event stream + SAME production base
            for label, gate_kw in CONFIGS:
                params = dict(MID_BASE)
                params["size"] = clip
                params["max_inv"] = int(round(MAXINV_CLIPS * clip))
                params["soft_inv"] = int(round(SOFTINV_CLIPS * clip))
                params["session_scale"] = scales[sym]
                rmp, clf = windows.get(sym, (5.0, 1.0))
                params["eod_ramp_start_min"] = rmp
                params["eod_cliff_min"] = clf
                # PRODUCTION unwind is ALWAYS on (this is the locked strategy)
                params["unwind_profile"] = profiles[sym]
                params["unwind_pov"] = MAX_POV
                params["session_segments"] = segs
                # the gate config (off for baseline)
                params.update(gate_kw)
                cfg = dict(R.CFG, session=(t0_, t1_),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
                bt = Backtester(strat, cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                nf, cap_b, mk_b, net_b = C.score_bps(fills, fs_day)
                liq = bt.eod["equity_liquidated"] if bt.eod is not None else None
                dark = strat.stats.get("reactive_darkened", 0)
                a = acc[(sym, label)]
                if liq is not None:
                    a["pnl"] += float(liq)
                    a["days"] += 1
                a["fills"] += nf
                a["darkened"] += int(dark)
                if isinstance(net_b, (float, np.floating)) and not np.isnan(net_b):
                    a["net"].append(float(net_b))
                if bt.eod is not None and bt.eod["liquidation_clean"] is False:
                    a["unclean"] += 1
                rows.append({"date": str(date), "symbol": sym, "config": label,
                             "clip_sh": clip, "fills": nf,
                             "pnl_pkr": (round(float(liq), 2) if liq is not None else np.nan),
                             "darkened": int(dark),
                             "net_bps": (round(net_b, 3) if isinstance(net_b, float) else np.nan)})
            if sd % 10 == 0 or sd == total:
                el = time.perf_counter() - t0_all
                eta = el / sd * (total - sd)
                print(f"  {sd}/{total} symbol-days  elapsed {C._fmt(el)}  "
                      f"ETA {C._fmt(eta)}", flush=True)

    # ---- outputs ----
    daily_csv = RESULTS / f"reactive_gate_daily_{stamp}.csv"
    pd.DataFrame(rows).to_csv(daily_csv, index=False)

    # ---- verdict: each config's total P&L vs the OFF baseline, per name + total ----
    base_label = "OFF|k0|c0"
    print(f"\n=== REACTIVE GATE vs BASELINE (gate off), total PKL over the run ===")
    print("(delta = gate_pnl - baseline_pnl; POSITIVE = the gate ADDS value)\n")
    # per-name baseline
    base = {sym: acc[(sym, base_label)]["pnl"] for sym in NAMES}
    # summary rows
    summ = []
    for label, _ in CONFIGS:
        if label == base_label:
            continue
        tot_delta = 0.0
        tot_dark = 0
        for sym in NAMES:
            d = acc[(sym, label)]["pnl"] - base[sym]
            tot_delta += d
            tot_dark += acc[(sym, label)]["darkened"]
        summ.append({"config": label, "total_delta": tot_delta,
                     "total_darkened": tot_dark})
    S = pd.DataFrame(summ).sort_values("total_delta", ascending=False)
    print(f"{'config':22s} {'total_delta':>13s} {'darken_events':>14s}")
    for _, r in S.iterrows():
        print(f"{r['config']:22s} {r['total_delta']:>+13,.0f} {int(r['total_darkened']):>14,}")
    # the headline
    best = S.iloc[0]
    print(f"\nBASELINE total P&L (gate off): {sum(base.values()):>+12,.0f}")
    print(f"BEST config: {best['config']}  delta {best['total_delta']:+,.0f}")
    if best["total_delta"] <= 0:
        print("\n>>> NO gate config beats baseline. The reactive gate does NOT add")
        print(">>> value on these names in-backtest (reversion capture > jump")
        print(">>> protection). Note: the clean-EOD backtest may UNDERSTATE live")
        print(">>> value, but there is no in-sample case to deploy it here.")
    else:
        print(f"\n>>> {best['config']} adds {best['total_delta']:+,.0f}. Inspect whether")
        print(">>> it concentrates in the true blowup names (real) or is broad (suspect).")
    rank_csv = RESULTS / f"reactive_gate_summary_{stamp}.csv"
    S.to_csv(rank_csv, index=False)
    print(f"\nwrote {rank_csv}\nwrote {daily_csv}")


if __name__ == "__main__":
    main()
