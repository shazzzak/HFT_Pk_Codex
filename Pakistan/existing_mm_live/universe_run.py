# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# universe_run.py -- the universe screen: does MID (microprice off + both triggers)
# make markets on each of the 38 shortlist names, and how big is the per-fill edge?
# Runs TWO configs per name -- naive (dumb baseline) and MID -- at a 1x MEDIAN-TRADE-
# SIZE clip (walk-forward, per-name), across all 207 days, on the TREC MM fee tier.
#
# Reads calibrated per-name session_scale from the newest session_scales_*.csv
# (produced by calibrate_all_scales.py). Fails loud if a symbol has no scale.
#
# Outputs:
#   * a per-day CSV (timestamped) with one row per (date, symbol, config) -> daily
#     P&L, so edge-over-time trends are visible (rising / flat / eroding).
#   * a ranked per-name summary: MID net_bps, MID P&L, naive P&L, fills, participation,
#     unclean-liq days -- sorted by MID net_bps (the edge ranking).
#
# Run from existing_mm_live/:  python universe_run.py

# paths
from pathlib import Path
# timing + stamp
import time
from datetime import datetime
# frames + arrays
import pandas as pd
import numpy as np
# driver + engine + strategies
import run_legacy_mm as R
# NaiveSymmetricMM lives in mm_backtest (run_legacy_mm re-imports it from there)
from mm_backtest import Backtester, LatencyModel, NaiveSymmetricMM
from micro_mm import MicrostructureMM
# reuse confirm's Path-B scorer and _fmt (single source of truth)
import confirm_micro_vs_naive as C

# raw store
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# feature store (Path B context)
# Resolve this filesystem path through the canonical checkout/data configuration.
FS_ROOT = Path(str(_hft_paths.RESULTS_ROOT / 'feature_store'))
# shortlist
# Resolve this filesystem path through the canonical checkout/data configuration.
WATCHLIST = Path(str(_hft_paths.RESULTS_ROOT / 'mm_watchlist_final.csv'))
# results dir (outputs + the scales CSV live here)
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS = Path(str(_hft_paths.RESULTS_ROOT))

# trailing window (days) for the median trade size + ADV (walk-forward)
TRAIL_DAYS = 10
# the clip multiple of the trailing median trade size (1x = the universe screen size)
CLIP_MULT = 1.0
# inventory limits as multiples of the clip (same ratios as 500/50 and 150/50)
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
# the locked production micro config: MID (microprice off) + both triggers
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_eod_trigger=True, enable_lock_trigger=True)
# None = all days (after the trailing warm-up)
N_DAYS = None


# load the newest calibrated scale table -> {symbol: session_scale}
def load_scales():
    # every scales CSV, newest last (timestamp sorts chronologically)
    cands = sorted(RESULTS.glob("session_scales_*.csv"))
    # must have one
    if not cands:
        raise SystemExit("no session_scales_*.csv -- run calibrate_all_scales.py first")
    # read the newest
    df = pd.read_csv(cands[-1])
    # keep only successfully-calibrated names
    ok = df[df["note"] == "ok"] if "note" in df.columns else df
    # build the dict
    scales = dict(zip(ok["symbol"], ok["session_scale"]))
    # report
    print(f"loaded {len(scales)} calibrated scales from {cands[-1].name}")
    return scales


# load the newest POV time-window table -> {symbol: (ramp_min, cliff_min)}
def load_windows():
    # every windows CSV, newest last
    cands = sorted(RESULTS.glob("time_windows_*.csv"))
    # must have one (produced by calibrate_time_windows.py)
    if not cands:
        raise SystemExit("no time_windows_*.csv -- run calibrate_time_windows.py first")
    # read the newest
    df = pd.read_csv(cands[-1])
    # keep calibrated rows
    ok = df[df["note"] == "ok"] if "note" in df.columns else df
    # build the dict symbol -> (ramp, cliff)
    win = {r["symbol"]: (float(r["eod_ramp_start_min"]), float(r["eod_cliff_min"]))
           for _, r in ok.iterrows()}
    # report
    print(f"loaded {len(win)} POV time windows from {cands[-1].name}")
    return win


# PRE-PASS: daily median trade size (shares) + traded notional (PKR), per name.
# (Same logic as size_sweep; trades table uses 'price', not 'px'.)
def daily_trade_stats(dates, syms):
    stats = {sym: {} for sym in syms}
    t0 = time.perf_counter()
    for i, date in enumerate(dates, 1):
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        for sym in syms:
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0:
                continue
            med_qty = float(t["qty"].median())
            notional = float((t["price"] * t["qty"]).sum())
            stats[sym][str(date)] = (med_qty, notional)
        if i % 50 == 0 or i == len(dates):
            print(f"  pre-pass {i}/{len(dates)} days  "
                  f"elapsed {C._fmt(time.perf_counter() - t0)}", flush=True)
    return stats


# trailing median trade size + median daily notional, strictly BEFORE a date
def trailing(stats_sym, all_dates, date):
    prior = [stats_sym[str(d)] for d in all_dates
             if str(d) < str(date) and str(d) in stats_sym]
    if len(prior) < TRAIL_DAYS:
        return None, None
    win = prior[-TRAIL_DAYS:]
    return float(np.median([w[0] for w in win])), float(np.median([w[1] for w in win]))


def main():
    # run stamp for the output filenames
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # the 38 shortlist symbols (with tier, for the ranked report)
    wl = pd.read_csv(WATCHLIST)[["tier", "symbol"]]
    syms = wl["symbol"].tolist()
    tier_of = dict(zip(wl["symbol"], wl["tier"]))
    # calibrated per-name scales
    scales = load_scales()
    # POV-calibrated per-name unwind windows (ramp, cliff) in minutes
    windows = load_windows()
    # every name must have a scale (fail loud -- never run on a guess)
    missing = [s for s in syms if s not in scales]
    if missing:
        raise SystemExit(f"no calibrated scale for: {missing} -- re-run calibration")

    # dates
    all_dates = R.discover_dates()
    # PRE-PASS for the median clip + ADV
    print(f"pre-pass: daily median trade size + notional, {len(all_dates)} days "
          f"x {len(syms)} names", flush=True)
    tstats = daily_trade_stats(all_dates, syms)
    # run days: skip warm-up, then optionally stratify
    run_dates = all_dates[TRAIL_DAYS:]
    if N_DAYS is not None and N_DAYS < len(run_dates):
        idx = [round(i * (len(run_dates) - 1) / (N_DAYS - 1)) for i in range(N_DAYS)]
        run_dates = [run_dates[i] for i in idx]
    # the two configs
    CONFIGS = [("naive", None), ("MID", MID_BASE)]
    print(f"\nuniverse_run: {len(CONFIGS)} configs x {len(syms)} names x "
          f"{len(run_dates)} days ({run_dates[0]} .. {run_dates[-1]})\n", flush=True)

    # accumulators keyed (sym, config)
    acc = {}
    for sym in syms:
        for cfg_name, _ in CONFIGS:
            acc[(sym, cfg_name)] = {"pnl": 0.0, "fills": 0, "days": 0, "net": [],
                                    "part": [], "clip": [], "unclean": 0}
    # per-day rows
    rows = []
    # timers
    t0_all = time.perf_counter()
    sd = 0
    sd_total = len(run_dates) * len(syms)

    # OUTER: dates
    for date in run_dates:
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # MIDDLE: symbols
        for sym in syms:
            sd += 1
            # trailing clip + ADV (walk-forward)
            med_qty, adv = trailing(tstats[sym], all_dates, date)
            if med_qty is None or med_qty <= 0:
                continue
            # feature-store day for Path B
            fs_path = FS_ROOT / sym / f"date={date}.parquet"
            if not fs_path.exists():
                continue
            fs_day = pd.read_parquet(
                fs_path, columns=["ts_exch", "mid", "spread_bps", "obi_1",
                                  "toxicity", "realized_vol_bps"])
            # build events ONCE per symbol-day (shared by both configs)
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0 or len(s) == 0:
                continue
            events, snap_groups, t = R.build_events(u, s, t)
            cont_snap = s[s["phase"] == "CONTINUOUS_AUCTION"]
            if len(cont_snap) == 0:
                continue
            t0, t1 = int(cont_snap["ts_exch"].min()), int(cont_snap["ts_exch"].max())
            # the 1x-median clip and its scaled inventory limits
            clip = max(1, int(round(CLIP_MULT * med_qty)))
            # INNER: naive then MID on the same event stream
            for cfg_name, overrides in CONFIGS:
                # fresh engine config per run (seeded latency, corrected window)
                cfg = dict(R.CFG, session=(t0, t1),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                # build the strategy
                if cfg_name == "naive":
                    # dumb baseline: the stock's naive params (its own fixed size)
                    strat = NaiveSymmetricMM(**R.STRAT)
                else:
                    # MID: locked base + this name's clip + scaled limits + calib scale
                    params = dict(overrides)
                    params["size"] = clip
                    params["max_inv"] = int(round(MAXINV_CLIPS * clip))
                    params["soft_inv"] = int(round(SOFTINV_CLIPS * clip))
                    params["session_scale"] = scales[sym]
                    # per-name POV unwind windows (default 5/1 if a name is missing)
                    ramp_min, cliff_min = windows.get(sym, (5.0, 1.0))
                    params["eod_ramp_start_min"] = ramp_min
                    params["eod_cliff_min"] = cliff_min
                    strat = MicrostructureMM(session_ms=(t0, t1), **params)
                # run
                bt = Backtester(strat, cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                # Path B per-fill economics
                nf, cap_b, mk_b, net_b = C.score_bps(fills, fs_day)
                # participation = filled notional / trailing median daily value
                f = fills if isinstance(fills, pd.DataFrame) else pd.DataFrame(fills)
                if len(f) and "px" in f.columns and "qty" in f.columns:
                    fnot = float((f["px"] * f["qty"]).sum())
                else:
                    fnot = 0.0
                part = 100.0 * fnot / adv if adv and adv > 0 else np.nan
                # window-tagged fill counts (the "did we sell during the cliff?"
                # instrumentation): count fills per trigger window this day
                wcounts = {"none": 0, "time_ramp": 0, "time_cliff": 0,
                           "lock_ramp": 0, "lock_cliff": 0}
                if len(f) and "window" in f.columns:
                    for w, n in f["window"].value_counts().items():
                        if w in wcounts:
                            wcounts[w] = int(n)
                # Path A: the day's liquidated P&L + the liquidation split
                liq = bt.eod["equity_liquidated"] if bt.eod is not None else None
                # the split: what the mid-mark would have said, the residual mark,
                # the shares the book could not absorb, and the closing position
                eq_mid = bt.eod.get("equity_mid_mark") if bt.eod is not None else None
                resid = bt.eod.get("residual_marked", 0.0) if bt.eod is not None else None
                unf = bt.eod.get("unfilled_sh", 0.0) if bt.eod is not None else None
                pos_c = bt.eod.get("pos_at_close", 0.0) if bt.eod is not None else None
                # accumulate
                a = acc[(sym, cfg_name)]
                if liq is not None:
                    a["pnl"] += float(liq)
                    a["days"] += 1
                a["fills"] += nf
                if isinstance(net_b, (float, np.floating)) and not np.isnan(net_b):
                    a["net"].append(float(net_b))
                if not np.isnan(part):
                    a["part"].append(part)
                a["clip"].append(clip if cfg_name == "MID" else 50)
                if bt.eod is not None and bt.eod["liquidation_clean"] is False:
                    a["unclean"] += 1
                # per-day row
                rows.append({"date": str(date), "symbol": sym, "tier": tier_of[sym],
                             "config": cfg_name, "fills": nf,
                             "clip_sh": (clip if cfg_name == "MID" else 50),
                             "med_trade_sh": round(med_qty, 1),
                             "pnl_pkr": (round(float(liq), 2) if liq is not None else np.nan),
                             # what a mid-mark close would have reported (the gap to
                             # pnl_pkr IS the day's liquidation cost)
                             "pnl_mid_mark": (round(float(eq_mid), 2) if eq_mid is not None else np.nan),
                             # residual marked at mid-3% (only when the walk could not absorb)
                             "residual_marked": (round(float(resid), 2) if resid is not None else np.nan),
                             # shares the visible book could not absorb at the close
                             "unfilled_sh": (float(unf) if unf is not None else np.nan),
                             # signed position carried into the close
                             "pos_at_close": (float(pos_c) if pos_c is not None else np.nan),
                             # fills by trigger window: the cliff/ramp activity audit
                             "fills_time_ramp": wcounts["time_ramp"],
                             "fills_time_cliff": wcounts["time_cliff"],
                             "fills_lock_ramp": wcounts["lock_ramp"],
                             "fills_lock_cliff": wcounts["lock_cliff"],
                             "net_bps": (round(net_b, 3) if isinstance(net_b, float) else np.nan),
                             "participation_pct": (round(part, 4) if not np.isnan(part) else np.nan)})
            # heartbeat with ETA
            if sd % 25 == 0 or sd == sd_total:
                el = time.perf_counter() - t0_all
                eta = el / sd * (sd_total - sd)
                print(f"  {sd}/{sd_total} symbol-days  elapsed {C._fmt(el)}  "
                      f"ETA {C._fmt(eta)}", flush=True)

    # ---- per-day CSV (timestamped) ----
    daily_csv = RESULTS / f"universe_daily_{stamp}.csv"
    pd.DataFrame(rows).to_csv(daily_csv, index=False)

    # ---- ranked summary: one row per name, MID vs naive ----
    summ = []
    for sym in syms:
        m = acc[(sym, "MID")]
        n = acc[(sym, "naive")]
        # skip names that never ran
        if m["days"] == 0:
            continue
        summ.append({
            "tier": tier_of[sym], "symbol": sym,
            "mid_net_bps": (np.nanmean(m["net"]) if m["net"] else np.nan),
            "mid_pnl": m["pnl"], "naive_pnl": n["pnl"],
            "mid_beats_naive": m["pnl"] > n["pnl"],
            "mid_fills": m["fills"], "avg_clip": (np.mean(m["clip"]) if m["clip"] else np.nan),
            "participation_pct": (np.nanmean(m["part"]) if m["part"] else np.nan),
            "unclean_liq_days": m["unclean"], "days": m["days"]})
    S = pd.DataFrame(summ).sort_values("mid_net_bps", ascending=False)
    # write the ranked table too
    rank_csv = RESULTS / f"universe_ranked_{stamp}.csv"
    S.to_csv(rank_csv, index=False)

    # ---- print the ranked table ----
    print("\n=== UNIVERSE RANKING (by MID net_bps per fill; TREC MM fee tier) ===")
    print(f"{'tier':11s} {'sym':8s} {'net_bps':>8s} {'mid_pnl':>10s} {'naive_pnl':>10s} "
          f"{'beats':>5s} {'fills':>8s} {'clip':>6s} {'part%':>6s} {'unclean':>7s}")
    for _, r in S.iterrows():
        print(f"{r['tier']:11s} {r['symbol']:8s} {r['mid_net_bps']:>8.3f} "
              f"{r['mid_pnl']:>10,.0f} {r['naive_pnl']:>10,.0f} "
              f"{('YES' if r['mid_beats_naive'] else 'no'):>5s} {r['mid_fills']:>8,} "
              f"{r['avg_clip']:>6,.0f} {r['participation_pct']:>6.3f} "
              f"{int(r['unclean_liq_days']):>7d}")
    # headline counts
    pos = int((S["mid_net_bps"] > 0).sum())
    beat = int(S["mid_beats_naive"].sum())
    print(f"\n{len(S)} names run | {pos} with positive MID net_bps | "
          f"{beat} where MID beats naive")
    print(f"\nwrote {rank_csv}\nwrote {daily_csv}")
    print("\nREAD: positive net_bps = the quoting is profitable per fill (the real edge).")
    print("mid_pnl is total PKR over the period at a 1x-median clip -- small by design;")
    print("the RANK (net_bps) is what identifies which names to size up and pilot.")


if __name__ == "__main__":
    main()
