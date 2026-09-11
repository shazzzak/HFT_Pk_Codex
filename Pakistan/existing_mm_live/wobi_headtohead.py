# ============================================================================
# wobi_headtohead.py -- P&L HEAD-TO-HEAD: plain L1 OBI vs decay-weighted w_d2_r0.5
# ----------------------------------------------------------------------------
# The obi_decay_gate SCREEN said w_d2_r0.5 has the largest INCREMENTAL partial-
# Spearman over L1 (+0.023) but ~zero standalone advantage (vs_L1 +0.003). Per the
# project rule "do not close on the correlation metric alone," this settles it in
# money: run the production strategy TWICE across the universe -- once with plain L1
# (control), once with the weighted signal swapped in (treatment) -- and compare
# net bps / Sharpe paired by day. Everything else (OBI throttle, thresholds, clip,
# sizing, session) is held IDENTICAL; only the imbalance signal changes.
#
# CRITICAL PLUMBING (see mm_backtest.py:1209-1210): the engine passes ranked depth
# into quotes() ONLY when ofi_depth_levels > 1, and returns exactly that many levels.
# weighted_obi needs >= wobi_depth levels, so the treatment MUST set
# ofi_depth_levels >= wobi_depth or the weighted signal silently collapses to L1 and
# the whole test becomes L1-vs-L1. validate_arms() asserts this BEFORE the run.
#
# REQUIRES the micro_mm.py weighted_obi edits (weighted_obi / wobi_depth / wobi_decay
# kwargs + _weighted_imb). Without them MicrostructureMM(**params) raises TypeError
# loudly -- it will NOT silently pass.
#
# USAGE: python wobi_headtohead.py --self-test | --smoke | --run [--days N] [--workers K]
# ============================================================================

# CLI parsing
import argparse
# wall-clock timing / ETA
import time
# log timestamps
from datetime import datetime
# process pool
from multiprocessing import Pool
# numerics / frames
import numpy as np
import pandas as pd

# central paths (single source of truth)
from config_pk import PARSED_ROOT, RESULTS_ROOT


# log stamp helper
def _ts():
    # HH:MM:SS in brackets
    return datetime.now().strftime("[%H:%M:%S]")


# output directory for the head-to-head artifacts
OUT_DIR = RESULTS_ROOT / "diagnostics"
# the production tradeable universe (same watchlist the screen used)
WATCHLIST = RESULTS_ROOT / "mm_watchlist_final.csv"
# trailing days for the median trade size (clip sizing) -- production value
TRAIL_DAYS = 10
# clip multiplier -- production value
CLIP_MULT = 3.0
# sampled days (mirror the screen's 20-day sample; raise for the full-year confirm)
MAX_DAYS = 20
# worker processes
WORKERS = 9

# ---- PRODUCTION BASELINE held identical across BOTH arms (the OBI throttle) ----
# these are the confirmed-edge production knobs; neither arm changes them
BASELINE = dict(
    # OBI throttle ON (the production baseline edge)
    obi_throttle=True,
    # OFI throttle OFF
    ofi_throttle=False,
    # throttle engage threshold on |imb-0.5|
    obi_throttle_thresh=0.15,
    # throttled-side clip fraction
    throttle_frac=0.5,
    # throttle hold window (ms)
    throttle_hold_ms=300.0,
)

# ---- THE ARMS: only the imbalance signal differs across arms ----
# control = plain L1 imbalance (weighted_obi off); treatments = decay-weighted variants.
# NOTE ofi_depth_levels=2 in each weighted arm: REQUIRED so the engine actually supplies
# 2 ranked levels to quotes() (see the plumbing note above) or the signal collapses to L1.
ARMS = {
    # CONTROL: exact production L1 signal
    "L1": dict(weighted_obi=False),
    # TREATMENT A: L1 + 0.3*L2 (more L1-dominant; screen's overall-gap pick)
    "wobi_d2r0.3": dict(weighted_obi=True, wobi_depth=2, wobi_decay=0.3,
                        ofi_depth_levels=2),
    # TREATMENT B: L1 + 0.5*L2 (screen's partial-rho pick)
    "wobi_d2r0.5": dict(weighted_obi=True, wobi_depth=2, wobi_decay=0.5,
                        ofi_depth_levels=2),
}
# the control label used for the paired comparison
CONTROL = "L1"


def validate_arms(arms):
    # HARD pre-run guard against the silent-null trap: any weighted arm MUST request
    # at least wobi_depth ranked levels via ofi_depth_levels, else weighted_obi
    # collapses to L1 and the test is meaningless. Crash BEFORE spawning workers.
    for label, ov in arms.items():
        # only weighted arms need the depth plumbing
        if ov.get("weighted_obi"):
            # required depth for the weighting
            wd = int(ov.get("wobi_depth", 1))
            # levels the engine will actually supply
            odl = int(ov.get("ofi_depth_levels", 1))
            # assert enough levels, with an actionable message
            assert odl >= wd, (
                f"ARM '{label}' misconfigured: ofi_depth_levels={odl} < wobi_depth={wd}. "
                f"The engine (mm_backtest.py:1209) would supply < {wd} levels and the "
                f"weighted signal would silently collapse to L1. Set ofi_depth_levels>={wd}."
            )
    # visible confirmation in the log
    print(_ts() + f"validate_arms OK: {list(arms)} (control={CONTROL})")


# ---- net P&L + opened notional for one day via the VALIDATED harness FIFO ----
def _attr(dr, H):
    # the raw fills for the day
    f = dr.fills
    # no fills at all -> no intraday notional
    if f is None or len(f) == 0:
        # net is still the engine's headline (may be liquidation-only), notional 0
        return float(dr.pnl()), 0.0
    # EXCLUDE EOD liquidation fills (reason 'liq'/'liq_residual') -- feeding them into
    # fifo_attribution double-handles the haircut (the documented 1-2 PKR leak)
    if "reason" in f.columns:
        # keep only non-liq fills
        ff = f[~f["reason"].astype(str).str.startswith("liq")].copy()
    else:
        # no reason column -> use all fills
        ff = f.copy()
    # no intraday fills after excluding liq -> notional 0
    if len(ff) == 0:
        return float(dr.pnl()), 0.0
    # the engine's ground-truth day P&L (guaranteed non-None by the caller's guard)
    engine_pnl = float(dr.pnl())
    # residual lots are held to session end
    liq_t = float(ff["t"].max())
    # the validated attribution (reconciles to engine_pnl by construction)
    per = H.fifo_attribution(ff, engine_pnl, liq_t)
    # total opened notional across all session buckets (the bps denominator)
    onot = sum(float(v.get("opened_notional", 0.0)) for v in per.values())
    # net P&L is the reconciled engine headline; notional is the opened total
    return engine_pnl, onot


# per-process globals
_R = None; _H = None; _CALIB = None
# pool initializer: import the driver + harness once per worker
def _init(calib):
    # module-level handles
    global _R, _H, _CALIB
    # driver (loading, build_events, discover_dates) + harness (run_symbol_day, FIFO)
    import run_legacy_mm as R, mm_harness as H
    # point the driver at the local parsed store
    R.PARSED_ROOT = PARSED_ROOT
    # use the micro strategy path
    R.USE_MICRO = True
    # stash for the workers
    _R = R; _H = H; _CALIB = calib


# run ONE (arm, symbol, day) and return a single per-name-day row
def _one(arm_label, overrides, date, sym, dsets):
    # calibration bundle
    C = _CALIB
    # need per-symbol scale + volume profile
    if sym not in C["scales"] or sym not in C["profiles"]:
        # not calibrated -> skip
        return None
    # session segments for this date
    segs = C["segments"].get(str(date))
    # no segments -> skip
    if segs is None:
        return None
    # trailing median trade size -> clip
    med = _H.trailing_median(C["tstats"][sym], C["all_dates"], date, TRAIL_DAYS)
    # unusable median -> skip
    if med is None or med <= 0:
        return None
    # integer clip (>=1)
    clip = max(1, int(round(CLIP_MULT * med)))
    # assemble the production params for this symbol-day
    params = _H.build_micro_params(clip, C["scales"][sym], C["profiles"][sym],
                                   C["windows"].get(sym, (5.0, 1.0)), segs)
    # apply the shared production baseline (OBI throttle etc.)
    params.update(BASELINE)
    # apply THIS arm's signal overrides (L1 vs weighted)
    params.update(overrides)
    # PER-CALL PLUMBING GUARD (defense in depth; validate_arms already ran pre-pool):
    # a weighted arm without enough supplied levels would silently test L1.
    if params.get("weighted_obi") and int(params.get("ofi_depth_levels", 1)) < int(params.get("wobi_depth", 1)):
        # fail loudly rather than emit a meaningless row
        raise ValueError(f"{arm_label}: ofi_depth_levels < wobi_depth -> weighted signal would collapse to L1")
    # run the single canonical backtest path
    dr = _H.run_symbol_day(date, sym, dsets, params)
    # skip unpriceable days: run failed OR no clean close (dr.pnl() is None)
    if dr is None or dr.pnl() is None:
        # visible skip so coverage loss is not silent
        print(_ts() + f"SKIP {arm_label} {date} {sym}: unpriceable day")
        return None
    # net P&L + opened notional for the bps denominator
    net_pkr, onot = _attr(dr, _H)
    # one tidy row (config column named 'throttle' so sweep_risk_score.cfg_col finds it)
    return dict(throttle=arm_label, symbol=sym, date=str(date),
                net_pkr=net_pkr, opened_notional=onot, bucket="all")


# work one date across BOTH arms and ALL symbols
def _work_date(date):
    # open the day's datasets once
    dsets = _R.open_datasets(date)
    # missing day -> nothing
    if dsets is None:
        return []
    # accumulate rows for this date
    rows = []
    # each arm
    for arm_label, overrides in ARMS.items():
        # each symbol in the universe
        for sym in _CALIB["universe"]:
            # guard per (arm, symbol): a genuine data error should skip, not kill the run
            try:
                # compute the row
                r = _one(arm_label, overrides, date, sym, dsets)
            except Exception as e:
                # log and continue (matches the house SKIP pattern)
                print(_ts() + f"SKIP {arm_label} {date} {sym}: {e!r}")
                continue
            # keep non-empty rows
            if r is not None:
                rows.append(r)
    # this date's rows
    return rows


# build the calibration bundle for the whole universe
def _calib():
    # driver + harness
    import run_legacy_mm as R, mm_harness as H
    # local store
    R.PARSED_ROOT = PARSED_ROOT
    # all tradeable dates
    all_dates = R.discover_dates()
    # the production universe from the watchlist
    universe = sorted(pd.read_csv(WATCHLIST)["symbol"].dropna().astype(str).unique().tolist())
    # calibration loaders (same single-source-of-truth as every other runner)
    C = dict(scales=H.load_scales(), profiles=H.load_profiles(),
             windows=H.load_windows(), segments=H.load_segments(),
             all_dates=all_dates, universe=universe,
             tstats=H.trailing_median_trade_size(all_dates, universe, TRAIL_DAYS))
    # bundle + the date list
    return C, all_dates


# ---- inline paired day-as-unit comparison (so you don't strictly need the scorer) ----
def _report(df):
    # daily portfolio P&L + notional per arm (sum across symbols within a day)
    g = (df.groupby(["throttle", "date"])
           .agg(pkr=("net_pkr", "sum"), opn=("opened_notional", "sum")).reset_index())
    # daily net bps (scale-free MM return series)
    g["bps"] = np.where(g["opn"] > 0, g["pkr"] / g["opn"] * 1e4, np.nan)
    # wide day x arm matrices
    w_bps = g.pivot(index="date", columns="throttle", values="bps")
    w_pkr = g.pivot(index="date", columns="throttle", values="pkr")
    # header
    print(_ts() + "===== L1 vs w_d2_r0.5 : P&L HEAD-TO-HEAD (day-as-unit) =====")
    print(_ts() + f"  {'arm':>12} {'net_PKR':>12} {'mean_bps':>9} {'Sharpe~':>8} {'win%':>5} {'n-days':>7}")
    # per-arm summary (Sharpe here is a quick 252-annualized proxy; use sweep_risk_score for the full set)
    for arm in w_bps.columns:
        # daily bps / pkr series
        d = w_bps[arm].to_numpy(); p = w_pkr[arm].fillna(0).to_numpy()
        # finite bps only
        dd = d[~np.isnan(d)]
        # quick annualized Sharpe proxy
        sh = (dd.mean() / dd.std(ddof=1) * np.sqrt(252.0)) if len(dd) >= 3 and dd.std(ddof=1) > 0 else float("nan")
        # print the arm line
        print(_ts() + f"  {arm:>12} {np.nansum(p):>12,.0f} {np.nanmean(d):>9.3f} {sh:>8.2f} "
                      f"{np.mean(p > 0)*100:>4.0f}% {len(dd):>7}")
    # PAIRED delta (treatment - control) on the SAME days -- the decisive number
    if CONTROL in w_bps.columns:
        # each non-control arm vs control
        for arm in [c for c in w_bps.columns if c != CONTROL]:
            # paired daily bps difference on days both arms have
            delta = (w_bps[arm] - w_bps[CONTROL]).dropna().to_numpy()
            # need a couple of days for a t-stat
            if len(delta) >= 3:
                # paired mean and t-stat (day-as-unit)
                m = delta.mean(); se = delta.std(ddof=1) / np.sqrt(len(delta)); t = m / se if se > 0 else float("nan")
                # the verdict line
                print(_ts() + f"  PAIRED {arm} - {CONTROL}: {m:+.3f} bps/day  (t={t:+.2f}, n={len(delta)})")
                # plain-language read
                verdict = ("treatment WINS" if (m > 0 and t > 2) else
                           "treatment LOSES" if (m < 0 and t < -2) else
                           "FLAT (within noise) -> keep L1")
                print(_ts() + f"         READ: {verdict}")
    # remind them of the fuller scorer
    print(_ts() + "  (full Sharpe/Sortino/maxDD: run sweep_risk_score.py on the CSV, --control L1 --bucket all)")


# full run
def run_real(out_dir=OUT_DIR, workers=WORKERS, max_days=MAX_DAYS):
    # FAIL FAST on any misconfigured arm before doing expensive work
    validate_arms(ARMS)
    # calibration pre-pass
    print(_ts() + "pre-pass: calibration")
    # build the bundle
    calib, all_dates = _calib()
    # drop the warmup window, then subsample days like the screen
    dates = all_dates[TRAIL_DAYS:]
    # even subsample down to max_days
    if max_days and len(dates) > max_days:
        step = max(1, len(dates) // max_days); dates = dates[::step][:max_days]
    # announce the plan
    print(_ts() + f"{len(calib['universe'])} names x {len(dates)} dates x {len(ARMS)} arms, {workers} workers")
    # collect rows + start the clock
    rows = []; t0 = time.perf_counter()
    # process pool over dates
    with Pool(processes=workers, initializer=_init, initargs=(calib,)) as pool:
        # completed counter
        done = 0
        # stream results with progress + ETA
        for res in pool.imap_unordered(_work_date, dates):
            # gather
            rows.extend(res); done += 1
            # elapsed minutes
            el = (time.perf_counter() - t0) / 60.0
            # heartbeat with ETA
            print(_ts() + f"  date {done}/{len(dates)} ({el:.1f} min, ETA {el/done*(len(dates)-done):.1f} min)")
    # nothing produced
    if not rows:
        print(_ts() + "no rows."); return
    # assemble the per-name-day frame
    df = pd.DataFrame(rows)
    # ensure output dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # write the artifact of record as verified parquet
    _safe_parquet(df, out_dir / "wobi_headtohead.parquet")
    # ALSO write the CSV that sweep_risk_score.py consumes (it uses pd.read_csv)
    csv_path = out_dir / "wobi_headtohead.csv"
    # plain CSV for the standing scorer
    df.to_csv(csv_path, index=False)
    # note where it went
    print(_ts() + f"wrote {csv_path}")
    # inline paired verdict
    _report(df)
    # final pointer
    print(_ts() + f"[wobi-headtohead] outputs -> {out_dir}")


# safe atomic+verified parquet (same contract as the sweeps)
def _safe_parquet(df, path):
    # local os
    import os as _os
    # temp path
    tmp = str(path) + ".tmp"
    try:
        # write temp
        df.to_parquet(tmp, index=False)
        # verify row count on readback
        import pyarrow.parquet as _pq
        # row-count assertion
        assert _pq.ParquetFile(tmp).metadata.num_rows == len(df)
        # atomic move into place
        _os.replace(tmp, path)
        # log success
        print(_ts() + f"wrote {path} ({len(df)} rows, verified)")
    except Exception as e:
        # clean up temp on failure
        try: _os.remove(tmp)
        except OSError: pass
        # CSV fallback (data is fine, just relabeled)
        csv = str(path).rsplit(".", 1)[0] + ".csv"
        # write fallback
        df.to_csv(csv, index=False)
        # log the fallback
        print(_ts() + f"parquet failed ({e!r}) -> CSV {csv}")


# time one symbol-day per arm (fast wiring check on real data)
def smoke():
    # validate arms first
    validate_arms(ARMS)
    # driver + harness
    import run_legacy_mm as R, mm_harness as H
    # local store
    R.PARSED_ROOT = PARSED_ROOT
    # calibration
    calib, all_dates = _calib()
    # set worker globals for a direct (non-pool) call
    global _R, _H, _CALIB; _R, _H, _CALIB = R, H, calib
    # pick a mid-sample date
    date = all_dates[len(all_dates) // 2]
    # open datasets
    dsets = R.open_datasets(date)
    # first calibrated universe name
    sym = calib["universe"][0]
    # run both arms on this one symbol-day
    for arm_label, overrides in ARMS.items():
        # time it
        t0 = time.perf_counter()
        # single row
        r = _one(arm_label, overrides, date, sym, dsets)
        # report
        print(_ts() + f"[smoke] {arm_label} {date} {sym} in {time.perf_counter()-t0:.1f}s: {r}")


# offline self-test: NO data, NO project deps -- proves the guard logic + arm config
def self_test():
    # the real arms must pass the plumbing validator
    validate_arms(ARMS)
    # the treatment must actually enable weighting AND supply enough levels
    tv = ARMS["wobi_d2r0.5"]
    # weighting on
    assert tv["weighted_obi"] is True
    # depth plumbing sufficient (the trap guard's core invariant)
    assert tv["ofi_depth_levels"] >= tv["wobi_depth"], "treatment would collapse to L1"
    # the control must be pure L1 (no weighting)
    assert ARMS[CONTROL].get("weighted_obi", False) is False
    # a DELIBERATELY BROKEN arm must be rejected by validate_arms
    bad = {"bad": dict(weighted_obi=True, wobi_depth=2, ofi_depth_levels=1)}
    # expect an AssertionError
    raised = False
    try:
        validate_arms(bad)
    except AssertionError:
        raised = True
    # the guard must have fired
    assert raised, "validate_arms failed to catch ofi_depth_levels < wobi_depth"
    # done
    print(_ts() + "[self-test] arm config + plumbing guard OK (broken arm correctly rejected).")
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


# entry point
if __name__ == "__main__":
    # parser
    ap = argparse.ArgumentParser()
    # modes
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    # knobs
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    # parse
    a = ap.parse_args()
    # smoke wiring check
    if a.smoke:
        smoke()
    # default to self-test when not running
    elif a.self_test or not a.run:
        self_test()
    # the full head-to-head
    if a.run:
        run_real(workers=a.workers, max_days=(a.days or None))
