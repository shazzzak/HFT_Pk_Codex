# ============================================================================
# markout_horizon_diag.py
# ----------------------------------------------------------------------------
# MARKOUT-BY-HORIZON (from fill). For every real fill, the side-signed mid
# DRIFT from the fill instant to t+h, for h from 1s out past the mean hold:
#   markout_h = sign * (mid(t_fill + h) - mid(t_fill)) / mid(t_fill) * 1e4
# (sign = +1 buy, -1 sell; drift-only, matching the decomposition's split of
# capture vs markout). Read straight off dr.equity's mid path, so it needs NO
# feature-store rebuild and works at any horizon.
#
# WHY: the sweeps score adverse selection at 5s, but the middle-bucket mean hold
# is ~765s. This traces how the diffusion tax ACCRUES with holding time and
# where it plateaus (the "knee") -> decides whether flattening faster pays.
# Per bucket, per name, day-as-unit. Same calibrated winner runner as the sweeps.
#
# USAGE:
#   python markout_horizon_diag.py --self-test
#   python markout_horizon_diag.py --run            # 30 sampled days, 9 workers
# ============================================================================

# CLI, timing, paths.
import argparse
import time
from pathlib import Path
# Numerics + frames.
import numpy as np
import pandas as pd
# Headless plotting.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# stamp every printed line (standing convention).
from datetime import datetime
from multiprocessing import Pool
def _ts():
    return datetime.now().strftime("[%H:%M:%S] ")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Where the profile CSV + plots land.
OUT_DIR = Path("/Users/shazzak/Capital Stake - Results/diagnostics")
# parallelism (match the sweeps) and day sample (a profile needs ~30 days, not 197).
WORKERS = 9
MAX_DAYS = 60
# markout horizons in SECONDS, 1s out past the ~765s middle-bucket mean hold.
HORIZONS_S = [1, 5, 15, 30, 60, 120, 240, 600, 1200]
# "at the ceiling" = within 5% of max_inv (matches time_at_inventory_limit).
LIMIT_FRAC = 0.95
# thin / deep universes for plot coloring.
THIN = {"FNEL", "TPL", "PACE", "PIAHCLA", "HASCOL", "NPL", "TOMCL"}
DEEP = {"OGDC", "PSO", "HUBC", "FFC", "PPL", "UBL", "NBP", "MEBL"}


# ===========================================================================
# CORE: turn a (t, pos) path into inventory-profile stats (time-weighted)
# ===========================================================================
def markout_by_horizon(fills, eq_t, eq_mid, horizons_s):
    # fills: DataFrame with t (ms), side ('BUY'/'SELL'), bucket. eq_t/eq_mid: sorted mid path.
    # Returns DataFrame rows (bucket, horizon_s, markout_bps) per fill x horizon.
    eq_t = np.asarray(eq_t, dtype=float)
    eq_mid = np.asarray(eq_mid, dtype=float)
    if fills is None or len(fills) == 0 or eq_t.size == 0:
        return pd.DataFrame(columns=["bucket", "horizon_s", "markout_bps"])
    ft = fills["t"].to_numpy(dtype=float)
    # side sign: +1 buy (long -> up is good), -1 sell
    sgn = np.where(fills["side"].to_numpy() == "BUY", 1.0, -1.0)
    # mid at the fill instant: last mid AT/BEFORE the fill (no look-ahead)
    i0 = np.searchsorted(eq_t, ft, side="right") - 1
    ok0 = i0 >= 0
    mid0 = np.where(ok0, eq_mid[np.clip(i0, 0, eq_t.size - 1)], np.nan)
    out = []
    for h in horizons_s:
        # first mid AT/AFTER t_fill + h (genuine future; markout is an evaluation, not a signal)
        i1 = np.searchsorted(eq_t, ft + h * 1000.0, side="left")
        ok1 = i1 < eq_t.size
        mid1 = np.where(ok1, eq_mid[np.clip(i1, 0, eq_t.size - 1)], np.nan)
        # side-signed drift in bps; NaN where the horizon runs past the session end
        with np.errstate(invalid="ignore", divide="ignore"):
            mk = sgn * (mid1 - mid0) / mid0 * 1e4
        mk = np.where(ok0 & ok1 & (mid0 > 0), mk, np.nan)
        out.append(pd.DataFrame({"bucket": fills["bucket"].to_numpy(),
                                 "horizon_s": h, "markout_bps": mk}))
    return pd.concat(out, ignore_index=True)


# ===========================================================================
# ENGINE RUNNER: mirror the sweep worker (build_micro_params + run_symbol_day)
# ===========================================================================
# Winner config constants (identical to skew_sweep_2d.py: the frozen winner
# et1 / obi+ / tol0 / mid, 3x trailing-median clip).
TRAIL_DAYS = 10
CLIP_MULT = 3.0
EXIT_INV_THRESHOLD = 1.0
OBI_DEF_THRESH = 0.15
OBI_DEF_TICKS = 1.0


def _pos_path_for(R, H, calib, date, sym, dsets):
    # per-name calibration (same sourcing as the sweep's _process)
    scales = calib["scales"]; profiles = calib["profiles"]; windows = calib["windows"]
    segments = calib["segments"]; all_dates = calib["all_dates"]; tstats = calib["tstats"]
    # session segments for the day
    segs = segments.get(str(date))
    if segs is None:
        return None
    # trailing-median trade size -> clip
    med = H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
    if med is None or med <= 0:
        return None
    if sym not in scales or sym not in profiles:
        return None
    # the quote clip (shares)
    clip = max(1, int(round(CLIP_MULT * med)))
    # build the WINNER params (byte-identical overrides to the frozen winner)
    params = H.build_micro_params(
        clip, scales[sym], profiles[sym], windows.get(sym, (5.0, 1.0)), segs,
        overrides={"exit_ticks_inside": 1,
                   "exit_inv_threshold": EXIT_INV_THRESHOLD,
                   "obi_defensive": True,
                   "obi_defensive_thresh": OBI_DEF_THRESH,
                   "obi_defensive_ticks": OBI_DEF_TICKS,
                   "use_microprice": False,
                   "tol_ticks": 0.0,
                   "ofi_defensive": False,
                   "micro_lambda": None})
    # run the frozen engine via the harness (same call the sweep uses)
    dr = H.run_symbol_day(date, sym, dsets, params)
    if dr is None or dr.pnl() is None:
        return None
    # --- fills + the mid path ---
    fills = dr.fills if isinstance(dr.fills, pd.DataFrame) else pd.DataFrame(list(dr.fills))
    if fills is None or len(fills) == 0 or not {"t", "side", "bucket"}.issubset(fills.columns):
        return None
    eq = pd.DataFrame(dr.equity) if len(dr.equity) else pd.DataFrame()
    if not len(eq) or "mid" not in eq.columns or "t" not in eq.columns:
        return None
    eq = eq.sort_values("t")
    return dict(fills=fills, eq_t=eq["t"].to_numpy(dtype=float), eq_mid=eq["mid"].to_numpy(dtype=float))


# worker globals (set once per process by the Pool initializer)
_CALIB = None
_R = None
_H = None


def _init_worker(calib):
    global _CALIB, _R, _H
    import run_legacy_mm as R
    import mm_harness as H
    R.USE_MICRO = True
    _CALIB = calib
    _R = R
    _H = H


def _work_date(date):
    # all symbols for one date; returns per-(name,bucket,horizon) mean markout + n
    dsets = _R.open_datasets(date)
    if dsets is None:
        return []
    rows = []
    for sym in _CALIB["names"]:
        try:
            pp = _pos_path_for(_R, _H, _CALIB, date, sym, dsets)
        except Exception as e:
            print(_ts() + f"SKIP {date} {sym}: {e!r}", flush=True)
            continue
        if pp is None:
            continue
        mk = markout_by_horizon(pp["fills"], pp["eq_t"], pp["eq_mid"], HORIZONS_S)
        if mk.empty:
            continue
        # per (bucket, horizon): fill-mean markout + count (a day-as-unit value)
        g = mk.groupby(["bucket", "horizon_s"])["markout_bps"].agg(["mean", "count"]).reset_index()
        for _, r in g.iterrows():
            rows.append(dict(date=date, symbol=sym, bucket=r["bucket"], horizon_s=int(r["horizon_s"]),
                             markout_bps=float(r["mean"]), n_fills=int(r["count"])))
    return rows


def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    # frozen driver + harness (parent-side, for calibration + date discovery)
    import run_legacy_mm as R
    import mm_harness as H
    R.USE_MICRO = True
    print(_ts() + "pre-pass: calibration + trailing median trade size", flush=True)
    all_dates = R.discover_dates()
    # symbol universe: calibrated names (or a caller subset)
    names = symbols or (list(H.NAMES) if hasattr(H, "NAMES") else None)
    if not names:
        names = sorted(H.load_scales().keys())
    calib = {"scales": H.load_scales(), "profiles": H.load_profiles(),
             "windows": H.load_windows(), "segments": H.load_segments(),
             "all_dates": all_dates,
             "tstats": H.trailing_median_trade_size(all_dates, names, TRAIL_DAYS),
             "names": names}
    # skip the trailing-median warmup, then SAMPLE evenly (a profile is day-stable;
    # even spacing avoids one regime, e.g. the Iran/oil window)
    run_dates = all_dates[TRAIL_DAYS:]
    if max_days and len(run_dates) > max_days:
        stride = max(1, len(run_dates) // max_days)
        run_dates = run_dates[::stride][:max_days]
    print(_ts() + f"[inv] {len(names)} names x {len(run_dates)} dates (sampled), "
          f"{workers} workers", flush=True)
    # parallel over dates
    rows = []
    t0 = time.perf_counter()
    with Pool(processes=workers, initializer=_init_worker, initargs=(calib,)) as pool:
        done = 0
        for res in pool.imap_unordered(_work_date, run_dates):
            rows.extend(res)
            done += 1
            # per-date heartbeat (frequent enough to never look hung)
            print(_ts() + f"  date {done}/{len(run_dates)} done  "
                  f"({len(rows)} sym-days, {(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    if not rows:
        print(_ts() + "no runnable symbol-days.", flush=True)
        return
    df = pd.DataFrame(rows).dropna(subset=["markout_bps"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "markout_horizon_pername_day.csv", index=False)
    # ---- portfolio day-as-unit curve per bucket: unit = (date), names averaged ----
    def _t(x):
        x = x[~np.isnan(x)]; n = x.size
        if n < 2: return np.nan, np.nan, n
        m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); return m, se, n
    print(_ts() + "===== MARKOUT BY HORIZON FROM FILL (side-signed drift, bps; day-as-unit) =====", flush=True)
    curve = []
    for b_ in ["first15", "middle", "preclose45", "last15"]:
        line = f"  {b_:11s}"
        for h in HORIZONS_S:
            d = df[(df.bucket == b_) & (df.horizon_s == h)].groupby("date")["markout_bps"].mean().to_numpy()
            m, se, n = _t(d)
            curve.append(dict(bucket=b_, horizon_s=h, markout_bps=m, se=se, n_days=n))
            line += f"  {h:>5d}s:{m:+6.2f}"
        print(_ts() + line, flush=True)
    cv = pd.DataFrame(curve)
    cv.to_csv(out_dir / "markout_horizon_curve.csv", index=False)
    # knee: where does the middle-bucket curve stop getting worse?
    mid = cv[cv.bucket == "middle"].sort_values("horizon_s")
    print(_ts() + "  middle-bucket: markout at 5s = %+.2f  at 60s = %+.2f  at 600s = %+.2f  at 1200s = %+.2f"
          % tuple(mid.set_index("horizon_s").loc[[5, 60, 600, 1200], "markout_bps"]), flush=True)
    _plots(cv, out_dir)
    print(_ts() + f"[markout] outputs -> {out_dir}", flush=True)


def _color(sname):
    return "#c0392b" if sname in THIN else ("#2c6fbb" if sname in DEEP else "#888888")


def _plots(cv, out_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    for b_, c in [("first15", "#888"), ("middle", "#2c6fbb"), ("preclose45", "#c0392b"), ("last15", "#e67e22")]:
        d = cv[cv.bucket == b_].sort_values("horizon_s")
        ax.errorbar(d["horizon_s"], d["markout_bps"], yerr=d["se"], marker="o", capsize=3,
                    lw=2 if b_ == "middle" else 1.2, color=c, label=b_)
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.8)
    ax.axvline(765, color="#2c6fbb", ls=":", lw=1, label="middle mean hold (~765s)")
    ax.set_xlabel("horizon from fill (s, log)"); ax.set_ylabel("side-signed mid drift (bps)")
    ax.set_title("How adverse selection ACCRUES with holding time (day-as-unit +/-SE)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(out_dir / "markout_horizon.png", dpi=130); plt.close(fig)


# ===========================================================================
# SELF-TEST: profile math on synthetic paths with known answers
# ===========================================================================
def self_test():
    # synthetic mid path: mid rises 1 bps/sec (linear drift) from 100.0, 1 event per 500ms
    t = np.arange(0, 2_000_000, 500.0)            # 0..2000s in ms
    mid = 100.0 * (1 + 1e-4 * (t / 1000.0))       # +1 bps per second
    # one BUY fill at t=100s and one SELL fill at t=100s, in the middle bucket
    fills = pd.DataFrame({"t": [100_000.0, 100_000.0], "side": ["BUY", "SELL"],
                          "bucket": ["middle", "middle"]})
    mk = markout_by_horizon(fills, t, mid, [1, 5, 60, 600])
    # BUY should see +h bps (mid up = good for a long); SELL should see -h bps
    for h in [1, 5, 60, 600]:
        vals = mk[mk.horizon_s == h]["markout_bps"].to_numpy()
        buy, sell = vals[0], vals[1]
        # analytic truth: drift of h bps-of-100 measured against mid0=101.0 -> h/1.01
        exp = h / 1.01
        print(_ts() + f"[self-test] h={h:>4}s  BUY={buy:+8.3f}  SELL={sell:+8.3f}  (expect +/-{exp:.3f})")
        assert abs(buy - exp) < 1e-6, f"BUY markout wrong at {h}s"
        assert abs(sell + exp) < 1e-6, f"SELL markout wrong at {h}s"
    # horizon past session end -> NaN, never a bogus number
    mk2 = markout_by_horizon(fills, t, mid, [5000])
    assert mk2["markout_bps"].isna().all(), "past-end horizon must be NaN"
    print(_ts() + "[self-test] past-session horizon -> NaN OK")
    # no look-ahead at t0: mid0 uses the mid AT/BEFORE the fill
    fills2 = pd.DataFrame({"t": [250.0], "side": ["BUY"], "bucket": ["middle"]})  # between events 0 and 500ms
    mk3 = markout_by_horizon(fills2, t, mid, [1])
    # mid0 = mid at t=0 (=100.0); mid1 = first event >= 1250ms -> t=1500 -> 100*(1+1.5e-4) -> +1.5 bps
    assert abs(mk3["markout_bps"].iloc[0] - 1.5) < 0.02, "no-look-ahead anchor wrong"
    print(_ts() + "[self-test] mid0 anchored at/before fill (no look-ahead) OK")
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


# ===========================================================================
# ENTRY
# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inventory-profile diagnostic.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    args = ap.parse_args()
    # Default to self-test.
    if args.self_test or not args.run:
        self_test()
    if args.run:
        run_real(symbols=args.symbols, workers=args.workers, max_days=args.days)
