# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# fifo_holdtime_diag.py
# ----------------------------------------------------------------------------
# FIFO-realized P&L broken down by HOLDING TIME -- the proper "should we sell out
# faster?" test. The earlier markout-by-horizon used the AVERAGE fill (gentle);
# this weights by what you ACTUALLY held: match each buy to the sell that closes
# it (FIFO, exactly as mm_harness.fifo_attribution does), and bin each round
# trip's realized P&L by how long it was held. If long-held round trips earn far
# less (or lose), flattening faster pays; if flat, it doesn't.
# Per hold-time bin: realized bps (gross of fees; the ~1.55 RT fee line is drawn)
# and realized PKR, DAY-AS-UNIT. Winner config, sampled days. Smoke test + ETA.
#
# USAGE:
#   python fifo_holdtime_diag.py --self-test
#   python fifo_holdtime_diag.py --smoke
#   python fifo_holdtime_diag.py --run
# ============================================================================
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
# Resolve this filesystem path through the canonical checkout/data configuration.
OUT_DIR = Path(str(_hft_paths.RESULTS_ROOT / 'diagnostics'))
# parallelism (match the sweeps) and day sample (a profile needs ~30 days, not 197).
WORKERS = 9
MAX_DAYS = 30
# hold-time bin edges (ms) + labels; round trips binned by exit_t - entry_t.
HOLD_EDGES_MS = [15000, 60000, 300000, 900000]
HOLD_LABELS = ['<15s', '15-60s', '1-5m', '5-15m', '>15m']
FEE_RT_BPS = 1.55
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
def _bucket_of(t, t0, t1):
    # the four session buckets by TIME (what the sweeps' decomp uses), not the
    # engine's current_bucket field (which only ever labels middle/last15).
    FIRST = 15 * 60 * 1000; PRE = 45 * 60 * 1000; LAST = 15 * 60 * 1000
    if t <= t0 + FIRST:
        return "first15"
    if t >= t1 - LAST:
        return "last15"
    if t >= t1 - PRE:
        return "preclose45"
    return "middle"


def _hold_bin(ms):
    # index into HOLD_LABELS for a hold duration in ms
    i = 0
    for e in HOLD_EDGES_MS:
        if ms < e:
            return HOLD_LABELS[i]
        i += 1
    return HOLD_LABELS[-1]


def fifo_holdtime(fills):
    # FIFO match (identical logic to mm_harness.fifo_attribution) but bin each
    # matched round trip's realized P&L by HOLD TIME. Returns per-bin dict:
    # {label: {realized, opened_notional, n}}.
    from collections import deque
    f = fills.to_dict("records") if isinstance(fills, pd.DataFrame) else list(fills)
    if not f or not {"side", "px", "qty", "t"}.issubset(
            f[0].keys() if isinstance(f[0], dict) else []):
        return None
    f = sorted(f, key=lambda r: float(r["t"]))
    open_lots = deque()
    acc = {b: {"realized": 0.0, "opn": 0.0, "n": 0} for b in HOLD_LABELS}
    for fl in f:
        side = fl["side"]; px = float(fl["px"]); qty = float(fl["qty"]); t = float(fl["t"])
        # same side (or empty) -> OPEN a lot
        if not open_lots or open_lots[0]["side"] == side:
            open_lots.append({"qty": qty, "px": px, "t": t, "side": side})
            continue
        # opposite side -> CLOSE lots FIFO
        remaining = qty
        while remaining > 1e-9 and open_lots and open_lots[0]["side"] != side:
            lot = open_lots[0]
            matched = min(remaining, lot["qty"])
            # sign-correct realized round trip on the matched shares
            realized = ((px - lot["px"]) * matched if lot["side"] == "BUY"
                        else (lot["px"] - px) * matched)
            hb = _hold_bin(t - lot["t"])
            acc[hb]["realized"] += realized
            acc[hb]["opn"] += matched * lot["px"]     # opened notional of the matched shares
            acc[hb]["n"] += 1
            lot["qty"] -= matched; remaining -= matched
            if lot["qty"] <= 1e-9:
                open_lots.popleft()
    return acc


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
    # --- fills only (FIFO needs side/px/qty/t) ---
    fills = dr.fills if isinstance(dr.fills, pd.DataFrame) else pd.DataFrame(list(dr.fills))
    if fills is None or len(fills) == 0 or not {"side", "px", "qty", "t"}.issubset(fills.columns):
        return None
    return dict(fills=fills)


# worker globals
_CALIB = None
_R = None
_H = None


def _init_worker(calib):
    global _CALIB, _R, _H
    import run_legacy_mm as R
    import mm_harness as H
    R.USE_MICRO = True
    _CALIB = calib; _R = R; _H = H


def _work_date(date):
    dsets = _R.open_datasets(date)
    if dsets is None:
        return []
    rows = []
    for sym in _CALIB["names"]:
        try:
            pp = _pos_path_for(_R, _H, _CALIB, date, sym, dsets)
        except Exception as e:
            print(_ts() + f"SKIP {date} {sym}: {e!r}")
            continue
        if pp is None:
            continue
        acc = fifo_holdtime(pp["fills"])
        if acc is None:
            continue
        for lab in HOLD_LABELS:
            a = acc[lab]
            if a["n"] == 0:
                continue
            rows.append(dict(date=date, symbol=sym, hold_bin=lab,
                             realized_pkr=a["realized"], opened_notional=a["opn"], n=a["n"]))
    return rows


def smoke(symbols=None):
    # ONE stock-day in THIS process (no pool): confirms it runs + times a unit.
    import time as _tm
    import run_legacy_mm as R
    import mm_harness as H
    R.USE_MICRO = True
    all_dates = R.discover_dates()
    names = symbols or (list(H.NAMES) if hasattr(H, "NAMES") else sorted(H.load_scales().keys()))
    calib = {"scales": H.load_scales(), "profiles": H.load_profiles(),
             "windows": H.load_windows(), "segments": H.load_segments(),
             "all_dates": all_dates,
             "tstats": H.trailing_median_trade_size(all_dates, names, TRAIL_DAYS),
             "names": names}
    date = all_dates[len(all_dates) // 2]
    dsets = R.open_datasets(date)
    print(_ts() + f"[smoke] one stock-day: {names[0]} on {date}")
    t0 = _tm.perf_counter()
    pp = _pos_path_for(R, H, calib, date, names[0], dsets)
    acc = fifo_holdtime(pp["fills"]) if pp else None
    dt = _tm.perf_counter() - t0
    if acc is None:
        print(_ts() + f"[smoke] no fills for {names[0]} {date} ({dt:.1f}s)")
    else:
        print(_ts() + f"[smoke] {names[0]} {date} in {dt:.1f}s -- realized by hold bin:")
        for lab in HOLD_LABELS:
            a = acc[lab]
            if a["n"]:
                bps = a["realized"] / a["opn"] * 1e4 if a["opn"] > 0 else float("nan")
                print(_ts() + f"[smoke]   {lab:>7}: {a['n']:>4} trips  realized {bps:+.2f} bps")
    nd = MAX_DAYS if MAX_DAYS else len(all_dates)
    print(_ts() + f"[smoke] one unit ~{dt:.1f}s. Full run {len(names)}x{nd}/{WORKERS} workers "
          f"~= {len(names)*nd*dt/WORKERS/60:.0f} min")


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
            el = (time.perf_counter() - t0) / 60.0
            eta = el / done * (len(run_dates) - done)
            print(_ts() + f"  date {done}/{len(run_dates)} done  "
                  f"({len(rows)} sym-days, {el:.1f} min elapsed, ETA {eta:.1f} min)", flush=True)
    if not rows:
        print(_ts() + "no runnable symbol-days.", flush=True)
        return
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "fifo_holdtime_pername_day.csv", index=False)

    def _t(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]; n = x.size
        if n < 2: return np.nan, np.nan, n
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)), n

    print(_ts() + "===== FIFO REALIZED P&L BY HOLDING TIME (gross of fees; day-as-unit) =====")
    print(_ts() + f"  realized bps of a round trip vs how long it was held. fee ~{FEE_RT_BPS} bps RT.")
    curve = []
    for lab in HOLD_LABELS:
        sub = df[df.hold_bin == lab]
        if not len(sub):
            continue
        perday = sub.groupby(["date", "symbol"]).apply(
            lambda z: (z["realized_pkr"].sum() / z["opened_notional"].sum() * 1e4)
            if z["opened_notional"].sum() > 0 else np.nan, include_groups=False)
        m, se, n = _t(perday.to_numpy())
        tot_pkr = sub["realized_pkr"].sum(); tot_n = int(sub["n"].sum())
        curve.append(dict(hold_bin=lab, realized_bps=m, se=se, realized_pkr=tot_pkr,
                          n_roundtrips=tot_n, n_namedays=n))
        print(_ts() + f"  {lab:>7}: realized {m:+6.2f} +/- {se:4.2f} bps   "
              f"PKR {tot_pkr:>12,.0f}   round-trips {tot_n:>9,}   [nd={n}]")
    cv = pd.DataFrame(curve)
    cv.to_csv(out_dir / "fifo_holdtime_curve.csv", index=False)
    print(_ts() + f"  (a bin is net-profitable if realized bps > {FEE_RT_BPS})")
    _plots(cv, out_dir)
    print(_ts() + f"[fifo-holdtime] outputs -> {out_dir}")


def _color(sname):
    return "#c0392b" if sname in THIN else ("#2c6fbb" if sname in DEEP else "#888888")


def _plots(cv, out_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(cv))
    a1.bar(x, cv["realized_bps"], yerr=cv["se"], capsize=4, color="#2c6fbb")
    a1.axhline(0, color="k", lw=0.8)
    a1.axhline(FEE_RT_BPS, color="g", ls="--", lw=1, label=f"+{FEE_RT_BPS} fee")
    a1.set_xticks(x); a1.set_xticklabels(cv["hold_bin"])
    a1.set_xlabel("holding time of the round trip"); a1.set_ylabel("realized bps (gross)")
    a1.set_title("Realized P&L vs holding time (does holding longer cost?)")
    a1.legend(fontsize=8); a1.grid(alpha=0.25, axis="y")
    a2.bar(x, cv["realized_pkr"], color="#888")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels(cv["hold_bin"])
    a2.set_xlabel("holding time"); a2.set_ylabel("realized PKR (total)")
    a2.set_title("Where the money is made, by holding time")
    a2.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(out_dir / "fifo_holdtime.png", dpi=130); plt.close(fig)


def self_test():
    fills = pd.DataFrame([
        dict(side="BUY", px=100.0, qty=100, t=0),
        dict(side="SELL", px=101.0, qty=100, t=10_000),
        dict(side="BUY", px=100.0, qty=100, t=100_000),
        dict(side="SELL", px=99.0, qty=100, t=500_000),
    ])
    acc = fifo_holdtime(fills)
    print(_ts() + f"[self-test] <15s: {acc['<15s']}   5-15m: {acc['5-15m']}")
    assert abs(acc["<15s"]["realized"] - 100.0) < 1e-9, "short realized wrong"
    assert abs(acc["<15s"]["opn"] - 10000.0) < 1e-9, "short opened notional wrong"
    assert acc["<15s"]["n"] == 1
    assert abs(acc["5-15m"]["realized"] + 100.0) < 1e-9, "long realized wrong"
    assert acc["5-15m"]["n"] == 1
    f2 = pd.DataFrame([
        dict(side="BUY", px=100.0, qty=100, t=0),
        dict(side="BUY", px=100.0, qty=50, t=0),
        dict(side="SELL", px=102.0, qty=120, t=20_000),
    ])
    a2 = fifo_holdtime(f2)
    print(_ts() + f"[self-test] partial-close 15-60s: {a2['15-60s']}")
    assert abs(a2["15-60s"]["realized"] - 240.0) < 1e-9, "partial FIFO realized wrong"
    assert a2["15-60s"]["n"] == 2, "should be 2 matched pairs"
    assert _hold_bin(14_999) == "<15s" and _hold_bin(15_000) == "15-60s"
    assert _hold_bin(59_999) == "15-60s" and _hold_bin(300_000) == "5-15m"
    assert _hold_bin(900_000) == ">15m"
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inventory-profile diagnostic.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="time ONE stock-day, no pool")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    args = ap.parse_args()
    # Default to self-test.
    if args.smoke:
        smoke(symbols=args.symbols)
    elif args.self_test or not args.run:
        self_test()
    if args.run:
        run_real(symbols=args.symbols, workers=args.workers, max_days=args.days)
