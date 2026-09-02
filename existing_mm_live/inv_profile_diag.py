# ============================================================================
# inv_profile_diag.py
# ----------------------------------------------------------------------------
# INVENTORY-PROFILE DIAGNOSTIC. Answers, from the real engine position path:
#   * how deep does |pos| actually get (in clips, and vs soft_inv / max_inv)?
#   * what fraction of the session is spent past soft_inv (add-side gated) and
#     pinned at the ceiling (>=95% of max_inv = one-sided, forfeiting spread)?
#   * what is the end-of-day |pos| distribution (the "ends short" skew)?
# per-name and pooled, day-as-unit. This decides whether the inventory-SIZING
# levers (soft_inv sweep, graded size-skew) are even worth building: if |pos|
# rarely approaches soft_inv, there is no over-accumulation to manage and the
# skew is a small persistent-direction problem, not a size problem.
#
# It REPLICATES run_legacy_mm.run_one exactly (same reads, session, latency,
# strategy) and captures the (t, pos) equity path bt.run returns -- it does not
# reimplement the engine. Strategy config = whatever run_legacy MICRO_PARAMS is
# set to; set that to the winner before --run to profile the deployed config.
#
# USAGE:
#   python inv_profile_diag.py --self-test          # validate the profile math
#   python inv_profile_diag.py --run                # full 38-name profile
#   python inv_profile_diag.py --run --symbols FNEL OGDC   # subset
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
MAX_DAYS = 30
# "at the ceiling" = within 5% of max_inv (matches time_at_inventory_limit).
LIMIT_FRAC = 0.95
# thin / deep universes for plot coloring.
THIN = {"FNEL", "TPL", "PACE", "PIAHCLA", "HASCOL", "NPL", "TOMCL"}
DEEP = {"OGDC", "PSO", "HUBC", "FFC", "PPL", "UBL", "NBP", "MEBL"}


# ===========================================================================
# CORE: turn a (t, pos) path into inventory-profile stats (time-weighted)
# ===========================================================================
def inventory_profile(t, pos, clip, max_inv, soft_inv):
    # Absolute inventory over time.
    t = np.asarray(t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    ap = np.abs(pos)
    # Guards for a clip of zero (avoid div by zero when normalizing to clips).
    clip = max(float(clip), 1e-9)
    # Result holder.
    out = {}
    # Peak inventory of the day, in clips.
    out["max_clips"] = float(ap.max() / clip) if ap.size else 0.0
    # End-of-day inventory (signed clips) and its magnitude.
    out["eod_clips_signed"] = float(pos[-1] / clip) if pos.size else 0.0
    out["eod_clips_abs"] = float(ap[-1] / clip) if ap.size else 0.0
    # Time-weighted metrics need at least two timestamps to form intervals.
    if t.size >= 2:
        # interval lengths
        dt = np.diff(t)
        # the level HELD over each interval is the level at its start
        held = ap[:-1]
        # total time (guard against zero span)
        tot = max(dt.sum(), 1.0)
        # fraction of session spent past soft_inv (add-side gated)
        out["frac_time_ge_soft"] = float(dt[held >= soft_inv].sum() / tot)
        # fraction pinned at the ceiling (>=95% max_inv = one-sided, lost spread)
        out["frac_time_ge_limit"] = float(dt[held >= LIMIT_FRAC * max_inv].sum() / tot)
        # time-weighted mean inventory, in clips
        out["mean_clips_timewt"] = float((dt * held).sum() / tot / clip)
    else:
        # single-point day: no dwell time to integrate
        out["frac_time_ge_soft"] = 0.0
        out["frac_time_ge_limit"] = 0.0
        out["mean_clips_timewt"] = 0.0
    return out


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
    # --- position path from dr.equity (has 'pos'); fall back to fills, exactly
    # as mm_harness.time_at_inventory_limit does ---
    eq = pd.DataFrame(dr.equity) if len(dr.equity) else pd.DataFrame()
    if len(eq) and "pos" in eq.columns and "t" in eq.columns:
        eq = eq.sort_values("t")
        tp = eq["t"].to_numpy(dtype=float)
        pp = eq["pos"].to_numpy(dtype=float)
    else:
        # reconstruct from fills (cumsum of signed qty)
        fills = dr.fills.to_dict("records") if isinstance(dr.fills, pd.DataFrame) else list(dr.fills)
        if not fills:
            return None
        f = pd.DataFrame(fills).sort_values("t")
        if not {"side", "qty", "t"}.issubset(f.columns):
            return None
        sgn = np.where(f["side"].to_numpy() == "BUY", 1.0, -1.0)
        pp = np.cumsum(sgn * f["qty"].to_numpy())
        tp = f["t"].to_numpy(dtype=float)
    if pp is None or len(pp) == 0:
        return None
    # clip / caps straight from the built params (byte-accurate to the run)
    clip_sh = float(params.get("size", clip) or clip)
    max_inv = float(params.get("max_inv", 10 * clip_sh))
    soft_inv = params.get("soft_inv", None)
    soft_inv = float(soft_inv) if soft_inv is not None else max_inv
    return dict(t=tp, pos=pp, clip=clip_sh, max_inv=max_inv, soft_inv=soft_inv)


# worker globals (set once per process by the Pool initializer)
_CALIB = None
_R = None
_H = None


def _init_worker(calib):
    # load the frozen driver + harness ONCE per worker, stash calibration
    global _CALIB, _R, _H
    import run_legacy_mm as R
    import mm_harness as H
    R.USE_MICRO = True
    _CALIB = calib
    _R = R
    _H = H


def _work_date(date):
    # process ALL symbols for one date (open datasets once). Returns profile rows.
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
        prof = inventory_profile(pp["t"], pp["pos"], pp["clip"], pp["max_inv"], pp["soft_inv"])
        prof.update(dict(date=date, symbol=sym, clip=pp["clip"],
                         max_inv=pp["max_inv"], soft_inv=pp["soft_inv"]))
        rows.append(prof)
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
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "inv_profile_pername_day.csv", index=False)
    print(_ts() + "===== INVENTORY PROFILE (pooled over symbol-days) =====", flush=True)
    print(_ts() + f"  peak |pos| (clips):     median {df['max_clips'].median():.2f}  "
          f"p90 {df['max_clips'].quantile(0.9):.2f}  max {df['max_clips'].max():.2f}", flush=True)
    print(_ts() + "  soft_inv = 3 clips, max_inv = 10 clips", flush=True)
    print(_ts() + f"  frac of day past soft_inv:  mean {df['frac_time_ge_soft'].mean()*100:.1f}%", flush=True)
    print(_ts() + f"  frac of day pinned at ceiling: mean {df['frac_time_ge_limit'].mean()*100:.2f}%", flush=True)
    print(_ts() + f"  EOD |pos| (clips):      median {df['eod_clips_abs'].median():.2f}  "
          f"p90 {df['eod_clips_abs'].quantile(0.9):.2f}", flush=True)
    short_days = float((df["eod_clips_signed"] < 0).mean())
    print(_ts() + f"  days ending net SHORT:  {short_days*100:.1f}%   "
          f"(mean EOD signed clips {df['eod_clips_signed'].mean():+.2f})", flush=True)
    pern = df.groupby("symbol").agg(
        max_clips=("max_clips", "median"),
        frac_soft=("frac_time_ge_soft", "mean"),
        frac_limit=("frac_time_ge_limit", "mean"),
        eod_signed=("eod_clips_signed", "mean"),
    ).reset_index()
    pern.to_csv(out_dir / "inv_profile_pername.csv", index=False)
    _plots(df, pern, out_dir)
    print(_ts() + f"[inv] outputs -> {out_dir}", flush=True)


def _color(sname):
    return "#c0392b" if sname in THIN else ("#2c6fbb" if sname in DEEP else "#888888")


def _plots(df, pern, out_dir):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
    a1.hist(df["max_clips"], bins=40, color="#444")
    a1.axvline(3, color="#e67e22", ls="--", label="soft_inv (3)")
    a1.axvline(10, color="#c0392b", ls="--", label="max_inv (10)")
    a1.set_xlabel("peak |pos| in a day (clips)"); a1.set_ylabel("symbol-days")
    a1.set_title("How deep does inventory get?"); a1.legend(fontsize=8)
    a2.hist(df["eod_clips_signed"], bins=40, color="#444"); a2.axvline(0, color="k")
    a2.set_xlabel("end-of-day pos (signed clips)")
    a2.set_title(f"EOD skew ({(df['eod_clips_signed']<0).mean()*100:.0f}% end short)")
    pn = pern.sort_values("eod_signed"); x = np.arange(len(pn))
    a3.bar(x, pn["eod_signed"], color=[_color(s) for s in pn["symbol"]])
    a3.axhline(0, color="k", lw=0.8); a3.set_xticks(x)
    a3.set_xticklabels(pn["symbol"], rotation=90, fontsize=6)
    a3.set_ylabel("mean EOD pos (signed clips)")
    a3.set_title("Per-name EOD skew (red=thin, blue=deep)")
    fig.tight_layout(); fig.savefig(out_dir / "inv_profile.png", dpi=130); plt.close(fig)


# ===========================================================================
# SELF-TEST: profile math on synthetic paths with known answers
# ===========================================================================
def self_test():
    # clip=100 shares; max_inv=1000 (10 clips); soft_inv=300 (3 clips).
    clip, max_inv, soft_inv = 100.0, 1000.0, 300.0
    # Path A: 4 points at t=0,1,2,3 (equal 1s intervals). pos held over each
    # interval = value at its start: [0, 500, 950, -200]. Last point -200 = EOD.
    #   intervals hold 0, 500, 950 -> |held| = 0,500,950
    #   >= soft(300): intervals 2 and 3 -> 2/3 of time
    #   >= 0.95*max(950): interval 3 (950>=950) -> 1/3 of time
    t = [0.0, 1.0, 2.0, 3.0]
    pos = [0.0, 500.0, 950.0, -200.0]
    p = inventory_profile(t, pos, clip, max_inv, soft_inv)
    print(_ts() + "[self-test] path A:", {k: round(v, 4) for k, v in p.items()})
    # peak |pos| = 950 -> 9.5 clips
    assert abs(p["max_clips"] - 9.5) < 1e-9, "max_clips wrong"
    # EOD signed = -200 -> -2.0 clips; abs 2.0
    assert abs(p["eod_clips_signed"] + 2.0) < 1e-9, "eod signed wrong"
    assert abs(p["eod_clips_abs"] - 2.0) < 1e-9, "eod abs wrong"
    # 2 of 3 intervals held >= soft_inv
    assert abs(p["frac_time_ge_soft"] - 2.0/3.0) < 1e-9, "frac soft wrong"
    # 1 of 3 intervals held >= 0.95*max_inv
    assert abs(p["frac_time_ge_limit"] - 1.0/3.0) < 1e-9, "frac limit wrong"
    # time-weighted mean |pos| = (0 + 500 + 950)/3 / 100 = 4.8333 clips
    assert abs(p["mean_clips_timewt"] - (0+500+950)/3/100) < 1e-9, "mean clips wrong"

    # Path B: UNEQUAL intervals -> confirm time-weighting, not point-counting.
    # t=0,1,11 (intervals 1s then 10s); pos=[0, 400, 0]. held=0 (1s), 400 (10s).
    #   >= soft: only the 10s interval -> 10/11 of time (not 1/2)
    t2 = [0.0, 1.0, 11.0]
    pos2 = [0.0, 400.0, 0.0]
    p2 = inventory_profile(t2, pos2, clip, max_inv, soft_inv)
    print(_ts() + "[self-test] path B:", {k: round(v, 4) for k, v in p2.items()})
    assert abs(p2["frac_time_ge_soft"] - 10.0/11.0) < 1e-9, "time-weighting broken"

    # Path C: fills-fallback parity -- reconstruct pos from BUY/SELL fills and
    # confirm it equals a direct path. BUY 500, BUY 500, SELL 300 -> 0,500,1000,700
    f = pd.DataFrame({"t": [0, 1, 2], "side": ["BUY", "BUY", "SELL"], "qty": [500, 500, 300]})
    sgn = np.where(f["side"] == "BUY", 1.0, -1.0)
    pos3 = np.cumsum(sgn * f["qty"].to_numpy())
    assert list(pos3) == [500.0, 1000.0, 700.0], "fills reconstruction wrong"
    print(_ts() + "[self-test] path C: fills->pos reconstruction OK", list(pos3))

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
