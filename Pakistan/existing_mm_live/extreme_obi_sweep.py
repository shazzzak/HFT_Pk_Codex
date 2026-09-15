# extreme_obi_sweep.py -- at EXTREME book imbalance the exposed side loses money.
# Is the cause the LEAN crossing the touch, or just too much size in a bad state?
#
# THE FINDING THIS TESTS (probe_conditional_markout.py, 518,285 fills, 207 days).
# Splitting the old '0.30+' bucket exposed a population that the coarse bucket had
# been averaging away:
#
#   obi_1 bucket   side         n_fills   capture   markout    gross
#   0.60-0.80      exposed       33,291   +1.4381   -0.1740   +1.2641
#   0.80+          exposed       86,238   -0.4234   -0.3584   -0.7817   <-- here
#   0.80+          favourable   158,769   +1.4030   +0.6785   +2.0815
#
# CAPTURE IS NEGATIVE on the exposed side above obi_1 0.80 -- 86,238 fills, 16.6%
# of the book, losing 0.78 bps each. Negative capture is the A.2 signature: the
# lean walking the quote THROUGH the opposite touch (KEL/PIBTL/TPL went +1.64 ->
# -1.79 the same way).
#
# ===========================================================================
# UNITS. READ THIS BEFORE CHANGING ANY THRESHOLD IN THIS FILE.
# ===========================================================================
# TWO DIFFERENT SCALES ARE IN PLAY and they differ by exactly 2x:
#   micro_mm   fires on (imb - 0.5) where imb = B/(B+A) in [0,1]
#              so the trigger quantity |imb - 0.5| lives in [0, 0.5]
#   the fills  store obi_1 = (B-A)/(B+A) in [-1,+1]
#   and        obi_1 = 2 * (imb - 0.5)   [algebraic identity, not a convention]
#
# Therefore:
#   the shipped queue_skew_thresh = 0.15  IS  obi_1 = 0.30
#   the probe's 0.80 boundary      IS  |imb - 0.5| = 0.40
#
# EVERY THRESHOLD IN THIS FILE IS IN MICRO_MM UNITS (|imb - 0.5|), because that is
# what the engine consumes. The obi_1 equivalent is printed beside each one so a
# mix-up is visible in the output rather than silent in the result.
#
# ===========================================================================
# WHAT THE PROBE COULD NOT SEPARATE, AND WHY THIS SWEEP EXISTS
# ===========================================================================
# In the fill data, "the book is lopsided" and "the lean is firing" are COLLINEAR
# by construction -- the lean fires BECAUSE the book is lopsided. So the probe
# cannot tell whether extreme imbalance is intrinsically bad, or whether the lean
# is what makes it bad. Only holding the state fixed and varying the mechanism
# separates them. That is this sweep.
#
# THE ARMS. All are compared against the shipped config, day-as-unit.
#   baseline        what ships today: lean 2 ticks past |imb-0.5| = 0.15
#   lean_band       lean fires only INSIDE 0.15 .. 0.40, off above -- treats the
#                   cause if negative capture is the lean crossing the touch.
#                   NEEDS A NEW micro_mm KWARG (see CAPABILITY PROBE below).
#   throttle        baseline + cut the exposed side above 0.40 -- treats the
#                   symptom: same mechanism, less size behind it
#   boost           baseline + more size on the favourable side above 0.40 --
#                   the offensive twin; that side earns +2.08 bps up there
#   throttle_boost  both size levers, lean untouched
#   all             lean_band + both size levers
#
# PRIOR, stated before the run so it is falsifiable: lean_band wins. Negative
# capture means the quote is crossing; cutting size scales that loss down but does
# not remove it, while switching the lean off in that band removes its cause.
#
# Run: caffeinate -is python extreme_obi_sweep.py --smoke
#      caffeinate -is python extreme_obi_sweep.py --run
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json, sys
import numpy as np, pandas as pd

# --- PATHS from config_pk; never a literal in this file ----------------------
try:
    # the project's central path module
    from config_pk import RESULTS_ROOT
except Exception as _e:
    # a wrong store is worse than a crash
    raise SystemExit("extreme_obi_sweep: could not import RESULTS_ROOT from "
                     "config_pk. Run from existing_mm_live/. Original: %r" % _e)
# the shared production harness -- one backtest path, so results cannot diverge
import mm_harness as H
# the driver module the harness itself wraps (datasets, date discovery)
import run_legacy_mm as R
# the strategy, for the capability probe below
from micro_mm import MicrostructureMM

# run stamp on every output, so a re-run never collides with an earlier one
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
# where results go
OUT_DIR = Path(RESULTS_ROOT) / "extreme_obi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# THRESHOLDS -- micro_mm units, |imb - 0.5| in [0, 0.5]
# ===========================================================================
# the shipped lean trigger (obi_1 0.30)
THRESH_LO = 0.15
# where capture goes negative (obi_1 0.80) -- the boundary this sweep is about
THRESH_HI = 0.40
# the lean, in ticks -- unchanged from what ships
LEAN_TICKS = 2.0
# how far the throttle cuts the exposed side (0.5 = half size)
THROTTLE_FRAC = 0.5
# how long a throttle trigger persists, ms
THROTTLE_HOLD_MS = 300.0
# how much the boost adds to the favourable side
BOOST_MULT = 1.5
# ===========================================================================
# THE THRESHOLD GRIDS -- micro_mm units |imb-0.5|; obi_1 equivalent is 2x
# ===========================================================================
# WHY THESE ARE SWEPT RATHER THAN FIXED (SZ, 2026-09-15). The first pass put
# every mechanism at 0.40 because that is where CAPTURE GOES NEGATIVE. But that
# is where the EXPOSED side breaks. The FAVOURABLE side is profitable at every
# imbalance level measured -- 2.08 bps at the extreme, 3.35 bps in the middle --
# so there is no reason the boost should wait until 0.40 to fire. It may want to
# start far earlier, and a single guessed threshold cannot find that out.
# 0.05 = obi_1 0.10, 0.40 = obi_1 0.80
BOOST_THRESHES = [0.05, 0.10, 0.15, 0.25, 0.40]
# the throttle is ALREADY live at 0.15; sweep where a DEEPER cut should start
THROTTLE_THRESHES = [0.15, 0.25, 0.40]
# how deep the deeper cut goes (production is 0.5)
THROTTLE_DEEP_FRAC = 0.25
# where the lean band should close, once micro_mm supports it
BAND_THRESHES = [0.25, 0.40]
# names and dates for --smoke
SMOKE_NAMES, SMOKE_DAYS = 6, 10
# names and dates for --run (0 = every quoted name / every date)
RUN_NAMES, RUN_DAYS = 30, 20
# the clip multiple on the trailing median trade size -- production value
CLIP_MULT = 3.0
# how many days of trailing history set that median -- production value
TRAIL_DAYS = 10
# the shipped assignment, for the name list and each name's own lean setting
ASSIGNMENT = Path(RESULTS_ROOT) / "config_assignment_20260915_0043.csv"

print(f"extreme_obi_sweep  stamp={STAMP}")
print(f"  thresholds, micro_mm units |imb-0.5|  ->  obi_1 equivalent (x2):")
print(f"    lean trigger  {THRESH_LO:.2f}  ->  obi_1 {2*THRESH_LO:.2f}")
print(f"    extreme band  {THRESH_HI:.2f}  ->  obi_1 {2*THRESH_HI:.2f}")

# ===========================================================================
# CAPABILITY PROBE -- does this micro_mm support the lean band?
# ===========================================================================
# The lean_band arm needs an UPPER bound on the lean's firing band, which the
# current engine does not have: the condition is `(imb-0.5) > thresh` with no cap.
# Probe for the kwarg rather than assuming it, so this script runs usefully
# either way instead of dying on a TypeError deep in a backtest.
HAS_BAND = "queue_skew_thresh_hi" in MicrostructureMM.__init__.__code__.co_varnames
print(f"\n  micro_mm supports queue_skew_thresh_hi: {HAS_BAND}")
if not HAS_BAND:
    print("    -> the lean_band and all arms will be SKIPPED. Apply the micro_mm")
    print("       patch (delivered alongside this file) to enable them. The size")
    print("       arms run regardless and are still informative on their own.")

# the params builder, confirmed by reading universe_expand.py's working call
BUILD = H.build_micro_params

# ===========================================================================
# THE PRODUCTION BASE -- copied from universe_expand.py's _OBI, not invented
# ===========================================================================
# CORRECTION 2026-09-15: obi_throttle is ALREADY ON in the shipped config, at
# threshold 0.15 (obi_1 0.30), cutting the exposed side to half size. An earlier
# reading of micro_mm's DEFAULTS said it was off; the runner overrides them.
#
# That matters for what this sweep can conclude. The throttle is already firing
# across the ENTIRE region where capture goes negative, and it is not preventing
# it -- halving the size did not stop the exposed side losing 0.78 bps above
# obi_1 0.80. So "too much size" is already partly treated and did not work,
# which is evidence FOR the lean being the cause rather than the size.
#
# size_boost_mult = 1.0 here, so the BOOST is genuinely untested.
PROD_BASE = dict(
    # the size throttle, ON in production
    obi_throttle=True, ofi_throttle=False, obi_throttle_thresh=0.15,
    throttle_frac=THROTTLE_FRAC, throttle_hold_ms=THROTTLE_HOLD_MS,
    qdr_throttle=False, enable_pov_cap=False, flow_throttle=False,
    enable_run_reprice=False, enable_aggr_lean=False, enable_age_cross=False,
    # the boost, OFF in production
    size_boost_mult=1.0,
    # the lean is set PER NAME from the assignment, not here
    queue_skew_bps=0.0, enable_inv_taper=False,
    # the asymmetric 1-tick widen, ON in both shipped configs
    obi_defensive=True, obi_defensive_thresh=0.15, obi_defensive_ticks=1.0,
    # the exit path, at its production settings
    exit_ticks_inside=1, exit_inv_threshold=1.0,
    # axes held at their production values
    use_microprice=False, tol_ticks=0.0, ofi_defensive=False, micro_lambda=None,
)
# the 1-tick-book names forced to plain OBI in the shipped book (A.2)
CHEAP_EXCLUDED = {"KEL", "PIBTL", "TPL"}

# ===========================================================================
# THE ARMS -- overrides applied on top of PROD_BASE + the name's own lean
# ===========================================================================
# every arm, in the order they are reported. The control comes first and every
# other arm is PAIRED against it on the same days.
ARMS = {"baseline": {}}
# BOOST swept over where it starts firing. Only size_boost_thresh varies, so any
# difference between these arms is attributable to the threshold alone.
for _t in BOOST_THRESHES:
    ARMS[f"boost@{_t:.2f}"] = dict(size_boost_mult=BOOST_MULT,
                                   size_boost_thresh=_t)
# DEEPER THROTTLE swept over where the deeper cut starts. 0.15 is the shipped
# trigger, so that arm is "same trigger, cut twice as hard"; the others move the
# trigger up so the deeper cut applies only in the more extreme states.
for _t in THROTTLE_THRESHES:
    ARMS[f"thr{THROTTLE_DEEP_FRAC:.2f}@{_t:.2f}"] = dict(
        throttle_frac=THROTTLE_DEEP_FRAC, obi_throttle_thresh=_t)
# LEAN BAND, only if this engine can express it
if HAS_BAND:
    for _t in BAND_THRESHES:
        ARMS[f"leanband@{_t:.2f}"] = dict(queue_skew_thresh_hi=_t)
# drop the arms this engine cannot express, rather than crashing mid-run
if not HAS_BAND:
    ARMS = {k: v for k, v in ARMS.items() if "queue_skew_thresh_hi" not in v}

# JOURNAL TAG. universe_expand's lesson: a resume journal keyed only on the cell
# identity will silently fold results from a DIFFERENT run definition into the
# same totals and reconcile perfectly.
#
# THE FIRST VERSION HASHED ONLY THE ARMS AND THAT WAS NOT ENOUGH. Changing the
# name sampler from alphabetical to stratified left the arms untouched, so the
# tag was unchanged, so the run RESUMED onto the old sample and reported a
# union of eleven names under a header that said six. The sample is part of the
# run's identity exactly as much as the arms are.
#
# The tag is now computed in resolve_tag() AFTER the names and dates are known,
# because those are not known at import time.
def resolve_tag(names, dates):
    """Hash arms + names + dates, so any change to the run starts a new journal."""
    # the arm definitions
    sig = {"arms": {k: sorted(v.items()) for k, v in ARMS.items()},
           # the exact symbols, order-independent
           "names": sorted(names),
           # the exact dates, order-independent
           "dates": sorted(str(d) for d in dates)}
    # a short stable digest of the whole run definition
    return hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:8]

print(f"  arms ({len(ARMS)}): {', '.join(ARMS)}")


def per_name_setting(assignment_path):
    """Each name's own shipped lean setting, so the baseline IS the shipped book.

    A sweep that puts every name on one uniform lean is not measuring the thing
    that ships -- the whole point of the three-bucket assignment is that names
    differ. Returns {symbol: (skew_ticks, skew_thresh)} and drops the DROP names.
    """
    # the assignment written by build_config_assignment.py
    df = pd.read_csv(assignment_path)
    # the columns the harness is meant to read (labels are for humans)
    need = {"symbol", "skew_ticks", "skew_thresh"}
    # fail naming the mismatch rather than KeyError-ing later
    if not need.issubset(df.columns):
        raise SystemExit(f"{assignment_path.name}: need {sorted(need)}, "
                         f"has {sorted(df.columns)}")
    # a NaN skew_ticks marks a name that is not quoted at all
    df = df[df.skew_ticks.notna()]
    # keep the P&L column when it is there, so the sampler can stratify on it
    pkr = (df.set_index("symbol")["assigned_pkr"].to_dict()
           if "assigned_pkr" in df.columns else {})
    # map symbol -> its two engine numbers
    setting = {r.symbol: (float(r.skew_ticks),
                          0.15 if pd.isna(r.skew_thresh) else float(r.skew_thresh))
               for r in df.itertuples()}
    return setting, pkr


def sample_names(setting, pkr, n):
    """A STRATIFIED sample across the P&L range, not the alphabet.

    The first version took sorted(setting)[:n], which on this book returns AGHA,
    AGP, AHCL, AIRLINK, AKBL, APL -- six names beginning with A, one of which
    (AKBL) swings +/-7,000 PKR on a single day. That is not a sample of the book,
    it is a sample of one letter, and it inflates the variance on every paired
    t-statistic. Take every k-th name down the P&L-ranked list instead, so the
    sample spans big and small contributors in proportion.
    """
    # everything, biggest contributor first when the column exists
    ranked = sorted(setting, key=lambda s: -pkr.get(s, 0.0)) if pkr \
        else sorted(setting)
    # 0 or a request for everything returns the full list
    if not n or n >= len(ranked):
        return ranked
    # evenly spaced picks down the ranking
    step = len(ranked) / float(n)
    return [ranked[int(i * step)] for i in range(n)]


def run(n_names, n_days):
    """Run every arm over the selected names and dates, day-as-unit."""
    # each name's shipped setting, plus its assigned PKR for the sampler
    setting, pkr = per_name_setting(ASSIGNMENT)
    # the dates available in the parsed store
    all_dates = R.discover_dates()
    dates = list(all_dates)
    # THE CALIBRATION, loaded exactly as universe_expand.py loads it. These four
    # files ARE the shipped calibration; a sweep on a different one is not
    # measuring the shipped book.
    print("\n  loading calibration ...")
    scales = H.load_scales(); profiles = H.load_profiles()
    windows = H.load_windows(); segments = H.load_segments()
    # trailing median trade size per name, which sets each name's clip
    tstats = H.trailing_median_trade_size(all_dates, sorted(setting), TRAIL_DAYS)
    # name the calibration files in the output so a run is self-identifying
    for pat in ("session_scales_*.csv", "volume_profile_*.csv",
                "time_windows_*.csv", "session_segments_*.csv"):
        try:
            print(f"    {H.newest(pat).name}")
        except Exception:
            print(f"    {pat}: NOT FOUND")
    # DROP THE WARM-UP DATES. trailing_median needs TRAIL_DAYS of history, so the
    # first TRAIL_DAYS dates in the store can NEVER produce a cell -- every name
    # is skipped and the date silently contributes nothing. The first smoke lost
    # one of five days to exactly this.
    if len(dates) > TRAIL_DAYS:
        dates = dates[TRAIL_DAYS:]
    # thin the dates evenly rather than taking a contiguous block, so a single
    # unusual week cannot drive the result
    if n_days and len(dates) > n_days:
        step = max(1, len(dates) // n_days); dates = dates[::step][:n_days]
    # STRATIFIED across the P&L range, never alphabetical
    names = sample_names(setting, pkr, n_names)
    print(f"\n  {len(names)} names x {len(dates)} dates x {len(ARMS)} arms "
          f"= {len(names)*len(dates)*len(ARMS):,} symbol-days")
    # name the sample, so an unrepresentative one is visible before it costs an hour
    print(f"  names: {', '.join(names)}")

    # the journal is keyed on arms AND names AND dates, so a change to ANY of
    # them starts a fresh file instead of resuming onto a different run
    JOURNAL = OUT_DIR / f"journal_{resolve_tag(names, dates)}.csv"
    print(f"  run tag: {JOURNAL.stem.split('_')[-1]}  (journal: {JOURNAL.name})")

    # resume: cells already done in a previous run of THIS EXACT definition
    done = set()
    if JOURNAL.exists():
        j = pd.read_csv(JOURNAL)
        done = {(r.arm, r.symbol, r.date) for r in j.itertuples()}
        print(f"  resuming: {len(done):,} cells already journalled")

    # accumulate one row per (arm, symbol, date)
    rows = []
    # walk dates outermost so each date's parquet is opened once for all names
    for di, d in enumerate(dates, 1):
        # the three tables for this date
        dsets = R.open_datasets(d)
        # a missing partition is a skip, not a crash
        if dsets is None:
            print(f"  [{d}] SKIP, partition missing"); continue
        # the day's continuous-session segments; no segments -> unrunnable day
        segs = segments.get(str(d))
        if segs is None:
            print(f"  [{d}] SKIP, no session segments"); continue
        # every selected name
        for sym in names:
            # a name without a scale or a volume profile cannot be run
            if sym not in scales or sym not in profiles:
                continue
            # the trailing median trade size that sets this name's clip
            med = H.trailing_median(tstats[sym], all_dates, d, TRAIL_DAYS)
            # no trailing history yet -> skip, same rule as universe_expand
            if med is None or med <= 0:
                continue
            # the clip, at the production multiple
            clip = max(1, int(round(CLIP_MULT * med)))
            # that name's shipped lean numbers
            ticks, thresh = setting[sym]
            # A.2: the 1-tick-book names run plain OBI in the shipped book, so
            # they must run plain OBI here too or the baseline is not the baseline
            if sym in CHEAP_EXCLUDED:
                ticks, thresh = 0.0, 0.15
            # every arm
            for arm, extra in ARMS.items():
                # already have this cell from an earlier run
                if (arm, sym, d) in done:
                    continue
                # production base, then the name's own lean, then the arm on top
                ov = dict(PROD_BASE)
                ov["queue_skew_ticks"] = ticks
                ov["queue_skew_thresh"] = thresh
                ov.update(extra)
                # a cheap-tick name has no lean, so a lean-band arm is identical
                # to baseline on it -- harmless, and keeps the panel balanced
                try:
                    # the assembled kwargs, arm overrides applied last.
                    # Signature confirmed against universe_expand.py's call:
                    # build_micro_params(clip, scale, profile, window, segments,
                    #                    overrides=dict)
                    params = BUILD(clip, scales[sym], profiles[sym],
                                   windows.get(sym, (5.0, 1.0)), segs,
                                   overrides=ov)
                    # the single shared backtest path
                    dr = H.run_symbol_day(d, sym, dsets, params)
                except TypeError as e:
                    # a signature mismatch is a setup error, not a data error --
                    # stop immediately rather than logging thousands of identical
                    # failures across the whole run
                    raise SystemExit(
                        f"build_micro_params signature mismatch: {e!r}\n"
                        f"Called as build_micro_params(clip, scale, profile, "
                        f"window, segments, overrides=dict), which is how "
                        f"universe_expand.py line 372 calls it. Send me the def "
                        f"line if it has changed.")
                except Exception as e:
                    print(f"  [{d}] {sym} {arm}: {e!r}"); continue
                # an unrunnable symbol-day returns None
                if dr is None:
                    continue
                # the day's P&L for this arm
                rows.append({"arm": arm, "symbol": sym, "date": d,
                             "pnl": float(dr.pnl() or 0.0),
                             "fills": int(len(dr.fills))})
        # journal what is done so a crash does not lose the day, THEN report a
        # CUMULATIVE count. The first version printed len(rows) after clearing
        # the buffer each date, so it showed the per-date count and looked stuck.
        n_new = len(rows)
        if rows:
            pd.DataFrame(rows).to_csv(
                JOURNAL, mode="a", header=not JOURNAL.exists(), index=False)
            done |= {(r["arm"], r["symbol"], r["date"]) for r in rows}
            rows = []
        # a date that produced nothing is worth naming, not passing over
        if n_new == 0:
            print(f"  [{di}/{len(dates)}] {d}: NO CELLS -- no name had a trailing "
                  f"median, a scale and a profile on this date")
        else:
            print(f"  [{di}/{len(dates)}] {d}: +{n_new} cells, "
                  f"{len(done):,} total")

    # everything journalled for this arm set, including earlier runs
    if not JOURNAL.exists():
        raise SystemExit("no results were produced")
    return pd.read_csv(JOURNAL)


def report(df):
    """Day-as-unit comparison of each arm against the baseline."""
    print("\n=== RESULT ===")
    # total PKR per arm, the headline
    tot = df.groupby("arm").pnl.sum().sort_values(ascending=False)
    print("\n  total PKR by arm:")
    for a, v in tot.items():
        print(f"    {a:<16} {v:>15,.0f}")

    # DAY AS UNIT. Pool the fills and a busy day counts as many independent
    # observations, which inflates every t-statistic. One number per day, then
    # a PAIRED test against the baseline on the same days.
    daily = df.groupby(["arm", "date"]).pnl.sum().unstack(0)
    # the baseline column must exist for any comparison to mean anything
    if "baseline" not in daily.columns:
        print("\n  no baseline arm in the results -- cannot compare"); return
    # fills per arm, so the mechanism is visible and not just its P&L: the boost
    # MUST raise fill count or it is not doing what it claims to do
    fills = df.groupby("arm").fills.sum()
    print(f"\n  paired against baseline, day-as-unit, {len(daily)} days:")
    print(f"    {'arm':<18} {'mean diff PKR/day':>18} {'t':>8} {'days>0':>8} "
          f"{'fills vs base':>14}")
    out = []
    # report in threshold order within each family, so the curve reads left to right
    for a in sorted(daily.columns, key=lambda s: (s.split("@")[0], s)):
        # the control is not compared with itself
        if a == "baseline":
            continue
        # per-day difference, on days both arms produced a number
        d = (daily[a] - daily["baseline"]).dropna()
        # too few days to say anything
        if len(d) < 3:
            continue
        # paired t across days
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) \
            if d.std(ddof=1) > 0 else np.nan
        # how much this arm changed the number of fills
        fd = 100.0 * (fills.get(a, 0) / max(fills.get("baseline", 1), 1) - 1)
        print(f"    {a:<18} {d.mean():>18,.0f} {t:>8.2f} "
              f"{int((d>0).sum()):>5}/{len(d)} {fd:>13.1f}%")
        out.append({"arm": a, "mean_diff_pkr_day": d.mean(), "t": t,
                    "days": len(d), "days_positive": int((d > 0).sum()),
                    "fills_pct_vs_baseline": fd})
    # THE THRESHOLD CURVE. The point of sweeping is to see the SHAPE, not to pick
    # the best cell -- with this many arms the best one is partly luck. A real
    # effect is monotone or single-peaked across thresholds; noise is ragged.
    if out:
        o = pd.DataFrame(out)
        # split the arm label into its family and its threshold
        o["family"] = o.arm.str.split("@").str[0]
        o["thresh"] = pd.to_numeric(o.arm.str.split("@").str[-1], errors="coerce")
        for fam, g in o.dropna(subset=["thresh"]).groupby("family"):
            g = g.sort_values("thresh")
            print(f"\n  {fam} across thresholds "
                  f"(micro_mm units; obi_1 is 2x these):")
            for r in g.itertuples():
                print(f"    thresh {r.thresh:.2f} (obi_1 {2*r.thresh:.2f}): "
                      f"{r.mean_diff_pkr_day:>10,.0f} PKR/day  t={r.t:+.2f}")
            print("    ^ read the SHAPE. Monotone or single-peaked = a real")
            print("      threshold effect. Ragged = noise, and the best cell here")
            print("      is the winner's curse, not a setting.")
    # persist the comparison
    if out:
        p = OUT_DIR / f"extreme_obi_summary_{STAMP}.csv"
        pd.DataFrame(out).to_csv(p, index=False)
        print(f"\n  wrote {p}")
    print("\n  READ: the PRIOR was that lean_band wins -- negative capture means")
    print("  the quote is crossing the touch, and cutting size only scales that")
    print("  loss down rather than removing its cause. If throttle beats lean_band,")
    print("  that prior was wrong and the problem is size, not placement.")
    print("  |t| > 2 with most days positive is the bar; anything less is noise.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="every quoted name, every date -- long")
    a = ap.parse_args()
    # default to the smoke so an accidental bare invocation is cheap
    if a.full:
        df = run(0, 0)
    elif a.run:
        df = run(RUN_NAMES, RUN_DAYS)
    else:
        df = run(SMOKE_NAMES, SMOKE_DAYS)
    report(df)
