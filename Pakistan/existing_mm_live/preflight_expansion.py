# ============================================================================
# preflight_expansion.py -- STEP 6: verify all 114 names have what the engine
# needs, BEFORE committing ~9.5 hours to the backtest.
# ============================================================================
# Read-only. Writes one new timestamped CSV. Prints a go / no-go.
#
# This is preflight_coverage.py extended to the expanded book. The original is
# hardcoded to the 38 names and to the PRE-MOVE feature-store path
# (/Users/shazzak/Capital Stake - Results/feature_store), so it cannot be run
# against the 114 as-is. Same checks, same loaders, correct paths, and three
# additions.
#
# WHAT THE ENGINE ACTUALLY REQUIRES, read out of source rather than assumed:
#   fullyear_confirm._one():
#       segments.get(str(date)) is None      -> return None   (per DATE)
#       trailing_median(tstats[sym],...) None-> return None   (needs >=10 prior
#                                                              days with trades)
#       sym not in scales or sym not in profiles -> return None   <-- THE ONE
#                                                   that cost 2026-09-12
#   mm_harness.load_windows() -> OPTIONAL; _one falls back to (5.0, 1.0)
#   mm_harness._load_table()  -> filters note == "ok", so a row whose note is
#                                anything else is INVISIBLE to the loader even
#                                though it exists in the file
#
# THREE ADDITIONS OVER THE ORIGINAL
#   1. The note == "ok" check. A name can be present in the CSV and still be
#      dropped by the loader. The original tests membership of the loaded dict,
#      which already reflects the filter -- this reports it explicitly so a
#      silently-filtered row is legible rather than looking like an absent one.
#   2. Calibration QUALITY flags, not just presence: names whose calibration
#      rests on too few days, or that cannot be unwound inside the policy cap.
#      Present-but-garbage passes the engine's checks and produces numbers.
#   3. A single go / no-go line at the end.
#
# Run from existing_mm_live/:
#   caffeinate -is python preflight_expansion.py
# ============================================================================

# frames
import pandas as pd
# numeric
import numpy as np

# the driver -- imported FIRST so the rebind below is the last word
import run_legacy_mm as R
# the harness: the SAME loaders the sweep uses, so this cannot diverge from it
import mm_harness as H
# shared constants, guards and safe_out
import expansion_names as EX

# ---------------------------------------------------------------------------
# PATH REBIND + store guard
# ---------------------------------------------------------------------------
# push the canonical raw-store root on and prove it reads
ALL_DATES = EX.bind_parsed_root(R)
# the trailing window the sweep requires before a day is tradeable
TRAIL_DAYS = 10
# the feature store, under the REAL results root
FS_ROOT = EX.RESULTS_ROOT / "feature_store"
# below this many calibration days, a median is not a median
MIN_CAL_DAYS = 30


def main():
    # ---- load calibration EXACTLY as the sweep does ------------------------
    print("=" * 78)
    print(f"PRE-FLIGHT: {len(EX.ALL_NAMES)} names "
          f"({len(EX.INCUMBENT)} production + {len(EX.NEW_NAMES)} new)")
    print("=" * 78)
    # per-symbol session_scale
    scales = H.load_scales()
    # per-symbol 4-bucket volume profile
    profiles = H.load_profiles()
    # per-symbol EOD windows (optional -- _one defaults to (5.0, 1.0))
    windows = H.load_windows()
    # per-DATE continuous segments
    segments = H.load_segments()
    # name the exact files in force, so a stale one is visible immediately
    print(f"  scales   : {H.newest('session_scales_*.csv').name}  "
          f"({len(scales)} names)")
    print(f"  profiles : {H.newest('volume_profile_*.csv').name}  "
          f"({len(profiles)} names)")
    print(f"  windows  : {H.newest('time_windows_*.csv').name}  "
          f"({len(windows)} names)")
    print(f"  segments : {H.newest('session_segments_*.csv').name}  "
          f"({len(segments)} dates)")
    print(f"  parsed   : {R.PARSED_ROOT}")

    # the run window, after the trailing-median warm-up
    run_dates = ALL_DATES[TRAIL_DAYS:]
    # report it
    print(f"\n  {len(ALL_DATES)} dates, {len(run_dates)} after the "
          f"{TRAIL_DAYS}-day warm-up")
    # GUARD: every run date needs segments or _one returns None for ALL names
    no_seg = [d for d in run_dates if str(d) not in segments]
    # a short calendar is fatal for the whole run
    if no_seg:
        print(f"\n*** {len(no_seg)} run dates have NO session segments: "
              f"{no_seg[:5]}")
        print("    _one() returns None for every name on those days.")
    else:
        print(f"  every run date has segments: OK")

    # ---- the quality inputs, read from the raw calibration files -----------
    # the scale file, for its note and day-count columns
    sc_df = pd.read_csv(H.newest("session_scales_*.csv")).set_index("symbol")
    # the profile file
    pr_df = pd.read_csv(H.newest("volume_profile_*.csv")).set_index("symbol")
    # the window file
    wn_df = pd.read_csv(H.newest("time_windows_*.csv")).set_index("symbol")

    # ---- per-name check ----------------------------------------------------
    rows = []
    # walk every name in the expanded book
    for sym in EX.ALL_NAMES:
        # the two that CAUSE a silent skip
        has_scale = sym in scales
        has_prof = sym in profiles
        # optional -- absence means the (5.0, 1.0) default, not a skip
        has_win = sym in windows
        # feature-store day coverage (not needed by the engine run itself,
        # which calls run_symbol_day with want_fs=False, but several analysis
        # tools do need it, so it is reported)
        fs_days = sum(1 for d in run_dates
                      if (FS_ROOT / sym / f"date={d}.parquet").exists())
        # ---- quality, not just presence ----
        # how many days the profile rests on
        prof_days = (pr_df.loc[sym, "days"]
                     if sym in pr_df.index and "days" in pr_df.columns else np.nan)
        # whether the window says this name cannot be unwound in the cap
        cap_flag = (wn_df.loc[sym, "capacity_flag"]
                    if sym in wn_df.index and "capacity_flag" in wn_df.columns
                    else "")
        # the scale's own note, which the loader filters on
        sc_note = (sc_df.loc[sym, "note"]
                   if sym in sc_df.index and "note" in sc_df.columns else "")
        # RUNNABLE means the engine will produce cells for it
        runnable = has_scale and has_prof
        # the quality warnings that do NOT stop the engine but should stop you
        warn = []
        # a profile built on a handful of days
        if not pd.isna(prof_days) and prof_days < MIN_CAL_DAYS:
            warn.append(f"profile on {int(prof_days)}d")
        # a name that cannot be unwound inside the policy cap
        if cap_flag and cap_flag != "ok":
            warn.append(str(cap_flag))
        # the verdict
        if not runnable:
            # name the missing piece so the fix is obvious
            miss = []
            if not has_scale:
                miss.append("no scale")
            if not has_prof:
                miss.append("no profile")
            verdict = "SKIP -> " + ", ".join(miss)
        elif warn:
            verdict = "RUN (warn: " + ", ".join(warn) + ")"
        else:
            verdict = "OK"
        # one row per name
        rows.append({"symbol": sym,
                     "in_production": sym in EX.INCUMBENT,
                     "scale": has_scale, "profile": has_prof,
                     "window": has_win, "fs_days": fs_days,
                     "scale_note": sc_note, "profile_days": prof_days,
                     "capacity_flag": cap_flag, "verdict": verdict})
    # the assembled table
    df = pd.DataFrame(rows)

    # ---- report ------------------------------------------------------------
    # the names the engine would silently drop
    skips = df[df.verdict.str.startswith("SKIP")]
    # the names that will run but whose calibration is thin or uncapacitated
    warns = df[df.verdict.str.startswith("RUN")]
    # the clean ones
    clean = df[df.verdict == "OK"]
    # the headline counts
    print("\n" + "=" * 78)
    print(f"  clean          : {len(clean)}")
    print(f"  run with warns : {len(warns)}")
    print(f"  WOULD SKIP     : {len(skips)}")
    print("=" * 78)
    # the fatal ones first
    if len(skips):
        print("\nWOULD SILENTLY PRODUCE ZERO CELLS -- this is the 2026-09-12 failure:")
        print(skips[["symbol", "in_production", "scale", "profile",
                     "scale_note"]].to_string(index=False))
    # then the quality warnings
    if len(warns):
        print("\nWILL RUN, BUT THE CALIBRATION IS QUESTIONABLE:")
        print(warns[["symbol", "in_production", "profile_days", "capacity_flag",
                     "verdict"]].to_string(index=False))
        print("\n  profile on <30d : the median rests on too few days to be stable.")
        print("  EXCEEDS_CAP     : max_inv cannot be cleared inside the 60-min cap")
        print("                    at 10% POV. Reduce clips for that name rather")
        print("                    than trusting the clipped window.")
    # names with no feature store, which only matters for the analysis tools
    no_fs = df[df.fs_days == 0]
    # report separately so it is not confused with a fatal gap
    if len(no_fs):
        print(f"\nNO FEATURE-STORE DAYS ({len(no_fs)}) -- not fatal for the engine "
              f"run (want_fs=False) but breaks the markout lens:")
        print(list(no_fs.symbol))

    # ---- write + go/no-go ---------------------------------------------------
    # a fresh timestamped destination; never overwrites
    out = EX.safe_out("preflight_expansion", "csv")
    # write the full table
    df.to_csv(out, index=False)
    # say where it went
    print(f"\nwrote {out}")
    # the single line that decides whether to start the run
    print("\n" + "=" * 78)
    # a skip means the run would repeat the silent-failure mode
    if len(skips):
        print(f"NO-GO: {len(skips)} names would produce zero cells. Fix their")
        print("       calibration before starting the backtest.")
    elif no_seg:
        print("NO-GO: the session-segment calendar does not cover the run window.")
    else:
        # everything the engine needs is present
        print(f"GO: all {len(df)} names have scale + profile. The engine will")
        print(f"    produce cells for every one of them.")
        # but say plainly what is still questionable
        if len(warns):
            print(f"    {len(warns)} carry calibration warnings above -- decide")
            print(f"    whether to exclude them before committing the run.")
    print("=" * 78)


# entry point
if __name__ == "__main__":
    main()
