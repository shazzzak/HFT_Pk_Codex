# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# preflight_coverage.py -- verify the 38 names have what the sweep needs BEFORE
# committing to the full run. Catches the silent skips (_process returns None)
# that would otherwise only show up as missing cells 4 hours in.
# ============================================================================
# Checks, per name:
#   scales   : session_scale present (else _process line ~147 skips the name)
#   profiles : volume profile present (same skip)
#   windows  : time window present (optional -- _process defaults to (5,1))
#   fs_days  : how many of the run days have a feature-store partition (the
#              context the run needs; 0 -> the name contributes nothing)
# ============================================================================

# path + fs check
from pathlib import Path
# the harness loaders (identical source to the sweep's calibration)
import mm_harness as H
# the driver (dates, feature-store root)
import run_legacy_mm as R

# the SAME 38-name list the sweep uses
NAMES = ['AKBL', 'ATRL', 'BAFL', 'BOP', 'DGKC', 'ENGROH', 'FFC', 'FNEL',
         'HASCOL', 'HBL', 'HUBC', 'KEL', 'LUCK', 'MARI', 'MEBL', 'MLCF',
         'NBP', 'NCPL', 'NML', 'NPL', 'NRL', 'OGDC', 'PACE', 'PAEL',
         'PIAHCLA', 'PIBTL', 'PIOC', 'PPL', 'PSO', 'PTC', 'SAZEW', 'SEARL',
         'SYS', 'THCCL', 'TOMCL', 'TPL', 'TRG', 'UBL']
# trailing-median window (matches the sweep)
TRAIL_DAYS = 10
# feature-store root (matches persist / harness)
# Resolve this filesystem path through the canonical checkout/data configuration.
FS_ROOT = Path(str(_hft_paths.RESULTS_ROOT / 'feature_store'))


def main():
    # load calibration EXACTLY as the sweep does
    scales = H.load_scales()
    profiles = H.load_profiles()
    windows = H.load_windows()
    segments = H.load_segments()
    # all trading dates, then the run window (drop the trailing-median warm-up)
    all_dates = R.discover_dates()
    run_dates = all_dates[TRAIL_DAYS:]
    # header
    print(f"pre-flight coverage for {len(NAMES)} names over "
          f"{len(run_dates)} run days\n")
    print(f"{'name':>9s} {'scales':>7s} {'profile':>8s} {'window':>7s} "
          f"{'fs_days':>8s}  verdict")
    # per-name check
    ok_names = []
    bad_names = []
    for sym in NAMES:
        # the two that CAUSE a skip (line ~147: sym not in scales or profiles)
        has_scale = sym in scales
        has_prof = sym in profiles
        # optional (defaults if absent)
        has_win = sym in windows
        # feature-store day coverage over the run window
        fs_days = sum(1 for d in run_dates
                      if (FS_ROOT / sym / f"date={d}.parquet").exists())
        # a name is RUNNABLE only if it has scale + profile + >=1 fs day
        runnable = has_scale and has_prof and fs_days > 0
        # verdict text
        if runnable:
            verdict = "OK"
            ok_names.append(sym)
        else:
            # name the missing piece(s) explicitly
            miss = []
            if not has_scale:
                miss.append("no scale")
            if not has_prof:
                miss.append("no profile")
            if fs_days == 0:
                miss.append("no fs")
            verdict = "SKIP -> " + ", ".join(miss)
            bad_names.append((sym, verdict))
        # row
        print(f"{sym:>9s} {str(has_scale):>7s} {str(has_prof):>8s} "
              f"{str(has_win):>7s} {fs_days:>8d}  {verdict}")
    # summary
    print(f"\n{len(ok_names)}/{len(NAMES)} runnable.")
    # loud list of the ones that would silently vanish
    if bad_names:
        print("WOULD SILENTLY SKIP in the full run:")
        for sym, why in bad_names:
            print(f"  {sym}: {why}")
        print("\n-> fix calibration/feature-store for these, OR trim NAMES to the "
              "runnable set, BEFORE the full run.")
    else:
        print("all names runnable -- safe to launch the full run.")


if __name__ == "__main__":
    main()
