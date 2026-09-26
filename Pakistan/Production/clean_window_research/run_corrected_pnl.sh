#!/bin/bash
# Stop immediately if verification or any prerequisite fails.
set -euo pipefail
# Resolve the corrected research directory independently of the caller's working directory.
research_dir='/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/Production/clean_window_research'
# Use the established backtest environment.
python_bin='/Users/shazzak/PycharmProjects/HFT_Pk_Codex/backtest/bin/python'
# Preserve immutable historical dependencies behind the corrected modules.
export PYTHONPATH="$research_dir:/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/existing_mm_live"
# Prevent any bytecode writes in the historical code directory.
export PYTHONDONTWRITEBYTECODE=1
# Show progress immediately in the user's terminal.
export PYTHONUNBUFFERED=1
# Render reports without opening graphical windows.
export MPLBACKEND=Agg
# Keep plotting cache writes in an authorized temporary directory.
export MPLCONFIGDIR=/private/tmp/source-coverage-mpl
# Prevent each worker from spawning additional OpenMP compute threads.
export OMP_NUM_THREADS=1
# Prevent each worker from spawning additional BLAS compute threads.
export OPENBLAS_NUM_THREADS=1
# Restrict extra arguments to the explicit read-only preflight mode.
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--check-only' ) ]]; then
    # Explain accepted usage without starting a partial or unintended replay.
    printf 'Usage: bash run_corrected_pnl.sh [--check-only]\n' >&2
    # Return an explicit command-line error.
    exit 2
# Finish argument validation.
fi
# Check validated code, unchanged calibration and the complete intended history.
caffeinate -is "$python_bin" "$research_dir/verify_book_release.py"
# Recheck book and replay regressions before the long run.
# Prevent the caller's historical directory from taking precedence over PYTHONPATH.
cd "$research_dir"
# Discover tests only after entering the corrected source directory.
caffeinate -is "$python_bin" -m unittest discover -s "$research_dir" -p 'test_*.py'
# Permit verification without launching the full strategy replay.
if [[ "${1:-}" == '--check-only' ]]; then
    # End successfully after all bounded preflight checks.
    exit 0
# Finish the optional preflight-only branch.
fi
# Allocate a unique output directory so earlier P&L is never overwritten.
run_output=$(mktemp -d '/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/clean_window_refresh_v4_XXXXXXXX')
# Show the exact new evidence location before replay starts.
printf 'New P&L results: %s\n' "$run_output"
# Run all original stocks, dates and strategies with eight workers and sleep prevention.
exec caffeinate -is "$python_bin" "$research_dir/clean_window_run.py" --manifest '/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/spacing_history_oct2025_jun2026_v1/preparation/history_jobs_manifest.json' --assignment '/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/config_assignment_20260915_0043.csv' --workers 8 --output-dir "$run_output"
