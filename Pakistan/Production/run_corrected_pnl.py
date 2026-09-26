# Parse an optional verification-only request before starting expensive work.
import argparse
# Configure deterministic imports and replace this launcher with the final replay process.
import os
# Resolve installation paths without trusting the terminal's current directory.
from pathlib import Path
# Run each prerequisite in a fresh interpreter and propagate failures.
import subprocess
# Use the explicitly selected backtest Python interpreter throughout.
import sys
# Allocate a unique result directory without overwriting earlier runs.
import tempfile

# Launch the corrected research from a controlled directory and environment.
def main():
    # Support checking the package without starting the full P&L calculation.
    parser=argparse.ArgumentParser(description='Run corrected clean-window P&L with verified imports')
    # Expose bounded preflight for installation checks.
    parser.add_argument('--check-only',action='store_true')
    # Reject unknown arguments instead of silently changing the requested scope.
    args=parser.parse_args()
    # Resolve the Production installation from this file itself.
    production=Path(__file__).resolve().parent
    # Keep the corrected modules first for this and every child interpreter.
    research=production/'clean_window_research'
    # Retain read-only historical dependencies behind corrected modules.
    legacy=production.parent/'existing_mm_live'
    # Locate the user's existing results root.
    results=Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex')
    # Preserve the exact verified installation expected by the source fingerprints.
    expected=Path('/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/Production')
    # Fail clearly if the ZIP was unpacked into a different directory.
    if production!=expected:
        # Explain the required installation destination without rewriting source identities.
        raise ValueError('Unpack the ZIP into '+str(expected))
    # Copy the environment without changing unrelated user configuration.
    env=os.environ.copy()
    # Prevent legacy current-directory imports and control all child import paths.
    env.update(PYTHONPATH=str(research)+os.pathsep+str(legacy),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/source-coverage-mpl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    # Keep each verification interpreter awake and use the same Python as the launcher.
    prefix=['/usr/bin/caffeinate','-is',sys.executable]
    # Verify code hashes, frozen input data, assignments and the full requested universe.
    subprocess.run(prefix+[str(research/'verify_book_release.py')],cwd=research,env=env,check=True)
    # Crucially run module-based test discovery inside the corrected directory.
    subprocess.run(prefix+['-m','unittest','discover','-s',str(research),'-p','test_*.py'],cwd=research,env=env,check=True)
    # Permit installation verification without launching strategy work.
    if args.check_only:
        # Confirm that all subprocesses passed from the controlled import location.
        print('Installation check passed; no full P&L run started.',flush=True)
        # Leave the caller's terminal and historical results untouched.
        return
    # Create a unique destination only after all checks pass.
    output=tempfile.mkdtemp(prefix='clean_window_refresh_v4_',dir=results)
    # Tell the user where the new results will be saved.
    print('New P&L results: '+output,flush=True)
    # Keep spawned workers independent of the caller's original working directory too.
    os.chdir(research)
    # Preserve the full original scope, assignment and eight-worker setting.
    command=prefix+[str(research/'clean_window_run.py'),'--manifest',str(results/'spacing_history_oct2025_jun2026_v1/preparation/history_jobs_manifest.json'),'--assignment','/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/config_assignment_20260915_0043.csv','--workers','8','--output-dir',output]
    # Replace the launcher so signals and terminal progress belong to the real replay.
    os.execve(command[0],command,env)

# Avoid starting work if a spawned process imports this entry point.
if __name__=='__main__':
    # Run only the explicitly requested launch or installation check.
    main()
