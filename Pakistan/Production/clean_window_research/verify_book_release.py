# Read the independently produced validation stamp before launching a long replay.
import json
# Resolve fixed evidence paths independently of the terminal's directory.
from pathlib import Path
# Reuse the validator's complete source fingerprint and import-path setup.
from validate_snapshot_book import code_hashes,ROOT,SAMPLE
# Check the original calibration and complete job universe without running strategies.
from clean_window_run import verify_inputs,jobs_from
# Check the frozen portfolio assignment independently of run scope.
from stock_search_util import digest

# Verify the exact release that passed source reconciliation and integration checks.
def main():
    # Keep the launch tied to the final generation-aware validation evidence.
    folder=ROOT/'book_reconstruction_validation_20260926_v3'
    # Require a completed validation stamp; partial attempts never qualify.
    report=json.loads((folder/'validation.json').read_text())
    # Refuse any Python source change since validation, including legacy dependencies.
    if report.get('passed') is not True or report['source_hashes']!=code_hashes():
        # Require fresh validation rather than silently launching a different implementation.
        raise ValueError('Book validation is missing or does not match current code')
    # Require the complete predeclared diagnostic sample.
    if {(r['symbol'],r['date']) for r in report['cells']}!=set(SAMPLE):
        # Prevent an incomplete sample from being mistaken for the completed check.
        raise ValueError('Incomplete source validation sample')
    # Require actual exact comparisons for every diagnostic stock-day.
    if any(r['compared_snapshots']<=0 or r['compared_snapshots']!=r['exact_matches'] for r in report['cells']):
        # Stop before spending time on a known book discrepancy.
        raise ValueError('Source reconciliation failed')
    # Require all twelve strategy arms to process all snapshots and finish flat.
    integration=report['integration']
    # Refuse incomplete execution checks or closing exclusions in this fixed fixture.
    if len(integration['arms'])!=12 or any(not r['closed'] or r['refreshes']!=integration['expected_refreshes'] for r in integration['arms'].values()):
        # Keep bounded source and execution requirements explicit.
        raise ValueError('Incomplete twelve-arm integration validation')
    # Read the original frozen full-history sizing manifest.
    manifest=json.loads((ROOT/'spacing_history_oct2025_jun2026_v1/preparation/history_jobs_manifest.json').read_text())
    # Verify calibration hashes and parsed partition identities before expensive replay.
    verify_inputs(manifest)
    # Validate all 20,905 jobs and their strictly historical clip sizing.
    jobs=jobs_from(manifest)
    # Preserve the originally specified stock assignments.
    assignment=Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/config_assignment_20260915_0043.csv')
    # Require the exact frozen assignment content.
    if digest(assignment)!=manifest['hashes'].get(str(assignment)):
        # Do not silently change the incumbent portfolio comparison.
        raise ValueError('Frozen assignment changed')
    # Explain the successful checks without claiming a full-history profit result.
    print(f'Preflight passed: validated book code; {len(jobs)} stock-days; 113 stocks; 185 dates; 12 arms.',flush=True)

# Run only when explicitly invoked by the guarded launcher.
if __name__=='__main__':
    # Propagate any failed verification as a nonzero exit status.
    main()
