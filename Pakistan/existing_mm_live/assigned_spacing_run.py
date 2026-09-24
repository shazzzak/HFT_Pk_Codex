# Parse one readable command for preparation and replay.
import argparse
# Forward only the declared replay arguments.
import sys
# Resolve external output folders.
from pathlib import Path
# Reuse saved sizing while explicitly assembling the requested configuration.
from assigned_spacing_config import prepare
# Import the additive assigned runner.
import assigned_spacing_pnl as replay

# Keep spawned processes from launching another preparation or worker pool.
def main():
    # Require the previous frozen sizing evidence and shipped assignment.
    parser = argparse.ArgumentParser()
    # Use saved sizes rather than recomputing the completed feature/sizing build.
    parser.add_argument('--source-manifest', required=True, type=Path)
    # Pin the intended three-tier configuration file.
    parser.add_argument('--assignment', required=True, type=Path)
    # Require the explicit baseline rule for previously excluded stocks.
    parser.add_argument('--drop-policy', required=True, choices=('best-recorded','qt15'))
    # Store all new evidence outside the repository.
    parser.add_argument('--output-dir', required=True, type=Path)
    # Use eight of the user's ten cores.
    parser.add_argument('--workers', type=int, default=8)
    # Allow interrupted identical runs to continue.
    parser.add_argument('--resume', action='store_true')
    # Allow a quick configuration-only inspection without historical replay.
    parser.add_argument('--prepare-only', action='store_true')
    # Read the user's explicit invocation.
    args = parser.parse_args()
    # Reject oversubscription before even preparing jobs.
    if not 1 <= args.workers <= 8:
        # Keep the declared resource budget bounded.
        parser.error('workers must be between 1 and 8')
    # Announce preparation without suggesting market replay has begun.
    print('Checking frozen sizing, assignments and actual strategy settings...', flush=True)
    # Build the complete assigned manifest under a new results directory.
    manifest = prepare(args.source_manifest, args.assignment, args.output_dir / 'preparation', args.drop_policy)
    # Configuration-only checks must never start market-data workers.
    if args.prepare_only:
        # Leave the verified manifest ready for later use.
        return
    # Forward the exact one-seed experiment to the replay entry point.
    sys.argv = [sys.argv[0], '--manifest', str(manifest), '--output-dir', str(args.output_dir / 'replay'), '--seeds', '0', '--workers', str(args.workers)]
    # Preserve the user's explicit resume choice.
    if args.resume:
        # Resume only under the runner's identical-input checks.
        sys.argv.append('--resume')
    # Execute the four-version comparison with periodic parent heartbeats.
    replay.main()

# Run only as the command-line entry point.
if __name__ == '__main__':
    # Keep multiprocessing spawn imports side-effect free.
    main()
