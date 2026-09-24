# Parse one complete, reproducible terminal invocation.
import argparse
# Copy frozen job settings without changing previous evidence.
import copy
# Read the original three-tier assignment for comparator reporting only.
import csv
# Load and compare durable checkpoints.
import json
# Validate dated clip calculations.
import math
# Spawn independent stock-day workers.
import multiprocessing as mp
# Keep runtime graphics caches outside the repository.
import os
# Quote complete command arguments safely in generated resume scripts.
import shlex
# Check available space before writing a full-history fill archive.
import shutil
# Reuse the same interpreter selected by the user's invocation.
import sys
# Resolve data and evidence paths.
from pathlib import Path
# Recompute each saved ten-day median exactly.
import statistics
# Time progress and estimate the full run from a bounded pilot.
import time
# Format concise local timestamps.
from datetime import datetime
# Bind the verified twelve-grid worker.
from stock_search_cell import cell, initialize, R
# Verify every actual constructed strategy before sending jobs to workers.
from stock_search_engine import ARMS, CHEAP, make_strategy
# Pin all common requested controls.
from stock_search_contract import COMMON
# Use streaming hashes and atomic evidence writes.
from stock_search_util import digest, save
# Generate reports from saved results, not another replay.
from stock_search_reports import report

# Check inputs before work and again after it finishes.
def verify_inputs(manifest):
    # Read every recorded source/calibration identity from the completed sizing build.
    for path, expected in manifest['hashes'].items():
        # Do not accept missing or changed historical sizing dependencies.
        if not Path(path).is_file() or digest(path) != expected:
            # Name the input that needs review.
            raise ValueError('Frozen input changed: ' + path)
    # Check all raw partitions, including September sizing history.
    for path, expected in manifest['raw_metadata'].items():
        # Refuse changed raw data without silently rebuilding clips.
        if not Path(path).is_file() or [Path(path).stat().st_size, Path(path).stat().st_mtime_ns] != expected:
            # Metadata checks are not advertised as content hashes.
            raise ValueError('Frozen raw metadata changed: ' + path)
    # Check additions as well as changes to previously recorded partitions.
    folders = [R.date_dir(table, date) for date in manifest['history_dates'] + manifest['evaluation_dates'] for table in ('ob_updates', 'ob_snapshot', 'trades')]
    # Reconstruct membership without reading parquet contents.
    current = {str(path) for folder in folders for path in Path(folder).rglob('*.parquet')}
    # New same-partition input files would change what the loader sees.
    if current != set(manifest['raw_metadata']):
        # Refuse silently expanded or reduced market input.
        raise ValueError('Frozen raw partition membership changed')

# Freeze the full universe with strictly prior-date saved clip sizes.
def prepare(source):
    # Read the completed sizing evidence once.
    manifest = json.loads(Path(source).read_text())
    # Verify lineage before trusting any saved configuration.
    verify_inputs(manifest)
    # Check the exact requested universe and period.
    dates, symbols = manifest['evaluation_dates'], manifest['symbols']
    # Do not launch a partial or quietly narrowed full-history experiment.
    if len(symbols) != 113 or len(set(symbols)) != 113 or len(dates) != 185 or dates != sorted(set(dates)) or dates[0][:7] != '2025-10' or dates[-1] != '2026-06-30' or manifest['excluded']:
        # Stop before workers start under an unexpected scope.
        raise ValueError('Expected all 113 names and 185 October-June dates with no exclusions')
    # Compare exact stock/date coverage, including duplicates.
    jobs = copy.deepcopy(manifest['jobs'])
    # Require the full Cartesian scope.
    if len(jobs) != len(symbols) * len(dates) or {(j['symbol'], j['date']) for j in jobs} != {(s, d) for s in symbols for d in dates}:
        # Neither omitted losing stocks nor duplicate jobs are acceptable.
        raise ValueError('Incomplete or duplicated stock-day scope')
    # Validate every saved clip without rereading historical events.
    checked_at = time.monotonic()
    # Inspect every job's actual sizing and controls.
    for index, job in enumerate(jobs, 1):
        # Extract the dates and daily median quantities actually used.
        history = job['clip_history']
        # Rebuild the clip from ten distinct strictly earlier trading dates.
        if len(history) != 10 or len({d for d, q in history}) != 10 or any(not '2025-09-01' <= d < job['date'] or not math.isfinite(q) or q <= 0 for d, q in history) or job['clip'] != max(1, round(3 * statistics.median(q for d, q in history))):
            # Never reuse a size influenced by the current or a future date.
            raise ValueError('Invalid dated clip history: ' + job['symbol'] + ' ' + job['date'])
        # Require unchanged clip and inventory ratios in actual constructor inputs.
        if (job['params']['size'], job['params']['max_inv'], job['params']['soft_inv']) != (job['clip'], 10 * job['clip'], 3 * job['clip']):
            # Reject hidden DROP or default sizing overrides.
            raise ValueError('Saved sizing ratio mismatch')
        # Fix the one requested latency stream.
        job['seed'] = 0
        # Validate the explicit common controls before replay.
        make_strategy(job, 'OBI_best', (job['params']['session_segments'][0][0], job['params']['session_segments'][-1][1]))
        # Keep configuration preparation visible without per-job printing.
        if time.monotonic() - checked_at >= 15:
            # Report bounded preparation progress.
            print(f'[{datetime.now():%H:%M:%S}] Config checks | Done {index}/{len(jobs)}', flush=True)
            # Restart the same fifteen-second throttle.
            checked_at = time.monotonic()
    # Return all jobs; assignment labels do not restrict this search.
    return manifest, jobs

# Select a bounded timing pilot before committing to another long run.
def pilot_jobs(jobs):
    # Include cheap-name, previously OBI, liquid and thinner-book examples.
    symbols = ('KEL', 'TPL', 'MLCF', 'NRL', 'WTL', 'PACE', 'HBL', 'AGP')
    # Spread the small sample across the entire requested period.
    dates = sorted({j['date'] for j in jobs})
    # Pair one different historical date with each pilot stock.
    chosen = {(s, dates[round(i * (len(dates) - 1) / (len(symbols) - 1))]) for i, s in enumerate(symbols)}
    # Retain complete sessions and all twelve configurations per pilot cell.
    result = [j for j in jobs if (j['symbol'], j['date']) in chosen]
    # Fail if an expected pilot stock is absent rather than benchmarking less work.
    if len(result) != 8:
        # Preserve the declared timing sample.
        raise ValueError('Pilot coverage missing')
    # Return a deterministic, profit-independent pilot.
    return result

# Run checkpointed cells with one parent heartbeat every fifteen seconds.
def execute(jobs, output, workers):
    # Reuse only successful, internally matching durable cell evidence.
    remaining, resumed = [], 0
    # Inspect completed summaries without loading their fill tables.
    for job in jobs:
        # Match the worker's exact naming convention.
        key = f"{job['symbol']}_{job['date']}_seed0"
        # Locate a previous checkpoint under this verified run identity.
        path = output / (key + '.json')
        # Read only existing evidence.
        old = json.loads(path.read_text()) if path.exists() else None
        # Refuse changed settings under the same cell identity.
        if old is not None and old['job'] != json.loads(json.dumps(job)):
            # Never mix different configurations in an apparently resumed run.
            raise ValueError('Checkpoint job mismatch: ' + key)
        # Require complete arm evidence and all non-aliased compressed fill files.
        valid = old is not None and old.get('passed') and set(old['arms']) == set(ARMS) and (output / key / 'arms.json').exists()
        # Check arm checkpoint agreement before adopting the cell.
        if valid:
            # Missing raw fill evidence invalidates reuse of a supposedly complete cell.
            valid = json.loads((output / key / 'arms.json').read_text()) == old['arms'] and all((output / key / (arm + '_fills.csv.gz')).is_file() and digest(output / key / (arm + '_fills.csv.gz')) == r['fill_file_sha256'] for arm, r in old['arms'].items() if 'reused_from' not in r)
        # Count fully reusable cells without replaying them.
        if valid:
            # Retain completed work across pilot/full and interrupted runs.
            resumed += 1
        # Regenerate incomplete cells without dropping them from the denominator.
        else:
            # Keep the original frozen job for this replay.
            remaining.append(job)
    # Use isolated spawn workers, never fork a mutated strategy.
    context = mp.get_context('spawn')
    # Measure only this invocation's additional work.
    started, completed = time.monotonic(), resumed
    # Track errors instead of continuing a long run after a systematic failure.
    failures = []
    # Share compact progress, not full events or fills.
    with context.Manager() as manager:
        # Use one progress record per submitted cell.
        progress = manager.dict()
        # Create at most the requested eight processes.
        with context.Pool(workers, initializer=initialize, initargs=(progress,)) as pool:
            # Bound queued work to avoid retaining thousands of futures.
            queue, pending = iter(remaining), []
            # Submit the first small batch.
            for job in remaining[:workers]:
                # Advance the shared iterator once per submitted job.
                next(queue)
                # Start one full-session comparison.
                pending.append(pool.apply_async(cell, ((job, str(output)),)))
            # Print the first loading heartbeat immediately.
            last_print = started - 15
            # Collect completed cells and refill available worker slots.
            while pending:
                # Iterate a copy because ready tasks are removed.
                for future in pending[:]:
                    # Never block on a worker that has not finished.
                    if future.ready():
                        # Surface unexpected worker exceptions normally.
                        result = future.get()
                        # Remove this completed task.
                        pending.remove(future)
                        # Advance the completed-cell counter.
                        completed += 1
                        # Stop new submissions after any cell fails.
                        if not result['passed']:
                            # Retain the exact failure for the user.
                            failures.append(result)
                        # Refill only while the run remains clean.
                        if not failures:
                            # Get the next frozen job without expanding the scope.
                            job = next(queue, None)
                            # Stop adding tasks at the fixed endpoint.
                            if job is not None:
                                # Preserve bounded concurrency.
                                pending.append(pool.apply_async(cell, ((job, str(output)),)))
                # Use a single parent status line at a measured cadence.
                if time.monotonic() - last_print >= 15:
                    # Include running arm/event fractions for an earlier useful ETA.
                    state = dict(progress)
                    # Count completed work only once.
                    done = resumed + sum(v[1] for v in state.values())
                    # Measure real elapsed minutes.
                    elapsed = (time.monotonic() - started) / 60
                    # Estimate from this invocation's measured progress, not old resumed work.
                    eta = f'~{elapsed * (len(jobs) - done) / (done - resumed):.1f}m' if done > resumed else 'estimating'
                    # Show one currently active cell's identity and configuration.
                    active = next(((k, v[0]) for k, v in state.items() if v[0] != 'done'), ('loading', 'loading'))
                    # Keep status compact and timestamped.
                    print(f'[{datetime.now():%H:%M:%S}] {active[0]} | {active[1]} | Elapsed {elapsed:.1f}m | ETA {eta} | Done {completed}/{len(jobs)} | Replay {100 * done / len(jobs):.0f}%', flush=True)
                    # Restart the print interval.
                    last_print = time.monotonic()
                # Avoid busy-polling completed workers.
                time.sleep(.2)
    # Preserve failures before raising an error.
    if failures:
        # Save a concise diagnostic outside the repository.
        save(output / 'failures.json', failures)
        # Stop without publishing a partial comparison as complete.
        raise ValueError('Replay failed; see failures.json. Completed cells are preserved.')
    # Return timing with the exact count of newly replayed cells.
    return dict(seconds=time.monotonic() - started, new_cells=len(remaining), reused_cells=resumed)

# Prepare, benchmark or execute the complete replacement search.
def main():
    # Keep all parameters in one copy/paste invocation.
    parser = argparse.ArgumentParser()
    # Reuse saved September-warmed historical clips.
    parser.add_argument('--source-manifest', required=True, type=Path)
    # Use the recorded assignment solely for the incumbent comparator.
    parser.add_argument('--assignment', required=True, type=Path)
    # Store all evidence externally.
    parser.add_argument('--output-dir', required=True, type=Path)
    # Use eight of ten available cores.
    parser.add_argument('--workers', type=int, default=8)
    # Measure a small full-session pilot before the expensive run.
    parser.add_argument('--pilot-only', action='store_true')
    # Permit identical-input reuse only when explicitly requested.
    parser.add_argument('--resume', action='store_true')
    # Allow configuration checks without replaying market history.
    parser.add_argument('--prepare-only', action='store_true')
    # Read the requested invocation.
    args = parser.parse_args()
    # The priced-book repair does not establish snapshot/tick synchronization.
    if not args.pilot_only and not args.prepare_only:
        # Stop before reading historical partitions or launching workers.
        parser.error('Full history blocked: snapshot/tick chronology remains unresolved; this package repairs unpriced aggregate leakage only')
    # Refuse accidental oversubscription.
    if not 1 <= args.workers <= 8:
        # Explain the permitted range.
        parser.error('workers must be 1..8')
    # Announce cheap configuration checks before any historical work begins.
    print(f'[{datetime.now():%H:%M:%S}] Checking saved clips and all-stock configuration...', flush=True)
    # Load and validate all saved stock-days once.
    manifest, jobs = prepare(args.source_manifest)
    # Read assignment labels without live overrides or profit-driven exclusions.
    with args.assignment.open() as stream:
        # Preserve the shipped symbol-to-label mapping for comparison.
        records = list(csv.DictReader(stream))
    # Reject missing or duplicate assignments.
    if len(records) != len(manifest['symbols']) or {r['symbol'] for r in records} != set(manifest['symbols']):
        # Do not label a different universe the incumbent portfolio.
        raise ValueError('Assignment coverage mismatch')
    # Use the actual assignment CSV label column.
    assignment = {r['symbol']: r['assigned_config'] for r in records}
    # Reject unknown labels before replay starts.
    if not set(assignment.values()) <= {'OBI', 'QT_2t@15', 'QT_2t@20', 'DROP'}:
        # Do not guess a configuration for unknown labels.
        raise ValueError('Unexpected assignment labels')
    # Freeze all actual local sources plus assignment and sizing evidence.
    sources = sorted(Path(__file__).parent.glob('*.py')) + [args.source_manifest, args.assignment]
    # Keep runtime settings and inputs in a resume identity independent of pilot/full mode.
    identity = dict(hashes={str(p.resolve()): digest(p) for p in sources}, common=COMMON, arms=ARMS, raw_metadata=manifest['raw_metadata'], seed=0)
    # Normalize tuples to the same JSON representation used on disk.
    identity = json.loads(json.dumps(identity))
    # Resolve the durable identity path.
    frozen = args.output_dir / 'search_identity.json'
    # Never adopt arbitrary old results without explicit identical-input resume.
    if args.output_dir.exists() and (not args.resume or not frozen.is_file()):
        # Preserve existing output rather than overwriting it.
        raise ValueError('Use a new output folder, or --resume for this exact search')
    # Refuse mixed-source or mixed-setting resumptions.
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        # Keep old evidence intact.
        raise ValueError('Resume identity changed; do not mix runs')
    # Create the external evidence directory once checks pass.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep matplotlib font-cache writes outside source control.
    os.environ.setdefault('MPLCONFIGDIR', str(args.output_dir / 'matplotlib_cache'))
    # Save the exact experiment identity before any worker is created.
    save(frozen, identity)
    # Preserve explicit methodological limits rather than claim causal calibrations.
    qualification = manifest['calibration_status'] + ' Twelve-grid search uses these same retrospective calibrations. Monthly choices use previous months only, but this does NOT remove calibration/universe look-ahead. One seed; no independent validation.'
    # Freeze the complete scope and the component/selection policy.
    save(args.output_dir / 'search_plan.json', dict(stock_days=len(jobs), naive_replays=len(jobs) * 12, distinct_replays=sum(4 if j['symbol'] in CHEAP else 12 for j in jobs), common=COMMON, arms=ARMS, assignment=assignment, qualification=qualification, midpoint='pre-fill two-sided book; missing values explicitly unattributed', bucket='opening fill time: first15, middle, final60-to-final15 preclose45, last15', inventory='Full entry-to-close P&L of EOD-held lots, split book walk and unfilled haircut mark; fees separate', selection='previous3months; >=40dates; >=2positive months; positive net; zero residual/unattributed shares'))
    # Give the user complete resume commands without separate parameters to assemble.
    for filename, pilot in (('resume_pilot.sh', True), ('run_full_after_review.sh', False)):
        # Keep every path explicit and preserve the current interpreter and worker count.
        command = ['caffeinate', '-is', sys.executable, '-u', str(Path(__file__).resolve()), '--source-manifest', str(args.source_manifest.resolve()), '--assignment', str(args.assignment.resolve()), '--output-dir', str(args.output_dir.resolve()), '--workers', str(args.workers), '--resume']
        # Preserve pilot scope in the pilot-resume script.
        if pilot:
            # Never turn an interrupted pilot into a full run accidentally.
            command.append('--pilot-only')
        # Write safely quoted bash that is run only by the user.
        (args.output_dir / filename).write_text('#!/bin/bash\n# Stop if the working directory cannot be selected.\ncd ' + shlex.quote(str(Path(__file__).resolve().parent)) + ' || exit 1\n# Resume only this frozen run; suppress bytecode and prevent sleep.\nPYTHONDONTWRITEBYTECODE=1 ' + shlex.join(command) + '\n')
    # Configuration-only mode must not start historical workers.
    if args.prepare_only:
        # Confirm exact workload without implying a successful replay.
        print(f'Prepared {len(jobs)} stock-days, 12 configurations, seed 0. No replay started.', flush=True)
        # Leave the manifest ready for explicit resume.
        return
    # Run only the requested pilot or the complete universe/date grid.
    selected = pilot_jobs(jobs) if args.pilot_only else jobs
    # Place fills and detailed replay evidence under one persistent subdirectory.
    replay = args.output_dir / 'replay'
    # Reuse successful pilot cells during the full run.
    replay.mkdir(exist_ok=True)
    # Require a reviewed, accounting-complete pilot before a full-history launch.
    if not args.pilot_only:
        # Locate the previous pilot or interrupted full-run summary.
        pilot_summary = replay / 'summary.json'
        # Full execution must explicitly reuse this same identity-checked search.
        if not args.resume or not pilot_summary.is_file():
            # Prevent an accidental multi-day run before timing and accounting checks.
            raise ValueError('Run --pilot-only first, then resume this same folder without --pilot-only')
        # Read the completed pilot's qualifications.
        previous = json.loads(pilot_summary.read_text())
        # An incomplete midpoint decomposition must be reviewed before spending full-run compute.
        if not previous.get('completed') or not previous.get('attribution_complete'):
            # Preserve pilot fills for diagnosis without automatically launching the full history.
            raise ValueError('Pilot accounting is incomplete; inspect summary.json before full history')
        # Reserve margin above the pilot's approximate compressed-fill footprint.
        estimated_bytes = previous.get('rough_full_fill_gib', 0) * 1024 ** 3
        # Fail before a multi-day run that is already likely to exhaust storage.
        if shutil.disk_usage(args.output_dir).free < estimated_bytes * 1.5 + 1024 ** 3:
            # Preserve existing evidence and explain the concrete resource shortfall.
            raise ValueError('Insufficient free space for estimated fills plus 50% margin and 1 GiB reports')
    # Execute bounded workers and preserve all completed evidence.
    timing = execute(selected, replay, args.workers)
    # Recheck every frozen input after worker completion.
    verify_inputs(manifest)
    # Detect source changes made while workers were running.
    if any(not Path(p).is_file() or digest(p) != h for p, h in identity['hashes'].items()):
        # Do not publish a clean aggregate from changed code.
        raise ValueError('Source/input changed during replay')
    # Report from saved results without replaying original market data.
    summary = report(replay, selected, assignment, qualification)
    # Keep the pilot denominator and full-history completion distinct.
    summary.update(scope='pilot' if args.pilot_only else 'full', full_history_completed=not args.pilot_only, timing=timing)
    # Estimate full runtime only from new measured pilot work.
    if args.pilot_only:
        # This small sample has heterogeneous stocks; report an estimate, not a promise.
        pilot_results = [json.loads((replay / f"{j['symbol']}_{j['date']}_seed0.json").read_text()) for j in selected]
        # Normalize cheap-name duplicate removal rather than assuming equal cell costs.
        seconds_per_replay = sum(r['elapsed_seconds'] for r in pilot_results) / sum(r['distinct_replays'] for r in pilot_results)
        # Convert estimated total worker time to wall time at the same worker count.
        summary['rough_full_run_days'] = seconds_per_replay * sum(4 if j['symbol'] in CHEAP else 12 for j in jobs) / args.workers / 86400
        # Explicitly qualify a small heterogeneous sample and sustained machine load.
        summary['timing_qualification'] = 'Eight-stock sample; approximate scaling at the same workers. Event counts, disk contention and thermal throttling can materially change runtime.'
        # Measure compressed fill bytes actually written by this pilot.
        fill_bytes = sum((replay / f"{j['symbol']}_{j['date']}_seed0" / (arm + '_fills.csv.gz')).stat().st_size for j, result in zip(selected, pilot_results) for arm, value in result['arms'].items() if 'reused_from' not in value)
        # Estimate full storage using distinct replays, not aliased logical arms.
        summary['rough_full_fill_gib'] = fill_bytes / sum(r['distinct_replays'] for r in pilot_results) * sum(4 if j['symbol'] in CHEAP else 12 for j in jobs) / 1024 ** 3
    # Save final scope qualification after report construction.
    save(replay / 'summary.json', summary)
    # Print a compact completion statement and point to detailed reports.
    print(json.dumps({k: summary[k] for k in ('completed', 'scope', 'stock_days', 'attribution_complete', 'full_history_completed', 'timing')} | {'rough_full_run_days': summary.get('rough_full_run_days'), 'reports': str(replay)}), flush=True)

# Do not launch another pool when multiprocessing imports this file.
if __name__ == '__main__':
    # Enter only from the user's explicit command.
    main()
