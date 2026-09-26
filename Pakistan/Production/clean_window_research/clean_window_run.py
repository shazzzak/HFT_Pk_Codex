# Parse an explicit research scope and output directory.
import argparse
# Read immutable sizing and checkpoint evidence.
import json
# Read the authoritative per-stock assignment CSV.
import csv
# Validate the dated clip calculation independently.
import statistics
# Reject nonfinite sizing inputs.
import math
# Freeze all current local Python source identities.
from pathlib import Path
# Spawn workers without inheriting a mutated trading account.
import multiprocessing as mp
# Wait for progress while keeping one parent heartbeat.
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
# Measure elapsed time and observed-work ETA.
import time
# Format local heartbeat timestamps.
from datetime import datetime
# Read canonical paths from this checkout only.
from config_pk import PARSED_ROOT
# Screen channel gaps before any individual-symbol replay.
from clean_window_data import channel_intervals
# Dispatch checkpointed independent stock-days.
from clean_window_cell import cell,initialize
# Assert all twelve real strategy configurations before a long run.
from stock_search_engine import ARMS,make_strategy
# Save strict atomic evidence and hashes.
from stock_search_util import save,digest
# Build summaries and plots from saved results without replaying them again.
from clean_window_reports import report

# Hash both the corrected local implementation and its read-only legacy dependencies.
def source_identity():
    # Resolve source roots without depending on the shell's working directory.
    roots=(Path(__file__).resolve().parent,Path(__file__).resolve().parents[2]/'existing_mm_live')
    # Record all Python sources conservatively, including imported legacy helpers.
    return {str(path):digest(path) for root in roots for path in sorted(root.glob('*.py'))}

# Validate original calibration and data lineage, explicitly superseding only old code hashes.
def verify_inputs(manifest):
    # Saved Python hashes describe the old engine, not the corrected research implementation.
    for path,expected in manifest['hashes'].items():
        # Preserve every non-code calibration, assignment and sizing dependency unchanged.
        if Path(path).suffix!='.py' and digest(path)!=expected:
            # Refuse a hidden calibration change.
            raise ValueError('Frozen calibration changed: '+path)
    # Check every original raw partition, including prior-date clip-sizing history.
    for path,expected in manifest['raw_metadata'].items():
        # Refuse an incomplete or partially replaced parsed store.
        if not Path(path).is_file() or [Path(path).stat().st_size,Path(path).stat().st_mtime_ns]!=expected:
            # Do not blend old and new parser outputs in one experiment.
            raise ValueError('Frozen raw metadata changed: '+path)
    # Detect added files as well as replaced or deleted existing files.
    current={str(p) for day in manifest['history_dates']+manifest['evaluation_dates'] for table in ('ob_updates','trades','ob_snapshot') for p in (PARSED_ROOT/table/('date='+day)).rglob('*.parquet')}
    # Membership must match the saved sizing build exactly.
    if current!=set(manifest['raw_metadata']):
        # An expanded partition would otherwise silently change replay inputs.
        raise ValueError('Frozen raw partition membership changed')

# Validate complete scope and strictly prior-date production clip history.
def jobs_from(manifest):
    # Retain every stock, including formerly unprofitable and DROP assignments.
    jobs=manifest['jobs']
    # Require the requested all-name October-to-June universe.
    if len(manifest['symbols'])!=113 or len(manifest['evaluation_dates'])!=185 or manifest['excluded'] or len(jobs)!=20905:
        # Stop before a silently narrowed full-history comparison.
        raise ValueError('Expected 113 stocks by 185 dates, with no saved exclusions')
    # Prevent duplicate jobs or missing requested stock-days.
    if {(j['symbol'],j['date']) for j in jobs}!={(s,d) for s in manifest['symbols'] for d in manifest['evaluation_dates']}:
        # Coverage is part of the run contract.
        raise ValueError('Duplicated or missing stock-days')
    # Validate every job before selecting even the timing pilot.
    for job in jobs:
        # Recompute the frozen trailing-ten-day size from saved evidence.
        history=job['clip_history']
        # Reject same-day, future, duplicate or corrupt sizing observations.
        if len(history)!=10 or len({d for d,q in history})!=10 or any(not '2025-09-01'<=d<job['date'] or not math.isfinite(q) or q<=0 for d,q in history) or job['clip']!=max(1,round(3*statistics.median(q for d,q in history))):
            # Do not use an approximate replacement clip.
            raise ValueError('Invalid clip history: '+job['symbol']+' '+job['date'])
        # Preserve the requested static inventory multiples.
        if (job['params']['size'],job['params']['max_inv'],job['params']['soft_inv'])!=(job['clip'],10*job['clip'],3*job['clip']):
            # Expose hidden parameter overrides.
            raise ValueError('Invalid size ratios')
        # Use one latency seed as requested.
        job['seed']=0
    # Return all validated jobs unchanged apart from the explicit seed.
    return jobs

# Format one concise heartbeat for both source screening and replay.
def heartbeat(start,phase,done,total,progress=None):
    # Use observed work rather than an unexplained perpetual pending label.
    elapsed=time.monotonic()-start
    # Count partial cells only as a progress estimate, never completed evidence.
    active=[(key,value) for key,value in (progress or {}).items() if value[0]!='done']
    # Accumulate fractions for the worker whose state is currently visible.
    fraction=done+sum(value[1] for key,value in active)
    # Provide an estimate once there is any measured work.
    eta=f'~{elapsed/max(fraction,1e-9)*max(0,total-fraction)/60:.1f}m' if fraction>0 else 'estimating (no completed work yet)'
    # Keep the current instrument and arm visible without printing every worker.
    label=active[0][0]+' | '+active[0][1][0] if active else phase
    # Print one compact parent-owned line.
    print(f'[{datetime.now():%H:%M:%S}] {label} | Elapsed {elapsed/60:.1f}m | ETA {eta} | Done {done}/{total}',flush=True)

# Adapt one channel-screen task to the process-pool interface.
def screen(task):
    # Each date is scanned once across all names.
    return channel_intervals(*task)

# Dispatch a bounded number of workers and retain periodic progress.
def collect(pool,fn,tasks,start,phase,progress=None):
    # Submit independent tasks to the same fixed worker pool.
    pending={pool.submit(fn,task) for task in tasks}
    # Preserve results without printing a line for each completion.
    results=[]
    # Establish a single fifteen-second terminal cadence.
    last=0.0
    # Wait until every submitted task has returned.
    while pending:
        # Wake at least once a second so the heartbeat remains responsive.
        ready,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
        # Propagate worker failures instead of silently dropping cells.
        for future in ready:
            # Preserve the completed worker's diagnostic before evaluating success.
            result=future.result()
            # Stop a long run on an implementation failure, not on ordinary excluded data.
            if isinstance(result,dict) and result.get('passed') is False:
                # Cancel work that has not started instead of wasting a whole overnight run.
                for outstanding in pending:
                    # Running workers may finish their bounded current cells.
                    outstanding.cancel()
                # Release pending tasks while preserving completed disk evidence.
                pool.shutdown(wait=True,cancel_futures=True)
                # Surface the exact failed cell immediately.
                raise RuntimeError('Replay stopped after failed cell: '+str(result))
            # Keep successful completed results for progress counting.
            results.append(result)
        # Keep console writes bounded independently of worker activity.
        if time.monotonic()-last>=15:
            # Show measured partial progress and an estimated remaining duration.
            heartbeat(start,phase,len(results),len(tasks),dict(progress) if progress is not None else None)
            # Restart the print interval.
            last=time.monotonic()
    # Return all results for durable reporting.
    return results

# Run only from an explicit command, so spawned children never launch another pool.
def main():
    # Require explicit input and output evidence paths.
    parser=argparse.ArgumentParser(description='Retrospective clean-window research; NOT full-day live P&L')
    # Use the existing dated clip-sizing manifest.
    parser.add_argument('--manifest',required=True)
    # Keep the intended three-tier incumbent independent of old sizing-manifest labels.
    parser.add_argument('--assignment',required=True)
    # Keep all generated evidence outside the source repository.
    parser.add_argument('--output-dir',required=True)
    # Preserve the user's eight-worker preference on ten cores.
    parser.add_argument('--workers',type=int,default=8)
    # Offer a complete single-date scope without changing default full-history coverage.
    parser.add_argument('--date')
    # Offer the same small deterministic eight-cell timing pilot as before.
    parser.add_argument('--pilot',action='store_true')
    # Read the declared experiment scope once.
    args=parser.parse_args()
    # Reject nonsensical or contradictory scope requests.
    if args.workers<1 or (args.date and args.pilot):
        # Stop before any source scan.
        parser.error('Use positive workers and either --date or --pilot')
    # Load completed sizing evidence without rebuilding features or median clips.
    manifest=json.loads(Path(args.manifest).read_text())
    # Verify immutable calibration and raw-store membership.
    verify_inputs(manifest)
    # Validate the full requested universe and sizing rules.
    jobs=jobs_from(manifest)
    # The assignment CSV must be a verified dependency of this sizing evidence.
    if str(Path(args.assignment).resolve()) not in manifest['hashes'] or digest(args.assignment)!=manifest['hashes'][str(Path(args.assignment).resolve())]:
        # Refuse a different or modified incumbent portfolio.
        raise ValueError('Assignment is not the frozen calibration input')
    # Read actual per-stock assigned configurations without guessing from clip-manifest labels.
    with Path(args.assignment).open() as stream:
        # Retain all original records to detect duplicate names.
        records=list(csv.DictReader(stream))
    # Construct the authoritative stock-to-style map.
    assignment={row['symbol']:row['assigned_config'] for row in records}
    # Require complete, unique assignment coverage and known style names.
    if len(assignment)!=len(records) or not set(manifest['symbols'])<=set(assignment) or not set(assignment.values())<={'OBI','QT_2t@15','QT_2t@20','DROP'}:
        # Stop before a uniform or incomplete portfolio can be reported.
        raise ValueError('Invalid assignment coverage or labels')
    # Replace reporting labels only; all twelve research arms still run on every name.
    for job in jobs:
        # Preserve the superseded sizing-manifest label for audit.
        job['sizing_manifest_assignment']=job['assignment']
        # Use the actual assigned portfolio in the comparator.
        job['assignment']=assignment[job['symbol']]
    # Preserve an explicit single-day request if provided.
    if args.date:
        # Keep all 113 names for the requested date.
        jobs=[job for job in jobs if job['date']==args.date]
    # Choose the existing eight-cell sample independent of profit.
    if args.pilot:
        # Import only the deterministic sample selector, not its blocked full-run path.
        from stock_search_run import pilot_jobs
        # Preserve the established pilot's distribution across names and dates.
        jobs=pilot_jobs(jobs)
    # An unsupported date must not produce an empty success.
    if not jobs:
        # Explain the scope error before launching workers.
        raise ValueError('No jobs in requested scope')
    # Assert actual constructed controls for every logical candidate on each requested name.
    for job in {j['symbol']:j for j in jobs}.values():
        # Verify every arm, including weighted variants of the OBI style.
        for arm in ARMS:
            # Constructor assertions cover sizing, protection and cap settings.
            make_strategy(job,arm,(job['params']['session_segments'][0][0],job['params']['session_segments'][-1][1]))
    # Freeze the corrected implementation separately from superseded historical code hashes.
    source=source_identity()
    # Keep resume compatibility tied to source, scope and sizing evidence.
    identity=dict(assignment_sha256=digest(args.assignment),manifest_sha256=digest(args.manifest),source=source,jobs=[(j['symbol'],j['date']) for j in jobs],experiment='retrospective_clean_windows_snapshot_refresh_v4',seed=0,arms=list(ARMS),min_window_ms=60000,cancel_buffer_ms=5000,exit_send_buffer_ms=2000,snapshot_refresh='every validated causal checkpoint; account and order state retained',queue_refresh='known arrivals preserve priority; anonymous snapshot quantity conservatively ahead; out-of-depth live queues fail explicitly')
    # Create the user's explicitly supplied evidence directory.
    output=Path(args.output_dir)
    # Keep all reports and fill archives together.
    output.mkdir(parents=True,exist_ok=True)
    # Compare serialized identities consistently across process restarts.
    identity=json.loads(json.dumps(identity))
    # Refuse to resume old code or another scope into this run.
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        # Ask for a new output directory rather than overwrite evidence.
        raise ValueError('Output directory belongs to different code or scope')
    # Freeze the corrected experiment identity before any replay.
    save(output/'identity.json',identity)
    # Preserve current settings and retrospective qualifications explicitly.
    save(output/'run_contract.json',dict(identity=identity,calibration_status=manifest['calibration_status'],live_approved=False,full_day_pnl=False,window_end_known_in_advance=True,flat_start_each_window=True,source_snapshot_alignment='quiet source-second cutovers and causal order-generation linkage; no common exchange watermark',symbol_staleness_replaced_by='full-channel 7-second capture pause and application-sequence screening',inventory_capacity='original volume profile restricted to each window exit time',closing='delayed IOC at reported prices; failed closes exclude the same window across every arm'))
    # Use isolated spawn workers to avoid cross-arm state leakage.
    context=mp.get_context('spawn')
    # Keep the channel screen separate from strategy execution.
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=context) as pool:
        # Scan only distinct requested dates, once each.
        dates=sorted({j['date'] for j in jobs})
        # Reuse a completed screen only under this verified run identity and raw metadata.
        quality=json.loads((output/'channel_quality.json').read_text()) if (output/'channel_quality.json').exists() else None
        # Perform the full channel screen when no matching durable result exists.
        if quality is None:
            # Keep periodic screen progress visible even before trading begins.
            values=collect(pool,screen,[(str(PARSED_ROOT),day) for day in dates],time.monotonic(),'channel screen')
            # Key by date for shared access across all stocks.
            quality={value['date']:value for value in values}
            # Preserve sequence-gap exclusions separately from strategy outcomes.
            save(output/'channel_quality.json',quality)
    # Preserve completed cells rather than repeating their replay after interruption.
    remaining=[]
    # Validate each existing cell's frozen job and compressed fill evidence.
    for job in jobs:
        # Match the worker's output naming.
        folder=output/(job['symbol']+'_'+job['date'])
        # Load a prior completed result if available.
        old=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
        # Resume only complete successful evidence with unmodified fills and settings.
        if old is None or not old.get('passed') or old['job']!=job or not (folder/'fills.csv.gz').exists() or digest(folder/'fills.csv.gz')!=old['fill_sha256']:
            # Failed or interrupted cells are rerun; no stock is silently omitted.
            remaining.append((job,quality[job['date']],str(output)))
    # Keep worker state local to this run and print only from the parent.
    with context.Manager() as manager:
        # Publish a compact ticker/arm progress state.
        progress=manager.dict()
        # Spawn the requested worker count for remaining cells only.
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,initializer=initialize,initargs=(progress,)) as pool:
            # Complete every requested remaining stock-day.
            results=collect(pool,cell,remaining,time.monotonic(),'replay',progress) if remaining else []
    # Verify both source and raw data remained fixed for the entire computation.
    verify_inputs(manifest)
    # A mid-run code install invalidates comparability even if workers finished.
    if source!=source_identity():
        # Preserve results but refuse a complete status.
        raise ValueError('Source changed during replay; results require review')
    # Build all requested money components, coverage tables and plots from saved cells.
    summary=report(output,jobs)
    # Print one compact final machine-readable result.
    print(json.dumps(summary),flush=True)
    # Programming/source-contract failures must produce a nonzero terminal status.
    if summary['failed_stock_days']:
        # Do not leave a failed full experiment looking successful to shell chaining.
        raise SystemExit(1)

# Spawn-safe entry point.
if __name__=='__main__':
    # Run the requested experiment once in the parent process.
    main()
