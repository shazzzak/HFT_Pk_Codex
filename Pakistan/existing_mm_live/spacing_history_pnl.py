# Parse the reproducible pilot command.
import argparse
# Copy mutable strategy settings and snapshots between arms.
import copy
# Fingerprint source and decoded inputs.
import hashlib
# Read manifests and write evidence.
import json
# Validate arithmetic and calculate distance weights.
import math
# Use isolated workers for independent stock-days.
import multiprocessing as mp
# Resolve installed source and output paths.
from pathlib import Path
# Serialize normalized inputs for checksums.
import pickle
# Time the single parent heartbeat.
import time
# Format local clock timestamps.
from datetime import datetime, timezone
# Import the original strategy without modifying it.
from micro_mm import MicrostructureMM as Original
# Import the isolated strategy copy with one queue-signal hook.
from spacing_pnl_micro import MicrostructureMM as Candidate
# Reuse the existing simulator and latency distribution.
from mm_backtest import Backtester, LatencyModel
# Reuse the original event builder and configuration.
import run_legacy_mm as R
# Use the canonical parsed-data root.
from config_pk import PARSED_ROOT
# Keep the loader rooted in this project's data configuration.
R.PARSED_ROOT = PARSED_ROOT

# Fix every research arm before examining the results.
ARMS = {'baseline': None, 'copy_control': None, 'distance_3': (3, 0.1), 'distance_4': (4, 0.1), 'distance_5': (5, 0.1), 'equal_3': (3, 0.0)}

# Fingerprint a file without loading it all into memory.
def digest(path):
    # Initialize the content checksum.
    h = hashlib.sha256()
    # Read the file as bytes.
    with Path(path).open('rb') as stream:
        # Bound memory per read.
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            # Include every byte.
            h.update(chunk)
    # Return a portable checksum.
    return h.hexdigest()

# Publish evidence atomically.
def save(path, value):
    # Keep partially written JSON out of the report directory.
    tmp = Path(str(path) + '.tmp')
    # Refuse non-finite JSON numbers.
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    # Replace only this run's own evidence file.
    tmp.replace(path)

# Compute selected-depth price-distance bid share, falling back to the unchanged signal.
def distance_share(bids, asks, bb, ba, tick, fallback, depth=3, decay=0.1):
    # Reject unsupported configurations rather than silently changing the experiment.
    if depth not in (2, 3, 4, 5) or not math.isfinite(decay) or decay < 0:
        # Invalid parameters are programmer errors.
        raise ValueError('depth must be 2..5 and decay must be finite and nonnegative')
    # Require the declared number of known levels on both sides and a valid tick.
    if len(bids) < depth or len(asks) < depth or not math.isfinite(tick) or tick <= 0:
        # Preserve the original decision on incomplete books.
        return fallback, False
    # Require known-price depth to agree with the simulator's touch.
    if bb is None or ba is None or bb >= ba or abs(bids[0][0]-bb) > tick*1e-6 or abs(asks[0][0]-ba) > tick*1e-6:
        # Avoid substituting a different touch when opaque depth matters.
        return fallback, False
    # Accumulate one weighted volume for each side.
    volumes = []
    # Measure distances away from each side's own best quote.
    for levels, touch, sign in ((bids[:depth], bb, -1), (asks[:depth], ba, 1)):
        # Start the side's weighted sum.
        total = 0.0
        # Remember the prior level to reject malformed ordering.
        previous = -1.0
        # Examine exactly the declared number of quoted prices.
        for price, qty in levels:
            # Convert price distance to ticks.
            distance = sign * (price - touch) / tick
            # Require finite, positive quantities and distinct ordered tick levels.
            if not all(math.isfinite(x) for x in (price, qty, distance)) or qty <= 0 or distance <= previous or abs(distance-round(distance)) > 1e-5:
                # Retain the current strategy signal when data are invalid.
                return fallback, False
            # Discount quantities by actual price distance, with the declared decay (zero means equal weights).
            total += qty * math.exp(-decay * max(0.0, distance))
            # Advance the ordering check.
            previous = distance
        # Retain the side's weighted volume.
        volumes.append(total)
    # Reject overflow instead of emitting a non-finite signal.
    if not math.isfinite(sum(volumes)) or sum(volumes) <= 0:
        # Use the unchanged signal.
        return fallback, False
    # Preserve the existing strategy's zero-to-one bid-share convention.
    return volumes[0] / sum(volumes), True

# Receive shared progress storage in each spawned worker.
def initialize(progress):
    # Keep the proxy private to the worker process.
    global PROGRESS
    # Retain the parent-owned progress dictionary.
    PROGRESS = progress

# Replay one matched stock-day using the original event stream.
def cell(task):
    # Unpack immutable job settings and output location.
    job, output = task
    # Name this evidence directory deterministically.
    key = job['symbol'] + '_' + job['date'] + '_seed' + str(job['seed'])
    # Preserve failures as part of the requested denominator.
    try:
        # Show loading while parquet decoding is in progress.
        PROGRESS[key] = ('loading', 0.0)
        # Open original parsed exchange data.
        datasets = R.open_datasets(job['date'])
        # Fail explicitly if a requested partition is absent.
        if datasets is None:
            # Never silently drop a requested day.
            raise ValueError('Missing parsed date')
        # Load REG rows from all three tables, requiring the market column.
        frames = [R.read_symbol(datasets[name], list(dict.fromkeys(cols + ['market'])), job['symbol'], market='REG') for name, cols in (('ob_updates', R.REQ_UPDATES), ('ob_snapshot', R.REQ_SNAP), ('trades', R.REQ_TRADES))]
        # Use the unchanged canonical event builder.
        events, snapshots, trades = R.build_events(*frames)
        # Derive the original continuous-session bounds.
        continuous = frames[1][frames[1]['phase'] == 'CONTINUOUS_AUCTION']
        # Reject unusable sessions.
        if continuous.empty or trades.empty or not events:
            # Report this cell as failed.
            raise ValueError('No continuous session or trades')
        # Set identical session bounds for every arm.
        session = (int(continuous.ts_exch.min()), int(continuous.ts_exch.max()))
        # Normalize pandas namedtuples before hashing the decoded event data.
        fingerprint = hashlib.sha256(pickle.dumps(([tuple(e[:4]) + (tuple(e[4]),) for e in events], snapshots), protocol=5)).hexdigest()
        # Retain only compact arm summaries between runs.
        results = {}
        # Preserve each cell's fills for review.
        folder = Path(output) / key
        # Never mix this cell with previous evidence.
        folder.mkdir(exist_ok=True)
        # Run a copy-equivalence control before the new signal.
        for arm_index, arm in enumerate(ARMS):
            # Build a fresh strategy with identical frozen production settings.
            strategy = (Original if arm == 'baseline' else Candidate)(session_ms=session, **copy.deepcopy(job['params']))
            # Reset the same latency random stream for each arm.
            cfg = dict(copy.deepcopy(R.CFG), session=session, latency_model=LatencyModel(seed=job['seed']), use_cfo=True, cross_on_arrival=True, log_equity=False)
            # Build a fresh simulated exchange.
            engine = Backtester(strategy, cfg)
            # Count how often the new signal is available and changes the skew direction.
            counts = dict(calls=0, valid=0, fallback=0, insufficient_depth=0, invalid_book=0, changed_skew=0)
            # Select depth and decay only for the declared research arms.
            spec = ARMS[arm]
            # Bind the new calculation only to the candidate arm.
            def signal(bb, ba, original):
                # Read the current reconstructed book, with no future features.
                bids, asks = engine.book.ranked_depth(spec[0], include_deep=False)
                # Calculate the fixed candidate signal.
                value, valid = distance_share(bids, asks, bb, ba, strategy.tick, original, depth=spec[0], decay=spec[1])
                # Count every queue-skew evaluation.
                counts['calls'] += 1
                # Separate usable depth from fallbacks.
                counts['valid' if valid else 'fallback'] += 1
                # Separate absent depth from invalid prices or a mismatched touch.
                if not valid:
                    # Identify why this arm retained the best-level signal.
                    counts['insufficient_depth' if len(bids) < spec[0] or len(asks) < spec[0] else 'invalid_book'] += 1
                # Compare the unchanged threshold decisions.
                decision = lambda x: int(x-0.5 > strategy.queue_skew_thresh) - int(0.5-x > strategy.queue_skew_thresh)
                # Record whether the actual skew trigger changes.
                counts['changed_skew'] += int(decision(value) != decision(original))
                # Supply only the queue-skew signal.
                return value
            # Leave both original and copy-control behavior unchanged.
            if spec is not None:
                # Install the process-local callback on this one strategy instance.
                strategy.spacing_signal = signal
            # Throttle worker-to-parent updates to at most one per second.
            last_update = [0.0]
            # Preserve event order while publishing progress.
            def stream():
                # Yield each original event exactly once.
                for index, event in enumerate(events):
                    # Check the clock only every thousand events.
                    if index % 1000 == 0 and time.monotonic()-last_update[0] >= 1:
                        # Store completion across all six replay arms.
                        PROGRESS[key] = (arm, (arm_index + index/len(events))/len(ARMS))
                        # Restart the update throttle.
                        last_update[0] = time.monotonic()
                    # Forward the unchanged market event.
                    yield event
            # Prevent snapshot mutation in one arm from affecting another.
            fills, equity, stats = engine.run(stream(), copy.deepcopy(snapshots))
            # Require finite end-of-day accounting.
            if not engine.eod or not math.isfinite(float(engine.eod['equity_liquidated'])):
                # Fail the entire comparison rather than dropping an arm.
                raise ValueError('Missing finite end-of-day accounting')
            # Save detailed fills, including separately tagged residual marks.
            fill_bytes = fills.to_csv(index=False).encode()
            # Preserve exact fill equality checks without mandatory large disk output.
            fill_hash = hashlib.sha256(fill_bytes).hexdigest()
            # Detailed fills remain explicitly opt-in for the full-history run.
            if job.get('save_fills', False):
                # Compress large fill artifacts without changing their logical contents.
                fills.to_csv(folder / (arm + '_fills.csv.gz'), index=False, compression='gzip')
            # Include starting equity zero when measuring the worst drop.
            values = [0.0] + equity.equity.tolist() + [float(engine.eod['equity_liquidated'])]
            # Track the running equity peak.
            peak, drawdown = 0.0, 0.0
            # Calculate the largest within-session marked loss from a previous peak.
            for value in values:
                # Update the running peak.
                peak = max(peak, value)
                # Update the largest peak-to-trough fall.
                drawdown = max(drawdown, peak-value)
            # Exclude haircut-marked residuals from actual simulated execution counts.
            actual = fills[fills.reason != 'liq_residual'] if not fills.empty else fills
            # Save directly comparable metrics and liquidation qualifications.
            results[arm] = dict(net_pkr=float(engine.eod['equity_liquidated']), fills=len(actual), shares=float(actual.qty.sum()) if not actual.empty else 0.0, max_abs_inventory=float(equity.pos.abs().max()) if not equity.empty else 0.0, max_drawdown_pkr=drawdown, eod=engine.eod, stats=stats, signal=counts, fills_sha256=fill_hash)
            # Save a cell checkpoint before starting the next arm.
            save(folder / 'arms.json', results)
        # Require the strategy-copy control to reproduce complete fills and accounting.
        control_ok = all(results['baseline'][field] == results['copy_control'][field] for field in ('net_pkr', 'fills_sha256', 'max_abs_inventory', 'max_drawdown_pkr', 'eod', 'stats'))
        # Refuse to interpret a strategy whose disabled copy changes behavior.
        if not control_ok:
            # Keep saved evidence for diagnosis.
            raise ValueError('Original versus disabled-copy reconciliation failed')
        # Retain the original job and decoded-input checksum.
        row = dict(job=job, passed=True, decoded_input_sha256=fingerprint, arms=results, delta_pkr={arm: value['net_pkr']-results['baseline']['net_pkr'] for arm, value in results.items() if ARMS[arm] is not None})
    # Capture failures without hiding incomplete scope.
    except Exception as error:
        # Preserve the failed cell's identity and reason.
        row = dict(job=job, passed=False, error=repr(error))
    # Save each completed cell immediately.
    save(Path(output) / (key + '.json'), row)
    # Mark completion for parent progress accounting.
    PROGRESS[key] = ('done', 1.0)
    # Return small evidence only.
    return row

# Validate a prepared full-history manifest before launching any replay.
def select_jobs(manifest, seeds, period):
    # Require three or more independent declared random streams.
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        # Never count a repeated seed as new evidence.
        raise ValueError('At least three distinct nonnegative seeds are required')
    # Use only the dates declared before any profit replay.
    dates = manifest['evaluation_dates']
    # Keep September out of profit measurement.
    if not dates or dates != sorted(set(dates)) or dates[0][:7] != '2025-10' or dates[-1] != '2026-06-30' or any(not '2025-10-01' <= date <= '2026-06-30' for date in dates):
        # Reject the old June pilot manifest or an altered date range.
        raise ValueError('Expected October 2025 through June 2026 evaluation dates')
    # Require exactly the full declared universe/date Cartesian scope.
    expected = {(symbol,date) for symbol in manifest['symbols'] for date in dates}
    # Include predeclared exclusions in scope verification.
    entries = manifest['jobs']+manifest['excluded']
    # Reject duplicates, omissions and invented job identities.
    if len(entries) != len(expected) or {(j['symbol'],j['date']) for j in entries} != expected:
        # Incomplete scopes cannot produce a clean comparison.
        raise ValueError('Manifest job/exclusion coverage is incomplete or duplicated')
    # Check every clip's dated sizing history independently of the preparer.
    for job in manifest['jobs']:
        # Require exactly ten strictly prior observed dates.
        history = job['clip_history']
        # Read the canonical median implementation.
        import statistics
        # Verify the clip can be reconstructed from its recorded contributors.
        if len(history) != 10 or len({d for d,q in history}) != 10 or any(d >= job['date'] or d < '2025-09-01' or not math.isfinite(q) or q <= 0 for d,q in history) or job['clip'] != max(1,int(round(3*statistics.median(q for d,q in history)))) or job['params']['size'] != job['clip']:
            # Fail before trading with a current-day or future-dependent clip.
            raise ValueError('Invalid strictly prior-date sizing history')
    # Pair each stock-day over the same fixed seed list.
    return [dict(copy.deepcopy(job),seed=seed) for job in manifest['jobs'] for seed in seeds],dates

# Keep parent memory bounded while detailed evidence stays on disk.
def compact(row):
    # Retain only the job identity and required report fields.
    result = dict(row)
    # Avoid retaining duplicate full parameter/history dictionaries per seed.
    result['job'] = {k:row['job'][k] for k in ('symbol','date','seed')}
    # Strip large lifecycle counters after each completed cell is saved.
    if row.get('passed'):
        # Preserve all reporting metrics and signal-coverage counters.
        result['arms'] = {arm:{k:v for k,v in value.items() if k != 'stats'} for arm,value in row['arms'].items() if arm != 'copy_control'}
    # Retain only closing fields used by the report; full evidence stays on disk.
    for value in result.get('arms',{}).values():
        # Avoid duplicating the full EOD diagnostic object for every seed and arm.
        value['eod'] = {key:value['eod'][key] for key in ('liquidation_clean','unfilled_sh')}
    # Return the compact report row.
    return result

# Run the expanded, predeclared paired profit experiment.
def main():
    # Configure the command-line interface.
    parser = argparse.ArgumentParser()
    # Require the user's production sizing and strategy manifest.
    parser.add_argument('--manifest', required=True)
    # Keep evidence outside the checkout.
    parser.add_argument('--output-dir', required=True)
    # Use eight workers on the user's ten-core machine.
    parser.add_argument('--workers', type=int, default=8)
    # Pair all arms across three fixed random streams by default.
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    # Keep validation evidence separate from the June comparison.
    parser.add_argument('--period', choices=('comparison',), default='comparison')
    # Resume only when all frozen input identities still match.
    parser.add_argument('--resume',action='store_true')
    # Avoid excessive full-history fill output unless explicitly needed.
    parser.add_argument('--save-fills',action='store_true')
    # Parse this invocation.
    args = parser.parse_args()
    # Require a valid bounded process count.
    if not 1 <= args.workers <= 8:
        # Refuse accidental oversubscription.
        parser.error('workers must be between 1 and 8')
    # Load the frozen production jobs.
    manifest = json.loads(Path(args.manifest).read_text())
    # Materialize all dates and seeds before starting any worker.
    jobs, selected_dates = select_jobs(manifest, args.seeds, args.period)
    # Freeze evidence-storage preferences with each job.
    for job in jobs:
        # Store full fills only when explicitly requested.
        job['save_fills'] = args.save_fills
    # Verify all preparation inputs before trusting cached sizing.
    for path,expected in manifest['hashes'].items():
        # Refuse changed code or calibration from the sizing pass.
        if not Path(path).exists() or digest(path) != expected:
            # A fresh preparation is needed after any such change.
            raise ValueError('Prepared input hash mismatch: '+path)
    # Check warmup data as well as evaluation data.
    for path,expected in manifest['raw_metadata'].items():
        # Preserve the original historical clip inputs.
        if not Path(path).exists() or [Path(path).stat().st_size,Path(path).stat().st_mtime_ns] != expected:
            # Avoid replaying jobs sized from changed raw data.
            raise ValueError('Prepared raw metadata mismatch: '+path)
    # Keep every denominator tied to the actual declared scope.
    planned = len(jobs)
    # Fail before replay if any required raw table is absent.
    for date in selected_dates:
        # Check all market-event table partitions.
        for name in ('ob_updates','ob_snapshot','trades'):
            # Do not spend hours completing only part of the intended period.
            if not any(R.date_dir(name,date).rglob('*.parquet')):
                # Explain the exact missing partition.
                raise ValueError(f'Missing parsed partition: {name} {date}')
    # Create a fresh output directory without overwriting prior work.
    output = Path(args.output_dir)
    # A reused run directory is an error.
    output.mkdir(parents=True, exist_ok=args.resume)
    # Fingerprint every local Python source and the sizing manifest.
    paths = sorted(Path(__file__).resolve().parent.glob('*.py')) + [Path(args.manifest)]
    # Record metadata for every original partition used by this pilot.
    raw_paths = sorted({p for date in selected_dates for name in ('ob_updates', 'ob_snapshot', 'trades') for p in R.date_dir(name, date).rglob('*.parquet')})
    # Save file identity, size and nanosecond modification time without another full-data scan.
    raw_metadata = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in raw_paths}
    # Keep raw-file metadata distinct from cryptographic decoded-input checksums.
    save(output / 'raw_metadata.json', raw_metadata)
    # Freeze source bytes before creating workers.
    hashes = {str(p): digest(p) for p in paths}
    # Pin resume to identical sources, manifest and run preferences.
    identity = dict(hashes=hashes,seeds=args.seeds,save_fills=args.save_fills,raw_metadata=raw_metadata)
    # Locate durable replay identity.
    identity_path = output/'resume_identity.json'
    # Refuse incompatible or unidentified existing output.
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        # Never mix outputs across code or input revisions.
        raise ValueError('Resume identity mismatch; use a fresh run directory')
    # Existing cell files without identity cannot be trusted.
    if not identity_path.exists() and any(output.glob('*_seed*.json')):
        # Stop rather than adopt unrelated results.
        raise ValueError('Existing cell evidence has no resume identity')
    # Store resume settings before launching workers.
    save(identity_path,identity)
    # Record simulator defaults, seed and the exact one-change policy.
    save(output / 'frozen.json', dict(jobs=jobs, hashes=hashes, cfg=R.CFG, latency_seeds=args.seeds, period=args.period, latency_defaults=vars(LatencyModel(seed=R.LATENCY_SEED)), arms=ARMS, policy='Only queue-skew bid share changes: exp(-k * own-touch distance in ticks) at declared depth; insufficient/invalid depth falls back to L1; original thresholds unchanged', use_cfo=True, cross_on_arrival=True, market='REG', scope='Full assignment-file universe, October-June; September warmup only; paired seeds averaged; retrospective fixed calibration; no independent validation or live approval'))
    # Use spawn so workers cannot inherit prior strategy mutations.
    context = mp.get_context('spawn')
    # Start the elapsed-time clock.
    started = time.monotonic()
    # Collect completed cells for reporting.
    rows = []
    # Reuse only fully successful cells with an exactly matching frozen job.
    remaining = []
    # Inspect durable cell summaries without loading market data.
    for job in jobs:
        # Match the worker's stable output key.
        key = job['symbol']+'_'+job['date']+'_seed'+str(job['seed'])
        # Locate any completed checkpoint.
        checkpoint = output/(key+'.json')
        # Read only checkpoints from this identity-verified run.
        previous = json.loads(checkpoint.read_text()) if args.resume and checkpoint.exists() else None
        # A complete cell with changed settings cannot be reused.
        if previous is not None and previous.get('job') != job:
            # Stop rather than overwrite unexplained evidence.
            raise ValueError('Checkpoint job mismatch: '+key)
        # Verify saved arm evidence before accepting a cached result.
        evidence = output/key/'arms.json'
        # Reuse successful cells only when durable evidence agrees exactly.
        if previous is not None and previous.get('passed') and evidence.exists() and json.loads(evidence.read_text()) == previous['arms']:
            # Retain compact metrics for the final report.
            rows.append(compact(previous))
        # Missing or failed cells must be regenerated from the same raw inputs.
        else:
            # Keep the complete job settings for the worker.
            remaining.append(job)
    # Preserve the number of pre-completed cells in progress denominators.
    resumed = len(rows)
    # Create shared progress and bounded independent processes.
    with context.Manager() as manager:
        # Maintain one compact progress record per cell.
        progress = manager.dict()
        # Initialize every worker with the progress proxy.
        with context.Pool(args.workers, initializer=initialize, initargs=(progress,)) as pool:
            # Submit each paired cell as an independent job.
            queue = iter(remaining)
            # Keep at most two jobs per worker submitted at a time.
            pending = []
            # Fill a bounded initial queue.
            for _ in range(min(len(remaining),args.workers*2)):
                # Submit one independent paired cell.
                pending.append(pool.apply_async(cell,((next(queue),str(output)),)))
            # Schedule a single parent print every fifteen seconds.
            last_print = started - 15
            # Wait without printing from worker processes.
            while pending:
                # Collect completed results and surface process failures.
                for result in pending[:]:
                    # Only read results that are ready.
                    if result.ready():
                        # Retain failed cells as well as successful ones.
                        rows.append(compact(result.get()))
                        # Remove this completed asynchronous task.
                        pending.remove(result)
                        # Refill only after a completed result frees a queue slot.
                        job = next(queue,None)
                        # Stop submitting when the fixed scope is exhausted.
                        if job is not None:
                            # Submit the next stock-day-seed comparison.
                            pending.append(pool.apply_async(cell,((job,str(output)),)))
                # Print a compact heartbeat at the user's requested cadence.
                if time.monotonic()-last_print >= 15:
                    # Snapshot progress atomically from the proxy.
                    state = dict(progress)
                    # Include partial replay progress in the estimate.
                    done = resumed + sum(value[1] for value in state.values())
                    # Measure elapsed minutes.
                    elapsed = (time.monotonic()-started)/60
                    # Estimate remaining time from completed arm fractions.
                    eta = f'~{elapsed*(planned-done)/(done-resumed):.1f}m' if done > resumed else 'estimating (loading)'
                    # Choose one active cell to explain current activity.
                    active = next(((key, value) for key, value in state.items() if value[0] != 'done'), ('starting', ('loading', 0)))
                    # Emit one timestamped line for the whole run.
                    print(f'[{datetime.now():%H:%M:%S}] {active[0]} | {active[1][0]} | Elapsed {elapsed:.1f}m | ETA {eta} | Done {len(rows)}/{planned} | Replay {100*done/planned:.0f}%', flush=True)
                    # Restart the print throttle.
                    last_print = time.monotonic()
                # Sleep briefly so results and cancellation stay responsive.
                time.sleep(0.2)
    # Detect source changes during this run.
    changed = [str(p) for p in paths if not p.exists() or digest(p) != hashes[str(p)]]
    # Detect raw-data rewrites while workers were running.
    changed.extend(str(p) for p in raw_paths if not p.exists() or [p.stat().st_size, p.stat().st_mtime_ns] != raw_metadata[str(p)])
    # Detect additions or removals within the selected raw partitions.
    raw_after = {p for date in selected_dates for name in ('ob_updates', 'ob_snapshot', 'trades') for p in R.date_dir(name, date).rglob('*.parquet')}
    # Include changed partition membership in the provenance failure list.
    changed.extend(str(p) for p in set(raw_paths).symmetric_difference(raw_after))
    # Recheck all inputs used for September warmup as well as later sizing.
    changed.extend(path for path,meta in manifest['raw_metadata'].items() if not Path(path).exists() or [Path(path).stat().st_size,Path(path).stat().st_mtime_ns] != meta)
    # Recheck every frozen calibration and preparation source.
    changed.extend(path for path,h in manifest['hashes'].items() if not Path(path).exists() or digest(path) != h)
    # Require all predeclared cells and unchanged sources.
    clean = len(rows) == planned and all(r['passed'] for r in rows) and not changed
    # Write the overall status before report generation.
    summary = dict(completed=clean, planned=planned, period=args.period, seeds=args.seeds, validation_completed=False, requested_stock_days=len(manifest['symbols'])*len(selected_dates), excluded_stock_days=len(manifest['excluded']), calibration_status=manifest['calibration_status'], completed_cells=len(rows), failed=sum(not r['passed'] for r in rows), changed_inputs=changed, live_approved=False)
    # Generate comparison tables only when every paired cell passed.
    if clean:
        # Import tabular reporting after historical replay completes.
        import pandas as pd
        # Flatten matching arm metrics without pooling failed cells.
        table = pd.DataFrame([dict(symbol=r['job']['symbol'], date=r['job']['date'], seed=r['job']['seed'], arm=arm, **{k: value[k] for k in ('net_pkr', 'fills', 'shares', 'max_abs_inventory', 'max_drawdown_pkr')}, liquidation_clean=value['eod']['liquidation_clean'], residual_shares=value['eod']['unfilled_sh'], **value['signal']) for r in rows for arm, value in r['arms'].items() if arm != 'copy_control'])
        # Save readable per-stock-day comparisons.
        table.to_csv(output / 'comparison.csv', index=False)
        # Aggregate daily profits across the four independently simulated names.
        seed_daily = table.groupby(['seed','date','arm']).net_pkr.sum().unstack()
        # Retain each seed's daily results for sensitivity checks.
        seed_daily.to_csv(output / 'daily_profit_by_seed.csv')
        # Seeds are repeated simulations of the same day, not extra trading days.
        averaged = table.groupby(['symbol','date','arm']).net_pkr.mean().reset_index()
        # Aggregate four stocks only after averaging seeds within each stock-day.
        daily = averaged.groupby(['date','arm']).net_pkr.sum().unstack()
        # Keep the profit difference for each seed, without claiming independent dates.
        seed_totals = seed_daily.groupby(level='seed').sum()
        # Save matched per-seed totals for every strategy.
        seed_totals.to_csv(output / 'profit_by_seed.csv')
        # Preserve the paired difference by date.
        differences = daily.drop(columns='baseline').subtract(daily.baseline, axis=0)
        # Save each arm's paired daily difference from the original.
        differences.to_csv(output / 'daily_differences.csv')
        # Write the daily profit comparison.
        daily.to_csv(output / 'daily_profit.csv')
        # Report absolute profit and differences rather than unstable percentage gains.
        summary.update(baseline_net_pkr=float(daily.baseline.sum()), tested_dates=len(daily), unique_stock_days=len(averaged[averaged.arm == 'baseline']), arms={})
        # Keep every candidate visible rather than selecting a winner automatically.
        for arm in differences.columns:
            # Select all matched stock-days and seeds for this arm.
            selected = table[table.arm == arm]
            # Count all queue-skew opportunities for honest fallback reporting.
            calls = int(selected.calls.sum())
            # Report realized simulation results and signal coverage separately.
            summary['arms'][arm] = dict(net_pkr=float(daily[arm].sum()), differences_by_seed={str(seed):float(seed_totals.loc[seed,arm]-seed_totals.loc[seed,'baseline']) for seed in args.seeds}, difference_pkr=float(differences[arm].sum()), winning_dates=int((differences[arm] > 0).sum()), valid_signal_pct=100*float(selected.valid.sum())/calls if calls else None, fallback_pct=100*float(selected.fallback.sum())/calls if calls else None, insufficient_depth=int(selected.insufficient_depth.sum()), invalid_book=int(selected.invalid_book.sum()), changed_skew=int(selected.changed_skew.sum()), runs_with_residual_inventory=int((selected.residual_shares > 0).sum()), worst_stock_day_drawdown_pkr=float(selected.max_drawdown_pkr.max()), worst_stock_day_net_pkr=float(selected.net_pkr.min()))
        # Add per-stock totals so pooled profit cannot hide a losing name.
        averaged.groupby(['symbol', 'arm']).net_pkr.sum().unstack().to_csv(output / 'profit_by_stock.csv')
        # Report profits by month without pooling repeated seeds as extra days.
        averaged['month'] = averaged.date.str[:7]
        # Save aggregate monthly P&L for the requested horizon.
        monthly = averaged.groupby(['month','arm']).net_pkr.sum().unstack()
        # Preserve monthly totals for review.
        monthly.to_csv(output/'monthly_profit.csv')
        # Separate stock-specific seasonal performance.
        averaged.groupby(['symbol','month','arm']).net_pkr.sum().unstack().to_csv(output/'profit_by_stock_month.csv')
        # Preserve predeclared missing-data and assignment exclusions.
        save(output/'exclusions.json',manifest['excluded'])
        # Use a non-interactive plotting backend.
        import matplotlib
        # Keep plots usable from a terminal.
        matplotlib.use('Agg')
        # Import the figure interface.
        import matplotlib.pyplot as plt
        # Draw actual daily net profit for both strategies.
        axis = monthly.plot.bar(figsize=(14, 5), ylabel='Net profit (PKR)', title=f'{args.period}: monthly profit averaged across latency seeds')
        # Mark the boundary between profit and loss.
        axis.axhline(0, color='black', linewidth=0.7)
        # Keep labels visible.
        plt.tight_layout()
        # Save the chart alongside its source table.
        plt.savefig(output / 'monthly_profit.png', dpi=150)
        # Release graphics resources.
        plt.close()
    # Publish the final qualification and totals.
    save(output / 'summary.json', summary)
    # Print the final summary only once.
    print(json.dumps(summary), flush=True)
    # Return a failure exit code when scope or provenance failed.
    if not clean:
        # Prevent a failed pilot being mistaken for a successful comparison.
        raise SystemExit(1)

# Keep spawned workers from starting another pool.
if __name__ == '__main__':
    # Execute only in the command-line entry process.
    main()
