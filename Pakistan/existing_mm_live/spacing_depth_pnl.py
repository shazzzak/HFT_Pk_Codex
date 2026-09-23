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
from datetime import datetime
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
ARMS = {'baseline': None, 'copy_control': None, 'distance_2': (2, 0.1), 'distance_3': (3, 0.1), 'distance_4': (4, 0.1), 'distance_5': (5, 0.1), 'equal_3': (3, 0.0)}

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
    key = job['symbol'] + '_' + job['date']
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
        folder.mkdir()
        # Run a copy-equivalence control before the new signal.
        for arm_index, arm in enumerate(ARMS):
            # Build a fresh strategy with identical frozen production settings.
            strategy = (Original if arm == 'baseline' else Candidate)(session_ms=session, **copy.deepcopy(job['params']))
            # Reset the same latency random stream for each arm.
            cfg = dict(copy.deepcopy(R.CFG), session=session, latency_model=LatencyModel(seed=R.LATENCY_SEED), use_cfo=True, cross_on_arrival=True, log_equity=False)
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
                        # Store completion across all seven replay arms.
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
            fills.to_csv(folder / (arm + '_fills.csv'), index=False)
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
            results[arm] = dict(net_pkr=float(engine.eod['equity_liquidated']), fills=len(actual), shares=float(actual.qty.sum()) if not actual.empty else 0.0, max_abs_inventory=float(equity.pos.abs().max()) if not equity.empty else 0.0, max_drawdown_pkr=drawdown, eod=engine.eod, stats=stats, signal=counts, fills_sha256=digest(folder / (arm + '_fills.csv')))
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

# Run the small, predeclared paired profit experiment.
def main():
    # Configure the command-line interface.
    parser = argparse.ArgumentParser()
    # Require the user's production sizing and strategy manifest.
    parser.add_argument('--manifest', required=True)
    # Keep evidence outside the checkout.
    parser.add_argument('--output-dir', required=True)
    # Use eight workers on the user's ten-core machine.
    parser.add_argument('--workers', type=int, default=8)
    # Parse this invocation.
    args = parser.parse_args()
    # Require a valid bounded process count.
    if not 1 <= args.workers <= 8:
        # Refuse accidental oversubscription.
        parser.error('workers must be between 1 and 8')
    # Load the frozen production jobs.
    manifest = json.loads(Path(args.manifest).read_text())
    # Select four predeclared names, without inspecting new profit results.
    symbols = ('NRL', 'MLCF', 'NPL', 'NCPL')
    # Choose the first, middle and last dates from the frozen June sample.
    dates = sorted({j['date'] for j in manifest['jobs']})
    # Require the expected manifest coverage.
    if len(dates) != 20:
        # Avoid silently changing the planned dates.
        raise ValueError('Expected the frozen 20-date manifest')
    # Fix three spread-out dates before observing results.
    selected_dates = (dates[0], dates[9], dates[-1])
    # Retain all production settings for the twelve selected cells.
    jobs = [j for j in manifest['jobs'] if j['symbol'] in symbols and j['date'] in selected_dates]
    # Validate unique cells and active queue skew.
    if len(jobs) != 12 or len({(j['symbol'], j['date']) for j in jobs}) != 12 or any(j['params'].get('queue_skew_ticks', 0) == 0 or j['params']['size'] != j['clip'] for j in jobs):
        # Fail before loading any historical events.
        raise ValueError('Invalid or inactive frozen pilot jobs')
    # Create a fresh output directory without overwriting prior work.
    output = Path(args.output_dir)
    # A reused run directory is an error.
    output.mkdir(parents=True, exist_ok=False)
    # Fingerprint every local Python source and the sizing manifest.
    paths = sorted(Path(__file__).resolve().parent.glob('*.py')) + [Path(args.manifest)]
    # Record metadata for every original partition used by this pilot.
    raw_paths = sorted({p for date in selected_dates for name in ('ob_updates', 'ob_snapshot', 'trades') for p in R.date_dir(name, date).rglob('*.parquet')})
    # Save file identity, size and nanosecond modification time without another full-data scan.
    raw_metadata = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in raw_paths}
    # Keep raw-file metadata distinct from cryptographic decoded-input checksums.
    save(output / 'raw_metadata.json', raw_metadata)
    # Freeze source bytes before creating workers.
    hashes = {str(p): digest(p) for p in paths}
    # Record simulator defaults, seed and the exact one-change policy.
    save(output / 'frozen.json', dict(jobs=jobs, hashes=hashes, cfg=R.CFG, latency_seed=R.LATENCY_SEED, latency_defaults=vars(LatencyModel(seed=R.LATENCY_SEED)), arms=ARMS, policy='Only queue-skew bid share changes: exp(-k * own-touch distance in ticks) at declared depth; insufficient/invalid depth falls back to L1; original thresholds unchanged', use_cfo=True, cross_on_arrival=True, market='REG', scope='12 cells, one seed, exploratory; not an independent holdout or live approval'))
    # Use spawn so workers cannot inherit prior strategy mutations.
    context = mp.get_context('spawn')
    # Start the elapsed-time clock.
    started = time.monotonic()
    # Collect completed cells for reporting.
    rows = []
    # Create shared progress and bounded independent processes.
    with context.Manager() as manager:
        # Maintain one compact progress record per cell.
        progress = manager.dict()
        # Initialize every worker with the progress proxy.
        with context.Pool(args.workers, initializer=initialize, initargs=(progress,)) as pool:
            # Submit each paired cell as an independent job.
            pending = [pool.apply_async(cell, ((job, str(output)),)) for job in jobs]
            # Schedule a single parent print every fifteen seconds.
            last_print = started - 15
            # Wait without printing from worker processes.
            while pending:
                # Collect completed results and surface process failures.
                for result in pending[:]:
                    # Only read results that are ready.
                    if result.ready():
                        # Retain failed cells as well as successful ones.
                        rows.append(result.get())
                        # Remove this completed asynchronous task.
                        pending.remove(result)
                # Print a compact heartbeat at the user's requested cadence.
                if time.monotonic()-last_print >= 15:
                    # Snapshot progress atomically from the proxy.
                    state = dict(progress)
                    # Include partial replay progress in the estimate.
                    done = sum(value[1] for value in state.values())
                    # Measure elapsed minutes.
                    elapsed = (time.monotonic()-started)/60
                    # Estimate remaining time from completed arm fractions.
                    eta = f'~{elapsed*(12-done)/done:.1f}m' if done > 0 else 'estimating (loading)'
                    # Choose one active cell to explain current activity.
                    active = next(((key, value) for key, value in state.items() if value[0] != 'done'), ('starting', ('loading', 0)))
                    # Emit one timestamped line for the whole run.
                    print(f'[{datetime.now():%H:%M:%S}] {active[0]} | {active[1][0]} | Elapsed {elapsed:.1f}m | ETA {eta} | Done {len(rows)}/12 | Replay {100*done/12:.0f}%', flush=True)
                    # Restart the print throttle.
                    last_print = time.monotonic()
                # Sleep briefly so results and cancellation stay responsive.
                time.sleep(0.2)
    # Detect source changes during this run.
    changed = [str(p) for p in paths if not p.exists() or digest(p) != hashes[str(p)]]
    # Detect raw-data rewrites while workers were running.
    changed.extend(str(p) for p in raw_paths if not p.exists() or (p.stat().st_size, p.stat().st_mtime_ns) != raw_metadata[str(p)])
    # Detect additions or removals within the selected raw partitions.
    raw_after = {p for date in selected_dates for name in ('ob_updates', 'ob_snapshot', 'trades') for p in R.date_dir(name, date).rglob('*.parquet')}
    # Include changed partition membership in the provenance failure list.
    changed.extend(str(p) for p in set(raw_paths).symmetric_difference(raw_after))
    # Require all predeclared cells and unchanged sources.
    clean = len(rows) == 12 and all(r['passed'] for r in rows) and not changed
    # Write the overall status before report generation.
    summary = dict(completed=clean, planned=12, completed_cells=len(rows), failed=sum(not r['passed'] for r in rows), changed_inputs=changed, live_approved=False)
    # Generate comparison tables only when every paired cell passed.
    if clean:
        # Import tabular reporting after historical replay completes.
        import pandas as pd
        # Flatten matching arm metrics without pooling failed cells.
        table = pd.DataFrame([dict(symbol=r['job']['symbol'], date=r['job']['date'], arm=arm, **{k: value[k] for k in ('net_pkr', 'fills', 'shares', 'max_abs_inventory', 'max_drawdown_pkr')}, liquidation_clean=value['eod']['liquidation_clean'], residual_shares=value['eod']['unfilled_sh'], **value['signal']) for r in rows for arm, value in r['arms'].items() if arm != 'copy_control'])
        # Save readable per-stock-day comparisons.
        table.to_csv(output / 'comparison.csv', index=False)
        # Aggregate daily profits across the four independently simulated names.
        daily = table.groupby(['date', 'arm']).net_pkr.sum().unstack()
        # Preserve the paired difference by date.
        differences = daily.drop(columns='baseline').subtract(daily.baseline, axis=0)
        # Save each arm's paired daily difference from the original.
        differences.to_csv(output / 'daily_differences.csv')
        # Write the daily profit comparison.
        daily.to_csv(output / 'daily_profit.csv')
        # Report absolute profit and differences rather than unstable percentage gains.
        summary.update(baseline_net_pkr=float(daily.baseline.sum()), tested_dates=len(daily), arms={})
        # Keep every candidate visible rather than selecting a winner automatically.
        for arm in differences.columns:
            # Select all twelve matched stock-days for this arm.
            selected = table[table.arm == arm]
            # Count all queue-skew opportunities for honest fallback reporting.
            calls = int(selected.calls.sum())
            # Report realized simulation results and signal coverage separately.
            summary['arms'][arm] = dict(net_pkr=float(daily[arm].sum()), difference_pkr=float(differences[arm].sum()), winning_dates=int((differences[arm] > 0).sum()), valid_signal_pct=100*float(selected.valid.sum())/calls if calls else None, fallback_pct=100*float(selected.fallback.sum())/calls if calls else None, insufficient_depth=int(selected.insufficient_depth.sum()), invalid_book=int(selected.invalid_book.sum()), changed_skew=int(selected.changed_skew.sum()), residual_shares=float(selected.residual_shares.sum()))
        # Add per-stock totals so pooled profit cannot hide a losing name.
        table.groupby(['symbol', 'arm']).net_pkr.sum().unstack().to_csv(output / 'profit_by_stock.csv')
        # Use a non-interactive plotting backend.
        import matplotlib
        # Keep plots usable from a terminal.
        matplotlib.use('Agg')
        # Import the figure interface.
        import matplotlib.pyplot as plt
        # Draw actual daily net profit for both strategies.
        axis = daily.plot.bar(figsize=(9, 4), ylabel='Net profit (PKR)', title='Same dates, sizes, fees and latency settings')
        # Mark the boundary between profit and loss.
        axis.axhline(0, color='black', linewidth=0.7)
        # Keep labels visible.
        plt.tight_layout()
        # Save the chart alongside its source table.
        plt.savefig(output / 'daily_profit.png', dpi=150)
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
