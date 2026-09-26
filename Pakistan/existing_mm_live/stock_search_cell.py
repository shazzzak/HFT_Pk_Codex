# Copy immutable per-arm simulator settings.
import copy
# Fingerprint decoded event order and fill evidence.
import hashlib
# Validate closing equity.
import math
# Resolve per-cell output paths.
from pathlib import Path
# Serialize normalized raw-event identities.
import pickle
# Throttle shared progress updates.
import time
# Reuse the unchanged historical latency distribution.
from mm_backtest import LatencyModel
# Read the canonical parsed data with explicit REG filters.
import run_legacy_mm as R
# Use this checkout's configured data root.
from config_pk import PARSED_ROOT
# Apply the authoritative root to the legacy loader.
R.PARSED_ROOT = PARSED_ROOT
# Use the new signal, risk, and observation implementation.
from stock_search_engine import ARMS, CHEAP, Engine, make_strategy
# Reconcile FIFO accounting before accepting a result.
from stock_search_accounting import attribute
# Use strict evidence writes and the tested distance formula.
from stock_search_util import save, distance_share, digest
# Keep raw pointers exact without the old parser-ID dependency.
from psx_reference_rows import reference
# Screen the complete channel before filtering to one symbol.
from psx_sequence_audit import audit_date

# Receive shared progress storage in each spawned worker.
def initialize(progress):
    # Keep the proxy private to the worker process.
    global PROGRESS
    # Retain the parent-owned progress dictionary.
    PROGRESS = progress

# Replay one matched stock-day using the original event stream.
def cell(task):
    # Measure per-cell runtime for a workload-normalized pilot estimate.
    cell_started = time.monotonic()
    # Unpack immutable job settings and output location.
    job, output = task
    # Name this evidence directory deterministically.
    key = job['symbol'] + '_' + job['date'] + '_seed' + str(job['seed'])
    # Preserve failures as part of the requested denominator.
    try:
        # Show loading while parquet decoding is in progress.
        PROGRESS[key] = ('loading', 0.0)
        # Missing messages on any symbol can invalidate this channel's reconstruction.
        quality=audit_date((str(PARSED_ROOT),job['date']))
        # Refuse a P&L result when source completeness is not established.
        if not quality['sequence_complete']:
            # Preserve unknown profit as a failure; do not replay a guessed book.
            raise ValueError('DATA_QUALITY_BLOCKED: '+str(quality))
        # Open original parsed exchange data.
        datasets = R.open_datasets(job['date'])
        # Fail explicitly if a requested partition is absent.
        if datasets is None:
            # Never silently drop a requested day.
            raise ValueError('Missing parsed date')
        # Load REG rows from all three tables, requiring the market column.
        frames = [R.read_symbol(datasets[name], list(dict.fromkeys(cols + ['market'] + (['channel', 'buy_ref', 'sell_ref'] if name != 'ob_snapshot' else []))), job['symbol'], market='REG') for name, cols in (('ob_updates', R.REQ_UPDATES), ('ob_snapshot', R.REQ_SNAP), ('trades', R.REQ_TRADES))]
        # Preserve original exchange IDs for audit while matching queues by source reference.
        frames[0]['exchange_order_id']=frames[0]['order_id']
        # Adds identify themselves; cancellations identify their referenced original add.
        frames[0]['order_id']=[f"ref:{reference(r.channel)}:{reference(r.appl_seq) if r.event=='ORDER_ADD' else (reference(r.buy_ref) or reference(r.sell_ref))}" for r in frames[0].itertuples()]
        # Trades with one reference get the same canonical ID as their original order.
        frames[2]['resting_order_id']=[f"ref:{reference(r.channel)}:{reference(r.buy_ref) or reference(r.sell_ref)}" if bool(reference(r.buy_ref)) != bool(reference(r.sell_ref)) else None for r in frames[2].itertuples()]
        # Retain event clocks; snapshot liquidity is ignored by the exact book.
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
        # Replay every style and signal, independent of the old assignment.
        for arm_index, arm in enumerate(ARMS):
            # Reuse only identical cheap-stock styles at the SAME signal depth.
            canonical = 'OBI_' + arm.split('_')[1] if job['symbol'] in CHEAP else arm
            # All three styles are identical on named exceptions, but depths are not.
            if canonical in results:
                # Retain explicit alias provenance rather than claiming another replay.
                results[arm] = dict(results[canonical], reused_from=canonical, effective_params=make_strategy(job, arm, session)[1])
                # Persist the exact alias after the original has completed.
                save(folder / 'arms.json', results)
                # Avoid eight duplicate replays per cheap stock-day.
                continue
            # Build and validate the actual twelve-grid settings.
            strategy, effective = make_strategy(job, arm, session)
            # Reset the same latency random stream for each arm.
            cfg = dict(copy.deepcopy(R.CFG), session=session, latency_model=LatencyModel(seed=job['seed']), use_cfo=True, cross_on_arrival=True, log_equity=False)
            # Build a fresh simulated exchange.
            engine = Engine(strategy, cfg)
            # Count how often the new signal is available and changes the skew direction.
            counts = dict(calls=0, valid=0, fallback=0, insufficient_depth=0, invalid_book=0, changed_throttle_decision=0, changed_defensive_decision=0, changed_queue_decision=0)
            # Select depth and decay only for the declared research arms.
            spec = (ARMS[arm][2], 0.1) if ARMS[arm][2] is not None else None
            # Bind the new calculation only to the candidate arm.
            def signal(bb, ba, original):
                # Read the current reconstructed book, with no future features.
                bids, asks = engine.book.ranked_depth(spec[0], include_deep=False)
                # Calculate the fixed candidate signal.
                value, valid = distance_share(bids, asks, bb, ba, strategy.tick, original, depth=spec[0], decay=spec[1])
                # Count every weighted decision-signal evaluation.
                counts['calls'] += 1
                # Separate usable depth from fallbacks.
                counts['valid' if valid else 'fallback'] += 1
                # Separate absent depth from invalid prices or a mismatched touch.
                if not valid:
                    # Identify why this arm retained the best-level signal.
                    counts['insufficient_depth' if len(bids) < spec[0] or len(asks) < spec[0] else 'invalid_book'] += 1
                # Compare the unchanged threshold decisions.
                decision = lambda x, threshold: int(x-0.5 > threshold) - int(0.5-x > threshold)
                # Count changes separately for the three declared decision rules.
                counts['changed_throttle_decision'] += int(decision(value, strategy.obi_throttle_thresh) != decision(original, strategy.obi_throttle_thresh))
                # Preserve the defensive threshold independently of the queue threshold.
                counts['changed_defensive_decision'] += int(decision(value, strategy.obi_defensive_thresh) != decision(original, strategy.obi_defensive_thresh))
                # Queue skew remains inert for OBI and one-tick books.
                counts['changed_queue_decision'] += int(strategy.queue_skew_ticks > 0 and ba - bb > strategy.tick + 1e-10 and decision(value, strategy.queue_skew_thresh) != decision(original, strategy.queue_skew_thresh))
                # Supply the same share to throttle, defensive retreat and enabled queue skew.
                return value
            # Attach the replacement share to weighted variants in every quote style.
            if spec is not None:
                # Install the process-local callback on this one strategy instance.
                strategy.search_signal = signal
            # Throttle worker-to-parent updates to at most one per second.
            last_update = [0.0]
            # Preserve event order while publishing progress.
            def stream():
                # Yield each original event exactly once.
                for index, event in enumerate(events):
                    # Check the clock only every thousand events.
                    if index % 1000 == 0 and time.monotonic()-last_update[0] >= 1:
                        # Store completion across all twelve logical configurations.
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
            # Preserve an uncompressed identity as well as compressed fill evidence.
            fill_hash = hashlib.sha256(fill_bytes).hexdigest()
            # Persist fill-time evidence so decomposition can be revisited without replay.
            fills.to_csv(folder / (arm + '_fills.csv.gz'), index=False, compression={'method': 'gzip', 'compresslevel': 1})
            # Attribute every closed lot to its original acquisition-time bucket.
            attribution = attribute(fills.to_dict('records'), job['params']['session_segments'], float(engine.eod['equity_liquidated']))
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
            results[arm] = dict(net_pkr=float(engine.eod['equity_liquidated']), fills=len(actual), shares=float(actual.qty.sum()) if not actual.empty else 0.0, max_abs_inventory=float(equity.pos.abs().max()) if not equity.empty else 0.0, max_drawdown_pkr=drawdown, eod=engine.eod, stats=stats, signal=counts, fills_sha256=fill_hash, fill_file_sha256=digest(folder / (arm + '_fills.csv.gz')), attribution=attribution, effective_params=effective, capacity_reductions=engine.capacity_reductions)
            # Save a cell checkpoint before starting the next arm.
            save(folder / 'arms.json', results)
            # Break the engine/strategy callback cycle before allocating the next replay.
            del strategy.capacity_guard
            # Drop the weighted-book callback after this arm's final accounting.
            if hasattr(strategy, 'search_signal'):
                # Keep completed engines and event-level equity logs reclaimable promptly.
                del strategy.search_signal
        # Retain the original job and decoded-input checksum.
        row = dict(job=job, passed=True, decoded_input_sha256=fingerprint, arms=results, elapsed_seconds=time.monotonic()-cell_started, distinct_replays=sum('reused_from' not in value for value in results.values()), delta_pkr={arm: value['net_pkr']-results['OBI_best']['net_pkr'] for arm, value in results.items()})
    # Capture failures without hiding incomplete scope.
    except Exception as error:
        # Preserve the failed cell's identity and reason.
        row = dict(job=job, passed=False, error=repr(error))
    # Save each completed cell immediately.
    save(Path(output) / (key + '.json'), row)
    # Mark completion for parent progress accounting.
    PROGRESS[key] = ('done', 1.0)
    # Return small evidence only.
    return dict(key=key, passed=row['passed'], error=row.get('error'))
