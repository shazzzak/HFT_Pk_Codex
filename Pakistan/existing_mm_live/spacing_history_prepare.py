# Parse the full-history preparation command.
import argparse
# Freeze configuration objects without shared mutations.
import copy
# Read and write reproducible manifests.
import json
# Validate finite production sizes.
import math
# Calculate medians without extra historical-model fitting.
import statistics
# Print one heartbeat from a background thread during long reads.
import threading
# Measure wall-clock progress.
import time
# Format timestamps and validate session dates.
from datetime import datetime, timezone
# Resolve external output paths.
from pathlib import Path
# Reuse canonical calibration and strategy assembly.
import mm_harness as H
# Reuse canonical parsed data loading.
import run_legacy_mm as R
# Use the production assignment validator.
from live_config import LiveConfig
# Reuse atomic evidence writing and source fingerprinting.
from spacing_history_pnl import save, digest

# Preserve the original median-of-daily-medians production sizing rule.
def prior_clip(history, date):
    # Exclude current and future days before selecting the last ten observations.
    prior = sorted((day, value) for day,value in history.items() if day < date)
    # Require ten days on which this symbol actually traded.
    if len(prior) < 10:
        # Keep insufficient warmup visible as an exclusion.
        return None
    # Select only the last ten prior observed trading days.
    selected = prior[-10:]
    # Match the existing production median-of-medians calculation.
    median = statistics.median(value for day,value in selected)
    # Reject invalid sizing history.
    if not math.isfinite(median) or median <= 0:
        # Never substitute a guessed production size.
        raise ValueError('Invalid trailing trade-size median')
    # Record exactly which dates determined this production clip.
    return max(1, int(round(3*median))), selected

# Prepare every stock-day before any profit comparison starts.
def main():
    # Require explicit inputs and output locations.
    parser = argparse.ArgumentParser()
    # Freeze the existing assignment/universe file.
    parser.add_argument('--assignment', required=True, type=Path)
    # Pin calibration files through the already-tested June manifest.
    parser.add_argument('--reference-manifest', required=True, type=Path)
    # Distinguish a uniform rule experiment from an assigned-strategy test.
    parser.add_argument('--strategy-mode', required=True, choices=('uniform','assigned'))
    # Store resumable preparation outside the checkout.
    parser.add_argument('--output-dir', required=True, type=Path)
    # Resolve the requested preparation.
    args = parser.parse_args()
    # Create or resume only this preparation directory.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Load the frozen calibration provenance already used in June.
    reference = json.loads(args.reference_manifest.read_text())
    # Select exact calibration paths rather than the newest available file.
    paths = {}
    # Require each shared production calibration input.
    for pattern in ('session_scales_*.csv','volume_profile_*.csv','time_windows_*.csv','session_segments_*.csv'):
        # Resolve only paths explicitly frozen in the reference manifest.
        matches = [Path(p) for p in reference['hashes'] if Path(p).match(pattern)]
        # Ambiguous calibration inputs are an error.
        if len(matches) != 1 or not matches[0].exists() or digest(matches[0]) != reference['hashes'][str(matches[0])]:
            # Fail before using a changed or missing calibration.
            raise ValueError('Calibration mismatch: '+pattern)
        # Pin this calibration input.
        paths[pattern] = matches[0]
    # Keep all loaders on the same known calibration snapshot.
    H.newest = lambda pattern: paths[pattern]
    # Load current calibrated constants once without fitting on new profit results.
    scales, profiles, windows, segments = H.load_scales(), H.load_profiles(), H.load_windows(), H.load_segments()
    # Read assignments without unrequested local overrides.
    config = LiveConfig(args.assignment, None)
    # Include the full assignment-file universe, including historical DROP names.
    names = config.symbols(quoting_only=False)
    # Use September as the earliest permitted sizing history.
    dates = [d for d in R.discover_dates() if '2025-09-01' <= str(d) <= '2026-06-30']
    # Normalize date strings for deterministic manifests.
    dates = sorted(map(str,dates))
    # Require all requested months and both period boundaries.
    if not dates or dates[0][:7] != '2025-09' or dates[-1] != '2026-06-30' or {d[:7] for d in dates} != {'2025-09','2025-10','2025-11','2025-12','2026-01','2026-02','2026-03','2026-04','2026-05','2026-06'}:
        # A truncated store must not look like the full requested history.
        raise ValueError('Missing requested September-June date coverage')
    # Fingerprint preparation source and immutable calibration inputs.
    files = sorted(Path(__file__).parent.glob('*.py')) + list(paths.values()) + [args.assignment,args.reference_manifest]
    # Include the production assignment validator outside the research source directory.
    files.append(Path(R.__file__).resolve().parent.parent/'Production/venues/psx_config.py')
    # Freeze hashes before scanning any raw data.
    hashes = {str(p.resolve()):digest(p) for p in files}
    # Keep schema, universe and policy in the preparation identity.
    identity = dict(hashes=hashes,dates=dates,symbols=names,strategy_mode=args.strategy_mode)
    # Locate a previous interrupted preparation, if any.
    freeze = args.output_dir/'preparation_identity.json'
    # Refuse to resume with changed code, calibration, dates or strategy policy.
    if freeze.exists() and json.loads(freeze.read_text()) != identity:
        # Keep old evidence untouched.
        raise ValueError('Preparation inputs changed; use a new output directory')
    # Persist the intended scope atomically.
    save(freeze,identity)
    # Import only the columnar and arithmetic helpers needed for this small pre-pass.
    import pyarrow.dataset as ds
    # Use numpy for finite quantity validation.
    import numpy as np
    # Track progress independently of long table reads.
    status = dict(done=0,date=dates[0])
    # Stop the heartbeat cleanly after preparation.
    stop = threading.Event()
    # Start elapsed-time accounting.
    started = time.monotonic()
    # Print one line every fifteen seconds even while a table is decoding.
    def heartbeat():
        # Stop promptly when preparation finishes or fails.
        while not stop.wait(15):
            # Measure the completed-date rate.
            elapsed = (time.monotonic()-started)/60
            # Estimate remaining preparation time from finished dates only.
            eta = f"~{elapsed*(len(dates)-status['done'])/status['done']:.1f}m" if status['done'] else 'estimating'
            # Emit one compact timestamped preparation heartbeat.
            print(f"[{datetime.now():%H:%M:%S}] Sizing {status['date']} | Elapsed {elapsed:.1f}m | ETA {eta} | Done {status['done']}/{len(dates)}",flush=True)
    # Run no parallel data scans in this lightweight one-read-per-date pass.
    thread = threading.Thread(target=heartbeat,daemon=True)
    # Start the nonblocking heartbeat.
    thread.start()
    # Retain only per-symbol daily medians and eligibility, not raw market rows.
    stats = {name:{} for name in names}
    # Record each day's available continuous-session symbols and trade medians.
    available = {}
    # Freeze raw metadata for preparation and later replay validation.
    raw_metadata = {}
    # Ensure failures terminate the heartbeat.
    try:
        # Scan dates chronologically so progress remains understandable.
        for index,date in enumerate(dates):
            # Publish the date currently loading.
            status['date'] = date
            # Locate all three required event partitions.
            partitions = {name:sorted(R.date_dir(name,date).rglob('*.parquet')) for name in ('trades','ob_updates','ob_snapshot')}
            # Missing partitions are failures rather than holidays inferred from silence.
            if any(not files for files in partitions.values()):
                # Expose the incomplete date explicitly.
                raise ValueError('Missing raw partition: '+date)
            # Capture size and nanosecond modification time for each file.
            metadata = {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for files in partitions.values() for p in files}
            # Keep completed preparation dates resumable.
            checkpoint = args.output_dir/('sizing_'+date+'.json')
            # Reuse only metadata-matching checkpoints under the frozen source identity.
            record = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
            # A rewritten input date invalidates the preparation run.
            if record is not None and record['raw_metadata'] != metadata:
                # Do not silently mix new data with old medians.
                raise ValueError('Raw data changed: '+date)
            # Build this date only if no verified checkpoint exists.
            if record is None:
                # Open canonical original event tables.
                datasets = R.open_datasets(date)
                # Select the regular stock market and the complete declared universe.
                predicate = (ds.field('market') == 'REG') & ds.field('symbol').isin(names)
                # Read trade quantities once for all stocks rather than once per stock.
                trades = datasets['trades'].to_table(columns=['symbol','qty','market'],filter=predicate).to_pandas()
                # Reject invalid regular-market trade sizes instead of changing their meaning.
                if not np.isfinite(trades.qty.to_numpy(dtype=float)).all() or (trades.qty <= 0).any():
                    # Preserve invalid raw data for explicit review.
                    raise ValueError('Nonpositive or nonfinite REG trade quantity on '+date)
                # Match the canonical daily median followed by trailing median sizing rule.
                medians = {str(k):float(v) for k,v in trades.groupby('symbol').qty.median().items()}
                # Read only phase metadata to avoid replaying unusable cells.
                snapshot = datasets['ob_snapshot'].to_table(columns=['symbol','phase','market'],filter=predicate & (ds.field('phase') == 'CONTINUOUS_AUCTION')).to_pandas()
                # Record absence before looking at any strategy outcome.
                continuous = sorted(snapshot.symbol.unique().tolist())
                # Retain source identity along with each date checkpoint.
                record = dict(medians=medians,continuous=continuous,raw_metadata=metadata)
                # Fail if raw inputs changed while this date was read.
                if any(not Path(p).exists() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns] != meta for p,meta in metadata.items()):
                    # Do not checkpoint a torn preparation read.
                    raise ValueError('Raw input changed during sizing: '+date)
                # Publish this completed date before advancing.
                save(checkpoint,record)
            # Add this day's per-stock trade sizes to strictly dated history.
            for symbol,value in record['medians'].items():
                # Retain only one daily median per stock/date.
                stats[symbol][date] = value
            # Freeze observed continuous-session coverage.
            available[date] = set(record['continuous'])
            # Merge raw provenance for both warmup and profit periods.
            raw_metadata.update(metadata)
            # Advance the completed-date count.
            status['done'] = index+1
    # End the background heartbeat on success and failure.
    finally:
        # Request immediate heartbeat shutdown.
        stop.set()
        # Wait only briefly for the idle thread to exit.
        thread.join(timeout=1)
    # Save every requested stock/date decision, including exclusions.
    jobs,excluded = [],[]
    # Profit reporting begins only in October.
    evaluation_dates = [d for d in dates if d >= '2025-10-01']
    # Assemble all parameters before the first simulated order.
    for date in evaluation_dates:
        # Missing date calibration is a preparation failure.
        if date not in segments:
            # Never use a June session on an October day.
            raise ValueError('Missing session segments: '+date)
        # Verify that calibrated segments belong to this trading date.
        if any(datetime.fromtimestamp(start/1000,timezone.utc).strftime('%Y-%m-%d') != date or end <= start for start,end in segments[date]):
            # Stop on mismatched historical session bounds.
            raise ValueError('Invalid session segments: '+date)
        # Cover the whole fixed universe without profit-based selection.
        for symbol in names:
            # Start from the stock's recorded assignment.
            setting = config.params_for(symbol)
            # Preserve exclusions explicitly without substituting zero profits.
            reason = None
            # Assigned DROP stocks remain unquoted under assigned mode only.
            if args.strategy_mode == 'assigned' and not setting['quote']:
                # Preserve the production choice as a visible scope exclusion.
                reason = 'assigned_DROP'
            # Require all non-sizing calibrated constants.
            elif symbol not in scales or symbol not in profiles or symbol not in windows:
                # Never invent missing calibration.
                reason = 'missing_calibration'
            # No continuous regular-market book means no comparable replay.
            elif symbol not in available[date]:
                # Retain this known absence in the manifest.
                reason = 'no_REG_continuous_snapshot'
            # Match the existing runner's requirement for daily trades.
            elif date not in stats[symbol]:
                # Zero observed trades is an explicit eligibility condition.
                reason = 'no_REG_trades'
            # Calculate a production clip using strictly earlier observations.
            sizing = prior_clip(stats[symbol],date)
            # Insufficient history cannot be filled with future observations.
            if sizing is None and reason is None:
                # A newly active symbol warms up independently.
                reason = 'fewer_than_10_prior_traded_dates'
            # Record exclusions before any strategy replay.
            if reason is not None:
                # Keep requested denominators auditable.
                excluded.append(dict(symbol=symbol,date=date,reason=reason))
                # Skip all five versions together for this unavailable cell.
                continue
            # Unpack the exact clip and its historical contributors.
            clip,history = sizing
            # Build date-appropriate sizes and sessions with existing calibrated constants.
            params = H.build_micro_params(clip,scales[symbol],profiles[symbol],windows[symbol],segments[date])
            # Preserve the chosen baseline convention consistently across all arms.
            override = dict(queue_skew_ticks=2.0,queue_skew_thresh=.15) if args.strategy_mode == 'uniform' else config.strategy_kwargs(symbol)
            # Apply the declared strategy assignment after shared production parameters.
            params.update(override)
            # Store every strategy argument and sizing date for replay and inspection.
            jobs.append(dict(symbol=symbol,date=date,clip=clip,assignment='QT_2t@15' if args.strategy_mode == 'uniform' else setting['label'],params=params,clip_history=history))
    # Recheck all frozen sources and raw inputs before releasing the manifest.
    changed = [p for p,h in hashes.items() if not Path(p).exists() or digest(p) != h]
    # Detect changes since any preparation checkpoint was read.
    changed.extend(p for p,m in raw_metadata.items() if not Path(p).exists() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns] != m)
    # Do not launch the long replay on mixed inputs.
    if changed:
        # Expose exact mismatched paths.
        raise ValueError('Preparation inputs changed: '+repr(changed))
    # A manifest with no comparable jobs cannot be a completed experiment.
    if not jobs:
        # Stop without running workers.
        raise ValueError('No eligible stock-days')
    # Record calibration limitations alongside the experiment definition.
    manifest = dict(jobs=jobs,excluded=excluded,symbols=names,evaluation_dates=evaluation_dates,history_dates=[d for d in dates if d < '2025-10-01'],hashes=hashes,raw_metadata=raw_metadata,strategy_mode=args.strategy_mode,calibration_status='Retrospective fixed calibrated constants and universe; only clip sizing is strictly prior-date. Not an as-of historical production simulation or independent validation.')
    # Atomically publish the complete historical job specification.
    save(args.output_dir/'history_jobs_manifest.json',manifest)
    # Report concise scope counts before replay begins.
    print(json.dumps(dict(prepared=True,symbols=len(names),profit_dates=len(evaluation_dates),eligible_stock_days=len(jobs),excluded_stock_days=len(excluded),strategy_mode=args.strategy_mode)),flush=True)

# Keep preparation out of imported worker processes.
if __name__ == '__main__':
    # Run only the explicit preparation entry point.
    main()
