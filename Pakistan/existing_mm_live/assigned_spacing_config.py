# Copy frozen jobs without changing the previous experiment.
import copy
# Inspect constructor defaults instead of relying on comments.
import inspect
# Resolve frozen evidence and installed source paths.
from pathlib import Path
# Read reproducible configuration files.
import json
# Hash source and calibration bytes.
import hashlib
# Read recorded per-setting outcomes only to resolve the user's DROP policy.
import csv
# Reject missing or nonfinite recorded comparison values.
import math
# Measure bounded configuration preparation progress.
import time
# Format the user's requested heartbeat timestamps.
from datetime import datetime
# Count the complete assigned universe.
from collections import Counter
# Use the project's single assignment validator without local overrides.
from live_config import LiveConfig
# Validate original and distance-hook strategy constructors.
from assigned_spacing_strategy import Original
# Use the existing isolated queue-signal hook.
from assigned_spacing_strategy import Candidate

# Freeze the user's requested common settings in all four versions.
COMMON = dict(obi_throttle=True, obi_throttle_thresh=0.15, throttle_frac=0.5,
              throttle_hold_ms=300.0, ofi_throttle=False, qdr_throttle=False,
              flow_throttle=False, enable_pov_cap=True, pov_cap_mult=1.0,
              enable_run_reprice=False, enable_aggr_lean=False, enable_age_cross=False,
              enable_inv_taper=False, size_boost_mult=1.0, queue_skew_bps=0.0,
              exit_ticks_inside=1, exit_inv_threshold=1.0, obi_defensive=True,
              obi_defensive_thresh=0.15, obi_defensive_ticks=1.0,
              use_microprice=False, tol_ticks=0.0, ofi_defensive=False,
              min_edge_pct=0.0005, improve_ticks=0.0, gamma=0.15, kappa=1.5,
              tick=0.01, require_viable=True, unwind_pov=0.10,
              enable_eod_trigger=True, enable_lock_trigger=True)
# Preserve the explicit zero-skew exceptions, including revived TPL.
CHEAP = frozenset(('KEL', 'PIBTL', 'TPL'))
# Freeze the shipped three-tier universe rather than silently choosing a new one.
EXPECTED = {'QT_2t@15': 68, 'QT_2t@20': 13, 'OBI': 17, 'DROP': 15}

# Fingerprint inputs without loading large files into memory.
def digest(path):
    # Initialize SHA-256.
    value = hashlib.sha256()
    # Open an immutable input.
    with Path(path).open('rb') as stream:
        # Read bounded chunks.
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            # Include every byte.
            value.update(chunk)
    # Return the complete content identity.
    return value.hexdigest()

# Validate historical inputs before reusing the completed sizing build.
def verify_inputs(manifest):
    # Verify every frozen source and calibration file.
    for path, expected in manifest['hashes'].items():
        # Refuse changed historical inputs.
        if not Path(path).is_file() or digest(path) != expected:
            # Name the mismatched input.
            raise ValueError('Frozen input changed: ' + path)
    # Check raw partition metadata including September sizing history.
    for path, expected in manifest['raw_metadata'].items():
        # Resolve each recorded partition.
        item = Path(path)
        # Never quietly reuse sizing after data changes.
        if not item.is_file() or [item.stat().st_size, item.stat().st_mtime_ns] != expected:
            # Metadata identity is not claimed to be a raw-content hash.
            raise ValueError('Frozen raw metadata changed: ' + path)

# Assemble one complete trading job under an explicit former-DROP policy.
def configure_job(job, config, replacements=None):
    # Preserve the recorded size, calibration and prior-date contributors.
    result = copy.deepcopy(job)
    # Read the authoritative assignment.
    setting = config.params_for(job['symbol'])
    # Missing assignments must not receive a guessed baseline.
    if setting is None:
        # Explain the unmapped stock.
        raise ValueError('Missing assignment: ' + job['symbol'])
    # Preserve the recorded label even when the full-universe experiment revives it.
    original_label = setting['label']
    # Resolve previously excluded stocks under an explicitly selected rule.
    if not setting['quote']:
        # TPL is explicitly forced to plain OBI by the user.
        label = 'OBI' if job['symbol'] == 'TPL' else (replacements or {}).get(job['symbol'])
        # Refuse silently selecting a baseline for a formerly excluded stock.
        if label not in ('OBI','QT_2t@15','QT_2t@20'):
            # Make the missing policy explicit.
            raise ValueError('DROP replacement missing: ' + job['symbol'])
        # Resolve queue knobs from the same canonical label table.
        from venues.psx_config import ENGINE_PARAMS
        # Recover the chosen label's exact queue settings.
        ticks, threshold, quote = ENGINE_PARAMS[label]
        # Mark this experiment's stock as tradable.
        setting = dict(label=label,quote=quote)
        # Set only the per-tier knobs, never DROP's soft_inv=0 override.
        specific = dict(queue_skew_ticks=ticks,queue_skew_thresh=threshold)
    # Keep every existing tradable stock on its assigned tier.
    else:
        # Obtain the unchanged production assignment parameters.
        specific = config.strategy_kwargs(job['symbol'])
    # Apply the winning shared settings and the requested capacity switch.
    result['params'].update(COMMON)
    # Apply this stock's assigned queue adjustment without changing its clip.
    result['params'].update(specific)
    # Honor the named exception independently of the assigned queue label.
    if job['symbol'] in CHEAP:
        # Disable both possible queue-adjustment mechanisms.
        result['params'].update(queue_skew_ticks=0.0, queue_skew_bps=0.0)
    # Record the actual portfolio label.
    result['assignment'] = setting['label']
    # Preserve original DROP provenance separately from the trading setting.
    result['original_assignment'] = original_label
    # Retain the exception as reviewable evidence.
    result['cheap_zero_skew'] = job['symbol'] in CHEAP
    # Return an isolated job with all shared settings explicit.
    return result

# Check actual constructed strategy attributes before it receives any events.
def validate_strategy(strategy, job):
    # Check all user-specified shared settings.
    for key, expected in COMMON.items():
        # Fail if a constructor ignores or changes a requested switch.
        if getattr(strategy, key) != expected:
            # Identify the actual runtime mismatch.
            raise ValueError('Effective strategy mismatch: ' + key)
    # Verify stock-specific queue knobs as well as shared controls.
    for key in ('queue_skew_ticks', 'queue_skew_thresh'):
        # Compare actual attributes to the frozen job.
        if getattr(strategy, key) != job['params'][key]:
            # Stop before an incorrectly assigned replay.
            raise ValueError('Effective assignment mismatch: ' + key)
    # Verify the actual clip and static inventory ratios.
    if strategy.size0 != job['clip'] or strategy.max_inv != 10 * job['clip'] or strategy.soft_inv != 3 * job['clip']:
        # Do not allow hidden sizing overrides.
        raise ValueError('Effective size or inventory ratio mismatch')
    # Require both inputs needed by the enabled acquisition cap.
    if strategy.unwind_profile is None or strategy.session_segments is None:
        # A cap silently bypassed for missing calibration is unacceptable.
        raise ValueError('Missing capacity calibration')
    # Reject corrupt volume inputs rather than silently bypassing the cap.
    if len(strategy.unwind_profile) != 4 or any(not math.isfinite(v) or v < 0 for v in strategy.unwind_profile):
        # Zero expected volume remains a legitimate no-capacity condition.
        raise ValueError('Invalid four-bucket volume profile')

# Freeze assignments on top of the already-completed historical sizing manifest.
def prepare(source, assignment, output, drop_policy):
    # Read the completed sizing evidence.
    old = json.loads(Path(source).read_text())
    # Check its complete original lineage.
    verify_inputs(old)
    # Read the assignment with no implicit intervention file.
    config = LiveConfig(Path(assignment), None)
    # Require the same fixed universe as the previous sizing build.
    if set(config.symbols(False)) != set(old['symbols']):
        # Never substitute stocks while claiming a matched comparison.
        raise ValueError('Assignment universe differs from frozen sizing')
    # Count assignments before outcomes influence scope.
    counts = Counter(config.params_for(s)['label'] for s in old['symbols'])
    # Confirm the intended shipped portfolio precisely.
    if dict(counts) != EXPECTED:
        # A different file needs explicit review rather than fallback defaults.
        raise ValueError('Unexpected three-tier assignment counts: ' + repr(counts))
    # Require the full original 113-stock historical scope before reviving DROP names.
    if old['excluded'] or len(old['jobs']) != len(old['symbols']) * len(old['evaluation_dates']):
        # This run must not silently omit names or dates from the requested universe.
        raise ValueError('Full-universe run requires complete saved sizing coverage')
    # Require an explicit user-selected baseline rule for the formerly excluded names.
    if drop_policy not in ('best-recorded','qt15'):
        # Avoid inventing a trading policy from a nontrading label.
        raise ValueError('Choose DROP policy best-recorded or qt15')
    # Preserve the recorded per-setting numbers used by the chosen rule.
    with Path(assignment).open() as stream:
        # Map rows once rather than rereading the file per stock-day.
        records = {r['symbol']: r for r in csv.DictReader(stream)}
    # Resolve replacements once before examining any new replay result.
    replacements = {}
    # Evaluate only stocks whose existing label was DROP.
    for symbol in old['symbols']:
        # Leave already tradable stock assignments unchanged.
        if config.params_for(symbol)['label'] != 'DROP':
            # Continue to the next stock.
            continue
        # Use the explicit cheap-stock override for TPL.
        if symbol == 'TPL':
            # TPL must trade plain OBI in all four versions.
            replacements[symbol] = 'OBI'
        # Apply the alternative uniform replacement only if selected.
        elif drop_policy == 'qt15':
            # Keep this rule independent of recorded per-setting profits.
            replacements[symbol] = 'QT_2t@15'
        # Resolve the highest recorded trading-setting result with deterministic ties.
        else:
            # Prefer OBI, then QT@15, then QT@20 on exact ties.
            choices = [('OBI',float(records[symbol]['obi_pkr'])),('QT_2t@15',float(records[symbol]['lean15_pkr'])),('QT_2t@20',float(records[symbol]['lean20_pkr']))]
            # Reject unavailable or corrupt historical scores.
            if not all(math.isfinite(value) for label,value in choices):
                # Do not substitute an arbitrary label.
                raise ValueError('Invalid recorded DROP scores: ' + symbol)
            # Freeze the chosen label across all dates and weighted depths.
            replacements[symbol] = max(choices,key=lambda item:item[1])[0]
    # Preserve original calibration and eligibility evidence.
    result = copy.deepcopy(old)
    # Construct new independent jobs.
    result['jobs'] = []
    # Preserve all pre-existing non-profit-based exclusions.
    result['excluded'] = copy.deepcopy(old['excluded'])
    # Materialize the assigned portfolio without rereading market events.
    for job in old['jobs']:
        # Apply the complete explicit configuration.
        configured = configure_job(job, config, replacements)
        # Preserve every original stock-date with its resolved trading setting.
        result['jobs'].append(configured)
    # Identify this experiment unambiguously.
    result.update(strategy_mode='assigned', experiment='assigned_spacing_v1', assignment_counts=dict(counts), seed=0, common_settings=COMMON, drop_policy=drop_policy, drop_replacements=replacements, all_stocks_trade=True)
    # Keep the previous study's calibration qualification.
    result['calibration_status'] = old['calibration_status'] + ' Assigned labels and common controls are retrospective; one latency seed only.'
    # Pin the old manifest and the selected assignment.
    for path in (Path(source).resolve(), Path(assignment).resolve()):
        # Preserve exact bytes as part of the new run's identity.
        result['hashes'][str(path)] = digest(path)
    # Include every installed source, including this additive package.
    for path in sorted(Path(__file__).resolve().parent.glob('*.py')):
        # Freeze the effective implementation for reproducible resume.
        result['hashes'][str(path)] = digest(path)
    # Measure the actual constructor-validation phase.
    started, last_print = time.monotonic(), time.monotonic()
    # Validate one actual object per job before releasing the manifest.
    for index, job in enumerate(result['jobs'],1):
        # Use its frozen continuous-session endpoints for constructor validation.
        session = (job['params']['session_segments'][0][0], job['params']['session_segments'][-1][1])
        # Construct the exact original strategy with all requested switches.
        validate_strategy(Original(session_ms=session, **job['params']), job)
        # Keep long preflight validation visible without flooding the terminal.
        if time.monotonic() - last_print >= 15:
            # Estimate remaining constructor checks from measured completed work.
            elapsed = (time.monotonic() - started) / 60
            # Print at most one concise line each fifteen seconds.
            print(f'[{datetime.now():%H:%M:%S}] Config checks | Elapsed {elapsed:.1f}m | ETA ~{elapsed*(len(result["jobs"])-index)/index:.1f}m | Done {index}/{len(result["jobs"])}',flush=True)
            # Restart the print interval.
            last_print = time.monotonic()
    # Recheck inputs after assembly to catch concurrent modifications.
    verify_inputs(result)
    # Create an external evidence directory.
    target = Path(output)
    # Allow preparation to be reissued only with identical content.
    target.mkdir(parents=True, exist_ok=True)
    # Keep a deterministic manifest filename.
    path = target / 'assigned_jobs_manifest.json'
    # Reject incompatible preparation resumes.
    if path.exists() and json.loads(path.read_text()) != result:
        # Leave the earlier evidence intact.
        raise ValueError('Prepared manifest changed; choose a fresh output folder')
    # Write the new frozen jobs without altering any previous output.
    path.write_text(json.dumps(result, indent=2))
    # Export every effective explicit setting in a human-readable stock-date table.
    fields = ['symbol','date','original_assignment','assignment','cheap_zero_skew','clip','max_inv','soft_inv','queue_skew_ticks','queue_skew_thresh'] + list(COMMON)
    # Write one row per real stock-day, retaining changing sizes explicitly.
    with (target/'effective_config.csv').open('w',newline='') as stream:
        # Preserve a stable column order.
        writer = csv.DictWriter(stream,fieldnames=fields)
        # Label every setting rather than relying on implied defaults.
        writer.writeheader()
        # Persist the actual explicit overrides used by each replay.
        for job in result['jobs']:
            # Combine job identity and parameters without hiding per-date clips.
            values = dict(job, **job['params'])
            # Write only the declared human-review columns.
            writer.writerow({key:values[key] for key in fields})
    # Export constructor defaults so omitted settings are visible.
    defaults = {k: ('REQUIRED' if v.default is inspect.Parameter.empty else v.default) for k, v in inspect.signature(Original).parameters.items()}
    # Save defaults separately from the explicit per-job overrides.
    (target / 'constructor_defaults.json').write_text(json.dumps(defaults, indent=2))
    # Print concise preparation evidence.
    print(json.dumps(dict(prepared=True, original_assignments=dict(counts), jobs=len(result['jobs']), excluded=len(result['excluded']), drop_policy=drop_policy, seed=0)), flush=True)
    # Return the new manifest for the replay command.
    return path
