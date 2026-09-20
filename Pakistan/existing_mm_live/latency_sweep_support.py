# Store provenance and reports using portable paths.
from pathlib import Path
# Serialize complete experiment settings deterministically.
import json
# Hash the values actually passed to the worker processes.
import hashlib
# Flush progress immediately even when stdout is piped through tee.
import builtins
# Implement a heartbeat that also runs during calibration and slow cells.
import threading
# Measure elapsed durations using a monotonic clock.
import time
# Record dependency versions in the experiment manifest.
import importlib.metadata
# Preserve platform and Python information in the manifest.
import platform
# Calculate numeric summaries and validate finite inputs.
import numpy as np
# Aggregate experiment cells into seed-level and date-level portfolios.
import pandas as pd
# Compute Student-t intervals using market dates as the sampling units.
from scipy import stats


# Print every progress message immediately rather than relying on -u.
def print(*args, **kwargs):
    # Override buffering for this module and callers importing this function.
    kwargs['flush'] = True
    # Delegate formatting to Python's standard print function.
    builtins.print(*args, **kwargs)


# Convert calibration structures into stable, explicitly typed JSON values.
def canonical(value):
    # Sort mapping keys and preserve non-string key types through key/value pairs.
    if isinstance(value, dict):
        # Represent mappings without relying on JSON's implicit key coercion.
        return {'mapping': [[canonical(k), canonical(v)] for k, v in sorted(value.items(), key=lambda item: repr(item[0]))]}
    # Keep ordered calibration vectors and session segments in their original order.
    if isinstance(value, (tuple, list)):
        # Recursively normalize every nested value.
        return [canonical(v) for v in value]
    # Convert NumPy scalar values produced by pandas into Python scalars.
    if isinstance(value, np.generic):
        # Normalize again in case the scalar represents a nonfinite number.
        return canonical(value.item())
    # Preserve missing and infinite calibration values explicitly in the identity.
    if isinstance(value, float) and not np.isfinite(value):
        # Avoid nonstandard JSON NaN literals.
        return {'nonfinite': str(value)}
    # Resolve filesystem paths into stable text values.
    if isinstance(value, Path):
        # Preserve the supplied path without reading additional files.
        return str(value)
    # Normalize date objects without changing strings or numeric values.
    if hasattr(value, 'isoformat'):
        # Store an ISO-formatted date or timestamp.
        return value.isoformat()
    # Let JSON validate the remaining primitive types.
    return value


# Identify an experiment from its complete serialized configuration.
def digest(value):
    # Encode the normalized values without insignificant whitespace.
    raw = json.dumps(canonical(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    # Use the full SHA-256 digest in manifests and worker acknowledgements.
    return hashlib.sha256(raw).hexdigest()


# Fingerprint the source bytes of a code module or assignment file.
def file_digest(path):
    # Read the exact bytes currently present on disk.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Keep long-running stages observable even before the first date completes.
class Heartbeat:
    # Configure a single progress thread for each parent run.
    def __init__(self, interval=30.0):
        # Validate the requested heartbeat period.
        if interval <= 0:
            # Reject an accidental busy-loop configuration.
            raise ValueError('heartbeat interval must be positive')
        # Store the reporting period in seconds.
        self.interval = interval
        # Keep the current stage readable by the heartbeat thread.
        self.stage = 'starting'
        # Signal the thread to stop when the run exits.
        self.stop = threading.Event()
        # Record elapsed time independently of wall-clock changes.
        self.started = time.monotonic()
        # Prepare a daemon thread so abnormal shutdown cannot hang on logging.
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # Emit progress at bounded intervals until the caller finishes.
    def _loop(self):
        # Event.wait also wakes immediately when the run exits.
        while not self.stop.wait(self.interval):
            # Report the latest completed stage or work unit.
            print(f'[heartbeat] {self.stage}; elapsed {(time.monotonic()-self.started)/60:.1f} min')

    # Start progress reporting when entering the protected run.
    def __enter__(self):
        # Start the dedicated reporting thread.
        self.thread.start()
        # Let the caller update the current stage.
        return self

    # Stop progress reporting whether the run succeeds or fails.
    def __exit__(self, *exc):
        # Wake the waiting heartbeat thread.
        self.stop.set()
        # Wait briefly for the thread to exit cleanly.
        self.thread.join(timeout=2)


# Capture code, dependency, and effective-calibration provenance.
def provenance(arms, seeds, floor, dates, calib, assignment, modules, constants):
    # Hash each explicitly declared source dependency by its current bytes.
    sources = {name: {'path': str(Path(module.__file__).resolve()), 'sha256': file_digest(module.__file__)} for name, module in modules.items()}
    # Include the scientific dependencies that can affect replay and reporting.
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'pyarrow', 'scipy')}
    # Return the actual worker settings and hashes of the loaded calibration values.
    return {'schema': 2, 'arms': arms, 'seeds': seeds, 'shape_floor_ms': floor, 'dates': list(map(str, dates)), 'universe': calib['universe'], 'calibration_sha256': digest(calib), 'assignment': {'path': str(assignment), 'sha256': file_digest(assignment)}, 'sources': sources, 'constants': constants, 'python': platform.python_version(), 'platform': platform.platform(), 'dependencies': versions, 'raw_data_note': 'Raw market files are not content-hashed; preserve the parsed-store snapshot separately.'}


# Reject comparisons with mismatched experiment cells before portfolio aggregation.
def validate_cells(df, control='lat40'):
    # Define the source schema required for every reported metric.
    required = ['arm', 'seed', 'date', 'symbol', 'net_pkr', 'opened_notional', 'capture_pkr', 'crossed_orders', 'crossed_shares', 'crossed_value']
    # Refuse empty or structurally incomplete inputs.
    if df.empty or not set(required).issubset(df.columns):
        # Name the missing schema instead of producing a misleading empty report.
        raise ValueError(f'Empty sweep or missing columns: {sorted(set(required)-set(df.columns))}')
    # Define the unique experiment-cell key.
    keys = ['arm', 'seed', 'date', 'symbol']
    # Missing identifiers cannot participate in a paired experiment.
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        # Stop rather than silently collapsing duplicate rows.
        raise ValueError('Missing or duplicate experiment-cell identifiers')
    # Every comparison requires the same baseline.
    if control not in set(df.arm):
        # Prevent accidental comparison to an absent control.
        raise ValueError(f'Missing control arm {control}')
    # Identify the full Cartesian cohort represented by the source file.
    expected = pd.MultiIndex.from_product([sorted(df.seed.unique()), sorted(df.date.unique()), sorted(df.symbol.unique())])
    # Verify that every arm contains every recorded seed/date/symbol combination.
    for arm, a in df.groupby('arm'):
        # Build the actual cohort for this arm.
        got = pd.MultiIndex.from_frame(a[['seed', 'date', 'symbol']])
        # Any missing combination changes the portfolio being compared.
        if len(got) != len(expected) or len(expected.difference(got)):
            # Require explicit repair of incomplete coverage before statistics.
            raise ValueError(f'Unbalanced coverage for {arm}: {len(got)} of {len(expected)} cells')
    # Require finite accounting inputs; capture may be unavailable and is handled separately.
    numeric = ['net_pkr', 'opened_notional', 'crossed_orders', 'crossed_shares', 'crossed_value']
    # Convert numeric input explicitly so malformed CSV fields fail visibly.
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        # Do not let NaN-aware summation hide an accounting failure.
        raise ValueError('Nonfinite accounting or crossing inputs')
    # Quantities and unsigned traded values cannot be negative.
    if df[numeric[1:]].lt(0).any().any():
        # Reject impossible negative activity values.
        raise ValueError('Negative notional or crossing activity')


# Build corrected reports without rerunning the expensive exchange simulation.
def analyse_cells(df, control='lat40'):
    # Validate experiment coverage before any aggregation.
    validate_cells(df, control)
    # Work on a private frame to preserve caller-owned results.
    frame = df.copy()
    # Detect missing or infinite capture observations.
    unavailable = ~np.isfinite(frame.capture_pkr.to_numpy(dtype=float))
    # A truly inactive cell has no measurable capture contribution.
    inactive = frame[['net_pkr', 'opened_notional', 'crossed_orders', 'crossed_shares', 'crossed_value']].eq(0).all(axis=1)
    # Normalize missing capture only on fully inactive cells.
    frame.loc[unavailable & inactive, 'capture_pkr'] = 0.0
    # Mark active cells with unavailable capture instead of treating them as zeros.
    frame['_capture_missing'] = unavailable & ~inactive
    # Translate detailed columns to the existing daily artifact's vocabulary.
    mapping = {'net_pkr': 'pkr', 'opened_notional': 'opn', 'capture_pkr': 'cap', 'crossed_orders': 'cx', 'crossed_value': 'cxv'}
    # Aggregate each seed's portfolio separately for each date.
    g = frame.groupby(['arm', 'seed', 'date'])[list(mapping)].sum().rename(columns=mapping)
    # Propagate active missing capture through the whole portfolio observation.
    bad = frame.groupby(['arm', 'seed', 'date'])._capture_missing.any()
    # Keep unavailable capture explicit in seed-level output.
    g.loc[bad, 'cap'] = np.nan
    # Record how many active cells were unmeasured in each observation.
    g['capture_missing_cells'] = frame.groupby(['arm', 'seed', 'date'])._capture_missing.sum()
    # A zero-exposure portfolio date has no defined basis-point rate.
    denominator = g.opn.where(g.opn.gt(0))
    # Compute net basis points on opened notional.
    g['bps'] = g.pkr / denominator * 10000
    # Compute the legacy capture proxy on the same denominator.
    g['cap_bps'] = g.cap / denominator * 10000
    # Label this as crossing/opened notional, not a fraction of total turnover.
    g['cx_pct'] = g.cxv / denominator * 100
    # Average seed-level rates and currency totals within each market date.
    day = g.groupby(['arm', 'date']).mean()
    # Refuse to average only the measurable seeds of an otherwise incomplete rate.
    for col in ['bps', 'cap', 'cap_bps', 'cx_pct']:
        # Find dates on which at least one seed lacks the metric.
        missing_day = g[col].isna().groupby(level=['arm', 'date']).any()
        # Preserve missingness through the seed average.
        day.loc[missing_day, col] = np.nan
    # Select the control's seed-averaged daily observations.
    ctl = day.loc[control]
    # Collect one comparison record per observed arm.
    records = []
    # Derive the reporting arm list from the artifact rather than module defaults.
    for arm in sorted(frame.arm.unique()):
        # Align each arm with the same baseline dates.
        a = day.loc[arm].reindex(ctl.index)
        # Form paired differences after averaging simulations of the same day.
        delta = (a.bps - ctl.bps).dropna()
        # Count usable market dates, not seed/date observations.
        n = len(delta)
        # Estimate uncertainty from the paired differences themselves.
        se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        # Compute the paired mean only when a defined rate exists.
        mean = float(delta.mean()) if n else np.nan
        # Use a Student-t critical value for the available date count.
        half = float(stats.t.ppf(.975, n-1) * se) if n > 1 else np.nan
        # Handle constant differences explicitly instead of dividing by zero.
        tval = mean / se if se > 0 else (0.0 if se == 0 and mean == 0 else np.nan)
        # Avoid claiming infinite confidence when all differences are identical and nonzero.
        pval = float(2*stats.t.sf(abs(tval), n-1)) if np.isfinite(tval) and n > 1 else np.nan
        # Save seed-averaged monetary totals and date-based inference together.
        records.append({'arm': arm, 'dates': len(a), 'paired_dates': n, 'seeds': int(frame.seed.nunique()), 'net_pkr': float(a.pkr.sum()), 'daily_bps': float(a.bps.mean()), 'pooled_bps': float(a.pkr.sum()/a.opn.sum()*10000) if a.opn.sum() > 0 else np.nan, 'delta_bps': mean, 'se': se, 'ci_low': mean-half, 'ci_high': mean+half, 't': tval, 'p': pval if arm != control else np.nan, 'delta_pkr': float((a.pkr-ctl.pkr).sum()), 'capture_bps': float(a.cap_bps.mean()) if a.cap_bps.notna().all() else np.nan, 'crossing_opened_pct': float(a.cx_pct.mean()), 'notional_change_pct': float((a.opn.sum()/ctl.opn.sum()-1)*100) if ctl.opn.sum() > 0 else np.nan, 'capture_missing_cells': int(g.loc[arm].capture_missing_cells.sum())})
    # Assemble the complete comparison table.
    summary = pd.DataFrame(records).set_index('arm')
    # Order the tested comparisons for Holm's step-down correction.
    ordered = summary.p.dropna().sort_values()
    # Count every attempted non-control comparison, including undefined tests conservatively.
    comparisons = len(summary) - 1
    # Correct for the complete family of comparisons against the baseline.
    corrected = np.minimum(1.0, np.maximum.accumulate(ordered.to_numpy() * (comparisons-np.arange(len(ordered))))) if len(ordered) else np.array([])
    # Attach adjusted p-values without changing the display order.
    summary['holm_p'] = pd.Series(corrected, index=ordered.index)
    # Return all levels so callers can inspect rather than trust a printed headline.
    return g.reset_index(), day.reset_index(), summary.reset_index()


# Persist reproducible reporting artifacts for new runs and existing CSVs.
def write_report(df, directory, source=None, plot=False):
    # Calculate all tables before publishing a successful report.
    g, day, summary = analyse_cells(df)
    # Resolve the caller-selected output location.
    directory = Path(directory)
    # Avoid mixing this analysis with a previous report.
    directory.mkdir(parents=True, exist_ok=False)
    # Retain the original per-seed daily table for reconciliation.
    g.to_csv(directory / 'daily_by_seed.csv', index=False)
    # Save the market-date units used by the paired test.
    day.to_csv(directory / 'daily_seed_average.csv', index=False)
    # Save intervals, exposure changes, and corrected probabilities.
    summary.to_csv(directory / 'paired_summary.csv', index=False)
    # Identify the exact detailed input when reanalysing an existing artifact.
    meta = {'source': str(source) if source else None, 'source_sha256': file_digest(source) if source else None, 'support_sha256': file_digest(__file__), 'rows': len(df), 'symbols': int(df.symbol.nunique()), 'dates': int(df.date.nunique()), 'seeds': sorted(df.seed.unique().tolist()), 'capture_note': 'Legacy equity-mid lookup; fill-time event alignment has not been independently validated.', 'inference_note': 'Seeds averaged within dates; Student-t intervals assume independent sampled dates. No market-impact or fill-model uncertainty is included.'}
    # Save the reporting assumptions alongside the numerical output.
    (directory / 'report_manifest.json').write_text(json.dumps(meta, indent=2))
    # Print the key corrected results with unbuffered output.
    print(summary[['arm', 'net_pkr', 'delta_bps', 'ci_low', 'ci_high', 'paired_dates', 'holm_p']].to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    # State the two assumptions that most directly limit interpretation.
    print(meta['inference_note'])
    # Keep the unresolved capture-timing limitation visible.
    print(meta['capture_note'])
    # Report missing active capture observations explicitly.
    if summary.capture_missing_cells.sum():
        # Net P&L remains available; affected capture comparisons are suppressed.
        print('WARNING: active cells have unavailable capture; affected capture summaries are NaN.')
    # Render the optional chart only when the caller requests an image.
    if plot:
        # Import the renderer lazily so non-plotting runs do not require matplotlib.
        import matplotlib
        # Save the chart without opening a GUI.
        matplotlib.use('Agg')
        # Load plotting primitives after selecting the backend.
        import matplotlib.pyplot as plt
        # Select non-control arms with defined intervals.
        shown = summary[summary.arm.ne('lat40')].dropna(subset=['delta_bps', 'ci_low', 'ci_high'])
        # Create enough vertical space for all experiment labels.
        fig, ax = plt.subplots(figsize=(9, max(3, len(shown)*.3+1.5)))
        # Plot daily paired differences and pointwise 95% intervals.
        ax.errorbar(shown.delta_bps, np.arange(len(shown)), xerr=shown.ci_high-shown.delta_bps, fmt='o', capsize=3)
        # Mark the no-difference reference line.
        ax.axvline(0, color='gray', linewidth=1)
        # Label each observed arm.
        ax.set_yticks(np.arange(len(shown)), shown.arm)
        # Identify the basis-point metric and reference arm.
        ax.set_xlabel('Daily net-bps difference versus lat40; pointwise 95% interval')
        # State the sampling unit and seed treatment.
        ax.set_title('Latency sweep: seeds averaged within each market date')
        # Fit all labels inside the image boundary.
        fig.tight_layout()
        # Save a standalone review image.
        fig.savefig(directory / 'paired_comparison.png', dpi=160)
        # Release renderer resources after saving.
        plt.close(fig)
    # Report the output directory as the last line for terminal users.
    print(f'Report saved to {directory}')
    # Return the frames for offline regression tests.
    return g, day, summary
