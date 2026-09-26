# Read immutable saved evidence without importing the trading engine.
import json, csv, time, hashlib
# Accumulate diagnostic counts without interpreting them as durations.
from collections import Counter
# Locate input evidence and separate audit outputs.
from pathlib import Path
# Render a static research figure.
import matplotlib.pyplot as plt
# Locate the completed run.
root = Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/clean_window_full_20260925_v1')
# Store new reports separately from every frozen artifact.
out = root.parent / 'Availability_Audit_20260925_v1'
# Create the authorized report destination.
out.mkdir(exist_ok=True)
# Read saved channel screening, which combines sequence gaps and timing anomalies.
quality = json.loads((root / 'channel_quality.json').read_text())
# Read the requested six-month stock-day universe.
cells = [r for r in csv.DictReader((root / 'coverage.csv').open()) if '2026-01-01' <= r['date'] <= '2026-06-30']
# Combine overlapping intervals before measuring duration.
def union(intervals):
    # Keep the merged intervals in chronological order.
    merged = []
    # Process each nonempty interval once.
    for a, b in sorted(intervals):
        # Ignore empty or reversed ranges.
        if b <= a:
            # Continue to the next actual duration.
            continue
        # Extend a touching or overlapping previous range.
        if merged and a <= merged[-1][1]:
            # Preserve the greatest right boundary.
            merged[-1][1] = max(merged[-1][1], b)
        # Otherwise retain a new disjoint range.
        else:
            # Copy the endpoints into a mutable pair.
            merged.append([a, b])
    # Return nonoverlapping intervals.
    return merged
# Measure intersection of disjoint interval collections.
def overlap(left, right):
    # Sum exact integer milliseconds without midpoint approximations.
    return sum(max(0, min(b, d)-max(a, c)) for a, b in left for c, d in right)
# Write reviewable CSV diagnostics.
def save_csv(name, rows):
    # Open only the new output path.
    with (out / name).open('w', newline='') as stream:
        # Use the first row's explicit schema.
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        # Preserve column names.
        writer.writeheader()
        # Save every record without an index column.
        writer.writerows(rows)
# Accumulate exact durations separately from counts.
totals = Counter()
# Preserve source diagnostic counts and closing failures.
reasons, failures, cell_rows = Counter(), [], []
# Cache date-level interval calculations.
cache = {}
# Measure elapsed time for bounded progress reporting.
began = last = time.monotonic()
# Inspect each saved stock-day result once.
for index, cell in enumerate(cells, 1):
    # Locate the immutable stock-day result.
    path = root / (cell['symbol'] + '_' + cell['date']) / 'result.json'
    # Decode saved evidence only.
    result = json.loads(path.read_text())
    # Require a successful completed cell.
    assert result['passed']
    # Retrieve the saved individual continuous sessions.
    segments = result['job']['params']['session_segments']
    # Collect all candidate and accepted intervals.
    candidates = [(w['start'], w['end']) for w in result['windows']]
    # Require disjoint candidate windows.
    assert sum(b-a for a,b in candidates) == sum(b-a for a,b in union(candidates))
    # Independently reconcile saved durations with window detail.
    assert sum(b-a for a,b in candidates) == result['source_window_ms']
    # Independently reconcile accepted duration.
    assert sum(w['end']-w['start'] for w in result['windows'] if w['accepted']) == result['accepted_ms']
    # Check the summary CSV against stock-day evidence.
    for column, key in [('requested_minutes','requested_ms'),('source_window_minutes','source_window_ms'),('matched_minutes','accepted_ms')]:
        # Allow only floating conversion noise below a microsecond.
        assert abs(float(cell[column])*60000-result[key]) < 0.001
    # Build disjoint known channel-unavailable ranges once per date.
    if cell['date'] not in cache:
        # Retrieve the saved channel screen for this date.
        q = quality[cell['date']]
        # Treat whole-channel rejection separately from timed gap flags.
        flagged = segments if q['unusable'] else union([(a,b) for a,b,_ in q['intervals']] + [(0,q['first_ms']), (q['last_ms'],max(b for a,b in segments))])
        # Clip the unavailable ranges to continuous sessions.
        clipped = union([(max(a,c),min(b,d)) for a,b in flagged for c,d in segments])
        # Cache the exact clipped ranges and session contract.
        cache[cell['date']] = (clipped, segments)
    # Retrieve the cached date-level screening.
    flagged, saved_segments = cache[cell['date']]
    # Require identical session contracts across stocks on a date.
    assert segments == saved_segments
    # Measure channel-unavailable time actually outside candidate windows.
    channel_ms = sum(b-a for a,b in flagged)-overlap(flagged,candidates)
    # Preserve any unexpected candidate overlap as a separate diagnostic.
    totals['candidate_channel_overlap_ms'] += overlap(flagged,candidates)
    # Compute remaining unavailable time without assigning unsupported causes.
    other_ms = result['requested_ms']-result['source_window_ms']
    # Verify the partial attribution does not exceed the missing duration.
    assert 0 <= channel_ms <= other_ms
    # Record each disjoint duration component in exact milliseconds.
    row = dict(symbol=cell['symbol'], date=cell['date'], requested_ms=result['requested_ms'], accepted_ms=result['accepted_ms'], closing_excluded_ms=result['source_window_ms']-result['accepted_ms'], channel_flagged_ms=channel_ms, remaining_unattributed_ms=other_ms-channel_ms)
    # Preserve the stock-day duration audit.
    cell_rows.append(row)
    # Accumulate numeric components only.
    totals.update({key:value for key,value in row.items() if key.endswith('_ms')})
    # Preserve counts, including overlapping diagnostics, without duration inference.
    reasons.update(result['source_exclusions'])
    # Inspect every candidate's closing decision.
    for window in result['windows']:
        # Count candidates and common exclusions exactly once.
        totals['candidate_windows'] += 1
        # Read failures only for rejected common windows.
        if not window['accepted']:
            # Require an explicit failure rather than an unexplained rejection.
            assert window['exit_failures']
            # Count a stock-specific rejected window once across arms.
            totals['excluded_windows'] += 1
            # Preserve each arm's saved failure exposure.
            for arm, failure in window['exit_failures'].items():
                # Retain cash as cash, never as profit when inventory remains.
                failures.append(dict(symbol=cell['symbol'],date=cell['date'],window_id=window['window_id'],start=window['start'],end=window['end'],ending=window['ending'],arm=arm,reason=failure['reason'].split(':')[0],position=failure['position'],cash=failure['cash'],orders=failure['orders'],pending=failure['pending']))
    # Emit a concise heartbeat at most once per fifteen seconds.
    if time.monotonic()-last >= 15:
        # Measure elapsed wall time.
        elapsed = time.monotonic()-began
        # Report evidence-reading progress rather than strategy replay progress.
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {cell["symbol"]}/{cell["date"]}/saved-audit elapsed={elapsed:.0f}s remaining={elapsed*(len(cells)/index-1):.0f}s jobs={index}/{len(cells)} progress={100*index/len(cells):.1f}%',flush=True)
        # Reset the heartbeat clock.
        last = time.monotonic()
# Verify complete duration accounting backward to the requested denominator.
assert sum(totals[k] for k in ['accepted_ms','closing_excluded_ms','channel_flagged_ms','remaining_unattributed_ms']) == totals['requested_ms']
# Require the stated six-month exclusion count to match actual evidence.
assert totals['excluded_windows'] == sum(int(c['exit_excluded_windows']) for c in cells)
# Save source counters with their non-duration meaning in the filename.
save_csv('source_diagnostic_counts_NOT_DURATIONS.csv',[dict(reason=k,count=v) for k,v in reasons.most_common()])
# Save complete per-arm failure exposure for independent inspection.
save_csv('closing_failure_exposure.csv',failures)
# Save disjoint stock-day duration components.
save_csv('stock_day_availability.csv',cell_rows)
# Summarize overlapping arm failure counts without treating them as unique windows.
failure_counts = Counter(f['reason'] for f in failures)
# Preserve the largest absolute share position with its exact context.
largest = max(failures,key=lambda f:abs(f['position']))
# Assemble a compact machine-readable summary.
summary = dict(scope='January–June 2026, 113 tickers',stock_days=len(cells),totals=dict(totals),source_diagnostic_counts=dict(reasons),failure_arm_counts=dict(failure_counts),failure_arm_records=len(failures),nonflat_failure_arm_records=sum(abs(f['position'])>1e-9 for f in failures),largest_absolute_share_position=largest,source=root.as_posix(),limitations=['Channel flags combine missing sequences and capture timing; they do not isolate actual lost messages.','Remaining unavailable duration cannot be separated by cause from saved counters.','Failure arm records are correlated alternatives, not simultaneous portfolio exposure.','Cash with residual inventory is not profit; share counts are not comparable rupee risk across tickers.'])
# Save exact results before rendering.
(out/'summary.json').write_text(json.dumps(summary,indent=2))
# Use descriptive labels and distinct colors for the exact disjoint components.
labels = ['Accepted windows','Closing exclusions','Channel flags / feed extent','Other unavailable: cause untimed']
# Convert milliseconds to shares of requested stock-time.
values = [100*totals[k]/totals['requested_ms'] for k in ['accepted_ms','closing_excluded_ms','channel_flagged_ms','remaining_unattributed_ms']]
# Allocate a readable single-panel static figure.
fig, ax = plt.subplots(figsize=(11,4.8),layout='constrained')
# Render the disjoint time breakdown.
ax.barh(labels,values,color=['#24776b','#c87937','#8b657f','#8b929b'])
# Put the principal accepted category first.
ax.invert_yaxis()
# Leave room for numeric labels.
ax.set_xlim(0,80)
# Explain the actual denominator.
ax.set_xlabel('Percent of requested stock-time within saved continuous sessions')
# Identify the period and evidence source.
ax.set_title('January–June 2026: saved-window availability audit')
# Label every component with exact displayed percentage and stock-minutes.
for i,value in enumerate(values):
    # Keep annotations outside the bars for contrast.
    ax.text(value+0.7,i,f'{value:.4f}%',va='center')
# Save a durable visual beside the audit tables.
fig.savefig(out/'availability_breakdown.png',dpi=160)
# Print only concise final results.
print(json.dumps(summary,indent=2))
# Count unique windows per failure category without counting alternative arms twice.
sets = {reason:{(f['symbol'],f['date'],f['window_id']) for f in failures if f['reason']==reason} for reason in failure_counts}
# Preserve mutually exclusive failure combinations.
depth = sets.get('INSUFFICIENT_REPORTED_EXIT_DEPTH',set())
# Retrieve windows containing a cancellation-settlement failure.
cancel = sets.get('CANCELS_NOT_SETTLED_BEFORE_EXIT',set())
# Verify the two observed categories cover every rejected window.
assert len(depth | cancel) == totals['excluded_windows']
# Record counts suitable for comparison against the unique window total.
summary['unique_failure_windows'] = dict(depth_only=len(depth-cancel),cancel_only=len(cancel-depth),both=len(depth&cancel))
# Persist the extended summary.
(out/'summary.json').write_text(json.dumps(summary,indent=2))
# Build a concise report using calculated values.
lines = ['# Saved-window availability audit','', 'Scope: January–June 2026, 113 tickers, 13,560 requested stock-days. Saved evidence only; no strategy replay or settings changes.','', '| Component | Stock-minutes | Requested time |','|---|---:|---:|']
# Populate the duration table from exact millisecond totals.
for label,key in zip(labels,['accepted_ms','closing_excluded_ms','channel_flagged_ms','remaining_unattributed_ms']):
    # Preserve precise duration and percentage units.
    lines.append(f'| {label} | {totals[key]/60000:,.4f} | {100*totals[key]/totals["requested_ms"]:.4f}% |')
# Explain the source-count limitation and closing diagnostics.
lines += ['',f'Candidate windows: {totals["candidate_windows"]:,}. Common closing exclusions: {totals["excluded_windows"]:,}.', '',f'Unique rejected windows: {len(depth-cancel):,} reported-exit-depth failures only; {len(cancel-depth):,} cancellation-settlement failures only; {len(depth&cancel):,} with both categories across strategy alternatives.', '',f'There are {len(failures):,} failing arm-window records, of which {sum(abs(f["position"])>1e-9 for f in failures):,} retain a nonzero position. Alternatives must not be summed as concurrent portfolio exposure.', '',f'Largest absolute recorded share position: TELE, 2026-04-29, window 5, QT20_w3, 10,807 shares. Cash is -PKR 91,598.9061; this is not a loss estimate because inventory remains unvalued. Share size is not rupee risk across different stocks.', '', 'Source diagnostics record 9,027,012 ambiguous snapshot attempts, 3,681,415 future-order snapshot rejections, 116,973 unresolved-reference terminations, 86,016 invalid/exhausted-touch terminations and 76,058 short attempts. Counts overlap and do not measure unavailable duration. SOURCE_BOUNDARY also includes normal session boundaries. Invalid/noncontinuous checkpoints are skipped without a separate counter.', '', 'Timed channel attribution uses the union of saved flagged ranges and time outside the observed channel extent, clipped to the saved continuous sessions. It explains direct unavailable duration only; downstream waiting for a usable checkpoint remains unattributed. Sequence loss and capture-timing anomalies share one recorded category. No candidate window overlaps these timed ranges.', '', 'The remaining unavailable duration cannot honestly be divided into missing feed, snapshot uncertainty, book failure and short intervals from these saved counters. A source-only window-planner audit would need to record rejected interval boundaries and disjoint causes. It need not rerun strategy arms, but its runtime has not been measured.', '', 'Validation: exact integer duration reconciliation within each stock-day; agreement with coverage.csv to less than 0.001 ms; disjoint candidate intervals; identical session contracts across tickers; closing-exclusion count reconciliation; backward reconciliation of the four duration components. Inspected clean_window_data.py, clean_window_cell.py and clean_window_engine.py match recorded source hashes.', '', 'Permissions: disposable-file create/remove succeeded in all four authorized project folders. existing_mm_live was inspected read-only. Frozen selection and completed results were not edited.', '', 'Input: '+str(root), '', 'Full-day profitability, intraday account drawdown, official calendar alignment and unobserved-period exposure remain unestablished.']
# Save a readable durable report.
(out/'READ_ME.md').write_text('\n'.join(lines)+'\n')
# Print the final unique-window breakdown.
print('UNIQUE_FAILURE_WINDOWS',summary['unique_failure_windows'])
