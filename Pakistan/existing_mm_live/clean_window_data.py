# Locate source partitions without writing to them.
from pathlib import Path
# Normalize captured rows and exact window boundaries.
import pandas as pd
# Read only required source columns.
import pyarrow.dataset as ds
# Search timestamp boundaries without rescanning whole symbol histories.
from bisect import bisect_left, bisect_right
# Preserve mutable simulator-compatible event rows.
from types import SimpleNamespace
# Reuse the saved snapshot schema conversion.
from snapshot_prep import prep_snapshot
# Reuse canonical timestamp conversion and partition access only.
import run_legacy_mm as R
# Normalize exact pointers and quantities.
from psx_reference_rows import reference
# Validate reported price and quantity units.
from psx_reference_book import units
# Build a bounded known-price reconstruction for each independent window.
from clean_window_book import WindowBook, WindowDataError

# Screen the entire channel once per date, before filtering to individual names.
def channel_intervals(root,date):
    # Combine adds, cancels and trades, which share one sequence stream.
    frames=[]
    # Preserve file metadata as a consistency check, not a claimed content hash.
    metadata={}
    # Read only channel sequence and timing columns.
    for name in ('ob_updates','trades'):
        # Select the actual daily partition.
        folder=Path(root)/name/('date='+date)
        # Require the source files rather than silently treating absence as zero trades.
        files=sorted(folder.glob('*.parquet'))
        # Missing partitions cannot supply any usable windows.
        if not files:
            # Fail before a long strategy search.
            raise ValueError('Missing '+str(folder))
        # Freeze metadata before reading.
        metadata.update({str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in files})
        # Read the entire REG channel, not one symbol's naturally sparse numbering.
        frame=ds.dataset(str(folder),format='parquet').to_table(columns=['appl_seq','transact_time','capture_ts'],filter=ds.field('channel')==2011,use_threads=False).to_pandas()
        # Preserve every source record for duplicate and gap checks.
        frames.append(frame)
    # Combine both message families before checking consecutive sequence numbers.
    frame=pd.concat(frames,ignore_index=True)
    # Reject missing or duplicate pointers rather than silently choosing a payload.
    if frame.empty or frame.transact_time.isna().any() or frame.capture_ts.isna().any() or frame.appl_seq.isna().any() or frame.appl_seq.duplicated().any():
        # Keep this date in the requested denominator with explicit unavailability.
        return dict(date=date,unusable=True,reason='EMPTY_NULL_OR_DUPLICATE_CHANNEL',intervals=[],metadata=metadata)
    # Preserve the exchange's application order.
    frame=frame.sort_values('appl_seq')
    # Convert the two clocks without assuming their offsets are identical.
    times=R.to_ms(frame.transact_time).to_numpy()
    # Capture time is used only to identify conservative feed pauses.
    captures=R.to_ms(frame.capture_ts).to_numpy()
    # Preserve exact integer application pointers.
    sequences=frame.appl_seq.to_numpy(dtype='int64')
    # Reject an inconsistent exchange chronology in this first research implementation.
    if (times[1:]<times[:-1]).any():
        # Do not reorder market mutations to hide a source inconsistency.
        return dict(date=date,unusable=True,reason='EXCHANGE_TIME_REVERSAL',intervals=[],metadata=metadata)
    # Collect inclusive intervals that cannot support a continuous replay.
    intervals=[]
    # Exclude an unavailable opening prefix without excluding the entire day.
    if sequences[0]!=1:
        # A later checkpoint can restart the separate flat-start experiment.
        intervals.append([0,int(times[0]),'MISSING_OPENING_PREFIX'])
    # Detect missing application records and long channel-level receive pauses.
    for index in ((sequences[1:]!=sequences[:-1]+1)|((captures[1:]-captures[:-1]>7000)|(captures[1:]<captures[:-1]))).nonzero()[0]:
        # Preserve both bordering source times; windows may not cross this interval.
        intervals.append([int(times[index]),int(times[index+1]),'MISSING_SEQUENCE_OR_CHANNEL_TIMING'])
    # Check raw files were not replaced while the screen was running.
    if any([Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=value for p,value in metadata.items()):
        # Never mix a reparse with an ongoing experiment.
        raise ValueError('Channel inputs changed during read')
    # Preserve a narrow claim: this identifies unusable intervals, not live recovery.
    return dict(date=date,unusable=False,intervals=sorted(intervals),metadata=metadata,records=len(frame),first_ms=int(times[0]),last_ms=int(times[-1]))

# Load a symbol-day once, preserving raw references and snapshot message identity.
def load_symbol(date,symbol):
    # Open existing parquet partitions in the active configured data root.
    datasets=R.open_datasets(date)
    # Fail explicitly when the date is incomplete.
    if datasets is None:
        # Do not silently skip a requested name/date.
        raise ValueError('Missing source date '+date)
    # Retain all fields required to reconstruct orders, queues and checkpoint timing.
    extra=['channel','buy_ref','sell_ref']
    # Load REG updates only.
    updates=R.read_symbol(datasets['ob_updates'],list(dict.fromkeys(R.REQ_UPDATES+['market']+extra)),symbol,market='REG')
    # Load REG trades only.
    trades=R.read_symbol(datasets['trades'],list(dict.fromkeys(R.REQ_TRADES+['market']+extra)),symbol,market='REG')
    # Keep per-message capture identity and explicit reported level indices.
    snapshots=R.read_symbol(datasets['ob_snapshot'],list(dict.fromkeys(R.REQ_SNAP+['channel','level'])),symbol,market='REG')
    # Preserve each add's actual immutable source identity and price.
    adds={}
    # Build one merged incremental event list.
    events=[]
    # Normalize exchange and observed clocks once per source row.
    for frame,kind in ((updates,'U'),(trades,'T')):
        # Reject missing source time fields.
        if frame.transact_time.isna().any() or frame.capture_ts.isna().any():
            # The clean-window experiment must not guess event times.
            raise WindowDataError('MISSING_EVENT_TIME')
        # Preserve millisecond source timestamps.
        frame['ts_exch']=R.to_ms(frame.transact_time)
        # Prevent negative effective feed delay from putting orders before their source event.
        frame['ts_cap']=R.to_ms(frame.capture_ts).clip(lower=frame.ts_exch)
        # Create mutable rows for queue-identity resolution at fill time.
        for record in frame.to_dict('records'):
            # Use the raw application identity unchanged.
            row=SimpleNamespace(**record)
            # Validate positive quantities at the source boundary.
            if units(row.qty,100,'quantity')<=0:
                # A malformed mutation cannot enter a profitable window.
                raise WindowDataError('NONPOSITIVE_EVENT_QUANTITY')
            # Source adds provide reference-to-price mapping, never future remaining state.
            if kind=='U' and row.event=='ORDER_ADD':
                # Normalize the authoritative application pointer.
                seq=reference(row.appl_seq)
                # Require one valid source add per pointer.
                if seq in adds or row.side not in ('BUY','SELL') or units(row.price,10000,'price')<=0:
                    # Duplicate or corrupt adds invalidate this input.
                    raise WindowDataError('INVALID_ADD_INDEX')
                # Preserve an exchange ID when present, otherwise use its genuine source pointer.
                oid=str(row.order_id) if pd.notna(row.order_id) and str(row.order_id) else f'ref:2011:{seq}'
                # Store no future fill/cancel-derived quantity in this lookup.
                adds[seq]=dict(oid=oid,side=row.side,price=float(row.price),ts=int(row.ts_exch))
                # Give the simulator the same identity used by the book.
                row.order_id=oid
            # Queue identities of reductions are resolved against the actual window book later.
            if kind=='T':
                # Do not trust old parser-resolved IDs.
                row.rest_oid=None
            # Preserve source sequence ordering for equal timestamps.
            events.append((int(row.ts_exch),1,reference(row.appl_seq),kind,row))
    # Keep the original source mutation order.
    events.sort(key=lambda event:event[2])
    # Reject repeated pointers or timestamp reversal before selecting windows.
    if any(b[2]<=a[2] or b[0]<a[0] for a,b in zip(events,events[1:])):
        # Do not repair chronology by sorting inconsistent prices into place.
        raise WindowDataError('SYMBOL_EVENT_ORDER')
    # Keep snapshot origin and capture times as separate fields.
    snapshots['origin_ms']=R.to_ms(snapshots.orig_time)
    # A captured snapshot cannot be used before receipt.
    snapshots['capture_ms']=R.to_ms(snapshots.capture_ts)
    # Group each distinct received message, not just its reusable FIX sequence.
    checkpoints=[]
    # Preserve all scalar identities that separate snapshot messages.
    for key,group in snapshots.groupby(['channel','msg_seq','orig_time','capture_ts'],sort=False):
        # Read the actual source clock interval and receipt time.
        origin=int(group.origin_ms.iloc[0])
        # Wait through the whole coarse source second, and until receipt if later.
        available=max(origin+1000,int(group.capture_ms.iloc[0]))
        # Validate that visible level numbering is contiguous on each side.
        valid=True
        # Neither duplicated nor missing visible ranks can define a certified price range.
        for side in ('BID','OFFER'):
            # Read only rows representing actual quoted prices.
            ranks=sorted(group.loc[group.entry_type==side,'level'].dropna().astype(int).tolist())
            # Accept fewer than ten levels only when they begin at the actual touch.
            valid &= bool(ranks) and ranks==list(range(1,len(ranks)+1))
        # Preserve status snapshots too, so phase changes terminate windows.
        checkpoints.append(dict(origin=origin,start=available,snapshot=prep_snapshot(group),valid=valid))
    # Use source origin order, with delayed availability retained explicitly.
    checkpoints.sort(key=lambda value:(value['origin'],value['start']))
    # Return decoded source once for all twelve configurations.
    return events,checkpoints,adds

# Select common nonoverlapping windows using source quality only, never strategy profits.
def plan_windows(events,checkpoints,adds,quality,segments,min_ms=60000):
    # Preserve every rejection category for the coverage report.
    counts={}
    # Retain no fictitious usable interval for an unusable channel day.
    if quality['unusable']:
        # Expose the reason without generating a zero-profit result.
        return [],{quality['reason']:1}
    # Use fast timestamp lookups for snapshot alignment and event slices.
    times=[event[0] for event in events]
    # Keep accepted windows disjoint across every arm.
    windows=[]
    # Never select a new checkpoint inside an already evaluated interval.
    next_start=-1
    # Cache observed phases that definitively end continuous trading.
    phase_ends=[c['origin'] for c in checkpoints if c['snapshot'].phase!='CONTINUOUS_AUCTION']
    # Cache actual band-change times once instead of scanning all snapshots per candidate.
    band_changes=[c['origin'] for previous,c in zip(checkpoints,checkpoints[1:]) if (c['snapshot'].limit_up,c['snapshot'].limit_dn)!=(previous['snapshot'].limit_up,previous['snapshot'].limit_dn)]
    # Preserve sorted channel-gap starts for logarithmic next-boundary lookup.
    gap_starts=sorted(a for a,b,reason in quality['intervals'])
    # Reject snapshots containing a source add known to occur after their availability.
    by_oid={value['oid']:value for value in adds.values()}
    # Test source checkpoints in their original sequence.
    for checkpoint in checkpoints:
        # Read the conservative snapshot availability boundary.
        start=checkpoint['start']
        # Skip checkpoints already covered, malformed, or outside continuous trading.
        if start<next_start or not checkpoint['valid'] or checkpoint['snapshot'].phase!='CONTINUOUS_AUCTION':
            # Keep searching without running any strategy.
            continue
        # Find the actual calibrated continuous segment containing this checkpoint.
        segment=next(((a,b) for a,b in segments if a<=start<b),None)
        # Exclude breaks and out-of-session records.
        if segment is None:
            # No trading is authorized in this interval.
            continue
        # A coarse snapshot can seed a window only if no symbol mutation intervened.
        if bisect_right(times,start)!=bisect_left(times,checkpoint['origin']):
            # Do not move an unknown snapshot ahead of same-second ticks.
            counts['SNAPSHOT_TIME_AMBIGUOUS']=counts.get('SNAPSHOT_TIME_AMBIGUOUS',0)+1
            # Wait for a later unambiguous checkpoint.
            continue
        # Reject checkpoints whose uncertainty interval intersects a channel outage.
        if any(a<=start and b>=checkpoint['origin'] for a,b,reason in quality['intervals']):
            # The missing records might have changed the snapshot state before availability.
            counts['CHECKPOINT_OVERLAPS_GAP']=counts.get('CHECKPOINT_OVERLAPS_GAP',0)+1
            # Try a genuinely later checkpoint instead.
            continue
        # Check disclosed order identities against all known add times without importing future state.
        if any(oid in by_oid and by_oid[oid]['ts']>start for side,price,qty,ids,qs in checkpoint['snapshot'].levels for oid in ids.split('|') if oid):
            # A snapshot containing a later order is not a verified restart boundary.
            counts['FUTURE_ORDER_IN_SNAPSHOT']=counts.get('FUTURE_ORDER_IN_SNAPSHOT',0)+1
            # Never quote from that future-containing book.
            continue
        # End before the next channel outage, non-continuous phase, or calibrated segment close.
        limits=[segment[1],quality['last_ms']]
        # Read only the first later change in each sorted boundary stream.
        for boundaries in (band_changes,gap_starts,phase_ends):
            # Find the first boundary strictly after this checkpoint.
            index=bisect_right(boundaries,start)
            # A remaining change ends the window before its entire timestamp.
            if index<len(boundaries):
                # Exclude all same-time mutations at the boundary.
                limits.append(boundaries[index]-1)
        # Use the earliest source-quality boundary for every strategy arm.
        end=min(limits)
        # Build a fresh checkpoint book solely to validate the market-data slice.
        try:
            # Aggregate undisclosed quantity only at actually reported prices.
            book=WindowBook(checkpoint['snapshot'],adds)
        # Reject a malformed checkpoint without invoking a strategy.
        except WindowDataError as error:
            # Count the specific quality failure.
            counts[str(error)]=counts.get(str(error),0)+1
            # Continue to the next real checkpoint.
            continue
        # Start after the quiet checkpoint boundary.
        left=bisect_right(times,start)
        # Retain the first inconsistent timestamp as an excluded whole batch.
        failure='SOURCE_BOUNDARY'
        # Validate incrementals until the common prospective boundary.
        for event in events[left:bisect_right(times,end)]:
            # Preserve original source kind and payload.
            ts,rank,sequence,kind,row=event
            # Test the mutation without any strategy state.
            try:
                # Apply source quantities exactly once to the bounded book.
                (book.trade if kind=='T' else book.add if row.event=='ORDER_ADD' else book.cancel)(row)
                # Stop when the certified price range no longer supplies a valid touch.
                book.check_touch()
            # Data errors terminate the window before this entire exchange timestamp.
            except WindowDataError as error:
                # Avoid selecting earlier sub-events of a bad same-timestamp batch.
                end=ts-1
                # Preserve the precise reason for the coverage report.
                failure=str(error)
                # No later event in this interval may be used by any arm.
                break
        # Record the common termination cause even for short discarded intervals.
        counts[failure]=counts.get(failure,0)+1
        # Advance beyond this attempted interval, avoiding repeated overlapping searches.
        next_start=max(start+1,end+1)
        # Keep only intervals long enough for quoting and explicit exit/cancel buffers.
        if end-start>=min_ms:
            # Retain the checkpoint and exact source-event range without copying rows per arm.
            windows.append(dict(start=start,end=end,left=left,right=bisect_right(times,end),checkpoint=checkpoint['snapshot'],ending=failure))
        # Account for unusably short clean slices separately.
        else:
            # No zero-profit window is added to the result.
            counts['TOO_SHORT']=counts.get('TOO_SHORT',0)+1
    # Return the source-only matched design and every exclusion category.
    return windows,counts
