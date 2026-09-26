# Parse the explicit historical coverage requested by the user.
import argparse
# Read dates without loading strategy or simulator modules.
from datetime import datetime
# Preserve a machine-readable result for every requested date.
import json
# Resolve paths without changing the parsed store.
from pathlib import Path
# Monitor actual elapsed wall time.
import time
# Bound independent daily readers to the requested worker count.
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
# Compare integer application sequences efficiently.
import numpy as np
# Read only sequence columns from the REG channel.
import pyarrow.dataset as ds

# Describe missing application ranges without treating missing events as zero profit.
def ranges(sequences):
    # Reject malformed numeric data before integer arithmetic.
    values=np.asarray(sequences)
    # Handle a wholly absent channel explicitly.
    if values.size == 0:
        # No finite ending sequence can be inferred from an empty capture.
        return dict(sequence_complete=False, reason='EMPTY_CHANNEL', missing_ranges=[], missing_messages=None, duplicates=0, observed=0)
    # Require integer parquet sequences, not rounded floating-point pointers.
    if values.dtype.kind not in 'iu' or np.any(values <= 0):
        # Preserve invalid input as a failure rather than repairing it silently.
        raise ValueError('Nonpositive or noninteger application sequence')
    # Keep one copy for locating gaps, but report duplicates as unresolved.
    unique=np.unique(values)
    # Count duplicate rows without asserting their payloads match.
    duplicates=int(values.size-unique.size)
    # Locate every missing interior sequence range.
    boundaries=np.flatnonzero(np.diff(unique)>1)
    # Record endpoints in ordinary JSON-compatible integers.
    missing=[[int(unique[i])+1,int(unique[i+1])-1] for i in boundaries]
    # The opening prefix must begin with the documented first message.
    if int(unique[0]) != 1:
        # Preserve missing opening messages as another incomplete range.
        missing.insert(0,[1,int(unique[0])-1])
    # Contiguity is necessary but not sufficient for an executable book.
    return dict(sequence_complete=not missing and duplicates==0, reason='SEQUENCE_GAPS' if missing else ('DUPLICATES_REQUIRE_PAYLOAD_CHECK' if duplicates else 'SEQUENCE_ONLY_PASSED'), missing_ranges=missing, missing_messages=sum(b-a+1 for a,b in missing), duplicates=duplicates, observed=int(values.size), first_sequence=int(unique[0]), last_sequence=int(unique[-1]))

# Inspect one date across every symbol on the channel, never per-symbol sequence gaps.
def audit_date(task):
    # Unpack the user-selected parsed root and date.
    root,date=task
    # Retain partition fingerprints to detect concurrent replacement.
    before={}
    # Combine both message families because they share application numbering.
    sequences=[]
    # Return errors as explicit dated evidence.
    try:
        # Neither the update nor trade table alone has contiguous sequence numbering.
        for name in ('ob_updates','trades'):
            # Locate this exact existing daily partition.
            folder=Path(root)/name/('date='+date)
            # Require actual saved source files.
            files=sorted(folder.glob('*.parquet'))
            # Never silently interpret an absent partition as no trading.
            if not files:
                # Retain the missing partition in the error output.
                raise ValueError('Missing partition: '+str(folder))
            # Freeze sizes and modification timestamps before reading.
            before.update({str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files})
            # Open this partition without loading price, quantity or snapshot columns.
            data=ds.dataset(str(folder),format='parquet')
            # Select the complete REG application channel, including every stock.
            column=data.to_table(columns=['appl_seq'],filter=ds.field('channel')==2011,use_threads=False).column('appl_seq')
            # Reject missing pointers instead of losing them in numpy conversion.
            if column.null_count:
                # This date cannot pass a completeness check.
                raise ValueError('Null application sequence')
            # Preserve exact integer sequence data.
            sequences.append(column.to_numpy(zero_copy_only=False))
        # Recheck saved-file identities after both reads finish.
        if any(not Path(p).is_file() or (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=identity for p,identity in before.items()):
            # Refuse mixed inputs during an active reparse.
            raise ValueError('Parsed inputs changed during audit')
        # Report observed contiguity without claiming opening or closing coverage.
        return dict(date=date,**ranges(np.concatenate(sequences)),files=before)
    # Preserve a failed date rather than dropping its denominator.
    except Exception as error:
        # Never mark an errored date as safe.
        return dict(date=date,sequence_complete=False,reason='AUDIT_ERROR',error=repr(error))

# Run a lightweight completeness screen before any expensive profit replay.
def main():
    # Keep the command self-contained and independent of the old checkout.
    parser=argparse.ArgumentParser()
    # Require an explicit immutable source location.
    parser.add_argument('--parsed-root',type=Path,required=True)
    # Default to the agreed comparison period, excluding calibration-only September.
    parser.add_argument('--start-date',default='2025-10-01')
    # Preserve the agreed final history date.
    parser.add_argument('--end-date',default='2026-06-30')
    # Bound disk and CPU concurrency.
    parser.add_argument('--workers',type=int,default=8)
    # Keep every output outside source control.
    parser.add_argument('--output-dir',type=Path,required=True)
    # Read the actual terminal arguments.
    args=parser.parse_args()
    # Reject accidental oversubscription or reversed dates.
    if not 1<=args.workers<=8 or args.start_date>args.end_date:
        # Fail before reading any source partitions.
        parser.error('Require 1..8 workers and start-date <= end-date')
    # Form the union so a missing table is reported rather than omitted.
    dates=sorted({p.name[5:] for table in ('ob_updates','trades','ob_snapshot') for p in (args.parsed_root/table).glob('date=*') if args.start_date<=p.name[5:]<=args.end_date})
    # Do not manufacture a successful empty audit.
    if not dates:
        # Explain the source-range problem directly.
        raise ValueError('No date partitions found in the requested range')
    # Preserve previous evidence by requiring a new directory.
    args.output_dir.mkdir(parents=True,exist_ok=False)
    # Measure total elapsed time and heartbeat cadence.
    started=last=time.monotonic()
    # Retain one small result per date.
    results=[]
    # Announce that this command cannot launch a profit simulation.
    print(f'[{datetime.now():%H:%M:%S}] Source sequence audit | Elapsed 0m | ETA estimating | Done 0/{len(dates)}',flush=True)
    # Use independent date readers with no strategy simulation.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # Track actual unfinished dates.
        pending={pool.submit(audit_date,(str(args.parsed_root),d)):d for d in dates}
        # Collect all outcomes, including bad dates.
        while pending:
            # Wake for progress without flooding the terminal.
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            # Persist each completed date immediately.
            for future in done:
                # Preserve worker exceptions with their requested date.
                date=pending.pop(future)
                # Convert exceptional worker failures into explicit audit failures.
                try:
                    # Read the completed audit result.
                    result=future.result()
                # Keep worker failures visible in the final denominator.
                except Exception as error:
                    # Do not treat failure as a clean channel.
                    result=dict(date=date,sequence_complete=False,reason='WORKER_ERROR',error=repr(error))
                # Retain the result for summary counts.
                results.append(result)
                # Save independently so interrupted runs still have dated evidence.
                (args.output_dir/(date+'.json')).write_text(json.dumps(result,indent=2))
            # Print at most one compact heartbeat every fifteen seconds.
            if time.monotonic()-last>=15:
                # Estimate remaining audit time from completed dates only.
                elapsed=(time.monotonic()-started)/60
                # Keep a missing estimate honest until the first date finishes.
                eta=f'~{elapsed*(len(dates)-len(results))/len(results):.1f}m' if results else 'estimating'
                # Include an unfinished date without verbose event details.
                print(f'[{datetime.now():%H:%M:%S}] Date {next(iter(pending.values()),"done")} | Source audit | Elapsed {elapsed:.1f}m | ETA {eta} | Done {len(results)}/{len(dates)}',flush=True)
                # Reset the heartbeat clock.
                last=time.monotonic()
    # Count only the narrow contiguity check as passed.
    passed=sum(r['sequence_complete'] for r in results)
    # Never turn this preliminary screen into approval to trade or publish P&L.
    summary=dict(completed=True,requested_dates=len(dates),sequence_only_passed=passed,failed_or_incomplete=len(dates)-passed,pnl_run_ready=False,live_approved=False,qualification='Sequence contiguity only. Opening state, closing coverage, retransmission payloads, snapshot recovery and simulator integration remain separate checks. Missing-data dates have unknown profit, not zero profit.')
    # Preserve the qualification next to the audit result.
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    # Display the concise result and its explicit limits.
    print(json.dumps(summary),flush=True)

# Avoid spawning workers while importing this helper in tests.
if __name__=='__main__':
    # Execute only the user-requested source audit.
    main()
