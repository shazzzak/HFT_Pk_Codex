# Parse a bounded pilot or an explicitly requested full source-only audit.
import argparse
# Store reproducible evidence and exact source identities.
import csv, hashlib, importlib.util, json, os, sys, time
# Summarize deterministic workload strata and observed runtimes.
import statistics
# Group source records without silently dropping empty strata.
from collections import defaultdict
# Spawn isolated source-reconstruction workers.
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
# Use spawn rather than inheriting imported mutable book modules.
import multiprocessing as mp
# Classify saved session dates and label heartbeat timestamps.
from datetime import date, datetime
# Resolve all paths against this authorized implementation.
from pathlib import Path
# Locate this script and the read-only legacy dependency directory.
HERE=Path(__file__).resolve().parent
# Never use the superseded HFT/Pakistan project.
LEGACY=HERE.parents[1]/'existing_mm_live'
# Make current local modules take priority over read-only dependencies.
sys.path.insert(0,str(HERE))
# Permit legacy dependency imports without writing bytecode there.
sys.path.append(str(LEGACY))
# Disable bytecode writes even if the caller omitted the environment variable.
sys.dont_write_bytecode=True
# Locate the completed baseline experiment.
DEFAULT_RUN=Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/clean_window_full_20260925_v1')
# Read exact JSON evidence using UTF-8.
def read_json(path):
    # Return decoded content without modifying the evidence file.
    return json.loads(Path(path).read_text())
# Hash exact bytes rather than modification dates for Python source identity.
def sha(path):
    # Return an ordinary SHA-256 content fingerprint.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
# Persist completed evidence atomically and reject nonfinite output numbers.
def save(path,value):
    # Write beside the destination so replacement remains atomic.
    temporary=Path(str(path)+'.tmp')
    # Keep JSON strict and readable.
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    # Publish only a fully serialized record.
    temporary.replace(path)
# Capture current source files and their historical read-only dependencies.
def source_hashes():
    # Include unused modules conservatively rather than miss a helper import.
    return {str(p):sha(p) for root in (HERE,LEGACY) for p in sorted(root.glob('*.py'))}
# Normalize the exact candidate-window fields saved by the historical runner.
def signature(windows):
    # Deliberately exclude strategy closing decisions from source comparison.
    return [[w['start'],w['end'],w['ending']] for w in windows]
# Form nonoverlapping unions before comparing source-window coverage.
def intervals(windows):
    # Sort fresh endpoint pairs without mutating window objects.
    values=sorted((w['start'],w['end']) for w in windows)
    # Require positive durations and disjoint candidate windows.
    if any(b<=a for a,b in values) or any(a<previous for (_,previous),(a,_) in zip(values,values[1:])):
        # An invalid planner result must not enter coverage arithmetic.
        raise ValueError('Nonpositive or overlapping source windows')
    # Return validated sorted intervals.
    return values
# Measure shared duration with a linear sweep over disjoint intervals.
def intersection_ms(left,right):
    # Initialize two independent cursors and an exact millisecond total.
    i=j=total=0
    # Stop when either input has no remaining interval.
    while i<len(left) and j<len(right):
        # Add the nonnegative overlap of the current interval pair.
        total+=max(0,min(left[i][1],right[j][1])-max(left[i][0],right[j][0]))
        # Advance the interval that ends first, including ties.
        if left[i][1]<=right[j][1]:
            # Move the left cursor only after its duration has been consumed.
            i+=1
        # Otherwise the right interval has ended first.
        else:
            # Move the right cursor forward.
            j+=1
    # Preserve exact integer units through aggregation.
    return total
# Build a deterministic diagnostic sample using source structure, never profit.
def select_jobs(rows):
    # Group every requested stock-day by symbol.
    by_symbol=defaultdict(list)
    # Preserve the original coverage records for workload classification.
    for row in rows:
        # Collect all six-month observations for each ticker.
        by_symbol[row['symbol']].append(row)
    # Rank symbols by mean source-window count, a fragmentation proxy, not liquidity.
    ranked=sorted(by_symbol,key=lambda s:(statistics.mean(int(r['source_windows']) for r in by_symbol[s]),s))
    # Split the ranked universe into three deterministic workload strata.
    strata={s:min(2,3*i//len(ranked)) for i,s in enumerate(ranked)}
    # Always retain the three existing cheap-tick exceptions when present.
    chosen=[s for s in ('KEL','PIBTL','TPL') if s in by_symbol]
    # Add three evenly spaced names from each workload stratum.
    for group in range(3):
        # Exclude already selected exceptions from supplementary picks.
        candidates=[s for s in ranked if strata[s]==group and s not in chosen]
        # Choose low, middle and high positions within each remaining group.
        for index in sorted({0,len(candidates)//2,len(candidates)-1}):
            # Ignore an empty stratum without indexing a missing symbol.
            if candidates:
                # Keep these predetermined names independently of observed correction gains.
                chosen.append(candidates[index])
    # Require the intended twelve-name pilot on the saved 113-name universe.
    if len(chosen)!=12 or len(set(chosen))!=12:
        # Reject a changed universe rather than silently narrowing the pilot.
        raise ValueError('Expected 12 unique pilot names including KEL, PIBTL and TPL')
    # Index saved per-date session duration without inferring official schedules.
    daily={r['date']:float(r['requested_minutes']) for r in rows}
    # Group dates by month and observed session type.
    date_groups=defaultdict(lambda:defaultdict(list))
    # Classify continuous-session totals using a documented diagnostic threshold.
    for day,minutes in sorted(daily.items()):
        # Distinguish Fridays from other weekdays.
        weekday='Friday' if date.fromisoformat(day).weekday()==4 else 'weekday'
        # Ordinary split Fridays total about 300 minutes, so use a distinct shorter-session threshold.
        threshold=270 if weekday=='Friday' else 300
        # These observed-duration categories are diagnostics, not official Ramadan classifications.
        kind=('short ' if minutes<threshold else 'regular ')+weekday
        # Preserve every date in its explicit month/session group.
        date_groups[day[:7]][kind].append(day)
    # Preserve requested cells for lookup and full-run stratification.
    lookup={(r['symbol'],r['date']):r for r in rows}
    # Accumulate the predetermined pilot manifest.
    pilot=[]
    # Select one date per chosen ticker in every month.
    for month,groups in sorted(date_groups.items()):
        # Rotate equally across the available observed-session categories.
        kinds=sorted(groups)
        # Avoid repeatedly choosing only the first date of a month.
        for index,symbol in enumerate(chosen):
            # Assign a session category independently of profit or coverage improvement.
            kind=kinds[index%len(kinds)]
            # Choose a deterministic spread of dates within that category.
            dates=groups[kind]
            # Use a reproducible position across the complete category range.
            day=dates[min(len(dates)-1,(index//len(kinds))*len(dates)//max(1,(len(chosen)+len(kinds)-1)//len(kinds))) ]
            # Require this exact requested stock-day to exist in the saved baseline.
            if (symbol,day) not in lookup:
                # Do not replace missing selected jobs with favorable alternatives.
                raise ValueError('Missing selected stock-day: '+symbol+' '+day)
            # Record the actual session group alongside the workload stratum.
            pilot.append(dict(symbol=symbol,date=day,stratum=strata[symbol],session_group=kind))
    # Construct every full-period job with the same workload strata.
    full=[dict(symbol=r['symbol'],date=r['date'],stratum=strata[r['symbol']]) for r in rows]
    # Return both scopes so runtime projection uses the actual population size.
    return pilot,full
# Inspect partition membership and metadata without reading entire parquet contents.
def partition_metadata(root,day):
    # Preserve all three required source tables, including checkpoint snapshots.
    paths=[p for table in ('ob_updates','trades','ob_snapshot') for p in sorted((root/table/('date='+day)).glob('*.parquet'))]
    # Refuse missing input tables before source reconstruction.
    if any(not list((root/table/('date='+day)).glob('*.parquet')) for table in ('ob_updates','trades','ob_snapshot')):
        # Unknown missing data cannot become an empty successful cell.
        raise ValueError('Missing parquet partition: '+day)
    # Record exact membership, size and nanosecond modification time.
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
# Load an isolated baseline module without replacing corrected modules in sys.modules.
def module_at(name,path):
    # Resolve source definitions from the explicit historical location.
    spec=importlib.util.spec_from_file_location(name,path)
    # Allocate a separate module namespace.
    module=importlib.util.module_from_spec(spec)
    # Load definitions only; no script entry point is invoked.
    spec.loader.exec_module(module)
    # Return the isolated historical implementation.
    return module
# Initialize each spawned worker once to avoid repeated module imports.
def initialize(run,quality,metadata,progress):
    # Keep worker-local references separate from each stock-day's book state.
    global BASELINE,NEW,BOOK,RUN,QUALITY,METADATA,PROGRESS
    # Import the corrected source planner, not a strategy runner.
    import clean_window_data as NEW
    # Retain the corrected class before installing per-cell instrumentation.
    BOOK=NEW.WindowBook
    # Load the original book with its own exception type.
    old_book=module_at('audit_original_book',LEGACY/'clean_window_book.py')
    # Load the original planner into a separate namespace.
    BASELINE=module_at('audit_original_data',LEGACY/'clean_window_data.py')
    # Bind the original implementation and its matching data exception.
    BASELINE.WindowBook,BASELINE.WindowDataError=old_book.WindowBook,old_book.WindowDataError
    # Retain immutable run evidence and per-date input fingerprints.
    RUN,QUALITY,METADATA,PROGRESS=Path(run),quality,metadata,progress
# Compare one source-day and measure successful anonymous reductions.
def audit_cell(job):
    # Keep exact elapsed time for workload-based projection.
    began=time.monotonic()
    # Identify the source cell in the shared heartbeat state.
    key=job['symbol']+'_'+job['date']
    # Publish the source-loading phase before accessing parquet data.
    PROGRESS[key]='loading source'
    # Preserve explicit failures instead of dropping cells from summaries.
    try:
        # Read the baseline's saved session contract and exact candidate windows.
        saved=read_json(RUN/key/'result.json')
        # Require the completed historical cell to be valid.
        if not saved.get('passed'):
            # An unsuccessful baseline needs investigation, not comparison.
            raise ValueError('Saved baseline cell did not pass')
        # Check source membership and metadata immediately before reading.
        if partition_metadata(NEW.R.PARSED_ROOT,job['date'])!=METADATA[job['date']]:
            # Prevent a concurrent source replacement from contaminating comparison.
            raise ValueError('Source partitions changed before load')
        # Decode this symbol-day once for both book implementations.
        events,checkpoints,adds=NEW.load_symbol(job['date'],job['symbol'])
        # Mark the historical reproduction phase.
        PROGRESS[key]='baseline source planner'
        # Recreate original candidate windows and rejection counters.
        old,old_counts=BASELINE.plan_windows(events,checkpoints,adds,QUALITY[job['date']],saved['job']['params']['session_segments'])
        # Require exact agreement with saved evidence, not rounded duration only.
        if signature(old)!=signature(saved['windows']) or old_counts!=saved['source_exclusions']:
            # Stop interpretation if the historical baseline cannot be reproduced.
            raise ValueError('Historical candidate windows or counters do not reproduce')
        # Preserve unique successfully applied anonymous source events.
        successful={}
        # Count attempts separately because short/rejected slices are also examined.
        calls=0
        # Instrument the book without changing its trading or reconstruction rules.
        class CountingBook(BOOK):
            # Observe reductions only after the actual quantity update succeeds.
            def reduce(self,row):
                # Update the cell-local attempt counter rather than global results.
                nonlocal calls
                # Resolve before mutation without learning or changing depth.
                pool,side,price=self.resolve(row)
                # Apply the exact corrected reduction once.
                super().reduce(row)
                # Count only visible reductions to an actual anonymous price pool.
                if pool is not None and pool.startswith('__WINDOW_POOL_'):
                    # Classify the source event from its explicit update label.
                    kind='cancel' if getattr(row,'event',None)=='CANCEL' else 'trade'
                    # Read the authoritative one-sided reference with exact conversion.
                    ref=NEW.reference(getattr(row,'buy_ref',None)) or NEW.reference(getattr(row,'sell_ref',None))
                    # Distinguish missing-add correction events from preexisting anonymous handling.
                    missing=ref not in self.adds
                    # Deduplicate by source kind and exact application pointer.
                    successful[(kind,NEW.reference(row.appl_seq))]=(int(row.ts_exch),kind,missing)
                    # Retain all successful application calls, including discarded short attempts.
                    calls+=1
        # Use instrumentation only in this worker's corrected planner.
        NEW.WindowBook=CountingBook
        # Mark the corrected source-planning phase in the heartbeat.
        PROGRESS[key]='corrected source planner'
        # Restore the original class even if a source error escapes.
        try:
            # Use identical events, checkpoints, channel screen and session boundaries.
            new,new_counts=NEW.plan_windows(events,checkpoints,adds,QUALITY[job['date']],saved['job']['params']['session_segments'])
        # Never leak one cell's instrumentation into the next task.
        finally:
            # Restore the corrected uninstrumented class.
            NEW.WindowBook=BOOK
        # Verify source files stayed unchanged throughout this cell.
        if partition_metadata(NEW.R.PARSED_ROOT,job['date'])!=METADATA[job['date']]:
            # Reject mixed input generations rather than publishing partial findings.
            raise ValueError('Source partitions changed during comparison')
        # Validate disjoint interval arithmetic for both reconstructions.
        left,right=intervals(old),intervals(new)
        # Calculate original and corrected usable milliseconds exactly.
        old_ms,new_ms=sum(b-a for a,b in left),sum(b-a for a,b in right)
        # Distinguish newly available time from time lost after changed restart choices.
        shared=intersection_ms(left,right)
        # Count only events lying inside retained corrected windows.
        retained=[v for v in successful.values() if any(a<v[0]<=b for a,b in right)]
        # Record all dimensions necessary to interpret counts and timing honestly.
        result=dict(job=job,passed=True,elapsed_seconds=time.monotonic()-began,event_count=len(events),checkpoint_count=len(checkpoints),requested_ms=saved['requested_ms'],old_source_ms=old_ms,new_source_ms=new_ms,gained_ms=new_ms-shared,lost_ms=old_ms-shared,net_change_ms=new_ms-old_ms,old_windows=len(old),new_windows=len(new),anonymous_successful_application_calls_all_attempts=calls,anonymous_unique_events_all_attempts=len(successful),anonymous_unique_events_retained=len(retained),missing_add_trades_retained=sum(kind=='trade' and missing for ts,kind,missing in retained),missing_add_cancels_retained=sum(kind=='cancel' and missing for ts,kind,missing in retained),unpriced_cancel_attempts=new_counts.get('UNPRICED_ANONYMOUS_CANCEL',0),old_counts=old_counts,new_counts=new_counts,old_intervals=signature(old),new_intervals=signature(new))
        # Expose how often retained windows actually replace their market picture.
        result['snapshot_refreshes_retained']=sum(len(w.get('refreshes',[])) for w in new)
        # Preserve exact replacement times for reproducible source-only comparisons.
        result['snapshot_refresh_times']=[c['start'] for w in new for c in w.get('refreshes',[])]
        # Reconcile gross gains minus gross losses to the net change.
        if result['gained_ms']-result['lost_ms']!=result['net_change_ms']:
            # Treat accounting disagreement as an implementation error.
            raise ValueError('Coverage difference failed reconciliation')
        # Return completed evidence for atomic parent-owned persistence.
        return result
    # Keep failed jobs visible with their original requested identity.
    except Exception as error:
        # Preserve the failure without assigning it zero coverage or zero profit.
        return dict(job=job,passed=False,error=repr(error),elapsed_seconds=time.monotonic()-began)
    # Remove the active heartbeat label on either success or failure.
    finally:
        # Completed jobs are counted only by the parent after receiving their results.
        PROGRESS.pop(key,None)
# Write tabular summaries without changing source artifacts.
def write_csv(path,rows):
    # Skip an empty table rather than invent column meanings.
    if not rows:
        # The JSON summary still records the empty/failed scope.
        return
    # Open only a newly generated report path.
    with path.open('w',newline='') as stream:
        # Preserve explicit column order from the first record.
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        # Include an ordinary CSV header for independent analysis.
        writer.writeheader()
        # Preserve every selected record.
        writer.writerows(rows)
# Execute only the explicitly chosen source-only scope.
def main():
    # Define a safe pilot default and an explicit full-audit gate.
    parser=argparse.ArgumentParser(description='Source-only coverage audit; never runs trading strategies')
    # Reuse saved baseline evidence rather than creating a new strategy run.
    parser.add_argument('--saved-run',type=Path,default=DEFAULT_RUN)
    # Default to a 72-cell purposive pilot across twelve names and six months.
    parser.add_argument('--scope',choices=['smoke','pilot','full'],default='pilot')
    # Require explicit intent before processing all January–June source cells.
    parser.add_argument('--confirm-full-source-audit',action='store_true')
    # Keep the user's usual worker preference configurable.
    parser.add_argument('--workers',type=int,default=8)
    # Permit scope review without reading source parquet or starting workers.
    parser.add_argument('--plan-only',action='store_true')
    # Require an explicit output destination so resumptions are deliberate.
    parser.add_argument('--output-dir',type=Path,required=True)
    # Resume only exact matching input/code identities and successful saved cells.
    parser.add_argument('--resume',action='store_true')
    # Decode the requested scope once.
    args=parser.parse_args()
    # Refuse invalid worker counts and accidental full-scope dispatch.
    if args.workers<1 or (args.scope=='full' and not args.confirm_full_source_audit):
        # Explain the missing scope control before loading any market data.
        parser.error('Use positive workers; full scope requires --confirm-full-source-audit')
    # Read the saved source-coverage universe within the requested six months.
    with (args.saved_run/'coverage.csv').open() as stream:
        # Preserve every requested stock-day, including zero-window observations.
        rows=[r for r in csv.DictReader(stream) if '2026-01-01'<=r['date']<='2026-06-30']
    # Reject a changed or duplicated universe rather than call it the agreed audit.
    if len(rows)!=13560 or len({r['symbol'] for r in rows})!=113 or len({(r['symbol'],r['date']) for r in rows})!=len(rows):
        # Require explicit investigation if the saved baseline differs.
        raise ValueError('Expected 13,560 distinct January–June stock-days over 113 names')
    # Freeze the reproducible pilot selection before observing corrected outcomes.
    pilot,full=select_jobs(rows)
    # Use the full population only with the separate explicit gate above.
    jobs=pilot if args.scope=='pilot' else full
    # Offer one known source cell for end-to-end command and output validation.
    if args.scope=='smoke':
        # This smoke scope cannot be mistaken for a representative timing pilot.
        jobs=[j for j in full if j['symbol']=='TELE' and j['date']=='2026-04-29']
    # Record exact source identities before workers import implementation code.
    sources=source_hashes()
    # Verify original reconstruction dependencies against the historical manifest.
    old_identity=read_json(args.saved_run/'identity.json')
    # Check the essential shared data and book modules before trusting cached baseline evidence.
    for name in ('clean_window_book.py','clean_window_data.py','run_legacy_mm.py','snapshot_prep.py','mm_backtest.py','psx_reference_rows.py','psx_reference_book.py'):
        # Require unchanged historical dependencies, not just matching file names.
        if old_identity['source'].get(str(LEGACY/name))!=sha(LEGACY/name):
            # Refuse an invalid comparison before any expensive source read.
            raise ValueError('Historical dependency changed: '+name)
    # Use the canonical current parsed-store root read-only.
    from config_pk import PARSED_ROOT
    # Reuse saved full-channel timing and sequence flags.
    quality=read_json(args.saved_run/'channel_quality.json')
    # Freeze all selected partitions, including snapshots, for run/resume consistency.
    metadata={day:partition_metadata(PARSED_ROOT,day) for day in sorted({j['date'] for j in jobs})}
    # Verify the saved channel screen still describes the same update/trade files.
    for day,files in metadata.items():
        # Keep snapshot metadata distinct from the older screen's two source tables.
        screened={p:info for p,info in files.items() if '/ob_snapshot/' not in p}
        # Refuse stale channel screening or changed partition membership.
        if screened!=quality[day]['metadata']:
            # A changed feed needs a new screen, outside this saved-evidence audit.
            raise ValueError('Saved channel metadata differs: '+day)
    # Tie resume behavior to scope, code, channel screen and source partitions.
    contract=dict(version=2,scope=args.scope,workers=args.workers,jobs=jobs,source_hashes=sources,metadata=metadata,baseline_cell_hashes={str(args.saved_run/(j['symbol']+'_'+j['date'])/'result.json'):sha(args.saved_run/(j['symbol']+'_'+j['date'])/'result.json') for j in jobs},baseline_identity_sha256=sha(args.saved_run/'identity.json'),coverage_sha256=sha(args.saved_run/'coverage.csv'),channel_quality_sha256=sha(args.saved_run/'channel_quality.json'),selection='12 names: cheap-tick exceptions plus three per source-window-fragmentation tertile; one observed-session-stratified date per month; no profit selection',strategy_replay=False,snapshot_policy='Replace market depth at every validated causal snapshot inside a window; coarse-time ambiguous snapshots remain explicitly rejected')
    # Avoid writing reports into the completed baseline experiment or any source directory.
    output=args.output_dir.resolve()
    # Enforce a separate results destination under the authorized Codex results root.
    if not output.is_relative_to(DEFAULT_RUN.parent) or output==DEFAULT_RUN.parent or output.is_relative_to(args.saved_run.resolve()):
        # Protect source code and immutable baseline artifacts from accidental output paths.
        raise ValueError('Use a new subfolder of Capital Stake - Results Codex outside the baseline run')
    # Handle deliberate resume only when the complete source contract matches.
    if output.exists():
        # Refuse existing destinations unless explicit resume is requested.
        if not args.resume or not (output/'contract.json').exists() or read_json(output/'contract.json')!=contract:
            # Never blend different versions into a plausible single audit.
            raise ValueError('Output exists or resume contract differs; choose a new output folder')
    # Create a fresh result directory when none exists.
    else:
        # Preserve the user's explicit location without touching prior runs.
        output.mkdir(parents=True)
    # Persist the complete selection and provenance before source processing.
    save(output/'contract.json',contract)
    # Write a convenient human-readable sample manifest.
    write_csv(output/'selected_jobs.csv',jobs)
    # Stop after metadata inspection when the user asks to review the plan.
    if args.plan_only:
        # Report exactly what would run without decoding source data.
        print(json.dumps(dict(scope=args.scope,jobs=len(jobs),symbols=len({j['symbol'] for j in jobs}),dates=len(metadata),output=str(output),source_data_loaded=False)))
        # Return successfully without worker processes or strategy execution.
        return
    # Keep completed cell records separate from summary outputs.
    cells=output/'cells'
    # Reuse the same directory for a verified interrupted audit.
    cells.mkdir(exist_ok=True)
    # Identify completed verified jobs and unfinished work.
    completed,pending_jobs=[],[]
    # Inspect every selected job without silently omitting failures.
    for job in jobs:
        # Match the worker identity convention.
        path=cells/(job['symbol']+'_'+job['date']+'.json')
        # Read resumable evidence only when it already exists.
        prior=read_json(path) if path.exists() else None
        # Reuse only successful records with exactly the same job identity.
        if prior is not None and prior.get('passed') and prior.get('job')==job:
            # Keep original measured per-cell timing for stratified runtime estimates.
            completed.append(prior)
        # Failed or unfinished cells remain explicit work items.
        else:
            # Reattempt only this unmatched stock-day.
            pending_jobs.append(job)
    # Record resumptions so wall-time extrapolation is not falsely based on skipped jobs.
    resumed=len(completed)
    # Measure this dispatch's actual elapsed wall time.
    began=last=time.monotonic()
    # Spawn workers with clean interpreter state.
    context=mp.get_context('spawn')
    # Keep per-worker labels small and avoid printing from every event.
    with context.Manager() as manager:
        # Expose source-processing phases to the parent heartbeat.
        progress=manager.dict()
        # Respect the declared worker cap without parallel strategy execution.
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,initializer=initialize,initargs=(str(args.saved_run),quality,metadata,progress)) as pool:
            # Submit the selected bounded source cells.
            pending={pool.submit(audit_cell,job):job for job in pending_jobs}
            # Receive results incrementally and persist each completed cell.
            while pending:
                # Wake frequently enough to maintain the requested heartbeat cadence.
                ready,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                # Persist every result before waiting for more work.
                for future in ready:
                    # Remove only the job whose worker has completed.
                    job=pending.pop(future)
                    # Let unexpected worker-process failures stop the command explicitly.
                    result=future.result()
                    # Save successes and failures atomically for later review.
                    save(cells/(job['symbol']+'_'+job['date']+'.json'),result)
                    # Preserve all requested outcomes in the parent summary.
                    completed.append(result)
                # Print one concise heartbeat every fifteen seconds.
                if time.monotonic()-last>=15:
                    # Count newly processed cells rather than resumed records for ETA.
                    done=len(completed)-resumed
                    # Measure elapsed wall time across this dispatch.
                    elapsed=time.monotonic()-began
                    # Estimate remaining duration only after at least one cell finishes.
                    eta=f'{elapsed/done*len(pending):.0f}s' if done else 'estimating'
                    # Show one active ticker/date and its current source-only phase.
                    active=next(iter(dict(progress).items()),('waiting','source-only'))
                    # Keep requested-job completion distinct from successful-cell count.
                    print(f'{datetime.now():%Y-%m-%d %H:%M:%S} {active[0]}/{active[1]} elapsed={elapsed:.0f}s remaining={eta} completed={len(completed)}/{len(jobs)} progress={100*len(completed)/len(jobs):.1f}%',flush=True)
                    # Restart the fifteen-second reporting interval.
                    last=time.monotonic()
    # Reject an audit that mixed changing code or changing source partitions.
    if source_hashes()!=sources or any(partition_metadata(PARSED_ROOT,day)!=files for day,files in metadata.items()) or any(sha(path)!=value for path,value in contract['baseline_cell_hashes'].items()):
        # Retain cell evidence but do not label the audit complete.
        raise ValueError('Source code or partitions changed during audit')
    # Separate unavailable comparisons rather than assigning failed cells zero impact.
    good=[r for r in completed if r['passed']]
    # Preserve failures with their reasons in a dedicated table.
    failures=[dict(**r['job'],error=r['error']) for r in completed if not r['passed']]
    # Select scalar fields suitable for direct CSV inspection.
    scalar=[dict(**r['job'],**{k:v for k,v in r.items() if k!='job' and not isinstance(v,(dict,list))}) for r in sorted(good,key=lambda r:(r['job']['date'],r['job']['symbol']))]
    # Add readable minute units without discarding exact millisecond evidence.
    for row in scalar:
        # Convert only durations, keeping event counts and runtime units separate.
        row.update({k.replace('_ms','_minutes'):row[k]/60000 for k in ('requested_ms','old_source_ms','new_source_ms','gained_ms','lost_ms','net_change_ms')})
    # Save the exact comparison and explicit failure tables.
    write_csv(output/'coverage_impact.csv',scalar)
    # Keep failure evidence separate from successful measurements.
    write_csv(output/'failures.csv',failures)
    # Aggregate rejection counters as counts, never as missing-time attribution.
    reason_rows=[]
    # Retain the original and corrected planner diagnostics separately.
    for version in ('old','new'):
        # Accumulate every recorded reason across successfully reproduced cells.
        reasons=defaultdict(int)
        # Count attempts, including short or unusable slices, with their original semantics.
        for result in good:
            # Preserve all observed reason categories rather than selected headline errors.
            for reason,count in result[version+'_counts'].items():
                # Sum the source planner's recorded attempt count.
                reasons[reason]+=count
        # Save explicit reason-count rows with no duration claim.
        reason_rows.extend(dict(version=version,reason=reason,attempt_count=count) for reason,count in sorted(reasons.items()))
    # Provide an independently reviewable table of remaining rejection diagnostics.
    write_csv(output/'reason_counts_NOT_DURATIONS.csv',reason_rows)
    # Estimate full source-audit compute from month-by-fragmentation workload strata.
    samples=defaultdict(list)
    # Preserve a distinct sample timing distribution for each population stratum.
    for result in good:
        # Use source-only load plus both planner times as the unit of work.
        samples[(result['job']['date'][:7],result['job']['stratum'])].append(result['elapsed_seconds'])
    # Count all requested jobs in each full-period workload stratum.
    population=defaultdict(int)
    # Include stock-days with no original usable windows in the projection.
    for job in full:
        # Match the sample's exact month/fragmentation classification.
        population[(job['date'][:7],job['stratum'])]+=1
    # Refuse a complete runtime projection if any stratum lacks successful observations.
    projected=sum(population[k]*statistics.mean(samples[k]) for k in population)/args.workers if all(samples[k] for k in population) and not failures else None
    # Preserve the sample support behind every runtime-estimation stratum.
    write_csv(output/'runtime_strata.csv',[dict(month=k[0],fragmentation_stratum=k[1],population_jobs=population[k],sample_jobs=len(samples[k]),mean_cell_seconds=statistics.mean(samples[k]) if samples[k] else None) for k in sorted(population)])
    # Aggregate exact measured duration changes and clearly scoped event counts.
    totals={k:sum(r[k] for r in good) for k in ('requested_ms','old_source_ms','new_source_ms','gained_ms','lost_ms','net_change_ms','missing_add_trades_retained','missing_add_cancels_retained','unpriced_cancel_attempts','anonymous_unique_events_retained')}
    # Preserve an explicit conservative next-step recommendation without arbitrary profit thresholds.
    recommendation='Investigate failed baseline/source cells before expansion.' if failures else 'Review timing and gross gained/lost intervals before a separate full source audit; coverage alone cannot justify a profit claim or automatic strategy replay.'
    # Keep pilot scope, timing limitations and successful counts in the final summary.
    summary=dict(scope=args.scope,selected_jobs=len(jobs),passed_jobs=len(good),failed_jobs=len(failures),resumed_jobs=resumed,workers=args.workers,dispatch_wall_seconds=time.monotonic()-began,full_requested_jobs=len(full),projected_full_source_wall_hours=projected/3600 if projected is not None else None,projection_method='Sum month-by-fragmentation population sizes times observed mean cell seconds; divide by requested workers. Diagnostic planning estimate, not a confidence bound; storage contention, memory pressure, startup and cache differences can change runtime.',totals=totals,strategy_replay=False,selection_warning='Purposive diagnostic diversity, not a probability sample. Do not extrapolate coverage gains or unresolved-event rates to all stocks.',counter_scope='Anonymous retained counts are unique source events inside retained corrected candidates, excluding short discarded attempts and failed timestamp batches. Unpriced-cancel counts are planner rejection attempts, not unavailable minutes.',next_step=recommendation)
    # Present duration totals in stock-minutes alongside the exact underlying units.
    summary['stock_minutes']={k.replace('_ms',''):value/60000 for k,value in totals.items() if k.endswith('_ms')}
    # Persist the completed summary before rendering any optional plotting output.
    save(output/'summary.json',summary)
    # Render useful evidence only when at least one comparison succeeded.
    if good:
        # Use a noninteractive backend so this command requires no GUI access.
        import matplotlib
        # Set the backend before importing the plotting interface.
        matplotlib.use('Agg')
        # Load only the available plotting dependency.
        import matplotlib.pyplot as plt
        # Compare gross gains, losses and net changes by calendar month.
        months=sorted({r['job']['date'][:7] for r in good})
        # Allocate one legible monthly impact panel and one runtime distribution.
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        # Plot distinct gain and loss components rather than hide offsetting shifts.
        for field,label,color in [('gained_ms','Gained','#2f7f6f'),('lost_ms','Lost','#c78244')]:
            # Convert exact milliseconds to stock-minutes for plotting only.
            values=[sum(r[field] for r in good if r['job']['date'].startswith(m))/60000 for m in months]
            # Offset the two component bars within each month.
            axes[0].bar([i+(-0.18 if field=='gained_ms' else 0.18) for i in range(len(months))],values,width=0.36,label=label,color=color)
        # Label the actual observed months.
        axes[0].set_xticks(range(len(months)),[m[5:] for m in months])
        # Explain the unit and the source-only nature of the result.
        axes[0].set_ylabel('Candidate stock-minutes')
        # Keep pilot interpretation visibly separate from full-period inference.
        axes[0].set_title(args.scope.capitalize()+' source coverage changes by month')
        # Identify both gross components.
        axes[0].legend()
        # Show measured per-cell runtime variation relevant to expansion cost.
        axes[1].hist([r['elapsed_seconds'] for r in good],bins=min(20,len(good)),color='#516d91')
        # Identify measured seconds per completed source cell.
        axes[1].set_xlabel('Load + original planner + corrected planner (seconds)')
        # Identify the count of successful cell comparisons.
        axes[1].set_ylabel('Stock-days')
        # Avoid labeling a runtime estimate as a guaranteed completion time.
        axes[1].set_title('Observed source-audit runtime distribution')
        # Save the visual alongside exact CSV and JSON evidence.
        fig.savefig(output/'coverage_and_runtime.png',dpi=160)
        # Release plotting memory before process exit.
        plt.close(fig)
    # Explain methodological limits in the generated report itself.
    (output/'READ_ME.md').write_text('# Source-only coverage audit\n\n'+json.dumps(summary,indent=2)+'\n\nNo trading strategy was replayed. Source windows are candidates; closing-success acceptance and profit have not been recomputed. Existing historical selection and profit reports remain unchanged. Snapshot metadata is frozen at audit start; update/trade metadata must also match the saved channel screen. Exact baseline window and rejection-count reproduction is required for every successful cell.\n')
    # Print a compact final summary suitable for sharing back for interpretation.
    print(json.dumps(summary,indent=2),flush=True)
    # Return a failure exit status when any selected comparison failed.
    if failures:
        # Do not let incomplete validation look like a successful complete audit.
        raise SystemExit(1)
# Keep worker imports from recursively launching another pool.
if __name__=='__main__':
    # Dispatch only the explicitly requested command-line scope.
    main()
