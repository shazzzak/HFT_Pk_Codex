# Keep per-window candidate objects independent.
import copy
# Write durable compressed fill evidence without retaining full-day arrays.
import gzip
# Serialize explicit matched/excluded fill rows.
import csv
# Measure progress without printing per event.
import time
# Resolve per-cell evidence locations.
from pathlib import Path
# Reconstruct source-only usable intervals before any strategy is run.
from clean_window_data import load_symbol, plan_windows
# Use explicit delayed cancellations and costed window exits.
from clean_window_engine import build_engine, WindowExitError
# Keep all twelve settings and the declared cheap-name aliases.
from stock_search_engine import ARMS, CHEAP, make_strategy
# Retain the original additive accounting fields and bucket order.
from stock_search_accounting import MONEY, BUCKETS
# Save atomic compact evidence and checksum raw fills.
from stock_search_util import save, digest

# Install only a lightweight parent-owned progress proxy in worker processes.
def initialize(progress):
    # Retain progress for this worker, not an external shared file.
    global PROGRESS
    # Assign the manager proxy created by the runner.
    PROGRESS=progress

# Process one requested stock-day, sharing only source data between candidate settings.
def cell(task):
    # Unpack a frozen job, channel-quality screen and evidence directory.
    job,quality,output=task
    # Start elapsed-time measurement for this complete cell.
    began=time.monotonic()
    # Preserve deterministic checkpoint names.
    key=job['symbol']+'_'+job['date']
    # Place every artifact outside the repository.
    folder=Path(output)/key
    # Permit a failed cell to be rerun in its existing verified run directory.
    folder.mkdir(exist_ok=True)
    # Keep input failures explicit rather than generating a zero-profit row.
    try:
        # Report the actual loading phase.
        PROGRESS[key]=('loading',0.0)
        # Reuse one decoded source set across all twelve arms.
        events,checkpoints,adds=load_symbol(job['date'],job['symbol'])
        # Build common windows without looking at any strategy's profit.
        windows,reasons=plan_windows(events,checkpoints,adds,quality,job['params']['session_segments'])
        # Create fixed additive summaries for accepted common windows only.
        totals={arm:dict(net_pkr=0.0,fills=0,capacity_reductions=0,max_inventory=0.0,max_window_drawdown_pkr=0.0,signal=dict(calls=0,valid=0,fallback=0),buckets={b:dict(bucket=b,**{m:0.0 for m in MONEY}) for b in BUCKETS}) for arm in ARMS}
        # Retain both matched and unclosed-window evidence.
        details=[]
        # Accumulate the actual shared coverage, not the requested full-day duration.
        accepted_ms=0
        # Save raw fills from excluded windows too, clearly marked as excluded.
        with gzip.open(folder/'fills.csv.gz','wt',newline='') as stream:
            # Keep a stable schema while allowing additional original execution labels.
            writer=csv.DictWriter(stream,fieldnames=['arm','window_id','accepted','t','side','px','qty','reason','mid0','mid_source','pre_bid','pre_ask','oid'],extrasaction='ignore')
            # Write headers even for cells with no usable windows.
            writer.writeheader()
            # Evaluate every independently flat-start window in source order.
            for index,window in enumerate(windows):
                # Retain only this interval's fills and results in memory.
                results,records,errors={},{},{}
                # Keep IPC messages infrequent; only the parent prints heartbeats.
                last=[0.0]
                # Run all declared styles and depths on this exact interval.
                for arm_index,arm in enumerate(ARMS):
                    # Cheap-name styles share a replay only at the identical signal depth.
                    canonical='OBI_'+arm.split('_')[1] if job['symbol'] in CHEAP else arm
                    # Reuse proven identical style settings without hiding depth comparisons.
                    if canonical in results:
                        # Retain the exact alias in the diagnostic result.
                        results[arm]=copy.deepcopy(results[canonical])
                        # Save equivalent executions under the logical arm's name too.
                        records[arm]=records[canonical]
                        # Skip only this exact duplicate replay.
                        continue
                    # Construct a fresh flat account for this one independent experiment.
                    engine,effective,counts=build_engine(job,arm,window,adds)
                    # Expose progress across windows and twelve settings.
                    def progress(fraction):
                        # Avoid manager overhead on almost all source events.
                        if time.monotonic()-last[0]>=1:
                            # Report the actual candidate and source progress.
                            PROGRESS[key]=(arm,(index+(arm_index+fraction)/len(ARMS))/max(1,len(windows)))
                            # Restart the one-second IPC throttle.
                            last[0]=time.monotonic()
                    # Keep exit failures separate from programming/data-contract failures.
                    try:
                        # This path cannot use the legacy instant end-of-day order deletion.
                        results[arm]=engine.replay(events,window,progress)
                        # Retain weighted-signal availability evidence.
                        results[arm]['signal']=dict(counts)
                    # A failed delayed exit makes the entire matched window unavailable.
                    except WindowExitError as error:
                        # Save exposure rather than inventing an execution or cash adjustment.
                        errors[arm]=dict(reason=str(error),position=engine.pos,cash=engine.cash,orders=len(engine._all_orders()),pending=len(engine.pending))
                    # Always retain actual simulated fills even if the closing check failed.
                    records[arm]=list(engine.fills)
                    # Break callbacks retaining this engine after this window.
                    del engine.strat.capacity_guard
                    # Release optional weighted callbacks too.
                    if hasattr(engine.strat,'search_signal'):
                        # Keep long-run memory bounded across all candidate replays.
                        del engine.strat.search_signal
                # The comparison uses the same set of successfully closed windows in every arm.
                accepted=not errors and set(results)==set(ARMS)
                # Persist source boundaries and all reasons for a matched-window exclusion.
                details.append(dict(window_id=index,start=window['start'],end=window['end'],ending=window['ending'],accepted=accepted,exit_failures=errors))
                # Preserve excluded executions, visibly distinct from accepted P&L.
                for arm,fills in records.items():
                    # Keep original fill order for later independent FIFO reconciliation.
                    for fill in fills:
                        # Add experiment identity without altering the source execution record.
                        writer.writerow(dict(fill,arm=arm,window_id=index,accepted=accepted))
                # Never count unavailable windows as zero-return matched observations.
                if not accepted:
                    # Continue other independently selected source intervals.
                    continue
                # Measure only matched usable duration.
                accepted_ms+=window['end']-window['start']
                # Sum additive cash and bucket components for accepted intervals.
                for arm,result in results.items():
                    # Update this arm's compact aggregate.
                    total=totals[arm]
                    # Accumulate only quantities that are additive across independent accounts.
                    for field in ('net_pkr','fills','capacity_reductions'):
                        # Preserve actual simulated totals.
                        total[field]+=result[field]
                    # Preserve the largest held position across independent intervals.
                    total['max_inventory']=max(total['max_inventory'],result['max_inventory'])
                    # Avoid calling a per-window risk measure a full-day drawdown.
                    total['max_window_drawdown_pkr']=max(total['max_window_drawdown_pkr'],result['drawdown_pkr'])
                    # Preserve weighted availability counters across the accepted comparison.
                    for field,value in result['signal'].items():
                        # Count decision calls, not calendar observations.
                        total['signal'][field]+=value
                    # Retain original opening-time buckets regardless of window position.
                    for row in result['buckets']:
                        # Reconcile each additive monetary component.
                        for field in MONEY:
                            # Sum only the matching opening cohort.
                            total['buckets'][row['bucket']][field]+=row[field]
        # Record all settings even when no window qualifies for P&L.
        effective={arm:make_strategy(job,arm,(job['params']['session_segments'][0][0],job['params']['session_segments'][-1][1]))[1] for arm in ARMS}
        # Make lack of coverage explicit with null arm totals.
        row=dict(passed=True,job=job,elapsed_seconds=time.monotonic()-began,requested_ms=sum(b-a for a,b in job['params']['session_segments']),source_window_ms=sum(w['end']-w['start'] for w in windows),accepted_ms=accepted_ms,windows=details,source_exclusions=reasons,arms=totals if accepted_ms else None,effective_params=effective,fill_sha256=digest(folder/'fills.csv.gz'))
    # A programming or malformed-data contract error does not disappear into exclusions.
    except Exception as error:
        # Preserve the requested cell and stop it with an actionable diagnostic.
        row=dict(passed=False,job=job,error=repr(error),elapsed_seconds=time.monotonic()-began)
    # Persist the complete cell atomically for restart/resume.
    save(folder/'result.json',row)
    # Publish completion without a burst of per-arm terminal output.
    PROGRESS[key]=('done',1.0)
    # Return a compact parent result; detailed evidence stays on disk.
    return dict(key=key,passed=row['passed'],error=row.get('error'))
