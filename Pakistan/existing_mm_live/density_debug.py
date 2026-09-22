"""Bounded first-mismatch diagnostic; never changes book or trading source."""
# Parse one failing instrument/day and an external evidence directory.
import argparse
# Serialize captured primitive book state.
import json
# Resolve external evidence paths.
from pathlib import Path
# Measure replay limits independently of clock adjustments.
import time
# Retain only a small rolling event history.
from collections import deque
# Read canonical parsed data with the existing loader.
import run_legacy_mm as R
# Intercept only this process's research collector book constructor.
import density_collector as collector
# Reuse the installed global fifteen-second heartbeat and source hash helper.
from density_research import Reporter, digest
# Preserve the original book implementation unchanged.
from mm_backtest import Book, Order
# Signal an intentional diagnostic stop distinctly from unexpected failures.
class DiagnosticStop(Exception):
    # Carry the stop reason in the normal exception message.
    pass
# Run a bounded diagnostic only when invoked directly.
def main():
    # Require explicit external output and expose conservative replay bounds.
    parser = argparse.ArgumentParser(description=__doc__)
    # Default to an observed failing pilot cell.
    parser.add_argument('--symbol',default='PPL')
    # Preserve the failing session date exactly.
    parser.add_argument('--date',default='2026-06-01')
    # Bound processed events after loading the selected session.
    parser.add_argument('--max-events',type=int,default=100000)
    # Bound replay wall time; input loading is separately reported.
    parser.add_argument('--max-seconds',type=float,default=120)
    # Keep all diagnostic evidence outside the checkout.
    parser.add_argument('--output-dir',type=Path,required=True)
    # Parse the user's single terminal command.
    args=parser.parse_args()
    # Reject invalid limits before reading market data.
    if args.max_events<1 or args.max_seconds<=0:
        # Report an actionable argument error.
        parser.error('positive limits required')
    # Preserve old diagnostics by requiring a new directory.
    args.output_dir.mkdir(parents=True,exist_ok=False)
    # Start the single parent heartbeat, including input loading.
    reporter=Reporter()
    # Retain the latest primitive events without a full event dump.
    recent=deque(maxlen=20)
    # Collect a truthful outcome even when the mismatch does not reproduce.
    evidence=dict(symbol=args.symbol,date=args.date,status='loading')
    # Save evidence and stop the heartbeat on all exit paths.
    try:
        # Announce the selected input cell.
        reporter.tick({'debug':dict(symbol=args.symbol,date=args.date,stage='Loading',done=0,total=0)},0,1)
        # Open the canonical existing parsed partitions read-only.
        datasets=R.open_datasets(args.date)
        # Reject unavailable whole-date partitions.
        if datasets is None:
            # A missing dataset is not a successful reproduction.
            raise ValueError('missing date partitions')
        # Read the single selected instrument's update stream.
        updates=R.read_symbol(datasets['ob_updates'],R.REQ_UPDATES,args.symbol,market='REG')
        # Read the selected instrument's snapshots.
        snapshots=R.read_symbol(datasets['ob_snapshot'],R.REQ_SNAP,args.symbol,market='REG')
        # Read the selected instrument's actual trades.
        trades=R.read_symbol(datasets['trades'],R.REQ_TRADES,args.symbol,market='REG')
        # Use the exact installed snapshot/event preparation path.
        events,groups,_=R.build_events(updates,snapshots,trades)
        # Reproduce the pilot's capture-clock ordering.
        events=[(int(event[4].ts_cap),*event[1:]) for event in events]
        # Preserve the pilot's same-millisecond tie ordering.
        events.sort(key=lambda event:(event[0],event[1],event[2]))
        # Bound replay independently of the initial read and preparation time.
        started=time.monotonic()
        # Record the current processed-event count at each timestamp boundary.
        processed=[0]
        # Add diagnostics without altering book transitions or returned depth.
        class DebugBook(Book):
            # Capture only the exact depth access that the collector requests.
            def ranked_depth(self,n=10,include_deep=False):
                # Delegate geometry reconstruction to the unchanged implementation.
                bids,asks=super().ranked_depth(n,include_deep)
                # Obtain the competing historical touch definition.
                bid,bq,ask,aq=self.bbo()
                # Reproduce the collector's current midpoint disagreement condition.
                mismatch=not bids or not asks or abs((bids[0][0]+asks[0][0])/2-(bid+ask)/2)>1e-7
                # Preserve a complete primitive order state at the first disagreement.
                if mismatch:
                    # Label this as observed evidence, not an assumed root cause.
                    evidence.update(status='mismatch_reproduced',events_processed=processed[0],historical_bbo=[bid,bq,ask,aq],known_bids=bids,known_asks=asks,phase=self.phase,recent_events=list(recent),orders=[dict(key=str(key),side=value.side,price=value.price,qty=value.qty) for key,value in self.o.items()])
                    # Stop before collecting or publishing any feature data.
                    raise DiagnosticStop('first density/touch mismatch captured')
                # Return the original result for all preceding valid samples.
                return bids,asks
        # Override only the collector module in this standalone diagnostic process.
        collector.Book=DebugBook
        # Stream event history while preserving the collector's sized-list contract.
        class TracedEvents(list):
            # Capture payloads as the collector consumes them.
            def __iter__(self):
                # Keep original selected-clock event order.
                for event in super().__iter__():
                    # Include both source payload and the chosen replay timestamp.
                    recent.append(dict(ts=event[0],kind=event[3],payload=event[4]._asdict()))
                    # Yield the exact original event without modification.
                    yield event
        # Publish concise replay progress and enforce bounded execution.
        def progress(done,total):
            # Record the last fully applied timestamp group's event count.
            processed[0]=done
            # Maintain the standard fifteen-second parent heartbeat.
            reporter.tick({'debug':dict(symbol=args.symbol,date=args.date,stage='Debug replay',done=done,total=min(total,args.max_events))},0,1)
            # Stop at the next timestamp boundary after either diagnostic limit.
            if done>=args.max_events or time.monotonic()-started>=args.max_seconds:
                # Distinguish a bounded non-reproduction from a successful fix.
                raise DiagnosticStop('replay limit reached without reproducing mismatch')
        # Run the same collector parameters as the pilot without writing feature parquet.
        collector.collect(TracedEvents(events),groups,.01,1000,2000,[250],progress)
        # A completed session is a non-reproduction, not automatic validation.
        evidence['status']='session_completed_without_mismatch'
    # Expected stops preserve the detailed first-failure state.
    except DiagnosticStop as error:
        # Keep the first-mismatch status when already recorded.
        evidence.setdefault('reason',str(error))
        # A replay cap without a mismatch must remain explicitly inconclusive.
        if evidence['status']=='loading':
            # Never report a cap as a pass.
            evidence['status']='bounded_without_mismatch'
    # Unexpected loader or collector errors need exact traceback evidence.
    except Exception:
        # Import only on the diagnostic error path.
        import traceback
        # Preserve the full failure without misclassifying it as a reproduced mismatch.
        evidence.update(status='error',traceback=traceback.format_exc())
    # Always persist diagnostic output and shut down the timer.
    finally:
        # Record hashes for the modules involved in the competing definitions.
        evidence['source_hashes']={str(path):digest(path) for path in (Path(__file__),Path(collector.__file__),Path(R.__file__),Path(__file__).parent/'mm_backtest.py')}
        # Store evidence outside source with a stable filename.
        (args.output_dir/'diagnostic.json').write_text(json.dumps(evidence,indent=2,default=str)+'\n')
        # Stop periodic output before the final summary.
        reporter.close()
        # Give the user one concise outcome and its evidence path.
        print(evidence['status']+' | '+str(args.output_dir/'diagnostic.json'),flush=True)
# Avoid running the diagnostic merely because a test imports it.
if __name__=='__main__':
    # Invoke the bounded command-line entry point.
    main()
