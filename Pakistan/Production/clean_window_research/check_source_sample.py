# Compare source-only window planning without executing strategy arms.
import json, time, importlib.util, hashlib, csv
# Locate explicitly authorized evidence and read-only dependencies.
from pathlib import Path
# Use a worker thread solely for a fifteen-second progress heartbeat.
import threading
# Render a simple coverage comparison from calculated sample results.
import matplotlib.pyplot as plt
# Import the corrected Production planner.
import clean_window_data as corrected
# Inspect exact anonymous-pool availability when a reduction remains rejected.
from clean_window_book import WindowBook, WindowDataError
# Collect bounded explanatory examples without changing book behavior.
examples=[]
# Extend the corrected book only with read-only failure diagnostics.
class DiagnosticBook(WindowBook):
    # Preserve resolution behavior and record the actual quantity available on failure.
    def resolve(self,row):
        # Delegate every successful operation to the corrected implementation.
        try:
            # Return the ordinary pool or named-order resolution.
            return super().resolve(row)
        # Add context only when the actual book rejects the event.
        except WindowDataError as error:
            # Keep a small diagnostic sample rather than full market-data dumps.
            if len(examples)<30:
                # Read the event's resting side directly from its one-sided reference.
                side='BUY' if getattr(row,'buy_ref',0) else 'SELL'
                # Read the event's price without inventing a cancellation price.
                price=getattr(row,'price',None)
                # Collect actual retained orders at the reported execution price.
                levels=[dict(oid=k,qty=o.qty) for k,o in self.o.items() if o.side==side and o.price==price]
                # Record the current sample identity and actual remaining level quantities.
                examples.append(dict(cell=state['label'],error=str(error),event=getattr(row,'event','TRADE'),sequence=row.appl_seq,price=price,qty=row.qty,side=side,orders_at_price=levels))
            # Preserve the original rejection and planner control flow.
            raise
# Use the diagnostic subclass only in this source-sample script.
corrected.WindowBook=DiagnosticBook
# Resolve the read-only original implementation.
legacy=Path(__file__).resolve().parents[2]/'existing_mm_live'
# Read the completed historical experiment without altering it.
saved=Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/clean_window_full_20260925_v1')
# Keep sample outputs separate from historical results.
out=saved.parent/'Anonymous_Reference_Fix_20260925_v1'
# Create only the new report directory.
out.mkdir(exist_ok=True)
# Load old modules under distinct names for an exact baseline comparison.
def load(name,path):
    # Build a module specification from the explicit read-only source.
    spec=importlib.util.spec_from_file_location(name,path)
    # Allocate an isolated namespace.
    module=importlib.util.module_from_spec(spec)
    # Execute definitions without invoking a strategy replay.
    spec.loader.exec_module(module)
    # Return the independently loaded module.
    return module
# Load the historical book implementation.
old_book=load('baseline_book',legacy/'clean_window_book.py')
# Load the historical planner implementation.
baseline=load('baseline_data',legacy/'clean_window_data.py')
# Bind only the isolated baseline planner to its original book.
baseline.WindowBook=old_book.WindowBook
# Bind its matching exception class for exact historical rejection behavior.
baseline.WindowDataError=old_book.WindowDataError
# Read saved run source identities.
identity=json.loads((saved/'identity.json').read_text())
# Refuse comparison against drifted original source definitions.
for name in ['clean_window_book.py','clean_window_data.py']:
    # Require the current baseline to match its recorded hash.
    assert hashlib.sha256((legacy/name).read_bytes()).hexdigest()==identity['source'][str(legacy/name)]
# Reuse the saved channel screen without scanning whole-market feed again.
quality=json.loads((saved/'channel_quality.json').read_text())
# Predeclare three diagnostic cells without selecting them by profit improvement.
sample=[('AGHA','2026-01-01'),('AICL','2026-04-06'),('TELE','2026-04-29')]
# Store compact report rows.
rows=[]
# Track the active cell and completed jobs for the heartbeat.
state=dict(label='starting',done=0)
# Establish a wall-clock origin.
began=time.monotonic()
# Coordinate heartbeat termination without blocking completion.
stop=threading.Event()
# Report progress independently of data loading and book reconstruction.
def heartbeat():
    # Wait no more than fifteen seconds between concise status lines.
    while not stop.wait(15):
        # Derive elapsed time and measured job-based remaining estimate.
        elapsed=time.monotonic()-began
        # Avoid claiming an estimate before any cell completes.
        remaining=f'{elapsed*(len(sample)/state["done"]-1):.0f}s' if state['done'] else 'estimating'
        # Report the source-only nature and exact completion count.
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {state["label"]}/source-only elapsed={elapsed:.0f}s remaining={remaining} jobs={state["done"]}/{len(sample)} progress={100*state["done"]/len(sample):.0f}%',flush=True)
# Start a daemon reporter that cannot keep a failed job alive.
threading.Thread(target=heartbeat,daemon=True).start()
# Ensure the heartbeat stops even when an assertion fails.
try:
    # Process each bounded diagnostic cell serially.
    for symbol,date in sample:
        # Identify the current source-only comparison.
        state['label']=symbol+'/'+date
        # Read its exact saved session and source-window evidence.
        record=json.loads((saved/(symbol+'_'+date)/'result.json').read_text())
        # Load one symbol-day once for both planners.
        events,checkpoints,adds=corrected.load_symbol(date,symbol)
        # Reconstruct historical candidate windows with the unchanged old book.
        old,old_counts=baseline.plan_windows(events,checkpoints,adds,quality[date],record['job']['params']['session_segments'])
        # Reproduce every historical candidate's boundaries and termination label.
        assert [(w['start'],w['end'],w['ending']) for w in old]==[(w['start'],w['end'],w['ending']) for w in record['windows']]
        # Verify diagnostic baseline counters independently too.
        assert old_counts==record['source_exclusions']
        # Plan with the corrected anonymous reduction behavior on identical inputs.
        new,new_counts=corrected.plan_windows(events,checkpoints,adds,quality[date],record['job']['params']['session_segments'])
        # Preserve complete candidate-level evidence for review.
        (out/(symbol+'_'+date+'.json')).write_text(json.dumps(dict(symbol=symbol,date=date,baseline_reproduced=True,old_counts=old_counts,new_counts=new_counts,old_windows=[{k:w[k] for k in ('start','end','ending')} for w in old],new_windows=[{k:w[k] for k in ('start','end','ending')} for w in new]),indent=2))
        # Record source coverage only, with no claim about accepted closing windows.
        rows.append(dict(symbol=symbol,date=date,old_source_minutes=sum(w['end']-w['start'] for w in old)/60000,new_source_minutes=sum(w['end']-w['start'] for w in new)/60000,old_windows=len(old),new_windows=len(new),old_unresolved=old_counts.get('UNRESOLVED_SOURCE_REFERENCE',0),new_unresolved=new_counts.get('UNRESOLVED_SOURCE_REFERENCE',0),unpriced_cancels=new_counts.get('UNPRICED_ANONYMOUS_CANCEL',0),excess_reductions=new_counts.get('UNRESOLVED_OR_EXCESS_REDUCTION',0)))
        # Mark this cell complete only after baseline reproduction and corrected planning.
        state['done']+=1
        # Print one result per completed bounded cell.
        print(json.dumps(rows[-1]),flush=True)
# Always stop the reporting thread after the bounded investigation.
finally:
    # Release the heartbeat promptly.
    stop.set()
# Save exact sample metrics as a reviewable table.
with (out/'sample_comparison.csv').open('w',newline='') as stream:
    # Use the explicit compact metric schema.
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    # Write readable column names.
    writer.writeheader()
    # Preserve numeric values without presentation rounding.
    writer.writerows(rows)
# Create a labeled comparison figure with no profit implication.
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
# Draw original source-window minutes to the left of each sample label.
ax.bar([i-0.18 for i in range(len(rows))],[r['old_source_minutes'] for r in rows],width=0.36,label='Original',color='#92989e')
# Draw corrected source-window minutes to the right of each sample label.
ax.bar([i+0.18 for i in range(len(rows))],[r['new_source_minutes'] for r in rows],width=0.36,label='Corrected anonymous reductions',color='#27796e')
# Identify exact tickers and dates rather than implying a universe-wide result.
ax.set_xticks(range(len(rows)),[r['symbol']+'\n'+r['date'] for r in rows])
# State the source-only metric explicitly.
ax.set_ylabel('Candidate source-window minutes')
# Avoid confusing these results with a strategy or closing replay.
ax.set_title('Three-cell source-only check; closing outcomes not rerun')
# Label both compared implementations.
ax.legend()
# Save the visual evidence beside its underlying table.
fig.savefig(out/'sample_coverage.png',dpi=150)
# Save final scope and corrected implementation identities.
(out/'summary.json').write_text(json.dumps(dict(sample=rows,remaining_failure_examples=examples,elapsed_seconds=time.monotonic()-began,strategy_replay=False,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}),indent=2))
