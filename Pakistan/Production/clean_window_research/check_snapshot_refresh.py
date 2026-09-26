# Validate bounded source coverage and one multi-arm replay without a full-history run.
import json,sys,importlib.util,time,threading,hashlib
# Locate immutable baselines and new validation artifacts.
from pathlib import Path
# Configure local corrected imports before read-only legacy dependencies.
HERE=Path(__file__).resolve().parent
# Make the corrected modules authoritative.
sys.path.insert(0,str(HERE))
# Reuse the established strategy dependencies without editing them.
sys.path.append(str(HERE.parents[1]/'existing_mm_live'))
# Load current source reconstruction and interval arithmetic.
import clean_window_data as current
# Reuse the independently tested overlap calculation.
from audit_source_coverage import intersection_ms,signature
# Exercise actual replay integration on one bounded interval.
from clean_window_engine import build_engine,WindowExitError
# Preserve all existing strategy settings.
from stock_search_engine import ARMS
# Render source-only comparisons rather than profit charts.
import matplotlib.pyplot as plt
# Locate the authorized results root.
ROOT=Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex')
# Write only to the new refresh validation directory.
OUT=ROOT/'snapshot_refresh_validation_20260925_v1'
# Preserve the completed full historical run as read-only input.
SAVED=ROOT/'clean_window_full_20260925_v1'
# Load explicitly frozen pre-refresh modules in separate namespaces.
def load(name,path):
    # Resolve this exact source file without importing its CLI.
    spec=importlib.util.spec_from_file_location(name,path)
    # Allocate the isolated module.
    module=importlib.util.module_from_spec(spec)
    # Execute definitions only.
    spec.loader.exec_module(module)
    # Return the frozen implementation.
    return module
# Load the archived pre-refresh book.
prior_book=load('prior_refresh_book',OUT/'prior_sources/clean_window_book.py')
# Load the archived pre-refresh planner.
prior=load('prior_refresh_data',OUT/'prior_sources/clean_window_data.py')
# Bind that planner to its own unchanged book.
prior.WindowBook=prior_book.WindowBook
# Bind its matching data exception too.
prior.WindowDataError=prior_book.WindowDataError
# Read immutable channel-quality evidence.
quality=json.loads((SAVED/'channel_quality.json').read_text())
# Select two large loss cases and two prior regression cells.
sample=[('KEL','2026-05-08'),('BOP','2026-05-20'),('TELE','2026-04-29'),('AGHA','2026-01-01')]
# Accumulate exact comparisons.
rows=[]
# Keep the eventual replay outcome separate from coverage.
replay={}
# Expose bounded progress during source loading and replay.
state=dict(label='starting',done=0)
# Establish elapsed wall time.
began=time.monotonic()
# Stop the reporter even when a verification fails.
stop=threading.Event()
# Report status no more than once per fifteen seconds.
def heartbeat():
    # Wait interruptibly for each reporting interval.
    while not stop.wait(15):
        # Identify the real current phase and bounded sample progress.
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {state["label"]} elapsed={time.monotonic()-began:.0f}s completed={state["done"]}/4',flush=True)
# Start a lightweight progress reporter.
threading.Thread(target=heartbeat,daemon=True).start()
# Ensure reporter cleanup after all validation paths.
try:
    # Reconstruct the predeclared four stock-days without strategy selection by profit.
    for symbol,date in sample:
        # Publish the active source cell.
        state['label']=symbol+'/'+date+'/source'
        # Read its saved calibrated session.
        saved=json.loads((SAVED/(symbol+'_'+date)/'result.json').read_text())
        # Share identical decoded data across both planners.
        events,checkpoints,adds=current.load_symbol(date,symbol)
        # Reproduce the anonymous-reference version before snapshot replacement.
        old,old_counts=prior.plan_windows(events,checkpoints,adds,quality[date],saved['job']['params']['session_segments'])
        # Match exact prior evidence for the two pilot loss cells.
        if symbol in ('KEL','BOP'):
            # Read the user's completed pilot result.
            evidence=json.loads((ROOT/'source_coverage_pilot_20260925_194110/cells'/(symbol+'_'+date+'.json')).read_text())
            # Require exact pre-refresh boundaries and rejection counters.
            assert signature(old)==evidence['new_intervals'] and old_counts==evidence['new_counts']
        # Match exact earlier three-cell evidence for the remaining two cells.
        else:
            # Read the frozen first correction's diagnostic output.
            evidence=json.loads((ROOT/'Anonymous_Reference_Fix_20260925_v1'/(symbol+'_'+date+'.json')).read_text())
            # Require exact reproduction rather than comparing only rounded minutes.
            assert signature(old)==[[w['start'],w['end'],w['ending']] for w in evidence['new_windows']] and old_counts==evidence['new_counts']
        # Reconstruct with periodic market-picture replacement.
        new,new_counts=current.plan_windows(events,checkpoints,adds,quality[date],saved['job']['params']['session_segments'])
        # Preserve exact disjoint interval pairs.
        left,right=[(w['start'],w['end']) for w in old],[(w['start'],w['end']) for w in new]
        # Measure independently shared time.
        shared=intersection_ms(left,right)
        # Keep source-only coverage and replacement counts separate.
        row=dict(symbol=symbol,date=date,prior_ms=sum(b-a for a,b in left),refresh_ms=sum(b-a for a,b in right),shared_ms=shared,refreshes_retained=sum(len(w['refreshes']) for w in new),counts=new_counts,old_intervals=signature(old),new_intervals=signature(new))
        # Retain the exact net coverage difference.
        row['net_ms']=row['refresh_ms']-row['prior_ms']
        # Save complete cell evidence immediately after source validation.
        (OUT/(symbol+'_'+date+'.json')).write_text(json.dumps(row,indent=2))
        # Accumulate the compact reporting rows.
        rows.append(row)
        # Exercise strategy integration only once on a bounded TELE interval with real refreshes.
        if symbol=='TELE':
            # Choose by duration and refresh presence, never by profit.
            window=min((w for w in new if w['refreshes'] and w['right']>w['left']),key=lambda w:w['end']-w['start'])
            # Preserve the exact source-selected replay interval.
            replay=dict(symbol=symbol,date=date,start=window['start'],end=window['end'],expected_refreshes=len(window['refreshes']),arms={})
            # Exercise every unchanged strategy adapter on the same interval.
            for arm in ARMS:
                # Publish the active bounded integration phase.
                state['label']='TELE/'+date+'/'+arm
                # Retain original settings, seed and order-state machinery.
                engine,effective,counts=build_engine(saved['job'],arm,window,adds)
                # Keep expected closing exclusions distinct from program failures.
                try:
                    # Replay actual incrementals and the planner's exact snapshot schedule.
                    result=engine.replay(events,window)
                    # Require complete snapshot delivery in every successful replay.
                    assert engine.snapshot_refreshes==len(window['refreshes'])
                    # Preserve actual successful integration evidence without publishing profits.
                    replay['arms'][arm]=dict(closed=True,refreshes=engine.snapshot_refreshes,fills=len(engine.fills))
                # Explicit uncertainty or closing failure remains a visible excluded outcome.
                except WindowExitError as error:
                    # Retain exposure and the exact reason without manufacturing a close.
                    replay['arms'][arm]=dict(closed=False,reason=str(error),refreshes=engine.snapshot_refreshes,position=engine.pos)
            # Persist bounded replay results separately from source coverage.
            (OUT/'single_window_replay.json').write_text(json.dumps(replay,indent=2))
        # Mark one complete source comparison.
        state['done']+=1
        # Print only scalar source metrics.
        print(json.dumps({k:v for k,v in row.items() if not isinstance(v,(list,dict))}),flush=True)
# Always terminate the daemon heartbeat.
finally:
    # Wake the reporter immediately.
    stop.set()
# Compare matched stock-days without a profit implication.
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
# Display previous source coverage to the left of each label.
ax.bar([i-.18 for i in range(len(rows))],[r['prior_ms']/60000 for r in rows],width=.36,label='Before snapshot refresh',color='#999999')
# Display the new refreshed coverage to the right.
ax.bar([i+.18 for i in range(len(rows))],[r['refresh_ms']/60000 for r in rows],width=.36,label='With validated snapshot refresh',color='#2f7f6f')
# Identify exact sample dates.
ax.set_xticks(range(len(rows)),[r['symbol']+'\n'+r['date'] for r in rows])
# Keep the metric clearly source-only.
ax.set_ylabel('Candidate source-window minutes')
# State the limited validation scope.
ax.set_title('Four-stock-day validation; no full strategy or profit rerun')
# Label both implementations explicitly.
ax.legend()
# Save the requested visual evidence.
fig.savefig(OUT/'coverage.png',dpi=150)
# Preserve source fingerprints for this exact validation.
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
# Save the final scope and exact results.
(OUT/'summary.json').write_text(json.dumps(dict(cells=rows,replay=replay,source_hashes=hashes,elapsed_seconds=time.monotonic()-began,full_history_run=False),indent=2))
