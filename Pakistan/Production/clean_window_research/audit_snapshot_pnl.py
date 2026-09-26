# Measure a fixed bounded P&L comparison without changing strategy parameters.
import argparse,json,sys,time,hashlib,importlib.util,threading
# Resolve explicit project and result paths.
from pathlib import Path
# Run independent versions in isolated processes without a manager socket.
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
# Use spawn to isolate engine module globals between comparisons.
import multiprocessing as mp
# Make corrected modules authoritative for ordinary dependencies.
HERE=Path(__file__).resolve().parent
# Add the research module directory.
sys.path.insert(0,str(HERE))
# Reuse legacy strategy modules read-only.
LEGACY=HERE.parents[1]/'existing_mm_live'
# Keep dependencies behind corrected local files.
sys.path.append(str(LEGACY))
# Disable bytecode writes even if the caller omitted its environment flag.
sys.dont_write_bytecode=True
# Preserve the original full experiment as read-only evidence.
ROOT=Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex')
# Read all fixed calibration and seed settings from this baseline.
SAVED=ROOT/'clean_window_full_20260925_v1'
# Use exact archived versions from before snapshot replacement.
PRIOR=ROOT/'snapshot_refresh_validation_20260925_v1/prior_sources'
# Fix diagnostic stock-days before observing P&L changes.
SAMPLE=[('KEL','2026-05-08'),('BOP','2026-05-20'),('TELE','2026-04-29'),('AGHA','2026-01-01')]

# Load one isolated implementation without running its CLI.
def module(name,path):
    # Build a specification for the exact requested file.
    spec=importlib.util.spec_from_file_location(name,path)
    # Allocate its own global namespace.
    result=importlib.util.module_from_spec(spec)
    # Execute only definitions in the selected module.
    spec.loader.exec_module(result)
    # Return the isolated version.
    return result

# Replay one stock-day and version with ordinary common-arm acceptance rules.
def worker(task):
    # Unpack only serializable process inputs.
    symbol,date,version,output=task
    # Read immutable saved job parameters and all session boundaries.
    saved=json.loads((SAVED/(symbol+'_'+date)/'result.json').read_text())
    # Read the saved channel-quality screen for the same day.
    quality=json.loads((SAVED/'channel_quality.json').read_text())[date]
    # Load the unchanged cell orchestration into this process.
    cell=module('pnl_cell',HERE/'clean_window_cell.py')
    # Choose the exact prior anonymous-only or current refreshed implementation.
    source=PRIOR if version=='without_refresh' else HERE
    # Load version-specific book rules.
    book=module('pnl_book',source/'clean_window_book.py')
    # Load version-specific source planning.
    data=module('pnl_data',source/'clean_window_data.py')
    # Bind the actual book and matching exception into the planner.
    data.WindowBook,data.WindowDataError=book.WindowBook,book.WindowDataError
    # Load the corresponding strategy adapter.
    engine=module('pnl_engine',source/'clean_window_engine.py')
    # Bind the same book implementation into execution.
    engine.WindowBook,engine.WindowDataError=book.WindowBook,book.WindowDataError
    # Keep source decoding and planning on the selected version.
    cell.load_symbol,cell.plan_windows=data.load_symbol,data.plan_windows
    # Keep closing exceptions tied to that exact engine module.
    cell.build_engine,cell.WindowExitError=engine.build_engine,engine.WindowExitError
    # Store lightweight worker-local progress with no manager permissions.
    progress={}
    # Install the ordinary cell's progress contract.
    cell.initialize(progress)
    # Identify this one bounded run.
    key=symbol+'_'+date
    # Start elapsed-time reporting before source loading.
    began=time.monotonic()
    # Ensure heartbeat cleanup on every exit.
    stop=threading.Event()
    # Report progress using the existing window/arm fraction.
    def heartbeat():
        # Avoid per-event output and long silent strategy replays.
        while not stop.wait(15):
            # Read the latest cell phase and fraction.
            phase,fraction=progress.get(key,('starting',0))
            # Print a bounded version-labelled progress message.
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {version}/{key}/{phase} elapsed={time.monotonic()-began:.0f}s progress={100*fraction:.1f}%',flush=True)
    # Start a worker-local daemon reporter.
    threading.Thread(target=heartbeat,daemon=True).start()
    # Run all twelve settings with their existing conditional-closing rule.
    try:
        # Persist fills, failed closes, settings and accounting under this version only.
        result=cell.cell((saved['job'],quality,Path(output)/version))
    # Stop reporting even when a programming failure escapes.
    finally:
        # Wake the heartbeat immediately.
        stop.set()
    # Preserve the version and measured runtime with the completion status.
    return dict(result,version=version,elapsed_seconds=time.monotonic()-began)

# Freeze all relevant source definitions before comparing versions.
def hashes():
    # Include dependencies and archived versions, not only directly edited files.
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in (HERE,LEGACY,PRIOR) for p in sorted(folder.glob('*.py'))}

# Execute only the fixed diagnostic sample, with no full-history option.
def main():
    # Require a distinct explicit results location.
    parser=argparse.ArgumentParser(description='Four-stock-day paired snapshot-refresh P&L diagnostic; no full-history mode')
    # Keep reports separate from every prior run.
    parser.add_argument('--output-dir',type=Path,required=True)
    # Default to the user-requested bounded concurrency.
    parser.add_argument('--workers',type=int,default=8,choices=range(1,9))
    # Parse the explicit diagnostic invocation.
    args=parser.parse_args()
    # Resolve the destination before allowing writes.
    output=args.output_dir.resolve()
    # Refuse existing evidence or any directory outside the authorized results root.
    if output.parent!=ROOT or output.exists():
        # Never overwrite baseline or earlier diagnostic results.
        raise ValueError('Choose a new direct subfolder of Capital Stake - Results Codex')
    # Freeze every relevant source byte hash.
    before=hashes()
    # Freeze relevant parsed partition metadata using the source audit's established contract.
    from audit_source_coverage import partition_metadata
    # Resolve the same parsed data root used by the source planner.
    from config_pk import PARSED_ROOT
    # Freeze source partitions once per selected date.
    metadata={date:partition_metadata(PARSED_ROOT,date) for symbol,date in SAMPLE}
    # Freeze exact source job records used in this comparison.
    baselines={symbol+'_'+date:hashlib.sha256((SAVED/(symbol+'_'+date)/'result.json').read_bytes()).hexdigest() for symbol,date in SAMPLE}
    # Create the distinct diagnostic output directory.
    output.mkdir()
    # Document comparison scope before any P&L is observed.
    contract=dict(sample=SAMPLE,versions=['without_refresh','with_refresh'],source_hashes=before,metadata=metadata,baseline_hashes=baselines,workers=args.workers,selection='Four previously diagnosed stock-days; purposive, not representative or selected by new P&L',comparison='Both versions include the anonymous-reference correction; refresh also requires queue reconciliation. Each version retains its own source windows and common twelve-arm closing exclusions. Not a same-window execution-only treatment effect.',full_history=False)
    # Save exact provenance before dispatch.
    (output/'contract.json').write_text(json.dumps(contract,indent=2))
    # Create each version's parent before its stock-day workers create child folders.
    for version in contract['versions']:
        # Keep the two implementations' evidence physically separate.
        (output/version).mkdir()
    # Prepare exactly eight independent cell-version tasks.
    tasks=[(symbol,date,version,str(output)) for symbol,date in SAMPLE for version in contract['versions']]
    # Retain explicit success or failure for every requested comparison.
    results=[]
    # Use isolated spawned workers with no cross-version mutable state.
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        # Submit only the eight declared tasks.
        pending={pool.submit(worker,task) for task in tasks}
        # Collect every result without treating missing comparisons as zero profit.
        while pending:
            # Poll in bounded intervals while workers report real progress.
            ready,pending=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
            # Preserve each completed result and fail loudly on an escaped exception.
            for future in ready:
                # Append its explicit cell outcome.
                results.append(future.result())
                # Print a compact completion record.
                print(json.dumps(results[-1]),flush=True)
    # Refuse a mixed-code or mixed-source result.
    assert hashes()==before
    # Verify every input partition after all processes finish.
    assert all(partition_metadata(PARSED_ROOT,date)==value for date,value in metadata.items())
    # Verify calibration and saved baseline records remained unchanged.
    assert all(hashlib.sha256((SAVED/key/'result.json').read_bytes()).hexdigest()==value for key,value in baselines.items())
    # Save task-level failures independently of P&L.
    (output/'tasks.json').write_text(json.dumps(results,indent=2))
    # Do not summarize an incomplete paired comparison.
    if not all(r['passed'] for r in results):
        # Leave all detailed failure artifacts available for diagnosis.
        raise RuntimeError('At least one paired cell failed; inspect tasks.json')
    # Read completed per-version outcomes.
    cells={version:{symbol+'_'+date:json.loads((output/version/(symbol+'_'+date)/'result.json').read_text()) for symbol,date in SAMPLE} for version in contract['versions']}
    # Load the fixed arm names without selecting winners.
    from stock_search_engine import ARMS
    # Retain per-cell values as well as sample totals.
    comparisons=[]
    # Compare one fixed strategy setting at a time.
    for arm in ARMS:
        # Read the two conditional sample totals without imputing unavailable cells.
        totals={v:sum(r['arms'][arm]['net_pkr'] for r in cells[v].values()) if all(r['arms'] is not None for r in cells[v].values()) else None for v in cells}
        # Record the exact paired difference only when both totals exist.
        comparisons.append(dict(arm=arm,without_refresh_pkr=totals['without_refresh'],with_refresh_pkr=totals['with_refresh'],difference_pkr=totals['with_refresh']-totals['without_refresh'] if all(t is not None for t in totals.values()) else None))
    # Measure accepted time and excluded windows separately from profit.
    coverage={v:dict(accepted_minutes=sum(r['accepted_ms'] for r in cells[v].values())/60000,excluded_windows=sum(not w['accepted'] for r in cells[v].values() for w in r['windows'])) for v in cells}
    # Preserve the original published baseline separately from the two-version effect.
    original={symbol+'_'+date:json.loads((SAVED/(symbol+'_'+date)/'result.json').read_text()) for symbol,date in SAMPLE}
    # Save all fixed-arm results and the narrow interpretation.
    summary=dict(comparisons=comparisons,coverage=coverage,original_published_sample_pkr={arm:sum(r['arms'][arm]['net_pkr'] for r in original.values()) for arm in ARMS},scope=contract['comparison'],full_portfolio_impact='Not measured; do not extrapolate the four diagnostic stock-days')
    # Persist complete scalar summary for review.
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    # Create a source-backed visual of each fixed strategy's measured difference.
    import matplotlib.pyplot as plt
    # Exclude unavailable comparisons rather than depicting them as zero.
    available=[r for r in comparisons if r['difference_pkr'] is not None]
    # Construct an explicitly bounded-scope horizontal chart.
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    # Plot observed direction without averaging or choosing a winner.
    ax.barh([r['arm'] for r in available],[r['difference_pkr'] for r in available],color=['#2f7f6f' if r['difference_pkr']>=0 else '#c78244' for r in available])
    # Mark no change exactly.
    ax.axvline(0,color='black',linewidth=.8)
    # Explain that each bar is a different portfolio configuration, not additive profit.
    ax.set_xlabel('P&L with refresh minus without refresh, PKR (each setting separately)')
    # State sample and conditional-window scope prominently.
    ax.set_title('Four diagnostic stock-days only; accepted windows differ between versions')
    # Save the chart beside its numeric source.
    fig.savefig(output/'pnl_difference.png',dpi=150)
    # Print the bounded measured results.
    print(json.dumps(summary,indent=2),flush=True)

# Run only when explicitly invoked as a script.
if __name__=='__main__':
    # Keep worker imports from recursively launching the parent pool.
    main()
