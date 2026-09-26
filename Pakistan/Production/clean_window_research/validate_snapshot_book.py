# Independently reconcile source reconstruction against subsequent exchange snapshots.
import argparse,json,sys,time,threading,heapq,hashlib,tarfile
# Preserve exact decimal source units independently of the book helper.
from decimal import Decimal
# Accumulate independently reconstructed price-level totals.
from collections import defaultdict
# Resolve source and evidence paths explicitly.
from pathlib import Path
# Prefer corrected source definitions.
HERE=Path(__file__).resolve().parent
# Make this directory importable from an arbitrary working directory.
sys.path.insert(0,str(HERE))
# Reuse immutable legacy dependencies without writing there.
LEGACY=HERE.parents[1]/'existing_mm_live'
# Keep corrected modules first.
sys.path.append(str(LEGACY))
# Disable bytecode output throughout source validation.
sys.dont_write_bytecode=True
# Import only reconstruction and evidence helpers at startup.
import clean_window_data as data
# Use the corrected generation-aware book for actual replay.
from clean_window_book import WindowBook
# Verify source metadata before and after each bounded cell.
from audit_source_coverage import partition_metadata
# Locate the configured actual parsed store.
from config_pk import PARSED_ROOT
# Define the authorized immutable results root.
ROOT=Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex')
# Reuse saved calibration and channel evidence.
SAVED=ROOT/'clean_window_full_20260925_v1'
# Cover the reused-ID case and four earlier diagnostic cells without profit selection.
SAMPLE=[('SNGP','2026-01-21'),('KEL','2026-05-08'),('BOP','2026-05-20'),('TELE','2026-04-29'),('AGHA','2026-01-01')]

# Reject any loss of exactness in independently calculated source units.
def integer(value,scale):
    # Avoid binary floating-point rounding in the validation oracle.
    scaled=Decimal(str(value))*scale
    # Require finite exact units rather than silently rounding malformed input.
    assert scaled.is_finite() and scaled==scaled.to_integral_value()
    # Return a primitive integer for exact mapping equality.
    return int(scaled)

# Aggregate actual book state without using its price-level or best-quote functions.
def book_levels(book):
    # Keep both sides independently keyed by integer price.
    result=defaultdict(int)
    # Read reconstructed order quantities directly.
    for order in book.o.values():
        # Require positive quantities for every retained executable order.
        assert integer(order.qty,100)>0
        # Accumulate each exact reported-price quantity.
        result[(order.side,integer(order.price,10000))]+=integer(order.qty,100)
    # Freeze the independent aggregate view.
    return dict(result)

# Read authoritative snapshot totals directly from reported level tuples.
def snapshot_levels(snapshot):
    # Do not reconstruct expected totals from the book's identity allocation.
    return {(side,integer(price,10000)):integer(qty,100) for side,price,qty,ids,amounts in snapshot.levels}

# Freeze code and dependency identities for the later guarded launch.
def code_hashes():
    # Include every research and legacy Python dependency conservatively.
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in (HERE,LEGACY) for p in sorted(folder.glob('*.py'))}

# Inspect original FIX records for the actual reused-ID cancellation and replacement.
def raw_generation_proof(output,state):
    # Read the recovered original archive without extraction or modification.
    path=ROOT/'Raw_Capture_Recovery/2026-01-21.tar.gz'
    # Select exact immutable application references on the regular-market channel.
    wanted={180376,454949,454950}
    # Retain complete source lines for independent review.
    records={}
    # Report the bounded original-capture phase.
    state['label']='SNGP/2026-01-21/original FIX references'
    # Stream compressed contents without loading the archive into memory.
    with tarfile.open(path,'r|gz') as archive:
        # Visit actual archive members in their stored order.
        for member in archive:
            # Ignore directory entries.
            if not member.isfile():
                # Only textual capture payloads contain market records.
                continue
            # Obtain a streaming member reader.
            stream=archive.extractfile(member)
            # Search source lines for the three exact channel records.
            for line in stream:
                # Avoid decoding unrelated market traffic.
                if b'10201=2011' not in line or not any(('1181='+str(seq)+'^').encode() in line for seq in wanted):
                    # Keep the scan bounded by cheap byte predicates.
                    continue
                # Preserve original capture bytes as readable text.
                raw=line.decode('utf-8',errors='strict').strip()
                # Parse scalar fields independently of the production parser.
                fields=dict(part.split('=',1) for part in raw.split('|',1)[-1].replace('\x01','^').split('^') if '=' in part)
                # Require exact symbol and channel rather than matching another sequence namespace.
                if fields.get('10201')=='2011' and fields.get('55')=='SNGP' and int(fields.get('1181','0')) in wanted:
                    # Keep the exact record for each requested application reference.
                    records[int(fields['1181'])]=dict(fields=fields,raw=raw)
                # Stop once all causal generations and their cancellation are present.
                if set(records)==wanted:
                    # Exit this member without scanning irrelevant later trading.
                    break
            # Stop the archive scan after complete recovery of the requested records.
            if set(records)==wanted:
                # No full-history raw scan is needed for this regression.
                break
    # Require the entire real cancel/re-add chain.
    assert set(records)==wanted,'Original generation records not found'
    # Verify the same exchange ID exists in two additions at different prices.
    assert records[180376]['fields']['37']==records[454950]['fields']['37']=='0010T96R3F0055LY'
    # Check each original source price directly.
    assert Decimal(records[180376]['fields']['44'])==Decimal('120.7')
    # Check the replacement's source price independently.
    assert Decimal(records[454950]['fields']['44'])==Decimal('120.5')
    # Verify the cancellation points at the earlier application generation.
    assert records[454949]['fields']['10117']=='180376' and records[454949]['fields']['150']=='4'
    # Save all three original messages as reviewable evidence.
    (output/'raw_generation_records.json').write_text(json.dumps(records,indent=2))
    # Return exact source identity metadata for the validation report.
    return dict(archive=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,verified_sequences=sorted(wanted))

# Validate source conservation, next-snapshot reconciliation and one actual execution adapter slice.
def main():
    # Require a distinct output directory.
    parser=argparse.ArgumentParser(description='Book validation only; no full P&L run')
    # Preserve earlier evidence by refusing reused destinations.
    parser.add_argument('--output-dir',type=Path,required=True)
    # Parse this explicit validation invocation.
    args=parser.parse_args()
    # Resolve the output once.
    output=args.output_dir.resolve()
    # Keep validation artifacts within the authorized results root.
    if output.parent!=ROOT or output.exists():
        # Never overwrite previous validation or research results.
        raise ValueError('Choose a new direct results subdirectory')
    # Create the new evidence directory.
    output.mkdir()
    # Freeze all implementation identities before source replay.
    hashes=code_hashes()
    # Preserve completed saved channel-quality evidence.
    quality=json.loads((SAVED/'channel_quality.json').read_text())
    # Accumulate per-cell validation evidence.
    results=[]
    # Retain bounded execution outcomes separately from book equality checks.
    integration={}
    # Report current phase without noisy per-event logging.
    state=dict(label='starting',done=0)
    # Establish a monotonic elapsed-time clock.
    began=time.monotonic()
    # Coordinate heartbeat shutdown.
    stop=threading.Event()
    # Emit one actual phase message per fifteen seconds.
    def heartbeat():
        # Use interruptible waits so completion is immediate.
        while not stop.wait(15):
            # Report scope and elapsed time without an invented completion estimate.
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {state["label"]} elapsed={time.monotonic()-began:.0f}s cells={state["done"]}/5',flush=True)
    # Run the lightweight reporter as a daemon.
    threading.Thread(target=heartbeat,daemon=True).start()
    # Preserve error cleanup without masking validation failures.
    try:
        # Reconcile every retained snapshot in each predeclared source cell.
        for symbol,date in SAMPLE:
            # Identify the source-loading phase.
            state['label']=symbol+'/'+date+'/book reconciliation'
            # Freeze source membership and file metadata.
            metadata=partition_metadata(PARSED_ROOT,date)
            # Read only this stock-day's saved session calibration.
            saved=json.loads((SAVED/(symbol+'_'+date)/'result.json').read_text())
            # Decode source records once.
            events,checkpoints,adds=data.load_symbol(date,symbol)
            # Select source-valid intervals using the current snapshot-refresh planner.
            windows,counts=data.plan_windows(events,checkpoints,adds,quality[date],saved['job']['params']['session_segments'])
            # Track independent exact equality, not just successful function calls.
            compared=matched=mutations=0
            # Keep all detected discrepancies as explicit evidence.
            mismatches=[]
            # Replay each retained source candidate without running a strategy.
            for window in windows:
                # Construct the actual initial generation-aware book.
                book=WindowBook(window['checkpoint'],adds,asof=window['start'])
                # Validate full initial level conservation against raw reported totals.
                assert book_levels(book)==snapshot_levels(window['checkpoint'])
                # Merge exact source ordering with the planner's causal snapshot schedule.
                stream=heapq.merge(events[window['left']:window['right']],[(c['start'],0,i,'S',c) for i,c in enumerate(window['refreshes'])])
                # Inspect every accepted mutation and replacement.
                for ts,rank,sequence,kind,row in stream:
                    # Compare the old reconstructed state before allowing a snapshot to replace it.
                    if kind=='S':
                        # Read authoritative expected totals independently.
                        expected=snapshot_levels(row['snapshot'])
                        # Limit comparisons to depth whose prices both snapshots disclose.
                        bounds={'BUY':max(integer(book.boundary['BUY'],10000),min(p for side,p in expected if side=='BUY')),'SELL':min(integer(book.boundary['SELL'],10000),max(p for side,p in expected if side=='SELL'))}
                        # Include disappeared levels inside that shared known range too.
                        keep=lambda key:key[1]>=bounds['BUY'] if key[0]=='BUY' else key[1]<=bounds['SELL']
                        # Aggregate the incrementally reconstructed state independently.
                        before={k:v for k,v in book_levels(book).items() if keep(k)}
                        # Restrict the authoritative picture to comparable depth.
                        target={k:v for k,v in expected.items() if keep(k)}
                        # Count this independent boundary comparison.
                        compared+=1
                        # Record exact integer-state agreement.
                        matched+=before==target
                        # Preserve any discrepancy instead of letting refresh conceal it.
                        if before!=target:
                            # Record the exact differing quantities for every affected level.
                            mismatches.append(dict(start=window['start'],snapshot=ts,origin=row['origin'],differences=[dict(side=k[0],price=k[1]/10000,reconstructed=before.get(k,0)/100,reported=target.get(k,0)/100) for k in sorted(set(before)|set(target)) if before.get(k,0)!=target.get(k,0)]))
                        # Replace the full reported book at the validated causal cutoff.
                        book.refresh(row['snapshot'],asof=ts)
                        # Verify every newly disclosed level too, not only the shared range.
                        assert book_levels(book)==expected
                    # Apply each incremental event once in original application order.
                    else:
                        # Keep exact trade, addition and cancellation semantics.
                        (book.trade if kind=='T' else book.add if row.event=='ORDER_ADD' else book.cancel)(row)
                        # Count accepted source mutations independently of fill activity.
                        mutations+=1
                        # Inspect retained positive quantities after each mutation.
                        book_levels(book)
            # Refuse source replacement during validation.
            assert partition_metadata(PARSED_ROOT,date)==metadata
            # Save each independent boundary check's count and any actual discrepancies.
            result=dict(symbol=symbol,date=date,windows=len(windows),source_minutes=sum(w['end']-w['start'] for w in windows)/60000,compared_snapshots=compared,exact_matches=matched,source_mutations=mutations,mismatches=mismatches,source_counts=counts,metadata=metadata)
            # Preserve evidence even if a later cell fails.
            (output/(symbol+'_'+date+'.json')).write_text(json.dumps(result,indent=2))
            # Accumulate the predeclared sample's actual checks.
            results.append(result)
            # Never certify an unexplained snapshot discrepancy.
            assert compared>0 and not mismatches,(symbol,date,'snapshot discrepancies',len(mismatches))
            # Exercise actual account-preserving replay once on a deterministic short interval.
            if symbol=='TELE':
                # Import execution only for this bounded integration check.
                from clean_window_engine import build_engine,WindowExitError
                # Preserve all twelve fixed configurations.
                from stock_search_engine import ARMS
                # Select a short refreshed slice with real source activity, never by its profit.
                candidates=[w for w in windows if w['refreshes'] and w['right']>w['left'] and w['end']-w['start']<=180000]
                # Require a real bounded test slice rather than silently skipping execution coverage.
                assert candidates,'No short refreshed TELE integration interval'
                # Choose the most event-active eligible short slice.
                window=max(candidates,key=lambda w:w['right']-w['left'])
                # Record the exact shared interval and expected market replacements.
                integration=dict(symbol=symbol,date=date,start=window['start'],end=window['end'],expected_refreshes=len(window['refreshes']),arms={})
                # Test each original setting on identical source events and snapshots.
                for arm in ARMS:
                    # Report the real bounded strategy integration phase.
                    state['label']='TELE/'+date+'/integration/'+arm
                    # Preserve original calibration, costs, seed and order handling.
                    engine,effective,signals=build_engine(saved['job'],arm,window,adds)
                    # Distinguish declared closing exclusions from reconstruction defects.
                    try:
                        # Exercise source handling, queue refresh and actual account accounting.
                        engine.replay(events,window)
                        # Require delivery of the complete source-selected refresh schedule.
                        assert engine.snapshot_refreshes==len(window['refreshes'])
                        # Preserve success without claiming a full profit comparison.
                        integration['arms'][arm]=dict(closed=True,refreshes=engine.snapshot_refreshes,fills=len(engine.fills))
                    # Keep source-safe closing uncertainty visible in the report.
                    except WindowExitError as error:
                        # Preserve the actual account exposure and exclusion reason.
                        integration['arms'][arm]=dict(closed=False,reason=str(error),refreshes=engine.snapshot_refreshes,position=engine.pos)
                # Require all source snapshots to be traversed for this integration fixture.
                assert all(r['refreshes']==len(window['refreshes']) for r in integration['arms'].values())
                # Save actual integration outcomes separately from source equality.
                (output/'integration.json').write_text(json.dumps(integration,indent=2))
            # Mark this predeclared cell complete.
            state['done']+=1
            # Print the independently measured counts only.
            print(json.dumps({k:result[k] for k in ('symbol','date','source_minutes','compared_snapshots','exact_matches','source_mutations')}),flush=True)
        # Confirm the actual original amendment messages after parsed-source validation.
        raw=raw_generation_proof(output,state)
        # Produce a reproducible picture of actual comparisons rather than estimated coverage.
        import matplotlib.pyplot as plt
        # Keep stock-day labels readable without implying a random sample.
        labels=[r['symbol']+'\n'+r['date'] for r in results]
        # Plot the exact number of next-snapshot agreements per diagnostic cell.
        figure,axis=plt.subplots(figsize=(10,5))
        # Show only completed comparisons in shared reported depth.
        bars=axis.bar(labels,[r['exact_matches'] for r in results],color='#287d68')
        # Label each count directly for comparison against the saved JSON evidence.
        axis.bar_label(bars,padding=4)
        # Name the measured unit precisely.
        axis.set_ylabel('Exact next-snapshot matches')
        # State the diagnostic limit alongside the zero-discrepancy result.
        axis.set_title('Book validation: zero mismatches in shared reported depth\nFive diagnostic stock-days; not a full-history P&L check')
        # Leave space above the tallest count label.
        axis.margins(y=0.15)
        # Keep all labels inside the saved image.
        figure.tight_layout()
        # Preserve the plot beside its underlying per-cell source counts.
        figure.savefig(output/'snapshot_matches.png',dpi=160)
        # Release graphical resources before final source fingerprint verification.
        plt.close(figure)
        # Refuse to certify a mixture of source-code revisions.
        assert code_hashes()==hashes
        # Produce a bounded validation stamp rather than a full-history certification.
        report=dict(passed=True,source_hashes=hashes,cells=[{k:r[k] for k in ('symbol','date','source_minutes','compared_snapshots','exact_matches','source_mutations')} for r in results],raw_generation_proof=raw,integration=integration,elapsed_seconds=time.monotonic()-began,scope='Five diagnostic stock-days; exact next-snapshot agreement within shared reported depth; every adopted full snapshot conserves reported quantities; one short twelve-arm integration; not full-history or full-depth certification',snapshot_timing='Only quiet source-second cutovers are adopted; ambiguous timing and unreconstructible intervals remain explicitly unavailable')
        # Publish a successful stamp only after every required assertion passes.
        (output/'validation.json').write_text(json.dumps(report,indent=2))
        # Display final bounded verification counts.
        print(json.dumps(dict(passed=True,compared=sum(r['compared_snapshots'] for r in results),matched=sum(r['exact_matches'] for r in results),output=str(output))),flush=True)
    # Always stop progress reporting on either success or failure.
    finally:
        # Release the reporter immediately.
        stop.set()

# Never run validation merely by importing its fingerprint helper.
if __name__=='__main__':
    # Execute the explicit bounded validation request.
    main()
