# Validate one recovered interval through all twelve unchanged strategy adapters.
import json, time
# Locate immutable saved settings and separate validation evidence.
from pathlib import Path
# Load the corrected source planner.
from clean_window_data import load_symbol, plan_windows
# Use the corrected local engine and its explicit closing exception.
from clean_window_engine import build_engine, WindowExitError
# Preserve the exact declared arm set.
from stock_search_engine import ARMS
# Read exact nullable resting references without relying on parser-resolved IDs.
from psx_reference_rows import reference
# Read settings from the completed historical run.
saved=Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/clean_window_full_20260925_v1')
# Keep validation evidence separate from published research profits.
out=saved.parent/'Anonymous_Reference_Fix_20260925_v1'
# Use the source sample's improved TELE stock-day.
old=json.loads((saved/'TELE_2026-04-29/result.json').read_text())
# Reuse the saved channel-quality contract.
quality=json.loads((saved/'channel_quality.json').read_text())['2026-04-29']
# Load one stock-day without a full-market reparse.
events,checkpoints,adds=load_symbol('2026-04-29','TELE')
# Select corrected source-only candidate windows.
windows,reasons=plan_windows(events,checkpoints,adds,quality,old['job']['params']['session_segments'])
# Find a retained interval containing an actual trade whose add reference is missing.
window=next(w for w in windows if any(kind=='T' and (reference(getattr(row,'buy_ref',None)) or reference(getattr(row,'sell_ref',None))) not in adds for ts,rank,seq,kind,row in events[w['left']:w['right']]))
# Keep each arm's validation result distinct from historical performance reports.
results={}
# Establish one heartbeat clock across the bounded validation.
began=last=time.monotonic()
# Replay the same single interval under every existing arm.
for index,arm in enumerate(ARMS):
    # Reuse the original job, seed, sizing, fees and controls.
    engine,effective,counts=build_engine(old['job'],arm,window,adds)
    # Enforce local corrected-book usage before running anything.
    assert hasattr(engine.book,'observed_references')
    # Publish bounded validation progress when needed.
    def progress(fraction):
        # Retain a single shared heartbeat clock.
        global last
        # Avoid per-event terminal output.
        if time.monotonic()-last>=15:
            # Estimate work from completed arms and current interval progress.
            done=index+fraction
            # Measure elapsed time only after validation started.
            elapsed=time.monotonic()-began
            # Print a concise scoped replay heartbeat.
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} TELE/2026-04-29/{arm} elapsed={elapsed:.0f}s remaining={elapsed*(len(ARMS)/max(done,0.001)-1):.0f}s jobs={index}/{len(ARMS)} progress={100*done/len(ARMS):.1f}%',flush=True)
            # Reset the heartbeat clock.
            last=time.monotonic()
    # Preserve genuine closing failures as outcomes, not cash-adjustment fixes.
    try:
        # Exercise source resolution, quoting, queue updates, fills and closing.
        result=engine.replay(events,window,progress)
        # Require a fully flat ending account for a successful close.
        assert abs(engine.pos)<1e-9
        # Save validation status and actual anonymous reductions.
        results[arm]=dict(closed=True,anonymous_trade_reductions=engine.book.anonymous_reference_reductions,fills=len(engine.fills))
    # An explicit close failure is distinct from a programming or source error.
    except WindowExitError as error:
        # Retain exposure for review without counting the interval as accepted profit.
        results[arm]=dict(closed=False,reason=str(error),position=engine.pos,cash=engine.cash,anonymous_trade_reductions=engine.book.anonymous_reference_reductions)
# Require every strategy to traverse at least one formerly unresolvable trade.
assert all(r['anonymous_trade_reductions']>0 for r in results.values())
# Save bounded integration evidence without publishing performance claims.
(out/'single_window_integration.json').write_text(json.dumps(dict(symbol='TELE',date='2026-04-29',start=window['start'],end=window['end'],arms=results,elapsed_seconds=time.monotonic()-began,scope='One recovered window; not a full stock-day or profitability comparison'),indent=2))
# Print exact validation outcomes.
print(json.dumps(results,indent=2))
