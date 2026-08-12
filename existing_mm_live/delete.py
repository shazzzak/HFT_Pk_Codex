# verify_snapshot_fix.py -- confirm the snapshot optimization preserves fills.
# run from existing_mm_live/:  python verify_snapshot_fix.py
# WANT: 480 fills (unchanged) and a much lower time / higher events/sec.
import time
from pathlib import Path
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel
from micro_mm import MicrostructureMM
# point loader at the raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# one representative PPL day
dsets = R.open_datasets("2026-06-30")
u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, "PPL")
s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, "PPL")
t = R.read_symbol(dsets["trades"], R.REQ_TRADES, "PPL")
# build events (now pre-parses snapshots)
events, snap_groups, t = R.build_events(u, s, t)
# session window
cont = t[t["initiator"] != "AUCTION"]
t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
# untuned micro, equity logging off
p = dict(R.MICRO_PARAMS); p["min_edge_pct"] = 0.0; p["improve_ticks"] = 0.0
cfg = dict(R.CFG, session=(t0, t1),
           latency_model=LatencyModel(seed=R.LATENCY_SEED), log_equity=False)
# time one backtest
st = time.time()
bt = Backtester(MicrostructureMM(session_ms=(t0, t1), **p), cfg)
fills, eq, stats = bt.run(events, snap_groups)
el = time.time() - st
# report: fills MUST be 480; time should drop sharply
print(f"{el:.2f}s, {len(fills)} fills (want 480), {len(events)/el:,.0f} events/sec")