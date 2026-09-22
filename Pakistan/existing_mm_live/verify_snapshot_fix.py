# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# verify_snapshot_fix.py -- confirm the snapshot optimization preserves fills.
# Run from existing_mm_live/:  python verify_snapshot_fix.py
# WANT: 480 fills (unchanged) and a much lower time / higher events/sec.

# timing
import time
# paths
from pathlib import Path
# driver + engine
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel
from micro_mm import MicrostructureMM

# point loader at the raw store
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# one representative PPL day
dsets = R.open_datasets("2026-06-30")
# load the three tables
u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, "PPL")
s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, "PPL")
t = R.read_symbol(dsets["trades"], R.REQ_TRADES, "PPL")
# build events (now pre-parses snapshots into PreparsedSnapshot structs)
events, snap_groups, t = R.build_events(u, s, t)
# confirm the pre-parse took effect: snap_groups values should NOT be DataFrames
import pandas as pd
sample_val = next(iter(snap_groups.values()))
print(f"snap_groups value type: {type(sample_val).__name__} "
      f"(want 'PreparsedSnapshot', NOT 'DataFrame')")
# session window
cont = t[t["initiator"] != "AUCTION"]
t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
# untuned micro, equity logging off (matches the diagnostic that gave 480 fills)
p = dict(R.MICRO_PARAMS); p["min_edge_pct"] = 0.0; p["improve_ticks"] = 0.0
cfg = dict(R.CFG, session=(t0, t1),
           latency_model=LatencyModel(seed=R.LATENCY_SEED), log_equity=False)
# time one backtest
st = time.time()
bt = Backtester(MicrostructureMM(session_ms=(t0, t1), **p), cfg)
fills, eq, stats = bt.run(events, snap_groups)
el = time.time() - st
# report: fills MUST be 480; time should drop sharply from ~13s
print(f"{el:.2f}s, {len(fills)} fills (want 480), {len(events)/el:,.0f} events/sec")
# explicit pass/fail on the fill count (the correctness gate)
print("FILLS MATCH -- fix is behavior-neutral" if len(fills) == 480
      else "!!! FILLS CHANGED -- pre-parse dropped/altered something, do NOT trust")
