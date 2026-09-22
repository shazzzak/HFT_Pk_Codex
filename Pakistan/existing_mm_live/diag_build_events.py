# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# diag_build_events.py -- isolate build_events timing + snapshot count, to see
# whether it is slow (large snapshot volume -> many prep_snapshot calls) or hung.
# Run from existing_mm_live/:  python diag_build_events.py

# timing + flush so output appears immediately
import time, sys
# paths
from pathlib import Path
# frames
import pandas as pd
# driver
import run_legacy_mm as R

# raw store
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# the day under test
DATE, SYM = "2026-06-30", "PPL"

# open + read
dsets = R.open_datasets(DATE)
u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, SYM)
s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, SYM)
t = R.read_symbol(dsets["trades"], R.REQ_TRADES, SYM)

# how many distinct snapshot messages? (each becomes one prep_snapshot call)
n_msgs = s["msg_seq"].nunique()
print(f"ob_snapshot rows={len(s):,}  distinct msg_seq={n_msgs:,}", flush=True)
print(f"  -> prep_snapshot will be called {n_msgs:,} times at build", flush=True)

# time JUST the groupby (no pre-parse) to separate grouping cost
t0 = time.time()
groups = list(s.groupby("msg_seq"))
print(f"groupby only          {time.time()-t0:.2f}s  ({len(groups):,} groups)", flush=True)

# time JUST the pre-parse loop (the prep_snapshot calls)
from snapshot_prep import prep_snapshot
t0 = time.time()
snap_groups = {ms: prep_snapshot(grp) for ms, grp in groups}
print(f"prep_snapshot loop    {time.time()-t0:.2f}s", flush=True)

# time the FULL build_events for comparison
t0 = time.time()
events, sg, t2 = R.build_events(u, s, t)
print(f"full build_events     {time.time()-t0:.2f}s  ({len(events):,} events)", flush=True)

# verdict
print(f"\n-> if prep_snapshot loop ~= full build_events time, the snapshot", flush=True)
print(f"   pre-parse ({n_msgs:,} calls) is the cost -- and the confirmation pays", flush=True)
print(f"   it 4x per symbol-day (once per config) instead of once.", flush=True)
