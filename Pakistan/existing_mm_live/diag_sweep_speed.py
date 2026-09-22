# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# diag_sweep_speed.py -- find WHY each backtest is slow. Times one symbol-day's
# stages, audits event-stream dtypes (the datetime-object-cast hypothesis), and
# times a single backtest with a breakdown. Run FRESH from existing_mm_live/.
# ============================================================================

# timers
import time
# paths
from pathlib import Path
# frames
import pandas as pd
# arrays
import numpy as np

# engine + latency
from mm_backtest import Backtester, LatencyModel
# micro strategy
from micro_mm import MicrostructureMM
# driver
import run_legacy_mm as R
# context join
import persist_fills as PF
# economics
import fill_attribution as FA

# raw store
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# feature store
# Resolve this filesystem path through the canonical checkout/data configuration.
FS = Path(str(_hft_paths.RESULTS_ROOT / 'feature_store'))

# one representative day + the high-fill symbol
DATE, SYM = "2026-06-30", "PPL"

# rolling stage timer
_tp = time.time()
# open datasets for the date
dsets = R.open_datasets(DATE)
# time the open
print(f"open_datasets           {time.time()-_tp:.2f}s", flush=True); _tp = time.time()

# load the three tables
u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, SYM)
s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, SYM)
t = R.read_symbol(dsets["trades"], R.REQ_TRADES, SYM)
# time the parse/load
print(f"read_symbol x3 (parse)  {time.time()-_tp:.2f}s", flush=True); _tp = time.time()

# build the event stream
events, snap_groups, t = R.build_events(u, s, t)
# time the build
print(f"build_events            {time.time()-_tp:.2f}s", flush=True); _tp = time.time()

# ---- DTYPE AUDIT: the datetime-object-cast hypothesis ----
# events is a list of tuples (ts_exch, kind_rank, appl_seq, kind, row-obj).
# Slow comparisons come from ts_exch being a Python object/Timestamp rather
# than a plain int. Inspect the first event's element types.
print("\n--- DTYPE AUDIT (the object-cast suspicion) ---")
# how many events
print(f"n_events: {len(events):,}")
# type of the sort key (ts_exch) on the first event -- want int, NOT Timestamp/object
if events:
    e0 = events[0]
    print(f"event[0] tuple types: {[type(x).__name__ for x in e0[:4]]}")
    print(f"  ts_exch type: {type(e0[0]).__name__}  (want 'int'; 'Timestamp'/'str' = slow)")
# the trades frame's ts_exch dtype (used for the session window + labels)
print(f"trades ts_exch dtype:   {t['ts_exch'].dtype}  (want int64)")
# check the row objects inside events -- are they namedtuples (fast attr) or dicts/Series (slow)?
if events:
    print(f"event row-obj type:     {type(events[0][4]).__name__}  (want a namedtuple/itertuples row; 'Series' = slow)")

# feature-store day (context)
fs_day = pd.read_parquet(FS/SYM/f"date={DATE}.parquet",
    columns=["ts_exch","mid","spread_bps","obi_1","toxicity","realized_vol_bps"])
# audit its ts_exch dtype too (used in the ASOF join)
print(f"fs_day ts_exch dtype:   {fs_day['ts_exch'].dtype}  (want int64)")
print("--- end audit ---\n")

# session window
cont = t[t["initiator"] != "AUCTION"]
t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())

# ---- time ONE backtest (the cost multiplied by 15 cells x 414 symbol-days) ----
# cfg + seeded latency
cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED))
# untuned micro (the max-fill, slowest cell)
p = dict(R.MICRO_PARAMS); p["min_edge_pct"] = 0.0; p["improve_ticks"] = 0.0
# build strategy
strat = MicrostructureMM(session_ms=(t0, t1), **p)
# time the single backtest
_tp = time.time()
bt = Backtester(strat, cfg)
fills, equity, stats = bt.run(events, snap_groups)
# the key number: one backtest's wall time
bt_s = time.time() - _tp
print(f"ONE backtest.run()      {bt_s:.2f}s  ({len(fills):,} fills)")
# extrapolate the full grid from this
print(f"  -> x15 cells x414 symbol-days = {bt_s*15*414/3600:.1f} hours of pure backtests")

# time the context join (per cell)
_tp = time.time()
fv = PF.join_fill_context(fills, fs_day)
print(f"join_fill_context       {time.time()-_tp:.2f}s")

# ---- if a backtest is slow, is it the event loop or something per-event? ----
# re-time a backtest to see variance, and report events/sec throughput
_tp = time.time()
bt2 = Backtester(MicrostructureMM(session_ms=(t0, t1), **p),
                 dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED)))
f2, e2, s2 = bt2.run(events, snap_groups)
bt2_s = time.time() - _tp
print(f"\nsecond backtest.run()   {bt2_s:.2f}s")
print(f"throughput:             {len(events)/bt2_s:,.0f} events/sec  "
      f"(healthy is >100k/s; <20k/s suggests per-event Python overhead)")
