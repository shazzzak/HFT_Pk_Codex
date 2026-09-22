# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# diag_confirm_day.py -- time EACH sub-step of the confirmation's per-day path,
# to find the ~12s that score_bps did NOT account for. Replicates exactly what
# confirm_micro_vs_naive.run_one + score_bps do for ONE naive-PPL day, timing
# every stage separately. Run from existing_mm_live/:  python diag_confirm_day.py

# timing
import time
# paths
from pathlib import Path
# frames + arrays
import pandas as pd
import numpy as np
# driver + engine + strategy
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel, NaiveSymmetricMM
# join + economics
import persist_fills as PF
import fill_attribution as FA

# raw store + feature store
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# Resolve this filesystem path through the canonical checkout/data configuration.
FS_ROOT = Path(str(_hft_paths.RESULTS_ROOT / 'feature_store'))

# one naive-PPL day (the first cell that ran at ~14s/day)
DATE, SYM = "2026-06-30", "PPL"

# time opening datasets
t = time.time()
dsets = R.open_datasets(DATE)
print(f"open_datasets        {time.time()-t:.2f}s")

# time each read_symbol
t = time.time()
u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, SYM)
print(f"read ob_updates      {time.time()-t:.2f}s  ({len(u):,} rows)")
t = time.time()
s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, SYM)
print(f"read ob_snapshot     {time.time()-t:.2f}s  ({len(s):,} rows)")
t = time.time()
tr = R.read_symbol(dsets["trades"], R.REQ_TRADES, SYM)
print(f"read trades          {time.time()-t:.2f}s  ({len(tr):,} rows)")

# time build_events (includes the snapshot pre-parse)
t = time.time()
events, snap_groups, tr = R.build_events(u, s, tr)
print(f"build_events         {time.time()-t:.2f}s  ({len(events):,} events)")

# session window
cont = tr[tr["initiator"] != "AUCTION"]
t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())

# time the backtest
t = time.time()
cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED))
bt = Backtester(NaiveSymmetricMM(**R.STRAT), cfg)
fills, eq, st = bt.run(events, snap_groups)
print(f"backtest.run()       {time.time()-t:.2f}s  ({len(fills)} fills)")

# time reading bt.eod (Path A) -- should be instant
t = time.time()
_ = bt.eod["equity_liquidated"] if bt.eod else 0.0
print(f"read bt.eod          {time.time()-t:.4f}s")

# time the Path B feature read + join + economics (score_bps equivalent)
t = time.time()
fs_day = pd.read_parquet(FS_ROOT/SYM/f"date={DATE}.parquet",
    columns=["ts_exch","mid","spread_bps","obi_1","toxicity","realized_vol_bps"])
f = PF.join_fill_context(fills, fs_day)
side_sgn = np.where(f["side"]=="BUY",1.0,-1.0)
f["net"] = FA.net_bps(side_sgn, f["px"], f["mid0"], f["mid_h"])
print(f"score_bps (B path)   {time.time()-t:.2f}s")

# TOTAL per-day cost as the confirmation experiences it
print("\n-> sum of the above is the confirmation's real per-day cost")
print("-> if backtest.run() dominates, the confirmation's 14s/day is the BACKTEST,")
print("   not score_bps -- meaning the ENGINE is slow on this path for some reason")
