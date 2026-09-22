# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# diag_scorebps_speed.py -- find why Path B (score_bps) is ~12s/day.
# Times the feature-store READ vs the join vs the economics, on ONE naive-PPL day.
# Run from existing_mm_live/:  python diag_scorebps_speed.py

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

# one high-fill day for naive PPL
DATE, SYM = "2026-06-30", "PPL"

# --- run the backtest once (this part we know is ~1.3s) ---
# open datasets
dsets = R.open_datasets(DATE)
# load tables
u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, SYM)
s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, SYM)
t = R.read_symbol(dsets["trades"], R.REQ_TRADES, SYM)
# build events
events, snap_groups, t = R.build_events(u, s, t)
# session window
cont = t[t["initiator"] != "AUCTION"]
t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
# cfg
cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED))
# time the backtest
tb = time.time()
bt = Backtester(NaiveSymmetricMM(**R.STRAT), cfg)
fills, eq, st = bt.run(events, snap_groups)
print(f"backtest.run()        {time.time()-tb:.2f}s  ({len(fills)} fills)")

# --- now time each piece of score_bps SEPARATELY ---
# 1) the feature-store parquet READ
tr = time.time()
fs_day = pd.read_parquet(
    FS_ROOT / SYM / f"date={DATE}.parquet",
    columns=["ts_exch", "mid", "spread_bps", "obi_1", "toxicity", "realized_vol_bps"])
print(f"read_parquet(fs_day)  {time.time()-tr:.2f}s  ({len(fs_day):,} rows)")

# 2) the join_fill_context call
tj = time.time()
f = PF.join_fill_context(fills, fs_day)
print(f"join_fill_context     {time.time()-tj:.2f}s")

# 3) the economics (vectorized bps)
te = time.time()
side_sgn = np.where(f["side"] == "BUY", 1.0, -1.0)
f["capture"] = FA.capture_bps(side_sgn, f["px"], f["mid0"])
f["markout"] = FA.markout_bps(side_sgn, f["mid0"], f["mid_h"])
f["net"] = FA.net_bps(side_sgn, f["px"], f["mid0"], f["mid_h"])
print(f"economics (bps)       {time.time()-te:.2f}s")

# the verdict: which of read / join / economics is the ~12s cost?
print("\n-> the largest of read/join/economics is what to optimize")
