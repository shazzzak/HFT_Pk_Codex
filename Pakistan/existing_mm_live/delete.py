
import sys; sys.path.insert(0,'existing_mm_live')
import run_legacy_mm as R, numpy as np, pandas as pd
from pathlib import Path
R.PARSED_ROOT = Path('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Parsed')
names = sorted(pd.read_csv('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/mm_watchlist_final.csv')['symbol'].astype(str).unique())
d = R.discover_dates()[len(R.discover_dates())//2]
ds = R.open_datasets(d)
rows=[]
for s in names:
    t = R.read_symbol(ds['trades'], ['symbol','price'], s)
    if len(t): rows.append((s, float(np.median(t['price']))))
for s,p in rows: print(f'{s},{p:.2f}')
