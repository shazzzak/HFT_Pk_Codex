import pandas as pd
import numpy as np

df = pd.read_csv('/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/taper_sweep_PERNAME_20260910_2331.parquet')

# net PKR per (config, symbol) summed across buckets+days
g = df.groupby(['throttle', 'symbol'])['net_pkr'].sum().unstack(0)

# the key comparison: does the taper add ON TOP of queue skew, per name?
g['TAPER_vs_QSKEW'] = g['QSKEW+TAPER_m0.25'] - g['QSKEW']
# and does queue skew itself still beat production, per name (context)
g['QSKEW_vs_PROD'] = g['QSKEW'] - g['PROD']

print('=== does TAPER add on top of QUEUE SKEW, per ticker? (sorted) ===')
show = g[['QSKEW', 'QSKEW+TAPER_m0.25', 'TAPER_vs_QSKEW']].sort_values('TAPER_vs_QSKEW', ascending=False)
print(show.round(0).to_string())

print()
helped = (g['TAPER_vs_QSKEW'] > 0).sum()
hurt = (g['TAPER_vs_QSKEW'] < 0).sum()
total = g['TAPER_vs_QSKEW'].sum()
print(f'taper-on-top-of-qskew: helps {helped}/{len(g)} names, hurts {hurt}, total {total:,.0f} PKR')

# concentration check: how much of the total gain comes from the top 3 names?
top3 = g['TAPER_vs_QSKEW'].sort_values(ascending=False).head(3).sum()
print(f'top-3 names contribute {top3:,.0f} of {total:,.0f} ({100*top3/total:.0f}% if total>0)')

# taper-only vs production too (bar 1 per-ticker)
g['TAPERonly_vs_PROD'] = g['TAPER_m0.25_k1'] - g['PROD']
print()
print(f"taper-ALONE vs PROD: helps {(g['TAPERonly_vs_PROD']>0).sum()}/{len(g)} names, total {g['TAPERonly_vs_PROD'].sum():,.0f}")