
import mm_harness as H, run_legacy_mm as R
from pathlib import Path
R.PARSED_ROOT = Path('/Users/shazzak/Capital Stake - Parsed')
d = R.discover_dates()[15]
ds = R.open_datasets(d)
sc,pr,wi,sg = H.load_scales(),H.load_profiles(),H.load_windows(),H.load_segments()
sym='PPL'
p = H.build_micro_params(50, sc[sym], pr[sym], wi.get(sym,(5.,1.)), sg.get(str(d)))
dr = H.run_symbol_day(d, sym, ds, p)
print('fills columns:', list(dr.fills.columns))
if 'bucket' in dr.fills.columns:
    print('bucket counts:'); print(dr.fills['bucket'].value_counts())
else:
    print('NO bucket column -> both files default everything to middle')
