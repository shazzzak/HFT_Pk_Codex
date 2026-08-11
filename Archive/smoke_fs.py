# smoke_fs.py -- one symbol-day sanity check before the 2-hour full build.
# Run from the folder containing build_feature_store.py:  python smoke_fs.py

# Import the feature-store module (must be build_feature_store.py in this folder).
import build_feature_store as B

# Open the datasets for one known-good date.
dsets = B.R.open_datasets("2026-06-30")
# Guard: fail clearly if the date/partition isn't found.
assert dsets is not None, "no datasets for 2026-06-30 -- check PARSED_ROOT path in build_feature_store.py"

# Build the feature+label frame for one symbol-day.
df = B.build_one("2026-06-30", "UBL", dsets)
# Guard: fail clearly if nothing was produced.
assert df is not None and len(df) > 0, "build_one returned no rows -- inspect the traceback / book replay"

# Row count -- expect a few thousand two-sided-touch rows for a liquid name.
print("rows:", len(df))
# Feature sanity -- ranges should be finite and sensible.
print(df[["obi_1", "micro_dev_bps", "ofi_l1", "toxicity",
          "realized_vol_bps", "markout_5000ms_bps"]].describe().to_string())
# Label coverage per horizon -- high at 1s/5s, lower at 30s (rows near close lack a +30s mid).
for h in [1000, 5000, 30000]:
    print(f"label coverage {h}ms: {df[f'markout_{h}ms_bps'].notna().mean():.3f}")