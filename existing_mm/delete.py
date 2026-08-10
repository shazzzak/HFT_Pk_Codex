# what ARE the 64 giant OFI rows? Use columns that exist in the store.
import build_feature_store as B
dsets = B.R.open_datasets("2026-06-30")
df = B.build_one("2026-06-30", "UBL", dsets)
big = df[df["ofi_l1"].abs() > 10000]
print(f"{len(big)} rows with |ofi_l1|>10000")
# columns that exist: ts_exch, mid, spread_bps, obi_1, ofi_l1, toxicity, etc.
print(big[["ts_exch","mid","spread_bps","obi_1","ofi_l1","realized_vol_bps"]].head(15).to_string(index=False))
# are the big-OFI rows at abnormal spreads (halt/gap = suspicious) or normal (real relevel)?
print(f"\nbig-OFI median spread: {big['spread_bps'].median():.1f} bps")
print(f"all-rows median spread: {df['spread_bps'].median():.1f} bps")
# do they cluster at one timestamp (corruption) or spread through the day (real)?
print(f"\nbig-OFI time span: {big['ts_exch'].min()} to {big['ts_exch'].max()}")
print(f"unique timestamps among big rows: {big['ts_exch'].nunique()} of {len(big)}")