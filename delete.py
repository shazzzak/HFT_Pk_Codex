# does the 30s label near session close correctly go NaN? Check the last rows.
import build_feature_store as B
dsets = B.R.open_datasets("2026-06-30")
df = B.build_one("2026-06-30", "UBL", dsets)
# last 5 rows: their +30s label MUST be NaN (no future mid 30s past the close)
print(df[["ts_exch","mid","markout_5000ms_bps","markout_30000ms_bps"]].tail(5).to_string(index=False))
# how many rows are within 30s of the last timestamp?
last_t = df["ts_exch"].max()
near_end = (df["ts_exch"] > last_t - 30000).sum()
print(f"\nrows within 30s of close: {near_end} -- these SHOULD have NaN 30s label")
print(f"rows with non-null 30s label: {df['markout_30000ms_bps'].notna().sum()} of {len(df)}")