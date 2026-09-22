# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# inspect_fills_schema.py -- does the on-disk fills parquet already carry a
# fill-instant mid (mid / bb / ba)? If so, zeroing the decomposition residual is
# a one-line change (read the stored mid instead of looking it up). Prints the
# schema, sample rows, and null-counts. Pure read-only inspection.
#
# Run:  python3 inspect_fills_schema.py

# DuckDB drives the parquet read (already used elsewhere in the project)
import duckdb

# the fills file to inspect (one naive-strategy symbol-day)
# Resolve this filesystem path through the canonical checkout/data configuration.
PARQUET = (str(_hft_paths.RESULTS_ROOT / 'fills/naive/UBL/date=2025-09-01.parquet'))

# open an in-memory DuckDB connection
con = duckdb.connect()

# ---- 1. COLUMN SCHEMA: the decisive check -- is a mid/bb/ba column present? ----
print("=" * 70)
print("1. SCHEMA (column name + type)")
print("=" * 70)
# DESCRIBE returns one row per column: name, type, null?, key, default, extra
schema = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{PARQUET}')").fetchdf()
# print the full schema (all columns, no truncation)
print(schema[["column_name", "column_type"]].to_string(index=False))

# ---- 2. SAMPLE ROWS: eyeball the actual values ----
print("\n" + "=" * 70)
print("2. FIRST 10 ROWS")
print("=" * 70)
# pull the first 10 fills
sample = con.execute(
    f"SELECT * FROM read_parquet('{PARQUET}') LIMIT 10").fetchdf()
# print without column truncation
import pandas as pd
# show all columns
pd.set_option("display.max_columns", None)
# widen so nothing is cut
pd.set_option("display.width", 200)
# print the sample
print(sample.to_string(index=False))

# ---- 3. NULL-COUNTS for any mid-like column that exists ----
print("\n" + "=" * 70)
print("3. MID-COLUMN POPULATION (only for columns that actually exist)")
print("=" * 70)
# the set of column names present on disk
cols = set(schema["column_name"].tolist())
# candidate fill-instant-mid columns to check
candidates = [c for c in ("mid", "bb", "ba", "mid_at_fill", "fair") if c in cols]
# none present -> the mid is NOT logged; engine edit needed
if not candidates:
    print("no mid/bb/ba/mid_at_fill/fair column present on disk.")
    print("-> the fill-instant mid is NOT logged; zeroing the residual needs")
    print("   the engine _fill edit (with sign-off), not a decomposition change.")
else:
    # for each present candidate, count non-null and show range
    for c in candidates:
        # total rows, non-null count, min, max of this column
        row = con.execute(f"""
            SELECT COUNT(*)            AS n_fills,
                   COUNT({c})          AS n_nonnull,
                   MIN({c})            AS min_val,
                   MAX({c})            AS max_val
            FROM read_parquet('{PARQUET}')
        """).fetchdf().iloc[0]
        # report population and range
        print(f"  {c}: {int(row.n_nonnull)}/{int(row.n_fills)} non-null "
              f"(min={row.min_val}, max={row.max_val})")
    # verdict
    print("\n-> if n_nonnull == n_fills for a mid (or both bb AND ba), the")
    print("   fill-instant mid IS on disk: the residual fix is a decomposition")
    print("   one-liner (use the stored mid), no engine change.")

# close the connection
con.close()
