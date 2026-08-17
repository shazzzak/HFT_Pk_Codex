# check_feature_stores.py -- which of the 38 shortlist names have a feature store
# built, and is each one COMPLETE (enough day-files for a full 207-day run)?
# Uses DuckDB to read the watchlist and glob the feature-store day-files per symbol.

# DuckDB: queries parquet/csv in place, no server
import duckdb
# pathlib for the directory existence check
from pathlib import Path

# ---- paths (edit if your layout differs) ----
# the ranked shortlist (has 'symbol' and 'tier' columns)
WATCHLIST = "/Users/shazzak/Capital Stake - Results/mm_watchlist_final.csv"
# per-symbol feature store root: feature_store/{SYM}/date=YYYY-MM-DD.parquet
FEATURE_STORE = "/Users/shazzak/Capital Stake - Results/feature_store"
# how many trading days a COMPLETE store should have (full sample)
EXPECTED_DAYS = 207
# --------------------------------------------

# open an in-memory DuckDB connection
con = duckdb.connect()

# 1) pull the shortlist symbols (with tier, so the report is ordered sensibly)
syms = con.execute(f"""
    -- read the watchlist csv directly
    SELECT tier, symbol
    FROM read_csv_auto('{WATCHLIST}')
    ORDER BY tier, symbol
""").fetchall()

# 2) for each symbol, glob its feature-store day-files and summarise coverage.
#    DuckDB's glob() lists matching paths; we count files and read the date range
#    from the Hive-partitioned 'date=' folder names via parquet_scan's filename.
print(f"{'tier':14s} {'symbol':8s} {'built':6s} {'day_files':>9s} "
      f"{'first_date':>12s} {'last_date':>12s} {'status':>10s}")

# tallies for the summary line
n_built = 0
n_complete = 0

# walk each shortlist symbol
for tier, sym in syms:
    # the symbol's feature-store directory
    d = Path(FEATURE_STORE) / sym
    # not built at all -> report and move on
    if not d.is_dir():
        print(f"{tier:14s} {sym:8s} {'NO':6s} {0:>9d} {'-':>12s} {'-':>12s} {'MISSING':>10s}")
        continue
    # count the day-files and read the min/max partition date via the file paths.
    # glob returns one row per matching file; we parse the date out of 'date=...'.
    q = f"""
        WITH files AS (
            -- every day-file for this symbol
            SELECT file
            FROM glob('{FEATURE_STORE}/{sym}/date=*/*.parquet')
            UNION ALL
            -- also match the flat 'date=YYYY-MM-DD.parquet' layout
            SELECT file
            FROM glob('{FEATURE_STORE}/{sym}/date=*.parquet')
        )
        SELECT
            -- how many day-files exist
            COUNT(*) AS n_files,
            -- earliest / latest date parsed from the 'date=' token in the path
            MIN(regexp_extract(file, 'date=([0-9-]+)', 1)) AS first_date,
            MAX(regexp_extract(file, 'date=([0-9-]+)', 1)) AS last_date
        FROM files
    """
    # run it
    n_files, first_date, last_date = con.execute(q).fetchone()
    # built if any files exist
    built = n_files and n_files > 0
    # complete if it has (nearly) the full sample -- allow a small margin
    complete = built and n_files >= EXPECTED_DAYS - 5
    # tally
    if built:
        n_built += 1
    if complete:
        n_complete += 1
    # status label
    status = "COMPLETE" if complete else ("PARTIAL" if built else "MISSING")
    # one row
    print(f"{tier:14s} {sym:8s} {('YES' if built else 'NO'):6s} {int(n_files or 0):>9d} "
          f"{str(first_date or '-'):>12s} {str(last_date or '-'):>12s} {status:>10s}")

# summary
print(f"\n{len(syms)} shortlist symbols | {n_built} built | {n_complete} complete "
      f"(>= {EXPECTED_DAYS-5} day-files) | {len(syms)-n_built} missing")
print("\nCOMPLETE = ready for the universe run. PARTIAL = needs a feature-store rebuild.")
print("MISSING = build_feature_store.py must be run for it first.")