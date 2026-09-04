# describe_pace_features.py -- print the column names + types of PACE's feature
# store, so we know exactly what's available for the session_scale back-solve.
# Run from anywhere:  python describe_pace_features.py

# glob for the PACE feature-store parquet files (Hive-style date=... files)
PACE_GLOB = "/Users/shazzak/Capital Stake - Results/feature_store/PACE/*.parquet"

# try DuckDB first (matches the CLI DESCRIBE you asked for)
try:
    # import the DuckDB engine
    import duckdb
    # DESCRIBE reads the schema without loading all rows
    q = f"DESCRIBE SELECT * FROM read_parquet('{PACE_GLOB}') LIMIT 1;"
    # run it and fetch as a dataframe
    df = duckdb.sql(q).df()
    # print the column/type table
    print(df.to_string(index=False))
# if DuckDB isn't importable for any reason, fall back to pandas
except Exception as e:
    # note why we fell back
    print(f"(duckdb path failed: {e}; using pandas)")
    # pandas + glob to read a single file
    import pandas as pd, glob
    # find the parquet files
    files = sorted(glob.glob(PACE_GLOB))
    # bail clearly if none found (wrong path / layout)
    if not files:
        raise SystemExit(f"no parquet files at {PACE_GLOB}")
    # read just the first file
    d = pd.read_parquet(files[0])
    # print each column name and dtype, one per line
    for name, dt in d.dtypes.items():
        print(f"{name}\t{dt}")
