"""
feature_store_wiring.py

Wires the decile-markout pipeline to your ACTUAL data layout:
  Capital Stake - Results/feature_store/{SYMBOL}/date={DATE}.parquet

Provides:
  * COLS               -- logical role -> real feature_store column name
  * enumerate_partitions() -- yields (symbol, date, file_path) for all 414 symbol-days
  * verify_markout_boundary() -- confirms 5s markouts are NULL at each day's close (hygiene)
  * detect_ts_units()  -- infers whether ts_exch is ns / us / ms (needed later)

Run it directly to enumerate partitions and sanity-check a few days.
"""

# os for path handling, glob for partition discovery
import os, glob

# ---------------- CONFIG ----------------

# root of the row-level feature store
FEATURE_STORE_ROOT = "/Users/shazzak/Capital Stake - Results/feature_store"

# logical role -> REAL column name in feature_store (this is the corrected mapping)
COLS = {
    "symbol":         "symbol",               # symbol (also encoded in the folder)
    "date":           "date",                 # day (VARCHAR), also in the filename
    "ts":             "ts_exch",              # exchange timestamp (BIGINT; units detected below)
    "mid":            "mid",                  # mid price
    "markout_5s":     "markout_5000ms_bps",   # the 5s label (NOT 'markout_5s')
    "markout_1s":     "markout_1000ms_bps",   # 1s label
    "markout_30s":    "markout_30000ms_bps",  # 30s label
    "ms_since_trade": "time_since_trade_ms",  # trade-conditioning column (NOT 'ms_since_trade')
    "obi_1":          "obi_1",                # feature
    "micro_dev_bps":  "micro_dev_bps",        # feature
}

# 5s window in milliseconds (for the boundary check)
MARKOUT_WINDOW_MS = 5000


# ---------------- PARTITION ENUMERATION (symbol-first layout) ----------------

def enumerate_partitions(root=FEATURE_STORE_ROOT):
    # glob every per-day file under every symbol folder: feature_store/<SYMBOL>/date=<DATE>.parquet
    files = sorted(glob.glob(os.path.join(root, "*", "date=*.parquet")))
    # collected (symbol, date, path) tuples
    parts = []
    # parse each path
    for f in files:
        # symbol is the parent directory name (e.g. .../feature_store/UBL/date=... -> 'UBL')
        sym = os.path.basename(os.path.dirname(f))
        # filename like 'date=2026-01-09.parquet'
        base = os.path.basename(f)
        # strip the 'date=' prefix and the '.parquet' suffix to get the date string
        dt = base[len("date="):-len(".parquet")]
        # record it
        parts.append((sym, dt, f))
    # hand back all symbol-days (expected: 2 * 207 = 414)
    return parts


# ---------------- BOUNDARY HYGIENE CHECK ----------------

def verify_markout_boundary(sample_paths, tail_rows=2000):
    # import duckdb lazily
    import duckdb
    # one connection for all checks
    con = duckdb.connect()
    # real column names
    mk = COLS["markout_5s"]
    ts = COLS["ts"]
    # header
    print(f"\nBoundary check on {mk} (last {tail_rows} rows of each sampled day, ordered by {ts}):")
    # check each sampled file
    for path in sample_paths:
        # pull the tail rows by timestamp and measure how many carry a non-null markout
        q = f"""
        WITH ordered AS (
            SELECT {mk} AS mk_val
            FROM read_parquet('{path}')
            ORDER BY {ts} DESC          -- newest first = end of the session
            LIMIT {tail_rows}
        )
        SELECT
            COUNT(*)                                            AS n_tail,
            SUM(CASE WHEN mk_val IS NULL THEN 1 ELSE 0 END)     AS n_null,
            AVG(CASE WHEN mk_val IS NULL THEN 1.0 ELSE 0.0 END) AS frac_null,
            AVG(ABS(mk_val))                                    AS mean_abs_nonnull
        FROM ordered
        """
        # run it
        r = con.execute(q).df().iloc[0]
        # short day label from the path
        day = os.path.basename(path)
        # interpret: high frac_null at the tail = correct hygiene; ~0 null with real values = investigate
        flag = "OK (nulled near close)" if r["frac_null"] > 0.5 else "INVESTIGATE (tail markouts populated)"
        # report
        print(f"  {day:35s} frac_null@tail={r['frac_null']:.2f}  "
              f"mean|mk| of non-null tail={r['mean_abs_nonnull']:.2f}bps  -> {flag}")


# ---------------- ts_exch UNIT DETECTION ----------------

def detect_ts_units(sample_path):
    # import duckdb lazily
    import duckdb
    # real ts column
    ts = COLS["ts"]
    # span of ts across one trading day tells us the units
    span = duckdb.connect().execute(
        f"SELECT MAX({ts}) - MIN({ts}) AS span FROM read_parquet('{sample_path}')"
    ).df().iloc[0]["span"]
    # a PSX session is ~4-6 hours; map the observed integer span to a unit
    # ~2e13 -> nanoseconds, ~2e10 -> microseconds, ~2e7 -> milliseconds, ~2e4 -> seconds
    guess = ("nanoseconds"  if span > 1e12 else
             "microseconds" if span > 1e9  else
             "milliseconds" if span > 1e6  else
             "seconds")
    # report the raw span and the inference
    print(f"\nts_exch span over one day = {span:,}  ->  units look like {guess}")
    # return the guess for downstream use
    return guess


# ---------------- MAIN ----------------

def main():
    # enumerate all symbol-days
    parts = enumerate_partitions()
    # report the count and a couple of examples
    print(f"enumerated {len(parts)} symbol-day partitions")
    # show the first and last few
    for sym, dt, f in parts[:2] + parts[-2:]:
        print(f"  {sym}  {dt}  {f}")
    # sample a few files across the sample for the hygiene + units checks
    if parts:
        # take an early, middle, and late day
        idxs = [0, len(parts) // 2, len(parts) - 1]
        # collect their paths
        sample_paths = [parts[i][2] for i in idxs]
        # detect timestamp units from the first sample
        detect_ts_units(sample_paths[0])
        # verify markout boundary hygiene on the samples
        verify_markout_boundary(sample_paths)


# entry point
if __name__ == "__main__":
    # run it
    main()
