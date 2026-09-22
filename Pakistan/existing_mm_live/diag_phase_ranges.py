# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# diag_phase_ranges.py -- show each snapshot PHASE's time span per symbol, so we
# can see (a) whether an after-hours / close-auction phase exists after the main
# continuous session, and (b) whether CONTINUOUS_AUCTION snapshot timestamps
# themselves bleed into that tail (which is why bounding trades by
# max(continuous time) failed to remove the post-close cliff-tail).
#
# Run from anywhere:  python3 diag_phase_ranges.py

# duckdb queries the parquet store in place
import duckdb
# pandas for display
import pandas as pd

# wide display
pd.set_option("display.width", 200)

# the snapshot glob (hive layout: table before date=)
# Resolve this filesystem path through the canonical checkout/data configuration.
GLOB = str(_hft_paths.PARSED_ROOT / 'ob_snapshot/date=*/*.parquet')
# a couple of symbols to inspect
SYMS = ["TRG", "PPL"]

# for each symbol, list phases with their row count and time span (minutes into
# the day), ordered by when they start -- so an after-hours tail is obvious
for sym in SYMS:
    # per-phase min/max time + count for this symbol, one representative date
    q = f"""
    WITH one_day AS (
      -- pick a single busy date so the spans are one session, not stacked
      SELECT date
      FROM read_parquet('{GLOB}', hive_partitioning=1)
      WHERE symbol = '{sym}'
      GROUP BY date
      ORDER BY COUNT(*) DESC
      LIMIT 1
    ),
    s AS (
      SELECT phase, orig_time
      FROM read_parquet('{GLOB}', hive_partitioning=1)
      WHERE symbol = '{sym}'
        AND date = (SELECT date FROM one_day)
    )
    SELECT phase,
           COUNT(*)                                   AS n_rows,
           MIN(orig_time)                             AS first_time,
           MAX(orig_time)                             AS last_time
    FROM s
    GROUP BY phase
    ORDER BY first_time
    """
    # print the per-phase table
    print(f"\n=== {sym}: snapshot phases on its busiest day ===")
    # run + show
    print(duckdb.sql(q).df().to_string())
    # explicit hint about the overlap we are hunting
    print("  -> if CONTINUOUS_AUCTION.last_time is close to an AFTER_HOUR_TRADING")
    print("     or CLOSE_CALL_AUCTION last_time, the continuous phase bleeds into")
    print("     the tail and max(continuous) is the WRONG session-close bound.")
