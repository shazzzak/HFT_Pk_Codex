"""
dump_codes_2.py

The corrected vocab queries (Q2, Q3, Q5, Q7, Q8, Q8b) with real paths wired in.
Fixes the earlier bug: 'rows' is a reserved word in DuckDB, so every alias is now 'n'.
Q8b parses the market-wide TradingPhaseCode straight out of the raw FIX 'h' messages,
which is the authoritative market-halt source (market_phase='H' == market halt).
"""

# duckdb for the queries
import duckdb

# ---- parsed-data globs (edit if your paths differ) ----
ROOT = "/Users/shazzak/Capital Stake - Parsed"
OB   = f"{ROOT}/ob_snapshot/date=*/*.parquet"
MISC = f"{ROOT}/misc/date=*/*.parquet"

# one reused connection
con = duckdb.connect()

# helper: run a query and print it under a labeled header
def show(title, sql):
    # section header
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)
    # run and print, guarding failures so one bad query doesn't abort the rest
    try:
        print(con.execute(sql).df().to_string(index=False))
    except Exception as e:
        print(f"(query failed: {e})")

# Q2  trading_status vocabulary
show("Q2  ob_snapshot.trading_status vocabulary", f"""
    SELECT trading_status, COUNT(*) AS n
    FROM read_parquet('{OB}')
    GROUP BY trading_status ORDER BY n DESC
""")

# Q3  halt reasons
show("Q3  ob_snapshot.break_reason vocabulary (halt reasons)", f"""
    SELECT break_reason, COUNT(*) AS n
    FROM read_parquet('{OB}')
    WHERE break_reason IS NOT NULL
    GROUP BY break_reason ORDER BY n DESC
""")

# Q5  entry types (MDEntryType 269: bid / ask / auction aggregates)
show("Q5  ob_snapshot.entry_type vocabulary", f"""
    SELECT entry_type_code, entry_type, COUNT(*) AS n
    FROM read_parquet('{OB}')
    GROUP BY 1,2 ORDER BY n DESC
""")

# Q7  phase by hour-of-day (sanity: OPEN in morning, CONTINUOUS midday, CLOSE at end)
show("Q7  phase by hour-of-day (sanity check)", f"""
    SELECT phase, EXTRACT(hour FROM snapshot_time) AS hr, COUNT(*) AS n
    FROM read_parquet('{OB}')
    GROUP BY 1,2 ORDER BY phase, hr
""")

# Q8  misc msg_type vocabulary
show("Q8  misc.msg_type vocabulary ('h' = Trading Session Status)", f"""
    SELECT msg_type, COUNT(*) AS n, COUNT(DISTINCT date) AS days
    FROM read_parquet('{MISC}')
    GROUP BY msg_type ORDER BY n DESC
""")

# Q8b  market-wide phase timeline parsed from raw FIX (authoritative market-halt source)
show("Q8b market phase timeline from raw FIX 8538 ('H' == market halt day)", f"""
    SELECT date,
           regexp_extract(raw, '8538=([A-Z])', 1) AS market_phase,
           COUNT(*) AS n
    FROM read_parquet('{MISC}')
    WHERE msg_type = 'h'
    GROUP BY 1,2 ORDER BY date, market_phase
""")

# done
print("\nDone. Q8b: any date with market_phase='H' is a market-wide halt to exclude.")
