"""
dump_codes.py

Dump the actual code vocabularies your parser wrote, across all four parsed tables,
so we can map them to the PSX FIX spec and build an authoritative halt/continuous mask.

Spec decode (TradingPhaseCode tag 8538, 1st char):
  T=Continuous(KEEP)  S=Starting  O/N=pre-open auctions  B=break(2=Fri lunch)
  H=Temporary Suspension(HALT)  V=resume auction  C=pre-close  A=post-close  E=closed
  2nd char '1' = suspended whole day.   MsgType 'h' in misc = Trading Session Status (market halts).
"""

# duckdb for the queries
import duckdb

# ---- parsed-data globs (edit if your paths differ) ----
ROOT = "/Users/shazzak/Capital Stake - Parsed"
OB   = f"{ROOT}/ob_snapshot/date=*/*.parquet"
MISC = f"{ROOT}/misc/date=*/*.parquet"
TR   = f"{ROOT}/trades/date=*/*.parquet"
OBU  = f"{ROOT}/ob_updates/date=*/*.parquet"

# one connection reused for all queries
con = duckdb.connect()

# helper: run a query, print a labeled section
def show(title, sql):
    # section header
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)
    # run and print (guard so one failure doesn't abort the whole dump)
    try:
        print(con.execute(sql).df().to_string(index=False))
    except Exception as e:
        print(f"(query failed: {e})")

# ---------- ob_snapshot: per-stock trading state ----------

# Q1 phase vocabulary + first char + coverage (THE key one)
show("Q1  ob_snapshot.phase vocabulary (KEEP first-char 'T'; 'H'=halt)", f"""
    SELECT phase, LEFT(phase,1) AS phase0, COUNT(*) AS rows,
           COUNT(DISTINCT symbol||'|'||CAST(date AS VARCHAR)) AS sym_days
    FROM read_parquet('{OB}') GROUP BY phase ORDER BY rows DESC
""")

# Q2 trading_status vocabulary
show("Q2  ob_snapshot.trading_status vocabulary", f"""
    SELECT trading_status, COUNT(*) rows
    FROM read_parquet('{OB}') GROUP BY trading_status ORDER BY rows DESC
""")

# Q3 halt reasons
show("Q3  ob_snapshot.break_reason vocabulary (halt reasons)", f"""
    SELECT break_reason, COUNT(*) rows
    FROM read_parquet('{OB}') WHERE break_reason IS NOT NULL
    GROUP BY break_reason ORDER BY rows DESC
""")

# Q4 whole-day suspensions per symbol
show("Q4  suspended_all_day days per symbol", f"""
    SELECT symbol, COUNT(DISTINCT date) suspended_days
    FROM read_parquet('{OB}') WHERE suspended_all_day
    GROUP BY symbol ORDER BY suspended_days DESC
""")

# Q5 entry types (MDEntryType 269: bid/ask/auction)
show("Q5  ob_snapshot.entry_type vocabulary", f"""
    SELECT entry_type_code, entry_type, COUNT(*) rows
    FROM read_parquet('{OB}') GROUP BY 1,2 ORDER BY rows DESC
""")

# Q6 individual halt windows (phase H): which symbol-days, how long
show("Q6  individual HALT windows (phase starts 'H')", f"""
    SELECT symbol, date, COUNT(DISTINCT snapshot_time) halt_snapshots
    FROM read_parquet('{OB}') WHERE LEFT(phase,1) = 'H'
    GROUP BY 1,2 ORDER BY halt_snapshots DESC LIMIT 50
""")

# Q7 sanity: phase-first-char by hour-of-day
show("Q7  phase (first char) by hour-of-day  (validate: O morning, T midday, C/E close)", f"""
    SELECT LEFT(phase,1) phase0, EXTRACT(hour FROM snapshot_time) hr, COUNT(*) rows
    FROM read_parquet('{OB}') GROUP BY 1,2 ORDER BY phase0, hr
""")

# ---------- misc: market-wide session status ----------

# Q8 msg_type vocabulary
show("Q8  misc.msg_type vocabulary ('h' = Trading Session Status = market halts)", f"""
    SELECT msg_type, COUNT(*) rows, COUNT(DISTINCT date) days
    FROM read_parquet('{MISC}') GROUP BY msg_type ORDER BY rows DESC
""")

# Q9 sample raw session-status messages
show("Q9  sample raw Trading Session Status messages (adjust msg_type from Q8)", f"""
    SELECT date, msg_type, LEFT(raw, 300) raw_snippet
    FROM read_parquet('{MISC}') WHERE msg_type = 'h' LIMIT 20
""")

# ---------- bonus: exec/side codes ----------

# Q10 trade exec codes
show("Q10 trades exec/aggressor codes", f"""
    SELECT exec_type_code, exec_type, exec_inst, aggressor_side, COUNT(*) rows
    FROM read_parquet('{TR}') GROUP BY 1,2,3,4 ORDER BY rows DESC LIMIT 30
""")

# Q11 book-update event/side codes
show("Q11 ob_updates event/side codes", f"""
    SELECT event, exec_type_code, exec_type, side_code, side, COUNT(*) rows
    FROM read_parquet('{OBU}') GROUP BY 1,2,3,4,5 ORDER BY rows DESC LIMIT 40
""")

# ---------- clock alignment helper (for the phase->feature_store interval join) ----------

# print ob_snapshot time range so we can align it to feature_store ts_exch (ms)
show("CLOCK  ob_snapshot snapshot_time range (for aligning to feature_store ts_exch)", f"""
    SELECT MIN(snapshot_time) min_ts, MAX(snapshot_time) max_ts
    FROM read_parquet('{OB.replace("date=*","date=2026-01-09")}')
""")

# done
print("\nDone. Paste Q1, Q2, Q3, Q8 back — those pin the halt mask.")
