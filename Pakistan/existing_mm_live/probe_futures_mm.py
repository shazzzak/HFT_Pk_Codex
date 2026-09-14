# probe_futures_mm.py -- the mechanics the futures MM engine must get right,
# that may DIFFER from spot. Read-only. Answers, per the active-month futures:
#   1. tick size (spot is 0.01; futures may differ -> half-spread/edge math)
#   2. lot size in the actual data (you said 500; confirm the traded qty granularity)
#   3. the ob_updates stream: same event vocabulary as spot? (build_events reuse)
#   4. spread distribution in bps (is there enough spread to make a market?)
#   5. which contract is the "active month" on any given date (front-month roll)
#
# 2026-09-15: the hardcoded PARSED path was stale -- the store moved under
# "~/HFT Data/Pakistan/" and every query raised IOException "No files found".
# Now resolved from config_pk, with the current literal only as a fallback.
from pathlib import Path
import duckdb, pandas as pd, numpy as np

# --- PARSED STORE: one source of truth, never a literal in this file ---------
try:
    # the project's central path module (same one mm_harness/config use)
    import config_pk
    # accept either spelling the module may expose
    _P = getattr(config_pk, "PARSED_ROOT", None) or getattr(config_pk, "PARSED", None)
    # as a string for the f-string globs below
    PARSED = str(_P) if _P else ""
except Exception:
    # config_pk not importable from this working directory
    PARSED = ""
# fallback to the CURRENT literal only if config_pk did not supply one
if not PARSED:
    PARSED = "/Users/shazzak/HFT Data/Pakistan/Capital Stake - Parsed"
# fail loudly and immediately rather than inside a DuckDB glob 40 lines later
if not Path(PARSED).is_dir():
    raise SystemExit(f"PARSED store not found: {PARSED}\n"
                     f"  set PARSED_ROOT in config_pk.py to the real location.")
# say which store this run read, so the output is self-describing
print(f"parsed store: {PARSED}\n")

con = duckdb.connect(); pd.set_option("display.width",160,"display.max_columns",30)
TRADES=f"{PARSED}/trades/date=*/*.parquet"; OBUPD=f"{PARSED}/ob_updates/date=*/*.parquet"

print("1. TICK SIZE: distinct price granularity in BOP futures vs BOP spot")
q=f"""
WITH fut AS (SELECT DISTINCT price FROM read_parquet('{TRADES}')
             WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%')
SELECT 'BOP_future' AS inst, MIN(ABS(a.price-b.price)) AS min_tick
FROM fut a, fut b WHERE a.price>b.price
"""
print(con.execute(q).df().to_string(index=False))

print("\n2. LOT SIZE: is traded qty always a multiple of 500? (BOP future)")
q=f"""
SELECT MIN(qty) AS min_qty, MAX(qty) AS max_qty,
       COUNT(*) AS n, SUM(CASE WHEN qty % 500 = 0 THEN 1 ELSE 0 END) AS mult_of_500
FROM read_parquet('{TRADES}') WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%'
"""
print(con.execute(q).df().to_string(index=False))

print("\n3. OB_UPDATES event vocabulary: futures vs spot (can we reuse build_events?)")
q=f"""
SELECT market, event, COUNT(*) AS n
FROM read_parquet('{OBUPD}')
WHERE (market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%') OR (market='REG' AND symbol='BOP')
GROUP BY market, event ORDER BY market, n DESC
"""
print(con.execute(q).df().to_string(index=False))

print("\n4. SPREAD in bps: is there room to make a market? (BOP active future, recent)")
q=f"""
WITH d AS (SELECT MAX(date) dd FROM read_parquet('{TRADES}')
           WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%'),
snap AS (
  SELECT s.symbol, s.orig_time,
         MAX(CASE WHEN entry_type='BID' AND level=1 THEN px END) AS bb,
         MIN(CASE WHEN entry_type='OFFER' AND level=1 THEN px END) AS ba
  FROM read_parquet('{PARSED}/ob_snapshot/date=*/*.parquet') s, d
  WHERE s.market='STOCK_DEL_FUT' AND s.symbol LIKE 'BOP-%' AND s.date=d.dd
        AND s.phase='CONTINUOUS_AUCTION'
  GROUP BY s.symbol, s.orig_time
)
SELECT symbol, COUNT(*) AS snaps,
       AVG((ba-bb)/((ba+bb)/2)*1e4) AS mean_spread_bps,
       MEDIAN((ba-bb)/((ba+bb)/2)*1e4) AS med_spread_bps
FROM snap WHERE bb>0 AND ba>0 AND ba>=bb GROUP BY symbol ORDER BY snaps DESC
"""
print(con.execute(q).df().to_string(index=False))

# --- 4b. THE DECISIVE NUMBER FOR THE LEAN -----------------------------------
# The lean shifts BOTH quotes 2 ticks. On a book only 1-2 ticks wide that walks
# the quote through the opposite side and INVERTS capture (A.2 on KEL/PIBTL/TPL:
# capture +1.64 -> -1.79 bps). bps is not the unit that decides this -- TICKS is.
# This reports the spread in WHOLE TICKS, and the share of time the book is wide
# enough for a 2-tick shift to still leave the quote inside the opposite touch.
print("\n4b. SPREAD in TICKS: is there room for a 2-TICK LEAN? (the A.2 test)")
q=f"""
WITH d AS (SELECT MAX(date) dd FROM read_parquet('{TRADES}')
           WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%'),
tick AS (
  -- the minimum observed price increment = the tick, measured not assumed
  SELECT MIN(ABS(a.price-b.price)) AS t FROM
    (SELECT DISTINCT price FROM read_parquet('{TRADES}')
     WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%') a,
    (SELECT DISTINCT price FROM read_parquet('{TRADES}')
     WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%') b
  WHERE a.price>b.price
),
snap AS (
  SELECT s.symbol, s.orig_time,
         MAX(CASE WHEN entry_type='BID' AND level=1 THEN px END) AS bb,
         MIN(CASE WHEN entry_type='OFFER' AND level=1 THEN px END) AS ba
  FROM read_parquet('{PARSED}/ob_snapshot/date=*/*.parquet') s, d
  WHERE s.market='STOCK_DEL_FUT' AND s.symbol LIKE 'BOP-%' AND s.date=d.dd
        AND s.phase='CONTINUOUS_AUCTION'
  GROUP BY s.symbol, s.orig_time
)
SELECT snap.symbol, COUNT(*) AS snaps,
       ROUND(MEDIAN((ba-bb)/tick.t),2)  AS med_spread_ticks,
       ROUND(AVG((ba-bb)/tick.t),2)     AS mean_spread_ticks,
       ROUND(100.0*AVG(CASE WHEN (ba-bb)/tick.t >= 3 THEN 1 ELSE 0 END),1) AS pct_ge_3_ticks,
       ROUND(100.0*AVG(CASE WHEN (ba-bb)/tick.t >= 5 THEN 1 ELSE 0 END),1) AS pct_ge_5_ticks
FROM snap, tick WHERE bb>0 AND ba>0 AND ba>=bb
GROUP BY snap.symbol ORDER BY snaps DESC
"""
print(con.execute(q).df().to_string(index=False))
print("  READ: a 2-tick lean needs the book at least ~3 ticks wide to stay inside")
print("  the opposite touch. If med_spread_ticks is 1-2, the lean will invert")
print("  capture exactly as it did on the cheap-tick spot names -- do not run the sweep.")

print("\n5. ACTIVE-MONTH ROLL: which BOP contract is most liquid each month?")
q=f"""
SELECT strftime(date,'%Y-%m') AS mon, symbol, SUM(qty) AS qty, COUNT(*) AS trades
FROM read_parquet('{TRADES}') WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%'
GROUP BY 1,2 QUALIFY ROW_NUMBER() OVER (PARTITION BY mon ORDER BY SUM(qty) DESC)=1
ORDER BY mon
"""
print(con.execute(q).df().to_string(index=False))
print("\nDONE -- these five answers set the futures MM engine's tick/lot/spread params.")
