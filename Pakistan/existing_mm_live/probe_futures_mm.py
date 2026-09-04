# probe_futures_mm.py -- the mechanics the futures MM engine must get right,
# that may DIFFER from spot. Read-only. Answers, per the active-month futures:
#   1. tick size (spot is 0.01; futures may differ -> half-spread/edge math)
#   2. lot size in the actual data (you said 500; confirm the traded qty granularity)
#   3. the ob_updates stream: same event vocabulary as spot? (build_events reuse)
#   4. spread distribution in bps (is there enough spread to make a market?)
#   5. which contract is the "active month" on any given date (front-month roll)
from pathlib import Path
import duckdb, pandas as pd, numpy as np
PARSED = "/Users/shazzak/Capital Stake - Parsed"
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

print("\n5. ACTIVE-MONTH ROLL: which BOP contract is most liquid each month?")
q=f"""
SELECT strftime(date,'%Y-%m') AS mon, symbol, SUM(qty) AS qty, COUNT(*) AS trades
FROM read_parquet('{TRADES}') WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%'
GROUP BY 1,2 QUALIFY ROW_NUMBER() OVER (PARTITION BY mon ORDER BY SUM(qty) DESC)=1
ORDER BY mon
"""
print(con.execute(q).df().to_string(index=False))
print("\nDONE -- these five answers set the futures MM engine's tick/lot/spread params.")
