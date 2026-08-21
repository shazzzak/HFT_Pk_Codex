# futures_top20.py -- FIX for the broken top-20 query: trades/min across all
# futures names in the most recent full month, to pick the tradeable universe.
# Read-only. Run:  python futures_top20.py
from pathlib import Path
import duckdb, pandas as pd
PARSED="/Users/shazzak/Capital Stake - Parsed"
con=duckdb.connect(); pd.set_option("display.width",170,"display.max_columns",30)
TRADES=f"{PARSED}/trades/date=*/*.parquet"

# step 1: the most recent month present in the futures data
mon = con.execute(f"""
  SELECT strftime(MAX(date),'%Y-%m') FROM read_parquet('{TRADES}')
  WHERE market='STOCK_DEL_FUT'
""").fetchone()[0]
print(f"most recent futures month: {mon}\n")

# step 2: per root symbol, the single most-liquid contract that month, and its
# per-day trade count + trades/continuous-minute (span = first..last trade).
q=f"""
WITH fut AS (
  SELECT regexp_extract(symbol,'^([A-Z]+)-',1) AS root, symbol, date,
         transact_time, qty
  FROM read_parquet('{TRADES}')
  WHERE market='STOCK_DEL_FUT' AND strftime(date,'%Y-%m')='{mon}'
),
vol AS (  -- pick each root's dominant contract by total qty
  SELECT root, symbol,
         ROW_NUMBER() OVER (PARTITION BY root ORDER BY SUM(qty) DESC) AS rn
  FROM fut GROUP BY root, symbol
),
active AS (SELECT root, symbol FROM vol WHERE rn=1),
daily AS (
  SELECT a.root, a.symbol, f.date,
         COUNT(*) AS n_trades,
         (epoch(MAX(f.transact_time))-epoch(MIN(f.transact_time)))/60.0 AS span_min,
         SUM(f.qty) AS day_qty
  FROM fut f JOIN active a ON f.symbol=a.symbol
  GROUP BY a.root, a.symbol, f.date
)
SELECT root,
       any_value(symbol) AS contract,
       COUNT(*) AS n_days,
       ROUND(AVG(n_trades),0) AS avg_trades_day,
       ROUND(AVG(n_trades/NULLIF(span_min,0)),2) AS trades_per_min,
       ROUND(AVG(day_qty),0) AS avg_qty_day,
       ROUND(AVG(day_qty)/500.0,0) AS avg_lots_day
FROM daily
GROUP BY root
ORDER BY avg_trades_day DESC
LIMIT 25
"""
df = con.execute(q).df()
print(df.to_string(index=False))
print(f"\n{len(df)} names shown. trades_per_min = tape rate; our passive fills are")
print("a fraction of it. avg_lots_day = daily volume in 500-share contracts.")
print("Names with trades_per_min >~5 and many lots/day are the MM candidates.")
