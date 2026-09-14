# futures_trade_intensity.py -- how much would we actually TRADE on the active
# futures? trades/min tells us the fill opportunity (and thus P&L potential).
# Also compute the proprietary DFC round-trip fee in bps so we know the hurdle.
# Read-only. Run:  python futures_trade_intensity.py
from pathlib import Path
import duckdb, pandas as pd, numpy as np
# ---- PATHS: from config_pk, never a literal in this file -------------------
# The literals that were here pointed at the OLD store location. When the data
# moved under "~/HFT Data/Pakistan/" every query in this file started failing
# with IOException. A path literal is a defect: it is valid syntax, so nothing
# warns you, and it fails deep into a run instead of at the top.
# config_pk exposes PARSED_ROOT and RESULTS_ROOT as Paths.
from config_pk import PARSED_ROOT, RESULTS_ROOT
# the parsed store as a string, for the f-string globs below
PARSED = str(PARSED_ROOT)
# fail here, with the path named, rather than inside the first query
if not PARSED_ROOT.is_dir():
    raise SystemExit(f"PARSED store not found: {PARSED}")
# say which store this run read
print(f"parsed store: {PARSED}\n")
con=duckdb.connect(); pd.set_option("display.width",170,"display.max_columns",30)
TRADES=f"{PARSED}/trades/date=*/*.parquet"

# ---- proprietary DFC fee, computed in bps, squared-up (no delivery) ----
# PSX Laga/CCPF: 0.93809 PKR per 100,000 traded value = 0.00094% per side
LAGA_PCT_PER_SIDE = 0.93809/100000.0            # fraction of notional, per side
# CDC per-share handling ONLY on delivery -> we square up -> ZERO
# round-trip (buy+sell) exchange levy in bps:
rt_bps = LAGA_PCT_PER_SIDE*2*1e4
print(f"PROPRIETARY DFC round-trip exchange fee: {rt_bps:.4f} bps "
      f"(Laga {LAGA_PCT_PER_SIDE*1e4:.4f} bps/side x2; CDC=0 since squared up)")
print(f"  vs spot TREC fee 1.554 bps round-trip -> futures are ~{1.554/rt_bps:.0f}x CHEAPER per trade\n")

print("="*70)
print("TRADES PER MINUTE on the active-month BOP future, by month")
print("="*70)
# for each month, the dominant BOP contract, its trades, and trades/continuous-min.
# continuous session ~ derive from trade timestamps (first..last) per day.
q=f"""
WITH fut AS (
  SELECT date, strftime(date,'%Y-%m') AS mon, symbol, transact_time, qty
  FROM read_parquet('{TRADES}')
  WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%'
),
active AS (  -- the most-traded BOP contract each month
  SELECT mon, symbol FROM (
    SELECT mon, symbol, SUM(qty) q,
           ROW_NUMBER() OVER (PARTITION BY mon ORDER BY SUM(qty) DESC) rn
    FROM fut GROUP BY mon, symbol) WHERE rn=1
),
daily AS (
  SELECT f.mon, f.symbol, f.date,
         COUNT(*) AS trades,
         (epoch(MAX(f.transact_time))-epoch(MIN(f.transact_time)))/60.0 AS span_min
  FROM fut f JOIN active a ON f.mon=a.mon AND f.symbol=a.symbol
  GROUP BY f.mon, f.symbol, f.date
)
SELECT mon, any_value(symbol) AS active_contract,
       COUNT(*) AS days,
       AVG(trades) AS avg_trades_per_day,
       AVG(trades/NULLIF(span_min,0)) AS avg_trades_per_min,
       MAX(trades/NULLIF(span_min,0)) AS peak_day_trades_per_min
FROM daily GROUP BY mon ORDER BY mon
"""
print(con.execute(q).df().round(2).to_string(index=False))

print("\n"+"="*70)
print("TRADES/MIN across the TOP-20 futures names (most recent full month)")
print("="*70)
# broaden: for the latest month, top-20 active futures by volume + their trades/min
q2=f"""
WITH d AS (SELECT strftime(MAX(date),'%Y-%m') AS mon FROM read_parquet('{TRADES}')
           WHERE market='STOCK_DEL_FUT'),
fut AS (
  SELECT strftime(date,'%Y-%m') AS mon,
         regexp_extract(symbol,'^([A-Z]+)-',1) AS root, symbol, date, transact_time, qty
  FROM read_parquet('{TRADES}') WHERE market='STOCK_DEL_FUT'),
active AS (
  SELECT f.root, f.symbol FROM fut f, d WHERE f.mon=d.mon
  QUALIFY ROW_NUMBER() OVER (PARTITION BY f.root ORDER BY SUM(qty) OVER (PARTITION BY f.symbol)) =1
),
daily AS (
  SELECT f.root, f.symbol, f.date, COUNT(*) trades,
         (epoch(MAX(f.transact_time))-epoch(MIN(f.transact_time)))/60.0 span_min
  FROM fut f, d WHERE f.mon=d.mon GROUP BY f.root,f.symbol,f.date)
SELECT root, any_value(symbol) AS contract, COUNT(*) days,
       AVG(trades) avg_trades_day, AVG(trades/NULLIF(span_min,0)) trades_per_min
FROM daily GROUP BY root ORDER BY avg_trades_day DESC LIMIT 20
"""
try:
    print(con.execute(q2).df().round(2).to_string(index=False))
except Exception as e:
    print("top-20 query issue:", e)
print("\nREAD: trades/min ~ the tape rate. Our passive fills are a FRACTION of this")
print("(we only fill when someone crosses to us). Spot BOP for comparison ~ see prior runs.")
