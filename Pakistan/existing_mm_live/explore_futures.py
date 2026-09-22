# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# explore_futures.py -- LEARN the futures data before building any arb backtest.
# Read-only DuckDB queries against the parsed store. Safe to run alongside other
# jobs. Answers: what markets exist, which symbols trade as futures, how liquid
# the futures books are, and whether spot vs future prices ever diverge enough to
# arb after costs. Run:  python explore_futures.py
#
# Nothing here is a backtest -- it is reconnaissance. The output tells us whether
# an arb backtest is even worth building.

# paths + duckdb
from pathlib import Path
import duckdb
import pandas as pd

# parsed store (Hive-partitioned by date under each table dir)
# Resolve this filesystem path through the canonical checkout/data configuration.
PARSED = str(_hft_paths.PARSED_ROOT)
con = duckdb.connect()
pd.set_option("display.width", 160, "display.max_columns", 40)

# a recent date glob to keep the recon fast; widen if a table looks empty
TRADES = f"{PARSED}/trades/date=*/*.parquet"
OBSNAP = f"{PARSED}/ob_snapshot/date=*/*.parquet"
OBUPD  = f"{PARSED}/ob_updates/date=*/*.parquet"

print("="*70)
print("1. WHAT MARKETS EXIST IN THE TRADES TABLE (and their trade counts)")
print("="*70)
# every distinct market value + how much it trades -- confirms the futures labels
q1 = f"""
SELECT market, COUNT(*) AS trades, COUNT(DISTINCT symbol) AS symbols,
       COUNT(DISTINCT date) AS days, SUM(qty) AS total_qty
FROM read_parquet('{TRADES}')
GROUP BY market ORDER BY trades DESC
"""
print(con.execute(q1).df().to_string(index=False))

print("\n" + "="*70)
print("2. FUTURES SYMBOLS: naming, count, per-symbol liquidity (DEL_FUT)")
print("="*70)
# delivery-future symbols (BOP-JUL style) -- volume, trade count, days present
q2 = f"""
SELECT symbol, COUNT(*) AS trades, COUNT(DISTINCT date) AS days,
       SUM(qty) AS total_qty, AVG(price) AS avg_px,
       MIN(date) AS first_day, MAX(date) AS last_day
FROM read_parquet('{TRADES}')
WHERE market = 'STOCK_DEL_FUT'
GROUP BY symbol ORDER BY trades DESC LIMIT 30
"""
print(con.execute(q2).df().to_string(index=False))

print("\n" + "="*70)
print("3. CASH-SETTLED FUTURES (STOCK_CS_FUT): do they trade at all?")
print("="*70)
# you flagged CS futures may be dead -- confirm
q3 = f"""
SELECT symbol, COUNT(*) AS trades, COUNT(DISTINCT date) AS days, SUM(qty) AS total_qty
FROM read_parquet('{TRADES}')
WHERE market = 'STOCK_CS_FUT'
GROUP BY symbol ORDER BY trades DESC LIMIT 30
"""
cs = con.execute(q3).df()
print(cs.to_string(index=False) if len(cs) else "  (no STOCK_CS_FUT trades found -- confirms cash-settled is dead)")

print("\n" + "="*70)
print("4. SPOT vs FUTURE side-by-side for one name (BOP) on the latest shared day")
print("="*70)
# find a date where BOTH BOP (spot) and a BOP-* future traded, then compare VWAPs
q4 = f"""
WITH fut AS (
  SELECT date, symbol, SUM(price*qty)/SUM(qty) AS fut_vwap, SUM(qty) AS fut_qty,
         COUNT(*) AS fut_trades
  FROM read_parquet('{TRADES}')
  WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%'
  GROUP BY date, symbol
),
spot AS (
  SELECT date, SUM(price*qty)/SUM(qty) AS spot_vwap, SUM(qty) AS spot_qty
  FROM read_parquet('{TRADES}')
  WHERE market='REG' AND symbol='BOP'
  GROUP BY date
)
SELECT f.date, f.symbol AS future, s.spot_vwap, f.fut_vwap,
       (f.fut_vwap - s.spot_vwap) AS basis,
       (f.fut_vwap - s.spot_vwap)/s.spot_vwap*1e4 AS basis_bps,
       s.spot_qty, f.fut_qty, f.fut_trades
FROM fut f JOIN spot s ON f.date = s.date
ORDER BY f.date DESC LIMIT 20
"""
print(con.execute(q4).df().to_string(index=False))

print("\n" + "="*70)
print("5. FUTURES ORDER-BOOK DEPTH: are the snapshot books usable? (BOP future)")
print("="*70)
# the futures snapshot schema -- levels, agg tags, depth. shows what columns exist
q5 = f"""
SELECT entry_type, COUNT(*) AS rows, COUNT(DISTINCT level) AS n_levels,
       AVG(qty) AS avg_qty, COUNT(DISTINCT symbol) AS symbols
FROM read_parquet('{OBSNAP}')
WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%'
GROUP BY entry_type ORDER BY rows DESC
"""
try:
    print(con.execute(q5).df().to_string(index=False))
except Exception as e:
    print("  snapshot query failed:", e)

print("\n" + "="*70)
print("6. AGG_BID / AGG_OFFER and other entry_type tags in the futures book")
print("="*70)
# you mentioned agg_bid/agg_offer tags -- enumerate all entry_type values
q6 = f"""
SELECT DISTINCT entry_type, entry_type_code
FROM read_parquet('{OBSNAP}')
WHERE market='STOCK_DEL_FUT'
LIMIT 40
"""
try:
    print(con.execute(q6).df().to_string(index=False))
except Exception as e:
    print("  entry_type query failed:", e)

print("\n" + "="*70)
print("7. INTRADAY BASIS: does spot-future spread move enough to arb? (BOP, 1 day)")
print("="*70)
# pick the most recent day BOP future traded; sample basis through the day using
# trades as a rough price proxy (book-level basis comes in the real backtest)
q7 = f"""
WITH d AS (
  SELECT MAX(date) AS dd FROM read_parquet('{TRADES}')
  WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%'
),
fut AS (
  SELECT date_trunc('minute', transact_time) AS min, AVG(price) AS fut_px, SUM(qty) AS q
  FROM read_parquet('{TRADES}'), d
  WHERE market='STOCK_DEL_FUT' AND symbol LIKE 'BOP-%' AND date=d.dd
  GROUP BY 1
),
spot AS (
  SELECT date_trunc('minute', transact_time) AS min, AVG(price) AS spot_px
  FROM read_parquet('{TRADES}'), d
  WHERE market='REG' AND symbol='BOP' AND date=d.dd
  GROUP BY 1
)
SELECT COUNT(*) AS matched_minutes,
       AVG((f.fut_px-s.spot_px)/s.spot_px*1e4) AS mean_basis_bps,
       STDDEV((f.fut_px-s.spot_px)/s.spot_px*1e4) AS std_basis_bps,
       MIN((f.fut_px-s.spot_px)/s.spot_px*1e4) AS min_basis_bps,
       MAX((f.fut_px-s.spot_px)/s.spot_px*1e4) AS max_basis_bps
FROM fut f JOIN spot s ON f.min = s.min
"""
try:
    print(con.execute(q7).df().to_string(index=False))
    print("\n  (wide min-max range + high std = basis MOVES intraday = arb MAY exist.")
    print("   narrow/stable basis = fairly priced = little to arb after costs.)")
except Exception as e:
    print("  intraday basis query failed:", e)

print("\nDONE. Paste the output back and we design the arb backtest around it.")
