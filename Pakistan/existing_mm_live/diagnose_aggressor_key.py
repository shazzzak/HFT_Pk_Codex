# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# diagnose_aggressor_key.py -- figure out which field identifies an aggressor
# order, so the sweep-depth grouping is correct. Run from anywhere.
#
# We suspect `initiator` is NOT the per-order aggressor id (the sweep grouping
# collapsed ~2000 trades per group). This inspects the real values so the fix
# keys on the right field.

# duckdb queries parquet in place
import duckdb
# pandas just for display width
import pandas as pd

# show wide tables without wrapping
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

# the parsed trades glob -- hive layout is <ROOT>/trades/date=*/*.parquet
# (table name comes BEFORE the date= partition, not after)
# Resolve this filesystem path through the canonical checkout/data configuration.
GLOB = str(_hft_paths.PARSED_ROOT / 'trades/date=*/*.parquet')
# the symbol to inspect (liquid spot name)
SYM = "UBL"

# 1) raw sample: look at 40 consecutive trades and eyeball the ref fields
sample = duckdb.sql(f"""
    SELECT transact_time, aggressor_side, initiator,
           buy_ref, sell_ref, resting_ref, price, qty
    FROM read_parquet('{GLOB}')
    WHERE symbol = '{SYM}'
    ORDER BY transact_time
    LIMIT 40
""").df()
# print the raw sample
print("=== 40 consecutive trades (eyeball which ref stays constant in a sweep) ===")
print(sample.to_string())

# 2) cardinality: how many distinct values each candidate key takes vs row count
card = duckdb.sql(f"""
    SELECT
      COUNT(*)                                             AS n_rows,
      COUNT(DISTINCT initiator)                            AS d_initiator,
      COUNT(DISTINCT buy_ref)                              AS d_buy_ref,
      COUNT(DISTINCT sell_ref)                             AS d_sell_ref,
      COUNT(DISTINCT resting_ref)                          AS d_resting_ref,
      SUM(CASE WHEN initiator IS NULL
               OR CAST(initiator AS VARCHAR) IN ('','None') THEN 1 ELSE 0 END)
                                                           AS initiator_null
    FROM read_parquet('{GLOB}')
    WHERE symbol = '{SYM}'
""").df()
# print the cardinality summary
print("\n=== cardinality of candidate keys (a per-ORDER key should have MANY "
      "distinct values, ~ n_rows / avg_fills_per_sweep) ===")
print(card.to_string())

# 3) fills-per-key for the by-side aggressor ref (buy_ref for buy-aggressor,
#    sell_ref for sell-aggressor) -- the hypothesised correct key. A good key
#    gives a SMALL mean (a few fills per sweep), not thousands.
byside = duckdb.sql(f"""
    WITH t AS (
      SELECT CASE WHEN UPPER(CAST(aggressor_side AS VARCHAR)) LIKE 'B%'
                  THEN buy_ref ELSE sell_ref END AS agg_key
      FROM read_parquet('{GLOB}')
      WHERE symbol = '{SYM}'
    ),
    g AS (SELECT agg_key, COUNT(*) AS fills FROM t GROUP BY agg_key)
    SELECT COUNT(*)          AS n_aggressor_orders,
           AVG(fills)        AS mean_fills,
           MEDIAN(fills)     AS median_fills,
           MAX(fills)        AS max_fills,
           QUANTILE_CONT(fills, 0.95) AS p95_fills
    FROM g
""").df()
# print the by-side ref grouping stats
print("\n=== by-side aggressor ref (buy_ref/sell_ref) grouping -- the hypothesised "
      "correct key ===")
print(byside.to_string())
# interpretation hint
print("\nREAD: if by-side ref gives mean_fills of a few (say 1-10) and MANY "
      "aggressor orders, THAT is the right key. If initiator has few distinct "
      "values / high null, it is NOT the per-order id and must be dropped.")
