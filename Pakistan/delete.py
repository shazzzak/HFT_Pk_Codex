
import duckdb, config_pk
g = str(config_pk.PARSED_ROOT / 'trades' / '*' / '*.parquet')
q = duckdb.connect().execute(f'''
  SELECT date, COUNT(*) AS trades,
         SUM(CASE WHEN resting_order_id IS NOT NULL
                   AND CAST(resting_order_id AS VARCHAR) <> '' THEN 1 ELSE 0 END) AS with_id
  FROM read_parquet('{g}', hive_partitioning=1)
  GROUP BY date ORDER BY date''').df()
q['pct'] = (q.with_id / q.trades * 100).round(2)
print(q.to_string(index=False))
print(f'\nOVERALL {q.with_id.sum()/q.trades.sum()*100:.2f}% of {q.trades.sum():,} trades carry a resting order id')
