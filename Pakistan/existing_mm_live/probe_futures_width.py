# probe_futures_width.py -- how wide is the ACTIVE-MONTH futures book, in TICKS,
# for EVERY futures root, across many days?
#
# WHY THIS EXISTS. probe_futures_mm.py answers the question for BOP on the single
# most recent date. That is one name on one day. The lean (a 2-tick shift of both
# quotes) only works where the book is wide enough for the shifted quote to stay
# inside the opposite touch -- on a 1-2 tick book it INVERTS capture (A.2 on
# KEL/PIBTL/TPL: +1.64 -> -1.79 bps). So the sweep's per-root result is
# uninterpretable without knowing each root's width first.
#
# Read-only, ~1-2 minutes. Run: caffeinate -is python probe_futures_width.py
from pathlib import Path
import duckdb, pandas as pd

# --- PARSED STORE from config_pk; never a literal in this file ---------------
try:
    # the project's central path module
    import config_pk
    # accept either spelling
    _P = getattr(config_pk, "PARSED_ROOT", None) or getattr(config_pk, "PARSED", None)
    # as a string for the globs
    PARSED = str(_P) if _P else ""
except Exception:
    # not importable from this directory
    PARSED = ""
# current literal only as a fallback
if not PARSED:
    PARSED = "/Users/shazzak/HFT Data/Pakistan/Capital Stake - Parsed"
# fail immediately with the path named, not inside a glob
if not Path(PARSED).is_dir():
    raise SystemExit(f"PARSED store not found: {PARSED}")
# self-describing output
print(f"parsed store: {PARSED}")

# how many of the most recent trading dates to sample
N_DAYS = 20
# the minimum snapshots for a root-day to count (thin days give noise widths)
MIN_SNAPS = 200

con = duckdb.connect(); pd.set_option("display.width", 170, "display.max_columns", 30)
TRADES = f"{PARSED}/trades/date=*/*.parquet"
SNAP = f"{PARSED}/ob_snapshot/date=*/*.parquet"
print(f"sampling the last {N_DAYS} trading dates, min {MIN_SNAPS} snaps per root-day\n")

q = f"""
WITH dates AS (
  -- the most recent N trading dates that have any futures activity
  SELECT DISTINCT date FROM read_parquet('{TRADES}')
  WHERE market='STOCK_DEL_FUT' ORDER BY date DESC LIMIT {N_DAYS}
),
fut AS (
  -- every futures trade on those dates, with the root parsed off the suffix
  SELECT t.date, regexp_extract(t.symbol,'^([A-Z0-9]+)-',1) AS root, t.symbol, t.qty
  FROM read_parquet('{TRADES}') t, dates d
  WHERE t.market='STOCK_DEL_FUT' AND t.date=d.date
),
active AS (
  -- the ACTIVE contract per root per date = the most-traded one that day.
  -- This is what excludes the near-dead back months whose books are nonsense
  -- (BOP-AUG showed a 696-tick spread on 26 snapshots).
  SELECT date, root, symbol FROM (
    SELECT date, root, symbol, SUM(qty) q,
           ROW_NUMBER() OVER (PARTITION BY date, root ORDER BY SUM(qty) DESC) rn
    FROM fut GROUP BY date, root, symbol) WHERE rn=1
),
touch AS (
  -- L1 bid/ask per snapshot message, continuous phase only, active contracts
  SELECT s.date, a.root, s.symbol, s.orig_time,
         MAX(CASE WHEN s.entry_type='BID'   AND s.level=1 THEN s.px END) AS bb,
         MIN(CASE WHEN s.entry_type='OFFER' AND s.level=1 THEN s.px END) AS ba
  FROM read_parquet('{SNAP}') s
  JOIN active a ON s.date=a.date AND s.symbol=a.symbol
  WHERE s.market='STOCK_DEL_FUT' AND s.phase='CONTINUOUS_AUCTION'
  GROUP BY s.date, a.root, s.symbol, s.orig_time
),
ok AS (
  -- two-sided, uncrossed touches only
  SELECT * FROM touch WHERE bb>0 AND ba>0 AND ba>=bb
),
per_day AS (
  -- one width observation per root-day, on days with enough snapshots.
  -- TICK IS 0.01 ON PSX FUTURES -- confirmed by probe_futures_mm.py section 1.
  SELECT date, root, COUNT(*) AS snaps,
         MEDIAN((ba-bb)/0.01) AS med_ticks,
         MEDIAN((ba-bb)/((ba+bb)/2)*1e4) AS med_bps,
         AVG(CASE WHEN (ba-bb)/0.01 >= 3 THEN 1.0 ELSE 0.0 END) AS frac_ge3
  FROM ok GROUP BY date, root HAVING COUNT(*) >= {MIN_SNAPS}
)
SELECT root,
       COUNT(*)                          AS days,
       ROUND(MEDIAN(med_ticks),2)        AS med_spread_ticks,
       ROUND(MIN(med_ticks),2)           AS min_day_ticks,
       ROUND(MAX(med_ticks),2)           AS max_day_ticks,
       ROUND(MEDIAN(med_bps),1)          AS med_spread_bps,
       ROUND(100*MEDIAN(frac_ge3),1)     AS pct_time_ge_3_ticks,
       ROUND(AVG(snaps))                 AS avg_snaps_day
FROM per_day GROUP BY root ORDER BY med_spread_ticks DESC
"""
df = con.execute(q).df()
print(df.to_string(index=False))

# the verdict line per root, using the same rule the spot work established
print("\nVERDICT (the 2-tick lean needs ~3 ticks of room to stay inside the touch):")
# roots wide enough to try the lean
wide = df[df.med_spread_ticks >= 3.0]
# roots that look like the cheap-tick spot names
thin = df[df.med_spread_ticks < 3.0]
print(f"  WIDE ENOUGH ({len(wide)}): {', '.join(wide.root.tolist()) or '-'}")
print(f"  TOO THIN    ({len(thin)}): {', '.join(thin.root.tolist()) or '-'}")
print("\n  Include only the WIDE roots in the lean sweep. A thin root in the mix")
print("  drags the total down for a reason that has nothing to do with the lean.")
