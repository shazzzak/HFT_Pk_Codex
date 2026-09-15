# probe_futures_width.py -- how wide is the ACTIVE-MONTH futures book, and what
# does CROSSING it cost, for every futures root, across many days?
#
# WHY THIS EXISTS. Two questions, one measurement:
#   (a) MARKET MAKING. The lean (a 2-tick shift of both quotes) only works where
#       the book is wide enough for the shifted quote to stay inside the opposite
#       touch. On a 1-2 tick book it INVERTS capture (A.2 on KEL/PIBTL/TPL:
#       +1.64 -> -1.79 bps).
#   (b) HEDGING. The futures FEE is 0.19 bps round trip -- cheap. But the fee is
#       not the cost. Crossing the quoted spread is. If the active-month book is
#       12 bps wide, putting a hedge on and taking it off costs 12 bps, which is
#       4.5x the entire 2.67 bps gross edge of the share book it is protecting.
#
# WHAT CHANGED IN THIS VERSION (and why the previous numbers were not trustworthy)
#   1. TRADE COUNTS. v1 reported avg_snaps_day -- QUOTE MESSAGES, not trades. A
#      dead contract with one flickering market maker produces thousands of quote
#      messages a day and almost no trades, and v1 could not tell that apart from
#      a real market. Now every row carries the selected contract's trade count
#      and traded volume, plus what share of the root's day volume it was.
#   2. DEAD-TIME WEIGHTING. v1 took the median over EVERY continuous-auction
#      snapshot, so a contract quoted 5% wide all session and 0.3% wide in the
#      twenty minutes it actually traded reported 5%. Now the width is ALSO
#      measured in-session only (between the day's first and last trade in that
#      contract), and p25 is reported so the tight end of the distribution is
#      visible. If med_ticks_all and med_ticks_insess diverge, v1 was inflated.
#   3. THE VERDICT GATE WAS WRONG. v1 gated on med_spread_ticks >= 3.0, which
#      passed KOSM/AGHA/KEL -- names at 3 ticks or more only ~48% of the time,
#      and KEL is on the A.2 cheap-tick blacklist. The gate is now the FRACTION
#      of time at >= 3 ticks, which is the quantity that actually matters.
#   4. THE CHOSEN SYMBOL IS PRINTED. v1 silently collapsed BOP-OCT and BOP-OCTB
#      to root BOP and picked one by volume without saying which. Now you can see
#      the pick and how ambiguous it was.
#   5. SNAPSHOT SHAPE IS VERIFIED. The touch is built by grouping on orig_time,
#      which is only correct if each orig_time carries a full book image. Section 0
#      prints rows-per-orig_time so that assumption is checked, not assumed.
#
# Read-only, ~3-5 minutes. Run: caffeinate -is python probe_futures_width.py
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
# the minimum TRADES for a root-day to count as a real market, not a quote ghost
MIN_TRADES = 10
# the PSX futures tick, confirmed by probe_futures_mm.py section 1
TICK = 0.01
# the gross markout budget per round trip in the share book, bps -- the hurdle
EDGE_BUDGET_BPS = 2.67
# futures round-trip fee (Laga + CCPF; CDC=0 because squared up, no delivery)
FUT_FEE_RT_BPS = 0.19
# the fraction of time at >= 3 ticks a root must clear for the lean to be usable
WIDE_GATE = 0.85

con = duckdb.connect(); pd.set_option("display.width", 210, "display.max_columns", 40)
TRADES = f"{PARSED}/trades/date=*/*.parquet"
SNAP = f"{PARSED}/ob_snapshot/date=*/*.parquet"
print(f"sampling the last {N_DAYS} trading dates, "
      f"min {MIN_SNAPS} snaps and {MIN_TRADES} trades per root-day\n")

# =============================================================================
# SECTION 0 -- verify the snapshot shape before trusting any width
# =============================================================================
# The touch is built as MAX(bid at level 1) / MIN(offer at level 1) grouped by
# orig_time. That is only a real touch if one orig_time carries a full book
# image (one row per side per level). If the table is an incremental event log,
# one orig_time carries a SINGLE message and the "touch" is nonsense.
q0 = f"""
WITH d AS (
  -- the single most recent futures date, to keep this check cheap
  SELECT MAX(date) AS date FROM read_parquet('{TRADES}') WHERE market='STOCK_DEL_FUT'
),
s AS (
  -- every futures snapshot row on that date
  SELECT s.symbol, s.orig_time, s.entry_type, s.level
  FROM read_parquet('{SNAP}') s, d
  WHERE s.market='STOCK_DEL_FUT' AND s.date=d.date AND s.phase='CONTINUOUS_AUCTION'
)
SELECT
  -- how many rows share one (symbol, orig_time): 1 means event log, >1 means image
  ROUND(AVG(n),2) AS avg_rows_per_orig_time,
  MIN(n) AS min_rows, MAX(n) AS max_rows,
  -- how often a single orig_time carries BOTH a level-1 bid and a level-1 offer
  ROUND(100.0*AVG(CASE WHEN has_bid1=1 AND has_ask1=1 THEN 1 ELSE 0 END),1) AS pct_two_sided
FROM (
  SELECT symbol, orig_time, COUNT(*) AS n,
         MAX(CASE WHEN entry_type='BID'   AND level=1 THEN 1 ELSE 0 END) AS has_bid1,
         MAX(CASE WHEN entry_type='OFFER' AND level=1 THEN 1 ELSE 0 END) AS has_ask1
  FROM s GROUP BY symbol, orig_time)
"""
shape = con.execute(q0).df()
print("SECTION 0 -- snapshot shape check (is one orig_time a full book image?)")
print(shape.to_string(index=False))
# tell the reader what the answer means instead of leaving them to infer it
_avg = float(shape.avg_rows_per_orig_time.iloc[0])
if _avg < 2.0:
    print("  WARNING: ~1 row per orig_time -- this looks like an EVENT LOG, not a")
    print("  book image. Every width below is then unreliable. Stop and fix the")
    print("  touch construction before using any number in this script.\n")
else:
    print(f"  OK: {_avg:.1f} rows per orig_time -- consistent with a book image.\n")

# =============================================================================
# SECTION 1 -- per-root width and hedge cost
# =============================================================================
q = f"""
WITH dates AS (
  -- the most recent N trading dates that have any futures activity
  SELECT DISTINCT date FROM read_parquet('{TRADES}')
  WHERE market='STOCK_DEL_FUT' ORDER BY date DESC LIMIT {N_DAYS}
),
fut AS (
  -- every futures trade on those dates, with the root parsed off the suffix
  SELECT t.date, regexp_extract(t.symbol,'^([A-Z0-9]+)-',1) AS root,
         t.symbol, t.qty, t.transact_time
  FROM read_parquet('{TRADES}') t, dates d
  WHERE t.market='STOCK_DEL_FUT' AND t.date=d.date
),
by_sym AS (
  -- per root-day-contract: trade count, traded volume, and the day's time span
  -- NOTE: trades.transact_time is the exchange timestamp; ob_snapshot calls the
  -- same thing orig_time. Both are TIMESTAMP WITH TIME ZONE, so the BETWEEN
  -- comparison in per_day is comparing like with like.
  SELECT date, root, symbol,
         COUNT(*) AS n_trades, SUM(qty) AS vol,
         MIN(transact_time) AS first_trade, MAX(transact_time) AS last_trade
  FROM fut GROUP BY date, root, symbol
),
root_vol AS (
  -- the root's TOTAL volume that day, across every contract month
  SELECT date, root, SUM(vol) AS root_day_vol FROM by_sym GROUP BY date, root
),
active AS (
  -- the ACTIVE contract per root per date = the most-traded one that day.
  -- This excludes the near-dead back months whose books are nonsense
  -- (BOP-AUG showed a 696-tick spread on 26 snapshots).
  -- vol_share says how DOMINANT that pick was: 0.95 is an unambiguous front
  -- month, 0.45 means the roll was in progress and the pick is arbitrary.
  SELECT b.date, b.root, b.symbol, b.n_trades, b.vol,
         b.first_trade, b.last_trade,
         b.vol / NULLIF(r.root_day_vol,0) AS vol_share
  FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY date, root ORDER BY vol DESC) rn
        FROM by_sym) b
  JOIN root_vol r ON b.date=r.date AND b.root=r.root
  WHERE b.rn=1
),
touch AS (
  -- L1 bid/ask per snapshot message, continuous phase only, active contracts.
  -- first_trade/last_trade ride along so in-session width can be split out.
  SELECT s.date, a.root, s.symbol, s.orig_time,
         a.n_trades, a.vol, a.vol_share, a.first_trade, a.last_trade,
         MAX(CASE WHEN s.entry_type='BID'   AND s.level=1 THEN s.px END) AS bb,
         MIN(CASE WHEN s.entry_type='OFFER' AND s.level=1 THEN s.px END) AS ba
  FROM read_parquet('{SNAP}') s
  JOIN active a ON s.date=a.date AND s.symbol=a.symbol
  WHERE s.market='STOCK_DEL_FUT' AND s.phase='CONTINUOUS_AUCTION'
  GROUP BY s.date, a.root, s.symbol, s.orig_time,
           a.n_trades, a.vol, a.vol_share, a.first_trade, a.last_trade
),
ok AS (
  -- two-sided, uncrossed touches only
  SELECT *, (ba-bb)/{TICK} AS tk, (ba-bb)/((ba+bb)/2)*1e4 AS bps
  FROM touch WHERE bb>0 AND ba>0 AND ba>=bb
),
per_day AS (
  -- one width observation per root-day, on days with enough snapshots AND
  -- enough trades. MIN_TRADES is what stops a quote-ghost contract -- thousands
  -- of flickering quote messages, almost no trades -- from being reported as a
  -- market. That filter did not exist in v1.
  SELECT date, root, ANY_VALUE(symbol) AS symbol, COUNT(*) AS snaps,
         ANY_VALUE(n_trades) AS n_trades, ANY_VALUE(vol) AS vol,
         ANY_VALUE(vol_share) AS vol_share,
         -- width over the WHOLE continuous session (what v1 reported)
         MEDIAN(tk) AS med_tk_all,
         -- width IN SESSION only: between the day's first and last trade.
         -- The gap between this and med_tk_all is the dead-time inflation.
         MEDIAN(CASE WHEN orig_time BETWEEN first_trade AND last_trade
                     THEN tk END) AS med_tk_insess,
         -- the tight end of the distribution, so a fat tail cannot hide it
         QUANTILE_CONT(tk, 0.25) AS p25_tk,
         MEDIAN(CASE WHEN orig_time BETWEEN first_trade AND last_trade
                     THEN bps END) AS med_bps_insess,
         -- the gate quantity: how OFTEN the book is at least 3 ticks wide
         AVG(CASE WHEN tk >= 3 THEN 1.0 ELSE 0.0 END) AS frac_ge3
  FROM ok GROUP BY date, root
  HAVING COUNT(*) >= {MIN_SNAPS} AND ANY_VALUE(n_trades) >= {MIN_TRADES}
)
SELECT root,
       COUNT(*)                            AS days,
       ANY_VALUE(symbol)                   AS example_contract,
       ROUND(MEDIAN(n_trades))             AS trades_day,
       ROUND(MEDIAN(vol))                  AS vol_day,
       ROUND(MEDIAN(vol_share),2)          AS vol_share,
       ROUND(MEDIAN(p25_tk),1)             AS p25_ticks,
       ROUND(MEDIAN(med_tk_insess),1)      AS med_ticks_insess,
       ROUND(MEDIAN(med_tk_all),1)         AS med_ticks_all,
       ROUND(MEDIAN(med_bps_insess),1)     AS med_bps_insess,
       ROUND(100*MEDIAN(frac_ge3),1)       AS pct_time_ge_3_ticks,
       ROUND(AVG(snaps))                   AS avg_snaps_day
FROM per_day GROUP BY root ORDER BY med_bps_insess
"""
df = con.execute(q).df()

# the cost of using this contract as a hedge, in bps, derived not re-measured.
# Crossing to put the hedge on costs half the quoted spread against mid;
# crossing to take it off costs the other half. One round trip = one full spread.
df["hedge_rt_bps"] = (df.med_bps_insess + FUT_FEE_RT_BPS).round(2)
# and what that is as a multiple of the edge it is supposed to protect
df["x_edge"] = (df.hedge_rt_bps / EDGE_BUDGET_BPS).round(1)
# how badly v1's all-session median overstated the in-session one
df["deadtime_infl"] = (df.med_ticks_all / df.med_ticks_insess).round(2)

print("SECTION 1 -- per-root width and hedge cost, sorted cheapest hedge first")
print(df.to_string(index=False))

# =============================================================================
# SECTION 2 -- did the dead-time weighting matter?
# =============================================================================
print("\nSECTION 2 -- dead-time inflation (med_ticks_all / med_ticks_insess)")
# a ratio near 1.0 means v1's number was fine; well above 1.0 means it was not
print(f"  median inflation across roots: {df.deadtime_infl.median():.2f}x")
print(f"  roots inflated by >1.5x: {(df.deadtime_infl > 1.5).sum()} of {len(df)}")
print(f"  worst: {df.nlargest(5,'deadtime_infl')[['root','deadtime_infl']].to_string(index=False)}")

# =============================================================================
# SECTION 3 -- the two verdicts
# =============================================================================
print("\nSECTION 3a -- MARKET MAKING: can this root take the 2-tick lean?")
print(f"  gate: at least 3 ticks wide at least {100*WIDE_GATE:.0f}% of the time")
# gate on the FRACTION of time wide enough, not on the median -- v1's bug
wide = df[df.pct_time_ge_3_ticks >= 100*WIDE_GATE]
thin = df[df.pct_time_ge_3_ticks < 100*WIDE_GATE]
print(f"  WIDE ENOUGH ({len(wide)}): {', '.join(wide.root.tolist()) or '-'}")
print(f"  TOO THIN    ({len(thin)}): {', '.join(thin.root.tolist()) or '-'}")

print("\nSECTION 3b -- HEDGING: what does one hedge round trip cost?")
print(f"  hurdle: the share book's entire gross edge is {EDGE_BUDGET_BPS} bps per round trip")
# the only roots where the hedge is not self-defeating
cheap = df[df.hedge_rt_bps < EDGE_BUDGET_BPS]
print(f"  hedge COSTS LESS than the edge it protects ({len(cheap)}): "
      f"{', '.join(cheap.root.tolist()) or '-- none --'}")
print(f"  cheapest root: {df.iloc[0].root} at {df.iloc[0].hedge_rt_bps} bps "
      f"= {df.iloc[0].x_edge}x the edge")
print(f"  median across roots: {df.hedge_rt_bps.median():.1f} bps "
      f"= {df.x_edge.median():.1f}x the edge")
print("\n  Read this as a per-hedge toll. A book that re-hedges every inventory")
print("  blip pays it on every blip. Only a residual/end-of-day hedge amortises it.")

# write the table out so the next step does not have to re-run the query
OUT = Path(PARSED).parent / "Capital Stake - Results" / "futures_width_20260915.csv"
try:
    # keep the numbers, do not overwrite anything that exists
    if not OUT.exists():
        df.to_csv(OUT, index=False)
        print(f"\nwrote {OUT}")
    else:
        print(f"\nNOT written, already exists: {OUT}")
except Exception as e:
    print(f"\ncould not write csv: {e}")
