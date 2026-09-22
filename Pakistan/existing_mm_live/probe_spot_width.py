# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# probe_spot_width.py -- how wide is the SHARE book, in bps, for the names that
# also have an active-month deliverable future? And which of the two is cheaper
# to cross?
#
# WHY THIS EXISTS. probe_futures_width.py measured the futures book: BOP 11.6 bps,
# MLCF 12.0, TRG 12.6, median 42.9 across the 89 watchlist names with a contract.
# That number is currently compared against nothing. The delta-hedge ladder says
# first preference for a hedge is the same-ticker active-month future -- but if
# you are long BOP shares, selling BOP futures and selling BOP shares both take
# you flat, so the real question is which book is cheaper to cross.
#
# THE INFERENCE THIS TESTS. The share book earns ~2.67 bps gross per round trip,
# which implies a captured half-spread near 1.3 bps and therefore a quoted spot
# spread on the order of 3 bps -- against 11.6 bps in BOP futures. If that holds,
# crossing in shares is ~4x cheaper than crossing in the future, and the first
# rung of the hedge ladder is wrong for every name with a liquid share book.
# THIS SCRIPT MEASURES IT INSTEAD OF INFERRING IT.
#
# WHAT IT DOES NOT SETTLE. Short-selling constraints. If PSX restricts
# establishing or carrying a short in the cash market, the future is not the
# more expensive hedge -- it is the only one, and the comparison below binds on
# the long side only. That is an exchange-rules question, not a data question.
#
# Read-only, ~5-10 minutes (the spot book is far larger than the futures book).
# Run: caffeinate -is python probe_spot_width.py
from pathlib import Path
import duckdb, pandas as pd

# --- PARSED STORE from config_pk; never a literal in this file ---------------
try:
    # the project's central path module
    import config_pk
    # accept either spelling for the parsed store
    _P = getattr(config_pk, "PARSED_ROOT", None) or getattr(config_pk, "PARSED", None)
    # accept either spelling for the results root
    _R = getattr(config_pk, "RESULTS_ROOT", None) or getattr(config_pk, "RESULTS", None)
    # as strings for the globs
    PARSED = str(_P) if _P else ""
    RESULTS = str(_R) if _R else ""
except Exception:
    # not importable from this directory
    PARSED, RESULTS = "", ""
# current literals only as a fallback
if not PARSED:
    # Resolve this filesystem path through the canonical checkout/data configuration.
    PARSED = str(_hft_paths.PARSED_ROOT)
if not RESULTS:
    # Resolve this filesystem path through the canonical checkout/data configuration.
    RESULTS = str(_hft_paths.RESULTS_ROOT)
# fail immediately with the path named, not inside a glob
if not Path(PARSED).is_dir():
    raise SystemExit(f"PARSED store not found: {PARSED}")
if not Path(RESULTS).is_dir():
    raise SystemExit(f"RESULTS root not found: {RESULTS}")
# self-describing output
print(f"parsed store: {PARSED}")
print(f"results root: {RESULTS}")

# how many of the most recent trading dates to sample -- same window as the
# futures probe so the two numbers are measured over the same days
N_DAYS = 20
# the minimum snapshots for a symbol-day to count
MIN_SNAPS = 200
# the minimum trades for a symbol-day to count as a real market
MIN_TRADES = 10
# the PSX tick, flat 0.01 PKR on both books
TICK = 0.01
# the gross markout budget per round trip in the share book, bps -- the hurdle
EDGE_BUDGET_BPS = 2.67
# spot round-trip fee: TREC 7.77e-5 = 0.777 bps/side
SPOT_FEE_RT_BPS = 1.554
# futures round-trip fee (Laga + CCPF; CDC=0 because squared up, no delivery)
FUT_FEE_RT_BPS = 0.19
# the futures result this joins against, written by probe_futures_width.py
FUT_CSV = Path(RESULTS) / "futures_width_20260915.csv"

con = duckdb.connect(); pd.set_option("display.width", 220, "display.max_columns", 40)
TRADES = f"{PARSED}/trades/date=*/*.parquet"
SNAP = f"{PARSED}/ob_snapshot/date=*/*.parquet"

# =============================================================================
# SECTION 0 -- what market codes exist? DO NOT GUESS THE SPOT CODE.
# =============================================================================
# The futures probe used market='STOCK_DEL_FUT' because that string was known.
# The spot code is not, and guessing it is the same class of mistake as assuming
# trades had an orig_time column. Read it off the data.
q0 = f"""
WITH dates AS (
  -- the most recent N trading dates in the store, any market
  SELECT DISTINCT date FROM read_parquet('{TRADES}') ORDER BY date DESC LIMIT {N_DAYS}
)
SELECT t.market,
       COUNT(*)                     AS trades,
       COUNT(DISTINCT t.symbol)     AS symbols,
       COUNT(DISTINCT t.date)       AS days,
       MIN(t.date)                  AS first_date,
       MAX(t.date)                  AS last_date
FROM read_parquet('{TRADES}') t, dates d
WHERE t.date = d.date
GROUP BY t.market ORDER BY trades DESC
"""
mk = con.execute(q0).df()
print("\nSECTION 0 -- market codes present in the last "
      f"{N_DAYS} trading dates (pick the spot code from here)")
print(mk.to_string(index=False))

# the spot book is the non-futures market with the most symbols: futures symbols
# carry a month suffix and are far fewer. Choose it programmatically, then SAY
# which one was chosen so a wrong pick is visible rather than silent.
cand = mk[~mk.market.str.contains("FUT", case=False, na=False)]
if cand.empty:
    raise SystemExit("no non-futures market code found -- inspect SECTION 0 and set SPOT_MKT by hand")
# most symbols = the main board, not an odd-lot or negotiated-deal market
SPOT_MKT = cand.sort_values("symbols", ascending=False).market.iloc[0]
print(f"\n  --> using SPOT_MKT = {SPOT_MKT!r}")
print( "      If that is wrong, set SPOT_MKT by hand from the table above and re-run.")

# =============================================================================
# SECTION 1 -- spot book width, same machinery as the futures probe
# =============================================================================
q1 = f"""
WITH dates AS (
  -- the most recent N trading dates that have spot activity
  SELECT DISTINCT date FROM read_parquet('{TRADES}')
  WHERE market='{SPOT_MKT}' ORDER BY date DESC LIMIT {N_DAYS}
),
by_sym AS (
  -- per symbol-day: trade count, traded volume, and the day's trading span.
  -- trades.transact_time is the exchange timestamp; ob_snapshot calls the same
  -- thing orig_time. Both are TIMESTAMP WITH TIME ZONE, so the BETWEEN in
  -- per_day compares like with like.
  SELECT t.date, t.symbol,
         COUNT(*) AS n_trades, SUM(t.qty) AS vol,
         MIN(t.transact_time) AS first_trade, MAX(t.transact_time) AS last_trade
  FROM read_parquet('{TRADES}') t, dates d
  WHERE t.market='{SPOT_MKT}' AND t.date=d.date
  GROUP BY t.date, t.symbol
),
touch AS (
  -- L1 bid/ask per snapshot message, continuous phase only
  SELECT s.date, s.symbol, s.orig_time,
         b.n_trades, b.vol, b.first_trade, b.last_trade,
         MAX(CASE WHEN s.entry_type='BID'   AND s.level=1 THEN s.px END) AS bb,
         MIN(CASE WHEN s.entry_type='OFFER' AND s.level=1 THEN s.px END) AS ba
  FROM read_parquet('{SNAP}') s
  JOIN by_sym b ON s.date=b.date AND s.symbol=b.symbol
  WHERE s.market='{SPOT_MKT}' AND s.phase='CONTINUOUS_AUCTION'
  GROUP BY s.date, s.symbol, s.orig_time,
           b.n_trades, b.vol, b.first_trade, b.last_trade
),
ok AS (
  -- two-sided, uncrossed touches only
  SELECT *, (ba-bb)/{TICK} AS tk, (ba-bb)/((ba+bb)/2)*1e4 AS bps
  FROM touch WHERE bb>0 AND ba>0 AND ba>=bb
),
per_day AS (
  -- one observation per symbol-day, on days with enough snapshots AND trades
  SELECT date, symbol, COUNT(*) AS snaps,
         ANY_VALUE(n_trades) AS n_trades, ANY_VALUE(vol) AS vol,
         -- width IN SESSION: between the day's first and last trade
         MEDIAN(CASE WHEN orig_time BETWEEN first_trade AND last_trade
                     THEN tk END) AS med_tk_insess,
         MEDIAN(CASE WHEN orig_time BETWEEN first_trade AND last_trade
                     THEN bps END) AS med_bps_insess,
         -- the tight end, so a fat tail cannot hide it
         QUANTILE_CONT(bps, 0.25) AS p25_bps
  FROM ok GROUP BY date, symbol
  HAVING COUNT(*) >= {MIN_SNAPS} AND ANY_VALUE(n_trades) >= {MIN_TRADES}
)
SELECT symbol                              AS root,
       COUNT(*)                            AS spot_days,
       ROUND(MEDIAN(n_trades))             AS spot_trades_day,
       ROUND(MEDIAN(vol))                  AS spot_vol_day,
       ROUND(MEDIAN(med_tk_insess),1)      AS spot_ticks,
       ROUND(MEDIAN(p25_bps),1)            AS spot_p25_bps,
       ROUND(MEDIAN(med_bps_insess),2)     AS spot_bps
FROM per_day GROUP BY symbol ORDER BY spot_bps
"""
sp = con.execute(q1).df()
print(f"\nSECTION 1 -- spot book width, {len(sp)} symbols cleared the floors")

# =============================================================================
# SECTION 2 -- the comparison: which book is cheaper to cross?
# =============================================================================
if not FUT_CSV.exists():
    # say exactly which file is missing rather than failing on an empty join
    print(f"\nSECTION 2 SKIPPED -- futures result not found at {FUT_CSV}")
    print("  Run probe_futures_width.py first, then re-run this.")
    cmp = sp
else:
    # the futures side, as written by the futures probe
    fu = pd.read_csv(FUT_CSV)
    # keep only the columns needed, under names that say which book they are
    fu = fu[["root","trades_day","med_bps_insess","hedge_rt_bps"]].rename(columns={
        "trades_day": "fut_trades_day",
        "med_bps_insess": "fut_bps",
        "hedge_rt_bps": "fut_hedge_rt_bps"})
    # inner join: only names that have BOTH a share book and a futures contract
    cmp = sp.merge(fu, on="root", how="inner")
    # round-trip cost of hedging in the SHARE book: one full spread + spot fee
    cmp["spot_hedge_rt_bps"] = (cmp.spot_bps + SPOT_FEE_RT_BPS).round(2)
    # how many times more expensive the future is than the shares (>1 = shares win)
    cmp["fut_over_spot"] = (cmp.fut_hedge_rt_bps / cmp.spot_hedge_rt_bps).round(2)
    # each book's cost as a multiple of the edge it is meant to protect
    cmp["spot_x_edge"] = (cmp.spot_hedge_rt_bps / EDGE_BUDGET_BPS).round(1)
    cmp["fut_x_edge"] = (cmp.fut_hedge_rt_bps / EDGE_BUDGET_BPS).round(1)
    # cheapest share book first
    cmp = cmp.sort_values("spot_hedge_rt_bps")

    print(f"\nSECTION 2 -- {len(cmp)} names with BOTH a share book and a futures contract")
    print(cmp[["root","spot_trades_day","fut_trades_day","spot_ticks","spot_bps","fut_bps",
               "spot_hedge_rt_bps","fut_hedge_rt_bps","fut_over_spot",
               "spot_x_edge","fut_x_edge"]].to_string(index=False))

    print("\nSECTION 3 -- the verdict on the hedge ladder's first rung")
    # the count that decides it
    n_spot_cheaper = int((cmp.fut_over_spot > 1).sum())
    print(f"  shares cheaper to cross than the future: {n_spot_cheaper} of {len(cmp)} names")
    print(f"  median futures/spot cost ratio: {cmp.fut_over_spot.median():.2f}x")
    print(f"  median spot hedge round trip:  {cmp.spot_hedge_rt_bps.median():6.2f} bps "
          f"= {cmp.spot_x_edge.median():.1f}x the {EDGE_BUDGET_BPS} bps edge")
    print(f"  median fut  hedge round trip:  {cmp.fut_hedge_rt_bps.median():6.2f} bps "
          f"= {cmp.fut_x_edge.median():.1f}x the {EDGE_BUDGET_BPS} bps edge")
    # the only names where crossing at all is not self-defeating
    ok_spot = cmp[cmp.spot_hedge_rt_bps < EDGE_BUDGET_BPS]
    print(f"\n  names where crossing the SHARE book costs less than the edge "
          f"({len(ok_spot)}): {', '.join(ok_spot.root.tolist()) or '-- none --'}")
    print("\n  Reminder: this settles COST only. If PSX restricts cash-market shorts,")
    print("  the future is not the dearer hedge on the short side -- it is the only")
    print("  one, and this comparison binds on the long side alone.")

# write the table out; never overwrite an existing file
OUT = Path(RESULTS) / "spot_vs_futures_width_20260915.csv"
try:
    if not OUT.exists():
        cmp.to_csv(OUT, index=False)
        print(f"\nwrote {OUT}")
    else:
        print(f"\nNOT written, already exists: {OUT}")
except Exception as e:
    print(f"\ncould not write csv: {e}")
