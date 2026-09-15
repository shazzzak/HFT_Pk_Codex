# probe_etf_hedge.py -- can the ETF leg of the hedge ladder be crossed at all,
# and is there anything in the markout for it to hedge?
#
# WHY THIS EXISTS. The delta-hedge ladder falls back to a sector ETF, then to a
# broad-market ETF, when a name has no liquid future. probe_futures_width.py and
# probe_spot_width.py settled the first two rungs:
#   - crossing the share book beats crossing the same-ticker future on 99 of 100
#     names (median 3.30x), so the ladder's stated first preference is inverted
#   - NO name clears the 2.67 bps gross edge by crossing in EITHER book: best is
#     EFERT shares at 5.30 bps = 2.0x the edge, median share book 13.79 bps
# The ETF rung has never been measured. It is the rung that matters most, because
# it is the only one that can hedge the whole portfolio at once, and portfolio
# netting is what the whole design now rests on.
#
# THE SCREEN THAT COSTS NOTHING. PSX ticks are flat 0.01 PKR, so a ONE-TICK
# spread is 100/P bps. Therefore, before measuring anything:
#     spread < 2.67 bps (the gross edge)          requires price > 37.45 PKR
#     round trip < 2.67 bps including 1.554 fees  requires price > 89.6  PKR
# An ETF trading below ~90 PKR cannot be crossed round-trip inside the edge even
# with a perfect one-tick book. Section 2 applies that before any spread work.
#
# WHAT THIS SCRIPT DOES NOT ANSWER. How much of the book's markout is actually
# market beta -- i.e. how much there is for ANY market hedge to remove. If the
# adverse move after a fill is mostly name-specific (someone picking you off on
# information about that company), a market hedge removes almost none of it and
# the ETF rung is dead however cheap the ETF is. That measurement needs the
# fill-level markout data, so Section 5 prints the fills schema rather than
# guessing at column names -- the regression is the next script, written against
# what Section 5 reports.
#
# Read-only. ~5-10 minutes. Run: caffeinate -is python probe_etf_hedge.py
from pathlib import Path
import duckdb, pandas as pd, numpy as np

# --- PATHS from config_pk; never a literal in this file ----------------------
try:
    # the project's central path module
    import config_pk
    # the parsed store
    PARSED = str(config_pk.PARSED_ROOT)
    # the results root, for the output csv
    RESULTS = str(config_pk.RESULTS_ROOT)
    # where fill_attribution.py writes its per-fill output
    FILLS = getattr(config_pk, "FILLS_DIR", None)
except Exception as _e:
    # a wrong store is worse than a crash -- fail naming the fix
    raise SystemExit("probe_etf_hedge: could not import paths from config_pk. Run "
                     "from existing_mm_live/. Original: %r" % _e)
# fail immediately with the path named, not inside a glob
for _lbl, _p in (("PARSED", PARSED), ("RESULTS", RESULTS)):
    if not Path(_p).is_dir():
        raise SystemExit(f"{_lbl} not found: {_p}")
# self-describing output
print(f"parsed store: {PARSED}")

# the ETF universe SZ is working from
ETFS = ["NITGETF", "UBLPETF", "NBPGETF",   # broad market proxies
        "MZNPETF", "MIIETF",               # Islamic
        "JSMFETF",                         # momentum factor
        "ACIETF",                          # consumer index
        "JSGBETF",                         # banking sector
        "HBLTETF"]                         # debt
# how many recent trading dates to sample -- same window as the width probes
N_DAYS = 20
# minimum snapshots for a symbol-day to count
MIN_SNAPS = 200
# minimum trades for a symbol-day to count as a real market, not a quote ghost
MIN_TRADES = 10
# the PSX tick, flat on every board
TICK = 0.01
# the share book's gross markout budget per round trip, bps -- the hurdle
EDGE_BPS = 2.67
# spot round-trip fee: TREC 7.77e-5 = 0.777 bps/side
SPOT_FEE_RT_BPS = 1.554
# price below which even a one-tick spread exceeds the edge
PRICE_FLOOR_SPREAD = TICK * 1e4 / EDGE_BPS
# price below which a one-tick ROUND TRIP plus fees exceeds the edge
PRICE_FLOOR_RT = TICK * 1e4 / max(EDGE_BPS - SPOT_FEE_RT_BPS, 1e-9)
# return horizons in minutes for the beta / R-squared work
HORIZONS_MIN = [1, 5, 15]

con = duckdb.connect(); pd.set_option("display.width", 200, "display.max_columns", 30)
TRADES = f"{PARSED}/trades/date=*/*.parquet"
SNAP = f"{PARSED}/ob_snapshot/date=*/*.parquet"
# the SQL list literal for the ETF symbols
ETF_SQL = ",".join(f"'{s}'" for s in ETFS)

# =============================================================================
# SECTION 1 -- do these ETFs exist in the store, and do they trade?
# =============================================================================
q1 = f"""
WITH dates AS (
  -- the most recent N trading dates on the main board
  SELECT DISTINCT date FROM read_parquet('{TRADES}')
  WHERE market='REG' ORDER BY date DESC LIMIT {N_DAYS}
)
SELECT t.symbol,
       COUNT(DISTINCT t.date)        AS days,
       ROUND(COUNT(*)*1.0
             / COUNT(DISTINCT t.date))       AS trades_day,
       ROUND(SUM(t.qty)*1.0
             / COUNT(DISTINCT t.date))       AS shares_day,
       ROUND(MEDIAN(t.price), 2)             AS px,
       ROUND(SUM(t.price*t.qty)*1.0
             / COUNT(DISTINCT t.date) / 1e6, 2) AS notional_m_day
FROM read_parquet('{TRADES}') t, dates d
WHERE t.market='REG' AND t.date=d.date AND t.symbol IN ({ETF_SQL})
GROUP BY t.symbol ORDER BY notional_m_day DESC
"""
etf = con.execute(q1).df()
print(f"\nSECTION 1 -- ETF activity on the main board, last {N_DAYS} trading dates")
# name the ones that are simply not there, rather than letting them vanish
missing = sorted(set(ETFS) - set(etf.symbol))
if len(etf):
    print(etf.to_string(index=False))
else:
    print("  none of the listed ETFs traded at all in this window")
print(f"\n  NOT PRESENT in the store at all ({len(missing)}): {', '.join(missing) or '-'}")

# =============================================================================
# SECTION 2 -- the tick-grid screen, applied BEFORE any spread measurement
# =============================================================================
print("\nSECTION 2 -- tick-grid screen (PSX tick is flat 0.01 PKR, so one tick = 100/P bps)")
print(f"  an instrument under {PRICE_FLOOR_SPREAD:.2f} PKR cannot have a spread under {EDGE_BPS} bps")
print(f"  an instrument under {PRICE_FLOOR_RT:.2f} PKR cannot have a ROUND TRIP inside the edge")
print(f"  (round trip must fit {EDGE_BPS} - {SPOT_FEE_RT_BPS} = {EDGE_BPS-SPOT_FEE_RT_BPS:.3f} bps of spread)")
if len(etf):
    # one tick expressed in bps at each ETF's own price -- the floor on its spread
    etf["one_tick_bps"] = (TICK / etf.px * 1e4).round(2)
    # the best possible round trip: one tick of spread plus the spot fee
    etf["best_rt_bps"] = (etf.one_tick_bps + SPOT_FEE_RT_BPS).round(2)
    # does the arithmetic alone rule it out?
    etf["passes_screen"] = etf.best_rt_bps < EDGE_BPS
    print()
    print(etf[["symbol", "px", "trades_day", "notional_m_day",
               "one_tick_bps", "best_rt_bps", "passes_screen"]].to_string(index=False))
    # the survivors, if any
    survivors = etf[etf.passes_screen].symbol.tolist()
    print(f"\n  SURVIVE the screen ({len(survivors)}): {', '.join(survivors) or '-- none --'}")
    # say plainly what a zero here means, so the result is not misread
    if not survivors:
        print("  Every listed ETF is priced too low for a one-tick round trip to fit")
        print("  inside the edge. The ETF rung cannot work on cost grounds alone, and")
        print("  no amount of measurement changes that -- it is the tick grid, not")
        print("  liquidity. Sections 3-4 below are then informational only.")
else:
    survivors = []

# =============================================================================
# SECTION 3 -- measured spread for the ETFs that actually trade
# =============================================================================
# measure every ETF that traded, not just the screen survivors: a name that fails
# the screen can still be worth knowing about if the screen is later revisited
# with a different edge number
live_etfs = etf.symbol.tolist() if len(etf) else []
if live_etfs:
    LIVE_SQL = ",".join(f"'{s}'" for s in live_etfs)
    q3 = f"""
    WITH dates AS (
      SELECT DISTINCT date FROM read_parquet('{TRADES}')
      WHERE market='REG' ORDER BY date DESC LIMIT {N_DAYS}
    ),
    by_sym AS (
      -- per symbol-day: trade count and the day's trading span.
      -- trades.transact_time is the exchange clock; ob_snapshot calls the same
      -- thing orig_time. Both are TIMESTAMP WITH TIME ZONE.
      SELECT t.date, t.symbol, COUNT(*) AS n_trades,
             MIN(t.transact_time) AS first_trade, MAX(t.transact_time) AS last_trade
      FROM read_parquet('{TRADES}') t, dates d
      WHERE t.market='REG' AND t.date=d.date AND t.symbol IN ({LIVE_SQL})
      GROUP BY t.date, t.symbol
    ),
    touch AS (
      -- L1 bid/ask per snapshot message, continuous phase only
      SELECT s.date, s.symbol, s.orig_time, b.n_trades, b.first_trade, b.last_trade,
             MAX(CASE WHEN s.entry_type='BID'   AND s.level=1 THEN s.px END) AS bb,
             MIN(CASE WHEN s.entry_type='OFFER' AND s.level=1 THEN s.px END) AS ba
      FROM read_parquet('{SNAP}') s
      JOIN by_sym b ON s.date=b.date AND s.symbol=b.symbol
      WHERE s.market='REG' AND s.phase='CONTINUOUS_AUCTION'
      GROUP BY s.date, s.symbol, s.orig_time, b.n_trades, b.first_trade, b.last_trade
    ),
    ok AS (
      -- two-sided, uncrossed touches only
      SELECT *, (ba-bb)/{TICK} AS tk, (ba-bb)/((ba+bb)/2)*1e4 AS bps
      FROM touch WHERE bb>0 AND ba>0 AND ba>=bb
    ),
    per_day AS (
      -- one observation per symbol-day, on days with enough snapshots AND trades
      SELECT date, symbol, COUNT(*) AS snaps,
             -- in-session only: between the day's first and last trade
             MEDIAN(CASE WHEN orig_time BETWEEN first_trade AND last_trade
                         THEN tk END) AS med_tk,
             MEDIAN(CASE WHEN orig_time BETWEEN first_trade AND last_trade
                         THEN bps END) AS med_bps,
             -- how often the book is even two-sided and quotable
             AVG(1.0) AS quotable
      FROM ok GROUP BY date, symbol
      HAVING COUNT(*) >= {MIN_SNAPS} AND ANY_VALUE(n_trades) >= {MIN_TRADES}
    )
    SELECT symbol, COUNT(*) AS days,
           ROUND(MEDIAN(med_tk),1)  AS med_ticks,
           ROUND(MEDIAN(med_bps),2) AS med_bps,
           ROUND(AVG(snaps))        AS avg_snaps_day
    FROM per_day GROUP BY symbol ORDER BY med_bps
    """
    sp = con.execute(q3).df()
    print(f"\nSECTION 3 -- measured ETF spread, in-session, days with >={MIN_SNAPS} "
          f"snaps and >={MIN_TRADES} trades")
    if len(sp):
        # the real round-trip cost of using this ETF as a hedge
        sp["hedge_rt_bps"] = (sp.med_bps + SPOT_FEE_RT_BPS).round(2)
        # as a multiple of the edge it is meant to protect
        sp["x_edge"] = (sp.hedge_rt_bps / EDGE_BPS).round(1)
        print(sp.to_string(index=False))
        # the ETFs that traded but never cleared the day floors
        thin = sorted(set(live_etfs) - set(sp.symbol))
        print(f"\n  traded but never cleared the day floors ({len(thin)}): "
              f"{', '.join(thin) or '-'}")
    else:
        sp = pd.DataFrame()
        print("  no ETF-day cleared the floors -- these books are not quotable markets")
else:
    sp = pd.DataFrame()
    print("\nSECTION 3 -- skipped, no ETF traded in the window")

# =============================================================================
# SECTION 4 -- how much of a name's move does the ETF actually explain?
# =============================================================================
# Cost is only half the question. The other half: if you hedge a name with an
# ETF, how much of the name's move does the ETF capture? That is the R-squared
# of the name's return on the ETF's return. A cheap hedge that explains 5% of the
# move is not a hedge.
#
# NOTE ON HORIZON. Correlations between two assets measured over short intervals
# are biased DOWNWARD when the two trade at different frequencies (the Epps
# effect). The ETF is the thinner leg here, so the 1-minute R-squared is a floor
# and the 15-minute figure is closer to the true relationship. Read the decay
# across horizons, not any single number.
if len(sp):
    # the most traded surviving ETF is the natural market proxy
    proxy = sp.sort_values("med_bps").symbol.iloc[0]
    print(f"\nSECTION 4 -- beta and R-squared against {proxy} "
          f"(the tightest ETF book), at {HORIZONS_MIN} minute horizons")
    q4 = f"""
    WITH dates AS (
      SELECT DISTINCT date FROM read_parquet('{TRADES}')
      WHERE market='REG' ORDER BY date DESC LIMIT {N_DAYS}
    ),
    touch AS (
      -- L1 mid per snapshot for every main-board symbol on those dates
      SELECT s.date, s.symbol, s.orig_time,
             MAX(CASE WHEN s.entry_type='BID'   AND s.level=1 THEN s.px END) AS bb,
             MIN(CASE WHEN s.entry_type='OFFER' AND s.level=1 THEN s.px END) AS ba
      FROM read_parquet('{SNAP}') s, dates d
      WHERE s.market='REG' AND s.date=d.date AND s.phase='CONTINUOUS_AUCTION'
      GROUP BY s.date, s.symbol, s.orig_time
    )
    -- one mid per symbol per MINUTE: the last two-sided touch in that minute.
    -- Minute bars give every symbol a common time grid, which is what makes a
    -- cross-symbol regression possible at all.
    SELECT date, symbol, date_trunc('minute', orig_time) AS m,
           LAST(( bb + ba ) / 2 ORDER BY orig_time) AS mid
    FROM touch
    WHERE bb>0 AND ba>0 AND ba>=bb
    GROUP BY date, symbol, m
    """
    bars = con.execute(q4).df()
    # wide: one column per symbol, one row per minute
    px = bars.pivot_table(index=["date", "m"], columns="symbol", values="mid")
    # the proxy must be present or there is nothing to regress against
    if proxy not in px.columns:
        print(f"  {proxy} has no minute bars -- cannot regress")
    else:
        # accumulate one row per (symbol, horizon)
        rows = []
        # walk each horizon so the Epps decay is visible rather than hidden
        for H in HORIZONS_MIN:
            # H-minute log returns, taken WITHIN a date so an overnight gap never
            # enters a return
            r = np.log(px).groupby(level="date").diff(H)
            # the market leg
            rm = r[proxy]
            # every other symbol is a candidate hedge target
            for sym in r.columns:
                # the ETF against itself is not informative
                if sym == proxy:
                    continue
                # align and drop minutes where either leg is missing
                pair = pd.concat([r[sym], rm], axis=1).dropna()
                # too few observations to say anything
                if len(pair) < 200:
                    continue
                y = pair.iloc[:, 0].values
                x = pair.iloc[:, 1].values
                # the market leg must actually move, or beta is undefined
                if x.std() == 0 or y.std() == 0:
                    continue
                # ordinary least squares slope: cov(y,x)/var(x)
                beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
                # correlation, and R-squared as its square
                rho = np.corrcoef(y, x)[0, 1]
                rows.append({"symbol": sym, "horizon_min": H, "n": len(pair),
                             "beta": round(float(beta), 3),
                             "r2": round(float(rho ** 2), 4)})
        # assemble
        fit = pd.DataFrame(rows)
        if len(fit):
            # the headline: median explanatory power at each horizon
            print("\n  median R-squared across all names, by horizon "
                  "(the fraction of a name's move the ETF explains):")
            for H in HORIZONS_MIN:
                sub = fit[fit.horizon_min == H]
                print(f"    {H:3d} min: median R2 = {sub.r2.median():.4f}  "
                      f"| 90th pct = {sub.r2.quantile(0.90):.4f}  "
                      f"| names = {len(sub)}")
            # the names the ETF explains best, at the longest horizon
            longest = fit[fit.horizon_min == max(HORIZONS_MIN)]
            print(f"\n  best-explained names at {max(HORIZONS_MIN)} min:")
            print(longest.nlargest(15, "r2")[
                ["symbol", "beta", "r2", "n"]].to_string(index=False))
            # write it out for the next script
            out = Path(RESULTS) / "etf_hedge_fit_20260915.csv"
            if not out.exists():
                fit.to_csv(out, index=False)
                print(f"\n  wrote {out}")
            else:
                print(f"\n  NOT written, already exists: {out}")
            # the interpretation, stated so the number is not over-read
            med = fit[fit.horizon_min == max(HORIZONS_MIN)].r2.median()
            print(f"\n  READ THIS AS: hedging with {proxy} can remove at most about "
                  f"{100*med:.1f}% of the")
            print("  typical name's price move. Everything else is name-specific and a")
            print("  market hedge cannot touch it. Compare that fraction against the")
            print("  hedge's cost before designing any tolerance band.")
        else:
            print("  no symbol had enough overlapping minutes to fit")
else:
    print("\nSECTION 4 -- skipped, no ETF has a quotable book to regress against")

# =============================================================================
# SECTION 5 -- the fills schema, so the markout regression can be written
# =============================================================================
# The decisive number for the whole ETF rung is not in this script: it is what
# fraction of the book's MARKOUT is market beta rather than name-specific
# information. That needs the per-fill markout data. Rather than guess at column
# names -- which has already cost two runs this week -- print the schema.
print("\nSECTION 5 -- fills schema (input for the markout-vs-market regression)")
if FILLS is None:
    print("  config_pk has no FILLS_DIR; name the fills path and re-run")
else:
    # the fills directory as config_pk reports it
    fp = Path(FILLS)
    print(f"  {fp}")
    if not fp.exists():
        print("  directory does not exist -- has fill_attribution.py been run?")
    else:
        # the first parquet found, just to read its column list
        cand = sorted(fp.rglob("*.parquet"))[:1]
        if not cand:
            print("  no parquet files found under it")
        else:
            # describe rather than load: we only want names and types
            desc = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{cand[0]}')").df()
            print(f"  from {cand[0].name}:")
            print(desc[["column_name", "column_type"]].to_string(index=False))
            print(f"\n  total fill files: {len(list(fp.rglob('*.parquet')))}")
