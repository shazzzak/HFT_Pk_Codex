# probe_market_beta.py -- is there market beta in the book at all, and how much
# of the MARKOUT would a portfolio overlay hedge actually remove?
#
# WHY THIS EXISTS. probe_etf_hedge.py killed the ETF rung on three grounds: cost
# (best real ETF 16.21 bps round trip, worse than the median share book at 13.79),
# explanatory power (median R2 0.0060 at 15 min), and horizon (markout is a
# 5-SECOND number; the tightest ETF trades once per ~44 s).
#
# But that R2 was measured against MZNPETF -- an Islamic index ETF that excludes
# conventional banks, and therefore a poor market proxy. The tell is in its own
# output: UBLPETF vs MZNPETF came out at R2 = 0.19. Two broad ETFs on the same
# exchange sharing 19% of their 15-minute variance is a fact about stale, wide,
# thinly-traded ETF prices, not a fact about the market.
#
# So two questions were conflated and only one was answered:
#   ANSWERED     "should I hedge with those ETFs?"  No. You trade the instrument
#                and you get the instrument's staleness, so the measured R2 is
#                the right number there.
#   UNANSWERED   "is there market beta in my book at all?"  If there is, the
#                overlay could be a basket of liquid SHARES, which cost 5.30-13.79
#                bps and actually trade.
# This script answers the second one, and then answers the only question that
# really decides the overlay: how much of the MARKOUT is common-factor.
#
# THE TWO MEASUREMENTS, AND WHY THE SECOND IS THE ONE THAT MATTERS
#   SECTION 3  R2 of each name's RETURN on a synthetic market factor, at
#              5s / 30s / 60s / 300s. Tells you whether a common factor exists.
#   SECTION 4  R2 of each fill's MARKOUT on the factor return over the SAME
#              5-second window. This is the hedgeable fraction of the thing you
#              are actually trying to remove. A book can have high return-beta
#              and near-zero markout-beta at the same time: if you are picked off
#              on name-specific information, the adverse move is idiosyncratic
#              even though the name's daily return tracks the market closely.
#              SECTION 4 IS THE DECIDING NUMBER. Section 3 is context for it.
#
# METHOD NOTES, so the numbers are not over-read:
#   - The factor is built from the most liquid names by traded notional, as an
#     equal-weighted mean of their log mid returns. Equal-weighted rather than
#     cap-weighted because market caps are not in the parsed store; flagged as a
#     simplification -- the production version weights by free-float cap.
#   - LEAVE-ONE-OUT is enforced: a name is never regressed on a factor that
#     contains it. Without this, a factor constituent's R2 is inflated purely by
#     self-inclusion, and the largest names (the ones you most want to hedge)
#     are inflated most.
#   - Returns use MID, never trade price. Bid-ask bounce induces negative
#     autocorrelation that contaminates any covariance estimate.
#   - The EPPS effect biases short-horizon correlation DOWNWARD when the two legs
#     trade at different rates. The 5s figure is therefore a floor. Read the
#     decay across horizons rather than any single number -- but note that 5s is
#     also the horizon that matters, so a low 5s R2 is a real obstacle even when
#     it is partly bias.
#   - Overnight gaps never enter a return: differencing is done within a date.
#
# Read-only. 10-20 minutes. Run: caffeinate -is python probe_market_beta.py
from pathlib import Path
from datetime import datetime
import duckdb, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- PATHS from config_pk; never a literal in this file ----------------------
try:
    # the project's central path module
    import config_pk
    # the parsed store
    PARSED = str(config_pk.PARSED_ROOT)
    # the results root
    RESULTS = Path(config_pk.RESULTS_ROOT)
    # where fill_attribution.py writes its per-fill output
    FILLS = Path(config_pk.FILLS_DIR)
except Exception as _e:
    # a wrong store is worse than a crash
    raise SystemExit("probe_market_beta: could not import paths from config_pk. "
                     "Run from existing_mm_live/. Original: %r" % _e)
# fail immediately with the path named
if not Path(PARSED).is_dir():
    raise SystemExit(f"PARSED not found: {PARSED}")
print(f"parsed store: {PARSED}")

# RUN STAMP, YYYYMMDD_HHMM, on every output this script writes, so a re-run never
# collides with an earlier one and never has to refuse to write
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
# how many recent trading dates to sample
N_DAYS = 10
# how many names form the synthetic market factor (most liquid by notional)
N_FACTOR = 40
# the grid the factor is built on, in seconds -- 5s because that is the markout
# horizon (HORIZON_MS = 5000 in fill_attribution.py)
GRID_SEC = 5
# return horizons in seconds; 5 is the one that matters, the rest show the decay
HORIZONS_SEC = [5, 30, 60, 300]
# the markout horizon, from fill_attribution.py HORIZON_MS
MARKOUT_MS = 5000
# minimum paired observations before a regression is reported
MIN_OBS = 500
# the share book's gross markout budget per round trip, bps
EDGE_BPS = 2.67

con = duckdb.connect(); pd.set_option("display.width", 200, "display.max_columns", 30)
TRADES = f"{PARSED}/trades/date=*/*.parquet"
SNAP = f"{PARSED}/ob_snapshot/date=*/*.parquet"

# =============================================================================
# SECTION 0 -- WHICH DATES? Anchor to the FILLS, not to the store's latest.
# =============================================================================
# The first version of this script sampled "the most recent N dates in the parsed
# store" and got 0 fills in Section 4. The store runs to 2026-06-30; the backtest
# fills are from the 197-session run and start in 2025. The two windows did not
# overlap at all, so the join was empty. Section 4 is the whole point of the
# script, so the dates must be chosen where the fills ARE.
fill_dates = []
try:
    # the distinct dates present in the fill output, most recent first
    fd = con.execute(f"""
        SELECT DISTINCT CAST(date AS VARCHAR) AS d
        FROM read_parquet('{FILLS}/**/*.parquet', hive_partitioning=1,
                          union_by_name=1)
        ORDER BY d DESC
    """).df()
    fill_dates = fd.d.tolist()
except Exception as _e:
    # say so rather than silently falling through to the wrong window
    print(f"  (could not read dates from fills: {_e!r})")
# take the most recent N that have fills
if fill_dates:
    DATES = sorted(fill_dates[:N_DAYS])
    print(f"\nSECTION 0 -- dates anchored to the FILLS: {len(DATES)} dates, "
          f"{DATES[0]} to {DATES[-1]}")
    print(f"  ({len(fill_dates)} dates have fills in total)")
else:
    # no fills at all: Section 3 can still run, Section 4 cannot
    DATES = sorted(con.execute(f"""
        SELECT DISTINCT CAST(date AS VARCHAR) AS d FROM read_parquet('{TRADES}')
        WHERE market='REG' ORDER BY d DESC LIMIT {N_DAYS}
    """).df().d.tolist())
    print(f"\nSECTION 0 -- NO FILLS FOUND. Falling back to the store's most "
          f"recent {len(DATES)} dates; Section 4 will be skipped.")
# the SQL literal every query below shares, so no two sections can disagree
D_SQL = ",".join(f"'{d}'" for d in DATES)

# =============================================================================
# SECTION 1 -- pick the factor constituents by traded notional
# =============================================================================
q1 = f"""
WITH dates AS (
  -- exactly the dates chosen in Section 0
  SELECT DISTINCT date FROM read_parquet('{TRADES}')
  WHERE market='REG' AND CAST(date AS VARCHAR) IN ({D_SQL})
)
SELECT t.symbol,
       COUNT(DISTINCT t.date) AS days,
       SUM(t.price*t.qty) / COUNT(DISTINCT t.date) / 1e6 AS notional_m_day
FROM read_parquet('{TRADES}') t, dates d
WHERE t.market='REG' AND t.date=d.date
GROUP BY t.symbol
-- a constituent must be present on essentially every sampled day, or the factor
-- changes composition from day to day and its variance is not comparable
HAVING COUNT(DISTINCT t.date) >= {N_DAYS} - 1
ORDER BY notional_m_day DESC LIMIT {N_FACTOR}
"""
fac = con.execute(q1).df()
print(f"\nSECTION 1 -- synthetic market factor: top {len(fac)} names by traded "
      f"notional over {N_DAYS} dates")
print(fac.head(15).to_string(index=False))
print(f"  ... total daily notional in the factor: "
      f"{fac.notional_m_day.sum():,.0f} m PKR")
# the constituent list for the SQL and for leave-one-out
FACTOR_SYMS = fac.symbol.tolist()
FAC_SQL = ",".join(f"'{s}'" for s in FACTOR_SYMS)

# =============================================================================
# SECTION 2 -- build the 5-second mid grid
# =============================================================================
q2 = f"""
WITH dates AS (
  -- the same dates as Section 1, so the factor and the fills share a window
  SELECT DISTINCT date FROM read_parquet('{TRADES}')
  WHERE market='REG' AND CAST(date AS VARCHAR) IN ({D_SQL})
),
touch AS (
  -- L1 bid/ask per snapshot message, continuous phase only, every symbol
  SELECT s.date, s.symbol, s.orig_time,
         MAX(CASE WHEN s.entry_type='BID'   AND s.level=1 THEN s.px END) AS bb,
         MIN(CASE WHEN s.entry_type='OFFER' AND s.level=1 THEN s.px END) AS ba
  FROM read_parquet('{SNAP}') s, dates d
  WHERE s.market='REG' AND s.date=d.date AND s.phase='CONTINUOUS_AUCTION'
  GROUP BY s.date, s.symbol, s.orig_time
)
-- one mid per symbol per GRID_SEC bucket: the LAST two-sided touch in the
-- bucket. Bucketing gives every symbol a common clock, which is the only way a
-- cross-symbol factor can be formed at all.
SELECT date, symbol,
       -- floor the timestamp to the grid
       to_timestamp(floor(epoch(orig_time) / {GRID_SEC}) * {GRID_SEC}) AS t,
       LAST((bb + ba) / 2 ORDER BY orig_time) AS mid
FROM touch
WHERE bb > 0 AND ba > 0 AND ba >= bb
GROUP BY date, symbol, t
"""
print(f"\nSECTION 2 -- building the {GRID_SEC}-second mid grid "
      f"(this is the slow step)")
bars = con.execute(q2).df()
print(f"  {len(bars):,} symbol-bars, {bars.symbol.nunique()} symbols, "
      f"{bars.date.nunique()} dates")
# wide frame: rows are (date, t), columns are symbols
px = bars.pivot_table(index=["date", "t"], columns="symbol", values="mid").sort_index()
# forward-fill WITHIN a date only: a stale mid is the right value to carry, but
# yesterday's close must never leak into today's first bucket
px = px.groupby(level="date").ffill()

# =============================================================================
# SECTION 3 -- does a common factor exist? R2 of returns, leave-one-out
# =============================================================================
print(f"\nSECTION 3 -- R2 of each name's return on the synthetic factor, "
      f"leave-one-out, horizons {HORIZONS_SEC}s")
# the constituents actually present in the grid
present = [s for s in FACTOR_SYMS if s in px.columns]
# log prices once; every horizon differences the same matrix
lp = np.log(px)
# accumulate one row per (symbol, horizon)
rows = []
# walk the horizons so the Epps decay is visible
for H in HORIZONS_SEC:
    # how many grid steps that horizon is
    k = max(1, H // GRID_SEC)
    # returns within a date, so no overnight gap enters
    r = lp.groupby(level="date").diff(k)
    # the factor's constituent returns
    rf = r[present]
    # the equal-weighted factor: mean across constituents present in that bucket
    f_all = rf.mean(axis=1)
    # how many constituents contributed to each bucket, for leave-one-out
    n_all = rf.notna().sum(axis=1)
    # every symbol in the book is a candidate
    for sym in r.columns:
        # LEAVE-ONE-OUT: rebuild the factor without this name when it is a
        # constituent, otherwise self-inclusion inflates its own R2
        if sym in present:
            # the factor's total, minus this name's contribution
            tot = f_all * n_all
            # this name's own return, zero where it is missing
            own = rf[sym].fillna(0.0)
            # how many others contributed
            n_oth = n_all - rf[sym].notna().astype(int)
            # the leave-one-out factor; undefined where nobody else contributed
            fk = (tot - own) / n_oth.replace(0, np.nan)
        else:
            # not a constituent, so the full factor is already leave-one-out
            fk = f_all
        # align and drop buckets where either leg is missing
        pair = pd.concat([r[sym].rename("y"), fk.rename("x")], axis=1).dropna()
        # too few observations to say anything
        if len(pair) < MIN_OBS:
            continue
        y = pair.y.to_numpy(); x = pair.x.to_numpy()
        # a degenerate leg makes beta undefined
        if x.std() == 0 or y.std() == 0:
            continue
        # OLS slope
        beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
        # correlation, squared for R2
        rho = float(np.corrcoef(y, x)[0, 1])
        rows.append({"symbol": sym, "horizon_s": H, "n": len(pair),
                     "beta": round(beta, 3), "r2": round(rho ** 2, 4)})
# assemble
fit = pd.DataFrame(rows)
if len(fit):
    print("\n  median R2 across all names, by horizon:")
    # one line per horizon, so the decay is explicit
    for H in HORIZONS_SEC:
        sub = fit[fit.horizon_s == H]
        print(f"    {H:4d}s: median R2 = {sub.r2.median():.4f} | "
              f"75th = {sub.r2.quantile(.75):.4f} | "
              f"90th = {sub.r2.quantile(.90):.4f} | names = {len(sub)}")
    # the 5-second row is the one that matches the markout horizon
    at5 = fit[fit.horizon_s == 5]
    print(f"\n  best-explained names at the {GRID_SEC}s markout horizon:")
    print(at5.nlargest(15, "r2")[["symbol", "beta", "r2", "n"]].to_string(index=False))
    # persist for the next step, timestamped so a re-run keeps both
    o1 = RESULTS / f"market_beta_returns_{STAMP}.csv"
    fit.to_csv(o1, index=False)
    print(f"\n  wrote {o1}")
else:
    print("  no symbol had enough paired buckets to fit")

# =============================================================================
# SECTION 4 -- THE DECIDING NUMBER: how much of MARKOUT is common-factor?
# =============================================================================
# Section 3 says whether a factor exists. This says whether hedging it removes
# any of the thing you are actually trying to remove. Each fill carries a
# side-signed markout in bps over MARKOUT_MS. Regress that on the side-signed
# factor return over the SAME window. The R2 is the hedgeable fraction.
print(f"\nSECTION 4 -- markout ({MARKOUT_MS} ms) regressed on the factor return "
      f"over the same window")
# the fill files for the sampled dates
fill_files = sorted(FILLS.rglob("*.parquet"))
if not fill_files:
    print(f"  no fill parquets under {FILLS} -- run fill_attribution.py first")
else:
    # Section 0 already resolved the shared window; reuse it rather than
    # re-deriving it from the grid, so a name that dropped out of the grid
    # cannot silently narrow the fill load
    # load only the columns needed, only for the sampled dates
    fq = f"""
    SELECT symbol, date, ts, side, markout
    FROM read_parquet('{FILLS}/**/*.parquet', hive_partitioning=1, union_by_name=1)
    WHERE CAST(date AS VARCHAR) IN ({D_SQL}) AND markout IS NOT NULL
    """
    try:
        fills = con.execute(fq).df()
    except Exception as e:
        print(f"  could not read fills: {e}")
        fills = pd.DataFrame()
    print(f"  {len(fills):,} fills on the sampled dates")
    if len(fills):
        # the factor return over exactly the markout window, on the grid
        k = max(1, MARKOUT_MS // 1000 // GRID_SEC)
        # FORWARD return: from this bucket to MARKOUT_MS later, within a date
        fwd = lp[present].groupby(level="date").diff(k).shift(-k).mean(axis=1)
        # a lookup frame keyed the same way the fills will be
        fdf = fwd.rename("fac_fwd").reset_index()
        # fills carry ts in milliseconds; floor it to the same grid
        fills["t"] = pd.to_datetime(
            (fills.ts // (GRID_SEC * 1000)) * (GRID_SEC * 1000), unit="ms", utc=True)
        # the grid's own timestamps, normalised for the join
        fdf["t"] = pd.to_datetime(fdf["t"], utc=True)
        # dates as strings on both sides
        fills["date"] = fills.date.astype(str); fdf["date"] = fdf.date.astype(str)
        # attach the factor's forward move to each fill
        m = fills.merge(fdf, on=["date", "t"], how="inner").dropna(
            subset=["fac_fwd", "markout"])
        print(f"  {len(m):,} fills matched to a factor bucket")
        if len(m) >= MIN_OBS:
            # +1 for a buy, -1 for a sell; markout is already side-signed, so the
            # factor return must be signed the same way for the regression to
            # mean "did the market move against my position"
            sgn = np.where(m.side > 0, 1.0, -1.0)
            # the signed factor move over the markout window, in bps
            x = sgn * m.fac_fwd.to_numpy() * 1e4
            # the fill's own signed markout, already in bps
            y = m.markout.to_numpy()
            # keep finite pairs only
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            # OLS slope and R2
            beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
            rho = float(np.corrcoef(y, x)[0, 1])
            print(f"\n  n = {len(x):,} fills")
            print(f"  mean markout          = {y.mean():+.4f} bps")
            print(f"  mean signed factor    = {x.mean():+.4f} bps")
            print(f"  beta (markout on factor) = {beta:+.4f}")
            print(f"  R2                    = {rho**2:.4f}")
            # the part a perfect overlay could remove, and the part it could not
            removable = beta * x.mean()
            print(f"\n  MARKOUT DECOMPOSITION, per fill:")
            print(f"    explained by the market factor : {removable:+.4f} bps")
            print(f"    left over (name-specific)      : {y.mean()-removable:+.4f} bps")
            print(f"    common-factor share of variance: {100*rho**2:.2f}%")
            print(f"\n  READ THIS AS: a PERFECT, FREE overlay hedge removes "
                  f"{100*rho**2:.2f}% of the")
            print(f"  variance of your adverse selection and {abs(removable):.4f} bps of its")
            print(f"  mean. Everything else is name-specific and no market hedge")
            print(f"  touches it. The cheapest crossing available is 5.30 bps "
                  f"(EFERT shares),")
            print(f"  against a {EDGE_BPS} bps edge -- so the overlay has to clear that too.")
            # ---- the picture -------------------------------------------------
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor="white")
            # panel 1: the scatter the regression actually saw, thinned to plot
            idx = np.random.default_rng(0).choice(
                len(x), size=min(len(x), 20000), replace=False)
            a1.scatter(x[idx], y[idx], s=4, alpha=.18, color="#2f6fb5",
                       edgecolor="none")
            # the fitted line across the plotted range
            xs = np.linspace(np.percentile(x, .5), np.percentile(x, 99.5), 50)
            a1.plot(xs, beta * xs + (y.mean() - beta * x.mean()),
                    color="#b3261e", lw=2)
            a1.set_xlabel("signed market-factor move over the 5s window, bps", fontsize=10)
            a1.set_ylabel("fill markout, bps", fontsize=10)
            a1.set_title(f"Markout vs market factor\nbeta {beta:+.3f}, "
                         f"R2 {rho**2:.4f}, n {len(x):,}", fontsize=11.5, loc="left")
            a1.set_xlim(np.percentile(x, .5), np.percentile(x, 99.5))
            a1.set_ylim(np.percentile(y, .5), np.percentile(y, 99.5))
            a1.grid(True, color="#dfe3e8", lw=.8); a1.set_axisbelow(True)
            for s in ("top", "right"): a1.spines[s].set_visible(False)
            # panel 2: how the return-R2 decays with horizon, from Section 3
            if len(fit):
                med = [fit[fit.horizon_s == H].r2.median() for H in HORIZONS_SEC]
                p90 = [fit[fit.horizon_s == H].r2.quantile(.90) for H in HORIZONS_SEC]
                a2.plot(HORIZONS_SEC, med, "o-", color="#2f6fb5", lw=2, label="median name")
                a2.plot(HORIZONS_SEC, p90, "s--", color="#c4762a", lw=2, label="90th pct")
                a2.axvline(MARKOUT_MS / 1000, color="#b3261e", lw=1.8)
                a2.annotate("markout horizon", xy=(MARKOUT_MS/1000, max(p90)*.92),
                            xytext=(6, 0), textcoords="offset points",
                            fontsize=9.5, color="#b3261e", weight="bold")
                a2.set_xscale("log"); a2.set_xticks(HORIZONS_SEC)
                a2.set_xticklabels([str(h) for h in HORIZONS_SEC])
                a2.set_xlabel("return horizon, seconds (log)", fontsize=10)
                a2.set_ylabel("R2 vs the market factor", fontsize=10)
                a2.set_title("Common-factor share rises with horizon\n"
                             "but the markout you must hedge is at 5s",
                             fontsize=11.5, loc="left")
                a2.legend(fontsize=9.5, frameon=False)
                a2.grid(True, color="#dfe3e8", lw=.8); a2.set_axisbelow(True)
                for s in ("top", "right"): a2.spines[s].set_visible(False)
            plt.tight_layout()
            # timestamped: every run keeps its own figure, none is overwritten
            o2 = RESULTS / f"market_beta_{STAMP}.png"
            plt.savefig(o2, dpi=150, facecolor="white")
            print(f"\n  wrote {o2}")
        else:
            print(f"  only {len(m)} fills matched a factor bucket -- too few to fit")
