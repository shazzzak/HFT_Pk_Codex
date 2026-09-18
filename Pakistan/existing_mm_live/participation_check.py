# ============================================================================
# participation_check.py -- HOW BIG AM I IN THIS MARKET?
# ----------------------------------------------------------------------------
# THE QUESTION: the backtest assumes my orders do not move the price. That is
# only defensible while my traded value is a SMALL share of the name's traded
# value on the day. This script measures that share and plots it four ways.
#
# READ THIS BEFORE TRUSTING A NUMBER -- three conventions decide the answer and
# all three are set explicitly at the top of the file:
#
#   1. MY TRADED VALUE. The PERNAME parquet stores `opened_notional` -- the
#      value of positions OPENED. A round trip trades twice (the open leg and
#      the close leg), so my total traded value is about 2x that. LEG_MULT
#      makes this explicit. It is EXACT when the day ends flat and slightly
#      over-counted when inventory is marked out rather than traded out.
#      The exact version needs a shares/notional column emitted by the runner
#      -- see the note at the bottom of this file.
#
#   2. WHOSE VOLUME IS IN THE DENOMINATOR. In a backtest my fills are
#      counterfactual: they would have DISPLACED someone else's fill, not added
#      to the day's volume. So `pov_excl = mine / market` is the honest and
#      CONSERVATIVE (higher) figure, and `pov_incl = mine / (market + mine)` is
#      what a live POV monitor would print. Both are computed. The truth sits
#      between them and they only diverge once the number is already large.
#
#   3. WHICH PRINTS COUNT AS MARKET VOLUME. NDM (negotiated blocks) and ODD_LOT
#      are excluded, matching run_daily_stats.py -- an NDM block averages ~300x
#      a REG trade and would flatter every ratio. Auction prints are excluded
#      too by default, because the strategy only quotes in the continuous
#      session, so continuous volume is the like-for-like denominator.
#
# USAGE: python participation_check.py --run
# ============================================================================

# command-line flags
import argparse
# filesystem paths
import pathlib
# numerics
import numpy as np
# dataframes
import pandas as pd
# plotting, headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# central paths -- never hardcode a root (config_pk is the single source of truth)
from config_pk import PARSED_ROOT, RESULTS_ROOT

# --------------------------------------------------------------- CONFIG ----
# the per-name-day output of the 113-name corrected run
PERNAME = RESULTS_ROOT / "universe_expand_c1_PERNAME_20260918_1230.parquet"
# which arm to measure; the shipped book is the 0.15 lean, so that is the default
ARM = "QT_2t@15"
# where the charts go
OUT_DIR = RESULTS_ROOT / "diagnostics"
# open leg + close leg = 2 legs per round trip (see convention 1 above)
LEG_MULT = 2.0
# exclude auction prints from the market denominator (see convention 3 above)
EXCLUDE_AUCTION = True
# the participation level above which the no-impact assumption stops being safe
WARN_PCT = 5.0
# the level above which the backtest should simply not be believed for that cell
ALARM_PCT = 15.0
# Okabe-Ito colourblind-safe hues, assigned in FIXED order and never cycled
C_BLUE, C_VERM, C_GREEN, C_PURPLE, C_GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6B7280"


# ------------------------------------------------------- MARKET VOLUME -----
def market_value_by_symbol_day(dates, symbols):
    """Total traded VALUE per (date, symbol) from the parsed store.

    One DuckDB query per date over that date's hive partition. Returns a tidy
    frame: date, symbol, mkt_value, mkt_shares, mkt_trades.
    """
    # duckdb is imported lazily so the file still imports without it
    import duckdb
    # one in-memory connection reused across dates
    con = duckdb.connect()
    # the symbols we care about, as a SQL list literal
    sym_list = ",".join("'" + s.replace("'", "''") + "'" for s in symbols)
    # collected per-date frames
    out = []
    # walk every date in the run
    for i, date in enumerate(dates, 1):
        # the hive partition for this date's trades
        glob = str(PARSED_ROOT / "trades" / f"date={date}" / "*.parquet")
        # skip a date with no partition rather than crash the whole pass
        if not list((PARSED_ROOT / "trades" / f"date={date}").glob("*.parquet")):
            print(f"  [skip] no trades partition for {date}")
            continue
        # NDM/ODD_LOT exclusion matches run_daily_stats.py -- and it must be an
        # exclusion, not market='REG', because STOCK_DEL_FUT / STOCK_CS_FUT are
        # also `market` values and an equality filter would delete all futures
        where = ["symbol IS NOT NULL", "market NOT IN ('NDM','ODD_LOT')",
                 f"symbol IN ({sym_list})"]
        # the strategy never quotes in the auction, so exclude auction prints
        if EXCLUDE_AUCTION:
            where.append("initiator <> 'AUCTION'")
        # aggregate value, shares and print count per symbol for this date
        q = (f"SELECT '{date}' AS date, symbol, "
             f"SUM(price * qty) AS mkt_value, SUM(qty) AS mkt_shares, "
             f"COUNT(*) AS mkt_trades "
             f"FROM read_parquet('{glob}') "
             f"WHERE {' AND '.join(where)} GROUP BY symbol")
        # run it and keep the frame
        out.append(con.execute(q).df())
        # a heartbeat every 20 dates so a long pass is not silent
        if i % 20 == 0:
            print(f"  market volume {i}/{len(dates)} dates")
    # one frame for the whole panel
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------- MY VOLUME ---
def my_value_by_symbol_day(arm=ARM):
    """My traded VALUE per (date, symbol) for one arm, from the run output."""
    # the per-name-day table, which is at (date, symbol, arm, bucket) grain
    df = pd.read_parquet(PERNAME)
    # keep only the arm being measured
    df = df[df["throttle"] == arm]
    # collapse the four intraday buckets away to one row per symbol-day
    g = (df.groupby(["date", "symbol"], as_index=False)
           .agg(opened_notional=("opened_notional", "sum"), fills=("fills", "sum")))
    # open leg + close leg -> total traded value (see convention 1)
    g["my_value"] = g["opened_notional"] * LEG_MULT
    # return the tidy frame
    return g[["date", "symbol", "my_value", "opened_notional", "fills"]]


# ------------------------------------------------------------- THE RATIOS --
def build(arm=ARM):
    """Join my volume to market volume and compute both participation ratios."""
    # my side first, because it defines the names and dates that matter
    mine = my_value_by_symbol_day(arm)
    # the dates and symbols actually present in the run
    dates = sorted(mine["date"].unique())
    symbols = sorted(mine["symbol"].unique())
    # announce the scope so the log records what was measured
    print(f"  my side: {len(mine):,} symbol-days | {len(symbols)} names | {len(dates)} dates")
    # the market side for exactly that scope
    mkt = market_value_by_symbol_day(dates, symbols)
    # inner join: a symbol-day with no market row cannot produce a ratio
    df = mine.merge(mkt, on=["date", "symbol"], how="inner")
    # say how many rows were lost to the join, so coverage loss is never silent
    print(f"  joined: {len(df):,} rows ({len(mine) - len(df):,} of my rows had no market row)")
    # backtest convention: my fills displace other fills, so the market total
    # already represents the whole day -- this is the conservative ratio
    df["pov_excl_pct"] = np.where(df["mkt_value"] > 0,
                                  df["my_value"] / df["mkt_value"] * 100.0, np.nan)
    # live-monitor convention: my volume is additive on top of the market's
    df["pov_incl_pct"] = np.where((df["mkt_value"] + df["my_value"]) > 0,
                                  df["my_value"] / (df["mkt_value"] + df["my_value"]) * 100.0,
                                  np.nan)
    # drop symbol-days with no measurable market volume
    df = df[df["pov_excl_pct"].notna()].copy()
    # a real date type makes the time-series plots order themselves correctly
    df["date"] = pd.to_datetime(df["date"])
    # sorted, tidy
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


# ------------------------------------------------------ DAILY AGGREGATES ---
def daily_stats(df):
    """Collapse the per-name-day ratios into one row per date, five ways."""
    # work on the conservative ratio
    r = "pov_excl_pct"
    # group once, then build each statistic from the same groups
    rows = []
    # walk each trading date
    for date, g in df.groupby("date"):
        # my total value and the market's total value on this date
        tot_mine = g["my_value"].sum(); tot_mkt = g["mkt_value"].sum()
        # #3a WEIGHTED BY MARKET VALUE: this is algebraically the PORTFOLIO
        # FOOTPRINT -- sum(mine) / sum(market) -- i.e. how big I am in the book
        # as a whole on this day
        w_mkt = tot_mine / tot_mkt * 100.0 if tot_mkt > 0 else np.nan
        # #3b WEIGHTED BY MY OWN VALUE: the participation rate my P&L actually
        # EXPERIENCES, because it over-weights the names I traded most. This is
        # the number that matters for impact on my own fills, and it is always
        # >= the footprint above
        w_mine = ((g[r] * g["my_value"]).sum() / tot_mine) if tot_mine > 0 else np.nan
        # #4 the plain median across names -- the typical name on this day
        med = g[r].median()
        # the tail is the risk, not the middle: one name at 40% breaks the
        # no-impact assumption for that name however low the median is
        p90 = g[r].quantile(0.90); p99 = g[r].quantile(0.99); mx = g[r].max()
        # the count of names over each policy line
        n_warn = int((g[r] > WARN_PCT).sum()); n_alarm = int((g[r] > ALARM_PCT).sum())
        # one tidy row per date
        rows.append(dict(date=date, n_names=len(g), footprint_pct=w_mkt,
                         wavg_mine_pct=w_mine, median_pct=med, p25=g[r].quantile(0.25),
                         p75=g[r].quantile(0.75), p90=p90, p99=p99, max_pct=mx,
                         n_over_warn=n_warn, n_over_alarm=n_alarm,
                         my_value=tot_mine, mkt_value=tot_mkt))
    # a frame ordered by date
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------- THE CHARTS --
def charts(df, dly, out_dir, arm):
    """Four charts, one file each, plus a fifth that ranks the risk by name."""
    # make sure the directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    # a recessive house style: thin marks, light grid, no chartjunk
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 10, "axes.titlesize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.frameon": False, "legend.fontsize": 9,
                         "font.size": 10})
    # the ratio column every chart reads
    r = "pov_excl_pct"

    # ---- CHART 1: the distribution of per-name-day participation -----------
    # a histogram, because #1 is a population of ~22,000 numbers, not a series
    fig, ax = plt.subplots(figsize=(9, 4.6))
    # clip at a sane upper edge so one 300% outlier does not eat the axis
    v = df[r].clip(upper=60.0)
    # log-spaced bins: participation spans three orders of magnitude
    bins = np.logspace(np.log10(max(1e-3, df[r][df[r] > 0].min())), np.log10(60.0), 60)
    # the bars themselves
    ax.hist(v[v > 0], bins=bins, color=C_BLUE, edgecolor="white", linewidth=0.4)
    # log x, because the distribution is heavily right-skewed
    ax.set_xscale("log")
    # reserve a clear band above the bars so no annotation lands on top of data
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.22)
    # the y position every annotation sits at, inside that reserved band
    ylab = top * 1.10
    # the median, labelled to the LEFT of its line so it cannot collide with 5%
    med = df[r].median()
    ax.axvline(med, color=C_GREEN, linewidth=1.6)
    ax.text(med, ylab, f"median {med:.2f}%  ", color=C_GREEN, fontsize=9,
            va="center", ha="right")
    # the two policy lines, labelled directly to the RIGHT rather than in a legend
    for x, c, lab in ((WARN_PCT, C_VERM, f"{WARN_PCT:.0f}% warn"),
                      (ALARM_PCT, C_PURPLE, f"{ALARM_PCT:.0f}% alarm")):
        ax.axvline(x, color=c, linewidth=1.6, linestyle="--")
        ax.text(x, ylab, "  " + lab, color=c, fontsize=9, va="center", ha="left")
    # axis labels carry the units so the chart reads standalone
    ax.set_xlabel("my traded value as % of the name's traded value, that day (log scale)")
    ax.set_ylabel("symbol-days")
    ax.set_title(f"1. Participation per name-day — {arm}, {len(df):,} symbol-days")
    fig.tight_layout(); fig.savefig(out_dir / "participation_1_distribution.png"); plt.close(fig)

    # ---- CHART 2: the time series, as percentile bands ---------------------
    # plotting 113 lines would be unreadable, so show the spread of the
    # population each day: the band is where the names are, the line is typical
    fig, ax = plt.subplots(figsize=(11, 4.6))
    # the interquartile band, drawn first so the lines sit on top of it
    ax.fill_between(dly["date"], dly["p25"], dly["p75"], color=C_BLUE, alpha=0.18,
                    linewidth=0, label="25th–75th percentile of names")
    # the median name
    ax.plot(dly["date"], dly["median_pct"], color=C_BLUE, linewidth=2.0, label="median name")
    # the 90th percentile -- the first place impact shows up
    ax.plot(dly["date"], dly["p90"], color=C_VERM, linewidth=1.4, label="90th percentile")
    # the worst name on the day -- thin, because it is a tail marker not a trend
    ax.plot(dly["date"], dly["max_pct"], color=C_GREY, linewidth=0.9, alpha=0.8, label="worst name")
    # the policy lines again, for continuity with chart 1
    ax.axhline(WARN_PCT, color=C_VERM, linewidth=1.0, linestyle="--")
    ax.axhline(ALARM_PCT, color=C_PURPLE, linewidth=1.0, linestyle="--")
    # log y for the same skew reason
    ax.set_yscale("log")
    ax.set_ylabel("% of the name's daily traded value")
    ax.set_title(f"2. Participation over time, spread across names — {arm}")
    ax.legend(ncol=4, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out_dir / "participation_2_timeseries.png"); plt.close(fig)

    # ---- CHART 3: the two value-weighted daily averages --------------------
    fig, ax = plt.subplots(figsize=(11, 4.6))
    # my-value-weighted: the rate my own P&L is exposed to
    ax.plot(dly["date"], dly["wavg_mine_pct"], color=C_VERM, linewidth=1.8,
            label="weighted by MY traded value (what my fills experience)")
    # market-value-weighted == total mine / total market: the book-wide footprint
    ax.plot(dly["date"], dly["footprint_pct"], color=C_BLUE, linewidth=1.8,
            label="weighted by MARKET traded value (= total mine / total market)")
    # the warn line for scale
    ax.axhline(WARN_PCT, color=C_GREY, linewidth=1.0, linestyle="--")
    ax.set_ylabel("% of traded value")
    ax.set_title(f"3. Value-weighted daily participation, two weightings — {arm}")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out_dir / "participation_3_weighted.png"); plt.close(fig)

    # ---- CHART 4: the daily median on its own ------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.0))
    # the single headline series he asked for
    ax.plot(dly["date"], dly["median_pct"], color=C_BLUE, linewidth=1.8, label="median across names")
    # a 21-day rolling mean of it, so the trend is visible through the noise
    ax.plot(dly["date"], dly["median_pct"].rolling(21, min_periods=5).mean(),
            color=C_VERM, linewidth=1.6, label="21-day rolling mean")
    ax.set_ylabel("% of the name's daily traded value")
    ax.set_title(f"4. Median participation across names, per day — {arm}")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out_dir / "participation_4_median.png"); plt.close(fig)

    # ---- CHART 5 (ADDED): which NAMES carry the impact risk ----------------
    # the aggregate is reassuring and the per-name tail is what actually breaks
    # a backtest, so rank the names by their own mean participation
    per = (df.groupby("symbol")
             .agg(mean_pct=(r, "mean"), p90_pct=(r, lambda s: s.quantile(0.90)),
                  max_pct=(r, "max"), my_value=("my_value", "sum"))
             .sort_values("mean_pct", ascending=False))
    # the worst 25 names, which is where any cap would be applied
    top = per.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    # horizontal bars, thin, with the 90th percentile drawn as a marker on top
    ax.barh(top.index, top["mean_pct"], color=C_BLUE, height=0.65, label="mean across days")
    ax.plot(top["p90_pct"], range(len(top)), "o", color=C_VERM, markersize=6,
            label="90th percentile day")
    # the policy lines
    ax.axvline(WARN_PCT, color=C_VERM, linewidth=1.0, linestyle="--")
    ax.axvline(ALARM_PCT, color=C_PURPLE, linewidth=1.0, linestyle="--")
    ax.set_xlabel("% of the name's daily traded value")
    ax.set_title(f"5. The 25 names where I am biggest — {arm}")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_dir / "participation_5_by_name.png"); plt.close(fig)
    # hand the per-name table back for the printed summary
    return per


# ------------------------------------------------------------- THE REPORT --
def report(df, dly, per, arm):
    """Print the numbers a chart cannot carry, and the verdict."""
    # the conservative ratio again
    r = "pov_excl_pct"
    print()
    print(f"===== PARTICIPATION: {arm} =====")
    print(f"  symbol-days measured        {len(df):,}")
    print(f"  my traded value (total)     {df['my_value'].sum():>18,.0f} PKR")
    print(f"  market traded value (total) {df['mkt_value'].sum():>18,.0f} PKR")
    print(f"  BOOK-WIDE FOOTPRINT         {df['my_value'].sum()/df['mkt_value'].sum()*100:>18.3f} %")
    print()
    print("  per name-day distribution (% of the name's daily traded value):")
    # the whole distribution, because the median alone hides the tail
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00):
        print(f"    p{q*100:>5.1f}  {df[r].quantile(q):>8.3f} %")
    print()
    # the counts that decide whether this is a problem
    n_warn = int((df[r] > WARN_PCT).sum()); n_alarm = int((df[r] > ALARM_PCT).sum())
    print(f"  symbol-days over {WARN_PCT:.0f}%:  {n_warn:,} ({n_warn/len(df)*100:.2f}%)")
    print(f"  symbol-days over {ALARM_PCT:.0f}%: {n_alarm:,} ({n_alarm/len(df)*100:.2f}%)")
    # the share of P&L exposure that sits in the risky cells is the real metric
    hot = df[df[r] > WARN_PCT]["my_value"].sum() / df["my_value"].sum() * 100.0
    print(f"  share of MY traded value in symbol-days over {WARN_PCT:.0f}%: {hot:.2f}%")
    print()
    print("  worst 10 names by mean participation:")
    for sym, row in per.head(10).iterrows():
        print(f"    {sym:10s} mean {row['mean_pct']:6.2f} %   p90 {row['p90_pct']:6.2f} %   "
              f"max {row['max_pct']:6.2f} %")
    print()
    # the verdict, stated as a rule rather than a feeling
    med = df[r].median()
    if med < 1.0 and hot < 5.0:
        print("  READ: the no-impact assumption is defensible for the book as a whole.")
        print("        Apply a per-name cap to the names listed above rather than")
        print("        discounting the whole backtest.")
    elif med < 3.0:
        print("  READ: borderline. The median is small but a material share of traded")
        print("        value sits in high-participation cells -- re-run with a POV cap")
        print("        and compare P&L, do not discount by assumption.")
    else:
        print("  READ: participation is high enough that the backtest's fill model is")
        print("        the binding assumption, not the strategy. Impact must be modelled")
        print("        before any of these numbers are used for sizing.")


# ------------------------------------------------------------- ENTRY POINT -
if __name__ == "__main__":
    # the parser
    ap = argparse.ArgumentParser()
    # which arm to measure
    ap.add_argument("--arm", default=ARM)
    # where to write
    ap.add_argument("--out", default=str(OUT_DIR))
    # run it
    ap.add_argument("--run", action="store_true")
    # parse
    a = ap.parse_args()
    # build the joined panel
    print(f"building participation panel for arm={a.arm}")
    d = build(a.arm)
    # collapse to daily
    dl = daily_stats(d)
    # draw
    p = charts(d, dl, pathlib.Path(a.out), a.arm)
    # the per-name-day panel and the daily series, saved so they can be re-plotted
    d.to_csv(pathlib.Path(a.out) / f"participation_panel_{a.arm.replace('@','')}.csv", index=False)
    dl.to_csv(pathlib.Path(a.out) / f"participation_daily_{a.arm.replace('@','')}.csv", index=False)
    # print the numbers
    report(d, dl, p, a.arm)
    # say where everything went
    print(f"\n  charts + CSVs -> {a.out}")

# ============================================================================
# THE EXACT VERSION, when you want it
# ----------------------------------------------------------------------------
# LEG_MULT = 2.0 is an approximation of my traded value. To remove it, emit the
# real number from the runner: in universe_expand.py, where the per-name row is
# built, add the traded value and traded shares straight off the fills --
#
#     traded_value  = float((f["price"] * f["qty"]).sum())
#     traded_shares = float(f["qty"].sum())
#
# -- and carry both into the PERNAME parquet. Then this script drops LEG_MULT
# and divides traded_value by mkt_value directly. That also unlocks the
# SHARE-based ratio, which is the one an exchange would recognise.
#
# THE MEASUREMENT THIS SCRIPT DOES NOT DO, AND WHY IT MATTERS MORE THAN THE
# DAILY NUMBER: participation is not uniform through the day. Being 4% of a
# name's daily volume while being 25% of its last fifteen minutes is a real
# impact problem that a daily ratio cannot see. The PERNAME parquet already
# carries the `bucket` column, so my side splits for free; the market side needs
# each print bucketed by the SAME boundaries mm_harness uses (session_segments),
# and those boundaries must be read from the calibration file rather than
# guessed. Do that next -- do not assume the daily number settles the question.
# ============================================================================
