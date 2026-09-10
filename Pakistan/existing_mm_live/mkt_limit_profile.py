# ============================================================================
# mkt_limit_profile.py -- MARKET vs LIMIT order composition by 15-MINUTE bucket.
# ----------------------------------------------------------------------------
# Classifies every incoming order from the feed:
#   MARKET = a TRADE (a marketable order that hit a resting limit order).
#            COLLAPSED: prints sharing the same exchange timestamp AND the same
#            aggressor side are ONE market order (a sweep fragments across the
#            resting orders it consumes). Volume is summed across the fragments.
#   LIMIT  = an ORDER_ADD in ob_updates (new resting liquidity entering the book).
#
# Reports, per 15-minute interval of the trading session:
#   * pct_market_cnt / pct_limit_cnt  -- share of ORDER COUNT
#   * pct_market_vol / pct_limit_vol  -- share of SHARE VOLUME
# both PER TICKER and AGGREGATED at exchange level (all names pooled), with the
# per-name-day rows saved so any later cut (by name, by day) is possible.
#
# Reads trades + ob_updates directly (no book replay needed) -> fast.
# Paths come from config_pk (single source of truth).
#
# USAGE:  python mkt_limit_profile.py --self-test | --smoke | --run
# ============================================================================

# CLI parsing
import argparse
# timing + heartbeat
import time
# timestamps on printed lines
from datetime import datetime
# parallel over dates
from multiprocessing import Pool
# numerics
import numpy as np
# dataframes
import pandas as pd
# headless plotting for the profile chart
import matplotlib
# no display needed
matplotlib.use("Agg")
# plot API
import matplotlib.pyplot as plt

# central path config (raw store + results roots)
from config_pk import PARSED_ROOT, RESULTS_ROOT


# bracketed HH:MM:SS stamp for every printed line
def _ts():
    # current local time
    return datetime.now().strftime("[%H:%M:%S]")


# diagnostics output directory
OUT_DIR = RESULTS_ROOT / "diagnostics"
# the 38-name shortlist (falls back to all traded symbols if absent)
WATCHLIST = RESULTS_ROOT / "mm_watchlist_final.csv"
# interval length in minutes
BUCKET_MIN = 15
# interval length in milliseconds
BUCKET_MS = BUCKET_MIN * 60 * 1000
# default number of sampled days
MAX_DAYS = 207
# default parallel workers
WORKERS = 6



# convert a datetime-like column to EPOCH MILLISECONDS, unit-safely.
# (pandas .astype("int64") returns ns or us depending on the dtype resolution;
# assuming one of them silently rescales every timestamp -- so derive it.)
def _to_ms(series):
    # parse to UTC datetimes (tz-aware)
    dt = pd.to_datetime(series, utc=True)
    # drop the timezone so the epoch subtraction is well-defined
    naive = dt.dt.tz_localize(None) if hasattr(dt, "dt") else dt.tz_localize(None)
    # milliseconds since the epoch, computed by pandas (no unit assumption)
    return (naive - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")


# assign each timestamp to a 15-minute interval index measured from session open
def _bucket_idx(ts_ms, t0):
    # integer number of whole 15-min intervals since the session start
    return ((np.asarray(ts_ms, dtype=float) - float(t0)) // BUCKET_MS).astype(int)


# core: from trades + updates for ONE symbol-day, produce per-interval composition
def composition(trades, updates):
    # need both tables to have rows
    if trades is None or updates is None or len(trades) == 0 or len(updates) == 0:
        # nothing to compute
        return None
    # ---- MARKET orders: trades, collapsed by (exchange ts, aggressor side) ----
    # keep only rows with a usable aggressor side
    tr = trades.dropna(subset=["transact_time", "aggressor_side", "qty"]).copy()
    # normalize the side to a single letter
    tr["sd"] = tr["aggressor_side"].astype(str).str.upper().str[0]
    # keep only buy/sell aggressors
    tr = tr[tr["sd"].isin(["B", "S"])]
    # unrunnable if no trades survive
    if len(tr) == 0:
        # nothing to compute
        return None
    # exact-equality key for the collapse: the raw integer epoch (unit-agnostic,
    # equality is all we need here, not the unit)
    tr["tns"] = pd.to_datetime(tr["transact_time"], utc=True).astype("int64")
    # epoch MILLISECONDS (unit-safe) for bucketing
    tr["tms"] = _to_ms(tr["transact_time"])
    # sort into true event order
    tr = tr.sort_values("tns")
    # a print starts a NEW market order unless it shares ts AND side with the previous
    newo = np.ones(len(tr), dtype=bool)
    # compare each row to its predecessor (same ts and same side = same sweep)
    newo[1:] = (tr["tns"].to_numpy()[1:] != tr["tns"].to_numpy()[:-1]) | \
               (tr["sd"].to_numpy()[1:] != tr["sd"].to_numpy()[:-1])
    # mark which prints begin a distinct market order
    tr["new_order"] = newo
    # ---- LIMIT orders: every ORDER_ADD in ob_updates ----
    # keep adds with a timestamp and quantity
    up = updates.dropna(subset=["transact_time", "qty"]).copy()
    # only ORDER_ADD rows count as new limit orders
    up = up[up["event"].astype(str) == "ORDER_ADD"]
    # unrunnable if no adds
    if len(up) == 0:
        # nothing to compute
        return None
    # epoch MILLISECONDS (unit-safe) for bucketing
    up["tms"] = _to_ms(up["transact_time"])
    # ---- session origin: the earliest event across both streams ----
    t0 = min(tr["tms"].min(), up["tms"].min())
    # bucket index for every trade print
    tr["bkt"] = _bucket_idx(tr["tms"], t0)
    # bucket index for every limit add
    up["bkt"] = _bucket_idx(up["tms"], t0)
    # ---- per-bucket tallies ----
    # market order COUNT counts only collapse-new prints (one per sweep)
    m_cnt = tr[tr["new_order"]].groupby("bkt").size().rename("mkt_cnt")
    # market VOLUME sums every print's shares (all fragments of the sweep)
    m_vol = tr.groupby("bkt")["qty"].sum().rename("mkt_vol")
    # limit order COUNT is one per ORDER_ADD
    l_cnt = up.groupby("bkt").size().rename("lim_cnt")
    # limit VOLUME sums the added shares
    l_vol = up.groupby("bkt")["qty"].sum().rename("lim_vol")
    # combine the four series on the bucket index
    out = pd.concat([m_cnt, m_vol, l_cnt, l_vol], axis=1).fillna(0.0).reset_index()
    # drop any negative bucket (defensive; cannot occur since t0 is the min)
    out = out[out["bkt"] >= 0]
    # the raw per-bucket tallies for this symbol-day
    return out


# per-process globals set by the pool initializer
_R = None
_NAMES = None


# pool initializer: import the driver, point it at the store, resolve the universe
def _init(names):
    # expose the globals
    global _R, _NAMES
    # the parquet driver
    import run_legacy_mm as R
    # point at the configured raw store
    R.PARSED_ROOT = PARSED_ROOT
    # stash the driver
    _R = R
    # default the universe to the watchlist, else all traded symbols
    if names is None:
        # try to read the shortlist
        try:
            # symbol column of the final watchlist
            names = sorted(pd.read_csv(WATCHLIST)["symbol"].dropna().astype(str).unique().tolist())
        except Exception:
            # fall back to per-date discovery
            names = None
    # stash the universe
    _NAMES = names


# columns needed from each table (column pushdown keeps the read light)
_TR_COLS = ["symbol", "transact_time", "aggressor_side", "qty"]
# updates need the event type to isolate ORDER_ADD
_UP_COLS = ["symbol", "transact_time", "event", "qty"]


# process one date: every symbol's per-bucket tallies
def _work_date(date):
    # open the date's datasets
    dsets = _R.open_datasets(date)
    # missing partition -> nothing to do
    if dsets is None:
        # empty result
        return []
    # the symbol universe for this date
    names = _NAMES or _R.list_symbols(dsets["trades"])
    # accumulated rows
    rows = []
    # loop the symbols
    for sym in names:
        # guard each symbol so one bad read cannot kill the date
        try:
            # this symbol's trades
            tr = _R.read_symbol(dsets["trades"], _TR_COLS, sym)
            # this symbol's book updates
            up = _R.read_symbol(dsets["ob_updates"], _UP_COLS, sym)
        except Exception as e:
            # report and continue
            print(_ts() + f"SKIP {date} {sym}: {e!r}")
            # next symbol
            continue
        # compute the per-bucket composition
        c = composition(tr, up)
        # skip symbols with nothing usable
        if c is None or len(c) == 0:
            # next symbol
            continue
        # tag with the date
        c["date"] = str(date)
        # tag with the symbol
        c["symbol"] = sym
        # collect the rows
        rows.append(c)
    # this date's frames
    return rows


# format a wall-clock label for a bucket index (session-relative)
def _bucket_label(b):
    # minutes from the session open at the start of this interval
    m0 = int(b) * BUCKET_MIN
    # minutes at the end of the interval
    m1 = m0 + BUCKET_MIN
    # a session-relative label, e.g. "+00:00-00:15"
    return f"+{m0//60:02d}:{m0%60:02d}-{m1//60:02d}:{m1%60:02d}"


# print the exchange-level table and write the outputs
def _report(df, out_dir):
    # ensure the output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    # save the full per-name-day-bucket tallies (any later cut is possible from this)
    df.to_csv(out_dir / "mkt_limit_profile_pername_day.csv", index=False)
    # ---- EXCHANGE LEVEL: pool every name and day, per bucket ----
    ex = df.groupby("bkt")[["mkt_cnt", "lim_cnt", "mkt_vol", "lim_vol"]].sum().reset_index()
    # total order count in the bucket
    ex["tot_cnt"] = ex["mkt_cnt"] + ex["lim_cnt"]
    # total share volume in the bucket
    ex["tot_vol"] = ex["mkt_vol"] + ex["lim_vol"]
    # percent of ORDERS that were market
    ex["pct_mkt_cnt"] = ex["mkt_cnt"] / ex["tot_cnt"].replace(0, np.nan) * 100
    # percent of SHARES that were market
    ex["pct_mkt_vol"] = ex["mkt_vol"] / ex["tot_vol"].replace(0, np.nan) * 100
    # percent of orders that were limit (the complement)
    ex["pct_lim_cnt"] = 100 - ex["pct_mkt_cnt"]
    # percent of shares that were limit
    ex["pct_lim_vol"] = 100 - ex["pct_mkt_vol"]
    # save the exchange-level profile
    ex.to_csv(out_dir / "mkt_limit_profile_exchange.csv", index=False)
    # header
    print(_ts() + "===== MARKET vs LIMIT by 15-MIN INTERVAL -- EXCHANGE LEVEL (all names pooled) =====")
    # column header
    print(_ts() + "  interval          %mkt(cnt)  %lim(cnt)   %mkt(vol)  %lim(vol)     mkt_orders     limit_orders")
    # one line per interval
    for _, r in ex.iterrows():
        # print the formatted row
        print(_ts() + f"  {_bucket_label(r['bkt']):<16} {r['pct_mkt_cnt']:8.1f}%  {r['pct_lim_cnt']:8.1f}%  "
              f"{r['pct_mkt_vol']:8.1f}%  {r['pct_lim_vol']:8.1f}%   {int(r['mkt_cnt']):>12,}   {int(r['lim_cnt']):>12,}")
    # ---- PER TICKER: pooled across days, per bucket ----
    pn = df.groupby(["symbol", "bkt"])[["mkt_cnt", "lim_cnt", "mkt_vol", "lim_vol"]].sum().reset_index()
    # totals per name-bucket
    pn["tot_cnt"] = pn["mkt_cnt"] + pn["lim_cnt"]
    # volume totals
    pn["tot_vol"] = pn["mkt_vol"] + pn["lim_vol"]
    # market share of orders
    pn["pct_mkt_cnt"] = pn["mkt_cnt"] / pn["tot_cnt"].replace(0, np.nan) * 100
    # market share of volume
    pn["pct_mkt_vol"] = pn["mkt_vol"] / pn["tot_vol"].replace(0, np.nan) * 100
    # save the per-ticker profile
    pn.to_csv(out_dir / "mkt_limit_profile_pername.csv", index=False)
    # per-ticker DAY AVERAGE (all buckets pooled) so names can be ranked
    tot = df.groupby("symbol")[["mkt_cnt", "lim_cnt", "mkt_vol", "lim_vol"]].sum()
    # market share of orders per name
    tot["pct_mkt_cnt"] = tot["mkt_cnt"] / (tot["mkt_cnt"] + tot["lim_cnt"]) * 100
    # market share of volume per name
    tot["pct_mkt_vol"] = tot["mkt_vol"] / (tot["mkt_vol"] + tot["lim_vol"]) * 100
    # rank by the volume measure
    tot = tot.sort_values("pct_mkt_vol", ascending=False)
    # header
    print(_ts() + "\n===== PER TICKER (all intervals pooled), sorted by %market of VOLUME =====")
    # column header
    print(_ts() + "  symbol     %mkt(cnt)   %mkt(vol)")
    # one line per name
    for sym, r in tot.iterrows():
        # print the name's shares
        print(_ts() + f"  {sym:<10} {r['pct_mkt_cnt']:8.1f}%   {r['pct_mkt_vol']:8.1f}%")
    # ---- CHART: exchange-level intraday profile, both measures ----
    _plot(ex, out_dir)
    # where things went
    print(_ts() + f"[mkt-limit] outputs -> {out_dir}")


# draw the exchange-level intraday profile
def _plot(ex, out_dir):
    # one figure, one axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # x positions are the interval indices
    x = ex["bkt"].to_numpy()
    # market share of order count
    ax.plot(x, ex["pct_mkt_cnt"], marker="o", lw=2, color="#c0392b", label="% market (order count)")
    # market share of share volume
    ax.plot(x, ex["pct_mkt_vol"], marker="s", lw=2, color="#2c6fbb", label="% market (share volume)")
    # 50% reference line
    ax.axhline(50, color="#888", ls=":", lw=1)
    # label the x axis with session-relative interval labels
    ax.set_xticks(x)
    # rotate for readability
    ax.set_xticklabels([_bucket_label(b) for b in x], rotation=60, fontsize=7)
    # axis labels
    ax.set_xlabel("15-minute interval (from session open)")
    # y axis
    ax.set_ylabel("% of incoming orders that were MARKET orders")
    # title
    ax.set_title("Market vs limit order composition through the session (exchange level)\n"
                 "market = trades (sweep-collapsed) | limit = ORDER_ADDs")
    # legend + grid
    ax.legend()
    # light grid
    ax.grid(alpha=0.25)
    # tidy layout
    fig.tight_layout()
    # save the figure
    fig.savefig(out_dir / "mkt_limit_profile.png", dpi=130)
    # free the figure
    plt.close(fig)


# full run across sampled dates
def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    # the driver (for date discovery in the parent process)
    import run_legacy_mm as R
    # point at the configured store
    R.PARSED_ROOT = PARSED_ROOT
    # all available dates
    dates = R.discover_dates()
    # stop if the store is empty/misconfigured
    if not dates:
        # explain and exit
        print(_ts() + "discover_dates() empty -- check config_pk PARSED_ROOT."); return
    # sample dates evenly when a cap is set
    if max_days and len(dates) > max_days:
        # stride to hit roughly max_days
        step = max(1, len(dates) // max_days)
        # take the strided sample
        dates = dates[::step][:max_days]
    # announce the run
    print(_ts() + f"{len(dates)} dates, {workers} workers -- market vs limit by {BUCKET_MIN}-min interval")
    # collected frames
    frames = []
    # overall timer
    t0 = time.perf_counter()
    # worker pool
    with Pool(processes=workers, initializer=_init, initargs=(symbols,)) as pool:
        # completed-date counter
        done = 0
        # consume results as dates complete
        for res in pool.imap_unordered(_work_date, dates):
            # collect this date's frames
            frames.extend(res)
            # bump the counter
            done += 1
            # elapsed minutes
            el = (time.perf_counter() - t0) / 60.0
            # progress with ETA
            print(_ts() + f"  date {done}/{len(dates)} ({el:.1f} min, ETA {el/done*(len(dates)-done):.1f} min)")
    # nothing collected
    if not frames:
        # stop
        print(_ts() + "no rows."); return
    # one frame of all per-name-day-bucket tallies
    df = pd.concat(frames, ignore_index=True)
    # report + save
    _report(df, out_dir)


# time ONE symbol-day in this process (fast sanity check)
def smoke(symbols=None):
    # the driver
    import run_legacy_mm as R
    # point at the store
    R.PARSED_ROOT = PARSED_ROOT
    # available dates
    dates = R.discover_dates()
    # report how many were found
    print(_ts() + f"discover_dates -> {len(dates)} dates")
    # stop if none
    if not dates:
        # exit
        return
    # a mid-panel date
    date = dates[len(dates) // 2]
    # open the datasets
    dsets = R.open_datasets(date)
    # the symbol to test
    sym = symbols[0] if symbols else R.list_symbols(dsets["trades"])[0]
    # time the read + computation
    t0 = time.perf_counter()
    # trades for this symbol
    tr = R.read_symbol(dsets["trades"], _TR_COLS, sym)
    # updates for this symbol
    up = R.read_symbol(dsets["ob_updates"], _UP_COLS, sym)
    # compute the composition
    c = composition(tr, up)
    # elapsed
    dt = time.perf_counter() - t0
    # report timing and shape
    print(_ts() + f"[smoke] {date} {sym} in {dt:.1f}s: "
          f"{0 if c is None else len(c)} intervals, trades={len(tr):,}, adds={len(up):,}")
    # show the intervals if present
    if c is not None and len(c):
        # totals per bucket
        c = c.copy()
        # market share of count
        c["pct_mkt_cnt"] = c["mkt_cnt"] / (c["mkt_cnt"] + c["lim_cnt"]) * 100
        # market share of volume
        c["pct_mkt_vol"] = c["mkt_vol"] / (c["mkt_vol"] + c["lim_vol"]) * 100
        # print a compact preview
        for _, r in c.iterrows():
            # one line per interval
            print(_ts() + f"[smoke]   {_bucket_label(r['bkt']):<16} "
                  f"%mkt_cnt={r['pct_mkt_cnt']:5.1f}  %mkt_vol={r['pct_mkt_vol']:5.1f}")


# validate the collapse + bucketing + percentage math on hand-built data
def self_test():
    # ---- build trades: a 3-print sweep at t=0 (ONE market order, 300 shares),
    # then a single sell at t=1s (one order, 50), then a buy in the NEXT interval.
    trades = pd.DataFrame({
        # exchange timestamps (the 3 sweep prints share one timestamp)
        "transact_time": pd.to_datetime([
            "2026-01-01T10:00:00.000Z", "2026-01-01T10:00:00.000Z", "2026-01-01T10:00:00.000Z",
            "2026-01-01T10:00:01.000Z",
            "2026-01-01T10:20:00.000Z"], utc=True),
        # aggressor sides
        "aggressor_side": ["BUY", "BUY", "BUY", "SELL", "BUY"],
        # share quantities
        "qty": [100.0, 100.0, 100.0, 50.0, 200.0],
        # symbol column (unused by composition but present in the real read)
        "symbol": ["X"] * 5,
    })
    # ---- build updates: 2 adds in the first interval, 1 add in the second,
    # plus a CANCEL that must NOT be counted as a limit order.
    updates = pd.DataFrame({
        # timestamps
        "transact_time": pd.to_datetime([
            "2026-01-01T10:00:00.500Z", "2026-01-01T10:05:00.000Z",
            "2026-01-01T10:06:00.000Z",
            "2026-01-01T10:20:30.000Z"], utc=True),
        # event types (the CANCEL is ignored)
        "event": ["ORDER_ADD", "ORDER_ADD", "CANCEL", "ORDER_ADD"],
        # quantities
        "qty": [400.0, 600.0, 999.0, 500.0],
        # symbol column
        "symbol": ["X"] * 4,
    })
    # run the composition
    c = composition(trades, updates)
    # index by bucket for assertions
    c = c.set_index("bkt")
    # show it
    print(_ts() + f"[self-test] buckets:\n{c.to_string()}")
    # ---- interval 0 (first 15 min) ----
    # market COUNT: the 3-print sweep collapses to 1, plus the single sell = 2
    assert c.loc[0, "mkt_cnt"] == 2, f"market count should be 2 (sweep collapsed), got {c.loc[0,'mkt_cnt']}"
    # market VOLUME: all prints count -> 100+100+100+50 = 350
    assert c.loc[0, "mkt_vol"] == 350.0, f"market volume should be 350, got {c.loc[0,'mkt_vol']}"
    # limit COUNT: 2 adds (the CANCEL is excluded)
    assert c.loc[0, "lim_cnt"] == 2, f"limit count should be 2 (cancel excluded), got {c.loc[0,'lim_cnt']}"
    # limit VOLUME: 400+600 = 1000 (cancel's 999 excluded)
    assert c.loc[0, "lim_vol"] == 1000.0, f"limit volume should be 1000, got {c.loc[0,'lim_vol']}"
    # ---- interval 1 (15-30 min) ----
    # one market order of 200 shares
    assert c.loc[1, "mkt_cnt"] == 1 and c.loc[1, "mkt_vol"] == 200.0
    # one limit add of 500 shares
    assert c.loc[1, "lim_cnt"] == 1 and c.loc[1, "lim_vol"] == 500.0
    # ---- percentage math on interval 0 ----
    # market share of count = 2 / (2+2) = 50%
    pct_cnt = c.loc[0, "mkt_cnt"] / (c.loc[0, "mkt_cnt"] + c.loc[0, "lim_cnt"]) * 100
    # market share of volume = 350 / (350+1000) = 25.93%
    pct_vol = c.loc[0, "mkt_vol"] / (c.loc[0, "mkt_vol"] + c.loc[0, "lim_vol"]) * 100
    # verify both
    assert abs(pct_cnt - 50.0) < 1e-9, f"pct market count wrong: {pct_cnt}"
    # volume percentage check
    assert abs(pct_vol - (350/1350*100)) < 1e-9, f"pct market vol wrong: {pct_vol}"
    # report
    print(_ts() + f"[self-test] interval0: %mkt_cnt={pct_cnt:.1f} %mkt_vol={pct_vol:.2f}  OK")
    # ---- bucket labelling ----
    # interval 0 label
    assert _bucket_label(0) == "+00:00-00:15", _bucket_label(0)
    # interval 5 spans 75-90 minutes
    assert _bucket_label(5) == "+01:15-01:30", _bucket_label(5)
    # report
    print(_ts() + "[self-test] bucket labels OK")
    # all good
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


# CLI entry point
if __name__ == "__main__":
    # argument parser
    ap = argparse.ArgumentParser(description="Market vs limit order composition by 15-min interval.")
    # run the self-test
    ap.add_argument("--self-test", action="store_true")
    # time one symbol-day
    ap.add_argument("--smoke", action="store_true")
    # do the full run
    ap.add_argument("--run", action="store_true")
    # restrict to specific symbols
    ap.add_argument("--symbols", nargs="*", default=None)
    # worker count
    ap.add_argument("--workers", type=int, default=WORKERS)
    # number of days (0 = all)
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    # parse the arguments
    a = ap.parse_args()
    # dispatch: smoke mode
    if a.smoke:
        # time one symbol-day
        smoke(symbols=a.symbols)
    # otherwise default to the self-test unless --run was given
    elif a.self_test or not a.run:
        # validate the math
        self_test()
    # full run
    if a.run:
        # scan the universe
        run_real(symbols=a.symbols, workers=a.workers, max_days=(a.days or None))
