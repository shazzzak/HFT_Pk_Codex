# lock_screener.py -- pick a thin/locky third name that will actually EXERCISE the
# time-to-close and distance-to-lock triggers. For each watchlist symbol it counts,
# from raw snapshots (no book reconstruction): (a) how often the CONTINUOUS_AUCTION
# touch sits at the published UPPER/LOWER circuit-breaker price (a lock), and (b) how
# often the continuous session CLOSES one-sided. Cross-referenced with spread/notional
# from the watchlist so we pick a name that is both locky AND genuinely thin.
#
# Run from existing_mm_live/:  python lock_screener.py

# filesystem paths
from pathlib import Path
# timing for the heartbeat
import time
# dataframes
import pandas as pd
# numeric helpers
import numpy as np
# driver module (dataset discovery, per-symbol reads)
import run_legacy_mm as R

# point the driver at the parsed data root
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# feature-store root (to confirm each candidate is runnable in confirm's Path B)
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
# the watchlist CSV (symbol column + context columns)
WATCHLIST = Path("/Users/shazzak/Capital Stake - Results/mm_watchlist_final.csv")
# tolerance (in price units) for calling the touch "at" the published limit
LOCK_TOL = 0.01
# cap days for a fast first pass; None = all 207 (definitive but ~5x slower)
DAYS_LIMIT = 40


# format seconds as mm:ss
def _fmt(sec):
    # minutes and zero-padded seconds
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# compute per-symbol-day lock/one-sided stats from one day's snapshot rows (no book)
def day_stats(s):
    # keep only continuous-trading snapshots (locks/closes we care about happen here)
    c = s[s["phase"] == "CONTINUOUS_AUCTION"]
    # no continuous data -> nothing to measure
    if len(c) == 0:
        return None
    # the day's published upper limit = first non-null UPPER_CIRCUIT_BREAKER price
    up_rows = c.loc[c["entry_type"] == "UPPER_CIRCUIT_BREAKER", "px"].dropna()
    # the day's published lower limit = first non-null LOWER_CIRCUIT_BREAKER price
    dn_rows = c.loc[c["entry_type"] == "LOWER_CIRCUIT_BREAKER", "px"].dropna()
    # day-constant limits (None if the feed never published them)
    lim_up = float(up_rows.iloc[0]) if len(up_rows) else None
    lim_dn = float(dn_rows.iloc[0]) if len(dn_rows) else None
    # best bid per message = max px among BID rows
    bid = c[c["entry_type"] == "BID"].groupby("msg_seq")["px"].max()
    # best ask per message = min px among OFFER rows
    ask = c[c["entry_type"] == "OFFER"].groupby("msg_seq")["px"].min()
    # union of messages that carried any book level (bid or ask)
    msgs = bid.index.union(ask.index)
    # messages with a real book
    n_msgs = len(msgs)
    # no priced book all day -> skip
    if n_msgs == 0:
        return None
    # align bid/ask onto the full message set (NaN where a side was absent)
    bid = bid.reindex(msgs)
    ask = ask.reindex(msgs)
    # upper-lock per message: best bid at/above the published upper limit
    if lim_up is not None:
        up_lock = (bid >= (lim_up - LOCK_TOL))
    else:
        # no upper limit published -> never upper-locked
        up_lock = pd.Series(False, index=msgs)
    # lower-lock per message: best ask at/below the published lower limit
    if lim_dn is not None:
        dn_lock = (ask <= (lim_dn + LOCK_TOL))
    else:
        # no lower limit published -> never lower-locked
        dn_lock = pd.Series(False, index=msgs)
    # a message is locked if either side is pinned (NaN sides count as not-locked)
    locked = (up_lock.fillna(False) | dn_lock.fillna(False))
    # fraction of continuous messages that were locked
    lock_frac = float(locked.mean())
    # did the day touch a lock at all?
    any_lock = bool(locked.any())
    # one-sided CLOSE: look at the LAST message that had a book
    last_msg = msgs.max()
    # sides present in that last message
    has_bid = bool(pd.notna(bid.get(last_msg, np.nan)))
    has_ask = bool(pd.notna(ask.get(last_msg, np.nan)))
    # one-sided if exactly one side is present at the continuous close
    one_sided_close = has_bid != has_ask
    # return the per-day summary
    return {"n_msgs": n_msgs, "lock_frac": lock_frac,
            "any_lock": any_lock, "one_sided_close": one_sided_close}


def main():
    # load the watchlist
    wl = pd.read_csv(WATCHLIST)
    # the symbols to screen
    symbols = wl["symbol"].astype(str).tolist()
    # feature-store availability per symbol (user says all built; verify cheaply)
    fs_ok = {sym: (FS_ROOT / sym).exists() for sym in symbols}
    # all dates, optionally truncated
    dates = R.discover_dates()
    # truncate for the fast first pass
    if DAYS_LIMIT is not None:
        dates = dates[:DAYS_LIMIT]
    # per-symbol accumulators
    acc = {sym: {"days": 0, "lock_days": 0, "onesided_days": 0, "lock_frac_sum": 0.0}
           for sym in symbols}
    # timers / counter
    t0_all = time.perf_counter()
    sd = 0
    # total symbol-days for ETA
    sd_total = len(dates) * len(symbols)
    # announce
    print(f"lock_screener: {len(symbols)} symbols x {len(dates)} days\n", flush=True)
    # OUTER over dates
    for date in dates:
        # open datasets once per day
        dsets = R.open_datasets(date)
        # skip missing days
        if dsets is None:
            continue
        # MIDDLE over watchlist symbols
        for sym in symbols:
            # read this symbol's snapshot rows (entry_type/px/phase/msg_seq)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            # advance the counter regardless
            sd += 1
            # skip symbols absent this day
            if len(s) == 0:
                continue
            # compute the per-day stats
            st = day_stats(s)
            # skip days with no continuous book
            if st is None:
                continue
            # accumulate
            a = acc[sym]
            # count this as a screened day
            a["days"] += 1
            # count lock days / one-sided-close days
            a["lock_days"] += int(st["any_lock"])
            a["onesided_days"] += int(st["one_sided_close"])
            # sum lock fraction for a mean later
            a["lock_frac_sum"] += st["lock_frac"]
            # heartbeat every 100 symbol-days
            if sd % 100 == 0:
                # elapsed
                el = time.perf_counter() - t0_all
                # projected total
                proj = el / sd * sd_total
                # print progress + ETA
                print(f"  {sd}/{sd_total} symbol-days  elapsed {_fmt(el)}  "
                      f"ETA {_fmt(proj - el)}", flush=True)

    # assemble results
    rows = []
    for sym in symbols:
        a = acc[sym]
        # skip symbols with no screened days
        if a["days"] == 0:
            continue
        rows.append({
            "symbol": sym,
            # days screened
            "days": a["days"],
            # % of days that touched a lock
            "pct_lock_days": round(100.0 * a["lock_days"] / a["days"], 1),
            # % of days that closed one-sided (continuous)
            "pct_onesided_close": round(100.0 * a["onesided_days"] / a["days"], 1),
            # mean fraction of the continuous session spent locked
            "mean_lock_frac": round(a["lock_frac_sum"] / a["days"], 4),
            # feature-store present?
            "fs": fs_ok.get(sym, False),
        })
    # to dataframe
    out = pd.DataFrame(rows)
    # bring in thinness context from the watchlist
    out = out.merge(wl[["symbol", "tier", "spread_bps_median", "notional_m_median"]],
                    on="symbol", how="left")
    # rank by lock activity (pct of days locked, then time spent locked)
    out = out.sort_values(["pct_lock_days", "mean_lock_frac"], ascending=False)
    # full-width print
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 60)
    # show the ranked table
    print("\n--- lock activity, ranked (fast first pass) ---")
    print(out.to_string(index=False))

    # guidance
    print("\nPICK: a good third name is high pct_lock_days / pct_onesided_close AND")
    print("thin (wide spread_bps_median, low notional_m_median) AND fs=True. That")
    print("exercises the new triggers AND is runnable in confirm's Path B. This is")
    print("the 40-day fast pass -- set DAYS_LIMIT=None for the definitive 207-day rank.")


if __name__ == "__main__":
    main()
