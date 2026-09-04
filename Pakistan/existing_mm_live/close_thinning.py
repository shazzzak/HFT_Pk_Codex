# close_thinning.py -- WHEN do books go one-sided before the bell?
# Decides whether "cross to flatten at T-120s" can work: on the days that CLOSED
# one-sided, was the book already one-sided at T-120s (no counterparty ever ->
# crossing early can't help), or did it collapse only in the final seconds
# (a 120s margin genuinely rescues them)?
#
# Pure book reconstruction: replays snapshots through book.snapshot() exactly like
# mm_backtest.run() (line 1070), reads book.bbo() after each. No strategy.
#
# Run from existing_mm_live/:  python close_thinning.py

# filesystem paths
from pathlib import Path
# wall-clock timing for the heartbeat
import time
# dataframes for bucketing/aggregation
import pandas as pd
# numeric arrays / binning
import numpy as np
# plotting library
import matplotlib
# force the non-interactive backend BEFORE importing pyplot (headless-safe)
matplotlib.use("Agg")
# the actual plotting interface
import matplotlib.pyplot as plt
# driver module: dataset discovery, per-symbol reads, event building
import run_legacy_mm as R
# the order-book object we drive with snapshots
from mm_backtest import Book

# point the driver at the parsed data root on this machine
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# symbols under study
SYMBOLS = ["PPL", "UBL"]
# how many seconds before the close to study
WINDOW_S = 300
# width of each time-to-close bucket, in seconds
BUCKET_S = 10
# the residual you actually need to flatten under MID+band150 (~66 sh PPL / ~49 UBL);
# a single conservative figure used to ask "is there enough depth to absorb it?"
RESIDUAL_SH = 66
# cap the number of days for a quick first pass; None = all days
DAYS_LIMIT = None


# format a duration in seconds as compact mm:ss
def _fmt(sec):
    # integer-divide for minutes, modulo for seconds
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# reconstruct book state through the last WINDOW_S seconds of one symbol-day;
# returns (list of per-snapshot sample dicts, close-type bool) or (rows, None)
def scan_day(events, snap_groups, t1):
    # a fresh empty book we will drive with snapshots only
    book = Book()
    # exchange-ms timestamp where the study window opens (WINDOW_S before the bell)
    w0 = t1 - WINDOW_S * 1000
    # collected samples that fall inside the window
    rows = []
    # remembers the two-sidedness of the most recent sample (for the close-type verdict)
    last_two_sided = None
    # walk every event in time order (build_events already sorted them)
    for ts_exch, _, _, kind, obj in events:
        # only snapshot events change the book state we measure
        if kind == "S":
            # apply this snapshot to the book (full replacement, like the backtester)
            book.snapshot(snap_groups[obj.msg_seq])
            # only SAMPLE inside the final window and up to the bell
            if w0 <= ts_exch <= t1:
                # read best bid / bid qty / best ask / ask qty from the book
                bb, bq, ba, aq = book.bbo()
                # seconds remaining until the close (0 at the bell)
                ttc = (t1 - ts_exch) / 1000.0
                # True only if BOTH sides have a resting price
                two = (bb is not None and ba is not None)
                # remember it so the last value = the close-type
                last_two_sided = two
                # store this sample
                rows.append({
                    # seconds-to-close for bucketing
                    "ttc_s": ttc,
                    # whether the book was two-sided at this instant
                    "two_sided": two,
                    # bid-side touch depth (0 if the bid side is empty)
                    "bq": bq if bb is not None else 0.0,
                    # ask-side touch depth (0 if the ask side is empty)
                    "aq": aq if ba is not None else 0.0,
                })
    # the day's close-type = two-sidedness of the LAST sample near the bell
    return rows, last_two_sided


# main driver: scan every symbol-day's final window, aggregate, plot
def main():
    # discover all available dates
    dates = R.discover_dates()
    # optionally truncate for a fast first pass
    if DAYS_LIMIT is not None:
        dates = dates[:DAYS_LIMIT]
    # per-symbol list of sample dicts (tagged with the day's close-type)
    samples = {s: [] for s in SYMBOLS}
    # per-symbol day counts split by close-type
    day_counts = {s: {"one": 0, "two": 0} for s in SYMBOLS}
    # whole-run timer start
    t0_all = time.perf_counter()
    # symbol-day counter for the heartbeat
    sd = 0
    # total symbol-days for the ETA
    sd_total = len(dates) * len(SYMBOLS)
    # announce the run
    print(f"close_thinning: {len(SYMBOLS)} symbols x {len(dates)} days\n", flush=True)
    # OUTER loop over dates
    for date in dates:
        # open the day's datasets once
        dsets = R.open_datasets(date)
        # skip days with no data
        if dsets is None:
            continue
        # MIDDLE loop over symbols
        for sym in SYMBOLS:
            # read this symbol's order-book updates
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            # read this symbol's snapshots
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            # read this symbol's trades
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # skip symbol-days with no trades or no snapshots
            if len(t) == 0 or len(s) == 0:
                continue
            # build the event stream + snapshot groups (same as the backtester)
            events, snap_groups, t = R.build_events(u, s, t)
            # restrict to continuous-session trades (drop auctions)
            cont = t[t["initiator"] != "AUCTION"]
            # skip days with no continuous session
            if len(cont) == 0:
                continue
            # the bell = last continuous-session trade timestamp
            t1 = int(cont["ts_exch"].max())
            # reconstruct the book through the final window
            rows, closed_two = scan_day(events, snap_groups, t1)
            # skip days that produced no in-window samples
            if not rows or closed_two is None:
                # still count the symbol-day so the heartbeat stays honest
                sd += 1
                continue
            # label this day one-sided or two-sided at the close
            ctype = "two" if closed_two else "one"
            # increment the matching day counter
            day_counts[sym][ctype] += 1
            # tag every sample with this day's close-type and stash it
            for r in rows:
                # attach the close-type to the sample
                r["close_type"] = ctype
                # append to this symbol's collection
                samples[sym].append(r)
            # advance the symbol-day counter
            sd += 1
            # heartbeat every 25 symbol-days
            if sd % 25 == 0:
                # elapsed wall-clock
                el = time.perf_counter() - t0_all
                # projected total time at the current rate
                proj = el / sd * sd_total
                # print progress + ETA
                print(f"  {sd}/{sd_total} symbol-days  elapsed {_fmt(el)}  "
                      f"ETA {_fmt(proj - el)}", flush=True)

    # bucket edges from 0 (bell) to WINDOW_S, spaced BUCKET_S apart
    edges = np.arange(0, WINDOW_S + BUCKET_S, BUCKET_S)
    # bucket centers, used as the x-axis / group labels
    centers = edges[:-1] + BUCKET_S / 2.0
    # a 2-row (two-sidedness / depth) x per-symbol grid of subplots
    fig, axes = plt.subplots(2, len(SYMBOLS), figsize=(7 * len(SYMBOLS), 9))
    # iterate symbols (column index j)
    for j, sym in enumerate(SYMBOLS):
        # assemble this symbol's samples into a dataframe
        df = pd.DataFrame(samples[sym])
        # skip symbols that produced nothing
        if len(df) == 0:
            # note it and move on
            print(f"{sym}: no samples")
            continue
        # assign each sample to a time-to-close bucket
        df["bucket"] = pd.cut(df["ttc_s"], bins=edges, labels=centers, include_lowest=True)
        # print the close-type day split for this symbol
        print(f"\n=== {sym} : {day_counts[sym]['two']} two-sided-close days, "
              f"{day_counts[sym]['one']} one-sided-close days ===")

        # TOP subplot handle: two-sidedness vs time-to-close, by close-type
        axT = axes[0, j]
        # draw one line per close-type (two-sided-close days blue, one-sided red)
        for ctype, color in (("two", "#1f77b4"), ("one", "#d62728")):
            # rows for this close-type
            sub = df[df["close_type"] == ctype]
            # skip if this close-type never occurred
            if len(sub) == 0:
                continue
            # mean two-sidedness per bucket (fraction of samples with both sides)
            frac = sub.groupby("bucket", observed=True)["two_sided"].mean()
            # plot fraction vs bucket center
            axT.plot(frac.index.astype(float), frac.values, "o-", color=color,
                     label=f"{ctype}-sided-close days")
        # vertical reference at the proposed T-120s early-flatten time
        axT.axvline(120, color="black", ls="--", lw=1, label="T-120s (proposed flatten)")
        # x-axis label
        axT.set_xlabel("seconds to close (0 = bell)")
        # y-axis label
        axT.set_ylabel("fraction of samples two-sided")
        # subplot title
        axT.set_title(f"{sym}: two-sidedness before the close")
        # invert x so time runs left(earlier)->right(bell)
        axT.invert_xaxis()
        # fraction axis fixed to [0, ~1]
        axT.set_ylim(0, 1.02)
        # legend
        axT.legend(fontsize=8)
        # light grid
        axT.grid(alpha=0.3)

        # BOTTOM subplot handle: thinner-touch depth vs time-to-close, one-sided days
        axB = axes[1, j]
        # one-sided-close days only (the days that hurt)
        one = df[df["close_type"] == "one"].copy()
        # only plot depth if such days exist
        if len(one):
            # tradable depth = the SMALLER of bid/ask depth (0 if a side is empty)
            one["min_depth"] = one[["bq", "aq"]].min(axis=1)
            # median of that per bucket
            med = one.groupby("bucket", observed=True)["min_depth"].median()
            # plot median depth vs bucket center
            axB.plot(med.index.astype(float), med.values, "s-", color="#d62728",
                     label="median min(bid,ask) depth")
        # horizontal reference at the residual you'd need to flatten
        axB.axhline(RESIDUAL_SH, color="green", ls=":", lw=1.5,
                    label=f"residual to flatten (~{RESIDUAL_SH} sh)")
        # vertical reference at T-120s
        axB.axvline(120, color="black", ls="--", lw=1)
        # x-axis label
        axB.set_xlabel("seconds to close (0 = bell)")
        # y-axis label
        axB.set_ylabel("shares at the thinner touch")
        # subplot title
        axB.set_title(f"{sym}: depth on one-sided-close days (can we cross early?)")
        # invert x to match the top subplot
        axB.invert_xaxis()
        # legend
        axB.legend(fontsize=8)
        # light grid
        axB.grid(alpha=0.3)

        # printed verdict: two-sidedness right around the T-120s mark, per close-type
        for ctype in ("one", "two"):
            # rows for this close-type
            sub = df[(df["close_type"] == ctype)]
            # skip if none
            if len(sub) == 0:
                continue
            # samples in a +/-10s band around T-120s
            near120 = sub[(sub["ttc_s"] >= 110) & (sub["ttc_s"] <= 130)]
            # print the two-sided fraction there if we have samples
            if len(near120):
                print(f"  {ctype}-sided-close days: at ~T-120s, "
                      f"{100 * near120['two_sided'].mean():.0f}% of samples still two-sided")

    # tidy layout
    fig.tight_layout()
    # write the figure to disk
    fig.savefig("close_thinning.png", dpi=140)
    # free the figure
    plt.close(fig)
    # report the output path
    print("\nwrote close_thinning.png")
    # explain how to read the result
    print("\nDECIDES: on the ONE-sided-close days, if two-sidedness at T-120s is")
    print("already low, the counterparty was gone well before the bell -> crossing")
    print("early can't help those days. If it's still high at T-120s and only")
    print("collapses later, a T-120s flatten reaches a real counterparty.")


# standard entry point
if __name__ == "__main__":
    # run it
    main()
