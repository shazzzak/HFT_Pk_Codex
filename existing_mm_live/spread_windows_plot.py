# spread_windows_plot.py -- time series of four spread measures for ONE symbol-day:
#   1. avg spread over the trailing TIME_WINDOW_S seconds   (true rolling time window)
#   2. avg spread over the trailing TICK_WINDOW ticks/events (true rolling count window)
#   3. max(1, 2)  -- the dual-window reference we considered for the widen cap
#   4. the current (instantaneous) spread
# Prototypes the "time-window vs tick-window, take the max" upgrade: shows whether
# the two windows actually diverge enough to matter on this name/day.
#
# Run in PyCharm (uses the .backtest interpreter):
#     python spread_windows_plot.py

# filesystem paths
from pathlib import Path
# ordered container for the rolling tick window
from collections import deque
# dataframes
import pandas as pd
# numeric
import numpy as np
# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# driver (datasets, reads, to_ms)
import run_legacy_mm as R

# ------------------------------- knobs (edit) --------------------------------
# raw parsed store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# which symbol to plot
SYMBOL = "PACE"
# which trading day (YYYY-MM-DD); pick a lock-heavy day for PACE
DATE = "2026-05-08"
# trailing TIME window, in seconds (the "YY")
TIME_WINDOW_S = 300
# trailing TICK/event window, in number of book updates (the "XX")
TICK_WINDOW = 100
# output image
OUT = "spread_windows.png"
# -----------------------------------------------------------------------------


# build the per-message best bid/ask/spread series for the continuous phase
def spread_series(s):
    # ts_exch isn't present until build_events; derive it as build_events does
    s = s.copy()
    # exchange-ms timestamp from orig_time
    s["ts_exch"] = R.to_ms(s["orig_time"])
    # continuous-trading snapshots only (spreads outside are meaningless)
    c = s[s["phase"] == "CONTINUOUS_AUCTION"]
    # best bid per message (max BID px)
    bid = c[c["entry_type"] == "BID"].groupby("msg_seq")["px"].max()
    # best ask per message (min OFFER px)
    ask = c[c["entry_type"] == "OFFER"].groupby("msg_seq")["px"].min()
    # message timestamp
    ts = c.groupby("msg_seq")["ts_exch"].first()
    # assemble, keep only two-sided messages (both bid and ask present), sort by time
    m = pd.DataFrame({"ts": ts, "bid": bid, "ask": ask}).dropna().sort_values("ts")
    # spread in PKR
    m["spread"] = m["ask"] - m["bid"]
    # drop non-positive spreads (crossed/degenerate)
    m = m[m["spread"] > 0].reset_index(drop=True)
    return m


# compute the four trailing measures walking the series once (O(n))
def rolling_measures(m):
    # timestamps as a numpy array (ms)
    ts = m["ts"].to_numpy()
    # spreads as a numpy array (PKR)
    sp = m["spread"].to_numpy()
    # window length in ms
    win_ms = TIME_WINDOW_S * 1000
    # outputs
    time_avg = np.empty(len(sp))
    tick_avg = np.empty(len(sp))
    # a deque of recent spreads for the TICK window (fixed max length)
    tickbuf = deque(maxlen=TICK_WINDOW)
    # running sum for the tick window (O(1) updates)
    tick_sum = 0.0
    # left pointer for the TIME window (two-pointer sliding sum)
    left = 0
    # running sum for the time window
    time_sum = 0.0
    # walk each event in time order
    for i in range(len(sp)):
        # --- tick window: push newest, pop oldest if the deque was full ---
        if len(tickbuf) == TICK_WINDOW:
            # subtract the value about to be evicted
            tick_sum -= tickbuf[0]
        # add the new spread
        tickbuf.append(sp[i])
        tick_sum += sp[i]
        # mean over however many are in the buffer (<= TICK_WINDOW)
        tick_avg[i] = tick_sum / len(tickbuf)
        # --- time window: extend the running sum, then shrink from the left ---
        time_sum += sp[i]
        # evict anything older than win_ms behind the current timestamp
        while ts[i] - ts[left] > win_ms:
            time_sum -= sp[left]
            left += 1
        # mean over the events still inside the time window
        time_avg[i] = time_sum / (i - left + 1)
    # the dual-window reference = elementwise max of the two
    dual_max = np.maximum(time_avg, tick_avg)
    # return all four aligned to m
    return time_avg, tick_avg, dual_max, sp


def main():
    # open the day's datasets
    dsets = R.open_datasets(DATE)
    # guard
    if dsets is None:
        raise SystemExit(f"no datasets for {DATE}")
    # read the symbol's snapshots
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, SYMBOL)
    # guard
    if len(s) == 0:
        raise SystemExit(f"no snapshot rows for {SYMBOL} on {DATE}")
    # build the spread series
    m = spread_series(s)
    # guard: need some two-sided continuous data
    if len(m) == 0:
        raise SystemExit(f"no two-sided continuous spreads for {SYMBOL} on {DATE}")
    # compute the four measures
    time_avg, tick_avg, dual_max, cur = rolling_measures(m)
    # x-axis: minutes since the first continuous quote (readable)
    x_min = (m["ts"].to_numpy() - m["ts"].iloc[0]) / 60000.0

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(13, 6))
    # current spread: thin/faint (it's the noisy raw signal)
    ax.plot(x_min, cur, color="#bbbbbb", lw=0.6, label="current (instantaneous) spread")
    # trailing time-window average
    ax.plot(x_min, time_avg, color="#1f77b4", lw=1.4,
            label=f"trailing {TIME_WINDOW_S}s time-window avg")
    # trailing tick-window average
    ax.plot(x_min, tick_avg, color="#2ca02c", lw=1.4,
            label=f"trailing {TICK_WINDOW}-tick window avg")
    # the dual-window max (the candidate cap reference)
    ax.plot(x_min, dual_max, color="#d62728", lw=1.6, ls="--",
            label="max(time, tick)  [dual-window reference]")
    # labels
    ax.set_xlabel("minutes since continuous open")
    ax.set_ylabel("spread (PKR)")
    ax.set_title(f"{SYMBOL} {DATE}: spread measures\n"
                 f"time-window ({TIME_WINDOW_S}s) vs tick-window ({TICK_WINDOW}) vs max vs current")
    # legend + grid
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # save
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    # report + a quick numeric read on whether the two windows diverge
    print(f"wrote {OUT}  ({len(m)} continuous two-sided ticks)")
    # how often does time-window exceed tick-window (and vice versa)?
    t_gt = float((time_avg > tick_avg).mean()) * 100
    print(f"time-window > tick-window on {t_gt:.0f}% of ticks; "
          f"tick-window > time-window on {100 - t_gt:.0f}%")
    # median gap between them (relative to the spread level)
    med_gap = float(np.median(np.abs(time_avg - tick_avg)))
    med_sp = float(np.median(cur))
    print(f"median |time-avg - tick-avg| = {med_gap:.4f} PKR "
          f"({100 * med_gap / med_sp:.1f}% of median spread)")
    print("\nREAD: if the two windows track each other closely (small gap, ~50/50),")
    print("the dual-window upgrade buys little over a single window. If they diverge")
    print("materially (esp. near lock-driven activity spikes), the max() is doing work.")


if __name__ == "__main__":
    main()
