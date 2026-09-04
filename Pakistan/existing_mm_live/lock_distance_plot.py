# lock_distance_plot.py -- HOW MANY SPREADS is price from the 10% lock, over a day,
# and WHEN does it breach the trigger threshold -- comparing two ways to measure it:
#   * INSTANTANEOUS spread as the yardstick  (what the trigger uses today; jumpy)
#   * TRAILING spread as the yardstick        (the proposed fix; smooth)
# The lock_activation_diag showed the instantaneous version dips below the threshold
# spuriously when the spread momentarily widens. This plot shows, visually, whether
# a trailing-spread yardstick keeps the distance above the line except near a REAL lock.
#
# Output OVERWRITES a fixed absolute path each run (no stale-file confusion).
#
# Run in PyCharm:  python lock_distance_plot.py

# filesystem paths
from pathlib import Path
# ordered buffer for the tick window
from collections import deque
# dataframes
import pandas as pd
# numeric
import numpy as np
# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# driver (datasets, reads, to_ms, discover_dates)
import run_legacy_mm as R

# ------------------------------- knobs (edit) --------------------------------
# raw parsed store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# which symbol
SYMBOL = "PACE"
# auto-pick the most informative day (most two-sided ticks that ACTUALLY breaches);
# set False and fill DATE to force a specific day
AUTO_PICK_DAY = False
# used only when AUTO_PICK_DAY is False
DATE = "2026-06-30"
# trailing TIME window (seconds) and TICK window (events) for the trailing spread
TIME_WINDOW_S = 300
TICK_WINDOW = 100
# the breach lines, in spread-multiples: ramp-start (widen) and cliff (go dark).
# These are the distances at which the trigger would engage; when a distance curve
# dips BELOW a line, that's a breach (trigger fires).
RAMP_SPREADS = 20.0
CLIFF_SPREADS = 5.0
# the PERCENT-OF-PRICE breach lines (the metric we are ADOPTING). RAMP_PCT (start
# widening once price has moved to within this % of the band) and CLIFF_PCT (go
# dark once within this %). Expressed as "% of price still to the band": ramp fires
# at 2% remaining (i.e. +8% move on a 10% band), cliff at 0.5% remaining (+9.5%).
RAMP_PCT = 2.0
CLIFF_PCT = 0.5
# skip days with fewer than this many two-sided ticks (degenerate/halt days)
MIN_TICKS = 300
# where the image goes (absolute, overwritten each run)
OUT = Path("/Users/shazzak/Capital Stake - Results/exports/lock_distance.png")
# -----------------------------------------------------------------------------


# per-message best bid/ask/spread for the continuous phase of one symbol-day
def day_frame(s):
    # derive ts_exch (build_events not called here)
    s = s.copy()
    s["ts_exch"] = R.to_ms(s["orig_time"])
    # continuous only
    c = s[s["phase"] == "CONTINUOUS_AUCTION"]
    # day-constant published limits
    up = c.loc[c["entry_type"] == "UPPER_CIRCUIT_BREAKER", "px"].dropna()
    dn = c.loc[c["entry_type"] == "LOWER_CIRCUIT_BREAKER", "px"].dropna()
    lim_up = float(up.iloc[0]) if len(up) else None
    lim_dn = float(dn.iloc[0]) if len(dn) else None
    # best bid / ask per message
    bid = c[c["entry_type"] == "BID"].groupby("msg_seq")["px"].max()
    ask = c[c["entry_type"] == "OFFER"].groupby("msg_seq")["px"].min()
    ts = c.groupby("msg_seq")["ts_exch"].first()
    # two-sided, positive-spread, time-sorted
    m = pd.DataFrame({"ts": ts, "bid": bid, "ask": ask}).dropna().sort_values("ts")
    m["spread"] = m["ask"] - m["bid"]
    m = m[m["spread"] > 0].reset_index(drop=True)
    return m, lim_up, lim_dn


# rolling trailing spread: time-window and tick-window means (two-pointer, O(n))
def trailing_spread(m):
    ts = m["ts"].to_numpy()
    sp = m["spread"].to_numpy()
    win_ms = TIME_WINDOW_S * 1000
    time_avg = np.empty(len(sp))
    tick_avg = np.empty(len(sp))
    buf = deque(maxlen=TICK_WINDOW)
    tick_sum = 0.0
    left = 0
    time_sum = 0.0
    for i in range(len(sp)):
        # tick window
        if len(buf) == TICK_WINDOW:
            tick_sum -= buf[0]
        buf.append(sp[i]); tick_sum += sp[i]
        tick_avg[i] = tick_sum / len(buf)
        # time window
        time_sum += sp[i]
        while ts[i] - ts[left] > win_ms:
            time_sum -= sp[left]; left += 1
        time_avg[i] = time_sum / (i - left + 1)
    # the trailing yardstick = the larger of the two (robust: a momentarily tight
    # book can't shrink it, matching the cap-reference logic)
    return np.maximum(time_avg, tick_avg)


# distance to the NEAREST band, in price, in each spread-yardstick, and in percent
def distances(m, lim_up, lim_dn, trail):
    bid = m["bid"].to_numpy()
    ask = m["ask"].to_numpy()
    inst = m["spread"].to_numpy()
    # distance in PKR to each band, clamped at 0 (pinned)
    d_up = np.maximum(0.0, lim_up - bid)
    d_dn = np.maximum(0.0, ask - lim_dn)
    # nearest band distance in PKR
    d_price = np.minimum(d_up, d_dn)
    # d_inst (distance to lock in INSTANTANEOUS-spread widths -- current buggy metric)
    d_inst = d_price / inst
    # d_trail (distance to lock in TRAILING-avg-spread widths -- the smoothed patch)
    d_trail = d_price / trail
    # d_pct (distance to lock as PERCENT OF PRICE -- the spread-free metric we adopt).
    # Measured from the MID to the nearer band, NOT from the touch: from-mid is
    # invariant to spread width, whereas from-touch drifts as the bid/ask move when
    # the spread widens (a faint echo of the very bug we're removing). Fully
    # instantaneous (only the current mid + the fixed daily band), no spread involved.
    mid = 0.5 * (bid + ask)
    # distance from mid up to the upper band, and down to the lower band, in PKR
    dm_up = np.maximum(0.0, lim_up - mid)
    dm_dn = np.maximum(0.0, mid - lim_dn)
    # nearest-band distance from mid, as % of price
    d_pct = np.minimum(dm_up, dm_dn) / mid * 100.0
    return d_price, d_inst, d_trail, d_pct


# compute everything for one day; return a dict or None if unusable
def analyse(date):
    dsets = R.open_datasets(date)
    if dsets is None:
        return None
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, SYMBOL)
    if len(s) == 0:
        return None
    m, lim_up, lim_dn = day_frame(s)
    # need limits and enough ticks
    if lim_up is None or lim_dn is None or len(m) < MIN_TICKS:
        return None
    trail = trailing_spread(m)
    d_price, d_inst, d_trail, d_pct = distances(m, lim_up, lim_dn, trail)
    return {"date": str(date), "m": m, "d_inst": d_inst, "d_trail": d_trail,
            "d_pct": d_pct, "n": len(m), "min_trail": float(np.nanmin(d_trail))}


def main():
    # choose the day
    if AUTO_PICK_DAY:
        # scan every day; keep the one with the MOST ticks that actually breaches
        # the ramp line on the TRAILING metric (so the plot shows a real approach)
        print(f"auto-picking the best {SYMBOL} day (most ticks that breaches "
              f"{RAMP_SPREADS} trailing-spreads)...", flush=True)
        best = None
        for date in R.discover_dates():
            r = analyse(date)
            if r is None:
                continue
            # a breach = the trailing distance dipped below the ramp line that day
            breached = r["min_trail"] < RAMP_SPREADS
            # prefer breaching days; among them, the one with the most ticks
            key = (breached, r["n"])
            if best is None or key > best[0]:
                best = (key, r)
        if best is None:
            raise SystemExit(f"no usable {SYMBOL} days found")
        r = best[1]
        print(f"chosen day: {r['date']}  ({r['n']} ticks, "
              f"min trailing-distance {r['min_trail']:.1f} spreads)")
    else:
        # forced day
        r = analyse(DATE)
        if r is None:
            raise SystemExit(f"{SYMBOL} {DATE} not usable (missing limits or <{MIN_TICKS} ticks)")

    # unpack
    m = r["m"]
    # x-axis: minutes since the first continuous quote
    x_min = (m["ts"].to_numpy() - m["ts"].iloc[0]) / 60000.0

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(13, 6))
    # PRIMARY (left) axis = the two SPREAD-BASED metrics (both in spread multiples)
    # d_inst (distance to lock in INSTANTANEOUS spreads -- current metric): faint/jumpy
    ax.plot(x_min, r["d_inst"], color="#bbbbbb", lw=0.8,
            label="d_inst: dist in INSTANTANEOUS spreads (current)")
    # d_trail (distance to lock in TRAILING-avg spreads -- the smoothed patch): bold
    ax.plot(x_min, r["d_trail"], color="#1f77b4", lw=1.6,
            label="d_trail: dist in TRAILING spreads (patch)")
    # spread-metric breach lines (left-axis units): ramp 20, cliff 5
    ax.axhline(RAMP_SPREADS, color="#ff7f0e", ls="--", lw=1.1,
               label=f"spread ramp ({RAMP_SPREADS:.0f})")
    ax.axhline(CLIFF_SPREADS, color="#d62728", ls="--", lw=1.1,
               label=f"spread cliff ({CLIFF_SPREADS:.0f})")
    # left axis kept to its natural spread-multiple range (breach zone legible)
    ax.set_ylim(0, RAMP_SPREADS * 3)
    ax.set_xlabel("minutes since continuous open")
    ax.set_ylabel("LEFT: distance to lock (spread multiples)")
    ax.grid(alpha=0.3)

    # SECONDARY (right) axis = the PERCENT-OF-PRICE metric (the one we ADOPT).
    # Independent, natural percent scale -- NOT aligned to the left axis, so the two
    # scales are not forced to imply a false relationship.
    ax2 = ax.twinx()
    # d_pct (distance to lock as % OF PRICE -- spread-free, fully instantaneous)
    ax2.plot(x_min, r["d_pct"], color="#2ca02c", lw=1.6,
             label="d_pct: dist as % of price (ADOPTED)")
    # percent breach lines: ramp 2% remaining (+8% move), cliff 0.5% (+9.5%)
    ax2.axhline(RAMP_PCT, color="#2ca02c", ls=":", lw=1.3,
                label=f"% ramp ({RAMP_PCT:.1f}% remaining = +{10-RAMP_PCT:.1f}% move)")
    ax2.axhline(CLIFF_PCT, color="#8c564b", ls=":", lw=1.3,
                label=f"% cliff ({CLIFF_PCT:.1f}% remaining = +{10-CLIFF_PCT:.1f}% move)")
    # right axis: 0 to just above the band (~11%) so the low breach lines are visible
    ax2.set_ylim(0, 11.0)
    ax2.set_ylabel("RIGHT: distance to lock (% of price)  [adopted metric]", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # combined legend (both axes) in one box
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=2)

    # title
    ax.set_title(f"{SYMBOL} {r['date']}: distance to the 10% lock\n"
                 f"LEFT = spread-based metrics (rejected: d_inst jumpy, d_trail smoothing "
                 f"a bad signal) | RIGHT = % of price (adopted)")
    # make sure the export folder exists, then OVERWRITE the image
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    plt.close(fig)

    # numeric read: breach rates for each metric on its own threshold
    inst_breach = float(np.mean(r["d_inst"] < RAMP_SPREADS)) * 100
    trail_breach = float(np.mean(r["d_trail"] < RAMP_SPREADS)) * 100
    pct_breach = float(np.mean(r["d_pct"] < RAMP_PCT)) * 100
    print(f"\nwrote {OUT}")
    print(f"  ramp-line breaches (fires): d_inst {inst_breach:.1f}% | "
          f"d_trail {trail_breach:.1f}% | d_pct {pct_breach:.1f}% of ticks")
    print("  READ: d_pct (green, right axis) should sit near the top of its range")
    print("  (far from the band) almost all day and only dip toward the % breach")
    print("  lines during a genuine approach -- spread-free, so no wide-spread noise.")


if __name__ == "__main__":
    main()
