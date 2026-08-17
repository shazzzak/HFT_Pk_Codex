# daily_pnl_plot.py -- is the P&L a steady grind or a few outlier days?
# Reads the per-day P&L that confirm ALREADY saves: eod_positions.csv has one row
# per (symbol, variant, date) and its 'liquidated' column IS that day's P&L in PKR
# (verified: summing it reproduces total_pnl_pkr exactly). No new run needed.
#
# Produces, per symbol: cumulative P&L curves (steady = straight line; outlier-driven
# = staircase with big jumps) and a daily-P&L bar panel for the chosen variants.
# Prints win-rate and top-N-day concentration so the read is numeric, not just visual.
#
# Output OVERWRITES a fixed absolute path each run.
#
# Run in PyCharm:  python daily_pnl_plot.py

# paths
from pathlib import Path
# dataframes
import pandas as pd
# numeric
import numpy as np
# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------- knobs (edit) --------------------------------
# where confirm wrote the per-day file
EOD_CSV = Path("/Users/shazzak/Capital Stake - Results/eod_positions.csv")
# which variants to draw (keep it few or the chart gets unreadable)
VARIANTS = ["naive", "MID", "MID+BOTH"]
# symbols to plot (one row of panels each)
SYMBOLS = ["PPL", "UBL", "PACE"]
# output image (absolute, overwritten)
OUT = Path("/Users/shazzak/Capital Stake - Results/exports/daily_pnl.png")
# -----------------------------------------------------------------------------


# concentration stats for one series of daily P&L
def concentration(x):
    # total across all days
    tot = x.sum()
    # sorted best-to-worst
    srt = np.sort(x)[::-1]
    # share of the total contributed by the best N days
    out = {}
    for n in (1, 5, 10, 20):
        # guard against series shorter than n
        if len(srt) >= n and tot != 0:
            out[n] = 100.0 * srt[:n].sum() / tot
        else:
            out[n] = np.nan
    # percentage of days that were profitable
    out["win"] = 100.0 * np.mean(x > 0)
    # median and mean day
    out["med"] = float(np.median(x))
    out["mean"] = float(np.mean(x))
    # total with the best 5 days removed (does the edge survive without them?)
    out["ex5"] = float(tot - srt[:5].sum()) if len(srt) >= 5 else np.nan
    # the raw total
    out["tot"] = float(tot)
    return out


def main():
    # load the per-day table
    e = pd.read_csv(EOD_CSV)
    # parse dates so the x-axis is real time
    e["date"] = pd.to_datetime(e["date"])
    # 'liquidated' is the day's P&L in PKR
    e = e.rename(columns={"liquidated": "pnl"})

    # two rows of panels per symbol: cumulative (top), daily bars (bottom)
    fig, axes = plt.subplots(2, len(SYMBOLS), figsize=(6.5 * len(SYMBOLS), 9))
    # colour per variant, consistent across panels
    colors = {"naive": "#d62728", "MID": "#1f77b4", "MID+BOTH": "#2ca02c"}

    # printed stats table
    print(f"{'sym':5s} {'variant':9s} {'total':>10s} {'win%':>5s} {'med/day':>8s} "
          f"{'top1%':>6s} {'top5%':>6s} {'top20%':>7s} {'excl-top5':>10s}")
    # walk symbols
    for j, sym in enumerate(SYMBOLS):
        # top panel: cumulative P&L
        axT = axes[0, j]
        # bottom panel: daily bars (only the first variant, else unreadable)
        axB = axes[1, j]
        # per variant
        for v in VARIANTS:
            # this symbol+variant, time-ordered
            d = e[(e.symbol == sym) & (e.variant == v)].sort_values("date")
            # skip if missing
            if len(d) == 0:
                continue
            # daily P&L array
            x = d["pnl"].to_numpy()
            # cumulative sum over the period
            cum = np.cumsum(x)
            # draw the cumulative curve
            axT.plot(d["date"], cum, lw=1.6, color=colors.get(v), label=v)
            # stats
            st = concentration(x)
            print(f"{sym:5s} {v:9s} {st['tot']:>10,.0f} {st['win']:>5.0f} "
                  f"{st['med']:>8,.0f} {st[1]:>6.1f} {st[5]:>6.1f} {st[20]:>7.1f} "
                  f"{st['ex5']:>10,.0f}")
        # cumulative panel dressing
        axT.axhline(0, color="black", lw=0.8)
        axT.set_title(f"{sym}: cumulative P&L\n(straight line = steady; big steps = outlier-driven)")
        axT.set_ylabel("cumulative P&L (PKR)")
        axT.legend(fontsize=8)
        axT.grid(alpha=0.3)
        axT.tick_params(axis="x", rotation=45, labelsize=7)

        # daily bars for the FIRST variant only (visual sense of day-to-day spread)
        v0 = VARIANTS[0]
        d0 = e[(e.symbol == sym) & (e.variant == v0)].sort_values("date")
        if len(d0):
            x0 = d0["pnl"].to_numpy()
            # green bars for profitable days, red for losing days
            axB.bar(d0["date"], x0, width=1.0,
                    color=np.where(x0 > 0, "#2ca02c", "#d62728"))
            axB.axhline(0, color="black", lw=0.8)
        axB.set_title(f"{sym}: daily P&L — {v0}")
        axB.set_ylabel("daily P&L (PKR)")
        axB.grid(alpha=0.3)
        axB.tick_params(axis="x", rotation=45, labelsize=7)

    # write it out (overwrite)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"\nwrote {OUT}")
    print("\nREAD: 'top5%' = share of total P&L from the 5 best days. Under ~20% with a")
    print("high win-rate means a steady grind (the edge is real and repeatable).")
    print("Over ~50% means a few lucky days carry it -- treat the headline with suspicion.")


if __name__ == "__main__":
    main()
