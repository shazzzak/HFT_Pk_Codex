"""
effectiveness_over_time.py

Did OBI / micro_dev effectiveness drift over the 207-day sample?
Built entirely from the daily decile summaries you already have -- no raw data needed.

Two bounded, outlier-resistant effectiveness metrics per (symbol, day):
  * Spearman rho between decile (1..10) and mean markout  -> "is the monotone shape still there?"  in [-1, 1]
  * top-minus-bottom edge (bps)                           -> magnitude (read via rolling MEDIAN, not mean)
Plus a rolling hit-rate (fraction of trailing days with edge > 0) -> "how often does it still work?"
"""

# pandas for frames, numpy for math, matplotlib for the time-series plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# os for building output paths
import os

# ---------------- CONFIG (edit these to your machine) ----------------

# folder holding the two decile-summary parquets
SUMMARY_DIR = "/Users/shazzak/Capital Stake - Results/markout_validation"

# feature name -> summary filename
FILES = {
    "obi_1":         "obi_1_daily_decile_summary.parquet",
    "micro_dev_bps": "micro_dev_bps_daily_decile_summary.parquet",
}

# which markout column to score ('mean_markout_all' = zeros-included; 'mean_markout_nonzero' = mid-moved only)
VALUE_COL = "mean_markout_all"

# the day column in the summary is named 'date' (VARCHAR) -- confirmed from your schema
DATE_COL = "date"

# rolling window in TRADING DAYS (21 ~ one month); a bias/variance smoothing choice, tune to taste
ROLL = 21

# minimum days before a rolling value is shown (avoids jumpy early estimates)
MIN_PERIODS = 10

# where to write the figures
OUT_DIR = SUMMARY_DIR


# ---------------- IO ----------------

def load_summary(path):
    # import duckdb lazily so this module imports even where duckdb is absent
    import duckdb
    # read the parquet straight into a pandas DataFrame via duckdb (no pyarrow needed)
    return duckdb.connect().execute(f"SELECT * FROM read_parquet('{path}')").df()


# ---------------- METRICS ----------------

def daily_metrics(df, value_col):
    # collect one record per (symbol, day)
    recs = []
    # iterate each symbol-day block (its up-to-10 decile rows)
    for (sym, day), sub in df.groupby(["symbol", DATE_COL]):
        # order the deciles ascending so 1..10 is clean
        sub = sub.sort_values("decile")
        # skip degenerate days that somehow have too few deciles to judge shape
        if sub["decile"].nunique() < 3:
            continue
        # Spearman rank corr between decile and markout: monotonicity in [-1,1], outlier-immune
        rho = sub[value_col].corr(sub["decile"], method="spearman")
        # top decile mean markout (highest decile present that day)
        top = sub.loc[sub["decile"] == sub["decile"].max(), value_col].iloc[0]
        # bottom decile mean markout (lowest decile present)
        bot = sub.loc[sub["decile"] == sub["decile"].min(), value_col].iloc[0]
        # the magnitude edge in bps
        edge = top - bot
        # store the day's metrics
        recs.append({"symbol": sym, "date": day, "rho": rho, "edge": edge})
    # assemble into a frame
    out = pd.DataFrame(recs)
    # parse the string date into a real datetime for a proper time axis
    out["date"] = pd.to_datetime(out["date"])
    # sort by symbol then time
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def add_rolling(g):
    # sort this symbol's series by date and index by date
    g = g.sort_values("date").set_index("date").copy()
    # rolling MEDIAN of rho: robust "shape effectiveness" trend
    g["rho_roll"] = g["rho"].rolling(ROLL, min_periods=MIN_PERIODS).median()
    # rolling hit-rate: fraction of trailing window with a positive edge
    g["hit_roll"] = (g["edge"] > 0).astype(float).rolling(ROLL, min_periods=MIN_PERIODS).mean()
    # rolling MEDIAN of edge (bps): robust magnitude trend, immune to blow-up days
    g["edge_roll"] = g["edge"].rolling(ROLL, min_periods=MIN_PERIODS).median()
    # hand back the enriched frame (date is the index)
    return g


# ---------------- PLOT ----------------

def plot_feature(feature, metrics):
    # one figure, three stacked panels sharing the time axis
    fig, (ax_rho, ax_hit, ax_edge) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    # a fixed colour per symbol so the two lines are distinguishable
    colors = {"UBL": "tab:blue", "PPL": "tab:orange"}

    # draw each symbol's three series
    for sym, g in metrics.groupby("symbol"):
        # compute rolling columns for this symbol
        gr = add_rolling(g)
        # pick a colour (fallback grey for unexpected symbols)
        c = colors.get(sym, "grey")

        # --- panel 1: Spearman rho ---
        # faint raw daily rho so you can see the noise behind the trend
        ax_rho.plot(gr.index, gr["rho"], color=c, alpha=0.12, linewidth=0.7)
        # bold rolling-median rho: the effectiveness trend
        ax_rho.plot(gr.index, gr["rho_roll"], color=c, linewidth=2, label=f"{sym}")

        # --- panel 2: rolling hit-rate ---
        # bold rolling hit-rate (raw hit is 0/1 so plotting it is meaningless)
        ax_hit.plot(gr.index, gr["hit_roll"], color=c, linewidth=2, label=f"{sym}")

        # --- panel 3: rolling-median edge (bps) ---
        # bold rolling-median edge magnitude
        ax_edge.plot(gr.index, gr["edge_roll"], color=c, linewidth=2, label=f"{sym}")

    # panel 1 cosmetics: zero = no monotone relationship
    ax_rho.axhline(0, color="red", linestyle="--", linewidth=1)
    # rho is bounded, so fix the y-range for honest comparison
    ax_rho.set_ylim(-1.05, 1.05)
    # label
    ax_rho.set_ylabel("Spearman rho\n(decile vs markout)")
    # title carries the feature and window
    ax_rho.set_title(f"{feature}: effectiveness over time  (rolling {ROLL} trading days, median)")
    # legend
    ax_rho.legend(loc="lower left")

    # panel 2 cosmetics: 0.5 = coin flip, the death line
    ax_hit.axhline(0.5, color="red", linestyle="--", linewidth=1)
    # hit-rate is a fraction
    ax_hit.set_ylim(0, 1)
    # label
    ax_hit.set_ylabel("rolling hit-rate\n(P[edge > 0])")
    # legend
    ax_hit.legend(loc="lower left")

    # panel 3 cosmetics: zero reference
    ax_edge.axhline(0, color="red", linestyle="--", linewidth=1)
    # label
    ax_edge.set_ylabel("rolling median\nedge (bps)")
    # x label only on the bottom panel
    ax_edge.set_xlabel("date")
    # legend
    ax_edge.legend(loc="lower left")

    # tidy spacing
    plt.tight_layout()
    # build the output path
    out_png = os.path.join(OUT_DIR, f"{feature}_effectiveness_over_time.png")
    # save the figure
    plt.savefig(out_png, dpi=130)
    # free the figure
    plt.close(fig)
    # report where it went
    print(f"[{feature}] saved {out_png}")


# ---------------- TEXT DRIFT READOUT ----------------

def print_drift(feature, metrics):
    # header
    print(f"\n[{feature}] first-half vs second-half drift:")
    # per symbol, split the time-ordered series in half and compare
    for sym, g in metrics.groupby("symbol"):
        # order by date
        g = g.sort_values("date")
        # midpoint index
        h = len(g) // 2
        # first and second halves
        first, second = g.iloc[:h], g.iloc[h:]
        # hit-rate (fraction positive edge) in each half
        hit1, hit2 = (first["edge"] > 0).mean(), (second["edge"] > 0).mean()
        # mean rho in each half
        rho1, rho2 = first["rho"].mean(), second["rho"].mean()
        # print the comparison
        print(f"   {sym}: hit-rate {hit1:.0%} -> {hit2:.0%} | mean rho {rho1:+.2f} -> {rho2:+.2f} "
              f"| n={len(g)} days")


# ---------------- MAIN ----------------

def main():
    # loop over both features
    for feature, fname in FILES.items():
        # full path to this feature's summary
        path = os.path.join(SUMMARY_DIR, fname)
        # load it
        df = load_summary(path)
        # compute per-(symbol, day) metrics
        metrics = daily_metrics(df, VALUE_COL)
        # draw the three-panel time series
        plot_feature(feature, metrics)
        # print the half-vs-half drift table
        print_drift(feature, metrics)


# standard entry point
if __name__ == "__main__":
    # run everything
    main()
