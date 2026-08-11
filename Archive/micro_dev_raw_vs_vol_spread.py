"""
micro_dev_raw_vs_vol_spread.py

Literal request: plot the raw micro_dev feature against daily vol and daily spread, per ticker.

Daily statistic = median |micro_dev_bps|  (the MAGNITUDE).
Reason: micro_dev = imbalance * half-spread, so the SIGNED daily median ~ 0 (book leans both ways
over a day) and would show no relationship. The magnitude is what scales with spread/vol.
The signed daily median is also computed and printed so you can confirm it sits near zero.

Per ticker, one figure:
  top-left  : |micro_dev| and vol over time (twin axis)     -> micro_dev vs VOL over time
  top-right : |micro_dev| and spread over time (twin axis)  -> micro_dev vs SPREAD over time
  bottom-left : scatter |micro_dev| vs vol      (Spearman annotated, coloured by date)
  bottom-right: scatter |micro_dev| vs spread   (Spearman annotated, coloured by date)

All series smoothed with a trailing median before plotting.
"""

# stdlib + numeric/plot stack + wiring for partition enumeration
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import feature_store_wiring as W

# ---------------- CONFIG ----------------

# raw feature-store column names
MICRO_COL, VOL_COL, SPREAD_COL = "micro_dev_bps", "realized_vol_bps", "spread_bps"
# where to cache the daily aggregation and write figures
OUT_DIR = "/Users/shazzak/Capital Stake - Results/markout_validation"
# cache path for the one-time raw pass
CACHE = os.path.join(OUT_DIR, "daily_micro_vol_spread.parquet")
# trailing-median smoothing window (trading days)
SMOOTH, MIN_PERIODS = 10, 5


# ---------------- IO ----------------

def load_parquet(path):
    # lazy duckdb import
    import duckdb
    # read parquet via duckdb into pandas
    return duckdb.connect().execute(f"SELECT * FROM read_parquet('{path}')").df()


# ---------------- daily aggregation from raw feature store (cached) ----------------

def build_daily():
    # reuse cache if present (skip the heavy raw pass)
    if os.path.exists(CACHE):
        # load cached daily table
        return load_parquet(CACHE)
    # otherwise aggregate one median-row per symbol-day
    import duckdb
    # single connection
    con = duckdb.connect()
    # accumulate rows
    rows = []
    # all (symbol, date, path)
    parts = W.enumerate_partitions()
    # one light pass per file
    for i, (sym, dt, path) in enumerate(parts, 1):
        # median |micro_dev| (magnitude), signed median micro_dev (~0 check), median vol, median spread
        r = con.execute(
            f"SELECT median(abs({MICRO_COL})) AS micro_abs, "
            f"       median({MICRO_COL})       AS micro_signed, "
            f"       median({VOL_COL})         AS vol, "
            f"       median({SPREAD_COL})      AS spread "
            f"FROM read_parquet('{path}')"
        ).df().iloc[0]
        # record
        rows.append({"symbol": sym, "date": dt,
                     "micro_abs": r["micro_abs"], "micro_signed": r["micro_signed"],
                     "vol": r["vol"], "spread": r["spread"]})
        # progress
        if i % 50 == 0:
            print(f"  agg {i}/{len(parts)}")
    # assemble and cache
    out = pd.DataFrame(rows)
    out.to_parquet(CACHE)
    return out


# ---------------- smoothing ----------------

def smooth(g):
    # sort by date and index by datetime
    g = g.sort_values("dt").copy()
    # trailing-median smooth each series (leakage-free: uses only past+current)
    for c in ["micro_abs", "vol", "spread"]:
        g[c + "_s"] = g[c].rolling(SMOOTH, min_periods=MIN_PERIODS).median()
    # drop the initial rows with no smoothed value
    return g.dropna(subset=["micro_abs_s", "vol_s", "spread_s"])


# ---------------- per-ticker plot ----------------

def plot_ticker(sym, g):
    # 2x2 figure
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    # datetime index for the time-series panels
    gi = g.set_index("dt")

    # --- top-left: |micro_dev| (left) and vol (right) over time ---
    a = ax[0, 0]
    a.plot(gi.index, gi["micro_abs_s"], color="tab:orange", lw=2, label="|micro_dev| (smoothed)")
    a.set_ylabel("median |micro_dev_bps|", color="tab:orange")
    a2 = a.twinx()
    a2.plot(gi.index, gi["vol_s"], color="tab:green", lw=2, label="vol (smoothed)")
    a2.set_ylabel("median realized_vol_bps", color="tab:green")
    a.set_title(f"{sym}: |micro_dev| vs VOL over time")

    # --- top-right: |micro_dev| (left) and spread (right) over time ---
    b = ax[0, 1]
    b.plot(gi.index, gi["micro_abs_s"], color="tab:orange", lw=2)
    b.set_ylabel("median |micro_dev_bps|", color="tab:orange")
    b2 = b.twinx()
    b2.plot(gi.index, gi["spread_s"], color="tab:purple", lw=2)
    b2.set_ylabel("median spread_bps", color="tab:purple")
    b.set_title(f"{sym}: |micro_dev| vs SPREAD over time")

    # --- Spearman correlations on the smoothed series ---
    rv = g["micro_abs_s"].corr(g["vol_s"], method="spearman")
    rs = g["micro_abs_s"].corr(g["spread_s"], method="spearman")
    # a time index for colouring the scatter (shows trajectory)
    tcol = np.arange(len(g))

    # --- bottom-left: scatter |micro_dev| vs vol ---
    c = ax[1, 0]
    sc = c.scatter(g["vol_s"], g["micro_abs_s"], c=tcol, cmap="viridis", s=18)
    c.set_xlabel("median vol (bps, smoothed)")
    c.set_ylabel("median |micro_dev| (bps, smoothed)")
    c.set_title(f"{sym}: |micro_dev| vs VOL   (Spearman {rv:+.2f})")
    fig.colorbar(sc, ax=c, label="time -->")

    # --- bottom-right: scatter |micro_dev| vs spread ---
    d = ax[1, 1]
    sc2 = d.scatter(g["spread_s"], g["micro_abs_s"], c=tcol, cmap="viridis", s=18)
    d.set_xlabel("median spread (bps, smoothed)")
    d.set_ylabel("median |micro_dev| (bps, smoothed)")
    d.set_title(f"{sym}: |micro_dev| vs SPREAD   (Spearman {rs:+.2f})")
    fig.colorbar(sc2, ax=d, label="time -->")

    # save
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"micro_dev_raw_vs_vol_spread_{sym}.png")
    plt.savefig(out, dpi=130)
    plt.close(fig)
    # report + the signed-median sanity check
    print(f"[{sym}] saved {out}")
    print(f"   Spearman: |micro_dev| vs vol = {rv:+.2f} | vs spread = {rs:+.2f}")
    print(f"   signed micro_dev daily median: mean over days = {g['micro_signed'].mean():+.4f} bps "
          f"(near zero by construction -> why magnitude is used)")


# ---------------- main ----------------

def main():
    # daily table (cached raw aggregation)
    daily = build_daily()
    # datetime for ordering/plotting
    daily["dt"] = pd.to_datetime(daily["date"].astype(str))
    # per ticker: smooth then plot
    for sym in sorted(daily["symbol"].unique()):
        # smooth this symbol's series
        g = smooth(daily[daily.symbol == sym])
        # plot
        plot_ticker(sym, g)


# entry point
if __name__ == "__main__":
    main()
