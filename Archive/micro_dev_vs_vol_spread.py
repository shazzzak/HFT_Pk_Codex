"""
micro_dev_vs_vol_spread.py

Tests the mechanism: does micro_dev's *effectiveness* rise with volatility and fall with spread,
while OBI's (control) does not?

Per ticker it produces:
  * a time-series panel (micro_dev vs OBI rolling rho; daily vol; daily spread), and
  * two scatters: effectiveness vs vol, and effectiveness vs spread, OBI overlaid as control,
    each annotated with a Spearman correlation, plus vol/spread-tercile means.
And it prints a per-ticker tercile table (mean effectiveness in low/mid/high vol and spread).

Effectiveness = daily Spearman rho(decile -> markout) from the decile summaries.
Vol/spread    = daily median of realized_vol_bps / spread_bps from the raw feature store.
"""

# stdlib + numeric/plot stack + the wiring module for partition enumeration
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import feature_store_wiring as W

# ---------------- CONFIG ----------------

# decile summaries (daily effectiveness source)
SUMMARY_DIR = "/Users/shazzak/Capital Stake - Results/markout_validation"
# feature name -> summary filename
SUMMARY_FILES = {"micro_dev_bps": "micro_dev_bps_daily_decile_summary.parquet",
                 "obi_1":         "obi_1_daily_decile_summary.parquet"}
# which markout column the effectiveness is scored on
VALUE_COL = "mean_markout_all"
# day column in the summaries
DATE_COL = "date"
# raw feature-store columns for the daily descriptors
VOL_COL, SPREAD_COL = "realized_vol_bps", "spread_bps"
# cache for the raw daily vol/spread aggregation (so the raw pass runs once)
VOLSPREAD_CACHE = os.path.join(SUMMARY_DIR, "daily_vol_spread.parquet")
# rolling window for the time-series panel
ROLL, MIN_PERIODS = 21, 10
# output dir
OUT_DIR = SUMMARY_DIR


# ---------------- IO ----------------

def load_parquet(path):
    # lazy duckdb import (module still imports where duckdb is absent)
    import duckdb
    # read parquet into pandas via duckdb
    return duckdb.connect().execute(f"SELECT * FROM read_parquet('{path}')").df()


# ---------------- daily effectiveness (rho) from a decile summary ----------------

def daily_rho(df, value_col):
    # one rho per (symbol, day)
    recs = []
    # iterate symbol-day decile blocks
    for (sym, day), sub in df.groupby(["symbol", DATE_COL]):
        # order deciles
        sub = sub.sort_values("decile")
        # need enough deciles to judge monotonicity
        if sub["decile"].nunique() < 3:
            continue
        # Spearman rho between decile and markout
        rho = sub[value_col].corr(sub["decile"], method="spearman")
        # store
        recs.append({"symbol": sym, "date": str(day), "rho": rho})
    # frame of daily rho
    return pd.DataFrame(recs)


# ---------------- raw daily vol/spread (median per symbol-day), cached ----------------

def build_vol_spread():
    # reuse the cache if present so we do the heavy raw pass only once
    if os.path.exists(VOLSPREAD_CACHE):
        # load cached daily descriptors
        return load_parquet(VOLSPREAD_CACHE)
    # otherwise aggregate from the raw feature store
    import duckdb
    # one connection
    con = duckdb.connect()
    # accumulate one row per symbol-day
    rows = []
    # loop partitions (bounded memory: one file at a time)
    parts = W.enumerate_partitions()
    # progress + aggregation
    for i, (sym, dt, path) in enumerate(parts, 1):
        # median vol and spread over the whole day (robust to skew)
        r = con.execute(
            f"SELECT median({VOL_COL}) AS vol, median({SPREAD_COL}) AS spread "
            f"FROM read_parquet('{path}')"
        ).df().iloc[0]
        # record it
        rows.append({"symbol": sym, "date": dt, "vol": r["vol"], "spread": r["spread"]})
        # light progress
        if i % 50 == 0:
            print(f"  vol/spread agg {i}/{len(parts)}")
    # assemble
    out = pd.DataFrame(rows)
    # cache for next time
    out.to_parquet(VOLSPREAD_CACHE)
    # return
    return out


# ---------------- analysis + plots for one merged frame ----------------

def analyze_and_plot(daily):
    # daily has: symbol, date, rho_micro, rho_obi, vol, spread ; add a datetime for the time axis
    daily = daily.copy()
    # parse date
    daily["dt"] = pd.to_datetime(daily["date"])
    # per ticker
    for sym in sorted(daily["symbol"].unique()):
        # this symbol's series, time-ordered
        g = daily[daily.symbol == sym].sort_values("dt").reset_index(drop=True)

        # --- correlations (Spearman): effectiveness vs vol and vs spread, micro + obi ---
        def sp(a, b):
            # guard against all-nan
            return g[a].corr(g[b], method="spearman")
        # micro_dev correlations
        cmv, cms = sp("rho_micro", "vol"), sp("rho_micro", "spread")
        # obi (control) correlations
        cov, cos = sp("rho_obi", "vol"), sp("rho_obi", "spread")

        # --- tercile table: mean effectiveness in low/mid/high vol and spread ---
        print(f"\n[{sym}] Spearman(effectiveness, driver):")
        print(f"   micro_dev: vs vol = {cmv:+.2f}   vs spread = {cms:+.2f}")
        print(f"   obi_1     : vs vol = {cov:+.2f}   vs spread = {cos:+.2f}   (control)")
        # build terciles for vol and spread
        for driver in ["vol", "spread"]:
            # qcut into 3 equal-count bins
            g[f"{driver}_t"] = pd.qcut(g[driver], 3, labels=["low", "mid", "high"])
            # mean micro/obi rho per tercile
            tab = g.groupby(f"{driver}_t", observed=True)[["rho_micro", "rho_obi"]].mean()
            # print the tercile means
            print(f"   mean rho by {driver} tercile:")
            for lvl, r in tab.iterrows():
                print(f"      {driver}={lvl:4s}  micro={r['rho_micro']:+.2f}  obi={r['rho_obi']:+.2f}")

        # --- figure: 2x2 (effectiveness ts, driver ts, scatter vs vol, scatter vs spread) ---
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        # (0,0) rolling effectiveness: micro vs obi
        gi = g.set_index("dt")
        ax[0, 0].plot(gi.index, gi["rho_micro"].rolling(ROLL, min_periods=MIN_PERIODS).median(),
                      color="tab:orange", linewidth=2, label="micro_dev")
        ax[0, 0].plot(gi.index, gi["rho_obi"].rolling(ROLL, min_periods=MIN_PERIODS).median(),
                      color="tab:blue", linewidth=2, label="obi_1 (control)")
        ax[0, 0].axhline(0, color="red", ls="--", lw=1)
        ax[0, 0].set_title(f"{sym}: rolling effectiveness (rho)")
        ax[0, 0].set_ylabel("rolling median rho"); ax[0, 0].legend(fontsize=8)

        # (0,1) rolling drivers: vol (left axis) and spread (right axis)
        axv = ax[0, 1]
        axv.plot(gi.index, gi["vol"].rolling(ROLL, min_periods=MIN_PERIODS).median(),
                 color="tab:green", linewidth=2, label="vol")
        axv.set_ylabel("median realized_vol_bps", color="tab:green")
        axs = axv.twinx()
        axs.plot(gi.index, gi["spread"].rolling(ROLL, min_periods=MIN_PERIODS).median(),
                 color="tab:purple", linewidth=2, label="spread")
        axs.set_ylabel("median spread_bps", color="tab:purple")
        axv.set_title(f"{sym}: daily volatility and spread")

        # (1,0) scatter: effectiveness vs vol (micro + obi control)
        ax[1, 0].scatter(g["vol"], g["rho_micro"], s=12, alpha=0.5, color="tab:orange", label="micro_dev")
        ax[1, 0].scatter(g["vol"], g["rho_obi"],   s=12, alpha=0.3, color="tab:blue",   label="obi_1")
        # overlay vol-tercile means for micro (big markers cut through the noise)
        tv = g.groupby("vol_t", observed=True).agg(vol=("vol", "median"), rm=("rho_micro", "mean")).reset_index()
        ax[1, 0].plot(tv["vol"], tv["rm"], "o-", color="darkorange", ms=10, label="micro tercile mean")
        ax[1, 0].axhline(0, color="red", ls="--", lw=1)
        ax[1, 0].set_xlabel("daily median vol (bps)"); ax[1, 0].set_ylabel("daily rho")
        ax[1, 0].set_title(f"effectiveness vs VOL   (micro {cmv:+.2f} | obi {cov:+.2f})")
        ax[1, 0].legend(fontsize=8)

        # (1,1) scatter: effectiveness vs spread (micro + obi control)
        ax[1, 1].scatter(g["spread"], g["rho_micro"], s=12, alpha=0.5, color="tab:orange", label="micro_dev")
        ax[1, 1].scatter(g["spread"], g["rho_obi"],   s=12, alpha=0.3, color="tab:blue",   label="obi_1")
        ts = g.groupby("spread_t", observed=True).agg(spread=("spread", "median"), rm=("rho_micro", "mean")).reset_index()
        ax[1, 1].plot(ts["spread"], ts["rm"], "o-", color="darkorange", ms=10, label="micro tercile mean")
        ax[1, 1].axhline(0, color="red", ls="--", lw=1)
        ax[1, 1].set_xlabel("daily median spread (bps)"); ax[1, 1].set_ylabel("daily rho")
        ax[1, 1].set_title(f"effectiveness vs SPREAD   (micro {cms:+.2f} | obi {cos:+.2f})")
        ax[1, 1].legend(fontsize=8)

        # save per ticker
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"micro_dev_vs_vol_spread_{sym}.png")
        plt.savefig(out, dpi=130); plt.close(fig)
        print(f"[{sym}] saved {out}")


# ---------------- main ----------------

def main():
    # daily effectiveness for both features
    rho = {}
    # load each summary and compute daily rho
    for feat, fname in SUMMARY_FILES.items():
        rho[feat] = daily_rho(load_parquet(os.path.join(SUMMARY_DIR, fname)), VALUE_COL)
    # merge micro and obi rho on (symbol, date)
    eff = rho["micro_dev_bps"].rename(columns={"rho": "rho_micro"}).merge(
        rho["obi_1"].rename(columns={"rho": "rho_obi"}), on=["symbol", "date"], how="outer")
    # daily vol/spread from the raw feature store (cached)
    vs = build_vol_spread()
    # normalize date dtype for the join
    vs["date"] = vs["date"].astype(str)
    eff["date"] = eff["date"].astype(str)
    # join effectiveness with drivers
    daily = eff.merge(vs, on=["symbol", "date"], how="inner").dropna(subset=["vol", "spread"])
    # analyze + plot per ticker
    analyze_and_plot(daily)


# entry point
if __name__ == "__main__":
    main()
