#!/usr/bin/env python3
# per_ticker_stats.py -- per-ticker risk/return from the fullyear PERNAME parquet.
# Two views: (1) per ticker OVERALL (buckets collapsed), (2) per ticker x bucket.
# Metrics per ticker: net_pkr, net_bps (notional-weighted), Sharpe, Sortino, maxDD,
# win%, plus capture/markout/fee/liq/fills. Sharpe/Sortino/maxDD/win% are computed
# on the per-ticker DAILY series (day-as-unit), matching the run's own scoring.

# frames + numerics
import pandas as pd
import numpy as np

# ---- EDIT: parquet path + which config to profile ----
# path to the PERNAME parquet from the full-year run
PERNAME = "/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/fullyear_confirm_PERNAME_20260911_1456.parquet"
# which throttle config to profile ("QBPS_2","QT_2t","QT_1t","OBI",... or None = all)
CONFIG = "QBPS_2"
# annualization factor for Sharpe/Sortino (trading days/yr)
ANN = np.sqrt(252.0)


def _risk(daily_pkr, daily_bps):
    # risk metrics from a per-ticker DAILY series.
    # daily_pkr: daily net PKR (one per date); daily_bps: daily net bps (NaN where no notional)
    # number of active days
    n = len(daily_pkr)
    # output dict
    out = {}
    # win rate = fraction of days with positive PKR
    out["win_pct"] = float(np.mean(daily_pkr > 0) * 100) if n else np.nan
    # finite daily-bps observations only (drop no-notional days)
    b = daily_bps[np.isfinite(daily_bps)]
    # need >=3 days and non-zero dispersion for a ratio
    if len(b) >= 3 and b.std(ddof=1) > 0:
        # annualized Sharpe on the daily bps series
        out["sharpe"] = float(b.mean() / b.std(ddof=1) * ANN)
        # target-0 downside semideviation: sqrt(mean(min(r,0)^2)) over ALL obs
        downside = np.minimum(b, 0.0)
        dd_dev = float(np.sqrt(np.mean(downside ** 2)))
        # annualized Sortino (inf if the ticker never had a down day)
        out["sortino"] = float(b.mean() / dd_dev * ANN) if dd_dev > 0 else np.inf
    else:
        # too few days -> undefined
        out["sharpe"] = np.nan
        out["sortino"] = np.nan
    # max drawdown on the cumulative PKR path (peak-to-trough, negative PKR)
    cum = np.cumsum(daily_pkr)
    # running peak
    peak = np.maximum.accumulate(cum)
    # drawdown series
    ddown = cum - peak
    # worst drawdown (most negative)
    out["maxDD_pkr"] = float(ddown.min()) if n else np.nan
    # return the metric bundle
    return out


def per_ticker(df, config, by_bucket=False):
    # per-ticker table for one config. by_bucket=True -> one row per (symbol,bucket).
    # filter to the requested config
    d = df[df["throttle"] == config].copy()
    # choose grouping keys
    grouped = d.groupby(["symbol", "bucket"]) if by_bucket else d.groupby("symbol")
    # accumulate one record per group
    rows = []
    # iterate groups
    for k, g in grouped:
        # collapse to a per-DATE daily series (sum across whatever isn't a key)
        daily = (g.groupby("date")
                   .agg(pkr=("net_pkr", "sum"), opn=("opened_notional", "sum"),
                        cap=("capture_pkr", "sum"), mko=("markout_pkr", "sum"),
                        fee=("fee_pkr", "sum"), liq=("liq_pkr", "sum"),
                        fills=("fills", "sum"))
                   .sort_index())
        # daily PKR array
        dp = daily["pkr"].to_numpy(dtype=float)
        # daily opened notional array
        do = daily["opn"].to_numpy(dtype=float)
        # daily net bps = pkr/notional*1e4, NaN where no notional that day
        dbps = np.where(do > 0, dp / np.where(do > 0, do, np.nan) * 1e4, np.nan)
        # risk metrics on the daily series
        r = _risk(dp, dbps)
        # start the record with the key(s)
        rec = {}
        # unpack the grouping key
        if by_bucket:
            rec["symbol"], rec["bucket"] = k
        else:
            rec["symbol"] = k
        # active-day count
        rec["n_days"] = int(len(daily))
        # total net PKR
        rec["net_pkr"] = float(daily["pkr"].sum())
        # notional-weighted net bps (the correct aggregate for a ratio)
        rec["net_bps"] = float(daily["pkr"].sum() / daily["opn"].sum() * 1e4) if daily["opn"].sum() > 0 else np.nan
        # simple mean of daily bps (context for the Sharpe numerator)
        rec["mean_daily_bps"] = float(np.nanmean(dbps)) if np.isfinite(dbps).any() else np.nan
        # merge in Sharpe/Sortino/maxDD/win%
        rec.update(r)
        # decomposition totals (capture is gross, markout/fee/liq are the drags)
        rec["capture_pkr"] = float(daily["cap"].sum())
        rec["markout_pkr"] = float(daily["mko"].sum())
        rec["fee_pkr"] = float(daily["fee"].sum())
        rec["liq_pkr"] = float(daily["liq"].sum())
        # total fills
        rec["fills"] = int(daily["fills"].sum())
        # keep it
        rows.append(rec)
    # assemble + sort by total money (most profitable ticker first)
    out = pd.DataFrame(rows).sort_values("net_pkr", ascending=False).reset_index(drop=True)
    # return the table
    return out


def main():
    # load the per-name parquet
    df = pd.read_parquet(PERNAME)
    # which configs to profile
    configs = [CONFIG] if CONFIG else sorted(df["throttle"].unique())
    # column order for printing
    cols = ["symbol", "n_days", "net_pkr", "net_bps", "sharpe", "sortino",
            "maxDD_pkr", "win_pct", "capture_pkr", "markout_pkr", "fee_pkr",
            "liq_pkr", "fills"]
    # each requested config
    for cfg in configs:
        # ---- VIEW 1: per ticker OVERALL (all buckets collapsed) ----
        overall = per_ticker(df, cfg, by_bucket=False)
        # banner
        print(f"\n{'='*90}\n{cfg} -- PER TICKER, OVERALL (all buckets)\n{'='*90}")
        # print with 2-dp money/ratios
        with pd.option_context("display.float_format", lambda x: f"{x:,.2f}"):
            print(overall[cols].to_string(index=False))
        # portfolio-level check line (sum of tickers should match the run's config total)
        print(f"  [check] sum net_pkr = {overall['net_pkr'].sum():,.0f}  "
              f"(should match the run's {cfg} net_PKR)")
        # save
        overall.to_parquet(f"per_ticker_OVERALL_{cfg}.parquet", index=False)

        # ---- VIEW 2: per ticker x bucket ----
        bybuck = per_ticker(df, cfg, by_bucket=True)
        # banner
        print(f"\n{'-'*90}\n{cfg} -- PER TICKER x BUCKET\n{'-'*90}")
        # bucket order for readability
        border = {"first15": 0, "middle": 1, "preclose45": 2, "last15": 3}
        # sort by symbol then session order
        bybuck["_o"] = bybuck["bucket"].map(border).fillna(9)
        bybuck = bybuck.sort_values(["symbol", "_o"]).drop(columns="_o")
        # print
        with pd.option_context("display.float_format", lambda x: f"{x:,.2f}"):
            print(bybuck[["symbol", "bucket"] + cols[1:]].to_string(index=False))
        # save
        bybuck.to_parquet(f"per_ticker_BYBUCKET_{cfg}.parquet", index=False)
        # tell where files went
        print(f"\nwrote per_ticker_OVERALL_{cfg}.parquet and per_ticker_BYBUCKET_{cfg}.parquet")


if __name__ == "__main__":
    main()
