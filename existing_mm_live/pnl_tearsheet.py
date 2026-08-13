# pnl_tearsheet.py -- QuantStats tearsheets from the per-day P&L CSV produced by
# daily_pnl_charts.py. Converts daily PKR P&L into daily RETURNS on a per-symbol
# capital-at-risk base (max_inv x median price), because QuantStats' Sharpe/Sortino/
# drawdown/CAGR math assumes fractional returns, not raw cash.
#
# WHAT'S TRUSTWORTHY vs the capital base:
#   * Sharpe, Sortino, win-rate, worst-day, best-day, volatility-of-returns,
#     tail ratio -- these are SCALE-INVARIANT (dividing all P&L by the same base
#     cancels in mean/std), so they're meaningful regardless of the exact base.
#   * CAGR, cumulative-return %, drawdown % -- these DEPEND on the base; they are
#     meaningful ONLY because we commit to max_inv x median price as the capital.
#
# Requires the price reference. We read median mid per symbol from the feature store
# so the base reflects actual price levels over the period.
#
# Run from existing_mm_live/ AFTER daily_pnl_charts.py has written daily_pnl.csv:
#   python pnl_tearsheet.py

# paths
from pathlib import Path
# frames + arrays
import pandas as pd
import numpy as np
# QuantStats for the tearsheet
import quantstats as qs
# driver (for MICRO_PARAMS max_inv + feature-store path)
import run_legacy_mm as R

# where daily_pnl_charts.py wrote its output
PNL_DIR = Path("/Users/shazzak/Capital Stake - Results/daily_pnl")
# the per-day CSV
CSV = PNL_DIR / "daily_pnl.csv"
# feature store (for median price per symbol -> capital base)
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
# output dir for the tearsheets
OUT_DIR = PNL_DIR / "tearsheets"
# ensure it exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# inventory cap (shares) from the strategy params -> capital-at-risk base
MAX_INV = R.MICRO_PARAMS["max_inv"]


# median mid price per symbol over the period (from the feature store), for the base
def median_price(sym, sample_days=20):
    # gather this symbol's feature-store day files
    files = sorted((FS_ROOT / sym).glob("date=*.parquet"))
    # nothing -> cannot price
    if not files:
        return np.nan
    # sample a spread of days (every Nth) to estimate the median cheaply
    step = max(1, len(files) // sample_days)
    # collect median mids
    mids = []
    # walk sampled files
    for f in files[::step]:
        # read just the mid column
        d = pd.read_parquet(f, columns=["mid"])
        # this day's median mid
        if len(d):
            mids.append(d["mid"].median())
    # overall median across sampled days
    return float(np.median(mids)) if mids else np.nan


# build a returns series (indexed by date) for one (strategy, symbol) cell
def returns_series(g, capital_base):
    # sort by date
    g = g.sort_values("date")
    # daily return = daily PKR P&L / capital-at-risk base
    r = pd.Series(g["pnl"].values / capital_base,
                  index=pd.to_datetime(g["date"].values))
    # QuantStats wants a clean DatetimeIndex; name it for the report title
    r.index.name = "date"
    return r


# main
def main():
    # load the per-day P&L (must exist -- produced by daily_pnl_charts.py)
    if not CSV.exists():
        raise SystemExit(f"missing {CSV} -- run daily_pnl_charts.py first")
    # read it
    df = pd.read_csv(CSV, parse_dates=["date"])
    # compute the per-symbol capital base = max_inv x median price
    bases = {}
    # for each symbol present
    for sym in df["symbol"].unique():
        # median price
        px = median_price(sym)
        # capital at risk = shares cap x price
        bases[sym] = MAX_INV * px
        # report the base so it's explicit, not hidden
        print(f"{sym}: median price {px:.2f} PKR x max_inv {MAX_INV} "
              f"= capital base {bases[sym]:,.0f} PKR", flush=True)
    # one tearsheet per (strategy, symbol)
    for (strat, sym), g in df.groupby(["strategy", "symbol"]):
        # the capital base for this symbol
        base = bases.get(sym, np.nan)
        # skip if we could not price
        if not np.isfinite(base) or base <= 0:
            print(f"  skip {strat} {sym}: no capital base", flush=True)
            continue
        # returns series on that base
        r = returns_series(g, base)
        # output filename
        out = OUT_DIR / f"tearsheet_{strat}_{sym}.html"
        # generate the full QuantStats HTML report
        # title carries the base so the reader knows what the %s are relative to
        try:
            qs.reports.html(
                r,
                output=str(out),
                title=f"{strat} {sym} (capital base {base:,.0f} PKR)")
            print(f"  tearsheet: {out}", flush=True)
        except Exception as e:
            # QuantStats can choke on degenerate series (all-zero, too few points)
            print(f"  {strat} {sym}: tearsheet failed ({e}); "
                  f"printing key scale-invariant stats instead", flush=True)
            # fall back to the metrics that don't need a working HTML render
            print(f"    Sharpe={qs.stats.sharpe(r):.2f}  "
                  f"Sortino={qs.stats.sortino(r):.2f}  "
                  f"win%={qs.stats.win_rate(r):.1%}  "
                  f"worst day={r.min():.4%}", flush=True)
    # done
    print(f"\ntearsheets in: {OUT_DIR}", flush=True)
    print("NOTE: Sharpe/Sortino/win-rate/worst-day are trustworthy; "
          "CAGR/drawdown-% are relative to the max_inv x price capital base above.",
          flush=True)


# entry point
if __name__ == "__main__":
    main()
