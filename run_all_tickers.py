"""
Per-symbol liquidity/spread screen across ALL tickers in one day's parquet.

Goal: find symbols with WIDE spreads AND real liquidity — specifically volume
that trades WHILE the spread is wide (co-incidence), not merely a wide
time-averaged spread that never overlaps with trading.

Run:  python run_all_tickers.py
Output: ticker_stats_2026-06-30.csv  (one row per symbol, ranked)

Memory: processes ONE symbol at a time via pyarrow row-group filtering, so it
never loads a full market table into RAM. Snapshots are the big table; we read
only the columns needed.
"""
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds

from ticker_stats_core import stats_for_symbol, ROUND_TRIP_BPS, WIDE_BPS

BASE = Path(r"G:\My Drive\HFT\Capital Stake Day\parsed\2026-06-30")
DATE = "2026-06-30"
F_SNAP  = BASE / f"{DATE}_ob_snapshot.parquet"
F_TRADE = BASE / f"{DATE}_trades.parquet"
OUT     = Path(f"ticker_stats_{DATE}.csv")

# only the columns each stat needs -> less I/O, less RAM
SNAP_COLS  = ["symbol", "msg_seq", "orig_time", "entry_type", "level", "px"]
TRADE_COLS = ["symbol", "transact_time", "initiator", "price", "qty"]


def list_symbols(dataset):
    """Distinct symbols without loading the whole column into one array:
    scan the symbol column in batches and union."""
    syms = set()
    scanner = dataset.scanner(columns=["symbol"])
    for batch in scanner.to_batches():
        syms.update(batch.column("symbol").to_pylist())
    return sorted(s for s in syms if s is not None)


def read_symbol(dataset, columns, symbol):
    tbl = dataset.to_table(columns=columns, filter=ds.field("symbol") == symbol)
    return tbl.to_pandas()


def main():
    snap_ds  = ds.dataset(F_SNAP,  format="parquet")
    trade_ds = ds.dataset(F_TRADE, format="parquet")

    # enumerate symbols from the TRADES file (only traded names are screenable)
    symbols = list_symbols(trade_ds)
    print(f"{len(symbols)} symbols with trades on {DATE}. Round-trip fee = {ROUND_TRIP_BPS:.1f} bps.")

    rows, errors = [], []
    for i, sym in enumerate(symbols, 1):
        try:
            t = read_symbol(trade_ds, TRADE_COLS, sym)
            if len(t) == 0:
                continue
            s = read_symbol(snap_ds, SNAP_COLS, sym)
            r = stats_for_symbol(s, t, sym)
            if r is not None:
                rows.append(r)
        except Exception as e:                     # never let one bad symbol kill the run
            errors.append((sym, repr(e)))
        if i % 25 == 0:
            print(f"  ...{i}/{len(symbols)} processed")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No symbols produced stats — check column names/paths.")
        return

    # --- rank by the DECISION metric: how much profit was physically available ---
    df = df.sort_values("ceiling_pkr", ascending=False).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT.resolve()}")
    if errors:
        print(f"{len(errors)} symbols errored (e.g. {errors[0]})")

    # --- console summary: the screen you actually asked for ---
    pd.set_option("display.width", 220, "display.max_columns", None)
    show = ["symbol", "trades", "notional_m", "median_spread_bps", "p99_spread_bps",
            "pct_time_wide", "pct_vol_qualifying", "qualifying_notional_m", "ceiling_pkr"]

    print(f"\n=== TOP 20 by profit ceiling at current fees ({ROUND_TRIP_BPS:.0f} bps round trip) ===")
    print(df[show].head(20).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # The specific thing you asked: WIDE spread AND liquid, co-incident
    screen = df[(df["p99_spread_bps"] > WIDE_BPS) & (df["notional_m"] > 50)]
    print(f"\n=== SCREEN: p99 spread > {WIDE_BPS:.0f} bps AND notional > 50M PKR "
          f"({len(screen)} symbols) ===")
    print(screen[show].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Sensitivity: how many symbols become viable if MM program cuts fees to ~1bp/side
    from ticker_stats_core import FEE_TOTAL_PCT
    print(f"\nSymbols with ANY qualifying volume at current fees: "
          f"{(df['ceiling_pkr'] > 0).sum()} / {len(df)}")
    print("(Re-run with reduced FEE_* in ticker_stats_core.py to test the MM-program scenario.)")


if __name__ == "__main__":
    main()
