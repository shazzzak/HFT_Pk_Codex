# run_naive_two.py -- naive MM, TREC fees, over a small ticker list x all dates.
# Run as a FRESH process:  python run_naive_two.py
# (never from a stale Jupyter kernel -- the fee/import can be cached and lie).

# The frozen engine module. Imported first so the assert-guard below can verify
# the TREC fee is live before any backtest runs.
import mm_backtest
# Guard 1: the TREC fee toggle must be on. Fails loudly if a stale/wrong module loaded.
assert mm_backtest.USE_TREC_FEE is True, "wrong/stale mm_backtest (USE_TREC_FEE not True)"
# Guard 2: the per-side fee must equal the TREC value (~0.78 bps = 7.77e-05), not retail.
assert abs(mm_backtest.FEE_TOTAL_PCT - 7.77e-05) < 1e-6, f"fee={mm_backtest.FEE_TOTAL_PCT}"
# Confirm to stdout what fee this run actually used.
print("fee confirmed TREC:", mm_backtest.FEE_TOTAL_PCT)

# DataFrame handling for the result table.
import pandas as pd
# Filesystem path helper for the output directory.
from pathlib import Path
# Wall-clock timing for the per-date / per-ticker stats.
import time
# Group timings by symbol at the end.
from collections import defaultdict
# The frozen single-symbol-day runner, the dataset opener, and the date lister.
from run_legacy_mm import run_one, open_datasets, discover_dates
# Timestamp for the output filename.
from datetime import datetime


# ------------------------------- CONFIG -------------------------------------
# The tickers to backtest. Keep UBL here: its 2026-06-30 result is the verified
# baseline (net_pnl -851, pos 404, resolved 0.186) used to prove the batch is faithful.
SYMBOLS = ["UBL", "PPL"]
# One-way latency in ms. Hoisted here as a single visible constant rather than
# buried in run_legacy_mm's CFG. NOTE: this is the flat 120ms reference value;
# it is PESSIMISTIC vs your ~45ms TREC target, so results here are a conservative
# floor. Held constant across naive/micro so the attribution stays clean.
LATENCY_MS = 120
# Timestamped, strategy-tagged filename -> no run ever overwrites another.
STAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
# Where the result CSV is written.
OUT_DIR = Path("mm_results")
# The result filename.
OUT_FILE = OUT_DIR / f"naive_two_{STAMP}.csv"

# --------------------------------- run --------------------------------------
# Every trading date in the store, ascending.
dates = discover_dates()
# Announce the scope of the run.
print(f"{len(SYMBOLS)} symbols x {len(dates)} dates")

# Accumulated result rows (one per symbol-day).
rows = []
# Per-symbol wall-clock totals, for the end-of-ticker stats.
sym_seconds = defaultdict(float)
# Per-symbol processed-file counts (symbol-days that produced a row).
sym_files = defaultdict(int)

# Walk every date in order.
for di, date in enumerate(dates, 1):
    # Open the three parquet datasets for this date once (shared across symbols).
    dsets = open_datasets(date)
    # open_datasets returns None if a required partition is missing -> skip the date.
    if dsets is None:
        # Note the skip and move on.
        print(f"  [{date}] no datasets; skipped")
        continue
    # Mark the start of this date's processing for timing.
    date_t0 = time.perf_counter()
    # Run each ticker for this date.
    for sym in SYMBOLS:
        # Time each symbol-day separately so per-ticker totals are exact.
        sym_t0 = time.perf_counter()
        # Isolate one symbol-day so a single failure cannot abort the batch.
        try:
            # The frozen runner: returns a summary dict, or None if not runnable.
            r = run_one(date, sym, dsets)
            # Keep the row only if the symbol-day was runnable.
            if r is not None:
                # Accumulate the result.
                rows.append(r)
                # Count this as a processed file for the symbol.
                sym_files[sym] += 1
        # Record, but do not re-raise, any per-symbol failure.
        except Exception as e:
            # Store the error inline so it shows up in the output CSV.
            rows.append({"date": date, "symbol": sym, "error": repr(e)})
        # Add this symbol-day's elapsed time to the symbol's running total.
        sym_seconds[sym] += time.perf_counter() - sym_t0
    # Elapsed wall-clock for the whole date (all symbols).
    date_dt = time.perf_counter() - date_t0
    # Print progress for EVERY date with the time it took (as requested).
    print(f"  [{di}/{len(dates)}] {date}  processed in {date_dt:.2f}s")

# Assemble all rows into one DataFrame.
out = pd.DataFrame(rows)
# Ensure the output directory exists.
OUT_DIR.mkdir(exist_ok=True)
# Write the full result table.
out.to_csv(OUT_FILE, index=False)
# Confirm the write.
print(f"\nwrote {OUT_FILE} -- {len(out)} rows")

# --------------------------- per-ticker timing ------------------------------
# Header for the timing block.
print("\n=== per-ticker timing ===")
# Report each symbol's total time, file count, and average per file.
for sym in SYMBOLS:
    # Total wall-clock spent on this symbol across all dates.
    total = sym_seconds[sym]
    # Number of symbol-days that produced a row.
    n = sym_files[sym]
    # Average time per processed file, guarding against divide-by-zero.
    avg = (total / n) if n else float("nan")
    # One line per ticker: total time, file count, average per file.
    print(f"  {sym:8s} total {total:7.2f}s | files {n:4d} | avg {avg:.3f}s/file")

# ----------------------------- reconciliation -------------------------------
# The gate: UBL 2026-06-30 inside this batch MUST reproduce the verified baseline.
ubl = out[(out.get("symbol") == "UBL") & (out.get("date") == "2026-06-30")] \
    if "symbol" in out.columns else out.iloc[0:0]
# Show it if present.
if len(ubl):
    # Header.
    print("\nUBL 2026-06-30 (seeded LatencyModel baseline -- record this net_pnl):")
    # The four fields that define the baseline.
    print(ubl[["net_pnl", "pos_at_close", "rest_oid_resolved_frac",
               "liquidation_clean"]].to_string(index=False))

# ------------------------------- summary ------------------------------------
# Full-period P&L per symbol, if the run produced P&L.
if "net_pnl" in out.columns:
    # Header.
    print("\nnet_pnl by symbol (full period, naive, TREC):")
    # Sum liquidated P&L per symbol.
    print(out.groupby("symbol")["net_pnl"].sum().round(0).to_string())
    # Liquidation-clean count, so haircut-affected rows are visible.
    print("clean:", int((out.get("liquidation_clean") == True).sum()), "of", len(out))