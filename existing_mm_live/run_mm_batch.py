# run_mm_batch.py -- run naive OR micro MM (whichever run_legacy_mm.USE_MICRO
# selects), over a ticker list x all dates, with TREC fees and seeded latency.
# Run as a FRESH process:  python run_mm_batch.py
# Output is timestamped and strategy-tagged, so runs NEVER overwrite each other
# and you never have to rename anything between runs.

# The frozen engine module -- imported first so the fee guard runs before work.
import mm_backtest
# run_legacy_mm owns the strategy toggle (USE_MICRO) and the per-symbol-day runner.
import run_legacy_mm
# The single-symbol-day runner, dataset opener, and date lister.
from run_legacy_mm import run_one, open_datasets, discover_dates

# Guard 1: the TREC fee toggle must be on (catches a stale/wrong module import).
assert mm_backtest.USE_TREC_FEE is True, "wrong/stale mm_backtest (USE_TREC_FEE not True)"
# Guard 2: the per-side fee must equal the TREC value (~0.78 bps), not retail.
assert abs(mm_backtest.FEE_TOTAL_PCT - 7.77e-05) < 1e-6, f"fee={mm_backtest.FEE_TOTAL_PCT}"

# DataFrame handling for the result table.
import pandas as pd
# Filesystem path helper.
from pathlib import Path
# Wall-clock timing for per-date / per-ticker stats.
import time
# Per-symbol timing accumulators.
from collections import defaultdict
# Timestamp for the output filename.
from datetime import datetime

# ------------------------------- CONFIG -------------------------------------
# The tickers to backtest. UBL stays for baseline reconciliation.
SYMBOLS = ["UBL", "PPL"]
# Which strategy ran -- READ from run_legacy_mm so there is ONE source of truth.
# Set USE_MICRO in run_legacy_mm.py; this runner labels its output to match.
STRATEGY_TAG = "micro" if run_legacy_mm.USE_MICRO else "naive"
# Output directory.
OUT_DIR = Path("mm_results")
# Timestamped, strategy-tagged filename -> no run ever overwrites another.
STAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
# Final output path, e.g. mm_results/naive_two_2026-08-10_143205.csv
OUT_FILE = OUT_DIR / f"{STRATEGY_TAG}_two_{STAMP}.csv"

# --------------------------------- run --------------------------------------
# Announce which strategy this run uses and where it will write -- unambiguous.
print(f"strategy = {STRATEGY_TAG.upper()}  (run_legacy_mm.USE_MICRO={run_legacy_mm.USE_MICRO})")
# Confirm the live fee.
print(f"fee = {mm_backtest.FEE_TOTAL_PCT} (TREC)")
# Confirm the output target up front.
print(f"output -> {OUT_FILE}")

# Every trading date in the store, ascending.
dates = discover_dates()
# Announce scope.
print(f"{len(SYMBOLS)} symbols x {len(dates)} dates\n")

# Accumulated result rows (one per symbol-day).
rows = []
# Per-symbol wall-clock totals.
sym_seconds = defaultdict(float)
# Per-symbol processed-file counts.
sym_files = defaultdict(int)

# Walk every date in order.
for di, date in enumerate(dates, 1):
    # Open the three parquet datasets for this date once (shared across symbols).
    dsets = open_datasets(date)
    # open_datasets returns None if a partition is missing -> skip the date.
    if dsets is None:
        # Note and continue.
        print(f"  [{di}/{len(dates)}] {date}  no datasets; skipped")
        continue
    # Start timing this date.
    date_t0 = time.perf_counter()
    # Run each ticker for this date.
    for sym in SYMBOLS:
        # Time each symbol-day separately.
        sym_t0 = time.perf_counter()
        # Isolate one symbol-day so a single failure cannot abort the batch.
        try:
            # The frozen runner: summary dict, or None if not runnable.
            r = run_one(date, sym, dsets)
            # Keep the row if runnable.
            if r is not None:
                # Accumulate.
                rows.append(r)
                # Count a processed file.
                sym_files[sym] += 1
        # Record, do not re-raise, any failure.
        except Exception as e:
            # Store the error inline so it appears in the CSV.
            rows.append({"date": date, "symbol": sym, "error": repr(e)})
        # Add this symbol-day's time to the symbol total.
        sym_seconds[sym] += time.perf_counter() - sym_t0
    # Elapsed wall-clock for the whole date.
    date_dt = time.perf_counter() - date_t0
    # Print progress for EVERY date with its processing time.
    print(f"  [{di}/{len(dates)}] {date}  processed in {date_dt:.2f}s")

# Assemble all rows.
out = pd.DataFrame(rows)
# Ensure the output directory exists.
OUT_DIR.mkdir(exist_ok=True)
# Write the timestamped, tagged result file.
out.to_csv(OUT_FILE, index=False)
# Confirm the write.
print(f"\nwrote {OUT_FILE} -- {len(out)} rows")

# --------------------------- per-ticker timing ------------------------------
# Header.
print("\n=== per-ticker timing ===")
# One line per ticker: total time, file count, average per file.
for sym in SYMBOLS:
    # Total wall-clock for this symbol.
    total = sym_seconds[sym]
    # Files (symbol-days) processed.
    n = sym_files[sym]
    # Average per file, guarded against zero.
    avg = (total / n) if n else float("nan")
    # Report.
    print(f"  {sym:8s} total {total:8.2f}s | files {n:4d} | avg {avg:.3f}s/file")

# ----------------------------- reconciliation -------------------------------
# UBL 2026-06-30 -- the reference symbol-day. Its value depends on the strategy,
# so this is a sanity/record line, not a fixed-number assertion.
ubl = out[(out.get("symbol") == "UBL") & (out.get("date") == "2026-06-30")] \
    if "symbol" in out.columns else out.iloc[0:0]
# Show it if present.
if len(ubl):
    # Header names the strategy so the recorded number is unambiguous.
    print(f"\nUBL 2026-06-30 [{STRATEGY_TAG}] -- record this as the {STRATEGY_TAG} baseline:")
    # The fields that define the baseline; resolved_frac should be ~0.186 either way.
    print(ubl[["net_pnl", "pos_at_close", "rest_oid_resolved_frac",
               "liquidation_clean"]].to_string(index=False))

# ------------------------------- summary ------------------------------------
# Full-period P&L per symbol.
if "net_pnl" in out.columns:
    # Header names the strategy.
    print(f"\nnet_pnl by symbol (full period, {STRATEGY_TAG}, TREC):")
    # Sum liquidated P&L per symbol.
    print(out.groupby("symbol")["net_pnl"].sum().round(0).to_string())
    # Liquidation-clean count.
    print("clean:", int((out.get("liquidation_clean") == True).sum()), "of", len(out))
