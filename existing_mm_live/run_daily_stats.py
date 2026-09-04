"""
Stage 1 -- Multi-day screen. Compute ticker_stats_core.stats_for_symbol for
EVERY symbol on EVERY date, land the result in a DuckDB table `daily_stats`
inside a single hft.duckdb file, and merge the per-date staging files into one
combined parquet PLUS one CSV (for Tableau, which cannot read parquet).

Reuses the validated stats function verbatim -- the ceiling metric, the fee grid,
the co-incidence as-of join and the pairing adjustment are NOT reimplemented here.

Usage (run in this order):
    python run_daily_stats.py --inspect                 # print source schema + validate
    python run_daily_stats.py --smoke --date 2026-06-30 # 3 symbols, one date
    python run_daily_stats.py --date 2026-06-30          # one full date
    python run_daily_stats.py --all --workers 3          # every date, parallel compute
    python run_daily_stats.py --load                     # build daily_stats in hft.duckdb
    python run_daily_stats.py --merge                    # combine staging -> parquet + CSV
"""
# Standard-library command-line argument parser.
import argparse
# Filename globbing, used to enumerate date partitions and staging files.
import glob
# OS helpers (path and file-size checks).
import os
# Object-oriented filesystem paths.
from pathlib import Path

# Numerical library (used by the imported stats function).
import numpy as np
# DataFrame library used to assemble result rows.
import pandas as pd

# The validated stats function and its fee grid -- imported, never reimplemented.
from ticker_stats_core import stats_for_symbol, FEE_RT_GRID

# ------------------------------- CONFIG -------------------------------------
# Root of the parsed store: the folder that CONTAINS trades/ ob_snapshot/ etc.
PARSED_ROOT = Path(
    # First half of the Google Drive path.
    "/Users/shazzak/Library/CloudStorage/"
    # Second half of the path to the parsed store.
    "GoogleDrive-shazzak@gmail.com/My Drive/Capital Stake - Parsed"
)
# The single embedded DuckDB database file the daily_stats table lives in.
DUCKDB_PATH = "hft.duckdb"
# Directory where each date's intermediate stats parquet is written.
STAGING = Path("./daily_stats_staging")
# Combined parquet produced by --merge.
MERGED_PARQUET = "daily_stats_ALL.parquet"
# Combined CSV twin of that parquet, for Tableau.
MERGED_CSV = "daily_stats_ALL.csv"

# Columns each source table must provide for stats_for_symbol.
# NOTE: 'aggressor_side' is REQUIRED (the pairing ratio uses it). Omitting it is
# the exact bug that made run_all_tickers error on every symbol -- do not drop it.
TRADE_COLS = ["symbol", "transact_time", "initiator", "price", "qty", "aggressor_side", "segment"]
# Snapshot columns stats_for_symbol needs (L1 spread series + session bounds).
SNAP_COLS = ["symbol", "msg_seq", "orig_time", "entry_type", "level", "px"]


# --------------------------- pure compute (testable) ------------------------
# Compute one symbol-day stats row from already-loaded frames; no I/O here.
def stats_one(snap_df, trades_df, sym, date):
    """Compute one symbol-day stats row (or None). Pure: no I/O, no DuckDB."""
    # Skip symbols with no book or no trades -- nothing to screen.
    if trades_df is None or snap_df is None or len(trades_df) == 0 or len(snap_df) == 0:
        # Signal "no row" to the caller.
        return None
    # Delegate the real computation to the validated function.
    row = stats_for_symbol(snap_df, trades_df, sym)
    # stats_for_symbol returns None when there is no continuous session / L1 series.
    if row is None:
        # Propagate the skip.
        return None
    # Key the row to its partition so daily_stats is keyed (symbol, date).
    row["date"] = date
    # Segment comes from the trades rows themselves (REG / STOCK_DEL_FUT / STOCK_CS_FUT).
    # A symbol-date should map to exactly one segment; keep the busiest and record
    # the count so an anomaly is visible rather than silently coalesced.
    if "segment" in trades_df.columns:
        segs = trades_df["segment"].dropna()
        row["segment"] = segs.mode().iloc[0] if len(segs) else "UNKNOWN"
        row["n_segments"] = int(segs.nunique())
    else:
        row["segment"] = "UNKNOWN"
        row["n_segments"] = 0
    # Hand the completed row back.
    return row


# ------------------------------- DuckDB I/O ---------------------------------
# Open a fresh in-memory DuckDB connection (used for reading parquet).
def _con():
    # Import here so the module still imports even if duckdb is absent until needed.
    import duckdb
    # Return a new connection; the caller closes it.
    return duckdb.connect()


# Build the glob that points at one table's single date partition.
def date_glob(table, date):
    # e.g. <root>/ob_snapshot/date=2026-06-30/*.parquet
    return str(PARSED_ROOT / table / f"date={date}" / "*.parquet")


# Fail loudly if a required column is missing, listing exactly what is absent.
def validate_schema(con, date):
    """Fail loudly if a required column is missing, listing the mismatch."""
    # Accumulate one message per offending table.
    problems = []
    # Check the trades table then the snapshot table.
    for tbl, req in (("trades", TRADE_COLS), ("ob_snapshot", SNAP_COLS)):
        # Ask DuckDB for the column names without reading any rows.
        cols = con.execute(
            # DESCRIBE of a zero-row SELECT returns just the schema.
            f"DESCRIBE SELECT * FROM read_parquet('{date_glob(tbl, date)}') LIMIT 0"
        ).df()["column_name"].tolist()
        # Which required columns are not present.
        missing = [c for c in req if c not in cols]
        # Record a problem line if anything is missing.
        if missing:
            # Include what we do have, to make name-mapping easy.
            problems.append(f"{tbl}: missing {missing}; has {cols}")
    # If any table was short a column, stop before wasting a run.
    if problems:
        # Raise with the full diff so names can be mapped, not guessed.
        raise KeyError("Schema mismatch -- send me these and I map the names:\n  "
                       + "\n  ".join(problems))


# Compute every symbol's stats for one date and write a staging parquet.
def compute_date(date, symbol_limit=None):
    """Compute every symbol's stats for one date -> staging parquet. Returns count."""
    # Open a connection for this date's reads.
    con = _con()
    # Abort early on any schema mismatch.
    validate_schema(con, date)

    # Comma-joined projection list for the trades read.
    tcols = ", ".join(TRADE_COLS)
    # Read the whole day's trades once (small table); pull only needed columns.
    t_all = con.execute(
        # Continuous order-book trades only. NDM is the Negotiated Deals Market:
        # bilaterally agreed blocks reported to the exchange, with no book behind
        # them -- averaging ~300x a REG trade, so one block can dominate a day's
        # volume-weighted ceiling with flow a market maker could never capture.
        # ODD_LOT is sub-round-lot flow, also not continuous-board matching.
        # NOTE: exclusion, NOT market='REG' -- STOCK_DEL_FUT and STOCK_CS_FUT are
        # also `market` values, so an equality filter would delete all futures.
        f"SELECT {tcols} FROM read_parquet('{date_glob('trades', date)}') "
        f"WHERE symbol IS NOT NULL AND market NOT IN ('NDM', 'ODD_LOT')"
    ).df()
    # If the day has no trades at all, there is nothing to screen.
    if len(t_all) == 0:
        # Tell the operator and return zero rows.
        print(f"  [{date}] no trades; skipped")
        # Nothing produced.
        return 0
    # Split the day's trades into one small frame per symbol (in memory).
    trades_by_sym = dict(tuple(t_all.groupby("symbol")))

    # Comma-joined projection list for the snapshot reads.
    scols = ", ".join(SNAP_COLS)
    # The symbol universe for the day, sorted for deterministic order.
    symbols = sorted(trades_by_sym.keys())
    # In --smoke mode, only process the first few symbols.
    if symbol_limit:
        # Truncate the symbol list.
        symbols = symbols[:symbol_limit]

    # Collected result rows and per-symbol errors.
    rows, errors = [], []
    # Walk every symbol for the day.
    for i, sym in enumerate(symbols, 1):
        # Isolate one symbol so a single bad name cannot kill the whole date.
        try:
            # Read just this symbol's snapshot rows (row-group pruning on symbol).
            s_sym = con.execute(
                # Project only the needed snapshot columns, filtered to the symbol.
                f"SELECT {scols} FROM read_parquet('{date_glob('ob_snapshot', date)}') "
                f"WHERE symbol = ?", [sym]
            ).df()
            # Compute the stats row from this symbol's snapshot + trades.
            r = stats_one(s_sym, trades_by_sym[sym], sym, date)
            # Keep the row if the symbol was screenable.
            if r is not None:
                # Accumulate it.
                rows.append(r)
        # Record, but do not re-raise, any per-symbol failure.
        except Exception as e:
            # Store the symbol and the error text for the summary line.
            errors.append((sym, repr(e)))
        # Progress heartbeat every 100 symbols.
        if i % 100 == 0:
            # Show how far into the day we are.
            print(f"  [{date}] {i}/{len(symbols)} symbols")

    # Make sure the staging directory exists.
    STAGING.mkdir(parents=True, exist_ok=True)
    # Destination staging file for this date.
    out = STAGING / f"daily_stats_{date}.parquet"
    # Only write if at least one symbol produced a row.
    if rows:
        # Assemble all rows into a single DataFrame.
        df = pd.DataFrame(rows)
        # Register the DataFrame with DuckDB so it can be written natively.
        con.register("df_v", df)
        # Write the staging parquet via DuckDB (no pyarrow dependency).
        con.execute(f"COPY df_v TO '{out.as_posix()}' (FORMAT PARQUET)")
        # Drop the temporary registration.
        con.unregister("df_v")
    # Build the summary line for this date.
    msg = f"  [{date}] {len(rows)} symbols -> {out}"
    # Append an error note if any symbol failed.
    if errors:
        # Show the count and one example.
        msg += f"  ({len(errors)} errored, e.g. {errors[0]})"
    # Emit the summary.
    print(msg)
    # Close the connection for this date.
    con.close()
    # Return how many rows were produced (for the driver).
    return len(rows)


# List every date partition present under trades/.
def discover_dates():
    # Find all date= folders.
    dirs = glob.glob(str(PARSED_ROOT / "trades" / "date=*"))
    # Return just the YYYY-MM-DD portion, sorted chronologically.
    return sorted(d.split("date=")[-1] for d in dirs)


# True if a date's staging file already exists and is non-empty (for resume).
def already_done(date):
    # The staging file path for this date.
    f = STAGING / f"daily_stats_{date}.parquet"
    # Exists and has content.
    return f.exists() and f.stat().st_size > 0


# Bulk-load all staging parquet into the daily_stats table in hft.duckdb.
def load_to_duckdb():
    """Bulk-load all staging parquet into daily_stats inside hft.duckdb (single writer)."""
    # DuckDB is only needed here.
    import duckdb
    # Glob covering every per-date staging file.
    files = str(STAGING / "daily_stats_*.parquet")
    # Nothing to load if no staging files exist yet.
    if not glob.glob(files):
        # Inform and stop.
        print("No staging files to load."); return
    # Open (or create) the persistent database file.
    con = duckdb.connect(DUCKDB_PATH)
    # Rebuild the table from all staging files in one SELECT.
    con.execute(f"""
        CREATE OR REPLACE TABLE daily_stats AS
        SELECT * FROM read_parquet('{files}', union_by_name=true)
    """)
    # Read back row and date counts as a sanity check.
    n, d = con.execute("SELECT count(*), count(DISTINCT date) FROM daily_stats").fetchone()
    # Report what was loaded.
    print(f"Loaded daily_stats: {n} rows across {d} dates -> {DUCKDB_PATH}")
    # Compute the current fee schedule's column tag (e.g. _35p45) for a convenience view.
    cur_tag = f"_{2*_fee_total()*1e4:.2f}".replace(".", "p")
    # Create a small ranking view keyed to the current fee schedule.
    con.execute(f"""
        CREATE OR REPLACE VIEW daily_rank AS
        SELECT date, symbol, notional_m, median_spread_bps, p99_spread_bps,
               ceiling_pkr, ceiling_paired_rt{cur_tag} AS ceiling_paired_cur
        FROM daily_stats
    """)
    # Done with the database connection.
    con.close()


# Merge every per-date staging parquet into ONE parquet and ONE CSV.
def merge_staging(out_parquet=MERGED_PARQUET, out_csv=MERGED_CSV):
    """Combine all daily_stats_<date>.parquet into a single parquet + a single CSV.

    The CSV is a content twin of the parquet, produced because Tableau cannot
    ingest parquet directly.
    """
    # DuckDB performs the read and both writes; no pyarrow dependency.
    import duckdb
    # Glob covering every per-date staging file.
    files = str(STAGING / "daily_stats_*.parquet")
    # Enumerate the matching files so we can report how many were merged.
    matched = sorted(glob.glob(files))
    # If nothing has been computed yet, there is nothing to merge.
    if not matched:
        # Inform and stop.
        print(f"No staging files to merge under {STAGING}/"); return
    # Open a throwaway in-memory connection.
    con = duckdb.connect()
    # One SELECT over all staging files; union_by_name guards column-order drift.
    src = f"SELECT * FROM read_parquet('{files}', union_by_name=true) ORDER BY date, symbol"
    # Write the combined parquet.
    con.execute(f"COPY ({src}) TO '{out_parquet}' (FORMAT PARQUET)")
    # Write the combined CSV with a header row for Tableau.
    con.execute(f"COPY ({src}) TO '{out_csv}' (FORMAT CSV, HEADER)")
    # Count rows and distinct dates in the merged output as a sanity check.
    n, d = con.execute(
        f"SELECT count(*), count(DISTINCT date) FROM read_parquet('{out_parquet}')"
    ).fetchone()
    # Report the inputs merged and both outputs written.
    print(f"Merged {len(matched)} staging files -> {n} rows across {d} dates")
    # Show the parquet path and its size in MB.
    print(f"  parquet: {out_parquet}  ({os.path.getsize(out_parquet)/1e6:.2f} MB)")
    # Show the CSV path and its size in MB.
    print(f"  csv    : {out_csv}  ({os.path.getsize(out_csv)/1e6:.2f} MB)")
    # Close the connection.
    con.close()


# Fetch the current total one-way fee fraction from the stats module.
def _fee_total():
    # Import lazily to avoid a hard dependency at module import time.
    from ticker_stats_core import FEE_TOTAL_PCT
    # Return it.
    return FEE_TOTAL_PCT


# --------------------------------- CLI --------------------------------------
# Print the source schema for the first date and validate it.
def inspect_one():
    # All available dates.
    dates = discover_dates()
    # Bail if the store has no partitions where expected.
    if not dates:
        # Tell the operator where we looked.
        print(f"No date= partitions under {PARSED_ROOT/'trades'}"); return
    # Use the first date; open a connection.
    d = dates[0]; con = _con()
    # Print columns for each source table.
    for tbl in ("trades", "ob_snapshot"):
        # Ask DuckDB for the schema without reading rows.
        cols = con.execute(
            # DESCRIBE of a zero-row SELECT.
            f"DESCRIBE SELECT * FROM read_parquet('{date_glob(tbl, d)}') LIMIT 0"
        ).df()["column_name"].tolist()
        # Show the table's columns.
        print(f"[{tbl}] columns: {cols}")
    # Run the required-column check and report pass/fail.
    try:
        # Raises on mismatch.
        validate_schema(con, d); print("Schema validation: PASS")
    # Surface a clean failure message instead of a traceback.
    except KeyError as e:
        # Show exactly what is missing.
        print("Schema validation: FAIL\n", e)
    # Close the connection.
    con.close()

# Classify WHY a symbol-day is unscreenable, so skips are recorded rather than
# silently vanishing from daily_stats. Mirrors stats_for_symbol's guards without
# touching that validated function.
def skip_reason(snap_df, trades_df):
    if trades_df is None or len(trades_df) == 0:
        return "no_trades"
    if snap_df is None or len(snap_df) == 0:
        return "no_snapshot"
    sb = snap_df[snap_df["entry_type"].isin(["BID", "OFFER"]) & (snap_df["level"] == 1)]
    if len(sb) == 0:
        return "no_l1_book"
    has_bid = bool((sb["entry_type"] == "BID").any())
    has_ask = bool((sb["entry_type"] == "OFFER").any())
    if not (has_bid and has_ask):
        # DMC 2025-09-23: locked limit-up, bid at the cap, zero offers all session.
        return "one_sided_bid_only" if has_bid else "one_sided_ask_only"
    if int((trades_df["initiator"] != "AUCTION").sum()) == 0:
        return "auction_only"
    return "other"

# Command-line entry point.
def main():
    # Build the argument parser.
    ap = argparse.ArgumentParser()
    # Print schema + validate and exit.
    ap.add_argument("--inspect", action="store_true")
    # Process only the first 3 symbols of one date.
    ap.add_argument("--smoke", action="store_true")
    # Which single date to process (also used by --smoke).
    ap.add_argument("--date", type=str, default=None)
    # Process every date.
    ap.add_argument("--all", action="store_true")
    # Load staging files into the DuckDB table.
    ap.add_argument("--load", action="store_true")
    # Merge staging files into one parquet + one CSV.
    ap.add_argument("--merge", action="store_true")
    # Number of parallel worker processes for --all.
    ap.add_argument("--workers", type=int, default=1)
    # Parse the command line.
    a = ap.parse_args()

    # --inspect: schema only.
    if a.inspect:
        # Run and return.
        inspect_one(); return
    # --smoke: first 3 symbols of one date.
    if a.smoke:
        # Default to the earliest date if none given.
        d = a.date or discover_dates()[0]
        # Compute just a few symbols.
        compute_date(d, symbol_limit=3)
        # Point the operator at the output.
        print(f"Smoke done. Inspect {STAGING/f'daily_stats_{d}.parquet'}"); return
    # --merge: combine staging into one parquet + CSV (checked before --date).
    if a.merge:
        # Merge and return.
        merge_staging(); return
    # --date without --all: one full date, then load it.
    if a.date and not a.all:
        # Compute the date, then refresh the DuckDB table.
        compute_date(a.date); load_to_duckdb(); return
    # --load: (re)build the table from whatever staging exists.
    if a.load:
        # Load and return.
        load_to_duckdb(); return
    # --all: every not-yet-done date, optionally in parallel.
    if a.all:
        # Skip dates already computed (resumable).
        dates = [d for d in discover_dates() if not already_done(d)]
        # Announce the plan.
        print(f"{len(dates)} dates to compute (resumable), workers={a.workers}")
        # Single-process path.
        if a.workers <= 1:
            # Compute each date in turn.
            for d in dates:
                # One date.
                compute_date(d)
        # Parallel path.
        else:
            # Process pool primitives.
            from concurrent.futures import ProcessPoolExecutor, as_completed
            # Spin up the pool.
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                # Submit one task per date.
                futs = {ex.submit(compute_date, d): d for d in dates}
                # Drain results (re-raising any worker exception).
                for f in as_completed(futs):
                    # Surface worker errors.
                    f.result()
        # After all dates are computed, build the table.
        load_to_duckdb()
        # Then produce the combined parquet + CSV as well.
        merge_staging(); return
    # No recognised flag: show help.
    ap.print_help()


# Only run main() when executed as a script, not when imported.
if __name__ == "__main__":
    # Dispatch.
    main()