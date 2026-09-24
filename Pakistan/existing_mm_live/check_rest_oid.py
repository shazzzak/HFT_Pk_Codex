# ============================================================================
# check_rest_oid.py -- HOW MANY TRADES ACTUALLY CARRY A RESTING ORDER ID?
# ----------------------------------------------------------------------------
# WHY: PSX_Parser_Mac.py resolves each trade's resting order (_build_trades,
# line 637) BEFORE loading that chunk's new orders into the lookup table
# (_build_ob_updates, line 715). So any trade whose resting order was added in
# the SAME 250,000-line chunk gets a blank id. The lookup table persists across
# chunks, so orders from earlier chunks are fine -- which means the damage is
# real but bounded, and this script measures exactly how bounded.
#
# HOW TO READ THE RESULT:
#   95%+        the bug is cosmetic; only same-chunk trades lose their id
#   70-90%      real -- worth fixing and re-parsing, but the P&L stands
#   below 50%   something else is also wrong; investigate before trusting it
#
# A date near 0% is probably one of the known feed gaps (1 Oct, 21 Jan), not
# this bug. Watch for that separately.
#
# RUN IT WITH THE SAME INTERPRETER AS YOUR BACKTEST (the .backtest venv), or
# duckdb will not be importable.
# ============================================================================

# standard library: filesystem paths
from pathlib import Path
# standard library: clean exit with a message
import sys

# ----------------------------------------------------------------- CONFIG --
# The parsed store, hardcoded (this is config_pk.PARSED_ROOT).
PARSED_ROOT = Path("/Users/shazzak/HFT Data/Pakistan/Capital Stake - Parsed")
# The trades table lives in date-partitioned folders: trades/date=YYYY-MM-DD/
TRADES_DIR = PARSED_ROOT / "trades"
# Set to a number to check only the first N dates (a quick smoke); None = all.
LIMIT_DATES = None

# duckdb is imported here so a missing install fails with a clear message
try:
    import duckdb
except ImportError:
    # tell him exactly what to do rather than dumping a traceback
    sys.exit("duckdb is not installed in this interpreter.\n"
             "Run this with your .backtest venv, or: pip install duckdb")


def main():
    # every date partition under trades/, sorted so the output reads in order
    date_dirs = sorted(d for d in TRADES_DIR.glob("date=*") if d.is_dir())
    # nothing found -> the path is wrong, say so plainly
    if not date_dirs:
        sys.exit(f"No date folders under {TRADES_DIR}\n"
                 f"Check the path at the top of this file.")
    # optionally cut the list down for a quick look
    if LIMIT_DATES:
        date_dirs = date_dirs[:LIMIT_DATES]
    # announce the scope before a long pass
    print(f"checking {len(date_dirs)} dates under {TRADES_DIR}\n")

    # one in-memory connection reused for every date
    con = duckdb.connect()
    # running totals across all dates
    tot_trades = 0
    tot_with_id = 0
    # collected per-date rows, so the worst dates can be listed at the end
    rows = []

    # the column header
    print(f"  {'date':<12} {'trades':>12} {'with id':>12} {'resolved':>9}")

    # walk every date partition
    for i, d in enumerate(date_dirs, 1):
        # the date string is the folder name after "date="
        date = d.name.split("=", 1)[1]
        # every parquet file in this date's folder
        glob = str(d / "*.parquet")
        # skip an empty partition rather than crashing on it
        if not list(d.glob("*.parquet")):
            print(f"  {date:<12} {'(no parquet files)':>35}")
            continue
        # count the rows and the rows with a usable id. NULL, empty string and
        # the pandas string forms of missing values all count as NOT resolved.
        q = f"""
            SELECT COUNT(*) AS trades,
                   SUM(CASE WHEN resting_order_id IS NOT NULL
                             AND CAST(resting_order_id AS VARCHAR) NOT IN ('', '<NA>', 'nan', 'None')
                            THEN 1 ELSE 0 END) AS with_id
            FROM read_parquet('{glob}')
        """
        # run it; a malformed partition should not kill the whole pass
        try:
            trades, with_id = con.execute(q).fetchone()
        except Exception as e:
            print(f"  {date:<12} ERROR: {e!r}")
            continue
        # guard against a date with no rows at all
        trades = int(trades or 0)
        with_id = int(with_id or 0)
        # the resolution rate for this date
        pct = (with_id / trades * 100.0) if trades else float("nan")
        # accumulate
        tot_trades += trades
        tot_with_id += with_id
        rows.append((date, trades, with_id, pct))
        # one line per date
        print(f"  {date:<12} {trades:>12,} {with_id:>12,} {pct:>8.2f}%")

    # nothing measurable
    if tot_trades == 0:
        sys.exit("\nNo trades counted. Check the path and the column name.")

    # the headline number
    overall = tot_with_id / tot_trades * 100.0
    print(f"\n  {'OVERALL':<12} {tot_trades:>12,} {tot_with_id:>12,} {overall:>8.2f}%")

    # the worst dates are where a feed gap would show, separate from the bug
    worst = sorted(rows, key=lambda r: r[3])[:10]
    print("\n  10 worst dates (a near-zero date is probably a feed gap, not this bug):")
    for date, trades, with_id, pct in worst:
        print(f"    {date}  {pct:6.2f}%   ({with_id:,} of {trades:,})")

    # the plain-language verdict, stated as a rule rather than a feeling
    print()
    if overall >= 95.0:
        print("  READ: the parser bug is COSMETIC. Fix it, but nothing downstream moves.")
    elif overall >= 70.0:
        print("  READ: REAL but bounded. Worth fixing and re-parsing. The P&L conclusions")
        print("        still stand -- queue position was measured at t = -0.03, so a gap")
        print("        in queue tracking cannot move the result much.")
    elif overall >= 50.0:
        print("  READ: MATERIAL. Fix the parser, re-parse everything, and re-check any")
        print("        result that depends on queue position before trusting it.")
    else:
        print("  READ: something beyond the chunk-ordering bug is wrong. Do not assume")
        print("        the swap fixes this -- investigate the column itself first.")


# run it when executed directly
if __name__ == "__main__":
    main()
