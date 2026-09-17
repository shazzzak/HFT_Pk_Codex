# ============================================================================
# dump_facts.py -- what the store ACTUALLY contains, written where I can read it
# ============================================================================
# WHY THIS EXISTS.
#
# I cannot read the parsed store. Every assumption I made about its columns,
# their dtypes and the shape of the FIX bodies inside them was therefore a
# guess that I shipped to you to test on my behalf, and session_calendar.py
# took six rounds because of it: pooled boards, a null column, a timestamp
# resolution, a union across boards, a dropped timezone, an unstable pick.
# Every one of those was a fact about the data that one look would have
# settled.
#
# This reads the store and writes ONE text file into Production/sim/, which
# I can already read. The store itself is never written to, and no new folder
# permission is needed.
#
# WHAT IT RECORDS, for one date:
#   * every table's row count, column names and EXACT dtypes -- including
#     timestamp resolution (ns / us / ms) and timezone, which is what broke
#     the per-second grid and then broke pick_board twice
#   * for each low-cardinality text column, its distinct values
#   * for each message type in misc, three whole raw FIX bodies
#   * the same for the raw bodies in ob_snapshot's text columns
#
# It prints nothing sensitive: column names, dtypes, a few sample values and
# a few protocol messages. No prices, no positions, no P&L.
#
# READ-ONLY ON THE STORE. Writes one file:
#   Production/sim/_facts/store_facts_<date>.txt
#
# Run from Production/:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/dump_facts.py
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/dump_facts.py --date 2026-06-30
# ============================================================================

# command-line flags
import argparse
# path handling
from pathlib import Path
# the date comes off the filename
import re

# frames
import pandas as pd

# the store, from the one config every runner uses
try:
    from config_pk import PARSED_ROOT
except Exception as _e:                                       # noqa: BLE001
    raise ImportError(
        "dump_facts: could not import PARSED_ROOT from config_pk (%r). Run "
        "with PYTHONPATH=../existing_mm_live." % _e)

# the four tables the parser writes
TABLES = ("trades", "ob_updates", "misc", "ob_snapshot")

# a column with at most this many distinct values gets all of them listed;
# above it, only the count and a few examples
LOW_CARD = 30

# how many rows to sample for the per-column value survey. The whole point is
# to be cheap: ob_snapshot is 33 million rows a date and nothing here needs
# more than a representative slice.
SAMPLE_ROWS = 200_000


def partition(table, date):
    """Where one table's partition for one date lives, hive layout."""
    # <root>/<table>/date=YYYY-MM-DD/YYYY-MM-DD_<table>.parquet
    return (Path(PARSED_ROOT) / table / f"date={date}"
            / f"{date}_{table}.parquet")


def newest_date():
    """The most recent trading date in the store."""
    # the trades partitions are the calendar
    files = sorted(Path(PARSED_ROOT).rglob("*_trades.parquet"))
    # nothing to go on
    if not files:
        raise SystemExit("no trades partitions found under the store")
    # the date prefix the parser writes on every filename
    rx = re.compile(r"^(\d{4}-\d{2}-\d{2})_trades\.parquet$")
    # every filename that matches
    got = [m.group(1) for m in (rx.match(f.name) for f in files) if m]
    # nothing named as expected
    if not got:
        raise SystemExit("no trades partition is named YYYY-MM-DD_trades.parquet")
    # the newest
    return sorted(got)[-1]


def describe(out, table, path):
    """Append one table's facts to the growing report."""
    # the header
    out.append("")
    out.append("=" * 74)
    out.append(f"TABLE: {table}")
    out.append("=" * 74)
    # a missing partition is a fact too
    if not path.exists():
        out.append("  PARTITION MISSING")
        return
    # row count and schema WITHOUT reading the data: parquet metadata carries
    # both, and ob_snapshot is far too large to read whole for this
    import pyarrow.parquet as pq
    # the file's metadata
    pf = pq.ParquetFile(path)
    # the total rows
    out.append(f"  rows        : {pf.metadata.num_rows:,}")
    out.append(f"  row groups  : {pf.metadata.num_row_groups:,}")
    out.append(f"  file size   : {path.stat().st_size / 1e6:,.1f} MB")
    # the arrow schema, which carries timestamp unit and timezone exactly
    out.append("")
    out.append("  ARROW SCHEMA (the authority on resolution and timezone)")
    # one line per field
    for f in pf.schema_arrow:
        out.append(f"    {f.name:<22s} {str(f.type)}")

    # a bounded slice for the value survey
    n = min(SAMPLE_ROWS, pf.metadata.num_rows)
    # read only that many rows, from the first row groups
    it = pf.iter_batches(batch_size=min(65_536, max(n, 1)))
    # collect until we have enough
    parts, got = [], 0
    # walk the batches
    for b in it:
        # this batch as a frame
        parts.append(b.to_pandas())
        # how many rows so far
        got += len(parts[-1])
        # enough
        if got >= n:
            break
    # nothing came back
    if not parts:
        out.append("\n  (no rows to sample)")
        return
    # the sample
    df = pd.concat(parts, ignore_index=True).head(n)
    out.append("")
    out.append(f"  PANDAS DTYPES AS READ ({len(df):,}-row sample)")
    # one line per column, because the pandas dtype is what the code sees
    for c in df.columns:
        out.append(f"    {c:<22s} {str(df[c].dtype)}")

    # the value survey
    out.append("")
    out.append("  COLUMN VALUES")
    # every column
    for c in df.columns:
        # the non-null values
        s = df[c].dropna()
        # a column that is entirely null is worth saying so about, loudly:
        # this is exactly what `segment` turned out to be on 35=h messages
        if len(s) == 0:
            out.append(f"    {c:<22s} ALL NULL in this sample")
            continue
        # how much of it is null
        nullpct = 100.0 * (len(df) - len(s)) / len(df)
        # the raw FIX body column gets its own treatment below
        if c == "raw":
            out.append(f"    {c:<22s} {nullpct:5.1f}% null  "
                       f"(bodies sampled separately below)")
            continue
        # how many distinct values
        nu = s.nunique()
        # a low-cardinality column: list every value, which is what identifies
        # a grouping key
        if nu <= LOW_CARD:
            # sorted for a stable report
            vals = ", ".join(str(v)[:24] for v in sorted(s.unique(),
                                                         key=str))
            out.append(f"    {c:<22s} {nullpct:5.1f}% null  "
                       f"{nu} distinct: {vals}")
        else:
            # otherwise the count and a few examples
            ex = ", ".join(str(v)[:24] for v in s.head(3))
            out.append(f"    {c:<22s} {nullpct:5.1f}% null  "
                       f"{nu:,} distinct, e.g. {ex}")

    # WHOLE RAW BODIES, per message type. This is the thing no summary
    # substitutes for, and the thing that settled the 35=h question in one
    # look after four rounds of guessing.
    if "raw" in df.columns and "msg_type" in df.columns:
        out.append("")
        out.append("  THREE WHOLE RAW BODIES PER MESSAGE TYPE")
        # each type present in the sample
        for mt, g in df.dropna(subset=["raw"]).groupby("msg_type"):
            # the header
            out.append(f"    --- 35={mt}  ({len(g):,} in sample)")
            # three, spread through the slice rather than three adjacent ones
            for i in (0, len(g) // 2, len(g) - 1):
                # truncated, because a snapshot body is enormous
                out.append(f"      {str(g['raw'].iloc[i])[:600]}")


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # which date to describe
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD; default is the newest in the store")
    # where the report goes
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to Production/sim/_facts")
    args = ap.parse_args()

    # the date
    date = args.date or newest_date()

    # the report, built in memory and written once
    out = []
    out.append("=" * 74)
    out.append(f"STORE FACTS -- {date}")
    out.append("=" * 74)
    out.append(f"  store : {PARSED_ROOT}")
    out.append("")
    out.append("  Written so that analysis code can be checked against what")
    out.append("  the store actually holds, rather than against an assumption")
    out.append("  about it. Column names, dtypes, sample values and protocol")
    out.append("  messages only -- no prices, positions or P&L.")

    # each table
    for table in TABLES:
        # say which, as it goes, because ob_snapshot takes a moment
        print(f"  reading {table} ...", flush=True)
        # its facts
        describe(out, table, partition(table, date))

    # where the file goes: beside this script, in a folder of its own
    outdir = Path(args.out) if args.out else Path(__file__).parent / "_facts"
    # create it if needed
    outdir.mkdir(parents=True, exist_ok=True)
    # the file, named for the date so several can coexist
    path = outdir / f"store_facts_{date}.txt"
    # written once
    path.write_text("\n".join(out) + "\n")
    # say where
    print(f"\nwrote {path}")
    print(f"  {len(out):,} lines. Nothing in the store was modified.")


# entry point
if __name__ == "__main__":
    main()
