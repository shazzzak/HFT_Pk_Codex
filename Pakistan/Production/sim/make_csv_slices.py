# ============================================================================
# make_csv_slices.py -- convert the bits of the store I need into readable CSV
# ============================================================================
# WHY THIS EXISTS.
#
# The data you copied into Production/Delete is reachable from my side, but
# neither my environment nor the one I can run commands in has a parquet
# engine, and both are blocked from installing one. Your .backtest venv has
# pyarrow. So this runs once under your venv and writes gzipped CSV, which
# needs no engine at all, into the same folder.
#
# After this I can check every assumption about the data myself -- dtypes,
# timezones, timestamp resolution, which tags are populated, what the session
# actually looks like on each of the four day types -- and validate changes
# to session_calendar.py before sending them to you. That is the loop that
# has been running through you for the last six rounds.
#
# WHAT IT WRITES, into Production/Delete/_csv/:
#
#   misc_h_<date>.csv.gz   every 35=h TradingSessionStatus message: the
#                          arrival timestamp and the whole raw FIX body.
#                          ~190k rows a date, ~4 MB gzipped.
#
#   trades_span.csv        ONE ROW PER DATE, not per trade: how many regular
#                          market prints there were and the 5th/50th/95th
#                          percentile and min/max of their arrival times.
#                          That is everything pick_board needs, and it avoids
#                          moving 400,000 rows a date for no reason.
#
#   schema.txt             the arrow schema of every table, which is the
#                          authority on timestamp resolution and timezone.
#
# THE DEFAULT FOUR DATES cover all four day types:
#   2026-03-11  Ramadan, regular day
#   2026-03-13  Ramadan, Friday
#   2026-03-25  ordinary, regular day
#   2026-03-27  ordinary, Friday
#
# READ-ONLY ON THE SOURCE. Writes only into <root>/_csv/.
#
# Run from Production/ under the venv that has pyarrow:
#   caffeinate -is python sim/make_csv_slices.py
#   caffeinate -is python sim/make_csv_slices.py --dates 2026-03-11,2026-04-01
#   caffeinate -is python sim/make_csv_slices.py --all-dates
# ============================================================================

# command-line flags
import argparse
# path handling
from pathlib import Path
# the date comes off the directory name
import re

# frames
import pandas as pd

# WHERE THE COPIED DATA LIVES. Deliberately NOT config_pk's PARSED_ROOT: this
# reads the copy you made, so the real store is not touched at all.
DEFAULT_ROOT = ("/Users/shazzak/PycharmProjects/HFT/Pakistan/Production/"
                "Delete")

# the four dates that cover all four day types
DEFAULT_DATES = ("2026-03-11", "2026-03-13", "2026-03-25", "2026-03-27")

# the tables that exist under the root
TABLES = ("trades", "ob_updates", "misc", "ob_snapshot")


def partition(root, table, date):
    """Where one table's partition for one date lives, hive layout."""
    # <root>/<table>/date=YYYY-MM-DD/YYYY-MM-DD_<table>.parquet
    return Path(root) / table / f"date={date}" / f"{date}_{table}.parquet"


def dates_under(root):
    """Every date present under the root, oldest first."""
    # the misc partitions are the calendar here
    dirs = sorted(Path(root).glob("misc/date=*"))
    # the date off each directory name
    rx = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
    # only the ones named as expected
    return sorted({m.group(1) for m in
                   (rx.match(d.name) for d in dirs) if m})


def write_schema(root, out_dir, date):
    """Record every table's arrow schema, which pandas dtypes do not show."""
    # the report
    lines = [f"ARROW SCHEMAS -- {date}", "=" * 70, ""]
    # metadata only; no data pages are read
    import pyarrow.parquet as pq
    # each table
    for table in TABLES:
        # its partition
        p = partition(root, table, date)
        # the header
        lines.append(f"--- {table}")
        # a missing partition is a fact too
        if not p.exists():
            lines.append("    PARTITION MISSING")
            lines.append("")
            continue
        # the file's metadata
        pf = pq.ParquetFile(p)
        # the row count
        lines.append(f"    rows: {pf.metadata.num_rows:,}")
        # one line per field, with the exact arrow type
        for f in pf.schema_arrow:
            lines.append(f"    {f.name:<22s} {f.type}")
        lines.append("")
    # written once
    (out_dir / "schema.txt").write_text("\n".join(lines) + "\n")
    # say so
    print(f"  wrote {out_dir / 'schema.txt'}")


def write_misc_h(root, out_dir, date):
    """Every 35=h status message for one date, as gzipped CSV."""
    # the partition
    p = partition(root, "misc", date)
    # nothing to convert
    if not p.exists():
        print(f"  {date}: no misc partition")
        return
    # two columns plus the type, which is all the calendar reads
    m = pd.read_parquet(p, columns=["capture_ts", "msg_type", "raw"])
    # status messages only
    h = m[m["msg_type"].astype("string") == "h"]
    # a date with none
    if len(h) == 0:
        print(f"  {date}: no 35=h messages")
        return
    # THE TIMESTAMP IS WRITTEN IN ISO 8601 WITH ITS OFFSET, so the timezone
    # survives the trip through CSV. A bare "2026-03-11 04:30:00" would come
    # back timezone-naive and reintroduce the exact bug this is meant to help
    # me stop making.
    out = pd.DataFrame({
        "capture_ts": pd.to_datetime(h["capture_ts"], utc=True)
                        .dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
        "raw": h["raw"],
    })
    # the file
    f = out_dir / f"misc_h_{date}.csv.gz"
    # gzipped, because the raw bodies compress about eight to one
    out.to_csv(f, index=False, compression="gzip")
    # say what came out
    print(f"  wrote {f}  ({len(out):,} rows, "
          f"{f.stat().st_size / 1e6:.1f} MB)")


def trades_span(root, date):
    """One date's regular-market trade span, as a dict."""
    # the partition
    p = partition(root, "trades", date)
    # nothing to measure
    if not p.exists():
        return None
    # two columns only
    t = pd.read_parquet(p, columns=["capture_ts", "market"])
    # the regular market
    reg = t[t["market"].astype("string") == "REG"]
    # fall back to every trade if the column is not populated
    src = reg if len(reg) else t
    # no trades at all
    if len(src) == 0:
        return None
    # arrival times, timezone-aware
    tt = pd.to_datetime(src["capture_ts"], utc=True)
    # the span, as ISO strings so nothing is lost in the CSV
    return {
        # the date
        "date": date,
        # how many regular-market prints
        "n_reg": int(len(reg)),
        # and how many trades in total, so the fallback is visible
        "n_all": int(len(t)),
        # the extremes
        "min_utc": tt.min().strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
        "max_utc": tt.max().strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
        # the percentiles pick_board uses
        "p05_utc": tt.quantile(0.05).strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
        "p50_utc": tt.quantile(0.50).strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
        "p95_utc": tt.quantile(0.95).strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
    }


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # where the copied data is
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="the folder holding the copied partitions")
    # which dates to convert
    ap.add_argument("--dates", default=None,
                    help="comma-separated YYYY-MM-DD; default is the four "
                         "that cover all four day types")
    # or all of them
    ap.add_argument("--all-dates", action="store_true",
                    help="convert every date under the root (larger)")
    args = ap.parse_args()

    print("=" * 70)
    print("CSV SLICES FOR OFFLINE CHECKING")
    print("=" * 70)
    print(f"  root : {args.root}")

    # every date present
    present = dates_under(args.root)
    # nothing there
    if not present:
        raise SystemExit(f"no misc/date=* partitions under {args.root}")

    # which to convert
    if args.all_dates:
        # everything
        want = present
    elif args.dates:
        # exactly what was asked for
        want = [d.strip() for d in args.dates.split(",") if d.strip()]
    else:
        # the four defaults, narrowed to what is actually present
        want = [d for d in DEFAULT_DATES if d in present]
        # fall back to the first four present if none of the defaults are
        if not want:
            want = present[:4]
    print(f"  dates: {', '.join(want)}")
    print(f"  ({len(present)} dates are present under the root)")

    # where the output goes
    out_dir = Path(args.root) / "_csv"
    # create it if needed
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out  : {out_dir}")
    print()

    # the schema, from the first date being converted
    write_schema(args.root, out_dir, want[0])

    # the status messages, per date
    for d in want:
        # say which, as it goes
        print(f"  {d} ...", flush=True)
        # the conversion
        write_misc_h(args.root, out_dir, d)

    # the trade spans, one row per date, for EVERY date present -- it is one
    # small read each and having the whole range makes the calendar checkable
    print("\n  trade spans ...", flush=True)
    # each date
    spans = [s for s in (trades_span(args.root, d) for d in present)
             if s is not None]
    # nothing measurable
    if spans:
        # the file
        f = out_dir / "trades_span.csv"
        # written once
        pd.DataFrame(spans).to_csv(f, index=False)
        # say so
        print(f"  wrote {f}  ({len(spans)} dates)")

    print()
    print("  Done. Nothing outside _csv/ was written.")


# entry point
if __name__ == "__main__":
    main()
