# ============================================================================
# dump_schema.py -- write schema_output.txt by READING THE PARQUET FILES
# ============================================================================
# WHY THIS EXISTS. The schema_output.txt in the project lists a `date` column
# on the trades table. A script that read a partition FILE and asked for that
# column failed on the first file with:
#
#   ArrowInvalid: No match for FieldRef.Name(date) in transact_time: ...
#
# CORRECTED 2026-09-16. THE OLD DUMP WAS NOT WRONG. The store is
# hive-partitioned:
#
#   <PARSED_ROOT>/trades/date=2026-06-30/2026-06-30_trades.parquet
#
# A dataset read of trades/** with hive partitioning on -- which is how
# run_legacy_mm opens the store -- synthesises `date` from the DIRECTORY NAME
# and presents it as an ordinary column. That view genuinely has 21 columns.
# A direct read of the FILE has 20, because the partition key lives in the
# path, not in the footer.
#
# So there are two correct schemas and they differ by exactly the partition
# keys. This dump reports the FILE schema, and names any partition keys it
# found in the path so the difference is stated rather than discovered by a
# crash six months from now.
#
# WHAT IT READS. The four tables the parser writes, per day:
#   {day}_trades.parquet       tick-level trades      (UA202 ExecType=F)
#   {day}_ob_updates.parquet   order adds and cancels (UA201, UA202 ExecType=4)
#   {day}_ob_snapshot.parquet  full book snapshots    (35=W)
#   {day}_misc.parquet         heartbeats, session status, news, everything else
#                              (the parser header calls it "other"; the store
#                               names it "misc" -- both are tried)
#
# It takes the MOST RECENT partition of each and names it in the output, so
# the dump says what it is a schema of. A schema with no date on it is how
# this problem started.
#
# COLUMN EXISTS != COLUMN POPULATED. --nulls reads the file and reports the
# fraction of nulls per column, which is the other half of the question and
# the reason a column can be present and still useless. It is off by default
# because it reads the data rather than the footer.
#
# READ-ONLY on the store. Writes ONE timestamped file; never overwrites the
# existing schema_output.txt.
#
# Run from existing_mm_live/:
#   caffeinate -is python dump_schema.py
#   caffeinate -is python dump_schema.py --nulls
# ============================================================================

# command-line flags
import argparse
# timestamped output name
import datetime as dt
# path handling
from pathlib import Path

# frames, for the --nulls pass and the printed tables
import pandas as pd

# the parquet reader. Reading the FOOTER only -- no data is loaded unless
# --nulls is passed.
try:
    # read_schema and ParquetFile are both long-standing, stable APIs
    import pyarrow.parquet as pq
# pandas reads parquet through pyarrow already, so this should never fire;
# say so plainly rather than failing on an AttributeError later
except Exception as _e:                                       # noqa: BLE001
    raise ImportError(
        "dump_schema: pyarrow is required to read a parquet schema (%r). "
        "It is already a dependency of the parser and of pandas' parquet "
        "reader, so if this fails the environment is not the one the store "
        "was written with." % _e)

# the project's single source of paths -- never hardcoded in a script
try:
    # PARSED_ROOT = the raw store; RESULTS_ROOT = where tools write
    from config_pk import PARSED_ROOT, RESULTS_ROOT
# no config_pk on the path -> stop with an explicit message
except Exception as _e:                                       # noqa: BLE001
    raise ImportError(
        "dump_schema: could not import PARSED_ROOT / RESULTS_ROOT from "
        "config_pk (%r). Run from the existing_mm_live/ dir, or add it to "
        "sys.path." % _e)

# THE FOUR TABLES, each as (label, description, filename suffixes to try),
# in the order the parser's own header lists them.
#
# The fourth is looked up under BOTH names on purpose. The parser's header
# comment calls it {day}_other.parquet; its internal table map keys the same
# schema as "misc". Only the store knows which name reached disk, so both are
# tried rather than one being assumed.
TABLES = [
    ("trades", "tick-level trades (UA202 ExecType=F)", ["trades"]),
    ("ob_updates", "order adds and cancels (UA201, UA202 ExecType=4)",
     ["ob_updates"]),
    ("ob_snapshot", "full order-book snapshots (35=W)", ["ob_snapshot"]),
    ("other/misc", "heartbeats, session status, news, everything else",
     ["other", "misc"]),
]


def newest_partition(root, suffixes):
    """The most recent {day}_{suffix}.parquet under the store, or None.

    Takes a LIST of suffixes because one table is written under two possible
    names; the first that matches anything wins.
    """
    # try each candidate name in order
    for suffix in suffixes:
        # every partition under that name, in filename order -- which is date
        # order, because the parser names them {day}_{suffix}.parquet
        files = sorted(Path(root).rglob(f"*_{suffix}.parquet"))
        # first name that exists is the one this store uses
        if files:
            return files[-1]
    # absent is a reportable state, not a crash: a store may legitimately not
    # carry every table
    return None


def hive_keys(path, root):
    """Partition keys encoded in the path, as key=value directory segments.

    THIS IS THE COLUMN THAT BIT US. A hive-partitioned dataset read presents
    these as ordinary columns; a direct file read does not have them. Naming
    them here is the whole reason the previous dump looked wrong.
    """
    # the path segments between the store root and the file
    try:
        # relative, so the store root's own directory names are not scanned
        parts = path.relative_to(root).parts
    # a path outside the root should not happen, but must not crash the dump
    except ValueError:
        parts = path.parts
    # any segment of the form key=value is a hive partition key
    return [p.split("=", 1) for p in parts if "=" in p and not p.endswith(".parquet")]


def describe(path, want_nulls):
    """One table's schema as a frame, plus its row count.

    Reads only the parquet FOOTER unless want_nulls is set. The footer carries
    the column names, their arrow types, and the row count -- everything this
    dump needs -- without touching a single data page.
    """
    # the arrow schema from the footer
    schema = pq.read_schema(path)
    # the row count, also from the footer
    n_rows = pq.ParquetFile(path).metadata.num_rows
    # one row per column: position, name, and the ACTUAL arrow type
    rows = [{"i": i, "column_name": name, "column_type": str(typ)}
            for i, (name, typ) in enumerate(zip(schema.names, schema.types))]
    # the frame
    df = pd.DataFrame(rows)
    # OPTIONAL: how much of each column is actually populated. A column can
    # exist and be entirely null, which is indistinguishable from absent for
    # any consumer that reads it.
    if want_nulls:
        # this reads the data, which is why it is opt-in
        data = pd.read_parquet(path)
        # fraction null, per column, as a percentage
        df["pct_null"] = [
            100.0 * data[c].isna().mean() if c in data.columns else float("nan")
            for c in df["column_name"]]
        # and whether the column is entirely empty, called out plainly
        df["all_null"] = df["pct_null"] >= 100.0
    # the frame and the row count
    return df, n_rows


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # the expensive, opt-in null pass
    ap.add_argument("--nulls", action="store_true",
                    help="also report the fraction of nulls per column "
                         "(reads the data, not just the footer)")
    args = ap.parse_args()

    # the lines of the output file, accumulated then written once
    out = []
    # a header that says what this is and where it came from
    out.append("=" * 78)
    out.append("PARSED STORE SCHEMA -- read from the parquet footers")
    out.append("=" * 78)
    out.append(f"generated : {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    out.append(f"store     : {PARSED_ROOT}")
    out.append(f"nulls     : {'yes (data read)' if args.nulls else 'no (footer only)'}")
    out.append("")
    out.append("This file is GENERATED. Do not hand-edit it, and do not trust")
    out.append("any copy that does not name the partition it was read from.")
    out.append("")
    out.append("TWO SCHEMAS, BOTH CORRECT. The store is hive-partitioned, so a")
    out.append("DATASET read of <table>/** synthesises the partition keys below")
    out.append("as ordinary columns, while a direct read of the FILE does not")
    out.append("have them. The column lists here are the FILE schema; the")
    out.append("partition keys are named per table so the difference is stated")
    out.append("rather than discovered by a crash.")
    out.append("")

    # every table
    for table, desc, suffixes in TABLES:
        # the most recent partition of it, under whichever name it uses
        path = newest_partition(PARSED_ROOT, suffixes)
        # a table with no partitions is reported, not skipped silently
        if path is None:
            out.append("=" * 78)
            out.append(f"Table: {table}  --  NOT PRESENT IN THIS STORE")
            out.append(f"  {desc}")
            # name every pattern tried, so "not found" cannot be confused
            # with "looked for under the wrong name"
            for suffix in suffixes:
                out.append(f"  searched: {PARSED_ROOT}/**/*_{suffix}.parquet")
            out.append("  Nothing matched any of those. Either the parser was")
            out.append("  not asked to keep this table, or it is stored under a")
            out.append("  name none of the patterns above cover.")
            out.append("")
            # tell the console too
            print(f"  {table:12s} NOT PRESENT (tried: "
                  f"{', '.join(suffixes)})")
            continue
        # read it
        try:
            # the schema frame and the row count
            df, n_rows = describe(path, args.nulls)
        # a failure on one table must not lose the other three
        except Exception as exc:                              # noqa: BLE001
            out.append("=" * 78)
            out.append(f"Table: {table}  --  COULD NOT READ")
            out.append(f"  file  : {path.name}")
            out.append(f"  error : {exc!r}")
            out.append("")
            print(f"  {table:12s} COULD NOT READ: {exc!r}")
            continue
        # the section header, naming the partition the schema came from
        out.append("=" * 78)
        out.append(f"Table: {table}")
        out.append(f"  {desc}")
        out.append(f"  source file : {path.name}")
        out.append(f"  full path   : {path}")
        out.append(f"  rows        : {n_rows:,}")
        out.append(f"  columns     : {len(df)}  (in the FILE)")
        # the hive partition keys this file sits under
        keys = hive_keys(path, Path(PARSED_ROOT))
        # named explicitly, with what a dataset read would show
        if keys:
            # each key, with the value this particular file carries
            for k, v in keys:
                out.append(f"  partition   : {k} = {v}  "
                           f"(from the PATH, not the footer)")
            out.append(f"  -> a hive-partitioned dataset read shows "
                       f"{len(df) + len(keys)} columns: these "
                       f"{len(df)} plus "
                       f"{', '.join(k for k, _ in keys)}.")
        else:
            out.append("  partition   : none (flat layout)")
        out.append("")
        # the column table itself
        out.append(df.to_string(index=False,
                                float_format=lambda v: f"{v:.2f}"))
        out.append("")
        # console progress
        print(f"  {table:12s} {len(df):3d} cols, {n_rows:>12,} rows  "
              f"({path.name})")

    # ---- write, never overwriting ---------------------------------------
    # a timestamped name, so the existing schema_output.txt is untouched
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    # results root from config, not typed here
    dest = Path(RESULTS_ROOT) / f"schema_output_{stamp}.txt"
    # refuse rather than clobber, on the vanishingly unlikely name collision
    if dest.exists():
        raise SystemExit(f"{dest} already exists; refusing to overwrite")
    # one write
    dest.write_text("\n".join(out) + "\n")
    # say where it went
    print(f"\nwrote {dest}")
    print("Replace schema_output.txt with this ONLY after reading it --")
    print("it describes the most recent partition of each table, which is")
    print("not necessarily every partition in the store.")


# entry point
if __name__ == "__main__":
    main()
