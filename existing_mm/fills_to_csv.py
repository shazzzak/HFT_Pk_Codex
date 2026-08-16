# fills_to_csv.py -- combine fill parquet files into CSVs for Tableau, STREAMING so
# memory never holds more than one batch (safe on many-GB data). Processes BOTH
# fill sources by default, writing ONE csv per source (their schemas differ, so they
# stay as separate tables):
#     * attribution fills (config.FILLS_DIR): queue-position fills WITH net/gross/
#       capture/markout -- the source of truth for P&L. -> fills_attribution.csv
#     * raw fills (config.FILLS_RAW_DIR): optimistic counterparty-to-every-trade
#       set, no net-P&L columns -- reference only.               -> fills_raw.csv
#
# Run in PyCharm (uses the .backtest interpreter):
#     python fills_to_csv.py                 # both sources (default)
#     python fills_to_csv.py --attribution   # attribution only
#     python fills_to_csv.py --raw           # raw only

# stdlib
import sys
import csv
import glob
import os
import time
# parquet reader (streams row-group batches, never loads whole files)
import pyarrow.parquet as pq
# central paths (edit paths in config.py, not here)
from config import FILLS_DIR, FILLS_RAW_DIR, EXPORT_DIR

# rows per streamed batch: peak memory ~= one batch. Lower this if RAM is tight.
BATCH_ROWS = 100_000


# format a duration as compact mm:ss
def _fmt(sec):
    # minutes and zero-padded seconds
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# find every parquet under a root: flat files AND nested subfolders (the plain
# fills set is micro/PPL/date=...parquet; attribution is flat) -- recursive covers both
def find_parquet(root):
    # recursive glob + set de-dup, sorted for deterministic (date) order
    return sorted(set(glob.glob(os.path.join(str(root), "**", "*.parquet"),
                                recursive=True)))


# stream one source folder into one csv; returns (rows, columns) or None if empty
def combine_source(label, root, out_path):
    # header per source
    print(f"\n=== {label} ===")
    print(f"source: {root}")
    # gather the files
    files = find_parquet(root)
    # nothing to do -> report and skip (do NOT crash the whole run)
    if not files:
        print(f"  no parquet found under {root} -- skipped")
        return None
    # report count
    print(f"  found {len(files)} parquet files")

    # header written yet for this csv?
    header_written = False
    # canonical column order, locked from the first file of THIS source
    columns = None
    # running totals + timer
    total_rows = 0
    t0 = time.perf_counter()
    # sanity: which (strategy, symbol) combos appear (parsed from filename if present)
    seen_combos = set()

    # open this source's csv once and stream every file's batches into it
    with open(out_path, "w", newline="") as fh:
        # a plain csv writer
        writer = csv.writer(fh)
        # walk files in sorted order
        for i, f in enumerate(files, 1):
            # open parquet metadata only (does NOT load the file into memory)
            pf = pq.ParquetFile(f)
            # lock the column order from the first file of this source
            if columns is None:
                columns = pf.schema_arrow.names
            # write the header exactly once (per source)
            if not header_written:
                writer.writerow(columns)
                header_written = True
            # stream this file in row-group-sized batches
            for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=columns):
                # convert just this batch to python columns (bounded memory)
                cols = [batch.column(c).to_pylist() for c in range(batch.num_columns)]
                # transpose columns -> rows and write them
                writer.writerows(zip(*cols))
                # tally rows
                total_rows += batch.num_rows
            # cheap sanity capture from filename like micro_PPL_2025-09-01.parquet
            # OR path like .../micro/PPL/date=2025-09-01.parquet
            parts = os.path.basename(f).replace(".parquet", "").split("_")
            if len(parts) >= 2:
                seen_combos.add((parts[0], parts[1]))
            # heartbeat every 50 files
            if i % 50 == 0 or i == len(files):
                # elapsed wall-clock
                el = time.perf_counter() - t0
                # progress line
                print(f"  {i}/{len(files)} files  {total_rows:,} rows  "
                      f"elapsed {_fmt(el)}", flush=True)

    # per-source report
    print(f"  wrote {out_path}")
    print(f"  {total_rows:,} rows, {len(columns)} columns")
    print(f"  columns: {columns}")
    # return a small summary for the final recap
    return total_rows, len(columns), sorted(seen_combos)


# main entry point
def main():
    # which sources to run: default = both; flags narrow it
    do_attr = ("--raw" not in sys.argv) or ("--attribution" in sys.argv)
    do_raw = ("--attribution" not in sys.argv) or ("--raw" in sys.argv)
    # if the user passed exactly one flag, honor only that one
    if "--attribution" in sys.argv and "--raw" not in sys.argv:
        do_raw = False
    if "--raw" in sys.argv and "--attribution" not in sys.argv:
        do_attr = False

    # make sure the export folder exists
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # collected recaps for the final summary
    recap = {}
    # attribution set (source of truth: has net/gross/capture/markout)
    if do_attr:
        recap["attribution"] = combine_source(
            "attribution fills (net P&L, source of truth)",
            FILLS_DIR, EXPORT_DIR / "fills_attribution.csv")
    # raw set (optimistic, reference only)
    if do_raw:
        recap["raw"] = combine_source(
            "raw fills (optimistic, reference only)",
            FILLS_RAW_DIR, EXPORT_DIR / "fills_raw.csv")

    # final recap so you can eyeball both at once
    print("\n================ SUMMARY ================")
    for name, r in recap.items():
        if r is None:
            print(f"  {name}: (no files / skipped)")
        else:
            rows, ncols, combos = r
            print(f"  {name}: {rows:,} rows, {ncols} cols, combos={combos}")
    # remind which is which for the P&L analysis
    print("\nP&L NOTE: use the ATTRIBUTION csv for money -- its 'net' column bakes in")
    print("fees ('gross' is pre-fee; 'capture'/'markout' are bps quality diagnostics).")
    print("The RAW csv has no net-P&L columns; it's for fill-behaviour inspection only.")


# standard entry point
if __name__ == "__main__":
    # run it
    main()