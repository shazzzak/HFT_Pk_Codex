"""sort_all_days.py -- one-time rewrite of the canonical store, sorted by
(symbol, wire-sequence) so parquet row-group stats enable symbol pruning.
Restricted to September 2025: days after that were parsed with the in-parser
sorter already. Safe to re-run: skips files already sorted."""
from pathlib import Path
import duckdb
import pyarrow.parquet as pq

PARSED = Path(r"/Users/shazzak/Capital Stake - Parsed")          # <- your parsed_root
SORT_KEYS = {"trades": "symbol, appl_seq",
             "ob_updates": "symbol, appl_seq",
             "ob_snapshot": "symbol, msg_seq",
             "misc": "symbol, msg_seq"}

# Only rewrite September 2025 days; later days were born sorted.
DATE_PREFIX = "2025-09"

def in_scope(path):
    # Take the date from the PARTITION DIRECTORY name (date=YYYY-MM-DD),
    # not the filename -- the directory is the authoritative label.
    part = path.parent.name                       # e.g. "date=2025-09-15"
    return part.startswith(f"date={DATE_PREFIX}")

def sort_key_for(path, keys):
    """Return the ORDER BY clause valid for THIS file, dropping any column the
    file doesn't have. misc/status tables often have no `symbol` column, and
    schema can drift across days, so the key is resolved per file, not assumed."""
    have = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c.strip() for c in keys.split(",")]
    usable = [c for c in cols if c in have]
    return ", ".join(usable) if usable else None


def already_sorted(path):
    """True if the file already looks symbol-sorted. Defensive: a file with no
    `symbol` column, no row groups, or no statistics is reported NOT sorted and
    handled by the caller rather than raising."""
    md = pq.ParquetFile(path).metadata
    if md.num_row_groups == 0:
        return True                      # empty file: nothing to sort
    i = md.schema.to_arrow_schema().get_field_index("symbol")
    if i < 0:
        return True                      # no symbol column: pruning N/A
    st = md.row_group(0).column(i).statistics
    return st is not None and st.min == st.max


print(f"Scanning directory: {PARSED}")
files_found = 0

for label, keys in SORT_KEYS.items():
    for f in sorted((PARSED / label).glob("date=*/*.parquet")):
        files_found += 1

        if already_sorted(f):
            print(f"skip (sorted / no symbol col): {f.name}")
            continue
        order_by = sort_key_for(f, keys)
        if order_by is None:
            print(f"skip (no usable sort column): {f.name}")
            continue
        tmp = f.with_suffix(".parquet.sorted")
        # External sort: DuckDB spills to disk -- no RAM spike on 58M-row days.
        duckdb.sql(f"""
                    COPY (SELECT * FROM read_parquet('{f.as_posix()}') ORDER BY {order_by})
                    TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
                """)
        tmp.replace(f)
        print(f"sorted ({order_by}): {f.name}")

print(f"Process complete. Total parquet files evaluated: {files_found}")