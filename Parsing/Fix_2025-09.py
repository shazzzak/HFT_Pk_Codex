"""sort_all_days.py -- one-time rewrite of the canonical store, sorted by
(symbol, wire-sequence) so parquet row-group stats enable symbol pruning.
Restricted to September 2025: days after that were parsed with the in-parser
sorter already. Safe to re-run: skips files already sorted."""
from pathlib import Path
import duckdb
import pyarrow.parquet as pq

PARSED = Path(r"D:\HFT\parsed")          # <- your parsed_root
SORT_KEYS = {"trades": "symbol, appl_seq", "ob_updates": "symbol, appl_seq",
             "ob_snapshot": "symbol, msg_seq", "misc": "symbol, msg_seq"}

# Only rewrite September 2025 days; later days were born sorted.
DATE_PREFIX = "2025-09"

def in_scope(path):
    # Take the date from the PARTITION DIRECTORY name (date=YYYY-MM-DD),
    # not the filename -- the directory is the authoritative label.
    part = path.parent.name                       # e.g. "date=2025-09-15"
    return part.startswith(f"date={DATE_PREFIX}")

def already_sorted(path):
    # Sorted file signature: first row group's symbol min == max (single symbol).
    md = pq.ParquetFile(path).metadata
    i = md.schema.to_arrow_schema().get_field_index("symbol")
    st = md.row_group(0).column(i).statistics
    return st is not None and st.min == st.max

for label, keys in SORT_KEYS.items():
    for f in sorted((PARSED / label).glob("date=*/*.parquet")):
        # Outside September 2025 -> parsed with the sorting merge, leave alone.
        if not in_scope(f):
            continue
        if already_sorted(f):
            print(f"skip (sorted): {f.name}")
            continue
        tmp = f.with_suffix(".parquet.sorted")
        # External sort: DuckDB spills to disk -- no RAM spike on 58M-row days.
        duckdb.sql(f"""
            COPY (SELECT * FROM read_parquet('{f.as_posix()}') ORDER BY {keys})
            TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
        """)
        tmp.replace(f)
        print(f"sorted: {f.name}")