"""
Convert the daily PSX Parquet files to CSV.

CSV has no row limit, so each Parquet becomes exactly one CSV, written in
streaming batches (bounded RAM even for the 33M-row orderbook).

Size expectation: the orderbook CSV will be roughly 6-10 GB of text
(Parquet's zstd compression is doing a lot of work for you). Set
COMPRESS = True to write .csv.gz instead (~10x smaller; pandas/polars
read .csv.gz directly, and Excel does not).

Timestamps are written as ISO-8601 UTC (e.g. 2026-06-30 04:32:01.123+00:00).
Set TIMEZONE = "Asia/Karachi" to write PKT wall-clock times instead.
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# ----------------------------- configuration ------------------------------
IN_DIR = Path(r"/Users/shazzak/HFT Data/Thailand/")
FILES = [
    IN_DIR / "l2-XBKK-20260903.parquet"
]
OUT_DIR = IN_DIR / "csv"

TIMEZONE = "UTC"                 # or "Asia/Karachi" for PKT wall-clock
COMPRESS = False                 # True -> write .csv.gz (~10x smaller)
BATCH_ROWS = 262_144             # parquet read batch size


def parquet_to_csv(src: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (src.stem + (".csv.gz" if COMPRESS else ".csv"))
    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    print(f"{src.name}: {total:,} rows -> {dst.name}")
    written = 0
    try:
        for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_ROWS)):
            df = batch.to_pandas()
            if TIMEZONE != "UTC":
                for col in df.columns:
                    if isinstance(df[col].dtype, pd.DatetimeTZDtype):
                        df[col] = df[col].dt.tz_convert(TIMEZONE)
            df.to_csv(dst,
                      mode="w" if i == 0 else "a",
                      header=(i == 0),
                      index=False,
                      compression="gzip" if COMPRESS else None)
            written += len(df)
            if written % (BATCH_ROWS * 8) == 0:
                print(f"  {written:>12,} / {total:,} rows", flush=True)
    finally:
        pf.close()
    print(f"  done: {written:,} rows -> {dst}")


if __name__ == "__main__":
    for f in FILES:
        parquet_to_csv(f, OUT_DIR)
    print("all done")
