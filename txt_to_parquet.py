"""
Compress a raw FIX capture .txt into a single Parquet file — no parsing,
no splitting. Each line of the text file becomes one row in a one-column
Parquet ("raw_line"), written in batches so memory stays bounded.

The original file is recoverable exactly:
    df = pd.read_parquet(dst)
    open("restored.txt", "w", encoding="utf-8", newline="").write(
        "\\n".join(df["raw_line"]) + "\\n")

Expect ~3 GB text -> roughly 400-600 MB parquet (zstd), similar to the
original .tar.gz.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------- configuration ------------------------------
SRC = Path(r"C:\Users\shahz\OneDrive\Desktop\Del\Capital Stake\2026-06-30.txt")
DST = SRC.with_suffix(".parquet")
BATCH_LINES = 500_000

SCHEMA = pa.schema([("raw_line", pa.string())])


def txt_to_parquet(src: Path = SRC, dst: Path = DST,
                   batch_lines: int = BATCH_LINES):
    writer = pq.ParquetWriter(dst, SCHEMA, compression="zstd")
    n = 0
    buf = []
    try:
        with open(src, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                buf.append(line.rstrip("\n").rstrip("\r"))
                if len(buf) >= batch_lines:
                    writer.write_table(
                        pa.Table.from_arrays([pa.array(buf, pa.string())],
                                             schema=SCHEMA))
                    n += len(buf)
                    print(f"  {n:>12,} lines written", flush=True)
                    buf = []
            if buf:
                writer.write_table(
                    pa.Table.from_arrays([pa.array(buf, pa.string())],
                                         schema=SCHEMA))
                n += len(buf)
    finally:
        writer.close()
    mb_in = src.stat().st_size / 1e6
    mb_out = dst.stat().st_size / 1e6
    print(f"done: {n:,} lines | {mb_in:,.0f} MB -> {mb_out:,.0f} MB "
          f"({mb_in/mb_out:.1f}x) | {dst}")
    return dst


if __name__ == "__main__":
    txt_to_parquet()
