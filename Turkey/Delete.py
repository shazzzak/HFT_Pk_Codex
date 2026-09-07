import zipfile
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import duckdb


ZIP_PATH = Path("/Users/shazzak/HFT Data/Turkey/PP_GUNICIEMIR.M.202605.zip")
N_ROWS = 5000
OUT_CSV = ZIP_PATH.with_name(f"{ZIP_PATH.stem}_sample_{N_ROWS}.csv")
OUT_PARQUET = ZIP_PATH.with_name(f"{ZIP_PATH.stem}_sample_{N_ROWS}.parquet")


def list_zip_contents(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as z:
        print("Files inside zip:")
        for info in z.infolist():
            print(f" - {info.filename} | {info.file_size:,} bytes")
        return z.infolist()


def parse_fix_sample(zip_path: Path, member_name: str, n_rows: int = 5000):
    """
    Basic FIX parser.
    Converts each FIX message into a dict of tag=value pairs.
    Handles both SOH-delimited and pipe-delimited FIX-ish data.
    """
    rows = []

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member_name) as f:
            for raw_line in f:
                line = raw_line.decode("utf-8", errors="replace").strip()

                if not line:
                    continue

                if "\x01" in line:
                    parts = line.split("\x01")
                else:
                    parts = line.split("|")

                row = {}
                for part in parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        row[key] = value

                if row:
                    rows.append(row)

                if len(rows) >= n_rows:
                    break

    df = pd.DataFrame(rows)

    print("\nFIX-style parsed sample using Pandas:")
    print(df.head(20))
    print("\nColumns / FIX tags:")
    print(df.columns.tolist())

    return df


def read_with_pandas(zip_path: Path, member_name: str, delimiter=None, n_rows: int = 5000):
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member_name) as f:
            df = pd.read_csv(
                f,
                sep=delimiter if delimiter else None,
                engine="python",
                nrows=n_rows,
                on_bad_lines="skip",
            )

    print("\nPandas sample:")
    print(df.head(20))
    print("\nPandas dtypes:")
    print(df.dtypes)

    return df


def read_with_polars(zip_path: Path, member_name: str, delimiter=",", n_rows: int = 5000):
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member_name) as f:
            df = pl.read_csv(
                f,
                separator=delimiter,
                n_rows=n_rows,
                ignore_errors=True,
                infer_schema_length=100,
            )

    print("\nPolars sample:")
    print(df.head(20))
    print("\nPolars schema:")
    print(df.schema)

    return df


def read_with_duckdb(zip_path: Path, member_name: str, n_rows: int = 5000):
    """
    DuckDB works most reliably if we extract the compressed member
    to a temporary file and query that file.
    """
    suffix = Path(member_name).suffix or ".txt"

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member_name) as src, tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(src.read())
            tmp_path = tmp.name

    con = duckdb.connect()

    query = f"""
        SELECT *
        FROM read_csv_auto('{tmp_path}', sample_size=10000, ignore_errors=true)
        LIMIT {n_rows}
    """

    df = con.execute(query).df()

    print("\nDuckDB sample:")
    print(df.head(20))
    print("\nDuckDB dtypes:")
    print(df.dtypes)

    return df


def export_sample_csv(df, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nExported {len(df):,} rows to:")
    print(out_csv)


def export_sample_parquet(df, out_parquet: Path):
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pl.DataFrame):
        df.write_parquet(out_parquet)
    else:
        df.to_parquet(out_parquet, index=False)
    print(f"\nExported {len(df):,} rows to:")
    print(out_parquet)


def preview_raw_bytes(zip_path: Path, member_name: str, n_bytes: int = 1024):
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member_name) as f:
            return f.read(n_bytes)


def looks_like_fix(raw_bytes: bytes):
    return b"\x01" in raw_bytes or raw_bytes.count(b"|") > raw_bytes.count(b",")


def guess_delimiter(raw_bytes: bytes):
    text = raw_bytes.decode("utf-8", errors="ignore")
    if text.count(",") > text.count("\t") and text.count(",") > text.count(";"):
        return ","
    if text.count("\t") > text.count(","):
        return "\t"
    if text.count(";") > text.count(","):
        return ";"
    return ","


def main():
    members = list_zip_contents(ZIP_PATH)

    if not members:
        raise RuntimeError("Zip file is empty")

    # Pick the first non-directory file
    file_members = [m for m in members if not m.is_dir()]
    if not file_members:
        raise RuntimeError("No files found inside zip")

    member_name = file_members[0].filename
    print(f"\nUsing file inside zip: {member_name}")

    text_preview = preview_raw_bytes(ZIP_PATH, member_name)

    if looks_like_fix(text_preview):
        print("\nDetected likely FIX-style data.")
        fix_df = parse_fix_sample(ZIP_PATH, member_name, N_ROWS)
        export_sample_csv(fix_df, OUT_CSV)
        export_sample_parquet(fix_df, OUT_PARQUET)
        return fix_df

    delimiter = guess_delimiter(text_preview)
    print(f"\nGuessed delimiter: {repr(delimiter)}")

    pandas_df = None
    polars_df = None
    duckdb_df = None

    try:
        pandas_df = read_with_pandas(ZIP_PATH, member_name, delimiter, N_ROWS)
        export_sample_csv(pandas_df, OUT_CSV)
        export_sample_parquet(pandas_df, OUT_PARQUET)
    except Exception as e:
        print("\nPandas failed:")
        print(repr(e))

    try:
        polars_delimiter = delimiter if delimiter else ","
        polars_df = read_with_polars(ZIP_PATH, member_name, polars_delimiter, N_ROWS)
    except Exception as e:
        print("\nPolars failed:")
        print(repr(e))

    try:
        duckdb_df = read_with_duckdb(ZIP_PATH, member_name, N_ROWS)
        if pandas_df is None:
            export_sample_csv(duckdb_df, OUT_CSV)
            export_sample_parquet(duckdb_df, OUT_PARQUET)
    except Exception as e:
        print("\nDuckDB failed:")
        print(repr(e))

    return pandas_df, polars_df, duckdb_df


if __name__ == "__main__":
    result = main()