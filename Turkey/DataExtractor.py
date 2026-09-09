import zipfile
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

import polars as pl
import pyarrow.parquet as pq


# ============================================================
# CONFIGURATION
# ============================================================

ZIP_PATHS = [
    Path("C:/Users/shahzeb/Desktop/Del/PP_GUNICIEMIR.M.202607.zip")
]

EXTRACT_ONE_DAY = True
TARGET_DATE = "20260701"

OUTPUT_DIR = Path("C:/Users/shahzeb/Desktop/Del/Output")

# IMPORTANT:
# Polars itself uses multiple CPU threads.
# Start with 1 worker. Try 2 only if processing multiple ZIPs.
NUM_WORKERS = 1

# Faster than zstd and perfectly reasonable for intermediate market data.
PARQUET_COMPRESSION = "snappy"

# Larger row groups usually improve write efficiency.
ROW_GROUP_SIZE = 250_000


# ============================================================
# HELPERS
# ============================================================

DATE_INDICATORS = [
    "date",
    "time",
    "transact",
    "timestamp",
    "created",
    "updated",
    "datetime",
    "dt",
    "trade",
    "settle",
    "tarih",
    "zaman",
    "islem",
    "giris",
    "degistirilme",
    "guncelleme",
    "takas",
    "saat",
]


DATE_FORMATS = [
    ("%Y-%m-%d", 10, "YYYY-MM-DD"),
    ("%Y/%m/%d", 10, "YYYY/MM/DD"),
    ("%d-%m-%Y", 10, "DD-MM-YYYY"),
    ("%d/%m/%Y", 10, "DD/MM/YYYY"),
    ("%d.%m.%Y", 10, "DD.MM.YYYY"),
    ("%Y%m%d", 8, "YYYYMMDD"),
]


def is_likely_header_line(line, delimiter):
    """Determine whether a line looks like a header."""

    if not line.strip():
        return False

    parts = line.split(delimiter)

    if len(parts) < 2:
        return False

    digit_parts = sum(
        1
        for p in parts
        if any(c.isdigit() for c in p)
    )

    if digit_parts > len(parts) * 0.5:
        return False

    alpha_parts = sum(
        1
        for p in parts
        if any(c.isalpha() for c in p)
        and not any(c.isdigit() for c in p)
    )

    return alpha_parts > len(parts) * 0.5


def make_headers_unique(headers):
    """
    Polars requires unique column names.

    Example:
        PRICE, PRICE
    becomes:
        PRICE, PRICE_2
    """

    counts = {}
    result = []

    for header in headers:

        header = header.strip()

        if not header:
            header = "unnamed"

        count = counts.get(header, 0) + 1
        counts[header] = count

        if count == 1:
            result.append(header)
        else:
            result.append(f"{header}_{count}")

    return result


def inspect_zip(zip_path):
    """
    Inspect the ZIP without parsing the entire CSV.

    Returns:
        member_name
        delimiter
        headers
        number_of_header_rows
    """

    with zipfile.ZipFile(zip_path, "r") as z:

        members = [
            m for m in z.infolist()
            if not m.is_dir()
        ]

        if not members:
            raise RuntimeError("ZIP contains no files")

        member_name = members[0].filename

        with z.open(member_name) as f:
            sample_bytes = f.read(8192)

        sample_text = sample_bytes.decode(
            "utf-8",
            errors="ignore"
        )

        # ----------------------------------------------------
        # Detect delimiter
        # ----------------------------------------------------

        if "\x01" in sample_text:
            delimiter = "\x01"

        elif "|" in sample_text:
            delimiter = "|"

        elif (
            ";" in sample_text
            and sample_text.count(";") > sample_text.count(",")
        ):
            delimiter = ";"

        elif "\t" in sample_text:
            delimiter = "\t"

        else:
            delimiter = ","

        # ----------------------------------------------------
        # Get first two lines
        # ----------------------------------------------------

        with z.open(member_name) as f:

            first_line = (
                f.readline()
                .decode("utf-8", errors="ignore")
                .strip()
            )

            second_line = (
                f.readline()
                .decode("utf-8", errors="ignore")
                .strip()
            )

        first_headers = first_line.split(delimiter)

        second_headers = (
            second_line.split(delimiter)
            if second_line
            else []
        )

        # ----------------------------------------------------
        # Determine whether second line is English header
        # ----------------------------------------------------

        if (
            second_headers
            and is_likely_header_line(
                second_line,
                delimiter
            )
        ):
            headers = second_headers
            header_rows = 2
            header_type = "English"

        else:
            headers = first_headers
            header_rows = 1
            header_type = "Turkish"

        headers = make_headers_unique(headers)

    return (
        member_name,
        delimiter,
        headers,
        header_rows,
        header_type,
    )


def detect_date_column(df):
    """
    Detect most likely date column.
    """

    candidates = []

    for col in df.columns:

        lower = col.lower()

        if not any(
            indicator in lower
            for indicator in DATE_INDICATORS
        ):
            continue

        values = (
            df[col]
            .drop_nulls()
            .head(20)
            .to_list()
        )

        score = 0

        for value in values:

            value = str(value)

            if (
                any(
                    separator in value
                    for separator in [
                        "/",
                        "-",
                        ":",
                        ".",
                    ]
                )
                or value[:8].isdigit()
            ):
                score += 1

        if score:
            candidates.append(
                (col, score)
            )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    if not candidates:
        return None

    return candidates[0][0]


def detect_date_prefix(df, date_column, target_date):
    """
    Determine the textual date layout once.

    Instead of parsing millions of date strings into datetime
    values, we detect the representation once and then use
    starts_with().

    Example:

        2026-07-01 10:25:31

    becomes:

        starts_with("2026-07-01")
    """

    values = (
        df[date_column]
        .drop_nulls()
        .head(1000)
        .to_list()
    )

    best_format = None
    best_width = None
    best_description = None
    best_score = 0

    for fmt, width, description in DATE_FORMATS:

        score = 0

        for value in values:

            text = str(value).strip()

            if len(text) < width:
                continue

            candidate = text[:width]

            try:
                datetime.strptime(
                    candidate,
                    fmt
                )
                score += 1
            except ValueError:
                pass

        if score > best_score:

            best_score = score
            best_format = fmt
            best_width = width
            best_description = description

    if best_format is None:
        return None, None

    target_dt = datetime.strptime(
        target_date,
        "%Y%m%d"
    )

    prefix = target_dt.strftime(
        best_format
    )

    return prefix, best_description


def get_parquet_metadata(path):
    """
    Read only Parquet metadata.

    Does NOT load the full file.
    """

    parquet = pq.ParquetFile(path)

    return (
        parquet.metadata.num_rows,
        parquet.metadata.num_columns,
    )


# ============================================================
# MAIN ZIP PROCESSOR
# ============================================================

def process_single_zip(
    zip_path,
    target_date=None,
    worker_id=0,
):
    start = time.time()

    print("\n" + "=" * 80)
    print(
        f"Worker {worker_id}: "
        f"Processing {zip_path.name}"
    )
    print("=" * 80)

    out_parquet = (
        OUTPUT_DIR
        / f"{zip_path.stem}_extracted.parquet"
    )

    temp_parquet = (
        OUTPUT_DIR
        / f"{zip_path.stem}_extracted.tmp.parquet"
    )

    # --------------------------------------------------------
    # File-level resume
    # --------------------------------------------------------

    if out_parquet.exists():

        try:

            rows, columns = get_parquet_metadata(
                out_parquet
            )

            print(
                f"Already completed: "
                f"{rows:,} rows"
            )

            return {
                "file": zip_path.name,
                "rows": rows,
                "columns": columns,
                "output": str(out_parquet),
            }

        except Exception:

            print(
                "Existing output appears corrupted. "
                "Reprocessing."
            )

            out_parquet.unlink(
                missing_ok=True
            )

    temp_parquet.unlink(
        missing_ok=True
    )

    # --------------------------------------------------------
    # Inspect ZIP
    # --------------------------------------------------------

    (
        member_name,
        delimiter,
        headers,
        header_rows,
        header_type,
    ) = inspect_zip(zip_path)

    print(f"File: {member_name}")
    print(f"Delimiter: {repr(delimiter)}")
    print(f"Columns: {len(headers)}")
    print(f"Header: {header_type}")

    # --------------------------------------------------------
    # Extract compressed CSV to temporary disk location
    #
    # This lets Polars use its native multithreaded CSV scanner.
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        dir=OUTPUT_DIR
    ) as temp_dir:

        extracted_file = (
            Path(temp_dir)
            / "source.csv"
        )

        print("Extracting ZIP...")

        extract_start = time.time()

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as z:

            with z.open(
                member_name
            ) as src:

                with open(
                    extracted_file,
                    "wb",
                    buffering=16 * 1024 * 1024,
                ) as dst:

                    shutil.copyfileobj(
                        src,
                        dst,
                        length=16 * 1024 * 1024,
                    )

        print(
            f"Extraction: "
            f"{time.time() - extract_start:.2f}s"
        )

        # ----------------------------------------------------
        # Everything remains String.
        #
        # This avoids expensive and unnecessary schema
        # inference.
        # ----------------------------------------------------

        schema = {
            column: pl.String
            for column in headers
        }

        # ----------------------------------------------------
        # Read only a tiny sample for date-column detection.
        #
        # This uses native Polars parsing, NOT Python loops
        # over the full file.
        # ----------------------------------------------------

        sample = pl.read_csv(
            extracted_file,
            has_header=False,
            skip_rows=header_rows,
            schema=schema,
            separator=delimiter,
            n_rows=1000,
            encoding="utf8-lossy",
            ignore_errors=True,
            truncate_ragged_lines=True,
            low_memory=False,
            rechunk=False,
        )

        date_column = None
        date_prefix = None

        if target_date is not None:

            date_column = detect_date_column(
                sample
            )

            if date_column is None:
                raise RuntimeError(
                    "A target date was requested, "
                    "but no date column could be detected."
                )

            date_prefix, date_format = (
                detect_date_prefix(
                    sample,
                    date_column,
                    target_date,
                )
            )

            if date_prefix is None:
                raise RuntimeError(
                    f"Could not determine date format "
                    f"for column {date_column!r}"
                )

            print(
                f"Date column: {date_column}"
            )

            print(
                f"Date format: {date_format}"
            )

            print(
                f"Filtering: {date_prefix}"
            )

        # ----------------------------------------------------
        # FAST PATH
        #
        # Polars now:
        #
        #   1. parses CSV in native Rust
        #   2. uses multiple threads
        #   3. filters while scanning
        #   4. writes directly to Parquet
        #
        # There are NO Python row dictionaries.
        # There are NO repeated Parquet reads.
        # ----------------------------------------------------

        print("Scanning/filtering/writing Parquet...")

        scan_start = time.time()

        lf = pl.scan_csv(
            extracted_file,
            has_header=False,
            skip_rows=header_rows,
            schema=schema,
            separator=delimiter,
            encoding="utf8-lossy",
            ignore_errors=True,
            truncate_ragged_lines=True,
            low_memory=False,
            rechunk=False,
        )

        if target_date is not None:

            lf = lf.filter(
                pl.col(date_column)
                .str.strip_chars()
                .str.starts_with(date_prefix)
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # maintain_order=True is deliberate.
        #
        # This appears to be exchange/market data.
        # Event ordering may be important for your backtester.
        # Do NOT turn it off merely for a small speed gain.
        # ----------------------------------------------------

        lf.sink_parquet(
            temp_parquet,
            compression=PARQUET_COMPRESSION,
            statistics=True,
            row_group_size=ROW_GROUP_SIZE,
            maintain_order=True,
        )

        # ----------------------------------------------------
        # Atomic finalization
        # ----------------------------------------------------

        temp_parquet.replace(
            out_parquet
        )

        processing_time = (
            time.time()
            - scan_start
        )

    # --------------------------------------------------------
    # Get counts from metadata ONLY.
    #
    # No pl.read_parquet(out_parquet)
    # --------------------------------------------------------

    rows, columns = get_parquet_metadata(
        out_parquet
    )

    elapsed = time.time() - start

    size_mb = (
        out_parquet.stat().st_size
        / 1024
        / 1024
    )

    print(
        f"\nCompleted {zip_path.name}"
    )

    print(
        f"Rows: {rows:,}"
    )

    print(
        f"Columns: {columns}"
    )

    print(
        f"Size: {size_mb:,.1f} MB"
    )

    print(
        f"CSV -> Parquet: "
        f"{processing_time:.2f}s"
    )

    print(
        f"Total: {elapsed:.2f}s"
    )

    return {
        "file": zip_path.name,
        "rows": rows,
        "columns": columns,
        "output": str(out_parquet),
    }


# ============================================================
# STREAMING PARQUET MERGE
# ============================================================

def merge_parquet_files(
    output_dir,
    target_date,
):

    parquet_files = sorted(
        output_dir.glob(
            "*_extracted.parquet"
        )
    )

    if not parquet_files:
        return None

    merged_path = (
        output_dir
        / f"merged_{target_date}_all_data.parquet"
    )

    temp_path = (
        output_dir
        / f"merged_{target_date}_all_data.tmp.parquet"
    )

    temp_path.unlink(
        missing_ok=True
    )

    print("\n" + "=" * 80)
    print(
        f"Merging {len(parquet_files)} "
        f"Parquet files..."
    )
    print("=" * 80)

    # Streaming merge.
    #
    # The original code reads every complete Parquet into RAM,
    # stores all DataFrames in a list, concatenates them and
    # then writes everything again.
    #
    # This does not.

    pl.scan_parquet(
        parquet_files
    ).sink_parquet(
        temp_path,
        compression=PARQUET_COMPRESSION,
        statistics=True,
        row_group_size=ROW_GROUP_SIZE,
        maintain_order=True,
    )

    temp_path.replace(
        merged_path
    )

    rows, columns = get_parquet_metadata(
        merged_path
    )

    print(
        f"Merged rows: {rows:,}"
    )

    print(
        f"Merged columns: {columns}"
    )

    print(
        f"Output: {merged_path}"
    )

    return merged_path


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_paths = [
        path
        for path in ZIP_PATHS
        if path.exists()
    ]

    if not valid_paths:
        print("No valid ZIP files found.")
        return

    print("=" * 80)
    print("FAST DATA EXTRACTION TOOL")
    print("=" * 80)

    print(
        f"ZIP files: {len(valid_paths)}"
    )

    print(
        f"Target date: "
        f"{TARGET_DATE if EXTRACT_ONE_DAY else 'ALL'}"
    )

    print(
        f"Workers: {NUM_WORKERS}"
    )

    print("=" * 80)

    start = time.time()

    target_date = (
        TARGET_DATE
        if EXTRACT_ONE_DAY
        else None
    )

    results = []

    # --------------------------------------------------------
    # Do NOT spawn a process just to process one ZIP.
    # --------------------------------------------------------

    if (
        len(valid_paths) == 1
        or NUM_WORKERS == 1
    ):

        for i, path in enumerate(
            valid_paths
        ):

            result = process_single_zip(
                path,
                target_date,
                i,
            )

            if result:
                results.append(result)

    else:

        with ProcessPoolExecutor(
            max_workers=NUM_WORKERS
        ) as executor:

            futures = {
                executor.submit(
                    process_single_zip,
                    path,
                    target_date,
                    i,
                ): path
                for i, path
                in enumerate(valid_paths)
            }

            for future in as_completed(
                futures
            ):

                path = futures[future]

                try:

                    result = future.result()

                    if result:
                        results.append(result)

                except Exception as exc:

                    print(
                        f"Failed: "
                        f"{path.name}: {exc}"
                    )

    elapsed = (
        time.time()
        - start
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_rows = sum(
        result["rows"]
        for result in results
    )

    for result in results:

        print(
            f"{result['file']}: "
            f"{result['rows']:,} rows"
        )

    print(
        f"\nTotal rows: {total_rows:,}"
    )

    print(
        f"Total time: {elapsed:.2f}s"
    )

    if len(results) > 1:

        merge_parquet_files(
            OUTPUT_DIR,
            TARGET_DATE,
        )

    print("\nCOMPLETE")


if __name__ == "__main__":

    try:
        mp.set_start_method(
            "spawn",
            force=True,
        )
    except RuntimeError:
        pass

    main()