import zipfile
import time
from pathlib import Path
from datetime import datetime

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ============================================================
# CONFIGURATION
# ============================================================

ZIP_PATHS = [
    Path(
        "C:/Users/shahzeb/Desktop/Del/"
        "PP_GUNICIEMIR.M.202607.zip"
    )
]

EXTRACT_ONE_DAY = True
TARGET_DATE = "20260701"

OUTPUT_DIR = Path(
    "C:/Users/shahzeb/Desktop/Del/Output"
)


# ============================================================
# MEMORY / PERFORMANCE SETTINGS
# ============================================================

# Amount of UNCOMPRESSED CSV data Arrow reads at a time.
#
# 16 MB is deliberately conservative.
#
# This does NOT mean total RAM = 16 MB because Arrow needs
# buffers for parsed strings, filtering, Parquet encoding, etc.
#
# But RAM remains bounded instead of creating a 298 GB CSV.
CSV_BLOCK_SIZE = 16 * 1024 * 1024


# Accumulate this many MATCHED rows before writing them to
# Parquet.
#
# This prevents thousands of tiny Parquet row groups while
# still keeping memory usage controlled.
WRITE_BUFFER_ROWS = 250_000


# Target Parquet row-group size.
PARQUET_ROW_GROUP_SIZE = 250_000


# Fast/lightweight compression.
PARQUET_COMPRESSION = "snappy"


# Print progress at least approximately this often.
PROGRESS_INTERVAL_SECONDS = 10


# ============================================================
# BAD ROW HANDLING
# ============================================================

# For exchange data I recommend "error".
#
# If a malformed CSV row occurs, the program STOPS instead of
# silently throwing data away.
#
# Alternatives:
#
#     "error"  -> STOP immediately
#     "skip"   -> skip malformed rows and count them
#
BAD_ROW_POLICY = "error"


# ============================================================
# DATE DETECTION
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


# ============================================================
# PRINTING
# ============================================================

def log(message=""):
    print(message, flush=True)


def format_elapsed(seconds):

    seconds = int(seconds)

    hours, remainder = divmod(
        seconds,
        3600
    )

    minutes, seconds = divmod(
        remainder,
        60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# HEADER HELPERS
# ============================================================

def is_likely_header_line(
    line,
    delimiter,
):

    if not line.strip():
        return False

    parts = line.split(
        delimiter
    )

    if len(parts) < 2:
        return False

    digit_parts = sum(
        1
        for part in parts
        if any(
            char.isdigit()
            for char in part
        )
    )

    if digit_parts > len(parts) * 0.5:
        return False

    alpha_parts = sum(
        1
        for part in parts
        if (
            any(
                char.isalpha()
                for char in part
            )
            and not any(
                char.isdigit()
                for char in part
            )
        )
    )

    return (
        alpha_parts
        > len(parts) * 0.5
    )


def make_headers_unique(headers):

    counts = {}
    result = []

    for header in headers:

        header = header.strip()

        if not header:
            header = "unnamed"

        count = (
            counts.get(
                header,
                0
            )
            + 1
        )

        counts[header] = count

        if count == 1:

            result.append(
                header
            )

        else:

            result.append(
                f"{header}_{count}"
            )

    return result


# ============================================================
# ZIP INSPECTION
# ============================================================

def inspect_zip(
    zip_path,
    sample_rows_to_read=1000,
):
    """
    Inspect only a tiny portion of the ZIP.

    NO extraction to disk.

    Returns:
        member name
        uncompressed size
        compressed size
        delimiter
        headers
        number of header rows
        header type
        sample rows
    """

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as z:

        members = [
            member
            for member in z.infolist()
            if not member.is_dir()
        ]

        if not members:

            raise RuntimeError(
                "ZIP contains no files"
            )

        member = members[0]

        member_name = (
            member.filename
        )

        uncompressed_size = (
            member.file_size
        )

        compressed_size = (
            member.compress_size
        )

        # ----------------------------------------------------
        # Determine delimiter from first 8 KB.
        # ----------------------------------------------------

        with z.open(
            member_name
        ) as f:

            sample_bytes = f.read(
                8192
            )

        sample_text = (
            sample_bytes.decode(
                "utf-8",
                errors="ignore",
            )
        )

        if "\x01" in sample_text:

            delimiter = "\x01"

        elif "|" in sample_text:

            delimiter = "|"

        elif (
            ";" in sample_text
            and
            sample_text.count(";")
            >
            sample_text.count(",")
        ):

            delimiter = ";"

        elif "\t" in sample_text:

            delimiter = "\t"

        else:

            delimiter = ","

        # ----------------------------------------------------
        # Read headers and tiny sample.
        # ----------------------------------------------------

        with z.open(
            member_name
        ) as f:

            first_line = (
                f.readline()
                .decode(
                    "utf-8",
                    errors="ignore",
                )
                .strip()
            )

            second_line = (
                f.readline()
                .decode(
                    "utf-8",
                    errors="ignore",
                )
                .strip()
            )

            first_headers = (
                first_line.split(
                    delimiter
                )
            )

            second_headers = (
                second_line.split(
                    delimiter
                )
                if second_line
                else []
            )

            # ------------------------------------------------
            # Your Borsa Istanbul files appear to have:
            #
            # row 1 = Turkish headers
            # row 2 = English headers
            # ------------------------------------------------

            if (
                second_headers
                and is_likely_header_line(
                    second_line,
                    delimiter,
                )
            ):

                headers = (
                    second_headers
                )

                header_rows = 2

                header_type = (
                    "English"
                )

            else:

                headers = (
                    first_headers
                )

                header_rows = 1

                header_type = (
                    "Turkish"
                )

                # We already consumed the second line,
                # and it is actually DATA.
                #
                # Save it as the first sample row.
                first_data_line = (
                    second_line
                )

            headers = (
                make_headers_unique(
                    headers
                )
            )

            sample_rows = []

            # ------------------------------------------------
            # If only one header exists, second_line was data.
            # ------------------------------------------------

            if (
                header_rows == 1
                and second_line
            ):

                values = (
                    second_line.split(
                        delimiter
                    )
                )

                if len(values) >= len(headers):

                    values = (
                        values[
                            :len(headers)
                        ]
                    )

                else:

                    values = (
                        values
                        +
                        [""] *
                        (
                            len(headers)
                            -
                            len(values)
                        )
                    )

                sample_rows.append(
                    values
                )

            # ------------------------------------------------
            # Read a small sample only.
            # ------------------------------------------------

            while (
                len(sample_rows)
                <
                sample_rows_to_read
            ):

                line_bytes = (
                    f.readline()
                )

                if not line_bytes:
                    break

                line = (
                    line_bytes
                    .decode(
                        "utf-8",
                        errors="ignore",
                    )
                    .strip()
                )

                if not line:
                    continue

                values = (
                    line.split(
                        delimiter
                    )
                )

                # Only for DATE DETECTION.
                #
                # The real CSV parser later handles the data.
                if len(values) >= len(headers):

                    values = (
                        values[
                            :len(headers)
                        ]
                    )

                else:

                    values = (
                        values
                        +
                        [""] *
                        (
                            len(headers)
                            -
                            len(values)
                        )
                    )

                sample_rows.append(
                    values
                )

    return {
        "member_name":
            member_name,

        "uncompressed_size":
            uncompressed_size,

        "compressed_size":
            compressed_size,

        "delimiter":
            delimiter,

        "headers":
            headers,

        "header_rows":
            header_rows,

        "header_type":
            header_type,

        "sample_rows":
            sample_rows,
    }


# ============================================================
# DATE COLUMN DETECTION
# ============================================================

def detect_date_column(
    headers,
    sample_rows,
):

    candidates = []

    for column_index, column in enumerate(
        headers
    ):

        lower = (
            column.lower()
        )

        # Give columns whose names look date-related
        # a large preference.
        name_score = (
            100
            if any(
                indicator in lower
                for indicator
                in DATE_INDICATORS
            )
            else 0
        )

        value_score = 0

        for row in sample_rows[:100]:

            if (
                column_index
                >=
                len(row)
            ):
                continue

            value = (
                str(
                    row[column_index]
                )
                .strip()
            )

            if not value:
                continue

            if (
                any(
                    separator
                    in value
                    for separator
                    in [
                        "-",
                        "/",
                        ".",
                        ":",
                    ]
                )
                or
                value[:8].isdigit()
            ):

                value_score += 1

        total_score = (
            name_score
            +
            value_score
        )

        if total_score > 0:

            candidates.append(
                (
                    column,
                    total_score,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return (
        candidates[0][0]
    )


# ============================================================
# DATE FORMAT DETECTION
# ============================================================

def detect_date_prefix(
    headers,
    sample_rows,
    date_column,
    target_date,
):

    column_index = (
        headers.index(
            date_column
        )
    )

    values = []

    for row in sample_rows:

        if (
            column_index
            <
            len(row)
        ):

            value = (
                str(
                    row[
                        column_index
                    ]
                )
                .strip()
            )

            if value:

                values.append(
                    value
                )

    best_format = None
    best_description = None
    best_score = 0

    for (
        fmt,
        width,
        description,
    ) in DATE_FORMATS:

        score = 0

        for value in values:

            if len(value) < width:
                continue

            candidate = (
                value[:width]
            )

            try:

                datetime.strptime(
                    candidate,
                    fmt,
                )

                score += 1

            except ValueError:

                pass

        if score > best_score:

            best_score = score
            best_format = fmt
            best_description = (
                description
            )

    if best_format is None:

        return (
            None,
            None,
        )

    target_dt = (
        datetime.strptime(
            target_date,
            "%Y%m%d",
        )
    )

    prefix = (
        target_dt.strftime(
            best_format
        )
    )

    return (
        prefix,
        best_description,
    )


# ============================================================
# PARQUET METADATA
# ============================================================

def get_parquet_metadata(
    file_path,
):

    parquet = (
        pq.ParquetFile(
            file_path
        )
    )

    return (
        parquet.metadata.num_rows,
        parquet.metadata.num_columns,
    )


# ============================================================
# PROCESS ONE ZIP
# ============================================================

def process_single_zip(
    zip_path,
    target_date=None,
):

    overall_start = (
        time.monotonic()
    )

    log()
    log("=" * 90)

    log(
        f"PROCESSING: "
        f"{zip_path.name}"
    )

    log("=" * 90)

    # --------------------------------------------------------
    # Inspect source.
    # --------------------------------------------------------

    info = inspect_zip(
        zip_path
    )

    member_name = (
        info[
            "member_name"
        ]
    )

    uncompressed_size = (
        info[
            "uncompressed_size"
        ]
    )

    compressed_size = (
        info[
            "compressed_size"
        ]
    )

    delimiter = (
        info[
            "delimiter"
        ]
    )

    headers = (
        info[
            "headers"
        ]
    )

    header_rows = (
        info[
            "header_rows"
        ]
    )

    sample_rows = (
        info[
            "sample_rows"
        ]
    )

    log(
        f"File inside ZIP: "
        f"{member_name}"
    )

    log(
        f"Compressed size: "
        f"{compressed_size / 1024**3:,.2f} GB"
    )

    log(
        f"Uncompressed size: "
        f"{uncompressed_size / 1024**3:,.2f} GB"
    )

    log(
        f"Delimiter: "
        f"{repr(delimiter)}"
    )

    log(
        f"Columns: "
        f"{len(headers)}"
    )

    log(
        f"Header type: "
        f"{info['header_type']}"
    )

    # --------------------------------------------------------
    # Detect date filtering.
    # --------------------------------------------------------

    date_column = None
    date_prefix = None

    if target_date is not None:

        date_column = (
            detect_date_column(
                headers,
                sample_rows,
            )
        )

        if date_column is None:

            raise RuntimeError(
                "Could not detect "
                "a date column."
            )

        (
            date_prefix,
            date_format_description,
        ) = detect_date_prefix(
            headers,
            sample_rows,
            date_column,
            target_date,
        )

        if date_prefix is None:

            raise RuntimeError(
                f"Could not determine "
                f"date format for "
                f"{date_column!r}."
            )

        log(
            f"Date column: "
            f"{date_column}"
        )

        log(
            f"Date format: "
            f"{date_format_description}"
        )

        log(
            f"Filtering target: "
            f"{date_prefix}"
        )

    # --------------------------------------------------------
    # Output paths.
    # --------------------------------------------------------

    out_parquet = (
        OUTPUT_DIR
        /
        f"{zip_path.stem}_extracted.parquet"
    )

    partial_parquet = (
        OUTPUT_DIR
        /
        f"{zip_path.stem}_extracted.partial.parquet"
    )

    done_marker = (
        OUTPUT_DIR
        /
        f"{zip_path.stem}_extracted.done"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Skip an already completed file.
    # --------------------------------------------------------

    if (
        out_parquet.exists()
        and
        done_marker.exists()
    ):

        rows, columns = (
            get_parquet_metadata(
                out_parquet
            )
        )

        log(
            f"Already completed: "
            f"{rows:,} rows"
        )

        return {
            "file":
                zip_path.name,

            "rows":
                rows,

            "columns":
                columns,

            "output":
                str(
                    out_parquet
                ),
        }

    # --------------------------------------------------------
    # Remove leftovers from an interrupted run.
    # --------------------------------------------------------

    partial_parquet.unlink(
        missing_ok=True
    )

    done_marker.unlink(
        missing_ok=True
    )

    # --------------------------------------------------------
    # Malformed row handling.
    # --------------------------------------------------------

    invalid_rows = {
        "count": 0
    }

    def invalid_row_handler(
        row
    ):

        invalid_rows[
            "count"
        ] += 1

        return "skip"

    if BAD_ROW_POLICY == "skip":

        parse_options = (
            pacsv.ParseOptions(
                delimiter=delimiter,
                invalid_row_handler=(
                    invalid_row_handler
                ),
            )
        )

    elif BAD_ROW_POLICY == "error":

        parse_options = (
            pacsv.ParseOptions(
                delimiter=delimiter
            )
        )

    else:

        raise ValueError(
            "BAD_ROW_POLICY must be "
            "'error' or 'skip'"
        )

    # --------------------------------------------------------
    # Tell Arrow to treat EVERYTHING as string.
    #
    # Important:
    # No expensive type inference across market-data columns.
    # --------------------------------------------------------

    column_types = {
        column: pa.string()
        for column in headers
    }

    read_options = (
        pacsv.ReadOptions(

            # open_csv is currently single-threaded anyway,
            # but make our intention explicit.
            use_threads=False,

            block_size=(
                CSV_BLOCK_SIZE
            ),

            # Skip Turkish + English header rows.
            skip_rows=(
                header_rows
            ),

            # We supply names explicitly.
            column_names=(
                headers
            ),

            encoding="utf8",
        )
    )

    convert_options = (
        pacsv.ConvertOptions(

            column_types=(
                column_types
            ),

            # Preserve string values instead of interpreting
            # things like "NA" as null.
            strings_can_be_null=False,

            # Avoid expensive UTF-8 validation across every
            # string column.
            check_utf8=False,
        )
    )

    # --------------------------------------------------------
    # Counters.
    # --------------------------------------------------------

    processed_rows = 0
    matched_rows = 0
    written_rows = 0

    buffered_rows = 0
    buffered_batches = []

    last_progress_time = (
        time.monotonic()
    )

    stream_start = (
        time.monotonic()
    )

    writer = None

    # ========================================================
    # DIRECT:
    #
    # ZIP -> Arrow CSV -> date filter -> Parquet
    #
    # NOTHING IS EXTRACTED TO DISK.
    # ========================================================

    try:

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as z:

            with z.open(
                member_name,
                "r",
            ) as zip_stream:

                # --------------------------------------------
                # Arrow reads directly from ZipExtFile.
                # --------------------------------------------

                csv_reader = (
                    pacsv.open_csv(

                        zip_stream,

                        read_options=(
                            read_options
                        ),

                        parse_options=(
                            parse_options
                        ),

                        convert_options=(
                            convert_options
                        ),
                    )
                )

                # --------------------------------------------
                # Create ONE ParquetWriter for the entire
                # output file.
                #
                # We never reread previous output.
                # --------------------------------------------

                writer = (
                    pq.ParquetWriter(

                        partial_parquet,

                        csv_reader.schema,

                        compression=(
                            PARQUET_COMPRESSION
                        ),

                        use_dictionary=True,

                        write_statistics=True,
                    )
                )

                date_column_index = None

                if date_column is not None:

                    date_column_index = (
                        csv_reader.schema
                        .get_field_index(
                            date_column
                        )
                    )

                    if date_column_index < 0:

                        raise RuntimeError(
                            f"Date column "
                            f"{date_column!r} "
                            f"not present in "
                            f"parsed schema."
                        )

                # ============================================
                # STREAM RECORD BATCHES
                # ============================================

                for batch in csv_reader:

                    processed_rows += (
                        batch.num_rows
                    )

                    # ----------------------------------------
                    # Filter date immediately.
                    # ----------------------------------------

                    if date_column is not None:

                        date_values = (
                            batch.column(
                                date_column_index
                            )
                        )

                        # Date is ASCII; this is cheaper than
                        # parsing every value to datetime.
                        cleaned_dates = (
                            pc.ascii_trim_whitespace(
                                date_values
                            )
                        )

                        mask = (
                            pc.starts_with(
                                cleaned_dates,
                                date_prefix,
                            )
                        )

                        filtered_batch = (
                            batch.filter(
                                mask,
                                null_selection_behavior="drop",
                            )
                        )

                    else:

                        filtered_batch = (
                            batch
                        )

                    batch_matched = (
                        filtered_batch.num_rows
                    )

                    matched_rows += (
                        batch_matched
                    )

                    # ----------------------------------------
                    # Buffer ONLY MATCHED DATA.
                    # ----------------------------------------

                    if batch_matched > 0:

                        buffered_batches.append(
                            filtered_batch
                        )

                        buffered_rows += (
                            batch_matched
                        )

                    # ----------------------------------------
                    # Flush matched rows periodically.
                    # ----------------------------------------

                    if (
                        buffered_rows
                        >=
                        WRITE_BUFFER_ROWS
                    ):

                        table = (
                            pa.Table.from_batches(
                                buffered_batches,
                                schema=(
                                    csv_reader.schema
                                ),
                            )
                        )

                        writer.write_table(
                            table,
                            row_group_size=(
                                PARQUET_ROW_GROUP_SIZE
                            ),
                        )

                        written_rows += (
                            table.num_rows
                        )

                        buffered_batches.clear()

                        buffered_rows = 0

                        # Release table immediately.
                        del table

                    # ----------------------------------------
                    # PROGRESS
                    # ----------------------------------------

                    now = (
                        time.monotonic()
                    )

                    if (
                        now
                        -
                        last_progress_time
                        >=
                        PROGRESS_INTERVAL_SECONDS
                    ):

                        elapsed = (
                            now
                            -
                            stream_start
                        )

                        # ZipExtFile.tell() reports our logical
                        # position in the UNCOMPRESSED member.
                        bytes_read = (
                            zip_stream.tell()
                        )

                        if uncompressed_size > 0:

                            progress_pct = (
                                bytes_read
                                /
                                uncompressed_size
                                *
                                100
                            )

                        else:

                            progress_pct = 0

                        gb_read = (
                            bytes_read
                            /
                            1024**3
                        )

                        gb_total = (
                            uncompressed_size
                            /
                            1024**3
                        )

                        mb_per_second = (
                            bytes_read
                            /
                            1024**2
                            /
                            elapsed
                            if elapsed > 0
                            else 0
                        )

                        rows_per_second = (
                            processed_rows
                            /
                            elapsed
                            if elapsed > 0
                            else 0
                        )

                        match_pct = (
                            matched_rows
                            /
                            processed_rows
                            *
                            100
                            if processed_rows > 0
                            else 0
                        )

                        try:

                            output_mb = (
                                partial_parquet
                                .stat()
                                .st_size
                                /
                                1024**2
                            )

                        except OSError:

                            output_mb = 0

                        log(
                            f"[{format_elapsed(elapsed)}] "
                            f"{progress_pct:5.1f}% | "
                            f"{gb_read:,.1f}/"
                            f"{gb_total:,.1f} GB | "
                            f"Rows: "
                            f"{processed_rows:,} | "
                            f"Matched: "
                            f"{matched_rows:,} "
                            f"({match_pct:.2f}%) | "
                            f"Written: "
                            f"{written_rows:,} | "
                            f"Output: "
                            f"{output_mb:,.1f} MB | "
                            f"{rows_per_second:,.0f} rows/s | "
                            f"{mb_per_second:,.1f} MB/s | "
                            f"Bad rows: "
                            f"{invalid_rows['count']:,}"
                        )

                        last_progress_time = (
                            now
                        )

                # ============================================
                # FINAL BUFFER
                # ============================================

                if buffered_batches:

                    table = (
                        pa.Table.from_batches(
                            buffered_batches,
                            schema=(
                                csv_reader.schema
                            ),
                        )
                    )

                    writer.write_table(
                        table,
                        row_group_size=(
                            PARQUET_ROW_GROUP_SIZE
                        ),
                    )

                    written_rows += (
                        table.num_rows
                    )

                    buffered_batches.clear()

                    buffered_rows = 0

                    del table

                writer.close()

                writer = None

    except Exception:

        # ----------------------------------------------------
        # Make sure file handle is closed.
        # ----------------------------------------------------

        if writer is not None:

            try:
                writer.close()
            except Exception:
                pass

        log()
        log(
            "ERROR: processing did not complete."
        )

        log(
            f"Partial output remains at:"
        )

        log(
            f"   {partial_parquet}"
        )

        raise

    # ========================================================
    # SUCCESS
    # ========================================================

    # Atomic-ish final rename on same filesystem.
    partial_parquet.replace(
        out_parquet
    )

    done_marker.write_text(
        datetime.now().isoformat(),
        encoding="utf-8",
    )

    rows, columns = (
        get_parquet_metadata(
            out_parquet
        )
    )

    elapsed = (
        time.monotonic()
        -
        overall_start
    )

    output_size_gb = (
        out_parquet.stat().st_size
        /
        1024**3
    )

    log()
    log("=" * 90)
    log("COMPLETED")
    log("=" * 90)

    log(
        f"Input rows processed: "
        f"{processed_rows:,}"
    )

    log(
        f"Rows matching target: "
        f"{matched_rows:,}"
    )

    log(
        f"Rows written: "
        f"{rows:,}"
    )

    log(
        f"Columns: "
        f"{columns}"
    )

    log(
        f"Bad rows: "
        f"{invalid_rows['count']:,}"
    )

    log(
        f"Output size: "
        f"{output_size_gb:,.2f} GB"
    )

    log(
        f"Total time: "
        f"{format_elapsed(elapsed)}"
    )

    log(
        f"Output:"
    )

    log(
        f"   {out_parquet}"
    )

    return {
        "file":
            zip_path.name,

        "rows":
            rows,

        "columns":
            columns,

        "output":
            str(
                out_parquet
            ),
    }


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

        log(
            "No valid ZIP files found."
        )

        return

    log("=" * 90)
    log("DIRECT ZIP -> FILTER -> PARQUET")
    log("=" * 90)

    log(
        f"ZIP files: "
        f"{len(valid_paths)}"
    )

    log(
        f"Target date: "
        f"{TARGET_DATE if EXTRACT_ONE_DAY else 'ALL'}"
    )

    log(
        "Python worker processes: 1"
    )

    log(
        "CSV parser: PyArrow streaming"
    )

    log(
        f"CSV block size: "
        f"{CSV_BLOCK_SIZE / 1024**2:.0f} MB"
    )

    log(
        f"Matched-row write buffer: "
        f"{WRITE_BUFFER_ROWS:,}"
    )

    log(
        f"Progress interval: "
        f"{PROGRESS_INTERVAL_SECONDS} sec"
    )

    log(
        "Temporary extracted CSV: NONE"
    )

    log("=" * 90)

    target_date = (
        TARGET_DATE
        if EXTRACT_ONE_DAY
        else None
    )

    results = []

    overall_start = (
        time.monotonic()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Sequential only.
    #
    # ZIP 1 must finish before ZIP 2 begins.
    #
    # NO multiprocessing.
    # NO ProcessPoolExecutor.
    # --------------------------------------------------------

    for number, zip_path in enumerate(
        valid_paths,
        start=1,
    ):

        log()
        log(
            f"Starting ZIP "
            f"{number} of "
            f"{len(valid_paths)}"
        )

        try:

            result = (
                process_single_zip(
                    zip_path,
                    target_date,
                )
            )

            results.append(
                result
            )

        except KeyboardInterrupt:

            log()
            log(
                "Interrupted by user."
            )

            log(
                "Final output was NOT marked complete."
            )

            return

    total_elapsed = (
        time.monotonic()
        -
        overall_start
    )

    total_rows = sum(
        result["rows"]
        for result in results
    )

    log()
    log("=" * 90)
    log("SUMMARY")
    log("=" * 90)

    for result in results:

        log(
            f"{result['file']}: "
            f"{result['rows']:,} rows"
        )

    log()
    log(
        f"Total rows: "
        f"{total_rows:,}"
    )

    log(
        f"Total time: "
        f"{format_elapsed(total_elapsed)}"
    )

    log()
    log("=" * 90)
    log("COMPLETE")
    log("=" * 90)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()