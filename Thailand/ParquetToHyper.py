import os
import stat
import subprocess
from pathlib import Path

import polars as pl
import pantab
from tableauhyperapi import (
    HyperProcess,
    Telemetry,
    Connection,
    CreateMode,
    TableDefinition,
    SqlType,
    TableName,
    Inserter,
)


input_parquet = Path("C:/Users/shahzeb/Desktop/Del/Output/")
output_hyper = Path("C:/Users/shahzeb/Desktop/Del/Output/")
table_name = TableName("Extract")


def prepare_hyper_binary():
    """
    On macOS, the bundled hyperd executable can sometimes be blocked by quarantine
    or missing executable permissions after installation.
    """
    pantab_dir = Path(pantab.__file__).resolve().parent
    hyperd = pantab_dir / "hyper" / "hyperd"

    if not hyperd.exists():
        print(f"Warning: Could not find bundled hyperd at: {hyperd}")
        return

    try:
        current_mode = hyperd.stat().st_mode
        hyperd.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception as exc:
        print(f"Warning: Could not update execute permission for hyperd: {exc}")

    if os.name == "posix":
        try:
            subprocess.run(
                ["xattr", "-dr", "com.apple.quarantine", str(hyperd)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"Warning: Could not remove macOS quarantine attribute: {exc}")


def polars_dtype_to_hyper_type(dtype):
    """
    Convert common Polars data types to Tableau Hyper SQL types.
    Extend this mapping if your Parquet file contains more specialized columns.
    """
    if dtype in (pl.Int8, pl.Int16, pl.Int32):
        return SqlType.int()

    if dtype == pl.Int64:
        return SqlType.big_int()

    if dtype in (pl.UInt8, pl.UInt16, pl.UInt32):
        return SqlType.big_int()

    if dtype == pl.UInt64:
        return SqlType.numeric(20, 0)

    if dtype in (pl.Float32, pl.Float64):
        return SqlType.double()

    if dtype == pl.Boolean:
        return SqlType.bool()

    if dtype == pl.Date:
        return SqlType.date()

    if isinstance(dtype, pl.Datetime):
        return SqlType.timestamp()

    if dtype == pl.Time:
        return SqlType.time()

    return SqlType.text()


def sanitize_value(value):
    """
    Convert values into forms acceptable to tableauhyperapi.
    Polars usually returns Python-native values here, but this keeps the insert path safer.
    """
    if value is None:
        return None

    return value


def convert_parquet_to_hyper(input_file: Path, output_file: Path):
    if not input_file.exists():
        raise FileNotFoundError(f"Input Parquet file does not exist: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        output_file.unlink()

    print(f"\nScanning {input_file}...")

    lazy_frame = pl.scan_parquet(input_file)

    print("Reading schema...")
    schema = lazy_frame.collect_schema()

    columns = [
        TableDefinition.Column(column_name, polars_dtype_to_hyper_type(dtype))
        for column_name, dtype in schema.items()
    ]

    table_definition = TableDefinition(table_name=table_name, columns=columns)

    print(f"Streaming data to {output_file}...")

    with HyperProcess(
        telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
        parameters={
            "log_config": "",
            "date_style": "MDY",
            "date_style_lenient": "false",
        },
    ) as hyper:
        with Connection(
            endpoint=hyper.endpoint,
            database=str(output_file),
            create_mode=CreateMode.CREATE_AND_REPLACE,
        ) as connection:
            connection.catalog.create_table(table_definition)

            column_names = list(schema.keys())

            for batch_index, chunk in enumerate(lazy_frame.collect_batches(), start=1):
                rows = [
                    [sanitize_value(row[column]) for column in column_names]
                    for row in chunk.iter_rows(named=True)
                ]

                if rows:
                    with Inserter(connection, table_definition) as inserter:
                        inserter.add_rows(rows)
                        inserter.execute()

                print(f"Appended chunk {batch_index} ({chunk.height} rows)")

    print(f"Finished: {input_file.name} -> {output_file.name}")


def main():
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input Parquet folder does not exist: {input_parquet}")

    if not input_parquet.is_dir():
        raise NotADirectoryError(f"input_parquet must be a folder: {input_parquet}")

    output_hyper.mkdir(parents=True, exist_ok=True)

    prepare_hyper_binary()

    parquet_files = sorted(input_parquet.glob("*.parquet"))

    if not parquet_files:
        print(f"No .parquet files found in: {input_parquet}")
        return

    print(f"Found {len(parquet_files)} parquet file(s) in {input_parquet}")

    try:
        for index, parquet_file in enumerate(parquet_files, start=1):
            hyper_file = output_hyper / f"{parquet_file.stem}.hyper"

            print(f"\n[{index}/{len(parquet_files)}] Converting {parquet_file.name}")
            convert_parquet_to_hyper(parquet_file, hyper_file)

        print("\nAll parquet files converted successfully.")

    except RuntimeError as exc:
        print("\nHyper failed to start or write an extract.")
        print("Original error:")
        print(exc)
        print("\nThings to check:")
        print("1. Make sure macOS is not blocking the Hyper executable.")
        print("2. Check whether your Python environment matches your CPU architecture.")
        print("3. Check the Hyper log file mentioned by the error message.")
        print("4. Try reinstalling pantab/tableauhyperapi in a clean virtual environment.")
        raise


if __name__ == "__main__":
    main()