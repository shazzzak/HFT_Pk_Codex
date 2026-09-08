import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# ===== CONFIGURATION =====
ZIP_PATHS = [
    Path("C:/Users/shahzeb/Desktop/Del/PP_GUNICIEMIR.M.202607.zip"),
    Path("C:/Users/shahzeb/Desktop/Del/PP_GUNICIISLEM.M.202607.zip")
]

EXTRACT_ONE_DAY = True
CHUNK_SIZE = 50000
TARGET_DATE = "20260701"
NUM_WORKERS = 4
OUTPUT_DIR = Path("C:/Users/shahzeb/Desktop/Del/Output")
# =========================

stop_processing = False


def signal_handler(sig, frame):
    global stop_processing
    if not stop_processing:
        print("\n\n⚠️  Stop signal received. Finishing current chunk and exiting...")
        stop_processing = True
    else:
        print("\n🛑 Force stopping...")
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


def detect_date_column(df_sample):
    """Detect columns with date-like names and values."""
    date_indicators = [
        'date', 'time', 'transact', 'timestamp', 'created', 'updated', 'datetime', 'dt', 'trade', 'settle',
        'tarih', 'zaman', 'islem', 'giris', 'degistirilme', 'guncelleme', 'takas', 'saat'
    ]
    candidates = []
    for col in df_sample.columns:
        col_lower = col.lower()
        if any(ind in col_lower for ind in date_indicators):
            sample_vals = df_sample[col].head(10).to_list()
            date_like = sum(1 for v in sample_vals if v is not None and isinstance(v, str) and
                            any(sep in v for sep in ['/', '-', ':', '.']) and any(c.isdigit() for c in v))
            if date_like > 0:
                candidates.append((col, date_like))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def determine_date_format(df, date_col, target_date):
    """
    Detect the date format from sample values, then return a filter expression
    that compares the parsed date to the target date.
    Invalid values (e.g., "DATE") become null and are ignored.
    """
    # Get non-null, non-empty sample values, but skip the literal "DATE"
    sample = df.select(pl.col(date_col)).filter(
        (pl.col(date_col).is_not_null()) & (pl.col(date_col) != "") & (pl.col(date_col) != "DATE")
    ).head(1000)

    if len(sample) == 0:
        return None, "No valid data"

    vals = sample[date_col].to_list()
    vals_str = [str(v) for v in vals if v is not None]

    # Try common datetime formats
    formats_to_try = [
        ("%Y-%m-%d", "YYYY-MM-DD"),
        ("%Y/%m/%d", "YYYY/MM/DD"),
        ("%d-%m-%Y", "DD-MM-YYYY"),
        ("%d/%m/%Y", "DD/MM/YYYY"),
        ("%d.%m.%Y", "DD.MM.YYYY"),
        ("%Y%m%d", "YYYYMMDD"),
    ]

    best_format = None
    best_desc = None
    best_count = 0

    for fmt, desc in formats_to_try:
        parsed = []
        for v in vals_str:
            try:
                dt = datetime.strptime(v, fmt)
                parsed.append(dt)
            except:
                pass
        if len(parsed) > best_count:
            best_count = len(parsed)
            best_format = fmt
            best_desc = desc
            if best_count / len(vals_str) > 0.8:
                break

    if best_format is None:
        # Fallback: try string contains of target date in various formats
        fallback_formats = [
            target_date,
            f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}",
            f"{target_date[:4]}/{target_date[4:6]}/{target_date[6:8]}",
            f"{target_date[6:8]}-{target_date[4:6]}-{target_date[:4]}",
            f"{target_date[6:8]}/{target_date[4:6]}/{target_date[:4]}",
        ]
        for pattern in fallback_formats:
            # Check if any sample value contains the pattern
            if any(pattern in v for v in vals_str):
                expr = pl.col(date_col).cast(pl.Utf8).str.contains(pattern)
                return expr, f"fallback contains '{pattern}'"
        return None, "No format detected"

    # Build filter: convert to date using the detected format, with strict=False
    expr = pl.col(date_col).cast(pl.Utf8).str.strptime(pl.Datetime, best_format, strict=False)
    target_dt = datetime.strptime(target_date, "%Y%m%d")
    filter_expr = expr.dt.date() == target_dt.date()

    return filter_expr, f"parsed with {best_desc}"


def append_to_parquet(df, out_path):
    if len(df) == 0:
        return
    arrow_table = df.to_arrow()
    if out_path.exists():
        try:
            existing = pq.read_table(out_path)
            combined = pa.concat_tables([existing, arrow_table])
            pq.write_table(combined, out_path)
        except:
            pq.write_table(arrow_table, out_path)
    else:
        pq.write_table(arrow_table, out_path)


def process_single_zip(zip_path: Path, target_date: str = None, chunk_size: int = 50000, worker_id: int = 0):
    print(f"\n{'='*80}")
    print(f"🔧 Worker {worker_id}: Processing {zip_path.name}")
    print(f"{'='*80}")

    if not zip_path.exists():
        print(f"❌ Worker {worker_id}: File not found: {zip_path}")
        return None

    try:
        out_parquet = OUTPUT_DIR / f"{zip_path.stem}_extracted.parquet"
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        if out_parquet.exists():
            out_parquet.unlink()
            print("   Removed existing output file")

        with zipfile.ZipFile(zip_path, "r") as z:
            file_members = [m for m in z.infolist() if not m.is_dir()]
            if not file_members:
                print("❌ No files in zip")
                return None
            member_name = file_members[0].filename
            print(f"📄 Using file: {member_name}")

            with z.open(member_name) as f:
                # Detect delimiter
                sample_bytes = f.read(8192)
                f.seek(0)
                sample_text = sample_bytes.decode('utf-8', errors='ignore')
                if '\x01' in sample_text:
                    delimiter = '\x01'
                elif '|' in sample_text:
                    delimiter = '|'
                else:
                    if ';' in sample_text and sample_text.count(';') > sample_text.count(','):
                        delimiter = ';'
                    elif '\t' in sample_text:
                        delimiter = '\t'
                    else:
                        delimiter = ','
                print(f"   Delimiter: {repr(delimiter)}")

                # Read header
                header_line = f.readline().decode('utf-8', errors='ignore').strip()
                f.seek(0)
                headers = header_line.split(delimiter)
                print(f"   Found {len(headers)} columns")

                # Skip header
                f.readline()

                # Determine date column and filter expression from first chunk
                date_column = None
                filter_expr = None
                date_format_desc = "Not determined"

                first_chunk_data = []
                for _ in range(min(chunk_size, 10000)):  # read a sample for detection
                    line_bytes = f.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    values = line.split(delimiter)
                    if len(values) >= len(headers):
                        values = values[:len(headers)]
                        row_dict = dict(zip(headers, values))
                        first_chunk_data.append(row_dict)
                    elif len(values) > 0:
                        padded = values + [''] * (len(headers) - len(values))
                        row_dict = dict(zip(headers, padded))
                        first_chunk_data.append(row_dict)

                if first_chunk_data:
                    sample_df = pl.DataFrame(first_chunk_data)
                    candidates = detect_date_column(sample_df)
                    if candidates:
                        date_column = candidates[0][0]
                        print(f"   ✅ Selected '{date_column}' for date filtering")
                        filter_expr, date_format_desc = determine_date_format(sample_df, date_column, target_date)
                        print(f"   📅 Using format: {date_format_desc}")
                    else:
                        print("   ⚠️  No date column detected, will process all rows")
                else:
                    print("   ⚠️  No data in first chunk")
                    return None

                # Reset file pointer and skip header
                f.seek(0)
                f.readline()

                # Process remaining data in chunks
                print(f"   Processing CSV rows...")
                start_time = time.time()
                chunk_data = []
                processed = 0
                matched_count = 0
                dropped_count = 0

                def process_chunk(chunk_list):
                    nonlocal matched_count, dropped_count
                    if not chunk_list:
                        return
                    df_chunk = pl.DataFrame(chunk_list)
                    if filter_expr is not None and date_column in df_chunk.columns:
                        filtered, dropped = filter_by_date_expr(df_chunk, date_column, filter_expr)
                        dropped_count += dropped
                    else:
                        filtered = df_chunk
                    if len(filtered) > 0:
                        matched_count += len(filtered)
                        append_to_parquet(filtered, out_parquet)

                for line_bytes in f:
                    if stop_processing:
                        print(f"\n🛑 Worker {worker_id}: Stopping...")
                        break
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    values = line.split(delimiter)
                    if len(values) >= len(headers):
                        values = values[:len(headers)]
                        row_dict = dict(zip(headers, values))
                        chunk_data.append(row_dict)
                        processed += 1
                    elif len(values) > 0:
                        padded = values + [''] * (len(headers) - len(values))
                        row_dict = dict(zip(headers, padded))
                        chunk_data.append(row_dict)
                        processed += 1

                    if len(chunk_data) >= chunk_size:
                        process_chunk(chunk_data)
                        chunk_data = []
                        if processed % (chunk_size * 10) == 0:
                            elapsed = time.time() - start_time
                            rate = processed / elapsed if elapsed > 0 else 0
                            match_pct = (matched_count / processed * 100) if processed > 0 else 0
                            # Print with newline so workers don't overwrite each other
                            print(
                                f"   Worker {worker_id}: ⏳ Processed: {processed:,} | Matched: {matched_count:,} ({match_pct:.1f}%) | Dropped: {dropped_count:,} | Rate: {rate:.0f}/s")

                # Final chunk
                if chunk_data and not stop_processing:
                    process_chunk(chunk_data)

                elapsed = time.time() - start_time
                print(f"   Worker {worker_id}: ✅ Completed in {elapsed:.2f}s")
                print(f"   Processed: {processed:,} | Matched: {matched_count:,} | Dropped: {dropped_count:,}")

                if out_parquet.exists():
                    df = pl.read_parquet(out_parquet)
                    print(f"   Output: {len(df):,} rows, {len(df.columns)} columns")
                    return {
                        'file': zip_path.name,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'output': str(out_parquet),
                        'df': df
                    }
                else:
                    print("   ⚠️  No data matched the date filter")
                    return None

    except Exception as e:
        print(f"❌ Worker {worker_id}: Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def filter_by_date_expr(df, date_col, filter_expr):
    if filter_expr is None or date_col not in df.columns:
        return df, 0
    original_len = len(df)
    filtered = df.filter(filter_expr)
    dropped = original_len - len(filtered)
    return filtered, dropped


def merge_parquet_files(output_dir: Path, target_date: str):
    print("\n" + "=" * 80)
    print("🔀 Merging all Parquet files...")
    print("=" * 80)
    parquet_files = list(output_dir.glob("*_extracted.parquet"))
    if not parquet_files:
        print("⚠️  No Parquet files found")
        return None
    print(f"Found {len(parquet_files)} files")
    dfs = []
    total_rows = 0
    for i, f in enumerate(parquet_files, 1):
        try:
            df = pl.read_parquet(f)
            rows = len(df)
            total_rows += rows
            dfs.append(df)
            print(f"   {i}. {f.name}: {rows:,} rows")
        except Exception as e:
            print(f"   ❌ Error reading {f.name}: {e}")
    if not dfs:
        return None
    print(f"\n🔄 Concatenating {len(dfs)} DataFrames...")
    combined = pl.concat(dfs)
    merged_path = output_dir / f"merged_{target_date}_all_data.parquet"
    combined.write_parquet(merged_path)
    size_mb = merged_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Merged: {merged_path}")
    print(f"   Total rows: {total_rows:,}")
    print(f"   File size: {size_mb:.2f} MB")
    return combined


def main():
    print("=" * 80)
    print("📊 DATA EXTRACTION TOOL (Optimized)")
    print("=" * 80)
    print(f"📁 Files: {len(ZIP_PATHS)}")
    for p in ZIP_PATHS:
        print(f"   - {p.name}")
    print(f"📅 Target Date: {TARGET_DATE}")
    print(f"📏 Chunk Size: {CHUNK_SIZE:,}")
    print(f"🔧 Workers: {NUM_WORKERS}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    valid_paths = [p for p in ZIP_PATHS if p.exists()]
    if not valid_paths:
        print("\n❌ No valid files found")
        return

    all_results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i, path in enumerate(valid_paths):
            future = executor.submit(
                process_single_zip,
                path,
                TARGET_DATE if EXTRACT_ONE_DAY else None,
                CHUNK_SIZE,
                i
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"❌ Worker failed: {e}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"⏱️  Time: {elapsed:.2f}s")
    total_rows = 0
    for r in all_results:
        print(f"\n   📄 {r['file']}")
        print(f"      Rows: {r['rows']:,}")
        print(f"      Columns: {r['columns']}")
        total_rows += r['rows']
    print(f"\n📊 Total rows: {total_rows:,}")

    if len(all_results) > 1:
        merge_parquet_files(OUTPUT_DIR, TARGET_DATE)
    elif len(all_results) == 1:
        print(f"\n✅ Output: {all_results[0]['output']}")

    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()