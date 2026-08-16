# probe_fills.py -- show what's actually in each fills folder + the parquet schema.
# Run in PyCharm (uses the .backtest interpreter, which has pyarrow).

import os
import glob
import pyarrow.parquet as pq

# the two candidate fills roots
ROOTS = {
    "fills": "/Users/shazzak/Capital Stake - Results/fills",
    "fill_attribution/fills": "/Users/shazzak/Capital Stake - Results/fill_attribution/fills",
}

# walk each root
for name, root in ROOTS.items():
    # header per folder
    print(f"\n=== {name} ===")
    # missing folder?
    if not os.path.isdir(root):
        print("  (folder does not exist)")
        continue
    # show the immediate contents so we see subfolder structure (micro/naive/etc.)
    print("  top-level:", sorted(os.listdir(root)))
    # find parquet anywhere below this root (any nesting depth)
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    # report how many and a few example relative paths
    print(f"  parquet files found (recursive): {len(files)}")
    for f in files[:3]:
        print("   ", os.path.relpath(f, root))
    # print the column schema of the first parquet, if any
    if files:
        print("  columns:", pq.ParquetFile(files[0]).schema_arrow.names)