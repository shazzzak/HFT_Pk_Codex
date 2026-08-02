import pyarrow.dataset as ds
from pathlib import Path

PARSED = Path("/Users/shazzak/Capital Stake - Parsed")
OUT_DIR = Path("/Users/shazzak/Capital Stake - Extracts")   # keep extracts out of the canonical store
DATE = "2025-09-15"          # pick a representative (not screen-topping) day

SYMBOLS = ["KTML", "FFC", "SYS", "BAFL", "PSO", "HBL",
           "KOHC", "PKGS", "ITANZ", "AKBL", "NBP", "SGPL"]

TABLES = [("trades", "trades"), ("ob_snapshot", "Ob_snapshot"),
          ("ob_updates", "Ob_updates")]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Open each day-partition ONCE, not once per symbol: 12 symbols x 3 tables would
# otherwise re-open the same three datasets 36 times.
datasets = {table: ds.dataset(PARSED / table / f"date={DATE}", format="parquet")
            for table, _ in TABLES}

missing = []
for symbol in SYMBOLS:
    counts = {}
    for table, tag in TABLES:
        df = (datasets[table]
              .to_table(filter=ds.field("symbol") == symbol)
              .to_pandas())
        counts[tag] = len(df)
        # Skip empty extracts -- an empty CSV is worse than no file (it looks
        # like data until something downstream divides by zero).
        if len(df) == 0:
            continue
        out = OUT_DIR / f"{symbol}_{tag}_{DATE}.csv"
        df.to_csv(out)
    if counts["trades"] == 0:
        missing.append(symbol)
        print(f"{symbol:<6} NO TRADES on {DATE} -- skipped")
    else:
        print(f"{symbol:<6} trades {counts['trades']:>7,} | "
              f"snap {counts['Ob_snapshot']:>8,} | upd {counts['Ob_updates']:>7,}")

print(f"\nWrote to {OUT_DIR}")
if missing:
    print(f"No data for: {', '.join(missing)} — check the ticker spelling or "
          f"whether they traded on {DATE}")