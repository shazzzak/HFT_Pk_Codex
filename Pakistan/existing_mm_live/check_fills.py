# check_fills.py
# Reconciliation preview for the persisted real fills, BEFORE running attribution.
# Confirms persist_fills.py produced 4 (strategy x symbol) fill sets and that the
# fill counts match the known backtest numbers -- the reconciliation gate, run cheap.
# Run from anywhere:  python check_fills.py

# duckdb for fast parquet row counts.
import duckdb
# pathlib for listing the persisted fill folders.
from pathlib import Path

# Root of the persisted fills (outside the git project).
FILLS_ROOT = Path("/Users/shazzak/Capital Stake - Results/fills")

# Known backtest fill counts (207 days) for the reconciliation check.
# From the earlier full-period runs: attribution counts should match these.
EXPECTED = {
    ("naive", "PPL"): 93286,
    ("micro", "PPL"): 166328,
    ("naive", "UBL"): 174820,
    ("micro", "UBL"): 83968,
}

# --- 1. list the persisted fill folders (expect 4: naive/micro x PPL/UBL) ---
# Header.
print("=== persisted fill folders (expect naive/micro x PPL/UBL) ===")
# Glob every {strategy}/{symbol} folder under fills/.
folders = sorted(p for p in FILLS_ROOT.glob("*/*") if p.is_dir())
# Print each folder relative to the fills root.
for f in folders:
    # Show strategy/symbol.
    print(f"  {f.relative_to(FILLS_ROOT)}")
# If none found, the persist step didn't write where expected.
if not folders:
    print("  (none found -- check persist_fills.py output path / that it ran)")

# --- 2. fill counts per (strategy, symbol) vs expected backtest counts ---
# Header.
print("\n=== fill counts vs expected backtest numbers (reconciliation) ===")
# Column header.
print(f"{'strategy':6} {'sym':4} {'fills':>10} {'days':>5} {'expected':>10} {'match?':>8}")
# Loop both strategies.
for strat in ["naive", "micro"]:
    # Loop both pilot symbols.
    for sym in ["PPL", "UBL"]:
        # The parquet glob for this (strategy, symbol).
        glob = str(FILLS_ROOT / strat / sym / "date=*.parquet")
        # Guard each read so one missing set doesn't abort the whole check.
        try:
            # Total fill rows.
            n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()[0]
            # Distinct days covered.
            d = duckdb.sql(f"SELECT count(DISTINCT date) FROM read_parquet('{glob}')").fetchone()[0]
            # Expected count for this cell.
            exp = EXPECTED.get((strat, sym), None)
            # Match flag: within 2% of expected (context join should drop ~nothing).
            if exp:
                # Relative difference.
                rel = abs(n - exp) / exp
                # Tick if close, cross if far.
                flag = "OK" if rel < 0.02 else f"OFF {rel:.0%}"
            else:
                # No expectation on record.
                flag = "-"
            # Print the row.
            print(f"{strat:6} {sym:4} {n:>10,} {d:>5} {str(exp):>10} {flag:>8}")
        except Exception as e:
            # Report a read failure for this cell.
            print(f"{strat:6} {sym:4}  ERROR: {e}")

# --- 3. quick schema peek so attribution's expected columns are present ---
# Header.
print("\n=== columns in one fill parquet (attribution needs these) ===")
# Try the naive/PPL set as the sample.
sample = str(FILLS_ROOT / "naive" / "PPL" / "date=*.parquet")
# Guard the read.
try:
    # DESCRIBE the schema.
    cols = duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{sample}')").df()
    # Print column names only (types omitted for brevity).
    print("  " + ", ".join(cols["column_name"].tolist()))
    # Flag the must-have columns for attribution.
    need = ["t", "side", "px", "qty", "reason", "mid0", "mid_h",
            "spread_bps", "obi_1", "toxicity", "realized_vol_bps", "strategy", "symbol", "date"]
    # Which are missing?
    have = set(cols["column_name"].tolist())
    # Compute the gap.
    missing = [c for c in need if c not in have]
    # Report.
    print(f"  MISSING (attribution will break on these): {missing if missing else 'none'}")
except Exception as e:
    # Report a read failure.
    print(f"  ERROR reading sample schema: {e}")
