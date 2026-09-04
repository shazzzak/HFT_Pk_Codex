#!/usr/bin/env python3
"""
decile_markout_validation.py -- ADAPTED to the PSX feature-store layout.

Only three things changed from the generic version, all forced by your schema:
  (1) partition layout: feature_store/{symbol}/date={date}.parquet
      (symbol is a PLAIN folder, not symbol=...; date is Hive-style).
  (2) columns: label is markout_5000ms_bps; there is NO ms-to-close column --
      near-close rows already carry NaN markout (the ASOF join returns null when
      no +5s future mid exists), so the NaN-label filter replaces the ms_to_close
      hygiene predicate exactly.
  (3) trade-conditioning column is time_since_trade_ms.
Everything else (day-as-unit SE, per-day edge histogram, checkpointing) is unchanged.
"""

# os for path checks and partition enumeration.
import os
# glob for discovering parquet partitions on disk.
import glob
# duckdb for out-of-core SQL over parquet.
import duckdb
# pandas for the small post-aggregation summaries + plotting frames.
import pandas as pd
# numpy for standard-error arithmetic.
import numpy as np
# matplotlib for the robustness plots.
import matplotlib.pyplot as plt


# =====================================================================
# CONFIG -- the only block you normally edit
# =====================================================================

# Root of the feature store (outside the git project).
DATA_ROOT = "/Users/shazzak/Capital Stake - Results/feature_store"

# Where per-partition summaries and plots are written.
OUT_DIR = "/Users/shazzak/Capital Stake - Results/markout_validation"

# DuckDB memory ceiling (keep below physical RAM).
DUCKDB_MEMORY_LIMIT = "8GB"

# DuckDB spill directory for large sorts/hashes.
DUCKDB_TEMP_DIR = "/Users/shazzak/Capital Stake - Results/duckdb_spill"

# Thread count (modest to bound concurrent buffer memory).
DUCKDB_THREADS = 4

# Logical role -> ACTUAL column name in YOUR store. These match build_feature_store.py.
COLS = {
    "symbol": "symbol",                       # symbol column (also the folder name)
    "date": "date",                           # date column (Hive partition value)
    "markout_5s": "markout_5000ms_bps",       # the 5s label, in bps (your name)
    "ms_since_trade": "time_since_trade_ms",  # ms since last trade (your name)
    "obi_1": "obi_1",                         # feature 1
    "micro_dev_bps": "micro_dev_bps",         # feature 2
}
# NOTE: no 'ms_to_close' entry -- your store has no such column; NaN-label filter covers it.

# Number of quantile buckets.
N_DECILES = 10

# Trade-conditioning cutoff (ms) for micro_dev_bps (the earlier finding).
TRADE_COND_MS = 100

# Features to run, and whether each is trade-conditioned.
FEATURE_SPECS = [
    {"name": "obi_1",         "trade_conditioned": False},  # OBI: full sample
    {"name": "micro_dev_bps", "trade_conditioned": True},   # micro_dev: <100ms since trade
]


# =====================================================================
# STEP 0 -- schema check (run manually once)
# =====================================================================

def describe_schema():
    # Grab any one parquet under the store to introspect columns.
    files = glob.glob(os.path.join(DATA_ROOT, "*", "*.parquet"))
    # Guard: wrong path if nothing found.
    if not files:
        raise FileNotFoundError(f"No parquet under {DATA_ROOT}/*/*.parquet")
    # Throwaway connection just for DESCRIBE.
    con = duckdb.connect()
    # Print columns/types so COLS can be verified.
    print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{files[0]}')").df().to_string())
    # Close.
    con.close()


# =====================================================================
# STEP 1 -- enumerate (symbol, date) partitions WITHOUT scanning rows
# =====================================================================

def enumerate_partitions():
    # YOUR layout: feature_store/{symbol}/date={date}.parquet
    # symbol is a plain folder; each date is its own parquet FILE (not a folder).
    pattern = os.path.join(DATA_ROOT, "*", "date=*.parquet")
    # List matching parquet files.
    part_files = sorted(glob.glob(pattern))
    # Container for (symbol, date, files_glob) tuples.
    partitions = []
    # Walk each parquet file.
    for f in part_files:
        # symbol = the parent folder name (plain, e.g. "PPL").
        sym = os.path.basename(os.path.dirname(f))
        # date = strip "date=" prefix and ".parquet" suffix from the filename.
        base = os.path.basename(f)                      # "date=2026-06-30.parquet"
        dt = base[len("date="):-len(".parquet")]        # "2026-06-30"
        # The single file IS the partition's data.
        partitions.append((sym, dt, f))
    # Expected: 2 symbols x 207 days = 414 partitions.
    return partitions


# =====================================================================
# STEP 2 -- per-partition aggregation query (bounded memory)
# =====================================================================

def build_partition_sql(files_glob, feature_col, trade_conditioned):
    # Hygiene: valid label only. This SINGLE predicate replaces the generic
    # version's (markout NOT NULL) AND (ms_to_close >= window): in your store,
    # rows whose +5s window would cross the close ALREADY have NaN markout, so
    # filtering non-null labels excludes exactly those boundary rows.
    where = [f"{COLS['markout_5s']} IS NOT NULL"]
    # Add trade-conditioning for features that need it (micro_dev_bps).
    if trade_conditioned:
        where.append(f"{COLS['ms_since_trade']} < {TRADE_COND_MS}")
    # Join predicates.
    where_sql = " AND ".join(where)
    # One partition per query -> NTILE needs no PARTITION BY (exact within-day deciles).
    return f"""
    WITH src AS (
        SELECT
            {COLS['markout_5s']} AS mk,               -- the label (bps)
            {feature_col}        AS feat              -- the feature under test
        FROM read_parquet('{files_glob}')
        WHERE {where_sql}                             -- non-null label (+ optional trade cond)
    ),
    binned AS (
        SELECT
            mk,
            NTILE({N_DECILES}) OVER (ORDER BY feat) AS decile   -- within-day deciles
        FROM src
    )
    SELECT
        decile,
        AVG(mk)                                     AS mean_markout_all,      -- zeros included
        AVG(CASE WHEN mk <> 0 THEN mk END)          AS mean_markout_nonzero,  -- mid-moved only
        COUNT(*)                                    AS n_rows,                -- bucket size
        AVG(CASE WHEN mk = 0 THEN 1.0 ELSE 0.0 END) AS frac_zero              -- zero-mass share
    FROM binned
    GROUP BY decile
    ORDER BY decile
    """


# =====================================================================
# STEP 3 -- partition loop with per-feature checkpointing
# =====================================================================

def run_extract(feature_spec):
    # Feature name + conditioning flag.
    fname, tcond = feature_spec["name"], feature_spec["trade_conditioned"]
    # Resolve to the real column.
    fcol = COLS[fname]
    # Checkpoint/summary path for this feature.
    out_path = os.path.join(OUT_DIR, f"{fname}_daily_decile_summary.parquet")
    # Ensure output + spill dirs exist.
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
    # Resume from an existing checkpoint if present.
    if os.path.exists(out_path):
        done = pd.read_parquet(out_path)
        done_keys = set(zip(done["symbol"], done["date"]))
        acc = [done]
    else:
        done_keys = set()
        acc = []
    # One reused DuckDB connection.
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory='{DUCKDB_TEMP_DIR}'")
    con.execute(f"SET threads TO {DUCKDB_THREADS}")
    # All partitions.
    partitions = enumerate_partitions()
    # Loop with progress index.
    for i, (sym, dt, files_glob) in enumerate(partitions, start=1):
        # Skip completed partitions.
        if (sym, dt) in done_keys:
            continue
        # Build + run this partition's aggregation.
        part_df = con.execute(build_partition_sql(files_glob, fcol, tcond)).df()
        # Stamp identity.
        part_df["symbol"] = sym
        part_df["date"] = dt
        # Collect.
        acc.append(part_df)
        # Checkpoint every 25 partitions.
        if i % 25 == 0:
            pd.concat(acc, ignore_index=True).to_parquet(out_path)
            print(f"[{fname}] checkpointed after {i}/{len(partitions)} partitions")
    # Final write.
    pd.concat(acc, ignore_index=True).to_parquet(out_path)
    con.close()
    print(f"[{fname}] wrote {out_path}")
    return out_path


# =====================================================================
# STEP 4 -- day-as-unit aggregation (honest error bar)
# =====================================================================

def aggregate_day_as_unit(summary_path, value_col="mean_markout_all"):
    # Load the tiny per-(symbol,date,decile) summary.
    df = pd.read_parquet(summary_path)
    # Per (symbol, decile): mean/std/count OF THE DAILY MEANS.
    agg = (
        df.groupby(["symbol", "decile"])[value_col]
          .agg(mean_of_daily="mean", std_of_daily="std", n_days="count")
          .reset_index()
    )
    # SE = day-to-day std / sqrt(n_days) -- NOT std/sqrt(n_rows) (rows autocorrelate).
    agg["se"] = agg["std_of_daily"] / np.sqrt(agg["n_days"])
    # 95% normal-approx band.
    agg["ci95"] = 1.96 * agg["se"]
    # Per-(symbol,date) decile table for the edge scalar.
    wide = df.pivot_table(index=["symbol", "date"], columns="decile", values=value_col)
    # Edge = top-decile minus bottom-decile mean markout (expected > 0).
    wide["edge"] = wide[N_DECILES] - wide[1]
    # Return curve aggregation + per-day edges.
    return df, agg, wide.reset_index()


# =====================================================================
# STEP 5 -- robustness plots (curve+band+spaghetti, edge histogram)
# =====================================================================

def plot_feature(feature_name, df, agg, wide, value_col="mean_markout_all"):
    # One figure per symbol.
    for sym in sorted(df["symbol"].unique()):
        # Left = decile curve; right = daily-edge distribution.
        fig, (ax_c, ax_h) = plt.subplots(1, 2, figsize=(15, 5))
        # Faint per-day curves (spaghetti) to expose sign-flips.
        for _, day_df in df[df.symbol == sym].groupby("date"):
            ax_c.plot(day_df["decile"], day_df[value_col],
                      color="steelblue", alpha=0.04, linewidth=0.6)
        # Bold mean-of-daily-means curve.
        a = agg[agg.symbol == sym].sort_values("decile")
        ax_c.plot(a["decile"], a["mean_of_daily"], color="black",
                  marker="o", linewidth=2, label="mean of daily means")
        # Honest 95% band from day-to-day SE.
        ax_c.fill_between(a["decile"], a["mean_of_daily"] - a["ci95"],
                          a["mean_of_daily"] + a["ci95"],
                          color="orange", alpha=0.35, label="95% CI (day-as-unit)")
        # Zero reference.
        ax_c.axhline(0, color="red", linestyle="--", linewidth=1)

        # --- NEW: Dynamic Y-Axis Limiting ---
        # Calculate the 2nd and 98th percentiles of the daily spaghetti lines
        # This prevents a single extreme outlier day from squashing the visual
        y_data = df.loc[df.symbol == sym, value_col].dropna()
        if not y_data.empty:
            y_min, y_max = np.percentile(y_data, [20, 80])
            # Add a 20% padding margin above and below the core distribution
            pad = (y_max - y_min) * 0.2
            # Force the y-axis bounds, letting extreme days extend off-canvas
            ax_c.set_ylim(y_min - pad, y_max + pad)

        # Labels/title.
        ax_c.set_xlabel(f"{feature_name} decile (within-day rank)")
        ax_c.set_ylabel("mean markout_5s (bps)")
        ax_c.set_title(f"{sym}: {feature_name} decile curve  (n_days={int(a['n_days'].iloc[0])})")
        ax_c.legend()
        # Right: histogram of per-day top-minus-bottom edge.
        edges = wide.loc[wide.symbol == sym, "edge"].dropna()
        ax_h.hist(edges, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
        # Zero line.
        ax_h.axvline(0, color="red", linestyle="--", linewidth=1.5)
        # Sign-agreement fraction.
        frac_pos = float((edges > 0).mean()) * 100.0
        ax_h.set_title(f"{sym}: daily top-bottom edge  (positive on {frac_pos:.0f}% of days)")
        ax_h.set_xlabel("top minus bottom decile markout_5s (bps), per day")
        ax_h.set_ylabel("day count")
        # Layout + save.
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f"{feature_name}_robustness_{sym}.png")
        plt.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"[{feature_name}] {sym}: saved {out_png}  (edge>0 on {frac_pos:.0f}% of days)")


# =====================================================================
# MAIN
# =====================================================================

def main():
    # Each feature end to end.
    for spec in FEATURE_SPECS:
        # Step 3: per-day decile summaries (checkpointed).
        summary_path = run_extract(spec)
        # Step 4+5: zeros-included label.
        df, agg, wide = aggregate_day_as_unit(summary_path, value_col="mean_markout_all")
        plot_feature(spec["name"], df, agg, wide, value_col="mean_markout_all")
        # Step 4+5: zeros-excluded (edge conditional on the mid moving).
        df_nz, agg_nz, wide_nz = aggregate_day_as_unit(summary_path, value_col="mean_markout_nonzero")
        plot_feature(spec["name"] + "_nonzero", df_nz, agg_nz, wide_nz, value_col="mean_markout_nonzero")


# Entry point.
if __name__ == "__main__":
    # Uncomment to verify columns before the first full run:
    # describe_schema()
    main()
