"""
Stage 2 -- Persistence analysis over daily_stats_ALL.parquet.

Turns 207 single-day snapshots into a watchlist. A single day is not rankable:
KTML swung 236x in ceiling between two sessions, so anything ranked on one date
is noise. This script ranks on PERSISTENCE instead.

Four corrections are built in, each from a finding earlier in the project:

  1. MATERIALITY GATE. pct_days_top20 as originally specified is a RANK
     threshold, so it ignores magnitude: an episodic name worth ~1,300 PKR still
     outranks every dead name and scores 1.000, identical to a stable name worth
     100,000. Verified on planted archetypes: rank-only scored stable and
     episodic BOTH at 1.000; adding the money gate separated them to 1.000 vs
     0.085 (recovering the 8% spike rate that was planted). Top-N AND >= money.

  2. DENOMINATOR RE-BASED ON DAYS TRADED. Unscreenable days (limit-locked,
     one-sided book, auction-only) VANISH from daily_stats rather than counting
     as zero -- DMC 2025-09-23 was bid-only at the +10% cap all session. Using
     days-present as the denominator therefore flatters exactly the thin,
     volatile names most likely to be locked. Re-based on days actually traded.

  3. SESSION REGIME. PSX ran shortened sessions in Ramadan 2026 (Feb 19-Mar 19):
     ~4.2h weekdays, ~3.2h Fridays, vs ~6.0h normal; normal Fridays have a
     ~152-minute Jumu'ah break (7.22h elapsed, 4.68h traded). A daily ceiling is
     a TOTAL, so pooling regimes inflates cross-day dispersion for reasons
     unrelated to episodic-vs-stable. Within-day metrics (top-N, rank
     autocorrelation) are immune -- they compare symbols on the SAME date.
     Cross-day level metrics are not, so they use LEVEL_REGIMES only.

     Deliberately NOT normalising per hour: opportunity is not linear in session
     length (open and close carry disproportionate volume and spread), and
     measured Friday intensity is ~24% HIGHER per traded hour. Tag, do not scale.

  4. SEGMENT SPLIT. Ready equities (REG) and single-stock futures
     (STOCK_DEL_FUT / STOCK_CS_FUT) are different instruments; ranking them on
     one leaderboard is meaningless. Uses the segment column if daily_stats
     carries it, else joins it from the trades store.

Outputs:
    session_calendar.parquet        one row per date: open/close, traded hours, regime
    daily_stats_enriched.parquet    daily_stats + regime + traded_h + day_idx
    persistence_<seg>_<fee>.csv     per-symbol persistence metrics
    rank_autocorr_<seg>_<fee>.csv   leaderboard stability at lag N

Run:
    python persistence_metrics.py
"""
# Numerical helpers: where/nan handling and the day index.
import numpy as np
# DataFrame library: all metric computation happens here.
import pandas as pd
# Filesystem checks for the optional skips table.
import os

# ------------------------------- CONFIG -------------------------------------
# Root of the parsed store, used for the session calendar and days-traded map.
PARSED_ROOT = ("/Users/shazzak/Library/CloudStorage/"
               "GoogleDrive-shazzak@gmail.com/My Drive/Capital Stake - Parsed")
# The merged Stage-1 output (should now carry a `segment` column natively).
DAILY_STATS = "daily_stats_ALL.parquet"
# Optional: merged skip records, if run_daily_stats.py was run with skip tracking.
SKIPS_FILE = "skips_ALL.parquet"

# Ramadan 1447 window, taken from MEASURED session times, not a calendar guess.
RAMADAN_START, RAMADAN_END = "2026-02-19", "2026-03-19"

# Which fee scenarios to produce. rt_2p00 is the operative TREC case (~1.6bps RT);
# rt_35p45 is the current retail schedule. Ranking INVERTS between them.
FEE_TAGS = ["2p00", "35p45"]

# Regimes whose daily totals are comparable enough to pool for LEVEL metrics
# (median / IQR). Ramadan is excluded: same intensity, genuinely shorter session.
# Fridays are INCLUDED by default because measured intensity is ~24% higher per
# traded hour, so a 4.68h Friday is much closer to a 5.96h normal day than the
# hour count suggests -- excluding them would discard ~37 days for little bias.
# main() prints a paired friday/normal ratio; set this to ("normal",) if that
# ratio comes back near the time ratio (~0.79) rather than near 1.0.
LEVEL_REGIMES = ("normal", "friday")

# Top-N membership threshold for the persistence metric.
TOP_N = 20

# Economic floor (PKR/day of pairing-adjusted ceiling) a symbol-day must clear to
# count as materially attractive -- PER FEE, because the ceiling scales with the
# fee. The two scenarios live on different scales: REG normal-day median ceiling
# is 6,641 at 2bps but 2,149 at 35bps (p90 82,575 vs 10,365). One constant would
# starve one scenario or water down the other. Each value sits ~p75-p85 of its
# own distribution, which is what "the same economic floor" means once the scales
# differ. Anchor: the daily gross below which quoting is not worth the risk,
# deflated by ~10-25% realistic capture of this upper-bound ceiling.
MATERIAL_PKR = {"2p00": 30_000, "35p45": 5_000}

# Lag, in trading days, for the leaderboard rank autocorrelation.
AUTOCORR_LAG = 5

# Minimum days TRADED before a symbol is scored (avoids one-day wonders).
MIN_DAYS = 100


# --------------------------- session calendar -------------------------------
# Build one row per date describing that session's shape and regime.
def build_session_calendar(con):
    """One row per date: open, close, elapsed hours, largest gap, traded hours,
    day of week, regime, and a trading-day index."""
    # max_gap matters because a mid-session break (normal Fridays) inflates
    # elapsed time; traded_h subtracts the single largest gap.
    q = f"""
        WITH t AS (
            SELECT date, transact_time AS ts
            FROM read_parquet('{PARSED_ROOT}/trades/*/*.parquet', hive_partitioning=true)
            WHERE initiator <> 'AUCTION'
        ),
        g AS (
            SELECT date, ts,
                   date_diff('second', lag(ts) OVER (PARTITION BY date ORDER BY ts), ts) AS gap_s
            FROM t
        )
        SELECT date,
               min(ts) AS open_ts,
               max(ts) AS close_ts,
               date_diff('second', min(ts), max(ts)) / 3600.0 AS elapsed_h,
               coalesce(max(gap_s), 0) / 60.0                 AS max_gap_min,
               (date_diff('second', min(ts), max(ts))
                  - coalesce(max(gap_s), 0)) / 3600.0         AS traded_h,
               count(*)                                       AS trades
        FROM g
        GROUP BY date
        ORDER BY date
    """
    # Execute and bring the (tiny, ~207 row) result into pandas.
    cal = con.sql(q).df()
    # Same hive date auto-cast issue as above: normalise to YYYY-MM-DD strings so
    # the merge against daily_stats (which stores date as text) has matching keys.
    cal["date"] = cal["date"].astype(str).str.slice(0, 10)
    # Day name from the partition date string, for the Friday split.
    cal["dow"] = pd.to_datetime(cal["date"]).dt.day_name()
    # Ramadan flag from the measured window.
    in_ram = (cal["date"] >= RAMADAN_START) & (cal["date"] <= RAMADAN_END)
    # Four regimes: the Friday/non-Friday split exists inside and outside Ramadan.
    cal["regime"] = np.where(
        # Ramadan Fridays close earliest (~3.2h, no break).
        in_ram & (cal["dow"] == "Friday"), "ramadan_friday",
        # Ramadan weekdays (~4.2h, no break).
        np.where(in_ram, "ramadan",
                 # Normal Fridays (~7.2h elapsed, ~4.7h traded, Jumu'ah break).
                 np.where(cal["dow"] == "Friday", "friday", "normal")))
    # Sort chronologically so the day index is meaningful.
    cal = cal.sort_values("date").reset_index(drop=True)
    # Trading-day index, used for the lag-N rank autocorrelation.
    cal["day_idx"] = np.arange(len(cal))
    # Hand back the calendar.
    return cal


# Fallback: build symbol -> segment from the trades store if daily_stats lacks it.
def build_segment_map(con):
    """(date, symbol) -> segment. Only used when daily_stats has no segment column."""
    # Count trades per (date, symbol, segment) so ties can be broken by activity.
    q = f"""
        WITH counts AS (
            SELECT date, symbol, segment, count(*) AS n
            FROM read_parquet('{PARSED_ROOT}/trades/*/*.parquet', hive_partitioning=true)
            WHERE symbol IS NOT NULL
            GROUP BY date, symbol, segment
        ),
        ranked AS (
            SELECT date, symbol, segment,
                   row_number() OVER (PARTITION BY date, symbol ORDER BY n DESC) AS rn,
                   count(*)    OVER (PARTITION BY date, symbol)                  AS n_segments
            FROM counts
        )
        SELECT date, symbol, segment, n_segments FROM ranked WHERE rn = 1
    """
    # Execute and return.
    seg = con.sql(q).df()
    # hive_partitioning auto-casts date= to DATE; daily_stats stores date as a
    # plain string. Normalise so the pandas merge keys have matching dtypes.
    seg["date"] = seg["date"].astype(str).str.slice(0, 10)
    return seg


# Days each symbol actually TRADED -- the honest denominator (correction 2).
def build_days_traded(con):
    """symbol -> number of distinct dates the symbol traded at all.

    This is larger than the number of days it appears in daily_stats, because
    unscreenable days (limit-locked, one-sided, auction-only) produce no stats row.
    """
    # One row per symbol; counts dates present in the trades store.
    q = f"""
        SELECT symbol, count(DISTINCT date) AS days_traded
        FROM read_parquet('{PARSED_ROOT}/trades/*/*.parquet', hive_partitioning=true)
        WHERE symbol IS NOT NULL
        GROUP BY symbol
    """
    # Return as a Series indexed by symbol for a fast .map().
    return con.sql(q).df().set_index("symbol")["days_traded"]


# --------------------------- persistence metrics ----------------------------
# Rank symbols within each (date, segment) and apply the materiality gate.
def add_daily_rank(df, fee_tag):
    """Rank WITHIN each (date, segment) by pairing-adjusted ceiling.

    Within-day, so session length cancels out -- this metric is regime-immune.
    """
    # The pairing-adjusted ceiling column for this fee scenario.
    col = f"ceiling_paired_rt_{fee_tag}"
    # Work on a copy so the caller's frame is untouched.
    df = df.copy()
    # Rank descending within the day and segment; ties share the better rank.
    df["rank_in_day"] = (df.groupby(["date", "segment"])[col]
                           .rank(ascending=False, method="min"))
    # Materiality gate: the day must also be worth a meaningful amount of money.
    df["is_material"] = df[col] >= MATERIAL_PKR[fee_tag]
    # THE metric: top-N AND materially large. Rank on this.
    df["in_top_n"] = (df["rank_in_day"] <= TOP_N) & df["is_material"]
    # Rank-only version retained so the two can be compared (see correction 1).
    df["in_top_n_rankonly"] = df["rank_in_day"] <= TOP_N
    # Hand back the annotated frame.
    return df


# Collapse the daily frame into one row per symbol.
def persistence_table(df, fee_tag, days_traded, min_days=MIN_DAYS):
    """Per-symbol persistence metrics.

    Within-day metrics use ALL days (regime-immune). Cross-day LEVEL metrics use
    LEVEL_REGIMES only. Denominators are days TRADED, not days present.
    """
    # Pairing-adjusted ceiling column for this fee scenario.
    col = f"ceiling_paired_rt_{fee_tag}"
    # Raw (un-paired) ceiling, kept for comparison.
    raw_col = f"ceiling_pkr_rt_{fee_tag}"

    # --- within-day metrics: every screened day counts, regime-immune ---
    # Group the daily rows by symbol.
    g = df.groupby("symbol")
    # Assemble the count-based block.
    out = pd.DataFrame({
        # Days the symbol appeared in daily_stats (i.e. was screenable).
        "n_days_screened": g.size(),
        # Days it was both top-N and material -- the numerator that matters.
        "n_days_top20": g["in_top_n"].sum(),
        # Days it cleared the money floor regardless of rank.
        "n_days_material": g["is_material"].sum(),
        # Days it had any positive pairing-adjusted ceiling at all.
        "n_days_tradeable": g[col].apply(lambda s: int((s > 0).sum())),
        # Typical rank on days it was screenable.
        "median_rank": g["rank_in_day"].median(),
        # Best rank it ever achieved.
        "best_rank": g["rank_in_day"].min(),
        # Rank-only top-N count, for the correction-1 comparison.
        "n_days_top20_rankonly": g["in_top_n_rankonly"].sum(),
    })

    # --- denominator correction: days TRADED, not days screened ---
    # Map in the total days the symbol traded (from the trades store).
    out["days_traded"] = out.index.map(days_traded)
    # Days it traded but could not be screened (locked, one-sided, auction-only).
    out["days_unscreenable"] = out["days_traded"] - out["n_days_screened"]
    # Share of traded days that were unscreenable -- a red flag in its own right.
    out["pct_days_unscreenable"] = out["days_unscreenable"] / out["days_traded"]
    # THE headline metric: attractive days as a share of days you could have traded.
    out["pct_days_top20"] = out["n_days_top20"] / out["days_traded"]
    # The old, optimistic version (denominator = days screened), for comparison.
    out["pct_days_top20_screened"] = out["n_days_top20"] / out["n_days_screened"]
    # Rank-only variant on the honest denominator, to expose the gate's effect.
    out["pct_days_top20_rankonly"] = out["n_days_top20_rankonly"] / out["days_traded"]
    # Share of traded days clearing the money floor.
    out["pct_days_material"] = out["n_days_material"] / out["days_traded"]
    # Share of traded days with any positive ceiling.
    out["pct_days_tradeable"] = out["n_days_tradeable"] / out["days_traded"]

    # --- cross-day LEVEL metrics: comparable sessions only ---
    # Restrict to the regimes whose daily totals are comparable.
    lvl_src = df[df["regime"].isin(LEVEL_REGIMES)]
    # Group those rows by symbol.
    gn = lvl_src.groupby("symbol")
    # Assemble the level block.
    lvl = pd.DataFrame({
        # How many comparable-regime days fed these level statistics.
        "n_days_level": gn.size(),
        # Typical daily ceiling on comparable days.
        "ceiling_median": gn[col].median(),
        # Lower quartile of daily ceiling.
        "ceiling_p25": gn[col].quantile(0.25),
        # Upper quartile of daily ceiling.
        "ceiling_p75": gn[col].quantile(0.75),
        # Best single comparable day.
        "ceiling_max": gn[col].max(),
        # Raw (un-paired) median, to see how much pairing costs.
        "ceiling_raw_median": gn[raw_col].median(),
        # Typical daily traded value.
        "notional_m_median": gn["notional_m"].median(),
        # Typical spread level.
        "spread_bps_median": gn["median_spread_bps"].median(),
        # Typical share of session spent wide.
        "pct_time_wide_median": gn["pct_time_wide"].median(),
    })
    # Interquartile range of the daily ceiling.
    lvl["ceiling_iqr"] = lvl["ceiling_p75"] - lvl["ceiling_p25"]
    # Dispersion relative to level: episodic names show a large ratio. Guard the
    # divide -- a symbol whose median ceiling is 0 has no stable level at all.
    lvl["iqr_over_median"] = np.where(
        lvl["ceiling_median"] > 0, lvl["ceiling_iqr"] / lvl["ceiling_median"], np.nan)

    # Join the level block onto the count block.
    out = out.join(lvl, how="left")
    # Score only symbols with enough traded history to be meaningful.
    out = out[out["days_traded"] >= min_days]
    # Rank on persistence first, level second -- never on a single day.
    out = out.sort_values(["pct_days_top20", "ceiling_median"], ascending=False)
    # Move symbol out of the index into a column.
    return out.reset_index()


# Leaderboard stability: does the same set of names stay attractive week to week?
def rank_autocorr(df, fee_tag, lag=AUTOCORR_LAG):
    """Spearman correlation of the symbol leaderboard between day t and t+lag.

    High -> a standing watchlist works. Low -> attractiveness rotates, and the
    episodic population needs an event trigger rather than a standing quote.
    """
    # Pairing-adjusted ceiling column for this fee scenario.
    col = f"ceiling_paired_rt_{fee_tag}"
    # Distinct dates with their trading-day index, in order.
    idx = df[["date", "day_idx"]].drop_duplicates().sort_values("day_idx")
    # Pre-split the frame by day index so the pair loop is cheap.
    by_day = {int(r.day_idx): df.loc[df["date"] == r.date, ["symbol", col]]
              for r in idx.itertuples()}
    # Accumulated per-pair correlations.
    rows = []
    # Walk every day that has a partner `lag` days later.
    for i in sorted(by_day):
        # The partner day index.
        j = i + lag
        # Skip if the partner day is missing (end of sample, or a gap).
        if j not in by_day:
            continue
        # The two days' frames.
        a, b = by_day[i], by_day[j]
        # Keep only symbols present on BOTH days.
        m = a.merge(b, on="symbol", suffixes=("_t", "_t5"))
        # Too few overlapping symbols makes the correlation meaningless.
        if len(m) < 30:
            continue
        # Rank on day t (descending: rank 1 = best).
        ra = m[f"{col}_t"].rank(ascending=False)
        # Rank on day t+lag.
        rb = m[f"{col}_t5"].rank(ascending=False)
        # Spearman = Pearson correlation of the ranks.
        rows.append({"day_idx_t": i, "n_symbols": len(m),
                     "spearman": float(ra.corr(rb))})
    # Return one row per date pair.
    return pd.DataFrame(rows)


# Diagnostic: is a Friday's daily ceiling comparable to a normal day's?
def friday_ratio_check(ds, fee_tag):
    """Paired per-symbol median ceiling by regime, relative to normal days.

    Paired (each symbol against itself) so the comparison isolates the session
    effect from changes in which symbols traded. Informs LEVEL_REGIMES.
    """
    # Pairing-adjusted ceiling column for this fee scenario.
    col = f"ceiling_paired_rt_{fee_tag}"
    # Median daily ceiling per symbol per regime.
    piv = ds.pivot_table(index="symbol", columns="regime", values=col, aggfunc="median")
    # Only compare names that have a real level on normal days.
    if "normal" not in piv.columns:
        # Cannot compare without a baseline.
        return pd.DataFrame()
    # Restrict to symbols above the money floor on normal days.
    piv = piv[piv["normal"] > MATERIAL_PKR[fee_tag]]
    # Accumulated ratios.
    rows = []
    # Compare each non-normal regime against normal.
    for r in ("friday", "ramadan", "ramadan_friday"):
        # Skip regimes absent from the sample.
        if r not in piv.columns:
            continue
        # Per-symbol ratio, dropping infinities and missing values.
        ratio = (piv[r] / piv["normal"]).replace([np.inf, -np.inf], np.nan).dropna()
        # Skip if nothing to compare.
        if len(ratio) == 0:
            continue
        # Record the median ratio and the sample size.
        rows.append({"regime": r, "n_symbols": len(ratio),
                     "median_ratio_vs_normal": round(float(ratio.median()), 3)})
    # Return the comparison table.
    return pd.DataFrame(rows)


# --------------------------------- driver -----------------------------------
# Orchestrate: calendar -> enrich -> metrics per segment per fee scenario.
def main():
    # Every fee scenario must have a money floor, or add_daily_rank KeyErrors deep
    # in the loop. Fail here with a clear message instead.
    missing = [t for t in FEE_TAGS if t not in MATERIAL_PKR]
    assert not missing, f"MATERIAL_PKR missing floors for {missing}"
    # DuckDB is used only for reading parquet; all metrics are pandas.
    import duckdb
    # In-memory connection.
    con = duckdb.connect()
    # Bound memory so the trades scan cannot swap a 16GB machine.
    con.execute("PRAGMA memory_limit='8GB'")

    # ---- 1. session calendar ----
    # Announce the step.
    print("1/5 building session calendar ...")
    # Derive open/close/regime per date from the trades store.
    cal = build_session_calendar(con)
    # Register so DuckDB can write it out.
    con.register("cal_v", cal)
    # Persist the calendar for reuse by other scripts.
    con.execute("COPY cal_v TO 'session_calendar.parquet' (FORMAT PARQUET)")
    # Drop the registration.
    con.unregister("cal_v")
    # Show the measured regime summary.
    print(cal.groupby("regime")[["elapsed_h", "traded_h", "max_gap_min", "trades"]]
             .median().round(2).to_string())

    # ---- 2. days traded (the honest denominator) ----
    # Announce the step.
    print("\n2/5 building days-traded map ...")
    # symbol -> distinct dates traded.
    days_traded = build_days_traded(con)
    # Report coverage.
    print(f"  {len(days_traded)} symbols; median days traded = {int(days_traded.median())}")

    # ---- 3. load and enrich daily_stats ----
    # Announce the step.
    print("\n3/5 loading and enriching daily_stats ...")
    # Read the merged Stage-1 output into pandas (small: ~123k rows).
    ds = con.sql(f"SELECT * FROM '{DAILY_STATS}'").df()
    # Use the native segment column if Stage 1 wrote one.
    if "segment" in ds.columns:
        # Nothing to join.
        print("  segment column present -- using it")
    # Otherwise fall back to joining it from the trades store.
    else:
        # Warn so the fallback is visible.
        print("  segment column ABSENT -- joining from trades store")
        # Build the map and merge it on.
        ds = ds.merge(build_segment_map(con), on=["date", "symbol"], how="left")
    # Any row without a segment becomes UNKNOWN rather than being dropped.
    ds["segment"] = ds["segment"].fillna("UNKNOWN")
    # PSX segment codes -> readable names (011 ready equities, 031 deliverable
    # futures, 041 cash-settled futures). Output files are then named by instrument
    # (persistence_REG_2p00.csv) and the REG filter in the Friday check matches.
    ds["segment"] = ds["segment"].map(
        {"011": "REG", "031": "STOCK_DEL_FUT", "041": "STOCK_CS_FUT"}).fillna(ds["segment"])
    # Attach the regime, traded hours and day index by date.
    ds = ds.merge(cal[["date", "regime", "traded_h", "day_idx", "dow"]],
                  on="date", how="left")
    # Register and persist the enriched frame for ad-hoc querying.
    con.register("ds_v", ds)
    # Write it out.
    con.execute("COPY ds_v TO 'daily_stats_enriched.parquet' (FORMAT PARQUET)")
    # Drop the registration.
    con.unregister("ds_v")
    # Show the composition by segment.
    print("  rows by segment:\n" + ds["segment"].value_counts().to_string())
    # Show the composition by regime.
    print("  rows by regime:\n" + ds["regime"].value_counts().to_string())
    # If skip records exist, summarise why days were lost.
    if os.path.exists(SKIPS_FILE):
        # Report the reason breakdown.
        print("  unscreenable day reasons:")
        # Query the merged skips table.
        print(con.sql(f"""
            SELECT reason, count(*) AS n, count(DISTINCT symbol) AS symbols
            FROM '{SKIPS_FILE}' GROUP BY reason ORDER BY n DESC
        """).df().to_string(index=False))

    # ---- 4. Friday regime diagnostic ----
    # Announce the step.
    print("\n4/5 regime comparability check (informs LEVEL_REGIMES) ...")
    # Friday-vs-normal ceiling ratio is instrument-specific: futures roll and
    # thin out differently from equities, so measure it PER SEGMENT rather than
    # pooling. LEVEL_REGIMES is one global switch, so if the segments disagree
    # (e.g. equities ~0.86 but futures ~0.75) that is itself the finding -- you
    # may want Fridays in for REG level metrics but out for futures.
    for seg_name in sorted(ds["segment"].unique()):
        # This segment's rows only.
        seg_df = ds[ds["segment"] == seg_name]
        # Paired per-symbol ratio versus normal days, at the operative fee.
        fr = friday_ratio_check(seg_df, FEE_TAGS[0])
        # Show it if computable.
        if len(fr):
            # Pull out just the friday row for the headline call.
            fri = fr[fr["regime"] == "friday"]
            # Format the friday ratio, or note its absence.
            fri_txt = (f"friday={fri['median_ratio_vs_normal'].iloc[0]:.3f} "
                       f"(n={int(fri['n_symbols'].iloc[0])})") if len(fri) else "friday=n/a"
            # One headline line per segment.
            print(f"  {seg_name:15s} {fri_txt}")
            # Full breakdown (friday / ramadan / ramadan_friday) indented under it.
            print(fr.to_string(index=False).replace("\n", "\n    "))
        # Otherwise say why not, per segment.
        else:
            # No normal-regime baseline above the money floor for this segment.
            print(f"  {seg_name:15s} not computable (no normal baseline above MATERIAL_PKR)")
    # Reminder of the current global switch and how to read the per-segment numbers.
    print(f"  LEVEL_REGIMES is currently {LEVEL_REGIMES}. Decide on the REG row:")
    print("  keep Fridays if REG friday >= ~0.82, else set ('normal',). Note if")
    print("  futures disagree -- that is a real instrument difference, not noise.")

    # ---- 5. persistence metrics per segment per fee scenario ----
    # Announce the step.
    print("\n5/5 persistence metrics ...")
    # One leaderboard per segment: instruments are not comparable across segments.
    for seg_name in sorted(ds["segment"].unique()):
        # Rows for this segment only.
        sub_all = ds[ds["segment"] == seg_name]
        # One leaderboard per fee scenario: ranking inverts with the fee level.
        for fee in FEE_TAGS:
            # The ceiling column must exist for this fee tag.
            if f"ceiling_paired_rt_{fee}" not in sub_all.columns:
                # Skip a fee tag the Stage-1 grid did not produce.
                continue
            # Rank within each day and apply the materiality gate.
            sub = add_daily_rank(sub_all, fee)
            # Collapse to one row per symbol.
            p = persistence_table(sub, fee, days_traded)
            # Nothing scored (all below MIN_DAYS) -> skip this combination.
            if len(p) == 0:
                # Move on.
                continue
            # Persistence output filename.
            f1 = f"persistence_{seg_name}_{fee}.csv"
            # Write it.
            p.to_csv(f1, index=False)
            # Leaderboard stability for the same slice.
            ac = rank_autocorr(sub, fee)
            # Autocorrelation output filename.
            f2 = f"rank_autocorr_{seg_name}_{fee}.csv"
            # Write it.
            ac.to_csv(f2, index=False)
            # Median stability across all date pairs.
            med = ac["spearman"].median() if len(ac) else float("nan")
            # Report the combination.
            print(f"  {seg_name:15s} fee={fee:6s} -> {len(p):4d} symbols | "
                  f"lag{AUTOCORR_LAG} autocorr median={med:.3f} | {f1}")

    # Close the connection.
    con.close()
    # Final note on how to read the output.
    print("\nDone. Rank on pct_days_top20 (denominator = days TRADED).")
    print("Cross-check: high pct_days_top20_rankonly but low pct_days_top20 = episodic.")
    print("             high pct_days_unscreenable = frequently locked/one-sided.")


# Only run when executed as a script.
if __name__ == "__main__":
    # Dispatch.
    main()
