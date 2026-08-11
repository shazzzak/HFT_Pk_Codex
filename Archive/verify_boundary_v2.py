"""
verify_boundary_v2.py

Corrected close-boundary hygiene check for markout_5000ms_bps.

v1 mistake: it inspected the last 2000 ROWS, which is not the last 5 SECONDS. This version
uses the true final-5s time window (ts_exch is milliseconds, confirmed) and reports the
BLAST RADIUS: how many rows fall in the last 5s, what fraction of the day that is, and how
big their markouts are. Severity follows from magnitude:
  * no markouts in the final 5s            -> CLEAN
  * small markouts, tiny row fraction      -> LOW-IMPACT (clamp/edge effect; ignore)
  * large markouts (overnight-sized)       -> CONTAMINATED (cross-boundary; must fix)

Reads feature_store via the wiring module. Samples several days across the full range.
"""

# stdlib + the wiring module for COLS / enumerate_partitions
import os
import feature_store_wiring as W

# how many days to sample per symbol (evenly spread across the date range)
SAMPLE_PER_SYMBOL = 5
# the 5s window in ms
WINDOW_MS = 5000
# magnitude (bps) above which tail markouts look overnight-sized rather than intraday
CONTAM_BPS = 15.0


# ---- verdict: classify a day's metrics ----
def verdict(n_end, n_end_mk, mean_abs_end_mk, n_total):
    # no markout-bearing rows in the final 5s -> the boundary was excluded cleanly
    if n_end_mk == 0:
        return "CLEAN (no markout in final 5s)"
    # fraction of the whole day sitting in that final-5s window
    frac = n_end / n_total if n_total else 0.0
    # large tail markouts imply a cross-boundary reference (overnight/auction)
    if mean_abs_end_mk is not None and mean_abs_end_mk >= CONTAM_BPS:
        return f"CONTAMINATED (tail |mk|~{mean_abs_end_mk:.1f}bps -> cross-boundary; FIX)"
    # otherwise it is a mild edge effect on a tiny row fraction
    return (f"LOW-IMPACT ({n_end_mk} rows, {frac:.4%} of day, |mk|~"
            f"{(mean_abs_end_mk or 0):.1f}bps)")


# ---- pandas mirror of the SQL metrics, used ONLY for offline testing ----
def compute_metrics_pandas(df):
    # numpy for nan handling
    import numpy as np
    # total rows
    n = len(df)
    # min/max timestamp
    min_ts, max_ts = df["ts"].min(), df["ts"].max()
    # rows that carry a markout
    notnull = df["mk"].notna()
    # latest timestamp that still has a markout
    max_ts_mk = df.loc[notnull, "ts"].max() if notnull.any() else np.nan
    # rows within the final 5s time window
    end_mask = df["ts"] > (max_ts - WINDOW_MS)
    # count of those, and of those that carry a markout
    n_end = int(end_mask.sum())
    n_end_mk = int((end_mask & notnull).sum())
    # mean absolute markout among the final-5s rows that have one
    sub = df.loc[end_mask & notnull, "mk"]
    mean_abs_end_mk = float(sub.abs().mean()) if len(sub) else None
    # assemble the metric row
    return dict(n=n, min_ts=min_ts, max_ts=max_ts, span_ms=max_ts - min_ts,
                gap_ms=(max_ts - max_ts_mk), frac_null_all=float((~notnull).mean()),
                n_end=n_end, n_end_mk=n_end_mk, mean_abs_end_mk=mean_abs_end_mk)


# ---- real metrics via duckdb over one parquet file ----
def compute_metrics_duckdb(con, path):
    # real column names from the wiring map
    ts, mk = W.COLS["ts"], W.COLS["markout_5s"]
    # single-pass query: aggregate min/max/null, then FILTER for the final-5s window
    q = f"""
    WITH t AS (
        SELECT CAST({ts} AS DOUBLE) AS ts, {mk} AS mk
        FROM read_parquet('{path}')
    ),
    m AS (
        SELECT COUNT(*) AS n, MIN(ts) AS min_ts, MAX(ts) AS max_ts,
               MAX(CASE WHEN mk IS NOT NULL THEN ts END) AS max_ts_mk,
               AVG(CASE WHEN mk IS NULL THEN 1.0 ELSE 0.0 END) AS frac_null_all
        FROM t
    )
    SELECT m.n, m.min_ts, m.max_ts, (m.max_ts - m.min_ts) AS span_ms,
           (m.max_ts - m.max_ts_mk) AS gap_ms, m.frac_null_all,
           COUNT(*) FILTER (WHERE t.ts > m.max_ts - {WINDOW_MS}) AS n_end,
           COUNT(*) FILTER (WHERE t.ts > m.max_ts - {WINDOW_MS} AND t.mk IS NOT NULL) AS n_end_mk,
           AVG(ABS(t.mk)) FILTER (WHERE t.ts > m.max_ts - {WINDOW_MS} AND t.mk IS NOT NULL) AS mean_abs_end_mk
    FROM t CROSS JOIN m
    GROUP BY m.n, m.min_ts, m.max_ts, m.max_ts_mk, m.frac_null_all
    """
    # run and return the single row as a dict
    return con.execute(q).df().iloc[0].to_dict()


# ---- sample evenly across dates, per symbol ----
def sample_partitions():
    # all 414 (symbol, date, path)
    parts = W.enumerate_partitions()
    # collect a spread per symbol
    chosen = []
    # group by symbol
    for sym in sorted({p[0] for p in parts}):
        # this symbol's partitions in date order
        sp = [p for p in parts if p[0] == sym]
        # evenly spaced indices across the range
        if sp:
            idxs = [round(i * (len(sp) - 1) / (SAMPLE_PER_SYMBOL - 1)) for i in range(SAMPLE_PER_SYMBOL)]
            # dedupe while preserving order, then collect
            for i in sorted(set(idxs)):
                chosen.append(sp[i])
    # hand back the sample
    return chosen


# ---- main ----
def main():
    # lazy duckdb import
    import duckdb
    # one connection
    con = duckdb.connect()
    # header
    print(f"{'symbol/date':40s} {'N':>10s} {'gap_ms':>8s} {'null%':>6s} "
          f"{'n_end':>6s} {'end_mk':>6s} {'|mk|':>6s}  verdict")
    # characterize the clock once (epoch-ms vs since-open) from the first sample
    clock_printed = False
    # evaluate each sampled day
    for sym, dt, path in sample_partitions():
        # compute the metrics
        m = compute_metrics_duckdb(con, path)
        # print the ts clock characterization once
        if not clock_printed:
            # epoch-ms for 2025-2026 is ~1.75e12; much smaller means since-open/since-midnight
            kind = "epoch-ms" if m["min_ts"] > 1e12 else "since-open/midnight-ms"
            print(f"# ts_exch clock looks like: {kind} (min_ts={m['min_ts']:.0f})\n")
            clock_printed = True
        # classify severity
        v = verdict(m["n_end"], m["n_end_mk"], m["mean_abs_end_mk"], m["n"])
        # one row per sampled day
        print(f"{sym+' '+dt:40s} {m['n']:>10,.0f} {m['gap_ms']:>8.0f} "
              f"{m['frac_null_all']*100:>5.2f}% {m['n_end']:>6.0f} {m['n_end_mk']:>6.0f} "
              f"{(m['mean_abs_end_mk'] or 0):>6.2f}  {v}")


# entry point
if __name__ == "__main__":
    main()
