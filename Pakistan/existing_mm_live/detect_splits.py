# ============================================================================
# detect_splits.py  (v2 -- corrected threshold + exchange confirmation)
# ============================================================================
# Read-only. Writes one new timestamped CSV. Deletes nothing.
#
# v1 WAS WRONG AND THIS DOCUMENTS WHY
#   v1 flagged any overnight |log move| > 0.15 on the claim that "PSX circuit
#   bands keep genuine overnight gaps well inside 15%". The data refuted it:
#   43 of 49 flags were ordinary 16-24% moves (ratios 0.78-1.21, nowhere near
#   any split factor), and they clustered on shared dates across unrelated
#   names -- four on 2026-02-02, three on 2026-03-02. Market moves, not
#   corporate actions.
#
#   The separator is the SIZE of the move, not how near the ratio lands to a
#   split factor. The smallest real split (1:2) is a log move of 0.693; the
#   observed price-move noise topped out at 0.24. Hence JUMP_THRESHOLD = 0.5,
#   which sits in the empty band between the two populations.
#
# TWO STAGES
#   Stage 1 -- SCREEN. Median mid per symbol-day from the feature store, log
#     ratio between consecutive trading days, flag |log move| > 0.5. Fast,
#     since it queries the parquet in place.
#   Stage 2 -- CONFIRM. For flagged names ONLY, compare the exchange's own
#     published prev_close on day t against the last traded price on day t-1.
#     The exchange ADJUSTS prev_close through a corporate action, so a large
#     divergence is the exchange itself telling you an action occurred. This
#     is independent of the price series and is what corp_action_detector was
#     designed around. A price move alone cannot produce it.
#
# WHAT A CONFIRMED ACTION INVALIDATES
#   session_scale  -- WRONG. scale ~ med_spread_pkr / fair_ref^2, both medians
#                     taken over a series with a step in it. After a k:1 split
#                     the correct scale is k TIMES the pre-split one, so a
#                     blended median is right for neither regime.
#   volume_profile -- WRONG. shares/min medians blend both regimes.
#   time_windows   -- mostly OK. minutes = max_inv/(vol_per_min*POV); both are
#                     share counts scaling by k, so the ratio ~cancels.
#   engine clip    -- OK. trailing 10-day median, walk-forward, re-adapts.
#   intraday replay-- OK. one day at a time, internally consistent.
#
# Run from existing_mm_live/:
#   caffeinate -is python detect_splits.py
# ============================================================================

# wall-clock timing
import time
# numeric
import numpy as np
# frames
import pandas as pd

# the driver: discover_dates, open_datasets, read_symbol, REQ_* specs
import run_legacy_mm as R
# shared constants, path guards and safe_out
import expansion_names as EX

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# push the canonical raw-store root on and prove it reads
ALL_DATES = EX.bind_parsed_root(R)
# the feature store built in step 2
FS_ROOT = EX.RESULTS_ROOT / "feature_store"
# every name whose calibration depends on a stable price series: the 76
# candidates AND the 38 production names, which carry the same exposure
NAMES = list(EX.ALL_NAMES)
# Stage-1 screen threshold. 0.5 sits in the empty band between the observed
# price-move population (max 0.24) and the smallest real split (1:2 = 0.693).
JUMP_THRESHOLD = 0.5
# Stage-2 confirmation threshold: how far the exchange's prev_close may sit
# from the previous day's last trade before it counts as an adjustment.
# 5% is well above tick/rounding noise and far below any split factor.
PREV_CLOSE_TOL = 0.05
# the split factors worth naming
COMMON_RATIOS = [10.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5,
                 1 / 1.5, 0.5, 1 / 2.5, 1 / 3, 0.25, 0.2, 0.1]


def median_mid_per_day(sym):
    # one row per date: that day's median mid, read from the feature store
    glob = f"{FS_ROOT}/{sym}/date=*.parquet"
    # DuckDB queries the parquet in place -- much faster than loading it
    try:
        # in-place parquet query
        import duckdb
        # the date is encoded in the partition filename
        q = f"""
            SELECT regexp_extract(filename, 'date=([0-9-]+)', 1) AS date,
                   median(mid) AS med_mid
            FROM read_parquet('{glob}', filename=true)
            WHERE mid > 0
            GROUP BY 1
            ORDER BY 1
        """
        # execute and hand back a frame
        return duckdb.sql(q).df()
    except Exception:
        # pandas fallback, one partition at a time
        import glob as _g
        # every partition for this symbol
        files = sorted(_g.glob(glob))
        # nothing to read
        if not files:
            return pd.DataFrame(columns=["date", "med_mid"])
        # collect one row per file
        rows = []
        # walk the partitions
        for f in files:
            # the date from the filename
            dt = f.rsplit("date=", 1)[1].replace(".parquet", "")
            # just the mid column
            m = pd.read_parquet(f, columns=["mid"])
            # positive mids only
            m = m[m["mid"] > 0]
            # skip an empty day
            if len(m) == 0:
                continue
            # that day's median
            rows.append({"date": dt, "med_mid": float(m["mid"].median())})
        # assembled, date-ordered
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def nearest_ratio(r):
    # name the closest standard split factor so 0.203 reads as "1:5"
    best = min(COMMON_RATIOS, key=lambda c: abs(np.log(r) - np.log(c)))
    # how far off, as a percentage in log space
    err = abs(np.log(r) - np.log(best)) * 100
    # human-readable
    label = (f"{best:g}:1 reverse-split-like" if best >= 1
             else f"1:{1/best:.0f} split-like")
    # the match and its quality
    return label, err


def stage1_screen():
    # median mid per day per name, then consecutive-day log ratios
    print("=" * 78)
    print(f"STAGE 1 -- SCREEN: {len(NAMES)} names "
          f"({len(EX.NEW_NAMES)} candidates + {len(EX.INCUMBENT)} production)")
    print("=" * 78)
    print(f"  feature store  : {FS_ROOT}")
    print(f"  threshold      : |log move| > {JUMP_THRESHOLD}")
    print(f"  why this value : the smallest real split (1:2) is 0.693; the")
    print(f"                   observed price-move population tops out at 0.24.")
    print(f"                   v1 used 0.15 and produced 43 false positives.\n")
    # start the clock
    t0 = time.perf_counter()
    # every flagged jump
    flags = []
    # names with no feature store
    missing = []
    # walk the names
    for i, sym in enumerate(NAMES, 1):
        # that name's per-day median mid
        d = median_mid_per_day(sym)
        # no feature store for this name
        if len(d) < 2:
            missing.append(sym)
            continue
        # date order matters
        d = d.sort_values("date").reset_index(drop=True)
        # previous day's median mid
        prev = d["med_mid"].shift(1)
        # today over yesterday
        ratio = d["med_mid"] / prev
        # symmetric for splits and reverse splits
        logmove = np.log(ratio)
        # the days that clear the threshold
        for idx in d.loc[logmove.abs() > JUMP_THRESHOLD].index:
            # the implied ratio
            r = float(ratio.loc[idx])
            # closest standard factor
            label, err = nearest_ratio(r)
            # one row per flagged day
            flags.append({
                "symbol": sym,
                "in_production": sym in EX.INCUMBENT,
                "date": d.loc[idx, "date"],
                "prev_date": d.loc[idx - 1, "date"],
                "prev_med_mid": round(float(prev.loc[idx]), 4),
                "med_mid": round(float(d.loc[idx, "med_mid"]), 4),
                "ratio": round(r, 4),
                "log_move": round(float(logmove.loc[idx]), 4),
                "nearest": label,
                "match_err_pct": round(err, 2),
            })
        # progress
        if i % 25 == 0 or i == len(NAMES):
            print(f"  {i}/{len(NAMES)} names  {time.perf_counter()-t0:,.0f}s",
                  flush=True)
    # report names that could not be screened
    if missing:
        print(f"\n  NO FEATURE STORE for {len(missing)} names: {missing}")
    # the screen result
    return pd.DataFrame(flags)


def stage2_confirm(screen_df):
    # THE INDEPENDENT TEST. The exchange publishes prev_close in ob_snapshot
    # and ADJUSTS it through a corporate action. Comparing prev_close on day t
    # against the last traded price on day t-1 therefore asks the exchange
    # directly whether an action happened -- a price move cannot produce a
    # divergence, only an adjustment can.
    print("\n" + "=" * 78)
    print("STAGE 2 -- CONFIRM against the exchange's published prev_close")
    print("=" * 78)
    # nothing to confirm
    if len(screen_df) == 0:
        print("  no stage-1 flags to confirm.")
        return screen_df
    # only the flagged names need the raw-store pass
    syms = sorted(screen_df.symbol.unique())
    # say what is being checked
    print(f"  checking {len(syms)} flagged names: {syms}")
    print(f"  tolerance: prev_close within {PREV_CLOSE_TOL:.0%} of the previous")
    print(f"             day's last trade counts as UNADJUSTED\n")
    # the dates of interest, plus the preceding day for each
    want_dates = set(screen_df.date) | set(screen_df.prev_date)
    # per (symbol, date): the exchange prev_close and the day's last trade
    obs = {}
    # start the clock
    t0 = time.perf_counter()
    # walk only the dates we need
    for i, date in enumerate(sorted(want_dates), 1):
        # that date's datasets
        dsets = R.open_datasets(date)
        # missing partition
        if dsets is None:
            continue
        # each flagged symbol
        for sym in syms:
            # the day's snapshots, which carry prev_close
            snap = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP + ["prev_close"], sym)
            # the day's trades, for the last traded price
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # the exchange's published previous close for this day
            pc = None
            # take the first non-null prev_close of the day
            if len(snap) and "prev_close" in snap.columns:
                # non-null values only
                vals = snap["prev_close"].dropna()
                # the day's published prev_close
                if len(vals):
                    pc = float(vals.iloc[0])
            # the day's last traded price, by transact_time
            lt = None
            # only when the name traded
            if len(t):
                # order by exchange time and take the final print
                ts = R.to_ms(t["transact_time"])
                # the last trade of the day
                lt = float(t.loc[ts.idxmax(), "price"])
            # record both
            obs[(sym, str(date))] = {"prev_close": pc, "last_trade": lt}
        # progress
        if i % 5 == 0 or i == len(want_dates):
            print(f"  {i}/{len(want_dates)} dates  {time.perf_counter()-t0:,.0f}s",
                  flush=True)

    # attach the confirmation to each flag
    out = screen_df.copy()
    # the exchange's prev_close on the jump day
    out["exch_prev_close"] = [obs.get((r.symbol, r.date), {}).get("prev_close")
                              for r in out.itertuples()]
    # the actual last trade the day before
    out["prior_last_trade"] = [obs.get((r.symbol, r.prev_date), {}).get("last_trade")
                               for r in out.itertuples()]
    # the ratio between them: 1.0 means the exchange did NOT adjust
    out["adj_ratio"] = out["exch_prev_close"] / out["prior_last_trade"]
    # the verdict, per flag
    verdicts = []
    # walk the flags
    for r in out.itertuples():
        # the exchange data is missing
        if pd.isna(r.adj_ratio):
            verdicts.append("NO_DATA")
        # prev_close matches the prior close -> no adjustment -> price move
        elif abs(np.log(r.adj_ratio)) <= np.log(1 + PREV_CLOSE_TOL):
            verdicts.append("PRICE_MOVE")
        # the exchange adjusted -> a corporate action occurred
        else:
            verdicts.append("CORP_ACTION")
    # attach it
    out["verdict"] = verdicts
    # the confirmed set
    return out


def main():
    # stage 1
    screen = stage1_screen()
    # stage 2
    full = stage2_confirm(screen)

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    # the clean case
    if len(full) == 0:
        print("No jumps above the threshold. session_scale and volume_profile")
        print("are safe over the full 207-day window for every name.")
    else:
        # split by verdict
        confirmed = full[full.verdict == "CORP_ACTION"]
        # show everything, confirmed first
        print(full.sort_values(["verdict", "in_production", "symbol"])
              .to_string(index=False))
        # the names that actually need action
        if len(confirmed):
            # production names are the urgent ones
            prod = sorted(confirmed[confirmed.in_production].symbol.unique())
            # candidates are pre-deployment
            cand = sorted(confirmed[~confirmed.in_production].symbol.unique())
            # the headline
            print(f"\nCONFIRMED corporate actions: {confirmed.symbol.nunique()} names")
            print(f"  production : {prod}")
            print(f"  candidates : {cand}")
            # what to do
            print("\nFOR EACH CONFIRMED NAME:")
            print("  session_scale and volume_profile must be recomputed on the")
            print("  POST-action window only -- that is the regime that goes live.")
            print("  Its 207-day backtest spans two incompatible price regimes, so")
            print("  no single scale is correct across the whole run. Either")
            print("  restrict the run for that name, or exclude it from the")
            print("  expansion decision.")
            # the production warning
            if prod:
                print(f"\n*** {len(prod)} PRODUCTION names affected: {prod}")
                print("    Their live session_scale in session_scales_20260818_0129.csv")
                print("    was computed over the blended window and is wrong NOW.")
                print("    scale ~ spread/price^2, so after a k:1 split the correct")
                print("    value is k TIMES the live one.")

    # a fresh timestamped destination; never overwrites
    out = EX.safe_out("corp_action_scan", "csv")
    # write whatever was found
    (full if len(full) else pd.DataFrame(columns=["symbol", "date", "verdict"])
     ).to_csv(out, index=False)
    # say where it went
    print(f"\nwrote {out}")


# entry point
if __name__ == "__main__":
    main()
