# ofi_windows_horserace.py -- the FAIR OFI race: trailing-WINDOWED OFI vs OBI.
#
# The original horserace raced SINGLE-EVENT OFI increments (a trailing window of
# exactly one book event) against OBI -- the weakest possible form of OFI, so
# its "OFI adds nothing" verdict was closed-by-construction. This script races
# OFI properly aggregated over trailing windows:
#
#   EVENT windows :  sum of ofi_l1 over the last 5 / 20 / 50 book events
#   TIME  windows :  sum of ofi_l1 over the last 1 / 2 / 3 / 5 seconds
#   HYBRID (min)  :  min(20 events, 2s) and min(50 events, 5s) -- whichever
#                    window contains FEWER events binds (SZ's "minimum of time
#                    and events" spec), so the window can't balloon in slow
#                    tape nor span minutes at the open.
#
# DISSECTED BY SESSION BUCKET (first15/middle/preclose45/last15): 50 events in
# first15 is seconds of wall-clock; in middle it can be minutes -- so a fixed
# event window means different things by bucket, and the race is run per bucket
# as well as pooled.
#
# DECISION GATE (same as the original): wire windowed OFI into the quotes only
# if median incremental OOS R^2 over obi_1 exceeds +0.0005.
#
# Label / split (identical to ofi_obi_horserace.py): forward mid-return over
# FWD_EVENTS book events (never crossing a day), winsorized, chronological
# 70/30 train/test, z-scored on train only.
#
# Run:  python3 existing_mm_live/ofi_windows_horserace.py     (~2-5 min)

from pathlib import Path
import numpy as np
import pandas as pd
import mm_harness as H

# results root + feature store (same convention as build_feature_store.py)
RESULTS_ROOT = Path("/Users/shazzak/Capital Stake - Results")
FS_ROOT = RESULTS_ROOT / "feature_store"

# the five race symbols (same as the original horserace)
SYMBOLS = ["PPL", "UBL", "MLCF", "TRG", "BOP"]
# the benchmark signal (the original race's winner at every depth)
BENCH = "obi_1"
# base per-event OFI column to window (L1 = the informative depth per the race)
OFI_BASE = "ofi_l1"
# trailing EVENT windows (book events)
EV_WINDOWS = [5, 20, 50, 60, 70, 80, 90]
# trailing TIME windows (seconds)
T_WINDOWS = [1, 2, 3, 5, 6, 7, 8, 9]
# HYBRID windows: (N events, T seconds) -- the smaller (fewer-events) one binds
HYBRIDS = [(20, 2), (50, 5), (60, 6), (70, 7), (80, 8), (90, 9)]
# forward-return label horizon (book events), as in the original race
FWD_EVENTS = 20
# chronological train fraction
TRAIN_FRAC = 0.70
# winsor percentile (each tail)
WINSOR_PCT = 0.5
# the four session buckets, chronological
BUCKETS = ("first15", "middle", "preclose45", "last15")
# the decision threshold on median incremental OOS R^2
GATE = 0.0005


# canonical session bucket for an exchange-ms timestamp, keyed off segments
def bucket_of(t, segs):
    # session open = first segment start; close = last segment end
    open_ms = segs[0][0]
    close_ms = segs[-1][1]
    # first 15 minutes after the open
    if t < open_ms + 15 * 60000:
        return "first15"
    # last 15 minutes before the close
    if t >= close_ms - 15 * 60000:
        return "last15"
    # the 45 minutes before last15
    if t >= close_ms - 60 * 60000:
        return "preclose45"
    # everything else
    return "middle"


# load one symbol's full feature store, tagged by day
def load_symbol(sym):
    # the per-symbol subtree
    d = FS_ROOT / sym
    # nothing there -> None
    if not d.exists():
        return None
    # all daily parquet files (both layout variants)
    files = sorted(d.glob("date=*/*.parquet")) or sorted(d.glob("date=*.parquet"))
    # none found
    if not files:
        return None
    # one frame per day, tagged so labels/windows never cross the overnight gap
    parts = []
    # read each day
    for f in files:
        # the day's frame
        df = pd.read_parquet(f)
        # tag the source day ("date=YYYY-MM-DD")
        df["_day"] = f.stem
        # collect
        parts.append(df)
    # the concatenated symbol frame
    return pd.concat(parts, ignore_index=True)


# winsorize a vector to [p, 100-p] percentiles
def winsor(s):
    # clip bounds
    lo, hi = np.nanpercentile(s, WINSOR_PCT), np.nanpercentile(s, 100 - WINSOR_PCT)
    # clipped copy
    return np.clip(s, lo, hi)


# out-of-sample R^2 of a 1-column (or k-column) regression, train->test
def oos_r2(X, y, train_mask):
    # design matrices with intercept
    Xtr = np.column_stack([np.ones(train_mask.sum()), X[train_mask]])
    Xte = np.column_stack([np.ones((~train_mask).sum()), X[~train_mask]])
    # split targets
    ytr, yte = y[train_mask], y[~train_mask]
    # least-squares on train (lstsq is robust to mild collinearity, unlike
    # normal-equation inversion -- avoids the original race's matmul warnings)
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    # test prediction
    yhat = Xte @ beta
    # OOS R^2 vs the test mean
    ss_res = float(np.sum((yte - yhat) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    # guard degenerate
    if ss_tot <= 0:
        return float("nan"), beta[1:]
    # the out-of-sample R^2
    return 1.0 - ss_res / ss_tot, beta[1:]


# build ALL trailing-window OFI columns for one day's frame (sorted by ts_exch)
def add_windows(day):
    # work on a copy (day is a group slice from the per-day concat)
    day = day.copy()
    # per-event OFI series
    s = day[OFI_BASE].astype(float)
    # ---- EVENT windows: rolling sums, but ONLY valid once the window is FULL.
    # min_periods=n -> the first (n-1) rows are NaN (not a partial-window sum),
    # so we never quote a "trailing-50" signal built from 5 events (SZ's warm-up
    # point). NaN rows are dropped downstream, i.e. the signal is OFF until warm.
    for n in EV_WINDOWS:
        # trailing N-event flow sum, valid only when N events exist
        day[f"ofi_ev{n}"] = s.rolling(n, min_periods=n).sum().to_numpy()
    # ---- TIME windows: sum over the last T seconds, valid only once we have BOTH
    # (a) at least T seconds of history since the day's first event, AND (b) a
    # minimum event count in the window (>=2) so a lone early event isn't a
    # "trailing-Ts" signal. Otherwise NaN (off until warm).
    idx = pd.to_datetime(day["ts_exch"].astype("int64"), unit="ms")
    # the series on the time index
    st = pd.Series(s.to_numpy(), index=idx)
    # elapsed ms since the day's first event, per row
    elapsed_ms = (day["ts_exch"] - day["ts_exch"].iloc[0]).to_numpy()
    for t in T_WINDOWS:
        # trailing T-second flow sum
        tsum = st.rolling(f"{t}s").sum().to_numpy()
        # events inside the T-second window (warm-up count guard)
        tcnt = st.rolling(f"{t}s").count().to_numpy()
        # valid only once T seconds have actually elapsed AND >=2 events present
        warm = (elapsed_ms >= t * 1000) & (tcnt >= 2)
        # off (NaN) until warm
        day[f"ofi_t{t}s"] = np.where(warm, tsum, np.nan)
    # ---- HYBRID min(N events, T seconds): whichever holds FEWER events binds.
    # Valid only when BOTH component windows are valid (both warm).
    for (n, t) in HYBRIDS:
        # events inside the trailing T-second window at each row
        cnt = st.rolling(f"{t}s").count().to_numpy()
        # the (already warm-guarded) time-window sum, and the full-window event sum
        tsum = day[f"ofi_t{t}s"].to_numpy()
        esum = day[f"ofi_ev{n}"].to_numpy()
        # min window = the one with FEWER events: if the T-sec window holds < n
        # events it is smaller -> use the time sum; else the n-event window binds.
        raw = np.where(cnt <= n, tsum, esum)
        # valid only when BOTH inputs are valid (NaN propagates the warm-up guard)
        both_warm = np.isfinite(tsum) & np.isfinite(esum)
        # off until both component windows are warm
        day[f"ofi_min{n}ev{t}s"] = np.where(both_warm, raw, np.nan)
    # the frame with all window columns added
    return day


def main():
    # session segments for the bucket assignment
    segments = H.load_segments()
    # the full list of windowed-variant column names (built once)
    variants = ([f"ofi_ev{n}" for n in EV_WINDOWS]
                + [f"ofi_t{t}s" for t in T_WINDOWS]
                + [f"ofi_min{n}ev{t}s" for (n, t) in HYBRIDS])
    # per-symbol pooled incremental rows, and per-(symbol,bucket) rows
    pooled_rows, bucket_rows = [], []
    # walk the race symbols
    for sym in SYMBOLS:
        # the symbol's feature store
        df = load_symbol(sym)
        # skip absent/thin
        if df is None or len(df) < 1000:
            print(f"{sym}: no/insufficient feature store -- skipped")
            continue
        # required base columns
        if OFI_BASE not in df.columns or BENCH not in df.columns:
            print(f"{sym}: missing {OFI_BASE}/{BENCH} -- rebuild the store")
            continue
        # chronological order within each day
        df = df.sort_values(["_day", "ts_exch"]).reset_index(drop=True)
        # trailing windows, PER DAY (never spanning overnight). Build per-day and
        # concat, preserving _day for the downstream label shift (avoids both the
        # groupby-apply deprecation and the include_groups=False column drop).
        df = pd.concat([add_windows(g) for _, g in df.groupby("_day", sort=False)],
                       ignore_index=True)
        # ---- forward label, per day ----
        fwd_mid = df.groupby("_day")["mid"].shift(-FWD_EVENTS)
        # forward return in bps
        df["y_fwd"] = (fwd_mid - df["mid"]) / df["mid"] * 1e4
        # ---- session bucket per row (from the day's segments) ----
        # date string "YYYY-MM-DD" from the "date=YYYY-MM-DD" day tag
        df["_date"] = df["_day"].str.replace("date=", "", regex=False)
        # bucket per row (rows on days without segments get None -> dropped)
        df["_bucket"] = [
            bucket_of(float(ts), segments[d]) if d in segments else None
            for ts, d in zip(df["ts_exch"], df["_date"])]
        # drop rows missing the LABEL, benchmark, or bucket only -- NOT on the
        # variant windows. Warm-up NaNs differ per variant (ev90 is NaN for 89
        # rows, ev5 for 4); dropping on all variants would score every variant on
        # the intersection of warm rows, biasing the comparison. Each variant is
        # filtered to its OWN warm rows inside score().
        df = df.dropna(subset=["y_fwd", BENCH, "_bucket"])
        # thin after cleaning
        if len(df) < 1000:
            print(f"{sym}: insufficient rows after labeling -- skipped")
            continue

        # score one subset (pooled or a bucket): incremental R^2 of each
        # windowed variant over the obi_1 benchmark. Each variant is scored on
        # its OWN warm rows (where the variant is finite), with the benchmark
        # recomputed on that same subset, so warm-up NaNs never bias the compare.
        def score(sub, label):
            # too thin to bother
            if len(sub) < 500:
                return
            # per-variant evaluation
            for v in variants:
                # this variant's warm rows within the subset
                vsub = sub[np.isfinite(sub[v].to_numpy())]
                # too thin after the warm-up filter -> skip this variant
                if len(vsub) < 500:
                    (pooled_rows if label == "ALL" else bucket_rows).append(
                        {"symbol": sym, "bucket": label, "variant": v,
                         "r2_obi": float("nan"), "incremental": float("nan")})
                    continue
                # winsorized label on THIS variant's rows
                y = winsor(vsub["y_fwd"].to_numpy())
                # chronological split on this subset
                m = np.arange(len(vsub)) < int(TRAIN_FRAC * len(vsub))
                # z-score helper (train stats only) on this subset
                def z(col):
                    # winsorized raw
                    x = winsor(vsub[col].to_numpy())
                    # train moments
                    mu, sd = x[m].mean(), x[m].std()
                    # degenerate
                    if sd == 0:
                        return None
                    # standardized
                    return (x - mu) / sd
                # benchmark + variant, both on the SAME warm rows
                zb = z(BENCH)
                zv = z(v)
                # either degenerate -> nan
                if zb is None or zv is None:
                    r2_obi, inc = float("nan"), float("nan")
                else:
                    # OBI-only baseline on this subset
                    r2_obi, _ = oos_r2(zb.reshape(-1, 1), y, m)
                    # OBI + variant blend
                    r2_blend, _ = oos_r2(np.column_stack([zb, zv]), y, m)
                    # incremental (the decision number), apples-to-apples
                    inc = r2_blend - r2_obi
                # store
                (pooled_rows if label == "ALL" else bucket_rows).append(
                    {"symbol": sym, "bucket": label, "variant": v,
                     "r2_obi": r2_obi, "incremental": inc})
        # pooled race
        score(df, "ALL")
        # per-bucket race
        for b in BUCKETS:
            score(df[df["_bucket"] == b], b)
        # progress
        print(f"{sym}: scored (pooled + buckets)", flush=True)

    # ---- REPORT ----
    pooled = pd.DataFrame(pooled_rows)
    byb = pd.DataFrame(bucket_rows)
    # nothing scored
    if pooled.empty:
        print("nothing scored -- is the feature store built with ofi_l1?")
        return
    print("\n=== WINDOWED-OFI incremental OOS R^2 over obi_1 (POOLED) ===")
    # variant x symbol matrix of incrementals
    piv = pooled.pivot_table(index="variant", columns="symbol",
                             values="incremental")
    # median across symbols (the decision column)
    piv["MEDIAN"] = piv.median(axis=1)
    # stable print order: events, then time, then hybrids
    piv = piv.reindex(variants)
    print(piv.to_string(float_format=lambda v: f"{v:+.5f}"))
    # the verdict against the gate
    best = piv["MEDIAN"].max()
    best_v = piv["MEDIAN"].idxmax()
    print(f"\nbest median incremental: {best_v} = {best:+.5f}  (gate: +{GATE})")
    if best > GATE:
        print("-> WINDOWED OFI clears the gate: wire this variant into the quotes.")
    else:
        print("-> no windowed variant clears the gate: OBI-only stands, now on a FAIR race.")
    # per-bucket medians (across symbols) for every variant
    if not byb.empty:
        print("\n=== per-BUCKET median incremental R^2 (across symbols) ===")
        # bucket x variant medians
        pb = byb.pivot_table(index="variant", columns="bucket",
                             values="incremental", aggfunc="median")
        # chronological bucket order, stable variant order
        pb = pb.reindex(variants)[list(BUCKETS)]
        print(pb.to_string(float_format=lambda v: f"{v:+.5f}"))
        print("\nREAD: a variant can fail pooled yet clear the gate in ONE bucket")
        print("(e.g. first15, where 50 events is seconds) -- that would justify a")
        print("bucket-gated signal rather than an always-on one.")
    # persist
    out = RESULTS_ROOT / "ofi_windows_horserace.csv"
    pd.concat([pooled, byb]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
