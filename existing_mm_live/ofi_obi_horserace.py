# ofi_obi_horserace.py -- DECISION GATE: does OFI (or an OBI/OFI blend) predict
# the next mid-move better than OBI alone? This is Step-2 research. It reads the
# feature store, builds a forward-return label, and horse-races the directional
# signals per symbol. NOTHING is wired into the live engine here -- this analysis
# decides whether anything SHOULD be (Step B).
#
# WHY THIS EXISTS: the microprice lesson. A signal well-supported in the
# literature (Cont-Kukanov OFI) may or may not help on a given exchange. Wiring
# it into micro_mm.py before this test is exactly the "theorize ahead of
# empirics" failure mode. So: measure predictive power FIRST, per symbol, with a
# train/test split, then decide.
#
# WHAT IT MEASURES, per symbol:
#   * univariate predictive R^2 of each signal on the forward mid-return:
#       obi_1, obi_5, obi_deep, ofi_l1, ofi_5, ofi_deep
#   * the sign of each coefficient (does the signal point the RIGHT way?)
#   * blends: OBI+OFI multivariate R^2, and the incremental R^2 OFI adds ON TOP
#     of OBI (the number that actually matters -- does OFI add anything NEW?)
#   * everything on a HELD-OUT test set (fit on the first 70% of each day's
#     events, score on the last 30%) so the R^2 is out-of-sample, not fitted.
#
# HOW TO READ IT: if OFI's incremental R^2 over OBI is materially positive and
# its coefficient sign is stable across symbols, OFI belongs in the skew and the
# blend weight follows from the fitted coefficients. If it adds ~0 (like
# micro_dev did), leave the skew OBI-only.
#
# Run from existing_mm_live/:  python3 ofi_obi_horserace.py

# paths
from pathlib import Path
# arrays + frames
import numpy as np
import pandas as pd

# feature-store location (same convention as build_feature_store.py)
RESULTS_ROOT = Path("/Users/shazzak/Capital Stake - Results")
# feature_store/{symbol}/date=*/*.parquet
FS_ROOT = RESULTS_ROOT / "feature_store"

# ------------------------------ config ---------------------------------------
# symbols to test (edit to the names whose feature store exists)
SYMBOLS = ["PPL", "UBL", "MLCF", "TRG", "BOP"]
# the directional signals to horse-race (must exist as feature-store columns)
SIGNALS = ["obi_1", "obi_5", "obi_deep", "ofi_l1", "ofi_5", "ofi_deep"]
# forward horizon for the label, in EVENTS (predict the mid this many rows ahead)
FWD_EVENTS = 20
# train fraction (fit on the first this-much of each day, score on the rest)
TRAIN_FRAC = 0.70
# winsorize signals/label at this percentile each tail (robustness to outliers)
WINSOR_PCT = 0.5
# -----------------------------------------------------------------------------


# load one symbol's full feature store (all dates) as a single frame
def load_symbol(sym):
    # the per-symbol subtree
    d = FS_ROOT / sym
    # nothing there -> None
    if not d.exists():
        return None
    # all daily parquet files
    files = sorted(d.glob("date=*/*.parquet")) or sorted(d.glob("date=*.parquet"))
    # none found
    if not files:
        return None
    # read + concatenate, tagging the source day so labels never cross days
    parts = []
    # one frame per day
    for f in files:
        # read the day
        df = pd.read_parquet(f)
        # tag the day (label horizon must not span the overnight gap)
        df["_day"] = f.stem
        # collect
        parts.append(df)
    # the full symbol frame
    return pd.concat(parts, ignore_index=True)


# winsorize a series to [p, 100-p] percentiles
def winsor(s):
    # lower / upper clips
    lo, hi = np.nanpercentile(s, WINSOR_PCT), np.nanpercentile(s, 100 - WINSOR_PCT)
    # clipped
    return np.clip(s, lo, hi)


# out-of-sample R^2 for a linear fit of X (columns) on y, with a train/test split
def oos_r2(X, y, train_mask):
    # design matrices with an intercept column
    Xtr = np.column_stack([np.ones(train_mask.sum()), X[train_mask]])
    Xte = np.column_stack([np.ones((~train_mask).sum()), X[~train_mask]])
    # targets
    ytr, yte = y[train_mask], y[~train_mask]
    # least-squares fit on train
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    # predict on test
    yhat = Xte @ beta
    # test-set SS
    ss_res = np.sum((yte - yhat) ** 2)
    # total SS around the TRAIN mean (honest OOS baseline)
    ss_tot = np.sum((yte - ytr.mean()) ** 2)
    # OOS R^2 (can be negative if worse than the mean)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # coefficients excluding the intercept
    return r2, beta[1:]


def main():
    # per-symbol result rows
    rows = []
    # walk each symbol
    for sym in SYMBOLS:
        # load its feature store
        df = load_symbol(sym)
        # skip if absent
        if df is None or len(df) < 1000:
            print(f"{sym}: no/insufficient feature store -- skipped")
            continue
        # ---- build the FORWARD mid-return label, per day (no overnight span) ----
        # forward mid = mid FWD_EVENTS rows ahead, within the same day
        df = df.sort_values(["_day", "ts_exch"]).reset_index(drop=True)
        # grouped forward shift so the label never crosses a day boundary
        fwd_mid = df.groupby("_day")["mid"].shift(-FWD_EVENTS)
        # forward return in bps (the thing we are trying to predict)
        df["y_fwd"] = (fwd_mid - df["mid"]) / df["mid"] * 1e4
        # drop rows with no label (end-of-day tail) or missing signals
        need = ["y_fwd"] + [s for s in SIGNALS if s in df.columns]
        df = df.dropna(subset=need)
        # not enough after cleaning
        if len(df) < 1000:
            print(f"{sym}: insufficient rows after labeling -- skipped")
            continue
        # winsorize the label
        y = winsor(df["y_fwd"].to_numpy())
        # chronological train/test split (first TRAIN_FRAC = train)
        n = len(df)
        # boolean train mask (time-ordered, no shuffle -> honest OOS)
        train_mask = np.arange(n) < int(TRAIN_FRAC * n)
        # ---- univariate OOS R^2 + coefficient sign for each signal ----
        uni = {}
        # each candidate signal
        for s in SIGNALS:
            # skip signals not present
            if s not in df.columns:
                continue
            # standardized, winsorized signal
            x = winsor(df[s].to_numpy())
            # z-score on the TRAIN portion only (no test leakage)
            mu, sd = x[train_mask].mean(), x[train_mask].std()
            # guard degenerate
            if sd == 0:
                continue
            # standardized column
            xz = (x - mu) / sd
            # OOS R^2 + coefficient
            r2, beta = oos_r2(xz.reshape(-1, 1), y, train_mask)
            # store R^2 (in bps^2 units of the label) and coef sign/size
            uni[s] = (r2, float(beta[0]))
        # ---- blends: OBI-only vs OBI+OFI, and OFI's INCREMENTAL R^2 ----
        # helper to build a standardized multi-column matrix
        def zmat(cols):
            # each column standardized on train
            mats = []
            # per column
            for c in cols:
                # winsorized values
                x = winsor(df[c].to_numpy())
                # train stats
                mu, sd = x[train_mask].mean(), x[train_mask].std()
                # skip degenerate
                if sd == 0:
                    return None
                # standardized
                mats.append((x - mu) / sd)
            # stacked design
            return np.column_stack(mats)
        # OBI-only baseline (best OBI depth = obi_5, the cleanest)
        obi_cols = [c for c in ["obi_1", "obi_5"] if c in df.columns]
        # OBI+OFI blend (add the best OFI depth = ofi_l1 + ofi_5)
        blend_cols = obi_cols + [c for c in ["ofi_l1", "ofi_5"] if c in df.columns]
        # OBI-only OOS R^2
        Xo = zmat(obi_cols)
        r2_obi = oos_r2(Xo, y, train_mask)[0] if Xo is not None else np.nan
        # blend OOS R^2
        Xb = zmat(blend_cols)
        r2_blend = oos_r2(Xb, y, train_mask)[0] if Xb is not None else np.nan
        # the decision number: incremental R^2 OFI adds ON TOP of OBI
        incr = (r2_blend - r2_obi) if (np.isfinite(r2_blend)
                                       and np.isfinite(r2_obi)) else np.nan
        # record the symbol's row
        row = {"symbol": sym, "n": n,
               "r2_obi_only": r2_obi, "r2_obi+ofi": r2_blend,
               "ofi_incremental_r2": incr}
        # add univariate R^2 + coef sign per signal
        for s, (r2, b) in uni.items():
            row[f"{s}_r2"] = r2
            row[f"{s}_coef"] = b
        # collect
        rows.append(row)
        # progress line
        print(f"{sym}: OBI-only R2={r2_obi:.5f}  OBI+OFI R2={r2_blend:.5f}  "
              f"OFI incremental={incr:+.5f}")

    # assemble + save
    res = pd.DataFrame(rows)
    # output path
    out = RESULTS_ROOT / "ofi_obi_horserace.csv"
    # write
    res.to_csv(out, index=False)
    # ---- verdict summary ----
    print("\n=== HORSE-RACE VERDICT ===")
    # only if we have results
    if len(res):
        # univariate R^2 table
        uni_cols = [c for c in res.columns if c.endswith("_r2")
                    and c != "ofi_incremental_r2"]
        print("\nUnivariate out-of-sample R^2 (higher = more predictive):")
        print(res[["symbol"] + uni_cols].to_string(index=False))
        # coefficient-sign stability (a signal must point the same way everywhere)
        coef_cols = [c for c in res.columns if c.endswith("_coef")]
        print("\nCoefficient signs (must be STABLE across symbols to be usable):")
        print(res[["symbol"] + coef_cols].to_string(index=False))
        # the decision line
        print("\nOFI incremental R^2 over OBI (the DECISION number):")
        print(res[["symbol", "r2_obi_only", "r2_obi+ofi",
                   "ofi_incremental_r2"]].to_string(index=False))
        # median incremental across symbols
        med_incr = res["ofi_incremental_r2"].median()
        # the plain-language verdict
        print(f"\nmedian OFI incremental R^2 = {med_incr:+.5f}")
        # interpretation
        if med_incr > 0.0005:
            print("-> OFI adds meaningful predictive power beyond OBI. Worth")
            print("   wiring into the skew as a blend (Step B), behind a toggle.")
        else:
            print("-> OFI adds ~no predictive power beyond OBI (like micro_dev).")
            print("   Leave the skew OBI-only; do NOT wire OFI in.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
