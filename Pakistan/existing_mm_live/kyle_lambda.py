# ============================================================================
# kyle_lambda.py -- estimate Kyle's lambda (price-impact / illiquidity) for all
# 38 names from the feature store. Kyle (1985): dP = lambda * Q, where Q is
# signed order flow and lambda is the price move per unit net flow. High lambda
# = thin/high-impact book (where defensive leaning should matter most).
# ============================================================================
# PRODUCTION CHOICES (each flagged; a naive tick-by-tick regression is biased):
#   1. TRUE trade signing: uses the feature store's signed_volume, which is built
#      from PSX's aggressor_side (no Lee-Ready approximation).
#   2. TIME BINNING: Kyle's lambda is an INTERVAL slope, not a tick slope (tick
#      level is dominated by bid-ask bounce). We bin into fixed BIN_MS windows,
#      sum signed volume per bin, take mid change per bin, regress.
#   3. PER-DAY estimation, DAY-AS-UNIT aggregation: one lambda per name per day,
#      then the name's lambda = median across days + day-as-unit SE. NOT one
#      pooled regression (which a single volatile day would dominate).
#   4. THROUGH-ORIGIN regression: Kyle's model has no intercept (zero net flow ->
#      zero expected price change). We fit dP = lambda*Q with no constant.
#   5. TWO FORMS: raw lambda (price units per share) is not comparable across
#      names at different price levels, so we also report a NORMALIZED lambda in
#      bps-of-mid per unit signed NOTIONAL -- the cross-name-comparable version.
#
# Run from existing_mm_live/ :  python kyle_lambda.py
# Writes: Capital Stake - Results/kyle_lambda_<stamp>.csv (per name)
#     and kyle_lambda_daily_<stamp>.csv (per name per day, for trend/scatter).
# ============================================================================

# paths + timing
from pathlib import Path
from datetime import datetime
import time
# numeric
import numpy as np
import pandas as pd
# dates
import run_legacy_mm as R

# feature-store root
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
# results root
RESULTS = Path("/Users/shazzak/Capital Stake - Results")
# the 38 names
NAMES = ['AKBL', 'ATRL', 'BAFL', 'BOP', 'DGKC', 'ENGROH', 'FFC', 'FNEL',
         'HASCOL', 'HBL', 'HUBC', 'KEL', 'LUCK', 'MARI', 'MEBL', 'MLCF',
         'NBP', 'NCPL', 'NML', 'NPL', 'NRL', 'OGDC', 'PACE', 'PAEL',
         'PIAHCLA', 'PIBTL', 'PIOC', 'PPL', 'PSO', 'PTC', 'SAZEW', 'SEARL',
         'SYS', 'THCCL', 'TOMCL', 'TPL', 'TRG', 'UBL']
# BIN size in ms. 60s is a standard Kyle horizon on thin books: long enough that
# a bin accumulates real net flow, short enough for many bins/day. Flagged as a
# choice -- sensitivity to it should be checked (30s / 300s) if lambda is used.
# bin sizes to compare (ms). Longer bins = less microstructure noise per bin
# (higher R2, more permanent-impact) but fewer bins/day (noisier daily slope).
# Running all three lets us check whether the RANKING is bin-size-stable.
BIN_SIZES = [60_000, 300_000, 900_000]   # 1min, 5min, 15min
# minimum bins in a day to estimate that day's lambda (else skip -- too few pts)
MIN_BINS = 8


# load one day's feature rows once (reused across all bin sizes)
def load_day(fs_path):
    # need signed_volume, mid, ts_exch
    df = pd.read_parquet(fs_path, columns=["ts_exch", "mid", "signed_volume"])
    # drop rows without a usable mid, sort by time
    df = df[(df["mid"] > 0)].sort_values("ts_exch").reset_index(drop=True)
    return df if len(df) >= 2 else None


# estimate one day's lambda at a given bin size from a preloaded day frame
def day_lambda(df, bin_ms):
    # assign each event to a fixed bin_ms bin (relative to the day's first ts)
    t0 = int(df["ts_exch"].iloc[0])
    b_idx = ((df["ts_exch"] - t0) // bin_ms).astype(int)
    # per bin: NET signed volume (sum) and price change = last mid - first mid
    g = df.groupby(b_idx)
    # net signed flow per bin (Q)
    Q = g["signed_volume"].sum()
    # mid at the end and start of each bin
    mid_last = g["mid"].last()
    mid_first = g["mid"].first()
    # price change per bin (dP). Using within-bin change pairs flow to move.
    dP = (mid_last - mid_first)
    # reference mid per bin (for the normalized/bps form)
    mid_ref = g["mid"].first()
    # assemble, drop empty/degenerate bins
    b = pd.DataFrame({"Q": Q, "dP": dP, "mid": mid_ref}).dropna()
    # need enough bins and some flow variation
    if len(b) < MIN_BINS or b["Q"].abs().sum() == 0:
        return None
    # ---- RAW lambda: through-origin OLS  dP = lambda * Q  ----
    # closed-form no-intercept slope: sum(Q*dP)/sum(Q^2)
    denom = float((b["Q"] ** 2).sum())
    if denom <= 0:
        return None
    lam_raw = float((b["Q"] * b["dP"]).sum() / denom)
    # R^2 of the through-origin fit
    ss_res = float(((b["dP"] - lam_raw * b["Q"]) ** 2).sum())
    ss_tot = float((b["dP"] ** 2).sum())  # through-origin -> vs 0, not mean
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # ---- NORMALIZED lambda: bps of mid per unit signed NOTIONAL ----
    # dP/mid*1e4 = bps; Q*mid = signed notional. Regress bps on notional so the
    # number is comparable across names at different price levels.
    mid_med = float(b["mid"].median())
    bps = (b["dP"] / mid_med) * 1e4
    notional = b["Q"] * mid_med
    dn = float((notional ** 2).sum())
    lam_bps_per_notional = (float((notional * bps).sum() / dn)
                            if dn > 0 else float("nan"))
    # return the day's estimates
    return {"lambda_raw": lam_raw, "lambda_bps_per_notional": lam_bps_per_notional,
            "r2": r2, "n_bins": int(len(b))}


def _bin_label(ms):
    # human label for a bin size
    return f"{ms // 60000}min"


def main():
    # trading dates
    all_dates = R.discover_dates()
    # per-(bin_size) per-day records
    daily = {bm: [] for bm in BIN_SIZES}
    # timing
    t0 = time.perf_counter()
    # walk names
    for si, sym in enumerate(NAMES, 1):
        for d in all_dates:
            fp = FS_ROOT / sym / f"date={d}.parquet"
            if not fp.exists():
                continue
            # load the day ONCE, reuse across all bin sizes
            try:
                df = load_day(fp)
            except Exception as e:
                print(f"  {sym} {d} LOAD ERROR {e!r}")
                continue
            if df is None:
                continue
            # estimate at each bin size from the same loaded frame
            for bm in BIN_SIZES:
                try:
                    r = day_lambda(df, bm)
                except Exception as e:
                    print(f"  {sym} {d} {_bin_label(bm)} ERROR {e!r}")
                    continue
                if r is None:
                    continue
                daily[bm].append({"symbol": sym, "date": d, **r})
        print(f"  [{si}/{len(NAMES)}] {sym}  {time.perf_counter()-t0:.1f}s")
    # stamp shared across the bin-size outputs
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # per-name ranking per bin size, kept for the cross-bin stability check
    rankings = {}
    for bm in BIN_SIZES:
        dd = pd.DataFrame(daily[bm])
        if len(dd) == 0:
            print(f"{_bin_label(bm)}: no estimates")
            continue
        # per-name day-as-unit aggregation
        rows = []
        for sym, g in dd.groupby("symbol"):
            ln = g["lambda_bps_per_notional"].to_numpy()
            lr = g["lambda_raw"].to_numpy()
            n = len(ln)
            se_n = (ln.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")
            rows.append({
                "symbol": sym, "n_days": n,
                "lambda_raw_median": float(np.median(lr)),
                "lambda_bps_per_notional_median": float(np.median(ln)),
                "lambda_bps_per_notional_se": se_n,
                "r2_median": float(np.median(g["r2"].to_numpy())),
                "bins_median": float(np.median(g["n_bins"].to_numpy()))})
        nm = pd.DataFrame(rows).sort_values(
            "lambda_bps_per_notional_median", ascending=False).reset_index(drop=True)
        # write per-bin-size files
        lab = _bin_label(bm)
        nm.to_csv(RESULTS / f"kyle_lambda_{lab}_{stamp}.csv", index=False)
        dd.to_csv(RESULTS / f"kyle_lambda_{lab}_daily_{stamp}.csv", index=False)
        # keep the ranking (symbol -> rank) for the stability comparison
        rankings[bm] = {r["symbol"]: i + 1 for i, r in nm.iterrows()}
        # print this bin size's ranking + its median R2 (the denoise check)
        med_r2 = float(nm["r2_median"].median())
        print(f"\n=== Kyle's lambda @ {lab} bins  (median R2 across names = {med_r2:.3f}) ===")
        print(f"{'rank':>4s} {'name':>9s} {'lam_bps/notl':>13s} {'se':>10s} "
              f"{'r2':>6s} {'bins':>6s} {'n_days':>7s}")
        for i, r in nm.iterrows():
            print(f"{i+1:>4d} {r['symbol']:>9s} "
                  f"{r['lambda_bps_per_notional_median']:>13.4g} "
                  f"{r['lambda_bps_per_notional_se']:>10.4g} "
                  f"{r['r2_median']:>6.3f} {r['bins_median']:>6.0f} "
                  f"{int(r['n_days']):>7d}")
    # ---- CROSS-BIN RANKING STABILITY (Spearman between bin sizes) ----
    # If the illiquidity RANKING is stable across bin sizes, the ordinal measure
    # is trustworthy regardless of horizon. If it shuffles, impact is horizon-
    # dependent and the bin choice matters.
    if len(rankings) >= 2:
        print("\n=== ranking stability across bin sizes (Spearman rho) ===")
        bms = [bm for bm in BIN_SIZES if bm in rankings]
        # common symbols across all
        common = set.intersection(*[set(rankings[bm]) for bm in bms])
        for i in range(len(bms)):
            for j in range(i + 1, len(bms)):
                a, b = bms[i], bms[j]
                xa = np.array([rankings[a][s] for s in common])
                xb = np.array([rankings[b][s] for s in common])
                # Spearman = Pearson on ranks (these ARE ranks already)
                rho = np.corrcoef(xa, xb)[0, 1]
                print(f"  {_bin_label(a)} vs {_bin_label(b)}: rho = {rho:.3f}  "
                      f"(n={len(common)})")
        print("  rho ~1 => same illiquidity ordering at all horizons (trust the rank)")
    print(f"\nwrote per-bin-size CSVs with stamp {stamp}")


if __name__ == "__main__":
    main()
