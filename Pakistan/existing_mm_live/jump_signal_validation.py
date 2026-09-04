# jump_signal_validation.py -- v2: does any LEADING microstructure feature spike
# BEFORE a price jump on PSX? Rewritten to read the FEATURE STORE's already-built,
# event-resolution features (ofi_l1, qdr_bid, qdr_ask, ewma_trade_flow, vpin,
# toxicity) instead of re-deriving them from sparse snapshots. The v1 bug was
# reading the periodic ob_snapshot table (coarse) and hand-rolling OFI/QDR badly;
# the feature store already computes these from the update-reconstructed book at
# event resolution, so this is both correct and fast (seconds, on-disk data).
#
# Jump label (SZ's design, 2026-08-20):
#   * PER-TICKER, SIGMA-based: a jump = |mid move over the next H seconds| >=
#     K_SIGMA * sigma_ticker, where sigma is a CAUSAL trailing estimate (the
#     feature store's realized_vol_bps, i.e. strictly-past info -- no peeking).
#   * SIGN-AGNOSTIC (up-jumps hurt the structurally-short book as much as down).
#   * RECORDED ALONGSIDE in spread-units, so we can later test whether sigma- or
#     spread-normalisation better predicts P&L damage (a maker cares about "moved
#     past my quote" = spread-relative, as much as "statistically unusual").
#   * primary horizon H=10s; cooldown so one episode = one event.
#
# Lift metric (fixes v1's near-zero-denominator 111x artifact):
#   Z-SCORE = (feature - day_median) / day_std, robust to signed mean-zero OFI.
#   A LEADING signal shows z >> 0 at short leads, DECAYING toward 0 at long leads.
#   Flat-across-leads = not a precursor (an artifact or a day-level constant).
#
# Output: timestamped per-jump-lead CSV + the verdict table. Run from
# existing_mm_live/:  python jump_signal_validation.py

# paths
from pathlib import Path
# timing + stamp
import time
from datetime import datetime
# frames + arrays
import pandas as pd
import numpy as np

# feature store + results
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ experiment knobs ------------------------------
# blowup/reject names (jumps kill us) + liquid anchors for contrast
SYMBOLS = ["BOP", "PACE", "FNEL", "UBL", "PPL"]
# regime windows (from the KSE-100 chart: melt-up, crash, both matter)
WINDOWS = [("RALLY", "2025-12-01", "2026-01-31"),
           ("CRASH", "2026-02-15", "2026-04-15")]
# jump threshold in trailing-sigma units (1.64 ~ 95th pct; optimisable later)
K_SIGMA = 1.64
# primary jump horizon (seconds) -- a jump that hurts a maker happens fast
PRIMARY_H = 10.0
# min gap between labelled jumps (seconds) -- one event per episode
JUMP_COOLDOWN = 300.0
# feature leads BEFORE the jump (seconds) -- the event-study x-axis
LEADS = [2.0, 5.0, 10.0, 20.0, 40.0, 60.0]
# candidate LEADING features already in the store (+ toxicity/vpin as slower
# gating features, and ewma_trade_flow as the "just volume" control)
FEATURES = ["ofi_l1", "qdr", "ewma_trade_flow", "toxicity", "vpin"]
# min events needed before we trust a per-day baseline
MIN_ROWS = 500
# ------------------------------------------------------------------------------


# load one symbol-day's feature rows; None if absent/too sparse
def load_day(sym, date):
    p = FS_ROOT / sym / f"date={date}.parquet"
    if not p.exists():
        return None
    # only the columns we need (vpin may be absent on older builds -> handle)
    want = ["ts_exch", "mid", "spread_bps", "ofi_l1", "qdr_bid", "qdr_ask",
            "ewma_trade_flow", "toxicity", "realized_vol_bps", "vpin"]
    have = pd.read_parquet(p).columns
    cols = [c for c in want if c in have]
    df = pd.read_parquet(p, columns=cols)
    if len(df) < MIN_ROWS:
        return None
    # combined QDR = the depleting side (a sweep hits ONE side); both >= 0
    df["qdr"] = np.maximum(df.get("qdr_bid", 0).fillna(0),
                           df.get("qdr_ask", 0).fillna(0))
    # |OFI| magnitude (direction not needed for a sign-agnostic precursor)
    df["ofi_l1"] = df["ofi_l1"].abs()
    # vpin may be missing -> neutral column so the feature loop never crashes
    if "vpin" not in df.columns:
        df["vpin"] = np.nan
    return df.sort_values("ts_exch").reset_index(drop=True)


# label sign-agnostic per-ticker jumps at horizon H (seconds)
def label_jumps(df, H):
    ts = df["ts_exch"].to_numpy()
    mid = df["mid"].to_numpy()
    rvol_bps = df["realized_vol_bps"].to_numpy()
    spread_bps = df["spread_bps"].to_numpy()
    # last row within H seconds ahead
    hi = np.searchsorted(ts, ts + H * 1000.0, side="right") - 1
    hi = np.clip(hi, 0, len(mid) - 1)
    # sign-agnostic forward move in bps
    move_bps = np.abs(mid[hi] - mid) / mid * 1e4
    # threshold = K_SIGMA * trailing sigma (realized_vol_bps is already in bps)
    sig = np.where(np.isfinite(rvol_bps) & (rvol_bps > 0), rvol_bps, np.nan)
    thresh = K_SIGMA * sig
    is_jump = np.isfinite(thresh) & (move_bps >= thresh)
    jumps = []
    last_t = -1e18
    for i in np.where(is_jump)[0]:
        if ts[i] - last_t >= JUMP_COOLDOWN * 1000.0:
            jumps.append((i, move_bps[i] / sig[i],
                          move_bps[i] / spread_bps[i] if spread_bps[i] > 0 else np.nan))
            last_t = ts[i]
    return jumps


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    win_dates = {}
    files = sorted((FS_ROOT / SYMBOLS[0]).glob("date=*.parquet"))
    dts = [f.stem.replace("date=", "") for f in files]
    for name, d0, d1 in WINDOWS:
        win_dates[name] = [d for d in dts if d0 <= d <= d1]
        print(f"{name}: {len(win_dates[name])} days ({d0}..{d1})")

    recs = []
    t0 = time.perf_counter()
    total = sum(len(v) for v in win_dates.values()) * len(SYMBOLS)
    sd = 0
    for wname, dates in win_dates.items():
        for date in dates:
            for sym in SYMBOLS:
                sd += 1
                df = load_day(sym, date)
                if df is None:
                    continue
                ts = df["ts_exch"].to_numpy()
                # per-day baselines: median + std (robust denominator)
                base = {}
                for f in FEATURES:
                    v = df[f].to_numpy()
                    v = v[np.isfinite(v)]
                    if len(v) == 0:
                        base[f] = (0.0, 1.0)
                    else:
                        med = float(np.median(v)); s = float(np.std(v))
                        base[f] = (med, s if s > 0 else 1.0)
                for j, mv_sig, mv_spr in label_jumps(df, PRIMARY_H):
                    for lead in LEADS:
                        k = np.searchsorted(ts, ts[j] - lead * 1000.0,
                                            side="right") - 1
                        if k < 0:
                            continue
                        row = {"window": wname, "date": date, "symbol": sym,
                               "lead_s": lead, "move_sigma": mv_sig,
                               "move_spread": mv_spr}
                        for f in FEATURES:
                            val = float(df[f].iloc[k])
                            med, s = base[f]
                            row[f"{f}_z"] = (val - med) / s
                        recs.append(row)
                if sd % 25 == 0 or sd == total:
                    print(f"  {sd}/{total}  elapsed {int(time.perf_counter()-t0)}s",
                          flush=True)

    df = pd.DataFrame(recs)
    out = RESULTS / f"jump_leads_{stamp}.csv"
    df.to_csv(out, index=False)
    days = df["date"].nunique() if len(df) else 0
    jpd = (len(df) / len(LEADS) / df.groupby(["date", "symbol"]).ngroups) if len(df) else 0
    print(f"\n{len(df)//len(LEADS) if len(df) else 0} jumps across {days} days "
          f"({jpd:.1f} jumps/symbol-day -- want ~1-3; v1 had 26)\n")
    print("=== Z-SCORE OF EACH FEATURE, N SECONDS BEFORE THE JUMP ===")
    print("(z=(feature-day_median)/day_std. LEADING: z>0 at short leads, DECAYING")
    print(" toward 0 at long leads. Flat near 0 = not a precursor.)\n")
    for wname in (df["window"].unique() if len(df) else []):
        w = df[df["window"] == wname]
        print(f"--- {wname} ({w.groupby(['date','symbol']).ngroups} symbol-days) ---")
        print(f"{'lead_s':>7s} | " + " ".join(f"{f[:10]:>10s}" for f in FEATURES))
        for lead in LEADS:
            x = w[w["lead_s"] == lead]
            if len(x) == 0:
                continue
            print(f"{lead:>7.0f} | " +
                  " ".join(f"{x[f'{f}_z'].median():>10.2f}" for f in FEATURES))
        print()
    print("READ: a column whose z RISES as lead->0 LEADS jumps -> build a gate on")
    print("it. Flat near 0 = no lead. Compare RALLY vs CRASH; ewma_trade_flow is")
    print("the 'just volume' control -- OFI/QDR must beat it to be worth anything.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
