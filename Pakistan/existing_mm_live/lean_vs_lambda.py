# ============================================================================
# lean_vs_lambda.py -- PRE-REGISTERED payoff test for the microprice defensive
# lean. Question: does leaning defensively (lambda<0) pay SPECIFICALLY on the
# thin / high-impact names (high Kyle's lambda)? If yes -> structural PSX thesis
# holds. If ~0 -> the lean's effect is unrelated to book depth.
# ============================================================================
# PRE-REGISTERED DESIGN (fixed BEFORE looking at Run A, to avoid p-hacking):
#   - metric per name: lean_benefit = best_lean_net_bps - mid_net_bps, where
#       mid_net_bps  = the lambda=None (mid) baseline, portfolio net over the run
#       best_lean    = the SINGLE best net_bps among lambda in {-1.5..-5.0}
#     (portfolio = all buckets, all days for that name; day-weighted by notional)
#   - illiquidity: the 15min Kyle's lambda rank (chosen because 15min had the
#       best R2 and the ranking is horizon-stable, rho~0.99)
#   - ONE test: Spearman rho between lean_benefit and Kyle's-lambda rank.
#       Spearman (not Pearson) because Kyle R2~0.05 -> cardinal lambda is noisy
#       but the RANK is trustworthy. H1: rho > 0. Report rho, p, and the scatter.
#   - NO subset searches, NO alternative "best lean" definitions after the fact.
#
# Usage:
#   python lean_vs_lambda.py <runA_PERNAME.csv> <kyle_lambda_15min.csv>
# Writes: lean_vs_lambda_<stamp>.csv (per-name table) + .png (scatter).
# ============================================================================

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
# headless backend (no display on the run box)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("/Users/shazzak/Capital Stake - Results")


def main():
    # ---- inputs ----
    if len(sys.argv) >= 3:
        pername_csv = Path(sys.argv[1])
        kyle_csv = Path(sys.argv[2])
    else:
        print("usage: python lean_vs_lambda.py <runA_PERNAME.csv> "
              "<kyle_lambda_15min.csv>")
        sys.exit(1)
    # per-name Run A rows: (date, symbol, ofi_window, ofi_thresh, bucket, net_bps,
    # net_pkr, capture_pkr, markout_pkr, liq_pkr, fee_pkr, markout_bps,
    # opened_notional, fills). ofi_window encodes the lambda as "OFF lam-3.0"/"OFF".
    pn = pd.read_csv(pername_csv)
    # keep only OFF-family rows (the lambda sweep lives on OFF); Run A has only
    # these anyway, but guard in case a combined file is passed.
    pn = pn[pn["ofi_window"].astype(str).str.startswith("OFF")].copy()

    # parse the lambda out of the window label: "OFF" -> baseline (mid, lam=0),
    # "OFF lam-3.0" -> -3.0
    def parse_lam(w):
        w = str(w)
        if w == "OFF":
            return 0.0            # the mid baseline
        if w.startswith("OFF lam"):
            try:
                return float(w.replace("OFF lam", ""))
            except ValueError:
                return np.nan
        return np.nan
    pn["lam"] = pn["ofi_window"].map(parse_lam)

    # ---- per (symbol, lam) PORTFOLIO net_bps: sum net_pkr and opened_notional
    # across ALL days and buckets for that name+lambda, then bps = 1e4*net/on.
    # (notional-weighted, matching how the sweep reports portfolio net_bps.)
    grp = pn.groupby(["symbol", "lam"]).agg(
        net_pkr=("net_pkr", "sum"),
        opened_notional=("opened_notional", "sum")).reset_index()
    grp["net_bps"] = np.where(grp["opened_notional"] > 0,
                              1e4 * grp["net_pkr"] / grp["opened_notional"],
                              np.nan)

    # ---- per name: mid baseline (lam=0) and BEST negative-lambda net_bps ----
    rows = []
    for sym, g in grp.groupby("symbol"):
        # baseline (mid)
        base = g.loc[g["lam"] == 0.0, "net_bps"]
        if base.empty or not np.isfinite(base.iloc[0]):
            continue
        mid_bps = float(base.iloc[0])
        # negative-lambda rows
        neg = g[g["lam"] < 0.0]
        if neg.empty:
            continue
        # the single best lean (pre-registered: best net_bps among the leans)
        best_row = neg.loc[neg["net_bps"].idxmax()]
        best_bps = float(best_row["net_bps"])
        best_lam = float(best_row["lam"])
        rows.append({"symbol": sym, "mid_net_bps": mid_bps,
                     "best_lean_net_bps": best_bps, "best_lam": best_lam,
                     "lean_benefit_bps": best_bps - mid_bps})
    lb = pd.DataFrame(rows)
    if len(lb) == 0:
        print("no per-name lean data found -- check the PERNAME csv path/content")
        sys.exit(1)

    # ---- join Kyle's lambda (15min) and rank ----
    ky = pd.read_csv(kyle_csv)
    # expected column: lambda_bps_per_notional_median (the comparable measure)
    kcol = "lambda_bps_per_notional_median"
    if kcol not in ky.columns:
        print(f"expected column '{kcol}' not in kyle csv; has {list(ky.columns)}")
        sys.exit(1)
    ky = ky[["symbol", kcol]].copy()
    # illiquidity RANK: 1 = most illiquid (highest lambda)
    ky["kyle_rank"] = ky[kcol].rank(ascending=False, method="average")
    # merge
    m = lb.merge(ky, on="symbol", how="inner")
    n = len(m)
    if n < 4:
        print(f"only {n} names matched -- too few for a meaningful test")
        # still write what we have
    # ---- THE pre-registered test: Spearman(lean_benefit, Kyle's lambda) ----
    # positive rho => lean helps the thin/high-impact names (thesis holds)
    rho, p = stats.spearmanr(m["lean_benefit_bps"], m[kcol])
    # also report using the rank directly (same thing, sanity)
    print("=" * 64)
    print("PRE-REGISTERED TEST: does the defensive lean pay on thin names?")
    print("=" * 64)
    print(f"n names                 : {n}")
    print(f"Spearman rho            : {rho:+.3f}")
    print(f"p-value                 : {p:.4f}")
    print(f"mean lean_benefit (bps) : {m['lean_benefit_bps'].mean():+.3f}")
    print(f"names lean HELPS (>0)    : {(m['lean_benefit_bps'] > 0).sum()} / {n}")
    print(f"median best_lam          : {m['best_lam'].median():.1f}")
    # verdict, stated plainly
    if p < 0.05 and rho > 0:
        print("\nVERDICT: positive & significant -> lean pays MORE on thinner "
              "names. Structural thesis SUPPORTED (ordinally).")
    elif p < 0.05 and rho < 0:
        print("\nVERDICT: negative & significant -> lean pays LESS on thinner "
              "names (opposite of thesis).")
    else:
        print("\nVERDICT: not significant -> no reliable link between the lean's "
              "benefit and book illiquidity. Consistent with the weak-impact "
              "(low Kyle R2) picture.")
    # honest caveat: is the mean benefit even positive?
    tb, pb = stats.ttest_1samp(m["lean_benefit_bps"], 0.0)
    print(f"\n(is the lean benefit positive AT ALL? mean-vs-0 t={tb:+.2f} "
          f"p={pb:.4f} -- 'best-of-8 leans' is upward-biased, so expect a small "
          f"positive even under the null; the Spearman is the real test.)")

    # ---- outputs ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    mout = RESULTS / f"lean_vs_lambda_{stamp}.csv"
    m.sort_values("kyle_rank").to_csv(mout, index=False)
    # scatter: x = Kyle's lambda (log, illiquidity), y = lean benefit
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(m[kcol], m["lean_benefit_bps"], s=40)
    for _, r in m.iterrows():
        ax.annotate(r["symbol"], (r[kcol], r["lean_benefit_bps"]),
                    fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Kyle's lambda (15min, bps per signed notional) -- higher = thinner")
    ax.set_ylabel("lean benefit (best negative-lambda net_bps - mid net_bps)")
    ax.set_title(f"Defensive lean benefit vs illiquidity\n"
                 f"Spearman rho={rho:+.3f}, p={p:.3f}, n={n}")
    fig.tight_layout()
    pout = RESULTS / f"lean_vs_lambda_{stamp}.png"
    fig.savefig(pout, dpi=130)
    print(f"\nwrote {mout}")
    print(f"wrote {pout}")


if __name__ == "__main__":
    main()
