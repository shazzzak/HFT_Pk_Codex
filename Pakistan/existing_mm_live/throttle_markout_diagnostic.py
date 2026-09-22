# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# throttle_markout_diagnostic.py
# ----------------------------------------------------------------------------
# GATE (before touching micro_mm.py): does a size throttle triggered by adverse
# OBI or adverse OFI actually dodge adverse selection by a TRADEABLE magnitude?
#
# The proposed live rule: when the book signal is against us, cut clip size
# (e.g. 0.5x) for a short window instead of pulling the quote. That only pays
# if fills taken while the signal is adverse mark out MEASURABLY worse than
# fills taken when it is neutral -- by more than a tick, net of the capture we
# forfeit by quoting smaller. This script measures that gap. It does NOT change
# any quoting; it reads the feature store and computes conditional markout.
#
# WHY THIS IS THE RIGHT GATE, NOT A LIVE BUILD:
#   - OFI-defensive (widen) already FAILED at n=197 (paired t: no config beat
#     OFF). A size throttle is a gentler lever than a widen, so it must clear a
#     LOWER bar -- but it is the SAME underlying question: is the adverse move
#     the signal predicts big enough to act on? Our decile work suggested the
#     move is often sub-tick ("real effect, untradeable magnitude" -- the
#     rejected dark-gate). So the prior is skeptical; measure before building.
#
# SIGNAL / MARKOUT SEMANTICS (must be exact so the verdict maps to the engine):
#   - obi_1  = (bq-aq)/(bq+aq) in [-1,+1]. obi_1 < 0 => ask-heavy (sellers
#     stacked) => our BID is the exposed side. obi_1 > 0 => bid-heavy => our
#     ASK is exposed. "Adverse for a maker" = the book is stacked against the
#     side we would be filled on.
#   - ofi_l1 = Cont-Kukanov L1 increment. ofi_l1 > 0 = net buying pressure
#     (our ASK exposed); ofi_l1 < 0 = net selling pressure (our BID exposed).
#   - markout_1000ms_bps = (mid(t+1s) - mid(t)) / mid(t) * 1e4 (signed, +=up).
#     Finest label in the store is 1000ms; the 100-500ms window is NOT in the
#     labels. 1s is a strict superset: no 1s signal => almost certainly no
#     tradeable 100-500ms signal (would have to appear AND decay inside 500ms).
#
# ADVERSE-MARKOUT CONVENTION (the key correctness point):
#   A resting BID that fills makes us LONG; we are hurt if mid then FALLS
#   (markout < 0). A resting ASK that fills makes us SHORT; we are hurt if mid
#   then RISES (markout > 0). To make "adverse selection" a single signed
#   number where MORE NEGATIVE = WORSE regardless of side, we orient markout by
#   the EXPOSED side the signal points to:
#     - signal points to BID exposed  -> adverse_mo =  markout (fall hurts -> neg)
#     - signal points to ASK exposed  -> adverse_mo = -markout (rise hurts -> neg)
#   Then compare adverse_mo in ADVERSE-signal rows vs NEUTRAL rows. If adverse
#   rows are meaningfully more negative, the throttle has something to dodge.
#
# OUTPUT: per-name and pooled markout gap (adverse - neutral) with DAY-AS-UNIT
#   error bars (never pool row-count SEs), a tick-size comparison (is the gap >
#   1 tick?), a Spearman of per-name gap vs Kyle's-lambda rank (does the effect
#   concentrate on thin names?), plus histograms and a scatter PNG.
#
# Usage:
#   python throttle_markout_diagnostic.py [--signal obi|ofi|both]
#                                         [--adverse-quantile 0.2]
#                                         [--neutral-band 0.1]
#                                         [--kyle CSV]
# ============================================================================

import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
# headless: write PNGs, no display on the run box
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- paths (match the project layout) ----
# feature store root: .../feature_store/{SYM}/date=YYYY-MM-DD/*.parquet
# Resolve this filesystem path through the canonical checkout/data configuration.
FEATURE_STORE = Path(
    str(_hft_paths.RESULTS_ROOT / 'feature_store'))
# results output
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS = Path(str(_hft_paths.RESULTS_ROOT))
# PSX tick size in price units (0.01 PKR) -- the tradeability yardstick
TICK = 0.01
# the markout horizon column we test (finest available in the store)
MO_COL = "markout_1000ms_bps"


# ---------------------------------------------------------------------------
# per-name computation: read the store, orient markout, split adverse/neutral
# ---------------------------------------------------------------------------
def analyse_name(sym, signal, adv_q, neut_band):
    # directory for this symbol
    d = FEATURE_STORE / sym
    # no data -> skip
    if not d.exists():
        return None
    # collect this name's day partitions
    parts = sorted(d.glob("date=*.parquet"))
    # nothing to read
    if not parts:
        return None
    # per-day accumulators: (day -> mean adverse_mo in adverse rows,
    #                         mean adverse_mo in neutral rows, n_adv, n_neu)
    day_rows = []
    # walk each day separately so error bars are DAY-as-unit (not row-pooled)
    for p in parts:
        # date string from the hive partition
        day = p.stem.replace("date=", "")
        # columns we need; read narrow for speed
        cols = ["mid", MO_COL, "obi_1", "ofi_l1", "spread_bps"]
        # read the day's features
        try:
            f = pd.read_parquet(p, columns=cols)
        except Exception:
            # a column may be absent in an older partition -> read all, subset
            f = pd.read_parquet(p)
            # keep only the columns we need that exist
            f = f[[c for c in cols if c in f.columns]]
        # need the markout label and the chosen signal present
        if MO_COL not in f.columns:
            continue
        # drop rows with no forward markout (end-of-day tail)
        f = f[np.isfinite(f[MO_COL])]
        # need a non-empty frame
        if len(f) == 0:
            continue
        # pick the signal series and the exposed-side orientation
        if signal == "obi":
            # need obi_1
            if "obi_1" not in f.columns:
                continue
            # signal value in [-1,+1]
            sig = f["obi_1"].astype(float)
            # exposed side: obi_1<0 (ask-heavy) -> BID exposed -> +markout adverse
            #               obi_1>0 (bid-heavy) -> ASK exposed -> -markout adverse
            # oriented adverse markout: sign(-obi_1)*markout gives "more neg=worse"
            # (bid exposed when obi_1<0: want +markout; -sign(obi_1)=+1 -> +mo)
            orient = -np.sign(sig)
        elif signal == "ofi":
            # need ofi_l1
            if "ofi_l1" not in f.columns:
                continue
            # signal value (unbounded); we rank within-name to threshold
            sig = f["ofi_l1"].astype(float)
            # exposed side: ofi_l1>0 buying pressure -> ASK exposed -> -markout;
            #               ofi_l1<0 selling pressure -> BID exposed -> +markout
            orient = -np.sign(sig)
        else:
            # unknown signal
            return None
        # oriented adverse markout: more NEGATIVE = worse adverse selection
        adverse_mo = orient * f[MO_COL].astype(float)
        # magnitude of the signal for thresholding (adverse = large |signal|)
        mag = sig.abs()
        # within-DAY quantile thresholds (per-day so intraday regime shifts
        # don't let one heavy day dominate the split)
        # adverse rows: |signal| in the top adv_q quantile
        hi = mag.quantile(1.0 - adv_q)
        # neutral rows: |signal| within a small band around 0 (bottom neut_band)
        lo = mag.quantile(neut_band)
        # adverse mask: strong signal AND oriented so it points at an exposed side
        adv_mask = (mag >= hi) & (orient != 0)
        # neutral mask: weak signal (near-balanced book / no flow)
        neu_mask = (mag <= lo)
        # need both buckets populated this day
        if adv_mask.sum() < 5 or neu_mask.sum() < 5:
            continue
        # day means of oriented adverse markout
        adv_mean = float(adverse_mo[adv_mask].mean())
        neu_mean = float(adverse_mo[neu_mask].mean())
        # record the day
        day_rows.append({
            "day": day, "sym": sym,
            "adv_mo_bps": adv_mean, "neu_mo_bps": neu_mean,
            # the per-day gap: adverse minus neutral (negative = throttle helps)
            "gap_bps": adv_mean - neu_mean,
            "n_adv": int(adv_mask.sum()), "n_neu": int(neu_mask.sum()),
            # median spread that day, to convert bps->ticks for tradeability
            "spread_bps": float(f["spread_bps"].median())
            if "spread_bps" in f.columns else np.nan,
            # median mid that day (bps<->price conversion for the tick test)
            "mid": float(f["mid"].median()),
        })
    # nothing usable
    if not day_rows:
        return None
    # per-name day-level frame
    return pd.DataFrame(day_rows)


# ---------------------------------------------------------------------------
# day-as-unit summary for one name: mean gap +/- SE over DAYS
# ---------------------------------------------------------------------------
def summarise(dfn):
    # number of days (the statistical unit)
    n = len(dfn)
    # mean per-day gap (bps): negative => adverse rows mark out worse => throttle
    # has something to dodge
    mean_gap = float(dfn["gap_bps"].mean())
    # day-as-unit standard error of the gap
    se_gap = float(dfn["gap_bps"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    # t-stat of the gap vs 0 (is the adverse-vs-neutral difference real?)
    t = mean_gap / se_gap if (se_gap and se_gap > 0) else np.nan
    # convert the gap to TICKS using this name's typical mid: bps of mid -> price
    # gap_price = mean_gap/1e4 * mid ; ticks = gap_price / TICK
    mid = float(dfn["mid"].median())
    gap_ticks = (mean_gap / 1e4 * mid) / TICK
    # return the per-name summary row
    return {
        "sym": dfn["sym"].iloc[0], "n_days": n,
        "mean_gap_bps": mean_gap, "se_gap_bps": se_gap, "t": t,
        "gap_ticks": gap_ticks,
        "median_spread_bps": float(dfn["spread_bps"].median()),
    }


def main():
    # ---- args ----
    ap = argparse.ArgumentParser()
    # which signal to test
    ap.add_argument("--signal", choices=["obi", "ofi", "both"], default="both")
    # adverse quantile: top X of |signal| = adverse rows
    ap.add_argument("--adverse-quantile", type=float, default=0.20)
    # neutral band: bottom X of |signal| = neutral rows
    ap.add_argument("--neutral-band", type=float, default=0.20)
    # optional Kyle's-lambda csv for the concentration test
    ap.add_argument("--kyle", type=str, default=None)
    args = ap.parse_args()
    # signals to run
    signals = ["obi", "ofi"] if args.signal == "both" else [args.signal]
    # discover names in the feature store
    names = sorted([d.name for d in FEATURE_STORE.iterdir() if d.is_dir()]) \
        if FEATURE_STORE.exists() else []
    # guard: no store
    if not names:
        print(f"no feature store at {FEATURE_STORE}")
        sys.exit(1)
    # run stamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # optional Kyle rank
    kyle = None
    if args.kyle:
        # read the comparable-lambda column
        k = pd.read_csv(args.kyle)
        # expected column name from kyle_lambda.py
        kc = "lam_bps_per_notional" if "lam_bps_per_notional" in k.columns \
            else ("lambda_bps_per_notional_median"
                  if "lambda_bps_per_notional_median" in k.columns else None)
        # build a name->rank map (1 = most illiquid) if we found the column
        if kc:
            k = k[["name", kc]].rename(columns={"name": "sym"}) \
                if "name" in k.columns else k.rename(columns={k.columns[0]: "sym"})
            k["kyle_rank"] = k[kc].rank(ascending=False, method="average")
            kyle = k[["sym", "kyle_rank", kc]]

    # ---- run each signal ----
    for signal in signals:
        print("=" * 70)
        print(f"THROTTLE DIAGNOSTIC -- signal = {signal.upper()}  "
              f"(adverse top {args.adverse_quantile:.0%} of |signal|, "
              f"neutral bottom {args.neutral_band:.0%})")
        print(f"markout horizon = {MO_COL} (finest label in the store)")
        print("=" * 70)
        # per-name summaries
        summaries = []
        # per-name day frames (kept for pooled + plots)
        all_days = []
        # walk names
        for i, sym in enumerate(names, 1):
            # per-day gaps for this name
            dfn = analyse_name(sym, signal, args.adverse_quantile,
                               args.neutral_band)
            # heartbeat every 5 names + timer-free (fast read)
            if i % 5 == 0 or i == len(names):
                print(f"  scanned {i}/{len(names)} names", flush=True)
            # skip names with no usable days
            if dfn is None or len(dfn) == 0:
                continue
            # keep the day frame
            all_days.append(dfn)
            # per-name day-as-unit summary
            summaries.append(summarise(dfn))
        # guard: nothing computed
        if not summaries:
            print("  no usable data for this signal")
            continue
        # per-name summary table, sorted by the gap (most-negative = throttle
        # helps most) first
        S = pd.DataFrame(summaries).sort_values("mean_gap_bps")
        # ---- POOLED day-as-unit test across ALL names ----
        # stack every (name,day) gap; the unit is the name-day
        pooled = pd.concat(all_days, ignore_index=True)
        # mean gap across all name-days
        pg = float(pooled["gap_bps"].mean())
        # day-as-unit SE across all name-days
        pse = float(pooled["gap_bps"].std(ddof=1) / np.sqrt(len(pooled)))
        # pooled t-stat
        pt = pg / pse if pse > 0 else np.nan
        # pooled gap in ticks (using the global median mid)
        pmid = float(pooled["mid"].median())
        pg_ticks = (pg / 1e4 * pmid) / TICK
        # ---- print the headline ----
        print(f"\n  POOLED (name-day as unit, n={len(pooled)}):")
        print(f"    mean adverse-minus-neutral markout gap = {pg:+.4f} bps  "
              f"(SE {pse:.4f}, t={pt:+.2f})")
        print(f"    that gap in ticks = {pg_ticks:+.3f} ticks")
        # interpret the tradeability, plainly
        if pt > -2:
            print("    VERDICT: gap not significantly negative -> adverse rows do "
                  "NOT mark out worse. A throttle has nothing to dodge. DO NOT "
                  "build it. (Consistent with the OFI-defensive n=197 failure.)")
        elif abs(pg_ticks) < 1.0:
            print("    VERDICT: gap is significant but < 1 tick -> real effect, "
                  "likely UNTRADEABLE magnitude (same shape as the rejected "
                  "dark-gate). A size throttle MIGHT clear the lower bar; only "
                  "worth a backtester test if the per-name concentration is "
                  "strong (see below).")
        else:
            print("    VERDICT: gap is significant AND > 1 tick -> potentially "
                  "tradeable. Justifies rebuilding 100-500ms labels and a "
                  "backtester test of the throttle (net of queue-forfeit cost).")
        # ---- per-name table (never drop the per-name axis) ----
        print("\n  per-name gap (most-negative first = throttle helps most):")
        print("    sym      n_days  gap_bps     SE       t     gap_ticks  spr_bps")
        for _, r in S.iterrows():
            print(f"    {r['sym']:>7s}  {r['n_days']:>5d}  {r['mean_gap_bps']:>+8.4f}  "
                  f"{r['se_gap_bps']:>6.4f}  {r['t']:>+5.1f}  {r['gap_ticks']:>+8.3f}  "
                  f"{r['median_spread_bps']:>6.1f}")
        # ---- optional: does the gap concentrate on thin names? ----
        if kyle is not None:
            # join per-name gap to Kyle rank
            m = S.merge(kyle, on="sym", how="inner")
            # need enough names
            if len(m) >= 4:
                # Spearman: does a MORE-NEGATIVE gap go with HIGHER illiquidity?
                # (thin names = high lambda = low rank number). We correlate gap
                # with the comparable lambda directly.
                from scipy import stats
                lam_col = [c for c in m.columns if "lam" in c.lower()][0]
                rho, pv = stats.spearmanr(m["mean_gap_bps"], m[lam_col])
                print(f"\n  concentration test (gap vs Kyle's lambda): "
                      f"Spearman rho={rho:+.3f} p={pv:.3f} (n={len(m)})")
                print("    negative rho => throttle helps MORE on thinner/"
                      "higher-impact names (would support a name-targeted rule)")
        # ---- outputs: CSV + plots ----
        # per-name summary csv
        scsv = RESULTS / f"throttle_diag_{signal}_pername_{stamp}.csv"
        S.to_csv(scsv, index=False)
        # all name-day gaps csv (for any re-analysis without re-reading the store)
        dcsv = RESULTS / f"throttle_diag_{signal}_daily_{stamp}.csv"
        pooled.to_csv(dcsv, index=False)
        # histogram of per-name-day gaps (visualise the distribution + zero line)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pooled["gap_bps"], bins=60)
        ax.axvline(0.0, color="k", lw=1, ls="--")
        ax.axvline(pg, color="r", lw=1.5, label=f"mean {pg:+.4f} bps")
        ax.set_xlabel("per-(name,day) adverse-minus-neutral markout gap (bps)")
        ax.set_ylabel("count of name-days")
        ax.set_title(f"{signal.upper()} throttle diagnostic: markout gap "
                     f"(t={pt:+.2f})")
        ax.legend()
        fig.tight_layout()
        hpng = RESULTS / f"throttle_diag_{signal}_hist_{stamp}.png"
        fig.savefig(hpng, dpi=130)
        plt.close(fig)
        print(f"\n  wrote {scsv}")
        print(f"  wrote {dcsv}")
        print(f"  wrote {hpng}")


if __name__ == "__main__":
    main()
