"""
feature_incremental_value.py

For every candidate feature, does it add anything to OBI for predicting markout_5000ms_bps?
Same test that killed micro_dev, run across the whole candidate set:
  per masked day, partial correlation of feature with markout AFTER obi_1 is removed.
Ranks features by |mean partial-rho| (day-as-unit). Also reports each feature's MARGINAL rho
(feature vs markout alone) for contrast -- a feature can look good marginally yet add nothing.

Reuses the validated solver in residualize_micro_on_obi.py.
"""

# stdlib + numeric/plot + the tested residualization module (solver, config) + wiring
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import residualize_micro_on_obi as R
import feature_store_wiring as W

# ---------------- CONFIG ----------------

# anchor feature and label
ANCHOR, Y = "obi_1", "markout_5000ms_bps"
# spread column for the crossed-row mask
SPREAD = "spread_bps"
# candidate features to test for incremental value over OBI
CANDIDATES = ["obi_5", "obi_deep", "ofi_l1", "ewma_trade_flow", "signed_volume",
              "qdr_bid", "qdr_ask", "toxicity", "vpin", "spread_z",
              "realized_vol_bps", "micro_dev_bps"]   # micro_dev included as a known-dead control
# only these symbols (scopes the mask to the right universe -- fixes the earlier over-broad scan)
SYMBOLS = ["PPL", "UBL"]
# rank (Spearman) vs linear
RANK_MODE = True
# crossed-spread day fraction above which a day is dropped
CROSSED_DAY_FRAC = 0.05
# output dir
OUT_DIR = "/Users/shazzak/Capital Stake - Results/markout_validation"


# ---------------- per (file, feature) moment query ----------------

def moment_sql(path, feat):
    # x1 = OBI, x2 = candidate feature, y = markout
    if RANK_MODE:
        # rank each within the day for a Spearman-style partial correlation
        x1e = f"CAST(RANK() OVER (ORDER BY {ANCHOR}) AS DOUBLE)"
        x2e = f"CAST(RANK() OVER (ORDER BY {feat})   AS DOUBLE)"
        ye  = f"CAST(RANK() OVER (ORDER BY {Y})      AS DOUBLE)"
    else:
        # raw values
        x1e, x2e, ye = ANCHOR, feat, Y
    # masked base rows (non-null, spread>0), then rank, then the 10 cross-moment sums
    return f"""
    WITH base AS (
        SELECT {ANCHOR} AS a, {feat} AS f, {Y} AS y
        FROM read_parquet('{path}')
        WHERE {ANCHOR} IS NOT NULL AND {feat} IS NOT NULL AND {Y} IS NOT NULL AND {SPREAD} > 0
    ),
    r AS (
        SELECT {x1e.replace(ANCHOR,'a')} AS x1,
               {x2e.replace(feat,'f')}   AS x2,
               {ye.replace(Y,'y')}       AS y
        FROM base
    )
    SELECT COUNT(*) n,
           SUM(x1) sx1, SUM(x2) sx2, SUM(y) sy,
           SUM(x1*x1) sx1x1, SUM(x2*x2) sx2x2, SUM(x1*x2) sx1x2,
           SUM(x1*y) sx1y, SUM(x2*y) sx2y, SUM(y*y) syy,
           NULL vol
    FROM r
    """


# ---------------- marginal correlation of feature with y, from the sums ----------------

def marginal_rho(m):
    # Pearson-style correlation of x2 (feature/rank) with y (markout/rank)
    n = m["n"]
    # covariance and variances from the raw sums
    cov = n * m["sx2y"] - m["sx2"] * m["sy"]
    vx = n * m["sx2x2"] - m["sx2"] ** 2
    vy = n * m["syy"] - m["sy"] ** 2
    # guard degenerate variance
    if vx <= 0 or vy <= 0:
        return np.nan
    # correlation
    return cov / np.sqrt(vx * vy)


# ---------------- lightweight bad-day set (crossed-spread days, per target symbol) ----------------

def build_bad_days(con):
    # (symbol,date) with too many crossed-spread rows -> drop
    bad = set()
    # only scan the target symbols' feature-store files
    for sym, dt, path in W.enumerate_partitions():
        # skip other symbols
        if sym not in SYMBOLS:
            continue
        # fraction of crossed rows that day
        try:
            frac = con.execute(f"SELECT AVG(CASE WHEN {SPREAD} <= 0 THEN 1.0 ELSE 0.0 END) f "
                               f"FROM read_parquet('{path}')").df().iloc[0]["f"]
            # flag heavily-crossed days
            if frac is not None and frac > CROSSED_DAY_FRAC:
                bad.add((sym, dt))
        except Exception:
            pass
    print(f"  crossed-spread days excluded: {len(bad)}")
    return bad


# ---------------- main ----------------

def main():
    # lazy duckdb
    import duckdb
    con = duckdb.connect()
    # build the day mask
    print("building crossed-spread mask...")
    bad = build_bad_days(con)
    # per (symbol, feature): collect per-day partial-rho and marginal-rho
    recs = []
    # iterate target partitions
    for sym, dt, path in W.enumerate_partitions():
        # scope + mask
        if sym not in SYMBOLS or (sym, dt) in bad:
            continue
        # test each candidate feature on this day
        for feat in CANDIDATES:
            # compute this day's moments for (obi, feat, y)
            try:
                m = con.execute(moment_sql(path, feat)).df().iloc[0].to_dict()
            except Exception:
                continue
            # solve for partial-rho of feat given obi
            out = R.solve_from_moments(m)
            # skip unusable days
            if not out:
                continue
            # record partial (incremental) and marginal correlations
            recs.append(dict(symbol=sym, date=dt, feature=feat,
                             partial_rho=out["pcorr"], marginal_rho=marginal_rho(m)))
    # assemble
    df = pd.DataFrame(recs)
    # guard empty
    if df.empty:
        print("no usable data"); return

    # per symbol: day-as-unit summary + ranking + plot
    for sym, g in df.groupby("symbol"):
        # aggregate across days per feature
        summ = (g.groupby("feature")
                  .agg(partial_mean=("partial_rho", "mean"),
                       partial_se=("partial_rho", lambda s: s.std() / np.sqrt(len(s))),
                       marginal_mean=("marginal_rho", "mean"),
                       n_days=("partial_rho", "count"))
                  .reset_index())
        # rank by absolute incremental value over OBI
        summ["abs_partial"] = summ["partial_mean"].abs()
        summ = summ.sort_values("abs_partial", ascending=False)
        # print the ranked table
        print(f"\n[{sym}] incremental value over OBI (mode={'rank' if RANK_MODE else 'linear'}), "
              f"ranked by |partial rho|:")
        print(f"   {'feature':18s} {'partial|obi':>12s} {'+/-SE':>7s} {'marginal':>9s}  n")
        for _, r in summ.iterrows():
            print(f"   {r['feature']:18s} {r['partial_mean']:>+12.3f} {r['partial_se']:>7.3f} "
                  f"{r['marginal_mean']:>+9.3f}  {int(r['n_days'])}")

        # figure: partial (incremental) vs marginal per feature
        fig, ax = plt.subplots(figsize=(10, 7))
        y = np.arange(len(summ))
        # marginal as faint bars (what the feature looks like alone)
        ax.barh(y + 0.2, summ["marginal_mean"], height=0.4, color="lightgray", label="marginal (alone)")
        # partial as solid bars with day-as-unit SE (what it adds over OBI)
        ax.barh(y - 0.2, summ["partial_mean"], height=0.4, xerr=summ["partial_se"],
                color="tab:blue", label="partial | OBI (incremental)")
        # zero line
        ax.axvline(0, color="red", ls="--", lw=1)
        # feature labels
        ax.set_yticks(y); ax.set_yticklabels(summ["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("correlation with markout_5000ms")
        ax.set_title(f"{sym}: does each feature ADD to OBI?  (solid = incremental over OBI)")
        ax.legend()
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"feature_incremental_value_{sym}.png")
        plt.savefig(out, dpi=130); plt.close(fig)
        print(f"[{sym}] saved {out}")


# entry point
if __name__ == "__main__":
    main()
