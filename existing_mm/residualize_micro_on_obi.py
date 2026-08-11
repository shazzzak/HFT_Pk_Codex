"""
residualize_micro_on_obi.py

Question: after OBI is accounted for, does micro_dev add any predictive information about
markout_5000ms_bps -- especially in the high-vol regime where micro is at its best?

Method (per masked (symbol, day)):
  regress markout on OBI, then on OBI + micro_dev, and report micro_dev's PARTIAL correlation
  with markout given OBI (= the part of micro_dev orthogonal to OBI, correlated with markout).
  Everything is computed from per-day cross-moment SUMS in SQL (no row transfer); the tiny
  normal equations are solved in numpy. Days are then split by vol tercile.

Halt masking (see notes in the accompanying message):
  * drop rows with spread_bps <= 0 (crossed/auction/halt artifact),
  * drop whole (symbol, date) that are suspended_all_day (ob_snapshot),
  * drop whole days where crossed-spread rows exceed a threshold (catches the UBL -1400 bug),
  * drop user-supplied market-wide halt dates (from misc; fill after discovery).
  Intraday partial-halt phase join is a Tier-2 refinement, NOT included here.

RANK_MODE=True -> Spearman partial correlation (default, matches prior analysis).
RANK_MODE=False -> linear Pearson.
"""

# stdlib + numeric/plot + wiring
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import feature_store_wiring as W

# ---------------- CONFIG ----------------

# feature-store column roles
X1, X2, Y = "obi_1", "micro_dev_bps", "markout_5000ms_bps"      # OBI, micro, label
SPREAD = "spread_bps"                                            # for the crossed-spread mask
VOLC = "realized_vol_bps"                                        # daily vol proxy for terciles
# parsed-data roots (for halt discovery / suspended-day detection)
OB_SNAPSHOT_ROOT = "/Users/shazzak/Capital Stake - Parsed/ob_snapshot"
MISC_ROOT        = "/Users/shazzak/Capital Stake - Parsed/misc"
# output dir
OUT_DIR = "/Users/shazzak/Capital Stake - Results/markout_validation"
# rank (Spearman) vs linear (Pearson)
RANK_MODE = True
# fraction of crossed-spread rows above which a whole day is discarded
CROSSED_DAY_FRAC = 0.05
# minimum masked rows for a day to be usable
MIN_ROWS = 2000
# market-wide halt dates (fill from discovery: dates where misc shows a market halt)
MARKET_HALT_DATES = set()   # e.g. {"2025-12-04", "2026-02-24"}


# ---------------- IO ----------------

def q(con, sql):
    # run a query and return a DataFrame
    return con.execute(sql).df()


# ---------------- halt vocabulary discovery (run once to configure) ----------------

def discover_halts(con):
    # ob_snapshot: which trading_status / phase / break_reason strings exist?
    print("=== ob_snapshot vocab (fill HALT_* config from these) ===")
    for col in ["trading_status", "phase", "break_reason"]:
        # value counts over a broad glob (cheap: reads one column)
        sql = (f"SELECT {col}, COUNT(*) c FROM read_parquet('{OB_SNAPSHOT_ROOT}/date=*/*.parquet') "
               f"GROUP BY {col} ORDER BY c DESC LIMIT 20")
        print(f"\n-- {col} --"); print(q(con, sql).to_string(index=False))
    # suspended_all_day days per symbol
    print("\n-- suspended_all_day (symbol, date) count --")
    sql = (f"SELECT symbol, COUNT(DISTINCT date) n_suspended_days "
           f"FROM read_parquet('{OB_SNAPSHOT_ROOT}/date=*/*.parquet') "
           f"WHERE suspended_all_day GROUP BY symbol ORDER BY n_suspended_days DESC")
    print(q(con, sql).to_string(index=False))
    # misc: which msg_type codes exist (one is the market-halt message)
    print("\n=== misc vocab (find the market-halt msg_type here) ===")
    sql = (f"SELECT msg_type, COUNT(*) c FROM read_parquet('{MISC_ROOT}/date=*/*.parquet') "
           f"GROUP BY msg_type ORDER BY c DESC LIMIT 30")
    print(q(con, sql).to_string(index=False))


# ---------------- build the day-exclusion set ----------------

def build_bad_days(con):
    # (symbol, date) pairs to drop entirely
    bad = set()
    # 1) suspended-all-day, per symbol+date
    try:
        d = q(con, f"SELECT DISTINCT symbol, CAST(date AS VARCHAR) date "
                   f"FROM read_parquet('{OB_SNAPSHOT_ROOT}/date=*/*.parquet') WHERE suspended_all_day")
        # add each pair
        for _, r in d.iterrows():
            bad.add((r["symbol"], r["date"]))
        print(f"  suspended_all_day days: {len(d)}")
    except Exception as e:
        # ob_snapshot may not be reachable; continue with the other masks
        print(f"  (suspended check skipped: {e})")
    # 2) crossed-spread-heavy days, from feature_store itself
    for sym, dt, path in W.enumerate_partitions():
        # fraction of rows with non-positive spread that day
        try:
            f = q(con, f"SELECT AVG(CASE WHEN {SPREAD} <= 0 THEN 1.0 ELSE 0.0 END) frac "
                       f"FROM read_parquet('{path}')").iloc[0]["frac"]
            # flag the day if too many crossed rows
            if f is not None and f > CROSSED_DAY_FRAC:
                bad.add((sym, dt))
        except Exception:
            # unreadable file -> skip
            pass
    # 3) user-supplied market-wide halt dates (apply to both symbols)
    if MARKET_HALT_DATES:
        for sym, dt, _ in W.enumerate_partitions():
            if dt in MARKET_HALT_DATES:
                bad.add((sym, dt))
    print(f"  total bad (symbol,date) excluded: {len(bad)}")
    return bad


# ---------------- per-day cross-moments (masked) ----------------

def moment_sql(path):
    # x1/x2/y expressions: raw values, or dense ranks for Spearman mode
    if RANK_MODE:
        # rank within the day; ties get RANK()'s standard competition ranking
        x1e = f"CAST(RANK() OVER (ORDER BY {X1}) AS DOUBLE)"
        x2e = f"CAST(RANK() OVER (ORDER BY {X2}) AS DOUBLE)"
        ye  = f"CAST(RANK() OVER (ORDER BY {Y})  AS DOUBLE)"
    else:
        # raw values
        x1e, x2e, ye = X1, X2, Y
    # masked, non-null base rows; then rank (if needed); then aggregate the 10 sums
    return f"""
    WITH base AS (
        SELECT {X1} AS x1r, {X2} AS x2r, {Y} AS yr, {SPREAD} AS sp, {VOLC} AS vol
        FROM read_parquet('{path}')
        WHERE {X1} IS NOT NULL AND {X2} IS NOT NULL AND {Y} IS NOT NULL AND {SPREAD} > 0
    ),
    r AS (
        SELECT {x1e.replace(X1,'x1r')} AS x1,
               {x2e.replace(X2,'x2r')} AS x2,
               {ye.replace(Y,'yr')}   AS y,
               vol
        FROM base
    )
    SELECT COUNT(*) n,
           SUM(x1) sx1, SUM(x2) sx2, SUM(y) sy,
           SUM(x1*x1) sx1x1, SUM(x2*x2) sx2x2, SUM(x1*x2) sx1x2,
           SUM(x1*y) sx1y, SUM(x2*y) sx2y, SUM(y*y) syy,
           MEDIAN(vol) vol
    FROM r
    """


# ---------------- solve normal equations from the moment sums ----------------

def solve_from_moments(m):
    # row count
    n = m["n"]
    # too few rows -> unusable
    if n is None or n < MIN_ROWS:
        return None
    # total SS of y (centered)
    ss_tot = m["syy"] - m["sy"] * m["sy"] / n
    # guard degenerate y
    if ss_tot <= 0:
        return None
    # Model B design moments: [1, x1, x2]
    MB = np.array([[n,        m["sx1"],   m["sx2"]],
                   [m["sx1"], m["sx1x1"], m["sx1x2"]],
                   [m["sx2"], m["sx1x2"], m["sx2x2"]]], float)
    bB = np.array([m["sy"], m["sx1y"], m["sx2y"]], float)
    # Model A design moments: [1, x1]
    MA = np.array([[n,        m["sx1"]],
                   [m["sx1"], m["sx1x1"]]], float)
    bA = np.array([m["sy"], m["sx1y"]], float)
    # solve both (lstsq handles near-singular collinear cases)
    try:
        betaB, *_ = np.linalg.lstsq(MB, bB, rcond=None)
        betaA, *_ = np.linalg.lstsq(MA, bA, rcond=None)
    except np.linalg.LinAlgError:
        return None
    # SS_res = syy - beta . X'y  (valid for OLS normal-equation solution)
    ssB = m["syy"] - betaB @ bB
    ssA = m["syy"] - betaA @ bA
    # R^2 of each model
    r2b = 1 - ssB / ss_tot
    r2a = 1 - ssA / ss_tot
    # incremental R^2 from adding micro
    dr2 = r2b - r2a
    # micro's coefficient in Model B
    coef_micro = betaB[2]
    # partial correlation of micro with y given x1 (signed)
    denom = max(1 - r2a, 1e-12)
    pcorr = np.sign(coef_micro) * np.sqrt(max(dr2, 0.0) / denom)
    # return the day's metrics
    return dict(n=int(n), r2_a=r2a, r2_b=r2b, dr2=dr2, coef_micro=coef_micro,
                pcorr=pcorr, vol=m["vol"])


# ---------------- main ----------------

def main():
    # lazy duckdb
    import duckdb
    con = duckdb.connect()
    # (optional) print the halt vocab to help configure -- comment out once configured
    try:
        discover_halts(con)
    except Exception as e:
        print(f"(discovery skipped: {e})")
    # build the day-exclusion set
    print("\nBuilding halt/bad-day mask...")
    bad = build_bad_days(con)
    # per-day residualization over the masked partitions
    recs = []
    for sym, dt, path in W.enumerate_partitions():
        # skip excluded days
        if (sym, dt) in bad:
            continue
        # compute the day's cross-moments
        try:
            m = q(con, moment_sql(path)).iloc[0].to_dict()
        except Exception:
            continue
        # solve for the day's metrics
        out = solve_from_moments(m)
        # keep usable days
        if out:
            out.update(symbol=sym, date=dt)
            recs.append(out)
    # assemble
    df = pd.DataFrame(recs)
    # guard empty
    if df.empty:
        print("no usable days after masking"); return

    # per symbol: vol terciles + summary + plot
    for sym, g in df.groupby("symbol"):
        # tercile by daily vol
        g = g.copy()
        g["vol_t"] = pd.qcut(g["vol"], 3, labels=["low", "mid", "high"])
        # day-as-unit summary of the partial correlation and dR2 by tercile
        print(f"\n[{sym}] micro_dev partial signal after OBI (mode={'rank' if RANK_MODE else 'linear'}):")
        for lvl, gg in g.groupby("vol_t", observed=True):
            # mean partial corr and its day-as-unit SE
            pc, se = gg["pcorr"].mean(), gg["pcorr"].std() / np.sqrt(len(gg))
            # mean incremental R^2 and OBI's own R^2 for context
            print(f"   vol={lvl:4s}  partial_rho(micro|obi)={pc:+.3f} +/- {se:.3f}  "
                  f"mean dR2={gg['dr2'].mean():.5f}  OBI R2={gg['r2_a'].mean():.4f}  n={len(gg)}")

        # figure: partial-rho over time (coloured by vol) + dR2 histogram by tercile
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        g2 = g.sort_values("date")
        g2["dt"] = pd.to_datetime(g2["date"])
        sc = ax[0].scatter(g2["dt"], g2["pcorr"], c=g2["vol"], cmap="viridis", s=14)
        ax[0].axhline(0, color="red", ls="--", lw=1)
        ax[0].set_title(f"{sym}: partial rho(micro|OBI) over time"); ax[0].set_ylabel("partial rho")
        fig.colorbar(sc, ax=ax[0], label="daily vol")
        for lvl, gg in g.groupby("vol_t", observed=True):
            ax[1].hist(gg["pcorr"], bins=25, alpha=0.5, label=f"vol={lvl}")
        ax[1].axvline(0, color="red", ls="--", lw=1)
        ax[1].set_title(f"{sym}: partial rho(micro|OBI) by vol tercile"); ax[1].legend()
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"residualize_micro_on_obi_{sym}.png")
        plt.savefig(out, dpi=130); plt.close(fig)
        print(f"[{sym}] saved {out}")


# entry point
if __name__ == "__main__":
    main()
