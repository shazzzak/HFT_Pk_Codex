# ============================================================================
# sweep_risk_score.py -- STANDING risk scorer for any sweep's per-name-day CSV.
# Always reports, per config: net PKR, mean daily bps, Sharpe, Sortino, max
# drawdown (bps + PKR), win rate, and paired-vs-control -- so no sweep is ever
# scored on return alone again. Uses quantstats if installed, else identical
# hand formulas. Metrics are on the DAILY net-bps series (P&L / that day's
# opened notional) -- the scale-free return series for a market maker.
# USAGE: python sweep_risk_score.py <PERNAME.csv> [--control LABEL] [--bucket middle]
# ============================================================================
import sys, argparse
import numpy as np, pandas as pd
try:
    import quantstats as qs; _QS = True
except Exception:
    _QS = False

ANN = 252.0

def sharpe(d):
    d = d[~np.isnan(d)]
    if len(d) < 3 or d.std(ddof=1) == 0: return np.nan
    return d.mean() / d.std(ddof=1) * np.sqrt(ANN)

def sortino(d):
    d = d[~np.isnan(d)]
    dn = d[d < 0]
    if len(d) < 3 or len(dn) == 0: return np.nan
    dd = np.sqrt((dn ** 2).mean())        # downside deviation vs 0 target
    return d.mean() / dd * np.sqrt(ANN) if dd > 0 else np.nan

def maxdd(series_pkr):
    c = np.cumsum(series_pkr); peak = np.maximum.accumulate(c)
    return float((c - peak).min())         # most-negative peak-to-trough (PKR)

def cfg_col(df):
    for c in ["throttle", "ofi_window"]:
        if c in df.columns and df[c].nunique() > 1: return c
    return "throttle" if "throttle" in df.columns else "ofi_window"

def score(path, control=None, bucket=None):
    df = pd.read_csv(path)
    cc = cfg_col(df)
    if bucket:
        df = df[df["bucket"] == bucket]
    # a bucket with no rows (e.g. a name that never traded then) -> signal empty
    if len(df) == 0:
        return None, None
    g = df.groupby([cc, "date"]).agg(pkr=("net_pkr", "sum"), opn=("opened_notional", "sum")).reset_index()
    g["bps"] = np.where(g["opn"] > 0, g["pkr"] / g["opn"] * 1e4, np.nan)
    w_bps = g.pivot(index="date", columns=cc, values="bps")
    w_pkr = g.pivot(index="date", columns=cc, values="pkr")
    rows = []
    for cfg in w_bps.columns:
        d = w_bps[cfg].to_numpy(); p = w_pkr[cfg].fillna(0).to_numpy()
        # prefer quantstats for Sharpe/Sortino when available
        if _QS:
            r = pd.Series(d).dropna() / 1e4  # bps -> fraction for qs
            sh = qs.stats.sharpe(r); so = qs.stats.sortino(r)
        else:
            sh = sharpe(d); so = sortino(d)
        rows.append(dict(config=str(cfg), net_pkr=np.nansum(p), mean_bps=np.nanmean(d),
                         sharpe=sh, sortino=so, maxDD_pkr=maxdd(p),
                         win=np.mean(p > 0), n=len(d)))
    r = pd.DataFrame(rows)
    # paired vs control
    ctrl = control or ("clip3x" if "clip3x" in w_bps.columns else
                       ("OBI" if "OBI" in w_bps.columns else r.sort_values("net_pkr").iloc[-1]["config"]))
    def paired(cfg):
        dd = (w_bps[cfg] - w_bps[ctrl]).dropna().to_numpy()
        if len(dd) < 2: return np.nan, np.nan
        return dd.mean(), dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
    r["vs_ctrl_bps"], r["vs_ctrl_t"] = zip(*[paired(c) for c in r["config"]])
    return r, ctrl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--control", default=None); ap.add_argument("--bucket", default=None)
    a = ap.parse_args()
    for bk in ([a.bucket] if a.bucket else [None, "first15", "middle", "preclose45", "last15"]):
        r, ctrl = score(a.csv, a.control, bk)
        # nothing in this bucket -> say so and move on
        if r is None or not len(r):
            print(f"\n===== RISK-SCORED: {('BUCKET='+bk) if bk else 'WHOLE DAY'} -- no rows =====")
            continue
        tag = f"BUCKET={bk}" if bk else "WHOLE DAY (all buckets)"
        print(f"\n===== RISK-SCORED: {tag}  (control={ctrl}; quantstats={_QS}) =====")
        print(f"{'config':10s} {'net_PKR':>10s} {'bps':>6s} {'Sharpe':>7s} {'Sortino':>8s} {'maxDD_PKR':>11s} {'win%':>5s} {'vs_ctrl_bps':>11s} {'t':>6s}")
        for _, x in r.sort_values("net_pkr", ascending=False).iterrows():
            print(f"{x['config']:10s} {x['net_pkr']:>10,.0f} {x['mean_bps']:>6.2f} {x['sharpe']:>7.2f} {x['sortino']:>8.2f} {x['maxDD_pkr']:>11,.0f} {x['win']*100:>4.0f}% {x['vs_ctrl_bps']:>+11.3f} {x['vs_ctrl_t']:>+6.2f}")
