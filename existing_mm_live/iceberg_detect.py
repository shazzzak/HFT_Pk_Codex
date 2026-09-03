# ============================================================================
# iceberg_detect.py -- order-id iceberg / hidden-replenishment detector.
# A resting order is an ICEBERG if it gets FILLED far more than the most it ever
# DISPLAYED (it kept refilling hidden size). Pure query on trades + ob_updates:
#   filled(id)    = sum of trades.qty where resting_order_id == id
#   displayed(id) = max of ob_updates.qty where order_id == id
#   iceberg  <=>  filled >= ICE_MULT * displayed   (and displayed > 0)
# Reports per symbol-day: iceberg count, their share of traded volume, hidden
# ratio, and which side. Portable -- this is a reusable capability for any venue.
#
# USAGE:
#   python iceberg_detect.py --self-test
#   python iceberg_detect.py --smoke              # ONE date, verbose (read check)
#   python iceberg_detect.py --run                # sampled days, parallel
# ============================================================================
import argparse
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

def _ts():
    return datetime.now().strftime("[%H:%M:%S]")

OUT_DIR = Path("/Users/shazzak/Capital Stake - Results/diagnostics")
ICE_MULT = 2.0        # filled >= 2x max displayed => iceberg (hidden replenishment)
WORKERS = 6
MAX_DAYS = 30


def _as_df(x, cols=None):
    # open_datasets may return a DataFrame, a DuckDB relation, or a path.
    if isinstance(x, pd.DataFrame):
        return x
    for m in ("df", "to_df", "fetchdf"):
        if hasattr(x, m):
            try:
                return getattr(x, m)()
            except Exception:
                pass
    return pd.read_parquet(x, columns=cols)


def detect_one(trades, obu, symbol=None):
    # trades: resting_order_id, qty, symbol, aggressor_side.  obu: order_id, qty, symbol, side.
    tr = trades; ob = obu
    if symbol is not None:
        if "symbol" in tr.columns: tr = tr[tr["symbol"] == symbol]
        if "symbol" in ob.columns: ob = ob[ob["symbol"] == symbol]
    if not {"resting_order_id", "qty"}.issubset(tr.columns) or not {"order_id", "qty"}.issubset(ob.columns):
        return None
    # keep populated ids only
    tr = tr[tr["resting_order_id"].notna() & (tr["resting_order_id"].astype(str).str.len() > 0)]
    ob = ob[ob["order_id"].notna() & (ob["order_id"].astype(str).str.len() > 0)]
    if len(tr) == 0 or len(ob) == 0:
        return None
    filled = tr.groupby("resting_order_id")["qty"].sum().rename("filled")
    displayed = ob.groupby("order_id")["qty"].max().rename("displayed")
    # side of each resting order (from its book updates), for the side split
    side = (ob.groupby("order_id")["side"].last().rename("side")
            if "side" in ob.columns else None)
    j = pd.concat([filled, displayed] + ([side] if side is not None else []), axis=1).dropna(subset=["filled", "displayed"])
    j = j[j["displayed"] > 0]
    if len(j) == 0:
        return None
    j["ratio"] = j["filled"] / j["displayed"]
    j["iceberg"] = j["ratio"] >= ICE_MULT
    ice = j[j["iceberg"]]
    out = dict(
        n_orders=int(len(j)),
        n_iceberg=int(len(ice)),
        iceberg_order_frac=float(len(ice) / len(j)),
        # share of MATCHED filled volume that flowed through icebergs
        iceberg_vol_frac=float(ice["filled"].sum() / j["filled"].sum()) if j["filled"].sum() > 0 else np.nan,
        median_hidden_ratio=float(ice["ratio"].median()) if len(ice) else np.nan,
        p95_hidden_ratio=float(ice["ratio"].quantile(0.95)) if len(ice) else np.nan,
    )
    if side is not None and len(ice):
        # BID-side icebergs = hidden buyers (support); ASK = hidden sellers (resistance)
        sc = ice["side"].astype(str).str.upper().str[0].value_counts()
        out["iceberg_bid_frac"] = float(sc.get("B", 0) / len(ice))
    return out


# --- worker plumbing ---
_R = None; _NAMES = None
def _init(names):
    global _R, _NAMES
    import run_legacy_mm as R
    _R = R; _NAMES = names


def _work_date(date):
    dsets = _R.open_datasets(date)
    if dsets is None:
        return []
    tr = _as_df(dsets["trades"]); ob = _as_df(dsets["ob_updates"])
    rows = []
    names = _NAMES or (sorted(tr["symbol"].dropna().unique()) if "symbol" in tr.columns else [None])
    for sym in names:
        r = detect_one(tr, ob, sym)
        if r is None:
            continue
        r.update(dict(date=str(date), symbol=sym))
        rows.append(r)
    return rows


def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    import run_legacy_mm as R
    print(_ts() + "discovering dates")
    dates = R.discover_dates()
    if not dates:
        print(_ts() + "discover_dates() returned NOTHING -- check the data path / run dir.")
        return
    if max_days and len(dates) > max_days:
        step = max(1, len(dates) // max_days); dates = dates[::step][:max_days]
    print(_ts() + f"{len(dates)} dates, {workers} workers, iceberg = filled >= {ICE_MULT}x displayed")
    rows = []; t0 = time.perf_counter()
    with Pool(processes=workers, initializer=_init, initargs=(symbols,)) as pool:
        done = 0
        for res in pool.imap_unordered(_work_date, dates):
            rows.extend(res); done += 1
            el = (time.perf_counter() - t0) / 60.0
            print(_ts() + f"  date {done}/{len(dates)} ({len(rows)} sym-days, {el:.1f} min, "
                  f"ETA {el/done*(len(dates)-done):.1f} min)")
    if not rows:
        print(_ts() + "no rows -- resting_order_id / order_id empty on these days?")
        return
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "iceberg_detect_pername_day.csv", index=False)
    def m(c): 
        x = df[c].dropna(); return (x.mean(), x.median()) if len(x) else (np.nan, np.nan)
    print(_ts() + "===== ICEBERG PREVALENCE (day-as-unit; order-id method) =====")
    print(_ts() + f"  iceberg orders: mean {m('iceberg_order_frac')[0]*100:.2f}% of resting orders")
    print(_ts() + f"  iceberg VOLUME: mean {m('iceberg_vol_frac')[0]*100:.2f}% of matched fill volume")
    print(_ts() + f"  hidden ratio (filled/displayed) of icebergs: median {m('median_hidden_ratio')[1]:.1f}x")
    if "iceberg_bid_frac" in df:
        print(_ts() + f"  iceberg side: {m('iceberg_bid_frac')[0]*100:.0f}% on the BID (hidden buyers)")
    print(_ts() + f"[iceberg] outputs -> {out_dir}")


def smoke(symbols=None):
    import run_legacy_mm as R
    dates = R.discover_dates()
    print(_ts() + f"discover_dates -> {len(dates)} dates" + (f"; first={dates[0]}" if dates else " (EMPTY!)"))
    if not dates:
        return
    date = dates[len(dates) // 2]
    dsets = R.open_datasets(date)
    tr = _as_df(dsets["trades"]); ob = _as_df(dsets["ob_updates"])
    print(_ts() + f"[smoke] {date}: trades rows={len(tr):,}, ob_updates rows={len(ob):,}")
    print(_ts() + f"[smoke] trades cols: {list(tr.columns)}")
    if "resting_order_id" in tr.columns:
        pop = (tr["resting_order_id"].astype(str).str.len() > 0).mean()
        print(_ts() + f"[smoke] resting_order_id populated: {pop*100:.1f}%")
    sym = (symbols[0] if symbols else (sorted(tr["symbol"].dropna().unique())[0] if "symbol" in tr.columns else None))
    r = detect_one(tr, ob, sym)
    print(_ts() + f"[smoke] {sym}: {r}")


def self_test():
    # order A: displayed max 100, filled 500 (5 refills) -> ratio 5 -> ICEBERG
    # order B: displayed 100, filled 80 -> ratio 0.8 -> normal
    # order C: displayed 50, filled 120 -> ratio 2.4 -> ICEBERG
    trades = pd.DataFrame({
        "resting_order_id": ["A", "A", "A", "A", "A", "B", "C", "C"],
        "qty": [100, 100, 100, 100, 100, 80, 60, 60],
        "symbol": ["X"] * 8,
    })
    obu = pd.DataFrame({
        "order_id": ["A", "A", "B", "C"],
        "qty": [100, 80, 100, 50],           # A's max display = 100
        "side": ["BID", "BID", "ASK", "BID"],
        "symbol": ["X"] * 4,
    })
    r = detect_one(trades, obu, "X")
    print(_ts() + f"[self-test] {r}")
    assert r["n_orders"] == 3
    assert r["n_iceberg"] == 2, "A and C are icebergs"
    assert abs(r["iceberg_order_frac"] - 2/3) < 1e-9
    # iceberg filled volume = A(500)+C(120)=620; total filled=500+80+120=700 -> 0.886
    assert abs(r["iceberg_vol_frac"] - 620/700) < 1e-6, "iceberg volume share wrong"
    # A ratio 5, C ratio 2.4 -> median 3.7
    assert abs(r["median_hidden_ratio"] - 3.7) < 1e-9, "hidden ratio wrong"
    assert abs(r["iceberg_bid_frac"] - 1.0) < 1e-9, "both icebergs on BID"
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    a = ap.parse_args()
    if a.smoke:
        smoke(symbols=a.symbols)
    elif a.self_test or not a.run:
        self_test()
    if a.run:
        run_real(symbols=a.symbols, workers=a.workers, max_days=(a.days or None))
