# ============================================================================
# iceberg_feasibility.py -- 10-second check: can we detect icebergs by order-id,
# or must we fall back to swept-vs-visible-depth (replay)?
# Reports, on a few sampled days: are resting_order_id (trades) and order_id
# (ob_updates) populated, and -- if so -- a rough iceberg prevalence (orders
# filled far beyond their max displayed size = hidden replenishment).
# USAGE:  python iceberg_feasibility.py            (samples ~5 days)
#         python iceberg_feasibility.py --days 5 --symbols OGDC UBL
# ============================================================================
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

def _ts():
    return datetime.now().strftime("[%H:%M:%S]")

def main(n_days, symbols):
    import run_legacy_mm as R
    dates = R.discover_dates()
    # sample a handful of evenly-spaced dates
    step = max(1, len(dates) // n_days)
    samp = dates[::step][:n_days]
    print(_ts() + f"probing {len(samp)} dates: {list(samp)}")
    tr_total = tr_nonnull = tr_distinct = 0
    ob_total = ob_nonnull = 0
    iceberg_hits = 0; iceberg_vol = 0.0; matched_vol = 0.0
    for date in samp:
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # ---- trades: resting_order_id populated? ----
        tr = dsets["trades"]
        if hasattr(tr, "to_df"):   # duckdb relation
            tr = tr.to_df()
        cols = set(tr.columns)
        rid = "resting_order_id" if "resting_order_id" in cols else None
        sym_col = "symbol" if "symbol" in cols else None
        if symbols and sym_col:
            tr = tr[tr[sym_col].isin(symbols)]
        tr_total += len(tr)
        if rid:
            nn = tr[rid].notna() & (tr[rid].astype(str).str.len() > 0)
            tr_nonnull += int(nn.sum())
            tr_distinct += int(tr.loc[nn, rid].nunique())
        # ---- ob_updates: order_id populated? ----
        ob = dsets["ob_updates"]
        if hasattr(ob, "to_df"):
            ob = ob.to_df()
        ocols = set(ob.columns)
        oid = "order_id" if "order_id" in ocols else None
        if symbols and "symbol" in ocols:
            ob = ob[ob["symbol"].isin(symbols)]
        ob_total += len(ob)
        if oid:
            onn = ob[oid].notna() & (ob[oid].astype(str).str.len() > 0)
            ob_nonnull += int(onn.sum())
        # ---- rough iceberg prevalence by order-id (only if both ids present) ----
        if rid and oid and "qty" in cols and "qty" in ocols:
            # max displayed size per resting order (from book updates)
            disp = ob.loc[ob[oid].astype(str).str.len() > 0].groupby(oid)["qty"].max()
            # cumulative filled per resting order (from trades)
            filled = tr.loc[tr[rid].astype(str).str.len() > 0].groupby(rid)["qty"].sum()
            j = pd.concat({"disp": disp, "filled": filled}, axis=1).dropna()
            if len(j):
                # iceberg = filled >= 2x the most it ever displayed
                ice = j[j["filled"] >= 2.0 * j["disp"]]
                iceberg_hits += len(ice)
                iceberg_vol += float(ice["filled"].sum())
                matched_vol += float(j["filled"].sum())
    print()
    print(_ts() + "===== TRADES.resting_order_id =====")
    print(_ts() + f"  rows={tr_total:,}  populated={tr_nonnull:,} "
          f"({100*tr_nonnull/max(tr_total,1):.1f}%)  distinct ids={tr_distinct:,}")
    print(_ts() + "===== OB_UPDATES.order_id =====")
    print(_ts() + f"  rows={ob_total:,}  populated={ob_nonnull:,} "
          f"({100*ob_nonnull/max(ob_total,1):.1f}%)")
    print(_ts() + "===== ROUGH ICEBERG PREVALENCE (order-id method) =====")
    if matched_vol > 0:
        print(_ts() + f"  resting orders filled >=2x their max display: {iceberg_hits:,}")
        print(_ts() + f"  their share of matched volume: {100*iceberg_vol/matched_vol:.1f}%")
    else:
        print(_ts() + "  could not match by id (ids empty or missing) -> use swept-vs-visible method")
    print()
    # verdict
    if tr_nonnull > 0.5 * max(tr_total, 1) and ob_nonnull > 0.5 * max(ob_total, 1):
        print(_ts() + "VERDICT: order-ids populated -> build the FAST order-id iceberg detector.")
    else:
        print(_ts() + "VERDICT: order-ids sparse/empty -> build the swept-vs-visible (replay) detector.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--symbols", nargs="*", default=None)
    a = ap.parse_args()
    main(a.days, a.symbols)
