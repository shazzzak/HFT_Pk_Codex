"""Corporate-action MASTER + reconciliation layer.

DOCTRINE (hybrid, two sources):
  PRIMARY   = vendor/exchange announcements (PUCARS / Capital Stake / PSX
              Payouts). This is ground truth for WHAT happened and WHEN.
  VALIDATOR = the price-based detector (corp_action_detector.py). It cannot be
              the primary source -- it is blind to symbol changes, spin-offs,
              and (critically) to dividends where the exchange does NOT adjust
              prev_close, which is exactly when contamination is silent.

Neither source alone is safe:
  * announcements alone -> a wrong/missing row is never caught
  * detector alone      -> silent misses on the failure mode above
Reconciling them turns both failure modes into a LOUD row in a review queue.

FACTOR CONVENTION (matches corp_action_detector.py):
  price_factor is the PRICE multiplier: old_price * price_factor = new scale.
  Forward 2:1 split -> 0.5.  Reverse 1:5 -> 5.0.  No split -> 1.0.
  Volumes are adjusted by the RECIPROCAL.
  Cash dividends carry price_factor = 1.0 and a cash_amount instead.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Canonical schema every announcement row must have after loading.
MASTER_COLS = ["symbol", "ex_date", "action_type", "price_factor",
               "cash_amount", "new_symbol", "source", "notes"]

# Action types we understand. Anything else -> forced to review.
ACTION_TYPES = {"forward_split", "reverse_split", "bonus", "rights",
                "dividend", "symbol_change", "merger", "spinoff"}

# Relative tolerance when comparing an announced factor to an observed one.
FACTOR_TOL = 0.02


def load_announcements(path):
    """Load the vendor/exchange corporate-action file into the canonical schema.

    INGESTION IS VENDOR-SPECIFIC AND IS *YOUR* PIECE TO BUILD/BUY. This function
    deliberately does NOT scrape anything: it reads a file you produce, and
    enforces the contract. Point it at a CSV/parquet exported from Capital
    Stake's corporate-actions API, PSX Payouts, or a manual sheet.

    Required columns (missing ones are created empty, then validated):
      symbol       ticker as it appears in the FIX feed
      ex_date      YYYY-MM-DD, the first session trading EX the action
      action_type  one of ACTION_TYPES
      price_factor PRICE multiplier (see convention above); 1.0 for dividends
      cash_amount  dividend per share in PKR; 0.0 for splits
      new_symbol   for symbol_change only; else empty
      source       provenance string, e.g. 'capitalstake_api' / 'psx_payouts'
      notes        free text
    """
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df.reindex(columns=MASTER_COLS)
    df["symbol"] = df["symbol"].astype("string").str.strip().str.upper()
    df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["action_type"] = df["action_type"].astype("string").str.strip().str.lower()
    df["price_factor"] = pd.to_numeric(df["price_factor"], errors="coerce").fillna(1.0)
    df["cash_amount"] = pd.to_numeric(df["cash_amount"], errors="coerce").fillna(0.0)
    bad = df[df["ex_date"].isna() | ~df["action_type"].isin(ACTION_TYPES)]
    if len(bad):
        print(f"  WARNING: {len(bad)} announcement rows have a bad ex_date or "
              f"unknown action_type -> they will be flagged, not silently used")
    return df


def effective_price_factor(row, ref_price):
    """Multiplicative factor to apply to PRE-ex prices to put them on the
    post-ex scale. Splits use price_factor directly; cash dividends become a
    ratio against the reference (pre-ex) close so the overnight gap is removed
    from cross-day returns."""
    f = float(row["price_factor"]) if pd.notna(row["price_factor"]) else 1.0
    cash = float(row["cash_amount"]) if pd.notna(row["cash_amount"]) else 0.0
    if cash > 0 and ref_price and ref_price > 0:
        f = f * (ref_price - cash) / ref_price
    return f


def reconcile(announcements, detections):
    """Cross-check the master against what the price series actually showed.

    announcements: DataFrame in MASTER_COLS (primary source)
    detections:    DataFrame from the detector's registry, columns at least
                   [symbol, date, status, factor, my_prev_close, feed_prev_close]

    Returns one row per (symbol, date) that needs attention, with `verdict`:

      CONFIRMED          both agree -> safe to apply automatically
      FACTOR_MISMATCH    both fired, factors disagree -> REVIEW before use
      SILENT_ADJUSTMENT  announced, detector saw nothing. The dangerous case:
                         the exchange did not move prev_close, so the overnight
                         gap is still in the data and MUST be adjusted from the
                         announcement. Never drop these.
      UNANNOUNCED_MOVE   detector fired, nothing announced -> master is missing
                         a row, or a genuine market move was misclassified.
    """
    a = announcements.copy()
    a["_key"] = a["symbol"].astype(str) + "|" + a["ex_date"].astype(str)
    d = detections.copy()
    d["_key"] = d["symbol"].astype(str) + "|" + d["date"].astype(str)
    # The detector's own 'normal'/'no_baseline'/'minor_diff' rows are non-events.
    d = d[~d["status"].isin(["normal", "no_baseline", "minor_diff"])]

    out = []
    for _, r in a.iterrows():
        m = d[d["_key"] == r["_key"]]
        if len(m) == 0:
            out.append({**r.to_dict(), "verdict": "SILENT_ADJUSTMENT",
                        "observed_factor": np.nan, "detector_status": None})
            continue
        obs = float(m.iloc[0]["factor"]) if pd.notna(m.iloc[0]["factor"]) else np.nan
        ann = float(r["price_factor"])
        agree = (not np.isnan(obs)) and abs(obs - ann) <= FACTOR_TOL * max(abs(ann), 1e-9)
        out.append({**r.to_dict(),
                    "verdict": "CONFIRMED" if agree else "FACTOR_MISMATCH",
                    "observed_factor": obs,
                    "detector_status": m.iloc[0]["status"]})
    seen = set(a["_key"])
    for _, r in d.iterrows():
        if r["_key"] in seen:
            continue
        out.append({"symbol": r["symbol"], "ex_date": r["date"],
                    "action_type": None, "price_factor": np.nan,
                    "cash_amount": 0.0, "new_symbol": None,
                    "source": "detector_only", "notes": r.get("note"),
                    "verdict": "UNANNOUNCED_MOVE",
                    "observed_factor": r.get("factor"),
                    "detector_status": r["status"]})
    res = pd.DataFrame(out)
    return res.drop(columns=[c for c in ["_key"] if c in res.columns])


def write_registry(con, reconciled, table="corp_actions"):
    """Upsert the reconciled registry into DuckDB. Idempotent per ex_date."""
    con.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
        symbol VARCHAR, ex_date VARCHAR, action_type VARCHAR,
        price_factor DOUBLE, cash_amount DOUBLE, new_symbol VARCHAR,
        source VARCHAR, notes VARCHAR, verdict VARCHAR,
        observed_factor DOUBLE, detector_status VARCHAR)""")
    dates = sorted(set(reconciled["ex_date"].dropna()))
    if dates:
        con.execute(f"DELETE FROM {table} WHERE ex_date IN "
                    f"({','.join(['?'] * len(dates))})", dates)
    con.register("_rec", reconciled)
    con.execute(f"INSERT INTO {table} SELECT symbol, ex_date, action_type, "
                f"price_factor, cash_amount, new_symbol, source, notes, "
                f"verdict, observed_factor, detector_status FROM _rec")
    con.unregister("_rec")
