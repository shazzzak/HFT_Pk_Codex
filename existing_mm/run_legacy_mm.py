"""
Batch-run the legacy single-name pure market maker (NaiveSymmetricMM) over the
ENTIRE parsed store: every symbol, every date.

The store is hive-partitioned parquet:
    <PARSED_ROOT>/trades/date=YYYY-MM-DD/*.parquet
    <PARSED_ROOT>/ob_updates/date=YYYY-MM-DD/*.parquet
    <PARSED_ROOT>/ob_snapshot/date=YYYY-MM-DD/*.parquet
    <PARSED_ROOT>/misc/date=YYYY-MM-DD/*.parquet          (not needed here)

It REUSES the frozen engine from mm_backtest.py (Book, Backtester,
NaiveSymmetricMM, fee_for) -- it never reimplements them. The only new code is
(a) a parquet event loader that mirrors mm_backtest.load_events exactly, and
(b) a driver that loops dates x symbols and writes one summary row each.

Usage (run in this order -- see the chat walkthrough):
    python run_legacy_mm.py --inspect                 # print schema of one partition
    python run_legacy_mm.py --smoke --date 2026-06-30 # one date, first 3 symbols
    python run_legacy_mm.py --date 2026-06-30          # one full date (single process)
    python run_legacy_mm.py --all --workers 3          # every date, parallel

FLAGGED SIMPLIFICATIONS:
  * Strategy = NaiveSymmetricMM (the baseline "pure MM" in mm_backtest, the one
    with the documented MCB reference). To use MicrostructureMM instead, see
    make_strategy() -- but fix its short-cap None-return bug first (separate note).
  * Session window = [first, last] NON-AUCTION trade per symbol-day, matching the
    ticker-stats convention. If your reference run used a different window, set it here.
  * fill_on_crossing_adds defaults False (conservative). The exact cfg behind the
    documented 417 fills / -5,611 PKR is not recorded in the files; use --smoke on
    MCB 2026-06-30 to find the setting that reproduces it before trusting absolute PnL.
"""
import argparse
import ast
import glob
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# The frozen engine. This import is the single source of truth for fills, fees,
# book reconstruction, and EOD liquidation -- do not copy these, import them.
from mm_backtest import Backtester, NaiveSymmetricMM, fee_for, FEE_TOTAL_PCT

# ------------------------------- CONFIG -------------------------------------
# Point this at your store root (the folder that CONTAINS trades/ ob_updates/ ...).
PARSED_ROOT = Path(
    "/Users/shazzak/Library/CloudStorage/"
    "GoogleDrive-shazzak@gmail.com/My Drive/Capital Stake - Parsed"
)
# Where per-date result CSVs and the combined file are written.
OUT_DIR = Path("./mm_results")

# Strategy parameters (the documented MCB baseline).
STRAT = dict(half_spread=0.20, size=50, max_inv=500)

# Backtester config. fee schedule itself comes from mm_backtest (17.73 bps/side).
CFG = dict(
    latency_ms=120,             # constant one-way latency, matches the reference run
    at_price_mode="queue",      # realistic queue consumption; "never"/"always" bracket it
    fill_on_crossing_adds=False,  # conservative; see flagged simplification above
    unfilled_haircut_pct=0.10,  # haircut on inventory the closing book cannot absorb
)

# The columns each table MUST provide. Validated up front so a name mismatch
# fails LOUDLY on symbol 1, not silently after hours. If validation raises,
# send me the printed schema and I map the names -- do not guess.
REQ_TRADES = ["symbol", "transact_time", "capture_ts", "price", "qty",
              "initiator", "aggressor_side", "resting_order_id", "appl_seq"]
REQ_UPDATES = ["symbol", "transact_time", "capture_ts", "order_id", "side",
               "price", "qty", "event", "appl_seq"]
REQ_SNAP = ["symbol", "msg_seq", "orig_time", "capture_ts", "entry_type",
            "px", "phase", "order_ids", "order_qtys", "qty"]


# --------------------------- loader helpers ---------------------------------
def to_ms(s: pd.Series) -> pd.Series:
    """ISO-string OR datetime64 column -> int64 milliseconds since epoch (UTC).

    Mirrors mm_backtest.load_events' ms(): parse to UTC, force ns, cast int64,
    // 1e6. The dtype guard is the only addition -- parquet stores these as
    datetime64[ns, UTC] already, whereas the CSV path had strings.
    """
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, utc=True, format="ISO8601")
    else:
        s = pd.to_datetime(s, utc=True)
    return s.dt.as_unit("ns").astype("int64") // 1_000_000


def parse_rest_oid(x):
    """Extract the resting order id from trades.resting_order_id.

    CSV stored it as a stringified tuple "('0010...R0K', 407.31)". Parquet may
    store the same string, or a real list/tuple. Handle both; never eval() disk
    data (ast.literal_eval only). Anything else -> None (Book.trade falls back
    to its __NEG_ placeholder path, exactly as documented for ~2/3 of trades).
    """
    if isinstance(x, str) and x.startswith("("):
        try:
            return ast.literal_eval(x)[0]
        except Exception:
            return None
    if isinstance(x, (list, tuple)) and len(x) >= 1:
        return x[0]
    # PSX stores resting_order_id as a bare order-ID string (e.g. 0010THF0D00017T6),
    # which shares a namespace with ob_updates.order_id -- return it directly so
    # queue-position resolution can match it. (The tuple/list branches above handle
    # the legacy formats.)
    if isinstance(x, str) and x:
        return x
    return None


def build_events(u: pd.DataFrame, s: pd.DataFrame, t: pd.DataFrame):
    """In-memory twin of mm_backtest.load_events (post-read section).

    Same timestamp derivation, same rest_oid parse, same snap_groups keyed by
    msg_seq, same event tuples and the same (ts_exch, kind_rank, appl_seq) sort.
    Returns (events, snap_groups, t) just like load_events.
    """
    # Exchange clock: transact_time for updates/trades, orig_time for snapshots.
    u["ts_exch"] = to_ms(u["transact_time"])
    t["ts_exch"] = to_ms(t["transact_time"])
    s["ts_exch"] = to_ms(s["orig_time"])
    # Capture clock everywhere -> knowledge time.
    for df in (u, t, s):
        df["ts_cap"] = to_ms(df["capture_ts"])
    # appl_seq is the same channel-2011 sequence for updates+trades; coerce to a
    # plain int so heap/sort never compares NA. Snapshots carry a 0 placeholder.
    for df in (u, t):
        df["appl_seq"] = pd.to_numeric(df["appl_seq"], errors="coerce").fillna(-1).astype("int64")
    # Resting order id for the exact trade-consumption path.
    t["rest_oid"] = t["resting_order_id"].map(parse_rest_oid)

    # One frame per snapshot message (all its rows: book + AGG + status).
    snap_groups = dict(tuple(s.groupby("msg_seq")))
    # One (ts_exch, ts_cap) per message; status-only messages become events too.
    snap_ev = s.groupby("msg_seq", as_index=False)[["ts_exch", "ts_cap"]].min()

    # Build the merged event list -- identical ordering contract to load_events.
    events = [(r.ts_exch, 1, r.appl_seq, "U", r) for r in u.itertuples()]
    events += [(r.ts_exch, 1, r.appl_seq, "T", r) for r in t.itertuples()]
    events += [(r.ts_exch, 0, 0, "S", r) for r in snap_ev.itertuples()]
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return events, snap_groups, t


def date_dir(table: str, date: str) -> Path:
    # Hive partition folder for one table/date.
    return PARSED_ROOT / table / f"date={date}"


def open_datasets(date: str):
    """pyarrow datasets for the three tables of one date, or None if incomplete."""
    paths = {tbl: date_dir(tbl, date) for tbl in ("trades", "ob_updates", "ob_snapshot")}
    for tbl, p in paths.items():
        if not p.exists():
            return None
    return {tbl: ds.dataset(str(p), format="parquet") for tbl, p in paths.items()}


def validate_schema(dsets):
    """Fail loudly if any required column is absent, listing the mismatch."""
    checks = [("trades", REQ_TRADES), ("ob_updates", REQ_UPDATES), ("ob_snapshot", REQ_SNAP)]
    problems = []
    for tbl, req in checks:
        have = set(dsets[tbl].schema.names)
        missing = [c for c in req if c not in have]
        if missing:
            problems.append(f"{tbl}: missing {missing}; has {sorted(have)}")
    if problems:
        raise KeyError("Schema mismatch -- send me these and I will map the names:\n  "
                       + "\n  ".join(problems))


def read_symbol(dset, cols, sym):
    # Predicate pushdown on symbol (leading sort key -> row-group pruning).
    tbl = dset.to_table(columns=cols, filter=ds.field("symbol") == sym)
    return tbl.to_pandas()


def list_symbols(trade_ds):
    # Distinct traded symbols, scanned in batches to bound memory.
    syms = set()
    for batch in trade_ds.scanner(columns=["symbol"]).to_batches():
        syms.update(batch.column("symbol").to_pylist())
    return sorted(s for s in syms if s is not None)


# ----------------------------- one backtest ---------------------------------
def make_strategy():
    # Swap point: return MicrostructureMM(...) here to run the microstructure MM
    # instead (it needs session_ms and per-symbol params, and its short-cap bug
    # must be fixed first). Default is the documented baseline.
    return NaiveSymmetricMM(**STRAT)


def run_one(date, sym, dsets):
    """Run one symbol-day. Returns a summary dict, or None if not runnable."""
    u = read_symbol(dsets["ob_updates"], REQ_UPDATES, sym)
    s = read_symbol(dsets["ob_snapshot"], REQ_SNAP, sym)
    t = read_symbol(dsets["trades"], REQ_TRADES, sym)
    # A pure MM needs a trade stream and a book; skip empties.
    if len(t) == 0 or len(s) == 0:
        return None

    events, snap_groups, t = build_events(u, s, t)

    # Session = continuous (non-auction) trading span, in exchange-ms.
    cont = t[t["initiator"] != "AUCTION"]
    if len(cont) == 0:
        return None
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())

    cfg = dict(CFG, session=(t0, t1))
    bt = Backtester(make_strategy(), cfg)
    fills, equity, stats = bt.run(events, snap_groups)
    eod = bt.eod or {}

    # rest_oid resolution rate: a health check. If ~0, resting_order_id is not in
    # the expected format and queue tracking silently degrades to the fallback.
    resolved = float(t["rest_oid"].notna().mean()) if len(t) else np.nan

    return {
        "date": date, "symbol": sym,
        "n_events": len(events), "n_fills": int(len(fills)),
        "n_orders_sent": stats.get("n_orders_sent", 0),
        "n_cancels": stats.get("n_cancels", 0),
        "rejected_crossing": stats.get("rejected_crossing", 0),
        "halted_requotes": stats.get("halted_requotes", 0),
        "pos_at_close": eod.get("pos_at_close"),
        "net_pnl": eod.get("equity_liquidated"),
        "equity_mid_mark": eod.get("equity_mid_mark"),
        "liquidation_clean": eod.get("liquidation_clean"),
        "unfilled_sh": eod.get("unfilled_sh"),
        "eod_fired": bool(bt.eod is not None),
        "rest_oid_resolved_frac": round(resolved, 4),
    }


def process_date(date, symbol_limit=None):
    """Run every symbol for one date; write mm_results/mm_<date>.csv. Returns row count."""
    dsets = open_datasets(date)
    if dsets is None:
        print(f"  [{date}] SKIP: a table partition is missing")
        return 0
    validate_schema(dsets)
    symbols = list_symbols(dsets["trades"])
    if symbol_limit:
        symbols = symbols[:symbol_limit]

    rows, errors = [], []
    for i, sym in enumerate(symbols, 1):
        try:
            r = run_one(date, sym, dsets)
            if r is not None:
                rows.append(r)
        except Exception as e:
            errors.append((sym, repr(e)))
        if i % 50 == 0:
            print(f"  [{date}] {i}/{len(symbols)} symbols")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"mm_{date}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    msg = f"  [{date}] wrote {len(rows)} rows -> {out}"
    if errors:
        msg += f"  ({len(errors)} errored, e.g. {errors[0]})"
    print(msg)
    return len(rows)


def discover_dates():
    # Dates are the date=YYYY-MM-DD folders under trades/.
    dirs = glob.glob(str(PARSED_ROOT / "trades" / "date=*"))
    return sorted(d.split("date=")[-1] for d in dirs)


def already_done(date):
    # Resumability: skip a date whose output already exists and is non-empty.
    f = OUT_DIR / f"mm_{date}.csv"
    return f.exists() and f.stat().st_size > 0


def combine():
    # Stitch all per-date CSVs into one combined file at the end.
    files = sorted(glob.glob(str(OUT_DIR / "mm_*.csv")))
    if not files:
        print("No per-date results to combine."); return
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    comb = OUT_DIR / "mm_ALL.csv"
    df.to_csv(comb, index=False)
    print(f"Combined {len(files)} dates, {len(df)} symbol-days -> {comb}")


# --------------------------------- CLI --------------------------------------
def inspect_one():
    dates = discover_dates()
    if not dates:
        print(f"No date= partitions under {PARSED_ROOT/'trades'}"); return
    d = dates[0]
    print(f"Inspecting partition date={d}")
    dsets = open_datasets(d)
    for tbl in ("trades", "ob_updates", "ob_snapshot"):
        print(f"\n[{tbl}] columns:")
        print("  ", dsets[tbl].schema.names)
    try:
        validate_schema(dsets)
        print("\nSchema validation: PASS -- required columns present.")
    except KeyError as e:
        print("\nSchema validation: FAIL\n", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--date", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    if a.inspect:
        inspect_one(); return
    if a.smoke:
        d = a.date or discover_dates()[0]
        n = process_date(d, symbol_limit=3)
        print(f"Smoke: {n} rows for {d}. Inspect mm_results/mm_{d}.csv"); return
    if a.date and not a.all:
        process_date(a.date); combine(); return
    if a.all:
        dates = [d for d in discover_dates() if not already_done(d)]
        print(f"{len(dates)} dates to run (resumable), workers={a.workers}")
        if a.workers <= 1:
            for d in dates:
                process_date(d)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(process_date, d): d for d in dates}
                for f in as_completed(futs):
                    f.result()
        combine(); return
    ap.print_help()


if __name__ == "__main__":
    main()
