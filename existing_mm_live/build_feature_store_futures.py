# build_feature_store_futures.py -- build the feature store for the DELIVERY
# FUTURES, reusing the EXACT spot builder (build_one) so futures scoring routes
# through the engine's own Book -- the same price basis the backtest uses. This
# is the ROOT-CAUSE fix for the futures MM scorer bug: the prior run used a
# separate local scorer (snapshot-mid reimplementation) that diverged from the
# engine's fills, producing the impossible "positive net_bps + negative P&L".
# With a real feature store, futures MM uses C.score_bps (the validated spot
# scorer) reading feature_store_fut mids -- one price basis, no divergence.
#
# Keyed by CONTRACT (BOP-JUL, BOP-AUG, ...), not root. The futures MM runner then
# selects the active contract per date via its roll map -- same as it already does
# for trading. We only build the CONTRACTS THAT ARE ACTIVE at some point (max
# trailing volume), to avoid building dead back-month books we never trade.
#
# Reuse: imports build_one, FS internals, and the driver from the spot builder.
# Output: feature_store_fut/{contract}/date={date}.parquet  (separate root so it
# never collides with the spot feature_store).
#
# Run from existing_mm_live/:  caffeinate -is python3 build_feature_store_futures.py

from pathlib import Path
import time
import pandas as pd
import numpy as np
import duckdb
# the spot builder module -- we reuse its build_one verbatim (engine parity)
import build_feature_store as BFS
import run_legacy_mm as R

R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
PARSED = str(R.PARSED_ROOT)
RESULTS = Path("/Users/shazzak/Capital Stake - Results")
# SEPARATE store root so futures never collide with spot feature_store/
FS_FUT = RESULTS / "feature_store_fut"

# ------------------------------ knobs ----------------------------------------
# the liquid futures roots (same universe as the MM run)
ROOTS = ["TRG", "MLCF", "BOP", "SSGC", "PTC", "TPLP", "PAEL", "KOSM",
         "DGKC", "THCCL", "PIAHCLA", "PACE", "PIBTL", "TPL", "CNERGY"]
# trailing window (days) for the active-contract decision (matches the MM run)
ROLL_TRAIL = 5
# a contract-day needs at least this many trades to be worth building
MIN_TRADES_DAY = 200
# -----------------------------------------------------------------------------


# per (root, contract, date) trade counts + volume, for the roll map
def load_futures_calendar():
    con = duckdb.connect()
    q = f"""
    SELECT regexp_extract(symbol,'^([A-Z]+)-',1) AS root, symbol,
           CAST(date AS VARCHAR) AS date, COUNT(*) AS n_trades, SUM(qty) AS qty
    FROM read_parquet('{PARSED}/trades/date=*/*.parquet')
    WHERE market='STOCK_DEL_FUT'
    GROUP BY root, symbol, date
    """
    cal = con.execute(q).df()
    con.close()
    return cal[cal["root"].isin(ROOTS)].copy()


# {(root,date): active_contract} via max trailing-ROLL_TRAIL-day volume (causal)
def build_roll_map(cal, all_dates):
    dates = [str(d) for d in all_dates]
    roll = {}
    for root, g in cal.groupby("root"):
        piv = g.pivot_table(index="date", columns="symbol", values="qty",
                            aggfunc="sum").reindex(dates).fillna(0.0)
        trail = piv.rolling(ROLL_TRAIL, min_periods=1).sum().shift(1)
        for d in dates:
            row = trail.loc[d]
            if row.notna().any() and row.max() > 0:
                roll[(root, d)] = row.idxmax()
    return roll


def main():
    # point the reused builder's output at the futures store root
    BFS.FS_ROOT = FS_FUT
    FS_FUT.mkdir(parents=True, exist_ok=True)

    all_dates = R.discover_dates()
    print("pre-pass: futures calendar + roll map (DuckDB)", flush=True)
    cal = load_futures_calendar()
    roll = build_roll_map(cal, all_dates)
    # trade-count lookup to skip dead contract-days
    tc = {(r["symbol"], r["date"]): r["n_trades"] for _, r in cal.iterrows()}
    # the set of (date -> active contract) build tasks
    tasks = []
    for (root, date), contract in roll.items():
        if tc.get((contract, date), 0) >= MIN_TRADES_DAY:
            tasks.append((date, contract))
    # group by date so we open each date's datasets once
    by_date = {}
    for date, contract in tasks:
        by_date.setdefault(date, []).append(contract)
    print(f"  {len(tasks)} active contract-days to build across "
          f"{len(by_date)} dates -> {FS_FUT}\n", flush=True)

    t0 = time.perf_counter()
    built = 0
    for di, date in enumerate(sorted(by_date), 1):
        dsets = R.open_datasets(date)
        if dsets is None:
            print(f"  [{di}/{len(by_date)}] {date} no datasets; skip", flush=True)
            continue
        dt0 = time.perf_counter()
        for contract in by_date[date]:
            out_dir = FS_FUT / contract
            out_path = out_dir / f"date={date}.parquet"
            # resume support: skip already-built
            if out_path.exists():
                continue
            try:
                # build_one is engine-parity: same Book, same features as spot
                df = BFS.build_one(date, contract, dsets)
            except Exception as e:
                print(f"    {contract} {date} ERROR {e!r}", flush=True)
                continue
            if df is None or len(df) == 0:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            built += 1
        # ETA heartbeat
        el = time.perf_counter() - t0
        eta = el / di * (len(by_date) - di)
        print(f"  [{di}/{len(by_date)}] {date} {time.perf_counter()-dt0:.2f}s  "
              f"built {built}  ETA {int(eta//60)}m{int(eta%60):02d}s", flush=True)
    print(f"\ndone. {built} contract-days written to {FS_FUT}")
    print("next: point futures_mm_run's scorer at feature_store_fut and use "
          "C.score_bps (delete the local scorer).")


if __name__ == "__main__":
    main()
