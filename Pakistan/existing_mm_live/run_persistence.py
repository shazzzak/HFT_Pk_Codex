# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# run_persistence.py -- TABLE A: does a run of same-side aggressor trades predict
# the NEXT trade's side on PSX? Replicates the Aldridge FBL ladder
# (P(buy | 1,2,3,4 consecutive buys) = 46/62/69/72%) on the PSX trades table.
# Pure trades-table computation -- no engine, no book, seconds to run.
#
# METHOD: within each symbol-day's continuous-session trades (time-ordered),
# walk the aggressor_side sequence keeping a CONSECUTIVE-run counter. For each
# trade, its "prior run length k" = how many immediately-preceding trades shared
# the same side. Then P(next same side | run length k) = fraction of the trades
# whose prior run was exactly k that CONTINUE the run (i.e. same side as the run).
# Reported per run length (1..5, 6+), sign-agnostic (buys and sells pooled by
# treating "same as the run" as the event), day-as-unit + pooled, per name.
#
# 50% = no persistence (dead). A rising ladder = runs predict continuation.
#
# USAGE:  python run_persistence.py --self-test
#         python run_persistence.py --run [--days N] [--symbols ...]
# ============================================================================

# CLI parsing
import argparse
# timing + heartbeat
import time
# timestamps on log lines
from datetime import datetime
# parallel over dates
from multiprocessing import Pool
# filesystem paths
from pathlib import Path
# numerics
import numpy as np
# dataframes
import pandas as pd


# bracketed HH:MM:SS stamp
def _ts():
    # current local time
    return datetime.now().strftime("[%H:%M:%S]")


# local parsed store (run_legacy_mm default points at the empty Google Drive)
# Resolve this filesystem path through the canonical checkout/data configuration.
LOCAL_STORE = Path(str(_hft_paths.PARSED_ROOT))
# results root (for the watchlist name list)
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS_ROOT = Path(str(_hft_paths.RESULTS_ROOT))
# output directory
# Resolve this filesystem path through the canonical checkout/data configuration.
OUT_DIR = Path(str(_hft_paths.RESULTS_ROOT / 'diagnostics'))
# run-length buckets to report (5 = exactly 5; 6 = 6-or-more)
RUN_BUCKETS = [1, 2, 3, 4, 5, 6]
# default sampled days
MAX_DAYS = 20
# default workers
WORKERS = 6


# ------------------------------------------------------------------ core -----
# from a time-ordered array of aggressor sides, compute, for each prior-run
# length k, (n_at_k, n_continued): how many trades had prior run exactly k, and
# of those how many CONTINUED the run (next-same-side). Returns two dicts.
def ladder_counts(sides):
    # dict k -> number of trades whose immediately-prior run had length k
    n_at = {}
    # dict k -> of those, how many continued (this trade == the run's side)
    n_cont = {}
    # need at least two trades to have a "prior run" and a "continue" decision
    if sides.size < 2:
        # nothing to count
        return n_at, n_cont
    # current run length of the sequence UP TO (not including) trade i
    run = 0
    # side of the current run
    run_side = None
    # walk trades in time order
    for i in range(sides.size):
        # the side of THIS trade
        s = sides[i]
        # for i>=1 we can classify: prior run length is `run`, and this trade
        # either continues (s == run_side) or breaks it
        if run >= 1 and run_side is not None:
            # bucket the prior run length (cap at 6 = "6 or more")
            k = run if run < 6 else 6
            # count one trade observed at prior-run-length k
            n_at[k] = n_at.get(k, 0) + 1
            # count a continuation if this trade matches the run's side
            if s == run_side:
                # continued
                n_cont[k] = n_cont.get(k, 0) + 1
        # update the running counter to INCLUDE this trade for the next iteration
        if s == run_side:
            # same side -> extend the run
            run += 1
        else:
            # side flipped (or first trade) -> new run of length 1
            run = 1
            # remember the new run's side
            run_side = s
    # per-k observed and continued counts
    return n_at, n_cont


# per-process globals
_R = None
_NAMES = None
_COLLAPSE = False


# pool initializer: import driver, point at local store, set universe
def _init(names, collapse=False):
    # expose globals
    global _R, _NAMES, _COLLAPSE
    # remember whether to collapse sweeps
    _COLLAPSE = collapse
    # the parquet driver
    import run_legacy_mm as R
    # local store (Drive default is empty)
    R.PARSED_ROOT = LOCAL_STORE
    # stash
    _R = R
    # default to the shortlist from the watchlist CSV; else all traded symbols
    if names is None:
        # try to read the 38-name watchlist
        try:
            # the final MM watchlist at the new results root
            wl = pd.read_csv(RESULTS_ROOT / "mm_watchlist_final.csv")
            # column is 'symbol'
            names = sorted(wl["symbol"].dropna().astype(str).unique().tolist())
        except Exception:
            # fall back to all symbols present that day (resolved per date)
            names = None
    # stash the universe
    _NAMES = names


# columns needed from trades (continuous-session filter + side + time order)
_TR_COLS = ["symbol", "transact_time", "aggressor_side"]


# scan one date: per symbol, compute the ladder counts; return rows
def _work_date(date):
    # open the date's datasets
    dsets = _R.open_datasets(date)
    # missing partition -> nothing
    if dsets is None:
        # empty
        return []
    # universe (resolved 38 names, else all traded symbols)
    names = _NAMES or _R.list_symbols(dsets["trades"])
    # accumulated rows
    rows = []
    # loop symbols
    for sym in names:
        # read this symbol's trades
        try:
            # pull only the needed columns
            tr = _R.read_symbol(dsets["trades"], _TR_COLS, sym)
        except Exception as e:
            # report + continue on any read failure
            print(_ts() + f"SKIP {date} {sym}: {e!r}")
            # next symbol
            continue
        # need trades with a valid aggressor side
        if tr is None or len(tr) == 0 or "aggressor_side" not in tr.columns:
            # skip
            continue
        # keep only BUY/SELL aggressor rows, in time order
        tr = tr[tr["aggressor_side"].astype(str).str.upper().str[0].isin(["B", "S"])]
        # need at least a handful of trades
        if len(tr) < 20:
            # too few to be meaningful
            continue
        # sort by exchange time to get the true sequence
        tr = tr.sort_values("transact_time")
        # +1 for buy, -1 for sell
        side = np.where(tr["aggressor_side"].astype(str).str.upper().str[0].eq("B"), 1, -1)
        # COLLAPSE MODE: merge consecutive prints that share the SAME exchange
        # timestamp AND side into ONE aggressor order (strips sweep fragmentation).
        if _COLLAPSE:
            # exchange timestamps as int ns
            ts = pd.to_datetime(tr["transact_time"], utc=True).astype("int64").to_numpy()
            # start a new "order" when EITHER the side flips OR the timestamp changes
            new_order = np.ones(side.size, dtype=bool)
            # for i>=1, it's a continuation of the same order only if same ts AND same side
            new_order[1:] = (ts[1:] != ts[:-1]) | (side[1:] != side[:-1])
            # keep one side value per collapsed order (the first print of each)
            sides = side[new_order]
        else:
            # raw trade-print sequence (no collapsing)
            sides = side
        # compute the ladder counts for this symbol-day
        n_at, n_cont = ladder_counts(sides)
        # emit one row per run-length bucket that had observations
        for k in RUN_BUCKETS:
            # observed count at this run length
            na = n_at.get(k, 0)
            # skip empties
            if na == 0:
                # nothing at this k today
                continue
            # continuation count
            nc = n_cont.get(k, 0)
            # record date/symbol/k/counts (probability computed at aggregation)
            rows.append(dict(date=str(date), symbol=sym, run_len=k, n_at=na, n_cont=nc))
    # this date's rows
    return rows


# full run: scan dates in parallel, aggregate the ladder, print + save
def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS, collapse=False):
    # driver (for dates)
    import run_legacy_mm as R
    # local store
    R.PARSED_ROOT = LOCAL_STORE
    # all dates
    dates = R.discover_dates()
    # guard empty store
    if not dates:
        # explain + stop
        print(_ts() + "discover_dates() empty -- check PARSED_ROOT."); return
    # sample days evenly
    if max_days and len(dates) > max_days:
        # stride
        step = max(1, len(dates) // max_days)
        # take sample
        dates = dates[::step][:max_days]
    # announce
    mode = "COLLAPSED (distinct aggressor orders)" if collapse else "RAW (trade prints)"
    print(_ts() + f"{len(dates)} dates, {workers} workers -- runs-persistence ladder [{mode}]")
    # accumulate rows
    rows = []
    # timer
    t0 = time.perf_counter()
    # pool
    with Pool(processes=workers, initializer=_init, initargs=(symbols, collapse)) as pool:
        # completed counter
        done = 0
        # consume results
        for res in pool.imap_unordered(_work_date, dates):
            # collect
            rows.extend(res)
            # bump
            done += 1
            # progress
            print(_ts() + f"  date {done}/{len(dates)} ({len(rows)} rows, {(time.perf_counter()-t0)/60:.1f} min)")
    # nothing collected
    if not rows:
        # stop
        print(_ts() + "no rows."); return
    # per-(symbol,date,run_len) counts
    df = pd.DataFrame(rows)
    # ensure output dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # save the raw counts
    tag = "_collapsed" if collapse else ""
    df.to_csv(out_dir / f"run_persistence_counts{tag}.csv", index=False)

    # ---- POOLED ladder: sum counts across everything, P = cont/at ----
    print(_ts() + "===== RUNS-PERSISTENCE LADDER: P(next trade CONTINUES the run | run length) =====")
    print(_ts() + "  (50% = no persistence; Aldridge FBL 2009 was 46/62/69/72% at k=1/2/3/4)")
    # pooled probability per run length
    pooled = df.groupby("run_len").agg(n_at=("n_at", "sum"), n_cont=("n_cont", "sum")).reset_index()
    # continuation probability
    pooled["P_continue"] = pooled["n_cont"] / pooled["n_at"]
    # print pooled ladder
    for _, r in pooled.iterrows():
        # label 6 as "6+"
        lab = f"{int(r['run_len'])}" if r["run_len"] < 6 else "6+"
        # one line per rung
        print(_ts() + f"    run={lab:>3}:  P(continue)={r['P_continue']*100:5.1f}%   (n={int(r['n_at']):,})")

    # ---- DAY-AS-UNIT: per (symbol,date) P at each k, then mean +/- SE ----
    # per name-day probability at each run length
    df["P"] = df["n_cont"] / df["n_at"]
    print(_ts() + "  --- day-as-unit (each name-day one obs; needs n_at>=20 that day) ---")
    # rows with enough observations that day to trust the daily P
    solid = df[df["n_at"] >= 20]
    # per run length
    for k in RUN_BUCKETS:
        # this rung's daily P values
        vals = solid[solid.run_len == k]["P"].to_numpy()
        # need a couple of name-days
        if vals.size < 2:
            # skip
            continue
        # mean and SE across name-days
        m = vals.mean(); se = vals.std(ddof=1) / np.sqrt(vals.size)
        # label
        lab = f"{k}" if k < 6 else "6+"
        # print
        print(_ts() + f"    run={lab:>3}:  {m*100:5.1f}% +/- {se*100:4.1f}   (name-days={vals.size:,})")

    # ---- per-name pooled ladder saved for the follow-up ----
    pern = (df.groupby(["symbol", "run_len"]).agg(n_at=("n_at", "sum"), n_cont=("n_cont", "sum")).reset_index())
    # probability per name-rung
    pern["P_continue"] = pern["n_cont"] / pern["n_at"]
    # save
    pern.to_csv(out_dir / f"run_persistence_pername{tag}.csv", index=False)
    # verdict hint
    p2 = pooled.set_index("run_len")["P_continue"]
    # compare k=1 to k=4 to state the direction
    if 1 in p2.index and 4 in p2.index:
        # rising ladder -> persistence exists
        trend = "RISING -> runs predict continuation (build Table B)" if p2[4] > p2[1] + 0.02 else "FLAT -> no usable persistence (stop)"
        # print the verdict
        print(_ts() + f"  VERDICT: k=1 {p2[1]*100:.1f}% -> k=4 {p2[4]*100:.1f}%  => {trend}")
    # outputs location
    print(_ts() + f"[run-persistence] outputs -> {out_dir}")


# validate ladder_counts on hand-built sequences with known answers
def self_test():
    # sequence: B B B S  -> classify trades i=1..3
    #   i=1: prior run=1 (B), this=B -> at[1]+1, cont[1]+1
    #   i=2: prior run=2 (B), this=B -> at[2]+1, cont[2]+1
    #   i=3: prior run=3 (B), this=S -> at[3]+1, cont[3]+0
    seq = np.array([1, 1, 1, -1])
    # compute
    n_at, n_cont = ladder_counts(seq)
    # show
    print(_ts() + f"[self-test] BBBS: n_at={n_at} n_cont={n_cont}")
    # assert the exact counts
    assert n_at == {1: 1, 2: 1, 3: 1}, f"n_at wrong: {n_at}"
    # continuations: only k=1 and k=2 continued
    assert n_cont == {1: 1, 2: 1}, f"n_cont wrong: {n_cont}"
    # a perfectly alternating sequence -> every prior run is length 1, never continues
    alt = np.array([1, -1, 1, -1, 1, -1])
    # compute
    a2, c2 = ladder_counts(alt)
    # show
    print(_ts() + f"[self-test] alternating: n_at={a2} n_cont={c2}")
    # 5 classifiable trades, all prior-run=1, none continue
    assert a2 == {1: 5} and c2 == {}, f"alt wrong: {a2} {c2}"
    # a long single run BBBBBBB (7) -> k caps at 6
    lon = np.ones(8, dtype=int)
    # compute
    a3, c3 = ladder_counts(lon)
    # show
    print(_ts() + f"[self-test] 8xB: n_at={a3} n_cont={c3}")
    # prior runs seen: 1,2,3,4,5,6,6 (the 7th classifiable trade has prior run 7->capped 6)
    assert a3 == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}, f"long n_at wrong: {a3}"
    # every one continued (all same side)
    assert c3 == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}, f"long n_cont wrong: {c3}"
    # pooled probability sanity on the long run = 100% continue at every k
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


# entry point
if __name__ == "__main__":
    # parser
    ap = argparse.ArgumentParser()
    # self-test
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--collapse", action="store_true", help="collapse same-ts+side prints into one aggressor order")
    # full run
    ap.add_argument("--run", action="store_true")
    # symbol subset
    ap.add_argument("--symbols", nargs="*", default=None)
    # workers
    ap.add_argument("--workers", type=int, default=WORKERS)
    # days
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    # parse
    a = ap.parse_args()
    # dispatch
    if a.self_test or not a.run:
        # run the self-test
        self_test()
    # full run
    if a.run:
        # scan
        run_real(symbols=a.symbols, workers=a.workers, max_days=(a.days or None), collapse=a.collapse)
