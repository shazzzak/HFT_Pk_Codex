# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# iceberg_detect.py -- iceberg / hidden-refill detector.
# Rides the ALREADY-reconstructed order book (mm_backtest.Book, driven by
# R.build_events) -- no book rebuild here. Walks the same merged S/U/T event
# stream the sweeps use and OBSERVES only. Two complementary signals:
#   (A) LIVE REFILL-CYCLE counter (pre-sweep): after a level at price P/side S is
#       hit by a trade, a NEW ORDER_ADD at EXACTLY P, same side, within <=100ms is
#       one refill cycle. 2 = Provisional wall, 3+ = Confirmed wall. (PSX has only
#       ORDER_ADD/CANCEL, no MODIFY, so only the synthetic-reload path applies.)
#   (B) M1/M2 (post-sweep): cluster trades at one exchange ts+side into a sweep;
#       M1 = distinct print prices; M2 = levels the total_qty would consume walking
#       the LIVE pre-trade book. Iceberg when M1<=2 & M2>=4.
#   Pairing: count sweeps that M1/M2-flag a price/side already flagged as a wall.
# USAGE:  python iceberg_detect.py --self-test | --smoke | --run
# ============================================================================

# argument parsing for the CLI
import argparse
# wall-clock timing for heartbeats/ETA
import time
# timestamps on every printed line
from datetime import datetime
# parallel scan across dates
from multiprocessing import Pool
# filesystem paths
from pathlib import Path
# numerics
import numpy as np
# dataframes
import pandas as pd


# return a bracketed HH:MM:SS stamp for log lines
def _ts():
    # format the current local time
    return datetime.now().strftime("[%H:%M:%S]")


# directory for diagnostic outputs
# Resolve this filesystem path through the canonical checkout/data configuration.
OUT_DIR = Path(str(_hft_paths.RESULTS_ROOT / 'diagnostics'))
# local parsed store (run_legacy_mm's default points at Google Drive, which is empty)
# Resolve this filesystem path through the canonical checkout/data configuration.
LOCAL_STORE = Path(str(_hft_paths.PARSED_ROOT))
# a refill must arrive within this many milliseconds of the level being hit
REFILL_MS = 100
# a sweep counts as low-impact if it printed at this many price levels or fewer
M1_MAX = 2
# a sweep is "big" (should have walked the book) at this many expected levels or more
M2_MIN = 4
# refill cycles that mark a level a "provisional" wall
PROVISIONAL = 2
# refill cycles that mark a level a "confirmed" wall
CONFIRMED = 3
# default parallel workers
WORKERS = 6
# default number of sampled days (None/0 = all)
MAX_DAYS = 30


# count how many best-first levels a total quantity would consume (the M2 walk)
def _walk_levels(levels, total):
    # running cumulative visible quantity
    c = 0.0
    # walk levels best-first, counting how many are needed to cover `total`
    for i, (_, q) in enumerate(levels, 1):
        # add this level's visible quantity
        c += q
        # stop once cumulative depth covers the swept quantity
        if c >= total:
            # number of levels consumed
            return i
    # total exceeds all visible depth -> it would consume every present level
    return len(levels)


# single observing pass over the merged event stream; returns a per-symbol-day dict
def scan_events(events, Book):
    # the live reconstructed book (engine's validated aggregation: handles the
    # __NEG_ unknown-resting-id placeholders and snapshot reconciliation)
    book = Book()
    # cache the ranked levels; recompute only after the book actually changes
    depth_cache = {"dirty": True, "bids": [], "asks": []}
    # return the top-N ranked levels per side, recomputing only when dirty
    def _depth(n=10):
        # recompute only after an add/cancel/trade dirtied the cache
        if depth_cache["dirty"]:
            # one ranked scan via the engine's aggregation
            depth_cache["bids"], depth_cache["asks"] = book.ranked_depth(n=n)
            # mark clean
            depth_cache["dirty"] = False
        # hand back the cached levels
        return depth_cache["bids"], depth_cache["asks"]
    # (side, price) -> exchange-ns time that level was last hit by a trade
    last_hit = {}
    # (side, price) -> refill-cycle count in the current episode
    cycles = {}
    # list of wall events reached: (ts, side, price, cycle_count)
    walls = []
    # (side, price) -> max cycle count reached (used for the sweep-agreement check)
    wall_active = {}
    # the running trade cluster being accumulated: [ts, buy, total_qty, {prices}]
    cur = None
    # count of sweeps flagged by the M1/M2 gate
    n_ice_m2 = 0
    # count of "big" sweeps (M2 >= M2_MIN), the ones large enough to test
    n_big = 0
    # count of all sweeps seen
    n_sweeps = 0
    # count of M1/M2 sweeps that landed on a price/side already flagged as a wall
    agree = 0

    # finalize one accumulated trade cluster into a sweep and run the M1/M2 gate
    def close_sweep(cl):
        # allow the counters above to be mutated from this closure
        nonlocal n_ice_m2, n_big, n_sweeps, agree
        # nothing to close
        if cl is None:
            # exit early
            return
        # unpack the cluster
        ts, buy, qty, prices = cl
        # M1 = number of distinct price levels the sweep actually printed at
        m1 = len(prices)
        # read the CACHED ranked levels (recomputed only when the book changed)
        bids, asks = _depth()
        # a buyer sweeps the ASK side; a seller sweeps the BID side
        levels = asks if buy else bids
        # no depth on the hit side -> cannot compute a counterfactual
        if not levels:
            # skip this sweep
            return
        # M2 = levels the swept quantity would consume against the pre-trade book
        m2 = _walk_levels(levels, qty)
        # every closed cluster is one sweep
        n_sweeps += 1
        # tally sweeps big enough to be testable
        if m2 >= M2_MIN:
            # increment the big-sweep counter
            n_big += 1
        # the iceberg signature: large expected impact but tiny realized impact
        if m1 <= M1_MAX and m2 >= M2_MIN:
            # count the M1/M2 iceberg
            n_ice_m2 += 1
            # the resting wall being hit sits on the side opposite the aggressor
            side = "BUY" if buy else "SELL"
            # the wall price is the extreme print on the hit side
            hit_px = max(prices) if buy else min(prices)
            # look up whether a live wall was already flagged there
            w = wall_active.get((side, hit_px))
            # agreement if that wall had reached at least provisional strength
            if w and w >= PROVISIONAL:
                # count the corroboration
                agree += 1

    # iterate the time-ordered merged stream (kind: S=snapshot, U=update, T=trade)
    for (ts, rank, seq, kind, obj) in events:
        # a trade print
        if kind == "T":
            # aggressor side: True if the buyer crossed the spread
            buy = str(getattr(obj, "aggressor_side", "")).upper().startswith("B")
            # this print's price
            px = float(obj.price)
            # this print's quantity
            q = float(obj.qty)
            # a new cluster starts when the timestamp or side changes
            if cur is not None and (cur[0] != ts or cur[1] != buy):
                # finalize the previous cluster first
                close_sweep(cur)
                # clear it
                cur = None
            # open a fresh cluster if none is active
            if cur is None:
                # [ts, buy, running_qty, set of print prices]
                cur = [ts, buy, 0.0, set()]
            # accumulate swept quantity
            cur[2] += q
            # record the print price
            cur[3].add(px)
            # the resting order that was hit sits on the side opposite the aggressor
            rest_side = "BUY" if not buy else "SELL"
            # remember when this resting level was last hit (for the refill window)
            last_hit[(rest_side, px)] = ts
            # apply the trade to the engine book (validated fill decrement +
            # __NEG_ placeholder for unknown resting ids)
            try:
                # decrement resting qty for the fill
                book.trade(obj)
                # book changed -> depth cache is stale
                depth_cache["dirty"] = True
            except Exception:
                # engine-specific edge cases (e.g. auction prints) -> skip
                pass
        # a book update (add or cancel)
        elif kind == "U":
            # close any open cluster before a non-trade event at a later timestamp
            if cur is not None and cur[0] != ts:
                # finalize the sweep
                close_sweep(cur)
                # clear it
                cur = None
            # the update's event type
            ev = getattr(obj, "event", None)
            # a new order entering the book
            if ev == "ORDER_ADD":
                # normalize the side to BUY/SELL
                side = str(getattr(obj, "side", "")).upper()
                # map anything starting with B to BUY, else SELL
                side = "BUY" if side.startswith("B") else "SELL"
                # the add's price
                px = float(getattr(obj, "price"))
                # when this exact level was last hit by a trade
                ht = last_hit.get((side, px))
                # a refill: a new add at a just-hit level, same side, inside the window
                if ht is not None and 0 <= (ts - ht) <= REFILL_MS:
                    # increment this level's refill-cycle count
                    cycles[(side, px)] = cycles.get((side, px), 0) + 1
                    # current cycle count
                    n = cycles[(side, px)]
                    # a provisional or stronger wall records an event
                    if n >= PROVISIONAL:
                        # append the wall event
                        walls.append((ts, side, px, n))
                        # track the max strength reached at this level
                        wall_active[(side, px)] = max(wall_active.get((side, px), 0), n)
                    # reset the hit clock so each refill must follow a fresh hit
                    last_hit[(side, px)] = ts
                # apply the add to the engine book
                book.add(obj)
                # book changed -> depth cache is stale
                depth_cache["dirty"] = True
            # otherwise it is a cancel
            else:
                # apply the cancel to the engine book
                book.cancel(obj)
                # book changed -> depth cache is stale
                depth_cache["dirty"] = True
        # a snapshot message
        elif kind == "S":
            # close any open cluster at a snapshot boundary
            if cur is not None:
                # finalize the sweep
                close_sweep(cur)
                # clear it
                cur = None
    # finalize any trailing cluster at end of day
    close_sweep(cur)

    # number of wall events that were exactly at provisional strength
    prov = sum(1 for (_, _, _, n) in walls if n == PROVISIONAL)
    # distinct (side, price) levels that reached confirmed strength
    conf = len(set((s, p) for (_, s, p, n) in walls if n >= CONFIRMED))
    # assemble the per-symbol-day summary
    return dict(
        # all sweeps observed
        n_sweeps=n_sweeps,
        # sweeps large enough to test (M2 >= M2_MIN)
        n_big_sweeps=n_big,
        # sweeps flagged by the M1/M2 gate
        n_iceberg_m2=n_ice_m2,
        # M1/M2 iceberg fraction of all sweeps
        m2_iceberg_frac=(n_ice_m2 / n_sweeps) if n_sweeps else np.nan,
        # total wall events recorded (provisional or stronger)
        n_refill_events=len(walls),
        # distinct confirmed walls
        n_confirmed_walls=conf,
        # M1/M2 sweeps that hit a live-flagged wall
        n_agreements=agree,
        # deepest refill cycle observed
        max_cycles=max([n for (_, _, _, n) in walls], default=0),
    )


# per-process globals set once by the pool initializer
_R = None
_BOOK = None
_NAMES = None


# pool initializer: import the engine, point it at the local store, pick the universe
def _init(names):
    # expose the globals for mutation
    global _R, _BOOK, _NAMES
    # the parquet reader / event builder module
    import run_legacy_mm as R
    # override the store path (default points at the empty Google Drive)
    R.PARSED_ROOT = LOCAL_STORE
    # the engine module that provides the Book reconstruction
    import mm_backtest as MB
    # stash the reader module
    _R = R
    # stash the Book class
    _BOOK = MB.Book
    # default the universe to the calibrated 38 names, not all ~577 symbols
    if names is None:
        # try to read the calibrated name list
        try:
            # the harness holds the trading universe
            import mm_harness as H
            # prefer an explicit NAMES list, else the scales-file keys
            names = list(H.NAMES) if hasattr(H, "NAMES") else sorted(H.load_scales().keys())
        except Exception:
            # fall back to "all symbols" only if the harness is unavailable
            names = None
    # stash the resolved universe
    _NAMES = names


# columns to pull from each table (predicate/column pushdown for speed)
_UNUSED = None


# scan one symbol-day: read the three tables, build the event stream, observe it
def _one(date, sym):
    # open the date's datasets
    dsets = _R.open_datasets(date)
    # the day's order-book updates for this symbol
    u = _R.read_symbol(dsets["ob_updates"], _R.REQ_UPDATES, sym)
    # the day's book snapshots for this symbol
    s = _R.read_symbol(dsets["ob_snapshot"], _R.REQ_SNAP, sym)
    # the day's trades for this symbol
    t = _R.read_symbol(dsets["trades"], _R.REQ_TRADES, sym)
    # unrunnable without both trades and a book
    if len(t) == 0 or len(s) == 0:
        # skip
        return None
    # build the merged, time-ordered event stream (adds ts_exch in place)
    events, snap_groups, t = _R.build_events(u, s, t)
    # observe the stream and return the summary
    return scan_events(events, _BOOK)


# scan every symbol for one date; emit a per-symbol heartbeat
def _work_date(date):
    # open the date's datasets
    dsets = _R.open_datasets(date)
    # missing partition -> nothing to do
    if dsets is None:
        # empty result
        return []
    # the universe to scan (resolved 38 names, or all traded symbols as fallback)
    names = _NAMES or _R.list_symbols(dsets["trades"])
    # accumulated rows for this date
    rows = []
    # per-date timer for the heartbeat
    t0 = time.perf_counter()
    # loop the symbols
    for i, sym in enumerate(names, 1):
        # scan one symbol-day, guarding against per-symbol failures
        try:
            # run the scan
            r = _one(date, sym)
        except Exception as e:
            # report and continue on any symbol error
            print(_ts() + f"SKIP {date} {sym}: {e!r}")
            # next symbol
            continue
        # keep non-empty results
        if r is not None:
            # tag with date and symbol
            r.update(dict(date=str(date), symbol=sym))
            # collect
            rows.append(r)
        # heartbeat every 5 symbols so a slow date still shows movement
        if i % 5 == 0:
            # elapsed seconds for this date so far
            print(_ts() + f"  {date}: {i}/{len(names)} syms ({time.perf_counter()-t0:.0f}s)")
    # this date's rows
    return rows


# full run: scan sampled (or all) dates in parallel and summarize
def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    # the reader module (also used to discover dates in the parent process)
    import run_legacy_mm as R
    # point at the local store
    R.PARSED_ROOT = LOCAL_STORE
    # all available dates
    dates = R.discover_dates()
    # bail if the store path is wrong/empty
    if not dates:
        # explain and stop
        print(_ts() + "discover_dates() empty -- check PARSED_ROOT.")
        # exit
        return
    # optionally subsample dates evenly
    if max_days and len(dates) > max_days:
        # stride to hit roughly max_days
        step = max(1, len(dates) // max_days)
        # take the strided sample
        dates = dates[::step][:max_days]
    # announce the run parameters
    print(_ts() + f"{len(dates)} dates, {workers} workers; refill<= {REFILL_MS}ms, "
          f"confirm>= {CONFIRMED} cycles; M1<={M1_MAX}&M2>={M2_MIN}")
    # accumulated rows across all dates
    rows = []
    # overall timer
    t0 = time.perf_counter()
    # worker pool, universe resolved once per worker in the initializer
    with Pool(processes=workers, initializer=_init, initargs=(symbols,)) as pool:
        # completed-date counter
        done = 0
        # consume results as dates finish
        for res in pool.imap_unordered(_work_date, dates):
            # collect this date's rows
            rows.extend(res)
            # bump the counter
            done += 1
            # elapsed minutes
            el = (time.perf_counter() - t0) / 60.0
            # per-date progress + ETA
            print(_ts() + f"  date {done}/{len(dates)} ({len(rows)} sym-days, {el:.1f} min, "
                  f"ETA {el/done*(len(dates)-done):.1f} min)")
    # nothing collected
    if not rows:
        # report and stop
        print(_ts() + "no rows.")
        # exit
        return
    # assemble the full per-symbol-day frame
    df = pd.DataFrame(rows)
    # ensure the output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    # write the per-symbol-day CSV
    df.to_csv(out_dir / "iceberg_detect_pername_day.csv", index=False)
    # total sweeps
    sw = df["n_sweeps"].sum()
    # total M1/M2 icebergs
    ice = df["n_iceberg_m2"].sum()
    # total big sweeps
    big = df["n_big_sweeps"].sum()
    # header
    print(_ts() + "===== ICEBERG DETECTION =====")
    # M1/M2 post-sweep prevalence
    print(_ts() + f"  [M1/M2 post-sweep]  sweeps {sw:,} | iceberg {ice:,} "
          f"({100*ice/max(sw,1):.1f}% of all, {100*ice/max(big,1):.1f}% of big M2>={M2_MIN})")
    # live refill-wall prevalence
    print(_ts() + f"  [live refill walls] confirmed(>=3 cycles): {df['n_confirmed_walls'].sum():,} | "
          f"refill events: {df['n_refill_events'].sum():,} | max cycles seen: {df['max_cycles'].max()}")
    # pairing between the two signals
    print(_ts() + f"  [pairing] M2 sweeps that hit a live-flagged wall: {df['n_agreements'].sum():,}")
    # per-name aggregation
    pern = df.groupby("symbol").agg(sw=("n_sweeps", "sum"), ice=("n_iceberg_m2", "sum"),
                                    conf=("n_confirmed_walls", "sum"))
    # keep names with enough sweeps to be meaningful
    pern = pern[pern["sw"] >= 50].sort_values("conf", ascending=False)
    # write the per-name CSV
    pern.to_csv(out_dir / "iceberg_detect_pername.csv")
    # top names by confirmed walls
    print(_ts() + "  top names by confirmed walls:")
    # print the leaders
    for s_, r in pern.head(8).iterrows():
        # one line per name
        print(_ts() + f"    {s_}: {int(r['conf'])} walls, {int(r['ice'])} M2-icebergs / {int(r['sw'])} sweeps")
    # done
    print(_ts() + f"[iceberg] outputs -> {out_dir}")


# time ONE stock-day in this process (no pool): a fast crash/timing check
def smoke(symbols=None):
    # the reader module
    import run_legacy_mm as R
    # point at the local store
    R.PARSED_ROOT = LOCAL_STORE
    # the engine module with the Book class
    import mm_backtest as MB
    # available dates
    dates = R.discover_dates()
    # report how many were found
    print(_ts() + f"discover_dates -> {len(dates)} dates")
    # stop if none
    if not dates:
        # exit
        return
    # a mid-panel date
    date = dates[len(dates) // 2]
    # open its datasets
    dsets = R.open_datasets(date)
    # pick the requested symbol, else the first traded one
    sym = symbols[0] if symbols else R.list_symbols(dsets["trades"])[0]
    # set the module globals _one() relies on
    global _R, _BOOK
    # reader
    _R = R
    # Book class
    _BOOK = MB.Book
    # time one scan
    t0 = time.perf_counter()
    # run it
    r = _one(date, sym)
    # elapsed
    dt = time.perf_counter() - t0
    # report result and timing
    print(_ts() + f"[smoke] {date} {sym} in {dt:.1f}s -> {r}")


# validate the detector logic on a synthetic event stream with a known answer
def self_test():
    # the engine module for the real Book
    import mm_backtest as MB

    # a duck-typed event payload (mimics an itertuples row)
    class Row:
        # set arbitrary attributes from kwargs
        def __init__(self, **k):
            # store them
            self.__dict__.update(k)

    # timestamps are in MILLISECONDS (matches build_events ts_exch); 1 ms unit
    MS = 1
    # the synthetic event list
    ev = []
    # seed asks 100..104 with 50 qty each as ADDs at t=0
    for i, px in enumerate([100, 101, 102, 103, 104]):
        # append an ORDER_ADD on the SELL side
        ev.append((0, 1, i, "U", Row(event="ORDER_ADD", side="SELL", price=float(px), qty=50.0, order_id=f"a{i}")))
    # a BUY sweep of 200 printing only at 100 at t=5s -> M1=1, walk asks -> M2=4
    for j in range(4):
        # append a BUY trade print at price 100
        ev.append((5000 * MS, 1, 10 + j, "T", Row(aggressor_side="BUY", price=100.0, qty=50.0)))
    # base time for the refill episode
    tbase = 6000 * MS
    # three refill cycles at ASK 100: trade hit @100, then ADD @100 within 100ms
    for c in range(3):
        # the hit time for this cycle
        th = tbase + c * 1000 * MS
        # a small BUY trade that hits ASK 100
        ev.append((th, 1, 100 + c, "T", Row(aggressor_side="BUY", price=100.0, qty=10.0)))
        # a refill ADD at ASK 100, 30ms later (inside the window)
        ev.append((th + 30 * MS, 1, 200 + c, "U", Row(event="ORDER_ADD", side="SELL", price=100.0, qty=10.0, order_id=f"r{c}")))
    # sort by (ts, rank, seq) like build_events does
    ev.sort(key=lambda e: (e[0], e[1], e[2]))
    # run the detector
    r = scan_events(ev, MB.Book)
    # show the result
    print(_ts() + f"[self-test] {r}")
    # exactly one M1/M2 iceberg expected
    assert r["n_iceberg_m2"] == 1, f"expected 1 M2 iceberg, got {r['n_iceberg_m2']}"
    # three refill cycles expected
    assert r["max_cycles"] >= 3, f"expected 3 refill cycles, got {r['max_cycles']}"
    # at least one confirmed wall expected
    assert r["n_confirmed_walls"] >= 1, "expected a confirmed wall at ASK 100"
    # negative case: refills 500ms apart must NOT confirm a wall
    ev2 = [e for e in ev if not (e[3] == "U" and getattr(e[4], "order_id", "").startswith("r"))]
    # add slow refills 500ms after each hit
    for c in range(3):
        # the hit time for this cycle
        th = tbase + c * 1000 * MS
        # a refill ADD 500ms later (outside the window)
        ev2.append((th + 500 * MS, 1, 300 + c, "U", Row(event="ORDER_ADD", side="SELL", price=100.0, qty=10.0, order_id=f"s{c}")))
    # re-sort
    ev2.sort(key=lambda e: (e[0], e[1], e[2]))
    # run again
    r2 = scan_events(ev2, MB.Book)
    # show the slow-refill result
    print(_ts() + f"[self-test] slow-refill (500ms): confirmed={r2['n_confirmed_walls']} cycles={r2['max_cycles']}")
    # no confirmed wall expected when refills are outside the 100ms window
    assert r2["n_confirmed_walls"] == 0, "500ms refills must NOT confirm a wall"
    # all checks passed
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


# CLI entry point
if __name__ == "__main__":
    # argument parser
    ap = argparse.ArgumentParser()
    # run the self-test
    ap.add_argument("--self-test", action="store_true")
    # time one stock-day
    ap.add_argument("--smoke", action="store_true")
    # do the full run
    ap.add_argument("--run", action="store_true")
    # restrict to specific symbols
    ap.add_argument("--symbols", nargs="*", default=None)
    # worker count
    ap.add_argument("--workers", type=int, default=WORKERS)
    # day count (0 = all)
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    # parse
    a = ap.parse_args()
    # smoke mode
    if a.smoke:
        # time one stock-day
        smoke(symbols=a.symbols)
    # otherwise default to self-test unless --run
    elif a.self_test or not a.run:
        # run the self-test
        self_test()
    # full run
    if a.run:
        # scan the universe
        run_real(symbols=a.symbols, workers=a.workers, max_days=(a.days or None))
