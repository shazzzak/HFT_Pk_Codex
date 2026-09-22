# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# iceberg_sanity.py -- does M1/M2 "absorption" hold price, or was the depth illusory?
# For every sweep we record the post-sweep mid and the mid 5s/30s/60s later, then
# compare the FORWARD move (in the aggressor's direction, bps) for three groups:
#   flagged   : M1<=2 & M2>=4  (the "absorbed" sweeps)
#   walked    : M2>=4 & M1>2   (big sweeps that DID walk the book -- the control)
#   small     : M2<4           (too small to test)
# If flagged sweeps show ~0 forward move while walked sweeps continue, the absorbing
# level really HELD (absorption is real, if slow/human). If flagged sweeps CATCH UP
# (large positive forward move), the depth M2 walked was illusory -> artifact.
# Rides R.build_events + mm_backtest.Book (validated aggregation, ms timestamps).
# USAGE:  python iceberg_sanity.py --self-test | --smoke | --run
# ============================================================================

# CLI parsing
import argparse
# timing for heartbeats
import time
# timestamps on printed lines
from datetime import datetime
# parallel over dates
from multiprocessing import Pool
# filesystem paths
from pathlib import Path
# numerics
import numpy as np
# dataframes
import pandas as pd


# bracketed HH:MM:SS stamp for log lines
def _ts():
    # format now
    return datetime.now().strftime("[%H:%M:%S]")


# output directory
# Resolve this filesystem path through the canonical checkout/data configuration.
OUT_DIR = Path(str(_hft_paths.RESULTS_ROOT / 'diagnostics'))
# local parsed store (run_legacy_mm's default points at the empty Google Drive)
# Resolve this filesystem path through the canonical checkout/data configuration.
LOCAL_STORE = Path(str(_hft_paths.PARSED_ROOT))
# M1/M2 gate (same as the detector)
M1_MAX = 2
# minimum expected levels to count a sweep as "big"
M2_MIN = 4
# forward horizons in seconds
HORIZONS_S = [5, 30, 60]
# default liquid names for the quick check (icebergs live on liquid names)
DEFAULT_SYMS = ["OGDC", "PPL", "PSO", "HBL", "FFC", "NBP"]
# default parallel workers
WORKERS = 6
# default sampled days (a 20-minute budget)
MAX_DAYS = 10


# count how many best-first levels a total quantity would consume
def _walk_levels(levels, total):
    # cumulative visible quantity
    c = 0.0
    # walk best-first
    for i, (_, q) in enumerate(levels, 1):
        # add this level
        c += q
        # covered the swept quantity
        if c >= total:
            # levels consumed
            return i
    # exceeds all visible depth
    return len(levels)


# one pass: record every sweep with its post-sweep mid, plus the mid timeline
def collect_sweeps(events, Book):
    # the engine's reconstructed book
    book = Book()
    # cached ranked depth, recomputed only when the book changes
    cache = {"dirty": True, "bids": [], "asks": []}

    # top-N levels per side from the cache
    def _depth():
        # recompute if stale
        if cache["dirty"]:
            # engine aggregation
            cache["bids"], cache["asks"] = book.ranked_depth(n=10)
            # now clean
            cache["dirty"] = False
        # cached levels
        return cache["bids"], cache["asks"]

    # mid timeline: parallel lists of (ts_ms, mid)
    mid_t = []
    # mid values
    mid_v = []
    # per-sweep records: (ts, buy, m1, m2, flag, pre_depth_ok)
    sweeps = []
    # running cluster: [ts, buy, qty, {prices}, pre_bids, pre_asks]
    cur = None

    # current mid from the book, or None if one-sided
    def _mid():
        # best bid/ask
        bb, _, ba, _ = book.bbo()
        # need both sides
        if bb is None or ba is None:
            # no mid
            return None
        # midpoint
        return 0.5 * (bb + ba)

    # finalize a cluster into a sweep record (M2 uses the PRE-sweep depth we cached)
    def close_sweep(cl):
        # nothing to close
        if cl is None:
            # done
            return
        # unpack
        ts, buy, qty, prices, pre_bids, pre_asks = cl
        # M1 = distinct print prices
        m1 = len(prices)
        # walk the hit side of the PRE-sweep book
        levels = pre_asks if buy else pre_bids
        # no depth -> skip
        if not levels:
            # done
            return
        # M2 = expected levels consumed
        m2 = _walk_levels(levels, qty)
        # the iceberg flag
        flag = (m1 <= M1_MAX) and (m2 >= M2_MIN)
        # record the sweep with its timestamp (post-sweep mid looked up later)
        sweeps.append((ts, buy, m1, m2, flag))

    # walk the merged stream
    for (ts, rank, seq, kind, obj) in events:
        # a trade print
        if kind == "T":
            # aggressor side
            buy = str(getattr(obj, "aggressor_side", "")).upper().startswith("B")
            # print price
            px = float(obj.price)
            # print quantity
            q = float(obj.qty)
            # new cluster on ts/side change
            if cur is not None and (cur[0] != ts or cur[1] != buy):
                # finalize previous
                close_sweep(cur)
                # clear
                cur = None
            # open a cluster, snapshotting the PRE-sweep depth before any trade applies
            if cur is None:
                # pre-sweep ranked depth
                pb, pa = _depth()
                # [ts, buy, qty, prices, pre_bids, pre_asks]
                cur = [ts, buy, 0.0, set(), list(pb), list(pa)]
            # accumulate
            cur[2] += q
            # record price
            cur[3].add(px)
            # apply the trade to the book (validated decrement / __NEG_ handling)
            try:
                # decrement resting qty
                book.trade(obj)
                # book changed
                cache["dirty"] = True
            except Exception:
                # auction prints etc.
                pass
        # a book update
        elif kind == "U":
            # close a cluster before a later-ts non-trade event
            if cur is not None and cur[0] != ts:
                # finalize
                close_sweep(cur)
                # clear
                cur = None
            # add or cancel on the engine book
            if getattr(obj, "event", None) == "ORDER_ADD":
                # add
                book.add(obj)
            else:
                # cancel
                book.cancel(obj)
            # book changed
            cache["dirty"] = True
        # a snapshot
        elif kind == "S":
            # close any open cluster
            if cur is not None:
                # finalize
                close_sweep(cur)
                # clear
                cur = None
        # after every event, record the mid if the book is two-sided
        m = _mid()
        # keep a dense mid timeline for forward lookups
        if m is not None:
            # timestamp
            mid_t.append(ts)
            # mid value
            mid_v.append(m)
    # trailing cluster
    close_sweep(cur)

    # nothing to analyze
    if not sweeps or not mid_t:
        # empty
        return None
    # mid timeline as arrays
    mt = np.asarray(mid_t, dtype=float)
    # mid values
    mv = np.asarray(mid_v, dtype=float)
    # sweeps as a frame
    df = pd.DataFrame(sweeps, columns=["ts", "buy", "m1", "m2", "flag"])
    # post-sweep mid: first mid at/after the sweep timestamp
    i0 = np.searchsorted(mt, df["ts"].to_numpy(), side="left")
    # clamp to valid
    ok0 = i0 < mt.size
    # post-sweep mid
    df["mid0"] = np.where(ok0, mv[np.clip(i0, 0, mt.size - 1)], np.nan)
    # aggressor sign: +1 buy (up = continued), -1 sell
    sgn = np.where(df["buy"].to_numpy(), 1.0, -1.0)
    # forward move at each horizon, signed in the aggressor's direction (bps)
    for h in HORIZONS_S:
        # first mid at/after ts + h seconds (ms timestamps)
        i1 = np.searchsorted(mt, df["ts"].to_numpy() + h * 1000.0, side="left")
        # valid if within the day
        ok1 = i1 < mt.size
        # forward mid
        mid1 = np.where(ok1, mv[np.clip(i1, 0, mt.size - 1)], np.nan)
        # signed forward move in bps: positive = price continued through the level
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"fwd_{h}"] = sgn * (mid1 - df["mid0"].to_numpy()) / df["mid0"].to_numpy() * 1e4
    # group label
    df["group"] = np.where(df["flag"], "flagged",
                  np.where(df["m2"] >= M2_MIN, "walked", "small"))
    # the per-sweep frame
    return df


# per-process globals
_R = None
_BOOK = None
_NAMES = None


# pool initializer
def _init(names):
    # expose globals
    global _R, _BOOK, _NAMES
    # reader module
    import run_legacy_mm as R
    # local store
    R.PARSED_ROOT = LOCAL_STORE
    # engine Book
    import mm_backtest as MB
    # stash
    _R = R
    # stash
    _BOOK = MB.Book
    # universe
    _NAMES = names or DEFAULT_SYMS


# one symbol-day -> per-sweep frame
def _one(date, sym):
    # datasets
    dsets = _R.open_datasets(date)
    # updates
    u = _R.read_symbol(dsets["ob_updates"], _R.REQ_UPDATES, sym)
    # snapshots
    s = _R.read_symbol(dsets["ob_snapshot"], _R.REQ_SNAP, sym)
    # trades
    t = _R.read_symbol(dsets["trades"], _R.REQ_TRADES, sym)
    # unrunnable
    if len(t) == 0 or len(s) == 0:
        # skip
        return None
    # merged stream
    events, snap_groups, t = _R.build_events(u, s, t)
    # collect sweeps + forward moves
    return collect_sweeps(events, _BOOK)


# all symbols for one date
def _work_date(date):
    # datasets
    dsets = _R.open_datasets(date)
    # missing
    if dsets is None:
        # nothing
        return []
    # collected frames
    out = []
    # per-symbol loop with heartbeat
    for i, sym in enumerate(_NAMES, 1):
        # guard per symbol
        try:
            # run
            df = _one(date, sym)
        except Exception as e:
            # report and continue
            print(_ts() + f"SKIP {date} {sym}: {e!r}")
            # next
            continue
        # keep
        if df is not None and len(df):
            # tag
            df["date"] = str(date)
            # tag
            df["symbol"] = sym
            # collect
            out.append(df)
        # heartbeat
        print(_ts() + f"  {date}: {i}/{len(_NAMES)} syms")
    # this date's frames
    return out


# summarize forward moves by group, day-as-unit-ish (per name-day means then average)
def _report(df):
    # header
    print(_ts() + "===== DOES M1/M2 'ABSORPTION' HOLD PRICE? (forward move in aggressor direction, bps) =====")
    # meaning
    print(_ts() + "  flagged = M1<=2&M2>=4 (absorbed?) | walked = big sweep that DID walk the book | small = M2<4")
    # per group x horizon
    for g in ["flagged", "walked", "small"]:
        # this group's sweeps
        sub = df[df["group"] == g]
        # skip empty
        if not len(sub):
            # next
            continue
        # per name-day mean per horizon, then mean +/- se across name-days
        parts = []
        # each horizon
        for h in HORIZONS_S:
            # name-day means
            nd = sub.groupby(["date", "symbol"])[f"fwd_{h}"].mean().dropna().to_numpy()
            # stats
            m = nd.mean() if nd.size else np.nan
            # standard error
            se = nd.std(ddof=1) / np.sqrt(nd.size) if nd.size > 1 else np.nan
            # cell
            parts.append(f"{h:>2d}s:{m:+6.2f}+/-{se:4.2f}")
        # line
        print(_ts() + f"  {g:8s} n={len(sub):>7,}   " + "   ".join(parts))
    # interpretation
    print(_ts() + "  READ: flagged ~0 (or <0) while walked >0  => the level HELD (absorption real).")
    print(_ts() + "        flagged strongly >0 (catching up)     => the walked depth was illusory (artifact).")


# full run
def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    # reader
    import run_legacy_mm as R
    # local store
    R.PARSED_ROOT = LOCAL_STORE
    # dates
    dates = R.discover_dates()
    # none
    if not dates:
        # stop
        print(_ts() + "discover_dates() empty."); return
    # sample days
    if max_days and len(dates) > max_days:
        # stride
        step = max(1, len(dates) // max_days)
        # sample
        dates = dates[::step][:max_days]
    # announce
    print(_ts() + f"{len(dates)} dates x {len(symbols or DEFAULT_SYMS)} names, {workers} workers")
    # frames
    frames = []
    # timer
    t0 = time.perf_counter()
    # pool
    with Pool(processes=workers, initializer=_init, initargs=(symbols,)) as pool:
        # counter
        done = 0
        # consume
        for res in pool.imap_unordered(_work_date, dates):
            # collect
            frames.extend(res)
            # bump
            done += 1
            # elapsed
            el = (time.perf_counter() - t0) / 60.0
            # progress + ETA
            print(_ts() + f"  date {done}/{len(dates)} ({el:.1f} min, ETA {el/done*(len(dates)-done):.1f} min)")
    # nothing
    if not frames:
        # stop
        print(_ts() + "no sweeps collected."); return
    # concat
    df = pd.concat(frames, ignore_index=True)
    # ensure dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # write per-sweep rows
    df.to_csv(out_dir / "iceberg_sanity_sweeps.csv", index=False)
    # report
    _report(df)
    # done
    print(_ts() + f"[sanity] outputs -> {out_dir}")


# one stock-day, timed
def smoke(symbols=None):
    # reader
    import run_legacy_mm as R
    # local store
    R.PARSED_ROOT = LOCAL_STORE
    # engine
    import mm_backtest as MB
    # dates
    dates = R.discover_dates()
    # none
    if not dates:
        # stop
        print(_ts() + "no dates"); return
    # mid date
    date = dates[len(dates) // 2]
    # symbol
    sym = (symbols or DEFAULT_SYMS)[0]
    # globals
    global _R, _BOOK
    # set
    _R = R
    # set
    _BOOK = MB.Book
    # time it
    t0 = time.perf_counter()
    # run
    df = _one(date, sym)
    # elapsed
    dt = time.perf_counter() - t0
    # report
    print(_ts() + f"[smoke] {date} {sym} in {dt:.1f}s: {0 if df is None else len(df)} sweeps")
    # quick group summary if any
    if df is not None and len(df):
        # tag
        df["date"] = str(date)
        # tag
        df["symbol"] = sym
        # summarize
        _report(df)


# validate the forward-move math on a hand-built timeline
def self_test():
    # sweeps: a BUY at t=0 with mid0=100; mid at +5s=100.10 (+10bps), +30s=100.00, +60s=99.90
    df = pd.DataFrame({"ts": [0.0], "buy": [True], "m1": [1], "m2": [5], "flag": [True]})
    # mid timeline (ms, value)
    mt = np.array([0.0, 5000.0, 30000.0, 60000.0])
    # mids
    mv = np.array([100.0, 100.10, 100.0, 99.90])
    # replicate the forward-move computation
    i0 = np.searchsorted(mt, df["ts"].to_numpy(), side="left")
    # post-sweep mid (scalar: one test sweep)
    mid0 = float(mv[i0][0])
    # sign
    sgn = 1.0
    # expected forward moves in bps: +10, 0, -10
    exp = {5: 10.0, 30: 0.0, 60: -10.0}
    # each horizon
    for h in HORIZONS_S:
        # forward index
        i1 = np.searchsorted(mt, df["ts"].to_numpy() + h * 1000.0, side="left")
        # forward mid (scalar)
        mid1 = float(mv[i1][0])
        # signed move
        got = float(sgn * (mid1 - mid0) / mid0 * 1e4)
        # show
        print(_ts() + f"[self-test] fwd_{h}s = {got:+.3f} bps (expect {exp[h]:+.1f})")
        # check
        assert abs(got - exp[h]) < 1e-6, f"forward move wrong at {h}s"
    # a SELL sweep flips the sign: same mids -> -10, 0, +10
    got_s = float(-1.0 * (mv[1] - mv[0]) / mv[0] * 1e4)
    # show
    print(_ts() + f"[self-test] SELL fwd_5s = {got_s:+.3f} (expect -10.0)")
    # check
    assert abs(got_s + 10.0) < 1e-6
    # walk-levels sanity: 200 into [50,50,50,50,50] -> 4 levels
    assert _walk_levels([(1, 50), (2, 50), (3, 50), (4, 50), (5, 50)], 200) == 4
    # all good
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


# entry point
if __name__ == "__main__":
    # parser
    ap = argparse.ArgumentParser()
    # flags
    ap.add_argument("--self-test", action="store_true")
    # smoke
    ap.add_argument("--smoke", action="store_true")
    # run
    ap.add_argument("--run", action="store_true")
    # symbols
    ap.add_argument("--symbols", nargs="*", default=None)
    # workers
    ap.add_argument("--workers", type=int, default=WORKERS)
    # days
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    # parse
    a = ap.parse_args()
    # dispatch
    if a.smoke:
        # smoke
        smoke(symbols=a.symbols)
    elif a.self_test or not a.run:
        # self-test
        self_test()
    # run
    if a.run:
        # full
        run_real(symbols=a.symbols, workers=a.workers, max_days=(a.days or None))
