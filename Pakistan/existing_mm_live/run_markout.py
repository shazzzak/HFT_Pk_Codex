# ============================================================================
# run_markout.py -- TABLE B: after a run of k consecutive same-side aggressor
# ORDERS (sweep-collapsed distinct orders), how much does price move IN THE RUN'S
# DIRECTION over the next 1/3/5s? That forward move IS the adverse markout the
# maker's EXPOSED side suffers (a buy-run lifts our resting ASK, then price runs
# up against the short). If it grows with k and beats fee+forfeited-capture, a
# graded skew/step-aside-on-long-runs gate could pay. Plus the ADDS-ON-TOP-OF-OBI
# check (does run length predict adverse markout even when OBI is calm?).
#
# Rides R.build_events + mm_backtest.Book (validated aggregation, ms timestamps).
# Run length is on COLLAPSED distinct orders (same exchange ts + same side = ONE
# order), matching the collapsed ladder (k=1 ~50%, rising to 75% at 6+).
#
# momentum_h(order) = sign(run_side) * (mid[t+h] - mid[t]) / mid[t] * 1e4
#   +ve => price CONTINUED with the run => ADVERSE for the exposed maker side.
#
# USAGE:  python run_markout.py --self-test | --smoke | --run
# ============================================================================

# CLI parsing
import argparse
# timing / heartbeat
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


# NEW data store (moved this session)
LOCAL_STORE = Path("/Users/shazzak/HFT Data/Pakistan/Capital Stake - Parsed")
# results root (watchlist name list)
RESULTS_ROOT = Path("/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results")
# output directory
OUT_DIR = RESULTS_ROOT / "diagnostics"
# forward horizons in seconds (the ones SZ named)
HORIZONS_S = [1, 3, 5]
# run-length buckets (match the collapsed ladder: 5 exact, 6 = 6+)
RUN_BUCKETS = [1, 2, 3, 4, 5, 6]
# OBI "calm" threshold: |obi_1| below this = OBI sees nothing (incumbent trigger 0.30)
OBI_CALM = 0.30
# default sampled days
MAX_DAYS = 20
# default workers
WORKERS = 6
# minimum orders in a (name-day, run bucket) cell to trust that day's mean
MIN_ROWS = 20


# ------------------------------------------------------------------ core -----
# one observing pass over the merged S/U/T stream; returns a per-order DataFrame
# with (bucket, run_len bucket, obi, mom_1/3/5) rows for this symbol-day.
def scan(events, Book):
    # the engine's reconstructed book
    book = Book()
    # mid timeline for forward lookups: parallel (ts_ms, mid)
    mid_t = []
    # mid values
    mid_v = []
    # per distinct-order records
    recs = []
    # collapse state: last order's exchange ts and side
    last_ts = None
    # last order's side (+1 buy / -1 sell)
    last_side = None
    # current consecutive-run length (distinct orders, same side)
    run = 0
    # side of the current run
    run_side = 0

    # current mid from the book (None if one-sided)
    def _mid_obi():
        # best bid/ask + their sizes
        bb, bq, ba, aq = book.bbo()
        # need both sides for a mid
        if bb is None or ba is None:
            # no two-sided book
            return None, None
        # midpoint
        m = 0.5 * (bb + ba)
        # L1 order-book imbalance in [-1,+1]; 0 if sizes missing
        denom = (bq or 0) + (aq or 0)
        # imbalance or 0
        ob = ((bq - aq) / denom) if denom > 0 else 0.0
        # return both
        return m, ob

    # walk the time-ordered merged stream
    for (ts, rank, seq, kind, obj) in events:
        # a trade print
        if kind == "T":
            # aggressor side: +1 buy, -1 sell
            side = 1 if str(getattr(obj, "aggressor_side", "")).upper().startswith("B") else -1
            # COLLAPSE: a new distinct order iff timestamp changed OR side flipped
            is_new_order = (last_ts is None) or (ts != last_ts) or (side != last_side)
            # update collapse state to this print
            last_ts = ts
            # remember the side
            last_side = side
            # only a NEW distinct order advances the run + gets recorded
            if is_new_order:
                # extend the run if same side, else start a new run of length 1
                if side == run_side:
                    # same-side distinct order -> longer run
                    run += 1
                else:
                    # side flipped -> new run
                    run = 1
                    # remember the run's side
                    run_side = side
                # read mid + obi at THIS order's instant (pre-forward)
                m, ob = _mid_obi()
                # record only when the book is two-sided (mid exists)
                if m is not None:
                    # bucket the run length (cap 6 = "6+")
                    rb = run if run < 6 else 6
                    # store (ts, run bucket, run side, obi, mid-now)
                    recs.append((ts, rb, run_side, ob, m))
            # apply the trade to the book (validated decrement / __NEG_ handling)
            try:
                # decrement resting qty for the fill
                book.trade(obj)
            except Exception:
                # engine edge cases (auction prints) -> ignore
                pass
        # a book update
        elif kind == "U":
            # add or cancel on the engine book
            if getattr(obj, "event", None) == "ORDER_ADD":
                # new resting order
                book.add(obj)
            else:
                # cancel
                book.cancel(obj)
        # a snapshot event
        elif kind == "S":
            # snapshots don't change the collapse/run logic here
            pass
        # after every event, record the mid if the book is two-sided
        m, _ = _mid_obi()
        # dense mid timeline for forward lookups
        if m is not None:
            # timestamp (ms)
            mid_t.append(ts)
            # mid value
            mid_v.append(m)

    # nothing to analyze
    if not recs or not mid_t:
        # empty frame
        return None
    # mid timeline as arrays
    mt = np.asarray(mid_t, dtype=float)
    # mid values
    mv = np.asarray(mid_v, dtype=float)
    # records to frame
    df = pd.DataFrame(recs, columns=["ts", "run", "run_side", "obi", "mid0"])
    # session bounds from the mid timeline (for the bucket split)
    t0, t1 = mt.min(), mt.max()
    # 15/45/15-min windows in ms
    F = 15 * 60 * 1000; P = 45 * 60 * 1000; L = 15 * 60 * 1000
    # default bucket = middle
    b = np.full(len(df), "middle", dtype=object)
    # first 15 min
    b[df["ts"].to_numpy() <= t0 + F] = "first15"
    # 45->15 before close
    b[df["ts"].to_numpy() >= t1 - P] = "preclose45"
    # last 15 min (overwrites preclose in its range)
    b[df["ts"].to_numpy() >= t1 - L] = "last15"
    # attach bucket
    df["bucket"] = b
    # forward momentum in the run direction at each horizon
    for h in HORIZONS_S:
        # first mid at/after order ts + h seconds
        i1 = np.searchsorted(mt, df["ts"].to_numpy() + h * 1000.0, side="left")
        # valid if within the session
        ok = i1 < mt.size
        # forward mid
        mid1 = np.where(ok, mv[np.clip(i1, 0, mt.size - 1)], np.nan)
        # signed by run direction: +ve = price ran WITH the run (adverse to exposed side)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"mom_{h}"] = df["run_side"].to_numpy() * (mid1 - df["mid0"].to_numpy()) / df["mid0"].to_numpy() * 1e4
    # drop the raw mid col
    return df.drop(columns=["mid0"])


# per-process globals
_R = None
_BOOK = None
_NAMES = None


# pool initializer: driver + engine + local store + universe
def _init(names):
    # expose globals
    global _R, _BOOK, _NAMES
    # parquet driver
    import run_legacy_mm as R
    # point at the new local store
    R.PARSED_ROOT = LOCAL_STORE
    # engine (Book)
    import mm_backtest as MB
    # stash
    _R = R
    # Book class
    _BOOK = MB.Book
    # resolve the universe: watchlist CSV else all symbols
    if names is None:
        # try the 38-name watchlist
        try:
            # read it from the new results root
            wl = pd.read_csv(RESULTS_ROOT / "mm_watchlist_final.csv")
            # symbol column
            names = sorted(wl["symbol"].dropna().astype(str).unique().tolist())
        except Exception:
            # fall back to all traded symbols
            names = None
    # stash
    _NAMES = names


# needed columns for the three tables
_U = None


# one symbol-day -> per-order frame
def _one(date, sym):
    # open datasets
    dsets = _R.open_datasets(date)
    # updates / snapshots / trades for this symbol
    u = _R.read_symbol(dsets["ob_updates"], _R.REQ_UPDATES, sym)
    # snapshots
    s = _R.read_symbol(dsets["ob_snapshot"], _R.REQ_SNAP, sym)
    # trades
    t = _R.read_symbol(dsets["trades"], _R.REQ_TRADES, sym)
    # unrunnable without trades + a book
    if len(t) == 0 or len(s) == 0:
        # skip
        return None
    # merged, time-ordered event stream (adds ts_exch)
    events, snap_groups, t = _R.build_events(u, s, t)
    # observe -> per-order frame
    return scan(events, _BOOK)


# all symbols for one date
def _work_date(date):
    # datasets
    dsets = _R.open_datasets(date)
    # missing partition
    if dsets is None:
        # nothing
        return []
    # universe
    names = _NAMES or _R.list_symbols(dsets["trades"])
    # collected frames
    out = []
    # loop symbols
    for sym in names:
        # guard per symbol
        try:
            # run
            df = _one(date, sym)
        except Exception as e:
            # report + continue
            print(_ts() + f"SKIP {date} {sym}: {e!r}")
            # next
            continue
        # keep non-empty
        if df is not None and len(df):
            # tag
            df["date"] = str(date)
            # tag
            df["symbol"] = sym
            # collect
            out.append(df)
    # this date's frames
    return out


# day-as-unit mean +/- SE at a run bucket, optionally OBI-calm only
def _cell(df, rb, hcol, obi_calm_only):
    # filter to the run bucket
    sub = df[df["run"] == rb]
    # optionally restrict to OBI-calm orders
    if obi_calm_only:
        # |obi| below the incumbent trigger
        sub = sub[sub["obi"].abs() < OBI_CALM]
    # per name-day mean of the momentum, requiring enough orders that day
    g = sub.groupby(["date", "symbol"])[hcol].agg(["mean", "count"]).reset_index()
    # keep name-days with enough orders in this cell
    g = g[g["count"] >= MIN_ROWS]
    # the per-name-day means
    vals = g["mean"].to_numpy()
    # need >=2 name-days
    if vals.size < 2:
        # not enough
        return np.nan, np.nan, vals.size
    # mean and SE across name-days
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(vals.size)), vals.size


# print one table (all rows or OBI-calm rows) across run buckets x horizons
def _print_table(df, obi_calm_only, title):
    # header
    print(_ts() + f"  --- {title} ---")
    # column header
    print(_ts() + "     run   " + "   ".join(f"{h}s(adv bps)" for h in HORIZONS_S) + "   [name-days]")
    # each run bucket
    for rb in RUN_BUCKETS:
        # build the row across horizons (all share the same name-day count roughly)
        cells = []
        # last name-day count for display
        nlast = 0
        # each horizon
        for h in HORIZONS_S:
            # compute the cell
            m, se, n = _cell(df, rb, f"mom_{h}", obi_calm_only)
            # remember n
            nlast = n
            # format
            cells.append(f"{m:+6.2f}+/-{se:4.2f}" if not np.isnan(m) else "   n/a    ")
        # label 6 as 6+
        lab = f"{rb}" if rb < 6 else "6+"
        # print the row
        print(_ts() + f"     {lab:>3}   " + "   ".join(cells) + f"   [{nlast}]")


# full run
def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    # driver for dates
    import run_legacy_mm as R
    # local store
    R.PARSED_ROOT = LOCAL_STORE
    # dates
    dates = R.discover_dates()
    # guard empty
    if not dates:
        # stop
        print(_ts() + "discover_dates() empty -- check PARSED_ROOT."); return
    # sample days
    if max_days and len(dates) > max_days:
        # stride
        step = max(1, len(dates) // max_days)
        # sample
        dates = dates[::step][:max_days]
    # announce
    print(_ts() + f"{len(dates)} dates, {workers} workers -- run-length vs forward markout")
    # accumulate frames
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
        print(_ts() + "no orders collected."); return
    # concat
    df = pd.concat(frames, ignore_index=True)
    # ensure dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # save per-order rows (big but useful)
    df.to_csv(out_dir / "run_markout_orders.csv", index=False)
    # headline
    print(_ts() + "===== RUN-LENGTH vs FORWARD MARKOUT (adverse to exposed side, bps; day-as-unit) =====")
    print(_ts() + "  +ve = price ran WITH the run = ADVERSE for the maker side facing it. fee ~1.55 bps RT.")
    # Table B1: all orders, pooled across buckets
    _print_table(df, obi_calm_only=False, title="ALL orders (all buckets)")
    # Table B2: adds on top of OBI (OBI-calm orders only)
    _print_table(df, obi_calm_only=True, title="OBI-CALM only (|obi_1|<%.2f) -- does run predict when OBI is quiet?" % OBI_CALM)
    # Table B3: middle bucket only (the business), all orders
    mid = df[df.bucket == "middle"]
    # print if present
    if len(mid):
        # middle-only view
        _print_table(mid, obi_calm_only=False, title="MIDDLE bucket only")
    # read
    print(_ts() + "  READ: rising adverse bps with run length, LARGER than fee+capture, esp at k>=5 and")
    print(_ts() + "        still present in OBI-CALM -> a graded step-aside-on-long-runs gate could pay.")
    print(_ts() + "        flat / < fee / vanishes when OBI-calm -> it's momentum OBI already sees (drop).")
    # location
    print(_ts() + f"[run-markout] outputs -> {out_dir}")


# one stock-day, timed
def smoke(symbols=None):
    # driver
    import run_legacy_mm as R
    # local store
    R.PARSED_ROOT = LOCAL_STORE
    # engine
    import mm_backtest as MB
    # dates
    dates = R.discover_dates()
    # report
    print(_ts() + f"discover_dates -> {len(dates)} dates")
    # stop if none
    if not dates:
        # exit
        return
    # mid date
    date = dates[len(dates) // 2]
    # datasets
    dsets = R.open_datasets(date)
    # symbol
    sym = symbols[0] if symbols else (R.list_symbols(dsets["trades"])[0])
    # set globals _one needs
    global _R, _BOOK
    # driver
    _R = R
    # Book
    _BOOK = MB.Book
    # time one
    t0 = time.perf_counter()
    # run
    df = _one(date, sym)
    # elapsed
    dt = time.perf_counter() - t0
    # report
    print(_ts() + f"[smoke] {date} {sym} in {dt:.1f}s: {0 if df is None else len(df)} distinct orders")
    # quick table if any
    if df is not None and len(df):
        # tag
        df["date"] = str(date); df["symbol"] = sym
        # print the all-orders table (single name-day, so SE will be nan -> shows the point estimate via count)
        _print_table(df, obi_calm_only=False, title="smoke (single name-day)")


# validate momentum + collapse + run-counter math on a hand-built stream
def self_test():
    # engine Book
    import mm_backtest as MB

    # duck-typed event payload
    class Row:
        # store kwargs
        def __init__(self, **k):
            # set attributes
            self.__dict__.update(k)

    # 1 ms in ms-units (build_events ts_exch is ms)
    MS = 1
    # event list
    ev = []
    # seed a two-sided book: bid 100 (50), ask 101 (50)
    ev.append((0, 1, 0, "U", Row(event="ORDER_ADD", side="BUY", price=100.0, qty=50.0, order_id="b0")))
    # ask side
    ev.append((0, 1, 1, "U", Row(event="ORDER_ADD", side="SELL", price=101.0, qty=50.0, order_id="a0")))
    # add deeper asks so a run of buys can walk price up as they lift levels
    for i, px in enumerate([102, 103, 104, 105, 106, 107], start=1):
        # deeper ask liquidity
        ev.append((0, 1, 1 + i, "U", Row(event="ORDER_ADD", side="SELL", price=float(px), qty=50.0, order_id=f"a{i}")))
    # a run of 6 BUY orders at distinct timestamps, each lifting the touch up 1 tick
    # (as each ask level is consumed the mid rises -> price runs WITH the buy run)
    for k in range(6):
        # buy order at t = (k+1) seconds
        tk = (k + 1) * 1000 * MS
        # a BUY trade lifting the current best ask (qty 50 = full level)
        ev.append((tk, 1, 10 + k, "T", Row(aggressor_side="BUY", price=float(101 + k), qty=50.0)))
        # re-post a bid so the book stays two-sided and the mid tracks up
        ev.append((tk, 1, 100 + k, "U", Row(event="ORDER_ADD", side="BUY", price=float(101 + k), qty=50.0, order_id=f"nb{k}")))
    # sort by (ts, rank, seq)
    ev.sort(key=lambda e: (e[0], e[1], e[2]))
    # scan
    df = scan(ev, MB.Book)
    # must have produced orders
    assert df is not None and len(df) >= 5, f"no orders: {None if df is None else len(df)}"
    # run lengths should increase 1,2,3,... for the consecutive buys
    runs = df.sort_values("ts")["run"].tolist()
    # show
    print(_ts() + f"[self-test] run sequence: {runs}")
    # first few must be 1,2,3,4,5 (6th capped at 6)
    assert runs[:5] == [1, 2, 3, 4, 5], f"run counter wrong: {runs}"
    # momentum at each order should be >=0 (price ran up WITH the buy run) for early orders
    early = df.sort_values("ts").iloc[:4]
    # 3s momentum should be non-negative (price kept rising)
    assert (early["mom_3"].fillna(0) >= -1e-9).all(), f"expected +ve momentum on a rising buy run: {early['mom_3'].tolist()}"
    # COLLAPSE check: two same-ts+side prints must count as ONE order
    ev2 = [(0, 1, 0, "U", Row(event="ORDER_ADD", side="BUY", price=100.0, qty=50.0, order_id="b0")),
           (0, 1, 1, "U", Row(event="ORDER_ADD", side="SELL", price=101.0, qty=200.0, order_id="a0")),
           # two BUY prints at the SAME ts -> one collapsed order
           (1000, 1, 2, "T", Row(aggressor_side="BUY", price=101.0, qty=50.0)),
           (1000, 1, 3, "T", Row(aggressor_side="BUY", price=101.0, qty=50.0)),
           (1000, 1, 4, "U", Row(event="ORDER_ADD", side="BUY", price=101.0, qty=50.0, order_id="nb"))]
    # sort
    ev2.sort(key=lambda e: (e[0], e[1], e[2]))
    # scan
    d2 = scan(ev2, MB.Book)
    # the two same-ts buys collapse to a single order (run==1), not two
    assert d2 is not None and len(d2) == 1 and int(d2.iloc[0]["run"]) == 1, f"collapse failed: {None if d2 is None else d2['run'].tolist()}"
    # print
    print(_ts() + "[self-test] collapse: two same-ts buys -> 1 order (run=1) OK")
    # all good
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


# entry point
if __name__ == "__main__":
    # parser
    ap = argparse.ArgumentParser()
    # self-test
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
    # full run
    if a.run:
        # scan
        run_real(symbols=a.symbols, workers=a.workers, max_days=(a.days or None))
