# ============================================================================
# obi_decay_gate.py -- does a DECAY-WEIGHTED multi-level OBI predict forward
# markout better than plain L1 OBI?
# ----------------------------------------------------------------------------
# The earlier obi_depth_gate tested CUMULATIVE EQUAL-WEIGHT OBI (obi_1/3/5/7)
# and found L1 best, deeper worse -- because equal weighting let a stale L5 order
# vote as loudly as the touch. This tests the fix SZ proposed: EXPONENTIAL DECAY
# weighting, so the touch dominates but deeper levels contribute a little:
#     wobi = SUM_i w_i*(bid_i - ask_i) / SUM_i w_i*(bid_i + ask_i),  w_i = rho^(i-1)
# swept over depth (L1-2 / L1-3 / L1-5) x decay rho (0.3 steep / 0.5 / 0.7 flat).
#
# THE DECISIVE TEST is not "does wobi predict" (it will, it contains L1) but
# "does wobi ADD on top of L1" -- i.e. does it predict forward markout among rows
# where plain L1 is NEUTRAL? If yes, deep levels carry conditional signal and a
# decay-weighted trigger would sharpen the throttle/queue-skew. If the adds-on-top
# gap is ~0, wobi collapses to L1 and we keep plain L1 (question closed cleanly).
#
# Rides R.build_events + mm_backtest.Book (validated aggregation), reads no feature
# store -- computes wobi and forward mid from the replay directly. Day-as-unit.
# USAGE: python obi_decay_gate.py --self-test | --smoke | --run
# ============================================================================

# CLI
import argparse
# timing
import time
# stamps
from datetime import datetime
# parallel
from multiprocessing import Pool
# numerics / frames
import numpy as np
import pandas as pd

# central paths
from config_pk import PARSED_ROOT, RESULTS_ROOT


# log stamp
def _ts():
    return datetime.now().strftime("[%H:%M:%S]")


# outputs
OUT_DIR = RESULTS_ROOT / "diagnostics"
# watchlist
WATCHLIST = RESULTS_ROOT / "mm_watchlist_final.csv"
# forward horizons (seconds) for the markout
HORIZONS_S = [1, 3, 5]
# depths to weight over
DEPTHS = [2, 3, 5]
# decay rates (rho): 0.3 steep (near-L1), 0.7 flat (more deep weight)
DECAYS = [0.3, 0.5, 0.7]
# "L1 neutral" band for the adds-on-top test (|L1 obi| < this = L1 says nothing)
L1_NEUTRAL = 0.10
# top/bottom decile for the predictive gap
DECILE = 0.10
# min rows per name-day cell
MIN_ROWS = 200
# sampled days / workers
MAX_DAYS = 20
WORKERS = 6


# decay-weighted OBI from ranked levels. bids/asks are best-first [(px,qty),...].
def wobi(bids, asks, depth, rho):
    # weights rho^0, rho^1, ... for levels 1..depth
    num = 0.0; den = 0.0
    # walk paired levels up to depth
    for i in range(depth):
        # weight for this level
        w = rho ** i
        # bid qty at level i (0 if absent)
        bq = bids[i][1] if i < len(bids) else 0.0
        # ask qty at level i
        aq = asks[i][1] if i < len(asks) else 0.0
        # weighted signed + total
        num += w * (bq - aq)
        den += w * (bq + aq)
    # imbalance in [-1,+1], 0 on an empty book
    return (num / den) if den > 0 else 0.0


# plain L1 obi from the touch sizes
def l1obi(bids, asks):
    # best bid/ask qty
    bq = bids[0][1] if bids else 0.0
    aq = asks[0][1] if asks else 0.0
    # imbalance
    return ((bq - aq) / (bq + aq)) if (bq + aq) > 0 else 0.0


# observe one symbol-day: at each event record L1 obi, each wobi variant, and mid;
# then attach forward side-signed markout. Returns a per-observation frame.
def scan(events, Book):
    # engine book
    book = Book()
    # mid timeline
    mt = []; mv = []
    # records
    recs = []
    # walk stream
    for (ts, rank, seq, kind, obj) in events:
        # apply the event to the book first
        if kind == "U":
            if getattr(obj, "event", None) == "ORDER_ADD":
                book.add(obj)
            else:
                book.cancel(obj)
        elif kind == "T":
            try: book.trade(obj)
            except Exception: pass
        # read ranked depth once (best-first, enough levels for max depth)
        bids, asks = book.ranked_depth(n=6)
        # need a two-sided touch
        if not bids or not asks:
            continue
        # mid
        m = 0.5 * (bids[0][0] + asks[0][0])
        # timeline
        mt.append(ts); mv.append(m)
        # only sample at a manageable cadence: every event is fine (thin book)
        # record L1 + each (depth,rho) wobi
        row = {"ts": ts, "mid": m, "l1": l1obi(bids, asks)}
        # each depth x decay
        for d in DEPTHS:
            for r in DECAYS:
                row[f"w_d{d}_r{r:g}"] = wobi(bids, asks, d, r)
        recs.append(row)
    # nothing
    if not recs or len(mt) < 50:
        return None
    # frame
    df = pd.DataFrame(recs)
    # arrays for forward lookup
    A = np.asarray(mt, float); V = np.asarray(mv, float)
    # forward side-signed markout at each horizon, per SIGNAL (sign by that signal)
    for h in HORIZONS_S:
        # forward mid index
        i1 = np.searchsorted(A, df["ts"].to_numpy() + h * 1000.0, side="left")
        ok = i1 < A.size
        mid1 = np.where(ok, V[np.clip(i1, 0, A.size - 1)], np.nan)
        # raw forward move in bps (unsigned; each signal signs it itself)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"fwd_{h}"] = (mid1 - df["mid"].to_numpy()) / df["mid"].to_numpy() * 1e4
    return df


# predictive gap for a signal: mean(sign(sig)*fwd | top decile |sig|) -- i.e. does
# a strong signal predict continuation? higher = more predictive. day-as-unit.
def _gap(df, sig, hcol):
    s = df[sig].to_numpy(); f = df[hcol].to_numpy()
    ok = ~(np.isnan(s) | np.isnan(f))
    s, f = s[ok], f[ok]
    if s.size < MIN_ROWS:
        return np.nan
    # top decile by |signal|
    thr = np.quantile(np.abs(s), 1 - DECILE)
    strong = np.abs(s) >= thr
    if strong.sum() < 20:
        return np.nan
    # momentum score on strong-signal rows
    return float(np.mean(np.sign(s[strong]) * f[strong]))


# ADDS-ON-TOP-OF-L1 via PARTIAL SPEARMAN done ENTIRELY IN RANK SPACE.
# BUG FIXED: the previous version regressed forward markout on L1 with OLS (a LINEAR
# fit) then evaluated the residual with Spearman (a RANK stat). OLS only forces
# PEARSON corr(L1, residual)=0 -- the RANK relationship survives, so the L1-self
# sanity check floored at ~-0.044 instead of 0. The old docstring claim that
# "corr(L1, residual)==0 by construction" is true for Pearson, FALSE for Spearman.
# FIX: rank-transform first, residualize BOTH sides on L1 in rank space, evaluate
# with Pearson on the rank residuals (== partial Spearman). Now:
#   - L1-self is 0 BY CONSTRUCTION (sr_resid is the zero vector) -> sanity reads ~0,
#   - a wobi variant "adds on top" iff its partial Spearman over L1 is clearly > 0.
# NOTE: still a SCREEN. A clean partial>0 must be confirmed by the engine-accurate
# P&L head-to-head (wire wobi into the trigger vs L1) -- do not close on rho alone.
def _partial_corr_over_l1(df, sig, hcol):
    # scipy for rankdata (Spearman == Pearson on ranks)
    from scipy import stats
    # L1 control (x), candidate signal (s), forward-markout label (f)
    x = df["l1"].to_numpy(); s = df[sig].to_numpy(); f = df[hcol].to_numpy()
    # keep only rows finite in all three so the arrays stay aligned
    ok = ~(np.isnan(x) | np.isnan(s) | np.isnan(f))
    x, s, f = x[ok], s[ok], f[ok]
    # too few rows for a stable rank correlation on this name-day
    if x.size < MIN_ROWS:
        return np.nan
    # rank-transform ALL THREE (average ranks handle ties) -- this IS the Spearman space
    xr = stats.rankdata(x); sr = stats.rankdata(s); fr = stats.rankdata(f)
    # center the control ranks (equivalent to an intercept; needed for orthogonality)
    xr_c = xr - xr.mean()
    # squared norm of the centered control ranks
    denom = float(xr_c @ xr_c)
    # degenerate control (all equal ranks) -> partialling undefined -> no incremental info
    if denom < 1e-12:
        return 0.0
    # rank residual of the LABEL on the control (OLS slope in rank space, then subtract fit)
    fr_resid = (fr - fr.mean()) - ((xr_c @ (fr - fr.mean())) / denom) * xr_c
    # rank residual of the CANDIDATE on the control (FULL partial residualizes BOTH sides)
    sr_resid = (sr - sr.mean()) - ((xr_c @ (sr - sr.mean())) / denom) * xr_c
    # candidate collinear with L1 in rank space (e.g. sig=='l1') -> its residual is ~0
    # -> 0 incremental BY CONSTRUCTION (this is why the sanity check now reads 0)
    if np.std(sr_resid) < 1e-12 or np.std(fr_resid) < 1e-12:
        return 0.0
    # partial Spearman = Pearson correlation of the two rank residuals
    rho = float(np.corrcoef(fr_resid, sr_resid)[0, 1])
    # NaN guard
    return rho if np.isfinite(rho) else np.nan


# per-process
_R = None; _BOOK = None; _NAMES = None
def _init(names):
    global _R, _BOOK, _NAMES
    import run_legacy_mm as R, mm_backtest as MB
    R.PARSED_ROOT = PARSED_ROOT
    _R = R; _BOOK = MB.Book
    if names is None:
        try:
            names = sorted(pd.read_csv(WATCHLIST)["symbol"].dropna().astype(str).unique().tolist())
        except Exception:
            names = None
    _NAMES = names


def _one(date, sym, dsets):
    u = _R.read_symbol(dsets["ob_updates"], _R.REQ_UPDATES, sym)
    s = _R.read_symbol(dsets["ob_snapshot"], _R.REQ_SNAP, sym)
    t = _R.read_symbol(dsets["trades"], _R.REQ_TRADES, sym)
    if len(s) == 0 or len(u) == 0:
        return None
    events, snap_groups, t = _R.build_events(u, s, t)
    return scan(events, _BOOK)


def _work_date(date):
    dsets = _R.open_datasets(date)
    if dsets is None:
        return []
    names = _NAMES or _R.list_symbols(dsets["trades"])
    rows = []
    for sym in names:
        try:
            df = _one(date, sym, dsets)
        except Exception as e:
            print(_ts() + f"SKIP {date} {sym}: {e!r}")
            continue
        if df is None or len(df) == 0:
            continue
        # headline horizon 5s; compute L1 and each wobi gap, all + L1-neutral
        h = "fwd_5"
        rec = {"date": str(date), "symbol": sym}
        # L1 baseline
        rec["l1_gap"] = _gap(df, "l1", h)
        # each wobi variant: overall gap + adds-on-top-of-L1 gap
        for d in DEPTHS:
            for r in DECAYS:
                sig = f"w_d{d}_r{r:g}"
                rec[f"{sig}_gap"] = _gap(df, sig, h)
                rec[f"{sig}_addl1"] = _partial_corr_over_l1(df, sig, h)
        # L1's own adds-on-top is by definition ~0 (it's neutral there) -- for ref
        rec["l1_addl1"] = _partial_corr_over_l1(df, "l1", h)
        rows.append(rec)
    return rows


def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    import run_legacy_mm as R
    R.PARSED_ROOT = PARSED_ROOT
    dates = R.discover_dates()
    if not dates:
        print(_ts() + "discover_dates empty."); return
    if max_days and len(dates) > max_days:
        step = max(1, len(dates) // max_days); dates = dates[::step][:max_days]
    print(_ts() + f"{len(dates)} dates, {workers} workers -- decay-weighted OBI gate")
    rows = []; t0 = time.perf_counter()
    with Pool(processes=workers, initializer=_init, initargs=(symbols,)) as pool:
        done = 0
        for res in pool.imap_unordered(_work_date, dates):
            rows.extend(res); done += 1
            el = (time.perf_counter() - t0) / 60.0
            print(_ts() + f"  date {done}/{len(dates)} ({el:.1f} min, ETA {el/done*(len(dates)-done):.1f} min)")
    if not rows:
        print(_ts() + "no rows."); return
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_parquet(df, out_dir / "obi_decay_gate.parquet")
    # day-as-unit mean of each gap
    def dau(col):
        x = df[col].dropna().to_numpy()
        return (x.mean(), x.std(ddof=1) / np.sqrt(len(x)), len(x)) if len(x) >= 2 else (np.nan, np.nan, len(x))
    # L1 reference
    l1m, l1se, l1n = dau("l1_gap")
    print(_ts() + "===== DECAY-WEIGHTED OBI vs L1 (forward markout gap @5s, day-as-unit) =====")
    print(_ts() + f"  L1 OBI baseline gap: {l1m:+.3f} +/- {l1se:.3f}  [name-days {l1n}]")
    print(_ts() + "  a wobi variant is only interesting if (a) its overall gap >= L1, AND")
    print(_ts() + "  (b) its ADDS-ON-TOP partial corr over L1 (unique signal) is clearly > 0.")
    print(_ts() + f"  {'variant':>12} {'overall_gap':>12} {'vs_L1':>8} {'partial_rho':>12}")
    best = None
    for d in DEPTHS:
        for r in DECAYS:
            sig = f"w_d{d}_r{r:g}"
            gm, gse, gn = dau(f"{sig}_gap")
            am, ase, an = dau(f"{sig}_addl1")
            vs = gm - l1m
            print(_ts() + f"  {sig:>12} {gm:+8.3f}+/-{gse:.2f} {vs:+8.3f} {am:+8.3f}+/-{ase:.2f}")
            # track the best adds-on-top
            if not np.isnan(am) and (best is None or am > best[1]):
                best = (sig, am)
    # verdict
    print(_ts() + f"  L1's own addl1 (should be ~0, sanity): {dau('l1_addl1')[0]:+.3f}")
    if best:
        print(_ts() + f"  STRONGEST adds-on-top: {best[0]} at {best[1]:+.3f}")
        print(_ts() + "  READ: if that adds-on-top is clearly >0 (and > L1's ~0), deep levels carry")
        print(_ts() + "        conditional signal -> wire decay-weighted OBI into the trigger + P&L test.")
        print(_ts() + "        if ~0, wobi collapses to L1 -> keep plain L1 (question closed).")
    print(_ts() + f"[obi-decay] outputs -> {out_dir}")


def _safe_parquet(df, path):
    import os as _os
    tmp = str(path) + ".tmp"
    try:
        df.to_parquet(tmp, index=False)
        import pyarrow.parquet as _pq
        assert _pq.ParquetFile(tmp).metadata.num_rows == len(df)
        _os.replace(tmp, path)
        print(_ts() + f"wrote {path} ({len(df)} rows, verified)")
    except Exception as e:
        try: _os.remove(tmp)
        except OSError: pass
        csv = str(path).rsplit(".", 1)[0] + ".csv"
        df.to_csv(csv, index=False)
        print(_ts() + f"parquet failed ({e!r}) -> CSV {csv}")


def smoke(symbols=None):
    import run_legacy_mm as R, mm_backtest as MB
    R.PARSED_ROOT = PARSED_ROOT
    dates = R.discover_dates()
    print(_ts() + f"discover_dates -> {len(dates)}")
    if not dates: return
    global _R, _BOOK; _R, _BOOK = R, MB.Book
    date = dates[len(dates) // 2]; dsets = R.open_datasets(date)
    sym = symbols[0] if symbols else R.list_symbols(dsets["trades"])[0]
    t0 = time.perf_counter(); df = _one(date, sym, dsets); dt = time.perf_counter() - t0
    print(_ts() + f"[smoke] {date} {sym} in {dt:.1f}s: {0 if df is None else len(df)} obs, "
          f"{0 if df is None else df.shape[1]} cols")


def self_test():
    # wobi math: 2 levels, bid-heavy L1 but ask-heavy L2, rho steep -> ~L1 sign
    bids = [(100.0, 900.0), (99.99, 100.0)]
    asks = [(100.01, 100.0), (100.02, 900.0)]
    # L1 obi: (900-100)/1000 = +0.8
    assert abs(l1obi(bids, asks) - 0.8) < 1e-9
    # depth-2, rho=0.3 (steep): w=[1,0.3]; num=1*(900-100)+0.3*(100-900)=800-240=560
    # den=1*1000+0.3*1000=1300 -> 560/1300=+0.4308 (still bid-heavy but pulled down by L2)
    w = wobi(bids, asks, 2, 0.3)
    assert abs(w - (560/1300)) < 1e-9, w
    print(_ts() + f"[self-test] wobi d2 r0.3 = {w:+.4f} (L1 +0.80 pulled toward 0 by ask-heavy L2)  OK")
    # rho=0.7 (flatter): w=[1,0.7]; num=800+0.7*(-800)=800-560=240; den=1000+700=1700 -> +0.141
    w2 = wobi(bids, asks, 2, 0.7)
    assert abs(w2 - (240/1700)) < 1e-9, w2
    # flatter decay weights L2 MORE -> pulls further from L1
    assert w2 < w, "flatter decay should pull further from L1"
    print(_ts() + f"[self-test] wobi d2 r0.7 = {w2:+.4f} (flatter decay pulls further from L1)  OK")
    # steep decay -> converges TOWARD L1 (not exact with a real ask-heavy L2 present,
    # since L2 still gets weight rho=0.05; it should be much closer to L1 than the
    # flatter-decay values above, which is the property that matters)
    w3 = wobi(bids, asks, 5, 0.05)
    # closer to L1 (0.80) than the rho=0.3 value was
    assert w3 > w > w2, f"steeper decay must be closer to L1: w3={w3} w={w} w2={w2}"
    assert abs(w3 - 0.8) < abs(w - 0.8), "steep decay closer to L1 than rho=0.3"
    print(_ts() + f"[self-test] wobi steep rho -> {w3:+.3f} (closer to L1 +0.80 than flatter decays)  OK")
    # gap function sanity
    df = pd.DataFrame({"s": np.r_[np.ones(300), -np.ones(300)],
                       "f": np.r_[np.ones(300)*2, -np.ones(300)*2], "l1": 0.0})
    g = _gap(df, "s", "f")
    assert abs(g - 2.0) < 1e-9, g  # sign(s)*f = +2 everywhere
    print(_ts() + f"[self-test] gap fn: perfect predictor -> {g:+.1f}  OK")
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
    if a.smoke: smoke(symbols=a.symbols)
    elif a.self_test or not a.run: self_test()
    if a.run: run_real(symbols=a.symbols, workers=a.workers, max_days=(a.days or None))
