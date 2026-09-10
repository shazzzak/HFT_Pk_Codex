# ============================================================================
# flow_composition.py -- FOUR-WAY ORDER-FLOW COMPOSITION feature + gate.
# ----------------------------------------------------------------------------
# Classifies every incoming order into one of four categories from the feed:
#   market_buy  = a TRADE with aggressor_side BUY   (marketable order lifting ask)
#   market_sell = a TRADE with aggressor_side SELL  (hitting the bid)
#   limit_buy   = an ORDER_ADD on the BID side       (new resting bid = supply)
#   limit_sell  = an ORDER_ADD on the ASK side       (new resting ask = supply)
# Market orders are COLLAPSED (same exchange ts + side = ONE order) for COUNTING,
# with volume summed -- so a sweep's fragmentation does not inflate market counts.
#
# Over 15 trailing windows -- 5 EVENT (10/20/30/40/50), 5 TIME (1/2/3/4/5s), and
# 5 MIN-of-the-two (min(10,1s)..min(50,5s), whichever holds FEWER events) -- it
# computes, in BOTH count and volume weighting:
#   * the 4 raw category fractions  (emitted as features for later modelling)
#   * aggression imbalance = mbuy% - msell%   (directional: +ve = net buying)
#   * supply     imbalance = lbuy% - lsell%   (directional: +ve = bid support added)
#   * market share = (mbuy% + msell%)         (how aggressive the flow is)
#
# GATE (does it predict, on top of OBI): for the two DIRECTIONAL imbalances, the
# momentum score = mean( sign(signal) * forward_mid_move ) at 1/3/5s -- +ve means
# price continues with the imbalance. Reported per window, day-as-unit, and again
# among OBI-CALM rows (|obi_1|<0.30) to see if it adds over the OBI edge.
#
# USAGE:  python flow_composition.py --self-test | --smoke | --run
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
# bisect for the time-window suffix start index
import bisect
# numerics
import numpy as np
# dataframes
import pandas as pd

# central config: raw store + results roots (single source of truth)
from config_pk import PARSED_ROOT, RESULTS_ROOT


# bracketed HH:MM:SS stamp for log lines
def _ts():
    # current local time
    return datetime.now().strftime("[%H:%M:%S]")


# where diagnostic outputs go
OUT_DIR = RESULTS_ROOT / "diagnostics"
# results root (for the watchlist name list)
WATCHLIST = RESULTS_ROOT / "mm_watchlist_final.csv"
# forward horizons in seconds
HORIZONS_S = [1, 3, 5]
# event-count windows
EVENT_WINS = [10, 20, 30, 40, 50]
# time windows in seconds
TIME_WINS = [1, 2, 3, 4, 5]
# category indices (fixed order everywhere)
MBUY, MSELL, LBUY, LSELL = 0, 1, 2, 3
# OBI "calm" threshold (the incumbent OBI trigger level)
OBI_CALM = 0.30
# minimum orders in the trailing window before a signal is trusted
MIN_WIN = 5
# minimum observations in a (name-day) cell to trust that day's mean
MIN_ROWS = 30
# cap on sampled feature rows saved per name-day (keeps the feature CSV small)
FEATURE_SAMPLE_PER_DAY = 300
# default sampled days
MAX_DAYS = 20
# default workers
WORKERS = 6


# tally counts + volumes per category over a slice of the trailing buffer.
# cats/qtys/isnew are numpy arrays for the last n events; returns (counts4, vols4).
def _tally(cats, qtys, isnew):
    # count each category using ONLY collapse-new orders (isnew mask)
    counts = np.bincount(cats[isnew], minlength=4).astype(float)
    # sum volume per category using ALL prints (fragmented sweep volume all counts)
    vols = np.bincount(cats, weights=qtys, minlength=4).astype(float)
    # hand back both 4-vectors
    return counts, vols


# from a category 4-vector, derive (mbuy%, msell%, lbuy%, lsell%, aggr, supply, mkt_share)
def _signals(vec):
    # total across the four categories
    tot = vec.sum()
    # degenerate window -> all NaN
    if tot <= 0:
        # nothing to normalize
        return (np.nan,) * 7
    # the four raw fractions
    mb, ms, lb, ls = vec / tot
    # aggression imbalance (net marketable direction)
    aggr = mb - ms
    # supply imbalance (net resting-liquidity direction)
    supply = lb - ls
    # how aggressive the flow is (market share of all orders)
    share = mb + ms
    # bundle
    return mb, ms, lb, ls, aggr, supply, share


# one observing pass over the merged S/U/T stream; returns a per-observation frame
def scan(events, Book):
    # the engine's reconstructed book (for obi + mid)
    book = Book()
    # trailing buffer of recent orders: parallel lists ts / cat / qty / isnew
    b_ts = []
    # category index per buffered event
    b_cat = []
    # volume per buffered event
    b_qty = []
    # collapse-new flag per buffered event (market collapse; limits always new)
    b_new = []
    # collapse memory for market orders: last trade ts and side
    last_tr_ts = None
    # last trade side (+1 buy / -1 sell)
    last_tr_side = 0
    # mid timeline for forward lookups: parallel (ts, mid)
    mid_t = []
    # mid values
    mid_v = []
    # per-observation records
    recs = []

    # read current mid + L1 obi from the book (None if one-sided)
    def _mid_obi():
        # best bid/ask + sizes
        bb, bq, ba, aq = book.bbo()
        # need both sides
        if bb is None or ba is None:
            # no mid
            return None, None
        # midpoint
        m = 0.5 * (bb + ba)
        # L1 imbalance
        denom = (bq or 0) + (aq or 0)
        # imbalance or 0
        ob = ((bq - aq) / denom) if denom > 0 else 0.0
        # return both
        return m, ob

    # append one classified order to the trailing buffer
    def _push(ts, cat, qty, isnew):
        # record time
        b_ts.append(ts)
        # record category
        b_cat.append(cat)
        # record volume
        b_qty.append(float(qty))
        # record collapse-new flag
        b_new.append(bool(isnew))
        # evict from the left while we still have >=50 events AND the oldest is >5s old
        while len(b_ts) > 50 and b_ts[0] < ts - 5000:
            # drop the stale oldest
            b_ts.pop(0); b_cat.pop(0); b_qty.pop(0); b_new.pop(0)

    # walk the time-ordered merged stream
    for (ts, rank, seq, kind, obj) in events:
        # a trade print = a market order
        if kind == "T":
            # aggressor side +1 buy / -1 sell
            sd = 1 if str(getattr(obj, "aggressor_side", "")).upper().startswith("B") else -1
            # category: market buy or market sell
            cat = MBUY if sd == 1 else MSELL
            # collapse: NOT a new order if same ts AND same side as the previous trade
            isnew = not (last_tr_ts is not None and ts == last_tr_ts and sd == last_tr_side)
            # update the collapse memory
            last_tr_ts = ts; last_tr_side = sd
            # push the market order into the trailing buffer
            _push(ts, cat, float(getattr(obj, "qty", 0.0) or 0.0), isnew)
            # ---- OBSERVATION at each trade (a market order just arrived) ----
            # read the mid + obi now
            m, ob = _mid_obi()
            # only observe when the book is two-sided
            if m is not None and len(b_ts) >= MIN_WIN:
                # snapshot the trailing buffer as arrays (small: <=~ last 5s/50)
                A_ts = np.asarray(b_ts); A_cat = np.asarray(b_cat)
                # volumes + new flags
                A_qty = np.asarray(b_qty); A_new = np.asarray(b_new, dtype=bool)
                # number buffered
                L = A_ts.size
                # per-window signal container for this observation
                row = {"ts": ts, "obi": ob, "mid0": m}
                # count of events within each time window (suffix via bisect)
                # (A_ts is ascending, so items >= ts - T*1000 form a suffix)
                for wi in range(5):
                    # ---- EVENT window k: last E events ----
                    nE = min(EVENT_WINS[wi], L)
                    # slice indices for the event window
                    cE, vE = _tally(A_cat[L-nE:], A_qty[L-nE:], A_new[L-nE:])
                    # ---- TIME window k: events in the last T seconds ----
                    startT = bisect.bisect_left(A_ts, ts - TIME_WINS[wi] * 1000)
                    # number in the time window
                    nT = L - startT
                    # tally the time window
                    cT, vT = _tally(A_cat[startT:], A_qty[startT:], A_new[startT:])
                    # ---- MIN-of-two window k: whichever holds FEWER events ----
                    nM = min(nE, nT)
                    # tally the min window (last nM events)
                    cM, vM = _tally(A_cat[L-nM:], A_qty[L-nM:], A_new[L-nM:])
                    # derive signals for each family (count + volume) and store the
                    # directional imbalances (the gate) + raw fractions (features)
                    for fam, cc, vv, nn in (("E", cE, vE, nE), ("T", cT, vT, nT), ("M", cM, vM, nM)):
                        # skip too-thin windows
                        if nn < MIN_WIN:
                            # mark NaN so the gate ignores this cell
                            row[f"aggr_c_{fam}{wi}"] = np.nan; row[f"aggr_v_{fam}{wi}"] = np.nan
                            row[f"supply_c_{fam}{wi}"] = np.nan; row[f"supply_v_{fam}{wi}"] = np.nan
                            # raw fractions NaN too
                            for nm in ("mbuy","msell","lbuy","lsell","share"):
                                row[f"{nm}_c_{fam}{wi}"] = np.nan; row[f"{nm}_v_{fam}{wi}"] = np.nan
                            # next family
                            continue
                        # count-based signals
                        mb,ms,lb,ls,ag,su,sh = _signals(cc)
                        # volume-based signals
                        mbv,msv,lbv,lsv,agv,suv,shv = _signals(vv)
                        # store the two directional imbalances (gate targets)
                        row[f"aggr_c_{fam}{wi}"]=ag; row[f"aggr_v_{fam}{wi}"]=agv
                        row[f"supply_c_{fam}{wi}"]=su; row[f"supply_v_{fam}{wi}"]=suv
                        # store the raw four fractions + market share (features)
                        row[f"mbuy_c_{fam}{wi}"]=mb; row[f"mbuy_v_{fam}{wi}"]=mbv
                        row[f"msell_c_{fam}{wi}"]=ms; row[f"msell_v_{fam}{wi}"]=msv
                        row[f"lbuy_c_{fam}{wi}"]=lb; row[f"lbuy_v_{fam}{wi}"]=lbv
                        row[f"lsell_c_{fam}{wi}"]=ls; row[f"lsell_v_{fam}{wi}"]=lsv
                        row[f"share_c_{fam}{wi}"]=sh; row[f"share_v_{fam}{wi}"]=shv
                # keep the observation
                recs.append(row)
        # a book update
        elif kind == "U":
            # only ORDER_ADD counts as a new limit order (supply)
            if getattr(obj, "event", None) == "ORDER_ADD":
                # side of the resting order
                sd = str(getattr(obj, "side", "")).upper()
                # limit buy (bid) or limit sell (ask)
                cat = LBUY if sd.startswith("B") else LSELL
                # each add is a distinct limit order (isnew=True)
                _push(ts, cat, float(getattr(obj, "qty", 0.0) or 0.0), True)
                # apply to the book
                book.add(obj)
            else:
                # a cancel: apply to the book (does not add supply to the tally)
                book.cancel(obj)
        # a snapshot event -> nothing to classify
        elif kind == "S":
            # (book reconciliation happens inside the engine's own path if used)
            pass
        # after every event, record the mid for forward lookups
        m, _ = _mid_obi()
        # dense mid timeline
        if m is not None:
            # timestamp
            mid_t.append(ts)
            # value
            mid_v.append(m)

    # nothing to analyze
    if not recs or not mid_t:
        # empty
        return None
    # observations to frame
    df = pd.DataFrame(recs)
    # mid timeline arrays
    mt = np.asarray(mid_t, float); mv = np.asarray(mid_v, float)
    # session bounds for the bucket split
    t0, t1 = mt.min(), mt.max()
    # window edges (ms)
    F = 15*60*1000; P = 45*60*1000; L = 15*60*1000
    # default bucket middle
    bk = np.full(len(df), "middle", dtype=object)
    # first 15
    bk[df["ts"].to_numpy() <= t0 + F] = "first15"
    # preclose 45->15
    bk[df["ts"].to_numpy() >= t1 - P] = "preclose45"
    # last 15
    bk[df["ts"].to_numpy() >= t1 - L] = "last15"
    # attach
    df["bucket"] = bk
    # forward UNSIGNED mid move (bps) at each horizon (gate signs it per signal)
    for h in HORIZONS_S:
        # first mid at/after ts+h
        i1 = np.searchsorted(mt, df["ts"].to_numpy() + h*1000.0, side="left")
        # valid within session
        ok = i1 < mt.size
        # forward mid
        m1 = np.where(ok, mv[np.clip(i1,0,mt.size-1)], np.nan)
        # unsigned forward move in bps
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"fwd_{h}"] = (m1 - df["mid0"].to_numpy())/df["mid0"].to_numpy()*1e4
    # drop the raw mid col
    return df.drop(columns=["mid0"])


# per-process globals
_R=None; _BOOK=None; _NAMES=None
# pool init
def _init(names):
    # expose globals
    global _R,_BOOK,_NAMES
    # driver
    import run_legacy_mm as R
    # local store (config-driven, but be explicit for the worker)
    R.PARSED_ROOT = PARSED_ROOT
    # engine
    import mm_backtest as MB
    # stash
    _R=R; _BOOK=MB.Book
    # universe: watchlist else all symbols
    if names is None:
        # try the watchlist
        try:
            # read the 38-name shortlist
            names=sorted(pd.read_csv(WATCHLIST)["symbol"].dropna().astype(str).unique().tolist())
        except Exception:
            # fall back to all traded symbols
            names=None
    # stash
    _NAMES=names


# one symbol-day -> per-observation frame
def _one(date, sym):
    # datasets
    dsets=_R.open_datasets(date)
    # updates / snapshots / trades
    u=_R.read_symbol(dsets["ob_updates"], _R.REQ_UPDATES, sym)
    # snapshots
    s=_R.read_symbol(dsets["ob_snapshot"], _R.REQ_SNAP, sym)
    # trades
    t=_R.read_symbol(dsets["trades"], _R.REQ_TRADES, sym)
    # unrunnable
    if len(t)==0 or len(s)==0:
        # skip
        return None
    # merged stream
    events, snap_groups, t = _R.build_events(u, s, t)
    # observe
    return scan(events, _BOOK)


# gate: momentum score for a directional signal, day-as-unit, all + OBI-calm
def _gate_rows(df, sym, date):
    # collect long-format gate rows
    out=[]
    # the two directional signals x two weightings
    sigs=["aggr_c","aggr_v","supply_c","supply_v"]
    # each window family x level
    for fam in ("E","T","M"):
        # each of the 5 levels
        for wi in range(5):
            # each directional signal
            for base in sigs:
                # the signal column name
                col=f"{base}_{fam}{wi}"
                # skip if absent
                if col not in df.columns:
                    # next
                    continue
                # the signal values
                sig=df[col].to_numpy()
                # each horizon
                for h in HORIZONS_S:
                    # forward move
                    fwd=df[f"fwd_{h}"].to_numpy()
                    # momentum score = sign(signal)*forward move (all rows)
                    mom=np.sign(sig)*fwd
                    # valid mask
                    ok=~(np.isnan(sig)|np.isnan(fwd)|(sig==0))
                    # all-rows mean (if enough)
                    m_all=float(np.mean(mom[ok])) if ok.sum()>=MIN_ROWS else np.nan
                    # OBI-calm subset
                    calm=ok & (np.abs(df["obi"].to_numpy())<OBI_CALM)
                    # OBI-calm mean
                    m_calm=float(np.mean(mom[calm])) if calm.sum()>=MIN_ROWS else np.nan
                    # record one gate row
                    out.append(dict(symbol=sym,date=date,family=fam,level=wi,
                                    signal=base,horizon=h,mom_all=m_all,mom_calm=m_calm,
                                    n_all=int(ok.sum())))
    # the day's gate rows
    return out


# all symbols for one date
def _work_date(date):
    # datasets
    dsets=_R.open_datasets(date)
    # missing
    if dsets is None:
        # nothing
        return [], []
    # universe
    names=_NAMES or _R.list_symbols(dsets["trades"])
    # gate rows + sampled feature rows
    gate=[]; feats=[]
    # loop symbols
    for sym in names:
        # guard
        try:
            # run one symbol-day
            df=_one(date, sym)
        except Exception as e:
            # report + continue
            print(_ts()+f"SKIP {date} {sym}: {e!r}")
            # next
            continue
        # skip empty
        if df is None or len(df)==0:
            # nothing
            continue
        # gate rows for this symbol-day
        gate.extend(_gate_rows(df, sym, str(date)))
        # sampled feature rows (raw fractions etc.) for later modelling
        take=min(FEATURE_SAMPLE_PER_DAY, len(df))
        # evenly-spaced sample
        idx=np.linspace(0, len(df)-1, take).astype(int)
        # tag + collect the sample
        fs=df.iloc[idx].copy(); fs["symbol"]=sym; fs["date"]=str(date)
        # collect
        feats.append(fs)
    # this date's outputs
    return gate, feats


# day-as-unit summary of the gate (mean +/- SE across name-days)
def _summarize(gate_df):
    # helper: mean/SE/n across name-days
    def t(x):
        # drop NaN
        x=np.asarray(x,float); x=x[~np.isnan(x)]; n=x.size
        # need >=2
        if n<2: return np.nan,np.nan,n
        # mean + SE
        return float(x.mean()), float(x.std(ddof=1)/np.sqrt(n)), n
    # header
    print(_ts()+"===== FLOW-COMPOSITION GATE: momentum score by signal x window (day-as-unit) =====")
    print(_ts()+"  +ve = price continues with the imbalance (predictive). 'calm' = among |obi|<%.2f rows." % OBI_CALM)
    # family label map
    famlab={"E":"events","T":"secs","M":"min(ev,sec)"}
    # for the strongest read, show the 5s horizon (report others in the CSV)
    h=5
    # each signal
    for base in ["aggr_v","aggr_c","supply_v","supply_c"]:
        # header per signal
        print(_ts()+f"  --- {base} @ {h}s ---   (all | OBI-calm), by window level 0..4")
        # each family
        for fam in ("E","T","M"):
            # build the row across levels
            cells=[]
            # each level
            for wi in range(5):
                # this cell's name-day values
                sub=gate_df[(gate_df.signal==base)&(gate_df.family==fam)&
                            (gate_df.level==wi)&(gate_df.horizon==h)]
                # all + calm means
                ma,_,na=t(sub["mom_all"].to_numpy()); mc,_,_=t(sub["mom_calm"].to_numpy())
                # window size label
                wl=EVENT_WINS[wi] if fam=="E" else (f"{TIME_WINS[wi]}s" if fam=="T" else f"{EVENT_WINS[wi]}/{TIME_WINS[wi]}s")
                # format cell
                cells.append(f"{wl}:{ma:+.2f}/{mc:+.2f}")
            # print the family row
            print(_ts()+f"      {famlab[fam]:>11}: " + "  ".join(cells))
    # reading guide
    print(_ts()+"  READ: |mom| large & consistent, and still nonzero in the CALM column -> predictive on")
    print(_ts()+"        top of OBI (worth a P&L test). ~0, or vanishes when calm -> redundant with OBI.")


# full run
def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    # driver
    import run_legacy_mm as R
    # local store
    R.PARSED_ROOT = PARSED_ROOT
    # dates
    dates=R.discover_dates()
    # guard
    if not dates:
        # stop
        print(_ts()+"discover_dates() empty -- check config_pk PARSED_ROOT."); return
    # sample days
    if max_days and len(dates)>max_days:
        # stride
        step=max(1,len(dates)//max_days); dates=dates[::step][:max_days]
    # announce
    print(_ts()+f"{len(dates)} dates, {workers} workers -- 4-way flow composition (15 windows)")
    # accumulate
    gate=[]; feats=[]
    # timer
    t0=time.perf_counter()
    # pool
    with Pool(processes=workers, initializer=_init, initargs=(symbols,)) as pool:
        # counter
        done=0
        # consume
        for g,f in pool.imap_unordered(_work_date, dates):
            # collect gate + features
            gate.extend(g); feats.extend(f)
            # bump
            done+=1
            # elapsed
            el=(time.perf_counter()-t0)/60.0
            # progress + ETA
            print(_ts()+f"  date {done}/{len(dates)} ({len(gate)} gate rows, {el:.1f} min, ETA {el/done*(len(dates)-done):.1f} min)")
    # nothing
    if not gate:
        # stop
        print(_ts()+"no rows."); return
    # gate frame
    gdf=pd.DataFrame(gate)
    # ensure dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # save the full gate table (all signals x windows x horizons, per name-day)
    gdf.to_csv(out_dir/"flow_composition_gate.csv", index=False)
    # save the sampled feature table (raw 4 fractions etc.) for later modelling
    if feats:
        # concat + write
        pd.concat(feats, ignore_index=True).to_csv(out_dir/"flow_composition_features.csv", index=False)
    # print the day-as-unit summary
    _summarize(gdf)
    # location
    print(_ts()+f"[flow-composition] outputs -> {out_dir}")


# one stock-day, timed
def smoke(symbols=None):
    # driver + engine
    import run_legacy_mm as R, mm_backtest as MB
    # local store
    R.PARSED_ROOT = PARSED_ROOT
    # dates
    dates=R.discover_dates()
    # report
    print(_ts()+f"discover_dates -> {len(dates)} dates")
    # stop if none
    if not dates: return
    # mid date
    date=dates[len(dates)//2]
    # datasets
    dsets=R.open_datasets(date)
    # symbol
    sym=symbols[0] if symbols else R.list_symbols(dsets["trades"])[0]
    # globals _one needs
    global _R,_BOOK; _R=R; _BOOK=MB.Book
    # time one
    t0=time.perf_counter(); df=_one(date, sym); dt=time.perf_counter()-t0
    # report
    print(_ts()+f"[smoke] {date} {sym} in {dt:.1f}s: {0 if df is None else len(df)} observations, "
          f"{0 if df is None else df.shape[1]} feature cols")


# validate tally + collapse + imbalance + momentum math
def self_test():
    # ---- tally + collapse ----
    # events (arrays): cats and qtys and isnew
    # sequence: mbuy(50,new), mbuy(50,SAME order->not new), lsell(add,new), msell(30,new)
    cats=np.array([MBUY,MBUY,LSELL,MSELL])
    # volumes
    qtys=np.array([50.,50.,20.,30.])
    # collapse flags: 2nd mbuy is a fragment of the 1st (not new)
    isnew=np.array([True,False,True,True])
    # tally the whole slice
    c,v=_tally(cats,qtys,isnew)
    # counts: mbuy=1 (collapsed), msell=1, lbuy=0, lsell=1
    assert list(c)==[1,1,0,1], f"count tally wrong: {c}"
    # volumes: mbuy=100 (both prints), msell=30, lsell=20
    assert list(v)==[100.,30.,0.,20.], f"vol tally wrong: {v}"
    print(_ts()+f"[self-test] tally+collapse: counts={list(c)} vols={list(v)}  OK")
    # ---- signals ----
    # from counts [1,1,0,1]: tot=3; mbuy%=1/3, msell%=1/3, lsell%=1/3
    mb,ms,lb,ls,ag,su,sh=_signals(c)
    # aggression imbalance = mbuy%-msell% = 0
    assert abs(ag-0.0)<1e-9, f"aggr wrong: {ag}"
    # supply imbalance = lbuy%-lsell% = 0 - 1/3
    assert abs(su-(-1/3))<1e-9, f"supply wrong: {su}"
    # market share = mbuy%+msell% = 2/3
    assert abs(sh-(2/3))<1e-9, f"share wrong: {sh}"
    print(_ts()+f"[self-test] signals: aggr={ag:+.3f} supply={su:+.3f} share={sh:.3f}  OK")
    # from volumes [100,30,0,20]: tot=150; aggr_v=(100-30)/150
    mbv,msv,lbv,lsv,agv,suv,shv=_signals(v)
    # check volume aggression
    assert abs(agv-(70/150))<1e-9, f"aggr_v wrong: {agv}"
    print(_ts()+f"[self-test] vol signals: aggr_v={agv:+.3f}  OK")
    # ---- momentum sign convention ----
    # signal +0.5 (net buying), forward move +2 bps -> momentum = +1 (continues)
    assert np.sign(0.5)*2.0 == 2.0
    # signal -0.5 (net selling), forward move -2 bps -> momentum = +1 (continues down = with signal)
    assert np.sign(-0.5)*(-2.0) == 2.0
    print(_ts()+"[self-test] momentum sign: continuation -> +ve both directions  OK")
    # all good
    print(_ts()+"[self-test] ALL ASSERTIONS PASSED.")


# entry point
if __name__=="__main__":
    # parser
    ap=argparse.ArgumentParser()
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
    a=ap.parse_args()
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
