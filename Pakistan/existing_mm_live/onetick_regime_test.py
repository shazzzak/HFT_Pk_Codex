# ============================================================================
# onetick_regime_test.py -- CAN WE MARKET-MAKE ON LOCKED 1-TICK BOOKS?
# ----------------------------------------------------------------------------
# Runs OBI (control) and the ONE-TICK MM strategy on a small subset of cheap /
# locked-book names, then splits every fill by the engine's `regime` tag
# ("onetick" = filled while the spread was 1 tick | "normal" = 2t+ spread) and
# reports FIFO-attributed P&L (capture / markout / fee / net) SEPARATELY per
# regime. That separation is the whole point: it answers "is the 1-tick regime
# itself profitable" independent of the normal regime.
#
# Strategy recap (SZ): on a 1-tick book, when FLAT, quote ONLY the OBI-favorable
# side when |obi|>=thresh (sit out otherwise); once filled, revert to normal
# touch-following exit. Sweeps the OBI entry threshold.
#
# USAGE: python onetick_regime_test.py --self-test | --smoke | --run
# ============================================================================

# CLI
import argparse
# timing
import time
# timestamps
from datetime import datetime
# parallel
from multiprocessing import Pool
# numerics / frames
import numpy as np
import pandas as pd

# central paths
from config_pk import PARSED_ROOT, RESULTS_ROOT


# stamp for log lines
def _ts():
    # current time
    return datetime.now().strftime("[%H:%M:%S]")


# output dir
OUT_DIR = RESULTS_ROOT / "diagnostics"
# 1-tick-heavy test universe (cheap/locked-book names)
SUBSET = ["KEL", "PIBTL", "TPL", "BOP", "PACE", "FNEL", "HASCOL", "PIAHCLA"]
# OBI entry thresholds to sweep for the one-tick regime
THRESHOLDS = [0.15, 0.25, 0.40]
# round-trip fee in bps (TREC), for the net line
FEE_BPS = 1.554
# trailing days for the median trade size (clip sizing)
TRAIL_DAYS = 10
# clip multiplier
CLIP_MULT = 3.0
# sampled days
MAX_DAYS = 20
# workers
WORKERS = 6


# split fills by regime and FIFO-attribute P&L within each regime.
# Returns {regime: {capture, markout, fee, net, fills, pnl}} using the harness FIFO.
def attribute_by_regime(dr, H):
    """Per-regime P&L via the VALIDATED harness FIFO attribution -- reconciled to
    the engine by construction (residual = engine_pnl - matched_realized). We do
    NOT reinvent FIFO/fees/residual here; we call mm_harness.fifo_attribution and
    simply use the fill's REGIME as the attribution key, by (a) copying regime
    into the 'bucket' field the harness reads, and (b) temporarily widening its
    BUCKETS set to include the regime labels so they accumulate. Round trips book
    to the OPENING regime and the residual reconciles -- both for free from the
    proven function. Returns {regime: {net, net_bps, n_fills}} + a _recon record.
    """
    f = dr.fills
    if f is None or len(f) == 0 or "regime" not in f.columns:
        return {}
    # EXCLUDE the engine's own EOD liquidation fills (reason 'liq' / 'liq_residual').
    # THE ROOT CAUSE of the 1-2 PKR non-reconciliation: fifo_attribution is DESIGNED
    # for the liquidation to be ABSENT from the fill stream -- it recovers the
    # residual as engine_pnl - matched and attributes it to the still-OPEN lots.
    # The engine now also emits liq fills into dr.fills; feeding those in makes FIFO
    # close every lot (open_lots empty at the end), so residual_total has no lot to
    # land on and is silently dropped. It is non-zero because 'liq_residual' is a
    # HAIRCUT MARK, not an execution: the engine adds it to equity with NO fee, but
    # fifo_attribution charges fee_for on it as if it were a fill. That fee gap is
    # exactly the observed -1.15/-1.96 PKR. Dropping liq rows restores the function's
    # contract: intraday fills leave the true residual OPEN, the plug captures the
    # liquidation (haircut + correct fees) and books it to the OPENING regime. Exact.
    if "reason" in f.columns:
        ff = f[~f["reason"].astype(str).str.startswith("liq")].copy()
    else:
        ff = f.copy()
    # nothing intraday -> nothing to attribute (all P&L is liquidation-only)
    if len(ff) == 0:
        return {}
    # remember the real session bucket in case anything else needs it (unused here)
    ff["bucket"] = ff["regime"]
    # the engine's ground-truth total P&L for this symbol-day
    engine_pnl = dr.pnl()
    # the liquidation time (residual lots are held to here) -- session end
    liq_t = float(ff["t"].max()) if len(ff) else 0.0
    # TEMPORARILY widen the harness BUCKETS to the regime labels so per[] accumulates
    regimes = sorted(ff["regime"].dropna().unique().tolist())
    saved = H.BUCKETS
    try:
        # the harness builds per = {b: {...} for b in BUCKETS}; make those the regimes
        H.BUCKETS = tuple(regimes)
        # call the VALIDATED attribution (matched->opening bucket, residual reconciled)
        per = H.fifo_attribution(ff, engine_pnl, liq_t)
    finally:
        # always restore, even on error
        H.BUCKETS = saved
    # assemble the per-regime view + net bps on opened notional
    out = {}
    total_realized = 0.0
    for reg in regimes:
        d = per.get(reg, {})
        rz = float(d.get("realized", 0.0))
        onot = float(d.get("opened_notional", 0.0))
        total_realized += rz
        out[reg] = dict(net=rz,
                        net_bps=(rz / onot * 1e4) if onot > 0 else float("nan"),
                        n_fills=int(d.get("fills", 0)),
                        matched_notional=onot)
    # reconciliation: harness guarantees sum(realized) == engine_pnl by construction
    err = total_realized - engine_pnl
    # tolerance: STRICT. The attribution is exact by construction once the liq
    # fills are excluded, so only float64 rounding remains (~1e-12 relative). Any
    # error above 0.01 PKR (or 1e-9 relative) is a REAL bug and must surface.
    out["_recon"] = dict(engine_total=engine_pnl, our_realized=total_realized,
                         recon_err=err,
                         recon_ok=abs(err) <= max(0.01, 1e-9 * abs(engine_pnl)))
    return out


# per-process globals
_R = None; _H = None; _CALIB = None
# pool init
def _init(calib):
    # globals
    global _R, _H, _CALIB
    # driver + harness
    import run_legacy_mm as R, mm_harness as H
    # local store
    R.PARSED_ROOT = PARSED_ROOT
    # micro strategy
    R.USE_MICRO = True
    # stash
    _R = R; _H = H; _CALIB = calib


# run one (config, symbol, day) and return per-regime attribution rows
def _one(thr_label, thr, date, sym, dsets):
    # calibration
    C = _CALIB
    # need scales + profile
    if sym not in C["scales"] or sym not in C["profiles"]:
        # skip
        return []
    # segments
    segs = C["segments"].get(str(date))
    # skip if none
    if segs is None:
        return []
    # trailing median trade size -> clip
    med = _H.trailing_median(C["tstats"][sym], C["all_dates"], date, TRAIL_DAYS)
    # unusable
    if med is None or med <= 0:
        return []
    # clip
    clip = max(1, int(round(CLIP_MULT * med)))
    # base production params
    params = _H.build_micro_params(clip, C["scales"][sym], C["profiles"][sym],
                                   C["windows"].get(sym, (5.0, 1.0)), segs)
    # OBI throttle on (production)
    params.update(dict(obi_throttle=True, ofi_throttle=False, obi_throttle_thresh=0.15,
                       throttle_frac=0.5, throttle_hold_ms=300.0))
    # one-tick MM on for the treatment configs (thr is None for control)
    if thr is not None:
        params.update(dict(enable_onetick_mm=True, onetick_obi_thresh=thr))
    # run the day
    dr = _H.run_symbol_day(date, sym, dsets, params)
    # nothing
    # skip unpriceable days: the run failed (dr None) OR there was no clean close
    # (dr.pnl() is None -- eod None, or equity_liquidated None on a broken close).
    # Without the pnl() half, engine_pnl=None reaches fifo_attribution and raises
    # TypeError at float(engine_pnl) (mm_harness.py:380); _work_date's try/except
    # then swallows it as a misleading "SKIP ... TypeError" and drops the day.
    if dr is None or dr.pnl() is None:
        return []
    # attribute per regime
    attr = attribute_by_regime(dr, _H)
    # pull + REMOVE the reconciliation record so it is not iterated as a regime
    recon = attr.pop("_recon", {})
    # rows (real regimes only now)
    rows = []
    # each regime
    for regime, m in attr.items():
        # record, carrying the reconciliation flag/err for the trust gate
        rows.append(dict(config=thr_label, symbol=sym, date=str(date), regime=regime,
                         net=m["net"], net_bps=m["net_bps"], n_fills=m["n_fills"],
                         matched_notional=m["matched_notional"],
                         recon_ok=recon.get("recon_ok", True),
                         recon_err=recon.get("recon_err", 0.0)))
    # this cell's rows
    return rows


# work one date across all configs+symbols
def _work_date(date):
    # datasets
    dsets = _R.open_datasets(date)
    # missing
    if dsets is None:
        return []
    # accumulate
    rows = []
    # configs: control (None) + each threshold
    configs = [("OBI", None)] + [(f"OT_t{t:g}", t) for t in THRESHOLDS]
    # each config
    for lab, thr in configs:
        # each symbol
        for sym in SUBSET:
            # guard
            try:
                rows.extend(_one(lab, thr, date, sym, dsets))
            except Exception as e:
                print(_ts() + f"SKIP {lab} {date} {sym}: {e!r}")
    # date rows
    return rows


# build calibration bundle
def _calib():
    import run_legacy_mm as R, mm_harness as H
    R.PARSED_ROOT = PARSED_ROOT
    all_dates = R.discover_dates()
    return dict(scales=H.load_scales(), profiles=H.load_profiles(),
                windows=H.load_windows(), segments=H.load_segments(),
                all_dates=all_dates,
                tstats=H.trailing_median_trade_size(all_dates, SUBSET, TRAIL_DAYS)), all_dates


# full run
def run_real(out_dir=OUT_DIR, workers=WORKERS, max_days=MAX_DAYS):
    # calibration
    print(_ts() + "pre-pass: calibration")
    calib, all_dates = _calib()
    # sample days
    dates = all_dates[TRAIL_DAYS:]
    if max_days and len(dates) > max_days:
        step = max(1, len(dates) // max_days); dates = dates[::step][:max_days]
    # announce
    print(_ts() + f"{len(SUBSET)} names x {len(dates)} dates x 4 configs, {workers} workers")
    # collect
    rows = []; t0 = time.perf_counter()
    with Pool(processes=workers, initializer=_init, initargs=(calib,)) as pool:
        done = 0
        for res in pool.imap_unordered(_work_date, dates):
            rows.extend(res); done += 1
            el = (time.perf_counter() - t0) / 60.0
            print(_ts() + f"  date {done}/{len(dates)} ({el:.1f} min, ETA {el/done*(len(dates)-done):.1f} min)")
    # nothing
    if not rows:
        print(_ts() + "no rows."); return
    # frame
    df = pd.DataFrame(rows)
    # ensure dir + save (safe parquet)
    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_parquet(df, out_dir / "onetick_regime_test.parquet")
    # RECONCILIATION GATE: how many (config,name,day) cells tied to engine P&L?
    if "recon_ok" in df.columns:
        cells = df.drop_duplicates(["config", "symbol", "date"])
        ok = int(cells["recon_ok"].sum()); tot = len(cells)
        print(_ts() + f"  reconciliation: {ok}/{tot} symbol-days tied to engine P&L "
              f"({100*ok/max(tot,1):.0f}%) -- non-reconciling cells are suspect.")
    # ---- REPORT: P&L by config x regime, day-as-unit ----
    print(_ts() + "===== ONE-TICK MM: P&L BY REGIME (day-as-unit) =====")
    print(_ts() + "  the question: is the 'onetick' regime (1-tick spread fills) profitable on its own?")
    # THREE numbers, deliberately distinct so nothing looks contradictory:
    #  net_PKR    = total cash summed over name-days (dominated by big names/days)
    #  bps_wtd    = PKR-weighted bps = sum(net)/sum(notional) -- SAME SIGN as net_PKR
    #               by construction (this is the honest "rate the money earned")
    #  bps_eqwt   = equal-weighted mean of per-name-day bps (every name-day counts
    #               the same). Can differ in SIGN from net_PKR when a few big-loss
    #               name-days sink the cash while most small name-days are green --
    #               that divergence is INFORMATION (concentration), not an error.
    print(_ts() + f"  {'config':>8} {'regime':>8} {'net_PKR':>12} {'bps_wtd':>8} {'bps_eqwt':>9} {'fills':>9} {'n-days':>7} {'win%':>5}")
    for cfg in ["OBI", "OT_t0.15", "OT_t0.25", "OT_t0.4"]:
        for regime in ["onetick", "normal"]:
            sub = df[(df.config == cfg) & (df.regime == regime)]
            if len(sub) == 0:
                continue
            # total cash
            tot = sub["net"].sum()
            # PKR-weighted bps (consistent in sign with the cash sum)
            notl = sub["matched_notional"].sum()
            bps_wtd = (tot / notl * 1e4) if notl > 0 else float("nan")
            # equal-weighted mean of per-name-day bps
            nd = sub.groupby(["symbol", "date"]).agg(pkr=("net", "sum"),
                                                     bps=("net_bps", "mean"))
            bps_eqwt = nd["bps"].mean()
            # win rate on name-day CASH (what fraction of name-days made money)
            win = (nd["pkr"] > 0).mean() * 100
            fills = int(sub["n_fills"].sum())
            print(_ts() + f"  {cfg:>8} {regime:>8} {tot:>12,.0f} {bps_wtd:>8.2f} {bps_eqwt:>9.2f} {fills:>9,} {len(nd):>7} {win:>4.0f}%")
    # the headline: onetick-regime P&L under the best one-tick config vs OBI's onetick fills
    print(_ts() + "  READ: if 'onetick' net_bps is clearly >0 under OT_* configs, MM works on 1-tick books.")
    print(_ts() + "        if ~0 or <0, locked books are not passively quotable -- keep excluding them.")
    print(_ts() + f"[onetick-regime] outputs -> {out_dir}")


# safe atomic+verified parquet (same as the sweeps)
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


# time one symbol-day
def smoke():
    import run_legacy_mm as R, mm_harness as H
    R.PARSED_ROOT = PARSED_ROOT
    calib, all_dates = _calib()
    global _R, _H, _CALIB; _R, _H, _CALIB = R, H, calib
    date = all_dates[len(all_dates) // 2]
    dsets = R.open_datasets(date)
    t0 = time.perf_counter()
    rows = _one("OT_t0.15", 0.15, date, SUBSET[0], dsets)
    print(_ts() + f"[smoke] {date} {SUBSET[0]} in {time.perf_counter()-t0:.1f}s: {len(rows)} regime rows")
    for r in rows:
        print(_ts() + f"   regime={r['regime']} net={r['net']:.0f} bps={r['net_bps']:.2f} fills={r['n_fills']}")


# validate the per-regime FIFO attribution on hand-built fills
def self_test():
    # A faithful local stub of mm_harness.fifo_attribution (matching the real source)
    # + a fake harness module, so we can test the WRAPPER + reconciliation offline.
    from collections import deque
    def fee_for(px, qty):
        # flat TREC-ish fee for the test (the real one is fee_for; value irrelevant
        # to the reconciliation because residual = engine_pnl - matched absorbs it)
        return 1.554/1e4 * px * qty
    def stub_fifo(fills, engine_pnl, liq_time, residual_hint=None):
        f = fills.to_dict("records") if hasattr(fills, "to_dict") else fills
        open_lots = deque()
        per = {b: {"realized":0.0,"opened_qty":0.0,"opened_notional":0.0,"holds":[],"fills":0}
               for b in FakeH.BUCKETS}
        matched = 0.0
        for fl in f:
            side=fl["side"]; px=float(fl["px"]); qty=float(fl["qty"]); b=fl.get("bucket","middle"); t=float(fl["t"])
            if b in per: per[b]["fills"]+=1
            if not open_lots or open_lots[0]["side"]==side:
                open_lots.append({"qty":qty,"px":px,"bucket":b,"t":t,"side":side})
                if b in per: per[b]["opened_qty"]+=qty; per[b]["opened_notional"]+=qty*px
                continue
            rem=qty
            while rem>1e-9 and open_lots and open_lots[0]["side"]!=side:
                lot=open_lots[0]; m=min(rem,lot["qty"])
                rz=((px-lot["px"])*m) if lot["side"]=="BUY" else ((lot["px"]-px)*m)
                rz-=fee_for(lot["px"],m); rz-=fee_for(px,m)
                if lot["bucket"] in per: per[lot["bucket"]]["realized"]+=rz
                matched+=rz; lot["qty"]-=m; rem-=m
                if lot["qty"]<=1e-9: open_lots.popleft()
            if rem>1e-9:
                open_lots.append({"qty":rem,"px":px,"bucket":b,"t":t,"side":side})
                if b in per: per[b]["opened_qty"]+=rem; per[b]["opened_notional"]+=rem*px
        residual_total=float(engine_pnl)-matched
        resid_notional=sum(l["qty"]*l["px"] for l in open_lots)
        for lot in open_lots:
            w=(lot["qty"]*lot["px"]/resid_notional) if resid_notional>0 else 0.0
            if lot["bucket"] in per: per[lot["bucket"]]["realized"]+=residual_total*w
        return per
    class FakeH:
        BUCKETS=("first15","middle","preclose45","last15")
        fifo_attribution=staticmethod(stub_fifo)
    class DR:
        def __init__(self,fills,eod): self.fills=fills; self.eod=eod
        def pnl(self): return self.eod["equity_liquidated"]

    # CASE: onetick entry (buy 100@100), partial exit in normal (sell 60@100.02),
    # a normal round trip (buy 50@99.99, sell 50@100.01), leaving 40 residual open.
    fills=pd.DataFrame([
      dict(t=1,side="BUY", px=100.00,qty=100,reason="x",regime="onetick"),
      dict(t=2,side="SELL",px=100.02,qty=60, reason="x",regime="normal"),
      dict(t=3,side="BUY", px=99.99, qty=50, reason="x",regime="normal"),
      dict(t=4,side="SELL",px=100.01,qty=50, reason="x",regime="normal"),
    ])
    # pick ANY engine_pnl -- the harness reconciles to it by construction
    dr=DR(fills, {"equity_liquidated": 123.45, "liq_vwap":100.005})
    out=attribute_by_regime(dr, FakeH)
    rec=out.pop("_recon")
    print(_ts()+f"[self-test] regimes: {list(out.keys())}")
    print(_ts()+f"[self-test] sum realized={rec['our_realized']:.4f} engine={rec['engine_total']:.2f} err={rec['recon_err']:.2e}")
    # THE KEY GUARANTEE: reconciles to engine_pnl by construction
    assert rec["recon_ok"], f"must reconcile: err={rec['recon_err']}"
    assert abs(rec["our_realized"]-123.45)<1e-6, "sum of regimes must equal engine_pnl"
    print(_ts()+"[self-test] per-regime sum == engine_pnl EXACTLY (reconciled by construction)  OK")
    # onetick opened the 100-lot; its matched(60) + residual share book to onetick
    assert "onetick" in out and "normal" in out
    print(_ts()+f"[self-test] onetick net={out['onetick']['net']:.2f}  normal net={out['normal']['net']:.2f}")
    # no _recon leaks as a regime
    assert "_recon" not in out
    print(_ts()+"[self-test] ALL ASSERTIONS PASSED.")


# entry
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    a = ap.parse_args()
    if a.smoke: smoke()
    elif a.self_test or not a.run: self_test()
    if a.run: run_real(workers=a.workers, max_days=(a.days or None))
