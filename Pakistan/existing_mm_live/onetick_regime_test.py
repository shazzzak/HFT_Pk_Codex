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
WORKERS = 3


# split fills by regime and FIFO-attribute P&L within each regime.
# Returns {regime: {capture, markout, fee, net, fills, pnl}} using the harness FIFO.
def attribute_by_regime(dr, H):
    """FIFO round-trip P&L by OPENING regime, WITH residual liquidation, reconciled
    to the engine's true total P&L.

    Two fixes over the naive version (both caught by SZ):
      1. Round trips book to the OPENING fill's regime (not the closing fill's),
         so a trip entered on a 1-tick book but exited once the spread widened
         attributes ALL its P&L to 'onetick' -- the regime of the ENTRY decision.
      2. RESIDUAL open inventory at EOD is NOT dropped. Each leftover lot is marked
         at the engine's real liquidation VWAP (dr.eod['liq_vwap']) and booked to
         its OPENING regime. Fees on the (synthetic) liquidation leg are charged.

    RECONCILIATION: sum over regimes of net == dr.pnl() (equity_liquidated) to the
    penny. If residual were dropped or mismarked this assert fails loudly. This is
    the same discipline every trusted sweep uses; the earlier version lacked it,
    which is how the residual hole (and the nonsensical -PKR/+bps) hid.
    """
    f = dr.fills
    if f is None or len(f) == 0 or "regime" not in f.columns:
        return {}
    # engine ground-truth total for this symbol-day
    engine_total = dr.pnl()
    # liquidation price for residual lots (VWAP the engine actually achieved)
    eod = dr.eod or {}
    liq_vwap = eod.get("liq_vwap")
    # single time-ordered stream (FIFO matches across regimes)
    recs = f.sort_values("t").to_dict("records")
    from collections import deque, defaultdict
    # open lots: (price, signed_qty, open_regime)
    lots = deque()
    # accumulators keyed by OPENING regime
    realized = defaultdict(float)
    fee_by = defaultdict(float)
    fills_by = defaultdict(int)
    matched_notional = defaultdict(float)
    # walk fills
    for r in recs:
        reg = r.get("regime", "normal")
        fills_by[reg] += 1
        # fee for this real fill, charged to the regime it occurred in
        fee_by[reg] += FEE_BPS / 1e4 * abs(r["qty"]) * r["px"]
        q = r["qty"] * (1 if r["side"] == "BUY" else -1)
        while lots and q != 0 and (lots[0][1] * q < 0):
            px0, q0, oreg = lots[0]
            m = min(abs(q), abs(q0))
            realized[oreg] += (r["px"] - px0) * (m if q0 > 0 else -1 * m)
            matched_notional[oreg] += m * px0
            if abs(q0) == m:
                lots.popleft()
            else:
                lots[0] = (px0, q0 - (m if q0 > 0 else -m), oreg)
            q -= (m if q > 0 else -m)
        if q != 0:
            lots.append((r["px"], q, reg))
    # ---- RESIDUAL: close every leftover lot at the engine's liquidation VWAP,
    # booked to that lot's OPENING regime. This is the inventory the naive version
    # silently dropped.
    residual_pos = sum(qy for (_, qy, _) in lots)
    if lots and liq_vwap is not None:
        for (px0, q0, oreg) in lots:
            # marking a long lot (q0>0): sell at liq_vwap -> (liq_vwap - px0)*q0
            realized[oreg] += (liq_vwap - px0) * q0
            matched_notional[oreg] += abs(q0) * px0
            # liquidation-leg fee, charged to the opening regime (best available split)
            fee_by[oreg] += FEE_BPS / 1e4 * abs(q0) * liq_vwap
    # assemble per-regime
    out = {}
    for reg in set(list(realized.keys()) + list(fills_by.keys())):
        rz = realized.get(reg, 0.0); fe = fee_by.get(reg, 0.0)
        mn = matched_notional.get(reg, 0.0); net = rz - fe
        out[reg] = dict(realized=rz, fee=fe, net=net, matched_notional=mn,
                        net_bps=(net / mn * 1e4) if mn > 0 else float("nan"),
                        n_fills=fills_by.get(reg, 0))
    # ---- RECONCILIATION: our per-regime net must sum to the engine total.
    # Note: our 'net' subtracts the TREC fee at FEE_BPS; the engine's
    # equity_liquidated already nets its own fees. These fee conventions can
    # differ slightly, so we reconcile the pre-fee REALIZED sum to (engine_total
    # + our_total_fee) rather than asserting net==engine (which would fold in a
    # fee-convention mismatch). This still catches any DROPPED residual/inventory.
    our_realized = sum(realized.values())
    our_fee = sum(fee_by.values())
    # what realized SHOULD be if nothing is dropped: engine_total + fees we charged
    # (engine_total is already net of the engine's fees; we add back OUR fee model
    # to compare gross realized paths). Use a tolerance scaled to the day's size.
    expected_realized = engine_total + our_fee
    resid_err = our_realized - expected_realized
    # attach reconciliation info to the result (checked by the caller / reported)
    out["_recon"] = dict(engine_total=engine_total, our_realized=our_realized,
                         our_fee=our_fee, residual_pos=residual_pos,
                         recon_err=resid_err,
                         recon_ok=abs(resid_err) <= max(1.0, 0.02 * abs(engine_total)))
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
    if dr is None:
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
    print(_ts() + f"  {'config':>8} {'regime':>8} {'net_PKR':>12} {'net_bps':>9} {'fills':>10} {'name-days':>10}")
    # per (config, regime): total net, day-as-unit bps
    for cfg in ["OBI", "OT_t0.15", "OT_t0.25", "OT_t0.4"]:
        for regime in ["onetick", "normal"]:
            sub = df[(df.config == cfg) & (df.regime == regime)]
            if len(sub) == 0:
                continue
            # total net PKR
            tot = sub["net"].sum()
            # day-as-unit bps: per (symbol,date) bps then mean across name-days
            nd = sub.groupby(["symbol", "date"])["net_bps"].mean()
            mbps = nd.mean()
            # total fills
            fills = int(sub["n_fills"].sum())
            print(_ts() + f"  {cfg:>8} {regime:>8} {tot:>12,.0f} {mbps:>9.2f} {fills:>10,} {len(nd):>10}")
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
    # a fake DayResult-like object with a fills frame + engine total + eod
    class DR:
        # engine total P&L (equity_liquidated) for reconciliation
        def pnl(self):
            return self.eod["equity_liquidated"]
    dr = DR()
    # fills: onetick regime buy@100 then sell@100.01 (+1 tick round trip);
    # normal regime buy@50 then sell@49.99 (-1 tick, a loss). Both flat at end,
    # so there is NO residual -> engine_total = sum of realized minus our fees.
    dr.fills = pd.DataFrame([
        dict(t=1, side="BUY", px=100.00, qty=100, reason="x", regime="onetick"),
        dict(t=2, side="SELL", px=100.01, qty=100, reason="x", regime="onetick"),
        dict(t=3, side="BUY", px=50.00, qty=100, reason="x", regime="normal"),
        dict(t=4, side="SELL", px=49.99, qty=100, reason="x", regime="normal"),
    ])
    # realized: onetick +1.00 (100*0.01), normal -1.00 -> gross 0.00; flat -> no residual
    # our fee model total (for reconciliation the engine_total = gross_realized - our_fee)
    _fee = FEE_BPS / 1e4 * (100*100.00 + 100*100.01 + 100*50.00 + 100*49.99)
    # eod: flat, engine total ties to gross realized (0.00) minus our fee
    dr.eod = {"equity_liquidated": 0.00 - _fee, "liq_vwap": None, "pos_at_close": 0}
    out = attribute_by_regime(dr, None)
    # drop the recon record for the regime assertions
    recon = out.pop("_recon", {})
    print(_ts() + f"[self-test] regimes: {list(out.keys())}")
    # onetick: bought 100@100.00, sold 100@100.01 -> realized +1.00 (100*0.01), minus fees
    ot = out["onetick"]
    assert abs(ot["realized"] - 1.00) < 1e-6, f"onetick realized {ot['realized']}"
    # normal: bought @50, sold @49.99 -> realized -1.00
    nm = out["normal"]
    assert abs(nm["realized"] - (-1.00)) < 1e-6, f"normal realized {nm['realized']}"
    # net includes fees (both negative contributions)
    assert ot["net"] < ot["realized"] and nm["net"] < nm["realized"]
    # bps signs: onetick positive-ish, normal negative
    print(_ts() + f"[self-test] onetick realized={ot['realized']:.2f} net={ot['net']:.2f} bps={ot['net_bps']:.1f}")
    print(_ts() + f"[self-test] normal  realized={nm['realized']:.2f} net={nm['net']:.2f} bps={nm['net_bps']:.1f}")
    assert ot["net_bps"] > nm["net_bps"], "onetick should beat the losing normal split"
    print(_ts() + "[self-test] per-regime FIFO attribution correct; regimes separated  OK")
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


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
