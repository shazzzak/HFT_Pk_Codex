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
WORKERS = 2


# split fills by regime and FIFO-attribute P&L within each regime.
# Returns {regime: {capture, markout, fee, net, fills, pnl}} using the harness FIFO.
def attribute_by_regime(dr, H):
    """FIFO round-trip P&L attributed to the OPENING fill's regime.

    THE FIX (SZ): a fill's own regime tag is the regime at the instant THAT fill
    happened -- so a trip opened in the 1-tick regime but closed in the normal
    regime would split its two legs across buckets (entry->onetick, exit profit->
    normal), making onetick look worse and normal better. That is an attribution
    artifact, not economics. Here every open LOT carries the regime it was opened
    in, and the ENTIRE realized round-trip P&L (both legs) books to that OPENING
    regime. "onetick P&L" then means "money on positions ENTERED during 1-tick
    moments" -- the actual question. Exit regime is incidental.
    """
    f = dr.fills
    if f is None or len(f) == 0 or "regime" not in f.columns:
        return {}
    # single time-ordered stream (NOT split by regime -- FIFO matches across them)
    recs = f.sort_values("t").to_dict("records")
    from collections import deque
    # signed FIFO queue of open lots: each is (price, signed_qty, open_regime)
    lots = deque()
    # per-opening-regime accumulators
    from collections import defaultdict
    realized = defaultdict(float)      # realized round-trip P&L, by OPENING regime
    matched_notional = defaultdict(float)
    fee_by = defaultdict(float)        # fees booked to the regime of the fill that paid them
    fills_by = defaultdict(int)
    # walk the single stream
    for r in recs:
        reg = r.get("regime", "normal")
        # count the fill + its fee under the regime it occurred in
        fills_by[reg] += 1
        fee_by[reg] += FEE_BPS / 1e4 * abs(r["qty"]) * r["px"]
        # signed incoming qty
        q = r["qty"] * (1 if r["side"] == "BUY" else -1)
        # match against opposite-sign open lots (FIFO)
        while lots and q != 0 and (lots[0][1] * q < 0):
            px0, q0, oreg = lots[0]
            m = min(abs(q), abs(q0))
            # realized P&L of this round trip, booked to the OPENING regime (oreg)
            realized[oreg] += (r["px"] - px0) * (m if q0 > 0 else -1 * m)
            matched_notional[oreg] += m * px0
            # shrink/remove the lot (keep its opening regime)
            if abs(q0) == m:
                lots.popleft()
            else:
                lots[0] = (px0, q0 - (m if q0 > 0 else -m), oreg)
            q -= (m if q > 0 else -m)
        # remainder opens a NEW lot, tagged with THIS fill's regime as its open regime
        if q != 0:
            lots.append((r["px"], q, reg))
    # assemble per-regime result. Fees are attributed to the regime the fill
    # occurred in (a fill's fee is paid when the fill happens, regardless of the
    # round trip's opening regime) -- so net = realized(open-regime) minus fees is
    # only exact at the TOTAL level; per-regime net uses that regime's own fees as
    # a reasonable split. Flagged: the clean per-regime number is `realized`.
    out = {}
    regimes = set(list(realized.keys()) + list(fills_by.keys()))
    for reg in regimes:
        rz = realized.get(reg, 0.0)
        fe = fee_by.get(reg, 0.0)
        mn = matched_notional.get(reg, 0.0)
        net = rz - fe
        out[reg] = dict(realized=rz, fee=fe, net=net,
                        matched_notional=mn,
                        net_bps=(net / mn * 1e4) if mn > 0 else float("nan"),
                        n_fills=fills_by.get(reg, 0))
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
    # rows
    rows = []
    # each regime
    for regime, m in attr.items():
        # record
        rows.append(dict(config=thr_label, symbol=sym, date=str(date), regime=regime,
                         net=m["net"], net_bps=m["net_bps"], n_fills=m["n_fills"],
                         matched_notional=m["matched_notional"]))
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
    # a fake DayResult-like object with a fills frame
    class DR: pass
    dr = DR()
    # fills: onetick regime buy@100 then sell@100.01 (+1 tick round trip);
    # normal regime buy@50 then sell@49.99 (-1 tick, a loss)
    dr.fills = pd.DataFrame([
        dict(t=1, side="BUY", px=100.00, qty=100, reason="x", regime="onetick"),
        dict(t=2, side="SELL", px=100.01, qty=100, reason="x", regime="onetick"),
        dict(t=3, side="BUY", px=50.00, qty=100, reason="x", regime="normal"),
        dict(t=4, side="SELL", px=49.99, qty=100, reason="x", regime="normal"),
    ])
    out = attribute_by_regime(dr, None)
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
