# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# aggressor_flow_full.py
# ----------------------------------------------------------------------------
# FULL aggressor-flow study (the fast gate proved flow adds on top of OBI).
# Computes the volume-weighted signed-flow signal at TIME-BASED fade speeds
# (EMA half-lives 2s / 5s / 15s) -- which the feature store's fixed ~10-trade
# EMA could not give us -- via the order-book replay, then reports:
#
#   REGIME  : momentum score = mean( sign(flow) * forward_move ) at 1s/5s/30s.
#             +ve => price continues with the flow (MOMENTUM: pull the exposed
#             side). -ve => it reverts (MEAN REVERSION: lean INTO the flow).
#   ADDS-ON-TOP : the same momentum score computed ONLY on rows where the OBI
#             trigger is CALM -> does flow flag moves OBI misses?
#
#   swept over: half-life {2,5,15}s x fire threshold N-std {1,1.5,2,2.5},
#   per session bucket, DAY-AS-UNIT. OBI's own regime shown for reference.
#
# Signal per half-life hl: two time-decayed accumulators over trades --
#   signed += (+q buy / -q sell);  absv += q;  both decayed by 2^(-dt/hl).
#   flow_hl = signed / absv   in [-1, +1]   (volume-weighted, exp-decayed).
# FIRING: top (100-pct)% of |flow_hl| within each (symbol,day,bucket) -- percentile,
# NOT std-units (a smoothed EMA rarely reaches 2 std, which starved the first run).
#
# Built on the hardened replay scaffold: depth-accepting collector patch,
# built-in --smoke timing test, 6 workers, within-day progress, timestamps.
#
# USAGE:
#   python aggressor_flow_full.py --self-test
#   python aggressor_flow_full.py --smoke          # time ONE stock-day, no pool
#   python aggressor_flow_full.py --run            # overnight (default 60 days)
#   python aggressor_flow_full.py --run --days 0   # all days
# ============================================================================

import argparse
import time
import math
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import pandas as pd

_bi_print = print
def print(*a, **k):
    _bi_print(datetime.now().strftime("[%H:%M:%S]"), *a, **k)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Resolve this filesystem path through the canonical checkout/data configuration.
OUT_DIR = Path(str(_hft_paths.RESULTS_ROOT / 'diagnostics'))
HALFLIVES_S = [2.0, 5.0, 15.0]          # EMA fade speeds to sweep (seconds)
FIRE_PCTS = [80.0, 90.0, 95.0]          # fire on the top (100-pct)% of |flow| that
                                        # day/bucket -> a fixed FRACTION always fires,
                                        # so a smooth EMA is never starved of samples
MIN_NAMEDAYS = 20                       # don't report a cell built on fewer name-days
HORIZONS_MS = [1000, 5000, 30000]       # momentum measured at 1s/5s/30s
HEADLINE_MS = 5000
OBI_FIRE = 0.30                         # |obi_1|>=0.30 == the incumbent trigger
MIN_ROWS = 50
BUCKETS = ["first15", "middle", "preclose45", "last15"]
WORKERS = 6
MAX_DAYS = 60                           # overnight-friendly sample; --days 0 = all
TRAIL_DAYS = 10
LN2 = math.log(2.0)


def _bucket_of(ts, t0, t1):
    F = 15 * 60 * 1000; P = 45 * 60 * 1000; L = 15 * 60 * 1000
    b = np.full(ts.shape, "middle", dtype=object)
    b[ts <= t0 + F] = "first15"
    b[ts >= t1 - P] = "preclose45"
    b[ts >= t1 - L] = "last15"
    return b


# ===========================================================================
# COLLECTOR PATCH: add time-decayed signed-flow EMAs at each half-life
# ===========================================================================
def _install_flow_collector():
    import build_feature_store as FS
    C = FS.FeatureCollector
    if getattr(C, "_FLOW_PATCHED", False):
        return FS
    _orig_observe = C.observe

    def observe(self, kind, obj, ts_exch, mid):
        # lazy-init the per-half-life decayed accumulators on this collector
        if not hasattr(self, "_flow_s"):
            self._flow_s = {hl: 0.0 for hl in HALFLIVES_S}   # decayed signed shares
            self._flow_a = {hl: 0.0 for hl in HALFLIVES_S}   # decayed abs shares
            self._flow_last = None                           # last trade ts (ms)
        # update the flow EMAs on aggressor trades (same detection as the base:
        # a trade object carries aggressor_side)
        side = getattr(obj, "aggressor_side", None)
        if side in ("BUY", "SELL"):
            q = float(getattr(obj, "qty", 0.0) or 0.0)
            if q > 0.0:
                sv = q if side == "BUY" else -q
                for hl in HALFLIVES_S:
                    if self._flow_last is not None:
                        f = 2.0 ** (-(ts_exch - self._flow_last) / (hl * 1000.0))
                        self._flow_s[hl] *= f
                        self._flow_a[hl] *= f
                    self._flow_s[hl] += sv
                    self._flow_a[hl] += q
                self._flow_last = ts_exch
        # let the base collector do its thing (append a row on book events)
        n0 = len(self.rows)
        _orig_observe(self, kind, obj, ts_exch, mid)
        # if a row was emitted, snapshot the current flow signals onto it
        if len(self.rows) > n0:
            for hl in HALFLIVES_S:
                a = self._flow_a[hl]
                self.rows[-1][f"flow_{hl:g}s"] = (self._flow_s[hl] / a) if a > 1e-9 else 0.0

    def quotes(self, bb, bq, ba, aq, pos, depth=None, **kw):
        # passive collector; accept depth= (deep-OFI engine passes it)
        return {}

    C.observe = observe
    C.quotes = quotes
    C._FLOW_PATCHED = True
    return FS


_FS = None
_R = None
_NAMES = None


def _pool_init(names):
    global _FS, _R, _NAMES
    import run_legacy_mm as R
    _FS = _install_flow_collector()
    _R = R
    _NAMES = names


# ===========================================================================
# PER SYMBOL-DAY: build the frame, compute regime + adds-on-top rows
# ===========================================================================
def _one_symbol_day(FS, R, date, sym, dsets):
    df = FS.build_one(date, sym, dsets)
    if df is None or f"flow_{HALFLIVES_S[0]:g}s" not in df.columns:
        return [], "no-data"
    return _rows_from_df(df, date, sym), "ok"


def _rows_from_df(df, date, sym):
    need = ["ts_exch", "obi_1"] + [f"markout_{h}ms_bps" for h in HORIZONS_MS] \
           + [f"flow_{hl:g}s" for hl in HALFLIVES_S]
    d = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    if len(d) < 4 * MIN_ROWS:
        return []
    t0, t1 = d["ts_exch"].min(), d["ts_exch"].max()
    d["bucket"] = _bucket_of(d["ts_exch"].to_numpy(), t0, t1)
    obi = d["obi_1"].to_numpy()
    obi_calm = np.abs(obi) < OBI_FIRE
    moves = {h: d[f"markout_{h}ms_bps"].to_numpy() for h in HORIZONS_MS}
    out = []
    for b in BUCKETS:
        bm = d["bucket"].to_numpy() == b
        if bm.sum() < 2 * MIN_ROWS:
            continue
        # OBI reference regime (its own trigger), momentum at each horizon
        oa = np.abs(obi[bm]) >= OBI_FIRE
        if oa.sum() >= MIN_ROWS:
            rec = dict(date=date, symbol=sym, bucket=b, hl=np.nan, N=np.nan, kind="obi",
                       n_active=int(oa.sum()))
            for h in HORIZONS_MS:
                rec[f"mom_{h}"] = float(np.mean((np.sign(obi[bm]) * moves[h][bm])[oa]))
            out.append(rec)
        # flow, per half-life x FIRE percentile (self-calibrating, sample-safe)
        for hl in HALFLIVES_S:
            fl = d[f"flow_{hl:g}s"].to_numpy()[bm]
            afl = np.abs(fl)
            if not np.any(afl > 0):
                continue
            for pct in FIRE_PCTS:
                thr = np.percentile(afl, pct)
                act = afl >= thr
                if act.sum() < MIN_ROWS:
                    continue
                rec = dict(date=date, symbol=sym, bucket=b, hl=hl, N=pct, kind="flow",
                           n_active=int(act.sum()))
                for h in HORIZONS_MS:
                    mv = moves[h][bm]
                    rec[f"mom_{h}"] = float(np.mean((np.sign(fl) * mv)[act]))
                # ADDS-ON-TOP: momentum among OBI-calm rows only (5s)
                oc = obi_calm[bm]
                act_oc = act & oc
                rec["mom_ontop_5000"] = (float(np.mean((np.sign(fl) * moves[HEADLINE_MS][bm])[act_oc]))
                                         if act_oc.sum() >= MIN_ROWS else np.nan)
                out.append(rec)
    return out


def _work_date(date):
    dsets = _R.open_datasets(date)
    if dsets is None:
        return []
    rows = []
    t0 = time.perf_counter()
    for i, sym in enumerate(_NAMES, 1):
        try:
            rr, _ = _one_symbol_day(_FS, _R, date, sym, dsets)
            rows.extend(rr)
        except Exception as e:
            print(f"SKIP {date} {sym}: {e!r}")
            continue
        if i % 10 == 0:
            print(f"  {date}: {i}/{len(_NAMES)} stocks ({time.perf_counter()-t0:.0f}s)")
    return rows


# ===========================================================================
# STATS + DRIVER
# ===========================================================================
def _t(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = x.size
    if n < 2:
        return np.nan, np.nan, n
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)), n


def smoke(symbols=None):
    import run_legacy_mm as R
    import mm_harness as H
    FS = _install_flow_collector()
    all_dates = R.discover_dates()
    names = symbols or (list(H.NAMES) if hasattr(H, "NAMES") else sorted(H.load_scales().keys()))
    date = all_dates[len(all_dates) // 2]
    dsets = R.open_datasets(date)
    print(f"[smoke] one stock-day: {names[0]} on {date}")
    t0 = time.perf_counter()
    rows, note = _one_symbol_day(FS, R, date, names[0], dsets)
    dt = time.perf_counter() - t0
    print(f"[smoke] {names[0]} {date}: {note}, {len(rows)} rows in {dt:.1f}s")
    ndays = MAX_DAYS if MAX_DAYS else len(all_dates)
    print(f"[smoke] one unit ~{dt:.1f}s. Full run {len(names)}x{ndays} / {WORKERS} workers "
          f"~= {len(names)*ndays*dt/WORKERS/60:.0f} min")


def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    import run_legacy_mm as R
    import mm_harness as H
    print("pre-pass: dates + symbol universe")
    all_dates = R.discover_dates()
    names = symbols or (list(H.NAMES) if hasattr(H, "NAMES") else sorted(H.load_scales().keys()))
    run_dates = all_dates[TRAIL_DAYS:]
    if max_days and len(run_dates) > max_days:
        stride = max(1, len(run_dates) // max_days)
        run_dates = run_dates[::stride][:max_days]
    print(f"[flow-full] {len(names)} names x {len(run_dates)} dates, {workers} workers, "
          f"half-lives {HALFLIVES_S}s")
    rows = []
    t0 = time.perf_counter()
    with Pool(processes=workers, initializer=_pool_init, initargs=(names,)) as pool:
        done = 0
        for res in pool.imap_unordered(_work_date, run_dates):
            rows.extend(res); done += 1
            el = (time.perf_counter() - t0) / 60.0
            eta = el / done * (len(run_dates) - done)
            print(f"  date {done}/{len(run_dates)} done ({len(rows)} rows, {el:.1f} min elapsed, ETA {eta:.1f} min)")
    if not rows:
        print("no runnable symbol-days."); return
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "aggressor_flow_full_raw.csv", index=False)
    _report(df)
    print(f"[flow-full] outputs -> {out_dir}")


def _report(df):
    print("===== AGGRESSOR FLOW (full): MOMENTUM vs REVERSION by fade speed =====")
    print("  momentum score = avg forward move IN THE FLOW DIRECTION (day-as-unit).")
    print("  +ve = MOMENTUM (pull the exposed side).  -ve = MEAN REVERSION (lean in).\n")
    # A) regime by half-life, at N=2.0, per bucket: momentum 1s/5s/30s
    print("  --- REGIME (fire on top 10% of |flow|) ---")
    for b in BUCKETS:
        print(f"   [{b}]")
        obi = df[(df.bucket == b) & (df.kind == "obi")]
        if len(obi):
            v = "  ".join(f"{h//1000}s:{_t(obi[f'mom_{h}'].to_numpy())[0]:+5.2f}" for h in HORIZONS_MS)
            print(f"      OBI(ref)      {v}")
        for hl in HALFLIVES_S:
            s = df[(df.bucket == b) & (df.kind == "flow") & (df.hl == hl) & (df.N == 90.0)]
            if len(s):
                def cell(h):
                    m, se, nn = _t(s[f'mom_{h}'].to_numpy())
                    return f"{h//1000}s:{m:+5.2f}" if nn >= MIN_NAMEDAYS else f"{h//1000}s: n/a"
                print(f"      flow {hl:g}s       " + "  ".join(cell(h) for h in HORIZONS_MS)
                      + f"   [nd={_t(s['mom_5000'].to_numpy())[2]}]")
    # B) adds-on-top-of-OBI (momentum at 5s among OBI-calm rows), per half-life x N
    print("\n  --- ADDS ON TOP OF OBI (5s momentum among OBI-calm rows) ---")
    print("  (clearly non-zero => flow flags 5s moves OBI misses)")
    for b in ["middle", "preclose45"]:                 # the buckets with signal + sample
        print(f"   [{b}]")
        for hl in HALFLIVES_S:
            parts = []
            for pct in FIRE_PCTS:
                s = df[(df.bucket == b) & (df.kind == "flow") & (df.hl == hl) & (df.N == pct)]
                m, se, n = _t(s["mom_ontop_5000"].to_numpy())
                top = int(100 - pct)
                parts.append(f"top{top}%:{m:+5.2f}" if n >= MIN_NAMEDAYS else f"top{top}%: n/a")
            print(f"      flow {hl:g}s   " + "  ".join(parts))
    print("\n  READ: pick the half-life with the strongest, most consistent signal.")
    print("        all + across horizons -> momentum -> aggressor-flow THROTTLE.")
    print("        + at 1s then - at 30s -> short momentum then reversion.")


# ===========================================================================
# SELF-TEST
# ===========================================================================
def self_test():
    # 1) time-decayed EMA math on a hand sequence
    class Obj:
        def __init__(self, side, qty): self.aggressor_side = side; self.qty = qty
    import types
    # minimal fake collector exercising just the flow-EMA wrapper logic
    s = {hl: 0.0 for hl in HALFLIVES_S}; a = {hl: 0.0 for hl in HALFLIVES_S}; last = [None]
    def upd(side, qty, ts):
        sv = qty if side == "BUY" else -qty
        for hl in HALFLIVES_S:
            if last[0] is not None:
                f = 2.0 ** (-(ts - last[0]) / (hl * 1000.0))
                s[hl] *= f; a[hl] *= f
            s[hl] += sv; a[hl] += qty
        last[0] = ts
    upd("BUY", 100, 0)
    # all-buy so far -> signal = +1 at every half-life
    for hl in HALFLIVES_S:
        assert abs(s[hl] / a[hl] - 1.0) < 1e-12
    # one half-life (2s) later, add an equal SELL: 2s EMA should be ~0 (old buy halved,
    # new sell full) -> signed = 0.5*100 - 100 = -50, abs = 0.5*100 + 100 = 150 -> -1/3
    upd("SELL", 100, 2000)
    sig2 = s[2.0] / a[2.0]
    print(f"[self-test] 2s flow after BUY@0, SELL@2s = {sig2:+.4f} (expect -0.3333)")
    assert abs(sig2 - (-1.0 / 3.0)) < 1e-9
    # the 15s EMA barely decayed the buy -> closer to 0 from the top: signed = ~0.912*100-100
    sig15 = s[15.0] / a[15.0]
    print(f"[self-test] 15s flow same = {sig15:+.4f} (less decayed -> nearer 0 from +side? check sign)")
    # 15s: f=2^(-2/15)=0.912; signed=91.2-100=-8.8; abs=91.2+100=191.2 -> -0.046
    assert abs(sig15 - (-8.79/191.2)) < 0.02

    # 2) regime recovery on synthetic frames (momentum + reversion + null)
    rng = np.random.default_rng(0); n = 6000
    T0 = 0; T1 = int(5.5 * 3600 * 1000); ts = np.sort(rng.uniform(T0, T1, n))
    z = rng.normal(0, 1, n)
    base = {"ts_exch": ts, "obi_1": rng.normal(0, 1, n)}
    for hl in HALFLIVES_S:
        base[f"flow_{hl:g}s"] = z + rng.normal(0, 0.4, n)      # flow tracks latent z
    mom = 3.0 * z + rng.normal(0, 3.0, n)
    dfm = pd.DataFrame(dict(base, markout_1000ms_bps=mom, markout_5000ms_bps=mom, markout_30000ms_bps=mom))
    r = pd.DataFrame(_rows_from_df(dfm, "d", "s"))
    got = r[(r.kind == "flow") & (r.N == 90.0)]["mom_5000"].dropna().mean()
    print(f"[self-test] MOMENTUM world 5s momentum = {got:+.3f} (expect clearly +)")
    assert got > 0.5
    rev = -3.0 * z + rng.normal(0, 3.0, n)
    dfr = pd.DataFrame(dict(base, markout_1000ms_bps=rev, markout_5000ms_bps=rev, markout_30000ms_bps=rev))
    rr = pd.DataFrame(_rows_from_df(dfr, "d", "s"))
    gotr = rr[(rr.kind == "flow") & (rr.N == 90.0)]["mom_5000"].dropna().mean()
    print(f"[self-test] REVERSION world 5s momentum = {gotr:+.3f} (expect clearly -)")
    assert gotr < -0.5
    nul = rng.normal(0, 3.0, n)
    dfn = pd.DataFrame(dict(base, markout_1000ms_bps=nul, markout_5000ms_bps=nul, markout_30000ms_bps=nul))
    rn = pd.DataFrame(_rows_from_df(dfn, "d", "s"))
    gotn = rn[(rn.kind == "flow") & (rn.N == 90.0)]["mom_5000"].dropna().mean()
    print(f"[self-test] NULL world 5s momentum = {gotn:+.3f} (expect ~0)")
    assert abs(gotn) < 0.5
    bk = _bucket_of(np.array([T0+1, T0+16*60000, T1-46*60000, T1-40*60000, T1-1]), T0, T1)
    assert list(bk) == ["first15", "middle", "middle", "preclose45", "last15"]
    print("[self-test] ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Full aggressor-flow study (time-decayed, swept).")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="time ONE stock-day, no pool")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--days", type=int, default=MAX_DAYS, help="0 = all days")
    args = ap.parse_args()
    if args.smoke:
        smoke(symbols=args.symbols)
    elif args.self_test or not args.run:
        self_test()
    if args.run:
        run_real(symbols=args.symbols, workers=args.workers, max_days=(args.days or None))
