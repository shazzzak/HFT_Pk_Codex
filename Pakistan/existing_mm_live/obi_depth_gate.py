# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# obi_depth_gate.py
# ----------------------------------------------------------------------------
# Which order-book-imbalance DEPTH is the best toxicity predictor?
# Compares obi_1 / obi_3 / obi_5 / obi_7 -- each CUMULATIVE over levels 1..N
# (Book.obi(n) sums the top n levels) -- as predictors of forward side-signed
# markout. The throttle/defensive logic only ever used L1 (imb = bq/(bq+aq));
# obi_5/obi_deep are computed but never wired as the trigger, and 3/7 never
# existed. This gate says whether a deeper OBI separates toxic from benign
# better than L1 -- i.e. whether a deeper-OBI throttle trigger is worth building.
#
# METHOD (mirrors the throttle markout gate + the boost/QDR gates):
#   * reuse build_feature_store.build_one (validated Book replay + leak-checked
#     forward-ASOF markout); a 3-line FeatureCollector subclass adds obi_3/obi_7.
#   * per depth d, per symbol-day: side-signed ADVERSE markout, where the adverse
#     side is the one obi_d points against (obi_d>0 bid-heavy -> up -> a SELL is
#     adverse). gap = mean(adverse | |obi_d| top decile) - mean(adverse | bottom
#     decile). A stronger predictor => a MORE NEGATIVE gap. Plus Spearman(obi_d, m).
#   * DAY-AS-UNIT error bars; PER-NAME axis retained. Headline horizon 5s
#     (matches the throttle diagnostic's ~-0.93 bps L1 gap -> d=1 should ~reproduce it).
#
# USAGE:
#   python obi_depth_gate.py --self-test
#   python obi_depth_gate.py --run                 # 30 sampled days, 9 workers
# ============================================================================

import argparse
import time
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# standing convention: timestamp every printed line.
_bi_print = print
def print(*a, **k):
    _bi_print(datetime.now().strftime("[%H:%M:%S]"), *a, **k)

# optional SciPy for Spearman; fall back to a rank-Pearson if absent.
try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:
    _sps = None
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Resolve this filesystem path through the canonical checkout/data configuration.
OUT_DIR = Path(str(_hft_paths.RESULTS_ROOT / 'diagnostics'))
# cumulative OBI depths to compare (1..N).
DEPTHS = [1, 3, 5, 7]
# markout horizons (ms); 5s headline (matches the throttle gate).
HORIZONS_MS = [1000, 5000, 30000]
HEADLINE_MS = 5000
# active = top-decile |obi_d|; neutral = bottom-decile |obi_d| (per symbol-day).
ACTIVE_Q = 0.90
NEUTRAL_Q = 0.10
# need this many rows in each state per symbol-day to trust that day.
MIN_ROWS = 50
WORKERS = 9
MAX_DAYS = 10
THIN = {"FNEL", "TPL", "PACE", "PIAHCLA", "HASCOL", "NPL", "TOMCL"}
DEEP = {"OGDC", "PSO", "HUBC", "FFC", "PPL", "UBL", "NBP", "MEBL"}


# ===========================================================================
# GATE MATH (operates on a df with obi_{d} + markout_{h}ms_bps columns)
# ===========================================================================
def _spearman(x, y):
    # rank correlation; NaN-safe; fall back to rank-Pearson without SciPy.
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 10 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    if _HAVE_SCIPY:
        return float(_sps.spearmanr(x, y).correlation)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def symbolday_gate(df, depth, horizon_ms):
    # obi at this depth and the forward markout label.
    ocol, mcol = f"obi_{depth}", f"markout_{horizon_ms}ms_bps"
    if ocol not in df.columns or mcol not in df.columns:
        return None
    d = df[[ocol, mcol]].dropna()
    if len(d) < 2 * MIN_ROWS:
        return None
    obi = d[ocol].to_numpy(dtype=float)
    m = d[mcol].to_numpy(dtype=float)
    # ADVERSE side-signed markout: obi>0 (bid-heavy) -> up -> a SELL is adverse,
    # so the adverse fill loses when mid rises => adverse = -sign(obi)*m.
    adverse = -np.sign(obi) * m
    # active / neutral by |obi_d| deciles WITHIN this symbol-day.
    a = np.abs(obi)
    hi, lo = np.quantile(a, ACTIVE_Q), np.quantile(a, NEUTRAL_Q)
    act = adverse[a >= hi]
    neu = adverse[a <= lo]
    if act.size < MIN_ROWS or neu.size < MIN_ROWS:
        return None
    # gap<0 => adverse states really are toxic (stronger = more negative).
    gap = float(np.mean(act) - np.mean(neu))
    # directional rank corr: does obi_d predict the (signed) forward drift?
    rho = _spearman(obi, m)
    return dict(depth=depth, gap_bps=gap, rho=rho, n=int(len(d)))


# ===========================================================================
# WORKER: reuse build_one, with a collector that also records obi_3 / obi_7
# ===========================================================================
def _install_depth_collector():
    # Patch FeatureCollector's methods IN PLACE (not subclass+reassign, which
    # depended on build_one picking up a rebound name). This mutates the class
    # object itself, so EVERY FeatureCollector instance build_one creates gets:
    #   * quotes(depth=...) accepted (the deep-OFI engine passes depth=)
    #   * obi_3 / obi_7 recorded onto each emitted row
    import build_feature_store as FS
    C = FS.FeatureCollector
    if getattr(C, "_DEPTH_PATCHED", False):
        return FS
    _orig_observe = C.observe

    def observe(self, kind, obj, ts_exch, mid):
        # super()'s observe appends 0/1 row and does NOT mutate the book, so the
        # book state after == state at append time -> obi(3)/obi(7) are correct.
        n0 = len(self.rows)
        _orig_observe(self, kind, obj, ts_exch, mid)
        if len(self.rows) > n0:
            self.rows[-1]["obi_3"] = self.book.obi(3)
            self.rows[-1]["obi_7"] = self.book.obi(7)

    def quotes(self, bb, bq, ba, aq, pos, depth=None, **kw):
        # passive collector: never quotes. Accept depth= (deep-OFI engine) + any kwarg.
        return {}

    C.observe = observe
    C.quotes = quotes
    C._DEPTH_PATCHED = True
    return FS


_FS = None
_R = None


def _init_worker():
    global _FS, _R
    import run_legacy_mm as R
    _FS = _install_depth_collector()
    _R = R


def _one_symbol_day(date, sym, dsets):
    # build one stock-day and its per-depth/per-horizon gate rows (shared by the
    # smoke test and the workers). Returns (rows, note).
    df = _FS.build_one(date, sym, dsets)
    if df is None or "obi_3" not in df.columns:
        return [], "no-data"
    rows = []
    for h in HORIZONS_MS:
        for d in DEPTHS:
            r = symbolday_gate(df, d, h)
            if r is None:
                continue
            r.update(dict(date=date, symbol=sym, horizon_ms=h))
            rows.append(r)
    return rows, "ok"


def _work_date(date):
    import time as _tm
    dsets = _R.open_datasets(date)
    if dsets is None:
        return []
    rows = []
    syms = _CALIB_NAMES
    t0 = _tm.perf_counter()
    for i, sym in enumerate(syms, 1):
        try:
            rr, _ = _one_symbol_day(date, sym, dsets)
            rows.extend(rr)
        except Exception as e:
            print(f"SKIP {date} {sym}: {e!r}")
            continue
        # within-day heartbeat so a slow date still shows movement
        if i % 10 == 0:
            print(f"  {date}: {i}/{len(syms)} stocks ({_tm.perf_counter()-t0:.0f}s)")
    return rows


# names shared to workers (set in run_real, read in _work_date)
_CALIB_NAMES = None


def _pool_init(names):
    global _CALIB_NAMES
    _init_worker()
    _CALIB_NAMES = names


# ===========================================================================
# DAY-AS-UNIT stats + driver
# ===========================================================================
def _t(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = x.size
    if n < 2:
        return np.nan, np.nan, n
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    return m, se, n


def smoke(symbols=None):
    # ONE stock-day, in THIS process (no pool) -> catches crashes/hangs in seconds
    # and times a single unit so we can predict the full run before launching it.
    import time as _tm
    import run_legacy_mm as R
    import mm_harness as H
    _init_worker()  # installs the collector patch + sets _R/_FS in this process
    all_dates = R.discover_dates()
    names = symbols or (list(H.NAMES) if hasattr(H, "NAMES") else sorted(H.load_scales().keys()))
    date = all_dates[len(all_dates)//2]          # a mid-panel date
    dsets = R.open_datasets(date)
    print(f"[smoke] one stock-day: {names[0]} on {date}")
    t0 = _tm.perf_counter()
    rows, note = _one_symbol_day(date, names[0], dsets)
    dt = _tm.perf_counter() - t0
    print(f"[smoke] {names[0]} {date}: {note}, {len(rows)} rows in {dt:.1f}s")
    if rows:
        hd = [r for r in rows if r["horizon_ms"] == HEADLINE_MS]
        for r in sorted(hd, key=lambda x: x["depth"]):
            print(f"[smoke]   obi_{r['depth']}: gap={r['gap_bps']:+.3f}  rho={r['rho']:+.4f}")
    # predict the full run
    n_units = len(names) * min(len(all_dates), MAX_DAYS)
    print(f"[smoke] one unit = {dt:.1f}s. Full run {len(names)}x{min(len(all_dates),MAX_DAYS)} "
          f"= {n_units} units / {WORKERS} workers ~= {n_units*dt/WORKERS/60:.1f} min")
    return dt


def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    import run_legacy_mm as R
    import mm_harness as H
    print("pre-pass: dates + symbol universe")
    all_dates = R.discover_dates()
    names = symbols or (list(H.NAMES) if hasattr(H, "NAMES") else sorted(H.load_scales().keys()))
    # sample days evenly (a signal-quality gate is day-stable)
    run_dates = all_dates
    if max_days and len(run_dates) > max_days:
        stride = max(1, len(run_dates) // max_days)
        run_dates = run_dates[::stride][:max_days]
    print(f"[obi-depth] {len(names)} names x {len(run_dates)} dates (sampled), {workers} workers")
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
        print("no runnable symbol-days.")
        return
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "obi_depth_gate_pername_day.csv", index=False)
    # ---- headline horizon: gap & rho by depth, day-as-unit ----
    hd = df[df.horizon_ms == HEADLINE_MS]
    print(f"===== OBI-DEPTH TOXICITY GATE @ {HEADLINE_MS}ms (day-as-unit; unit=name-day) =====")
    print(f"  depth   adverse_gap_bps (more negative = stronger)     rank_rho(obi,markout)")
    summ = []
    for d in DEPTHS:
        sub = hd[hd.depth == d]
        gm, gse, gn = _t(sub["gap_bps"].to_numpy())
        rm, rse, rn = _t(sub["rho"].to_numpy())
        summ.append(dict(depth=d, gap=gm, gap_se=gse, rho=rm, rho_se=rse, n=gn))
        print(f"   obi_{d}     {gm:+7.3f} +/- {gse:.3f}                        {rm:+.4f} +/- {rse:.4f}")
    cv = pd.DataFrame(summ)
    cv.to_csv(out_dir / "obi_depth_gate_summary.csv", index=False)
    # best depth by strongest (most negative) gap and by |rho|
    best_gap = cv.loc[cv["gap"].idxmin(), "depth"]
    best_rho = cv.loc[cv["rho"].abs().idxmax(), "depth"]
    print(f"  strongest toxicity gap: obi_{best_gap}   |   best directional rho: obi_{best_rho}")
    # ---- per-name: which depth wins per name (headline) ----
    pern = (hd.groupby(["symbol", "depth"])["gap_bps"].mean().reset_index()
            .pivot(index="symbol", columns="depth", values="gap_bps"))
    pern.to_csv(out_dir / "obi_depth_gate_pername.csv")
    _plots(cv, hd, out_dir)
    print(f"[obi-depth] outputs -> {out_dir}")


def _plots(cv, hd, out_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    a1.errorbar(cv["depth"], cv["gap"], yerr=cv["gap_se"], marker="o", capsize=4, lw=1.8, color="#c0392b")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xticks(DEPTHS); a1.set_xlabel("cumulative OBI depth (levels 1..N)")
    a1.set_ylabel("adverse-vs-neutral markout gap (bps)")
    a1.set_title("Toxicity gap by depth (more negative = better predictor)")
    a1.grid(alpha=0.25)
    a2.errorbar(cv["depth"], cv["rho"].abs(), yerr=cv["rho_se"], marker="o", capsize=4, lw=1.8, color="#2c6fbb")
    a2.set_xticks(DEPTHS); a2.set_xlabel("cumulative OBI depth (levels 1..N)")
    a2.set_ylabel("|rank corr| of obi_d with forward markout")
    a2.set_title("Directional predictive power by depth")
    a2.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_dir / "obi_depth_gate.png", dpi=130); plt.close(fig)


# ===========================================================================
# SELF-TEST: inject a KNOWN depth ordering, assert the gate recovers it
# ===========================================================================
def self_test():
    rng = np.random.default_rng(0)
    n = 20000
    # build obi_1..7 that get progressively MORE predictive of the forward move:
    # a shared latent direction 'z'; deeper obi tracks z more tightly (less noise).
    z = rng.normal(0, 1, n)
    obi = {}
    for d, noise in [(1, 1.2), (3, 0.8), (5, 0.5), (7, 0.3)]:
        obi[d] = np.tanh(z + rng.normal(0, noise, n))   # in (-1,1), deeper=cleaner
    # forward markout driven by the latent direction (+ noise): up when z>0
    m = 3.0 * np.tanh(z) + rng.normal(0, 3.0, n)
    df = pd.DataFrame({f"obi_{d}": obi[d] for d in DEPTHS})
    df["markout_5000ms_bps"] = m
    print("[self-test] depth ->  gap_bps   rho")
    res = {}
    for d in DEPTHS:
        r = symbolday_gate(df, d, 5000)
        res[d] = r
        print(f"[self-test]  obi_{d}:  {r['gap_bps']:+7.3f}   {r['rho']:+.4f}")
    # deeper must be a STRONGER predictor: more negative gap and higher rho
    assert res[7]["gap_bps"] < res[1]["gap_bps"], "gap should strengthen (more neg) with depth"
    assert res[5]["gap_bps"] < res[1]["gap_bps"], "obi_5 should beat obi_1 here"
    assert abs(res[7]["rho"]) > abs(res[1]["rho"]), "rho should rise with depth"
    # NULL: an obi column unrelated to markout -> ~zero gap, ~zero rho
    df["obi_1"] = rng.normal(0, 1, n)  # scramble depth-1 to pure noise
    r0 = symbolday_gate(df, 1, 5000)
    print(f"[self-test]  NULL obi_1 (scrambled): gap={r0['gap_bps']:+.3f} rho={r0['rho']:+.4f}")
    assert abs(r0["gap_bps"]) < 0.6 and abs(r0["rho"]) < 0.05, "null should show no signal"
    # side-signing sanity: flipping obi sign flips the adverse gap's driver, not its sign
    # (adverse = -sign(obi)*m; a genuinely toxic signal stays negative) -- covered above.
    print("[self-test] ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OBI-depth toxicity gate (obi_1/3/5/7).")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="time ONE stock-day, no pool")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    args = ap.parse_args()
    if args.smoke:
        smoke(symbols=args.symbols)
    elif args.self_test or not args.run:
        self_test()
    if args.run:
        run_real(symbols=args.symbols, workers=args.workers, max_days=args.days)
