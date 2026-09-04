# ============================================================================
# aggressor_flow_gate.py
# ----------------------------------------------------------------------------
# Does aggressor trade-flow warn us of toxic fills -- BETTER THAN, or ON TOP OF,
# the OBI trigger we already run? Fast check straight off the feature store
# (no order-book replay): uses the already-computed volume-weighted signed-flow
# EMA (ewma_trade_flow) and the leak-checked forward-markout columns.
#
# YOUR DESIGN, tested offline:
#   * signal = ewma_trade_flow (volume-weighted signed flow, exponentially decayed)
#   * NORMALIZE per (symbol, day, bucket): z = flow / std(flow in that bucket)
#     -> "how many standard deviations from normal", self-calibrating per stock
#     and per time-of-day. (Offline we have the whole bucket, so no cold-start.)
#   * FIRE when |z| >= N; SWEEP N in {1.0, 1.5, 2.0, 2.5}.
#   * REGIME FIRST (momentum vs mean reversion), NOT an assumed 'toxic side':
#     for extreme flow, MOMENTUM SCORE = mean( sign(flow) * forward_move ).
#       > 0  => price moves WITH the flow (MOMENTUM): the side flow lifts is
#              toxic -> the defensive 'pull the exposed side' response applies.
#       < 0  => price moves AGAINST the flow (MEAN REVERSION): selling into the
#              spike is PROFITABLE -> LEAN INTO the flow (harvest reversion),
#              the opposite of pulling.
#     Reported at 1s / 5s / 30s (short momentum can flip to longer reversion),
#     per bucket, DAY-AS-UNIT. OBI's own momentum score shown for reference.
#
# LIMITATION (honest): the EMA fade is fixed (~10-trade half-life, whatever the
# feature store built). This gate CANNOT sweep the fade -- if it proves out, the
# time-based half-life sweep needs the replay path. This is the cheap first look.
#
# USAGE:
#   python aggressor_flow_gate.py --self-test
#   python aggressor_flow_gate.py --run
# ============================================================================

import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# standing convention: timestamp every printed line.
_bi_print = print
def print(*a, **k):
    _bi_print(datetime.now().strftime("[%H:%M:%S]"), *a, **k)

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:
    _sps = None
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
OUT_DIR = Path("/Users/shazzak/Capital Stake - Results/diagnostics")
HEADLINE_MS = 5000                      # markout horizon (matches the throttle gate)
HORIZONS_MS = [1000, 5000, 30000]
N_STDS = [1.0, 1.5, 2.0, 2.5]           # fire thresholds (standard deviations from 0)
# OBI incumbent trigger level (|imb-0.5|>0.15 == |obi_1|>0.30 on the (bq-aq)/(bq+aq) scale)
OBI_FIRE = 0.30
MIN_ROWS = 50                           # per (symbol,day,bucket) to trust it
# session buckets (minutes): first 15 / last 15 / preclose 45->15 / middle
BUCKETS = ["first15", "middle", "preclose45", "last15"]
THIN = {"FNEL", "TPL", "PACE", "PIAHCLA", "HASCOL", "NPL", "TOMCL"}
DEEP = {"OGDC", "PSO", "HUBC", "FFC", "PPL", "UBL", "NBP", "MEBL"}


def _bucket(ts, t0, t1):
    # time-of-day bucket from the event timestamp and the day's session bounds.
    F = 15 * 60 * 1000; P = 45 * 60 * 1000; L = 15 * 60 * 1000
    b = np.full(ts.shape, "middle", dtype=object)
    b[ts <= t0 + F] = "first15"
    b[ts >= t1 - P] = "preclose45"     # 45->15 before close
    b[ts >= t1 - L] = "last15"         # last 15 (overwrites preclose in its range)
    return b


# ===========================================================================
# GATE MATH -- operates on ONE symbol-day frame; returns per-bucket gaps
# ===========================================================================
def _mom(signal, move, active):
    # MOMENTUM SCORE: average forward move measured IN THE SIGNAL'S DIRECTION,
    # over the rows where the signal fired. +ve = price continued with the flow
    # (momentum); -ve = price reverted against it (mean reversion).
    a = (np.sign(signal) * move)[active]
    if a.size < MIN_ROWS:
        return np.nan, int(a.size)
    return float(np.mean(a)), int(a.size)


def symbolday(df):
    # one symbol-day; returns per (bucket, N) the flow momentum score at each
    # horizon, plus the OBI momentum score for reference.
    need = ["ewma_trade_flow", "obi_1"] + [f"markout_{h}ms_bps" for h in HORIZONS_MS]
    d = df.dropna(subset=need).copy()
    if len(d) < 4 * MIN_ROWS:
        return []
    t0, t1 = d["ts_exch"].min(), d["ts_exch"].max()
    d["bucket"] = _bucket(d["ts_exch"].to_numpy(), t0, t1)
    flow = d["ewma_trade_flow"].to_numpy()
    obi = d["obi_1"].to_numpy()
    moves = {h: d[f"markout_{h}ms_bps"].to_numpy() for h in HORIZONS_MS}
    out = []
    for b in BUCKETS:
        mask = d["bucket"].to_numpy() == b
        if mask.sum() < 2 * MIN_ROWS:
            continue
        fl = flow[mask]; ob = obi[mask]
        sd = np.std(fl, ddof=1)
        if sd <= 0:
            continue
        z = fl / sd
        # OBI reference (fires on the incumbent level), momentum at each horizon
        obi_active = np.abs(ob) >= OBI_FIRE
        rec = dict(bucket=b, N=np.nan, kind="obi")
        for h in HORIZONS_MS:
            mm, nn = _mom(ob, moves[h][mask], obi_active)
            rec[f"mom_{h}"] = mm
        rec["n_active"] = int(obi_active.sum())
        out.append(rec)
        # flow, at each fire threshold N, momentum at each horizon
        for N in N_STDS:
            act = np.abs(z) >= N
            rec = dict(bucket=b, N=N, kind="flow")
            for h in HORIZONS_MS:
                mm, nn = _mom(fl, moves[h][mask], act)
                rec[f"mom_{h}"] = mm
            rec["n_active"] = int(act.sum())
            out.append(rec)
    return out


# ===========================================================================
# DRIVER (feature-store read; no replay)
# ===========================================================================
def _t(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = x.size
    if n < 2:
        return np.nan, np.nan, n
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)), n


def run_real(root=FS_ROOT, out_dir=OUT_DIR, symbols=None):
    if not root.exists():
        print(f"feature store not found: {root}"); return
    sym_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if symbols:
        sym_dirs = [p for p in sym_dirs if p.name in set(symbols)]
    cols = ["ts_exch", "ewma_trade_flow", "obi_1"] + [f"markout_{h}ms_bps" for h in HORIZONS_MS]
    rows = []
    t0 = time.perf_counter(); nfile = 0
    for sp in sym_dirs:
        for fp in sorted(sp.glob("date=*.parquet")):
            try:
                df = pd.read_parquet(fp, columns=cols)
            except Exception as e:
                print(f"SKIP {sp.name}/{fp.name}: {e!r}"); continue
            for r in symbolday(df):
                r.update(dict(symbol=sp.name, date=fp.stem.replace("date=", "")))
                rows.append(r)
            nfile += 1
            if nfile % 500 == 0:
                print(f"  {nfile} files ({(time.perf_counter()-t0)/60:.1f} min)")
    if not rows:
        print("no usable symbol-days."); return
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "aggressor_flow_gate_raw.csv", index=False)
    print("===== AGGRESSOR FLOW: MOMENTUM vs MEAN-REVERSION (day-as-unit) =====")
    print("  momentum score = avg forward move IN THE FLOW DIRECTION.")
    print("  +ve = MOMENTUM (price continues; pull the exposed side).")
    print("  -ve = MEAN REVERSION (spike snaps back; LEAN INTO the flow).\n")
    def line(sub, label):
        vals = []
        for h in HORIZONS_MS:
            m, se, n = _t(sub[f"mom_{h}"].to_numpy())
            vals.append(f"{h//1000:>2d}s:{m:+6.2f}+/-{se:4.2f}")
        return f"   {label:22s} " + "   ".join(vals)
    for b in BUCKETS:
        print(f"  --- {b} ---")
        obi = df[(df.bucket == b) & (df.kind == "obi")]
        if len(obi):
            print(line(obi, "OBI (ref)"))
        for N in N_STDS:
            fs = df[(df.bucket == b) & (df.kind == "flow") & (df.N == N)]
            if len(fs):
                print(line(fs, f"flow |z|>={N}"))
    print("\n  READ each row left->right (1s,5s,30s): a sign FLIP (+ then -) = short")
    print("  momentum then reversion. All + = momentum (defensive throttle).")
    print("  All - = pure reversion (lean-in opportunity, not a throttle).")
    print(f"\n[flow-gate] outputs -> {out_dir}")

# ===========================================================================
# SELF-TEST: inject flow that predicts markout, OBI that doesn't -> gate finds it
# ===========================================================================
def self_test():
    rng = np.random.default_rng(0)
    n = 8000
    T0 = 0; T1 = int(5.5 * 3600 * 1000)
    ts = np.sort(rng.uniform(T0, T1, n))
    z = rng.normal(0, 1, n)
    flow = z + rng.normal(0, 0.4, n)                 # unbounded (real EMA), |z|>=2 reachable
    obi = rng.normal(0, 1, n)
    # --- MOMENTUM world: price moves WITH the flow ---
    m_mom = 3.0 * z + rng.normal(0, 3.0, n)
    dfm = pd.DataFrame({"ts_exch": ts, "ewma_trade_flow": flow, "obi_1": obi,
                        "markout_1000ms_bps": m_mom, "markout_5000ms_bps": m_mom,
                        "markout_30000ms_bps": m_mom})
    r = pd.DataFrame(symbolday(dfm))
    mm = r[(r.kind == "flow") & (r.N == 2.0)]["mom_5000"].dropna().mean()
    print(f"[self-test] MOMENTUM world: flow momentum @|z|>=2 = {mm:+.3f}  (expect clearly +)")
    assert mm > 0.5, "should read momentum (+) when price moves with flow"
    # --- REVERSION world: price moves AGAINST the flow ---
    m_rev = -3.0 * z + rng.normal(0, 3.0, n)
    dfr = dfm.copy()
    for h in ("1000", "5000", "30000"):
        dfr[f"markout_{h}ms_bps"] = m_rev
    rr = pd.DataFrame(symbolday(dfr))
    mr = rr[(rr.kind == "flow") & (rr.N == 2.0)]["mom_5000"].dropna().mean()
    print(f"[self-test] REVERSION world: flow momentum @|z|>=2 = {mr:+.3f}  (expect clearly -)")
    assert mr < -0.5, "should read reversion (-) when price moves against flow"
    # --- NULL: move unrelated to flow -> ~0 ---
    dfn = dfm.copy()
    mn = rng.normal(0, 3.0, n)
    for h in ("1000", "5000", "30000"):
        dfn[f"markout_{h}ms_bps"] = mn
    rn = pd.DataFrame(symbolday(dfn))
    m0 = rn[(rn.kind == "flow") & (rn.N == 2.0)]["mom_5000"].dropna().mean()
    print(f"[self-test] NULL world: flow momentum @|z|>=2 = {m0:+.3f}  (expect ~0)")
    assert abs(m0) < 0.5, "unrelated move should read ~0"
    b = _bucket(np.array([T0+1, T0+16*60000, T1-46*60000, T1-40*60000, T1-1]), T0, T1)
    assert list(b) == ["first15", "middle", "middle", "preclose45", "last15"], f"bucketer wrong: {list(b)}"
    print("[self-test] ALL ASSERTIONS PASSED.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aggressor trade-flow gate (fast, feature-store).")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    if args.self_test or not args.run:
        self_test()
    if args.run:
        run_real(symbols=args.symbols)
