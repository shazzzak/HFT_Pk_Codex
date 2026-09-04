# calibrate_all_scales.py -- back-solve session_scale for EVERY shortlist name,
# using the EXACT rule proven on PACE (calibrate_pace_scale.py):
#
#   session_scale = median_spread_pkr / (gamma * (sigma*fair)^2 * tau * pos_lots_max)
#     tau = 1 (worst case, the open);  pos_lots_max = max_inv/size = 10
#
# sigma is the REAL EMA vol from replaying each name through MicrostructureMM.observe()
# (subclassed only to record self.sigma) -- byte-identical to the live path, no proxy.
# spread + price come from each name's feature store (DuckDB).
#
# Writes a cached table (symbol -> session_scale + the inputs) so the 38-name
# universe run reads calibrated scales instead of crashing on the [] lookup.
# Output filename carries the run timestamp (standing convention).
#
# Run from existing_mm_live/:  python calibrate_all_scales.py

# filesystem paths
from pathlib import Path
# timing + stamp
import time
from datetime import datetime
# numeric
import numpy as np
# frames (for the output table + watchlist read)
import pandas as pd
# driver (datasets, reads, event building, params)
import run_legacy_mm as R
# engine + latency (exact sigma path)
from mm_backtest import Backtester, LatencyModel
# the real strategy, subclassed to record sigma
from micro_mm import MicrostructureMM

# raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# feature-store root (per-name spread + price)
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
# the shortlist (symbol column)
WATCHLIST = Path("/Users/shazzak/Capital Stake - Results/mm_watchlist_final.csv")
# where the scale table is written (timestamped)
OUT_DIR = Path("/Users/shazzak/Capital Stake - Results")
# days sampled per name for the sigma median (sigma is a fast EMA -> stable;
# spread the sample across the whole range to span regimes). Same as PACE used.
N_CAL_DAYS = 30
# worst-case horizon for the skew target (largest at the open)
TAU = 1.0


# the sigma-recording subclass: identical to the live strategy, records sigma
class SigmaRecorder(MicrostructureMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (ts, sigma) samples in the quotable regime
        self._sig = []

    def observe(self, kind, obj, ts_exch, mid):
        # exact live behaviour
        super().observe(kind, obj, ts_exch, mid)
        # sample sigma only where a two-sided mid existed (matches usage)
        if mid is not None:
            self._sig.append((ts_exch, self.sigma))


# compact mm:ss
def _fmt(sec):
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# median spread (PKR) and median price for one symbol, from its feature store
def spread_and_price(sym):
    # this name's feature files
    glob = f"{FS_ROOT}/{sym}/date=*.parquet"
    try:
        # DuckDB path (fast, queries parquet in place)
        import duckdb
        # per-row PKR spread = spread_bps * mid / 1e4; medians over clean rows
        q = f"""
            SELECT median(mid) AS fair_ref,
                   median(spread_bps * mid / 10000.0) AS med_spread_pkr
            FROM read_parquet('{glob}')
            WHERE mid > 0 AND spread_bps > 0
        """
        row = duckdb.sql(q).df().iloc[0]
        return float(row["fair_ref"]), float(row["med_spread_pkr"])
    except Exception:
        # pandas fallback
        import glob as _g
        files = sorted(_g.glob(glob))
        # some layouts use date=.../file.parquet; catch both
        if not files:
            files = sorted(_g.glob(f"{FS_ROOT}/{sym}/date=*/*.parquet"))
        if not files:
            return None, None
        d = pd.concat([pd.read_parquet(f, columns=["mid", "spread_bps"]) for f in files],
                      ignore_index=True)
        d = d[(d["mid"] > 0) & (d["spread_bps"] > 0)]
        if len(d) == 0:
            return None, None
        return float(d["mid"].median()), float((d["spread_bps"] * d["mid"] / 1e4).median())


# median exact-EMA sigma for one symbol, replaying the real observe()
def sigma_median(sym, sample_dates):
    # collected steady-state sigma values
    sigs = []
    # loop the sampled days
    for date in sample_dates:
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # this name's three tables
        u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
        s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
        t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
        if len(t) == 0 or len(s) == 0:
            continue
        # build events (adds ts_exch to s)
        events, snap_groups, t = R.build_events(u, s, t)
        # continuous window (phase-based)
        cont = s[s["phase"] == "CONTINUOUS_AUCTION"]
        if len(cont) == 0:
            continue
        t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
        # real params + placeholder scale (sigma is independent of the scale)
        params = dict(R.MICRO_PARAMS)
        params["session_scale"] = 1.0
        params["use_microprice"] = False
        # the recording strategy over the continuous window
        strat = SigmaRecorder(session_ms=(t0, t1), **params)
        cfg = dict(R.CFG, session=(t0, t1),
                   latency_model=LatencyModel(seed=R.LATENCY_SEED))
        # run the real backtester (keeps the book exact)
        Backtester(strat, cfg).run(events, snap_groups)
        # keep warmed-up sigma inside the continuous window
        for ts, sg in strat._sig:
            if t0 <= ts <= t1 and sg > 0.0:
                sigs.append(sg)
    # median across all sampled days
    return float(np.median(sigs)) if sigs else None


def main():
    # run stamp for the output filename
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # strategy constants (single source of truth)
    gamma = float(R.MICRO_PARAMS.get("gamma", 0.15))
    size0 = float(R.MICRO_PARAMS.get("size", 50))
    max_inv = float(R.MICRO_PARAMS.get("max_inv", 500))
    # inventory in lots at the cap
    pos_lots_max = max_inv / size0

    # the 38 shortlist symbols
    syms = pd.read_csv(WATCHLIST)["symbol"].tolist()
    # evenly-spaced calibration days spanning the full range
    dates = R.discover_dates()
    step = max(1, len(dates) // N_CAL_DAYS)
    sample_dates = dates[::step][:N_CAL_DAYS]
    print(f"calibrate_all_scales: {len(syms)} names x {len(sample_dates)} sampled days\n",
          flush=True)

    # collected rows
    rows = []
    # timer
    t0_all = time.perf_counter()
    # per-name loop
    for i, sym in enumerate(syms, 1):
        # spread + price
        fair_ref, med_spread = spread_and_price(sym)
        # exact sigma
        sigma_ref = sigma_median(sym, sample_dates) if fair_ref is not None else None
        # guard: missing inputs -> record NaN, flag it, do NOT crash the batch
        if fair_ref is None or med_spread is None or sigma_ref is None or sigma_ref <= 0:
            rows.append({"symbol": sym, "session_scale": np.nan, "fair_ref": fair_ref,
                         "med_spread_pkr": med_spread, "sigma_ref": sigma_ref,
                         "note": "MISSING_INPUTS"})
            print(f"  [{i}/{len(syms)}] {sym:8s}  MISSING INPUTS -> NaN", flush=True)
            continue
        # PKR price vol
        sigma_p = sigma_ref * fair_ref
        # back-solve (the exact PACE rule)
        scale = med_spread / (gamma * (sigma_p ** 2) * TAU * pos_lots_max)
        # record
        rows.append({"symbol": sym, "session_scale": round(scale, 4),
                     "fair_ref": round(fair_ref, 4),
                     "med_spread_pkr": round(med_spread, 5),
                     "sigma_ref": sigma_ref, "note": "ok"})
        # progress with ETA
        el = time.perf_counter() - t0_all
        eta = el / i * (len(syms) - i)
        print(f"  [{i}/{len(syms)}] {sym:8s}  scale={scale:>9.4f}  "
              f"(px {fair_ref:.2f}, spr {med_spread:.4f})  "
              f"elapsed {_fmt(el)}  ETA {_fmt(eta)}", flush=True)

    # assemble + write the cached table
    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"session_scales_{stamp}.csv"
    out.to_csv(out_csv, index=False)

    # smell test vs the two anchors we trust
    print("\n--- smell test vs known anchors (PPL~7.6, UBL~3.9, PACE~46.15) ---")
    for anchor in ("PPL", "UBL", "PACE"):
        r = out[out.symbol == anchor]
        if len(r):
            print(f"  {anchor}: {r.iloc[0]['session_scale']}")
    # report any names that failed
    bad = out[out.note != "ok"]
    if len(bad):
        print(f"\n!!! {len(bad)} names missing inputs (excluded / need attention): "
              f"{list(bad.symbol)}")
    # a Python dict block ready to paste into the universe runner, if preferred
    print(f"\nwrote {out_csv}")
    print("\nThe universe runner should read this CSV into SESSION_SCALE_BASE. Also")
    print("printed as a dict for convenience:\n")
    d = {r["symbol"]: r["session_scale"] for _, r in out.iterrows() if r["note"] == "ok"}
    print("SESSION_SCALE_BASE = {")
    for k, v in d.items():
        print(f'    "{k}": {v},')
    print("}")


if __name__ == "__main__":
    main()
