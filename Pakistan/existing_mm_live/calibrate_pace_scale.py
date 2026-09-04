# calibrate_pace_scale.py -- back-solve PACE's session_scale so the inventory skew
# at MAX inventory ~ 1x PACE's median spread. EXACT sigma: we run PACE through the
# REAL MicrostructureMM.observe() (subclassed only to record self.sigma each event),
# so the vol is byte-identical to what the strategy uses live -- no proxy. Spread and
# price come from PACE's feature store (no sigma-style ambiguity there).
#
# session_scale = median_spread_pkr / (gamma * (sigma*fair)^2 * tau * pos_lots_max)
#   tau=1 (worst case, the open);  pos_lots_max = max_inv/size = 10.
#
# Run from existing_mm_live/:  python calibrate_pace_scale.py

# filesystem paths
from pathlib import Path
# timing
import time
# numeric median
import numpy as np
# driver (datasets, reads, event building, params)
import run_legacy_mm as R
# the engine + latency model (exact code path for sigma)
from mm_backtest import Backtester, LatencyModel
# the real strategy we subclass to record sigma
from micro_mm import MicrostructureMM

# raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# PACE feature-store glob (median spread + price come from here)
PACE_FS_GLOB = "/Users/shazzak/Capital Stake - Results/feature_store/PACE/*.parquet"
# the symbol we're calibrating
SYM = "PACE"
# how many days to sample for the sigma median (sigma is a fast EMA -> stable;
# we spread the sample across the whole range to span Ramadan/other regimes)
N_CAL_DAYS = 30
# worst-case horizon for the target (skew is largest at the open, tau=1)
TAU = 1.0


# subclass that records (ts, sigma) after every real observe() call -- the ONLY
# change from the live strategy; sigma itself is computed by the parent, unchanged.
class SigmaRecorder(MicrostructureMM):
    # extend __init__ to add a samples list
    def __init__(self, *args, **kwargs):
        # build the real strategy exactly as usual
        super().__init__(*args, **kwargs)
        # storage for (ts_exch, sigma) samples
        self._sig = []

    # wrap observe: run the real one, then record the resulting sigma
    def observe(self, kind, obj, ts_exch, mid):
        # exact live behaviour
        super().observe(kind, obj, ts_exch, mid)
        # record only when a two-sided mid existed (mid is not None) so we sample
        # the quotable regime, matching where the strategy actually uses sigma
        if mid is not None:
            self._sig.append((ts_exch, self.sigma))


# compact mm:ss
def _fmt(sec):
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# median spread (PKR) and price (fair) from PACE's feature store, via DuckDB
def spread_and_price():
    # import here so the script still loads if duckdb is missing (we fall back)
    try:
        import duckdb
        # median PKR spread computed per-row (spread_bps * mid / 1e4), and median mid.
        # filter mid>0 and spread_bps>0 to drop pre-open/auction contamination.
        q = f"""
            SELECT median(mid) AS fair_ref,
                   median(spread_bps * mid / 10000.0) AS med_spread_pkr
            FROM read_parquet('{PACE_FS_GLOB}')
            WHERE mid > 0 AND spread_bps > 0
        """
        # run and pull the single row
        row = duckdb.sql(q).df().iloc[0]
        # return the two references
        return float(row["fair_ref"]), float(row["med_spread_pkr"])
    # pandas fallback if duckdb isn't available
    except Exception as e:
        # note the fallback
        print(f"(duckdb unavailable: {e}; using pandas over the feature store)")
        # pandas + glob
        import pandas as pd, glob
        # gather all PACE feature files
        files = sorted(glob.glob(PACE_FS_GLOB))
        # read just the columns we need across all days
        d = pd.concat([pd.read_parquet(f, columns=["mid", "spread_bps"]) for f in files],
                      ignore_index=True)
        # clean rows
        d = d[(d["mid"] > 0) & (d["spread_bps"] > 0)]
        # medians
        return float(d["mid"].median()), float((d["spread_bps"] * d["mid"] / 1e4).median())


# collect exact sigma samples by replaying PACE through the real observe()
def sigma_samples():
    # all dates
    dates = R.discover_dates()
    # evenly spaced sample across the full range (spans regimes, not just the start)
    step = max(1, len(dates) // N_CAL_DAYS)
    # the sampled dates
    sample = dates[::step][:N_CAL_DAYS]
    # collected sigma values (steady-state only)
    sigs = []
    # timer
    t0_all = time.perf_counter()
    # progress counter
    done = 0
    # announce
    print(f"sigma replay: {len(sample)} PACE days (exact observe())\n", flush=True)
    # loop the sampled days
    for date in sample:
        # open datasets
        dsets = R.open_datasets(date)
        # skip missing
        if dsets is None:
            continue
        # read PACE's three tables
        u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, SYM)
        s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, SYM)
        t = R.read_symbol(dsets["trades"], R.REQ_TRADES, SYM)
        # need book + trades
        if len(t) == 0 or len(s) == 0:
            continue
        # build events (adds ts_exch to s)
        events, snap_groups, t = R.build_events(u, s, t)
        # PHASE-BASED continuous window (same fix as confirm/run_legacy)
        cont_snap = s[s["phase"] == "CONTINUOUS_AUCTION"]
        # skip days with no continuous phase
        if len(cont_snap) == 0:
            continue
        # continuous open/close
        t0, t1 = int(cont_snap["ts_exch"].min()), int(cont_snap["ts_exch"].max())
        # strategy params: real MICRO_PARAMS + placeholder scale (sigma is
        # independent of it) + mid-based fair (sigma independent of that too)
        params = dict(R.MICRO_PARAMS)
        # placeholder scale just to construct; does NOT affect sigma
        params["session_scale"] = 1.0
        # mid-based (matches the MID baseline; irrelevant to sigma)
        params["use_microprice"] = False
        # build the recording strategy over the continuous window
        strat = SigmaRecorder(session_ms=(t0, t1), **params)
        # config with the production latency model, same as confirm
        cfg = dict(R.CFG, session=(t0, t1),
                   latency_model=LatencyModel(seed=R.LATENCY_SEED))
        # run the real backtester so the book is maintained exactly
        bt = Backtester(strat, cfg)
        # execute (fills/latency are wasted work here but keep the path exact)
        bt.run(events, snap_groups)
        # keep sigma samples inside the continuous window with sigma>0 (warmed up)
        for ts, sg in strat._sig:
            if t0 <= ts <= t1 and sg > 0.0:
                sigs.append(sg)
        # progress
        done += 1
        if done % 5 == 0:
            el = time.perf_counter() - t0_all
            print(f"  {done}/{len(sample)} days  elapsed {_fmt(el)}", flush=True)
    # median of all collected sigma values
    return float(np.median(sigs)) if sigs else None


def main():
    # strategy constants from MICRO_PARAMS (single source of truth)
    gamma = float(R.MICRO_PARAMS.get("gamma", 0.15))
    # base clip size
    size0 = float(R.MICRO_PARAMS.get("size", 50))
    # hard inventory cap
    max_inv = float(R.MICRO_PARAMS.get("max_inv", 500))
    # inventory in lots at the cap
    pos_lots_max = max_inv / size0

    # 1) spread + price from the feature store
    fair_ref, med_spread_pkr = spread_and_price()
    # 2) exact sigma from the real observe() replay
    sigma_ref = sigma_samples()
    # guard: no sigma collected
    if sigma_ref is None:
        raise SystemExit("no sigma samples collected for PACE")

    # PKR price vol
    sigma_p = sigma_ref * fair_ref
    # back-solve: session_scale so skew at max inventory == 1x median spread
    session_scale = med_spread_pkr / (gamma * (sigma_p ** 2) * TAU * pos_lots_max)

    # report the inputs and the answer
    print("\n--- PACE session_scale back-solve (exact sigma) ---")
    print(f"  fair_ref (median mid)      : {fair_ref:.4f} PKR")
    print(f"  median spread              : {med_spread_pkr:.4f} PKR")
    print(f"  sigma_ref (exact EMA, med) : {sigma_ref:.3e} (fractional)")
    print(f"  sigma_p = sigma*fair       : {sigma_p:.5f} PKR")
    print(f"  gamma / pos_lots_max / tau : {gamma} / {pos_lots_max:.0f} / {TAU}")
    print(f"\n  derived session_scale      : {session_scale:.4f}")
    # the sweep grid to actually run in confirm (don't trust the point estimate)
    grid = [round(session_scale * m, 4) for m in (0.25, 0.5, 1.0, 2.0)]
    print(f"  sweep grid {{0.25,0.5,1,2}}x : {grid}")
    # sanity check vs the two known symbols (PPL~7.6, UBL~3.9) for a smell test
    print("\n  (smell test: PPL back-solved to ~7.6, UBL ~3.9 -- PACE should land in a")
    print("   plausible range given its wider spread; if it's wildly off, inspect inputs.)")


if __name__ == "__main__":
    main()
