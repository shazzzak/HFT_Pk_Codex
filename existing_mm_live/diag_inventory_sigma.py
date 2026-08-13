# diag_inventory_sigma.py -- confirm micro's inventory-carry problem and calibrate
# the session_scale skew fix, per symbol, from MEASURED sigma.
#
# Two purposes:
#   (1) CONFIRM the mechanism: is micro carrying large, LOSING inventory into the
#       close? Reports EOD-position distribution + how much of micro's negative
#       P&L is the liquidation mark vs the intraday fills. If EOD positions are big
#       and the liquidation mark is the bleed, the inert inventory skew is proven
#       to be the cause (per-fill capture is fine; carry kills it).
#   (2) CALIBRATE session_scale: measure the per-event fractional-return sigma per
#       symbol, then back-solve the session_scale that makes the A-S inventory skew
#       hit a SPREAD-AWARE target at max inventory (e.g. skew = 1x the median spread,
#       not a round-number 15 ticks). Prints the derived session_scale per symbol.
#
# Runs micro at the sweep-winner 0.0005/0.0 (current broken skew) so the EOD-position
# distribution reflects the strategy we are trying to fix.
#
# Heartbeat + timer built in. Run from existing_mm_live/:  python diag_inventory_sigma.py

# paths
from pathlib import Path
# timing
import time
# frames + arrays
import pandas as pd
import numpy as np
# driver + engine + micro
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel
from micro_mm import MicrostructureMM

# raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# feature store (for sigma + spread measurement)
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")

# pilot symbols
SYMBOLS = ["PPL", "UBL"]
# the config whose inventory behaviour we are diagnosing (sweep winner)
MICRO_CFG = {"min_edge_pct": 0.0005, "improve_ticks": 0.0}
# micro's baseline lot size + inventory cap (from MICRO_PARAMS) for skew calibration
SIZE0 = R.MICRO_PARAMS["size"]
MAX_INV = R.MICRO_PARAMS["max_inv"]
# risk aversion (for the back-solve; same gamma micro uses)
GAMMA = R.MICRO_PARAMS["gamma"]
# how many lots at max inventory (pos_lots at the cap)
MAX_LOTS = MAX_INV / SIZE0
# target skew at max inventory, expressed as a MULTIPLE of the symbol's median
# spread (spread-aware, not a round tick count). 1.0 = shift a full spread's worth.
TARGET_SKEW_IN_SPREADS = 1.0


# compact mm:ss
def _fmt(sec):
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# run micro for one symbol-day; return (eod_pos, eod_pnl, equity_mid_mark, liq_clean)
def run_micro_day(sym, date, dsets):
    # load tables
    u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
    t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
    # need book + trades
    if len(t) == 0 or len(s) == 0:
        return None
    # build events
    events, snap_groups, t = R.build_events(u, s, t)
    # session window
    cont = t[t["initiator"] != "AUCTION"]
    if len(cont) == 0:
        return None
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
    # cfg with seeded latency
    cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED))
    # micro at the diagnosed config
    params = dict(R.MICRO_PARAMS); params.update(MICRO_CFG)
    bt = Backtester(MicrostructureMM(session_ms=(t0, t1), **params), cfg)
    # run
    fills, equity, stats = bt.run(events, snap_groups)
    # need an eod report to read inventory outcome
    if bt.eod is None:
        return None
    # extract the EOD accounting
    return {
        # position carried into the close (signed shares)
        "eod_pos": float(bt.eod["pos_at_close"]),
        # true P&L with book-walk liquidation
        "eod_pnl": float(bt.eod["equity_liquidated"]),
        # the OPTIMISTIC mid-mark P&L (ignores liquidation cost) -- None if one-sided
        "mid_mark": bt.eod["equity_mid_mark"],
        # whether liquidation was clean
        "clean": bool(bt.eod["liquidation_clean"]),
    }


# measure per-symbol sigma (per-event fractional-return vol) + median spread from
# the feature store, sampling a subset of days for speed.
def measure_sigma_spread(sym, dates, stride=7):
    # collectors
    sigmas, spreads, fairs = [], [], []
    # sample days
    for date in dates[::stride]:
        # feature-store partition
        fp = FS_ROOT / sym / f"date={date}.parquet"
        # skip missing
        if not fp.exists():
            continue
        # read mid + spread (mid for return-vol, spread for the target)
        d = pd.read_parquet(fp, columns=["mid", "spread_bps"])
        # need enough rows
        if len(d) < 100:
            continue
        # per-event fractional returns of the mid
        rets = d["mid"].pct_change().dropna()
        # per-event return volatility (the strategy's self.sigma analogue)
        sigmas.append(rets.std())
        # median spread in bps (for the spread-aware skew target)
        spreads.append(d["spread_bps"].median())
        # median fair price (for PKR conversion)
        fairs.append(d["mid"].median())
    # aggregate across sampled days (median = robust central estimate)
    return {
        "sigma": float(np.median(sigmas)) if sigmas else np.nan,
        "spread_bps": float(np.median(spreads)) if spreads else np.nan,
        "fair": float(np.median(fairs)) if fairs else np.nan,
    }


# back-solve session_scale so that at MAX inventory the A-S skew equals the target.
# skew formula (from the vetted fix):  skew = gamma * (sigma*fair)^2 * session_scale * tau * pos_lots
# at open tau=1, pos_lots=MAX_LOTS. Solve for session_scale given target skew in PKR.
def solve_session_scale(sigma, fair, spread_bps):
    # price volatility per event in PKR
    sigma_p = sigma * fair
    # per-event price variance (PKR^2)
    var_p = sigma_p ** 2
    # target skew in PKR = TARGET_SKEW_IN_SPREADS * (median spread in PKR)
    # median spread in PKR = spread_bps/1e4 * fair
    spread_pkr = spread_bps / 1e4 * fair
    # the desired skew magnitude at max inventory
    target_skew_pkr = TARGET_SKEW_IN_SPREADS * spread_pkr
    # skew at max inv = gamma * var_p * session_scale * 1.0 * MAX_LOTS
    # => session_scale = target_skew_pkr / (gamma * var_p * MAX_LOTS)
    denom = GAMMA * var_p * MAX_LOTS
    # guard divide-by-zero
    session_scale = target_skew_pkr / denom if denom > 0 else np.nan
    # also report what the target skew is in TICKS (fair/... ) for sanity
    tick = R.MICRO_PARAMS["tick"]
    target_skew_ticks = target_skew_pkr / tick
    return {
        "sigma_p_pkr": sigma_p,
        "spread_pkr": spread_pkr,
        "target_skew_pkr": target_skew_pkr,
        "target_skew_ticks": target_skew_ticks,
        "session_scale": session_scale,
    }


# main
def main():
    # all dates
    dates = R.discover_dates()
    # timers
    t0_all = time.perf_counter()
    # per symbol
    for sym in SYMBOLS:
        # ---- Part 1: EOD-position distribution (confirm the carry mechanism) ----
        # header
        print(f"\n=== {sym}: EOD inventory diagnosis (micro {MICRO_CFG}) ===", flush=True)
        # collectors
        eod_positions, eod_pnls, mid_marks, unclean = [], [], [], 0
        # symbol-day counter
        n = 0
        # walk all dates
        for date in dates:
            # open datasets
            dsets = R.open_datasets(date)
            if dsets is None:
                continue
            # run micro for the day
            res = run_micro_day(sym, date, dsets)
            if res is None:
                continue
            # collect
            eod_positions.append(res["eod_pos"])
            eod_pnls.append(res["eod_pnl"])
            # mid-mark may be None on one-sided closes
            if res["mid_mark"] is not None:
                mid_marks.append(res["mid_mark"])
            if not res["clean"]:
                unclean += 1
            # heartbeat every 50 days
            n += 1
            if n % 50 == 0:
                el = time.perf_counter() - t0_all
                print(f"  {sym}: {n} days processed, elapsed {_fmt(el)}", flush=True)
        # arrays
        pos = np.array(eod_positions)
        pnl = np.array(eod_pnls)
        # position distribution
        print(f"  EOD position (shares): mean={pos.mean():.0f}  std={pos.std():.0f}  "
              f"min={pos.min():.0f}  max={pos.max():.0f}", flush=True)
        # how often is EOD position near the inventory cap (|pos| > 50% of max_inv)?
        near_cap = (np.abs(pos) > 0.5 * MAX_INV).mean()
        print(f"  fraction of days ending with |pos| > 50% of cap ({0.5*MAX_INV:.0f}): "
              f"{near_cap:.1%}", flush=True)
        # total liquidated P&L vs what mid-mark WOULD have claimed (the carry cost)
        print(f"  total EOD P&L (liquidated):  {pnl.sum():>14,.0f} PKR", flush=True)
        if mid_marks:
            print(f"  total EOD P&L (mid-mark):    {np.sum(mid_marks):>14,.0f} PKR "
                  f"(optimistic; ignores liquidation cost)", flush=True)
            # the gap = the inventory liquidation cost the per-fill view misses
            gap = np.sum(mid_marks) - pnl.sum()
            print(f"  => liquidation/carry cost:   {gap:>14,.0f} PKR "
                  f"(this is what a WORKING inventory skew targets)", flush=True)
        print(f"  unclean-liquidation days: {unclean}", flush=True)

        # ---- Part 2: sigma + spread measurement and session_scale back-solve ----
        # measure
        ms = measure_sigma_spread(sym, dates)
        # solve
        sol = solve_session_scale(ms["sigma"], ms["fair"], ms["spread_bps"])
        # report
        print(f"  --- session_scale calibration ---", flush=True)
        print(f"  measured sigma (per-event frac return): {ms['sigma']:.2e}", flush=True)
        print(f"  median fair: {ms['fair']:.2f} PKR   median spread: {ms['spread_bps']:.2f} bps "
              f"({sol['spread_pkr']:.4f} PKR)", flush=True)
        print(f"  sigma_p (PKR/event): {sol['sigma_p_pkr']:.5f}", flush=True)
        print(f"  target skew at max inv: {sol['target_skew_pkr']:.4f} PKR "
              f"(= {sol['target_skew_ticks']:.1f} ticks, {TARGET_SKEW_IN_SPREADS}x median spread)", flush=True)
        print(f"  => DERIVED session_scale = {sol['session_scale']:.1f}", flush=True)
    # done
    print(f"\ntotal elapsed {_fmt(time.perf_counter()-t0_all)}", flush=True)
    print("Use the DERIVED session_scale per symbol in the skew fix; verify the "
          "target skew in ticks is sane vs the spread before trusting it.", flush=True)


# entry point
if __name__ == "__main__":
    main()
