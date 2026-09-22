# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# calibrate_kappa.py -- fit the Avellaneda-Stoikov fill-intensity decay per
# symbol, so the currently-INERT base half-spread term in micro_mm.py can be
# activated with a data-derived kappa instead of the placeholder 1.5.
#
# =========================== DATA-QUALITY GATE ===============================
# THIS CALIBRATION REQUIRES A TRADE FEED THAT RELIABLY IDENTIFIES THE RESTING
# ORDER EACH FILL CONSUMED (trades.resting_order_id). On PSX it does NOT: only
# ~24% of trades carry a resolvable resting_order_id, and the unresolvable 76%
# are concentrated AT THE TOUCH (fast near-touch fills use the anonymized
# matching path). The result is an INVERTED fill-rate-vs-distance curve (fills
# appear to RISE with distance: ~7% at 1bp -> ~22% at 15-20bp on PPL), which
# yields a NEGATIVE kappa and a nan base spread. That is a DATA limitation, not
# a code bug -- kappa is not cleanly measurable on PSX.
#
# BEFORE TRUSTING kappa ON ANY EXCHANGE, run diag_kappa.py first:
#   * resolution rate should be HIGH (well above ~50%),
#   * fill RATE by distance bin should DECREASE with distance.
# If it looks like PSX (low resolution, inverted curve), leave as_base_weight=0
# (its current value) -- the base term simply cannot be calibrated on that data,
# and the working spread (risk + adverse-selection + floor) is unaffected.
# ============================================================================
#
# THE MODEL (Avellaneda & Stoikov 2008): the probability that a passive order
# resting at distance delta from the mid gets filled decays exponentially with
# distance:   lambda(delta) = A * exp(-kappa * delta)
# The optimal base half-spread is then (1/gamma)*ln(1 + gamma/kappa). micro_mm
# already COMPUTES that term; it is weighted 0 (as_base_weight=0) because kappa
# was never fit. This script fits (A, kappa) per symbol from real resting-order
# outcomes.
#
# METHOD (event-accurate, uses ob_updates order lifecycles):
#   1. For each resting limit order, measure its distance-from-mid at placement
#      (delta, in ticks or bps) and whether it was FILLED (a trade consumed it)
#      vs CANCELLED/expired.
#   2. Bin by delta; the empirical fill INTENSITY per bin is
#      fills / (orders * exposure_time) -- fills per resting-order-second.
#   3. Fit ln(intensity) = ln(A) - kappa*delta by OLS. kappa is the decay slope.
#
# HONEST CAVEAT (the double-counting question): micro_mm's spread already covers
# risk + adverse-selection + a cost floor, and it works (Sharpe 5.67). The A-S
# base term is a SEPARATE, top-down way of sizing the same cushion. Activating it
# (as_base_weight>0) may be additive, redundant, or harmful (double-charging).
# This script only DERIVES kappa; whether to switch the term on is an A/B test
# (see the note at the end) -- do not assume activation helps.
#
# Run from existing_mm_live/:  python3 calibrate_kappa.py

# paths
from pathlib import Path
# math
import numpy as np
import pandas as pd
# driver (dates, readers)
import run_legacy_mm as R

# store + results
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS = Path(str(_hft_paths.RESULTS_ROOT))

# ------------------------------ config ---------------------------------------
# symbols to calibrate
SYMBOLS = ["PPL", "UBL", "MLCF", "TRG", "BOP"]
# how many days to use (kappa is stable; a sample suffices)
N_DAYS = 210
# distance bins from the mid, in BPS (passive orders rest within a few bps)
DELTA_BINS_BPS = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20])
# minimum orders in a bin to trust its intensity
MIN_ORDERS_PER_BIN = 30
# -----------------------------------------------------------------------------


# calibrate one symbol: return (A, kappa, n_orders, r2) or None
def calibrate_symbol(sym, dates):
    # per-order records: (delta_bps, filled?, exposure_ms)
    deltas = []
    filled = []
    exposure = []
    # walk the sampled days
    for date in dates:
        # open datasets
        dsets = R.open_datasets(date)
        # skip missing
        if dsets is None:
            continue
        # order-level updates: contains ORDER_ADD (placement) + CANCEL rows
        u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
        # trades: a FILL lives HERE (event TRADE), keyed by resting_order_id --
        # NOT in ob_updates. This is the schema fact the first version got wrong.
        t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
        # snapshots (for the mid at placement time)
        s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
        # need updates + snapshots at minimum
        if len(u) == 0 or len(s) == 0:
            continue
        # a mid time series from the snapshots (best bid/ask midpoint)
        mids = _mid_series(s)
        # unusable book
        if mids is None:
            continue
        # reconstruct each order's outcome from the REAL schema
        recs = _order_lifecycles(u, t, mids)
        # accumulate
        for d_bps, was_filled, exp_ms in recs:
            # keep only sensible passive distances with positive exposure
            if 0 < d_bps <= DELTA_BINS_BPS[-1] and exp_ms > 0:
                deltas.append(d_bps)
                filled.append(1.0 if was_filled else 0.0)
                exposure.append(exp_ms / 1000.0)
    # nothing usable
    if len(deltas) < MIN_ORDERS_PER_BIN:
        return None
    # arrays
    deltas = np.array(deltas)
    filled = np.array(filled)
    exposure = np.array(exposure)
    # bin by distance and compute empirical fill INTENSITY per bin.
    # SATURATION CAVEAT (verified on synthetic data): near the touch, fill
    # probability can approach 1, so fills/exposure UNDER-estimates the true
    # intensity there and biases kappa UPWARD (a too-steep decay). Using fill
    # INTENSITY (fills per resting-second) rather than fill RATE mitigates this,
    # but if fit_r2 is high yet the near-touch points visibly bend below the
    # log-linear line, drop the most-saturated near-touch bin(s) and refit. On a
    # new exchange, always eyeball the ln(intensity)-vs-delta points before
    # trusting kappa.
    bin_centers = []
    intensities = []
    # each bin [lo, hi)
    for lo, hi in zip(DELTA_BINS_BPS[:-1], DELTA_BINS_BPS[1:]):
        # orders whose placement distance falls in this bin
        m = (deltas >= lo) & (deltas < hi)
        # need enough orders to trust the estimate
        if m.sum() < MIN_ORDERS_PER_BIN:
            continue
        # intensity = fills per resting-order-second in this bin
        total_exposure = exposure[m].sum()
        # guard
        if total_exposure <= 0:
            continue
        # empirical lambda for the bin
        lam = filled[m].sum() / total_exposure
        # need a positive intensity to take a log
        if lam <= 0:
            continue
        # bin center (midpoint) and its intensity
        bin_centers.append(0.5 * (lo + hi))
        intensities.append(lam)
    # need at least 3 bins for a slope
    if len(bin_centers) < 3:
        return None
    # arrays
    x = np.array(bin_centers)
    # ln(lambda) = ln(A) - kappa * delta  -> linear in delta
    ylog = np.log(np.array(intensities))
    # OLS fit
    A_design = np.column_stack([np.ones(len(x)), x])
    # solve
    coef, *_ = np.linalg.lstsq(A_design, ylog, rcond=None)
    # ln(A) intercept, -kappa slope
    lnA, neg_kappa = coef[0], coef[1]
    # recover parameters
    A = float(np.exp(lnA))
    kappa = float(-neg_kappa)
    # fit quality (R^2 of the log-linear fit)
    yhat = A_design @ coef
    ss_res = np.sum((ylog - yhat) ** 2)
    ss_tot = np.sum((ylog - ylog.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # (A, kappa, order count, fit R^2)
    return A, kappa, len(deltas), r2


# a best-bid/ask mid series (ms, mid) from snapshots
def _mid_series(s):
    # continuous rows only
    c = s[s["phase"] == "CONTINUOUS_AUCTION"]
    # empty
    if len(c) == 0:
        return None
    # per message, best bid + best ask
    # (reuse a simple groupby on msg_seq for the touch)
    bids = c[c["entry_type"] == "BID"]
    offs = c[c["entry_type"] == "OFFER"]
    # need both sides
    if len(bids) == 0 or len(offs) == 0:
        return None
    # best bid per message (max price)
    bb = bids.groupby("msg_seq")["px"].max()
    # best ask per message (min price)
    ba = offs.groupby("msg_seq")["px"].min()
    # align on msg_seq
    touch = pd.DataFrame({"bb": bb, "ba": ba}).dropna()
    # message time per msg_seq (first)
    mt = c.groupby("msg_seq")["orig_time"].first()
    # mid + time, sorted by time
    touch["t_ms"] = R.to_ms(mt.reindex(touch.index)).to_numpy()
    # mid
    touch["mid"] = 0.5 * (touch["bb"] + touch["ba"])
    # sorted arrays
    touch = touch.sort_values("t_ms")
    # (times, mids)
    return touch["t_ms"].to_numpy(), touch["mid"].to_numpy()


# reconstruct order outcomes -> list of (delta_bps_at_placement, filled?, exposure_ms)
# REAL SCHEMA: placements are ob_updates ORDER_ADD rows (order_id, price, time);
# a FILL is recorded in the TRADES table with the resting order's id in
# resting_order_id (same namespace as ob_updates.order_id). CANCEL rows in
# ob_updates carry order_id=None, so we identify fills via the trades join, and
# treat every added order NOT seen as a resting fill as unfilled (cancelled).
def _order_lifecycles(u, t, mids):
    # unpack the mid series (times, mids)
    tmid, midv = mids
    # ORDER_ADD rows only = placements
    adds = u[u["event"] == "ORDER_ADD"].copy()
    # nothing placed
    if len(adds) == 0:
        return []
    # placement times in ms
    adds["t_ms"] = R.to_ms(adds["transact_time"])
    # ---- which placed orders were FILLED? (from the trades table) ----
    # resting order id -> the FIRST fill time for it (ms)
    fill_time = {}
    # only if we have trades with a resting id
    if t is not None and len(t) > 0 and "resting_order_id" in t.columns:
        # resolve the resting order id (shares ob_updates.order_id namespace)
        t = t.copy()
        # parse each resting_order_id into a bare order id
        t["rest_oid"] = t["resting_order_id"].map(R.parse_rest_oid)
        # trade times in ms
        t["t_ms"] = R.to_ms(t["transact_time"])
        # earliest fill time per resting order id
        for oid_, tt in zip(t["rest_oid"].to_numpy(), t["t_ms"].to_numpy()):
            # skip unresolved ids
            if oid_ is None:
                continue
            # keep the earliest fill timestamp for this resting order
            if oid_ not in fill_time or tt < fill_time[oid_]:
                fill_time[oid_] = tt
    # mid at a given time (asof: last mid <= t)
    def mid_at(tt):
        # binary search for the last mid at/before tt
        pos = np.searchsorted(tmid, tt, side="right") - 1
        # none before -> no mid
        if pos < 0:
            return None
        # the mid value
        return midv[pos]
    # a censoring horizon for UNFILLED orders: the session end (last mid time),
    # so an unfilled order's "exposure" is placement -> end of observable book
    sess_end = float(tmid[-1]) if len(tmid) else None
    # completed records
    recs = []
    # placement arrays
    oid_arr = adds["order_id"].to_numpy()
    px_arr = adds["price"].to_numpy(dtype=float)
    tms_arr = adds["t_ms"].to_numpy()
    # walk each placement
    for i in range(len(adds)):
        # this order's id
        k = oid_arr[i]
        # skip rows with no id or no price
        if k is None or not np.isfinite(px_arr[i]):
            continue
        # mid at placement
        m = mid_at(float(tms_arr[i]))
        # need a positive mid
        if m is None or m <= 0:
            continue
        # placement distance from mid, in bps
        d_bps = abs(px_arr[i] - m) / m * 1e4
        # was this order filled? (its id appears as a resting order in a trade)
        if k in fill_time:
            # filled: exposure = placement -> first fill
            exp_ms = fill_time[k] - float(tms_arr[i])
            # record as a fill (guard non-positive exposure)
            if exp_ms > 0:
                recs.append((d_bps, True, exp_ms))
        else:
            # unfilled (cancelled/expired): exposure = placement -> session end
            if sess_end is not None:
                # censoring exposure
                exp_ms = sess_end - float(tms_arr[i])
                # record as not filled
                if exp_ms > 0:
                    recs.append((d_bps, False, exp_ms))
    # the outcome records
    return recs


def main():
    # dates to sample
    dates = [str(d) for d in R.discover_dates()[:N_DAYS]]
    # per-symbol result rows
    rows = []
    # announce
    print(f"calibrating kappa on {len(dates)} days, {len(SYMBOLS)} symbols\n")
    # walk symbols
    for sym in SYMBOLS:
        # calibrate
        res = calibrate_symbol(sym, dates)
        # skip if it failed
        if res is None:
            print(f"{sym}: insufficient data -- no fit")
            continue
        # unpack
        A, kappa, n, r2 = res
        # the A-S base half-spread this kappa implies (fraction of price), at gamma=0.15
        gamma = 0.15
        # (1/gamma)*ln(1+gamma/kappa) -- the term micro_mm computes
        as_base = (1.0 / gamma) * np.log(1.0 + gamma / kappa) if kappa > 0 else np.nan
        # record
        rows.append({"symbol": sym, "A": round(A, 4), "kappa": round(kappa, 4),
                     "n_orders": n, "fit_r2": round(r2, 3),
                     "as_base_frac": round(as_base, 6)})
        # progress
        print(f"{sym}: kappa={kappa:.3f}  A={A:.3f}  n={n:,}  fit_R2={r2:.2f}  "
              f"implied A-S base={as_base:.5f} of price")
    # save
    res_df = pd.DataFrame(rows)
    # output path
    out = RESULTS / "kappa_calibration.csv"
    # write
    res_df.to_csv(out, index=False)
    # ---- how to USE this (and the A/B caveat) ----
    print("\n=== USING THESE KAPPAS ===")
    print("Pass the per-symbol kappa into MicrostructureMM(kappa=...), and raise")
    print("as_base_weight ABOVE 0 to activate the base term. Then A/B it:")
    print("  * baseline: as_base_weight=0 (current, Sharpe 5.67)")
    print("  * candidate: as_base_weight in {0.25, 0.5, 1.0} with the fitted kappa")
    print("Compare net PnL and fill ratio. If activation only WIDENS the spread")
    print("and cuts fills without raising PnL, the base term is double-counting")
    print("the risk+adverse+floor cushion already in place -- leave it at 0.")
    print("Only adopt as_base_weight>0 if it measurably improves out-of-sample PnL.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
