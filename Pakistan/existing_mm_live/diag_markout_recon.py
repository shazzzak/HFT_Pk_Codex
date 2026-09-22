# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# diag_markout_recon.py -- settle three questions raised by the spot capture/
# markout decomposition run:
#
# (1) LEVELS: does the run's per-bucket (capture + markout - fees) match the
#     exact-recon attribution's realized_pnl per bucket? If yes -> the levels
#     are trustworthy; if no -> residual-lot handling is undercounting and the
#     ratios (jump/diff/OBI) are still trustworthy but the absolute PnL isn't.
#
# (2) JUMP-DETECTOR FRACTION: what % of mid events are flagged as jumps at
#     JUMP_K=4? Expected 0.1-1% on liquid names. Near zero would mean the
#     threshold is too high for PSX and the "diffusive dominates" reading is
#     partly an artifact of under-detection.
#
# (3) INVARIANT: does markout == jump_markout + diff_markout on real fills?
#     The user spotted first15 7x showing -231,868 vs (-415 + -234,838) =
#     -235,253 -- off by 3,385 PKR. Confirms the bug and locates it.
#
# Small sample (5 days x 10 names x one clip), no parallelism, finishes fast.

# paths + timing
from pathlib import Path
import time
# arrays + frames
import numpy as np
import pandas as pd
# harness + driver
import mm_harness as H
import run_legacy_mm as R

# store paths
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS = Path(str(_hft_paths.RESULTS_ROOT))

# ------------------------------ config ---------------------------------------
# the top-10 book (same as production)
NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
# base production clip
CLIP_MULT = 3.0
# trailing-median window
TRAIL_DAYS = 10
# smoke sample: enough to check reconciliation but fast
N_DAYS = 5
# jump detector params (same as spot_capture_markout_decomp.py)
JUMP_K = 4.0
LOCAL_VOL_WIN = 100
# -----------------------------------------------------------------------------


# same helpers as the main decomposition
def _mid_at(eq_t, eq_mid, t):
    # empty -> nan
    if len(eq_t) == 0:
        return np.nan
    # last equity row with time <= t
    pos = np.searchsorted(eq_t, t, side="right") - 1
    # nothing before -> nan
    if pos < 0:
        return np.nan
    # the mid
    return float(eq_mid[pos])


# THE SUSPECTED-BUG FUNCTION: as used in spot_capture_markout_decomp.py
# returns (jump_move, diff_move) computed as sum-of-per-event-steps over an
# asof-indexed slice of the mid path -- and independently, the total move
# computed as a single asof-lookup at exit_t minus a single asof-lookup at
# fill_t. We return BOTH so we can measure the gap.
def _split_move_diagnostic(eq_t, eq_mid, fill_t, exit_t):
    # empty / degenerate
    if len(eq_t) == 0 or exit_t <= fill_t:
        return 0.0, 0.0, 0.0
    # anchor indices (as in the main file)
    a = np.searchsorted(eq_t, fill_t, side="right") - 1
    b = np.searchsorted(eq_t, exit_t, side="right") - 1
    # bounds
    a = max(0, a); b = max(a, b)
    # need at least 2 points
    if b <= a:
        # single asof endpoint diff -- the "markout path" answer
        m_open = _mid_at(eq_t, eq_mid, fill_t)
        m_exit = _mid_at(eq_t, eq_mid, exit_t)
        # total move by asof, but no per-event steps to split
        total_asof = (m_exit - m_open) if (np.isfinite(m_open) and
                                            np.isfinite(m_exit)) else 0.0
        return 0.0, 0.0, total_asof
    # slice the mid path
    m = np.asarray(eq_mid[a:b + 1], dtype=float)
    # need positive, finite
    if len(m) < 2 or not np.isfinite(m).all() or (m <= 0).any():
        return 0.0, 0.0, 0.0
    # per-event LOG returns for the jump test
    r = np.diff(np.log(m))
    ar = np.abs(r)
    # robust local sigma (MAD/0.6745) or mean abs for small samples
    if len(ar) >= 5:
        sigma_local = np.median(ar) / 0.6745
    else:
        sigma_local = ar.mean()
    # degenerate sigma -> all treated as diffusion
    if not np.isfinite(sigma_local) or sigma_local <= 0:
        # total by sum-of-steps
        return 0.0, float(m[-1] - m[0]), float(m[-1] - m[0])
    # per-return jump mask
    is_jump = ar > (JUMP_K * sigma_local)
    # signed per-event PRICE steps
    dpx = np.diff(m)
    # jump / diffusion contributions (sum-of-steps method)
    jump_move = float(dpx[is_jump].sum())
    diff_move = float(dpx[~is_jump].sum())
    # ALSO compute the endpoint difference via ASOF (as markout does)
    m_open_asof = _mid_at(eq_t, eq_mid, fill_t)
    m_exit_asof = _mid_at(eq_t, eq_mid, exit_t)
    # this is the value markout uses
    total_asof = (m_exit_asof - m_open_asof) if (
        np.isfinite(m_open_asof) and np.isfinite(m_exit_asof)) else 0.0
    # return (jump, diff, total_used_by_markout) so we can measure the gap
    return jump_move, diff_move, total_asof


# process one symbol-day and return the per-bucket sums + jump-fraction data
def process_day(date, sym, calib):
    # session segments
    segs = calib["segments"].get(str(date))
    if segs is None:
        return None
    # walk-forward clip anchor
    med = H.trailing_median(calib["tstats"][sym], calib["all_dates"],
                            date, TRAIL_DAYS)
    if med is None or med <= 0:
        return None
    # calibration presence
    if sym not in calib["scales"] or sym not in calib["profiles"]:
        return None
    # datasets
    dsets = R.open_datasets(date)
    if dsets is None:
        return None
    # locked production params
    clip = max(1, int(round(CLIP_MULT * med)))
    params = H.build_micro_params(
        clip, calib["scales"][sym], calib["profiles"][sym],
        calib["windows"].get(sym, (5.0, 1.0)), segs)
    # run backtest
    dr = H.run_symbol_day(date, sym, dsets, params)
    if dr is None or dr.pnl() is None:
        return None
    # equity -> mid series
    eq = pd.DataFrame(dr.equity) if len(dr.equity) else pd.DataFrame()
    if len(eq) == 0 or "mid" not in eq.columns:
        return None
    eq = eq.sort_values("t")
    eq_t = eq["t"].to_numpy(dtype=float)
    eq_mid = eq["mid"].to_numpy(dtype=float)
    # ---- run the EXACT SAME FIFO matching as fifo_attribution to get the
    # ground-truth realized_pnl per bucket the earlier panel used ----
    per_recon = H.fifo_attribution(
        dr.fills.to_dict("records"), float(dr.pnl()), dr.session[1])
    # ---- and, independently, our decomposition: capture + markout - fees ----
    per_dec = {b: {"capture": 0.0, "markout": 0.0, "fee": 0.0,
                   "jump_markout": 0.0, "diff_markout": 0.0,
                   "markout_asof_sumsteps_gap": 0.0}
               for b in H.BUCKETS}
    # jump-fraction sanity: count events per fill window and flagged fraction
    total_events = 0
    total_jumps = 0
    # invariant check on individual matched round-trips
    invariant_gaps = []
    # FIFO walk
    open_lots = []
    fills = dr.fills.to_dict("records") if isinstance(dr.fills, pd.DataFrame) \
        else list(dr.fills)
    for fl in fills:
        side = fl["side"]; px = float(fl["px"]); qty = float(fl["qty"])
        t = float(fl["t"]); b = fl.get("bucket", "middle")
        # mid at this fill
        mid_at_fill = _mid_at(eq_t, eq_mid, t)
        # capture sign
        cap_sign = 1.0 if side == "BUY" else -1.0
        # capture in PKR
        cap = (cap_sign * (mid_at_fill - px) * qty
               if np.isfinite(mid_at_fill) else 0.0)
        # opens if empty or same side
        if not open_lots or open_lots[0]["side"] == side:
            open_lots.append({"qty": qty, "px": px, "bucket": b, "t": t,
                              "side": side, "mid_at_fill": mid_at_fill})
            if b in per_dec:
                per_dec[b]["capture"] += cap
            continue
        # opposite side -> close FIFO
        remaining = qty
        while remaining > 1e-9 and open_lots and open_lots[0]["side"] != side:
            lot = open_lots[0]
            matched = min(remaining, lot["qty"])
            m_open = lot.get("mid_at_fill", np.nan)
            m_exit = _mid_at(eq_t, eq_mid, t)
            # markout via ASOF endpoint difference (what spot_capture uses)
            if np.isfinite(m_open) and np.isfinite(m_exit):
                open_sign = 1.0 if lot["side"] == "BUY" else -1.0
                mko_asof = open_sign * (m_exit - m_open) * matched
                # split via SUM-OF-STEPS over the mid path
                jm_step, dm_step, total_asof_step = _split_move_diagnostic(
                    eq_t, eq_mid, lot["t"], t)
                # scale to matched shares
                mko_jump = open_sign * jm_step * matched
                mko_diff = open_sign * dm_step * matched
                # count events + jumps in this window for the flagged fraction
                a = np.searchsorted(eq_t, lot["t"], side="right") - 1
                bidx = np.searchsorted(eq_t, t, side="right") - 1
                a = max(0, a); bidx = max(a, bidx)
                if bidx > a:
                    m = np.asarray(eq_mid[a:bidx + 1], dtype=float)
                    if len(m) >= 2 and np.isfinite(m).all() and (m > 0).all():
                        r = np.abs(np.diff(np.log(m)))
                        # local sigma
                        sigma_local = (np.median(r) / 0.6745
                                       if len(r) >= 5 else r.mean())
                        if sigma_local > 0 and np.isfinite(sigma_local):
                            # count this window's events + jumps
                            total_events += len(r)
                            total_jumps += int((r > JUMP_K * sigma_local).sum())
                # invariant check: |markout - (jump + diff)| for this trip
                gap = mko_asof - (mko_jump + mko_diff)
                invariant_gaps.append(gap)
            else:
                mko_asof = 0.0; mko_jump = 0.0; mko_diff = 0.0
            # fees (two legs)
            fee = H.fee_for(lot["px"], matched) + H.fee_for(px, matched)
            # attribute to opening bucket
            ob = lot["bucket"]
            if ob in per_dec:
                per_dec[ob]["markout"] += mko_asof
                per_dec[ob]["jump_markout"] += mko_jump
                per_dec[ob]["diff_markout"] += mko_diff
                per_dec[ob]["fee"] += fee
                per_dec[ob]["markout_asof_sumsteps_gap"] += (
                    mko_asof - (mko_jump + mko_diff))
            # shrink lot + remaining
            lot["qty"] -= matched
            remaining -= matched
            if lot["qty"] <= 1e-9:
                open_lots.pop(0)
        # flip: remainder opens the other side
        if remaining > 1e-9:
            open_lots.append({"qty": remaining, "px": px, "bucket": b, "t": t,
                              "side": side, "mid_at_fill": mid_at_fill})
            if b in per_dec:
                per_dec[b]["capture"] += cap * (remaining / qty) if qty > 0 else 0.0
    # residual lots -> mark to last mid at liquidation time (same as main file)
    liq_t = float(dr.session[1]) if dr.session is not None else 0.0
    liq_px = float(eq_mid[-1]) if len(eq_mid) else np.nan
    for lot in open_lots:
        ob = lot["bucket"]
        m_open = lot.get("mid_at_fill", np.nan)
        if np.isfinite(m_open) and np.isfinite(liq_px):
            open_sign = 1.0 if lot["side"] == "BUY" else -1.0
            mko_asof = open_sign * (liq_px - m_open) * lot["qty"]
            jm_step, dm_step, _ = _split_move_diagnostic(
                eq_t, eq_mid, lot["t"], liq_t)
            mko_jump = open_sign * jm_step * lot["qty"]
            mko_diff = open_sign * dm_step * lot["qty"]
            if ob in per_dec:
                per_dec[ob]["markout"] += mko_asof
                per_dec[ob]["jump_markout"] += mko_jump
                per_dec[ob]["diff_markout"] += mko_diff
                per_dec[ob]["markout_asof_sumsteps_gap"] += (
                    mko_asof - (mko_jump + mko_diff))
    # bundle return
    return {"per_recon": per_recon, "per_dec": per_dec,
            "total_events": total_events, "total_jumps": total_jumps,
            "invariant_gaps": invariant_gaps,
            "engine_pnl": float(dr.pnl())}


def main():
    # timing
    t0 = time.perf_counter()
    # calibration bundle (like the fast harness)
    print("pre-pass: loading calibration + trailing median trade size",
          flush=True)
    all_dates = R.discover_dates()
    calib = {"scales": H.load_scales(), "profiles": H.load_profiles(),
             "windows": H.load_windows(), "segments": H.load_segments(),
             "all_dates": all_dates,
             "tstats": H.trailing_median_trade_size(all_dates, NAMES,
                                                    TRAIL_DAYS)}
    # smoke dates
    run_dates = [str(d) for d in all_dates[TRAIL_DAYS:TRAIL_DAYS + N_DAYS]]
    # accumulators
    agg_recon = {b: 0.0 for b in H.BUCKETS}
    agg_dec = {b: {"capture": 0.0, "markout": 0.0, "fee": 0.0,
                   "jump_markout": 0.0, "diff_markout": 0.0}
               for b in H.BUCKETS}
    total_events = 0
    total_jumps = 0
    all_gaps = []
    engine_pnl_total = 0.0
    # walk
    print(f"\nrunning {len(run_dates)} days x {len(NAMES)} names at "
          f"{CLIP_MULT}x clip...\n", flush=True)
    for i, date in enumerate(run_dates, 1):
        for sym in NAMES:
            res = process_day(date, sym, calib)
            if res is None:
                continue
            # earlier-panel realized
            for b in H.BUCKETS:
                agg_recon[b] += res["per_recon"][b]["realized"]
            # decomposition components
            for b in H.BUCKETS:
                for k in ("capture", "markout", "fee", "jump_markout",
                          "diff_markout"):
                    agg_dec[b][k] += res["per_dec"][b][k]
            total_events += res["total_events"]
            total_jumps += res["total_jumps"]
            all_gaps.extend(res["invariant_gaps"])
            engine_pnl_total += res["engine_pnl"]
        print(f"  {i}/{len(run_dates)} days done", flush=True)
    # ---- REPORTING ----
    print("\n" + "=" * 78)
    print(f"### RUNTIME: {H._fmt(time.perf_counter() - t0)} ###")
    print("=" * 78)
    # (1) LEVELS RECONCILIATION
    print("\n=== (1) LEVELS: capture+markout-fees vs earlier realized_pnl ===")
    print(f"{'bucket':12s} {'earlier realized':>18s} "
          f"{'this run (cap+mko-fee)':>24s} {'gap':>14s} {'gap %':>8s}")
    total_earlier = 0.0
    total_dec = 0.0
    for b in H.BUCKETS:
        earlier = agg_recon[b]
        dec = agg_dec[b]["capture"] + agg_dec[b]["markout"] - agg_dec[b]["fee"]
        gap = dec - earlier
        pct = (100 * gap / earlier) if abs(earlier) > 1 else float("nan")
        total_earlier += earlier
        total_dec += dec
        print(f"{b:12s} {earlier:>18,.0f} {dec:>24,.0f} {gap:>14,.0f} "
              f"{pct:>7.1f}%")
    print(f"{'TOTAL':12s} {total_earlier:>18,.0f} {total_dec:>24,.0f} "
          f"{total_dec - total_earlier:>14,.0f}")
    print(f"\nengine P&L sum across all symbol-days: {engine_pnl_total:,.0f}")
    # verdict
    if abs(total_dec - total_earlier) < 0.01 * abs(total_earlier):
        print("-> LEVELS RECONCILE (within 1%): the new run's absolute PnL is "
              "trustworthy.")
    else:
        print(f"-> LEVELS DO NOT RECONCILE. Gap = {total_dec - total_earlier:,.0f} "
              f"({100 * (total_dec - total_earlier) / total_earlier:.1f}% of earlier).")
        print("   The decomposition's absolute PnL is not the same measure as "
              "the earlier attribution.")
        print("   Ratios (jump/diff/OBI/clip-monotonicity) remain trustworthy; "
              "absolute levels do not.")
    # (2) JUMP-FLAG FRACTION
    print("\n=== (2) JUMP DETECTOR: what fraction of events flagged as jumps? ===")
    frac = (100 * total_jumps / total_events) if total_events > 0 else 0.0
    print(f"total mid events across all fill windows: {total_events:,}")
    print(f"events flagged as jumps at k={JUMP_K}:      {total_jumps:,}")
    print(f"flagged fraction:                          {frac:.4f}%")
    # verdict
    if frac < 0.01:
        print(f"-> Jump detector is essentially INERT on PSX (flagged {frac:.4f}%). "
              "The 'diffusive dominates' reading is partly an artifact of "
              "under-detection. Consider lowering JUMP_K.")
    elif frac > 5.0:
        print(f"-> Flagged fraction {frac:.2f}% is unusually high. Consider "
              "raising JUMP_K -- many normal returns are being called jumps.")
    else:
        print(f"-> Flagged fraction {frac:.2f}% is in the expected range "
              "(0.01-5%). The jump split is meaningful.")
    # (3) INVARIANT CHECK
    print("\n=== (3) INVARIANT: does markout == jump + diff per round-trip? ===")
    if all_gaps:
        g = np.array(all_gaps)
        print(f"round-trips checked: {len(g):,}")
        print(f"max |gap|:  {np.max(np.abs(g)):,.2f}")
        print(f"mean gap:   {g.mean():,.4f}")
        print(f"median gap: {np.median(g):,.4f}")
        # verdict
        if np.max(np.abs(g)) < 1.0:
            print("-> Invariant HOLDS (max gap < 1 PKR). No bug.")
        else:
            print(f"-> Invariant BROKEN: max per-trip gap is {np.max(np.abs(g)):,.2f} "
                  "PKR.")
            print("   Cause: markout uses asof(exit) - asof(open) endpoints, but "
                  "jump/diff sums per-event STEPS between them.")
            print("   When asof indices for the endpoints differ from the slice "
                  "indices used by split_move, the endpoint value and the "
                  "sum-of-steps don't match.")
            print("   FIX: force diff = markout_asof - jump_component (residual "
                  "plug pattern -- guarantees reconciliation by construction).")


if __name__ == "__main__":
    main()
