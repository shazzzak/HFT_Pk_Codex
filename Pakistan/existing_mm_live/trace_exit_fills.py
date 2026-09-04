# trace_exit_fills.py -- the smallest possible check: does exit_ticks_inside=1
# make our EXIT fills happen at self-harming prices, or is the P&L collapse an
# accounting artifact? Reads the engine's OWN fills + equity + eod directly.
# NO decomposition, NO plug, NO reconstruction -- just raw engine output.
#
# For one symbol, one day, it runs exit_ticks=0 and exit_ticks=1 and prints:
#   * raw engine P&L (dr.pnl()) for each -- the ground truth
#   * a sample of EXIT-side fills with: fill price, touch (bb/ba) at that moment,
#     ticks inside the touch, and the signed cash the fill produced
# If exit_ticks=1 fills sit at sane prices (0-1 tick inside) and P&L is similar,
# the earlier "collapse" was an accounting artifact. If they sit deep inside the
# touch / cross, and cash is worse, the collapse is real placement behavior.
#
# Run:  python3 existing_mm_live/trace_exit_fills.py

from pathlib import Path
import numpy as np
import pandas as pd
import mm_harness as H
import run_legacy_mm as R

R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")

# ------------------------------ config ---------------------------------------
# one liquid name, one day (index into the trading calendar after the warmup)
SYM = "PPL"
DAY_INDEX = 15
CLIP_MULT = 3.0
TRAIL_DAYS = 10
# how many exit fills to print per config
N_SAMPLE = 25
# -----------------------------------------------------------------------------


# run one config and return (dr, params)
def run_config(date, dsets, calib, exit_ticks):
    segs = calib["segments"].get(str(date))
    med = H.trailing_median(calib["tstats"][SYM], calib["all_dates"], date, TRAIL_DAYS)
    clip = max(1, int(round(CLIP_MULT * med)))
    params = H.build_micro_params(
        clip, calib["scales"][SYM], calib["profiles"][SYM],
        calib["windows"].get(SYM, (5.0, 1.0)), segs,
        overrides={"exit_ticks_inside": exit_ticks, "exit_inv_threshold": 1.0,
                   "obi_defensive": False})
    dr = H.run_symbol_day(date, SYM, dsets, params)
    return dr


# print the exit-fill trace for one config
def trace(dr, exit_ticks):
    # engine ground-truth P&L
    pnl = dr.pnl()
    print(f"\n{'='*76}")
    print(f"exit_ticks_inside={exit_ticks}   RAW ENGINE P&L = {pnl:,.2f} PKR")
    if dr.eod:
        # show the measured liquidation directly from the engine
        print(f"  eod: pos_at_close={dr.eod.get('pos_at_close')}  "
              f"liq_vwap={dr.eod.get('liq_vwap')}  "
              f"unfilled_sh={dr.eod.get('unfilled_sh')}  "
              f"liq_slippage_per_sh={dr.eod.get('liq_slippage_per_sh')}")
    print("=" * 76)
    # equity log -> touch series (bb/ba) for the moment of each fill
    eq = dr.equity if isinstance(dr.equity, pd.DataFrame) else pd.DataFrame(dr.equity)
    eq = eq.sort_values("t")
    eq_t = eq["t"].to_numpy(float)
    have_touch = "bb" in eq.columns and "ba" in eq.columns
    eq_bb = eq["bb"].to_numpy(float) if have_touch else None
    eq_ba = eq["ba"].to_numpy(float) if have_touch else None
    eq_pos = eq["pos"].to_numpy(float) if "pos" in eq.columns else None
    # session close (exchange ms) for minutes-to-close + EOD-zone labelling
    sess_close = float(dr.session[1]) if dr.session is not None else None
    sess_open = float(dr.session[0]) if dr.session is not None else None
    # session bucket by timestamp (first15/middle/preclose45/last15) -- this is
    # the TIME-OF-DAY bucket, DISTINCT from the trigger 'window' (none/time_ramp/
    # time_cliff/lock_ramp/lock_cliff). They are different things: bucket = when in
    # the day; window = whether an EOD/lock trigger was firing.
    def _bucket_of(t):
        if sess_open is None or sess_close is None:
            return "?"
        if t < sess_open + 15 * 60000:
            return "first15"
        if t >= sess_close - 15 * 60000:
            return "last15"
        if t >= sess_close - 60 * 60000:
            return "preclose45"
        return "middle"
    # asof index of the touch just before a fill time
    def touch_before(t):
        pos = np.searchsorted(eq_t, t, side="right") - 1
        pos = max(0, pos - 0)  # the row at/just before the fill
        return pos
    # walk fills, reconstruct running position to identify EXIT fills
    fills = dr.fills.to_dict("records") if isinstance(dr.fills, pd.DataFrame) \
        else list(dr.fills)
    # ---- posting-time map: fill oid -> when the order went LIVE (o.t_active).
    # The live engine's DayResult carries an order_log (per-order lifecycle).
    # Its exact columns are live-side instrumentation, so DETECT them instead of
    # assuming: id column from {oid, order_id, id}; live-time column from
    # {t_active, t_live, t_sent, t_place, t}. Prints what it found so the join
    # is auditable, and degrades to '-' when unavailable.
    post_t = {}
    olog = getattr(dr, "order_log", None)
    if olog is not None and len(olog):
        ocols = list(olog.columns)
        # detect the id column shared with fills' oid
        id_col = next((c for c in ("oid", "order_id", "id") if c in ocols), None)
        # detect the went-live time column (t_active is the engine's name)
        t_col = next((c for c in ("t_active", "t_live", "t_sent", "t_place", "t")
                      if c in ocols), None)
        print(f"  order_log columns: {ocols}  -> join on id={id_col}, time={t_col}")
        if id_col and t_col:
            # one posting time per order id (first live time)
            post_t = olog.groupby(id_col)[t_col].first().to_dict()
    else:
        print("  order_log: not available -> posted_at/touch-at-post shown as '-'")
    running_pos = 0.0
    printed = 0
    print(f"{'time':>12s} {'side':>4s} {'fill_px':>9s} {'bb':>9s} {'ba':>9s} "
          f"{'tks_vs_mid':>8s} {'reason':>12s} {'window':>10s} {'bucket':>10s} "
          f"{'min2close':>9s} {'age_s':>9s} {'bb@post':>8s} {'ba@post':>8s} "
          f"{'signed_cash':>12s}")
    for fl in fills:
        side = fl["side"]; px = float(fl["px"]); qty = float(fl["qty"]); t = float(fl["t"])
        # minutes remaining to the session close (negative = after close/liquidation)
        min2close = ((sess_close - t) / 60000.0) if sess_close is not None else float("nan")
        # is this fill in the last-15-min EOD zone (where an EOD unwind would fire)?
        eod_zone = "YES" if (sess_close is not None and 0 <= (sess_close - t) <= 15 * 60000) \
            else ("POST" if (sess_close is not None and t > sess_close) else "no")
        # ---- posting time + touch at posting (the pick-off check): when did THIS
        # order go live, and what was the touch then? A SELL filled below the
        # CURRENT bid but posted ABOVE the THEN-bid = stale-quote pick-off, not a
        # crossing. age_s = seconds the quote rested before filling.
        fo = fl.get("oid")
        tp = post_t.get(fo)
        if tp is not None and have_touch:
            # resting age: fill time minus the moment the order went LIVE
            # (t_live = after send latency, i.e. when it actually started
            # resting in the book). Millisecond resolution so you can see
            # whether pick-offs happen in ms, seconds, or minutes.
            age_s = (t - float(tp)) / 1000.0
            j = touch_before(float(tp))
            bb_p, ba_p = eq_bb[j], eq_ba[j]
        else:
            age_s, bb_p, ba_p = float("nan"), float("nan"), float("nan")
        # is this fill REDUCING the position (an exit)? BUY reduces a short,
        # SELL reduces a long.
        is_exit = (side == "BUY" and running_pos < 0) or \
                  (side == "SELL" and running_pos > 0)
        # touch at the fill moment
        if have_touch:
            i = touch_before(t)
            bb, ba = eq_bb[i], eq_ba[i]
            # mid, and SIGNED ticks vs mid: + = passive (earned spread),
            # - = crossed the mid (paid the spread). Measuring vs the opposite
            # touch was the bug -- it turned a sell that crossed the bid into a
            # nonsense "+26 inside" when it was really paying to cross.
            mid_ref = 0.5 * (bb + ba) if (np.isfinite(bb) and np.isfinite(ba)) else float("nan")
            if side == "BUY":
                tks = (mid_ref - px) / 0.01
            else:
                tks = (px - mid_ref) / 0.01
        else:
            bb = ba = mid_ref = tks = float("nan")
        # signed cash this fill produced (buying spends, selling earns), minus fee
        sgn = 1.0 if side == "BUY" else -1.0
        signed_cash = -sgn * qty * px - H.fee_for(px, qty)
        # the fill's reason + trigger window (reveals EOD/lock unwind crosses)
        reason = fl.get("reason", "?")
        window = fl.get("window", "?")
        # session bucket for this fill (distinct from the trigger window)
        bkt = _bucket_of(t)
        # print only EXIT fills (the ones exit_ticks_inside affects), up to N.
        # ALSO always print any fill the engine tagged with a non-none window
        # (the real EOD/lock trigger fills, which live in the last few minutes) so
        # we can confirm the window tagging fires when a fill is genuinely in it.
        show = (is_exit and printed < N_SAMPLE) or (window not in ("none", "?"))
        if show:
            print(f"{t:>12.0f} {side:>4s} {px:>9.2f} {bb:>9.2f} {ba:>9.2f} "
                  f"{tks:>8.2f} {reason:>12s} {window:>10s} {bkt:>10s} "
                  f"{min2close:>9.1f} {age_s:>9.3f} {bb_p:>8.2f} {ba_p:>8.2f} "
                  f"{signed_cash:>12.2f}")
            if is_exit and printed < N_SAMPLE:
                printed += 1
        # update running position
        running_pos += (qty if side == "BUY" else -qty)
    # summary of exit-fill aggressiveness (vs mid, signed), SPLIT by whether the
    # fill is mid-session or in the last-15-min EOD zone -- this answers the key
    # question: are the crossing exits the exit-skew MECHANISM (mid-session) or
    # the legitimate EOD unwind (EOD zone / post-close)?
    mid_sess_tks = []   # exit fills during normal trading (> 15 min to close)
    eod_zone_tks = []   # exit fills in the last 15 min or after close
    running_pos = 0.0
    for fl in fills:
        side = fl["side"]; px = float(fl["px"]); t = float(fl["t"])
        is_exit = (side == "BUY" and running_pos < 0) or \
                  (side == "SELL" and running_pos > 0)
        if is_exit and have_touch:
            i = touch_before(t)
            mid_ref = 0.5 * (eq_bb[i] + eq_ba[i])
            tks = (mid_ref - px) / 0.01 if side == "BUY" else (px - mid_ref) / 0.01
            # classify by time-to-close
            in_eod = (sess_close is not None and (sess_close - t) <= 15 * 60000)
            (eod_zone_tks if in_eod else mid_sess_tks).append(tks)
        running_pos += (float(fl["qty"]) if side == "BUY" else -float(fl["qty"]))
    # report each cohort separately
    def _summ(name, lst):
        if not lst:
            print(f"  {name}: (none)")
            return
        a = np.array(lst)
        crossed = 100 * (a < 0).mean()
        print(f"  {name}: n={len(a)}  median tks_vs_mid={np.median(a):+.2f}  "
              f"mean={a.mean():+.2f}  crossed_mid={crossed:.0f}%")
    print(f"\n  EXIT-fill breakdown (+ = passive/earned spread, - = crossed/paid):")
    # MID-SESSION is the one that tests the exit-skew mechanism
    _summ("MID-SESSION (>15min to close, tests the mechanism)", mid_sess_tks)
    # EOD zone is the legitimate end-of-day flatten
    _summ("EOD-ZONE     (<=15min to close, the unwind)       ", eod_zone_tks)
    print("  -> if MID-SESSION crossed_mid is high, the mechanism itself is")
    print("     crossing (the bug); if only EOD-ZONE crosses, that's the normal unwind.")
    # window-tag tally: how many fills fired in each trigger window. Confirms the
    # tagging works and shows the EOD window is only the last few minutes (so most
    # fills are correctly 'none'). eod_ramp/cliff default to (5,1) min -> tiny.
    from collections import Counter
    wc = Counter(fl.get("window", "?") for fl in fills)
    print(f"\n  window-tag tally (all fills): {dict(wc)}")
    print("  (the EOD trigger window is only the last eod_ramp_start_min minutes")
    print("   before close (default 5), so 'none' for mid-session fills is CORRECT.)")
    # ---- RESTING-AGE distribution over ALL fills (not just the printed sample):
    # how long did each filled order rest (t_fill - t_live) before executing?
    # Split by reason: 'through' = market ran over the quote (pick-off speed);
    # 'at_queue' = queue cleared to us (benign passive fill). Answers whether the
    # market moves through us in ms, seconds, or minutes.
    ages = {}
    # per-bucket ages (first15/middle/preclose45/last15) to quantify the
    # bucket-dependence SZ observed in the per-row age_s values
    ages_by_bucket = {}
    for fl in fills:
        tp = post_t.get(fl.get("oid"))
        if tp is None:
            continue
        # resting age in seconds
        a = (float(fl["t"]) - float(tp)) / 1000.0
        ages.setdefault(fl.get("reason", "?"), []).append(a)
        # bucket of the FILL time (when the market reached us)
        ages_by_bucket.setdefault(_bucket_of(float(fl["t"])), []).append(a)
    if ages:
        print("\n  RESTING AGE before fill (t_fill - t_live), seconds, ALL fills:")
        for rsn in sorted(ages):
            arr = np.array(ages[rsn])
            print(f"    {rsn:>12s}: n={len(arr):>4d}  p25={np.percentile(arr,25):>8.3f}  "
                  f"median={np.median(arr):>8.3f}  p75={np.percentile(arr,75):>8.3f}  "
                  f"max={arr.max():>9.1f}")
        # pooled
        allages = np.concatenate([np.array(v) for v in ages.values()])
        print(f"    {'ALL':>12s}: n={len(allages):>4d}  p25={np.percentile(allages,25):>8.3f}  "
              f"median={np.median(allages):>8.3f}  p75={np.percentile(allages,75):>8.3f}  "
              f"max={allages.max():>9.1f}")
        # by session bucket (chronological order, only buckets with fills)
        print("  RESTING AGE by session bucket:")
        for bkt_ in ("first15", "middle", "preclose45", "last15"):
            if bkt_ not in ages_by_bucket:
                continue
            arr = np.array(ages_by_bucket[bkt_])
            print(f"    {bkt_:>12s}: n={len(arr):>4d}  p25={np.percentile(arr,25):>8.3f}  "
                  f"median={np.median(arr):>8.3f}  p75={np.percentile(arr,75):>8.3f}  "
                  f"max={arr.max():>9.1f}")


def main():
    all_dates = R.discover_dates()
    date = all_dates[TRAIL_DAYS + DAY_INDEX]
    print(f"trace: {SYM} on {date}, exit_ticks 0 vs 1")
    calib = {"scales": H.load_scales(), "profiles": H.load_profiles(),
             "windows": H.load_windows(), "segments": H.load_segments(),
             "all_dates": all_dates,
             "tstats": H.trailing_median_trade_size(all_dates, [SYM], TRAIL_DAYS)}
    dsets = R.open_datasets(date)
    if dsets is None:
        print("no data for this day"); return
    # run both configs and trace each
    for et in (0, 1):
        dr = run_config(date, dsets, calib, et)
        if dr is None or dr.pnl() is None:
            print(f"exit_ticks={et}: no result"); continue
        trace(dr, et)
    # the decisive comparison
    print(f"\n{'='*76}")
    print("READ: if exit_ticks=1 shows exit fills at 0-1 ticks inside and P&L")
    print("close to exit_ticks=0, the earlier collapse was an ACCOUNTING artifact.")
    print("If exit fills sit deep inside the touch with worse signed_cash and the")
    print("raw engine P&L genuinely drops, the collapse is REAL placement behavior.")
    print("=" * 76)


if __name__ == "__main__":
    main()
