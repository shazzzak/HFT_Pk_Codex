# skew_sweep_2d.py -- 2D sweep of inventory-exit aggressiveness x OBI-defensive
# skew, to test whether shortening hold time (the confirmed driver of diffusive
# markout) and avoiding adverse-imbalance fills improves net edge.
#
# THE HYPOTHESIS (stated so the data can refute it): median hold of 4-6 minutes
# is NOT high-frequency; the markout loss is dominated by DIFFUSION accrued while
# holding (diff_mko >> jump_mko, and it grows with hold time). Therefore:
#   AXIS 1 (exit_ticks_inside): when loaded, post the EXIT side N ticks inside
#     the touch -> fills that leave faster -> shorter hold -> less diffusive
#     markout. Cost: less capture (posting inside earns less than the half-spread).
#   AXIS 2 (obi_defensive): OBI-at-fill is systematically against us; quotes
#     ignore book imbalance. Suppressing the side the book leans against should
#     raise (toward 0) the direction-signed OBI-at-fill and cut adverse selection.
# If net_bps rises, one/both work; if it falls, holding longer was cheaper and the
# markout is not hold-time-fixable -- either way we learn the mechanism.
#
# GRID: exit_ticks_inside in {0,1,2,3}  x  obi_defensive in {False, True}
#       = 8 configs. N=0 & obi_defensive=False == the validated baseline
#       (byte-identical placement -- unit-tested in micro_mm.py).
# CLIP: production 3x only (the capacity question is already settled; this sweep
#       is about the hold-time/adverse-selection mechanism, not capacity).
#
# PER-CONFIG REPORT (per bucket): capture, markout (jump/diff), fees, liq_loss,
#   net-bps, median+mean hold, realized median ticks-inside-the-touch, raw
#   OBI-at-fill AND direction-signed OBI-at-fill split by buy/sell fills, number
#   of trades, and average shares per trade.
#
# Runtime: ~8x a single-config decomposition. Expect a few hours; heartbeat +
# ETA are printed. Run with caffeinate.
#
#   caffeinate -is python3 existing_mm_live/skew_sweep_2d.py

# paths + timing
from pathlib import Path
import time
from datetime import datetime
# parallelism
import multiprocessing as mp
# arrays + frames
import numpy as np
import pandas as pd
# significance
from scipy import stats
# harness + driver
import mm_harness as H
import run_legacy_mm as R
# reuse the validated decomposition helpers (mid asof + jump/diff split)
from spot_capture_markout_decomp import _mid_at, _split_move

# store paths
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ config ---------------------------------------
# the top-10 production book
NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
# single production clip (capacity already settled; this is a mechanism sweep)
CLIP_MULT = 3.0
# trailing median-trade-size window (days)
TRAIL_DAYS = 10
# AXIS 1: ticks inside the touch on the exit side when loaded (0 = baseline)
EXIT_TICKS = [0, 1, 2, 3]
# AXIS 2: OBI-defensive skew on/off
OBI_MODES = [False, True]
# inventory threshold (lots) beyond which the tick-exit engages
EXIT_INV_THRESHOLD = 1.0
# OBI-defensive engage threshold (|imb-0.5|) and widen ticks
OBI_DEF_THRESH = 0.15
OBI_DEF_TICKS = 1.0
# jump detector (same as the decomposition)
JUMP_K = 4.0
# smoke: first N days (None -> full ~207)
SMOKE_DAYS = None
# workers
WORKERS = None
# -----------------------------------------------------------------------------

# worker globals
_G = {}


# worker init
def _init_worker(calib):
    _G.update(calib)


# one (date, symbol, exit_ticks, obi_defensive) cell
def _process(args):
    # unpack the work tuple
    date, sym, exit_ticks, obi_def = args
    # calibration
    scales = _G["scales"]; profiles = _G["profiles"]; windows = _G["windows"]
    segments = _G["segments"]; all_dates = _G["all_dates"]; tstats = _G["tstats"]
    # segments for the day
    segs = segments.get(str(date))
    if segs is None:
        return None
    # trailing median trade size
    med = H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
    if med is None or med <= 0:
        return None
    if sym not in scales or sym not in profiles:
        return None
    # datasets
    dsets = R.open_datasets(date)
    if dsets is None:
        return None
    # clip
    clip = max(1, int(round(CLIP_MULT * med)))
    # build params, then INJECT the sweep axes via overrides (baseline when
    # exit_ticks=0 and obi_def=False -- unit-tested byte-identical)
    params = H.build_micro_params(
        clip, scales[sym], profiles[sym], windows.get(sym, (5.0, 1.0)), segs,
        overrides={"exit_ticks_inside": exit_ticks,
                   "exit_inv_threshold": EXIT_INV_THRESHOLD,
                   "obi_defensive": obi_def,
                   "obi_defensive_thresh": OBI_DEF_THRESH,
                   "obi_defensive_ticks": OBI_DEF_TICKS})
    # run
    dr = H.run_symbol_day(date, sym, dsets, params)
    if dr is None or dr.pnl() is None:
        return None
    # mid + touch series from the equity log
    eq = pd.DataFrame(dr.equity) if len(dr.equity) else pd.DataFrame()
    if len(eq) == 0 or "mid" not in eq.columns:
        return None
    eq = eq.sort_values("t")
    eq_t = eq["t"].to_numpy(dtype=float)
    eq_mid = eq["mid"].to_numpy(dtype=float)
    # best bid/ask for ticks-inside + OBI (if logged)
    have_touch = "bb" in eq.columns and "ba" in eq.columns
    eq_bb = eq["bb"].to_numpy(dtype=float) if have_touch else None
    eq_ba = eq["ba"].to_numpy(dtype=float) if have_touch else None
    # OBI at fill (if logged)
    have_obi = "obi_5" in eq.columns and "obi_deep" in eq.columns
    eq_obi5 = eq["obi_5"].to_numpy(dtype=float) if have_obi else None
    eq_obi_d = eq["obi_deep"].to_numpy(dtype=float) if have_obi else None
    # tick size for ticks-inside
    tick = params.get("tick", 0.01)
    # per-bucket accumulators
    per = {b: {"capture": 0.0, "markout": 0.0, "fee": 0.0, "liq_fee": 0.0,
               "jump_markout": 0.0, "diff_markout": 0.0, "liq_loss": 0.0,
               "opened_notional": 0.0, "opened_qty": 0.0, "holds": [],
               "fills": 0, "trades": 0, "shares": 0.0,
               # raw OBI at fill (book-signed) and DIRECTION-signed OBI at fill,
               # split by buy vs sell fills (the corrected metric)
               "obi5_raw_sum": 0.0, "obi5_signed_sum": 0.0,
               "obi5_buy_sum": 0.0, "obi5_buy_n": 0,
               "obi5_sell_sum": 0.0, "obi5_sell_n": 0, "obi_n": 0,
               # realized ticks inside the touch on our fills
               "ticks_inside": []}
           for b in H.BUCKETS}
    # asof index helper
    def _asof_idx(t):
        pos = np.searchsorted(eq_t, t, side="right") - 1
        return pos if pos >= 0 else None
    # FIFO queue
    open_lots = []
    fills = dr.fills.to_dict("records") if isinstance(dr.fills, pd.DataFrame) \
        else list(dr.fills)
    # walk fills
    for fl in fills:
        side = fl["side"]; px = float(fl["px"]); qty = float(fl["qty"])
        t = float(fl["t"]); b = fl.get("bucket", "middle")
        # count trade + shares + fill in its bucket
        if b in per:
            per[b]["fills"] += 1
            per[b]["trades"] += 1
            per[b]["shares"] += qty
        # asof book state at the fill
        idx = _asof_idx(t)
        # OBI at fill (raw + direction-signed) and ticks-inside
        if idx is not None and b in per:
            # direction sign: +1 if we BOUGHT, -1 if we SOLD -> signed OBI < 0
            # ALWAYS means "book leaned against our fill", regardless of side.
            dir_sign = 1.0 if side == "BUY" else -1.0
            if have_obi:
                # raw book-signed OBI (what the earlier table showed)
                per[b]["obi5_raw_sum"] += eq_obi5[idx]
                # direction-signed OBI (the corrected adverse-selection metric)
                per[b]["obi5_signed_sum"] += dir_sign * eq_obi5[idx]
                # split by side so we can see WHICH side gets picked
                if side == "BUY":
                    per[b]["obi5_buy_sum"] += eq_obi5[idx]
                    per[b]["obi5_buy_n"] += 1
                else:
                    per[b]["obi5_sell_sum"] += eq_obi5[idx]
                    per[b]["obi5_sell_n"] += 1
                per[b]["obi_n"] += 1
            # realized ticks inside the touch: for a BUY fill, how far above bb;
            # for a SELL fill, how far below ba. Measures the sweep's actual
            # aggressiveness on filled orders.
            if have_touch and np.isfinite(eq_bb[idx]) and np.isfinite(eq_ba[idx]):
                if side == "BUY":
                    ti = (px - eq_bb[idx]) / tick
                else:
                    ti = (eq_ba[idx] - px) / tick
                per[b]["ticks_inside"].append(ti)
        # mid at fill for capture
        mid_at_fill = _mid_at(eq_t, eq_mid, t)
        cap_sign = 1.0 if side == "BUY" else -1.0
        cap = (cap_sign * (mid_at_fill - px) * qty
               if np.isfinite(mid_at_fill) else 0.0)
        # opens if empty/same side
        if not open_lots or open_lots[0]["side"] == side:
            open_lots.append({"qty": qty, "px": px, "bucket": b, "t": t,
                              "side": side, "mid_at_fill": mid_at_fill})
            if b in per:
                per[b]["opened_notional"] += qty * px
                per[b]["opened_qty"] += qty
                per[b]["capture"] += cap
            continue
        # opposite side -> close FIFO
        remaining = qty
        while remaining > 1e-9 and open_lots and open_lots[0]["side"] != side:
            lot = open_lots[0]
            matched = min(remaining, lot["qty"])
            m_open = lot.get("mid_at_fill", np.nan)
            m_exit = _mid_at(eq_t, eq_mid, t)
            if np.isfinite(m_open) and np.isfinite(m_exit):
                open_sign = 1.0 if lot["side"] == "BUY" else -1.0
                mko = open_sign * (m_exit - m_open) * matched
                jm, _ = _split_move(eq_t, eq_mid, lot["t"], t)
                mko_jump = open_sign * jm * matched
                mko_diff = mko - mko_jump
            else:
                mko = 0.0; mko_jump = 0.0; mko_diff = 0.0
            # round-trip fees (two legs)
            fee = H.fee_for(lot["px"], matched) + H.fee_for(px, matched)
            ob = lot["bucket"]
            if ob in per:
                per[ob]["markout"] += mko
                per[ob]["jump_markout"] += mko_jump
                per[ob]["diff_markout"] += mko_diff
                per[ob]["fee"] += fee
                per[ob]["holds"].append(t - lot["t"])
            lot["qty"] -= matched
            remaining -= matched
            if lot["qty"] <= 1e-9:
                open_lots.pop(0)
        # flip remainder opens the other side
        if remaining > 1e-9:
            open_lots.append({"qty": remaining, "px": px, "bucket": b, "t": t,
                              "side": side, "mid_at_fill": mid_at_fill})
            if b in per:
                per[b]["opened_notional"] += remaining * px
                per[b]["opened_qty"] += remaining
                if qty > 0:
                    per[b]["capture"] += cap * (remaining / qty)
    # residual lots -> liquidation mark + itemized liq fee
    liq_time = float(dr.session[1]) if dr.session is not None else 0.0
    liq_px = H.liquidation_price(dr) if hasattr(H, "liquidation_price") else float("nan")
    if not (isinstance(liq_px, float) and np.isfinite(liq_px)) or liq_px <= 0:
        liq_px = float(eq_mid[-1]) if len(eq_mid) else 0.0
    for lot in open_lots:
        ob = lot["bucket"]
        open_sign = 1.0 if lot["side"] == "BUY" else -1.0
        m_open = lot.get("mid_at_fill", np.nan)
        if np.isfinite(m_open):
            mko = open_sign * (liq_px - m_open) * lot["qty"]
            jm, _ = _split_move(eq_t, eq_mid, lot["t"], liq_time)
            mko_jump = open_sign * jm * lot["qty"]
            mko_diff = mko - mko_jump
            if ob in per:
                per[ob]["markout"] += mko
                per[ob]["jump_markout"] += mko_jump
                per[ob]["diff_markout"] += mko_diff
                per[ob]["holds"].append(max(0.0, liq_time - lot["t"]))
                # itemized liquidation fee (engine charges it; surface it)
                per[ob]["liq_fee"] += H.fee_for(liq_px, lot["qty"])
    # reconciliation plug: liq_loss = engine_pnl - (capture + markout - fee)
    dec_total = sum(per[b]["capture"] + per[b]["markout"] - per[b]["fee"]
                    for b in H.BUCKETS)
    liq_gap = float(dr.pnl()) - dec_total
    tot_opened = sum(per[b]["opened_notional"] for b in H.BUCKETS)
    for b in H.BUCKETS:
        w = (per[b]["opened_notional"] / tot_opened) if tot_opened > 0 \
            else (1.0 / len(H.BUCKETS))
        per[b]["liq_loss"] += liq_gap * w
    # bundle
    return {"exit_ticks": exit_ticks, "obi_def": obi_def, "per": per,
            "daily_pnl": float(dr.pnl()), "date": str(date)}


def main():
    # stamp + timing
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    t0 = time.perf_counter()
    # calibration (once in parent)
    print("pre-pass: calibration + trailing median trade size", flush=True)
    all_dates = R.discover_dates()
    calib = {"scales": H.load_scales(), "profiles": H.load_profiles(),
             "windows": H.load_windows(), "segments": H.load_segments(),
             "all_dates": all_dates,
             "tstats": H.trailing_median_trade_size(all_dates, NAMES, TRAIL_DAYS)}
    # dates
    run_dates = all_dates[TRAIL_DAYS:]
    if SMOKE_DAYS is not None:
        run_dates = run_dates[:SMOKE_DAYS]
    # workers
    nproc = WORKERS or mp.cpu_count()
    # the 8 configs
    configs = [(et, od) for et in EXIT_TICKS for od in OBI_MODES]
    # work list: every (date, symbol, exit_ticks, obi_def)
    work = [(date, sym, et, od)
            for date in run_dates for sym in NAMES for (et, od) in configs]
    total = len(work)
    print(f"\n2D skew sweep: {len(EXIT_TICKS)} tick-levels x {len(OBI_MODES)} "
          f"OBI-modes = {len(configs)} configs", flush=True)
    print(f"  x {len(NAMES)} names x {len(run_dates)} days = {total} cells",
          flush=True)
    print(f"  workers: {nproc}\n", flush=True)
    # per-config accumulators keyed by (exit_ticks, obi_def)
    agg = {(et, od): {b: {"capture": 0.0, "markout": 0.0, "fee": 0.0,
                          "liq_fee": 0.0, "jump_markout": 0.0,
                          "diff_markout": 0.0, "liq_loss": 0.0,
                          "opened_notional": 0.0, "opened_qty": 0.0,
                          "holds": [], "fills": 0, "trades": 0, "shares": 0.0,
                          "obi5_raw_sum": 0.0, "obi5_signed_sum": 0.0,
                          "obi5_buy_sum": 0.0, "obi5_buy_n": 0,
                          "obi5_sell_sum": 0.0, "obi5_sell_n": 0, "obi_n": 0,
                          "ticks_inside": []}
                      for b in H.BUCKETS}
           for (et, od) in configs}
    # per-config daily portfolio markout for significance
    daily_mko = {(et, od): {} for (et, od) in configs}
    # run
    done = 0
    with mp.Pool(processes=nproc, initializer=_init_worker,
                 initargs=(calib,)) as pool:
        for res in pool.imap_unordered(_process, work, chunksize=4):
            done += 1
            if done % 200 == 0 or done == total:
                el = time.perf_counter() - t0
                print(f"  {done}/{total}  {H._fmt(el)}  "
                      f"ETA {H._fmt(el / done * (total - done))}", flush=True)
            if res is None:
                continue
            key = (res["exit_ticks"], res["obi_def"])
            for b in H.BUCKETS:
                s = res["per"][b]; d = agg[key][b]
                # sum scalar accumulators
                for k in ("capture", "markout", "fee", "liq_fee",
                          "jump_markout", "diff_markout", "liq_loss",
                          "opened_notional", "opened_qty", "fills", "trades",
                          "shares", "obi5_raw_sum", "obi5_signed_sum",
                          "obi5_buy_sum", "obi5_buy_n", "obi5_sell_sum",
                          "obi5_sell_n", "obi_n"):
                    d[k] += s[k]
                # extend lists
                d["holds"].extend(s["holds"])
                d["ticks_inside"].extend(s["ticks_inside"])
            # daily portfolio markout
            day = res["date"]
            dsum = sum(res["per"][b]["markout"] for b in H.BUCKETS)
            daily_mko[key][day] = daily_mko[key].get(day, 0.0) + dsum
    # ---- reporting ----
    print("\n" + "=" * 84)
    print(f"### 2D SKEW SWEEP DONE: {H._fmt(time.perf_counter() - t0)} "
          f"for {total} cells ###")
    print("=" * 84)
    # summary matrix: net-bps (portfolio) per (exit_ticks x obi_def)
    print("\n=== SUMMARY: portfolio net-bps by config (the decision matrix) ===")
    print(f"{'exit_ticks':>10s} {'obi_off_net_bps':>16s} {'obi_on_net_bps':>16s}")
    for et in EXIT_TICKS:
        row = {}
        for od in OBI_MODES:
            a = agg[(et, od)]
            net = sum(a[b]["capture"] + a[b]["markout"]
                      - (a[b]["fee"] + a[b]["liq_fee"])
                      + (a[b]["liq_loss"] + a[b]["liq_fee"]) for b in H.BUCKETS)
            on = sum(a[b]["opened_notional"] for b in H.BUCKETS)
            row[od] = (1e4 * net / on) if on > 0 else float("nan")
        print(f"{et:>10d} {row[False]:>16.3f} {row[True]:>16.3f}")
    # detailed per-config tables
    for (et, od) in configs:
        a = agg[(et, od)]
        print(f"\n{'=' * 84}")
        print(f"CONFIG: exit_ticks_inside={et}   obi_defensive={od}")
        print("=" * 84)
        # per-bucket detail
        print(f"{'bucket':11s} {'cap_bps':>8s} {'mko_bps':>8s} {'jmp_bps':>8s} "
              f"{'dif_bps':>8s} {'fee_bps':>8s} {'liq_bps':>8s} {'net_bps':>8s} "
              f"{'med_hld':>8s} {'mean_hld':>8s} {'med_tk_in':>9s} "
              f"{'trades':>8s} {'sh/trd':>8s}")
        for b in H.BUCKETS:
            d = a[b]
            on = d["opened_notional"]
            def _bps(x):
                return (1e4 * x / on) if on > 0 else float("nan")
            fee_all = d["fee"] + d["liq_fee"]
            liq_net = d["liq_loss"] + d["liq_fee"]
            net_bps = _bps(d["capture"] + d["markout"] - fee_all + liq_net)
            hs = np.array(d["holds"]) / 1000.0 if d["holds"] else np.array([0.0])
            tks = np.array(d["ticks_inside"]) if d["ticks_inside"] else np.array([0.0])
            sh_per = (d["shares"] / d["trades"]) if d["trades"] > 0 else 0.0
            print(f"{b:11s} {_bps(d['capture']):>8.3f} {_bps(d['markout']):>8.3f} "
                  f"{_bps(d['jump_markout']):>8.3f} {_bps(d['diff_markout']):>8.3f} "
                  f"{_bps(-fee_all):>8.3f} {_bps(liq_net):>8.3f} {net_bps:>8.3f} "
                  f"{np.median(hs):>8.1f} {hs.mean():>8.1f} {np.median(tks):>9.2f} "
                  f"{d['trades']:>8,d} {sh_per:>8.0f}")
        # OBI-at-fill: raw vs direction-signed, split by buy/sell
        print(f"\n{'bucket':11s} {'raw_obi5':>10s} {'signed_obi5':>12s} "
              f"{'buy_obi5':>10s} {'sell_obi5':>10s}  (signed<0 = book against us)")
        for b in H.BUCKETS:
            d = a[b]
            n = d["obi_n"]
            raw = (d["obi5_raw_sum"] / n) if n > 0 else float("nan")
            signed = (d["obi5_signed_sum"] / n) if n > 0 else float("nan")
            buy = (d["obi5_buy_sum"] / d["obi5_buy_n"]) if d["obi5_buy_n"] > 0 else float("nan")
            sell = (d["obi5_sell_sum"] / d["obi5_sell_n"]) if d["obi5_sell_n"] > 0 else float("nan")
            print(f"{b:11s} {raw:>10.4f} {signed:>12.4f} {buy:>10.4f} {sell:>10.4f}")
        # daily markout significance (day as unit)
        dm = pd.Series(daily_mko[(et, od)]).sort_index().to_numpy()
        if len(dm) >= 20:
            t_stat, t_p = stats.ttest_1samp(dm, 0.0, nan_policy="omit")
            dm_nz = dm[dm != 0]
            try:
                w_stat, w_p = stats.wilcoxon(dm_nz)
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
            print(f"\n  daily markout (n={len(dm)}): mean {dm.mean():,.0f}  "
                  f"median {np.median(dm):,.0f}  t={t_stat:.2f} p={t_p:.3g}  "
                  f"Wilcoxon p={w_p:.3g}")
    # save the summary matrix
    rows = []
    for (et, od) in configs:
        a = agg[(et, od)]
        net = sum(a[b]["capture"] + a[b]["markout"]
                  - (a[b]["fee"] + a[b]["liq_fee"])
                  + (a[b]["liq_loss"] + a[b]["liq_fee"]) for b in H.BUCKETS)
        on = sum(a[b]["opened_notional"] for b in H.BUCKETS)
        allhold = [h for b in H.BUCKETS for h in a[b]["holds"]]
        rows.append({"exit_ticks": et, "obi_defensive": od,
                     "net_bps": (1e4 * net / on) if on > 0 else np.nan,
                     "median_hold_s": (np.median(allhold) / 1000.0
                                       if allhold else np.nan),
                     "trades": sum(a[b]["trades"] for b in H.BUCKETS)})
    out = RESULTS / f"skew_sweep_2d_{stamp}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
