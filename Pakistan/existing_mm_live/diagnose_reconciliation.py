# diagnose_reconciliation.py -- find and dissect the worst-reconciling symbol-day
# so we can SEE where FIFO-attributed realized P&L diverges from the engine's own
# cash accounting. Scans the top-10 names, recomputes per-day reconciliation, then
# for the single worst day dumps: the engine's cash P&L decomposition, the FIFO
# match ledger, and the residual liquidation -- side by side.
#
# Run from existing_mm_live/:  caffeinate -is python3 diagnose_reconciliation.py

# filesystem paths
from pathlib import Path
# timing for the heartbeat
import time
# numeric arrays
import numpy as np
# dataframes
import pandas as pd
# the shared harness (driver + FIFO + loaders)
import mm_harness as H
# the driver module (dates + datasets)
import run_legacy_mm as R
# the engine fee function (to reconstruct the engine's own cash math)
from mm_backtest import fee_for

# ------------------------------ config ---------------------------------------
# the same top-10 the attribution ran on
NAMES = ["ENGROH", "LUCK", "UBL", "PSO", "PPL", "HBL", "SAZEW", "MLCF",
         "ATRL", "SYS"]
# production clip multiple
CLIP_MULT = 3.0
# trailing window for the clip anchor
TRAIL_DAYS = 10
# how many worst days to dump in detail
TOP_N_WORST = 3
# -----------------------------------------------------------------------------


# reconstruct the ENGINE's own daily P&L from fills + liquidation, independently
# of the FIFO attribution. This is cash accounting: every fill moves cash by
# -sign*qty*px minus its fee; residual inventory is closed at the liq price.
# It must equal bt.eod["equity_liquidated"]; if it doesn't, the mismatch is in
# how the harness reads the engine, not in FIFO. If it DOES equal the engine but
# FIFO doesn't, the bug is in FIFO. This isolates the two.
def engine_cash_pnl(fills, liq_px):
    # no fills -> zero traded cash (residual handled below is also zero)
    if len(fills) == 0:
        return 0.0, 0.0, 0.0
    # +1 for buys (cash out), -1 for sells (cash in)
    sgn = np.where(fills["side"].to_numpy() == "BUY", 1.0, -1.0)
    # shares per fill
    q = fills["qty"].to_numpy()
    # price per fill
    px = fills["px"].to_numpy()
    # cash from trading: sells add, buys subtract
    trade_cash = float(np.sum(-sgn * q * px))
    # fees on every fill (per-side)
    fees = float(np.sum([fee_for(p, x) for p, x in zip(px, q)]))
    # net signed position left open at end of day
    net_pos = float(np.sum(sgn * q))
    # close the residual at the liquidation price (buy back shorts / sell longs)
    resid_cash = -(-net_pos) * liq_px if False else net_pos * liq_px
    # residual liquidation fee
    resid_fee = fee_for(liq_px, abs(net_pos)) if net_pos != 0 else 0.0
    # total engine-style P&L = trade cash + residual close - all fees
    total = trade_cash + resid_cash - fees - resid_fee
    # return components for the dissection
    return total, trade_cash + resid_cash, fees + resid_fee


def main():
    # calibration via the harness
    scales = H.load_scales()
    profiles = H.load_profiles()
    windows = H.load_windows()
    segments = H.load_segments()
    # all trading dates
    all_dates = R.discover_dates()

    # trailing median trade size pre-pass
    print("pre-pass: trailing median trade size", flush=True)
    tstats = H.trailing_median_trade_size(all_dates, NAMES, TRAIL_DAYS)

    # trading window
    run_dates = all_dates[TRAIL_DAYS:]
    # per-symbol-day reconciliation records: (abs_err, date, sym, details)
    recs = []
    # scan timer
    t0 = time.perf_counter()
    # counter
    sd = 0
    total = len(run_dates) * len(NAMES)
    # scan every symbol-day, recomputing reconciliation
    for date in run_dates:
        # datasets for the date
        dsets = R.open_datasets(date)
        # skip missing partitions
        if dsets is None:
            continue
        # session segments
        segs = segments.get(str(date))
        # skip days without segments
        if segs is None:
            continue
        # each symbol
        for sym in NAMES:
            # heartbeat counter
            sd += 1
            # the clip anchor
            med = H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
            # skip on missing history/calibration
            if med is None or med <= 0 or sym not in scales or sym not in profiles:
                continue
            # the production clip
            clip = max(1, int(round(CLIP_MULT * med)))
            # locked production params
            params = H.build_micro_params(clip, scales[sym], profiles[sym],
                                          windows.get(sym, (5.0, 1.0)), segs)
            # the single backtest path
            dr = H.run_symbol_day(date, sym, dsets, params)
            # skip unrunnable days
            if dr is None or dr.pnl() is None:
                continue
            # the true liquidation price
            liq_px = H.liquidation_price(dr)
            # FIFO attribution total
            per = H.fifo_attribution(dr.fills.to_dict("records"), liq_px,
                                     dr.session[1])
            # attributed sum across buckets
            attr = sum(per[b]["realized"] for b in H.BUCKETS)
            # engine headline P&L
            eng = float(dr.pnl())
            # independent cash reconstruction (isolates harness-read vs FIFO bug)
            cash_pnl, gross_cash, all_fees = engine_cash_pnl(dr.fills, liq_px)
            # the reconciliation error (FIFO attributed vs engine headline)
            err = attr - eng
            # store the record with everything needed to dissect
            recs.append({"abs_err": abs(err), "err": err, "date": str(date),
                         "sym": sym, "attr": attr, "engine": eng,
                         "cash_recon": cash_pnl, "liq_px": liq_px,
                         "n_fills": len(dr.fills),
                         "eod_inv": dr.eod.get("pos_at_close"),
                         "liq_clean": dr.eod.get("liquidation_clean"),
                         "unfilled": dr.eod.get("unfilled_sh"),
                         # keep the fills + per for the deep dump on worst days
                         "_fills": dr.fills, "_per": per, "_dr": dr})
            # heartbeat
            if sd % 20 == 0 or sd == total:
                el = time.perf_counter() - t0
                print(f"  {sd}/{total}  {H._fmt(el)}  "
                      f"ETA {H._fmt(el/sd*(total-sd))}", flush=True)

    # sort by absolute reconciliation error, worst first
    recs.sort(key=lambda r: r["abs_err"], reverse=True)

    # ---- summary of reconciliation health ----
    errs = np.array([r["abs_err"] for r in recs])
    print(f"\n=== RECONCILIATION SCAN: {len(recs)} symbol-days ===")
    print(f"  mean abs err: {errs.mean():.4f} PKR")
    print(f"  median abs err: {np.median(errs):.4f} PKR")
    print(f"  days over 1 PKR: {int((errs > 1.0).sum())} "
          f"({100*(errs>1.0).mean():.1f}%)")
    print(f"  days over 100 PKR: {int((errs > 100).sum())}")
    print(f"  max abs err: {errs.max():.2f} PKR")

    # ---- dissect the worst TOP_N_WORST days ----
    for r in recs[:TOP_N_WORST]:
        print("\n" + "=" * 68)
        print(f"WORST DAY: {r['sym']} {r['date']}  |  err {r['err']:+,.2f} PKR")
        print("=" * 68)
        # the three P&L numbers that should all agree
        print(f"  engine equity_liquidated : {r['engine']:>14,.2f}")
        print(f"  FIFO attributed total    : {r['attr']:>14,.2f}  "
              f"(err vs engine {r['err']:+,.2f})")
        print(f"  independent cash recon   : {r['cash_recon']:>14,.2f}  "
              f"(err vs engine {r['cash_recon']-r['engine']:+,.2f})")
        # this triangulation tells us WHERE the bug is:
        print("\n  DIAGNOSIS:")
        # if cash recon matches engine but FIFO doesn't -> bug is in FIFO
        if abs(r["cash_recon"] - r["engine"]) < 1.0 and r["abs_err"] > 1.0:
            print("    cash recon MATCHES engine, FIFO does NOT -> bug is in FIFO matching")
        # if cash recon also fails -> the harness reads the engine wrong (liq px etc)
        elif abs(r["cash_recon"] - r["engine"]) > 1.0:
            print("    cash recon ALSO fails -> harness misreads engine (liq px / eod path),")
            print("    NOT a FIFO bug. Likely liq_vwap != the price the engine actually used,")
            print("    or a multi-segment / unclean-liquidation day.")
        # day context that usually explains the outliers
        print(f"\n  day context:")
        print(f"    n_fills           : {r['n_fills']}")
        print(f"    eod inventory     : {r['eod_inv']}")
        print(f"    liquidation clean : {r['liq_clean']}")
        print(f"    unfilled shares   : {r['unfilled']}")
        print(f"    liq price used    : {r['liq_px']}")
        # per-bucket attributed (to see if one bucket carries the error)
        print(f"\n  FIFO per-bucket realized:")
        for b in H.BUCKETS:
            print(f"    {b:12s} {r['_per'][b]['realized']:>12,.2f}  "
                  f"(fills {r['_per'][b]['fills']}, "
                  f"opened_qty {r['_per'][b]['opened_qty']:.0f})")
        # net signed position from fills (should equal eod inventory pre-liq)
        f = r["_fills"]
        sgn = np.where(f["side"].to_numpy() == "BUY", 1.0, -1.0)
        net = float(np.sum(sgn * f["qty"].to_numpy()))
        print(f"\n  fills net position: {net:.0f}  (engine pos_at_close: {r['eod_inv']})")
        # if these disagree, the fills the harness sees != the fills the engine
        # booked -> the discrepancy is upstream of FIFO
        if r["eod_inv"] is not None and abs(net - float(r["eod_inv"])) > 1e-6:
            print("    ** MISMATCH: fills net != engine close position -> the harness")
            print("    ** is not seeing the same fills the engine used for P&L")

    print("\nread the DIAGNOSIS lines: they localize the bug to FIFO vs engine-read.")


if __name__ == "__main__":
    main()
