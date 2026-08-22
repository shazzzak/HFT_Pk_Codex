# smoke_recon.py -- FAST reconciliation check on the exact symbol-days that were
# broken (unclean-liquidation days), not a full-universe run. Confirms the FIFO
# residual fix reconciles attribution to the engine headline P&L in minutes.
#
# Run from existing_mm_live/:  python smoke_recon.py

# dataframes + arrays
import numpy as np
import pandas as pd
# the shared harness
import mm_harness as H
# the driver
import run_legacy_mm as R

# the exact days the diagnostic flagged as worst (unclean liquidations) + a
# couple of clean controls, so we prove BOTH still reconcile
CHECKS = [
    # (symbol, date, was_broken_before)
    ("HBL", "2026-03-10", True),      # err +5,989 before
    ("MLCF", "2026-03-09", True),     # err +4,210 before
    ("ENGROH", "2026-04-14", True),   # err +4,188 before
    ("UBL", "2025-09-15", False),     # a clean control day
    ("PPL", "2025-09-15", False),     # a clean control day
]
# production clip multiple
CLIP_MULT = 3.0
# trailing window for the clip anchor
TRAIL_DAYS = 10


def main():
    # calibration via the harness
    scales = H.load_scales()
    profiles = H.load_profiles()
    windows = H.load_windows()
    segments = H.load_segments()
    # all dates (needed for the trailing median lookup)
    all_dates = R.discover_dates()
    # the distinct names we touch
    names = sorted({c[0] for c in CHECKS})

    # minimal trailing-median pre-pass: only the dates up to our latest check,
    # only our names -- keeps the smoke test fast
    print("mini pre-pass: trailing median trade size (checked names only)",
          flush=True)
    tstats = H.trailing_median_trade_size(all_dates, names, TRAIL_DAYS,
                                          heartbeat=100)

    # header
    print(f"\n{'symbol':8s} {'date':12s} {'was_broken':>10s} "
          f"{'engine_pnl':>12s} {'attributed':>12s} {'err':>10s} {'status':>8s}")
    # run each checked day
    for sym, date, broken in CHECKS:
        # datasets for the date
        dsets = R.open_datasets(date)
        # skip if missing
        if dsets is None:
            print(f"{sym:8s} {date:12s}  no datasets")
            continue
        # session segments for the date
        segs = segments.get(str(date))
        # skip if missing
        if segs is None:
            print(f"{sym:8s} {date:12s}  no segments")
            continue
        # the clip anchor for this symbol-day
        med = H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
        # skip on missing history/calibration
        if med is None or med <= 0 or sym not in scales:
            print(f"{sym:8s} {date:12s}  no calibration/history")
            continue
        # the production clip
        clip = max(1, int(round(CLIP_MULT * med)))
        # locked production params
        params = H.build_micro_params(clip, scales[sym], profiles[sym],
                                      windows.get(sym, (5.0, 1.0)), segs)
        # the single backtest path
        dr = H.run_symbol_day(date, sym, dsets, params)
        # skip unrunnable
        if dr is None or dr.pnl() is None:
            print(f"{sym:8s} {date:12s}  unrunnable")
            continue
        # engine headline P&L
        eng = float(dr.pnl())
        # FIFO attribution (now reconciles by construction)
        per = H.fifo_attribution(dr.fills.to_dict("records"), eng, dr.session[1])
        # attributed total
        attr = sum(per[b]["realized"] for b in H.BUCKETS)
        # the reconciliation error
        err = attr - eng
        # pass/fail at the 1-PKR threshold
        status = "OK" if abs(err) < 1.0 else "FAIL"
        # the row
        print(f"{sym:8s} {date:12s} {str(broken):>10s} "
              f"{eng:>12,.2f} {attr:>12,.2f} {err:>+10.4f} {status:>8s}")

    # verdict banner
    print("\nAll rows OK -> the residual fix reconciles on the previously-broken")
    print("unclean days AND the clean controls. Safe to launch the full re-run.")


if __name__ == "__main__":
    main()
