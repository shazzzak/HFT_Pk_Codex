# calibrate_time_windows.py -- per-name EOD unwind windows from a PARTICIPATION cap.
#
# The rule (optimal-execution style, replaces the old fixed 5min/1min):
#   Minutes_Needed = max_inv / (closing_vol_per_min x MAX_POV)
# i.e. start the unwind exactly early enough that clearing max inventory never
# exceeds MAX_POV of the market's natural closing volume. Liquid names come out
# near the 5-min floor; thin names get structurally longer windows -- no hand-tuned
# multiplier, the one dial (MAX_POV) is an interpretable participation policy.
#
# closing_vol_per_min = median over ALL sample days of (shares traded in the last
# CLOSING_WIN_MIN minutes of the session) / CLOSING_WIN_MIN. Production version
# recomputes this daily over a trailing 20 days (a daily batch job); for the
# backtest a static per-name table is the agreed design (robust, no intraday
# Poisson noise -- a live mid-day estimate on a thin name can collapse to ~zero
# and tell you to start flattening at 10am).
#
# max_inv is per-name = MAXINV_CLIPS x (median trade size), matching the universe
# runner's sizing. Output: timestamped CSV symbol -> ramp/cliff minutes + inputs,
# plus a paste-ready dict.
#
# Run from existing_mm_live/:  python calibrate_time_windows.py

# paths
from pathlib import Path
# timing + stamp
import time
from datetime import datetime
# frames + arrays
import pandas as pd
import numpy as np
# driver (datasets, reads)
import run_legacy_mm as R
# reuse the mm:ss formatter
import confirm_micro_vs_naive as C

# raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# shortlist + output dir
WATCHLIST = Path("/Users/shazzak/Capital Stake - Results/mm_watchlist_final.csv")
OUT_DIR = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ policy knobs ---------------------------------
# maximum participation of the closing volume we allow our unwind to be.
# Sweep 0.05..0.20; 0.10 = "never more than 10% of the tape while unwinding".
MAX_POV = 0.10
# the closing window we measure natural volume over (minutes before the last trade)
CLOSING_WIN_MIN = 30
# inventory cap in clips (matches the universe runner: max_inv = 10 x clip)
MAXINV_CLIPS = 10.0
# ramp floor/cap in minutes: never tighter than the old 5-min default; never wider
# than 60 (a name needing >60min at 10% POV is a capacity problem, not a window one)
RAMP_MIN, RAMP_MAX = 5.0, 60.0
# cliff = ramp/5 (preserves today's 5:1 shape), floored at the 1-min hard stop
CLIFF_RATIO, CLIFF_MIN = 5.0, 1.0
# ------------------------------------------------------------------------------


def main():
    # run stamp for the output filename
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # shortlist symbols
    syms = pd.read_csv(WATCHLIST)["symbol"].tolist()
    # all dates
    dates = R.discover_dates()
    print(f"calibrate_time_windows: {len(syms)} names x {len(dates)} days, "
          f"MAX_POV={MAX_POV:.0%}, closing window {CLOSING_WIN_MIN}min\n", flush=True)

    # per-symbol accumulators: daily closing-window volumes + all trade sizes' medians
    close_vol = {s: [] for s in syms}   # shares traded in the last CLOSING_WIN_MIN, per day
    med_trade = {s: [] for s in syms}   # daily median trade size, per day
    # timer
    t0 = time.perf_counter()
    # one pass over days (trades table only -- cheap)
    for i, date in enumerate(dates, 1):
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        for sym in syms:
            # this name's trades (uses 'price'/'qty'; ts from transact_time)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0:
                continue
            # exchange-ms timestamps
            ts = R.to_ms(t["transact_time"])
            # session close proxy = the last trade's time
            t_close = ts.max()
            # shares traded in the closing window [close - WIN, close]
            in_win = t.loc[ts >= (t_close - CLOSING_WIN_MIN * 60_000), "qty"].sum()
            close_vol[sym].append(float(in_win))
            # daily median trade size (for the per-name max_inv)
            med_trade[sym].append(float(t["qty"].median()))
        # heartbeat
        if i % 50 == 0 or i == len(dates):
            print(f"  {i}/{len(dates)} days  elapsed {C._fmt(time.perf_counter() - t0)}",
                  flush=True)

    # ---- derive the windows per name ----
    rows = []
    for sym in syms:
        # need data
        if not close_vol[sym]:
            rows.append({"symbol": sym, "note": "NO_TRADES"})
            continue
        # median closing-window volume (shares) across days -> per-minute rate
        med_close_vol = float(np.median(close_vol[sym]))
        vol_per_min = med_close_vol / CLOSING_WIN_MIN
        # per-name inventory cap: 10 x the median trade size (matches the runner)
        med_sz = float(np.median(med_trade[sym]))
        max_inv = MAXINV_CLIPS * med_sz
        # degenerate guard: a closing window with ~no volume -> cap the window
        if vol_per_min <= 0:
            minutes = RAMP_MAX
        else:
            # THE RULE: minutes needed at the participation cap
            minutes = max_inv / (vol_per_min * MAX_POV)
        # clip into policy bounds
        ramp = float(np.clip(minutes, RAMP_MIN, RAMP_MAX))
        # cliff preserves the 5:1 shape, floored at the 1-min hard stop
        cliff = float(max(CLIFF_MIN, ramp / CLIFF_RATIO))
        # a capacity flag: names whose UNCLIPPED need exceeds the cap are names
        # where even a 10%-POV unwind cannot clear max_inv in an hour -- reduce
        # max_inv there rather than pretending a longer window fixes it
        rows.append({"symbol": sym, "eod_ramp_start_min": round(ramp, 1),
                     "eod_cliff_min": round(cliff, 1),
                     "minutes_needed_raw": round(minutes, 1),
                     "med_close_vol_sh": round(med_close_vol, 0),
                     "vol_per_min_sh": round(vol_per_min, 0),
                     "max_inv_sh": round(max_inv, 0),
                     "capacity_flag": ("EXCEEDS_CAP" if minutes > RAMP_MAX else "ok"),
                     "note": "ok"})

    # write the table
    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"time_windows_{stamp}.csv"
    out.to_csv(out_csv, index=False)

    # ---- report ----
    ok = out[out.note == "ok"].sort_values("eod_ramp_start_min", ascending=False)
    print(f"\n{'symbol':8s} {'ramp_min':>8s} {'cliff':>6s} {'raw_need':>9s} "
          f"{'vol/min':>8s} {'max_inv':>8s} {'flag':>12s}")
    for _, r in ok.iterrows():
        print(f"{r.symbol:8s} {r.eod_ramp_start_min:>8.1f} {r.eod_cliff_min:>6.1f} "
              f"{r.minutes_needed_raw:>9.1f} {r.vol_per_min_sh:>8,.0f} "
              f"{r.max_inv_sh:>8,.0f} {r.capacity_flag:>12s}")
    # capacity warnings up front
    bad = ok[ok.capacity_flag != "ok"]
    if len(bad):
        print(f"\n!!! {len(bad)} names cannot clear max_inv within {RAMP_MAX:.0f}min at "
              f"{MAX_POV:.0%} POV: {list(bad.symbol)}")
        print("    -> for these, REDUCE max_inv (fewer clips) rather than widening further.")
    print(f"\nwrote {out_csv}")
    # paste-ready dict for the runner
    print("\nTIME_WINDOWS = {")
    for _, r in ok.iterrows():
        print(f'    "{r.symbol}": ({r.eod_ramp_start_min}, {r.eod_cliff_min}),')
    print("}")


if __name__ == "__main__":
    main()
