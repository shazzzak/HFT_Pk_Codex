# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# trades_by_phase.py -- is AFTER_HOUR_TRADING a real exit venue or just quote churn?
# Decides how the corrected EOD flatten should mark: if after-hours has genuine
# fixed-price executions, the realistic model flattens INTO after-hours at the
# closing price; if it's churn only, the continuous-phase end is a hard wall.
#
# Reads the TRADES table (executions, not messages) and buckets them by the
# snapshot phase in effect at each trade's time, per symbol-day. Also reports the
# distinct trade prices per phase (after-hours should be ~one price if fixed-price).
#
# Run from existing_mm_live/:  python trades_by_phase.py

# filesystem paths
from pathlib import Path
# wall-clock timing
import time
# dataframes
import pandas as pd
# numeric helpers
import numpy as np
# driver module
import run_legacy_mm as R

# point the driver at the parsed data root
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# symbols under study
SYMBOLS = ["PPL", "UBL"]
# how many days to sample (a handful is plenty to characterise phases); None = all
DAYS_LIMIT = 15


# format seconds as mm:ss
def _fmt(sec):
    # minutes and zero-padded seconds
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# for each trade, find the phase in effect by as-of joining to the snapshot phase timeline
def label_trades_with_phase(t, s):
    # snapshot phase timeline: one row per snapshot message with its time + phase
    # keep only rows that actually carry a phase string
    ph = s[["ts_exch", "phase"]].dropna(subset=["phase"]).copy()
    # nothing to join against -> return trades with unknown phase
    if len(ph) == 0:
        # mark every trade's phase as unknown
        t = t.copy()
        # unknown phase label
        t["phase"] = "UNKNOWN"
        return t
    # sort both by time for the as-of merge
    ph = ph.sort_values("ts_exch")
    # trades sorted by time
    tt = t.sort_values("ts_exch").copy()
    # as-of join: each trade gets the most recent phase at or before its time
    merged = pd.merge_asof(tt, ph, on="ts_exch", direction="backward")
    # trades before the first phase row get NaN -> label UNKNOWN
    merged["phase"] = merged["phase"].fillna("UNKNOWN")
    return merged


# main driver
def main():
    # all dates, optionally truncated for a quick pass
    dates = R.discover_dates()
    # truncate if requested
    if DAYS_LIMIT is not None:
        dates = dates[:DAYS_LIMIT]
    # accumulator: list of per-(symbol, phase) trade summaries
    rows = []
    # timer + counter
    t0_all = time.perf_counter()
    sd = 0
    # announce
    print(f"trades_by_phase: {len(SYMBOLS)} symbols x {len(dates)} days\n", flush=True)
    # OUTER over dates
    for date in dates:
        # open datasets once
        dsets = R.open_datasets(date)
        # skip missing
        if dsets is None:
            continue
        # MIDDLE over symbols
        for sym in SYMBOLS:
            # read snapshots (for the phase timeline) and trades (the executions)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # skip if either is empty
            if len(s) == 0 or len(t) == 0:
                continue
            # attach exchange-ms timestamps the same way build_events does
            s = s.copy()
            # snapshot time from orig_time
            s["ts_exch"] = R.to_ms(s["orig_time"])
            # trades copy
            t = t.copy()
            # trade time from transact_time
            t["ts_exch"] = R.to_ms(t["transact_time"])
            # label each trade with the phase in effect at its time
            tl = label_trades_with_phase(t, s)
            # price column on trades is 'px' (fall back to 'price' if named that way)
            pxcol = "px" if "px" in tl.columns else ("price" if "price" in tl.columns else None)
            # summarise per phase for this symbol-day
            for phase, g in tl.groupby("phase"):
                # number of executions in this phase
                n = len(g)
                # distinct prices (a fixed-price session shows very few)
                ndist = g[pxcol].nunique() if pxcol else np.nan
                # record the summary row
                rows.append({
                    "symbol": sym, "date": str(date), "phase": str(phase),
                    "n_trades": n, "distinct_px": ndist,
                })
            # progress
            sd += 1
            # heartbeat every 10 symbol-days
            if sd % 10 == 0:
                # elapsed
                el = time.perf_counter() - t0_all
                # print progress
                print(f"  {sd} symbol-days  elapsed {_fmt(el)}", flush=True)

    # assemble
    df = pd.DataFrame(rows)
    # guard
    if len(df) == 0:
        print("no trades summarised"); return

    # ---- aggregate across days, per (symbol, phase) ----
    # total executions and typical distinct-price count per phase
    agg = df.groupby(["symbol", "phase"]).agg(
        # total trades in this phase across sampled days
        total_trades=("n_trades", "sum"),
        # days on which this phase had any trade
        days_with_trades=("n_trades", lambda x: int((x > 0).sum())),
        # median distinct prices per day in this phase (1-2 => fixed-price)
        median_distinct_px=("distinct_px", "median"),
    ).reset_index()
    # order phases by trade volume within each symbol
    agg = agg.sort_values(["symbol", "total_trades"], ascending=[True, False])
    # full-width print
    pd.set_option("display.width", 200)
    # show the table
    print("\n--- executions by phase (summed over sampled days) ---")
    print(agg.to_string(index=False))

    # the decision text
    print("\nDECIDES:")
    print("  CONTINUOUS_AUCTION should hold the bulk of trades (normal matching).")
    print("  If AFTER_HOUR_TRADING has meaningful total_trades AND median_distinct_px")
    print("  ~1-2, it's a REAL fixed-price post-close session -> the corrected flatten")
    print("  should try to clear at the closing price there, then mark the rest.")
    print("  If AFTER_HOUR_TRADING trades are ~0, it's quote churn only -> continuous")
    print("  phase-end is a hard wall and the flatten must mark/liquidate there.")


if __name__ == "__main__":
    main()
