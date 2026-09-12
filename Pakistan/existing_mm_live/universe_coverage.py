# ============================================================================
# universe_coverage.py -- STEP 1: raw-data coverage for the expanded universe
# ============================================================================
# Read-only. Creates one new CSV. Touches nothing else.
#
# WHY: universe_expand.py dispatched 44,916 cells and 76 of the 114 names
# produced zero rows, silently, because _one() does
#     if sym not in scales or sym not in profiles: return None
# Before generating calibration for those 76, establish which of them are
# even RUNNABLE against the raw store. Calibrating a name that has no
# continuous-auction phase or no trades is wasted work twice over.
#
# Runnability is defined by the engine itself, not by guesswork. From
# mm_harness.run_symbol_day():
#     len(t) == 0 or len(s) == 0                      -> return None
#     s[s["phase"] == "CONTINUOUS_AUCTION"] is empty  -> return None
# and from fullyear_confirm._one():
#     trailing_median(...) is None or <= 0            -> return None
#       (needs >= TRAIL_DAYS prior days WITH trades)
# This script measures exactly those four conditions, per symbol, per day.
#
# OUTPUT: universe_coverage_<stamp>.csv in the results dir, one row per symbol.
# ============================================================================

# filesystem paths
from pathlib import Path
# run timestamp for the output filename
from datetime import datetime
# wall-clock timing for the heartbeat
import time
# numeric arrays
import numpy as np
# dataframes
import pandas as pd

# the driver module: open_datasets, read_symbol, discover_dates, REQ_* specs
import run_legacy_mm as R
# the shared harness: pulls PARSED_ROOT/RESULTS from config_pk and pushes onto R
import mm_harness as H

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# results dir comes from the harness, which sources it from config_pk --
# never hardcoded, so this follows the machine like every other runner
RESULTS = H.RESULTS
# trailing window fullyear_confirm requires before a day is tradeable
TRAIL_DAYS = 10
# heartbeat every N dates so a long pass shows progress
HEARTBEAT = 25

# the 114-name expanded universe (same list as universe_expand.py NAMES)
NAMES = ['AGHA', 'AGP', 'AHCL', 'AICL', 'AIRLINK', 'AKBL', 'APL', 'ASL',
         'ATRL', 'AVN', 'BAFL', 'BAHL', 'BBFL', 'BECO', 'BFBIO',
         'BML', 'BNL', 'BOP', 'CEPB', 'CHCC', 'CNERGY', 'CPHL',
         'CSAP', 'DCL', 'DFML', 'DGKC', 'EFERT', 'ENGROH', 'EPCL',
         'FABL', 'FATIMA', 'FCCL', 'FCEPL', 'FCL', 'FECTC', 'FFC',
         'FFL', 'FNEL', 'GAL', 'GCIL', 'GCWL', 'GGL', 'GHNI',
         'GLAXO', 'HALEON', 'HASCOL', 'HBL', 'HCAR', 'HMB',
         'HUBC', 'HUMNL', 'ILP', 'IMAGE', 'ISL', 'JVDC', 'KAPCO',
         'KEL', 'KOHC', 'KOIL', 'KOSM', 'LCI', 'LOADS', 'LOTCHEM',
         'LUCK', 'MARI', 'MCB', 'MEBL', 'MLCF', 'MTL', 'MUGHAL',
         'NATF', 'NBP', 'NCPL', 'NETSOL', 'NML', 'NPL', 'NRL',
         'OGDC', 'PACE', 'PAEL', 'PIAHCLA', 'PIBTL', 'PIOC',
         'POL', 'POWER', 'PPL', 'PREMA', 'PRL', 'PSO', 'PSX',
         'PTC', 'QUICE', 'SAZEW', 'SEARL', 'SGF', 'SGPL', 'SLGL',
         'SNGP', 'SSGC', 'SYS', 'TBL', 'TELE', 'TGL', 'THCCL',
         'TOMCL', 'TPL', 'TPLP', 'TREET', 'TRG', 'UBL', 'UNITY',
         'WAVES', 'WTL', 'ZAL']

# the 38 names already in the production book, flagged in the output so the
# new names can be read against a known-good baseline in the same table
INCUMBENT = {'AKBL', 'ATRL', 'BAFL', 'BOP', 'DGKC', 'ENGROH', 'FFC', 'FNEL',
             'HASCOL', 'HBL', 'HUBC', 'KEL', 'LUCK', 'MARI', 'MEBL', 'MLCF',
             'NBP', 'NCPL', 'NML', 'NPL', 'NRL', 'OGDC', 'PACE', 'PAEL',
             'PIAHCLA', 'PIBTL', 'PIOC', 'PPL', 'PSO', 'PTC', 'SAZEW',
             'SEARL', 'SYS', 'THCCL', 'TOMCL', 'TPL', 'TRG', 'UBL'}


# create a path that does not exist, so nothing is ever overwritten
def safe_out(stem, ext):
    # this run's timestamp, matching the house naming convention
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    # the candidate path
    p = RESULTS / f"{stem}_{ts}.{ext}"
    # disambiguating counter, only used on a same-minute collision
    n = 1
    # walk until the name is free
    while p.exists():
        # append the counter
        p = RESULTS / f"{stem}_{ts}_{n}.{ext}"
        # advance it
        n += 1
    # a path guaranteed not to exist
    return p


def main():
    # start the wall clock for the heartbeat
    t0 = time.perf_counter()
    # every trading date in the parsed store
    all_dates = R.discover_dates()
    # announce the scope so the log is self-documenting
    print(f"coverage pass: {len(NAMES)} names x {len(all_dates)} dates", flush=True)
    # report which store is being read, so a stale config_pk is visible
    print(f"  parsed store : {R.PARSED_ROOT}", flush=True)
    # report where the output will land
    print(f"  results dir  : {RESULTS}", flush=True)

    # per-symbol accumulators, one dict per name
    acc = {s: {"days_trades": 0, "days_snap": 0, "days_continuous": 0,
               "days_runnable": 0, "trade_counts": [], "med_qty": [],
               "med_price": [], "runnable_dates": []} for s in NAMES}

    # walk every date once, opening the day's datasets a single time
    for i, date in enumerate(all_dates, 1):
        # the day's parquet datasets (trades / ob_snapshot / ob_updates)
        dsets = R.open_datasets(date)
        # a missing partition is a dead date for every symbol
        if dsets is None:
            # heartbeat still fires below; nothing to accumulate
            continue
        # each symbol's slice of this date
        for s in NAMES:
            # the symbol's trades for the day, via the same reader the engine uses
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, s)
            # the symbol's book snapshots for the day
            snap = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, s)
            # did the symbol trade at all?
            has_t = len(t) > 0
            # did the symbol have a book at all?
            has_s = len(snap) > 0
            # count the day against each table's coverage
            if has_t:
                acc[s]["days_trades"] += 1
            if has_s:
                acc[s]["days_snap"] += 1
            # the engine needs a CONTINUOUS_AUCTION phase to derive session bounds
            has_cont = has_s and (snap["phase"] == "CONTINUOUS_AUCTION").any()
            # count the day against continuous-phase coverage
            if has_cont:
                acc[s]["days_continuous"] += 1
            # a day is runnable iff trades AND book AND a continuous phase exist
            if has_t and has_s and has_cont:
                # count it
                acc[s]["days_runnable"] += 1
                # remember the date, so the trailing-window test below is exact
                acc[s]["runnable_dates"].append(str(date))
                # the day's trade count (liquidity signal)
                acc[s]["trade_counts"].append(len(t))
                # the day's median trade size -- the clip driver in _one()
                acc[s]["med_qty"].append(float(t["qty"].median()))
                # the day's median trade price -- needed by the scale calibration
                acc[s]["med_price"].append(float(t["price"].median()))
        # progress heartbeat on the cadence, and always on the final date
        if i % HEARTBEAT == 0 or i == len(all_dates):
            # elapsed wall clock in mm:ss
            print(f"  {i}/{len(all_dates)} dates  "
                  f"{time.perf_counter() - t0:,.0f}s", flush=True)

    # assemble one output row per symbol
    rows = []
    # walk the names in a stable order
    for s in NAMES:
        # this symbol's accumulator
        a = acc[s]
        # how many days had a median trade size recorded (the trailing input)
        n_qty = len(a["med_qty"])
        # fullyear_confirm drops the first TRAIL_DAYS dates, then requires
        # TRAIL_DAYS of PRIOR days with trades before any day is tradeable.
        # So the number of days that can actually produce a cell is at most
        # the runnable days beyond the warm-up.
        tradeable_days = max(0, n_qty - TRAIL_DAYS)
        # the verdict, using the engine's own conditions in priority order
        if a["days_trades"] == 0:
            verdict = "NO_TRADES"
        elif a["days_snap"] == 0:
            verdict = "NO_BOOK"
        elif a["days_continuous"] == 0:
            verdict = "NO_CONTINUOUS_PHASE"
        elif tradeable_days <= 0:
            verdict = "INSUFFICIENT_HISTORY"
        elif tradeable_days < 100:
            verdict = "THIN_HISTORY"
        else:
            verdict = "RUNNABLE"
        # one row per symbol
        rows.append({
            # the symbol
            "symbol": s,
            # is it already in the production book (a known-good control row)
            "incumbent": s in INCUMBENT,
            # the verdict driving what happens next
            "verdict": verdict,
            # days the symbol traded
            "days_trades": a["days_trades"],
            # days the symbol had a book
            "days_snap": a["days_snap"],
            # days with a continuous-auction phase
            "days_continuous": a["days_continuous"],
            # days meeting all three engine preconditions
            "days_runnable": a["days_runnable"],
            # days that survive the TRAIL_DAYS warm-up
            "days_tradeable_est": tradeable_days,
            # typical daily trade count (liquidity)
            "median_trades_per_day": (float(np.median(a["trade_counts"]))
                                      if a["trade_counts"] else np.nan),
            # typical daily median trade size -- clip = 3.0 x this
            "median_trade_qty": (float(np.median(a["med_qty"]))
                                 if a["med_qty"] else np.nan),
            # implied production clip at CLIP_MULT = 3.0
            "implied_clip": (int(round(3.0 * float(np.median(a["med_qty"]))))
                             if a["med_qty"] else np.nan),
            # typical price -- the session_scale back-solve needs it
            "median_price": (float(np.median(a["med_price"]))
                             if a["med_price"] else np.nan),
            # first and last runnable date, to spot listings and delistings
            "first_runnable": (a["runnable_dates"][0]
                               if a["runnable_dates"] else ""),
            "last_runnable": (a["runnable_dates"][-1]
                              if a["runnable_dates"] else ""),
        })
    # the assembled table
    df = pd.DataFrame(rows)

    # ---- console summary -----------------------------------------------------
    # blank line before the summary block
    print("\n" + "=" * 74)
    # headline: how many names clear every engine precondition
    print("VERDICT COUNTS (all 114)")
    # counts by verdict
    print(df["verdict"].value_counts().to_string())
    # the same split for the 76 names that have never run
    print("\nVERDICT COUNTS (the 76 NEW names only)")
    # filter to non-incumbents
    print(df[~df["incumbent"]]["verdict"].value_counts().to_string())
    # the incumbents are the control: they MUST all come back RUNNABLE
    bad_incumbent = df[df["incumbent"] & (df["verdict"] != "RUNNABLE")]
    # a non-runnable incumbent means this script is wrong, not the data
    if len(bad_incumbent):
        print("\n*** CONTROL FAILURE: incumbent names not marked RUNNABLE.")
        print("    The 38 production names all ran last night, so a non-RUNNABLE")
        print("    verdict here means THIS SCRIPT's test is wrong. Do not act on")
        print("    the new-name verdicts until this is resolved.")
        print(bad_incumbent[["symbol", "verdict", "days_trades", "days_snap",
                             "days_continuous", "days_tradeable_est"]]
              .to_string(index=False))
    else:
        # the control passed, so the new-name verdicts can be trusted
        print(f"\ncontrol OK: all {int(df['incumbent'].sum())} incumbent names "
              f"verdict=RUNNABLE")

    # the new names that are good to calibrate
    ready = df[(~df["incumbent"]) & (df["verdict"] == "RUNNABLE")]
    # the new names that are not worth calibrating
    drop = df[(~df["incumbent"]) & (df["verdict"] != "RUNNABLE")]
    # report the split plainly
    print(f"\nNEW names ready to calibrate : {len(ready)}")
    print(f"NEW names to drop            : {len(drop)}")
    # name the ones being dropped and why
    if len(drop):
        print(drop[["symbol", "verdict", "days_trades", "days_continuous",
                    "days_tradeable_est"]].to_string(index=False))

    # ---- write ---------------------------------------------------------------
    # a fresh timestamped path; never overwrites a prior run
    out = safe_out("universe_coverage", "csv")
    # write the full table
    df.sort_values(["incumbent", "verdict", "symbol"]).to_csv(out, index=False)
    # say where it landed
    print(f"\nwrote {out}")
    # total elapsed
    print(f"elapsed {time.perf_counter() - t0:,.0f}s")


# entry point
if __name__ == "__main__":
    main()
