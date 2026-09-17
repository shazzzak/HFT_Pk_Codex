# ============================================================================
# verify_fix.py -- did the snapshot-grouping fix actually take?
# ============================================================================
# RUN THIS BEFORE COMMITTING HOURS TO A FULL BACKTEST.
#
# Two bugs were found in the loader on 2026-09-17 and both are fixed in
# run_legacy_mm.py, mm_harness.py and mm_backtest.py. This file proves the fix
# is live rather than assuming it, because a full run against a store that is
# still being grouped wrongly is hours spent producing numbers that have to be
# thrown away.
#
# BUG 1 -- TWO SNAPSHOT MESSAGES MERGED INTO ONE BOOK.
#   msg_seq is not unique per 35=W message. The loader grouped snapshots on
#   msg_seq alone, so two messages for one symbol were concatenated: two
#   level-1 bids, two level-1 offers, and a touch that reads as one message's
#   bid against the other's ask. Measured on NRL and MLCF for 2026-06-30:
#     symbol+msg_seq             90 of 12,476 groups held TWO level-1 bids
#     symbol+msg_seq+market      90   (market does not separate them)
#     symbol+msg_seq+channel     90   (nor does channel)
#     symbol+msg_seq+orig_time    0 of 12,566   -- clean
#   The fix keys snapshots on "msg_seq|orig_time".
#
# BUG 2 -- A SECOND MARKET REPLACING THE BOOK.
#   MLCF carries 69 EQ_SQUARE_UP rows beside 272,894 REG rows, and nothing
#   filtered on market. A square-up snapshot replaced MLCF's whole book --
#   square-up best bid 122.49 against the regular market's 95.86 ask -- until
#   the next regular snapshot arrived. The fix filters to market="REG".
#
# WHAT THIS FILE CHECKS, in order, stopping at the first failure:
#   1. the loader selects `market` and read_symbol accepts a market argument
#   2. snapshots are grouped on snap_key, and snap_key is on the event rows
#   3. the regular-market filter is actually applied by the harness
#   4. every snapshot group holds exactly ONE level-1 row per side
#   5. one real symbol-day runs end to end, and crossed_book_requotes is
#      reported -- THE NUMBER THAT MATTERS. Before the fix the engine was
#      quoting off merged books; after it this should be at or near zero.
#
# READ-ONLY. Runs one symbol-day. Writes nothing.
#
# Run from existing_mm_live/:
#   caffeinate -is python verify_fix.py
#   caffeinate -is python verify_fix.py --symbol MLCF
# ============================================================================

# command-line flags
import argparse
# exit codes
import sys
# frames
import pandas as pd

# the loader under test
import run_legacy_mm as R
# the shared harness, which is the path every runner actually uses
import mm_harness as H

# every check's outcome, so the run reports a total rather than dying on the
# first problem where it can usefully continue
results = []


def check(name, condition, detail=""):
    """Record one check and print it as it happens."""
    # keep it for the summary
    results.append((name, bool(condition)))
    # and show it immediately
    print(("PASS  " if condition else "FAIL  ") + name)
    # a failure gets its explanation on the next line, indented
    if not condition and detail:
        print("        " + detail)


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # which symbol to run end to end; NRL is the most profitable name
    ap.add_argument("--symbol", default="NRL",
                    help="the symbol to run one full day of")
    # which day; the most recent by default
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD; default is the most recent in the store")
    args = ap.parse_args()

    print("=" * 74)
    print("VERIFYING THE SNAPSHOT-GROUPING FIX")
    print("=" * 74)

    # ---- 1. is the loader even the fixed one? --------------------------
    print("\n1. THE LOADER ITSELF")
    # `market` has to be selected or it cannot be filtered on
    check("REQ_SNAP selects `market`", "market" in R.REQ_SNAP,
          "run_legacy_mm.py is the old version -- the market filter cannot "
          "work without this column")
    # read_symbol has to accept the argument the callers now pass
    import inspect
    # its parameters
    params = inspect.signature(R.read_symbol).parameters
    # the new one
    check("read_symbol accepts a market argument", "market" in params,
          "run_legacy_mm.py is the old version")

    # ---- 2. is the grouping key the composite one? ---------------------
    print("\n2. THE GROUPING KEY")
    # the source of build_events, read rather than assumed
    src = inspect.getsource(R.build_events)
    # the composite key has to be built
    check("build_events builds snap_key", 'snap_key' in src,
          "still grouping on msg_seq alone")
    # and grouped on
    check("snapshots are grouped on snap_key",
          'groupby("snap_key")' in src,
          "the key is built but not used")
    # mm_backtest has to look up the same key
    import mm_backtest as MB
    # the run loop's source
    run_src = inspect.getsource(MB.Backtester.run)
    # the lookup
    check("mm_backtest.run looks up snap_key",
          "snap_groups[obj.snap_key]" in run_src,
          "mm_backtest.py is the old version -- it will look up the wrong "
          "snapshot, or raise KeyError")

    # stop here if the files are not the fixed ones: everything below would
    # be testing the old code and reporting confusing results
    if any(not ok for _, ok in results):
        print("\nSTOPPING: the files above are not the fixed versions.")
        print("Move all four downloaded files into existing_mm_live/ first.")
        sys.exit(1)

    # ---- 3. the data, as the harness actually reads it -----------------
    print("\n3. THE DATA AS THE HARNESS READS IT")
    # the day to use
    dates = R.discover_dates()
    # the one asked for, or the newest
    date = args.date or dates[-1]
    # that date's partitions
    dsets = R.open_datasets(date)
    # a missing partition is a stop, not a guess
    if dsets is None:
        raise SystemExit(f"no datasets for {date}")
    print(f"   {args.symbol} on {date}")
    # the snapshots, read exactly the way run_symbol_day reads them
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, args.symbol,
                      market="REG")
    # how many rows came back
    print(f"   {len(s):,} snapshot rows")
    # only the regular market should be present now
    markets = sorted(s["market"].dropna().unique()) if "market" in s else []
    check("only the regular market came back", markets == ["REG"],
          f"got {markets} -- the filter is not being applied")

    # ---- 4. one level-1 row per side, per snapshot ---------------------
    print("\n4. ONE SNAPSHOT PER GROUP")
    # build the same key the loader builds
    s2 = s.copy()
    # the composite, exactly as build_events forms it
    s2["snap_key"] = (s2["msg_seq"].astype("int64").astype(str) + "|"
                      + s2["orig_time"].astype(str))
    # the book rows, by their decoded type name
    book = s2[s2["entry_type"].astype("string").isin(["BID", "OFFER"])]
    # level-1 rows only
    lvl1 = book[book["level"] == 1] if "level" in book.columns else book
    # a well-formed snapshot has exactly one level 1 per side
    dupes_new = (lvl1.groupby(["snap_key", "entry_type"]).size() > 1).sum()
    # and what the OLD key would have given, for the comparison
    dupes_old = (lvl1.groupby(["msg_seq", "entry_type"]).size() > 1).sum()
    # the before and after, side by side
    print(f"   grouped on msg_seq alone : {dupes_old:,} groups with two "
          f"level-1 rows")
    print(f"   grouped on snap_key      : {dupes_new:,}")
    # the fix must eliminate them
    check("no snapshot group holds two level-1 rows", dupes_new == 0,
          "the composite key is not separating the messages; orig_time is "
          "second-precision, so two snapshots inside one second would still "
          "merge -- send me the numbers above")
    # and it must have been doing something: if the old key was already clean
    # on this symbol-day, this run has not tested anything
    if dupes_old == 0:
        print("   NOTE: the old key was already clean on this symbol-day, so")
        print("   this particular day does not exercise the fix. Try MLCF, or")
        print("   another date, before trusting a pass here.")

    # ---- 5. one symbol-day, end to end ---------------------------------
    print("\n5. ONE SYMBOL-DAY, END TO END")
    # the calibration the harness needs
    scales, profiles = H.load_scales(), H.load_profiles()
    # and the rest
    windows, segments = H.load_windows(), H.load_segments()
    # this date's session segments
    segs = segments.get(str(date))
    # a name missing calibration cannot be run
    if (segs is None or args.symbol not in scales
            or args.symbol not in profiles or args.symbol not in windows):
        raise SystemExit(f"{args.symbol} has no calibration for {date}")
    # the strategy parameters, assembled the way every runner assembles them
    params = H.build_micro_params(50, scales[args.symbol],
                                  profiles[args.symbol],
                                  windows[args.symbol], segs)
    # the run
    dr = H.run_symbol_day(date, args.symbol, dsets, params)
    # an unrunnable day
    if dr is None:
        raise SystemExit(f"{args.symbol} {date} did not run")
    # the headline
    print(f"   net P&L : {dr.pnl():,.2f} PKR")
    print(f"   fills   : {len(dr.fills):,}")
    print(f"   orders  : {dr.stats.get('n_orders_sent', 0):,}")
    # THE NUMBER THAT MATTERS
    crossed = int(dr.stats.get("crossed_book_requotes", 0))
    print(f"\n   crossed_book_requotes : {crossed:,}")
    print("   This counts the cycles where the engine refused to quote")
    print("   because the reconstructed book had its best bid at or above")
    print("   its best ask. Before the grouping fix those books were two")
    print("   snapshots stacked on top of each other.")
    # near zero is the expected outcome; a large number means the fix did not
    # reach the reconstruction
    check("crossed-book requotes are rare after the fix", crossed < 20,
          f"{crossed} is high. The grouping fix has not reached the "
          f"reconstruction, or there is a second cause. Do not start the "
          f"full run.")

    # ---- summary --------------------------------------------------------
    print("\n" + "=" * 74)
    # anything that failed
    failed = [n for n, ok in results if not ok]
    # the headline
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    # named, so a failure says what broke
    for n in failed:
        print("  FAILED:", n)
    # and the go/no-go, stated rather than left to be inferred
    if failed:
        print("\nDO NOT START THE FULL RUN.")
    else:
        print("\nClear. The full backtest is worth starting.")
    # non-zero exit on any failure, so this can gate a script
    sys.exit(1 if failed else 0)


# entry point
if __name__ == "__main__":
    main()
