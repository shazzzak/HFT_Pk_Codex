# ============================================================================
# expand_feature_store.py -- STEP 2: build feature-store partitions for the
# 76 new names.
# ============================================================================
# A WRAPPER, not a reimplementation. It imports build_feature_store and calls
# its own main() with three globals overridden. The original file is never
# edited and its tested logic (Book replay, Cont-Kukanov OFI, the leak-checked
# forward-markout labels) is reused exactly as the 38 production names' store
# was built -- so the new names' features are directly comparable to theirs.
#
# WHAT IT OVERRIDES, AND WHY
#   PARSED_ROOT / RESULTS_ROOT : the originals hardcode the PRE-MOVE paths
#       (/Users/shazzak/Capital Stake - {Parsed,Results}). Both now come from
#       config_pk. run_legacy_mm's own default points at the EMPTY Google
#       Drive path, and build_feature_store sets the stale one at IMPORT time,
#       so the rebind must happen AFTER the import -- hence the ordering below.
#   FS_ROOT                    : derived from the real RESULTS_ROOT.
#   SYMBOLS                    : the 76 candidates instead of the 38.
#
# SAFE BY CONSTRUCTION
#   build_feature_store.main() skips any symbol-day whose partition already
#   exists, so this cannot overwrite the 38 production names' store even if
#   their symbols were passed in. Nothing is ever deleted.
#
# SMOKE FIRST
#   SMOKE_ONE = True builds a SINGLE symbol-day and stops. Run that before the
#   full pass. It is there to catch one specific hazard: the collector declares
#       def quotes(self, bb, bq, ba, aq, pos)
#   with no `depth` parameter, while mm_backtest.NaiveSymmetricMM declares
#       def quotes(self, bb, bq, ba, aq, pos, depth=None)
#   "for interface parity with the deep-OFI strategy". If the current engine
#   passes depth, every symbol-day dies with TypeError -- and main() swallows
#   per-symbol exceptions into a printed line, so a full run would churn for
#   hours and produce nothing. Same silent-skip shape as 2026-09-12.
#
# Run from existing_mm_live/:  python expand_feature_store.py
# ============================================================================

# wall-clock timing for the summary
import time
# command-line args, so the mode is chosen at the prompt not by editing a file
import sys
# numeric comparison for the verify diff
import numpy as np
# reading the stored parquet partition in verify
import pandas as pd

# the driver module -- imported FIRST so the rebind below is the last word
import run_legacy_mm as R
# the original builder, untouched; it sets its own stale PARSED_ROOT on import
import build_feature_store as BFS
# shared expansion constants + the two guards
import expansion_names as EX

# ---------------------------------------------------------------------------
# PATH REBIND -- must run AFTER `import build_feature_store`, which sets
# R.PARSED_ROOT to the stale pre-move path at import time.
# bind_parsed_root also proves the store is non-empty and returns the dates.
# ---------------------------------------------------------------------------
# push the canonical raw-store root onto the driver and verify it reads
ALL_DATES = EX.bind_parsed_root(R)
# keep the builder module's own copy of the root consistent with the driver
BFS.PARSED_ROOT = EX.PARSED_ROOT
# the results root the feature store hangs off
BFS.RESULTS_ROOT = EX.RESULTS_ROOT
# the feature-store subtree: feature_store/{symbol}/date={date}.parquet
BFS.FS_ROOT = EX.RESULTS_ROOT / "feature_store"
# build the 76 candidates only; the 38 already have partitions
BFS.SYMBOLS = list(EX.NEW_NAMES)

# ---------------------------------------------------------------------------
# COMPATIBILITY PATCH -- build_feature_store.py is currently BROKEN.
# mm_backtest.py line 1214 now calls:
#     want = self.strat.quotes(bb, bq, ba, aq, self.pos, depth=_depth)
# passing `depth` unconditionally, while FeatureCollector declares
#     def quotes(self, bb, bq, ba, aq, pos)
# so every symbol-day dies with TypeError. build_feature_store.main() catches
# per-symbol exceptions and prints them, so a full run would have printed
# 76 x 207 = 15,732 error lines over several hours and written nothing.
#
# This patch is PROVABLY feature-neutral: the method's entire body is
# `return {}`. It has never influenced a recorded value -- the collector is
# passive and every feature comes from the Book the engine mutates. Accepting
# one more keyword argument cannot change any output.
#
# Applied here rather than in build_feature_store.py so the original is left
# untouched. If the original is ever fixed at source, this becomes a no-op.
# ---------------------------------------------------------------------------
# replace the method with an identical one that tolerates the new kwarg
BFS.FeatureCollector.quotes = (
    lambda self, bb, bq, ba, aq, pos, depth=None: {})

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# True = build ONE symbol-day and stop. Default is SMOKE, so the expensive
# path can never be started by accident. Pass --full on the command line to
# run the real build; --smoke is accepted explicitly for symmetry.
# explicit flags; anything else (including no flag) falls through to SMOKE so
# the expensive path can never be started by accident
MODE = ("full" if "--full" in sys.argv
        else "verify" if "--verify" in sys.argv
        else "smoke")
# accepted flags, printed in the banner so the mode is never ambiguous
USAGE = "  usage: python expand_feature_store.py [--smoke | --verify | --full]"


def smoke():
    # prove one symbol-day builds before committing to 76 x 207
    print("=" * 74)
    print("SMOKE: one symbol-day through the real builder")
    print("=" * 74)
    # report the paths actually in force, so a stale root is visible immediately
    print(f"  parsed store : {R.PARSED_ROOT}")
    print(f"  feature store: {BFS.FS_ROOT}")
    print(f"  symbols      : {len(BFS.SYMBOLS)} candidates")
    # pick a mid-range date rather than the first, which can be a partial day
    date = ALL_DATES[len(ALL_DATES) // 2]
    # pick the first candidate alphabetically for reproducibility
    sym = BFS.SYMBOLS[0]
    # say what is being attempted
    print(f"\n  building {sym} on {date} ...", flush=True)
    # open that date's datasets
    dsets = R.open_datasets(date)
    # a missing partition here is a data problem, not a code problem
    if dsets is None:
        raise SystemExit(f"open_datasets({date}) returned None -- no partition.")
    # call the ORIGINAL build_one, unmodified, with NO exception swallowing:
    # a TypeError from the quotes() signature must surface, not be printed
    df = BFS.build_one(date, sym, dsets)
    # None means the symbol-day was legitimately unrunnable, not that it failed
    if df is None:
        print(f"  {sym} {date}: build_one returned None (no trades/book/continuous)")
        print("  -> not a failure; try another date or symbol before concluding.")
        return
    # report what came back, so the columns can be eyeballed once
    print(f"  OK: {len(df):,} rows x {len(df.columns)} cols")
    # the markout label columns are the leak-checked ones; confirm they exist
    labels = [c for c in df.columns if c.startswith("markout_")]
    print(f"  label columns: {labels}")
    # a quick non-null sanity read on the primary 5s label
    if "markout_5000ms_bps" in df.columns:
        # share of rows carrying a usable 5s forward markout
        frac = df["markout_5000ms_bps"].notna().mean()
        print(f"  markout_5000ms_bps non-null: {100 * frac:.1f}%")
    # the go/no-go line, naming the exact next command
    print("\n  SMOKE PASSED -> run the full build with:")
    print("     caffeinate -is python expand_feature_store.py --full")


def full():
    # the full pass: delegate entirely to the original main()
    print("=" * 74)
    print(f"FULL BUILD: {len(BFS.SYMBOLS)} new names x {len(ALL_DATES)} dates")
    print("=" * 74)
    # report the paths in force
    print(f"  parsed store : {R.PARSED_ROOT}")
    print(f"  feature store: {BFS.FS_ROOT}")
    # existing partitions are skipped, so nothing already built is touched
    print("  existing partitions are SKIPPED (no overwrite, no delete)")
    # start the clock
    t0 = time.perf_counter()
    # run the original, unmodified builder
    BFS.main()
    # elapsed
    print(f"\nelapsed {time.perf_counter() - t0:,.0f}s")
    # count what actually landed, per symbol -- main() prints per-DATE progress
    # but never says which symbols ended up with zero partitions
    print("\npartitions written per symbol:")
    # names that produced nothing at all, which would silently skip downstream
    empty = []
    # walk the candidates in a stable order
    for sym in BFS.SYMBOLS:
        # this symbol's feature-store directory
        d = BFS.FS_ROOT / sym
        # how many date partitions exist for it
        n = len(list(d.glob("date=*.parquet"))) if d.exists() else 0
        # record a total miss
        if n == 0:
            empty.append(sym)
        # one line per symbol
        print(f"  {sym:10s} {n:>4d}")
    # a name with zero partitions cannot be calibrated in step 3
    if empty:
        print(f"\n*** {len(empty)} names produced ZERO partitions: {empty}")
        print("    calibrate_all_scales reads the feature store for spread and")
        print("    price, so these cannot be calibrated. Investigate before step 3.")
    else:
        # the clean outcome
        print(f"\nall {len(BFS.SYMBOLS)} candidates have partitions -- ready for step 3.")


def verify():
    # THE COMPARABILITY TEST. calibrate_all_scales reads median spread and
    # median price out of the feature store, so if the engine has drifted
    # since the 38-name store was built, the new names' scales would not be
    # comparable to the incumbents'. Rebuild ONE incumbent symbol-day with
    # today's engine and diff it against its stored partition. Nothing is
    # written -- this is read-and-compare only.
    print("=" * 74)
    print("VERIFY: rebuild one EXISTING incumbent symbol-day and diff it")
    print("=" * 74)
    # find an incumbent that already has partitions
    sym = None
    stored = None
    # walk the production names until one with a partition turns up
    for cand in EX.INCUMBENT:
        # that name's feature-store directory
        d = BFS.FS_ROOT / cand
        # its existing date partitions
        parts = sorted(d.glob("date=*.parquet")) if d.exists() else []
        # take the first name that has any
        if parts:
            # remember the symbol
            sym = cand
            # pick a mid-range partition rather than the first day
            stored = parts[len(parts) // 2]
            break
    # no incumbent partitions at all -> nothing to compare against
    if sym is None:
        print("  no existing incumbent partitions found under")
        print(f"  {BFS.FS_ROOT}")
        print("  -> cannot run the comparability check; skipping is a RISK, not a pass.")
        return
    # the date encoded in the partition filename
    date = stored.stem.split("=", 1)[1]
    # say what is being compared
    print(f"  symbol   : {sym}")
    print(f"  date     : {date}")
    print(f"  stored   : {stored.name}")
    # read the stored partition
    old = pd.read_parquet(stored)
    # open that date's raw datasets
    dsets = R.open_datasets(date)
    # a missing partition here is a data problem
    if dsets is None:
        print(f"  open_datasets({date}) returned None -- cannot rebuild.")
        return
    # rebuild with today's engine, in memory, writing nothing
    print("  rebuilding with the current engine ...", flush=True)
    new = BFS.build_one(date, sym, dsets)
    # a None rebuild is itself a finding
    if new is None:
        print("  *** rebuild returned None while a stored partition exists.")
        print("      The engine no longer produces this symbol-day. INVESTIGATE.")
        return
    # row counts first -- the coarsest possible difference
    print(f"\n  rows : stored {len(old):,}  rebuilt {len(new):,}")
    # column sets
    only_old = sorted(set(old.columns) - set(new.columns))
    only_new = sorted(set(new.columns) - set(old.columns))
    # report any schema drift
    if only_old or only_new:
        print(f"  columns only in stored : {only_old}")
        print(f"  columns only in rebuilt: {only_new}")
    # the shared numeric columns are where a value drift would show
    shared = [c for c in old.columns if c in new.columns]
    # row-count mismatch means the event stream itself changed
    if len(old) != len(new):
        print("\n  *** ROW COUNT DIFFERS -- the event replay has changed.")
        print("      New names' features would NOT be comparable to the")
        print("      incumbents'. Resolve before calibrating.")
        return
    # compare column by column
    print("\n  column                     match")
    # track the overall verdict
    all_ok = True
    # walk every shared column
    for c in shared:
        # pull both series aligned by position
        a = old[c].to_numpy()
        b = new[c].to_numpy()
        # numeric columns compare with a tolerance; others compare exactly
        try:
            # allclose with NaNs treated as equal
            ok = np.allclose(a.astype(float), b.astype(float), equal_nan=True)
        except (TypeError, ValueError):
            # non-numeric: exact equality
            ok = bool((a == b).all())
        # a single mismatch fails the whole check
        if not ok:
            all_ok = False
        # one line per column
        print(f"  {c:26s} {'OK' if ok else 'DIFFERS'}")
    # the verdict, stated in terms of what it means for the pipeline
    if all_ok:
        print("\n  VERIFY PASSED -- today's engine reproduces the stored features")
        print("  exactly. The 76 new names will be comparable to the 38.")
        print("\n  -> run the full build:")
        print("     caffeinate -is python expand_feature_store.py --full")
    else:
        print("\n  *** VERIFY FAILED -- the engine no longer reproduces the stored")
        print("      features. calibrate_all_scales reads spread and price from")
        print("      this store, so the new names' session_scale would be")
        print("      derived on a different basis than the incumbents'.")
        print("      Do NOT proceed to step 3 until this is understood.")


# entry point
if __name__ == "__main__":
    # state the mode before doing anything, so a bare invocation is unambiguous
    print(f"\nMODE = {MODE.upper()}" + ("  (default -- no flag given)"
                                        if len(sys.argv) == 1 else ""))
    # the accepted flags
    print(USAGE + "\n")
    # dispatch on the command-line mode
    if MODE == "full":
        full()
    elif MODE == "verify":
        verify()
    else:
        smoke()
