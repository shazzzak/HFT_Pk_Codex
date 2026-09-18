"""Why did the two runs quote a different price at the same instant?

THE QUESTION THIS ANSWERS. On THCCL 2026-06-05 the backtest and the engine
produced byte-identical order logs for 1,026 records and then disagreed about
ONE price: order 1078, SELL, 25 shares, sent and landed at the same
milliseconds in both runs, priced 66.14 by the backtest and 66.89 by the
engine. Everything that differs after that -- the extra amendment, the
different timings, the three fewer kept places -- follows from that one
number.

Both runs ask the SAME strategy class for their prices, but they hand it
different objects:

    mm_backtest.py       self.strat.quotes(bb, bq, ba, aq, self.pos, depth=_depth)
    venues/psx_strategy   self._mm.quotes(bb, bq, ba, aq, position, depth=depth)

`self.pos` is the backtest's own running position; `position` is the order
manager's tracker. `_depth` is built by the backtest's book; `depth` by the
production book model. This script records every call to quotes() in both
runs -- the inputs and the answer -- and prints the first call where the
answers disagree, with the inputs side by side. Whichever input differs on
that line is the cause. Nothing is guessed.

It changes no behaviour: it wraps the strategy's methods, runs the two
existing runners from sim/gate.py unmodified, and unwraps afterwards.

RUN IT:
    cd Production
    PYTHONPATH=../existing_mm_live caffeinate -i python sim/diff_quote_inputs.py \
        --symbol THCCL --date 2026-06-05
"""
# command line parsing
import argparse
# the output path helper, same one every other script here uses
from pathlib import Path
# where sim/ sits, so `import sim.gate` works when run from Production
import sys

# tables for the side-by-side comparison
import pandas as pd

# make Production importable no matter where this is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# the gate's own loaders and runners, reused exactly so this measures the
# same two runs the gate measures -- not a reconstruction of them
from sim.gate import (load_symbol_day, run_baseline, run_engine)
# the driver module: dataset opening
import run_legacy_mm as R
# the shared harness: the calibration loaders
import mm_harness as H
# the strategy class whose methods are wrapped
from micro_mm import MicrostructureMM
# the shared never-overwrite output helper, the same one gate.py and
# diff_fills.py use -- so no results path is written down in this file
import expansion_names as EX


# every recorded call, from both runs, in the order they happened
RECORDS = []
# which run is being recorded right now; set before each run
CURRENT = {"run": "?"}


def _summarise(obj):
    """A short, comparable string for an object of unknown shape.

    depth is a list of levels in one run and may be a different container in
    the other. Comparing reprs is enough to answer 'are these the same?',
    which is the only question being asked of it.
    """
    # None compares cleanly as itself
    if obj is None:
        return "None"
    # anything else: its repr, truncated so the table stays readable
    s = repr(obj)
    # keep it to one screen width
    return s if len(s) <= 160 else s[:157] + "..."


def install(cls):
    """Wrap observe() and quotes() on the strategy class.

    observe() is wrapped ONLY to stamp the exchange time onto the instance:
    quotes() is not given a timestamp, so without this there is no way to
    line the two runs up against each other or against the order log.
    """
    # keep the originals so they can be restored
    orig_observe = cls.observe
    orig_quotes = cls.quotes

    def observe(self, kind, obj, ts_exch, mid, *a, **k):
        # stamp the instance with the latest exchange time seen
        self._dqi_ts = ts_exch
        # then do exactly what it always did
        return orig_observe(self, kind, obj, ts_exch, mid, *a, **k)

    def quotes(self, bb, bq, ba, aq, pos, *a, **k):
        # ask the real strategy first, so a raising call is not swallowed
        out = orig_quotes(self, bb, bq, ba, aq, pos, *a, **k)
        # the answer for each side, as (price, quantity) or None
        want_b = out.get("BUY") if isinstance(out, dict) else None
        want_s = out.get("SELL") if isinstance(out, dict) else None
        # one row per call: which run, when, what went in, what came out
        RECORDS.append({
            "run": CURRENT["run"],
            # the exchange time of the last event the strategy was shown
            "ts": getattr(self, "_dqi_ts", None),
            # the touch, which both runs read from the same historical book
            "bb": bb, "bq": bq, "ba": ba, "aq": aq,
            # the position, which the two runs compute in different places
            "pos": pos,
            # the depth object, which the two runs BUILD in different places
            "depth": _summarise(k.get("depth")),
            # the answer, split so a price difference is visible on its own
            "bid_px": (want_b[0] if want_b else None),
            "bid_qty": (want_b[1] if want_b else None),
            "ask_px": (want_s[0] if want_s else None),
            "ask_qty": (want_s[1] if want_s else None),
            # the strategy's own view of where it is in the day
            "window": getattr(self, "current_window", None),
            "bucket": getattr(self, "current_bucket", None),
            "regime": getattr(self, "current_regime", None),
        })
        # hand the real answer back unchanged
        return out

    # swap them in
    cls.observe = observe
    cls.quotes = quotes
    # give the caller what it needs to undo this
    return orig_observe, orig_quotes


def restore(cls, orig_observe, orig_quotes):
    """Put the strategy class back exactly as it was."""
    # the original event handler
    cls.observe = orig_observe
    # the original quoting method
    cls.quotes = orig_quotes


def main():
    # the command line, mirroring sim/diff_fills.py so the two are used alike
    ap = argparse.ArgumentParser()
    # which name
    ap.add_argument("--symbol", required=True)
    # which date, as the parsed store spells it
    ap.add_argument("--date", required=True)
    # the reprice mechanic; the gate passes under replace, so that is default
    ap.add_argument("--mode", choices=("replace", "cancel_new"),
                    default="replace")
    # how many calls either side of the first disagreement to print
    ap.add_argument("--context", type=int, default=8)
    args = ap.parse_args()
    # True when both sides use one amendment message per reprice
    use_g = (args.mode == "replace")

    print("=" * 78)
    print(f"QUOTE INPUT DIFF -- {args.symbol} {args.date} ({args.mode})")
    print("=" * 78)

    # the date's partitions
    dsets = R.open_datasets(args.date)
    # a missing partition is a clear stop, not a traceback
    if dsets is None:
        raise SystemExit(f"no datasets for {args.date}")
    # the calibration every runner uses
    scales, profiles = H.load_scales(), H.load_profiles()
    # the end-of-day window and the session segments
    windows, segments = H.load_windows(), H.load_segments()
    # this date's segments
    segs = segments.get(str(args.date))
    # without them the strategy cannot be built the way the gate builds it
    if segs is None:
        raise SystemExit(f"no session segments for {args.date}")
    # the strategy parameters, assembled exactly as every runner does
    params = H.build_micro_params(50, scales[args.symbol],
                                  profiles[args.symbol],
                                  windows[args.symbol], segs)
    # the day's events, built once and shared by both runs
    loaded = load_symbol_day(dsets, args.symbol)
    # an unrunnable symbol-day
    if loaded is None:
        raise SystemExit(f"{args.symbol} {args.date} is not runnable")
    # unpack
    events, snap_groups, t0, t1, ref_minor = loaded

    # wrap the strategy class for both runs
    saved = install(MicrostructureMM)
    try:
        # ---- the thing being reproduced --------------------------------
        CURRENT["run"] = "backtest"
        run_baseline(events, snap_groups, params, t0, t1, use_g)
        # ---- the production engine, under the gate's own policy ---------
        CURRENT["run"] = "engine"
        run_engine(events, snap_groups, params, t0, t1, ref_minor,
                   args.symbol, args.date, use_g, quantity_policy="exact")
    finally:
        # always put the class back, even if a run raises
        restore(MicrostructureMM, *saved)

    # every recorded call from both runs
    df = pd.DataFrame(RECORDS)
    # a run that recorded nothing means the wrap did not take
    if df.empty:
        raise SystemExit("no quote calls recorded -- the wrap did not take")
    # number the calls within each run so the two can be lined up
    df["seq"] = df.groupby("run").cumcount()
    print(f"  calls recorded: backtest {(df['run'] == 'backtest').sum():,}, "
          f"engine {(df['run'] == 'engine').sum():,}")

    # the two runs, side by side, aligned on call number
    a = df[df["run"] == "backtest"].set_index("seq")
    b = df[df["run"] == "engine"].set_index("seq")
    # the calls both runs made
    n = min(len(a), len(b))
    # the columns whose disagreement IS the finding
    out_cols = ["bid_px", "bid_qty", "ask_px", "ask_qty"]
    # the columns that explain it
    in_cols = ["ts", "bb", "bq", "ba", "aq", "pos", "depth",
               "window", "bucket", "regime"]

    # walk forward to the first call where the ANSWERS differ
    first = None
    for i in range(n):
        # the two answers at this call number
        if not a.iloc[i][out_cols].equals(b.iloc[i][out_cols]):
            first = i
            break

    # both runs answered identically for every shared call
    if first is None:
        print("  no answer differs on any shared call.")
        # a different number of calls is itself the finding, then
        if len(a) != len(b):
            print(f"  BUT the call COUNTS differ: {len(a):,} vs {len(b):,} "
                  f"-- the runs were asked a different number of times.")
    else:
        print(f"\n  FIRST DISAGREEMENT at call {first:,}\n")
        # the window either side of it
        lo, hi = max(0, first - args.context), first + args.context
        # the answers, which is what diverged
        print("  --- WHAT EACH RUN ANSWERED ---")
        print(pd.concat([a.loc[lo:hi, out_cols].add_prefix("bt_"),
                         b.loc[lo:hi, out_cols].add_prefix("en_")],
                        axis=1).to_string())
        # the inputs at that one call, which is what explains it
        print("\n  --- WHAT EACH RUN WAS GIVEN, at the disagreeing call ---")
        cmp = pd.DataFrame({"backtest": a.iloc[first][in_cols],
                            "engine": b.iloc[first][in_cols]})
        # the column that makes the answer obvious
        cmp["same"] = [x == y for x, y in zip(cmp["backtest"], cmp["engine"])]
        print(cmp.to_string())
        print("\n  THE INPUT MARKED False IS THE CAUSE. If every input is")
        print("  True, the two runs were given identical arguments and")
        print("  answered differently, which means the strategy's own")
        print("  internal state has diverged -- look at what each run tells")
        print("  it about fills, not at what it is asked.")

    # write the whole recording so it can be examined without re-running
    # a fresh timestamped destination; never overwrites
    path = EX.safe_out(f"quote_inputs_{args.symbol}_{args.date}", "csv")
    # the full recording, both runs, every call
    df.to_csv(path, index=False)
    print(f"\n  wrote {path}")


# the usual entry point
if __name__ == "__main__":
    main()
