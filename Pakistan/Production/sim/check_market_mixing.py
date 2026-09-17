# ============================================================================
# check_market_mixing.py -- is one symbol's book being built from two markets?
# ============================================================================
# THE HYPOTHESIS, AND IT IS ONLY A HYPOTHESIS UNTIL THIS RUNS.
#
# The data-quality run found 285 crossed books on NRL and MLCF over three
# days, and 275 of them had MORE THAN ONE bid level sitting through the best
# ask. A reconstruction that is one event behind inverts the touch and nothing
# deeper, so that explanation is dead.
#
# The misordered-level dump points somewhere else. MLCF msg_seq 97592 shows a
# price STEP on level 1 -- which can only happen if that group contains two
# level-1 rows. And msg_seq 36284 has bid levels 8, 9 and 10 sitting about
# 5.40 ABOVE level 7, then descending properly among themselves. That is not a
# corrupt ladder. That is TWO ladders concatenated.
#
# PSX lists the same symbol in more than one market: the Ready (cash) market
# and the Deliverable Futures market, among others. Both carry the same
# `symbol`. A futures book trades at a basis to cash -- which is exactly the
# size of the gaps above, and exactly what "bid 117.87 while the ask is
# 106.49" looks like when one of them is cash and the other is futures.
#
# WHY THIS MATTERS FAR BEYOND THE CHECKER. run_legacy_mm.REQ_SNAP selects
# symbol, msg_seq, orig_time, capture_ts, entry_type, px, phase, order_ids,
# order_qtys and qty. There is no `market` in that list and no filter on it.
# If the store holds two markets per symbol, then every book the backtest has
# ever reconstructed mixed them -- and every number this project has produced
# was computed against a book that does not exist.
#
# This script does not assume that. It measures it.
#
# READ-ONLY. Writes nothing.
#
# Run from Production/:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/check_market_mixing.py
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/check_market_mixing.py --names NRL,MLCF
# ============================================================================

# command-line flags
import argparse
# path handling
from pathlib import Path

# frames
import pandas as pd

# the store root, from the one config every runner uses
try:
    from config_pk import PARSED_ROOT
except Exception as _e:                                       # noqa: BLE001
    raise ImportError(
        "check_market_mixing: could not import PARSED_ROOT from config_pk "
        "(%r). Run with PYTHONPATH=../existing_mm_live." % _e)


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # which symbols
    ap.add_argument("--names", default="NRL,MLCF")
    # which day; the most recent by default
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    # the symbols to look at
    names = [s.strip().upper() for s in args.names.split(",") if s.strip()]

    # the snapshot partitions
    files = sorted(Path(PARSED_ROOT).rglob("*_ob_snapshot.parquet"))
    # the one asked for, or the newest
    path = ([f for f in files if f.name.startswith(args.date)][0]
            if args.date else files[-1])
    print(f"reading {path.name} for {', '.join(names)}\n")

    # only what the question needs
    df = pd.read_parquet(
        path,
        columns=["symbol", "msg_seq", "market", "segment", "trading_status",
                 "phase", "entry_type_code", "level", "px",
                 # section 5 tests these as candidate grouping keys
                 "channel", "orig_time", "snapshot_time"],
        filters=[("symbol", "in", names)])

    # ---- 1. how many markets carry each symbol? -------------------------
    print("=" * 74)
    print("1. MARKETS AND SEGMENTS CARRYING EACH SYMBOL")
    print("=" * 74)
    print("  More than one row per symbol here is the whole answer.")
    counts = df.groupby(["symbol", "market", "segment"]).size()
    print(counts.to_string())

    # ---- 2. does one msg_seq span two markets? --------------------------
    print("\n" + "=" * 74)
    print("2. DOES A SINGLE (symbol, msg_seq) SPAN MORE THAN ONE MARKET?")
    print("=" * 74)
    print("  This is the grouping the data-quality checker used, and the")
    print("  grouping run_legacy_mm's book reconstruction uses. If it spans")
    print("  two markets, both were being treated as one book.")
    # markets per (symbol, msg_seq)
    spread = df.groupby(["symbol", "msg_seq"])["market"].nunique()
    # how many snapshots mix
    mixed = int((spread > 1).sum())
    print(f"  {mixed} of {len(spread)} (symbol, msg_seq) groups span "
          f"more than one market")

    # ---- 3. the price gap between the markets ---------------------------
    print("\n" + "=" * 74)
    print("3. THE TOUCH IN EACH MARKET, SIDE BY SIDE")
    print("=" * 74)
    print("  If one market's bid sits above another's ask, then mixing them")
    print("  produces exactly the 'crossed book' the checker reported -- and")
    print("  the difference is a futures basis, not a fault in the feed.")
    # the book rows only
    book = df[df["entry_type_code"].astype("string").isin(["0", "1"])]
    # best bid and best ask per symbol per market
    rows = []
    # walk each symbol and market
    for (sym, mkt), g in book.groupby(["symbol", "market"]):
        # that market's best bid
        b = g[g["entry_type_code"].astype("string") == "0"]["px"].max()
        # and its best ask
        a = g[g["entry_type_code"].astype("string") == "1"]["px"].min()
        # keep both
        rows.append({"symbol": sym, "market": mkt, "best_bid_seen": b,
                     "best_ask_seen": a, "rows": len(g)})
    # as a table
    print(pd.DataFrame(rows).to_string(
        index=False, float_format=lambda v: f"{v:,.4f}"))

    # ---- 4. DOES ONE msg_seq HOLD MORE THAN ONE SNAPSHOT? ---------------
    print("\n" + "=" * 74)
    print("4. DOES A SINGLE (symbol, msg_seq) HOLD MORE THAN ONE SNAPSHOT?")
    print("=" * 74)
    print("  THIS IS THE ONE THAT MATTERS. run_legacy_mm groups snapshots by")
    print("  msg_seq ALONE -- snap_groups = {ms: prep(grp) for ms, grp in")
    print("  s.groupby('msg_seq')} -- and Book.snapshot() then replaces the")
    print("  whole book from that group. If two 35=W messages for one symbol")
    print("  ever share a msg_seq, their ladders are concatenated into a single")
    print("  book: two level-1 bids, two level-1 offers, and a 'crossed' touch")
    print("  that is really one message's bid against the other's ask.")
    print("  The misordered-level dump already showed a price step ON LEVEL 1,")
    print("  which can only happen if a group holds two level-1 rows.\n")
    # the book rows only
    book = df[df["entry_type_code"].astype("string").isin(["0", "1"])]
    # per (symbol, msg_seq, side): how many rows claim to be level 1
    lvl1 = (book[book["level"] == 1]
            .groupby(["symbol", "msg_seq", "entry_type_code"])
            .size().rename("level_1_rows").reset_index())
    # a well-formed snapshot has exactly one level 1 per side
    dupes = lvl1[lvl1["level_1_rows"] > 1]
    # the headline
    print(f"  groups with MORE THAN ONE level-1 row on a side: "
          f"{len(dupes)} of {len(lvl1)}")
    # and the worst of them, so the scale is visible
    if len(dupes):
        # biggest first
        worst = dupes.sort_values("level_1_rows", ascending=False).head(10)
        # named per side
        worst = worst.assign(
            side=worst["entry_type_code"].map({"0": "BID", "1": "OFFER"}))
        print(worst[["symbol", "msg_seq", "side", "level_1_rows"]]
              .to_string(index=False))
        # how many distinct snapshots that implies
        print(f"\n  A group with N level-1 rows on a side is N snapshots")
        print(f"  merged into one book.")

    # ---- 5. what SHOULD the grouping key be? ----------------------------
    print("\n" + "=" * 74)
    print("5. WHAT MAKES A SNAPSHOT UNIQUE?")
    print("=" * 74)
    # candidate keys, each tested for whether it yields exactly one level-1
    # bid per group
    for key in (["symbol", "msg_seq"],
                ["symbol", "msg_seq", "market"],
                ["symbol", "msg_seq", "channel"],
                ["symbol", "msg_seq", "orig_time"],
                ["symbol", "orig_time"]):
        # only the columns we actually have
        if not set(key) <= set(df.columns):
            print(f"  {'+'.join(key):40s} (column not selected)")
            continue
        # bids at level 1, grouped this way
        g = (book[(book["level"] == 1)
                  & (book["entry_type_code"].astype("string") == "0")]
             .groupby(key).size())
        # how many groups hold more than one
        bad = int((g > 1).sum())
        # a key that yields exactly one level-1 bid per group is a candidate
        mark = "OK  " if bad == 0 else "BAD "
        print(f"  {mark}{'+'.join(key):40s} {bad} of {len(g)} groups "
              f"hold more than one level-1 bid")
    print("\n  The first key marked OK is the one the loader should group on.")

    # ---- 6. the verdict --------------------------------------------------
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    # CORRECTED 2026-09-17: the first version of this verdict fired on
    # section 1 alone -- "a symbol appears in more than one market" -- without
    # consulting section 2, which is the question that actually matters. A
    # symbol can be listed in two markets and still never have them merged
    # into one book. Both conditions are now required.
    # how many markets each symbol appears in at all
    per_sym = df.groupby("symbol")["market"].nunique()
    # whether any single snapshot group actually SPANS two markets
    spans = int((df.groupby(["symbol", "msg_seq"])["market"].nunique() > 1).sum())
    # duplicate level-1 rows, the other candidate
    dup_groups = len(dupes)
    # the two findings, separately, because they need separate fixes
    if spans:
        print("  MARKETS ARE BEING MERGED INTO ONE BOOK. Fix the loader's")
        print("  grouping key and re-run everything affected.")
    elif (per_sym > 1).any():
        print("  A symbol is LISTED in more than one market, but no single")
        print("  snapshot group spans two, so markets are not being merged")
        print("  within a message. STILL A REAL BUG THOUGH: REQ_SNAP does not")
        print("  select `market`, so a snapshot message from the other market")
        print("  would replace that symbol's book wholesale until the next")
        print("  regular-market snapshot arrives. Small, and worth filtering.")
    else:
        print("  Each symbol appears in exactly one market. Markets are not")
        print("  the issue at all.")
    # the level-1 finding, which is the one that explains a crossed book
    if dup_groups:
        print()
        print("  AND THE CAUSE OF THE CROSSED BOOKS IS ABOVE: groups holding")
        print("  more than one level-1 row are two snapshots merged into one.")
        print("  Section 5 names the key that separates them.")
    else:
        print()
        print("  Every group holds exactly one level-1 row per side, so merged")
        print("  snapshots are NOT the cause either. The crossed books are")
        print("  something else again, and nothing measured so far explains")
        print("  them. Do not guess -- the next step is to dump one crossed")
        print("  snapshot in full, every row, and read it.")


# entry point
if __name__ == "__main__":
    main()
