# ============================================================================
# check_preopen_book.py -- during the pre-open, what does the feed show us?
# ============================================================================
# THE QUESTION. PSX runs a call auction before continuous trading. Orders
# accumulate through it and the exchange matches everything at one clearing
# price at the end. Do we SEE that book building, or are we blind until the
# bell?
#
# WHAT THE SPEC SAYS (PSX FIX Market Data Interface Specifications, the
# MDEntryType notes). For entry types 0 (buy) and 1 (sell):
#
#   "MDEntryPx shows the price, MDEntrySize shows the quantity ... Level 2
#    data discloses 10 levels at most. NumberOfOrders shows number of total
#    orders on this level. Repeated pairs of NoOrders, OrderQty show the order
#    details at this level."
#
#   "For after-hour trading business, only three fields of MDEntryType,
#    MDEntryPx, MDEntrySize are released."
#
# That cut-down disclosure is stated for AFTER-HOURS only. Nothing equivalent
# is stated for the pre-open. And there is an auction-only entry type:
#
#   x3, x4 -- "the aggregate total order of buy (x3), sell (x4) within the
#    effective auctions range in the order book, where MDEntryPx shows the
#    weighted average price of order quantity, MDEntrySize shows the total
#    quantity of orders"
#
# which is the closest thing the feed carries to an indicative auction price.
#
# BUT THE SPEC SAYS WHAT MAY BE PUBLISHED, NOT WHAT OUR CAPTURE RECEIVED.
# This script answers it from the store. It does not assume the answer, and
# where it has to guess (which decoded phase string means "pre-open") it says
# what it guessed so a wrong guess is visible rather than silent.
#
# WHAT IT PRINTS
#   1. every phase found, when it ran, and how many snapshot messages it holds
#   2. which entry types appear in which phase -- the disclosure question
#   3. for the pre-open: how deep the book goes, and whether the per-level
#      order detail (order_ids / order_qtys) is populated
#   4. the x3 / x4 auction aggregates, if the feed carries them
#   5. ONE pre-open snapshot printed in full, as the ladder actually looked
#
# READ-ONLY. Opens parquet, writes nothing, deletes nothing.
#
# Run from existing_mm_live/:
#   caffeinate -is python check_preopen_book.py
#   caffeinate -is python check_preopen_book.py --names UBL,OGDC --date 2026-06-19
#   caffeinate -is python check_preopen_book.py --phase OPEN_CALL_AUCTION
# ============================================================================

# command-line flags
import argparse
# path handling for the parquet store
from pathlib import Path
# the date comes out of the filename
import re

# frames
import pandas as pd

# the project's single source of paths -- never hardcoded in a script
try:
    # PARSED_ROOT is the raw parsed store
    from config_pk import PARSED_ROOT
# no config_pk on the path -> stop with an explicit message
except Exception as _e:                                       # noqa: BLE001
    raise ImportError(
        "check_preopen_book: could not import PARSED_ROOT from config_pk "
        "(%r). Run from the existing_mm_live/ dir, or add it to sys.path."
        % _e)

# THE COLUMNS THIS SCRIPT NEEDS, from ob_snapshot. Verified against the
# generated schema dump. ob_snapshot is ~33M rows a day, so asking for 13 of
# 27 columns and pushing the symbol filter into the reader is the difference
# between seconds and minutes.
COLS = ["snapshot_time", "msg_seq", "symbol", "phase",
        "entry_type_code", "entry_type", "level", "px", "qty",
        "n_orders_at_level", "n_orders_detailed", "order_ids", "order_qtys"]

# the partition naming the parser uses
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_ob_snapshot\.parquet$")

# THE PRE-OPEN PHASES, matched by SUBSTRING rather than exact name.
#
# The parser decodes TradingPhaseCode into a string, and this script does not
# know for certain what it produces. Every pre-open code the spec lists is a
# CALL auction -- 'O' Open Call Auction, 'N' Normal Call Auction (after the
# Friday break), 'V' Normal Call Auction (after a halt) -- while continuous
# trading is "Continuous Auction". So "CALL" separates them, and continuous is
# excluded explicitly in case a decoded name surprises us.
#
# Whatever this matches is PRINTED, so a wrong guess is visible immediately
# and --phase overrides it.
PREOPEN_HINT = "CALL"
CONTINUOUS_HINT = "CONTINUOUS"


def load_day(root, date, names):
    """One day of ob_snapshot for the named symbols."""
    # every ob_snapshot partition, filename order = date order
    files = sorted(Path(root).rglob("*_ob_snapshot.parquet"))
    # nothing to read is a clear message, not a stack trace
    if not files:
        raise SystemExit(f"no *_ob_snapshot.parquet under {root}")
    # the requested date, or the most recent day in the store
    if date:
        # match the partition whose filename carries that date
        want = [f for f in files if f.name.startswith(date)]
        # an unknown date is a stop, with the range named
        if not want:
            raise SystemExit(
                f"no ob_snapshot partition for {date}; store covers "
                f"{files[0].name[:10]} .. {files[-1].name[:10]}")
        path = want[0]
    else:
        # the newest partition
        path = files[-1]
    # the trading date, from the filename
    m = DATE_RE.match(path.name)
    # an unexpected filename is a stop, not a guess
    if m is None:
        raise SystemExit(f"{path.name} does not match the expected naming")
    # say exactly what is being read
    print(f"reading {path.name}  ({len(names) if names else 'all'} symbol(s))")
    # the rows, with the symbol filter pushed into the reader so whole row
    # groups for other symbols are never decompressed
    df = pd.read_parquet(
        path, columns=COLS,
        filters=([("symbol", "in", list(names))] if names else None))
    # belt and braces, in case the reader ignored the pushdown
    if names:
        df = df[df["symbol"].isin(names)]
    # the frame and the date it came from
    return df, m.group(1)


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # which symbols; two liquid names by default
    ap.add_argument("--names", default="UBL,OGDC",
                    help="comma-separated symbols, or ALL (slow: ~33M rows)")
    # which day; the most recent by default
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD; default is the most recent partition")
    # override the phase guess
    ap.add_argument("--phase", default=None,
                    help="exact phase string to treat as pre-open, if the "
                         "substring guess picks the wrong one")
    args = ap.parse_args()

    # the requested symbols, or None for all
    names = (None if args.names.strip().upper() == "ALL"
             else [s.strip().upper() for s in args.names.split(",") if s.strip()])
    # the day's snapshot rows
    df, date = load_day(PARSED_ROOT, args.date, names)
    # nothing matched
    if df.empty:
        raise SystemExit(f"no rows for {args.names} on {date}")
    print(f"{len(df):,} snapshot rows for {date}\n")

    # ---- 1. every phase, when it ran, how much of it we have -------------
    print("=" * 74)
    print("1. PHASES PRESENT -- when each ran and how many snapshot messages")
    print("=" * 74)
    # one row per phase: time span, distinct snapshot messages, total rows
    P = df.groupby("phase").agg(
        first=("snapshot_time", "min"),
        last=("snapshot_time", "max"),
        messages=("msg_seq", "nunique"),
        rows=("msg_seq", "size")).sort_values("first")
    print(P.to_string())

    # ---- which phase is the pre-open ------------------------------------
    # every distinct phase seen
    phases = list(P.index)
    # an explicit override wins
    if args.phase:
        # only the phase the user named
        preopen = [p for p in phases if p == args.phase]
        # a name that is not in the data is a stop, with the options listed
        if not preopen:
            raise SystemExit(
                f"--phase {args.phase} not present. Phases in this day: "
                f"{', '.join(phases)}")
    else:
        # a call auction, but not the continuous one
        preopen = [p for p in phases
                   if PREOPEN_HINT in str(p).upper()
                   and CONTINUOUS_HINT not in str(p).upper()]
    # SAY WHAT WAS GUESSED. A silent wrong guess is the failure mode here.
    print(f"\n  treating as PRE-OPEN: {preopen or 'NOTHING MATCHED'}")
    print(f"  (matched on the substring {PREOPEN_HINT!r}; override with "
          f"--phase)")
    # nothing to analyse
    if not preopen:
        raise SystemExit(
            f"no phase matched. Phases in this day: {', '.join(map(str, phases))}"
            f"\nRe-run with --phase <one of those>.")

    # ---- 2. which entry types appear in which phase ----------------------
    print("\n" + "=" * 74)
    print("2. WHAT IS PUBLISHED, BY PHASE -- rows per entry type")
    print("=" * 74)
    print("  (0/BID and 1/OFFER are the order book itself; x3/x4 are the")
    print("   auction aggregates; the rest are reference prices and limits)")
    # a cross-tab: phase down the side, entry type across
    X = pd.crosstab(df["phase"], [df["entry_type_code"], df["entry_type"]])
    print(X.to_string())

    # the pre-open rows only, from here on
    pre = df[df["phase"].isin(preopen)]
    # nothing captured in the pre-open is itself the answer
    if pre.empty:
        raise SystemExit(
            "the pre-open phase is present in the phase table but has no "
            "rows for these symbols -- nothing was published, or nothing "
            "was captured")

    # ---- 3. how deep does the pre-open book go? --------------------------
    print("\n" + "=" * 74)
    print("3. THE PRE-OPEN ORDER BOOK -- depth and per-level detail")
    print("=" * 74)
    # the two-sided book entries: codes '0' (buy) and '1' (sell)
    book = pre[pre["entry_type_code"].astype("string").isin(["0", "1"])]
    # no book entries at all means we ARE blind during the pre-open
    if book.empty:
        print("  NO BID/OFFER ENTRIES IN THE PRE-OPEN.")
        print("  The feed publishes no order book during this phase for these")
        print("  symbols. Quoting into the auction would be blind.")
    else:
        # per symbol and side: deepest level, and how much detail is filled in
        B = book.groupby(["symbol", "entry_type"]).agg(
            rows=("px", "size"),
            max_level=("level", "max"),
            pct_px=("px", lambda s: 100.0 * s.notna().mean()),
            pct_qty=("qty", lambda s: 100.0 * s.notna().mean()),
            pct_n_orders=("n_orders_at_level",
                          lambda s: 100.0 * s.notna().mean()),
            pct_order_ids=("order_ids", lambda s: 100.0 * s.notna().mean()))
        print(B.to_string(float_format=lambda v: f"{v:,.1f}"))
        print("\n  max_level is how many price levels the exchange disclosed")
        print("  (the spec caps Level 2 data at 10). pct_order_ids is whether")
        print("  the individual orders at each level are named -- that is what")
        print("  makes exact queue position knowable.")

    # ---- 4. the auction aggregates --------------------------------------
    print("\n" + "=" * 74)
    print("4. AUCTION AGGREGATES (x3 = total buy, x4 = total sell, within")
    print("   the effective auction range; px is their weighted average)")
    print("=" * 74)
    # the two auction-only entry types
    agg = pre[pre["entry_type_code"].astype("string").isin(["x3", "x4"])]
    # absent is a real answer and is stated as one
    if agg.empty:
        print("  NOT PRESENT for these symbols on this day. Either the")
        print("  exchange does not publish them here, or the parser did not")
        print("  keep them. entry_type_code holds the RAW code, so if they")
        print("  were captured they would appear -- their absence is the feed")
        print("  or the capture, not a decoding gap.")
    else:
        # how they evolve through the pre-open, per symbol and side
        A = agg.groupby(["symbol", "entry_type_code"]).agg(
            rows=("px", "size"),
            first_px=("px", "first"), last_px=("px", "last"),
            first_qty=("qty", "first"), last_qty=("qty", "last"))
        print(A.to_string(float_format=lambda v: f"{v:,.2f}"))
        print("\n  last_px is the exchange's own weighted-average view of")
        print("  where the auction was heading, and last_qty is how much was")
        print("  eligible to match. That is an indicative auction price in")
        print("  everything but name.")

    # ---- 5. one pre-open snapshot, in full -------------------------------
    print("\n" + "=" * 74)
    print("5. THE LAST PRE-OPEN SNAPSHOT, AS IT ACTUALLY LOOKED")
    print("=" * 74)
    # one symbol at a time
    for sym in sorted(pre["symbol"].unique()):
        # that symbol's pre-open rows
        s = pre[pre["symbol"] == sym]
        # the final snapshot message of the pre-open -- the state going into
        # the match, which is the one that would have informed a decision
        last_seq = s.loc[s["snapshot_time"].idxmax(), "msg_seq"]
        # every row of that one message
        snap = s[s["msg_seq"] == last_seq]
        # a heading naming the symbol and the moment
        print(f"\n  {sym}  msg_seq={last_seq}  "
              f"at {snap['snapshot_time'].iloc[0]}")
        # the book entries, deepest detail first
        ladder = snap[snap["entry_type_code"].astype("string").isin(["0", "1"])]
        # no ladder is itself informative
        if ladder.empty:
            print("    (no bid/offer entries in this message)")
        else:
            # readable: side, level, price, quantity, orders, and whether the
            # individual order ids were disclosed
            show = ladder[["entry_type", "level", "px", "qty",
                           "n_orders_at_level", "order_ids"]].copy()
            # the id list can be long; show only whether it is there and how
            # many ids it holds
            show["order_ids"] = show["order_ids"].map(
                lambda v: f"{len(str(v).split(','))} ids" if pd.notna(v)
                else "-")
            # sorted by side then level, which is how a ladder reads
            print(show.sort_values(["entry_type", "level"])
                  .to_string(index=False))
        # the non-book entries in the same message: limits, reference prices,
        # and the auction aggregates if present
        other = snap[~snap["entry_type_code"].astype("string").isin(["0", "1"])]
        # only print when there is something
        if not other.empty:
            print("\n    other entries in the same message:")
            print(other[["entry_type_code", "entry_type", "px", "qty"]]
                  .to_string(index=False))


# entry point
if __name__ == "__main__":
    main()
