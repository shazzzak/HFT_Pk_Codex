# ============================================================================
# check_auction_prints.py -- how much actually trades in the opening auction?
# ============================================================================
# WHAT AN OPENING AUCTION IS. PSX runs a call auction before continuous
# trading (FIX tag 8538 TradingPhaseCode 'O' = Open Call Auction, the morning
# pre-open). Nothing matches continuously during it: orders accumulate, and at
# the end the exchange computes ONE clearing price and matches all eligible
# buys against all eligible sells at that single price. Everyone gets the same
# price; there is no aggressor and no spread to pay.
#
# The parser tags the resulting trades: when BOTH BidApplSeqNum (10116) and
# OfferApplSeqNum (10117) are nonzero, neither side was the resting one, so
# the row is written with initiator = 'AUCTION' (PSX_Parser_Mac.py, Fix 8).
# ("Auction match" throughout, NOT "cross" -- that word also means sending an
# order that takes the other side of the book, which is a different thing and
# has its own guard in mm_backtest.)
#
# WHAT IT IS FOR. Characterising opening-auction liquidity per name: how much
# size is genuinely available at the open, and how the auction price compares
# with the first continuous print. Auction liquidity varies enormously by
# name -- a small-cap can match 26 shares while a large-cap matches tens of
# thousands -- and a number from one illiquid name says nothing about another.
#
# WHAT IT PRINTS, per symbol-day:
#   n_auction     how many auction trades exist
#   auction_qty   total shares matched -- the liquidity actually there
#   auction_vwap  their volume-weighted price
#   first_cont    the first CONTINUOUS print after the auction
#   gap_pct       how far apart those two are, as a percentage
#
# PLAIN PANDAS, NOT duckdb. Same reader mm_harness and run_legacy_mm use, so
# there is one fewer dependency and one fewer SQL dialect to be wrong about.
#
# READ-ONLY. Opens parquet, writes nothing, deletes nothing.
#
# Run from existing_mm_live/:
#   caffeinate -is python check_auction_prints.py
#   caffeinate -is python check_auction_prints.py --names UBL,OGDC --days 20
#   caffeinate -is python check_auction_prints.py --names ALL --days 40
# ============================================================================

# command-line flags
import argparse
# path handling for the parquet store
from pathlib import Path
# the date is parsed out of the filename, so a regex rather than slicing
import re

# frames; the store is parquet and pandas already reads it everywhere else
import pandas as pd

# the project's single source of paths -- never hardcoded in a script.
# Same import run_legacy_mm uses, so this script moves machines with
# everything else. Fail LOUD rather than guessing a path.
try:
    # PARSED_ROOT is the raw parsed store
    from config_pk import PARSED_ROOT
# no config_pk on the path -> stop with an explicit message
except Exception as _e:
    raise ImportError(
        "check_auction_prints: could not import PARSED_ROOT from config_pk "
        "(%r). Run from the existing_mm_live/ dir, or add it to sys.path."
        % _e)

# THE COLUMNS THIS SCRIPT NEEDS, from the trades table.
#
# CORRECTED 2026-09-16. The first version also asked for a `date` column,
# taken from schema_output.txt, which lists one. A read of the FILE fails with
# "No match for FieldRef.Name(date)", because the store is hive-partitioned:
#
#   <PARSED_ROOT>/trades/date=2026-06-30/2026-06-30_trades.parquet
#
# The `date` column is the PARTITION KEY, synthesised from the directory name
# by a dataset read and absent from the file's own footer. Both schemas are
# correct; they differ by exactly that key.
#
# This script reads files directly, so it takes the date from the FILENAME,
# which the parser controls and which is therefore reliable either way.
COLS = ["symbol", "transact_time", "price", "qty", "initiator"]

# The date prefix the parser writes on every partition: 2026-06-22_trades.parquet
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_trades\.parquet$")


def load_days(root, n_days, names=None):
    """Read the most recent n_days of trades partitions into one frame.

    `names` is pushed down into the parquet read as a row-group filter rather
    than applied afterwards. The parser sorts every partition by symbol first
    precisely so a single-symbol read can skip row groups, so on a named-symbol
    run this is the difference between reading two symbols and reading the
    whole market. None means every symbol.
    """
    # every trades partition under the store, in filename order (which is
    # date order, because the parser names them {day}_trades.parquet)
    files = sorted(Path(root).rglob("*_trades.parquet"))
    # nothing to read is a clear message, not a stack trace
    if not files:
        raise SystemExit(f"no *_trades.parquet under {root}")
    # the most recent N
    files = files[-n_days:]
    # say exactly what is being read, so a wrong store is obvious
    print(f"reading {len(files)} day(s) from {root}")
    print(f"  {files[0].name} .. {files[-1].name}\n")
    # one frame per file
    parts = []
    # read each, asking only for the columns we use
    for f in files:
        # THE DATE COMES FROM THE FILENAME. The partition has no date column
        # of its own (schema_output.txt is misleading on this point), and the
        # parser names every file {day}_trades.parquet, so the name is the
        # authoritative source.
        m = DATE_RE.match(f.name)
        # an unexpected filename is a stop, not a guess
        if m is None:
            raise SystemExit(
                f"{f.name} does not match the expected "
                f"YYYY-MM-DD_trades.parquet naming; not guessing its date")
        # a partition missing a column fails LOUDLY here with the file named,
        # rather than producing a silently wrong aggregate later
        try:
            # only the columns this script needs, and only the symbols asked
            # for -- the filter is pushed into the reader so whole row groups
            # for other symbols are never decompressed
            part = pd.read_parquet(
                f, columns=COLS,
                filters=([("symbol", "in", list(names))] if names else None))
        # name the file AND the columns in the error so the fix is obvious
        except Exception as exc:                              # noqa: BLE001
            raise SystemExit(
                f"could not read {f.name} with columns {COLS}: {exc!r}")
        # stamp the trading date from the filename
        part["date"] = m.group(1)
        # keep it
        parts.append(part)
    # one frame
    return pd.concat(parts, ignore_index=True)


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # which symbols to look at; default is the smoke pair
    ap.add_argument("--names", default="AGHA,AGP",
                    help="comma-separated symbols, or ALL for every symbol")
    # how many of the most recent dates to look at
    ap.add_argument("--days", type=int, default=5,
                    help="how many of the most recent trading days")
    args = ap.parse_args()

    # the requested symbols, or None for every symbol
    names = (None if args.names.strip().upper() == "ALL"
             else [s.strip().upper() for s in args.names.split(",") if s.strip()])
    # the trades rows for those days, symbol filter pushed into the read
    df = load_days(PARSED_ROOT, args.days, names)
    # a belt-and-braces filter in case the reader ignored the pushdown
    if names:
        df = df[df["symbol"].isin(names)]
    # nothing matched
    if df.empty:
        raise SystemExit(
            f"no rows for {args.names} in the last {args.days} day(s) -- "
            f"check the symbols against the store")

    # WHICH ROWS ARE THE CROSS. The parser writes initiator='AUCTION' for a
    # auction trade and a side ('BUY'/'SELL') for a continuous print.
    # Compared as a
    # string so a null initiator lands on the continuous side rather than
    # silently disappearing from both halves.
    is_auction = df["initiator"].astype("string").fillna("") == "AUCTION"

    # ---- the auction half ------------------------------------------------
    # only the auction trades
    auc = df[is_auction].copy()
    # value traded, for the volume-weighted price
    auc["value"] = auc["price"] * auc["qty"]
    # per symbol-day: how many, how much, at what average price
    A = auc.groupby(["date", "symbol"]).agg(
        n_auction=("price", "size"),
        auction_qty=("qty", "sum"),
        auction_value=("value", "sum"))
    # the vwap, guarded against a zero-quantity match
    A["auction_vwap"] = A["auction_value"] / A["auction_qty"].where(
        A["auction_qty"] > 0)
    # drop the working column
    A = A.drop(columns=["auction_value"])

    # ---- the continuous half ---------------------------------------------
    # everything that is not an auction trade, earliest first
    cont = df[~is_auction].sort_values("transact_time")
    # the FIRST continuous print per symbol-day -- what the buffer costs today
    C = cont.groupby(["date", "symbol"]).agg(
        first_cont=("price", "first"),
        n_cont=("price", "size"))

    # ---- join, continuous-led --------------------------------------------
    # LEFT join from the continuous side, because a symbol-day with no
    # continuous print is not a tradeable day at all and should not appear
    out = C.join(A, how="left")
    # days with no auction trade get 0, not NaN -- the count is truly zero
    out["n_auction"] = out["n_auction"].fillna(0).astype(int)
    # how far the two references are apart, in percent
    out["gap_pct"] = 100.0 * (out["first_cont"] - out["auction_vwap"]) \
        / out["auction_vwap"].where(out["auction_vwap"] > 0)
    # a readable column order
    out = out.reset_index()[["date", "symbol", "n_auction", "auction_qty",
                             "auction_vwap", "first_cont", "gap_pct",
                             "n_cont"]]
    # print the table
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ---- the verdict, stated rather than left to the reader --------------
    # how many symbol-days matched nothing at the open
    n_missing = int((out["n_auction"] == 0).sum())
    # the headline
    print(f"\n{len(out)} symbol-days, {n_missing} with NO auction trade at all")
    # what that means
    if n_missing == 0:
        print("  Every symbol-day matched something at the open.")
    else:
        print("  On those days the opening auction matched nothing: either no")
        print("  orders were entered, or none overlapped in price. Anything")
        print("  that assumes an opening auction price needs a fallback.")

    # ---- how much size is actually available at the open? ----------------
    # the matched size, where there was one
    q = out["auction_qty"].dropna()
    # only report when at least one day matched something
    if len(q):
        print(f"\n  shares matched at the open: {q.median():,.0f} median, "
              f"{q.min():,.0f} smallest, {q.max():,.0f} largest")
        print("  THIS IS THE NUMBER THAT VARIES BY NAME. Submitting size that")
        print("  is a large fraction of what would otherwise match does not")
        print("  get you that size at the printed price -- you move the")
        print("  clearing price and become it. Size that is a small fraction")
        print("  of the match is genuinely available at one price, with no")
        print("  spread to pay, which is why the open matters more in a")
        print("  liquid name than the continuous session's touch suggests.")

    # ---- how much the reference moves if we switch -----------------------
    # the gaps, where both numbers existed
    gaps = out["gap_pct"].dropna()
    # only when at least one day had both
    if len(gaps):
        print(f"\n  first continuous print vs auction vwap:")
        print(f"    mean {gaps.mean():+.3f}%, median {gaps.median():+.3f}%, "
              f"sd {gaps.std(ddof=1):.3f}%, worst {gaps.abs().max():.3f}%")
        print("  A positive number means the first continuous trade is ABOVE")
        print("  the auction price -- the open gapped up off the auction. A")
        print("  mean near zero with a wide spread is noise; a mean that is a")
        print("  large fraction of the spread is a systematic step, and that")
        print("  step is a cost or an edge depending on which side you are on.")


# entry point
if __name__ == "__main__":
    main()
