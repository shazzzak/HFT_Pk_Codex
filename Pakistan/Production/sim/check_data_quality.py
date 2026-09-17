# ============================================================================
# check_data_quality.py -- is the parsed store fit to draw conclusions from?
# ============================================================================
# WHY THIS EXISTS, AND WHY IT SHOULD HAVE EXISTED FIRST.
#
# Every result in this project was computed before anyone asked whether the
# data underneath it is internally consistent. The reconcile gate then found
# 24 snapshots where the best bid was at or above the best ask -- which cannot
# happen on a real exchange -- and the explanation offered for them was a
# guess, not a measurement. This file replaces the guess.
#
# WHAT IT CHECKS, and what a failure would mean for the results:
#
#   BOOK INTEGRITY
#     crossed book      best bid >= best ask during CONTINUOUS trading. Cannot
#                       exist at the exchange. Either the feed sent it, the
#                       parser mangled it, or events are being applied out of
#                       order. Every mid price computed on such a snapshot is
#                       meaningless, and mid drives the strategy's every quote.
#     locked book       best bid == best ask. Legal on some venues, and worth
#                       counting separately from crossed rather than lumped in.
#     levels misordered bids must descend and asks must ascend by level. If
#                       they do not, "level 1" is not the touch and every depth
#                       calculation reads the wrong rows.
#     non-positive qty  a book level with zero or negative size is not a level.
#
#   CIRCUIT LIMITS
#     band vs prev_close   the published upper limit against prev_close x 1.1
#                          and lower against x 0.9. THIS IS A CHECK, NOT A
#                          FORMULA: the exchange bands off the ADJUSTED close,
#                          so a stock split legitimately breaks the arithmetic.
#                          Mismatches are counted and shown, never corrected.
#     quote outside band   a bid above the upper limit or an ask below the
#                          lower one. The matching engine rejects those, so
#                          their presence means the band or the book is wrong.
#
#   SEQUENCE INTEGRITY
#     gaps              missing ApplSeqNum within a channel. The FIX spec makes
#                       this detectable on purpose; a gap is a known-missing
#                       message, which means the reconstructed book is missing
#                       an add or a cancel and every queue position after it is
#                       wrong.
#     duplicates        the same ApplSeqNum twice in a channel.
#     time inversions   messages whose sequence order disagrees with their
#                       exchange timestamps. The engine sorts by one of them;
#                       if they disagree, it is applying events in an order the
#                       exchange did not.
#
#   TIME SANITY
#     captured early    capture_ts earlier than transact_time: we received a
#                       message before the exchange sent it. Impossible, and it
#                       would mean the latency model is calibrated on nonsense.
#
#   TRADES
#     outside the band  a trade printed beyond the published circuit limits.
#     unknown resting   a trade naming a resting order that was never added in
#                       ob_updates, so the queue drain has nothing to drain.
#
# SEQUENCE CHECKS READ EVERY SYMBOL, DELIBERATELY. ApplSeqNum is assigned per
# CHANNEL, and a channel carries many instruments -- so filtering to two names
# and then looking for gaps would report a gap at every message belonging to
# some other symbol. Those checks therefore read the whole partition, with
# only the four columns they need.
#
# READ-ONLY. Writes ONE timestamped CSV. Never overwrites, never deletes.
#
# Run from existing_mm_live/:
#   caffeinate -is python check_data_quality.py
#   caffeinate -is python check_data_quality.py --names NRL,MLCF --days 5
#   caffeinate -is python check_data_quality.py --names ALL --days 1
# ============================================================================

# command-line flags
import argparse
# path handling for the parquet store
from pathlib import Path
# the date is parsed out of the filename
import re

# numeric
import numpy as np
# frames
import pandas as pd

# the project's single source of paths -- never hardcoded in a script
try:
    # PARSED_ROOT is the raw store; RESULTS_ROOT is where tools write
    from config_pk import PARSED_ROOT, RESULTS_ROOT
# no config_pk on the path -> stop with an explicit message
except Exception as _e:                                       # noqa: BLE001
    raise ImportError(
        "check_data_quality: could not import paths from config_pk (%r). Run "
        "from the existing_mm_live/ dir, or add it to sys.path." % _e)

# the partition naming the parser uses, per table
DATE_RE = {t: re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})_{t}\.parquet$")
           for t in ("trades", "ob_updates", "ob_snapshot")}

# THE SAMPLE THE GATE USES, so the data behind the gate is what gets checked.
# Ranked by measured net P&L over the 113-name run.
DEFAULT_NAMES = ["NRL", "MLCF"]

# One paisa. Two prices closer together than this are the same price, and the
# comparison has to allow for it because the feed carries more decimals than
# the paisa grid.
PAISA = 0.005


def partitions(table, n_days):
    """The most recent n_days partitions of one table, newest last."""
    # every partition of this table, filename order = date order
    files = sorted(Path(PARSED_ROOT).rglob(f"*_{table}.parquet"))
    # nothing to read is a clear message, not a stack trace
    if not files:
        raise SystemExit(f"no *_{table}.parquet under {PARSED_ROOT}")
    # the most recent N
    return files[-n_days:]


def read(path, table, cols, names=None):
    """One partition, only the columns asked for, optionally one symbol set."""
    # the symbol filter is pushed into the reader so whole row groups for
    # other symbols are never decompressed
    df = pd.read_parquet(
        path, columns=cols,
        filters=([("symbol", "in", list(names))] if names else None))
    # belt and braces in case the reader ignored the pushdown
    if names and "symbol" in df.columns:
        df = df[df["symbol"].isin(names)]
    # the trading date, from the filename the parser controls
    m = DATE_RE[table].match(path.name)
    # an unexpected filename is a stop, not a guess
    if m is None:
        raise SystemExit(f"{path.name} does not match the expected naming")
    # stamp it, since the file itself has no date column
    df["date"] = m.group(1)
    # done
    return df


def add(findings, date, table, check, n, detail=""):
    """Record one check's result. Zero is recorded too -- a check that ran and
    found nothing is different from a check that never ran, and only one of
    those is reassuring."""
    findings.append({"date": date, "table": table, "check": check,
                     "count": int(n), "detail": detail})


# ---------------------------------------------------------------------------
# BOOK INTEGRITY
# ---------------------------------------------------------------------------
def check_book(snap, date, findings, examples, level_examples):
    """Crossed and locked books, level ordering, and empty levels."""
    # the two entry types that are the order book itself
    book = snap[snap["entry_type_code"].astype("string").isin(["0", "1"])]
    # nothing to check on a day with no book rows
    if book.empty:
        add(findings, date, "ob_snapshot", "book_rows", 0, "no book entries")
        return
    # how many book rows were examined, so a zero elsewhere is interpretable
    add(findings, date, "ob_snapshot", "book_rows", len(book))

    # ---- crossed and locked, per snapshot message ----------------------
    # CONTINUOUS TRADING ONLY. A crossed book during the pre-open auction is
    # NORMAL and expected: orders priced at the circuit limits sit on both
    # sides because nothing matches until the auction clears. Counting those
    # as faults would bury the ones that matter.
    cont = book[book["phase"] == "CONTINUOUS_AUCTION"]
    # the touch of each side, per snapshot message per symbol
    if not cont.empty:
        # best bid = highest bid price in that message
        bids = (cont[cont["entry_type_code"].astype("string") == "0"]
                .groupby(["symbol", "snap_key"])["px"].max())
        # best ask = lowest ask price in that message
        asks = (cont[cont["entry_type_code"].astype("string") == "1"]
                .groupby(["symbol", "snap_key"])["px"].min())
        # only messages carrying both sides can be crossed
        both = pd.concat([bids.rename("bid"), asks.rename("ask")],
                         axis=1).dropna()
        # how many two-sided continuous snapshots were examined
        add(findings, date, "ob_snapshot", "continuous_two_sided_snapshots",
            len(both))
        # CROSSED: best bid strictly above best ask, beyond a paisa
        crossed = both[both["bid"] - both["ask"] > PAISA]
        add(findings, date, "ob_snapshot", "crossed_book", len(crossed),
            "best bid above best ask during continuous trading")
        # LOCKED: the two sides at the same price
        locked = both[(both["bid"] - both["ask"]).abs() <= PAISA]
        add(findings, date, "ob_snapshot", "locked_book", len(locked),
            "best bid equal to best ask during continuous trading")
        # ---- IS IT ONE LEVEL OR THE WHOLE LADDER? -----------------------
        # A one-tick inversion between events is a different animal from a bid
        # ladder sitting three rupees through the offers. Counting the levels
        # that are inverted separates them, and this is the number that says
        # whether the reconstruction is momentarily behind or the snapshot is
        # simply wrong.
        if len(crossed):
            # the snapshots that are crossed, as a lookup
            bad_keys = set(crossed.index)
            # every book row belonging to one of them
            rows = cont.set_index(["symbol", "snap_key"])
            # restricted to the crossed messages
            sub = rows[rows.index.isin(bad_keys)].reset_index()
            # per crossed snapshot: how many bid levels sit at or above the
            # best ask, which is the depth of the inversion
            depth = []
            # walk each crossed snapshot once
            for (sym, seq), g in sub.groupby(["symbol", "snap_key"]):
                # that snapshot's best ask
                a = g[g["entry_type_code"].astype("string") == "1"]["px"].min()
                # its bid levels sitting at or through that ask
                nb = int((g[g["entry_type_code"].astype("string") == "0"]["px"]
                          >= a - PAISA).sum())
                # and how many levels the bid side published at all
                tot = int((g["entry_type_code"].astype("string") == "0").sum())
                # keep both, so "2 of 10" reads differently from "10 of 10"
                depth.append({"symbol": sym, "snap_key": seq,
                              "bid_levels_through_ask": nb,
                              "bid_levels_total": tot})
            # as a frame
            D = pd.DataFrame(depth)
            # the headline: is it the touch only, or the whole ladder?
            add(findings, date, "ob_snapshot", "crossed_touch_only",
                int((D["bid_levels_through_ask"] <= 1).sum()),
                "crossed by the best bid alone -- consistent with the "
                "reconstruction being one event behind")
            add(findings, date, "ob_snapshot", "crossed_multiple_levels",
                int((D["bid_levels_through_ask"] > 1).sum()),
                "more than one bid level through the offer -- NOT a one-event "
                "lag; the snapshot itself is inconsistent")

        # keep a few crossed examples, because the count alone says nothing
        # about whether it is one symbol or all of them
        if len(crossed):
            # the worst ones first, by how far through they are
            worst = crossed.assign(
                through=crossed["bid"] - crossed["ask"]
            ).sort_values("through", ascending=False).head(5).reset_index()
            # tag them so the output file says where they came from
            worst["date"] = date
            examples.append(worst)

    # ---- level ordering ------------------------------------------------
    # bids must DESCEND with level and asks must ASCEND. If they do not, then
    # level 1 is not the touch and every depth read is off.
    bad_order = 0
    # examine each side separately, because the expected direction differs
    for code, ascending in (("0", False), ("1", True)):
        # that side's rows, with a usable level
        side = book[(book["entry_type_code"].astype("string") == code)
                    & book["level"].notna()]
        # nothing on this side
        if side.empty:
            continue
        # within one message and symbol, sorted by level, the price must move
        # monotonically in the expected direction
        ordered = side.sort_values(["symbol", "snap_key", "level"])
        # the price step from one level to the next, within a group
        step = ordered.groupby(["symbol", "snap_key"])["px"].diff()
        # a violation is a step in the wrong direction beyond a paisa
        bad = (step > PAISA) if not ascending else (step < -PAISA)
        # count them
        bad_order += int(bad.sum())
        # KEEP THE ROWS THEMSELVES. A count says 6,347 and nothing else; the
        # rows say whether this is one symbol, one message, or the whole day,
        # and whether the jump is a paisa or a rupee. Guessing at a cause from
        # a count is how the last explanation turned out to be invented.
        if bad.any():
            # the offending rows with the step that flagged them
            ex = ordered[bad].copy()
            # what the price did from the level above
            ex["step_from_prev_level"] = step[bad]
            # which side, in words
            ex["side"] = "BID" if not ascending else "OFFER"
            # keep the biggest jumps, which are the most diagnostic
            level_examples.append(
                ex.reindex(ex["step_from_prev_level"].abs()
                           .sort_values(ascending=False).index).head(5))
    # one line for both sides
    add(findings, date, "ob_snapshot", "levels_misordered", bad_order,
        "bids not descending or asks not ascending by level")

    # ---- empty levels ---------------------------------------------------
    # a level with no size is not a level
    empty = int((book["qty"].fillna(0) <= 0).sum())
    add(findings, date, "ob_snapshot", "level_qty_not_positive", empty,
        "a book level with zero or negative quantity")


# ---------------------------------------------------------------------------
# CIRCUIT LIMITS
# ---------------------------------------------------------------------------
def check_bands(snap, date, findings, band_examples):
    """The published limits against 10% of the previous close, and quotes
    sitting outside them.

    THE ARITHMETIC IS A CHECK, NOT A FORMULA, AND IT IS NOT IMPLEMENTED
    ANYWHERE ELSE. The exchange sets the band off the ADJUSTED previous close,
    so a stock split or reverse split legitimately breaks the +/-10% identity.
    A mismatch here means "look at this symbol", never "the exchange is wrong",
    and this file never substitutes its own arithmetic for the published value.
    """
    # the published upper and lower limits
    up = snap[snap["entry_type_code"].astype("string") == "xe"]
    dn = snap[snap["entry_type_code"].astype("string") == "xf"]
    # the last published pair per symbol, with the previous close beside it
    rows = []
    # walk each symbol once
    for sym, g in snap.groupby("symbol"):
        # the previous close the exchange published for this symbol
        pc = g["prev_close"].dropna()
        # without it there is nothing to compare against
        if pc.empty:
            continue
        # one value: it does not change within a day
        prev_close = float(pc.iloc[0])
        # a zero or missing previous close cannot anchor a band
        if prev_close <= 0:
            continue
        # this symbol's published limits
        u = up[up["symbol"] == sym]["px"].dropna()
        d = dn[dn["symbol"] == sym]["px"].dropna()
        # both are needed for the comparison
        if u.empty or d.empty:
            continue
        # the published values
        upper, lower = float(u.iloc[0]), float(d.iloc[0])
        # what a flat 10% band off the unadjusted close would be
        exp_up, exp_dn = prev_close * 1.1, prev_close * 0.9
        # how far each published value is from that, in percent of the close
        rows.append({"date": date, "symbol": sym, "prev_close": prev_close,
                     "published_upper": upper, "published_lower": lower,
                     "tenpct_upper": exp_up, "tenpct_lower": exp_dn,
                     "upper_diff_pct": 100.0 * (upper - exp_up) / prev_close,
                     "lower_diff_pct": 100.0 * (lower - exp_dn) / prev_close})
    # nothing comparable on this day
    if not rows:
        add(findings, date, "ob_snapshot", "band_vs_prev_close", 0,
            "no symbol had both a previous close and published limits")
        return
    # as a frame
    B = pd.DataFrame(rows)
    # a mismatch is either side off by more than a paisa's worth
    off = B[(B["upper_diff_pct"].abs() > 0.01)
            | (B["lower_diff_pct"].abs() > 0.01)]
    # how many symbols were compared, and how many disagreed
    add(findings, date, "ob_snapshot", "band_symbols_compared", len(B))
    add(findings, date, "ob_snapshot", "band_vs_prev_close", len(off),
        "published limit not 10% of the UNADJUSTED previous close -- "
        "expected after a split; investigate, do not 'fix'")
    # keep the mismatches themselves, because the count is not the finding
    if len(off):
        band_examples.append(off)

    # ---- quotes outside the published band ------------------------------
    # the book rows, which must sit inside the limits
    book = snap[snap["entry_type_code"].astype("string").isin(["0", "1"])]
    # nothing to check
    if book.empty or B.empty:
        return
    # the limits, per symbol, to join on
    lim = B.set_index("symbol")[["published_upper", "published_lower"]]
    # attach each book row's limits
    j = book.join(lim, on="symbol")
    # a bid above the ceiling or an ask below the floor should be impossible:
    # the matching engine rejects orders priced outside the band
    outside = int(((j["px"] > j["published_upper"] + PAISA)
                   | (j["px"] < j["published_lower"] - PAISA)).sum())
    # report it
    add(findings, date, "ob_snapshot", "quote_outside_band", outside,
        "a book level priced beyond the published circuit limits")


# ---------------------------------------------------------------------------
# SEQUENCE INTEGRITY -- every symbol, never a subset
# ---------------------------------------------------------------------------
def check_sequence(df, table, date, findings, gap_examples):
    """Gaps, duplicates, and sequence-versus-time disagreement, per channel.

    READS EVERY SYMBOL, AND EVERY TABLE THAT SHARES THE SEQUENCE, BY DESIGN.

    CORRECTED 2026-09-17, after the first real run reported 8,710,890 gaps
    against 3,408,253 rows -- more missing messages than messages, which is
    impossible, and was a bug HERE rather than a fault in the store.

    ApplSeqNum is assigned per CHANNEL, running continuously across the whole
    message stream. The parser then SPLITS that one stream into four tables:
    UA202 executions to trades, UA201 adds and UA202 cancels to ob_updates,
    everything else to misc. So every ob_updates message looks like a missing
    number when you examine trades alone, and vice versa. Counting gaps table
    by table counts the other table's messages as losses.

    The caller therefore hands this the CONCATENATION of the tables that carry
    appl_seq, never one table at a time.

    A subset of SYMBOLS is wrong for the same reason: a channel carries many
    instruments, so two names would report a gap at every message belonging to
    any other name on that channel.
    """
    # no sequence column means nothing to check
    if "appl_seq" not in df.columns or df.empty:
        add(findings, date, table, "sequence_rows", 0, "no appl_seq column")
        return
    # rows carrying a sequence number
    d = df[df["appl_seq"].notna()]
    # how many were examined
    add(findings, date, table, "sequence_rows", len(d))
    # nothing to do
    if d.empty:
        return
    # totals across every channel
    gaps = dupes = inversions = 0
    # each channel is its own sequence space
    for ch, g in d.groupby("channel"):
        # the sequence numbers, in the order the exchange assigned them
        seq = g["appl_seq"].astype("int64").sort_values()
        # DUPLICATES: the same number twice in one channel
        dupes += int(seq.duplicated().sum())
        # GAPS: the span covered, minus the distinct numbers actually present
        distinct = seq.nunique()
        # span from first to last, inclusive
        span = int(seq.iloc[-1] - seq.iloc[0]) + 1
        # whatever is missing from that span
        gaps += max(0, span - distinct)
        # TIME INVERSIONS: sort by sequence, and the exchange timestamps must
        # not go backwards. If they do, sorting by time and sorting by sequence
        # give different books, and the engine sorts by one of them.
        if "transact_time" in g.columns:
            # in sequence order
            t = g.sort_values("appl_seq")["transact_time"]
            # a step backwards in time
            inversions += int((t.diff() < pd.Timedelta(0)).sum())
    # ---- WHERE IN THE DAY ARE THE GAPS? ---------------------------------
    # THIS IS THE QUESTION THE TOTAL CANNOT ANSWER AND THE ONLY ONE THAT
    # DECIDES ANYTHING.
    #
    # A block of missing sequence numbers BEFORE the first continuous print,
    # or AFTER the last one, costs nothing: the capture started late or
    # stopped early, and no quoting decision depended on those messages. The
    # same block in the middle of the session means the reconstructed book was
    # missing adds and cancels while the strategy was quoting against it, and
    # every queue position after it is wrong.
    #
    # Same number of missing messages. Opposite conclusions. So the breaks are
    # reported with the exchange timestamps either side of them.
    breaks = []
    # each channel separately, since each has its own sequence
    for ch, g in d.groupby("channel"):
        # sequence and time together, in sequence order
        gg = (g[["appl_seq", "transact_time"]].dropna()
              .astype({"appl_seq": "int64"})
              .sort_values("appl_seq").drop_duplicates("appl_seq"))
        # nothing to compare
        if len(gg) < 2:
            continue
        # the step between consecutive sequence numbers
        step = gg["appl_seq"].diff()
        # the rows that follow a break
        after = gg[step > 1]
        # and the rows that precede one
        before = gg.shift(1)[step > 1]
        # one record per break
        for (_, a), (_, b), n in zip(after.iterrows(), before.iterrows(),
                                     (step[step > 1] - 1)):
            breaks.append({"channel": ch, "missing": int(n),
                           "last_seen_at": b["transact_time"],
                           "resumed_at": a["transact_time"]})
    # ---- HOW MUCH OF THE SESSION WAS THE FEED DOWN? ---------------------
    # WITHDRAWN 2026-09-17. The measure that used to live here was wrong.
    #
    # It took the time between the two messages surrounding a SEQUENCE BREAK
    # and called that an outage. On a low-traffic channel one missing message
    # can sit between two messages that are naturally minutes apart -- that is
    # one lost message in a quiet period, not a blackout. Summing those spans
    # produced 8,040 seconds and a claim that 12.5% of the day had a frozen
    # book, while every break actually listed spanned 20 to 33 seconds. The
    # two could not both be true, and the sum was the wrong one.
    #
    # Outage duration is now measured directly, in check_feed_liveness, from
    # message ARRIVAL times across every table. The feed sends a heartbeat
    # every 3 seconds per channel, so a stretch with no message of any kind is
    # a receiver that stopped, measured rather than inferred.
    #
    # ---- (the old block, kept for the break shape only) -----------------
    # THE NUMBER THAT DECIDES WHETHER THE STORE IS USABLE.
    #
    # The evidence says these are CAPTURE OUTAGES, not exchange losses:
    # several independent channels stop at the same microsecond and resume
    # together. An exchange dropping messages would hit one channel.
    #
    # What matters is not how many messages went missing -- it is how many
    # SECONDS of the trading session the book was frozen while the strategy
    # went on quoting against it. Sum the outage durations, and only the ones
    # inside continuous trading.
    #
    # PSX continuous trading is 09:32-15:30 PKT = 04:32-10:30 UTC. Breaks
    # whose two timestamps are equal are the session boundary itself, not an
    # outage, and are excluded.
    if breaks:
        # as a frame
        _B = pd.DataFrame(breaks)
        # how long each break lasted
        _B["seconds"] = (_B["resumed_at"] - _B["last_seen_at"]
                         ).dt.total_seconds()
        # the hour of day the break started, in UTC
        _hh = _B["last_seen_at"].dt.hour + _B["last_seen_at"].dt.minute / 60.0
        # inside continuous trading, and an actual outage rather than the
        # boundary jump at the close
        _in = _B[(_hh >= 4.533) & (_hh <= 10.5) & (_B["seconds"] > 0.5)]
        # DEDUPLICATED BY TIME. The same outage appears once per channel --
        # three channels stopping together is one outage, not three -- so
        # counting each channel's break separately would treble the total.
        _u = (_in.assign(t=_in["last_seen_at"].dt.floor("s"))
              .groupby("t")["seconds"].max())
        # the totals

    # keep them for the report, biggest first
    if breaks:
        # as a frame
        B = pd.DataFrame(breaks).sort_values("missing", ascending=False)
        # tagged with the date so several days can be read together
        B["date"] = date
        # hand them up
        gap_examples.append(B.head(8))

    # ---- WHAT SHAPE ARE THE GAPS? ---------------------------------------
    # A FEW ENORMOUS GAPS AND MANY SMALL ONES ARE DIFFERENT DIAGNOSES, and the
    # total cannot tell them apart. One gap of 400,000 is a sequence reset or a
    # capture that started late -- nothing was lost mid-session. Four hundred
    # thousand gaps of one are genuinely dropped messages, and the book is
    # missing an add or a cancel four hundred thousand times.
    #
    # The distribution is the diagnosis; the sum is not.
    biggest = 0
    n_runs = 0
    singles = 0
    # each channel again, now looking at the steps rather than the total
    for ch, g in d.groupby("channel"):
        # the sequence numbers in order
        seq = g["appl_seq"].astype("int64").sort_values().drop_duplicates()
        # the step from one to the next
        step = seq.diff()
        # a step above 1 is a gap of (step - 1) missing numbers
        holes = step[step > 1] - 1
        # nothing missing on this channel
        if holes.empty:
            continue
        # how many separate breaks, the largest, and how many are a single
        # missing message
        n_runs += len(holes)
        biggest = max(biggest, int(holes.max()))
        singles += int((holes == 1).sum())
    # each as its own line, because each means something different
    add(findings, date, table, "sequence_break_count", n_runs,
        "separate breaks in the sequence, regardless of how many messages "
        "each one spans")
    add(findings, date, table, "sequence_largest_break", biggest,
        "the biggest single break; one enormous break is a reset or a late "
        "capture start, not lost traffic")
    add(findings, date, table, "sequence_single_message_gaps", singles,
        "breaks of exactly one message -- these are the ones that mean a "
        "genuinely dropped add or cancel")
    # one line each, with the consequence spelled out
    add(findings, date, table, "sequence_gaps", gaps,
        "missing ApplSeqNum: a lost add or cancel, so the reconstructed "
        "book and every queue position after it is wrong")
    add(findings, date, table, "sequence_duplicates", dupes,
        "the same ApplSeqNum twice in one channel")
    add(findings, date, table, "sequence_time_inversions", inversions,
        "exchange time goes backwards when sorted by sequence")


# ---------------------------------------------------------------------------
# TIME SANITY
# ---------------------------------------------------------------------------
def check_feed_liveness(paths, date, findings, outage_examples):
    """For how long did the capture receive NOTHING? Measured, not inferred.

    THE ONLY HONEST WAY TO MEASURE AN OUTAGE. Every sequence-based attempt in
    this file reasons from missing numbers, and a missing number tells you a
    message was lost -- it says nothing about how long the receiver was down.
    A previous version of this check confused the two and reported 12.5% of
    the trading day as a frozen book, from breaks that actually spanned half a
    minute each.

    The feed sends a HEARTBEAT EVERY 3 SECONDS PER CHANNEL. So if the capture
    is alive, something arrives at least that often. A stretch of wall-clock
    time in which NO message of ANY kind arrived -- not a trade, not an update,
    not a snapshot, not a heartbeat -- is the receiver having stopped. That is
    a fact about arrival times and requires no inference at all.

    Measured on capture_ts (when WE received it), not transact_time (when the
    exchange sent it), because the question is about our receiver.
    """
    # every arrival timestamp we hold, from every table
    stamps = []
    # each table in turn; a missing one is skipped rather than fatal
    for label, path in paths.items():
        # not every store carries every table
        if not path.exists():
            continue
        # only the arrival clock is needed
        try:
            # capture_ts is on all four tables
            col = pd.read_parquet(path, columns=["capture_ts"])["capture_ts"]
        # a table without it contributes nothing rather than breaking the run
        except Exception:                                     # noqa: BLE001
            continue
        # keep them
        stamps.append(col.dropna())
    # nothing to measure
    if not stamps:
        add(findings, date, "all", "liveness_rows", 0, "no capture_ts anywhere")
        return
    # one sorted series of arrival times across the whole capture
    t = pd.concat(stamps, ignore_index=True).sort_values().reset_index(drop=True)
    # how many messages the measurement rests on
    add(findings, date, "all", "liveness_rows", len(t))
    # the wall-clock gap between consecutive arrivals
    gap = t.diff().dt.total_seconds()
    # THE THRESHOLD: the heartbeat is every 3 seconds per channel, so two
    # missed intervals is the standard disconnect test. 7 seconds allows for
    # jitter without letting a real stall through.
    SILENT = 7.0
    # the stretches that exceed it
    idx = gap[gap > SILENT].index
    # none is the good answer and is worth recording as such
    if len(idx) == 0:
        add(findings, date, "all", "feed_silent_periods", 0,
            f"no stretch longer than {SILENT:.0f}s without a single message")
        add(findings, date, "all", "feed_silent_seconds_in_session", 0)
        return
    # each silence, with the times either side
    rows = pd.DataFrame({"silent_from": t[idx - 1].values,
                         "silent_to": t[idx].values,
                         "seconds": gap[idx].values})
    # CONTINUOUS TRADING ONLY. Silence before the open or after the close is
    # the capture starting late or stopping early and costs nothing.
    hh = rows["silent_from"].dt.hour + rows["silent_from"].dt.minute / 60.0
    # 04:32-10:30 UTC is 09:32-15:30 PKT
    inside = rows[(hh >= 4.533) & (hh <= 10.5)]
    # the three numbers that decide whether the store is usable
    add(findings, date, "all", "feed_silent_periods", len(inside),
        f"stretches during continuous trading with no message of any kind "
        f"for more than {SILENT:.0f}s -- the receiver stopped")
    add(findings, date, "all", "feed_silent_seconds_in_session",
        int(inside["seconds"].sum()),
        "SECONDS of continuous trading with nothing arriving. The session is "
        "21,480 seconds, so divide for the share of the day the book was "
        "frozen")
    add(findings, date, "all", "feed_silent_longest_seconds",
        int(inside["seconds"].max()) if len(inside) else 0,
        "the longest single silence during trading")
    # ---- ARE THE OUTAGES AT THE BUSY MOMENTS? ---------------------------
    # THE QUESTION THAT DECIDES WHETHER EXCLUDING THEM IS SAFE.
    #
    # If the receiver drops under load, the outages land on the busiest
    # periods -- and cutting them from the sample would quietly delete the
    # hardest trading of the day and flatter every result. If they land
    # wherever a clock falls, cutting them is unbiased.
    #
    # Measured directly: the message rate in the 60 seconds BEFORE each
    # silence, against the median 60-second rate across the whole session.
    # A ratio above 1 means outages happen when it is busier than usual.
    if len(inside):
        # every arrival time in the continuous session
        _sess = t[(t.dt.hour + t.dt.minute / 60.0 >= 4.533)
                  & (t.dt.hour + t.dt.minute / 60.0 <= 10.5)]
        # messages per minute across the session, as a baseline
        _per_min = _sess.dt.floor("min").value_counts()
        # the typical minute
        _median_rate = float(_per_min.median()) if len(_per_min) else 0.0
        # the rate in the minute before each silence began
        _rates = []
        # each silence in turn
        for _from in inside["silent_from"]:
            # the window immediately before it
            _win = _sess[(_sess >= _from - pd.Timedelta(seconds=60))
                         & (_sess < _from)]
            # how many messages arrived in that minute
            _rates.append(len(_win))
        # the comparison, as a ratio
        _ratio = ((sum(_rates) / len(_rates)) / _median_rate
                  if _median_rate > 0 else 0.0)
        # reported to two decimals as a percentage, so 100 means "typical"
        add(findings, date, "all", "busyness_before_outage_pct",
            int(round(100 * _ratio)),
            "message rate in the 60s before each outage, as a PERCENTAGE of "
            "the median minute. ~100 means outages happen at ordinary times "
            "and excluding them is unbiased; well above 100 means the "
            "receiver drops under load and excluding them would delete the "
            "busiest trading")

    # keep the worst for the report
    if len(inside):
        # biggest first
        ex = inside.sort_values("seconds", ascending=False).head(8).copy()
        # tagged with the date
        ex["date"] = date
        outage_examples.append(ex)


def check_heartbeat_sequence(misc_path, seq_df, date, findings):
    """Compare what we HAVE against what the exchange SAID it sent.

    THE ONLY NON-INFERENTIAL GAP CHECK. Everything else in this file reasons
    about gaps from the numbers we hold: if we have 100 and 102, we conclude
    101 is missing. That reasoning collapses if the sequence is not dense, if
    it resets, or if some message type we route elsewhere also consumes a
    number. It has already been wrong once here.

    The UA001 heartbeat carries ApplLastSeqNum (tag 1350) -- the exchange
    stating the last sequence number it has sent on that channel. The parser
    promotes it to `appl_last_seq`. Comparing the exchange's own high-water
    mark against ours answers the question directly:

      ours == theirs   we have everything up to that point, and any "gaps"
                       this file reports are an artefact of the counting.
      ours <  theirs   we are genuinely behind and messages were lost.
      ours >  theirs   our sequence space is not the one the heartbeat refers
                       to, and the comparison itself is meaningless.
    """
    # the heartbeats live in misc
    if not misc_path.exists():
        add(findings, date, "misc", "heartbeat_check", 0,
            "no misc partition, so the exchange's own count is unavailable")
        return
    # only the two columns needed
    hb = pd.read_parquet(misc_path, columns=["msg_type", "channel",
                                             "appl_last_seq"])
    # the rows that actually carry the declaration
    hb = hb[hb["appl_last_seq"].notna()]
    # none is a reportable state
    if hb.empty:
        add(findings, date, "misc", "heartbeat_check", 0,
            "no heartbeat carried appl_last_seq")
        return
    # the exchange's high-water mark per channel
    theirs = hb.groupby("channel")["appl_last_seq"].max()
    # ours, from the message stream itself
    ours = seq_df.dropna(subset=["appl_seq"]).groupby(
        "channel")["appl_seq"].max()
    # compare only the channels both know about
    common = theirs.index.intersection(ours.index)
    # nothing comparable
    if len(common) == 0:
        add(findings, date, "misc", "heartbeat_check", 0,
            "no channel appears in both the heartbeats and the message stream")
        return
    # the shortfall per channel: how many the exchange says it sent beyond
    # the last one we hold
    behind = (theirs[common] - ours[common]).astype("int64")
    # the total, which is the number that matters
    add(findings, date, "misc", "messages_behind_the_exchange",
        int(behind.clip(lower=0).sum()),
        "the exchange's own ApplLastSeqNum minus the highest we hold, per "
        "channel. NOT inferred from gaps -- this is the exchange saying what "
        "it sent")
    # a negative means our numbers run ahead of the heartbeat's, which means
    # the two are not the same sequence space and no gap arithmetic on them
    # means anything
    add(findings, date, "misc", "sequence_space_mismatch",
        int((behind < 0).sum()),
        "channels where OUR highest sequence exceeds the exchange's declared "
        "last -- the heartbeat counts a different space, so every gap number "
        "computed from it is meaningless")


def check_time(df, table, date, findings):
    """Messages we appear to have received before the exchange sent them."""
    # both timestamps are needed
    if not {"capture_ts", "transact_time"} <= set(df.columns) or df.empty:
        return
    # rows carrying both
    d = df.dropna(subset=["capture_ts", "transact_time"])
    # nothing to check
    if d.empty:
        return
    # received strictly before sent, by more than a millisecond of slop
    early = int((d["capture_ts"] < d["transact_time"]
                 - pd.Timedelta(milliseconds=1)).sum())
    # this would invalidate the latency model, which is calibrated on the gap
    add(findings, date, table, "captured_before_sent", early,
        "capture_ts earlier than transact_time; the latency model is "
        "calibrated on this difference")


# ---------------------------------------------------------------------------
# TRADES
# ---------------------------------------------------------------------------
def check_trades(tr, upd, date, findings):
    """Trade prices, sizes, and whether the order they hit was ever seen."""
    # nothing to check
    if tr.empty:
        add(findings, date, "trades", "trade_rows", 0)
        return
    # how many were examined
    add(findings, date, "trades", "trade_rows", len(tr))
    # a trade at zero or negative price or size is not a trade
    bad = int(((tr["price"].fillna(0) <= 0) | (tr["qty"].fillna(0) <= 0)).sum())
    add(findings, date, "trades", "trade_price_or_qty_not_positive", bad)
    # the aggressor side should be present on a continuous print
    cont = tr[tr["initiator"].astype("string").fillna("") != "AUCTION"]
    # how many continuous prints name no aggressor
    no_aggr = int(cont["aggressor_side"].isna().sum())
    add(findings, date, "trades", "continuous_trade_no_aggressor", no_aggr,
        "a continuous print with no taker side; the fill model tests against "
        "this to decide which of our orders could have been hit")

    # ---- did the resting order ever exist? -------------------------------
    # the surgical queue drain consumes exactly the named resting order. If
    # that order was never added in ob_updates, the drain has nothing to find
    # and silently falls back to consuming the pool front-to-back.
    if "resting_order_id" in tr.columns and not upd.empty:
        # trades that name a resting order
        named = tr["resting_order_id"].dropna()
        # how many name one at all -- the rest use the pool path by necessity
        add(findings, date, "trades", "trades_naming_a_resting_order",
            len(named))
        # every order id the updates stream ever added
        if "order_id" in upd.columns:
            # the set of ids that exist
            known = set(upd["order_id"].dropna().astype(str))
            # THE NAMED ID IS A STRINGIFIED TUPLE in this store -- "('...', px)"
            # -- so compare on the leading id rather than the whole string.
            def head(v):
                # pull the first quoted field out of the tuple text
                m = re.search(r"'([^']+)'", str(v))
                # or fall back to the raw value when it is a plain id
                return m.group(1) if m else str(v)
            # how many named a resting order that was never added
            unknown = sum(1 for v in named if head(v) not in known)
            # report it
            add(findings, date, "trades", "resting_order_never_added",
                unknown,
                "a trade naming a resting order that ob_updates never added; "
                "the exact queue drain cannot find it and falls back to the "
                "pool, silently")


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # which symbols the per-symbol checks cover
    ap.add_argument("--names", default=",".join(DEFAULT_NAMES),
                    help="comma-separated symbols, or ALL (slow on snapshots)")
    # how many recent days
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args()
    # the symbol filter, or None for everything
    names = (None if args.names.strip().upper() == "ALL"
             else [s.strip().upper() for s in args.names.split(",") if s.strip()])

    print("=" * 78)
    print("DATA QUALITY -- is the store fit to draw conclusions from?")
    print("=" * 78)
    print(f"  store   : {PARSED_ROOT}")
    print(f"  symbols : {args.names}  (book and trade checks)")
    print(f"  sequence: EVERY symbol -- ApplSeqNum is per CHANNEL, so a subset")
    print(f"            would report another symbol's message as a gap")
    print(f"  days    : {args.days}\n")

    # one row per (date, table, check)
    findings = []
    # crossed-book examples, because a count does not say which symbol
    examples = []
    # circuit-band mismatches, same reason
    band_examples = []
    # misordered price levels, ditto
    level_examples = []
    # sequence breaks with the times either side, which is what decides
    # whether a missing block matters at all
    gap_examples = []
    # stretches where the receiver went silent entirely
    outage_examples = []

    # the snapshot columns the book and band checks need
    SNAP = ["symbol", "msg_seq", "phase", "entry_type_code", "level", "px",
            "qty", "prev_close", "capture_ts", "orig_time", "market"]
    # the update columns
    UPD = ["symbol", "appl_seq", "channel", "transact_time", "capture_ts",
           "order_id", "event", "price", "qty"]
    # the trade columns
    TRD = ["symbol", "appl_seq", "channel", "transact_time", "capture_ts",
           "price", "qty", "initiator", "aggressor_side", "resting_order_id"]

    # walk the most recent days, using the trades partitions as the calendar
    for path in partitions("trades", args.days):
        # the date from this partition's name
        date = DATE_RE["trades"].match(path.name).group(1)
        # progress, because the snapshot read is the slow part
        print(f"  {date} ...", flush=True)
        # the matching partitions of the other two tables
        upd_path = path.parent.parent.parent / "ob_updates" / f"date={date}" \
            / f"{date}_ob_updates.parquet"
        snap_path = path.parent.parent.parent / "ob_snapshot" / f"date={date}" \
            / f"{date}_ob_snapshot.parquet"
        # misc, where the UA001 heartbeats live. The heartbeat carries
        # ApplLastSeqNum -- the exchange's own statement of the last sequence
        # number it sent on that channel -- which is the only gap check in
        # this file that does not depend on its own arithmetic being right.
        # NOTE THE TABLE IS CALLED misc ON DISK, not "other": the parser's
        # header comment says {day}_other.parquet, its internal table map says
        # misc, and misc is what reached the store.
        misc_path = path.parent.parent.parent / "misc" / f"date={date}" \
            / f"{date}_misc.parquet"

        # ---- per-symbol checks -----------------------------------------
        # the trades for the sample
        tr = read(path, "trades", TRD, names)
        # the updates for the sample
        upd = (read(upd_path, "ob_updates", UPD, names)
               if upd_path.exists() else pd.DataFrame())
        # the snapshots for the sample
        snap = (read(snap_path, "ob_snapshot", SNAP, names)
                if snap_path.exists() else pd.DataFrame())
        # ---- GROUP SNAPSHOTS THE WAY THE LOADER DOES, from 2026-09-17 ----
        # msg_seq is NOT unique per 35=W message: two messages for one symbol
        # can share it, and grouping on it alone concatenates their ladders
        # into one book. That is what produced the 285 "crossed books" this
        # checker reported, and it is fixed in run_legacy_mm.build_events by
        # keying on msg_seq|orig_time. This checker has to use the SAME key or
        # it will go on reporting a bug that no longer exists.
        if not snap.empty:
            # the composite, formed exactly as build_events forms it
            snap["snap_key"] = (snap["msg_seq"].astype("int64").astype(str)
                                + "|" + snap["orig_time"].astype(str))
            # REGULAR MARKET ONLY, matching the loader's filter. A symbol can
            # be listed in more than one market under the same ticker and a
            # snapshot from the wrong one replaces the whole book.
            if "market" in snap.columns:
                snap = snap[snap["market"] == "REG"]
        # the book, if there is one
        if not snap.empty:
            check_book(snap, date, findings, examples, level_examples)
            check_bands(snap, date, findings, band_examples)
        # the trades
        check_trades(tr, upd, date, findings)
        # timestamps on both message streams
        check_time(tr, "trades", date, findings)
        # and on the updates
        if not upd.empty:
            check_time(upd, "ob_updates", date, findings)

        # ---- sequence checks: EVERY symbol, BOTH STREAMS TOGETHER -------
        # only the four columns these need, so reading whole partitions is
        # affordable even at two million rows
        seq_cols = ["appl_seq", "channel", "transact_time", "capture_ts"]
        # ONE STREAM, SPLIT ACROSS TWO TABLES BY THE PARSER. ApplSeqNum runs
        # continuously per channel over everything the exchange sent, so the
        # halves have to be put back together before a gap means anything.
        parts = [read(path, "trades", seq_cols)]
        # the other half of the same stream
        if upd_path.exists():
            parts.append(read(upd_path, "ob_updates", seq_cols))
        # the combined stream, used twice
        seq_df = pd.concat(parts, ignore_index=True)
        # checked as one stream, because that is what it is
        check_sequence(seq_df, "trades+ob_updates", date, findings,
                       gap_examples)
        # AND CHECKED AGAINST THE EXCHANGE'S OWN COUNT, which is the only
        # answer that does not depend on this file's arithmetic being right
        check_heartbeat_sequence(misc_path, seq_df, date, findings)
        # AND THE RECEIVER'S OWN LIVENESS, from arrival times across every
        # table -- the direct measure of how long the capture was down
        check_feed_liveness({"trades": path, "ob_updates": upd_path,
                             "ob_snapshot": snap_path, "misc": misc_path},
                            date, findings, outage_examples)

    # nothing ran
    if not findings:
        raise SystemExit("no partitions read; check the store")
    # the results
    F = pd.DataFrame(findings)

    # ---- the report -----------------------------------------------------
    # the checks whose non-zero count is a FAULT, as opposed to a row count
    # NOTE crossed_touch_only and crossed_multiple_levels are DIAGNOSTICS of
    # crossed_book, not separate faults -- they split the same events by cause
    # sequence_break_count / _largest_break / _single_message_gaps are the
    # SHAPE of sequence_gaps, not extra faults -- they split the same number
    FAULTS = ["crossed_book", "levels_misordered", "level_qty_not_positive",
              "quote_outside_band", "sequence_gaps", "sequence_duplicates",
              "sequence_time_inversions", "captured_before_sent",
              "trade_price_or_qty_not_positive",
              "messages_behind_the_exchange", "sequence_space_mismatch",
              "feed_silent_periods", "feed_silent_seconds_in_session",
              "continuous_trade_no_aggressor", "resting_order_never_added"]
    # everything else is context: how much was examined
    print("\n" + "=" * 78)
    print("WHAT WAS EXAMINED")
    print("=" * 78)
    ctx = F[~F["check"].isin(FAULTS + ["band_vs_prev_close", "locked_book"])]
    print(ctx.pivot_table(index="check", columns="date", values="count",
                          aggfunc="sum").to_string())

    print("\n" + "=" * 78)
    print("FAULTS -- a non-zero number here is a problem in the data")
    print("=" * 78)
    # the fault rows
    f = F[F["check"].isin(FAULTS)]
    # summed per check across the sample
    tot = f.groupby("check")["count"].sum().sort_values(ascending=False)
    # print each with its explanation, clean ones included
    for check, n in tot.items():
        # the explanation recorded with the check
        why = f[f["check"] == check]["detail"].iloc[0]
        # a clean check is worth seeing: it ran and found nothing
        mark = "    ok" if n == 0 else f"  {n:>6,}"
        print(f"{mark}  {check}")
        # only explain the ones that fired
        if n:
            print(f"          {why}")

    # the band comparison is reported separately: a mismatch is EXPECTED after
    # a corporate action and is not a fault
    print("\n" + "=" * 78)
    print("CIRCUIT BANDS vs 10% OF THE UNADJUSTED PREVIOUS CLOSE")
    print("=" * 78)
    print("  A mismatch is EXPECTED after a split or bonus issue: the exchange")
    print("  bands off the ADJUSTED close. This is a pointer to look at a")
    print("  symbol, never a licence to compute the band ourselves.")
    # the compared and mismatched counts
    cmp_n = int(F[F["check"] == "band_symbols_compared"]["count"].sum())
    off_n = int(F[F["check"] == "band_vs_prev_close"]["count"].sum())
    print(f"  compared {cmp_n} symbol-days, {off_n} did not match")
    # the mismatches themselves
    if band_examples:
        BE = pd.concat(band_examples, ignore_index=True)
        print(BE.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # WHEN THE RECEIVER WAS SILENT -- the direct measure, no inference
    if outage_examples:
        print("\n" + "=" * 78)
        print("FEED SILENCE -- stretches with NO message of any kind arriving")
        print("=" * 78)
        print("  The feed heartbeats every 3 seconds per channel, so anything")
        print("  longer than 7 seconds with nothing at all arriving is the")
        print("  receiver having stopped. Measured on capture_ts -- when WE")
        print("  received it -- because the question is about our receiver,")
        print("  not about what the exchange sent.")
        print("  Continuous trading only. 21,480 seconds in a session.\n")
        OE = pd.concat(outage_examples, ignore_index=True)
        print(OE[["date", "silent_from", "silent_to", "seconds"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    # WHERE THE MISSING MESSAGES ARE, which is the whole question
    if gap_examples:
        print("\n" + "=" * 78)
        print("SEQUENCE BREAKS -- the biggest, with the times either side")
        print("=" * 78)
        print("  A block missing BEFORE the first continuous print or AFTER")
        print("  the last one costs nothing: the capture started late or")
        print("  stopped early. The same block DURING the session means the")
        print("  book was missing adds and cancels while we were quoting")
        print("  against it. Read the timestamps, not the totals.")
        print("  PSX continuous trading is 09:32-15:30 PKT, which is")
        print("  04:32-10:30 UTC.\n")
        GE = pd.concat(gap_examples, ignore_index=True)
        print(GE[["date", "channel", "missing", "last_seen_at", "resumed_at"]]
              .to_string(index=False))

    # misordered levels, with the price step that flagged each one
    if level_examples:
        print("\n" + "=" * 78)
        print("MISORDERED LEVELS -- the biggest jumps, so the cause can be seen")
        print("=" * 78)
        print("  A bid ladder must fall as the level number rises and an offer")
        print("  ladder must climb. Where it does not, 'level 1' is not the")
        print("  touch and every depth read takes the wrong rows.")
        LE = pd.concat(level_examples, ignore_index=True)
        print(LE[["date", "symbol", "msg_seq", "side", "level", "px",
                  "step_from_prev_level"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # crossed-book examples, because the count alone says nothing
    if examples:
        print("\n" + "=" * 78)
        print("CROSSED BOOKS -- the worst, so the cause can be looked at")
        print("=" * 78)
        CE = pd.concat(examples, ignore_index=True)
        print(CE.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ---- write ----------------------------------------------------------
    # a fresh timestamped destination; never overwrites
    import datetime as _dt
    out = Path(RESULTS_ROOT) / (
        f"data_quality_{_dt.datetime.now():%Y%m%d_%H%M}.csv")
    # refuse rather than clobber
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite")
    F.to_csv(out, index=False)
    print(f"\nwrote {out}")
    # non-zero exit when any fault fired, so this can gate a pipeline
    raise SystemExit(1 if int(f["count"].sum()) else 0)


# entry point
if __name__ == "__main__":
    main()
