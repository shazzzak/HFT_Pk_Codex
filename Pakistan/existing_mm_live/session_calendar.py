# ============================================================================
# session_calendar.py -- the measured trading session for every date, by type
# ============================================================================
# WHY THIS EXISTS.
#
# PSX does not run one session. It runs FOUR, and they have different clocks:
#
#   1. REGULAR TRADING DAY   -- Monday to Thursday, ordinary calendar
#   2. REGULAR FRIDAY        -- split by the Jumu'ah break
#   3. RAMADAN REGULAR DAY   -- Monday to Thursday during Ramadan, shorter
#   4. RAMADAN FRIDAY        -- shorter AND split
#
# Any script that hard-codes one window is wrong on three of the four. That is
# not hypothetical: report_feed_gaps.py assumed 09:32-15:30 PKT on all 207
# dates, which is wrong at BOTH ends on an ordinary day and wrong by a further
# hour or more during Ramadan, and it consequently reported tens of thousands
# of seconds of "feed outage" that were the market being closed.
#
# HOW IT MEASURES, AND WHY THIS WAY.
#
# The exchange states its own state in TradingSessionStatus messages (35=h),
# in TradingPhaseCode (tag 8538), a fixed-width string whose characters are
# [phase][suspended][break_reason]:
#
#   phase        T=continuous  O=open call auction  B=break  A=after hours
#                E=closed  S=starting  N/V=normal call auction  C=close
#                H=temporary suspension
#   break_reason 1=after pre-open  2=FRIDAY LUNCH BREAK
#                3=after pre-open PM Friday  4=before post-close
#
# There are roughly 190,000 of these a date -- about 7.6 a second -- so they
# are NOT one global announcement. They describe many things at once, and
# those things are not in the same phase at the same moment.
#
# TWO EARLIER ATTEMPTS AT THIS FILE GOT IT WRONG, both the same way:
#   * pooling every message into one timeline and collapsing it into spans
#     produced 8,500 traded seconds inside a 25,200-second session, because
#     the phase flipped every few messages as different instruments disagreed
#     and the continuous spans were chopped into fragments;
#   * filtering on `segment` produced nothing at all, because 35=h messages
#     do not carry tag 1500, so that column is null on every one of them.
#
# THIS VERSION DOES NOT LOOK FOR A DISCRIMINATOR AT ALL. It asks a different
# and better-posed question, second by second: of everything the exchange said
# about its state during this second, what did MOST of it say? The market is
# open in a second when most of that second's messages say continuous trading.
# That is robust to a handful of individually halted instruments, it needs no
# assumption about what an h message describes, and it yields traded seconds
# directly as a count of seconds rather than as a sum of inferred spans.
#
# Use --probe to see what the messages actually contain on one date, tag by
# tag, if the grouping question ever needs settling on its own.
#
# READ-ONLY. Writes one timestamped CSV. Never overwrites.
#
# Run from Production/:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/session_calendar.py --probe
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/session_calendar.py --days 15
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/session_calendar.py
# ============================================================================

# command-line flags
import argparse
# timestamped output names
import datetime as dt
# path handling
from pathlib import Path
# the date comes off the filename
import re

# run-length arithmetic on the per-second flags
import numpy as np
# frames
import pandas as pd

# the store and the results directory, from the one config every runner uses
try:
    from config_pk import PARSED_ROOT, RESULTS_ROOT
except Exception as _e:                                       # noqa: BLE001
    raise ImportError(
        "session_calendar: could not import paths from config_pk (%r). Run "
        "with PYTHONPATH=../existing_mm_live." % _e)

# THE PHASE DICTIONARY, copied from PSX_Parser_Mac.py's PHASE_MAP so the two
# cannot drift apart silently. Only the first character of tag 8538.
PHASE_MAP = {
    "S": "STARTING",
    "O": "OPEN_CALL_AUCTION",
    "T": "CONTINUOUS_AUCTION",
    "B": "TRADING_BREAK",
    "N": "NORMAL_CALL_AUCTION_PM",
    "H": "TEMPORARY_SUSPENSION",
    "V": "NORMAL_CALL_AUCTION_RESUME",
    "C": "CLOSE_CALL_AUCTION",
    "A": "AFTER_HOUR_TRADING",
    "E": "MARKET_CLOSED",
}

# THE MARKET DICTIONARY, from the PSX FIX Market Data Interface
# Specification, section 4.2.1, tag 336 TradingSessionID, described there as
# MarketCode. PSX publishes a SEPARATE Trading Session Status stream per
# market, every 3 seconds, and they are not in the same phase at the same
# moment -- which is why pooling them produces nonsense.
#
# THIS IS A DOCUMENTED PROTOCOL CONSTANT, not something to infer. An earlier
# version of this file tried to identify the regular market by matching each
# stream's opening and closing bell against the first and last regular-market
# print, and chose '08' -- the ODD LOT market -- because the odd lot bell sits
# closer to those prints than the regular market's own does. It does so for a
# reason worth knowing: the first and last REG prints of the day are AUCTION
# CROSSES, which happen at the end of the pre-open and at the close, outside
# the continuous session. Matching a continuous window against auction prints
# selects the wrong market. The spec was the answer all along.
MARKET_CODE = {
    "01": "Regular Market",
    "02": "Bills and Bond Market",
    "03": "Stock Deliverable Future Market",
    "04": "Stock Cash Settled Future Market",
    "05": "Stock Option Market",
    "06": "Index Option Market",
    "07": "Stock Index Future Market",
    "08": "Odd Lot Market",
    "09": "Negotiated Deal Market",
    "10": "Equities Square Up Market",
    "12": "Futures Square Up Market",
    "13": "Trade Rectification and Modification Market",
}

# THE MARKET THIS DESK QUOTES. '01' per the dictionary above.
REGULAR_MARKET = "01"

# THE BREAK DICTIONARY, likewise. Only meaningful when the phase is B.
BREAK_REASON_MAP = {
    "1": "AFTER_PRE_OPEN",
    "2": "FRIDAY_LUNCH_BREAK",
    "3": "AFTER_PRE_OPEN_PM_FRIDAY",
    "4": "BEFORE_POST_CLOSE",
}

# THE FIELD SEPARATOR THIS CAPTURE USES. Standard FIX is SOH (0x01); this
# capture writes a caret. Taken from PSX_Parser_Mac.py.
SOH = "^"

# PKT is UTC+5 and does not observe daylight saving, so one constant is
# correct all year.
PKT_OFFSET = pd.Timedelta(hours=5)

# HOW LONG A SECOND'S STATE CARRIES FORWARD when no message arrives, in
# seconds. A phase message states a condition that holds until it changes, so
# a short silence inherits the previous state. Beyond this the state is
# treated as unknown rather than assumed, which matters because the capture
# has genuine multi-minute silences and assuming "still trading" through one
# would manufacture session time that was never observed.
CARRY_SECONDS = 120

# HOW MUCH SHORTER THAN ITS PEERS A DATE MUST BE to be labelled a shortened
# session, in minutes. 30 is comfortably larger than the few-minutes jitter
# between ordinary dates and far smaller than the Ramadan shortening, so the
# classification is not sensitive to where this sits. It affects the LABEL
# only -- every measured time in the output is unaffected.
SHORT_SESSION_MINUTES = 30.0


def partition(table, date):
    """Where one table's partition for one date lives, hive layout."""
    # <root>/<table>/date=YYYY-MM-DD/YYYY-MM-DD_<table>.parquet
    return (Path(PARSED_ROOT) / table / f"date={date}"
            / f"{date}_{table}.parquet")


def all_dates():
    """Every trading date in the store, oldest first."""
    # the trades partitions are the calendar
    files = sorted(Path(PARSED_ROOT).rglob("*_trades.parquet"))
    # the date prefix the parser writes on every filename
    rx = re.compile(r"^(\d{4}-\d{2}-\d{2})_trades\.parquet$")
    # only the ones that match the expected naming
    out = []
    # walk them
    for f in files:
        # the date, or skip a file that cannot be attributed
        m = rx.match(f.name)
        # named unexpectedly: say so rather than guess
        if m is None:
            print(f"  skipping unexpected filename {f.name}")
            continue
        # keep it
        out.append(m.group(1))
    # oldest first, no duplicates
    return sorted(set(out))


def field(body, tag):
    """One FIX tag's value out of a raw message body, or None."""
    # a missing body cannot be read
    if not isinstance(body, str):
        return None
    # the separator-prefixed key, so "8538=" cannot match inside "18538=..."
    key = SOH + tag + "="
    # where it sits
    i = body.find(key)
    # not found mid-message; it may still be the very first field
    if i < 0:
        # the first field carries no leading separator
        if body.startswith(tag + "="):
            # value starts just past "tag="
            off = len(tag) + 1
        else:
            # genuinely absent
            return None
    else:
        # value starts just past the matched key
        off = i + len(key)
    # the end of this field
    j = body.find(SOH, off)
    # last field on the line has no trailing separator
    return body[off:] if j < 0 else body[off:j]


def median_clock(series):
    """The median wall-clock time of a timestamp series, as HH:MM:SS.

    Times are not numbers, so pandas cannot take their median directly. This
    converts to seconds since midnight, takes the median there, and converts
    back -- which is the only meaningful median for a clock time.
    """
    # seconds since midnight for each timestamp
    secs = [x.hour * 3600 + x.minute * 60 + x.second for x in series]
    # the median of those, as a whole number of seconds
    med = int(float(pd.Series(secs).median()))
    # back to a clock
    return f"{med // 3600:02d}:{(med % 3600) // 60:02d}:{med % 60:02d}"


def read_status(date):
    """Every TradingSessionStatus (35=h) message for one date.

    Returns a frame of (capture_ts, phase, break_reason, raw), sorted by
    arrival time, or None when the date has no misc partition or no status
    messages in it.
    """
    # where the misc partition lives
    p = partition("misc", date)
    # a missing partition is reported by the caller, never assumed empty
    if not p.exists():
        return None
    # Three columns only, and only the status messages: misc holds ~400k rows
    # a date of which ~190k are 35=h, and `raw` is a long string column, so
    # pushing the filter into the read rather than doing it afterwards halves
    # the bytes materialised. A parquet engine that cannot push it down still
    # returns the right rows, just more slowly.
    try:
        m = pd.read_parquet(p, columns=["capture_ts", "msg_type", "raw"],
                            filters=[("msg_type", "==", "h")])
    except Exception:                                          # noqa: BLE001
        # no filter support: read it all and filter here
        m = pd.read_parquet(p, columns=["capture_ts", "msg_type", "raw"])
    # TradingSessionStatus messages only (a no-op when the filter was pushed)
    h = m[m["msg_type"].astype("string") == "h"]
    # a date with no session-status messages cannot be measured this way
    if len(h) == 0:
        return None
    # TAG 8538, VECTORISED. A per-row Python call is ~190k invocations a date
    # and 207 dates of that is the difference between a coffee and an
    # afternoon. Neither 8538 nor 336 is ever the first field -- every body
    # starts with 8=FIXT.1.1 -- so the leading separator is always present
    # and the caret in the pattern is safe.
    codes = h["raw"].str.extract(r"\^8538=([^\^]*)", expand=False)
    # the frame this function promises. The three text columns go through
    # .values only to drop the source index; NOTE that capture_ts is
    # deliberately NOT built that way -- see below.
    out = pd.DataFrame({
        # WHICH BOARD THIS MESSAGE DESCRIBES. Tag 336 is TradingSessionID and
        # the probe found 12 distinct values on every message -- PSX publishes
        # a separate status per board. Without this the measured session is
        # the UNION across all twelve, so the day appears to open when the
        # earliest board opens and close when the latest board closes, which
        # is nobody's actual session.
        "session_id": h["raw"].str.extract(r"\^336=([^\^]*)",
                                           expand=False).values,
        # first character -> readable phase
        "phase": codes.str.slice(0, 1).map(PHASE_MAP).values,
        # third character -> break reason, meaningful only during a break
        "break_reason": codes.str.slice(2, 3).map(BREAK_REASON_MAP).values,
        # kept for --probe
        "raw": h["raw"].values,
    })
    # ARRIVAL TIME, ADDED SEPARATELY AND ON PURPOSE.
    #
    # `.values` on a timezone-aware datetime Series returns a plain numpy
    # datetime64 array and SILENTLY DROPS THE TIMEZONE. Putting that in the
    # frame gives a tz-naive column that looks right, prints right, and
    # measures right -- until it meets a tz-aware timestamp from anywhere
    # else, at which point pandas refuses to compare or subtract them. That
    # is how pick_board crashed twice: once on each side of the comparison.
    # Building the column from the numpy array with utc=True re-attaches the
    # timezone, so everything this function returns is tz-aware.
    out.insert(0, "capture_ts",
               pd.to_datetime(h["capture_ts"].to_numpy(), utc=True))
    # drop rows whose code could not be read at all
    out = out.dropna(subset=["phase"])
    # nothing usable
    if len(out) == 0:
        return None
    # in time order
    return out.sort_values("capture_ts").reset_index(drop=True)


def per_second_state(st):
    """Collapse the status stream into one phase per second of the day.

    THE CORE OF THIS FILE. For each second in which at least one status
    message arrived, the phase is whichever phase MOST of that second's
    messages reported. Seconds with no message inherit the previous second's
    phase for up to CARRY_SECONDS, after which the state is unknown (NaN)
    rather than assumed.

    Returns (t0, phase_by_second, reason_by_second) where t0 is the origin
    timestamp and the two Series are indexed by WHOLE SECONDS SINCE t0.

    EVERYTHING HERE IS INTEGER-INDEXED ON PURPOSE.

    The previous version built the grid with pd.date_range and reindexed the
    per-second phases onto it. A pandas timestamp carries a resolution --
    nanosecond, microsecond, millisecond -- that depends on how the parquet
    was written, and .dt.floor and pd.date_range do not always produce the
    same one. Reindex matches by exact label, so a microsecond-resolution
    index reindexed against a nanosecond one matches NOTHING and returns an
    all-empty Series. That presents as "the market never opened", on every
    date at once, with no error -- which is exactly what happened.

    Integer second offsets cannot drift like that, so the failure mode is
    removed rather than patched.
    """
    # THE ORIGIN: the first message of the day, floored to a whole second.
    # st is sorted by arrival, so this is the earliest.
    t0 = st["capture_ts"].iloc[0].floor("s")
    # WHOLE SECONDS SINCE THE ORIGIN, as plain integers
    off = (st["capture_ts"] - t0).dt.total_seconds().astype("int64")
    # how many messages in each (second, phase) pair. crosstab is vectorised;
    # a per-group lambda over ~25,000 seconds a date times 207 dates is slow.
    cnt = pd.crosstab(off, st["phase"])
    # the phase most of that second's messages reported. Cast to a string
    # dtype before any reindex or fill: an object-dtype Series triggers
    # pandas' silent-downcasting FutureWarning on ffill, which is noise.
    mode = cnt.idxmax(axis=1).astype("string")
    # the same for break reasons, which only exist during a break
    br = st.dropna(subset=["break_reason"])
    # a date with no breaks at all still needs a Series to align against
    if len(br):
        # that subset's own second offsets, on the same origin
        boff = (br["capture_ts"] - t0).dt.total_seconds().astype("int64")
        # count by (second, reason)
        bcnt = pd.crosstab(boff, br["break_reason"])
        # the reason most of that second's break messages gave
        bmode = bcnt.idxmax(axis=1).astype("string")
    else:
        # an empty Series of the right dtype and index type
        bmode = pd.Series(dtype="string", index=pd.Index([], dtype="int64"))
    # a contiguous run of integer seconds across the whole observed range
    idx = pd.RangeIndex(int(off.min()), int(off.max()) + 1)
    # reindex onto it, leaving unobserved seconds empty
    full = mode.reindex(idx)
    # carry the last known state forward, but only so far
    full = full.ffill(limit=CARRY_SECONDS)
    # break reasons on the same grid, carried the same way
    bfull = bmode.reindex(idx).ffill(limit=CARRY_SECONDS)
    # the origin and both series
    return t0, full, bfull


def seconds_of_day(ts):
    """A timestamp's clock time as seconds since midnight, as an integer."""
    # hours, minutes and seconds only -- the date is irrelevant to a session
    return int(ts.hour) * 3600 + int(ts.minute) * 60 + int(ts.second)


def board_scores(date, rows):
    """How far each board's bell is from this date's first and last REG print.

    Returns {board_id: error_seconds}, LOWER IS BETTER: the number is
    |open - first regular-market print| + |close - last regular-market print|.

    WHY THIS METRIC, measured on 2026-03-11, 03-13, 03-25 and 03-27 (one of
    each of the four day types):

        board 08          1s      2s      2s     10s
        board 01        121s    124s    125s      7s
        boards 03-07    121s    135s    129s     10s
        board 02      1,802s  3,602s      2s  3,603s
        board 09      1,801s  3,602s  1,802s  5,352s
        board 13     19,502s 15,903s 26,403s 30,019s

    Board 08 is the regular market: its bell sits within ten seconds of the
    real prints on every day type. Boards 01 and 03-07 are consistently about
    two minutes late at the open -- on 2026-03-25 board 05 reports the market
    as not yet trading at 09:32:13 while the first regular print is timestamped
    09:30:00.101. Boards 02 and 09 close half an hour to an hour early; board
    13 is the after-hours board.

    AN EARLIER VERSION SCORED JACCARD OVERLAP AGAINST THE 5TH-TO-95TH
    PERCENTILE of print times and picked board 07, 06 or 05 -- all three tied
    at 0.9130, separated in the fourth decimal. Trimming to percentiles threw
    away the opening and closing prints, which are precisely the signal that
    distinguishes one board from another; the extremes ARE the bell. On the
    same data this metric separates the right board from the rest by a factor
    of twelve.

    AN ERROR, NOT A VERDICT, because the verdict must not be taken one date at
    a time. The caller collects these across every date and chooses once, on
    the median, so one date with a stray print cannot move the answer.
    """
    # nothing to choose between
    if not rows:
        return {}
    # the trades partition, which is where the REG market leaves its mark
    p = partition("trades", date)
    # without it there is no evidence to choose on
    if not p.exists():
        return {}
    # two columns only
    t = pd.read_parquet(p, columns=["capture_ts", "market"])
    # the regular market
    reg = t[t["market"].astype("string") == "REG"]
    # fall back to every trade when the market column is not populated
    src = reg if len(reg) else t
    # no trades at all
    if len(src) == 0:
        return {}
    # arrival times of those trades
    tt = pd.to_datetime(src["capture_ts"], utc=True)
    # EPOCH SECONDS, AS PLAIN NUMBERS. Everything below is arithmetic on
    # floats rather than on timestamps, for the same reason per_second_state
    # works in integer seconds: pandas' timestamp comparisons are sensitive
    # to timezone-awareness and to resolution, and quantile() on a tz-aware
    # series returns a tz-NAIVE timestamp in some versions, which then cannot
    # be compared with the tz-aware ones here. Numbers have neither problem.
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    # every trade's arrival, in seconds since the epoch
    ts = (tt - epoch).dt.total_seconds()
    # THE FIRST AND LAST PRINT, untrimmed. See the docstring: trimming to
    # percentiles removes the opening and closing prints, which are the only
    # thing that tells the boards apart.
    lo, hi = float(ts.min()), float(ts.max())
    # board -> error in seconds
    out = {}
    # measure each board's bell against those two instants
    for r in rows:
        # this board's window, in the same units
        o = (r["open_utc"] - epoch).total_seconds()
        c = (r["close_utc"] - epoch).total_seconds()
        # how far its bell sits from the real one, at both ends
        out[r["session_id"]] = abs(o - lo) + abs(c - hi)
    # every board's error for this date
    return out


def pick_board(date, rows):
    """The single best-matching board on ONE date.

    Kept because it is the natural unit to test. main() does NOT use it --
    see board_scores for why a per-date verdict is the wrong shape.
    """
    # the errors
    s = board_scores(date, rows)
    # nothing to choose between
    if not s:
        return None
    # The SMALLEST error wins, and an exact tie is broken by the lowest board
    # id. Without that second key the answer depends on dictionary iteration
    # order, which is a different answer on a different day for no reason.
    return min(s, key=lambda k: (s[k], str(k)))


def measure(date, st, session_id=None):
    """One board's measured session on one date, as a dict for the frame."""
    # the second-by-second state, indexed by whole seconds since t0
    t0, ph, br = per_second_state(st)
    # which seconds the market was continuously trading in
    cont = (ph == "CONTINUOUS_AUCTION")
    # a date that never opened has no session to measure
    if not cont.any():
        # A CONTRADICTION IS NOT THE SAME AS AN ANSWER. If the raw messages
        # contain continuous trading but no second resolves to it, the
        # collapse is broken, not the date -- say so loudly rather than
        # returning a quiet "never opened" that reads like a fact about PSX.
        if st["phase"].eq("CONTINUOUS_AUCTION").any():
            print(f"      CONTRADICTION on {date}: "
                  f"{int(st['phase'].eq('CONTINUOUS_AUCTION').sum()):,} "
                  f"messages report CONTINUOUS_AUCTION, but no second "
                  f"resolves to it. This is a bug in per_second_state, not "
                  f"a closed market.")
        return None
    # the first and last such second, as offsets
    open_off = int(cont.index[cont][0])
    close_off = int(cont.index[cont][-1])
    # and as real timestamps, rebuilt from the origin
    open_utc = t0 + pd.Timedelta(seconds=open_off)
    close_utc = t0 + pd.Timedelta(seconds=close_off)
    # TRADED SECONDS: a straight count of continuous seconds. On a Friday this
    # is smaller than close minus open by the length of the Jumu'ah break, and
    # the traded figure is the one any rate metric wants.
    traded = int(cont.sum())
    # the part of the day between the open and the close, by offset
    inside = ph.loc[open_off:close_off]
    # seconds inside it that the exchange called a break
    brk = (inside == "TRADING_BREAK")
    # how long the market stood in a break inside the session
    brk_secs = int(brk.sum())
    # which breaks those were, named by the exchange
    names = br.loc[open_off:close_off][brk.values].dropna().unique()
    # as a stable string
    brk_names = ",".join(sorted(str(x) for x in names))
    # seconds inside the session whose state we never observed
    unknown = int(inside.isna().sum())
    # ---- THE REMAINDER, WHICH MUST NOT BE ALLOWED TO VANISH ------------
    # Seconds inside the session that were NOT continuous trading, NOT a
    # declared break, and NOT unobserved. Until this column existed those
    # seconds simply disappeared between traded_seconds and break_seconds,
    # and five dates in the 207 turned out to be missing about 65 minutes
    # each with nothing in the output to show it: 2026-03-02, 03-09, 03-10,
    # 04-01 and 04-08, all within nine seconds of 3,885. A market halt would
    # look exactly like that. The phases are named rather than guessed at.
    _other = inside[inside.notna()
                    & ~inside.isin(["CONTINUOUS_AUCTION", "TRADING_BREAK"])]
    # how much time that accounts for
    other_secs = int(len(_other))
    # and which phases they were, so the answer is in the output, not a theory
    other_phases = ",".join(sorted(str(x) for x in _other.unique()))
    # HOW MANY SEPARATE STRETCHES of continuous trading there were -- 2 on a
    # split Friday, 1 on an ordinary day. Counted as the number of times the
    # series turns continuous having not been continuous the second before.
    ic = cont.loc[open_off:close_off]
    # a run starts wherever a True follows a non-True
    spans = int((ic & ~ic.shift(1, fill_value=False)).sum())
    # THE TRADING INTERVALS THEMSELVES, not just their outer bounds.
    #
    # A consumer asking "was this moment inside the session?" CANNOT answer it
    # from open and close on a Friday: the two-and-a-quarter-hour Jumu'ah
    # break sits between them, and counting a feed silence during a closed
    # lunch break as an outage is the same class of error that put 80,000
    # phantom seconds in the first vendor report.
    #
    # Found vectorised rather than by walking the seconds: ten markets times
    # 207 dates times ~25,000 seconds is fifty million iterations in Python
    # and about two hundred in numpy.
    # na_value=False because a second whose state was never observed is not a
    # trading second. Without it this raises on any date with an unobserved
    # gap -- which is most of the interesting ones.
    a = ic.to_numpy(dtype=bool, na_value=False)
    # a run starts at a True whose predecessor is not True
    _starts = np.flatnonzero(a & ~np.r_[False, a[:-1]])
    # and ends at a True whose successor is not True
    _ends = np.flatnonzero(a & ~np.r_[a[1:], False])
    # the offsets, as real timestamps
    intervals = [(t0 + pd.Timedelta(seconds=int(ic.index[s])),
                  t0 + pd.Timedelta(seconds=int(ic.index[e])))
                 for s, e in zip(_starts, _ends)]
    # the weekday, which is half the day-type answer on its own
    wd = pd.Timestamp(date).day_name()
    # THE FRIDAY TEST, taken from the exchange rather than from the weekday:
    # break reasons 2 and 3 exist only on a Friday.
    friday_break = any(n in ("FRIDAY_LUNCH_BREAK", "AFTER_PRE_OPEN_PM_FRIDAY")
                       for n in names)
    # the row
    return {
        # the date
        "date": date,
        # its weekday
        "weekday": wd,
        # WHICH BOARD these numbers describe
        "session_id": session_id,
        # continuous open, both clocks
        "open_utc": open_utc,
        "open_pkt": open_utc + PKT_OFFSET,
        # continuous close, both clocks
        "close_utc": close_utc,
        "close_pkt": close_utc + PKT_OFFSET,
        # THE SPAN FROM OPEN TO CLOSE, breaks included, COUNTED THE SAME WAY
        # AS EVERYTHING ELSE IN THIS ROW.
        #
        # This was (close - open).total_seconds(), an elapsed-time
        # subtraction, while traded/break/unobserved/other are COUNTS of
        # seconds on the per-second grid. Mixing the two conventions made the
        # parts exceed the whole by exactly one second on every date -- a
        # clean day printed "traded 15,180s of 15,179s", which is nonsense on
        # its face -- and the balance check below only passed because I had
        # given it a tolerance. A check that needs slack to pass is not
        # checking. Counting inclusively, as the grid does, makes the parts
        # sum to the whole exactly and the check exact with it.
        "open_to_close_seconds": float(close_off - open_off + 1),
        # THE NUMBER THAT MATTERS: seconds of actual continuous trading
        "traded_seconds": float(traded),
        # how many continuous stretches made it up (2 on a split Friday)
        "continuous_spans": spans,
        # THE STRETCHES THEMSELVES, which is what a consumer needs to decide
        # whether a given moment was inside the session
        "intervals": intervals,
        # the break time inside the session
        "break_seconds": float(brk_secs),
        # and what the exchange called those breaks
        "break_reasons": brk_names,
        # seconds inside the session with no observed state at all. A large
        # number here means the capture went quiet and the session boundaries
        # are less certain than they look.
        "unobserved_seconds": float(unknown),
        # seconds inside the session in some OTHER phase -- not trading, not
        # a declared break, not missing. These four columns plus
        # traded_seconds account for open_to_close_seconds exactly.
        "other_seconds": float(other_secs),
        # and what those phases were
        "other_phases": other_phases,
        # whether the exchange itself declared a Friday break
        "exchange_says_friday": bool(friday_break),
        # how many status messages the measurement rests on
        "status_messages": int(len(st)),
    }


def probe(date):
    """Print what a date's status messages actually contain, tag by tag.

    This exists because two earlier versions of this file guessed at what an
    h message describes and both guesses were wrong. It asserts nothing; it
    prints the messages and the cardinality of every tag in them.
    """
    # the messages
    st = read_status(date)
    # nothing to probe
    if st is None:
        raise SystemExit(f"no 35=h status messages on {date}")
    print(f"\n  {len(st):,} TradingSessionStatus (35=h) messages on {date}")
    # a few whole bodies, which is the thing no summary substitutes for
    print("\n  THREE RAW BODIES")
    # spread across the day rather than three from the same instant
    for i in (0, len(st) // 2, len(st) - 1):
        # truncated, because a snapshot-sized body would swamp the output
        print(f"    {str(st['raw'].iloc[i])[:400]}")

    # ---- THE PHASE CENSUS ------------------------------------------------
    # WHY THIS IS HERE. A majority-vote measure of the session found no
    # second in which CONTINUOUS_AUCTION was the most common phase, while a
    # span-based measure found continuous trading easily. Both cannot be
    # right about what this stream is. This says what the phases actually
    # are, in what proportion, so the question stops being answered by
    # assumption.
    print("\n  EVERY PHASE ON THIS DATE")
    # counts, biggest first
    vc = st["phase"].value_counts()
    # each one with its share
    for ph, n in vc.items():
        print(f"    {ph:<28s} {n:>10,}  ({100 * n / len(st):5.2f}%)")

    # ---- WHAT CO-EXISTS INSIDE ONE SECOND --------------------------------
    print("\n  WHAT THE EXCHANGE SAID DURING FIVE SINGLE SECONDS")
    print("  If several phases appear in the same second, these messages")
    print("  describe different things and cannot be pooled. If only one")
    print("  appears, they are one global announcement and pooling is fine.")
    # the second each message landed in
    sec = st["capture_ts"].dt.floor("s")
    # every second that carries at least one message
    uniq = sec.drop_duplicates().sort_values().reset_index(drop=True)
    # five spread evenly through the day
    picks = [uniq.iloc[int(len(uniq) * f)]
             for f in (0.10, 0.30, 0.50, 0.70, 0.90)]
    # each one, with the full breakdown of that second
    for p in picks:
        # the messages that landed in that second
        g = st[sec == p]
        # what they said
        parts = ", ".join(f"{k}={v}" for k, v
                          in g["phase"].value_counts().items())
        # the second, in both clocks, and the breakdown
        print(f"    {str(p)[11:19]} UTC ({str(p + PKT_OFFSET)[11:19]} PKT)  "
              f"{len(g):>4d} msgs  {parts}")

    # ---- WHEN WAS CONTINUOUS TRADING EVER SEEN AT ALL --------------------
    print("\n  WHEN CONTINUOUS_AUCTION APPEARS, however few messages say it")
    # every message that reported continuous trading
    c = st[st["phase"] == "CONTINUOUS_AUCTION"]
    # a date with none is itself the answer
    if len(c) == 0:
        print("    NEVER on this date -- no message reported continuous "
              "trading at all.")
    else:
        # the window it spans
        print(f"    {len(c):,} messages, first "
              f"{str(c['capture_ts'].iloc[0])[11:19]} UTC "
              f"({str(c['capture_ts'].iloc[0] + PKT_OFFSET)[11:19]} PKT), "
              f"last {str(c['capture_ts'].iloc[-1])[11:19]} UTC "
              f"({str(c['capture_ts'].iloc[-1] + PKT_OFFSET)[11:19]} PKT)")
        # how many distinct seconds carried at least one such message
        csec = c["capture_ts"].dt.floor("s").nunique()
        # against the span those seconds cover
        span = (c["capture_ts"].iloc[-1] - c["capture_ts"].iloc[0]).total_seconds()
        # the density, which says whether this is a steady heartbeat of state
        # or an occasional announcement
        print(f"    covering {csec:,} distinct seconds inside a "
              f"{span:,.0f}-second span "
              f"({100 * csec / max(span, 1):.1f}% of them)")
    # tokenise a sample: full tokenisation of every message is wasteful and a
    # few thousand settles a cardinality question
    n = min(20_000, len(st))
    # evenly spaced through the day
    sample = st["raw"].iloc[:: max(1, len(st) // n)]
    # tag -> set of values seen, capped so a per-message id cannot blow memory
    seen = {}
    # tag -> how many messages carried it
    present = {}
    # walk the sample
    for body in sample:
        # not a string
        if not isinstance(body, str):
            continue
        # every field on this message
        for part in body.rstrip(SOH).split(SOH):
            # a malformed field
            if "=" not in part:
                continue
            # split once
            tag, val = part.split("=", 1)
            # count presence
            present[tag] = present.get(tag, 0) + 1
            # collect values, up to a cap
            s = seen.setdefault(tag, set())
            # stop growing a set that is clearly a unique-per-message field
            if len(s) <= 50:
                s.add(val)
    # the table
    print(f"\n  EVERY TAG IN A SAMPLE OF {len(sample):,} MESSAGES")
    print(f"    {'tag':>8s} {'present':>9s} {'distinct':>9s}  example values")
    # most-present first, then by tag
    for tag in sorted(present, key=lambda t: (-present[t], t)):
        # the values seen
        vals = sorted(seen[tag])
        # "50+" when the cap was hit, because the true count is unknown
        dist = f"{len(vals)}" if len(vals) <= 50 else "50+"
        # a few examples, truncated
        ex = ", ".join(str(v)[:18] for v in vals[:4])
        # the row
        print(f"    {tag:>8s} {present[tag]:>9,} {dist:>9s}  {ex}")
    print("\n  A tag that is present on every message and has a small number")
    print("  of distinct values is a candidate grouping key. A tag with one")
    print("  distinct value groups nothing. Nothing was written.")


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # a short run, for a smoke test
    ap.add_argument("--days", type=int, default=0,
                    help="only the most recent N dates (0 = every date)")
    # where the CSV goes
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to the results root")
    # print what the status messages contain and stop
    ap.add_argument("--probe", action="store_true",
                    help="print the raw status messages and every tag in "
                         "them for one date, then stop without writing")
    # which date to probe
    ap.add_argument("--date", default=None,
                    help="the date --probe examines; default is the newest")
    # which market to report as the session, if not the regular one
    ap.add_argument("--market", default=REGULAR_MARKET,
                    help=f"the TradingSessionID (tag 336) whose session is "
                         f"reported; default {REGULAR_MARKET} = Regular "
                         f"Market")
    args = ap.parse_args()

    print("=" * 78)
    print("PSX SESSION CALENDAR -- measured, not assumed")
    print("=" * 78)
    print(f"  store : {PARSED_ROOT}")

    # the dates
    dates = all_dates()
    # a store with no partitions is a stop, not an empty report
    if not dates:
        raise SystemExit("no trades partitions found under the store")
    print(f"  dates : {len(dates)}  ({dates[0]} .. {dates[-1]})")

    # ---- the probe, which writes nothing ---------------------------------
    if args.probe:
        # the date to look at
        probe(args.date or dates[-1])
        return

    # narrow if asked
    if args.days:
        dates = dates[-args.days:]
        print(f"  running: {len(dates)}  ({dates[0]} .. {dates[-1]})")
    print()

    # one row per board per date
    rows = []
    # dates that could not be measured, and why
    skipped = []
    # when the run started, for the estimate below
    t_start = dt.datetime.now()
    # board -> (summed score, number of dates it was scored on)
    board_total = {}
    # walk them
    for k, date in enumerate(dates, 1):
        # the exchange's own status messages
        st = read_status(date)
        # no status messages means no measurement from this source
        if st is None:
            # record it rather than let it vanish
            skipped.append((date, "no 35=h session-status messages in misc"))
            print(f"  [{k}/{len(dates)}] {date}  SKIPPED (no status messages)")
            continue
        # EVERY BOARD SEPARATELY. Measuring the pooled stream gives the union
        # across all twelve -- the earliest board's open and the latest
        # board's close -- which is nobody's session.
        boards = sorted(x for x in st["session_id"].dropna().unique())
        # no board id at all: measure the pooled stream and say so
        if not boards:
            boards = [None]
        # one measurement per board
        day_rows = []
        # walk them
        for b in boards:
            # that board's own messages
            sub = (st if b is None
                   else st[st["session_id"] == b]).reset_index(drop=True)
            # nothing from this board
            if len(sub) == 0:
                continue
            # its own session
            rb = measure(date, sub, b)
            # a board that never entered continuous trading is not an error
            if rb is not None:
                day_rows.append(rb)
        # no board opened
        if not day_rows:
            skipped.append((date, "no board reported CONTINUOUS_AUCTION"))
            print(f"  [{k}/{len(dates)}] {date}  SKIPPED (never opened)")
            continue
        # SCORE the boards against this date's trades, but do NOT pick a
        # winner yet -- the choice is made once, after every date, below.
        sc = board_scores(date, day_rows)
        # collect each board's error across the whole run. A LIST, not a
        # running total, because the choice is made on the MEDIAN: one date
        # with a stray print well outside the session would drag a mean.
        for b, s in sc.items():
            board_total.setdefault(b, []).append(s)
        # keep every market's row
        rows.extend(day_rows)
        # THE PROGRESS LINE SHOWS THE REGULAR MARKET, named by the spec. It
        # is not chosen from the data and so cannot wander from date to date.
        r = next((x for x in day_rows if x["session_id"] == args.market),
                 None)
        # a date on which the regular market never opened still gets a line,
        # from whatever did, rather than vanishing
        if r is None:
            # say so plainly on that date's line
            print(f"  [{k}/{len(dates)}] {date}  market {args.market} did "
                  f"not open; {len(day_rows)} other market(s) did")
            continue
        # progress, with the numbers worth watching
        print(f"  [{k}/{len(dates)}] {date} {r['weekday'][:3]}  "
              f"mkt {str(r['session_id']):>3s} of {len(day_rows):>2d}  "
              f"{str(r['open_pkt'])[11:19]}-{str(r['close_pkt'])[11:19]} PKT  "
              f"traded {r['traded_seconds']:>7,.0f}s  "
              f"break {r['break_seconds']:>6,.0f}s  "
              f"unobs {r['unobserved_seconds']:>5,.0f}s  "
              f"{r['break_reasons']}")
        # A RUNNING ESTIMATE, once there is enough history to make one. A
        # 207-date run reads a few hundred million rows, and a progress line
        # that does not say how long it will take is a reason to kill a job
        # that was nearly done.
        if k == 5 or (k % 25 == 0 and k < len(dates)):
            # seconds spent so far
            spent = (dt.datetime.now() - t_start).total_seconds()
            # seconds per date, and what remains at that rate
            left = spent / k * (len(dates) - k)
            # stated in minutes, which is the unit anyone waiting thinks in
            print(f"        ... {k}/{len(dates)} done in {spent / 60:.1f} "
                  f"min, about {left / 60:.1f} min left")

    # nothing measurable
    if not rows:
        raise SystemExit(
            "no date could be measured. Run with --probe to see what the "
            "status messages contain.")

    # EVERY board's row, which is what the CSV carries
    ALL = pd.DataFrame(rows)

    # ---- THE MARKET, FROM THE SPEC, NOT FROM A GUESS ---------------------
    # Tag 336 is MarketCode and the specification lists its values. There is
    # nothing here to infer.
    print("\n" + "=" * 78)
    print("MARKETS PRESENT (tag 336, TradingSessionID)")
    print("=" * 78)
    # every market seen, with its name and how its bell compares
    print(f"\n  {'code':>5s}  {'market':<38s} {'dates':>5s} "
          f"{'median bell gap (s)':>20s}")
    # each, in code order
    for b in sorted(board_total):
        # the spec's name for it, or a flag that the spec does not list it
        name = MARKET_CODE.get(str(b), "NOT IN THE SPEC'S MARKET LIST")
        # the median gap, for the cross-check below
        mu = float(pd.Series(board_total[b]).median())
        # the row
        print(f"  {str(b):>5s}  {name:<38s} {len(board_total[b]):>5d} "
              f"{mu:>20,.0f}")
    # what the gap column is, and why it is NOT the selector
    print("\n  Bell gap = |open - first REG print| + |close - last REG print|.")
    print("  It is a CROSS-CHECK, not the selector. The first and last REG")
    print("  prints of a day are AUCTION CROSSES, which sit outside the")
    print("  continuous session, so the market whose continuous bell is")
    print("  closest to them is not necessarily the regular market -- an")
    print("  earlier version of this file selected on it and chose the Odd")
    print("  Lot Market. The selector is the spec.")
    # the market to report
    reg = args.market
    # say which, by name
    print(f"\n  REPORTING: market {reg} = "
          f"{MARKET_CODE.get(str(reg), 'unknown code')}"
          + ("  (the spec's Regular Market)" if reg == REGULAR_MARKET
             else "  (given on the command line)"))
    # a code the spec does not list is worth flagging rather than accepting
    if str(reg) not in MARKET_CODE:
        print(f"  WARNING: {reg} is not one of the spec's market codes.")
    # THE CROSS-CHECK, stated rather than acted on. If the regular market's
    # bell is a long way from its own first and last print on most dates,
    # something is wrong with either the measurement or the assumption -- but
    # a gap of roughly the pre-open auction's length is EXPECTED, because the
    # first print is the opening cross.
    if reg in board_total:
        # its median gap
        gap = float(pd.Series(board_total[reg]).median())
        # stated with its interpretation
        print(f"\n  cross-check: market {reg}'s bell sits a median "
              f"{gap:,.0f}s from its own first and last print.")
        print("  A gap of a couple of minutes is expected -- the first REG")
        print("  print is the opening auction cross, which happens before")
        print("  continuous trading starts. A gap of hours would not be.")
    # flag the reported market on every row
    ALL["is_reg_board"] = (ALL["session_id"] == reg)
    # THE REPORTED MARKET ONLY, which is what the day types describe
    D = ALL[ALL["is_reg_board"]].copy().reset_index(drop=True)
    # the reported market is missing from every date
    if len(D) == 0:
        raise SystemExit(
            f"market {reg} has no measured session on any date. Every "
            f"market's measurements are in the CSV; report another one with "
            f"--market.")
    # dates where the reported market is absent, which the CSV would otherwise
    # hide by simply not having a row for them
    missing = sorted(set(ALL["date"]) - set(D["date"]))
    # named, because those dates have no session in the output
    if missing:
        print(f"\n  market {reg} is ABSENT on {len(missing)} date(s), which "
              f"therefore have no session in this calendar:")
        # up to fifteen, so a long list does not swamp the output
        for dd in missing[:15]:
            print(f"      {dd}")
        # and how many more
        if len(missing) > 15:
            print(f"      ... and {len(missing) - 15} more")

    # ---- CLASSIFY, using the measurements and nothing else ---------------
    # A date is a Friday if the exchange declared a Friday break on it. The
    # weekday is carried too, and a disagreement between the two is printed
    # below rather than silently resolved.
    D["is_friday"] = D["exchange_says_friday"] | (D["weekday"] == "Friday")
    # THE SHORTENED-SESSION TEST, ON THE CLOSING BELL rather than on traded
    # seconds.
    #
    # WHY NOT TRADED SECONDS. They are reduced by anything that stops the
    # feed, so a date with a bad capture looks like a short session. The
    # first run of this file labelled 2026-06-23 RAMADAN_REGULAR on exactly
    # that basis -- it is a June Tuesday, nowhere near Ramadan, and it closed
    # at 15:30:04 like every other regular day. What was actually different
    # about it was 429 seconds of lost feed. The closing bell does not move
    # when the capture drops a few minutes, so it is the right discriminator.
    # traded_seconds is still reported; it is simply not what classifies.
    D["close_sec"] = [seconds_of_day(x) for x in D["close_pkt"]]
    # the median closing bell within this date's own Friday class, so a
    # Friday is never called short merely for being a Friday
    D["class_median_close"] = D.groupby("is_friday")["close_sec"] \
                               .transform("median")
    # short by more than the threshold
    D["is_short"] = (
        (D["class_median_close"] - D["close_sec"])
        > SHORT_SESSION_MINUTES * 60.0)
    # THE FOUR TYPES, NAMED FOR WHAT WAS MEASURED, NOT FOR A CAUSE.
    #
    # These were called RAMADAN_REGULAR and RAMADAN_FRIDAY. What the code
    # actually measures is a closing bell well before its peers' -- and on the
    # full 207-date run one of the 22 dates it caught was 2025-09-23, a
    # September Tuesday that closed at 14:19:18 for some reason of its own.
    # Labelling that RAMADAN was the same mistake as choosing the Odd Lot
    # Market: naming a measurement after a cause that was inferred rather than
    # observed. SHORT_DAY and SHORT_FRIDAY say only what is known.
    #
    # That 21 of the 22 form one contiguous block from 2026-02-19 to
    # 2026-03-19, which is Ramadan 1447, is a fact about the output worth
    # writing down. It is not a fact the classifier is entitled to assert.
    D["day_type"] = [
        ("SHORT_FRIDAY" if f else "SHORT_DAY") if s
        else ("REGULAR_FRIDAY" if f else "REGULAR_DAY")
        for f, s in zip(D["is_friday"], D["is_short"])]

    # ---- REPORT -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("THE FOUR DAY TYPES, AS MEASURED")
    print("=" * 78)
    # the header
    print(f"\n  {'day type':<18s} {'dates':>6s} {'median open':>12s} "
          f"{'median close':>13s} {'median traded':>14s} {'break':>8s}")
    # one line per type, in a fixed order so the table reads the same each run
    for t in ("REGULAR_DAY", "REGULAR_FRIDAY", "SHORT_DAY", "SHORT_FRIDAY"):
        # that type's dates
        g = D[D["day_type"] == t]
        # a type with no dates still gets a line, so its absence is visible
        if len(g) == 0:
            print(f"  {t:<18s} {0:>6d}           --            -- "
                  f"            --       --")
            continue
        # the median clock times, in PKT. A time is not a number, so this
        # goes through seconds-since-midnight.
        o = median_clock(g["open_pkt"])
        c = median_clock(g["close_pkt"])
        # the row
        print(f"  {t:<18s} {len(g):>6d} {o:>12s} {c:>13s} "
              f"{g['traded_seconds'].median():>13,.0f}s "
              f"{g['break_seconds'].median():>7,.0f}s")

    # the shortened block, listed in full so it can be checked against the
    # actual Ramadan dates by eye
    short = D[D["is_short"]].sort_values("date")
    # only if there is one
    if len(short):
        print(f"\n  SHORTENED SESSIONS ({len(short)} dates) -- the bell rang "
              f"well before its peers'. NO CAUSE IS ASSERTED; a contiguous "
              f"block is worth\n  checking against the Ramadan calendar, and "
              f"an isolated date against that day's news.")
        print(f"    {short['date'].iloc[0]} .. {short['date'].iloc[-1]}")
        # each one, so a stray date in the middle is not hidden by a range
        for _, r in short.iterrows():
            print(f"      {r['date']} {r['weekday'][:3]}  close "
                  f"{str(r['close_pkt'])[11:19]} PKT  traded "
                  f"{r['traded_seconds']:,.0f}s  ({r['day_type']})")

    # any date where the weekday and the exchange's own break codes disagree
    odd = D[(D["weekday"] == "Friday") != D["exchange_says_friday"]]
    # reported rather than reconciled
    if len(odd):
        print(f"\n  {len(odd)} date(s) where the weekday and the exchange's "
              f"own Friday break codes DISAGREE:")
        # each one
        for _, r in odd.iterrows():
            print(f"      {r['date']} weekday={r['weekday']} "
                  f"exchange_says_friday={r['exchange_says_friday']} "
                  f"breaks={r['break_reasons']}")
        print("      These are worth a look before the calendar is trusted.")

    # dates whose measurement rests on a lot of unobserved time
    shaky = D[D["unobserved_seconds"] > 600]
    # named, because their boundaries are softer than the others'
    if len(shaky):
        print(f"\n  {len(shaky)} date(s) with more than 10 minutes of "
              f"UNOBSERVED time inside the session:")
        # each one
        for _, r in shaky.sort_values("unobserved_seconds",
                                      ascending=False).head(15).iterrows():
            print(f"      {r['date']}  {r['unobserved_seconds']:,.0f}s "
                  f"unobserved of {r['open_to_close_seconds']:,.0f}s")
        print("      Their open and close are measured, but the capture was")
        print("      quiet for part of the day, so treat them as softer.")

    # ---- TIME INSIDE THE SESSION THAT WAS NEITHER TRADING NOR A BREAK ----
    # The column that stops these seconds vanishing. On the 207-date run five
    # dates were each missing about 65 minutes with nothing in the output to
    # show for it.
    odd_phase = D[D["other_seconds"] > 600]
    # named, with the phases the exchange actually reported
    if len(odd_phase):
        print(f"\n  {len(odd_phase)} date(s) with more than 10 minutes inside "
              f"the session in some phase\n  that was NEITHER continuous "
              f"trading NOR a declared break NOR missing:")
        # the header
        print(f"      {'date':<12s} {'seconds':>9s}  phases the exchange "
              f"reported")
        # worst first
        for _, r in odd_phase.sort_values("other_seconds",
                                          ascending=False).head(20).iterrows():
            # the row
            print(f"      {r['date']:<12s} {r['other_seconds']:>8,.0f}s  "
                  f"{r['other_phases']}")
        print("      A market halt looks like this. The phases are the")
        print("      exchange's own words; no cause is asserted here.")

    # ---- THE SECONDS MUST ADD UP, EXACTLY --------------------------------
    # traded + break + unobserved + other == open-to-close, to the second, on
    # every date. NO TOLERANCE. Every one of those five numbers is a count of
    # seconds on the same per-second grid, so there is nothing for a
    # tolerance to absorb; an earlier version allowed a second per continuous
    # stretch and that slack silently hid a constant off-by-one in
    # open_to_close_seconds on all 207 dates.
    _sum = (D["traded_seconds"] + D["break_seconds"]
            + D["unobserved_seconds"] + D["other_seconds"])
    # how far off each date is
    _off = (D["open_to_close_seconds"] - _sum).abs()
    # anything at all
    _bad = D[_off > 0]
    # reported loudly, because it means the accounting is wrong
    if len(_bad):
        print(f"\n  ACCOUNTING DOES NOT BALANCE on {len(_bad)} date(s) -- "
              f"this is a bug in session_calendar.py, not a fact about PSX:")
        # each one
        for _, r in _bad.head(10).iterrows():
            print(f"      {r['date']}  open-to-close "
                  f"{r['open_to_close_seconds']:,.0f}s vs traded+break+"
                  f"unobserved+other "
                  f"{r['traded_seconds'] + r['break_seconds'] + r['unobserved_seconds'] + r['other_seconds']:,.0f}s")
    else:
        print(f"\n  seconds balance on all {len(D)} dates: traded + break + "
              f"unobserved + other = open-to-close.")

    # dates that could not be measured at all
    if skipped:
        print(f"\n  {len(skipped)} date(s) could not be measured:")
        # each with its reason
        for date, why in skipped:
            print(f"      {date}  {why}")

    # ---- WRITE ------------------------------------------------------------
    # where it goes
    outdir = Path(args.out) if args.out else Path(RESULTS_ROOT)
    # a stamp, so a rerun never overwrites an earlier calendar
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    # the file
    path = outdir / f"session_calendar_{stamp}.csv"
    # the columns, in a deliberate order
    cols = ["date", "weekday", "session_id", "is_reg_board", "day_type",
            "is_friday", "is_short",
            "open_utc", "open_pkt", "close_utc", "close_pkt",
            "open_to_close_seconds", "traded_seconds", "continuous_spans",
            "break_seconds", "break_reasons", "unobserved_seconds",
            "other_seconds", "other_phases",
            "exchange_says_friday", "status_messages"]
    # EVERY BOARD goes in the file, with the regular one flagged, so nothing
    # is hidden by the choice pick_board made and a wrong pick can be
    # overridden by filtering the CSV differently.
    OUT = ALL.merge(D[["date", "day_type", "is_friday", "is_short"]],
                    on="date", how="left")
    # write it
    OUT.reindex(columns=cols).to_csv(path, index=False)
    # say where, and what is in it
    print(f"\nwrote {path}")
    print(f"  {len(OUT):,} rows -- one per market per date, "
          f"{int(ALL['is_reg_board'].sum()):,} of them the regular market")

    # ---- THE INTERVALS FILE, which is the one a consumer actually uses ----
    # THE SUMMARY ABOVE CANNOT ANSWER "was this moment inside the session?"
    # On a Friday the Jumu'ah break sits between open and close, so a script
    # that tests open <= t <= close counts two and a quarter hours of closed
    # market as trading time. That is precisely the error that put 80,000
    # phantom seconds of "feed outage" in the first vendor report.
    spans_path = outdir / f"session_spans_{stamp}.csv"
    # one row per continuous stretch
    span_rows = []
    # every market's row, so a consumer can pick a different market if needed
    for r in rows:
        # each stretch of that market's day
        for i, (s, e) in enumerate(r.get("intervals") or [], 1):
            # the row
            span_rows.append({
                "date": r["date"],
                "session_id": r["session_id"],
                "is_reg_board": r["session_id"] == reg,
                # which stretch: 1 on an ordinary day, 1 and 2 on a Friday
                "span": i,
                # both clocks, as for the summary
                "start_utc": s,
                "start_pkt": s + PKT_OFFSET,
                "end_utc": e,
                "end_pkt": e + PKT_OFFSET,
                # its own length, so the parts sum to traded_seconds
                "seconds": float((e - s).total_seconds()),
            })
    # written only if there is something to write
    if span_rows:
        # the frame
        S = pd.DataFrame(span_rows)
        # out it goes
        S.to_csv(spans_path, index=False)
        # say where, and how many of them belong to the regular market
        print(f"wrote {spans_path}")
        print(f"  {len(S):,} rows -- one per continuous stretch, "
              f"{int(S['is_reg_board'].sum()):,} of them the regular market")
    print()
    print("  USE session_spans_*.csv, NOT the summary, to decide whether a")
    print("  moment was inside the session. Filter to is_reg_board and test")
    print("  membership of the stretches: on a Friday there are two, and the")
    print("  gap between them is a closed market, not a feed outage.")
    print()
    print("  traded_seconds in the summary is the denominator for any rate")
    print("  metric: it EXCLUDES the Friday break, which")
    print("  open_to_close_seconds does not.")


# entry point
if __name__ == "__main__":
    main()
