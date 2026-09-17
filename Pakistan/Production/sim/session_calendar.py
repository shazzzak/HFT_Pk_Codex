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
# dates and consequently reported 80,284 seconds of "feed outage" that was the
# market being closed during Ramadan. 64% of its headline number.
#
# THIS FILE DOES NOT ASSUME ANY CLOCK. It reads the exchange's own phase
# declarations and reports what each date actually did. A wrong idea about
# when Ramadan started cannot corrupt the answer, because no calendar is
# consulted to produce the session -- only to LABEL it afterwards, and the
# label is printed so it can be checked by eye.
#
# THE SOURCE. TradingSessionStatus messages (35=h) carry TradingPhaseCode
# (tag 8538), a fixed-width string whose characters are
# [phase][suspended][break_reason]:
#
#   phase        T=continuous  O=open call auction  B=break  A=after hours
#                E=closed  S=starting  N/V=normal call auction  C=close
#                H=temporary suspension
#   break_reason 1=after pre-open  2=FRIDAY LUNCH BREAK
#                3=after pre-open PM Friday  4=before post-close
#
# Break reasons 2 and 3 are the Friday split, stated by the exchange itself.
# They are the reason this file can tell a Friday from an ordinary day without
# looking at a calendar at all.
#
# The parser keeps 35=h messages in the `misc` table with the whole message
# body in `raw`, so this reads ~400k rows a date rather than the 33 million in
# ob_snapshot. The phase is identical in both; ob_snapshot is simply the
# expensive way to ask.
#
# READ-ONLY. Writes one timestamped CSV. Never overwrites.
#
# Run from Production/:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/session_calendar.py
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/session_calendar.py --days 20
# ============================================================================

# command-line flags
import argparse
# timestamped output names
import datetime as dt
# path handling
from pathlib import Path
# the date comes off the filename
import re

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

# HOW MUCH EARLIER THAN ITS PEERS A DATE MUST CLOSE to be labelled a shortened
# session, in minutes. 30 is comfortably larger than the few-minutes jitter
# between ordinary dates and far smaller than the ~65-minute Ramadan
# shortening, so the classification is not sensitive to where this sits. It
# affects the LABEL only -- every measured time in the output is unaffected.
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


def phase_codes(raw):
    """Tag 8538 out of a raw FIX body, or None.

    A targeted search rather than a full tokenize: this runs over hundreds of
    thousands of rows a date and reads exactly one field.
    """
    # a missing body cannot be read
    if not isinstance(raw, str):
        return None
    # the separator-prefixed key, so "8538=" cannot match inside "18538=..."
    key = SOH + "8538="
    # where it sits
    i = raw.find(key)
    # not found mid-message; it may still be the very first field
    if i < 0:
        # the first field carries no leading separator
        if raw.startswith("8538="):
            # value starts just past "8538="
            off = 5
        else:
            # genuinely absent
            return None
    else:
        # value starts just past the matched key
        off = i + len(key)
    # the end of this field
    j = raw.find(SOH, off)
    # last field on the line has no trailing separator
    return raw[off:] if j < 0 else raw[off:j]


def read_phases(date):
    """The exchange's phase timeline for one date, as a sorted frame.

    Returns a frame of (capture_ts, phase, break_reason), one row per 35=h
    message, or None when the date has no misc partition.
    """
    # where the misc partition lives
    p = partition("misc", date)
    # a missing partition is reported by the caller, never assumed empty
    if not p.exists():
        return None
    # three columns only: this table is ~400k rows a date and the rest is
    # irrelevant here
    m = pd.read_parquet(p, columns=["capture_ts", "msg_type", "raw"])
    # TradingSessionStatus messages only
    h = m[m["msg_type"].astype("string") == "h"]
    # a date with no session-status messages cannot be measured this way
    if len(h) == 0:
        return None
    # tag 8538 off each body
    codes = h["raw"].map(phase_codes)
    # the frame this function promises
    out = pd.DataFrame({
        # arrival time, which is the clock a session is quoted on
        "capture_ts": pd.to_datetime(h["capture_ts"], utc=True).values,
        # first character -> readable phase
        "phase": codes.str.slice(0, 1).map(PHASE_MAP).values,
        # third character -> break reason, meaningful only during a break
        "break_reason": codes.str.slice(2, 3).map(BREAK_REASON_MAP).values,
    })
    # drop rows whose code could not be read at all
    out = out.dropna(subset=["phase"])
    # in time order, which every span calculation below assumes
    return out.sort_values("capture_ts").reset_index(drop=True)


def spans(ph):
    """Collapse a phase timeline into contiguous spans.

    Returns a list of (phase, break_reason, start, end). The end of a span is
    the start of the next one, because a phase message states a state that
    holds until the next message changes it.
    """
    # nothing to collapse
    if ph is None or len(ph) == 0:
        return []
    # mark where the phase changes
    changed = (ph["phase"] != ph["phase"].shift(1))
    # the index of each change
    starts = list(ph.index[changed])
    # the result
    out = []
    # each span runs from its own start to the next span's start
    for k, i in enumerate(starts):
        # the last span ends at the last message we hold
        j = starts[k + 1] if k + 1 < len(starts) else len(ph) - 1
        # the span
        out.append((ph["phase"].iloc[i],
                    ph["break_reason"].iloc[i],
                    ph["capture_ts"].iloc[i],
                    ph["capture_ts"].iloc[j]))
    # in time order
    return out


def measure(date, ph):
    """One date's measured session, as a dict ready for the output frame."""
    # the phase spans
    sp = spans(ph)
    # every continuous-trading span
    cont = [s for s in sp if s[0] == "CONTINUOUS_AUCTION"]
    # a date with no continuous trading has no session to measure
    if not cont:
        return None
    # the day's first continuous open and last continuous close
    open_utc = min(s[2] for s in cont)
    close_utc = max(s[3] for s in cont)
    # TRADED SECONDS: the sum of the continuous spans, NOT close minus open.
    # On a Friday the two differ by the length of the Jumu'ah break, and the
    # traded figure is the one any rate metric wants.
    traded = sum((s[3] - s[2]).total_seconds() for s in cont)
    # every break that falls inside the trading day
    brk = [s for s in sp
           if s[0] == "TRADING_BREAK" and open_utc <= s[2] <= close_utc]
    # how long the market stood in a break inside the session
    brk_secs = sum((s[3] - s[2]).total_seconds() for s in brk)
    # which breaks they were, named by the exchange
    brk_names = ",".join(sorted({s[1] for s in brk if s[1]})) or ""
    # the weekday, which is half the day-type answer on its own
    wd = pd.Timestamp(date).day_name()
    # THE FRIDAY TEST, taken from the exchange rather than from the weekday:
    # break reasons 2 and 3 exist only on a Friday.
    friday_break = any(s[1] in ("FRIDAY_LUNCH_BREAK",
                                "AFTER_PRE_OPEN_PM_FRIDAY") for s in brk)
    # the row
    return {
        # the date
        "date": date,
        # its weekday
        "weekday": wd,
        # continuous open, both clocks
        "open_utc": open_utc,
        "open_pkt": open_utc + PKT_OFFSET,
        # continuous close, both clocks
        "close_utc": close_utc,
        "close_pkt": close_utc + PKT_OFFSET,
        # the span from open to close, breaks included
        "open_to_close_seconds": (close_utc - open_utc).total_seconds(),
        # THE NUMBER THAT MATTERS: seconds of actual continuous trading
        "traded_seconds": traded,
        # how many continuous stretches made it up (2 on a split Friday)
        "continuous_spans": len(cont),
        # the break time inside the session
        "break_seconds": brk_secs,
        # and what the exchange called those breaks
        "break_reasons": brk_names,
        # whether the exchange itself declared a Friday break
        "exchange_says_friday": friday_break,
    }


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # a short run, for a smoke test
    ap.add_argument("--days", type=int, default=0,
                    help="only the most recent N dates (0 = every date)")
    # where the CSV goes
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to the results root")
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
    # narrow if asked
    if args.days:
        dates = dates[-args.days:]
    print(f"  dates : {len(dates)}  ({dates[0]} .. {dates[-1]})")
    print()

    # one row per date
    rows = []
    # dates that could not be measured, and why
    skipped = []
    # walk them
    for k, date in enumerate(dates, 1):
        # the exchange's phase timeline
        ph = read_phases(date)
        # no session-status messages means no measurement from this source
        if ph is None:
            # record it rather than let it vanish
            skipped.append((date, "no 35=h session-status messages in misc"))
            print(f"  [{k}/{len(dates)}] {date}  SKIPPED (no phase messages)")
            continue
        # the measurement
        r = measure(date, ph)
        # a date that never entered continuous trading
        if r is None:
            skipped.append((date, "no CONTINUOUS_AUCTION phase"))
            print(f"  [{k}/{len(dates)}] {date}  SKIPPED (never opened)")
            continue
        # keep it
        rows.append(r)
        # progress, with the two numbers worth watching
        print(f"  [{k}/{len(dates)}] {date} {r['weekday'][:3]}  "
              f"{str(r['open_pkt'])[11:19]}-{str(r['close_pkt'])[11:19]} PKT  "
              f"traded {r['traded_seconds']:>7,.0f}s  "
              f"breaks {r['break_seconds']:>6,.0f}s {r['break_reasons']}")

    # nothing measurable
    if not rows:
        raise SystemExit("no date could be measured")

    # the frame
    D = pd.DataFrame(rows)

    # ---- CLASSIFY, using the measurements and nothing else ---------------
    # A date is a Friday if the exchange declared a Friday break on it. The
    # weekday is carried too, and a disagreement between the two is printed
    # below rather than silently resolved.
    D["is_friday"] = D["exchange_says_friday"] | (D["weekday"] == "Friday")
    # THE SHORTENED-SESSION TEST. Compare each date's traded seconds against
    # the MEDIAN for its own Friday class, so a Friday is never called short
    # merely for being a Friday.
    D["class_median_traded"] = D.groupby("is_friday")["traded_seconds"] \
                                .transform("median")
    # short by more than the threshold
    D["is_short"] = (
        (D["class_median_traded"] - D["traded_seconds"])
        > SHORT_SESSION_MINUTES * 60.0)
    # THE FOUR TYPES. "RAMADAN" is the label for a shortened session; it is
    # named for the cause SZ identified, and the measurement that produced it
    # is the shortening itself, not a religious calendar.
    D["day_type"] = [
        ("RAMADAN_FRIDAY" if f else "RAMADAN_REGULAR") if s
        else ("REGULAR_FRIDAY" if f else "REGULAR_DAY")
        for f, s in zip(D["is_friday"], D["is_short"])]

    # ---- REPORT -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("THE FOUR DAY TYPES, AS MEASURED")
    print("=" * 78)
    # the header
    print(f"\n  {'day type':<18s} {'dates':>6s} {'median open':>12s} "
          f"{'median close':>13s} {'median traded':>14s} {'breaks':>8s}")
    # one line per type, in a fixed order so the table reads the same each run
    for t in ("REGULAR_DAY", "REGULAR_FRIDAY", "RAMADAN_REGULAR",
              "RAMADAN_FRIDAY"):
        # that type's dates
        g = D[D["day_type"] == t]
        # a type with no dates still gets a line, so its absence is visible
        if len(g) == 0:
            print(f"  {t:<18s} {0:>6d}            --            -- "
                  f"            --       --")
            continue
        # the median clock times, in PKT
        o = pd.Series([x.time() for x in g["open_pkt"]]).astype(str).median()
        c = pd.Series([x.time() for x in g["close_pkt"]]).astype(str).median()
        # the row
        print(f"  {t:<18s} {len(g):>6d} {o[:8]:>12s} {c[:8]:>13s} "
              f"{g['traded_seconds'].median():>13,.0f}s "
              f"{g['break_seconds'].median():>7,.0f}s")

    # the shortened block, listed in full so it can be checked against the
    # actual Ramadan dates by eye
    short = D[D["is_short"]].sort_values("date")
    # only if there is one
    if len(short):
        print(f"\n  SHORTENED SESSIONS ({len(short)} dates) -- check this "
              f"block against the Ramadan calendar")
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
    cols = ["date", "weekday", "day_type", "is_friday", "is_short",
            "open_utc", "open_pkt", "close_utc", "close_pkt",
            "open_to_close_seconds", "traded_seconds", "continuous_spans",
            "break_seconds", "break_reasons", "exchange_says_friday"]
    # write it
    D[cols].to_csv(path, index=False)
    # say where
    print(f"\nwrote {path}")
    print()
    print("  Every script that needs a session window should read this file")
    print("  rather than hard-coding a clock. traded_seconds is the")
    print("  denominator for any rate metric: it EXCLUDES the Friday break,")
    print("  which open_to_close_seconds does not.")


# entry point
if __name__ == "__main__":
    main()
