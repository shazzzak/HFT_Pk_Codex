# ============================================================================
# test_session_calendar.py -- does the trading calendar measure what it claims?
# ============================================================================
# WHY THIS FILE EXISTS.
#
# session_calendar.py took four attempts to get right, and every one of the
# three failures was silent: it produced a number, or produced nothing, and
# said nothing about being broken. Each failure is a case below, so none of
# them can come back unnoticed.
#
#   ATTEMPT 1  pooled every board into one timeline and collapsed it into
#              spans -> 8,500 traded seconds inside a 25,200-second session
#   ATTEMPT 2  filtered on the `segment` column -> 35=h messages carry no
#              tag 1500, so the column is null and every date was skipped
#   ATTEMPT 3  built the per-second grid with pd.date_range and reindexed
#              onto it -> a timestamp-resolution mismatch matched nothing and
#              every date reported "never opened"
#   ATTEMPT 4  worked, but took the union across all twelve boards, so the
#              day closed whenever the LAST board closed; and it classified
#              shortened sessions on traded seconds, so a date that merely
#              lost feed was labelled Ramadan
#
# Run from Production/:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/test_session_calendar.py
#
# READ-ONLY. Builds its own synthetic data. Touches neither the store nor the
# results directory.
# ============================================================================

# exit codes
import sys
# stand in for the trades partition without a real parquet
import types

# frames
import pandas as pd

# the module under test, imported the way the sim package imports its siblings
try:
    import session_calendar as SC
except ImportError:                                            # noqa: BLE001
    # run from Production/ rather than from Production/sim/
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import session_calendar as SC


# every (name, passed?) pair
results = []


def check(name, condition, detail=""):
    # keep it for the summary, and show it as it happens
    results.append((name, bool(condition)))
    print(("PASS  " if condition else "FAIL  ") + name)
    # a failure explains itself on the next line
    if not condition and detail:
        print("        " + detail)


def status_frame(specs, unit="ns"):
    """Build a status stream from (board, phase_fn) pairs.

    phase_fn takes the clock hour in PKT as a float and returns
    (phase, break_reason). One message per board every 3 seconds, which is
    the cadence the real feed uses.
    """
    # the day, wide enough to hold a pre-open and an after-hours session
    start = pd.Timestamp("2026-06-30 02:00:00", tz="UTC")
    end = pd.Timestamp("2026-06-30 13:30:00", tz="UTC")
    # the rows
    rows = []
    # every third second
    for t in pd.date_range(start, end, freq="3s"):
        # the same instant in Pakistan time, which is how a session is quoted
        pkt = t + SC.PKT_OFFSET
        # as a decimal hour
        h = pkt.hour + pkt.minute / 60 + pkt.second / 3600
        # one message per board
        for board, fn in specs:
            # what that board was doing
            phase, reason = fn(h)
            # the message
            rows.append({"capture_ts": t, "session_id": board,
                         "phase": phase, "break_reason": reason, "raw": "x"})
    # sorted by arrival, as read_status returns it
    df = pd.DataFrame(rows).sort_values("capture_ts").reset_index(drop=True)
    # the timestamp resolution under test
    df["capture_ts"] = df["capture_ts"].astype(f"datetime64[{unit}, UTC]")
    return df


def equity(h):
    """The regular equity board: 09:30-15:30 PKT."""
    # continuous through the session
    if 9.5 <= h < 15.5:
        return "CONTINUOUS_AUCTION", None
    # closed after the bell
    if h >= 15.5:
        return "MARKET_CLOSED", None
    # pre-open before it
    return "STARTING", None


def after_hours(h):
    """An after-hours board: 15:45-16:30 PKT, i.e. entirely after the bell."""
    # continuous only after the equity board has closed
    if 15.75 <= h < 16.5:
        return "CONTINUOUS_AUCTION", None
    # otherwise idle
    return "STARTING", None


def friday_equity(h):
    """The equity board on a Friday: 09:15-12:00, lunch, 14:30-16:30 PKT."""
    # the morning session
    if 9.25 <= h < 12.0:
        return "CONTINUOUS_AUCTION", None
    # the Jumu'ah break, which the exchange names
    if 12.0 <= h < 14.5:
        return "TRADING_BREAK", "FRIDAY_LUNCH_BREAK"
    # the afternoon session
    if 14.5 <= h < 16.5:
        return "CONTINUOUS_AUCTION", None
    # closed after it
    if h >= 16.5:
        return "MARKET_CLOSED", None
    # pre-open before it
    return "STARTING", None


# ==========================================================================
# PART 1 -- READING THE MESSAGES. Attempt 2 died here.
# ==========================================================================
print("PART 1 -- READING TAGS OFF A REAL BODY")

# bodies copied verbatim from the probe's output, plus a break code
BODIES = [
    "8=FIXT.1.1^9=107^35=h^49=NMDU001Q0001^56=PSX^34=548^"
    "52=20260629-22:41:45.239^42=20260629-22:41:45.000^10201=1^336=02^"
    "8538=S^10=175^",
    "8=FIXT.1.1^9=109^35=h^49=NMDU001Q0001^56=PSX^34=11211^"
    "52=20260630-12:05:37.448^42=20260630-12:05:36.000^10201=1^336=13^"
    "8538=E^10=235^",
    # the THREE-character break code: phase, a SPACE, then the reason
    "8=FIXT.1.1^9=109^35=h^49=X^56=PSX^34=1^52=X^42=X^10201=1^336=01^"
    "8538=B 2^10=235^",
]
# as a column
S = pd.Series(BODIES)
# the vectorised extract the loader uses
codes = S.str.extract(r"\^8538=([^\^]*)", expand=False)
boards = S.str.extract(r"\^336=([^\^]*)", expand=False)
# it must agree with the single-message reader on every one
check("the vectorised tag read matches the per-message reader",
      list(codes) == [SC.field(b, "8538") for b in BODIES]
      and list(boards) == [SC.field(b, "336") for b in BODIES],
      f"got {list(codes)} / {list(boards)}")
# tag 336 is the board, and it is populated -- the column `segment` is NOT
check("tag 336 carries a board id on every message",
      list(boards) == ["02", "13", "01"], f"got {list(boards)}")
# the phase is character 1
check("the phase decodes from character 1",
      list(codes.str.slice(0, 1).map(SC.PHASE_MAP))
      == ["STARTING", "MARKET_CLOSED", "TRADING_BREAK"])
# the break reason is character 3, ACROSS THE SPACE at character 2
check("the break reason decodes from character 3, across the space",
      list(codes.str.slice(2, 3).map(SC.BREAK_REASON_MAP))[2]
      == "FRIDAY_LUNCH_BREAK",
      f"got {list(codes.str.slice(2, 3).map(SC.BREAK_REASON_MAP))}")

# ==========================================================================
# PART 1b -- THE WHOLE CHAIN, THROUGH read_status
# ==========================================================================
# WHY THIS CASE EXISTS. Every other case in this file builds its status
# frame by hand, with timezone-aware timestamps, and so never exercised
# read_status itself. read_status built its frame with `.values`, which on a
# timezone-aware datetime Series returns a plain numpy array and SILENTLY
# DROPS THE TIMEZONE. The result measured fine on its own and then crashed
# the moment it met a timezone-aware timestamp from the trades table. The
# hand-built cases all passed while the real run failed twice.
print("\nPART 1b -- THE WHOLE CHAIN, THROUGH read_status")


def misc_frame(unit="us"):
    """A misc partition as the parser writes one, for one board's day.

    Timestamps are timezone-aware, which is what parquet returns, and the
    bodies are real 35=h messages with the board and phase in them.
    """
    # the rows
    rows = []
    # every third second across an ordinary day
    for t in pd.date_range("2026-06-30 02:00:00", "2026-06-30 13:30:00",
                           freq="3s", tz="UTC"):
        # the same instant in Pakistan time
        pkt = t + SC.PKT_OFFSET
        # as a decimal hour
        hh = pkt.hour + pkt.minute / 60 + pkt.second / 3600
        # the equity board's phase code at that moment
        code = "T" if 9.5 <= hh < 15.5 else ("E" if hh >= 15.5 else "S")
        # a real message body
        rows.append({
            "capture_ts": t,
            "msg_type": "h",
            "raw": (f"8=FIXT.1.1^9=107^35=h^49=NMDU001Q0001^56=PSX^34=1^"
                    f"52=X^42=X^10201=1^336=01^8538={code}^10=175^"),
        })
        # and some other message type, which read_status must filter out
        rows.append({"capture_ts": t, "msg_type": "UA001", "raw": "8=X^35=UA001^"})
    # as a frame, at the resolution under test
    df = pd.DataFrame(rows)
    # parquet hands back tz-aware timestamps at some resolution
    df["capture_ts"] = df["capture_ts"].astype(f"datetime64[{unit}, UTC]")
    return df


# stand in for the misc partition and its parquet read
_rp, _rr = SC.partition, SC.pd.read_parquet
SC.partition = lambda table, date: types.SimpleNamespace(exists=lambda: True)
SC.pd.read_parquet = lambda *a, **k: misc_frame("us")
# the real loader
st_real = SC.read_status("2026-06-30")
# put the real ones back
SC.partition, SC.pd.read_parquet = _rp, _rr
# it must have returned something
check("read_status returns the status messages", st_real is not None
      and len(st_real) > 0)
# THE CASE THAT MATTERS: the column it returns must still carry a timezone
check("read_status keeps the timezone on capture_ts",
      st_real is not None
      and getattr(st_real["capture_ts"].dtype, "tz", None) is not None,
      "a tz-naive column here measures fine alone and then cannot be "
      "compared with the tz-aware timestamps from the trades table")
# it must have filtered to 35=h only
check("read_status keeps only 35=h messages",
      st_real is not None and len(st_real) == len(misc_frame("us")) // 2,
      f"got {len(st_real) if st_real is not None else None}")
# the board and phase must have decoded
check("read_status decodes the board and the phase",
      st_real is not None
      and set(st_real["session_id"].dropna().unique()) == {"01"}
      and "CONTINUOUS_AUCTION" in set(st_real["phase"].unique()))
# and the whole chain must run: measure, then pick_board against real trades
if st_real is not None:
    # the board's session
    r_real = SC.measure("2026-06-30", st_real, "01")
    # the regular-market trades, tz-aware as parquet returns them
    tr = pd.DataFrame({
        "capture_ts": pd.date_range("2026-06-30 04:30:00",
                                    "2026-06-30 10:30:00", freq="10s",
                                    tz="UTC"),
    })
    # all regular market
    tr["market"] = "REG"
    # stand in again
    _rp, _rr = SC.partition, SC.pd.read_parquet
    SC.partition = lambda table, date: types.SimpleNamespace(
        exists=lambda: True)
    SC.pd.read_parquet = lambda *a, **k: tr
    # the choice, and any exception
    try:
        picked_real = SC.pick_board("2026-06-30", [r_real])
        err_real = None
    except Exception as exc:                                   # noqa: BLE001
        picked_real, err_real = None, exc
    # restore
    SC.partition, SC.pd.read_parquet = _rp, _rr
    # the chain must complete without a timezone error
    check("read_status -> measure -> pick_board completes end to end",
          err_real is None and picked_real == "01",
          f"raised {err_real!r}" if err_real else f"picked {picked_real}")
    # and give the right session
    check("  and the measured session is 09:30-15:30",
          r_real is not None and str(r_real["open_pkt"])[11:16] == "09:30"
          and str(r_real["close_pkt"])[11:16] == "15:29",
          f"got {str(r_real['open_pkt'])[11:19]}-"
          f"{str(r_real['close_pkt'])[11:19]}" if r_real else "not measured")

# ==========================================================================
# PART 2 -- TIMESTAMP RESOLUTION. Attempt 3 died here.
# ==========================================================================
print("\nPART 2 -- TIMESTAMP RESOLUTION")

# the same day at three resolutions must give the same answer
for unit in ("ns", "us", "ms"):
    # one board, an ordinary day
    st = status_frame([("01", equity)], unit=unit)
    # measure it
    r = SC.measure("2026-06-30", st, "01")
    # a resolution mismatch shows up as "never opened"
    check(f"a {unit}-resolution capture is measured, not skipped",
          r is not None,
          "this is the pd.date_range reindex failure: mismatched timestamp "
          "resolutions match nothing and the whole day reads as closed")
    # and the answer must be the same one
    if r is not None:
        check(f"  and {unit} gives the 09:30-15:30 session",
              str(r["open_pkt"])[11:16] == "09:30"
              and str(r["close_pkt"])[11:16] == "15:29",
              f"got {str(r['open_pkt'])[11:19]}-{str(r['close_pkt'])[11:19]}")

# ==========================================================================
# PART 3 -- ONE BOARD, NOT TWELVE. Attempt 4's first fault.
# ==========================================================================
print("\nPART 3 -- PER BOARD, NOT POOLED")

# the equity board plus THREE after-hours boards, so the after-hours boards
# outnumber it and win any vote taken across the whole stream
st = status_frame([("01", equity), ("07", after_hours),
                   ("08", after_hours), ("12", after_hours)])
# the pooled measurement, which is what attempt 4 produced
pooled = SC.measure("2026-06-30", st, None)
# the equity board measured on its own
own = SC.measure("2026-06-30",
                 st[st["session_id"] == "01"].reset_index(drop=True), "01")
# the equity board's own session is the real one
check("the equity board's own session is 09:30-15:30",
      own is not None and str(own["open_pkt"])[11:16] == "09:30"
      and str(own["close_pkt"])[11:16] == "15:29",
      f"got {str(own['open_pkt'])[11:19]}-{str(own['close_pkt'])[11:19]}"
      if own else "not measured")
# pooling gives something else entirely, which is the point
check("pooling every board gives a DIFFERENT and wrong answer",
      pooled is not None
      and str(pooled["close_pkt"])[11:16] != str(own["close_pkt"])[11:16],
      "if these agree this case is not exercising the fault any more")
# say what pooling actually returned, so the size of the error is on record
if pooled is not None and own is not None:
    print(f"        pooled : {str(pooled['open_pkt'])[11:19]}-"
          f"{str(pooled['close_pkt'])[11:19]} PKT, "
          f"{pooled['traded_seconds']:,.0f}s traded")
    print(f"        board 01: {str(own['open_pkt'])[11:19]}-"
          f"{str(own['close_pkt'])[11:19]} PKT, "
          f"{own['traded_seconds']:,.0f}s traded")

# ---- and the right board is identified from the trades --------------------
# every board's measurement
day = [SC.measure("2026-06-30",
                  st[st["session_id"] == b].reset_index(drop=True), b)
       for b in ("01", "07", "08", "12")]
# the regular-market trades sit inside the equity board's session
trades = pd.DataFrame({
    "capture_ts": pd.date_range("2026-06-30 04:30:00", "2026-06-30 10:30:00",
                                freq="10s", tz="UTC"),
})
# all regular market
trades["market"] = "REG"
# EVERY TIMESTAMP RESOLUTION, because this is where a tz-aware quantile
# returning a tz-NAIVE timestamp crashed the first real run of the per-board
# version. The comparison is arithmetic now, so resolution cannot matter --
# this case is what keeps it that way.
for unit in ("ns", "us", "ms"):
    # the same trades at this resolution
    tr = trades.copy()
    tr["capture_ts"] = tr["capture_ts"].astype(f"datetime64[{unit}, UTC]")
    # stand in for the partition and the parquet read, so no store is needed
    _real_partition, _real_read = SC.partition, SC.pd.read_parquet
    SC.partition = lambda table, date: types.SimpleNamespace(
        exists=lambda: True)
    SC.pd.read_parquet = lambda *a, _tr=tr, **k: _tr
    # the choice, and any exception it raises
    try:
        picked = SC.pick_board("2026-06-30", day)
        err = None
    except Exception as exc:                                   # noqa: BLE001
        picked, err = None, exc
    # put the real ones back before anything else runs
    SC.partition, SC.pd.read_parquet = _real_partition, _real_read
    # it must be the equity board, and it must not raise
    check(f"pick_board picks the REG board from a {unit}-resolution capture",
          picked == "01",
          f"raised {err!r}" if err else f"picked {picked}, expected 01")

# ==========================================================================
# PART 3b -- THE CHOICE MUST BE STABLE ACROSS DATES
# ==========================================================================
# WHY THIS CASE EXISTS. The first working run picked board 05 on eleven of
# fifteen dates and boards 03, 07 and 09 on the other four. Several PSX
# boards keep near-identical hours, so their scores land within noise of each
# other and whichever happens to be a second wider on a given day wins it.
# The regular equity board does not change from Monday to Tuesday, so the
# choice has to be made once across every date, not once per date.
print("\nPART 3b -- A STABLE BOARD CHOICE")


def near_identical(seed):
    """Three boards trading the same hours, differing by a few seconds.

    `seed` shifts which of them happens to be widest, exactly as real
    second-level jitter does from one date to the next.
    """
    # a fixed close, and opens that differ by a second or two
    specs = []
    # three boards, each opening a hair earlier or later
    for i, b in enumerate(("03", "05", "09")):
        # this board's open, shifted by the seed so the widest one rotates
        shift = ((i + seed) % 3) / 3600.0
        # its phase function
        specs.append((b, (lambda h, s=shift:
                          ("CONTINUOUS_AUCTION", None)
                          if 9.5 + s <= h < 15.5
                          else (("MARKET_CLOSED", None) if h >= 15.5
                                else ("STARTING", None)))))
    # the stream
    return status_frame(specs)


# the trades, identical on every date
tr_stable = pd.DataFrame({
    "capture_ts": pd.date_range("2026-06-30 04:30:00", "2026-06-30 10:30:00",
                                freq="10s", tz="UTC"),
})
# regular market
tr_stable["market"] = "REG"
# accumulate scores across five dates, the way main() does
totals = {}
# and record each date's own winner, to show the wandering
per_date = []
# five synthetic dates, each with a different board happening to be widest
for seed in range(5):
    # that date's stream
    st_s = near_identical(seed)
    # every board's measurement
    day_s = [SC.measure("2026-06-30",
                        st_s[st_s["session_id"] == b].reset_index(drop=True), b)
             for b in ("03", "05", "09")]
    # stand in for the trades read
    _rp, _rr = SC.partition, SC.pd.read_parquet
    SC.partition = lambda table, date: types.SimpleNamespace(
        exists=lambda: True)
    SC.pd.read_parquet = lambda *a, **k: tr_stable
    # this date's errors
    sc_s = SC.board_scores("2026-06-30", day_s)
    # restore
    SC.partition, SC.pd.read_parquet = _rp, _rr
    # accumulate, as a list, because the choice is made on the median
    for b, v in sc_s.items():
        totals.setdefault(b, []).append(v)
    # this date's own winner -- the SMALLEST error
    per_date.append(min(sc_s, key=lambda kk: sc_s[kk]) if sc_s else None)

# errors are returned for every board, not just a winner
check("board_scores returns an error per board, not a single verdict",
      isinstance(totals, dict) and set(totals) == {"03", "05", "09"},
      f"got {set(totals)}")
# the per-date winner does wander, which is the fault being guarded against
print(f"        per-date winners: {per_date}")
# the run-level choice is one board, whatever the per-date winners did
meds = {b: float(pd.Series(v).median()) for b, v in totals.items()}
# the same tie-break main() uses: lowest error, then lowest board id
chosen = min(meds, key=lambda k: (meds[k], str(k)))
check("a single board is chosen for the whole run",
      chosen in {"03", "05", "09"},
      f"chose {chosen}")
# THE TIE CASE. These three boards are symmetric by construction, so their
# medians come out exactly equal and min() would otherwise return whichever
# the dictionary happened to yield first -- a different answer on a different
# run, for no reason in the data.
print(f"        median errors: "
      + ", ".join(f"{b}={meds[b]:.0f}s" for b in sorted(meds)))
# reversed insertion order must give the same answer
rev = {b: meds[b] for b in reversed(list(meds))}
check("  and the choice is the same whichever order the dates came in",
      chosen == min(rev, key=lambda k: (rev[k], str(k))),
      "the accumulation is order-dependent, which it must not be")

# ---- THE MARKET IS NAMED BY THE SPEC, NOT INFERRED FROM TIMING ----------
# THE MISTAKE THIS GUARDS AGAINST. The timing metric below was used to SELECT
# the regular market, and on real data it selected '08' -- the Odd Lot Market
# -- because the odd lot bell sits within 2 seconds of the first and last
# regular-market print while the Regular Market's own bell is about 2 minutes
# away. The reason is that the first and last REG prints of a day are AUCTION
# CROSSES, which happen outside the continuous session. The PSX FIX spec,
# section 4.2.1, lists tag 336 as MarketCode and says '01' is the Regular
# Market. It was in the project the whole time.
check("the spec's market codes are carried, not inferred",
      SC.MARKET_CODE.get("01") == "Regular Market"
      and SC.MARKET_CODE.get("08") == "Odd Lot Market",
      f"got 01={SC.MARKET_CODE.get('01')}, 08={SC.MARKET_CODE.get('08')}")
check("the regular market is 01, per the spec",
      SC.REGULAR_MARKET == "01", f"got {SC.REGULAR_MARKET}")
# every code the real feed was observed to carry, on 2026-03-11/13/25/27
SEEN_CODES = ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
              "12", "13")
# any the dictionary is missing
MISSING = [c for c in SEEN_CODES if c not in SC.MARKET_CODE]
check("and every code the feed carries is in the dictionary",
      not MISSING, f"missing {MISSING}")

# ---- THE METRIC ITSELF: minutes late must lose to seconds late -----------
# Kept as a CROSS-CHECK only. It is no longer what picks the market.
# This is the discovery that the real data forced. Scoring Jaccard overlap
# against the 5th-to-95th percentile of print times rated a board whose bell
# is two MINUTES from the real one at 0.9130 and the board whose bell is two
# SECONDS away at 0.9130 as well -- tied to four decimals, so the pick was
# decided by noise. Trimming to percentiles discards the opening and closing
# prints, which are the only thing telling the boards apart.
# the real prints, first and last
first = pd.Timestamp("2026-06-30 04:30:00", tz="UTC")
last = pd.Timestamp("2026-06-30 10:30:00", tz="UTC")
# two candidate boards: one two seconds out, one two minutes out
cands = [
    {"session_id": "08", "open_utc": first + pd.Timedelta(seconds=2),
     "close_utc": last},
    {"session_id": "05", "open_utc": first + pd.Timedelta(seconds=133),
     "close_utc": last - pd.Timedelta(seconds=1)},
]
# the trades those bells are measured against
tr_m = pd.DataFrame({"capture_ts": [first, last], "market": ["REG", "REG"]})
# stand in for the read
_rp, _rr = SC.partition, SC.pd.read_parquet
SC.partition = lambda table, date: types.SimpleNamespace(exists=lambda: True)
SC.pd.read_parquet = lambda *a, **k: tr_m
# the errors
err = SC.board_scores("2026-06-30", cands)
# restore
SC.partition, SC.pd.read_parquet = _rp, _rr
# the two-second board must win, and by a wide margin
check("a bell 2s from the real prints beats one 2 minutes away",
      err.get("08", 9e9) < err.get("05", 0),
      f"got {err}")
check("  and the gap is large enough not to be decided by noise",
      err.get("05", 0) > 10 * max(err.get("08", 1e-9), 1e-9),
      f"2s board scored {err.get('08')}, 2min board {err.get('05')} -- "
      f"the old percentile metric put these within 0.0002 of each other")

# ==========================================================================
# PART 4 -- THE FRIDAY SPLIT
# ==========================================================================
print("\nPART 4 -- THE FRIDAY SPLIT")

# a Friday on the equity board
st = status_frame([("01", friday_equity)])
r = SC.measure("2026-06-05", st, "01")
# two separate stretches of trading
check("a split Friday is two continuous stretches, not one",
      r is not None and r["continuous_spans"] == 2,
      f"got {r['continuous_spans'] if r else None}")
# the lunch break is named by the exchange and excluded from traded time
check("the Jumu'ah break is named", r is not None
      and "FRIDAY_LUNCH_BREAK" in r["break_reasons"],
      f"got {r['break_reasons'] if r else None}")
# 9,900s morning + 7,200s afternoon = 17,100s, against a 26,100s span
check("traded seconds EXCLUDE the break, open-to-close does not",
      r is not None
      and abs(r["traded_seconds"] - 17_100) <= 6
      and abs(r["open_to_close_seconds"] - 26_100) <= 6,
      f"traded {r['traded_seconds'] if r else None}, "
      f"open-to-close {r['open_to_close_seconds'] if r else None}")
# and the exchange's own codes identify the day as a Friday
check("the exchange's own break code marks it a Friday",
      r is not None and r["exchange_says_friday"] is True)

# ==========================================================================
# PART 5 -- A SILENT FEED IS NOT A SHORT SESSION
# ==========================================================================
print("\nPART 5 -- CARRYING STATE ACROSS A SILENCE")


def with_gap(gap_seconds):
    """An ordinary day whose feed stops for `gap_seconds` mid-session."""
    # the full day
    st = status_frame([("01", equity)])
    # the instant the silence starts: an hour into the session
    t0 = st["capture_ts"].iloc[0]
    # the window to delete
    lo = pd.Timestamp("2026-06-30 05:30:00", tz="UTC")
    # its far end
    hi = lo + pd.Timedelta(seconds=gap_seconds)
    # everything outside it
    return st[~st["capture_ts"].between(lo, hi)].reset_index(drop=True)


# a one-minute silence is inside the carry window
short_gap = SC.measure("2026-06-30", with_gap(60), "01")
# a five-minute one is not
long_gap = SC.measure("2026-06-30", with_gap(300), "01")
# the short one is carried, so nothing is reported unobserved
check("a 60s silence is carried and costs no traded seconds",
      short_gap is not None and short_gap["unobserved_seconds"] == 0,
      f"got {short_gap['unobserved_seconds'] if short_gap else None}s "
      f"unobserved")
# the long one is NOT assumed to be trading
check("a 300s silence is NOT assumed to be trading",
      long_gap is not None and long_gap["unobserved_seconds"] > 0,
      "assuming the market stayed open through a five-minute blackout "
      "manufactures session time nobody observed")
# THE STRETCHES MUST SURVIVE AN UNOBSERVED GAP. The run-length arithmetic
# that finds them works on a numpy bool array, and a second whose state was
# never observed is neither True nor False -- converting that without saying
# what to do with the missing values raises, on exactly the dates that matter.
check("the trading stretches are found even with unobserved seconds",
      long_gap is not None and isinstance(long_gap.get("intervals"), list)
      and len(long_gap["intervals"]) >= 1,
      f"got {long_gap.get('intervals') if long_gap else None}")
# and they must still sum to the traded seconds
if long_gap is not None and long_gap.get("intervals"):
    # the sum of the stretches
    _tot = sum((e - s).total_seconds() for s, e in long_gap["intervals"])
    # within one second per stretch, since each is measured inclusively
    check("  and they sum to traded_seconds",
          abs(_tot - long_gap["traded_seconds"])
          <= len(long_gap["intervals"]),
          f"stretches {_tot}s vs traded {long_gap['traded_seconds']}s")
# but the closing bell is unmoved either way, which is what the classifier
# keys off
# THE BALANCE IS EXACT. NO TOLERANCE.
# An earlier version of this check allowed a second per continuous stretch,
# and that slack hid a constant off-by-one: open_to_close_seconds was an
# elapsed-time subtraction while the four parts are counts of seconds on the
# per-second grid, so the parts exceeded the whole by exactly one second on
# every one of the 207 real dates. A clean day printed "traded 15,180s of
# 15,179s". Zero tolerance is what makes this check able to find that.
check("the seconds balance EXACTLY: traded + break + unobserved + other",
      long_gap is not None
      and (long_gap["open_to_close_seconds"]
           == (long_gap["traded_seconds"] + long_gap["break_seconds"]
               + long_gap["unobserved_seconds"]
               + long_gap["other_seconds"])),
      f"open-to-close {long_gap['open_to_close_seconds'] if long_gap else None}"
      f" vs parts "
      f"{(long_gap['traded_seconds'] + long_gap['break_seconds'] + long_gap['unobserved_seconds'] + long_gap['other_seconds']) if long_gap else None}")
# and traded can never exceed the window, which is what the bug looked like
check("  and traded_seconds never exceeds the window",
      long_gap is not None
      and long_gap["traded_seconds"] <= long_gap["open_to_close_seconds"],
      f"traded {long_gap['traded_seconds'] if long_gap else None} > window "
      f"{long_gap['open_to_close_seconds'] if long_gap else None}")
check("and the closing bell is unmoved by either silence",
      short_gap is not None and long_gap is not None
      and str(short_gap["close_pkt"])[11:19]
      == str(long_gap["close_pkt"])[11:19],
      f"{str(short_gap['close_pkt'])[11:19] if short_gap else None} vs "
      f"{str(long_gap['close_pkt'])[11:19] if long_gap else None}")

# ==========================================================================
# PART 6 -- THE CLASSIFIER. Attempt 4's second fault.
# ==========================================================================
print("\nPART 6 -- SHORTENED SESSIONS, ON THE BELL NOT ON TRADED SECONDS")

# four dates: three ordinary, one genuinely short. The date that merely lost
# feed has FEWER traded seconds than the genuinely short one, which is
# exactly why traded seconds cannot be the discriminator.
D = pd.DataFrame([
    dict(date="2026-06-22", weekday="Monday",
         close_pkt=pd.Timestamp("2026-06-22 15:29:59"),
         traded_seconds=20_399, exchange_says_friday=False),
    # 2,000 seconds of lost feed, but the bell rang at the usual time
    dict(date="2026-06-23", weekday="Tuesday",
         close_pkt=pd.Timestamp("2026-06-23 15:30:04"),
         traded_seconds=18_449, exchange_says_friday=False),
    dict(date="2026-06-24", weekday="Wednesday",
         close_pkt=pd.Timestamp("2026-06-24 15:29:59"),
         traded_seconds=20_393, exchange_says_friday=False),
    # a genuine early close, with MORE traded seconds than 06-23
    dict(date="2026-03-11", weekday="Wednesday",
         close_pkt=pd.Timestamp("2026-03-11 14:25:00"),
         traded_seconds=19_000, exchange_says_friday=False),
])
# the classification, exactly as main() performs it
D["is_friday"] = D["exchange_says_friday"] | (D["weekday"] == "Friday")
D["close_sec"] = [SC.seconds_of_day(x) for x in D["close_pkt"]]
D["class_median_close"] = D.groupby("is_friday")["close_sec"] \
                           .transform("median")
D["is_short"] = ((D["class_median_close"] - D["close_sec"])
                 > SC.SHORT_SESSION_MINUTES * 60.0)
# the date with the FEWEST traded seconds must not be called short
check("a date that lost feed is NOT called a shortened session",
      not bool(D.loc[D["date"] == "2026-06-23", "is_short"].iloc[0]),
      "this is how 2026-06-23, a June Tuesday, was labelled a short session")
# THE LABELS SAY WHAT WAS MEASURED, NOT A CAUSE. They were RAMADAN_REGULAR
# and RAMADAN_FRIDAY until the 207-date run caught 2025-09-23 -- a September
# Tuesday that closed at 14:19 -- and labelled it Ramadan.
_types = [("SHORT_FRIDAY" if f else "SHORT_DAY") if s
          else ("REGULAR_FRIDAY" if f else "REGULAR_DAY")
          for f, s in zip(D["is_friday"], D["is_short"])]
check("  and the day types name the measurement, not an inferred cause",
      not any("RAMADAN" in t for t in _types)
      and set(_types) <= {"SHORT_DAY", "SHORT_FRIDAY", "REGULAR_DAY",
                          "REGULAR_FRIDAY"},
      f"got {sorted(set(_types))}")
# and a real early close must be
check("a genuine early close IS called a shortened session",
      bool(D.loc[D["date"] == "2026-03-11", "is_short"].iloc[0]))
# stated plainly, because the ordering is the whole point
check("and it is called short DESPITE having more traded seconds than the "
      "date that is not",
      float(D.loc[D["date"] == "2026-03-11", "traded_seconds"].iloc[0])
      > float(D.loc[D["date"] == "2026-06-23", "traded_seconds"].iloc[0]))

# ---- summary -------------------------------------------------------------
print()
# anything that failed
failed = [n for n, ok in results if not ok]
# the headline
print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
# named, so a failure says what broke
for n in failed:
    print("  FAILED:", n)
# non-zero exit on any failure, so this can gate a script
sys.exit(1 if failed else 0)
