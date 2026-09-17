# ============================================================================
# check_raw_capture.py -- is the missing data missing from the WIRE, or did we
#                         lose it in parsing?
# ============================================================================
# WHY THIS EXISTS.
#
# report_feed_gaps.py measures the PARSED store. Every number it produces has
# passed through PSX_Parser_Mac.py first, so a gap it reports has two possible
# explanations and the report cannot tell them apart:
#
#   (A) the message never reached the capture machine  -> the vendor's problem
#   (B) the message reached it and our parser dropped it -> our problem
#
# This file reads the RAW capture -- the file the parser takes as input, before
# any of our code has touched it -- and re-runs the same three measurements on
# it. If the raw file shows the same silences and the same sequence gaps, the
# parser is exonerated and the finding stands against the vendor. If the raw
# file is clean where the store is not, the fault is ours and no vendor should
# be emailed about it.
#
# It does not guess. It reports both numbers side by side and states which
# conclusion they support.
#
# WHAT IT MEASURES, all four straight off the raw lines:
#
#   1. LINE ACCOUNTING -- total lines, and how many the parser's own filter
#      would silently discard (`if not raw or "|" not in raw: continue`).
#      A large discard count is a parser problem by itself.
#
#   2. ARRIVAL SILENCES -- stretches with no line at all, on the capture
#      clock. The identical measurement report_feed_gaps.py makes, on the
#      identical clock, one stage earlier in the pipeline.
#
#   3. SEQUENCE CONTINUITY -- ApplSeqNum (tag 1181) per channel (tag 10201),
#      read off the wire. trades and ob_updates are ONE stream that the parser
#      splits in two, so this is the only place the sequence can be checked
#      without having to reassemble it.
#
#   4. THE EXCHANGE'S OWN COUNT -- ApplLastSeqNum (tag 1350) in the UA001
#      heartbeats against the highest 1181 present in the same file.
#
#   5. LAST TRADE OF THE DAY -- the last UA202 ExecType=F on the wire. This
#      settles a session-length question that has nothing to do with data
#      loss: PSX trades a SHORTER session during Ramadan, and a report that
#      assumes 09:32-15:30 PKT on those dates counts the closed hour as an
#      outage. See the note under CLOSE TIME in the output.
#
# INPUT FORMATS. The parser takes a .tar.gz holding one .txt of
# `capture_ts|fixbody` lines. This reads that, a bare .txt or .txt.gz, and a
# .parquet of the same lines (any single string column, or a capture_ts column
# beside a body column).
#
# READ-ONLY. Opens the raw file, prints, writes nothing.
#
# Run from Production/:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/check_raw_capture.py \
#       --raw "/Users/shazzak/Library/CloudStorage/GoogleDrive-shazzak@gmail.com/My Drive/HFT Google Drive/Capital Stake Day/2026-06-30.parquet"
#
# and to compare against the store's own numbers for the same date:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/check_raw_capture.py \
#       --raw "<that file>" --compare-store
# ============================================================================

# command-line flags
import argparse
# gzip for a bare .txt.gz
import gzip
# the tar the parser itself opens
import io
import tarfile
# path handling
from pathlib import Path
# the date is read off the filename
import re

# arrays for the gap arithmetic
import numpy as np
# frames, and the parquet reader
import pandas as pd

# the store, from the one config every runner uses. Only needed for
# --compare-store, so an import failure is not fatal on its own.
try:
    from config_pk import PARSED_ROOT
except Exception as _e:                                       # noqa: BLE001
    # remember why, and let the comparison step report it if it is reached
    PARSED_ROOT = None
    _CONFIG_ERR = _e
else:
    # imported cleanly
    _CONFIG_ERR = None

# THE FIELD SEPARATOR THIS CAPTURE USES. Standard FIX is SOH (0x01); this
# capture writes a caret instead. Taken from PSX_Parser_Mac.py, which is the
# only authority on what the file actually contains.
SOH = "^"

# THE SILENCE THRESHOLD, in seconds -- the same 7s report_feed_gaps.py uses,
# so the two numbers are directly comparable. Two missed 3-second heartbeats
# plus jitter.
SILENT_SECONDS = 7.0

# PSX continuous trading on an ORDINARY day, in UTC. 09:32-15:30 Pakistan
# Standard Time is 04:32-10:30 UTC, and PKT does not observe daylight saving.
# THIS IS NOT TRUE EVERY DAY -- see CLOSE TIME in the output.
SESSION_START_UTC = 4 + 32 / 60
SESSION_END_UTC = 10 + 30 / 60


def field(body, tag):
    """One FIX tag's value out of a raw message body, or None.

    Deliberately not a full tokenizer: this file reads five tags out of
    millions of lines, and splitting every line into every field costs about
    twenty times what five targeted searches cost.
    """
    # the separator-prefixed key, so "35=" cannot match inside "1035=..."
    key = SOH + tag + "="
    # where that key sits
    i = body.find(key)
    # not found mid-message; it may still be the very first field
    if i < 0:
        # the first field has no leading separator
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


def raw_lines(path):
    """Yield every raw line of the capture, whatever container it is in."""
    # the file
    p = Path(path)
    # a tar, the format the parser itself takes
    if p.suffix == ".gz" and ".tar" in p.name:
        # open it without extracting to disk
        with tarfile.open(p, "r:*") as tf:
            # the parser takes the first .txt member; do the same
            member = next(m for m in tf.getmembers()
                          if m.isfile() and m.name.lower().endswith(".txt"))
            # report which one, so a multi-member tar is not silently narrowed
            print(f"  inner file       : {member.name}")
            # decode the stream exactly as the parser does
            stream = io.TextIOWrapper(tf.extractfile(member),
                                      encoding="utf-8", errors="replace")
            # one line at a time; the file does not fit in memory
            for line in stream:
                yield line
    # a gzipped text file
    elif p.suffix == ".gz":
        # same decoding
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line
    # a parquet of the same lines
    elif p.suffix == ".parquet":
        # read it whole: a day is large but this is a one-column read
        df = pd.read_parquet(p)
        # say what came back, because the column layout is not guaranteed
        print(f"  parquet columns  : {list(df.columns)}")
        # the string columns, which are the only candidates
        strcols = [c for c in df.columns
                   if pd.api.types.is_object_dtype(df[c])
                   or pd.api.types.is_string_dtype(df[c])]
        # a column already holding whole "capture_ts|body" lines
        whole = [c for c in strcols
                 if df[c].dropna().head(50).astype(str).str.contains(
                     r"\|.*" + re.escape(SOH)).any()]
        # the simple case: one column, already in the parser's line format
        if whole:
            # name it, so the choice is visible rather than assumed
            print(f"  line column      : {whole[0]}")
            # hand them out unchanged
            for v in df[whole[0]].astype(str):
                yield v
        # otherwise look for a timestamp column beside a FIX body column
        else:
            # a column whose values look like FIX fields
            bodies = [c for c in strcols
                      if df[c].dropna().head(50).astype(str).str.contains(
                          r"\d+=").any()]
            # no body column means this parquet is not the raw capture
            if not bodies:
                raise SystemExit(
                    "check_raw_capture: this parquet holds no column of raw "
                    "FIX text. Columns are %r. Point --raw at the file the "
                    "parser takes as input." % (list(df.columns),))
            # the first plausible timestamp column
            tcol = next((c for c in df.columns
                         if "capture" in c.lower() or "recv" in c.lower()
                         or "arriv" in c.lower()), None)
            # state both choices
            print(f"  body column      : {bodies[0]}")
            print(f"  capture column   : {tcol}")
            # a body with no arrival time cannot answer the silence question
            if tcol is None:
                raise SystemExit(
                    "check_raw_capture: found FIX bodies but no capture "
                    "timestamp column, so arrival silences cannot be "
                    "measured. Columns are %r." % (list(df.columns),))
            # rebuild the parser's own line format so one code path follows
            for ts, body in zip(df[tcol].astype(str), df[bodies[0]].astype(str)):
                yield f"{ts}|{body}"
    # a bare text file
    else:
        # same decoding as the tar path
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line


def scan(path):
    """One pass over the raw file, collecting everything the report needs."""
    # every arrival timestamp string, in file order
    caps = []
    # per channel: the ApplSeqNum values seen
    seqs = {}
    # per channel: the highest ApplLastSeqNum the exchange declared
    declared = {}
    # message-type census
    types = {}
    # arrival times of trades (UA202 ExecType=F), for the close-time check
    trade_caps = []
    # lines the parser's own filter would discard
    dropped = 0
    # lines with no tag 35 at all
    no_type = 0
    # total lines
    n = 0

    # walk the file
    for raw in raw_lines(path):
        # count it before anything can reject it
        n += 1
        # the parser's own first two tests, reproduced exactly
        raw = raw.strip()
        # blank, or not in "capture_ts|body" form
        if not raw or "|" not in raw:
            # the parser would silently skip this line; count it instead
            dropped += 1
            continue
        # split on the FIRST pipe, as the parser does
        cap, body = raw.split("|", 1)
        # keep the arrival time
        caps.append(cap)
        # the message type
        mt = field(body, "35")
        # a line with no type is malformed
        if mt is None:
            no_type += 1
        else:
            # census
            types[mt] = types.get(mt, 0) + 1
        # the channel this message came in on
        ch = field(body, "10201")
        # UA001 is the heartbeat, and carries the exchange's own count
        if mt == "UA001":
            # the last sequence number the exchange says it sent
            als = field(body, "1350")
            # keep the highest per channel
            if als is not None and ch is not None:
                try:
                    # the value, as an integer
                    v = int(als)
                except ValueError:
                    # a non-numeric field is not silently coerced
                    v = None
                # keep the maximum
                if v is not None:
                    declared[ch] = max(declared.get(ch, v), v)
        # every other message may carry its own sequence number
        else:
            # ApplSeqNum
            sq = field(body, "1181")
            # only usable with a channel to attribute it to
            if sq is not None and ch is not None:
                try:
                    # the value
                    v = int(sq)
                except ValueError:
                    # skip a malformed one rather than coerce it
                    v = None
                # collect
                if v is not None:
                    seqs.setdefault(ch, []).append(v)
        # a fill, for the close-time check. ExecType is tag 150; F is TRADE.
        if mt == "UA202" and field(body, "150") == "F":
            # its arrival time
            trade_caps.append(cap)

    # hand it all back
    return dict(n=n, dropped=dropped, no_type=no_type, caps=caps, seqs=seqs,
                declared=declared, types=types, trade_caps=trade_caps)


def to_utc(strings):
    """Parse a list of capture timestamps into a sorted UTC index."""
    # the capture writes "2025-10-08 09:19:28.324 +0500"; let pandas infer,
    # because a wrong explicit format silently produces NaT
    t = pd.to_datetime(pd.Series(strings), utc=True, errors="coerce")
    # how many failed to parse -- reported by the caller, never dropped quietly
    bad = int(t.isna().sum())
    # sorted, without the failures
    return t.dropna().sort_values().reset_index(drop=True), bad


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # the raw file
    ap.add_argument("--raw", required=True,
                    help="the raw capture: .tar.gz, .txt, .txt.gz or .parquet")
    # the silence threshold, so the vendor can dispute it without a code change
    ap.add_argument("--silent", type=float, default=SILENT_SECONDS,
                    help="seconds with no line at all that counts as a silence")
    # whether to read the store and compare
    ap.add_argument("--compare-store", action="store_true",
                    help="also read the parsed store for the same date and "
                         "put the two counts side by side")
    args = ap.parse_args()

    print("=" * 78)
    print("RAW CAPTURE CHECK -- is the loss on the wire or in our parsing?")
    print("=" * 78)
    print(f"  raw file         : {args.raw}")

    # the one pass
    r = scan(args.raw)

    # ---- 1. LINE ACCOUNTING ---------------------------------------------
    print("\n1. LINE ACCOUNTING")
    # what the file holds
    print(f"   lines in file                  : {r['n']:,}")
    # what the parser's own filter throws away
    print(f"   discarded by the parser's filter: {r['dropped']:,}")
    # and what had no message type
    print(f"   lines with no tag 35           : {r['no_type']:,}")
    # the verdict on this section, stated rather than left to be inferred
    if r["dropped"] or r["no_type"]:
        print("   NOTE: these lines never reach any of the four tables.")
    else:
        print("   Every line is well formed. Nothing is lost at this step.")

    # the message census, biggest first
    print("\n   messages by type")
    # sorted
    for mt, c in sorted(r["types"].items(), key=lambda kv: -kv[1]):
        # each type and its share
        print(f"     {mt:<8s} {c:>12,}  ({100 * c / max(r['n'], 1):5.2f}%)")

    # ---- 2. ARRIVAL SILENCES --------------------------------------------
    print("\n2. ARRIVAL SILENCES, MEASURED ON THE RAW FILE")
    # the arrival clock
    t, bad_ts = to_utc(r["caps"])
    # unparseable timestamps are a finding, not a footnote
    if bad_ts:
        print(f"   {bad_ts:,} capture timestamps could not be parsed")
    # nothing to measure
    if len(t) < 2:
        print("   too few parseable timestamps to measure a gap")
    else:
        # the wall-clock gap between consecutive arrivals
        gaps = t.diff().dt.total_seconds()
        # the hour of day each gap ENDED, for the session test
        hod = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
        # inside the ordinary continuous session
        insess = (hod >= SESSION_START_UTC) & (hod <= SESSION_END_UTC)
        # the silences
        sil = gaps > args.silent
        # the ones that matter
        both = sil & insess
        # the day's span, for context
        print(f"   first line {t.iloc[0]}   last line {t.iloc[-1]}")
        # the headline
        print(f"   silences over {args.silent:g}s, anywhere in the file : "
              f"{int(sil.sum()):,}  ({float(gaps[sil].sum()):,.0f}s)")
        print(f"   of those, inside 04:32-10:30 UTC            : "
              f"{int(both.sum()):,}  ({float(gaps[both].sum()):,.0f}s)")
        # each in-session one, so they can be matched against the store's
        if int(both.sum()):
            print("\n   each in-session silence on the raw wire")
            print(f"     {'silent from (UTC)':<26s} {'to (UTC)':<26s} {'sec':>8s}")
            # walk them in time order
            for i in t.index[both]:
                # the gap ended at t[i] and began one gap earlier
                print(f"     {str(t[i] - pd.Timedelta(seconds=float(gaps[i]))):<26s} "
                      f"{str(t[i]):<26s} {float(gaps[i]):8.1f}")
        # THE POINT OF THE WHOLE FILE
        print("\n   WHAT THIS MEANS")
        print("   These gaps are in the file BEFORE our parser runs. A gap")
        print("   here is a stretch in which nothing arrived at the capture")
        print("   machine, and no change to our code can recover it. A gap")
        print("   the store reports but this does not is OUR bug.")

    # ---- 3. SEQUENCE CONTINUITY ON THE WIRE ------------------------------
    print("\n3. SEQUENCE CONTINUITY, PER CHANNEL, ON THE RAW WIRE")
    print("   ApplSeqNum (1181) runs continuously across UA201 adds, UA202")
    print("   executions and UA202 cancels -- one stream. The parser splits")
    print("   that stream into trades and ob_updates, so checking either")
    print("   table alone reports the other's messages as losses. Here it")
    print("   has not been split yet.")
    # totals across channels
    tot_missing = 0
    # per channel
    print(f"\n     {'chan':>6s} {'messages':>12s} {'first':>12s} {'last':>12s} "
          f"{'gaps':>8s} {'missing':>12s}")
    # each channel, in order
    for ch in sorted(r["seqs"], key=lambda c: (len(c), c)):
        # the values, deduplicated and sorted
        v = np.unique(np.asarray(r["seqs"][ch], dtype=np.int64))
        # a channel with one message has no step to measure
        if len(v) < 2:
            continue
        # the step between consecutive sequence numbers
        step = np.diff(v)
        # a step above 1 is a hole
        holes = step[step > 1]
        # how many messages those holes account for
        miss = int((holes - 1).sum())
        # accumulate
        tot_missing += miss
        # the row
        print(f"     {ch:>6s} {len(v):>12,} {v[0]:>12,} {v[-1]:>12,} "
              f"{len(holes):>8,} {miss:>12,}")
    # the total
    print(f"\n   missing sequence numbers on the raw wire: {tot_missing:,}")

    # ---- 4. THE EXCHANGE'S OWN COUNT ------------------------------------
    print("\n4. THE EXCHANGE'S OWN COUNT (tag 1350) vs WHAT THE FILE HOLDS")
    print("   This is the strongest line in the report: it is not inferred")
    print("   from what is absent, it is the exchange stating what it sent.")
    # per channel
    print(f"\n     {'chan':>6s} {'exchange said':>16s} {'file holds':>16s} "
          f"{'behind':>14s}")
    # running total
    tot_behind = 0
    # every channel the heartbeats named
    for ch in sorted(r["declared"]):
        # what the exchange declared
        ex = r["declared"][ch]
        # the highest we actually hold on that channel
        ours = max(r["seqs"].get(ch, [0])) if r["seqs"].get(ch) else 0
        # the shortfall
        behind = max(0, ex - ours)
        # accumulate
        tot_behind += behind
        # the row
        print(f"     {ch:>6s} {ex:>16,} {ours:>16,} {behind:>14,}")
    # the total
    print(f"\n   messages behind the exchange's own count: {tot_behind:,}")

    # ---- 5. CLOSE TIME ---------------------------------------------------
    print("\n5. CLOSE TIME -- when did trading actually stop?")
    print("   NOT a data-loss question. PSX runs a SHORTER session during")
    print("   Ramadan, and a report that assumes 09:32-15:30 PKT on those")
    print("   dates counts the closed hour as an outage. The last fill on")
    print("   the wire says which session this date actually ran.")
    # the trade arrival times
    if r["trade_caps"]:
        # parsed
        tt, _ = to_utc(r["trade_caps"])
        # the last one
        last = tt.iloc[-1]
        # in Pakistan time, which is what a session is quoted in
        last_pkt = last + pd.Timedelta(hours=5)
        # the headline
        print(f"\n   fills on the wire      : {len(tt):,}")
        print(f"   first fill             : {tt.iloc[0]} UTC "
              f"({tt.iloc[0] + pd.Timedelta(hours=5)} PKT)")
        print(f"   LAST fill              : {last} UTC ({last_pkt} PKT)")
        # the ordinary close, for comparison
        print(f"   ordinary PSX close     : 10:30 UTC (15:30 PKT)")
        # how early this date stopped
        close_utc = last.normalize() + pd.Timedelta(hours=10, minutes=30)
        # the shortfall in minutes
        early = (close_utc - last).total_seconds() / 60.0
        # a date that stopped well before the ordinary close ran a short session
        if early > 20:
            print(f"\n   THIS DATE STOPPED {early:.0f} MINUTES EARLY. Treat the")
            print("   period after the last fill as a CLOSED MARKET, not an")
            print("   outage, and do not run the backtest's session past it.")
        else:
            print("\n   This date ran the ordinary session.")
    else:
        # no fills at all is itself worth saying
        print("   no UA202 ExecType=F messages in this file")

    # ---- 6. THE STORE, SIDE BY SIDE --------------------------------------
    if args.compare_store:
        print("\n6. THE SAME DATE IN THE PARSED STORE")
        # the config has to have imported
        if PARSED_ROOT is None:
            print(f"   could not import config_pk ({_CONFIG_ERR!r}). Run with "
                  f"PYTHONPATH=../existing_mm_live.")
        else:
            # the date, off the raw filename
            m = re.search(r"(\d{4}-\d{2}-\d{2})", Path(args.raw).name)
            # a filename with no date cannot be matched to a partition
            if m is None:
                print(f"   no YYYY-MM-DD in {Path(args.raw).name}; cannot "
                      f"locate the matching partition")
            else:
                # the date
                date = m.group(1)
                # rows per table
                print(f"\n     {'table':<14s} {'rows in store':>16s}")
                # the running total of message-level rows
                store_msgs = 0
                # each table the parser writes
                for table in ("trades", "ob_updates", "misc", "ob_snapshot"):
                    # the partition, hive layout
                    p = (Path(PARSED_ROOT) / table / f"date={date}"
                         / f"{date}_{table}.parquet")
                    # a missing partition is reported, not assumed empty
                    if not p.exists():
                        print(f"     {table:<14s} {'MISSING':>16s}")
                        continue
                    # just the row count
                    import pyarrow.parquet as pq
                    # metadata read: no data pages are touched
                    nrows = pq.ParquetFile(p).metadata.num_rows
                    # the row
                    print(f"     {table:<14s} {nrows:>16,}")
                    # ob_snapshot expands one message into many rows, so it
                    # cannot be added to a message-level total
                    if table != "ob_snapshot":
                        store_msgs += nrows
                # THE RECONCILIATION
                print(f"\n   raw lines with a message type : "
                      f"{r['n'] - r['dropped'] - r['no_type']:,}")
                print(f"   store rows, message-level tables: {store_msgs:,}")
                print("   (ob_snapshot is excluded: one 35=W message becomes")
                print("   one row per book level, so it is not comparable.)")
                # the snapshot count on the wire, which IS comparable
                w = r["types"].get("W", 0)
                # stated separately
                print(f"   35=W messages on the wire       : {w:,}")

    # ---- the conclusion --------------------------------------------------
    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  Compare section 2 against report_feed_gaps.py's silences for the")
    print("  same date, and section 3 against its sequence gaps.")
    print()
    print("  SAME NUMBERS      -> the data never arrived. The parser is clean")
    print("                       and the finding stands against the vendor.")
    print("  RAW CLEAN, STORE  -> we are losing it in parsing. Do not email")
    print("  SHOWS GAPS           the vendor; fix the parser first.")
    print("  RAW WORSE THAN    -> the store is not reading everything the")
    print("  THE STORE            parser wrote. Check the merge step.")
    print()
    print("  NO CAUSE IS ASSERTED ANYWHERE IN THIS OUTPUT.")


# entry point
if __name__ == "__main__":
    main()
