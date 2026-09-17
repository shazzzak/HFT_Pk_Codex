# ============================================================================
# report_feed_gaps.py -- every gap in the captured feed, for the data vendor
# ============================================================================
# WHAT THIS IS FOR. Producing a factual record of where the captured PSX feed
# is incomplete, across every trading date in the store, in a form a vendor can
# check against their own logs.
#
# IT MAKES NO CLAIM ABOUT CAUSE. It reports three measured things and leaves
# the explanation to whoever ran the capture:
#
#   1. FEED SILENCE -- stretches of wall-clock time in which NO message of any
#      kind arrived. The PSX feed heartbeats every 3 seconds per channel
#      (UA001, ApplLastSeqNum in tag 1350), so if the receiver is connected,
#      something arrives at least that often. A longer stretch with nothing at
#      all means nothing was being received. Measured on capture_ts -- our
#      arrival clock -- because this is a question about the receiver, not
#      about what the exchange sent.
#
#   2. SEQUENCE GAPS -- missing ApplSeqNum values within a channel. The
#      exchange numbers its messages per channel specifically so a consumer
#      can detect loss. A gap means a message the exchange sent did not reach
#      the store.
#
#   3. THE EXCHANGE'S OWN COUNT -- the heartbeat states the last sequence
#      number sent on that channel. Comparing it against the highest we hold
#      is the strongest evidence in this report, because it is the exchange's
#      own declaration rather than anything inferred from what is missing.
#
# THE TWO ARE DIFFERENT AND BOTH ARE REPORTED. A sequence gap with no
# corresponding silence means messages were lost while the connection was up.
# A silence with a matching gap means the receiver stopped. Which of those it
# is matters to the vendor, so the report does not merge them.
#
# WHY ob_snapshot IS NOT READ FOR LIVENESS. It is ~33 million rows a day and
# adds nothing: the heartbeats in misc already establish liveness every three
# seconds. Reading it would multiply the runtime for no extra information.
#
# READ-ONLY. Writes two timestamped CSVs and prints a summary that can be
# pasted into an email. Never overwrites.
#
# Run from Production/:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/report_feed_gaps.py
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/report_feed_gaps.py --days 20
# ============================================================================

# command-line flags
import argparse
# timestamped output names
import datetime as dt
# path handling
from pathlib import Path
# the date comes out of the filename
import re

# frames
import pandas as pd

# the store and the results directory, from the one config every runner uses
try:
    from config_pk import PARSED_ROOT, RESULTS_ROOT
except Exception as _e:                                       # noqa: BLE001
    raise ImportError(
        "report_feed_gaps: could not import paths from config_pk (%r). Run "
        "with PYTHONPATH=../existing_mm_live." % _e)

# THE SILENCE THRESHOLD, in seconds.
#
# The feed heartbeats every 3 seconds per channel, so two missed intervals is
# the conventional disconnect test. 7 seconds allows for scheduling jitter on
# the receiver without letting a genuine stall through. Raise it with --silent
# if the vendor disputes the threshold; the raw gaps are in the detail CSV
# either way, so the conclusion does not depend on where the line is drawn.
SILENT_SECONDS = 7.0

# PSX continuous trading, in UTC. 09:32-15:30 Pakistan Standard Time is
# 04:32-10:30 UTC, and PKT does not observe daylight saving.
SESSION_START_UTC = 4 + 32 / 60
SESSION_END_UTC = 10 + 30 / 60
# how long that session is, in seconds -- the denominator for every share
SESSION_SECONDS = int((SESSION_END_UTC - SESSION_START_UTC) * 3600)

# the four tables the parser writes, and whether each is read for liveness
TABLES = {"trades": True, "ob_updates": True, "misc": True,
          "ob_snapshot": False}


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
        # the date, or skip a file we cannot attribute
        m = rx.match(f.name)
        # named unexpectedly: report rather than guess
        if m is None:
            print(f"  skipping unexpected filename {f.name}")
            continue
        # keep it
        out.append(m.group(1))
    # oldest first
    return out


def feed_silences(date):
    """Stretches in which NO message of any kind arrived.

    Returns one row per silence, with the channels that were carrying traffic
    either side of it -- which is what tells the vendor whether a single
    channel stopped or the whole connection did.
    """
    # every arrival time we hold, with the channel it came in on
    parts = []
    # only the tables that establish liveness; ob_snapshot is skipped on
    # purpose (see the header)
    for table, use in TABLES.items():
        # not a liveness table
        if not use:
            continue
        # where it lives
        path = partition(table, date)
        # a missing table contributes nothing rather than failing the date
        if not path.exists():
            continue
        # arrival clock and channel only -- two columns keeps this affordable
        try:
            df = pd.read_parquet(path, columns=["capture_ts", "channel"])
        # a table without those columns is reported and skipped
        except Exception as exc:                              # noqa: BLE001
            print(f"  {date} {table}: could not read ({exc!r})")
            continue
        # keep the rows that have an arrival time
        parts.append(df.dropna(subset=["capture_ts"]))
    # nothing readable for this date
    if not parts:
        return pd.DataFrame(), 0
    # one stream, in arrival order
    t = (pd.concat(parts, ignore_index=True)
         .sort_values("capture_ts").reset_index(drop=True))
    # how many messages this rests on, for the record
    n_msgs = len(t)
    # the wall-clock gap between consecutive arrivals
    gap = t["capture_ts"].diff().dt.total_seconds()
    # the stretches longer than the threshold
    idx = gap[gap > SILENT_SECONDS].index
    # a clean day
    if len(idx) == 0:
        return pd.DataFrame(), n_msgs
    # one row per silence, with what was arriving either side
    rows = pd.DataFrame({
        # when the last message before the silence arrived
        "silent_from_utc": t["capture_ts"].iloc[idx - 1].values,
        # and when traffic resumed
        "silent_to_utc": t["capture_ts"].iloc[idx].values,
        # how long nothing arrived
        "seconds": gap.iloc[idx].values,
        # the channel the last message came in on
        "channel_before": t["channel"].iloc[idx - 1].values,
        # and the first one after
        "channel_after": t["channel"].iloc[idx].values})
    # the hour of day the silence began, UTC
    hh = (rows["silent_from_utc"].dt.hour
          + rows["silent_from_utc"].dt.minute / 60.0)
    # whether it fell inside continuous trading, which is the only part that
    # affects anything
    rows["in_continuous_session"] = ((hh >= SESSION_START_UTC)
                                     & (hh <= SESSION_END_UTC))
    # stamped with the date
    rows.insert(0, "date", date)
    # and labelled, since the detail file holds two kinds of finding
    rows.insert(1, "finding", "feed_silence")
    # done
    return rows, n_msgs


def sequence_gaps(date):
    """Missing ApplSeqNum values, per channel, with the times either side.

    trades and ob_updates are ONE message stream that the parser splits in
    two: UA202 executions to trades, UA201 adds and UA202 cancels to
    ob_updates. ApplSeqNum runs continuously across both, so they have to be
    put back together before a gap means anything. Checking either alone
    reports the other's messages as losses.
    """
    # both halves of the stream
    parts = []
    # the two tables that carry appl_seq
    for table in ("trades", "ob_updates"):
        # where it lives
        path = partition(table, date)
        # a missing table is skipped
        if not path.exists():
            continue
        # sequence, channel, and both clocks
        try:
            parts.append(pd.read_parquet(
                path, columns=["appl_seq", "channel", "transact_time",
                               "capture_ts"]))
        # reported and skipped rather than fatal
        except Exception as exc:                              # noqa: BLE001
            print(f"  {date} {table}: could not read ({exc!r})")
    # nothing readable
    if not parts:
        return pd.DataFrame()
    # one stream, which is what it is
    d = pd.concat(parts, ignore_index=True).dropna(subset=["appl_seq"])
    # nothing to check
    if d.empty:
        return pd.DataFrame()
    # one record per break, per channel
    out = []
    # each channel has its own sequence space
    for ch, g in d.groupby("channel"):
        # sequence and time together, in sequence order, deduplicated
        gg = (g[["appl_seq", "transact_time", "capture_ts"]]
              .astype({"appl_seq": "int64"})
              .sort_values("appl_seq").drop_duplicates("appl_seq"))
        # nothing to compare
        if len(gg) < 2:
            continue
        # the step from one sequence number to the next
        step = gg["appl_seq"].diff()
        # the rows that follow a break
        after = gg[step > 1]
        # and those that precede one
        before = gg.shift(1)[step > 1]
        # how many numbers are missing in each
        missing = (step[step > 1] - 1).astype("int64")
        # one row per break
        for (_, a), (_, b), n in zip(after.iterrows(), before.iterrows(),
                                     missing):
            out.append({
                "date": date, "finding": "sequence_gap", "channel": ch,
                # the sequence numbers either side of the hole
                "last_seq_held": int(b["appl_seq"]),
                "next_seq_held": int(a["appl_seq"]),
                # how many the exchange numbered that we do not have
                "missing_messages": int(n),
                # and when, by the exchange's clock
                "last_seen_utc": b["transact_time"],
                "resumed_utc": a["transact_time"]})
    # nothing missing on any channel
    if not out:
        return pd.DataFrame()
    # as a frame
    rows = pd.DataFrame(out)
    # the hour the break began, UTC
    hh = rows["last_seen_utc"].dt.hour + rows["last_seen_utc"].dt.minute / 60.0
    # whether it fell inside continuous trading
    rows["in_continuous_session"] = ((hh >= SESSION_START_UTC)
                                     & (hh <= SESSION_END_UTC))
    # done
    return rows


def exchange_shortfall(date):
    """The exchange's own ApplLastSeqNum against the highest we hold.

    THE STRONGEST EVIDENCE IN THIS REPORT, because it is the exchange stating
    what it sent rather than us inferring what is absent. The UA001 heartbeat
    carries ApplLastSeqNum in tag 1350; the parser promotes it to
    appl_last_seq.

    A positive shortfall means the exchange numbered messages beyond the last
    one that reached the store.
    """
    # the heartbeats
    misc = partition("misc", date)
    # without them there is nothing to compare against
    if not misc.exists():
        return pd.DataFrame()
    # their declaration of the last sequence sent, per channel
    try:
        hb = pd.read_parquet(misc, columns=["channel", "appl_last_seq"])
    # reported and skipped
    except Exception as exc:                                  # noqa: BLE001
        print(f"  {date} misc: could not read ({exc!r})")
        return pd.DataFrame()
    # only the rows that carry it
    hb = hb.dropna(subset=["appl_last_seq"])
    # none did
    if hb.empty:
        return pd.DataFrame()
    # the exchange's high-water mark per channel
    theirs = hb.groupby("channel")["appl_last_seq"].max()
    # and ours, from the message stream
    parts = []
    # both halves again
    for table in ("trades", "ob_updates"):
        # where it lives
        path = partition(table, date)
        # skip a missing table
        if not path.exists():
            continue
        # sequence and channel only
        parts.append(pd.read_parquet(path, columns=["appl_seq", "channel"]))
    # nothing to compare
    if not parts:
        return pd.DataFrame()
    # our high-water mark per channel
    ours = (pd.concat(parts, ignore_index=True).dropna(subset=["appl_seq"])
            .groupby("channel")["appl_seq"].max())
    # only channels both know about
    common = theirs.index.intersection(ours.index)
    # nothing in common means the two count different spaces
    if len(common) == 0:
        return pd.DataFrame()
    # one row per channel
    return pd.DataFrame({
        "date": date, "finding": "exchange_shortfall",
        "channel": common,
        # what the exchange says it sent
        "exchange_last_seq": theirs[common].astype("int64").values,
        # what we hold
        "our_last_seq": ours[common].astype("int64").values,
        # the difference, which is the number that matters
        "messages_behind": (theirs[common] - ours[common]).astype("int64").values,
    }).reset_index(drop=True)


def main():
    # rebinding the module-level threshold, so the declaration comes first
    global SILENT_SECONDS
    # the command line
    ap = argparse.ArgumentParser()
    # limit the scan while testing; the default is the whole store
    ap.add_argument("--days", type=int, default=0,
                    help="scan only the most recent N dates (0 = every date)")
    # the vendor may dispute the threshold, so it is adjustable
    ap.add_argument("--silent", type=float, default=SILENT_SECONDS,
                    help="seconds with no message at all before it counts as "
                         "a silence (default 7; the feed heartbeats every 3)")
    args = ap.parse_args()
    # apply the override
    SILENT_SECONDS = args.silent

    # every date, or the most recent N
    dates = all_dates()
    # narrowed if asked
    if args.days:
        dates = dates[-args.days:]

    print("=" * 78)
    print("PSX FEED GAP REPORT")
    print("=" * 78)
    print(f"  store            : {PARSED_ROOT}")
    print(f"  dates            : {len(dates)}  "
          f"({dates[0]} .. {dates[-1]})" if dates else "  no dates found")
    print(f"  silence threshold: {SILENT_SECONDS:.0f}s with NO message of any "
          f"kind")
    print(f"                     (the feed heartbeats every 3s per channel)")
    print(f"  session          : 04:32-10:30 UTC = 09:32-15:30 PKT, "
          f"{SESSION_SECONDS:,}s\n")
    # nothing to do
    if not dates:
        raise SystemExit("no trades partitions found under the store")

    # every finding, one row each
    detail = []
    # one summary row per date
    summary = []
    # walk the calendar
    for i, date in enumerate(dates, 1):
        # the three measurements
        sil, n_msgs = feed_silences(date)
        seq = sequence_gaps(date)
        short = exchange_shortfall(date)
        # keep whatever came back
        for frame in (sil, seq, short):
            # a clean day contributes nothing
            if len(frame):
                detail.append(frame)
        # the silences that fell inside trading, which are the ones that count
        sil_in = sil[sil["in_continuous_session"]] if len(sil) else sil
        # and the sequence gaps likewise
        seq_in = seq[seq["in_continuous_session"]] if len(seq) else seq
        # one row for this date
        summary.append({
            "date": date,
            # how many messages the liveness measure rests on
            "messages_examined": n_msgs,
            # silences during trading
            "silences_in_session": len(sil_in),
            "silent_seconds_in_session": float(sil_in["seconds"].sum())
            if len(sil_in) else 0.0,
            "longest_silence_seconds": float(sil_in["seconds"].max())
            if len(sil_in) else 0.0,
            # as a share of the session, which is the headline
            "pct_of_session_silent": (100.0 * sil_in["seconds"].sum()
                                      / SESSION_SECONDS) if len(sil_in) else 0.0,
            # sequence gaps during trading
            "sequence_gaps_in_session": len(seq_in),
            "missing_messages_in_session": int(seq_in["missing_messages"].sum())
            if len(seq_in) else 0,
            # and the exchange's own count
            "messages_behind_exchange": int(short["messages_behind"].clip(
                lower=0).sum()) if len(short) else 0,
        })
        # progress, because a full store is 200-odd dates
        print(f"  [{i}/{len(dates)}] {date}  "
              f"{len(sil_in)} silence(s) in session, "
              f"{summary[-1]['silent_seconds_in_session']:,.0f}s", flush=True)

    # as frames
    S = pd.DataFrame(summary)
    # the detail, with a consistent column order across the three finding types
    D = (pd.concat(detail, ignore_index=True) if detail else pd.DataFrame())

    # ---- the summary a vendor can read ----------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    # the totals that matter
    tot_sil = S["silent_seconds_in_session"].sum()
    # the session time across the whole sample
    tot_session = SESSION_SECONDS * len(S)
    print(f"  dates examined            : {len(S)}")
    print(f"  messages examined         : {S['messages_examined'].sum():,}")
    print(f"  silences during trading   : {S['silences_in_session'].sum():,}")
    print(f"  seconds silent in session : {tot_sil:,.0f} of {tot_session:,} "
          f"({100.0 * tot_sil / tot_session:.2f}%)")
    print(f"  longest single silence    : "
          f"{S['longest_silence_seconds'].max():,.0f}s")
    print(f"  missing sequence numbers  : "
          f"{S['missing_messages_in_session'].sum():,} during trading")
    print(f"  behind the exchange's own count: "
          f"{S['messages_behind_exchange'].sum():,}")
    # the worst days, which is what a vendor will want to look up first
    if len(S):
        print("\n  WORST DATES BY TIME SILENT DURING TRADING")
        worst = S.sort_values("silent_seconds_in_session",
                              ascending=False).head(10)
        print(worst[["date", "silences_in_session",
                     "silent_seconds_in_session", "longest_silence_seconds",
                     "pct_of_session_silent"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    print("\n  WHAT THESE NUMBERS ARE")
    print("  * A SILENCE is a stretch in which no message of any kind reached")
    print("    the capture -- no trade, no book update, no heartbeat. The PSX")
    print("    feed heartbeats every 3 seconds per channel, so a stretch")
    print(f"    longer than {SILENT_SECONDS:.0f}s means nothing was arriving.")
    print("    Measured on the capture's own arrival timestamps.")
    print("  * A SEQUENCE GAP is a missing ApplSeqNum within a channel. The")
    print("    exchange numbers messages per channel so a consumer can detect")
    print("    loss; a gap is a message that did not reach the store.")
    print("  * BEHIND THE EXCHANGE'S COUNT compares the last sequence number")
    print("    the exchange itself declared in its heartbeat (tag 1350,")
    print("    ApplLastSeqNum) against the highest we hold. This one is not")
    print("    inferred from what is absent -- it is the exchange's own")
    print("    statement of what it sent.")
    print("  * NO CAUSE IS ASSERTED anywhere in this report.")

    # ---- write ----------------------------------------------------------
    # a timestamp for both files
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    # the per-date summary
    out_s = Path(RESULTS_ROOT) / f"feed_gaps_summary_{stamp}.csv"
    # and every individual finding
    out_d = Path(RESULTS_ROOT) / f"feed_gaps_detail_{stamp}.csv"
    # refuse rather than clobber
    for p in (out_s, out_d):
        # a collision is vanishingly unlikely but never silently overwritten
        if p.exists():
            raise SystemExit(f"{p} already exists; refusing to overwrite")
    # write them
    S.to_csv(out_s, index=False)
    D.to_csv(out_d, index=False)
    print(f"\nwrote {out_s}")
    print(f"wrote {out_d}")
    print("\n  Send both. The summary is one row per trading date; the detail")
    print("  is one row per individual gap, with the timestamps either side,")
    print("  so each one can be checked against the vendor's own logs.")


# entry point
if __name__ == "__main__":
    main()
