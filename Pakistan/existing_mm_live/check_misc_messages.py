# ============================================================================
# check_misc_messages.py -- what is actually in the misc table?
# ============================================================================
# WHY. The generated schema dump says misc holds 404,086 rows a day across 12
# columns, and that most of those columns are mostly null:
#
#   end_of_channel   100.00% null   -- parser Fix 15 promoted it; it never comes
#   segment           93.55% null
#   orig_time         90.55% null
#   appl_last_seq     64.60% null
#   heartbeat_time    64.60% null
#   msg_type           0.00% null   -- always there
#   raw                0.00% null   -- always there
#
# A column being null on most rows usually means "this row is a different kind
# of message", not "the field is broken". So the useful question is not how
# null each column is, it is WHICH MESSAGE TYPES are in here and which columns
# each one populates. That is what this prints.
#
# It matters because misc is where the TradingSessionStatus messages live --
# the ones carrying TradingPhaseCode every three seconds -- and where news
# messages would be. `raw` is never null, so anything the parser did not
# decode into a column is still recoverable from here without re-parsing the
# original capture.
#
# READ-ONLY. Opens parquet, writes nothing, deletes nothing.
#
# Run from existing_mm_live/:
#   caffeinate -is python check_misc_messages.py
#   caffeinate -is python check_misc_messages.py --date 2026-06-19 --sample 2
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
        "check_misc_messages: could not import PARSED_ROOT from config_pk "
        "(%r). Run from the existing_mm_live/ dir, or add it to sys.path."
        % _e)

# the partition naming the parser uses. NOTE the table is called misc on disk,
# even though the parser's own header comment calls it "other".
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_misc\.parquet$")


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # which day; the most recent by default
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD; default is the most recent partition")
    # how many raw examples to show per message type
    ap.add_argument("--sample", type=int, default=1,
                    help="raw message examples to print per type (0 for none)")
    args = ap.parse_args()

    # every misc partition, filename order = date order
    files = sorted(Path(PARSED_ROOT).rglob("*_misc.parquet"))
    # nothing to read is a clear message, not a stack trace
    if not files:
        raise SystemExit(f"no *_misc.parquet under {PARSED_ROOT}")
    # the requested date, or the most recent day
    if args.date:
        # the partition whose filename carries that date
        want = [f for f in files if f.name.startswith(args.date)]
        # an unknown date is a stop, with the range named
        if not want:
            raise SystemExit(
                f"no misc partition for {args.date}; store covers "
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
    print(f"reading {path.name}")
    # the whole partition: 404k rows is small, and every column is wanted
    df = pd.read_parquet(path)
    print(f"{len(df):,} rows, {len(df.columns)} columns, for {m.group(1)}\n")

    # ---- 1. the message mix ---------------------------------------------
    print("=" * 74)
    print("1. MESSAGE TYPES -- what is in here")
    print("=" * 74)
    # count per type, with its share of the table
    counts = df["msg_type"].value_counts(dropna=False)
    # as a frame so the share can sit beside the count
    M = pd.DataFrame({"rows": counts,
                      "pct": 100.0 * counts / len(df)})
    print(M.to_string(float_format=lambda v: f"{v:,.2f}"))

    # ---- 2. which columns each message type actually populates -----------
    print("\n" + "=" * 74)
    print("2. POPULATED COLUMNS BY MESSAGE TYPE (% non-null)")
    print("=" * 74)
    print("  A column that is 100% null overall may be fully populated on one")
    print("  message type and absent on the rest -- which is normal. A column")
    print("  that is 0% everywhere is genuinely never delivered.")
    # every column except the type itself and the raw text
    cols = [c for c in df.columns if c not in ("msg_type", "raw")]
    # percentage non-null, per type, per column
    F = df.groupby("msg_type")[cols].apply(
        lambda g: 100.0 * g.notna().mean())
    print(F.to_string(float_format=lambda v: f"{v:,.1f}"))

    # ---- 3. columns delivered by NOTHING ---------------------------------
    # a column null on every row of every type is a genuine gap
    dead = [c for c in cols if df[c].notna().sum() == 0]
    # say so plainly, because this is the actionable finding
    print("\n" + "=" * 74)
    print("3. COLUMNS NEVER DELIVERED BY ANY MESSAGE TYPE")
    print("=" * 74)
    # named, or confirmed absent
    if dead:
        # one line each
        for c in dead:
            print(f"  {c}")
        print("\n  These are parsed for but never arrive. If anything depends")
        print("  on one of them, it depends on nothing.")
    else:
        print("  None -- every column is populated by at least one type.")

    # ---- 4. a raw example of each type -----------------------------------
    # skipped entirely when --sample 0
    if args.sample > 0:
        print("\n" + "=" * 74)
        print(f"4. RAW EXAMPLES ({args.sample} per type)")
        print("=" * 74)
        print("  `raw` is never null, so this is the ground truth for what")
        print("  each message carries, including fields the parser did not")
        print("  decode into a column.")
        # each type in descending frequency, so the common ones come first
        for t in counts.index:
            # that type's rows
            g = df[df["msg_type"] == t]
            # a heading with the count
            print(f"\n  --- msg_type={t!r}  ({len(g):,} rows) ---")
            # the first N raw strings, truncated so one message does not
            # fill the terminal
            for raw in g["raw"].head(args.sample):
                # keep it to a readable width
                text = str(raw)
                print(f"    {text[:600]}{'...' if len(text) > 600 else ''}")


# entry point
if __name__ == "__main__":
    main()
