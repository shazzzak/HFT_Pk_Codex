"""
Parse the `raw` column of an existing *_other.parquet into typed tables.
No re-parse of the day file needed — this reads the parquet you already have.

Outputs (written next to the input, same day prefix):
  <day>_session_status.parquet   35=h   market phase timeline (halts, auctions)
  <day>_security_switches.parquet 35=f  per-symbol switches (short-sell etc.)
  <day>_news.parquet             35=B   bulletins (id, headline, payload)
  <day>_rejects.parquet          35=j   business rejects
  <day>_channel_stats.parquet    UA004  per-stream close status, exploded

UA001 / 0 rows are left alone: your file already carries their only fields
(appl_last_seq, end_of_channel, heartbeat_time) — heartbeats have nothing
more to parse.

Field sources: PSX FIX Market Data Interface Specifications (Aug 2024).
Tested against the sample capture for h / B / UA004 / UA001; f and j do not
occur in the sample day, so their parsers follow the spec but are untested
on real data — eyeball the first day that contains them.

Usage:
    python parse_other_messages.py "C:\\...\\parsed\\2026-06-30\\2026-06-30_other.parquet"
or edit OTHER_PARQUET below and run without arguments.
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OTHER_PARQUET = Path(
    r"G:\My Drive\HFT\Capital Stake Day\parsed\2026-06-30\2026-06-30_other.parquet")

SOH = "^"

# ------------------------------- dictionaries ------------------------------
MARKET_CODE_336 = {
    "01": "REG", "02": "BILLS_BONDS", "03": "STOCK_DELIV_FUTURE",
    "04": "STOCK_CASH_SETTLED_FUTURE", "05": "STOCK_OPTION",
    "06": "INDEX_OPTION", "07": "STOCK_INDEX_FUTURE", "08": "ODD_LOT",
    "09": "NEGOTIATED_DEAL", "10": "EQUITIES_SQUARE_UP",
    "12": "FUTURES_SQUARE_UP", "13": "TRADE_RECTIFICATION",
}
PHASE_0 = {
    "S": "STARTING", "O": "OPEN_CALL_AUCTION", "T": "CONTINUOUS",
    "B": "BREAK", "N": "PM_CALL_AUCTION", "C": "CLOSE_CALL_AUCTION",
    "H": "HALT", "A": "AFTER_HOURS", "V": "RESUME_CALL_AUCTION",
    "E": "CLOSED",
}
BREAK_REASON = {"1": "AFTER_PREOPEN", "2": "FRIDAY_LUNCH",
                "3": "AFTER_PM_PREOPEN", "4": "PRE_POSTCLOSE"}
SWITCH_TYPE = {"1": "LEVERAGED_BUY", "2": "SELL_SHORT",
               "3": "BLANK_SELL", "4": "MSF_BUY"}


def _tokens(raw):
    return [p.split("=", 1) for p in raw.rstrip(SOH).split(SOH) if "=" in p]


def _utc(v):
    return pd.to_datetime(v.replace("-", " ", 1) if isinstance(v, str) else v,
                          utc=True, errors="coerce")


def _decode_phase(code):
    code = code or ""
    d0 = code[0] if len(code) > 0 else None
    d2 = code[2] if len(code) > 2 else None
    return (PHASE_0.get(d0), d0 == "H",
            d0 in ("O", "N", "C", "V"),
            BREAK_REASON.get(d2) if d0 == "B" else None)


# ------------------------------ per-type parsers ---------------------------
def parse_h(raw):
    f = dict(_tokens(raw))
    phase, is_halt, is_auction, br = _decode_phase(f.get("8538"))
    return {"orig_time": _utc(f.get("42")), "channel": f.get("10201"),
            "market_code": f.get("336"),
            "market": MARKET_CODE_336.get(f.get("336")),
            "phase_code": f.get("8538"), "phase": phase,
            "is_halt": is_halt, "is_auction": is_auction,
            "break_reason": br}


def parse_f(raw):
    """One input row -> list of rows (one per switch in the 10202 group)."""
    head, switches, cur = {}, [], None
    for tag, val in _tokens(raw):
        if tag == "10203":
            cur = {"switch_type_code": val,
                   "switch_type": SWITCH_TYPE.get(val, f"TYPE_{val}")}
        elif tag == "10204" and cur is not None:
            cur["enabled"] = (val == "Y")
            switches.append(cur)
            cur = None
        else:
            head[tag] = val
    base = {"orig_time": _utc(head.get("42")), "channel": head.get("10201"),
            "symbol": head.get("55")}
    return [{**base, **s} for s in switches] or [base]


def parse_B(raw):
    f = dict(_tokens(raw))
    # RawData (96) may contain '^' / '=': recover it as the span 96=...^10=
    payload = None
    i = raw.find(f"{SOH}96=")
    if i >= 0:
        j = raw.rfind(f"{SOH}10=")
        payload = raw[i + 4: j if j > i else None]
    return {"orig_time": _utc(f.get("42")), "channel": f.get("10201"),
            "news_id": f.get("1472"),
            "is_summary": not f.get("1472"),
            "headline": f.get("148"),
            "raw_data_format": f.get("10209"),
            "raw_data_len": pd.to_numeric(f.get("95"), errors="coerce"),
            "raw_data": payload}


def parse_j(raw):
    f = dict(_tokens(raw))
    return {"ref_seq": pd.to_numeric(f.get("45"), errors="coerce"),
            "ref_msg_type": f.get("372"),
            "reject_ref_id": f.get("379"),
            "reject_reason": f.get("380"), "text": f.get("58")}


def parse_ua004(raw):
    """One input row -> list of rows (one per MDStreamID in the 10208 group)."""
    head, streams, cur = {}, [], None
    for tag, val in _tokens(raw):
        if tag == "1500":
            if cur:
                streams.append(cur)
            cur = {"stream_id": val}
        elif tag == "10207" and cur is not None:
            cur["stock_num"] = pd.to_numeric(val, errors="coerce")
        elif tag == "8538" and cur is not None:
            cur["close_status"] = val
        else:
            head[tag] = val
    if cur:
        streams.append(cur)
    base = {"orig_time": _utc(head.get("42")), "channel": head.get("10201"),
            "n_streams": pd.to_numeric(head.get("10208"), errors="coerce")}
    return [{**base, **s} for s in streams] or [base]


# --------------------------------- driver ----------------------------------
def process(other_parquet: Path):
    other_parquet = Path(other_parquet)
    out_dir = other_parquet.parent
    day = other_parquet.stem.replace("_other", "")
    df = pd.read_parquet(other_parquet, columns=["capture_ts", "sending_time",
                                                 "msg_seq", "msg_type", "raw"])
    print(f"{other_parquet.name}: {len(df):,} rows; msg types:",
          df.msg_type.value_counts().to_dict())

    outputs = {}

    def collect(name, rows):
        if rows:
            outputs[name] = pd.DataFrame(rows)

    passthrough = ["capture_ts", "sending_time", "msg_seq"]
    for name, mtypes, fn, exploded in [
            ("session_status", ("h",), parse_h, False),
            ("security_switches", ("f",), parse_f, True),
            ("news", ("B",), parse_B, False),
            ("rejects", ("j",), parse_j, False),
            ("channel_stats", ("UA004",), parse_ua004, True)]:
        sub = df[df.msg_type.isin(mtypes)]
        rows = []
        for rec in sub.itertuples(index=False):
            base = {c: getattr(rec, c) for c in passthrough}
            parsed = fn(rec.raw)
            if exploded:
                rows.extend({**base, **r} for r in parsed)
            else:
                rows.append({**base, **parsed})
        collect(name, rows)

    for name, out in outputs.items():
        path = out_dir / f"{day}_{name}.parquet"
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False),
                       path, compression="zstd")
        print(f"  {name}: {len(out):,} rows -> {path.name}")
    if "session_status" in outputs:
        s = outputs["session_status"]
        changes = s[s["phase_code"].ne(
            s.groupby("market_code")["phase_code"].shift())]
        print("\nphase CHANGES per market (the timeline that matters):")
        with pd.option_context("display.max_rows", 200, "display.width", 200):
            print(changes[["sending_time", "market", "phase_code", "phase",
                           "is_halt", "is_auction", "break_reason"]]
                  .to_string(index=False))
    return outputs


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else OTHER_PARQUET
    process(src)
