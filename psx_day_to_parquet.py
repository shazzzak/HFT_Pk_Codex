# %% [markdown]
# # PSX FIX daily capture -> 3 Parquet files (chunked, bounded RAM)
#
# Workflow, exactly as specified:
#   1. Stream a chunk of lines straight out of the .tar.gz (never extracted).
#   2. Parse it with `parse_fix_file` (same function as the original parser,
#      input generalised from a path to any iterable of lines — parsing logic
#      is byte-for-byte identical).
#   3. Write 3 partial Parquet files: ticks / orderbook / other.
#   4. Free the memory (del + gc.collect()).
#   5. Repeat until the day is done.
#   6. Combine the partials into 3 final files (streaming, row-group by
#      row-group — the full day is never in RAM) and delete the partials.
#
# Final output in OUT_DIR:
#   2026-06-30_ticks.parquet      (UA201 adds + UA202 trades/cancels,
#                                  incl. buyer/seller-initiated flag)
#   2026-06-30_orderbook.parquet  (35=W snapshots, one row per book entry)
#   2026-06-30_other.parquet      (h / B / UA001 / UA004 / heartbeats, raw)

# %%
import gc
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------- configuration ------------------------------
SRC = Path(r"C:\Users\shahz\OneDrive\Desktop\Del\Capital Stake\2026-06-30.tar.gz")
OUT_DIR = SRC.parent / "parsed" / SRC.name.replace(".tar.gz", "")
CHUNK_LINES = 250_000            # ~250 MB of text per chunk; lower if RAM-tight

SOH = "^"

SIDE_MAP = {"1": "BUY", "2": "SELL", "5": "SELL_SHORT", "8": "CROSS",
            "G": "LEVERAGED_BUY"}
EXECTYPE_MAP = {"0": "NEW", "1": "PARTIAL_FILL", "2": "FILL", "4": "CANCELED",
                "5": "REPLACED", "6": "PENDING_CANCEL", "8": "REJECTED",
                "9": "SUSPENDED", "A": "PENDING_NEW", "E": "PENDING_REPLACE",
                "F": "TRADE"}
MARKET_SEGMENT_MAP = {"010": "REG_SNAPSHOT", "011": "REG",
                      "020": "ODL_SNAPSHOT_?", "030": "FUT_SNAPSHOT",
                      "031": "FUT", "070": "SEG_070", "080": "SEG_080",
                      "081": "SEG_081", "100": "SEG_100", "900": "INDEX"}
MDENTRY_MAP = {"0": "BID", "1": "OFFER", "2": "LAST_TRADE", "3": "INDEX_VALUE",
               "4": "OPENING_PRICE", "7": "SESSION_HIGH", "8": "SESSION_LOW",
               "x1": "NET_CHANGE", "x2": "CUSTOM_x2",
               "x3": "AGG_BID (VWAP px / total qty)",
               "x4": "AGG_OFFER (VWAP px / total qty)",
               "x7": "CUSTOM_x7", "x8": "CUSTOM_x8", "xa": "INDEX_OPEN",
               "xb": "INDEX_HIGH", "xc": "INDEX_LOW_?", "xd": "INDEX_LOW/PREV_?",
               "xe": "UPPER_CIRCUIT_BREAKER", "xf": "LOWER_CIRCUIT_BREAKER",
               "xg": "CUSTOM_xg (futures only, likely open interest)"}


# ----------------------- parsing (identical logic) ------------------------
def _tokenize(body):
    return [p.split("=", 1) for p in body.rstrip(SOH).split(SOH) if "=" in p]


def _parse_snapshot_entries(pairs):
    header, entries, cur = {}, [], None
    it = iter(pairs)
    for tag, val in it:
        if tag == "268":
            header["268"] = val
            break
        header[tag] = val
    for tag, val in it:
        if tag == "269":
            if cur is not None:
                entries.append(cur)
            cur = {"269": val, "orders": []}
        elif tag == "10":
            break
        elif cur is None:
            header[tag] = val
        elif tag == "38":
            cur["orders"].append({"qty": val})
        elif tag == "37":
            if cur["orders"] and "id" not in cur["orders"][-1]:
                cur["orders"][-1]["id"] = val
            else:
                cur["orders"].append({"id": val})
        else:
            cur[tag] = val
    if cur is not None:
        entries.append(cur)
    return header, entries


def parse_fix_file(lines):
    """
    The original parse_fix_file, with the input generalised: takes any
    iterable of raw lines (a chunk) instead of a path.
    Returns (tick_records, book_records, other_records).
    """
    ticks, book, other = [], [], []
    for raw in lines:
        raw = raw.strip()
        if not raw or "|" not in raw:
            continue
        capture_ts, body = raw.split("|", 1)
        pairs = _tokenize(body)
        msg = dict(pairs)
        mtype = msg.get("35")
        base = {"capture_ts": capture_ts, "msg_seq": msg.get("34"),
                "sending_time": msg.get("52"), "msg_type": mtype,
                "channel": msg.get("10201"), "segment": msg.get("1500")}
        if mtype == "UA201":
            ticks.append({**base, "event": "ORDER_ADD",
                          "appl_seq": msg.get("1181"), "symbol": msg.get("55"),
                          "side_code": msg.get("54"), "price": msg.get("44"),
                          "qty": msg.get("38"), "order_id": msg.get("37"),
                          "transact_time": msg.get("60"),
                          "exec_type_code": None,
                          "buy_ref": None, "sell_ref": None})
        elif mtype == "UA202":
            et = msg.get("150")
            ticks.append({**base,
                          "event": "TRADE" if et == "F"
                                   else ("CANCEL" if et == "4" else f"EXEC_{et}"),
                          "appl_seq": msg.get("1181"), "symbol": msg.get("55"),
                          "side_code": None, "price": msg.get("31"),
                          "qty": msg.get("32"), "order_id": None,
                          "transact_time": msg.get("60"),
                          "exec_type_code": et,
                          "buy_ref": msg.get("10116"),
                          "sell_ref": msg.get("10117")})
        elif mtype == "W":
            header, entries = _parse_snapshot_entries(pairs)
            snap = {**base, "orig_time": header.get("42"),
                    "symbol": header.get("55"),
                    "trading_status": header.get("8538"),
                    "prev_close": header.get("140"),
                    "num_trades": header.get("8503"),
                    "cum_volume": header.get("387"),
                    "cum_value": header.get("8504"),
                    "n_entries": header.get("268")}
            for e in entries:
                book.append({**snap,
                             "entry_type_code": e.get("269"),
                             "px": e.get("270"), "qty": e.get("271"),
                             "level": e.get("1023"),
                             "n_orders_at_level": e.get("346"),
                             "n_orders_detailed": e.get("73"),
                             "order_ids": [o.get("id") for o in e["orders"]] or None,
                             "order_qtys": [o.get("qty") for o in e["orders"]] or None})
        else:
            other.append({**base, "raw": body})
    return ticks, book, other


# ------------------- records -> typed DataFrames (3 kinds) ----------------
def _to_utc(s):
    return pd.to_datetime(s, format="mixed", utc=True, errors="coerce")


def build_frames(tick_recs, book_recs, other_recs, adds_index):
    """One chunk -> (df_ticks, df_book, df_other), dtypes locked so every
    partial Parquet has an identical schema. `adds_index` persists
    {(channel, appl_seq): order_id} across chunks for OrderID resolution."""
    # ---------------- ticks ----------------
    df_t = pd.DataFrame(tick_recs)
    if not df_t.empty:
        for c in ("price", "qty"):
            df_t[c] = pd.to_numeric(df_t[c], errors="coerce")
        for c in ("appl_seq", "msg_seq", "buy_ref", "sell_ref"):
            df_t[c] = pd.to_numeric(df_t[c], errors="coerce").astype("Int64")
        df_t["transact_time"] = _to_utc(df_t["transact_time"].str.replace("-", " ", n=1))
        df_t["sending_time"] = _to_utc(df_t["sending_time"].str.replace("-", " ", n=1))
        df_t["capture_ts"] = pd.to_datetime(df_t["capture_ts"], utc=True,
                                            format="mixed", errors="coerce")
        df_t["side"] = df_t["side_code"].map(SIDE_MAP)
        df_t["exec_type"] = df_t["exec_type_code"].map(EXECTYPE_MAP)
        df_t["market"] = df_t["segment"].map(MARKET_SEGMENT_MAP)

        is_add = df_t["event"] == "ORDER_ADD"
        for ch, sq, oid in zip(df_t.loc[is_add, "channel"],
                               df_t.loc[is_add, "appl_seq"],
                               df_t.loc[is_add, "order_id"]):
            if pd.notna(sq):
                adds_index[(ch, int(sq))] = oid

        ref = df_t["buy_ref"].where(df_t["buy_ref"].fillna(0) > 0, df_t["sell_ref"])
        df_t["resting_ref"] = ref.where(ref.fillna(0) > 0).astype("Int64")
        keys = zip(df_t["channel"],
                   df_t["resting_ref"].astype("float").fillna(-1).astype(int))
        resolved = pd.Series([adds_index.get(k) for k in keys], index=df_t.index)
        not_add = df_t["event"] != "ORDER_ADD"
        df_t.loc[not_add, "order_id"] = resolved[not_add]

        cxl_side = np.where(df_t["buy_ref"].fillna(0) > 0, "1",
                   np.where(df_t["sell_ref"].fillna(0) > 0, "2", None))
        is_cxl = df_t["event"] == "CANCEL"
        df_t.loc[is_cxl, "side_code"] = cxl_side[is_cxl]
        df_t.loc[is_cxl, "side"] = df_t.loc[is_cxl, "side_code"].map(SIDE_MAP)

        is_trade = df_t["event"] == "TRADE"
        init = pd.Series(pd.NA, index=df_t.index, dtype="object")
        init[is_trade & (df_t["sell_ref"].fillna(0) > 0)] = "BUYER_INITIATED"
        init[is_trade & (df_t["buy_ref"].fillna(0) > 0)] = "SELLER_INITIATED"
        df_t["initiator"] = init
        df_t["aggressor_side"] = df_t["initiator"].map(
            {"BUYER_INITIATED": "BUY", "SELLER_INITIATED": "SELL"})

        df_t = df_t[["transact_time", "sending_time", "capture_ts", "msg_seq",
                     "appl_seq", "channel", "segment", "market", "event",
                     "exec_type", "symbol", "side_code", "side", "price",
                     "qty", "order_id", "buy_ref", "sell_ref", "resting_ref",
                     "initiator", "aggressor_side"]]
        for c in ("channel", "segment", "market", "event", "exec_type",
                  "symbol", "side_code", "side", "order_id", "initiator",
                  "aggressor_side"):
            df_t[c] = df_t[c].astype("string")

    # ---------------- order book ----------------
    df_b = pd.DataFrame(book_recs)
    if not df_b.empty:
        for c in ("px", "qty", "prev_close", "cum_volume", "cum_value"):
            df_b[c] = pd.to_numeric(df_b[c], errors="coerce")
        for c in ("msg_seq", "num_trades", "n_entries", "level",
                  "n_orders_at_level", "n_orders_detailed"):
            df_b[c] = pd.to_numeric(df_b[c], errors="coerce").astype("Int64")
        df_b["snapshot_time"] = _to_utc(df_b["sending_time"].str.replace("-", " ", n=1))
        df_b["capture_ts"] = pd.to_datetime(df_b["capture_ts"], utc=True,
                                            format="mixed", errors="coerce")
        df_b["entry_type"] = df_b["entry_type_code"].map(MDENTRY_MAP)
        df_b["market"] = df_b["segment"].map(MARKET_SEGMENT_MAP)
        df_b["visible_qty_sum"] = df_b["order_qtys"].map(
            lambda q: float(np.sum([float(x) for x in q]))
            if isinstance(q, list) else np.nan)
        df_b["order_ids"] = df_b["order_ids"].map(
            lambda x: "|".join(x) if isinstance(x, list) else None)
        df_b["order_qtys"] = df_b["order_qtys"].map(
            lambda x: "|".join(x) if isinstance(x, list) else None)
        df_b = df_b[["snapshot_time", "capture_ts", "msg_seq", "channel",
                     "segment", "market", "symbol", "trading_status",
                     "prev_close", "num_trades", "cum_volume", "cum_value",
                     "entry_type_code", "entry_type", "level", "px", "qty",
                     "n_orders_at_level", "n_orders_detailed", "order_ids",
                     "order_qtys", "visible_qty_sum"]]
        for c in ("channel", "segment", "market", "symbol", "trading_status",
                  "entry_type_code", "entry_type", "order_ids", "order_qtys"):
            df_b[c] = df_b[c].astype("string")

    # ---------------- other ----------------
    df_o = pd.DataFrame(other_recs)
    if not df_o.empty:
        df_o["msg_seq"] = pd.to_numeric(df_o["msg_seq"],
                                        errors="coerce").astype("Int64")
        df_o["capture_ts"] = pd.to_datetime(df_o["capture_ts"], utc=True,
                                            format="mixed", errors="coerce")
        df_o["sending_time"] = _to_utc(df_o["sending_time"].str.replace("-", " ", n=1))
        df_o = df_o[["capture_ts", "sending_time", "msg_seq", "msg_type",
                     "channel", "segment", "raw"]]
        for c in ("msg_type", "channel", "segment", "raw"):
            df_o[c] = df_o[c].astype("string")
    return df_t, df_b, df_o


# ----------------------------- main pipeline ------------------------------
def run_day(src=SRC, out_dir=OUT_DIR, chunk_lines=CHUNK_LINES):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = Path(src).name.replace(".tar.gz", "")
    kinds = ("ticks", "orderbook", "other")

    # ---- pass 1: chunk -> parse -> 3 partial parquets -> free RAM ----
    adds_index = {}
    n_chunk = 0
    totals = dict.fromkeys(kinds, 0)

    tf = tarfile.open(src, "r:*")
    member = next(m for m in tf.getmembers()
                  if m.isfile() and m.name.lower().endswith(".txt"))
    stream = io.TextIOWrapper(tf.extractfile(member), encoding="utf-8",
                              errors="replace")
    buf = []
    for line in stream:
        buf.append(line)
        if len(buf) < chunk_lines:
            continue
        n_chunk += 1
        _write_chunk(buf, n_chunk, adds_index, out_dir, day, totals)
        buf = []
        gc.collect()                                   # clear RAM
    if buf:
        n_chunk += 1
        _write_chunk(buf, n_chunk, adds_index, out_dir, day, totals)
        buf = []
        gc.collect()
    tf.close()
    del adds_index
    gc.collect()
    print(f"\npass 1 done: {n_chunk} chunks | rows:", totals)

    # ---- pass 2: combine partials -> 3 final files, delete partials ----
    for kind in kinds:
        parts = sorted(out_dir.glob(f"_part_{kind}_*.parquet"))
        if not parts:
            print(f"{kind}: no data")
            continue
        final = out_dir / f"{day}_{kind}.parquet"
        writer = None
        for p in parts:
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=131_072):
                if writer is None:
                    writer = pq.ParquetWriter(final, batch.schema,
                                              compression="zstd")
                writer.write_batch(batch)
        writer.close()
        for p in parts:                                # delete partial files
            p.unlink()
        print(f"{kind}: {totals[kind]:,} rows -> {final}")
    return out_dir


def _write_chunk(buf, n_chunk, adds_index, out_dir, day, totals):
    t, b, o = parse_fix_file(buf)                      # <- the parser
    df_t, df_b, df_o = build_frames(t, b, o, adds_index)
    for kind, df in (("ticks", df_t), ("orderbook", df_b), ("other", df_o)):
        if df.empty:
            continue
        path = out_dir / f"_part_{kind}_{n_chunk:04d}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       path, compression="zstd")
        totals[kind] += len(df)
    print(f"chunk {n_chunk:>3}: {len(buf):>9,} msgs -> "
          f"ticks {len(df_t):>9,} | book {len(df_b):>10,} | other {len(df_o):>7,}")
    del t, b, o, df_t, df_b, df_o                      # clear RAM


# %%
run_day()

# %% [markdown]
# Afterwards you have exactly three files in
# `...\Capital Stake\parsed\2026-06-30\`:
# `2026-06-30_ticks.parquet`, `2026-06-30_orderbook.parquet`,
# `2026-06-30_other.parquet` — all partials deleted.
#
# ```python
# import pandas as pd
# trades = pd.read_parquet(OUT_DIR / "2026-06-30_ticks.parquet",
#                          filters=[("event", "==", "TRADE")])
# ```
