# To run in CMD
# .backtest\Scripts\python.exe psx_unified_output.py

# =============================================================================
# PSX FIX daily capture -> ONE unified CSV + ONE unified Parquet file
# Spec: PSX FIX Market Data Interface Specifications v1.05
#
# WHY ONE TABLE (unlike the earlier 4-table version):
#   You want to filter by symbol + market, then sort by time, and see -
#   in one continuous chronological stream - when an order book snapshot
#   (35=W) was received, when the book was updated (UA201 add / UA202
#   cancel), and when a trade happened (UA202 ExecType=F) - all interleaved
#   as they actually occurred on the wire. Splitting into 4 tables makes
#   that interleaved view harder to reconstruct, so this version keeps
#   every message type as ROWS of one common (wide, mostly-nullable) schema.
#
# ROW TYPES (all in the same "event_type" column):
#   ORDER_ADD    <- UA201                 (order book update)
#   CANCEL       <- UA202 ExecType=4      (order book update)
#   TRADE        <- UA202 ExecType=F      (tick-level trade)
#   OB_SNAPSHOT  <- 35=W (one row per MDEntry line, i.e. per price level/stat)
#   HEARTBEAT    <- UA001
#   SESSION_STATUS <- h
#   SECURITY_STATUS <- f
#   NEWS         <- B
#   RETRANSMIT   <- UA002
#   CHANNEL_STATS <- UA004
#   BUSINESS_REJECT <- j
#
# All the same correctness fixes from the 4-table version are preserved:
#   - auction cross handling (both refs nonzero -> AUCTION, resting_ref NULL)
#   - cancel price resolved from adds_index (tag 31 is unreliable on cancels)
#   - exec_inst forced null on ORDER_ADD (UA201 has no tag 18)
#   - corrected MDEntryType dictionary (xa/xb/xc/xd/5/6/x5/x6/x7/x8/xg/xl)
#   - xe/xf circuit-breaker sentinel handled asymmetrically
#   - TradingPhaseCode decoded into phase / suspended_all_day / break_reason
#   - UA001 heartbeat fields promoted (appl_last_seq, end_of_channel, hb_time)
#
# CHRONOLOGICAL ORDERING - see the printed note at the bottom of this file
# for exactly which two columns to sort by (msg_seq vs appl_seq) and why.
# =============================================================================

import gc
import io
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ─────────────────────────── configuration ───────────────────────────────────
SRC         = Path(r"C:\Users\shahz\OneDrive\Desktop\Del\Capital Stake\2026-06-30.tar.gz")
OUT_DIR     = SRC.parent / "parsed_unified" / SRC.name.replace(".tar.gz", "")
CHUNK_LINES = 250_000      # lines per batch; lower to 100_000 if RAM is tight

SOH = "^"

SIDE_MAP = {"1": "BUY", "2": "SELL"}
EXECTYPE_MAP = {"4": "CANCELLED", "F": "TRADE"}

SNAPSHOT_SEGMENT_MAP = {
    "010": "REG", "020": "BILLS_BOND", "030": "STOCK_DEL_FUT",
    "040": "STOCK_CS_FUT", "050": "STOCK_DEL_OPT", "060": "INDEX_OPT",
    "070": "STOCK_IDX_FUT", "080": "ODD_LOT", "100": "EQ_SQUARE_UP",
    "120": "FUT_SQUARE_UP", "900": "INDEX",
}

TICK_SEGMENT_MAP = {
    "011": "REG", "031": "STOCK_DEL_FUT", "041": "STOCK_CS_FUT",
    "051": "STOCK_DEL_OPT", "061": "INDEX_OPT", "071": "STOCK_IDX_FUT",
    "081": "ODD_LOT", "091": "NDM",
}

MDENTRY_MAP = {
    "0": "BID", "1": "OFFER", "2": "LAST_TRADE", "3": "INDEX_VALUE",
    "4": "OPENING_PRICE", "5": "CLOSING_PRICE", "6": "SETTLEMENT_PRICE",
    "7": "SESSION_HIGH", "8": "SESSION_LOW",
    "x1": "NET_CHANGE_1", "x2": "NET_CHANGE_2",
    "x3": "AGG_BID", "x4": "AGG_OFFER",
    "x5": "PE_RATIO_1", "x6": "PE_RATIO_2",
    "x7": "FUND_PREV_NAV", "x8": "ETF_INAV",
    "xa": "PREV_CLOSE_INDEX", "xb": "OPEN_INDEX",
    "xc": "HIGH_INDEX", "xd": "LOW_INDEX",
    "xe": "UPPER_CIRCUIT_BREAKER", "xf": "LOWER_CIRCUIT_BREAKER",
    "xg": "OPEN_INTEREST", "xl": "CLOSE_INDEX",
}

XE_NO_LIMIT_SENTINEL = 999999999.9999

PHASE_MAP = {
    "S": "STARTING", "O": "OPEN_CALL_AUCTION", "T": "CONTINUOUS_AUCTION",
    "B": "TRADING_BREAK", "N": "NORMAL_CALL_AUCTION_PM",
    "H": "TEMPORARY_SUSPENSION", "V": "NORMAL_CALL_AUCTION_RESUME",
    "C": "CLOSE_CALL_AUCTION", "A": "AFTER_HOUR_TRADING", "E": "MARKET_CLOSED",
}

BREAK_REASON_MAP = {
    "1": "AFTER_PRE_OPEN", "2": "FRIDAY_LUNCH_BREAK",
    "3": "AFTER_PRE_OPEN_PM_FRIDAY", "4": "BEFORE_POST_CLOSE",
}

# The single unified schema every row (regardless of message type) is
# reindexed into before being written out. Order matters only for readability.
UNIFIED_COLUMNS = [
    # ── identity / ordering ──────────────────────────────────────────────
    "event_type",        # ORDER_ADD / CANCEL / TRADE / OB_SNAPSHOT / etc.
    "capture_ts",         # wall-clock receive time (log prefix)
    "sending_time",       # SendingTime(52)
    "orig_time",          # OrigTime(42)  (snapshots / UA004)
    "transact_time",      # TransactTime(60) (orders/trades/heartbeat)
    "msg_seq",            # MsgSeqNum(34)  - session-level sequence
    "appl_seq",           # ApplSeqNum(1181) - channel-level sequence
    "channel",            # ChannelNo(10201)
    "segment",            # raw MDStreamID(1500)
    "market",             # human label derived from segment
    "symbol",             # Symbol(55)

    # ── order book update / trade fields ─────────────────────────────────
    "side",               # BUY/SELL (adds); back-filled on cancels
    "price",               # order Price(44) / resolved cancel price / LastPx(31) on trade
    "qty",                 # OrderQty(38) / cancel qty / LastQty(32) on trade
    "order_id",            # OrderID(37), resolved via adds_index on cancels
    "exec_type_code",      # ExecType(150) raw: '4' or 'F'
    "exec_type",           # CANCELLED / TRADE
    "exec_inst",           # ExecInst(18); structurally null on ORDER_ADD
    "buy_ref",             # BidApplSeqNum(10116)
    "sell_ref",            # OfferApplSeqNum(10117)
    "resting_ref",         # derived: whichever ref is nonzero (NULL if auction)
    "initiator",           # trades only: BUYER_INITIATED/SELLER_INITIATED/AUCTION
    "aggressor_side",      # trades only: BUY/SELL/AUCTION
    "raw_last_px",         # trades: same as price; cancels: raw unresolved tag31 (QA)

    # ── order book snapshot fields (OB_SNAPSHOT rows only) ───────────────
    "trading_status",      # raw TradingPhaseCode(8538)
    "phase",               # decoded phase label
    "suspended_all_day",   # decoded suspended flag
    "break_reason",        # decoded break reason (phase B only)
    "prev_close",          # PrevClosePx(140)
    "num_trades",          # NumTrades(8503)
    "cum_volume",          # TotalVolumeTrade(387)
    "cum_value",           # TotalValueTrade(8504)
    "entry_type_code",     # MDEntryType(269) raw
    "entry_type",          # decoded label
    "level",               # MDPriceLevel(1023)
    "n_orders_at_level",   # NumberOfOrders(346)
    "n_orders_detailed",   # NoOrders(73)
    "order_ids",           # pipe-joined OrderID(37) list at this level
    "order_qtys",          # pipe-joined OrderQty(38) list at this level
    "visible_qty_sum",     # sum of the disclosed order_qtys above
    "is_xf_tick_floor",    # heuristic flag for xf "no-limit" rows

    # ── heartbeat / misc message fields ──────────────────────────────────
    "n_streams",           # NoMDStreamID(10208) on UA004
    "appl_last_seq",       # ApplLastSeqNum(1350) on UA001
    "end_of_channel",      # EndOfChannel(10205) on UA001
    "heartbeat_time",      # TransactTime(60) on UA001

    "raw",                 # full raw FIX body, kept for any unclassified msg
]


# ─────────────────────────── low-level parsing ───────────────────────────────

def _tokenize(body: str):
    return [p.split("=", 1) for p in body.rstrip(SOH).split(SOH) if "=" in p]


def _parse_snapshot_entries(pairs):
    """Parse repeating MDEntry group of a 35=W message."""
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


def parse_fix_chunk(lines):
    """
    Parse one chunk of raw FIX lines into ONE flat list of dict records.
    Each dict already carries an 'event_type' key so downstream building
    can branch on it, but everything lives in one list/one DataFrame.
    """
    records = []

    for raw in lines:
        raw = raw.strip()
        if not raw or "|" not in raw:
            continue

        capture_ts, body = raw.split("|", 1)
        pairs  = _tokenize(body)
        msg    = dict(pairs)
        mtype  = msg.get("35")

        base = {
            "capture_ts":   capture_ts,
            "msg_seq":      msg.get("34"),
            "sending_time": msg.get("52"),
            "msg_type":     mtype,
            "channel":      msg.get("10201"),
            "segment":      msg.get("1500"),
        }

        # ── UA201: Tick Order → order book add ───────────────────────────
        if mtype == "UA201":
            records.append({
                **base,
                "event_type":    "ORDER_ADD",
                "appl_seq":      msg.get("1181"),
                "symbol":        msg.get("55"),
                "side_code":     msg.get("54"),
                "price":         msg.get("44"),
                "qty":           msg.get("38"),
                "order_id":      msg.get("37"),
                "transact_time": msg.get("60"),
                "exec_type_code": None,
                "exec_inst":     None,   # UA201 has no tag 18 in spec
                "buy_ref":       None,
                "sell_ref":      None,
            })

        # ── UA202: Tick Execution → trade or cancel ───────────────────────
        elif mtype == "UA202":
            et        = msg.get("150")
            exec_inst = msg.get("18")

            if et == "F":
                records.append({
                    **base,
                    "event_type":    "TRADE",
                    "appl_seq":      msg.get("1181"),
                    "symbol":        msg.get("55"),
                    "price":         msg.get("31"),
                    "qty":           msg.get("32"),
                    "transact_time": msg.get("60"),
                    "exec_type_code": "F",
                    "exec_inst":     exec_inst,
                    "buy_ref":       msg.get("10116"),
                    "sell_ref":      msg.get("10117"),
                })
            elif et == "4":
                records.append({
                    **base,
                    "event_type":    "CANCEL",
                    "appl_seq":      msg.get("1181"),
                    "symbol":        msg.get("55"),
                    "side_code":     None,
                    "raw_last_px":   msg.get("31"),
                    "qty":           msg.get("32"),
                    "order_id":      None,
                    "transact_time": msg.get("60"),
                    "exec_type_code": "4",
                    "exec_inst":     exec_inst,
                    "buy_ref":       msg.get("10116"),
                    "sell_ref":      msg.get("10117"),
                })
            else:
                records.append({**base, "event_type": "OTHER", "raw": body})

        # ── 35=W: Snapshot Data ───────────────────────────────────────────
        elif mtype == "W":
            header, entries = _parse_snapshot_entries(pairs)
            snap_base = {
                **base,
                "event_type":     "OB_SNAPSHOT",
                "orig_time":      header.get("42"),
                "symbol":         header.get("55"),
                "trading_status": header.get("8538"),
                "prev_close":     header.get("140"),
                "num_trades":     header.get("8503"),
                "cum_volume":     header.get("387"),
                "cum_value":      header.get("8504"),
                "n_entries":      header.get("268"),
            }
            for e in entries:
                records.append({
                    **snap_base,
                    "entry_type_code":   e.get("269"),
                    "px":                e.get("270"),
                    "qty":               e.get("271"),
                    "level":             e.get("1023"),
                    "n_orders_at_level": e.get("346"),
                    "n_orders_detailed": e.get("73"),
                    "order_ids":  [o.get("id")  for o in e["orders"]] or None,
                    "order_qtys": [o.get("qty") for o in e["orders"]] or None,
                })

        elif mtype == "UA004":
            records.append({
                **base,
                "event_type": "CHANNEL_STATS",
                "raw":        body,
                "orig_time":  msg.get("42"),
                "n_streams":  msg.get("10208"),
            })

        elif mtype == "UA001":
            records.append({
                **base,
                "event_type":       "HEARTBEAT",
                "raw":              body,
                "appl_last_seq":    msg.get("1350"),
                "end_of_channel":   msg.get("10205"),
                "heartbeat_time":   msg.get("60"),
            })

        elif mtype == "h":
            records.append({**base, "event_type": "SESSION_STATUS", "raw": body,
                             "orig_time": msg.get("42")})
        elif mtype == "f":
            records.append({**base, "event_type": "SECURITY_STATUS", "raw": body,
                             "orig_time": msg.get("42"), "symbol": msg.get("55")})
        elif mtype == "B":
            records.append({**base, "event_type": "NEWS", "raw": body,
                             "orig_time": msg.get("42")})
        elif mtype == "UA002":
            records.append({**base, "event_type": "RETRANSMIT", "raw": body})
        elif mtype == "j":
            records.append({**base, "event_type": "BUSINESS_REJECT", "raw": body})
        else:
            records.append({**base, "event_type": "OTHER", "raw": body})

    return records


# ─────────────────────────── DataFrame builder ────────────────────────────────

def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype(str).str.replace("-", " ", n=1),
        format="mixed", utc=True, errors="coerce"
    )


def build_unified_frame(records, adds_index: dict) -> pd.DataFrame:
    """
    Build ONE DataFrame from the flat record list, applying all the same
    correctness fixes as the 4-table version, then reindex into the fixed
    UNIFIED_COLUMNS schema so every chunk's parquet part has an identical
    schema (this is what prevents the earlier merge-schema-mismatch bug).
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # numeric coercions (only touch columns that exist in this chunk)
    for c in ("price", "qty", "prev_close", "cum_volume", "cum_value",
              "raw_last_px"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("appl_seq", "msg_seq", "buy_ref", "sell_ref", "channel",
              "num_trades", "n_entries", "level", "n_orders_at_level",
              "n_orders_detailed", "n_streams", "appl_last_seq",
              "end_of_channel"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # timestamps
    df["capture_ts"] = pd.to_datetime(df["capture_ts"], utc=True,
                                      format="mixed", errors="coerce")
    for c in ("sending_time", "orig_time", "transact_time", "heartbeat_time"):
        if c in df.columns:
            df[c] = _to_utc(df[c])
        else:
            df[c] = pd.NaT

    # snapshot px/qty were parsed under "px"/"qty" - unify into price/qty
    if "px" in df.columns:
        is_snap = df["event_type"] == "OB_SNAPSHOT"
        df.loc[is_snap, "price"] = pd.to_numeric(df.loc[is_snap, "px"],
                                                 errors="coerce")
        df.drop(columns=["px"], inplace=True)

    df["exec_type"] = df.get("exec_type_code", pd.Series(dtype="object")) \
                        .map(EXECTYPE_MAP)
    df["market"] = df["segment"].map({**SNAPSHOT_SEGMENT_MAP,
                                      **TICK_SEGMENT_MAP})

    # ── side (adds only initially) ────────────────────────────────────────
    if "side_code" in df.columns:
        df["side"] = df["side_code"].map(SIDE_MAP)
    else:
        df["side"] = None

    # ── auction / initiator logic for TRADE rows ─────────────────────────
    is_trade = df["event_type"] == "TRADE"
    if is_trade.any():
        buy_pos  = df["buy_ref"].fillna(0)  > 0
        sell_pos = df["sell_ref"].fillna(0) > 0
        is_auction = buy_pos & sell_pos & is_trade

        ref = df["buy_ref"].where(buy_pos, df["sell_ref"])
        df["resting_ref"] = ref.where(ref.fillna(0) > 0).astype("Int64")
        df.loc[is_auction, "resting_ref"] = pd.NA

        keys = list(zip(df["channel"].astype("object"),
                        df["resting_ref"].astype("float").fillna(-1).astype(int)))
        df["order_id_resolved"] = [
            adds_index.get(k, (None, None))[0] if is_trade.iloc[i] else None
            for i, k in enumerate(keys)
        ]

        init = pd.Series(pd.NA, index=df.index, dtype="object")
        init[sell_pos & is_trade & ~is_auction] = "BUYER_INITIATED"
        init[buy_pos  & is_trade & ~is_auction] = "SELLER_INITIATED"
        init[is_auction]                         = "AUCTION"
        df["initiator"] = init
        df["aggressor_side"] = df["initiator"].map(
            {"BUYER_INITIATED": "BUY", "SELLER_INITIATED": "SELL",
             "AUCTION": "AUCTION"})

        # for trades, order_id column = resolved resting order id; raw_last_px = price
        df.loc[is_trade, "order_id"] = df.loc[is_trade, "order_id_resolved"]
        df.loc[is_trade, "raw_last_px"] = df.loc[is_trade, "price"]
        df.drop(columns=["order_id_resolved"], inplace=True)
    else:
        df["resting_ref"] = pd.array([pd.NA] * len(df), dtype="Int64")
        df["initiator"] = None
        df["aggressor_side"] = None

    # ── ORDER_ADD: populate adds_index cache; exec_inst forced null ──────
    is_add = df["event_type"] == "ORDER_ADD"
    if is_add.any():
        df.loc[is_add, "exec_inst"] = pd.NA
        for ch, sq, oid, px in zip(df.loc[is_add, "channel"],
                                    df.loc[is_add, "appl_seq"],
                                    df.loc[is_add, "order_id"],
                                    df.loc[is_add, "price"]):
            if pd.notna(sq) and pd.notna(ch):
                adds_index[(int(ch), int(sq))] = (oid, px)

    # ── CANCEL: resolve order_id / price via adds_index ───────────────────
    is_cxl = df["event_type"] == "CANCEL"
    if is_cxl.any():
        ref = df["buy_ref"].where(df["buy_ref"].fillna(0) > 0, df["sell_ref"])
        cxl_resting_ref = ref.where(ref.fillna(0) > 0).astype("Int64")
        df.loc[is_cxl, "resting_ref"] = cxl_resting_ref[is_cxl]

        keys = list(zip(df["channel"].astype("object"),
                        cxl_resting_ref.astype("float").fillna(-1).astype(int)))
        resolved_oid = pd.Series(
            [adds_index.get(k, (None, None))[0] for k in keys], index=df.index)
        resolved_px = pd.Series(
            [adds_index.get(k, (None, None))[1] for k in keys], index=df.index)

        df.loc[is_cxl, "order_id"] = resolved_oid[is_cxl]
        df.loc[is_cxl, "price"]    = resolved_px[is_cxl]

        cxl_side = np.where(df["buy_ref"].fillna(0)  > 0, "1",
                   np.where(df["sell_ref"].fillna(0) > 0, "2", None))
        df.loc[is_cxl, "side"] = pd.Series(cxl_side, index=df.index).map(SIDE_MAP)[is_cxl]

    # ── OB_SNAPSHOT: entry-type decode, circuit breaker, phase decode ────
    is_snap = df["event_type"] == "OB_SNAPSHOT"
    if is_snap.any():
        df["entry_type"] = df.get("entry_type_code", pd.Series(dtype="object")) \
                              .map(MDENTRY_MAP)

        is_xe = df["entry_type_code"] == "xe"
        df.loc[is_xe & (df["price"] == XE_NO_LIMIT_SENTINEL), "price"] = np.nan

        is_xf = df["entry_type_code"] == "xf"
        df["is_xf_tick_floor"] = pd.NA
        df.loc[is_xf, "is_xf_tick_floor"] = df.loc[is_xf, "price"] <= 1.0

        ts = df["trading_status"].astype(str)
        phase_char     = ts.str.slice(0, 1)
        suspended_char = ts.str.slice(1, 2)
        break_char     = ts.str.slice(2, 3)

        df["phase"]             = phase_char.map(PHASE_MAP).astype("string")
        df["suspended_all_day"] = suspended_char.map({"1": True, "0": False})
        df["break_reason"]      = np.where(
            phase_char == "B", break_char.map(BREAK_REASON_MAP), None)
        df["break_reason"] = df["break_reason"].astype("string")

        df["visible_qty_sum"] = df.get(
            "order_qtys", pd.Series([None] * len(df))
        ).map(lambda q: float(np.sum([float(x) for x in q]))
              if isinstance(q, list) else np.nan)
        df["order_ids"] = df.get("order_ids", pd.Series([None] * len(df))).map(
            lambda x: "|".join(x) if isinstance(x, list) else None)
        df["order_qtys"] = df.get("order_qtys", pd.Series([None] * len(df))).map(
            lambda x: "|".join(x) if isinstance(x, list) else None)
    else:
        for c in ("entry_type", "is_xf_tick_floor", "phase",
                  "suspended_all_day", "break_reason", "visible_qty_sum"):
            if c not in df.columns:
                df[c] = None

    # ── reindex into the fixed unified schema ─────────────────────────────
    df = df.reindex(columns=UNIFIED_COLUMNS)

    string_cols = ["event_type", "segment", "market", "symbol", "side",
                   "order_id", "exec_type_code", "exec_type", "exec_inst",
                   "initiator", "aggressor_side", "trading_status", "phase",
                   "break_reason", "entry_type_code", "entry_type",
                   "order_ids", "order_qtys", "raw"]
    for c in string_cols:
        df[c] = df[c].astype("string")

    return df


# ─────────────────────────── chunk writer ────────────────────────────────────

def _write_chunk(buf, n_chunk, adds_index, out_dir, totals, t0):
    print(f"chunk {n_chunk:>3}: parsing  {len(buf):>9,} lines …", flush=True)
    records = parse_fix_chunk(buf)

    print(f"chunk {n_chunk:>3}: building unified frame …", flush=True)
    df = build_unified_frame(records, adds_index)

    if not df.empty:
        path = out_dir / f"_part_{n_chunk:04d}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       path, compression="zstd")
        totals["rows"] += len(df)

    elapsed = time.time() - t0
    print(f"chunk {n_chunk:>3} DONE ({elapsed:6.1f}s) | rows {len(df):>9,} | "
          f"running total {totals['rows']:>10,}", flush=True)

    del records, df
    gc.collect()


# ─────────────────────────── main pipeline ───────────────────────────────────

def run_day(src=SRC, out_dir=OUT_DIR, chunk_lines=CHUNK_LINES):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day    = Path(src).name.replace(".tar.gz", "")
    totals = {"rows": 0}

    adds_index = {}   # (channel, appl_seq) -> (order_id, price)
    n_chunk    = 0
    n_lines    = 0
    t0         = time.time()

    print(f"\nOpening {Path(src).name} …", flush=True)
    with tarfile.open(src, "r:*") as tf:
        member = next(m for m in tf.getmembers()
                      if m.isfile() and m.name.lower().endswith(".txt"))
        print(f"  inner file : {member.name}", flush=True)
        stream = io.TextIOWrapper(tf.extractfile(member),
                                  encoding="utf-8", errors="replace")

        buf = []
        for line in stream:
            buf.append(line)
            n_lines += 1
            if n_lines % 500_000 == 0:
                print(f"  … {n_lines:,} lines read ({time.time()-t0:.1f}s)",
                      flush=True)
            if len(buf) >= chunk_lines:
                n_chunk += 1
                _write_chunk(buf, n_chunk, adds_index, out_dir, totals, t0)
                buf = []

        if buf:
            n_chunk += 1
            _write_chunk(buf, n_chunk, adds_index, out_dir, totals, t0)
            del buf

    del adds_index
    gc.collect()
    print(f"\nPass 1 done: {n_chunk} chunks, {n_lines:,} lines, "
          f"{totals['rows']:,} rows.", flush=True)

    print("\nMerging partials into ONE parquet + ONE csv …", flush=True)
    parts = sorted(out_dir.glob("_part_*.parquet"))
    if not parts:
        print("  no data — nothing to merge")
        return

    final_parquet = out_dir / f"{day}_unified.parquet"
    final_csv     = out_dir / f"{day}_unified.csv"

    schemas = [pq.ParquetFile(p).schema_arrow for p in parts]
    try:
        unified_schema = pa.unify_schemas(schemas, promote_options="permissive")
    except TypeError:
        unified_schema = pa.unify_schemas(schemas)

    writer = None
    csv_header_written = False
    for p in parts:
        pf = pq.ParquetFile(p)
        try:
            for batch in pf.iter_batches(batch_size=131_072):
                tbl = pa.Table.from_batches([batch]).cast(unified_schema)

                if writer is None:
                    writer = pq.ParquetWriter(final_parquet, unified_schema,
                                              compression="zstd")
                writer.write_table(tbl)

                chunk_df = tbl.to_pandas()
                chunk_df.to_csv(final_csv, mode="a", index=False,
                               header=not csv_header_written)
                csv_header_written = True
        finally:
            pf.close()
    if writer:
        writer.close()

    for p in parts:
        try:
            p.unlink()
        except PermissionError:
            print(f"  WARNING: could not delete {p.name} "
                  f"(OneDrive lock?) — delete manually")

    mb_pq  = final_parquet.stat().st_size / 1_048_576
    mb_csv = final_csv.stat().st_size / 1_048_576
    print(f"  unified table : {totals['rows']:>12,} rows")
    print(f"    parquet -> {final_parquet.name}  ({mb_pq:.1f} MB)")
    print(f"    csv     -> {final_csv.name}  ({mb_csv:.1f} MB)")

    print(f"\nAll done in {time.time()-t0:.1f}s", flush=True)
    print(f"Output : {out_dir}", flush=True)


# ─────────────────────────── entry point ─────────────────────────────────────
if __name__ == "__main__":
    run_day()

    # To process all days in a folder, replace the line above with:
    # for f in sorted(SRC.parent.glob("*.tar.gz")):
    #     run_day(src=f, out_dir=f.parent / "parsed_unified" / f.name.replace(".tar.gz",""))
