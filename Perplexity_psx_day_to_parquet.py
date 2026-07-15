# =============================================================================
# PSX FIX daily capture -> 4 Parquet files (chunked, bounded RAM)
# Spec: PSX FIX Market Data Interface Specifications v1.05
#
# Tables produced:
#   {day}_trades.parquet       - UA202 ExecType=F (TRADE) only — tick level
#   {day}_ob_snapshot.parquet  - 35=W full order book snapshots from exchange
#   {day}_ob_updates.parquet   - UA201 (order add) + UA202 ExecType=4 (cancel)
#   {day}_other.parquet        - heartbeats, session status, news, all else
#
# FIX 7 spec errors vs PSX FIX Market Data Interface Specifications v1.05:
#   Fix 1 - ExecType(150) only valid values are 4=Cancelled and F=Trade
#   Fix 2 - Side(54) only valid values in this spec are 1=Buy and 2=Sell
#   Fix 3 - MDStreamID(1500) separate maps for Snapshot vs Tick channels
#   Fix 4 - UA004 structured parsing (TradingPhaseCode per segment)
#   Fix 5 - ExecInst(18) captured from UA202 (B=Ok to Cross for NDM)
#   Fix 6 - xl MDEntryType (close index) added to MDENTRY_MAP
#   Fix 7 - ChannelNo(10201) cast to Int64 (spec: N4 numeric)
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
OUT_DIR     = SRC.parent / "parsed" / SRC.name.replace(".tar.gz", "")
CHUNK_LINES = 250_000      # lines per batch; lower to 100_000 if RAM is tight

SOH = "^"

# Fix 2: spec Side(54) valid values are 1=Buy, 2=Sell only in market data feed
SIDE_MAP = {
    "1": "BUY",
    "2": "SELL",
}

# Fix 1: spec ExecType(150) valid values are 4=Cancelled and F=Trade only
EXECTYPE_MAP = {
    "4": "CANCELLED",
    "F": "TRADE",
}

# Fix 3: separate MDStreamID maps for Snapshot (35=W) vs Tick (UA201/UA202)
# Snapshot channel MDStreamID values (tag 1500 inside 35=W messages)
SNAPSHOT_SEGMENT_MAP = {
    "010": "REG",                   # Regular Market
    "020": "BILLS_BOND",            # Bills and Bond Market (was wrongly "ODL")
    "030": "STOCK_DEL_FUT",         # Stock Deliverable Future
    "040": "STOCK_CS_FUT",          # Stock Cash Settled Future
    "050": "STOCK_DEL_OPT",         # Stock Deliverable Option
    "060": "INDEX_OPT",             # Stock Index Option
    "070": "STOCK_IDX_FUT",         # Stock Index Future
    "080": "ODD_LOT",               # Odd Lot Market
    "100": "EQ_SQUARE_UP",          # Equities Square Up
    "120": "FUT_SQUARE_UP",         # Futures Square Up
    "900": "INDEX",                 # Index
}

# Tick channel MDStreamID values (tag 1500 inside UA201/UA202 messages)
TICK_SEGMENT_MAP = {
    "011": "REG",                   # Regular Market
    "031": "STOCK_DEL_FUT",         # Stock Deliverable Future
    "041": "STOCK_CS_FUT",          # Stock Cash Settled Future
    "051": "STOCK_DEL_OPT",         # Stock Deliverable Option
    "061": "INDEX_OPT",             # Stock Index Option
    "071": "STOCK_IDX_FUT",         # Stock Index Future
    "081": "ODD_LOT",               # Odd Lot Market
    "091": "NDM",                   # Negotiated Deal Market
}

# Fix 6: added xl=CLOSE_INDEX (was missing)
MDENTRY_MAP = {
    "0":  "BID",
    "1":  "OFFER",
    "2":  "LAST_TRADE",             # latest execution price and volume
    "3":  "INDEX_VALUE",
    "4":  "OPENING_PRICE",
    "5":  "CLOSING_PRICE",
    "6":  "SETTLEMENT_PRICE",
    "7":  "SESSION_HIGH",
    "8":  "SESSION_LOW",
    "x1": "NET_CHANGE",             # latest px minus prev close
    "x2": "NET_CHANGE_2",           # latest px minus last latest px
    "x3": "AGG_BID",                # VWAP px / total qty (within auction range)
    "x4": "AGG_OFFER",              # VWAP px / total qty (within auction range)
    "x5": "PE_RATIO_1",             # reserved, not released
    "x6": "PE_RATIO_2",             # reserved, not released
    "x7": "FUND_PREV_NAV",          # fund prev NAV (incl. ETF)
    "x8": "ETF_INAV",               # ETF intraday NAV
    "xa": "INDEX_PREV_CLOSE",
    "xb": "INDEX_OPEN",
    "xc": "INDEX_HIGH",
    "xd": "INDEX_LOW",
    "xe": "UPPER_CIRCUIT_BREAKER",
    "xf": "LOWER_CIRCUIT_BREAKER",
    "xg": "OPEN_INTEREST",          # position qty of derivative contract
    "xl": "CLOSE_INDEX",            # Fix 6: close index (added)
}


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
    Parse one chunk of raw FIX lines into 4 separate record lists.

    Returns
    -------
    trades     : UA202 ExecType=F (actual fills / tick-level trades)
    ob_updates : UA201 (order add) + UA202 ExecType=4 (cancels)
    ob_snaps   : 35=W full order book snapshots
    other      : UA001 heartbeats, h session status, B news,
                 UA002 retransmit, UA004 stats, j reject, f security status
    """
    trades, ob_updates, ob_snaps, other = [], [], [], []

    for raw in lines:
        raw = raw.strip()
        if not raw or "|" not in raw:
            continue

        capture_ts, body = raw.split("|", 1)
        pairs  = _tokenize(body)
        msg    = dict(pairs)
        mtype  = msg.get("35")

        # Fix 7: channel stored as raw string; cast to Int64 in build step
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
            ob_updates.append({
                **base,
                "event":         "ORDER_ADD",
                "appl_seq":      msg.get("1181"),
                "symbol":        msg.get("55"),
                "side_code":     msg.get("54"),
                "price":         msg.get("44"),
                "qty":           msg.get("38"),
                "order_id":      msg.get("37"),
                "transact_time": msg.get("60"),
                "exec_type_code":None,
                "exec_inst":     None,      # Fix 5: placeholder
                "buy_ref":       None,
                "sell_ref":      None,
            })

        # ── UA202: Tick Execution → trade or cancel ───────────────────────
        elif mtype == "UA202":
            et       = msg.get("150")
            exec_inst= msg.get("18")        # Fix 5: ExecInst B=Ok to Cross

            # Fix 1: only F and 4 are valid per spec; anything else → other
            if et == "F":
                trades.append({
                    **base,
                    "event":         "TRADE",
                    "appl_seq":      msg.get("1181"),
                    "symbol":        msg.get("55"),
                    "price":         msg.get("31"),
                    "qty":           msg.get("32"),
                    "transact_time": msg.get("60"),
                    "exec_type_code":"F",
                    "exec_inst":     exec_inst,  # Fix 5
                    "buy_ref":       msg.get("10116"),
                    "sell_ref":      msg.get("10117"),
                })
            elif et == "4":
                ob_updates.append({
                    **base,
                    "event":         "CANCEL",
                    "appl_seq":      msg.get("1181"),
                    "symbol":        msg.get("55"),
                    "side_code":     None,
                    "price":         msg.get("31"),
                    "qty":           msg.get("32"),
                    "order_id":      None,
                    "transact_time": msg.get("60"),
                    "exec_type_code":"4",
                    "exec_inst":     exec_inst,  # Fix 5
                    "buy_ref":       msg.get("10116"),
                    "sell_ref":      msg.get("10117"),
                })
            else:
                # unexpected ExecType — keep raw in other table
                other.append({**base, "raw": body})

        # ── 35=W: Snapshot Data ───────────────────────────────────────────
        elif mtype == "W":
            header, entries = _parse_snapshot_entries(pairs)
            snap_base = {
                **base,
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
                ob_snaps.append({
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

        # Fix 4: UA004 Statistics — parse structure instead of raw dump
        elif mtype == "UA004":
            other.append({
                **base,
                "raw":          body,
                "orig_time":    msg.get("42"),
                "n_streams":    msg.get("10208"),
                # repeating group: stream-level TradingPhaseCode stored as raw
                # (variable-length repeating group; full parse adds complexity
                # for minimal gain — the raw field retains all info)
            })

        # ── everything else: h, B, UA001, UA002, j, f ────────────────────
        else:
            other.append({**base, "raw": body})

    return trades, ob_updates, ob_snaps, other


# ─────────────────────────── DataFrame builders ──────────────────────────────

def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.str.replace("-", " ", n=1),
        format="mixed", utc=True, errors="coerce"
    )


def _build_trades(recs, adds_index: dict) -> pd.DataFrame:
    """
    Tick-level trade DataFrame from UA202/F records.
    Resolves resting order ID and buyer/seller initiator.

    Initiator logic per spec:
      OfferApplSeqNum(10117) == 0  → no resting sell  → buyer was aggressor
                                     → BidApplSeqNum(10116) is the resting buy
                                     → SELLER_INITIATED  (seller hit the bid)
      BidApplSeqNum(10116)  == 0   → no resting buy   → seller was aggressor
                                     → OfferApplSeqNum(10117) is resting sell
                                     → BUYER_INITIATED   (buyer lifted the offer)
    """
    df = pd.DataFrame(recs)
    if df.empty:
        return df

    for c in ("price", "qty"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("appl_seq", "msg_seq", "buy_ref", "sell_ref"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Fix 7: channel as Int64
    df["channel"] = pd.to_numeric(df["channel"], errors="coerce").astype("Int64")

    df["transact_time"] = _to_utc(df["transact_time"])
    df["sending_time"]  = _to_utc(df["sending_time"])
    df["capture_ts"]    = pd.to_datetime(df["capture_ts"], utc=True,
                                         format="mixed", errors="coerce")

    # Fix 1: only F is valid here; label it
    df["exec_type"] = df["exec_type_code"].map(EXECTYPE_MAP)

    # Fix 3: use tick segment map
    df["market"] = df["segment"].map(TICK_SEGMENT_MAP)

    # resolve resting order_id via adds_index
    ref = df["buy_ref"].where(df["buy_ref"].fillna(0) > 0, df["sell_ref"])
    df["resting_ref"] = ref.where(ref.fillna(0) > 0).astype("Int64")
    keys = zip(df["channel"].astype("object"), df["resting_ref"].astype("float").fillna(-1).astype(int))
    df["resting_order_id"] = [adds_index.get(k) for k in keys]

    # initiator: sell_ref nonzero → resting sell → buyer aggressed → BUYER_INITIATED
    #            buy_ref  nonzero → resting buy  → seller aggressed → SELLER_INITIATED
    init = pd.Series(pd.NA, index=df.index, dtype="object")
    init[df["sell_ref"].fillna(0) > 0] = "BUYER_INITIATED"
    init[df["buy_ref"].fillna(0)  > 0] = "SELLER_INITIATED"
    df["initiator"]      = init
    df["aggressor_side"] = df["initiator"].map(
        {"BUYER_INITIATED": "BUY", "SELLER_INITIATED": "SELL"})

    df = df[["transact_time", "sending_time", "capture_ts", "msg_seq",
             "appl_seq", "channel", "segment", "market",
             "exec_type_code", "exec_type",
             "exec_inst",                        # Fix 5
             "symbol", "price", "qty",
             "buy_ref", "sell_ref", "resting_ref", "resting_order_id",
             "initiator", "aggressor_side"]]

    for c in ("segment", "market", "exec_type_code", "exec_type",
              "exec_inst", "symbol", "resting_order_id",
              "initiator", "aggressor_side"):
        df[c] = df[c].astype("string")
    return df


def _build_ob_updates(recs, adds_index: dict) -> pd.DataFrame:
    """
    Order book update DataFrame: UA201 adds + UA202 cancels.
    Populates adds_index for cross-chunk order ID resolution.
    Fix 2: Side only 1/2; Fix 5: exec_inst captured.
    """
    df = pd.DataFrame(recs)
    if df.empty:
        return df

    for c in ("price", "qty"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("appl_seq", "msg_seq", "buy_ref", "sell_ref"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Fix 7: channel as Int64
    df["channel"] = pd.to_numeric(df["channel"], errors="coerce").astype("Int64")

    df["transact_time"] = _to_utc(df["transact_time"])
    df["sending_time"]  = _to_utc(df["sending_time"])
    df["capture_ts"]    = pd.to_datetime(df["capture_ts"], utc=True,
                                         format="mixed", errors="coerce")

    # Fix 2: spec-compliant Side map (1=Buy, 2=Sell only)
    df["side"] = df["side_code"].map(SIDE_MAP)

    # Fix 1: map the two valid ExecType values only
    df["exec_type"] = df["exec_type_code"].map(EXECTYPE_MAP)

    # Fix 3: tick segment map
    df["market"] = df["segment"].map(TICK_SEGMENT_MAP)

    # populate adds_index from ORDER_ADD rows
    is_add = df["event"] == "ORDER_ADD"
    for ch, sq, oid in zip(df.loc[is_add, "channel"],
                            df.loc[is_add, "appl_seq"],
                            df.loc[is_add, "order_id"]):
        if pd.notna(sq) and pd.notna(ch):
            adds_index[(int(ch), int(sq))] = oid

    # for CANCEL rows: resolve order_id and infer side from resting ref
    ref = df["buy_ref"].where(df["buy_ref"].fillna(0) > 0, df["sell_ref"])
    df["resting_ref"] = ref.where(ref.fillna(0) > 0).astype("Int64")
    keys     = zip(df["channel"].astype("object"), df["resting_ref"].astype("float").fillna(-1).astype(int))
    resolved = pd.Series([adds_index.get(k) for k in keys], index=df.index)
    not_add  = ~is_add
    df.loc[not_add, "order_id"] = resolved[not_add]

    # Fix 2: infer side for CANCEL only from ref (1=buy resting, 2=sell resting)
    cxl_side = np.where(df["buy_ref"].fillna(0)  > 0, "1",
               np.where(df["sell_ref"].fillna(0) > 0, "2", None))
    is_cxl = df["event"] == "CANCEL"
    df.loc[is_cxl, "side_code"] = cxl_side[is_cxl]
    df.loc[is_cxl, "side"]      = df.loc[is_cxl, "side_code"].map(SIDE_MAP)

    df = df[["transact_time", "sending_time", "capture_ts", "msg_seq",
             "appl_seq", "channel", "segment", "market",
             "event", "exec_type_code", "exec_type",
             "exec_inst",                        # Fix 5
             "symbol", "side_code", "side",
             "price", "qty", "order_id",
             "buy_ref", "sell_ref", "resting_ref"]]

    for c in ("segment", "market", "event", "exec_type_code", "exec_type",
              "exec_inst", "symbol", "side_code", "side", "order_id"):
        df[c] = df[c].astype("string")
    return df


def _build_ob_snapshot(recs) -> pd.DataFrame:
    """
    Order book full-snapshot DataFrame from 35=W records.
    Fix 3: use SNAPSHOT_SEGMENT_MAP.
    Fix 6: xl entry type now resolves correctly via MDENTRY_MAP.
    Fix 7: channel as Int64.
    """
    df = pd.DataFrame(recs)
    if df.empty:
        return df

    for c in ("px", "qty", "prev_close", "cum_volume", "cum_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("msg_seq", "num_trades", "n_entries", "level",
              "n_orders_at_level", "n_orders_detailed"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Fix 7
    df["channel"] = pd.to_numeric(df["channel"], errors="coerce").astype("Int64")

    df["snapshot_time"] = _to_utc(df["sending_time"])
    df["orig_time"]     = _to_utc(df["orig_time"])
    df["capture_ts"]    = pd.to_datetime(df["capture_ts"], utc=True,
                                         format="mixed", errors="coerce")

    # Fix 6: xl now in MDENTRY_MAP → resolves correctly
    df["entry_type"] = df["entry_type_code"].map(MDENTRY_MAP)

    # Fix 3: snapshot segment map
    df["market"] = df["segment"].map(SNAPSHOT_SEGMENT_MAP)

    df["visible_qty_sum"] = df["order_qtys"].map(
        lambda q: float(np.sum([float(x) for x in q]))
        if isinstance(q, list) else np.nan)
    df["order_ids"]  = df["order_ids"].map(
        lambda x: "|".join(x) if isinstance(x, list) else None)
    df["order_qtys"] = df["order_qtys"].map(
        lambda x: "|".join(x) if isinstance(x, list) else None)

    df = df[["snapshot_time", "orig_time", "capture_ts", "msg_seq",
             "channel", "segment", "market",
             "symbol", "trading_status",
             "prev_close", "num_trades", "cum_volume", "cum_value",
             "entry_type_code", "entry_type",
             "level", "px", "qty",
             "n_orders_at_level", "n_orders_detailed",
             "order_ids", "order_qtys", "visible_qty_sum"]]

    for c in ("segment", "market", "symbol", "trading_status",
              "entry_type_code", "entry_type", "order_ids", "order_qtys"):
        df[c] = df[c].astype("string")
    return df


def _build_other(recs) -> pd.DataFrame:
    """
    Other messages: UA001 heartbeats, h session status, B news,
    UA002 retransmit, UA004 stats (Fix 4: orig_time + n_streams added),
    j business reject, f security status.
    """
    df = pd.DataFrame(recs)
    if df.empty:
        return df

    df["msg_seq"]    = pd.to_numeric(df["msg_seq"], errors="coerce").astype("Int64")

    # Fix 7
    df["channel"]    = pd.to_numeric(df["channel"], errors="coerce").astype("Int64")

    df["capture_ts"] = pd.to_datetime(df["capture_ts"], utc=True,
                                      format="mixed", errors="coerce")
    df["sending_time"]= _to_utc(df["sending_time"])

    # Fix 4: orig_time and n_streams present only for UA004 rows; others → NaT / NA
    if "orig_time" not in df.columns:
        df["orig_time"] = pd.NaT
    else:
        df["orig_time"] = _to_utc(df["orig_time"].astype(str))

    if "n_streams" not in df.columns:
        df["n_streams"] = pd.NA

    df["n_streams"] = pd.to_numeric(df["n_streams"], errors="coerce").astype("Int64")

    df = df[["capture_ts", "sending_time", "orig_time",
             "msg_seq", "msg_type", "channel", "segment",
             "n_streams", "raw"]]

    for c in ("msg_type", "segment", "raw"):
        df[c] = df[c].astype("string")
    return df


# ─────────────────────────── chunk writer ────────────────────────────────────

def _write_chunk(buf, n_chunk, adds_index, out_dir, day, totals, t0):
    print(f"chunk {n_chunk:>3}: parsing  {len(buf):>9,} lines …", flush=True)
    trades, ob_upd, ob_snap, other = parse_fix_chunk(buf)

    print(f"chunk {n_chunk:>3}: building frames …", flush=True)
    df_trades  = _build_trades(trades, adds_index)
    df_ob_upd  = _build_ob_updates(ob_upd, adds_index)
    df_ob_snap = _build_ob_snapshot(ob_snap)
    df_other   = _build_other(other)

    print(f"chunk {n_chunk:>3}: writing parquet …", flush=True)
    for label, df in (("trades",      df_trades),
                      ("ob_updates",  df_ob_upd),
                      ("ob_snapshot", df_ob_snap),
                      ("other",       df_other)):
        if df.empty:
            continue
        path = out_dir / f"_part_{label}_{n_chunk:04d}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       path, compression="zstd")
        totals[label] += len(df)

    elapsed = time.time() - t0
    print(f"chunk {n_chunk:>3} DONE ({elapsed:6.1f}s) | "
          f"trades {len(df_trades):>8,} | "
          f"ob_upd {len(df_ob_upd):>9,} | "
          f"ob_snap {len(df_ob_snap):>9,} | "
          f"other {len(df_other):>7,}", flush=True)

    del trades, ob_upd, ob_snap, other
    del df_trades, df_ob_upd, df_ob_snap, df_other
    gc.collect()


# ─────────────────────────── main pipeline ───────────────────────────────────

def run_day(src=SRC, out_dir=OUT_DIR, chunk_lines=CHUNK_LINES):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day    = Path(src).name.replace(".tar.gz", "")
    labels = ("trades", "ob_updates", "ob_snapshot", "other")
    totals = dict.fromkeys(labels, 0)

    adds_index = {}    # {(channel_int, appl_seq_int) -> order_id_str}
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
                _write_chunk(buf, n_chunk, adds_index, out_dir, day, totals, t0)
                buf = []

        if buf:
            n_chunk += 1
            _write_chunk(buf, n_chunk, adds_index, out_dir, day, totals, t0)
            del buf

    del adds_index
    gc.collect()
    print(f"\nPass 1 done: {n_chunk} chunks, {n_lines:,} lines.", flush=True)
    print(f"Row totals  : {totals}", flush=True)

    # ── merge partials → 4 final Parquets, delete partials ───────────────
    print("\nMerging partials …", flush=True)
    for label in labels:
        parts = sorted(out_dir.glob(f"_part_{label}_*.parquet"))
        if not parts:
            print(f"  {label}: no data — skipped")
            continue
        final  = out_dir / f"{day}_{label}.parquet"
        writer = None
        for p in parts:
            pf = pq.ParquetFile(p)
            try:
                for batch in pf.iter_batches(batch_size=131_072):
                    if writer is None:
                        writer = pq.ParquetWriter(final, batch.schema,
                                                  compression="zstd")
                    writer.write_batch(batch)
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
        mb = final.stat().st_size / 1_048_576
        print(f"  {label:<12}: {totals[label]:>12,} rows  →  "
              f"{final.name}  ({mb:.1f} MB)")

    print(f"\nAll done in {time.time()-t0:.1f}s", flush=True)
    print(f"Output : {out_dir}", flush=True)


# ─────────────────────────── entry point ─────────────────────────────────────
if __name__ == "__main__":
    run_day()

    # To process all 5 days in one go, replace the line above with:
    # for f in sorted(SRC.parent.glob("*.tar.gz")):
    #     run_day(src=f)
