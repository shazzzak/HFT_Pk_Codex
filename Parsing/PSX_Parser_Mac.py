# To run in CMD
#.backtest\Scripts\python.exe Perplexity_psx_day_to_parquet.py

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
# CHANGELOG (this revision, on top of the previous 7-fix version):
#   Fix 8  - Auction crosses: both BidApplSeqNum(10116) & OfferApplSeqNum(10117)
#            nonzero -> initiator = AUCTION, resting_ref = NULL (was ill-posed
#            under the old "whichever ref nonzero" rule during phase O/N/V)
#   Fix 9  - Cancel price resolved from the referenced UA201 via adds_index,
#            instead of trusting tag 31 (LastPx), which is 0.0000 on cancels
#            (spec marks it optional/trade-price; not meaningful for cancels)
#   Fix 10 - exec_inst documented/enforced as cancel-and-trade-only; UA201 has
#            no tag 18 in the spec, so it is structurally always null on adds
#   Fix 11 - MDEntryType dictionary corrected against spec text:
#            xa=PREV_CLOSE_INDEX, xb=OPEN_INDEX, xc=HIGH_INDEX, xd=LOW_INDEX,
#            5=CLOSING_PRICE, 6=SETTLEMENT_PRICE, x5/x6=PE_RATIO_1/2,
#            x7=FUND_PREV_NAV, x8=ETF_INAV, xg=OPEN_INTEREST, xl=CLOSE_INDEX
#   Fix 12 - Circuit breaker sentinels handled asymmetrically per spec:
#            xe (up limit) sentinel 999999999.9999 -> NULL
#            xf (down limit) has NO large sentinel; its "no-limit" value is the
#            market's price tick size (e.g. 0.01 for Regular Market) and is
#            NOT nulled automatically — flagged via is_xf_tick_floor instead
#   Fix 13 - TradingPhaseCode(8538) decoded into phase / suspended_all_day /
#            break_reason columns instead of only storing the raw C8 string
#   Fix 14 - after-hours (phase A) snapshot rows keep level/order-count
#            columns nullable (spec: only 269/270/271 released in phase A)
#   Fix 15 - UA001 heartbeat fields promoted to first-class columns:
#            appl_last_seq(1350), end_of_channel(10205), heartbeat_time(60)
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
import psutil, os
proc = psutil.Process(os.getpid())

import sqlite3

import resource

class DiskBackedIndex:
    """
    Bounded-memory replacement for the plain adds_index dict. Keeps a hot
    LRU-style cache in memory and spills everything else to SQLite, so
    per-chunk memory growth stops being a function of total resting-order
    count for the day. Fixes the unbounded growth seen at chunks 1-5
    (43k -> 410k entries and climbing).
    """
    def __init__(self, db_path, hot_cache_size=200_000):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS idx (k TEXT PRIMARY KEY, oid TEXT, px REAL)"
        )
        self.hot = {}
        self.hot_cache_size = hot_cache_size

    def __setitem__(self, key, value):
        self.hot[key] = value
        if len(self.hot) > self.hot_cache_size:
            self._flush()

    def get(self, key, default=(None, None)):
        if key in self.hot:
            return self.hot[key]
        row = self.conn.execute("SELECT oid, px FROM idx WHERE k=?", (str(key),)).fetchone()
        return row if row else default

    def __contains__(self, key):
        return key in self.hot or self.get(key) != (None, None)

    def __len__(self):
        count = self.conn.execute("SELECT COUNT(*) FROM idx").fetchone()[0]
        return count + len(self.hot)

    def _flush(self):
        rows = [(str(k), v[0], v[1]) for k, v in self.hot.items()]
        self.conn.executemany("INSERT OR REPLACE INTO idx (k, oid, px) VALUES (?, ?, ?)", rows)
        self.conn.commit()
        self.hot.clear()

    def close(self):
        self._flush()
        self.conn.close()


# ─────────────────────────── configuration ───────────────────────────────────
IN_DIR      = Path("/Users/shazzak/Library/CloudStorage/"
                   "GoogleDrive-shazzak@gmail.com/My Drive/Capital Stake")
#PARSED_ROOT = Path("/Users/shazzak/Library/CloudStorage/GoogleDrive-shazzak@gmail.com/My Drive/Capital Stake - Parsed")
PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")

LABELS      = ("trades", "ob_updates", "ob_snapshot", "misc")
CHUNK_LINES = 250_000      # lines per batch; lower to 100_000 if RAM is tight

SOH = "^"

# spec Side(54) valid values are 1=Buy, 2=Sell only in market data feed
SIDE_MAP = {
    "1": "BUY",
    "2": "SELL",
}

# spec ExecType(150) valid values are 4=Cancelled and F=Trade only
EXECTYPE_MAP = {
    "4": "CANCELLED",
    "F": "TRADE",
}

# separate MDStreamID maps for Snapshot (35=W) vs Tick (UA201/UA202)
SNAPSHOT_SEGMENT_MAP = {
    "010": "REG",
    "020": "BILLS_BOND",
    "030": "STOCK_DEL_FUT",
    "040": "STOCK_CS_FUT",
    "050": "STOCK_DEL_OPT",
    "060": "INDEX_OPT",
    "070": "STOCK_IDX_FUT",
    "080": "ODD_LOT",
    "100": "EQ_SQUARE_UP",
    "120": "FUT_SQUARE_UP",
    "900": "INDEX",
}

TICK_SEGMENT_MAP = {
    "011": "REG",
    "031": "STOCK_DEL_FUT",
    "041": "STOCK_CS_FUT",
    "051": "STOCK_DEL_OPT",
    "061": "INDEX_OPT",
    "071": "STOCK_IDX_FUT",
    "081": "ODD_LOT",
    "091": "NDM",
}

# Fix 11: MDEntryType dictionary corrected against spec text verbatim
MDENTRY_MAP = {
    "0":  "BID",
    "1":  "OFFER",
    "2":  "LAST_TRADE",
    "3":  "INDEX_VALUE",
    "4":  "OPENING_PRICE",
    "5":  "CLOSING_PRICE",          # Fix 11
    "6":  "SETTLEMENT_PRICE",       # Fix 11
    "7":  "SESSION_HIGH",
    "8":  "SESSION_LOW",
    "x1": "NET_CHANGE_1",           # latest px minus prev close
    "x2": "NET_CHANGE_2",           # latest px minus last latest px
    "x3": "AGG_BID",                # VWAP px / total qty within auction range
    "x4": "AGG_OFFER",              # VWAP px / total qty within auction range
    "x5": "PE_RATIO_1",             # Fix 11: reserved, not released
    "x6": "PE_RATIO_2",             # Fix 11: reserved, not released
    "x7": "FUND_PREV_NAV",          # Fix 11: fund prev NAV incl. ETF
    "x8": "ETF_INAV",               # Fix 11: ETF intraday NAV
    "xa": "PREV_CLOSE_INDEX",       # Fix 11: corrected (was INDEX_OPEN)
    "xb": "OPEN_INDEX",             # Fix 11: corrected
    "xc": "HIGH_INDEX",             # Fix 11: corrected (was INDEX_LOW_?)
    "xd": "LOW_INDEX",              # Fix 11: corrected
    "xe": "UPPER_CIRCUIT_BREAKER",
    "xf": "LOWER_CIRCUIT_BREAKER",
    "xg": "OPEN_INTEREST",          # position qty of derivative contract
    "xl": "CLOSE_INDEX",
}

# Fix 12: sentinel value for "no limit" on xe (up circuit breaker) ONLY.
# xf (down circuit breaker) has no universal sentinel — its no-limit value
# equals the market's minimum price tick (e.g. 0.01 for Regular Market),
# which varies by market/segment and must not be hard-coded/nulled blindly.
XE_NO_LIMIT_SENTINEL = 999999999.9999

# Fix 13: TradingPhaseCode(8538) 0th-digit phase code -> human label
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

# Fix 13: break-reason 2nd digit (only meaningful when phase == B)
BREAK_REASON_MAP = {
    "1": "AFTER_PRE_OPEN",
    "2": "FRIDAY_LUNCH_BREAK",
    "3": "AFTER_PRE_OPEN_PM_FRIDAY",
    "4": "BEFORE_POST_CLOSE",
}


# ─────────────────────── canonical schemas (Fix 16) ───────────────────────

TRADES_RAW_COLS = {
    "capture_ts": "datetime64[ns, UTC]", "msg_seq": "Int64",
    "sending_time": "datetime64[ns, UTC]", "channel": "Int64",
    "segment": "string", "appl_seq": "Int64", "symbol": "string",
    "price": "float64", "qty": "float64",
    "transact_time": "datetime64[ns, UTC]",
    "exec_type_code": "string", "exec_inst": "string",
    "buy_ref": "Int64", "sell_ref": "Int64",
}

TRADES_FINAL_COLS = {
    "transact_time": "datetime64[ns, UTC]", "sending_time": "datetime64[ns, UTC]",
    "capture_ts": "datetime64[ns, UTC]", "msg_seq": "Int64", "appl_seq": "Int64",
    "channel": "Int64", "segment": "string", "market": "string",
    "exec_type_code": "string", "exec_type": "string", "exec_inst": "string",
    "symbol": "string", "price": "float64", "qty": "float64",
    "buy_ref": "Int64", "sell_ref": "Int64", "resting_ref": "Int64",
    "resting_order_id": "string", "initiator": "string", "aggressor_side": "string",
}

OB_UPDATES_RAW_COLS = {
    "capture_ts": "datetime64[ns, UTC]", "msg_seq": "Int64",
    "sending_time": "datetime64[ns, UTC]", "channel": "Int64",
    "segment": "string", "event": "string", "appl_seq": "Int64",
    "symbol": "string", "side_code": "string", "price": "float64",
    "qty": "float64", "order_id": "string",
    "transact_time": "datetime64[ns, UTC]",
    "exec_type_code": "string", "exec_inst": "string",
    "buy_ref": "Int64", "sell_ref": "Int64", "raw_last_px": "float64",
}

OB_UPDATES_FINAL_COLS = {
    "transact_time": "datetime64[ns, UTC]", "sending_time": "datetime64[ns, UTC]",
    "capture_ts": "datetime64[ns, UTC]", "msg_seq": "Int64", "appl_seq": "Int64",
    "channel": "Int64", "segment": "string", "market": "string",
    "event": "string", "exec_type_code": "string", "exec_type": "string",
    "exec_inst": "string", "symbol": "string", "side_code": "string",
    "side": "string", "price": "float64", "qty": "float64", "order_id": "string",
    "buy_ref": "Int64", "sell_ref": "Int64", "resting_ref": "Int64",
    "raw_last_px": "float64",
}

OB_SNAPSHOT_RAW_COLS = {
    "capture_ts": "datetime64[ns, UTC]", "msg_seq": "Int64",
    "sending_time": "datetime64[ns, UTC]", "channel": "Int64",
    "segment": "string", "orig_time": "datetime64[ns, UTC]",
    "symbol": "string", "trading_status": "string",
    "prev_close": "float64", "num_trades": "Int64",
    "cum_volume": "float64", "cum_value": "float64", "n_entries": "Int64",
    "entry_type_code": "string", "px": "float64", "qty": "float64",
    "level": "Int64", "n_orders_at_level": "Int64", "n_orders_detailed": "Int64",
    "order_ids": "object", "order_qtys": "object",
}

OB_SNAPSHOT_FINAL_COLS = {
    "snapshot_time": "datetime64[ns, UTC]", "orig_time": "datetime64[ns, UTC]",
    "capture_ts": "datetime64[ns, UTC]", "msg_seq": "Int64",
    "channel": "Int64", "segment": "string", "market": "string",
    "symbol": "string", "trading_status": "string", "phase": "string",
    "suspended_all_day": "boolean", "break_reason": "string",
    "prev_close": "float64", "num_trades": "Int64",
    "cum_volume": "float64", "cum_value": "float64",
    "entry_type_code": "string", "entry_type": "string",
    "level": "Int64", "px": "float64", "qty": "float64",
    "n_orders_at_level": "Int64", "n_orders_detailed": "Int64",
    "order_ids": "string", "order_qtys": "string", "visible_qty_sum": "float64",
    "is_xf_tick_floor": "boolean",
}

OTHER_RAW_COLS = {
    "capture_ts": "datetime64[ns, UTC]", "msg_seq": "Int64",
    "sending_time": "datetime64[ns, UTC]", "msg_type": "string",
    "channel": "Int64", "segment": "string", "raw": "string",
    "orig_time": "datetime64[ns, UTC]", "n_streams": "Int64",
    "appl_last_seq": "Int64", "end_of_channel": "Int64",
    "heartbeat_time": "datetime64[ns, UTC]",
}

OTHER_FINAL_COLS = {
    "capture_ts": "datetime64[ns, UTC]", "sending_time": "datetime64[ns, UTC]",
    "orig_time": "datetime64[ns, UTC]", "msg_seq": "Int64", "msg_type": "string",
    "channel": "Int64", "segment": "string", "n_streams": "Int64",
    "appl_last_seq": "Int64", "end_of_channel": "Int64",
    "heartbeat_time": "datetime64[ns, UTC]", "raw": "string",
}


def _ensure_cols(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """
    Ensure all columns in `cols` exist with the given dtype.
    `cols` maps column_name -> dtype string (e.g. "Int64", "float64", "string").
    """
    for c, dtype in cols.items():
        if c not in df.columns:
            # Column missing: create with appropriate nulls
            if dtype in ("Int64", "Float64", "boolean", "string"):
                df[c] = pd.Series(pd.NA, index=df.index, dtype=dtype)
            elif dtype == "float64":
                df[c] = pd.Series(np.nan, index=df.index, dtype=dtype)
            else:
                # fallback: treat other numeric vs non-numeric
                if dtype.startswith("float") or dtype.startswith("int"):
                    df[c] = pd.Series(np.nan, index=df.index, dtype=dtype)
                else:
                    df[c] = pd.Series(pd.NA, index=df.index, dtype=dtype)
        else:
            # Column exists: enforce dtype safely
            if dtype in ("Int64", "Float64"):
                # coerce bad literals (e.g. 'N') to NA before casting
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(dtype)
            elif dtype == "float64":
                df[c] = pd.to_numeric(df[c], errors="coerce")
            elif dtype in ("boolean", "string"):
                df[c] = df[c].astype(dtype)
            else:
                # generic fallback, still ok for object/string-like
                df[c] = df[c].astype(dtype)

    return df


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
                "exec_inst":     None,   # Fix 10: UA201 has no tag 18 in spec
                "buy_ref":       None,
                "sell_ref":      None,
            })

        # ── UA202: Tick Execution → trade or cancel ───────────────────────
        elif mtype == "UA202":
            et        = msg.get("150")
            exec_inst = msg.get("18")

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
                    "exec_inst":     exec_inst,
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
                    # Fix 9: tag 31 is unreliable (0.0000) on cancels; keep the
                    # raw value here so build step can compare vs resolved px,
                    # but the FINAL "price" column is resolved from adds_index
                    "raw_last_px":   msg.get("31"),
                    "qty":           msg.get("32"),
                    "order_id":      None,
                    "transact_time": msg.get("60"),
                    "exec_type_code":"4",
                    "exec_inst":     exec_inst,
                    "buy_ref":       msg.get("10116"),
                    "sell_ref":      msg.get("10117"),
                })
            else:
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

        elif mtype == "UA004":
            other.append({
                **base,
                "raw":       body,
                "orig_time": msg.get("42"),
                "n_streams": msg.get("10208"),
            })

        # Fix 15: UA001 heartbeat fields promoted
        elif mtype == "UA001":
            other.append({
                **base,
                "raw":              body,
                "appl_last_seq":    msg.get("1350"),
                "end_of_channel":   msg.get("10205"),
                "heartbeat_time":   msg.get("60"),
            })

        # ── everything else: h, B, UA002, j, f ───────────────────────────
        else:
            other.append({**base, "raw": body})

    return trades, ob_updates, ob_snaps, other


# ─────────────────────────── DataFrame builders ──────────────────────────────

def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype(str).str.replace("-", " ", n=1),
        format="mixed", utc=True, errors="coerce"
    )


def _build_trades(recs, adds_index: dict) -> pd.DataFrame:
    """
    Tick-level trade DataFrame from UA202/F records.

    Initiator logic (Fix 8 — handles auction crosses):
      both buy_ref & sell_ref > 0  -> AUCTION (call-auction cross matches
                                       two resting orders; ref rule is
                                       ill-posed here) -> resting_ref = NULL
      sell_ref > 0, buy_ref == 0   -> resting sell order  -> buyer aggressed
                                       -> BUYER_INITIATED
      buy_ref  > 0, sell_ref == 0  -> resting buy order   -> seller aggressed
                                       -> SELLER_INITIATED
    """
    df = pd.DataFrame(recs)
    if df.empty:
        return _ensure_cols(df, TRADES_FINAL_COLS)
    df = _ensure_cols(df, TRADES_RAW_COLS)

    for c in ("price", "qty"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("appl_seq", "msg_seq", "buy_ref", "sell_ref"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["channel"] = pd.to_numeric(df["channel"], errors="coerce").astype("Int64")

    df["transact_time"] = _to_utc(df["transact_time"])
    df["sending_time"]  = _to_utc(df["sending_time"])
    df["capture_ts"]    = pd.to_datetime(df["capture_ts"], utc=True,
                                         format="mixed", errors="coerce")

    df["exec_type"] = df["exec_type_code"].map(EXECTYPE_MAP)
    df["market"]    = df["segment"].map(TICK_SEGMENT_MAP)

    buy_pos  = df["buy_ref"].fillna(0)  > 0
    sell_pos = df["sell_ref"].fillna(0) > 0
    is_auction = buy_pos & sell_pos                       # Fix 8

    ref = df["buy_ref"].where(buy_pos, df["sell_ref"])
    df["resting_ref"] = ref.where(ref.fillna(0) > 0).astype("Int64")
    df.loc[is_auction, "resting_ref"] = pd.NA             # Fix 8

    keys = zip(df["channel"].astype("object"),
               df["resting_ref"].astype("float").fillna(-1).astype(int))

    # PRODUCTION FIX: Extract index [0] to avoid Tuple injection.
    # Use pd.NA instead of None to ensure zero-overhead casting to string[pyarrow] later.
    # We explicitly use .get() here (not .pop()) because trades can be PARTIAL fills;
    # popping the order here would break subsequent partial fills on the same order.
    df["resting_order_id"] = [
        adds_index.get(k, (pd.NA, None))[0]
        for k in keys
    ]

    df.loc[is_auction, "resting_order_id"] = pd.NA        # Fix 8

    init = pd.Series(pd.NA, index=df.index, dtype="object")
    init[sell_pos & ~is_auction] = "BUYER_INITIATED"
    init[buy_pos  & ~is_auction] = "SELLER_INITIATED"
    init[is_auction]             = "AUCTION"              # Fix 8
    df["initiator"] = init

    df["aggressor_side"] = df["initiator"].map(
        {"BUYER_INITIATED": "BUY", "SELLER_INITIATED": "SELL",
         "AUCTION": "AUCTION"})

    df = df[list(TRADES_FINAL_COLS.keys())]

    for c, dtype in TRADES_FINAL_COLS.items():
        if dtype in ("string", "Int64", "boolean", "float64"):
            df[c] = df[c].astype(dtype)

    return df


def _build_ob_updates(recs, adds_index: dict) -> pd.DataFrame:
    """
    Order book update DataFrame: UA201 adds + UA202 cancels.

    Fix 9: cancel row's "price" is resolved from the referenced UA201 order
           (via adds_index price cache) instead of trusting tag 31, which is
           empirically 0.0000 on cancels and not spec-guaranteed meaningful.
    Fix 10: exec_inst is structurally None for ORDER_ADD rows (UA201 has no
            tag 18 per spec) — enforced explicitly here, not just inherited.
    """
    df = pd.DataFrame(recs)
    if df.empty:
        return _ensure_cols(df, OB_UPDATES_FINAL_COLS)
    df = _ensure_cols(df, OB_UPDATES_RAW_COLS)

    # keep raw_last_px separate; do not use as final price for cancels
    # ALWAYS create this column (even if this chunk has zero cancels) so
    # every ob_updates chunk has an identical schema for the Parquet merge.
    if "raw_last_px" not in df.columns:
        df["raw_last_px"] = np.nan
    df["raw_last_px"] = pd.to_numeric(df["raw_last_px"], errors="coerce")
    has_raw_px = True

    for c in ("price", "qty"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("appl_seq", "msg_seq", "buy_ref", "sell_ref"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["channel"] = pd.to_numeric(df["channel"], errors="coerce").astype("Int64")

    df["transact_time"] = _to_utc(df["transact_time"])
    df["sending_time"]  = _to_utc(df["sending_time"])
    df["capture_ts"]    = pd.to_datetime(df["capture_ts"], utc=True,
                                         format="mixed", errors="coerce")

    df["side"]      = df["side_code"].map(SIDE_MAP)
    df["exec_type"] = df["exec_type_code"].map(EXECTYPE_MAP)
    df["market"]    = df["segment"].map(TICK_SEGMENT_MAP)

    is_add = df["event"] == "ORDER_ADD"

    # Fix 10: enforce exec_inst is always null on ORDER_ADD rows
    df.loc[is_add, "exec_inst"] = pd.NA

    # populate adds_index with (order_id, price) so cancels can resolve both
    mask = is_add & df["channel"].notna() & df["appl_seq"].notna()

    for ch, sq, oid, px in zip(
            df.loc[mask, "channel"].astype(int),
            df.loc[mask, "appl_seq"].astype(int),
            df.loc[mask, "order_id"],
            df.loc[mask, "price"],
    ):
        adds_index[(ch, sq)] = (oid, px)

    ref = df["buy_ref"].where(df["buy_ref"].fillna(0) > 0, df["sell_ref"])
    df["resting_ref"] = ref.where(ref.fillna(0) > 0).astype("Int64")
    keys = list(zip(df["channel"].astype("object"),
                    df["resting_ref"].astype("float").fillna(-1).astype(int)))

    resolved_oid = pd.Series(
        [adds_index.get(k, (None, None))[0] for k in keys], index=df.index)
    resolved_px = pd.Series(
        [adds_index.get(k, (None, None))[1] for k in keys], index=df.index)

    not_add = ~is_add
    df.loc[not_add, "order_id"] = resolved_oid[not_add]

    # Fix 9: overwrite cancel price with the resolved order price
    is_cxl = df["event"] == "CANCEL"
    df.loc[is_cxl, "price"] = resolved_px[is_cxl]

    cxl_side = np.where(df["buy_ref"].fillna(0)  > 0, "1",
               np.where(df["sell_ref"].fillna(0) > 0, "2", None))
    df.loc[is_cxl, "side_code"] = cxl_side[is_cxl]
    df.loc[is_cxl, "side"]      = df.loc[is_cxl, "side_code"].map(SIDE_MAP)

    df = df[list(OB_UPDATES_FINAL_COLS.keys())]

    for c, dtype in OB_UPDATES_FINAL_COLS.items():
        if dtype in ("string", "Int64", "boolean", "float64"):
            df[c] = df[c].astype(dtype)

    return df


def _build_ob_snapshot(recs) -> pd.DataFrame:
    """
    Order book full-snapshot DataFrame from 35=W records.

    Fix 11: corrected MDENTRY_MAP applied.
    Fix 12: xe sentinel (999999999.9999) -> px NULL; xf is NOT auto-nulled
            (no universal sentinel — flagged via is_xf_tick_floor instead).
    Fix 13: TradingPhaseCode decoded into phase / suspended_all_day /
            break_reason.
    Fix 14: level/order-count columns remain nullable for phase A rows
            (already nullable by construction; documented here).
    """
    df = pd.DataFrame(recs)
    if df.empty:
        return _ensure_cols(df, OB_SNAPSHOT_FINAL_COLS)
    df = _ensure_cols(df, OB_SNAPSHOT_RAW_COLS)

    for c in ("px", "qty", "prev_close", "cum_volume", "cum_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("msg_seq", "num_trades", "n_entries", "level",
              "n_orders_at_level", "n_orders_detailed"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["channel"] = pd.to_numeric(df["channel"], errors="coerce").astype("Int64")

    df["snapshot_time"] = _to_utc(df["sending_time"])
    df["orig_time"]     = _to_utc(df["orig_time"])
    df["capture_ts"]    = pd.to_datetime(df["capture_ts"], utc=True,
                                         format="mixed", errors="coerce")

    df["entry_type"] = df["entry_type_code"].map(MDENTRY_MAP)   # Fix 11
    df["market"]     = df["segment"].map(SNAPSHOT_SEGMENT_MAP)

    # Fix 12: xe no-limit sentinel -> NULL. xf intentionally left untouched;
    # instead flag rows where px looks like a tick-size floor (heuristic:
    # px <= 1.0 for xf entries) so downstream users can decide per market.
    is_xe = df["entry_type_code"] == "xe"
    df.loc[is_xe & (df["px"] == XE_NO_LIMIT_SENTINEL), "px"] = np.nan

    is_xf = df["entry_type_code"] == "xf"
    df["is_xf_tick_floor"] = pd.NA
    df.loc[is_xf, "is_xf_tick_floor"] = df.loc[is_xf, "px"] <= 1.0

    # Fix 13: decode TradingPhaseCode(8538) -> phase / suspended / break
    ts = df["trading_status"].astype(str)
    phase_char     = ts.str.slice(0, 1)
    suspended_char = ts.str.slice(1, 2)
    break_char     = ts.str.slice(2, 3)

    df["phase"]             = phase_char.map(PHASE_MAP).astype("string")
    df["suspended_all_day"] = suspended_char.map({"1": True, "0": False})
    df["break_reason"]      = np.where(
        phase_char == "B", break_char.map(BREAK_REASON_MAP), None)
    df["break_reason"] = df["break_reason"].astype("string")

    # Fix 14: level/order-count columns already Int64 (nullable) — after-hour
    # (phase A) rows will naturally carry <NA> since spec releases only
    # 269/270/271 in that phase and the source message omits 1023/346/73.

    df["visible_qty_sum"] = df["order_qtys"].map(
        lambda q: float(np.sum([float(x) for x in q]))
        if isinstance(q, list) else np.nan)
    df["order_ids"]  = df["order_ids"].map(
        lambda x: "|".join(x) if isinstance(x, list) else None)
    df["order_qtys"] = df["order_qtys"].map(
        lambda x: "|".join(x) if isinstance(x, list) else None)

    df = df[list(OB_SNAPSHOT_FINAL_COLS.keys())]

    for c, dtype in OB_SNAPSHOT_FINAL_COLS.items():
        if dtype in ("string", "Int64", "boolean", "float64"):
            df[c] = df[c].astype(dtype)

    return df


def _build_other(recs) -> pd.DataFrame:
    """
    Other messages: UA001 heartbeats (Fix 15: fields promoted),
    h session status, B news, UA002 retransmit, UA004 stats,
    j business reject, f security status.
    """
    df = pd.DataFrame(recs)
    if df.empty:
        return _ensure_cols(df, OTHER_FINAL_COLS)
    df = _ensure_cols(df, OTHER_RAW_COLS)

    df["msg_seq"] = pd.to_numeric(df["msg_seq"], errors="coerce").astype("Int64")
    df["channel"] = pd.to_numeric(df["channel"], errors="coerce").astype("Int64")

    df["capture_ts"]   = pd.to_datetime(df["capture_ts"], utc=True,
                                        format="mixed", errors="coerce")
    df["sending_time"] = _to_utc(df["sending_time"])

    if "orig_time" not in df.columns:
        df["orig_time"] = pd.NaT
    else:
        df["orig_time"] = _to_utc(df["orig_time"])

    if "n_streams" not in df.columns:
        df["n_streams"] = pd.NA
    df["n_streams"] = pd.to_numeric(df["n_streams"], errors="coerce").astype("Int64")

    # Fix 15: UA001 heartbeat fields
    for col in ("appl_last_seq", "end_of_channel", "heartbeat_time"):
        if col not in df.columns:
            df[col] = pd.NA

    df["appl_last_seq"]  = pd.to_numeric(df["appl_last_seq"], errors="coerce").astype("Int64")
    df["end_of_channel"] = pd.to_numeric(df["end_of_channel"], errors="coerce").astype("Int64")
    df["heartbeat_time"] = _to_utc(df["heartbeat_time"])

    df = df[list(OTHER_FINAL_COLS.keys())]

    for c, dtype in OTHER_FINAL_COLS.items():
        if dtype in ("string", "Int64", "boolean", "float64"):
            df[c] = df[c].astype(dtype)

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
                      ("misc",        df_other)):
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

def run_day(src, parsed_root=PARSED_ROOT, chunk_lines=CHUNK_LINES):
    day     = Path(src).name.replace(".tar.gz", "").replace(".txt", "")
    out_dir = Path(parsed_root) / "_tmp" / day     # partials live here
    out_dir.mkdir(parents=True, exist_ok=True)
    labels  = LABELS
    totals  = dict.fromkeys(labels, 0)

    # adds_index now stores (order_id, price) tuples — needed for Fix 9
    #adds_index = {}
    adds_index = DiskBackedIndex(out_dir / f"{day}_adds_index.sqlite")
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

                proc = psutil.Process()
                rss_mb = proc.memory_info().rss / (1024 ** 2)  # live current RSS, not historical peak
                vm = psutil.virtual_memory()
                print(f"  [chunk {n_chunk}] adds_index size={len(adds_index):,} entries, "
                      f"live RSS={rss_mb:,.0f} MB, system available={vm.available / (1024 ** 2):,.0f} MB, "
                      f"swap used={psutil.swap_memory().used / (1024 ** 2):,.0f} MB")

                _write_chunk(buf, n_chunk, adds_index, out_dir, day, totals, t0)
                buf = []

        if buf:
            n_chunk += 1
            _write_chunk(buf, n_chunk, adds_index, out_dir, day, totals, t0)
            del buf

    adds_index.close()
    del adds_index


    gc.collect()
    print(f"\nPass 1 done: {n_chunk} chunks, {n_lines:,} lines.", flush=True)
    print(f"Row totals  : {totals}", flush=True)
    print(f"    RSS {proc.memory_info().rss / 1e9:.2f} GB")

    print("\nMerging partials …", flush=True)
    for label in labels:
        parts = sorted(out_dir.glob(f"_part_{label}_*.parquet"))
        if not parts:
            print(f"  {label}: no data — skipped")
            continue
        final_dir = Path(parsed_root) / label / f"date={day}"
        final_dir.mkdir(parents=True, exist_ok=True)
        final = final_dir / f"{day}_{label}.parquet"

        # Schema-tolerant merge: union all part schemas first (promote_options
        # handles missing/added columns and mismatched nullability across
        # chunks), then re-write every part's batches against that unified
        # schema. This prevents ValueError crashes if any chunk had a column
        # a neighboring chunk lacked (e.g. an all-adds chunk with no cancels).
        schemas = [pq.ParquetFile(p).schema_arrow for p in parts]
        try:
            unified = pa.unify_schemas(schemas, promote_options="permissive")
        except TypeError:
            # older pyarrow without promote_options kwarg
            unified = pa.unify_schemas(schemas)

        writer = None
        for p in parts:
            pf = pq.ParquetFile(p)
            try:
                for batch in pf.iter_batches(batch_size=131_072):
                    tbl = pa.Table.from_batches([batch]).cast(unified)
                    if writer is None:
                        writer = pq.ParquetWriter(final, unified,
                                                  compression="zstd")
                    writer.write_table(tbl)
            finally:
                pf.close()
        if writer:
            writer.close()

        for p in parts:
            try:
                p.unlink()
            except PermissionError:
                print(f"  WARNING: could not delete {p.name} "
                      f"(cloud-sync lock?) — delete manually")
        mb = final.stat().st_size / 1_048_576
        print(f"  {label:<12}: {totals[label]:>12,} rows  →  "
              f"{final.name}  ({mb:.1f} MB)")

    try:                                   # remove empty tmp dir
        out_dir.rmdir()
    except OSError:
        pass
    print(f"\nAll done in {time.time()-t0:.1f}s", flush=True)
    print(f"Output : {parsed_root}/<table>/date={day}/", flush=True)


# ─────────────────────────── entry point ─────────────────────────────────────
def day_outputs_exist(day: str) -> bool:
    return all((PARSED_ROOT / lbl / f"date={day}" / f"{day}_{lbl}.parquet").exists()
               for lbl in LABELS)


if __name__ == "__main__":
    files = sorted(IN_DIR.glob("*.tar.gz"))          # get names, sort, loop
    print(f"{len(files)} day file(s) found in {IN_DIR}")
    for i, f in enumerate(files, 1):
        day = f.name.replace(".tar.gz", "")
        if day_outputs_exist(day):
            print(f"[{i}/{len(files)}] {day}: all 4 outputs exist — skipping")
            continue
        print(f"\n[{i}/{len(files)}] ===== {day} =====")
        run_day(src=f)
    print("\nALL DAYS DONE")

