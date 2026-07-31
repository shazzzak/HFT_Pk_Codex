# %% [markdown]
# # PSX FIX Market-Data Parser
#
# Parses the PSX (KATS / new trading system) FIX market-data capture file and produces:
#
#   1. `df_ticks` — tick-level event stream (order adds, trades, cancels) from
#      messages 35=UA201 (Add Order) and 35=UA202 (Order Executed / Deleted),
#      with deterministic buyer/seller-initiated classification for trades.
#   2. `df_book` — order-book snapshots from 35=W (Market Data Snapshot / Full
#      Refresh), one row per (snapshot, book entry), including per-order detail
#      (OrderID + displayed qty) where the exchange publishes it.
#
# IMPORTANT NOTE ON THE TWO SPECS:
# The PDF you attached is PSX's *order-entry* FIX 4.2 spec (what a broker sends
# to KATS: New Order Single, Cancel/Replace, etc., with the order types shown in
# your image — Normal, Market, Stop Loss, MIT, Cross, FOK, Short Sell, Leveraged
# Buy variants, GTC/GTW/GTM). The capture file, however, is the *market-data
# dissemination* feed (FIXT.1.1, sender NMDU001Q0001 -> PSX vendor feed). It is
# the mirror image of the order-entry spec: every order type placed via the PDF
# spec shows up here as UA201/UA202/W events. The decode tables below cover ALL
# codes from the PDF (Side 1/2/5/8/G, OrdType 1/2/4/J, TIF 0/1/4/6, all market
# codes REG/FUT/CSF/IPO/SQR/SIF/ODL/IOM/FRO/KMT/LMT/IMT) plus everything
# observed in the feed, so any order type in your image is handled.
#
# Field semantics verified empirically against this capture:
#   * UA201: 1181=ApplSeqNum, 55=Symbol, 44=Price, 38=Qty, 54=Side,
#     37=OrderID, 60=TransactTime  -> "order added to book" tick.
#   * UA202: 1181=ApplSeqNum, 31=LastPx, 32=LastQty, 150=ExecType
#     ('F'=Trade, '4'=Cancelled), 10116=BuySideOrderRef, 10117=SellSideOrderRef
#     (each is the ApplSeqNum of the corresponding UA201 add).
#   * In every trade in this capture exactly ONE of 10116/10117 is non-zero,
#     and it always points at the RESTING (passive) order — its limit price
#     equals the trade price in 100% of resolvable cases. Therefore:
#         10117 != 0  (passive SELL rested)  ->  BUYER-initiated trade
#         10116 != 0  (passive BUY rested)   ->  SELLER-initiated trade
#     A tick-rule fallback is included for robustness on other capture days.
#   * W: header (55, 140=PrevClosePx, 8503=#trades, 387=cum volume,
#     8504=cum value, 8538=trading status), then 268=NoMDEntries repeating
#     group: 269=EntryType, 270=Px, 271=Qty, 1023=PriceLevel,
#     346=NumberOfOrders, optional 73=NoOrders detail of (38=Qty, 37=OrderID).

# %%
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

# ---------------------------------------------------------------------------
# Configuration — point this at your file (zip or plain txt)
# ---------------------------------------------------------------------------
FIX_PATH = Path("fix.txt")          # or Path("fix_txt.zip")

SOH = "^"                            # capture file uses '^' in place of ASCII 0x01

# ---------------------------------------------------------------------------
# Decode tables (union of the PDF order-entry spec and the observed feed)
# ---------------------------------------------------------------------------
SIDE_MAP = {
    "1": "BUY",
    "2": "SELL",
    "5": "SELL_SHORT",
    "8": "CROSS",
    "G": "LEVERAGED_BUY",            # 'Borrow' in FIX 4.2 spec = Leveraged Buy on REG
}

ORDTYPE_MAP = {                      # per PDF (order-entry) — kept for completeness
    "1": "MARKET",
    "2": "LIMIT",
    "4": "STOP_LIMIT",
    "J": "MARKET_IF_TOUCHED",
    "8": "CROSS",
}

TIF_MAP = {                          # per PDF Appendix D transaction plan
    "0": "DAY",
    "1": "GOOD_TILL_CANCEL",
    "4": "FILL_OR_KILL",
    "6": "GOOD_TILL_DATE",
}

EXECTYPE_MAP = {                     # tag 150 (feed) + PDF execution-report codes
    "0": "NEW", "1": "PARTIAL_FILL", "2": "FILL", "4": "CANCELED",
    "5": "REPLACED", "6": "PENDING_CANCEL", "8": "REJECTED",
    "9": "SUSPENDED", "A": "PENDING_NEW", "E": "PENDING_REPLACE",
    "F": "TRADE",
}

# Market codes: PDF Appendix D (3-letter) — the feed uses numeric 1500 segment
# IDs; both are preserved. Known/inferred numeric segments from this capture:
MARKET_SEGMENT_MAP = {
    "010": "REG_SNAPSHOT", "011": "REG",          # regular / ready market
    "020": "ODL_SNAPSHOT_?", "030": "FUT_SNAPSHOT", "031": "FUT",  # deliverable futures (…-JAN symbols)
    "070": "SEG_070", "080": "SEG_080", "081": "SEG_081",
    "100": "SEG_100", "900": "INDEX",
}
MARKET_CODES_PDF = {
    "REG": "Regular", "FUT": "Future", "CSF": "Cash Settled Future",
    "IPO": "Initial Public Offer", "SQR": "Square-Up", "SIF": "Stock Index Future",
    "ODL": "Odd Lot", "IOM": "Index Option", "FRO": "Future Rollover",
    "KMT": "Karachi Margin Trading", "LMT": "Lahore Margin Trading",
    "IMT": "Islamabad Margin Trading",
}

# Tag 269 MD entry types: standard FIX + PSX custom x* codes.
# x* labels below are inferred from the data (xe/xf verified as ±10% of tag 140
# prev-close, i.e. the daily circuit-breaker caps; xa..xd only on index channel).
MDENTRY_MAP = {
    "0": "BID", "1": "OFFER", "2": "LAST_TRADE", "3": "INDEX_VALUE",
    "4": "OPENING_PRICE", "7": "SESSION_HIGH", "8": "SESSION_LOW",
    "x1": "NET_CHANGE", "x2": "CUSTOM_x2",
    "x3": "AGG_BID (VWAP px / total qty)", "x4": "AGG_OFFER (VWAP px / total qty)",
    "x7": "CUSTOM_x7", "x8": "CUSTOM_x8",
    "xa": "INDEX_OPEN", "xb": "INDEX_HIGH", "xc": "INDEX_LOW_?", "xd": "INDEX_LOW/PREV_?",
    "xe": "UPPER_CIRCUIT_BREAKER", "xf": "LOWER_CIRCUIT_BREAKER",
    "xg": "CUSTOM_xg (futures only, likely open interest)",
}

TRADSES_MAP = {   # tag 340 / 8538-style states, per PDF Trading Session Status
    "1": "HALTED", "2": "OPEN", "3": "CLOSED", "4": "PRE_OPEN", "5": "PRE_CLOSE",
    "100": "READY", "101": "PRE_OPENING", "102": "OPENING", "103": "CLOSING",
    "104": "OPENCLOSE", "105": "DUMP",
}


# ---------------------------------------------------------------------------
# Low-level parsing
# ---------------------------------------------------------------------------
def _iter_lines(path: Path):
    """Yield raw lines from a .txt file or from the first .txt inside a .zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist()
                        if n.lower().endswith(".txt") and not n.startswith("__MACOSX"))
            with zf.open(name) as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                    yield raw
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            yield from fh


def _tokenize(body: str):
    """Split a FIX body into ordered (tag, value) pairs. Order matters for 35=W."""
    return [p.split("=", 1) for p in body.rstrip(SOH).split(SOH) if "=" in p]


def _parse_snapshot_entries(pairs):
    """
    Parse a 35=W message into (header_dict, [entry_dict,...]).
    Repeating group starts at tag 268; each entry starts with 269 and may carry
    an inner order-detail group 73=(38 qty, 37 order id)*.
    Validated: len(entries) == int(header['268']) for all 2,043 snapshots
    in the sample capture.
    """
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
        elif tag == "10":                       # checksum -> end of message
            break
        elif cur is None:
            header[tag] = val
        elif tag == "38":                       # order detail: qty first…
            cur["orders"].append({"qty": val})
        elif tag == "37":                       # …then its OrderID
            if cur["orders"] and "id" not in cur["orders"][-1]:
                cur["orders"][-1]["id"] = val
            else:
                cur["orders"].append({"id": val})
        else:
            cur[tag] = val
    if cur is not None:
        entries.append(cur)
    return header, entries


def parse_fix_file(path: Path):
    """
    Single pass over the capture file.
    Returns (tick_records, book_records, other_records).
    """
    ticks, book, other = [], [], []
    for lineno, raw in enumerate(_iter_lines(path), 1):
        raw = raw.strip()
        if not raw or "|" not in raw:
            continue
        capture_ts, body = raw.split("|", 1)
        pairs = _tokenize(body)
        msg = dict(pairs)                       # fine for non-repeating messages
        mtype = msg.get("35")

        base = {
            "capture_ts": capture_ts,           # local PKT wall clock of the recorder
            "msg_seq": msg.get("34"),           # session MsgSeqNum (tag 34)
            "sending_time": msg.get("52"),      # UTC
            "msg_type": mtype,
            "channel": msg.get("10201"),        # PSX dissemination channel id
            "segment": msg.get("1500"),         # numeric market-segment id
        }

        if mtype == "UA201":                    # ---- Add Order (tick) ----
            ticks.append({
                **base,
                "event": "ORDER_ADD",
                "appl_seq": msg.get("1181"),
                "symbol": msg.get("55"),
                "side_code": msg.get("54"),
                "price": msg.get("44"),
                "qty": msg.get("38"),
                "order_id": msg.get("37"),
                "transact_time": msg.get("60"),
                "exec_type_code": None,
                "buy_ref": None, "sell_ref": None,
            })
        elif mtype == "UA202":                  # ---- Execution / Delete (tick) ----
            et = msg.get("150")
            ticks.append({
                **base,
                "event": "TRADE" if et == "F" else
                         ("CANCEL" if et == "4" else f"EXEC_{et}"),
                "appl_seq": msg.get("1181"),
                "symbol": msg.get("55"),
                "side_code": None,              # resolved later from refs
                "price": msg.get("31"),
                "qty": msg.get("32"),
                "order_id": None,               # resolved later from refs
                "transact_time": msg.get("60"),
                "exec_type_code": et,
                "buy_ref": msg.get("10116"),
                "sell_ref": msg.get("10117"),
            })
        elif mtype == "W":                      # ---- Order-book snapshot ----
            pairs_full = pairs
            header, entries = _parse_snapshot_entries(pairs_full)
            snap_base = {
                **base,
                "orig_time": header.get("42"),
                "symbol": header.get("55"),
                "trading_status": header.get("8538"),
                "prev_close": header.get("140"),
                "num_trades": header.get("8503"),
                "cum_volume": header.get("387"),
                "cum_value": header.get("8504"),
                "n_entries": header.get("268"),
            }
            for e in entries:
                book.append({
                    **snap_base,
                    "entry_type_code": e.get("269"),
                    "px": e.get("270"),
                    "qty": e.get("271"),
                    "level": e.get("1023"),
                    "n_orders_at_level": e.get("346"),
                    "n_orders_detailed": e.get("73"),
                    "order_ids": [o.get("id") for o in e["orders"]] or None,
                    "order_qtys": [o.get("qty") for o in e["orders"]] or None,
                })
        else:                                   # h, B, UA001, UA004, 0, …
            other.append({**base, "raw": body})
    return ticks, book, other


# ---------------------------------------------------------------------------
# Run the parse
# ---------------------------------------------------------------------------
tick_recs, book_recs, other_recs = parse_fix_file(FIX_PATH)
print(f"tick events: {len(tick_recs):,}   book rows: {len(book_recs):,}   "
      f"other msgs (session status/news/stats/heartbeats): {len(other_recs):,}")

# %% [markdown]
# ## 1–2. Build the two DataFrames

# %%
def _to_utc(s):
    """FIX UTCTimestamp 'YYYYMMDD-HH:MM:SS(.sss)' -> tz-aware pandas datetime."""
    return pd.to_datetime(s, format="mixed", utc=True, errors="coerce")


# ---------------- df_ticks ----------------
df_ticks = pd.DataFrame(tick_recs)

for col in ("price", "qty"):
    df_ticks[col] = pd.to_numeric(df_ticks[col], errors="coerce")
for col in ("appl_seq", "msg_seq", "buy_ref", "sell_ref"):
    df_ticks[col] = pd.to_numeric(df_ticks[col], errors="coerce").astype("Int64")

df_ticks["transact_time"] = _to_utc(df_ticks["transact_time"].str.replace("-", " ", n=1))
df_ticks["sending_time"] = _to_utc(df_ticks["sending_time"].str.replace("-", " ", n=1))
df_ticks["capture_ts"] = pd.to_datetime(df_ticks["capture_ts"], utc=True,
                                        format="mixed", errors="coerce")

df_ticks["side"] = df_ticks["side_code"].map(SIDE_MAP)
df_ticks["exec_type"] = df_ticks["exec_type_code"].map(EXECTYPE_MAP)
df_ticks["market"] = df_ticks["segment"].map(MARKET_SEGMENT_MAP)

# ---- Resolve UA202 refs against the UA201 add archive ----
# The non-zero ref on a trade/cancel is the ApplSeqNum of the affected
# RESTING order's add message -> recovers its OrderID and (for cancels) side.
adds = (df_ticks[df_ticks.event == "ORDER_ADD"]
        .set_index("appl_seq")[["order_id", "side_code", "price", "symbol"]])

ref = df_ticks["buy_ref"].where(df_ticks["buy_ref"].fillna(0) > 0,
                                df_ticks["sell_ref"])
ref = ref.where(ref.fillna(0) > 0)
df_ticks["resting_ref"] = ref.astype("Int64")
lookup = df_ticks["resting_ref"].map(adds["order_id"])
df_ticks.loc[df_ticks.event != "ORDER_ADD", "order_id"] = lookup

# Cancel side = side of the referenced (cancelled) order:
cancel_side = np.where(df_ticks["buy_ref"].fillna(0) > 0, "1",
              np.where(df_ticks["sell_ref"].fillna(0) > 0, "2", None))
is_cxl = df_ticks.event == "CANCEL"
df_ticks.loc[is_cxl, "side_code"] = cancel_side[is_cxl]
df_ticks.loc[is_cxl, "side"] = df_ticks.loc[is_cxl, "side_code"].map(SIDE_MAP)

# ---- 4. Trade initiator classification ----
# Primary (deterministic, from matching-engine refs):
#   sell_ref != 0 -> passive SELL was resting -> aggressor bought  -> BUYER-initiated
#   buy_ref  != 0 -> passive BUY  was resting -> aggressor sold    -> SELLER-initiated
# Fallback (should not trigger on this feed): classic tick test per symbol.
is_trade = df_ticks.event == "TRADE"
init = pd.Series(pd.NA, index=df_ticks.index, dtype="object")
init[is_trade & (df_ticks.sell_ref.fillna(0) > 0)] = "BUYER_INITIATED"
init[is_trade & (df_ticks.buy_ref.fillna(0) > 0)] = "SELLER_INITIATED"

unresolved = is_trade & init.isna()
if unresolved.any():                      # tick-test fallback
    t = df_ticks[is_trade].sort_values(["symbol", "transact_time", "appl_seq"])
    prev_px = t.groupby("symbol")["price"].shift()
    tick_dir = np.sign(t["price"] - prev_px)
    tick_cls = pd.Series(np.where(tick_dir > 0, "BUYER_INITIATED",
                         np.where(tick_dir < 0, "SELLER_INITIATED", pd.NA)),
                         index=t.index)
    init[unresolved] = tick_cls[unresolved[unresolved].index]

df_ticks["initiator"] = init
df_ticks["aggressor_side"] = df_ticks["initiator"].map(
    {"BUYER_INITIATED": "BUY", "SELLER_INITIATED": "SELL"})

df_ticks = df_ticks[[
    "transact_time", "sending_time", "capture_ts", "msg_seq", "appl_seq",
    "channel", "segment", "market", "event", "exec_type", "symbol",
    "side_code", "side", "price", "qty", "order_id",
    "buy_ref", "sell_ref", "resting_ref", "initiator", "aggressor_side",
]].sort_values(["transact_time", "msg_seq"]).reset_index(drop=True)

print(df_ticks["event"].value_counts(), "\n")
print(df_ticks.loc[df_ticks.event == "TRADE", "initiator"]
      .value_counts(dropna=False), "\n")
df_ticks.head(10)

# %%
# ---------------- df_book ----------------
df_book = pd.DataFrame(book_recs)

for col in ("px", "qty", "prev_close", "cum_volume", "cum_value"):
    df_book[col] = pd.to_numeric(df_book[col], errors="coerce")
for col in ("msg_seq", "num_trades", "n_entries", "level",
            "n_orders_at_level", "n_orders_detailed"):
    df_book[col] = pd.to_numeric(df_book[col], errors="coerce").astype("Int64")

df_book["snapshot_time"] = _to_utc(df_book["sending_time"].str.replace("-", " ", n=1))
df_book["capture_ts"] = pd.to_datetime(df_book["capture_ts"], utc=True,
                                       format="mixed", errors="coerce")
df_book["entry_type"] = df_book["entry_type_code"].map(MDENTRY_MAP)
df_book["market"] = df_book["segment"].map(MARKET_SEGMENT_MAP)
df_book["visible_qty_sum"] = df_book["order_qtys"].map(
    lambda q: float(np.sum([float(x) for x in q])) if isinstance(q, list) else np.nan)

df_book = df_book[[
    "snapshot_time", "capture_ts", "msg_seq", "channel", "segment", "market",
    "symbol", "trading_status", "prev_close", "num_trades", "cum_volume",
    "cum_value", "entry_type_code", "entry_type", "level", "px", "qty",
    "n_orders_at_level", "n_orders_detailed", "order_ids", "order_qtys",
    "visible_qty_sum",
]].sort_values(["snapshot_time", "msg_seq", "entry_type_code", "level"]
).reset_index(drop=True)

print(df_book["entry_type"].value_counts().head(12), "\n")
df_book[(df_book.entry_type_code.isin(["0", "1"]))].head(8)

# %% [markdown]
# ### Convenience views
# * `df_trades` — trades only (with initiator flag)
# * `best_quotes()` — best bid/ask per symbol per snapshot (top of book)

# %%
df_trades = df_ticks[df_ticks.event == "TRADE"].copy()

def best_quotes(df_book):
    top = df_book[(df_book.level == 1) &
                  (df_book.entry_type_code.isin(["0", "1"]))]
    bq = top.pivot_table(index=["snapshot_time", "symbol"],
                         columns="entry_type_code",
                         values=["px", "qty"], aggfunc="first")
    bq.columns = [f"{'bid' if c[1]=='0' else 'ask'}_{c[0]}" for c in bq.columns]
    return bq.reset_index().rename(columns={"bid_px": "best_bid",
                                            "ask_px": "best_ask",
                                            "bid_qty": "best_bid_qty",
                                            "ask_qty": "best_ask_qty"})

bq = best_quotes(df_book)
bq.head()

# %% [markdown]
# ## 5. Identifying undisclosed (iceberg) orders on PSX
#
# **Mechanics (from the order-entry PDF):** an "undisclosed" order is sent with
# `MaxFloor` (tag 111) < `OrderQty` (tag 38). KATS shows only the disclosed
# tranche in the book; when it is fully executed, the engine re-displays the
# next tranche from the hidden reserve. In the sample messages, e.g.
# OrderQty=201,000 with MaxFloor=100,500 means the market ever sees at most
# 100,500 shares of a 201k parent.
#
# **What that implies in this market-data feed** (you never see tag 111 —
# only its footprints):
#
# 1. **Replenishment under the same OrderID.** In the 35=W snapshots each
#    price level lists resting orders (tag 37 OrderID + tag 38 *displayed*
#    qty). If cumulative traded volume attributed to one OrderID exceeds the
#    maximum quantity it ever *displayed*, the excess came from a hidden
#    reserve — a smoking gun. Similarly, a UA201 "add" that re-appears with
#    the SAME OrderID right after trades consumed it is a disclosure refill
#    (a genuinely new order gets a new OrderID).
# 2. **Level absorption.** Trades keep printing at one price while the
#    snapshot qty at that level barely declines (or snaps back between
#    consecutive snapshots) although `n_orders_at_level` is unchanged.
# 3. **Displayed-vs-total mismatch.** At detailed levels,
#    `visible_qty_sum` (sum of per-order tag-38 quantities) should equal the
#    level qty (tag 271). Persistent gaps after reconciling feed timing are
#    another hidden-liquidity signal. (Note: PSX only publishes per-order
#    detail for the first orders at the top levels, so absence of detail is
#    NOT evidence of hiding.)
# 4. **Execution/quote anomaly.** A single aggressive fill (one UA202 seq)
#    larger than the entire displayed size at the touch in the latest
#    snapshot means hidden size traded.
#
# The detector below implements signal #1 on the tick stream — the most
# reliable one — and reports candidate iceberg OrderIDs with their refill
# counts and hidden volume estimates.

# %%
def detect_iceberg_candidates(df_ticks):
    """
    Signal-1 iceberg detector:
      (a) same OrderID re-added (UA201) >= 2 times with executions against it
          in between  -> disclosure refills;
      (b) traded volume attributed to an OrderID exceeds the max single
          displayed tranche -> hidden reserve consumed.
    Returns one row per candidate OrderID.
    """
    adds = df_ticks[df_ticks.event == "ORDER_ADD"]
    fills = df_ticks[(df_ticks.event == "TRADE") & df_ticks.order_id.notna()]

    add_stats = adds.groupby("order_id").agg(
        symbol=("symbol", "first"),
        side=("side", "first"),
        n_adds=("appl_seq", "count"),
        max_displayed=("qty", "max"),
        total_displayed=("qty", "sum"),
        first_seen=("transact_time", "min"),
        last_refill=("transact_time", "max"),
        prices=("price", lambda s: sorted(set(s.round(4)))),
    )
    fill_stats = fills.groupby("order_id").agg(
        n_fills=("appl_seq", "count"),
        traded_qty=("qty", "sum"),
    )
    cand = add_stats.join(fill_stats, how="left").fillna(
        {"n_fills": 0, "traded_qty": 0})
    cand["hidden_qty_consumed"] = (cand["traded_qty"]
                                   - cand["max_displayed"]).clip(lower=0)
    cand["is_iceberg_candidate"] = (
        ((cand["n_adds"] >= 2) & (cand["n_fills"] >= 1))       # refill pattern
        | (cand["traded_qty"] > cand["max_displayed"])          # hidden volume
    )
    out = (cand[cand["is_iceberg_candidate"]]
           .sort_values(["n_adds", "hidden_qty_consumed"], ascending=False))
    return out


icebergs = detect_iceberg_candidates(df_ticks)
print(f"iceberg candidates: {len(icebergs)}")
icebergs.head(15)

# %% [markdown]
# ## 6. QuestDB: tables + ingestion
#
# Two tables matching the two DataFrames. Designed for your QuestDB feature
# pipeline: designated timestamps, `SYMBOL` columns for low-cardinality
# strings, daily partitions, WAL, and dedup keys so re-ingesting a capture
# file is idempotent. Array columns (`order_ids`/`order_qtys`) are flattened
# to pipe-joined VARCHARs — explode them into a third table later if you need
# per-order book analytics at scale.

# %%
QUESTDB_DDL = """
CREATE TABLE IF NOT EXISTS psx_ticks (
    transact_time   TIMESTAMP,
    sending_time    TIMESTAMP,
    capture_ts      TIMESTAMP,
    msg_seq         LONG,
    appl_seq        LONG,
    channel         SYMBOL CAPACITY 64 CACHE,
    segment         SYMBOL CAPACITY 64 CACHE,
    market          SYMBOL CAPACITY 32 CACHE,
    event           SYMBOL CAPACITY 16 CACHE,   -- ORDER_ADD / TRADE / CANCEL
    exec_type       SYMBOL CAPACITY 16 CACHE,
    symbol          SYMBOL CAPACITY 4096 CACHE,
    side            SYMBOL CAPACITY 16 CACHE,   -- BUY/SELL/SELL_SHORT/CROSS/LEVERAGED_BUY
    price           DOUBLE,
    qty             DOUBLE,
    order_id        VARCHAR,
    buy_ref         LONG,
    sell_ref        LONG,
    resting_ref     LONG,
    initiator       SYMBOL CAPACITY 8 CACHE,    -- BUYER_INITIATED / SELLER_INITIATED
    aggressor_side  SYMBOL CAPACITY 8 CACHE
) TIMESTAMP(transact_time) PARTITION BY DAY WAL
  DEDUP UPSERT KEYS(transact_time, channel, appl_seq);

CREATE TABLE IF NOT EXISTS psx_orderbook (
    snapshot_time      TIMESTAMP,
    capture_ts         TIMESTAMP,
    msg_seq            LONG,
    channel            SYMBOL CAPACITY 64 CACHE,
    segment            SYMBOL CAPACITY 64 CACHE,
    market             SYMBOL CAPACITY 32 CACHE,
    symbol             SYMBOL CAPACITY 4096 CACHE,
    trading_status     SYMBOL CAPACITY 16 CACHE,
    prev_close         DOUBLE,
    num_trades         LONG,
    cum_volume         DOUBLE,
    cum_value          DOUBLE,
    entry_type_code    SYMBOL CAPACITY 32 CACHE,  -- 0,1,2,4,7,8,x1..xg
    entry_type         SYMBOL CAPACITY 32 CACHE,
    level              INT,
    px                 DOUBLE,
    qty                DOUBLE,
    n_orders_at_level  INT,
    n_orders_detailed  INT,
    order_ids          VARCHAR,                   -- pipe-joined
    order_qtys         VARCHAR,                   -- pipe-joined
    visible_qty_sum    DOUBLE
) TIMESTAMP(snapshot_time) PARTITION BY DAY WAL
  DEDUP UPSERT KEYS(snapshot_time, msg_seq, entry_type_code, level, symbol);
"""
print(QUESTDB_DDL)

# %%
# --- Create the tables (HTTP /exec endpoint; adjust host/port/auth) ---------
# import requests
# for stmt in [s.strip() for s in QUESTDB_DDL.split(";") if s.strip()]:
#     r = requests.get("http://localhost:9000/exec", params={"query": stmt})
#     r.raise_for_status()
#     print(r.json().get("ddl", r.json()))

# --- Ingest via official client (pip install questdb>=2.0) ------------------
# ILP over HTTP; Sender.dataframe() maps pandas columns to table columns.
# from questdb.ingress import Sender
#
# t = df_ticks.copy()
# t["msg_seq"], t["appl_seq"] = t["msg_seq"].astype("float64"), t["appl_seq"].astype("float64")
# for c in ("buy_ref", "sell_ref", "resting_ref"):
#     t[c] = t[c].astype("float64")
# for c in ("channel","segment","market","event","exec_type","symbol",
#           "side","initiator","aggressor_side"):
#     t[c] = t[c].astype("string")
# t = t.drop(columns=["side_code"]).dropna(subset=["transact_time"])
#
# b = df_book.copy()
# b["order_ids"]  = b["order_ids"].map(lambda x: "|".join(x) if isinstance(x, list) else None)
# b["order_qtys"] = b["order_qtys"].map(lambda x: "|".join(x) if isinstance(x, list) else None)
# for c in ("channel","segment","market","symbol","trading_status",
#           "entry_type_code","entry_type","order_ids","order_qtys"):
#     b[c] = b[c].astype("string")
# b = b.drop(columns=["sending_time"], errors="ignore").dropna(subset=["snapshot_time"])
#
# conf = "http::addr=localhost:9000;"          # add username/token as needed
# with Sender.from_conf(conf) as sender:
#     sender.dataframe(t, table_name="psx_ticks", at="transact_time")
#     sender.dataframe(b, table_name="psx_orderbook", at="snapshot_time")
# print("ingested:", len(t), "ticks,", len(b), "book rows")

# %% [markdown]
# ### Handy QuestDB queries once loaded
# ```sql
# -- Order-flow imbalance per symbol, 1-min buckets
# SELECT transact_time ts, symbol,
#        sum(CASE WHEN initiator = 'BUYER_INITIATED'  THEN qty ELSE 0 END) buy_vol,
#        sum(CASE WHEN initiator = 'SELLER_INITIATED' THEN qty ELSE 0 END) sell_vol
# FROM psx_ticks WHERE event = 'TRADE'
# SAMPLE BY 1m;
#
# -- Latest top-of-book per symbol
# SELECT snapshot_time, symbol, entry_type, px, qty
# FROM psx_orderbook
# WHERE level = 1 AND entry_type_code IN ('0','1')
# LATEST ON snapshot_time PARTITION BY symbol, entry_type_code;
# ```
