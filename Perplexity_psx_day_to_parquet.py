
# =============================================================================
# PSX FIX daily capture -> 4 Parquet files (chunked, bounded RAM)
#
# Tables produced:
#   {day}_trades.parquet      - UA202 exec_type=F (TRADE) only — tick level
#   {day}_ob_snapshot.parquet - 35=W full order book snapshots from exchange
#   {day}_ob_updates.parquet  - UA201 (add) + UA202 non-trade (cancel/modify)
#   {day}_other.parquet       - heartbeats, session status, news, all else
#
# Run:
#   python psx_day_to_parquet.py
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

SIDE_MAP = {
    "1": "BUY", "2": "SELL", "5": "SELL_SHORT",
    "8": "CROSS", "G": "LEVERAGED_BUY",
}
EXECTYPE_MAP = {
    "0": "NEW",           "1": "PARTIAL_FILL",  "2": "FILL",
    "4": "CANCELED",      "5": "REPLACED",      "6": "PENDING_CANCEL",
    "8": "REJECTED",      "9": "SUSPENDED",      "A": "PENDING_NEW",
    "E": "PENDING_REPLACE", "F": "TRADE",
}
MARKET_SEGMENT_MAP = {
    "010": "REG_SNAPSHOT", "011": "REG",    "020": "ODL",
    "030": "FUT_SNAPSHOT", "031": "FUT",    "070": "SEG_070",
    "080": "SEG_080",      "081": "SEG_081","100": "SEG_100",
    "900": "INDEX",
}
MDENTRY_MAP = {
    "0": "BID",             "1": "OFFER",           "2": "LAST_TRADE",
    "3": "INDEX_VALUE",     "4": "OPENING_PRICE",   "7": "SESSION_HIGH",
    "8": "SESSION_LOW",     "x1": "NET_CHANGE",     "x2": "CUSTOM_x2",
    "x3": "AGG_BID",        "x4": "AGG_OFFER",      "x7": "CUSTOM_x7",
    "x8": "CUSTOM_x8",      "xa": "INDEX_OPEN",     "xb": "INDEX_HIGH",
    "xc": "INDEX_LOW",      "xd": "INDEX_PREV_CLOSE",
    "xe": "UPPER_CIRCUIT_BREAKER", "xf": "LOWER_CIRCUIT_BREAKER",
    "xg": "OPEN_INTEREST",
}


# ─────────────────────────── low-level parsing ───────────────────────────────

def _tokenize(body: str):
    return [p.split("=", 1) for p in body.rstrip(SOH).split(SOH) if "=" in p]


def _parse_snapshot_entries(pairs):
    """Parse repeating group of a 35=W message."""
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
    trades     : UA202 with exec_type = F  (actual fills)
    ob_updates : UA201 (order add) + UA202 non-trade (cancel/modify/etc.)
    ob_snaps   : 35=W  full order-book snapshots from the exchange
    other      : everything else (heartbeats, session status, news, …)
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

        # ── UA201: Order Add → order book update ──────────────────────────
        if mtype == "UA201":
            ob_updates.append({
                **base,
                "event":          "ORDER_ADD",
                "appl_seq":       msg.get("1181"),
                "symbol":         msg.get("55"),
                "side_code":      msg.get("54"),
                "price":          msg.get("44"),
                "qty":            msg.get("38"),
                "order_id":       msg.get("37"),
                "transact_time":  msg.get("60"),
                "exec_type_code": None,
                "buy_ref":        None,
                "sell_ref":       None,
            })

        # ── UA202: split into trades vs. book updates ─────────────────────
        elif mtype == "UA202":
            et = msg.get("150")
            rec = {
                **base,
                "appl_seq":       msg.get("1181"),
                "symbol":         msg.get("55"),
                "price":          msg.get("31"),
                "qty":            msg.get("32"),
                "transact_time":  msg.get("60"),
                "exec_type_code": et,
                "buy_ref":        msg.get("10116"),
                "sell_ref":       msg.get("10117"),
            }

            if et == "F":
                # ── actual trade fill ──────────────────────────────────────
                trades.append({**rec, "event": "TRADE", "side_code": None})
            else:
                # ── cancel / modify / reject / etc → order book update ─────
                event = "CANCEL" if et == "4" else f"EXEC_{et}"
                ob_updates.append({**rec, "event": event, "side_code": None, "order_id": None})

        # ── 35=W: full order book snapshot ───────────────────────────────
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
                    "entry_type_code":  e.get("269"),
                    "px":               e.get("270"),
                    "qty":              e.get("271"),
                    "level":            e.get("1023"),
                    "n_orders_at_level":e.get("346"),
                    "n_orders_detailed":e.get("73"),
                    "order_ids":  [o.get("id")  for o in e["orders"]] or None,
                    "order_qtys": [o.get("qty") for o in e["orders"]] or None,
                })

        # ── everything else ───────────────────────────────────────────────
        else:
            other.append({**base, "raw": body})

    return trades, ob_updates, ob_snaps, other


# ─────────────────────────── DataFrame builders ───────────────────────────────

def _to_utc(series):
    return pd.to_datetime(
        series.str.replace("-", " ", n=1),
        format="mixed", utc=True, errors="coerce"
    )


def _build_trades(trade_recs, adds_index):
    """
    Build tick-level trade DataFrame from UA202/F records.
    Resolves buyer/seller initiator from the resting-order reference.
    """
    df = pd.DataFrame(trade_recs)
    if df.empty:
        return df

    for c in ("price", "qty"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("appl_seq", "msg_seq", "buy_ref", "sell_ref"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["transact_time"] = _to_utc(df["transact_time"])
    df["sending_time"]  = _to_utc(df["sending_time"])
    df["capture_ts"]    = pd.to_datetime(df["capture_ts"], utc=True,
                                         format="mixed", errors="coerce")
    df["exec_type"] = df["exec_type_code"].map(EXECTYPE_MAP)
    df["market"]    = df["segment"].map(MARKET_SEGMENT_MAP)

    # resolve order_id of the resting order (from adds_index built from UA201)
    ref = df["buy_ref"].where(df["buy_ref"].fillna(0) > 0, df["sell_ref"])
    df["resting_ref"] = ref.where(ref.fillna(0) > 0).astype("Int64")
    keys = zip(df["channel"], df["resting_ref"].astype("float").fillna(-1).astype(int))
    df["resting_order_id"] = [adds_index.get(k) for k in keys]

    # initiator logic:
    # sell_ref != 0  → passive SELL was resting → buyer aggressed → BUYER_INITIATED
    # buy_ref  != 0  → passive BUY  was resting → seller aggressed → SELLER_INITIATED
    init = pd.Series(pd.NA, index=df.index, dtype="object")
    init[df["sell_ref"].fillna(0) > 0] = "BUYER_INITIATED"
    init[df["buy_ref"].fillna(0)  > 0] = "SELLER_INITIATED"
    df["initiator"]     = init
    df["aggressor_side"]= df["initiator"].map(
        {"BUYER_INITIATED": "BUY", "SELLER_INITIATED": "SELL"})

    df = df[["transact_time", "sending_time", "capture_ts", "msg_seq",
             "appl_seq", "channel", "segment", "market",
             "exec_type_code", "exec_type", "symbol",
             "price", "qty",
             "buy_ref", "sell_ref", "resting_ref", "resting_order_id",
             "initiator", "aggressor_side"]]

    for c in ("channel", "segment", "market", "exec_type_code", "exec_type",
              "symbol", "resting_order_id", "initiator", "aggressor_side"):
        df[c] = df[c].astype("string")
    return df


def _build_ob_updates(upd_recs, adds_index):
    """
    Build order-book update DataFrame from UA201 (add) and UA202 non-trade.
    Also populates adds_index for cross-chunk order ID resolution.
    """
    df = pd.DataFrame(upd_recs)
    if df.empty:
        return df

    for c in ("price", "qty"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("appl_seq", "msg_seq", "buy_ref", "sell_ref"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["transact_time"] = _to_utc(df["transact_time"])
    df["sending_time"]  = _to_utc(df["sending_time"])
    df["capture_ts"]    = pd.to_datetime(df["capture_ts"], utc=True,
                                         format="mixed", errors="coerce")
    df["side"]      = df["side_code"].map(SIDE_MAP)
    df["exec_type"] = df["exec_type_code"].map(EXECTYPE_MAP)
    df["market"]    = df["segment"].map(MARKET_SEGMENT_MAP)

    # build adds_index from ORDER_ADD rows (for resolving refs in trades)
    is_add = df["event"] == "ORDER_ADD"
    for ch, sq, oid in zip(df.loc[is_add, "channel"],
                            df.loc[is_add, "appl_seq"],
                            df.loc[is_add, "order_id"]):
        if pd.notna(sq):
            adds_index[(ch, int(sq))] = oid

    # for CANCEL rows: resolve the order_id from the resting ref
    ref = df["buy_ref"].where(df["buy_ref"].fillna(0) > 0, df["sell_ref"])
    df["resting_ref"] = ref.where(ref.fillna(0) > 0).astype("Int64")
    keys    = zip(df["channel"], df["resting_ref"].astype("float").fillna(-1).astype(int))
    resolved= pd.Series([adds_index.get(k) for k in keys], index=df.index)
    not_add = ~is_add
    df.loc[not_add, "order_id"] = resolved[not_add]

    # infer side for CANCEL from whichever ref is nonzero
    cxl_side = np.where(df["buy_ref"].fillna(0) > 0, "1",
               np.where(df["sell_ref"].fillna(0) > 0, "2", None))
    is_cxl = df["event"] == "CANCEL"
    df.loc[is_cxl, "side_code"] = cxl_side[is_cxl]
    df.loc[is_cxl, "side"]      = df.loc[is_cxl, "side_code"].map(SIDE_MAP)

    df = df[["transact_time", "sending_time", "capture_ts", "msg_seq",
             "appl_seq", "channel", "segment", "market", "event",
             "exec_type_code", "exec_type", "symbol",
             "side_code", "side", "price", "qty",
             "order_id", "buy_ref", "sell_ref", "resting_ref"]]

    for c in ("channel", "segment", "market", "event", "exec_type_code",
              "exec_type", "symbol", "side_code", "side", "order_id"):
        df[c] = df[c].astype("string")
    return df


def _build_ob_snapshot(snap_recs):
    """Build order-book full-snapshot DataFrame from 35=W records."""
    df = pd.DataFrame(snap_recs)
    if df.empty:
        return df

    for c in ("px", "qty", "prev_close", "cum_volume", "cum_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("msg_seq", "num_trades", "n_entries", "level",
              "n_orders_at_level", "n_orders_detailed"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["snapshot_time"] = _to_utc(df["sending_time"])
    df["capture_ts"]    = pd.to_datetime(df["capture_ts"], utc=True,
                                         format="mixed", errors="coerce")
    df["entry_type"] = df["entry_type_code"].map(MDENTRY_MAP)
    df["market"]     = df["segment"].map(MARKET_SEGMENT_MAP)

    df["visible_qty_sum"] = df["order_qtys"].map(
        lambda q: float(np.sum([float(x) for x in q]))
        if isinstance(q, list) else np.nan)
    df["order_ids"]  = df["order_ids"].map(
        lambda x: "|".join(x) if isinstance(x, list) else None)
    df["order_qtys"] = df["order_qtys"].map(
        lambda x: "|".join(x) if isinstance(x, list) else None)

    df = df[["snapshot_time", "capture_ts", "msg_seq", "channel",
             "segment", "market", "symbol", "trading_status",
             "prev_close", "num_trades", "cum_volume", "cum_value",
             "entry_type_code", "entry_type", "level", "px", "qty",
             "n_orders_at_level", "n_orders_detailed",
             "order_ids", "order_qtys", "visible_qty_sum"]]

    for c in ("channel", "segment", "market", "symbol", "trading_status",
              "entry_type_code", "entry_type", "order_ids", "order_qtys"):
        df[c] = df[c].astype("string")
    return df


def _build_other(other_recs):
    """Build other-messages DataFrame (heartbeats, session, news, etc.)"""
    df = pd.DataFrame(other_recs)
    if df.empty:
        return df
    df["msg_seq"]      = pd.to_numeric(df["msg_seq"], errors="coerce").astype("Int64")
    df["capture_ts"]   = pd.to_datetime(df["capture_ts"], utc=True,
                                        format="mixed", errors="coerce")
    df["sending_time"] = _to_utc(df["sending_time"])
    df = df[["capture_ts", "sending_time", "msg_seq", "msg_type",
             "channel", "segment", "raw"]]
    for c in ("msg_type", "channel", "segment", "raw"):
        df[c] = df[c].astype("string")
    return df


# ─────────────────────────── chunk writer ────────────────────────────────────

def _write_chunk(buf, n_chunk, adds_index, out_dir, day, totals, t0):
    print(f"chunk {n_chunk:>3}: parsing  {len(buf):>9,} lines …", flush=True)
    trades, ob_updates, ob_snaps, other = parse_fix_chunk(buf)

    print(f"chunk {n_chunk:>3}: building frames …", flush=True)
    df_trades   = _build_trades(trades, adds_index)
    df_ob_upd   = _build_ob_updates(ob_updates, adds_index)
    df_ob_snap  = _build_ob_snapshot(ob_snaps)
    df_other    = _build_other(other)

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

    del trades, ob_updates, ob_snaps, other
    del df_trades, df_ob_upd, df_ob_snap, df_other
    gc.collect()


# ─────────────────────────── main pipeline ───────────────────────────────────

def run_day(src=SRC, out_dir=OUT_DIR, chunk_lines=CHUNK_LINES):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day    = Path(src).name.replace(".tar.gz", "")
    labels = ("trades", "ob_updates", "ob_snapshot", "other")
    totals = dict.fromkeys(labels, 0)

    adds_index = {}   # {(channel, appl_seq) -> order_id}  persists across chunks
    n_chunk    = 0
    n_lines    = 0
    t0         = time.time()

    print(f"\nOpening {Path(src).name} …", flush=True)
    with tarfile.open(src, "r:*") as tf:
        member = next(m for m in tf.getmembers()
                      if m.isfile() and m.name.lower().endswith(".txt"))
        print(f"  inner file: {member.name}", flush=True)
        stream = io.TextIOWrapper(tf.extractfile(member),
                                  encoding="utf-8", errors="replace")

        buf = []
        for line in stream:
            buf.append(line)
            n_lines += 1
            if n_lines % 500_000 == 0:
                print(f"  … {n_lines:,} lines read ({time.time()-t0:.1f}s)", flush=True)
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
    print(f"\nPass 1 done: {n_chunk} chunks, {n_lines:,} lines. Row totals: {totals}", flush=True)

    # ── merge partials → 4 final Parquets ──────────────────────────────────
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
        print(f"  {label:<12}: {totals[label]:>12,} rows  →  {final.name}  ({mb:.1f} MB)")

    print(f"\nAll done in {time.time()-t0:.1f}s", flush=True)
    print(f"Output: {out_dir}", flush=True)


# ─────────────────────────── entry point ─────────────────────────────────────
if __name__ == "__main__":
    run_day()
