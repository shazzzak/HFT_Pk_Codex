"""Core per-symbol statistics. Validated on sample CSVs, reused by the parquet script."""
import numpy as np
import pandas as pd

# ---- FEES (mirror your mm_backtest schedule) -------------------------------
FEE_COMMISSION_PCT = 0.0015
FEE_SST_RATE       = 0.13
FEE_PSX_LAGA_PCT   = 0.000035
FEE_SECP_PCT       = 0.0000065
FEE_IPF_PCT        = 0.0000062
FEE_CLEARING_PCT   = 0.00003
FEE_TOTAL_PCT = (FEE_COMMISSION_PCT * (1 + FEE_SST_RATE)
                 + FEE_PSX_LAGA_PCT + FEE_SECP_PCT + FEE_IPF_PCT + FEE_CLEARING_PCT)
ROUND_TRIP_BPS = 2 * FEE_TOTAL_PCT * 1e4        # per-round-trip fee in bps

WIDE_BPS = 30.0        # "wide spread" screen threshold (your image uses 30)

# Round-trip fee grid in decimal. The current-schedule point is DERIVED from
# FEE_TOTAL_PCT (not hardcoded) so ceiling_pkr_rt35 always agrees exactly with
# the legacy ceiling_pkr column. Others: 2, 4, 10, 20, 60 bps round-trip.
FEE_RT_GRID = [0.0002, 0.0004, 0.0010, 0.0020, 2 * FEE_TOTAL_PCT, 0.0060]

def _ms(series):
    return (pd.to_datetime(series, utc=True, format="ISO8601")
            .dt.as_unit("ns").astype("int64") // 1_000_000)


def stats_for_symbol(snap, trades, symbol):
    """Compute descriptive + co-incidence stats for ONE symbol.
    snap, trades: DataFrames already filtered to this symbol.
    Returns a dict (one row of the final table)."""
    # No trades -> nothing to measure for this symbol.
    if len(trades) == 0:
        return None
    # Copy so the ts columns we add don't mutate the caller's frames.
    snap = snap.copy(); trades = trades.copy()
    # Exchange time (ms) from the snapshot's orig_time.
    snap["ts"] = _ms(snap["orig_time"])
    # Exchange time (ms) from each trade's transact_time.
    trades["ts"] = _ms(trades["transact_time"])
    # ---- L1 spread series from snapshots ----
    # Keep only best-bid / best-offer rows (level 1) -> the touch.
    sb = snap[snap["entry_type"].isin(["BID", "OFFER"]) & (snap["level"] == 1)]
    # No level-1 quotes -> can't build a spread series.
    if len(sb) == 0:
        return None
    # Pivot to one row per snapshot message with BID and OFFER as columns.
    l1 = sb.pivot_table(index="msg_seq", columns="entry_type", values="px", aggfunc="first")
    # Attach each message's timestamp (min ts across its rows).
    l1["ts"] = sb.groupby("msg_seq")["ts"].min()
    # Drop half-quotes (one side missing) and order by time.
    l1 = l1.dropna(subset=["BID", "OFFER"]).sort_values("ts")
    # No complete two-sided snapshots -> nothing to measure.
    if len(l1) == 0:
        return None
    # continuous session = between first and last non-auction trade
    # Exclude auction prints; sort the remaining continuous trades by time.
    tc = trades[trades["initiator"] != "AUCTION"].sort_values("ts")
    # No continuous trades -> skip this symbol.
    if len(tc) == 0:
        return None
    # Session bounds = first and last continuous-trade timestamps.
    open_ms, close_ms = int(tc["ts"].min()), int(tc["ts"].max())
    # Session length in hours, floored away from zero to avoid divide-by-zero.
    hours = max((close_ms - open_ms) / 3.6e6, 1e-9)
    # Restrict the spread series to the continuous session window.
    l1 = l1[(l1["ts"] >= open_ms) & (l1["ts"] <= close_ms)].copy()
    # No snapshots inside the session -> skip.
    if len(l1) == 0:
        return None
    # Absolute spread in PKR.
    l1["spread"] = l1["OFFER"] - l1["BID"]
    # Midpoint of the touch.
    l1["mid"] = (l1["OFFER"] + l1["BID"]) / 2
    # Drop non-positive mids (bad/empty quotes) before dividing by mid.
    l1 = l1[l1["mid"] > 0]
    # Spread in basis points of the mid.
    l1["spread_bps"] = 1e4 * l1["spread"] / l1["mid"]
    # guard against crossed/locked snapshots
    # Drop negative-spread rows (crossed books) that would distort stats.
    l1 = l1[l1["spread"] >= 0]
    # ---- DESCRIPTIVE (time-averaged) ----
    # Total shares traded across the whole session.
    total_vol = float(trades["qty"].sum())
    # Total traded value in PKR.
    notional = float((trades["qty"] * trades["price"]).sum())
    # Assemble the descriptive half of the output row.
    row = {
        # Ticker symbol.
        "symbol": symbol,
        # Trade count (all trades, including auction).
        "trades": int(len(trades)),
        # Continuous trades per hour = activity rate.
        "trades_per_hr": len(tc) / hours,
        # Median gap between consecutive continuous trades, in seconds
        # (NaN if fewer than two trades to form a gap).
        "median_gap_s": float(np.median(np.diff(np.sort(tc["ts"].to_numpy()))) / 1000)
                        if len(tc) > 1 else np.nan,
        # Total volume in shares.
        "volume_sh": total_vol,
        # Total notional in millions of PKR.
        "notional_m": notional / 1e6,
        # Median trade price.
        "median_px": float(trades["price"].median()),
        # Median time-averaged spread (bps) -- descriptive, NOT the decision metric.
        "median_spread_bps": float(l1["spread_bps"].median()),
        # 90th-percentile spread (bps).
        "p90_spread_bps": float(l1["spread_bps"].quantile(0.90)),
        # 99th-percentile spread (bps).
        "p99_spread_bps": float(l1["spread_bps"].quantile(0.99)),
        # Fraction of session time the spread exceeded the WIDE_BPS threshold.
        "pct_time_wide": float((l1["spread_bps"] > WIDE_BPS).mean()),
    }
    # ---- CO-INCIDENCE (the decision metric): spread & volume at the SAME instant ----
    # Snapshot timestamps and spreads as arrays for as-of matching.
    sp_ts = l1["ts"].to_numpy(); sp = l1["spread"].to_numpy()
    # For each trade, find the index of the most recent PRIOR snapshot.
    # searchsorted(..., "right")-1 = last snapshot at or before the trade.
    idx = np.searchsorted(sp_ts, tc["ts"].to_numpy(), side="right") - 1
    # Trades before the first snapshot have idx = -1 -> drop them.
    ok = idx >= 0
    # Keep only matchable trades, and align idx to them.
    tcv = tc[ok].copy(); idx = idx[ok]
    # The spread that was prevailing at each trade's instant.
    tcv["prev_spread"] = sp[idx]
    # That prevailing spread in bps of the trade price.
    tcv["prev_spread_bps"] = 1e4 * tcv["prev_spread"] / tcv["price"]

    # ---- CO-INCIDENCE at each fee level on the grid ----
    for fee_rt in FEE_RT_GRID:
        # Net edge per share at this fee level.
        net = tcv["prev_spread"] - fee_rt * tcv["price"]
        q = tcv[net > 0]
        # Tag = round-trip bps to 2dp, '.' -> 'p' so the column name stays
        # SQL-safe, with a '_' separator: ceiling_pkr_rt_35p45, pct_vol_qual_rt_2p00.
        tag = f"_{fee_rt * 1e4:.2f}".replace(".", "p")  # 35.4531 -> "_35p45", 2.0 -> "_2p00"
        row[f"ceiling_pkr_rt{tag}"] = float((net[net > 0] * q["qty"]).sum() / 2)
        row[f"pct_vol_qual_rt{tag}"] = float(q["qty"].sum()) / total_vol if total_vol else 0.0

    # Legacy alias: current-schedule ceiling, kept for existing queries/notebooks.
    _cur = f"_{2 * FEE_TOTAL_PCT * 1e4:.2f}".replace(".", "p")
    row["ceiling_pkr"] = row[f"ceiling_pkr_rt{_cur}"]

    # One completed stats row for this symbol.
    return row
