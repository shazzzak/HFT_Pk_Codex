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
# FEE_TOTAL_PCT (not hardcoded) so ceiling_pkr_rt_35p45 always agrees exactly
# with the legacy ceiling_pkr column. Others: 2, 4, 10, 20, 60 bps round-trip.
FEE_RT_GRID = [0.0002, 0.0004, 0.0010, 0.0020, 2 * FEE_TOTAL_PCT, 0.0060]


def _ms(series):
    return (pd.to_datetime(series, utc=True, format="ISO8601")
            .dt.as_unit("ns").astype("int64") // 1_000_000)


def stats_for_symbol(snap, trades, symbol):
    """Compute descriptive + co-incidence stats for ONE symbol.

    snap, trades: DataFrames already filtered to this symbol.
    Returns a dict (one row of the final table)."""
    if len(trades) == 0:
        return None

    snap = snap.copy(); trades = trades.copy()
    snap["ts"] = _ms(snap["orig_time"])
    trades["ts"] = _ms(trades["transact_time"])

    # ---- L1 spread series from snapshots ----
    sb = snap[snap["entry_type"].isin(["BID", "OFFER"]) & (snap["level"] == 1)]
    if len(sb) == 0:
        return None
    l1 = sb.pivot_table(index="msg_seq", columns="entry_type", values="px", aggfunc="first")
    l1["ts"] = sb.groupby("msg_seq")["ts"].min()
    l1 = l1.dropna(subset=["BID", "OFFER"]).sort_values("ts")
    if len(l1) == 0:
        return None

    # continuous session = between first and last non-auction trade
    tc = trades[trades["initiator"] != "AUCTION"].sort_values("ts")
    if len(tc) == 0:
        return None
    open_ms, close_ms = int(tc["ts"].min()), int(tc["ts"].max())
    hours = max((close_ms - open_ms) / 3.6e6, 1e-9)

    l1 = l1[(l1["ts"] >= open_ms) & (l1["ts"] <= close_ms)].copy()
    if len(l1) == 0:
        return None
    l1["spread"] = l1["OFFER"] - l1["BID"]
    l1["mid"] = (l1["OFFER"] + l1["BID"]) / 2
    l1 = l1[l1["mid"] > 0]
    l1["spread_bps"] = 1e4 * l1["spread"] / l1["mid"]
    # guard against crossed/locked snapshots
    l1 = l1[l1["spread"] >= 0]

    # ---- DESCRIPTIVE (time-averaged) ----
    total_vol = float(trades["qty"].sum())
    notional = float((trades["qty"] * trades["price"]).sum())
    row = {
        "symbol": symbol,
        "trades": int(len(trades)),
        "trades_per_hr": len(tc) / hours,
        "median_gap_s": float(np.median(np.diff(np.sort(tc["ts"].to_numpy()))) / 1000)
                        if len(tc) > 1 else np.nan,
        "volume_sh": total_vol,
        "notional_m": notional / 1e6,
        "median_px": float(trades["price"].median()),
        "median_spread_bps": float(l1["spread_bps"].median()),
        "p90_spread_bps": float(l1["spread_bps"].quantile(0.90)),
        "p99_spread_bps": float(l1["spread_bps"].quantile(0.99)),
        "pct_time_wide": float((l1["spread_bps"] > WIDE_BPS).mean()),
    }

    # ---- CO-INCIDENCE (the decision metric): spread & volume at the SAME instant ----
    sp_ts = l1["ts"].to_numpy(); sp = l1["spread"].to_numpy()
    idx = np.searchsorted(sp_ts, tc["ts"].to_numpy(), side="right") - 1
    ok = idx >= 0
    tcv = tc[ok].copy(); idx = idx[ok]
    tcv["prev_spread"] = sp[idx]
    tcv["prev_spread_bps"] = 1e4 * tcv["prev_spread"] / tcv["price"]
    # ---- CO-INCIDENCE at each fee level on the grid ----
    for fee_rt in FEE_RT_GRID:
        # Net edge per share at this round-trip fee level.
        net = tcv["prev_spread"] - fee_rt * tcv["price"]
        # Trades where spread and liquidity coincided profitably.
        q = tcv[net > 0]
        # Upper-bound daily profit: net edge x qualifying volume, halved
        # because one round trip consumes two fills.
        ub = float((net[net > 0] * q["qty"]).sum() / 2)
        # PAIRING ADJUSTMENT: a round trip needs a buy-side AND a sell-side
        # fill. Directional sweeps inflate the /2 ceiling, so scale by how
        # two-sided the qualifying flow actually was (1.0 = balanced, ->0 = one-way).
        vb = float(q.loc[q["aggressor_side"] == "BUY", "qty"].sum())
        vs = float(q.loc[q["aggressor_side"] == "SELL", "qty"].sum())
        pair_ratio = (2.0 * min(vb, vs) / (vb + vs)) if (vb + vs) > 0 else 0.0
        # Tag = round-trip bps to 2dp, '.' -> 'p' so the column name stays
        # SQL-safe, with a '_' separator: ceiling_pkr_rt_35p45.
        tag = f"_{fee_rt * 1e4:.2f}".replace(".", "p")
        # Raw ceiling (optimistic).
        row[f"ceiling_pkr_rt{tag}"] = ub
        # Pairing-adjusted ceiling -- the one to RANK on.
        row[f"ceiling_paired_rt{tag}"] = ub * pair_ratio
        # Share of the day's volume that cleared the fee at this level.
        row[f"pct_vol_qual_rt{tag}"] = (float(q["qty"].sum()) / total_vol) if total_vol else 0.0

    # Legacy alias: current-schedule ceiling, kept for existing queries/notebooks.
    _cur = f"_{2 * FEE_TOTAL_PCT * 1e4:.2f}".replace(".", "p")
    row["ceiling_pkr"] = row[f"ceiling_pkr_rt{_cur}"]

    # One completed stats row for this symbol.
    return row
