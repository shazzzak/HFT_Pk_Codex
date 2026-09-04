"""Core per-symbol statistics. Validated on sample CSVs, reused by the parquet script."""
# Numerical library: medians, diffs, searchsorted for the as-of join.
import numpy as np
# DataFrame library: all frame handling and the pivot that builds the L1 series.
import pandas as pd

# ---- FEES (mirror your mm_backtest schedule) -------------------------------
# Broker commission as a fraction of traded value (0.15%). Dominates the stack.
FEE_COMMISSION_PCT = 0.0015
# Sales tax charged ON THE COMMISSION (not on value): 13% Sindh.
FEE_SST_RATE       = 0.13
# PSX trading fee ("laga"): PKR 3.50 per 100,000 = 0.0035%.
FEE_PSX_LAGA_PCT   = 0.000035
# SECP supervisory levy: PKR 0.65 per 100,000.
FEE_SECP_PCT       = 0.0000065
# PSX regulatory / Investor Protection Fund: PKR 0.62084 per 100,000.
FEE_IPF_PCT        = 0.0000062
# NCCPL + CDC clearing, per leg (low end assumes intraday-squared, no delivery).
FEE_CLEARING_PCT   = 0.00003
# All-in ONE-SIDE fee: commission grossed up for SST, plus the regulatory stack.
FEE_TOTAL_PCT = (FEE_COMMISSION_PCT * (1 + FEE_SST_RATE)
                 # The four non-commission components add linearly on traded value.
                 + FEE_PSX_LAGA_PCT + FEE_SECP_PCT + FEE_IPF_PCT + FEE_CLEARING_PCT)
# Round-trip cost in basis points (two sides), for display and screening.
ROUND_TRIP_BPS = 2 * FEE_TOTAL_PCT * 1e4        # per-round-trip fee in bps

# Threshold above which a spread counts as "wide" for the pct_time_wide screen.
WIDE_BPS = 30.0        # "wide spread" screen threshold (your image uses 30)

# Round-trip fee grid in decimal. The current-schedule point is DERIVED from
# FEE_TOTAL_PCT (not hardcoded) so ceiling_pkr_rt_35p45 always agrees exactly
# with the legacy ceiling_pkr column. Others: 2, 4, 10, 20, 60 bps round-trip.
FEE_RT_GRID = [0.0002, 0.0004, 0.0010, 0.0020, 2 * FEE_TOTAL_PCT, 0.0060]


# Convert a timestamp column to int64 milliseconds since epoch (UTC).
def _ms(series):
    # Parse to UTC-aware datetimes; ISO8601 covers both strings and tz-aware columns.
    return (pd.to_datetime(series, utc=True, format="ISO8601")
            # Force nanosecond unit so .astype("int64") is always ns, then // 1e6 -> ms.
            .dt.as_unit("ns").astype("int64") // 1_000_000)


# Compute one row of the screen for a single symbol on a single day.
def stats_for_symbol(snap, trades, symbol):
    """Compute descriptive + co-incidence stats for ONE symbol.

    snap, trades: DataFrames already filtered to this symbol.
    Returns a dict (one row of the final table)."""
    # No trades at all -> nothing to measure.
    if len(trades) == 0:
        # Caller treats None as "skip this symbol".
        return None

    # Copy both frames so the caller's data is never mutated.
    snap = snap.copy(); trades = trades.copy()
    # Snapshot exchange clock: orig_time (tag 42) -> ms.
    snap["ts"] = _ms(snap["orig_time"])
    # Trade exchange clock: transact_time (tag 60) -> ms.
    trades["ts"] = _ms(trades["transact_time"])

    # ---- L1 spread series from snapshots ----
    # Keep only top-of-book rows: level 1, and only the BID/OFFER entry types.
    sb = snap[snap["entry_type"].isin(["BID", "OFFER"]) & (snap["level"] == 1)]
    # No touch rows -> no spread series can be built.
    if len(sb) == 0:
        # Skip this symbol.
        return None
    # One row per snapshot message, with BID and OFFER as separate columns.
    l1 = sb.pivot_table(index="msg_seq", columns="entry_type", values="px", aggfunc="first")
    # A symbol whose level-1 book was one-sided for the WHOLE session (bid-only,
    # offer-only, or pinned at a circuit limit) yields a pivot with the other
    # column absent entirely -- so dropna(subset=[...]) would raise KeyError.
    # This is a legitimate "not screenable" case, not an error: skip it cleanly
    # like the other guards, so it does not pollute the error count.
    if not {"BID", "OFFER"}.issubset(l1.columns):
        return None
    # Timestamp each message with the earliest ts among its rows.
    l1["ts"] = sb.groupby("msg_seq")["ts"].min()
    # Drop one-sided quotes (no valid spread) and order chronologically.
    l1 = l1.dropna(subset=["BID", "OFFER"]).sort_values("ts")
    # Nothing two-sided left -> skip.
    if len(l1) == 0:
        # Skip this symbol.
        return None

    # continuous session = between first and last non-auction trade
    # Auction prints have no continuous-market aggressor, so exclude them.
    tc = trades[trades["initiator"] != "AUCTION"].sort_values("ts")
    # No continuous trades -> no session to measure.
    if len(tc) == 0:
        # Skip this symbol.
        return None
    # Session bounds in exchange-ms: first and last continuous trade.
    open_ms, close_ms = int(tc["ts"].min()), int(tc["ts"].max())
    # Session length in hours, floored away from zero to avoid divide-by-zero.
    hours = max((close_ms - open_ms) / 3.6e6, 1e-9)

    # Restrict the quote series to the continuous session window.
    l1 = l1[(l1["ts"] >= open_ms) & (l1["ts"] <= close_ms)].copy()
    # No quotes inside the session -> skip.
    if len(l1) == 0:
        # Skip this symbol.
        return None
    # Absolute spread in PKR.
    l1["spread"] = l1["OFFER"] - l1["BID"]
    # Touch midpoint.
    l1["mid"] = (l1["OFFER"] + l1["BID"]) / 2
    # Drop non-positive mids (divide-by-zero guard).
    l1 = l1[l1["mid"] > 0]
    # Spread in basis points of the mid -- comparable across price levels.
    l1["spread_bps"] = 1e4 * l1["spread"] / l1["mid"]
    # guard against crossed/locked snapshots
    # Negative spreads are data artefacts; remove them.
    l1 = l1[l1["spread"] >= 0]

    # ---- DESCRIPTIVE (time-averaged) ----
    # Day's total traded shares (all trades, including auction).
    total_vol = float(trades["qty"].sum())
    # Day's total traded value in PKR.
    notional = float((trades["qty"] * trades["price"]).sum())
    # Assemble the descriptive block of the output row.
    row = {
        # The ticker this row describes.
        "symbol": symbol,
        # Total trade count for the day.
        "trades": int(len(trades)),
        # Continuous trades per hour -- an activity measure.
        "trades_per_hr": len(tc) / hours,
        # Median seconds between continuous trades (burstiness / pacing).
        "median_gap_s": float(np.median(np.diff(np.sort(tc["ts"].to_numpy()))) / 1000)
                        # Undefined with fewer than two trades.
                        if len(tc) > 1 else np.nan,
        # Total shares traded.
        "volume_sh": total_vol,
        # Total value traded, in PKR millions.
        "notional_m": notional / 1e6,
        # Median trade price -- the reference for bps conversions elsewhere.
        "median_px": float(trades["price"].median()),
        # Typical spread in bps (the central tendency of quoting).
        "median_spread_bps": float(l1["spread_bps"].median()),
        # 90th percentile spread -- how wide it gets in the tail.
        "p90_spread_bps": float(l1["spread_bps"].quantile(0.90)),
        # 99th percentile spread -- extreme widening.
        "p99_spread_bps": float(l1["spread_bps"].quantile(0.99)),
        # Fraction of quoted time the spread exceeded WIDE_BPS.
        "pct_time_wide": float((l1["spread_bps"] > WIDE_BPS).mean()),
    }

    # ---- CO-INCIDENCE (the decision metric): spread & volume at the SAME instant ----
    # Quote timestamps and spreads as arrays, for the as-of join.
    sp_ts = l1["ts"].to_numpy(); sp = l1["spread"].to_numpy()
    # For each trade, the index of the most recent PRIOR snapshot.
    idx = np.searchsorted(sp_ts, tc["ts"].to_numpy(), side="right") - 1
    # Trades occurring before the first snapshot get -1 and are unmatchable.
    ok = idx >= 0
    # Keep only matchable trades and align the index array to them.
    tcv = tc[ok].copy(); idx = idx[ok]
    # The spread that prevailed at each trade's instant (the co-incidence join).
    tcv["prev_spread"] = sp[idx]
    # Same, expressed in bps of the trade price.
    tcv["prev_spread_bps"] = 1e4 * tcv["prev_spread"] / tcv["price"]
    # ---- CO-INCIDENCE at each fee level on the grid ----
    # Evaluate the ceiling at every fee scenario in one pass over the day.
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
        # Qualifying volume that arrived as sell-side aggression.
        vs = float(q.loc[q["aggressor_side"] == "SELL", "qty"].sum())
        # Two-sidedness ratio in [0,1]; 0 when there is no qualifying volume.
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
        # Qualifying NOTIONAL (PKR millions) at this fee level -- consumed by the screen.
        row[f"qual_notional_m_rt{tag}"] = float((q["qty"] * q["price"]).sum()) / 1e6

    # Legacy alias: current-schedule ceiling, kept for existing queries/notebooks.
    # Build the column tag for the CURRENT fee schedule (derived, never hardcoded).
    _cur = f"_{2 * FEE_TOTAL_PCT * 1e4:.2f}".replace(".", "p")
    # Point ceiling_pkr at the current-schedule ceiling.
    row["ceiling_pkr"] = row[f"ceiling_pkr_rt{_cur}"]

    # Current-schedule aliases consumed by run_all_tickers' screen columns.
    # Qualifying volume share at the current schedule.
    row["pct_vol_qualifying"] = row[f"pct_vol_qual_rt{_cur}"]
    # Qualifying notional (PKR m) at the current schedule.
    row["qualifying_notional_m"] = row[f"qual_notional_m_rt{_cur}"]
    # ---- ADVERSE SELECTION / MARKOUTS (ported verbatim from multi_sym.py) ----
    # Reuses the SAME session-filtered L1 mid series built above (l1["ts"],
    # l1["mid"]) -- no new data. The ceiling is an UPPER bound on opportunity;
    # these measure how much of it SURVIVES adverse selection, which is what
    # actually determines realized MM edge. net60_bps > 0 is THE viability test.
    # Mid series as arrays for forward/backward lookups.
    mt, mv = l1["ts"].to_numpy(), l1["mid"].to_numpy()

    # First mid at or AFTER time w, unless the nearest quote is stale (> tol ms).
    def m_at(w, tol=20000):
        # Index of the first quote at or after w.
        i = np.searchsorted(mt, w, side="left")
        # Return it only if it exists and is within tol milliseconds.
        return mv[i] if i < len(mt) and mt[i] - w <= tol else np.nan

    # Last mid at or BEFORE time w (the pre-trade reference).
    def m_bf(w):
        # Index of the last quote at or before w.
        i = np.searchsorted(mt, w, side="right") - 1
        # Return it only if such a quote exists.
        return mv[i] if i >= 0 else np.nan

    # Accumulate one record per signable continuous trade.
    rows = []
    # Walk the continuous-session trades.
    for x in tc.itertuples():
        # Need a known aggressor to sign the trade.
        if x.aggressor_side not in ("BUY", "SELL"):
            # Skip unsignable prints.
            continue
        # Sign from the AGGRESSOR's perspective: +1 buy, -1 sell.
        d = 1 if x.aggressor_side == "BUY" else -1
        # Mid immediately before the trade.
        m0 = m_bf(x.ts)
        # No prior quote -> cannot measure this trade.
        if np.isnan(m0):
            # Skip it.
            continue
        # Effective half-spread the AGGRESSOR paid (positive = paid up).
        rec = {"eff": d * (x.price - m0)}
        # Forward markouts across horizons. Short ones (1-10s) matter for HFT:
        # a maker who flattens in seconds never holds long enough to eat the
        # 60s+ directional drift, so 60s understates a fast strategy's edge.
        for h in (1, 5, 10, 30, 60, 300):
            # Mid h seconds after the trade.
            mh = m_at(x.ts + h * 1000)
            # Realized half-spread the PASSIVE side kept after h seconds.
            # Positive = passive side profited; negative = adverse selection.
            rec[f"r{h}"] = d * (x.price - mh) if not np.isnan(mh) else np.nan
        # Store this trade's record.
        rows.append(rec)
    # Assemble the per-trade markout records.
    dec = pd.DataFrame(rows)
    # Average effective half-spread paid, in bps of the median price.
    row["eff_bps"] = 1e4 * dec["eff"].mean() / row["median_px"] if len(dec) else np.nan
    # Average passive realized half-spread at each horizon, in bps.
    for h in (1, 5, 10, 30, 60, 300):
        # Mean realized markout at horizon h, in bps of median price.
        row[f"mk{h}_bps"] = 1e4 * dec[f"r{h}"].mean() / row["median_px"] if len(dec) else np.nan
    # THE DECISION NUMBER: gross passive edge at 60s minus ONE side's fee.
    # Positive -> passive MM is viable on this symbol; negative -> the fee exceeds
    # the surviving edge and no strategy quality can fix it.
    row["net60_bps"] = row["mk60_bps"] - FEE_TOTAL_PCT * 1e4

    # One completed stats row for this symbol.
    return row