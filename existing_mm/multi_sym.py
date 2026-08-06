"""Batch per-symbol MM analysis: spreads, fee-grid ceilings, adverse selection.

One row per (symbol, day). Reads the three per-symbol CSVs, and returns a dict
of descriptive stats, profit ceilings across a fee grid, and markout-based
adverse-selection measures. Designed so new symbols need no code change: it
globs by symbol name, so dropping more CSVs in and re-running is enough.
"""
import glob
import numpy as np, pandas as pd

# Directory holding the per-symbol CSV extracts.
UP = "/mnt/user-data/uploads"
# All-in PSX fee PER SIDE as a fraction of traded value: commission x (1+SST)
# plus the exchange/regulatory stack. 17.73 bps.
FEE_SIDE = 0.0015*1.13 + 0.000035 + 0.0000065 + 0.0000062 + 0.00003   # 17.73 bps
# ROUND-TRIP fee scenarios to price the ceiling at: 2, 10, 20, current, 60 bps.
# The current-schedule point is DERIVED (2 x FEE_SIDE), never hardcoded.
FEE_GRID = [0.0002, 0.0010, 0.0020, 2*FEE_SIDE, 0.0060]

def _ms(df, c):
    # ISO timestamp column -> int64 milliseconds since epoch.
    # as_unit("ns") first so .astype("int64") is nanoseconds on any pandas
    # version (pandas 3.0 parses to microseconds and would silently be 1000x off).
    return (pd.to_datetime(df[c], utc=True, format="ISO8601")
            .dt.as_unit("ns").astype("int64") // 1_000_000)

def find_files(sym, date):
    """Locate the 3 CSVs for a symbol regardless of upload-id prefix."""
    # Maps the filename tag to the short key used downstream.
    out = {}
    for tag, key in [("trades","trades"), ("Ob_snapshot","snap"), ("Ob_updates","upd")]:
        # Wildcard the upload-id prefix so "1785677768184_KTML_trades_...csv" matches.
        hits = glob.glob(f"{UP}/*_{sym}_{tag}_{date}.csv")
        # If several uploads of the same file exist, take the last (newest id).
        if hits:
            out[key] = sorted(hits)[-1]
    # Require all three; a partial set can't produce a valid analysis.
    return out if len(out) == 3 else None

def analyse(sym, date):
    # Resolve the three input files.
    f = find_files(sym, date)
    # Missing any file -> skip this symbol silently (caller filters None).
    if f is None:
        return None
    # low_memory=False: prev_close has mixed types in some snapshot files.
    s = pd.read_csv(f["snap"], index_col=0, low_memory=False)
    t = pd.read_csv(f["trades"], index_col=0)
    u = pd.read_csv(f["upd"], index_col=0)
    # No trades -> nothing to measure.
    if len(t) == 0:
        return None
    # Exchange clock for all three tables (snapshots use orig_time, tag 42).
    s["ts"], t["ts"], u["ts"] = _ms(s,"orig_time"), _ms(t,"transact_time"), _ms(u,"transact_time")

    # Level-1 rows only: the touch.
    sb = s[s.entry_type.isin(["BID","OFFER"]) & (s.level == 1)]
    # One row per snapshot message, with BID and OFFER as columns.
    l1 = sb.pivot_table(index="msg_seq", columns="entry_type", values="px", aggfunc="first")
    # Timestamp each message (min ts across its rows).
    l1["ts"] = sb.groupby("msg_seq")["ts"].min()
    # Drop one-sided quotes and order by time.
    l1 = l1.dropna(subset=["BID","OFFER"]).sort_values("ts")
    # Continuous session only: auction prints have no continuous aggressor.
    tc = t[t.initiator != "AUCTION"].sort_values("ts")
    # Need both a trade stream and a quote stream to proceed.
    if len(tc) == 0 or len(l1) == 0:
        return None
    # Session bounds = first and last continuous trade.
    o, c = int(tc.ts.min()), int(tc.ts.max())
    # Session length in hours, floored away from zero.
    hrs = max((c-o)/3.6e6, 1e-9)
    # Restrict the quote series to the continuous session.
    l1 = l1[(l1.ts >= o) & (l1.ts <= c)].copy()
    # Absolute spread in PKR.
    l1["spread"] = l1.OFFER - l1.BID
    # Touch midpoint.
    l1["mid"] = (l1.OFFER + l1.BID)/2
    # Drop bad quotes: non-positive mid (divide-by-zero) and crossed books.
    l1 = l1[(l1["mid"] > 0) & (l1.spread >= 0)]
    # Nothing usable left -> skip.
    if len(l1) == 0:
        return None
    # Spread in basis points of the mid -- comparable across price levels.
    l1["sbps"] = 1e4*l1.spread/l1["mid"]

    # Median trade price: the reference for all bps conversions below.
    px = float(tc.price.median())
    # Descriptive block: activity, size, and the spread distribution.
    r = {"symbol": sym, "px": px, "trades": len(tc),
         "vol_sh": float(tc.qty.sum()), "notional_m": float((tc.qty*tc.price).sum()/1e6),
         "tr_hr": len(tc)/hrs, "gap_s": float(np.median(np.diff(np.sort(tc.ts.to_numpy())))/1000) if len(tc)>1 else np.nan,
         "med_size": float(tc.qty.median()), "tick_bps": 1e4*0.01/px,
         "sp50": float(l1.sbps.median()), "sp90": float(l1.sbps.quantile(.90)),
         "sp99": float(l1.sbps.quantile(.99)), "pct_wide30": float((l1.sbps>30).mean()),
         "n_snap": len(l1)}

    # ceilings on the fee grid, with pairing adjustment
    # Quote timestamps and spreads as arrays for as-of matching.
    sp_ts, sp = l1.ts.to_numpy(), l1.spread.to_numpy()
    # For each trade, index of the most recent PRIOR snapshot.
    idx = np.searchsorted(sp_ts, tc.ts.to_numpy(), side="right") - 1
    # Trades before the first snapshot get idx = -1 -> unmatchable.
    ok = idx >= 0
    # Keep matchable trades and align the index to them.
    tcv = tc[ok].copy(); idx = idx[ok]
    # The spread prevailing at each trade's instant (the co-incidence join).
    tcv["ps"] = sp[idx]
    for rt in FEE_GRID:
        # Net edge per share at this round-trip fee level.
        net = tcv.ps - rt*tcv.price
        # Trades where spread and liquidity coincided profitably.
        q = tcv[net > 0]
        # Upper-bound daily profit: net edge x qualifying volume, halved
        # because one round trip consumes two fills.
        ub = float((net[net>0]*q.qty).sum()/2)
        # Qualifying volume by aggressor side.
        vb = float(q.loc[q.aggressor_side=="BUY","qty"].sum())
        vs = float(q.loc[q.aggressor_side=="SELL","qty"].sum())
        # PAIRING RATIO: a round trip needs one buy-side and one sell-side fill.
        # 1.0 = perfectly two-sided flow, ->0 = one-way sweep. Directional flow
        # inflates the raw /2 ceiling, so scale it down.
        pair = (2*min(vb,vs)/(vb+vs)) if (vb+vs) > 0 else 0.0
        # Column suffix = round-trip bps, e.g. "35".
        tag = f"{rt*1e4:.0f}"
        # Raw ceiling (optimistic).
        r[f"ceil{tag}"] = ub
        # Pairing-adjusted ceiling -- the one to RANK on.
        r[f"paired{tag}"] = ub*pair
        # Share of the day's volume that cleared the fee at this level.
        r[f"qual{tag}"] = float(q.qty.sum())/float(tc.qty.sum()) if tc.qty.sum() else 0.0

    # adverse selection / markouts
    # Mid series as arrays for forward/backward lookups.
    mt, mv = l1.ts.to_numpy(), l1["mid"].to_numpy()
    def m_at(w, tol=20000):
        # First mid at or AFTER time w (for forward markouts).
        i = np.searchsorted(mt, w, side="left")
        # Reject if the nearest quote is more than tol ms away (stale).
        return mv[i] if i < len(mt) and mt[i]-w <= tol else np.nan
    def m_bf(w):
        # Last mid at or BEFORE time w (the pre-trade reference).
        i = np.searchsorted(mt, w, side="right")-1
        return mv[i] if i >= 0 else np.nan
    rows = []
    for x in tc.itertuples():
        # Need a known aggressor to sign the trade.
        if x.aggressor_side not in ("BUY","SELL"):
            continue
        # Sign from the AGGRESSOR's perspective: +1 buy, -1 sell.
        d = 1 if x.aggressor_side == "BUY" else -1
        # Mid immediately before the trade.
        m0 = m_bf(x.ts)
        # No prior quote -> can't measure.
        if np.isnan(m0):
            continue
        # Effective half-spread the AGGRESSOR paid (positive = paid up).
        rec = {"eff": d*(x.price-m0)}
        for h in (60, 300):
            # Mid h seconds later.
            mh = m_at(x.ts+h*1000)
            # Realized half-spread the PASSIVE side kept after h seconds.
            # Positive = the passive side profited; negative = adverse selection.
            rec[f"r{h}"] = d*(x.price-mh) if not np.isnan(mh) else np.nan
        rows.append(rec)
    dec = pd.DataFrame(rows)
    # Average effective half-spread paid, in bps of the median price.
    r["eff_bps"] = 1e4*dec.eff.mean()/px if len(dec) else np.nan
    for h in (60, 300):
        # Average passive realized half-spread at each horizon, in bps.
        r[f"mk{h}_bps"] = 1e4*dec[f"r{h}"].mean()/px if len(dec) else np.nan
    # THE DECISION NUMBER: gross passive edge minus one side's fee.
    # Positive -> passive MM is viable on this symbol at this fee. Negative ->
    # the fee exceeds the edge and no strategy quality can fix it.
    r["net60_bps"] = r["mk60_bps"] - FEE_SIDE*1e4        # net of ONE side's fee
    # Merged event stream (updates + trades + snapshot messages) to measure pacing.
    ev = np.sort(np.concatenate([u.ts.to_numpy(), t.ts.to_numpy(),
                                 s.groupby("msg_seq")["ts"].min().to_numpy()]))
    # Inter-event gaps; drop any negative artefacts from ties.
    g = np.diff(ev); g = g[g >= 0]
    # Median inter-event gap -- drives latency sensitivity and window sizing.
    r["ev_gap_ms"] = float(np.median(g)) if len(g) else np.nan
    # Total events in the day for this symbol.
    r["n_events"] = len(ev)
    # One completed stats row.
    return r