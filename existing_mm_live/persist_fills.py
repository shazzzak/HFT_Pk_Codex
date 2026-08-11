# ============================================================================
# persist_fills.py -- SYSTEM FILE (goes in existing_mm_live/, run repeatedly)
# ============================================================================
# Runs the REAL backtester (same engine, same seeded latency, same fees as
# run_one) for BOTH strategies (naive + micro) per symbol-day, captures the
# engine's actual queue-gated fills (t/side/px/qty/reason), ASOF-joins each
# fill to the feature store (book state at fill time + forward mid for the
# markout), and persists one parquet per (strategy, symbol, day):
#
#   /Users/shazzak/Capital Stake - Results/fills/{strategy}/{sym}/date={dt}.parquet
#
# fill_attribution.py then reads THESE real fills instead of synthesizing a
# fill-on-every-trade proxy from the trades table. Attribution reconciles to
# the backtest by construction, because the fills ARE the backtest's fills.
#
# RECONCILIATION GUARANTEES (why this matches your backtests exactly):
#   - Same Backtester, same CFG, same LatencyModel(seed=LATENCY_SEED) as
#     run_one -- identical latency draws per symbol-day, order-independent.
#   - Strategy construction mirrors make_strategy(): micro gets session_ms,
#     naive gets STRAT -- but we build BOTH here explicitly (no USE_MICRO
#     flag dependence, so one run persists both strategies' fills).
#   - Fill-time features come from the SAME feature store the labels used;
#     the ASOF direction is backward (last feature row AT OR BEFORE the fill)
#     so no future book state leaks into a fill's context.
#   - Forward mid for markout is the feature store's own event-level mid,
#     first row AT OR AFTER t+HORIZON -- same tested forward-ASOF pattern as
#     the store's own labels (leak-guarded by assertion).
#
# Run FRESH from existing_mm_live/:  python persist_fills.py
# Resume-safe: skips (strategy, symbol, day) parquets that already exist.
# ============================================================================

# Filesystem paths.
from pathlib import Path
# Per-day wall-clock timing.
import time
# Numeric arrays.
import numpy as np
# DataFrames + the tested ASOF joins.
import pandas as pd

# The frozen engine: Backtester + strategies + seeded latency.
from mm_backtest import Backtester, NaiveSymmetricMM, LatencyModel
# The micro strategy (constructed explicitly per symbol-day).
from micro_mm import MicrostructureMM
# The driver module: loader, event builder, CFG, STRAT, MICRO_PARAMS, dates.
import run_legacy_mm as R

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# Raw parsed store (the moved location; run_legacy_mm's constant is stale).
PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# Override the loader's root IN THIS PROCESS ONLY (run_legacy_mm.py untouched).
R.PARSED_ROOT = PARSED_ROOT
# Results root (outside the git project).
RESULTS_ROOT = Path("/Users/shazzak/Capital Stake - Results")
# The feature store built by build_feature_store.py (fill-time context source).
FS_ROOT = RESULTS_ROOT / "feature_store"
# Where the real-fill parquets land: fills/{strategy}/{sym}/date={dt}.parquet
FILLS_ROOT = RESULTS_ROOT / "fills"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Symbols to persist fills for. Start with the two validated pilots; widen to
# the 38-name watchlist once the feature store build completes for them
# (fills REQUIRE the symbol's feature store to exist for the context join).
SYMBOLS = ["PPL", "UBL"]
SYMBOLS = ['AKBL', 'ATRL', 'BAFL', 'BOP', 'DGKC', 'ENGROH', 'FFC', 'FNEL', 'HASCOL', 'HBL', 'HUBC', 'KEL', 'LUCK', 'MARI', 'MEBL', 'MLCF', 'NBP', 'NCPL', 'NML', 'NPL', 'NRL', 'OGDC', 'PACE', 'PAEL', 'PIAHCLA', 'PIBTL', 'PIOC', 'PPL', 'PSO', 'PTC', 'SAZEW', 'SEARL', 'SYS', 'THCCL', 'TOMCL', 'TPL', 'TRG', 'UBL']
# Markout horizon for the fill-level forward mid (matches attribution's 5s).
HORIZON_MS = 5000
# Both strategies, persisted side by side under fills/{strategy}/...
STRATEGIES = ["naive", "micro"]

# ---------------------------------------------------------------------------
# STRATEGY FACTORY -- mirrors run_legacy_mm.make_strategy() but takes the
# strategy name explicitly, so ONE run persists both without flipping USE_MICRO.
# ---------------------------------------------------------------------------
def build_strategy(name, session_ms):
    # Micro: session-aware (Ho-Stoll horizon tau needs the session span).
    if name == "micro":
        # Same params dict run_one uses -- single source of truth.
        return MicrostructureMM(session_ms=session_ms, **R.MICRO_PARAMS)
    # Naive: session-agnostic symmetric quoting, same STRAT params as run_one.
    return NaiveSymmetricMM(**R.STRAT)


# ---------------------------------------------------------------------------
# FILL-TIME CONTEXT JOIN -- attach book state (backward ASOF) and forward mid
# (forward ASOF) from the feature store to each real fill.
# ---------------------------------------------------------------------------
def join_fill_context(fills, fs_day):
    # Sort fills by fill time (ASOF requires sorted keys).
    fills = fills.sort_values("t").reset_index(drop=True)
    # Feature timeline sorted by event time.
    fs_day = fs_day.sort_values("ts_exch").reset_index(drop=True)
    # --- BACKWARD ASOF: book state AT OR BEFORE the fill (no future leak) ---
    # Context columns the attribution needs at fill time.
    ctx_cols = ["ts_exch", "mid", "spread_bps", "obi_1", "toxicity", "realized_vol_bps"]
    # Last feature row at or before each fill's timestamp.
    ctx = pd.merge_asof(
        fills[["t"]], fs_day[ctx_cols],
        left_on="t", right_on="ts_exch", direction="backward")
    # HARD GUARD: context must never postdate the fill (backward means <= t).
    ok_b = (ctx["ts_exch"].dropna() <= ctx.loc[ctx["ts_exch"].notna(), "t"]).all()
    assert ok_b, "CONTEXT LEAK: a fill's book-state context postdates the fill"
    # Attach context under attribution's expected names (mid0 = mid at fill).
    fills["mid0"] = ctx["mid"].values
    fills["spread_bps"] = ctx["spread_bps"].values
    fills["obi_1"] = ctx["obi_1"].values
    fills["toxicity"] = ctx["toxicity"].values
    fills["realized_vol_bps"] = ctx["realized_vol_bps"].values
    # --- FORWARD ASOF: first mid AT OR AFTER t + HORIZON (markout reference) ---
    # Target time per fill.
    tgt = fills[["t"]].copy()
    tgt["ts_target"] = fills["t"] + HORIZON_MS
    # Forward join against the same event-level mid timeline the labels used.
    fwd = pd.merge_asof(
        tgt.sort_values("ts_target"),
        fs_day[["ts_exch", "mid"]].rename(columns={"mid": "mid_h"}),
        left_on="ts_target", right_on="ts_exch",
        direction="forward", suffixes=("", "_f"))
    # Restore fill order.
    fwd = fwd.sort_values("t").reset_index(drop=True)
    # HARD GUARD: the matched forward mid must be dated >= t + HORIZON.
    ok_f = (fwd["ts_exch"].fillna(fwd["ts_target"]) >= fwd["ts_target"]).all()
    assert ok_f, "LOOK-AHEAD LEAK: a fill's forward mid predates t+HORIZON"
    # Attach the forward mid (NaN near session close -- correct, not fabricated).
    fills["mid_h"] = fwd["mid_h"].values
    # Done: every real fill now carries book state + forward mid.
    return fills


# ---------------------------------------------------------------------------
# ONE (strategy, symbol, day): run the REAL backtest, join context, persist.
# ---------------------------------------------------------------------------
def persist_one(strategy, date, sym, dsets, fs_day):
    # Load the three tables with the SAME loader run_one uses.
    u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
    t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
    # Not runnable without a book and trades.
    if len(t) == 0 or len(s) == 0:
        return None
    # Merged event stream (identical contract to run_one).
    events, snap_groups, t = R.build_events(u, s, t)
    # Continuous session window (auctions excluded) -- same as run_one.
    cont = t[t["initiator"] != "AUCTION"]
    if len(cont) == 0:
        return None
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
    # cfg parity with run_one: same CFG, same seeded latency per symbol-day.
    cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED))
    # Build THIS strategy explicitly (no USE_MICRO flag dependence).
    strat = build_strategy(strategy, session_ms=(t0, t1))
    # Run the real backtest.
    bt = Backtester(strat, cfg)
    fills, equity, stats = bt.run(events, snap_groups)
    # No fills -> nothing to persist for this (strategy, symbol, day).
    if fills is None or len(fills) == 0:
        return 0
    # Join book-state context + forward mid from the feature store.
    fills = join_fill_context(fills, fs_day)
    # Stamp identity so attribution can group without path parsing.
    fills["strategy"] = strategy
    fills["symbol"] = sym
    fills["date"] = date
    # Output path: fills/{strategy}/{sym}/date={dt}.parquet
    out_dir = FILLS_ROOT / strategy / sym
    out_dir.mkdir(parents=True, exist_ok=True)
    fills.to_parquet(out_dir / f"date={date}.parquet", index=False)
    # Report the fill count persisted.
    return int(len(fills))


# ---------------------------------------------------------------------------
# MAIN -- loop dates x symbols x strategies, resume-safe.
# ---------------------------------------------------------------------------
def main():
    # All trading dates in the parsed store.
    dates = R.discover_dates()
    # Announce scope.
    print(f"{len(STRATEGIES)} strategies x {len(SYMBOLS)} symbols x {len(dates)} dates -> {FILLS_ROOT}")
    # Walk every date.
    for di, date in enumerate(dates, 1):
        # Open the date's datasets once (shared across symbols + strategies).
        dsets = R.open_datasets(date)
        # Missing partition -> skip the date.
        if dsets is None:
            print(f"  [{di}/{len(dates)}] {date} no datasets; skip"); continue
        # Time the date.
        dt0 = time.perf_counter()
        # Each symbol.
        for sym in SYMBOLS:
            # The symbol-day's feature store partition (fill-time context source).
            fs_path = FS_ROOT / sym / f"date={date}.parquet"
            # Fills REQUIRE the feature store for the context join; skip if absent.
            if not fs_path.exists():
                continue
            # Load the day's feature rows once (shared by both strategies).
            fs_day = pd.read_parquet(
                fs_path,
                columns=["ts_exch", "mid", "spread_bps", "obi_1",
                         "toxicity", "realized_vol_bps"])
            # Each strategy.
            for strategy in STRATEGIES:
                # Resume support: skip (strategy, symbol, day) already on disk
                # BEFORE the expensive backtest replay.
                out = FILLS_ROOT / strategy / sym / f"date={date}.parquet"
                if out.exists():
                    continue
                # Isolate failures per unit.
                try:
                    n = persist_one(strategy, date, sym, dsets, fs_day)
                except Exception as e:
                    print(f"    {strategy}/{sym} {date} ERROR {e!r}"); continue
        # Per-date progress with timing.
        print(f"  [{di}/{len(dates)}] {date} {time.perf_counter()-dt0:.2f}s")
    # Done.
    print("done.")


# Entry point.
if __name__ == "__main__":
    main()
