# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# persist_fills_v2.py -- Stage 1 persistence of LIQUIDATION-INCLUSIVE fills.
# ============================================================================
# Identical in spirit to persist_fills.py, but:
#   (1) uses the engine that now EMITS per-level EOD liquidation fills
#       (reason="liq") plus one haircut-marked residual fill (reason="liq_residual"),
#   (2) writes to a BRAND-NEW tree so nothing existing is touched:
#         /Users/shazzak/Capital Stake - Results/fills_v2/{strategy}/{sym}/date={dt}.parquet
#   (3) attaches mid0/mid_h via the SAME join_fill_context as the original, so the
#       new parquet is a strict superset (same columns + the liq rows).
#
# The intraday rows are byte-identical to fills/ (same engine, same seed, same
# join). The ONLY difference is the extra reason in ("liq","liq_residual") rows
# at session end. Downstream tools that read fills/ are unaffected; tools that
# want the liquidation itemization read fills_v2/.
#
# Run from existing_mm_live/:  python persist_fills_v2.py
# Resume-safe: skips (strategy, symbol, day) already in fills_v2/.
# ============================================================================

# filesystem paths
from pathlib import Path
# per-day timing
import time
# dataframes
import pandas as pd

# the frozen engine (now with liq-fill emission) + strategies + latency
from mm_backtest import Backtester, NaiveSymmetricMM, LatencyModel
# the micro strategy
from micro_mm import MicrostructureMM
# the driver module (loader, CFG, params, dates)
import run_legacy_mm as R
# REUSE the original context-join + strategy factory -- single source of truth,
# so the intraday rows match fills/ exactly.
import persist_fills as PF

# raw parsed store (same override the original uses)
# Resolve this filesystem path through the canonical checkout/data configuration.
PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# override the loader root in-process only
R.PARSED_ROOT = PARSED_ROOT
# results root
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS_ROOT = Path(str(_hft_paths.RESULTS_ROOT))
# feature store (context source for the mid0/mid_h join)
FS_ROOT = RESULTS_ROOT / "feature_store"
# NEW tree -- never touches the existing fills/ folder
FILLS_V2_ROOT = RESULTS_ROOT / "fills_v2"

# symbols (mirror the original's active set)
SYMBOLS = ["PPL", "UBL"]
# both strategies
# STAGE 1: naive only. Micro requires per-symbol session_scale calibration
# (loaded from session_scales_*.csv and injected by mm_harness.build_micro_params),
# which is not in the raw MICRO_PARAMS dict -- so micro cannot be built here.
# Naive is strategy-independent for the EOD liquidation path and fully exercises
# the new liq-fill emission, which is all Stage 1 needs to verify. Micro fills
# get persisted in Stage 2 via the calibrated harness path.
STRATEGIES = ["naive"]


# one (strategy, symbol, day): run the real backtest (with liq emission),
# join context, persist to fills_v2/.
def persist_one_v2(strategy, date, sym, dsets, fs_day):
    # load the three tables with the SAME loader
    u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
    t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
    # unrunnable without a book and trades
    if len(t) == 0 or len(s) == 0:
        return None
    # merged event stream (same contract)
    events, snap_groups, t = R.build_events(u, s, t)
    # continuous session window (auctions excluded) -- same as the original
    cont = t[t["initiator"] != "AUCTION"]
    if len(cont) == 0:
        return None
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
    # cfg parity: same CFG, same seeded latency per symbol-day
    cfg = dict(R.CFG, session=(t0, t1),
               latency_model=LatencyModel(seed=R.LATENCY_SEED))
    # build THIS strategy explicitly (reuse the original factory)
    strat = PF.build_strategy(strategy, session_ms=(t0, t1))
    # run the real backtest (engine now emits liq fills at EOD)
    bt = Backtester(strat, cfg)
    fills, equity, stats = bt.run(events, snap_groups)
    # no fills -> nothing to persist
    if fills is None or len(fills) == 0:
        return 0
    # SPLIT: intraday fills get the feature-store context join exactly as before;
    # liq fills already carry their own mid0 (the closing mid) from the engine and
    # must NOT be asof-joined (they occur AT/after session end, where a backward
    # asof would mis-assign context). We join intraday, then re-attach liq rows.
    is_liq = fills["reason"].isin(["liq", "liq_residual"])
    # intraday portion (everything that isn't a liquidation fill)
    intraday = fills[~is_liq].copy()
    # liquidation portion (already has mid0 from the engine)
    liqrows = fills[is_liq].copy()
    # join context to intraday only (mirrors persist_fills exactly)
    if len(intraday):
        intraday = PF.join_fill_context(intraday, fs_day)
    # give liq rows the SAME columns as the joined intraday frame:
    if len(liqrows):
        # mid_h is undefined for EOD liq fills (no forward horizon after close):
        # set NaN -- correct, not fabricated. mid0 already present from engine.
        liqrows["mid_h"] = float("nan")
        # the context columns the join adds to intraday but liq rows lack:
        # spread_bps/obi_1/toxicity/realized_vol_bps are NaN for liq (no book
        # state context defined at the crossing event) -- again, honest NaN.
        for c in ("spread_bps", "obi_1", "toxicity", "realized_vol_bps"):
            if c not in liqrows.columns:
                liqrows[c] = float("nan")
    # recombine, restore fill-time order
    out = pd.concat([intraday, liqrows], ignore_index=True).sort_values("t")
    # stamp identity (same as original)
    out["strategy"] = strategy
    out["symbol"] = sym
    out["date"] = date
    # NEW tree path: fills_v2/{strategy}/{sym}/date={dt}.parquet
    out_dir = FILLS_V2_ROOT / strategy / sym
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"date={date}.parquet", index=False)
    # report count persisted (incl. liq rows)
    return int(len(out))


# main loop: dates x symbols x strategies, resume-safe
def main():
    # all trading dates
    dates = R.discover_dates()
    # announce scope + destination
    print(f"{len(STRATEGIES)} strategies x {len(SYMBOLS)} symbols x "
          f"{len(dates)} dates -> {FILLS_V2_ROOT}")
    # walk dates
    for di, date in enumerate(dates, 1):
        # open the date's datasets once
        dsets = R.open_datasets(date)
        # missing partition -> skip the date
        if dsets is None:
            print(f"  [{di}/{len(dates)}] {date} no datasets; skip")
            continue
        # time the date
        dt0 = time.perf_counter()
        # each symbol
        for sym in SYMBOLS:
            # the feature-store partition (context source)
            fs_path = FS_ROOT / sym / f"date={date}.parquet"
            # require the feature store for the context join
            if not fs_path.exists():
                continue
            # load the day's feature rows once
            fs_day = pd.read_parquet(
                fs_path,
                columns=["ts_exch", "mid", "spread_bps", "obi_1",
                         "toxicity", "realized_vol_bps"])
            # each strategy
            for strategy in STRATEGIES:
                # resume: skip already-persisted units
                out = FILLS_V2_ROOT / strategy / sym / f"date={date}.parquet"
                if out.exists():
                    continue
                # isolate failures per unit
                try:
                    persist_one_v2(strategy, date, sym, dsets, fs_day)
                except Exception as e:
                    print(f"    {strategy}/{sym} {date} ERROR {e!r}")
                    continue
        # per-date progress
        print(f"  [{di}/{len(dates)}] {date} {time.perf_counter()-dt0:.2f}s")
    # done
    print("done.")


# entry point
if __name__ == "__main__":
    main()
