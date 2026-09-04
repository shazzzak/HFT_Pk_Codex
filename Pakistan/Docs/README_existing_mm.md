# existing_mm/ — TOOLS & ANALYSIS (not the system)

These CONSUME the engine's outputs (feature store, backtest fills). None of them
is needed to RUN the market maker — that's existing_mm_live/. Split into three
buckets so you know what to keep, what to re-run, and what to archive.

## BUCKET A — Durable tools you'll re-run (KEEP)
| File | What it does |
|------|--------------|
| `build_watchlist.py` | Builds the persistence-screened watchlist -> mm_watchlist_final.csv (the 38 names). |
| `persistence_metrics.py` | The screen's core metrics (pct_days_top20, edge5_pos, etc.). |
| `ticker_stats_core.py` | Fee definitions + net5/markout logic used across screening. |
| `decile_markout_validation.py` | The EDA you ran: decile curves, day-as-unit CIs, edge histograms. Re-run per feature set. |
| `fill_attribution.py` | Fee-aware capture/markout/net-bps attribution. **NOTE: currently uses a TRADES PROXY, not real engine fills — needs wiring to Backtester (in progress).** |
| `run_mm_batch.py` | Non-destructive batch backtest runner (naive/micro, timestamped output). |
| `sweep_micro.py` | Single-day micro parameter sweep (diagnostic; falsified the min_edge fix). |
| `run_naive_two.py` | Two-name naive backtest runner. |
| `run_all_tickers.py` | Full-universe batch runner. |
| `run_daily_stats.py` | Per-day summary statistics. |
| `feature_store_wiring.py` | Partition enumerator for feature_store/{sym}/date=*.parquet (imported by attribution). |
| `halt_state.py` | Market-halt date handling (imported by attribution). |
| `corp_action_detector.py` / `corp_actions_master.py` | Corporate-action detection (BAMM Phase 4; audited clean). |

## BUCKET B — One-shot diagnostics (ARCHIVE — they answered one question)
| File | Answered |
|------|----------|
| `smoke_fs.py` | Feature-store one-day smoke test (its job is done). |
| `delete.py` | Scratch pad for ad-hoc queries. |
| `dump_codes.py` / `dump_codes_2.py` | One-off code/enum dumps. |
| `map_parquet_datasets.py` | One-off schema/partition mapping. |
| `micro_dev_raw_vs_vol_spread.py` / `micro_dev_vs_vol_spread.py` | micro_dev vs vol/spread investigation (settled: micro_dev dead after OBI). |
| `residualize_micro_on_obi.py` | Confirmed micro_dev carries no info beyond OBI. |
| `effectiveness_over_time.py` | Signal-stability-over-time check. |
| `verify_boundary_v2.py` | Session-boundary / label-coverage verification. |
| `feature_incremental_value.py` | Per-feature incremental-value check. |
| `multi_sym.py` | Early multi-symbol scaffolding. |

## BUCKET C — Resolved / moved
| File | Status |
|------|--------|
| `queue_replay.py` / `queue_sim.py` | Were here on Aug 11; now absent (moved/deleted). If they reimplemented fills, that logic must reconcile to mm_backtest.py's Backtester — the engine is the only valid fill model. |

## Rule going forward
- Need to RUN the system? -> existing_mm_live/ (4 files).
- Need a screen, validation, or attribution? -> Bucket A here.
- Bucket B can be deleted or moved to an archive/ subfolder; they won't be re-run.
