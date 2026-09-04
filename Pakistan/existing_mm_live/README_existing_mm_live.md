# existing_mm_live/ — THE SYSTEM (4 files)

This folder is the runnable market-making system. If you only keep four files,
these are them. Everything here is load-bearing; nothing is a throwaway.

| File                     | What it is                                                                                                                                                                                                                                                     | You run it? |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| `mm_backtest.py`         | **The engine.** `Book` (order-book reconstruction: add/cancel/trade/snapshot), `Backtester` (event loop + queue-position fill logic, the `o.ahead` pool-drain at lines ~700–871), `NaiveSymmetricMM`, fees, `LatencyModel`. Everything else imports from here. | No — imported by the others |
| `micro_mm.py`            | **The micro strategy** (`MicrostructureMM`): Avellaneda-Stoikov inventory skew + adverse-selection spread. The thing being recalibrated. Imports `FEE_TOTAL_PCT` from the engine.                                                                              | No — imported by the driver |
| `run_legacy_mm.py`       | **The driver you execute** for backtests. Loads parquet, builds events, wires the strategy (`USE_MICRO` flag at line 203 switches micro vs naive), runs `Backtester`. Has `main()` + argparse.                                                                 | **YES — this runs backtests** |
| `build_feature_store.py` | **Builds the feature store.** Rides the engine's event loop via a passive `FeatureCollector` to emit features + markout labels per two-sided book event. Edit `SYMBOLS` + run.                                                                                 | **YES — this builds features** |
| `persist_fills.py`       | **Runs the real backtest for both strategies per symbol-day**                                                                                                                                                                                                  |


## To run a backtest
- Set `USE_MICRO` (line 203) True (micro) or False (naive).
- `python run_legacy_mm.py` (from this folder).

## To build the feature store
- Edit `SYMBOLS` list in `build_feature_store.py`.
- `python build_feature_store.py` (from this folder).

## Dependencies
- Third-party: numpy, pandas, pyarrow (in the .backtest venv).
- Data: /Users/shazzak/Capital Stake - Parsed/{trades,ob_updates,ob_snapshot}/date=*/
- Output: /Users/shazzak/Capital Stake - Results/feature_store/{symbol}/date={date}.parquet

## The one fact that matters most
The ONLY valid fill model is `Backtester` in mm_backtest.py (lines 700–871).
Any script that computes fills WITHOUT calling this engine is using a proxy and
will not reconcile. Attribution/analysis must consume the engine's real fills.
