# Source-only coverage pilot

Run `audit_source_coverage.py`. No edits to existing scripts are required. It loads source events and reconstructs books; it does not invoke the trading engine, evaluate strategies, recalculate profits or change settings.

## Pilot selection

The default pilot contains 72 stock-days: twelve tickers, one date per month from January through June 2026. KEL, PIBTL and TPL are included. Nine further tickers are selected across three groups ranked by mean saved source-window count. This measures fragmentation, not liquidity or volume.

Dates rotate through observed shorter/regular weekday and Friday session categories where available. Friday continuous-session totals below 270 minutes are called shorter; other weekdays use 300 minutes. These are diagnostic categories based on saved sessions, not independently verified exchange schedules. Selection never uses profit or observed improvement from the correction.

The sample provides diagnostic diversity. It is not a probability sample; do not extrapolate its coverage improvement to all 113 tickers.

## Run the pilot

```bash
# Choose a fresh results directory for this pilot run.
audit_output="/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/source_coverage_pilot_$(date +%Y%m%d_%H%M%S)"

# Run source reconstruction with eight workers, no strategy replay, and sleep prevention.
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/source-coverage-mpl \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 caffeinate -is \
  "/Users/shazzak/PycharmProjects/HFT_Pk_Codex/backtest/bin/python" \
  "/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/Production/clean_window_research/audit_source_coverage.py" \
  --scope pilot \
  --workers 8 \
  --output-dir "$audit_output"
```

The script sets its own local/dependency import paths. It prints one progress heartbeat about every 15 seconds with the current ticker/date/phase, elapsed time, estimated remaining time, completed jobs and percentage.

Optional `--plan-only` writes the manifest and checks metadata without decoding parquet. To execute that exact plan afterward, use the same directory with `--resume` and omit `--plan-only`. Interrupted runs likewise resume with the original directory, code, worker count and inputs. Changed code or input fingerprints require a fresh directory.

## What to inspect afterward

- `summary.json`: successful/failed cells, gross gained and lost stock-time, net coverage change, anonymous-reduction counts and projected January–June source-audit runtime.
- `coverage_impact.csv`: per-stock-day durations in both exact milliseconds and readable minutes, event counts and measured runtime.
- `reason_counts_NOT_DURATIONS.csv`: original/corrected planner rejection counts. These counts overlap in meaning and are not unavailable minutes.
- `runtime_strata.csv`: observations supporting each month-by-fragmentation runtime estimate.
- `coverage_and_runtime.png`: gross coverage changes and observed per-cell runtime distribution.
- `selected_jobs.csv`, `contract.json`, `cells/`: exact scope, source hashes/input metadata, before/after windows, rejection counts and atomic checkpoints.
- `failures.csv`: present when failures occurred. Failed cells are not assigned zero impact; any failure gives a nonzero command exit status.

Successful anonymous reductions are deduplicated by event kind and application sequence. Retained counts exclude short discarded attempts and source events outside the retained corrected windows, including the failed timestamp batch. Missing-add trades and missing-add cancellations are reported separately from all anonymous reductions already supported by the original model.

Each successful cell must exactly reproduce the original saved candidate intervals, ending reasons and rejection counters before comparing the corrected book. Update/trade file membership and metadata must match the saved channel screen. Snapshot file metadata is frozen at audit start. These are metadata checks, not complete raw-file byte hashes.

## Expansion decision

1. Resolve any failed baseline reproduction or input-integrity check first.
2. Review gross gains and losses, not only net time. A different restart can shift later windows even when net minutes barely change.
3. Review remaining unpriced cancellations and invalid/exhausted-book terminations. A reduced missing-reference count does not establish that all unavailable time is repaired.
4. Use the runtime projection as a planning estimate. It weights observed mean source-cell times by month and fragmentation-group population, then divides by the worker count. Storage contention, memory pressure, startup, cache state and imperfect workload proxies affect it. No runtime estimate is issued if a stratum is missing or any cell failed.
5. Agree the full source-audit scope before using `--scope full --confirm-full-source-audit` with a new output directory. That scope covers all 13,560 requested January–June stock-days and still executes no strategies.
6. Only then decide whether a separate strategy replay is warranted. Candidate source coverage does not establish closing success or profit, and historical accepted-window results cannot simply be transplanted onto changed windows.

The `--scope smoke` option processes TELE on 2026-04-29 only. It verifies execution and output handling, not representative coverage or full-run runtime.
# Snapshot-refresh version

The current code compares the saved original baseline against anonymous-reference handling **plus validated snapshot replacement inside active windows**. It is a different source contract from the 72-cell pilot completed at 19:41 on 25 September. Use a fresh output directory. `snapshot_refreshes_retained` and `snapshot_refresh_times` record actual adopted replacements; `SNAPSHOT_TIME_AMBIGUOUS` still identifies timing-rejected pictures. This source audit runs no trading strategies. The prior 2.93-hour full-audit estimate predates this change and needs remeasurement before expansion.
