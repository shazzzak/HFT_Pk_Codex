# Anonymous-reference correction

## Snapshot refresh update — 25 September 2026

The current planner and engine replace market depth at every **validated, causally ordered snapshot**, including snapshots arriving inside an active research window. They apply subsequent additions, cancellations and trades to that replacement. A replacement does not create a new flat account: cash, inventory, live orders, pending requests and acknowledgments remain in the same engine. Old market quantities and prices absent from the replacement are discarded. Earlier reference-to-price observations are retained as historical evidence, not carried-forward quantity.

This corrects the earlier clean-window implementation's decision to skip snapshots inside active windows. That earlier restriction was introduced in the assistant-authored clean-window package, not requested by the user as the intended market-book design.

**This does not yet adopt every five-second snapshot unconditionally.** The parsed book timestamp has one-second precision and no shared application-sequence watermark. A snapshot with intervening symbol mutations between its coarse source time and conservative availability still fails `SNAPSHOT_TIME_AMBIGUOUS`; malformed levels, gaps and noncontinuous phases also remain excluded. The planner now considers these replacements even while a window is active and reports their timing rejections. Adoption occurs no earlier than receipt and the end of the coarse source second. Resolving ambiguous snapshots requires better alignment evidence, not an assumed event order.

The 26 September repair also distinguishes cancel/re-add generations that reuse an exchange order ID. A snapshot links only to the latest matching generation known at its causal cutoff. Individual execution keys use channel and add sequence, not the reusable ID. An earlier valid snapshot is no longer rejected because that ID appears in a later amendment. Superseded references cannot consume a replacement generation or an anonymous pool. The original exchange messages for SNGP on 21 January confirm one such cancellation/replacement chain.

Queue handling preserves individually identified orders' known arrival times. Anonymous replacement quantity is conservatively treated as ahead of an already-live simulated order because its arrival order is unknown. This is a conservative queue bound, **not an exact reconstruction of anonymous priority**. A live quote outside the new reported depth produces `SNAPSHOT_QUEUE_OUTSIDE_REPORTED_DEPTH` rather than an assumed empty queue. Such failures remain visible excluded replay outcomes.

The source-audit CLI now records contract version 2, retained refresh counts and exact refresh times. Strategy runs use experiment identity `retrospective_clean_windows_snapshot_refresh_v4`. Use a new output folder; old pilot/profit results have not been overwritten and must not be resumed with changed code.

Validation scripts: `test_snapshot_refresh.py` exercises replacement, timing and account/queue invariants; `check_snapshot_refresh.py` compares four fixed real stock-days with archived pre-refresh code and replays one bounded TELE interval across twelve arms. Prior-source copies and new results are in `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/snapshot_refresh_validation_20260925_v1`. The checks below describe the earlier anonymous-only version, not a new full-history result.

This is the corrected clean-window research implementation. The historical implementation in `existing_mm_live` remains read-only. This directory does not change the live trading application or historical profit reports.

## Earlier anonymous-only correction

The following describes the earlier version and its original validation. The snapshot-refresh changes above supersede its initial-pool-only queue assumption.

Snapshot quantity at each reported price is split into:

- Individually tracked orders whose snapshot IDs have original add-reference mappings.
- A price-specific anonymous pool containing undisclosed quantity and disclosed snapshot IDs that cannot be addressed through those mappings.

A missing-add trade consumes that anonymous pool using its actual execution price and single resting-side reference, validated against the aggressor side. A successfully applied trade establishes a reference-price observation local to that window. A later unpriced cancellation of the same reference can use that earlier observation. All quantities remain at their actual reported prices. Every initial pool predates simulated orders in the flat-start window, and partial cancellations reduce only the relevant queue-ahead pool quantity.

An unpriced cancellation without prior reference-price evidence remains unavailable as `UNPRICED_ANONYMOUS_CANCEL`. Overconsumption, exhausted known identities, contradictory side/price evidence and future references remain rejected. This correction does not infer an individual order's remaining size from aggregate liquidity or invent deeper prices.

## Validation completed

- Thirteen unit/integration regression tests passed.
- Three source-only stock-days reproduced the original saved candidate windows and source counters exactly before testing the correction: AGHA 2026-01-01, AICL 2026-04-06, TELE 2026-04-29.
- Corrected candidate coverage changed by approximately -4.199 seconds, 0 seconds and +55 seconds respectively. Longer reconstruction can change subsequent checkpoint choices, so coverage is not necessarily monotonic.
- One retained TELE interval containing a missing-add trade ran through all twelve original strategy arms. Each processed one anonymous-reference trade reduction and closed flat successfully. This was a bounded integration check, not a full-day profitability comparison.
- The corrected runner's `--help` import check passed. Full-history source coverage, closing outcomes and profitability have not been recomputed.

Evidence: `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/Anonymous_Reference_Fix_20260925_v1`.

## Execution

The guarded `run_corrected_pnl.sh` launcher verifies the completed source reconciliation stamp against current code, checks the frozen inputs, reruns the regression suite, and launches all 113 stocks, 185 dates (October 2025–June 2026), and twelve arms with eight workers into a unique new results directory. Invoke it with `bash` from any directory. Add `--check-only` to perform the same preflight without starting the full replay. The full strategy run has not been performed as part of this repair. It remains retrospective clean-window research, not continuous full-day or live P&L.

The files reuse legacy dependencies read-only. Put this directory before `existing_mm_live` on `PYTHONPATH`; the historical runner must not be used for corrected runs. The new runner records hashes of both source directories and has a distinct experiment identity, preventing resumption into the completed historical run.

Run the regression tests:

```bash
# Use the corrected local modules and the read-only historical dependencies.
PYTHONPATH='/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/Production/clean_window_research:/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/existing_mm_live' \
PYTHONDONTWRITEBYTECODE=1 caffeinate -is \
  '/Users/shazzak/PycharmProjects/HFT_Pk_Codex/backtest/bin/python' \
  '/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/Production/clean_window_research/test_anonymous_reductions.py'
```

With the same environment, `check_source_sample.py` repeats the bounded three-cell source audit and `check_replay_window.py` repeats the single-interval twelve-arm integration check. The latter performs an actual bounded strategy replay. Neither starts a full-history run.

Future broad runs require an explicitly agreed scope and a new results directory. Keep the frozen selection rule, portfolio assignments, sizing, latency seed, fees and risk controls unchanged unless separately authorized.
