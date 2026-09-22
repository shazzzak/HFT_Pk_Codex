"""Replay frozen assigned jobs through strategy, exchange, durable OMS and account risk."""
# Parse explicit scope and resource limits for user-operated large runs.
import argparse
# Hash source and decoded data rather than trusting filenames.
import hashlib
# Persist auditable manifests and verdicts.
import json
# Reject invalid numerical reconciliation results.
import math
# Resolve the current checkout without the original HFT project.
from pathlib import Path
# Capture the exact normalized input stream consumed by each cell.
import pickle
# Set imports from this script's current checkout.
import sys
# Time each cell so the first smoke informs full-run sizing.
import time
# Keep long-running cells observable without blocking the parent heartbeat.
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
# Start clean workers with no inherited SQLite ownership.
import multiprocessing
# Prefer this Production tree for all production imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Prefer its sibling strategy/backtest tree, never another checkout on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "existing_mm_live"))
# Reuse the existing gate's input loading and corrected baseline.
from sim import gate as G
# Reuse deterministic serialization and execution comparison utilities.
from sim.gate_refresh import digest, save, executions
# Run the real durable financial boundary against the simulator.
from sim.account_replay import build, ledger_check, restart_check, test_limits
# Reuse the exact simulator fee function for prefix valuation.
from mm_backtest import fee_for
# Freeze immutable financial test settings.
from dataclasses import asdict
# Report the selected canonical data roots explicitly.
import config_pk
# Report stage liveness and measured work without accessing account state from threads.
from sim.replay_progress import Heartbeat, hash_files


# Require condition checks even when Python is invoked with optimization enabled.
def require(condition, message):
    # A violated gate invariant is always a failure, never an omitted assertion.
    if not condition:
        # Preserve the actionable cause in per-cell evidence.
        raise ValueError(message)


# Value a prefix at its last observed book without fabricating an end-of-day event.
def valuation(engine, prefix):
    # Full sessions retain the original gate's analytical EOD convention.
    if not prefix:
        # No closing valuation is an error handled by the caller.
        return G.pnl_of(engine)
    # Walk only the current observed book; this never creates actual OMS executions.
    cash, unfilled, _, _ = engine.book.liquidation_value(engine.pos, fee_fn=fee_for)
    # Use the same conservative residual-mark convention as the underlying backtest.
    bid, _, ask, _ = engine.book.bbo()
    # A valid current mid takes precedence over the last two-sided observation.
    reference = (bid+ask)/2 if bid is not None and ask is not None else engine.last_good_mid
    # Unpriceable residual inventory must not disappear from the valuation.
    if unfilled and reference is None:
        # Report missing valuation to the caller's finite-accounting check.
        return None
    # Preserve the liquidation direction when applying the residual haircut.
    sign = 1 if engine.pos > 0 else -1
    # This is prefix valuation only, never realized cash or flattened inventory.
    return engine.cash + cash + (sign*unfilled*reference*(1-sign*engine.cfg.get("unfilled_haircut_pct", 0.03)) if unfilled else 0)


# Compare one frozen symbol-day with a separate deliberately unfunded account arm.
def _cell(job, output, max_events, budget, progress):
    # Preserve identity and clip size even if input loading fails.
    row = {key: job[key] for key in ("symbol", "date", "clip", "position_limit", "assignment")}
    # Measure realistic durable replay cost, including journal verification.
    started = time.monotonic()
    # Keep every opened ledger owned and closed within this worker.
    active = None
    # Never silently skip a cell with missing data or incomplete reconciliation.
    try:
        # Open the canonical parsed datasets for exactly the frozen trading date.
        progress.stage("opening datasets")
        # Preserve the existing loader and its input scope.
        datasets = G.R.open_datasets(job["date"])
        # Absence counts as a failed planned cell.
        require(datasets is not None, "missing parsed datasets")
        # Normalize input once and share it across all three replay arms.
        progress.stage("loading symbol-day")
        # Loading does not yet expose an event count; elapsed time remains visible.
        loaded = G.load_symbol_day(datasets, job["symbol"])
        # A missing symbol is not zero trading and cannot pass.
        require(loaded is not None, "unrunnable symbol-day")
        # Preserve original session hours even in a prefix smoke.
        events, snapshots, start, end, reference = loaded
        # Record full decoded scope before selecting a bounded prefix.
        row["available_events"] = len(events)
        # A smoke preserves preceding state and limits continuous-session events.
        if max_events:
            # Locate the requested event boundary without changing event ordering.
            indices = [i for i, event in enumerate(events) if event[0] >= start]
            # No session events is not a useful smoke.
            require(bool(indices), "no in-session events")
            # Keep all earlier state-building events and the selected session prefix.
            events = events[:indices[min(max_events, len(indices))-1]+1]
        # Freeze the actual consumed prefix as well as the full file-level inputs.
        row["consumed_events"] = len(events)
        # Normalize namedtuple records so hashing does not depend on generated classes.
        progress.stage("hashing decoded inputs")
        # Preserve the exact decoded-input fingerprint.
        row["decoded_input_sha256"] = hashlib.sha256(pickle.dumps(([tuple(e[:4]) + (tuple(e[4]),) for e in events], snapshots, start, end, reference), protocol=5)).hexdigest()
        # Run the corrected unmodified backtest on exactly these input events.
        baseline = G.run_baseline(progress.track(events, "baseline replay"), snapshots, job["params"], start, end, True)
        # Create an isolated durable database outside the repository.
        progress.stage("creating parity ledger")
        # Keep all durable initialization semantics unchanged.
        active = build(job, start, end, reference, Path(output) / "parity.sqlite", journal_budget_bytes=budget)
        # Run real strategy decisions through production account checks and commits.
        active.run(progress.track(events, "durable parity replay"), snapshots)
        # Both paths must retain execution of limits that cross during transit.
        require(baseline.cross_on_arrival and active.cross_on_arrival, "cross-on-arrival correction disabled")
        # Compare actual execution sequence including analytical EOD rows separately.
        row["fills_equal"] = executions(baseline) == executions(active)
        # Compare finite end-of-input liquidation valuation, not invented live flattening.
        bp, ep = valuation(baseline, bool(max_events)), valuation(active, bool(max_events))
        # Missing accounting cannot become a pass through NaN comparisons.
        require(bp is not None and ep is not None and all(math.isfinite(float(x)) for x in (bp, ep, baseline.cash, active.cash, baseline.pos, active.pos)), "missing/nonfinite accounting")
        # Record ordinary simulator equality independently of integer ledger equality.
        row.update(backtest_pnl=float(bp), engine_pnl=float(ep), pnl_delta=float(ep-bp), cash_delta=float(active.cash-baseline.cash), position_delta=float(active.pos-baseline.pos))
        # Preserve the same lifecycle counters checked by the earlier gate.
        keys = ("n_orders_sent", "n_cancels", "n_cfos", "crossed_on_arrival", "crossed_on_arrival_shares")
        # Report both values rather than hiding them behind a boolean.
        row["counters"] = {key: [baseline.stats.get(key, 0), active.stats.get(key, 0)] for key in keys}
        # Normal parity limits must not alter strategy output.
        row["risk_rejections"] = active.risk_rejections
        # An order silently overwriting another side invalidates reproduction.
        row["occupied_side"] = active.engine_stats.get("placed_onto_occupied_side", 0)
        # Check exact position, fees and cash from individual exchange executions.
        progress.stage("checking exact ledger")
        # Financial comparisons remain on the serial worker thread.
        row["ledger"] = ledger_check(active)
        # Retain pre-restart account status and journal size for operational review.
        row["account_status"] = active.manager.status()
        # Record the actual disk cost before reopening the account.
        row["journal_bytes"] = active.manager.store.path.stat().st_size
        # Reopen a fresh account owner, preserving liabilities and disabling quoting.
        progress.stage("reopening and verifying journal", detail=f"journal {row["journal_bytes"]/1000000:.1f} MB")
        # Verification can take time; report elapsed time without claiming record progress.
        row["restart"] = restart_check(active)
        # A restarted owner has already been closed by the restart check.
        active = None
        # Exercise risk binding using the same real strategy and captured market data.
        progress.stage("creating unfunded ledger")
        # Keep the same intentionally binding account limits.
        active = build(job, start, end, reference, Path(output) / "unfunded.sqlite", blocked=True, journal_budget_bytes=budget)
        # The unfunded account must remain unable to enter orders.
        active.run(progress.track(events, "unfunded replay"), snapshots)
        # Demand an actual account-check rejection, not merely an inactive strategy.
        denied = [item for item in active.risk_rejections if item["check"] == "account_risk"]
        # Record the financial and order outcomes independently of parity P&L.
        row["binding_risk"] = dict(account_rejections=len(denied), sample=denied[:3], orders_sent=active.stats.get("n_orders_sent", 0), fills=len(active.fills), position=active.pos, cash=active.cash)
        # Empty funding and zero short permissions must block all exchange activity.
        row["binding_risk"]["passed"] = bool(denied) and not active.fills and active.pos == 0 and active.cash == 0 and active.stats.get("n_orders_sent", 0) == 0
        # No unexplained simulator or production cash gap is acceptable.
        row["passed"] = row["fills_equal"] and row["pnl_delta"] == 0 and row["cash_delta"] == 0 and row["position_delta"] == 0 and all(a == b for a, b in row["counters"].values()) and not row["risk_rejections"] and row["ledger"]["passed"] and row["restart"]["passed"] and row["binding_risk"]["passed"] and not row["account_status"]["killed"] and not row["occupied_side"]
    # Errors remain inside the planned denominator with their exact cause.
    except Exception as error:
        # Keep any partial evidence and mark the cell incomplete/failed.
        row.update(passed=False, error=repr(error))
    # Close an account regardless of simulator or reconciliation outcome.
    finally:
        # Restart checks may already have released the original store.
        if active is not None and not active.manager.store.lock.closed:
            # Leave database evidence on disk but release single-writer ownership.
            active.manager.store.close()
    # Let users size a larger run from observed cost rather than guessed runtime.
    row["elapsed_seconds"] = time.monotonic() - started
    # No offline gate constitutes live trading approval.
    row["live_approved"] = False
    # Return compact evidence to the parent; retain databases outside the repo.
    return row


# Own one reporter per worker cell and stop it on every exit path.
def cell(job, output, max_events, budget):
    # Report the active symbol/date, including stages with no exposed counters.
    with Heartbeat(f"{job['symbol']} {job['date']}") as progress:
        # Keep exception handling and all financial logic inside the original worker.
        return _cell(job, output, max_events, budget, progress)


# Freeze selected scope, existing calibration and current sources before replay.
def main():
    # Require explicit frozen input and an unused evidence directory.
    parser = argparse.ArgumentParser(description=__doc__)
    # Reuse resolved production clips and strategy settings from the assigned gate.
    parser.add_argument("--frozen-jobs", type=Path, required=True)
    # Six cells provide plumbing evidence; full uses the complete frozen list.
    parser.add_argument("--smoke", action="store_true")
    # Zero means every event; a positive value is explicitly a prefix smoke.
    parser.add_argument("--max-events", type=int, default=0)
    # Bound disk usage of each durable parity or binding arm.
    parser.add_argument("--journal-budget-mb", type=int, default=2000)
    # Avoid saturating a laptop's synchronous storage.
    parser.add_argument("--workers", type=int, default=1)
    # Evidence must remain outside the source checkout.
    parser.add_argument("--output-dir", type=Path, required=True)
    # Parse the user's single terminal statement.
    args = parser.parse_args()
    # Prevent accidentally describing a prefix as the full replay gate.
    require(args.max_events >= 0 and (args.smoke or args.max_events == 0) and args.workers > 0 and args.journal_budget_mb > 0, "invalid scope or resource settings")
    # Resolve actual roots and refuse the original project or mixed parsed stores.
    root = Path(__file__).resolve().parents[2]
    # The checked-out code must be the canonical production/backtest pair.
    require(root == config_pk.PAKISTAN_ROOT.resolve(), "runner must use configured HFT_Pk_Codex checkout")
    # All backtest input loading must use the same canonical parsed store.
    require(G.R.PARSED_ROOT.resolve() == config_pk.PARSED_ROOT.resolve(), "parsed root mismatch")
    # Generated outputs must not be written into any part of the source repository.
    require(not args.output_dir.resolve().is_relative_to(config_pk.PROJECT_ROOT.resolve()), "output directory must be outside repository")
    # Load the earlier manifest as data, not executable instructions.
    frozen = json.loads(args.frozen_jobs.read_text())
    # Require production assignment scope, published-band gate and unchanged policy.
    require(frozen["profile"] == "assigned" and frozen["oms_policy"] == "exact" and frozen["use_replace"] is True and frozen["house_band_pct"] == G.HOUSE_BAND_PCT and frozen["latency_seed"] == G.R.LATENCY_SEED, "incompatible frozen gate policy")
    # Simulator settings must agree exactly with the previous corrected baseline.
    require(frozen["simulator_config"] == G.R.CFG, "simulator configuration changed")
    # Keep the entire explicit job list unless the user requests a smoke.
    jobs = frozen["jobs"]
    # Scope exclusions such as DROP remain explicit in the new evidence.
    excluded = frozen.get("excluded", [])
    # A smoke selects two names across the final three frozen dates.
    if args.smoke:
        # Preserve manifest symbol order without duplicating symbols.
        names = list(dict.fromkeys(job["symbol"] for job in jobs))[:2]
        # Select the most recent three frozen trading dates.
        dates = sorted({job["date"] for job in jobs})[-3:]
        # Retain the original calibrated parameters and assigned clip in every cell.
        jobs = [job for job in jobs if job["symbol"] in names and job["date"] in dates]
        # A shortened sample must not masquerade as the promised six cells.
        require(len(jobs) == 6, "smoke requires two names and three dates")
    # Empty scope or duplicate identities cannot satisfy a reconciliation gate.
    require(bool(jobs) and len({(job["symbol"], job["date"]) for job in jobs}) == len(jobs), "empty or duplicate jobs")
    # Freeze all current source modules, including dynamically imported local code.
    paths = {p for tree in (root / "Production", root / "existing_mm_live") for p in tree.rglob("*.py") if not {"docs", "__pycache__", ".git"}.intersection(p.relative_to(tree).parts)}
    # Bind the frozen job parameters themselves to the run.
    paths.add(args.frozen_jobs.resolve())
    # Previous code hashes are superseded; calibration/assignment bytes must match.
    calibrations = {Path(p): value for p, value in frozen["hashes"].items() if Path(p).is_relative_to(config_pk.RESULTS_ROOT)}
    # Require a real calibration closure rather than an empty permissive manifest.
    require(len(calibrations) >= 5, "frozen calibration closure missing")
    # Validate every previously selected data/config input before any new simulation.
    for path, expected in calibrations.items():
        # Fail rather than silently selecting a newer assignment or calibration.
        require(path.is_file() and digest(path) == expected, f"changed calibration: {path}")
        # Include verified inputs in this run's before/after freeze.
        paths.add(path)
    # Freeze all consumed parquet partitions once per date, not once per symbol.
    data_paths = set()
    # Keep the raw date scope explicit and finite.
    for day in sorted({job["date"] for job in jobs}):
        # Include each table used by the normalized market-data loader.
        for table in ("trades", "ob_updates", "ob_snapshot"):
            # Preserve the entire partition inventory to catch additions/removals.
            partition = set((config_pk.PARSED_ROOT / table / f"date={day}").rglob("*.parquet"))
            # A missing required partition invalidates the run before dispatch.
            require(bool(partition), f"missing partition: {table} {day}")
            # Freeze the selected raw input bytes.
            data_paths.update(partition)
    # Include both code/calibration and raw market inputs in provenance.
    paths.update(data_paths)
    # Never overwrite an earlier run's evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Hashing can take time for raw data; make the phase visible.
    print(f"Freezing {len(paths)} source/config/data files for {len(jobs)} cells", flush=True)
    # Record exact input bytes before worker processes import and consume them.
    hashes = hash_files(paths, "initial input freeze")
    # Scope and quantization limitations belong in the machine-readable evidence.
    manifest = dict(jobs=jobs, excluded=excluded, hashes=hashes, parsed_root=str(config_pk.PARSED_ROOT), results_root=str(config_pk.RESULTS_ROOT), python=sys.version, pandas=G.pd.__version__, account_limits=asdict(test_limits()), max_events=args.max_events, journal_budget_mb=args.journal_budget_mb, house_band_pct=G.HOUSE_BAND_PCT, latency_seed=G.R.LATENCY_SEED, simulator_config=G.R.CFG, oms_policy="exact", use_replace=True, fee_rounding="HALF_UP per actual execution to paisa; test convention pending actual fee statement", recovery_scope="fresh-owner checkpoint reload, no resumed venue session", mark_scope="historical BBO at quote checkpoints; one-day age limit, not a live feed timing test", portfolio_scope="one isolated account per symbol-day; cross-symbol checks remain in account_risk_check", live_approved=False)
    # Persist the frozen experiment before any replay starts.
    save(args.output_dir / "manifest.json", manifest)
    # Stream one durable cell result even if another cell fails.
    results = []
    # Fresh processes avoid inheriting mutable simulator state and writer locks.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        # Associate futures with explicit symbol/date scope.
        pending = {}
        # Submit each independent cell exactly once.
        for job in jobs:
            # Keep both SQLite ledgers and the cell result together.
            folder = args.output_dir / f"{job['symbol']}_{job['date']}"
            # Avoid accidental reuse of existing account databases.
            folder.mkdir()
            # A cell uses real durability even during a prefix smoke.
            pending[pool.submit(cell, job, str(folder), args.max_events, args.journal_budget_mb*1000000)] = folder
        # Keep printing liveness during synchronous journal writes.
        while pending:
            # Wait at most ten seconds before another progress update.
            ready, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            # Long cells must not look like a frozen terminal.
            print(f"integrated gate {len(results)}/{len(jobs)} complete", flush=True)
            # Persist every completed result before waiting again.
            for future in ready:
                # Resolve its already-created evidence directory.
                folder = pending.pop(future)
                # A worker crash belongs in the failed denominator too.
                try:
                    # Obtain the worker's explicit reconciliation evidence.
                    result = future.result()
                # Retain catastrophic process failures as incomplete cells.
                except Exception as error:
                    # The folder name still identifies the failed symbol-day.
                    result = dict(passed=False, error=repr(error), cell=folder.name)
                # Preserve all results for final coverage checks.
                results.append(result)
                # Do not wait for the whole matrix to persist completed evidence.
                save(folder / "result.json", result)
                # Report the verdict and concrete failure if present.
                print(f"{folder.name}: {'PASS' if result['passed'] else 'FAIL'} {result.get('error', '')}", flush=True)
    # An edit to any frozen source or input invalidates this experiment.
    after_hashes = hash_files([path for path in hashes if Path(path).is_file()], "final input verification")
    # Compare the same bytes and missing files as before, without silent hashing phases.
    changed = [path for path, expected in hashes.items() if after_hashes.get(path) != expected]
    # Detect new/deleted raw files as well as modified existing ones.
    after_data = {p for day in {job["date"] for job in jobs} for table in ("trades", "ob_updates", "ob_snapshot") for p in (config_pk.PARSED_ROOT / table / f"date={day}").rglob("*.parquet")}
    # Report partition inventory changes explicitly.
    changed.extend(str(p) for p in sorted(data_paths.symmetric_difference(after_data)))
    # At least one actual execution is required to exercise cash and fill deduplication.
    fills = sum(row.get("ledger", {}).get("execution_count", 0) for row in results)
    # A vacuous no-fill sample is incomplete coverage even if every zero matched.
    passed = len(results) == len(jobs) and all(row["passed"] for row in results) and fills > 0 and not changed
    # Label prefix smoke honestly; only all-event full scope can be the full gate.
    summary = dict(passed=passed, scope="prefix_smoke" if args.max_events else "six_cell_smoke" if args.smoke else "full_frozen_scope", planned=len(jobs), completed=len(results), failed=sum(not row["passed"] for row in results), execution_reports=fills, changed_inputs=changed, coverage_error="no actual fills; extend sample" if not fills else None, live_approved=False)
    # Save the final verdict after provenance and coverage checks.
    save(args.output_dir / "summary.json", summary)
    # Provide a compact pasteable result for follow-up review.
    print(json.dumps(summary), flush=True)
    # Stop chained terminal commands on any failure or incomplete coverage.
    return 0 if passed else 1


# Spawn workers only from the parent entry point.
if __name__ == "__main__":
    # Propagate the gate verdict to the shell.
    raise SystemExit(main())
