"""Strict, frozen reconciliation; no connectivity or live-order submission."""
# Parse explicit research scope and risk settings.
import argparse
# Bind all tested inputs to content digests.
import hashlib
# Persist human-readable manifests and results.
import json
# Reject non-finite reconciliation numbers.
import math
# Use spawn so workers do not inherit mutable simulator state.
import multiprocessing
# Resolve all paths independently of the shell's working directory.
from pathlib import Path
# Fingerprint the exact decoded market input consumed by the simulators.
import pickle
# Record interpreter and dependency provenance.
import sys
# Print liveness while long symbol-days run.
import time
# Limit process concurrency and retain failed cells explicitly.
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
# Permit invocation as either a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Reuse the existing replay plumbing and corrected exchange semantics.
from sim import gate as G
# Use the same validated assignment loader as the production adapter.
from venues.psx_config import LiveConfig


# Hash files without loading large inputs into RAM at once.
def digest(path):
    # Initialize the file hash.
    result = hashlib.sha256()
    # Stream the exact bytes.
    with Path(path).open("rb") as stream:
        # Bound the read buffer.
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            # Include every byte in order.
            result.update(block)
    # Return a portable manifest value.
    return result.hexdigest()


# Write complete JSON files into the user-selected run directory.
def save(path, value):
    # Preserve real newlines and reject non-finite JSON numeric values.
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


# Compare executions in order, independent of internal order identifiers.
def executions(engine):
    # Time, side, price and size define the observed execution stream.
    return [(float(f["t"]), str(f["side"]), float(f["px"]), float(f["qty"])) for f in engine.fills]


# Run one independent symbol-day using already resolved parameters.
def cell(job):
    # Preserve identity even if input loading or either simulator fails.
    row = {k: job[k] for k in ("symbol", "date", "clip", "position_limit", "assignment")}
    # Never silently skip a requested cell.
    try:
        # Read the same captured market data used by the original gate.
        datasets = G.R.open_datasets(job["date"])
        # Missing partitions invalidate the requested scope.
        if datasets is None:
            # Turn an absent day into a recorded failure.
            raise ValueError("missing parsed datasets")
        # Load and normalize one symbol-day once for both runs.
        loaded = G.load_symbol_day(datasets, job["symbol"])
        # Absence cannot count as agreement.
        if loaded is None:
            # Retain this cell as a failure.
            raise ValueError("unrunnable symbol-day")
        # Fingerprint exact decoded inputs, including the session and reference.
        row["decoded_input_sha256"] = hashlib.sha256(pickle.dumps(([tuple(e[:4]) + (tuple(e[4]),) for e in loaded[0]], *loaded[1:]), protocol=5)).hexdigest()
        # Unpack the shared input bundle.
        events, snapshots, start, end, reference = loaded
        # Run the corrected backtest with the declared replacement policy.
        base = G.run_baseline(events, snapshots, job["params"], start, end, True)
        # Run production risk, OMS and adapter against identical captured data.
        engine = G.run_engine(events, snapshots, job["params"], start, end, reference, job["symbol"], job["date"], True, "exact", job["position_limit"])
        # Require the correction to be active on both paths.
        row["cross_on_arrival"] = bool(base.cross_on_arrival and engine.cross_on_arrival)
        # Extract the same liquidation accounting on each side.
        bp, ep = G.pnl_of(base), G.pnl_of(engine)
        # Reject missing or non-finite accounting.
        if bp is None or ep is None or not all(math.isfinite(float(x)) for x in (bp, ep, base.cash, engine.cash, base.pos, engine.pos)):
            # A broken close must never pass a gate.
            raise ValueError("missing or non-finite accounting")
        # Compare P&L separately from fills and lifecycle counters.
        row.update(backtest_pnl=float(bp), engine_pnl=float(ep), pnl_delta=float(ep-bp), cash_delta=float(engine.cash-base.cash), position_delta=float(engine.pos-base.pos))
        # Compare the complete execution sequence, not just aggregate counts.
        row["fills_equal"] = executions(base) == executions(engine)
        # Record counts to make a mismatch easier to diagnose.
        row["fill_counts"] = [len(base.fills), len(engine.fills)]
        # Compare lifecycle behavior including crossing-on-arrival executions.
        keys = ("n_orders_sent", "n_cancels", "n_cfos", "crossed_on_arrival", "crossed_on_arrival_shares")
        # Preserve both values for every selected counter.
        row["counters"] = {key: [int(base.stats.get(key, 0)), int(engine.stats.get(key, 0))] for key in keys}
        # A refusal is material even if it happens to leave P&L unchanged.
        row["risk_rejections"] = engine.risk_rejections
        # Simulator order overwrite is an invalid comparison, never a pass.
        row["occupied_side"] = int(engine.engine_stats.get("placed_onto_occupied_side", 0))
        # Require agreement across all independently reported dimensions.
        row["passed"] = (row["cross_on_arrival"] and row["fills_equal"] and abs(row["pnl_delta"]) < 0.005 and abs(row["cash_delta"]) < 0.005 and row["position_delta"] == 0 and all(a == b for a, b in row["counters"].values()) and not row["risk_rejections"] and not row["occupied_side"])
    # Keep errors visible and let other requested cells finish.
    except Exception as error:
        # A failure is part of the denominator.
        row.update(passed=False, error=repr(error))
    # Return only compact results, not full market books.
    return row


# Freeze scope and settings before dispatching any replay.
def main():
    # Require an explicit output directory to prevent accidental result overwrites.
    parser = argparse.ArgumentParser(description=__doc__)
    # A small run is useful for plumbing but is not the full gate.
    parser.add_argument("--smoke", action="store_true")
    # Choose historical clips or production sizing and current assignment.
    parser.add_argument("--profile", choices=("historical", "assigned"), required=True)
    # Require a declared test envelope instead of guessing live capital limits.
    parser.add_argument("--risk-clips", type=int, required=True)
    # Make assignment selection explicit for the assigned comparison.
    parser.add_argument("--assignment", type=Path)
    # Apply only explicitly selected manual overrides.
    parser.add_argument("--overrides", type=Path)
    # Bound replay concurrency.
    parser.add_argument("--workers", type=int, default=2)
    # Keep all generated evidence together.
    parser.add_argument("--output-dir", type=Path, required=True)
    # Resolve command-line values.
    args = parser.parse_args()
    # Reject invalid or incomplete settings before creating artifacts.
    if args.workers < 1 or args.risk_clips < 1 or (args.profile == "assigned" and args.assignment is None):
        # Explain the missing inputs.
        parser.error("positive workers/risk-clips required; assigned profile requires --assignment")
    # Never overwrite a prior gate result.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Capture actual loaded calibration paths rather than only glob patterns.
    paths = {p: G.H.newest(p) for p in ("session_scales_*.csv", "volume_profile_*.csv", "time_windows_*.csv", "session_segments_*.csv")}
    # Pin the harness loader to this exact selection.
    G.H.newest = lambda pattern: paths[pattern]
    # Resolve calibration once, in the parent process.
    scales, profiles, windows, segments = G.H.load_scales(), G.H.load_profiles(), G.H.load_windows(), G.H.load_segments()
    # Select the established gate sample and latest recorded trading days.
    names = G.GATE_NAMES[:2] if args.smoke else G.GATE_NAMES
    # Keep the full historical calendar for strictly prior sizing observations.
    all_dates = G.R.discover_dates()
    # Freeze the exact requested dates.
    dates = all_dates[-(3 if args.smoke else 20):]
    # A short store cannot silently produce a smaller full gate.
    if len(dates) != (3 if args.smoke else 20):
        # Refuse an incomplete sample.
        raise ValueError("insufficient dates for requested gate scope")
    # Load assignment through the production validator.
    assignment = LiveConfig(args.assignment, args.overrides) if args.profile == "assigned" else None
    # Calculate production clips using the established trailing-ten-day rule.
    stats = G.H.trailing_median_trade_size(all_dates, names, 10, heartbeat=20) if assignment else None
    # Freeze every tested symbol-day's complete strategy parameters.
    jobs = []
    # Keep non-trading assignment decisions explicit.
    excluded = []
    # Materialize the full expected Cartesian scope.
    for date in dates:
        # Resolve each symbol before running any engine.
        for symbol in names:
            # Production assignment includes explicit DROP decisions.
            setting = assignment.params_for(symbol) if assignment else None
            # Missing assignments are errors rather than default configurations.
            if assignment and setting is None:
                # Make the unidentified symbol actionable.
                raise ValueError(f"missing assignment for {symbol}")
            # A flat DROP name should not be quoted by either side.
            if setting and not setting["quote"]:
                # Preserve exclusion in the frozen denominator.
                excluded.append({"symbol": symbol, "date": str(date), "reason": "DROP with zero opening inventory"})
                # Skip constructing a strategy that would quote a dropped name.
                continue
            # Historical gate used a fixed fifty-share clip.
            clip = 50
            # Production sizing excludes the day being tested.
            if assignment:
                # Require ten preceding observations for the symbol.
                median = G.H.trailing_median(stats[symbol], all_dates, date, 10)
                # Missing calibration must fail before replay.
                if median is None or not math.isfinite(median) or median <= 0:
                    # Identify the affected cell.
                    raise ValueError(f"missing sizing history: {symbol} {date}")
                # Match the production runner's three-times-median sizing rule.
                clip = max(1, int(round(3.0 * median)))
            # Require all calibrations; no silent default windows.
            params = G.H.build_micro_params(clip, scales[symbol], profiles[symbol], windows[symbol], segments[str(date)])
            # Feed the same validated assignment into both strategies.
            if assignment:
                # Strip control-plane fields through the loader's dedicated API.
                params.update(assignment.strategy_kwargs(symbol))
            # A per-cell limit is explicit and retained in the manifest.
            jobs.append(dict(symbol=symbol, date=str(date), clip=clip, position_limit=args.risk_clips*clip, assignment=setting["label"] if setting else "historical", params=params))
    # Track all local source modules loaded by the gate and research stack.
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, "__file__", None) and str(m.__file__).endswith(".py") and ("HFT_Pk_Codex" in str(m.__file__))}
    # Include this runner even if it was invoked as __main__.
    sources.add(Path(__file__).resolve())
    # Include calibration and configuration bytes in the provenance closure.
    sources.update(Path(p).resolve() for p in paths.values())
    # Record assignment and overrides when supplied.
    sources.update(p.resolve() for p in (args.assignment, args.overrides) if p is not None and p.exists())
    # Bind exact bytes before workers begin.
    hashes = {str(p): digest(p) for p in sorted(sources)}
    # Record simulator settings, policy, test limits and dependency versions.
    manifest = dict(profile=args.profile, smoke=args.smoke, jobs=jobs, excluded=excluded, hashes=hashes, python=sys.version, pandas=G.pd.__version__, simulator_config=G.R.CFG, latency_seed=G.R.LATENCY_SEED, oms_policy="exact", use_replace=True, house_band_pct=G.HOUSE_BAND_PCT, risk_clips=args.risk_clips, quantity_limit=1_000_000, parsed_root=str(G.R.PARSED_ROOT), live_approved=False)
    # Persist the entire frozen scope before dispatch.
    save(args.output_dir / "manifest.json", manifest)
    # Stream durable per-cell evidence as each worker finishes.
    results = []
    # Start workers with fresh module state.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        # Submit each frozen cell exactly once.
        pending = {pool.submit(cell, job): job for job in jobs}
        # Keep reporting while work remains.
        while pending:
            # A bounded wait provides a liveness heartbeat.
            ready, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            # Show progress even when a single cell takes several minutes.
            print(f"gate {len(results)}/{len(jobs)} complete", flush=True)
            # Handle completed results independently.
            for future in ready:
                # Remove it from the outstanding set.
                job = pending.pop(future)
                # Surface a worker failure as an explicit cell failure.
                try:
                    # Retrieve the worker's compact report.
                    row = future.result()
                # Process failures are not successful skips.
                except Exception as error:
                    # Preserve the cell identity in the failure result.
                    row = dict(symbol=job["symbol"], date=job["date"], passed=False, error=repr(error))
                # Retain one result per dispatched job.
                results.append(row)
                # Persist each completed cell separately for interrupted runs.
                save(args.output_dir / f"{job['symbol']}_{job['date']}.json", row)
                # Report its outcome immediately.
                print(f"{row['symbol']} {row['date']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    # Changes to source or calibration during a run invalidate the freeze.
    changed = [p for p, value in hashes.items() if not Path(p).exists() or digest(p) != value]
    # A gate requires every planned, nonempty cell and unchanged inputs.
    passed = bool(jobs) and len(results) == len(jobs) and all(r["passed"] for r in results) and not changed
    # Summarize scope without claiming smoke is full reconciliation.
    summary = dict(passed=passed, scope="smoke" if args.smoke else "full_12_names_20_dates", planned=len(jobs), completed=len(results), failed=sum(not r["passed"] for r in results), excluded=excluded, changed_inputs=changed, live_approved=False)
    # Persist the final verdict only after all requested work finishes.
    save(args.output_dir / "summary.json", summary)
    # Print a compact machine-readable final outcome.
    print(json.dumps(summary), flush=True)
    # Allow shell automation to stop on any mismatch or incomplete evidence.
    return 0 if passed else 1


# Only the parent process dispatches work.
if __name__ == "__main__":
    # Propagate the gate's failure status to the terminal.
    raise SystemExit(main())
