"""Offline crash/restart runner; no sockets, exchange access or trading credentials."""
# Parse reproducible offline run settings.
import argparse
# Fingerprint the actual code used by this runner.
import hashlib
# Read and write machine-readable evidence.
import json
# Terminate a child without running cleanup at selected crash boundaries.
import os
# Locate scripts and write results outside the repository.
from pathlib import Path
# Launch isolated fault-injection workers.
import subprocess
# Preserve the interpreter selected by the operator.
import sys
# Provide a stable local script import root independent of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Use the durable transaction owner.
from core.recovery_store import RecoveryStore
# Use the production-domain recovery facade.
from core.durable_runtime import DurableManager
# Build actual order and desired-state objects.
from core.model import DesiredQuotes, QuoteIntent, Side
# Keep the real OMS diff and reservation implementation.
from core.oms import OrderManager, QuoteTolerance
# Enforce the kill and per-symbol limits during the offline scenarios.
from core.risk import KillSwitch, KillSwitchCheck, PositionLimitCheck, RiskGateway
# Supply an explicitly synthetic trading schedule.
from core.venue import SessionSegment
# Use PSX price/tick validation without any exchange connection.
from venues.psx import PSXVenue, static_session_provider

# Fixed test time makes independent runs reproducible.
DAY, NOW = "2026-09-14", 39600000
# Each case ends a separate child at a different durability boundary.
CASES = ("before_batch_commit", "after_batch_commit", "after_attempt_commit", "during_transport", "before_fill_commit", "after_fill_commit")


# Capture recovery dependencies so restarts cannot silently use changed code.
def identity():
    # Locate the installed production package.
    root = Path(__file__).resolve().parents[1]
    # Bind the database to the actual risk, order model and recovery implementation.
    names = ("core/model.py", "core/oms.py", "core/risk.py", "core/venue.py", "core/recovery_store.py", "core/durable_runtime.py", "venues/psx.py")
    # Explicit test limits must not be confused with approved live account limits.
    return {"schema": 1, "account": "OFFLINE", "session": "RECOVERY", "position_limit": 1000, "quantity_policy": "exact", "code": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}


# Build the real OMS with a synthetic session and a durable local ledger.
def build(path, create=False):
    # Open the exclusively owned local journal first.
    store = RecoveryStore(path, identity(), create=create)
    # Use a full-day synthetic session rather than assuming live market hours.
    venue = PSXVenue(session_provider=static_session_provider({DAY: (SessionSegment(0, 86400000),)}))
    # Bind OMS and gateway to the same kill switch.
    switch = KillSwitch()
    # Retain the usual risk authorization path in every scenario.
    gateway = RiskGateway([KillSwitchCheck(switch), PositionLimitCheck(1000)])
    # Amendments use the existing simulator's remaining-quantity contract.
    oms = OrderManager(venue, gateway, switch, "RECOVERY", account="OFFLINE", tolerance=QuoteTolerance(quantity_policy="exact"), use_replace=True)
    # Restore state before returning the facade to a controller.
    return DurableManager(oms, store), store, switch


# Construct ordinary two-sided strategy intent for one test symbol.
def desired(qty=100):
    # Prices are paisa and sizes are shares throughout the test.
    return DesiredQuotes("OGDC", QuoteIntent(Side.BUY, 9900, qty), QuoteIntent(Side.SELL, 10100, qty))


# Supply a synchronized empty-account fixture; this is not a broker API.
def authorize(manager):
    # Real callers must obtain this complete snapshot from their recovery adapter.
    return manager.reconcile_external({"complete": True, "barrier": "synthetic-startup", "open_orders": [], "positions": {}, "cash_minor": 0}, "offline-test")


# Normalize a test execution with a stable scoped identity supplied separately.
def fill_payload(order_id):
    # Explicit fees allow exact cash checks across duplicate delivery.
    return {"cl_ord_id": order_id, "symbol": "OGDC", "side": "BUY", "price_minor": 9900, "quantity": 7, "timestamp_ms": NOW, "fee_minor": 3}


# Simulate abrupt process death without finally blocks or connection cleanup.
def worker(path, boundary):
    # Only workers initialize the case database.
    manager, store, _ = build(path, create=True)
    # The empty synthetic account is explicitly reconciled before quote permission.
    assert authorize(manager)
    # Capture the original durable commit for controlled fault injection.
    original = store.commit
    # Intercept only the chosen transactional boundary.
    def commit(kind, payload, state):
        # Kill before a mutation becomes durable in the selected case.
        if (boundary == "before_batch_commit" and kind == "reconcile") or (boundary == "before_fill_commit" and kind == "report" and payload["input"]["kind"] == "fill"):
            # Exit bypasses Python cleanup and emulates abrupt process loss.
            os._exit(73)
        # Preserve actual SQLite commit semantics in every other case.
        original(kind, payload, state)
        # Kill immediately after the selected durable transaction succeeds.
        if (boundary == "after_batch_commit" and kind == "reconcile") or (boundary == "after_attempt_commit" and kind == "send_attempt") or (boundary == "after_fill_commit" and kind == "report" and payload["input"]["kind"] == "fill"):
            # No subsequent application code gets to run.
            os._exit(73)
    # Install the fault at the journal boundary, not by fabricating recovered state.
    store.commit = commit
    # Persist the strategy's intent before planning any orders.
    manager.set_desired(desired())
    # Commit the entire approved order batch before dispatch.
    actions = manager.reconcile(NOW, DAY, {"OGDC": 10000})
    # Model a send that may have reached the counterparty before the process dies.
    if boundary == "during_transport":
        # The durable attempted marker must precede this callback.
        manager.dispatch(actions[0], lambda action: os._exit(73))
    # Remaining cases reach a normal local handoff boundary.
    for action in actions:
        # No real wire traffic occurs in this test.
        manager.dispatch(action, lambda action: None)
    # Acknowledgements establish actual resting orders before fill crash cases.
    for action in actions:
        # Use stable distinct event identities for execution deduplication.
        manager.apply_report("ack:" + action.cl_ord_id, "ack", {"cl_ord_id": action.cl_ord_id, "exchange_order_id": "X" + action.cl_ord_id})
    # Crash either before or after this fill is atomically committed.
    manager.apply_report("execution:day1:1", "fill", fill_payload(actions[0].cl_ord_id))
    # Every named scenario must terminate at its intended boundary.
    raise RuntimeError("fault boundary was not reached")


# Run one isolated crash case and inspect its recovered account invariants.
def check_case(directory, boundary):
    # Use a separate ledger per case; never reuse another case's state.
    directory.mkdir()
    # Keep the journal outside the source repository.
    database = directory / "ledger.sqlite"
    # Run the same script with the same interpreter in worker mode.
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", boundary, "--database", str(database)], capture_output=True, text=True, timeout=30)
    # Failure to reach the explicit crash marker invalidates the case.
    if result.returncode != 73:
        # Surface import errors and unexpected exceptions instead of reporting a pass.
        raise RuntimeError(f"worker failed: {result.returncode}: {result.stdout} {result.stderr}")
    # Reopen the database through the same production recovery code.
    manager, store, _ = build(database)
    # Always release the account lock after evaluating the case.
    try:
        # A process restart must never restore account authorization.
        assert not manager.reconciled
        # A pre-commit batch cannot have reached the transport.
        expected_orders = 0 if boundary == "before_batch_commit" else 2
        # All other crashes retain both sides, including the unsent batch member.
        assert len(manager.oms._orders) == expected_orders
        # A committed fill is the only scenario that may change recovered inventory.
        expected_position = 7 if boundary == "after_fill_commit" else 0
        # Inventory and cash must advance together, never one without the other.
        assert manager.position("OGDC") == expected_position
        # Exact integer accounting includes the explicit three-paisa fee.
        assert manager.cash_minor == (-69303 if expected_position else 0)
        # Buying exposure shrinks only for a durably committed fill.
        assert manager.reserved_quantity("OGDC", Side.BUY) == (100 - expected_position if expected_orders else 0)
        # Restarting always withdraws the previous desired quotes.
        assert all(item.bid is None and item.ask is None for item in manager.oms._desired.values())
        # Prepared actions from the old epoch are held rather than automatically retried.
        assert all(entry["status"] != "prepared" for entry in manager.outbox.values())
        # A committed fill must be deduplicated after restart as well.
        if expected_position:
            # Look up the stable original buy ID from restored order state.
            buy = next(o for o in manager.oms._orders.values() if o.side is Side.BUY)
            # Redelivery must not change cash, fees or position a second time.
            assert manager.apply_report("execution:day1:1", "fill", fill_payload(buy.cl_ord_id)) is False
            # Verify the unchanged cash balance after duplicate processing.
            assert manager.cash_minor == -69303 and manager.position("OGDC") == 7
        # Save a compact independent record of this case's recovered invariants.
        return {"boundary": boundary, "passed": True, "status": manager.status()}
    # Make each case independently reopenable for later inspection.
    finally:
        # Close without deleting the evidence database.
        store.close()


# Provide one self-contained command for the user's larger offline run.
def main():
    # Assertions are verification checks and must not be disabled by Python optimization.
    if sys.flags.optimize:
        # Refuse a run that could print success without checking invariants.
        raise RuntimeError("run without -O or PYTHONOPTIMIZE")
    # Both parent runner and isolated child use this parser.
    parser = argparse.ArgumentParser(description=__doc__)
    # Select one internal crash boundary only when launched as a child.
    parser.add_argument("--worker", choices=CASES)
    # The parent gives the child its explicit evidence database path.
    parser.add_argument("--database", type=Path)
    # Every independent run must have a new output directory.
    parser.add_argument("--output-dir", type=Path)
    # Repeating the matrix tests process recovery, not backtest profitability.
    parser.add_argument("--iterations", type=int, default=1)
    # Parse exact operator parameters.
    args = parser.parse_args()
    # Worker mode never touches any network or other run's database.
    if args.worker:
        # Missing paths are an invocation error, not permission to choose a default ledger.
        if args.database is None:
            # Stop before creating any files.
            parser.error("worker requires --database")
        # Deliberately exits with the crash marker at the chosen boundary.
        worker(args.database, args.worker)
    # Normal mode requires an external evidence destination and positive repetition count.
    if args.output_dir is None or args.iterations < 1:
        # Avoid accidentally writing output beside the source code.
        parser.error("provide --output-dir and positive --iterations")
    # Resolve the output path before checking that it is outside the Git repository.
    output = args.output_dir.resolve()
    # The installed project repository is the ancestor containing Production.
    production = Path(__file__).resolve().parents[1]
    # Walk up to the actual Git root instead of assuming a checkout nesting depth.
    repository = next((parent for parent in (production, *production.parents) if (parent / ".git").exists()), production)
    # Prohibit output beneath the repository even when the working directory differs.
    if output.is_relative_to(repository):
        # Explain the separation required by the user.
        parser.error("output directory must be outside the source repository")
    # Refuse to overwrite evidence from any earlier invocation.
    output.mkdir(parents=True, exist_ok=False)
    # Retain both failures and successful cases in the final summary.
    results = []
    # Exercise each transactional boundary in every requested iteration.
    for iteration in range(args.iterations):
        # Sequential cases keep ownership and result interpretation straightforward.
        for boundary in CASES:
            # Derive a unique evidence directory per process crash.
            destination = output / f"{iteration:04d}_{boundary}"
            # A failed case must be visible without hiding the rest of the matrix.
            try:
                # Run and independently recover one worker's state.
                result = check_case(destination, boundary)
            # Record controlled test failures while continuing the user's matrix.
            except Exception as error:
                # Preserve the failure and its case rather than printing a false pass.
                result = {"boundary": boundary, "passed": False, "error": str(error)}
            # Store this case in the aggregate evidence.
            results.append(result)
            # Emit progress suitable for a PyCharm terminal.
            print(f"{iteration + 1}/{args.iterations} {boundary}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    # A single failed boundary fails the entire recovery check.
    summary = {"passed": all(item["passed"] for item in results), "planned": len(CASES) * args.iterations, "completed": len(results), "failed": sum(not item["passed"] for item in results), "live_approved": False, "identity": identity(), "cases": results}
    # Write JSON evidence beside the per-case durable ledgers.
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Keep the final console line compact and easy to share.
    print(json.dumps({key: value for key, value in summary.items() if key not in ("cases", "identity")}), flush=True)
    # A failed matrix must propagate to shell/PyCharm status.
    return 0 if summary["passed"] else 1


# Importing the runner never launches a test or creates a database.
if __name__ == "__main__":
    # Propagate the test outcome to the terminal.
    raise SystemExit(main())
