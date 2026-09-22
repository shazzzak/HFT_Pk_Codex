"""Targeted local recovery, deduplication and controller integration tests."""
# Copy domain requests while changing one field for tamper tests.
from dataclasses import replace
# Inspect journal corruption without bypassing the runtime in normal cases.
import sqlite3
# Check fail-closed outcomes and parameterized faults.
import pytest
# Use the actual controller around the durable OMS facade.
from core.control_loop import ControlLoop
# Preserve action and side identity in assertions.
from core.model import DesiredQuotes, Side
# Verify database ownership and immutable identity checks.
from core.recovery_store import RecoveryStore
# Use the same deterministic setup as the delivered command-line runner.
from sim.recovery_check import build, authorize, desired, fill_payload, DAY, NOW, CASES, check_case, identity


# Isolate each test's durable database and release its account lock.
@pytest.fixture
# Build a clean offline account for each invariant.
def runtime(tmp_path):
    # Initialize explicitly rather than treating a missing file as recovery.
    manager, store, switch = build(tmp_path / "ledger.sqlite", create=True)
    # Return the account components to the test body.
    yield manager, store, switch
    # Tests that deliberately restart replace this connection themselves.
    if not store.lock.closed:
        # Release the original lock when still owned.
        store.close()


# Persist and dispatch both sides with stable acknowledgement identities.
def resting(manager):
    # Empty-account reconciliation precedes any new order.
    assert authorize(manager)
    # Ask the real OMS for ordinary two-sided quotes.
    manager.set_desired(desired())
    # Persist the complete batch before any send callback.
    actions = manager.reconcile(NOW, DAY, {"OGDC": 10000})
    # Deliver to a simulated transport and acknowledge each side.
    for action in actions:
        # Local transport completion is distinct from exchange acceptance.
        manager.dispatch(action, lambda _: None)
        # Apply the acknowledgement once through the normalized report boundary.
        manager.apply_report("ack:" + action.cl_ord_id, "ack", {"cl_ord_id": action.cl_ord_id, "exchange_order_id": "X" + action.cl_ord_id})
    # Preserve the original action IDs for later lifecycle tests.
    return actions


# Missing recovery files must never turn into an empty account silently.
def test_missing_database_requires_explicit_initialization(tmp_path):
    # An absent state path is an operator error during restart.
    with pytest.raises(FileNotFoundError):
        # The default mode is recovery, not creation.
        build(tmp_path / "missing.sqlite")


# A single local account cannot have two simultaneous process owners.
def test_exclusive_writer(runtime):
    # Keep the first instance's lock open throughout the attempted second open.
    _, store, _ = runtime
    # OS lock enforcement also prevents two owners inside one process.
    with pytest.raises(BlockingIOError):
        # The second owner must fail before reading or mutating the ledger.
        RecoveryStore(store.path, identity())


# A different account/configuration may not reuse committed trading history.
def test_identity_mismatch(runtime):
    # Release ownership so identity validation, not locking, is exercised.
    _, store, _ = runtime
    # Preserve the database path before closing the first owner.
    path = store.path
    # Finish the first ownership epoch.
    store.close()
    # Altering even one configured risk limit requires reviewed migration.
    with pytest.raises(RuntimeError, match="identity"):
        # Never accept the old state under a changed live configuration silently.
        RecoveryStore(path, {**identity(), "position_limit": 2000})


# Both sides are committed before the first transport side effect.
def test_batch_is_durable_before_transport(runtime):
    # Use the actual risk-authorized diff implementation.
    manager, store, _ = runtime
    # Establish the synthetic empty-account recovery barrier.
    assert authorize(manager)
    # Request two independent potential exposures.
    manager.set_desired(desired())
    # Receive only actions whose reservations already reached SQLite.
    actions = manager.reconcile(NOW, DAY, {"OGDC": 10000})
    # Read the committed checkpoint before calling any transport.
    state = store.load()
    # Both liabilities are preserved even if only the first send later happens.
    assert len(state["orders"]) == 2 and len(state["outbox"]) == 2
    # Initial durable status records intent, not exchange acknowledgement.
    assert all(entry["status"] == "prepared" for entry in state["outbox"].values())
    # Use an explicit callback to check the actual pre-send durability contract.
    def transport(action):
        # Disk state already says this delivery might have happened.
        assert store.load()["outbox"][action.cl_ord_id]["status"] == "attempted"
    # Only now hand off the first action.
    manager.dispatch(actions[0], transport)
    # Local handoff still does not release the reservation.
    assert manager.reserved_quantity("OGDC", Side.BUY) == 100


# Duplicate dispatch must never turn one intent into two exchange orders.
def test_duplicate_and_modified_dispatch_are_rejected(runtime):
    # Create the durable order batch without acknowledging it.
    manager, _, _ = runtime
    # Authorize only after explicit startup reconciliation.
    assert authorize(manager)
    # Preserve the original domain action in the outbox.
    manager.set_desired(desired())
    # Produce the full approved batch.
    actions = manager.reconcile(NOW, DAY, {"OGDC": 10000})
    # A changed size cannot reuse a committed order identifier.
    with pytest.raises(RuntimeError):
        # Reject tampering before any transport callback.
        manager.dispatch(replace(actions[0], quantity=101), lambda _: None)
    # The original unaltered action may still be handed off once.
    manager.dispatch(actions[0], lambda _: None)
    # A second handoff is not a reliable resend mechanism.
    with pytest.raises(RuntimeError):
        # Duplicate attempts require external recovery instead.
        manager.dispatch(actions[0], lambda _: None)


# Fill deduplication survives ownership epochs and preserves fees exactly once.
def test_fill_duplicate_after_restart(runtime):
    # Establish acknowledged orders in a durable account.
    manager, store, _ = runtime
    # Both orders use the same real OMS as ordinary production controls.
    actions = resting(manager)
    # Apply a partial execution with a small explicit fee.
    manager.apply_report("exec:1", "fill", fill_payload(actions[0].cl_ord_id))
    # Check exact integer cash and inventory before restart.
    assert manager.position("OGDC") == 7 and manager.cash_minor == -69303 and manager.fees_minor == 3
    # Preserve the existing account location.
    path = store.path
    # End the first process ownership epoch.
    store.close()
    # Construct a new OMS instance from committed state.
    recovered, reopened, _ = build(path)
    # Always release the second ownership epoch.
    try:
        # A replayed execution must be recognized from the persistent deduplication map.
        assert recovered.apply_report("exec:1", "fill", fill_payload(actions[0].cl_ord_id)) is False
        # All accounting remains exactly as before the redelivery.
        assert recovered.position("OGDC") == 7 and recovered.cash_minor == -69303 and recovered.fees_minor == 3
        # A restart never inherits account authorization.
        assert not recovered.reconciled
    # Cleanup is explicit so later tests can inspect these databases.
    finally:
        # Release the reopened account lock.
        reopened.close()


# Contradictory duplicate executions require persistent operator investigation.
def test_conflicting_duplicate_survives_restart(runtime):
    # Begin from two acknowledged quotes.
    manager, store, _ = runtime
    # Keep the buy ID for both conflicting reports.
    actions = resting(manager)
    # Commit the first execution normally.
    payload = fill_payload(actions[0].cl_ord_id)
    # The first stable execution identity affects cash once.
    manager.apply_report("exec:1", "fill", payload)
    # A changed execution size cannot reuse that identity.
    with pytest.raises(RuntimeError, match="conflicting"):
        # This must poison the account without changing its good ledger.
        manager.apply_report("exec:1", "fill", {**payload, "quantity": 8})
    # Preserve the path before releasing the poisoned instance.
    path = store.path
    # Simulate process shutdown after the incident.
    store.close()
    # Reopening cannot clear the recorded contradiction.
    recovered, reopened, _ = build(path)
    # Keep the lock cleanup independent of assertions.
    try:
        # Good cash and inventory remain available for investigation.
        assert recovered.poisoned and recovered.position("OGDC") == 7
        # New order processing remains forbidden after restart.
        with pytest.raises(RuntimeError, match="poisoned"):
            # Operator arming cannot bypass this unresolved input incident.
            recovered.set_desired(desired())
    # Release the incident store without editing its history.
    finally:
        # Preserve the evidence for later inspection.
        reopened.close()


# Invalid reports must never enter cash or position accounting.
@pytest.mark.parametrize("change", [{"quantity": 0}, {"quantity": 101}, {"quantity": 1.5}, {"symbol": "OTHER"}, {"side": "SELL"}, {"price_minor": 10000}, {"fee_minor": 0.5}])
# Exercise malformed units, overfills, side mismatch and limit-price violations.
def test_invalid_fill_fails_closed(runtime, change):
    # Use acknowledged orders so rejection concerns the report itself.
    manager, store, _ = runtime
    # Locate the actual buy order.
    actions = resting(manager)
    # Any invalid execution is a persistent recovery incident.
    with pytest.raises(Exception):
        # The runtime validates before accepting the input as processed.
        manager.apply_report("bad:1", "fill", {**fill_payload(actions[0].cl_ord_id), **change})
    # The durable ledger remains at its last good state.
    assert store.load()["positions"] == {} and store.load()["cash_minor"] == 0
    # An incident must be visible after a restart as well.
    assert store.load()["incident"] is not None and manager.poisoned


# A SQLite write error must prevent the send path from receiving a new batch.
def test_failed_batch_commit_poisoned(runtime, monkeypatch):
    # Build a clean account and establish authorization.
    manager, store, _ = runtime
    # Authorize the initial synthetic empty account.
    assert authorize(manager)
    # Commit desired intent before the injected disk failure.
    manager.set_desired(desired())
    # A failing journal must look like a real write failure to the runtime.
    def fail(*args):
        # No transaction successfully commits at this boundary.
        raise OSError("injected disk full")
    # Replace only the store's write operation for this test.
    monkeypatch.setattr(store, "commit", fail)
    # Planning may reserve in memory, but it must not return dispatchable actions.
    with pytest.raises(OSError):
        # The controller therefore cannot call the transport with these actions.
        manager.reconcile(NOW, DAY, {"OGDC": 10000})
    # Continuing with this partially mutated instance is prohibited.
    assert manager.poisoned
    # The last durable checkpoint still has no outbound orders.
    assert store.load()["orders"] == []


# A timeout after transport entry retains both attempted and unsent liabilities.
def test_transport_exception_never_retries(runtime):
    # Open the account and persist two orders.
    manager, store, _ = runtime
    # Startup requires explicit empty-account reconciliation.
    assert authorize(manager)
    # Record intended bid and ask.
    manager.set_desired(desired())
    # Make both potential exposures durable before dispatch.
    actions = manager.reconcile(NOW, DAY, {"OGDC": 10000})
    # Model an adapter that cannot tell whether a socket write succeeded.
    def uncertain(action):
        # An exception is not proof that the exchange received nothing.
        raise TimeoutError("unknown delivery")
    # Surface uncertainty to the host rather than silently retrying.
    with pytest.raises(TimeoutError):
        # The attempted marker precedes this callback.
        manager.dispatch(actions[0], uncertain)
    # The unsent second action remains conservatively reserved too.
    assert len(store.load()["orders"]) == 2 and manager.poisoned
    # Attempt status survives independently of transport completion.
    assert store.load()["outbox"][actions[0].cl_ord_id]["status"] == "attempted"


# A pending cancellation retains leaves across a restart and racing execution.
def test_cancel_fill_race_restart(runtime):
    # Start with two acknowledged quotes.
    manager, store, _ = runtime
    # Obtain stable original order identifiers.
    actions = resting(manager)
    # Ordinary desired-flat logic generates the cancellation batch.
    manager.set_desired(DesiredQuotes.flat("OGDC"))
    # Both cancels are committed while original exposure remains.
    cancels = manager.reconcile(NOW + 1, DAY, {"OGDC": 10000})
    # Apply a partial fill before cancellation confirms.
    manager.apply_report("race:fill", "fill", fill_payload(actions[0].cl_ord_id))
    # End this ownership epoch with cancellations outstanding.
    path = store.path
    # Close without manufacturing any cancel acknowledgement.
    store.close()
    # Reconstruct aliases, pending flags and reduced leaves.
    recovered, reopened, _ = build(path)
    # Test the race through the recovered facade.
    try:
        # Remaining buy liability survives the restart.
        assert recovered.reserved_quantity("OGDC", Side.BUY) == 93
        # Cancel IDs still resolve to original orders after recovery.
        for action in cancels:
            # Only authoritative cancellation releases the remaining leaves.
            recovered.apply_report("cancel:" + action.cl_ord_id, "cancelled", {"cl_ord_id": action.cl_ord_id})
        # Cancelled quotes do not liquidate the seven filled shares.
        assert recovered.reserved_quantity("OGDC", Side.BUY) == 0 and recovered.position("OGDC") == 7
        # A mismatched external account cannot authorize quoting.
        assert not authorize(recovered)
        # Matching synchronized positions and cash complete the drained-order barrier.
        assert recovered.reconcile_external({"complete": True, "barrier": "after-cancels", "open_orders": [], "positions": {"OGDC": 7}, "cash_minor": -69303}, "tester")
    # Leave the evidence ledger intact but unlocked.
    finally:
        # Close the recovered owner.
        reopened.close()


# Pending amendment exposure includes both generations after restart.
def test_replacement_reservation_restart(runtime):
    # Begin with resting one-hundred-share quotes.
    manager, store, _ = runtime
    # Acknowledge both original sides.
    resting(manager)
    # Ask exact-size OMS policy to increase both sides.
    manager.set_desired(desired(150))
    # Persist the replacement batch with original and proposed liabilities.
    replacements = manager.reconcile(NOW + 1, DAY, {"OGDC": 10000})
    # Old remaining and new proposed quantities both count during uncertainty.
    assert manager.reserved_quantity("OGDC", Side.BUY) == 250
    # Keep the same database for recovery.
    path = store.path
    # End the original owner before any amendment confirmation.
    store.close()
    # Construct fresh in-memory order objects from the checkpoint.
    recovered, reopened, _ = build(path)
    # Validate replacement aliases and reservation lifetime.
    try:
        # Conservative amendment exposure survives the crash boundary.
        assert recovered.reserved_quantity("OGDC", Side.BUY) == 250
        # Apply an authoritative remaining-size amendment acknowledgement.
        recovered.apply_report("replace:1", "replaced", {"cl_ord_id": replacements[0].cl_ord_id, "price_minor": 9900, "quantity": 150})
        # Only confirmation releases the superseded generation's reservation.
        assert recovered.reserved_quantity("OGDC", Side.BUY) == 150
        # ID allocation must continue beyond the previous session's durable counter.
        assert recovered.oms._seq >= 4
    # Release the reopened account on success or failure.
    finally:
        # Keep the journal available for forensic inspection.
        reopened.close()


# Controller timers must use the durable normal cancellation path.
def test_controller_feed_failure_integration(runtime):
    # Use the controller against the facade rather than a second OMS implementation.
    manager, _, switch = runtime
    # Establish a synthetic synchronized startup account.
    assert authorize(manager)
    # Track feed health independently of incoming trading messages.
    feed, sent = [True], []
    # Inject only a simulated transport; broker readiness comes from durable recovery.
    controller = ControlLoop(manager, switch, ("OGDC",), lambda action: manager.dispatch(action, sent.append), lambda: True, lambda: manager.reconciled, lambda symbol: feed[0], lambda now: None, manager.audit)
    # Explicit named arming is required even after account reconciliation.
    controller.arm("tester", ("OGDC",))
    # Deliver strategy intent through the normal controller boundary.
    assert controller.quote(desired())
    # The timer commits and dispatches the OMS batch.
    quotes = controller.cycle(1, NOW, DAY, {"OGDC": 10000})
    # Acknowledge both actual outbound orders through durable deduplication.
    for action in quotes:
        # Acknowledgements do not reset a valid same-epoch account barrier.
        manager.apply_report("ack:" + action.cl_ord_id, "ack", {"cl_ord_id": action.cl_ord_id, "exchange_order_id": "X" + action.cl_ord_id})
    # Lose feed health with no new market message.
    feed[0] = False
    # Independent timer processing withdraws both sides.
    cancels = controller.cycle(2, NOW + 1, DAY, {"OGDC": 10000})
    # Cancellation uses the same durable batch/transport path.
    assert len(cancels) == 2 and all(action.is_cancel for action in cancels)
    # Lost data removes arm permission rather than merely pausing the strategy.
    assert not controller.armed


# Logical checkpoint corruption must stop startup before any order is restored.
def test_checkpoint_tamper_detected(runtime):
    # Close the legitimate owner before simulating storage corruption.
    _, store, _ = runtime
    # Preserve the physical database path.
    path = store.path
    # Release SQLite and ownership locking.
    store.close()
    # Modify one checkpoint outside the legitimate transactional protocol.
    with sqlite3.connect(path) as connection:
        # Structural integrity alone would not detect this valid JSON replacement.
        connection.execute("UPDATE current SET state='{}'")
    # The journal's checkpoint digest must catch the mismatch.
    with pytest.raises(RuntimeError, match="checkpoint"):
        # Never restore or silently repair a logically damaged account.
        build(path)


# Exercise each abrupt process boundary once in the targeted test suite.
@pytest.mark.parametrize("boundary", CASES)
# Use real child exit and SQLite reopen rather than mocked recovery state.
def test_abrupt_process_boundaries(tmp_path, boundary):
    # One small six-case matrix is enough for local boundary validation.
    assert check_case(tmp_path / boundary, boundary)["passed"]


# Journal insertion and checkpoint publication must roll back as one transaction.
def test_mid_transaction_failure_rolls_back(runtime):
    # Capture the last fully committed account state and journal sequence.
    manager, store, _ = runtime
    # Load a detached checkpoint before fault injection.
    before, sequence = store.load(), store.sequence
    # Preserve the actual SQLite connection underneath a narrow failure proxy.
    connection = store.db
    # Fail only the checkpoint write after the journal insert has succeeded.
    class FailingConnection:
        # Delegate SQLite operations except the selected failing statement.
        def execute(self, sql, parameters=()):
            # Inject failure inside the actual transaction after its first write.
            if sql.startswith("INSERT OR REPLACE INTO current"):
                # Simulate storage failing on publication of the new checkpoint.
                raise OSError("checkpoint write failed")
            # The real connection executes BEGIN, journal insertion and ROLLBACK.
            return connection.execute(sql, parameters)
        # Report the real transaction state for correct rollback behavior.
        @property
        # Expose SQLite's actual active-transaction flag.
        def in_transaction(self):
            # No simulated transaction bookkeeping is used.
            return connection.in_transaction
    # Swap in the narrow proxy only for one attempted commit.
    store.db = FailingConnection()
    # Always restore the connection so the fixture can close it.
    try:
        # An event insert without a checkpoint must never remain committed.
        with pytest.raises(OSError):
            # Exercise the real store transaction rather than bypassing commit().
            store.commit("injected", {}, before)
    # Keep cleanup independent of the expected exception.
    finally:
        # Restore the real SQLite connection for inspection.
        store.db = connection
    # The failed transaction must preserve the previous checkpoint exactly.
    assert store.load() == before and store.sequence == sequence
    # Its journal insertion must have rolled back too.
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == sequence
    # Full logical chain verification remains valid after rollback.
    store.verify()


# Startup quote permission cannot be inherited from caller-provided truthy strings.
@pytest.mark.parametrize("snapshot", [{"complete": "yes", "barrier": "b"}, {"complete": True, "barrier": ""}])
# Require an explicit complete synchronized recovery barrier.
def test_reconciliation_requires_explicit_barrier(runtime, snapshot):
    # Use a genuinely empty account to isolate metadata validation.
    manager, _, _ = runtime
    # A partial or malformed response cannot authorize the account.
    with pytest.raises(ValueError):
        # Complete order/position data is insufficient without the barrier contract.
        manager.reconcile_external({"positions": {}, "open_orders": [], "cash_minor": 0, **snapshot}, "tester")
    # Failed input must not grant startup readiness.
    assert not manager.reconciled
