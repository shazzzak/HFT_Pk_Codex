"""Focused account-wide boundary, funding, loss and durable integration tests."""
# Change one immutable test limit at a time.
from dataclasses import replace
# Exercise fail-closed errors and deterministic scenario coverage.
import pytest
# Validate integer settings and process-local valuation data.
from core.account_risk import AccountLimits, AccountMark, AccountRiskManager, RiskUnavailable
# Construct actual order-side and withdrawal intentions.
from core.model import Side, DesiredQuotes
# Reuse the delivered runner's actual production assembly and numeric fixtures.
from sim.account_risk_check import build, start, marks, quote, plan, acknowledge, fill, scenario, LIMITS, SCENARIOS, DAY


# Own one temporary account per focused test without repository-side writes.
@pytest.fixture
# Give each test a fresh durable, funded and explicitly marked account.
def account(tmp_path):
    # Build the actual gateway/OMS/durable account combination.
    manager, store, switch, clock = build(tmp_path / "ledger.sqlite")
    # Establish the synthetic account reconciliation barrier and funded day.
    start(manager, clock)
    # Expose lifecycle state and controllable time to the test body.
    yield manager, store, switch, clock
    # Restart tests may already have closed the original owner.
    if not store.lock.closed:
        # Release the local single-writer lock.
        store.close()


# Each delivered matrix scenario must also pass as an independently reported test.
@pytest.mark.parametrize("name", SCENARIOS)
# Run one small account scenario per named financial invariant.
def test_delivered_scenario(tmp_path, name):
    # The returned monitoring result never claims approval to trade live.
    assert scenario(tmp_path, name)["live_approved"] is False


# Booleans, fractional amounts and invalid ceilings cannot become financial limits.
@pytest.mark.parametrize("changes", [{"max_gross_minor": True}, {"max_long_minor": 0}, {"max_short_minor": -1}, {"fee_reserve_bps": 0.5}, {"max_mark_age_ms": 0}, {"max_daily_loss_minor": -1}])
# Validate each independent configuration field through the immutable dataclass.
def test_invalid_limits(changes):
    # Configuration errors must be rejected before opening a trading account.
    with pytest.raises(ValueError):
        # Copy all valid fields and alter only the named invalid one.
        replace(LIMITS, **changes)


# External funding can bind more tightly than the house's maximum cash ceiling.
def test_authorized_funding_below_house_cap(tmp_path):
    # Keep house limits wide while supplying a smaller authorized cash envelope.
    manager, store, _, clock = build(tmp_path / "ledger.sqlite")
    # Release the store even if the expected risk boundary fails.
    try:
        # The account receives only 99,999 paisa of usable test authorization.
        start(manager, clock, funding=99999)
        # A 100,000-paisa buy fits the house cap but not the actual funded envelope.
        quote(manager)
        # The account control must reject before any order is created.
        assert plan(manager) == [] and not manager.oms._orders
    # Preserve evidence while releasing writer ownership.
    finally:
        # End this independent test account.
        store.close()


# Suspended orders remain potential liabilities for account reservations.
def test_suspended_order_keeps_account_funding(account):
    # Obtain a clean funded account with fresh marks.
    manager, _, _, _ = account
    # Request and acknowledge a real OMS bid.
    quote(manager)
    # Keep its client ID for the normalized suspension report.
    action = plan(manager)[0]
    # Establish exchange acceptance before suspension.
    acknowledge(manager, [action])
    # The venue may hold the order inactive without cancelling its liability.
    manager.apply_report("suspend:1", "suspended", {"cl_ord_id": action.cl_ord_id})
    # Suspended size remains fully charged to aggregate risk and cash funding.
    assert manager.measure()["cash_commitment_minor"] == 100000
    # It must not be mistaken for a terminal account state.
    assert manager.status()["unresolved_orders"] == 1


# Cancel rejection preserves both shares and cash previously reserved.
def test_cancel_rejection_keeps_reservation(account):
    # Use the same real OMS and durable account as normal planning.
    manager, _, _, _ = account
    # Establish one live bid.
    quote(manager)
    # Persist and acknowledge it before withdrawal.
    acknowledge(manager, plan(manager))
    # Ordinary flat desire plans the cancel request.
    manager.set_desired(DesiredQuotes.flat("OGDC"))
    # Keep the cancel's own alias for its rejection report.
    cancel = plan(manager)[0]
    # Rejection means the original order can still execute.
    manager.apply_report("cancel-reject:1", "cancel_rejected", {"cl_ord_id": cancel.cl_ord_id, "reason": "too late"})
    # Neither local cancellation nor rejection releases the financial liability.
    assert manager.measure()["cash_commitment_minor"] == 100000


# A price-only amendment can exceed funding even without increasing quantity.
def test_price_only_replacement_counts_old_and_new(tmp_path):
    # Original 100,000 plus proposed 120,000 must exceed this 210,000 envelope.
    limits = replace(LIMITS, max_cash_commitment_minor=210000)
    # Initialize this specific immutable risk contract.
    manager, store, _, clock = build(tmp_path / "ledger.sqlite", limits)
    # Ensure no failed assertion keeps the account locked.
    try:
        # Establish its synthetic day and funding.
        start(manager, clock)
        # Submit one hundred shares at 1,000 paisa.
        quote(manager)
        # Acknowledge before requesting a price change.
        acknowledge(manager, plan(manager))
        # Keep quantity unchanged while increasing the proposed limit.
        quote(manager, price=1200)
        # The old generation can still fill before the new one becomes effective.
        assert plan(manager) == []
        # A rejected proposal must not leave an unapproved reservation behind.
        assert manager.measure()["cash_commitment_minor"] == 100000
        # Proposed-price tracking only updates after the whole gateway approves.
        assert manager.account["replacement_prices"] == {}
    # Release the account after checking the rejected amendment.
    finally:
        # Preserve the failure/success journal on disk.
        store.close()


# No fresh marks are implicitly restored from a previous process's timestamps.
def test_restart_needs_fresh_marks_and_keeps_spend(account):
    # Create a fill whose budget effect must survive recovery.
    manager, store, _, clock = account
    # Establish one live bid before the partial execution.
    quote(manager)
    # Preserve its stable action identity.
    action = plan(manager)[0]
    # Accept it through the durable transport/report boundary.
    acknowledge(manager, [action])
    # Execute twenty shares with explicit fees.
    fill(manager, action, 20, 990, 7)
    # Retain the persistent database across the ownership change.
    path = store.path
    # Finish the initial process epoch.
    store.close()
    # Restore order and account limits into a new instance.
    recovered, reopened, _, clock = build(path, create=False, clock=clock)
    # Always release the recovered owner.
    try:
        # Persisted prior-process marks cannot authorize current account valuation.
        with pytest.raises(RiskUnavailable):
            # Remaining order and position exposure still require fresh prices.
            recovered.measure()
        # Publish fresh observations in the current clock domain.
        marks(recovered, clock)
        # Reserved and already spent cash must be identical across restart.
        assert recovered.measure()["cash_commitment_minor"] == 99807
        # Redelivery of the same execution cannot consume funding again.
        assert fill(recovered, action, 20, 990, 7) is False
        # A funding counter is not inferred from net cash or opposing fills.
        assert recovered.account["buy_spent_minor"] == 19800 and recovered.account["positive_fees_minor"] == 7
    # Leave the evidence database available for inspection.
    finally:
        # Release the second ownership epoch.
        reopened.close()


# Approval cannot be used after prices become stale before transport handoff.
def test_stale_before_dispatch_never_calls_transport(account):
    # Generate approved actions under fresh market observations.
    manager, store, switch, clock = account
    # Request one ordinary bid.
    quote(manager)
    # Commit its risk-approved prepared state before time advances.
    action = plan(manager)[0]
    # Move local time beyond the mark freshness bound.
    clock[0] += 1001
    # Record any simulated transport side effect.
    sends = []
    # A fresh dispatch check must reject rather than sending on stale approval.
    with pytest.raises(RuntimeError):
        # The simulated transport must remain untouched.
        manager.dispatch(action, sends.append)
    # Retain the uncertain prepared liability while latching a risk halt.
    assert sends == [] and switch.tripped
    # No attempted-send marker is allowed when transport was never entered.
    assert store.load()["outbox"][action.cl_ord_id]["status"] == "prepared"


# Unexpected actual fees must be booked, even if they exceed the pretrade allowance.
def test_actual_fees_exceed_funding_halt_after_recording(tmp_path):
    # A zero-fee test reserve makes the unexpected one-paisa fee cross the boundary.
    limits = replace(LIMITS, max_cash_commitment_minor=100000)
    # Start with exactly enough authorized cash for the order notional.
    manager, store, switch, clock = build(tmp_path / "ledger.sqlite", limits)
    # Financial facts must survive even when they reveal a budget breach.
    try:
        # Fund the account at the exact synthetic ceiling.
        start(manager, clock)
        # Submit a buy that fits before the unanticipated fee arrives.
        quote(manager)
        # Preserve its real order identity.
        action = plan(manager)[0]
        # Accept the order before reporting execution.
        acknowledge(manager, [action])
        # The fill is real and cannot be rejected merely because it breaches risk.
        fill(manager, action, 100, 1000, 1)
        # Book both inventory and the entire paid amount before halting future activity.
        assert manager.position("OGDC") == 100 and manager.cash_minor == -100001
        # The execution and funding breach share one durable commit.
        assert store.load()["account_risk"]["positive_fees_minor"] == 1 and store.load()["killed"]
        # Account risk must halt rather than silently increasing its authorization.
        assert switch.tripped
    # Release the account after checking its committed breach state.
    finally:
        # Preserve the durable fill and risk incident.
        store.close()


# Reaching the daily-loss threshold exactly is already a stop, including fees.
def test_exact_daily_loss_boundary(tmp_path):
    # One hundred shares lose 1,000 at bid, plus ten paisa of fees.
    limits = replace(LIMITS, max_daily_loss_minor=1010)
    # Keep every other financial limit nonbinding.
    manager, store, switch, clock = build(tmp_path / "ledger.sqlite", limits)
    # Release the writer after the boundary assertion.
    try:
        # Create the explicit zero-equity daily baseline.
        start(manager, clock)
        # Enter the hundred-share position through actual order processing.
        quote(manager)
        # Keep the risk-approved action for its fill report.
        action = plan(manager)[0]
        # Record transport handoff and exchange acknowledgement.
        acknowledge(manager, [action])
        # Mark-to-bid P&L is exactly -1,010 paisa after this fill.
        fill(manager, action, 100, 1000, 10)
        # Equality must trip rather than waiting for a greater loss.
        assert manager.measure()["daily_pnl_minor"] == -1010 and manager.account["loss_latched"] and switch.tripped
        # Resetting only the generic kill cannot clear the financial day's stop.
        switch.reset("test-reset")
        # Independent timer evaluation must reassert the persistent daily stop.
        manager.poll_account()
        # The account's daily latch is the authority, not the resettable display state.
        assert switch.tripped
    # Preserve the committed halt while releasing ownership.
    finally:
        # Close the SQLite connection and lock.
        store.close()


# Short inventory is marked at the ask needed to cover it, not a favorable bid.
def test_short_pnl_marks_at_ask(tmp_path):
    # A one-thousand-paisa daily loss is the exact cover-price boundary.
    limits = replace(LIMITS, max_daily_loss_minor=1000)
    # Create an explicitly short-authorized synthetic account.
    manager, store, switch, clock = build(tmp_path / "ledger.sqlite", limits)
    # Always close the database at the end of the case.
    try:
        # Short permission is separate from account-wide notional limits.
        start(manager, clock, short_caps={"OGDC": 10})
        # Sell ten shares at a thousand paisa each.
        quote(manager, side=Side.SELL, quantity=10)
        # Preserve the real OMS order ID.
        action = plan(manager)[0]
        # Acknowledge before reporting the short execution.
        acknowledge(manager, [action])
        # The proceeds do not replenish the cash commitment envelope.
        fill(manager, action, 10, 1000)
        # A wider ask raises actual short-cover valuation even if bid stays unchanged.
        marks(manager, clock, bid=1000, ask=1100)
        # Ten shares times one hundred paisa is a one-thousand-paisa loss.
        assert manager.measure()["daily_pnl_minor"] == -1000 and switch.tripped
    # Finish the synthetic account ownership epoch.
    finally:
        # Keep all short-risk evidence external to source control.
        store.close()


# Reduced external authorization must withdraw already resting liabilities.
def test_funding_tightening_cancels_existing_order(account):
    # Begin from an account with ample initial authorization.
    manager, _, switch, _ = account
    # Establish a resting bid consuming 100,000 paisa.
    quote(manager)
    # Pass through both durable and account-aware dispatch.
    acknowledge(manager, plan(manager))
    # Reduce the day's total envelope below outstanding commitment.
    manager.tighten_authorizations("tester", funding_minor=99999)
    # The account must stop adding exposure immediately.
    assert switch.tripped
    # The standard OMS diff remains able to cancel the original order.
    assert all(action.is_cancel for action in plan(manager))


# Revoked short availability must not leave authorized-looking sell orders resting.
def test_short_authorization_revocation(tmp_path):
    # Start with an explicitly granted small short allocation.
    manager, store, switch, clock = build(tmp_path / "ledger.sqlite")
    # Keep the scenario isolated from other account fixtures.
    try:
        # This grant is synthetic; a live adapter must supply genuine availability.
        start(manager, clock, short_caps={"OGDC": 10})
        # Place a sell entirely within that allocation.
        quote(manager, side=Side.SELL, quantity=10)
        # Confirm that it is actually resting before the revocation.
        acknowledge(manager, plan(manager))
        # A complete empty map removes all previously granted short quantities.
        manager.tighten_authorizations("tester", short_caps={})
        # Existing exposure outside the new authorization must halt quoting.
        assert switch.tripped and manager.account["short_caps"] == {}
        # Cancellation stays on the normal OMS path despite the risk breach.
        cancels = plan(manager)
        # Require a real cancellation rather than vacuous all([]) success.
        assert len(cancels) == 1 and cancels[0].is_cancel
    # Release the current account after its revocation check.
    finally:
        # Preserve the original grant and later restriction in the journal.
        store.close()


# Monitoring consumers must not be able to mutate live financial permissions.
def test_status_is_detached(account):
    # Request ordinary machine-readable status.
    manager, _, _, _ = account
    # Returned nested structures must not alias authoritative account state.
    status = manager.status()
    # A dashboard changes its local copy of the availability map.
    status["account_risk"]["short_caps"]["OGDC"] = 99999
    # No extra short permission may appear inside the risk engine.
    assert manager.account["short_caps"] == {}


# Invalid market inputs must cause durable withdrawal instead of retaining old permission.
def test_invalid_mark_halts_durably(account):
    # Start from valid current observations and an acknowledged order.
    manager, store, switch, clock = account
    # Request an ordinary bid under the initial valid marks.
    quote(manager)
    # Confirm its exchange acceptance.
    acknowledge(manager, plan(manager))
    # An observation claiming receipt in the future is not safe to use.
    with pytest.raises(ValueError):
        # Do not let future timestamps disable stale-feed protection.
        manager.update_marks({"OGDC": AccountMark(990, 1000, clock[0] + 1)})
    # The failed update must persist a halt before returning the error.
    assert switch.tripped and store.load()["killed"]
