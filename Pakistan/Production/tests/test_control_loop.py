"""Offline operator/timer/transport controls around the actual OMS."""
# Assert fail-closed operator and transport behavior.
import pytest
# Exercise the application controller and separate liquidation policy.
from core.control_loop import ControlLoop, LiquidationPolicy
# Use real fill and order-domain types.
from core.model import CancelOrder, Fill, Side
# Exact requoting exposes accidental liquidation top-ups.
from core.oms import QuoteTolerance
# The control loop still clears the actual position-risk gateway.
from core.risk import PositionLimitCheck
# Reuse a deterministic venue/session setup.
from tests.test_oms import build, want, DAY, NOW


# Construct an offline application with visible sends and mutable readiness.
def app():
    # No exchange connection exists in this fixture.
    oms, _, switch = build(tolerance=QuoteTolerance(quantity_policy="exact"), checks=[PositionLimitCheck(200)], use_replace=True)
    # Model independently changing transport, ledger and market-data readiness.
    state = {"transport": True, "ledger": True, "feed": True}
    # Retain outbound messages and operator audit records.
    sent, audit = [], []
    # Supply explicit readiness authorities rather than inferring an empty account.
    controller = ControlLoop(oms, switch, ("OGDC",), sent.append, lambda: state["transport"], lambda: state["ledger"], lambda _: state["feed"], lambda _: None, lambda *a: audit.append(a))
    # Expose every independently observable effect.
    return controller, oms, state, sent, audit


# Use a fixed exchange clock and a monotonically advanced local timer.
def cycle(controller, now=100):
    # References still pass through the gateway on every action.
    return controller.cycle(now, NOW, DAY, {"OGDC": 10000})


# A connected socket alone cannot authorize quoting.
def test_startup_requires_operator_and_broker_reconciliation():
    # Startup is disarmed despite every readiness callback initially being true.
    controller, oms, state, sent, _ = app()
    # Unarmed strategy output is rejected.
    assert not controller.quote(want())
    # No order reaches the transport.
    assert cycle(controller) == []
    # An unreconciled broker account prevents arming.
    state["ledger"] = False
    # A named operator cannot bypass an incomplete startup check.
    with pytest.raises(RuntimeError):
        # Request normal quote permission.
        controller.arm("operator", ("OGDC",))
    # Only verified account state enables the arming check.
    state["ledger"] = True
    # Deliberately authorize this symbol.
    controller.arm("operator", ("OGDC",))
    # Strategy intent remains unchanged.
    assert controller.quote(want())
    # Both sides are still approved through the real OMS and gateway.
    assert len(cycle(controller, 101)) == 2


# Recovery of data alone must not automatically restart market making.
def test_feed_failure_cancels_and_requires_manual_rearm():
    # Build and arm the test application.
    controller, oms, state, sent, _ = app()
    # Explicitly grant quote permission.
    controller.arm("operator", ("OGDC",))
    # Publish strategy intent.
    controller.quote(want())
    # Dispatch the two quotes.
    quotes = cycle(controller)
    # Acknowledge both orders before the feed fails.
    for order in quotes:
        # Keep the OMS and simulated exchange in agreement.
        oms.on_ack(order.cl_ord_id, order.cl_ord_id)
    # Feed validity is lost independently of order-entry connectivity.
    state["feed"] = False
    # Timer-driven controls withdraw the existing quotes.
    cancels = cycle(controller, 101)
    # The controller uses ordinary CancelOrder messages from OMS reconciliation.
    assert len(cancels) == 2 and all(isinstance(a, CancelOrder) for a in cancels)
    # Recovery alone cannot silently re-arm the strategy.
    state["feed"] = True
    # A stale strategy callback cannot restore quote permission.
    assert not controller.quote(want())
    # Acknowledgements resolve the original exposures.
    for cancel in cancels:
        # Cancel responses release reservations normally.
        oms.on_cancelled(cancel.cl_ord_id)
    # Explicit operator action is needed to resume.
    controller.arm("operator", ("OGDC",))
    # Only now may the strategy resume producing quotes.
    assert controller.quote(want())


# An uncertain transport outcome must retain all reservations and forbid retries.
def test_ambiguous_send_keeps_risk_reserved():
    # Create an armed application with no actual network connection.
    controller, oms, _, sent, _ = app()
    # Grant explicit quote permission.
    controller.arm("operator", ("OGDC",))
    # Ask for two sides.
    controller.quote(want())
    # Model a transport that might have sent before failing.
    def broken_send(action):
        # The local exception does not prove that the exchange received nothing.
        raise OSError("connection lost during enqueue")
    # Replace only the outbound transport for this failure test.
    controller.send = broken_send
    # The failure reaches the host loop rather than being silently ignored.
    with pytest.raises(OSError):
        # Attempt one normal cycle.
        cycle(controller)
    # The shared kill switch is tripped.
    assert controller.kill.tripped and controller.send_failed
    # The potentially delivered buy remains reserved.
    assert oms.reserved_quantity("OGDC", Side.BUY) == 100
    # Later timer calls do not blindly resend the ambiguous actions.
    assert cycle(controller, 101) == []


# Inventory liquidation waits for old quotes and never tops up a partially filled clip.
def test_liquidation_is_separate_and_does_not_top_up():
    # Use exact OMS policy so an incorrect fixed desired size would cause an amendment.
    controller, oms, _, _, _ = app()
    # Authorize initial market-making quotes.
    controller.arm("operator", ("OGDC",))
    # Request the initial quote pair.
    controller.quote(want())
    # Dispatch and acknowledge both orders.
    quotes = cycle(controller)
    # Establish both live orders.
    for order in quotes:
        # Acknowledgements preserve each unique order identity.
        oms.on_ack(order.cl_ord_id, order.cl_ord_id)
    # Fill the complete buy to create real inventory.
    buy = next(q for q in quotes if q.side is Side.BUY)
    # The sell quote remains outstanding.
    oms.on_fill(Fill(buy.cl_ord_id, "OGDC", Side.BUY, buy.price_minor, 100, NOW))
    # An explicit bounded-price liquidation instruction is separate from a stop.
    controller.request_liquidation("operator", "OGDC", LiquidationPolicy(50, 9900))
    # First cancel the old sell quote, with no overlapping unwind clip.
    cancel, = cycle(controller, 101)
    # The drain barrier uses the normal cancellation action.
    assert isinstance(cancel, CancelOrder)
    # Until the cancel is confirmed, no liquidation order is emitted.
    assert cycle(controller, 102) == []
    # Resolve the original quote.
    oms.on_cancelled(cancel.cl_ord_id)
    # Now send the independently authorized fifty-share liquidation clip.
    liquidation, = cycle(controller, 103)
    # Its side and size must reduce actual inventory.
    assert liquidation.side is Side.SELL and liquidation.quantity == 50
    # Acknowledge the new clip.
    oms.on_ack(liquidation.cl_ord_id, liquidation.cl_ord_id)
    # Partially execute twenty shares.
    oms.on_fill(Fill(liquidation.cl_ord_id, "OGDC", Side.SELL, liquidation.price_minor, 20, NOW))
    # Exact policy must not amend the remaining thirty back up to fifty.
    assert cycle(controller, 104) == []
    # The remaining order risk equals the actual unfilled clip.
    assert oms.reserved_quantity("OGDC", Side.SELL) == 30
    # Complete this clip before issuing another.
    oms.on_fill(Fill(liquidation.cl_ord_id, "OGDC", Side.SELL, liquidation.price_minor, 30, NOW))
    # Only the remaining fifty inventory shares may be offered next.
    next_clip, = cycle(controller, 105)
    # No instruction can overshoot zero inventory in this sequential policy.
    assert next_clip.quantity == 50 and oms.position("OGDC") == 50


# A shutdown request does not mean outstanding quotes have disappeared.
def test_shutdown_waits_for_cancel_responses():
    # Create an armed, flat-inventory test account.
    controller, oms, _, _, _ = app()
    # Grant quote permission.
    controller.arm("operator", ("OGDC",))
    # Send two quotes with zero filled inventory.
    controller.quote(want())
    # Dispatch the quotes.
    quotes = cycle(controller)
    # Make them live.
    for order in quotes:
        # The exchange knows these orders.
        oms.on_ack(order.cl_ord_id, order.cl_ord_id)
    # Stop means cancel, not assume-flat.
    controller.stop("operator", "end of session", NOW)
    # The account still has outstanding orders.
    assert not controller.shutdown_complete()
    # Ordinary reconciliation sends their cancellations.
    cancels = cycle(controller, 101)
    # No cancellation is assumed complete until its response arrives.
    for cancel in cancels:
        # Resolve each outstanding liability.
        oms.on_cancelled(cancel.cl_ord_id)
    # With zero inventory and no unresolved messages, shutdown is complete.
    assert controller.shutdown_complete()


# A broken feed poller must not bypass the cancellation cycle.
def test_feed_poller_exception_stops_and_cancels():
    # Build an armed application with live quotes.
    controller, oms, _, _, _ = app()
    # Authorize the test symbol.
    controller.arm("operator", ("OGDC",))
    # Publish the desired quote pair.
    controller.quote(want())
    # Send the quotes.
    quotes = cycle(controller)
    # Confirm both orders are resting.
    for order in quotes:
        # The cancellation must refer to known live order identities.
        oms.on_ack(order.cl_ord_id, order.cl_ord_id)
    # Inject a feed-driver failure during the next timer poll.
    def broken_poll(now):
        # Model a decoder or feed-state exception.
        raise ValueError("feed state corrupted")
    # Replace the feed poll callback only.
    controller.poll_feed = broken_poll
    # The same cycle stops quotes and emits normal cancellations.
    cancels = cycle(controller, 101)
    # The shared kill state prevents any subsequent new orders.
    assert controller.kill.tripped
    # Both outstanding quotes were cancelled through OMS reconciliation.
    assert len(cancels) == 2 and all(isinstance(a, CancelOrder) for a in cancels)
