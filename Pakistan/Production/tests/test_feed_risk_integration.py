"""Offline feed-failure integration through the real OMS cancellation path."""
# Use real order and fill domain objects.
from core.model import CancelOrder, DesiredQuotes, Fill, Side
# Exercise working-order exposure through the actual gateway.
from core.risk import PositionLimitCheck
# Use the PSX application recovery state machine.
from venues.psx_market_data import PSXTickRecovery, TickRecord
# Reuse established deterministic OMS fixture construction.
from tests.test_oms import build, reconcile, want, NOW


# A feed gap must use ordinary desired-flat reconciliation, not a second cancel path.
def test_gap_with_partial_fill_and_slow_cancel_confirmation():
    # Construct the real risk gateway and manager.
    oms, _, _ = build(checks=[PositionLimitCheck(100)])
    # Capture outbound messages without connecting to any exchange.
    outbound = []
    # Wire the recovery safety callback to the same ordinary diff path.
    def withdraw(channel, reason):
        # A channel failure withdraws its subscribed symbol.
        oms.set_desired(DesiredQuotes.flat("OGDC"))
        # Risk always permits cancellations through the ordinary gateway.
        outbound.extend(reconcile(oms))
    # Build the actual PSX recovery component with an inert transport.
    feed = PSXTickRecovery(2011, "E1", lambda record: None, withdraw, lambda *a: None, lambda *a: None)
    # Establish the initial contiguous data stream.
    feed.tick("E1", TickRecord("UA201", 1, ()), 100)
    # The test now explicitly enables the strategy's desired quote.
    oms.set_desired(want())
    # Dispatch both sides through real risk accounting.
    quotes = reconcile(oms)
    # Acknowledge the two resting orders.
    for quote in quotes:
        # Both sides are live before the feed failure.
        oms.on_ack(quote.cl_ord_id, quote.cl_ord_id)
    # A missing tick sequence invalidates the shared channel.
    feed.tick("E1", TickRecord("UA201", 3, ()), 101)
    # Both sides were cancelled using the ordinary OMS plan.
    assert len(outbound) == 2 and all(isinstance(a, CancelOrder) for a in outbound)
    # A sent cancel does not remove unconfirmed exposure.
    assert oms.reserved_quantity("OGDC", Side.BUY) == 100
    # A buy execution beats the cancel acknowledgement.
    buy = next(q for q in quotes if q.side is Side.BUY)
    # Apply that execution through the real manager.
    oms.on_fill(Fill(buy.cl_ord_id, "OGDC", Side.BUY, buy.price_minor, 25, NOW))
    # Seventy-five shares remain at risk until the cancellation is confirmed.
    assert oms.reserved_quantity("OGDC", Side.BUY) == 75
    # Confirm the outstanding cancellations.
    for cancel in outbound:
        # Terminal exchange responses finally release unfilled reservations.
        oms.on_cancelled(cancel.cl_ord_id)
    # The filled inventory survived the quote cancellation.
    assert oms.position("OGDC") == 25
    # Cancelling quotes is not inventory liquidation.
    assert not oms.is_flat()
    # No unfilled buy exposure remains.
    assert oms.reserved_quantity("OGDC", Side.BUY) == 0
