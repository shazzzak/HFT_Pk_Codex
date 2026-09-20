"""Risk reservations across asynchronous order lifecycles."""
# Parameterize both inventory directions.
import pytest
# Use the public domain messages.
from core.model import DesiredQuotes, Fill, QuoteIntent, Side, ReplaceOrder
# Exercise the gateway used by the OMS.
from core.risk import PositionLimitCheck
# Reuse only the venue/OMS setup, not expected arithmetic.
from tests.test_oms import build, reconcile, NOW
# Select a policy which can hold multiple orders per side.
from core.oms import QuoteTolerance


# Express one desired quote without an offsetting opposite side.
def desire(oms, side, qty, price=10000):
    # Construct the intent on the selected side only.
    intent = QuoteIntent(side, price, qty)
    # Update through the production API.
    oms.set_desired(DesiredQuotes("OGDC", bid=intent if side is Side.BUY else None,
                                 # The sell case mirrors the buy case.
                                 ask=intent if side is Side.SELL else None))


# Cover both signs without relying on netted open orders.
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_topup_counts_existing_leaves(side):
    # Two clips must not each pass against filled inventory alone.
    oms, _, _ = build(tolerance=QuoteTolerance(quantity_policy="queue_preserving"), checks=[PositionLimitCheck(100)])
    # Start with 60 shares reserved.
    desire(oms, side, 60)
    # Send through risk and acknowledge.
    first, = reconcile(oms)
    # Make the first order eligible for a top-up.
    oms.on_ack(first.cl_ord_id, "X1")
    # An extra 50 would make 110 shares executable.
    desire(oms, side, 110)
    # Reject that increment.
    assert reconcile(oms) == []
    # The rejected increment must not create a reservation.
    assert oms.reserved_quantity("OGDC", side) == 60
    # Exactly 100 is allowed.
    desire(oms, side, 100)
    # Only the 40-share increment is sent.
    second, = reconcile(oms)
    # Prove aggregation across independently identified orders.
    assert second.quantity == 40
    # The increment is reserved before its acknowledgement.
    assert oms.reserved_quantity("OGDC", side) == 100


# Partial executions must transfer shares from unfilled risk to inventory.
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_partial_fill_and_cancel_rejection_keep_exposure(side):
    # Use an exact-size order to make the lifecycle explicit.
    oms, _, _ = build(checks=[PositionLimitCheck(100)])
    # Reserve the complete limit.
    desire(oms, side, 100)
    # Dispatch and acknowledge.
    first, = reconcile(oms)
    # Establish a live order.
    oms.on_ack(first.cl_ord_id, "X1")
    # Fill 30 shares; the remaining liability is 70.
    oms.on_fill(Fill(first.cl_ord_id, "OGDC", side, 10000, 30, NOW))
    # Position plus same-side liability remains 100.
    assert oms.reserved_quantity("OGDC", side) == 70
    # Request cancellation through normal diff logic.
    oms.set_desired(DesiredQuotes.flat("OGDC"))
    # A sent cancel cannot free risk.
    cancel, = reconcile(oms)
    # The exchange can still fill all remaining shares.
    assert oms.reserved_quantity("OGDC", side) == 70
    # Another fill can beat the cancel.
    oms.on_fill(Fill(first.cl_ord_id, "OGDC", side, 10000, 20, NOW))
    # Keep only the actual remaining shares.
    assert oms.reserved_quantity("OGDC", side) == 50
    # A rejected cancel leaves those shares executable.
    oms.on_cancel_rejected(cancel.cl_ord_id, "too late")
    # No exposure is released on rejection.
    assert oms.reserved_quantity("OGDC", side) == 50
    # Retry the cancellation.
    cancel, = reconcile(oms)
    # Confirmation finally releases unfilled exposure.
    oms.on_cancelled(cancel.cl_ord_id)
    # Filled position remains, but no shares are outstanding.
    assert oms.reserved_quantity("OGDC", side) == 0


# Remaining-size amendments require old-plus-new reservation.
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
@pytest.mark.parametrize("accepted", [True, False])
def test_replace_fill_race(side, accepted):
    # Permit 60 old shares plus 40 new remaining shares.
    oms, _, _ = build(tolerance=QuoteTolerance(quantity_policy="exact"), checks=[PositionLimitCheck(100)], use_replace=True)
    # Establish the old generation.
    desire(oms, side, 60)
    # Dispatch the order.
    first, = reconcile(oms)
    # The old generation is live.
    oms.on_ack(first.cl_ord_id, "X1")
    # A same-size reprice could execute 60 old then another 60.
    desire(oms, side, 60, 10010)
    # Reject the unsafe replacement.
    assert reconcile(oms) == []
    # A 40-share replacement fits the conservative cap.
    desire(oms, side, 40, 10010)
    # Dispatch the replacement.
    amendment, = reconcile(oms)
    # Verify this is one replacement, not a cancel/new pair.
    assert isinstance(amendment, ReplaceOrder)
    # Reserve both executable generations.
    assert oms.reserved_quantity("OGDC", side) == 100
    # Thirty old shares fill while the replacement is in flight.
    oms.on_fill(Fill(first.cl_ord_id, "OGDC", side, 10000, 30, NOW))
    # Thirty old leaves plus forty proposed leaves remain.
    assert oms.reserved_quantity("OGDC", side) == 70
    # Resolve either possible exchange response.
    if accepted:
        # The accepted generation has forty remaining shares.
        oms.on_replaced(amendment.cl_ord_id, 10010, 40)
        # Release old leaves and keep only confirmed new terms.
        assert oms.reserved_quantity("OGDC", side) == 40
    # A rejection preserves the old generation.
    else:
        # Discard only the proposed generation.
        oms.on_cancel_rejected(amendment.cl_ord_id, "refused")
        # The old thirty shares remain live.
        assert oms.reserved_quantity("OGDC", side) == 30


# Suspended orders cannot disappear from accounting during quote planning.
def test_suspended_order_stays_reserved():
    # Use the real OMS risk path.
    oms, _, _ = build(checks=[PositionLimitCheck(100)])
    # Establish one order.
    desire(oms, Side.BUY, 100)
    # Send through the gateway.
    first, = reconcile(oms)
    # Acknowledge before suspension.
    oms.on_ack(first.cl_ord_id, "X1")
    # The exchange holds rather than cancels the order.
    oms.on_suspended(first.cl_ord_id)
    # Never replace an unresolved suspended order with another full clip.
    assert reconcile(oms) == []
    # The liability remains indexed after repeated reconciliation.
    assert oms.reserved_quantity("OGDC", Side.BUY) == 100
    # An operator must not mistake a suspended liability for a flat account.
    assert not oms.is_flat()


# Opposite orders cannot be credited as if they were guaranteed fills.
def test_opposite_side_does_not_offset():
    # Start flat and permit independent endpoints of plus/minus 100.
    oms, _, _ = build(tolerance=QuoteTolerance(quantity_policy="queue_preserving"), checks=[PositionLimitCheck(100)])
    # Both sides may execute independently.
    oms.set_desired(DesiredQuotes("OGDC", QuoteIntent(Side.BUY, 9900, 100), QuoteIntent(Side.SELL, 10100, 100)))
    # Two hundred gross shares are safe at these separate endpoints.
    orders = reconcile(oms)
    # Both quotes fit the cap.
    assert len(orders) == 2
    # Acknowledge each side.
    for order in orders:
        # Allow top-up planning.
        oms.on_ack(order.cl_ord_id, order.cl_ord_id)
    # An extra buy must fail despite the existing sell.
    oms.set_desired(DesiredQuotes("OGDC", QuoteIntent(Side.BUY, 9900, 101), QuoteIntent(Side.SELL, 10100, 100)))
    # Net-zero reasoning would wrongly allow this share.
    assert reconcile(oms) == []


# A terminal old generation cannot be resurrected by a delayed confirmation.
@pytest.mark.parametrize("reply", ["accepted", "rejected"])
def test_full_fill_before_amendment_reply(reply):
    # Allow enough risk to send the old and proposed generations.
    oms, _, _ = build(tolerance=QuoteTolerance(quantity_policy="exact"), checks=[PositionLimitCheck(100)], use_replace=True)
    # Establish the old generation.
    desire(oms, Side.BUY, 60)
    # Dispatch the first order.
    first, = reconcile(oms)
    # Allow replacement planning.
    oms.on_ack(first.cl_ord_id, "X1")
    # Reserve another forty potential shares.
    desire(oms, Side.BUY, 40, 10010)
    # Send the amendment.
    amendment, = reconcile(oms)
    # All old shares execute before the amendment is processed.
    oms.on_fill(Fill(first.cl_ord_id, "OGDC", Side.BUY, 10000, 60, NOW))
    # Terminal orders have no remaining exposure under the existing model.
    assert oms.reserved_quantity("OGDC", Side.BUY) == 0
    # Exercise either delayed lifecycle response.
    if reply == "accepted":
        # The current OMS treats terminal replacement confirmations as no-ops.
        oms.on_replaced(amendment.cl_ord_id, 10010, 40)
    # A rejection likewise cannot restore a finished generation.
    else:
        # Release any proposed terms without altering inventory.
        oms.on_cancel_rejected(amendment.cl_ord_id, "already filled")
    # The confirmed filled position is preserved.
    assert oms.position("OGDC") == 60
    # No risk reservation survives a terminal order.
    assert oms.reserved_quantity("OGDC", Side.BUY) == 0


# New-order rejection is the point at which its entire reservation is released.
def test_new_order_rejection_releases_reservation():
    # Use the actual OMS dispatch path.
    oms, _, _ = build(checks=[PositionLimitCheck(100)])
    # Prepare a complete clip.
    desire(oms, Side.BUY, 100)
    # Dispatch while leaving the order unacknowledged.
    first, = reconcile(oms)
    # Reserve on send, before acceptance.
    assert oms.reserved_quantity("OGDC", Side.BUY) == 100
    # Apply an explicit exchange rejection.
    oms.on_rejected(first.cl_ord_id, "refused")
    # Only that terminal response frees risk.
    assert oms.reserved_quantity("OGDC", Side.BUY) == 0
