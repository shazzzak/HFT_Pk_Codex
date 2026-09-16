"""Tests for the order manager and the audit log.

The tests that matter most here are the ones about orders IN FLIGHT. Every
other behaviour is visible in normal operation and would be noticed; a
double-send only happens when the exchange is slow, which is exactly when
nobody is in a position to notice it.
"""
# the test framework
import pytest
# the domain
from core.model import (CancelOrder, DesiredQuotes, Fill, Order, OrderState,
                        PlaceOrder, QuoteIntent, ReplaceOrder, Side)
# the risk layer
from core.risk import (KillSwitch, KillSwitchCheck, OrderQuantityCheck,
                       RiskGateway)
# the order manager
from core.oms import OrderManager, QuoteTolerance
# the venue
from core.venue import SessionSegment
from venues.psx import PSXVenue, static_session_provider

# a normal trading day, so the venue has a calendar
DAY, OPEN_MS, CLOSE_MS = "2026-09-14", 9 * 3600_000, 15 * 3600_000
# eleven o'clock, mid-session
NOW = 11 * 3600_000


def build(tolerance=None, switch=None, checks=None, use_replace=False):
    """An order manager with a permissive gateway, unless told otherwise.

    use_replace defaults to False, matching the manager's own default and
    mm_backtest's cancel-plus-new behaviour.
    """
    # the venue, with one continuous segment
    venue = PSXVenue(session_provider=static_session_provider(
        {DAY: (SessionSegment(OPEN_MS, CLOSE_MS),)}))
    # the kill switch, shared with the gateway
    sw = switch or KillSwitch()
    # a gateway that allows everything reasonable, so tests isolate the OMS
    gw = RiskGateway(checks or [OrderQuantityCheck(max_quantity=1_000_000)])
    # the order manager under test
    oms = OrderManager(venue=venue, gateway=gw, kill_switch=sw,
                       session_id="T", account="CLIENT001",
                       tolerance=tolerance, use_replace=use_replace)
    # everything the tests need to reach
    return oms, gw, sw


def want(symbol="OGDC", bid_px=9_900, ask_px=10_100, qty=100):
    """A two-sided desire, with defaults so each test states only its point."""
    # one bid and one ask
    return DesiredQuotes(symbol=symbol,
                         bid=QuoteIntent(Side.BUY, bid_px, qty),
                         ask=QuoteIntent(Side.SELL, ask_px, qty))


def reconcile(oms):
    """Run one cycle at a fixed time, with a reference price for risk."""
    # the same arguments every time, so tests differ only in their setup
    return oms.reconcile(NOW, DAY, {"OGDC": 10_000})


# --- the basic diff --------------------------------------------------------

def test_places_both_sides_when_nothing_is_resting():
    """The simplest case: we want quotes, there are none, so send them."""
    # a fresh manager
    oms, _, _ = build()
    # ask for a two-sided quote
    oms.set_desired(want())
    # one cycle
    acts = reconcile(oms)
    # two placements, one per side
    assert len(acts) == 2
    assert all(isinstance(a, PlaceOrder) for a in acts)
    assert {a.side for a in acts} == {Side.BUY, Side.SELL}


def test_does_nothing_when_the_resting_quote_already_matches():
    """THE CHURN SAVING. An unchanged desire must not produce a message."""
    # a fresh manager
    oms, _, _ = build()
    # ask, send, and acknowledge both orders
    oms.set_desired(want())
    for a in reconcile(oms):
        oms.on_ack(a.cl_ord_id, "X" + a.cl_ord_id)
    # ask for exactly the same thing again
    oms.set_desired(want())
    # nothing should go out
    assert reconcile(oms) == []


def test_cancels_when_the_desire_is_withdrawn():
    """A side that is no longer wanted gets cancelled."""
    # a fresh manager with both sides resting
    oms, _, _ = build()
    oms.set_desired(want())
    for a in reconcile(oms):
        oms.on_ack(a.cl_ord_id, "X" + a.cl_ord_id)
    # now want only the bid
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_900, 100)))
    # one cancel, for the ask
    acts = reconcile(oms)
    assert len(acts) == 1 and isinstance(acts[0], CancelOrder)


def test_a_price_change_is_a_cancel_by_default_not_an_amendment():
    """DEFAULT: cancel now, place on a later cycle. Matching the backtest.

    PSX accepts Cancel/Replace (MsgType G) and an amendment is the better wire
    mechanic -- there is no window in which we have cancelled and are not yet
    quoting. But mm_backtest._requote sends a cancel and a replacement as two
    independent messages, and the old order stays fillable until its cancel
    lands. Every measured result was produced under that behaviour, INCLUDING
    the fills taken in that window. An engine that amends is not the thing that
    was backtested, so amendment is opt-in.
    """
    # a fresh manager with a resting, acknowledged bid
    oms, _, _ = build()
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_900, 100)))
    first = reconcile(oms)
    oms.on_ack(first[0].cl_ord_id, "X1")
    # move the desired price one tick
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_901, 100)))
    # a cancel, carrying both our ids and the exchange's
    acts = reconcile(oms)
    assert len(acts) == 1 and isinstance(acts[0], CancelOrder)
    assert acts[0].orig_cl_ord_id == first[0].cl_ord_id
    assert acts[0].exchange_order_id == "X1"
    # NOTHING ELSE GOES OUT while the cancel is unconfirmed. PSX assigns
    # OrderID on the acknowledgement, so until this is answered there is one
    # order on this side and the in-flight rule forbids touching it.
    assert reconcile(oms) == []
    # the old order is still working and still fillable at the ORIGINAL price,
    # which is exactly the exposure the backtest models
    working = oms.working_orders("OGDC")
    assert len(working) == 1 and working[0].price_minor == 9_900
    # the exchange confirms the cancel
    oms.on_cancelled(acts[0].cl_ord_id)
    # NOW the replacement goes out, at the new price
    acts = reconcile(oms)
    assert len(acts) == 1 and isinstance(acts[0], PlaceOrder)
    assert acts[0].price_minor == 9_901


def test_a_price_change_becomes_one_amendment_when_replace_is_enabled():
    """OPT-IN: the better mechanic is available, behind a deliberate flag."""
    # the same manager, with amendments turned on
    oms, _, _ = build(use_replace=True)
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_900, 100)))
    first = reconcile(oms)
    oms.on_ack(first[0].cl_ord_id, "X1")
    # move the desired price one tick
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_901, 100)))
    # ONE amendment, carrying both our ids and the exchange's
    acts = reconcile(oms)
    assert len(acts) == 1 and isinstance(acts[0], ReplaceOrder)
    assert acts[0].price_minor == 9_901
    assert acts[0].orig_cl_ord_id == first[0].cl_ord_id
    assert acts[0].exchange_order_id == "X1"
    # nothing further goes out while the amendment is unconfirmed
    assert reconcile(oms) == []
    # THE OLD TERMS ARE STILL LIVE until the exchange confirms, so the order is
    # still working and can still be filled at the ORIGINAL price
    working = oms.working_orders("OGDC")
    assert len(working) == 1 and working[0].price_minor == 9_900
    # the exchange confirms the new terms
    oms.on_replaced(acts[0].cl_ord_id, 9_901, 100)
    # and the resting order now carries them
    working = oms.working_orders("OGDC")
    assert len(working) == 1 and working[0].price_minor == 9_901
    # with nothing left to do
    assert reconcile(oms) == []


def test_a_qty_tolerance_does_requote_when_asked_to():
    """The behaviour is available, it is just not the default.

    Run with use_replace=True so the assertion is about the TOLERANCE deciding
    to requote, not about which message a requote becomes.
    """
    # a manager that requotes once the remainder drops below 80% of the target
    oms, _, _ = build(tolerance=QuoteTolerance(price_ticks=0, qty_ratio=0.8),
                      use_replace=True)
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_900, 100)))
    first = reconcile(oms)
    oms.on_ack(first[0].cl_ord_id, "X1")
    # most of it executes, leaving 30 against a target of 100
    oms.on_fill(Fill(cl_ord_id=first[0].cl_ord_id, symbol="OGDC",
                     side=Side.BUY, price_minor=9_900, quantity=70,
                     timestamp_ms=NOW))
    # the same desire now produces an AMENDMENT restoring the size, because
    # 30 < 0.8 * 100. The spec is explicit that Cancel/Replace is the message
    # for changing quantity; Cancel Request is only for killing a remainder.
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_900, 100)))
    acts = reconcile(oms)
    assert len(acts) == 1 and isinstance(acts[0], ReplaceOrder)
    assert acts[0].quantity == 100


# --- the churn lever -------------------------------------------------------

def test_price_tolerance_suppresses_a_small_reprice():
    """One tick of hysteresis holds the quote and keeps the queue position."""
    # a manager that tolerates a one-tick drift
    oms, _, _ = build(tolerance=QuoteTolerance(price_ticks=1))
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_900, 100)))
    first = reconcile(oms)
    oms.on_ack(first[0].cl_ord_id, "X1")
    # a one-tick move is within tolerance: nothing goes out
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_901, 100)))
    assert reconcile(oms) == []
    # a two-tick move is not
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_902, 100)))
    assert len(reconcile(oms)) == 1


# --- risk integration ------------------------------------------------------

def test_a_risk_rejection_leaves_no_trace_in_the_order_state():
    """A refused order was never sent, so nothing may think it exists."""
    # a gateway that refuses anything over 10 shares
    oms, _, _ = build(checks=[OrderQuantityCheck(max_quantity=10)])
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_900, 100)))
    # nothing goes out
    assert reconcile(oms) == []
    # and nothing is recorded as working
    assert oms.working_orders() == []
    # a later cycle with an acceptable size still works, so the side is not stuck
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_900, 10)))
    assert len(reconcile(oms)) == 1


def test_an_exchange_reject_frees_the_side_to_be_requoted():
    """A refused order must not block that side forever."""
    # a fresh manager
    oms, _, _ = build()
    oms.set_desired(DesiredQuotes(symbol="OGDC",
                                  bid=QuoteIntent(Side.BUY, 9_900, 100)))
    first = reconcile(oms)
    # the exchange refuses it
    oms.on_rejected(first[0].cl_ord_id, "unknown symbol")
    # the side is empty again, so the next cycle re-quotes
    acts = reconcile(oms)
    assert len(acts) == 1 and isinstance(acts[0], PlaceOrder)


def test_an_event_for_an_unknown_order_is_recorded_not_raised():
    """Throwing here would take the session down with live orders resting."""
    # collect what the audit sink sees
    seen = []
    # a manager wired to that sink
    venue = PSXVenue(session_provider=static_session_provider(
        {DAY: (SessionSegment(OPEN_MS, CLOSE_MS),)}))
    oms = OrderManager(venue=venue,
                       gateway=RiskGateway([OrderQuantityCheck(1_000_000)]),
                       kill_switch=KillSwitch(), session_id="T",
                       account="CLIENT001",
                       on_event=lambda e, p: seen.append(e))
    # a fill arrives for an order we never sent
    oms.on_fill(Fill(cl_ord_id="nope", symbol="OGDC", side=Side.BUY,
                     price_minor=9_900, quantity=100, timestamp_ms=NOW))
    # it is recorded
    assert "fill_unknown_order" in seen
    # AND the position is still counted, because the shares are real whatever
    # our records say
    assert oms.position("OGDC") == 100


# --- the venue's wire rules ------------------------------------------------

def test_price_formatting_is_exact_and_has_one_decimal_point():
    """PSX Appendix C prohibits '.' everywhere except once in a price field."""
    # the venue
    venue = PSXVenue(session_provider=static_session_provider(
        {DAY: (SessionSegment(OPEN_MS, CLOSE_MS),)}))
    # whole rupees, sub-rupee values, and a value that a float would mangle
    assert venue.format_price(10_000) == "100.00"
    assert venue.format_price(9_901) == "99.01"
    assert venue.format_price(1) == "0.01"
    assert venue.format_price(28_868) == "288.68"
    # exactly one decimal point, which is the whole constraint
    assert venue.format_price(12_345).count(".") == 1
    # NO FLOAT ANYWHERE. 0.1 + 0.2 arithmetic is what this avoids: a price of
    # 70 paisa must render as 0.70, not 0.7000000000000001.
    assert venue.format_price(70) == "0.70"


def test_prohibited_characters_are_caught_where_the_id_is_made():
    """An exchange reject for a malformed id arrives mid-session; this does not."""
    # the venue
    venue = PSXVenue(session_provider=static_session_provider(
        {DAY: (SessionSegment(OPEN_MS, CLOSE_MS),)}))
    # every character PSX Appendix C prohibits, checked individually
    for bad in ";|`~#^'%*,?":
        with pytest.raises(ValueError):
            venue.validate_text(f"SESS{bad}01", "session_id")
    # non-printables are prohibited in every tag
    with pytest.raises(ValueError):
        venue.validate_text("SESS\x01", "session_id")
    # the hyphen our order ids use is NOT prohibited, which is why it was chosen
    assert venue.validate_text("T-00000001", "cl_ord_id") == "T-00000001"


def test_a_session_id_the_exchange_would_reject_fails_at_construction():
    """Discovered at startup, not as a reject on every order."""
    # the venue
    venue = PSXVenue(session_provider=static_session_provider(
        {DAY: (SessionSegment(OPEN_MS, CLOSE_MS),)}))
    # a session id containing a prohibited character
    with pytest.raises(ValueError):
        OrderManager(venue=venue,
                     gateway=RiskGateway([OrderQuantityCheck(1_000_000)]),
                     kill_switch=KillSwitch(), session_id="SESS#1",
                     account="CLIENT001")


def test_a_venue_that_requires_an_account_refuses_to_start_without_one():
    """PSX makes Account (tag 1) required; every order would be rejected."""
    # the venue
    venue = PSXVenue(session_provider=static_session_provider(
        {DAY: (SessionSegment(OPEN_MS, CLOSE_MS),)}))
    # constructing without a client code must fail here, not at go-live
    with pytest.raises(ValueError):
        OrderManager(venue=venue,
                     gateway=RiskGateway([OrderQuantityCheck(1_000_000)]),
                     kill_switch=KillSwitch(), session_id="T")


def test_the_account_is_carried_on_every_order():
    """It is a required field, so it must be on the action the encoder sees."""
    # a fresh manager
    oms, _, _ = build()
    oms.set_desired(want())
    # every placement carries the client code
    for a in reconcile(oms):
        assert a.account == "CLIENT001"


if __name__ == "__main__":
    # A pytest file is not a script: there is no runner, and the project root is
    # not on the import path. Say so rather than failing with ModuleNotFoundError.
    raise SystemExit(
        "This is a pytest file, not a script.\n"
        "Run the suite from the Production directory:\n"
        "    python -m pytest -q\n"
        "or one file:\n"
        "    python -m pytest tests/test_audit.py -q")
