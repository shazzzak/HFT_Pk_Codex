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


# ---------------------------------------------------------------------------
# NEVER GIVE UP QUEUE POSITION -- the second-order top-up
# ---------------------------------------------------------------------------
# PSX Regulations 8.5.2: an amendment that RAISES an order's size sends it to
# the back of the queue at that price. A reduction is applied in place and
# keeps its position. So there are two ways to show more size after a partial
# fill, and only one of them is free:
#
#   amend 20 -> 50   all fifty shares go to the back of the queue
#   add an order      the twenty keep their place, thirty join the back
#
# The order manager holds a LIST of orders per side so it can do the second.
# ---------------------------------------------------------------------------
def _queue_oms(use_replace=False):
    """An order manager running the queue-preserving design.

    NOT THE DEFAULT. The default is the single-order design every measured
    number came from, so a test of the list behaviour has to ask for it --
    otherwise it would silently test the wrong thing and pass.
    """
    # the list design, with price changes still triggering a requote
    return build(tolerance=QuoteTolerance(quantity_policy="queue_preserving"),
                 use_replace=use_replace)


def _bid_ids(oms):
    """The ids of every order currently resting on the bid."""
    return {o.cl_ord_id for o in oms.working_orders("OGDC")
            if o.side is Side.BUY}


def _bid_actions(actions, ids=frozenset()):
    """Actions aimed at the bid side.

    CancelOrder CARRIES NO SIDE -- only the id of the order it targets -- so
    filtering on `.side` alone silently drops every cancel, and a test looking
    for two cancels would see zero and fail for the wrong reason. The caller
    passes the bid order ids captured BEFORE the cycle.
    """
    out = []
    # each action in turn
    for a in actions:
        # a place or an amendment names its side directly
        if getattr(a, "side", None) is Side.BUY:
            out.append(a)
            continue
        # a cancel has to be resolved through the order it targets
        if getattr(a, "orig_cl_ord_id", "") in ids:
            out.append(a)
    return out


def _resting_after_partial_fill(oms, filled=30, clip=100):
    """Get one order resting, partially filled, and acknowledged."""
    # the strategy wants a clip on each side
    oms.set_desired(want(qty=clip))
    # the first cycle places them
    actions = reconcile(oms)
    # the bid, which is a place and so does name its side
    bid = [a for a in actions if getattr(a, "side", None) is Side.BUY][0]
    # the exchange acknowledges it
    oms.on_ack(bid.cl_ord_id, "EX-1")
    # and part of it trades
    oms.on_fill(Fill(cl_ord_id=bid.cl_ord_id, symbol="OGDC", side=Side.BUY,
                     price_minor=9_900, quantity=filled, timestamp_ms=OPEN_MS))
    # the order, and its id
    return bid.cl_ord_id


def test_a_partial_fill_is_topped_up_with_a_SECOND_order():
    """The remainder is not cancelled and not amended. It is left alone."""
    # a bid for 100 with 30 filled: 70 resting, 100 still wanted
    oms, _gw, _sw = _queue_oms()
    first_id = _resting_after_partial_fill(oms, filled=30, clip=100)
    # the next cycle
    actions = reconcile(oms)
    # the bid-side actions
    bid_actions = _bid_actions(actions)
    # exactly one, and it is a NEW ORDER -- not a cancel, not an amendment
    assert len(bid_actions) == 1
    assert isinstance(bid_actions[0], PlaceOrder)
    # FOR THE SHORTFALL ONLY. 100 wanted, 70 resting, so 30 -- never 100 again.
    assert bid_actions[0].quantity == 30
    # and the original is untouched: still resting, still 70 to go
    original = [o for o in oms.working_orders("OGDC")
                if o.cl_ord_id == first_id][0]
    assert original.leaves_quantity == 70
    assert original.state is OrderState.PARTIALLY_FILLED


def test_the_topped_up_side_shows_the_full_size_across_two_orders():
    """Two orders, one quote. The total is what the strategy asked for."""
    # a bid for 100 with 30 filled
    oms, _gw, _sw = _queue_oms()
    _resting_after_partial_fill(oms, filled=30, clip=100)
    # the top-up goes out and is acknowledged
    top_up = _bid_actions(reconcile(oms))[0]
    oms.on_ack(top_up.cl_ord_id, "EX-2")
    # both orders are resting on the bid
    bids = [o for o in oms.working_orders("OGDC") if o.side is Side.BUY]
    assert len(bids) == 2
    # and together they show exactly the clip
    assert sum(o.leaves_quantity for o in bids) == 100
    # nothing further is wanted while that holds
    assert not _bid_actions(reconcile(oms), _bid_ids(oms))


def test_the_oldest_order_keeps_its_place_in_the_list():
    """The list IS the queue order, and the reduce path depends on it."""
    # a bid for 100 with 30 filled, then topped up
    oms, _gw, _sw = _queue_oms()
    first_id = _resting_after_partial_fill(oms, filled=30, clip=100)
    top_up = _bid_actions(reconcile(oms))[0]
    oms.on_ack(top_up.cl_ord_id, "EX-2")
    # the bid side, in the order the manager holds it
    bids = [o for o in oms.working_orders("OGDC") if o.side is Side.BUY]
    # the original is first: oldest, and best queue position
    assert bids[0].cl_ord_id == first_id


def test_wanting_less_shrinks_the_YOUNGEST_order_first():
    """Reducing is free under 8.5.2, so spend it on the worst position."""
    # two orders resting: 70 (old) and 30 (new)
    oms, _gw, _sw = _queue_oms(use_replace=True)
    first_id = _resting_after_partial_fill(oms, filled=30, clip=100)
    top_up = _bid_actions(reconcile(oms))[0]
    oms.on_ack(top_up.cl_ord_id, "EX-2")
    # now the strategy wants only 80 on the bid
    oms.set_desired(want(qty=80))
    # the ids before the cycle, so a cancel can be attributed to this side
    ids = _bid_ids(oms)
    # the cycle that trims it
    actions = _bid_actions(reconcile(oms), ids)
    # one action, aimed at the YOUNGER order -- the older one is untouched
    assert len(actions) == 1
    assert actions[0].orig_cl_ord_id == top_up.cl_ord_id
    # and it is an amendment down to 10, which keeps its place under 8.5.2
    assert isinstance(actions[0], ReplaceOrder)
    assert actions[0].quantity == 10
    # the oldest order, with the best position, was never touched
    original = [o for o in oms.working_orders("OGDC")
                if o.cl_ord_id == first_id][0]
    assert original.leaves_quantity == 70


def test_a_price_change_still_moves_everything():
    """Priority at a price we are leaving is worth nothing."""
    # two orders resting on the bid at 9,900
    oms, _gw, _sw = _queue_oms()
    _resting_after_partial_fill(oms, filled=30, clip=100)
    top_up = _bid_actions(reconcile(oms))[0]
    oms.on_ack(top_up.cl_ord_id, "EX-2")
    # the ids before the cycle, because cancels carry no side
    ids = _bid_ids(oms)
    # the market moves and the strategy wants a different price
    oms.set_desired(want(bid_px=9_890, qty=100))
    # the cycle that follows
    actions = _bid_actions(reconcile(oms), ids)
    # nothing is left resting at the stale price
    assert len(actions) == 2
    # and every action is a cancel, since use_replace is off here
    assert all(isinstance(a, CancelOrder) for a in actions)


def test_pulling_the_side_cancels_every_order_on_it():
    """The kill switch route: one wish, and the diff does the rest."""
    # two orders resting on the bid
    oms, _gw, _sw = _queue_oms()
    _resting_after_partial_fill(oms, filled=30, clip=100)
    top_up = _bid_actions(reconcile(oms))[0]
    oms.on_ack(top_up.cl_ord_id, "EX-2")
    # the ids before the cycle, because cancels carry no side
    ids = _bid_ids(oms)
    # want nothing anywhere
    oms.flatten_all("test")
    # the cycle that clears it
    actions = _bid_actions(reconcile(oms), ids)
    # BOTH orders are cancelled, not just one
    assert len(actions) == 2
    assert all(isinstance(a, CancelOrder) for a in actions)
