"""Tests for the risk gateway.

These are the tests that matter most in the whole system, because this is the
layer that is supposed to hold when everything else is wrong. Each one asserts
a REJECTION path: a control that never rejects in a test has never been shown
to work, and a control that silently stopped working looks exactly like a
control that was never needed.
"""
# the test framework
import pytest
# the domain
from core.model import (Action, CancelOrder, Fill, InvalidTransition, Order,
                        OrderState, PlaceOrder, QuoteIntent, DesiredQuotes,
                        Side)
# the risk layer
from core.risk import (KillSwitch, KillSwitchCheck, MessageRateCheck,
                       OrderQuantityCheck, OrderToTradeRatioCheck,
                       OrderValueCheck, PositionLimitCheck, PriceBandCheck,
                       RiskContext, RiskGateway, TradingWindowCheck)
# the venue interface pieces
from core.venue import MarketPhase, PriceBand, SessionSegment
# the PSX implementation
from venues.psx import PSXVenue, static_session_provider


# --- fixtures --------------------------------------------------------------

# 09:32 to 15:30 as milliseconds past midnight, a normal PSX weekday
OPEN_MS, CLOSE_MS = 9 * 3600_000 + 32 * 60_000, 15 * 3600_000 + 30 * 60_000
# a Friday, which breaks for Jumu'ah: two segments with a gap in between
FRI_A, FRI_B = (OPEN_MS, 12 * 3600_000), (14 * 3600_000 + 30 * 60_000, CLOSE_MS)


@pytest.fixture
def venue():
    """A PSX venue with one ordinary day and one Friday."""
    # the calendar these tests run against
    return PSXVenue(session_provider=static_session_provider({
        # a normal weekday: one continuous segment
        "2026-09-14": (SessionSegment(OPEN_MS, CLOSE_MS),),
        # a Friday: two segments with the prayer break between them
        "2026-09-18": (SessionSegment(*FRI_A), SessionSegment(*FRI_B)),
    }))


@pytest.fixture
def ctx():
    """A context in the middle of a normal trading day, flat, with a mid."""
    # 11:00, no position, reference price 100.00 PKR = 10,000 paisa
    return RiskContext(date="2026-09-14", timestamp_ms=11 * 3600_000,
                       position=0, reference_price_minor=10_000)


def order(price_minor=10_000, quantity=100, side=Side.BUY, cl="c1"):
    """A placement, with sensible defaults so each test states only its point."""
    # one new order
    return PlaceOrder(symbol="OGDC", cl_ord_id=cl, side=side,
                      price_minor=price_minor, quantity=quantity)


# --- the cancel rule -------------------------------------------------------

def test_cancel_is_approved_even_when_kill_switch_is_tripped(ctx):
    """THE most important test here: a tripped switch must not trap us.

    The kill switch exists to remove exposure. If it also blocked the cancels
    that remove exposure, tripping it would leave the whole book resting with
    no way to pull it.
    """
    # a switch that is already tripped
    switch = KillSwitch()
    switch.trip("test", by="unit-test", timestamp_ms=ctx.timestamp_ms)
    # a gateway whose only control is that switch
    gw = RiskGateway([KillSwitchCheck(switch)])
    # a new order must be refused
    assert not gw.authorise(order(), ctx).allowed
    # and the cancel must go through
    assert gw.authorise(CancelOrder(symbol="OGDC", cl_ord_id="c1"),
                        ctx).allowed


def test_cancel_is_approved_even_at_the_message_rate_limit(ctx):
    """A rate limit must never be the reason an order cannot be pulled."""
    # allow one message per second, then exhaust it
    gw = RiskGateway([MessageRateCheck(max_per_second=1)])
    # the first placement consumes the budget
    assert gw.authorise(order(cl="c1"), ctx).allowed
    # the second placement is correctly refused
    assert not gw.authorise(order(cl="c2"), ctx).allowed
    # but a cancel still goes out
    assert gw.authorise(CancelOrder(symbol="OGDC", cl_ord_id="c1"),
                        ctx).allowed


# --- the kill switch -------------------------------------------------------

def test_kill_switch_keeps_the_first_reason():
    """A trip cascades; the cause must not be overwritten by a symptom."""
    # a fresh switch
    switch = KillSwitch()
    # the real cause trips it first
    switch.trip("position limit breached", by="risk", timestamp_ms=1)
    # a downstream failure tries to trip it again
    switch.trip("session disconnected", by="fix", timestamp_ms=2)
    # the original cause survives
    assert switch.reason == "position limit breached"
    assert switch.tripped_by == "risk"
    assert switch.tripped_at_ms == 1


def test_kill_switch_notifies_listeners_once():
    """The order manager relies on this to flatten the moment it trips."""
    # collect the notifications
    seen = []
    # a switch with one listener
    switch = KillSwitch()
    switch.add_listener(lambda s: seen.append(s.reason))
    # trip it twice
    switch.trip("first", by="op", timestamp_ms=1)
    switch.trip("second", by="op", timestamp_ms=2)
    # only the first trip fires, so the OMS does not flatten twice
    assert seen == ["first"]


# --- individual controls ---------------------------------------------------

def test_trading_window_blocks_before_the_open(venue, ctx):
    """SECP s7: no algorithmic orders outside continuous trading."""
    # a gateway with only the window control
    gw = RiskGateway([TradingWindowCheck(venue)])
    # 09:00, before the 09:32 open
    early = RiskContext(date="2026-09-14", timestamp_ms=9 * 3600_000)
    # the order is refused
    assert not gw.authorise(order(), early).allowed


def test_trading_window_blocks_during_the_friday_break(venue):
    """The Jumu'ah gap is closed, even though the day has not ended."""
    # only the window control
    gw = RiskGateway([TradingWindowCheck(venue)])
    # 13:00 on the Friday, inside the break
    mid = RiskContext(date="2026-09-18", timestamp_ms=13 * 3600_000)
    # refused
    assert not gw.authorise(order(), mid).allowed
    # 11:00 on the same Friday, inside the first segment
    early = RiskContext(date="2026-09-18", timestamp_ms=11 * 3600_000)
    # allowed
    assert gw.authorise(order(), early).allowed


def test_friday_tradeable_time_excludes_the_break(venue):
    """Wall-clock time to the close overstates what can be unwound."""
    # 11:00 on the Friday
    now = 11 * 3600_000
    # tradeable milliseconds remaining, summed over the two segments
    tradeable = venue.tradeable_ms_remaining("2026-09-18", now)
    # naive wall clock to the close
    wall_clock = CLOSE_MS - now
    # the break is real time that cannot be traded
    assert tradeable < wall_clock
    # and the difference is exactly the length of the break
    assert wall_clock - tradeable == FRI_B[0] - FRI_A[1]


def test_price_band_rejects_when_there_is_no_reference_price(venue):
    """No mid means the book is untrustworthy, which is when prices go wrong."""
    # the house band needs a reference to work against
    gw = RiskGateway([PriceBandCheck(venue, house_band_pct=5.0)])
    # a context with no reference price
    blind = RiskContext(date="2026-09-14", timestamp_ms=11 * 3600_000,
                        reference_price_minor=None)
    # the absence is a rejection, not a pass
    assert not gw.authorise(order(), blind).allowed


def test_price_band_rejects_outside_the_house_band(venue, ctx):
    """Catches a plausible-looking price computed from a stale book."""
    # 5% either side of the reference
    gw = RiskGateway([PriceBandCheck(venue, house_band_pct=5.0)])
    # 4% below the 10,000 reference: inside
    assert gw.authorise(order(price_minor=9_600), ctx).allowed
    # 6% below: outside
    assert not gw.authorise(order(price_minor=9_400), ctx).allowed


def test_price_band_rejects_outside_the_exchange_band(ctx):
    """The exchange would reject it on arrival, and that counts against us."""
    # a venue that publishes a tight circuit band
    v = PSXVenue(
        session_provider=static_session_provider(
            {"2026-09-14": (SessionSegment(OPEN_MS, CLOSE_MS),)}),
        band_provider=lambda sym: PriceBand(upper_minor=10_200,
                                            lower_minor=9_800))
    # a wide house band, so only the exchange band can bind
    gw = RiskGateway([PriceBandCheck(v, house_band_pct=50.0)])
    # inside the circuit band
    assert gw.authorise(order(price_minor=10_100), ctx).allowed
    # above it
    assert not gw.authorise(order(price_minor=10_300), ctx).allowed


def test_order_value_and_quantity_limits(ctx):
    """SECP s8.2 and s8.3."""
    # cap the order at 1,000,000 paisa = 10,000 PKR
    gw = RiskGateway([OrderValueCheck(max_order_value_minor=1_000_000)])
    # 100 shares at 10,000 paisa = 1,000,000: exactly at the limit, allowed
    assert gw.authorise(order(quantity=100), ctx).allowed
    # 101 shares takes it over
    assert not gw.authorise(order(quantity=101), ctx).allowed
    # a separate gateway for the share-count control
    gw2 = RiskGateway([OrderQuantityCheck(max_quantity=500)])
    # at the cap
    assert gw2.authorise(order(quantity=500), ctx).allowed
    # over it
    assert not gw2.authorise(order(quantity=501), ctx).allowed


def test_position_limit_uses_the_worst_case_not_the_current_position():
    """Checking the current position lets legal orders breach in aggregate."""
    # a 1,000-share ceiling
    gw = RiskGateway([PositionLimitCheck(max_absolute_position=1_000)])
    # already long 900, which is itself inside the limit
    ctx = RiskContext(date="2026-09-14", timestamp_ms=11 * 3600_000,
                      position=900, reference_price_minor=10_000)
    # buying 100 more lands exactly on the limit
    assert gw.authorise(order(quantity=100, side=Side.BUY), ctx).allowed
    # buying 200 more would breach it if filled, so it is refused now
    assert not gw.authorise(order(quantity=200, side=Side.BUY), ctx).allowed
    # selling is unaffected: it reduces the long
    assert gw.authorise(order(quantity=200, side=Side.SELL), ctx).allowed


def test_message_rate_burst_window_is_tighter_than_the_second(ctx):
    """A strategy oscillating at 100ms looks fine averaged over a second."""
    # 100 per second, but at most 2 in any 100ms
    check = MessageRateCheck(max_per_second=100, burst_window_ms=100,
                             max_per_burst=2)
    # only the rate control
    gw = RiskGateway([check])
    # two messages at the same instant are fine
    assert gw.authorise(order(cl="a"), ctx).allowed
    assert gw.authorise(order(cl="b"), ctx).allowed
    # the third inside the same 100ms is a burst
    assert not gw.authorise(order(cl="c"), ctx).allowed
    # 200ms later the window has moved on and quoting resumes
    later = RiskContext(date=ctx.date, timestamp_ms=ctx.timestamp_ms + 200,
                        reference_price_minor=ctx.reference_price_minor)
    assert gw.authorise(order(cl="d"), later).allowed


def test_order_to_trade_ratio_measures_but_does_not_bind_by_default(ctx):
    """Default is measure-only: it counts from day one and blocks nothing."""
    # no enforce flag, so the limit is inert
    check = OrderToTradeRatioCheck(max_ratio=1.0,
                                   min_messages_before_binding=1)
    # only the ratio control
    gw = RiskGateway([check])
    # send far more messages than a 1:1 ratio with no fills could ever allow
    for i in range(50):
        decision = gw.authorise(order(cl=f"c{i}"), ctx)
        # every one is approved
        assert decision.allowed
    # the ratio was tracked the whole time
    assert check.messages == 50 and check.trades == 0
    # and the reading reaches the audit log through the approval's details,
    # so an approved order records what the ratio was when it was sent
    assert any("measure-only" in d.reason for d in decision.details)


def test_order_to_trade_ratio_can_be_switched_on_at_runtime(ctx):
    """The toggle is flippable live, from the hot-reloaded config."""
    # start in measure-only
    check = OrderToTradeRatioCheck(max_ratio=5.0,
                                   min_messages_before_binding=4)
    # only the ratio control
    gw = RiskGateway([check])
    # ten messages, no fills: a ratio no limit would tolerate
    for i in range(10):
        assert gw.authorise(order(cl=f"c{i}"), ctx).allowed
    # turn the limit on without restarting anything
    check.enforce = True
    # the same state is now refused
    assert not gw.authorise(order(cl="c10"), ctx).allowed
    # and turning it back off restores quoting
    check.enforce = False
    assert gw.authorise(order(cl="c11"), ctx).allowed


def test_order_to_trade_ratio_does_not_bind_during_warm_up(ctx):
    """The first order before any fill has an infinite ratio by construction."""
    # bind at 5:1, but only after 10 messages, with enforcement on
    check = OrderToTradeRatioCheck(max_ratio=5.0,
                                   min_messages_before_binding=10,
                                   enforce=True)
    # only the ratio control
    gw = RiskGateway([check])
    # the first ten messages pass despite a mathematically infinite ratio,
    # because the control is inert until the warm-up count is REACHED
    for i in range(10):
        assert gw.authorise(order(cl=f"c{i}"), ctx).allowed
    # the ratio is indeed infinite, with no trades yet
    assert check.ratio == float("inf")
    # the eleventh is the first message evaluated with the warm-up satisfied,
    # so it is the first one the control can refuse
    assert not gw.authorise(order(cl="c10"), ctx).allowed


def test_order_to_trade_ratio_falls_as_fills_arrive(ctx):
    """Executions in the denominator are what let quoting continue."""
    # bind at 5:1 after 4 messages, with enforcement on
    check = OrderToTradeRatioCheck(max_ratio=5.0,
                                   min_messages_before_binding=4,
                                   enforce=True)
    # only the ratio control
    gw = RiskGateway([check])
    # four messages, no fills: ratio infinite, warm-up just satisfied
    for i in range(4):
        gw.authorise(order(cl=f"c{i}"), ctx)
    # the fifth is refused
    assert not gw.authorise(order(cl="c4"), ctx).allowed
    # one execution arrives, taking the ratio to 4/1
    gw.on_trade()
    # now under the 5:1 ceiling, so quoting resumes
    assert gw.authorise(order(cl="c5"), ctx).allowed


# --- the gateway itself ----------------------------------------------------

def test_gateway_with_no_checks_is_refused_at_construction():
    """An empty gateway approves everything, which is never what was meant."""
    # constructing one must fail loudly
    with pytest.raises(ValueError):
        RiskGateway([])


def test_rejected_actions_do_not_advance_counters(ctx):
    """A blocked order must not consume the message budget it never used."""
    # a switch that blocks everything
    switch = KillSwitch()
    switch.trip("halt", by="test", timestamp_ms=0)
    # the rate control sits behind the switch
    rate = MessageRateCheck(max_per_second=1)
    # the switch is evaluated first
    gw = RiskGateway([KillSwitchCheck(switch), rate])
    # the order is refused by the switch
    assert not gw.authorise(order(cl="a"), ctx).allowed
    # re-arm
    switch.reset(by="test")
    # the single message of budget must still be available
    assert gw.authorise(order(cl="b"), ctx).allowed


def test_every_decision_reaches_the_audit_sink(ctx):
    """SECP s12: the trail must answer why an order was SENT, not only blocked."""
    # collect what the sink receives
    log = []
    # a permissive gateway with the sink wired in
    gw = RiskGateway([OrderQuantityCheck(max_quantity=100)],
                     on_decision=lambda a, d: log.append((a, d)))
    # one approval and one rejection
    gw.authorise(order(quantity=50), ctx)
    gw.authorise(order(quantity=500), ctx)
    # both are recorded
    assert len(log) == 2
    # the approval names the gateway, the rejection names the control
    assert log[0][1].allowed and log[0][1].check == "gateway"
    assert not log[1][1].allowed and log[1][1].check == "order_quantity"


# --- the domain model ------------------------------------------------------

def test_tick_rounding_is_never_more_aggressive(venue):
    """Rounding a bid up would spend capture the strategy meant to keep."""
    # PSX ticks are whole paisa, so a fractional input must snap outward
    assert venue.round_to_tick(10_007, Side.BUY) == 10_007
    # a bid never rounds up
    assert venue.round_to_tick(10_007, Side.BUY) <= 10_007
    # an ask never rounds down
    assert venue.round_to_tick(10_007, Side.SELL) >= 10_007


def test_pending_cancel_still_counts_as_exposure():
    """A cancel can lose the race with an aggressor; the risk is real."""
    # an order that is resting
    o = Order(cl_ord_id="c1", symbol="OGDC", side=Side.BUY,
              price_minor=10_000, quantity=100, state=OrderState.PENDING_NEW)
    # acknowledge it
    o.on_ack("X1")
    # send a cancel
    o.on_cancel_sent()
    # it is not terminal and it can still be filled
    assert o.state.is_working
    assert o.leaves_quantity == 100
    # and indeed a fill may still arrive
    o.on_fill(100)
    assert o.state is OrderState.FILLED


def test_impossible_transitions_raise_rather_than_corrupt_state():
    """A silently-ignored bad transition means our position is wrong."""
    # an order the exchange has already rejected
    o = Order(cl_ord_id="c1", symbol="OGDC", side=Side.BUY,
              price_minor=10_000, quantity=100, state=OrderState.PENDING_NEW)
    o.on_rejected("unknown symbol")
    # a fill on it cannot be reconciled and must not be absorbed quietly
    with pytest.raises(InvalidTransition):
        o.on_fill(100)


def test_desired_quotes_absence_means_cancel_not_leave_alone():
    """The protocol that makes 'go flat' expressible at all."""
    # the flat state the kill switch asks for
    d = DesiredQuotes.flat("OGDC")
    # both sides are absent, which the OMS reads as 'nothing should rest'
    assert d.bid is None and d.ask is None
    # and a one-sided desire leaves the other side explicitly empty
    one = DesiredQuotes(symbol="OGDC",
                        bid=QuoteIntent(Side.BUY, 9_900, 100))
    assert one.side(Side.BUY) is not None
    assert one.side(Side.SELL) is None


# --- the exchange's published phase beats any calendar ---------------------

def test_an_unknown_phase_blocks_quoting(venue, ctx):
    """Before the first Trading Session Status we have not been told anything.

    "Not told" is not permission. The engine stays dark for up to one status
    interval after connecting, which is correct.
    """
    # a provider that has received nothing yet
    gw = RiskGateway([TradingWindowCheck(venue, phase_provider=lambda s: None)])
    # nothing goes out
    assert not gw.authorise(order(), ctx).allowed
    # and an unrecognised code is treated the same way, never as 'probably open'
    gw2 = RiskGateway([TradingWindowCheck(
        venue, phase_provider=lambda s: venue.parse_phase("Z"))])
    assert not gw2.authorise(order(), ctx).allowed


def test_the_published_phase_overrides_the_calendar(venue, ctx):
    """A halt at 11am is invisible to a calendar and obvious in the feed."""
    # the calendar says this instant is inside continuous trading
    assert venue.is_continuous(ctx.date, ctx.timestamp_ms)
    # but the exchange says the market is halted
    gw = RiskGateway([TradingWindowCheck(
        venue, phase_provider=lambda s: venue.parse_phase("H0"))])
    decision = gw.authorise(order(), ctx)
    # the exchange wins
    assert not decision.allowed and "HALTED" in decision.reason


def test_a_security_suspended_for_the_day_cannot_be_quoted(venue, ctx):
    """TradingPhaseCode 1st digit '1', carried per instrument on the snapshot."""
    # continuous market, but this security is suspended all day
    gw = RiskGateway([TradingWindowCheck(
        venue, phase_provider=lambda s: venue.parse_phase("T1"))])
    decision = gw.authorise(order(), ctx)
    # refused, naming the reason
    assert not decision.allowed and "suspended" in decision.reason


def test_continuous_matching_is_the_only_tradeable_phase(venue, ctx):
    """Call auctions accept orders; they do not match continuously."""
    # every phase the spec defines, with whether we may quote in it
    cases = {"T0": True, "S0": False, "O0": False, "N0": False, "V0": False,
             "B02": False, "H0": False, "C0": False, "A0": False, "E0": False}
    # each one, through the gateway
    for code, should_pass in cases.items():
        gw = RiskGateway([TradingWindowCheck(
            venue, phase_provider=lambda s, c=code: venue.parse_phase(c))])
        assert gw.authorise(order(), ctx).allowed is should_pass, code


def test_the_friday_break_is_named_by_the_exchange_not_inferred(venue, ctx):
    """'B' with 2nd digit '2' is the Jumu'ah break, straight from the feed."""
    # the code the exchange publishes during the Friday lunch break
    state = venue.parse_phase("B02")
    # parsed into a phase and a human-readable reason
    assert state.phase is MarketPhase.BREAK
    assert state.break_reason == "Friday lunch break"
    # and the rejection says so, so an operator reading the log understands
    gw = RiskGateway([TradingWindowCheck(
        venue, phase_provider=lambda s: state)])
    assert "Friday lunch break" in gw.authorise(order(), ctx).reason


# --- price limits: the exchange's "no limit" sentinels ---------------------

def test_an_absent_price_bound_constrains_nothing():
    """PSX publishes 999999999.9999 for 'no up limit'; None says it honestly."""
    # a band with no upper bound
    band = PriceBand(upper_minor=None, lower_minor=9_000)
    # anything above passes
    assert band.contains(99_999_999)
    # but the lower bound still binds
    assert not band.contains(8_999)
    # a band with neither bound is not a band, and says so
    assert PriceBand().is_unbounded


# --- feed precision: never truncate a price silently -----------------------

def test_parsing_a_price_refuses_to_truncate(venue):
    """Silent rounding of a price is the quietest possible way to be wrong."""
    # ordinary two-decimal prices round-trip exactly
    assert venue.parse_price("100.00") == 10_000
    assert venue.parse_price("288.68") == 28_868
    # the feed's four- and six-decimal padding is harmless when it is zeros
    assert venue.parse_price("288.6800") == 28_868
    assert venue.parse_price("0.010000") == 1
    # but real precision the minor unit cannot hold is an ERROR, not a rounding
    with pytest.raises(ValueError):
        venue.parse_price("100.0001")
    # and so is a malformed value
    with pytest.raises(ValueError):
        venue.parse_price("1.2.3")
    with pytest.raises(ValueError):
        venue.parse_price("")
    # a round trip through format_price is exact
    for minor in (1, 70, 9_901, 10_000, 28_868):
        assert venue.parse_price(venue.format_price(minor)) == minor


if __name__ == "__main__":
    # A pytest file is not a script: there is no runner, and the project root is
    # not on the import path. Say so rather than failing with ModuleNotFoundError.
    raise SystemExit(
        "This is a pytest file, not a script.\n"
        "Run the suite from the Production directory:\n"
        "    python -m pytest -q\n"
        "or one file:\n"
        "    python -m pytest tests/test_audit.py -q")
