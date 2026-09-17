"""Tests for the replay harness -- the bridge, not the exchange.

WHAT IS UNDER TEST. `EngineReplay` reuses mm_backtest's exchange wholesale and
replaces one method. So these tests assert the BRIDGE: that a production action
becomes the right message on Backtester's scheduler, that a Backtester lifecycle
event reaches the production order manager, and that the two id schemes stay in
step. The exchange itself is mm_backtest's and is not re-tested here.

These run against the REAL Backtester. A harness tested against a mock of the
thing it exists to integrate with proves nothing about the integration.
"""
# the test runner
import pytest

# skip the whole module cleanly where the research tree is not importable
mm_backtest = pytest.importorskip(
    "mm_backtest", reason="mm_backtest must be on PYTHONPATH for the harness "
                          "tests; run from a checkout where it is")

# the production side
from core.model import (BookLevel, BookSnapshot, DesiredQuotes, QuoteIntent,
                        Side)
from core.oms import OrderManager, QuoteTolerance
from core.risk import KillSwitch, OrderQuantityCheck, RiskGateway
from core.venue import SessionSegment
from venues.psx import PSXVenue
from venues.psx_strategy import MicroMMAdapter
from sim.replay import EngineReplay

# the exchange
from mm_backtest import LatencyModel


# ---------------------------------------------------------------------------
# a market day, in the shape both sides expect
# ---------------------------------------------------------------------------
DAY = "2026-09-16"
# 09:32 to 15:30 in exchange-ms, the shape of a non-Friday PSX session
OPEN_MS = 34_320_000
CLOSE_MS = 55_800_000


class StubMM:
    """A MicrostructureMM-shaped stub with a scripted answer.

    The real strategy is nine hundred lines and is tested elsewhere; using it
    here would mean any failure could be the bridge or the quoting logic.
    """
    # micro_mm advertises this, so Backtester syncs the circuit limits onto it
    wants_limits = True

    def __init__(self, returns=None):
        # what quotes() hands back
        self._returns = returns if returns is not None else {}
        # the attributes the adapter and Backtester read off a strategy
        self.tick = 0.01
        self.enable_age_cross = False
        self.allow_taker = False
        self.tol_ticks = 0.0
        self.ofi_depth_levels = 1
        self.want_taker_side = None
        self.log_fill_state = False
        self.current_window = "none"
        self.limit_up = None
        self.limit_dn = None

    def observe(self, kind, obj, ts_exch, mid):
        # the harness must not call this -- Backtester.run does
        return None

    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # whatever the test scripted
        return self._returns


def build(returns=None, quote_price=289.00, quote_qty=50):
    """A harness wired to a real Backtester, ready to be driven by hand."""
    # the venue, with one continuous segment and no published band
    venue = PSXVenue(session_provider=lambda d: [
        SessionSegment(start_ms=OPEN_MS, end_ms=CLOSE_MS)])
    # the strategy stub
    mm = StubMM(returns if returns is not None
                else {"BUY": (quote_price, quote_qty)})
    # the production adapter over it
    adapter = MicroMMAdapter("PPL", venue, mm, reference_price_minor=28900)
    # the production order manager, with a permissive gateway so these tests
    # isolate the bridge
    switch = KillSwitch()
    gateway = RiskGateway([OrderQuantityCheck(max_quantity=1_000_000)])
    oms = OrderManager(venue=venue, gateway=gateway, kill_switch=switch,
                       session_id="R", account="CLIENT001")
    # CONSTANT LATENCY, so a test can say exactly when a message lands
    cfg = {"latency_model": LatencyModel(decision_ms=0.0,
                                         wire_out_median_ms=100.0,
                                         wire_out_tail_ms=0.0,
                                         wire_in_median_ms=100.0,
                                         wire_in_tail_ms=0.0, tail_prob=0.0),
           "at_price_mode": "queue", "fill_on_crossing_adds": False,
           "log_equity": False, "session_ms": (OPEN_MS, CLOSE_MS)}
    # the harness
    replay = EngineReplay(strategy=mm, adapter=adapter, oms=oms,
                          symbol="PPL", cfg=cfg)
    # the date the risk gateway's window check needs
    replay.session_date = DAY
    # everything a test needs to reach
    return replay, oms, mm


def set_book(replay, bid=289.00, bid_qty=500, ask=289.02, ask_qty=300,
             phase="CONTINUOUS_AUCTION"):
    """Put a two-sided book into Backtester's own Book object.

    Book keeps ONE dict, order_id -> Order, and derives price levels on demand.
    Never a bids/asks mapping -- that order-level state is precisely what makes
    the exact queue tracking possible, so the test builds it the same way.
    """
    # Backtester's book, addressed the way its own handlers do
    book = replay.book
    # clear anything previously resting
    book.o.clear()
    # one historical order each side, keyed by id
    book.o["H1"] = mm_backtest.Order("BUY", bid, bid_qty)
    book.o["H2"] = mm_backtest.Order("SELL", ask, ask_qty)
    # the trading phase the requote gate reads
    book.phase = phase
    # no circuit limits published
    book.limit_up = None
    book.limit_dn = None


# ---------------------------------------------------------------------------
# a production action becomes the right message
# ---------------------------------------------------------------------------
def test_a_place_becomes_an_arrive_scheduled_at_the_latency():
    """One order out, landing exactly one send-latency later."""
    # a harness wanting a bid
    replay, oms, _ = build()
    set_book(replay)
    # one requote cycle at a known knowledge-time
    replay._requote(OPEN_MS + 1_000)
    # one message pending, and it is an arrival
    assert len(replay.pending) == 1
    t_land, _, action, order = replay.pending[0]
    assert action == "ARRIVE"
    # 100 ms of constant send latency
    assert t_land == OPEN_MS + 1_100
    # carrying the price and size the strategy asked for
    assert order.side == "BUY" and order.price == 289.00 and order.qty == 50
    # and Backtester's own counter moved, so its reporting still adds up
    assert replay.stats["n_orders_sent"] == 1


def test_the_order_manager_learns_the_exchange_id_when_the_order_rests():
    """PSX requires OrderID on every later cancel; it arrives with the ack."""
    # send one
    replay, oms, _ = build()
    set_book(replay)
    replay._requote(OPEN_MS + 1_000)
    # land it
    replay._activate_until(OPEN_MS + 2_000)
    # Backtester has it resting
    assert "BUY" in replay.work
    # and the production order manager has been acknowledged, with the oid as
    # the exchange's handle
    working = oms.working_orders("PPL")
    assert len(working) == 1
    assert working[0].exchange_order_id == str(replay.work["BUY"].oid)


def test_an_order_the_market_ran_past_is_reported_as_a_reject():
    """The book moves during the latency window. That is the real reject.

    Note the adapter refuses a quote that crosses the book it was GIVEN, so
    this can only happen the way it happens in life: the quote was passive when
    it was sent and marketable by the time it arrived, 100 ms later.
    """
    # a perfectly ordinary bid, one paisa inside the touch
    replay, oms, _ = build(returns={"BUY": (289.00, 50)})
    set_book(replay, bid=288.99, ask=289.02)
    replay._requote(OPEN_MS + 1_000)
    # while it is in flight the offer collapses onto our bid
    set_book(replay, bid=288.50, ask=289.00)
    # it lands -- _arrive refuses it under post-only semantics
    replay._activate_until(OPEN_MS + 2_000)
    # nothing rests
    assert "BUY" not in replay.work
    # and the order manager knows, so the side is free to be quoted again
    assert oms.working_orders("PPL") == []
    # the market comes back, and the next cycle re-quotes rather than waiting
    # on an order that never existed
    set_book(replay, bid=288.99, ask=289.02)
    replay._requote(OPEN_MS + 3_000)
    assert len(replay.pending) == 1


def test_a_price_change_cancels_and_the_replacement_waits_for_the_ack():
    """The default mechanic, and the one every measured result came from."""
    # a resting, acknowledged bid
    replay, oms, mm = build()
    set_book(replay)
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    oid = replay.work["BUY"].oid
    # the strategy now wants a different price
    mm._returns = {"BUY": (288.99, 50)}
    replay._requote(OPEN_MS + 3_000)
    # a cancel is scheduled against THAT order, and nothing else
    assert len(replay.pending) == 1
    t_land, _, action, payload = replay.pending[0]
    assert action == "CANCEL" and payload == ("BUY", oid)
    # the incumbent is still fillable until the cancel lands -- the exposure
    # mm_backtest models and an amendment would remove
    assert replay.work["BUY"].cancel_at == OPEN_MS + 3_100
    # NOTHING further goes out while the cancel is unconfirmed
    replay._requote(OPEN_MS + 3_050)
    assert len(replay.pending) == 1
    # the cancel lands
    replay._activate_until(OPEN_MS + 4_000)
    assert "BUY" not in replay.work
    # and only now does the replacement go
    replay._requote(OPEN_MS + 4_100)
    assert len(replay.pending) == 1
    _, _, action, order = replay.pending[0]
    assert action == "ARRIVE" and order.price == 288.99


def test_an_unchanged_desire_sends_nothing():
    """The no-churn check, which is the whole point of a diff."""
    # a resting, acknowledged bid
    replay, oms, _ = build()
    set_book(replay)
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # the same desire, again and again
    for t in range(3_000, 9_000, 1_000):
        replay._requote(OPEN_MS + t)
    # not one further message
    assert replay.pending == []
    assert replay.stats["n_orders_sent"] == 1


# ---------------------------------------------------------------------------
# the exchange's events reach the order manager
# ---------------------------------------------------------------------------
def test_a_fill_reaches_the_order_manager_at_our_own_price():
    """Position comes from fills, on both sides of the comparison."""
    # a resting bid
    replay, oms, _ = build()
    set_book(replay)
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # 20 of our 50 execute, at a print price that is NOT ours
    replay._fill("BUY", 288.50, 20, OPEN_MS + 2_500, "through")
    # Backtester booked it
    assert replay.pos == 20
    # and so did the production order manager, at OUR limit price
    assert oms.position("PPL") == 20
    working = oms.working_orders("PPL")
    assert len(working) == 1 and working[0].leaves_quantity == 30


def test_a_full_fill_frees_the_side_on_both_sides():
    """A fully consumed order leaves self.work; the order manager must agree."""
    # a resting bid
    replay, oms, _ = build()
    set_book(replay)
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # all 50 execute
    replay._fill("BUY", 289.00, 50, OPEN_MS + 2_500, "through")
    # gone from Backtester
    assert "BUY" not in replay.work
    # and from the order manager, which is now flat of working orders
    assert oms.working_orders("PPL") == []
    assert oms.position("PPL") == 50
    # so the next cycle re-sends the quote
    replay._requote(OPEN_MS + 3_000)
    assert len(replay.pending) == 1


def test_a_landed_cancel_is_reported_and_is_not_mistaken_for_a_fill():
    """An order leaves self.work inside _activate_until for one reason only."""
    # a resting bid
    replay, oms, _ = build()
    set_book(replay)
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # want nothing
    replay._adapter._mm._returns = {}
    replay._requote(OPEN_MS + 3_000)
    # the cancel lands
    replay._activate_until(OPEN_MS + 4_000)
    # both sides agree it is gone, and no fill was invented
    assert "BUY" not in replay.work
    assert oms.working_orders("PPL") == []
    assert oms.position("PPL") == 0


def test_a_cancel_racing_a_fill_becomes_a_cancel_reject_not_a_stuck_order():
    """The order filled before our cancel landed. There is nothing to cancel.

    Without this, the order manager sits in PENDING_CANCEL for the rest of the
    session and never quotes that side again.
    """
    # a resting bid
    replay, oms, _ = build()
    set_book(replay)
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # it fills completely
    replay._fill("BUY", 289.00, 50, OPEN_MS + 2_500, "through")
    # and only then do we decide to cancel something that no longer exists
    from core.model import CancelOrder
    replay._dispatch(CancelOrder(symbol="PPL", cl_ord_id="R-99",
                                 orig_cl_ord_id="R-00000001",
                                 exchange_order_id="1"), OPEN_MS + 3_000)
    # Backtester counts it the way a real exchange cancel-reject counts
    assert replay.stats["stale_cancels_ignored"] == 1


# ---------------------------------------------------------------------------
# the gate's own preconditions
# ---------------------------------------------------------------------------
def _amending_replay(returns, quantity_policy="exact"):
    """A replay harness whose order manager amends instead of cancel+new.

    `quantity_policy` defaults to "exact" -- mm_backtest's rule, which requotes
    on ANY size difference. The order manager's own default is "ignore", under
    which a size-only change produces no message at all, so a test of what an
    amendment does to queue position would silently test nothing.
    """
    # the venue, which is also what supplies the three priority rules
    venue = PSXVenue(session_provider=lambda d: [
        SessionSegment(start_ms=OPEN_MS, end_ms=CLOSE_MS)])
    # a scripted strategy
    mm = StubMM(returns)
    # the production adapter
    adapter = MicroMMAdapter("PPL", venue, mm, reference_price_minor=28900)
    # THE TOGGLE: use_replace=True sends one Change Former Order per reprice
    # instead of a cancel and a new order
    oms = OrderManager(venue=venue,
                       gateway=RiskGateway([OrderQuantityCheck(1_000_000)]),
                       kill_switch=KillSwitch(), session_id="R",
                       account="CLIENT001", use_replace=True,
                       tolerance=QuoteTolerance(
                           quantity_policy=quantity_policy))
    # constant latency, so a test can say precisely when a message lands
    cfg = {"latency_ms": 100, "at_price_mode": "queue",
           "fill_on_crossing_adds": False, "log_equity": False,
           "session_ms": (OPEN_MS, CLOSE_MS)}
    # the harness
    replay = EngineReplay(strategy=mm, adapter=adapter, oms=oms, symbol="PPL",
                          cfg=cfg)
    # the risk gateway's window check needs a date
    replay.session_date = DAY
    # a two-sided book to quote against
    set_book(replay)
    # everything the tests need
    return replay, oms, mm, venue


def test_the_venue_drives_the_engines_priority_rules():
    """The simulated exchange must not have its own opinion.

    Two sets of flags mean the same thing -- the venue's and the engine's --
    and if they ever drift, every fill after a reprice is wrong in the same
    direction with nothing raising. So they are wired, not coincidentally
    equal, and this asserts the wiring rather than the values.
    """
    # a harness built against the PSX venue
    replay, _oms, _mm, venue = _amending_replay({"BUY": (289.00, 50)})
    # each engine flag must equal the venue's answer, not a constant
    assert replay.cfo_price_keeps_priority is venue.replace_price_keeps_priority
    assert replay.cfo_qty_up_keeps_priority is venue.replace_qty_up_keeps_priority
    assert replay.cfo_qty_down_keeps_priority is venue.replace_qty_down_keeps_priority


def test_psx_answers_the_three_priority_questions_as_the_rulebook_does():
    """PSX Regulations 8.5.2, the only place these answers come from."""
    # the venue
    venue = PSXVenue(session_provider=lambda d: None)
    # a price change re-queues at the back of the new level
    assert venue.replace_price_keeps_priority is False
    # so does an increase -- only reduction is carved out
    assert venue.replace_qty_up_keeps_priority is False
    # a reduction is amended in place and keeps its position
    assert venue.replace_qty_down_keeps_priority is True


def test_an_amendment_is_one_message_not_two():
    """The whole point of the toggle: half the wire traffic per reprice."""
    # a resting bid
    replay, _oms, mm, _venue = _amending_replay({"BUY": (289.00, 50)})
    # get it live
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # now want a different price
    mm._returns = {"BUY": (288.99, 50)}
    # one requote cycle
    replay._requote(OPEN_MS + 3_000)
    # ONE message, and it is an amendment rather than a cancel
    assert len(replay.pending) == 1
    assert replay.pending[0][2] == "AMEND"
    # and the order manager did emit a ReplaceOrder to get here
    assert replay.engine_stats["replace_actions"] == 1


def test_the_old_terms_stay_live_until_the_amendment_lands():
    """The real exposure of the single message, and what it costs you."""
    # a resting bid
    replay, _oms, mm, _venue = _amending_replay({"BUY": (289.00, 50)})
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # reprice
    mm._returns = {"BUY": (288.99, 50)}
    replay._requote(OPEN_MS + 3_000)
    # nothing has been cancelled: the OLD price is still resting and still
    # matchable. Under cancel-plus-new the cancel would already be in flight.
    assert replay.work["BUY"].price == 289.00
    assert replay.work["BUY"].cancel_at is None
    # and it is marked in flight, so a second amendment is not stacked on it
    assert replay.work["BUY"].amend_at is not None


def test_a_price_change_loses_priority_and_the_manager_is_told():
    """PSX 8.5.2: a reprice rejoins the back of the queue at the new level."""
    # a resting bid
    replay, oms, mm, _venue = _amending_replay({"BUY": (289.00, 50)})
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # the production order, before anything changes
    order = oms.working_orders("PPL")[0]
    cl_ord_id = order.cl_ord_id
    # put a queue at the price we are moving TO, so losing priority is visible
    replay.book.o["H3"] = mm_backtest.Order("BUY", 288.99, 700)
    # reprice and land it
    mm._returns = {"BUY": (288.99, 50)}
    replay._requote(OPEN_MS + 3_000)
    replay._activate_until(OPEN_MS + 4_000)
    # we joined the back: all 700 are in front of us
    assert sum(replay.work["BUY"].ahead.values()) == 700
    # the engine counted an amendment that did NOT keep its place
    assert replay.stats["n_cfos"] == 1
    assert replay.stats.get("n_cfos_kept_priority", 0) == 0
    # AND THE PRODUCTION SIDE AGREES. The order kept its identity -- it was
    # amended, not cancelled and re-created -- and carries the new terms.
    still = oms.working_orders("PPL")
    assert len(still) == 1
    assert still[0].cl_ord_id == cl_ord_id
    assert still[0].price_minor == 28899


def test_a_size_reduction_keeps_priority():
    """PSX 8.5.2's one carve-out, and the only free amendment."""
    # a resting bid, with a queue in front of it
    replay, oms, mm, _venue = _amending_replay({"BUY": (289.00, 50)})
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # the queue state that must survive
    kept_ahead = dict(replay.work["BUY"].ahead)
    kept_t_active = replay.work["BUY"].t_active
    # SAME price, SMALLER size
    mm._returns = {"BUY": (289.00, 20)}
    replay._requote(OPEN_MS + 3_000)
    replay._activate_until(OPEN_MS + 4_000)
    # the line in front of us is untouched, and so is our join time
    assert replay.work["BUY"].ahead == kept_ahead
    assert replay.work["BUY"].t_active == kept_t_active
    # with the smaller size applied
    assert replay.work["BUY"].qty == 20
    # and counted as keeping its place
    assert replay.stats["n_cfos_kept_priority"] == 1
    # the production order carries the new size too
    assert oms.working_orders("PPL")[0].quantity == 20


def test_a_size_increase_loses_priority():
    """Only reduction is carved out; growing an order re-queues it."""
    # a resting bid
    replay, _oms, mm, _venue = _amending_replay({"BUY": (289.00, 50)})
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # SAME price, BIGGER size
    mm._returns = {"BUY": (289.00, 90)}
    replay._requote(OPEN_MS + 3_000)
    # the queue at our price grows while the message is in flight, so the
    # re-snapshot has to happen when it LANDS, not when it was sent
    replay.book.o["H9"] = mm_backtest.Order("BUY", 289.00, 111)
    replay._activate_until(OPEN_MS + 4_000)
    # 500 from the original queue plus 111 that arrived: we are behind both
    assert sum(replay.work["BUY"].ahead.values()) == 611
    # nothing kept its place
    assert replay.stats.get("n_cfos_kept_priority", 0) == 0


def test_an_amendment_that_loses_a_race_to_a_fill_is_rejected():
    """The exchange answers with an Order Cancel Reject; so must we."""
    # a resting bid
    replay, oms, mm, _venue = _amending_replay({"BUY": (289.00, 50)})
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # decide to reprice
    mm._returns = {"BUY": (288.99, 50)}
    replay._requote(OPEN_MS + 3_000)
    # the order fills completely BEFORE the amendment lands
    replay._fill("BUY", 289.00, 50, OPEN_MS + 3_500, "through")
    # land the now-pointless amendment
    replay._activate_until(OPEN_MS + 4_000)
    # nothing was resurrected, and the rejection was counted rather than silent
    assert "BUY" not in replay.work
    assert replay.stats.get("stale_cfos_ignored") == 1
    # AND THE PRODUCTION ORDER IS FINISHED, not resurrected. The amendment was
    # refused because it lost the race to a fill -- the order is done, and the
    # reject must not put it back into the working set. It did, before
    # 2026-09-17: it came back as PARTIALLY_FILLED with 50 of 50 filled, where
    # the diff would have seen a resting order matching what we wanted and left
    # the side dark for the rest of the session.
    assert not oms.working_orders("PPL")


def test_a_reject_never_resurrects_a_finished_order():
    """The bug the test above found, pinned on its own.

    A cancel or an amendment is most often refused because it LOST A RACE: the
    order filled before the exchange reached our message. Returning it to LIVE
    puts an order the exchange has finished with back in our working set, and
    the diff then leaves that side alone forever.
    """
    # a resting bid
    replay, oms, mm, _venue = _amending_replay({"BUY": (289.00, 50)})
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # the production order, and its id
    order = oms.working_orders("PPL")[0]
    # it fills completely
    replay._fill("BUY", 289.00, 50, OPEN_MS + 2_500, "through")
    # it is finished
    assert order.state.is_terminal
    # now a late reject arrives quoting that order
    oms.on_cancel_rejected(order.cl_ord_id, "too late, already filled")
    # it stays finished
    assert order.state.is_terminal
    # and it is not back in the working set
    assert not oms.working_orders("PPL")


def test_an_amended_order_can_still_be_cancelled():
    """The engine renames the order; the production side does not.

    Backtester gives an amended order a NEW internal id. The order manager
    still knows it by its original client order id. If the two are not kept in
    step, the cancel silently matches nothing and the quote stays in the book.
    """
    # a resting bid
    replay, oms, mm, _venue = _amending_replay({"BUY": (289.00, 50)})
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # amend it once, so the engine id changes
    mm._returns = {"BUY": (288.99, 50)}
    replay._requote(OPEN_MS + 3_000)
    replay._activate_until(OPEN_MS + 4_000)
    # the order survived the amendment on the production side
    assert len(oms.working_orders("PPL")) == 1
    # now pull the side entirely, which is a Cancel and never an amendment
    mm._returns = {}
    replay._requote(OPEN_MS + 5_000)
    # a cancel was actually scheduled -- not silently dropped
    assert len(replay.pending) == 1
    assert replay.pending[0][2] == "CANCEL"
    # and it lands, removing the quote
    replay._activate_until(OPEN_MS + 6_000)
    assert "BUY" not in replay.work


def test_a_halt_pulls_the_quote_through_the_production_path():
    """Backtester's gate, translated into a phase the adapter understands."""
    # a resting bid
    replay, oms, _ = build()
    set_book(replay)
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # the market halts
    set_book(replay, phase="TRADING_BREAK")
    replay._requote(OPEN_MS + 3_000)
    # the quote is pulled -- by the ordinary diff, not by a special path
    assert len(replay.pending) == 1
    _, _, action, payload = replay.pending[0]
    assert action == "CANCEL"
    # and Backtester's own counter agrees this was a halted cycle
    assert replay.stats["halted_requotes"] == 1


def test_a_one_sided_book_produces_no_messages():
    """Normal at the open and on thin names. Not an error, and not a quote."""
    # a book with no offer
    replay, oms, _ = build()
    set_book(replay)
    del replay.book.o["H2"]
    replay._requote(OPEN_MS + 1_000)
    # nothing sent, and the harness says why
    assert replay.pending == []
    assert replay.engine_stats["skipped_one_sided"] == 1


def test_the_book_reaches_the_strategy_in_paisa_without_a_rounding_error():
    """The harness and the adapter must round identically.

    If they did not, the reconcile gate would fail on prices rather than on
    logic, and the difference would look like an order-manager bug.
    """
    # a price where truncation and rounding disagree
    replay, oms, _ = build()
    set_book(replay, bid=280.03, ask=280.10)
    # the snapshot the production stack sees
    book = replay._snapshot(OPEN_MS + 1_000)
    # rounded, not truncated
    assert book.bids[0].price_minor == 28003
    assert book.asks[0].price_minor == 28010


# ---------------------------------------------------------------------------
# the three stand-downs the harness has to share with the backtest
# ---------------------------------------------------------------------------
# WHY THESE EXIST. The reconcile gate asks one question: does the production
# engine reproduce mm_backtest? That question is only answerable if the two
# stand down in the same places. Backtester grew two guards after this harness
# was written -- the stale-feed guard and the crossed-book guard -- and both
# live in `_requote`, the one method the harness replaces, so neither was
# inherited. A third case, the one-sided book, was never shared at all. In each
# case the backtest CANCELS and the harness left the quotes resting, so the
# gate would have reported a P&L difference that was this omission rather than
# the order manager.


def _resting_bid():
    """A harness with one of our bids actually resting on the book."""
    # the harness and its order manager
    replay, oms, _ = build()
    # an ordinary two-sided book in continuous trading
    set_book(replay)
    # ask for a quote and let it land
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # nothing in flight, so the next requote's messages are unambiguous
    replay.pending.clear()
    # ready to be disturbed
    return replay, oms


def test_a_stale_feed_pulls_the_quote():
    """Nothing has arrived for longer than a working feed ever goes quiet.

    Backtester stands down and stays dark until a snapshot restores the book.
    The DETECTION lives in Backtester.run and is inherited; only the guard was
    missing here.
    """
    # a resting bid
    replay, _ = _resting_bid()
    # the flag Backtester.run maintains, set directly so this case tests the
    # GUARD alone rather than the detector
    replay._feed_stale = True
    # the next cycle
    replay._requote(OPEN_MS + 3_000)
    # the quote is pulled, through the ordinary diff
    assert len(replay.pending) == 1
    _, _, action, _ = replay.pending[0]
    assert action == "CANCEL"
    # and counted the way Backtester counts it
    assert replay.stats["stale_feed_requotes"] == 1
    assert replay.stats["halted_requotes"] == 1


def test_a_crossed_book_pulls_the_quote():
    """A bid at or above the ask cannot rest at a continuously matching
    exchange, so the reconstruction is momentarily wrong and a mid computed
    from it is not a price. Backtester pulls; so must this.
    """
    # a resting bid
    replay, _ = _resting_bid()
    # the book crosses under us
    set_book(replay, bid=289.05, ask=289.00)
    # the next cycle
    replay._requote(OPEN_MS + 3_000)
    # pulled
    assert len(replay.pending) == 1
    _, _, action, _ = replay.pending[0]
    assert action == "CANCEL"
    # and counted under Backtester's own name for it
    assert replay.stats["crossed_book_requotes"] == 1


def test_a_one_sided_book_pulls_a_resting_quote():
    """Measured, not assumed.

    micro_mm.quotes opens with `if bb is None or ba is None or bq <= 0 or
    aq <= 0: return {}` -- the identical test _snapshot uses -- and an empty
    desire diffs to a full cancel in Backtester. So the backtest pulls its
    quotes on a one-sided book, and this harness used to leave them resting.
    """
    # a resting bid
    replay, _ = _resting_bid()
    # the offer disappears
    del replay.book.o["H2"]
    # the next cycle
    replay._requote(OPEN_MS + 3_000)
    # pulled
    assert len(replay.pending) == 1
    _, _, action, _ = replay.pending[0]
    assert action == "CANCEL"
    # NOT counted as a halt: Backtester does not count one here, because there
    # the strategy declines rather than the gate refusing
    assert replay.stats["halted_requotes"] == 0


if __name__ == "__main__":
    # A pytest file is not a script: there is no runner, and the project root is
    # not on the import path. Say so rather than failing with ModuleNotFoundError.
    raise SystemExit(
        "This is a pytest file, not a script.\n"
        "Run the suite from the Production directory, with mm_backtest on the "
        "path:\n"
        "    PYTHONPATH=/path/to/existing_mm_live python -m pytest -q")
