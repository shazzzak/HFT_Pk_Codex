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
from core.oms import OrderManager
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
def test_an_amendment_stops_the_run_rather_than_quietly_diverging():
    """mm_backtest has no amendment path, so a run using one is not comparable."""
    # a manager configured to amend
    venue = PSXVenue(session_provider=lambda d: [
        SessionSegment(start_ms=OPEN_MS, end_ms=CLOSE_MS)])
    mm = StubMM({"BUY": (289.00, 50)})
    adapter = MicroMMAdapter("PPL", venue, mm, reference_price_minor=28900)
    oms = OrderManager(venue=venue,
                       gateway=RiskGateway([OrderQuantityCheck(1_000_000)]),
                       kill_switch=KillSwitch(), session_id="R",
                       account="CLIENT001", use_replace=True)
    cfg = {"latency_ms": 100, "at_price_mode": "queue",
           "fill_on_crossing_adds": False, "log_equity": False,
           "session_ms": (OPEN_MS, CLOSE_MS)}
    replay = EngineReplay(strategy=mm, adapter=adapter, oms=oms, symbol="PPL",
                          cfg=cfg)
    replay.session_date = DAY
    set_book(replay)
    # rest an order
    replay._requote(OPEN_MS + 1_000)
    replay._activate_until(OPEN_MS + 2_000)
    # move the price, which now produces a ReplaceOrder
    mm._returns = {"BUY": (288.99, 50)}
    # the harness refuses rather than dropping the message
    with pytest.raises(ValueError, match="no amendment path"):
        replay._requote(OPEN_MS + 3_000)


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


if __name__ == "__main__":
    # A pytest file is not a script: there is no runner, and the project root is
    # not on the import path. Say so rather than failing with ModuleNotFoundError.
    raise SystemExit(
        "This is a pytest file, not a script.\n"
        "Run the suite from the Production directory, with mm_backtest on the "
        "path:\n"
        "    PYTHONPATH=/path/to/existing_mm_live python -m pytest -q")
