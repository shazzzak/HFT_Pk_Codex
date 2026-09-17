"""Tests for the in-flight order rule in mm_backtest.

WHAT IS UNDER TEST, AND WHY IT EXISTS.

The reconcile gate measured mm_backtest sending 6,833 new orders across four
symbol-days where the production engine sent 1,409. 4,901 of the difference
were DUPLICATES: mm_backtest wrote an order into self.work when it LANDED at
the exchange, not when it was SENT, so for one network latency after every
send the side read as empty and the next requote sent another one. When those
duplicates landed they overwrote whatever was resting -- 5,205 orders that were
never cancelled, never filled, and never closed out.

The fix reserves the side at SEND time and holds it until the exchange answers.
These tests assert the four properties that has to have, against the REAL
Backtester rather than a mock of it:

  1. Two requote cycles inside one latency window send ONE order.
  2. An order that has not landed cannot be filled.
  3. A reprice cancels and waits: no replacement goes out until the cancel has
     landed and freed the side.
  4. A rejected crossing order RELEASES the side. Get this wrong and a symbol
     goes dark for the rest of the session with no error anywhere.
"""
# the test runner
import pytest

# skip the whole module cleanly where the research tree is not importable
mm_backtest = pytest.importorskip(
    "mm_backtest", reason="mm_backtest must be on PYTHONPATH for these tests")

# the engine, its order struct, and the latency model
from mm_backtest import Backtester, LatencyModel, MyOrder, Order


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
# a session window wide enough that nothing here runs outside it
OPEN_MS = 9 * 3_600_000
CLOSE_MS = 15 * 3_600_000 + 30 * 60_000


class FixedQuoteMM:
    """A strategy that always wants the same quote, so the test controls churn.

    Backtester._requote asks exactly one thing of a strategy -- quotes(...) --
    so this is the whole interface.
    """

    def __init__(self, want=None):
        # what to return every cycle; None on a side means no quote there
        self.want = want if want is not None else {"BUY": (289.00, 50)}

    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # a fresh dict each call, because _requote rewrites it in place when
        # it clamps prices to the circuit band
        return dict(self.want)


def build(use_cfo=False, want=None, latency_ms=100.0):
    """A real Backtester with constant latency, ready to drive by hand."""
    # the strategy
    mm = FixedQuoteMM(want)
    # CONSTANT latency both legs, so a test can say exactly when a message
    # lands rather than probabilistically
    cfg = {"latency_model": LatencyModel(decision_ms=0.0,
                                         wire_out_median_ms=latency_ms,
                                         wire_out_tail_ms=0.0,
                                         wire_in_median_ms=latency_ms,
                                         wire_in_tail_ms=0.0, tail_prob=0.0),
           "at_price_mode": "queue", "fill_on_crossing_adds": False,
           "log_equity": False, "session": (OPEN_MS, CLOSE_MS),
           "use_cfo": use_cfo, "fee_per_share": 0.0}
    # the engine
    bt = Backtester(strategy=mm, cfg=cfg)
    # a two-sided book in continuous trading
    set_book(bt)
    # ready
    return bt


def set_book(bt, bid=289.00, bid_qty=500, ask=289.50, ask_qty=300,
             phase="CONTINUOUS_AUCTION"):
    """Put a two-sided book into the engine's Book object.

    The default ask is deliberately well ABOVE the default quote price, so a
    bid at 289.00 rests rather than being rejected as crossing.
    """
    # the engine's book
    book = bt.book
    # clear anything previously resting
    book.o.clear()
    # one historical order each side, keyed by id
    book.o["H1"] = Order("BUY", bid, bid_qty)
    book.o["H2"] = Order("SELL", ask, ask_qty)
    # the phase the requote gate reads
    book.phase = phase
    # no circuit limits published
    book.limit_up = None
    book.limit_dn = None


class FakeTrade:
    """The attributes _on_market_trade reads off a trade row."""

    def __init__(self, ts_exch, price, qty, aggressor_side,
                 rest_oid=None, initiator="BUYER"):
        self.ts_exch = ts_exch
        self.price = price
        self.qty = qty
        self.aggressor_side = aggressor_side
        self.rest_oid = rest_oid
        # anything other than "AUCTION": an auction print has no continuous
        # aggressor and the handler skips it
        self.initiator = initiator


# ---------------------------------------------------------------------------
# 1. the duplicate is gone
# ---------------------------------------------------------------------------
def test_two_cycles_inside_one_latency_window_send_one_order():
    """THE DEFECT, stated as a test. This failed before the fix."""
    # an engine wanting a bid, 100 ms each way
    bt = build()
    # first cycle: the order goes out
    bt._requote(OPEN_MS + 1_000)
    # exactly one message in the air, landing at +1_100
    assert len(bt.pending) == 1
    assert bt.stats["n_orders_sent"] == 1
    # the side is RESERVED, which is the fix -- the old code left this empty
    assert bt.work.get("BUY") is not None
    # and the reservation is marked as not yet live
    assert bt.work["BUY"].t_active is None
    # three more cycles, all inside the latency window
    for dt in (1_010, 1_050, 1_099):
        bt._requote(OPEN_MS + dt)
    # STILL one message and one order. The old code sent four.
    assert len(bt.pending) == 1, "a duplicate went out"
    assert bt.stats["n_orders_sent"] == 1
    # the duplicate counter agrees
    assert bt.stats["orders_sent_while_new_in_flight"] == 0
    # and the three cycles are accounted for as blocked, not as nothing
    assert bt.stats["requotes_blocked_in_flight"] == 3


def test_the_order_becomes_live_when_it_lands_and_quoting_resumes():
    """The hold is temporary: once the exchange answers, normal service."""
    # send one
    bt = build()
    bt._requote(OPEN_MS + 1_000)
    # land it
    bt._activate_until(OPEN_MS + 2_000)
    # it is live now, with a real activation time and a queue snapshot
    o = bt.work["BUY"]
    assert o.t_active == OPEN_MS + 1_100
    assert isinstance(o.ahead, dict)
    # a further cycle wanting the SAME quote sends nothing, via the no-churn
    # path rather than the in-flight path
    before = bt.stats["requotes_blocked_in_flight"]
    bt._requote(OPEN_MS + 2_100)
    assert bt.stats["n_orders_sent"] == 1
    assert bt.stats["requotes_blocked_in_flight"] == before


# ---------------------------------------------------------------------------
# 2. an order in the air cannot be filled
# ---------------------------------------------------------------------------
def test_an_order_still_on_the_wire_cannot_be_filled():
    """Reserving the side must not make the order matchable early.

    This is the risk the fix introduces and the reason t_active is a None
    rather than a flag: a reservation that could be filled would invent free
    money, and it would do it silently.
    """
    # send a bid at 289.00, still in the air
    bt = build()
    bt._requote(OPEN_MS + 1_000)
    # the order is on the side but not live
    assert bt.work["BUY"].t_active is None
    # a trade prints BELOW our bid -- a through-print, which would certainly
    # fill a resting order -- while ours is still on the wire
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 1_050, price=288.00,
                                  qty=100, aggressor_side="SELL"))
    # nothing filled, and no position
    assert bt.fills == []
    assert bt.pos == 0
    # now let it land
    bt._activate_until(OPEN_MS + 2_000)
    # the SAME print after it is live does fill it
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 2_100, price=288.00,
                                  qty=100, aggressor_side="SELL"))
    # filled at OUR price, for our whole size
    assert len(bt.fills) == 1
    assert bt.fills[0]["px"] == 289.00
    assert bt.pos == 50


# ---------------------------------------------------------------------------
# 3. a reprice cancels and WAITS
# ---------------------------------------------------------------------------
def test_a_reprice_cancels_and_does_not_replace_in_the_same_cycle():
    """The second orphan source: cancel and replacement raced each other.

    Both were sent together with independent latency draws, so whenever the
    replacement's draw was shorter it landed first and overwrote an order that
    was still resting. Now the replacement waits for the side to be freed.
    """
    # a resting bid at 289.00
    bt = build()
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    assert bt.work["BUY"].t_active is not None
    # one order sent so far
    assert bt.stats["n_orders_sent"] == 1
    # the strategy now wants a different price
    bt.strat.want = {"BUY": (289.10, 50)}
    # one cycle: this should CANCEL and send nothing else
    bt._requote(OPEN_MS + 2_100)
    # still one order sent -- no replacement went out beside the cancel
    assert bt.stats["n_orders_sent"] == 1, "the replacement did not wait"
    # the incumbent has a cancel in flight
    assert bt.work["BUY"].cancel_at == OPEN_MS + 2_200
    # a cycle while the cancel is in flight sends nothing either
    bt._requote(OPEN_MS + 2_150)
    assert bt.stats["n_orders_sent"] == 1
    # let the cancel land: the side is freed and the order is closed out
    bt._activate_until(OPEN_MS + 3_000)
    assert bt.work.get("BUY") is None
    assert bt.stats["n_cancels"] == 1
    # NOTHING WAS ORPHANED: the cancelled order has an end reason in the
    # lifecycle log, which is exactly what an overwritten order never got
    assert bt._olog[1]["end_reason"] == "cancelled"
    # and now the replacement goes out
    bt._requote(OPEN_MS + 3_100)
    assert bt.stats["n_orders_sent"] == 2
    assert bt.work["BUY"].price == 289.10
    # no orphans anywhere in the sequence
    assert bt.stats["orders_orphaned_by_overwrite"] == 0
    assert bt.stats["stale_arrivals_ignored"] == 0


def test_an_amendment_still_reprices_in_one_message():
    """With CFO on, the reprice path above is not used and is unaffected."""
    # a resting bid, CFO enabled
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    # the strategy wants a new price
    bt.strat.want = {"BUY": (289.10, 50)}
    # one cycle
    bt._requote(OPEN_MS + 2_100)
    # ONE message and NO cancel: the amendment does not pull the order
    assert bt.stats["n_orders_sent"] == 2
    assert bt.work["BUY"].cancel_at is None
    # it is marked as having an amendment on the wire
    assert bt.work["BUY"].amend_at == OPEN_MS + 2_200
    # the OLD terms are still what is resting until it lands, which is the
    # real exposure of an amendment
    assert bt.work["BUY"].price == 289.00
    # land it: the CFO is counted at the exchange, not at the send
    bt._activate_until(OPEN_MS + 3_000)
    assert bt.stats["n_cfos"] == 1
    # now the new terms are in place, with no order orphaned on the way
    assert bt.work["BUY"].price == 289.10
    assert bt.stats["orders_orphaned_by_overwrite"] == 0


# ---------------------------------------------------------------------------
# 4. a rejected order releases the side
# ---------------------------------------------------------------------------
def test_a_crossing_order_that_is_rejected_frees_the_side():
    """THE DANGEROUS FAILURE MODE OF THIS FIX.

    The side is reserved at send time. If a reservation is not released when
    the order is rejected at arrival, that side is occupied by something that
    will never come live and never be cancelled -- the in-flight rule then
    blocks every later cycle and the symbol goes dark for the rest of the
    session, with no error raised anywhere. This test is the guard on that.
    """
    # an engine quoting a bid at 289.00
    bt = build()
    bt._requote(OPEN_MS + 1_000)
    # the side is reserved
    assert bt.work.get("BUY") is not None
    # THE MARKET MOVES DOWN through our price while the order is on the wire,
    # so on arrival our bid would be marketable and is rejected post-only
    set_book(bt, bid=288.00, ask=288.50)
    # land it
    bt._activate_until(OPEN_MS + 2_000)
    # rejected, and counted
    assert bt.stats["rejected_crossing"] == 1
    # AND THE SIDE IS FREE AGAIN
    assert bt.work.get("BUY") is None
    # so the next cycle quotes normally rather than being blocked forever
    bt._requote(OPEN_MS + 2_100)
    assert bt.stats["n_orders_sent"] == 2


def test_a_full_fill_frees_the_side_and_the_next_cycle_requotes():
    """The ordinary path out: the order fills, the side is free, we requote."""
    # a resting bid
    bt = build()
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    # a through-print takes the whole order
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 2_100, price=288.00,
                                  qty=500, aggressor_side="SELL"))
    # fully filled, so the side is empty
    assert bt.pos == 50
    assert bt.work.get("BUY") is None
    # and the next cycle puts a fresh quote out
    bt._requote(OPEN_MS + 2_200)
    assert bt.stats["n_orders_sent"] == 2
    # nothing orphaned or stale anywhere
    assert bt.stats["orders_orphaned_by_overwrite"] == 0
    assert bt.stats["stale_arrivals_ignored"] == 0


# ---------------------------------------------------------------------------
# 5. a halt does not try to cancel something still on the wire
# ---------------------------------------------------------------------------
def test_a_halt_leaves_an_in_flight_order_alone_then_pulls_it():
    """There is no OrderID to put in a cancel for an order still on the wire.

    So the halt path skips it, and pulls it on the cycle after it lands.
    """
    # an order goes out
    bt = build()
    bt._requote(OPEN_MS + 1_000)
    # the market halts while it is in the air
    bt.book.phase = "TEMPORARY_SUSPENSION"
    # a requote during the halt
    bt._requote(OPEN_MS + 1_050)
    # the halt was counted, but no cancel was sent for the in-flight order
    assert bt.stats["halted_requotes"] == 1
    assert bt.work["BUY"].cancel_at is None
    # the order lands
    bt._activate_until(OPEN_MS + 2_000)
    # the next requote, still halted, now pulls it
    bt._requote(OPEN_MS + 2_100)
    assert bt.work["BUY"].cancel_at == OPEN_MS + 2_200
    # and it is cancelled when that lands
    bt._activate_until(OPEN_MS + 3_000)
    assert bt.work.get("BUY") is None
    assert bt.stats["n_cancels"] == 1


if __name__ == "__main__":
    raise SystemExit(
        "This is a pytest file, not a script.\n"
        "    PYTHONPATH=/path/to/existing_mm_live python -m pytest -q")


# ---------------------------------------------------------------------------
# 6. THE CRASH THE GATE HIT: a fill must not erase a message in flight
# ---------------------------------------------------------------------------
# WHAT HAPPENED. From the run of 2026-09-17:
#
#   NRL 2026-06-24 ERROR InvalidTransition('order GATE-00001068 (NRL SELL)
#                        is PENDING_CANCEL; cannot apply replaced')
#
# Two of six symbol-days died this way, and it is production code -- core/
# model.py and core/oms.py -- so in live trading it raises mid-session on the
# most active name on the book.
#
# THE SEQUENCE. An amendment goes out, so the order is PENDING_REPLACE. The
# OLD terms are still resting and still matchable -- that is the whole point
# of an amendment and the real exposure of the one message. A fill arrives on
# them, and Order.on_fill overwrites `state` with PARTIALLY_FILLED, which is
# correct for the lifecycle and destroys the only record that a message was on
# the wire. The order manager then reads a quiescent order and sends a cancel.
# When the amendment is finally confirmed, the order is PENDING_CANCEL and the
# transition raises.
#
# THE FIX separates the two facts: `state` is the lifecycle, and
# replace_in_flight / cancel_in_flight are the messages. A fill touches the
# first and cannot touch the second.
# ALIASED. mm_backtest.Order (a resting order in the historical book) is
# imported at the top of this file and the book helper builds them; importing
# core.model.Order under its own name would shadow it and every book in this
# module would fail to build. The tests caught it immediately, which is the
# argument for having them.
from core.model import InvalidTransition, OrderState, Side
from core.model import Order as EngineOrder


def _live_order(qty=100):
    """An order acknowledged by the exchange and resting."""
    # a plain order
    o = EngineOrder(cl_ord_id="C1", symbol="NRL", side=Side.SELL,
                    price_minor=28900, quantity=qty)
    # sent
    o.state = OrderState.PENDING_NEW
    # and acknowledged, which is what gives it an exchange id
    o.on_ack("EX1")
    # resting
    return o


def test_a_fill_does_not_erase_an_amendment_on_the_wire():
    """THE CRASH, stated as a test. This is the exact sequence that failed."""
    # a resting order
    o = _live_order()
    # we send an amendment: the order manager must now leave the side alone
    o.on_replace_sent()
    assert o.has_message_in_flight
    # A FILL ARRIVES ON THE OLD TERMS, which are still live
    o.on_fill(40)
    # the lifecycle moved, exactly as it should
    assert o.state is OrderState.PARTIALLY_FILLED
    # AND THE MESSAGE IS STILL ON THE WIRE. This is what was being lost.
    assert o.replace_in_flight is True
    assert o.has_message_in_flight is True
    # so the amendment can still be confirmed, without raising
    o.on_replaced(28950, 60)
    # the new terms are live and the flag is down
    assert o.price_minor == 28950
    assert o.state is OrderState.LIVE
    assert o.has_message_in_flight is False


def test_an_amendment_and_a_cancel_can_both_be_outstanding():
    """Both messages at once, which is how the crash actually arose."""
    # a resting order with an amendment on the wire
    o = _live_order()
    o.on_replace_sent()
    # a fill lands on the old terms
    o.on_fill(40)
    # the manager sends a cancel -- legal, the order is partially filled
    o.on_cancel_sent()
    # both messages are outstanding
    assert o.replace_in_flight and o.cancel_in_flight
    assert o.state is OrderState.PENDING_CANCEL
    # the amendment is confirmed FIRST
    o.on_replaced(28950, 60)
    # THE PENDING CANCEL SURVIVES. Returning the order to LIVE here would lose
    # it, and the cancel confirmation would then raise in its turn.
    assert o.state is OrderState.PENDING_CANCEL
    assert o.cancel_in_flight is True
    assert o.replace_in_flight is False
    # and the cancel confirmation lands cleanly
    o.on_cancelled()
    assert o.state is OrderState.CANCELLED
    assert o.has_message_in_flight is False


def test_an_amendment_confirmed_after_a_full_fill_does_not_resurrect_it():
    """The other race: the order finished while the message was on the wire."""
    # a resting order with an amendment out
    o = _live_order(qty=100)
    o.on_replace_sent()
    # it fills completely
    o.on_fill(100)
    assert o.state is OrderState.FILLED
    # the amendment is confirmed after the fact
    o.on_replaced(28950, 60)
    # NOTHING WAS APPLIED and the order stays finished. Putting it back to LIVE
    # would return an order the exchange is done with to the working set, where
    # the diff sees a resting order matching what it wants and leaves it alone
    # -- for the rest of the session, silently.
    assert o.state is OrderState.FILLED
    assert o.price_minor == 28900
    # but the message is answered, so nothing holds the side
    assert o.has_message_in_flight is False


def test_a_confirmation_with_no_amendment_outstanding_still_raises():
    """The fix must not turn a genuine impossibility into a silent no-op."""
    # a resting order with nothing on the wire
    o = _live_order()
    # a replace confirmation out of nowhere is a real inconsistency
    try:
        o.on_replaced(28950, 60)
    except InvalidTransition:
        # which is what should happen
        return
    # reaching here means the guard was lost
    raise AssertionError("on_replaced did not raise with no amendment out")


# ---------------------------------------------------------------------------
# 7. AN AMENDMENT ON THE WIRE HOLDS THE SIDE TOO
# ---------------------------------------------------------------------------
# FOUND BY sim/diff_fills.py ON NRL 2026-06-30. The two sides filled 51 times
# each and still differed by 145.74 PKR, because the engine bought 0.1124
# cheaper and sold 0.0426 dearer -- 0.155 a share on the round trip.
#
# The mechanism was one rule. When a reprice arrives while a CFO is already
# outstanding, mm_backtest could not send a second CFO (amend_at blocks that,
# correctly) and so fell through to the cancel path and PULLED THE QUOTE. The
# production order manager does nothing at all: an order with any message
# outstanding is left exactly where it is until the exchange answers.
#
# At 1782807500800 that cost a real fill -- the engine sold 10 shares at
# 364.34, over two rupees above the day's mid, off an order it had left
# resting; mm_backtest had cancelled its own and had nothing there.
def test_an_amendment_on_the_wire_holds_the_side():
    """A reprice during an outstanding CFO must send NOTHING, not a cancel."""
    # a resting bid, CFO enabled so a reprice is one message
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    # one order sent, resting, nothing outstanding
    assert bt.stats["n_orders_sent"] == 1
    assert bt.work["BUY"].amend_at is None
    # the strategy reprices: one amendment goes out
    bt.strat.want = {"BUY": (289.10, 50)}
    bt._requote(OPEN_MS + 2_100)
    assert bt.stats["n_orders_sent"] == 2
    # the amendment is on the wire and the OLD terms are still resting
    assert bt.work["BUY"].amend_at == OPEN_MS + 2_200
    assert bt.work["BUY"].price == 289.00
    # THE STRATEGY REPRICES AGAIN while that amendment is unanswered
    bt.strat.want = {"BUY": (289.20, 50)}
    bt._requote(OPEN_MS + 2_150)
    # NOTHING WENT OUT -- no second amendment, and crucially NO CANCEL
    assert bt.stats["n_orders_sent"] == 2
    assert bt.work["BUY"].cancel_at is None, "the quote was pulled"
    # the order is still resting and still fillable at the old terms
    assert bt.work["BUY"].t_active is not None
    # and the cycle is accounted for
    assert bt.stats["requotes_blocked_in_flight"] >= 1
    # when the amendment lands the side is free again and quoting resumes
    bt._activate_until(OPEN_MS + 3_000)
    assert bt.work["BUY"].amend_at is None
    bt._requote(OPEN_MS + 3_100)
    assert bt.stats["n_orders_sent"] == 3


def test_an_order_still_fills_while_its_amendment_is_on_the_wire():
    """Holding the side must not stop the OLD terms being hit.

    That exposure is the entire cost of a one-message reprice: until the
    amendment lands, the order the exchange holds is the old one.
    """
    # a resting bid with an amendment outstanding
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    bt.strat.want = {"BUY": (289.10, 50)}
    bt._requote(OPEN_MS + 2_100)
    assert bt.work["BUY"].amend_at is not None
    # a through-print hits the OLD price while the amendment is in the air
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 2_150, price=288.00,
                                  qty=100, aggressor_side="SELL"))
    # filled, at the OLD terms, which is exactly right
    assert len(bt.fills) == 1
    assert bt.fills[0]["px"] == 289.00
    assert bt.pos == 50


def test_a_refused_amendment_does_not_block_the_side_for_ever():
    """A held flag that is never lowered is worse than the bug it prevents.

    If the amendment's target has gone, nothing clears amend_at on the order
    the side is holding -- so this checks the side comes back rather than going
    quiet for the rest of the session.
    """
    # a resting bid with an amendment outstanding
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    bt.strat.want = {"BUY": (289.10, 50)}
    bt._requote(OPEN_MS + 2_100)
    # the order is FULLY FILLED before the amendment lands, so the amendment
    # has nothing left to modify
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 2_150, price=288.00,
                                  qty=500, aggressor_side="SELL"))
    # the side is empty: a fully filled order leaves self.work
    assert bt.work.get("BUY") is None
    # the amendment lands and is refused as stale
    bt._activate_until(OPEN_MS + 3_000)
    assert bt.stats["stale_cfos_ignored"] == 1
    # AND THE SIDE QUOTES AGAIN. Nothing is stuck holding it.
    bt._requote(OPEN_MS + 3_100)
    assert bt.work.get("BUY") is not None


# ---------------------------------------------------------------------------
# 8. A PARTIALLY FILLED ORDER HOLDS NOTHING UP
# ---------------------------------------------------------------------------
# I claimed the second-order (queue_preserving) policy was quiet because
# "holding two orders per side gives it more messages to wait on", and pointed
# at the whole-side block in _plan_side_multi. That was wrong, and wrong in the
# one case the policy exists for: after a partial fill the first order is
# ACKNOWLEDGED and PARTIALLY_FILLED, it has no message outstanding, and the
# block does not fire. The top-up goes out on the very next cycle.
#
# This test exists so that claim cannot be made again without failing.
from core.model import DesiredQuotes, Fill, PlaceOrder, QuoteIntent
from core.oms import OrderManager, QuoteTolerance
from core.risk import KillSwitch, OrderQuantityCheck, RiskGateway
from core.venue import SessionSegment
from venues.psx import PSXVenue

# the symbol and date these use
SYM, DATE = "NRL", "2026-06-30"


def _queue_preserving_oms():
    """An order manager on the second-order policy, with a permissive gateway."""
    # a venue with one all-day session, so the window check never bites
    venue = PSXVenue(session_provider=lambda d: [
        SessionSegment(start_ms=0, end_ms=86_400_000)])
    # the manager under test
    return OrderManager(
        venue=venue,
        gateway=RiskGateway([OrderQuantityCheck(max_quantity=1_000_000)]),
        kill_switch=KillSwitch(), session_id="T", account="A",
        tolerance=QuoteTolerance(quantity_policy="queue_preserving"))


def test_a_partial_fill_does_not_block_the_top_up():
    """THE CORRECTION, as a test. Clip 500, 400 fills, 400 goes back out."""
    # a manager wanting 500 on the bid
    oms = _queue_preserving_oms()
    oms.set_desired(DesiredQuotes(
        symbol=SYM, bid=QuoteIntent(side=Side.BUY, price_minor=36100,
                                    quantity=500)))
    # cycle one places the whole clip
    acts = oms.reconcile(1_000, DATE, {SYM: 36100})
    assert len(acts) == 1 and acts[0].quantity == 500
    # the exchange acknowledges it
    oms.on_ack(acts[0].cl_ord_id, "EX1")
    # 400 of the 500 fills, leaving 100 resting with its queue position
    oms.on_fill(Fill(cl_ord_id=acts[0].cl_ord_id, symbol=SYM, side=Side.BUY,
                     price_minor=36100, quantity=400, timestamp_ms=1_100))
    # the order is acknowledged and partially filled
    order = oms._orders[acts[0].cl_ord_id]
    assert order.state is OrderState.PARTIALLY_FILLED
    assert order.leaves_quantity == 100
    # AND IT HOLDS NOTHING UP. This is the line the wrong claim turned on.
    assert order.has_message_in_flight is False
    # the counters before the next cycle
    before = dict(oms.plan_counts)
    # cycle two: the top-up
    acts2 = oms.reconcile(2_000, DATE, {SYM: 36100})
    # ONE NEW ORDER FOR THE SHORTFALL ONLY -- not 500, and not an amendment.
    # The resting 100 is untouched and keeps the place it earned.
    assert len(acts2) == 1
    assert isinstance(acts2[0], PlaceOrder)
    assert acts2[0].quantity == 400
    assert acts2[0].price_minor == 36100
    # and the side was NOT held for any reason
    assert oms.plan_counts["held_message_in_flight"] == \
        before["held_message_in_flight"]
    assert oms.plan_counts["acted"] == before["acted"] + 1


def test_the_side_is_held_only_while_a_message_is_actually_outstanding():
    """Where the block DOES bite, so the counter can be read correctly."""
    # the same manager, with 500 resting after a partial fill
    oms = _queue_preserving_oms()
    oms.set_desired(DesiredQuotes(
        symbol=SYM, bid=QuoteIntent(side=Side.BUY, price_minor=36100,
                                    quantity=500)))
    acts = oms.reconcile(1_000, DATE, {SYM: 36100})
    oms.on_ack(acts[0].cl_ord_id, "EX1")
    oms.on_fill(Fill(cl_ord_id=acts[0].cl_ord_id, symbol=SYM, side=Side.BUY,
                     price_minor=36100, quantity=400, timestamp_ms=1_100))
    # the top-up goes out and is NOT yet acknowledged
    top_up = oms.reconcile(2_000, DATE, {SYM: 36100})[0]
    # the counters before the next cycle
    before = dict(oms.plan_counts)
    # NOW the side is held: one of its two orders has a message outstanding
    assert oms.reconcile(2_100, DATE, {SYM: 36100}) == []
    assert oms.plan_counts["held_message_in_flight"] == \
        before["held_message_in_flight"] + 1
    # once the exchange answers, the side is free again
    oms.on_ack(top_up.cl_ord_id, "EX2")
    # and with 500 resting against 500 wanted there is simply nothing to do
    before = dict(oms.plan_counts)
    assert oms.reconcile(3_000, DATE, {SYM: 36100}) == []
    # counted as no_change, NOT as held -- the distinction the gate now prints
    assert oms.plan_counts["no_change"] == before["no_change"] + 1
    assert oms.plan_counts["held_message_in_flight"] == \
        before["held_message_in_flight"]
