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
    assert bt._lead("BUY") is not None
    # and the reservation is marked as not yet live
    assert bt._lead("BUY").t_active is None
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
    o = bt._lead("BUY")
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
    assert bt._lead("BUY").t_active is None
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
    assert bt._lead("BUY").t_active is not None
    # one order sent so far
    assert bt.stats["n_orders_sent"] == 1
    # the strategy now wants a different price
    bt.strat.want = {"BUY": (289.10, 50)}
    # one cycle: this should CANCEL and send nothing else
    bt._requote(OPEN_MS + 2_100)
    # still one order sent -- no replacement went out beside the cancel
    assert bt.stats["n_orders_sent"] == 1, "the replacement did not wait"
    # the incumbent has a cancel in flight
    assert bt._lead("BUY").cancel_at == OPEN_MS + 2_200
    # a cycle while the cancel is in flight sends nothing either
    bt._requote(OPEN_MS + 2_150)
    assert bt.stats["n_orders_sent"] == 1
    # let the cancel land: the side is freed and the order is closed out
    bt._activate_until(OPEN_MS + 3_000)
    assert bt._lead("BUY") is None
    assert bt.stats["n_cancels"] == 1
    # NOTHING WAS ORPHANED: the cancelled order has an end reason in the
    # lifecycle log, which is exactly what an overwritten order never got
    assert bt._olog[1]["end_reason"] == "cancelled"
    # and now the replacement goes out
    bt._requote(OPEN_MS + 3_100)
    assert bt.stats["n_orders_sent"] == 2
    assert bt._lead("BUY").price == 289.10
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
    assert bt._lead("BUY").cancel_at is None
    # it is marked as having an amendment on the wire
    assert bt._lead("BUY").amend_at == OPEN_MS + 2_200
    # the OLD terms are still what is resting until it lands, which is the
    # real exposure of an amendment
    assert bt._lead("BUY").price == 289.00
    # land it: the CFO is counted at the exchange, not at the send
    bt._activate_until(OPEN_MS + 3_000)
    assert bt.stats["n_cfos"] == 1
    # now the new terms are in place, with no order orphaned on the way
    assert bt._lead("BUY").price == 289.10
    assert bt.stats["orders_orphaned_by_overwrite"] == 0


# ---------------------------------------------------------------------------
# 4. an order that arrives marketable TRADES, and releases the side
# ---------------------------------------------------------------------------
def test_a_crossing_order_executes_and_frees_the_side():
    """THE DANGEROUS FAILURE MODE OF THE SEND-TIME RESERVATION.

    The side is reserved at send time. If that reservation is not released
    when the order stops existing at arrival, the side is occupied by
    something that will never come live and never be cancelled -- the
    in-flight rule then blocks every later cycle and the symbol goes dark for
    the rest of the session, with no error raised anywhere.

    THIS TEST USED TO ASSERT A REJECTION. It no longer does, because PSX does
    not reject a marketable limit order: Regulation 8.4.2 says an order that
    cannot be executed immediately is queued, which leaves two outcomes and
    not three. 8.5.1 and 8.9(b) list every order type and time-in-force term
    the venue accepts and neither contains a post-only instruction. So the
    order TRADES -- and the side must still be freed, which is what this
    still guards.
    """
    # an engine quoting a bid at 289.00 for 50 shares
    bt = build()
    # send it; the side is reserved at this instant, before it lands
    bt._requote(OPEN_MS + 1_000)
    # the reservation is in place
    assert bt._lead("BUY") is not None
    # THE MARKET FALLS THROUGH OUR PRICE while the order is on the wire, so
    # our 289.00 bid arrives able to buy the 288.50 offer. 300 shares are
    # resting there, more than our 50, so all of it executes.
    set_book(bt, bid=288.00, ask=288.50)
    # land it
    bt._activate_until(OPEN_MS + 2_000)
    # it crossed on contact, and that is counted under its own name
    assert bt.stats["crossed_on_arrival"] == 1
    # all 50 shares traded
    assert bt.stats["crossed_on_arrival_shares"] == 50
    # the old post-only counter must NOT move: nothing was refused
    assert bt.stats["rejected_crossing"] == 0
    # we bought, so the position moved up by the full clip
    assert bt.pos == 50
    # A TAKER PAYS WHAT IS RESTING, not its own limit. Booking this at our
    # 289.00 would invent 25 rupees of profit that never existed.
    assert bt.fills[-1]["px"] == 288.50
    # and the fill carries its own reason, so it can be measured separately
    # from a passive fill and from a deliberate taker sweep
    assert bt.fills[-1]["reason"] == "crossed_on_arrival"
    # NOTHING RESTS, so the side is free again -- the original guard
    assert bt._lead("BUY") is None
    # so the next cycle quotes normally rather than being blocked forever
    bt._requote(OPEN_MS + 2_100)
    # two orders sent in total: the one that crossed, and the replacement
    assert bt.stats["n_orders_sent"] == 2


def test_a_crossing_order_rests_what_it_could_not_fill():
    """Regulation 8.4.4: the unfilled remainder keeps its place in the queue.

    A marketable limit order is not a sweep. It takes what is there and the
    rest RESTS -- so the side stays legitimately occupied, which is the one
    case where a non-empty side after arrival is correct rather than a ghost.
    """
    # the same engine, wanting 50 shares at 289.00
    bt = build()
    # send it
    bt._requote(OPEN_MS + 1_000)
    # the market falls through our price, but only 20 shares are offered
    set_book(bt, bid=288.00, ask=288.50, ask_qty=20)
    # land it
    bt._activate_until(OPEN_MS + 2_000)
    # it crossed
    assert bt.stats["crossed_on_arrival"] == 1
    # but only the 20 that were actually there could trade
    assert bt.stats["crossed_on_arrival_shares"] == 20
    # and the engine counted that this one left something working
    assert bt.stats["crossed_on_arrival_rested"] == 1
    # we bought 20
    assert bt.pos == 20
    # THE REMAINDER RESTS: the side is occupied, and correctly so
    assert bt._lead("BUY") is not None
    # at our own limit price, which is where an unfilled remainder belongs
    assert bt._lead("BUY").price == 289.00
    # carrying only what did not trade
    assert bt._lead("BUY").qty == 30
    # and the lifecycle record is still OPEN, because the order still exists
    assert bt._olog[bt._lead("BUY").oid]["end_reason"] is None


def test_a_crossing_order_never_trades_through_its_own_limit():
    """The whole difference between this and _taker_fill.

    A deliberate sweep pays through every level it needs. A limit order that
    merely arrived marketable stops at its own price -- otherwise the engine
    books fills at prices we never agreed to pay, which is invented loss in
    exactly the same way paying our own limit would be invented profit.
    """
    # wanting 50 at 289.00
    bt = build()
    # send it
    bt._requote(OPEN_MS + 1_000)
    # the market falls through us: 20 offered at 288.50, INSIDE our limit
    set_book(bt, bid=288.00, ask=288.50, ask_qty=20)
    # and a second offer at 289.60, OUTSIDE our limit -- a sweep would take
    # this, a limit order must not
    bt.book.o["H3"] = Order("SELL", 289.60, 100)
    # land it
    bt._activate_until(OPEN_MS + 2_000)
    # only the level inside our limit traded
    assert bt.stats["crossed_on_arrival_shares"] == 20
    # every fill happened at or below what we were willing to pay
    assert all(f["px"] <= 289.00 for f in bt.fills)
    # the 30 we could not fill within our limit rests, rather than paying up
    assert bt._lead("BUY").qty == 30


def test_the_old_post_only_behaviour_still_frees_the_side_when_enabled():
    """The escape hatch has to work, or it is not an escape hatch.

    cfg["cross_on_arrival"]=False restores the pre-2026-09-18 behaviour so a
    published number can be reproduced. It is the OLD BUG, not a cautious
    setting -- but while it exists it must still release the reservation, or
    turning it on to check an old figure silently kills the session.
    """
    # the same engine
    bt = build()
    # turn the venue-accurate behaviour off, as cfg would
    bt.cross_on_arrival = False
    # send the bid
    bt._requote(OPEN_MS + 1_000)
    # the market falls through our price while it is on the wire
    set_book(bt, bid=288.00, ask=288.50)
    # land it
    bt._activate_until(OPEN_MS + 2_000)
    # refused, and counted the old way
    assert bt.stats["rejected_crossing"] == 1
    # nothing traded
    assert bt.pos == 0
    # THE SIDE IS FREE AGAIN -- the guard this test has always really been
    assert bt._lead("BUY") is None
    # and the record says WHY it ended, rather than leaving a blank that reads
    # identically to an order still resting at the close
    assert bt.stats["crossed_on_arrival"] == 0
    # so the next cycle quotes normally
    bt._requote(OPEN_MS + 2_100)
    # two orders sent in total
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
    assert bt._lead("BUY") is None
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
    assert bt._lead("BUY").cancel_at is None
    # the order lands
    bt._activate_until(OPEN_MS + 2_000)
    # the next requote, still halted, now pulls it
    bt._requote(OPEN_MS + 2_100)
    assert bt._lead("BUY").cancel_at == OPEN_MS + 2_200
    # and it is cancelled when that lands
    bt._activate_until(OPEN_MS + 3_000)
    assert bt._lead("BUY") is None
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
    assert bt._lead("BUY").amend_at is None
    # the strategy reprices: one amendment goes out
    bt.strat.want = {"BUY": (289.10, 50)}
    bt._requote(OPEN_MS + 2_100)
    assert bt.stats["n_orders_sent"] == 2
    # the amendment is on the wire and the OLD terms are still resting
    assert bt._lead("BUY").amend_at == OPEN_MS + 2_200
    assert bt._lead("BUY").price == 289.00
    # THE STRATEGY REPRICES AGAIN while that amendment is unanswered
    bt.strat.want = {"BUY": (289.20, 50)}
    bt._requote(OPEN_MS + 2_150)
    # NOTHING WENT OUT -- no second amendment, and crucially NO CANCEL
    assert bt.stats["n_orders_sent"] == 2
    assert bt._lead("BUY").cancel_at is None, "the quote was pulled"
    # the order is still resting and still fillable at the old terms
    assert bt._lead("BUY").t_active is not None
    # and the cycle is accounted for
    assert bt.stats["requotes_blocked_in_flight"] >= 1
    # when the amendment lands the side is free again and quoting resumes
    bt._activate_until(OPEN_MS + 3_000)
    assert bt._lead("BUY").amend_at is None
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
    assert bt._lead("BUY").amend_at is not None
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
    assert bt._lead("BUY") is None
    # the amendment lands and is refused as stale
    bt._activate_until(OPEN_MS + 3_000)
    assert bt.stats["stale_cfos_ignored"] == 1
    # AND THE SIDE QUOTES AGAIN. Nothing is stuck holding it.
    bt._requote(OPEN_MS + 3_100)
    assert bt._lead("BUY") is not None


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
from core.model import DesiredQuotes, Fill, PlaceOrder, QuoteIntent, ReplaceOrder
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


# ---------------------------------------------------------------------------
# 9. THE THIRD POLICY: DON'T TOP UP
# ---------------------------------------------------------------------------
# Clip 500, a partial fill takes 400, 100 is left resting with its place.
#
#   exact             amend the 100 back up to 500 -- 8.5.2 sends the WHOLE
#                     order to the back of the queue for that
#   queue_preserving  leave the 100, send a SECOND order for 400
#   reduce_only       send NOTHING. Show 100 until someone hits it.
#
# reduce_only was implemented in _matches and documented in QuoteTolerance, but
# sim/gate.py never ran it, so it had never been measured against the other
# two. These tests pin its behaviour in both directions before it is.
def _oms_with(policy):
    """An order manager on one named size policy, permissive gateway."""
    # a venue with one all-day session
    venue = PSXVenue(session_provider=lambda d: [
        SessionSegment(start_ms=0, end_ms=86_400_000)])
    # amendments enabled, so the free reduction is available to it
    return OrderManager(
        venue=venue,
        gateway=RiskGateway([OrderQuantityCheck(max_quantity=1_000_000)]),
        kill_switch=KillSwitch(), session_id="T", account="A",
        tolerance=QuoteTolerance(quantity_policy=policy), use_replace=True)


def _partially_filled(policy, clip=500, taken=400, px=36100):
    """One order placed, acknowledged, and partly filled. Returns (oms, order)."""
    # the manager
    oms = _oms_with(policy)
    # ask for the full clip
    oms.set_desired(DesiredQuotes(
        symbol=SYM, bid=QuoteIntent(side=Side.BUY, price_minor=px,
                                    quantity=clip)))
    # it goes out
    acts = oms.reconcile(1_000, DATE, {SYM: px})
    # the exchange acknowledges it
    oms.on_ack(acts[0].cl_ord_id, "EX1")
    # and part of it trades
    oms.on_fill(Fill(cl_ord_id=acts[0].cl_ord_id, symbol=SYM, side=Side.BUY,
                     price_minor=px, quantity=taken, timestamp_ms=1_100))
    # ready for the cycle under test
    return oms, oms._orders[acts[0].cl_ord_id]


def test_dont_top_up_sends_nothing_after_a_partial_fill():
    """The defining behaviour: no message, smaller size, place kept."""
    # 100 of 500 left resting
    oms, order = _partially_filled("reduce_only")
    # the cycle that the other two policies use to restore size
    acts = oms.reconcile(2_000, DATE, {SYM: 36100})
    # NOTHING GOES OUT. Not an amendment, not a second order.
    assert acts == []
    # one order still resting, showing the remainder and nothing more
    working = oms.working_orders(SYM)
    assert len(working) == 1
    assert working[0].leaves_quantity == 100
    # it is the SAME order, so it still holds the place it earned
    assert working[0].cl_ord_id == order.cl_ord_id
    # and the cycle is counted as nothing-to-do, not as held
    assert oms.plan_counts["held_message_in_flight"] == 0


def test_dont_top_up_still_takes_the_free_reduction():
    """It pays for nothing, but it does take what 8.5.2 gives away."""
    # 100 of 500 left resting
    oms, _ = _partially_filled("reduce_only")
    # the strategy now wants LESS than is resting
    oms.set_desired(DesiredQuotes(
        symbol=SYM, bid=QuoteIntent(side=Side.BUY, price_minor=36100,
                                    quantity=50)))
    # one cycle
    acts = oms.reconcile(2_000, DATE, {SYM: 36100})
    # ONE amendment, downward. 8.5.2 applies a reduction in place, so this is
    # the one message the policy is willing to send.
    assert len(acts) == 1
    assert isinstance(acts[0], ReplaceOrder)
    assert acts[0].quantity == 50


def test_dont_top_up_returns_to_full_size_when_the_price_moves():
    """THE PROPERTY THAT DECIDES WHETHER THIS POLICY IS USABLE AT ALL.

    Size under this policy only ever shrinks -- a partial fill takes some and
    nothing puts it back. If a reprice did not restore the full clip, a quiet
    name would ratchet down to a token quote and stay there for the session.
    It does restore it, because a price move is a requote and a requote asks
    for the whole desired size.
    """
    # 100 of 500 left resting at 361.00
    oms, _ = _partially_filled("reduce_only")
    # the strategy wants the full clip at a NEW price
    oms.set_desired(DesiredQuotes(
        symbol=SYM, bid=QuoteIntent(side=Side.BUY, price_minor=36110,
                                    quantity=500)))
    # one cycle
    acts = oms.reconcile(2_000, DATE, {SYM: 36110})
    # one amendment, to the new price AND back to the full size
    assert len(acts) == 1
    assert acts[0].price_minor == 36110
    assert acts[0].quantity == 500


def test_the_three_policies_do_three_different_things():
    """Same partial fill, three policies, three answers. The whole comparison."""
    # what each one sends on the cycle after a 400-of-500 partial fill
    answers = {}
    # every policy the gate now runs
    for policy in ("exact", "queue_preserving", "reduce_only"):
        # the identical starting state
        oms, _ = _partially_filled(policy)
        # the identical cycle
        acts = oms.reconcile(2_000, DATE, {SYM: 36100})
        # what it did, and what is showing afterwards
        answers[policy] = (
            [type(a).__name__ for a in acts],
            sum(o.leaves_quantity for o in oms.working_orders(SYM)),
            len(oms.working_orders(SYM)))
    # AMEND UP: one message, the whole order re-sent at full size
    assert answers["exact"] == (["ReplaceOrder"], 100, 1)
    # SECOND ORDER: a new order for the shortfall, two resting, full size
    assert answers["queue_preserving"] == (["PlaceOrder"], 500, 2)
    # DON'T TOP UP: nothing at all, one resting, smaller size
    assert answers["reduce_only"] == ([], 100, 1)


# ---------------------------------------------------------------------------
# 10. THE HARNESS CANNOT HOLD TWO ORDERS ON ONE SIDE
# ---------------------------------------------------------------------------
# Backtester.work is dict[side] -> ONE MyOrder. EngineReplay inherits it. So a
# policy that wants several orders on a side has the second one land ON TOP of
# the first, and the first -- the one carrying the queue position the policy
# exists to protect -- is gone: no cancel, no fill, no lifecycle end.
#
# Measured on the real harness, 200 cycles with the price walking and one
# partial fill:
#
#   exact             oms approved 101   exchange accepted 101   resting 1 / 1
#   reduce_only       oms approved 101   exchange accepted 101   resting 1 / 1
#   queue_preserving  oms approved 172   exchange accepted  59   resting 1 / 0
#
# The last row is the order manager believing it has a quote resting while the
# exchange has nothing. Every queue_preserving number the gate has ever printed
# came from that state.
#
# These tests exist so the limitation is stated by the suite rather than
# rediscovered, and so that anyone who makes Backtester hold a list per side
# finds out immediately that they have fixed it.
from sim.replay import EngineReplay
from venues.psx_strategy import MicroMMAdapter


class _WalkingMM:
    """A strategy with the one attribute the adapter checks, and a price."""
    # the adapter refuses a strategy whose tick disagrees with the venue's
    tick = 0.01

    def __init__(self, want):
        # what it wants this cycle
        self.want = want

    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # a fresh dict, because _requote rewrites it in place
        return dict(self.want)


def _replay_on(policy):
    """The real EngineReplay harness on one named size policy."""
    # a venue with one all-day session
    venue = PSXVenue(session_provider=lambda d: [
        SessionSegment(start_ms=OPEN_MS, end_ms=CLOSE_MS)])
    # a strategy wanting a 500-share bid
    mm = _WalkingMM({"BUY": (289.00, 500)})
    # the production adapter over it
    adapter = MicroMMAdapter("PPL", venue, mm, reference_price_minor=28900)
    # the order manager on the policy under test
    oms = OrderManager(
        venue=venue,
        gateway=RiskGateway([OrderQuantityCheck(max_quantity=1_000_000)]),
        kill_switch=KillSwitch(), session_id="R", account="C1",
        tolerance=QuoteTolerance(quantity_policy=policy), use_replace=True)
    # constant latency, so timing is exact
    cfg = {"latency_model": LatencyModel(
               decision_ms=0.0, wire_out_median_ms=100.0, wire_out_tail_ms=0.0,
               wire_in_median_ms=100.0, wire_in_tail_ms=0.0, tail_prob=0.0),
           "at_price_mode": "queue", "fill_on_crossing_adds": False,
           "log_equity": False, "session": (OPEN_MS, CLOSE_MS)}
    # the harness
    rep = EngineReplay(strategy=mm, adapter=adapter, oms=oms, symbol="PPL",
                       cfg=cfg)
    # the date the window check needs
    rep.session_date = "2026-06-30"
    # a two-sided book in continuous trading
    set_book(rep)
    # everything the test reaches
    return rep, oms, mm


def test_the_second_order_policy_keeps_both_orders_resting():
    """THE FIX. Backtester.work is a LIST per side, so both orders rest.

    Until 2026-09-17 this was dict[side] -> ONE MyOrder, and the second order
    to land replaced the first -- including the very order whose queue
    position the policy exists to protect. This test was written the other way
    up, asserting the loss; it now asserts that nothing is lost.
    """
    # the harness on the policy that wants two orders per side
    rep, oms, _ = _replay_on("queue_preserving")
    # one order out and resting
    rep._requote(OPEN_MS + 1_000)
    rep._activate_until(OPEN_MS + 2_000)
    # a partial fill takes 400 of the 500
    rep._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 2_100, price=288.00,
                                   qty=400, aggressor_side="SELL"))
    # the order manager has 100 resting with its place
    assert [o.leaves_quantity for o in oms.working_orders("PPL")] == [100]
    # the top-up cycle: a SECOND order for the 400
    rep._requote(OPEN_MS + 2_200)
    rep._activate_until(OPEN_MS + 3_000)
    # THE ORDER MANAGER AND THE EXCHANGE NOW AGREE: two orders, 500 in total
    assert sorted(o.leaves_quantity
                  for o in oms.working_orders("PPL")) == [100, 400]
    assert sorted(o.qty for o in rep._side_orders("BUY")) == [100, 400]
    # nothing was overwritten
    assert rep.engine_stats.get("placed_onto_occupied_side", 0) == 1
    # and the OLDER order is still first in the list, which is what carries
    # its time priority when both sit at the same price
    assert rep._side_orders("BUY")[0].qty == 100


def test_the_one_order_policies_keep_the_two_views_in_step():
    """The same harness is sound for the policies that hold one order.

    This is what makes 'amend up' and 'don't top up' comparable to each other
    even though the third policy is not comparable to either.
    """
    # both single-order policies
    for policy in ("exact", "reduce_only"):
        # a fresh harness
        rep, oms, mm = _replay_on(policy)
        # one order out and resting
        rep._requote(OPEN_MS + 1_000)
        rep._activate_until(OPEN_MS + 2_000)
        # a partial fill
        rep._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 2_100, price=288.00,
                                       qty=400, aggressor_side="SELL"))
        # fifty cycles with the price walking, which is what the real strategy
        # does and what exposes a divergence if there is one
        t = OPEN_MS + 2_200
        for i in range(50):
            # a price that moves every cycle
            mm.want = {"BUY": (289.00 + (i % 7) * 0.01, 500)}
            rep._requote(t)
            rep._activate_until(t + 50)
            t += 100
        # NEVER MORE THAN ONE ORDER PER SIDE, so nothing is ever overwritten
        assert rep.engine_stats.get("placed_onto_occupied_side", 0) == 0, policy
        # and the two views agree on how many orders exist
        assert len(oms.working_orders("PPL")) == len(rep._all_orders()), policy


def test_the_side_survives_a_price_move_with_two_orders_resting():
    """The consequence of the fix, and the thing that used to kill the run.

    WHAT USED TO HAPPEN. B overwrote A on the single slot, so A became a ghost
    -- still listed by the order manager, gone from the simulator. The next
    price move then cancelled B (real, so the cancel worked) and amended A
    (a ghost, so the amendment was refused). The side was empty from that
    point and never recovered: every later cycle re-amended the ghost, was
    refused, and tried again.

    WHAT HAPPENS NOW. Both orders are real, so the cancel and the amendment
    both land, and the side still has a quote on it afterwards.
    """
    # the harness on the policy that rests two orders
    rep, oms, mm = _replay_on("queue_preserving")
    # one order out, resting, then partly filled
    rep._requote(OPEN_MS + 1_000)
    rep._activate_until(OPEN_MS + 2_000)
    rep._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 2_100, price=288.00,
                                   qty=400, aggressor_side="SELL"))
    # the top-up
    rep._requote(OPEN_MS + 2_200)
    rep._activate_until(OPEN_MS + 3_000)
    # two orders, on both sides of the comparison
    assert len(oms.working_orders("PPL")) == 2
    assert len(rep._side_orders("BUY")) == 2
    # THE PRICE MOVES -- the step that used to empty the side
    mm.want = {"BUY": (289.01, 500)}
    rep._requote(OPEN_MS + 3_100)
    rep._activate_until(OPEN_MS + 3_400)
    # NO AMENDMENT WAS REFUSED: both targets were really there
    assert rep.engine_stats["replace_rejected_stale"] == 0
    # AND THE SIDE STILL HAS A QUOTE ON IT
    assert len(rep._side_orders("BUY")) >= 1
    # five more cycles of moving price, and it never goes dark
    t = OPEN_MS + 3_500
    for i in range(5):
        # a price that keeps moving, as the real strategy's does
        mm.want = {"BUY": (289.02 + i * 0.01, 500)}
        rep._requote(t)
        rep._activate_until(t + 300)
        t += 400
        # something of ours is resting on every one of them
        assert len(rep._side_orders("BUY")) >= 1, f"went dark at cycle {i}"
    # and no amendment was ever aimed at an order that was not there
    assert rep.engine_stats["replace_rejected_stale"] == 0


# ---------------------------------------------------------------------------
# 11. THE FILL ENGINE, NOW THAT A SIDE CAN HOLD SEVERAL ORDERS
# ---------------------------------------------------------------------------
# _on_market_trade used to look at ONE order. It now walks our orders in the
# exchange's own priority -- price first, then time -- consuming the
# aggressor's quantity as it goes. Two things have to be true:
#
#   a) with one order resting it behaves exactly as it always did, because
#      `exact` is the baseline every published number in this project rests
#      on and the reconcile gate judges it;
#   b) with several, the flow reaches them in the right order and stops when
#      it is spent.
def _bt_with(orders, side="BUY"):
    """A Backtester with specific orders of ours already resting."""
    # an engine with a two-sided book
    bt = build()
    # drop whatever the harness placed
    bt._side_orders("BUY").clear()
    bt._side_orders("SELL").clear()
    # put ours in, in send order, already live
    for k, (px, qty) in enumerate(orders):
        bt._side_orders(side).append(
            MyOrder(side, px, qty, {}, OPEN_MS + k, oid=100 + k))
    # ready
    return bt


def test_one_order_fills_exactly_as_it_always_did():
    """(a) The single-order path is unchanged. This is the baseline."""
    # one bid for 50 at 289.00
    bt = _bt_with([(289.00, 50)])
    # a print BELOW it takes the lot
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 1_000, price=288.00,
                                  qty=500, aggressor_side="SELL"))
    # one fill, our whole size, at OUR price, and the side is empty
    assert len(bt.fills) == 1
    assert bt.fills[0]["qty"] == 50
    assert bt.fills[0]["px"] == 289.00
    assert bt.fills[0]["reason"] == "through"
    assert bt._side_orders("BUY") == []


def test_a_trade_smaller_than_our_order_fills_only_part_of_it():
    """(a) Partial fills still work, and the remainder keeps its place."""
    # one bid for 500
    bt = _bt_with([(289.00, 500)])
    # a print through it, for less than our size
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 1_000, price=288.00,
                                  qty=120, aggressor_side="SELL"))
    # we took the whole print and no more
    assert bt.fills[0]["qty"] == 120
    # and the rest is still resting, same order object
    assert bt._side_orders("BUY")[0].qty == 380


def test_the_aggressor_is_shared_between_our_orders_oldest_first():
    """(b) Two of ours at ONE price: the older fills first, then the younger."""
    # two bids at the same price, sent in this order
    bt = _bt_with([(289.00, 100), (289.00, 400)])
    # a print through both, big enough for the first and part of the second
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 1_000, price=288.00,
                                  qty=250, aggressor_side="SELL"))
    # THE OLDER ORDER FILLED FIRST AND IN FULL, then the younger took the rest
    assert [(f["qty"], f["oid"]) for f in bt.fills] == [(100, 100), (150, 101)]
    # the older is gone, the younger is short by what it took
    assert len(bt._side_orders("BUY")) == 1
    assert bt._side_orders("BUY")[0].qty == 250


def test_the_flow_stops_when_the_aggressor_is_spent():
    """(b) A small print cannot fill more of our size than it carried."""
    # two bids at one price
    bt = _bt_with([(289.00, 100), (289.00, 400)])
    # a print that is smaller than our FIRST order
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 1_000, price=288.00,
                                  qty=60, aggressor_side="SELL"))
    # only the older order was touched, for exactly the print size
    assert [(f["qty"], f["oid"]) for f in bt.fills] == [(60, 100)]
    # and both are still resting
    assert [o.qty for o in bt._side_orders("BUY")] == [40, 400]


def test_the_better_price_fills_first_whatever_order_it_was_sent_in():
    """(b) PRICE BEATS TIME. The exchange's rule, not our list's order."""
    # a bid at 289.00 sent FIRST, and a better bid at 289.05 sent second
    bt = _bt_with([(289.00, 100), (289.05, 100)])
    # a print below both
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 1_000, price=288.00,
                                  qty=150, aggressor_side="SELL"))
    # THE HIGHER BID FILLED FIRST even though it was sent second: a buyer at
    # 289.05 is ahead of a buyer at 289.00 no matter who arrived when
    assert [(f["px"], f["qty"]) for f in bt.fills] == [(289.05, 100),
                                                       (289.00, 50)]


def test_an_order_priced_away_from_the_print_does_not_fill():
    """(b) Walking best-first must stop, not keep going into worse prices."""
    # one bid at the touch and one well below it
    bt = _bt_with([(289.00, 100), (288.50, 100)])
    # a print that reaches the first but not the second
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 1_000, price=288.90,
                                  qty=500, aggressor_side="SELL"))
    # only the bid the print went through filled
    assert [(f["px"], f["qty"]) for f in bt.fills] == [(289.00, 100)]
    # the one below the print is untouched
    assert [o.price for o in bt._side_orders("BUY")] == [288.50]


def test_our_own_orders_at_one_price_share_the_market_queue_once():
    """(b) The queue in front of a level is drained ONCE, not per order.

    Each of our orders carries its own `ahead` snapshot of the SAME historical
    orders. Draining both would let the market queue be consumed twice and
    would fill us roughly twice as often as the exchange would have.
    """
    # two of ours at one price, each believing 200 shares sit in front
    bt = _bt_with([(289.00, 100), (289.00, 100)])
    # the same historical order ahead of both, as a real snapshot would give
    for o in bt._side_orders("BUY"):
        o.ahead = {"H9": 200.0}
    # a print AT our price for 250: 200 clears the queue, 50 reaches us
    bt._on_market_trade(FakeTrade(ts_exch=OPEN_MS + 1_000, price=289.00,
                                  qty=250, aggressor_side="SELL"))
    # ONLY 50 filled, and it went to the older order
    assert [(f["qty"], f["oid"], f["reason"]) for f in bt.fills] == [
        (50, 100, "at_queue")]
    # the queue was charged once: had it been charged per order, the second
    # order would have seen its own untouched 200 and filled nothing, and had
    # it not been charged at all both would have filled in full
    assert bt._side_orders("BUY")[0].qty == 50


# ---------------------------------------------------------------------------
# 12. THE AMENDMENT DECOMPOSITION
# ---------------------------------------------------------------------------
# PSX 8.5.2 applies a size REDUCTION in place and re-queues everything else.
# So "how many amendments kept their place" is the same question as "how many
# were reductions". The gate now prints the kinds side by side instead of
# predicting the answer from a policy's description -- which was done twice,
# wrongly, on 2026-09-17. These tests check the counters say what they claim.
def test_a_price_change_is_counted_as_a_price_change_and_loses_place():
    """A reprice re-queues: PSX 8.5.2, and the venue flag the engine reads."""
    # a resting bid, CFO enabled
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    # the strategy wants the same size at a different price
    bt.strat.want = {"BUY": (289.10, 50)}
    bt._requote(OPEN_MS + 2_100)
    bt._activate_until(OPEN_MS + 3_000)
    # one amendment, classified as a price move
    assert bt.stats["n_cfos"] == 1
    assert bt.stats["n_cfos_price_change"] == 1
    assert bt.stats["n_cfos_qty_up"] == 0
    assert bt.stats["n_cfos_qty_down"] == 0
    # and it did NOT keep its place
    assert bt.stats.get("n_cfos_kept_priority", 0) == 0


def test_a_size_reduction_is_counted_as_one_and_keeps_its_place():
    """The one amendment 8.5.2 makes free, and the only one that keeps."""
    # a resting bid for 50
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    # the strategy wants LESS at the same price
    bt.strat.want = {"BUY": (289.00, 20)}
    bt._requote(OPEN_MS + 2_100)
    bt._activate_until(OPEN_MS + 3_000)
    # one amendment, classified as a reduction
    assert bt.stats["n_cfos_qty_down"] == 1
    assert bt.stats["n_cfos_price_change"] == 0
    # AND IT KEPT ITS PLACE. This is the identity the gate prints: on PSX,
    # kept place should equal size down.
    assert bt.stats["n_cfos_kept_priority"] == 1


def test_a_size_increase_is_counted_as_one_and_loses_its_place():
    """Growing costs the queue on PSX, which is the whole reason for policy 2."""
    # a resting bid for 50
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    # the strategy wants MORE at the same price
    bt.strat.want = {"BUY": (289.00, 120)}
    bt._requote(OPEN_MS + 2_100)
    bt._activate_until(OPEN_MS + 3_000)
    # one amendment, classified as an increase
    assert bt.stats["n_cfos_qty_up"] == 1
    assert bt.stats["n_cfos_price_change"] == 0
    # and it did NOT keep its place
    assert bt.stats.get("n_cfos_kept_priority", 0) == 0


def test_the_kinds_add_up_to_the_amendment_total():
    """No amendment falls outside the three kinds, so the table balances."""
    # a resting bid
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    # a price move, then a reduction, then an increase
    for want, t in (((289.10, 50), 2_100), ((289.10, 20), 3_100),
                    ((289.10, 90), 4_100)):
        # what the strategy asks for this cycle
        bt.strat.want = {"BUY": want}
        bt._requote(OPEN_MS + t)
        bt._activate_until(OPEN_MS + t + 900)
    # three amendments, one of each kind, and they sum to the total
    kinds = (bt.stats["n_cfos_price_change"] + bt.stats["n_cfos_qty_up"]
             + bt.stats["n_cfos_qty_down"])
    assert kinds == bt.stats["n_cfos"] == 3
    # and exactly the reduction kept its place
    assert bt.stats["n_cfos_kept_priority"] == bt.stats["n_cfos_qty_down"] == 1


def test_both_policies_count_a_wait_the_same_way():
    """The two size policies must MEASURE waiting identically.

    They did not. The single-order path tests for an outstanding message
    before it tests for 'the strategy wants nothing here'; the list path
    tested in the other order and so recorded no wait on a halted or stale
    cycle. On six symbol-days that alone made the list policy look as though
    it waited half as often -- 8,387 against 16,785 -- when the difference was
    where the line sat in the function, not what either policy did.
    """
    # one manager per policy, each with an order resting and a cancel in
    # flight on it
    for policy in ("exact", "queue_preserving"):
        # a manager on this policy
        oms = _oms_with(policy)
        # ask for a quote
        oms.set_desired(DesiredQuotes(
            symbol=SYM, bid=QuoteIntent(side=Side.BUY, price_minor=36100,
                                        quantity=500)))
        # it goes out and is acknowledged
        acts = oms.reconcile(1_000, DATE, {SYM: 36100})
        oms.on_ack(acts[0].cl_ord_id, "EX1")
        # now the strategy wants NOTHING here -- a halt, a stale feed, a
        # one-sided book. The cancel goes out.
        oms.set_desired(DesiredQuotes.flat(SYM))
        oms.reconcile(2_000, DATE, {SYM: 36100})
        # the cancel is unanswered, so the order has a message in flight
        order = oms._orders[acts[0].cl_ord_id]
        assert order.has_message_in_flight
        # the count before the cycle under test
        before = oms.plan_counts["held_message_in_flight"]
        # another cycle, still wanting nothing, message still unanswered
        oms.reconcile(2_100, DATE, {SYM: 36100})
        # BOTH POLICIES RECORD THE WAIT. Neither sends anything, and neither
        # pretends the cycle was quiet for some other reason.
        assert oms.plan_counts["held_message_in_flight"] == before + 1, policy


# ---------------------------------------------------------------------------
# 13. THE ACKNOWLEDGEMENT WAIT
# ---------------------------------------------------------------------------
# After sending a CANCEL, mm_backtest holds that side until the exchange's
# reply comes back -- the cancel's landing time plus one inbound latency.
# Until then the old order may still be resting and may still fill, so putting
# a fresh order there could leave more size working than intended.
#
# IT IS A RULE ABOUT HOW FAST YOU LEARN, NOT ABOUT MESSAGE TYPE. A
# Cancel/Replace ('G') sets no wait on either side, because there is no
# separate cancel to be told about. Only a genuine Cancel does.
#
# The harness drew the latency and stored it in self.ack_until, then never
# read it back -- so the engine resumed quoting the instant the cancel landed.
# On PPL 2026-06-19 the backtest waited 1,465 times and the engine none, which
# is what put the two runs on different messages, different latency draws, and
# therefore different fills.
def test_a_reprice_sets_no_acknowledgement_wait():
    """THE POINT ABOUT 'G'. One message, nothing to be told about, no wait."""
    # a resting bid, amendments enabled so a reprice is one message
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    # nothing is waiting on an acknowledgement
    assert bt.ack_until["BUY"] == 0
    # the strategy reprices: ONE amendment goes out
    bt.strat.want = {"BUY": (289.10, 50)}
    bt._requote(OPEN_MS + 2_100)
    # STILL no wait. This is why 'we use G so we do not need the wait' is
    # already true, and why removing the wait would not change a reprice.
    assert bt.ack_until["BUY"] == 0
    assert bt.stats["requotes_blocked_by_ack"] == 0


def test_a_cancel_does_set_an_acknowledgement_wait():
    """And a genuine Cancel does, because there IS something to be told."""
    # a resting bid
    bt = build(use_cfo=True)
    bt._requote(OPEN_MS + 1_000)
    bt._activate_until(OPEN_MS + 2_000)
    # the strategy wants nothing here now -- a halt, a stale feed, whatever.
    # Pulling a side is a Cancel, not an amendment.
    bt.strat.want = {}
    bt._requote(OPEN_MS + 2_100)
    # the cancel is on the wire and the side is held until it is acknowledged
    assert bt.ack_until["BUY"] > OPEN_MS + 2_100
    # and the wait is LONGER than the cancel's own landing time, because it
    # includes the reply coming back
    assert bt.ack_until["BUY"] > bt._lead("BUY").cancel_at


def test_the_engine_honours_the_same_wait():
    """Both sides must hold for the same reason, or the gate is comparing two
    different rules and calling the difference a bug."""
    # the engine harness with a resting bid
    rep, oms, mm = _replay_on("exact")
    rep._requote(OPEN_MS + 1_000)
    rep._activate_until(OPEN_MS + 2_000)
    # the strategy wants nothing: the order manager sends a cancel
    mm.want = {}
    rep._requote(OPEN_MS + 2_100)
    # the wait was drawn and stored, exactly as the backtest does
    assert rep.ack_until["BUY"] > OPEN_MS + 2_100
    # the cancel lands, freeing the side as far as the exchange is concerned
    rep._activate_until(int(rep.ack_until["BUY"]) - 1)
    # the strategy wants a quote again, BEFORE the acknowledgement is due
    mm.want = {"BUY": (289.00, 500)}
    sent_before = rep.stats["n_orders_sent"]
    rep._requote(int(rep.ack_until["BUY"]) - 1)
    # NOTHING WENT OUT. Until this fix the engine quoted here and the backtest
    # did not, which is the whole divergence.
    assert rep.stats["n_orders_sent"] == sent_before
    assert rep.stats.get("requotes_blocked_by_ack", 0) >= 1
    # and the order manager was told, so nothing is stranded: once the wait
    # expires the side quotes again
    rep._requote(int(rep.ack_until["BUY"]) + 1)
    assert rep.stats["n_orders_sent"] == sent_before + 1


def test_a_held_message_does_not_strand_the_order_manager():
    """The dangerous failure mode of holding a message back.

    The manager marks its own state the moment it approves an action. Drop the
    message without telling it and that side is blocked for the rest of the
    session, with nothing ever arriving to clear it.
    """
    # a resting bid, then a cancel, then a blocked requote
    rep, oms, mm = _replay_on("exact")
    rep._requote(OPEN_MS + 1_000)
    rep._activate_until(OPEN_MS + 2_000)
    mm.want = {}
    rep._requote(OPEN_MS + 2_100)
    rep._activate_until(int(rep.ack_until["BUY"]) - 1)
    mm.want = {"BUY": (289.00, 500)}
    rep._requote(int(rep.ack_until["BUY"]) - 1)
    # the manager has NOTHING working that it believes is in flight
    assert not any(o.has_message_in_flight for o in oms.working_orders("PPL"))
    # so the side is free the moment the wait expires, and stays productive
    t = int(rep.ack_until["BUY"]) + 1
    for i in range(5):
        rep._requote(t); rep._activate_until(t + 300); t += 400
    # something of ours is resting at the end of it
    assert len(rep._side_orders("BUY")) >= 1
