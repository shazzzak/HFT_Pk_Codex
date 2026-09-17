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
