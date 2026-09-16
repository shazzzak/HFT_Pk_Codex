# ============================================================================
# test_buffer_isolation.py -- does removing the buffer's price move leave the
# quoting result behind, and nothing else?
# ============================================================================
# THE PROBLEM IT GUARDS. Under the long_buffer policy every backtest day buys
# a buffer at that day's first print and sells it back into that day's close.
# The stock's move between those two moments lands in the P&L. On a 500-share
# buffer that is several times the size of a day of quoting, so the arm's
# number is mostly a bet on the stock and only a little market making.
#
# The engine now records buffer_qty and buffer_px and reports
# equity_ex_buffer_move = equity_liquidated - buffer_qty*(mid_close - buffer_px)
#
# THE INVARIANT THIS FILE CHECKS: on a day where we never quote and never
# trade, that number must be the SAME whatever the stock did. What is left is
# the entry fee, the exit fee, and the cost of crossing the spread on the way
# out -- all real costs of carrying a buffer, all independent of direction.
#
# If the invariant holds, the long_buffer arm's ex-buffer column is comparable
# to the other arms. If it does not, the comparison is still measuring the
# stock and the policy conclusion would be drawn from noise.
#
# READ-ONLY. Touches no data, writes no file, needs no parsed store.
#
# Run from existing_mm_live/:
#   caffeinate -is python test_buffer_isolation.py
# Exit code 0 = every case passed, 1 = something is wrong.
# ============================================================================

# argv/exit only
import sys
# a throwaway record type for the fabricated trade print
from types import SimpleNamespace

# the engine, its latency model, the Order record the Book holds, and the fee
# function -- the SAME fee the engine charges, not a re-typed copy
from mm_backtest import Backtester, LatencyModel, Order, fee_for


class Stub:
    """A MicrostructureMM-shaped stand-in that never wants to quote.

    The whole point of these cases is a day with NO market making in it, so
    the only thing in the P&L is the buffer.
    """
    # the engine reads .tick off the strategy; PSX is a flat one paisa
    tick = 0.01

    def observe(self, *args):
        # the engine calls this on every event; nothing here needs the state
        return None

    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # no bid, no ask, ever
        return {}


def build(buffer_shares, bid_px, bid_qty):
    """An engine holding `buffer_shares`, with one bid resting to exit into."""
    # the configuration: latency is irrelevant here because nothing is sent,
    # but it is pinned anyway so a failure cannot be a random draw
    cfg = dict(
        # every source of randomness zeroed
        latency_model=LatencyModel(decision_ms=0.0, wire_out_median_ms=100.0,
                                   wire_out_tail_ms=0.0,
                                   wire_in_median_ms=100.0,
                                   wire_in_tail_ms=0.0, tail_prob=0.0),
        # exact queue consumption, the mode the shipped runs used
        at_price_mode="queue",
        # the per-event equity curve is not needed
        log_equity=False,
        # a session wide enough that no end-of-day logic engages on its own
        session=(0, 10 ** 9),
        # THE BUFFER: shares held from the first print
        opening_inventory=buffer_shares)
    # the engine, with a strategy that never quotes
    bt = Backtester(Stub(), cfg)
    # BUILD THE BOOK BY HAND: one bid deep enough to absorb the whole buffer,
    # so the exit is clean and `unfilled` is zero
    bt.book.o.clear()
    # the bid we will sell the buffer into
    bt.book.o["B1"] = Order("BUY", bid_px, bid_qty)
    # an offer, so the book is two-sided and a mid exists
    bt.book.o["A1"] = Order("SELL", round(bid_px + 0.02, 2), bid_qty)
    # continuous trading
    bt.book.phase = "CONTINUOUS_AUCTION"
    # no circuit limits published
    bt.book.limit_up = None
    bt.book.limit_dn = None
    # the engine, ready to be driven
    return bt


def first_print(bt, price):
    """Drive one market trade through the engine, which pays for the buffer."""
    # a record shaped like the parsed trade rows the engine consumes
    r = SimpleNamespace(
        # not an auction print, so the handler proceeds
        initiator="TRADE",
        # the traded price -- this is what the buffer gets bought at
        price=price,
        # some size; nothing of ours is working, so it cannot fill us
        qty=100.0,
        # exchange timestamp
        ts_exch=1000,
        # a taker direction the handler recognises
        aggressor_side="BUY",
        # no named resting victim
        rest_oid=None,
        # the passive side's own field, unused with no working order
        side="SELL")
    # run it
    bt._on_market_trade(r)


def eod_numbers(bt):
    """Reproduce the engine's end-of-day arithmetic for a no-trade day.

    Every expression here is the one mm_backtest uses in its EOD block; this
    file drives the pieces directly because that block lives inside run() and
    run() needs a parsed store.
    """
    # walk the book to flatten the position, fees included
    liq_cash, unfilled, vwap, _levels = bt.book.liquidation_value(
        bt.pos, fee_fn=fee_for)
    # the closing touch
    bb, _, ba, _ = bt.book.bbo()
    # the closing mid
    mid = (bb + ba) / 2
    # what the account shows: cash plus the liquidation proceeds
    equity_liquidated = bt.cash + liq_cash
    # the directional term the sweep removes
    buffer_price_move = bt.buffer_qty * (mid - bt.buffer_px)
    # the policy-comparison number
    equity_ex = equity_liquidated - buffer_price_move
    # everything the caller might want to assert on
    return equity_liquidated, equity_ex, unfilled, vwap, mid


# every (name, passed?) pair, so the run reports a total rather than dying on
# the first failure
results = []


def check(name, condition):
    """Record one assertion and print it as it happens."""
    # keep it for the summary
    results.append((name, condition))
    # and show it immediately
    print(("PASS  " if condition else "FAIL  ") + name)


# ---------------------------------------------------------------------------
# 1. the engine records what the buffer cost
# ---------------------------------------------------------------------------
# 500 shares, a bid at 300.00 to exit into
bt = build(500.0, 300.00, 5000)
# before any print, nothing has been paid and no price is known
check("buffer_px is None before the first print", bt.buffer_px is None)
# the size is known from the configuration alone
check("buffer_qty is the opening inventory", bt.buffer_qty == 500.0)
# the first print at 301.00
first_print(bt, 301.00)
# the acquisition price was captured
check("buffer_px is the first print", bt.buffer_px == 301.00)
# cash was debited for the shares AND the fee, not just the shares
check("cash debited for shares plus fee",
      abs(bt.cash - (-500.0 * 301.00 - fee_for(301.00, 500.0))) < 1e-9)
# and it is not charged twice
first_print(bt, 302.00)
check("the buffer is paid for exactly once",
      abs(bt.cash - (-500.0 * 301.00 - fee_for(301.00, 500.0))) < 1e-9)

# ---------------------------------------------------------------------------
# 2. THE INVARIANT: the ex-buffer number does not depend on where the stock went
# ---------------------------------------------------------------------------
# three days: the stock falls 3%, sits still, and rises 3%. Same buffer, same
# entry price, same exit book shape -- only the level differs.
# the results of each scenario
ex_values = []
raw_values = []
# the exit price achieved in each, needed for the fee arithmetic below
vwaps = []
# entry at 300, and closing books centred 3% below, flat, and 3% above
for close_bid in (291.00, 300.00, 309.00):
    # a fresh engine each time
    b = build(500.0, close_bid, 5000)
    # OVERRIDE THE ENTRY: buy at 300.00 regardless of where the day closes,
    # which is what makes this a test of the price move and nothing else
    b.book.o["B1"] = Order("BUY", close_bid, 5000)
    b.book.o["A1"] = Order("SELL", round(close_bid + 0.02, 2), 5000)
    # the first print sets the acquisition price
    first_print(b, 300.00)
    # the end-of-day arithmetic
    raw, ex, unfilled, vwap, mid = eod_numbers(b)
    # the exit must be clean or the comparison is contaminated by the haircut
    check(f"close {close_bid:.2f}: book absorbed the whole buffer",
          unfilled == 0)
    # keep the numbers
    raw_values.append(raw)
    ex_values.append(ex)
    vwaps.append(vwap)

# the raw numbers MUST differ -- that is the problem being corrected
check("as-run P&L swings with the stock (the problem)",
      max(raw_values) - min(raw_values) > 1000.0)
# THE RESIDUAL IS NOT EXACTLY ZERO, AND SHOULD NOT BE. PSX fees are
# ad-valorem: the exit fee is a percentage of the exit price, so selling the
# buffer at 309 costs marginally more in fees than selling it at 291. That
# difference is a REAL cost that genuinely depends on the price, so removing
# it would be removing something true. What must be gone is the first-order
# term, the Q x (price move) itself.
# the exact difference the ad-valorem fee accounts for, high scenario vs low
fee_gap = fee_for(vwaps[-1], 500.0) - fee_for(vwaps[0], 500.0)
# the ex-buffer spread must be exactly that and nothing more
check("ex-buffer spread == the ad-valorem fee difference, exactly",
      abs((max(ex_values) - min(ex_values)) - fee_gap) < 1e-6)
# and it must be negligible beside the raw swing it replaced
check("ex-buffer spread is <0.1% of the as-run swing",
      (max(ex_values) - min(ex_values))
      < 0.001 * (max(raw_values) - min(raw_values)))
# show both, so the size of the correction is visible in the run output
print(f"      as-run swing over a +/-3% move: "
      f"{max(raw_values) - min(raw_values):,.2f} PKR")
print(f"      ex-buffer swing over the same:  "
      f"{max(ex_values) - min(ex_values):,.2f} PKR (all of it ad-valorem fee)")

# ---------------------------------------------------------------------------
# 3. what is LEFT in the ex-buffer number is exactly the carrying cost
# ---------------------------------------------------------------------------
# a single clean case: 500 shares bought at 300.00, book closing around 300.00
b = build(500.0, 300.00, 5000)
# buy the buffer at 300.00
first_print(b, 300.00)
# the numbers
raw, ex, unfilled, vwap, mid = eod_numbers(b)
# the three components, each computed independently of the engine's answer:
#   the fee paid on the way in
fee_in = fee_for(300.00, 500.0)
#   the fee paid on the way out, at the price actually achieved
fee_out = fee_for(vwap, 500.0)
#   the spread crossed on the way out: we sold at the bid, not at the mid
cross = 500.0 * (mid - vwap)
# the ex-buffer number must be the negative of those three, and nothing else
check("ex-buffer P&L == -(entry fee + exit fee + spread crossed)",
      abs(ex - (-fee_in - fee_out - cross)) < 1e-6)
# and every component must be a COST, never a credit
check("all three components are costs",
      fee_in > 0 and fee_out > 0 and cross > 0)
# say what they are, so the run is readable without opening the file
print(f"      entry fee {fee_in:,.2f} + exit fee {fee_out:,.2f} + spread "
      f"{cross:,.2f} = {-ex:,.2f} PKR per day, on a "
      f"{500.0 * 300.00:,.0f} PKR buffer")
print(f"      A BUFFER THAT IS ACTUALLY HELD PAYS THIS ONCE, not daily.")

# ---------------------------------------------------------------------------
# 4. an arm with no buffer is untouched
# ---------------------------------------------------------------------------
# no opening inventory: the original behaviour
b = build(0.0, 300.00, 5000)
# a print goes by
first_print(b, 300.00)
# nothing was bought, so nothing was paid
check("no buffer -> cash untouched", b.cash == 0.0)
# and there is no acquisition price to record
check("no buffer -> buffer_qty 0 and buffer_px None",
      b.buffer_qty == 0.0 and b.buffer_px is None)

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
# blank line before the total
print()
# every case that did not pass
failed = [name for name, passed in results if not passed]
# the headline
print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
# name them, so a failing run says what broke without scrolling
for name in failed:
    print("  FAILED:", name)
# non-zero exit on any failure, so this can gate a run
sys.exit(1 if failed else 0)
