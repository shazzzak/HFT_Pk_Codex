# ============================================================================
# test_short_policy.py -- do the four short-sale modes behave as PSX Chapter 10
# says they must?
# ============================================================================
# READ-ONLY. No data, no parsed store. Builds a book by hand and drives the
# engine one message at a time.
#
# Run from existing_mm_live/:
#   caffeinate -is python test_short_policy.py
# Exit code 0 = every case passed, 1 = something is wrong.
# ============================================================================

# argv/exit only
import sys

# the engine, its latency model, and the Order record the Book holds
from mm_backtest import Backtester, LatencyModel, Order
# the real strategy, because the quoting half of the policy lives in it
from micro_mm import MicrostructureMM


def make_strategy(policy, eligible=False):
    """A real MicrostructureMM under one short-sale policy."""
    # session_scale is required and keyword-only; this value is PPL's derived
    # one and is irrelevant to what these tests check
    return MicrostructureMM(session_ms=(0, 10 ** 9), session_scale=7.6,
                            fee_pct=0.0000777, size=50, max_inv=500,
                            short_policy=policy, slb_eligible=eligible)


def build(policy="unrestricted", eligible=False, opening=0.0, **overrides):
    """An engine plus a real strategy, ready to be driven by hand."""
    # constant latency so a test can say exactly when a message lands
    cfg = dict(
        # every source of randomness zeroed
        latency_model=LatencyModel(decision_ms=0.0, wire_out_median_ms=100.0,
                                   wire_out_tail_ms=0.0,
                                   wire_in_median_ms=100.0,
                                   wire_in_tail_ms=0.0, tail_prob=0.0),
        # the mode the shipped runs used
        at_price_mode="queue",
        # the optimistic crossing-add fill stays off
        fill_on_crossing_adds=False,
        # the per-event equity curve is the slow path and is not needed
        log_equity=False,
        # a session wide enough that no end-of-day logic engages
        session=(0, 10 ** 9),
        # the long_buffer policy's opening position
        opening_inventory=opening)
    # whatever this test wants to change
    cfg.update(overrides)
    # the strategy under the policy being tested
    strategy = make_strategy(policy, eligible)
    # the engine
    bt = Backtester(strategy, cfg)
    # build a two-sided book by hand: Book holds order_id -> Order
    bt.book.o.clear()
    # a bid with depth
    bt.book.o["H1"] = Order("BUY", 289.00, 500)
    # an offer with depth
    bt.book.o["H2"] = Order("SELL", 289.50, 300)
    # continuous trading, the only phase _requote quotes in
    bt.book.phase = "CONTINUOUS_AUCTION"
    # no circuit limits published
    bt.book.limit_up = None
    bt.book.limit_dn = None
    # both, because the tests drive each directly
    return bt, strategy


# every (name, passed?) pair, so the run reports a total rather than dying first
results = []


def check(name, condition):
    """Record one assertion and print it as it happens."""
    # keep it for the summary
    results.append((name, condition))
    # and show it immediately
    print(("PASS  " if condition else "FAIL  ") + name)


def sides_quoted(bt, pos):
    """Which sides the strategy wants, at a given position."""
    # the touch, as _requote reads it
    bb, bq, ba, aq = bt.book.bbo()
    # ask the strategy directly, bypassing the message machinery
    return set(bt.strat.quotes(bb, bq, ba, aq, pos))


def ask_size(bt, pos):
    """The size the strategy wants on the offer, or None if it wants none."""
    # the touch
    bb, bq, ba, aq = bt.book.bbo()
    # what it wants
    want = bt.strat.quotes(bb, bq, ba, aq, pos)
    # the offer's size, if there is an offer at all
    return want["SELL"][1] if "SELL" in want else None


# ---------------------------------------------------------------------------
# 1. unrestricted -- what was measured, and it must be unchanged
# ---------------------------------------------------------------------------
# the default policy
bt, strategy = build("unrestricted")
# flat: it quotes both sides, which is exactly the behaviour that may not be
# permissible under 10.15 and is why the other three modes exist
check("unrestricted quotes BOTH sides when flat",
      sides_quoted(bt, 0) == {"BUY", "SELL"})
# already short: it keeps offering, going further short
check("unrestricted still offers when already short",
      "SELL" in sides_quoted(bt, -200))

# ---------------------------------------------------------------------------
# 2. no_short -- never sell what we do not hold (10.15)
# ---------------------------------------------------------------------------
# the never-go-short policy
bt, strategy = build("no_short")
# flat: no offer at all, because every share sold would be a Blank Sale
check("no_short quotes NO ask when flat", sides_quoted(bt, 0) == {"BUY"})
# short: still no offer, and it must not be adding to the short
check("no_short quotes NO ask when short", sides_quoted(bt, -100) == {"BUY"})
# holding more than a clip: the ask is quoted at the normal size
check("no_short offers a full clip when holding plenty",
      ask_size(bt, 500) == 50)
# holding LESS than a clip: the ask is capped at what we actually own
check("no_short caps the ask at the position when holding less than a clip",
      ask_size(bt, 20) == 20)

# ---------------------------------------------------------------------------
# 3. long_buffer -- same quoting rule, plus an opening position
# ---------------------------------------------------------------------------
# a run that starts holding 1,000 shares
# 200 shares: deliberately INSIDE max_inv (500), because a buffer above the
# cap would stop the bid for a completely unrelated reason -- the inventory
# limit -- and the test would be measuring that instead
bt, strategy = build("long_buffer", opening=200.0)
# the engine starts long rather than flat
check("long_buffer starts the run with the opening inventory",
      bt.pos == 200.0)
# and it quotes both sides, because the buffer means a sale is a real sale
check("long_buffer quotes BOTH sides while the buffer is intact",
      sides_quoted(bt, bt.pos) == {"BUY", "SELL"})
# run the buffer down to nothing and the ask disappears, exactly as no_short
check("long_buffer stops offering once the buffer is gone",
      sides_quoted(bt, 0) == {"BUY"})

# ---------------------------------------------------------------------------
# 4. slb_uptick on a name that is NOT SLB-eligible -> falls back to no_short
# ---------------------------------------------------------------------------
# 10.17 allows short selling only in Category A SLB-eligible securities
bt, strategy = build("slb_uptick", eligible=False)
# an ineligible name cannot be short sold at all, so it behaves as no_short
check("slb_uptick on an INELIGIBLE name quotes no ask when flat",
      sides_quoted(bt, 0) == {"BUY"})
# and the engine does not arm the uptick gate, because there is nothing to gate
check("slb_uptick on an INELIGIBLE name does not arm the uptick gate",
      bt.enforce_uptick is False)

# ---------------------------------------------------------------------------
# 5. slb_uptick on an ELIGIBLE name -- quotes freely, gated at execution
# ---------------------------------------------------------------------------
# an eligible name may be short sold
bt, strategy = build("slb_uptick", eligible=True)
# so it quotes both sides from flat, like unrestricted
check("slb_uptick on an ELIGIBLE name quotes BOTH sides when flat",
      sides_quoted(bt, 0) == {"BUY", "SELL"})
# and the gate is armed, because the constraint now applies at fill time
check("slb_uptick on an ELIGIBLE name arms the uptick gate",
      bt.enforce_uptick is True)


class Trade:
    """The shape _on_market_trade reads off a tape row."""
    # only the fields the handler touches
    def __init__(self, px, qty, aggressor, ts):
        # the print price
        self.price = px
        # the printed size
        self.qty = qty
        # which side took liquidity
        self.aggressor_side = aggressor
        # continuous market, not an auction
        self.initiator = "BROKER"
        # exchange time
        self.ts_exch = ts
        # no resting victim named -- the handler reads rest_oid, and a real
        # tape row carries it; None means "the print did not name whose order
        # it consumed", which is the common case
        self.resting_order_id = None
        self.rest_oid = None


def rest_an_ask(bt, price=289.20, qty=50):
    """Put one of OUR asks into the book, live and ready to be hit."""
    # the engine's own order record, arriving at a known time
    from mm_backtest import MyOrder
    # give it an id and no queue ahead of it, so a print at its price fills it
    bt._oid += 1
    # live immediately, with an empty ahead pool = first in line
    bt.work["SELL"] = MyOrder("SELL", price, qty, {}, 0, oid=bt._oid)
    # the lifecycle record the engine expects to exist
    bt._olog[bt._oid] = {"oid": bt._oid, "side": "SELL", "px": price,
                         "qty": qty, "t_sent": 0, "t_live": 0,
                         "t_end": None, "end_reason": None}


# ---------------------------------------------------------------------------
# 6. a short-taking fill on a DOWNTICK is blocked (10.16.1(a))
# ---------------------------------------------------------------------------
# an eligible name with the gate armed
bt, strategy = build("slb_uptick", eligible=True)
# ESTABLISH THE TICK HISTORY FIRST, with no ask resting. Prints above our
# eventual ask price would go THROUGH it and fill us, so resting it early
# would count blocks from trades this case is not about.
bt._on_market_trade(Trade(289.50, 10, "BUY", 1000))
bt._on_market_trade(Trade(289.30, 10, "BUY", 1100))
# now put our ask at 289.20, below the last executed price of 289.30
rest_an_ask(bt, 289.20, 50)
# flat, so any sale takes us short
bt.pos = 0.0
# a buyer lifts through it -- but OUR sale executes at 289.20, below the last
# executed price, which is a downtick
bt._on_market_trade(Trade(289.25, 40, "BUY", 1200))
# the short did not happen
check("short-taking fill BLOCKED on a downtick",
      bt.pos == 0.0 and bt.stats["short_fills_blocked_by_uptick"] == 1)
# and our order is still resting, untouched
check("the blocked order stays resting", bt.work["SELL"].qty == 50)

# ---------------------------------------------------------------------------
# 7. the same fill on an UPTICK goes through
# ---------------------------------------------------------------------------
# a fresh engine, gate armed
bt, strategy = build("slb_uptick", eligible=True)
# the last executed price is BELOW our ask, so selling at 289.20 is an uptick
bt._on_market_trade(Trade(289.10, 10, "BUY", 1000))
# now rest the ask
rest_an_ask(bt, 289.20, 50)
# flat
bt.pos = 0.0
# a buyer lifts us
bt._on_market_trade(Trade(289.20, 40, "BUY", 1100))
# the short executed
check("short-taking fill ALLOWED on an uptick",
      bt.pos == -40.0 and bt.stats["short_fills_blocked_by_uptick"] == 0)

# ---------------------------------------------------------------------------
# 8. a ZERO-PLUS tick is allowed; a zero tick after a DOWN move is not
# ---------------------------------------------------------------------------
# zero-plus: same price as the last, where the last move was UP
bt, strategy = build("slb_uptick", eligible=True)
# establish the upward move before resting anything
bt._on_market_trade(Trade(289.10, 5, "BUY", 1000))
bt._on_market_trade(Trade(289.20, 5, "BUY", 1100))
# then rest the ask at that same price
rest_an_ask(bt, 289.20, 50)
bt.pos = 0.0
# another print at the same 289.20: our sale is a zero tick, and the last move
# was up, so it is a Zero-Plus Tick
bt._on_market_trade(Trade(289.20, 40, "BUY", 1200))
check("short-taking fill ALLOWED on a zero-PLUS tick", bt.pos == -40.0)

# the mirror: the same equal price, but the last move was DOWN
bt, strategy = build("slb_uptick", eligible=True)
# establish the downward move before resting anything
bt._on_market_trade(Trade(289.30, 5, "BUY", 1000))
bt._on_market_trade(Trade(289.20, 5, "BUY", 1100))
# then rest the ask at that same price
rest_an_ask(bt, 289.20, 50)
bt.pos = 0.0
# a print at the same price is now a zero-MINUS tick, and is not permitted
bt._on_market_trade(Trade(289.20, 40, "BUY", 1200))
check("short-taking fill BLOCKED on a zero tick after a DOWN move",
      bt.pos == 0.0 and bt.stats["short_fills_blocked_by_uptick"] == 1)

# ---------------------------------------------------------------------------
# 9. selling stock we OWN is not a short sale and is never gated
# ---------------------------------------------------------------------------
# gate armed, but we are long
bt, strategy = build("slb_uptick", eligible=True)
# establish a downtick first, which would block a SHORT sale
bt._on_market_trade(Trade(289.50, 5, "BUY", 1000))
bt._on_market_trade(Trade(289.30, 5, "BUY", 1100))
# then rest the ask
rest_an_ask(bt, 289.20, 50)
# holding 200 shares, so a 40-share sale leaves us long
bt.pos = 200.0
# and get hit
bt._on_market_trade(Trade(289.25, 40, "BUY", 1200))
# an ordinary sale of owned stock: the uptick rule has nothing to say about it
check("selling stock we OWN is never blocked, even on a downtick",
      bt.pos == 160.0 and bt.stats["short_fills_blocked_by_uptick"] == 0)

# ---------------------------------------------------------------------------
# 10. before the first print there is no reference, so no short is permitted
# ---------------------------------------------------------------------------
# gate armed, tape empty
bt, strategy = build("slb_uptick", eligible=True)
rest_an_ask(bt, 289.20, 50)
bt.pos = 0.0
# the very first print of the day hits us
bt._on_market_trade(Trade(289.20, 40, "BUY", 1000))
# NOT TOLD IS NOT PERMISSION: with no previous executed price neither the
# uptick nor the zero-plus test can pass
check("no short before the first print establishes a reference price",
      bt.pos == 0.0 and bt.stats["short_fills_blocked_by_uptick"] == 1)

# ---------------------------------------------------------------------------
# 11. unrestricted ignores the tick entirely -- the measured behaviour
# ---------------------------------------------------------------------------
# the original policy
bt, strategy = build("unrestricted")
# a clear downtick, established before anything rests
bt._on_market_trade(Trade(289.50, 5, "BUY", 1000))
bt._on_market_trade(Trade(289.30, 5, "BUY", 1100))
# then rest the ask
rest_an_ask(bt, 289.20, 50)
bt.pos = 0.0
# and we are hit
bt._on_market_trade(Trade(289.25, 40, "BUY", 1200))
# the short happens, because no gate is armed
check("unrestricted shorts on a downtick, unchanged from the original engine",
      bt.pos == -40.0)

# ---------------------------------------------------------------------------
# 12. an unknown policy is refused at construction, not at run time
# ---------------------------------------------------------------------------
# a typo in the policy name
try:
    # this must not build
    make_strategy("no_shorts")
    # reaching here means it did
    refused = False
except ValueError:
    # refused, which is the point
    refused = True
check("an unknown short_policy is refused at construction", refused)

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
# blank line before the total
print()
# every case that did not pass
failed = [name for name, passed in results if not passed]
# the headline
print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
# name them
for name in failed:
    print("  FAILED:", name)
# non-zero exit on any failure
sys.exit(1 if failed else 0)
