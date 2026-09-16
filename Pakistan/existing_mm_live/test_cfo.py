# ============================================================================
# test_cfo.py -- does the Change Former Order path in mm_backtest do what PSX
# Regulations 8.5.2 says an amendment does?
# ============================================================================
# READ-ONLY. Touches no data, writes no file, needs no parsed store. It builds
# a book by hand, drives the engine one message at a time, and checks what
# happened to queue position.
#
# WHY A PLAIN SCRIPT AND NOT PYTEST. It has to run against the REAL Backtester,
# which needs numpy and pandas; a harness tested against a mock of the thing it
# integrates with proves nothing about the integration. Plain python keeps the
# dependency list to what the backtest already has.
#
# Run from existing_mm_live/:
#   caffeinate -is python test_cfo.py
# Exit code 0 = every case passed, 1 = something is wrong.
# ============================================================================

# argv/exit only
import sys

# the engine under test, its latency model, and the Order record the Book holds
from mm_backtest import Backtester, LatencyModel, Order


class Stub:
    """A MicrostructureMM-shaped stand-in with a scripted answer.

    The real strategy is nine hundred lines. Using it here would mean a failure
    could be the CFO path OR the quoting logic, and this file is only about the
    CFO path.
    """
    # the engine reads .tick off the strategy; PSX is a flat one paisa
    tick = 0.01

    def __init__(self, returns=None):
        # what quotes() will hand back, set per test
        self._r = returns or {}

    def observe(self, *args):
        # the engine calls this on every event; nothing here needs the state
        return None

    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # whatever the test scripted, ignoring the book entirely
        return self._r


def build(**cfg_overrides):
    """An engine with a hand-built book, ready to be driven message by message."""
    # CONSTANT LATENCY: every draw returns exactly 100 ms, so a test can say
    # precisely when a message lands. tail_prob=0 means the spike branch never
    # fires, which is what makes the timing deterministic.
    cfg = dict(
        # the model, with every source of randomness zeroed
        latency_model=LatencyModel(decision_ms=0.0, wire_out_median_ms=100.0,
                                   wire_out_tail_ms=0.0,
                                   wire_in_median_ms=100.0,
                                   wire_in_tail_ms=0.0, tail_prob=0.0),
        # exact queue consumption, the mode the shipped runs used
        at_price_mode="queue",
        # the optimistic crossing-add fill stays off, as in production
        fill_on_crossing_adds=False,
        # the per-event equity curve is not needed and is the slow path
        log_equity=False,
        # a session wide enough that no end-of-day logic ever engages
        session=(0, 10 ** 9))
    # whatever this particular test wants to change (use_cfo, the venue flags)
    cfg.update(cfg_overrides)
    # a strategy that wants 50 shares bid at 289.00
    strategy = Stub({"BUY": (289.00, 50)})
    # the engine
    bt = Backtester(strategy, cfg)
    # BUILD THE BOOK BY HAND. Book keeps ONE dict, order_id -> Order, and
    # derives price levels on demand -- that order-level state is exactly what
    # makes the queue tracking exact, so the test builds it the same way.
    bt.book.o.clear()
    # 500 shares resting at our bid price: this is the queue we will sit behind
    bt.book.o["H1"] = Order("BUY", 289.00, 500)
    # something on the offer, so the book is two-sided and quotable
    bt.book.o["H2"] = Order("SELL", 289.02, 300)
    # continuous trading, the only phase _requote will quote in
    bt.book.phase = "CONTINUOUS_AUCTION"
    # no circuit limits published, so nothing is clamped or pinned
    bt.book.limit_up = None
    bt.book.limit_dn = None
    # the engine and the strategy, both of which the tests drive directly
    return bt, strategy


def rest(bt):
    """One requote cycle, then land the order, so something is working."""
    # ask the strategy and send whatever it wants
    bt._requote(1000)
    # advance past the 100 ms send latency so the order actually rests
    bt._activate_until(2000)


# every (name, passed?) pair, so the run can report a total rather than dying
# on the first failure -- a partial picture of what broke is worth more
results = []


def check(name, condition):
    """Record one assertion and print it as it happens."""
    # keep it for the summary
    results.append((name, condition))
    # and show it immediately, so a hang is attributable to a case
    print(("PASS  " if condition else "FAIL  ") + name)


# ---------------------------------------------------------------------------
# 1. use_cfo=False must be byte-identical to the original engine
# ---------------------------------------------------------------------------
# the default configuration, with the CFO path off
bt, strategy = build()
# get a bid resting
rest(bt)
# now want a different price, which forces a reprice
strategy._r = {"BUY": (288.99, 50)}
# one requote cycle produces whatever messages the reprice needs
bt._requote(3000)
# TWO messages, a cancel and a new order, as the engine has always done
check("use_cfo=False still sends CANCEL + ARRIVE (2 msgs)",
      sorted(a for _, _, a, _ in bt.pending) == ["ARRIVE", "CANCEL"])

# ---------------------------------------------------------------------------
# 2. use_cfo=True sends exactly one message, and the old terms stay live
# ---------------------------------------------------------------------------
# the same setup with the CFO path switched on
bt, strategy = build(use_cfo=True)
# get a bid resting
rest(bt)
# remember which generation is resting and what is in front of it
old_oid = bt.work["BUY"].oid
old_ahead = dict(bt.work["BUY"].ahead)
# the 500 shares from H1 should be ahead of us
check("order rested with 500 ahead of it", sum(old_ahead.values()) == 500)
# want a different price
strategy._r = {"BUY": (288.99, 50)}
# one requote cycle
bt._requote(3000)
# ONE message this time, not two
check("use_cfo=True sends exactly ONE message", len(bt.pending) == 1)
# and it is an amendment rather than a cancel
check("and it is an AMEND", bt.pending[0][2] == "AMEND")
# THE REAL EXPOSURE OF AN AMENDMENT: until it lands, the OLD price is still
# resting and still fillable. Nothing has been cancelled.
check("old order still live at the OLD price until it lands",
      bt.work["BUY"].price == 289.00)

# ---------------------------------------------------------------------------
# 3. a price change LOSES priority -- PSX Regulations 8.5.2
# ---------------------------------------------------------------------------
# put a queue at the price we are amending TO, so losing priority is visible
bt.book.o["H3"] = Order("BUY", 288.99, 700)
# land the amendment
bt._activate_until(4000)
# we joined the back of the line at the new price: all 700 are in front
check("price change re-queued behind the 700 at the new price",
      sum(bt.work["BUY"].ahead.values()) == 700)
# and the amended terms are the ones now resting
check("and the new terms are live", bt.work["BUY"].price == 288.99)
# the counters should say one amendment landed and none kept its place
check("counted as a CFO that did NOT keep priority",
      bt.stats.get("n_cfos") == 1 and bt.stats.get("n_cfos_kept_priority", 0) == 0)

# ---------------------------------------------------------------------------
# 4. a quantity REDUCTION keeps priority -- the carve-out in 8.5.2
# ---------------------------------------------------------------------------
# a fresh engine with the CFO path on
bt, strategy = build(use_cfo=True)
# get a bid resting
rest(bt)
# remember the queue state we expect to survive the amendment
keep_ahead = dict(bt.work["BUY"].ahead)
keep_t_active = bt.work["BUY"].t_active
# SAME price, SMALLER size -- the one case 8.5.2 carves out
strategy._r = {"BUY": (289.00, 20)}
# send it and land it
bt._requote(3000)
bt._activate_until(4000)
# the queue in front of us is untouched, and so is our join time
check("size reduction KEPT its place in the queue",
      bt.work["BUY"].ahead == keep_ahead
      and bt.work["BUY"].t_active == keep_t_active)
# while the size itself did change
check("with the reduced size applied", bt.work["BUY"].qty == 20)
# and the counter records that this one held its place
check("counted as keeping priority", bt.stats.get("n_cfos_kept_priority") == 1)

# ---------------------------------------------------------------------------
# 5. a quantity INCREASE loses priority -- only reduction is carved out
# ---------------------------------------------------------------------------
# a fresh engine
bt, strategy = build(use_cfo=True)
# get a bid resting
rest(bt)
# SAME price, BIGGER size
strategy._r = {"BUY": (289.00, 90)}
# send the amendment
bt._requote(3000)
# the queue at our price grows while the message is in flight, so the
# re-snapshot has to be taken when it LANDS, not when it was sent
bt.book.o["H9"] = Order("BUY", 289.00, 111)
# land it
bt._activate_until(4000)
# 500 + 111 are now in front of us: we rejoined at the back, at landing time
check("size increase re-queued at the current depth",
      sum(bt.work["BUY"].ahead.values()) == 611)
# and the counter agrees nothing kept its place
check("counted as NOT keeping priority",
      bt.stats.get("n_cfos_kept_priority", 0) == 0)

# ---------------------------------------------------------------------------
# 6. the venue switch: an exchange that DOES hold priority on a reprice
# ---------------------------------------------------------------------------
# the same engine with one flag flipped -- this is the whole portability story
bt, strategy = build(use_cfo=True, cfo_price_keeps_priority=True)
# get a bid resting
rest(bt)
# remember the queue we expect to keep even across a price change
keep_ahead = dict(bt.work["BUY"].ahead)
# change the price
strategy._r = {"BUY": (288.99, 50)}
# send it
bt._requote(3000)
# a queue exists at the new price, and on THIS venue it does not matter
bt.book.o["H3"] = Order("BUY", 288.99, 700)
# land the amendment
bt._activate_until(4000)
# we carried our place across the reprice, and the new price is live
check("cfo_price_keeps_priority=True -> reprice HOLDS its place",
      bt.work["BUY"].ahead == keep_ahead and bt.work["BUY"].price == 288.99)

# ---------------------------------------------------------------------------
# 7. a CFO that arrives after the order already filled is rejected
# ---------------------------------------------------------------------------
# a fresh engine
bt, strategy = build(use_cfo=True)
# get a bid resting
rest(bt)
# decide to reprice
strategy._r = {"BUY": (288.99, 50)}
# send the amendment
bt._requote(3000)
# the order fills completely BEFORE the amendment lands -- the race a real
# exchange answers with an Order Cancel Reject
bt._fill("BUY", 289.00, 50, 3500, "through")
# land the (now pointless) amendment
bt._activate_until(4000)
# nothing was resurrected, and the rejection was counted rather than silent
check("CFO on a filled order is ignored and counted",
      "BUY" not in bt.work and bt.stats.get("stale_cfos_ignored") == 1)

# ---------------------------------------------------------------------------
# 8. pulling a side entirely is a Cancel Order (8.11), never a CFO
# ---------------------------------------------------------------------------
# a fresh engine with the CFO path on
bt, strategy = build(use_cfo=True)
# get a bid resting
rest(bt)
# the strategy now wants nothing on either side
strategy._r = {}
# one requote cycle
bt._requote(3000)
# a CFO modifies an order; removing one is a Cancel, and 8.11/8.12 keep them
# separate, so the CFO path must not swallow this case
check("pulling a side uses CANCEL even with use_cfo on",
      len(bt.pending) == 1 and bt.pending[0][2] == "CANCEL")

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
