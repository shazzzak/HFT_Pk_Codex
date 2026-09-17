# ============================================================================
# test_crossed_book.py -- does the engine refuse to quote off an invalid book?
# ============================================================================
# A book whose best bid is at or above its best ask cannot rest at a
# continuously matching exchange -- the two orders would have traded. It shows
# up in the reconstruction, and until 2026-09-17 the engine quoted against it,
# pricing every quote off a mid that is not a mid.
#
# Run from existing_mm_live/:
#   caffeinate -is python test_crossed_book.py
# ============================================================================

# argv/exit only
import sys
# the engine, its latency model, and the Order record the Book holds
from mm_backtest import Backtester, LatencyModel, Order


class Stub:
    """A strategy that always wants to quote, so any refusal is the engine's."""
    # PSX is a flat one paisa
    tick = 0.01

    def observe(self, *args):
        # the engine calls this on every event; nothing here needs it
        return None

    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # always wants a bid, whatever the book looks like
        return {"BUY": (289.00, 50)}


def build(bid, ask, **over):
    """An engine with a hand-built two-sided book at the given touch."""
    # every source of randomness pinned
    cfg = dict(latency_model=LatencyModel(decision_ms=0.0,
                                          wire_out_median_ms=100.0,
                                          wire_out_tail_ms=0.0,
                                          wire_in_median_ms=100.0,
                                          wire_in_tail_ms=0.0, tail_prob=0.0),
               at_price_mode="queue", fill_on_crossing_adds=False,
               log_equity=False, session=(0, 10 ** 9))
    # whatever this case changes
    cfg.update(over)
    # the engine
    bt = Backtester(Stub(), cfg)
    # the book, built the way Book itself stores it
    bt.book.o.clear()
    bt.book.o["H1"] = Order("BUY", bid, 500)
    bt.book.o["H2"] = Order("SELL", ask, 300)
    # continuous trading, no published band
    bt.book.phase = "CONTINUOUS_AUCTION"
    bt.book.limit_up = None
    bt.book.limit_dn = None
    # ready to drive
    return bt


# every (name, passed?) pair
results = []


def check(name, condition):
    # keep it, and show it as it happens
    results.append((name, condition))
    print(("PASS  " if condition else "FAIL  ") + name)


# ---- a NORMAL book still quotes ------------------------------------------
bt = build(289.00, 289.02)
bt._requote(1000)
check("a normal book quotes", len(bt.pending) == 1)
check("and is not counted as crossed",
      bt.stats.get("crossed_book_requotes", 0) == 0)

# ---- a CROSSED book sends nothing ----------------------------------------
bt = build(289.05, 289.00)
bt._requote(1000)
check("a crossed book sends NOTHING", len(bt.pending) == 0)
check("and is counted", bt.stats.get("crossed_book_requotes") == 1)

# ---- a LOCKED book is treated the same -----------------------------------
bt = build(289.00, 289.00)
bt._requote(1000)
check("a locked book (bid == ask) sends nothing", len(bt.pending) == 0)
check("and is counted too", bt.stats.get("crossed_book_requotes") == 1)

# ---- an existing quote is PULLED when the book goes crossed --------------
bt = build(289.00, 289.02)
bt._requote(1000)
bt._activate_until(2000)
check("an order is resting to begin with", "BUY" in bt.work)
# the book crosses under us
bt.book.o["H1"] = Order("BUY", 289.05, 500)
bt._requote(3000)
check("the resting quote is pulled, not left exposed",
      len(bt.pending) == 1 and bt.pending[0][2] == "CANCEL")

# ---- the flag reproduces the old behaviour exactly ------------------------
bt = build(289.05, 289.00, skip_crossed_book=False)
bt._requote(1000)
check("skip_crossed_book=False quotes anyway, as every run before 2026-09-17",
      len(bt.pending) == 1)
check("and counts nothing", bt.stats.get("crossed_book_requotes", 0) == 0)

# ---- summary -------------------------------------------------------------
print()
failed = [n for n, ok in results if not ok]
print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
for n in failed:
    print("  FAILED:", n)
sys.exit(1 if failed else 0)
