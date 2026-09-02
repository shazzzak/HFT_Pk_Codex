# test_itch_book.py
# Synthetic verification of the parser, book state machine, and queue fill sim.
# No real ITCH data needed: we hand-build byte messages with the exact formats.

# Import struct to pack synthetic messages exactly like the real feed.
import struct
# Pull in the components under test.
from itch_book import parse_message, OrderBook, QueueTracker, PRICE_SCALE


def hdr(mtype, ts=0, locate=1):
    # Build the 11-byte common header: type(1) locate(2) tracking(2) ts(6).
    return mtype + struct.pack(">HH", locate, 0) + ts.to_bytes(6, "big")


def add_msg(ref, side, shares, stock, price, ts=0):
    # Compose an 'A' Add Order message end to end.
    return hdr(b"A", ts) + struct.pack(">QcI8sI", ref, side.encode(),
                                        shares, stock.encode().ljust(8),
                                        int(round(price * PRICE_SCALE)))


def exec_msg(ref, shares, match, ts=0):
    # Compose an 'E' Order Executed message.
    return hdr(b"E", ts) + struct.pack(">QIQ", ref, shares, match)


def cancel_msg(ref, shares, ts=0):
    # Compose an 'X' Order Cancel (partial) message.
    return hdr(b"X", ts) + struct.pack(">QI", ref, shares)


def delete_msg(ref, ts=0):
    # Compose a 'D' Order Delete message.
    return hdr(b"D", ts) + struct.pack(">Q", ref)


def replace_msg(orig, new, shares, price, ts=0):
    # Compose a 'U' Order Replace message.
    return hdr(b"U", ts) + struct.pack(">QQII", orig, new, shares,
                                        int(round(price * PRICE_SCALE)))


def drive(book, raw):
    # Parse one raw message and apply it to the book (mirrors the real engine).
    m = parse_message(raw)
    # Ignore message types the parser skips.
    if m is None:
        return
    # Route by type into the book's handlers.
    t = m["type"]
    if t == "A":
        book.add(m["ref"], m["side"], m["price"], m["shares"])
    elif t == "E":
        book.execute(m["ref"], m["exec_shares"])
    elif t == "C":
        book.execute_with_price(m["ref"], m["exec_shares"])
    elif t == "X":
        book.cancel(m["ref"], m["cancelled_shares"])
    elif t == "D":
        book.delete(m["ref"])
    elif t == "U":
        book.replace(m["orig_ref"], m["new_ref"], m["shares"], m["price"])


# --- Test 1: basic add / best prices / aggregation --------------------------
b = OrderBook()
# Two bids at different prices and one ask.
drive(b, add_msg(1, "B", 100, "AAPL", 150.00))
drive(b, add_msg(2, "B", 200, "AAPL", 150.01))
drive(b, add_msg(3, "S", 300, "AAPL", 150.05))
# Best bid should be the higher price (150.01), best ask 150.05.
assert b.best_bid() == 150.01, b.best_bid()
assert b.best_ask() == 150.05, b.best_ask()
# Aggregated size at 150.01 must equal the single order's 200 shares.
assert b.bids[150.01] == 200
print("Test 1 passed: add / best bid-ask / aggregation")

# --- Test 2: partial execution then delete ---------------------------------
# Execute 50 of order 2 (200 -> 150 remaining).
drive(b, exec_msg(2, 50, match=1001))
assert b.bids[150.01] == 150, b.bids.get(150.01)
assert b.orders[2]["shares"] == 150
# Delete order 2 entirely; its price level should disappear.
drive(b, delete_msg(2))
assert 150.01 not in b.bids
# Best bid falls back to the remaining 150.00 order.
assert b.best_bid() == 150.00
print("Test 2 passed: partial execute + delete + level cleanup")

# --- Test 3: replace loses priority and moves size -------------------------
# Replace order 3 (ask 150.05 x300) with a new ref at 150.04 x300.
drive(b, replace_msg(3, 4, 300, 150.04))
# Old level gone, new level present, old ref removed, new ref present.
assert 150.05 not in b.asks
assert b.asks[150.04] == 300
assert 3 not in b.orders and 4 in b.orders
print("Test 3 passed: replace = delete + re-add at new level")

# --- Test 4: FIFO queue fill simulation ------------------------------------
# Fresh book with a queue of real orders at the same ask price we will join.
b2 = OrderBook()
# Real resting asks at 150.10: 500 then 300 shares (800 ahead in total).
drive(b2, add_msg(10, "S", 500, "AAPL", 150.10))
drive(b2, add_msg(11, "S", 300, "AAPL", 150.10))
# We place OUR passive sell of 200 @150.10 -> 800 shares ahead of us.
b2.sim = QueueTracker("S", 150.10, 200, b2)
assert b2.sim.shares_ahead == 800, b2.sim.shares_ahead
# A real add of 400 @150.10 lands BEHIND us (must not change shares_ahead).
drive(b2, add_msg(12, "S", 400, "AAPL", 150.10))
assert b2.sim.shares_ahead == 800
# Executions drain the 800 ahead of us: 500 (order10) then 300 (order11).
drive(b2, exec_msg(10, 500, match=2001))
drive(b2, exec_msg(11, 300, match=2002))
# Queue ahead is now 0 but we have NOT been filled yet.
assert b2.sim.shares_ahead == 0
assert b2.sim.remaining == 200
# Next execution (against the order behind us) now fills US for up to its qty.
# 150 shares trade -> we fill 150, 50 left.
drive(b2, exec_msg(12, 150, match=2003))
assert b2.sim.remaining == 50, b2.sim.remaining
assert b2.sim.fills == [(150.10, 150)], b2.sim.fills
# Another 100-share trade -> we fill our remaining 50 (min(50,100)).
drive(b2, exec_msg(12, 100, match=2004))
assert b2.sim.remaining == 0
assert b2.sim.fills[-1] == (150.10, 50)
print("Test 4 passed: FIFO queue-position fill model")

# --- Test 5: cancels AHEAD improve our position (but never fill us) ---------
b3 = OrderBook()
# 1000 shares resting ahead at a bid we will join.
drive(b3, add_msg(20, "B", 1000, "AAPL", 149.90))
# We join with 100 shares; 1000 ahead.
b3.sim = QueueTracker("B", 149.90, 100, b3)
assert b3.sim.shares_ahead == 1000
# A 600-share CANCEL ahead of us frees queue but is not a fill.
drive(b3, cancel_msg(20, 600))
assert b3.sim.shares_ahead == 400
assert b3.sim.remaining == 100  # unchanged: cancels don't fill
# Delete the rest (400) -> queue ahead now 0, still no fill.
drive(b3, delete_msg(20))
assert b3.sim.shares_ahead == 0
assert b3.sim.remaining == 100
print("Test 5 passed: cancels/deletes ahead free queue without filling us")

print("\nALL TESTS PASSED")
