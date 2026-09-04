# itch_book.py
# Nasdaq TotalView-ITCH 5.0 parser + full order-book (MBO/L3) reconstruction
# + FIFO queue-position passive-fill simulator for market-making backtests.
#
# Scope note (READ THIS):
#   This reconstructs ONE venue's book (Nasdaq). See notes at bottom for the
#   multi-venue / latency / economics additions required before trusting a
#   market-making backtest result.

# struct gives us big-endian binary unpacking; ITCH is big-endian throughout.
import struct
# We use a namedtuple-free plain-dict design for orders to keep it hackable.
from collections import defaultdict

# ITCH price fields are integers with 4 implied decimal places (price * 10000).
# We divide by this to recover a float price in dollars.
PRICE_SCALE = 10000.0

# ---------------------------------------------------------------------------
# LOW-LEVEL PARSING
# ---------------------------------------------------------------------------

# Pre-compiled struct objects for each message BODY (everything AFTER the
# 11-byte common header: type[1] + stock_locate[2] + tracking[2] + ts[6]).
# '>' = big-endian, no padding. Sizes are validated by the self-test.
_BODY = {
    # 'A' Add Order (no MPID): order_ref(Q) side(c) shares(I) stock(8s) price(I)
    b"A": struct.Struct(">QcI8sI"),
    # 'F' Add Order w/ MPID: same as A plus 4-char attribution (MPID)
    b"F": struct.Struct(">QcI8sI4s"),
    # 'E' Order Executed: order_ref(Q) executed_shares(I) match_number(Q)
    b"E": struct.Struct(">QIQ"),
    # 'C' Order Executed w/ Price: order_ref(Q) shares(I) match(Q) printable(c) exec_price(I)
    b"C": struct.Struct(">QIQcI"),
    # 'X' Order Cancel: order_ref(Q) cancelled_shares(I)
    b"X": struct.Struct(">QI"),
    # 'D' Order Delete: order_ref(Q)
    b"D": struct.Struct(">Q"),
    # 'U' Order Replace: orig_ref(Q) new_ref(Q) shares(I) price(I)
    b"U": struct.Struct(">QQII"),
    # 'P' Trade (non-cross, hidden liquidity): ref(Q) side(c) shares(I) stock(8s) price(I) match(Q)
    b"P": struct.Struct(">QcI8sIQ"),
    # 'Q' Cross Trade: shares(Q) stock(8s) cross_price(I) match(Q) cross_type(c)
    b"Q": struct.Struct(">Q8sIQc"),
}


def parse_message(msg):
    """Parse one raw ITCH 5.0 message (bytes, WITHOUT the 2-byte length prefix).

    Returns a plain dict with a 'type' key plus the decoded fields, or None for
    message types we don't need for book reconstruction.
    """
    # First byte is the message type as a single-byte bytes object, e.g. b'A'.
    mtype = msg[0:1]
    # Common header timestamp: 6 bytes at offset 5..10 = nanoseconds since midnight.
    ts = int.from_bytes(msg[5:11], "big")
    # stock_locate (offset 1..2) is the integer symbol id used by all messages.
    locate = int.from_bytes(msg[1:3], "big")

    # If this type isn't one we reconstruct from, skip it fast.
    if mtype not in _BODY:
        # We still surface 'R' (Stock Directory) so callers can map locate->ticker.
        if mtype == b"R":
            # Stock symbol sits right after the header: offset 11..18 (8 chars).
            stock = msg[11:19].decode("ascii", "replace").strip()
            # Return a directory record so the engine can build a symbol map.
            return {"type": "R", "ts": ts, "locate": locate, "stock": stock}
        # Everything else (S, H, Y, L, V, W, K, I, B, ...) is ignored here.
        return None

    # Unpack the body using the pre-compiled struct for this message type.
    body = _BODY[mtype].unpack(msg[11:])

    # Dispatch per type into a normalized dict. side is decoded to 'B'/'S'.
    if mtype == b"A":
        # Add order without attribution.
        return {"type": "A", "ts": ts, "locate": locate,
                "ref": body[0], "side": body[1].decode(),
                "shares": body[2], "stock": body[3].decode("ascii").strip(),
                "price": body[4] / PRICE_SCALE}
    if mtype == b"F":
        # Add order with attribution; MPID (body[5]) kept for toxicity tagging.
        return {"type": "A", "ts": ts, "locate": locate,
                "ref": body[0], "side": body[1].decode(),
                "shares": body[2], "stock": body[3].decode("ascii").strip(),
                "price": body[4] / PRICE_SCALE,
                "mpid": body[5].decode("ascii").strip()}
    if mtype == b"E":
        # Execution against a displayed order at that order's resting price.
        return {"type": "E", "ts": ts, "locate": locate,
                "ref": body[0], "exec_shares": body[1], "match": body[2]}
    if mtype == b"C":
        # Execution with an explicit print price (e.g. price improvement).
        # printable == b'Y' means the trade prints to the tape (counts as volume).
        return {"type": "C", "ts": ts, "locate": locate,
                "ref": body[0], "exec_shares": body[1], "match": body[2],
                "printable": body[3] == b"Y", "exec_price": body[4] / PRICE_SCALE}
    if mtype == b"X":
        # Partial cancel: some shares removed, order may remain.
        return {"type": "X", "ts": ts, "locate": locate,
                "ref": body[0], "cancelled_shares": body[1]}
    if mtype == b"D":
        # Full delete: the entire order leaves the book.
        return {"type": "D", "ts": ts, "locate": locate, "ref": body[0]}
    if mtype == b"U":
        # Replace = delete(orig) then add(new); ALWAYS loses time priority.
        return {"type": "U", "ts": ts, "locate": locate,
                "orig_ref": body[0], "new_ref": body[1],
                "shares": body[2], "price": body[3] / PRICE_SCALE}
    if mtype == b"P":
        # Hidden/non-displayed execution: volume only, does NOT touch the lit book.
        return {"type": "P", "ts": ts, "locate": locate,
                "ref": body[0], "side": body[1].decode(),
                "shares": body[2], "stock": body[3].decode("ascii").strip(),
                "price": body[4] / PRICE_SCALE, "match": body[5]}
    if mtype == b"Q":
        # Auction cross (open/close/halt): volume only, not the continuous book.
        return {"type": "Q", "ts": ts, "locate": locate,
                "shares": body[0], "stock": body[1].decode("ascii").strip(),
                "price": body[2] / PRICE_SCALE, "match": body[3],
                "cross_type": body[4].decode()}


def iter_messages(fh):
    """Yield raw ITCH messages from a historical BinaryFILE stream.

    Nasdaq historical TotalView-ITCH files frame each message with a 2-byte
    big-endian length prefix. (Live capture uses MoldUDP64 framing instead --
    see the bottom notes.) `fh` is a binary file object (already gunzipped).
    """
    # Loop until we run out of length prefixes.
    while True:
        # Read the 2-byte big-endian message length.
        hdr = fh.read(2)
        # Empty read means end of file.
        if len(hdr) < 2:
            return
        # Decode the length of the message body that follows.
        n = int.from_bytes(hdr, "big")
        # Read exactly n bytes for this message.
        msg = fh.read(n)
        # Guard against a truncated file.
        if len(msg) < n:
            return
        # Hand the raw bytes to the caller.
        yield msg


# ---------------------------------------------------------------------------
# ORDER BOOK (MBO / L3) RECONSTRUCTION
# ---------------------------------------------------------------------------

class OrderBook:
    """Single-symbol MBO book. Track it per stock_locate in production."""

    def __init__(self):
        # ref -> order dict. This IS the L3 state (every live order).
        self.orders = {}
        # Aggregated displayed size per price on each side (the L2 view).
        # price -> total_shares. Kept in sync with self.orders.
        self.bids = defaultdict(int)
        self.asks = defaultdict(int)
        # Monotonic sequence assigned to each add; used for FIFO queue priority.
        self._seq = 0
        # Optional hook: a QueueTracker for one simulated passive order (or None).
        self.sim = None

    def _level(self, side):
        # Return the correct price->size dict for a 'B' or 'S' order.
        return self.bids if side == "B" else self.asks

    def add(self, ref, side, price, shares):
        # Give this order the next FIFO sequence number (arrival order).
        self._seq += 1
        # Store the full order record (this is the order-by-order L3 detail).
        self.orders[ref] = {"side": side, "price": price,
                            "shares": shares, "seq": self._seq}
        # Add its size to the aggregated price level.
        self._level(side)[price] += shares
        # If we're simulating and this order lands at our price/side, it is
        # BEHIND us (arrived after we joined), so it does not affect our queue.
        # (No queue update needed on adds behind the simulated order.)

    def _reduce(self, ref, qty, is_execution):
        # Shared logic for E/C/X/D: remove `qty` shares from order `ref`.
        # Return the order's resting side/price so callers can record trades.
        o = self.orders.get(ref)
        # Order may be unknown if we started mid-session (no opening snapshot).
        if o is None:
            return None
        # Drive the simulated queue BEFORE we mutate, using this order's seq.
        if self.sim is not None:
            # An execution consumes the queue; a cancel/delete also frees space
            # ahead of us. The tracker decides based on seq (ahead vs behind).
            self.sim.on_book_event(o, qty, is_execution)
        # Decrement the order's remaining shares.
        o["shares"] -= qty
        # Decrement the aggregated level by the same amount.
        lvl = self._level(o["side"])
        lvl[o["price"]] -= qty
        # Clean up an emptied price level so best-bid/ask stays correct.
        if lvl[o["price"]] <= 0:
            del lvl[o["price"]]
        # If the order is fully consumed, drop it from the L3 map.
        if o["shares"] <= 0:
            del self.orders[ref]
        # Hand back side/price for trade recording.
        return o["side"], o["price"]

    def execute(self, ref, exec_shares):
        # 'E': trade at the order's own resting price.
        return self._reduce(ref, exec_shares, is_execution=True)

    def execute_with_price(self, ref, exec_shares):
        # 'C': book is reduced at the resting order's price; the PRINT price
        # (exec_price) is handled by the caller for tape/volume, not the book.
        return self._reduce(ref, exec_shares, is_execution=True)

    def cancel(self, ref, cancelled_shares):
        # 'X': partial cancel frees queue ahead of us but is NOT a fill.
        return self._reduce(ref, cancelled_shares, is_execution=False)

    def delete(self, ref):
        # 'D': full delete. Look up remaining shares, then reduce by that amount.
        o = self.orders.get(ref)
        # Nothing to do if we never saw the order.
        if o is None:
            return None
        # Reuse _reduce with the full remaining size (treated as a cancel).
        return self._reduce(ref, o["shares"], is_execution=False)

    def replace(self, orig_ref, new_ref, shares, price):
        # 'U': capture the original side BEFORE deleting (needed for the re-add).
        o = self.orders.get(orig_ref)
        # If we never saw the original, we can't infer side; skip safely.
        if o is None:
            return
        # Remember the side; price/shares come from the replace message itself.
        side = o["side"]
        # Delete the original order (frees its place in the queue).
        self.delete(orig_ref)
        # Re-add under the NEW reference at the NEW price/size (back of queue).
        self.add(new_ref, side, price, shares)

    def best_bid(self):
        # Highest bid price, or None if the bid side is empty. O(n) here;
        # production uses a sorted structure (see notes) for O(1)/O(log n).
        return max(self.bids) if self.bids else None

    def best_ask(self):
        # Lowest ask price, or None if the ask side is empty.
        return min(self.asks) if self.asks else None


# ---------------------------------------------------------------------------
# FIFO QUEUE-POSITION FILL SIMULATOR (the part we do NOT simplify)
# ---------------------------------------------------------------------------

class QueueTracker:
    """Simulate ONE resting passive order under Nasdaq price/time (FIFO) priority.

    Fill logic, given full MBO data:
      * At placement, `shares_ahead` = size of all orders already resting at our
        price on our side (they have priority over us).
      * Any execution/cancel/delete of an order with seq < our seq at our
        price/side reduces `shares_ahead` (queue drains in front of us).
      * Once `shares_ahead` hits 0, subsequent EXECUTIONS at our price/side
        fill US (cancels never fill us -- only trades do).
    """

    def __init__(self, side, price, size, book):
        # Our order's side ('B' or 'S'), limit price, and remaining size.
        self.side = side
        self.price = price
        self.remaining = size
        # Snapshot the queue ahead of us = current displayed size at our level.
        self.shares_ahead = book._level(side).get(price, 0)
        # Our FIFO seq is "now": every currently-resting order is ahead of us,
        # every future add is behind us. We freeze the book's counter as our seq.
        self.seq = book._seq
        # Accumulate our simulated fills as (ts-less) (price, shares) tuples.
        self.fills = []

    def on_book_event(self, order, qty, is_execution):
        # Only events at OUR price and OUR side can interact with our queue.
        if order["side"] != self.side or order["price"] != self.price:
            return
        # We are already fully filled: nothing more to do.
        if self.remaining <= 0:
            return
        # Case 1: the event hits an order AHEAD of us (resting at/before we
        # joined). Everything with seq <= our seq had priority over us.
        if order["seq"] <= self.seq:
            # Drain the queue in front of us by the affected quantity.
            self.shares_ahead -= qty
            # Clamp so it never goes negative.
            if self.shares_ahead < 0:
                self.shares_ahead = 0
            return
        # Case 2: the event hits an order BEHIND us (newer seq).
        #   - If it's an execution, it means a marketable order matched behind
        #     us; under strict FIFO that can't happen before us, so we ignore.
        #   - Cancels/deletes behind us don't affect our position.
        #   (We reach here only when shares_ahead should already be ~0.)
        if is_execution and self.shares_ahead <= 0:
            # We're at the front: this execution fills us for up to `qty`.
            filled = min(self.remaining, qty)
            # Record the fill at our limit price.
            self.fills.append((self.price, filled))
            # Reduce our remaining size.
            self.remaining -= filled
