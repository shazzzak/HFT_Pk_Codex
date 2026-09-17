"""Venue-agnostic domain model for a market-making engine.

Nothing in this module knows about PSX, FIX, or any specific exchange. Anything
that differs between venues lives behind the Venue interface in core/venue.py,
so this file can be imported unchanged by a second market (IDX, and so on).

PRICES ARE INTEGERS. A price is carried as a whole number of the currency's
MINOR unit -- paisa for PKR, whole rupiah for IDR -- never as a float. Floats
make `px == bb + tick` fail at random and make a tick grid impossible to test
against. How many minor units make one major unit is a property of the venue,
not of this module, and so is how one is rendered onto the wire.
"""
# dataclasses give value objects with equality and a readable repr for free
from dataclasses import dataclass, field
# enums, so a side or a state can never be an arbitrary string
from enum import Enum
# typing only; no runtime cost
from typing import Optional


class Side(Enum):
    """Which side of the book an order sits on.

    Deliberately only two values. A venue may distinguish more -- PSX FIX 4.2
    separates plain Sell from Sell Short, Cross and Borrow -- but that is a
    WIRE distinction driven by the account's position and the venue's rules,
    not a strategy one. The strategy decides to buy or sell; the venue encoder
    decides which of its codes expresses that. Putting venue order codes in
    here would put PSX's rulebook in the shared model.
    """
    # we are bidding: we want to buy
    BUY = "BUY"
    # we are offering: we want to sell
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        """The other side. Used wherever an exit or a cross is expressed."""
        # BUY -> SELL and SELL -> BUY, with no if-chain at the call site
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 for a buy, -1 for a sell, so position maths needs no branching."""
        # a fill on the buy side increases inventory; a sell decreases it
        return 1 if self is Side.BUY else -1


class OrderState(Enum):
    """The order lifecycle.

    The PENDING_* states exist because an order that has been SENT but not yet
    acknowledged is neither live nor dead, and the difference matters: we must
    not send a second order believing the first never left, and we must not
    assume a cancel worked before the exchange says so. An engine without these
    states double-sends under latency, which is the most common way an order
    management system loses money.

    The state set is aligned with PSX FIX 4.2 OrdStatus (tag 39), which carries
    New, Partial fill, Fill, Canceled, Replace, Pending Cancel, Rejected,
    Suspended, Pending New and Pending Replace. Two of those -- SUSPENDED and
    PENDING_REPLACE -- have no meaning in a pure quoting strategy but DO arrive
    on the wire, and an inbound status we cannot represent is an inbound status
    we will mishandle.
    """
    # created locally, not yet sent
    NEW = "NEW"
    # sent to the exchange, no acknowledgement yet
    PENDING_NEW = "PENDING_NEW"
    # acknowledged and resting on the book
    LIVE = "LIVE"
    # partially executed and still resting
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    # an amendment has been sent and not yet confirmed
    PENDING_REPLACE = "PENDING_REPLACE"
    # cancel sent, not yet acknowledged
    PENDING_CANCEL = "PENDING_CANCEL"
    # the exchange is holding the order inactive at our request
    SUSPENDED = "SUSPENDED"
    # fully executed
    FILLED = "FILLED"
    # cancelled and no longer on the book
    CANCELLED = "CANCELLED"
    # the exchange refused it
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        """True once the order can never change again."""
        # a terminal order can be forgotten; a non-terminal one must be tracked
        return self in (OrderState.FILLED, OrderState.CANCELLED,
                        OrderState.REJECTED)

    @property
    def is_working(self) -> bool:
        """True while the exchange may still execute this order.

        PENDING_CANCEL and PENDING_REPLACE both count: an amendment or a cancel
        can lose the race with an incoming aggressor, so the exposure is real
        until the exchange confirms. Treating a pending cancel as already dead
        is how a position appears from nowhere.

        SUSPENDED does NOT count -- a suspended order cannot trade -- but it is
        not terminal either, because it can be resumed.
        """
        # every state in which a fill can still arrive
        return self in (OrderState.PENDING_NEW, OrderState.LIVE,
                        OrderState.PARTIALLY_FILLED, OrderState.PENDING_CANCEL,
                        OrderState.PENDING_REPLACE)

    @property
    def is_in_flight(self) -> bool:
        """True while we are waiting for the exchange to answer.

        Nothing may be sent about an order in this state. The order manager
        keys its most important safety rule on this.
        """
        # the three states where a message of ours is outstanding
        return self in (OrderState.PENDING_NEW, OrderState.PENDING_CANCEL,
                        OrderState.PENDING_REPLACE)


@dataclass(frozen=True)
class QuoteIntent:
    """What the strategy WANTS on one side of one symbol, right now.

    This is a desire, not an order. The strategy produces intents; the order
    manager decides what messages, if any, are needed to make the book match.
    Keeping those two ideas apart is what lets the order-to-trade ratio be
    controlled in one place, and what lets the same strategy code drive both the
    backtest and the live engine.
    """
    # which side this intent is for
    side: Side
    # the limit price, in minor units (paisa for PKR)
    price_minor: int
    # how many shares we want resting at that price
    quantity: int

    def __post_init__(self):
        # a non-positive price is never a real quote, it is a bug upstream
        if self.price_minor <= 0:
            raise ValueError(f"price_minor must be positive, got "
                             f"{self.price_minor}")
        # zero quantity is expressed by omitting the intent, not by a zero
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")

    @property
    def notional_minor(self) -> int:
        """Value of this intent in minor units, for the value-limit check."""
        # price x size, still an integer, so no rounding creeps in
        return self.price_minor * self.quantity


@dataclass(frozen=True)
class DesiredQuotes:
    """The strategy's complete desired state for ONE symbol.

    COMPLETE is the operative word and it is a protocol, not a detail: a side
    that is absent means "nothing should be resting there", not "leave whatever
    is there alone". An engine that treats absence as 'no change' can never be
    made to go flat, because there is no message that means 'stop'.
    """
    # the symbol these intents are for
    symbol: str
    # the bid we want resting, or None for no bid
    bid: Optional[QuoteIntent] = None
    # the ask we want resting, or None for no ask
    ask: Optional[QuoteIntent] = None

    def __post_init__(self):
        # an intent filed under the wrong side would silently invert the book
        if self.bid is not None and self.bid.side is not Side.BUY:
            raise ValueError("bid intent must have side=BUY")
        # same check for the offer
        if self.ask is not None and self.ask.side is not Side.SELL:
            raise ValueError("ask intent must have side=SELL")

    @classmethod
    def flat(cls, symbol: str) -> "DesiredQuotes":
        """Want nothing resting. This is what the kill switch asks for."""
        # both sides absent, which the protocol above defines as 'cancel all'
        return cls(symbol=symbol)

    def side(self, side: Side) -> Optional[QuoteIntent]:
        """The intent for one side, so callers need no attribute branching."""
        # pick the matching field
        return self.bid if side is Side.BUY else self.ask


@dataclass
class Order:
    """A single order we have sent, or are about to send.

    Mutable by design: this object IS the record of what the exchange thinks,
    updated as acknowledgements and fills arrive. Its state must only ever be
    changed through the transition methods below, so an impossible transition
    raises instead of silently corrupting the position.
    """
    # our own identifier, unique for the life of this session
    cl_ord_id: str
    # the instrument
    symbol: str
    # which side
    side: Side
    # limit price in minor units
    price_minor: int
    # the size originally sent
    quantity: int
    # where it is in the lifecycle
    state: OrderState = OrderState.NEW
    # how much has executed so far
    filled_quantity: int = 0
    # the exchange's own identifier, once it gives us one. AN AMENDMENT OR A
    # CANCEL CANNOT BE SENT WITHOUT IT -- PSX requires OrderID (tag 37) on both
    # -- which is a second, harder reason never to act on an unacknowledged
    # order: we do not merely prefer to wait, we have nothing to send.
    exchange_order_id: Optional[str] = None
    # the client code this order trades for, required by PSX on every order
    account: Optional[str] = None
    # why it was rejected or cancelled, when that applies
    reason: Optional[str] = None
    # the id of the order this one replaced, when it came from an amendment
    orig_cl_ord_id: Optional[str] = None

    @property
    def leaves_quantity(self) -> int:
        """Shares still working. This is the live exposure, not `quantity`."""
        # nothing is working once the order reached a terminal state
        if self.state.is_terminal:
            return 0
        # otherwise it is whatever has not yet executed
        return self.quantity - self.filled_quantity

    def on_ack(self, exchange_order_id: str) -> None:
        """The exchange acknowledged the order; it is now resting."""
        # only an order we believe is in flight can be acknowledged
        if self.state is not OrderState.PENDING_NEW:
            raise InvalidTransition(self, "ack")
        # record the exchange's handle, which every later cancel or amend needs
        self.exchange_order_id = exchange_order_id
        # it is now live on the book
        self.state = OrderState.LIVE

    def on_fill(self, quantity: int) -> None:
        """An execution arrived for this order."""
        # a fill on a terminal order means our state and theirs disagree
        if not self.state.is_working:
            raise InvalidTransition(self, f"fill({quantity})")
        # more fill than the order ever had is a feed or mapping error
        if quantity <= 0 or self.filled_quantity + quantity > self.quantity:
            raise InvalidTransition(self, f"fill({quantity})")
        # accumulate the executed size
        self.filled_quantity += quantity
        # fully done, or still resting with a smaller remainder
        self.state = (OrderState.FILLED
                      if self.filled_quantity == self.quantity
                      else OrderState.PARTIALLY_FILLED)

    def on_cancel_sent(self) -> None:
        """We have sent a cancel and are waiting for confirmation."""
        # cancelling something that is not working is a logic error upstream
        if not self.state.is_working:
            raise InvalidTransition(self, "cancel_sent")
        # the exposure is still real until the exchange confirms
        self.state = OrderState.PENDING_CANCEL

    def on_cancelled(self) -> None:
        """The exchange confirmed the cancel."""
        # confirmation for an order that was never working is inconsistent
        if not self.state.is_working:
            raise InvalidTransition(self, "cancelled")
        # now genuinely off the book
        self.state = OrderState.CANCELLED

    def on_replace_sent(self) -> None:
        """We have sent an amendment and are waiting for confirmation."""
        # only a resting order can be amended
        if self.state not in (OrderState.LIVE, OrderState.PARTIALLY_FILLED):
            raise InvalidTransition(self, "replace_sent")
        # the OLD terms are still live until the exchange confirms the new ones
        self.state = OrderState.PENDING_REPLACE

    def on_replaced(self, price_minor: int, quantity: int) -> None:
        """The exchange confirmed the amendment; the new terms are live."""
        # only an order with an amendment outstanding can be replaced
        if self.state is not OrderState.PENDING_REPLACE:
            raise InvalidTransition(self, "replaced")
        # the new terms
        self.price_minor = price_minor
        self.quantity = quantity
        # THE FILL COUNTER RESTARTS FOR THIS GENERATION, and it must.
        #
        # PSX Regulations 8.12.2: a Change Former Order "can only modify price
        # and volume of an unfilled/outstanding Order in whole or in parts".
        # The quantity coming back is therefore the new RESTING size, not the
        # original total -- which is also how mm_backtest._amend treats it,
        # working off the remaining size rather than the order as sent.
        #
        # Without this reset, an order for 50 that had filled 30 and was then
        # amended back to a full 50 would carry filled_quantity = 30 against
        # quantity = 50, and the very next fill of 50 would be refused as
        # more fill than the order ever had. That is exactly the
        # InvalidTransition the first gate run hit on AGP, three days out of
        # three.
        #
        # POSITION IS UNAFFECTED. The order manager accumulates position from
        # fill events and never derives it from this counter, so rebasing here
        # cannot lose a share.
        self.filled_quantity = 0
        # a freshly amended order is resting with nothing yet taken against it
        self.state = OrderState.LIVE

    def on_suspended(self) -> None:
        """The exchange is holding the order inactive."""
        # a terminal order cannot be suspended
        if self.state.is_terminal:
            raise InvalidTransition(self, "suspended")
        # inactive, but not gone: it can be resumed
        self.state = OrderState.SUSPENDED

    def on_rejected(self, reason: str) -> None:
        """The exchange refused the order."""
        # a reject after the order was already live is a different event
        if self.state not in (OrderState.NEW, OrderState.PENDING_NEW):
            raise InvalidTransition(self, "rejected")
        # keep the reason: it is required in the audit trail
        self.reason = reason
        # terminal
        self.state = OrderState.REJECTED


class InvalidTransition(Exception):
    """An order was asked to do something its state does not permit.

    This is deliberately loud. A silently-ignored bad transition means our
    position and the exchange's have diverged, and every number downstream --
    P&L, risk limits, the kill switch's idea of what to cancel -- is then wrong.
    """

    def __init__(self, order: "Order", event: str):
        # name the order, its state, and what was attempted
        super().__init__(f"order {order.cl_ord_id} ({order.symbol} "
                         f"{order.side.value}) is {order.state.value}; "
                         f"cannot apply {event}")
        # keep the objects for a handler that wants to inspect them
        self.order = order
        self.event = event


# ---------------------------------------------------------------------------
# ACTIONS -- what the order manager wants to do, before risk sees it
# ---------------------------------------------------------------------------
# The order manager never talks to the wire. It produces Actions, the risk
# gateway approves or rejects each one, and only approved Actions are encoded.
# Making the request a value object is what allows every decision to be logged
# and replayed.

@dataclass(frozen=True)
class Action:
    """Base class for a requested exchange action."""
    # the instrument the action concerns
    symbol: str

    @property
    def is_cancel(self) -> bool:
        """Cancels are privileged: risk checks must never block them.

        A control that can block a cancel is a control that can trap us in a
        position. Every limit in this system restricts what we ADD, never what
        we remove. This property is what the gateway keys that rule on.
        """
        # only the cancel subclass overrides this to True
        return False


@dataclass(frozen=True)
class OrderRequest(Action):
    """An action that puts size on the book, or changes how much is there.

    THIS CLASS EXISTS TO PREVENT ONE SPECIFIC BUG. The risk controls have to
    apply to a PlaceOrder and to a ReplaceOrder identically -- an amendment can
    raise the price, increase the quantity, or push the position past a limit
    just as a new order can. If the controls tested for PlaceOrder by name, an
    amendment would sail past every one of them, and the failure would be
    invisible because nothing errors: orders simply stop being checked.

    Every risk control tests for THIS type, so a future action class that adds
    size is checked by default rather than by remembering to.
    """
    # our identifier for the resulting order
    cl_ord_id: str
    # which side
    side: Side
    # limit price, minor units
    price_minor: int
    # size in shares
    quantity: int
    # the client code this trades for, when the venue requires one
    account: Optional[str] = None

    @property
    def notional_minor(self) -> int:
        """Order value, for the value limit."""
        # integer price x integer size
        return self.price_minor * self.quantity


@dataclass(frozen=True)
class PlaceOrder(OrderRequest):
    """Send a new order."""
    # no extra fields: a placement is the plain case of an order request


@dataclass(frozen=True)
class ReplaceOrder(OrderRequest):
    """Amend a resting order's price or quantity in a single message.

    Only possible once the exchange has acknowledged the original, because the
    venue requires its own OrderID on the amendment. That is not a preference:
    without the ack there is literally nothing to send.
    """
    # the id of the order being amended
    orig_cl_ord_id: str = ""
    # the exchange's own handle for it, which the venue requires on the wire
    exchange_order_id: str = ""

    def __post_init__(self):
        # an amendment with nothing to amend is a bug, not a no-op
        if not self.orig_cl_ord_id:
            raise ValueError("ReplaceOrder needs orig_cl_ord_id")
        # and one the exchange cannot match to an order of its own is rejected
        if not self.exchange_order_id:
            raise ValueError("ReplaceOrder needs exchange_order_id; the "
                             "original must be acknowledged first")


@dataclass(frozen=True)
class CancelOrder(Action):
    """Cancel an order we believe is working."""
    # the order to cancel
    cl_ord_id: str
    # the id of the order being cancelled
    orig_cl_ord_id: str = ""
    # the exchange's handle, which the venue requires on a cancel
    exchange_order_id: str = ""

    @property
    def is_cancel(self) -> bool:
        # this is the one action class that risk must always let through
        return True


@dataclass(frozen=True)
class Fill:
    """One execution, as reported by the exchange."""
    # the order it belongs to
    cl_ord_id: str
    # the instrument
    symbol: str
    # which side we traded
    side: Side
    # the execution price in minor units
    price_minor: int
    # how many shares executed
    quantity: int
    # exchange timestamp, milliseconds since epoch
    timestamp_ms: int

    @property
    def signed_quantity(self) -> int:
        """Position delta: positive for a buy, negative for a sell."""
        # the sign lives on Side, so this stays a one-liner everywhere
        return self.side.sign * self.quantity

    @property
    def notional_minor(self) -> int:
        """Traded value in minor units."""
        # unsigned: this feeds turnover and fee calculations
        return self.price_minor * self.quantity


# ---------------------------------------------------------------------------
# MARKET DATA
#
# What the strategy reads. Kept here rather than in a feed module because these
# are domain objects: a strategy consumes a book and a trade, and neither idea
# is specific to FIX, to PSX, or to how the bytes arrived.
#
# PRICES ARE INTEGERS HERE TOO. The feed decoder is where a wire price becomes
# a minor-unit integer (Venue.parse_price), so nothing downstream of it ever
# sees a float.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BookLevel:
    """One price level of the order book."""
    # the level's price in minor units
    price_minor: int
    # the total shares resting at that price
    quantity: int


@dataclass(frozen=True)
class BookSnapshot:
    """The order book for one symbol at one instant.

    A SNAPSHOT, not a delta. Whatever assembles it from the feed is responsible
    for having applied every update up to `timestamp_ms`; by the time it gets
    here it is a complete statement of the book, and the strategy is entitled
    to treat it as one.

    EITHER SIDE MAY BE EMPTY. A book with no bid is a real state on PSX -- it
    happens at the open, after a halt, and on illiquid names all day -- and it
    is not an error. It means there is nothing to quote against, which is a
    different thing from bad data.
    """
    # the instrument
    symbol: str
    # exchange timestamp, milliseconds since epoch. THE EXCHANGE'S clock, not
    # ours: every time-based decision in the strategy counts against this, so
    # feeding it a local wall clock makes the end-of-day logic fire early or
    # late by however far the feed is behind.
    timestamp_ms: int
    # bid levels, BEST FIRST (highest price first)
    bids: tuple = ()
    # ask levels, BEST FIRST (lowest price first)
    asks: tuple = ()

    def __post_init__(self):
        # a book whose levels are out of order is corrupt, and every downstream
        # calculation that reads [0] as 'the touch' would be quietly wrong
        for i in range(1, len(self.bids)):
            # bids must descend
            if self.bids[i].price_minor >= self.bids[i - 1].price_minor:
                raise ValueError(f"{self.symbol}: bids not in descending price "
                                 f"order at level {i}")
        # same for the offers, ascending
        for i in range(1, len(self.asks)):
            # asks must ascend
            if self.asks[i].price_minor <= self.asks[i - 1].price_minor:
                raise ValueError(f"{self.symbol}: asks not in ascending price "
                                 f"order at level {i}")
        # a crossed book means the feed is broken or we have applied updates out
        # of sequence; quoting against it would place orders inside a spread
        # that does not exist
        if self.bids and self.asks:
            # best bid at or above best ask
            if self.bids[0].price_minor >= self.asks[0].price_minor:
                raise ValueError(
                    f"{self.symbol}: crossed book, bid "
                    f"{self.bids[0].price_minor} >= ask "
                    f"{self.asks[0].price_minor}")

    @property
    def best_bid(self) -> Optional[BookLevel]:
        """The touch on the bid side, or None when there is no bid."""
        # first level if there is one
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[BookLevel]:
        """The touch on the offer side, or None when there is no offer."""
        # first level if there is one
        return self.asks[0] if self.asks else None

    @property
    def is_two_sided(self) -> bool:
        """True when both sides have depth -- the only case worth quoting in."""
        # both touches present and both carrying shares
        return (bool(self.bids) and bool(self.asks)
                and self.bids[0].quantity > 0 and self.asks[0].quantity > 0)

    @property
    def mid_minor(self) -> Optional[int]:
        """The midpoint in minor units, or None when the book is one-sided.

        Returned as an INTEGER, which means a half-paisa mid is floored. That
        is deliberate: the mid is used here for reporting and for rejecting
        stale books, never for pricing a quote. Anything that prices against
        the mid must do it in the units it actually needs.
        """
        # no mid exists without two sides
        if not self.is_two_sided:
            return None
        # integer midpoint, floored
        return (self.bids[0].price_minor + self.asks[0].price_minor) // 2

    @property
    def spread_minor(self) -> Optional[int]:
        """Best ask minus best bid, or None when the book is one-sided."""
        # no spread exists without two sides
        if not self.is_two_sided:
            return None
        # the gap, always positive because __post_init__ refuses a crossed book
        return self.asks[0].price_minor - self.bids[0].price_minor


@dataclass(frozen=True)
class Trade:
    """One execution that happened in the market -- not necessarily ours.

    `aggressor` is the side that CROSSED the spread, and it is Optional on
    purpose: a feed that does not publish it must not have a side guessed for
    it. Every flow-direction signal in the strategy is built on this field, so
    a guessed value does not degrade the signal, it inverts it half the time.
    """
    # the instrument
    symbol: str
    # exchange timestamp, milliseconds since epoch
    timestamp_ms: int
    # the execution price in minor units
    price_minor: int
    # how many shares traded
    quantity: int
    # which side took liquidity, or None when the feed does not say
    aggressor: Optional[Side] = None
