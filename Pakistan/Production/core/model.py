"""Venue-agnostic domain model for a market-making engine.

Nothing in this module knows about PSX, FIX, or any specific exchange. Anything
that differs between venues lives behind the Venue interface in core/venue.py,
so this file can be imported unchanged by a second market (IDX, and so on).

PRICES ARE INTEGERS. A price is carried as a whole number of the currency's
MINOR unit -- paisa for PKR, whole rupiah for IDR -- never as a float. Floats
make `px == bb + tick` fail at random and make a tick grid impossible to test
against. How many minor units make one major unit is a property of the venue,
not of this module.
"""
# dataclasses give value objects with equality and a readable repr for free
from dataclasses import dataclass, field
# enums, so a side or a state can never be an arbitrary string
from enum import Enum
# typing only; no runtime cost
from typing import Optional


class Side(Enum):
    """Which side of the book an order sits on."""
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
    states double-sends under latency, which is the single most common way an
    order management system loses money.
    """
    # created locally, not yet sent
    NEW = "NEW"
    # sent to the exchange, no acknowledgement yet
    PENDING_NEW = "PENDING_NEW"
    # acknowledged and resting on the book
    LIVE = "LIVE"
    # partially executed and still resting
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    # cancel sent, not yet acknowledged
    PENDING_CANCEL = "PENDING_CANCEL"
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

        PENDING_CANCEL counts as working: the cancel may lose the race with an
        incoming aggressor, so the exposure is real until the exchange confirms.
        Treating a pending cancel as already dead is how a position appears from
        nowhere.
        """
        # every state in which a fill can still arrive
        return self in (OrderState.PENDING_NEW, OrderState.LIVE,
                        OrderState.PARTIALLY_FILLED, OrderState.PENDING_CANCEL)


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
    # our own identifier, unique for the life of the session
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
    # the exchange's own identifier, once it gives us one
    exchange_order_id: Optional[str] = None
    # why it was rejected or cancelled, when that applies
    reason: Optional[str] = None

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
        # record the exchange's handle, which every later cancel needs
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
# and replayed, which is the SECP audit-trail requirement (concept paper s12).

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
class PlaceOrder(Action):
    """Send a new order."""
    # our identifier for it
    cl_ord_id: str
    # which side
    side: Side
    # limit price, minor units
    price_minor: int
    # size in shares
    quantity: int

    @property
    def notional_minor(self) -> int:
        """Order value, for the value limit."""
        # integer price x integer size
        return self.price_minor * self.quantity


@dataclass(frozen=True)
class CancelOrder(Action):
    """Cancel an order we believe is working."""
    # the order to cancel
    cl_ord_id: str

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
