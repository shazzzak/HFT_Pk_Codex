"""The venue interface -- the single seam between shared logic and one market.

Everything that differs between exchanges is declared here as an abstract
method, so a new market is a new subclass rather than a new engine. The rules
this abstraction has to survive, drawn from the two markets in scope:

  * TICK SIZE IS NOT A CONSTANT. PSX is a flat 0.01 PKR at every price. IDX is
    tiered: the tick widens in steps as the price rises. Any code that hard-codes
    a tick is PSX-only code, so the tick is a FUNCTION OF PRICE here.
  * THE TRADING DAY IS NOT ONE INTERVAL. PSX breaks for Jumu'ah on Fridays, so a
    session is a LIST of segments and 'minutes remaining' must be summed over
    them. Treating the day as open-to-close overstates remaining time on Friday,
    which silently mis-sizes every end-of-day unwind.
  * REGULATORY LIMITS ARE VENUE PROPERTIES. The orders-per-second cap and the
    order-to-trade ratio threshold come from the exchange, not from us, and both
    are currently unknown for PSX pending the final SECP framework. They are
    declared here so the engine reads them from one place when they land.
"""
# the abstract base machinery
from abc import ABC, abstractmethod
# value objects for the session description
from dataclasses import dataclass
# typing only
from typing import Optional, Sequence
# the shared domain
from core.model import Side


@dataclass(frozen=True)
class SessionSegment:
    """One continuous-trading interval, in exchange milliseconds."""
    # when continuous trading starts in this segment
    start_ms: int
    # when it ends
    end_ms: int

    def contains(self, timestamp_ms: int) -> bool:
        """Is this timestamp inside the segment?"""
        # half-open so two adjacent segments cannot both claim the boundary
        return self.start_ms <= timestamp_ms < self.end_ms

    @property
    def duration_ms(self) -> int:
        """How long this segment lasts."""
        # never negative, even if a bad feed inverts the pair
        return max(0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class PriceBand:
    """The exchange's hard price limits for one instrument, right now."""
    # the highest price the exchange will accept
    upper_minor: int
    # the lowest price the exchange will accept
    lower_minor: int

    def contains(self, price_minor: int) -> bool:
        """Would the exchange accept an order at this price?"""
        # inclusive: an order exactly at the band is acceptable
        return self.lower_minor <= price_minor <= self.upper_minor


class Venue(ABC):
    """One exchange's rules. Subclass this; do not branch on venue name."""

    # ---- identity ---------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Short venue code, used in logs and in the audit trail."""

    @property
    @abstractmethod
    def minor_per_major(self) -> int:
        """How many minor currency units make one major unit.

        PKR: 100 (paisa). IDR: 1 (no subdivision is quoted). Every price in the
        engine is an integer count of minor units; this is the only place that
        knows how to turn one back into a human-readable number.
        """

    # ---- the price grid ---------------------------------------------------
    @abstractmethod
    def tick_minor(self, price_minor: int) -> int:
        """The tick size AT THIS PRICE, in minor units.

        Taking the price as an argument is what makes a tiered grid expressible.
        A flat-tick venue ignores it.
        """

    def round_to_tick(self, price_minor: int, side: Side) -> int:
        """Snap a price onto the grid, NEVER becoming more aggressive.

        The direction of rounding is a correctness property, not a style
        choice: rounding a bid UP or an ask DOWN moves the quote closer to the
        touch than the strategy asked for, which silently spends capture the
        strategy believed it was keeping. So a bid floors and an ask ceils.
        """
        # the grid spacing that applies around this price
        tick = self.tick_minor(price_minor)
        # a non-positive tick would divide by zero and is a venue bug
        if tick <= 0:
            raise ValueError(f"{self.name}: tick_minor returned {tick}")
        # a bid rounds DOWN, so it can only become less aggressive
        if side is Side.BUY:
            return (price_minor // tick) * tick
        # an ask rounds UP, likewise less aggressive
        return -((-price_minor) // tick) * tick

    # ---- the trading day --------------------------------------------------
    @abstractmethod
    def sessions(self, date: str) -> Sequence[SessionSegment]:
        """The continuous-trading segments for one date, as 'YYYY-MM-DD'.

        A list, not a pair, because some days have a break in the middle.
        """

    def is_continuous(self, date: str, timestamp_ms: int) -> bool:
        """Is continuous trading open at this instant?

        This is the gate behind the SECP requirement (concept paper s7) that
        algorithmic orders may not be placed in the pre-open or through any
        off-hours mechanism.
        """
        # true if the timestamp falls inside any one of the day's segments
        return any(seg.contains(timestamp_ms) for seg in self.sessions(date))

    def tradeable_ms_remaining(self, date: str, timestamp_ms: int) -> int:
        """Milliseconds of TRADEABLE time left, summed across segments.

        Wall-clock time to the close is the wrong number on a day with a break:
        it counts minutes in which nothing can be unwound. Every capacity and
        end-of-day calculation must use this instead.
        """
        # accumulate only the parts of each segment that are still ahead
        remaining = 0
        # walk every segment of the day
        for seg in self.sessions(date):
            # nothing left to count in a segment already finished
            if timestamp_ms >= seg.end_ms:
                continue
            # count from now, or from the segment start if it has not begun
            remaining += seg.end_ms - max(timestamp_ms, seg.start_ms)
        # total tradeable milliseconds still available
        return remaining

    # ---- costs ------------------------------------------------------------
    @property
    @abstractmethod
    def fee_bps_per_side(self) -> float:
        """All-in transaction cost per side, in basis points of traded value."""

    # ---- regulatory and exchange limits -----------------------------------
    @property
    @abstractmethod
    def max_orders_per_second(self) -> Optional[int]:
        """Exchange cap on order messages per second, or None if unknown.

        None means UNKNOWN, not UNLIMITED. The risk gateway treats a None here
        as a reason to fall back to our own house limit rather than as
        permission to send at any rate.
        """

    @property
    @abstractmethod
    def max_order_to_trade_ratio(self) -> Optional[float]:
        """Exchange order-to-trade ratio threshold, or None if unknown.

        Same convention as above: None means we have not been told.
        """

    @abstractmethod
    def price_band(self, symbol: str) -> Optional[PriceBand]:
        """The current exchange price limits for a symbol, if published.

        Returns None when the venue has not published a band for this symbol
        yet -- at which point the risk gateway falls back to its own house
        band rather than letting an unbounded price through.
        """

    # ---- order tagging ----------------------------------------------------
    @property
    @abstractmethod
    def requires_algo_tag(self) -> bool:
        """Must every order carry an exchange-issued algorithm identifier?

        SECP concept paper s4 proposes exactly this for PSX. The encoder needs
        a slot for it from the first line of code even while the value is a
        placeholder, because retrofitting a mandatory field across an order
        path is far more disruptive than carrying an unused one.
        """
