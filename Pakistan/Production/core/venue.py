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
# enums, so a market phase can never be an arbitrary string
from enum import Enum
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


class MarketPhase(Enum):
    """What the exchange says the market is doing, right now.

    WHY THIS EXISTS RATHER THAN A CALENDAR. A calendar is a guess about the
    future written down in advance. The exchange publishes the actual state --
    PSX does so every three seconds in TradingPhaseCode (tag 8538) on both the
    Trading Session Status and every snapshot -- and the published state covers
    things a calendar cannot know: an unscheduled halt, a security suspended for
    the day, a resumption after a halt.

    A calendar cannot tell you the market halted two minutes ago. It will
    cheerfully report that trading is open while the exchange has stopped
    matching, and the engine will quote into a market that is not there.
    """
    # before the market opens
    STARTING = "STARTING"
    # a call auction: orders accepted, no continuous matching
    PRE_OPEN = "PRE_OPEN"
    # continuous matching -- THE ONLY PHASE A MARKET MAKER QUOTES IN
    CONTINUOUS = "CONTINUOUS"
    # a scheduled break, including the Friday prayer break
    BREAK = "BREAK"
    # an unscheduled halt or suspension
    HALTED = "HALTED"
    # a closing auction
    PRE_CLOSE = "PRE_CLOSE"
    # after-hours trading
    POST_CLOSE = "POST_CLOSE"
    # the market is shut
    CLOSED = "CLOSED"
    # we have not been told, or were told something we do not recognise.
    # TREATED AS 'DO NOT TRADE'. A phase we cannot interpret is not a phase we
    # may assume is open.
    UNKNOWN = "UNKNOWN"

    @property
    def is_tradeable(self) -> bool:
        """Can a resting limit order be matched right now?"""
        # continuous matching, and nothing else
        return self is MarketPhase.CONTINUOUS


@dataclass(frozen=True)
class SecurityPhase:
    """The full state of one instrument, as the exchange publishes it."""
    # what the market as a whole is doing
    phase: MarketPhase = MarketPhase.UNKNOWN
    # this instrument is suspended for the entire day
    suspended_all_day: bool = False
    # why the market is on a break, when it is; venue-specific detail
    break_reason: Optional[str] = None

    @property
    def is_tradeable(self) -> bool:
        """Both conditions have to hold: market open AND security not suspended."""
        # a suspended security cannot trade however open the market is
        return self.phase.is_tradeable and not self.suspended_all_day


@dataclass(frozen=True)
class PriceBand:
    """The exchange's hard price limits for one instrument, right now.

    EITHER BOUND MAY BE ABSENT, and `None` is how that is said. A band built
    from a sentinel taken literally is arithmetically valid and completely
    meaningless.

    THE EXCHANGE PUBLISHES BOTH BOUNDS AS ORDINARY PRICES, on every order-book
    snapshot: MDEntryType `xe` is the upper circuit breaker and `xf` the lower.
    So a band provider READS them. It does not derive them.

    NEVER COMPUTE A BAND, EVEN THOUGH THE ARITHMETIC LOOKS EASY. On an ordinary
    day the published pair is exactly +/-10% of the previous close and it is
    tempting to reconstruct it -- as a cross-check, or as a fallback when a row
    is missing. Do not. On a stock split or a reverse split the exchange bands
    off the ADJUSTED close, so a computed band is wrong by the split ratio, on
    the one day the price is moving and the band decides whether an order is
    accepted. A missing row means the last published band still stands, not that
    we should invent one.

    THE TWO SENTINELS ARE NOT SYMMETRICAL, and this docstring previously said
    they were (corrected 2026-09-16):

      UP LIMIT (`xe`) has a clean sentinel: 999999999.9999 means no rise limit.
      Null it.

      DOWN LIMIT (`xf`) HAS NO UNIVERSAL SENTINEL. Its no-limit value equals the
      market's own minimum tick, which varies by market and segment.
      PSX_Parser_Mac.py is explicit that it "must not be hard-coded/nulled
      blindly", and does not: it flags a suspicious value (px <= 1.0) and leaves
      the decision downstream. Follow the parser, not a 0.01 rule -- nulling a
      real lower band removes the only thing stopping a quote below the floor.

    Both of those are edge-case guards on a value that is normally just a price.
    """
    # the highest price the exchange will accept, or None for no upper limit
    upper_minor: Optional[int] = None
    # the lowest price the exchange will accept, or None for no lower limit
    lower_minor: Optional[int] = None

    def contains(self, price_minor: int) -> bool:
        """Would the exchange accept an order at this price?"""
        # an absent bound constrains nothing
        if self.upper_minor is not None and price_minor > self.upper_minor:
            return False
        # likewise below
        if self.lower_minor is not None and price_minor < self.lower_minor:
            return False
        # inside every bound that exists
        return True

    @property
    def is_unbounded(self) -> bool:
        """True when the exchange published no usable limit at all."""
        # worth surfacing: a band that constrains nothing is not a band
        return self.upper_minor is None and self.lower_minor is None


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

    # ---- what the wire will and will not accept ---------------------------
    @property
    @abstractmethod
    def supports_replace(self) -> bool:
        """Can a resting order's price or size be amended in one message?

        Where it is available it removes the window in which we have cancelled
        and not yet replaced -- a window in which we are simply not quoting.
        Where it is not, the order manager falls back to cancel-then-place.

        This is a capability question with a factual answer in the venue's
        specification, and it must never be guessed: assuming an amendment is
        supported when it is not produces rejects on every reprice, and
        assuming it is not costs a round trip on every reprice forever.

        WHETHER IT KEEPS QUEUE POSITION IS A SEPARATE QUESTION, answered by
        the three properties below. Sending one message instead of two is a
        latency saving; keeping your place in the line is worth far more, and
        the two do not come together.
        """

    # ---- what an amendment does to queue position -------------------------
    # Three separate rules, because venues split them three ways: a venue that
    # keeps priority on a size reduction will usually still strip it on a
    # reprice. The simulated exchange reads these to decide whether an amended
    # order carries its place in the line or rejoins at the back, so one wrong
    # answer makes every fill downstream of a reprice wrong in the same
    # direction.
    #
    # THESE ARE FACTS ABOUT THE VENUE, published in its rulebook. They are not
    # tuning knobs. They are properties rather than constants so a second venue
    # can answer differently without a line of engine code changing.

    @property
    @abstractmethod
    def replace_price_keeps_priority(self) -> bool:
        """Does amending the PRICE keep our place in the queue?

        Almost universally no: a different price is a different queue, and a
        position held at the old price cannot mean anything at the new one. A
        venue answering True wants its citation beside the answer.
        """

    @property
    @abstractmethod
    def replace_qty_up_keeps_priority(self) -> bool:
        """Does amending the size UPWARD keep our place in the queue?

        Normally no. Growing an order without losing priority would let anyone
        hold a place in the line with one share and inflate it the moment the
        queue became valuable.
        """

    @property
    @abstractmethod
    def replace_qty_down_keeps_priority(self) -> bool:
        """Does amending the size DOWNWARD keep our place in the queue?

        Normally yes, and this is the case worth exploiting. Reducing takes
        nothing from anyone behind us -- they move up -- so venues generally
        amend the resting order in place.
        """

    @property
    @abstractmethod
    def requires_account(self) -> bool:
        """Must every order carry a client/account code?

        PSX makes Account (tag 1) a required field on New Order Single. An
        engine that discovers this at go-live discovers it as a reject on every
        single order, so the order manager refuses to start without one when
        the venue says it is needed.
        """

    @property
    @abstractmethod
    def prohibited_chars(self) -> frozenset:
        """Characters the venue refuses inside identifier and text fields.

        Exchanges use these as internal delimiters, so a stray one does not
        corrupt a value -- it gets the whole message rejected, and the reject
        arrives at the worst possible moment because the offending character is
        usually in something generated at runtime.
        """

    def validate_text(self, value: str, field: str) -> str:
        """Check a value the venue will see, and fail NOW rather than on reject.

        Called where an identifier is generated, not where it is encoded, so a
        bad one never reaches an order at all. An exchange reject for a
        malformed id is recoverable but arrives mid-session; a failure here
        arrives at startup or in a test.
        """
        # a missing identifier is a bug wherever the venue requires one
        if value is None or value == "":
            raise ValueError(f"{self.name}: {field} must not be empty")
        # anything in the venue's prohibited set, named individually
        bad = sorted({c for c in value if c in self.prohibited_chars})
        # report every offending character, so one fix clears them all
        if bad:
            raise ValueError(
                f"{self.name}: {field}={value!r} contains prohibited "
                f"character(s) {bad}; the exchange would reject the message")
        # non-printable characters are prohibited everywhere on every venue
        ctrl = sorted({ord(c) for c in value if ord(c) < 32 or ord(c) == 127})
        # likewise reported by code point, since they do not print
        if ctrl:
            raise ValueError(
                f"{self.name}: {field}={value!r} contains non-printable "
                f"character(s) at code points {ctrl}")
        # the value, so this can be used inline where an id is created
        return value

    @property
    @abstractmethod
    def feed_price_decimals(self) -> int:
        """How many decimal places the MARKET DATA feed carries.

        This is NOT the same as the tick grid and must not be assumed equal to
        the currency's minor unit. PSX quotes a 0.01 tick on the regular market
        -- two decimals -- while its market-data Price type is N13(4) and
        MDEntryPx is N18(6). A parser that assumes two decimals silently
        truncates anything finer, and silent truncation of a price is the
        quietest possible way to be wrong.
        """

    def parse_price(self, text: str) -> int:
        """Read a feed price into integer minor units, REFUSING to truncate.

        A value the minor unit cannot represent exactly is an error, not
        something to round. If PSX ever publishes a price with more precision
        than paisa, this raises and we find out immediately -- rather than
        trading for a year on prices that were quietly rounded.
        """
        # reject anything that is not a plain decimal number
        t = (text or "").strip()
        # an empty price is missing data, not a zero
        if not t:
            raise ValueError(f"{self.name}: empty price")
        # exactly one decimal point at most
        if t.count(".") > 1:
            raise ValueError(f"{self.name}: malformed price {text!r}")
        # split into whole and fractional parts
        whole, _, frac = t.partition(".")
        # the sign travels with the whole part
        neg = whole.startswith("-")
        # digits only from here
        whole_digits = whole.lstrip("+-")
        # a non-numeric price is a feed error worth naming
        if not whole_digits.isdigit() or (frac and not frac.isdigit()):
            raise ValueError(f"{self.name}: non-numeric price {text!r}")
        # how many decimals the minor unit can represent
        scale = len(str(self.minor_per_major)) - 1
        # anything beyond that must be zero, or precision would be lost
        if len(frac) > scale and frac[scale:].strip("0"):
            raise ValueError(
                f"{self.name}: price {text!r} carries more precision than the "
                f"minor unit can hold ({scale} decimals); refusing to truncate")
        # pad or trim the fraction to exactly the minor-unit scale
        frac_padded = (frac + "0" * scale)[:scale]
        # the integer value in minor units
        value = int(whole_digits or "0") * self.minor_per_major + \
            int(frac_padded or "0")
        # restore the sign
        return -value if neg else value

    @abstractmethod
    def format_price(self, price_minor: int) -> str:
        """Render an integer price as the venue's wire representation.

        The engine holds prices as integers precisely so that no float ever
        touches the tick grid. This is the single place that converts one back,
        and it is the venue's job because the number of decimals, the separator
        and the permitted characters are all venue rules.
        """

    # ---- what the exchange says the market is doing -----------------------
    @abstractmethod
    def parse_phase(self, code: str) -> "SecurityPhase":
        """Turn the venue's own trading-phase code into the shared enum.

        The codes are venue vocabulary; the enum is what the engine reasons
        about. An UNRECOGNISED code maps to UNKNOWN, which is treated as 'do
        not trade' -- never as 'probably fine'.
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
