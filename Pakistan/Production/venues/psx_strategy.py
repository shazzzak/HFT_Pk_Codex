"""The bridge from the researched strategy to the production engine.

WHAT THIS IS. `MicrostructureMM` (micro_mm.py) is the strategy every backtest
result was produced by. This file wraps it without changing it, so that the
engine trades the same code the evidence is about. It translates, and that is
all it does:

    integer minor units  ->  float major units   (what micro_mm expects)
    micro_mm's dict      ->  DesiredQuotes        (what the order manager reads)

WHY A TRANSLATOR RATHER THAN A REWRITE. Every measured result -- the fee floor,
the three quoting groups, the closed cross-asset programme -- is evidence about
one particular implementation. Run a reimplementation live and the evidence no
longer describes what is trading, and the discrepancy shows up as a P&L
difference nobody can attribute. The cost of this choice is that floats live in
the trading path, and this file is where every one of them is converted. That
makes the conversion the highest-risk code in the adapter, so it is isolated in
two functions and tested directly.

WHY IT LIVES IN venues/ AND NOT core/. micro_mm prices in rupees and rounds to
two decimals in its own source. That is a PSX assumption, so anything wrapping
it is PSX-specific by definition.
"""
# for the floating-point grid check
import math
# typing only
from typing import Any, Optional, Sequence

# the domain objects on both sides of the translation
from core.model import BookSnapshot, DesiredQuotes, QuoteIntent, Side, Trade
# the strategy interface this satisfies
from core.strategy import Strategy
# the venue interface and the phase type
from core.venue import SecurityPhase, Venue


class TakerNotSupported(Exception):
    """The strategy asked to cross the spread and this engine cannot.

    Version 1 is maker-only: `QuoteIntent` describes resting orders and nothing
    else. micro_mm's age-cross feature deliberately emits a crossing order to
    flatten aged inventory, and an engine that posted that as an ordinary
    resting limit would be sending an aggressive order it believes is passive.
    Refusing loudly is the only safe response.
    """


class _TradeView:
    """The shape micro_mm.observe() expects a trade to have.

    micro_mm reads exactly two attributes off the trade object -- the aggressor
    side as the string "BUY"/"SELL", and the quantity. Rather than putting
    those names on the shared `Trade` model (where they would be a private
    detail of one strategy leaking into the domain), the adapter builds this.
    """
    # a fixed attribute set: no per-instance dict, so this costs about as
    # little as an object can while still being a real object
    __slots__ = ("aggressor_side", "qty")

    def __init__(self, aggressor_side: str, qty: int):
        # "BUY" or "SELL" -- the literal strings micro_mm tests against
        self.aggressor_side = aggressor_side
        # share count
        self.qty = qty


class MicroMMAdapter(Strategy):
    """Runs one `MicrostructureMM` instance for one symbol.

    The strategy object is INJECTED rather than constructed here. Two reasons:
    its forty-odd parameters are per-symbol calibration that belongs in a config
    table and not in this file, and injecting it means the adapter can be tested
    against a stub without micro_mm being importable at all.
    """

    def __init__(self, symbol: str, venue: Venue, strategy: Any,
                 *, reference_price_minor: int):
        # which instrument this instance quotes
        self._symbol = symbol
        # the venue, for the tick grid and the minor-unit scale
        self._venue = venue
        # the wrapped MicrostructureMM
        self._mm = strategy
        # how many minor units make one major unit: 100 paisa to the rupee.
        # Held as a float because every conversion into micro_mm divides by it.
        self._scale = float(venue.minor_per_major)
        # the exchange's current view of this security; UNKNOWN until told
        self._phase: Optional[SecurityPhase] = None

        # ---- refuse the configurations this engine cannot honour ----------
        # THE TAKER PATH. Checked here, at construction, rather than at the
        # first aged position: a config error found on day one at start-up is a
        # config error; found fifteen minutes into a position it is a surprise
        # crossing order.
        if getattr(strategy, "enable_age_cross", False):
            raise TakerNotSupported(
                f"{symbol}: enable_age_cross=True asks the strategy to cross "
                f"the spread, and this engine is maker-only. Either turn it "
                f"off, or add taker support to QuoteIntent first.")
        # same switch seen from the engine's side
        if getattr(strategy, "allow_taker", False):
            raise TakerNotSupported(
                f"{symbol}: strategy.allow_taker=True, and this engine is "
                f"maker-only.")
        # DOUBLE HYSTERESIS. micro_mm's tol_ticks holds a previous quote until
        # the ideal drifts far enough; the order manager's QuoteTolerance does
        # the same job on the same quantity. Both on means requotes are
        # suppressed twice and the effective tolerance is neither setting.
        # The order manager is where it lives, because that is where the cost
        # it exists to control -- lost queue position -- is actually paid.
        if getattr(strategy, "tol_ticks", 0.0):
            raise ValueError(
                f"{symbol}: strategy.tol_ticks="
                f"{strategy.tol_ticks} duplicates the order manager's "
                f"QuoteTolerance. Set tol_ticks=0.0 and configure the "
                f"tolerance on the order manager instead.")
        # THE TICK GRID. micro_mm carries its own `tick` as a float and floors
        # onto it. If that disagrees with the venue's grid, every price it
        # returns is off-grid and the exchange rejects it -- or worse, accepts
        # a rounded version we did not choose.
        venue_tick_minor = venue.tick_minor(reference_price_minor)
        # the same tick as micro_mm states it, in major units
        expected_tick = venue_tick_minor / self._scale
        # compare as floats, because micro_mm's is one
        if not math.isclose(float(getattr(strategy, "tick", 0.0)),
                            expected_tick, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{symbol}: strategy.tick={getattr(strategy, 'tick', None)} "
                f"but {venue.name} ticks at {expected_tick} around "
                f"{reference_price_minor}. These must agree.")
        # A TIERED GRID CANNOT USE THIS ADAPTER UNCHANGED. micro_mm holds ONE
        # tick for the whole session, so a venue whose tick varies with price
        # would be validated at the reference price and wrong everywhere else.
        # PSX is flat at one paisa, so this never bites here -- but it must
        # fail loudly rather than silently on the market where it would.
        for probe in (reference_price_minor // 4 or 1, reference_price_minor * 4):
            # the grid at a very different price level
            if venue.tick_minor(probe) != venue_tick_minor:
                raise ValueError(
                    f"{venue.name} has a price-dependent tick "
                    f"({venue_tick_minor} at {reference_price_minor}, "
                    f"{venue.tick_minor(probe)} at {probe}). micro_mm assumes "
                    f"one fixed tick, so it cannot be wrapped unchanged here.")
        # cached for the output grid check
        self._tick_minor = venue_tick_minor

    @property
    def symbol(self) -> Optional[str]:
        # the instrument this instance is bound to
        return self._symbol

    # ---- unit conversion: the whole risk surface of this file -------------
    def _to_major(self, price_minor: int) -> float:
        """Integer minor units -> the float major-unit price micro_mm wants."""
        # exact for every integer that fits a double, which covers any price
        return price_minor / self._scale

    def _to_minor(self, price_major: float) -> int:
        """micro_mm's float price -> an integer minor-unit price.

        THE ROUND IS NOT OPTIONAL, and this is measured, not a worry. Of the
        499,900 paisa prices between Rs 1.00 and Rs 5,000.00, 32,808 -- 6.6% --
        land just below a whole number when multiplied by 100, because the
        nearest double to a two-decimal value is very slightly under it. In the
        Rs 280-290 band where PPL trades the rate is 16%. Rs 280.03 times 100
        is 28002.999999999996, so `int(px * 100)` gives 28002: one paisa below
        what the strategy asked for, on that price and not on the one next to
        it. `int(round(px * 100))` gives 28003.
        """
        # the scaled value, still a float
        scaled = price_major * self._scale
        # the nearest integer to it
        as_int = int(round(scaled))
        # A price that is not within a rounding error of a whole minor unit did
        # not come from a paisa grid at all -- it is a strategy bug, and
        # silently snapping it would hide that.
        if abs(scaled - as_int) > 1e-6:
            raise ValueError(
                f"{self._symbol}: strategy returned {price_major!r}, which is "
                f"not on the minor-unit grid (scaled to {scaled!r})")
        # the integer price everything downstream uses
        return as_int

    # ---- market data ------------------------------------------------------
    def on_trade(self, trade: Trade) -> None:
        """A print in the market: advance the clock and the flow signals."""
        # A trade whose aggressor the feed did not publish carries no direction.
        # micro_mm ignores a side outside {"BUY","SELL"}, which is the correct
        # behaviour -- but it must still see the trade, because the timestamp
        # drives the quiet-market test.
        view = _TradeView(
            # the string micro_mm compares against, or a value it will ignore
            trade.aggressor.value if trade.aggressor is not None else "",
            # share count
            trade.quantity)
        # "T" is micro_mm's kind code for a trade; mid is not needed on a print
        self._mm.observe("T", view, trade.timestamp_ms, None)

    def on_book(self, book: BookSnapshot, position: int) -> DesiredQuotes:
        """A new book: update state, then say what we want resting.

        SPLIT IN TWO on purpose. The replay harness drives mm_backtest's own
        Backtester, which already calls strategy.observe() and already syncs
        the circuit limits before asking for a quote. Calling observe twice per
        event would double-count every volatility and flow update, so the
        harness calls quote() alone. Live, on_book() does both -- this is the
        live entry point and nothing else should be.
        """
        # state first
        self.observe_book(book)
        # then the decision
        return self.quote(book, position)

    def observe_book(self, book: BookSnapshot) -> None:
        """State update only: the clock, and the published circuit limits."""
        # THE CLOCK ADVANCES ON EVERY EVENT, not only on trades. micro_mm's
        # end-of-day and circuit-band triggers count down against self.now, and
        # self.now is only ever set by observe(). Skipping this on book updates
        # leaves the clock frozen between prints -- which on a quiet name means
        # the close arrives while the strategy still thinks it is mid-session.
        mid_major = (self._to_major(book.mid_minor)
                     if book.mid_minor is not None else None)
        # anything that is not "T" is a book event to micro_mm
        self._mm.observe("B", None, book.timestamp_ms, mid_major)
        # THE PUBLISHED CIRCUIT LIMITS. micro_mm advertises `wants_limits` and
        # reads self.limit_up / self.limit_dn; its lock trigger tests both for
        # None first, so an absent band correctly disables the trigger rather
        # than defaulting it to something.
        if getattr(self._mm, "wants_limits", False):
            # whatever the feed has published for this symbol, or None
            band = self._venue.price_band(self._symbol)
            # each bound converted independently: PSX publishes sentinels for
            # "no limit" which the venue has already turned into None
            self._mm.limit_up = (self._to_major(band.upper_minor)
                                 if band is not None
                                 and band.upper_minor is not None else None)
            # same for the floor
            self._mm.limit_dn = (self._to_major(band.lower_minor)
                                 if band is not None
                                 and band.lower_minor is not None else None)

    def quote(self, book: BookSnapshot, position: int) -> DesiredQuotes:
        """What we want resting, given this book and our position."""
        # HALTED, SUSPENDED, PRE-OPEN, CLOSED: want nothing resting. The risk
        # gateway would refuse the orders anyway, but refusing is not the same
        # as cancelling -- a strategy that keeps desiring quotes through a halt
        # leaves the order manager re-offering rejected orders forever while
        # whatever is already resting stays there.
        if self._phase is not None and not self._phase.is_tradeable:
            return DesiredQuotes.flat(self._symbol)

        # Nothing to quote against. Not an error: a one-sided book is a normal
        # state at the open, after a halt, and all day on a thin name.
        if not book.is_two_sided:
            return DesiredQuotes.flat(self._symbol)

        # the touch, in the units micro_mm works in
        bb = self._to_major(book.bids[0].price_minor)
        bq = book.bids[0].quantity
        ba = self._to_major(book.asks[0].price_minor)
        aq = book.asks[0].quantity
        # Ranked depth, only when the strategy is configured to use more than
        # one level. Building it unconditionally would allocate two lists per
        # book update for a feature that is off by default.
        depth = None
        if getattr(self._mm, "ofi_depth_levels", 1) > 1:
            # (bids, asks) as (price, qty) pairs, best first, in major units
            depth = (self._levels(book.bids), self._levels(book.asks))

        # what the strategy wants: {"BUY": (price, size), "SELL": (...)}
        wanted = self._mm.quotes(bb, bq, ba, aq, position, depth=depth)

        # THE TAKER GUARD, a second time. The constructor refused the switch,
        # but the switch is a mutable attribute and a sweep or a control-plane
        # update could set it after construction. The cost of checking is one
        # attribute read per book; the cost of not checking is an aggressive
        # order sent as though it were passive.
        if getattr(self._mm, "want_taker_side", None) is not None:
            raise TakerNotSupported(
                f"{self._symbol}: strategy asked to cross on the "
                f"{self._mm.want_taker_side} side; this engine is maker-only.")

        # translate each side, or leave it out entirely
        return DesiredQuotes(
            symbol=self._symbol,
            # the bid we want resting, if any
            bid=self._intent(wanted.get("BUY"), Side.BUY, book),
            # the offer we want resting, if any
            ask=self._intent(wanted.get("SELL"), Side.SELL, book))

    def _clamp_to_band(self, price_minor: int) -> int:
        """Bring a price inside the exchange's published circuit limits.

        Matching mm_backtest._requote, which does exactly this before sending.
        An ABSENT bound constrains nothing -- PSX publishes sentinels meaning
        'no limit' and the venue has already turned those into None, so a
        missing bound must not be treated as a limit of zero or of infinity.
        """
        # whatever the feed has published for this symbol, or None
        band = self._venue.price_band(self._symbol)
        # no band published: nothing to clamp against
        if band is None:
            return price_minor
        # never above the upper circuit limit
        if band.upper_minor is not None:
            price_minor = min(price_minor, band.upper_minor)
        # never below the lower one
        if band.lower_minor is not None:
            price_minor = max(price_minor, band.lower_minor)
        # inside every bound that exists
        return price_minor

    def _levels(self, levels: Sequence) -> list:
        """Book levels as the (price, qty) float pairs the OFI code reads."""
        # one pair per level, price converted, quantity left as shares
        return [(self._to_major(lv.price_minor), lv.quantity) for lv in levels]

    def _intent(self, quote, side: Side,
                book: BookSnapshot) -> Optional[QuoteIntent]:
        """One side of micro_mm's output as a QuoteIntent, with the checks.

        These checks are not defensive clutter. Everything below this point
        treats a QuoteIntent as a statement of fact -- the risk gateway checks
        limits, not sanity, and it has no idea what the book looked like. This
        is the last place a nonsensical price can be recognised as one.
        """
        # the side was omitted, which means we want nothing resting there
        if quote is None:
            return None
        # micro_mm returns (price in major units, size in shares)
        price_major, size = quote
        # the integer price, with the rounding trap handled in one place
        price_minor = self._to_minor(price_major)
        # THE CIRCUIT BAND IS CLAMPED, NOT REJECTED, AND THAT IS A DELIBERATE
        # MATCH TO THE BACKTEST. mm_backtest._requote clamps each desired price
        # into [limit_dn, limit_up] before sending, so a quote that would have
        # been outside the band still rests -- at the band edge. Rejecting it
        # instead would leave that side empty, which on the days micro_mm's lock
        # trigger fires is a different book from the one that was measured.
        #
        # RiskGateway's PriceBandCheck still rejects out-of-band orders. It is
        # the backstop, and after this clamp it should never fire: if it does,
        # the band moved between here and the gateway, which is worth knowing.
        price_minor = self._clamp_to_band(price_minor)
        # sizes come back as floats from round(); shares are whole
        quantity = int(round(size))
        # A zero or negative size is expressed by omitting the side, never by a
        # zero, and QuoteIntent refuses it -- but the message it would raise
        # says nothing about which strategy produced it.
        if quantity <= 0:
            raise ValueError(f"{self._symbol}: strategy returned "
                             f"{size!r} shares on the {side.value} side")
        # OFF THE GRID. Always true on PSX, where the tick is one paisa and
        # every integer is on-grid; it is the venue where it is not that this
        # catches.
        if price_minor % self._tick_minor:
            raise ValueError(
                f"{self._symbol}: {side.value} price {price_minor} is not a "
                f"multiple of the {self._tick_minor} tick")
        # CROSSING. micro_mm clips both sides to stay inside the touch, so this
        # should be unreachable -- which is exactly why it is worth asserting:
        # if that clip is ever changed, the failure is a marketable order sent
        # by a strategy that believes it is quoting passively.
        if side is Side.BUY and price_minor >= book.asks[0].price_minor:
            raise ValueError(
                f"{self._symbol}: bid {price_minor} would cross the offer at "
                f"{book.asks[0].price_minor}")
        # the mirror
        if side is Side.SELL and price_minor <= book.bids[0].price_minor:
            raise ValueError(
                f"{self._symbol}: offer {price_minor} would cross the bid at "
                f"{book.bids[0].price_minor}")
        # a well-formed desire
        return QuoteIntent(side=side, price_minor=price_minor,
                           quantity=quantity)

    # ---- state the engine feeds in ---------------------------------------
    def on_phase(self, phase: SecurityPhase) -> None:
        # remember it; on_book gates on it
        self._phase = phase

    def on_session_start(self, date: str) -> None:
        """A new day.

        NOT IMPLEMENTED ON PURPOSE. micro_mm carries a day's worth of state --
        the volatility EMA, the flow and OFI windows, the spread EMA, the
        session boundaries t0/t1, the per-day volume profile and the trigger
        counters -- and there is no reset method on it. Clearing some of that
        and not the rest would be worse than not clearing it at all, because
        the result would look like a fresh strategy while carrying yesterday's
        volatility into this morning's quote widths.

        So a new day means a NEW STRATEGY OBJECT, built from that day's
        calibration. This method exists to say so where someone would look.
        """
        raise NotImplementedError(
            f"{self._symbol}: micro_mm has no reset. Build a new strategy "
            f"object and a new adapter for each trading day.")
