"""The strategy seam: what every market-making strategy must provide.

VENUE-AGNOSTIC. Nothing here knows about PSX or about any particular signal.
A second market reuses this file unchanged; a second strategy implements it.

WHY THIS INTERFACE IS SO SMALL. The strategy's only output is a statement of
what it wants resting. It does not send orders, does not cancel, does not know
what is currently on the exchange, and cannot reach the wire. Everything about
turning a desire into messages -- the diff, the in-flight rule, the cancel
ordering, the risk checks -- lives in the order manager and the risk gateway,
where it can be tested once instead of once per strategy.

That split is also what makes a backtest meaningful: the same strategy object
produces the same desires from the same book whether the book came from a file
or from a socket.
"""
# the abstract base machinery
from abc import ABC, abstractmethod
# typing only
from typing import Optional

# the domain objects a strategy reads and writes
from core.model import BookSnapshot, DesiredQuotes, Fill, Trade
# the exchange's view of what the market is doing
from core.venue import SecurityPhase


class Strategy(ABC):
    """One symbol's quoting logic.

    ONE INSTANCE PER SYMBOL. Deliberate, and it is the decision that the whole
    cross-asset research programme settled: futures lead-lag, sector lead-lag,
    ETF hedging and a portfolio beta overlay were all measured and all came out
    negative after fees, so no strategy here needs to see another symbol's
    state. Keeping it that way means 113 independent quoters that cannot
    contaminate each other, rather than one object holding the whole market.
    """

    # ---- market data ------------------------------------------------------
    @abstractmethod
    def on_book(self, book: BookSnapshot, position: int) -> DesiredQuotes:
        """A new book. Return what we want resting NOW, given our position.

        `position` is passed in rather than tracked here because the order
        manager owns it, and it owns it because it comes from fills -- never
        from what we believe we sent. A strategy keeping its own copy is a
        second version of the truth that drifts on the first missed fill.

        The return value is COMPLETE: a side left out means "nothing should be
        resting there". There is no way to say "leave that side alone", because
        a protocol with no way to say 'stop' cannot be made to go flat.
        """

    @abstractmethod
    def on_trade(self, trade: Trade) -> None:
        """A trade printed in the market. State update only -- no quote.

        Separate from `on_book` because a print and a book change are different
        events arriving at different times, and a strategy that only learns the
        time from book updates has a clock that stops whenever the book does.
        """

    # ---- everything below has a safe default ------------------------------
    def on_fill(self, fill: Fill) -> None:
        """One of OUR orders executed.

        Default does nothing: position is passed to `on_book`, so a strategy
        that only cares about inventory needs no fill handler at all. Override
        when the strategy needs the execution itself -- the price it got, or
        when it got it -- rather than the resulting position.
        """
        # nothing by default
        return None

    def on_phase(self, phase: SecurityPhase) -> None:
        """The exchange changed the market state for this symbol.

        Default does nothing. A strategy does not have to gate itself on the
        phase: the risk gateway refuses orders outside continuous trading in
        any case. But a strategy that DOES stop desiring quotes during a halt
        makes the order manager emit cancels instead of the gateway rejecting
        the same order over and over, which is both correct and quieter.
        """
        # nothing by default
        return None

    def on_session_start(self, date: str) -> None:
        """A new trading day. Clear anything carried from yesterday.

        Default does nothing. Any strategy holding rolling windows, volatility
        estimates or end-of-day timers MUST override this: a volatility EMA
        carried across an overnight gap treats the gap as an intraday move, and
        a session countdown carried from yesterday is already expired.
        """
        # nothing by default
        return None

    @property
    def symbol(self) -> Optional[str]:
        """Which instrument this instance quotes, when it is bound to one."""
        # unknown unless a subclass says otherwise
        return None
