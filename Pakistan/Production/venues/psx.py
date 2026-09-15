"""Pakistan Stock Exchange: the venue-specific half.

Everything PSX-specific lives here and nowhere else. Adding a second market
means adding a sibling of this file, not editing anything in core/.

WHAT IS MEASURED AND WHAT IS NOT. The numbers below are marked as one of:
  MEASURED  -- established from our own parsed store, with the finding on record
  PUBLISHED -- from the exchange or the regulator
  UNKNOWN   -- not yet established; the code must not pretend otherwise

An UNKNOWN is represented as None rather than as a guessed default. A guessed
default is indistinguishable from a measured one once it is in the code, and
the first time anyone notices is when it turns out to be wrong.
"""
# typing only
from typing import Callable, Dict, Optional, Sequence
# the venue interface and its value objects
from core.venue import PriceBand, SessionSegment, Venue


class PSXVenue(Venue):
    """PSX rules for the market-making engine.

    Two things are injected rather than hard-coded, because both change daily
    and neither belongs in source control:

      * `session_provider` returns the continuous-trading segments for a date.
        PSX breaks for Jumu'ah on Fridays, so a Friday has TWO segments and the
        tradeable time left is the sum over them. This is already produced by
        the calibration pipeline as `session_segments_*.csv`.

      * `band_provider` returns the exchange's published circuit limits for a
        symbol. These arrive on the market-data feed as the UPPER and LOWER
        CIRCUIT_BREAKER rows and move intraday, so they are read live rather
        than configured.
    """

    # MEASURED: the tick is a flat 0.01 PKR at every price on PSX, unlike the
    # tiered grids used by several regional exchanges. One paisa.
    _TICK_MINOR = 1

    # MEASURED: all-in transaction cost is 1.554 bps for a ROUND TRIP under the
    # TREC fee schedule, so one side is half of that. Stated per side here
    # because that is what a quoting decision needs; halving it at each call
    # site is how a round-trip number ends up charged twice.
    _FEE_BPS_PER_SIDE = 1.554 / 2.0

    def __init__(self,
                 session_provider: Callable[[str], Sequence[SessionSegment]],
                 band_provider: Optional[
                     Callable[[str], Optional[PriceBand]]] = None,
                 algo_code: Optional[str] = None,
                 require_algo_tag: bool = False):
        # returns the continuous-trading segments for a given 'YYYY-MM-DD'
        self._session_provider = session_provider
        # returns the live circuit band for a symbol, or None if unpublished
        self._band_provider = band_provider
        # the algorithm identifier to stamp on outbound orders, if we have one
        self._algo_code = algo_code
        # whether the tag is MANDATORY. Off by default: the SECP framework that
        # would have required it is not being pursued. The FIELD stays in the
        # order path regardless -- see requires_algo_tag below for why.
        self._require_algo_tag = require_algo_tag

    # ---- identity ---------------------------------------------------------
    @property
    def name(self) -> str:
        # short code for logs and the audit trail
        return "PSX"

    @property
    def minor_per_major(self) -> int:
        # 100 paisa to the rupee; every price in the engine is in paisa
        return 100

    # ---- the price grid ---------------------------------------------------
    def tick_minor(self, price_minor: int) -> int:
        # flat grid: the price argument is ignored, which is the point of the
        # interface taking one at all -- a tiered venue uses it
        return self._TICK_MINOR

    # ---- the trading day --------------------------------------------------
    def sessions(self, date: str) -> Sequence[SessionSegment]:
        # delegate to the injected calendar
        segments = self._session_provider(date)
        # a date with no segments is not a quiet day, it is missing data, and
        # silently returning nothing would make the engine think the market is
        # shut rather than that we cannot tell
        if not segments:
            raise ValueError(f"PSX: no session segments for {date}; the "
                             f"calendar is missing, not empty")
        # the day's continuous-trading intervals
        return segments

    # ---- costs ------------------------------------------------------------
    @property
    def fee_bps_per_side(self) -> float:
        # half the measured round-trip cost
        return self._FEE_BPS_PER_SIDE

    # ---- regulatory and exchange limits -----------------------------------
    @property
    def max_orders_per_second(self) -> Optional[int]:
        # UNKNOWN, and now for a plainer reason than before. No regulatory cap
        # is coming -- the SECP framework is not being pursued -- so the only
        # message-rate ceiling that exists is whatever the EXCHANGE or our
        # BROKER enforces on the session. That is a real limit and exceeding it
        # gets orders rejected or the session throttled; we simply have not
        # been told the number. Ask the broker. None means 'not told', which
        # makes the gateway fall back to a house limit rather than send at an
        # unbounded rate.
        return None

    @property
    def max_order_to_trade_ratio(self) -> Optional[float]:
        # UNKNOWN, and no longer a compliance number at all. With no regulator
        # penalising a high ratio, this matters for two practical reasons:
        # a broker or exchange that charges per message or throttles a chatty
        # session, and -- the larger one -- QUEUE POSITION. Every cancel and
        # repost sends us to the back of the queue at that price. See
        # OrderToTradeRatioCheck in core/risk.py.
        return None

    def price_band(self, symbol: str) -> Optional[PriceBand]:
        # no provider wired up yet: say so rather than inventing a band
        if self._band_provider is None:
            return None
        # the live published limits for this symbol
        return self._band_provider(symbol)

    # ---- order tagging ----------------------------------------------------
    @property
    def requires_algo_tag(self) -> bool:
        # NOT CURRENTLY REQUIRED. SECP concept paper s4 proposed that the
        # exchange issue a unique code per algorithm and that every order carry
        # it; that framework is not being pursued, so this defaults to False.
        #
        # The FIELD stays in the order path anyway, and that is a deliberate
        # cheap bet rather than dead code: retrofitting a mandatory tag across
        # a live order path is disruptive, carrying an unused optional one
        # costs nothing, and the day a regulator or a broker asks for order
        # attribution this becomes a constructor argument rather than a change
        # to the encoder, the OMS and every test that touches them.
        return self._require_algo_tag

    @property
    def algo_code(self) -> Optional[str]:
        """The algorithm identifier stamped on outbound orders, if any."""
        # None is the honest answer when nobody has issued or required one
        return self._algo_code


def static_session_provider(
        segments_by_date: Dict[str, Sequence[SessionSegment]]
) -> Callable[[str], Sequence[SessionSegment]]:
    """Build a session provider from an in-memory map.

    Used by tests and by the replay harness. The live engine passes a provider
    backed by the calibration file instead.
    """
    # look the date up, returning an empty sequence so PSXVenue can raise the
    # specific 'calendar missing' error rather than a bare KeyError
    def provider(date: str) -> Sequence[SessionSegment]:
        # the segments for that date, if any
        return segments_by_date.get(date, ())
    # the closure
    return provider
