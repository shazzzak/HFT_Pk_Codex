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
from core.venue import (MarketPhase, PriceBand, SecurityPhase,
                        SessionSegment, Venue)


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

    # ---- what the exchange says the market is doing -----------------------
    # PUBLISHED. PSX FIX Market Data Interface Specification v1.05 (2 Apr 2024),
    # TradingPhaseCode (tag 8538), published every 3 seconds on the Trading
    # Session Status message and carried on every snapshot.
    _PHASE_0 = {
        # before the market opens
        "S": MarketPhase.STARTING,
        # pre-open call auction, morning
        "O": MarketPhase.PRE_OPEN,
        # pre-open call auction, afternoon after the Friday lunch break
        "N": MarketPhase.PRE_OPEN,
        # pre-open call auction when trading resumes after a halt
        "V": MarketPhase.PRE_OPEN,
        # continuous auction -- the only phase we quote in
        "T": MarketPhase.CONTINUOUS,
        # a scheduled break, including the Friday prayer break
        "B": MarketPhase.BREAK,
        # an unscheduled halt or a security suspension
        "H": MarketPhase.HALTED,
        # pre-close call auction (the spec marks this reserved)
        "C": MarketPhase.PRE_CLOSE,
        # after-hours trading
        "A": MarketPhase.POST_CLOSE,
        # shut
        "E": MarketPhase.CLOSED,
    }
    # the 2nd digit, revealed only when the market is on a break
    _BREAK_REASON = {
        "1": "after pre-open",
        "2": "Friday lunch break",
        "3": "after the afternoon pre-open on Friday",
        "4": "market close before post-close",
    }

    def parse_phase(self, code: str) -> SecurityPhase:
        # a missing code is not an open market
        if not code:
            return SecurityPhase(phase=MarketPhase.UNKNOWN)
        # 0th digit: what the market as a whole is doing. An unrecognised
        # letter maps to UNKNOWN, which the risk gateway treats as closed --
        # a code we cannot read is never assumed to mean 'open'.
        phase = self._PHASE_0.get(code[0], MarketPhase.UNKNOWN)
        # 1st digit: '1' means THIS SECURITY is suspended for the whole day.
        # Carried on the snapshot, so it is per instrument, not per market.
        suspended = len(code) > 1 and code[1] == "1"
        # 2nd digit: why the break, revealed only when the 0th digit is 'B'
        reason = (self._BREAK_REASON.get(code[2])
                  if phase is MarketPhase.BREAK and len(code) > 2 else None)
        # the complete state of this instrument
        return SecurityPhase(phase=phase, suspended_all_day=suspended,
                             break_reason=reason)

    @property
    def feed_price_decimals(self) -> int:
        # PUBLISHED. Market Data spec v1.05, Data Dictionary: Price is N13(4)
        # and MDEntryPx is N18(6). The regular market's TICK is 0.01, so equity
        # prices arrive with trailing zeros -- but the FIELD carries more, and
        # index values use the full six. parse_price() refuses to truncate
        # anything the minor unit cannot hold rather than rounding it away.
        return 6

    # ---- what the wire will and will not accept ---------------------------
    @property
    def supports_replace(self) -> bool:
        # PUBLISHED. PSX FIX Specification v1.2, "Order Cancel Replace Request"
        # (MsgType 'G'): "Cancel/Replace will be used to change any valid
        # attribute of an open order (i.e. reduce/increase quantity, change
        # limit price, change instructions, etc.)"
        #
        # NOTE ON WHAT THIS DOES AND DOES NOT BUY. It removes the round trip in
        # which we have cancelled and not yet replaced -- a window where we are
        # not quoting at all. It does NOT buy queue position; see the three
        # properties below, which the rulebook answers explicitly.
        return True

    # ---- what an amendment does to queue position -------------------------
    # PSX Regulations, 8.5.2. The rulebook splits the cases, and so do these.
    #
    # 8.5.1(d) makes Change Former Order the ONLY modification path: 8.12.1
    # says "the terms of an Order placed in the Trading System can only be
    # modified through the CFO option", and 8.12.2 that it "can only modify
    # price and volume of an unfilled/outstanding Order in whole or in parts".
    # So there is no amend-without-CFO and no partial exemption to find.
    #
    # THE ONE CARVE-OUT IS A QUANTITY REDUCTION. Everything else -- any price
    # change, any increase -- is accepted as a single message and then placed
    # at the back of the FIFO queue for the price level it lands on. The
    # message count halves; the priority does not survive.

    @property
    def replace_price_keeps_priority(self) -> bool:
        # PUBLISHED, 8.5.2. A price change strips time priority: the amended
        # order joins the back of the queue at the new price level. This is
        # also the only answer consistent with 8.4.1's price-then-time
        # priority -- a reprice that kept its place would let an order buy
        # priority at a price it never queued at.
        return False

    @property
    def replace_qty_up_keeps_priority(self) -> bool:
        # PUBLISHED, 8.5.2. Only REDUCTION is carved out, so an increase is
        # treated like any other modification and re-queues at the back. The
        # asymmetry is deliberate on the exchange's part: letting size grow
        # without cost would make a one-share order a cheap option on the
        # front of the queue.
        return False

    @property
    def replace_qty_down_keeps_priority(self) -> bool:
        # PUBLISHED, 8.5.2, and consistent with 8.4.4 (a partial fill keeps
        # priority on the remainder -- which is a reduction the market made
        # rather than one we asked for). Reducing takes nothing from anyone
        # behind us, so the order is amended in place and keeps its position.
        #
        # THIS IS THE ONLY CASE WHERE THE AMENDMENT IS FREE, and it is the
        # reason to have the path at all.
        return True

    @property
    def requires_account(self) -> bool:
        # PUBLISHED. PSX FIX Specification v1.2, New Order Single: Account
        # (tag 1) is Required = Y, and carries the "Client Code" agreed between
        # broker and exchange. Without it every order is rejected.
        return True

    @property
    def prohibited_chars(self) -> frozenset:
        # PUBLISHED. PSX FIX Specification v1.2, Appendix C. These are rejected
        # by the FIX engine in Account (1), Symbol (55), ClOrdID (11),
        # OrderID (37), Price (44), StopPx (99) and LastPx (31).
        #
        # The '.' is listed as prohibited WITH AN EXCEPTION: price fields may
        # carry exactly one. It is therefore excluded from this set and the
        # single-occurrence rule is enforced in format_price() below, where the
        # only '.' this engine ever emits is produced.
        return frozenset(";|`~#^'%*,?")

    def format_price(self, price_minor: int) -> str:
        # paisa to rupees: the integer is exact, and dividing by 100 by string
        # surgery rather than by float arithmetic keeps it exact
        if price_minor < 0:
            raise ValueError(f"PSX: price_minor must not be negative, got "
                             f"{price_minor}")
        # whole rupees
        major = price_minor // self.minor_per_major
        # the remainder, zero-padded to the currency's two decimal places
        minor = price_minor % self.minor_per_major
        # exactly one '.', which is what Appendix C's exception permits
        return f"{major}.{minor:02d}"

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
