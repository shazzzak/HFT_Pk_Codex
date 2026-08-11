"""
halt_state.py

Single source of truth for PSX trading-state classification, shared by the backtest
and the live trader so both apply identical rules.

Three distinct mechanisms (do NOT conflate):
  1. MARKET HALT      -- KSE-30 index circuit breaker (+/-5% or +/-7.5% held 5 min).
                         All trading suspended. In data: misc FIX 8538='H' AND per-stock
                         phase 'TEMPORARY_SUSPENSION'. NOT TRADEABLE.
  2. TRADING SUSPENSION-- manual/disciplinary, days-weeks. In data: suspended_all_day.
                         NOT TRADEABLE.
  3. SCRIP CIRCUIT BREAKER -- +/-10% (or PKR 1.00, whichever larger) daily price lock.
                         Stock KEEPS trading (phase stays CONTINUOUS_AUCTION) but cannot
                         move past the band; book goes one-sided at the cap. TRADEABLE but
                         TOXIC for passive MM -> flagged, not silently kept.

Every function is pure (row-in / state-out) so it runs the same on a historical parquet
row and a live feed snapshot.
"""

# numeric helpers
import math

# ---- state constants ----
CONTINUOUS_NORMAL = "CONTINUOUS_NORMAL"    # normal continuous trading, quote freely
NEAR_UPPER_LIMIT  = "NEAR_UPPER_LIMIT"     # within the warning band below the upper cap
NEAR_LOWER_LIMIT  = "NEAR_LOWER_LIMIT"     # within the warning band above the lower cap
AT_UPPER_LOCK     = "AT_UPPER_LOCK"        # scrip locked up (no offers / bid at cap)
AT_LOWER_LOCK     = "AT_LOWER_LOCK"        # scrip locked down (no bids / ask at cap)
MARKET_HALT       = "MARKET_HALT"          # index circuit breaker: whole market suspended
STOCK_HALT        = "STOCK_HALT"           # per-stock temporary suspension
STOCK_SUSPENDED   = "STOCK_SUSPENDED"      # manual all-day suspension
NON_CONTINUOUS    = "NON_CONTINUOUS"       # auction / break / closed / starting

# states in which a passive market maker may quote normally
_TRADEABLE_NORMAL = {CONTINUOUS_NORMAL, NEAR_UPPER_LIMIT, NEAR_LOWER_LIMIT}


# =====================================================================
# SCRIP CIRCUIT-BREAKER BAND (from prev close)
# =====================================================================

def scrip_band(prev_close):
    # PSX scrip band: +/-10% of prev close, OR +/-PKR 1.00, whichever is WIDER
    if prev_close is None or (isinstance(prev_close, float) and math.isnan(prev_close)):
        # no reference -> no band
        return (None, None)
    # ten-percent half-width
    pct_width = 0.10 * prev_close
    # absolute PKR 1.00 half-width
    abs_width = 1.00
    # the effective half-width is the larger of the two
    width = max(pct_width, abs_width)
    # lower and upper caps
    return (prev_close - width, prev_close + width)


# =====================================================================
# ROW / SNAPSHOT STATE CLASSIFIER  (pure)
# =====================================================================

def classify(stock_phase, market_phase, suspended_all_day,
             best_bid, best_ask, upper_cap, lower_cap, near_frac=0.02):
    # ---- highest-priority, hard non-tradeable states first ----
    # manual all-day suspension dominates everything
    if suspended_all_day:
        return STOCK_SUSPENDED
    # market-wide index halt (misc FIX 8538='H')
    if market_phase == "H":
        return MARKET_HALT
    # per-stock temporary suspension (halt window)
    if stock_phase == "TEMPORARY_SUSPENSION":
        return STOCK_HALT
    # any non-continuous session state (auction / break / closed / starting)
    if stock_phase != "CONTINUOUS_AUCTION":
        return NON_CONTINUOUS

    # ---- from here we are in continuous trading; check scrip lock / proximity ----
    # a valid best bid means offers-side may be empty (locked up), and vice versa
    has_bid = best_bid is not None and not _isnan(best_bid)
    has_ask = best_ask is not None and not _isnan(best_ask)

    # locked UP: buyers stacked at the cap, no sellers (one-sided book at/above upper cap)
    if upper_cap is not None and not _isnan(upper_cap):
        # no offers at all, or the best bid has reached the upper cap
        if (not has_ask and has_bid) or (has_bid and best_bid >= upper_cap - 1e-9):
            return AT_UPPER_LOCK
    # locked DOWN: sellers stacked at the cap, no buyers
    if lower_cap is not None and not _isnan(lower_cap):
        # no bids at all, or the best ask has reached the lower cap
        if (not has_bid and has_ask) or (has_ask and best_ask <= lower_cap + 1e-9):
            return AT_LOWER_LOCK

    # ---- near-limit warning zone (both caps and a two-sided book present) ----
    if has_bid and has_ask and upper_cap and lower_cap and not _isnan(upper_cap):
        # mid reference for distance measurement
        mid = 0.5 * (best_bid + best_ask)
        # fractional distance from mid up to the upper cap
        if (upper_cap - mid) / mid <= near_frac:
            return NEAR_UPPER_LIMIT
        # fractional distance from mid down to the lower cap
        if (mid - lower_cap) / mid <= near_frac:
            return NEAR_LOWER_LIMIT

    # ---- otherwise: healthy continuous trading ----
    return CONTINUOUS_NORMAL


def is_tradeable(state):
    # True only where a passive maker should quote normally
    return state in _TRADEABLE_NORMAL


# small nan helper that tolerates None and non-floats
def _isnan(x):
    # only floats can be nan; anything else is "not nan"
    return isinstance(x, float) and math.isnan(x)


# =====================================================================
# LIVE KSE-30 MARKET-HALT EARLY-WARNING MONITOR (stateful)
# =====================================================================

class KSE30HaltMonitor:
    """
    Anticipates a market-wide halt from the KSE-30 index BEFORE the exchange's 'H' message,
    by replicating the trigger rule: index >= +/-5% (or +/-7.5%) from prev close, held for
    5 consecutive minutes, and NOT within the end-of-day exemption window.
    Feed it index observations as they arrive; it returns OK / WARNING / HALT_LIKELY.
    In the backtest, prefer the authoritative 'H' message; use this only for live anticipation.
    """

    # construct with the rule parameters (defaults per PSX spec you provided)
    def __init__(self, thresholds=(0.05, 0.075), hold_seconds=300, eod_exempt_seconds=90 * 60):
        # the two trigger thresholds (5% and 7.5%)
        self.thresholds = thresholds
        # how long the index must stay beyond a threshold to trigger (5 minutes)
        self.hold_seconds = hold_seconds
        # end-of-day exemption: no halt if this close to the bell (last ~1-2 hours)
        self.eod_exempt_seconds = eod_exempt_seconds
        # timestamp when the index first went beyond the (smallest) threshold, else None
        self._breach_start_ts = None

    # feed one index observation; returns the current warning state
    def update(self, ts_seconds, index_value, prev_close_index, seconds_to_close):
        # guard against missing reference
        if prev_close_index in (None, 0) or _isnan(prev_close_index):
            return "OK"
        # signed percentage move from previous close
        pct = (index_value - prev_close_index) / prev_close_index
        # are we beyond the smallest trigger threshold in either direction?
        beyond = abs(pct) >= min(self.thresholds)
        # end-of-day exemption: too close to the bell -> halts are not triggered
        in_eod_exemption = seconds_to_close <= self.eod_exempt_seconds

        # if not beyond threshold (or in EOD exemption), reset the breach clock
        if not beyond or in_eod_exemption:
            # clear any running breach
            self._breach_start_ts = None
            # explicitly OK during the exemption even if beyond threshold
            return "OK"

        # we are beyond threshold and outside the exemption -> start/continue the breach clock
        if self._breach_start_ts is None:
            # mark the first moment of the breach
            self._breach_start_ts = ts_seconds
        # how long have we been continuously beyond threshold?
        held = ts_seconds - self._breach_start_ts
        # once held long enough, a halt is (per rule) imminent/active
        if held >= self.hold_seconds:
            return "HALT_LIKELY"
        # beyond threshold but not yet long enough
        return "WARNING"
