# Copy per-arm settings without mutating the frozen manifest.
import copy
# Round allowable quantities down to whole shares.
import math
# Use the unchanged matching engine and latency implementation.
from mm_backtest import Backtester
# Exclude unpriced aggregate statistics from touch and queue position.
from stock_search_book import PricedBook
# Use the isolated source copy with the narrowly defined signal substitution.
from stock_search_micro import MicrostructureMM
# Pin all requested common control settings.
from stock_search_contract import COMMON, CHEAP, validate_strategy

# Freeze all twelve combinations before looking at outcomes.
ARMS = {f'{style}_{signal}': (ticks, threshold, depth) for style, ticks, threshold in (('OBI', 0.0, .15), ('QT15', 2.0, .15), ('QT20', 2.0, .20)) for signal, depth in (('best', None), ('w3', 3), ('w4', 4), ('w5', 5))}

# Apply the same one-tick and acquisition safeguards in every arm.
class Strategy(MicrostructureMM):
    # Obtain desired quotes before the engine sends any messages.
    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # Preserve the configured queue adjustment across temporary suppression.
        previous = self.queue_skew_ticks, self.queue_skew_bps, self.queue_skew_stairs
        # Detect a spread with no room to improve inside it.
        tight = bb is not None and ba is not None and ba - bb <= self.tick + 1e-10
        # Restore configuration even if quoting raises an exception.
        try:
            # Disable queue leaning on every one-tick book.
            if tight:
                # Keep throttle and defensive retreat enabled.
                self.queue_skew_ticks, self.queue_skew_bps, self.queue_skew_stairs = 0.0, 0.0, None
            # Run the otherwise unchanged strategy.
            want = super().quotes(bb, bq, ba, aq, pos, depth=depth)
        # Never leave a dynamic guard stuck on.
        finally:
            # Recover the stock's configured controls.
            self.queue_skew_ticks, self.queue_skew_bps, self.queue_skew_stairs = previous
        # Bound new orders before sending; sent orders remain executable.
        return self.capacity_guard(want) if hasattr(self, 'capacity_guard') else want

# Construct and verify each arm's actual settings.
def make_strategy(job, arm, session):
    # Read the predeclared combination.
    ticks, threshold, depth = ARMS[arm]
    # Preserve frozen size and calibration parameters.
    params = copy.deepcopy(job['params'])
    # Explicitly restore all common winning controls.
    params.update(COMMON)
    # Override the assigned label only as required by this exhaustive search.
    params.update(queue_skew_ticks=0.0 if job['symbol'] in CHEAP else ticks, queue_skew_thresh=threshold)
    # Instantiate the exact research strategy.
    strategy = Strategy(session_ms=session, **params)
    # Validate actual attributes, including acquisition controls and sizing.
    validate_strategy(strategy, dict(job, params=params))
    # Return settings alongside the runtime object for audit.
    return strategy, params

# Add fill-time observations and conservative send-time quantity checks.
class Engine(Backtester):
    # Bind capacity checking to this engine's actual working and pending orders.
    def __init__(self, strategy, cfg):
        # Keep the original engine initialization intact.
        super().__init__(strategy, cfg)
        # Keep all historical mutations, using consistent executable price views.
        self.book = PricedBook()
        # Do not permit opening buffers that are absent from the fill ledger.
        if self.pos != 0 or self.cash != 0:
            # Such a simulation needs explicit opening-lot accounting first.
            raise ValueError('Search requires zero starting inventory and cash')
        # Apply quantity checks only at quote decision time.
        strategy.capacity_guard = self.guard
        # Retain a compact count of capacity reductions.
        self.capacity_reductions = 0

    # Reserve old terms until they can no longer fill.
    def guard(self, want):
        # The estimate is a modelled capacity, not guaranteed future trading volume.
        cap = min(self.strat.max_inv, self.strat._pov_capacity() * self.strat.pov_cap_mult)
        # Reject corrupt capacity rather than silently bypassing it.
        if not math.isfinite(cap) or cap < 0:
            # Fail the cell with explicit calibration evidence.
            raise ValueError('Invalid remaining-volume capacity')
        # Copy the desired book before reducing quantities.
        result = dict(want)
        # Buy and sell fills are independent worst-case scenarios.
        for side, sign in (('BUY', 1), ('SELL', -1)):
            # Skip sides the strategy already pulled.
            if side not in result:
                # Let the usual cancel logic handle existing exposure.
                continue
            # Read proposed price and quantity.
            price, qty = result[side]
            # Include sent-but-not-yet-active and cancel-pending orders.
            working = list(self._side_orders(side))
            # An identical live quote needs no new reservation or message.
            if len(working) == 1 and working[0].price == price and working[0].qty == qty and working[0].cancel_at is None and working[0].amend_at is None and sign * self.pos + qty <= cap:
                # Preserve queue position when nothing needs changing.
                continue
            # Reserve every existing order until exchange processing removes it.
            reserved = sum(o.qty for o in working)
            # Outstanding amendments can restore quantity after partial fills.
            reserved += sum(p[3] for t, sequence, action, p in self.pending if action == 'AMEND' and p[0] == side)
            # Conservatively allow old fills plus the proposed amended remainder.
            allowed = max(0, math.floor(cap - sign * self.pos - reserved + 1e-9))
            # Do not enlarge a strategy's desired order.
            clipped = min(qty, allowed)
            # Record an actual capacity restriction.
            if clipped < qty:
                # Count decisions, not fictitious prevented fills.
                self.capacity_reductions += 1
            # Omit a side with no remaining headroom so normal cancellation runs.
            if clipped <= 0:
                # Never delete a sent order directly.
                result.pop(side)
            # Keep permitted whole-share quantity.
            else:
                # Only the next outbound request changes.
                result[side] = (price, clipped)
        # Return the risk-checked desired quotes.
        return result

    # Annotate only fills actually appended by the original matching method.
    def _record(self, method, *args, **kwargs):
        # Capture the pre-fill book, before the historical market event mutates it.
        bid, bq, ask, aq = self.book.bbo()
        # Keep unavailable or crossed midpoints explicitly missing.
        mid = (bid + ask) / 2 if bid is not None and ask is not None and bid < ask else None
        # Separate absent depth from locked and crossed reconstructed prices.
        source = 'missing_two_sided_book' if bid is None or ask is None else ('locked_book' if bid == ask else ('crossed_book' if bid > ask else 'pre_fill_book'))
        # Preserve the original execution order.
        start = len(self.fills)
        # Execute the unmodified matching logic exactly once.
        result = method(*args, **kwargs)
        # Attach observations without changing prices, sizes or matching decisions.
        for fill in self.fills[start:]:
            # Avoid a stale equity-table midpoint lookup.
            fill.update(mid0=mid, mid_source=source, pre_bid=bid, pre_ask=ask)
        # Preserve the original method's return value.
        return result

    # Observe passive executions without changing them.
    def _fill(self, *args, **kwargs):
        # Delegate through the common fill recorder.
        return self._record(super()._fill, *args, **kwargs)

    # Observe deliberate crossing fills at their actual book prices.
    def _taker_fill(self, *args, **kwargs):
        # Delegate through the same recorder.
        return self._record(super()._taker_fill, *args, **kwargs)

    # Preserve execution of limits that become marketable during latency.
    def _cross_on_arrival(self, *args, **kwargs):
        # Do not add a new arrival-time risk rejection.
        return self._record(super()._cross_on_arrival, *args, **kwargs)
