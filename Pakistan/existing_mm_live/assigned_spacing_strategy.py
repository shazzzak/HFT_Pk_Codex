# Import the unchanged original strategy implementation.
from micro_mm import MicrostructureMM as LegacyOriginal
# Import its reviewed isolated distance-signal hook.
from spacing_pnl_micro import MicrostructureMM as LegacyCandidate

# Apply the same one-tick spread rule to baseline and every weighted version.
class OneTickGuard:
    # Preserve the original strategy's public quote interface.
    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # Treat tiny floating-point representation errors as the same tick.
        tight = bb is not None and ba is not None and ba - bb <= self.tick + max(1e-10, self.tick * 1e-8)
        # Keep normal-spread behavior exactly unchanged.
        if not tight:
            # Delegate once to the original quoting implementation.
            return super().quotes(bb, bq, ba, aq, pos, depth)
        # Preserve all queue-skew modes so state cannot leak into the next quote.
        previous = self.queue_skew_ticks, self.queue_skew_bps, self.queue_skew_stairs
        # Restore controls even if downstream quote generation raises.
        try:
            # Disable fixed-tick, basis-point and staircase queue lean for this quote.
            self.queue_skew_ticks, self.queue_skew_bps, self.queue_skew_stairs = 0.0, 0.0, None
            # Preserve defensive retreat, size throttling and exit controls.
            return super().quotes(bb, bq, ba, aq, pos, depth)
        # Do not change the stock's assigned configuration permanently.
        finally:
            # Restore the exact original queue settings.
            self.queue_skew_ticks, self.queue_skew_bps, self.queue_skew_stairs = previous

# Use the spread guard on the original signal baseline.
class Original(OneTickGuard, LegacyOriginal):
    # Inherit all constructor and trading behavior through the reviewed mixin.
    pass

# Use the identical spread guard on the weighted-signal candidate.
class Candidate(OneTickGuard, LegacyCandidate):
    # Preserve the isolated hook and every other strategy rule.
    pass
