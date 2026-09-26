# Reuse the strict incremental reconstruction and channel sequence checks.
from psx_reference_book import ReferenceBook, SequenceGate, ReconstructionError

# Stop quoting after missing data without pretending existing exposure disappeared.
class BookHealth:
    # Bind the ordinary desired-quote cancellation callback, never an exchange shortcut.
    def __init__(self, channel, cancel_quotes):
        # Retain the caller's ordinary quote-manager callback.
        self.cancel_quotes = cancel_quotes
        # Start with the documented first sequence, not an arbitrary observed sequence.
        self.gate = SequenceGate(channel)
        # Reconstruct only explicitly reported orders and their referenced reductions.
        self.book = ReferenceBook()
        # Require an externally verified opening state before enabling quotes.
        self.opening_verified = False
        # Persist a sequence or reference failure until verified recovery is implemented.
        self.blocked = False
        # Record the first failure without manufacturing a profit adjustment.
        self.failure = None
        # Remember quote permission independently for each stock on this channel.
        self.permissions = {}
        # Preserve transitions for coverage and operations reports.
        self.transitions = []
        # Reject backward local observation clocks.
        self.last_observed_ms = None

    # Evaluate quote permission without treating sparsity as data corruption.
    def can_quote(self, symbol, phase):
        # Neither a later snapshot nor an ordinary tick clears a missing-message failure.
        if self.blocked or not self.opening_verified or phase != 'CONTINUOUS_AUCTION':
            # Keep new quote requests disabled.
            return False
        # Obtain only exact, reported price levels.
        bid, ask = self.book.bbo(symbol)
        # Require two sides and a positive spread; a one-tick spread is still a valid book.
        return bid is not None and ask is not None and bid[0] < ask[0]

    # Confirm an empty opening only before consuming any events, with explicit evidence.
    def confirm_empty_open(self, evidence):
        # Require an auditable external observation, not an inferred sequence start.
        if not isinstance(evidence, str) or not evidence.strip():
            # Sequence one alone does not prove that no overnight orders exist.
            raise ValueError('Verified empty-opening evidence is required')
        # Never use an opening declaration as an after-gap reset.
        if self.blocked or self.gate.last_event is not None:
            # Recovery needs a separately verified checkpoint and account reconciliation.
            raise ValueError('Cannot declare an empty opening after replay starts')
        # Preserve the verification identity for downstream evidence.
        self.opening_evidence = evidence
        # Permit subsequent healthy continuous-session books to quote.
        self.opening_verified = True

    # Latch failures before any event can be used to simulate another fill.
    def _block(self, observed_ms, reason):
        # Keep only the first failure as the start of the uncertain interval.
        if not self.blocked:
            # Disable quote permissions before calling any external component.
            self.blocked = True
            # Preserve the local detection time, not a fabricated earlier detection.
            self.failure = dict(detected_ms=observed_ms, reason=reason)
            # Record that the restart time remains unknown.
            self.transitions.append(dict(symbol=None, start_ms=observed_ms, end_ms=None, reason=reason))
            # Cancel every symbol on the failed channel through the normal quote manager.
            self.cancel_quotes(None)
        # No live order, pending order, cash balance or position is deleted here.
        return False

    # Process the complete channel before symbol filtering.
    def on_event(self, event, observed_ms, phase):
        # A latched stream cannot become healthy merely because another message arrived.
        if self.blocked:
            # Keep the existing uncertainty explicit.
            return False
        # Validate observation time before mutating the book.
        if not isinstance(observed_ms, int) or (self.last_observed_ms is not None and observed_ms < self.last_observed_ms):
            # Treat a broken observation clock as unusable input.
            return self._block(observed_ms, 'INVALID_OBSERVATION_CLOCK')
        # Preserve the current local observation time.
        self.last_observed_ms = observed_ms
        # Catch failures from both sequence and reference validation.
        try:
            # Ignore an identical immediate retransmission without applying it twice.
            fresh = self.gate.accept(event)
            # Only a newly accepted record may change the historical book.
            if fresh:
                # Atomically validate every referenced leg before any mutation.
                self.book.apply(event)
        # Make bad data a persistent no-quote state rather than an invented order.
        except (ReconstructionError, ValueError, TypeError) as error:
            # Preserve the exact failure in the audit trail.
            return self._block(observed_ms, str(error))
        # Decide permission from the current phase and reconstructed prices.
        allowed = self.can_quote(event.symbol, phase)
        # Detect a transition out of a previously quotable state.
        if self.permissions.get(event.symbol, False) and not allowed:
            # Ask the regular order manager to cancel this stock's quotes.
            self.cancel_quotes(event.symbol)
            # Record temporary price/phase unavailability separately from a channel gap.
            self.transitions.append(dict(symbol=event.symbol, start_ms=observed_ms, end_ms=None, reason='BOOK_OR_PHASE_NOT_QUOTABLE'))
        # Close a temporary pause only when a valid continuous book actually returns.
        if allowed and not self.permissions.get(event.symbol, False):
            # Find this stock's most recent still-open temporary pause.
            for interval in reversed(self.transitions):
                # Never close channel failures from an ordinary book update.
                if interval['symbol'] == event.symbol and interval['end_ms'] is None:
                    # Preserve the actual local recovery observation time.
                    interval['end_ms'] = observed_ms
                    # Only one temporary interval can remain open per stock.
                    break
        # Retain the new permission for the next transition.
        self.permissions[event.symbol] = allowed
        # Tell the caller whether new quotes are allowed, not whether old orders vanished.
        return allowed

    # Respond to an independently monitored channel timeout without waiting for a new tick.
    def on_timeout(self, observed_ms):
        # Use the same normal cancellation route and persistent failure state.
        return self._block(observed_ms, 'CHANNEL_TIMEOUT')

    # Snapshots without an application watermark cannot restore reference continuity.
    def on_unverified_snapshot(self, observed_ms):
        # Leave all book, order and position state untouched.
        return False
