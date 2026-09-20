"""PSX v1.05 application-layer tick recovery, independent of FIX sessions.

Supply decoded complete messages from a verified session/codec. Heartbeats
do not consume ApplSeqNum. Snapshots are a different, non-retransmitted stream;
this module never invents a snapshot/tick sequence bridge. Each instance owns
one channel and one feed session epoch, and runs on one serialized event loop.
The request callback returns UA002 business fields for the retransmit session.
It does not open sockets, authenticate, or send orders.
"""
# Preserve decoded messages as immutable values.
from dataclasses import dataclass
# Convert published decimal prices without binary-float rounding.
from decimal import Decimal
# Expose only validated immutable strategy books.
from core.model import BookLevel, BookSnapshot
# Keep published bounds distinct from absent reference data.
from core.venue import PriceBand
# Reuse the established PSX phase decoder.
from venues.psx import PSXVenue


# Application tick records preserve the channel sequence and original fields.
@dataclass(frozen=True)
class TickRecord:
    # UA201 or UA202, or a future application type handled by the consumer.
    kind: str
    # Channel-wide ApplSeqNum, tag 1181.
    sequence: int
    # Ordered tag/value pairs; do not collapse repeating fields into a dict.
    fields: tuple


# Recover ticks without treating FIX session sequences as application sequences.
class PSXTickRecovery:
    # Callbacks are required because losing data must have an immediate effect.
    def __init__(self, channel, epoch, deliver, on_unsafe, request_retransmit, reconnect, max_buffer=10000, recovery_timeout_ms=10000):
        # Reject ambiguous channel identities and unbounded buffering policies.
        if type(channel) is not int or channel <= 0 or not epoch or max_buffer < 1 or recovery_timeout_ms <= 0:
            # Require explicit usable configuration before accepting input.
            raise ValueError("invalid channel, epoch or recovery limits")
        # A new instance starts from the beginning of the channel's day/session.
        self.channel, self.epoch = channel, epoch
        # Deliver only contiguous ordered application records.
        self.deliver = deliver
        # The application withdraws desired quotes via ordinary OMS reconciliation.
        self.on_unsafe = on_unsafe
        # The connection layer sends these business fields on the retransmit session.
        self.request_retransmit = request_retransmit
        # The connection layer owns reconnect authentication and session reset.
        self.reconnect = reconnect
        # Bound queued out-of-order records independently of feed rate.
        self.max_buffer = max_buffer
        # Bound unsuccessful recovery independently of heartbeat liveness.
        self.recovery_timeout_ms = recovery_timeout_ms
        # The last contiguous record delivered to the consumer.
        self.applied = 0
        # Largest application sequence observed in data or a channel heartbeat.
        self.high = 0
        # Buffered records beyond a hole.
        self.buffer = {}
        # At most one request is outstanding because the server processes serially.
        self.outstanding = None
        # Monotonic receipt time of live data or channel heartbeats.
        self.last_live_ms = None
        # Track local monotonic time across all callbacks.
        self.clock_ms = None
        # Bound startup silence even before the first live message arrives.
        self.first_clock_ms = None
        # No heartbeat or tick has yet established liveness.
        self.ready = False
        # A fatal recovery failure requires an explicit new instance/epoch.
        self.failed = False
        # End-of-channel is terminal for quoting in this epoch.
        self.ended = False
        # Initial startup must not leave any previous desired quotes active.
        self.on_unsafe(channel, "awaiting contiguous tick stream")

    # Fail closed before asking the connection layer to reconnect.
    def _fail(self, reason):
        # Idempotency prevents reconnect storms from repeated bad traffic.
        if self.failed:
            # Preserve the first failure.
            return
        # Block quote permission before invoking callbacks.
        self.failed, self.ready = True, False
        # Withdraw all instruments depending on this channel.
        self.on_unsafe(self.channel, reason)
        # Recovery needs a new epoch and reconstruction from a verified checkpoint.
        self.reconnect(self.channel, reason)

    # Accept only timestamps from a local monotonic clock.
    def _clock(self, now_ms):
        # Reject bad units/types and clock rollback.
        if type(now_ms) is not int or now_ms < 0 or (self.clock_ms is not None and now_ms < self.clock_ms):
            # No timer can be trusted after rollback.
            self._fail("invalid monotonic clock")
            # Reject the associated input.
            return False
        # Establish the start of monitoring before any live traffic is observed.
        if self.first_clock_ms is None:
            # Do not assume a particular monotonic clock origin.
            self.first_clock_ms = now_ms
        # Check silence before an arriving message can revive an expired stream.
        since = self.last_live_ms if self.last_live_ms is not None else self.first_clock_ms
        # Two heartbeat intervals are six seconds in the supplied specification.
        if now_ms - since > 6000:
            # Require explicit reconnection instead of silently resuming stale state.
            self._fail("real-time tick channel silent beyond two heartbeat intervals")
        # Advance the known local clock.
        self.clock_ms = now_ms
        # Fatal or completed streams must not resume implicitly.
        return not self.failed and not self.ended

    # Ask for the first missing contiguous interval only.
    def _recover(self, now_ms):
        # Serialized recovery permits only one request at a time.
        if self.outstanding is not None or self.failed or self.applied >= self.high:
            # No additional request is currently needed or permitted.
            return
        # Withdraw before any retransmission is requested.
        self.ready = False
        # The first absent sequence immediately follows the contiguous watermark.
        begin = self.applied + 1
        # Stop before the first already buffered record, otherwise at known high.
        end = min(self.buffer) - 1 if self.buffer else self.high
        # Remember this exact requested interval and its deadline origin.
        self.outstanding = (begin, end, now_ms)
        # Gap recovery must not continue quoting on stale strategy state.
        self.on_unsafe(self.channel, f"tick sequence gap {begin}..{end}")
        # These are application-layer fields, not a FIX session ResendRequest.
        self.request_retransmit(self.epoch, ((35, "UA002"), (10077, "1"), (10201, str(self.channel)), (1182, str(begin)), (1183, str(end))))

    # Apply buffered records in channel order once each hole is repaired.
    def _drain(self):
        # Never skip a missing sequence even if newer records are available.
        while self.applied + 1 in self.buffer and not self.failed:
            # Peek until downstream processing succeeds.
            record = self.buffer[self.applied + 1]
            # Consumer errors must invalidate this whole reconstructed stream.
            try:
                # The consumer receives each application sequence only once.
                self.deliver(record)
            # Never retry an ambiguously applied record after a consumer error.
            except Exception:
                # A new epoch must reconstruct consumer state explicitly.
                self._fail("tick consumer failed; reconstruction required")
                # Retain the original exception for the application's audit path.
                raise
            # Commit the contiguous watermark after successful delivery.
            self.applied = record.sequence
            # Release the now-applied buffered record.
            del self.buffer[record.sequence]

    # Receive a decoded application tick from either live or retransmit session.
    def tick(self, epoch, record, now_ms, retransmitted=False):
        # Traffic from old sessions cannot affect this epoch's recovery.
        if epoch != self.epoch or not self._clock(now_ms):
            # Ignore stale epochs and failed streams.
            return False
        # Require a proper positive application sequence.
        if type(record.sequence) is not int or record.sequence < 1:
            # A malformed sequence cannot enter the buffer.
            self._fail("invalid application sequence")
            # Report rejection to the decoder.
            return False
        # Ignore sequences already delivered, including retransmitted duplicates.
        if record.sequence <= self.applied:
            # Duplicates do not establish fresh live data.
            return False
        # Do not let replay traffic conceal a dead real-time connection.
        if not retransmitted and record.sequence not in self.buffer:
            # Record genuine new live data arrival.
            self.last_live_ms = now_ms
        # Conflicting buffered records make recovery ambiguous.
        if record.sequence in self.buffer and self.buffer[record.sequence] != record:
            # Refuse to choose one silently.
            self._fail("conflicting buffered application sequence")
            # No version is delivered out of order.
            return False
        # Refuse unbounded accumulation while missing data cannot be recovered.
        if record.sequence not in self.buffer and record.sequence != self.applied + 1 and len(self.buffer) >= self.max_buffer:
            # Restart and rebuild rather than discard part of the order book.
            self._fail("tick recovery buffer exhausted")
            # The rejected record was not applied.
            return False
        # Keep the largest observed sequence even while recovery is pending.
        self.high = max(self.high, record.sequence)
        # Buffer the complete record until preceding sequences are delivered.
        self.buffer[record.sequence] = record
        # Fill any newly completed contiguous run.
        self._drain()
        # Request a missing interval when no earlier request is outstanding.
        self._recover(now_ms)
        # Recovery is not complete until both all data and the completion ack arrive.
        self.ready = not self.failed and self.outstanding is None and self.applied == self.high and self.last_live_ms is not None
        # The input was accepted, possibly into a bounded recovery buffer.
        return True

    # UA001 has no ApplSeqNum and must not advance the data watermark by itself.
    def heartbeat(self, epoch, last_sequence, now_ms, end=False):
        # Require the active feed epoch and a valid local clock.
        if epoch != self.epoch or not self._clock(now_ms):
            # Ignore obsolete or unusable messages.
            return False
        # ApplLastSeqNum may be zero before any ticks exist.
        if type(last_sequence) is not int or last_sequence < 0:
            # Malformed heartbeat metadata cannot prove liveness.
            self._fail("invalid heartbeat application watermark")
            # Refuse the heartbeat.
            return False
        # A live heartbeat proves transport liveness but not data completeness.
        self.last_live_ms = now_ms
        # Learn about missed tail records even when no newer tick has arrived.
        self.high = max(self.high, last_sequence)
        # End-of-channel stops quoting even if recovery is otherwise complete.
        if end:
            # Preserve the terminal state for this epoch.
            self.ended, self.ready = True, False
            # Pull quotes through the normal application callback.
            self.on_unsafe(self.channel, "end of tick channel")
            # Terminal status was accepted.
            return True
        # Recover records exposed only by the heartbeat's last-sequence field.
        self._recover(now_ms)
        # Never claim readiness while a gap request awaits its final ack.
        self.ready = not self.failed and self.outstanding is None and self.applied == self.high
        # The heartbeat was processed without consuming a tick sequence.
        return True

    # Match UA002 acknowledgements to the one serialized outstanding request.
    def retransmission_ack(self, epoch, status, now_ms):
        # A new connection epoch invalidates old acknowledgements.
        if epoch != self.epoch or not self._clock(now_ms):
            # Do not let a stale response complete a new request.
            return False
        # An unsolicited ack cannot safely advance recovery state.
        if self.outstanding is None:
            # Surface this protocol mismatch instead of guessing a correlation.
            self._fail("unsolicited retransmission acknowledgement")
            # No request was completed.
            return False
        # Status 2 explicitly says part of the requested data has not returned yet.
        if status == 2:
            # Keep the original deadline; partial replies cannot extend it forever.
            return True
        # Only status 1 denotes successful completion.
        if status != 1:
            # No-authority, unavailable-data and unknown statuses all fail closed.
            self._fail(f"retransmission failed with status {status}")
            # The gap remains unresolved.
            return False
        # A successful ack without the requested records is inconsistent.
        if self.applied < self.outstanding[1]:
            # Never convert that inconsistency into a skipped sequence.
            self._fail("retransmission completed with missing records")
            # Keep the engine disarmed.
            return False
        # Release the completed request so a later hole may be requested.
        self.outstanding = None
        # New live data may have exposed another gap during recovery.
        self._recover(now_ms)
        # Readiness requires both contiguous data and an alive real-time stream.
        self.ready = not self.failed and self.outstanding is None and self.applied == self.high and self.last_live_ms is not None
        # The completion acknowledgement was accepted.
        return True

    # Run even when neither connection delivers messages.
    def poll(self, now_ms):
        # Clock failure immediately disarms the application.
        if not self._clock(now_ms):
            # No additional checks can restore readiness.
            return
        # PSX specifies three-second channel heartbeats and failure after two intervals.
        if self.last_live_ms is not None and now_ms - self.last_live_ms > 6000:
            # Disconnect and reconnect as specified, while keeping quotes withdrawn.
            self._fail("real-time tick channel silent beyond two heartbeat intervals")
        # A busy live stream must not conceal an unrecoverable historical gap.
        if self.outstanding is not None and now_ms - self.outstanding[2] >= self.recovery_timeout_ms:
            # Recovery timeout requires reconstruction, never silent fast-forward.
            self._fail("application retransmission timed out")


# A decoder must preserve each complete NoMDEntries repeating group.
@dataclass(frozen=True)
class SnapshotEntry:
    # Tag 269, including 0/1 depth and xe/xf published limits.
    kind: str
    # Tag 270 as its exact decimal text; never pre-round with float.
    price: str | None = None
    # Tag 271 as its exact decimal text.
    quantity: str | None = None
    # Tag 1023 for depth levels, when present.
    level: int | None = None


# Maintain the independent periodic REG snapshot and phase streams.
class PSXSnapshotState:
    """Published snapshot books only; deliberately does not splice in tick data.

The supplied specification contains no shared snapshot/tick cut identifier.
An application must not call this a reconciled incremental book until a
verified bridge or independently rebuilt tick book is provided.
"""
    # Snapshot age is a declared house limit because the W interval is unspecified.
    def __init__(self, symbols, epoch, snapshot_stale_ms, on_unsafe, status_stale_ms=6000):
        # Require explicit scope, epoch and positive freshness policy.
        if not symbols or not epoch or snapshot_stale_ms <= 0 or status_stale_ms <= 0:
            # Fail before a partially configured feed can publish books.
            raise ValueError("symbols, epoch and snapshot freshness are required")
        # Keep subscription identity immutable for this feed epoch.
        self.symbols, self.epoch = frozenset(symbols), epoch
        # Timer policy is explicit and inspectable.
        self.snapshot_stale_ms = snapshot_stale_ms
        # This is a house status-age limit, not a claimed PSX disconnect rule.
        self.status_stale_ms = status_stale_ms
        # Withdraw affected symbols through the application's normal OMS path.
        self.on_unsafe = on_unsafe
        # The existing phase parser does not need a calendar for decoding.
        self.venue = PSXVenue(session_provider=lambda _: ())
        # No market-wide phase is assumed on startup.
        self.session_phase = None
        # Receipt timestamp of the latest REG h message.
        self.session_rx = None
        # Exchange timestamp of the latest REG h message.
        self.session_exchange_ms = None
        # Store only complete validated symbol snapshots.
        self.books = {}
        # Symbols remain disarmed until all required state is fresh.
        self.unsafe = set(self.symbols)
        # Track local monotonic time across both independent streams.
        self.clock_ms = None
        # Clock corruption requires explicit epoch replacement.
        self.failed = False
        # Startup itself is a quote-withdrawal condition.
        for symbol in self.symbols:
            # Never inherit a previous session's quote permission.
            self.on_unsafe(symbol, "awaiting published book, phase and bands")

    # Refuse future use of state whose local freshness clock is inconsistent.
    def _clock(self, now_ms):
        # A prior clock failure cannot be repaired by another event in this epoch.
        if self.failed:
            # Avoid repeated withdrawal callbacks after the first failure.
            return False
        # Validate integer monotonic milliseconds.
        if type(now_ms) is not int or now_ms < 0 or (self.clock_ms is not None and now_ms < self.clock_ms):
            # A new epoch is required after a clock failure.
            self.failed = True
            # Withdraw every dependent symbol.
            for symbol in self.symbols:
                # Preserve an explicit reason in the application audit stream.
                self.on_unsafe(symbol, "invalid snapshot freshness clock")
            # Refuse this update.
            return False
        # Advance the known local clock.
        self.clock_ms = now_ms
        # A fresh timestamp does not undo a previous clock failure.
        return not self.failed

    # Convert a decimal market value to exact integer units.
    @staticmethod
    def _integer(text, scale):
        # Decimal preserves the published precision and explicit sentinel values.
        value = Decimal(text) * scale
        # Reject missing, non-finite, fractional-unit and negative values.
        if not value.is_finite() or value != value.to_integral_value() or value < 0:
            # Never silently round a price or share quantity.
            raise ValueError("market value is not representable in integer units")
        # Return the exact integer domain value.
        return int(value)

    # Apply a decoded h message for the Regular Market only.
    def trading_status(self, epoch, market_code, phase_code, exchange_ms, now_ms):
        # Other markets and previous epochs cannot change REG quote permission.
        if epoch != self.epoch or market_code != "01" or not self._clock(now_ms):
            # Ignore irrelevant messages without refreshing status liveness.
            return False
        # Older status cannot overwrite a more recent exchange phase.
        if self.session_exchange_ms is not None and exchange_ms < self.session_exchange_ms:
            # Do not refresh freshness on an obsolete status.
            return False
        # Retain the exchange-published phase and both distinct clocks.
        self.session_phase, self.session_exchange_ms, self.session_rx = self.venue.parse_phase(phase_code), exchange_ms, now_ms
        # A halt withdraws quotes immediately, without waiting for W messages.
        self.poll(now_ms)
        # The REG phase was accepted.
        return True

    # Decode one complete W message after a verified FIX repeating-group parser.
    def snapshot(self, epoch, channel, stream, symbol, phase_code, exchange_ms, entries, now_ms):
        # Channel 1011 and stream 010 identify Regular Market share snapshots.
        if epoch != self.epoch or channel != 1011 or stream != "010" or symbol not in self.symbols or not self._clock(now_ms):
            # Never mix odd-lot, futures or square-up books into REG.
            return False
        # Retain the previous timestamp to reject obsolete snapshot generations.
        old = self.books.get(symbol)
        # Equal timestamps are allowed: OrigTime can be only second precision.
        if old is not None and exchange_ms < old[0].timestamp_ms:
            # A delayed older snapshot cannot refresh the book.
            return False
        # Build a complete new message off to the side.
        bids, asks, bounds, seen_levels = [], [], {}, set()
        # Malformed snapshots withdraw the symbol instead of retaining quote permission.
        try:
            # Preserve repeated entries rather than reducing tags to one dictionary.
            for entry in entries:
                # Only actual book rows enter the strategy's depth.
                if entry.kind in ("0", "1"):
                    # Require a valid, unique level rank for this side.
                    if type(entry.level) is not int or not 1 <= entry.level <= 10 or (entry.kind, entry.level) in seen_levels:
                        # Reject ambiguous snapshot ladders.
                        raise ValueError("invalid or duplicate snapshot depth level")
                    # Remember this declared rank.
                    seen_levels.add((entry.kind, entry.level))
                    # Convert price and shares exactly.
                    price, quantity = self._integer(entry.price, 100), self._integer(entry.quantity, 1)
                    # Zero or negative levels do not describe a usable resting order.
                    if price <= 0 or quantity <= 0:
                        # Treat malformed depth as a feed failure.
                        raise ValueError("nonpositive snapshot depth")
                    # Preserve the published level order for later validation.
                    (bids if entry.kind == "0" else asks).append((entry.level, BookLevel(price, quantity)))
                # Published circuit bounds are metadata, not liquidity levels.
                elif entry.kind in ("xe", "xf"):
                    # Multiple published bounds in one message are ambiguous.
                    if entry.kind in bounds:
                        # Refuse the entire symbol snapshot.
                        raise ValueError("duplicate published band")
                    # Interpret only documented no-limit sentinel values.
                    raw = Decimal(entry.price)
                    # Pages 18 and 20 give two lower no-limit representations.
                    unlimited = raw == Decimal("999999999.9999") if entry.kind == "xe" else raw in (Decimal("-999999999.9999"), Decimal("0.01"))
                    # Keep explicit unlimited distinct from absent tags.
                    bounds[entry.kind] = None if unlimited else self._integer(entry.price, 100)
            # Require both published bound entries in each usable full snapshot.
            if set(bounds) != {"xe", "xf"}:
                # A status-only or incomplete W must not refresh an old ladder.
                raise ValueError("full published bounds absent")
            # Finite limits must be positive; zero is not a documented sentinel.
            if any(value is not None and value <= 0 for value in bounds.values()):
                # Refuse malformed published reference data.
                raise ValueError("nonpositive published price band")
            # Reject inverted finite limits.
            if bounds["xf"] is not None and bounds["xe"] is not None and bounds["xf"] > bounds["xe"]:
                # Invalid reference data must fail closed.
                raise ValueError("inverted published price band")
            # Require contiguous ranks so a missing level cannot masquerade as full depth.
            for side in (bids, asks):
                # The disclosed ladder starts at one on each present side.
                if sorted(rank for rank, _ in side) != list(range(1, len(side) + 1)):
                    # Stop quoting on incomplete ranked depth.
                    raise ValueError("noncontiguous snapshot depth ranks")
            # BookSnapshot validates monotonic prices and crossing independently.
            book = BookSnapshot(symbol, exchange_ms, tuple(v for _, v in sorted(bids)), tuple(v for _, v in sorted(asks)))
            # Decode per-security suspension independently of the market h stream.
            phase = self.venue.parse_phase(phase_code)
            # Preserve explicitly unbounded published limits.
            band = PriceBand(upper_minor=bounds["xe"], lower_minor=bounds["xf"])
        # Decimal and type errors also make the whole snapshot unusable.
        except Exception:
            # Remove old state so it cannot be revived by a later status heartbeat.
            self.books.pop(symbol, None)
            # Record the unsafe state immediately.
            self.unsafe.add(symbol)
            # The application cancels the symbol through its normal diff path.
            self.on_unsafe(symbol, "invalid or incomplete REG snapshot")
            # A later complete periodic snapshot may recover this symbol.
            return False
        # Commit book, phase, bands and receipt time together.
        self.books[symbol] = (book, phase, band, now_ms)
        # Re-evaluate current quote permission using both independent streams.
        self.poll(now_ms)
        # A coherent snapshot was accepted; it may still be non-tradeable.
        return True

    # Return a snapshot only when both the market and security state are current.
    def view(self, symbol, now_ms):
        # A clock failure invalidates every book in this epoch.
        if not self._clock(now_ms):
            # No old state remains safe to use.
            return None
        # Resolve the latest complete symbol snapshot.
        item = self.books.get(symbol)
        # Require a current REG h stream as well as current symbol depth.
        if item is None or self.session_phase is None or not self.session_phase.is_tradeable or self.session_rx is None or now_ms - self.session_rx > self.status_stale_ms:
            # Book freshness alone cannot override a stale or halted market phase.
            return None
        # Unpack the coherent generation.
        book, phase, band, received = item
        # One-sided books and suspended securities cannot drive this market maker.
        if not phase.is_tradeable or now_ms - received >= self.snapshot_stale_ms or not book.bids or not book.asks:
            # Leave quote permission off until fresh usable depth arrives.
            return None
        # Return both depth and the corresponding published bounds.
        return book, band

    # Timer-driven staleness cannot depend on receipt of another market event.
    def poll(self, now_ms):
        # Check every subscribed symbol against both stream clocks.
        for symbol in self.symbols:
            # A usable fresh view ends the local unsafe transition.
            if self.view(symbol, now_ms) is not None:
                # Operator arming and tick recovery remain separate gates.
                self.unsafe.discard(symbol)
            # Emit one cancellation callback per loss-of-validity transition.
            elif symbol not in self.unsafe:
                # Remember that withdrawal has already been requested.
                self.unsafe.add(symbol)
                # Use ordinary OMS desired-flat reconciliation.
                self.on_unsafe(symbol, "stale, halted or unusable snapshot state")
