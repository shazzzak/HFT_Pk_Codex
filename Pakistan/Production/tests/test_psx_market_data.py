"""PSX v1.05 sequencing and snapshot safety without a network session."""
# Cover defined retransmission failure statuses.
import pytest
# Exercise venue-specific application contracts.
from venues.psx_market_data import PSXTickRecovery, TickRecord, PSXSnapshotState, SnapshotEntry


# Build one isolated channel with observable callbacks.
def stream(**kwargs):
    # Capture delivered sequences, quote withdrawals, requests and reconnects.
    delivered, unsafe, requests, reconnects = [], [], [], []
    # A new epoch starts at sequence zero and cannot inherit readiness.
    handler = PSXTickRecovery(2011, "E1", delivered.append, lambda *a: unsafe.append(a), lambda *a: requests.append(a), lambda *a: reconnects.append(a), **kwargs)
    # Expose the effects for assertions.
    return handler, delivered, unsafe, requests, reconnects


# Preserve the sequence and payload of a decoded tick message.
def record(sequence):
    # Fields remain ordered pairs, not a dictionary that could lose repetitions.
    return TickRecord("UA201", sequence, ((1181, str(sequence)), (55, "OGDC")))


# Live traffic may arrive beyond a hole while replay fills the missing interval.
def test_gap_recovery_buffers_and_delivers_contiguously():
    # Create one new stream.
    h, delivered, unsafe, requests, _ = stream()
    # Establish the first contiguous record.
    assert h.tick("E1", record(1), 100)
    # Sequence three exposes missing sequence two.
    assert h.tick("E1", record(3), 101)
    # Do not deliver three before two.
    assert [r.sequence for r in delivered] == [1]
    # Quoting is blocked during recovery.
    assert not h.ready
    # Encode an application UA002 request for the exact missing interval.
    assert dict(requests[0][1]) == {35: "UA002", 10077: "1", 10201: "2011", 1182: "2", 1183: "2"}
    # New live traffic remains buffered while recovery proceeds.
    assert h.tick("E1", record(4), 102)
    # Deliver the retransmitted missing record on the separate session.
    assert h.tick("E1", record(2), 103, retransmitted=True)
    # Drain all now-contiguous records exactly once.
    assert [r.sequence for r in delivered] == [1, 2, 3, 4]
    # The final acknowledgement is still required.
    assert not h.ready
    # Successful completion now closes this request.
    assert h.retransmission_ack("E1", 1, 104)
    # Recovery is complete without skipping a sequence.
    assert h.ready
    # A replay duplicate must not apply another order update.
    assert not h.tick("E1", record(2), 105, retransmitted=True)
    # Exactly four original records were delivered.
    assert len(delivered) == 4


# Idle channel heartbeats identify missing tail records without consuming a sequence.
def test_heartbeat_watermark_and_stale_live_connection():
    # Create a fresh application stream.
    h, delivered, _, requests, reconnects = stream()
    # Learn that two ticks exist although neither reached the live socket.
    assert h.heartbeat("E1", 2, 100)
    # Heartbeat itself is not tick sequence one.
    assert h.applied == 0
    # Request both missing records.
    assert dict(requests[0][1])[1183] == "2"
    # Recover the application stream using the retransmission connection.
    h.tick("E1", record(1), 101, retransmitted=True)
    # Complete the missing interval.
    h.tick("E1", record(2), 102, retransmitted=True)
    # Acknowledge completion.
    h.retransmission_ack("E1", 1, 103)
    # At exactly two heartbeat intervals the specification says not yet beyond it.
    h.poll(6100)
    # The stream has not exceeded the six-second boundary.
    assert not reconnects
    # One millisecond later the real-time connection is stale.
    h.poll(6101)
    # Recovery traffic did not conceal live-channel silence.
    assert reconnects and not h.ready


# Partial completion is not successful completion, and error statuses never fast-forward.
@pytest.mark.parametrize("status", [3, 4, 99])
def test_retransmission_failures_remain_disarmed(status):
    # Start a gap request.
    h, _, _, _, reconnects = stream()
    # Discover a missing initial record.
    h.tick("E1", record(2), 100)
    # Partial status keeps the request outstanding.
    assert h.retransmission_ack("E1", 2, 101)
    # No partially completed stream becomes ready.
    assert not h.ready and h.outstanding is not None
    # A final non-success status is fatal for this epoch.
    assert not h.retransmission_ack("E1", status, 102)
    # Reconstruction is explicitly requested.
    assert reconnects and h.failed


# An acknowledgement is not evidence that the missing records were delivered.
def test_false_completion_and_recovery_timeout():
    # Start a gap and receive an inconsistent success ack.
    h, _, _, _, reconnects = stream()
    # Sequence one has not arrived.
    h.tick("E1", record(2), 100)
    # The server cannot complete an undelivered range.
    assert not h.retransmission_ack("E1", 1, 101)
    # Do not silently skip the missing record.
    assert h.applied == 0 and reconnects
    # Independently test a busy live connection with stalled recovery.
    h, _, _, _, reconnects = stream(recovery_timeout_ms=100)
    # Start another missing interval.
    h.tick("E1", record(2), 100)
    # A live heartbeat proves transport liveness only.
    h.heartbeat("E1", 2, 150)
    # The recovery deadline still expires.
    h.poll(200)
    # Reconnect rather than accumulate data forever.
    assert reconnects and h.failed


# Memory exhaustion is an explicit failure, never an arbitrary dropped tick.
def test_buffer_limit_epoch_and_end_of_channel():
    # Allow one queued out-of-order record.
    h, _, _, _, reconnects = stream(max_buffer=1)
    # Old epochs cannot contaminate recovery.
    assert not h.tick("OLD", record(1), 100)
    # Buffer one record beyond a hole.
    h.tick("E1", record(2), 100)
    # A second out-of-order record exceeds the declared limit.
    assert not h.tick("E1", record(3), 101)
    # Explicit reconstruction is required.
    assert reconnects and h.failed
    # Test terminal channel status independently.
    h, _, unsafe, _, _ = stream()
    # Establish a quiet but live empty stream.
    h.heartbeat("E1", 0, 100)
    # Apply EndOfChannel from UA001.
    h.heartbeat("E1", 0, 101, end=True)
    # Terminal state cannot be rearmed by later data.
    assert h.ended and not h.ready and unsafe[-1][1] == "end of tick channel"


# A full gap buffer must still admit the missing record that drains it.
def test_missing_record_can_drain_full_buffer():
    # Bound out-of-order storage at a single record.
    h, delivered, _, _, _ = stream(max_buffer=1)
    # Fill that one slot beyond a missing first record.
    h.tick("E1", record(2), 100)
    # Admit the missing head even though the waiting buffer is full.
    assert h.tick("E1", record(1), 101, retransmitted=True)
    # Both records are now consumed in order and storage is released.
    assert [r.sequence for r in delivered] == [1, 2] and not h.buffer
    # The normal completion acknowledgement closes recovery.
    h.retransmission_ack("E1", 1, 102)
    # The stream is ready without reconnecting.
    assert h.ready and not h.failed


# Startup silence and late arrivals must not bypass a missed timer deadline.
def test_startup_silence_and_late_heartbeat_require_reconnect():
    # Start without any live data.
    h, _, _, _, reconnects = stream()
    # Establish a monotonic timer origin.
    h.poll(100)
    # A silent newly connected socket also requires recovery after two intervals.
    h.poll(6101)
    # No indefinitely silent startup is treated as healthy.
    assert reconnects and h.failed
    # Check a formerly healthy channel independently.
    h, _, _, _, reconnects = stream()
    # Establish liveness.
    h.heartbeat("E1", 0, 100)
    # An overdue heartbeat cannot resurrect the old epoch without reconnecting.
    assert not h.heartbeat("E1", 0, 6101)
    # The timer deadline is enforced even when poll was delayed.
    assert reconnects and not h.ready


# A minimal complete REG W message with exact decimal prices.
def entries(upper="110.0000", lower="90.0000"):
    # Distinguish depth from published metadata.
    return (SnapshotEntry("0", "99.0000", "10.00", 1), SnapshotEntry("1", "101.0000", "20.00", 1), SnapshotEntry("xe", upper), SnapshotEntry("xf", lower))


# Books require independently fresh market phase, security phase and published bands.
def test_snapshot_phase_bands_and_staleness():
    # Capture cancellations for a two-name book.
    unsafe = []
    # The snapshot freshness policy is explicit, not guessed from heartbeat timing.
    h = PSXSnapshotState(("A", "B"), "E1", 5000, lambda *a: unsafe.append(a))
    # A valid snapshot alone does not prove the market-wide phase is known.
    assert h.snapshot("E1", 1011, "010", "A", "T0", 100, entries(), 1000)
    # No h message yet means no quote permission.
    assert h.view("A", 1000) is None
    # A fresh REG continuous phase allows use of the snapshot.
    assert h.trading_status("E1", "01", "T0", 100, 1000)
    # Verify exact conversion to paisa.
    assert h.view("A", 1000)[0].best_bid.price_minor == 9900
    # Verify published bounds rather than a calculation from prior close.
    assert h.view("A", 1000)[1].lower_minor == 9000
    # A global halt immediately withdraws the symbol.
    h.trading_status("E1", "01", "H0", 101, 1100)
    # Fresh book depth cannot override the halt.
    assert h.view("A", 1100) is None
    # A later continuous market phase resumes this fresh book's eligibility.
    h.trading_status("E1", "01", "T0", 102, 1200)
    # Timer-driven snapshot expiry withdraws it even without new market messages.
    h.poll(6000)
    # Snapshot age reaches the configured five-second threshold.
    assert h.view("A", 6000) is None


# Market mixing, whole-day suspension and absent bounds cannot produce tradeable books.
def test_snapshot_rejects_wrong_market_and_incomplete_reference_data():
    # Set up a single subscribed REG instrument.
    h = PSXSnapshotState(("A",), "E1", 5000, lambda *a: None)
    # Establish the fresh global market phase.
    h.trading_status("E1", "01", "T0", 100, 1000)
    # An odd-lot snapshot must not populate the REG book.
    assert not h.snapshot("E1", 3021, "080", "A", "T0", 100, entries(), 1000)
    # A whole-day suspension remains nontradeable even during continuous market phase.
    assert h.snapshot("E1", 1011, "010", "A", "T1", 100, entries(), 1000)
    # Security suspension wins over global status.
    assert h.view("A", 1000) is None
    # A status-only W cannot refresh the old depth or price limits.
    assert not h.snapshot("E1", 1011, "010", "A", "T0", 101, (), 1100)
    # No old book survives as tradeable state.
    assert h.view("A", 1100) is None


# Sentinel bounds are explicitly unbounded, not huge or negative executable prices.
def test_snapshot_sentinels_and_fractional_units():
    # Create a fresh ready-market snapshot consumer.
    h = PSXSnapshotState(("A",), "E1", 5000, lambda *a: None)
    # Establish continuous market phase.
    h.trading_status("E1", "01", "T0", 100, 1000)
    # Use the two documented no-limit representations.
    assert h.snapshot("E1", 1011, "010", "A", "T0", 100, entries("999999999.9999", "0.01"), 1000)
    # Both endpoints are explicitly unbounded.
    assert h.view("A", 1000)[1].upper_minor is None
    # Missing bounds would have failed; this None has published provenance.
    assert h.view("A", 1000)[1].lower_minor is None
    # An off-paisa depth price must not be silently rounded.
    broken = (SnapshotEntry("0", "99.0001", "10.00", 1),) + entries()[1:]
    # Reject the malformed full snapshot.
    assert not h.snapshot("E1", 1011, "010", "A", "T0", 101, broken, 1100)
    # The old usable snapshot is invalidated.
    assert h.view("A", 1100) is None
