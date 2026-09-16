"""Tests for the audit log.

Two things are being asserted here and they pull against each other:

  * the log is COMPLETE and DURABLE for the records that matter
  * writing it costs the trading path almost nothing

The performance test is not decoration. The first version of this file used
`queue.Queue`, which notifies a condition variable on every put and cost the
producer 17 MICROSECONDS per record. That is the kind of regression nobody
notices by reading a diff, so it is pinned by an assertion instead.
"""
# the test framework
import json
import tempfile
import time
from pathlib import Path
import pytest
# the domain
from core.model import CancelOrder, PlaceOrder, ReplaceOrder, Side
# the risk layer
from core.risk import OrderQuantityCheck, RiskDecision, RiskGateway
# the log under test
from core.audit import AuditLog, CRITICAL_EVENTS

# pyarrow is required by the engine, but this suite should say so clearly
# rather than erroring in a way that looks like a bug in the log
try:
    import pyarrow.parquet as pq
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False


def _need_parquet():
    """Skip rather than fail when the engine's Parquet dependency is absent.

    The engine REFUSES TO START without pyarrow -- the audit log is not
    optional. That is right in production and unhelpful in a test run, where it
    turns one missing dependency into a page of identical errors.
    """
    # nothing to do when it is installed
    if not HAVE_PARQUET:
        pytest.skip("pyarrow not installed; the engine requires it")


def make(d, **kw):
    """A log in a temporary directory, with test-sized batching."""
    # small row groups and a short flush so tests do not wait
    kw.setdefault("row_group_rows", 100)
    kw.setdefault("flush_interval_s", 0.1)
    # the log
    return AuditLog(Path(d), session_id="T", **kw)


# --- the crash-survivable file ---------------------------------------------

def test_critical_records_are_on_disk_before_record_returns():
    """The crash is the event the log exists to explain.

    A compressed record of a normal morning is of limited interest. The records
    immediately before a crash are the whole point, so they are fsynced on the
    hot path rather than queued.
    """
    # the engine will not construct without pyarrow
    _need_parquet()
    # a log
    with tempfile.TemporaryDirectory() as d:
        log = make(d)
        # a critical event
        log.record("kill_switch_tripped", {"reason": "operator"})
        # WITHOUT closing, or waiting, or flushing anything: it is already there
        rows = [json.loads(l)
                for l in log.critical_path.read_text().splitlines()]
        # the trip is on disk
        assert any(r["event"] == "kill_switch_tripped" for r in rows)
        # tidy up
        log.close()


def test_only_critical_events_reach_the_sidecar():
    """The sidecar is the emergency file, not a second copy of everything."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # a log
    with tempfile.TemporaryDirectory() as d:
        log = make(d)
        # one of each
        log.record("risk_approved", {"symbol": "OGDC"})
        log.record("risk_rejected", {"symbol": "OGDC"})
        # read the sidecar
        rows = [json.loads(l)
                for l in log.critical_path.read_text().splitlines()]
        events = {r["event"] for r in rows}
        # the rejection is there; the approval is not
        assert "risk_rejected" in events
        assert "risk_approved" not in events
        # and every event in the file is one we declared critical
        assert events <= CRITICAL_EVENTS
        log.close()


def test_a_dropped_record_still_leaves_a_visible_gap():
    """A log that loses records silently is worse than no log."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # a log whose buffer holds almost nothing, so the writer cannot keep up
    with tempfile.TemporaryDirectory() as d:
        log = make(d, queue_size=1)
        # flood it
        for i in range(5_000):
            log.record("tick", {"i": i})
        # some were dropped
        assert log.dropped > 0
        # THE SEQUENCE NUMBER IS ASSIGNED BEFORE THE DROP, so the numbers that
        # did survive are non-contiguous and the loss is visible in the file
        # rather than being a file that is merely short.
        log.record("kill_switch_tripped", {"n": "final"})
        rows = [json.loads(l)
                for l in log.critical_path.read_text().splitlines()]
        # the last record's sequence number counts everything, dropped included
        assert rows[-1]["seq"] > 5_000
        log.close()


# --- the Parquet file ------------------------------------------------------

def test_everything_reaches_the_parquet_file():
    """The sidecar is a subset; the Parquet file is the whole record."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # a log
    with tempfile.TemporaryDirectory() as d:
        log = make(d)
        # a mix of critical and ordinary events
        for i in range(250):
            log.record("risk_approved", {"symbol": "OGDC", "quantity": i})
        log.record("risk_rejected", {"symbol": "OGDC", "reason": "too big"})
        # close flushes the tail and joins the writer
        log.close()
        # read it back
        tbl = pq.read_table(log.path)
        events = tbl.column("event").to_pylist()
        # both kinds are present
        assert events.count("risk_approved") == 250
        assert events.count("risk_rejected") == 1
        # the session bookends are there too
        assert events[0] == "session_start" and events[-1] == "session_end"
        # sequence numbers are contiguous from 1, so a gap would be visible
        seqs = tbl.column("seq").to_pylist()
        assert seqs == list(range(1, len(seqs) + 1))


def test_promoted_fields_are_real_columns_not_buried_in_json():
    """"Every rejection on OGDC" must not have to parse a JSON blob per row."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # a log fed through the real gateway hook
    with tempfile.TemporaryDirectory() as d:
        log = make(d)
        gw = RiskGateway([OrderQuantityCheck(max_quantity=100)],
                         on_decision=log.on_risk_decision)
        from core.risk import RiskContext
        ctx = RiskContext(date="2026-09-16", timestamp_ms=1,
                          reference_price_minor=10_000)
        # one that passes and one that does not
        gw.authorise(PlaceOrder(symbol="OGDC", cl_ord_id="a", side=Side.BUY,
                                price_minor=9_900, quantity=50), ctx)
        gw.authorise(PlaceOrder(symbol="OGDC", cl_ord_id="b", side=Side.SELL,
                                price_minor=10_100, quantity=500), ctx)
        log.close()
        # read only the columns a query would want -- no JSON decoding
        tbl = pq.read_table(log.path, columns=["event", "symbol", "side",
                                               "price_minor", "quantity",
                                               "allowed", "check"])
        rows = tbl.to_pylist()
        # the approval
        ok = next(r for r in rows if r["event"] == "risk_approved")
        assert ok["symbol"] == "OGDC" and ok["quantity"] == 50
        assert ok["side"] == "BUY" and ok["allowed"] is True
        # the rejection, with the control that produced it in its own column
        bad = next(r for r in rows if r["event"] == "risk_rejected")
        assert bad["quantity"] == 500 and bad["allowed"] is False
        assert bad["check"] == "order_quantity"


def test_an_amendment_is_logged_in_full():
    """A ReplaceOrder must not degrade to a bare event name."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # a log
    with tempfile.TemporaryDirectory() as d:
        log = make(d)
        # an amendment, approved
        log.on_risk_decision(
            ReplaceOrder(symbol="OGDC", cl_ord_id="new", side=Side.BUY,
                         price_minor=9_901, quantity=100, account="C1",
                         orig_cl_ord_id="old", exchange_order_id="X1"),
            RiskDecision.allow("gateway", "all checks passed"))
        log.close()
        # read it back
        rows = pq.read_table(log.path).to_pylist()
        row = next(r for r in rows if r["event"] == "risk_approved")
        # the promoted columns
        assert row["cl_ord_id"] == "new" and row["price_minor"] == 9_901
        # and what it was amending, in the JSON remainder
        extra = json.loads(row["extra"])
        assert extra["orig_cl_ord_id"] == "old"
        assert extra["exchange_order_id"] == "X1"


def test_the_file_is_actually_compressed():
    """Compression is the reason for Parquet; assert it rather than assume it."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # a log with many similar records, which is what a trading day looks like
    with tempfile.TemporaryDirectory() as d:
        log = make(d, row_group_rows=20_000)
        for i in range(20_000):
            log.record("risk_approved",
                       {"symbol": "OGDC", "cl_ord_id": f"T-{i:08d}",
                        "side": "BUY", "price_minor": 28_868 + (i % 5),
                        "quantity": 300, "check": "gateway", "allowed": True,
                        "reason": "all checks passed"})
        log.close()
        # what one uncompressed JSON line of the same content would cost
        one = len(json.dumps({"seq": 1, "ts_ns": time.time_ns(),
                              "event": "risk_approved", "symbol": "OGDC",
                              "cl_ord_id": "T-00000001", "side": "BUY",
                              "price_minor": 28_868, "quantity": 300,
                              "check": "gateway", "allowed": True,
                              "reason": "all checks passed"}))
        # the compressed file against the equivalent JSON lines
        ratio = (one * 20_000) / log.path.stat().st_size
        # dictionary encoding plus zstd on this much repetition should be
        # dramatic; anything under 5x means the schema is doing no work
        assert ratio > 5, f"only {ratio:.1f}x smaller than JSON lines"


# --- the hot path ----------------------------------------------------------

def test_recording_does_not_cost_the_trading_path_a_microsecond():
    """THE REGRESSION GUARD.

    The first version used queue.Queue, whose put() notifies a condition
    variable and forces a context switch on the PRODUCER's thread -- 17
    microseconds per record. Nobody spots that in a diff. This assertion does.
    """
    # the engine will not construct without pyarrow
    _need_parquet()
    # a log with a writer thread actually running, which is the realistic case
    with tempfile.TemporaryDirectory() as d:
        log = make(d, queue_size=2_000_000, row_group_rows=100_000)
        # a realistic payload
        payload = {"symbol": "OGDC", "cl_ord_id": "T-00000001", "side": "BUY",
                   "price_minor": 28_868, "quantity": 300, "check": "gateway",
                   "allowed": True, "reason": "all checks passed"}
        # warm up, so the first-call costs do not count
        for _ in range(20_000):
            log.record("risk_approved", payload)
        # per-call samples, so the TAIL is visible and not just the mean
        samples = []
        for _ in range(50_000):
            t0 = time.perf_counter_ns()
            log.record("risk_approved", payload)
            samples.append(time.perf_counter_ns() - t0)
        samples.sort()
        # the median, which is what the engine pays almost every time
        p50 = samples[len(samples) // 2]
        # and the 99th percentile
        p99 = samples[int(len(samples) * 0.99)]
        # nothing was dropped, so this measured the real path
        assert log.dropped == 0
        # ONE MICROSECOND is the bar. Measured around 300ns; the headroom is
        # for slower machines, not for a reintroduced lock.
        assert p50 < 1_000, f"p50 {p50}ns -- something is locking again"
        # the tail may include a GC pause, so it gets more room
        assert p99 < 20_000, f"p99 {p99}ns"
        log.close()


def test_a_full_buffer_never_blocks_the_caller():
    """Blocking the trading path to write a log is not a trade anyone takes."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # a buffer of one, so it is full essentially always
    with tempfile.TemporaryDirectory() as d:
        log = make(d, queue_size=1)
        # time a burst that cannot possibly fit
        t0 = time.perf_counter_ns()
        for i in range(20_000):
            log.record("tick", {"i": i})
        elapsed = time.perf_counter_ns() - t0
        # dropping is fast; blocking would not be
        assert elapsed / 20_000 < 5_000
        # and the drops were counted rather than hidden
        assert log.dropped > 0
        log.close()


# --- lifecycle -------------------------------------------------------------

def test_refuses_to_reuse_an_existing_file():
    """An audit file that can be overwritten is not an audit file."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # a fixed base path
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "audit_fixed"
        log = AuditLog(Path(d), session_id="T", path=target)
        log.close()
        # a second log at the same base must refuse
        with pytest.raises(FileExistsError):
            AuditLog(Path(d), session_id="T", path=target)


def test_two_logs_started_in_the_same_second_do_not_collide():
    """A crash loop restarts fast; the filename must survive that."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # two logs created back to back, same session id
    with tempfile.TemporaryDirectory() as d:
        a = make(d)
        b = make(d)
        # different files, so the restart is not refused by its own guard
        assert a.path != b.path
        a.close()
        b.close()


def test_closing_twice_is_safe():
    """Shutdown paths run more than once; close() must tolerate it."""
    # the engine will not construct without pyarrow
    _need_parquet()
    # a log
    with tempfile.TemporaryDirectory() as d:
        log = make(d)
        log.close()
        # the second call does nothing and does not raise
        log.close()


if __name__ == "__main__":
    # A pytest file is not a script: there is no runner, and the project root is
    # not on the import path. Say so rather than failing with ModuleNotFoundError.
    raise SystemExit(
        "This is a pytest file, not a script.\n"
        "Run the suite from the Production directory:\n"
        "    python -m pytest -q\n"
        "or one file:\n"
        "    python -m pytest tests/test_audit.py -q")
