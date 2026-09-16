"""The audit log: an append-only record of why every order was sent.

THE TEST THIS HAS TO PASS. Not "can we see what happened" -- almost any logging
passes that. The test is: **someone who was not here can reconstruct why a
specific order went out, from the file alone, months later.** That means the
record has to carry the decision, the inputs to the decision, and the state the
decision was made against -- not just the outcome.

It is also, in practice, the only thing that will exist after a bad day. Memory
of what the screen said is not evidence.

===========================================================================
WHY THERE ARE TWO FILES, AND WHY PARQUET ALONE WOULD BE A MISTAKE
===========================================================================
Parquet is COLUMNAR and BATCHED. A single row cannot be appended to it: rows are
accumulated, encoded, compressed and written as a row group. That is what makes
it small and fast to query, and it is exactly what makes it the wrong thing to
put on a hot path.

So the hot path never touches Parquet. It puts a tuple on a queue and returns.
A background thread does the encoding and the compression.

But that creates a second problem, and it is the important one: **whatever is
still in the queue when the process dies is gone -- and the crash is precisely
the event the log exists to explain.** A compressed, queryable record of a
normal morning is of limited interest. The five records before the crash are the
whole point.

Hence two files, with different jobs:

  * `*.parquet`  -- EVERYTHING, compressed, queryable, written in batches by a
                    background thread. Zero cost on the hot path beyond a queue
                    put. This is the file you analyse.
  * `*.critical.jsonl` -- the handful of records that must survive a crash:
                    the kill switch, a rejected order, a state mismatch, session
                    start and end. Written SYNCHRONOUSLY and fsynced, on the hot
                    path, because they are rare (a few dozen a day) and because
                    a durability guarantee that does not cover them is not worth
                    having. This is the file you read after something went wrong.

Every record appears in the Parquet file. Critical ones appear in both.

===========================================================================
WHAT THE HOT PATH ACTUALLY COSTS
===========================================================================
Three things were moved OFF it:

  * `datetime.now().isoformat()` -- allocates and formats a string. Replaced
    with `time.time_ns()`, an integer. Formatted in the writer thread.
  * `json.dumps()` -- the single most expensive thing the old version did per
    record. Now done in the writer thread.
  * the dict construction -- replaced with a tuple.

What remains per record: one `next()` on an atomic counter, one `time_ns()`,
one tuple build, one length check and one `deque.append`. MEASURED, on the
machine this was written on, with the writer thread running:

    non-critical record()      p50   302 ns    p99  2,149 ns
    CRITICAL record() (fsync)       ~122,000 ns

At a thousand records a second that is 0.03% of one core. The fsync figure is
the price of durability and it is paid only on the rare events listed in
CRITICAL_EVENTS -- a few dozen a day, not a few thousand a second.
`tests/test_audit.py` asserts the non-critical figure stays under a microsecond,
so a future change that reintroduces a lock or a string format fails the suite
rather than being discovered in production.

WHY A deque AND NOT queue.Queue. Measured on this machine:

    queue.put_nowait(), no consumer                   639 ns
    queue.put_nowait(), consumer blocked on get()  17,563 ns
    deque.append(), polling consumer                   55 ns

`queue.Queue` is thread-safe through a mutex and a condition variable, and every
put NOTIFIES a waiting consumer -- which forces a context switch on the
producer's thread. That is 17 microseconds of the trading path spent waking up a
logger. A `collections.deque` append is atomic in CPython and touches no lock at
all, so the writer polls instead of waiting. 320x cheaper, for a millisecond of
extra latency on the WRITER side, which nothing cares about.

Single producer (the engine loop) appends; single consumer (the writer thread)
popleft()s. Both operations are atomic, so this needs no lock.

BACKPRESSURE. The buffer is BOUNDED by an explicit length check. If the writer
cannot keep up, the hot path DROPS the record and counts it -- it never blocks,
and it never grows until the process runs out of memory. A dropped record is
still visible, because the sequence number is assigned before the drop, so the
gap shows in the file. Blocking the trading path to write a log is not a trade
anyone would take.
"""
# value objects
from dataclasses import dataclass
# wall-clock formatting, in the WRITER thread only
from datetime import datetime, timezone
# durable writes for the critical sidecar
import os
# the critical-record format, and the variable payload column
import json
# an atomic counter: next() on itertools.count is atomic in CPython, so the hot
# path needs no lock for the sequence number
import itertools
# paths
from pathlib import Path
# an atomic, lock-free buffer, and the background writer
from collections import deque
import threading
# monotonic and wall-clock time
import time
# typing only
from typing import Any, Dict, List, Optional, Tuple
# the risk layer's types, so a decision can be recorded whole
from core.model import Action, CancelOrder, OrderRequest, ReplaceOrder
from core.risk import RiskContext, RiskDecision

# EVENTS THAT MUST SURVIVE A CRASH. Each one is rare, and each one is something
# a person will be reading the file specifically to understand. These get a
# synchronous fsync on the hot path; everything else rides the queue.
CRITICAL_EVENTS = frozenset({
    # the kill switch tripping, and everything that follows from it
    "kill_switch_tripped", "kill_switch_reset", "flatten_all",
    # the exchange or our own controls refusing something
    "order_rejected", "risk_rejected", "cancel_rejected",
    # our state and the exchange's disagreeing, which is unrecoverable if lost
    "unknown_order", "fill_unknown_order", "cancel_unknown_order",
    "replace_unknown_order",
    # session lifecycle
    "session_start", "session_end", "config_change",
})

# THE PARQUET SCHEMA. The fields that get their own column are the ones worth
# filtering and sorting on without decoding anything: a query like "every
# rejection on OGDC between 11:00 and 11:05" must not have to parse JSON.
#
# Everything else goes into `extra` as a JSON string. That is not laziness --
# audit payloads are genuinely heterogeneous, and a column per field that any
# event might ever carry would be a table of mostly nulls. Dictionary encoding
# plus compression handles the repeated JSON keys well.
_COLUMNS = ("seq", "ts_ns", "event", "symbol", "cl_ord_id", "side",
            "price_minor", "quantity", "check", "allowed", "reason", "extra")
# the payload keys promoted out of `extra` into their own column
_PROMOTED = ("symbol", "cl_ord_id", "side", "price_minor", "quantity",
             "check", "allowed", "reason")


class AuditLog:
    """Two-file audit record: compressed Parquet for all of it, fsynced JSON
    lines for the records that must survive a crash."""

    def __init__(self, directory: Path, session_id: str,
                 queue_size: int = 200_000,
                 row_group_rows: int = 50_000,
                 flush_interval_s: float = 5.0,
                 compression: str = "zstd",
                 path: Optional[Path] = None):
        # where the files live
        self._dir = Path(directory)
        # create it if this is the first run
        self._dir.mkdir(parents=True, exist_ok=True)
        # the session these files belong to
        self._session_id = session_id
        # THE FILENAMES CARRY THE SESSION AND THE TIME AND ARE NEVER REUSED.
        # MICROSECONDS, not seconds: a process that restarts inside the same
        # second -- which is what a crash loop does -- would otherwise collide
        # with its own previous file and be refused by the guard below, turning
        # a recoverable restart into a failure to start.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        # the explicit base path is for tests; production derives one
        base = Path(path) if path is not None \
            else self._dir / f"audit_{session_id}_{stamp}"
        # everything, compressed
        self._parquet_path = base.with_suffix(".parquet")
        # the records that must survive a crash
        self._critical_path = base.with_suffix(".critical.jsonl")
        # refuse rather than append to, or overwrite, an existing file
        for pth in (self._parquet_path, self._critical_path):
            # an audit file that can be overwritten is not an audit file
            if pth.exists():
                raise FileExistsError(
                    f"audit file {pth} already exists; refusing to reuse it")
        # the sequence counter. next() on itertools.count is atomic in CPython,
        # so the hot path needs no lock even if the engine ever goes threaded.
        self._seq = itertools.count(1)
        # the critical sidecar, line-buffered
        self._critical_fh = open(self._critical_path, "a", buffering=1,
                                 encoding="utf-8")
        # THE BUFFER. A deque, not a queue.Queue: append is atomic and touches
        # no lock, where queue.Queue notifies a condition variable on every put
        # and costs the producer a context switch. See the module docstring.
        self._buf: "deque[Tuple]" = deque()
        # bounded by an explicit check, since a maxlen deque would silently
        # discard the OLDEST record and give us no way to count the loss
        self._max_buffered = max(1, queue_size)
        # how long the writer sleeps when the buffer is empty
        self._poll_s = 0.001
        # set to stop the writer
        self._stop = threading.Event()
        # the last sequence number issued. Tracked as a plain attribute rather
        # than read back out of the counter, because reading a counter consumes
        # a value -- which is exactly the bug the first version of session_end
        # had: it burned a sequence number to report the count, leaving a
        # one-record gap at the end of every file and making the gap detector
        # cry wolf on every clean shutdown.
        self._last_seq = 0
        # how many records the hot path had to drop
        self.dropped = 0
        # write failures on the critical sidecar, counted rather than raised
        self.write_failures = 0
        # how many records the writer has committed to Parquet
        self.written = 0
        # batching policy for the writer
        self._row_group_rows = max(1, row_group_rows)
        self._flush_interval_s = max(0.1, flush_interval_s)
        self._compression = compression
        # FAIL NOW, NOT AT SHUTDOWN. Without Parquet the full log is never
        # captured, and discovering that in close() means discovering it after
        # the session it was meant to record. Import it here.
        try:
            # the Parquet machinery, imported once and handed to the writer
            import pyarrow  # noqa: F401
            import pyarrow.parquet  # noqa: F401
        except Exception as e:
            # close the sidecar we already opened, so the failure leaves nothing
            self._critical_fh.close()
            # and say exactly what is wrong
            raise RuntimeError(
                f"audit: pyarrow is required for the Parquet log and is not "
                f"importable ({e!r}). The audit log is not optional, so the "
                f"engine refuses to start without it.") from e
        # kept for the writer thread to report an unexpected failure
        self._writer_error: Optional[BaseException] = None
        # the background thread; daemon=False so close() can join it
        self._thread = threading.Thread(target=self._writer_loop,
                                        name=f"audit-{session_id}",
                                        daemon=False)
        # closed once
        self._closed = False
        # start writing
        self._thread.start()
        # the first record names the session, so the files are self-identifying
        self.record("session_start", {"session_id": session_id,
                                      "parquet": str(self._parquet_path),
                                      "critical": str(self._critical_path)})

    # ---- paths, for an operator who needs to find the files ---------------
    @property
    def path(self) -> Path:
        """The Parquet file -- the one to analyse."""
        # everything is in here
        return self._parquet_path

    @property
    def critical_path(self) -> Path:
        """The crash-survivable file -- the one to read after something broke."""
        # only the records that matter when the process died
        return self._critical_path

    # =======================================================================
    # THE HOT PATH
    # =======================================================================
    def record(self, event: str, payload: Optional[Dict[str, Any]] = None,
               critical: Optional[bool] = None) -> None:
        """Append one record. Never raises. Never blocks.

        The payload dict is taken BY REFERENCE and encoded later, on the writer
        thread. The caller must not mutate it afterwards -- every caller in this
        engine builds a fresh dict per record, which is why that is safe.
        """
        # the sequence number is assigned FIRST and unconditionally, so a record
        # that is dropped below still leaves a visible gap in the file
        seq = next(self._seq)
        # remember it for session_end, without consuming another
        self._last_seq = seq
        # an integer, not a formatted string. Formatting moves to the writer.
        ts_ns = time.time_ns()
        # whether this one also gets a synchronous, fsynced write
        force = critical if critical is not None else (event in CRITICAL_EVENTS)
        # A TUPLE, not a dict: cheaper to build, and the writer wants positional
        # data anyway
        row = (seq, ts_ns, event, payload or {})
        # NEVER BLOCK. A full buffer means the writer is behind; dropping a log
        # line is a bad outcome, and blocking the trading path to avoid it is a
        # worse one. len() on a deque is O(1).
        if len(self._buf) >= self._max_buffered:
            # counted, and visible as a sequence gap in the file
            self.dropped += 1
        else:
            # atomic, lock-free
            self._buf.append(row)
        # CRITICAL RECORDS GO TO DISK NOW. This is a real syscall on the hot
        # path and it is deliberate: these events are rare (a few dozen a day)
        # and they are the ones that have to exist after a crash.
        if force:
            self._write_critical(seq, ts_ns, event, payload or {})

    def _write_critical(self, seq: int, ts_ns: int, event: str,
                        payload: Dict[str, Any]) -> None:
        """Synchronous, fsynced write of one record. Never raises."""
        # a dead logger is a problem; a dead engine with live orders is worse
        try:
            # the same shape as a Parquet row, readable without any tooling
            line = json.dumps({"seq": seq, "ts_ns": ts_ns,
                               "ts": _iso(ts_ns), "session": self._session_id,
                               "event": event, "data": payload}, default=str)
            # write it
            self._critical_fh.write(line + "\n")
            # push the buffer to the OS
            self._critical_fh.flush()
            # and the OS's buffer to the device
            os.fsync(self._critical_fh.fileno())
        except Exception:
            # counted and surfaced, never raised
            self.write_failures += 1

    # =======================================================================
    # THE WRITER THREAD -- everything expensive happens here
    # =======================================================================
    def _writer_loop(self) -> None:
        """Drain the buffer into Parquet row groups until told to stop."""
        # a writer that dies silently leaves a short file and no explanation
        try:
            # the real loop
            self._writer_body()
        except BaseException as e:  # noqa: BLE001 -- deliberately everything
            # close() turns this into a loud failure
            self._writer_error = e

    def _writer_body(self) -> None:
        """The loop itself."""
        # available: the constructor already proved it
        import pyarrow as pa
        import pyarrow.parquet as pq
        # the columnar buffer, one list per column
        cols: Dict[str, List[Any]] = {c: [] for c in _COLUMNS}
        # the schema, fixed so every row group matches
        schema = pa.schema([
            # monotonic within the file; a jump means records were lost
            ("seq", pa.int64()),
            # nanoseconds since epoch, an integer so the hot path stays cheap
            ("ts_ns", pa.int64()),
            # what happened
            ("event", pa.string()),
            # the promoted columns, all nullable because not every event has them
            ("symbol", pa.string()),
            ("cl_ord_id", pa.string()),
            ("side", pa.string()),
            ("price_minor", pa.int64()),
            ("quantity", pa.int64()),
            ("check", pa.string()),
            ("allowed", pa.bool_()),
            ("reason", pa.string()),
            # everything else, as JSON. Heterogeneous payloads would otherwise
            # need a column per field any event might ever carry.
            ("extra", pa.string()),
        ])
        # the writer, created on first use so an empty session leaves no file
        writer = None
        # when the current batch was started, for the time-based flush
        last_flush = time.monotonic()
        # how many rows are buffered
        buffered = 0

        def flush():
            """Encode, compress and append one row group."""
            # nothing to write
            nonlocal writer, buffered, last_flush
            if not buffered:
                return
            # create the file on first flush
            if writer is None:
                writer = pq.ParquetWriter(str(self._parquet_path), schema,
                                          compression=self._compression)
            # one record batch from the column buffers
            writer.write_table(pa.Table.from_pydict(
                {c: cols[c] for c in _COLUMNS}, schema=schema))
            # count what has reached the file
            self.written += buffered
            # reset the buffers
            for c in _COLUMNS:
                cols[c].clear()
            buffered = 0
            last_flush = time.monotonic()

        def take(row):
            """Split one record into promoted columns and a JSON remainder."""
            # unpack
            nonlocal buffered
            seq, ts_ns, event, payload = row
            # the fixed columns
            cols["seq"].append(seq)
            cols["ts_ns"].append(ts_ns)
            cols["event"].append(event)
            # the promoted ones, absent as None
            for key in _PROMOTED:
                cols[key].append(payload.get(key))
            # whatever is left, as JSON
            rest = {k: v for k, v in payload.items() if k not in _PROMOTED}
            # None rather than "{}" so the column compresses to almost nothing
            cols["extra"].append(json.dumps(rest, default=str) if rest else None)
            # one more row buffered
            buffered += 1

        # POLL, never block on a condition variable -- that is what keeps the
        # producer's append at 55ns instead of 17 microseconds
        while True:
            # how many records are waiting right now. Reading the length once
            # and draining exactly that many keeps this loop finite even while
            # the producer keeps appending.
            pending = len(self._buf)
            # nothing to do: flush on the timer, then sleep
            if not pending:
                # a time-based flush, so an idle session still reaches disk
                if time.monotonic() - last_flush >= self._flush_interval_s:
                    flush()
                # stop once the producer has finished AND the buffer is empty
                if self._stop.is_set():
                    # write the tail
                    flush()
                    # close the file if one was ever created
                    if writer is not None:
                        writer.close()
                    return
                # wait a millisecond; nothing cares about writer latency
                time.sleep(self._poll_s)
                continue
            # drain what is there. popleft is atomic and the producer only
            # appends, so single-producer/single-consumer needs no lock.
            for _ in range(pending):
                try:
                    # take one
                    take(self._buf.popleft())
                except IndexError:
                    # cannot happen with one consumer, but a log must not die
                    break
                # flush on size, mid-drain
                if buffered >= self._row_group_rows:
                    flush()

    # =======================================================================
    # The hooks the rest of the engine plugs into
    # =======================================================================
    def on_risk_decision(self, action: Action, decision: RiskDecision) -> None:
        """Wire this into RiskGateway(on_decision=...).

        Records approvals as well as rejections. An approval is the more
        important of the two for the test at the top of this file: "why was this
        order sent" is answered by the approval and the checks behind it, while
        a rejection only explains an order that never existed.
        """
        # the promoted columns come first, so they land in real Parquet columns
        row: Dict[str, Any] = {"symbol": action.symbol,
                               "check": decision.check,
                               "allowed": decision.allowed,
                               "reason": decision.reason,
                               "kind": type(action).__name__}
        # any order request -- a placement OR an amendment -- carries price,
        # size and account. Tested on the shared base so a future request type
        # is logged in full by default rather than by remembering to add it.
        if isinstance(action, OrderRequest):
            row.update({"cl_ord_id": action.cl_ord_id,
                        "side": action.side.value,
                        "price_minor": action.price_minor,
                        "quantity": action.quantity,
                        "account": action.account})
            # an amendment also names what it is amending
            if isinstance(action, ReplaceOrder):
                row.update({"orig_cl_ord_id": action.orig_cl_ord_id,
                            "exchange_order_id": action.exchange_order_id})
        # a cancel carries the order it refers to
        elif isinstance(action, CancelOrder):
            row.update({"cl_ord_id": action.cl_ord_id,
                        "orig_cl_ord_id": action.orig_cl_ord_id})
        # EVERY CONTROL'S VERDICT, not only the gateway's summary. Without this
        # an approval records "all checks passed", which says nothing about the
        # state the decision was made against -- and the measuring controls put
        # their readings here.
        if decision.details:
            row["checks"] = [{"check": d.check, "reason": d.reason}
                             for d in decision.details]
        # one record
        self.record("risk_approved" if decision.allowed else "risk_rejected",
                    row)

    def on_oms_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Wire this into OrderManager(on_event=...)."""
        # the order manager's vocabulary is already the right one, and its keys
        # (symbol, cl_ord_id, side, px, qty) mostly map straight to columns
        self.record(event, payload)

    def record_context(self, symbol: str, ctx: RiskContext,
                       extra: Optional[Dict[str, Any]] = None) -> None:
        """Record the market and inventory state a decision was made against.

        The decision alone does not answer "why". A price that looks wrong in
        hindsight was reasonable against the book at the time, and without the
        book at the time there is no way to tell the two apart.
        """
        # the state, plus whatever the caller wants to add
        row: Dict[str, Any] = {"symbol": symbol, "date": ctx.date,
                               "ts_ms": ctx.timestamp_ms,
                               "position": ctx.position,
                               "reference_px": ctx.reference_price_minor}
        # merge the extras
        row.update(extra or {})
        # one record
        self.record("context", row)

    # =======================================================================
    def close(self, timeout_s: float = 30.0) -> None:
        """Flush, join the writer and close. Safe to call more than once."""
        # nothing to do if already closed
        if self._closed:
            return
        self._closed = True
        # a final record, so a file that ends without one is visibly truncated
        self.record("session_end", {"records": self._last_seq,
                                    "written": self.written,
                                    "dropped": self.dropped,
                                    "write_failures": self.write_failures})
        # tell the writer to finish once it has drained. An Event, not a
        # sentinel record: a sentinel could itself be the thing that gets
        # dropped when the buffer is full.
        self._stop.set()
        # wait for the tail to reach the file
        self._thread.join(timeout=timeout_s)
        # the critical sidecar is already fsynced per record; just close it
        try:
            # release the handle
            self._critical_fh.close()
        except Exception:
            # counted, never raised -- we are shutting down either way
            self.write_failures += 1
        # a writer that died mid-session lost records, and that is not
        # something to discover by noticing the file is short
        if self._writer_error is not None:
            raise RuntimeError(
                f"audit: the Parquet writer failed ({self._writer_error!r}). "
                f"Critical records are still in {self._critical_path}, but the "
                f"full log is incomplete.")

    def __enter__(self) -> "AuditLog":
        # usable as a context manager, so close() is not forgotten
        return self

    def __exit__(self, *exc) -> None:
        # always close, even on the way out of an exception
        self.close()


def _iso(ts_ns: int) -> str:
    """Nanoseconds since epoch as a UTC ISO string. Writer/critical path only."""
    # seconds for the datetime, nanoseconds preserved separately in ts_ns
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()
