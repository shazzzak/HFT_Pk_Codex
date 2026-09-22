"""Read-only progress reporting; never touches trading state or SQLite connections."""
# Hash inputs incrementally while reporting real completed byte counts.
import hashlib
# Resolve file names for readable progress details.
from pathlib import Path
# Serialize progress snapshots and stop the reporter promptly.
import threading
# Measure durations with a monotonic clock.
import time


# Report liveness separately from measured work advancement.
class Heartbeat:
    # Tests can inject a short interval and capture output without a long wait.
    def __init__(self, label, interval=10.0, emit=None):
        # Reject a busy-loop interval before starting a background thread.
        if interval <= 0:
            # Timing configuration must be explicit and valid.
            raise ValueError("heartbeat interval must be positive")
        # Keep the label and reporting period immutable throughout this task.
        self.label, self.interval = label, interval
        # Flush terminal output so buffering does not look like a stalled run.
        self.emit = emit or (lambda message: print(message, flush=True))
        # Lock only lightweight diagnostic fields, never the account or simulator.
        self.lock, self.stop = threading.Lock(), threading.Event()
        # Initialize elapsed and progress timestamps from the same clock.
        self.started = self.stage_started = self.last_change = time.monotonic()
        # Begin with an explicit starting stage before any expensive work.
        self.name, self.done, self.total, self.detail = "starting", 0, None, ""
        # Create the thread only on context entry, keeping imports side-effect free.
        self.thread = None

    # Start periodic reporting before entering the expensive work.
    def __enter__(self):
        # A daemon reporter cannot hold a failed worker process open.
        self.thread = threading.Thread(target=self._loop, daemon=True)
        # Begin reporting while the main thread retains exclusive financial ownership.
        self.thread.start()
        # Return the progress handle to the serial worker.
        return self

    # Stop the reporter on successful completion, exception or interruption.
    def __exit__(self, kind, value, traceback):
        # Wake the waiting thread immediately instead of waiting ten seconds.
        self.stop.set()
        # Ensure no delayed progress output leaks into a later task.
        self.thread.join()
        # Exceptions belong to the original gate and must propagate unchanged.
        return False

    # Announce each new processing stage immediately.
    def stage(self, name, total=None, detail=""):
        # Keep a stage transition atomic for the reporting thread.
        with self.lock:
            # Reset measured progress for this stage only.
            self.name, self.done, self.total, self.detail = name, 0, total, str(detail)
            # Track time since both stage start and its latest completed work.
            self.stage_started = self.last_change = time.monotonic()
        # Stage changes should be visible even if they complete within ten seconds.
        self.report()

    # Publish completed work from the main thread without touching trading objects.
    def update(self, done, detail=""):
        # Preserve a consistent counter/detail/timestamp snapshot.
        with self.lock:
            # Repeated identical counters must not pretend that work advanced.
            if done != self.done or str(detail) != self.detail:
                # Measure the age of the most recent observed progress change.
                self.last_change = time.monotonic()
            # Store only immutable diagnostic values.
            self.done, self.detail = done, str(detail)

    # Format a detached progress snapshot with truthful idle time.
    def report(self):
        # Read all fields at one point without accessing simulator state.
        with self.lock:
            # Monotonic durations remain valid across wall-clock adjustments.
            now = time.monotonic()
            # Unknown totals are explicitly shown as a running stage.
            count = f"{self.done:,}/{self.total:,}" if self.total is not None else "running"
            # A heartbeat proves the reporter is alive, not that work advanced.
            message = f"[{self.label}] {self.name}: {count} | stage {now-self.stage_started:.0f}s | total {now-self.started:.0f}s | last progress {now-self.last_change:.0f}s ago | {self.detail}"
        # Do not hold the diagnostic lock while writing to a terminal.
        self.emit(message)

    # Periodically report even during a slow file read or synchronous commit.
    def _loop(self):
        # Event.wait allows prompt shutdown without a blocking sleep.
        while not self.stop.wait(self.interval):
            # This thread never inspects account state or uses its SQLite connection.
            self.report()

    # Preserve event identity and order while counting fully processed events.
    def track(self, events, name):
        # Announce the precise event denominator before the replay starts.
        self.stage(name, len(events), "events completed")
        # Feed the unchanged input objects to the existing simulator loop.
        for index, event in enumerate(events, 1):
            # The simulator completes this event before requesting the next one.
            yield event
            # Update after processing, never before a potentially slow commit.
            self.update(index, "events completed")
        # Ensure the final completed count appears even for very short stages.
        self.report()


# Freeze exact input bytes with periodic file and byte progress.
def hash_files(paths, label):
    # Materialize file order once for deterministic hashing and totals.
    paths = sorted(Path(path) for path in paths)
    # Keep returned hashes identical to the previous digest implementation.
    hashes = {}
    # Report throughout slow reads, including a stalled individual read.
    with Heartbeat(label) as progress:
        # Byte totals measure actual reading rather than merely file-open attempts.
        total = sum(path.stat().st_size for path in paths)
        # Announce the aggregate byte denominator.
        progress.stage("hashing bytes", total, f"0/{len(paths)} files")
        # Track completed bytes across every file in this phase.
        completed = 0
        # Preserve exact file-level SHA-256 values in the existing manifest shape.
        for index, path in enumerate(paths, 1):
            # Reset the digest at each file boundary.
            digest = hashlib.sha256()
            # Identify the active file before a potentially slow first read.
            progress.update(completed, f"file {index}/{len(paths)}: {path.name}")
            # Stream source/data files instead of retaining their bytes in memory.
            with path.open("rb") as stream:
                # Bound each read and update the heartbeat after actual data arrives.
                for block in iter(lambda: stream.read(1024*1024), b""):
                    # Hash every byte using the previous algorithm.
                    digest.update(block)
                    # Count only bytes successfully read and hashed.
                    completed += len(block)
                    # Publish progress without changing any provenance semantics.
                    progress.update(completed, f"file {index}/{len(paths)}: {path.name}")
            # Preserve the absolute/path-string keys expected by the gate.
            hashes[str(path)] = digest.hexdigest()
        # Show completion immediately before returning to the caller.
        progress.report()
    # Return the exact old digest mapping with additional observability only.
    return hashes
