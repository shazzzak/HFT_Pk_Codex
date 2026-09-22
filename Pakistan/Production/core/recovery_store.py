"""Single-writer local SQLite journal. No distributed ownership or wire replay."""
# Hash canonical records to detect accidental logical corruption.
import hashlib
# Enforce one owning process on supported Unix hosts.
import fcntl
# Encode only explicit JSON domain values, never executable pickle data.
import json
# Persist transactions through SQLite's crash recovery machinery.
import sqlite3
# Bind ownership to the creating process as well as its thread.
import os
# Bind ownership to the creating thread.
import threading
# Resolve one canonical database and lock location.
from pathlib import Path


# Reject NaN and serialize equivalent values identically.
def canonical(value):
    # Compact sorted JSON makes digests stable across restarts.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


# Digest a canonical domain record.
def digest(value):
    # This detects corruption, not a malicious writer who can rewrite the database.
    return hashlib.sha256(canonical(value).encode()).hexdigest()


# Own a synchronous durable journal and latest state in the same transaction.
class RecoveryStore:
    # Existing stores require exact identity/configuration agreement.
    def __init__(self, path, identity, create=False):
        # Resolve symlinks before choosing the ownership lock.
        self.path = Path(path).resolve()
        # Never silently create an empty account during a recovery attempt.
        if not create and not self.path.is_file():
            # A missing store requires explicit initialization.
            raise FileNotFoundError(self.path)
        # Initialization must not replace an existing account history.
        if create and self.path.exists():
            # Reusing a database is recovery, not initialization.
            raise FileExistsError(self.path)
        # Remember the serial event-loop owner.
        self.owner = (os.getpid(), threading.get_ident())
        # Refuse concurrent processes before opening SQLite.
        self.lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+b")
        # Release resources if ownership or verification fails.
        try:
            # Lock contention fails immediately; it never creates a second trader.
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Recheck initialization under the lock to close the creation race.
            if create and self.path.exists():
                # Another initializer may have won before the lock was acquired.
                raise FileExistsError(self.path)
            # Autocommit permits explicit transaction boundaries below.
            self.db = sqlite3.connect(str(self.path), isolation_level=None)
            # Use rollback journals with synchronous durable commit semantics.
            self.db.execute("PRAGMA journal_mode=DELETE")
            # Ask SQLite to flush both journal and database at commit.
            self.db.execute("PRAGMA synchronous=FULL")
            # Ask macOS to flush hardware caches where supported.
            self.db.execute("PRAGMA fullfsync=ON")
            # Structural corruption must stop startup before state is trusted.
            if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                # Recovery from damaged storage is an operator incident.
                raise RuntimeError("SQLite integrity check failed")
            # Create the schema only for an explicitly new store.
            if create:
                # Keep initialization atomic as well as subsequent mutations.
                self.db.executescript("BEGIN IMMEDIATE; CREATE TABLE meta (identity TEXT NOT NULL); CREATE TABLE current (id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL); CREATE TABLE events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, state_hash TEXT NOT NULL, previous TEXT NOT NULL, hash TEXT NOT NULL); COMMIT;")
                # Store immutable account, session, schema and code/config identity.
                self.db.execute("INSERT INTO meta VALUES (?)", (canonical(identity),))
            # Reject schema/account/config migration by accidental reuse.
            if self.db.execute("SELECT identity FROM meta").fetchall() != [(canonical(identity),)]:
                # Migration requires its own reviewed procedure.
                raise RuntimeError("recovery identity/configuration mismatch")
            # Verify the event chain and the latest checkpoint before loading it.
            self.verify()
        # Do not retain locks after any failed constructor.
        except BaseException:
            # Close a database if construction reached that point.
            if hasattr(self, "db"):
                # Rollback any incomplete SQLite transaction on close.
                self.db.close()
            # Closing the descriptor releases its ownership lock.
            self.lock.close()
            # Preserve the original failure for the caller.
            raise

    # All operations belong to one serialized event loop.
    def check_owner(self):
        # Concurrent callback threads must queue work into the owner instead.
        if (os.getpid(), threading.get_ident()) != self.owner:
            # Fail before touching either durable or in-memory state.
            raise RuntimeError("recovery store used outside owner thread")

    # Read and validate the complete audit chain at startup.
    def verify(self):
        # Protect the SQLite connection's ownership contract.
        self.check_owner()
        # An empty history has a fixed initial chain link.
        previous, count, state_hash = "", 0, None
        # Stream rows rather than copying the journal into memory.
        for seq, kind, payload, state_hash, prior, hashed in self.db.execute("SELECT * FROM events ORDER BY seq"):
            # Recompute every link from the stored canonical content.
            if seq != count + 1 or prior != previous or hashed != digest([seq, kind, json.loads(payload), state_hash, prior]):
                # Never repair or truncate corrupt history automatically.
                raise RuntimeError("recovery journal chain mismatch")
            # Advance only after the link has been verified.
            previous, count = hashed, seq
        # The latest checkpoint must match the final committed journal event.
        row = self.db.execute("SELECT state FROM current WHERE id=1").fetchone()
        # An orphan checkpoint or a missing checkpoint is equally unsafe.
        if (row is None) != (count == 0) or (row is not None and digest(json.loads(row[0])) != state_hash):
            # Do not reconstruct a plausible state from damaged metadata.
            raise RuntimeError("recovery checkpoint mismatch")
        # Cache the next append position only after full verification.
        self.sequence, self.head = count, previous

    # Return detached decoded state to the owning application.
    def load(self):
        # Enforce serialization even for monitoring reads.
        self.check_owner()
        # There is exactly one current checkpoint.
        row = self.db.execute("SELECT state FROM current WHERE id=1").fetchone()
        # A new store has no checkpoint until initialization commits.
        return None if row is None else json.loads(row[0])

    # Commit event evidence and the complete current state atomically.
    def commit(self, kind, payload, state):
        # Never mutate a store from a callback thread.
        self.check_owner()
        # Validate serializability before opening the write transaction.
        encoded, record, hashed_state = canonical(state), canonical(payload), digest(state)
        # Bind the new event to both its predecessor and resulting state.
        seq = self.sequence + 1
        # Compute the next chain link deterministically.
        hashed = digest([seq, kind, payload, hashed_state, self.head])
        # Use an explicit transaction so failed writes cannot half-apply a fill.
        self.db.execute("BEGIN IMMEDIATE")
        # Roll back either write if the transaction cannot commit.
        try:
            # Journal the command and resulting checkpoint identity.
            self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)", (seq, kind, record, hashed_state, self.head, hashed))
            # Publish the state in the same commit as its journal record.
            self.db.execute("INSERT OR REPLACE INTO current VALUES (1,?)", (encoded,))
            # No outbound call may precede this durable boundary.
            self.db.execute("COMMIT")
        # A disk failure is fatal to this application's current in-memory state.
        except BaseException:
            # A failed commit can already have ended its transaction.
            if self.db.in_transaction:
                # Roll back only when SQLite still owns a transaction.
                self.db.execute("ROLLBACK")
            # The application must poison itself instead of continuing.
            raise
        # Advance the cached head only after commit succeeds.
        self.sequence, self.head = seq, hashed

    # Release the single writer at an orderly process boundary.
    def close(self):
        # Shutdown belongs to the same owner as trading callbacks.
        self.check_owner()
        # Closing SQLite leaves committed history intact.
        self.db.close()
        # Closing the descriptor releases flock.
        self.lock.close()
