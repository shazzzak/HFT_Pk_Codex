"""Offline durable OMS boundary; transport reports require externally stable IDs.

The application must exclusively use this facade, not mutate its underlying OMS.
This is local crash recovery, not an implementation of PSX session recovery.
"""
# Serialize explicit domain fields without pickling executable objects.
from dataclasses import asdict
# Translate enums to portable JSON values.
from enum import Enum
# Reuse canonical hashing and durable transactions.
from core.recovery_store import digest
# Restore actual OMS domain objects and preserve pending-message flags.
from core.model import DesiredQuotes, Fill, Order, OrderState, Side


# Convert nested dataclasses and enums into JSON-compatible values.
def plain(value):
    # Enums must retain their stable external value rather than repr text.
    if isinstance(value, Enum):
        # Side and lifecycle state both use explicit string values.
        return value.value
    # Domain dataclasses have a finite explicit set of fields.
    if hasattr(value, "__dataclass_fields__"):
        # Recursively convert their dictionary representation.
        return plain(asdict(value))
    # JSON dictionaries retain their textual keys.
    if isinstance(value, dict):
        # Normalize every contained field.
        return {key: plain(item) for key, item in value.items()}
    # Tuples and lists share JSON array representation.
    if isinstance(value, (tuple, list)):
        # Preserve order because order priority is meaningful.
        return [plain(item) for item in value]
    # Integers, strings, booleans and None are already portable.
    return value


# Persist OMS changes before exposing them to a network adapter.
class DurableManager:
    # The supplied OMS must be new; recovery replaces its volatile state.
    def __init__(self, oms, store):
        # Retain the one underlying OMS and one owning store.
        self.oms, self.store = oms, store
        # Poisoning blocks all further order processing after uncertain local state.
        self.poisoned = False
        # Retain fatal input incidents across process restarts.
        self.incident = None
        # Capture OMS event details alongside each durable mutation.
        self.events = []
        # Avoid an independently failing callback after a partial OMS mutation.
        if oms._orders or oms._on_event is not None:
            # A fresh manager must not hide an existing state or audit sink.
            raise ValueError("fresh OMS with no event sink required")
        # The journal becomes the authoritative OMS event sink.
        oms._on_event = lambda kind, data: self.events.append([kind, plain(data)])
        # Restore committed state, never a caller's guessed account position.
        state = store.load()
        # New stores establish an explicit zero ledger before use.
        if state is None:
            # All pending outbound attempts live inside the atomic checkpoint.
            self.outbox, self.reports = {}, {}
            # Cash here is trading cash movement, not bank buying power.
            self.cash_minor, self.fees_minor = 0, 0
            # Opening equity and account positions need an external bootstrap later.
            self._commit("initialize", {})
        # Existing stores contain exact order and execution state.
        else:
            # Checkpoint schema changes must not be interpreted optimistically.
            if state["schema"] != 1 or state["session"] != oms._session_id or state["account"] != oms._account:
                # Keep the original store untouched on incompatible restart.
                raise ValueError("OMS checkpoint identity mismatch")
            # Restore all order fields, including terminal orders with pending messages.
            oms._orders = {item["cl_ord_id"]: Order(**{**item, "side": Side(item["side"]), "state": OrderState(item["state"])}) for item in state["orders"]}
            # Preserve per-side order priority instead of rebuilding by sorted IDs.
            oms._working = {(symbol, Side(side)): [oms._orders[key] for key in keys] for symbol, side, keys in state["working"]}
            # Preserve amendment/cancel aliases for late reports.
            oms._alias = state["aliases"]
            # Keep old and proposed replacement quantities reserved.
            oms._replacement_reservations = state["replacements"]
            # Restore positions directly from the committed execution ledger.
            oms._position = state["positions"]
            # Never reuse a previously allocated order identifier.
            oms._seq = state["next_id"]
            # Historical counters remain useful in operator status.
            oms.plan_counts = state["plan_counts"]
            # Preserve uncertain sends and execution deduplication across restarts.
            self.outbox, self.reports = state["outbox"], state["reports"]
            # A crash before dispatch must not turn into an automatic resend.
            for entry in self.outbox.values():
                # Prepared messages are uncertain across an ownership epoch.
                if entry["status"] == "prepared":
                    # Only external lifecycle evidence can resolve this liability.
                    entry["status"] = "recovery_held"
            # Unresolved input incidents remain an explicit startup blocker.
            self.incident = state.get("incident")
            # Restore exact integer cash and fees.
            self.cash_minor, self.fees_minor = state["cash_minor"], state["fees_minor"]
            # A prior kill remains latched until a reviewed external procedure clears it.
            if state["killed"]:
                # Never restore an armed controller or silently reset a kill.
                oms._kill.trip("restored durable kill", "recovery", 0)
        # Recovery always withdraws previous strategy intentions.
        symbols = set(oms._position) | {o.symbol for o in oms._orders.values()} | set(oms._desired)
        # Do not resume previously desired quotes after a restart.
        oms._desired = {symbol: DesiredQuotes.flat(symbol) for symbol in symbols}
        # Account equality is never inherited from an earlier connection epoch.
        self.reconciled = False
        # Write the restart boundary and withdrawn intent before processing new input.
        self._commit("startup_disarmed", {})
        # A journaled invalid report cannot be cleared merely by restarting.
        self.poisoned = self.incident is not None

    # Expose read capabilities required by the existing serialized controller.
    def __getattr__(self, name):
        # Inbound callbacks must carry deduplication identity through apply_report().
        if name.startswith("on_") or name.startswith("_mark"):
            # Prevent accidental use of the non-durable callback surface.
            raise AttributeError("use apply_report with a stable report identity")
        # The controller reads order maps, position and ordinary OMS queries.
        return getattr(self.oms, name)

    # Fail closed after any local mutation or storage error.
    def _guard(self):
        # Catch concurrency before mutating the OMS.
        self.store.check_owner()
        # A poisoned process must restart from its last committed state.
        if self.poisoned:
            # No further events or dispatch are safe in this instance.
            raise RuntimeError("durable runtime poisoned; restart and reconcile")

    # Capture every recovery-critical field explicitly.
    def _state(self):
        # Keep session identity, order priority and all reservation fields together.
        return {"schema": 1, "incident": self.incident, "session": self.oms._session_id, "account": self.oms._account, "orders": [plain(order) for order in self.oms._orders.values()], "working": [[symbol, side.value, [order.cl_ord_id for order in orders]] for (symbol, side), orders in self.oms._working.items()], "aliases": dict(self.oms._alias), "replacements": dict(self.oms._replacement_reservations), "positions": dict(self.oms._position), "next_id": self.oms._seq, "plan_counts": dict(self.oms.plan_counts), "outbox": self.outbox, "reports": self.reports, "cash_minor": self.cash_minor, "fees_minor": self.fees_minor, "killed": self.oms._kill.tripped, "desired": [plain(item) for item in self.oms._desired.values()]}

    # Commit command evidence and all events emitted by its OMS mutation.
    def _commit(self, kind, payload):
        # Require the same owner for state and journal operations.
        self._guard()
        # A commit failure invalidates in-memory state even if SQLite rolls back.
        try:
            # Record the full requested input and resulting OMS audit events.
            self.store.commit(kind, {"input": plain(payload), "oms_events": self.events}, self._state())
        # Disk-full and unexpected exceptions must never permit later dispatch.
        except BaseException:
            # Make the current object permanently unusable for trading.
            self.poisoned = True
            # Surface the failure rather than silently degrading durability.
            raise
        # Clear only events already safely committed.
        self.events = []

    # Persist strategy intent even though it will be withdrawn on restart.
    def set_desired(self, desired):
        # Never change desires after a failed transaction.
        self._guard()
        # Apply the existing OMS's domain validation and desired-state protocol.
        self.oms.set_desired(desired)
        # The audit can reconstruct what the strategy requested.
        self._commit("desired", desired)

    # Reserve and persist the entire approved batch before returning any action.
    def reconcile(self, now_ms, date, reference_prices=None):
        # Owner-thread checks precede risk/OMS mutations.
        self._guard()
        # A partially failed planning cycle cannot be used again.
        try:
            # Recovery permits cancellation only; new quotes require reconciliation.
            if not self.reconciled:
                # Existing liabilities remain while intentions become flat.
                self.oms.flatten_all("durable recovery awaiting reconciliation")
            # Use the unchanged risk and OMS diff implementation.
            actions = self.oms.reconcile(now_ms, date, reference_prices)
            # The entire batch becomes durable before even its first send.
            for action in actions:
                # Store action kind and terms to reject altered or duplicate dispatch.
                self.outbox[action.cl_ord_id] = {"type": type(action).__name__, "action": plain(action), "status": "prepared"}
            # Persist the decision inputs needed for later audit review.
            self._commit("reconcile", {"now_ms": now_ms, "date": date, "references": reference_prices or {}, "actions": [plain(a) for a in actions]})
            # Only committed actions are visible to the controller's send loop.
            return actions
        # Even a risk callback failure can leave earlier batch members reserved.
        except BaseException:
            # Do not continue from an incompletely persisted planning cycle.
            self.poisoned = True
            # Require restart from the last atomic checkpoint.
            raise

    # Persist an attempt marker before handing a message to an injected transport.
    def dispatch(self, action, transport):
        # A failed store or wrong thread may never call the transport.
        self._guard()
        # Lookup binds the exact committed action to its outbound attempt.
        entry = self.outbox.get(action.cl_ord_id)
        # Replaying an attempted or changed action is explicitly prohibited.
        if entry is None or entry["status"] != "prepared" or entry["action"] != plain(action) or entry["type"] != type(action).__name__:
            # This is not an automatic reliable-message resend layer.
            raise RuntimeError("outbound action absent, changed, or already attempted")
        # Restarted prepared actions are not automatically redispatched either.
        entry["status"] = "attempted"
        # The potential exposure must survive before any transport side effect.
        self._commit("send_attempt", {"id": action.cl_ord_id})
        # A transport return confirms handoff, not exchange acceptance.
        try:
            # The caller supplies a simulated transport until venue integration exists.
            transport(action)
        # Any exception could follow a successful socket write.
        except BaseException:
            # Keep reservations and prevent blind retries in this process.
            self.poisoned = True
            # Restart must reconcile the durable attempted action.
            raise
        # Keep handoff distinct from receipt and execution.
        entry["status"] = "handed_off"
        # A crash before this commit still recovers the conservative attempted state.
        self._commit("send_handoff", {"id": action.cl_ord_id})

    # Apply one normalized inbound report exactly once within a stable namespace.
    def apply_report(self, report_id, kind, payload):
        # Stable identity is supplied by the eventual venue/session adapter.
        self._guard()
        # Empty identities must not collapse unrelated executions.
        if not isinstance(report_id, str) or not report_id.strip():
            # Include venue/account/trading-session scope in the caller's report ID.
            raise ValueError("stable scoped report identity required")
        # Hash the complete normalized report, including execution fees.
        fingerprint = digest([kind, payload])
        # Exact duplicates have already changed both cash and inventory.
        if report_id in self.reports:
            # Reusing an identifier for different content is a reconciliation incident.
            if self.reports[report_id] != fingerprint:
                # Prevent further actions after contradictory execution evidence.
                good_state = self.store.load()
                # Conflicting execution identity remains fatal across restart.
                good_state["incident"] = {"id": report_id, "kind": kind, "payload": payload, "error": "conflicting duplicate"}
                # Journal contradictory evidence before disabling the runtime.
                try:
                    # Never change cash or position for conflicting duplicates.
                    self.store.commit("report_incident", good_state["incident"], good_state)
                # Even a failed incident write prohibits further activity.
                finally:
                    # The process must stop after contradictory execution evidence.
                    self.poisoned = True
                # Never guess which version is correct.
                raise RuntimeError("conflicting duplicate report")
            # Do not apply the same fill twice, including after restart.
            return False
        # All report mutations must either commit together or poison this process.
        try:
            # Reject malformed replacement terms before mutating the active order.
            if kind == "replaced" and (type(payload.get("price_minor")) is not int or payload["price_minor"] <= 0 or type(payload.get("quantity")) is not int or payload["quantity"] <= 0):
                # Fractional, zero or negative reservations are never admissible.
                raise ValueError("positive integer replacement terms required")
            # An acknowledgement needs a usable exchange handle for subsequent cancellation.
            if kind == "ack" and (not isinstance(payload.get("exchange_order_id"), str) or not payload["exchange_order_id"].strip()):
                # Never turn an unaddressable order into an apparently live one.
                raise ValueError("exchange order identity required")
            # Resolve both original order IDs and cancel/amendment aliases.
            key = payload["cl_ord_id"]
            # Unknown orders need external reconciliation, not speculative mutation.
            order = self.oms._orders.get(key) or self.oms._orders.get(self.oms._alias.get(key, ""))
            # Refuse unknown account activity until it is explicitly investigated.
            if order is None:
                # Preserve the raw report in the failure record below.
                raise ValueError("unknown order report")
            # Fills require strict units and order-side consistency before accounting.
            if kind == "fill":
                # Extract only the declared Fill fields, excluding fees.
                fill = Fill(**{name: Side(value) if name == "side" else value for name, value in payload.items() if name != "fee_minor"})
                # Do not silently accept fractional cash, wrong-symbol fills or impossible prices.
                if fill.symbol != order.symbol or fill.side != order.side or type(fill.quantity) is not int or fill.quantity <= 0 or type(fill.price_minor) is not int or fill.price_minor <= 0 or type(fill.timestamp_ms) is not int:
                    # Normalization errors must be resolved before trading resumes.
                    raise ValueError("invalid normalized fill")
                # A limit order cannot execute worse than its active limit.
                if (fill.price_minor - order.price_minor) * fill.side.sign > 0:
                    # Pending replacement pricing needs venue-specific mapping before live use.
                    raise ValueError("fill violates active limit")
                # Fee amounts are explicit integer minor units, allowing rebates.
                fee = payload.get("fee_minor", 0)
                # Unknown fee precision must not become floating-point ledger drift.
                if type(fee) is not int:
                    # Require the normalized fee contract.
                    raise ValueError("integer fee required")
                # Let the existing lifecycle reject overfills and terminal fills.
                self.oms.on_fill(fill)
                # Buy fills spend cash and sell fills receive cash.
                self.cash_minor -= fill.signed_quantity * fill.price_minor + fee
                # Track fees separately without losing their cash effect.
                self.fees_minor += fee
            # Non-fill lifecycle reports reuse the existing OMS transition rules.
            else:
                # Whitelist the normalized lifecycle contract instead of arbitrary getattr.
                callbacks = {"ack": "on_ack", "cancelled": "on_cancelled", "rejected": "on_rejected", "replaced": "on_replaced", "cancel_rejected": "on_cancel_rejected", "suspended": "on_suspended"}
                # Unsupported reports include trade busts/corrections pending a venue contract.
                if kind not in callbacks:
                    # Never mark an unsupported event as processed.
                    raise ValueError("unsupported normalized report")
                # Apply only known OMS lifecycle operations.
                getattr(self.oms, callbacks[kind])(**payload)
            # Deduplication and cash/order state share one commit boundary.
            self.reports[report_id] = fingerprint
            # Commit before acknowledging consumption to an upstream replay adapter.
            self._commit("report", {"id": report_id, "kind": kind, "payload": payload})
            # Exactly one normalized report was applied.
            return True
        # A rejected report is a persistent incident, not a harmless parser warning.
        except BaseException as error:
            # Save the observed input when storage remains usable.
            was_poisoned = self.poisoned
            # Set the fail-stop flag before an incident write can itself fail.
            self.poisoned = True
            # A previous storage error makes an additional write unsafe.
            if not was_poisoned:
                # Journal the incident against the last committed good state.
                incident = {"id": report_id, "kind": kind, "payload": payload, "error": str(error)}
                # Preserve the last good ledger instead of a partly applied transition.
                good_state = self.store.load()
                # A durable incident requires investigation even after a restart.
                good_state["incident"] = incident
                # Save the failed input and the unchanged good ledger together.
                self.store.commit("report_incident", incident, good_state)
            # No further sends may follow an unresolved report.
            self.poisoned = True
            # Expose the failure to the serialized host.
            raise

    # Compare externally supplied authoritative account state at a synchronized barrier.
    def reconcile_external(self, snapshot, operator):
        # External data must be delivered on the serialized owner thread.
        self._guard()
        # The caller must attest completeness at its actual session recovery barrier.
        if not isinstance(operator, str) or not operator.strip() or snapshot.get("complete") is not True or not isinstance(snapshot.get("barrier"), str) or not snapshot["barrier"].strip():
            # Socket connectivity or an empty partial response is not reconciliation.
            raise ValueError("named operator and complete synchronized snapshot required")
        # Include suspended orders and pending messages in the local unresolved set.
        unresolved = [o for o in self.oms._orders.values() if not o.state.is_terminal or o.has_message_in_flight]
        # This first recovery contract deliberately requires the account to be drained.
        expected_positions = {s: q for s, q in self.oms._position.items() if q}
        # Compare cash movement since the same ledger origin, not absolute bank cash.
        matches = not unresolved and snapshot.get("open_orders") == [] and snapshot.get("positions") == expected_positions and snapshot.get("cash_minor") == self.cash_minor
        # Record mismatches as evidence and leave quote permission disabled.
        self.reconciled = bool(matches)
        # Persist both the external evidence and the exact comparison outcome.
        self._commit("external_reconciliation", {"operator": operator, "snapshot": snapshot, "matched": matches})
        # Return a result; the controller still requires separate explicit arming.
        return self.reconciled

    # Route controller/operator audit records through the same durability boundary.
    def audit(self, kind, payload):
        # Capture kill state and ordinary OMS events with the operator decision.
        self._commit(kind, payload)

    # Provide a detached machine-readable monitoring snapshot.
    def status(self):
        # Monitoring must not race lifecycle processing.
        self.store.check_owner()
        # Preserve uncertain attempts even if the connection is currently healthy.
        return {"poisoned": self.poisoned, "reconciled": self.reconciled, "journal_sequence": self.store.sequence, "positions": dict(self.oms._position), "cash_minor": self.cash_minor, "fees_minor": self.fees_minor, "killed": self.oms._kill.tripped, "unresolved_orders": sum(not o.state.is_terminal or o.has_message_in_flight for o in self.oms._orders.values()), "outbound_status": {key: item["status"] for key, item in self.outbox.items()}, "live_approved": False}
