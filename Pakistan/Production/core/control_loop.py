"""Serialized offline-testable application controls; no network session implementation.

The eventual live driver must deliver inbound order events, then call cycle(),
and schedule cycle() independently of market traffic. Its transport and broker
reconciliation callbacks must be backed by certified session/recovery state.
Callbacks may withdraw through feed_unsafe(), but must not re-enter cycle()
or synchronously deliver order acknowledgements during outbound dispatch.
"""
# Immutable explicit liquidation instructions.
from dataclasses import dataclass
# All controls share the existing desired-state and order domain.
from core.model import DesiredQuotes, QuoteIntent, Side


# Liquidation is an explicit bounded-price instruction, separate from cancellation.
@dataclass(frozen=True)
class LiquidationPolicy:
    # Never submit more than this many shares per liquidation clip.
    max_clip: int
    # A buy cannot pay above this limit; a sell cannot sell below it.
    limit_price_minor: int

    # Validate operational quantities before any control can use them.
    def __post_init__(self):
        # Booleans and fractional units are not valid trading settings.
        if type(self.max_clip) is not int or type(self.limit_price_minor) is not int or self.max_clip <= 0 or self.limit_price_minor <= 0:
            # Refuse an unusable liquidation instruction.
            raise ValueError("positive integer clip and limit price required")

    # Express only the side which reduces the current signed inventory.
    def desired(self, symbol, position):
        # No inventory means there is nothing to liquidate.
        if position == 0:
            # Never start a new position from a flat account.
            return DesiredQuotes.flat(symbol)
        # Long inventory sells; short inventory buys.
        side = Side.SELL if position > 0 else Side.BUY
        # Cap each clip at the actual remaining inventory.
        intent = QuoteIntent(side, self.limit_price_minor, min(abs(position), self.max_clip))
        # A liquidation instruction never includes an acquiring quote.
        return DesiredQuotes(symbol, bid=intent if side is Side.BUY else None, ask=intent if side is Side.SELL else None)


# Own arming and timer behavior without inventing a FIX transport or ledger.
class ControlLoop:
    # Inject real components and explicit authoritative readiness callbacks.
    def __init__(self, oms, kill_switch, symbols, send, transport_ready, broker_reconciled, feed_ready, poll_feed, audit):
        # Keep the manager and gateway's shared kill state.
        self.oms, self.kill = oms, kill_switch
        # Freeze the subscribed operational scope.
        self.symbols = frozenset(symbols)
        # The transport must return only after accepting responsibility for an action.
        self.send = send
        # Connection/session readiness is supplied by the eventual certified transport.
        self.transport_ready = transport_ready
        # This must reflect fresh broker/engine reconciliation, not just an empty OMS.
        self.broker_reconciled = broker_reconciled
        # This must combine book, phase, reference-data and tick-recovery readiness.
        self.feed_ready = feed_ready
        # A real timer calls this even when market traffic stops.
        self.poll_feed = poll_feed
        # Audit must be durable in a live assembly; exceptions are not swallowed.
        self.audit = audit
        # Startup is always disarmed, regardless of connection state.
        self.armed = set()
        # Remember withdrawal reasons so timer polling does not flood the audit log.
        self.unsafe_reasons = {}
        # Liquidation instructions require a separate named operator action.
        self.liquidation = {}
        # Distinguish draining old quotes from maintaining an actual liquidation clip.
        self.liquidation_started = set()
        # Shutdown withdraws quotes; it does not implicitly flatten inventory.
        self.stopping = False
        # A transport ambiguity cannot be cleared by ordinary heartbeat recovery.
        self.send_failed = False
        # Track monotonic control-loop time separately from exchange time.
        self.last_monotonic_ms = None

    # Require named, deliberate operator actions for quote permission changes.
    def _operator(self, operator):
        # Empty identity makes the control audit uninterpretable.
        if not isinstance(operator, str) or not operator.strip():
            # Refuse unattributed intervention.
            raise ValueError("named operator required")

    # Arm only explicitly chosen symbols after all startup checks pass.
    def arm(self, operator, symbols):
        # Establish an attributable operator instruction.
        self._operator(operator)
        # Materialize the requested scope exactly once.
        requested = frozenset(symbols)
        # Unknown symbols cannot acquire quote permission dynamically.
        if not requested or not requested <= self.symbols:
            # Explain the invalid scope.
            raise ValueError("arm requires known subscribed symbols")
        # Live quoting requires all external recovery prerequisites.
        if self.stopping or self.send_failed or self.kill.tripped or not self.transport_ready() or not self.broker_reconciled() or not all(self.feed_ready(s) for s in requested):
            # A callback turning healthy later must not automatically rearm.
            raise RuntimeError("startup or recovery checks are incomplete")
        # Do not arm while any order in the requested scope remains unresolved.
        if any(not order.state.is_terminal or order.has_message_in_flight for order in self.oms._orders.values() if order.symbol in requested):
            # Outstanding orders need reconciliation or cancellation first.
            raise RuntimeError("orders remain outstanding or unresolved")
        # Record authorization before changing quote permission.
        self.audit("operator_arm", {"operator": operator, "symbols": sorted(requested)})
        # Explicit operator arming ends any liquidation instruction for these names.
        for symbol in requested:
            # A symbol cannot be in market-making and liquidation modes at once.
            self.liquidation.pop(symbol, None)
            # Arming removes all remaining liquidation state for that instrument.
            self.liquidation_started.discard(symbol)
            # A future safety loss must emit a fresh withdrawal record.
            self.unsafe_reasons.pop(symbol, None)
        # Grant only the approved symbol permissions.
        self.armed.update(requested)

    # Feed callbacks withdraw intentions; cycle() emits the ordinary OMS diff.
    def feed_unsafe(self, symbol, reason):
        # Ignore events outside this controller's configured scope.
        if symbol not in self.symbols:
            # Unknown feed subscriptions cannot create OMS state.
            return
        # Repeated identical timer observations do not create duplicate audit traffic.
        if self.unsafe_reasons.get(symbol) == reason and symbol not in self.armed and symbol not in self.liquidation:
            # The desired state is already flat from the earlier transition.
            return
        # Preserve the current withdrawal cause.
        self.unsafe_reasons[symbol] = reason
        # Loss of feed safety removes market-making permission immediately.
        self.armed.discard(symbol)
        # Remove any still-running liquidation instruction as well.
        self.liquidation.pop(symbol, None)
        # A future liquidation request must restart the drain barrier.
        self.liquidation_started.discard(symbol)
        # Withdrawal uses the same desired-state path as every ordinary quote change.
        self.oms.set_desired(DesiredQuotes.flat(symbol))
        # Preserve why this symbol stopped quoting.
        self.audit("feed_unsafe", {"symbol": symbol, "reason": reason})

    # Strategy output reaches the OMS only through current quote permission.
    def quote(self, desired):
        # Refuse stale strategy calls after a halt, stop or recovery transition.
        if desired.symbol not in self.armed or self.kill.tripped or not self.feed_ready(desired.symbol):
            # Remove any prior desire rather than retaining a stale quote.
            self.feed_unsafe(desired.symbol, "quote permission absent")
            # No new strategy intent was accepted.
            return False
        # Pass the original strategy intent unchanged to the OMS.
        self.oms.set_desired(desired)
        # Risk still evaluates it during reconcile(), not here.
        return True

    # Cancellation and inventory liquidation are deliberately separate operator actions.
    def request_liquidation(self, operator, symbol, policy):
        # Require an attributable instruction.
        self._operator(operator)
        # Accept only a validated policy for a configured instrument.
        if symbol not in self.symbols or not isinstance(policy, LiquidationPolicy):
            # A malformed operator instruction cannot become an order.
            raise ValueError("known symbol and explicit liquidation policy required")
        # Liquidation also needs trustworthy market and account state.
        if self.kill.tripped or self.send_failed or not self.transport_ready() or not self.broker_reconciled() or not self.feed_ready(symbol):
            # A kill switch cannot be bypassed by calling an order an unwind.
            raise RuntimeError("liquidation prerequisites are incomplete")
        # Log the price/size authorization before changing intentions.
        self.audit("operator_liquidate", {"operator": operator, "symbol": symbol, "max_clip": policy.max_clip, "limit_price_minor": policy.limit_price_minor})
        # Remove market-making permission for this symbol.
        self.armed.discard(symbol)
        # First cancel all old quotes; do not overlap them with liquidation.
        self.oms.set_desired(DesiredQuotes.flat(symbol))
        # Activate the independent policy only after those orders are terminal.
        self.liquidation[symbol] = policy
        # Do not mistake existing market-making orders for an unwind clip.
        self.liquidation_started.discard(symbol)

    # Emergency stop cancels through the manager's usual kill-switch listener.
    def stop(self, operator, reason, exchange_ms):
        # Require operator attribution for the stop request.
        self._operator(operator)
        # Remove all future quote permissions before recording the stop.
        self.armed.clear()
        # Stop does not implicitly authorize liquidation trades.
        self.liquidation.clear()
        # No clip may resume after an operator stop.
        self.liquidation_started.clear()
        # Keep the application in stopping mode.
        self.stopping = True
        # The OMS listener sets ordinary desired quotes to flat.
        self.kill.trip(reason, operator, exchange_ms)
        # Persist the intervention for later reconstruction.
        self.audit("operator_stop", {"operator": operator, "reason": reason})

    # Run after inbound lifecycle events and from an independent periodic timer.
    def cycle(self, monotonic_ms, exchange_ms, date, references):
        # Reject a broken local timebase before polling feeds or planning actions.
        valid_clock = type(monotonic_ms) is int and monotonic_ms >= 0 and (self.last_monotonic_ms is None or monotonic_ms >= self.last_monotonic_ms)
        # A broken timer must not prevent cancellation through the remaining loop.
        if not valid_clock:
            # A clock failure stops quotes through the same global control.
            self.stop("control_loop", "monotonic clock failure", exchange_ms)
        # Poll only with a valid timestamp; the kill state still permits cancels.
        if valid_clock:
            # Record the last usable monotonic reading.
            self.last_monotonic_ms = monotonic_ms
            # Feed silence must be detected even when no market messages arrive.
            try:
                # A feed parser or timer failure is a safety event, not permission to quote.
                self.poll_feed(monotonic_ms)
            # Stop before reconciliation so this same cycle can dispatch cancellations.
            except Exception as error:
                # The audit records the fault while OMS uses its normal cancel path.
                self.stop("control_loop", f"feed polling failed: {type(error).__name__}", exchange_ms)
        # A lost connection or account reconciliation disarms all symbols.
        if not self.transport_ready() or not self.broker_reconciled():
            # Retain order reservations until exchange recovery resolves them.
            for symbol in self.symbols:
                # This changes desires, not filled inventory or pending order state.
                self.feed_unsafe(symbol, "transport or broker reconciliation unavailable")
        # Freshness may change independently for each subscribed symbol.
        for symbol in tuple(self.armed | set(self.liquidation)):
            # Withdraw before asking the OMS to plan any action.
            if not self.feed_ready(symbol):
                # Require an explicit new operator action after recovery.
                self.feed_unsafe(symbol, "feed readiness lost")
        # Liquidation never reprices or tops up an unresolved existing clip.
        for symbol, policy in self.liquidation.items():
            # Retain unresolved terminal messages as well as executable orders.
            outstanding = [order for order in self.oms._orders.values() if order.symbol == symbol and (not order.state.is_terminal or order.has_message_in_flight)]
            # Wait for all old orders and late responses before issuing a fresh clip.
            if not outstanding:
                # A fresh clip is bounded by actual current inventory.
                self.oms.set_desired(policy.desired(symbol, self.oms.position(symbol)))
                # Subsequent cycles must retain this clip's actual remaining size.
                self.liquidation_started.add(symbol)
            # An active unwind clip must never be topped back up after a partial fill.
            elif symbol in self.liquidation_started:
                # This controller starts only one liquidation order at a time.
                live = [order for order in outstanding if not order.state.is_terminal]
                # Unanswered terminal messages keep the side flat until resolved.
                if not live:
                    # Retain the drain barrier without requesting another clip.
                    self.oms.set_desired(DesiredQuotes.flat(symbol))
                # One executable generation is the invariant of this policy.
                elif len(live) == 1:
                    # Express exactly the existing remainder at its authorized limit.
                    order = live[0]
                    # Match the remaining clip rather than restoring its original size.
                    intent = QuoteIntent(order.side, order.price_minor, order.leaves_quantity)
                    # No acquiring side is introduced during a partial liquidation.
                    self.oms.set_desired(DesiredQuotes(symbol, bid=intent if order.side is Side.BUY else None, ask=intent if order.side is Side.SELL else None))
                # Unexpected concurrent orders require a global reconciliation stop.
                else:
                    # Fail closed rather than guessing which order is the unwind.
                    self.stop("control_loop", "multiple liquidation orders", exchange_ms)
                    # stop() cleared the instruction map, so end iteration now.
                    break
        # Do not blindly retry a transport-ambiguous send.
        if self.send_failed or not self.transport_ready():
            # Reservations remain until an external recovery protocol resolves them.
            return []
        # This is the only action-generation path, including emergency cancellations.
        actions = self.oms.reconcile(exchange_ms, date, references)
        # Dispatch in the manager's cancellation-first order.
        for action in actions:
            # A transport exception means delivery may be ambiguous.
            try:
                # The eventual transport owns durable enqueue and wire serialization.
                self.send(action)
            # Never release exposure based on a local send failure alone.
            except Exception:
                # Block further dispatch until a new recovered controller is assembled.
                self.send_failed = True
                # Kill uses the same desired-state cancellation path.
                self.stop("control_loop", "ambiguous order transport outcome", exchange_ms)
                # Surface the failure so the host process cannot silently continue.
                raise
        # Return the dispatched actions for deterministic monitoring and tests.
        return actions

    # Operator shutdown completion must include inventory and unresolved message state.
    def shutdown_complete(self):
        # A stop request alone does not prove that anything was cancelled or flattened.
        return self.stopping and not self.send_failed and self.broker_reconciled() and self.oms.is_flat()
