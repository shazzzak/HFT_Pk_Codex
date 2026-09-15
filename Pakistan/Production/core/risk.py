"""The risk gateway: the only path from a trading decision to the wire.

ARCHITECTURAL RULE, and the reason this module exists at all: no component may
send an order without an approval from here. Not a debug path, not a manual
override, not a "just this once" flag. If an order can reach the exchange
without passing this gateway, then the kill switch is a suggestion rather than
a switch, and every limit below is decorative.

Each control is a separate class implementing one interface, so adding a
control is adding a class rather than editing a growing if-chain, and each one
can be unit-tested in isolation against its own rejection path.

THE CANCEL RULE. Every control here restricts what we ADD. None of them may
block a CANCEL. A control that can block a cancel is a control that can trap us
in a position -- the failure it causes is strictly worse than the one it
prevents. RiskGateway enforces this centrally so an individual check cannot get
it wrong.

WHY THESE CONTROLS, GIVEN THAT NOBODY IS MAKING US. The SECP concept paper of
30-05-2025 is not being pursued -- the leadership that sponsored it has changed
and electronic market making is not a current priority. So nothing here is
compliance. Every control below earns its place because it protects capital:

  KillSwitch            one action stops everything, when something is wrong
                        and we do not yet know what
  PriceBandCheck        catches a plausible-looking price computed from a stale
                        or corrupt book -- the failure an exchange band is too
                        wide to catch
  OrderValueCheck       bounds the cost of one bad number
  OrderQuantityCheck    same, in shares
  PositionLimitCheck    bounds inventory on the WORST case, not the current one
  MessageRateCheck      protects the session from a strategy that has started
                        oscillating, and respects whatever the broker enforces
  TradingWindowCheck    orders outside continuous trading get rejected or
                        behave in ways the backtest never modelled
  OrderToTradeRatioCheck  queue position, not compliance -- see its docstring

The concept-paper section is noted against each one anyway, so that if the
framework is ever revived the mapping is a filing exercise rather than a build:
s8.1 price limits, s8.2 order value, s8.3 volume, s8.4 burst, s7 orders per
second and order timing, s6 order-to-trade, s11 kill switch, s12 audit trail
(every RiskDecision is a loggable record, approvals included).
"""
# value objects
from dataclasses import dataclass, field
# the abstract base machinery
from abc import ABC, abstractmethod
# a bounded deque is the right shape for a sliding time window
from collections import deque
# typing only
from typing import Callable, Deque, Optional, Sequence
# the shared domain
from core.model import Action, PlaceOrder, Side
# venue rules and price bands
from core.venue import PriceBand, Venue


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskDecision:
    """The outcome of evaluating one action.

    Carries the CHECK NAME and a human-readable reason whether it passed or
    failed, because the audit trail (s12) has to answer "why was this order
    sent", not only "why was this order blocked".
    """
    # did the action pass
    allowed: bool
    # which control produced this decision
    check: str
    # why, in words a compliance reviewer can read
    reason: str = ""
    # the individual control decisions behind a gateway-level verdict.
    # WHY THIS EXISTS: without it an approval records only "all checks passed",
    # which cannot answer why an order was sent -- only that nothing objected.
    # Controls that measure rather than block (the order-to-trade ratio) put
    # their reading in their own reason, and it would be lost otherwise.
    details: tuple = ()

    @classmethod
    def allow(cls, check: str, reason: str = "",
              details: tuple = ()) -> "RiskDecision":
        """A passing decision."""
        # approval carries the check name so a full trace can be assembled
        return cls(allowed=True, check=check, reason=reason, details=details)

    @classmethod
    def reject(cls, check: str, reason: str) -> "RiskDecision":
        """A blocking decision. A reason is mandatory."""
        # an unexplained rejection is useless at 10am when quoting stops
        if not reason:
            raise ValueError("a rejection must carry a reason")
        # the blocking decision
        return cls(allowed=False, check=check, reason=reason)


@dataclass
class RiskContext:
    """Everything a control may need to judge one action.

    Passed by the caller rather than read from globals, so every check is a
    pure function of its inputs and can be tested without a running engine.
    """
    # trading date, 'YYYY-MM-DD', for the session lookup
    date: str
    # exchange time in milliseconds
    timestamp_ms: int
    # our current signed position in this symbol, in shares
    position: int = 0
    # value of everything we already have resting, in minor units
    working_notional_minor: int = 0
    # a reference price for the symbol in minor units, usually the mid
    reference_price_minor: Optional[int] = None


# ---------------------------------------------------------------------------
# The kill switch
# ---------------------------------------------------------------------------

@dataclass
class KillSwitch:
    """SECP s11: halt new order submission and cancel every resting order.

    This object holds only the STATE. The cancelling is done by the order
    manager, which reacts to the tripped state by setting every symbol's
    desired quotes to flat and letting its normal diff logic emit the cancels.

    That indirection is deliberate. A kill switch with its own dedicated
    cancel-everything code path is a path that is never exercised until the day
    it matters, and on that day it is the least-tested code in the system.
    Routing the emergency through the everyday mechanism means the everyday
    tests cover it.
    """
    # is the switch currently tripped
    tripped: bool = False
    # why it was tripped
    reason: Optional[str] = None
    # who or what tripped it: an operator name, or the control that fired
    tripped_by: Optional[str] = None
    # when, in exchange milliseconds
    tripped_at_ms: Optional[int] = None
    # callbacks to notify the moment it trips
    _listeners: list = field(default_factory=list, repr=False)

    def trip(self, reason: str, by: str, timestamp_ms: int) -> None:
        """Trip the switch. Idempotent: the FIRST reason is the one kept.

        Keeping the first reason matters because a trip usually causes a
        cascade of secondary failures, and the secondary ones would otherwise
        overwrite the cause with a symptom.
        """
        # already tripped: keep the original cause and do nothing else
        if self.tripped:
            return
        # record the cause
        self.tripped = True
        self.reason = reason
        self.tripped_by = by
        self.tripped_at_ms = timestamp_ms
        # tell everyone who asked to be told, so the OMS can go flat at once
        for listener in self._listeners:
            listener(self)

    def reset(self, by: str) -> None:
        """Re-arm after a trip. Deliberately requires a named human.

        There is no automatic reset and there should not be one: whatever
        tripped the switch has to be understood before quoting resumes, and an
        engine that can un-trip itself will do so in the middle of the event
        that tripped it.
        """
        # clear the state
        self.tripped = False
        self.reason = None
        self.tripped_by = None
        self.tripped_at_ms = None

    def add_listener(self, callback: Callable[["KillSwitch"], None]) -> None:
        """Register something to be notified the moment the switch trips."""
        # the order manager registers here to flatten its desired state
        self._listeners.append(callback)


# ---------------------------------------------------------------------------
# The control interface
# ---------------------------------------------------------------------------

class RiskCheck(ABC):
    """One pre-trade control."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in decisions and in the audit log."""

    @abstractmethod
    def evaluate(self, action: Action, ctx: RiskContext) -> RiskDecision:
        """Judge one action. Called only for order-adding actions."""

    def on_approved(self, action: Action, ctx: RiskContext) -> None:
        """Told that an action was approved and will be sent.

        Stateful controls -- rate limits, ratios -- update their counters here.
        Stateless ones ignore it, which is why this is a no-op by default
        rather than an abstract method.
        """
        # stateless controls need to do nothing
        return None


# ---------------------------------------------------------------------------
# The controls
# ---------------------------------------------------------------------------

class KillSwitchCheck(RiskCheck):
    """SECP s11. Blocks every new order while the switch is tripped."""

    def __init__(self, switch: KillSwitch):
        # the shared switch state
        self._switch = switch

    @property
    def name(self) -> str:
        # identifier for decisions and logs
        return "kill_switch"

    def evaluate(self, action: Action, ctx: RiskContext) -> RiskDecision:
        # while tripped, nothing new may be sent
        if self._switch.tripped:
            return RiskDecision.reject(
                self.name, f"kill switch tripped by {self._switch.tripped_by}: "
                           f"{self._switch.reason}")
        # armed and not tripped: no objection
        return RiskDecision.allow(self.name)


class TradingWindowCheck(RiskCheck):
    """SECP s7. No algorithmic orders outside continuous trading.

    The venue owns the session definition, including any mid-day break, so this
    control is venue-agnostic.
    """

    def __init__(self, venue: Venue):
        # the venue whose calendar decides the answer
        self._venue = venue

    @property
    def name(self) -> str:
        # identifier
        return "trading_window"

    def evaluate(self, action: Action, ctx: RiskContext) -> RiskDecision:
        # ask the venue whether continuous trading is open right now
        if not self._venue.is_continuous(ctx.date, ctx.timestamp_ms):
            return RiskDecision.reject(
                self.name, f"{ctx.timestamp_ms} is outside continuous trading "
                           f"on {ctx.date}")
        # inside the session
        return RiskDecision.allow(self.name)


class PriceBandCheck(RiskCheck):
    """SECP s8.1. Two bands, and the tighter one wins.

    The EXCHANGE band is the hard limit -- an order outside it is rejected at
    the gateway and counts against us. The HOUSE band is a percentage around
    our own reference price, and it is the one that catches the failure the
    exchange band cannot: a correct-looking price computed from a stale or
    corrupt book. The exchange band is wide enough that a badly wrong price can
    sit comfortably inside it.
    """

    def __init__(self, venue: Venue, house_band_pct: float):
        # for the published exchange limits
        self._venue = venue
        # our own tolerance around the reference price, in percent
        self._house_pct = house_band_pct
        # a non-positive band would block everything
        if house_band_pct <= 0:
            raise ValueError("house_band_pct must be positive")

    @property
    def name(self) -> str:
        # identifier
        return "price_band"

    def evaluate(self, action: Action, ctx: RiskContext) -> RiskDecision:
        # only order placements carry a price
        if not isinstance(action, PlaceOrder):
            return RiskDecision.allow(self.name)
        # the exchange's published limits, if it has published any
        band: Optional[PriceBand] = self._venue.price_band(action.symbol)
        # outside the exchange band the order would be rejected on arrival
        if band is not None and not band.contains(action.price_minor):
            return RiskDecision.reject(
                self.name, f"{action.price_minor} outside exchange band "
                           f"[{band.lower_minor}, {band.upper_minor}]")
        # without a reference price the house band cannot be evaluated
        if ctx.reference_price_minor is None:
            # NO REFERENCE IS A REJECTION, NOT A PASS. Quoting a symbol whose
            # own mid we cannot compute means the book is not trustworthy, and
            # that is exactly when a wrong price gets sent.
            return RiskDecision.reject(
                self.name, "no reference price available for the house band")
        # how far the order sits from our own reference, in percent
        distance = abs(action.price_minor - ctx.reference_price_minor)
        # the width the house band permits at this reference
        allowed = ctx.reference_price_minor * self._house_pct / 100.0
        # further than the house band tolerates
        if distance > allowed:
            return RiskDecision.reject(
                self.name, f"{action.price_minor} is "
                           f"{100.0 * distance / ctx.reference_price_minor:.2f}% "
                           f"from reference {ctx.reference_price_minor}, house "
                           f"band is {self._house_pct:.2f}%")
        # inside both bands
        return RiskDecision.allow(self.name)


class OrderValueCheck(RiskCheck):
    """SECP s8.2. Cap the value of any single order."""

    def __init__(self, max_order_value_minor: int):
        # the per-order ceiling in minor units
        self._max = max_order_value_minor
        # a non-positive cap would block everything
        if max_order_value_minor <= 0:
            raise ValueError("max_order_value_minor must be positive")

    @property
    def name(self) -> str:
        # identifier
        return "order_value"

    def evaluate(self, action: Action, ctx: RiskContext) -> RiskDecision:
        # only placements have a value
        if not isinstance(action, PlaceOrder):
            return RiskDecision.allow(self.name)
        # price x size against the ceiling
        if action.notional_minor > self._max:
            return RiskDecision.reject(
                self.name, f"order value {action.notional_minor} exceeds "
                           f"limit {self._max}")
        # within the cap
        return RiskDecision.allow(self.name)


class OrderQuantityCheck(RiskCheck):
    """SECP s8.3. Keep an unusually large order out of the book."""

    def __init__(self, max_quantity: int):
        # the per-order share ceiling
        self._max = max_quantity
        # a non-positive cap would block everything
        if max_quantity <= 0:
            raise ValueError("max_quantity must be positive")

    @property
    def name(self) -> str:
        # identifier
        return "order_quantity"

    def evaluate(self, action: Action, ctx: RiskContext) -> RiskDecision:
        # only placements have a size
        if not isinstance(action, PlaceOrder):
            return RiskDecision.allow(self.name)
        # compare against the ceiling
        if action.quantity > self._max:
            return RiskDecision.reject(
                self.name, f"quantity {action.quantity} exceeds limit "
                           f"{self._max}")
        # within the cap
        return RiskDecision.allow(self.name)


class PositionLimitCheck(RiskCheck):
    """House control. Refuse an order that would push inventory past a cap.

    Judged on the WORST CASE -- the order filling in full on top of the
    position we already hold -- not on the current position. Checking the
    current position permits a set of individually-legal orders whose combined
    fills breach the limit, which is the ordinary way a position limit fails.
    """

    def __init__(self, max_absolute_position: int):
        # the inventory ceiling, absolute value, in shares
        self._max = max_absolute_position
        # a non-positive cap would block everything
        if max_absolute_position <= 0:
            raise ValueError("max_absolute_position must be positive")

    @property
    def name(self) -> str:
        # identifier
        return "position_limit"

    def evaluate(self, action: Action, ctx: RiskContext) -> RiskDecision:
        # only placements can change inventory
        if not isinstance(action, PlaceOrder):
            return RiskDecision.allow(self.name)
        # where the position lands if this order fills entirely
        worst_case = ctx.position + action.side.sign * action.quantity
        # breach of the ceiling in either direction
        if abs(worst_case) > self._max:
            return RiskDecision.reject(
                self.name, f"position would reach {worst_case} from "
                           f"{ctx.position}, limit is +/-{self._max}")
        # inside the cap even if fully filled
        return RiskDecision.allow(self.name)


class MessageRateCheck(RiskCheck):
    """SECP s7 (orders per second) and s8.4 (burst control).

    Two windows, because they answer different questions. The per-second cap is
    the exchange's steady-state rule. The burst window is shorter and catches a
    strategy that has started oscillating -- repricing back and forth on every
    event -- which produces a rate that looks acceptable averaged over a second
    and is pathological at 100ms.

    CANCELS COUNT BUT ARE NEVER BLOCKED. They consume the exchange's message
    budget, so ignoring them would under-count the true rate; but blocking one
    to stay under a limit would leave an order resting that we decided to
    remove. The gateway guarantees cancels are never passed here for judgement;
    they arrive only through on_approved, which is exactly the asymmetry wanted.
    """

    def __init__(self, max_per_second: int, burst_window_ms: int = 100,
                 max_per_burst: Optional[int] = None):
        # the steady-state ceiling
        self._max_per_second = max_per_second
        # the short window, in milliseconds
        self._burst_window_ms = burst_window_ms
        # the short-window ceiling; default keeps bursts proportionate
        self._max_per_burst = (max_per_burst if max_per_burst is not None
                               else max(1, max_per_second * burst_window_ms
                                        // 1000))
        # a non-positive rate would block everything
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        # timestamps of messages sent, oldest first
        self._sent_ms: Deque[int] = deque()

    @property
    def name(self) -> str:
        # identifier
        return "message_rate"

    def _prune(self, now_ms: int) -> None:
        """Drop timestamps older than the longer of the two windows."""
        # anything older than one second cannot affect either window
        cutoff = now_ms - 1000
        # discard from the front, which is the oldest end
        while self._sent_ms and self._sent_ms[0] <= cutoff:
            self._sent_ms.popleft()

    def evaluate(self, action: Action, ctx: RiskContext) -> RiskDecision:
        # forget anything that has aged out of both windows
        self._prune(ctx.timestamp_ms)
        # messages in the trailing second
        in_second = len(self._sent_ms)
        # the steady-state rule
        if in_second >= self._max_per_second:
            return RiskDecision.reject(
                self.name, f"{in_second} messages in the last 1000ms, limit "
                           f"is {self._max_per_second}")
        # messages inside the short burst window
        burst_cutoff = ctx.timestamp_ms - self._burst_window_ms
        # count only the recent tail
        in_burst = sum(1 for t in self._sent_ms if t > burst_cutoff)
        # the burst rule
        if in_burst >= self._max_per_burst:
            return RiskDecision.reject(
                self.name, f"{in_burst} messages in the last "
                           f"{self._burst_window_ms}ms, burst limit is "
                           f"{self._max_per_burst}")
        # under both ceilings
        return RiskDecision.allow(self.name)

    def on_approved(self, action: Action, ctx: RiskContext) -> None:
        # every approved message counts, cancels included
        self._sent_ms.append(ctx.timestamp_ms)


class OrderToTradeRatioCheck(RiskCheck):
    """Messages per execution. The reason is QUEUE POSITION, not compliance.

    With no regulator penalising a high ratio, the cost of churn is paid to the
    market instead. Every cancel-and-repost surrenders queue priority at that
    price and rejoins at the back. For a strategy whose edge is measured as
    spread CAPTURE -- which is to say, as getting filled while resting -- queue
    position is close to the whole game, and a quote that reprices on every
    book event may be destroying the thing it is trying to earn.

    That cost is invisible in a backtest whose fill model does not simulate
    queue position, which is exactly the situation we are in until the fill
    logging work is done. So this number is a HEALTH METRIC first and a guard
    second: it says how hard the strategy is churning, and a high reading is a
    reason to look at the quoting cadence rather than to tighten this limit.

    The secondary reason is mundane and still real: a broker or exchange may
    charge per message or throttle a chatty session. That ceiling is unknown
    for PSX -- ask the broker.

    For the record, since the design anticipates the framework returning: this
    is SECP concept paper s6, where India penalises a high ratio with a
    15-minute trading ban and Borsa Istanbul charges per message above 5:1.

    WARM-UP MATTERS. Early in the day the ratio is meaningless -- the first
    order before any fill gives a ratio of infinity. The control therefore does
    nothing until a minimum number of messages has been sent, so it cannot
    strangle quoting at the open, which is precisely when it would do the most
    damage and have the least justification.
    """

    def __init__(self, max_ratio: float, min_messages_before_binding: int = 100,
                 enforce: bool = False):
        # the ceiling on messages per execution, applied only when enforcing
        self._max_ratio = max_ratio
        # how many messages must be sent before the control can reject
        self._warmup = min_messages_before_binding
        # MEASURE ALWAYS, ENFORCE ON REQUEST. Default off: nothing external
        # requires a ratio cap today, and a limit that can stop quoting should
        # not be switched on before we know what our normal ratio even is.
        # The counting runs regardless, so the number is on the record from the
        # first day and turning the limit on later is a decision made against
        # data rather than against a guess.
        self._enforce = enforce
        # a non-positive ratio would block everything once enforcing
        if max_ratio <= 0:
            raise ValueError("max_ratio must be positive")
        # total messages sent today
        self._messages = 0
        # total executions received today
        self._trades = 0

    @property
    def enforce(self) -> bool:
        """Is the ratio currently a limit, or only a measurement?"""
        # readable so an operator console can show which mode is live
        return self._enforce

    @enforce.setter
    def enforce(self, value: bool) -> None:
        """Turn the limit on or off, including while the engine is running.

        Settable at runtime so it can be driven from the hot-reloaded config
        rather than needing a restart -- the same mechanism that already
        switches a symbol between quoting configurations intraday.
        """
        # flip the mode; the counters are untouched either way
        self._enforce = bool(value)

    @property
    def messages(self) -> int:
        """Messages sent today. Exposed so the monitor can chart it."""
        # the numerator
        return self._messages

    @property
    def trades(self) -> int:
        """Executions today. Exposed so the monitor can chart it."""
        # the denominator
        return self._trades

    @property
    def name(self) -> str:
        # identifier
        return "order_to_trade_ratio"

    @property
    def ratio(self) -> float:
        """Current messages-per-trade. Infinite before the first execution."""
        # no executions yet: the ratio is undefined, reported as infinity
        if self._trades == 0:
            return float("inf") if self._messages else 0.0
        # messages divided by executions
        return self._messages / self._trades

    def evaluate(self, action: Action, ctx: RiskContext) -> RiskDecision:
        # MEASURE-ONLY MODE. The ratio still goes into the decision reason, so
        # it lands in the audit log on every action and the number is available
        # without a separate reporting path.
        if not self._enforce:
            return RiskDecision.allow(
                self.name, f"measure-only: ratio {self.ratio:.1f} "
                           f"({self._messages} messages, {self._trades} trades)")
        # too early in the day for the ratio to mean anything
        if self._messages < self._warmup:
            return RiskDecision.allow(
                self.name, f"warm-up: {self._messages}/{self._warmup} messages")
        # the current ratio against the ceiling
        if self.ratio > self._max_ratio:
            return RiskDecision.reject(
                self.name, f"order-to-trade ratio {self.ratio:.1f} exceeds "
                           f"{self._max_ratio:.1f} ({self._messages} messages, "
                           f"{self._trades} trades)")
        # under the ceiling
        return RiskDecision.allow(self.name)

    def on_approved(self, action: Action, ctx: RiskContext) -> None:
        # every message sent counts toward the numerator, cancels included
        self._messages += 1

    def on_trade(self) -> None:
        """Told that an execution arrived. Feeds the denominator."""
        # one more execution
        self._trades += 1


# ---------------------------------------------------------------------------
# The gateway
# ---------------------------------------------------------------------------

class RiskGateway:
    """Runs every control and decides whether one action may be sent."""

    def __init__(self, checks: Sequence[RiskCheck],
                 on_decision: Optional[Callable[[Action, RiskDecision],
                                                None]] = None):
        # the controls, evaluated in the order given
        self._checks = list(checks)
        # where every decision is recorded; this is the audit-trail hook (s12)
        self._on_decision = on_decision
        # an empty gateway would approve everything, which is never intended
        if not self._checks:
            raise ValueError("a gateway with no checks approves everything")

    def authorise(self, action: Action, ctx: RiskContext) -> RiskDecision:
        """Approve or reject one action.

        A CANCEL IS ALWAYS APPROVED. This is enforced here, once, rather than
        trusted to every individual check, because a single control that
        forgets the rule could trap the book in a position. Cancels are still
        reported to the stateful controls so their counters stay honest.
        """
        # removing exposure is never blocked, whatever the state of the system
        if action.is_cancel:
            # an explicit, logged approval rather than a silent bypass
            decision = RiskDecision.allow(
                "gateway", "cancels are never blocked")
            # the counters must still see it: it consumes exchange bandwidth
            self._record(action, ctx, decision)
            # approved
            return decision
        # keep each control's own verdict, so the approval can carry them
        results = []
        # evaluate each control in turn, stopping at the first rejection
        for check in self._checks:
            # ask this control
            decision = check.evaluate(action, ctx)
            # a rejection ends the evaluation; no counter is advanced
            if not decision.allowed:
                # log it, then refuse
                self._emit(action, decision)
                return decision
            # remember the passing verdict and its reason
            results.append(decision)
        # every control passed, and the approval carries what each one said
        decision = RiskDecision.allow("gateway", "all checks passed",
                                      details=tuple(results))
        # advance the stateful controls and log
        self._record(action, ctx, decision)
        # approved
        return decision

    def _record(self, action: Action, ctx: RiskContext,
                decision: RiskDecision) -> None:
        """Tell every stateful control that an action is going out, and log."""
        # counters advance only for actions that are actually sent
        for check in self._checks:
            check.on_approved(action, ctx)
        # write the decision to the audit trail
        self._emit(action, decision)

    def _emit(self, action: Action, decision: RiskDecision) -> None:
        """Hand one decision to the audit sink, if there is one."""
        # no sink configured: nothing to do
        if self._on_decision is None:
            return
        # record the decision against the action that produced it
        self._on_decision(action, decision)

    def on_trade(self) -> None:
        """Tell the controls that an execution arrived."""
        # only the ratio control cares, but it is found by capability not name
        for check in self._checks:
            # advance any control that tracks executions
            if hasattr(check, "on_trade"):
                check.on_trade()
