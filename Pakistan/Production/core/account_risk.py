"""Single-account conservative cash risk around the durable OMS.

Values are integer paisa/shares. This is not a PSX margin or settlement model.
The caller must provide complete account ownership and authoritative funding.
"""
# Freeze the explicit risk contract and serialize it for durable verification.
from dataclasses import dataclass, asdict
# Return detached monitoring state rather than leaking mutable dictionaries.
from copy import deepcopy
# Validate trading dates rather than comparing unvalidated strings.
from datetime import date as CalendarDate
# Use a process-local monotonic clock for quote freshness.
import time
# Extend the existing atomic OMS/account checkpoint boundary.
from core.durable_runtime import DurableManager
# Inspect every order request, including amendments.
from core.model import OrderRequest, ReplaceOrder, Side
# Use the existing mandatory risk-gateway interface.
from core.risk import RiskCheck, RiskDecision


# Reject booleans and fractional values in all integer financial settings.
def integer(value, name, minimum=0):
    # Numeric truthiness must never turn malformed data into an account limit.
    if type(value) is not int or value < minimum:
        # Explain which configuration or market-data field is invalid.
        raise ValueError(f"{name} must be an integer >= {minimum}")
    # Return the validated integer for expressions that need its value.
    return value


# No implicit financial defaults: callers must declare each test or approved limit.
@dataclass(frozen=True)
class AccountLimits:
    # Sum of per-symbol maximum possible long/short notional endpoints.
    max_gross_minor: int
    # Aggregate long endpoint across all symbols, without short offsets.
    max_long_minor: int
    # Aggregate short endpoint across all symbols, without long offsets.
    max_short_minor: int
    # House ceiling on the cash commitment envelope for the trading day.
    max_cash_commitment_minor: int
    # A loss equal to this amount trips the daily stop.
    max_daily_loss_minor: int
    # Quotes older than this process-local age cannot value risk.
    max_mark_age_ms: int
    # Conservative configurable fee allowance on outstanding order notional.
    fee_reserve_bps: int
    # Additional fee allowance per outstanding generation, in integer paisa.
    fee_reserve_fixed_minor: int

    # Validate every explicit limit at construction time.
    def __post_init__(self):
        # Most financial ceilings must be positive; zero shorts is a valid policy.
        for name, value in asdict(self).items():
            # Fee budgets and the short ceiling may deliberately be zero.
            minimum = 0 if name in ("max_short_minor", "fee_reserve_bps", "fee_reserve_fixed_minor") else 1
            # Prevent silently disabled controls through invalid configuration.
            integer(value, name, minimum)


# Keep market valuation and its local receive time in one immutable record.
@dataclass(frozen=True)
class AccountMark:
    # Price available to liquidate long inventory.
    bid_minor: int
    # Price available to cover short inventory and conservatively value exposure.
    ask_minor: int
    # Time received in this process's monotonic millisecond domain.
    received_ms: int

    # Fail before accepting corrupt or crossed valuation data.
    def __post_init__(self):
        # Bid and ask prices must use whole positive minor units.
        integer(self.bid_minor, "bid_minor", 1)
        # Ask prices obey the same integer price contract.
        integer(self.ask_minor, "ask_minor", 1)
        # Local receive time must be a nonnegative monotonic value.
        integer(self.received_ms, "received_ms")
        # A crossed mark cannot be used to manufacture account equity.
        if self.bid_minor > self.ask_minor:
            # Market-data recovery must resolve the inconsistency.
            raise ValueError("crossed account valuation mark")


# Missing or stale valuation is a control failure, not zero risk.
class RiskUnavailable(ValueError):
    # Distinguish unavailable inputs from an ordinary limit rejection.
    pass


# Evaluate aggregate exposure against the actual sequentially updated OMS state.
class AccountRiskCheck(RiskCheck):
    # Bind one check to one durable account owner.
    def __init__(self, manager):
        # Earlier approved actions in the same batch are already in this manager.
        self.manager = manager

    # Name the check in existing OMS rejection/audit records.
    @property
    # Keep decisions identifiable independently of symbol risk checks.
    def name(self):
        # This identifier is part of the audit contract.
        return "account_risk"

    # Every place and amendment is evaluated before the OMS marks it sent.
    def evaluate(self, action, ctx):
        # Gateway cancels bypass checks; retain that invariant in direct calls too.
        if not isinstance(action, OrderRequest):
            # Removing exposure is never blocked by account limits.
            return RiskDecision.allow(self.name)
        # Core order fields must be valid before accounting calculations.
        try:
            # Reject fractional or zero shares before worst-case multiplication.
            integer(action.quantity, "order quantity", 1)
            # Reject fractional or zero limit prices as well.
            integer(action.price_minor, "order price", 1)
            # Require the funded trading-day contract to match this OMS cycle.
            if self.manager.account["day"] != ctx.date:
                # A new date must never reset the daily loss limit automatically.
                raise RiskUnavailable("trading day not initialized or differs")
            # A persistent daily stop cannot be bypassed by resetting only the kill switch.
            if self.manager.account["loss_latched"] or self.manager.oms._kill.tripped:
                # Keep the account disarmed until an explicit valid recovery procedure.
                raise RiskUnavailable("account halt remains latched")
            # Compare aggregate endpoints including this proposed generation.
            metrics = self.manager.measure(action)
            # Retain all failing controls for a meaningful rejection explanation.
            breaches = self.manager.breaches(metrics)
            # Per-symbol short authorization is independent of account-wide short notional.
            if metrics["unauthorized_shorts"]:
                # Unfilled buys cannot be used to justify an uncovered sell.
                breaches.append("short availability: " + ",".join(metrics["unauthorized_shorts"]))
            # A single failed account control rejects the proposed action.
            if breaches:
                # Include concrete metrics for audit and debugging.
                return RiskDecision.reject(self.name, "; ".join(breaches))
            # Include the assessed aggregate endpoints in the passing decision.
            return RiskDecision.allow(self.name, f"gross={metrics['gross_minor']} long={metrics['long_minor']} short={metrics['short_minor']} cash_commitment={metrics['cash_commitment_minor']}")
        # Missing marks or invalid units must fail closed, not crash the gateway.
        except (ValueError, KeyError) as error:
            # Existing OMS audit infrastructure records the rejected proposal.
            return RiskDecision.reject(self.name, str(error))

    # Preserve amendment prices before the next action in the same batch is checked.
    def on_approved(self, action, ctx):
        # The existing OMS records replacement quantity but not proposed price.
        if isinstance(action, ReplaceOrder):
            # Persist the new price with the account checkpoint after batch planning.
            self.manager.account["replacement_prices"][action.orig_cl_ord_id] = action.price_minor


# Add account controls without changing the strategy or existing OMS diff rules.
class AccountRiskManager(DurableManager):
    # Require explicit limits and a testable monotonic clock.
    def __init__(self, oms, store, limits, clock_ms=None):
        # Refuse a loosely typed or partially populated limit configuration.
        if not isinstance(limits, AccountLimits):
            # Every limit must have passed its domain validation.
            raise TypeError("AccountLimits required")
        # Keep immutable limits available to both checks and monitoring.
        self.limits = limits
        # All freshness decisions use receive-time monotonic milliseconds.
        self.clock_ms = clock_ms or (lambda: time.monotonic_ns() // 1000000)
        # Marks are intentionally never trusted across a process restart.
        self.marks = {}
        # A clock reversal invalidates valuation in this process.
        self.last_clock_ms = None
        # Read the prior checkpoint before base initialization performs any writes.
        saved = store.load()
        # Recovery requires the exact account-risk contract used before the crash.
        if saved is not None:
            # Never reinterpret a recovery-only database as a funded risk account.
            if "account_risk" not in saved or saved["account_risk"]["limits"] != asdict(limits):
                # Upgrading existing account ledgers requires a reviewed migration.
                raise ValueError("account-risk checkpoint missing or limits changed")
            # Restore loss baseline, spent cash, short caps and amendment prices together.
            self.account = saved["account_risk"]
        # New accounts begin unfunded and cannot quote.
        else:
            # Funding and the trading-day baseline require a separate named instruction.
            self.account = {"limits": asdict(limits), "day": None, "opening_equity_minor": None, "funding_minor": 0, "buy_spent_minor": 0, "positive_fees_minor": 0, "loss_latched": False, "halt_reason": None, "short_caps": {}, "replacement_prices": {}, "latest_metrics": None}
        # Restore the real order/cash ledger before attaching the account gateway.
        super().__init__(oms, store)
        # Install the account control inside the same gateway as existing symbol checks.
        oms._gateway._checks.insert(0, AccountRiskCheck(self))

    # Include financial risk state in every OMS transaction, including fills.
    def _state(self):
        # Reuse the recovery schema's exact order, cash and pending-message snapshot.
        state = super()._state()
        # The state is serialized immediately by the owning store transaction.
        state["account_risk"] = self.account
        # Return a single atomic account/OMS checkpoint.
        return state

    # Fold spend and daily-loss evaluation into the same commit as each fill.
    def _commit(self, kind, payload):
        # Only an applied nonduplicate fill advances spent-cash counters.
        if kind == "report" and payload["kind"] == "fill":
            # The base runtime has already validated and applied this execution.
            fill = payload["payload"]
            # Do not replenish buying power from sell proceeds or rebates.
            if fill["side"] == "BUY":
                # Charge executed buy notional once, alongside cash and position.
                self.account["buy_spent_minor"] += fill["price_minor"] * fill["quantity"]
            # Positive fees consume funding; rebates do not manufacture new headroom.
            self.account["positive_fees_minor"] += max(0, fill.get("fee_minor", 0))
        # Lifecycle reports release replacement pricing only with the OMS reservation.
        if kind == "report":
            # Keep only amendment generations which still represent contingent exposure.
            self.account["replacement_prices"] = {key: value for key, value in self.account["replacement_prices"].items() if key in self.oms._replacement_reservations}
        # Persist current post-mutation metrics rather than the previous planning snapshot.
        if kind in ("report", "reconcile"):
            # Fills, acknowledgements and complete batches can change actual account exposure.
            self._enforce()
        # Commit risk, executions, cash, inventory and kill state together.
        super()._commit(kind, payload)

    # Validate the local time domain on every evaluation, not just market messages.
    def _now(self):
        # Call the injected production or deterministic test clock.
        now = integer(self.clock_ms(), "monotonic clock")
        # A backwards clock makes all received timestamps untrustworthy.
        if self.last_clock_ms is not None and now < self.last_clock_ms:
            # Retain the last good time so repeated evaluations remain blocked.
            raise RiskUnavailable("monotonic clock moved backwards")
        # Remember the latest accepted local reading.
        self.last_clock_ms = now
        # Return current process-local receive-time milliseconds.
        return now

    # Require fresh valuation for every symbol that can contribute account exposure.
    def _mark(self, symbol, now):
        # A missing book is unknown value, not a zero-price asset.
        mark = self.marks.get(symbol)
        # Freshness uses the same process-local timebase as receipt.
        if mark is None or mark.received_ms > now or now - mark.received_ms > self.limits.max_mark_age_ms:
            # Fail the entire aggregate valuation when even one exposed symbol is unknown.
            raise RiskUnavailable(f"missing, future or stale mark: {symbol}")
        # Return the complete bid/ask observation.
        return mark

    # Enumerate old and proposed generations without offsetting opposite sides.
    def _generations(self, proposed=None):
        # Order map contains each original generation exactly once, unlike the alias map.
        for order in self.oms._orders.values():
            # Terminal messages still block recovery but no longer have executable leaves.
            if not order.state.is_terminal:
                # Pending cancels, pending sends and suspended orders all retain liability.
                yield order.symbol, order.side, order.leaves_quantity, order.price_minor
                # A pending remaining-size amendment can follow fills of the old generation.
                quantity = self.oms._replacement_reservations.get(order.cl_ord_id, 0)
                # Proposed size needs its actual new price, not the old price.
                if quantity:
                    # Missing amendment pricing is a state inconsistency, never a zero reserve.
                    if order.cl_ord_id not in self.account["replacement_prices"]:
                        # Require recovery rather than undercounting amendment cost.
                        raise RiskUnavailable("missing replacement price")
                    # Reserve the old and proposed generations independently until confirmation.
                    yield order.symbol, order.side, quantity, self.account["replacement_prices"][order.cl_ord_id]
        # The gateway evaluates a candidate before it becomes part of the OMS state.
        if proposed is not None:
            # A replacement adds a contingent generation; it does not remove the old one yet.
            yield proposed.symbol, proposed.side, proposed.quantity, proposed.price_minor

    # Produce conservative aggregate risk and liquidation-marked daily P&L.
    def measure(self, proposed=None):
        # Risk methods belong to the same event-loop owner as order processing.
        self._guard()
        # A missing day baseline cannot be replaced with an assumed zero loss.
        if self.account["day"] is None:
            # The account starts unfunded even if an empty ledger reconciled.
            raise RiskUnavailable("account trading day is not initialized")
        # Read the clock once so every symbol shares one valuation instant.
        now = self._now()
        # Keep a detached list because several aggregate controls inspect each generation.
        generations = list(self._generations(proposed))
        # Positions and every potential fill contribute to the required mark universe.
        symbols = {s for s, q in self.oms._position.items() if q} | {item[0] for item in generations}
        # Initialize exact integer totals without any cross-symbol directional offset.
        gross, longs, shorts, buy_reserve, fee_reserve = 0, 0, 0, 0, 0
        # Start account equity from actual cash, which already includes execution fees.
        equity = self.cash_minor
        # Retain symbol endpoints for audit and short-availability checks.
        endpoints, unauthorized = {}, []
        # Group generations once rather than repeatedly scanning the entire order history.
        grouped = {symbol: [] for symbol in symbols}
        # Preserve each old/proposed generation for fee and cash reservation.
        for generation in generations:
            # The symbol is the first item in the normalized tuple.
            grouped[generation[0]].append(generation)
        # Sum each symbol's independently reachable inventory endpoints.
        for symbol in sorted(symbols):
            # All exposed symbols require fresh two-sided marks.
            mark = self._mark(symbol, now)
            # Filled inventory is the starting point, not the whole risk exposure.
            position = self.oms.position(symbol)
            # Only actual inventory enters marked equity; pending orders are not profit.
            equity += position * (mark.bid_minor if position >= 0 else mark.ask_minor)
            # Each side can fill without any opposite-side execution.
            buys, sells, price = 0, 0, mark.ask_minor
            # Count pending cancellations and amendment races conservatively.
            for _, side, quantity, limit in grouped[symbol]:
                # Use at least the higher of current ask and any committed order limit.
                price = max(price, limit)
                # Pending buys consume actual cash at their limit, not a favorable current bid.
                if side is Side.BUY:
                    # Worst-case cash assumes the full remaining quantity executes.
                    buy_reserve += quantity * limit
                    # Opposite-side sells are not guaranteed to reduce this endpoint.
                    buys += quantity
                # Pending sells widen the independent short endpoint.
                else:
                    # Unfilled buys do not create stock available to sell.
                    sells += quantity
                # Round fee allowances upward so fractional paisa never reduce reserve.
                fee_reserve += (quantity * limit * self.limits.fee_reserve_bps + 9999) // 10000 + self.limits.fee_reserve_fixed_minor
            # Positive upper endpoint is maximum possible long inventory.
            high = max(0, position + buys)
            # Negative lower endpoint is maximum possible short inventory.
            low = max(0, sells - position)
            # Long and short account caps are assessed independently.
            longs += high * price
            # No cross-symbol long inventory offsets this short risk.
            shorts += low * price
            # A symbol cannot finish simultaneously at both endpoints.
            gross += max(high, low) * price
            # Short inventory requires an explicit per-symbol authorized quantity.
            if low > self.account["short_caps"].get(symbol, 0):
                # Aggregate short notional permission alone does not establish borrow.
                unauthorized.append(symbol)
            # Preserve exact shares, valuation price and endpoints for diagnostics.
            endpoints[symbol] = {"position": position, "buys": buys, "sells": sells, "long_shares": high, "short_shares": low, "valuation_minor": price}
        # Funding is a cash commitment envelope, not a claim about PSX margin.
        commitment = self.account["buy_spent_minor"] + self.account["positive_fees_minor"] + buy_reserve + fee_reserve
        # Report the full risk calculation so a rejection can be reconstructed.
        return {"gross_minor": gross, "long_minor": longs, "short_minor": shorts, "pending_buy_minor": buy_reserve, "pending_fee_minor": fee_reserve, "cash_commitment_minor": commitment, "funding_minor": self.account["funding_minor"], "remaining_funding_minor": self.account["funding_minor"] - commitment, "equity_minor": equity, "daily_pnl_minor": equity - self.account["opening_equity_minor"], "unauthorized_shorts": unauthorized, "symbols": endpoints}

    # Convert measured limits into explicit rejection or halt reasons.
    def breaches(self, metrics):
        # Keep the configured cash ceiling separate from externally granted funding.
        caps = {"gross_minor": self.limits.max_gross_minor, "long_minor": self.limits.max_long_minor, "short_minor": self.limits.max_short_minor, "cash_commitment_minor": min(self.limits.max_cash_commitment_minor, self.account["funding_minor"])}
        # Equality is permitted for exposure caps; exceeding any one is not.
        reasons = [f"{key}={metrics[key]} exceeds {cap}" for key, cap in caps.items() if metrics[key] > cap]
        # The daily stop triggers at the limit, not one paisa after it.
        if metrics["daily_pnl_minor"] <= -self.limits.max_daily_loss_minor:
            # This condition is latched independently of future price recovery.
            reasons.append("daily loss limit reached")
        # Return every violated account constraint.
        return reasons

    # Withdraw through the existing kill switch, never through a separate cancel loop.
    def _halt(self, reason):
        # Retain the account-specific cause for status and recovery.
        self.account["halt_reason"] = reason
        # OMS's existing listener replaces desired quotes with flat intentions.
        self.oms._kill.trip(reason, "account_risk", 0)

    # Evaluate live positions on timers, mark changes and committed executions.
    def _enforce(self):
        # Initialization is blocked by the gateway until the named day setup occurs.
        if self.account["day"] is None:
            # There is no approved daily baseline to evaluate yet.
            return
        # A fill must still be committed even when its risk valuation is unavailable.
        try:
            # Measure current liabilities without proposing a new order.
            metrics = self.measure()
            # Save the current aggregate numbers for the durable audit trail.
            self.account["latest_metrics"] = metrics
            # An already latched daily stop cannot be cleared by favorable prices.
            if metrics["daily_pnl_minor"] <= -self.limits.max_daily_loss_minor:
                # This flag persists atomically with the triggering execution or mark update.
                self.account["loss_latched"] = True
            # Existing exposure can breach limits because of price changes or fills.
            reasons = self.breaches(metrics)
            # Missing short authorization is a safety condition even without a new proposal.
            if metrics["unauthorized_shorts"]:
                # Cancel quotes rather than allowing further unapproved short exposure.
                reasons.append("short availability exceeded")
            # Preserve the stop if an external caller resets only the generic kill switch.
            if self.account["loss_latched"]:
                # Daily-loss state has an independent persistent latch.
                reasons.append("daily loss halt latched")
            # Any existing breach withdraws every account symbol via normal OMS logic.
            if reasons:
                # Joining reasons preserves a concrete diagnostic for the operator.
                self._halt("; ".join(reasons))
        # Missing, stale or invalid marks cannot be treated as favorable equity.
        except (ValueError, KeyError) as error:
            # Do not display old metrics as a current valuation.
            self.account["latest_metrics"] = {"unavailable": str(error)}
            # Fill accounting still commits while quoting is halted.
            self._halt(str(error))

    # Accept complete independently timed marks without trusting persisted receive times.
    def update_marks(self, marks):
        # Serialize mark updates with fills, risk planning and durable state writes.
        self._guard()
        # Freeze one local receive-time validation instant.
        now = self._now()
        # Validate the whole batch before changing any existing observation.
        for symbol, mark in marks.items():
            # A malformed symbol or future/stale timestamp is not a usable account price.
            if not isinstance(symbol, str) or not symbol or not isinstance(mark, AccountMark) or mark.received_ms > now or now - mark.received_ms > self.limits.max_mark_age_ms:
                # Persist a halt instead of continuing on an older apparently valid mark.
                self._halt("invalid or stale account mark input")
                # Record the rejection without attempting to serialize malformed objects.
                self._commit("account_mark_rejected", {"reason": "invalid or stale account mark"})
                # Require market-data recovery and explicit operator rearming.
                raise ValueError("invalid or stale account mark")
            # A delayed snapshot must not overwrite a newer observation.
            if symbol in self.marks and mark.received_ms < self.marks[symbol].received_ms:
                # Decoder/session recovery must settle the ordering ambiguity.
                self._halt("out-of-order account mark input")
                # Persist the halt before surfacing the invalid market update.
                self._commit("account_mark_rejected", {"reason": "out-of-order account mark"})
                # An old quote cannot silently replace a newer account valuation.
                raise ValueError("out-of-order account mark")
        # Publish the validated mark batch in the serialized event loop.
        self.marks.update(marks)
        # Mark-to-market losses must trip without waiting for a strategy order.
        self._enforce()
        # Preserve mark inputs alongside any resulting daily stop.
        self._commit("account_marks", {symbol: asdict(mark) for symbol, mark in marks.items()})

    # Establish a named, explicit daily equity and authorized cash envelope.
    def begin_day(self, operator, day, funding_minor, short_caps):
        # Funding changes must be ordered with lifecycle events.
        self._guard()
        # A complete recovered account and attributable operator are prerequisites.
        if not isinstance(operator, str) or not operator.strip() or not self.reconciled:
            # An empty local ledger is not authoritative account reconciliation.
            raise ValueError("named operator and reconciled account required")
        # Validate ISO calendar format and reject same-day reset attempts.
        parsed = CalendarDate.fromisoformat(day)
        # Repeated baseline creation could otherwise erase a realized daily loss.
        if parsed.isoformat() != day or (self.account["day"] is not None and day <= self.account["day"]):
            # Day rollover must be explicit, forward-moving and once per date.
            raise ValueError("daily baseline cannot be reset or moved backwards")
        # Pending orders could belong to either day's budget and must drain first.
        if any(not order.state.is_terminal or order.has_message_in_flight for order in self.oms._orders.values()):
            # Require the same conservative barrier as restart reconciliation.
            raise ValueError("all orders and messages must drain before day setup")
        # Cash authorization is explicit and cannot exceed the configured house ceiling.
        integer(funding_minor, "funding_minor")
        # A funding input is not permission to raise the immutable house cap.
        if funding_minor > self.limits.max_cash_commitment_minor:
            # Refuse accidental limit relaxation through account bootstrap.
            raise ValueError("funding exceeds configured cash ceiling")
        # Validate the entire short-availability map before mutating risk state.
        for symbol, quantity in short_caps.items():
            # Instrument identifiers must be explicit strings.
            if not isinstance(symbol, str) or not symbol:
                # Invalid reference data is not a valid borrow authorization.
                raise ValueError("invalid short-availability symbol")
            # Zero is an explicit prohibition; negative/fractional caps are invalid.
            integer(quantity, "short availability")
        # Carry-in positions need fresh liquidation marks for the opening baseline.
        now, equity = self._now(), self.cash_minor
        # The existing durable ledger remains the source of all filled positions.
        for symbol, position in self.oms._position.items():
            # Zero inventory does not require a price to value it.
            if position:
                # Longs are marked at bid and shorts at ask at the day boundary.
                mark = self._mark(symbol, now)
                # Anchor the day's change to a conservatively executable valuation.
                equity += position * (mark.bid_minor if position > 0 else mark.ask_minor)
        # Publish the complete daily financial contract in one mutation.
        self.account.update(day=day, opening_equity_minor=equity, funding_minor=funding_minor, buy_spent_minor=0, positive_fees_minor=0, loss_latched=False, short_caps=dict(short_caps), latest_metrics=None)
        # Existing generic kill state is deliberately never reset by day setup.
        self._enforce()
        # Persist the named funding/baseline input before any quote is authorized.
        self._commit("account_begin_day", {"operator": operator, "day": day, "funding_minor": funding_minor, "short_caps": short_caps, "opening_equity_minor": equity})

    # Apply intraday reductions in cash authorization or available short quantity.
    def tighten_authorizations(self, operator, funding_minor=None, short_caps=None):
        # Authorization updates share the same serialized durable account owner.
        self._guard()
        # A named operator and initialized daily contract make the change attributable.
        if not isinstance(operator, str) or not operator.strip() or self.account["day"] is None:
            # No risk budget can be adjusted before an explicit day baseline exists.
            raise ValueError("named operator and initialized day required")
        # Omitted fields retain their previous ceiling.
        funding = self.account["funding_minor"] if funding_minor is None else integer(funding_minor, "funding_minor")
        # This is a cumulative cash envelope, not a fresh remaining bank balance.
        if funding > self.account["funding_minor"]:
            # Increasing intraday authorization requires a separate reviewed contract.
            raise ValueError("intraday funding may only tighten")
        # A supplied map replaces the prior one, so omitted symbols lose permission.
        caps = dict(self.account["short_caps"] if short_caps is None else short_caps)
        # Validate every proposed borrow/short restriction before applying any of them.
        for symbol, quantity in caps.items():
            # Reject malformed quantities rather than treating them as no change.
            integer(quantity, "short availability")
            # New permissions and larger short allocations cannot sneak into a tightening call.
            if not isinstance(symbol, str) or not symbol or quantity > self.account["short_caps"].get(symbol, 0):
                # The original daily permission remains in force until valid input arrives.
                raise ValueError("intraday short authorization may only tighten")
        # Publish both reduced authorizations in the same serialized operation.
        self.account.update(funding_minor=funding, short_caps=caps)
        # Outstanding liabilities exceeding the new envelope trigger normal quote withdrawal.
        self._enforce()
        # Persist the authorization change and any resulting kill atomically.
        self._commit("account_authorization_tightened", {"operator": operator, "funding_minor": funding, "short_caps": caps})

    # Poll account limits even when market data and strategy callbacks are silent.
    def poll_account(self):
        # Timer callbacks must run on the same owner thread as fills.
        self._guard()
        # Compare current clock, price ages, marked equity and reserved exposure.
        self._enforce()
        # Persist a newly discovered halt before a subsequent action can be dispatched.
        self._commit("account_timer", {})

    # Evaluate the current financial state before the normal cancellation-first diff.
    def reconcile(self, now_ms, date, reference_prices=None):
        # Reject attempts to plan in an inconsistent owning process.
        self._guard()
        # A date transition with open liabilities requires deliberate day setup.
        if self.account["day"] is not None and date != self.account["day"]:
            # Never let old-day funding or loss baselines authorize new-day orders.
            self._halt("trading day changed without account day setup")
        # Timers and fills can reveal breaches even when no new order is proposed.
        self._enforce()
        # The durable base commits all account mutations with the outgoing OMS batch.
        return super().reconcile(now_ms, date, reference_prices)

    # Recheck time-sensitive account safety before a previously approved new send.
    def dispatch(self, action, transport):
        # Cancel messages must never be trapped by account risk.
        if not action.is_cancel:
            # Persist current valuation and any resulting halt before transport entry.
            self.poll_account()
            # Recovery readiness and a latched kill remain mandatory at dispatch.
            if not self.reconciled or self.account["day"] is None or self.account["loss_latched"] or self.oms._kill.tripped:
                # Retain the prepared reservation without a transport side effect.
                raise RuntimeError("account safety changed before dispatch; action retained")
        # Preserve the existing durable transport boundary.
        return super().dispatch(action, transport)

    # Expose account metrics without disguising them as live trading approval.
    def status(self):
        # Reuse the recovery status's positions, cash and unresolved-order count.
        result = super().status()
        # Serialize a detached financial snapshot rather than exposing mutable state.
        result["account_risk"] = {**deepcopy(self.account), "limits": asdict(self.limits), "fresh_marks_required_after_restart": True}
        # The inherited live_approved field remains false.
        return result
