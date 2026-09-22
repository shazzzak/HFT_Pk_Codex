"""Offline replay bridge for the real durable account boundary; never live order entry.

Simulator lifecycle callbacks are normalized into stable, journaled reports.
Fees are rounded HALF_UP per execution to paisa for this test ledger only.
Analytical end-of-day liquidation rows are valuations, not venue executions.
"""
# Convert fee amounts with an explicit, reproducible rounding convention.
from decimal import Decimal, ROUND_HALF_UP
# Serialize immutable limits and normalized fill payloads.
from dataclasses import asdict
# Validate simulator floating-point grid representations.
import math
# Use the existing simulated exchange without changing its matching rules.
from mm_backtest import fee_for
# Reuse the production account controls and mark contract.
from core.account_risk import AccountLimits, AccountMark, AccountRiskManager
# Reuse the persisted lifecycle normalization contract.
from core.durable_runtime import plain
# Own an actual synchronous SQLite ledger, not an in-memory substitute.
from core.recovery_store import RecoveryStore
# Reuse the established engine-to-exchange bridge.
from sim.replay import EngineReplay
# Construct the same venue, strategy, OMS and symbol gateway as the existing gate.
from sim import gate as G


# Refuse nonintegral shares and off-grid prices instead of concealing rounding.
def whole(value, scale=1):
    # Convert simulator floats only within representation noise of an integer.
    scaled = float(value) * scale
    # Reject NaN, infinity and economically fractional values.
    if not math.isfinite(scaled) or abs(scaled - round(scaled)) > 0.000001:
        # The current production ledger only supports whole shares and paisa.
        raise ValueError(f"nonintegral ledger value: {value} x {scale}")
    # Return the checked exact domain unit.
    return int(round(scaled))


# Convert a simulator execution into one exact test-ledger fee.
def execution_fee(row):
    # Preserve the existing fee model while declaring its quantization boundary.
    return int((Decimal(str(fee_for(row["px"], row["qty"]))) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


# These intentionally wide values isolate reproduction; they are NOT live limits.
def test_limits():
    # A one-day mark age isolates ledger integration from live feed timing.
    return AccountLimits(max_gross_minor=10**15, max_long_minor=10**15, max_short_minor=10**15, max_cash_commitment_minor=10**15, max_daily_loss_minor=10**15, max_mark_age_ms=86400000, fee_reserve_bps=10, fee_reserve_fixed_minor=1)


# Adapt legacy callback signatures without exposing raw OMS mutation to replay.
class ReportBridge:
    # One bridge owns report identity for one isolated simulator/account pair.
    def __init__(self, manager):
        # Keep the durable account as the only lifecycle mutation destination.
        self.manager = manager
        # Bind the simulator after its constructor has read the venue.
        self.replay = None
        # Sequence identities are deterministic within the frozen test stream.
        self.sequence = 0
        # Keep the next unreported actual execution index.
        self.cursor = 0
        # Retain one fill for an explicit duplicate-after-restart assertion.
        self.last_fill = None
        # Count applied executions independently of simulator lifecycle summaries.
        self.fill_reports = 0

    # Delegate read/planning calls to the durable boundary.
    def __getattr__(self, name):
        # DurableManager itself refuses unnormalized lifecycle mutations.
        return getattr(self.manager, name)

    # Normalize and journal a lifecycle callback with a unique stable identity.
    def report(self, kind, payload):
        # Allocate a deterministic simulator report identifier.
        self.sequence += 1
        # Keep each identity local to this database's declared replay identity.
        report_id = f"sim:{self.sequence}"
        # Persist executions, reservations and duplicate protection atomically.
        self.manager.apply_report(report_id, kind, payload)
        # Retain exact normalized fill evidence for restart/deduplication tests.
        if kind == "fill":
            # Store the original ID and immutable payload content.
            self.last_fill = (report_id, kind, dict(payload))
            # Count per-level executions rather than averaged crossings.
            self.fill_reports += 1

    # Translate exchange acknowledgement into the production report schema.
    def on_ack(self, cl_ord_id, exchange_order_id):
        # Preserve the exchange's actual simulated order handle.
        self.report("ack", dict(cl_ord_id=cl_ord_id, exchange_order_id=exchange_order_id))

    # Translate a terminal cancellation using the cancellation message alias.
    def on_cancelled(self, cl_ord_id):
        # Release reservations only through the production lifecycle transition.
        self.report("cancelled", dict(cl_ord_id=cl_ord_id))

    # Translate a new-order rejection, including deliberate simulator holds.
    def on_rejected(self, cl_ord_id, reason):
        # Record the specific exchange/adapter reason in the durable journal.
        self.report("rejected", dict(cl_ord_id=cl_ord_id, reason=reason))

    # Translate a failed cancellation or amendment without releasing live exposure.
    def on_cancel_rejected(self, cl_ord_id, reason):
        # Let the unchanged OMS resolve its original-order alias.
        self.report("cancel_rejected", dict(cl_ord_id=cl_ord_id, reason=reason))

    # Translate an accepted amendment into exact remaining shares and price.
    def on_replaced(self, cl_ord_id, price_minor, quantity):
        # The durable boundary validates units and keeps pending-price reservations.
        self.report("replaced", dict(cl_ord_id=cl_ord_id, price_minor=price_minor, quantity=quantity))

    # Replace the legacy crossing VWAP callback with actual per-level executions.
    def on_fill(self, fill):
        # Consume only newly produced exchange fills, preserving their sequence.
        rows = self.replay.fills[self.cursor:]
        # A fill callback without actual exchange evidence is a bridge error.
        if not rows:
            # Never manufacture cash from an aggregate callback alone.
            raise AssertionError("fill callback without new exchange executions")
        # Verify the aggregate callback still refers to these exact shares/order.
        quantity = 0
        # Preserve every price level independently in the production cash ledger.
        for row in rows:
            # Analytical liquidation and unmapped order IDs are not executions.
            if row.get("oid") is None or self.replay._cl_by_oid.get(row["oid"]) != fill.cl_ord_id or row["side"] != fill.side.value:
                # Fail rather than silently assigning cash to the wrong order.
                raise AssertionError("unmapped or wrong-order execution")
            # Translate shares only after exact unit validation.
            count = whole(row["qty"])
            # Preserve the actual level price, not rounded volume-weighted price.
            price = whole(row["px"], 100)
            # Use the callback's documented integer timestamp convention.
            payload = dict(cl_ord_id=fill.cl_ord_id, symbol=fill.symbol, side=row["side"], price_minor=price, quantity=count, timestamp_ms=int(row["t"]), fee_minor=execution_fee(row))
            # Journal each level before exposing the next lifecycle change.
            self.report("fill", payload)
            # Advance the aggregate reconciliation count.
            quantity += count
        # A lost or duplicated level must invalidate the gate immediately.
        if quantity != fill.quantity:
            # Never accept a partial aggregate silently.
            raise AssertionError("per-level shares disagree with callback")
        # Advance only after every execution committed successfully.
        self.cursor = len(self.replay.fills)


# Preserve the real exchange/strategy loop while inserting durable boundaries.
class AccountReplay(EngineReplay):
    # The factory supplies the funded account, deterministic clock and limits.
    def __init__(self, *, manager, clock, journal_budget_bytes, **kwargs):
        # Keep the only financial state owner separate from the legacy callback API.
        self.manager, self.clock = manager, clock
        # Limit evidence growth without turning resource exhaustion into a pass.
        self.journal_budget_bytes = journal_budget_bytes
        # Construct the explicit callback normalizer.
        self.bridge = ReportBridge(manager)
        # Reuse all existing simulator semantics and quote decisions.
        super().__init__(oms=self.bridge, **kwargs)
        # Resolve the bridge's access to the actual per-level execution stream.
        self.bridge.replay = self
        # Existing durable bootstrap supports zero opening holdings only.
        if self.pos != 0 or self.cash != 0:
            # Carry-in reconciliation must not be faked for gate parity.
            raise ValueError("nonzero simulator opening account requires explicit bootstrap")

    # Publish a historical mark before each production quote/risk cycle.
    def _requote(self, ts_know):
        # Replay events use a synthetic monotonic clock, never the host wall clock.
        self.clock[0] = max(self.clock[0], int(ts_know))
        # Read the same reconstructed historical book as the strategy.
        bid, bq, ask, aq = self.book.bbo()
        # Invalid books cannot manufacture fresh liquidation prices.
        if bid is not None and ask is not None and bq > 0 and aq > 0 and bid <= ask:
            # One-day expiry is an explicit parity-test setting, not live freshness.
            self.manager.update_marks({self._symbol: AccountMark(whole(bid, 100), whole(ask, 100), self.clock[0])})
        # Reuse the real strategy adapter, account-aware OMS diff and gateway.
        super()._requote(ts_know)
        # Full-state journals currently grow with retained order/report history.
        if self.manager.store.path.stat().st_size > self.journal_budget_bytes:
            # Bound a user run honestly; this is a failed/incomplete cell.
            raise RuntimeError("journal budget exceeded; cell incomplete, no gate pass")

    # Commit the attempted send before the simulator's scheduling side effect.
    def _dispatch(self, action, ts_know):
        # Explicit base invocation prevents recursion through this override.
        self.manager.dispatch(action, lambda item: EngineReplay._dispatch(self, item, ts_know))

    # Simulator acknowledgement waits can reject locally before wire scheduling.
    def _unsend(self, action):
        # Record this adapter handoff and its local rejection through the same ledger.
        self.manager.dispatch(action, lambda item: EngineReplay._unsend(self, item))


# Build a fresh production OMS with the same price-band and exposure controls.
def make_oms(venue, position_limit, rejected):
    # Retain exact failed gateway decisions for parity and binding-risk assertions.
    def record(action, decision):
        # Passing decisions need no duplicate in-memory log.
        if not decision.allowed:
            # Preserve both the rejecting check and financial reason.
            rejected.append(dict(id=action.cl_ord_id, check=decision.check, reason=decision.reason))
    # Keep the previous gate's price band and working/inflight position check.
    checks = [G.OrderQuantityCheck(max_quantity=1000000), G.PriceBandCheck(venue, house_band_pct=G.HOUSE_BAND_PCT), G.PositionLimitCheck(position_limit)]
    # Exact top-ups and replacements must match the frozen backtest policy.
    return G.OrderManager(venue=venue, gateway=G.RiskGateway(checks, on_decision=record), kill_switch=G.KillSwitch(), session_id="ACCOUNT_REPLAY", account="OFFLINE_ONLY", tolerance=G.QuoteTolerance(price_ticks=0, qty_ratio=0.0, quantity_policy="exact"), use_replace=True)


# Assemble one isolated symbol-day; cross-symbol aggregation is covered separately.
def build(job, start, end, reference, database, blocked=False, journal_budget_bytes=2_000_000_000, strategy=None, cfg=None):
    # Late binding exposes published exchange bands after replay construction.
    exchange = []
    # The venue reads the same published limits as the established replay gate.
    def band(symbol):
        # No constructed exchange means no published band yet.
        if not exchange:
            # Do not invent exchange limits.
            return None
        # Read each published bound independently.
        up, down = exchange[0].book.limit_up, exchange[0].book.limit_dn
        # Preserve absence rather than substituting a house value.
        return None if up is None and down is None else G.PriceBand(upper_minor=None if up is None else whole(up, 100), lower_minor=None if down is None else whole(down, 100))
    # Use identical session hours and published-band source on both paths.
    venue = G.PSXVenue(session_provider=lambda day: [G.SessionSegment(start_ms=start, end_ms=end)], band_provider=band)
    # Preserve failed risk decisions and use a controllable receive clock.
    rejected, clock = [], [0]
    # Declare intentionally nonbinding limits separately from actual live approvals.
    limits = test_limits()
    # Bind the durable database to its frozen strategy and risk contract.
    identity = dict(kind="offline_account_replay_v1", job=job, limits=asdict(limits), blocked=blocked)
    # Refuse overwriting an existing run or importing a different account database.
    store = RecoveryStore(database, identity, create=True)
    # Close the writer if assembly fails before ownership reaches the caller.
    try:
        # Construct the existing production financial boundary.
        manager = AccountRiskManager(make_oms(venue, job["position_limit"], rejected), store, limits, clock_ms=lambda: clock[0])
        # The isolated simulator starts with no orders, inventory or cash movement.
        if not manager.reconcile_external(dict(complete=True, barrier="isolated-simulator-empty-start", open_orders=[], positions={}, cash_minor=0), "offline-gate"):
            # An unexpected opening mismatch cannot be ignored.
            raise AssertionError("empty simulator bootstrap failed")
        # The blocked arm supplies no buying power and no short authorization.
        manager.begin_day("offline-gate", job["date"], 0 if blocked else limits.max_cash_commitment_minor, {job["symbol"]: 0 if blocked else job["position_limit"]})
        # Use the real strategy unless a focused bridge unit test supplies a stub.
        strategy = strategy or G.MicrostructureMM(session_ms=(start, end), **job["params"])
        # Preserve the established replacement and seeded latency configuration.
        cfg = cfg if cfg is not None else dict(G.R.CFG, session=(start, end), latency_model=G.LatencyModel(seed=G.R.LATENCY_SEED), use_cfo=True)
        # Attach the same thin production strategy adapter.
        adapter = G.MicroMMAdapter(job["symbol"], venue, strategy, reference_price_minor=reference)
        # Construct the real simulator subclass with no matching-engine changes.
        replay = AccountReplay(manager=manager, clock=clock, journal_budget_bytes=journal_budget_bytes, strategy=strategy, adapter=adapter, symbol=job["symbol"], cfg=cfg)
        # Make the trading date available to the existing gateway.
        replay.session_date = job["date"]
        # Publish the reconstructed book to the venue's late-bound band provider.
        exchange.append(replay)
        # Expose compact factory metadata needed to verify a fresh restart.
        replay.recovery_spec = (identity, venue, limits, job["position_limit"])
        # Retain exact rejected proposals in the gate output.
        replay.risk_rejections = rejected
        # The caller owns store closure after the run and restart assertions.
        return replay
    # Assembly errors must release local writer ownership.
    except BaseException:
        # Preserve any already-written failure evidence.
        store.close()
        # Report the original failure to the gate.
        raise


# Independently reconstruct exact account cash from actual order executions.
def ledger_check(replay):
    # Analytical closing valuations must not enter the execution ledger.
    rows = [row for row in replay.fills if row.get("oid") is not None]
    # Unknown unowned rows cannot silently disappear from accounting.
    unowned = [row for row in replay.fills if row.get("oid") is None and row.get("reason") not in ("liq", "liq_residual")]
    # Opening inventory or new synthetic execution types need explicit support.
    if unowned:
        # Fail with an actionable explanation rather than adjusting the ledger.
        raise AssertionError("unsupported non-order fills in simulator")
    # Reconstruct signed executed shares independently of OMS position state.
    position = sum((1 if row["side"] == "BUY" else -1) * whole(row["qty"]) for row in rows)
    # Quantize the declared fee convention independently for every actual fill.
    fees = sum(execution_fee(row) for row in rows)
    # Compute cash using each actual execution price and whole-share quantity.
    cash = -sum((1 if row["side"] == "BUY" else -1) * whole(row["qty"]) * whole(row["px"], 100) for row in rows) - fees
    # Compare exact integers without allowing cash or inventory tolerances.
    equal = cash == replay.manager.cash_minor and fees == replay.manager.fees_minor and position == replay.manager.oms.position(replay._symbol) == replay.pos and len(rows) == replay.bridge.fill_reports
    # The simulator keeps fractional-paisa fees; expose that model difference.
    return dict(passed=equal, execution_count=len(rows), position=position, cash_minor=cash, fees_minor=fees, fee_quantization_delta_minor=cash-replay.cash*100, analytical_liquidation_rows=len(replay.fills)-len(rows))


# Verify durable reload and duplicate rejection without claiming transport recovery.
def restart_check(replay):
    # Capture the exact committed state before releasing single-writer ownership.
    before = replay.manager.store.load()
    # Preserve the database's declared identity, financial limits and venue.
    identity, venue, limits, position_limit = replay.recovery_spec
    # Keep the path after closing the original writer.
    path = replay.manager.store.path
    # A fresh owner must reopen and validate the journal chain from disk.
    replay.manager.store.close()
    # Recovery must not create a missing ledger or change its configuration.
    store = RecoveryStore(path, identity)
    # Release the recovered owner even if an invariant fails.
    try:
        # Reconstruct a new OMS and account manager from the persisted checkpoint.
        restored = AccountRiskManager(make_oms(venue, position_limit, []), store, limits, clock_ms=lambda: replay.clock[0])
        # Capture all restored fields through the production schema.
        after = restored._state()
        # Intent and prepared-message disposition intentionally change at restart.
        keys = ("orders", "working", "aliases", "replacements", "positions", "next_id", "plan_counts", "reports", "cash_minor", "fees_minor", "killed", "account_risk")
        # All ledger and contingent-exposure state must restore exactly.
        equal = all(before[key] == after[key] for key in keys)
        # Prepared actions become held; attempted/handoff actions retain status.
        expected = {key: dict(value, status="recovery_held" if value["status"] == "prepared" else value["status"]) for key, value in before["outbox"].items()}
        # No recovered message may become eligible for blind resend.
        equal = equal and expected == after["outbox"]
        # Previous desired quotes must be withdrawn, with reconciliation disabled.
        disarmed = not restored.reconciled and not restored.marks and all(desire.bid is None and desire.ask is None for desire in restored.oms._desired.values())
        # An actual prior fill, if present, must not alter any restored state twice.
        duplicate = None
        # Empty-execution runs cannot claim tested execution deduplication.
        if replay.bridge.last_fill is not None:
            # Reuse exactly the prior execution identity and financial payload.
            duplicate = restored.apply_report(*replay.bridge.last_fill) is False and restored._state() == after
        # Report the scope explicitly as checkpoint recovery, not mid-session reconnect.
        return dict(passed=equal and disarmed and duplicate is not False, checkpoint_equal=equal, disarmed=disarmed, duplicate_fill_ignored=duplicate, scope="fresh-owner checkpoint reload; no resumed exchange session")
    # Never leave a successful or failed verification holding the database lock.
    finally:
        # Complete this read/recovery ownership epoch.
        store.close()
