"""Offline multi-symbol risk checks. No network, credentials or exchange orders."""
# Parse an explicit external output directory.
import argparse
# Freeze executable source bytes in each run.
import hashlib
# Write machine-readable financial and recovery evidence.
import json
# Freeze and selectively vary declared test settings.
from dataclasses import asdict, replace
# Keep every generated artifact outside the repository.
from pathlib import Path
# Resolve installed imports independently of the terminal directory.
import sys
# Locate the installed Production tree from this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Exercise the account gateway around the actual durable OMS.
from core.account_risk import AccountLimits, AccountMark, AccountRiskManager
# Preserve the existing journal's atomic state boundary.
from core.recovery_store import RecoveryStore
# Express strategy intentions without bypassing OMS planning.
from core.model import DesiredQuotes, QuoteIntent, Side
# Keep the production pending-message and amendment lifecycle.
from core.oms import OrderManager, QuoteTolerance
# Retain symbol and kill controls alongside the new account control.
from core.risk import KillSwitch, KillSwitchCheck, PositionLimitCheck, RiskGateway
# Provide a clearly synthetic venue schedule.
from core.venue import SessionSegment
# Use PSX validation without opening an exchange session.
from venues.psx import PSXVenue, static_session_provider

# Fixed clocks keep the scenarios reproducible.
DAY, NOW = "2026-09-14", 39600000
# All values are explicit test limits, not approval for live trading.
LIMITS = AccountLimits(10000000, 10000000, 10000000, 10000000, 100000, 1000, 0, 0)
# Bind account state to the exact executable risk and recovery implementation.
DEPENDENCIES = ("core/account_risk.py", "core/durable_runtime.py", "core/recovery_store.py", "core/oms.py", "core/model.py", "core/risk.py", "core/venue.py", "venues/psx.py", "sim/account_risk_check.py")
# Name each independent financial boundary tested by the matrix.
SCENARIOS = ("gross", "long", "cash", "opposite_gross", "short", "fee", "partial_cancel", "replacement_restart", "daily_loss_restart", "stale_timer", "day_boundary")


# Assertions must stay active even when Python optimization is enabled.
def require(condition, message):
    # A failed risk invariant invalidates the entire scenario.
    if not condition:
        # Retain the reason in the user's result JSON.
        raise AssertionError(message)


# Compute dependency digests before and after a user run.
def code_hashes():
    # Resolve executable files relative to this installed runner.
    root = Path(__file__).resolve().parents[1]
    # Include both scenario logic and financial implementation in the freeze.
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in DEPENDENCIES}


# Assemble one synthetic account with explicit financial settings.
def build(path, limits=LIMITS, create=True, clock=None):
    # Each process owns its monotonic receive-time domain.
    clock = clock if clock is not None else [10000]
    # A different configuration or source version cannot silently reopen this ledger.
    identity = {"schema": "account-risk-1", "account": "OFFLINE", "session": "ACCTRISK", "limits": asdict(limits), "code": code_hashes()}
    # Missing state may be initialized only by an explicit create request.
    store = RecoveryStore(path, identity, create=create)
    # The test schedule is synthetic, not a live-market-hours assertion.
    venue = PSXVenue(session_provider=static_session_provider({DAY: (SessionSegment(0, 86400000),)}))
    # Share one kill switch across the OMS and all risk checks.
    switch = KillSwitch()
    # Account checks supplement rather than replace per-symbol safety.
    gateway = RiskGateway([KillSwitchCheck(switch), PositionLimitCheck(10000)])
    # Use the previously tested exact-size, remaining-quantity amendment contract.
    oms = OrderManager(venue, gateway, switch, "ACCTRISK", account="OFFLINE", tolerance=QuoteTolerance(quantity_policy="exact"), use_replace=True)
    # Attach account state to the existing durable recovery boundary.
    manager = AccountRiskManager(oms, store, limits, lambda: clock[0])
    # Expose the test clock so silence and restart are deterministic.
    return manager, store, switch, clock


# Fresh marks are receive-timed observations, not persisted restart assumptions.
def marks(manager, clock, bid=990, ask=1000):
    # Two different prices expose incorrect cross-symbol netting.
    manager.update_marks({"OGDC": AccountMark(bid, ask, clock[0]), "PPL": AccountMark(1990, 2000, clock[0])})


# Establish a funded day only after a complete synthetic account barrier.
def start(manager, clock, short_caps=None, funding=None):
    # Actual deployments need a real authoritative account adapter for this evidence.
    require(manager.reconcile_external({"complete": True, "barrier": "offline-start", "open_orders": [], "positions": {}, "cash_minor": 0}, "tester"), "startup reconciliation")
    # Mark every fixture symbol before placing orders.
    marks(manager, clock)
    # Explicit funding and short caps cannot be inferred from strategy preferences.
    manager.begin_day("tester", DAY, manager.limits.max_cash_commitment_minor if funding is None else funding, {} if short_caps is None else short_caps)


# Send one-sided desired quotes through the durable strategy boundary.
def quote(manager, symbol="OGDC", side=Side.BUY, quantity=100, price=1000):
    # A missing opposite side means no quote is wanted there.
    intent = QuoteIntent(side, price, quantity)
    # Preserve the actual strategy-domain protocol.
    manager.set_desired(DesiredQuotes(symbol, bid=intent if side is Side.BUY else None, ask=intent if side is Side.SELL else None))


# Plan through sequential OMS risk authorization and durable batch commit.
def plan(manager):
    # Existing symbol and account checks both see these reference inputs.
    return manager.reconcile(NOW, DAY, {"OGDC": 1000, "PPL": 2000})


# Treat local handoff and exchange acknowledgement as distinct events.
def acknowledge(manager, actions):
    # The controller dispatches each risk-approved batch in order.
    for action in actions:
        # This callback is a simulated transport with no network effects.
        manager.dispatch(action, lambda _: None)
        # Stable scoped report IDs exercise actual deduplication.
        manager.apply_report("ack:" + action.cl_ord_id, "ack", {"cl_ord_id": action.cl_ord_id, "exchange_order_id": "X" + action.cl_ord_id})


# Apply explicit price, size and fees through the persistent execution boundary.
def fill(manager, action, quantity, price, fee=0, identifier="fill:1"):
    # The durable manager validates side, limit price, quantity and duplicate identity.
    return manager.apply_report(identifier, "fill", {"cl_ord_id": action.cl_ord_id, "symbol": action.symbol, "side": action.side.value, "price_minor": price, "quantity": quantity, "timestamp_ms": NOW, "fee_minor": fee})


# Run one small independent account scenario and return its final evidence.
def scenario(directory, name):
    # Change only the specific boundary being tested.
    overrides = {"gross": {"max_gross_minor": 250000}, "long": {"max_long_minor": 250000}, "cash": {"max_cash_commitment_minor": 250000}, "opposite_gross": {"max_gross_minor": 250000}, "short": {"max_short_minor": 150000}, "fee": {"max_cash_commitment_minor": 100500, "fee_reserve_bps": 100}, "replacement_restart": {"max_cash_commitment_minor": 300000}, "daily_loss_restart": {"max_daily_loss_minor": 2000}}
    # Keep all remaining controls explicit and nonbinding for the chosen boundary.
    limits = replace(LIMITS, **overrides.get(name, {}))
    # Each scenario writes a separate recoverable ledger.
    manager, store, switch, clock = build(directory / "ledger.sqlite", limits)
    # Release the current owner even when a risk invariant fails.
    try:
        # Only scenarios testing shorts receive an explicit synthetic short allocation.
        start(manager, clock, short_caps={"PPL": 1000} if name in ("short", "opposite_gross") else {})
        # Test cross-symbol financial ceilings inside one OMS batch.
        if name in ("gross", "long", "cash", "opposite_gross", "short"):
            # The first symbol consumes 100,000 paisa of account exposure.
            quote(manager)
            # The second consumes 200,000; opposite-side cases must not be netted.
            quote(manager, "PPL", Side.SELL if name in ("opposite_gross", "short") else Side.BUY, 100, 2000)
            # Earlier approved actions must reserve budget before the second symbol is checked.
            actions = plan(manager)
            # The first fits while the second exceeds its targeted account ceiling.
            require(len(actions) == 1 and actions[0].symbol == "OGDC", "cross-symbol cap did not reject second order")
            # Keep gross valuation tied to the actually approved order.
            require(manager.measure()["gross_minor"] == 100000, "incorrect approved gross")
            # Unfilled buys also cannot manufacture a short-sale authorization.
            if name == "short":
                # Withdraw the bid and request an uncovered sell with no OGDC short allocation.
                quote(manager, side=Side.SELL)
                # The pending buy is not owned inventory.
                require(plan(manager) == [], "pending buy incorrectly authorized uncovered sell")
        # A notional that fits may still exceed funding once fees are reserved.
        elif name == "fee":
            # Reserve 100,000 notional plus a deliberately synthetic 1,000 fee allowance.
            quote(manager)
            # A 100,500 budget cannot support the resulting 101,000 commitment.
            require(plan(manager) == [], "fee allowance did not consume funding")
        # Exercise exact partial-fill transfer and pending-cancel funding lifetime.
        elif name == "partial_cancel":
            # Place one hundred shares through the real account gateway.
            quote(manager)
            # Keep the durable order identifier for every lifecycle report.
            action = plan(manager)[0]
            # Establish a resting order before it partially executes.
            acknowledge(manager, [action])
            # Twenty shares execute at a better price, including a seven-paisa fee.
            fill(manager, action, 20, 990, 7)
            # 19,800 spent + 7 fees + 80,000 remaining limit liability.
            require(manager.measure()["cash_commitment_minor"] == 99807, "partial-fill commitment incorrect")
            # An exact duplicate must not consume the cash envelope twice.
            require(fill(manager, action, 20, 990, 7) is False, "duplicate execution applied")
            # Ordinary flat desire initiates cancellation of the remainder.
            manager.set_desired(DesiredQuotes.flat("OGDC"))
            # Persist the pending cancel without releasing any leaves.
            cancel = plan(manager)[0]
            # A local cancel request is not an exchange cancellation.
            require(manager.measure()["cash_commitment_minor"] == 99807, "pending cancel released funding")
            # Only the normalized authoritative response removes the old liability.
            manager.apply_report("cancel:1", "cancelled", {"cl_ord_id": cancel.cl_ord_id})
            # Filled inventory and execution fees remain consumed.
            require(manager.measure()["cash_commitment_minor"] == 19807, "cancel released executed spend")
            # Offer only stock that actually exists in the account.
            quote(manager, side=Side.SELL, quantity=20, price=990)
            # Sell through the same risk, durability and acknowledgement boundaries.
            sale = plan(manager)[0]
            # Confirm the sale order before applying its execution.
            acknowledge(manager, [sale])
            # Charge a second explicit fee on the sale.
            fill(manager, sale, 20, 990, 2, "fill:2")
            # Sale proceeds and fee rebates never replenish conservative buying power.
            require(manager.measure()["cash_commitment_minor"] == 19809, "sale proceeds replenished funding")
            # With flat inventory, daily P&L is exactly the two fees paid.
            require(manager.measure()["daily_pnl_minor"] == -9, "fees absent from daily P&L")
        # Check old/new amendment reservations and their higher proposed price after restart.
        elif name == "replacement_restart":
            # Begin with an acknowledged hundred-share bid at 1,000 paisa.
            quote(manager)
            # Preserve its client identifier for later aliases.
            action = plan(manager)[0]
            # Put the original generation into the live state.
            acknowledge(manager, [action])
            # A partial fill leaves eighty shares under the old terms.
            fill(manager, action, 20, 990, 7)
            # Propose 150 remaining shares at a higher 1,100-paisa limit.
            quote(manager, quantity=150, price=1100)
            # Exact-size policy uses the real OMS replacement path.
            amendment = plan(manager)[0]
            # 19,800 + 7 + 80,000 old leaves + 165,000 contingent new leaves.
            require(manager.measure()["cash_commitment_minor"] == 264807, "amendment not fully reserved")
            # Keep the same database while changing process ownership.
            path = store.path
            # Close without confirming the pending amendment.
            store.close()
            # Restore both the original leaves and new-price reservation.
            manager, store, switch, clock = build(path, limits, create=False, clock=clock)
            # Restart requires new local receive-time observations.
            marks(manager, clock)
            # The higher proposed limit must remain reserved after recovery.
            require(manager.measure()["cash_commitment_minor"] == 264807, "replacement price lost on restart")
            # Resolve the persisted alias with the actual remaining-size confirmation.
            manager.apply_report("replace:1", "replaced", {"cl_ord_id": amendment.cl_ord_id, "price_minor": 1100, "quantity": 150})
            # Only the old unfilled generation is released.
            require(manager.measure()["cash_commitment_minor"] == 184807, "replacement reserve release incorrect")
            # No financial snapshot restores transport/account readiness automatically.
            require(not manager.reconciled, "restart restored authorization")
        # Test a fee-inclusive marked loss without waiting for another order request.
        elif name == "daily_loss_restart":
            # Buy inventory in the first symbol.
            quote(manager)
            # Keep a second symbol quoting to observe global cancellation.
            quote(manager, "PPL", quantity=10, price=2000)
            # Both requests initially fit within every account limit.
            actions = plan(manager)
            # Acknowledge both before the inventory execution.
            acknowledge(manager, actions)
            # At bid 990 this buy creates a 1,010-paisa loss including fees.
            fill(manager, actions[0], 100, 1000, 10)
            # The configured 2,000 loss threshold must not trigger early.
            require(not switch.tripped, "daily loss triggered too early")
            # Bid 980 raises the marked loss to 2,010 paisa.
            marks(manager, clock, bid=980, ask=1000)
            # The mark update itself must persist the loss latch and trip the kill.
            require(switch.tripped and manager.account["loss_latched"], "daily loss did not latch")
            # Cancellation goes through the ordinary OMS diff and durable batch path.
            cancels = plan(manager)
            # Only the still-resting second-symbol order needs cancellation.
            require(len(cancels) == 1 and cancels[0].is_cancel, "loss halt did not produce normal cancel")
            # A favorable price rebound must not silently unhalt the account.
            marks(manager, clock, bid=1200, ask=1210)
            # Keep the daily loss stop independent of later displayed P&L.
            require(manager.account["loss_latched"], "price rebound cleared loss halt")
            # End the current ownership epoch with the halt still active.
            path = store.path
            # Shutdown does not resolve the pending cancellation.
            store.close()
            # Restore persisted financial and kill state into a new owner.
            manager, store, switch, clock = build(path, limits, create=False, clock=clock)
            # A restart must not erase the daily-loss stop.
            require(manager.account["loss_latched"] and switch.tripped, "restart cleared loss halt")
        # A timer must enforce freshness when the feed sends nothing.
        elif name == "stale_timer":
            # Begin with one acknowledged live quote.
            quote(manager)
            # Dispatch through the account-aware durable boundary.
            acknowledge(manager, plan(manager))
            # Advance beyond the explicitly configured one-second mark age.
            clock[0] += 1001
            # Polling must work independently of new market or strategy messages.
            manager.poll_account()
            # Stale valuation cannot be interpreted as a safe zero account.
            require(switch.tripped, "stale account valuation did not halt")
            # An unavailable valuation cannot prevent cancellation.
            cancels = plan(manager)
            # Normal OMS cancellation remains the only withdrawal mechanism.
            require(len(cancels) == 1 and cancels[0].is_cancel, "stale valuation blocked cancel")
            # The dispatch guard also preserves unconditional cancel permission.
            manager.dispatch(cancels[0], lambda _: None)
        # Same-day reset and implicit date rollover must never erase loss/spend history.
        elif name == "day_boundary":
            # Attempt an invalid second baseline within the same trading day.
            try:
                # Even a named operator cannot silently reset the daily counters.
                manager.begin_day("tester", DAY, 10000000, {})
            # Refusal is the required control behavior.
            except ValueError:
                # Preserve the existing funded baseline unchanged.
                pass
            # Accepting the reset would invalidate every daily limit.
            else:
                # Report the precise broken invariant.
                raise AssertionError("same-day baseline reset accepted")
            # A new date arriving without explicit setup must halt.
            manager.reconcile(NOW, "2026-09-15", {})
            # Preserve the old baseline rather than implicitly rolling the ledger.
            require(switch.tripped and manager.account["day"] == DAY, "implicit day reset")
        # Unknown names must not accidentally produce an empty passing case.
        else:
            # Keep the matrix's coverage contract explicit.
            raise ValueError("unknown scenario")
        # Return detached durable monitoring evidence for the result file.
        return manager.status()
    # Always release whichever account owner is current after a restart.
    finally:
        # A failed reopen may leave the original store already closed.
        if not store.lock.closed:
            # Preserve the SQLite journal for independent inspection.
            store.close()


# Run the user's repeated validation matrix with independent evidence per case.
def main():
    # Keep every user-specified setting visible in the command line.
    parser = argparse.ArgumentParser(description=__doc__)
    # Require external output instead of silently writing under Production.
    parser.add_argument("--output-dir", required=True, type=Path)
    # Larger repetitions are run by the user, not the coding agent.
    parser.add_argument("--iterations", type=int, default=1)
    # Parse the selected run size and artifact location.
    args = parser.parse_args()
    # A run with no cases cannot validate risk behavior.
    if args.iterations < 1:
        # Explain the invalid size before creating any files.
        parser.error("iterations must be positive")
    # Resolve symlinks before checking the repository boundary.
    output, production = args.output_dir.resolve(), Path(__file__).resolve().parents[1]
    # Find the actual enclosing Git root regardless of project nesting depth.
    repository = next((p for p in (production, *production.parents) if (p / ".git").exists()), production)
    # Keep generated databases and reports out of source control.
    if output.is_relative_to(repository):
        # Stop before creating output under the source checkout.
        parser.error("output directory must be outside the source repository")
    # Preserve previous evidence by requiring a new output directory.
    output.mkdir(parents=True, exist_ok=False)
    # Freeze source inputs before the first financial scenario.
    before, results = code_hashes(), []
    # Run serially so each database has exactly one owning process.
    for iteration in range(args.iterations):
        # Every scenario gets an isolated funded account fixture.
        for name in SCENARIOS:
            # Give each evidence cell a stable unique directory.
            path = output / f"{iteration:04d}_{name}"
            # Keep each ledger and its result next to each other.
            path.mkdir()
            # Continue collecting diagnostics even after an ordinary case failure.
            try:
                # Each named scenario asserts its own financial invariants.
                result = {"scenario": name, "passed": True, "status": scenario(path, name)}
            # Unexpected errors are failures, never omitted evidence cells.
            except Exception as error:
                # Preserve the failure reason for the user to share.
                result = {"scenario": name, "passed": False, "error": str(error)}
            # Save the detailed per-case account result externally.
            (path / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            # Add the same case to the final run summary.
            results.append(result)
            # Emit immediate progress in the PyCharm terminal.
            print(f"{iteration + 1}/{args.iterations} {name}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    # Source drift during a run invalidates fixed-configuration evidence.
    changed = before != code_hashes()
    # Any failed case or changed input fails the whole matrix.
    summary = {"passed": not changed and all(r["passed"] for r in results), "planned": args.iterations * len(SCENARIOS), "completed": len(results), "failed": sum(not r["passed"] for r in results), "changed_inputs": changed, "live_approved": False, "test_limits": asdict(LIMITS), "code_hashes": before, "cases": results}
    # Persist the complete evidence next to the independent account ledgers.
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Keep the last console line compact enough to paste back for review.
    print(json.dumps({k: v for k, v in summary.items() if k not in ("test_limits", "code_hashes", "cases")}), flush=True)
    # A failed run must propagate through shell/PyCharm exit status.
    return 0 if summary["passed"] else 1


# Importing fixture helpers never starts a run or writes a database.
if __name__ == "__main__":
    # Return the risk matrix outcome to the terminal.
    raise SystemExit(main())
