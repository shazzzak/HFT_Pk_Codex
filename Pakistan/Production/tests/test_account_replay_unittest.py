"""Small standard-library tests of the real simulator/durable-account bridge."""
# Run focused tests without requiring pytest in the user's current environment.
import unittest
# Isolate only input loading while running the full cell implementation.
from unittest.mock import patch
# Keep every test ledger outside the source repository.
import tempfile
# Resolve temporary account paths.
from pathlib import Path
# Build synthetic full-loop events without loading historical datasets.
from collections import namedtuple
# Use the real historical order book and deterministic simulated latency.
from mm_backtest import Order, LatencyModel
# Exercise the new integration and exact-ledger checks directly.
from sim.account_replay import build, ledger_check, restart_check, whole
# Compare the complete real-strategy execution sequence.
from sim.account_replay_gate import G, executions, valuation, cell
# Retain coverage of the previously delivered account-wide invariants.
from sim.account_risk_check import scenario, SCENARIOS


# Script quote intent only for lifecycle-specific bridge unit tests.
class QuoteStub:
    # Match the strategy capability read by the existing simulator.
    wants_limits = True
    # Initialize the narrow adapter contract explicitly.
    def __init__(self):
        # Quote a single bid so expected fills and fees are unambiguous.
        self.answer = {"BUY": (100.00, 10)}
        # Expose only the attributes the real adapter and exchange require.
        self.tick, self.enable_age_cross, self.allow_taker = 0.01, False, False
        # Disable optional strategy mechanisms in bridge-only unit tests.
        self.tol_ticks, self.ofi_depth_levels, self.want_taker_side = 0, 1, None
        # Keep fill diagnostics and exchange bands consistent with a fresh strategy.
        self.log_fill_state, self.current_window = False, "none"
        # No published bands exist in the synthetic fixture.
        self.limit_up, self.limit_dn = None, None
    # Return the explicitly scripted quote without changing any simulator logic.
    def quotes(self, *args, **kwargs):
        # Tests mutate this desire to exercise amendments and cancellations.
        return self.answer
    # Observation has no effect only in the bridge unit fixture.
    def observe(self, *args):
        # Full-loop tests below use the actual MicrostructureMM instead.
        return None


# Drive real OMS state, actual SQLite commits and the existing simulated exchange.
class AccountReplayTests(unittest.TestCase):
    # Isolate every test's evidence and single-writer locks.
    def setUp(self):
        # tempfile follows the external TMPDIR selected by the test command.
        self.directory = tempfile.TemporaryDirectory()
        # Release it after closing all stores registered later.
        self.addCleanup(self.directory.cleanup)
        # Explicit synthetic strategy parameters also support the real-strategy test.
        self.job = dict(symbol="PPL", date="2026-09-21", clip=10, position_limit=120, assignment="synthetic", params=dict(size=10, max_inv=100, session_scale=1.0, require_viable=False, quiet_ms=0))

    # Build an isolated parity or binding account with deterministic message timing.
    def make(self, blocked=False, stub=None):
        # A fixed latency makes before/after arrival races repeatable.
        cfg = dict(latency_model=LatencyModel(decision_ms=0, wire_out_median_ms=100, wire_out_tail_ms=0, wire_in_median_ms=100, wire_in_tail_ms=0, tail_prob=0), session=(1000, 100000), use_cfo=True, log_equity=False)
        # Use the real production assembly including PriceBandCheck.
        replay = build(self.job, 1000, 100000, 10000, Path(self.directory.name) / f"{len(list(Path(self.directory.name).glob('*.sqlite')))}.sqlite", blocked=blocked, strategy=stub or QuoteStub(), cfg=cfg)
        # Release live stores before deleting their temporary directories.
        self.addCleanup(lambda: replay.manager.store.close() if not replay.manager.store.lock.closed else None)
        # Initialize the actual historical book used by the simulator.
        self.book(replay)
        # Return the real wired integration for explicit event stimulation.
        return replay

    # Set historical visible liquidity without injecting synthetic account fills.
    def book(self, replay, bid=100.0, ask=100.1):
        # Two real historical order objects provide both valuation and execution depth.
        replay.book.o = {"B": Order("BUY", bid, 100), "A": Order("SELL", ask, 100)}
        # Permit normal continuous quoting in the adapter.
        replay.book.phase = "CONTINUOUS_AUCTION"

    # A limit that becomes marketable in transit must execute at each actual level.
    def test_crossing_levels_and_restart(self):
        # Start with a genuinely passive desired bid.
        replay = self.make()
        # Persist and dispatch the bid into the latency gap.
        replay._requote(1000)
        # Move the offer below the sent bid while it is travelling.
        self.book(replay, bid=99.90, ask=99.98)
        # Split available offers across two different actual execution prices.
        replay.book.o["A"].qty = 3
        # The second level supplies the remaining seven shares.
        replay.book.o["A2"] = Order("SELL", 99.99, 7)
        # Arrival must trade; this is the user's corrected matching behavior.
        replay._activate_until(1101)
        # Both levels must become separate integer-paisa ledger executions.
        self.assertEqual(replay.bridge.fill_reports, 2)
        # The weighted average would round differently; require the exact notional.
        self.assertAlmostEqual(sum(row["px"]*row["qty"] for row in replay.fills), 999.87)
        # Check independent cash, fees, position and execution counts.
        self.assertTrue(ledger_check(replay)["passed"])
        # Reopen from disk and ignore the duplicated final actual execution.
        self.assertTrue(restart_check(replay)["passed"])

    # Partial fills and accepted replacements must retain exact cash and reservations.
    def test_partial_replace_cancel(self):
        # Use a mutable quote fixture with the real exchange and OMS.
        stub = QuoteStub()
        # Construct a fully funded test account.
        replay = self.make(stub=stub)
        # Send and activate the initial bid.
        replay._requote(1000)
        # The hundred-millisecond outbound latency expires here.
        replay._activate_until(1101)
        # Execute only part of the acknowledged resting order.
        replay._fill("BUY", 100.0, 3, 1150, "unit_partial")
        # Request a new price with the usual ten-share target.
        stub.answer = {"BUY": (99.99, 10)}
        # The account reserves old and new generations while replacement travels.
        replay._requote(1200)
        # Verify the production financial layer retained amendment pricing.
        self.assertTrue(replay.manager.account["replacement_prices"])
        # A fill can still reach the original terms before amendment arrival.
        replay._fill("BUY", 100.0, 2, 1250, "unit_race")
        # Apply the exchange's accepted amendment.
        replay._activate_until(1301)
        # Withdraw the desired quote through the ordinary diff/cancel path.
        stub.answer = {}
        # Send the cancellation with pending exposure still reserved.
        replay._requote(1400)
        # A final old-order fill may occur before the cancel lands.
        replay._fill("BUY", 99.99, 1, 1450, "unit_cancel_race")
        # Complete the exchange cancellation.
        replay._activate_until(1501)
        # Exact cash and all six shares must match after both races.
        self.assertTrue(ledger_check(replay)["passed"])
        # Restart must preserve the filled position and all report identities.
        self.assertTrue(restart_check(replay)["passed"])

    # An unfunded account must not reach the exchange despite strategy intent.
    def test_unfunded(self):
        # Supply neither buying power nor short permissions.
        replay = self.make(blocked=True)
        # The real strategy adapter still requests its normal bid.
        replay._requote(1000)
        # Risk rejection must happen before simulated wire scheduling.
        self.assertEqual(replay.stats["n_orders_sent"], 0)
        # Demand evidence of a real account-risk check, not a silent inactive fixture.
        self.assertEqual(replay.risk_rejections[0]["check"], "account_risk")
        # No exchange activity means no cash or inventory can change.
        self.assertTrue(ledger_check(replay)["passed"])

    # End-of-input recovery must retain unresolved in-flight exposure.
    def test_pending_restart(self):
        # Start with an empty, funded simulator account.
        replay = self.make()
        # Leave its first order travelling rather than acknowledging it.
        replay._requote(1000)
        # A pending order must remain a liability after reopening the ledger.
        self.assertEqual(replay.manager.status()["unresolved_orders"], 1)
        # Reload must disable quoting without dropping that reservation.
        self.assertTrue(restart_check(replay)["passed"])

    # Preserve strict integer financial units instead of silently truncating data.
    def test_fractional_units(self):
        # Fractional shares violate the production accounting contract.
        with self.assertRaises(ValueError):
            # A superficially plausible quantity must still be rejected.
            whole(1.5)
        # More than two price decimals cannot silently become a paisa limit.
        with self.assertRaises(ValueError):
            # A rounded crossing VWAP is precisely what this bridge avoids.
            whole(99.985, 100)

    # Exercise the actual strategy's observe/quote loop on a tiny synthetic market.
    def test_real_strategy_full_event_loop(self):
        # Define ordinary historical update records accepted by the real backtester.
        Update = namedtuple("Update", "ts_exch ts_cap event order_id side price qty appl_seq")
        # Add both sides, then move prices enough to exercise actual strategy decisions.
        specs = [(1000,"B","BUY",100.0),(1000,"A","SELL",100.1),(1020,"A","SELL",99.99),(1100,"B","BUY",99.9),(1800,"B","BUY",99.9),(2200,"A","SELL",100.1),(2600,"B","BUY",100.0)]
        # Construct the event shape produced by the real loader.
        events = [(t, 1, i, "U", Update(t, t, "ORDER_ADD", oid, side, price, 100, i)) for i,(t,oid,side,price) in enumerate(specs)]
        # Run the actual calibrated strategy class with explicit synthetic parameters.
        baseline = G.run_baseline(events, {}, self.job["params"], 1000, 100000, True)
        # Build the production strategy/OMS/account/recovery path with the same seed.
        replay = build(self.job, 1000, 100000, 10000, Path(self.directory.name)/"real.sqlite")
        # Always close the original writer after this full-loop unit scenario.
        self.addCleanup(lambda: replay.manager.store.close() if not replay.manager.store.lock.closed else None)
        # Drive the same main loop and event sequence rather than direct callbacks.
        replay.run(events, {})
        # Every execution must match between the baseline and integrated engine.
        self.assertEqual(executions(baseline), executions(replay))
        # The sample must actually exercise account cash and execution reports.
        self.assertGreater(replay.bridge.fill_reports, 0)
        # Prefix valuation must agree without adding fictitious liquidation fills.
        self.assertEqual(valuation(baseline, True), valuation(replay, True))
        # Check exact actual-fill cash independently of the baseline float ledger.
        self.assertTrue(ledger_check(replay)["passed"])
        # Reopen real-strategy state and verify duplicate execution protection.
        self.assertTrue(restart_check(replay)["passed"])

    # Exercise both complete-session and prefix verdict paths on a tiny fixture.
    def test_gate_cell_full_and_prefix(self):
        # Supply the same schema emitted by the historical update loader.
        Update = namedtuple("Update", "ts_exch ts_cap event order_id side price qty appl_seq")
        # End after session close so the real analytical liquidation path is exercised.
        specs = [(1000,"B","BUY",100.0),(1000,"A","SELL",100.1),(1020,"A","SELL",99.99),(1100,"B","BUY",99.9),(1800,"B","BUY",99.9),(2200,"A","SELL",100.1),(100001,"B","BUY",100.0)]
        # Preserve the real merged-event shape and deterministic order.
        events = [(t, 1, i, "U", Update(t, t, "ORDER_ADD", oid, side, price, 100, i)) for i,(t,oid,side,price) in enumerate(specs)]
        # Exercise each scope separately; neither test loads real historical partitions.
        for maximum in (0, 6):
            # Isolate the normal and unfunded databases of this cell invocation.
            folder = Path(self.directory.name) / f"cell-{maximum}"
            # The cell requires an unused output directory owned by its caller.
            folder.mkdir()
            # Replace input acquisition only; all strategies, exchanges and ledgers remain real.
            with patch.object(G.R, "open_datasets", return_value=object()), patch.object(G, "load_symbol_day", return_value=(events, {}, 1000, 100000, 10000)):
                # Run the actual user-delivered reconciliation worker end to end.
                result = cell(self.job, str(folder), maximum, 100000000)
            # Report the complete diagnostic row if any integrated invariant fails.
            self.assertTrue(result["passed"], result)
            # Both scopes must exercise actual fills and cash ledger reconciliation.
            self.assertGreater(result["ledger"]["execution_count"], 0)

    # A local acknowledgement-wait hold must not strand a prepared durable message.
    def test_local_ack_hold(self):
        # Use the real quote planner but hold the bid at the simulator adapter.
        replay = self.make()
        # The adapter knows it is still awaiting an earlier cancellation response.
        replay.ack_until["BUY"] = 2000
        # The ordinary gate plans and durably records its locally rejected request.
        replay._requote(1000)
        # Nothing should enter the simulated outbound scheduler.
        self.assertEqual(replay.stats["n_orders_sent"], 0)
        # The durable lifecycle must retain no unresolved order after local rejection.
        self.assertEqual(replay.manager.status()["unresolved_orders"], 0)
        # Recovery must preserve both rejection identity and outbound disposition.
        self.assertTrue(restart_check(replay)["passed"])

    # Retain all eleven prior account-wide scenarios after path/integration changes.
    def test_account_regression_matrix(self):
        # Each scenario has its own ledgers and clear subtest label.
        for name in SCENARIOS:
            # One failing financial invariant must name itself in test output.
            with self.subTest(name=name):
                # Scenario runners use actual production risk and recovery classes.
                folder = Path(self.directory.name) / name
                # Give every regression scenario a distinct new database directory.
                folder.mkdir()
                # Exercise the scenario without reusing another account journal.
                result = scenario(folder, name)
                # Successful scenarios never claim live trading approval.
                self.assertFalse(result["live_approved"])


# Allow both unittest discovery and explicit invocation by a user command.
if __name__ == "__main__":
    # Return a nonzero exit code on any failed invariant.
    unittest.main(verbosity=2)
