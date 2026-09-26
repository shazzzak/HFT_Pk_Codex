# Exercise market refreshes independently of any profit claim.
import unittest
# Build explicit timestamped source fixtures.
from types import SimpleNamespace as Row
# Import the corrected source planner.
import clean_window_data as data
# Exercise the actual market book and strategy adapter.
from clean_window_book import WindowBook, WindowDataError
# Preserve the real replay's explicit queue-unavailability behavior.
from clean_window_engine import WindowEngine, WindowExitError

# Create an authoritative two-sided market picture.
def snapshot(ask_qty=10,ask_price=10):
    # Leave identities anonymous so exact level conservation is visible.
    return Row(phase='CONTINUOUS_AUCTION',limit_up=20,limit_dn=1,levels=[('BUY',9,100,'',''),('SELL',ask_price,ask_qty,'','')])

# Create a checkpoint with the loader's one-second conservative availability rule.
def checkpoint(origin,**changes):
    # Keep the default capture inside its coarse source second.
    return dict(origin=origin,start=origin+1000,valid=True,snapshot=snapshot(**changes))

# Create a trade with a missing original add and explicit execution price.
def trade(ts=7000,qty=10):
    # Reference an earlier order and a later application sequence.
    return (ts,1,100,'T',Row(ts_exch=ts,ts_cap=ts,appl_seq=100,buy_ref=0,sell_ref=1,price=10,qty=qty,aggressor_side='BUY'))

# Use an uninterrupted synthetic channel and a long enough research segment.
QUALITY=dict(unusable=False,intervals=[],last_ms=100000)

# Verify replacement, timing, causal evidence and account-state invariants.
class SnapshotRefreshTests(unittest.TestCase):
    # New pictures supersede quantities rather than adding duplicate depth.
    def test_replaces_old_depth_and_applies_next_trade_once(self):
        # Start with ten shares offered at ten.
        book=WindowBook(snapshot(),{})
        # Replace them with twenty shares at the same price.
        book.refresh(snapshot(ask_qty=20))
        # Apply one actual ten-share execution.
        book.trade(trade()[4])
        # The correct remainder is ten, not twenty or thirty.
        self.assertEqual(book.bbo(),(9.0,100.0,10.0,10.0))
        # Replace the entire picture with a different ask price.
        book.refresh(snapshot(ask_price=11))
        # No stale order at the old ask may survive.
        self.assertEqual(book.qty_at('SELL',10),{})
    # A malformed refresh leaves the previous usable book intact.
    def test_rejected_replacement_is_atomic(self):
        # Preserve a known valid book.
        book=WindowBook(snapshot(),{})
        # Reject a crossed replacement before assignment.
        with self.assertRaises(WindowDataError):
            # An ask below the bid cannot replace a valid book.
            book.refresh(snapshot(ask_price=8))
        # The prior quote must remain unchanged.
        self.assertEqual(book.bbo(),(9.0,100.0,10.0,10.0))
    # Reference-price observations survive replacement but remaining quantities do not.
    def test_causal_reference_survives_refresh(self):
        # Begin with sufficient anonymous volume.
        book=WindowBook(snapshot(ask_qty=30),{})
        # Establish a successful reference-price observation.
        book.trade(trade()[4])
        # Adopt a new quantity at the same reported price.
        book.refresh(snapshot(ask_qty=50))
        # Use the earlier causal price for a later cancellation.
        book.cancel(Row(event='CANCEL',side='SELL',price=None,qty=5,appl_seq=101,sell_ref=1,buy_ref=0,ts_exch=8000))
        # Subtract from the replacement quantity exactly once.
        self.assertEqual(book.bbo()[3],45)
    # Refreshes occur inside a long window rather than producing five-second flat resets.
    def test_planner_refreshes_active_window(self):
        # Supply a picture every five seconds for over one minute.
        pictures=[checkpoint(t,ask_qty=20) for t in range(0,70000,5000)]
        # Make the initial picture insufficient for the later trade.
        pictures[0]=checkpoint(0)
        # Require the normal sixty-second minimum duration.
        windows,counts=data.plan_windows([trade()],pictures,{},QUALITY,[(0,100000)])
        # Every refresh must remain inside one continuous candidate.
        self.assertEqual(len(windows),1)
        # Preserve the entire supported interval.
        self.assertEqual((windows[0]['start'],windows[0]['end']),(1000,100000))
        # Apply all thirteen later snapshots rather than skipping them.
        self.assertEqual(len(windows[0]['refreshes']),13)
        # Keep refresh diagnostics explicit.
        self.assertEqual(counts['SNAPSHOT_REFRESH_APPLIED'],13)
    # A coarse snapshot boundary with concurrent updates cannot silently double-count events.
    def test_ambiguous_snapshot_is_reported(self):
        # Put a mutation inside the later snapshot's uncertain source second.
        events=[trade(ts=5500,qty=1)]
        # Keep the initial book sufficient throughout this test.
        windows,counts=data.plan_windows(events,[checkpoint(0),checkpoint(5000)],{},QUALITY,[(0,100000)])
        # Reject only the ambiguous replacement and retain the valid current reconstruction.
        self.assertEqual(windows[0]['refreshes'],[])
        # Do not hide why a five-second picture was not adopted.
        self.assertEqual(counts['SNAPSHOT_TIME_AMBIGUOUS'],1)
    # No snapshot may repair a historical channel gap before its receipt.
    def test_gap_still_ends_source_window(self):
        # Place a real channel gap after one usable refresh.
        quality=dict(QUALITY,intervals=[[8000,9000,'MISSING_SEQUENCE_OR_CHANNEL_TIMING']])
        # Inspect short candidates directly to verify exact boundaries.
        windows,counts=data.plan_windows([], [checkpoint(0),checkpoint(5000),checkpoint(10000)],{},quality,[(0,100000)],min_ms=0)
        # End before the first missing timestamp.
        self.assertEqual(windows[0]['end'],7999)
        # Restart only from the later independently usable picture.
        self.assertEqual(windows[1]['start'],11000)
    # A late older picture cannot overwrite an already received newer picture.
    def test_delayed_snapshot_does_not_roll_back_market(self):
        # Delay the older generation beyond a later picture's availability.
        delayed=dict(checkpoint(5000),start=20000)
        # Use a quiet interval so age is the only reason for rejection.
        windows,counts=data.plan_windows([], [checkpoint(0),delayed,checkpoint(10000)],{},QUALITY,[(0,100000)])
        # Preserve the newest generation's schedule only.
        self.assertEqual([c['origin'] for c in windows[0]['refreshes']],[10000])
        # Record the discarded stale generation explicitly.
        self.assertEqual(counts['STALE_SNAPSHOT'],1)
    # Simultaneous availability adopts the newest valid generation exactly once.
    def test_same_availability_uses_latest_generation(self):
        # Delay the older picture to the later picture's availability.
        delayed=dict(checkpoint(5000),start=11000)
        # Preserve both original records in the input.
        windows,counts=data.plan_windows([], [checkpoint(0),delayed,checkpoint(10000,ask_qty=30)],{},QUALITY,[(0,100000)])
        # Keep only the newest authoritative picture at that instant.
        self.assertEqual([c['origin'] for c in windows[0]['refreshes']],[10000])
    # A phase boundary cannot be erased by a later depth refresh.
    def test_phase_change_still_ends_window(self):
        # Construct an explicit halt-state observation.
        halted=checkpoint(10000)
        # Preserve the actual noncontinuous phase.
        halted['snapshot'].phase='HALTED'
        # Inspect the short pre-halt source interval without applying the normal minimum.
        windows,counts=data.plan_windows([], [checkpoint(0),checkpoint(5000),halted],{},QUALITY,[(0,100000)],min_ms=0)
        # Exclude the entire first halted source timestamp.
        self.assertEqual(windows[0]['end'],9999)
        # Deliver only the pre-halt replacement.
        self.assertEqual(len(windows[0]['refreshes']),1)
    # Market replacement must not erase our account, pending requests or order identity.
    def test_engine_preserves_account_and_order_state(self):
        # Build only the real refresh callback's account fixture.
        engine=object.__new__(WindowEngine)
        # Attach an ordinary anonymous source book.
        engine.book=WindowBook(snapshot(),{})
        # Keep a genuine live order and its pending cancellation marker.
        order=Row(side='SELL',price=10,t_active=2000,ahead={},cancel_at=9000,qty=3)
        # Make the same exchange-order object visible to the real callback.
        engine._all_orders=lambda:[order]
        # Retain immutable source arrival times.
        engine.source_times={}
        # Initialize the diagnostic counter.
        engine.snapshot_refreshes=0
        # Give the account nonzero inventory and cash.
        engine.pos,engine.cash=7,123.0
        # Preserve a pending-message object by identity.
        engine.pending=[('cancel',order)]
        # Keep a reference for a nonreplacement assertion.
        pending=engine.pending
        # Refresh historical depth using the actual engine method.
        engine._refresh_market(snapshot(ask_qty=50))
        # Inventory and cash must survive untouched.
        self.assertEqual((engine.pos,engine.cash),(7,123.0))
        # Pending messages cannot be recreated or erased.
        self.assertIs(engine.pending,pending)
        # The quote's cancellation and remaining size must remain unchanged.
        self.assertEqual((order.cancel_at,order.qty,order.t_active),(9000,3,2000))
        # Anonymous depth receives a conservative queue bound, not invented priority.
        self.assertEqual(sum(order.ahead.values()),50)
    # Known later arrivals stay behind our already-live quote after a replacement.
    def test_queue_preserves_known_arrival_order(self):
        # Create exact original add-time metadata for disclosed orders.
        adds={1:dict(oid='early',side='SELL',price=10,ts=1000),2:dict(oid='late',side='SELL',price=10,ts=3000)}
        # Allocate the refresh fixture.
        engine=object.__new__(WindowEngine)
        # Seed the source book with the same immutable add index.
        engine.book=WindowBook(snapshot(),adds)
        # Place our order between the two historical arrivals.
        order=Row(side='SELL',price=10,t_active=2000,ahead={})
        # Expose the live quote.
        engine._all_orders=lambda:[order]
        # Preserve both arrival timestamps.
        engine.source_times={'early':1000,'late':3000}
        # Initialize the counter required by the callback.
        engine.snapshot_refreshes=0
        # Disclose two named orders and five anonymous shares.
        picture=snapshot(ask_qty=35)
        # Replace only the ask's identity details in this test picture.
        picture.levels[1]=('SELL',10,35,'early|late','10|20')
        # Reconcile actual queue evidence.
        engine._refresh_market(picture)
        # Only the earlier named order and conservative anonymous quantity stand ahead.
        self.assertEqual(order.ahead,{'early':10,engine.book.pool('SELL',10):5})
    # Refreshing to shallower visible depth must not turn an unknown live queue into zero ahead.
    def test_unknown_live_queue_fails_without_deleting_order(self):
        # Allocate an isolated engine callback fixture.
        engine=object.__new__(WindowEngine)
        # Seed the existing source book.
        engine.book=WindowBook(snapshot(),{})
        # Keep a live bid below the newly reported depth boundary.
        order=Row(side='BUY',price=8,t_active=2000,ahead={'old':50},qty=3)
        # Expose the actual same order object.
        engine._all_orders=lambda:[order]
        # Preserve required immutable metadata.
        engine.source_times={}
        # Initialize diagnostics.
        engine.snapshot_refreshes=0
        # Require explicit queue uncertainty rather than a fabricated fill opportunity.
        with self.assertRaisesRegex(WindowExitError,'SNAPSHOT_QUEUE_OUTSIDE_REPORTED_DEPTH'):
            # Replace market depth with the ordinary reported bid boundary at nine.
            engine._refresh_market(snapshot())
        # Keep the simulated order alive and its previous queue evidence intact.
        self.assertEqual((order.qty,order.ahead),(3,{'old':50}))

# Run the focused regression suite only when explicitly invoked.
if __name__=='__main__':
    # Report individual test results and a nonzero exit on any failure.
    unittest.main(verbosity=2)
