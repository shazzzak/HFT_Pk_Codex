# Exercise quantity conservation, causal resolution and queue integration.
import unittest
# Construct explicit source rows and checkpoint fixtures.
from types import SimpleNamespace as Row
# Use the corrected Production book.
from clean_window_book import WindowBook, WindowDataError
# Exercise the actual clean-window cancellation callback.
from clean_window_engine import WindowEngine
# Create independent books with named and anonymous depth on both sides.
def book(adds=None):
    # Retain one named order and an anonymous remainder at each touch.
    snapshot=Row(phase='CONTINUOUS_AUCTION',limit_up=20,limit_dn=1,levels=[('BUY',9,100,'bid','20'),('SELL',10,100,'ask','20')])
    # Preserve linked named orders while leaving reference five missing by default.
    metadata={1:dict(oid='bid',side='BUY',price=9,ts=0),2:dict(oid='ask',side='SELL',price=10,ts=0)}
    # Apply explicit source-reference cases without removing ordinary linked identities.
    metadata.update(adds or {})
    # Construct the book with addressable named quantity and anonymous remainder.
    return WindowBook(snapshot,metadata)
# Create a missing-add trade with an explicit resting-side reference.
def trade(**changes):
    # Use an earlier resting reference and a later event pointer.
    values=dict(appl_seq=100,buy_ref=0,sell_ref=5,price=10,qty=30,aggressor_side='BUY',ts_exch=1000)
    # Override only fields relevant to a particular regression case.
    values.update(changes)
    # Return a simulator-compatible row.
    return Row(**values)
# Create an unpriced cancellation for a previously referenced ask.
def cancel(**changes):
    # Keep the cancellation later than the price-establishing trade.
    values=dict(appl_seq=101,buy_ref=0,sell_ref=5,price=None,qty=10,side='SELL',event='CANCEL',ts_exch=1001,order_id=None)
    # Override explicit conflict or boundary fields.
    values.update(changes)
    # Return the immutable-input-compatible row shape.
    return Row(**values)
# Verify the corrected paths and the rejection boundaries together.
class AnonymousReductionTests(unittest.TestCase):
    # Pool snapshot IDs that cannot be addressed through any original add reference.
    def test_unlinked_snapshot_ids_become_anonymous(self):
        # Build a fully disclosed checkpoint whose add-reference history is unavailable.
        snapshot=Row(phase='CONTINUOUS_AUCTION',limit_up=20,limit_dn=1,levels=[('BUY',9,100,'bid','100'),('SELL',10,100,'ask','100')])
        # Omit original add mappings as in the observed source failure.
        b=WindowBook(snapshot,{})
        # Apply a trade that previously could not address the disclosed ask ID.
        b.trade(trade())
        # Verify the exact reported-level quantity remains after execution.
        self.assertEqual(b.o[b.pool('SELL',10)].qty,70)
        # Confirm pooled identity is not double-counted as another live order.
        self.assertNotIn('ask',b.o)
    # Keep malformed duplicated snapshot IDs rejected even after pooling.
    def test_duplicate_unlinked_snapshot_ids_rejected(self):
        # Use the same unresolved ID on both sides of the checkpoint.
        snapshot=Row(phase='CONTINUOUS_AUCTION',limit_up=20,limit_dn=1,levels=[('BUY',9,100,'same','100'),('SELL',10,100,'same','100')])
        # Require the preexisting identity-consistency guarantee.
        with self.assertRaisesRegex(WindowDataError,'INVALID_SNAPSHOT_ORDER'):
            # Pooling cannot hide contradictory snapshot identities.
            WindowBook(snapshot,{})
    # Preserve named volume while applying an unidentified trade to its actual level.
    def test_trade_and_later_cancel(self):
        # Start with eighty anonymous ask shares.
        b=book()
        # Apply thirty shares at the reported execution price.
        b.trade(trade())
        # Apply ten more shares using the earlier reference-price observation.
        b.cancel(cancel())
        # Verify the exact anonymous remainder.
        self.assertEqual(b.o[b.pool('SELL',10)].qty,40)
        # Verify named quantity was never used as a balancing plug.
        self.assertEqual(b.o['ask'].qty,20)
    # Confirm the symmetric bid-side reduction.
    def test_bid_trade(self):
        # Create an independent book.
        b=book()
        # Consume an anonymous bid using the actual sell execution.
        b.trade(trade(buy_ref=5,sell_ref=0,price=9,aggressor_side='SELL'))
        # Verify the correct side and level were reduced.
        self.assertEqual(b.o[b.pool('BUY',9)].qty,50)
    # Prevent an unpriced cancellation from selecting an arbitrary level.
    def test_unpriced_cancel_rejected(self):
        # Create a book without prior trade evidence.
        b=book()
        # Require the precise unresolved-price diagnostic.
        with self.assertRaisesRegex(WindowDataError,'UNPRICED_ANONYMOUS_CANCEL'):
            # Try to apply the unpriced reduction.
            b.cancel(cancel())
        # Verify rejection is nonmutating.
        self.assertEqual(b.o[b.pool('SELL',10)].qty,80)
    # Prevent anonymous reductions from consuming named shares or negative volume.
    def test_excess_reduction_rejected_without_learning(self):
        # Create eighty anonymous shares and twenty named shares.
        b=book()
        # Ninety shares fit total depth but exceed the anonymous pool.
        with self.assertRaisesRegex(WindowDataError,'UNRESOLVED_OR_EXCESS_REDUCTION'):
            # Attempt the oversized anonymous execution.
            b.trade(trade(qty=90))
        # A rejected trade must not establish cancellation-price evidence.
        self.assertEqual(b.observed_references,{})
        # The anonymous pool must remain unchanged.
        self.assertEqual(b.o[b.pool('SELL',10)].qty,80)
    # Reject contradictory side, price and future-pointer evidence.
    def test_conflicting_trade_fields(self):
        # Cover independent invalid source contracts.
        for changes in [dict(aggressor_side='SELL'),dict(price=0),dict(sell_ref=100),dict(buy_ref=7),dict(price=9.5)]:
            # Name each regression case in failure output.
            with self.subTest(changes=changes):
                # Require a data error before any accepted reduction.
                with self.assertRaises(WindowDataError):
                    # Apply one contradictory source row to a fresh book.
                    book().trade(trade(**changes))
    # Prevent a later reference observation from changing the original price.
    def test_reference_cannot_change_price(self):
        # Create the book and establish a causal price.
        b=book()
        # Consume the first valid trade.
        b.trade(trade())
        # Reject the same resting reference at a different price.
        with self.assertRaisesRegex(WindowDataError,'ANONYMOUS_REFERENCE_CONFLICT'):
            # A deeper execution cannot evade consistency checks.
            b.trade(trade(appl_seq=101,ts_exch=1001,price=11))
    # Keep known-order exhaustion from spilling into anonymous liquidity.
    def test_known_order_exhaustion(self):
        # Bind the reference to the actual named ask.
        b=book({5:dict(oid='ask',side='SELL',price=10,ts=0)})
        # Consume the complete named quantity.
        b.trade(trade(qty=20))
        # Reject any further reduction of that exhausted identity.
        with self.assertRaisesRegex(WindowDataError,'UNRESOLVED_OR_EXCESS_REDUCTION'):
            # The anonymous remainder cannot repair a named-order inconsistency.
            b.trade(trade(appl_seq=101,qty=1))
        # Preserve the unrelated anonymous pool.
        self.assertEqual(b.o[b.pool('SELL',10)].qty,80)
    # Keep original metadata contradictions from taking the fallback path.
    def test_known_future_add_rejected(self):
        # Provide an add recorded after the trade's timestamp.
        b=book({5:dict(oid='ask',side='SELL',price=10,ts=2000)})
        # Require the original chronology check to remain active.
        with self.assertRaisesRegex(WindowDataError,'UNRESOLVED_SOURCE_REFERENCE'):
            # The execution price does not override contradictory known metadata.
            b.trade(trade(qty=10))
    # Verify reductions outside visible depth never synthesize liquidity.
    def test_deeper_trade_does_not_create_level(self):
        # Create the bounded checkpoint.
        b=book()
        # Observe a real execution beyond reported ask depth.
        b.trade(trade(price=11))
        # Preserve its price for subsequent reference checks without adding volume.
        b.cancel(cancel())
        # Confirm no executable deeper level was created.
        self.assertNotIn(b.pool('SELL',11),b.o)
    # Ensure repeated resolver calls do not reduce quantity or learn future evidence.
    def test_resolve_is_nonmutating(self):
        # Create an independent book.
        b=book()
        # Resolve the trade twice as the engine and book callbacks do.
        self.assertEqual(b.resolve(trade()),b.resolve(trade()))
        # Verify neither resolution changed depth.
        self.assertEqual(b.o[b.pool('SELL',10)].qty,80)
        # Verify observation is learned only after actual trade application.
        self.assertEqual(b.observed_references,{})
    # Validate the engine's queue-ahead callback uses the corrected anonymous key.
    def test_partial_cancel_queue_integration(self):
        # Prepare an anonymous reference with fifty shares left after a trade.
        b=book()
        # Establish the causal price and reduce depth.
        b.trade(trade())
        # Represent our already-resting quote's queue-ahead state.
        quote=Row(ahead={b.pool('SELL',10):50,'ask':20})
        # Construct only the callback fixture, without strategy replay.
        engine=object.__new__(WindowEngine)
        # Attach the actual corrected historical book.
        engine.book=b
        # Expose the single quote through the ordinary order iterator.
        engine._all_orders=lambda:[quote]
        # Apply the real cancellation queue callback.
        engine._on_market_cancel(cancel())
        # Apply the historical quantity reduction as the engine does afterward.
        b.cancel(cancel())
        # Verify the queue and book agree on partial anonymous quantity.
        self.assertEqual(quote.ahead[b.pool('SELL',10)],40)
        # Verify named queue priority is unchanged.
        self.assertEqual(quote.ahead['ask'],20)
        # Verify the depth was reduced exactly once.
        self.assertEqual(b.o[b.pool('SELL',10)].qty,40)
# Run only explicitly requested unit tests.
if __name__=='__main__':
    # Print individual test outcomes and a failure exit code if applicable.
    unittest.main(verbosity=2)
