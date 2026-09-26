# Test exchange-ID reuse independently of source planner selection.
import unittest
# Construct explicit source records.
from types import SimpleNamespace as Row
# Use the production generation-aware index and book.
from clean_window_identity import AddIndex
# Assert exact rejection boundaries as well as successful reconstruction.
from clean_window_book import WindowBook,WindowDataError
# Exercise snapshot eligibility with a later reused identity.
from clean_window_data import plan_windows

# Construct two source generations sharing one exchange order ID.
def metadata(second_price=11):
    # Preserve immutable references and their actual creation times.
    return AddIndex({1:dict(oid='reused',key='ref:2011:1',sequence=1,side='SELL',price=10,ts=1000),3:dict(oid='reused',key='ref:2011:3',sequence=3,side='SELL',price=second_price,ts=10000)}).seal()

# Create a snapshot disclosing the reusable exchange identity.
def picture(price=10,qty=20):
    # Report independent bid depth so ask generation changes remain testable.
    return Row(phase='CONTINUOUS_AUCTION',limit_up=20,limit_dn=1,levels=[('BUY',9,100,'',''),('SELL',price,qty,'reused',str(qty))])

# Verify causal snapshot linkage and exact immutable reduction targets.
class GenerationTests(unittest.TestCase):
    # A later amendment must not invalidate the earlier valid snapshot.
    def test_snapshot_links_earlier_generation(self):
        # Seed at a cutoff before the second generation exists.
        book=WindowBook(picture(),metadata(),asof=5000)
        # Keep the earlier sequence as the executable identity.
        self.assertEqual(book.o['ref:2011:1'].qty,20)
        # Exclude the future generation from current liquidity.
        self.assertNotIn('ref:2011:3',book.o)
    # Cancel/re-add updates with one exchange ID must remain two distinct generations.
    def test_cancel_then_readd_same_id(self):
        # Seed the first generation only.
        book=WindowBook(picture(),metadata(),asof=5000)
        # Remove the old generation using its actual source reference.
        book.cancel(Row(event='CANCEL',appl_seq=2,sell_ref=1,buy_ref=0,side='SELL',price=10,qty=20,ts_exch=10000))
        # Add the replacement at its new price without a duplicate-ID rejection.
        book.add(Row(event='ORDER_ADD',appl_seq=3,side='SELL',price=11,qty=30,ts_exch=10000))
        # A new ask beyond the old snapshot range is not fabricated as complete depth.
        self.assertNotIn('ref:2011:3',book.o)
        # Adopt the next authoritative picture of that new price.
        book.refresh(picture(price=11,qty=30),asof=11000)
        # Link the new picture to the replacement generation.
        self.assertEqual(book.o['ref:2011:3'].qty,30)
        # Never leave cancelled old quantity behind.
        self.assertNotIn('ref:2011:1',book.o)
    # An old reference cannot reduce a newer generation at the same price.
    def test_exhausted_reference_cannot_spill_after_refresh(self):
        # Put both generations at the same price to expose identity errors.
        book=WindowBook(picture(),metadata(second_price=10),asof=5000)
        # Exhaust the original generation.
        book.cancel(Row(event='CANCEL',appl_seq=2,sell_ref=1,buy_ref=0,side='SELL',price=10,qty=20,ts_exch=10000))
        # Add its replacement under a new application reference.
        book.add(Row(event='ORDER_ADD',appl_seq=3,side='SELL',price=10,qty=30,ts_exch=10000))
        # Refresh with fifty shares, including twenty anonymous shares at that price.
        snapshot=picture(qty=30)
        # Retain the replacement's thirty named shares plus anonymous depth.
        snapshot.levels[1]=('SELL',10,50,'reused','30')
        # Update market quantities while retaining exhausted-generation evidence.
        book.refresh(snapshot,asof=11000)
        # A stale cancel must not consume the replacement or anonymous shares.
        with self.assertRaisesRegex(WindowDataError,'STALE_ORDER_GENERATION'):
            # Reference the exhausted first generation explicitly.
            book.cancel(Row(event='CANCEL',appl_seq=4,sell_ref=1,buy_ref=0,side='SELL',price=10,qty=1,ts_exch=12000))
        # Preserve all fifty authoritative shares after rejection.
        self.assertEqual(book.bbo()[3],50)
    # Starting after a replacement must not forget which older references are stale.
    def test_stale_reference_after_fresh_start(self):
        # Report new named liquidity alongside an anonymous remainder at the same price.
        snapshot=picture(qty=30)
        # Preserve twenty anonymous shares to expose an erroneous fallback reduction.
        snapshot.levels[1]=('SELL',10,50,'reused','30')
        # Start after the replacement without having processed the earlier cancellation.
        book=WindowBook(snapshot,metadata(second_price=10),asof=11000)
        # A stale reference must not reduce this unrelated anonymous remainder.
        with self.assertRaisesRegex(WindowDataError,'STALE_ORDER_GENERATION'):
            # Attempt to consume one share using the superseded application's reference.
            book.cancel(Row(event='CANCEL',appl_seq=4,sell_ref=1,buy_ref=0,side='SELL',price=10,qty=1,ts_exch=12000))
        # Preserve the authoritative fifty-share level after rejection.
        self.assertEqual(book.bbo()[3],50)
    # IDs seen only in future additions cannot be linked using future metadata.
    def test_future_only_id_stays_anonymous(self):
        # A snapshot before either known generation still has real reported quantity.
        book=WindowBook(picture(),metadata(),asof=500)
        # Keep that quantity in the exact-price anonymous pool.
        self.assertEqual(book.o[book.pool('SELL',10)].qty,20)
        # Do not import the first later add as if already received.
        self.assertNotIn('ref:2011:1',book.o)
    # Real source books must never silently default to a whole-day ID lookup.
    def test_source_book_requires_causal_cutoff(self):
        # Require explicit time evidence for a real indexed source.
        with self.assertRaisesRegex(WindowDataError,'MISSING_SNAPSHOT_CUTOFF'):
            # Omitting cutoff used to permit last-add-of-day contamination.
            WindowBook(picture(),metadata())
    # The planner must retain a quiet earlier snapshot despite a later amendment.
    def test_later_add_does_not_reject_checkpoint(self):
        # Give the planner an earlier quiet snapshot and a bounded source segment.
        checkpoints=[dict(origin=4000,start=5000,valid=True,snapshot=picture())]
        # The metadata includes a later generation but no event inside this bounded slice.
        windows,counts=plan_windows([],checkpoints,metadata(),dict(unusable=False,intervals=[],last_ms=9000),[(0,9000)],min_ms=0)
        # Retain the independently valid earlier interval.
        self.assertEqual([(w['start'],w['end']) for w in windows],[(5000,9000)])
        # Do not reproduce the incorrect last-ID future rejection.
        self.assertNotIn('FUTURE_ORDER_IN_SNAPSHOT',counts)

# Run the focused regression suite only on explicit invocation.
if __name__=='__main__':
    # Report each generation and timing check separately.
    unittest.main(verbosity=2)
