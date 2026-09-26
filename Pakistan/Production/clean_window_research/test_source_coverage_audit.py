# Validate sampling, interval arithmetic and a real saved source-cell comparison.
import csv, unittest
# Import the audit without invoking its guarded CLI entry point.
import audit_source_coverage as audit
# Group observed sample categories for independent coverage checks.
from collections import Counter
# Test core audit contracts without launching any trading strategy.
class SourceCoverageAuditTests(unittest.TestCase):
    # Require disjoint gain/loss arithmetic even when windows move in both directions.
    def test_shifted_intervals(self):
        # Use two original intervals with a gap between them.
        old=[(0,10),(20,30)]
        # Use a shifted interval overlapping both old segments.
        new=[(5,25)]
        # The overlap consists of two independently known five-unit pieces.
        self.assertEqual(audit.intersection_ms(old,new),10)
        # Equal total time can conceal ten gained and ten lost units.
        self.assertEqual(sum(b-a for a,b in new)-audit.intersection_ms(old,new),10)
    # Verify adjacency contributes zero shared duration.
    def test_adjacent_and_empty(self):
        # Adjacent endpoints must not count as an extra millisecond.
        self.assertEqual(audit.intersection_ms([(0,10)],[(10,20)]),0)
        # Missing candidate windows must remain a valid empty duration.
        self.assertEqual(audit.intersection_ms([],[(0,10)]),0)
    # Reject overlapping planner output before generating misleading totals.
    def test_invalid_windows(self):
        # Attempt overlapping candidate intervals.
        with self.assertRaises(ValueError):
            # The guard must stop double-counting source time.
            audit.intervals([dict(start=0,end=10),dict(start=9,end=20)])
        # Reject a zero-duration interval too.
        with self.assertRaises(ValueError):
            # Empty windows are not retained one-minute candidates.
            audit.intervals([dict(start=4,end=4)])
    # Independently verify the real saved universe produces a deterministic 72-cell sample.
    def test_saved_sample_selection(self):
        # Read only the small completed coverage table.
        with (audit.DEFAULT_RUN/'coverage.csv').open() as stream:
            # Keep the exact January–June requested universe.
            rows=[r for r in csv.DictReader(stream) if '2026-01-01'<=r['date']<='2026-06-30']
        # Select the proposed pilot without loading any source parquet.
        pilot,full=audit.select_jobs(rows)
        # Reversing input row order must not change the pilot choices.
        reverse,_=audit.select_jobs(list(reversed(rows)))
        # Require deterministic selection rather than arrival-order dependence.
        self.assertEqual(pilot,reverse)
        # Require exactly one cell for each of twelve tickers in every month.
        self.assertEqual(len(pilot),72)
        # Require unique stock-day jobs.
        self.assertEqual(len({(j['symbol'],j['date']) for j in pilot}),72)
        # Preserve all requested full-period stock-days for the runtime denominator.
        self.assertEqual(len(full),13560)
        # Keep the three cheap-tick exceptions explicitly represented.
        self.assertTrue({'KEL','PIBTL','TPL'}<={j['symbol'] for j in pilot})
        # Require twelve observations in every selected month.
        self.assertEqual(set(Counter(j['date'][:7] for j in pilot).values()),{12})
        # Cover each source-window-fragmentation stratum.
        self.assertEqual({j['stratum'] for j in pilot},{0,1,2})
        # Require both shortened and regular Friday observations in the real calendar.
        self.assertTrue({'short Friday','regular Friday'}<={j['session_group'] for j in pilot})
    # Validate the new counters against the previously checked TELE source cell.
    def test_real_source_cell(self):
        # Import canonical paths without running strategy code.
        from config_pk import PARSED_ROOT
        # Read the historical channel-screen evidence.
        quality=audit.read_json(audit.DEFAULT_RUN/'channel_quality.json')
        # Freeze only the single bounded source date used by this regression.
        metadata={'2026-04-29':audit.partition_metadata(PARSED_ROOT,'2026-04-29')}
        # Use an ordinary dictionary for a serial single-cell smoke test.
        audit.initialize(str(audit.DEFAULT_RUN),quality,metadata,{})
        # Reconstruct source windows only, without any strategy arms.
        result=audit.audit_cell(dict(symbol='TELE',date='2026-04-29',stratum=0))
        # Require exact baseline reproduction and successful corrected planning.
        self.assertTrue(result['passed'],result.get('error'))
        # Require the new planner to adopt actual snapshots inside retained windows.
        self.assertGreater(result['snapshot_refreshes_retained'],0)
        # Require the counter to include the actual recovered missing-reference trade.
        self.assertGreaterEqual(result['missing_add_trades_retained'],1)
        # Retained events must be a subset of successfully applied attempt events.
        self.assertLessEqual(result['anonymous_unique_events_retained'],result['anonymous_unique_events_all_attempts'])
        # Verify both gross components reconcile to the measured net change.
        self.assertEqual(result['gained_ms']-result['lost_ms'],result['new_source_ms']-result['old_source_ms'])
# Run the bounded checks only when this test file is explicitly executed.
if __name__=='__main__':
    # Print exact checks performed and return nonzero on failure.
    unittest.main(verbosity=2)
