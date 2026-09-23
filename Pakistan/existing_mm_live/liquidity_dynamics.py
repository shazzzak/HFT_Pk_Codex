"""Causal rolling volumes with explicit order-ID attribution and reset boundaries."""
# Keep only events inside the requested rolling lookback.
from collections import deque
# Represent undefined ratios without inventing zero observations.
import math
# Track directly identified changes separately from reconstruction uncertainty.
class LiquidityDynamics:
    # Reset lookbacks at snapshots, invalid states and observation gaps.
    def __init__(self,window_ms=1000):
        # A nonpositive lookback cannot define observed flow rates.
        if window_ms<=0:
            # Reject invalid settings at initialization.
            raise ValueError('positive dynamic window required')
        # Preserve the declared rolling lookback.
        self.window_ms=window_ms
        # Hold timestamp, side, category and quantity records only.
        self.records=deque()
        # Track fixed-price execution recency without claiming causation.
        self.last_execution={}
        # No pre-session observation history can be assumed.
        self.since=None
    # Clear history without representing snapshots as economic order events.
    def reset(self,timestamp):
        # Snapshot replacement may remove orders for unknown reasons.
        self.records.clear()
        # Do not attribute additions after a snapshot to pre-snapshot removals.
        self.last_execution.clear()
        # Record the first timestamp of the new observable interval.
        self.since=timestamp
    # Capture an immutable copy of the directly referenced order before mutation.
    def before(self,book,kind,obj):
        # A snapshot is not an incrementally attributable order change.
        if kind=='S':
            # No order match is needed for a state replacement.
            return None
        # Use the trade's resting identifier or update identifier as appropriate.
        oid=getattr(obj,'rest_oid',None) if kind=='T' else getattr(obj,'order_id',None)
        # Synthetic price/aggregate placeholders do not identify an actual order.
        order=book.o.get(str(oid)) if oid is not None and not str(oid).startswith('__') else None
        # Preserve primitive values so later Book mutation cannot change this evidence.
        return (str(oid),order.side,order.price,order.qty) if order is not None else None
    # Attribute only observed primitive order changes after the real Book transition.
    def after(self,book,kind,obj,before,timestamp):
        # A snapshot starts a new attribution interval without counting adds/cancels.
        if kind=='S':
            # All quantities introduced by reconciliation remain unattributed.
            self.reset(timestamp)
            # There is no economic flow record for a snapshot replacement.
            return
        # Establish observation history even when starting from an incremental event.
        if self.since is None:
            # The collector will separately clear invalid book intervals.
            self.since=timestamp
        # Recover a directly referenced order's post-event state.
        after=book.o.get(before[0]) if before else None
        # Compute removed quantity only while the same order remains at the same price/side.
        remaining=max(0.,after.qty) if after and after.side==before[1] and after.price==before[2] else 0.
        # Direct trades can be bounded against known outstanding quantity.
        if kind=='T':
            # Unresolved trades stay separate from directly attributed consumption.
            side=before[1] if before else {'BUY':'SELL','SELL':'BUY'}.get(getattr(obj,'aggressor_side',None))
            # Unknown aggressor direction cannot be assigned to one book side.
            if side is None:
                # Preserve an explicit unattributed event counter.
                self.records.append((timestamp,'unknown','trade_count',1.))
                # Do not guess which side supplied liquidity.
                return
            # Never attribute more shares than were present at the matched order.
            matched=min(max(0.,before[3]-remaining),max(0.,float(obj.qty))) if before else 0.
            # Emit directly identified execution volume separately.
            self.records.append((timestamp,side,'executed',matched))
            # Only known executed orders identify a consumed price level.
            if matched>0:
                # Later same-price additions can be measured without inferring order intent.
                self.last_execution[(side,before[2])]=timestamp
            # Excess or unmatched quantity remains visible as unresolved flow.
            self.records.append((timestamp,side,'unresolved_trade',max(0.,float(obj.qty)-matched)))
        # Treat cancellation messages separately from replacement/addition messages.
        elif getattr(obj,'event',None)!='ORDER_ADD':
            # Known cancellations use the order's actual old side, not unreliable message price.
            if before:
                # An already-consumed quantity cannot be removed twice.
                self.records.append((timestamp,before[1],'cancelled',max(0.,before[3]-remaining)))
            # Fallback hidden-level removal is not a directly identified cancellation.
            else:
                # Count unresolved events without assigning their size as proven cancellation.
                self.records.append((timestamp,'unknown','cancel_count',1.))
        # A known add can overwrite an old identifier and therefore represent amendment.
        else:
            # Inspect the actual resulting order rather than trusting malformed message fields.
            current=book.o.get(str(getattr(obj,'order_id',None)))
            # A rejected/missing addition supplies no identified replenishment.
            if current is None:
                # Record explicit unresolved addition coverage.
                self.records.append((timestamp,'unknown','add_count',1.))
            # Brand-new IDs provide an observed addition, not proof of lasting liquidity.
            elif before is None:
                # Only positive post-event quantity contributes to addition volume.
                self.records.append((timestamp,current.side,'added',max(0.,current.qty)))
                # Require a recent directly identified execution at this same side and price.
                prior=self.last_execution.get((current.side,current.price))
                # Equal-timestamp ordering follows the preserved event sequence, not invented sub-ms time.
                if prior is not None and 0<=timestamp-prior<self.window_ms:
                    # Record observed replenishment, not a causal or survival claim.
                    self.records.append((timestamp,current.side,'added_at_consumed_price',max(0.,current.qty)))
            # Existing-ID changes are retained separately from new liquidity arrivals.
            else:
                # Record the removed component at its original side.
                self.records.append((timestamp,before[1],'amend_removed',max(0.,before[3]-remaining)))
                # Record the added component at its resulting side and price.
                self.records.append((timestamp,current.side,'amend_added',max(0.,current.qty-(before[3] if current.side==before[1] and current.price==before[2] else 0.))))
    # Expose causal window sums at the feature sample timestamp.
    def features(self,timestamp):
        # Drop events exactly at or before the open left boundary.
        while self.records and self.records[0][0]<=timestamp-self.window_ms:
            # Expired evidence cannot influence a current sample.
            self.records.popleft()
        # Expire price ancestry outside the same causal lookback.
        self.last_execution={key:time for key,time in self.last_execution.items() if time>timestamp-self.window_ms}
        # Record observation duration without interpreting partial history as a full window.
        age=max(0,timestamp-self.since) if self.since is not None else 0
        # Preserve both readiness and the requested lookback in each row.
        result=dict(dyn_window_ms=self.window_ms,dyn_observed_ms=min(age,self.window_ms),dyn_full_window=age>=self.window_ms)
        # Initialize explicit side/category volumes, including zero-activity windows.
        for side in ('BUY','SELL'):
            # Every category uses shares, not a mixture of shares and event counts.
            for category in ('added','added_at_consumed_price','executed','cancelled','unresolved_trade','amend_added','amend_removed'):
                # Zero is a valid absence of attributed activity within the observed interval.
                result[f'dyn_{side.lower()}_{category}_shares']=0.
        # Retain unknown-side event counts under distinct names and units.
        for category in ('trade_count','cancel_count','add_count'):
            # Unknown direction cannot be silently assigned to bid or ask flow.
            result['dyn_unknown_'+category]=0.
        # Accumulate only events available by the sample timestamp.
        for time,side,category,qty in self.records:
            # Future records must never enter the rolling statistics.
            if time>timestamp:
                # A caller violating causal ordering must fail visibly.
                raise ValueError('future dynamic event at sample')
            # Select a key whose suffix states shares or event counts unambiguously.
            key='dyn_unknown_'+category if side=='unknown' else f'dyn_{side.lower()}_{category}_shares'
            # Accumulate primitive evidence rather than inferred wall trust.
            result[key]+=qty
        # Ratios are undefined without directly identified consumption.
        for side in ('buy','sell'):
            # Only directly matched executions define the consumption denominator.
            consumption=result[f'dyn_{side}_executed_shares']
            # New-order additions are not guaranteed to replenish the same price.
            result[f'dyn_{side}_addition_to_execution']=result[f'dyn_{side}_added_shares']/consumption if consumption>0 else math.nan
        # This is an attribution layer, not a wall-survival or recovery-time estimator.
        return result
