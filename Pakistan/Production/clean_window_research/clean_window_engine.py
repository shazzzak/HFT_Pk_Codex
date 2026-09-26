# Preserve frozen parameter objects independently across arms.
import copy
# Merge source events with explicit cancellation and exit timers.
import heapq
# Round the arrival of a closing request conservatively to milliseconds.
import math
# Construct simulator-compatible timer rows.
from types import SimpleNamespace
# Keep the original message latency distribution and order representation.
from mm_backtest import LatencyModel, MyOrder
# Retain every common control and the existing fill observation hooks.
from stock_search_engine import Engine, make_strategy, ARMS
# Use actual reported-price reconstruction only.
from clean_window_book import WindowBook, WindowDataError
# Keep priority timestamps keyed by application generation, not a reusable exchange ID.
from clean_window_identity import order_key
# Reuse the engine configuration without changing installed defaults.
import run_legacy_mm as R
# Reconcile complete round trips by their original opening-time bucket.
from stock_search_accounting import attribute
# Keep the tested price-distance bid-share formula unchanged.
from stock_search_util import distance_share

# Distinguish an uncompleted experiment from a zero-profit window.
class WindowExitError(ValueError):
    # Retain normal diagnostic exception behavior.
    pass

# Replay one independent retrospective window, never carrying state across missing data.
class WindowEngine(Engine):
    # Preserve the normal strategy and execution setup with a bounded initial book.
    def __init__(self,strategy,cfg,window,adds):
        # Retain the existing risk reservation, matching and cash-accounting implementation.
        super().__init__(strategy,cfg)
        # Replace only the historical book with its reported-price checkpoint.
        self.book=WindowBook(window['checkpoint'],adds,asof=window['start'])
        # Preserve immutable arrival evidence for individually disclosed historical orders.
        self.source_times={order_key(source):source['ts'] for source in adds.values()}
        # Count market-picture replacements without treating them as account resets.
        self.snapshot_refreshes=0
        # Stop new quotes five seconds before the retrospectively known boundary.
        self.stop_ms=window['end']-5000
        # Send a closing request only after three seconds for ordinary quote cancellations.
        self.exit_ms=window['end']-2000
        # Preserve the source-selected boundary for pending-message checks.
        self.end_ms=window['end']
        # Track the acknowledgment of the explicit closing request.
        self.exit_ack=0
        # Keep a compact risk measure without retaining all marked states.
        self.peak,self.drawdown,self.max_inventory=0.0,0.0,0.0

    # Stop through the same cancellation path used by a halt, without deleting orders.
    def _requote(self,ts_know):
        # Do not acquire new inventory during the predeclared exit buffer.
        if ts_know>=self.stop_ms:
            # Existing orders remain executable until their simulated cancels arrive.
            return self._cancel_desired_quotes(ts_know)
        # Keep every original quoting and risk control before the buffer.
        return super()._requote(ts_know)

    # Partial historical cancellation removes only the actually cancelled quantity ahead.
    def _on_market_cancel(self,row):
        # Obtain the actual named order or reported-price anonymous pool.
        key,side,price=self.book.resolve(row)
        # A deeper-than-known cancellation cannot improve our visible queue.
        if key is None:
            # Preserve all current queue reservations.
            return
        # Preserve the immutable source row shared by every candidate.
        resolved=copy.copy(row)
        # Map its real reference onto the queue identity used by this checkpoint book.
        resolved.order_id=key
        # Apply the corrected common quantity-reduction logic exactly once.
        return super()._on_market_cancel(resolved)

    # Require exchange order state and acknowledgments to be settled before flat-start separation.
    def settled(self,ts):
        # Neither sent orders, amendment messages nor delayed acknowledgments may be erased.
        return not self._all_orders() and not self.pending and max(self.ack_until.values())<=ts

    # Refresh market liquidity while preserving cash, inventory and exchange-order state.
    def _refresh_market(self,snapshot,asof=None):
        # Replace the historical market picture atomically.
        self.book.refresh(snapshot,asof=asof)
        # Reconcile queue evidence for every live simulated order without replacing that order.
        for order in self._all_orders():
            # Pending arrivals have no exchange queue position yet.
            if order.t_active is None:
                # Their actual arrival will establish priority against the then-current book.
                continue
            # Do not infer an empty queue when a live quote lies beyond reported snapshot depth.
            if not self.book.inside(order.side,order.price):
                # Preserve exposure as an explicit unsuccessful replay instead of optimistic fills.
                raise WindowExitError('SNAPSHOT_QUEUE_OUTSIDE_REPORTED_DEPTH')
            # Read only quantity actually disclosed at this simulated order's price.
            reported=self.book.qty_at(order.side,order.price)
            # Keep proven earlier named orders ahead and conservatively place anonymous volume ahead.
            order.ahead={key:qty for key,qty in reported.items() if key not in self.source_times or self.source_times[key]<=order.t_active}
        # Count replacement only after every live queue has adequate evidence.
        self.snapshot_refreshes+=1

    # Execute the common source window without the legacy instant end-of-session cancellation.
    def replay(self,events,window,progress=None):
        # Preserve a heap so the closing IOC arrives at its sampled latency time.
        stream=list(events[window['left']:window['right']])
        # Replay precisely the snapshot replacements validated by the source planner.
        for index,checkpoint in enumerate(window.get('refreshes',[])):
            # Use causal availability rather than the snapshot's coarse book timestamp.
            stream.append((checkpoint['start'],0,index,'S',SimpleNamespace(ts_cap=checkpoint['start'],ts_exch=checkpoint['start'],snapshot=checkpoint['snapshot'])))
        # Add an initial observation and conservative periodic cancellation opportunities.
        timers=[window['start']]+list(range(self.stop_ms,self.end_ms+1,100))+[self.exit_ms,self.end_ms]
        # Put timer actions after source mutations with an identical timestamp.
        for index,ts in enumerate(sorted(set(timers))):
            # Timers have no fabricated market liquidity or volume.
            stream.append((ts,2,index,'N',SimpleNamespace(ts_cap=ts,ts_exch=ts)))
        # Preserve source ordering within equal exchange timestamps.
        heapq.heapify(stream)
        # Track the latest actually observed local receive time.
        know=window['start']
        # Send at most one window-end IOC after quotes have settled.
        sent_exit=False
        # Process every recorded mutation before declaring the window complete.
        while stream:
            # Read the next market event or explicit operator timer.
            ts,rank,sequence,kind,row=heapq.heappop(stream)
            # Land normal messages strictly before this exchange event.
            self._activate_until(ts)
            # Replace market depth without flattening inventory or erasing pending messages.
            if kind=='S':
                # Reconcile active queues against the newly observed market picture.
                self._refresh_market(row.snapshot,asof=ts)
            # Preserve price-time priority and fill observation on genuine trades.
            elif kind=='T':
                # Resolve queue identities from this window's current reported-price book.
                key,side,price=self.book.resolve(row)
                # Copy the source row so one arm cannot alter another's evidence.
                row=copy.copy(row)
                # Supply the genuine queue identity or the known-price initial pool.
                row.rest_oid=key
                # Match the historical trade against orders already sent.
                self._on_market_trade(row)
                # Apply the independently prevalidated source reduction.
                self.book.trade(row)
            # Preserve ordinary new-order and cancellation matching.
            elif kind=='U':
                # Adds may cross a previously resting simulated quote under the existing model.
                if row.event=='ORDER_ADD':
                    # Apply the established fill policy before the new order enters the book.
                    self._on_market_add(row)
                    # Add its actual priced quantity once.
                    self.book.add(row)
                # Cancels shrink only their resolved actual remaining quantity.
                else:
                    # Reduce the ahead quantity before mutating source liquidity.
                    self._on_market_cancel(row)
                    # Apply the exact partial or full book reduction.
                    self.book.cancel(row)
            # A separate IOC event represents a closing order's actual exchange arrival.
            elif kind=='X':
                # No quote or amendment may compete with the independent window closing request.
                if not self.settled(ts):
                    # Keep the entire matched window excluded rather than guessing remaining exposure.
                    raise WindowExitError('UNSETTLED_ORDERS_AT_EXIT')
                # Record only fills actually produced by this delayed closing request.
                before=len(self.fills)
                # Walk actual opposite-side reported prices and charge ordinary execution fees.
                self._taker_fill(ts,MyOrder('SELL' if self.pos>0 else 'BUY',0.0,abs(self.pos),{},ts))
                # Attribute these executions as window-end closing costs, not ordinary capture.
                for fill in self.fills[before:]:
                    # Reuse the established closing-execution accounting category.
                    fill['reason']='liq'
                # Unknown deeper liquidity cannot provide a fictional exit.
                if abs(self.pos)>1e-9:
                    # Keep remaining quantity in the diagnostic exception.
                    raise WindowExitError('INSUFFICIENT_REPORTED_EXIT_DEPTH: '+str(self.pos))
                # Require the closing acknowledgment to return before separation too.
                self.exit_ack=ts+self.lat.draw_ack()
            # Confirm the source view remains executable after each market mutation.
            self.book.check_touch()
            # Obtain the current midpoint without an equity-table interpolation.
            bid,bq,ask,aq=self.book.bbo()
            # Mark compact risk diagnostics at the currently observed book.
            mid=(bid+ask)/2
            # Preserve the largest marked decline including initial zero equity.
            marked=self.cash+self.pos*mid
            # Update the running peak.
            self.peak=max(self.peak,marked)
            # Update the largest peak-to-trough loss.
            self.drawdown=max(self.drawdown,self.peak-marked)
            # Track maximum held inventory separately from working-order reservations.
            self.max_inventory=max(self.max_inventory,abs(self.pos))
            # A timer advances time without inventing a market print or resetting queues.
            self.strat.observe(kind if kind in ('U','T') else 'S',row,ts,mid)
            # Keep the actual checkpoint's published circuit bands active.
            self.strat.limit_up,self.strat.limit_dn=self.book.limit_up,self.book.limit_dn
            # Never send an order before the triggering event was received.
            know=max(know,int(row.ts_cap),ts)
            # Reconcile quotes or send ordinary cancellations during the exit buffer.
            self._requote(know)
            # Send the explicitly costed closing order at the predeclared timer.
            if ts==self.exit_ms and kind=='N' and not sent_exit:
                # Retain a clear single-send invariant.
                sent_exit=True
                # Refuse to flatten while orders or acknowledgments remain uncertain.
                if not self.settled(ts):
                    # Do not clear the pending heap as an accounting shortcut.
                    raise WindowExitError('CANCELS_NOT_SETTLED_BEFORE_EXIT')
                # A flat window does not need a fictitious execution.
                if abs(self.pos)>1e-9:
                    # Draw the original unbounded latency distribution without clipping its tail.
                    arrival=math.ceil(ts+self.lat.draw_out())
                    # An IOC arriving outside observed data cannot be assigned an execution.
                    if arrival>=self.end_ms:
                        # Exclude this window across all arms with an explicit reason.
                        raise WindowExitError('EXIT_LATENCY_EXCEEDS_WINDOW')
                    # Insert the real exchange arrival after same-time historical mutations.
                    heapq.heappush(stream,(arrival,3,0,'X',SimpleNamespace(ts_cap=arrival,ts_exch=arrival)))
            # Publish lightweight worker progress through the caller's throttled callback.
            if progress is not None:
                # This callback does not print per event.
                progress((ts-window['start'])/(window['end']-window['start']))
        # Require genuine flat cash and settled messages, without the legacy EOD shortcut.
        if abs(self.pos)>1e-9 or not self.settled(self.end_ms) or self.exit_ack>self.end_ms:
            # Unknown exposure does not become zero profit or a balancing residual mark.
            raise WindowExitError('WINDOW_NOT_FULLY_CLOSED')
        # Reconcile direct cash to opening-cohort components using original day boundaries.
        buckets=attribute(self.fills,self.strat.session_segments,self.cash)
        # Missing midpoint attribution is a failed experiment, not a catch-all profit plug.
        if any(row['unattributed_qty'] for row in buckets):
            # Preserve the problematic fills in the worker's diagnostic archive.
            raise WindowExitError('UNATTRIBUTED_FILL')
        # Return only fully matched, costed window results.
        return dict(net_pkr=self.cash,fills=len(self.fills),max_inventory=self.max_inventory,drawdown_pkr=self.drawdown,buckets=buckets,capacity_reductions=self.capacity_reductions,snapshot_refreshes=self.snapshot_refreshes)

# Construct a candidate with identical baseline controls and a window-bounded capacity forecast.
def build_engine(job,arm,window,adds):
    # Preserve the original complete trading day's strategy calibration.
    segments=job['params']['session_segments']
    # Instantiate and assert every required user setting.
    strategy,effective=make_strategy(job,arm,(segments[0][0],segments[-1][1]))
    # Retain the original calibrated expected remaining-volume function.
    original=strategy._expvol_minsleft
    # Bound acquisition and unwind capacity to the known research exit-send time.
    def volume_left():
        # Preserve the actual strategy clock around a hypothetical boundary evaluation.
        now=strategy.now
        # Restore the clock even if calibration is malformed.
        try:
            # Calculate original remaining-day volume and minutes now.
            rate,left=original()
            # Calculate how much of that capacity lies after the research exit time.
            strategy.now=window['end']-2000
            # Do not borrow post-window trading volume to support window inventory.
            tail_rate,tail_left=original()
        # Keep the stateful strategy at its actual observation time.
        finally:
            # Restore the unmodified event clock.
            strategy.now=now
        # Subtract later capacity rather than relabeling a midday window as the market close.
        minutes=max(0.0,left-tail_left)
        # Retain the original volume-profile forecast inside the usable interval only.
        volume=max(0.0,(rate or 0.0)*left-(tail_rate or 0.0)*tail_left)
        # Preserve the original no-capacity convention.
        return (volume/minutes if minutes>0 and volume>0 else None),minutes
    # Both the inventory cap and unwind trigger use this same restricted forecast.
    strategy._expvol_minsleft=volume_left
    # Replace the per-symbol silence heuristic with source-wide channel-gap window selection.
    cfg=dict(copy.deepcopy(R.CFG),session=(window['start'],window['end']),latency_model=LatencyModel(seed=job['seed']),use_cfo=True,cross_on_arrival=True,log_equity=False,stale_feed_seconds=0)
    # Construct a fresh flat account and bounded source book for this arm and interval.
    engine=WindowEngine(strategy,cfg,window,adds)
    # Preserve a diagnostic of whether weighted depth was available.
    counts=dict(calls=0,valid=0,fallback=0)
    # Attach the unchanged exponential distance signal only to weighted arms.
    if ARMS[arm][2] is not None:
        # Freeze the requested number of quoted levels.
        depth=ARMS[arm][2]
        # Substitute the common imbalance signal used by all three protective decisions.
        def signal(bb,ba,original_share):
            # Read only current actual-priced levels.
            bids,asks=engine.book.ranked_depth(depth,include_deep=False)
            # Preserve the established decay coefficient and 0-to-1 normalization.
            value,valid=distance_share(bids,asks,bb,ba,strategy.tick,original_share,depth=depth,decay=.1)
            # Count every attempted signal calculation.
            counts['calls']+=1
            # Keep missing-depth fallback explicit.
            counts['valid' if valid else 'fallback']+=1
            # Supply the same share to throttle, defensive retreat and enabled queue skew.
            return value
        # This replaces signals in prior OBI names too, without enabling their queue skew.
        strategy.search_signal=signal
    # Retain effective parameters and substitution coverage for evidence.
    return engine,effective,counts
