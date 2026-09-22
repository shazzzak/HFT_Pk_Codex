"""Separate versioned density store; canonical feature files and trading code stay untouched."""
# Group selected-clock events sharing a millisecond before sampling their final state.
from itertools import groupby
# Use numerical vectors and explicit missing values.
import numpy as np
# Return ordinary parquet-ready feature and timeline frames.
import pandas as pd
# Reuse the exact historical order-book update implementation.
from mm_backtest import Book
# Share validated geometry and clock-horizon label definitions.
from book_density import density_features, attach_density_labels


# Derive a trusted quote while detecting artificial aggregate-price touch changes.
def research_touch(book):
    # Aggregate signed quantities at their known prices on each side.
    bids, asks = {}, {}
    # Retain the historical view only as a conservative contamination check.
    historical = book.bbo()
    # Preserve known-price hidden residuals and negative quantity netting.
    for key, order in book.o.items():
        # Opaque deep totals do not establish an observable price level.
        if key.startswith("__AGG_"):
            # Exclude synthetic prices from both predictive features and labels.
            continue
        # Select the same side convention as the shared historical book.
        side = bids if order.side == "BUY" else asks
        # Net all known-price quantities before choosing a best level.
        side[order.price] = side.get(order.price, 0.0) + order.qty
    # Fully consumed or negatively netted levels are not valid quotes.
    bids = {price:qty for price,qty in bids.items() if qty > 0}
    # Apply the same net-positive criterion on the ask side.
    asks = {price:qty for price,qty in asks.items() if qty > 0}
    # A missing known-price side cannot be replaced with an opaque aggregate.
    bid, ask = max(bids) if bids else None, min(asks) if asks else None
    # Preserve known-price quantities for OFI controls as well as labels.
    touch = (bid, bids.get(bid,0.0), ask, asks.get(ask,0.0))
    # Compare prices separately so offsetting quote errors cannot cancel in the mid.
    disagreement = any((a is None) != (b is None) or (a is not None and b is not None and abs(a-b)>1e-7) for a,b in ((bid,historical[0]),(ask,historical[2])))
    # An opaque quote disagreement is an exclusion, never a repaired market price.
    return touch, disagreement


# Collect fixed selected-clock samples without requesting or simulating any orders.
def collect(events, snapshots, tick, grid_ms, max_age_ms, horizons, progress=None, pd_decay_k=0.1):
    # Grid and freshness must be explicit positive clock quantities.
    if grid_ms <= 0 or max_age_ms <= 0:
        # Refuse an accidental every-event or unbounded-staleness experiment.
        raise ValueError("positive grid and maximum age required")
    # A new book isolates each instrument/day and avoids cross-session carry-in.
    book = Book()
    # Retain a compact event-mid timeline and sampled geometry separately.
    samples, timeline = [], []
    # Start without trusted phase, touch or prior-state information.
    previous, next_grid, epoch, previous_valid = None, None, 0, False
    # Keep causal return volatility and smoothed normalized OFI controls.
    old_touch, variance, flow = None, 0.0, 0.0
    # Preserve the input sequence rather than sorting away an upstream error.
    completed, last_time = 0, None
    # Count opaque-touch exclusions independently from ordinary phase/staleness loss.
    opaque_states, opaque_slots = 0, 0
    # Same-millisecond states are indivisible under this diagnostic clock convention.
    for timestamp, grouped in groupby(events, key=lambda event: event[0]):
        # Epoch-ms values must be monotonic under the established loader contract.
        timestamp = int(timestamp)
        # An out-of-order stream cannot yield a trustworthy markout timeline.
        if last_time is not None and timestamp <= last_time:
            # Fail instead of silently reordering exchange observations.
            raise ValueError("events not monotonic by selected timestamp")
        # Emit grid points strictly before this new event from the preceding book.
        if previous is not None:
            # Compute geometry once for all still-fresh grid points in this interval.
            cached = None
            # The current book has not yet consumed the new timestamp's events.
            while next_grid < timestamp:
                # A sample uses only information observed by its own grid instant.
                age = next_grid-previous["ts"]
                # Record fresh grid slots excluded because opaque liquidity changed the touch.
                if previous["opaque"] and age <= max_age_ms:
                    # This denominator exposes missing-side selection in thin names.
                    opaque_slots += 1
                # Invalid/old source quotes remain excluded and counted by coverage.
                if previous["valid"] and age <= max_age_ms:
                    # Avoid repeatedly sorting a book that has not changed.
                    if cached is None:
                        # Exclude unpriced deep residual aggregates from quoted levels.
                        bids, asks = book.ranked_depth(None, include_deep=False)
                        # Validate the explicit tick schedule before recording any feature.
                        cached = density_features(bids, asks, tick, pd_decay_k=pd_decay_k)
                        # The label touch must agree with the selected known-price book.
                        if not bids or not asks or abs(bids[0][0]-previous["bid"]) > 1e-7 or abs(asks[0][0]-previous["ask"]) > 1e-7:
                            # A deep aggregate cannot supply a fabricated top-of-book quote.
                            raise ValueError("known-price depth disagrees with historical touch")
                    # Keep prior-only feature state and freshness metadata in each sample.
                    samples.append(dict(cached, ts=next_grid, epoch=previous["epoch"], mid=previous["mid"], spread_bps=previous["spread_bps"], quote_age_ms=age, realized_vol_bps=np.sqrt(variance)*10000, ofi_ewma=flow))
                # Advance by clock time, independent of how many market events occurred.
                next_grid += grid_ms
        # Apply the same primitive market updates as Backtester.run, with no own orders.
        for _, _, _, kind, obj in grouped:
            # Snapshots replace historical state through the original book implementation.
            if kind == "S":
                # Use the loader's unique snapshot key, never a reused FIX sequence number.
                book.snapshot(snapshots[obj.snap_key])
            # Updates either add an order or remove historical liquidity.
            elif kind == "U":
                # Preserve original cancellation and negative-quantity netting rules.
                (book.add if obj.event == "ORDER_ADD" else book.cancel)(obj)
            # Trades consume the actual historical resting liquidity.
            elif kind == "T":
                # Do not substitute own hypothetical fills into market features.
                book.trade(obj)
            # New event types need an explicit book interpretation.
            else:
                # Unknown message semantics invalidate the feature stream.
                raise ValueError(f"unsupported event kind {kind}")
            # Count actual consumed events for the parent heartbeat.
            completed += 1
        # Use known-price touch for both event labels and predictive controls.
        (bid, bq, ask, aq), opaque = research_touch(book)
        # Keep contamination coverage explicit at full event-time resolution.
        opaque_states += int(opaque)
        # Treat missing phase as unknown, not as automatic continuous trading.
        valid = bool(not opaque and book.phase == "CONTINUOUS_AUCTION" and bid is not None and ask is not None and bq > 0 and aq > 0 and 0 < bid < ask)
        # Long observation gaps are conservatively censored, not called proven feed failures.
        gap = last_time is not None and timestamp-last_time > max_age_ms
        # Break label continuity on invalid books, phase interruptions or observation gaps.
        if not valid or not previous_valid or gap:
            # A later valid target cannot bridge an excluded interval.
            epoch += 1
            # Reset event-derived controls across discontinuities.
            old_touch, variance, flow = None, 0.0, 0.0
        # Update market controls only from valid contemporaneous states.
        if valid:
            # Arithmetic mid is the neutral reference for forward returns.
            mid = (bid+ask)/2
            # Compute causal volatility and standard touch OFI from the prior valid state.
            if old_touch is not None:
                # Recover the previous quote and its quantities.
                pb, pqb, pa, pqa = old_touch
                # Smooth squared mid changes without incorporating future returns.
                variance = .05*(mid/((pb+pa)/2)-1)**2+.95*variance
                # Standard L1 order-flow imbalance includes price improvement/worsening.
                ofi = (bq if bid >= pb else 0)-(pqb if bid <= pb else 0)-(aq if ask <= pa else 0)+(pqa if ask >= pa else 0)
                # Normalize by current/previous touch depth before smoothing.
                flow = .1*ofi/max(1.0,bq+aq+pqb+pqa)+.9*flow
            # Keep the last valid observation for the next causal control update.
            old_touch = (bid,bq,ask,aq)
        # Invalid states must also appear in the label timeline to prevent forward filling.
        else:
            # A missing mid is explicitly unpriceable rather than zero return.
            mid = np.nan
        # Preserve full-event label coverage separately from lower-frequency predictors.
        previous = dict(ts=timestamp, bid=bid, ask=ask, opaque=opaque, mid=mid, valid=valid, epoch=epoch, spread_bps=(ask-bid)/mid*10000 if valid else np.nan)
        # Only the required columns are retained on the dense timeline.
        timeline.append({key: previous[key] for key in ("ts","mid","valid","epoch")})
        # Initialize the next grid on the selected-clock boundary at/after the first event.
        if next_grid is None:
            # This point is emitted only after all updates at its time have been consumed.
            next_grid = ((timestamp+grid_ms-1)//grid_ms)*grid_ms
        # Retain state for the next interval and validity boundary.
        last_time, previous_valid = timestamp, valid
        # Shared progress publication is throttled by the caller, not printed per event.
        if progress is not None:
            # Expose actual completion while preserving the single-parent print policy.
            progress(completed, len(events))
    # Fixed-clock labels cannot extend beyond the final observed book event.
    frame = attach_density_labels(pd.DataFrame(samples), pd.DataFrame(timeline), horizons, max_age_ms) if samples else pd.DataFrame()
    # Coverage includes excluded source slots so sparse-name selection remains visible.
    slots = max(0, (last_time-1)//grid_ms-(int(events[0][0])+grid_ms-1)//grid_ms+1) if events else 0
    # Return data and reconstruction diagnostics without writing canonical stores.
    return frame, dict(opaque_touch_event_states=opaque_states, opaque_touch_grid_slots=opaque_slots, events=completed, candidate_clock_slots=int(slots), sampled_rows=len(frame), invalid_event_states=sum(not state["valid"] for state in timeline), clock="selected feed milliseconds; no execution simulation")
