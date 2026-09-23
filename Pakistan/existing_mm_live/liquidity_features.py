"""Known-price cumulative depth and static, fully covered book walks."""
# Validate settings and accumulate displayed quantities accurately.
import math
# Preserve missing full-size prices rather than inventing deeper liquidity.
import numpy as np
# Declare small fixed physical-distance and order-size grids.
BANDS=(1,2,5,10,20)
# Keep clip multipliers explicit and comparable across runs.
MULTIPLIERS=(1,3,5)
# Walk a known-price side without extrapolating beyond available orders.
def walk(levels,quantity,tick,mid,direction):
    # Reject malformed target quantities before computing apparent cost.
    if not math.isfinite(quantity) or quantity<=0:
        # A zero-sized walk has no meaningful execution price.
        raise ValueError('positive finite walk quantity required')
    # Begin with the entire hypothetical order unfilled.
    remaining,paid,filled,last=quantity,0.,0.,None
    # Consume displayed levels in their existing best-first order.
    for price,qty in levels:
        # Never consume more than the remaining request.
        take=min(remaining,qty)
        # Stop without visiting unnecessary deep prices.
        if take<=0:
            # A completed order has no additional marginal execution.
            break
        # Accumulate only actually covered consideration and quantity.
        paid+=take*price
        # Preserve covered shares independently of full-size completion.
        filled+=take
        # Track unfilled shares explicitly.
        remaining-=take
        # Retain the last price reached by this static walk.
        last=price
    # Treat the order as complete only when all requested shares are covered.
    complete=remaining<=max(1e-9,quantity*1e-12)
    # Keep partial VWAP separately from the unavailable full-size VWAP.
    partial=paid/filled if filled else np.nan
    # A missing full-size execution cannot look like a cheap complete fill.
    full=partial if complete else np.nan
    # Export price impact relative to same-side touch and a common mid denominator.
    return dict(requested=quantity,filled=filled,unfilled=max(0.,remaining),coverage=filled/quantity,complete=complete,partial_vwap=partial,partial_impact_bps=direction*(partial-levels[0][0])/mid*10000 if filled and levels else np.nan,vwap=full,impact_bps=direction*(full-levels[0][0])/mid*10000 if complete and levels else np.nan,ticks_to_fill=direction*(last-levels[0][0])/tick if complete and levels else np.nan)
# Calculate these descriptors on the same valid known-price book as density.
def liquidity_features(bids,asks,tick,clip_shares):
    # The caller validates price ordering and tick grid through density_features.
    if not bids or not asks or tick<=0 or bids[0][0]>=asks[0][0]:
        # Reject ambiguous input instead of fabricating a reference mid.
        raise ValueError('valid two-sided known-price book required')
    # Freeze the sizing basis in every emitted feature row.
    result=dict(walk_clip_shares=float(clip_shares))
    # Preserve a common economic reference for both sides.
    mid=(bids[0][0]+asks[0][0])/2
    # Treat known-price extent as coverage evidence, not latent liquidity.
    for side,levels,sign in (('bid',bids,-1),('ask',asks,1)):
        # Distances are relative to each side's own best quote.
        distances=[sign*(price-levels[0][0])/tick for price,_ in levels]
        # Record the extent of the reconstructed priced book.
        result['known_extent_ticks_'+side]=distances[-1]
        # Preserve first-gap distance and empty-tick counts separately.
        gaps=[distances[i]-distances[i-1] for i in range(1,len(levels))]
        # Missing second levels make a first gap unavailable.
        result['first_gap_ticks_'+side]=gaps[0] if gaps else np.nan
        # Do not call the endpoint span itself an internal gap.
        result['max_gap_ticks_'+side]=max(gaps) if gaps else np.nan
        # Explicit empty grid slots differ from distance between occupied prices.
        result['max_empty_ticks_'+side]=max(0.,max(gaps)-1) if gaps else np.nan
        # Every physical band carries both observed quantity and extent coverage.
        for band in BANDS:
            # A tolerance accommodates floating-point tick representation only.
            selected=[(p,q) for (p,q),d in zip(levels,distances) if d<=band+1e-7]
            # Observed quantities are lower bounds when known extent does not reach the band.
            result[f'depth_observed_shares_{side}_{band}t']=math.fsum(q for _,q in selected)
            # Retain actual price-weighted notional rather than shares multiplied by mid.
            result[f'depth_observed_pkr_{side}_{band}t']=math.fsum(p*q for p,q in selected)
            # Extent coverage does not certify reconstruction or feed completeness.
            result[f'depth_extent_covers_{side}_{band}t']=distances[-1]>=band-1e-7
    # Walk each size independently with explicit incomplete-book flags.
    for multiple in MULTIPLIERS:
        # A fixed reference clip can later be supplied per production job.
        quantity=float(clip_shares)*multiple
        # Buy through asks; sell through bids using positive cost directions.
        buy=walk(asks,quantity,tick,mid,1)
        # No opaque aggregates are present in the supplied known-price side.
        sell=walk(bids,quantity,tick,mid,-1)
        # Preserve every diagnostic for both order directions.
        for side,values in (('buy',buy),('sell',sell)):
            # Stable flat names fit the existing parquet feature schema.
            for name,value in values.items():
                # Record partial and full-size results without conflating them.
                result[f'walk_{side}_{multiple}x_{name}']=value
        # Full-size spread is defined only if both directions are covered.
        spread=buy['vwap']-sell['vwap']
        # Retain both absolute and relative measures of book hollowness.
        result[f'walk_spread_{multiple}x_bps']=spread/mid*10000
        # A positive uncrossed touch guarantees a nonzero denominator.
        result[f'walk_spread_{multiple}x_to_touch']=spread/(asks[0][0]-bids[0][0])
    # Return descriptive static liquidity, not an executable cost forecast.
    return result
