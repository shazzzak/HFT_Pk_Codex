"""Price-level geometry diagnostics; no changes to strategy or execution semantics."""
# Validate finite price/quantity inputs and compute causal control transforms.
import math
# Operate on arrays of event times without row-wise future searches.
import numpy as np
# Produce compatible feature-store columns and explicit missing labels.
import pandas as pd

# Fixed, preregistered visible price-level windows.
DEPTHS = (3, 5, 10, None)


# Measure positive, strictly ranked known-price levels; missing depth is not neutral.
def density_features(bids, asks, tick, pd_decay_k=0.1):
    # Tick size is an explicit instrument/research input, never inferred from gaps.
    if not math.isfinite(tick) or tick <= 0:
        # Invalid units must fail instead of producing plausible-looking ratios.
        raise ValueError("positive finite tick required")
    # A zero decay is the ordinary same-depth OBI control; negative decay is invalid.
    if not math.isfinite(pd_decay_k) or pd_decay_k < 0:
        # Refuse invalid or deeper-liquidity-amplifying decay values.
        raise ValueError("finite nonnegative price-distance decay required")
    # Validate each side without sorting away bad upstream ordering.
    for levels, sign in ((bids, -1), (asks, 1)):
        # Every supplied level must represent positive known-price liquidity.
        for i, (price, qty) in enumerate(levels):
            # Reject non-finite, negative or off-grid levels rather than rounding them.
            if not math.isfinite(price) or not math.isfinite(qty) or price <= 0 or qty <= 0 or abs(price/tick-round(price/tick)) > 1e-5:
                # Research on an incorrect tick schedule is invalid.
                raise ValueError("invalid/off-grid level; verify tick configuration")
            # Duplicate prices must have been aggregated by Book.ranked_depth().
            if i and sign*(price-levels[i-1][0]) <= 0:
                # Unsorted or duplicate levels do not define a usable span.
                raise ValueError("levels must be distinct and best-first")
    # Crossed/locked states are excluded before directional geometry is computed.
    valid_touch = bool(bids and asks and bids[0][0] < asks[0][0])
    # Retain level availability independently of any feature value.
    result = dict(visible_bid_levels=len(bids), visible_ask_levels=len(asks), pd_decay_k=pd_decay_k)
    # The conventional top-of-book OBI remains a separate baseline feature.
    result["obi_1"] = (bids[0][1]-asks[0][1])/(bids[0][1]+asks[0][1]) if valid_touch else np.nan
    # Calculate every requested horizon from the same aggregated book snapshot.
    for depth in DEPTHS:
        # Use stable flat column names in parquet and research outputs.
        key = str(depth) if depth is not None else "all"
        # Select quoted price levels, not tick slots or individual orders.
        bid, ask = bids[:depth], asks[:depth]
        # Fixed-depth measures require the full N on both sides; all requires two.
        valid = valid_touch and len(bid) >= (depth or 2) and len(ask) >= (depth or 2)
        # Keep insufficient-depth coverage auditable rather than substituting zero.
        result[f"valid_{key}"] = valid
        # Keep observed counts even when a fixed-depth measure is unavailable.
        result[f"n_bid_{key}"], result[f"n_ask_{key}"] = len(bid), len(ask)
        # Declare the same schema for valid and invalid states.
        names = ("pd_obi", "pd_volume_bid", "pd_volume_ask", "obi", "span_bid", "span_ask", "density_bid", "density_ask", "density", "density_inclusive", "geometry", "geometry_inclusive", "occupancy_bid", "occupancy_ask", "log_depth", "log_span")
        # HHI measures price-level concentration, not order-count concentration.
        names += tuple(f"{metric}_{side}" for side in ("bid","ask") for metric in ("hhi","hhi_normalized","effective_levels","touch_share","hhi_touch","hhi_distance_ticks","log_hhi_distance","largest_share","largest_distance_ticks"))
        # Undefined geometry must not masquerade as a balanced book.
        for name in names:
            # NaN propagates to explicit model coverage exclusions.
            result[f"{name}_{key}"] = np.nan
        # No arithmetic is meaningful without the declared depth coverage.
        if not valid:
            # Preserve the missingness flags and move to the next depth.
            continue
        # Compute raw quantity concentration independently of exponential proximity weighting.
        for side,levels,sign in (("bid",bid,-1),("ask",ask,1)):
            # Normalize by the largest quantity first to avoid squaring large share counts.
            qscale = max(q for _,q in levels)
            # Sum scaled positive quantities using stable floating-point accumulation.
            total = math.fsum(q/qscale for _,q in levels)
            # Shares sum to one within this exact side/depth window.
            shares = [(q/qscale)/total for _,q in levels]
            # Distances are from the same-side best quote in actual tick units.
            distances = [sign*(price-levels[0][0])/tick for price,_ in levels]
            # Standard HHI is invariant to where the same quantities are placed.
            hhi = math.fsum(share*share for share in shares)
            # Normalize the equal-share lower bound to zero using the observed level count.
            normalized = (hhi-1/len(levels))/(1-1/len(levels))
            # Quantity-squared weighting locates the concentration without identifying a tied maximum.
            distance = math.fsum(share*share*d for share,d in zip(shares,distances))/hhi
            # For tied largest levels choose the nearest one under the best-first ordering.
            largest = max(range(len(levels)),key=lambda i:shares[i])
            # Preserve the standard HHI definition separately from normalized concentration.
            result[f"hhi_{side}_{key}"] = hhi
            # Clamp floating-point roundoff only; valid mathematical values lie in [0,1].
            result[f"hhi_normalized_{side}_{key}"] = min(1.0,max(0.0,normalized))
            # Reciprocal HHI expresses the equivalent number of equally sized price levels.
            result[f"effective_levels_{side}_{key}"] = 1/hhi
            # Touch share distinguishes near-touch quantity from distant walls.
            result[f"touch_share_{side}_{key}"] = shares[0]
            # The touch's squared share is its contribution to total HHI, not total HHI itself.
            result[f"hhi_touch_{side}_{key}"] = shares[0]*shares[0]
            # Keep a distance-aware concentration descriptor beside the permutation-invariant HHI.
            result[f"hhi_distance_ticks_{side}_{key}"] = distance
            # A logarithmic distance control limits raw scale differences across thin books.
            result[f"log_hhi_distance_{side}_{key}"] = math.log1p(distance)
            # Preserve the largest level's relative mass independently of its location.
            result[f"largest_share_{side}_{key}"] = shares[largest]
            # Record nearest-largest distance with deterministic tie handling.
            result[f"largest_distance_ticks_{side}_{key}"] = distances[largest]
        # Convert endpoint distances to ticks without integer rounding distortion.
        sb, sa = (bid[0][0]-bid[-1][0])/tick, (ask[-1][0]-ask[0][0])/tick
        # Match the requested guarded-span definition exactly.
        rb, ra = max(1.0, sb), max(1.0, sa)
        # Each volume is the unweighted sum at the selected price levels.
        vb, va = sum(q for _, q in bid), sum(q for _, q in ask)
        # Density has units of shares per tick of endpoint distance.
        db, da = vb/rb, va/ra
        # Inclusive width counts price-grid slots, not just intervening gaps.
        ib, ia = vb/(sb+1), va/(sa+1)
        # Discount bid volume by tick distance below the best known bid.
        wb = math.fsum(q*math.exp(-pd_decay_k*((bid[0][0]-price)/tick)) for price,q in bid)
        # Discount ask volume by tick distance above the best known ask.
        wa = math.fsum(q*math.exp(-pd_decay_k*((price-ask[0][0])/tick)) for price,q in ask)
        # Preserve both effective side volumes, not just their normalized ratio.
        result[f"pd_volume_bid_{key}"], result[f"pd_volume_ask_{key}"] = wb, wa
        # Scale before taking the ratio to avoid overflowing the sum of large volumes.
        scale = max(wb,wa)
        # Touch weights equal one, so positive valid touch quantities ensure a denominator.
        result[f"pd_obi_{key}"] = (wb/scale-wa/scale)/(wb/scale+wa/scale)
        # Retain the comparable same-depth volume-only imbalance.
        result[f"obi_{key}"] = (vb-va)/(vb+va)
        # Preserve side-specific spans and densities for attribution.
        result[f"span_bid_{key}"], result[f"span_ask_{key}"] = sb, sa
        # Expose quantities per distance rather than only a bounded composite.
        result[f"density_bid_{key}"], result[f"density_ask_{key}"] = db, da
        # Primary requested density asymmetry mixes volume and spacing.
        result[f"density_{key}"] = (db-da)/(db+da)
        # Sensitivity variant uses the inclusive occupied price-grid interval.
        result[f"density_inclusive_{key}"] = (ib-ia)/(ib+ia)
        # Positive geometry means asks span farther than bids, independently of volume.
        result[f"geometry_{key}"] = (ra-rb)/(ra+rb)
        # Inclusive geometry gives the exact decomposition of inclusive density too.
        result[f"geometry_inclusive_{key}"] = (sa-sb)/(sa+sb+2)
        # Occupancy distinguishes equally sparse books from equally dense ones.
        result[f"occupancy_bid_{key}"], result[f"occupancy_ask_{key}"] = len(bid)/(sb+1), len(ask)/(sa+1)
        # Symmetric magnitude controls separate scale from directional asymmetry.
        result[f"log_depth_{key}"], result[f"log_span_{key}"] = math.log1p(vb+va), math.log1p(sb+sa)
    # Return values without altering any book, strategy or order state.
    return result


# Assign geometry groups using fixed thresholds independent of future markout.
def geometry_group(values):
    # Treat near-equal spans as a separate identity/control subset.
    return np.select([np.abs(values) < 1e-9, values > 1/3, values < -1/3], ["equal_span", "ask_span_gt_2x", "bid_span_gt_2x"], default="moderate_asymmetry")


# Label exact clock horizons using the book state at/before the target, not a later tick.
def attach_density_labels(samples, timeline, horizons, max_age_ms):
    # Copy feature rows so labeling cannot mutate an already-frozen input frame.
    result = samples.copy()
    # Empty samples still represent missing coverage, not a successful signal test.
    if result.empty:
        # The caller records the missing symbol-day explicitly.
        return result
    # Same-millisecond events resolve to the last observed state at that timestamp.
    states = timeline.sort_values("ts", kind="stable").drop_duplicates("ts", keep="last")
    # Validate the label timeline before binary searching its states.
    if states.empty:
        # No reconstructed book state means no possible label.
        raise ValueError("empty label timeline")
    # Use numeric vectors for deterministic nearest-prior state lookup.
    times, mids = states.ts.to_numpy(), states.mid.to_numpy()
    # Epoch changes identify invalid books and phase interruptions.
    epochs, valid = states.epoch.to_numpy(), states.valid.to_numpy()
    # One explicit forward-return target per requested clock horizon.
    for horizon in horizons:
        # Short-term labels must have strictly positive forward time.
        if horizon <= 0:
            # Reject mislabeled contemporaneous correlations.
            raise ValueError("positive horizons required")
        # Target the same-day selected-clock instant, not N future events.
        targets = result.ts.to_numpy()+horizon
        # Lookup the latest reconstructed state no later than that target.
        indices = np.searchsorted(times, targets, side="right")-1
        # Clip only for safe array indexing; validity still rejects absent states.
        safe = np.clip(indices, 0, len(times)-1)
        # Labeling never bridges invalid phases, missing endpoints or stale marks.
        ok = (indices >= 0) & (targets <= times[-1]) & valid[safe] & (epochs[safe] == result.epoch.to_numpy()) & (targets-times[safe] <= max_age_ms)
        # Retain target-state age even for exclusions to diagnose thin-name coverage.
        result[f"label_age_{horizon}ms"] = targets-times[safe]
        # A zero return is a valid observation and must not be filtered away.
        result[f"markout_{horizon}ms_bps"] = np.where(ok, (mids[safe]/result.mid.to_numpy()-1)*10000, np.nan)
    # Forward labels remain diagnostic selected-clock returns, not live execution claims.
    return result
