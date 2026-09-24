# Match closing trades to the oldest opening trades.
from collections import deque
# Reject missing or invalid fill values.
import math
# Use exactly the simulator's fee function.
from mm_backtest import fee_for

# Keep every bucket in a fixed, readable order.
BUCKETS = ('first15', 'middle', 'preclose45', 'last15')
# Separate additive money columns from diagnostic quantities.
MONEY = ('capture_pkr', 'fifo_markout_pkr', 'closing_walk_pnl_pkr', 'residual_inventory_pnl_pkr', 'unattributed_trade_pnl_pkr', 'fees_pkr', 'net_pkr')

# Classify the actual opening fill's exchange timestamp, never the exit timestamp.
def bucket_at(t, segments):
    # Read the day's first open and final close.
    start, end = segments[0][0], segments[-1][1]
    # Keep opening-period inventory in its own cohort.
    if t < start + 15 * 60000:
        # Name the first fifteen clock minutes.
        return 'first15'
    # Give the last fifteen minutes priority over the preceding window.
    if t >= end - 15 * 60000:
        # Name the final fifteen clock minutes.
        return 'last15'
    # The preceding forty-five minutes start sixty minutes before close.
    if t >= end - 60 * 60000:
        # Do not accidentally make this a thirty-minute bucket.
        return 'preclose45'
    # Include every other opening fill, including the afternoon reopening.
    return 'middle'

# Require a positive finite two-sided midpoint.
def valid_mid(value):
    # Accept missing midpoints only as explicitly unattributed evidence.
    return value is not None and math.isfinite(float(value)) and float(value) > 0

# Reconcile all cash flows with FIFO opening-cohort attribution.
def attribute(records, segments, expected_net):
    # Retain unmatched opening lots in actual execution order.
    lots = deque()
    # Emit even buckets containing no trades.
    totals = {b: dict(bucket=b, **{k: 0.0 for k in MONEY}, closed_qty=0.0, closing_walk_qty=0.0, residual_qty=0.0, unattributed_qty=0.0) for b in BUCKETS}
    # Independently accumulate the simulator's flat-start cash ledger.
    cash = 0.0
    # Follow original append order; sorting equal timestamps would change FIFO.
    for index, fill in enumerate(records):
        # Decode the actual direction of this execution or residual mark.
        side = fill['side']
        # Reject unknown directions instead of assuming sells.
        if side not in ('BUY', 'SELL'):
            # Expose corrupt fill evidence.
            raise ValueError('Unknown fill side')
        # Use positive direction for acquired long inventory.
        sign = 1 if side == 'BUY' else -1
        # Read execution price and full executed quantity.
        price, remaining = float(fill['px']), float(fill['qty'])
        # Refuse invalid prices or quantities.
        if not all(math.isfinite(v) and v > 0 for v in (price, remaining)):
            # A corrupt fill must fail the cell.
            raise ValueError('Invalid fill price/quantity')
        # Distinguish actual closing executions from hypothetical residual marks.
        reason = fill['reason']
        # Residual marking is not an execution and incurs no exit fee.
        fee_unit = 0.0 if reason == 'liq_residual' else fee_for(price, remaining) / remaining
        # Reconstruct cash independently of component formulas.
        cash += (-sign * price - fee_unit) * remaining
        # Read a recorded pre-fill midpoint without an equity-table lookup.
        mid = fill.get('mid0')
        # Close the oldest opposing inventory first.
        while remaining > 1e-9 and lots and lots[0]['sign'] != sign:
            # Inspect the first opening lot.
            lot = lots[0]
            # Allocate this closing fill across partial lots as necessary.
            qty = min(remaining, lot['qty'])
            # Attribute all components to the opening trade's time bucket.
            row = totals[lot['bucket']]
            # Compute the full gross round-trip amount independently.
            gross = lot['sign'] * (price - lot['price']) * qty
            # Charge both legs in the opening bucket.
            fees = (lot['fee_unit'] + fee_unit) * qty
            # Retain the actual end-of-day book walk separately.
            if reason == 'liq':
                # Include the full result from entry to closing execution.
                row['closing_walk_pnl_pkr'] += gross
                # Count shares still held for the closing walk.
                row['closing_walk_qty'] += qty
            # Show shares the closing book could not absorb separately.
            elif reason == 'liq_residual':
                # Include entry-to-haircut-mark P&L, not a balancing plug.
                row['residual_inventory_pnl_pkr'] += gross
                # Preserve the actual unfilled quantity.
                row['residual_qty'] += qty
            # Split ordinary completed trades only with valid recorded midpoints.
            elif valid_mid(lot['mid']) and valid_mid(mid):
                # Capture includes both entry and exit execution relative to their mids.
                row['capture_pkr'] += lot['sign'] * (lot['mid'] - lot['price'] + price - mid) * qty
                # Markout is the midprice movement while this lot was held.
                row['fifo_markout_pkr'] += lot['sign'] * (mid - lot['mid']) * qty
                # Count normally completed shares.
                row['closed_qty'] += qty
            # Never invent a midpoint just to make the decomposition look complete.
            else:
                # Preserve known profit with an explicit incomplete-attribution label.
                row['unattributed_trade_pnl_pkr'] += gross
                # A nonzero count prevents claiming complete capture attribution.
                row['unattributed_qty'] += qty
            # Accumulate positive costs independently of profit.
            row['fees_pkr'] += fees
            # Accumulate net from the direct entry/exit formula.
            row['net_pkr'] += gross - fees
            # Consume the matched amount on both sides.
            lot['qty'], remaining = lot['qty'] - qty, remaining - qty
            # Remove completely closed lots.
            if lot['qty'] <= 1e-9:
                # Advance FIFO to the next opening lot.
                lots.popleft()
        # Any excess ordinary execution opens new inventory after a reversal.
        if remaining > 1e-9:
            # A closing mark may never open a new position.
            if reason in ('liq', 'liq_residual'):
                # Fail instead of hiding an over-liquidation error.
                raise ValueError('Closing execution exceeds open inventory')
            # Store the opening fill's own timestamp and prorated fee.
            lots.append(dict(sign=sign, qty=remaining, price=price, mid=mid, fee_unit=fee_unit, bucket=bucket_at(float(fill['t']), segments), opening_index=index))
    # Every remaining position must have an explicit closing walk or residual mark.
    if lots:
        # Do not silently value unsold stock at zero.
        raise ValueError('Open inventory lacks end-of-day valuation')
    # Allow only numerical floating-point noise in rupee reconciliation.
    tolerance = max(0.01, abs(expected_net) * 1e-10)
    # Recheck each bucket using its independent components.
    for row in totals.values():
        # Reconstruct net without reading the stored net column.
        rebuilt = sum(row[k] for k in MONEY if k not in ('fees_pkr', 'net_pkr')) - row['fees_pkr']
        # Reject a component or allocation mistake.
        if abs(rebuilt - row['net_pkr']) > tolerance:
            # Identify a broken local accounting identity.
            raise ValueError('Bucket components do not reconcile')
        # Positive inventory loss means a loss; a gain appears as a negative loss.
        row['inventory_loss_pkr'] = -row['closing_walk_pnl_pkr'] - row['residual_inventory_pnl_pkr']
    # Check both component accounting and direct cash against engine equity.
    if abs(sum(r['net_pkr'] for r in totals.values()) - expected_net) > tolerance or abs(cash - expected_net) > tolerance:
        # Never let a failed reconciliation enter aggregate profit reports.
        raise ValueError(f'FIFO/cash reconciliation failed: cash={cash}, engine={expected_net}')
    # Keep four compact rows per stock-day/configuration.
    return list(totals.values())
