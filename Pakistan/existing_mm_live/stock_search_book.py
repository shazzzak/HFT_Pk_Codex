# Use the established book mutations and explicitly priced depth calculation.
from mm_backtest import Book

# Prevent unpriced whole-side statistics from becoming executable prices.
class PricedBook(Book):
    # Derive the touch from exactly the same priced levels used for execution.
    def bbo(self):
        # Exclude synthetic deep aggregate prices, retaining signed depletion.
        bids, asks = self.ranked_depth(n=1, include_deep=False)
        # Preserve the existing four-value contract, including absent sides.
        return (*(bids[0] if bids else (None, 0.0)), *(asks[0] if asks else (None, 0.0)))

    # Build queue-ahead quantities solely from orders with a reported price.
    def qty_at(self, side, price):
        # Keep disclosed and hidden quantities, excluding unpriced statistics.
        orders = {key: order.qty for key, order in self.o.items() if order.side == side and order.price == price and not key.startswith(('__NEG_', '__AGG_')) and order.qty > 0}
        # Preserve the existing anonymous-depletion convention at this price.
        depleted = -sum(order.qty for key, order in self.o.items() if key.startswith('__NEG_') and order.side == side and order.price == price)
        # Allocate anonymous depletion in the same insertion order as before.
        for key in list(orders):
            # Stop once all recorded depletion is reflected in queue quantities.
            if depleted <= 0:
                # Leave the remaining queue unchanged.
                break
            # Never deduct more than this order's remaining quantity.
            taken = min(orders[key], depleted)
            # Reduce the executable queue quantity.
            orders[key] -= taken
            # Carry only the unapplied depletion forward.
            depleted -= taken
            # Remove fully consumed queue entries.
            if orders[key] <= 0:
                # Do not return zero-sized orders as queue ahead.
                del orders[key]
        # Do not manufacture quantity when depletion exceeds visible depth.
        return orders
