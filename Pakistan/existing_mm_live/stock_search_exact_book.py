# Retain the simulator's public book interface and priced liquidation routines.
from mm_backtest import Book, Order
# Use one exact reference implementation for every quantity mutation.
from psx_reference_book import ReferenceBook
# Normalize raw pointers without trusting the parser's historical ID lookup.
from psx_reference_rows import normalize

# Maintain reported orders only; snapshots cannot overwrite incremental depth.
class ExactBook(Book):
    # Create an empty reconstruction whose opening still requires external validation.
    def __init__(self):
        # Preserve phase, band and statistics fields expected by the simulator.
        super().__init__()
        # Store all genuine deep orders using exact quantities.
        self.reference=ReferenceBook()

    # Commit one validated incremental mutation to the simulator's priced view.
    def _apply(self,row,table):
        # Use raw application references rather than possibly blank resolved IDs.
        event=normalize(row,table)
        # Identify only orders affected by this event.
        keys=[(event.channel,event.sequence)] if event.kind=='ADD' else [(event.channel,r) for r in (event.buy_ref,event.sell_ref) if r>0]
        # Validate all legs before updating either representation.
        self.reference.apply(event)
        # Update only affected orders, preserving all other deep levels and queue IDs.
        for key in keys:
            # Use a stable reference identity shared with the event loader.
            oid=f'ref:{key[0]}:{key[1]}'
            # Retrieve the authoritative remaining order.
            order=self.reference.orders.get(key)
            # Remove only fully consumed orders.
            if order is None:
                # An exhausted reference must not remain as executable liquidity.
                self.o.pop(oid,None)
            # Preserve genuine remaining quantity at its reported price.
            else:
                # Convert exact feed units only at the simulator interface.
                self.o[oid]=Order(order.side,order.price/10000,order.quantity/100)

    # Add one actually reported order.
    def add(self,row):
        # Reuse the common reference mutation.
        self._apply(row,'ob_updates')

    # Apply the actual cancellation quantity, retaining partial remainders.
    def cancel(self,row):
        # Never delete an entire partially cancelled order.
        self._apply(row,'ob_updates')

    # Consume both auction references or the single continuous passive reference.
    def trade(self,row):
        # Never invent anonymous negative volume to balance unknown trades.
        self._apply(row,'trades')

    # Receive phase and bands without importing unsynchronised snapshot liquidity.
    def snapshot(self,snapshot):
        # Preserve explicit market phase observations.
        if snapshot.phase is not None:
            # Phase timing remains the loader's separately qualified observation clock.
            self.phase=snapshot.phase
        # Retain a reported upper band without inventing one.
        if snapshot.limit_up is not None:
            # Update only the observed value.
            self.limit_up=snapshot.limit_up
        # Retain a reported lower band independently.
        if snapshot.limit_dn is not None:
            # Leave all incremental orders untouched.
            self.limit_dn=snapshot.limit_dn

    # Never expose the stale priced mirror after its reference ledger failed.
    def _require_healthy(self):
        # Delegate invalidity reporting to the exact ledger itself.
        if self.reference.failed:
            # A read cannot silently turn incomplete data back into liquidity.
            self.reference.depth('', 'BUY')

    # Protect touch reads after reconstruction errors.
    def bbo(self):
        # Reject a damaged ledger before exposing any prices.
        self._require_healthy()
        # Reuse the ordinary priced-level calculation.
        return super().bbo()

    # Protect queue-position reads after reconstruction errors.
    def qty_at(self,side,price):
        # Reject stale queue membership after a failed reduction.
        self._require_healthy()
        # Return only genuine current orders at this price.
        return super().qty_at(side,price)

    # Protect weighted signals and execution depth after reconstruction errors.
    def ranked_depth(self,n=10,include_deep=False):
        # Refuse to value or execute against an invalid book.
        self._require_healthy()
        # No synthetic aggregates exist in this implementation.
        return super().ranked_depth(n,include_deep=include_deep)

    # Protect closing inventory valuation from stale prices after a gap.
    def liquidation_value(self,*args,**kwargs):
        # Unknown closing liquidity must not become an apparent realized profit.
        self._require_healthy()
        # Preserve the original fee and price-walk arithmetic on valid books.
        return super().liquidation_value(*args,**kwargs)
