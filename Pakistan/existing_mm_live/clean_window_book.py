# Preserve the simulator's established price-level and liquidation interfaces.
from mm_backtest import Book, Order
# Keep quantities and prices exact when checking source consistency.
from psx_reference_book import units
# Read nullable application pointers without float rounding.
from psx_reference_rows import reference

# Signal an unusable historical window without manufacturing a balancing order.
class WindowDataError(ValueError):
    # Preserve a normal exception interface for data-only window selection.
    pass

# Reconstruct only the price range actually disclosed by a checkpoint snapshot.
class WindowBook(Book):
    # Start from reported price levels, never aggregate quantity at an invented price.
    def __init__(self, snapshot, adds):
        # Preserve phase and circuit-band fields expected by the simulator.
        super().__init__()
        # Retain raw add-reference metadata, not future remaining quantities.
        self.adds=adds
        # Track whether source add records have already been processed in this window.
        self.seen_adds=set()
        # Remember every individually tracked identity even after full depletion.
        self.individual=set()
        # Copy only the observed phase and price bands.
        self.phase,self.limit_up,self.limit_dn=snapshot.phase,snapshot.limit_up,snapshot.limit_dn
        # Record the outer limit of each disclosed price range.
        self.boundary={}
        # Collect actual reported prices independently for each side.
        prices={'BUY':[],'SELL':[]}
        # Never collapse duplicate reported price rows silently.
        seen=set()
        # Seed actual snapshot prices and disclosed orders.
        for side,price,qty,ids,quantities in snapshot.levels:
            # Reject invalid prices, quantities and repeated levels.
            if side not in prices or units(price,10000,'price')<=0 or units(qty,100,'quantity')<=0 or (side,price) in seen:
                # Avoid building an apparently healthy book from malformed rows.
                raise WindowDataError('INVALID_SNAPSHOT_LEVEL')
            # Mark this price row as consumed once.
            seen.add((side,price))
            # Preserve the actual reported price range.
            prices[side].append(price)
            # Split disclosed source identities and their remaining quantities.
            names=ids.split('|') if ids else []
            # Missing detail represents anonymous quantity at this reported price only.
            amounts=quantities.split('|') if quantities else []
            # Require matching list lengths instead of truncating with zip.
            if len(names)!=len(amounts) or len(set(names))!=len(names):
                # Inconsistent order details invalidate the checkpoint.
                raise WindowDataError('INVALID_SNAPSHOT_ORDER_LIST')
            # Sum disclosed quantities in exact feed units.
            disclosed=0
            # Preserve each genuine disclosed order identity.
            for oid,amount in zip(names,amounts):
                # Require strictly positive remaining quantity and globally unique IDs.
                q=units(amount,100,'quantity')
                # Do not overwrite an order listed twice on either side.
                if not oid or oid in self.o or q<=0:
                    # Reject contradictory source rows.
                    raise WindowDataError('INVALID_SNAPSHOT_ORDER')
                # Store reported-price liquidity without moving it by a tick.
                self.o[oid]=Order(side,float(price),q/100)
                # An exhausted named order cannot consume an unrelated anonymous pool.
                self.individual.add(oid)
                # Count disclosed source quantity exactly.
                disclosed+=q
            # Compute only the undisclosed remainder at this actual reported price.
            residual=units(qty,100,'quantity')-disclosed
            # A negative remainder indicates contradictory source quantities.
            if residual<0:
                # Do not clamp away the disagreement.
                raise WindowDataError('SNAPSHOT_DETAILS_EXCEED_LEVEL')
            # Preserve anonymous quantity at its known price, not beyond the visible book.
            if residual:
                # This level pool existed before any of our new window orders.
                self.o[self.pool(side,price)]=Order(side,float(price),residual/100)
        # Both price ranges must be explicitly reported.
        if not prices['BUY'] or not prices['SELL']:
            # An absent side is not inferred from aggregate totals.
            raise WindowDataError('ONE_SIDED_CHECKPOINT')
        # Preserve only prices at or better than the worst disclosed bid/offer.
        self.boundary={'BUY':min(prices['BUY']),'SELL':max(prices['SELL'])}
        # Require a usable initial spread.
        self.check_touch()

    # Identify a price-level pool without confusing it with an exchange order ID.
    @staticmethod
    def pool(side,price):
        # Exact integer price units avoid unstable decimal string representations.
        return f'__WINDOW_POOL_{side}_{units(price,10000,"price")}'

    # Ask whether a source price belongs to the certified snapshot range.
    def inside(self,side,price):
        # A new better price is inside the known range; a deeper one remains unknown.
        return price>=self.boundary['BUY'] if side=='BUY' else price<=self.boundary['SELL']

    # Refuse quotes and executions once the known price range is exhausted or crossed.
    def check_touch(self):
        # Read only real snapshot/add prices retained in the range.
        bid,_,ask,_=self.bbo()
        # Empty or crossed sides end this retrospective usable-data window.
        if bid is None or ask is None or bid>=ask:
            # Do not create a replacement level from undisclosed aggregate volume.
            raise WindowDataError('INVALID_OR_EXHAUSTED_TOUCH')

    # Resolve reductions through original add references and original source prices.
    def resolve(self,row):
        # Retain both references for explicit rejection of auction events in these windows.
        refs=[(reference(getattr(row,'buy_ref',None)),'BUY'),(reference(getattr(row,'sell_ref',None)),'SELL')]
        # Keep only actual source pointers.
        refs=[(ref,side) for ref,side in refs if ref]
        # The research windows contain continuous trading, not auction matching.
        if len(refs)!=1:
            # End the window rather than infer the missing aggressor or auction state.
            raise WindowDataError('NOT_SINGLE_REFERENCE_CONTINUOUS_EVENT')
        # Read the referenced source side.
        ref,side=refs[0]
        # Look up original add metadata, never a guessed cancel price.
        source=self.adds.get(ref)
        # Future adds and missing references cannot be used to repair the window.
        if source is None or ref>=reference(row.appl_seq) or source['ts']>row.ts_exch or source['side']!=side:
            # Unpriced missing messages terminate this clean interval.
            raise WindowDataError('UNRESOLVED_SOURCE_REFERENCE')
        # Preserve actual source identity and price for queue updates.
        oid,price=source['oid'],source['price']
        # A known deeper order cannot change the certified price range.
        if not self.inside(side,price):
            # No invented deeper price participates in matching or liquidation.
            return None,side,price
        # Prefer a disclosed or post-checkpoint individual order when it is present.
        key=oid if oid in self.individual else self.pool(side,price)
        # The source event must be covered by known quantity at its actual price.
        if key not in self.o or units(row.qty,100,'quantity')>units(self.o[key].qty,100,'quantity') or units(row.qty,100,'quantity')<=0:
            # Stop on overconsumption rather than plug it with negative volume.
            raise WindowDataError('UNRESOLVED_OR_EXCESS_REDUCTION')
        # Return the exact queue identity used by the current book view.
        return key,side,price

    # Add known new volume only within the snapshot's certified range.
    def add(self,row):
        # Reject duplicate source adds in a window.
        seq=reference(row.appl_seq)
        # Require actual source metadata for this very add.
        source=self.adds.get(seq)
        # Never accept an unindexed or repeated add.
        if source is None or seq in self.seen_adds:
            # The data plan must end before this inconsistent record.
            raise WindowDataError('INVALID_OR_DUPLICATE_ADD')
        # Remember all adds, including ones outside the bounded depth view.
        self.seen_adds.add(seq)
        # Store only price ranges certified by the checkpoint.
        if self.inside(source['side'],source['price']):
            # A snapshot that already contains this later add is not aligned.
            if source['oid'] in self.o:
                # Do not overwrite its quantity and conceal the timing problem.
                raise WindowDataError('ADD_ALREADY_IN_CHECKPOINT')
            # Preserve the actual reported order quantity and price.
            self.o[source['oid']]=Order(source['side'],source['price'],float(row.qty))
            # Never reinterpret this order as checkpoint anonymous liquidity.
            self.individual.add(source['oid'])

    # Apply known partial and full reductions to one price/identity exactly once.
    def reduce(self,row):
        # Resolve the current queue pool before changing quantity.
        key,side,price=self.resolve(row)
        # Known deeper events do not affect this bounded executable view.
        if key is None:
            # Keep the unknown deeper range excluded.
            return
        # Reduce exact hundredth-share units rather than clamping binary drift.
        remaining=units(self.o[key].qty,100,'quantity')-units(row.qty,100,'quantity')
        # Remove fully exhausted source quantity.
        if remaining==0:
            # Empty levels must not remain executable.
            del self.o[key]
        # Preserve the real partial remainder.
        else:
            # Convert only at the existing simulator boundary.
            self.o[key].qty=remaining/100

    # Cancellations use the same exact quantity-reduction path as trades.
    def cancel(self,row):
        # Never remove an entire partially cancelled pool.
        self.reduce(row)

    # Trades consume only their resolved source price and quantity.
    def trade(self,row):
        # Do not manufacture an anonymous negative level for missing references.
        key,side,price=self.resolve(row)
        # Continuous matching must execute at the referenced resting price.
        if units(row.price,10000,'price')!=units(price,10000,'price') or (hasattr(row,'aggressor_side') and getattr(row,'aggressor_side')!=('SELL' if side=='BUY' else 'BUY')):
            # An inconsistent print cannot validate a simulated fill.
            raise WindowDataError('TRADE_REFERENCE_PRICE_MISMATCH')
        # Consume only the independently resolved resting quantity.
        self.reduce(row)

    # Subsequent status/timer observations do not replace depth or queue positions.
    def snapshot(self,snapshot):
        # Update only explicit phase information.
        if snapshot.phase is not None:
            # Price levels remain incremental within this window.
            self.phase=snapshot.phase
        # Keep actual observed circuit limits.
        if snapshot.limit_up is not None:
            # Never synthesize a default band.
            self.limit_up=snapshot.limit_up
        # Keep the lower limit independently.
        if snapshot.limit_dn is not None:
            # Preserve reported risk information without resetting depth.
            self.limit_dn=snapshot.limit_dn
