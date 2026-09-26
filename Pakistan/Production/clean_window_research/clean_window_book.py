# Preserve the simulator's established price-level and liquidation interfaces.
from mm_backtest import Book, Order
# Keep quantities and prices exact when checking source consistency.
from psx_reference_book import units
# Read nullable application pointers without float rounding.
from psx_reference_rows import reference
# Resolve snapshot IDs without conflating earlier and later order generations.
from clean_window_identity import AddIndex, order_key

# Signal an unusable historical window without manufacturing a balancing order.
class WindowDataError(ValueError):
    # Preserve a normal exception interface for data-only window selection.
    pass

# Reconstruct only the price range actually disclosed by a checkpoint snapshot.
class WindowBook(Book):
    # Start from reported price levels, never aggregate quantity at an invented price.
    def __init__(self, snapshot, adds, asof=None):
        # Preserve phase and circuit-band fields expected by the simulator.
        super().__init__()
        # Retain raw add-reference metadata, not future remaining quantities.
        self.adds=adds
        # Real source snapshots require an explicit causal cutoff.
        if isinstance(adds,AddIndex) and asof is None:
            # Refuse a whole-day last-ID lookup that can import a later amendment.
            raise WindowDataError('MISSING_SNAPSHOT_CUTOFF')
        # Track whether source add records have already been processed in this window.
        self.seen_adds=set()
        # Remember every individually tracked identity even after full depletion.
        self.individual=set()
        # Learn missing-reference prices only from successfully applied earlier trades.
        self.observed_references={}
        # Count successful anonymous-reference reductions for source-quality audits.
        self.anonymous_reference_reductions=0
        # Retain simple fixture compatibility without changing real source generation lookup.
        fixture_sources={source['oid']:source for source in adds.values()} if not isinstance(adds,AddIndex) else None
        # Validate all reported snapshot identities even when their reference links are missing.
        snapshot_ids=set()
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
            # Retain only reference-addressable snapshot volume as individual orders.
            linked_quantity=0
            # Preserve each genuine disclosed order identity.
            for oid,amount in zip(names,amounts):
                # Require strictly positive remaining quantity and globally unique IDs.
                q=units(amount,100,'quantity')
                # Do not overwrite an order listed twice on either side.
                if not oid or oid in snapshot_ids or q<=0:
                    # Reject contradictory source rows.
                    raise WindowDataError('INVALID_SNAPSHOT_ORDER')
                # Preserve duplicate-ID validation across all levels and both sides.
                snapshot_ids.add(oid)
                # Resolve only the order generation already observed by this snapshot cutoff.
                source=adds.latest(oid,asof) if isinstance(adds,AddIndex) else fixture_sources.get(oid)
                # A missing or different-price generation cannot identify this snapshot order.
                linked=source is not None and source['side']==side and units(source['price'],10000,'price')==units(price,10000,'price') and (asof is None or source['ts']<=asof)
                # Preserve individual quantity only when this causal generation actually matches.
                if linked:
                    # Use its immutable application reference rather than the reusable exchange ID.
                    key=order_key(source)
                    # Store linked liquidity at its exact reported price.
                    self.o[key]=Order(side,float(price),q/100)
                    # Exhausted linked orders cannot spill into an anonymous pool.
                    self.individual.add(key)
                    # Exclude linked quantity from the pool calculated below.
                    linked_quantity+=q
                # Count disclosed source quantity exactly.
                disclosed+=q
            # Compute only the undisclosed remainder at this actual reported price.
            residual=units(qty,100,'quantity')-disclosed
            # A negative remainder indicates contradictory source quantities.
            if residual<0:
                # Do not clamp away the disagreement.
                raise WindowDataError('SNAPSHOT_DETAILS_EXCEED_LEVEL')
            # Pool both undisclosed quantity and disclosed IDs lacking reference mappings.
            residual=units(qty,100,'quantity')-linked_quantity
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
        # A future reference is contradictory even when the execution has a price.
        if ref>=reference(row.appl_seq):
            # Preserve strict application chronology for every reduction route.
            raise WindowDataError('UNRESOLVED_SOURCE_REFERENCE')
        # Resolve missing add metadata using causal price evidence at this checkpoint.
        if source is None:
            # Identify cancellations by their explicit source event label.
            cancellation=getattr(row,'event',None)=='CANCEL'
            # Retain only earlier successful trade observations for this reference.
            observed=self.observed_references.get(ref)
            # A cancellation without original add metadata needs prior price evidence.
            if cancellation:
                # Never choose an arbitrary anonymous level for an unpriced cancel.
                if observed is None:
                    # Preserve an actionable distinction from absent individual IDs.
                    raise WindowDataError('UNPRICED_ANONYMOUS_CANCEL')
                # Require a matching side and a strictly earlier application pointer.
                if observed['side']!=side or observed['sequence']>=reference(row.appl_seq) or observed['ts']>row.ts_exch or getattr(row,'side',side)!=side:
                    # Reject conflicting evidence without changing historical quantity.
                    raise WindowDataError('ANONYMOUS_REFERENCE_CONFLICT')
                # Reuse only the causally observed execution price.
                price=observed['price']
            # A trade independently reports its execution price and aggressor side.
            else:
                # Require an explicit trade-side field before interpreting a price.
                if getattr(row,'aggressor_side',None)!=('SELL' if side=='BUY' else 'BUY'):
                    # Do not infer a resting side from an ambiguous execution.
                    raise WindowDataError('TRADE_REFERENCE_PRICE_MISMATCH')
                # Validate a positive exact price before selecting its anonymous pool.
                if units(row.price,10000,'price')<=0:
                    # Refuse zero or negative execution prices.
                    raise WindowDataError('INVALID_ANONYMOUS_TRADE_PRICE')
                # Use the actual execution price, never the current best quote.
                price=float(row.price)
                # One resting reference cannot silently move to a different level.
                if observed is not None and (observed['side']!=side or units(observed['price'],10000,'price')!=units(price,10000,'price') or observed['sequence']>=reference(row.appl_seq) or observed['ts']>row.ts_exch):
                    # Surface contradictory repeated-reference evidence.
                    raise WindowDataError('ANONYMOUS_REFERENCE_CONFLICT')
            # Preserve known deeper events without introducing executable deep liquidity.
            if not self.inside(side,price):
                # No visible anonymous quantity is touched outside certified depth.
                return None,side,price
            # Missing identity can consume only the anonymous remainder at this price.
            key=self.pool(side,price)
        # Preserve original exact-order behavior when the original add is available.
        else:
            # Contradictory original metadata cannot fall back to anonymous volume.
            if source['ts']>row.ts_exch or source['side']!=side:
                # Reject inconsistent evidence before applying any reduction.
                raise WindowDataError('UNRESOLVED_SOURCE_REFERENCE')
            # A superseded generation cannot consume a newer snapshot's anonymous quantity.
            if isinstance(self.adds,AddIndex) and self.adds.latest(source['oid'],row.ts_exch,reference(row.appl_seq)) is not source:
                # Reject stale references even when reconstruction began after the replacement.
                raise WindowDataError('STALE_ORDER_GENERATION')
            # Preserve actual source identity and price for queue updates.
            oid,price=order_key(source),source['price']
            # A known deeper order cannot change the certified price range.
            if not self.inside(side,price):
                # No invented deeper price participates in matching or liquidation.
                return None,side,price
            # Prefer individually tracked orders, including known exhausted identities.
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
            if order_key(source) in self.o:
                # Do not overwrite its quantity and conceal the timing problem.
                raise WindowDataError('ADD_ALREADY_IN_CHECKPOINT')
            # Preserve the actual reported order quantity and price.
            self.o[order_key(source)]=Order(source['side'],source['price'],float(row.qty))
            # Never reinterpret this order as checkpoint anonymous liquidity.
            self.individual.add(order_key(source))

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
        # Preserve new reference evidence only after all trade checks and reduction pass.
        ref=reference(getattr(row,'buy_ref',None)) or reference(getattr(row,'sell_ref',None))
        # Keep learning local to this reconstructed window and absent-add references.
        if ref not in self.adds:
            # Record exact causal price evidence for subsequent unpriced cancellations.
            self.observed_references[ref]=dict(side=side,price=price,sequence=reference(row.appl_seq),ts=row.ts_exch)
            # Count only actually applied reductions at a reported visible level.
            self.anonymous_reference_reductions+=int(key is not None)

    # Replace market quantities from a validated snapshot without resetting causal evidence.
    def refresh(self,snapshot,asof=None):
        # Validate a replacement before changing any existing book state.
        fresh=WindowBook(snapshot,self.adds,asof=asof)
        # Replace all historical orders, including stale prices absent from this snapshot.
        self.o=fresh.o
        # Replace the set of individually addressable snapshot orders.
        self.individual.update(fresh.individual)
        # Expand or contract certified depth to the newly reported levels.
        self.boundary=fresh.boundary
        # Adopt the snapshot's actual phase and circuit bands.
        self.phase,self.limit_up,self.limit_dn=fresh.phase,fresh.limit_up,fresh.limit_dn

    # Explicit replay snapshots replace market depth while account state lives in the engine.
    def snapshot(self,snapshot,asof=None):
        # Use the same atomic replacement as the source planner.
        self.refresh(snapshot,asof=asof)
