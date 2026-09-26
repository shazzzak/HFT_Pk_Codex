# Keep decimal source quantities exact before converting to integer units.
from decimal import Decimal, InvalidOperation
# Store explicit immutable normalized events and mutable live orders.
from dataclasses import dataclass
# Aggregate only reported order prices.
from collections import defaultdict

# Report incomplete or inconsistent source data without inventing book state.
class ReconstructionError(ValueError):
    # Retain a machine-readable cause alongside the explanatory message.
    def __init__(self, code, sequence, message):
        # Initialize the standard exception text.
        super().__init__(f'{code} at sequence {sequence}: {message}')
        # Preserve the source-quality failure category.
        self.code = code
        # Preserve the event that cannot be applied safely.
        self.sequence = sequence

# Normalize quantities into exact feed units.
def units(value, scale, name):
    # Reject nonnumeric and nonfinite input explicitly.
    try:
        # Decimal(str(...)) retains the source's displayed decimal value.
        result = Decimal(str(value)) * scale
        # Require an exactly representable integer unit count.
        if not result.is_finite() or result != result.to_integral_value():
            # Never silently round a malformed source field.
            raise ValueError(f'Invalid {name}: {value}')
        # Return exact integer arithmetic for all state changes.
        return int(result)
    # Convert decimal parsing errors to an explicit data error.
    except InvalidOperation as error:
        # Preserve the field identity for diagnosis.
        raise ValueError(f'Invalid {name}: {value}') from error

# Use application references as identity; optional exchange order IDs are evidence.
@dataclass(frozen=True)
class Event:
    # Channel numbers scope application sequence numbers.
    channel: int
    # Sequence determines event order independently of snapshot timestamps.
    sequence: int
    # Symbol is required even though references are channel-wide.
    symbol: str
    # Kind is ADD, TRADE or CANCEL.
    kind: str
    # Quantity is in hundredths of a share as specified by the feed format.
    quantity: int
    # An add's price is in ten-thousandths of a currency unit.
    price: int = 0
    # An add's side is BUY or SELL.
    side: str = ''
    # A trade/cancel references the original buy add when nonzero.
    buy_ref: int = 0
    # A trade/cancel references the original sell add when nonzero.
    sell_ref: int = 0
    # Optional exchange identifier is retained without replacing sequence identity.
    order_id: str = ''

# Track exact remaining quantity at the original reported price.
@dataclass
class LiveOrder:
    # Retain symbol identity for channel-wide processing.
    symbol: str
    # Retain resting side.
    side: str
    # Retain integer reported price.
    price: int
    # Retain integer unexecuted and uncancelled quantity.
    quantity: int
    # Retain the optional exchange order ID for comparison with snapshots.
    order_id: str

# Validate a complete, ordered single-channel stream before trusting its book.
class SequenceGate:
    # Require an explicit channel and the exchange's daily start sequence.
    def __init__(self, channel, first_sequence=1):
        # Retain the channel boundary.
        self.channel = channel
        # Expect the first daily sequence unless a verified checkpoint is supplied.
        self.next_sequence = first_sequence
        # Retain only the last event for immediate duplicate recognition.
        self.last_event = None
        # A damaged stream cannot silently become valid after another event.
        self.failed = False

    # Accept the next event, rejecting gaps and conflicting or old replays.
    def accept(self, event):
        # Require a new verified stream after any sequencing failure.
        if self.failed:
            # Do not recover implicitly from missing data.
            raise ReconstructionError('STREAM_INVALID', event.sequence, 'A prior sequence failure requires recovery')
        # Verify the source channel before sequence comparisons.
        if event.channel != self.channel:
            # Permanently invalidate this gate until explicitly rebuilt.
            self.failed = True
            # Avoid cross-channel reference collisions.
            raise ReconstructionError('WRONG_CHANNEL', event.sequence, str(event.channel))
        # Ignore an identical immediate retransmission without consuming it twice.
        if event == self.last_event:
            # Inform the caller that no book mutation is required.
            return False
        # Every next application event must be present and in order.
        if event.sequence != self.next_sequence:
            # Freeze this invalid stream.
            self.failed = True
            # Distinguish missing records from conflicting/late replay.
            raise ReconstructionError('SEQUENCE_GAP' if event.sequence > self.next_sequence else 'SEQUENCE_REPLAY', event.sequence, f'Expected {self.next_sequence}')
        # Commit the sequence advance only after validation.
        self.next_sequence += 1
        # Retain the exact prior event for duplicate detection.
        self.last_event = event
        # Permit one book application.
        return True

# Reconstruct order depth from incrementals without overwriting it with snapshots.
class ReferenceBook:
    # Separate reference integrity from full-channel sequencing for filtered tests.
    def __init__(self):
        # Live orders are keyed by channel and original add sequence.
        self.orders = {}
        # Keep actual price-level totals in exact units.
        self.levels = defaultdict(dict)
        # Detect duplicate add references even after an order has completed.
        self.added = set()
        # Retain last applied sequence per channel to prevent backdated mutation.
        self.last_sequence = {}
        # Preserve invalidity after a failed event.
        self.failed = False

    # Validate and apply an entire event atomically, including auction pairs.
    def apply(self, event):
        # Do not continue an incomplete reconstruction after a prior error.
        if self.failed:
            # Require a new verified rebuild rather than guessed recovery.
            raise ReconstructionError('BOOK_INVALID', event.sequence, 'A prior event failed')
        # Convert all validation failures into persistent invalidity.
        try:
            # Validate identity, ordering and quantity before changing any state.
            if any(type(v) is not int for v in (event.channel, event.sequence, event.quantity, event.price, event.buy_ref, event.sell_ref)) or event.buy_ref < 0 or event.sell_ref < 0 or event.channel <= 0 or event.sequence <= self.last_sequence.get(event.channel, 0) or not event.symbol or event.quantity <= 0:
                # Reject zero/negative quantity and out-of-order filtered events.
                raise ReconstructionError('INVALID_EVENT', event.sequence, 'Invalid identity, ordering or quantity')
            # An add establishes a new authoritative reference.
            if event.kind == 'ADD':
                # Scope the original add sequence by its channel.
                key = (event.channel, event.sequence)
                # Require a real reported price and recognized resting side.
                if event.price <= 0 or event.side not in ('BUY', 'SELL') or key in self.added:
                    # Reject malformed adds without overwriting live state.
                    raise ReconstructionError('INVALID_ADD', event.sequence, 'Invalid price, side or repeated reference')
                # Record the actual order with its exact remaining quantity.
                order = LiveOrder(event.symbol, event.side, event.price, event.quantity, event.order_id)
                # Add it to the authoritative order map.
                self.orders[key] = order
                # Remember this reference even after its final reduction.
                self.added.add(key)
                # Update the corresponding exact price total.
                self._change(order, event.quantity)
            # Executions and cancellations reduce existing references.
            elif event.kind in ('TRADE', 'CANCEL'):
                # Preserve both legs when a trade references two resting orders.
                references = [(event.buy_ref, 'BUY'), (event.sell_ref, 'SELL')]
                # Keep only actual nonzero references.
                references = [(ref, side) for ref, side in references if ref > 0]
                # A cancel must identify one order; a trade identifies one or two.
                if not references or (event.kind == 'CANCEL' and len(references) != 1):
                    # Do not guess which price or side to deplete.
                    raise ReconstructionError('INVALID_REFERENCES', event.sequence, 'Missing or ambiguous references')
                # Validate every leg before applying any reduction.
                pending = []
                # Check reference chronology, symbol, side and sufficient quantity.
                for reference, side in references:
                    # Resolve only an already observed add on this channel.
                    key = (event.channel, reference)
                    # Retrieve the actual live order.
                    order = self.orders.get(key)
                    # Never manufacture an order from a trade price or snapshot.
                    if reference >= event.sequence or order is None:
                        # Report the exact missing or already completed add reference.
                        raise ReconstructionError('UNKNOWN_REFERENCE', event.sequence, f'Missing live add {reference}')
                    # Prevent applying another symbol's or opposite side's order.
                    if order.symbol != event.symbol or order.side != side:
                        # Reject a corrupt cross-symbol reference.
                        raise ReconstructionError('REFERENCE_MISMATCH', event.sequence, str(reference))
                    # An oversized reduction indicates corrupt or incomplete state.
                    if event.quantity > order.quantity:
                        # Never clamp an impossible reduction into an apparently valid book.
                        raise ReconstructionError('EXCESS_REDUCTION', event.sequence, str(reference))
                    # Retain this validated leg for the atomic commit.
                    pending.append((key, order))
                # Commit every leg after all checks pass.
                for key, order in pending:
                    # Reduce the priced total by exactly the event quantity.
                    self._change(order, -event.quantity)
                    # Reduce the corresponding order quantity by the same amount.
                    order.quantity -= event.quantity
                    # Remove fully consumed orders only.
                    if order.quantity == 0:
                        # Keep partially cancelled or partially filled orders live.
                        del self.orders[key]
            # Do not silently ignore an unsupported book-changing event.
            else:
                # Surface unknown semantics for explicit implementation.
                raise ReconstructionError('UNKNOWN_KIND', event.sequence, event.kind)
            # Commit the applied sequence after the event succeeds.
            self.last_sequence[event.channel] = event.sequence
        # Keep the book unusable after an invalid event.
        except ReconstructionError:
            # Preserve the failure instead of resuming with incomplete depth.
            self.failed = True
            # Return the precise original cause to the caller.
            raise

    # Update a reported price without adding any synthetic level.
    def _change(self, order, delta):
        # Select the symbol's actual resting side.
        levels = self.levels[(order.symbol, order.side)]
        # Apply exact integer quantity arithmetic.
        levels[order.price] = levels.get(order.price, 0) + delta
        # Empty price levels no longer belong to the book.
        if levels[order.price] == 0:
            # Remove exhausted prices rather than assigning nearby liquidity.
            del levels[order.price]

    # Return the entire known priced ladder in exchange units.
    def depth(self, symbol, side, limit=None):
        # Never expose a corrupted book as usable depth.
        if self.failed:
            # Downstream quoting must stop on incomplete reconstruction.
            raise ReconstructionError('BOOK_INVALID', 0, 'Cannot read invalid depth')
        # Use highest bids and lowest offers first.
        return sorted(self.levels.get((symbol, side), {}).items(), reverse=side == 'BUY')[:limit]

    # Return best reported prices without interpolating a missing side.
    def bbo(self, symbol):
        # Read at most one bid price.
        bids = self.depth(symbol, 'BUY', 1)
        # Read at most one offer price.
        asks = self.depth(symbol, 'SELL', 1)
        # Preserve absent sides explicitly.
        return (bids[0] if bids else None, asks[0] if asks else None)
