# Normalize source fields without relying on parser-resolved order IDs.
from psx_reference_book import Event, units
# Detect nullable pandas scalars at the parquet boundary.
import pandas as pd

# Read a nonnegative integer reference without float coercion.
def reference(value):
    # Missing references mean this side is not linked by the execution.
    if value is None or pd.isna(value):
        # Preserve the feed's absent-reference convention.
        return 0
    # Require exact integer representation, including 18-digit sequence values.
    result = units(value, 1, 'reference')
    # Reject invalid negative references instead of treating them as absent.
    if result < 0:
        # Surface corrupted source fields before applying any event.
        raise ValueError('Negative reference')
    # Return the exact application reference.
    return result

# Convert one saved incremental record to the reference-book contract.
def normalize(row, table):
    # Decide event kind from the actual table and update type.
    kind = 'TRADE' if table == 'trades' else {'ORDER_ADD': 'ADD', 'CANCEL': 'CANCEL'}.get(row.event)
    # Reject unsupported records rather than silently skipping mutations.
    if kind is None:
        # Include the original event label in the error.
        raise ValueError(f'Unsupported update {row.event}')
    # An optional exchange ID is diagnostic only; raw references are authoritative.
    order_id = getattr(row, 'order_id', None)
    # Avoid the literal strings nan or <NA> as order identifiers.
    order_id = '' if order_id is None or pd.isna(order_id) else str(order_id)
    # Normalize every field used in state changes without rounding.
    return Event(channel=reference(row.channel), sequence=reference(row.appl_seq), symbol=str(row.symbol), kind=kind, quantity=units(row.qty, 100, 'quantity'), price=units(row.price, 10000, 'price') if kind == 'ADD' else 0, side=str(row.side) if kind == 'ADD' else '', buy_ref=reference(getattr(row, 'buy_ref', None)), sell_ref=reference(getattr(row, 'sell_ref', None)), order_id=order_id)
