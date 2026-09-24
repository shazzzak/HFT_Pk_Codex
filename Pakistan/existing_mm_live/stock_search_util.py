# Resolve input and evidence paths.
from pathlib import Path
# Fingerprint immutable inputs.
import hashlib
# Serialize strict evidence.
import json
# Calculate finite distance weights.
import math

# Fingerprint a file without loading it all into memory.
def digest(path):
    # Initialize the content checksum.
    h = hashlib.sha256()
    # Read the file as bytes.
    with Path(path).open('rb') as stream:
        # Bound memory per read.
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            # Include every byte.
            h.update(chunk)
    # Return a portable checksum.
    return h.hexdigest()

# Publish evidence atomically.
def save(path, value):
    # Keep partially written JSON out of the report directory.
    tmp = Path(str(path) + '.tmp')
    # Refuse non-finite JSON numbers.
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    # Replace only this run's own evidence file.
    tmp.replace(path)

# Compute selected-depth price-distance bid share, falling back to the unchanged signal.
def distance_share(bids, asks, bb, ba, tick, fallback, depth=3, decay=0.1):
    # Reject unsupported configurations rather than silently changing the experiment.
    if depth not in (2, 3, 4, 5) or not math.isfinite(decay) or decay < 0:
        # Invalid parameters are programmer errors.
        raise ValueError('depth must be 2..5 and decay must be finite and nonnegative')
    # Require the declared number of known levels on both sides and a valid tick.
    if len(bids) < depth or len(asks) < depth or not math.isfinite(tick) or tick <= 0:
        # Preserve the original decision on incomplete books.
        return fallback, False
    # Require known-price depth to agree with the simulator's touch.
    if bb is None or ba is None or bb >= ba or abs(bids[0][0]-bb) > tick*1e-6 or abs(asks[0][0]-ba) > tick*1e-6:
        # Avoid substituting a different touch when opaque depth matters.
        return fallback, False
    # Accumulate one weighted volume for each side.
    volumes = []
    # Measure distances away from each side's own best quote.
    for levels, touch, sign in ((bids[:depth], bb, -1), (asks[:depth], ba, 1)):
        # Start the side's weighted sum.
        total = 0.0
        # Remember the prior level to reject malformed ordering.
        previous = -1.0
        # Examine exactly the declared number of quoted prices.
        for price, qty in levels:
            # Convert price distance to ticks.
            distance = sign * (price - touch) / tick
            # Require finite, positive quantities and distinct ordered tick levels.
            if not all(math.isfinite(x) for x in (price, qty, distance)) or qty <= 0 or distance <= previous or abs(distance-round(distance)) > 1e-5:
                # Retain the current strategy signal when data are invalid.
                return fallback, False
            # Discount quantities by actual price distance, with the declared decay (zero means equal weights).
            total += qty * math.exp(-decay * max(0.0, distance))
            # Advance the ordering check.
            previous = distance
        # Retain the side's weighted volume.
        volumes.append(total)
    # Reject overflow instead of emitting a non-finite signal.
    if not math.isfinite(sum(volumes)) or sum(volumes) <= 0:
        # Use the unchanged signal.
        return fallback, False
    # Preserve the existing strategy's zero-to-one bid-share convention.
    return volumes[0] / sum(volumes), True

