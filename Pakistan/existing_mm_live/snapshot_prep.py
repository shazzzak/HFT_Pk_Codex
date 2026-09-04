# snapshot_prep.py -- pre-parse snapshot DataFrames into plain-Python structs ONCE
# at build time, so Book.snapshot() does ZERO pandas per call. This removes the
# 89%-of-runtime bottleneck (8,056 pandas-filtering calls -> 8,056 dict iterations).
# A PreparsedSnapshot holds exactly what Book.snapshot() needs, already extracted.
#
# Goes in existing_mm_live/ (imported by run_legacy_mm.build_events).

# lightweight container: no pandas, just native lists/scalars
from dataclasses import dataclass, field
# typed fields for clarity
from typing import Optional


# one pre-parsed snapshot: everything Book.snapshot() reads, already pulled out
@dataclass
class PreparsedSnapshot:
    # phase string (or None if not a string in the source) -- from rows_all["phase"].iloc[0]
    phase: Optional[str] = None
    # upper circuit-breaker price (or None) -- from UPPER_CIRCUIT_BREAKER row
    limit_up: Optional[float] = None
    # lower circuit-breaker price (or None) -- from LOWER_CIRCUIT_BREAKER row
    limit_dn: Optional[float] = None
    # visible BID/OFFER levels as native tuples:
    #   (side, px, qty, order_ids_str, order_qtys_str)
    # side is already "BUY"/"SELL"; strings kept raw for the split() in snapshot()
    levels: list = field(default_factory=list)
    # aggregate totals: {"BUY": agg_qty or None, "SELL": agg_qty or None}
    agg: dict = field(default_factory=lambda: {"BUY": None, "SELL": None})
    # True if there were any visible BID/OFFER rows (status-only msg if False)
    has_visible: bool = False


# convert ONE snapshot DataFrame (all rows sharing a msg_seq) to a PreparsedSnapshot.
# Called ONCE per snapshot at build time -- all pandas work happens here, not in run().
def prep_snapshot(rows_all):
    # the output container
    ps = PreparsedSnapshot()
    # nothing to do on an empty group
    if len(rows_all) == 0:
        return ps
    # --- phase (message-level): first row's phase, only if it's a string ---
    ph = rows_all["phase"].iloc[0]
    # match snapshot()'s isinstance(ph, str) guard exactly
    if isinstance(ph, str):
        ps.phase = ph
    # --- SINGLE PASS over the whole group: bucket every row by entry_type in
    # one itertuples() loop, replacing the five separate boolean-mask scans
    # (UPPER/LOWER breaker .loc, BID/OFFER .isin, two AGG ==) that each rescanned
    # the group. Same fields, same values, same guards -- just one pass. This is
    # the ~9s-per-build hot spot (8,056 calls/day); one pass instead of six.
    for r in rows_all.itertuples():
        # the row's entry type drives which bucket it lands in
        et = r.entry_type
        # visible book level: BID or OFFER -> append a native level tuple
        if et == "BID" or et == "OFFER":
            # side label from the entry type
            side = "BUY" if et == "BID" else "SELL"
            # keep order_ids as a raw string, guarding non-string exactly as before
            oids = r.order_ids if isinstance(r.order_ids, str) else ""
            # order_qtys only meaningful when order_ids is a real non-empty string
            oqty = str(r.order_qtys) if (isinstance(r.order_ids, str) and r.order_ids) else ""
            # native tuple: (side, px, qty, order_ids_str, order_qtys_str)
            ps.levels.append((side, float(r.px), float(r.qty), oids, oqty))
            # at least one visible level seen -> not a status-only message
            ps.has_visible = True
        # upper circuit-breaker price (first one wins, mirrors old .iloc[0])
        elif et == "UPPER_CIRCUIT_BREAKER":
            # only set if not already captured (first occurrence)
            if ps.limit_up is None:
                ps.limit_up = float(r.px)
        # lower circuit-breaker price (first one wins)
        elif et == "LOWER_CIRCUIT_BREAKER":
            # only set if not already captured
            if ps.limit_dn is None:
                ps.limit_dn = float(r.px)
        # aggregate BID total -> whole-side BUY total (first one wins)
        elif et == "AGG_BID":
            # only set if not already captured
            if ps.agg["BUY"] is None:
                ps.agg["BUY"] = float(r.qty)
        # aggregate OFFER total -> whole-side SELL total (first one wins)
        elif et == "AGG_OFFER":
            # only set if not already captured
            if ps.agg["SELL"] is None:
                ps.agg["SELL"] = float(r.qty)
    # done: a pandas-free struct
    return ps
