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
    # --- circuit limits: pull the UPPER/LOWER breaker prices if present ---
    up = rows_all.loc[rows_all.entry_type == "UPPER_CIRCUIT_BREAKER", "px"]
    dn = rows_all.loc[rows_all.entry_type == "LOWER_CIRCUIT_BREAKER", "px"]
    # only set when a row exists (mirrors snapshot()'s len() guards)
    if len(up):
        ps.limit_up = float(up.iloc[0])
    if len(dn):
        ps.limit_dn = float(dn.iloc[0])
    # --- visible book levels: filter to BID/OFFER, extract to native tuples ---
    vis = rows_all[rows_all.entry_type.isin(["BID", "OFFER"])]
    # flag whether any visible levels exist (status-only message if not)
    ps.has_visible = len(vis) > 0
    # extract each visible level ONCE via itertuples (the only itertuples call now)
    for r in vis.itertuples():
        # map entry_type to our side label here, once
        side = "BUY" if r.entry_type == "BID" else "SELL"
        # keep order_ids / order_qtys as raw strings (snapshot() splits them);
        # guard non-string order_ids exactly as the original snapshot() did
        oids = r.order_ids if isinstance(r.order_ids, str) else ""
        # order_qtys only meaningful when order_ids is a real string
        oqty = str(r.order_qtys) if (isinstance(r.order_ids, str) and r.order_ids) else ""
        # native tuple: no pandas from here on
        ps.levels.append((side, float(r.px), float(r.qty), oids, oqty))
    # --- aggregate (L11) totals per side ---
    for side, agg_type in (("BUY", "AGG_BID"), ("SELL", "AGG_OFFER")):
        # the AGG row for this side, if present
        arow = rows_all[rows_all.entry_type == agg_type]
        # store the whole-side total (or leave None)
        if len(arow):
            ps.agg[side] = float(arow["qty"].iloc[0])
    # done: a pandas-free struct
    return ps
