# ============================================================================
# mm_backtest.py — Event-driven market-making backtester for PSX order data
# ============================================================================
#
# WHAT THIS FILE IS
# -----------------
# A backtest harness that replays one trading day of PSX order-level market
# data (three tables: incremental order updates, trades, and periodic
# exchange snapshots) and simulates a market-making strategy living inside
# that day: placing passive quotes, waiting in queue, getting filled by
# aggressive flow, paying fees, and carrying inventory.
#
# It is EVENT-DRIVEN: the strategy is re-evaluated after every single
# book-changing event (~4,500/day for MCB), not just on trades. A large
# order arriving at the best bid re-triggers quoting logic immediately,
# even if nothing trades for minutes.
#
# THE TWO-CLOCK PRINCIPLE (the core anti-cheat design)
# ----------------------------------------------------
# Every historical event carries two timestamps:
#
#   ts_exch — when it happened at the exchange's matching engine
#             (transact_time for updates/trades, orig_time for snapshots).
#             The book replay and all FILL decisions run on this clock,
#             because fills are determined by exchange reality.
#
#   ts_cap  — when our capture server actually RECEIVED the message.
#             The STRATEGY runs on this clock (specifically on its running
#             maximum, "knowledge time"), because a real trading system can
#             only react to information after it arrives — measured feed
#             latency on this data is ~80ms median with multi-second spikes.
#
# Our simulated orders cross between the clocks: a decision made at
# knowledge time k lands on the exchange timeline at k + latency_ms.
# Getting this boundary right is what separates a backtest from a
# lookahead-contaminated toy.
#
# COMPONENT MAP
# -------------
#   round_tick()      price-grid rounding that can never cross the market
#   Order             one resting order in the HISTORICAL book
#   Book              replay of the historical order book
#   MyOrder           one of OUR simulated orders (with queue + cancel state)
#   Backtester        the engine: latency plumbing, fill rules, accounting
#   NaiveSymmetricMM  baseline strategy (mid ± fixed half-spread)
#   load_events()     CSV/parquet -> unified, correctly ordered event stream
#
# KNOWN SIMPLIFICATIONS (flagged; see inline FLAGGED SIMPLIFICATION tags)
# -----------------------------------------------------------------------
#   1. Shadow fills: our fills do not alter subsequent history. The flow
#      that fills us also filled its real historical counterparty, and our
#      presence would in reality change others' behaviour. Every MM
#      backtest shares this counterfactual limit; mitigation = conservative
#      fill rules here + eventually a small live pilot.
#   2. Latency is a single symmetric constant (latency_ms) for both order
#      placement and cancels. Production: per-message stochastic draws,
#      separate out/ack legs, cancel-race asymmetry.
#   3. An arriving order that would cross the current touch is REJECTED
#      (post-only semantics) rather than executed as a taker.
#   4. Strategy reacts at the next market event after a fill; there is no
#      dedicated own-fill callback and no wall-clock timer events.
#      Exchange snapshots every ~5s bound quote staleness regardless.
#   5. On each snapshot, every order revealed at our price is assumed to be
#      AHEAD of us in queue (worst case), because snapshots do not disclose
#      arrival order.
#   6. End of day: final inventory is marked at the last mid; the closing
#      auction is not modelled.
# ============================================================================

import ast        # parses the stringified tuple in trades.resting_order_id
import heapq      # priority queue for our in-flight orders/cancels
from dataclasses import dataclass

import numpy as np
import pandas as pd

# PSX equity price grid: one paisa. Verified from the data (minimum positive
# increment across ~500 distinct prices in updates/trades/snapshots = 0.01).
TICK = 0.01

# ---- FEES: PSX schedule, charged PER SIDE on traded VALUE (price*qty) ------
# Source: PSX "PKR per 100,000" schedule converted to decimals.
FEE_COMMISSION_PCT = 0.0015      # broker commission 0.15% -- REPLACE with your
                                 #   negotiated HFT rate; this drives everything
FEE_SST_RATE       = 0.13        # sales tax ON THE COMMISSION (not on value):
                                 #   13% Sindh (SST) / 16% Punjab (PRA)
                                 #   -- TODO confirm your BROKER's province
FEE_PSX_LAGA_PCT   = 0.000035    # PSX trading fee: PKR 3.50 / 100,000 = 0.0035%
FEE_SECP_PCT       = 0.0000065   # SECP supervisory: PKR 0.65 / 100,000
FEE_IPF_PCT        = 0.0000062   # PSX regulatory (IPF): PKR 0.62084 / 100,000
                                 #   was scheduled to end Aug 2025 -- kept in as
                                 #   the conservative default; zero it once you
                                 #   confirm it is discontinued
FEE_CLEARING_PCT   = 0.00003     # NCCPL + CDC, ~0.003-0.005% per leg.
                                 #   FLAGGED ASSUMPTION: intraday-squared MM ->
                                 #   CDC delivery waived, NCCPL only -> low end
                                 #   0.003%. Positions held OVERNIGHT pay CDC
                                 #   delivery too; not modeled per-fill here.
FEE_PER_SHARE_FLAT = 0.0         # no flat per-share components in this schedule
FEE_MM_REBATE_PCT  = 0.0         # MM-program rebate; enter NEGATIVE when known
# CVT / WHT on turnover: abolished -> intentionally absent.

# Retail all-in per-side fee (commission dominates: ~17.73 bps/side).
FEE_TOTAL_RETAIL = (FEE_COMMISSION_PCT * (1.0 + FEE_SST_RATE)
                    + FEE_PSX_LAGA_PCT + FEE_SECP_PCT + FEE_IPF_PCT
                    + FEE_CLEARING_PCT + FEE_MM_REBATE_PCT)
# TREC own-account per-side fee: the broker COMMISSION (and its sales tax) drops
# to zero because you are your own broker; the regulatory stack (LAGA, SECP, IPF,
# clearing) remains. This sums to ~0.78 bps/side = ~1.6 bps round trip, which is
# exactly the rt_2p00 scenario the screen's net5 used -- so backtest P&L is now
# comparable to the screen. Flip USE_TREC_FEE to False to restore retail.
FEE_TOTAL_TREC = (FEE_PSX_LAGA_PCT + FEE_SECP_PCT + FEE_IPF_PCT
                  + FEE_CLEARING_PCT + FEE_MM_REBATE_PCT)
USE_TREC_FEE = True
FEE_TOTAL_PCT = FEE_TOTAL_TREC if USE_TREC_FEE else FEE_TOTAL_RETAIL


def fee_for(price, qty):
    """All-in per-side fee for a fill of `qty` shares at `price`."""
    return FEE_TOTAL_PCT * price * qty + FEE_PER_SHARE_FLAT * qty


def round_tick(p, side):
    """Snap a computed price onto the exchange tick grid, SAFELY.

    'Safely' means: rounding must never make a quote MORE aggressive than
    the strategy intended. So bids round DOWN (floor) and asks round UP
    (ceil). A bid computed at 405.117 becomes 405.11, never 405.12 —
    otherwise rounding alone could push a quote across the spread and turn
    an intended passive order into an accidental taker.
    """
    n = p / TICK
    return (np.floor(n) if side == "BUY" else np.ceil(n)) * TICK


@dataclass
class Order:
    """One resting order in the HISTORICAL book (not ours).

    A mutable struct: Book.trade() decrements qty in place as fills
    consume it. side is 'BUY' or 'SELL'; qty is the remaining quantity.
    """
    side: str
    price: float
    qty: float


class Book:
    """Replay of the historical order book from the three data feeds.

    State is ONE dict: order_id -> Order. Price levels are never stored;
    they are derived on demand (bbo, qty_at). Keeping order-level state is
    what makes exact queue-position tracking possible later.

    The comment is saying: we store every order separately by its ID (not lumped together by price)
     and recompute price totals whenever we need them — and we accept that recompute cost on purpose,
     because keeping orders individually distinguishable is the only way to know exactly where our
     own order sits in the fill queue.

    That trade-off — spend a little CPU re-summing levels, gain exact queue tracking — is a
    deliberate design choice that pays off precisely because you have order-level PSX data
    (with individual order IDs) rather than the level-aggregated feed most markets provide.


    Reconciliation model (validated earlier in this project: 97.9% of
    trades print inside the reconstructed pre-trade touch):
      * incremental adds/cancels/trades mutate the dict between snapshots;
      * each exchange snapshot (every ~5s) REPLACES the entire dict —
        full-state reconciliation, so any drift from unresolvable events
        is wiped at least every snapshot interval.
    """

    def __init__(self):
        self.o: dict[str, Order] = {}
        self.phase = None          # trading phase from latest snapshot
        self.limit_up = None       # individual-stock circuit limits (+/-10% prev close)
        self.limit_dn = None
        self.b2_hits = 0
        self.b2_ignored = 0

    # ---- incremental path: ob_updates rows ----
    def add(self, r):
        """ORDER_ADD: a new order enters the book (or overwrites same id)."""
        if pd.notna(r.order_id):                 # defensive: skip malformed rows
            self.o[str(r.order_id)] = Order(r.side, float(r.price), float(r.qty))

    def cancel(self, r):
        """Two-branch cancel, in priority order:

        BRANCH 1 — order_id found in the book: remove it exactly.
          The cancel's price field is IGNORED (it is wrong ~44% of the
          time; the id is authoritative). Handles 98.5% of cancels.

        BRANCH 2 — id unknown (pre-capture order or wiped by a snapshot
          replacement): if the cancel's (side, price) matches a HIDDEN
          lump ("__H_" synthetic order from undisclosed deep-level qty),
          decrement it: new_qty = qty - min(cancel_qty, qty), floor 0.
          Only hidden lumps are eligible — disclosed orders are never
          touched by price. Counted in b2_hits for auditing.

        Neither matches -> no-op (b2_ignored); next snapshot reconciles.
        """
        oid = str(r.order_id)
        if oid in self.o:  # BRANCH 1
            del self.o[oid]
            return
        if pd.notna(r.price) and pd.notna(r.side):  # BRANCH 2
            px = float(r.price)
            for k, o in self.o.items():
                if k.startswith("__H_") and o.side == r.side and o.price == px:
                    o.qty -= min(float(r.qty), o.qty)
                    if o.qty <= 0:
                        del self.o[k]
                    self.b2_hits += 1
                    return
        self.b2_ignored += 1

    # ---- trades consume resting liquidity ----
    def trade(self, r):
        """Apply one historical trade to the book: a fill REMOVES resting
        quantity. ob_updates carries only adds and cancels, so without this
        the book overstates depth between snapshots exactly when trading is
        most active.

        Two paths:
          exact    — the trade row identifies the resting order id
                     (r.rest_oid, pre-parsed in load_events): decrement it.
          fallback — id unknown (~2/3 of trades on the sample day): DON'T
                     guess which resting order was hit. Instead park the
                     traded qty as a signed "__NEG_{side}_{price}"
                     placeholder that SUBTRACTS from that level's net total.
                     This keeps the level aggregate exact for bbo/qty_at/obi
                     (which net by price and clamp at 0) while leaving the
                     real orders intact — so if one is later cancelled or
                     traded by id, branch 1 removes it cleanly with no
                     double-count. Placeholders are wiped at the next
                     snapshot (full replacement), bounding any residual
                     error to one snapshot interval at deep levels.
        """
        oid = r.rest_oid
        if oid and oid in self.o:                          # exact path
            self.o[oid].qty -= float(r.qty)
            if self.o[oid].qty <= 0:                       # fully consumed
                del self.o[oid]
            return
        passive = {"BUY": "SELL", "SELL": "BUY"}.get(r.aggressor_side)
        if passive is None:                                # AUCTION print:
            return                                         # snapshot reconciles it
        px = float(r.price)  # fallback path:
        key = f"__NEG_{passive}_{px}"  # signed placeholder
        if key in self.o:
            self.o[key].qty -= float(r.qty)
        else:
            self.o[key] = Order(passive, px, -float(r.qty))

    # NOTE: now receives a PreparsedSnapshot (from snapshot_prep.prep_snapshot),
    # NOT a DataFrame. All pandas parsing was moved to build time; this method
    # does only native-Python dict building. Behavior is identical -- same visible
    # levels, same __H_ hidden residual, same __AGG_ L11 lump, same state updates.
    def snapshot(self, ps):
        """Apply one pre-parsed exchange snapshot: FULL book replacement.
        `ps` is a PreparsedSnapshot with .phase, .limit_up/.limit_dn, .levels
        (list of (side, px, qty, order_ids_str, order_qtys_str)), .agg
        ({"BUY":total,"SELL":total}), and .has_visible. See snapshot_prep.py.
        """
        # --- market state (message-level): phase + circuit limits ---
        # set phase only when the source had a string phase
        if ps.phase is not None:
            self.phase = ps.phase
        # set circuit limits when present
        if ps.limit_up is not None:
            self.limit_up = ps.limit_up
        if ps.limit_dn is not None:
            self.limit_dn = ps.limit_dn
        # status-only message (no visible levels): update state, KEEP the book
        if not ps.has_visible:
            return
        # --- build the target book from scratch (full replacement) ---
        tgt = {}
        # iterate the pre-extracted native level tuples (no pandas)
        for side, px, qty, order_ids, order_qtys in ps.levels:
            # disclosed qty accumulator for this level
            disc = 0.0
            # if this level lists disclosed orders, add each one
            if order_ids:
                # split the parallel pipe-delimited id/qty strings
                for oid, q in zip(order_ids.split("|"), order_qtys.split("|")):
                    # each disclosed order becomes an Order; accumulate disclosed qty
                    tgt[oid] = Order(side, px, float(q));
                    disc += float(q)
            # hidden residual within this level (total qty beyond disclosed)
            if qty - disc > 0:
                # park the residual under a deterministic synthetic key
                tgt[f"__H_{side}_{px}"] = Order(side, px, qty - disc)
        # --- L11: deep residual beyond visible levels, from the AGG totals ---
        for side in ("BUY", "SELL"):
            # the whole-side aggregate total for this side (may be None)
            agg_qty = ps.agg.get(side)
            # no aggregate row -> no L11 lump
            if agg_qty is None:
                continue
            # sum of visible qty on this side already placed in tgt
            visible = sum(o.qty for o in tgt.values() if o.side == side and o.qty > 0)
            # L11 = whole-side total minus visible
            residual = agg_qty - visible
            # only park a lump when there is genuine deep residual
            if residual > 0:
                # prices currently on this side (to place the lump one tick past worst)
                prices = [o.price for o in tgt.values() if o.side == side]
                # need at least one visible price to anchor the lump
                if prices:
                    # one tick past the worst visible level (below best bid / above best ask)
                    edge = (min(prices) - TICK) if side == "BUY" else (max(prices) + TICK)
                    # single opaque aggregate lump, correct in total
                    tgt[f"__AGG_{side}"] = Order(side, edge, residual)
        # atomic full replacement: add-missing / remove-stale / correct-all-qty at once
        self.o = tgt

    # ---- derived views ----
    def bbo(self):
        """Best bid and best offer, each with its total quantity.

        Returns four values as (bb, bq, ba, aq):
          bb = best bid price   (highest price a buyer will pay)
          bq = total qty resting at that bid price
          ba = best offer price (lowest price a seller will accept)
          aq = total qty resting at that offer price

        Aggregates qty per price level across all entries — real orders,
        hidden "__H_" lumps, and signed "__NEG_" placeholders (which
        subtract already-traded unidentified qty). Levels whose net qty
        is <= 0 are treated as empty: a fully netted-out level cannot be
        the best price. Returns (None, 0.0, None, 0.0) components when a
        side has no positive level (pre-open, post-close).
        """
        bids, asks = {}, {}
        for o in self.o.values():  # net qty per level
            d = bids if o.side == "BUY" else asks
            d[o.price] = d.get(o.price, 0.0) + o.qty  # __NEG_ has negative qty -> subtracts
        bids = {p: q for p, q in bids.items() if q > 0}  # clamp: fully netted
        asks = {p: q for p, q in asks.items() if q > 0}  # level is GONE
        bb = max(bids) if bids else None
        ba = min(asks) if asks else None
        return bb, (bids[bb] if bb else 0.0), ba, (asks[ba] if ba else 0.0)


    def liquidation_value(self, pos, fee_fn=None):
        """Cash realised by flattening `pos` shares RIGHT NOW by walking the book.

        pos > 0 (long)  -> we SELL into the bids, best (highest) bid first.
        pos < 0 (short) -> we BUY from the asks, best (lowest) ask first.

        Returns (cash_realised, shares_unfilled, vwap).
          cash_realised  : signed cash from the liquidating trades, fees deducted.
          shares_unfilled: size the book could NOT absorb (book too thin). Genuinely
                           unpriceable residual -- do NOT silently mark it at mid.
          vwap           : volume-weighted avg price achieved, or None if nothing filled.

        Production alternative to marking inventory at mid. Mid-marking assumes an
        exit at the midpoint, which is impossible: you must cross the spread and eat
        successively worse levels. Includes __H_ hidden lumps; EXCLUDES the __AGG_
        deep residual (its price is a synthetic placeholder, not a tradable level).
        """
        # Nothing to liquidate -> zero cash, zero unfilled, no vwap, no fills.
        # (4th return is ADDITIVE: per-level liquidation fills for attribution.
        # Existing callers unpacking 3 values are unaffected -- see back-compat
        # test. Each entry is (price, qty) for one book level the walk consumed.)
        if pos == 0:
            return 0.0, 0.0, None, []
        # We hit the OPPOSITE side of the book: long sells into resting BUYs (bids),
        # short buys back from resting SELLs (asks).
        side_wanted = "BUY" if pos > 0 else "SELL"
        # Build net qty per price level on that side.
        levels = {}
        for k, o in self.o.items():
            # Skip the deep-residual lump: its price is a placeholder, not tradable.
            if k.startswith("__AGG_"):
                continue
            # Keep only orders on the side we are hitting.
            if o.side != side_wanted:
                continue
            # Accumulate qty at this price. __NEG_ entries are negative -> they net down.
            levels[o.price] = levels.get(o.price, 0.0) + o.qty
        # Clamp: a level netted to <= 0 has no real liquidity -> drop it.
        levels = {p: q for p, q in levels.items() if q > 0}
        # Order levels best-first: selling a long -> highest bid first (reverse=True);
        # buying back a short -> lowest ask first (reverse=False).
        ordered = sorted(levels.items(), reverse=(pos > 0))
        # Shares still to flatten.
        remaining = abs(float(pos))
        # Running signed cash from the liquidating trades.
        cash = 0.0
        # Shares actually filled (for the vwap).
        filled = 0.0
        # Sum of price*qty actually filled (for the vwap).
        notional = 0.0
        # ADDITIVE: per-level liquidation fills, (price, qty) each. This is the
        # itemization of the same walk -- it does NOT change cash/filled/vwap.
        liq_fills = []
        # Walk the levels best-first, consuming each until we are flat or the book runs out.
        for px, avail in ordered:
            # Fully flattened -> stop walking.
            if remaining <= 0:
                break
            # Take the smaller of this level's size or what we still need.
            take = min(avail, remaining)
            # Selling a long brings cash IN (+); buying back a short pays cash OUT (-).
            cash += (take * px) if pos > 0 else (-take * px)
            # Fees are a cost in either direction.
            if fee_fn is not None:
                cash -= fee_fn(px, take)
            # Track fill totals for the vwap.
            notional += take * px
            filled += take
            # ADDITIVE: record this level as a liquidation fill (price, qty).
            # Same px/take the cash line above used -- pure itemization.
            liq_fills.append((px, take))
            # Reduce what is left to flatten.
            remaining -= take
        # Volume-weighted average execution price, or None if the book had nothing.
        vwap = (notional / filled) if filled > 0 else None
        # remaining > 0 means the visible book could not absorb the full position.
        # 4th value (liq_fills) is ADDITIVE -- the per-level itemization.
        return cash, remaining, vwap, liq_fills

    def pinned(self):
        """Best bid at/above upper limit (limit-up) or best ask at/below lower
        (limit-down): one-sided market -- quoting into it is pure adverse
        selection, and prints there carry degenerate mids (exclude from
        markout labels via this flag)."""
        bb, _, ba, _ = self.bbo()
        if self.limit_up is not None and bb is not None and bb >= self.limit_up:
            return True
        if self.limit_dn is not None and ba is not None and ba <= self.limit_dn:
            return True
        return False

    def qty_at(self, side, price):
        """All resting orders at one (side, price): {order_id: qty}.

        Used the moment one of OUR orders arrives: this dict IS our queue —
        every order already at our price stands ahead of us under
        price-time priority. Includes synthetic "__H_" hidden entries,
        which is correct: hidden qty is genuinely ahead of us too.
        """
        d = {k: o.qty for k, o in self.o.items()
             if o.side == side and o.price == price and not k.startswith("__NEG_")}
        neg = sum(o.qty for k, o in self.o.items()
                  if k.startswith("__NEG_") and o.side == side and o.price == price)
        if neg < 0:  # only if there's traded-away qty to account for
            for k in list(d):  # walk the real orders in d
                if neg >= 0:  # once the full __NEG_ amount is absorbed,
                    break  # stop
                take = min(d[k], -neg)  # remove from THIS order: the smaller of
                #   its size or the remaining amount to absorb
                d[k] -= take  # shrink this order in our queue view
                neg += take  # move neg toward 0 by what we just absorbed
                if d[k] <= 0:  # order fully consumed ->
                    del d[k]  # remove it from the queue
        return d

    def obi(self, n=None, include_deep=False):
        """Order book imbalance (Qbid - Qask)/(Qbid + Qask) over the top
        n price levels per side (n=None -> all levels).

        Use obi(5) as the event-accurate signal (levels 1-5 are ~99%
        id-disclosed, exact tick by tick). Use obi(None) as the deep
        signal — snapshot-accurate, patched between snapshots by
        branch-2 cancels; treat it as the noisier of the two.
        Returns None when the book is empty.

        So you actually have three distinct signals from this one function:
        obi(5) (near touch, cleanest), obi(None) (all visible levels), and
        obi(include_deep=True) (whole book including the deep aggregate).
        """
        bids, asks = {}, {}  # two empty dicts to accumulate total qty per price: {price: qty}. bids = buy side, asks = sell side.
        for k, o in self.o.items():  # walk every entry in the book. k = order ID (dict key), o = the Order (side, price, qty).
            if k.startswith(
                    "__AGG_") and not include_deep:  # __AGG_ = the deep-book residual lump (liquidity beyond level 10).
                continue  # skip it UNLESS include_deep=True was requested; otherwise it would swamp the near-touch imbalance.
            d = bids if o.side == "BUY" else asks  # point d at the correct side's dict (buys -> bids, sells -> asks). d is a reference, not a copy.
            d[o.price] = d.get(o.price,
                               0.0) + o.qty  # add this entry's qty into its price level. .get(price, 0.0) = "total so far here, or 0 if new". __NEG_ has negative qty, so this SUBTRACTS for those entries (netting).
        bids = {p: q for p, q in bids.items() if q > 0}  # clamp: keep only price levels with positive net qty. A level netted to <=0 (e.g. __NEG_ cancelled the real orders) is dropped as empty.
        asks = {p: q for p, q in asks.items() if q > 0}  # same clamp for the ask side.
        bq = sum(q for _, q in sorted(bids.items(), reverse=True)[:n]) if bids else 0.0  # bid depth: sort levels HIGH-to-LOW (best bid first), take top n (n=None -> all), sum their qty. 0.0 if no bids (guards empty side).
        aq = sum(q for _, q in sorted(asks.items())[:n]) if asks else 0.0  # ask depth: sort levels LOW-to-HIGH (best ask first, no reverse), take top n, sum. 0.0 if no asks.
        return (bq - aq) / (bq + aq) if bq + aq > 0 else None  # imbalance = (bid depth - ask depth)/(total depth). Range [-1,+1]: +ve = buy pressure, -ve = sell pressure. None if book empty (avoid /0).

    def ranked_depth(self, n=10, include_deep=False):
        # Ranked (price, qty) levels per side, best-first, for multi-level OFI.
        # Returns (bids, asks): bids sorted HIGH->LOW (best bid first), asks
        # sorted LOW->HIGH (best ask first), each truncated to the top n levels.
        # Mirrors obi()'s level aggregation exactly (net qty per price, __AGG_
        # skipped unless include_deep, empty levels dropped) so the deep-OFI
        # signal is consistent with the OBI depth the engine already exposes.
        bids, asks = {}, {}
        # aggregate net qty per price level, same rules as obi()
        for k, o in self.o.items():
            if k.startswith("__AGG_") and not include_deep:
                continue
            d = bids if o.side == "BUY" else asks
            d[o.price] = d.get(o.price, 0.0) + o.qty
        # drop levels that netted to <= 0
        bids = {p: q for p, q in bids.items() if q > 0}
        asks = {p: q for p, q in asks.items() if q > 0}
        # best-first ranked (price, qty) tuples, truncated to n levels
        bid_levels = sorted(bids.items(), reverse=True)[:n]
        ask_levels = sorted(asks.items())[:n]
        # return as (bids, asks) lists of (price, qty)
        return (bid_levels, ask_levels)

@dataclass
class MyOrder:
    """One of OUR simulated orders, with the state a real order carries.

    ahead     — order_id -> qty for every historical order that was resting
                at our price when we arrived. Price-time priority means all
                of them fill before we do; this dict is maintained event by
                event (market cancels shrink it, trades at our price drain
                it) so our queue position is EXACT, not estimated. That
                precision is a luxury of order-level data.
    t_active  — exchange-ms when the order became live (after send latency),
                or None while it is still on the wire. THE None IS LOAD-BEARING:
                the order is written into self.work the moment it is SENT, so
                that the side is reserved and no duplicate goes out, and
                t_active is what distinguishes "sent" from "matchable". Every
                fill path tests it and skips an order that has not landed.
    cancel_at — exchange-ms when our in-flight cancel LANDS, or None.
                Between deciding to cancel and cancel_at, the order remains
                fully fillable — the in-flight window naive backtests skip.
    """
    side: str          # 'BUY' = our bid, 'SELL' = our ask
    price: float
    qty: float
    ahead: dict
    t_active: int | None   # None = sent but not yet landed; set at _arrive
    cancel_at: int = None
    # AMENDMENT IN FLIGHT (added 2026-09-16). exchange-ms when our in-flight
    # CFO lands, or None. Deliberately SEPARATE from cancel_at, which gates
    # fills: a cancel stops the exchange matching us, an amendment does not --
    # the old terms stay fully live until the new ones arrive. This field
    # exists only so _requote knows not to send a second amendment for a
    # reprice that is already on the wire.
    amend_at: int = None
    oid: int = 0       # unique id: cancels target THIS order, not just the side
    # TAKER FLAG (default False = a normal passive quote). When True AND the
    # engine's allow_taker flag is on, an order that CROSSES the touch executes
    # as a taker (walks the opposite book) instead of being post-only rejected.
    # Used by the age-cross flatten: a deliberate, tagged liquidity-taking exit.
    taker: bool = False

class LatencyModel:
    """Stochastic, two-leg latency (production model).

    Two independent legs:
      wire_out : you -> exchange. Applies to new orders AND cancel requests.
                 Median + occasional fat tail (exponential).
      wire_in  : exchange -> you (ack). Time until you KNOW a cancel landed.
    decision_ms = feed-in -> order-out compute, folded into the out leg.

    Draws are independent per message. The cancel leg costs money: a cancel
    racing an adverse trade loses when decision+wire_out(cancel) exceeds the
    trade's own latency to you.

    FLAGGED SIMPLIFICATION: tail params are PRIORS for PSX-remote access, not
    measured. Refit from colo telemetry once live. A constant-latency run is
    recoverable with tail_prob=0 and tail_ms=0.
    """

    def __init__(self, decision_ms=5.0, wire_out_median_ms=40.0,
                 # constructor. All args have defaults = PRIORS for PSX-remote access (not measured); refit from your colo telemetry once live.
                 wire_out_tail_ms=400.0, wire_in_median_ms=40.0,
                 wire_in_tail_ms=10.0, tail_prob=0.02, seed=0):
        self.decision_ms = decision_ms  # your compute time: feed-in -> order-out (strategy decides). Typical 5ms.
        self.wire_out_median_ms = wire_out_median_ms  # typical one-way network delay YOU -> exchange gateway. Applies to new orders AND cancels. ~40ms.
        self.wire_out_tail_ms = wire_out_tail_ms  # size of the OCCASIONAL spike on the send leg (GC pause, congestion). Mean of the tail draw. ~400ms.
        self.wire_in_median_ms = wire_in_median_ms  # typical delay exchange -> YOU for an ACK (confirmation a cancel landed). ~40ms.
        self.wire_in_tail_ms = wire_in_tail_ms  # size of the occasional spike on the ack leg. Smaller than send-side. ~10ms.
        self.tail_prob = tail_prob  # probability ANY given message hits the fat tail. 0.02 = 2% of messages spike.
        self.rng = np.random.default_rng(
            seed)  # seeded random generator. Same seed -> identical latency draws every run -> reproducible backtests.

    def draw_out(self):  # returns ONE random send latency (ms), for a new order OR a cancel request.
        base = self.decision_ms + self.wire_out_median_ms  # start with the normal case: your compute time + typical wire delay (5 + 40 = 45ms).
        if self.rng.random() < self.tail_prob:  # roll a die in [0,1): with probability tail_prob (2%), this message spikes.
            base += self.rng.exponential(self.wire_out_tail_ms)  # add a random spike drawn from an exponential distribution (mean = tail_ms). Most spikes small, occasionally huge -> models real fat tails.
        return base  # total one-way send latency for this message.

    def draw_ack(self):  # returns ONE random ACK latency (ms) -- time to LEARN a cancel succeeded.
        base = self.wire_in_median_ms  # normal case: typical return-path delay (40ms). No decision_ms here -- an ack is passive, no compute.
        if self.rng.random() < self.tail_prob:  # same 2% chance of a spike on the return path.
            base += self.rng.exponential(self.wire_in_tail_ms)  # add an exponential spike (smaller mean than send side).
        return base  # total ack latency for this message.


class Backtester:
    """The engine. Owns the historical book, our orders, and the accounting.

    Configuration (cfg dict):
      latency_ms   one-way decision+wire latency, applied to BOTH order
                   placement and cancels.
                   FLAGGED SIMPLIFICATION: constant and symmetric.
                   Production: stochastic per-message draws with separate
                   out/ack legs; the cancel leg racing toxic flow is the
                   one that costs money. Sensitivity on this day: PnL is
                   nearly flat 120ms -> 2000ms (5.6s median trade gap), so
                   the go/no-go conclusion is latency-robust; inventory and
                   toxicity results are the ones needing the richer model.
      fee_per_share  all-in cost, PKR per share per side. Placeholder 0.0
                   until the real broker + PSX MM-program schedule is known;
                   at ~3bps gross edge, this number decides everything.
      session      (start_ms, end_ms): quote only inside this window.
      fill_on_crossing_adds  True -> an incoming historical order whose
                   price crosses our quote fills us (it is marketable flow
                   that would have traded with us). See _on_market_add.
      at_price_mode  what happens when a trade prints exactly AT our price:
                   'queue'  exact queue consumption (default, realistic)
                   'never'  no at-price fills (conservative lower bound)
                   'always' full fill (optimistic upper bound)
                   Run all three to bracket results.

      ---- CHANGE FORMER ORDER (CFO), added 2026-09-16 --------------------
      use_cfo      False (default) -> a reprice is a CANCEL plus a NEW order,
                   two messages with two independent latency draws. This is
                   the original behaviour and is byte-identical when off.
                   True -> ONE message amends the resting order in place.
                   PSX Regulations 8.5.1(d) names this order type "Change
                   Former Order (CFO)"; 8.12.1 makes it the ONLY way to
                   modify an order's terms.

      WHAT AN AMENDMENT DOES TO QUEUE POSITION IS A PER-VENUE RULE, so it is
      three flags rather than one. PSX Regulations 8.5.2:

        "Modification of price in CFO shall be subject to fill allocation
         priorities, however, reduction of bid/offer quantity shall not be
         subject to the fill allocation priorities."

      cfo_price_keeps_priority     PSX: False. A price change re-queues.
      cfo_qty_down_keeps_priority  PSX: True.  A size REDUCTION is carved out
                                   of the priority rule and holds its place.
      cfo_qty_up_keeps_priority    PSX: False. Only reduction is carved out,
                                   so an increase re-queues.

      A venue that preserves priority on a reprice is expressed by flipping
      cfo_price_keeps_priority to True -- nothing else changes.

    Internal state:
      work     side -> MyOrder: at most ONE working order per side.
      pending  min-heap of our in-flight messages (t, seq, action, payload);
               seq is a monotone counter that (a) breaks time ties FIFO and
               (b) prevents heapq from ever comparing payload objects.
      pos/cash running position (shares, signed) and cash (PKR).
      fills/equity  accounting logs -> DataFrames returned by run().
    """

    def __init__(self, strategy,
                 cfg):  # constructor. strategy = the quoting logic (e.g. NaiveSymmetricMM); cfg = config dict (latency, fees, session window, fill rules).
        self.strat = strategy  # store the strategy object; _requote() calls self.strat.quotes(...) each event to ask where to quote.
        self.cfg = cfg  # store the config dict; read throughout for session window, fill rules, etc.
        # ALLOW_TAKER FLAG (default False = post-only engine, byte-identical to before).
        # Read from the strategy so ONE strategy switch (enable_age_cross) turns on
        # both the strategy's flatten behaviour and the engine's permission to take.
        # When False, tagged taker orders are still rejected like any crossing order.
        self.allow_taker = bool(getattr(strategy, "allow_taker", False))
        # LOG_FILL_STATE (default False = byte-identical): when True, each resting
        # order's log record also captures the MARKET STATE at the instant it
        # joined the queue (queue ahead, own/opp depth, spread, OBI, bucket). Pairs
        # with end_reason to give an unbiased fill-probability dataset: every
        # posted quote, filled or not, with the circumstances it was posted into.
        self.log_fill_state = bool(getattr(strategy, "log_fill_state", False))
        # ---- CFO (Change Former Order), added 2026-09-16 ------------------
        # Off by default: with use_cfo False nothing below this line executes
        # and the engine is byte-identical to the cancel-plus-new original.
        # master switch: one AMEND message instead of a CANCEL plus a NEW.
        # Absent from cfg -> False -> nothing in the CFO path ever executes.
        self.use_cfo = bool(cfg.get("use_cfo", False))
        # ---- CROSSED-BOOK GUARD, added 2026-09-17 ------------------------
        # DO NOT QUOTE OFF A BOOK WHERE THE BEST BID IS AT OR ABOVE THE BEST
        # ASK. Such a book cannot exist at the exchange: the two orders would
        # have matched. It appears in the reconstruction, and until this flag
        # existed the engine quoted against it anyway -- pricing every quote
        # off a mid that is not a mid.
        #
        # THE CAUSE HAS NOT BEEN ESTABLISHED. It may be events applied out of
        # sequence, a gap in the message stream, or the feed itself. That is
        # what sim/check_data_quality.py is for. This flag does not diagnose
        # it; it stops the engine trading on it.
        #
        # ON BY DEFAULT, AND THAT CHANGES RESULTS. Every number produced
        # before 2026-09-17 was computed with the engine quoting on these
        # books. Set skip_crossed_book=False in cfg to reproduce an older run
        # exactly; leave it on for anything new. The count is reported as
        # crossed_book_requotes so the size of the change is visible rather
        # than inferred.
        #
        # WHY HERE AND NOT IN THE STRATEGY. micro_mm already refuses a
        # one-sided or zero-size book (bb/ba None, or bq/aq <= 0). A crossed
        # book is not a quoting judgement though -- it is the book being
        # invalid -- so it belongs to whoever owns the book, which is the
        # engine. The production engine makes the same check in the same
        # place, in sim/replay.py's _snapshot.
        self.skip_crossed_book = bool(cfg.get("skip_crossed_book", True))
        # ---- STALE-FEED GUARD, added 2026-09-17 --------------------------
        # DO NOT QUOTE WHEN NOTHING HAS ARRIVED FOR A WHILE.
        #
        # MEASURED IN THE CAPTURE, not hypothetical. NRL and MLCF over three
        # days: 21 stretches during continuous trading with NO message of any
        # kind -- no trade, no book update, no heartbeat -- for 7 to 33
        # seconds. The PSX feed heartbeats every 3 seconds per channel, so
        # that is the receiver having stopped, and the gaps recur on a
        # ~47-minute cycle (17 of 18 intervals between 0.89x and 1.07x of the
        # 47.1-minute median), which is a schedule rather than load.
        #
        # WHAT THE ENGINE DID BEFORE THIS. Nothing, and that is the problem.
        # run() is event-driven: no events means _requote is never called, no
        # fill checks run, and nothing lands. The engine simply sat with its
        # orders resting through a blind window and picked up afterwards as
        # if nothing had happened. That is the MOST FLATTERING possible
        # assumption -- free queue position through exactly the period when a
        # real book would have moved without us and our quotes would have
        # been picked off.
        #
        # WHAT IT DOES NOW. On the first event after a silence longer than
        # stale_feed_seconds, it pulls every working quote, exactly as a halt
        # does, and counts the window. It stays dark until a SNAPSHOT lands,
        # because a snapshot is a full state replacement and is the only
        # message that restores a book we know is stale -- incremental
        # updates applied to a stale book leave it stale.
        #
        # THIS IS ALSO THE PRODUCTION BEHAVIOUR. The PSX spec's own answer is
        # a 3-second channel heartbeat with disconnect after two missed
        # intervals. A live engine that keeps quoting into a feed it cannot
        # see is the failure this models.
        #
        # 7 SECONDS: two missed 3-second heartbeats plus jitter. Set
        # stale_feed_seconds=0 to disable and reproduce a pre-2026-09-17 run.
        self.stale_feed_seconds = float(cfg.get("stale_feed_seconds", 7.0))
        # the capture time of the last event seen, for measuring the silence
        self._last_event_cap = None
        # True while we are standing down after a silence, until a snapshot
        # restores the book
        self._feed_stale = False
        # ---- BLIND-WINDOW LEDGER, added 2026-09-17 ----------------------
        # One record per silence, so the windows can be taken OUT of the
        # measurement rather than merely quoted through. Each record holds
        # the silence in seconds, how much of it fell inside the session
        # window, the mid on either side of it, and what WE had resting when
        # the feed went dark. The EOD block turns these into the reported
        # numbers -- and says plainly which distortion they can and cannot
        # remove.
        self.blind_windows = []
        # index into blind_windows of the record still waiting for its
        # closing mid, or None. Filled at the mark-to-market step of the
        # SAME event, because that is the first point where the post-silence
        # touch exists.
        self._pending_blind = None
        # ---- SHORT-SALE POLICY, added 2026-09-16 -------------------------
        # read off the strategy, the same way allow_taker and log_fill_state
        # are, so ONE variable drives quoting and execution together.
        self.short_policy = getattr(strategy, "short_policy", "unrestricted")
        # is this symbol on NCCPL's Category A SLB-eligible list (10.17)?
        self.slb_eligible = bool(getattr(strategy, "slb_eligible", False))
        # THE UPTICK GATE IS ON only for an SLB-eligible name under the
        # slb_uptick policy. Every other combination either forbids shorting
        # in the quote (no_short / long_buffer / ineligible) or permits it
        # outright (unrestricted), and in both cases there is nothing to gate.
        self.enforce_uptick = (self.short_policy == "slb_uptick"
                               and self.slb_eligible)
        # last executed price seen on the tape, for the tick test. None until
        # the first print, and an unknown tick is NOT treated as an uptick.
        self._last_exec_px = None
        # direction of the last price CHANGE: +1 up, -1 down, 0 not yet known.
        # Zero-Plus Tick is defined off this, not off the last trade.
        self._last_tick_dir = 0
        # PSX 8.5.2: "Modification of price in CFO shall be subject to fill
        # allocation priorities" -- a reprice goes to the back of the queue.
        # False is therefore the PSX default; True is for a venue that holds.
        self.cfo_price_keeps_priority = bool(
            cfg.get("cfo_price_keeps_priority", False))
        # PSX 8.5.2 continued: "...however, reduction of bid/offer quantity
        # shall not be subject to the fill allocation priorities." A size cut
        # is explicitly carved out, so it KEEPS its place -> default True.
        self.cfo_qty_down_keeps_priority = bool(
            cfg.get("cfo_qty_down_keeps_priority", True))
        # Only REDUCTION is carved out by 8.5.2, so an increase is subject to
        # the priority rule like any other modification -> default False.
        self.cfo_qty_up_keeps_priority = bool(
            cfg.get("cfo_qty_up_keeps_priority", False))
        # ARRIVAL-RATE buffer (only maintained when log_fill_state is on): a
        # trailing list of (ts_ms, side, qty) for EVERY continuous-market trade,
        # so _arrive can estimate the recent side-signed trade arrival rate the
        # quote's queue will actually clear against. Windows are min(T minutes,
        # N trades) -- trade-anchored so a churn-with-no-trades regime cannot
        # spuriously zero the event side; the time cap bounds slow tape, the
        # trade cap bounds fast tape. Swept to calibrate what "recent" means.
        self._arr_buf = []
        # (label, T_ms, N_trades) window grid for the sweep
        self._arr_windows = [("w1m50", 60000.0, 50),
                             ("w3m150", 180000.0, 150),
                             ("w5m300", 300000.0, 300)]
        # keep at most the largest N so the buffer never grows unbounded
        self._arr_max_n = max(n for _, _, n in self._arr_windows)
        # Skip the per-event equity+OBI logging when False (sweep speed path).
        # obi(5)/obi(None) scan book levels EVERY event and dominate runtime (~50x);
        # consumers needing only fills set cfg["log_equity"]=False. Defaults True so
        # real backtests keep the full equity curve unchanged. Fills are unaffected.
        self.log_equity = cfg.get("log_equity", True)
        # latency: use provided LatencyModel, else build a CONSTANT-latency
        # model from cfg['latency_ms'] (back-compat / go-no-go runs)
        self.lat = cfg.get(
            'latency_model')  # try to get a LatencyModel from cfg. Returns None if the caller didn't supply one.
        if self.lat is None:  # no stochastic model given -> caller wants simple constant latency.
            L = cfg.get('latency_ms', 120)  # read the fixed one-way latency in ms; default 120 if not specified.
            self.lat = LatencyModel(decision_ms=0.0,
                                    wire_out_median_ms=L, wire_out_tail_ms=0.0,
                                    # build a LatencyModel with ALL randomness zeroed: no compute time,
                                    wire_in_median_ms=L, wire_in_tail_ms=0.0,
                                    # no tail spikes (tail_ms=0), send & ack both = L.
                                    tail_prob=0.0)  # tail_prob=0 -> spike branch never fires -> every draw returns exactly L. A "constant latency" disguised as the same LatencyModel interface.
        self.book = Book()  # the reconstructed real order book (everyone else's orders). Empty until the first snapshot/update.
        # OUR live orders, keyed by side: {"BUY": [MyOrder, ...], "SELL": [...]}.
        #
        # A LIST, NOT ONE ORDER. Until 2026-09-17 this was dict[side] -> ONE
        # MyOrder, so the simulator could only ever record a single order of
        # ours per side. A strategy that legitimately rests TWO orders on a
        # side -- which is how you add size at a price without sending the
        # shares already resting to the back of the queue, PSX 8.5.2 -- had
        # its second order overwrite its first. The first then existed
        # nowhere: no cancel, no fill, no lifecycle end, and every later
        # message aimed at it was refused. A real exchange holds both.
        #
        # ORDER WITHIN THE LIST IS SEND ORDER, oldest first. That is what the
        # fill engine uses to break time priority between two of our own
        # orders resting at the same price.
        self.work: dict[str, list] = {"BUY": [], "SELL": []}
        self._oid = 0  # counter that generates a unique id for each order we send (cancels target a specific oid, not just a side).
        # DIAGNOSTIC STATE, NO BEHAVIOUR CHANGE. Per side: how many NEW orders
        # have been sent and have not yet reached the exchange. self.work above
        # only records an order once it LANDS, so this is the only way to see
        # the send-to-arrival window from inside _requote. Read and written by
        # the two diagnostic counters in self.stats; nothing decides anything
        # on it.
        self._new_in_flight = {"BUY": 0, "SELL": 0}
        self.ack_until = {"BUY": 0,
                          "SELL": 0}  # per side: exchange-ms until which that side's last cancel is UNCONFIRMED (ack not yet back). 0 = nothing pending.
        self.use_ack = 'latency_model' in cfg  # ack-realism gate: only enforce the "don't restack an unconfirmed side" rule in stochastic mode (when a real LatencyModel was given).
        self.pending = []  # min-heap of OUR in-flight messages (orders/cancels traveling to the exchange). Ordered by land-time.
        self._seq = 0  # monotonic counter: breaks ties in the heap FIFO AND stops heapq from ever comparing MyOrder payloads.
        # OPENING INVENTORY. Under the long_buffer policy the day starts with
        # shares already held, so every sale is a sale of stock we own and no
        # Blank Sale is ever made (PSX 10.15). 0 everywhere else, which is the
        # original behaviour.
        #
        # CORRECTED 2026-09-16. The first version set this position and left
        # cash at zero, with a comment claiming that was deliberate. It was
        # wrong. The engine liquidates the closing position into the book, so
        # starting with 1,000 shares and never paying for them books the entire
        # sale proceeds as profit -- about 289,000 PKR of phantom P&L on a
        # Rs 289 name, which would have made long_buffer look like the best
        # policy by a mile for a reason having nothing to do with policy.
        #
        # THE BUFFER IS BOUGHT, not conjured. It is acquired at the first
        # market print of the day (see _acquire_buffer), cash is debited and
        # the fee is charged. That makes the buffer's intraday price move a
        # real cost carried by the run -- which it is: holding inventory to
        # stay on the right side of 10.15 is a directional exposure, and a
        # comparison that hides it is not a comparison.
        self.pos = float(cfg.get("opening_inventory", 0.0))
        # True until the buffer has been paid for; False when there is none
        self._buffer_unpaid = self.pos != 0.0
        # ISOLATING THE BUFFER'S DIRECTIONAL BET FROM THE QUOTING POLICY.
        # The run buys the buffer at the first print and sells it back into the
        # closing book, so whatever the stock did between those two moments
        # lands in the P&L. That is a bet on the stock, not a market-making
        # result. At the two-clip buffer the sweep uses (100 shares) a 1%
        # intraday move on a Rs 300 name is 300 PKR against a baseline day of
        # roughly 148 PKR, so the stock is the larger of the two terms. Recording the size and the entry price lets the
        # sweep subtract the price move afterwards and compare quoting to
        # quoting. The ENTRY FEE and the EXIT EXECUTION COST are deliberately
        # NOT recorded for removal: those are real costs of carrying a buffer.
        # how many shares the buffer is (0.0 when there is none)
        self.buffer_qty = self.pos
        # the price it was acquired at; None until the first print sets it
        self.buffer_px = None
        # THE TOUCH AT THE MOMENT OF ACQUISITION. Recorded so the claim "this
        # acquisition is better than any real execution" becomes a number
        # instead of an assertion. The buffer is booked at the PRINT price,
        # whichever side that print was on; a real market buy pays the ASK.
        # The gap between them, times the size, is what the model hands us for
        # free every day. None on a one-sided book, where there is no ask to
        # compare against and the honest answer is "not measurable here".
        # best bid at the instant the buffer was paid for
        self.buffer_bid = None
        # best ask at that instant -- the price a market buy would have paid
        self.buffer_ask = None
        self.cash = 0.0  # our running cash in PKR (signed). Fills add/subtract price*qty and deduct fees.
        self.fills, self.equity = [], []  # accounting logs: fills = every trade we got; equity = mark-to-market curve (one row per event). Returned as DataFrames by run().
        self.eod = None        # EOD book-walk liquidation report (filled once, at session end).
        # Last mid seen with a two-sided book -- EOD reference if the close is one-sided.
        self.last_good_mid = None
        # ORDER LIFECYCLE LOG: oid -> {oid, side, px, qty, t_sent, t_live, t_end,
        # end_reason}. Powers quote-uptime, quoted-spread, time-to-fill, and the
        # per-order audit. Flushed to self.order_log (DataFrame) at end of run().
        self._olog = {}
        # outbound MESSAGE timestamps (order sends + cancel sends) for msgs/sec
        self._msg_ts = []
        self.stats = {
                      # how many short-taking fills the uptick rule refused.
                      # Declared here rather than beside the other short-sale
                      # state because self.stats does not exist yet up there.
                      # A run can then show what the constraint COST in fills,
                      # not only what it did to P&L.
                      "short_fills_blocked_by_uptick": 0,
                      "rejected_crossing": 0, "n_orders_sent": 0, "n_cancels": 0,
                      # diagnostic counters, all start at 0:
                      "stale_cancels_ignored": 0, "requotes_blocked_by_ack": 0,
                      # rejected_crossing = post-only rejects; n_orders_sent/n_cancels = message counts;
                      "halted_requotes": 0,  # stale_cancels_ignored = cancels for already-gone orders; requotes_blocked_by_ack = requotes skipped by ack guard; halted_requotes = requotes skipped during halts.
                      # DIAGNOSTIC COUNTERS, NO BEHAVIOUR CHANGE. Added after
                      # the reconcile gate showed this backtester sending 1,822
                      # messages where the production engine sent 359, against
                      # near-equal cancel counts. These two say whether the
                      # cause is what the arithmetic points at: an order that
                      # has been SENT is not recorded in self.work until it
                      # LANDS, so for one network latency the side reads as
                      # empty and the next requote sends another new order.
                      # orders_sent_while_new_in_flight counts those extra
                      # sends; orders_orphaned_by_overwrite counts the resting
                      # orders that are silently dropped when a duplicate lands
                      # on top of them.
                      "orders_sent_while_new_in_flight": 0,
                      "orders_orphaned_by_overwrite": 0,
                      # AND THE COUNTERS THE FIX ITSELF ADDED.
                      # requotes_blocked_in_flight = cycles where a side was
                      # left alone because its order was still on the wire.
                      # This should be LARGE: it is the duplicate sends, now
                      # correctly doing nothing instead of sending.
                      "requotes_blocked_in_flight": 0,
                      # stale_arrivals_ignored = an ARRIVE for an order the
                      # side is no longer holding. The old code applied these
                      # by overwriting whatever was there.
                      "stale_arrivals_ignored": 0}


    # ================= our-order plumbing (EXCHANGE side) =================
    def _side_orders(self, side):
        """Our orders on one side, oldest first. Never None."""
        # a side with nothing resting returns an empty list, not None, so
        # every caller can iterate without a guard
        return self.work.setdefault(side, [])

    def _all_orders(self):
        """Every order of ours, both sides, in no particular order."""
        # flattened, for the handlers that touch all of them
        return [o for orders in self.work.values() for o in orders]

    def _order_by_oid(self, side, oid):
        """The order on this side carrying that id, or None."""
        # ids are unique per order generation, so at most one matches
        for o in self._side_orders(side):
            if o.oid == oid:
                return o
        return None

    def _drop_order(self, side, o):
        """Remove one specific order from a side. BY IDENTITY, not value.

        MyOrder is a plain dataclass, so `list.remove` would compare fields --
        and two of our orders on the same side at the same price for the same
        size are equal by that test while being different orders with
        different places in the queue.
        """
        # the list this order lives in
        orders = self._side_orders(side)
        # find it by identity and remove that exact element
        for i, x in enumerate(orders):
            if x is o:
                del orders[i]
                return True
        # nothing removed
        return False

    def _lead(self, side):
        """The oldest order on a side, or None. The single-order view."""
        # mm_backtest's own _requote never rests more than one order per side,
        # so this IS that order for every run this file drives
        orders = self._side_orders(side)
        return orders[0] if orders else None

    def _push(self, t, action, payload):
        """Schedule one of our messages to land on the exchange at time t."""
        heapq.heappush(self.pending, (t, self._seq, action, payload))
        self._seq += 1

    def _activate_until(self, t_exch):
        """Land every in-flight message due BEFORE the next market event.

        Called at the top of the main loop with the next event's exchange
        time. Strict '<' is deliberate and conservative: one of our
        messages stamped the SAME millisecond as a market event is
        processed AFTER it — at ms resolution we cannot prove we beat the
        event to the gateway, so we assume we lost the race.

        ARRIVE -> _arrive() (may still be rejected as crossing)
        AMEND  -> _amend()  (a CFO; rejected if the target already filled)
        CANCEL -> remove the working order; if a fill already consumed it
                  (self.work.get returns None) the cancel is simply void —
                  which is exactly what an exchange cancel-reject is.
        """
        while self.pending and self.pending[0][0] < t_exch:  # loop while the heap is non-empty AND its earliest message's land-time (pending[0][0]) is before t_exch. Strict '<' = same-ms messages process AFTER the market event (conservative).
            t, _, action, p = heapq.heappop(self.pending)  # pop the earliest message off the heap. Unpack: t = land-time, _ = the _seq tiebreaker (ignored), action = "ARRIVE"/"CANCEL", p = payload.
            if action == "ARRIVE":  # this message is a NEW order reaching the exchange.
                self._arrive(t,p)  # hand it to _arrive(), which checks if it crosses the book (reject) and otherwise makes it live + snapshots its queue position.
            # this message is a Change Former Order reaching the exchange: it
            # modifies a resting order in place rather than replacing it.
            elif action == "AMEND":
                # hand it to _amend(), which applies the new price/size and
                # decides, per the venue's rules, what happens to queue position
                self._amend(t, p)
            elif action == "CANCEL":  # this message is a CANCEL request reaching the exchange.
                side, oid = p  # unpack the payload: which side, and the SPECIFIC order id this cancel was meant for.
                o = self._order_by_oid(side, oid)  # the SPECIFIC order this cancel targeted, or None if it has already gone.
                if o is not None:  # it is still resting, so the cancel lands on it
                    self._drop_order(side, o)  # remove that one order; anything else of ours on this side is untouched.
                    self.stats["n_cancels"] += 1  # count a successful cancel.
                    # LIFECYCLE: the order ended by cancellation at this land-time
                    if oid in self._olog:
                        self._olog[oid]["t_end"] = t
                        self._olog[oid]["end_reason"] = "cancelled"
                else:  # no matching order: it was already filled, or already replaced by a newer order.
                    self.stats["stale_cancels_ignored"] += 1  # count a no-op cancel. This mirrors a real exchange CANCEL-REJECT (nothing there to cancel).

    def _amend(self, t, payload):
        """A Change Former Order reaches the exchange. ONE message, in place.

        PSX Regulations 8.12.1: "The terms of an Order placed in the Trading
        System can only be modified through the CFO option." 8.12.2: it "can
        only modify price and volume of an unfilled/outstanding Order in whole
        or in parts" -- so an order that has partly filled is still amendable
        on what is left, which is why this works off o.qty (the remainder)
        rather than the original size.

        WHAT HAPPENS TO QUEUE POSITION is the whole question, and it is a
        per-venue rule read from the three cfo_*_keeps_priority flags. On PSX
        (8.5.2) a price change re-queues and a size reduction does not.

        Priority KEPT  -> the ahead dict and t_active carry over untouched.
                          We are the same order in the same place in the line.
        Priority LOST  -> ahead is re-snapshotted from the book at the new
                          price, exactly as a brand-new order's would be in
                          _arrive. Everything resting there is now in front.
        """
        # unpack what _requote scheduled: which side, WHICH order generation
        # this CFO was aimed at, the new terms, and the id the amended version
        # will carry from here on.
        side, oid, new_px, new_qty, new_oid = payload
        # THE SPECIFIC order this amendment was aimed at, or None.
        o = self._order_by_oid(side, oid)
        # THE TARGET IS GONE -- it filled, or a later message already replaced
        # it. A real exchange answers that with an Order Cancel Reject; there
        # is nothing left to modify.
        if o is None:
            # count it so a run can show how often a CFO raced a fill and lost.
            self.stats["stale_cfos_ignored"] = self.stats.get("stale_cfos_ignored", 0) + 1
            # nothing is applied: the book is left exactly as it was.
            return
        # did the price move? compared against what is actually resting.
        price_changed = (new_px != o.price)
        # is this a REDUCTION? measured against o.qty, the REMAINING size,
        # because 8.12.2 amends the outstanding part, not the original order.
        qty_down = (new_qty < o.qty)
        # is this an INCREASE? kept separate because the two follow different
        # rules under 8.5.2.
        qty_up = (new_qty > o.qty)
        # A PRICE CHANGE IS JUDGED FIRST and overrides the quantity rules: a
        # reprice moves us to a different price level, where whatever position
        # we held in the old level's queue means nothing at all.
        if price_changed:
            # the venue's rule for a reprice.
            keeps = self.cfo_price_keeps_priority
        # same price, bigger size -- the venue's rule for an increase.
        elif qty_up:
            # on PSX this is False, because 8.5.2 carves out only reduction.
            keeps = self.cfo_qty_up_keeps_priority
        # same price, smaller size -- the carve-out in 8.5.2.
        elif qty_down:
            # on PSX this is True: a reduction is not subject to the rule.
            keeps = self.cfo_qty_down_keeps_priority
        # neither price nor size moved, so there is nothing to re-queue.
        else:
            # a no-op amendment cannot cost priority; _requote should not have
            # sent one, and the no-churn check above it means it does not.
            keeps = True
        # WE HELD OUR PLACE: carry the queue state across unchanged.
        if keeps:
            # the same shares are still in front of us, order for order.
            ahead = o.ahead
            # and the join time is unchanged, because we never left the line.
            t_active = o.t_active
        # WE LOST OUR PLACE: rebuild the queue exactly as a new order would.
        else:
            # everything resting at the new price is now ahead of us. This is
            # the same call _arrive makes for a brand-new order, which is the
            # point -- a re-queued amendment IS a new order for priority.
            ahead = self.book.qty_at(side, new_px)
            # and we joined the line now, not when the original was sent.
            t_active = t
        # close the old lifecycle record, if the engine is logging them, so
        # time-to-fill stays measured per quote VERSION rather than per order.
        if oid in self._olog:
            # the old version stopped existing at this instant.
            self._olog[oid]["t_end"] = t
            # and it ended by being amended, not filled or cancelled.
            self._olog[oid]["end_reason"] = "amended"
        # open a record for the amended version. t_sent and t_live are the same
        # instant here: unlike a new order, an amendment is live the moment it
        # lands -- it was already in the book under its previous terms.
        self._olog[new_oid] = {"oid": new_oid, "side": side, "px": new_px,
                               "qty": new_qty, "t_sent": t, "t_live": t,
                               "t_end": None, "end_reason": None}
        # replace the amended order with its new generation AT THE SAME
        # POSITION IN THE LIST. Position is send order, which is what breaks
        # time priority between two of our own orders at one price -- an
        # amendment does not make an order younger than one sent after it, so
        # appending would silently reorder our own queue.
        # taker is always False: a CFO modifies a resting order and never
        # crosses.
        _new = MyOrder(side, new_px, new_qty, ahead, t_active,
                       oid=new_oid, taker=False)
        # find the order being amended and overwrite that slot
        _orders = self._side_orders(side)
        for _i, _x in enumerate(_orders):
            if _x is o:
                _orders[_i] = _new
                break
        # how many amendments actually landed this run.
        self.stats["n_cfos"] = self.stats.get("n_cfos", 0) + 1
        # WHAT KIND OF AMENDMENT IT WAS. Added 2026-09-17, after the kept-place
        # count was predicted wrongly twice in a row. Under 8.5.2 only a pure
        # size REDUCTION keeps its place, so "how many kept?" is really "how
        # many were reductions?" -- and that is a fact to be counted, not
        # reasoned about from the policy's description.
        self.stats["n_cfos_price_change"] = (
            self.stats.get("n_cfos_price_change", 0) + (1 if price_changed else 0))
        self.stats["n_cfos_qty_up"] = (
            self.stats.get("n_cfos_qty_up", 0)
            + (1 if (not price_changed and qty_up) else 0))
        self.stats["n_cfos_qty_down"] = (
            self.stats.get("n_cfos_qty_down", 0)
            + (1 if (not price_changed and qty_down) else 0))
        # and of those, how many held their place in the queue.
        if keeps:
            # if this stays at zero on a PSX run, the reductions are not firing
            # and the flag is doing no work -- worth noticing rather than not.
            self.stats["n_cfos_kept_priority"] = self.stats.get("n_cfos_kept_priority", 0) + 1

    def _arrive(self, t, o: MyOrder): # called when OUR order o reaches the exchange at time t (after its send latency). o is a MyOrder (side, price, qty, ...).
        """Our new order reaches the exchange. Two things happen:

        1. Marketability check against the CURRENT book (which may have
           moved during our latency window). If our bid >= best ask (or
           ask <= best bid) the order would execute as a taker.
           FLAGGED SIMPLIFICATION: we REJECT it (post-only semantics) and
           count it, instead of modelling taker execution. Rare when
           quoting inside a 62-tick spread (48 of 2,673 sends on the
           sample day) but a real production model would walk the book.

        2. Queue snapshot: everything resting at our price RIGHT NOW is
           ahead of us. Captured once here; maintained incrementally by
           the market-event handlers below.
        """
        # This order has arrived, so it is no longer in the air.
        if self._new_in_flight.get(o.side, 0) > 0:
            # one fewer order in the air on this side
            self._new_in_flight[o.side] -= 1
        # IDENTITY CHECK. The order was written into self.work when it was
        # SENT, so the side should still be holding THIS object. If it is
        # holding something else -- or nothing -- then this arrival belongs to
        # an order that has already been cancelled, filled or superseded, and
        # applying it would resurrect a dead order.
        #
        # THIS IS THE GUARD THAT REPLACED THE OVERWRITE. The old code assigned
        # self.work[o.side] = o unconditionally at the end of this method, so
        # whatever was resting there was silently dropped: never cancelled,
        # never filled, never closed out in the lifecycle log. The gate
        # measured 5,205 of those across four symbol-days.
        if o not in self._side_orders(o.side):
            # count it: with the in-flight rule in _requote this should be rare
            # and is worth seeing if it is not
            self.stats["stale_arrivals_ignored"] += 1
            # nothing is applied
            return
        bb, _, ba, _ = self.book.bbo()  # get the CURRENT best bid (bb) and best ask (ba) from the real book. The _ discard the qty fields (bq, aq) -- not needed here. Note: the book may have MOVED during our latency window.
        crosses = ((o.side == "BUY" and ba is not None and o.price >= ba) or  # would our order execute immediately instead of resting? For a BUY: our bid at/above the best ask means we'd cross and take.
                   (o.side == "SELL" and bb is not None and o.price <= bb))  # for a SELL: our ask at/below the best bid means we'd cross and take. (the 'is not None' guards an empty side.)
        if crosses:  # our order would be marketable (take liquidity) rather than post passively.
            # TAKER PATH (opt-in): a DELIBERATELY tagged taker order, with the engine's
            # allow_taker flag on, executes against the opposite book instead of being
            # rejected. Walks best-first levels, fills at each LEVEL price (a taker pays
            # the resting price), books pos/cash/fee exactly like _fill, reason="taker".
            # Shadow-fill semantics (does not mutate the historical book), same as
            # every other fill in this engine. Off (default) -> falls through to the
            # original post-only reject below, byte-identical.
            if self.allow_taker and getattr(o, "taker", False):
                # THE ORDER NEVER RESTS, so its slot must be released. It was
                # reserved at send time; leaving it there would have the
                # engine believe size is working that never existed.
                self._drop_order(o.side, o)
                # execute the taker and stop
                self._taker_fill(t, o)
                # done
                return
            self.stats["rejected_crossing"] += 1  # count it as a rejected crossing order.
            # REJECTED, SO RELEASE THE SLOT. Same reason as the taker path
            # above: the reservation made at send time has to be undone, or
            # the engine believes an order is working that never rested.
            self._drop_order(o.side, o)
            return  # FLAGGED SIMPLIFICATION: reject it (post-only behavior) instead of executing as a taker. The order never enters the book. Exit early.
        o.ahead = self.book.qty_at(o.side, o.price)  # order rests: snapshot our QUEUE POSITION -- {order_id: qty} of every order already resting at our price (all ahead of us under price-time priority).
        o.t_active = t  # record the exchange-time the order became live (used for timing/diagnostics). Until this line runs, t_active is None and every fill path skips the order.
        # LIFECYCLE: the order is now live at the exchange (resting, matchable)
        if o.oid in self._olog:
            self._olog[o.oid]["t_live"] = t
            # FILL-STATE CAPTURE (opt-in): record what the market looked like at
            # the instant this quote joined the queue. These are the features a
            # fill-probability model conditions on; end_reason is the outcome.
            if self.log_fill_state:
                # touch sizes for depth + OBI (bbo() returns bb, bq, ba, aq)
                bb2, bq2, ba2, aq2 = self.book.bbo()
                # shares resting ahead of us at our own price (the queue we wait behind)
                ahead_qty = float(sum(o.ahead.values())) if o.ahead else 0.0
                # visible depth on OUR side's touch and the OPPOSITE side's touch
                own_touch = float(bq2 or 0.0) if o.side == "BUY" else float(aq2 or 0.0)
                opp_touch = float(aq2 or 0.0) if o.side == "BUY" else float(bq2 or 0.0)
                # spread in ticks-equivalent price units (None if one-sided)
                spread = (ba2 - bb2) if (bb2 is not None and ba2 is not None) else None
                # L1 imbalance in [-1,+1]: +ve = bid-heavy
                den = float((bq2 or 0.0) + (aq2 or 0.0))
                obi1 = ((float(bq2 or 0.0) - float(aq2 or 0.0)) / den) if den > 0 else 0.0
                # how far inside/outside the touch we posted (ticks of price; +ve = inside)
                if o.side == "BUY" and bb2 is not None:
                    # a bid above the best bid is MORE aggressive (inside)
                    rel_px = o.price - bb2
                elif o.side == "SELL" and ba2 is not None:
                    # an ask below the best ask is more aggressive
                    rel_px = ba2 - o.price
                else:
                    # no reference touch
                    rel_px = None
                # write the features onto the order's log record
                rec = self._olog[o.oid]
                rec["ahead_qty"] = ahead_qty
                rec["own_touch_qty"] = own_touch
                rec["opp_touch_qty"] = opp_touch
                rec["spread"] = spread
                rec["obi1"] = obi1
                rec["rel_px"] = rel_px
                rec["mid_live"] = (0.5 * (bb2 + ba2)) if spread is not None else None
                # session bucket + window the strategy was in when it posted
                rec["bucket"] = getattr(self.strat, "current_bucket", "middle")
                rec["window"] = getattr(self.strat, "current_window", "none")
                # ARRIVAL-RATE / EXPECTED-WAIT per swept window. For each window,
                # rate = recent clearing-side shares/min; expected_wait_min =
                # (shares ahead + our own qty) / rate = minutes until we'd fill at
                # that rate. rate==0 (no clearing trades in window) -> wait is
                # CENSORED (won't fill at this rate): store a large sentinel + flag.
                # Also store the clearing-trade count and window event count so the
                # churn-with-no-trades state is measured, not just survived.
                for (wlab, wt, wn) in self._arr_windows:
                    # recent clearing-side rate + counts for this window
                    rate, n_clear, n_evt = self._arrival_rate(t, o.side, wt, wn)
                    # shares that must clear before us (queue ahead + our own size)
                    to_clear = ahead_qty + float(o.qty)
                    # expected wait in minutes; None-rate -> censored sentinel
                    if rate > 0.0:
                        # minutes to fill at the recent rate
                        rec[f"ewait_{wlab}"] = to_clear / rate
                        # not censored
                        rec[f"cens_{wlab}"] = 0
                    else:
                        # no clearing trades in the window -> won't fill at this rate
                        rec[f"ewait_{wlab}"] = float("inf")
                        # mark censored (the churn / dead-tape state)
                        rec[f"cens_{wlab}"] = 1
                    # the clearing-side rate itself (shares/min) for analysis
                    rec[f"rate_{wlab}"] = rate
                    # clearing trades in the window (0 = the pathological state)
                    rec[f"nclear_{wlab}"] = n_clear
                    # total trades in the window (for churn composition)
                    rec[f"nevt_{wlab}"] = n_evt

    # ============ fill engine (runs BEFORE the event mutates the book) ====
    # Ordering matters: fills are judged against the book AS IT WAS when
    # the aggressive order hit it. The main loop therefore calls these
    # handlers first, and only then applies the event to self.book.

    def _arrival_rate(self, now, side, t_ms, n_trades):
        """Recent trade arrival rate (shares/min) on the side that CLEARS a resting
        order on `side`. A resting BUY (bid) is cleared by SELL aggressors hitting
        it; a resting SELL (ask) by BUY aggressors. Window = min(t_ms, n_trades):
        take the last n_trades prints, then keep only those within t_ms of now --
        whichever is TIGHTER binds. Returns (rate_per_min, n_trades_in_window,
        total_events_considered). rate can be 0.0 (no clearing trades in window)."""
        # the aggressor side that fills a resting order on `side`
        clearing = "SELL" if side == "BUY" else "BUY"
        # nothing logged yet
        if not self._arr_buf:
            # zero rate, empty window
            return 0.0, 0, 0
        # last n_trades prints (trade-anchored event cap)
        tail = self._arr_buf[-n_trades:]
        # time floor for the window
        cut = now - t_ms
        # keep only prints within the time cap (min of the two windows)
        win = [(ts, sd, q) for (ts, sd, q) in tail if ts >= cut]
        # span of the retained window in minutes (guard divide-by-zero)
        if not win:
            # no trades in the time window: rate 0, 0 clearing, 0 IN-WINDOW events
            # (consistent with the non-empty return below, which reports in-window)
            return 0.0, 0, 0
        # elapsed minutes across the retained window (>= a small floor)
        span_min = max((now - win[0][0]) / 60000.0, 1.0 / 60000.0)
        # shares that CLEARED our side (clearing-side aggressor volume)
        cleared = sum(q for (ts, sd, q) in win if sd == clearing)
        # rate in shares per minute
        rate = cleared / span_min
        # rate, clearing-trade count usable, total prints in window
        return rate, sum(1 for (_, sd, _) in win if sd == clearing), len(win)

    def _taker_fill(self, t_exch, o):
        """Execute a tagged TAKER order by walking the opposite side of the book.

        A BUY taker consumes asks best-first (ascending); a SELL taker consumes
        bids best-first (descending). Each level is filled at the LEVEL price
        (the taker pays the resting price, never its own limit). Accounting is
        identical in form to _fill: pos moves with sgn*take, cash moves opposite
        at the level price, fee per fill, one fills-row per level with
        reason="taker". Unfilled remainder (book exhausted) is simply not filled.
        Shadow-fill: the historical book is NOT mutated (same assumption as every
        passive fill here). Counted in stats["taker_fills"].
        """
        # the opposite side's ranked levels, best-first: BUY taker eats asks
        bids, asks = self.book.ranked_depth(n=50)
        # levels to consume
        levels = asks if o.side == "BUY" else bids
        # position sign of this side
        sgn = 1 if o.side == "BUY" else -1
        # remaining quantity to fill
        remaining = float(o.qty)
        # walk the levels
        for px, avail in levels:
            # stop when done
            if remaining <= 0:
                break
            # take what this level has, up to what we still need
            take = min(float(avail), remaining)
            # skip empty levels
            if take <= 0:
                continue
            # position moves with the side
            self.pos += sgn * take
            # cash moves opposite, at the LEVEL price, minus the fee
            self.cash += -sgn * take * px - fee_for(px, take)
            # one fills row per level consumed, tagged as a taker fill
            self.fills.append({"t": t_exch, "side": o.side, "px": px, "qty": take,
                               "reason": "taker",
                               "window": getattr(self.strat, "current_window", "none"),
                               "bucket": getattr(self.strat, "current_bucket", "middle"),
                               "regime": getattr(self.strat, "current_regime", "normal"),
                               "oid": o.oid})
            # reduce the remainder
            remaining -= take
        # count the taker event
        self.stats["taker_fills"] = self.stats.get("taker_fills", 0) + 1
        # LIFECYCLE: the order ended by taking (fully or partially)
        if o.oid in self._olog:
            self._olog[o.oid]["t_end"] = t_exch
            self._olog[o.oid]["end_reason"] = "taker"

    def _fill(self, side, price, qty, t_exch, reason, order=None): # book a fill of OUR order. side = BUY/SELL; price = the trade's print price (not used for our cash);
        # qty = shares offered to us; t_exch = fill time; reason = provenance tag ("through"/"at_queue"/etc).

        """Book a (possibly partial) fill of our working order on `side`.

        take   = min(our remaining qty, the aggressive qty offered to us)
        sgn    = +1 our bid bought (position up), -1 our ask sold
        cash   moves opposite to position at OUR limit price — we always
               transact at our own quoted price, never at the print price
               (price improvement accrues to the aggressor's limit, not us)
        fees   deducted per share on every fill, both sides
        reason tags provenance: 'through' | 'at_queue' | 'at_optimistic'
               | 'crossing_add' — the fills DataFrame lets you audit how
               much PnL depends on each fill rule.
        Fully consumed orders leave self.work; partials stay with reduced qty.

        The reason tag is for honest attribution, not decoration. Each fill records
        why it happened — was it a trade printing through your price (certain fill
        by price priority), at your price after your queue drained, or a crossing add?
        When you later analyze PnL, you can group by reason and see how much of your profit
        depends on each fill rule. If most of your PnL comes from the more sp712577eculative rules
        (like crossing-add fills, which rely on a counterfactual assumption), that's a signal
        your edge is fragile. It turns the fill log into an auditable record rather than a
        black box.

        One thing to flag about what this method does not do: it never touches self.book
        (the real order book). When you get filled, the historical trade that filled you
        also consumed its real historical counterparty inside Book.trade() — your fill is
        recorded only in your ledger (pos, cash, fills), running in parallel. That's the
        "shadow fill" assumption: your simulated presence doesn't remove real liquidity or
        alter history. Correct for a backtest; the honest caveat is that it's mildly
        optimistic, since a real order of yours would have consumed that liquidity and
        changed what followed.

        """
        # THE SPECIFIC ORDER BEING FILLED. Passed in by the fill engine,
        # which decides -- by price, then by time -- which of our orders on
        # this side the flow reaches. Defaults to the lead order so the two
        # callers that can only ever mean one order (the taker path and the
        # crossing-add path) need no change.
        o = order if order is not None else self._lead(side)
        # nothing of ours resting there: no fill to book
        if o is None:
            return 0.0
        # we can only fill up to OUR remaining size; if the incoming qty is
        # larger, we take our whole order, not more. This is the actual fill.
        take = min(o.qty, qty)
        # ---- UPTICK GATE, added 2026-09-16 --------------------------------
        # PSX Regulations 10.16.1(a): a Short Sale must be "made at an Uptick
        # or Zero-Plus Tick". Only bites when the policy is slb_uptick AND the
        # name is SLB-eligible; every other configuration either forbids the
        # short in the quote or permits it outright.
        if self.enforce_uptick and side == "SELL":
            # would this fill actually take us NET SHORT? Selling stock we hold
            # is an ordinary sale and the uptick rule has nothing to say about
            # it. Only the part that goes below zero is a Short Sale.
            if (self.pos - take) < 0:
                # THE TICK OF *OUR* SALE, not of the print that triggered it.
                # We transact at o.price, never at the print price (price
                # improvement accrues to the aggressor), so o.price is what the
                # regulation measures. On a "through" fill the print is above
                # our ask, so the two genuinely differ and using the print
                # would be the more permissive, wrong answer.
                ref = self._last_exec_px
                # an uptick: our execution price is above the last executed one
                is_uptick = ref is not None and o.price > ref
                # a zero-plus tick: equal to the last executed price, where the
                # last actual move was upward
                is_zero_plus = (ref is not None and o.price == ref
                                and self._last_tick_dir == 1)
                # NOT TOLD IS NOT PERMISSION: before the first print there is
                # no reference price, so neither test can pass and the short
                # does not happen.
                if not (is_uptick or is_zero_plus):
                    # count what the constraint cost us, in fills
                    self.stats["short_fills_blocked_by_uptick"] += 1
                    # the exchange would not have matched a short sale here, so
                    # neither do we. The order stays resting, unchanged, and
                    # NOTHING was taken out of the aggressor's quantity.
                    return 0.0
        sgn = 1 if side == "BUY" else -1              # sign of the position change: a BUY adds shares (+1), a SELL removes them (-1).
        self.pos += sgn * take                        # update our position: +take if we bought, -take if we sold.
        self.cash += -sgn * take * o.price - fee_for(o.price, take)   # update cash: money moves OPPOSITE to position (buying spends cash, selling earns it), always at OUR limit price o.price -- then subtract the fee. Note: fee uses o.price, the price we transacted at.
        self.fills.append({"t": t_exch, "side": side, "px": o.price, "qty": take,   # log this fill: time, side, OUR price, and the amount filled...
                           "reason": reason,          # ...plus WHY it filled (through/at-queue/crossing-add) so you can audit which fill rule produced which PnL.
                           # ...plus WHICH trigger window the strategy was in when it
                           # filled (none/time_ramp/time_cliff/lock_ramp/lock_cliff).
                           # Read-only label set by the strategy each quote cycle;
                           # strategies without the attribute (naive) tag "none".
                           # Answers: "did we actually sell during the cliffs?"
                           "window": getattr(self.strat, "current_window", "none"),
                           # ...and WHICH time-of-day bucket (first15/middle/
                           # preclose45/last15) for per-bucket net_bps analysis.
                           "bucket": getattr(self.strat, "current_bucket", "middle"),
                           "regime": getattr(self.strat, "current_regime", "normal"),
                           # ...plus WHICH order this fill consumed (lifecycle join
                           # key for time-to-fill / queue-wait in the harness).
                           "oid": o.oid})
        o.qty -= take                                 # reduce our order's remaining quantity by what just filled.
        if o.qty <= 0:                                # if the order is now fully filled...
            self._drop_order(side, o)                 # ...remove THAT order (it's done). A partial fill leaves it in place with reduced qty, and anything else of ours on this side is untouched.
            # LIFECYCLE: the order ended by being fully filled at this time
            if o.oid in self._olog:
                self._olog[o.oid]["t_end"] = t_exch
                self._olog[o.oid]["end_reason"] = "filled"
        # HOW MUCH OF THE AGGRESSOR'S QUANTITY THIS CONSUMED. The fill engine
        # subtracts it before offering the rest to the next order in priority,
        # which is the whole reason this returns a number.
        return take

    def _on_market_trade(self, r):
        """A historical trade printed. Could its aggressive flow have hit us?

        Eligibility gates, in order:
          * AUCTION prints have no continuous-market aggressor -> skip.
          * We must have a working order on the PASSIVE side (a BUY
            aggressor hits asks -> checks our SELL order, and vice versa).
          * The order must still be live: if a cancel is in flight,
            fills stop the instant it lands (r.ts_exch >= cancel_at).
            Before that instant the order is fair game — that is the
            in-flight cancel risk being modelled, not a bug.

        Then three price cases:
          THROUGH — the print is at a price WORSE (for the aggressor) than
            ours: a BUY paid 407.31 while our ask sat at 407.20. Price
            priority is absolute: any order willing to pay 407.31 takes
            407.20 first. Certain fill, capped at the trade's qty.
          AT our price — queue position decides, per at_price_mode:
            'never'/'always' are the bracketing bounds; 'queue' consumes
            the ahead-pool first. If the trade row names its resting
            victim (rest_oid) and that victim is in our ahead dict, exactly
            that entry is drained — surgical. Otherwise qty drains the
            pool front-to-back (order within the pool is unknowable but
            irrelevant: only the TOTAL ahead of us gates our fill).
            Whatever aggressive qty survives the pool reaches us.
          BEHIND our price — the print didn't reach our level; no fill.
            (Implicit: neither branch triggers.)
        """
        # Auction prints have no continuous-market aggressor -> skip.
        if r.initiator == "AUCTION":
            return
        # ---- TICK STATE, maintained on every continuous print -------------
        # Computed BEFORE anything else uses it, because the tick of THIS
        # trade is defined against the price of the PREVIOUS one.
        # PSX Regulations, Chapter 1 definitions:
        #   Uptick         "the price of a Security above the last executed
        #                   price of that Security transacted through the
        #                   Trading System"
        #   Zero-Tick      the price with no difference from the last executed
        #   Zero-Plus Tick "the price without any difference in the previous
        #                   price of a trade of a security, WHICH WAS AN UPTICK"
        # so a zero tick only counts as zero-PLUS when the last actual move
        # was upward -- which is why the direction is tracked separately.
        # the price this print executed at
        _px_now = float(r.price)
        # the reference the rules measure against: the previous executed price
        _prev_exec = self._last_exec_px
        # advance the direction only on an actual change, so a run of equal
        # prices keeps pointing at whichever way the last real move went
        if _prev_exec is not None and _px_now > _prev_exec:
            # this print moved the price up
            self._last_tick_dir = 1
        elif _prev_exec is not None and _px_now < _prev_exec:
            # this print moved the price down
            self._last_tick_dir = -1
        # remember this print as the reference for the next one
        self._last_exec_px = _px_now
        # ---- PAY FOR THE OPENING BUFFER, once, at the first print ---------
        # Deferred to here rather than done in __init__ because the acquisition
        # price is not known until the market has printed something. Charged
        # like any other purchase: cash out at the price, plus the per-side fee.
        if self._buffer_unpaid:
            # THE TOUCH, READ BEFORE THE CASH MOVES. This is the pre-event
            # book: run() calls _on_market_trade before applying the trade to
            # the book, so bbo() here is the book the print hit, which is the
            # book a real order would have faced. Read first so nothing below
            # can disturb it.
            _bbid, _, _bask, _ = self.book.bbo()
            # cash moves opposite to the position, at the first traded price
            self.cash -= self.pos * _px_now
            # and the buy pays the same fee schedule every other fill pays
            self.cash -= fee_for(_px_now, abs(self.pos))
            # remember WHAT WE PAID, so the sweep can strip out the stock's
            # move afterwards and leave the quoting result behind
            self.buffer_px = _px_now
            # and what the market was showing at that instant, so the size of
            # the free lunch can be measured rather than argued about
            self.buffer_bid = _bbid
            self.buffer_ask = _bask
            # paid for; never again
            self._buffer_unpaid = False
        # ARRIVAL-RATE: log this trade (ts, aggressor side, qty) for the recent-rate
        # estimate. Guarded -> off = no buffer maintenance (byte-identical).
        if self.log_fill_state and r.aggressor_side in ("BUY", "SELL"):
            # append this print
            self._arr_buf.append((r.ts_exch, r.aggressor_side, float(r.qty)))
            # bound the buffer to the largest window's trade count
            if len(self._arr_buf) > self._arr_max_n:
                # drop the oldest
                self._arr_buf.pop(0)
        # Which side was the aggressor (taker): BUY lifted an ask, SELL hit a bid.
        aggr = r.aggressor_side
        # PASSIVE side is the opposite: BUY aggressor hits resting SELLs, SELL hits BUYs.
        # This is the side OUR order must be on to get hit.
        passive_side = {"BUY": "SELL", "SELL": "BUY"}.get(aggr)
        # EVERY ORDER OF OURS ON THE PASSIVE SIDE that the flow could reach.
        # Skipped here, per order: one still on the wire (t_active is None, so
        # nothing can hit it), and one whose in-flight cancel has ALREADY
        # landed (trade time >= cancel_at). Before cancel_at the order is
        # still fillable -- that is in-flight cancel risk, modelled
        # deliberately.
        live = [o for o in self._side_orders(passive_side)
                if o.t_active is not None
                and not (o.cancel_at is not None
                         and r.ts_exch >= o.cancel_at)]
        # nothing of ours could be hit
        if not live:
            return
        # The trade's execution (print) price.
        px = float(r.price)
        # HOW MUCH OF THE AGGRESSOR'S QUANTITY IS STILL AVAILABLE. It is
        # consumed as it works through the levels, so an order deeper in the
        # queue only sees what survived the ones in front of it.
        flow = float(r.qty)
        # PRICE FIRST, THEN TIME -- the exchange's own priority rule, and the
        # reason this is a sort rather than a loop over the list as stored. On
        # the SELL side the lowest ask is hit first; on the BUY side the
        # highest bid. Ties break on send order, which is the list's own
        # order, so `index` carries time priority.
        order_index = {id(o): k for k, o in
                       enumerate(self._side_orders(passive_side))}
        live.sort(key=lambda o: ((o.price if passive_side == "SELL"
                                  else -o.price), order_index[id(o)]))
        # OUR OWN ORDERS AT ONE PRICE SHARE ONE QUEUE. Each order carries its
        # own `ahead` dict, snapshotted from the book when it arrived, and the
        # historical orders in front of it are the SAME orders for both of
        # ours at that price. Draining each dict separately would let the
        # market queue be consumed twice. So the level is drained ONCE, using
        # the OLDEST of our orders there, and what survives is then offered to
        # ours in time order.
        levels = {}
        # group our live orders by price, preserving the sorted priority order
        for o in live:
            levels.setdefault(o.price, []).append(o)
        # walk the levels in the priority order the sort established
        for level_px in dict.fromkeys(o.price for o in live):
            # every order of ours at this price, oldest first
            ours = levels[level_px]
            # the aggressor is spent
            if flow <= 0:
                break
            # DID THE TRADE PRINT PAST THIS LEVEL? SELL: print above our ask.
            # BUY: print below our bid. A through-print forces a fill by price
            # priority -- the exchange could not have skipped us.
            through = (px > level_px) if passive_side == "SELL" else (px < level_px)
            # THROUGH -> certain fill, oldest of ours first
            if through:
                # offer the surviving flow to each of our orders in turn
                for o in ours:
                    # spent
                    if flow <= 0:
                        break
                    # fill it, and take off what it actually consumed
                    flow -= self._fill(passive_side, px, flow, r.ts_exch,
                                       "through", order=o)
                # on to the next level
                continue
            # PRINTED EXACTLY AT THIS LEVEL -> queue / time priority decides
            if px == level_px:
                # At-price fill policy: "never" | "always" | "queue".
                mode = self.cfg["at_price_mode"]
                # Conservative lower bound: never fill at our price.
                if mode == "never":
                    continue
                # Optimistic upper bound: assume we are first in line.
                if mode == "always":
                    # each of ours in turn, until the flow is spent
                    for o in ours:
                        # spent
                        if flow <= 0:
                            break
                        # fill it, and take off what it consumed
                        flow -= self._fill(passive_side, px, flow, r.ts_exch,
                                           "at_optimistic", order=o)
                    # on to the next level
                    continue
                # "queue" mode (realistic). THE LEVEL'S QUEUE IS THE OLDEST OF
                # OUR ORDERS' `ahead` DICT -- see the note above on why it is
                # drained once rather than per order.
                lead = ours[0]
                # SURGICAL DRAIN: the trade names the specific resting order it
                # hit, and it is in our queue.
                if r.rest_oid and r.rest_oid in lead.ahead:
                    # Consume the smaller of that order's remaining qty or the
                    # available flow.
                    take = min(lead.ahead[r.rest_oid], flow)
                    # Shrink that specific order in our queue.
                    lead.ahead[r.rest_oid] -= take
                    # If it is fully consumed, remove it from our queue-ahead.
                    if lead.ahead[r.rest_oid] <= 0:
                        del lead.ahead[r.rest_oid]
                    # Reduce remaining flow by what it consumed.
                    flow -= take
                # POOL DRAIN: id unknown or not in our queue -> front-to-back.
                else:
                    # Walk the orders ahead of us. list() allows safe deletion.
                    for k in list(lead.ahead):
                        # Flow exhausted before reaching us -> stop.
                        if flow <= 0:
                            break
                        # Consume the smaller of this order's qty or the flow.
                        take = min(lead.ahead[k], flow)
                        # Shrink this order and the remaining flow.
                        lead.ahead[k] -= take
                        flow -= take
                        # Order fully consumed -> remove it.
                        if lead.ahead[k] <= 0:
                            del lead.ahead[k]
                # WHAT SURVIVED THE MARKET QUEUE now reaches OUR orders at this
                # price, oldest first -- our own time priority among ourselves.
                for o in ours:
                    # spent
                    if flow <= 0:
                        break
                    # fill it, and take off what it consumed
                    flow -= self._fill(passive_side, px, flow, r.ts_exch,
                                       "at_queue", order=o)
                # on to the next level
                continue
            # WORSE PRICED THAN THE PRINT. Levels are walked best-first, so
            # every level after this one is worse too: nothing more can fill.
            break

    def _on_market_add(self, r):
        """A historical ORDER_ADD arrived. If its price CROSSES one of our
        quotes, it was marketable flow: had our quote truly been in the
        book, the exchange would have matched it against us instead of
        letting it rest. So (config permitting) we fill AT OUR PRICE.

        Historically that order rested only because the real touch was
        wider than ours. Filling it against us creates a small shadow
        inconsistency (the order also enters the historical book and lives
        its historical life) — accepted under the shadow-fill limitation
        in the header.

        Loop reads: for our SELL order, a crossing add is an opposing BUY
        at px >= our ask; for our BUY, a SELL at px <= our bid.
        """
        # Check both of our sides. opp = the opposing side an add must be on to cross us.
        # For our SELL (ask), a crossing add is a BUY; for our BUY (bid), it is a SELL.
        for side, opp in (("SELL", "BUY"), ("BUY", "SELL")):
            # the add is not on the side that could cross us
            if r.side != opp:
                continue
            # Only fill off crossing adds if the config enables this (it is optional/optimistic).
            if not self.cfg["fill_on_crossing_adds"]:
                continue
            # The incoming order's price.
            px = float(r.price)
            # HOW MUCH OF THE INCOMING ORDER IS STILL UNMATCHED. It is
            # consumed as it works through our orders, so the second of ours
            # it reaches only sees what the first left.
            flow = float(r.qty)
            # OUR ORDERS ON THIS SIDE, in the exchange's priority: best price
            # first, then oldest first. Sent order is the list's own order, so
            # the index carries time priority.
            idx = {id(o): k for k, o in enumerate(self._side_orders(side))}
            # only orders that have actually reached the exchange can be hit
            ours = sorted((o for o in self._side_orders(side)
                           if o.t_active is not None),
                          key=lambda o: ((o.price if side == "SELL"
                                          else -o.price), idx[id(o)]))
            # walk them until the incoming order is exhausted
            for o in ours:
                # nothing left of the incoming order
                if flow <= 0:
                    break
                # Does it cross us? Our SELL: add's buy price at/above our ask. Our BUY: add's sell price at/below our bid.
                if (side == "SELL" and px >= o.price) or (side == "BUY" and px <= o.price):
                    # It would have matched against us -> fill us AT OUR PRICE (o.price), tagged "crossing_add".
                    flow -= self._fill(side, o.price, flow, r.ts_exch,
                                       "crossing_add", order=o)
                else:
                    # priced best-first, so no later order can cross either
                    break

    def _on_market_cancel(self, r):
        """A historical order was cancelled. If it was queued AHEAD of one
        of our orders, our queue position just improved: remove it from
        every ahead dict it appears in. (It can only match at one price,
        so at most one dict actually contains it.)"""
        oid = str(r.order_id)
        for o in self._all_orders():
            o.ahead.pop(oid, None)

    def _on_snapshot_queue_reset(self):
        """A snapshot just REPLACED the historical book, so our carefully
        maintained ahead dicts reference a dead generation of the book.
        Rebuild each from the fresh book.

        FLAGGED SIMPLIFICATION (worst case): snapshots do not reveal
        arrival order, so every order now at our price is assumed AHEAD of
        us — including any that actually arrived after we did. Understates
        our priority, never overstates it. Cheap insurance given quotes
        inside a wide spread are usually alone at their price anyway.
        """
        for o in self._all_orders():
            # An order still on the wire has no queue position yet -- _arrive
            # snapshots it from the book at the instant it lands, which is the
            # only moment that snapshot is meaningful. Rebuilding it here would
            # hand it a position it does not hold.
            if o.t_active is None:
                continue
            o.ahead = self.book.qty_at(o.side, o.price)

    # ================ strategy plumbing (KNOWLEDGE side) ==================
    def _requote(self, ts_know):
        """Ask the strategy what it wants; reconcile with what's working.

        The strategy sees the book (bb, bq, ba, aq) and current position,
        and returns desired quotes {side: (price, qty)} — or omits a side
        to mean 'no quote there'.

        Reconciliation per side:
          unchanged (same px, same qty, no cancel already racing)
            -> do NOTHING. This no-churn check matters: the strategy is
               re-evaluated ~4,500 times/day and mustn't spam
               cancel/replace when its answer hasn't changed.
          changed or newly wanted
            -> cancel the incumbent (if any, and not already being
               cancelled) AND send the replacement. Each message gets an
               independent latency draw from self.lat; the cancel and its
               replacement land at different times.. Until the cancel
               lands the old order remains fillable (cancel_at gate in
               _on_market_trade); the new order only activates on arrival
               (_arrive), where it may still be rejected as crossing.

        Position note (flagged in header): self.pos is read in real time,
        i.e. the strategy 'knows' a fill the moment it happens rather than
        one ack-latency later. At ~100ms ack vs a 5.6s median trade gap
        the distortion is negligible; a production harness acks fills.
        """
        # HALT / BAND-PIN GATE: quote only in continuous trading, never into a
        # pinned band. On any other phase (TRADING_BREAK, OPEN_CALL_AUCTION,
        # MARKET_CLOSED...) pull all working quotes via the normal latency path.
        # FLAGGED ASSUMPTION: orders persist through halts until our cancel
        # lands; PSX may purge on some halt types -- verify with broker.

        # Quotable only in continuous trading AND when not pinned at a circuit limit.
        quotable = self.book.phase in (None, "CONTINUOUS_AUCTION") and not self.book.pinned()
        # STALE FEED -> STAND DOWN. Nothing has arrived for longer than a
        # working feed ever goes quiet, so the book in front of us describes a
        # market that has moved on without us. Pull everything and wait for a
        # snapshot, which is the only message that restores a book wholesale.
        # Counted separately from a halt: a halt is the exchange telling us to
        # stop, this is our own data having stopped arriving.
        if self._feed_stale:
            # count the requote refused for this reason
            self.stats["stale_feed_requotes"] = \
                self.stats.get("stale_feed_requotes", 0) + 1
            # and take the same path a halt takes
            quotable = False
        # CROSSED OR LOCKED BOOK -> STAND DOWN. bbo() is a single top-of-book
        # read, so testing it here costs nothing. `>=` covers both cases: a
        # crossed book (bid above ask) and a locked one (bid equal to ask).
        # Neither can rest at a continuously matching exchange, so both mean
        # the reconstruction is momentarily wrong, and a mid computed from
        # them is not a price. Falls through to the same pull-everything path
        # a halt uses -- which is also what the production engine does, so the
        # two agree.
        _gb, _, _ga, _ = self.book.bbo()
        # both sides present and inverted or equal
        if self.skip_crossed_book and _gb is not None and _ga is not None \
                and _gb >= _ga:
            # counted separately from halted_requotes: a halt is the exchange
            # telling us to stop, this is our own data being unusable
            self.stats["crossed_book_requotes"] = \
                self.stats.get("crossed_book_requotes", 0) + 1
            # treat it exactly as not quotable
            quotable = False
        # Not quotable (halt / auction / closed / pinned) -> pull all our quotes and stand down.
        if not quotable:
            # Count this halted requote for diagnostics.
            self.stats["halted_requotes"] += 1
            # Cancel every working order that isn't already being cancelled.
            for side, cur in [(s_, o_) for s_ in ("BUY", "SELL")
                              for o_ in list(self._side_orders(s_))]:
                # Skip orders with ANY message of ours outstanding: a cancel
                # already in flight, an order that has not reached the exchange
                # yet, or an amendment awaiting an answer. In each case there
                # is nothing we may send, and the order manager holds in
                # exactly the same three cases. Whichever message is
                # outstanding lands, and the next requote -- still not
                # quotable -- pulls the order then.
                if (cur.cancel_at is None and cur.t_active is not None
                        and cur.amend_at is None):
                    # Draw a send latency for this cancel.
                    a_out = self.lat.draw_out()
                    # The cancel lands (exchange stops matching) at knowledge time + latency.
                    cur.cancel_at = ts_know + a_out
                    # Schedule the cancel to land, targeting this specific order (side, oid).
                    self._push(cur.cancel_at, "CANCEL", (side, cur.oid))
                    # MESSAGE LOG: a halt-pull cancel is an outbound message too
                    self._msg_ts.append(ts_know)
                    # In stochastic mode, mark this side unconfirmed until the ack returns.
                    if self.use_ack:
                        self.ack_until[side] = cur.cancel_at + self.lat.draw_ack()
            # Done -- no new quotes while not quotable.
            return

        # Read the current best bid/offer (with qtys) from the reconstructed book.
        bb, bq, ba, aq = self.book.bbo()
        # DEEP-OFI SUPPORT: if the strategy uses multi-level OFI (ofi_depth_levels
        # > 1), supply ranked (price, qty) depth; otherwise pass None (L1 path,
        # zero extra cost -- byte-identical to the pre-deep-OFI call). getattr
        # guards strategies that lack the attribute entirely.
        _depth = (self.book.ranked_depth(getattr(self.strat, "ofi_depth_levels", 1))
                  if getattr(self.strat, "ofi_depth_levels", 1) > 1 else None)
        # Ask the strategy where it wants to quote: {side: (price, qty)}, sides may be omitted.
        want = self.strat.quotes(bb, bq, ba, aq, self.pos, depth=_depth)

        # BAND CLAMP: exchange rejects orders outside [limit_dn, limit_up]
        # Clamp each desired price into the allowed circuit band before sending.
        for side in ("BUY", "SELL"):
            # The desired quote for this side (or None if the strategy omitted it).
            w = want.get(side)
            # Only clamp if the strategy actually wants a quote on this side.
            if w is not None:
                # The desired price.
                px = w[0]
                # Never send above the upper circuit limit.
                if self.book.limit_up is not None:
                    px = min(px, self.book.limit_up)
                # Never send below the lower circuit limit.
                if self.book.limit_dn is not None:
                    px = max(px, self.book.limit_dn)
                # Write the clamped, tick-rounded price back into the desired quote.
                want[side] = (round(px, 2), w[1])

        # Reconcile desired quotes against what we already have working, per side.
        for side in ("BUY", "SELL"):
            # ACK GUARD (stochastic mode only): while this side's last cancel is
            # unconfirmed (sent, ack not back), don't stack another cancel/replace
            # on it -- a real risk system treats the old order as possibly-live
            # until acked. The original cancel+replace pair is unaffected.

            # If this side is still awaiting a cancel ack, skip repricing it for now.
            if self.use_ack and ts_know < self.ack_until[side]:
                # Count the blocked requote for diagnostics.
                self.stats["requotes_blocked_by_ack"] += 1
                continue
            # The desired quote for this side (or None).
            w = want.get(side)
            # Our current working order on this side (or None).
            # THE LEAD ORDER, which under this file's own requote logic is the
            # only one: mm_backtest never rests more than one order per side.
            # The list exists for the production order manager, which drives
            # the same exchange through sim/replay.py and does rest two.
            cur = self._lead(side)
            # ---- IN FLIGHT: WAIT --------------------------------------------
            # THE ORDER HAS BEEN SENT AND HAS NOT REACHED THE EXCHANGE YET
            # (t_active is None until _arrive sets it). Nothing can be done
            # with it: it cannot be cancelled, because PSX requires the
            # exchange's OrderID on a cancel and that only exists once the
            # order has been acknowledged; it cannot be amended, for the same
            # reason; and it must not be duplicated, which is exactly what the
            # old code did every cycle of the latency window.
            #
            # This one test is what takes the duplicate-send count from 4,901
            # to zero. The production order manager expresses the identical
            # rule as Order.has_message_in_flight.
            #
            # AN AMENDMENT ON THE WIRE COUNTS THE SAME WAY, and until
            # 2026-09-17 it did not. The order was live, so t_active was set,
            # and a reprice arriving while a CFO was outstanding could not send
            # a second CFO (amend_at blocks that) -- so it fell through to the
            # cancel path and PULLED THE QUOTE instead. The order manager does
            # no such thing: an order with any message outstanding is left
            # exactly where it is until the exchange answers.
            #
            # sim/diff_fills.py found it on NRL 2026-06-30. At 1782807500800
            # the engine sold 10 shares at 364.34, more than two rupees above
            # the day's mid, off an order it had left resting. mm_backtest had
            # cancelled its own and had nothing there. Over the day that one
            # difference in rule is worth 0.155 PKR a share on both sides of
            # the round trip.
            if cur is not None and (cur.t_active is None
                                    or cur.amend_at is not None):
                # count it, so a run can show how much of the session each side
                # spent waiting on the wire rather than quoting
                self.stats["requotes_blocked_in_flight"] += 1
                # nothing to send for this side this cycle
                continue
            # "same" = we already have exactly this quote live, with no cancel racing -> no churn.
            same = cur is not None and w is not None and \
                   cur.price == w[0] and cur.qty == w[1] and cur.cancel_at is None
            # If nothing changed, do nothing (avoid needless cancel/replace spam).
            if same:
                continue
            # ---- CFO PATH (use_cfo=True): ONE message instead of two -----
            # Only when we have a live incumbent that is not already being
            # cancelled AND we still want a quote on this side. Pulling a side
            # entirely is a Cancel Order (8.11), not a CFO, so that falls
            # through to the original path below.
            # FIVE conditions, and the fifth was missing until the first real
            # run: the switch is on, there IS an incumbent to modify, it is not
            # already being cancelled, NO AMENDMENT IS ALREADY IN FLIGHT for
            # it, and we still want a quote here (pulling a side is a Cancel,
            # not a CFO).
            #
            # WHY THE FOURTH CONDITION IS LOAD-BEARING. Under cancel-plus-new,
            # setting cur.cancel_at doubles as the in-flight marker: the next
            # cycle sees it and does not cancel twice. A CFO never sets
            # cancel_at -- the order stays live at the old terms until the
            # amendment lands -- so without amend_at the no-churn check still
            # sees a mismatched price, fires again, and keeps firing every
            # cycle until the message arrives. The first smoke run showed it:
            # 11,622 messages against 9,714 on the baseline, when one message
            # replacing two should have sent FEWER.
            if self.use_cfo and cur is not None and cur.cancel_at is None \
                    and cur.amend_at is None and w is not None:
                # one send-latency draw, because this is one message
                a_out = self.lat.draw_out()
                # when the amendment reaches the exchange
                t_land = ts_know + a_out
                # the amended version gets its own id, so a later cancel
                # targets the right generation and the lifecycle log stays
                # per quote-version
                self._oid += 1
                # record the send time so msgs/sec peaks can be computed from
                # timestamps rather than day totals.
                self._msg_ts.append(ts_know)
                # count it as an order sent, so message-rate reporting stays
                # comparable between the one-message and two-message paths.
                self.stats["n_orders_sent"] += 1
                # AND SEPARATELY, because n_orders_sent counts new orders AND
                # amendments together and a reader cannot tell them apart.
                self.stats["n_amends_sent"] = (
                    self.stats.get("n_amends_sent", 0) + 1)
                # schedule the amendment to land; the payload carries the
                # target generation (cur.oid) so a CFO that loses a race to a
                # fill is recognised and rejected rather than misapplied.
                self._push(t_land, "AMEND",
                           (side, cur.oid, w[0], w[1], self._oid))
                # MARK IT IN FLIGHT. Until this lands, no further amendment is
                # sent for this order. Note this does NOT gate fills the way
                # cancel_at does -- the old terms stay matchable, which is the
                # real exposure of an amendment.
                cur.amend_at = t_land
                # THE OLD TERMS STAY LIVE AND FILLABLE UNTIL THIS LANDS. That
                # is the real exposure of an amendment and the one thing that
                # differs from cancel-plus-new, where an early-arriving
                # replacement could cut the old order short.
                # this side is done for this cycle: one message covers both
                # the cancel and the replacement, so skip the two-message path.
                continue
            # ---- CANCEL PATH: pull the incumbent, and DO NOT replace it in
            # the same cycle --------------------------------------------------
            # THE REPLACEMENT NOW WAITS FOR THE CANCEL TO LAND. The old code
            # fell straight through to the new-order block below, so a cancel
            # and its replacement went out together with INDEPENDENT latency
            # draws. Whenever the replacement's draw was the shorter one it
            # landed first and overwrote an order that was still resting and
            # not yet cancelled -- the second source of the 5,205 orphans the
            # gate measured, and the reason orphans outnumbered duplicate
            # sends on three of four days.
            #
            # Waiting is also what the production order manager does, and it
            # is what PSX forces anyway: a cancel needs the exchange's OrderID,
            # which only exists once the order has been acknowledged.
            #
            # THIS IS A REAL BEHAVIOUR CHANGE, NOT ONLY A BUG FIX. A reprice
            # under cancel-plus-new now costs a full round trip instead of
            # being simultaneous, so fills and P&L will move. Under CFO
            # (use_cfo=True) the path above handles a reprice in one message
            # and is unaffected.
            if cur is not None:
                # already being cancelled: nothing to send, just wait
                if cur.cancel_at is not None:
                    continue
                # Independent send-latency draw for the cancel.
                a_out = self.lat.draw_out()  # independent cancel-send draw
                # Cancel lands (exchange stops matching) at knowledge time + latency.
                cur.cancel_at = ts_know + a_out  # exchange stops matching here
                # Schedule the cancel to land, targeting this specific order (side, oid).
                self._push(cur.cancel_at, "CANCEL", (side, cur.oid))
                # MESSAGE LOG: a cancel send is an outbound message (msgs/sec
                # peaks need timestamps of every message, not just day totals)
                self._msg_ts.append(ts_know)
                # In stochastic mode, this side stays unconfirmed until the ack returns.
                if self.use_ack:  # confirmed only after ack
                    self.ack_until[side] = cur.cancel_at + self.lat.draw_ack()
                # the side is now busy until the cancel lands and pops it out
                # of self.work; the replacement goes out on a later cycle
                continue
            # ---- PLACE PATH: the side is genuinely empty ---------------------
            # If the strategy wants a quote on this side, send it.
            if w is not None:
                # REGRESSION GUARD. With the order written into self.work at
                # SEND time and the in-flight rule at the top of this loop,
                # this can no longer fire. It is kept because it is what
                # measured the defect in the first place, and a non-zero value
                # in a later run means the hole has reopened.
                if self._new_in_flight[side] > 0:
                    # one more order sent on top of one still in the air
                    self.stats["orders_sent_while_new_in_flight"] += 1
                # this side now has one more order in the air
                self._new_in_flight[side] += 1
                # Count an order sent.
                self.stats["n_orders_sent"] += 1
                # AND SEPARATELY, as a NEW order. n_orders_sent is the sum of
                # this and n_amends_sent; reporting only the sum made the
                # amendment count look like a separate category when it is
                # part of the same total.
                self.stats["n_new_orders_sent"] = (
                    self.stats.get("n_new_orders_sent", 0) + 1)
                # Independent send-latency draw for the new order (separate from the cancel).
                a_out = self.lat.draw_out()  # independent new-order draw
                # The new order lands (becomes eligible to rest) at knowledge time + latency.
                t_land = ts_know + a_out
                # Assign a unique id so a future cancel can target THIS specific order.
                self._oid += 1
                # MESSAGE LOG: an order send is an outbound message
                self._msg_ts.append(ts_know)
                # LIFECYCLE LOG: one record per order, keyed by oid. t_sent = the
                # knowledge-time we decided to send; t_live set at _arrive;
                # t_end + end_reason set at cancel-land / full fill.
                self._olog[self._oid] = {"oid": self._oid, "side": side,
                                         "px": w[0], "qty": w[1],
                                         "t_sent": ts_know, "t_live": None,
                                         "t_end": None, "end_reason": None}
                # TAKER TAG: the strategy sets want_taker_side to the side it wants to
                # CROSS with (a deliberate flatten); any other order is a passive quote.
                _tk = (getattr(self.strat, "want_taker_side", None) == side)
                # THE ORDER OBJECT, built once and shared. t_active=None is the
                # marker that it is SENT BUT NOT YET LIVE: every fill path
                # tests for it and skips, so an order in the air cannot be hit.
                # _arrive sets it to the landing time, and from that instant the
                # order is matchable.
                _o = MyOrder(side, w[0], w[1], {}, None,
                             oid=self._oid, taker=_tk)
                # RESERVE THE SIDE NOW, AT SEND TIME. This is the fix. Until
                # this line existed, self.work[side] stayed empty for one whole
                # network latency after every send, the next requote read the
                # side as empty, and sent another order -- 4,901 of them across
                # four symbol-days. The in-flight rule at the top of the
                # reconcile loop now sees this entry and waits.
                self._side_orders(side).append(_o)
                # Schedule its arrival; the empty {} is the queue-ahead dict,
                # filled from the book at _arrive.
                self._push(t_land, "ARRIVE", _o)

    # ============================ main loop ================================
    def run(self, events, snap_groups):
        """Single pass over the merged event stream: the main replay loop.

        Two clocks are in play throughout (the core anti-lookahead rule):
        `ts_exch` (exchange time) drives the book and all fill decisions,
        while `know` (running max of ts_cap) drives what the strategy is
        allowed to see and act on.

        Per event, IN ORDER:

        1. _activate_until: land our in-flight messages due before this
           event (exchange timeline catches up).
        2. Fill checks against the PRE-event book: trades test our resting
           orders; adds test for crossings; cancels update our queues.
           (Must precede step 3 — a fill is judged against the book the
           aggressor actually hit.)
        3. Apply the event to the historical book (snapshot = full replace,
           then rebuild our queue dicts; update = add/cancel; trade =
           consume liquidity).
        4. Mark to market: equity = cash + pos * mid, plus the OBI feature
           columns (obi_5 near-touch, obi_deep all visible levels). One row
           per event that has a valid TWO-SIDED book; events during
           one-sided or empty books are skipped, so the equity curve can be
           shorter than the event stream.
        5. Advance knowledge time (running max of ts_cap — receive
           timestamps jitter, knowledge never runs backwards) and, inside
           the session window, let the strategy requote. _requote itself
           applies the halt/band-pin gate and the circuit-band clamp, so
           quotes are pulled automatically outside continuous trading.
           After session end, pull all working orders.

        FLAGGED SIMPLIFICATION: the end-of-session pull is instant and
        latency-free, and final inventory is marked at the last mid — the
        closing auction and a book-walk liquidation are not modelled.

        Returns (fills DataFrame, equity DataFrame, stats dict).
        """
        # Session window: quote only between t0 and t1 (exchange-ms).
        t0, t1 = self.cfg["session"]
        # Knowledge time: earliest wall-clock ms by which we could know the current state.
        know = 0
        # Walk the merged, time-ordered event stream. kind: "S"=snapshot, "U"=update, "T"=trade.
        for ts_exch, _, _, kind, obj in events:
            # ---- STALE-FEED DETECTION, before anything else -------------
            # Measured on the CAPTURE clock, because the question is how long
            # WE went without hearing anything -- not what the exchange's own
            # timestamps say about messages we never received.
            if self.stale_feed_seconds > 0:
                # this event's arrival time
                _cap = int(obj.ts_cap)
                # how long since the previous one arrived, in seconds
                if self._last_event_cap is not None:
                    # the silence that just ended
                    _silence = (_cap - self._last_event_cap) / 1000.0
                    # longer than the threshold means the receiver had stopped
                    if _silence > self.stale_feed_seconds:
                        # mark the book untrustworthy until a snapshot resets it
                        self._feed_stale = True
                        # count the window and how much time it cost
                        self.stats["stale_feed_windows"] = \
                            self.stats.get("stale_feed_windows", 0) + 1
                        self.stats["stale_feed_seconds"] = \
                            self.stats.get("stale_feed_seconds", 0.0) + _silence
                        # ---- LEDGER THE WINDOW -----------------------
                        # The touch we last saw. The book has NOT yet been
                        # updated with this event (that happens at step 3
                        # below), so bbo() here is the PRE-silence book.
                        _pb, _, _pa, _ = self.book.bbo()
                        # a mid only exists if both sides were present
                        _mid_before = ((_pb + _pa) / 2.0
                                       if (_pb is not None and _pa is not None)
                                       else None)
                        # WHERE THE SILENCE SAT ON THE CLOCK. It spans
                        # [ts_exch - silence, ts_exch]; clipping that to the
                        # session window means a gap straddling the open or
                        # the close contributes only its in-session part,
                        # which is the only part any denominator cares about.
                        _sil_start = ts_exch - int(_silence * 1000.0)
                        # overlap of the silence with [t0, t1], in ms
                        _ov_ms = min(ts_exch, t1) - max(_sil_start, t0)
                        # negative means the silence fell wholly outside the
                        # session, and contributes nothing
                        _in_sess = max(0.0, _ov_ms / 1000.0)
                        # the running total every time-denominated metric
                        # must subtract
                        self.stats["stale_feed_seconds_in_session"] = (
                            self.stats.get("stale_feed_seconds_in_session",
                                           0.0) + _in_sess)
                        # WHAT WE HAD RESTING WHEN THE LIGHTS WENT OUT.
                        # Live orders only: one not yet active could not have
                        # been hit, and one with a cancel already in flight is
                        # a different exposure. (side, price, qty) is all the
                        # exposure calculation below needs.
                        _resting = [
                            (o.side, float(o.price), float(o.qty))
                            for o in self._all_orders()
                            if o.t_active is not None
                            and o.t_active <= ts_exch
                            and o.cancel_at is None]
                        # open the record. mid_after and the exposure are
                        # filled in at the mark-to-market step below, which is
                        # the first moment the post-silence touch exists.
                        self.blind_windows.append({
                            # exchange-ms of the first event after the silence
                            "t_exch_end": int(ts_exch),
                            # the full silence
                            "seconds": _silence,
                            # the part of it inside the session
                            "seconds_in_session": _in_sess,
                            # the market we left
                            "mid_before": _mid_before,
                            # the market we came back to (filled below)
                            "mid_after": None,
                            # our exposure going in
                            "resting": _resting,
                            # the adverse selection the backtest was spared
                            # (filled below; None when it cannot be measured)
                            "spared_adverse_pkr": None})
                        # this record is the one awaiting its closing mid
                        self._pending_blind = len(self.blind_windows) - 1
                # remember this arrival for the next comparison
                self._last_event_cap = _cap
                # A SNAPSHOT IS THE ONLY THING THAT CLEARS IT. It replaces the
                # book wholesale, which is exactly what a book known to be
                # stale needs. An incremental update applied to a stale book
                # leaves it stale, so those do not clear the flag.
                if self._feed_stale and kind == "S":
                    # the book is trustworthy again from here
                    self._feed_stale = False
            # (1) Land any of OUR in-flight orders/cancels due before this event.
            self._activate_until(ts_exch)
            # (2) FILL CHECKS -- run against the PRE-event book, before it mutates below.
            # A trade: could its aggressive flow have hit our resting quote?
            if kind == "T":
                self._on_market_trade(obj)
            # An update: either a new order (may cross us) or a cancel (shrinks our queue).
            elif kind == "U":
                # A crossing add is marketable flow that would have matched against us.
                if obj.event == "ORDER_ADD":
                    self._on_market_add(obj)
                # A cancel of an order ahead of us improves our queue position.
                else:
                    self._on_market_cancel(obj)
            # (3) APPLY the event to the historical book.
            # Snapshot: full replacement of the book, then rebuild our queue dicts.
            if kind == "S":
                # KEYED BY snap_key, NOT msg_seq -- msg_seq repeats within a
                # day and two messages would be merged into one book. A loader
                # that has not been updated will raise AttributeError here
                # rather than silently looking up the wrong snapshot, which is
                # the failure mode worth having.
                self.book.snapshot(snap_groups[obj.snap_key])
                self._on_snapshot_queue_reset()
            # Update: dispatch to add or cancel on the book.
            elif kind == "U":
                (self.book.add if obj.event == "ORDER_ADD" else self.book.cancel)(obj)
            # Otherwise it's a trade: consume the resting liquidity it hit.
            else:
                self.book.trade(obj)
            # (4) MARK TO MARKET -- read the post-event touch.
            bb, _, ba, _ = self.book.bbo()
            # ---- CLOSE ANY BLIND WINDOW WAITING ON ITS EXIT MID ---------
            # The book now carries the first post-silence event, so this
            # touch is the market we came back to. Done OUTSIDE the
            # two-sided gate below so a one-sided return still closes the
            # record -- with mid_after None, which reads as "not measurable"
            # rather than silently leaving the record open and letting the
            # next window overwrite it.
            if self._pending_blind is not None:
                # the record opened when the silence was detected
                _rec = self.blind_windows[self._pending_blind]
                # the mid we came back to, if there is one
                _m_after = ((bb + ba) / 2.0
                            if (bb is not None and ba is not None) else None)
                # record it either way
                _rec["mid_after"] = _m_after
                # the exposure calculation needs a price to compare against
                if _m_after is not None:
                    # WHAT THE BACKTEST WAS HANDED FOR FREE. Every order we
                    # had resting sat through a window in which the engine
                    # saw no events, so it checked no fills -- it could not
                    # be hit, by construction. This puts a number on that.
                    _spared = 0.0
                    # each order that was live when the feed stopped
                    for _sd, _px, _q in _rec["resting"]:
                        # our BID, market now BELOW it: anyone could have
                        # sold into us at _px during the blackout, leaving
                        # us long at _px with the market at _m_after
                        if _sd == "BUY" and _m_after < _px:
                            _spared += _q * (_px - _m_after)
                        # our OFFER, market now ABOVE it: the same in reverse
                        elif _sd == "SELL" and _m_after > _px:
                            _spared += _q * (_m_after - _px)
                    # store it on the record
                    _rec["spared_adverse_pkr"] = _spared
                    # and accumulate for the EOD summary
                    self.stats["blind_spared_adverse_pkr"] = (
                        self.stats.get("blind_spared_adverse_pkr", 0.0)
                        + _spared)
                # the record is closed either way
                self._pending_blind = None
            # Only record a row when both sides exist (skips pre-open, halts, one-sided books).
            if bb is not None and ba is not None:
                # Mid price used for marking inventory.
                mid = (bb + ba) / 2
                # Remember this as the last reliable reference price.
                self.last_good_mid = mid
                # One equity row per event: PnL state plus the OBI features for later
                # analysis. GATED: obi(5)/obi(None) scan book levels EVERY event and
                # dominate runtime (~50x). Consumers that only need fills (e.g. the
                # capture sweep) set cfg["log_equity"]=False to skip this entirely --
                # fills are computed in steps (1)-(3) ABOVE and are unaffected.
                if self.log_equity:
                    # best bid/ask at this event, for realized ticks-inside-the-
                    # touch analysis (the skew sweep needs bb/ba to measure how far
                    # inside the touch each fill actually landed). Cheap: bbo() is a
                    # single top-of-book read, unlike the obi() level scans below.
                    _lbb, _, _lba, _ = self.book.bbo()
                    self.equity.append({"t": ts_exch, "mid": mid,
                                        "equity": self.cash + self.pos * mid,
                                        "pos": self.pos,
                                        "bb": _lbb, "ba": _lba,
                                        "obi_5": self.book.obi(5),
                                        "obi_deep": self.book.obi(None)})
                else:
                    # cheap path: keep only the mark needed for EOD inventory marking
                    self.equity.append({"t": ts_exch, "mid": mid,
                                        "equity": self.cash + self.pos * mid,
                                        "pos": self.pos})

            # Feed the event to the strategy so it can calibrate sigma / flow / quiet time.
            if hasattr(self.strat, "observe"):
                _bb, _, _ba, _ = self.book.bbo()
                _mid = (_bb + _ba) / 2 if (_bb is not None and _ba is not None) else None
                self.strat.observe(kind, obj, ts_exch, _mid)
                # Sync the published circuit-breaker band onto strategies that run
                # the distance-to-lock trigger. Duck-typed: micro_mm sets
                # wants_limits=True; naive lacks the attribute -> getattr returns
                # False -> skipped at near-zero cost.
                if getattr(self.strat, "wants_limits", False):
                    self.strat.limit_up = self.book.limit_up
                    self.strat.limit_dn = self.book.limit_dn

            # (5) Advance knowledge time. cummax: receive times jitter, knowledge never rewinds.
            know = max(know, int(obj.ts_cap))
            # Inside the session window: let the strategy react to this event.
            if t0 <= ts_exch <= t1:
                self._requote(know)
            # Past session end: pull every working quote (instant -- flagged simplification).
            elif ts_exch > t1:
                for side in ("BUY", "SELL"):
                    self._side_orders(side).clear()
                # EOD FLATTEN (production): the FIRST time we cross session end,
                # liquidate remaining inventory by walking the real book instead of
                # marking it at mid. Mid-marking assumes an impossible exit at the
                # midpoint; walking the book pays the spread and eats successively
                # worse levels, which is what getting flat actually costs.
                if self.eod is None:
                    # Void all in-flight messages: nothing of ours may land, fill,
                    # or change position after this report (makes eod final).
                    self.pending.clear()
                    # position we are about to flatten (sign fixes the liq side)
                    pos_at_liq = self.pos
                    # Walk the book to flatten the position, fees included. The 4th
                    # return (liq_level_fills) is the per-level itemization of THIS
                    # walk -- same cash, now attributable fill-by-fill.
                    liq_cash, unfilled, vwap, liq_level_fills = \
                        self.book.liquidation_value(self.pos, fee_fn=fee_for)
                    # Read the closing touch for the comparison mid-mark.
                    bb_e, _, ba_e, _ = self.book.bbo()
                    # Mid only exists if both sides are present.
                    mid_e = (bb_e + ba_e) / 2 if (bb_e is not None and ba_e is not None) else None
                    # The report: both numbers side by side so the overstatement is visible.
                    # Residual the book could not absorb. Mark it at the last good
                    # mid with a haircut, because we demonstrably could NOT trade
                    # out of it -- and flag the run as not cleanly liquidated.
                    # Reporting cash as equity here books an unclosed position as profit.
                    ref = mid_e if mid_e is not None else self.last_good_mid
                    residual_mark = 0.0
                    if unfilled > 0 and ref is not None:
                        # Residual haircut: 3% off the closing mid. The residual is
                        # HELD OVERNIGHT (a genuine option for this desk), so the mark
                        # is fair value less ~one day's adverse move -- NOT a fire-sale
                        # discount. (Was 0.10, which over-penalised trapping names.)
                        haircut = self.cfg.get("unfilled_haircut_pct", 0.03)
                        sgn = 1.0 if self.pos > 0 else -1.0
                        residual_mark = sgn * unfilled * ref * (1.0 - sgn * haircut)
                    self.eod = {
                        # Position we carried into the close.
                        "pos_at_close": self.pos,
                        # Mid at session end (None if the book was one-sided).
                        "mid_at_close": mid_e,
                        # What the old mid-mark WOULD have reported (diagnostic only).
                        "equity_mid_mark": (self.cash + self.pos * mid_e) if mid_e is not None else None,
                        # False means equity_liquidated is an estimate, not a realisable number.
                        "liquidation_clean": (unfilled == 0),
                        "residual_marked": residual_mark,
                        "equity_liquidated": self.cash + liq_cash + residual_mark,
                        # Achieved liquidation vwap.
                        "liq_vwap": vwap,
                        # Slippage vs mid, per share.
                        "liq_slippage_per_sh": (
                            abs(vwap - mid_e) if (vwap is not None and mid_e is not None) else None),
                        # Size the visible book could not absorb -- genuinely unpriceable.
                        "unfilled_sh": unfilled,
                        # ---- BUFFER ISOLATION FIELDS ----------------------
                        # Shares of opening buffer this run started with (0.0
                        # when the policy is not long_buffer).
                        "buffer_qty": self.buffer_qty,
                        # What the buffer was bought at (the day's first print).
                        # None if the buffer was never paid for, which can only
                        # happen if the session produced no continuous print.
                        "buffer_px": self.buffer_px,
                        # The touch at that instant, carried through so the
                        # acquisition can be audited rather than trusted.
                        "buffer_bid": self.buffer_bid,
                        "buffer_ask": self.buffer_ask,
                        # WHAT THE FREE ACQUISITION IS WORTH, in PKR. A real
                        # market buy pays the ask and the fee on the ask; the
                        # model pays the print and the fee on the print. Both
                        # terms are included -- the fee is ad-valorem, so a
                        # cheaper price is also a cheaper fee, and leaving that
                        # out would understate the understatement.
                        #
                        # A NEGATIVE VALUE IS NOT A BUG. It means the print was
                        # at or above the ask, so the model paid MORE than a
                        # market order would have. That happens and it is worth
                        # seeing, so it is not clamped at zero.
                        #
                        # None on a one-sided book: there is no ask to compare
                        # against, and inventing one would be the exact kind of
                        # silent interpolation this file exists to avoid.
                        "buffer_acq_understated_pkr": (
                            self.buffer_qty * (self.buffer_ask - self.buffer_px)
                            + (fee_for(self.buffer_ask, abs(self.buffer_qty))
                               - fee_for(self.buffer_px, abs(self.buffer_qty)))
                            if (self.buffer_qty and self.buffer_px is not None
                                and self.buffer_ask is not None)
                            else None),
                        # THE DIRECTIONAL TERM: what the buffer made or lost
                        # purely because the stock moved between the first print
                        # and the close. Marked at the CLOSING MID, not at the
                        # liquidation vwap, on purpose -- the gap between mid
                        # and vwap is execution cost, which is a real cost of
                        # carrying the buffer and stays in the P&L.
                        "buffer_price_move": (
                            self.buffer_qty * (
                                (mid_e if mid_e is not None else self.last_good_mid)
                                - self.buffer_px)
                            if (self.buffer_qty and self.buffer_px is not None
                                and (mid_e is not None or self.last_good_mid is not None))
                            else 0.0),
                    }
                    # EQUITY WITH THE STOCK'S MOVE TAKEN OUT. Still carries the
                    # buffer's entry fee and its exit execution cost, so it is
                    # not "the buffer for free" -- it is the buffer without the
                    # coin flip. This is the column the policy comparison should
                    # be run on; equity_liquidated is the column that answers
                    # "what would the account have done", buffer bet included.
                    self.eod["equity_ex_buffer_move"] = (
                        self.eod["equity_liquidated"] - self.eod["buffer_price_move"])
                    # ---- BLIND-WINDOW ACCOUNTING, added 2026-09-17 --------
                    # READ THIS BEFORE USING THESE NUMBERS.
                    #
                    # A blind window is a stretch with NO messages at all. It
                    # is not a period in which the engine did the wrong thing;
                    # it is a period in which the engine did NOTHING, because
                    # run() is event-driven and there were no events. So
                    # "taking the window out of the results" has exactly one
                    # honest meaning and one dishonest one.
                    #
                    # HONEST: take the seconds out of every denominator. The
                    # engine could not trade in them, so P&L per hour, fills
                    # per minute and quoted-time fractions must be measured on
                    # effective_seconds, not on the full session. That is what
                    # the fields below give.
                    #
                    # DISHONEST: subtract a P&L number for the window. There
                    # is no P&L inside the window to subtract -- no fills
                    # happened, because nothing arrived to fill against. The
                    # inventory mark that jumps across the gap is real money:
                    # in live trading we genuinely hold that position through
                    # the blackout, and deleting it would flatter the result,
                    # not clean it.
                    #
                    # THE DISTORTION THAT IS ACTUALLY THERE, and that NO
                    # subtraction fixes: our orders rested through the window
                    # and could not be hit. A real book moved without us and
                    # would have picked them off. blind_spared_adverse_pkr
                    # sizes that gift. It is an UPPER BOUND -- it assumes
                    # every resting order was filled in full at its limit and
                    # marks the loss at the post-window mid. Treat it as "the
                    # result is overstated by at most this much", and if it is
                    # large relative to the day's P&L, the day is not
                    # trustworthy however clean the other diagnostics look.
                    # the session's full length in seconds
                    _sess_s = (t1 - t0) / 1000.0
                    # the part of it the engine was blind for
                    _blind_s = float(
                        self.stats.get("stale_feed_seconds_in_session", 0.0))
                    # both, so a consumer can recompute anything
                    self.eod["session_seconds"] = _sess_s
                    self.eod["blind_seconds"] = _blind_s
                    # THE DENOMINATOR every rate metric should use
                    self.eod["effective_seconds"] = max(0.0, _sess_s - _blind_s)
                    # the same thing as a share, for eyeballing a day
                    self.eod["blind_pct_of_session"] = (
                        100.0 * _blind_s / _sess_s if _sess_s > 0 else None)
                    # how many separate silences made it up
                    self.eod["blind_windows"] = len(self.blind_windows)
                    # how many of them had a two-sided book on the far side,
                    # i.e. how many the exposure below could be measured on
                    self.eod["blind_windows_measured"] = sum(
                        1 for w in self.blind_windows
                        if w["spared_adverse_pkr"] is not None)
                    # THE UPPER BOUND ON THE GIFT, in PKR. See the note above.
                    self.eod["blind_spared_adverse_pkr"] = float(
                        self.stats.get("blind_spared_adverse_pkr", 0.0))
                    # P&L PER HOUR ON THE CLOCK THE ENGINE COULD SEE. cash
                    # starts at 0.0, so equity_liquidated IS the day's net.
                    self.eod["pnl_per_effective_hour"] = (
                        self.eod["equity_liquidated"]
                        / (self.eod["effective_seconds"] / 3600.0)
                        if self.eod["effective_seconds"] > 0 else None)
                    # the same number on the uncorrected clock, so the size of
                    # the correction is visible rather than asserted
                    self.eod["pnl_per_session_hour"] = (
                        self.eod["equity_liquidated"] / (_sess_s / 3600.0)
                        if _sess_s > 0 else None)
                    # THE GIFT AS A SHARE OF THE DAY'S P&L -- the one number
                    # that says whether the blind windows matter here. None
                    # when the day made nothing, because a ratio to zero says
                    # nothing and inventing one would be worse than a gap.
                    self.eod["blind_spared_pct_of_pnl"] = (
                        100.0 * self.eod["blind_spared_adverse_pkr"]
                        / abs(self.eod["equity_liquidated"])
                        if self.eod["equity_liquidated"] else None)
                    # ---- EMIT LIQUIDATION FILLS (Stage 1) --------------------
                    # The EOD walk flattened the position; book each consumed
                    # level as a real fill so the FIFO decomposition matches it
                    # against the open lots exactly (no modelled reconstruction,
                    # no plug). The liquidating side is OPPOSITE our position:
                    # long (pos>0) -> we SELL into bids; short -> we BUY asks.
                    if pos_at_liq != 0:
                        # side that flattens the position
                        liq_side = "SELL" if pos_at_liq > 0 else "BUY"
                        # timestamp for every liquidation fill = the crossing event
                        liq_ts = int(ts_exch)
                        # closing mid recorded on each liq fill (mid0 analogue);
                        # None-safe -- falls back to last good mid.
                        liq_mid = mid_e if mid_e is not None else self.last_good_mid
                        # each book level the walk consumed -> one fill at its px
                        for _lpx, _lqty in liq_level_fills:
                            # append with the SAME schema as _fill's rows, tagged
                            # reason="liq" so downstream can identify EOD exits
                            self.fills.append({
                                # crossing-event exchange time
                                "t": liq_ts,
                                # the flattening side
                                "side": liq_side,
                                # the actual level price the walk paid/received
                                "px": float(_lpx),
                                # shares taken at this level
                                "qty": float(_lqty),
                                # provenance: end-of-day book-walk liquidation
                                "reason": "liq",
                                # no trigger window applies to the EOD flatten
                                "window": "eod",
                                # last15 by construction (crossing session end)
                                "bucket": getattr(self.strat, "current_bucket", "last15"),
                                # closing mid, carried for the decomposition's capture
                                "mid0": (float(liq_mid) if liq_mid is not None else np.nan),
                                # liquidation fills have no working-order id
                                "oid": None})
                        # residual the book could NOT absorb: one fill at the
                        # haircut mark price, so equity_liquidated reconciles to
                        # the sum of ALL liq fills exactly. residual_mark is the
                        # signed CASH of the mark; recover a per-share price so
                        # the fill's px*qty*sign matches that cash.
                        if unfilled > 0 and ref is not None:
                            # sign of the flattening trade (opposite of position)
                            _rsgn = -1.0 if pos_at_liq > 0 else 1.0
                            # per-share haircut price implied by residual_mark:
                            # residual_mark = pos_sgn * unfilled * ref*(1-pos_sgn*hc)
                            # so the per-share mark price is ref*(1 - pos_sgn*hc).
                            _pos_sgn = 1.0 if pos_at_liq > 0 else -1.0
                            _hc = self.cfg.get("unfilled_haircut_pct", 0.03)
                            # the haircut per-share price (what we "exit" residual at)
                            _rpx = ref * (1.0 - _pos_sgn * _hc)
                            # append the residual as its own tagged fill
                            self.fills.append({
                                # same EOD timestamp
                                "t": liq_ts,
                                # same flattening side as the rest of the walk
                                "side": liq_side,
                                # the haircut mark price
                                "px": float(_rpx),
                                # the unfilled shares
                                "qty": float(unfilled),
                                # provenance: haircut-marked overnight residual
                                "reason": "liq_residual",
                                # no trigger window
                                "window": "eod",
                                # last15 bucket
                                "bucket": getattr(self.strat, "current_bucket", "last15"),
                                # closing mid reference
                                "mid0": (float(liq_mid) if liq_mid is not None else np.nan),
                                # no order id
                                "oid": None})
        # Return the accounting logs as DataFrames, plus the diagnostic counters.
        # flush the order lifecycle log to a DataFrame for the harness
        # (rows: oid, side, px, qty, t_sent, t_live, t_end, end_reason)
        self.order_log = pd.DataFrame(self._olog.values())
        # unchanged return contract; order_log + _msg_ts are attributes
        return pd.DataFrame(self.fills), pd.DataFrame(self.equity), self.stats


class NaiveSymmetricMM:
    """Baseline strategy: symmetric quotes at mid ± half_spread.

    Exists to exercise the harness and set the bar every smarter strategy
    must beat. It is deliberately naive: no inventory skew, no toxicity
    gating, no spread-regime awareness.

    Logic per evaluation:
      * no two-sided book -> no quotes (pre-open, halts, one-sided book);
      * hard inventory gate: long >= max_inv stops bidding, short <= -max_inv
        stops offering. This is a GATE, not a skew, and it is the baseline's
        main weakness: on the MCB sample day inventory pinned at the cap
        (|pos| reached 548 vs a 500 cap, and sat at >=90% of cap for 34.7%
        of the session) through a trending day. Inventory SKEW — shifting
        both quotes against the position instead of switching a side off —
        is the first improvement to make;
      * price = mid -/+ half_spread, safe-rounded (round_tick: bids floor,
        asks ceil, so rounding never makes a quote more aggressive), then
        post-only clipped one tick inside the opposite touch so the sent
        order is never marketable against the book we can see. It can still
        cross the book as it stands after latency -> handled by the
        crossing-reject check in _arrive;
      * round(px, 2) kills float dust (405.16999... -> 405.17) so the
        no-churn comparison in _requote sees identical prices as identical.

    Reference numbers (MCB 2026-06-30, hs=0.20, size=50, max_inv=500,
    constant 120ms latency, full 17.73 bps/side fee schedule, book-walk EOD
    liquidation): 417 fills, PnL -5,611 PKR. Note the loss is dominated by
    fees at this schedule, not by fill quality — see the fee analysis.

    Strategy interface contract: quotes() returns {side: (price, qty)} and
    may omit a side to mean 'no quote there'. Any replacement strategy needs
    only this method with the same signature.
    """

    def __init__(self, half_spread=0.20, size=50, max_inv=500):
        # hs = half-spread in PKR (distance from mid to each quote).
        # size = shares per quote. max_inv = hard inventory cap in shares.
        self.hs, self.size, self.max_inv = half_spread, size, max_inv

    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # No two-sided book (pre-open, halt, one side empty) -> quote nothing.
        # depth accepted for interface parity with the deep-OFI strategy; the
        # naive reference strategy ignores it.
        if bb is None or ba is None:
            return {}
        # Reference price: the arithmetic midpoint of the touch.
        mid = (bb + ba) / 2
        # Desired quotes, keyed by side. A missing key means 'no quote there'.
        out = {}
        # Bid only while we are not already at the long inventory cap.
        if pos < self.max_inv:
            # Target bid = mid - half_spread, floored onto the tick grid;
            # then post-only clip to one tick BELOW the best ask so it cannot take.
            px = min(round_tick(mid - self.hs, "BUY"), ba - TICK)
            # Store the bid, rounded to 2dp to kill float dust for the no-churn check.
            out["BUY"] = (round(px, 2), self.size)
        # Offer only while we are not already at the short inventory cap.
        if pos > -self.max_inv:
            # Target ask = mid + half_spread, ceiled onto the tick grid;
            # then post-only clip to one tick ABOVE the best bid so it cannot take.
            px = max(round_tick(mid + self.hs, "SELL"), bb + TICK)
            # Store the ask, rounded to 2dp for the same reason.
            out["SELL"] = (round(px, 2), self.size)
        # Return the desired quote set; _requote reconciles it against what's working.
        return out


# ---------------------------------------------------------------------------
# Data loading: three raw tables -> one correctly ordered event stream
# ---------------------------------------------------------------------------
def load_events(u_path, s_path, t_path):
    """Read updates/snapshot/trades CSVs and build the replay inputs.

    Returns:
      events      list of (ts_exch, kind_rank, appl_seq, kind, row),
                  fully sorted — see ordering rationale below
      snap_groups {snap_key: DataFrame of ALL rows of that snapshot message}
                  where snap_key is "msg_seq|orig_time" -- see the note at the
                  grouping itself; msg_seq alone is not unique per message
                  — book levels (BID/OFFER), the AGG_BID/AGG_OFFER totals,
                  and the status/circuit-breaker rows. snapshot() does its
                  own filtering. Pre-split once so the replay loop does
                  O(1) lookups.
      t           the trades DataFrame (with ts_exch/ts_cap/rest_oid added),
                  returned because callers use it to define the session
                  window and for post-run analysis

    Timestamps -> int64 milliseconds since epoch:
      exchange clock: transact_time (updates, trades), orig_time (snapshots;
        second-precision only — tolerable because snapshots are full-state
        replacements, so ±1s placement error self-corrects immediately).
      capture clock: capture_ts everywhere (feeds knowledge time).
      .dt.as_unit("ns") guards a pandas-3.0 trap: these strings parse to
        datetime64[us], where .astype("int64") silently yields MICROseconds;
        forcing ns first makes the //1_000_000 division correct always.

    rest_oid: trades carry the resting order as a stringified tuple
      "('0010THF0D0000R0K', 407.31)"; ast.literal_eval safely extracts the
      id (never eval() on data from disk). ~1/3 of trades resolve; the rest
      use the Book.trade fallback.

    Snapshot EVENTS come from ALL messages, not just book-bearing ones:
      a status-only 35=W (MARKET_CLOSED, TRADING_BREAK, ...) carries no
      BID/OFFER rows but must still reach snapshot() so Book.phase updates
      and the halt gate in _requote can pull quotes. snapshot() returns
      early on those, updating state without wiping the book.

    Event ordering — the load-bearing detail of the whole harness:
      key = (ts_exch, kind_rank, appl_seq)
      * ts_exch    exchange time first: the matching engine's own sequence.
      * kind_rank  snapshots (0) before incrementals (1) at the same
                   timestamp: a snapshot stamped T describes the book AS OF
                   T, so same-time incrementals must build on top of it,
                   not be wiped by it.
      * appl_seq   updates and trades share channel 2011 and their combined
                   appl_seq is strictly increasing (verified on the data) —
                   exact wire order within a same-millisecond burst, no
                   clock needed. Snapshots ride channel 1011 (different
                   sequence space), hence their 0 placeholder.
    """
    # Read the three raw tables. index_col=0 drops the writer's index column.
    u = pd.read_csv(u_path, index_col=0)
    s = pd.read_csv(s_path, index_col=0, dtype={"prev_close": "float64"})
    t = pd.read_csv(t_path, index_col=0)

    # Helper: ISO timestamp column -> int64 milliseconds since epoch.
    def ms(df, c):
        # as_unit("ns") first, so .astype("int64") is nanoseconds on any pandas
        # version; //1e6 then gives exact ms (this feed is ms-precision).
        return (pd.to_datetime(df[c], utc=True, format="ISO8601")
                .dt.as_unit("ns").astype("int64") // 1_000_000)

    # Exchange clock per table: tag 60 for updates/trades, tag 42 for snapshots.
    u["ts_exch"], t["ts_exch"], s["ts_exch"] = \
        ms(u, "transact_time"), ms(t, "transact_time"), ms(s, "orig_time")
    # Capture clock on all three: feeds knowledge time (what we could have known).
    for df, c in ((u, "capture_ts"), (t, "capture_ts"), (s, "capture_ts")):
        df["ts_cap"] = ms(df, c)
    # Extract the resting order id from the stringified tuple; None when absent.
    t["rest_oid"] = t["resting_order_id"].map(
        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith("(") else None)

    # ---- THE SNAPSHOT GROUPING KEY, CORRECTED 2026-09-17 ---------------
    # msg_seq ALONE IS NOT UNIQUE PER SNAPSHOT MESSAGE, and grouping on it
    # merged two different 35=W messages into one book.
    #
    # MEASURED, not inferred. On NRL and MLCF for 2026-06-30:
    #   symbol+msg_seq            90 of 12,476 groups held TWO level-1 bids
    #   symbol+msg_seq+market     90   (the market does not separate them)
    #   symbol+msg_seq+channel    90   (nor does the channel)
    #   symbol+msg_seq+orig_time   0 of 12,566   -- clean
    #
    # Same symbol, same channel, same market, DIFFERENT orig_time: the FIX
    # sequence number repeats within the day. Merging two messages puts two
    # ladders in one book -- two level-1 bids, two level-1 offers -- and the
    # touch then reads as one message's bid against the other's ask. That is
    # the crossed-book finding: 275 of 285 were inverted MORE THAN ONE LEVEL
    # deep, which no event-ordering explanation accounts for.
    #
    # AND THE CROSSINGS ARE ONLY THE VISIBLE PART. A merge whose two ladders
    # happen not to cross produces a book with doubled depth and a wrong
    # shape, and nothing flags it. 0.72% of snapshot messages were affected;
    # every quote priced off one of them was priced off a book that never
    # existed.
    #
    # CAVEAT ON orig_time: it is tag 42 and second-precision, so two genuinely
    # distinct snapshots inside one second would still merge. It separates
    # every case in the sample, but the durable fix is a per-message id from
    # the parser. This is the right key today, not forever.
    s["snap_key"] = (s["msg_seq"].astype("int64").astype(str) + "|"
                     + s["orig_time"].astype(str))
    # One dict entry per snapshot MESSAGE, holding ALL its rows (book + AGG +
    # status). snapshot() filters internally; it needs the AGG and phase rows.
    snap_groups = dict(tuple(s.groupby("snap_key")))
    # One (ts_exch, ts_cap) pair per snapshot message. Grouping on the FULL
    # frame means status-only messages also become events (phase changes).
    snap_ev = s.groupby("snap_key", as_index=False)[["ts_exch", "ts_cap"]].min()

    # Updates: kind "U", rank 1, ordered within a ms by appl_seq.
    events = [(r.ts_exch, 1, r.appl_seq, "U", r) for r in u.itertuples()]
    # Trades: kind "T", rank 1, sharing the same appl_seq sequence as updates.
    events += [(r.ts_exch, 1, r.appl_seq, "T", r) for r in t.itertuples()]
    # Snapshots: kind "S", rank 0 (apply first on ties), appl_seq not comparable.
    events += [(r.ts_exch, 0, 0, "S", r) for r in snap_ev.itertuples()]
    # Sort on the first three fields only; the payload row is carried, not compared.
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return events, snap_groups, t