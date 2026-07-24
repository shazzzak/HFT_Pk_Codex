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

# ---- FEES (edit as you confirm the real schedule) --------------------------
# Charged PER SIDE per fill. Percentage fees apply to traded VALUE (price*qty);
# flat fees per share. Total per-side cost = sum of all components.
# FLAGGED SIMPLIFICATION: 0.15% is broker commission ONLY. Production must add
# SECP, PSX transaction fee, CDC/NCCPL, CGT, and any MM-program rebate
# (which may be NEGATIVE). Fill these in as confirmed.
FEE_COMMISSION_PCT = 0.0015   # 0.15% broker commission (of traded value)
FEE_SECP_PCT       = 0.0      # TODO confirm
FEE_PSX_PCT        = 0.0      # TODO confirm
FEE_OTHER_PCT      = 0.0      # CDC/NCCPL/etc -- TODO confirm
FEE_PER_SHARE_FLAT = 0.0      # any flat per-share cost -- TODO confirm
FEE_MM_REBATE_PCT  = 0.0      # MM-program rebate (of value); ENTER AS NEGATIVE


FEE_TOTAL_PCT = (FEE_COMMISSION_PCT + FEE_SECP_PCT + FEE_PSX_PCT
                 + FEE_OTHER_PCT + FEE_MM_REBATE_PCT)   # per-side % of value

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
        self.b2_hits = 0       # branch-2 decrements applied (audit counter)
        self.b2_ignored = 0    # cancels with unknown id AND no matching hidden level

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

    # ---- snapshots replace the whole book ----
    def snapshot(self, rows):
        """Apply one exchange snapshot (35=W message): FULL replacement.

            `rows` = ALL rows sharing one msg_seq, including BID/OFFER book
            levels AND the AGG_BID/AGG_OFFER aggregate rows. The method filters
            to BID/OFFER for the visible book (up to ~10 levels per side, each
            row carrying pipe-separated parallel lists of disclosed orders,
            "ID1|ID2" / "500|1200") and separately reads the AGG rows for L11.

            Build the target book from scratch, then `self.o = tgt` in one
            assignment = add-missing / remove-stale / correct-every-qty at once.
            All prior incremental state (and all synthetic lumps) is
            deliberately forgotten, so lumps reset every snapshot.

            Undisclosed depth WITHIN a visible level: a level's total qty can
            exceed the sum of its disclosed orders. The residual is parked in a
            synthetic order keyed "__H_{side}_{price}" so depth totals stay
            correct; the deterministic key means successive snapshots overwrite
            rather than accumulate.

            Deep book BEYOND the visible levels (L11): the feed transmits only
            the top ~10 price levels per side, but AGG_BID/AGG_OFFER give the
            WHOLE-side total. The portion not covered by the visible levels
            (total - sum(visible)) is parked as a "__AGG_{side}" lump one tick
            past the worst visible level. This preserves whole-book depth for
            obi(include_deep=True) without affecting bbo() or touch-level
            quoting. The lump has no per-level detail — the feed doesn't provide
            it — so it is a single aggregate, correct in total but opaque in
            composition, and refreshed each snapshot. Zero/absent residual
            (common pre-open, when the side fits in the window) -> no lump.
            """

        rows_all = rows  # full msg incl AGG_*
        rows = rows[rows.entry_type.isin(["BID", "OFFER"])]  # visible levels

        tgt = {}
        for r in rows.itertuples():
            side = "BUY" if r.entry_type == "BID" else "SELL"
            px = float(r.px); disc = 0.0                   # disclosed qty sum
            if isinstance(r.order_ids, str) and r.order_ids:
                for oid, q in zip(r.order_ids.split("|"), str(r.order_qtys).split("|")):
                    tgt[oid] = Order(side, px, float(q)); disc += float(q)
            if float(r.qty) - disc > 0:                    # hidden residual
                tgt[f"__H_{side}_{px}"] = Order(side, px, float(r.qty) - disc)

        # L11: deep residual beyond the visible 10 levels. AGG_BID/AGG_OFFER
        # carry the WHOLE-side total; the part not covered by L1-10 is parked
        # as a "__AGG_" lump one tick past the worst visible level so bbo()
        # never sees it. Zero/missing residual (pre-open) -> no lump.
        # Resets each snapshot via the full replacement below.
        for side, agg_type in (("BUY", "AGG_BID"), ("SELL", "AGG_OFFER")):
            arow = rows_all[rows_all.entry_type == agg_type]
            if len(arow) == 0:
                continue
            agg_qty = float(arow["qty"].iloc[0])
            visible = sum(o.qty for o in tgt.values() if o.side == side and o.qty > 0)
            residual = agg_qty - visible  # L11 = AGG - sum(L1-10)
            if residual > 0:
                prices = [o.price for o in tgt.values() if o.side == side]
                if prices:
                    edge = (min(prices) - TICK) if side == "BUY" else (max(prices) + TICK)
                    tgt[f"__AGG_{side}"] = Order(side, edge, residual)

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
        if neg < 0:  # shrink our queue by
            for k in list(d):  # the already-traded qty
                if neg >= 0:
                    break
                take = min(d[k], -neg);
                d[k] -= take;
                neg += take
                if d[k] <= 0:
                    del d[k]
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
        bids, asks = {}, {}
        for k, o in self.o.items(): # Walk every entry in the book. k is the order ID (the dict key), o is the Order (side, price, qty).
            if k.startswith("__AGG_") and not include_deep:
                continue  # skip deep residual
            d = bids if o.side == "BUY" else asks
            d[o.price] = d.get(o.price, 0.0) + o.qty
        bids = {p: q for p, q in bids.items() if q > 0}  # clamp netted levels
        asks = {p: q for p, q in asks.items() if q > 0}
        bq = sum(q for _, q in sorted(bids.items(), reverse=True)[:n]) if bids else 0.0
        aq = sum(q for _, q in sorted(asks.items())[:n]) if asks else 0.0
        return (bq - aq) / (bq + aq) if bq + aq > 0 else None

@dataclass
class MyOrder:
    """One of OUR simulated orders, with the state a real order carries.

    ahead     — order_id -> qty for every historical order that was resting
                at our price when we arrived. Price-time priority means all
                of them fill before we do; this dict is maintained event by
                event (market cancels shrink it, trades at our price drain
                it) so our queue position is EXACT, not estimated. That
                precision is a luxury of order-level data.
    t_active  — exchange-ms when the order became live (after send latency).
    cancel_at — exchange-ms when our in-flight cancel LANDS, or None.
                Between deciding to cancel and cancel_at, the order remains
                fully fillable — the in-flight window naive backtests skip.
    """
    side: str          # 'BUY' = our bid, 'SELL' = our ask
    price: float
    qty: float
    ahead: dict
    t_active: int
    cancel_at: int = None

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
                 wire_out_tail_ms=400.0, wire_in_median_ms=40.0,
                 wire_in_tail_ms=10.0, tail_prob=0.02, seed=0):
        self.decision_ms = decision_ms
        self.wire_out_median_ms = wire_out_median_ms
        self.wire_out_tail_ms = wire_out_tail_ms
        self.wire_in_median_ms = wire_in_median_ms
        self.wire_in_tail_ms = wire_in_tail_ms
        self.tail_prob = tail_prob
        self.rng = np.random.default_rng(seed)

    def draw_out(self):
        base = self.decision_ms + self.wire_out_median_ms
        if self.rng.random() < self.tail_prob:
            base += self.rng.exponential(self.wire_out_tail_ms)
        return base

    def draw_ack(self):
        base = self.wire_in_median_ms
        if self.rng.random() < self.tail_prob:
            base += self.rng.exponential(self.wire_in_tail_ms)
        return base


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

    Internal state:
      work     side -> MyOrder: at most ONE working order per side.
      pending  min-heap of our in-flight messages (t, seq, action, payload);
               seq is a monotone counter that (a) breaks time ties FIFO and
               (b) prevents heapq from ever comparing payload objects.
      pos/cash running position (shares, signed) and cash (PKR).
      fills/equity  accounting logs -> DataFrames returned by run().
    """

    def __init__(self, strategy, cfg):
        self.strat = strategy
        self.cfg = cfg
        # latency: use provided LatencyModel, else build a CONSTANT-latency
        # model from cfg['latency_ms'] (back-compat / go-no-go runs)
        self.lat = cfg.get('latency_model')
        if self.lat is None:
            L = cfg.get('latency_ms', 120)
            self.lat = LatencyModel(decision_ms=0.0, wire_out_median_ms=L,
                                    wire_out_tail_ms=0.0, wire_in_median_ms=L,
                                    wire_in_tail_ms=0.0, tail_prob=0.0)
        self.book = Book()
        self.work: dict[str, MyOrder] = {}
        self.pending = []
        self._seq = 0
        self.pos = 0.0
        self.cash = 0.0
        self.fills, self.equity = [], []
        self.stats = {"rejected_crossing": 0, "n_orders_sent": 0, "n_cancels": 0}

    # ================= our-order plumbing (EXCHANGE side) =================
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
        CANCEL -> remove the working order; if a fill already consumed it
                  (self.work.get returns None) the cancel is simply void —
                  which is exactly what an exchange cancel-reject is.
        """
        while self.pending and self.pending[0][0] < t_exch:
            t, _, action, p = heapq.heappop(self.pending)
            if action == "ARRIVE":
                self._arrive(t, p)
            elif action == "CANCEL":
                o = self.work.get(p)
                if o is not None:
                    self.work.pop(p, None)
                    self.stats["n_cancels"] += 1

    def _arrive(self, t, o: MyOrder):
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
        bb, _, ba, _ = self.book.bbo()
        crosses = ((o.side == "BUY" and ba is not None and o.price >= ba) or
                   (o.side == "SELL" and bb is not None and o.price <= bb))
        if crosses:
            self.stats["rejected_crossing"] += 1
            return
        o.ahead = self.book.qty_at(o.side, o.price)
        o.t_active = t
        self.work[o.side] = o

    # ============ fill engine (runs BEFORE the event mutates the book) ====
    # Ordering matters: fills are judged against the book AS IT WAS when
    # the aggressive order hit it. The main loop therefore calls these
    # handlers first, and only then applies the event to self.book.

    def _fill(self, side, price, qty, t_exch, reason):
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
        """
        o = self.work.get(side)
        take = min(o.qty, qty)
        sgn = 1 if side == "BUY" else -1
        self.pos += sgn * take
        self.cash += -sgn * take * o.price - fee_for(o.price, take)
        self.fills.append({"t": t_exch, "side": side, "px": o.price, "qty": take,
                           "reason": reason})
        o.qty -= take
        if o.qty <= 0:
            self.work.pop(side, None)

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
        if r.initiator == "AUCTION":
            return
        aggr = r.aggressor_side
        passive_side = {"BUY": "SELL", "SELL": "BUY"}.get(aggr)
        o = self.work.get(passive_side)
        if o is None or (o.cancel_at is not None and r.ts_exch >= o.cancel_at):
            return
        px = float(r.price)
        through = (px > o.price) if passive_side == "SELL" else (px < o.price)
        if through:
            self._fill(passive_side, px, float(r.qty), r.ts_exch, "through")
        elif px == o.price:
            mode = self.cfg["at_price_mode"]
            if mode == "never":
                return
            if mode == "always":
                self._fill(passive_side, px, float(r.qty), r.ts_exch, "at_optimistic")
                return
            rem = float(r.qty)
            if r.rest_oid and r.rest_oid in o.ahead:       # surgical drain
                take = min(o.ahead[r.rest_oid], rem)
                o.ahead[r.rest_oid] -= take
                if o.ahead[r.rest_oid] <= 0:
                    del o.ahead[r.rest_oid]
                rem -= take
            else:                                          # pool drain
                for k in list(o.ahead):
                    if rem <= 0:
                        break
                    take = min(o.ahead[k], rem)
                    o.ahead[k] -= take; rem -= take
                    if o.ahead[k] <= 0:
                        del o.ahead[k]
            if rem > 0:                                    # flow reached us
                self._fill(passive_side, px, rem, r.ts_exch, "at_queue")

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
        for side, opp in (("SELL", "BUY"), ("BUY", "SELL")):
            o = self.work.get(side)
            if o is None or r.side != opp:
                continue
            if self.cfg["fill_on_crossing_adds"]:
                px = float(r.price)
                if (side == "SELL" and px >= o.price) or (side == "BUY" and px <= o.price):
                    self._fill(side, o.price, float(r.qty), r.ts_exch, "crossing_add")

    def _on_market_cancel(self, r):
        """A historical order was cancelled. If it was queued AHEAD of one
        of our orders, our queue position just improved: remove it from
        every ahead dict it appears in. (It can only match at one price,
        so at most one dict actually contains it.)"""
        oid = str(r.order_id)
        for o in self.work.values():
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
        for o in self.work.values():
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
        bb, bq, ba, aq = self.book.bbo()
        want = self.strat.quotes(bb, bq, ba, aq, self.pos)
        for side in ("BUY", "SELL"):
            w = want.get(side)
            cur = self.work.get(side)
            same = cur is not None and w is not None and \
                   cur.price == w[0] and cur.qty == w[1] and cur.cancel_at is None
            if same:
                continue
            if cur is not None and cur.cancel_at is None:
                a_out = self.lat.draw_out()  # independent cancel-send draw
                cur.cancel_at = ts_know + a_out  # exchange stops matching here
                self._push(cur.cancel_at, "CANCEL", side)
            if w is not None:
                self.stats["n_orders_sent"] += 1
                a_out = self.lat.draw_out()  # independent new-order draw
                t_land = ts_know + a_out
                self._push(t_land, "ARRIVE",
                           MyOrder(side, w[0], w[1], {}, t_land))

    # ============================ main loop ================================
    def run(self, events, snap_groups):
        """Single pass over the merged event stream. Per event, IN ORDER:

        1. _activate_until: land our in-flight messages due before this
           event (exchange timeline catches up).
        2. Fill checks against the PRE-event book: trades test our resting
           orders; adds test for crossings; cancels update our queues.
           (Must precede step 3 — a fill is judged against the book the
           aggressor actually hit.)
        3. Apply the event to the historical book (snapshot = full replace,
           then rebuild our queue dicts; update = add/cancel; trade =
           consume liquidity).
        4. Mark to market: equity = cash + pos * mid, one row per event
           with a valid two-sided book. This is the equity curve.
        5. Advance knowledge time (running max of ts_cap — receive
           timestamps jitter, knowledge never runs backwards) and, inside
           the session window, let the strategy requote. After session end,
           pull all working orders (instant, latency-free — acceptable at
           the close boundary; the closing auction is not modelled).

        Returns (fills DataFrame, equity DataFrame, stats dict).
        """
        t0, t1 = self.cfg["session"]
        know = 0
        for ts_exch, _, _, kind, obj in events:
            self._activate_until(ts_exch)                       # (1)
            if kind == "T":                                     # (2)
                self._on_market_trade(obj)
            elif kind == "U":
                if obj.event == "ORDER_ADD":
                    self._on_market_add(obj)
                else:
                    self._on_market_cancel(obj)
            if kind == "S":                                     # (3)
                self.book.snapshot(snap_groups[obj.msg_seq])
                self._on_snapshot_queue_reset()
            elif kind == "U":
                (self.book.add if obj.event == "ORDER_ADD" else self.book.cancel)(obj)
            else:
                self.book.trade(obj)
            bb, _, ba, _ = self.book.bbo()                      # (4)
            if bb is not None and ba is not None:
                mid = (bb + ba) / 2
                self.equity.append({"t": ts_exch, "mid": mid,
                                    "equity": self.cash + self.pos * mid,
                                    "pos": self.pos,
                                    "obi_5": self.book.obi(5),
                                    "obi_deep": self.book.obi(None)})
            know = max(know, int(obj.ts_cap))                   # (5)
            if t0 <= ts_exch <= t1:
                self._requote(know)
            elif ts_exch > t1:
                for side in list(self.work):
                    self.work.pop(side)
        return pd.DataFrame(self.fills), pd.DataFrame(self.equity), self.stats


class NaiveSymmetricMM:
    """Baseline strategy: symmetric quotes at mid ± half_spread.

    Exists to exercise the harness and set the bar every smarter strategy
    must beat. Logic per evaluation:
      * no two-sided book -> no quotes (pre-open, halts);
      * hard inventory gate: long >= max_inv stops bidding, short <= -max_inv
        stops offering (a gate, not a skew — the measured -437 PKR baseline
        loss is largely this gate pinning at the cap through a trending
        day; inventory SKEW is the first improvement to make);
      * price = mid -/+ half_spread, safe-rounded (round_tick), then
        post-only clipped one tick inside the opposite touch so the sent
        order is never marketable against the book we can see (it can
        still cross the book as it stands after latency -> _arrive check);
      * round(px, 2) kills float dust (405.16999... -> 405.17) so the
        no-churn comparison in _requote sees identical prices as identical.
    """

    def __init__(self, half_spread=0.20, size=50, max_inv=500):
        self.hs, self.size, self.max_inv = half_spread, size, max_inv

    def quotes(self, bb, bq, ba, aq, pos):
        if bb is None or ba is None:
            return {}
        mid = (bb + ba) / 2
        out = {}
        if pos < self.max_inv:
            px = min(round_tick(mid - self.hs, "BUY"), ba - TICK)
            out["BUY"] = (round(px, 2), self.size)
        if pos > -self.max_inv:
            px = max(round_tick(mid + self.hs, "SELL"), bb + TICK)
            out["SELL"] = (round(px, 2), self.size)
        return out


# ---------------------------------------------------------------------------
# Data loading: three raw tables -> one correctly ordered event stream
# ---------------------------------------------------------------------------
def load_events(u_path, s_path, t_path):
    """Read updates/snapshot/trades CSVs and build the replay inputs.

    Returns:
      events      list of (ts_exch, kind_rank, appl_seq, kind, row),
                  fully sorted — see ordering rationale below
      snap_groups {msg_seq: DataFrame of that snapshot's BID/OFFER rows},
                  pre-split once so the replay loop does O(1) lookups
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
    u = pd.read_csv(u_path, index_col=0)
    s = pd.read_csv(s_path, index_col=0)
    t = pd.read_csv(t_path, index_col=0)

    def ms(df, c):
        return (pd.to_datetime(df[c], utc=True, format="ISO8601")
                .dt.as_unit("ns").astype("int64") // 1_000_000)

    u["ts_exch"], t["ts_exch"], s["ts_exch"] = \
        ms(u, "transact_time"), ms(t, "transact_time"), ms(s, "orig_time")
    for df, c in ((u, "capture_ts"), (t, "capture_ts"), (s, "capture_ts")):
        df["ts_cap"] = ms(df, c)
    t["rest_oid"] = t["resting_order_id"].map(
        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith("(") else None)

    sb = s[s.entry_type.isin(["BID", "OFFER"])]  # book levels (for timing)
    snap_groups = dict(tuple(s.groupby("msg_seq")))  # FULL msg incl AGG_* rows
    #   snapshot() itself filters to BID/OFFER and reads AGG_BID/AGG_OFFER
    snap_ev = sb.groupby("msg_seq", as_index=False)[["ts_exch", "ts_cap"]].min()

    events = [(r.ts_exch, 1, r.appl_seq, "U", r) for r in u.itertuples()]
    events += [(r.ts_exch, 1, r.appl_seq, "T", r) for r in t.itertuples()]
    events += [(r.ts_exch, 0, 0, "S", r) for r in snap_ev.itertuples()]
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return events, snap_groups, t
