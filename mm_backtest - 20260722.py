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

FEE_TOTAL_PCT = (FEE_COMMISSION_PCT * (1.0 + FEE_SST_RATE)   # commission + tax ON it
                 + FEE_PSX_LAGA_PCT + FEE_SECP_PCT + FEE_IPF_PCT
                 + FEE_CLEARING_PCT + FEE_MM_REBATE_PCT)

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

        # --- market state (message-level): phase + individual circuit limits ---
        if len(rows_all):
            ph = rows_all["phase"].iloc[0]
            if isinstance(ph, str):
                self.phase = ph
            up = rows_all.loc[rows_all.entry_type == "UPPER_CIRCUIT_BREAKER", "px"]
            dn = rows_all.loc[rows_all.entry_type == "LOWER_CIRCUIT_BREAKER", "px"]
            if len(up):
                self.limit_up = float(up.iloc[0])
            if len(dn):
                self.limit_dn = float(dn.iloc[0])
        if len(rows) == 0:
            return  # status-only message: update state, KEEP the book

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
    oid: int = 0       # unique id: cancels target THIS order, not just the side

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
        self.work: dict[
            str, MyOrder] = {}  # OUR live orders, keyed by side: {"BUY": MyOrder, "SELL": MyOrder}. At most one per side.
        self._oid = 0  # counter that generates a unique id for each order we send (cancels target a specific oid, not just a side).
        self.ack_until = {"BUY": 0,
                          "SELL": 0}  # per side: exchange-ms until which that side's last cancel is UNCONFIRMED (ack not yet back). 0 = nothing pending.
        self.use_ack = 'latency_model' in cfg  # ack-realism gate: only enforce the "don't restack an unconfirmed side" rule in stochastic mode (when a real LatencyModel was given).
        self.pending = []  # min-heap of OUR in-flight messages (orders/cancels traveling to the exchange). Ordered by land-time.
        self._seq = 0  # monotonic counter: breaks ties in the heap FIFO AND stops heapq from ever comparing MyOrder payloads.
        self.pos = 0.0  # our current position in shares (signed: + long, - short). Updated on every fill.
        self.cash = 0.0  # our running cash in PKR (signed). Fills add/subtract price*qty and deduct fees.
        self.fills, self.equity = [], []  # accounting logs: fills = every trade we got; equity = mark-to-market curve (one row per event). Returned as DataFrames by run().
        self.stats = {"rejected_crossing": 0, "n_orders_sent": 0, "n_cancels": 0,
                      # diagnostic counters, all start at 0:
                      "stale_cancels_ignored": 0, "requotes_blocked_by_ack": 0,
                      # rejected_crossing = post-only rejects; n_orders_sent/n_cancels = message counts;
                      "halted_requotes": 0}  # stale_cancels_ignored = cancels for already-gone orders; requotes_blocked_by_ack = requotes skipped by ack guard; halted_requotes = requotes skipped during halts.


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
        while self.pending and self.pending[0][0] < t_exch:  # loop while the heap is non-empty AND its earliest message's land-time (pending[0][0]) is before t_exch. Strict '<' = same-ms messages process AFTER the market event (conservative).
            t, _, action, p = heapq.heappop(self.pending)  # pop the earliest message off the heap. Unpack: t = land-time, _ = the _seq tiebreaker (ignored), action = "ARRIVE"/"CANCEL", p = payload.
            if action == "ARRIVE":  # this message is a NEW order reaching the exchange.
                self._arrive(t,p)  # hand it to _arrive(), which checks if it crosses the book (reject) and otherwise makes it live + snapshots its queue position.
            elif action == "CANCEL":  # this message is a CANCEL request reaching the exchange.
                side, oid = p  # unpack the payload: which side, and the SPECIFIC order id this cancel was meant for.
                o = self.work.get(side)  # look up our current working order on that side (or None if there isn't one).
                if o is not None and o.oid == oid:  # is there an order on that side AND is it the SAME order this cancel targeted (matching oid)?
                    self.work.pop(side, None)  # yes -> remove it from our working orders (the cancel succeeded).
                    self.stats["n_cancels"] += 1  # count a successful cancel.
                else:  # no matching order: it was already filled, or already replaced by a newer order.
                    self.stats["stale_cancels_ignored"] += 1  # count a no-op cancel. This mirrors a real exchange CANCEL-REJECT (nothing there to cancel).

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
        bb, _, ba, _ = self.book.bbo()  # get the CURRENT best bid (bb) and best ask (ba) from the real book. The _ discard the qty fields (bq, aq) -- not needed here. Note: the book may have MOVED during our latency window.
        crosses = ((o.side == "BUY" and ba is not None and o.price >= ba) or  # would our order execute immediately instead of resting? For a BUY: our bid at/above the best ask means we'd cross and take.
                   (o.side == "SELL" and bb is not None and o.price <= bb))  # for a SELL: our ask at/below the best bid means we'd cross and take. (the 'is not None' guards an empty side.)
        if crosses:  # our order would be marketable (take liquidity) rather than post passively.
            self.stats["rejected_crossing"] += 1  # count it as a rejected crossing order.
            return  # FLAGGED SIMPLIFICATION: reject it (post-only behavior) instead of executing as a taker. The order never enters the book. Exit early.
        o.ahead = self.book.qty_at(o.side, o.price)  # order rests: snapshot our QUEUE POSITION -- {order_id: qty} of every order already resting at our price (all ahead of us under price-time priority).
        o.t_active = t  # record the exchange-time the order became live (used for timing/diagnostics).
        self.work[o.side] = o  # store the order as our working order on this side. It's now live and eligible to be filled by incoming flow.

    # ============ fill engine (runs BEFORE the event mutates the book) ====
    # Ordering matters: fills are judged against the book AS IT WAS when
    # the aggressive order hit it. The main loop therefore calls these
    # handlers first, and only then applies the event to self.book.

    def _fill(self, side, price, qty, t_exch, reason): # book a fill of OUR order. side = BUY/SELL; price = the trade's print price (not used for our cash);
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
        o = self.work.get(side)                       # fetch our working order on this side (the one being filled).
        take = min(o.qty, qty)                        # we can only fill up to OUR remaining size; if the incoming qty is larger, we take our whole order, not more. This is the actual fill amount.
        sgn = 1 if side == "BUY" else -1              # sign of the position change: a BUY adds shares (+1), a SELL removes them (-1).
        self.pos += sgn * take                        # update our position: +take if we bought, -take if we sold.
        self.cash += -sgn * take * o.price - fee_for(o.price, take)   # update cash: money moves OPPOSITE to position (buying spends cash, selling earns it), always at OUR limit price o.price -- then subtract the fee. Note: fee uses o.price, the price we transacted at.
        self.fills.append({"t": t_exch, "side": side, "px": o.price, "qty": take,   # log this fill: time, side, OUR price, and the amount filled...
                           "reason": reason})          # ...plus WHY it filled (through/at-queue/crossing-add) so you can audit which fill rule produced which PnL.
        o.qty -= take                                 # reduce our order's remaining quantity by what just filled.
        if o.qty <= 0:                                # if the order is now fully filled...
            self.work.pop(side, None)                 # ...remove it from working orders (it's done). A partial fill leaves it in place with reduced qty.

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
        # HALT / BAND-PIN GATE: quote only in continuous trading, never into a
        # pinned band. On any other phase (TRADING_BREAK, OPEN_CALL_AUCTION,
        # MARKET_CLOSED...) pull all working quotes via the normal latency path.
        # FLAGGED ASSUMPTION: orders persist through halts until our cancel
        # lands; PSX may purge on some halt types -- verify with broker.

        quotable = self.book.phase in (None, "CONTINUOUS_AUCTION") and not self.book.pinned()
        if not quotable:
            self.stats["halted_requotes"] += 1
            for side, cur in list(self.work.items()):
                if cur.cancel_at is None:
                    a_out = self.lat.draw_out()
                    cur.cancel_at = ts_know + a_out
                    self._push(cur.cancel_at, "CANCEL", (side, cur.oid))
                    if self.use_ack:
                        self.ack_until[side] = cur.cancel_at + self.lat.draw_ack()
            return

        bb, bq, ba, aq = self.book.bbo()
        want = self.strat.quotes(bb, bq, ba, aq, self.pos)

        # BAND CLAMP: exchange rejects orders outside [limit_dn, limit_up]
        for side in ("BUY", "SELL"):
            w = want.get(side)
            if w is not None:
                px = w[0]
                if self.book.limit_up is not None:
                    px = min(px, self.book.limit_up)
                if self.book.limit_dn is not None:
                    px = max(px, self.book.limit_dn)
                want[side] = (round(px, 2), w[1])

        for side in ("BUY", "SELL"):
            # ACK GUARD (stochastic mode only): while this side's last cancel is
            # unconfirmed (sent, ack not back), don't stack another cancel/replace
            # on it -- a real risk system treats the old order as possibly-live
            # until acked. The original cancel+replace pair is unaffected.
            if self.use_ack and ts_know < self.ack_until[side]:
                self.stats["requotes_blocked_by_ack"] += 1
                continue
            w = want.get(side)
            cur = self.work.get(side)
            same = cur is not None and w is not None and \
                   cur.price == w[0] and cur.qty == w[1] and cur.cancel_at is None
            if same:
                continue
            if cur is not None and cur.cancel_at is None:
                a_out = self.lat.draw_out()  # independent cancel-send draw
                cur.cancel_at = ts_know + a_out  # exchange stops matching here
                self._push(cur.cancel_at, "CANCEL", (side, cur.oid))
                if self.use_ack:  # confirmed only after ack
                    self.ack_until[side] = cur.cancel_at + self.lat.draw_ack()
            if w is not None:
                self.stats["n_orders_sent"] += 1
                a_out = self.lat.draw_out()  # independent new-order draw
                t_land = ts_know + a_out
                self._oid += 1
                self._push(t_land, "ARRIVE",
                           MyOrder(side, w[0], w[1], {}, t_land, oid=self._oid))

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
    snap_ev = s.groupby("msg_seq", as_index=False)[["ts_exch", "ts_cap"]].min()

    events = [(r.ts_exch, 1, r.appl_seq, "U", r) for r in u.itertuples()]
    events += [(r.ts_exch, 1, r.appl_seq, "T", r) for r in t.itertuples()]
    events += [(r.ts_exch, 0, 0, "S", r) for r in snap_ev.itertuples()]
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return events, snap_groups, t
