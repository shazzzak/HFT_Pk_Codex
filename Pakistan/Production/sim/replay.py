"""The replay harness: mm_backtest's exchange, the production engine's decisions.

WHAT THIS IS FOR. Before the engine trades, it has to be shown to produce the
same orders and the same P&L as the backtest every measured result came from.
That test is only meaningful if the FILL MODEL is identical on both sides --
otherwise a P&L gap could be the strategy adapter, the order manager, or the
simulated exchange, and there is no way to tell which.

So the harness does not simulate an exchange. It subclasses `mm_backtest.
Backtester` and reuses its exchange wholesale -- `Book`, `MyOrder`, the
`ahead`-dict queue, `LatencyModel`, `_arrive`, `_on_market_trade`, `_fill`, the
whole main loop -- and replaces exactly one method: `_requote`.

THAT ONE METHOD IS AN ORDER MANAGER. `_requote` reads the book, asks the
strategy what it wants, diffs that against what is working, suppresses no-change
requotes, cancels the incumbent, sends the replacement, draws a latency for each
message and gates on the trading phase. That is `core/oms.py`'s entire job. So
this harness is a controlled experiment with one variable: the same market, the
same fills, the same latency draws, and a different order manager.

WHAT IT DOES *NOT* TEST. `Backtester.run` calls `strategy.observe()` and syncs
the circuit limits itself, before asking for a quote. Those are the backtest's
own code paths and the harness leaves them alone, which is why it calls the
adapter's `quote()` rather than `on_book()` -- calling `on_book()` would observe
every event twice and double-count every volatility and flow update. The
adapter's observe path is covered by unit tests instead.

IMPORT DIRECTION. This module imports `mm_backtest`. That is correct for a test
harness and would be wrong anywhere else: the live engine must never import the
backtest, which is why `micro_mm` was changed to make that import optional.
"""
# for the id bookkeeping
from typing import Dict, Optional

# the exchange this harness reuses. Imported here and NOWHERE in core/ or
# venues/ -- if this import ever appears in the live path, the split is gone.
from mm_backtest import Backtester, MyOrder

# the production domain objects
from core.model import BookLevel, BookSnapshot, DesiredQuotes, Fill, Side
from core.model import CancelOrder, PlaceOrder, ReplaceOrder
# the phase the adapter gates on
from core.venue import MarketPhase, SecurityPhase


class EngineReplay(Backtester):
    """`Backtester` with `_requote` replaced by the production engine.

    Everything else is inherited unchanged and deliberately so.
    """

    def __init__(self, *, strategy, adapter, oms, symbol: str, **kwargs):
        # the real MicrostructureMM goes to Backtester as its strategy, so that
        # observe(), the limit sync, current_window and want_taker_side all
        # behave exactly as they do in a backtest run
        super().__init__(strategy=strategy, **kwargs)
        # the production adapter -- asked only for quotes, never for state
        self._adapter = adapter
        # the production order manager under test
        self._oms = oms
        # ONE SOURCE OF TRUTH FOR WHAT AN AMENDMENT COSTS. The venue publishes
        # the three priority rules; the engine carries its own three flags with
        # the same meaning. Wiring them here means the simulated exchange can
        # never disagree with the venue the live engine is configured against.
        # A disagreement would make every fill downstream of a reprice wrong in
        # the same direction, and it would never surface as an error -- just as
        # a P&L that quietly does not match.
        _venue = getattr(oms, "_venue", None)
        # a harness constructed without a venue keeps Backtester's own defaults
        if _venue is not None:
            # a reprice: does the amended order keep its place in the queue?
            self.cfo_price_keeps_priority = _venue.replace_price_keeps_priority
            # a size increase: same question
            self.cfo_qty_up_keeps_priority = _venue.replace_qty_up_keeps_priority
            # a size reduction: the one case venues usually allow in place
            self.cfo_qty_down_keeps_priority = _venue.replace_qty_down_keeps_priority
        # the instrument
        self._symbol = symbol
        # Backtester identifies our orders by an integer oid; the production
        # engine identifies them by a client order id string. One map each way,
        # because both directions are needed on different events.
        self._cl_by_oid: Dict[int, str] = {}
        # the client order id of the cancel currently in flight per oid, so the
        # order manager can be told which MESSAGE was answered -- PSX requires a
        # new ClOrdID on a cancel and the exchange's reply quotes that one, not
        # the order's
        self._cancel_cl_by_oid: Dict[int, str] = {}
        # the client order id of the amendment in flight, keyed by the NEW oid
        # the amended generation will carry. Separate from the cancel map
        # because an amendment and a cancel are different messages with
        # different replies, and conflating them is how a reject gets applied
        # to the wrong order.
        self._amend_cl_by_oid: Dict[int, str] = {}
        # oids that left self.work because they were AMENDED, not cancelled.
        # _activate_until reports any disappearance as a cancellation, which is
        # right under cancel-plus-new and wrong under an amendment: the order
        # did not go away, it became a new generation of itself.
        self._amended_away: set = set()
        # counters this harness adds, kept separate from Backtester's stats so
        # nothing it reports is altered
        self.engine_stats = {
            # books the production stack refused to quote against
            "skipped_crossed_book": 0,
            # one-sided books, which are a normal state and not an error
            "skipped_one_sided": 0,
            # requotes where the phase gate pulled everything
            "halted_requotes": 0,
            # actions the risk gateway refused
            "gateway_rejections": 0,
            # amendments emitted by the order manager and scheduled here
            "replace_actions": 0,
            # amendments that landed on an order that had already gone -- the
            # exchange's Order Cancel Reject, and the real cost of the one
            # message: until it lands the OLD terms are still matchable
            "replace_rejected_stale": 0,
        }

    # ---- the book, in the production engine's units -----------------------
    def _snapshot(self, ts_exch: int) -> Optional[BookSnapshot]:
        """Backtester's reconstructed book as a production BookSnapshot.

        Returns None when there is nothing to quote against, which is a normal
        state and not an error.
        """
        # the touch, in float rupees
        bb, bq, ba, aq = self.book.bbo()
        # a one-sided book: normal at the open, after a halt, and all day on a
        # thin name
        if bb is None or ba is None or bq <= 0 or aq <= 0:
            self.engine_stats["skipped_one_sided"] += 1
            return None
        # THE SAME ROUNDING RULE AS THE ADAPTER. int(px * 100) truncates on 6.6%
        # of PSX paisa prices; if the harness converted differently from the
        # adapter, the gate would fail on prices rather than on logic.
        bid_minor = int(round(bb * 100))
        ask_minor = int(round(ba * 100))
        # a crossed book means the reconstruction is momentarily inconsistent;
        # BookSnapshot refuses one, so skip rather than raise mid-replay
        if bid_minor >= ask_minor:
            self.engine_stats["skipped_crossed_book"] += 1
            return None
        # ranked depth, only when the strategy actually uses more than one level
        levels = getattr(self.strat, "ofi_depth_levels", 1)
        # one level each side is enough otherwise
        if levels > 1:
            # Backtester's own ranked depth, best first
            bids_raw, asks_raw = self.book.ranked_depth(levels)
            # converted, keeping the order
            bids = tuple(BookLevel(int(round(p * 100)), int(q))
                         for p, q in bids_raw)
            asks = tuple(BookLevel(int(round(p * 100)), int(q))
                         for p, q in asks_raw)
        else:
            # just the touch
            bids = (BookLevel(bid_minor, int(bq)),)
            asks = (BookLevel(ask_minor, int(aq)),)
        # the snapshot the production strategy reads
        return BookSnapshot(symbol=self._symbol, timestamp_ms=int(ts_exch),
                            bids=bids, asks=asks)

    # ---- the one method that is replaced ---------------------------------
    def _requote(self, ts_know):
        """Backtester's order manager, replaced by the production one.

        The shape is kept deliberately close to the original so the two can be
        read side by side: phase gate, ask for a desire, diff, cancel, send,
        one independent latency draw per message.
        """
        # BACKTESTER'S OWN GATE, TRANSLATED. It quotes only in continuous
        # trading and never while pinned at a circuit limit. Rather than
        # reimplementing that rule in the production path, it is handed to the
        # adapter as a phase -- which makes the adapter return nothing, which
        # makes the order manager emit cancels. Same outcome, through the code
        # that will actually run.
        quotable = (self.book.phase in (None, "CONTINUOUS_AUCTION")
                    and not self.book.pinned())
        # STALE FEED -> STAND DOWN, added 2026-09-17.
        #
        # Backtester grew this guard after this harness was written: when
        # nothing has arrived for longer than stale_feed_seconds it pulls
        # every quote and stays dark until a snapshot restores the book. The
        # DETECTION lives in Backtester.run, which this class inherits, so
        # self._feed_stale is already maintained correctly -- only the guard
        # was missing, because the guard lives in the one method this class
        # replaces. Without it the baseline stands down through roughly seven
        # silences a day and the engine keeps quoting, and the gate reports a
        # P&L difference that is this omission rather than the order manager.
        if getattr(self, "_feed_stale", False):
            # counted on both sides, as Backtester counts it
            self.stats["stale_feed_requotes"] = \
                self.stats.get("stale_feed_requotes", 0) + 1
            self.engine_stats["stale_feed_requotes"] = \
                self.engine_stats.get("stale_feed_requotes", 0) + 1
            # the same path a halt takes
            quotable = False
        # CROSSED OR LOCKED BOOK -> STAND DOWN, added 2026-09-17.
        #
        # Backtester's skip_crossed_book guard, which this harness also
        # predates. It matters that this sets `quotable` rather than merely
        # skipping: the backtest CANCELS on a crossed book, and a harness that
        # returned early would leave the quotes resting.
        _gb, _, _ga, _ = self.book.bbo()
        # both sides present and inverted or equal
        if getattr(self, "skip_crossed_book", True) \
                and _gb is not None and _ga is not None and _gb >= _ga:
            # counted the way Backtester counts it
            self.stats["crossed_book_requotes"] = \
                self.stats.get("crossed_book_requotes", 0) + 1
            # and stand down
            quotable = False
        # count a halted cycle the way Backtester does -- AFTER the two guards
        # above, because Backtester counts a crossed or stale cycle as halted
        # too
        if not quotable:
            self.stats["halted_requotes"] += 1
            self.engine_stats["halted_requotes"] += 1
        # the book the production stack sees
        book = self._snapshot(ts_know)
        # A ONE-SIDED BOOK IS ALSO A STAND-DOWN, and this is measured rather
        # than assumed: micro_mm.quotes opens with
        #     if bb is None or ba is None or bq <= 0 or aq <= 0: return {}
        # -- the identical test _snapshot uses -- and an empty desire diffs to
        # a full cancel in Backtester. So the backtest pulls its quotes on a
        # one-sided book. This harness used to `return` there, leaving them
        # resting. Routing it through the flat desire below reaches the same
        # outcome through the production order manager.
        #
        # It is NOT counted as a halt: Backtester does not count one, because
        # there the strategy declines rather than the gate refusing.
        # tell the adapter what the market is doing, which is what makes it
        # return nothing and the order manager emit cancels
        self._adapter.on_phase(SecurityPhase(
            phase=MarketPhase.CONTINUOUS if (quotable and book is not None)
            else MarketPhase.HALTED))
        # what the strategy wants. No book, or not quotable, means nothing.
        desired = (self._adapter.quote(book, self.pos)
                   if (quotable and book is not None)
                   else DesiredQuotes.flat(self._symbol))
        # hand it to the production order manager
        self._oms.set_desired(desired)
        # THE DIFF, THE RISK GATEWAY AND THE IN-FLIGHT RULE, all inside here.
        # This call is the thing under test.
        actions = self._oms.reconcile(now_ms=int(ts_know), date=self._date())
        # turn each action into a message on Backtester's own scheduler
        for action in actions:
            self._dispatch(action, ts_know)

    def _date(self) -> str:
        """The trading date, which the risk gateway's window check needs."""
        # Backtester does not carry one, so it is supplied at construction
        return getattr(self, "session_date", "")

    def _dispatch(self, action, ts_know) -> None:
        """One production action -> one message on Backtester's scheduler."""
        # AN AMENDMENT. One message, scheduled exactly as Backtester's own CFO
        # path schedules it, so the two runs are comparable message for
        # message. What the amendment does to QUEUE POSITION when it lands is
        # not decided here -- it is decided by the venue's three
        # replace_*_keeps_priority answers, which the engine reads.
        if isinstance(action, ReplaceOrder):
            # count it
            self.engine_stats["replace_actions"] += 1
            # find the working order this amendment targets, by the PRODUCTION
            # identity rather than the exchange id: an order that has already
            # been amended once carries a new exchange id, and the production
            # side still knows it by its original client order id.
            for side_str, cur in [(s_, o_) for s_ in ("BUY", "SELL")
                                  for o_ in list(self._side_orders(s_))]:
                # AN ORDER STILL ON THE WIRE CANNOT BE AMENDED: PSX requires
                # the exchange's OrderID on a CFO and it does not exist until
                # the order is acknowledged. Skipping it here sends this action
                # down the already-tested "nothing to amend" path, which tells
                # the order manager rather than leaving it in PENDING_REPLACE.
                if cur.t_active is None:
                    continue
                # the production id of whatever is resting on this side
                if self._cl_by_oid.get(cur.oid) != action.orig_cl_ord_id:
                    continue
                # NEVER AMEND SOMETHING ALREADY BEING CANCELLED, and never
                # stack a second amendment on one already in flight. Both are
                # the order manager's job to prevent; this is the belt.
                if cur.cancel_at is not None or cur.amend_at is not None:
                    return
                # one send-latency draw, because this is one message
                a_out = self.lat.draw_out()
                # when the amendment reaches the exchange
                t_land = ts_know + a_out
                # the amended generation gets its own engine id
                self._oid += 1
                # an outbound message, for the rate statistics
                self._msg_ts.append(ts_know)
                # counted the way Backtester counts it, so the message totals
                # of the two runs line up
                self.stats["n_orders_sent"] += 1
                # and separately as an amendment, same as Backtester
                self.stats["n_amends_sent"] = (
                    self.stats.get("n_amends_sent", 0) + 1)
                # remember which production message the exchange is answering
                self._amend_cl_by_oid[self._oid] = action.cl_ord_id
                # schedule it, carrying the target generation so an amendment
                # that loses a race to a fill is rejected, not misapplied
                self._push(t_land, "AMEND",
                           (side_str, cur.oid, action.price_minor / 100.0,
                            action.quantity, self._oid))
                # MARK IT IN FLIGHT. This deliberately does NOT gate fills the
                # way cancel_at does: the old terms stay matchable until the
                # amendment lands, which is the real exposure of the one
                # message and the thing it costs you.
                cur.amend_at = t_land
                # done
                return
            # nothing matched: the target filled or was already gone, which the
            # exchange answers with an Order Cancel Reject
            self.engine_stats["replace_rejected_stale"] += 1
            # tell the order manager, so the order does not sit in
            # PENDING_REPLACE for the rest of the session
            self._oms.on_cancel_rejected(action.cl_ord_id,
                                         "nothing to amend")
            return
        # A CANCEL. Find the working order it targets and start its cancel,
        # exactly as _requote does.
        if isinstance(action, CancelOrder):
            # which side, from the order manager's own record
            for side_str, cur in [(s_, o_) for s_ in ("BUY", "SELL")
                                  for o_ in list(self._side_orders(s_))]:
                # MATCH ON THE PRODUCTION IDENTITY, not the exchange id. An
                # order that has been amended carries a NEW engine id while the
                # order manager still knows it by its original client order id,
                # so matching on the exchange id would fail to find it and the
                # cancel would be silently dropped. The exchange id is still
                # checked as a fallback for an order that was never amended.
                # AN ORDER STILL ON THE WIRE CANNOT BE CANCELLED, for the
                # same reason: no OrderID exists yet. Falls through to the
                # "nothing to cancel" path below, which answers the order
                # manager instead of stranding it in PENDING_CANCEL.
                if cur.t_active is None:
                    continue
                if (self._cl_by_oid.get(cur.oid) != action.orig_cl_ord_id
                        and str(cur.oid) != action.exchange_order_id):
                    continue
                # never stack a second cancel on an order already being
                # cancelled -- the order manager's in-flight rule should
                # prevent it, and this is the belt to that braces
                if cur.cancel_at is not None:
                    return
                # independent send-latency draw for the cancel
                a_out = self.lat.draw_out()
                # the exchange stops matching at this instant
                cur.cancel_at = ts_know + a_out
                # schedule it, targeting this specific order
                self._push(cur.cancel_at, "CANCEL", (side_str, cur.oid))
                # an outbound message, for the rate statistics
                self._msg_ts.append(ts_know)
                # remember which MESSAGE the exchange will be answering
                self._cancel_cl_by_oid[cur.oid] = action.cl_ord_id
                # in stochastic mode this side stays unconfirmed until the ack
                if self.use_ack:
                    self.ack_until[side_str] = cur.cancel_at + self.lat.draw_ack()
                # done
                return
            # nothing matched: the order was already filled or already gone,
            # which is what an exchange cancel-reject is
            self.stats["stale_cancels_ignored"] += 1
            # tell the order manager, so the order returns to LIVE rather than
            # sitting in PENDING_CANCEL for the rest of the session
            self._oms.on_cancel_rejected(action.cl_ord_id, "nothing to cancel")
            return
        # A NEW ORDER.
        if isinstance(action, PlaceOrder):
            # DIAGNOSTIC, NO BEHAVIOUR CHANGE. The identical counter Backtester
            # keeps, incremented here so the two runs can be compared like for
            # like. The order manager marks an order PENDING_NEW the moment it
            # is sent and its in-flight rule then holds the side until the
            # exchange answers, so this side of the comparison should stay at
            # or near zero. If it does not, the order manager has the same hole
            # the backtester has and this is where it shows.
            if self._new_in_flight[action.side.value] > 0:
                # a second new order sent while the first is still in the air
                self.stats["orders_sent_while_new_in_flight"] += 1
            # this side now has one more order in the air
            self._new_in_flight[action.side.value] += 1
            # count it the way Backtester does
            self.stats["n_orders_sent"] += 1
            # and separately as a NEW order, same as Backtester
            self.stats["n_new_orders_sent"] = (
                self.stats.get("n_new_orders_sent", 0) + 1)
            # independent send-latency draw, separate from any cancel
            a_out = self.lat.draw_out()
            # when it becomes eligible to rest
            t_land = ts_know + a_out
            # Backtester's own id, so a later cancel can target this order
            self._oid += 1
            # an outbound message
            self._msg_ts.append(ts_know)
            # the side, as Backtester spells it
            side_str = action.side.value
            # the price, back in float rupees
            px = action.price_minor / 100.0
            # the lifecycle record Backtester's reporting reads
            self._olog[self._oid] = {"oid": self._oid, "side": side_str,
                                     "px": px, "qty": action.quantity,
                                     "t_sent": ts_know, "t_live": None,
                                     "t_end": None, "end_reason": None}
            # both directions of the id map
            self._cl_by_oid[self._oid] = action.cl_ord_id
            # TAKER TAG. The adapter refuses a taker request outright, so this
            # is always False -- kept so the constructed MyOrder is the same
            # shape Backtester builds.
            taker = (getattr(self.strat, "want_taker_side", None) == side_str)
            # THE ORDER OBJECT. t_active=None marks it as sent but not yet
            # live: Backtester's fill paths all test for it and skip, and
            # Backtester._arrive sets it to the landing time.
            order = MyOrder(side_str, px, action.quantity, {}, None,
                            oid=self._oid, taker=taker)
            # RESERVE THE SIDE AT SEND TIME, exactly as Backtester._requote now
            # does. This harness shares Backtester's _arrive, and that method
            # requires the side to be holding the very object that is arriving
            # -- the identity check that replaced the silent overwrite. Without
            # this line every arrival here would be discarded as stale.
            #
            # It changes nothing about the engine's own behaviour: the order
            # manager already refuses to send a second order while the first is
            # in flight, which is why engine_dup_sends measured zero.
            #
            # A SECOND ORDER ON A SIDE IS NOW LEGITIMATE. Backtester holds a
            # LIST per side, so this appends rather than overwriting. Until
            # 2026-09-17 it overwrote, and the order already resting -- the one
            # carrying the queue position the second-order policy exists to
            # protect -- vanished with no cancel and no fill. The counter is
            # kept so the old failure stays visible if it ever returns; it
            # should now be zero on every run.
            if self._side_orders(side_str):
                self.engine_stats["placed_onto_occupied_side"] = \
                    self.engine_stats.get("placed_onto_occupied_side", 0) + 1
            self._side_orders(side_str).append(order)
            # schedule the arrival; the empty dict is filled at _arrive
            self._push(t_land, "ARRIVE", order)
            # done
            return
        # anything else is an action type this harness has not been taught
        raise ValueError(f"EngineReplay cannot dispatch {type(action).__name__}")

    # ---- keeping the two order lifecycles in step -------------------------
    def _arrive(self, t, o: MyOrder):
        """The order reached the exchange. Tell the order manager which way."""
        # Backtester decides: rest, reject as crossing, or execute as a taker
        super()._arrive(t, o)
        # the production id for this order
        cl_ord_id = self._cl_by_oid.get(o.oid)
        # an order this harness did not send (there should be none)
        if cl_ord_id is None:
            return
        # it rested if it is now the working order on its side
        if o in self._side_orders(o.side):
            # the exchange's handle, which PSX requires on every later cancel
            self._oms.on_ack(cl_ord_id, str(o.oid))
        else:
            # post-only reject: the book moved during our latency window and
            # the order would have crossed. A real reject, not a lost message.
            self._oms.on_rejected(cl_ord_id, "post-only reject: would cross")

    def _amend(self, t, payload):
        """An amendment reached the exchange. Keep the order manager in step.

        Backtester's _amend replaces self.work[side] with a NEW MyOrder that
        carries a new engine id -- because for priority purposes a re-queued
        amendment is a new order. The production side does no such thing: the
        order manager keeps ONE Order object and applies the new terms to it.
        Reconciling those two views is all this override does.
        """
        # what this amendment was aimed at, and what it will become
        side, oid, new_px, new_qty, new_oid = payload
        # the production id of the amendment MESSAGE, which is what the
        # exchange's reply quotes
        amend_cl = self._amend_cl_by_oid.pop(new_oid, None)
        # the production id of the ORDER, which survives the amendment
        order_cl = self._cl_by_oid.get(oid)
        # let Backtester apply it, or refuse it as stale
        super()._amend(t, payload)
        # the amended generation, if it landed
        after = self._order_by_oid(side, new_oid)
        # THE AMENDMENT WAS REFUSED: the target had already filled or gone, so
        # the engine left the book untouched and counted it.
        if after is None:
            # count it on this harness's own ledger too
            self.engine_stats["replace_rejected_stale"] += 1
            # tell the order manager, so the order comes back out of
            # PENDING_REPLACE instead of being stuck there
            if amend_cl is not None:
                self._oms.on_cancel_rejected(amend_cl,
                                             "amendment target already gone")
            # nothing else to do
            return
        # IT LANDED. The engine gave the amended generation a new id, so carry
        # the production identity onto it -- every later fill and cancel is
        # matched through this map.
        if order_cl is not None:
            self._cl_by_oid[new_oid] = order_cl
        # the OLD generation left self.work by being amended, NOT cancelled.
        # _activate_until reports disappearances as cancellations, so record
        # this one to stop it being reported as something it was not.
        self._amended_away.add(oid)
        # tell the order manager the new terms are live. The amendment's own
        # id is what the exchange quotes, and the order manager aliases it back
        # to the original order.
        if amend_cl is not None:
            self._oms.on_replaced(amend_cl, int(round(new_px * 100)),
                                  int(new_qty))

    def _fill(self, side, price, qty, t_exch, reason, order=None):
        """One of our orders executed. Tell the order manager how much.

        `order` is chosen by Backtester's fill engine, by price then time, and
        passed through -- with several of our orders resting on one side, WHICH
        one filled is the whole question and must not be guessed at here.
        """
        # the order being filled, and its size before
        order = order if order is not None else self._lead(side)
        # nothing working on that side: Backtester answers for itself
        if order is None:
            return super()._fill(side, price, qty, t_exch, reason)
        # remaining size before the fill
        before = order.qty
        # Backtester books it against THAT order
        super()._fill(side, price, qty, t_exch, reason, order=order)
        # how much actually executed. Read from the SAME object, which survives
        # being dropped from its side's list on a full fill.
        take = before - order.qty
        # EVERY PATH OUT OF THIS METHOD RETURNS THE QUANTITY CONSUMED. The fill
        # engine subtracts it from the aggressor's remaining size before
        # offering what is left to the next order in priority, so a None here
        # stops the walk dead -- and with one order per side it never showed,
        # because there was never a next order.
        if take <= 0:
            return 0.0
        # the production id
        cl_ord_id = self._cl_by_oid.get(order.oid)
        # not ours: Backtester has still booked it, so report what it took
        if cl_ord_id is None:
            return take
        # POSITION COMES FROM FILLS, on both sides of the comparison. The fill
        # is booked at OUR limit price, never the print price -- the same rule
        # Backtester uses for cash.
        self._oms.on_fill(Fill(cl_ord_id=cl_ord_id, symbol=self._symbol,
                               side=Side.BUY if side == "BUY" else Side.SELL,
                               price_minor=int(round(order.price * 100)),
                               quantity=int(take),
                               timestamp_ms=int(t_exch)))
        # and tell the fill engine how much of the aggressor this consumed
        return take

    def _activate_until(self, t_exch):
        """Land in-flight messages, then report any cancel that completed.

        Only ARRIVE and CANCEL are processed in here, and an order leaves
        self.work in this method for exactly one reason: its cancel landed.
        Fills happen in _on_market_trade, on a different path. So a
        disappearance here is unambiguously a cancellation.
        """
        # which of our orders existed before, by id and by side
        before = {o.oid: s_ for s_ in ("BUY", "SELL")
                  for o in self._side_orders(s_)}
        # Backtester lands everything due
        super()._activate_until(t_exch)
        # which are still there
        after = {o.oid for o in self._all_orders()}
        # anything that left was cancelled -- unless it was AMENDED, in which
        # case the order did not go away, it became a new generation of itself
        # and _amend has already told the order manager so.
        for oid in before:
            # still working
            if oid in after:
                continue
            # left because an amendment replaced it, not because a cancel
            # landed. Reporting this as a cancellation would move the
            # production order to CANCELLED while it is still resting.
            if oid in self._amended_away:
                # consume the marker; each amendment is reported once
                self._amended_away.discard(oid)
                continue
            # the CANCEL's own client order id, which is what the exchange's
            # reply quotes and what the order manager is waiting to hear about
            cancel_cl = self._cancel_cl_by_oid.pop(oid, None)
            # a cancel we did not send
            if cancel_cl is None:
                continue
            # the order is gone
            self._oms.on_cancelled(cancel_cl)
