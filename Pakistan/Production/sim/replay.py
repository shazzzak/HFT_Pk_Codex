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
            # amendments emitted. MUST BE ZERO for a reconcile run: Backtester
            # has no amendment path, so a ReplaceOrder cannot be honoured here.
            "replace_actions": 0,
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
        # tell the adapter what the market is doing
        self._adapter.on_phase(SecurityPhase(
            phase=MarketPhase.CONTINUOUS if quotable else MarketPhase.HALTED))
        # count a halted cycle the way Backtester does
        if not quotable:
            self.stats["halted_requotes"] += 1
            self.engine_stats["halted_requotes"] += 1
        # the book the production stack sees
        book = self._snapshot(ts_know)
        # nothing to quote against -- but a halt must still pull our quotes, so
        # only skip entirely when the book is unusable AND we are quotable
        if book is None and quotable:
            return
        # what the strategy wants. On a halt there is no book, so the desire is
        # explicitly nothing; otherwise ask.
        desired = (self._adapter.quote(book, self.pos) if book is not None
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
        # AN AMENDMENT CANNOT BE HONOURED HERE. Backtester has no amendment
        # path: an order is cancelled and a new one is sent. The order manager
        # defaults to that behaviour for exactly this reason, so a ReplaceOrder
        # arriving means use_replace was turned on -- and the run would silently
        # stop being comparable.
        if isinstance(action, ReplaceOrder):
            self.engine_stats["replace_actions"] += 1
            raise ValueError(
                "EngineReplay received a ReplaceOrder. mm_backtest has no "
                "amendment path, so a run using amendments is not comparable "
                "with any measured result. Construct the OrderManager with "
                "use_replace=False (the default).")
        # A CANCEL. Find the working order it targets and start its cancel,
        # exactly as _requote does.
        if isinstance(action, CancelOrder):
            # which side, from the order manager's own record
            for side_str, cur in list(self.work.items()):
                # match on the exchange id, which is the oid as a string
                if str(cur.oid) != action.exchange_order_id:
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
            # count it the way Backtester does
            self.stats["n_orders_sent"] += 1
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
            # schedule the arrival; the empty dict is filled at _arrive
            self._push(t_land, "ARRIVE",
                       MyOrder(side_str, px, action.quantity, {}, t_land,
                               oid=self._oid, taker=taker))
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
        if self.work.get(o.side) is o:
            # the exchange's handle, which PSX requires on every later cancel
            self._oms.on_ack(cl_ord_id, str(o.oid))
        else:
            # post-only reject: the book moved during our latency window and
            # the order would have crossed. A real reject, not a lost message.
            self._oms.on_rejected(cl_ord_id, "post-only reject: would cross")

    def _fill(self, side, price, qty, t_exch, reason):
        """One of our orders executed. Tell the order manager how much."""
        # the order being filled, and its size before
        order = self.work.get(side)
        # nothing working on that side: Backtester would fail here anyway
        if order is None:
            return super()._fill(side, price, qty, t_exch, reason)
        # remaining size before the fill
        before = order.qty
        # Backtester books it
        super()._fill(side, price, qty, t_exch, reason)
        # how much actually executed. Read from the SAME object, which survives
        # being popped from self.work on a full fill.
        take = before - order.qty
        # a zero take would be a Backtester bug, not something to forward
        if take <= 0:
            return
        # the production id
        cl_ord_id = self._cl_by_oid.get(order.oid)
        # not ours
        if cl_ord_id is None:
            return
        # POSITION COMES FROM FILLS, on both sides of the comparison. The fill
        # is booked at OUR limit price, never the print price -- the same rule
        # Backtester uses for cash.
        self._oms.on_fill(Fill(cl_ord_id=cl_ord_id, symbol=self._symbol,
                               side=Side.BUY if side == "BUY" else Side.SELL,
                               price_minor=int(round(order.price * 100)),
                               quantity=int(take),
                               timestamp_ms=int(t_exch)))

    def _activate_until(self, t_exch):
        """Land in-flight messages, then report any cancel that completed.

        Only ARRIVE and CANCEL are processed in here, and an order leaves
        self.work in this method for exactly one reason: its cancel landed.
        Fills happen in _on_market_trade, on a different path. So a
        disappearance here is unambiguously a cancellation.
        """
        # which order was on each side before
        before = {side: order.oid for side, order in self.work.items()}
        # Backtester lands everything due
        super()._activate_until(t_exch)
        # which are still there
        after = {order.oid for order in self.work.values()}
        # anything that left was cancelled
        for oid in before.values():
            # still working
            if oid in after:
                continue
            # the CANCEL's own client order id, which is what the exchange's
            # reply quotes and what the order manager is waiting to hear about
            cancel_cl = self._cancel_cl_by_oid.pop(oid, None)
            # a cancel we did not send
            if cancel_cl is None:
                continue
            # the order is gone
            self._oms.on_cancelled(cancel_cl)
