"""The order manager: the only thing that decides what messages to send.

The strategy says what it WANTS resting. This decides what has to happen to
make that true, given what is actually resting right now. Those are two
different jobs and keeping them apart is what makes the rest of the system
tractable:

  * QUOTE CHURN IS CONTROLLED IN ONE PLACE -- the diff below. Hysteresis lives
    here, not in the strategy, so changing how often we reprice never touches
    strategy code.
  * THE KILL SWITCH HAS ONE THING TO DO. It sets the desired state to nothing,
    and the ordinary diff emits the cancels. A kill switch with its own
    dedicated cancel-everything path is code that is never exercised until the
    day it matters, and on that day it is the least-tested code in the system.
  * THE BACKTEST AND THE LIVE ENGINE BECOME COMPARABLE. Both are driven by a
    desired quote state, so feeding a recorded day through here and getting the
    same fills is a real check rather than a hope.

THE RULE THAT MATTERS MOST HERE. An order that has been SENT but not yet
acknowledged is neither live nor dead, and this file never acts on one. Not to
cancel it, not to replace it, not to send another. An order manager that
forgets this double-sends under latency, and double-sending is the ordinary way
an order management system loses money -- twice the intended size resting, from
a bug that only appears when the exchange is slow, which is exactly when the
market is moving.
"""
# value objects and typing
from dataclasses import dataclass, field
# typing only
from typing import Callable, Dict, Iterable, List, Optional, Tuple
# the shared domain
from core.model import (Action, CancelOrder, DesiredQuotes, Fill,
                        InvalidTransition, Order, OrderRequest, OrderState,
                        PlaceOrder, QuoteIntent, ReplaceOrder, Side)
# the risk layer this must not bypass
from core.risk import KillSwitch, RiskContext, RiskGateway
# venue rules, for the tick grid
from core.venue import Venue


@dataclass(frozen=True)
class QuoteTolerance:
    """How different the market's quote must be before we bother repricing.

    THIS IS THE CHURN LEVER, and it is the only one. Every cancel-and-repost
    surrenders queue priority: we leave the queue at the old price and rejoin at
    the back of the new one. For a strategy whose edge is earned by resting
    until someone trades with us, that is a real cost, and it is a cost a
    backtest without a queue model cannot see.

    Defaults reproduce the shipped behaviour exactly -- price_ticks = 0 means
    any price change at all triggers a requote, which is what micro_mm does
    today with tol_ticks = 0.0. Raise it deliberately, and measure what it
    costs, rather than inheriting it by accident.
    """
    # requote only when the desired price has moved MORE than this many ticks.
    # 0 = requote on any change, which is today's behaviour.
    price_ticks: int = 0
    # requote when the remaining size has fallen below this fraction of what we
    # want resting. 0 = never requote on size alone, so a partial fill does not
    # cost us our queue position for the remainder.
    qty_ratio: float = 0.0


class OrderManager:
    """Owns every order we have sent, and decides what to send next."""

    def __init__(self, venue: Venue, gateway: RiskGateway,
                 kill_switch: KillSwitch, session_id: str,
                 account: Optional[str] = None,
                 tolerance: Optional[QuoteTolerance] = None,
                 on_event: Optional[Callable[[str, dict], None]] = None,
                 use_replace: bool = False):
        # the venue's rules -- the tick grid, in particular
        self._venue = venue
        # ---- HOW A PRICE CHANGE IS EXPRESSED ------------------------------
        # False (the default) -- cancel the incumbent and place a new order, as
        # two independent messages. True -- one Order Cancel/Replace Request.
        #
        # THE DEFAULT IS FALSE AND THAT IS DELIBERATE, EVEN THOUGH PSX SUPPORTS
        # AMENDMENT AND AMENDMENT IS THE BETTER MECHANIC. mm_backtest._requote
        # sends a cancel and a replacement as two messages with two independent
        # latency draws, and the old order stays fillable until its cancel
        # actually lands. Every measured result -- the three buckets, the fee
        # floor, the 113-name assignment -- was produced under that behaviour,
        # including the fills taken in that window.
        #
        # An engine that amends is therefore NOT the thing that was backtested.
        # It is very likely better, but "better" is a claim, and shipping it by
        # default would make the reconcile gate compare two mechanics and be
        # unable to say anything exact. So: match what was measured, and turn
        # this on as a deliberate experiment with a before and an after.
        self._use_replace = use_replace
        # the risk gateway. EVERY action goes through it; there is no other path.
        self._gateway = gateway
        # the shared kill-switch state
        self._kill = kill_switch
        # a prefix that makes our order ids unique to this session. VALIDATED
        # NOW against the venue's prohibited-character list, because an id that
        # the exchange refuses would otherwise be discovered as a reject on
        # every single order, mid-session.
        self._session_id = venue.validate_text(session_id, "session_id")
        # the client code every order carries, where the venue requires one
        self._account = account
        # FAIL AT CONSTRUCTION, NOT AT GO-LIVE. PSX makes Account (tag 1) a
        # required field; without it the exchange rejects every order we send.
        # That is worth discovering here rather than on the first live morning.
        if venue.requires_account:
            if not account:
                raise ValueError(
                    f"{venue.name} requires an account (client code) on every "
                    f"order; OrderManager was constructed without one")
            # and it must survive the venue's own character rules
            venue.validate_text(account, "account")
        # the churn lever
        self._tol = tolerance or QuoteTolerance()
        # where every state change is recorded, for the audit trail
        self._on_event = on_event
        # what the strategy currently wants, per symbol
        self._desired: Dict[str, DesiredQuotes] = {}
        # the order resting on each (symbol, side), when there is one
        self._working: Dict[Tuple[str, Side], Order] = {}
        # every order we have ever sent this session, by its OWN id
        self._orders: Dict[str, Order] = {}
        # message ids that refer to an existing order -- the ClOrdID of a cancel
        # or of an amendment. PSX requires a NEW ClOrdID on each of those, and
        # the exchange's reply quotes it, so we must be able to resolve it back.
        # Kept SEPARATE from _orders: registering one order object under two
        # keys there made working_orders() count it twice, which in turn made
        # is_flat() wrong -- and is_flat() is what an operator reads after
        # tripping the kill switch.
        self._alias: Dict[str, str] = {}
        # net position per symbol, built ONLY from real fills
        self._position: Dict[str, int] = {}
        # monotonic counter behind the order ids
        self._seq = 0
        # register with the kill switch so a trip flattens us at once
        self._kill.add_listener(self._on_kill)

    # ---- identifiers ------------------------------------------------------
    def _next_id(self) -> str:
        """A client order id that is unique for the life of this session.

        Never reused, including for an order that was rejected: an id that comes
        back a second time makes the exchange's record and ours ambiguous, and
        ambiguity in an order id is unrecoverable after the fact.
        """
        # advance the counter
        self._seq += 1
        # session prefix plus the counter. The separator is deliberate: '-' is
        # NOT in the PSX prohibited set (Appendix C), while several obvious
        # alternatives -- '#', '%', '*', '|', '~' -- are.
        return self._venue.validate_text(
            f"{self._session_id}-{self._seq:08d}", "cl_ord_id")

    # ---- what the strategy wants ------------------------------------------
    def set_desired(self, desired: DesiredQuotes) -> None:
        """Record what the strategy wants resting for one symbol.

        Recording is not sending. Nothing leaves until reconcile() runs, which
        is what lets several symbols be updated and then reconciled once.
        """
        # store it against the symbol
        self._desired[desired.symbol] = desired

    def flatten_all(self, reason: str) -> None:
        """Want nothing resting, anywhere. The cancels follow from the diff."""
        # replace every symbol's desire with the empty one
        for sym in list(self._desired):
            self._desired[sym] = DesiredQuotes.flat(sym)
        # record why, because "everything cancelled at 11:04" needs a cause
        self._emit("flatten_all", {"reason": reason})

    def _on_kill(self, switch: KillSwitch) -> None:
        """Called the moment the kill switch trips."""
        # want nothing; reconcile() will emit the cancels on the next cycle
        self.flatten_all(f"kill switch: {switch.reason}")

    # ---- the diff ---------------------------------------------------------
    def _matches(self, order: Order, want: QuoteIntent) -> bool:
        """Is this resting order close enough to what we want to leave alone?"""
        # the tick size around this price; a tiered venue varies it by price
        tick = self._venue.tick_minor(want.price_minor)
        # how far the resting price is from the desired one, in ticks
        drift = abs(order.price_minor - want.price_minor) / tick
        # a price move beyond the tolerance means requote
        if drift > self._tol.price_ticks:
            return False
        # a size check only when one was asked for; 0 disables it entirely, so
        # a partial fill does not cost us the queue position for the remainder
        if self._tol.qty_ratio > 0.0:
            # what is still working against what we want working
            if order.leaves_quantity < self._tol.qty_ratio * want.quantity:
                return False
        # close enough: leave it where it is and keep its place in the queue
        return True

    def _plan_side(self, symbol: str, side: Side,
                   want: Optional[QuoteIntent]) -> Optional[Action]:
        """What, if anything, needs to happen on one side of one symbol."""
        # the order currently resting on this side, if any
        order = self._working.get((symbol, side))
        # a terminal order is not resting; forget it and treat the side as empty
        if order is not None and order.state.is_terminal:
            self._working.pop((symbol, side), None)
            order = None
        # NOTHING RESTING
        if order is None:
            # and nothing wanted: no action
            if want is None:
                return None
            # wanted but absent: send it
            return PlaceOrder(symbol=symbol, cl_ord_id=self._next_id(),
                              side=side, price_minor=want.price_minor,
                              quantity=want.quantity, account=self._account)
        # IN FLIGHT. The exchange has not answered yet, so we do not know what
        # it thinks exists. Acting now is how an order manager ends up with two
        # orders resting where it intended one.
        #
        # AND ON PSX IT IS NOT MERELY UNWISE, IT IS IMPOSSIBLE. Both the Order
        # Cancel Request (MsgType 'F') and the Order Cancel/Replace Request
        # ('G') carry OrderID (tag 37) as a REQUIRED field, and OrderID is
        # assigned by the exchange on the acknowledgement. Before the ack there
        # is literally nothing to put in the message. Wait.
        if order.state.is_in_flight:
            return None
        # SUSPENDED. The exchange is holding it inactive; it cannot trade and it
        # cannot be amended into a live quote. Leave it and let an operator
        # resolve it, rather than guessing at a resume.
        if order.state is OrderState.SUSPENDED:
            return None
        # RESTING, AND NOT WANTED
        if want is None:
            return self._cancel(order)
        # RESTING, AND CLOSE ENOUGH
        if self._matches(order, want):
            return None
        # RESTING, AND WRONG.
        #
        # RESOLVED 2026-09-16 BY THE PSX FIX SPECIFICATION v1.2. The earlier
        # version of this file cancelled and then placed on a later cycle,
        # because it could not be confirmed that the venue accepted an
        # amendment. It does: Order Cancel/Replace Request (MsgType 'G')
        # "will be used to change any valid attribute of an open order (i.e.
        # reduce/increase quantity, change limit price...)".
        #
        # One message therefore replaces two, and the window in which we had
        # cancelled and not yet replaced -- a window in which we were simply not
        # quoting -- disappears.
        #
        # WHAT IT DOES NOT BUY: the specification does not say whether an
        # amendment keeps queue position. On most venues a price change or a
        # size increase goes to the back of the queue. Treat the latency saving
        # as real and any queue saving as unproven until UAT measures it.
        # OFF BY DEFAULT -- see use_replace in the constructor. Amendment is
        # the better wire mechanic and PSX accepts it, but it is not what the
        # measured results were produced under.
        if (self._use_replace and self._venue.supports_replace
                and order.exchange_order_id):
            return ReplaceOrder(symbol=symbol, cl_ord_id=self._next_id(),
                                side=side, price_minor=want.price_minor,
                                quantity=want.quantity, account=self._account,
                                orig_cl_ord_id=order.cl_ord_id,
                                exchange_order_id=order.exchange_order_id)
        # CANCEL NOW, PLACE ON A LATER CYCLE. The replacement is not sent here
        # and must not be: PSX assigns OrderID on the acknowledgement, so until
        # this cancel is acknowledged there is one order on this side and the
        # in-flight rule above forbids acting on it. mm_backtest sends both at
        # once because it is not bound by that -- it knows its own order ids --
        # and the difference is one cycle of latency, not a different mechanic.
        # A venue with no amendment at all takes this same path.
        return self._cancel(order)

    def _cancel(self, order: Order) -> CancelOrder:
        """A cancel carrying every identifier the venue requires."""
        # PSX requires ClOrdID (a NEW one for the cancel itself), OrigClOrdID
        # (the order being cancelled) and OrderID (the exchange's own handle).
        return CancelOrder(symbol=order.symbol, cl_ord_id=self._next_id(),
                           orig_cl_ord_id=order.cl_ord_id,
                           exchange_order_id=order.exchange_order_id or "")

    def reconcile(self, now_ms: int, date: str,
                  reference_prices: Optional[Dict[str, int]] = None
                  ) -> List[Action]:
        """Work out what to send, clear it with risk, and hand it back.

        Returns only APPROVED actions. The caller encodes and sends them; this
        has already marked the affected orders as in flight, so calling it twice
        without sending in between will not produce duplicates.

        The run loop must call this every cycle, including immediately after a
        kill-switch trip -- the trip sets the desired state, and this is what
        turns that into cancels.
        """
        # the prices the risk gateway checks new orders against
        refs = reference_prices or {}
        # approved actions, cancels first (see the ordering note below)
        cancels: List[Action] = []
        places: List[Action] = []
        # every symbol we have a desire for
        for symbol, desired in self._desired.items():
            # THE KILL SWITCH WINS HERE, at the point of emission, not only
            # where it is handled. If anything set a desire after the trip --
            # a strategy that has not noticed, a race, a bug -- it is overridden
            # now. Defence in depth on the one control that has to work.
            effective = DesiredQuotes.flat(symbol) if self._kill.tripped \
                else desired
            # both sides of the book
            for side in (Side.BUY, Side.SELL):
                # what this side needs, if anything
                action = self._plan_side(symbol, side, effective.side(side))
                # nothing to do on this side
                if action is None:
                    continue
                # the state the risk controls judge against
                ctx = RiskContext(date=date, timestamp_ms=now_ms,
                                  position=self._position.get(symbol, 0),
                                  reference_price_minor=refs.get(symbol))
                # EVERY action clears the gateway. There is no other path out.
                decision = self._gateway.authorise(action, ctx)
                # refused: record it and leave the side as it is. A refused
                # place simply means no quote on that side this cycle, which is
                # the control doing its job, not an error to work around.
                if not decision.allowed:
                    self._emit("risk_rejected",
                               {"symbol": symbol, "side": side.value,
                                "check": decision.check,
                                "reason": decision.reason})
                    continue
                # approved: move our own state to match what is now in flight
                self._mark_sent(action)
                # cancels are collected separately so they can go out first
                (cancels if action.is_cancel else places).append(action)
        # CANCELS BEFORE PLACES, always. Within one cycle this keeps total
        # resting size at or below the intended amount at every instant; the
        # reverse order would briefly double it.
        return cancels + places

    def _mark_sent(self, action: Action) -> None:
        """Move our own record to match what is now in flight."""
        # a new order: create it and mark it pending
        if isinstance(action, PlaceOrder):
            # the order object that will carry its whole lifecycle
            order = Order(cl_ord_id=action.cl_ord_id, symbol=action.symbol,
                          side=action.side, price_minor=action.price_minor,
                          quantity=action.quantity,
                          state=OrderState.PENDING_NEW)
            # remember it by id and as the resting order for that side
            self._orders[order.cl_ord_id] = order
            self._working[(order.symbol, order.side)] = order
            # record it
            self._emit("order_sent",
                       {"cl_ord_id": order.cl_ord_id, "symbol": order.symbol,
                        "side": order.side.value, "px": order.price_minor,
                        "qty": order.quantity})
            return
        # an amendment: the ORIGINAL order goes to pending-replace. The new
        # terms are not applied until the exchange confirms them, because until
        # then the OLD terms are what is resting and what can be filled.
        if isinstance(action, ReplaceOrder):
            # the order being amended
            order = self._orders.get(action.orig_cl_ord_id)
            # an amendment for an order we do not know about is a bug
            if order is None:
                self._emit("replace_unknown_order",
                           {"cl_ord_id": action.orig_cl_ord_id})
                return
            # the old terms remain live until the exchange answers
            order.on_replace_sent()
            # the amendment's own id resolves to the same order, so the
            # exchange's reply -- which quotes the NEW ClOrdID -- can be matched
            self._alias[action.cl_ord_id] = order.cl_ord_id
            # record it
            self._emit("replace_sent",
                       {"cl_ord_id": action.cl_ord_id,
                        "orig_cl_ord_id": action.orig_cl_ord_id,
                        "symbol": action.symbol, "px": action.price_minor,
                        "qty": action.quantity})
            return
        # a cancel: move the existing order to pending-cancel
        if isinstance(action, CancelOrder):
            # the order being cancelled, found by the id it refers to
            order = self._orders.get(action.orig_cl_ord_id or action.cl_ord_id)
            # a cancel for an order we do not know about is a bug worth seeing
            if order is None:
                self._emit("cancel_unknown_order",
                           {"cl_ord_id": action.cl_ord_id})
                return
            # the exposure is still real until the exchange confirms
            order.on_cancel_sent()
            # the cancel's own id resolves to the same order, so a Cancel
            # Reject quoting that id can be matched back to it
            self._alias[action.cl_ord_id] = order.cl_ord_id
            # record it
            self._emit("cancel_sent", {"cl_ord_id": action.cl_ord_id,
                                       "orig_cl_ord_id": order.cl_ord_id,
                                       "symbol": order.symbol})

    # ---- what the exchange tells us ---------------------------------------
    def on_ack(self, cl_ord_id: str, exchange_order_id: str) -> None:
        """The exchange acknowledged an order; it is resting."""
        # find it
        order = self._require(cl_ord_id, "ack")
        # an ack we cannot place is a state mismatch, not a routine event
        if order is None:
            return
        # move it to live and remember the exchange's handle
        order.on_ack(exchange_order_id)
        # record it
        self._emit("order_acked", {"cl_ord_id": cl_ord_id,
                                   "exchange_order_id": exchange_order_id})

    def on_fill(self, fill: Fill) -> None:
        """An execution arrived. THIS is the only thing that moves position."""
        # find the order
        order = self._require(fill.cl_ord_id, "fill")
        # a fill for an unknown order means our record and theirs disagree, and
        # the position is still real, so it is counted anyway and flagged loudly
        if order is None:
            self._position[fill.symbol] = (self._position.get(fill.symbol, 0)
                                           + fill.signed_quantity)
            self._emit("fill_unknown_order",
                       {"cl_ord_id": fill.cl_ord_id, "symbol": fill.symbol,
                        "qty": fill.quantity})
            return
        # advance the order's own state
        order.on_fill(fill.quantity)
        # POSITION COMES FROM FILLS AND NOTHING ELSE -- never from what we
        # believe we sent, never from what we expected to happen.
        self._position[fill.symbol] = (self._position.get(fill.symbol, 0)
                                       + fill.signed_quantity)
        # the ratio control needs to know an execution happened
        self._gateway.on_trade()
        # a fully filled order is no longer resting
        if order.state is OrderState.FILLED:
            self._working.pop((order.symbol, order.side), None)
        # record it
        self._emit("fill", {"cl_ord_id": fill.cl_ord_id, "symbol": fill.symbol,
                            "side": fill.side.value, "px": fill.price_minor,
                            "qty": fill.quantity,
                            "position": self._position[fill.symbol]})

    def on_replaced(self, cl_ord_id: str, price_minor: int,
                    quantity: int) -> None:
        """The exchange confirmed an amendment; the new terms are live.

        `cl_ord_id` may be either the amendment's id or the original's -- the
        exchange quotes the new one, and both are registered against the same
        order object for exactly this reason.
        """
        # find it
        order = self._require(cl_ord_id, "replaced")
        # nothing to do for an order we do not have
        if order is None:
            return
        # apply the new terms
        order.on_replaced(price_minor, quantity)
        # record it
        self._emit("order_replaced", {"cl_ord_id": cl_ord_id,
                                      "px": price_minor, "qty": quantity})

    def on_suspended(self, cl_ord_id: str) -> None:
        """The exchange is holding the order inactive.

        Reachable through the venue's suspend instruction. The engine never
        sends one, but an operator or the exchange can, and an inbound status
        we cannot represent is an inbound status we will mishandle.
        """
        # find it
        order = self._require(cl_ord_id, "suspended")
        # nothing to do for an order we do not have
        if order is None:
            return
        # inactive, but not gone
        order.on_suspended()
        # record it
        self._emit("order_suspended", {"cl_ord_id": cl_ord_id})

    def on_cancel_rejected(self, cl_ord_id: str, reason: str) -> None:
        """The exchange refused a cancel or an amendment.

        THIS IS THE DANGEROUS ONE. A refused cancel means the order is STILL
        RESTING while our own state says a cancel is in flight -- and an order
        stuck in PENDING_CANCEL is one the diff will never touch again, so it
        would rest untouched for the rest of the session. The order goes back to
        live so the next cycle can try again.
        """
        # find it
        order = self._require(cl_ord_id, "cancel_rejected")
        # nothing to do for an order we do not have
        if order is None:
            return
        # back to resting, keeping any partial fill already taken
        order.state = (OrderState.PARTIALLY_FILLED if order.filled_quantity
                       else OrderState.LIVE)
        # record it, and mark it critical: a refused cancel is never routine
        self._emit("cancel_rejected", {"cl_ord_id": cl_ord_id,
                                       "reason": reason})

    def on_cancelled(self, cl_ord_id: str) -> None:
        """The exchange confirmed a cancel."""
        # find it
        order = self._require(cl_ord_id, "cancelled")
        # nothing to do for an order we do not have
        if order is None:
            return
        # now genuinely off the book
        order.on_cancelled()
        # and no longer the resting order for its side
        self._working.pop((order.symbol, order.side), None)
        # record it
        self._emit("order_cancelled", {"cl_ord_id": cl_ord_id})

    def on_rejected(self, cl_ord_id: str, reason: str) -> None:
        """The exchange refused an order."""
        # find it
        order = self._require(cl_ord_id, "rejected")
        # nothing to do for an order we do not have
        if order is None:
            return
        # terminal, with the reason kept for the audit trail
        order.on_rejected(reason)
        # the side is empty again, so the next reconcile may re-quote it
        self._working.pop((order.symbol, order.side), None)
        # record it
        self._emit("order_rejected", {"cl_ord_id": cl_ord_id,
                                      "reason": reason})

    def _require(self, cl_ord_id: str, event: str) -> Optional[Order]:
        """Look up an order, recording the miss rather than raising.

        An event for an order we do not know about means our state and the
        exchange's have diverged. That is serious, but throwing from inside an
        inbound-message handler would take the session down and leave live
        orders resting with nothing watching them. Record it and continue.
        """
        # the order by its own id, or by the id of a cancel/amendment that
        # refers to it
        order = self._orders.get(cl_ord_id) or \
            self._orders.get(self._alias.get(cl_ord_id, ""))
        # a miss is recorded, not raised
        if order is None:
            self._emit("unknown_order", {"cl_ord_id": cl_ord_id,
                                         "event": event})
        # whatever we found
        return order

    # ---- state, for anything that needs to look ---------------------------
    def position(self, symbol: str) -> int:
        """Net position in one symbol, from fills only."""
        # zero when we have never traded it
        return self._position.get(symbol, 0)

    def working_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Every order that can still be executed, optionally for one symbol.

        PENDING_CANCEL counts. A cancel can lose the race with an incoming
        aggressor, so the exposure is real until the exchange confirms it.
        """
        # every order still capable of being filled
        out = [o for o in self._orders.values() if o.state.is_working]
        # narrowed to one symbol when asked
        return [o for o in out if symbol is None or o.symbol == symbol]

    def is_flat(self) -> bool:
        """Nothing resting, nothing in flight, no position anywhere.

        This is the condition the kill switch is trying to reach, and the thing
        an operator actually wants to see after tripping it.
        """
        # no working orders and no non-zero position
        return (not self.working_orders()
                and all(p == 0 for p in self._position.values()))

    def _emit(self, event: str, payload: dict) -> None:
        """Hand one state change to the audit sink, if there is one."""
        # no sink configured: nothing to do
        if self._on_event is None:
            return
        # record it
        self._on_event(event, payload)
