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
    # WHAT TO DO WHEN THE SIZE RESTING IS NOT THE SIZE WANTED.
    #
    # PSX Regulations 8.5.2 makes this asymmetric, and the policy has to be
    # too. An amendment that REDUCES the quantity is applied in place and keeps
    # its queue position. An amendment that raises it, or changes the price,
    # goes to the BACK of the queue at that price. So shrinking is free and
    # growing is expensive, and a single boolean cannot express that.
    #
    #   "queue_preserving"
    #                  A LIST of orders per side. Never amend upward and never
    #                  cancel a remainder: show more size by sending a SECOND
    #                  ORDER for the increment, so the shares already resting
    #                  keep the place they earned and only the new ones join
    #                  the back. Shrink the YOUNGEST order first, since a
    #                  reduction is free and the oldest has the best position.
    #                  This is the design worth running; the others exist to
    #                  measure it against.
    #
    #   "exact"        Requote on ANY size difference. This is mm_backtest's
    #                  rule -- its no-churn check compares price AND quantity
    #                  -- so it is what every measured number was produced
    #                  under, and it is what the reconcile gate runs. It tops
    #                  the clip back up after every partial fill, and under
    #                  8.5.2 each of those top-ups surrenders the queue
    #                  position the order had earned.
    #
    #   "reduce_only"  Requote only when we want LESS than is resting. That
    #                  amendment is the one 8.5.2 carves out, so it costs
    #                  nothing. When we want MORE -- after a partial fill --
    #                  leave the remainder where it is and keep its place,
    #                  showing less size until it is hit. This is the policy
    #                  that takes every free amendment and pays for none.
    #
    #   "ignore"       Never requote on size alone. Simplest, and it also
    #                  declines the free reductions.
    #
    # THE DEFAULT IS "ignore" ONLY BECAUSE IT IS THE BEHAVIOUR THAT SHIPPED.
    # It is not a recommendation. Pick one deliberately, with a number.
    quantity_policy: str = "ignore"


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
        # ---- WHY A CYCLE PRODUCED NO ACTIONS -----------------------------
        # Plain counters, read by sim/gate.py. They exist because the two
        # quantity policies came back with a threefold difference in orders
        # sent and the reason was GUESSED AT rather than measured -- twice.
        # A quiet side has exactly four causes and each one is counted here,
        # so the next comparison starts from a number.
        self.plan_counts = {
            # some order on this side has a message outstanding, so nothing
            # may be sent for ANY of them until the exchange answers. Under
            # the list policy this waits on the SLOWEST outstanding answer,
            # which is why it is worth separating from the rest.
            "held_message_in_flight": 0,
            # an order the exchange is holding inactive
            "held_suspended": 0,
            # what is resting already matches what is wanted
            "no_change": 0,
            # the side was asked for a quote and produced one or more actions
            "acted": 0,
        }
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
        # A TYPO IN THE POLICY MUST NOT BE A SILENT NO-OP. Checked once here
        # rather than on every quote, and refused rather than defaulted: an
        # unrecognised policy that quietly behaved like "ignore" would produce
        # a full run of numbers that answer a different question.
        if self._tol.quantity_policy not in ("queue_preserving", "exact",
                                             "reduce_only", "ignore"):
            raise ValueError(
                f"OrderManager: unknown quantity_policy "
                f"{self._tol.quantity_policy!r}; expected one of "
                f"queue_preserving / exact / reduce_only / ignore")
        # where every state change is recorded, for the audit trail
        self._on_event = on_event
        # what the strategy currently wants, per symbol
        self._desired: Dict[str, DesiredQuotes] = {}
        # the order resting on each (symbol, side), when there is one
        # ORDERS RESTING ON EACH SIDE, OLDEST FIRST.
        #
        # A LIST, not a single order, and the order of the list is its queue
        # order. Under PSX 8.5.2 an amendment that raises an order's size sends
        # it to the BACK of the queue at that price -- so topping a quote back
        # up after a partial fill surrenders the position it had earned. A
        # SECOND ORDER for the incremental size does not: the remainder keeps
        # its place and only the new shares join the back.
        #
        # That is the whole reason this is a list. With one order per side the
        # only way to show more size is to amend, and amending always pays.
        self._working: Dict[Tuple[str, Side], List[Order]] = {}
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
    def _unwork(self, order: Order) -> None:
        """Take one order off its side, leaving the others where they are."""
        # the side's list, if it has one
        key = (order.symbol, order.side)
        # nothing recorded for this side
        if key not in self._working:
            return
        # drop this order by identity, not by value: two orders on a side can
        # carry the same price and size and still be different orders
        self._working[key] = [o for o in self._working[key] if o is not order]
        # an empty side keeps no entry, so "is there anything resting" stays a
        # simple truth test everywhere else
        if not self._working[key]:
            del self._working[key]

    def _resting(self, symbol: str, side: Side) -> List[Order]:
        """What is actually resting on one side, oldest first.

        Terminal orders are swept out here rather than everywhere else: they
        are not resting, and leaving them in the list would make the size
        arithmetic below count shares that no longer exist.
        """
        # whatever the side has
        orders = self._working.get((symbol, side), [])
        # only the ones the exchange may still fill
        live = [o for o in orders if o.state.is_working]
        # keep the list clean so the next call does less work
        if len(live) != len(orders):
            # write back, or drop the side entirely when nothing survives
            if live:
                self._working[(symbol, side)] = live
            else:
                self._working.pop((symbol, side), None)
        # oldest first, which is best queue position first
        return live

    def _matches(self, order: Order, want: QuoteIntent) -> bool:
        """Is this resting order close enough to what we want to leave alone?"""
        # the tick size around this price; a tiered venue varies it by price
        tick = self._venue.tick_minor(want.price_minor)
        # how far the resting price is from the desired one, in ticks
        drift = abs(order.price_minor - want.price_minor) / tick
        # a price move beyond the tolerance means requote
        if drift > self._tol.price_ticks:
            return False
        # THE SIZE POLICY, per QuoteTolerance above. Checked before the ratio
        # because "exact" is the stricter rule and subsumes it.
        policy = self._tol.quantity_policy
        # mm_backtest's rule: any difference at all is a requote
        if policy == "exact":
            # including the size lost to a partial fill, which it tops back up
            if order.leaves_quantity != want.quantity:
                return False
        # take the free amendment, never the expensive one
        elif policy == "reduce_only":
            # wanting LESS than is resting: 8.5.2 applies that in place
            if want.quantity < order.leaves_quantity:
                return False
            # wanting MORE: leave it, because growing costs the queue position
        # a size check only when one was asked for; 0 disables it entirely, so
        # a partial fill does not cost us the queue position for the remainder
        if self._tol.qty_ratio > 0.0:
            # what is still working against what we want working
            if order.leaves_quantity < self._tol.qty_ratio * want.quantity:
                return False
        # close enough: leave it where it is and keep its place in the queue
        return True

    def _plan_side(self, symbol: str, side: Side,
                   want: Optional[QuoteIntent]) -> List[Action]:
        """Dispatch to whichever quoting design this manager is configured for.

        TWO DESIGNS, AND THE DIFFERENCE IS QUEUE POSITION.

        "queue_preserving" holds a LIST of orders per side. After a partial
        fill it tops the quote back up with a SECOND ORDER, so the remainder
        keeps the place it earned and only the increment joins the back of the
        queue. When it has to shrink, it shrinks the youngest order first.
        This is the design worth running.

        "exact" and "ignore" hold ONE order per side and express every change
        as a cancel-and-replace or an amendment of that single order. "exact"
        is mm_backtest's rule -- requote on any difference in price or size --
        and it exists so the reconcile gate can reproduce the measured numbers.
        It surrenders queue position on every top-up, which is precisely the
        cost the other design avoids and the thing worth measuring.
        """
        # the list design
        if self._tol.quantity_policy == "queue_preserving":
            return self._plan_side_multi(symbol, side, want)
        # the single-order design, which is what shipped
        return self._plan_side_single(symbol, side, want)

    def _plan_side_single(self, symbol: str, side: Side,
                          want: Optional[QuoteIntent]) -> List[Action]:
        """ONE order per side. The behaviour every measured number came from.

        Kept so the reconcile gate has something to compare against. Under this
        design a top-up is an amendment or a cancel-and-replace of the whole
        order, and PSX 8.5.2 sends that to the back of the queue every time.
        """
        # whatever is resting, at most one under this design
        resting = self._resting(symbol, side)
        # the single order, or nothing
        order = resting[0] if resting else None
        # ---- NOTHING RESTING ------------------------------------------
        if order is None:
            # and nothing wanted: no action
            if want is None:
                return []
            # wanted but absent: send it
            return [self._place(symbol, side, want.price_minor, want.quantity)]
        # ---- IN FLIGHT: WAIT ------------------------------------------
        # PSX requires OrderID on a cancel or an amendment, and OrderID is
        # assigned on the acknowledgement. Before the ack there is nothing to
        # put in the message.
        # NOTE `has_message_in_flight`, NOT `state.is_in_flight`. A fill
        # overwrites `state`, so a partially filled order with an amendment
        # still on the wire reads as quiescent to the state property. Acting on
        # it is what produced the InvalidTransition that killed two of six
        # symbol-days in the reconcile gate.
        if order.has_message_in_flight:
            self.plan_counts["held_message_in_flight"] += 1
            return []
        # ---- SUSPENDED: leave it for an operator -----------------------
        if order.state is OrderState.SUSPENDED:
            self.plan_counts["held_suspended"] += 1
            return []
        # ---- RESTING, NOT WANTED ---------------------------------------
        if want is None:
            return [self._cancel(order)]
        # ---- RESTING, CLOSE ENOUGH -------------------------------------
        if self._matches(order, want):
            self.plan_counts["no_change"] += 1
            return []
        # ---- RESTING, WRONG: amend if the venue takes one, else replace --
        if (self._use_replace and self._venue.supports_replace
                and order.exchange_order_id):
            return [self._replace(order, want)]
        # cancel now, place on a later cycle -- the in-flight rule above holds
        # the side until the exchange has answered
        return [self._cancel(order)]

    def _plan_side_multi(self, symbol: str, side: Side,
                         want: Optional[QuoteIntent]) -> List[Action]:
        """What, if anything, needs to happen on one side of one symbol.

        RETURNS A LIST, because showing more size at a price we are already
        resting at takes a SECOND ORDER, not an amendment. PSX Regulations
        8.5.2: an amendment that raises the size goes to the back of the queue
        at that price, while a second order leaves the first exactly where it
        is and sends only the new shares to the back. The first order keeps
        the priority it earned; we pay only on the increment.

        The four cases:

          nothing wanted        cancel everything resting on this side.
          nothing resting       place the full size.
          resting, wrong price  the price moved, so priority at the old level
                                is worthless. Move the oldest order to the new
                                price and cancel any others.
          resting, right price  compare TOTAL resting size against what we
                                want. Short -> add a second order for the
                                difference. Over -> reduce, shrinking the
                                YOUNGEST order first so the oldest keeps its
                                place. A reduction is the one amendment 8.5.2
                                applies in place, so it costs nothing.
        """
        # what is actually resting here, oldest first
        resting = self._resting(symbol, side)

        # ---- NOTHING WANTED -------------------------------------------
        if want is None:
            # pull everything that is not already on its way out
            return [self._cancel(o) for o in resting
                    if not o.has_message_in_flight]

        # ---- ANYTHING IN FLIGHT: WAIT ----------------------------------
        # The exchange has not answered yet, so we do not know what it thinks
        # exists. AND ON PSX IT IS NOT MERELY UNWISE: Order Cancel ('F') and
        # Cancel/Replace ('G') both carry OrderID (tag 37) as REQUIRED, and
        # OrderID is assigned on the acknowledgement. Before the ack there is
        # literally nothing to put in the message.
        #
        # NOTE THIS BLOCKS THE WHOLE SIDE, not just the in-flight order. Adding
        # a second order while the first is unacknowledged would be safe on the
        # wire, but it would make the resting total ambiguous at the moment the
        # risk gateway judges it, and the gateway is not a place for ambiguity.
        if any(o.has_message_in_flight for o in resting):
            self.plan_counts["held_message_in_flight"] += 1
            return []

        # ---- SUSPENDED --------------------------------------------------
        # Held inactive by the exchange: it cannot trade and cannot be amended
        # into a live quote. Leave it for an operator rather than guessing.
        if any(o.state is OrderState.SUSPENDED for o in resting):
            self.plan_counts["held_suspended"] += 1
            return []

        # ---- NOTHING RESTING --------------------------------------------
        if not resting:
            # the whole size, in one order
            return [self._place(symbol, side, want.price_minor, want.quantity)]

        # ---- SPLIT BY PRICE ---------------------------------------------
        # orders already at the price we want, oldest first
        at_price = [o for o in resting
                    if self._matches_price(o, want)]
        # and everything sitting at a price we no longer want.
        # BY IDENTITY, NOT BY VALUE. Order is a plain dataclass, so `in` would
        # compare every field -- and two orders on the same side at the same
        # price for the same size are equal by that test while being different
        # orders with different places in the queue.
        at_ids = {id(o) for o in at_price}
        off_price = [o for o in resting if id(o) not in at_ids]

        # ---- THE PRICE MOVED --------------------------------------------
        # Priority at a price we are leaving is worth nothing, so there is
        # nothing to protect here. Move the oldest order across and cancel the
        # rest; the size arithmetic below then applies on the next cycle.
        if off_price and not at_price:
            # the one we move
            first = off_price[0]
            # an amendment when the venue takes one and we have its handle,
            # otherwise the cancel-and-replace path on the next cycle
            head = ([self._replace(first, want)]
                    if (self._use_replace and self._venue.supports_replace
                        and first.exchange_order_id)
                    else [self._cancel(first)])
            # anything else on this side is surplus at a stale price
            return head + [self._cancel(o) for o in off_price[1:]]

        # a stale order alongside good ones is simply cancelled
        actions: List[Action] = [self._cancel(o) for o in off_price]

        # ---- SAME PRICE: COMPARE TOTALS ---------------------------------
        # every share we have resting at this price
        total = sum(o.leaves_quantity for o in at_price)

        # SHORT OF WHAT WE WANT -> A SECOND ORDER FOR THE DIFFERENCE.
        # This is the case the whole list exists for. The orders already there
        # keep their queue position untouched; only the increment joins the
        # back.
        if total < want.quantity:
            # just the shortfall, never the whole size again
            actions.append(self._place(symbol, side, want.price_minor,
                                       want.quantity - total))
            return actions

        # OVER WHAT WE WANT -> REDUCE, YOUNGEST FIRST.
        # A reduction keeps its place under 8.5.2, so this costs nothing. Doing
        # it youngest-first matters: the oldest order has the best position in
        # the queue and is the last thing to give up.
        if total > want.quantity:
            # how many shares have to come off
            excess = total - want.quantity
            # walk from the back of the list, which is the back of the queue
            for o in reversed(at_price):
                # done
                if excess <= 0:
                    break
                # how much this order can give up
                take = min(excess, o.leaves_quantity)
                # it goes entirely: a cancel, not a zero-size amendment
                if take >= o.leaves_quantity:
                    actions.append(self._cancel(o))
                # it shrinks: the free amendment, when the venue takes one
                elif (self._use_replace and self._venue.supports_replace
                      and o.exchange_order_id):
                    # same price, smaller size -- applied in place
                    actions.append(self._replace(
                        o, QuoteIntent(side=side,
                                       price_minor=o.price_minor,
                                       quantity=o.leaves_quantity - take)))
                # no amendment available: cancelling is the only way to shrink,
                # and it costs the position. Better than showing size we do not
                # want.
                else:
                    actions.append(self._cancel(o))
                # account for what came off
                excess -= take
            return actions

        # EXACTLY RIGHT: leave every one of them alone. `actions` may still
        # carry cancels for orders at a stale price, so this is only a quiet
        # cycle when it is empty.
        if not actions:
            self.plan_counts["no_change"] += 1
        return actions

    def _matches_price(self, order: Order, want: QuoteIntent) -> bool:
        """Is this order at the price we want, within the tolerance?"""
        # the tick around this price; a tiered venue varies it
        tick = self._venue.tick_minor(want.price_minor)
        # distance in ticks
        drift = abs(order.price_minor - want.price_minor) / tick
        # inside the tolerance counts as the same price
        return drift <= self._tol.price_ticks

    def _place(self, symbol: str, side: Side, price_minor: int,
               quantity: int) -> PlaceOrder:
        """A new order for a stated size at a stated price."""
        # every field the venue requires on a New Order Single
        return PlaceOrder(symbol=symbol, cl_ord_id=self._next_id(), side=side,
                          price_minor=price_minor, quantity=quantity,
                          account=self._account)

    def _replace(self, order: Order, want: QuoteIntent) -> ReplaceOrder:
        """An amendment carrying every identifier the venue requires."""
        # PSX needs a NEW ClOrdID for the amendment, the original's id, and the
        # exchange's own handle for the order being changed
        return ReplaceOrder(symbol=order.symbol, cl_ord_id=self._next_id(),
                            side=order.side, price_minor=want.price_minor,
                            quantity=want.quantity, account=self._account,
                            orig_cl_ord_id=order.cl_ord_id,
                            exchange_order_id=order.exchange_order_id)

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
                # what this side needs -- a LIST, because showing more size
                # at a price we already rest at takes a second order
                for action in self._plan_side(symbol, side,
                                              effective.side(side)):
                    # the state the risk controls judge against
                    ctx = RiskContext(date=date, timestamp_ms=now_ms,
                                      position=self._position.get(symbol, 0),
                                      reference_price_minor=refs.get(symbol))
                    # EVERY action clears the gateway. There is no other path.
                    decision = self._gateway.authorise(action, ctx)
                    # refused: record it and move on. A refused place means no
                    # quote there this cycle, which is the control working.
                    #
                    # NOTE ONE REFUSAL DOES NOT CANCEL THE OTHERS. Each action
                    # on a side is judged on its own: a rejected top-up must
                    # not also drop the cancel that was going out beside it,
                    # because the cancel is the half that REDUCES exposure.
                    if not decision.allowed:
                        self._emit("risk_rejected",
                                   {"symbol": symbol, "side": side.value,
                                    "check": decision.check,
                                    "reason": decision.reason})
                        continue
                    # approved: move our own state to match what is in flight
                    self._mark_sent(action)
                    # cancels are collected separately so they go out first
                    (cancels if action.is_cancel else places).append(action)
                    # this side did something this cycle
                    self.plan_counts["acted"] += 1
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
            # remember it by id, and APPEND to the side -- appending is what
            # keeps the list in queue order, oldest first, which is what the
            # reduce path relies on to shrink the youngest first
            self._orders[order.cl_ord_id] = order
            self._working.setdefault((order.symbol, order.side), []).append(order)
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
            self._unwork(order)
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
        # the state before, so a race can be reported rather than inferred
        was_terminal = order.state.is_terminal
        # apply the new terms -- a no-op on an order that already finished
        order.on_replaced(price_minor, quantity)
        # AN AMENDMENT THAT LOST ITS RACE. The order filled or was cancelled
        # while the message was on the wire, so nothing was applied. Worth a
        # line of its own: it is a real cost of the one-message reprice and the
        # count of it says how often the amendment window bites.
        if was_terminal:
            self._emit("replace_confirmed_after_terminal",
                       {"cl_ord_id": cl_ord_id, "state": order.state.value})
            return
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
        # A TERMINAL ORDER STAYS TERMINAL, added 2026-09-17.
        #
        # The commonest reason a cancel or an amendment is refused is that it
        # LOST A RACE: the order filled, or was already cancelled, before the
        # exchange reached our message. The order is finished. Putting it back
        # to LIVE would return an order the exchange has done with to our
        # working set, where _plan_side sees a resting order that matches what
        # we want and leaves it alone -- forever. The side then never quotes
        # again for the rest of the session and nothing raises.
        #
        # Found by the reconcile-gate tests: an amendment that lost to a full
        # fill came back as PARTIALLY_FILLED with 50 of 50 filled.
        # THE MESSAGE HAS BEEN ANSWERED -- with a refusal, but answered. Both
        # flags are cleared FIRST, before the terminal check returns, or a
        # terminal order would keep a raised flag and nothing would ever lower
        # it. A flag that is never lowered holds its side for the rest of the
        # session and raises nothing.
        order.replace_in_flight = False
        order.cancel_in_flight = False
        if order.state.is_terminal:
            # say so, because a reject arriving after the end is worth seeing
            self._emit("cancel_rejected_after_terminal",
                       {"cl_ord_id": cl_ord_id, "reason": reason,
                        "state": order.state.value})
            # and change nothing else
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
        # and no longer resting on its side
        self._unwork(order)
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
        # one fewer order resting, so the next reconcile may replace it
        self._unwork(order)
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
