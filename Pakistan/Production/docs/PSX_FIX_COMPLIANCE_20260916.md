# PSX FIX 4.2 — compliance review of the production engine

Spec reviewed: **PSX Financial Information eXchange (FIX) Specification v1.2**,
effective 24-Sep-2012, errata 20-Sep-2012. FIX 4.2.
Reviewed against: `Production/` as of 2026-09-16.

---

## 0. Read this first — is this specification current?

The document is **fourteen years old** and its Appendix D is titled *"KATS
Transactions & Tags Selection"*. KATS is the trading system PSX ran at the time
it was written. **Confirm with PSX or the broker that v1.2 is still the live
order-entry specification** before a line of encoder code is written against it.

I am not asserting that it has been superseded — I do not know. I am saying the
cost of checking is one email and the cost of not checking is an encoder built
to the wrong message set.

Everything below is true *of this document*.

---

## 1. What the specification changed in code already written

### 1.1 Amendments are supported — the OMS was being conservative for nothing

`Order Cancel/Replace Request` (MsgType `G`) exists and explicitly covers
*"reduce/increase quantity, change limit price"*. The order manager previously
cancelled and then placed on a **later cycle**, with a comment saying it could
not confirm the venue supported an amendment.

**Changed, and then deliberately reversed on 2026-09-16.** A reprice became one
`ReplaceOrder`; it is now a cancel-plus-new again, with amendment available
behind `OrderManager(use_replace=True)`.

Why the reversal: `mm_backtest._requote` sends a cancel and a replacement as two
independent messages with two independent latency draws, and **the old order
stays fillable until its cancel lands**. Every measured result was produced under
that behaviour, including the fills taken inside that window. An engine that
amends is not the thing that was backtested — very likely better, but "better" is
a claim, and shipping it by default makes the reconcile gate compare two
mechanics and unable to say anything exact. `sim/replay.py` raises on a
`ReplaceOrder` rather than dropping it.

**What this does NOT buy:** the spec does not say whether an amendment keeps
queue position. Most venues send an order to the back of the queue on a price
change or a size increase. Treat the latency saving as real and any queue saving
as **unproven until UAT measures it**.

### 1.2 The in-flight rule turned out to be mandatory, not merely prudent

Both `Order Cancel Request` (`F`) and `Order Cancel/Replace Request` (`G`) carry
**`OrderID` (tag 37) as Required = Y**, and `OrderID` is assigned by the exchange
on the acknowledgement.

So "never act on an unacknowledged order" is not caution. **Before the ack there
is nothing to put in the message.** The existing behaviour was right; the reason
given for it was weaker than the real one. Comment corrected.

### 1.3 A latent bug that the spec exposed

Adding `ReplaceOrder` would have walked straight past **every risk control**. The
controls tested `isinstance(action, PlaceOrder)`, and an amendment is not a
placement — so an amendment could have raised the price, increased the quantity,
or pushed the position past its limit with no check firing and **no error
anywhere**. Orders would simply have stopped being checked.

**Fixed structurally**, not by adding a case: `PlaceOrder` and `ReplaceOrder` now
share an `OrderRequest` base and every control tests for that. A future action
type that adds size is checked by default rather than by someone remembering.

---

## 2. Gaps the review found and closed

| # | Requirement | Where | Status |
|---|---|---|---|
| 1 | `Account` (tag 1) **Required = Y** on New Order Single — the Client Code | New Order Single | **Closed.** `Venue.requires_account`; the OMS refuses to construct without one |
| 2 | Appendix C prohibited characters in `Account`, `Symbol`, `ClOrdID`, `OrderID`, `Price`, `StopPx`, `LastPx` | Appendix C | **Closed.** `Venue.validate_text()`, applied where ids are generated |
| 3 | `.` permitted **once** in price fields only | Appendix C | **Closed.** `PSXVenue.format_price()` is the only place a `.` is emitted |
| 4 | Non-printables (0–31, 127) prohibited in **all** tags | Appendix C | **Closed.** Same validator |
| 5 | `OrdStatus` (39) carries Suspended, Pending Replace | Execution Report | **Closed.** Added to `OrderState` |
| 6 | `Order Cancel Reject` (MsgType `9`) | p.23 | **Closed.** `on_cancel_rejected()` — see below |
| 7 | Cancel requires a **new** `ClOrdID` plus `OrigClOrdID` plus `OrderID` | Order Cancel Request | **Closed.** Alias map resolves the exchange's reply back to the order |

**On item 6 — this one could have cost real money.** A refused cancel means the
order is *still resting* while our state says a cancel is in flight. An order
stuck in `PENDING_CANCEL` is one the diff never touches again, so it would have
rested untouched for the rest of the session with nothing trying to pull it. The
handler returns it to live so the next cycle retries.

**And one bug found while fixing item 7:** registering a single order object
under two ids made `working_orders()` count it twice, which made `is_flat()`
wrong. `is_flat()` is what an operator reads after tripping the kill switch.
Fixed with a separate alias map; a test now pins it.

---

## 3. Three questions for the broker — I will not guess these

### 3.1 How does a market maker's sell get entered when flat?

`Side` (54) has **five** values: `1` Buy, `2` Sell, `5` **Sell Short**, `8`
Cross, `G` Borrow. Appendix D's execution plan shows Short Sell requires
`LocateReqd` (114) = `N (FALSE)` and is accepted on REG and FUT.

**Our engine quotes two-sided from flat, so it will sell when it holds nothing.**
Whether that is entered as `2` or as `5` is a settlement and regulatory
question, not a coding one, and getting it wrong is not a bug we can fix after
the fact. Ask before UAT.

### 3.2 What is the Client Code, and is it one account or many?

`Account` (1) is required and the spec calls it the Client Code "as agreed
between broker and exchange". The engine currently takes one per session. If a
proprietary book needs a different code, or per-symbol codes, that changes the
order manager's shape.

### 3.3 Is there a session message-rate limit?

Still unanswered, and still the only real ceiling that exists now that no
regulator is setting one. `PSXVenue.max_orders_per_second` remains `None`,
which means *not told*, not *unlimited*.

---

## 4. A correction to the scope document

`PSX_LIVE_ENGINE_SCOPE_20260915_1900.md` said the conformance test was no longer
mandatory, because the SECP framework that required it is not being pursued.

**That was wrong.** PSX requires it independently:

> "Certification testing is required and can be arranged through our FIX
> Connectivity Department. **Before a subscriber can go live, it is mandatory to
> complete approved test scripts to become FIX Certified.**"

Certification is back on the critical path. It is an exchange requirement, not a
regulatory one, and it does not lapse with the concept paper.

---

## 5. Still to build against this spec

Nothing below is started. All of it is now unblocked.

- **Encoder / decoder** for MsgTypes `D`, `F`, `G`, `H` outbound and `8`, `9`,
  `B`, `0`, `A`, `5` inbound, with the checksum from Appendix B.
- **Header fields we do not model:** `SenderCompID` (49) = TraderID,
  `OnBehalfOfCompID` (115) = MemberID — required on *every* inbound message
  except Heartbeat — and `TargetLocationID` (143) = market code, required on
  `D`/`F`/`G`/`H`. Ours is `REG`.
- **Fixed order fields:** `HandlInst` (21) = `1`, `OrdType` (40) = `2` for a
  limit, `TimeInForce` (59) = `0` for Day.
- **Session layer:** Logon carries the password in `RawData` (96) with
  `RawDataLength` (95) and `EncryptMethod` (98) = `0`; Test Request after
  `HeartBtInt` plus a margin, connection considered lost after another.
- **`MaxFloor` (111)** — the disclosed/undisclosed decision. Every order sample
  in Appendix A comes in both forms. Showing full size on a thin book is
  information we may not want to give away; that is a strategy decision nobody
  has made yet.

---

## 6. One thing we are stricter than required about, deliberately

Appendix D shows the pre-open state accepts **Normal Order** on REG. Our
`TradingWindowCheck` blocks everything outside continuous trading.

That is a **choice, not a requirement**: there is no continuous matching in
pre-open, so a market maker resting a quote there is taking on risk with no
mechanism to earn the spread. Keeping the check as it is, and recording here
that it is stricter than the exchange demands so nobody later "fixes" it.
