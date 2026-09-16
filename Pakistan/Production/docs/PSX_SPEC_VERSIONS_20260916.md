# The two PSX specs do not describe the same system

Reviewed 2026-09-16.

| | Order entry | Market data |
|---|---|---|
| Document | FIX Specification **v1.2** | FIX Market Data Interface **Ver 1.05** |
| Dated | 19-Sep-2012, effective 24-Sep-2012 | 2-Apr-2024 (first draft 2-Mar-2020) |
| Protocol | **FIX 4.2** | **FIX 5.0 SP2** + extensions |
| Session layer | standard FIX 4.2 | **"Light-weight FIX Session Layer Protocol"** |
| Market codes | alphabetic — `REG`, `FUT`, `CSF`, `SQR` | numeric — `'01'` Regular, `'03'` Deliverable Future, `'04'` CSF |
| Appendix titled | **"KATS Transactions & Tags Selection"** | — |
| Page footer | — | **STSV5** |
| Version tag | — | `FIX5.00_PSX_1.00` in `DefaultCstmApplVerID` |

## What follows from that

These are not two halves of one interface. They are **twelve years and a
platform apart**. The order-entry document names KATS — the trading system PSX
ran when it was written. The market-data document is a channel-based FIX 5.0 SP2
feed with its own session protocol, its own numeric market codes, a
`TradingPhaseCode` state machine, and a versioning scheme (`FIX5.00_PSX_n.xy`)
that the 2012 document knows nothing about.

An exchange does not run a 2024 FIX 5.0 SP2 market-data gateway against a 2012
FIX 4.2 order-entry gateway on the same matching engine. **The FIX 4.2
order-entry specification is very probably superseded.**

I am not asserting that as fact — I have not seen a newer order-entry document
and I will not invent one. What is fact is that the two documents in hand are
mutually inconsistent, and the market-data one is current.

### The one thing to ask for

> **The order-entry (trading) interface specification that matches the STSV5 /
> FIX 5.0 platform.** Its market codes should be numeric and its
> `DefaultCstmApplVerID` should read `FIX5.00_PSX_n.xy`, matching the
> market-data document.

Everything in `PSX_FIX_COMPLIANCE_20260916.md` about message types, required
tags and prohibited characters is accurate **for the 2012 document** and should
be re-derived against whatever comes back.

---

## What I changed, using the spec that IS current

The market-data specification is current, and three of its provisions changed
real behaviour in the engine.

### 1. The exchange tells us what the market is doing — stop guessing

`TradingPhaseCode` (tag 8538) is published **every three seconds** on the
Trading Session Status message and carried on every snapshot.

`TradingWindowCheck` previously read a session calendar built from historical
data (`session_segments_*.csv`). **A calendar is a guess about the future
written down in advance.** It cannot know that the market halted two minutes
ago, or that one security is suspended for the day. It reports the market open
while the exchange has stopped matching, and the engine quotes into a market
that is not there.

**Changed.** When a phase feed is wired in it decides; the calendar stays as the
fallback for replay and for a feedless session. The phase covers, from the
exchange's own mouth:

| Code | Meaning | Quote? |
|---|---|---|
| `T` | Continuous Auction | **yes — the only one** |
| `O` `N` `V` | call auctions (morning, post-Friday-break, post-halt) | no |
| `B` | Trading Break — 2nd digit `2` is the **Jumu'ah break** | no |
| `H` | Temporary Suspension (halt) | no |
| `S` `C` `A` `E` | starting, pre-close, after-hours, closed | no |
| 1st digit `1` | **this security suspended all day** | no |

That last one is per instrument and has no calendar equivalent at all.

**Verified 2026-09-16** against the specification itself and against
`PSX_Parser_Mac.py`'s `PHASE_MAP` / `BREAK_REASON_MAP`. Every code matches, and
the digit numbering above follows the spec's own 0-based scheme. Two nuances
this table omits:

- the spec marks `C` (Close Call Auction) **"(reserved)"** — it may never be
  emitted. `psx.py` maps it anyway, which is harmless;
- the 1st digit carries the all-day-suspension flag on the **snapshot**, but is
  **"(reserved)"** on the Trading Session Status message. `psx.py:parse_phase`
  reads it unconditionally, so a session-status code whose reserved digit
  happened to be `1` would mark a security suspended. Low probability, one-line
  guard, recorded rather than fixed.

**An unknown or unrecognised phase is a rejection, not a pass.** Before the
first status message arrives we have not been told what the market is doing, and
"not told" is not permission. The engine stays dark for up to three seconds
after connecting. That is correct.

The Friday break also stops being something we infer from a data-derived
calendar — the exchange names it.

### 2. "No price limit" has a sentinel value, and taking it literally is wrong

`MDEntryType` `xe` is the up limit and `xf` the down limit. **999999999.9999
means no rise limit** — verified: it is `XE_NO_LIMIT_SENTINEL` in
`PSX_Parser_Mac.py`.

> **CORRECTED 2026-09-16.** This paragraph also said "a value equal to one tick
> (0.01 on the regular market) means no fall limit", as though the spec gives a
> clean rule for `xf`. **`PSX_Parser_Mac.py` says the opposite and says it
> explicitly:** `xf` has *no universal sentinel* — its no-limit value equals the
> market's minimum tick, "which varies by market/segment and **must not be
> hard-coded/nulled blindly**". The parser acts on that: it nulls `xe` outright,
> and for `xf` it sets a heuristic flag (`is_xf_tick_floor`, px ≤ 1.0) and leaves
> the decision downstream.
>
> Nothing in the engine currently nulls an `xf` value — `PSXVenue.price_band()`
> delegates to a `band_provider` that is not built yet. **Whoever writes that
> provider must follow the parser, not the sentence this replaced.**
>
> **AMENDED same day, on a real snapshot row.** The sentinel is an edge case,
> not the design. In the data both bands are ordinary published prices: a row
> carrying LAST_TRADE 336.00 and NET_CHANGE_1 0.80 (previous close 335.20)
> publishes `xe` 368.72 and `xf` 301.68 — exactly ±10%, to the paisa. The
> provider is a **read of `xe` and `xf` off the ob snapshot**, sticky when a
> snapshot omits the row, with the sentinel handled as a guard.
>
> **The ±10% is a reconciliation, not a rule to code.** On a split or reverse
> split the exchange bands off the ADJUSTED close, so a derived band is wrong by
> the split ratio, on the day the band matters most. Read the published values;
> never compute them.

A parser that takes those literally builds a band that is arithmetically valid
and completely meaningless. `PriceBand` bounds are now `Optional`; `None` says
"no limit" honestly, and `is_unbounded` surfaces a band that constrains nothing.

### 3. The feed carries more precision than paisa

Market-data `Price` is **N13(4)** and `MDEntryPx` is **N18(6)**. Our entire
engine holds prices as integer paisa — two decimals.

For regular-market equities with a 0.01 tick that is exact and the extra
decimals are zeros. But the *field* carries more, and index values use all six.
`Venue.parse_price()` now **raises rather than truncates** when a value carries
precision the minor unit cannot hold. If PSX ever publishes a finer price we
find out on the first message, rather than trading for a year on prices that
were quietly rounded.

---

## Also worth having from the market-data spec, not yet built

- **`SecurityStatus` (MsgType `f`)** publishes per-symbol switches every 15
  seconds: **Sell Short**, Borrow (Leverage Buy), Blank Sell, MSF Buy, each
  `Y`/`N`. This is part of the answer to the open short-selling question — the
  exchange says per symbol whether short selling is open.
- **Tick data gives full order-by-order flow** — `TickOrder` (UA201) and
  `TickExecution` (UA202) with `BidApplSeqNum` / `OfferApplSeqNum` linking each
  execution to the two orders behind it.
  **CORRECTED 2026-09-16:** this bullet ended "which is precisely what the
  backtest's fill model lacks". It is not. `mm_backtest` already tracks queue
  position exactly, from the order-level data in the parsed store — see
  `PSX_DOC_AUDIT_20260916.md` §1.1. The tick channel is a *live* source for the
  same thing, which matters for the live engine, not a gap in the backtest.
- **Snapshot depth is 10 levels** (`MDPriceLevel` 1–10) with `NumberOfOrders`
  per level, and optionally the individual `OrderQty` values at each level.
- **Sequence gaps are detectable and recoverable** — `ApplSeqNum` per channel,
  with a retransmission session. A gap is a known-missing, not a silent hole.
- **Channel heartbeat every 3 seconds**; no message for more than two intervals
  means disconnect and reconnect. That is the connectivity-loss trigger the
  engine needs in order to go flat.
