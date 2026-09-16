# Truth pass over the engine documents

Date: 2026-09-16.
Audited: `Production/README.md` and the three documents in `Production/docs/`.
Method: every factual claim about SZ's code or about PSX checked against a file
in hand. Claims I could not check are marked unverified rather than left to look
authoritative.

**Why this exists.** `README.md` asserted that the backtest cannot see the cost
of losing queue position. It was written without opening `mm_backtest.py`, which
models queue position exactly. That was not a one-off: the same claim appears in
two more places, and the audit found several other assertions made about files I
had not read. This document lists all of them.

Files I now hold and audited against: `micro_mm.py`, `mm_backtest.py`,
`live_config.py`, `build_config_assignment.py`, three `config_assignment_*.csv`,
`live_overrides.csv`, `PSX_Parser_Mac.py`, the PSX FIX Market Data Interface
Specification, and the `Production/` tree itself.

**Not held:** `mm_harness.py`, `config_pk.py`, `expansion_names.py`, and the PSX
FIX order-entry specification v1.2 (reviewed earlier in the session, not
available now). Claims resting on those are marked.

---

## 1. Wrong, and the claim was load-bearing

### 1.1 "The backtest cannot see the cost of losing queue position"

**Appears in:** `PSX_LIVE_ENGINE_SCOPE` §3 (the section's whole argument) and §2
(the order-to-trade row points at it); `PSX_SPEC_VERSIONS`, final bullet
("a queue-position dataset, which is precisely what the backtest's fill model
lacks"); `README.md` (corrected 2026-09-16).

**It is false.** `mm_backtest.MyOrder` carries `ahead` — an order-id → qty dict
of everything resting at our price at the moment we arrived, built from
order-level data and maintained event by event:

- `_on_market_cancel` removes a cancelled order from every `ahead` dict, so our
  queue position improves when someone ahead of us pulls;
- `_on_market_trade` drains the pool, and when the trade row names its resting
  victim (`rest_oid`) it drains *exactly that entry*;
- `_on_snapshot_queue_reset` rebuilds it conservatively after a snapshot
  replaces the book — everything now at our price is assumed ahead of us, which
  understates our priority and never overstates it;
- on top sits `LatencyModel`, two-leg and stochastic, under which an order stays
  fillable until its cancel actually *lands*.

The scope document says "a quote resting for ten minutes and a quote posted one
millisecond ago fill identically." They do not. The ten-minute quote has a
drained `ahead` pool and the fresh one has a full one.

**Consequence.** The queue cost of `tol_ticks = 0.0` is already priced into every
measured result. §3's argument that the backtest "is structurally incapable of
charging us for churn" is wrong, and the urgency it assigns to the churn
measurement does not follow from it. Knowing the ratio is still worth an hour —
for the broker's message limit, which is a real unknown — but not for the reason
given.

### 1.2 "The same gap as the unaddressed `log_fill_state` item (C.18)"

**Appears in:** `PSX_LIVE_ENGINE_SCOPE` §3, closing line.

**It is false**, and for a second reason on top of 1.1. `log_fill_state` records,
at the instant a quote joins the queue, the features a fill-probability model
conditions on — and the first one it writes is `ahead_qty`, commented in
`mm_backtest` as *"shares resting ahead of us at our own price (the queue we wait
behind)"*. It is a logging switch over a queue the engine already tracks, not a
missing mechanism. There is no gap for it to be the other side of.

### 1.3 What survives from §8's `log_fill_state` note

> "If the fill model flatters us, Phase 2 will reconcile perfectly against a
> number that was never real."

**Still true and still important.** The reconcile gate compares the engine to the
backtest; it cannot tell you whether the backtest is right. `log_fill_state`
gives every posted quote, filled or not, with the circumstances it was posted
into — which is the dataset that would answer that. Keep the item; drop the
stated reason ("the most important unaddressed item"), which rested on 1.1.

---

## 2. Contradicted by SZ's own code

### 2.1 The circuit bands — right finding, wrong emphasis

**Amended 2026-09-16, same day, after SZ supplied a real snapshot row.**

The original finding: `PSX_SPEC_VERSIONS` §2 stated *"a value equal to one tick
(0.01 on the regular market) means no fall limit"* as a clean spec rule, and
`core/venue.py`'s `PriceBand` docstring repeated it. `PSX_Parser_Mac.py` says
`xf` has **no universal sentinel** — its no-limit value equals the market's own
minimum tick, "which varies by market/segment and **must not be
hard-coded/nulled blindly**" — and the parser acts on that: it nulls `xe`
outright and only *flags* `xf` (`is_xf_tick_floor`, px <= 1.0).

That part stands. **But the emphasis was wrong, and the wrong emphasis is the
more misleading half.** In the actual data both bands are ordinary published
prices. From one ob-snapshot row:

| Entry type | Code | Px |
|---|---|---|
| LAST_TRADE | `2` | 336.00 |
| NET_CHANGE_1 | `x1` | 0.80 |
| UPPER_CIRCUIT_BREAKER | `xe` | **368.72** |
| LOWER_CIRCUIT_BREAKER | `xf` | **301.68** |

Previous close = 336.00 − 0.80 = 335.20. Then 335.20 x 1.10 = **368.7200** and
335.20 x 0.90 = **301.6800** — both exact to the paisa. This is an ordinary
±10% band, published on both sides, on every snapshot.

> **THAT ARITHMETIC IS A CHECK, NOT A FORMULA. DO NOT IMPLEMENT IT.** (SZ,
> 2026-09-16.) It reconciles on an ordinary day and that is all it is for —
> confirming the published values are real prices rather than sentinels.
>
> **On a stock split or a reverse split it breaks.** The exchange bands off the
> **adjusted** close; a previous close carried over unadjusted does not. Anything
> deriving the band from it would be wrong by the split ratio — on a 1:10 reverse
> split, wrong by a factor of ten — and wrong on precisely the day the price is
> moving and the band matters most. The reconstruction above is doubly exposed,
> because `NET_CHANGE_1` is itself computed against the adjusted close, so
> last_trade − net_change would not even recover the number it was derived from.
>
> The exchange does this adjustment for us and publishes the result. **Read `xe`
> and `xf`. Never compute them, never validate a published band against a
> computed one, and never fall back to a computed one when the row is missing** —
> a missing row means the last known band still stands (see the stickiness note
> below), not that we should invent one.

**So the band provider is not a sentinel-handling problem. It is a read.** Take
`xe` and `xf` off the ob snapshot. `mm_backtest.Book.snapshot()` already does
exactly this and is the shape to copy — note it updates only when the value is
present (`if ps.limit_up is not None`), so a snapshot that omits the row leaves
the previous band standing rather than clearing it.

The sentinel is an **edge case to guard, not the design**: null `xe` at
999999999.9999, and for `xf` follow the parser's flag rather than a hard-coded
0.01. For any tradeable equity a legitimate lower band cannot be near one tick
anyway — 0.01 as a 10% floor implies a previous close of 0.0111.

**The `xe` sentinel value is correct**: 999999999.9999 matches
`XE_NO_LIMIT_SENTINEL`.

---

## 3. RESOLVED — `at_price_mode` is `"queue"`

Settled 2026-09-16. It is not set in `mm_harness.py` at all; it is in
**`run_legacy_mm.py`**, which the harness imports as `R`:

```python
CFG = dict(
    latency_ms=120,
    at_price_mode="queue",      # realistic queue consumption; "never"/"always" bracket it
    fill_on_crossing_adds=False,
    unfilled_haircut_pct=0.10,
)
```

and `mm_harness.run_symbol_day` builds
`cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED))`.

So at-price fills drain the `ahead` pool. **§1.1 stands in full, with no
caveat**: the exact queue is engaged, and the queue cost of `tol_ticks = 0.0`
is priced into every measured result.

`fill_on_crossing_adds=False` too — the optimistic crossing-add fill is off.

---

## 3a. NEW FINDING, from that same line: the gate cannot be run on the
production latency model

`mm_harness` passes `latency_model=LatencyModel(seed=R.LATENCY_SEED)` with
`LATENCY_SEED = 0`. That **overrides** `latency_ms=120`, so every shipped result
used the STOCHASTIC two-leg model at its defaults: 5 ms decision, 40 ms each
way, 2% of messages taking an exponential tail of mean 400 ms out / 10 ms back.

A seeded RNG makes that reproducible — **but only if the draws happen in the
same order.** They will not.

`_requote` on a price change draws **two** latencies in one cycle: one for the
cancel, one for the replacement, back to back. The production order manager
emits the cancel this cycle and the replacement on a **later** cycle, once the
exchange has acknowledged — because PSX assigns `OrderID` on the ack and the
in-flight rule forbids touching the order before it. So the engine draws one,
then one later, with whatever other messages fall in between.

From the first reprice onward the two runs consume the shared RNG in a
different order, every subsequent latency differs, orders land at different
milliseconds, and fills diverge. **Not because the order manager is wrong —
because the random numbers moved.** Chasing that as a bug would cost a day.

**How to run the gate.** Both sides on a CONSTANT latency model:

```python
LatencyModel(decision_ms=5.0, wire_out_median_ms=40.0, wire_out_tail_ms=0.0,
             wire_in_median_ms=40.0, wire_in_tail_ms=0.0, tail_prob=0.0)
```

`tail_prob=0.0` makes every draw return the same value whatever order it is
taken in (the RNG is still consumed, but the branch never fires), so timing is
deterministic and identical on both sides. Any remaining difference is the
order manager, which is the only thing the gate is meant to measure.

Then a SECOND pass with the production stochastic model on both sides, over
enough days to say whether the engine's P&L sits inside the spread the tails
create. That pass answers a different question — "does the cancel-then-place
delay cost anything under realistic latency?" — and it is a real question,
because the engine waits an ack where `_requote` does not.

**Consequence for the build:** the reconcile gate is two runs, not one, and the
first one is not the production configuration. That has to be said out loud or
someone will read the constant-latency pass as proof the engine reproduces the
shipped numbers.

---

## 3b. BIGGER FINDING: exact reconciliation is impossible, and the reason is a
## real behavioural difference, not the RNG

§3a said a seeded RNG consumed in a different order makes the two runs diverge.
True, and fixable with a constant-latency pass. But checking the latency model
surfaced something that does not go away with any latency configuration.

**The shipped latency model, measured over 20,000 draws** (`LatencyModel(seed=0)`
at its defaults, which is what `mm_harness` passes): decision 5 ms + wire 40 ms
each way, 2% of messages taking an exponential tail of mean 400 ms out. Median
**45.0 ms**, mean **52.4 ms**, **1.69%** above 100 ms, worst draw **2,817 ms**.

**`_requote` sends the cancel and the replacement in the SAME cycle**, two
independent draws, and the replacement does not wait for the cancel. The
production order manager cannot: it emits the cancel, and the replacement waits
for a later cycle once the order is known gone.

So the engine's replacement rests **at least one full latency later** than the
backtest's, and on a quiet name later still, because it waits for the next
market event to trigger the next requote cycle. Different resting times mean
different queue positions mean different fills. **No latency model makes those
two runs agree fill for fill.** The gate has to be P&L within a stated
tolerance, with the difference attributed -- which is what the scope document
already says, and which I had quietly started treating as "identical".

### The fork this opens, and it is a real decision

**PSX's `OrderID` requirement blocks CANCELLING an unacknowledged order. It does
not block PLACING a new one** -- New Order Single carries no `OrderID`. So the
engine *could* emit cancel and place in the same cycle, exactly as `_requote`
does, and reproduce the measured behaviour.

The cost is the thing `README.md` currently claims as a virtue: cancels are
emitted before places so resting size never briefly doubles. Send both at once
and, if the cancel is slow, both rest -- twice the intended size, at two prices,
on a fast market.

**And the backtest does not model that risk at all.** `Backtester._arrive` does
`self.work[o.side] = o` -- a plain overwrite. If the replacement lands before
the cancel, the old order is silently dropped from `work` and can never fill
again; the cancel then finds a non-matching oid and counts
`stale_cancels_ignored`. At most one order per side ever exists in the model. So
the measured results were produced by a simulation in which double-resting is
impossible by construction, and choosing to match `_requote` would take on a
real risk the numbers being matched never priced.

**Neither option is free:**

| | reproduces the measured results | double-size risk |
|---|---|---|
| wait for the cancel (current) | no -- replacement rests later | none |
| cancel + place together | yes | real, and never modelled |

This needs a decision, and it should be made explicitly rather than inherited
from whichever one was written first.

---

## 4. Numbers checked

| Claim | Where | Verdict |
|---|---|---|
| 15.4M PKR | SCOPE §5, §8 | **Verified as a figure, wrong as a word.** Summing each name's assigned-setting P&L in `config_assignment_20260915_0043.csv` gives **15,366,558 PKR**. That is P&L over 197 days, not a "book". §5 then uses it as a capital figure to argue one name at minimum clip risks little — a different quantity, and that inference is unsupported by this number. |
| everyone on lean 0.15 | — | 13,285,941 PKR, so the assignment is **+15.7% in sample**. Consistent with `build_config_assignment.py`'s own warning that the walk-forward figure is +6.41% and the in-sample one overstates it. |
| 113 names, three buckets | SCOPE §8 | **Verified.** 68 / 13 / 17 / 15 across 113 rows. |
| TREC fee 1.554 bps round trip | README, `psx.py` | **Verified.** `FEE_TOTAL_TREC` = 0.000035 + 0.0000065 + 0.0000062 + 0.00003 = 0.0000777 per side = 0.777 bps. A test now pins the venue to it. |
| `queue_skew_thresh_hi` patch unapplied | SCOPE §8 | **Verified.** Zero occurrences in `micro_mm.py`. |
| `tol_ticks = 0.0` | SCOPE §3 | **Effectively verified, loosely worded.** It is `micro_mm`'s constructor default and no config sets it; the doc says "set to 0.0 in the production config", which implies someone chose it. Same outcome. |
| "170 files of research" | SCOPE §0 | **Unverified.** I never counted. |

---

## 5. Verified against the market-data specification

Searched the PSX FIX Market Data Interface Specification directly.

**The phase table in `PSX_SPEC_VERSIONS` §1 is correct.** Every code matches, and
matches `PSX_Parser_Mac.py`'s own `PHASE_MAP` and `BREAK_REASON_MAP`: `S` `O` `T`
`B` `N` `C` `H` `A` `V` `E` on the 0th digit, `1` on the 1st digit meaning
suspended for a whole day, and `2` on the 2nd digit meaning the Friday lunch
break. The spec's own digit numbering is 0-based and the document follows it.

Two nuances the document omits:

- The spec marks `C` (Close Call Auction) **"(reserved)"** — it may never be
  emitted. `psx.py` maps it anyway, which is harmless.
- The 1st digit carries the all-day-suspension flag on the **snapshot**, but is
  **"(reserved)"** on the Trading Session Status message. `psx.py:parse_phase`
  reads it as the suspension flag unconditionally, so a Trading Session Status
  code whose reserved digit happened to be `1` would mark a security suspended.
  Low probability, one-line guard, recorded rather than fixed.

---

## 6. Stale — true when written, not now

| Claim | Where | Now |
|---|---|---|
| "A reprice is now one `ReplaceOrder`" | FIX_COMPLIANCE §1.1 | Reversed deliberately. The default is cancel-plus-new, matching `_requote`; amendment is behind `use_replace=True`. Every measured result was produced under cancel-plus-new, including the fills taken while the old order was still live. |
| Phase 2 = "simulated exchange replaying the parsed store" | SCOPE §5 | Superseded. `sim/replay.py` subclasses `mm_backtest.Backtester` and replaces one method, so the fill model is identical by construction rather than by assertion. |
| "`live_config.py` … done" | SCOPE §8 | Moved into `Production/venues/psx_config.py` with three changes, all marked in the source. |
| "Do you want Phase 0 before I write the order manager?" | SCOPE §7 | The order manager is written, and the question's rationale rested on §1.1. |

---

## 7. Unverifiable in this session

Everything sourced from the **PSX FIX order-entry specification v1.2** — message
types, required tags, Appendix C's prohibited characters, Appendix D, the
certification requirement, the five `Side` values. That document was reviewed
earlier in the session and is not available now.

The *code* claims in `PSX_FIX_COMPLIANCE` were all re-checked against the tree
and hold: every risk control tests `OrderRequest` rather than `PlaceOrder`;
`OrderManager` refuses to construct without an account where the venue requires
one; `OrderState` carries `SUSPENDED` and `PENDING_REPLACE`; `on_cancel_rejected`
exists; the alias map resolves a cancel's reply back to its order;
`format_price` is the only place a `.` is emitted.

What is **not** independently verified is that the specification says what the
document reports it as saying. `PSX_SPEC_VERSIONS` already argues that v1.2 is
very probably superseded, so this should be re-derived against whatever the
broker returns rather than re-read.

---

## 8. What the pattern was

Every error above is the same mistake: **a claim about a file, written without
opening the file.** Not context loss — the queue-position claim was written when
`mm_backtest.py` had simply never been requested.

The rule going forward: read the file before asserting anything about it, and
mark a claim unverified rather than letting it read as established. Documents in
this project are read months later by someone who will not re-derive them.
