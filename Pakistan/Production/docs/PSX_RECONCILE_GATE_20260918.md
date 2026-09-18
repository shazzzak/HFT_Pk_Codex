# PSX — the reconcile gate passes, and the two venue rules it exposed

**2026-09-18.** Final run: `gate_replace_20260918_1115.csv`, 12 names × 20 dates
(2026-06-01 → 2026-06-30), 240 symbol-days, four engine runs each.

Supersedes the 02:06 run. That one also passed, but under two models of PSX
that were wrong — see §3.

---

## 1. The result

The production engine reproduces the research backtest exactly.

| | |
|---|---|
| exact matches | **240 of 240** |
| total, backtest | 30,703.16 PKR |
| total, engine | 30,703.16 PKR |
| total difference | **0.00 PKR** |

Checked against the CSV rather than the printed summary:

- days with any P&L difference: **0**
- days where fill counts differ: **0**
- days where cancel counts differ: **0**
- days where order counts differ: **0**

That last line is new. The 02:06 run had one day out — THCCL 2026-06-05, one
extra message. It is now zero, and the reason it was ever non-zero turned out
to be a missing venue rule rather than a reconciliation quirk (§3.7).

Reconciliation is exact at the **record** level, not just the paisa. Every
order-disposal column matches between the two runs:

| | records | cancelled | amended | filled | crossed | open | in flight | unlabelled |
|---|---|---|---|---|---|---|---|---|
| backtest | 193,936 | 106,694 | 75,229 | 10,693 | 1,247 | 61 | 12 | 0 |
| engine | 193,936 | 106,694 | 75,229 | 10,693 | 1,247 | 61 | 12 | 0 |

The columns sum to the record count exactly, on all four runs. There is no
remainder any more — see §5.

This was step 5 of the live-engine plan: *"does the engine reproduce the
backtest? If it doesn't line up, nothing downstream is worth building."* It
lines up.

---

## 2. Five defects in the engines

Each number measured, not estimated.

### 2.1 Duplicate orders fired into the engine's own latency window
*(`mm_backtest.py`)*

`self.work[side]` recorded an order when it **landed**, not when it was
**sent**. For one network latency after every send the side read as empty, and
the next requote — event-driven, able to fire many times in that window — sent
another order.

- **4,901** duplicate sends across four symbol-days
- **5,205** resting orders overwritten: never cancelled, never filled, never
  closed out in the lifecycle log

On MLCF 2026-06-24 the backtester sent 1,756 new orders when only about 265
orders had ever left a side. That arithmetic exposed it.

**Fix:** the side is reserved at send time, with `t_active = None` marking the
order as on the wire. Every fill path tests for it and skips.

### 2.2 A cancel racing its own replacement
*(`mm_backtest.py`)*

Narrower than it first appeared. A normal reprice under a Cancel/Replace is one
message with nothing to race. The race happened only when an amendment was
**already** on the wire and the strategy wanted a different price again: a
second amendment cannot be stacked, so the code fell through to the old
cancel-plus-new path and fired both in one cycle with independent latency draws.

**Fix:** the side is held while **any** message is outstanding.

### 2.3 A crash in the live order manager
*(`core/model.py`, `core/oms.py`)*

`InvalidTransition('order GATE-00001068 (NRL SELL) is PENDING_CANCEL; cannot
apply replaced')` — killed **2 of every 6** symbol-days, in production code that
would raise mid-session on the most active name on the book.

An amendment goes out, so the order is `PENDING_REPLACE`. The old terms are
still resting and still matchable — the whole point of a one-message reprice. A
fill arrives on them and `on_fill` overwrites `state`, destroying the only
record that a message was on the wire. The manager then reads a quiescent order
and sends a cancel; when the amendment confirms, the transition raises.

**Fix:** `state` carries the lifecycle; `replace_in_flight` and
`cancel_in_flight` carry the messages.

### 2.4 The simulator could not hold two orders on one side
*(`mm_backtest.py`)*

`Backtester.work` was `dict[side] → ONE MyOrder`. A policy that legitimately
rests two orders had its second **overwrite** the first — including the very
order whose queue position the policy exists to protect.

Traced on the real harness, 200 cycles with the price walking:

| policy | manager approved | exchange accepted | resting: manager / exchange |
|---|---|---|---|
| exact | 101 | 101 | 1 / 1 |
| reduce_only | 101 | 101 | 1 / 1 |
| queue_preserving | 172 | 59 | **1 / 0** |

The last row is the manager believing it has a quote resting while the exchange
has nothing.

**Fix:** `work` is a list per side; the fill engine walks our orders in the
exchange's own priority.

### 2.5 The acknowledgement wait the engine never read

After a **cancel**, `mm_backtest` holds that side until the exchange's reply
returns. `sim/replay.py` had been drawing that latency into `ack_until` since it
was written and never reading it back. On PPL 2026-06-19 the backtest waited
1,465 times and the engine none.

**Fix:** the engine honours the wait it was already computing.

---

## 3. Two defects in the model of PSX itself

These are different in kind from §2. Nothing was inconsistent between the two
engines — both agreed, and both were wrong about the exchange.

### 3.6 A post-only order type PSX does not have

A quote that leaves passive can **arrive** marketable, because the book moves
during the latency window. Both engines threw those orders away and counted
them as `rejected_crossing`.

That is post-only behaviour. Checked against the documents rather than assumed:

| source | says |
|---|---|
| Regulations **8.5.1** | every order type PSX accepts: Limit, Market, Market-to-Limit, CFO, CXL |
| Regulations **8.9(b)** | every time-in-force term: GTD, FOK, IOC |
| Regulations **8.4.2** | *"Orders that cannot be immediately executed shall be queued for future execution."* |
| Regulations **8.4.4** | *"In case an Order is executed partly, the remaining part of such Order shall not lose its priority."* |
| FIX Market Data spec v1.05 | `ExecInst(18)` exists with exactly one value, `'B' = Ok to Cross`, on an outbound execution report |

No post-only instruction exists, so it cannot be asked for. 8.4.2 leaves two
outcomes and not three: an order that can execute immediately does.

**The old behaviour deleted trades, and not neutral ones.** A bid only becomes
marketable if the offer *fell* to meet it — so these were buys into a falling
market and sells into a rising one. Adverse fills. Dropping them flattered P&L.
A simplification that errs toward profit is the worst kind to leave in.

**Fix:** `Backtester._cross_on_arrival`. Consumes the resting book best-first,
**only while the level price is within our own limit**, filling at the level
price; the remainder rests under 8.4.4. Deliberately *not* `_taker_fill`, which
is a sweep: it pays through every level and drops the remainder, which is right
for a flatten and wrong for a limit order. `sim/replay.py` now reports those
shares to the order manager, or its position would drift from the exchange's.

Measured over the run: **1,284 orders, 58,648 shares** — identical on both
sides, as they must be on one exchange with one latency seed.

Reachable as `cfg["cross_on_arrival"]=False` for reproducing an older figure.
That is the old bug, not a cautious setting.

### 3.7 The engine was quoting with no price band

`mm_backtest._requote` clamps every desired price into `[limit_dn, limit_up]`
before sending. The engine has the same clamp — `MicroMMAdapter._clamp_to_band`
— and a `PriceBandCheck` besides, both written and tested.

**`sim/gate.py` built `PSXVenue` with no `band_provider`**, so
`price_band()` returned `None` for every symbol. The clamp clamped to nothing,
and the adapter also set `micro_mm.limit_up/limit_dn` to `None` from the same
call. The machinery was all there, wired to a dead input.

THCCL 2026-06-05 is the whole visible symptom. At 1780652342851 the strategy
asked for an ask of **66.89 in both runs** — verified by recording all 30,095
`quotes()` calls in each run and finding no answer differs anywhere. The
backtest clamped to the published **66.14**; the engine sent 66.89.
`66.14 / 1.10 = 60.13`, a 10% band off the previous close.

Live, PSX rejects 66.89 and that side stops quoting. It surfaced as one extra
message on one day only because the band rarely binds — and the money matched
only because both prices sat above a market trading 65.6–66.0, so neither
traded. Luck, not a passing grade.

**Fix:** `gate.py` passes a `band_provider` reading `limit_up`/`limit_dn` off
the exchange's own Book. It **reads** the band and never computes one:
`PriceBand`'s docstring is explicit that on a split the exchange bands off the
*adjusted* close, so a ±10% reconstruction is wrong by the split ratio on
precisely the day it matters.

### 3.8 The backstop is now installed

`PriceBandCheck` — the gateway control that refuses an order priced outside the
published band — existed, was tested, and was not in the gate's gateway. It
could not be: its house band reads `ctx.reference_price_minor`, which comes
from `reference_prices` on the order manager's `reconcile()` call, and
`sim/replay.py` passed none. A missing reference is a **rejection**, not a pass,
by design — quoting a symbol whose own mid cannot be computed means the book is
not trustworthy — so installing it would have refused every order.

**Fixed:** `replay.py` now passes the live mid as the reference, and `gate.py`
installs the check. The mid rather than the day's opening trade, because the
house band is a tolerance around where the market *is*; an opening price anchors
it somewhere the market may have left hours ago.

Verified rather than assumed: a one-sided book returns `None` from the
book-builder (`replay.py:129`), so the strategy wants nothing, only cancels are
emitted, and `PriceBandCheck` passes anything that is not an order request.
Every book that can produce a quote is two-sided, so the reference is never
absent when it matters.

**It must never fire.** The clamp runs first and pulls every price inside the
band, so a rejection means the band moved between the clamp and the gateway, or
the clamp was bypassed. `gateway_rejections` is already a must-be-zero column.
On the 6-day smoke the output is bit-identical to the run without the check —
same P&L, same order counts, same disposal table — which is the evidence it
rejected nothing.

**The house band here is 25% and is not a live value.** `PriceBandCheck`
enforces the tighter of the exchange band and the house band; in the gate the
exchange band is the one that should bind, so the house figure is set well
outside anything micro_mm can produce. Live it should be much tighter: the
house band is the control that catches a plausible-looking price computed from
a stale book, and the exchange band at +/-10% is far too wide to do that.

---

## 4. What the two model fixes cost

Comparing the same 240 symbol-days before and after:

| | |
|---|---|
| before (02:06 run) | 30,785.24 PKR |
| after (11:15 run) | 30,703.16 PKR |
| net | **−82.08 PKR, −0.27%** |

**In aggregate: nothing.** But that near-zero total hides real movement.

| | |
|---|---|
| days that moved at all | **224 of 240** |
| moved down | 103 days, −7,646.40 PKR |
| moved up | 121 days, +7,564.32 PKR |
| sd of the per-day change | **106.28 PKR** |
| t on the change | −0.05 |

The per-day standard deviation is comparable to the mean daily P&L of about
128 PKR. Individual days moved by up to ±524. **The corrections are material
per day and immaterial in total.**

That matters for anything selected per name:

| | old | new | moved | t old | t new | rank |
|---|---|---|---|---|---|---|
| NRL | 4,019.78 | 3,021.46 | **−998.32** | 2.95 | 2.42 | 3 → 5 |
| PPL | 2,427.12 | 3,040.44 | +613.32 | 2.25 | 2.63 | 6 → 4 |
| ENGROH | 6,772.14 | 7,355.11 | +582.97 | 4.73 | 4.77 | 1 → 1 |
| THCCL | −178.35 | −704.59 | −526.25 | −0.17 | −0.52 | 12 → 12 |
| MLCF | 1,643.26 | 2,066.70 | +423.44 | 2.21 | 2.50 | 8 → 7 |
| NCPL | 846.76 | 892.65 | +45.89 | 2.16 | **1.92** | 11 → 11 |

**6 of 12 names changed rank. No name changed sign. NCPL crossed below t = 2**,
a bar it previously passed.

**The aggregate is safe; per-name and per-day figures are not.** Anything
derived from them — the tradeability gate, the config assignment, per-name
rankings, capacity exclusions — was computed on the old basis and needs
re-deriving before it decides a deploy.

A six-day sample taken mid-way through this work showed those days going from
+184 to −156 and looked alarming. It was noise. The full run is the answer.

---

## 5. Order accounting now balances to zero

The disposal table used to carry a "remainder" of about 12,000 records per
policy that pooled four different outcomes, two of which had no label at all:
`rejected_crossing` never wrote an end reason, and neither did an order still
resting at the close. Both came out as blank and were indistinguishable.

Every record now closes with exactly one reason, and `gate.py` checks the
arithmetic rather than asserting it:

| policy | records | cancelled | amended | filled | crossed | refused | open | in flight | unlabelled |
|---|---|---|---|---|---|---|---|---|---|
| amend up | 193,936 | 106,694 | 75,229 | 10,693 | 1,247 | 0 | 61 | 12 | 0 |
| second order | 191,957 | 112,997 | 66,569 | 11,057 | 1,241 | 0 | 78 | 15 | 0 |
| don't top up | 185,460 | 106,723 | 66,687 | 10,736 | 1,242 | 0 | 59 | 13 | 0 |

`refused` is 0 because the post-only path is off. `unlabelled` is 0 because
nothing escaped a reason. One record per order **generation**, not per distinct
order: an amendment closes the record it amends and opens a new one.

---

## 6. What queue position is worth on this strategy: nothing

Three quoting policies, same days, same exchange, same latency seed. The clip is
500; a partial fill takes 400 and leaves 100 resting with the place it earned.

1. **amend up** (`exact`) — amend the 100 back to 500. PSX 8.5.2 sends the whole
   order to the back of the queue. mm_backtest's rule, and the gate's baseline.
2. **second order** (`queue_preserving`) — leave the 100, send a separate order
   for the 400. Priority paid only on the increment.
3. **don't top up** (`reduce_only`) — send nothing. Show 100 until someone hits
   it. The cost is the size not working.

| policy | total P&L | vs amend up | better on | mean | se | t |
|---|---|---|---|---|---|---|
| amend up | 30,703.16 | — | — | — | — | — |
| second order | 30,688.36 | −14.81 | 55 of 240 | −0.06 | 2.34 | **−0.03** |
| don't top up | 30,042.98 | −660.18 | 74 of 240 | −2.75 | 3.00 | **−0.92** |

**Neither clears |t| > 2, and the conclusion is now stronger than it was.**
On the old basis the second-order policy looked worth +562.63 (t 0.85), which
was at least suggestive. On the corrected basis it is worth **−14.81, t −0.03** —
indistinguishable from zero on 240 symbol-days.

The reason is in the amendment breakdown:

| policy | price move | size up | size down | kept place |
|---|---|---|---|---|
| amend up | 62,918 | **6,763** | 5,548 | 5,548 |
| second order | 62,831 | 1 | 3,737 | 3,737 |
| don't top up | 62,981 | 0 | 3,706 | 3,706 |

Of 75,229 landed amendments on the baseline, **69,681 — 92.6% — lose their
place**, and 62,918 of those (83.6% of all landed amendments) are price moves,
which re-queue under 8.5.2 regardless of policy. This strategy reprices on
nearly every tick, so it almost never reaches the decision the three policies
disagree about.

Queue position is not worthless in general. It is worth nothing **to this
strategy**, because the strategy surrenders it constantly for a different
reason. Changing that means widening the requote tolerance (`price_ticks`), not
changing the size policy.

Three invariants that came out clean and are worth keeping:

- `kept place` equals `size down` exactly on all three policies. On PSX only a
  size reduction is applied in place, so those two must agree.
- `orders rested alongside another on the same side`: 6,802 for the second-order
  policy, **0** for both single-order policies.
- `crossed_on_arrival` is identical on both sides, 1,284 orders and 58,648
  shares. One exchange, one seed — a difference here would be a divergence in
  the exchange itself and would outrank everything else in the run.

---

## 7. Still open

**The full run has not been repeated with `PriceBandCheck` installed.** The
6-day smoke is bit-identical with and without it, so it rejects nothing there.
Only the 240-day run proves it rejects nothing anywhere, and
`gateway_rejections` is the column that says so.

**The live house band is unset.** 25% is a gate value, chosen so the exchange
band binds. Picking the live figure is a real decision and nobody has made it.

**`engine_blocked_in_flight` is a dead column.** The counter lives in
`Backtester._requote`, which the engine replaces, so it can only read zero.
Printed beside a real backtest number it invites the conclusion that the engine
never waits, which is false — its equivalent is `engine_held_inflight`
(781,820 over this run). To be removed.

**`kept place` is a misleading column name.** It counts only amendments that
were size reductions. It does not count an order that kept its place by having
nothing sent for it — which is exactly what the second-order policy does, so
the column understates the policy it was meant to show off. The column that
answers "how often did this policy give away its place" is `size up`.

**Shadow fills can leave a visibly crossed book.** When a crossing order fills
partly, the remainder rests at our limit while the historical book still shows
the level we just consumed. Same assumption every other fill in this engine
makes; crossing is the first place it is visible. No P&L effect — those shares
are already paid for — and left alone deliberately rather than changing the
fill model under a passing gate.

---

## 8. Already covered elsewhere, recorded so it is not rebuilt

**The split detector exists.** `sim/check_data_quality.py`, `check_bands()`,
lines 293-353. It reads the published `xe`/`xf` limits and `prev_close`,
computes the flat +/-10% band, and flags any symbol off by more than a paisa as
`band_vs_prev_close`, keeping the offending rows rather than only a count. It
also checks `quote_outside_band`, which **is** in the `FAULTS` list.

`band_vs_prev_close` is deliberately kept OUT of `FAULTS` (line 993) and
reported through the examples section instead. That is correct: the exchange
bands off the ADJUSTED close, so a split legitimately breaks the +/-10%
identity. A mismatch is "look at this symbol", never "the data is wrong", and
the file never substitutes its own arithmetic for the published value.

---

## 9. What this unlocks

Steps 1–4 of the live-engine plan (order manager, audit log, strategy adapter,
simulated exchange) were already built. Step 5 was the gate. It passes, and it
passes against a model of PSX that now matches the rulebook on the two points
where it previously did not.

Outside the engine and unchanged by this work: the order-entry specification and
the connectivity decision — own TREC or broker DMA.
