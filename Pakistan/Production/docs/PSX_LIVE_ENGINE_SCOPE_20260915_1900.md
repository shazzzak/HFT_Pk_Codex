# PSX live market-making engine — scope

Date: 2026-09-15
Status: **nothing of this exists yet.** 170 files of research, none of which can send an order.
Regulatory input: SECP *Concept Paper — Regulating Algorithmic Trading in Pakistan's Capital
Market*, 30-05-2025 — **not being pursued; taken as engineering guidance, not compliance.**

---

## 0. The regulation is not a constraint, and that changes what this document is

**Revised 2026-09-15 on SZ's information: the SECP framework is not being pursued.** The
leadership that sponsored the concept paper has changed and electronic market making is not a
current priority. There is no registration requirement, no mandatory conformance test, no
algorithm code, no order-to-trade penalty.

So the paper stops being a compliance specification and becomes what it actually is underneath:
**a summary of what eight other regulators learned the hard way about automated trading going
wrong.** Those lessons are worth taking on their own merit — we are the ones whose capital is at
risk — and they are worth taking *in a form that can be pointed at later* if the framework ever
revives.

That produces one design rule for everything below:

> **Build every control because it protects capital. Record the paper's section number against
> it. Never let the regulatory framing drive a design decision.**

The difference is not cosmetic. A control built for a regulator is built to the regulator's
threshold and then forgotten. A control built for ourselves is built to the threshold that
actually protects us, and it gets measured. Two of the controls change materially under this
reading — see §3.

**What genuinely lapses:** registration of algorithms and developers, the mandatory PSX UAT
conformance test, the three SLAs, the annual certified system audit, the unique algorithm code,
and the institutional-investor-only restriction. None of these are now gating anything.

**What does not lapse:** everything in §1, because none of it was ever regulatory. A broker
still has to give us an order-entry session and a specification for it.

---

## 1. The blocker that is not code

We have the **PSX FIX Market Data Interface Specification**. We do **not** have the order-entry
specification. Nothing in Python, C++ or Rust fixes that.

Before a single order can be sent, these must exist, and none of them are engineering tasks:

| # | Item | Why it blocks | Source |
|---|---|---|---|
| 1 | PSX order-entry FIX spec (exact version) | cannot encode an order without it | PSX / broker |
| 2 | Connectivity path: own TREC + direct session, or broker DMA | decides the whole session layer | commercial decision |
| 3 | SenderCompID / TargetCompID, session credentials | cannot log on | PSX / broker |
| 4 | UAT environment access | **still wanted** — see below | PSX / broker |
| 5 | Broker's session message-rate limit, if any | sets the rate ceiling we code to | broker |

All five are operational, not regulatory, and none of them lapse with the framework. They run in
**weeks** and they are the critical path. Start them now.

**Item 4 deserves a note.** The conformance test is no longer mandatory, but UAT access is worth
asking for anyway, and for a better reason than compliance: it is the only place to find out what
the exchange actually does with a malformed order, a cancel for an unknown ID, or a session that
drops mid-order — before finding out with money on. A regulator was going to force us into that
environment; we should go voluntarily.

**Item 5 replaces the regulatory cap.** No regulator is setting an orders-per-second limit, but a
broker or exchange session usually has one, and exceeding it gets orders rejected or the session
throttled. One question to the broker.

**One thing that becomes simpler:** the algorithm code (SECP §4) is no longer issued or required.
The *field* stays in the order encoder anyway — carrying an unused optional tag costs nothing,
and retrofitting a mandatory one across a live order path is disruptive. It is a constructor
argument, defaulted off.

---

## 2. What we build, and why — with the regulator removed from the argument

Each of these survives on its own merit. The § column is bookkeeping for the day the framework
returns, not a justification.

| Control | Why *we* want it | § |
|---|---|---|
| **Kill switch** — stop new orders AND cancel all resting | one action stops everything when something is wrong and we don't yet know what | 11 |
| **Price band** — exchange band *and* a house band around our own mid | the house band is the one that matters: it catches a plausible-looking price computed from a stale or corrupt book, which the exchange band is far too wide to catch | 8.1 |
| **Order value / quantity caps** | bounds the cost of one bad number reaching the wire | 8.2, 8.3 |
| **Position limit on the worst case** | checking the *current* position permits a set of individually-legal orders whose combined fills breach the limit | — |
| **Message rate + burst window** | a strategy that has started oscillating looks acceptable averaged over a second and pathological at 100ms | 7, 8.4 |
| **Trading-window gate** | orders outside continuous trading get rejected or behave in ways the backtest never modelled | 7 |
| **Order-to-trade ratio** | **queue position** — see §3, this one changes meaning entirely | 6 |
| **Audit log** | when a day goes wrong, "why was this order sent" has to be answerable from the record, not reconstructed from memory | 12 |
| **Connectivity fault detection → go flat** | a session that drops while we have orders resting is the single most dangerous state the system can be in | 5 |

### What lapses, and the one piece worth keeping anyway

Registration, conformance testing, the three SLAs, the certified annual audit, and the
institutional-only restriction are all gone.

The one item worth keeping voluntarily is the **stress-test set** (§5): extreme volatility,
volume spikes, multiple algorithms interacting, and connectivity failure. Not because anyone
will ask for the evidence, but because the harness that runs them **is the Phase 2 simulated
exchange** — we are building it regardless to prove the live engine reproduces the backtest.
Running four adverse scenarios through a harness that already exists is nearly free, and
"multiple algorithms interacting" is not hypothetical for us: we intend to run 113 of them.

---

## 3. Quote churn — the item that got MORE important when the regulation went away

With the framework dead, the order-to-trade ratio stops being a compliance number. It does not
stop mattering. It stops being someone else's problem and becomes ours, and the cost moves from
a fine to something worse.

**Every cancel-and-repost surrenders queue priority.** Pull a resting bid and put it back one
paisa away, and we rejoin at the back of the queue at the new price. For a strategy whose entire
measured edge is spread **capture** — which is to say, getting filled *while resting* — queue
position is close to the whole game.

`micro_mm` has `tol_ticks`, a pegging tolerance that exists precisely to hold a quote rather than
repost it. **It is set to 0.0 in the production config.** No hysteresis at all: any recomputation
that moves the desired price by one paisa cancels and reposts.

> **CORRECTED 2026-09-16. The three paragraphs that stood here were wrong.** They said the
> backtest's fill model does not simulate queue position and therefore cannot charge us for
> churn. They were written without opening `mm_backtest.py`.
>
> `MyOrder.ahead` is an order-id → qty dict of everything resting at our price when we arrived,
> built from order-level data and maintained event by event: `_on_market_cancel` removes an order
> that pulls ahead of us, `_on_market_trade` drains the pool (surgically, by `rest_oid`, when the
> trade names its resting victim), and `_on_snapshot_queue_reset` rebuilds it conservatively after
> a snapshot. A two-leg stochastic `LatencyModel` keeps an order fillable until its cancel lands.
> A quote resting ten minutes and one posted a millisecond ago do **not** fill identically.
>
> So the queue cost of `tol_ticks = 0.0` **is already in every measured result**. It also follows
> that the `log_fill_state` sentence below was wrong twice: that switch records `ahead_qty` — the
> queue we wait behind — so it is logging over a queue the engine already tracks, not the other
> side of a missing mechanism.
>
> **Checked and settled 2026-09-16:** `run_legacy_mm.CFG` sets
> `at_price_mode="queue"`, and `mm_harness` builds every run from it. The exact queue is engaged.
> The correction above stands with no caveat. See `PSX_DOC_AUDIT_20260916.md` §3.

The churn ratio is still worth knowing — for the broker's session message limit, which is a real
unknown — but not for the reason this section originally gave.

### How this is handled in the code

**Measure always, enforce on a toggle** (`OrderToTradeRatioCheck(enforce=False)`, the default).
Counting runs from the first day so the number is on the record; the limit blocks nothing until
it is switched on, and the switch is settable at runtime from the hot-reloaded config rather than
needing a restart. A limit that can stop quoting should not be armed before we know what our
normal ratio even is.

The reading travels in the approval's `details`, so every order that goes out records what the
ratio was at the moment it was sent — no separate reporting path.

### The measurement — still worth an hour, for a different reason

**Reframed 2026-09-16.** Not because the backtest is blind to churn (it is not), but because a
broker or exchange session message cap is the one real ceiling that exists and we have not been
told what it is.

The backtest holds every quote the strategy wanted; the ratio of desired quote changes to fills
is computable from data we already have, at no risk and with no new dependency. What the answer
would mean:

- **under ~5:1** — churn is not our problem; note it and move on
- **5:1 to ~20:1** — `tol_ticks` stops being a dormant sweep axis and becomes a live parameter
  whose P&L cost needs measuring
- **above ~20:1** — the quoting cadence needs rethinking before the order manager is written,
  because the OMS diff is where hysteresis lives and it should be designed knowing roughly what
  it has to absorb

The last case is the one that would change the build order, which is why this is worth an hour
before Phase 2 rather than after it.

**One concern that genuinely lapses.** I previously flagged §6's prohibition on *placing orders
without genuine intention to execute*, and said our defensive mechanisms — the OBI widen, the
throttle, the end-of-day cliff — would need documenting so a surveillance team did not read them
as quote withdrawal ahead of flow. With no registration regime, that is gone. The rationales stay
in the research documents where they already are.

## 4. Architecture

The single non-negotiable rule: **the risk gateway sits between the strategy and the wire, and
nothing can go around it.** Not a debug path, not a manual override, not a "just this once"
flag. If an order can reach the exchange without passing the gateway, the kill switch is a
suggestion rather than a switch.

```
   PSX FIX MD  ──▶  [1] Market data handler  ──▶  [2] Book state
                                                       │
                                                       ▼
                                              [3] Strategy  × 113
                                                  (micro_mm)
                                                       │  desired quote state
                                                       ▼
                                          ┌────────────────────────┐
                                          │ [4] RISK GATEWAY       │  ◀── kill switch
                                          │   price / value /      │  ◀── live config
                                          │   volume / burst       │
                                          │   OTR + rate limiter   │
                                          │   position + notional  │
                                          └────────────────────────┘
                                                       │  permitted actions
                                                       ▼
                                          [5] Order manager (OMS)
                                          desired vs actual → min messages
                                          order state machine, ClOrdID,
                                          fills, reconciliation
                                                       │
                                                       ▼
                                          [6] FIX session layer
                                          logon, seq nums, heartbeat,
                                          resend/gap-fill, cancel-on-disconnect
                                                       │
                                                       ▼
                                                   PSX / broker

   everything above writes to  [7] append-only audit log
   [8] control plane: kill switch, config reload, operator console
```

**What exists:** [1], [2] and [3] exist in research form — `PSX_Parser_Mac.py`, the book
reconstruction, and `micro_mm.py` — but [1] reads parsed files, not a live FIX feed, and the
strategy is driven by a backtest loop rather than an event loop.

**What does not exist at all:** [4], [5], [6], [7], [8]. That is the build.

### The design decision that matters most

**The strategy emits a desired quote state; the OMS computes the diff against what is actually
resting and emits the minimum set of messages.** The strategy never sends an order.

Three things fall out of that boundary:

1. **OTR is controlled in one place** — the diff. Hysteresis, tolerance bands and cancel
   suppression all live there, and none of them require touching strategy code.
2. **The backtest and the live engine can be reconciled.** The backtest already computes a
   desired quote state. If the live OMS is fed a recorded day and produces the same fills, the
   two agree. If it does not, we find out in Phase 2 and not with money on.
3. **The kill switch has one thing to do** — tell the OMS that the desired state is "nothing
   resting, no new orders" and let the existing diff machinery cancel everything. A kill switch
   that has its own special-case cancel path is a code path that is never tested until the day
   it matters.

---

## 5. Build order

Each phase is gated: it does not start until the previous one is demonstrably true.

| Phase | What | Gate to pass | Depends on |
|---|---|---|---|
| **0** | Measure quote churn from the existing backtest | a number, with a decision attached | nothing |
| **1** | Risk gateway + kill switch, as pure functions | unit tests incl. every rejection path; kill switch tested from a live-ish state | nothing — **DONE** |
| **2** | OMS state machine + **replay harness**, run TWICE — once on a constant-latency model where both sides are deterministic (the gate), once on the production stochastic model to price the ack wait. See audit §3a: a seeded RNG only reproduces if the draws happen in the same order, and the engine's cancel-then-place-later changes that order — `sim/replay.py`, which subclasses `mm_backtest.Backtester` and replaces `_requote` only, so the fill model is identical by construction (revised 2026-09-16; was "simulated exchange") | engine reproduces backtest P&L on the same days, within a stated tolerance | 1 |
| **3** | FIX session layer against UAT | logon, heartbeat, seq recovery, deliberate disconnect → cancel-on-disconnect verified | UAT access, order-entry spec |
| **4** | The four adverse scenarios through the Phase 2 harness | volatility, volume spike, 113 algos together, connectivity loss — each ends flat, none breaches a limit | 2, 3 |
| **5** | **One name**, minimum clip, live, watched | a full day with no unexplained order, no breach, fills reconcile to the broker's record | 4 |
| **6** | Scale toward 113 names | per-name and aggregate limits hold under real message rates | 5 |

Phase 2 is the one people skip and it is the one that decides whether any of the research
transfers.

Phase 4 was a compliance gate and is now a self-interest gate, which does not make it optional.
"113 algorithms operating simultaneously" is the scenario the SGX list names and the one we are
actually planning, and connectivity loss with orders resting is the most dangerous state the
system can reach. The harness exists from Phase 2 either way; running four scenarios through it
is nearly free.

Phase 5 is deliberately one name: at minimum clip it risks almost nothing and tests every line of
the system. **Corrected 2026-09-16** — this sentence previously read "the measured book is 15.4M
PKR across 113 names", which used a P&L figure as though it were capital. The capital at risk in
Phase 5 is one clip on one name, and it has nothing to do with that number.

---

## 6. Python now, Rust later — where the boundary goes

You said Python first, then C++ or Rust. Here are both paths honestly.

### The staged path (Python first) — and why it is *not* merely a shortcut here

The usual reason to write the matching path in Rust is latency. **That reason is weaker on PSX
than almost anywhere**, and by your own observation: PSX is human-traded, there are no
competing bots, and our edge is measured as spread **capture**, not as winning a race. Nothing
in the research says we lose money by being a millisecond late.

So Python for the strategy is defensible on the merits, not just as a learning step.

Where Python is genuinely uncomfortable is **[4] the risk gateway and [6] the session layer**,
and the problem is not throughput — it is **determinism**. A garbage-collection pause at the
wrong moment delays a cancel, and the moment you care about cancels is the moment the market is
moving fast. Those two components have a worst-case requirement, not an average-case one.

### The production path

Rust for [4] risk gateway, [5] OMS and [6] FIX session from day one — no GC, no pause,
exhaustive state machines checked by the compiler, and a `Result` type that makes an unhandled
reject a compile error rather than a silent drop. Strategy either also in Rust, or in Python
across a process boundary with the risk gateway on the Rust side of it.

### What I recommend

Build the whole thing in Python **with the boundary at [4]/[5]/[6] made hard from the first
commit** — separate processes or at minimum separate modules with a narrow, typed, byte-level
interface and no shared mutable state. Then port those three to Rust after Phase 5, when their
behaviour is pinned by a test suite that already passes.

That gets a working system sooner, and the port is mechanical rather than a rewrite. The cost of
being wrong about this is one port; the cost of building a Rust FIX engine against a
specification we have not read yet is months spent on the wrong abstraction.

**The honest risk in my recommendation:** teams say "we'll port it later" and never do, and a
Python risk gateway becomes permanent by inertia. The defence is the Phase-5 gate — porting is
not optional after one name goes live, it is the condition for scaling past it.

---

## 7. What I need from you

1. **Order-entry spec and connectivity path** — own TREC with a direct session, or a broker's
   DMA? This changes [6] entirely and it is the longest-lead item by a wide margin.
2. **Does the broker or the exchange cap session message rate?** One question, and it sets the
   number the rate limiter is coded to. With no regulator setting one, this is the only real
   ceiling that exists.
3. ~~**Do you want Phase 0 (the churn measurement) before I write the order manager?**~~
   **Overtaken 2026-09-16.** The order manager is written, and the urgency behind this question
   rested on the queue-position error corrected in §3. What replaces it: `mm_harness.py`'s
   `at_price_mode` (audit §3) and the broker's session message cap (item 2 above).

---

## 8. Carried over, not forgotten

- **`micro_mm.py` `queue_skew_thresh_hi` patch** — still unapplied. It only matters if we return
  to the extreme-OBI question, which I have recommended stopping. Parked, not lost.
- **`log_fill_state` (item C.18)** — **reason corrected 2026-09-16.** It is not a missing piece
  of the fill model; it logs `ahead_qty` over a queue the engine already tracks. What stands is
  the second half: Phase 2 reconciles the engine against the backtest and cannot tell you whether
  the backtest is right. `log_fill_state` produces every posted quote, filled or not, with the
  circumstances it was posted into — the dataset that would answer that. Still wanted, for that.
- **The shipped config assignment** (three buckets, 68 / 13 / 17 / 15 across 113 names) and
  **`live_config.py`** are done. `live_config.py` has since moved into
  `Production/venues/psx_config.py` with three changes, all marked in that source.
  **15.4M PKR corrected 2026-09-16:** the figure verifies (15,366,558, summing each name's
  assigned-setting P&L) but it is **P&L over 197 days, not a book** — see the audit, §4.
