# PSX market-making BACKTEST — complete mechanism specification

**Written 2026-09-18. Three passes; Part III (§37) is the correction register.**
This describes the **research backtest only** — the
simulated exchange (`mm_backtest.py`), the strategy (`micro_mm.py`), the
calibration layer (`mm_harness.py`) and the sweep runner (`universe_expand.py`).
It deliberately does **not** describe the production live engine in
`Production/` (order manager, risk gateway, FIX adapter); that is a separate
system whose only relationship to this one is that it must reproduce it.

## 0. Provenance — what this was checked against

Read in full and quoted from:

| file | md5 | lines |
|---|---|---|
| `mm_backtest.py` | `55b239ed60d3e5d914cedf768780aa5c` | 3,029 |
| `micro_mm.py` | `362f8a5b429dd55ff292abef706345e2` | 1,994 |
| `mm_harness.py` | `f273a49209147bfcf2694fc6ef562ca4` | 1,010 |
| `universe_expand.py` | `3c529ec8c5d2642cedff4126877d9ef1` | 1,433 |
| `build_config_assignment.py` | (attached 2026-09-18) | 514 |

Also read: `Production/README.md`, `Production/docs/PSX_DOC_AUDIT_20260916.md`,
`PSX_LIVE_ENGINE_SCOPE_20260915_1900.md`, `PSX_SPEC_VERSIONS_20260916.md`,
`PSX_FIX_COMPLIANCE_20260916.md`, `PSX_RECONCILE_GATE_20260918.md`, and the
project to-do `PSX_HFT_TODO_20260915_1245.md`.

**Not readable at the time Part I was written**, so nothing in §1–§23 depends on
them: `run_legacy_mm.py`, `config_pk.py`, `spot_capture_markout_decomp.py`,
`expansion_names.py`, and the generators of the four calibration CSVs.
`/Users/shazzak/PycharmProjects/HFT/Pakistan/Docs` is not a folder I can reach,
so anything only written there is not reflected here.

**PART II (§24–§35, plus the §36 index) was written after eleven more files were supplied and read
in full.** `run_legacy_mm.py` and `spot_capture_markout_decomp.py` moved from
that gap list into §24 and §26.

**PART III (§37) was written after three more — `config_pk.py`,
`build_feature_store.py` and `leadlag_screen.py` — plus the lead-lag screen's
own output file.** It closes three §34 gaps and **corrects two statements made
earlier in this document**; the corrections are set out in §37.4 rather than
edited away. The files still unread are listed, one per row with what would
close each, in **§34**.

Where the code and a document disagree, **the code wins and I say so**.
Where I have not read the file, it goes in §34 and I do not describe it.

**How to read this document.** Part I (§1–§23) is the backtest as a machine:
clocks, latency, fees, book, fills, venue rules, strategy, skews, time
structure, calibration, outputs. Part II (§24–§35) is everything measured
around it: the canonical parameter sets, the event stream, the P&L
decomposition, the imbalance family and the conditional-markout finding,
persistence and order collapsing, icebergs, lead-lag, hedging, corporate
actions, the screening funnel, the open gaps, and the lessons. §36 is a
single-page index of every number in the document with its source.

---

## 1. THE SHAPE OF THE WHOLE THING

One run = one `(date, symbol, arm)` cell. For each cell:

1. Open the day's parsed data for that symbol: order-book updates, order-book
   snapshots, and trades.
2. Merge them into one event stream, ordered by exchange time.
3. Build a `MicrostructureMM` strategy object with per-name calibrated
   parameters and a per-name clip size.
4. Build a `Backtester` — the simulated exchange — with a seeded latency model.
5. Replay the day. The exchange reconstructs the real book event by event; the
   strategy is asked where it wants to quote; orders travel with latency, rest
   in a modelled queue, and are filled by the real historical trade flow.
6. At the close, flatten whatever is left against the visible book.
7. Emit fills, an equity curve and a stats dictionary.

The sweep runner does this across ~197 dates × 113 names × N arms in a process
pool, decomposes every fill into capture / markout / fees by FIFO round trip,
and writes three files: a per-config CSV, a per-config-day-bucket parquet, and a
per-config-day-**name**-bucket parquet.

---

## 2. TWO CLOCKS, AND THE ANTI-LOOKAHEAD RULE

This is the most important structural idea in the backtest. Every row in the
parsed store carries two timestamps:

- **`ts_exch`** — the exchange's own clock. For updates and trades this is
  `transact_time`; for snapshots it is `orig_time` (second precision only).
- **`ts_cap`** — `capture_ts`, the moment our capture process saw the message.

The book, and every fill decision, run on **`ts_exch`**. What the strategy is
allowed to *see* runs on knowledge time:

```python
know = max(know, int(obj.ts_cap))     # running maximum, never rewinds
```

The running maximum is deliberate — receive timestamps jitter, and knowledge
must not go backwards. `_requote(know)` is the only consumer.

Our own messages cross between the clocks: a decision taken at knowledge time
`k` lands on the exchange timeline at `k + draw_out()`.

**Known asymmetry, flagged in the code itself:** there is no reverse conversion.
An exchange-time fill updates `self.pos` immediately, and the strategy reads the
new position on the next requote with no acknowledgement delay. Real fills
arrive on an execution report one inbound hop later. The code flags this at
lines 1991–1994 as a known distortion that has not been corrected.

---

## 3. THE LATENCY MODEL

`class LatencyModel`, two independent legs, drawn per message.

```python
def __init__(self, decision_ms=5.0, wire_out_median_ms=40.0,
             wire_out_tail_ms=400.0, wire_in_median_ms=40.0,
             wire_in_tail_ms=10.0, tail_prob=0.02, seed=0):
```

| leg | method | formula |
|---|---|---|
| out (us → exchange) | `draw_out()` | `decision_ms + wire_out_median_ms`, plus with probability `tail_prob` an `Exponential(mean=wire_out_tail_ms)` spike |
| in / ack (exchange → us) | `draw_ack()` | `wire_in_median_ms`, plus with probability `tail_prob` an `Exponential(mean=wire_in_tail_ms)` spike. No `decision_ms` — an ack is passive |

So the normal case is 45 ms out and 40 ms back, with 2% of messages taking an
extra exponential spike averaging 400 ms outbound / 10 ms inbound.

- The out leg applies to **new orders, amendments and cancel requests** alike.
- `rng = np.random.default_rng(seed)`. **Same seed → identical draw sequence.**
  This is what makes two runs comparable, and it is also the property that makes
  the reconcile gate meaningful: if two runs send the same messages in the same
  order they consume the same draws, and any divergence in message sequence
  makes everything downstream diverge.
- Set `tail_prob=0` and the tail means to 0 to recover a constant-latency run.
- **The tail parameters are priors for PSX-remote access, not measurements.**
  The docstring says so explicitly and says to refit from colo telemetry.

---

## 4. FEES

```python
FEE_PSX_LAGA_PCT   = 0.000035    # PSX trading fee, PKR 3.50 / 100,000
FEE_SECP_PCT       = 0.0000065   # SECP supervisory
FEE_IPF_PCT        = 0.0000062   # PSX regulatory (IPF)
FEE_CLEARING_PCT   = 0.00003     # NCCPL + CDC
FEE_MM_REBATE_PCT  = 0.0         # negative when a rebate is known
FEE_PER_SHARE_FLAT = 0.0

FEE_TOTAL_TREC = LAGA + SECP + IPF + CLEARING + REBATE      # 0.0000777
USE_TREC_FEE   = True
FEE_TOTAL_PCT  = FEE_TOTAL_TREC

def fee_for(price, qty):
    return FEE_TOTAL_PCT * price * qty + FEE_PER_SHARE_FLAT * qty
```

**0.777 bps per side, 1.554 bps round trip.** That is the *own-TREC* number:
broker commission (0.15%) and its 13% sales tax are zero because you are your
own broker. Flipping `USE_TREC_FEE = False` restores the retail schedule at
~17.73 bps per side, which no market-making edge survives.

Assumptions recorded in the code and worth re-checking before deploy: clearing
is set at the low end (0.003%) on the assumption that an intraday-flat MM has
CDC delivery waived; IPF was scheduled to end Aug 2025 and is kept as the
conservative default; SST is 13% Sindh, and the broker's province is unconfirmed.
CVT and WHT on turnover are abolished and deliberately absent.

Every fill charges `fee_for` on both legs. The one exception is the end-of-day
**residual mark** (§9, item 3), which is a mark and not a trade, so it pays no fee.

---

## 5. THE BOOK

`class Book` reconstructs the real order book from the parsed feed.

- `self.o` is a dict of `{order_id: Order(side, price, qty)}` — the historical
  resting orders, not ours.
- Aggregated deep levels arrive with ids prefixed `__AGG_` and are skipped by
  imbalance/depth calculations unless `include_deep=True`.
- `bbo()` returns `(best_bid, bid_qty, best_ask, ask_qty)`.
- `qty_at(side, price)` returns `{order_id: qty}` of everything resting at that
  price — this is what our queue position is measured against.
- `ranked_depth(n=10)` returns `(bids, asks)` as `(price, qty)` lists, bids
  high→low and asks low→high, aggregating net quantity per price and dropping
  levels that net to ≤ 0.
- `obi()` is the level-aggregated imbalance.
- `phase` carries the exchange's own trading phase.
- `limit_up` / `limit_dn` come from the feed's `UPPER_CIRCUIT_BREAKER` /
  `LOWER_CIRCUIT_BREAKER` snapshot rows. **They are read, never computed.** A
  ±10% reconstruction would be wrong by the split ratio on exactly the day it
  matters, because the exchange bands off the *adjusted* previous close.
- `pinned()` is true when `best_bid >= limit_up` or `best_ask <= limit_dn`.

`TICK = 0.01` PKR across PSX; `round_tick()` snaps to it.

---

## 6. OUR ORDERS AND THE QUOTING CYCLE

### 6.1 `MyOrder` and `work`

```python
self.work: dict[str, list] = {"BUY": [], "SELL": []}
```

A **list per side**, ordered by send time, oldest first. It held one order per
side until 2026-09-17; that was a defect, because a policy that legitimately
rests two orders on a side had its second order overwrite its first.

`MyOrder` carries side, price, qty, `ahead` (the queue snapshot), `t_active`,
`oid`, `cancel_at`, `amend_at`, `taker`.

**`t_active = None` means "sent but not yet landed".** The side is reserved at
**send** time, not at land time. Before this fix the side read as empty for one
network latency after every send, and the event-driven requote fired again into
that window — 4,901 duplicate sends and 5,205 silently orphaned orders across
four symbol-days.

Accessors: `_side_orders(side)`, `_all_orders()`, `_order_by_oid(side, oid)`,
`_drop_order(side, o)` (by identity, not value), `_lead(side)` (the oldest).

### 6.2 The requote gate

`_requote(ts_know)` is called on every event while `t0 <= ts_exch <= t1`. It
refuses to quote when any of these hold, cancelling everything resting first:

1. **Phase** is not `CONTINUOUS_AUCTION` (and not `None`) — counted as
   `halted_requotes`.
2. **`book.pinned()`** — the touch is at or through a circuit band.
3. **Stale feed** — `stale_feed_requotes`.
4. **Crossed book** — `skip_crossed_book` (on by default): best bid at or above
   best ask cannot exist at the exchange, so quoting off that mid means pricing
   off a mid that is not a mid. Counted as `crossed_book_requotes`. The cause of
   the crossed snapshots has never been established; this flag stops the engine
   trading on them, it does not diagnose them.

It also holds a side while **any** message is outstanding on it
(`t_active is None or amend_at is not None`), and — after a genuine **cancel**
— until the acknowledgement would have returned (§8.2).

### 6.3 The band clamp

After the strategy returns its desired prices, the backtest clamps each one into
the published band and re-rounds:

```python
if self.book.limit_up is not None: px = min(px, self.book.limit_up)
if self.book.limit_dn is not None: px = max(px, self.book.limit_dn)
want[side] = (round(px, 2), w[1])
```

This is a strategy-side self-cap: quote at the band edge rather than have the
exchange reject the order. It is the mechanism the production engine was missing
until 2026-09-18.

---

## 7. THE FILL MODEL — how our resting orders get hit

This is the heart of the backtest and the place where a market-making simulation
is most easily flattered.

### 7.1 The queue snapshot

When an order **lands** (`_arrive`), it takes a snapshot of everything already
resting at its own price:

```python
o.ahead = self.book.qty_at(o.side, o.price)     # {order_id: qty}
o.t_active = t
```

Under price-time priority every one of those shares is in front of us. The dict
is then maintained incrementally by the market-event handlers: when one of those
historical orders cancels or trades, its entry shrinks or disappears.

### 7.2 What happens when a historical trade prints

`_on_market_trade(r)` asks: could this aggressor have hit us?

Eligibility gates, in order:
- **Auction prints are skipped** — there is no continuous-market aggressor.
- We must have a working order on the **passive** side (a buy is hit by a seller).
- The order must be live (`t_active` is not `None`).

Then the engine groups our live orders by price, sorts them price-then-time
(the exchange's own priority), and walks levels best-first consuming the
aggressor's quantity. At each price:

1. The historical queue in front (`ours[0].ahead`) is drained **once** for that
   price level — not once per order of ours — so the queue ahead cannot be
   consumed twice by two of our own orders sitting at the same price.
2. Whatever survives is offered to our orders **oldest first**.

### 7.3 The fill reasons

Every fill row carries a `reason` tag so P&L can be attributed to the rule that
produced it:

| reason | meaning |
|---|---|
| `through` | the trade printed **through** our price — certain fill by price priority |
| `at_queue` | the trade printed **at** our price and our queue had drained |
| `at_optimistic` | at our price under the optimistic `at_price_mode` |
| `crossing_add` | filled by a crossing add (`fill_on_crossing_adds`, off in production) |
| `taker` | a deliberately tagged taker order swept the book (`allow_taker`) |
| `crossed_on_arrival` | our passive quote arrived marketable and traded (§8.3) |
| `liq` | end-of-day liquidation against the visible book |
| `liq_residual` | the synthetic mark on shares the book could not absorb |

`at_price_mode` is `"queue"` in production — a fill at our price requires the
queue in front of us to have been consumed first. The optimistic mode exists to
measure what the queue assumption is worth.

### 7.4 Cash convention

**A passive fill books at OUR limit price, never the print price.** Price
improvement accrues to the aggressor's limit, not to us. A taker fill books at
the **level** price — a taker pays what is resting.

```python
self.cash += -sgn * take * o.price - fee_for(o.price, take)   # passive
self.cash += -sgn * take * px      - fee_for(px, take)        # taker, level px
```

`sgn` is +1 when our bid bought, −1 when our ask sold. Position moves with `sgn`,
cash moves opposite.

### 7.5 Shadow fills

**Our fills never mutate the historical book.** If we buy 500 shares from a
resting offer, that offer is still there for the next event. This is the
standard assumption for a small participant replaying real tape and it is
applied to every fill path. It is visible as a modelling artefact in exactly one
place: when a crossing order fills partly and rests the remainder, the book still
shows the level we just consumed, so the reconstructed book can look crossed.
There is no P&L effect — those shares are already paid for.

---

## 8. THE THREE VENUE RULES THE BACKTEST MODELS

### 8.1 Amendments — PSX Regulation 8.5.2

> *"Modification of price in CFO shall be subject to fill allocation priorities,
> however, reduction of bid/offer quantity shall not be subject to the fill
> allocation priorities."*

In plain words, and this is exactly what the code implements:

| change | queue position |
|---|---|
| price change | **lost** — order goes to the back at the new price |
| size **increase** | **lost** — the whole order re-queues, including the shares already resting |
| size **reduction** | **kept** — applied in place |

`use_cfo=True` sends one Change Former Order per reprice instead of a cancel
plus a new order. An amendment **closes the lifecycle record it amends**
(`end_reason="amended"`) and opens a new one, because for priority purposes a
re-queued amendment is a new order. Counters: `n_cfos` (landed),
`n_cfos_price_change`, `n_cfos_qty_up`, `n_cfos_qty_down`,
`n_cfos_kept_priority`.

**Invariant worth keeping:** `kept place` must equal `size down` exactly. If
those two columns ever disagree, the engine is not reading the venue rule the
way this project believes it is.

### 8.2 The acknowledgement wait

After sending a genuine **cancel**, the side is held until the exchange's reply
would have returned — landing time plus one `draw_ack()`. Until then the old
order may still be resting and may still fill, so quoting again could leave more
size working than intended.

This is a rule about **how fast you learn**, not about message type. A
Cancel/Replace sets **no** wait on either side, because there is no separate
cancel to be told about. Only a genuine cancel does — pulling a quote on a halt,
a crossed book, a stale feed. Counter: `requotes_blocked_by_ack`.

### 8.3 Crossing on arrival — PSX Regulation 8.4.2

A quote that leaves passive can **arrive** marketable, because the book moves
during the latency window.

> **8.4.2** *"Orders that cannot be immediately executed shall be queued for
> future execution."*

Two outcomes, not three. An order that can execute immediately does. Checked
against the rulebook rather than assumed: **8.5.1** lists every order type PSX
accepts (Limit, Market, Market-to-Limit, CFO, CXL) and **8.9(b)** every
time-in-force term (GTD, FOK, IOC). Neither list contains a post-only or
book-or-cancel instruction, and the FIX market-data spec v1.05 defines
`ExecInst(18)` with exactly one value, `'B' = Ok to Cross`, on an outbound
execution report. **There is no way to ask PSX to reject a marketable order.**

`_cross_on_arrival(t, o)` therefore:

1. walks the opposite book best-first,
2. **stops at our own limit price** — a limit order never trades through itself,
3. fills at each **level** price,
4. returns the remainder, which **rests** under 8.4.4 (*"the remaining part of
   such Order shall not lose its priority"*).

This is deliberately **not** `_taker_fill`, which is a sweep-to-flat tool: that
one pays through every level it needs and drops whatever it cannot fill, which is
right for a deliberate flatten and wrong for a limit order.

Counters: `crossed_on_arrival`, `crossed_on_arrival_shares`,
`crossed_on_arrival_rested`. End reason `crossed_filled` when fully consumed.

**Until 2026-09-18 the backtest threw these orders away** and counted
`rejected_crossing` — post-only semantics for a venue with no post-only. That
deleted trades, and not neutral ones: a bid only becomes marketable if the offer
*fell* to meet it, so these were buys into a falling market and sells into a
rising one. Adverse fills. Dropping them flattered P&L. `cfg["cross_on_arrival"]
= False` restores the old behaviour for reproducing a published figure; it is
the old bug, not a cautious setting.

### 8.4 Short selling

`short_policy` takes four values: `"unrestricted"`, `"no_short"`,
`"long_buffer"`, `"slb_uptick"` (with `slb_eligible`). Anything else raises at
construction. Under `no_short` / `long_buffer`, and under `slb_uptick` when the
name is not SLB-eligible, the ask is **capped at the current long position** —
flat means no ask at all. The engine also counts
`short_fills_blocked_by_uptick`, so a run shows what the constraint cost in
fills and not only in P&L.

---

## 9. END OF DAY

The strategy contains no forced-liquidation code. The **engine** flattens, once,
the first time an event arrives past `t1`:

1. Every working order is cleared from both sides.
2. `book.liquidation_value(pos, fee_fn=fee_for)` walks the visible book and
   consumes levels until the position is closed. Each consumed level becomes a
   fill with `reason="liq"`, `window="eod"`.
3. Whatever the visible book **cannot absorb** becomes one synthetic fill,
   `reason="liq_residual"`, marked at
   `ref * (1.0 - pos_sgn * unfilled_haircut_pct)`. **The haircut in production
   is 10%** — the runner sets it explicitly; `0.03` is only the fallback inside
   `mm_backtest` when the key is absent, and it never binds (see §24.2). It pays
   no fee — it is a mark, not a trade.
4. An `eod` report is attached to the engine and the headline number is the
   post-liquidation equity.

The residual is a real weakness and is recorded as such: it puts one synthetic
observation per symbol-day into fill counts, fill rates and markout
distributions, which is item B.13 on the to-do (fix with a flag column).

---

## 10. THE STRATEGY — `MicrostructureMM`

89 constructor parameters, one of which (`session_scale`) is **required with no
default** — omitting it is a deliberate `TypeError`.

### 10.1 Fair value

```python
_mid = 0.5 * (bb + ba)
_spr = ba - bb
imb  = bq / (bq + aq)                       # bid share of touch depth, [0,1]
fair = _mid + _lam * (imb - 0.5) * _spr
```

`_lam` is `micro_lambda` when set, else `1.0` if `use_microprice` else `0.0`.
At λ=1 this is exactly the textbook microprice `(bb·aq + ba·bq)/(bq+aq)`; at λ=0
it is the plain mid; λ<0 leans *against* the imbalance.

**Production runs with `use_microprice=False`, i.e. λ=0, the plain mid.** The
sweep froze it there: OBI used as a fair-value lean was strongly destructive.
OBI survives only defensively.

### 10.2 The half-spread

```python
tau          = (t1 - now) / (t1 - t0), clamped to [0,1]
inv_risk     = gamma * sigma**2 * tau
half_risk    = 0.5 * inv_risk * fair
tox          = |sum(flow)| / sum(|flow|)          over the last flow_window=50 signed trades
if quiet:  tox *= 0.5                              # quiet = no trade for quiet_ms (2000)
half_adverse = tox * (sigma * fair) * 2.0
as_base      = (1/gamma) * log(1 + gamma/kappa)    # Avellaneda-Stoikov, weight 0 by default
cost_floor   = (fee_pct + min_edge_pct) * fair
half         = half_risk + half_adverse + as_base_weight*as_base*fair + cost_floor
half         = max(half, tick)
```

Then the wide-market floor, applied after the viability gate:

```python
mkt_half = (ba - bb) / 2.0
half     = max(half, mkt_half - improve_ticks * tick)
```

With `improve_ticks = 0.0` in production this pins the quote **at the touch** on
any book wider than the cost floor — the dominant term in practice.
`min_edge_pct = 0.0005` (5 bps) is the required edge above fees.

### 10.3 Inventory skew — "the lean" in the risk sense

```python
pos_lots      = pos / size0
remaining_var = (sigma*fair)**2 * session_scale * tau
skew          = gamma * remaining_var * pos_lots
reservation   = fair - skew
```

Long → `skew > 0` → **both** quotes shift **down** by the same amount: sell more
eagerly, buy less. Short is the mirror. It **shifts the pair; it does not change
the width**.

Inventory is measured in **lots**, not shares. `session_scale` is the
dimensional bridge, back-solved per symbol so that the skew at maximum inventory
is about one median spread (PPL ≈ 7.6, UBL ≈ 3.9, PACE ≈ 46.15). There is no
clamp on `skew`, so a wrong `session_scale` produces absurd prices rather than a
bounded error.

### 10.4 Quote placement and tick rounding

```python
px_buy  = floor((reservation - half_buy)  / tick) * tick
px_sell = ceil ((reservation + half_sell) / tick) * tick
```

`floor` on the bid, `ceil` on the ask — **rounding is always in the
conservative direction and can never make a quote more aggressive**.

### 10.5 The post-only clip

Applied last, immediately before emission:

```python
px_buy  = min(px_buy,  ba - tick)
px_sell = max(px_sell, bb + tick)
```

One-directional: it can only pull a quote back, never push it toward the touch.
On a one-tick book the bounds collapse to joining the touch. This is why the
production adapter's crossing assertion should be unreachable — the strategy
clips itself.

### 10.6 Sizing

```python
base_size = size_notional / fair  if size_notional else size0
size      = base_size * (0.5 if tox > 0.5 else 1.0)
size      = max(1.0, round(size))
```

A hard 50% cut when toxicity exceeds 0.5 — a step, not a ramp.

The clip itself is **not** a strategy parameter. It is computed per name per day
by the runner:

```python
clip     = max(1, int(round(CLIP_MULT * trailing_median_trade_qty)))   # CLIP_MULT = 3.0
size     = clip
max_inv  = round(10.0 * clip)          # hard inventory ceiling = 10 clips
soft_inv = round(3.0  * clip)          # soft band = 3 clips
```

`max_inv` is a hard stop on the adding side; `soft_inv` is **also a hard cut**,
not a taper — past `+soft_inv` the bid disappears entirely.

### 10.7 The sub-clip flat rule

```python
is_flat = abs(pos) < size0
```

A residue smaller than one full clip counts as flat for every trigger purpose.
This is item B.11 on the to-do: up to 10% of the inventory cap therefore carries
no aging clock. Note the file uses **three different notions of flat** —
`abs(pos) < size0` for triggers, `abs(pos) < 1e-9` for sign tests, and
`pos != 0.0` for the size branches.

---

## 11. THE THREE QUOTING GROUPS — OBI, QT_2t, QBPS_2

This is the axis the whole book was split on. All three share the **same** base
configuration `_OBI`; they differ only in how the quote is skewed toward the
favourable side of an imbalanced book.

```python
_OBI = dict(obi_throttle=True,  ofi_throttle=False, obi_throttle_thresh=0.15,
            throttle_frac=0.5,  throttle_hold_ms=300.0, qdr_throttle=False,
            enable_pov_cap=False, flow_throttle=False, enable_run_reprice=False,
            enable_aggr_lean=False, enable_age_cross=False,
            size_boost_mult=1.0, queue_skew_ticks=0.0, queue_skew_bps=0.0,
            enable_inv_taper=False)
```

| group | definition | label |
|---|---|---|
| **OBI** | `_OBI` unchanged — no queue skew at all. The control. | `OBI` |
| **QT_2t** | `_OBI` + `queue_skew_ticks=2.0`, `queue_skew_thresh` 0.15 or 0.20 | `QT_2t@15` / `QT_2t@20` |
| **QBPS_2** | `_OBI` + `queue_skew_bps=2.0` — two basis points of mid, converted to whole ticks per name | `QBPS_2` |

### 11.1 The OBI throttle — in all three groups

Present in every arm, including the control. It is a **size** control.

```python
buy_trig  = obi_throttle and (0.5 - imb) > obi_throttle_thresh   # ask-heavy -> protect the BUY
sell_trig = obi_throttle and (imb - 0.5) > obi_throttle_thresh   # bid-heavy -> protect the SELL
```

- The test quantity is `imb - 0.5`, which lives in **[−0.5, +0.5]**. So
  `obi_throttle_thresh = 0.15` means the bid holds **more than 65%** (or less
  than 35%) of touch depth.
- On trigger, that side's clip is multiplied by `throttle_frac = 0.5`, floored at
  one share. **It never removes a side and never moves a price.**
- The cut is held for `throttle_hold_ms = 300` ms of exchange time, per side. A
  fresh trigger re-arms the hold rather than extending it.

Beware the scale: `onetick_obi_thresh` and `aggr_lean_obi_calm` are tested on
the **[−1, +1]** scale instead. All three default to the literal `0.15`/`0.30`
but they do not mean the same book state.

### 11.2 The queue skew — what separates the groups

```python
if   (imb - 0.5) > queue_skew_thresh:   # bid-heavy: BUY is the favourable side
        qs_half_buy  = max(0.0, half - _qs)      # bid steps CLOSER to the touch
        qs_half_sell = half + _qs                # ask steps BACK
elif (0.5 - imb) > queue_skew_thresh:   # ask-heavy: SELL is favourable
        qs_half_sell = max(0.0, half - _qs)
        qs_half_buy  = half + _qs
```

Three mutually exclusive ways to compute the distance `_qs`, in priority order:

1. **Staircase** — `queue_skew_stairs`, a list of `(threshold, ticks)` rungs.
   The **last rung the imbalance clears wins**. Validated at construction:
   thresholds must ascend, and `stairs[0][0]` must equal `queue_skew_thresh`.
2. **Basis points** — `queue_skew_bps` (this is **QBPS_2** at 2.0):
   ```python
   _shift_px = queue_skew_bps / 1e4 * mid
   _nticks   = max(1, round(_shift_px / tick))
   _qs       = _nticks * tick
   ```
   Whole ticks, floored at one. This makes the skew **price-relative**: on a
   300-PKR name 2 bps is 6 ticks; on a 16-PKR name it is 1 tick.
3. **Fixed ticks** — `queue_skew_ticks` (this is **QT_2t** at 2.0). Same two
   ticks on every name regardless of price.

**The threshold is tested on order-book imbalance, not on inventory.** This is
worth stating plainly because the word "lean" is used for both this and the
inventory skew of §10.3, and they are different mechanisms with different
inputs. The queue skew changes **fill probability only** — not size, not fair
value, not the reservation price.

### 11.3 The cheap-tick exclusion

```python
CHEAP_EXCLUDED = {"KEL", "PIBTL", "TPL"}
HONOUR_CHEAP_EXCLUDED = True
```

Those three names have roughly one-tick books — there is no room to skew inside
the touch, so the skew can only hurt. When honoured, **all three** skew
mechanisms are cleared for them (`ticks`, `bps` **and** `stairs` — clearing only
two would silently un-exclude the name, because the staircase branch is tested
first). Their P&L under a skew arm is then byte-identical to OBI, which is why
the effect size `d` comes out NaN for them and they are classified "lean inert".

The label is **not** changed, so an excluded name still writes rows tagged with
the skew arm's name — an OBI result wearing a skew name. That is why they were
dropped from `RUN_ONLY` rather than relying on the exclusion.

---

## 12. EVERY SKEW — PRICE AND SIZE, OFFENSIVE AND DEFENSIVE

**Ten** separate mechanisms move a quote's **price** (P1–P10 below) and six move its **size** (S1–S6).
They are applied in a fixed order and several of them compound, so the order is
part of the specification, not an implementation detail.

### 12.1 The master table

**Price skews.** "Offensive" = toward the touch, more fill probability.
"Defensive" = away from the touch, less.

| # | mechanism | direction | trigger | magnitude | default |
|---|---|---|---|---|---|
| P1 | **inventory skew** | shifts **both** sides together | any non-zero position | `gamma × remaining_var × pos_lots` | always on |
| P2 | **aggression lean** | shifts **both** (moves `fair`) | calm book only, `\|imb−0.5\|×2 < 0.30` | `k × (signed/abs flow) × spread`, `k=0.5` | **off** |
| P3 | **queue skew, favourable side** | offensive | `\|imb−0.5\| > queue_skew_thresh` | `half − _qs` | **the arm** |
| P4 | **queue skew, exposed side** | defensive | same test, other side | `half + _qs` | **the arm** |
| P5 | **EOD / lock ramp widen** | defensive | time or lock ramp, **flat only** | `half × (1+u)`, capped at `10 × ref_spread` | on |
| P6 | **exit ticks inside** | offensive | `\|pos_lots\| ≥ 1.0` or aged | `bb + N·tick` / `ba − N·tick`, fee-bounded | **N = 1** |
| P7 | **exit lean** | offensive, maximal | holding + (time unwind or lock cliff) | `ba − tick` / `bb + tick` | on |
| P8 | **OBI defensive** | defensive | imbalance against us `> 0.15` | `∓ 1.0 tick` | **on in production** |
| P9 | **OFI defensive** | defensive | trailing order flow against us `> 0.30` | `∓ 1.0 tick` | off |
| P10 | **run reprice** | defensive | ≥ 3 consecutive same-side aggressors | `∓ 1 tick` | off |

**Size skews.**

| # | mechanism | direction | trigger | magnitude | default |
|---|---|---|---|---|---|
| S1 | **toxicity halving** | defensive | `tox > 0.5` | `× 0.5` — a step, not a ramp | always on |
| S2 | **throttle** (OBI / OFI / QDR / flow) | defensive | see §11.1 | `× throttle_frac = 0.5`, held 300 ms | **OBI on** |
| S3 | **size boost** | **offensive** | book leans **our** way, `\|imb−0.5\| > 0.15` | `× size_boost_mult` | **off (1.0)** |
| S4 | **inventory taper, adding side** | defensive | any position, POV-based | `max(floor, (1−util)^k)` | off |
| S5 | **inventory taper, reducing side** | offensive | same, only if `inv_taper_both` | `× (1 + util)` | off |
| S6 | **hard suppression** | kill | `kill_buy`/`kill_sell`, `soft_inv`, `max_inv`, short cap | side omitted entirely | on |

### 12.2 The three defensive price skews, verbatim

All three are **retreat-only** — they can never improve a quote toward the
pressure. That is deliberate and is called the anti-microprice guard: leaning
fair value into an imbalance signal was measured as destructive.

```python
# P8 — OBI defensive. imb < 0.5 means ask-heavy, sellers stacked, so buying
# here is adverse. Push the bid DOWN.
if self.obi_defensive and (0.5 - imb) > self.obi_defensive_thresh:
    px = px - self.obi_defensive_ticks * self.tick

# P9 — OFI defensive. Sustained SELLING flow -> price likely to fall -> the bid
# retreats. Mirrors OBI on the trailing flow signal instead of the book.
if ofi_sig is not None and ofi_sig < -self.ofi_defensive_thresh:
    px = px - self.ofi_defensive_ticks * self.tick

# P10 — run reprice. A live SELL run of >= N distinct aggressors is hitting the
# bid, so quote it LOWER and buy into the run cheaper.
if (self.enable_run_reprice and self._rr_run_side == -1
        and self._rr_run >= self.run_reprice_n):
    px = px - self.run_reprice_ticks * self.tick
```

The SELL side is the exact mirror: `+` instead of `−`, and the tests use
`(imb − 0.5)`, `ofi_sig > +thresh`, `_rr_run_side == +1`.

Defaults: `obi_defensive_thresh = 0.15` on the `[−0.5, +0.5]` scale,
`obi_defensive_ticks = 1.0`; `ofi_defensive_thresh = 0.30` on the `[−1, +1]`
scale, `ofi_defensive_ticks = 1.0`; `run_reprice_n = 3`,
`run_reprice_ticks = 1`.

**The run counter collapses a sweep into one order.** A new distinct aggressor
is counted only when the timestamp changes **or** the side flips — same
timestamp plus same side is one order hitting many levels, not many orders.

**`obi_defensive = True` is the production setting.** It was frozen ON by the
sweep. So in every live arm there is a one-tick defensive retreat running
underneath the queue skew, and it is **not gated on `lean_exit`** — which is
why an exit quote can still be pushed a tick away while unwinding (§14.4).

### 12.3 The offensive size skew

```python
if self.size_boost_mult != 1.0:
    boost_buy  = (imb - 0.5) > self.size_boost_thresh   # bid-heavy -> boost the BID
    boost_sell = (0.5 - imb) > self.size_boost_thresh   # ask-heavy -> boost the ASK
...
if boost_buy and not buy_throttled:
    size_buy = max(1.0, round(size * self.size_boost_mult))
```

Two things a reimplementation must copy exactly:

1. **Throttle wins over boost.** The boost is gated on `not throttled` — you
   never boost a side you just flagged as toxic. They are mutually exclusive by
   construction.
2. **The boost multiplies `size`, not `size_buy`.** It replaces the throttled
   value rather than compounding with it. Harmless today because of the gate,
   but it matters if the gate is ever removed.

`size_boost_thresh = 0.15` on the `[−0.5, +0.5]` scale. `size_boost_mult = 1.0`
means off, and it **is** off in every production arm.

### 12.4 The inventory taper

```python
if self.enable_inv_taper and abs(pos) > 0.0:
    cap  = self._pov_capacity() * self.inv_taper_pov_mult
    util = min(1.0, abs(pos) / cap) if cap > 0.0 else 1.0
    add_taper = max(self.inv_taper_floor, (1.0 - util) ** self.inv_taper_k)
    if self.inv_taper_both:
        red_boost = 1.0 + util
...
if self.enable_inv_taper and pos != 0.0:
    _f = add_taper if pos > 0 else red_boost     # BUY adds when long
    size_buy = max(1.0, round(size_buy * _f))
```

A continuous taper of the **adding** side as the position consumes the capacity
we could still clear by the bell, and optionally a boost of the **reducing**
side. Defaults `inv_taper_k = 1.0` (linear), `inv_taper_floor = 0.25`,
`inv_taper_pov_mult = 1.0`, `inv_taper_both = False`.

**Note the failure mode:** `cap == 0` forces `util = 1.0`, so the adding side
collapses to the floor. And unlike `enable_pov_cap`, this path is **not guarded**
against a missing `unwind_profile` — it will raise.

### 12.5 Order of application — BUY side, exactly

Price:

1. base grid price — `floor((reservation − half_buy) / tick) × tick`
2. **P7** exit lean overwrite — `px = ba − tick` (if short and leaning)
3. **P6** exit ticks — `px = max(px, min(bb + N·tick, fee_ceiling))`
4. **P8** OBI defensive — `px −= ticks`
5. **P9** OFI defensive — `px −= ticks`
6. **P10** run reprice — `px −= ticks`
7. **post-only clip** — `px = min(px, ba − tick)`
8. `round(px, 2)`

Size:

1. base `size`, already toxicity-halved (**S1**)
2. **S2** throttle — `size × throttle_frac` if throttled
3. **S3** boost — `size × size_boost_mult` if favourable **and not** throttled
4. **S4/S5** taper — `size_buy × (add_taper | red_boost)`
5. emit

The consequence worth stating: **steps 4–6 run after the exit skews of 2–3**, so
the defensive retreats can and do partially undo an exit improvement. The
post-only clip at step 7 bounds aggression only — it never undoes a widen.

### 12.6 The two thresholds scales — a trap

Three parameters all default to the literal `0.15` and **do not mean the same
book state**:

| parameter | tested on | scale | `0.15` means |
|---|---|---|---|
| `queue_skew_thresh` | `imb − 0.5` | `[−0.5, +0.5]` | bid holds > 65% of touch |
| `obi_throttle_thresh` | `imb − 0.5` | `[−0.5, +0.5]` | bid holds > 65% |
| `obi_defensive_thresh` | `imb − 0.5` | `[−0.5, +0.5]` | bid holds > 65% |
| `size_boost_thresh` | `imb − 0.5` | `[−0.5, +0.5]` | bid holds > 65% |
| `onetick_obi_thresh` | `(bq−aq)/(bq+aq)` | `[−1, +1]` | bid holds > 57.5% |
| `aggr_lean_obi_calm` | `\|imb−0.5\|×2` | `[−1, +1]` | — |
| `ofi_*_thresh` | normalised OFI | `[−1, +1]` | — |

Since `(imb − 0.5) × 2 ≡ (bq − aq)/(bq + aq)`, a `[−0.5, +0.5]` threshold of
0.15 is equivalent to 0.30 on the other scale. Get this wrong and every
threshold is off by a factor of two.

---

## 12A. POV — 10% PARTICIPATION, USED IN THREE PLACES

`unwind_pov = 0.10` is set explicitly by the harness and is the participation
rate the whole inventory framework is built on: **we assume we can trade 10% of
expected volume.**

All three uses share one volume estimate, `_expvol_minsleft()` (§13.3), which
weights the four-bucket volume profile by the tradeable minutes actually left —
so it is break-aware and shrinks correctly on a Friday or a short Ramadan day.

### Use 1 — the exit trigger (always on)

```python
def _unwind_needed(self, pos):
    exp_vol, left_min = self._expvol_minsleft()
    if left_min <= 0.0 or exp_vol is None:
        return True                              # no time / dead tape -> engage
    my_rate     = exp_vol * self.unwind_pov      # shares per minute we can clear
    mins_needed = abs(pos) / my_rate
    return mins_needed >= left_min
```

**This is the mechanism that makes the exit state time-of-day independent**
(§14.3). It asks one question: at 10% of expected volume, can the position
still be cleared passively before the bell? The moment the answer is no, the
adding side is pulled and the exit side leans, whatever the clock says.

### Use 2 — the acquisition cap (`enable_pov_cap`, off by default)

```python
def _pov_capacity(self):
    exp_vol, left_min = self._expvol_minsleft()
    if left_min <= 0.0 or exp_vol is None:
        return 0.0
    return exp_vol * self.unwind_pov * left_min   # shares still clearable
...
eff_max = self.max_inv
if self.enable_pov_cap and unwind_profile and session_segments:
    eff_max = min(self.max_inv, self._pov_capacity() * self.pov_cap_mult)
```

The **same** rate, integrated over the remaining minutes, gives the most
inventory we could still clear. It replaces the static `max_inv` with the
tighter of the two, so the ceiling **decays through the day** as the
high-volume buckets fall behind. `pov_cap_mult = 1.0`.

### Use 3 — the inventory taper (`enable_inv_taper`, off by default)

The same capacity, scaled by `inv_taper_pov_mult`, becomes the denominator of
`util` in §12.4 — a continuous taper instead of a hard ceiling.

### The symmetry worth preserving

Use 1 is the **exit** side of the same arithmetic that Use 2 is the **entry**
side of: *can I still get out?* versus *how much more may I take on?* Both use
`exp_vol × 0.10`, both use the same volume-profile weighting, and both are
therefore consistent by construction. If you reimplement only one of them, the
entry and exit tests will disagree about capacity, which is exactly the bug the
shared `_expvol_minsleft()` exists to prevent.

**In production only Use 1 is live.** `enable_pov_cap` and `enable_inv_taper`
are both off in every arm, so inventory is bounded by the static `max_inv` (10
clips) and `soft_inv` (3 clips), and POV governs only when the exit engages.

---

## 13. TIME STRUCTURE — BUCKETS AND DAY TYPES

### 13.1 The four buckets

```python
BUCKETS = ("first15", "middle", "preclose45", "last15")

open_ms  = session_segments[0][0]
close_ms = session_segments[-1][-1]

if t <  open_ms  + 15*60000:  return "first15"
if t >= close_ms - 15*60000:  return "last15"
if t >= close_ms - 60*60000:  return "preclose45"
return "middle"
```

| bucket | interval |
|---|---|
| `first15` | `[open, open+15min)` |
| `middle` | `[open+15min, close−60min)` |
| `preclose45` | `[close−60min, close−15min)` |
| `last15` | `[close−15min, close]` |

Order of the tests is load-bearing: `first15` beats `last15` beats
`preclose45`. On a session shorter than 75 minutes the zones overlap and that
precedence resolves it.

**A defect you must know about if you rebuild this.** The strategy has a
`_bucket_now()` and the engine stamps every fill with
`getattr(self.strat, "current_bucket", "middle")` — but **`current_bucket` is
never assigned anywhere**. `MicrostructureMM` defines `current_window` and
`current_regime`, not `current_bucket`. So the fills' own `bucket` column reads
`"middle"` for every quoting fill and `"last15"` for every liquidation fill,
regardless of the actual time of day. The sweep runner knows this and works
around it by recomputing the bucket from the fill timestamp:

```python
# the fills' own 'bucket' column is never populated by the engine
# -> was defaulting everything to middle
b = _bucket_of(t, segs)
```

`mm_harness.fifo_attribution` still reads the fill's own column, so **the
harness's attribution path puts 100% of quoting P&L in `middle`**. Only the
runner's path is correct. Either set `current_bucket` in the strategy or delete
the attribute and always recompute from the timestamp — but do not leave both.

### 13.2 The four day types

Regular day, Friday, Ramadan regular, Ramadan Friday.

**There is no day-type classifier anywhere in the code.** No constant, no
branch, no enum. I searched all four files for `ramadan`, `friday` and `jumu`;
the only hits are explanatory comments.

The four types are handled **implicitly and data-driven**, through
`session_segments_*.csv`:

- a **Friday** is simply a date whose row contains **two or more** `start:end`
  segments — the Jumu'ah prayer break splits the session;
- a **Ramadan** day is simply a date whose segment boundaries differ from the
  norm (shorter session, earlier close);
- a **Ramadan Friday** is both: two segments, shifted.

Nothing anywhere hard-codes a session clock time. This is the right design — the
calendar lives in data, and the code only ever asks "which spans are tradeable
today".

Two separate sources of session time, and they are **not** reconciled:

1. **`session_segments`** — per **date**, for the whole market. Text field
   `start:end;start:end`, integers in exchange milliseconds. Drives bucket
   boundaries and all tradeable-minutes maths.
2. **`session_ms = (t0, t1)`** — per **symbol-day**, derived empirically as the
   min and max `ts_exch` of that symbol's snapshots whose phase is
   `CONTINUOUS_AUCTION`. Drives `tau`, the EOD time trigger and the run window.

On a Friday, `t1 - t0` spans **across** the break and therefore overstates
tradeable time, which is exactly why the segment list exists separately. If a
symbol's own continuous phase is shorter than the market segments the two
disagree and nothing detects it.

During the break the phase is not `CONTINUOUS_AUCTION`, so the requote gate
refuses, everything resting is cancelled, and `halted_requotes` increments. The
strategy's clock still advances, because `observe()` is called on every event.

### 13.3 Tradeable-minutes maths (the only genuinely day-type-sensitive code)

```python
left_ms = 0; total_ms = 0
for s, e in session_segments:
    total_ms += (e - s)
    if now < e:
        left_ms += (e - max(now, s))
```

Minutes remaining is the **sum of overlaps of `[now, ∞)` with each segment**, so
the Jumu'ah break is excluded automatically: sitting in the break gives the whole
afternoon segment as remaining, not the wall-clock gap.

Bucket day-lengths are then **derived, not assumed**:

```python
p_total   = clamp(total_min - 30.0, 0, 45)    # preclose45 shrinks on a short day
mid_total = max(total_min - 30.0 - p_total, 1.0)
```

and the remaining minutes are allocated **close-backward** — `last15` first,
then `preclose45`, then `middle`, then `first15` — to produce an expected
volume rate:

```python
a_last  = min(15.0, left_min)
a_pre   = min(p_total,   left_min - a_last)
a_mid   = min(mid_total, left_min - a_last - a_pre)
a_first = max(0.0, left_min - a_last - a_pre - a_mid)
exp_vol = (a_first*vf + a_mid*vm + a_pre*vp + a_last*vl) / left_min
```

So a short Ramadan day automatically shrinks `middle` first, then `preclose45`,
with `first15` and `last15` fixed at 15 minutes each.

---

## 14. THE EXIT RAMP AND CLIFF (time)

All of this lives in `_trigger_state(bb, ba, pos)`, called once per quote cycle,
**before** the viability gate. Master switch `enable_eod_trigger`, forced `True`
by the harness.

| parameter | production source | default |
|---|---|---|
| `eod_ramp_start_min` | `time_windows_*.csv` column 1 | 5.0 min |
| `eod_cliff_min` | `time_windows_*.csv` column 2 | 1.0 min |
| `widen_cap_spreads` | constant | 10.0 |
| `spread_alpha` | constant | 0.05 |

```python
mins_left = (t1 - now) / 60000.0

if mins_left <= eod_cliff_min:                 # THE CLIFF
    in_time_window = True
    if is_flat:
        kill_buy = kill_sell = True            # go dark on BOTH sides

elif mins_left < eod_ramp_start_min:           # THE RAMP
    in_time_window = True
    u_t = (eod_ramp_start_min / mins_left) - 1.0
    if is_flat:
        u_buy = u_sell = max(..., u_t)         # widen BOTH sides
```

- **Ramp zone:** 5 minutes to 1 minute before the close.
- **Cliff zone:** the final 1 minute.
- **Urgency** `u = ramp_start/mins_left − 1`. Zero at the ramp edge, rising to
  `5/1 − 1 = 4.0` at the cliff. Unbounded if `eod_cliff_min = 0` — no guard.
- When **flat**, both sides are affected: any acquisition near the bell is
  unwanted.

### 14.1 What the ramp does to the quote — price only, never size

```python
ema_spread = spread_alpha*spr + (1-spread_alpha)*ema_spread
ref_spr    = max(spr, ema_spread)
raw        = qs_half * (1.0 + u)
cap        = widen_cap_spreads * ref_spr
half       = max(qs_half, min(raw, cap))
```

Widened half = `clamp(base × (1+u), base, 10 × max(spread, EMA spread))`. The
cap is a ceiling, never a target. **No trigger path ever changes size** — size
moves only through toxicity, the throttles, the boost and the taper.

### 14.2 What the cliff does

`kill_buy` / `kill_sell` suppress the side entirely. The protocol between
strategy and engine is that **an omitted side means cancel**, so a cliff kill
actively pulls the resting quote rather than merely not refreshing it.

### 14.3 The holding rule — where the real exit behaviour is

```python
if unwind_profile is not None and session_segments is not None:
    time_unwind = (not is_flat) and enable_eod_trigger and _unwind_needed(pos)
else:
    time_unwind = in_time_window and not is_flat

if (time_unwind or in_lock_cliff) and not is_flat:
    kill the ADDING side              # long -> kill BUY; short -> kill SELL
    lean_exit = True
    un-kill and un-widen the EXIT side
```

with

```python
def _unwind_needed(pos):
    exp_vol, left_min = _expvol_minsleft()
    if left_min <= 0 or exp_vol is None: return True
    my_rate     = exp_vol * unwind_pov          # unwind_pov = 0.10
    mins_needed = abs(pos) / my_rate
    return mins_needed >= left_min
```

**This is the single most consequential piece of time logic, and it is not
clock-bounded.** With a volume profile loaded — which is always, in production —
`time_unwind` does **not** test `in_time_window`. It is purely a solvency test:
the exit state engages at **any time of day**, including mid-morning, as soon as

> position ÷ (expected shares per minute × 10% participation) ≥ tradeable
> minutes left

The 5-minute ramp and 1-minute cliff only govern the **flat** case (widen, then
go dark) and the `current_window` label. Without a profile it falls back to the
clock-only rule.

Zone split, deliberate: **the lock ramp does not pull the adding side — only the
lock cliff does.** A ramp is a price-proximity warning, not a liquidity budget.

### 14.4 The exit lean

```python
if lean_exit and pos < 0:  px = ba - tick      # short -> BUY is the exit
if lean_exit and pos > 0:  px = bb + tick      # long  -> SELL is the exit
```

A pure overwrite of the skew-derived price, and structurally the most aggressive
**passive** placement possible — post-only cannot cross, so one tick inside the
opposite touch is the bound. **Never crosses the spread.**

**One nuance a reimplementation must not miss:** the comments claim the exit
side is "always quoted, never widened or pulled". That is true only of the
*trigger's own* widen. The OBI-defensive widen, the OFI-defensive widen and the
run-reprice push are applied **after** the lean-exit overwrite and are not gated
on it. With `obi_defensive=True` — the frozen production setting — an exit bid
can still be pushed a tick away while unwinding.

### 14.5 `current_window` — five values, and a sixth from the engine

Precedence, top down: `time_cliff` > `lock_cliff` > `time_ramp` > `lock_ramp` >
`none`. Cliffs beat ramps; within a tier, time beats lock. It stays `"none"`
permanently when both triggers are disabled, because `_trigger_state`
early-returns. The engine stamps EOD liquidation fills with a sixth value,
`"eod"`, that the strategy never produces.

---

## 15. THE LOCK-LIMIT RAMP AND CLIFF (price)

The exchange publishes `limit_up` and `limit_dn` on the snapshot feed; the
engine syncs them onto the strategy after every event because the strategy sets
the class attribute `wants_limits = True`. **The strategy never computes a
band.**

### 15.1 Distance metric

```python
mid_px    = 0.5 * (bb + ba)
tick_pct  = (tick / mid_px) * 100.0
cliff_pct = max(lock_cliff_pct, min_cliff_ticks * tick_pct)
ramp_pct  = max(lock_ramp_pct,  cliff_pct + min_ramp_gap_ticks * tick_pct)
d_up_pct  = max(0.0, limit_up - mid_px) / mid_px * 100.0
d_dn_pct  = max(0.0, mid_px - limit_dn) / mid_px * 100.0
```

- Distance is measured in **percent of price, from the mid** — deliberately
  spread-free. Spread-unit distances fired spuriously: the median fire was 3–4%
  from the band on names that never lock.
- Defaults: `lock_cliff_pct = 0.5`, `lock_ramp_pct = 2.0`,
  `min_cliff_ticks = 2.0`, `min_ramp_gap_ticks = 2.0`.
- **The low-price tick tier** is why the two `max()` calls exist. At PACE ≈ 16
  PKR one tick is 0.06% of price and the floor never binds. On a 1-PKR stock one
  tick is 1% of price, so the cliff auto-widens to 2% and the ramp to 4%. Note
  `ramp_pct` is built on the already-floored `cliff_pct`, so the ramp is always
  at least two ticks beyond the cliff and can never fall inside it.

### 15.2 The two bands

```python
if d_up_pct <= cliff_pct:          # UPPER cliff
    in_lock_cliff = True
    if is_flat: kill_sell = True
elif d_up_pct < ramp_pct:          # UPPER ramp
    u_l = ramp_pct / max(d_up_pct, 1e-9) - 1.0
    if is_flat: u_sell = max(u_sell, u_l)

if d_dn_pct <= cliff_pct:          # LOWER cliff
    in_lock_cliff = True
    if is_flat: kill_buy = True
elif d_dn_pct < ramp_pct:          # LOWER ramp
    u_l = ramp_pct / max(d_dn_pct, 1e-9) - 1.0
    if is_flat: u_buy = max(u_buy, u_l)
```

**There is no threshold asymmetry** — the same `cliff_pct` and `ramp_pct` apply
to both bands, with the same urgency formula and the same counters. The
asymmetry is in **which side is acted on**:

| band | the trapped position would be | side suppressed | side left quoting |
|---|---|---|---|
| upper (near limit-up) | **short** — you cannot buy back above the cap | **SELL** | BUY |
| lower (near limit-dn) | **long** — you cannot sell below the floor | **BUY** | SELL |

The surviving side near limit-up is kept deliberately: being **long** into a
limit-up close is the safe side, because you sell into stacked bids.

Both bands can only fire at once if the band is narrower than `2 × cliff_pct` of
the mid; there is no guard, and a flat book would then go dark on both sides.

### 15.3 Two engine-level behaviours on top

1. **`pinned()` → not quotable.** When the touch reaches a band, the engine
   cancels everything and does not consult the strategy at all. This makes the
   lock cliff largely a **pre-pin** guard.
2. **The band clamp** (§6.3) — every desired price is clamped into the band
   before sending.

### 15.4 Measured context, recorded in the code

> Lock approaches **snap**: on PPL / UBL / PACE over 207 days the price sits
> 17–42 spreads out until T−10s and covers the distance in the final seconds.
> So the **ramp will almost never engage**, and "the ramp made no difference" in
> a backtest is the *expected* result, not evidence it is broken. It is kept as
> defence in depth. Check `stats["lock_ramp_widen"]` to see when it actually
> fired.

### 15.5 One unresolved risk

`PSX_SPEC_VERSIONS_20260916.md` records that `999999999.9999` is the "no upper
limit" sentinel, and that the **lower** band has no universal sentinel — its
no-limit value equals the market's own minimum tick. I found **no sentinel
handling** in either `mm_backtest.py` or `micro_mm.py`. If the sentinel reaches
`limit_up` unfiltered the upper trigger simply never fires, which is safe in
direction but silent. Whether the parser strips it is not determinable from the
files I can read.

---

## 16. EXIT TICKS

Three separate mechanisms make the exit side more aggressive. They are ranked:
2 dominates 1 whenever both fire, and 3 returns before either is reached.

| # | mechanism | trigger | price | crosses? |
|---|---|---|---|---|
| 1 | `exit_ticks_inside` | `\|pos_lots\| ≥ exit_inv_threshold` or aged | `bb + N·tick` / `ba − N·tick`, fee-bounded | no |
| 2 | `lean_exit` | holding **and** (`time_unwind` or `lock_cliff`) | `ba − tick` / `bb + tick` | no |
| 3 | `enable_age_cross` | held ≥ `age_cross_ms` (15 min) | `bb` for a sell, `ba` for a buy, size `\|pos\|` | **yes, taker** |

Production runs `exit_ticks_inside = 1` and `exit_inv_threshold = 1.0` — the
skew engages at **one full clip** of inventory. Both are injected by the sweep
runner as overrides; the harness itself does not set them.

```python
# BUY exit (we are short): improve our bid N ticks above the best bid
improved        = bb + exit_ticks_inside * tick
fee_ceiling_buy = true_mid / (1.0 + fee_pct)     # mid - px >= fee_pct * px
improved        = min(improved, fee_ceiling_buy)
px              = max(px, improved)              # only ever MORE aggressive

# SELL exit (we are long): improve our ask N ticks below the best ask
improved        = ba - exit_ticks_inside * tick
fee_floor_sell  = true_mid / (1.0 - fee_pct)     # px - mid >= fee_pct * px
improved        = max(improved, fee_floor_sell)
px              = min(px, improved)
```

Two things to preserve exactly:

- **The fee bound is exact and price-is-its-own-fee-base.** It is deliberately
  *not* `fee_pct × fair` or `fee_pct × mid`. The exit skew may spend the
  `min_edge_pct` cushion but never the fee itself.
- `max`/`min` against the base quote means the exit skew can only make the quote
  **more** aggressive, never less.

### 16.1 The viability gate and its exit bypass

```python
gate_would_block = require_viable and (ba - bb) < 2.0 * half
if require_viable and (ba-bb) < 2.0*half and not lean_exit and not holding_exit:
    return {}                                      # no quote at all
```

A book narrower than twice our required half-spread is not worth quoting. **Both
exit states bypass this gate** — getting flat outranks earning edge — and in
both cases the adding side is suppressed, so the bypass can only ever post the
exit.

### 16.2 Position aging — two independent clocks

- **`enable_age_exit`** (off by default, 15 min): lets aged inventory use the
  tick exit even below `exit_inv_threshold`. Motivated by FIFO evidence that
  round trips longer than 15 minutes turn toxic — realized P&L goes negative.
- **`enable_age_cross`** (off by default, 15 min): crosses the spread to
  flatten. The only taker path in the strategy.

Both measure age from the moment the current net position's **sign** was
established, so a position that is added to inherits the older timestamp. This
is item B.12 on the to-do: FIFO lots aged from fill acknowledgement is correct,
and this degrades most on names that flip often.

---

## 17. CALIBRATION — the four CSVs

All resolved by `newest(pattern)`: a **non-recursive** glob of the results
directory, sorted **lexicographically**, last match wins. "Newest" is therefore
a filename convention, correct only because the stamps are fixed-width
`YYYYMMDD_HHMM`. A missing file is `SystemExit`, never a silent default.

Three of the four go through one loader:

```python
df = pd.read_csv(newest(pattern))
ok = df[df["note"] == "ok"] if "note" in df.columns else df
return {r["symbol"]: tuple(r[c] for c in cols) for _, r in ok.iterrows()}
```

Consequences to preserve: a `note` column, when present, admits **only** rows
where it equals exactly `"ok"`; duplicate symbols mean **last row wins**,
silently; every other column is ignored.

| file | columns read | what it is |
|---|---|---|
| `session_scales_*.csv` | `session_scale` | per-symbol back-solved scalar that makes the inventory skew dimensionally right |
| `volume_profile_*.csv` | `vol_first15`, `vol_middle`, `vol_preclose45`, `vol_last15` | shares per minute in each bucket — drives every POV calculation |
| `time_windows_*.csv` | `eod_ramp_start_min`, `eod_cliff_min` | the exit ramp and cliff, in minutes before the close. **Also carries `capacity_flag`** |
| `session_segments_*.csv` | `date`, `segments` | `start:end;start:end` in exchange ms, **per date, not per symbol** |

`session_segments` has its own reader and **no `note` filter**.

Only `time_windows` has a fallback: a missing symbol defaults to `(5.0, 1.0)`
and the pre-flight treats it as non-fatal. Missing scales, profiles or trade
stats are fatal.

**`capacity_flag` is read by `build_config_assignment.py`, not by the
backtest.** It lives in `time_windows_*.csv`, is loaded there with
`tw["capacity_flag"]`, defaults to `"ok"` when absent, and any value other than
`"ok"` forces `DROP` ahead of every P&L consideration — a risk limit, not an
opinion. No loader in `mm_harness.py` or `universe_expand.py` reads it, so the
backtest itself is unaware of it.

### 17.1 `build_micro_params`

```python
p = dict(R.MICRO_PARAMS)                      # canonical base, file not readable
p.update(dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
              enable_eod_trigger=True, enable_lock_trigger=True))
p["size"]               = clip
p["max_inv"]            = round(10.0 * clip)
p["soft_inv"]           = round(3.0  * clip)
p["session_scale"]      = scale
p["eod_ramp_start_min"] = window[0]
p["eod_cliff_min"]      = window[1]
p["unwind_profile"]     = profile
p["unwind_pov"]         = 0.10
p["session_segments"]   = segments
if overrides: p.update(overrides)             # applied LAST, can override anything
```

Those five hardcoded values are the **locked production defaults shared by every
experiment in the project**, and **three** of them differ from the strategy's own
defaults: `improve_ticks` 1.0 → 0.0, `use_microprice` True → False, and
`min_edge_pct` 0.0000 → 0.0005 (§24.1). Of the nine `MICRO_PARAMS` keys, two are
overridden — `use_microprice` is not one of the nine.

### 17.2 The clip, per name per day

```python
med  = trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)   # TRAIL_DAYS = 10
clip = max(1, int(round(CLIP_MULT * med)))                         # CLIP_MULT = 3.0
```

`med` is the median of that day's **tape** trade sizes for that symbol — not our
fills — taken over the last 10 days on which the symbol actually traded, and
**strictly prior** to the day being simulated:

```python
prior = [stats_sym[str(d)] for d in all_dates
         if str(d) < str(date) and str(d) in stats_sym]
if len(prior) < ndays: return None
return float(np.median(prior[-ndays:]))
```

Fewer than 10 prior trading days → the cell is **dropped**, silently. For a thin
name the 10-value window can span far more than 10 calendar days.

The first `TRAIL_DAYS` dates are dropped from trading outright as warmup, but
still feed the median.

The clip is recomputed for every `(symbol, date)` and is **identical across
arms**, which is what makes the paired-by-day comparisons valid.

---

## 18. THE RUN AND ITS OUTPUTS

A work item is a 10-tuple: `(date, symbol, exit_ticks, obi_def, use_micro, tol,
ofi_win, ofi_thresh, micro_lambda, arm_index)`. Cell count is
`dates × names × arms`. Parallelism is `multiprocessing.Pool(WORKERS=9)` with
`imap_unordered(chunksize=4)`; calibration is built once in the parent and
pickled to each worker once.

Frozen sweep axes, each a list of one because the sweep already settled them:
`EXIT_TICKS=[1]`, `OBI_MODES=[True]` (obi_defensive on), `MICRO_MODES=[False]`
(microprice off), `TOL_MODES=[0.0]` (no pegging), `OFI_MODES=[None]`,
`MICRO_LAMBDA=[None]`.

Three outputs, all timestamped and never overwritten:

| file | grain |
|---|---|
| `{stem}_{stamp}.csv` | one row per config |
| `{stem}_DAILY_{stamp}.parquet` | (config, date, bucket) + an `ALL` row per config-day |
| `{stem}_PERNAME_{stamp}.parquet` | (config, date, **symbol**, bucket) — the finest grain, the source the other two summarise |

PERNAME columns: `net_pkr`, `capture_pkr`, `markout_pkr`, `liq_pkr`, `fee_pkr`,
`markout_bps`, `opened_notional`, `fills`, `net_bps`. The identity
`net = capture + markout + liq − fee` holds per row by construction.

### 18.1 The reconciliation discipline

This project checks arithmetic rather than asserting it, in four places:

1. **Per cell:** `recon_gap = engine_pnl − decomposed_net`, spread pro-rata
   across buckets as `liq_loss` — a **diagnostic**, deliberately excluded from
   `net`, never a plug.
2. **Per run:** decomposed net must be within 2% of engine P&L, or under 1 PKR.
   **This is a different and far looser check than the assignment anchors**,
   which use `RECON_TOL = 1e-4` PKR and hard-stop (§19, §35). Two tolerances
   four orders of magnitude apart live in this project; know which one you are
   quoting.
3. **Grain guards:** every parquet must be unique on
   `(throttle, symbol, date, bucket)`; an `ALL` bucket row, if present, is
   dropped before summing so the four real buckets are not double-counted.
4. **Journal identity:** the resume journal is named after a sha1 of the
   calibration filenames **and** the arm definitions — because resume keys a
   completed cell on `(date, sym, thr)` where `thr` is a bare integer index.
   Redefine the arms and `thr=2` silently changes meaning, and the error would
   be invisible in every anchor, because each cell is internally consistent.

   **As of 2026-09-18 the engine belongs in that digest too** and did not use to
   be: a journal written by one `mm_backtest` was indistinguishable from one
   written by another. The corrected runner hashes the bytes of `mm_backtest.py`
   and `micro_mm.py` as actually imported.

5. **Resume is a hard stop by default.** A resume skips work but does **not**
   put its results back into the aggregation, because the journal carries only
   part of the per-bucket record. `ALLOW_RESUME=False` aborts rather than
   silently emitting a partial universe.

---

## 19. THE CONFIG ASSIGNMENT — four buckets per name

`build_config_assignment.py` decides, per name, which of four settings to run.
It is read-only on every input and writes one timestamped CSV atomically
(temp file → `fsync` → `os.replace`), because the live config layer re-reads the
file every 3 minutes and must never see it half-written.

| label | plain words |
|---|---|
| `QT_2t@15` | lean, trigger 0.15 |
| `QT_2t@20` | lean, trigger 0.20 |
| `OBI` | lean off |
| `DROP` | not quoted |

**The decision rule, in order — order matters:**

1. **Capacity override.** `capacity_flag != "ok"` → `DROP`, regardless of P&L.
   A risk limit, not an opinion.
2. **Drop if every setting loses money** over the sample.
3. **Classify on the per-day effect size** of (lean 0.15 − lean off):

   ```
   d = mean(daily diff) / sd(daily diff)
   D_BAND = 2.0 / sqrt(98) = 0.202
   ```

   - `d > +D_BAND` → lean at 0.15
   - `d < −D_BAND` → lean off
   - in between → the better of {lean off, lean 0.20} by sample PKR

   **Not the t-statistic.** `t = d·√n` grows with the number of days even when
   the underlying effect is unchanged, so a band written in `t` means a
   different thing on 98 days than on 197. `D_BAND` is exactly "t = 2 on a
   98-day window" — the window the walk-forward validated — and means the same
   thing at any sample size.

   **Hysteresis:** a name already in a bucket must cross the far edge
   (`D_EXIT = D_BAND/2`) before it leaves, so the config does not churn on noise
   between refits.

4. **Assign then gate.** The P&L test is applied to the setting the file
   actually **chooses**, never to the best of the alternatives. (Two names were
   once assigned a money-losing setting because the gate tested the wrong
   column.)
5. **Re-entry:** a previously dropped name needs its chosen setting's own daily
   `t > 2.0` before it comes back.

A name whose lean is inert (the cheap-tick three) has `d = NaN` because the two
settings are byte-identical every day, and is classified `OBI`.

`DROP` is implemented as `no_add`: stop quoting the adding side and work
inventory off passively, using `soft_inv = 0`. **Caveat recorded in the code:
`no_add` stops you getting deeper, it does not get you out.**

---

## 20. EVERY EXPERIMENT AND WHAT IT RETURNED

Results as recorded in the project's own documents and run outputs. Where a
result later moved, both figures are given.

### 20.1 The queue-skew family — how the three groups were chosen

| experiment | result |
|---|---|
| **QT_2t vs QBPS_2**, full-year confirmation | QT_2t **8,736,780 PKR** vs QBPS_2 **8,278,591**. QT_2t is the deploy pick. Paired edge of QT_2t over OBI: **+1.3036 bps/day, t = +9.42** |
| **Tick depth 0 → 2 → 3** | Markout gain **exhausted at 2 ticks**. 0→2t: markout +1.08 bps for capture −0.15. 2→3t: markout +0.05 for capture −1.21. Three ticks is **destructive**, t = −26 against two |
| **Graded staircases** | **Dead.** −1.93 bps, t = −34 |
| **Trigger 0.15 → 0.20, uniform** | +0.13 bps (t = +5.2) but **PKR flat** (t = +0.6). It helps lean-losers (+209k) and hurts lean-winners (−154k) by about the same. Spearman −0.57, t = −7.3. **The trigger is a per-name dial, not a global one.** A claim that drawdown nearly halves did **not** replicate on holdout (−10%) — it was an artefact of a subset overweighted with lean-losers |
| **Gate sweep** | 30 names × 8 arms, 47,280 cells; then a 110-name confirmation of lean 0.20, 21,670 cells. All anchors reconciled to 0.000000 |
| **Cheap-tick names (KEL, PIBTL, TPL)** | Lean **off** wins on every name, paired t = **+4.61**. Mechanism: capture goes **negative** (+1.64 → −1.79 bps) while markout barely moves — on a one-tick book the lean walks the quote through the other side |

### 20.2 The config assignment — walk-forward, 110 names, 10 folds

60-day fit, 20-day test, step 12. Gain over always-0.15, with the drop gate
applied to both rules:

| rule | mean | se | t | folds > 0 |
|---|---|---|---|---|
| two buckets (off / 0.15 + drop) | +5.45% | 0.94 | **+5.83** | 10/10 |
| three buckets (shipped) | +6.41% | 1.22 | **+5.25** | 10/10 |
| **three minus two** | **+0.96pp** | 0.58 | **+1.64** | 8/10 |

**Read honestly: the DROP gate carries nearly all the value and is solid. The
third bucket does not clear the |t| > 2 bar used everywhere else in this
project.** It was shipped anyway as a small positive-expectation bet, because it
only weakens an already-validated mechanism on 13 names, and 8/10 folds is a
second weaker line of evidence. Re-measure at the next refit.

Shipped split: **68 at QT_2t@15, 13 at QT_2t@20, 17 at OBI, 15 DROP = 113
names.** In-sample 15,366,558 PKR against 13,285,941 if every name ran lean
0.15 — **+15.7%**, and **98.8%** of the per-name in-sample ceiling of 15,555,884.

**Correction, computed from `config_assignment_20260915_0043.csv` directly.**
The project to-do records this share as **91.7%**. It is not: summing each
name's assigned-setting P&L gives 15,366,558, summing `max(OBI, lean 0.15,
lean 0.20, 0)` per name gives 15,555,884, and **15,366,558 / 15,555,884 =
98.8%**. (The `0` in that max is the DROP option — without it the best-of-three
ceiling is only 14,539,731, which the assignment *beats*, because dropping a
loser is worth more than the best of three bad settings.) The bucket counts
68 / 13 / 17 / 15 and both PKR totals reconcile exactly against that CSV; only
the percentage was wrong. **In sample the assignment leaves ~1.2% on the table,
not 8.3% — which makes the third bucket's marginal +0.96pp look smaller still,
not larger.**

Two edge cases tested and left alone because neither improved the walk-forward:
QUICE is dropped although lean 0.15 earned +6,336 in sample (a fallback changed
the walk-forward by 0.00pp); and the middle bucket never considers lean 0.15,
costing ~2,167 PKR on FCL.

### 20.3 The reconcile gate — does the production engine reproduce this backtest

240 symbol-days (12 names × 20 dates, June 2026), four engine runs each.

**240 of 240 exact, 0.00 PKR difference**, and exact at the **record** level —
every order-disposal column matches between the two runs, not just the money.

Five engine defects were found on the way, each by the gate failing and then
being instrumented rather than guessed at:

1. **Duplicate orders into the latency window** — 4,901 duplicate sends, 5,205
   orphaned resting orders across four symbol-days. Fixed by reserving the side
   at send time.
2. **A cancel racing its own replacement** — only when an amendment was already
   on the wire. Fixed by holding the side while any message is outstanding.
3. **An `InvalidTransition` crash** in the live order manager that killed 2 of
   every 6 symbol-days.
4. **The simulator could not hold two orders on one side** — which made the
   queue-preserving policy untestable, and had its second order overwrite the
   very order whose queue position the policy exists to protect.
5. **The acknowledgement wait the engine drew, stored and never read.** On PPL
   2026-06-19 the backtest waited 1,465 times and the engine none.

And two defects in the **model of PSX itself**, where both engines agreed with
each other and both were wrong about the exchange: the post-only rejection
(§8.3) and a missing price band on the engine side.

### 20.4 What the two model corrections cost

| | |
|---|---|
| before | 30,785.24 PKR |
| after | 30,703.16 PKR |
| net | **−82.08 PKR, −0.27%** |

Aggregate impact: nothing. But **224 of 240 days moved** — 103 down totalling
−7,646, 121 up totalling +7,564 — with a per-day standard deviation of
**106.28 PKR** against a mean daily P&L of about 128. Individual days moved up
to ±524.

**The corrections are material per day and immaterial in total.** Per name:
**6 of 12 names changed rank**, NRL lost 998 PKR over 20 days, and NCPL fell
from t = 2.16 to **1.92** — below a bar it previously passed. No name changed
sign.

The lesson for anything built on this: the aggregate is safe, per-name figures
are not.

### 20.5 What queue position is worth — three requote policies

Same days, same exchange, same latency seed. The clip is 500; a partial fill
takes 400 and leaves 100 resting with the place it earned.

1. **amend up** — amend the 100 back to 500. Under 8.5.2 the whole order
   re-queues. This is `mm_backtest`'s rule and the baseline.
2. **second order** — leave the 100, send a separate order for the 400.
   Priority paid only on the increment.
3. **don't top up** — send nothing. Show 100 until someone hits it.

| policy | total | vs baseline | better on | t |
|---|---|---|---|---|
| amend up | 30,703.16 | — | — | — |
| second order | 30,688.36 | −14.81 | 55 of 240 | **−0.03** |
| don't top up | 30,042.98 | −660.18 | 74 of 240 | **−0.92** |

**Neither clears |t| > 2, and the conclusion strengthened after the
corrections.** On the old basis the second-order policy looked worth +562.63
(t = 0.85) — suggestive. Corrected, it is **−14.81, t = −0.03**:
indistinguishable from zero on 240 symbol-days.

The mechanism is in the amendment breakdown. Of 75,229 landed amendments on the
baseline, **69,681 — 92.6% — lose their place**, and 62,918 of those (83.6% of
all landed amendments) are **price moves**, which re-queue under 8.5.2
regardless of policy.

**Queue position is not worthless in general. It is worth nothing to this
strategy**, because the strategy reprices on nearly every tick and so surrenders
its place constantly for a different reason. Changing that means widening the
requote tolerance (`price_ticks`), not changing the size policy.

### 20.6 Cross-asset work — all negative, recorded so it is not redone

| experiment | result |
|---|---|
| **PSX single-stock futures liquidity** | Almost entirely dead. Of 100 roots with an active-month contract, only **13** trade more than 1,000 times a day; MCB trades 12, NATF 10. Spread in bps is almost purely a function of that deadness: **Spearman ρ = −0.875** between trades/day and spread, n = 100, p = 1.3e-32 |
| **Futures market making, four arms** | **All four lose.** Markout is 1.19–1.28× capture in every arm; capture stayed positive, so the failure is simply that the adverse move after each trade exceeds the spread earned. Only the size throttle helped (3/3 names, +3,323 PKR). Even at its nine-month average futures MM would run at ~24 PKR/name-day against ~597 for the share book |
| **Delta-hedge ladder** | **First rung inverted.** Crossing the **share** book is cheaper than crossing the same-ticker future on **99 of 100 names**, median 3.30× — so the stated first preference was the most expensive rung. **No name clears the 2.67 bps edge by crossing in either book**: best is EFERT shares at 5.30 bps = 2.0× the edge; median share book 13.79 bps = 5.2×; median futures 48.09 bps = 18.0× |
| **Why that is structural** | A maker **earns** the spread by posting; a hedger **pays** it by crossing; and the edge is derived from that same spread. A one-for-one delta hedge by crossing cannot be profitable anywhere |
| **Netting** | Now load-bearing. At a budget of 20% of the edge the median name can hedge only **3.9%** of traded notional; BOP 7.4%; EFERT 10.1%; median futures 1.1%. The median name needs ~96% netting to afford a hedge |
| **ETF rung screen** | PSX's tick is a flat 0.01 PKR, so a one-tick spread is `100/P` bps. An instrument under **37.45 PKR** cannot have a spread under 2.67 bps; under **89.6 PKR** it cannot have a round trip inside the edge including the 1.554 bps of fees. Apply that before measuring anything |
| **Cash-settled futures** | Do not exist as a market: 6 trades on 1 day across 20 dates |
| **Sector lead-lag** | `leadlag_screen.py` is built and validated on synthetic ground truth (recovered a +1000 ms peak against a true +800 ms; corr +0.464 vs +0.155 at lag 0), sector map complete at 107 names / 19 sectors. **It HAS been run** — 20 days, 157 pairs, output verified in §37.3. An earlier draft of this document said it never had; see the correction in **§37.4** |

### 20.7 Data-quality findings

- **285 crossed books** on NRL and MLCF over three days were traced to a real
  cause and are why `skip_crossed_book` exists.
- A ticker listed in more than one PSX market (MLCF in `REG` and
  `EQ_SQUARE_UP`) had its book wholesale replaced by another instrument's
  prices. Fixed by filtering the snapshot read to `market="REG"` — **the only
  market-filtered read in the harness**. The trade-stats pre-pass is **not**
  filtered, so a dual-listed name's clip is still sized from both markets'
  trades.
- The parsed store ends **2026-06-30**.

---

## 21. THE RULES, GATHERED IN ONE PLACE

Venue rules the backtest implements:

1. **8.4.1** Orders match on price, then time of entry.
2. **8.4.2** An order that cannot be executed immediately is queued. Two
   outcomes, not three — there is no rejection of a marketable limit order.
3. **8.4.4** A partly executed order keeps its priority for the remainder.
4. **8.5.1** The complete order-type list: Limit, Market, Market-to-Limit, CFO,
   CXL.
5. **8.5.2** A price change or a size **increase** re-queues; a size
   **reduction** is applied in place.
6. **8.9(b)** The complete time-in-force list: GTD, FOK, IOC.
7. **Circuit bands** are published on the feed and read, never computed.

House rules the backtest imposes on itself:

8. **Never quote off a crossed book.**
9. **Never quote when the touch is pinned at a band.**
10. **Never quote outside the published band** — clamp to the edge instead.
11. **Never cross the spread** except on the one opt-in age-cross path.
12. **Book a passive fill at our own limit**, never the print price.
13. **Book a taker fill at the level price**, never our limit.
14. **A fill never mutates the historical book.**
15. **Tick rounding is always away from aggression** — floor the bid, ceil the
    ask.
16. **An omitted side means cancel.**
17. **The exit side is never suppressed by the trigger that suppressed the
    adding side.**
18. **Never let the exit skew spend the fee** — it may spend the edge cushion.
19. **A missing calibration file is a hard stop**, never a silent default.
20. **A resume that would drop cells from the outputs aborts.**

---

## 22. KNOWN DEFECTS AND FLAGGED SIMPLIFICATIONS

Carry these forward; they are all recorded in the code or the to-do, and none
are fixed.

| # | thing | why it matters |
|---|---|---|
| 1 | **`current_bucket` is never assigned** | Every fill is stamped `middle` (or `last15` for liquidation). The harness's attribution path is therefore wrong; only the runner's recomputation is right |
| 2 | **Fills reach the strategy with no ack delay** | Exchange-time fills update `pos` immediately; real fills arrive one inbound hop later |
| 3 | **`liq_residual` is a synthetic fill** | One fabricated observation per symbol-day pollutes fill counts, fill rates and markout distributions. Needs a flag column |
| 4 | **The unfilled haircut is a guess — and it is 10%, not the 3% an earlier draft of this document recorded** (§24.2). `0.03` is the never-binding fallback | It prices whatever the visible book cannot absorb at the close, so end-of-day P&L on every unfilled position scales with it |
| 5 | **Latency tail parameters are priors**, not measurements | Refit from colo telemetry |
| 6 | **Sub-clip inventory counts as flat** | Up to 10% of the inventory cap carries no aging clock |
| 7 | **Position age is measured from a sign change** | New shares inherit the old age; worst on names that flip often |
| 8 | **No sentinel handling on the circuit bands** | The upper trigger silently never fires if `999999999.9999` reaches it |
| 9 | **`ofi_throttle` silently enables the OFI price retreat** | The guard tests `ofi_sig is not None`, not `ofi_defensive` |
| 10 | **`enable_inv_taper` is unguarded** | Without an `unwind_profile` it raises; `enable_pov_cap` is guarded, this is not |
| 11 | **`session_segments` and `session_ms` are never reconciled** | Market-wide segments vs a symbol's own observed continuous window |
| 12 | **The trade-stats pre-pass is not market-filtered** | A dual-listed name's clip is sized from both markets' trades |
| 13 | **Shadow fills can show a crossed book** | Only visible on a partial crossing-on-arrival. No P&L effect |
| 14 | **P&L bps denominator** | Buy-entry notional vs buy-plus-sell turnover is a 2× difference — larger than the entire fee load. Unconfirmed which the reporting uses |
| 15 | **`_taker_fill` may look ahead** | Unchecked whether the first liquidation can read a book later than the decision timestamp, and whether repeated walks reuse one depth snapshot |
| 16 | **Asymmetric quote suppression** | The early returns in `quotes()` kill **both** sides. Killing the *reducing* side removes the only passive exit and forces a later spread-crossing unwind. This is now load-bearing, because `DROP`/`no_add` relies on the same mechanism |
| 17 | **`log_fill_state` has never been used** | The whole edge is capture-driven, which is exactly what a passive-fill assumption flatters. This is the single most important unaddressed item for believing the P&L |

---

## 23. REBUILD CHECKLIST — the order to implement in

1. `TICK`, the fee schedule, `fee_for`.
2. `Book` — reconstruct from snapshots and updates; `bbo`, `qty_at`,
   `ranked_depth`, `obi`, `pinned`, phase, circuit bands.
3. `LatencyModel` — seeded, two legs, exponential tails.
4. `MyOrder` and `work` as a **list per side**; send-time reservation with
   `t_active = None`.
5. The event loop with **two clocks** and the running-maximum knowledge time.
6. `_requote` and its four stand-down reasons (phase, pinned, stale feed,
   crossed book); the in-flight hold; the ack wait
   after a genuine cancel; the band clamp.
7. `_arrive`: identity check, crossing-on-arrival, queue snapshot.
8. `_on_market_trade`: price-then-time ordering, drain the level queue **once**,
   offer survivors oldest-first; the reason tags.
9. Amendments under 8.5.2, with the `kept place == size down` invariant.
10. End-of-day liquidation and the residual mark.
11. The strategy: fair value → half-spread → inventory skew → reservation →
    per-side price → queue skew → exit ticks → defensive widens → post-only
    clip → size → throttle → boost → taper → emit.
12. `_trigger_state`: the time ramp/cliff, the lock ramp/cliff, the holding rule,
    `current_window`.
13. The calibration loaders and `build_micro_params`.
14. The runner: work items, the pool, the FIFO decomposition, the three outputs,
    the reconciliation anchors, the journal identity **including the engine
    digest**.
15. `build_config_assignment.py` and its four-bucket rule.

**The test that the rebuild is correct** is not that it produces a plausible
number. It is that:

- every anchor reconciles to 0.000000,
- `kept place` equals `size down` exactly,
- the disposal columns sum to the record count with no remainder,
- and a second engine driving the same exchange off the same latency seed
  reproduces it **to the paisa, order for order**.

That last one is what the reconcile gate is, and it is the only check that has
ever caught anything structural.

---

# PART II — ADDED 2026-09-18 (second pass)

Sources added in this pass, all read in full: `run_legacy_mm.py`,
`spot_capture_markout_decomp.py`, `iceberg_feasibility.py`,
`leadlag_diagnose.py`, `run_persistence.py`, `persistence_metrics.py`,
`run_daily_stats.py`, `ticker_stats_core.py`, `corp_actions_master.py`,
`corp_action_detector.py`, `extreme_obi_sweep.py`, plus the project findings
`PSX_HEDGE_COST_FINDING_20260915_0700.md` and
`PSX_CROSS_ASSET_CLOSURE_20260915_1833.md`.

`run_legacy_mm.py` closes the largest gap in Part I: every parameter default
that Part I traced to "a file I cannot read" is now resolved in §24.

---

## 24. THE CANONICAL PARAMETER SETS — `run_legacy_mm.py` (module `R`)

### 24.1 `MICRO_PARAMS` — the base every strategy starts from

```python
MICRO_PARAMS = dict(
    size=50,                # match naive's size so sizing is not a confound
    max_inv=500,            # match naive's inventory cap
    gamma=0.15,             # risk aversion (default)
    kappa=1.5,              # A-S base intensity (inert until as_base_weight>0)
    min_edge_pct=0.0000,    # no edge demanded above fees yet (default)
    tick=0.01,              # PSX Ready-Market tick is a flat 1 paisa (verified)
    improve_ticks=1.0,      # placement: quote 1 tick inside the touch (default)
    tol_ticks=0.0,          # quote pegging off (default)
    require_viable=True,    # stand aside when the spread cannot cover cost
)
```

**Nine keys. That is the whole dict.** Everything else in the 89-parameter
constructor comes from `MicrostructureMM`'s own defaults.

Two keys are **deliberately absent**:

- **`fee_pct`** — omitted so the strategy falls back to `mm_backtest.FEE_TOTAL_PCT`.
  `micro_mm` raises rather than inventing a default if `mm_backtest` is not
  importable, so the fee can never be silently wrong.
- **`session_ms`** — injected per symbol-day, not carried in the dict.

A comment instructs that `gamma`, `kappa`, `min_edge_pct`, `improve_ticks` and
`tol_ticks` must **not** be tuned here: *"that is Stage-C calibration, done
later."* And indeed `build_micro_params` overrides three of them
(`min_edge_pct → 0.0005`, `improve_ticks → 0.0`, `use_microprice → False`), so
**the production values differ from this dict**. Part I §17.1 has the override
list; this is where the base comes from.

### 24.2 `CFG` and the latency ambiguity

```python
CFG = dict(
    latency_ms=120,               # constant one-way latency
    at_price_mode="queue",        # realistic queue consumption
    fill_on_crossing_adds=False,  # conservative
    unfilled_haircut_pct=0.10,    # haircut on inventory the book cannot absorb
)
LATENCY_SEED = 0
```

Never passed as-is. The effective config is built per symbol-day:

```python
cfg = dict(CFG, session=(t0, t1), latency_model=LatencyModel(seed=LATENCY_SEED))
```

**Two corrections to Part I, both material:**

1. **The unfilled haircut is 10%, not 3%.** Part I said 3% from
   `cfg.get("unfilled_haircut_pct", 0.03)` — that is the *fallback* inside
   `mm_backtest`, and `run_legacy_mm` overrides it to **0.10**. So a real run
   marks the residual the book could not absorb at **10%** away from the
   reference, not 3%. The 3% only applies to a config that omits the key.

2. **`latency_ms=120` and a stochastic `latency_model` coexist in the same
   dict.** A comment claims the stochastic model is what runs (~45 ms median,
   2% fat tails, "not the flat-120ms constant"), but the constant is never
   deleted. Which one binds is decided inside `Backtester`, not here. A
   reimplementation must pick one and delete the other; leaving both is how a
   run gets described with the wrong latency in a write-up.

**`LATENCY_SEED = 0`, and a fresh `LatencyModel` is constructed per symbol-day.**
So every symbol-day draws the identical latency sequence regardless of batch
order or worker count. That is what makes parallel runs reproducible and what
makes the reconcile gate's seeded-draw argument valid.

### 24.3 `USE_MICRO`

```python
USE_MICRO = True
```

The master strategy switch. `make_strategy()` returns `MicrostructureMM` when
True, else the `NaiveSymmetricMM` baseline (`half_spread=0.20, size=50,
max_inv=500`). The module docstring still says the strategy is the naive
baseline and warns to "fix its short-cap None-return bug first" — **the
docstring is stale**; the module as shipped runs the micro strategy. Whether
that bug was ever fixed is not recorded anywhere I can read.

### 24.4 The session window — where `t0`/`t1` really come from

```python
cont_snap = s[s["phase"] == "CONTINUOUS_AUCTION"]
if len(cont_snap) == 0:
    return None
t0, t1 = int(cont_snap["ts_exch"].min()), int(cont_snap["ts_exch"].max())
```

Phase-based, from **REG snapshots only**. The recorded reason: the previous
rule (first/last non-auction trade) let the end-of-day flatten fire in
`AFTER_HOURS` / `MARKET_CLOSED` against a one-sided post-close book — the
"one-sided close" bug that inflated liquidation loss. Phase-based is also the
only definition robust to Ramadan hours (recorded as 9:17–13:30) and the Friday
split.

### 24.5 Column contracts

```python
REQ_TRADES  = [symbol, transact_time, capture_ts, price, qty,
               initiator, aggressor_side, resting_order_id, appl_seq]
REQ_UPDATES = [symbol, transact_time, capture_ts, order_id, side,
               price, qty, event, appl_seq]
REQ_SNAP    = [symbol, msg_seq, orig_time, capture_ts, entry_type,
               px, phase, order_ids, order_qtys, qty, market]
```

These lists serve double duty: they are the schema contract **and** the literal
column projection pushed down to parquet. Adding a column to a list also changes
what is read off disk.

### 24.6 The market filter — narrower than Part I implied

```python
def read_symbol(dset, cols, sym, market=None):
    ...
    if market is not None and "market" in cols:
        pred = pred & (ds.field("market") == market)
```

Two conditions must both hold, and `"market"` appears in **`REQ_SNAP` only**.
At the call sites:

| read | filtered? |
|---|---|
| `ob_updates` | **no** |
| `ob_snapshot` | **yes**, `market="REG"` |
| `trades` | **no** |

So the 2026-09-17 square-up fix covers **snapshots only**. For a dual-listed
ticker such as MLCF, square-up **order updates and trades still enter the event
stream**. The measured contamination that motivated the fix: MLCF carried 69
`EQ_SQUARE_UP` rows against 272,894 `REG` rows, with a square-up best bid of
122.49 against the regular market's 95.86 ask.

This is a live, unresolved gap, and it is wider than Part I recorded.

---

## 25. THE EVENT STREAM — `build_events`

### 25.1 Timestamps

| table | `ts_exch` | `ts_cap` |
|---|---|---|
| `ob_updates` | `transact_time` | `capture_ts` |
| `trades` | `transact_time` | `capture_ts` |
| `ob_snapshot` | **`orig_time`** (FIX tag 42, second precision) | `capture_ts` |

All converted to **integer milliseconds since epoch, UTC**, by truncating
division — sub-millisecond precision is discarded.

### 25.2 The ordering contract

```python
events  = [(r.ts_exch, 1, r.appl_seq, "U", r) for r in u.itertuples()]
events += [(r.ts_exch, 1, r.appl_seq, "T", r) for r in t.itertuples()]
events += [(r.ts_exch, 0, 0,          "S", r) for r in snap_ev.itertuples()]
events.sort(key=lambda e: (e[0], e[1], e[2]))
```

- Sort key is `(ts_exch, kind_rank, appl_seq)` — three elements only.
- **Snapshots rank 0, updates and trades rank 1.** At an identical exchange
  timestamp, **a snapshot is always applied first**.
- `list.sort` is **stable**, so equal keys keep build order: **all updates
  before all trades**, then original row order within each. There is no explicit
  update-vs-trade tiebreak — it is an artefact of construction order plus sort
  stability. A reimplementation that builds the lists in a different order gets
  a different event sequence and therefore different fills.
- Ordering is by **`ts_exch` only**. `ts_cap` rides on the row and plays no part
  in the sort; knowledge time is the `Backtester`'s job.
- `appl_seq` is coerced to int64 with missing → **−1**, so unparseable sequence
  numbers sort *before* everything at that millisecond. Snapshots carry a
  literal 0.

### 25.3 `snap_groups` and the key that fixed the crossed books

```python
s["snap_key"] = s["msg_seq"].astype(str) + "|" + s["orig_time"].astype(str)
snap_groups = {k: prep_snapshot(grp) for k, grp in s.groupby("snap_key")}
```

`snap_groups` maps the composite key to a **pre-parsed** snapshot object, not a
DataFrame. The pre-parse removed what was measured as **89% of runtime** in
`Book.snapshot()`.

**This composite key is the fix for the crossed-book finding.** Measured on NRL
and MLCF, 2026-06-30:

| grouping key | groups holding two level-1 bids |
|---|---|
| `symbol + msg_seq` | 90 of 12,476 |
| `+ market` | still 90 |
| `+ channel` | still 90 |
| `symbol + msg_seq + orig_time` | **0 of 12,566** |

0.72% of snapshot messages were affected, and **275 of the 285 crossed books
were inverted more than one level deep** — a genuinely merged book, not a
one-tick artefact.

**The caveat is recorded and still stands:** `orig_time` is second-precision, so
two genuinely distinct snapshots inside one second would still merge. The
durable fix is a per-message id from the parser, which does not exist yet.

### 25.4 `rest_oid`

`resting_order_id` is parsed by `parse_rest_oid`: a string starting with `(` is
`ast.literal_eval`'d and element 0 taken; a list/tuple yields element 0; a
non-empty string is returned as-is (the PSX bare order-id case, e.g.
`0010THF0D00017T6`); anything else → `None`. **Never `eval()`.**

The per-run CSV carries `rest_oid_resolved_frac`. Near zero means queue tracking
has silently degraded to the fallback path — a health metric worth watching.

---

## 26. THE FIFO DECOMPOSITION — where the P&L actually comes from

`spot_capture_markout_decomp.py`. This is the file that answers *where does the
edge live and how much of it does adverse selection eat.*

### 26.1 The identity

```
net = capture + markout − fees
```

and it must equal engine realized cash **exactly**.

**Two corrections to how that is stated, both found by re-reading the code
against this document.**

**First: there are TWO decompositions in this project and their identities are
not the same.** `universe_expand.py` — the runner that produces the shipped
per-name numbers — uses

```python
# universe_expand.py
net = capture + markout + liq_cap + liq_mko - fee - liq_fee
```

where `liq_cap` and `liq_mko` are **measured** components of lots whose closing
fill was a forced end-of-day liquidation, booked separately so the total
telescopes. `liq_fee` is a legacy column, zero since liquidation fees started
routing through `fee`. That identity is clean: every term is measured.

`spot_capture_markout_decomp.py` — the analysis tool this section describes —
uses a different one, and it is **not** clean. See below.

**Second: the file does not `assert` the identity.** The word appears once, in a
header comment (*"decomposition correct by construction; we assert it per
bucket"*), and there is no `assert` statement in the executable code. The
identity is **printed** as a reconciliation line, not tested. An earlier draft of
this document said the file asserts it; that was reading the comment as code.

The three components:

**Capture** — the half-spread earned at the moment of the fill:

```python
cap_sign = 1.0 if side == "BUY" else -1.0
cap = cap_sign * (mid_at_fill - px) * qty
```

A buy below the mid is positive; a sell above the mid is positive.

**Markout** — how the mid moved between opening a position and actually closing
it, measured on the **FIFO-matched exit**, not a fixed horizon:

```python
open_sign = 1.0 if lot["side"] == "BUY" else -1.0
mko = open_sign * (m_exit - m_open) * matched
```

**A negative markout means the price moved against the open position — that is
adverse selection, and it is what eats capture.**

**Fees** — both legs on the matched shares, with one exception:

```python
close_is_mark = (fl.get("reason", "") == "liq_residual")
fee = H.fee_for(lot["px"], matched) + (0.0 if close_is_mark else H.fee_for(px, matched))
```

A residual mark is a haircut, not a traded exit, so it pays no closing fee.
Charging one would re-introduce a reconciliation residual.

### 26.2 Two corrections recorded in the file, both of which were real bugs

**The missing closing-leg capture.** The exit fill's own half-spread, pro-rated
to the matched shares, is booked to the **opening** bucket:

```python
if qty > 0:
    per[ob]["capture"] += cap * (matched / qty)
```

The comment is explicit: *"Without it, measured_net undercounts by the exit
capture (~half of gross capture)."* The phrase *"the root of the −70%
unexplained"* names the same bug, but it lives in **`universe_expand.py`**, not
in this file — an earlier draft of this document attributed it here.

**The liquidation plug was half-deleted — and this document previously said it
was fully deleted. It is not.** The improvement is real: an earlier version
*reconstructed* the end-of-day residual and added a liquidation-loss plug; now
the engine emits the EOD book-walk as real `liq` / `liq_residual` fills that the
FIFO loop matches like any other. But what is left over is still spread:

```python
recon_gap = float(dr.pnl()) - dec_total          # the unexplained remainder
...
w = (per[b]["opened_notional"] / tot_opened) if tot_opened > 0 else 1.0/len(H.BUCKETS)
per[b]["liq_loss"] = recon_gap * w               # distributed PRO-RATA by notional
...
liq_net = a["liq_loss"] + a["liq_fee"]
net     = a["capture"] + a["markout"] - fee_all + liq_net    # and added into net
```

So `net == capture + markout − fee + recon_gap`, which sums over buckets to
engine P&L **by construction**. The file's own comment says the residual is
*"never added into net_pnl to force the identity"*; the code three hundred lines
later does exactly that.

**What is actually true, stated carefully**, because the distinction still
matters:

- The residual is **itemised in its own printed column** (`liq_loss`), so a
  reader can see how big it is. That is much better than a silent plug.
- But it is **not excluded from `net`**, and `net` therefore cannot be used as
  evidence that the decomposition is complete. **The number to look at is the
  `liq_loss` column itself.** If it is not ~0, the decomposition is incomplete
  and `net` is hiding it in plain sight.
- A rebuild should make this explicit: compute `net_explained = capture +
  markout − fee` **without** the residual, print `recon_gap` beside it, and
  **assert** `|recon_gap| < tol` rather than printing it.

The diagnostic-versus-plug discipline is still the right principle. This file
states it and then does not quite follow it — which is exactly why the principle
is worth writing down.

### 26.3 The jump / diffusion split

Markout is further decomposed into the part that came from a **jump** and the
part from ordinary **diffusion**, so you can tell whether adverse selection is
"the price drifted against me" or "I was run over by news".

Method, Lee–Mykland (2008)-style thresholding over the mid path between the
opening fill and its matched exit:

```python
r  = np.diff(np.log(m))          # per-event log returns over the window
ar = np.abs(r)
if len(ar) >= 5:
    sigma_local = np.median(ar) / 0.6745     # MAD -> normal-consistent sigma
else:
    sigma_local = ar.mean()
is_jump = ar > (JUMP_K * sigma_local)        # JUMP_K = 4.0
dpx = np.diff(m)
jump_move = float(dpx[is_jump].sum())
diff_move = float(dpx[~is_jump].sum())
```

Parameters: `JUMP_K = 4.0` (the Lee–Mykland default), `LOCAL_VOL_WIN = 100`
events.

**Two things a reimplementation must get right, both flagged in the code:**

1. **`LOCAL_VOL_WIN` is declared but not used as written.** The docstring
   describes a rolling estimate over the prior 100 returns; the implementation
   uses the **in-window median absolute return** as a robust proxy, because
   prior returns are not passed in. The code says so. So the effective local
   sigma is the window's own MAD, not a trailing estimate — on a short window
   that is a weak estimator.

2. **Diffusion is the residual plug, jump is the detector's output:**

```python
mko_jump = open_sign * jm * matched
mko_diff = mko - mko_jump        # plug, so jump + diff == markout EXACTLY
```

The per-event sum of steps and the as-of endpoint difference can differ by an
off-by-one at the window boundary. Defining diffusion as the remainder
guarantees reconciliation by construction. **Only the remainder is a plug; the
jump component is measured.**

### 26.4 Significance testing — day as the unit

```python
day_sum = sum(fm["markout"] for fm in res["fill_markouts"])
daily_markouts[cm][date] += day_sum
...
t_stat, t_p = stats.ttest_1samp(dm, 0.0)
w_stat, w_p = stats.wilcoxon(dm[dm != 0])
```

Both a t-test and a Wilcoxon signed-rank against zero, on a **daily portfolio
markout series** of about 207 days. The reason is stated and is the right one:

> *"Using per-fill markouts as the unit would overstate significance
> (pseudo-replication — fills within a day are correlated)."*

This is the same day-as-unit discipline the rest of the project applies, and it
is why every t-statistic quoted in Part I §20 is on symbol-days rather than
fills.

### 26.5 The clip sweep

`CLIP_MULTS = [3.0, 5.0, 7.0, 10.0]` × the trailing median trade size, on the
top-10 book (`ENGROH, LUCK, UBL, PSO, PPL, HBL, SAZEW, MLCF, ATRL, SYS`). The
range deliberately brackets the earlier capacity finding — **3× optimal, 5×
incremental too small against overnight-inventory risk** — which is why
production runs `CLIP_MULT = 3.0`.

### 26.6 What it reports

Four tables per clip multiple, per bucket: totals in PKR
(capture / markout / jump / diffusion / fees / liq / net / fills); the same in
**bps on opened notional**; per-opened-share averages with median and mean hold
time; and **mean OBI at fill** (`obi_5` and `obi_deep`), available only when
`log_equity` was on. Then the daily markout distribution with both tests and a
plain-language verdict.

---

## 27. ORDER BOOK IMBALANCE — every variant, and the weighting question

There are **four distinct imbalance quantities** in this codebase. They are
easily confused because three of them are called "OBI".

| name | where | levels | what it is |
|---|---|---|---|
| `imb` | `micro_mm.quotes` | **level 1 only** | `bq / (bq + aq)`, range `[0, 1]` — **this is the one every trading decision uses** |
| `obi_5` | `Book.obi(5)` | top 5 per side | `(Qbid − Qask) / (Qbid + Qask)`, range `[−1, +1]` — logged as a feature |
| `obi_deep` | `Book.obi(None)` | all visible levels | same formula, all levels — logged as a feature |
| `obi(include_deep=True)` | `Book.obi` | + the `__AGG_` lump | the whole book including deep residual |

### 27.1 The weighting, stated exactly

```python
bq = sum(q for _, q in sorted(bids.items(), reverse=True)[:n]) if bids else 0.0
aq = sum(q for _, q in sorted(asks.items())[:n]) if asks else 0.0
return (bq - aq) / (bq + aq) if bq + aq > 0 else None
```

**Every level inside the top `n` carries equal weight. There is no decay, no
distance weighting, no per-level coefficient.** The near-touch emphasis is
achieved by **truncation** — `obi(5)` versus `obi(None)` — not by a weight curve.

The multi-level OFI says so in its own comment, and this is the closest the code
comes to answering the decay question directly:

> `# Equal-weight multi-level OFI from a ranked depth snapshot. The scalar`
> `# increment is the SUM of the canonical increment at ranks 1..N. Leaves`
> `# depth N as an experiment axis rather than baking in a decay curve.`

So the design decision, as recorded, was: **sweep the depth N, do not fit a
decay.** `ofi_depth_levels` is validated to `1..10` and raises outside that.

### 27.2 The weighted / decay-sensitivity question — what I found, and what is still open

**What every file I have read does.** A weighted-average imbalance with more
weight on levels near the best, and a sensitivity analysis over decay constants,
is **not present in any file I have read**. I checked `mm_backtest.py`,
`micro_mm.py`, `mm_harness.py`, `universe_expand.py`,
`spot_capture_markout_decomp.py`, `run_legacy_mm.py`, `run_persistence.py`,
`persistence_metrics.py`, `run_daily_stats.py`, `ticker_stats_core.py`,
`iceberg_feasibility.py`, `leadlag_diagnose.py`, `corp_actions_master.py`,
`corp_action_detector.py` and `extreme_obi_sweep.py`. Case-insensitive searches
for `weight`, `decay`, `half_life` and `lambda` over the depth aggregation
return nothing but the spread EMA and ordinary Python lambdas.

**`extreme_obi_sweep.py` has now been read** — it was one of the four named
candidates — and it is written up in full as **§27A**. It does **not** contain a
decay sweep. **It sweeps the threshold at which the mechanism fires, not the
weight given to each level.** That is worth stating plainly, because it is a
different axis and answers a different question. The near-touch emphasis in this
codebase is still produced by **truncation** (`obi(5)` vs `obi(None)`) and by
the fact that the trading path reads **level 1 only**.

**Where a decay sweep could still be.** `build_feature_store.py` — the most
likely home, since a weighted feature would be built there rather than in the
strategy — **has now been read and does not contain one** (§37.2). The only
depth weighting in the whole codebase is the level-1 microprice. Two candidates
remain: **`expand_feature_store.py`**, and **`probe_conditional_markout.py`**,
which is the source of the 518,285-fill conditional table quoted in §27A.1 and
which I want next regardless.

**The honest reading of the evidence I do have.** Two independent places in the
code record that equal weighting was a *deliberate choice, not an oversight*:

- `Book.obi(n)` sums the top `n` levels flat, and
- the multi-level OFI comment says *"Leaves depth N as an experiment axis rather
  than baking in a decay curve."*

So the recorded design intent is **sweep N, do not fit a decay**. If a decay
sensitivity analysis was run, it was run outside these files, and I am not going
to reconstruct its results from memory of a conversation. It stays in **§34 row
1** until a file shows it.

### 27.3 Why the distinction matters for a rebuild

The **trading** path never uses a multi-level imbalance at all. Every threshold
in Part I §11 and §12 — the OBI throttle, the queue skew, the OBI-defensive
retreat, the size boost — tests `imb − 0.5` computed from **level 1 only**.
`obi_5` and `obi_deep` exist solely as **logged features** on the equity row,
for the fill-state dataset and for the "OBI at fill" table in the decomposition.

A reimplementation that wires `obi_5` into the throttle would be building a
different strategy, not this one.

### 27.4 The netting subtlety in the book

Both `obi()` and `ranked_depth()` aggregate **net** quantity per price:

```python
d[o.price] = d.get(o.price, 0.0) + o.qty
bids = {p: q for p, q in bids.items() if q > 0}
```

`__NEG_` entries carry negative quantity, so they subtract. A level that nets to
zero or below is **dropped as empty** rather than reported as a level with no
size. `__AGG_` (the residual lump beyond level 10) is excluded unless
`include_deep=True`, otherwise it would swamp the near-touch imbalance.

`ranked_depth()` mirrors these rules exactly, so the deep-OFI signal and the OBI
feature see the same book.

---

## 27A. CONDITIONAL MARKOUT BY BOOK STATE — the finding that drives everything

`extreme_obi_sweep.py`, and behind it `probe_conditional_markout.py`
(**518,285 fills, 207 days**). This is the conditional analysis: fills bucketed
by the book imbalance at the moment they happened, with capture and markout
measured **separately for the side the book was leaning against (exposed) and
the side it was leaning toward (favourable)**.

### 27A.1 The measured table

Splitting the old coarse "0.30+" bucket exposed a population the average had
been hiding:

| `obi_1` bucket | side | n fills | capture | markout | gross |
|---|---|---|---|---|---|
| 0.60–0.80 | exposed | 33,291 | +1.4381 | −0.1740 | +1.2641 |
| **0.80+** | **exposed** | **86,238** | **−0.4234** | **−0.3584** | **−0.7817** |
| 0.80+ | favourable | 158,769 | +1.4030 | +0.6785 | +2.0815 |

All in bps.

**Capture goes NEGATIVE on the exposed side above `obi_1` 0.80 — 86,238 fills,
16.6% of the book, losing 0.78 bps each.**

Negative capture is not adverse selection. Adverse selection shows up in
**markout**. Negative *capture* means **the quote was filled on the wrong side
of the mid at the instant of the fill** — the quote had been walked through the
opposite touch. It is the exact signature of the A.2 finding on the cheap-tick
names, where KEL / PIBTL / TPL went **+1.64 → −1.79 bps** the same way.

### 27A.2 The units, stated in the source

This file writes down the trap that Part I §12.6 flags, and it is worth quoting
because it is the cleanest statement of it anywhere in the codebase:

> **TWO DIFFERENT SCALES ARE IN PLAY and they differ by exactly 2×:**
> `micro_mm` fires on `(imb − 0.5)` where `imb = B/(B+A)` in `[0,1]`, so the
> trigger quantity `|imb − 0.5|` lives in `[0, 0.5]`.
> The fills store `obi_1 = (B−A)/(B+A)` in `[−1,+1]`.
> **`obi_1 = 2 × (imb − 0.5)` — an algebraic identity, not a convention.**
>
> Therefore: **the shipped `queue_skew_thresh = 0.15` IS `obi_1 = 0.30`**, and
> the probe's 0.80 boundary IS `|imb − 0.5| = 0.40`.

Every threshold in that file is in micro_mm units, with the `obi_1` equivalent
**printed beside it**, *"so a mix-up is visible in the output rather than silent
in the result."*

### 27A.3 Why a sweep was needed at all — collinearity

> *"In the fill data, 'the book is lopsided' and 'the lean is firing' are
> **COLLINEAR by construction** — the lean fires BECAUSE the book is lopsided.
> So the probe cannot tell whether extreme imbalance is intrinsically bad, or
> whether the lean is what makes it bad. **Only holding the state fixed and
> varying the mechanism separates them.**"*

That is the whole reason the sweep exists, and it is the right reason.

### 27A.4 The arms

**Read from the code, not from the file's header prose** — an earlier draft of
this document listed a `throttle_boost` and an `all` arm from the header. Neither
is built. The `ARMS` dict is constructed as follows, and this is the whole of it:

| arm | override | how many |
|---|---|---|
| `baseline` | `{}` — what ships: lean 2 ticks past `\|imb−0.5\| = 0.15`, throttle on at 0.15 to half size | 1 |
| `boost@0.10` | `size_boost_mult=1.5, size_boost_thresh=0.10` — more size on the favourable side, which earns +2.08 bps up there | 1 |
| `thr{f}@{t}` | `throttle_frac=f, obi_throttle_thresh=t` for `f ∈ {0.25, 0.50}` × `t ∈ {0.15, 0.25, 0.40}`, **skipping (0.50, 0.15) because that IS production** and a duplicate of the control costs a multiple-testing slot for nothing | 5 |
| `throttle_off` | `obi_throttle=False` | 1 |
| `leanband@{t}` | `queue_skew_thresh_hi=t` for `t ∈ {0.25, 0.40}` — **only if the engine can express it** | 2, currently **0** |

**Eight arms actually run; ten if the engine is patched.**

**The arm this document previously missed is the most interesting one.**
`throttle_off` is described in the source as:

> *"The one control never run: production has thrown this switch since before
> the fill study, and 'is the throttle earning its keep at all' has never been
> tested against not having it. **Without this arm, every throttle comparison is
> relative to a setting that was itself never validated.**"*

That is the right instinct and it generalises: **when you sweep a knob, include
the arm that removes it.** Otherwise every number is measured against a baseline
nobody ever justified.

The arms that *would* test the lean band directly:

**The prior was stated before the run, so it is falsifiable:**

> *"lean_band wins. Negative capture means the quote is crossing; cutting size
> scales that loss down but does not remove it, while switching the lean off in
> that band removes its cause."*

### 27A.5 The capability probe — and the missing engine feature

```python
HAS_BAND = "queue_skew_thresh_hi" in MicrostructureMM.__init__.__code__.co_varnames
```

**`micro_mm` has no upper bound on the lean's firing band.** The condition is
`(imb − 0.5) > thresh` with no cap. So the `lean_band` and `all` arms — the ones
testing the **stated prior** — are **skipped** unless the engine is patched.

The script probes for the kwarg rather than assuming it, *"so this script runs
usefully either way instead of dying on a TypeError deep in a backtest."*

**This is an open engine gap, and it is the one that matters most here: the
hypothesis with the strongest prior cannot currently be tested.**

### 27A.6 What the throttle sweep already established

The throttle is **already on in production** at threshold 0.15, cutting the
exposed side to half size. A correction is recorded about this:

> *"An earlier reading of micro_mm's DEFAULTS said it was off; the runner
> overrides them. That matters for what this sweep can conclude. The throttle is
> already firing across the ENTIRE region where capture goes negative, and it is
> not preventing it — halving the size did not stop the exposed side losing 0.78
> bps above obi_1 0.80. **So 'too much size' is already partly treated and did
> not work, which is evidence FOR the lean being the cause rather than the
> size.**"*

**The decomposition insight, which is subtle and easy to get wrong.**
`micro_mm` has a **single** throttle threshold. So raising it from 0.15 to 0.40
does **not** deepen the cut in the middle:

> *"it REMOVES the cut entirely between `|imb−0.5|` 0.15 and 0.40 and applies
> the deeper one only above 0.40. So that arm is really: **stop throttling at
> moderate imbalance + cut to quarter at extreme**. Those two have opposite
> signs on size and cannot be told apart from one arm."*

Hence `THROTTLE_FRACS = [0.25, 0.50]` — the 0.50 arm isolates "stop throttling
the middle" at production depth, and the gap between the two measures what the
deeper cut above the threshold adds.

`thr0.25@0.40` was the best arm in run `d86e4f22`: **+669 PKR/day, margin +0.59
bps pooled** — but it is exactly the arm that changes two things at once.

### 27A.7 The boost question — settled, and cut to one arm

> *"The five-threshold sweep answered the boost question at the MECHANISM level
> and the answer did not depend on the threshold: **every boost arm put 15–35%
> more shares at risk and every one earned essentially the SAME margin per PKR
> traded** (day-as-unit change in bps: **+0.09 to +0.29, none with |t| above
> 2**). **More volume at an unchanged rate is a capital-and-risk decision, not
> an edge**, so sweeping the threshold further only spends statistical power on
> a question already settled."*

One arm is kept as a live control, at threshold 0.10 — *"kept because it was the
strongest of the five, which makes it the hardest test of 'the boost is
inert'."* Keeping the strongest arm as the control against your own hypothesis
is the right way round.

### 27A.8 The measurement lessons in this file

These are the most transferable part of it.

**1. Fill count is blind to a size lever.**

> *"A SIZE lever changes shares per fill, not the number of fills: multiplying a
> quote by 1.5 puts more shares behind it, and an aggressor sweeping through
> produces one fill either way. `thr0.25@0.40` made this visible — it moved P&L
> by **+669 PKR/day on a 0.0% change in fill COUNT**, so the whole effect was in
> size and the fill column was blind to it."*

So the journal carries **shares and notional**, not just fills, and P&L is
reported as **bps of the PKR actually traded** — a rate, not a level.

**2. The mechanism check — did the lever even fire?**

```python
inert = abs(shares_pct_vs_baseline) < 1.0
```

> *"an arm marked NOT BINDING tested nothing. Its P&L difference comes from
> backtest path noise, not from the setting."*

This separates *"the boost did not help"* from *"the boost never fired"* — and
those two have opposite implications for what to try next. Almost nothing else
in this project does this, and it should be standard.

**3. Read the shape, not the best cell.**

> *"The point of sweeping is to see the SHAPE, not to pick the best cell — with
> this many arms the best one is partly luck. A real effect is monotone or
> single-peaked across thresholds; noise is ragged."*

**4. Multiple testing, stated with the result rather than argued after it.**

> *"Under the null that none of them does anything, the LARGEST |t| among that
> many independent tests is expected to land near **1.8–2.2** on its own. So a
> single best arm at |t| ~ 2 is what chance produces here, not evidence. What is
> NOT explained by chance: **a consistent SIGN across a family, a shape that
> tracks the threshold, and a margin that moves in the same direction as the
> P&L.**"*

**5. The pooled margin has no error bar; the daily one does.**

Pooled bps over the whole panel is dominated by the biggest-notional
symbol-days — *"a single heavy day on OGDC or CPHL can set it."* So the margin
is also computed **per day** and paired-tested, and both are printed side by
side (`bps` and `d_bps t`).

**6. Two journal-identity bugs, both found the hard way.**

> *"THE FIRST VERSION HASHED ONLY THE ARMS AND THAT WAS NOT ENOUGH. Changing the
> name sampler from alphabetical to stratified left the arms untouched, so the
> tag was unchanged, so the run RESUMED onto the old sample and **reported a
> union of eleven names under a header that said six**. The sample is part of
> the run's identity exactly as much as the arms are."*

> *"THE SCHEMA IS PART OF THE IDENTITY TOO. The journal is appended to, so a run
> that writes MORE columns than the existing file produces a ragged CSV…"*

The tag is now `sha1(arms + names + dates + column schema)`. This is the same
lesson as the engine digest in Part I §18.1 — **anything that changes what a
cell means belongs in the key** — learned independently in two places.

**7. Stratified sampling, not alphabetical.**

> *"The first version took `sorted(setting)[:n]`, which on this book returns
> AGHA, AGP, AHCL, AIRLINK, AKBL, APL — six names beginning with A, one of which
> (AKBL) swings ±7,000 PKR on a single day. **That is not a sample of the book,
> it is a sample of one letter**, and it inflates the variance on every paired
> t-statistic."*

Now: rank by assigned P&L and take every k-th name, so the sample spans big and
small contributors in proportion.

**8. The baseline must be the shipped book.**

The lean is set **per name from the assignment CSV**, not uniformly — *"A sweep
that puts every name on one uniform lean is not measuring the thing that ships."*
And the cheap-tick three are forced to plain OBI here too, *"or the baseline is
not the baseline."*

**9. Warm-up dates are dropped explicitly.**

`trailing_median` needs 10 days of history, so the first 10 dates can never
produce a cell — *"The first smoke lost one of five days to exactly this."*

### 27A.9 What this settles, and what it does not

**It settles the mechanism question.** The interesting dimension turned out to
be **where the lean fires**, not how depth is aggregated — and the shipped
single-threshold design cannot express the band that the strongest hypothesis
needs, so that hypothesis has never actually been tested.

**It does not settle the weighting question.** This file sweeps **thresholds,
not weights**. There is still no per-level weight function and no decay constant
in anything I have read, and §27.2 leaves that open with
`build_feature_store.py`, `expand_feature_store.py` and
`probe_conditional_markout.py` as the remaining candidates. The last of those is
the source of the conditional table in §27A.1, so it is the one I want next
regardless of the weighting question.

---

## 28. TRADE PERSISTENCE AND ORDER COLLAPSING — `run_persistence.py`

### 28.1 The question

> *"does a run of same-side aggressor trades predict the NEXT trade's side on
> PSX? Replicates the Aldridge FBL ladder (P(buy | 1,2,3,4 consecutive buys) =
> 46/62/69/72%) on the PSX trades table."*

This is the conditional-probability work. Pure trades-table computation — no
engine, no book.

### 28.2 The ladder

Walk each symbol-day's trades in time order keeping a consecutive-run counter.
For each trade, its **prior run length k** is how many immediately preceding
trades shared the same side. Then:

```
P(continue | run length k) = fraction of trades whose prior run was exactly k
                             that CONTINUE the run
```

```python
k = run if run < 6 else 6          # bucket 6 means "6 or more"
n_at[k]   += 1
n_cont[k] += 1 if sides[i] == run_side
```

Reported for k = 1, 2, 3, 4, 5, 6+. **Sign-agnostic** — buy runs and sell runs
are pooled by treating "same as the run" as the event.

**50% means no persistence.** A rising ladder means runs predict continuation.

### 28.3 The verdict rule

```python
VERDICT: p2[4] > p2[1] + 0.02
  True  -> "RISING -> runs predict continuation (build Table B)"
  False -> "FLAT -> no usable persistence (stop)"
```

A two-percentage-point rise from k=1 to k=4 is the bar.

### 28.4 Order collapsing — the mechanism you asked about

```
# COLLAPSE MODE: merge consecutive prints that share the SAME exchange
# timestamp AND side into ONE aggressor order (strips sweep fragmentation).
```

```python
new_order[1:] = (ts[1:] != ts[:-1]) | (side[1:] != side[:-1])
```

A single aggressive order that sweeps four price levels prints as four trades.
Counted raw, that one decision looks like a run of four and inflates every rung
of the ladder. Collapsing by **(timestamp, side)** turns it back into one
aggressor order. The run is labelled `RAW (trade prints)` or
`COLLAPSED (distinct aggressor orders)` so the two can never be confused in a
write-up.

This is the same collapse rule the run-reprice detector uses inside the strategy
(Part I §12.2): *a new distinct aggressor is counted only when the timestamp
changes or the side flips.* The two are consistent by design.

### 28.5 Two aggregation schemes, deliberately contrasted

- **Pooled** — `sum(n_cont) / sum(n_at)` across everything. Trade-count
  weighted, so busy symbol-days dominate.
- **Day-as-unit** — the mean of per-symbol-day `P`, gated at `n_at >= 20`, with
  a standard error. Every name-day counts once.

Same discipline as the markout significance test: when the two disagree, the
pooled number is being driven by a handful of busy days.

### 28.6 Parameters

| name | value |
|---|---|
| `RUN_BUCKETS` | `[1,2,3,4,5,6]` — 5 exact, 6 = six-or-more |
| `MAX_DAYS` | 20 (evenly strided across the calendar, not most-recent) |
| `WORKERS` | 6 |
| minimum trades per symbol-day | 20 |
| day-as-unit minimum `n_at` | 20 |
| verdict delta | 0.02 |
| universe | `mm_watchlist_final.csv`, fallback: all symbols traded that date |

### 28.7 Outputs

`run_persistence_counts{tag}.csv` at grain **(date, symbol, run_len)** with
columns `date, symbol, run_len, n_at, n_cont` — note `P` is computed *after* the
file is written, so it is **not** in it. And
`run_persistence_pername{tag}.csv` at grain **(symbol, run_len)** with
`P_continue`. `tag` is `_collapsed` or empty. The pooled and day-as-unit ladders
are **printed only**.

### 28.8 No PSX result is recorded

**The file contains no measured PSX outcome.** No ladder values, no verdict, no
date range. The only numbers in it are the external Aldridge benchmark
(46/62/69/72%) and three hand-verified self-test cases. Whether PSX runs are
persistent is, on the evidence I can read, **not yet recorded anywhere**.

### 28.9 One real inconsistency

The docstring says the ladder is computed "within each symbol-day's
**continuous-session** trades". The code filters only on `aggressor_side`
starting with B or S. It does **not** exclude `initiator == 'AUCTION'` and does
**not** exclude `market IN ('NDM','ODD_LOT')` — both of which the daily-stats
pipeline does exclude. If auction prints or negotiated blocks carry an aggressor
side in the store, they enter this ladder. The two pipelines disagree about what
a tradeable print is.

---

## 29. ICEBERG FEASIBILITY — `iceberg_feasibility.py`

### 29.1 What it actually is

Not an iceberg study. A **ten-second data-availability probe** whose deliverable
is a verdict on which of two detector architectures to build:

> *"can we detect icebergs by order-id, or must we fall back to
> swept-vs-visible-depth (replay)?"*

### 29.2 The detection rule, in full

```python
disp   = ob[order_id nonempty].groupby(order_id)["qty"].max()   # most it ever showed
filled = tr[resting_order_id nonempty].groupby(rid)["qty"].sum() # all it ever traded
j      = concat({disp, filled}).dropna()                         # ids in BOTH
ice    = j[j["filled"] >= 2.0 * j["disp"]]                       # the whole rule
```

**One threshold: `2.0`**, a bare literal. If displayed size is refreshed back to
the same clip, `max(displayed)` ≈ the clip, so `filled ≥ 2 × clip` means the
order was refilled at least once beyond anything ever visible.

Using `max` rather than last or initial display is **conservative** — the
largest visible size is the denominator, so it under-flags.

**What the rule does not have**, each checked: no time window, no session
filter, no inter-arrival gap, no side filter, no price filter, **no
price-continuity test** (a real iceberg detector would require replenishment at
the *same* price), no minimum order size, no minimum replenishment count. The
aggregation spans the whole day.

### 29.3 The verdict

```python
if tr_nonnull > 0.5*tr_total and ob_nonnull > 0.5*ob_total:
    "order-ids populated -> build the FAST order-id iceberg detector"
else:
    "order-ids sparse/empty -> build the swept-vs-visible (replay) detector"
```

**The verdict ignores the prevalence number entirely** — it depends only on how
populated the id columns are. A run could report 0% iceberg prevalence and still
say "build the order-id detector".

### 29.4 No conditional probability, and no recorded result

There is **no conditional-probability analysis in this file**. The three outputs
are plain ratios: share of trade rows with a populated resting id, the same for
book updates, and flagged-order fill volume as a share of matched volume — a
volume share, not a probability.

**Nothing is written to disk and no result is recorded in any comment.** No
population rate, no prevalence, no verdict outcome. On the evidence I can read,
**what this probe returned when it was run is not recorded anywhere**.

### 29.5 The load-bearing assumption

Nothing establishes that `ob_updates.qty` means *displayed* size rather than
total size. The entire method rests on that, and it is untested.

---

## 30. LEAD-LAG — `leadlag_diagnose.py` and the closure

### 30.1 What the diagnostic does

It does **not** compute correlations. It consumes `leadlag_screen.parquet` and
applies the prose acceptance rule mechanically, because:

> *"Reading 157 rows against five simultaneous conditions by eye is exactly how
> a table gets mined for its best t-statistic."*

Six boolean gates, applied as a cumulative AND so the funnel shows where the
population collapses:

```python
c1_faster    = leader_faster
c2_clears_se = |peak_lag_mean| > peak_lag_se
c3_positive  = peak_lag_mean > 0
c4_stable    = frac_leader_leads > 0.70
c5_real_peak = (peak_corr_mean > 0) & (peak_corr_mean > 2.0 * |corr0_mean|)
c6_clears_fee= ind_bps_median > 1.554
```

`EPPS_MULT = 2.0` is the Epps gate: a peak that is not at least twice the
lag-zero correlation is the non-synchronicity artefact, not a lead.

### 30.2 The multiple-testing arithmetic — the most valuable part

> *"With N pairs tested over D days, a pair whose lag sign is pure coin-flip
> still shows a perfectly stable sign (D of D) with probability 2 × 0.5^D. At
> D=5 that is 6.25%, so out of 157 pairs roughly TEN will look perfectly stable
> having no relationship at all."*

**Read the D carefully — this document previously did not.** The `D=5` in that
comment is a **worked example** matching `SMOKE_DAYS = 5`, not the production
run. The code takes D from the data:

```python
D = int(df.n_days.median())          # the run's own median days per pair
p_perfect   = 2.0 * (0.5 ** D)
exp_perfect = len(df) * p_perfect
```

The real run used `MAX_DAYS = 20`, so D is near 20 and the expected number of
perfectly-stable-by-chance pairs is **essentially zero**, not ten. **That cuts
the other way from how the example reads**: on a 20-day pair,
`frac_leader_leads = 1.00` is *not* explained by chance, and the screen's
negative conclusion rests on the **economic** gate (1 of 157 clears the fee),
not on the stability gate. The illustration is still the right piece of
machinery — it is just an illustration.

```python
p_perfect   = 2.0 * (0.5 ** D)
exp_perfect = len(df) * p_perfect
obs_perfect = (frac_leader_leads.isin([0.0, 1.0])).sum()
```

**If observed matches expected, `frac_leader_leads = 1.00` is not evidence of
anything.** Printing the expected count beside the observed one is what stops a
screen being mined.

Plus an exact one-sided binomial tail — computed with `math.comb`, not quoted —
testing whether the largest name leads more often than a coin flip. The comment
is disciplined about it: *"A prior is only supported if this is small. It is a
PRIOR, not a finding."*

### 30.3 The measured outcome (from the project closure document)

- Design: top **2** names per sector by median **traded value** — not volume,
  because "a leader on share count is a cheap stock, not an informed one".
  **157 UNORDERED (leader, follower) pairs** — the loop keys on
  `frozenset((leader, foll))` and skips a pair already measured from the other
  direction, so each pair is tested once, in one direction. (An earlier draft of
  this document called them "ordered"; they are not, and §30.2's
  multiple-testing count depends on it.) **Hayashi-Yoshida** asynchronous
  covariance across a **39-point lag grid spanning −30 s to +30 s**.
- **Result: 1 of 157 pairs clears the 1.554 bps round-trip fee**, before any of
  the other conditions. The **median anticipatable move is about 0.22 bps** —
  roughly one seventh of the fee. *"The signal is not small relative to the
  cost; it is invisible relative to the cost."*
- Correlation at the fitted peak is **not higher than at lag 0** for most pairs
  — points sit on the 45° line, the Epps signature of no lead at all.
- Fitted peak lags **spread −20 s to +20 s and centre on zero** — the shape of
  noise, not of a lead.
- The pre-registered prior, "largest traded value leads its sector", was tested
  and **not supported**.

**A correction recorded against the analysis itself:** the first chart was
captioned "a microstructure lead should be sub-second". That was wrong for this
venue — PSX is human-traded, so a genuine lead lives in **seconds**. The lag
grid at the time had only three points between 3 s and 15 s and could not have
resolved a lead where one would actually be. The grid was widened to 39 points
and re-run. **The conclusion did not change, but it was only decision-grade
after the correction.**

One design note carried in the code: `GRID_EDGE_MS = ±30 s` is **drawn but never
gated on**, deliberately — *"the LOCATION of the peak is not evidence either way
on this venue… Only a peak pinned to the grid EDGE is diagnostic, because that
means the search never found an interior maximum."*

---

## 31. HEDGING — futures, ETFs and the market index, all closed negative

Three rungs were specified and all three are dead. This section exists so they
are not rebuilt.

### 31.1 Same-ticker futures — the first rung is inverted

Crossing the **share** book is cheaper than crossing the same-ticker
active-month **future** on **99 of 100 names**, median **3.30×**. The one
exception, WTL, is a tie at 1.00×.

That inverts the stated design, whose first preference for a hedge was the
same-ticker active-month future.

**The harder result: no name can be crossed for less than the 2.67 bps gross
edge in either book.**

| | best | median |
|---|---|---|
| share book round trip | EFERT 5.30 bps = 2.0× edge | 13.79 bps = 5.2× edge |
| futures round trip | BOP 11.79 bps = 4.4× edge | 48.09 bps = 18.0× edge |

Not one name's spot spread is below 2.67 bps even **before** fees (minimum
3.75 bps, EFERT).

**Why this is structural, not a PSX quirk.** A market maker **earns** the spread
by posting. A hedger **pays** it by crossing. The edge is derived from that same
spread. So a hedge executed by crossing always costs at least what the round
trip that created the inventory earned. **A one-for-one delta hedge by crossing
cannot be profitable in any market, on any instrument.**

The design survives only through **netting** — paying the toll on net residual
exposure rather than per position. That moves netting from a nice-to-have to the
load-bearing assumption.

**Capacity at a 20% budget of the 2.67 bps edge:**

| venue | cost | max hedged as % of traded notional |
|---|---|---|
| EFERT shares (best) | 5.30 bps | 10.1% |
| BOP shares | 7.24 bps | 7.4% |
| median share book | 13.79 bps | 3.9% |
| median futures | 48.09 bps | 1.1% |

At a 50% budget: 25.2% / 18.4% / 9.7% / 2.8%. **For the median name the book
must net down to ~96% before hedging is affordable.**

Where the futures rung is worst is exactly where a hedge is most wanted — the
large caps: MCB 39.2×, NATF 28.4×, POL 28.3×, MTL 25.4×, BAHL 20.2×.

**Supporting reads:** futures width is deadness, not an artefact — Spearman
ρ = **−0.875** between trades/day and spread in bps, n=100, p=1.3e-32;
dead-time inflation is 1.01×, so the widths are real all day. Contract selection
was never in doubt (`vol_share` = 1.00 on 95 of 100 roots). Cash-settled futures
do not exist as a market: 6 trades, 1 symbol, 1 day, across 20 dates.

**Three corrections recorded against earlier claims in this workstream:**

1. *"Futures are the cheap hedge because the fee is 0.19 bps vs 1.554 spot."*
   **Wrong axis.** The fee is not the cost of a hedge; crossing is. Futures cost
   3.3× more despite the fee advantage.
2. *"The spot spread is probably ~3 bps."* Directionally right, **magnitude
   wrong by ~4×** — median spot spread is 12.24 bps.
3. *"Dead-time weighting is inflating the wide futures names."* **Wrong** —
   inflation is 1.01×. The missing control was a minimum-trade floor.

### 31.2 The tick-grid screen — arithmetic that kills a whole rung before measurement

PSX's tick is a flat 0.01 PKR, so a one-tick spread is exactly `100/P` bps.
Therefore:

- a spread below 2.67 bps requires price > **37.45 PKR**
- a round trip inside the edge including 1.554 bps of fees requires spread
  < 1.116 bps, i.e. price > **89.6 PKR**

**Any hedge instrument trading below ~90 PKR cannot have a round-trip crossing
cost inside the edge, even with a perfect one-tick book.** This is arithmetic,
not an estimate, and it is the screen to apply before measuring anything.

Several names are **tick-floored in spot** — a one-tick spread that is still
wide in bps because the price is low: WTL 78.1 bps, BECO 17.8, KOSM 17.7,
CNERGY 12.3, KEL 12.2, UNITY 8.7, FFL 5.7. For these the spread cannot be
tighter anywhere.

### 31.3 ETF hedge — dead

The tick screen kills it first. Measured anyway, R² of each name's return on the
ETF return, 465–477 names per horizon:

| horizon | median R² | 90th pct | best | median beta |
|---|---|---|---|---|
| 1 min | 0.0004 | 0.0031 | 0.0303 | 0.019 |
| 5 min | 0.0027 | 0.0176 | 0.0753 | 0.097 |
| 15 min | 0.0060 | 0.0436 | 0.1931 | 0.152 |

Seven listed ETFs appear. Only three reach any explanatory power at all —
UBLPETF 0.193, MIIETF 0.172, JSMFETF 0.137 — and **only at 15 minutes**. At one
minute the best of those three is 0.024, while the best name overall at one
minute is **REWM at 0.0303** — the table's "best" column is the best name at
that horizon, which is not the same population as the three ETFs that lead at
fifteen minutes. (Recomputed from `etf_hedge_fit_20260915.csv`: 477 names at
1 min, 468 at 5 min, 465 at 15 min; best at 15 min is UBLPETF 0.1931.)

**A hedge that needs fifteen minutes to explain a sixth of the variance is not a
hedge for a book holding inventory for seconds to minutes.**

### 31.4 Portfolio market-beta overlay — dead

Remove the common market factor at portfolio level with a tolerance band.
Measured on 500 names × 4 horizons:

| horizon | median R² | 90th pct | max | median beta |
|---|---|---|---|---|
| 5 s | 0.0001 | 0.0005 | 0.0036 | 0.022 |
| 30 s | 0.0003 | 0.0038 | 0.0243 | 0.086 |
| 60 s | 0.0007 | 0.0111 | 0.0618 | 0.110 |
| 300 s | 0.0037 | 0.0619 | 0.2644 | 0.208 |

At the horizon a market maker actually holds inventory — **seconds** — the
market factor explains about **0.01%** of a typical PSX name's return, at a
median beta of **0.02**.

**There is no common factor to hedge out. The residual *is* the position.** Beta
only becomes visible at five minutes, by which time the inventory is gone. The
tolerance band is moot as a consequence: a no-trade band exists to stop
over-trading a hedge, and there is no hedge here to over-trade.

### 31.5 What this settles for the architecture

| rung | status | why |
|---|---|---|
| same-ticker futures hedge | inverted | shares cheaper on 99/100, median 3.30× |
| futures → spot lead-lag | dead | only 13 of 100 roots trade >1,000×/day |
| basis-aware futures MM | premise undermined | all four arms lose |
| sector-leader lead-lag | dead | 1 of 157 pairs clears the fee |
| ETF hedge | dead | tick screen, and R² 0.0004 at one minute |
| portfolio beta overlay | dead | R² 0.0001 at five seconds |

**Consequence: the engine needs no cross-symbol state, no portfolio risk
aggregator and no hedge execution path. It is 113 independent quoters, each
seeing only its own book.** That is a settled design constraint, and the useful
outcome of the whole block is not a new signal but a smaller system.

**Caveats that remain open.** Every R² is a linear, contemporaneous,
unconditional fit — a relationship that only appears in a regime would not show.
None of the rungs was rejected on a marginal statistic, though; the margins are
one to two orders of magnitude. Hedge costs assume crossing at the touch with
**no size impact**, so every figure is a **floor**. And if PSX restricts
cash-market shorts, the future is not the dearer hedge on the short side — it is
the only one, and the comparison binds on the long side alone. That is an
exchange-rules question and it is still unresolved.

---

## 32. CORPORATE ACTIONS — two independent detectors, and why both exist

There are **two completely separate corporate-action detectors** in this project.
They use different inputs, different arithmetic, and catch different failures.
Part I described only the first.

### 32.1 Detector A — band versus previous close

In `sim/check_data_quality.py`, `check_bands()`. Reads the published
`xe` / `xf` circuit limits and `prev_close`, computes the flat ±10% band, and
flags any symbol off by more than a paisa as `band_vs_prev_close`. Deliberately
**not** in the `FAULTS` list, because a split legitimately breaks the identity —
it is "look at this symbol", never "the data is wrong".

### 32.2 Detector B — feed prev_close versus my own close

`corp_action_detector.py`. **It does not use the circuit band at all** — no
`limit_up`, no `limit_dn`, no `band` anywhere in the file. It compares the
feed's published `prev_close` against the close **it computed itself** from the
trade tape on the previous session it saw:

```python
ratio = feed_prev / my_close      # = price_factor
```

where `my_close` is the **last non-auction trade price**, not an official close.
The file flags this itself: *"If your feed has an official closing-price field,
prefer that instead."*

**The classification, in full:**

```python
_SPLIT_FACTORS = [10, 5, 4, 3, 2, 1.5] + reciprocals
_RATIO_TOL   = 0.02      # ±2% RELATIVE TO THE CANDIDATE FACTOR
_DIV_MAX_PCT = 0.15
_FLAG_PCT    = 0.15
```

| ratio | status | factor recorded |
|---|---|---|
| within ±2% of a round split factor | `forward_split` (f<1) / `reverse_split` (f>1) | the **raw observed ratio** |
| [0.85, 0.999) | `dividend` | 1.0 |
| [0.999, 1.001] | `normal` | 1.0 |
| (1.001, 1.15] | `minor_diff` | 1.0 |
| anything else | `review` | NaN |

**Sign convention, shared by both files:** `price_factor` is the **price
multiplier** — `old_price × factor = new scale`. Forward 2:1 → 0.5. Reverse 1:5
→ 5.0. Volumes adjust by the **reciprocal**. *"Use this number, not the word."*

### 32.3 Four defects in Detector B, all arithmetic from its own thresholds

1. **Bonus issues are structurally mislabelled as dividends with factor 1.0.**
   `ACTION_TYPES` includes `bonus`, but `_classify` has no bonus branch. A 10%
   bonus has factor `1/1.1 = 0.909`; a 15% bonus `0.870`. Both land in the
   dividend band `[0.85, 0.999)` and are recorded with **`factor = 1.0`** — so
   prices would be left unadjusted. A 20% bonus (0.833) crosses the line and
   becomes `review` with `NaN`. **For the entire 0–15% bonus range the registry
   records a factor that does nothing.**
2. **`minor_diff` is unreachable on the downside**, because `_DIV_MAX_PCT` and
   `_FLAG_PCT` are both 0.15. Any downward gap between 0.1% and 15% is
   `dividend`. And upward gaps of 0.1–15% are **never recorded at all**, because
   `minor_diff` is on the suppression list — so a reverse split or consolidation
   in that range leaves no trace anywhere.
3. **No staleness check on the baseline.** The history stores
   `{symbol: (date, close)}` and the classifier reads only the close. If a
   symbol does not trade for several sessions, today's `prev_close` is compared
   against an arbitrarily old close and the accumulated drift is attributed to a
   corporate action.
4. **`close_history.csv` is not a history.** The comment says "persist full
   close history"; the data structure is `{symbol: (date, close)}` — a
   latest-value snapshot, one row per symbol.

Also: the ±2% split window is spent partly on the definitional mismatch (last
trade vs official close), not only on real overnight movement. A genuine 2-for-1
with any real move beyond ±2% misses the split test and falls to `review`.

### 32.4 The reconciliation layer — `corp_actions_master.py`

The doctrine is the valuable part:

> **PRIMARY** = vendor/exchange announcements. Ground truth for what happened
> and when.
> **VALIDATOR** = the price-based detector. *"It cannot be the primary source —
> it is blind to symbol changes, spin-offs, and (critically) to dividends where
> the exchange does NOT adjust prev_close, which is exactly when contamination
> is silent."*
>
> Neither source alone is safe: announcements alone → a wrong or missing row is
> never caught; detector alone → silent misses on that failure mode.
> **Reconciling them turns both failure modes into a loud row in a review
> queue.**

Four verdicts:

| verdict | meaning |
|---|---|
| `CONFIRMED` | both agree — safe to apply automatically |
| `FACTOR_MISMATCH` | both fired, factors disagree — review before use |
| `SILENT_ADJUSTMENT` | announced, detector saw nothing. **The dangerous case:** the exchange did not move `prev_close`, so the overnight gap is still in the data and must be adjusted from the announcement. **Never drop these.** |
| `UNANNOUNCED_MOVE` | detector fired, nothing announced — the master is missing a row, or a real move was misclassified |

Agreement test: `|observed − announced| ≤ 0.02 × |announced|`. NaN always fails.

**Three defects here too:** dividends **always** reconcile as `CONFIRMED`
regardless of amount, because both sides carry `price_factor = 1.0` and the
announced `cash_amount` is never compared against the observed markdown.
`write_registry` deletes by **date, not (symbol, date)**, so a partial re-run
silently drops the other symbols on that date. And `effective_price_factor` —
which converts a cash dividend to a multiplicative `(P − D)/P` — is **defined,
documented and never called**, and its convention contradicts the detector
header's "dividends: additive".

### 32.5 What neither file does

**Neither file adjusts any price or quantity.** Both only compute and record a
factor. The adjustment — *"splits: price×=factor, volume/=factor; dividends:
additive"* — is delegated to a downstream consumer, and **no such consumer
exists in any file I have read.**

The backtest itself is unaffected: *"Intraday replay uses RAW prices and needs
NO adjustment (actions take effect overnight)."* The exposure is in any
**cross-day** feature — rolling vol, overnight returns, multi-day VPIN buckets,
fair-value training targets — which sees a phantom jump on the action date.

---

## 33. THE SCREENING STACK — how 500 names became 113

Three stages, run before any backtest, that produced the universe.

### 33.1 Stage 0 — `ticker_stats_core.py`, per symbol-day

**The decision metric is the paired ceiling.**

```python
net  = prev_spread - fee_rt * price          # PKR per share, at an as-of spread
q    = trades[net > 0]                       # qualifying trades
ub   = (net[net>0] * q["qty"]).sum() / 2     # halved: a round trip is two fills
pair_ratio = 2.0 * min(vb, vs) / (vb + vs)   # two-sidedness of qualifying flow
ceiling_paired = ub * pair_ratio             # THE one to rank on
```

`prev_spread` is an **as-of join** — `np.searchsorted(sp_ts, trade_ts,
side="right") - 1`, i.e. the most recent snapshot **at or before** each trade.
Note *at or* : a snapshot stamped in the same millisecond as the trade is the
one selected. That is a defensible choice on a millisecond feed, but it is not
"strictly before", which is how an earlier draft of this document described it —
and a no-lookahead claim is exactly the kind that should be stated to the tick.
Trades before the first snapshot are dropped.

**`pair_ratio` is the only genuine weighting function in the screening stack.**
It is a trade-flow balance score: 1.0 when buy- and sell-initiated qualifying
volume are equal, → 0 as flow becomes one-directional. Its purpose:

> *"a round trip needs a buy-side AND a sell-side fill. Directional sweeps
> inflate the /2 ceiling, so scale by how two-sided the qualifying flow actually
> was."*

It is a **balance** score, not a directional signal, and it is not used
predictively. Note it is computed **separately per fee level**, over qualifying
trades only.

**The fee grid** — six round-trip scenarios, `[2, 4, 10, 20, 35.45, 60]` bps,
tagged `_2p00` … `_60p00`. The current-schedule point is **derived** from
`FEE_TOTAL_PCT` rather than hardcoded, so `ceiling_pkr_rt_35p45` always agrees
with the legacy `ceiling_pkr` column.

**Note the fee basis differs from the backtest.** This file uses the **retail**
schedule: commission 0.15% plus 13% SST plus the regulatory stack = **17.727 bps
per side, 35.454 round trip**. The backtest uses the **TREC** schedule at 0.777
bps per side. Same project, two fee regimes, 22× apart — and the ranking
**inverts between them**. That is deliberate two-scenario design, but the shared
word "net" invites confusion.

**The markout ladder** — six horizons, 1/5/10/30/60/300 seconds:

```python
eff  = d * (price - m_before)         # what the aggressor paid
r{h} = d * (price - m_at(ts + h))     # what the PASSIVE side kept after h sec
net60_bps = mk60_bps - 17.727
```

*"Positive = passive side profited; negative = adverse selection."*
**`net60_bps > 0` is the viability test** — *"positive → passive MM is viable on
this symbol; negative → the fee exceeds the surviving edge and no strategy
quality can fix it."*

**The measurement caveat that matters most.** `m_at(w, tol=20000)` accepts a
quote up to **20,000 ms after** the requested time. So `mk1_bps` — nominally the
one-second markout — can be measured from a quote up to **21 seconds** later,
and `mk5_bps` up to 25. On thin names the 1/5/10-second rungs can collapse onto
nearly the same quote. Since the whole short-horizon case rests on the
`mk1 4.5 → mk60 2.0` decay and the `net1/net5/net10 = 3.76/3.13/2.66` ramp,
**this tolerance is the single biggest caveat on those numbers.** It is not a
module constant and is not swept.

Two smaller ones: auction trades leak into `trades`, `volume_sh` and
`notional_m` while everything else excludes them, so `pct_vol_qual` has a
continuous-only numerator over an auction-inclusive denominator. And
`pct_time_wide` is the share of **snapshot messages** above 30 bps, not of
elapsed time — messages arrive more densely when the book is active, so quiet
wide stretches are under-weighted despite the name.

### 33.2 Stage 1 — `run_daily_stats.py`, every symbol every day

Orchestration only; it computes no metric of its own beyond `segment` and
`n_segments`. It lands one row per `(symbol, date)`.

**The one substantive rule is the trade filter:**

```sql
WHERE symbol IS NOT NULL AND market NOT IN ('NDM', 'ODD_LOT')
```

> *"NDM is the Negotiated Deals Market: bilaterally agreed blocks reported to
> the exchange, with no book behind them — **averaging ~300× a REG trade**, so
> one block can dominate a day's volume-weighted ceiling with flow a market
> maker could never capture."*

And it must be an **exclusion, not `market='REG'`**, because `STOCK_DEL_FUT` and
`STOCK_CS_FUT` are also `market` values — an equality filter would delete all
futures.

**A dead-code warning:** `skip_reason()` classifies an unscreenable symbol-day
(`no_trades`, `no_l1_book`, `one_sided_bid_only`, `auction_only`, …) and is
**never called**. Nothing writes `skips_ALL.parquet`. So Stage 2's
"unscreenable day reasons" block can never fire, and `days_unscreenable` can
only be inferred arithmetically, never attributed.

### 33.3 Stage 2 — `persistence_metrics.py`, turning 207 days into a watchlist

> *"A single day is not rankable: **KTML swung 236× in ceiling between two
> sessions**, so anything ranked on one date is noise. This script ranks on
> PERSISTENCE instead."*

**The headline metric:**

```python
rank_in_day = groupby(["date","segment"])[ceiling].rank(ascending=False)
is_material = ceiling >= MATERIAL_PKR[fee_tag]     # 30,000 at 2bps; 5,000 at 35bps
in_top_n    = (rank_in_day <= 20) & is_material
pct_days_top20 = n_days_top20 / days_traded
```

**Four corrections are baked into that one line, and each one is a lesson:**

1. **The materiality gate.** A rank threshold ignores magnitude: an episodic
   name worth ~1,300 PKR outranks every dead name and scores 1.000, identical to
   a stable name worth 100,000. **Verified on planted archetypes** — rank-only
   scored a stable and an episodic name **both at 1.000**; adding the money gate
   separated them to **1.000 vs 0.085**, recovering the **8% spike rate that was
   planted**. Top-N **AND** ≥ money.
2. **The denominator is days TRADED, not days present.** Unscreenable days —
   limit-locked, one-sided, auction-only — **vanish** from daily stats rather
   than counting as zero. *"DMC 2025-09-23 was bid-only at the +10% cap all
   session."* Using days-present would flatter exactly the thin, volatile names
   most likely to be locked.
3. **Session regime is tagged, not scaled.** Ramadan 2026 ran ~4.2 h weekdays
   and ~3.2 h Fridays against ~6.0 h normal; a normal Friday has a ~152-minute
   Jumu'ah break (7.22 h elapsed, 4.68 h traded). A daily ceiling is a total, so
   pooling regimes inflates dispersion for reasons unrelated to the question.
   **Deliberately not normalised per hour:** opportunity is not linear in
   session length, and **measured Friday intensity is ~24% HIGHER per traded
   hour**. *"Tag, do not scale."* Within-day metrics are immune anyway — they
   compare symbols on the same date.
4. **Segments are split.** Ready equities and single-stock futures are different
   instruments; one leaderboard across them is meaningless.

**Leaderboard stability** is tested by `rank_autocorr` — Spearman between
day *t* and day *t+5* ranks, over symbols present on both, skipped below 30
overlapping names. The interpretation is stated: *"High → a standing watchlist
works. Low → attractiveness rotates, and the episodic population needs an event
trigger rather than a standing quote."*

**Session calendar construction** deserves a note, because it is where the four
day types actually come from. Per date: `traded_h = elapsed − max_gap`, i.e. it
subtracts **only the single largest gap** (the Jumu'ah break), and the regime is
a four-way classification into `ramadan_friday` / `ramadan` / `friday` /
`normal` from measured Ramadan bounds `2026-02-19 … 2026-03-19` — *"taken from
MEASURED session times, not a calendar guess."* `day_idx` is a dense
**trading-day** index, so lag 5 means five trading days, not five calendar days.

Other recorded findings: markout decays with hold time (**mk1 ~4.5 → mk60 ~2.0
bps**), so 60 s understates a fast strategy's edge; the net ramp at TREC fees is
smooth (**3.76 / 3.13 / 2.66** at 1/5/10 s), which is why **mk5** was chosen as
the working horizon; and **ranking inverts between the 2 bps and 35 bps fee
scenarios.** `MIN_DAYS = 100` traded days before a symbol is scored at all.

---

## 34. WHAT REMAINS UNVERIFIED

Everything below is a gap in what I could read, not a conclusion.

| # | gap | what would close it |
|---|---|---|
| 1 | **The weighted / decay-sensitivity OBI work.** Every file read is equal-weight. `extreme_obi_sweep.py` sweeps **thresholds, not weights** (§27A) and `build_feature_store.py` holds no per-level decay either (§37.2) — so on the evidence now in hand, **no per-level decay exists in this codebase**, and the OFI comment says that was deliberate | `expand_feature_store.py`, `probe_conditional_markout.py` — the last two places it could hide |
| 1b | **`probe_conditional_markout.py`** — the 518,285-fill / 207-day conditional markout table in §27A.1 is quoted from the header of `extreme_obi_sweep.py`, not verified against the script that produced it | the file, and its output CSV |
| 1c | **`micro_mm` has no `queue_skew_thresh_hi`.** The lean has a lower bound and no upper bound, so the `lean_band` and `all` arms — the ones testing the strongest stated prior in §27A.4 — are skipped at runtime and **have never run** | add the kwarg to `MicrostructureMM.__init__`, then re-run the sweep |
| 2 | ~~`leadlag_screen.py`~~ — **CLOSED in §37.3.** Read in full, and its 157-row output parquet opened directly | — |
| 3 | **`probe_etf_hedge.py` / `probe_market_beta.py`** — the R² tables are quoted from the closure document, not verified against code | the files |
| 4 | ~~`config_pk.py`~~ — **READ, and the hazard is CONFIRMED, not hypothetical** (§37.1). `build_feature_store.py` hardcodes `/Users/shazzak/Capital Stake - Parsed`, which is **not** `config_pk.PARSED_ROOT` (`/Users/shazzak/HFT Data/Pakistan/Capital Stake - Parsed`). `universe_expand.py` likewise rebinds `R.PARSED_ROOT`. Nothing reconciles them | make every script import `config_pk` and print both roots at start-up |
| 5 | **`snapshot_prep.prep_snapshot`** — the pre-parsed snapshot object the whole book reconstruction consumes | the file |
| 6 | **`expansion_names.py`** — `ALL_NAMES`, `INCUMBENT`, `safe_out` | the file |
| 7 | **Whether `latency_ms=120` or `latency_model` binds** — both are in the effective config | read `Backtester.__init__`'s handling of `cfg["latency_ms"]` |
| 8 | **The `log_fill_state` dataset has never been produced.** It remains the single most important unaddressed item for believing the P&L, because the whole edge is capture-driven and that is exactly what a passive-fill assumption flatters | run it |
| 9 | **No measured PSX result is recorded for the persistence ladder or the iceberg probe.** Both scripts exist and neither records what it returned | run them and record the output |
| 10 | `Pakistan/Docs` is not a folder I can reach | connect it or attach its contents |

---

## 35. THE LESSONS, SEPARATED FROM THE MECHANISMS

If the code is ever rebuilt and only one page survives, it should be this one.

**On measurement**

1. **A diagnostic is not a plug — and this project states the rule but does not
   fully follow it.** In `spot_capture_markout_decomp.py` the unexplained
   remainder is spread pro-rata by notional into `liq_loss` and then added into
   `net`, so `net` reconciles to engine P&L by construction while the file's own
   comment claims it does not. It is itemised rather than hidden, which is much
   better than a silent plug — but **the moment a residual is added back, the
   identity stops being evidence, and the only number worth reading is the
   residual column itself.** Assert it; do not print it. (§26.2)
2. **Reconcile to the cent, and abort if you cannot.** Every anchor in this
   project is an equality check with a tolerance of 1e-4 PKR and a hard stop.
3. **The day is the unit, not the fill.** Fills within a day are correlated;
   using them as independent observations manufactures significance.
4. **Effect size, not t.** `t = d·√n` grows with sample size at a fixed effect,
   so a band written in t means different things on 98 days and 197. The config
   assignment uses `d = mean/sd` for exactly this reason.
5. **Print what chance alone would produce, beside what you observed.** The
   lead-lag screen's `2 × 0.5^D` calculation is what stops 157 rows being mined.
6. **Plant an archetype and check the metric recovers it.** The materiality gate
   was validated by planting a known 8% spike rate and confirming it came back.

**On modelling a venue**

7. **Read the rulebook, do not infer it.** The post-only rejection survived for
   months because nobody checked 8.5.1 and 8.9(b) for an instruction that does
   not exist.
8. **Read published values, never reconstruct them.** A ±10% band computed from
   the previous close is wrong by the split ratio on precisely the day it
   matters.
9. **A simplification that errs toward profit is the worst kind.** Deleting
   crossing fills removed adverse selection and flattered P&L.
10. **Round away from aggression.** Floor the bid, ceil the ask — never let
    arithmetic make a quote better than intended.

**On building the thing**

11. **Hash what actually ran.** The journal tag covers calibration, arms and now
    the engine, because a cell computed under a different engine is
    indistinguishable otherwise and reconciles perfectly while being wrong.
12. **A guard that is never installed is not a guard.** `PriceBandCheck` was
    written, tested, and wired to a dead input for weeks.
13. **A counter that only exists sometimes is a footgun.** Declare them at
    construction.
14. **Two engines agreeing is not evidence they are right.** The gate proved
    reproduction; it could not have caught a shared misreading of the venue, and
    twice it did not.
15. **The aggregate can be safe while every per-name number moves.** −0.27% in
    total, 6 of 12 names re-ranked, one crossing a selection threshold.

**On knowing what you know** (added in Part III. Every one of these was learned
by getting it wrong in an earlier draft of this document — the full register is
§37.4)

16. **A comment is evidence of intent, never of behaviour.** This is the single
    most common error found: four separate claims in this document came from a
    source file's own comment while the code did something else. The liquidation
    residual is documented as "never added into net_pnl" and is added into
    net_pnl. The decomposition is documented as asserted per bucket and is
    printed, not asserted. The sweep's header lists arms the `ARMS` dict does not
    build. **Read the code; quote the comment only as intent.** (§37.4)
17. **A plausible explanation for a gap is worse than admitting the gap.** Faced
    with 176 nominal pairs and 157 rows, an earlier draft invented a
    reconciliation. There was no gap — the code enumerates 157 — and the
    invented reason looked like understanding and would have survived review.
    **When a count does not match, find the code that produces it.** (§37.4 #2)
18. **A to-do is a record of intention, not of state.** "`leadlag_screen.py` has
    never been run" was carried forward from an old to-do into §20.6 of this
    document, while the screen's 157-row output file sat in the same folder as
    the script. **Check a claim about state against an artefact, not against a
    note.** (§37.4)
19. **A coarse grid in the band of interest can depress the very statistics used
    to declare a result noise.** The lead-lag grid had three points between 3 s
    and 15 s, which is exactly where a human-speed lead would live; that
    manufactures sign instability and inflates the standard error — two of the
    gates the screen failed on. The conclusion survived the re-run at 39 points,
    but it was only decision-grade afterwards. **Before accepting a negative,
    confirm the measurement could have seen the effect.** (§37.3)
20. **Check the arithmetic, not the comment.** `ALPHA_FAST = 0.1` is documented
    as a "~10-trade half-life"; the half-life is 6.58 and 10 is the mean lag.
    Harmless here, and exactly the kind of thing that is not harmless somewhere
    else. (§37.2)
21. **Two roots are one root too many.** A hardcoded path that shadows
    `config_pk` fails loudly when the directory is missing and silently when it
    is merely stale. **Import the root, and print it.** (§37.1)
22. **A correction applied in one section is not applied.** §24.2 corrected the
    unfilled haircut from 3% to 10%; the §22 defect register — the page a
    rebuilder actually works from — still said 3% two passes later. **Sweep a
    correction through every place the number appears, and keep an index of
    where numbers live so that sweep is mechanical.** That is what §36 is for.
23. **Recompute a derived number before repeating it.** "91.7% of the per-name
    ceiling" came from a project to-do and is wrong; the two figures it is
    derived from are both correct and give **98.8%**. A ratio quoted from a
    document is not a measurement. (§37.4 #8) A hardcoded path that shadows
    `config_pk` fails loudly when the directory is missing and silently when it
    is merely stale. **Import the root, and print it.** (§37.1)

---

## 36. EVERY NUMBER IN THIS DOCUMENT, WITH ITS SOURCE

One page. If a figure is quoted anywhere downstream of this document, this is
where to check what it came from and whether it was measured or assumed.

**How to read the "basis" column:**

- **code** — a literal or an expression read directly out of a source file.
- **measured** — an output of a run recorded in a project document or a run log.
- **derived** — arithmetic on the two above, done in this document and shown.
- **assumed** — a value chosen, with no measurement behind it. Every one of
  these is a place a rebuild can go wrong quietly.

### 36.1 Money, fees and the edge

| quantity | value | basis | § |
|---|---|---|---|
| TREC fee, one side | `FEE_TOTAL_TREC = 0.0000777` → **0.777 bps** | code | 4 |
| TREC fee, round trip | **1.554 bps** | derived | 4 |
| Retail fee, one side (screening only) | **17.727 bps** | code | 33.1 |
| Retail fee, round trip (screening only) | **35.454 bps** | code | 33.1 |
| Gross edge used in every hedge screen | **2.67 bps** | measured | 31.1 |
| Fee-scenario grid in screening | `[2, 4, 10, 20, 35.45, 60]` bps | code | 33.1 |
| PSX tick | flat **0.01 PKR** | code | 31.2 |
| Price above which a 1-tick spread < 2.67 bps | **37.45 PKR** | derived | 31.2 |
| Price above which a 1-tick round trip fits inside the edge | **89.6 PKR** | derived | 31.2 |

### 36.2 Latency

| quantity | value | basis | § |
|---|---|---|---|
| decision | **5.0 ms** | code | 3 |
| wire out, median | **40.0 ms** | code | 3 |
| wire out, tail | **400.0 ms** | code | 3 |
| wire in, median | **40.0 ms** | code | 3 |
| wire in, tail | **10.0 ms** | code | 3 |
| tail probability | **0.02** | code | 3 |
| seed | **0** (`LATENCY_SEED`) | code | 3, 24 |
| the unresolved one | `latency_ms = 120` sits in the same config as `latency_model` | code | 24, 34 |

### 36.3 The lean, and the units trap

| quantity | value | basis | § |
|---|---|---|---|
| shipped trigger, micro_mm units | `queue_skew_thresh = 0.15` | code | 11 |
| the identity | `obi_1 = 2 × (imb − 0.5)` | derived, stated in source | 12.6, 27A.2 |
| shipped trigger, fill-data units | **`obi_1 = 0.30`** | derived | 27A.2 |
| second bucket | `0.20` → `obi_1 = 0.40` | derived | 19 |
| tick depth | `queue_skew_ticks = 2.0` | code | 11 |
| bps variant | `queue_skew_bps = 2.0` | code | 11 |
| the probe boundary | `obi_1 = 0.80` → `\|imb − 0.5\| = 0.40` | derived | 27A.2 |

### 36.4 The conditional markout table

518,285 fills, 207 days, all bps. **measured**, §27A.1.

| `obi_1` | side | n | capture | markout | gross |
|---|---|---|---|---|---|
| 0.60–0.80 | exposed | 33,291 | +1.4381 | −0.1740 | +1.2641 |
| 0.80+ | exposed | **86,238** | **−0.4234** | −0.3584 | **−0.7817** |
| 0.80+ | favourable | 158,769 | +1.4030 | +0.6785 | +2.0815 |

86,238 of 518,285 = **16.6%** of the book (derived).

### 36.5 The three quoting groups and the assignment

| quantity | value | basis | § |
|---|---|---|---|
| QT_2t full year | **8,736,780 PKR** | measured | 20.1 |
| QBPS_2 full year | **8,278,591 PKR** | measured | 20.1 |
| QT_2t over OBI, paired | **+1.3036 bps/day, t = +9.42** | measured | 20.1 |
| 0→2 ticks | markout **+1.08** for capture **−0.15** | measured | 20.1 |
| 2→3 ticks | markout **+0.05** for capture **−1.21**, t = **−26** | measured | 20.1 |
| graded staircases | **−1.93 bps, t = −34** | measured | 20.1 |
| cheap-tick three (KEL, PIBTL, TPL) | lean off wins, paired t = **+4.61**; capture **+1.64 → −1.79 bps** | measured | 20.1 |
| walk-forward, two buckets | **+5.45%**, se 0.94, **t = +5.83**, 10/10 folds | measured | 20.2 |
| walk-forward, three buckets | **+6.41%**, se 1.22, **t = +5.25**, 10/10 folds | measured | 20.2 |
| the third bucket alone | **+0.96pp**, se 0.58, **t = +1.64**, 8/10 | measured | 20.2 |
| shipped split | **68 / 13 / 17 / 15 = 113** | measured | 19, 20.2 |
| in-sample total | **15,366,558 PKR** vs 13,285,941 flat-0.15 = **+15.7%** | measured | 20.2 |
| share of per-name ceiling | **98.8%** of 15,555,884 — the project to-do's 91.7% is wrong | **recomputed from the assignment CSV** | 20.2 |
| best-of-three ceiling, no DROP option | 14,539,731 — the assignment **beats** it | recomputed | 20.2 |
| the assignment band | `D_BAND = 2.0/√98 = 0.202` — an **effect size**, not a t | derived | 19 |

### 36.6 The reconcile gate

| quantity | value | basis | § |
|---|---|---|---|
| scope | 12 names × 20 dates = **240 symbol-days**, June 2026 | measured | 20.3 |
| result | **240/240 exact, 0.00 PKR**, exact at record level | measured | 20.3 |
| before the two venue corrections | **30,785.24 PKR** | measured | 20.4 |
| after | **30,703.16 PKR** | measured | 20.4 |
| net cost of correctness | **−82.08 PKR = −0.27%** | derived | 20.4 |
| days that moved | **224 of 240** (103 down −7,646; 121 up +7,564) | measured | 20.4 |
| per-day sd of the change | **106.28 PKR** against mean daily ≈ 128 | measured | 20.4 |
| names re-ranked | **6 of 12**; NCPL t 2.16 → **1.92** | measured | 20.4 |
| house band used by the gate | `HOUSE_BAND_PCT = 25.0` in `sim/gate.py` | code (**assumed** value; not discussed elsewhere in this document) | 36.13 |

### 36.7 Queue position

| policy | total PKR | vs baseline | better on | t | basis |
|---|---|---|---|---|---|
| amend up (shipped) | 30,703.16 | — | — | — | measured |
| second order | 30,688.36 | **−14.81** | 55/240 | **−0.03** | measured |

*(The differences are taken from the unrounded run totals. The two-decimal
figures above differ by 14.80; both are the same number, printed at different
precision. Same for the capture/markout row in §27A.1, where −0.4234 and
−0.3584 print as −0.7817 rather than −0.7818.)*
| don't top up | 30,042.98 | **−660.18** | 74/240 | **−0.92** | measured |

Landed amendments **75,229**; **69,681 (92.6%) lose place**; **62,918 (83.6% of
all landed)** are price moves that re-queue under 8.5.2 regardless of policy.
All measured, §20.5.

### 36.8 Cross-asset — every rung, every number

| quantity | value | basis | § |
|---|---|---|---|
| futures roots with >1,000 trades/day | **13 of 100** | measured | 20.6 |
| futures deadness ↔ spread | Spearman **ρ = −0.875**, n=100, p=1.3e-32 | measured | 31.1 |
| dead-time inflation | **1.01×** (i.e. not an artefact) | measured | 31.1 |
| shares cheaper than same-ticker future | **99 of 100 names, median 3.30×** | measured | 31.1 |
| best share round trip | EFERT **5.30 bps = 2.0×** edge | measured | 31.1 |
| median share round trip | **13.79 bps = 5.2×** edge | measured | 31.1 |
| best futures round trip | BOP **11.79 bps = 4.4×** edge | measured | 31.1 |
| median futures round trip | **48.09 bps = 18.0×** edge | measured | 31.1 |
| tightest spot spread anywhere | **3.75 bps (EFERT)** — still above the 2.67 edge | measured | 31.1 |
| hedgeable share of notional, 20% budget | EFERT 10.1% / BOP 7.4% / median 3.9% / futures 1.1% | derived | 31.1 |
| netting required for the median name | **~96%** | derived | 31.1 |
| ETF R², 1 min | median **0.0004**, best 0.0303 | measured | 31.3 |
| ETF R², 15 min | median **0.0060**, best **0.1931** (UBLPETF) | measured | 31.3 |
| market beta R², 5 s | median **0.0001**, median beta **0.022** | measured | 31.4 |
| market beta R², 300 s | median 0.0037, max 0.2644 | measured | 31.4 |
| lead-lag pairs tested | **157 unordered** pairs, 39-point grid, ±30 s, Hayashi-Yoshida | **verified: enumerating the code's own loop over `SECTORS` gives exactly 157, and the output parquet holds 157 rows** | 30.3, 37.3 |
| pairs clearing the fee | **1 of 157** | measured | 30.3 |
| median anticipatable move | **≈0.22 bps** ≈ 1/7 of the fee | measured | 30.3 |
| the multiple-testing correction | `2 × 0.5^D`, **D = median days per pair, taken from the data**. The source comment's "D=5 → 6.25% → ~10 of 157" is a worked example at `SMOKE_DAYS`; the real run had `MAX_DAYS = 20`, where the expected count is ≈0 | code | 30.2 |
| Epps gate | `EPPS_MULT = 2.0` | code | 30.1 |
| cash-settled futures | **6 trades, 1 symbol, 1 day, across 20 dates** | measured | 20.6, 31.1 |

### 36.9 The decomposition

| quantity | value | basis | § |
|---|---|---|---|
| the identity, `universe_expand.py` | `net = capture + markout + liq_cap + liq_mko − fee − liq_fee` — every term measured | code | 26.1 |
| the identity, `spot_capture_markout_decomp.py` | `net = capture + markout − fee + recon_gap` — the residual **is** added back | code | 26.2 |
| is it asserted? | **no** — printed, not tested; the word `assert` appears only in a header comment | code | 26.1 |
| jump threshold | `JUMP_K = 4.0` (Lee–Mykland default) | code | 26.3 |
| local vol window | `LOCAL_VOL_WIN = 100` — **declared, not used as written** | code | 26.3 |
| robust sigma | `median(\|r\|) / 0.6745` | code | 26.3 |
| which component is the plug | **diffusion**; jump is measured | code | 26.3 |
| significance unit | **the day**, ~207 days, t-test + Wilcoxon | code | 26.4 |
| clip sweep | `CLIP_MULTS = [3.0, 5.0, 7.0, 10.0]` | code | 26.5 |
| production clip | `CLIP_MULT = 3.0` | code | 26.5 |
| unfilled haircut | `unfilled_haircut_pct = 0.10` | code | 24 |
| POV | `unwind_pov = 0.10` — live in **one** of three places | code | 12A |

### 36.10 Screening

| quantity | value | basis | § |
|---|---|---|---|
| universe | 500 names → **113** | measured | 33 |
| minimum history to be scored | `MIN_DAYS = 100` traded days | code | 33.3 |
| materiality gate | **30,000 PKR** at 2 bps; **5,000** at 35 bps | code | 33.3 |
| top-N rank gate | **20** | code | 33.3 |
| planted-archetype validation | rank-only 1.000 / 1.000 → with money gate **1.000 / 0.085**, recovering a planted **8%** spike rate | measured | 33.3 |
| why one day cannot rank | **KTML swung 236×** between two sessions | measured | 33.3 |
| NDM block size | **~300× a REG trade** | measured | 33.2 |
| markout decay | **mk1 ≈ 4.5 → mk60 ≈ 2.0 bps** | measured | 33.3 |
| net ramp at TREC fees | **3.76 / 3.13 / 2.66** at 1/5/10 s → **mk5 chosen** | measured | 33.3 |
| the caveat on all of those | `m_at(tol=20000)` — a "1-second" markout can be measured up to **21 s** later | code | 33.1 |
| stability lag | **5 trading days**, Spearman, ≥30 overlapping names | code | 33.3 |
| Ramadan bounds | **2026-02-19 … 2026-03-19**, measured not assumed | code | 33.3 |
| session lengths | ~6.0 h normal; Ramadan ~4.2 h; Ramadan Friday ~3.2 h; normal Friday 7.22 h elapsed / 4.68 h traded (~152-min break) | measured | 33.3 |
| Friday intensity | **~24% higher per traded hour** — the reason regimes are tagged, not scaled | measured | 33.3 |
| rank inversion | ranking **inverts** between the 2 bps and 35 bps fee scenarios | measured | 33.3 |

### 36.11 Iceberg

| quantity | value | basis | § |
|---|---|---|---|
| the entire detection rule | `filled ≥ 2.0 × max(displayed)` | code | 29.2 |
| id-population verdict threshold | **>50%** on both trade and book id columns | code | 29.3 |
| recorded result | **none — nothing is written to disk and no run output is recorded anywhere** | — | 29.4 |

### 36.12 Data quality

| quantity | value | basis | § |
|---|---|---|---|
| crossed books found | **285** on NRL and MLCF over three days | measured | 20.7 |
| parsed store end date | **2026-06-30** | measured | 20.7 |
| the dual-listing bug | MLCF in `REG` and `EQ_SQUARE_UP`; fixed by `market="REG"` on the **snapshot** read only | code | 20.7 |
| the residue of that fix | the trade-stats pre-pass is **not** filtered, so a dual-listed name's clip is still sized from both markets | code | 20.7 |

### 36.13 The numbers that are assumed, gathered deliberately

Every one of these is a chosen value with nothing measured behind it. They are
the first place to look when a rebuild disagrees with this one.

| value | where | what choosing differently would do |
|---|---|---|
| `HOUSE_BAND_PCT = 25.0` | gate | it is a gate convenience, **not a risk limit**. A live house band must be set on purpose |
| `2.0` iceberg multiple | `iceberg_feasibility.py` | the whole prevalence figure; a bare literal with no sensitivity |
| `0.5` id-population threshold | same | which detector architecture gets built |
| `JUMP_K = 4.0` | decomposition | the jump/diffusion split, though it is the published default |
| `tol = 20000 ms` | screening markouts | the short-horizon markout ladder, which is the case for a fast strategy |
| `CLIP_MULT = 3.0` | production | capacity, bounded by an earlier finding but not re-swept |
| `unfilled_haircut_pct = 0.10` | runner config | the residual mark, and therefore end-of-day P&L on every unfilled position |
| `EPPS_MULT = 2.0` | lead-lag | how aggressively the non-synchronicity artefact is filtered |

### 36.14 What has never been run

Stated once, so nothing downstream reads a gap as a result.

1. The **`log_fill_state` dataset** — §34 row 8. The largest single threat to
   believing the P&L, because the edge is capture-driven.
2. ~~`leadlag_screen.py`~~ — **this entry was wrong and is withdrawn.** The
   screen has been run: 20 days, 157 pairs, output parquet verified in §37.3.
   What has never been run is an *engine test* of a throttle built on it, and
   nothing justifies one at 1 of 157 pairs clearing the fee. See §37.4.
3. The **persistence ladder** and the **iceberg probe** — both exist, neither
   records an output.
4. The **`leanband@0.25` and `leanband@0.40` arms** of the extreme-OBI sweep —
   dropped at runtime because `micro_mm` has no `queue_skew_thresh_hi`. These
   are the arms testing the strongest stated prior in the project. (There is no
   `all` arm; an earlier draft of this document invented one from the file's
   header prose.)
5. The **113-name re-run under the corrected engine** — the configuration
   assignment in §19 and §20.2 was produced under the pre-correction engine.

### 36.15 Part III additions

| quantity | value | basis | § |
|---|---|---|---|
| `config_pk.PARSED_ROOT` | `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Parsed` | code | 37.1 |
| `config_pk.RESULTS_ROOT` | `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results` | code | 37.1 |
| the root `build_feature_store.py` hardcodes instead | `/Users/shazzak/Capital Stake - Parsed` — **different, unreconciled** | code | 37.1 |
| `SESSION_SCALE` | PPL 7.6 / UBL 3.9 / PACE 46.15 | code | 37.1 |
| feature-store label horizons | `[1000, 5000, 30000]` ms, **5 s primary** | code | 37.2 |
| flow window | `FLOW_WINDOW = 50` trades | code | 37.2 |
| signed-flow EWMA | `ALPHA_FAST = 0.1` → half-life **6.58** trades (the source comment's "10" is the mean lag `1/α`) | derived | 37.2 |
| realized-vol EWMA | `VOL_ALPHA = 0.05` → half-life **13.5**; matches `micro_mm.vol_alpha` exactly | derived, verified | 37.2 |
| spread-Z window | `SPREAD_Z_WINDOW = 200`, NaN below **20** samples | code | 37.2 |
| VPIN bucket | `1/50` of running daily volume — flagged as a tuning knob, not swept | code (**assumed**) | 37.2 |
| feature-store symbols | **38** (the third and only live `SYMBOLS` assignment) | code | 37.2 |
| lead-lag sector map | **107 names / 19 screenable sectors + 6 singletons = 113** | code, counted | 37.3 |
| sector-map source | PSX daily quotation sheet, Section 4, **2026-09-14**; all 113 matched, no UNCLASSIFIED | code | 37.3 |
| lag grid | **39 points**, −30 s … +30 s (was 19) | code, counted | 37.3 |
| lead-lag run cost | 20-day pass ≈ **7 minutes** | measured | 37.3 |
| lead-lag screen output | **157 rows**, read from the parquet footer | **verified from the output file** | 37.3 |
| nominal pair count | 2 leaders × followers = **176**; 19 produced no row | derived | 37.3 |
| `MIN_PAIR_DAYS` / `MIN_TRADES` / `MAX_DAYS` / `N_LEADERS` | 3 / 100 / 20 / 2 | code | 37.3 |

---

# PART III — ADDED 2026-09-18 (third pass)

Three more files were located and read in full: **`config_pk.py`**,
**`build_feature_store.py`** and **`leadlag_screen.py`** — all three of which
earlier sections of this document list as gaps. The screen's own **output
parquet** was also opened directly.

**This pass corrects two statements made earlier in this document.** They are
set out in §37.4 rather than quietly edited, because a document that silently
fixes itself teaches nothing.

---

## 37. THE THREE FILES THAT CLOSED THE REMAINING GAPS

### 37.1 `config_pk.py` — the canonical paths, and a live hazard confirmed

Fifty-five lines. It is the single source of truth for every path, and it names
its own purpose: *"Import from here instead of hardcoding strings at the top of
each script. When you move machines or reorganise, change paths HERE only."*

| constant | value |
|---|---|
| `PARSED_ROOT` | `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Parsed` |
| `RESULTS_ROOT` | `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results` |
| `PROJECT_ROOT` | `/Users/shazzak/PycharmProjects/HFT` |
| `FEATURE_STORE` | `RESULTS_ROOT / "feature_store"` |
| `FILLS_DIR` | `RESULTS_ROOT / "fill_attribution" / "fills"` |
| `FILLS_RAW_DIR` | `RESULTS_ROOT / "fills"` |
| `EOD_POSITIONS`, `PNL_SUMMARY`, `CONFIRM_CSV`, `WATCHLIST`, `EXPORT_DIR`, `DIAGNOSTICS` | all under `RESULTS_ROOT` |
| `PARSED` | alias of `PARSED_ROOT`, *"under the name run_legacy_mm / mm_harness expect to SET"* |
| `DEV_SYMBOLS` | `["PPL", "UBL"]` |
| `LOCK_SYMBOL` | `"PACE"` — *"thin/locky third name (measured; exercises the EOD/lock triggers)"* |
| `SESSION_SCALE` | `{"PPL": 7.6, "UBL": 3.9, "PACE": 46.15}` — back-solved so skew at max inventory ≈ 1× median spread |

**The two fills folders are not interchangeable, and the file says so.**
`FILLS_DIR` is the **queue-position attribution** set and is the source of truth
for net P&L. `FILLS_RAW_DIR` is the optimistic *"counterparty to every trade"*
set, kept for reference and explicitly **not for net-P&L analysis**. A rebuild
that reads the raw folder gets a materially better-looking P&L with no error
raised anywhere. The file even says to confirm which is which from the schema
probe before trusting either.

**The path hazard flagged in §34 is now confirmed concretely, not just
suspected.** `build_feature_store.py` opens with:

```python
PARSED_ROOT  = Path("/Users/shazzak/Capital Stake - Parsed")
R.PARSED_ROOT = PARSED_ROOT                     # override, this process only
RESULTS_ROOT = Path("/Users/shazzak/Capital Stake - Results")
```

Those are **not** the `config_pk.py` paths — the `HFT Data/Pakistan/` segment is
missing from both. So the feature store and the backtests can be reading and
writing under **two different roots**, and nothing detects it: a missing
directory raises, but a *stale* one that happens to exist returns wrong data
silently. `leadlag_screen.py` gets this right — it tries `config_pk` first and
falls back to the literals only on `ImportError`. `build_feature_store.py` does
not try at all.

**Rule for the rebuild: every path comes from `config_pk`, and a script that
overrides a root prints both the override and the config value at start-up.**

### 37.2 `build_feature_store.py` — and the final answer on depth weighting

**What it is.** A passive observer that rides the real engine. It installs a
`FeatureCollector` as the strategy, wired to *the same `Book` the `Backtester`
mutates* — not a copy, not a reimplementation — so the features are recorded at
**the engine's own no-look-ahead point**. It quotes nothing: `quotes()` returns
`{}`. One instance per symbol-day, so all state resets daily.

Output: `feature_store/{symbol}/date={date}.parquet`. 38 symbols in the live
`SYMBOLS` list.

**The one design decision worth copying.** Inventory is **excluded from the
features on purpose**:

> *"INVENTORY IS DELIBERATELY EXCLUDED from features (endogeneity: the model must
> learn market toxicity, not our historical policy). pos stays in A-S skew only."*

That is the right call and the reason is exactly right: a feature that records
what your own past policy did teaches a model to predict your policy, not the
market.

**The 17 features, per event:**

| feature | what it is |
|---|---|
| `mid` | arithmetic mid `(bb+ba)/2` — the **label reference** |
| `spread_bps` | `(ba−bb)/mid × 1e4` |
| `obi_1` | `(bq−aq)/(bq+aq)`, **level 1 only** |
| `obi_5` | `Book.obi(5)` — disclosed depth |
| `obi_deep` | `Book.obi(None)` — all visible levels |
| `micro_dev_bps` | microprice deviation from mid, in bps |
| `ofi_l1` | Cont–Kukanov level-1 order flow imbalance |
| `qdr_bid`, `qdr_ask` | queue-depletion rate, per side |
| `ewma_trade_flow` | EWMA of signed trade flow, `ALPHA_FAST = 0.1` |
| `toxicity` | `\|Σ signed\| / Σ \|signed\|` over the last `FLOW_WINDOW = 50` trades |
| `signed_volume` | the raw sum over that window |
| `spread_z` | Z-score of spread in ticks over `SPREAD_Z_WINDOW = 200`; **NaN below 20 samples** |
| `realized_vol_bps` | `√ema_var × 1e4`, `VOL_ALPHA = 0.05` |
| `vpin` | mean of closed bucket imbalances; bucket = `1/50` of running daily volume |
| `time_since_trade_ms` | the quiet clock |
| `ts_exch` | event time |

#### The weighting question, finally answered

This was the last plausible home for a decay-weighted OBI, and it is not there.
**The only depth weighting anywhere in this codebase is the microprice, and it
is a weighting over the two touch *prices*, not over levels:**

```python
# L1 depth-imbalance weight (bid share of touch depth).
imb = bq / (bq + aq)
# Microprice: depth-weighted touch price (heavier bid pulls fair UP).
microprice = ba * imb + bb * (1.0 - imb)
```

Note the crossing: the **bid** share multiplies the **ask** price. That is
correct and is the standard microprice — a heavy bid queue means the next trade
is more likely to lift the offer, so fair value is pulled up toward `ba`. A
rebuild that "fixes" this to `bb * imb + ba * (1 − imb)` inverts the signal.

So, stated once and for all:

- **Across price levels: equal weight, always.** `Book.obi(n)` sums the top `n`
  flat; multi-level OFI sums ranks 1..N flat and says in its own comment that it
  *"leaves depth N as an experiment axis rather than baking in a decay curve"*.
- **Across the two sides at the touch: depth-weighted, once, as the microprice.**
  Emitted as a feature (`micro_dev_bps`), never wired into a trading threshold.
- **Across time: two EWMAs**, on signed flow and on realized variance — and
  those are the only decay constants in the system.
- **`extreme_obi_sweep.py` sweeps thresholds, not weights** (§27A).

**There is no per-level decay and no decay sensitivity sweep in any file read
for this document.** If that work exists it lives outside this codebase, and I
am not going to reconstruct its results.

#### Three things a rebuild must get right here

**1. The Cont–Kukanov OFI is capped on a price move, deliberately.**

```python
if bb == self.prev_bb:   dq_bid =  bq - self.prev_bq   # same price -> true delta
elif bb >  self.prev_bb: dq_bid =  bq                  # improved -> +new depth
else:                    dq_bid = -self.prev_bq        # worsened -> -old depth
ofi_l1 = dq_bid - dq_ask
```

The reason is recorded: on a price move the level *re-levels* rather than flows,
so the contribution is capped at the touch quantity itself, *"preventing the
100k+ artifacts seen when a whole queue is (dis)counted as flow."* And the first
event returns `0.0` rather than a spurious full-queue spike — a guard, not an
accident.

**2. The label join is forward-ASOF with a hard no-leak assertion.**

`LABEL_HORIZONS_MS = [1000, 5000, 30000]`, **5 s primary** because it matches
the flatten horizon. Labels are `markout_{h}ms_bps`, joined with
`merge_asof(direction="forward")` against **the feature rows' own event-level
mid timeline**, then:

```python
assert ok, f"LOOK-AHEAD LEAK at horizon {h}ms: a label mid predates t+h"
```

Three bugs are named in the header as having been fixed in this merge, and the
third is the one that matters: the label timeline used to come from **snapshots
only**, which is stale between snapshots and therefore biased the 1-second
labels. **Labels must come from the same event-level replay as the features.**

**3. One comment is mislabelled — check the arithmetic, not the comment.**

`ALPHA_FAST = 0.1` is described as *"~10-trade half-life"*. With
`x ← α·new + (1−α)·x`, the half-life is `ln 0.5 / ln 0.9` = **6.58 trades**;
**10 is `1/α`, the mean lag**, not the half-life. `VOL_ALPHA = 0.05` is
described as matching `micro_mm.vol_alpha` — **that one checks out exactly**:
`micro_mm` defaults `vol_alpha=0.05` and uses the identical update
`ema_var = α·r² + (1−α)·ema_var`, so the two realized-vol series reconcile.
(Its half-life is 13.5 mid-moves.)

**Dead code to not copy:** `SYMBOLS` is assigned three times in a row; only the
third assignment (38 names) survives. `VPIN_BUCKET_FRACTION = 1/50` is flagged
in the source as a tuning knob and is not swept.

### 37.3 `leadlag_screen.py` — verified against the file and its own output

This closes §34 row 2. Everything §30 says about the closure is consistent with
the file, and several figures move from *quoted* to *verified*.

**The four steps, each able to kill the idea, stopped at the first failure:**

0. **Leader selection** — within each sector, the name with the highest **median
   daily traded value**. Data-driven, not asserted. `N_LEADERS = 2`.
1. **Update-rate asymmetry** — median inter-trade time, leader vs follower. If
   the leader does not update materially faster there is no lead to exploit.
2. **Async-robust lead-lag** — Hayashi–Yoshida cross-correlation over the lag
   grid, per (leader, follower)-day. HY is the fix for the Epps effect; *"a
   lead-lag peak at lag 0, or one that vanishes under HY, was a sampling
   artifact, not information."*
3. **Day-as-unit stability** — is the peak lag's **sign** stable across days?
4. **Economic gate** — does the anticipatable move (HY beta × leader per-event
   vol) clear the fee hurdle? *"A statistically real 0.2 bps lead is useless."*

**Constants:** `FEE_BPS = 1.554`, `MIN_TRADES = 100` per symbol-day,
`MIN_PAIR_DAYS = 3`, `MAX_DAYS = 20`, `SMOKE_DAYS = 5`, `WORKERS = 6`.

**The sector map is not a guess, and the arithmetic closes exactly.** *"PSX
OFFICIAL SECTORS, parsed from the exchange's daily quotation sheet (Section 4,
MARKET IN DETAIL) for 2026-09-14. All 113 production names matched exactly — no
guesses, no UNCLASSIFIED bucket."* Counted from the file:

**107 names in 19 screenable sectors (≥ 2 names) + 6 singleton sectors
(IMAGE, TGL, AICL, SGF, CEPB, KOSM) = 113.**

Singletons have no follower and cannot be screened, and are listed by name
*"so nothing looks silently dropped"* — which is the right way to handle an
exclusion.

**The lag grid is 39 points, −30 s to +30 s, verified by counting the list.**
The widening is documented in the file and the reasoning is the sharpest version
of it anywhere:

> *"The previous grid ran 0, 250, 500, 1000, 2000, 3000, 5000, 10000, 15000,
> 30000 — only THREE sample points between 3 and 15 seconds. PSX is quoted by
> people, not by machines, so the propagation scale to expect here is SECONDS…
> If the true lag is, say, 7 s, each day's peak lands on either 5000 or 10000
> depending on noise and flips between them day to day. That MANUFACTURES sign
> instability and INFLATES peak_lag_se — which are two of the gates the screen
> failed on. **A coarse grid in the band of interest can depress the very
> statistics used to declare the result noise.**"*

**That is the most transferable lesson in the whole cross-asset block**, and it
generalises past lead-lag: whenever a result is declared noise, check that the
measurement had the resolution to see a signal in the band where one would live.
19 points → 39; the 20-day pass runs about 7 minutes.

**The estimator.** `_hy_cov` is a two-pointer sweep over the two interval
sequences, amortized `O(nX + nY + overlaps)`, accumulating the cross-product of
every overlapping return pair. Zero-duration intervals (repeated timestamps) are
dropped because they break the overlap logic. `hy_beta_bps(...)` converts the
peak correlation and the two realized vols into the economic number the fee gate
tests.

**The self-test needs no data**: it plants a known lag in synthetic series and
confirms the estimator recovers it, writing `leadlag_selftest.png`. It is the
default action when the script is run with no flag — so the honest path is the
one you get by accident.

**The output was opened directly.** `leadlag_screen.parquet` carries
**157 rows**, read from the file's own footer, with exactly the columns §30.1
describes:

```
sector, leader, follower, leader_rank, n_days, leader_ms, foll_ms,
leader_faster, peak_lag_mean, peak_lag_se, frac_leader_leads,
peak_corr_mean, corr0_mean, ind_bps_median, clears_fee
```

**157 is now verified twice over, and a second error is corrected with it.**
Enumerating the code's own loop over the `SECTORS` dict gives **exactly 157**,
and the output file holds **exactly 157 rows** — so **nothing was dropped**.

The naive count, `Σ 2·(n−1)` over the sectors, is 176, and an earlier draft of
this document used it and then invented an explanation for the missing 19
("consistent with `MIN_TRADES` or no overlapping quotes"). That was a fabricated
reconciliation. Two things in the code make 157 the correct nominal figure:

```python
leaders[sec] = ranked[:min(N_LEADERS, len(ranked) - 1)]   # never leave 0 followers
...
key = (sec, frozenset((leader, foll)))                    # UNORDERED identity
if key in seen_pairs: continue                            # measured from the other side
```

The `frozenset` means leader #1 → leader #2 and leader #2 → leader #1 are the
same pair, measured once; and a 2-name sector gets one leader, not two.

**The lesson is the one that matters here: when a count does not match, find the
code that produces it before writing down a reason it might differ.** A
plausible explanation for a gap that does not exist is worse than no explanation
at all.

### 37.4 THE CORRECTION REGISTER — every claim this document got wrong

An earlier draft of this document was checked, claim by claim, against the
source files. **Eleven statements were wrong.** They are listed here, with the
evidence, rather than edited away, because the *pattern* in them is the useful
part — and every one of them is the kind of error a rebuilder would inherit
silently.

Each is already fixed in the section it belongs to. This is the register.

| # | what the draft said | what is true | how the error arose |
|---|---|---|---|
| 1 | `leadlag_screen.py` **has never been run** (§20.6) | It has: 20 days, 157 pairs, output parquet on disk | Carried forward from a **to-do**, never checked against the artefact |
| 2 | **157 "ordered" pairs**; nominal 176, so 19 were dropped (§30.3) | 157 **unordered** pairs; enumerating the code's own loop gives **exactly 157** and nothing was dropped | Used the naive `Σ 2(n−1)`, then **invented a reconciliation** for a gap that did not exist |
| 3 | The multiple-testing correction gives **~10 of 157** by chance (§30.2) | `D = median(n_days)` from the data. `D=5 → ~10` is the source comment's **worked example** at `SMOKE_DAYS`; the real run had `MAX_DAYS = 20`, where the expectation is ≈0 | Quoted a header comment as if it were the computation |
| 4 | The liquidation residual is **"never distributed"** (§26.2, §35) | It **is** spread pro-rata by notional and added into `net`. `net` reconciles by construction | Quoted the file's own comment; the code 300 lines later contradicts it |
| 5 | The file **asserts** the identity per bucket (§26.1) | There is no `assert` statement. The word appears once, in a header comment. The identity is printed, not tested | Same: comment read as code |
| 6 | *"the root of the −70% unexplained"* is in the decomposition file (§26.2) | It is in **`universe_expand.py`** | Attributed a real quote to the wrong file |
| 7 | The sweep has **`throttle_boost` and `all`** arms (§27A.4) | Neither exists. There are 8 arms, and the one the draft **missed** is `throttle_off`, the control that has never been run | Transcribed the file's **header prose** instead of reading the `ARMS` dict |
| 8 | The assignment captures **91.7%** of the per-name ceiling (§20.2) | **98.8%**. Both PKR totals are right; the percentage in the project to-do is wrong | Propagated a project document without recomputing it |
| 9 | The unfilled haircut is **3%** (§22) | **10%** — the runner overrides the fallback. §24.2 had already corrected this, and the §22 defect register was not updated with it | A correction applied in one section and not swept through the others |
| 10 | Nine price mechanisms (§12); five `_requote` stand-down reasons (§23) | **Ten** (P1–P10) and **four** | Counts written from memory, not from the list directly below them |
| 11 | `prev_spread` is the snapshot **strictly before** each trade (§33.1) | `searchsorted(..., side="right") - 1` → **at or before** | A no-lookahead claim stated loosely |

**The pattern, which is the point of keeping this register.** Nine of the eleven
are one of three failure modes:

- **A comment quoted as behaviour** (#4, #5, #3, #7). Four separate times, a
  source file said what it intended and the code did something else. *Comments
  are evidence of intent, never of behaviour.*
- **A derived number reproduced without recomputing it** (#8, #2). Both came
  from a document; one was wrong and one invited an invented explanation.
- **A correction applied locally and not swept** (#9, #1). §24.2 fixed the
  haircut and §22 kept the old value; a to-do said "not run" and the artefact
  said otherwise.

**The deepest one is #2**, and it is worth naming on its own: faced with a count
that did not match, the draft produced a *plausible reason for the gap* instead
of finding the code that produced the count. **A plausible explanation for a
discrepancy that does not exist is worse than admitting the discrepancy**, because
it looks like understanding and it survives review.

**What none of these changed.** No P&L figure moved except #8, which moved in
the *unfavourable* direction for the shipped design (less headroom left, not
more). No threshold, no venue rule, no experiment's sign or significance. Every
other correction is about **what is known and how it is known**.

---

**The evidence behind #1, kept in full because it is the cleanest example.**

§20.6 said the screen *"has never been run"*. Three independent pieces of
evidence say otherwise:

1. `leadlag_screen.parquet` exists and holds **157 rows**.
2. `leadlag_diagnose.py` consumes that parquet, and the §30.3 outcome is derived
   from it.
3. The source comment records an observed runtime — *"the 20-day pass took about
   7 minutes"* — which is a measurement, not a plan.

**The correct statement:** the screen has been **built, self-tested on synthetic
ground truth, and run over 20 days**, producing the 157-row table that §30.3
reports. What has *not* happened is any engine test of a defensive throttle
built on it — and there is no reason to run one, because **1 of 157 pairs clears
the fee.** §20.6 and §36.14 item 2 are amended accordingly.

**And three §34 gaps close.**

| §34 row | was | now |
|---|---|---|
| 1 | decay-sensitivity work might be in `build_feature_store.py` | **read — it is not there.** The candidate list is down to `expand_feature_store.py` and `probe_conditional_markout.py`, and the balance of evidence is now firmly that no per-level decay exists |
| 2 | `leadlag_screen.py` described only second-hand | **read, and its output opened.** Closed |
| 4 | `config_pk.py` unread; path divergence suspected | **read, and the divergence is confirmed in `build_feature_store.py`** — a different root, hardcoded, with no reconciliation. Upgraded from "if those two directories ever differ" to "they do differ in at least one script" |

**What this pass did not change.** No P&L figure, no threshold, no venue rule,
no experiment result. Every correction is about **what is known and how it is
known**, not about what the system does.

---

*End of specification.*
