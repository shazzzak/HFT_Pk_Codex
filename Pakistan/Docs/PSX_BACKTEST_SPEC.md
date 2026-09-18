# PSX market-making BACKTEST — complete mechanism specification

**Written 2026-09-18.** This describes the **research backtest only** — the
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

**Not readable, so nothing below depends on them:** `run_legacy_mm.py` (supplies
`R.MICRO_PARAMS`, `R.CFG`, `R.LATENCY_SEED`, date discovery and the parsed-store
readers), `config_pk.py` (path roots), `spot_capture_markout_decomp.py`
(`_mid_at`, `_split_move`), `expansion_names.py`, and the generators of the four
calibration CSVs. `/Users/shazzak/PycharmProjects/HFT/Pakistan/Docs` is not a
folder I can reach, so anything only written there is not reflected here.

Where the code and a document disagree, **the code wins and I say so**.

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
**residual mark** (§9.4), which is a mark and not a trade, so it pays no fee.

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
   `ref * (1.0 - pos_sgn * unfilled_haircut_pct)` with the haircut defaulting to
   **3%**. It pays no fee — it is a mark, not a trade.
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

## 12. THE OTHER THROTTLES AND SWITCHES

All off in the production arms. Listed because they exist, were built, and can
be turned on.

| switch | default | what it does |
|---|---|---|
| `obi_defensive` | **True in production** | **Price** retreat, not size: widen the side the book leans against by `obi_defensive_ticks=1.0` when `|imb-0.5| > 0.15` |
| `ofi_defensive` | False | Same, on trailing order-flow imbalance over a hybrid `(N events, T seconds)` window |
| `ofi_throttle` | False | Size cut on trailing OFI beyond `ofi_throttle_thresh=0.30` |
| `qdr_throttle` | False | Queue-depletion-rate: fraction of our own touch queue that vanished at an unchanged price; fires at 0.40 |
| `flow_throttle` | False | Aggressor-flow z-score over a 15 s half-life with a 300 s variance EWMA; fires at ±2.0 |
| `enable_pov_cap` | False | Replaces `max_inv` with `min(max_inv, POV capacity)` |
| `enable_run_reprice` | False | After 3 consecutive same-side aggressors, push the exposed side 1 tick away |
| `enable_aggr_lean` | False | Shifts **fair value** by `k·(signed/abs flow)·spread`, only in a calm book |
| `enable_age_cross` | False | Once a position is held ≥ 15 min, cross the spread to flatten — the only taker path |
| `enable_inv_taper` | False | Tapers the adding side as POV utilisation rises |
| `size_boost_mult` | 1.0 (off) | Boost the favourable side's size when the book leans that way |
| `enable_onetick_mm` | False | A separate regime for one-tick books: quote one side only, by imbalance sign |
| `tol_ticks` | 0.0 (off) | Pegging hysteresis — hold the previous quote until it drifts `tol_ticks` |

**Two couplings found in the code that are not obvious from the parameter
names**, and which matter if you re-enable them:

1. Turning on `ofi_throttle` **also** enables the OFI *price* retreat, because
   the retreat is guarded on `ofi_sig is not None` rather than on
   `ofi_defensive`. The comments describe the throttle as size-only.
2. `enable_inv_taper` without an `unwind_profile` reaches `_pov_capacity()`
   unguarded and will raise. `enable_pov_cap` **is** guarded; the taper is not.

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
experiment in the project**, and two of them differ from the strategy's own
defaults (`improve_ticks` 1.0 → 0.0, `use_microprice` True → False).

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
0.15 — **+15.7%**, and 91.7% of the per-name in-sample ceiling of 15,555,884.

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
| **Sector lead-lag** | `leadlag_screen.py` is built and validated on synthetic ground truth (recovered a +1000 ms peak against a true +800 ms; corr +0.464 vs +0.155 at lag 0), sector map complete at 107 names / 19 sectors. **It has never been run** |

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
| 4 | **3% unfilled haircut** is a guess | It prices whatever the visible book cannot absorb at the close |
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
6. `_requote` and its five stand-down reasons; the in-flight hold; the ack wait
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
