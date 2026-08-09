# `mm_backtest.py` — Class & Method Reference

One line per item. Grouped by what it is, in file order.

---

## Module-level constants

| Name | What it is |
|---|---|
| `TICK` | PSX price grid, 0.01 PKR — the smallest legal price increment. |
| `FEE_COMMISSION_PCT` | Broker commission as a fraction of traded value (0.15%); replace with your negotiated HFT rate. |
| `FEE_SST_RATE` | Provincial sales tax charged **on the commission**, not on value (13% Sindh / 16% Punjab). |
| `FEE_PSX_LAGA_PCT` | PSX trading fee (laga), 0.0035% of value. |
| `FEE_SECP_PCT` | SECP supervisory fee, 0.00065% of value. |
| `FEE_IPF_PCT` | PSX regulatory / Investor Protection Fund fee, 0.00062% of value. |
| `FEE_CLEARING_PCT` | NCCPL + CDC clearing, assumed 0.003% (intraday-squared: CDC delivery waived). |
| `FEE_PER_SHARE_FLAT` | Any flat per-share cost; zero in the current schedule. |
| `FEE_MM_REBATE_PCT` | Market-maker programme rebate — enter as a **negative** number when known. |
| `FEE_TOTAL_PCT` | All-in per-side cost = commission×(1+SST) + exchange stack + rebate. Currently 17.73 bps. |

## Module-level functions

| Function | What it does |
|---|---|
| `fee_for(price, qty)` | Returns the all-in fee in PKR for one fill of `qty` shares at `price`. |
| `round_tick(p, side)` | Snaps a price onto the tick grid **safely** — bids floor, asks ceil, so rounding never makes a quote more aggressive. |
| `load_events(u_path, s_path, t_path)` | Reads the three CSVs, builds ms timestamps and `rest_oid`, and returns the fully sorted event stream plus snapshot groups. |

---

## `Order` (dataclass)

One resting order in the **historical** book (someone else's order).

| Field | Meaning |
|---|---|
| `side` | `"BUY"` or `"SELL"`. |
| `price` | Limit price. |
| `qty` | Remaining quantity (mutated in place as trades consume it). |

## `MyOrder` (dataclass)

One of **our** simulated orders, carrying the state a real order carries.

| Field | Meaning |
|---|---|
| `side` | `"BUY"` = our bid, `"SELL"` = our ask. |
| `price` | Our limit price. |
| `qty` | Remaining quantity we still want filled. |
| `ahead` | `{order_id: qty}` of every order queued **in front of us** at our price. |
| `t_active` | Exchange-ms when the order became live (after send latency). |
| `cancel_at` | Exchange-ms when our in-flight cancel lands; `None` if no cancel sent. |
| `oid` | Unique id so a cancel targets **this** order, not just the side. |

---

## `Book` — replay of the historical order book

State is one dict, `self.o` = `{order_id: Order}`. Price levels are derived on demand, never stored.

| Method | What it does |
|---|---|
| `__init__()` | Creates the empty book plus market-state fields (`phase`, `limit_up`, `limit_dn`) and audit counters. |
| `add(r)` | Applies an `ORDER_ADD`: inserts the order, skipping rows with no `order_id`. |
| `cancel(r)` | Two-branch cancel: remove by exact `order_id`; else decrement a matching `__NEG_`-eligible hidden lump by price; else no-op. |
| `trade(r)` | Applies a historical trade: decrements the named resting order, or parks the qty as a signed `__NEG_` placeholder when the id is unknown. |
| `snapshot(rows)` | Full-state replacement from a 35=W message; also reads `phase`, circuit limits, `__H_` hidden depth and the `__AGG_` deep residual. |
| `bbo()` | Returns `(best_bid, bid_qty, best_ask, ask_qty)`, netting all lumps per price level and discarding levels that net to ≤ 0. |
| `liquidation_value(pos, fee_fn)` | Walks the real book best-level-first to flatten `pos`, returning `(cash, unfilled_shares, vwap)`. |
| `pinned()` | True when the market is stuck at a circuit limit (best bid ≥ upper, or best ask ≤ lower). |
| `qty_at(side, price)` | Returns `{order_id: qty}` of orders ahead of us at one price, already reduced by any already-traded `__NEG_` quantity. |
| `obi(n, include_deep)` | Order-book imbalance over the top `n` levels per side; `include_deep=True` adds the `__AGG_` deep residual. |

### The three synthetic keys `Book` creates

| Key prefix | Represents | Price is |
|---|---|---|
| `__H_{side}_{price}` | Undisclosed quantity **inside** a visible level (level total minus named orders). | Real and tradable. |
| `__AGG_{side}` | The deep tail **beyond** the visible ~10 levels (`AGG_BID` minus sum of visible). | Fictional placeholder. |
| `__NEG_{side}_{price}` | Quantity already traded away by an **unattributable** trade; stored negative so it subtracts. | Real (the trade price). |

---

## `LatencyModel` — stochastic two-leg latency

| Method | What it does |
|---|---|
| `__init__(...)` | Stores median/tail parameters for both legs plus a seeded RNG for reproducible runs. |
| `draw_out()` | Returns one random **send** latency (decision + wire out), occasionally adding a fat-tail spike. |
| `draw_ack()` | Returns one random **ack** latency — how long until we learn a cancel landed. |

Constant latency is the degenerate case: `tail_prob=0` and zero tail sizes make every draw return the same number.

---

## `Backtester` — the engine

| Method | What it does |
|---|---|
| `__init__(strategy, cfg)` | Wires up the book, our order state, the latency model (stochastic or constant fallback), accounting and counters. |
| `_push(t, action, payload)` | Schedules one of our messages onto the exchange timeline heap at land-time `t`. |
| `_activate_until(t_exch)` | Lands every in-flight message due strictly before the next market event; cancels match by `(side, oid)`. |
| `_arrive(t, o)` | Our order reaches the exchange: rejects it if it would cross, else snapshots its queue position and makes it live. |
| `_fill(side, price, qty, t_exch, reason)` | Books a fill at **our** limit price, updates position/cash/fees, and logs it with a provenance tag. |
| `_on_market_trade(r)` | Decides whether a historical trade's flow would have hit our quote — through, at-price by queue, or not at all. |
| `_on_market_add(r)` | Fills us when an incoming order crosses our quote (marketable flow that would have matched us). |
| `_on_market_cancel(r)` | Removes a cancelled order from our `ahead` queues, improving our position. |
| `_on_snapshot_queue_reset()` | Rebuilds our `ahead` dicts after a snapshot replaced the book; assumes worst case (everything ahead of us). |
| `_requote(ts_know)` | Applies the halt/pin gate and band clamp, asks the strategy for quotes, and reconciles them against working orders. |
| `run(events, snap_groups)` | The main loop: one pass over the event stream, returning `(fills, equity, stats)` and populating `self.eod`. |

### Key `Backtester` attributes

| Attribute | Meaning |
|---|---|
| `book` | The reconstructed historical book (everyone else's orders). |
| `work` | `{side: MyOrder}` — our live quotes, at most one per side. |
| `pending` | Min-heap of our in-flight messages, ordered by land-time. |
| `pos` / `cash` | Our position in shares (signed) and cash in PKR. |
| `fills` / `equity` | Accounting logs, returned as DataFrames. |
| `stats` | Diagnostic counters (crossings rejected, cancels, ack blocks, halts, stale cancels). |
| `eod` | End-of-session liquidation report — **read `equity_liquidated` as headline PnL**. |
| `last_good_mid` | Last mid seen with a two-sided book; the EOD fallback reference price. |

### What `eod` contains

| Key | Meaning |
|---|---|
| `pos_at_close` | Position carried into the close. |
| `mid_at_close` | Mid at session end, or `None` if the book was one-sided. |
| `equity_mid_mark` | What the old (flattering) mid-marking would have reported — diagnostic only. |
| `liquidation_clean` | **`False` means `equity_liquidated` is an estimate, not realisable.** Always check this. |
| `residual_marked` | Haircut mark applied to inventory the book could not absorb. |
| `equity_liquidated` | Headline PnL: cash + book-walk proceeds + residual mark. |
| `liq_vwap` | Volume-weighted price achieved by the liquidation walk. |
| `liq_slippage_per_sh` | Per-share slippage of that walk versus the closing mid. |
| `unfilled_sh` | Shares the visible book could not absorb — genuinely unpriceable. |

---

## `NaiveSymmetricMM` — baseline strategy

| Method | What it does |
|---|---|
| `__init__(half_spread, size, max_inv)` | Stores the fixed half-spread, quote size and hard inventory cap. |
| `quotes(bb, bq, ba, aq, pos)` | Quotes mid ± half_spread, safe-rounded and post-only clipped, switching a side off at the inventory cap. |

**Strategy interface contract:** any strategy needs only `quotes(bb, bq, ba, aq, pos)` returning `{side: (price, qty)}`, omitting a side to mean "no quote". Optionally add `observe(kind, obj, ts_exch, mid)` to receive every event for calibration — `run()` calls it only if it exists.
