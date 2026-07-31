# `mm_backtest.py` — Execution Flow Walkthrough

Follow this top to bottom to trace exactly what runs, in order, when you launch a backtest.

---

## PHASE 0 — Import time (before you call anything)

```
import mm_backtest
   │
   ├─ TICK = 0.01
   ├─ FEE_* constants evaluated
   └─ FEE_TOTAL_PCT computed  =  commission×(1+SST) + laga + SECP + IPF + clearing + rebate
                               =  0.001773  (17.73 bps per side)
```

Nothing else executes. `fee_for()` and `round_tick()` are defined but not called.

---

## PHASE 1 — You call `load_events(...)`

This runs **once**, before any simulation.

```
load_events(u_path, s_path, t_path)
   │
   ├─ 1. pd.read_csv × 3            → u (updates), s (snapshots), t (trades)
   │
   ├─ 2. ms() on each table         → ts_exch  (exchange clock)
   │        u,t: transact_time            ← tag 60, millisecond precision
   │        s  : orig_time                ← tag 42, second precision only
   │
   ├─ 3. ms() on each table         → ts_cap   (capture clock = when WE received it)
   │
   ├─ 4. ast.literal_eval on trades → rest_oid (which resting order the trade hit)
   │
   ├─ 5. groupby("msg_seq") on s    → snap_groups  {msg_seq: all rows of that message}
   │                                    includes BID/OFFER + AGG_* + status rows
   │
   ├─ 6. groupby("msg_seq").min()   → snap_ev  one (ts_exch, ts_cap) per snapshot message
   │
   ├─ 7. Build the event list — three sources, one stream:
   │        updates    → (ts_exch, 1, appl_seq, "U", row)
   │        trades     → (ts_exch, 1, appl_seq, "T", row)
   │        snapshots  → (ts_exch, 0, 0,        "S", row)
   │
   └─ 8. events.sort(key = ts_exch, kind_rank, appl_seq)
            ts_exch    → exchange time is the primary order
            kind_rank  → snapshots (0) apply BEFORE incrementals (1) at the same ms
            appl_seq   → exact wire order within a same-millisecond burst

RETURNS → (events, snap_groups, t)
```

On the MCB sample day this yields **5,353 events** and **2,195 snapshot groups**.

---

## PHASE 2 — You construct the `Backtester`

```
Backtester(strategy, cfg)
   │
   └─ __init__
        ├─ self.strat = strategy
        ├─ latency:  cfg['latency_model'] if given
        │            else build LatencyModel with all randomness zeroed
        │                 (constant latency = degenerate stochastic model)
        ├─ self.use_ack = ('latency_model' in cfg)   ← ack realism only in stochastic mode
        ├─ self.book = Book()        (empty; no orders yet)
        ├─ self.work = {}            (our live quotes — none yet)
        ├─ self.pending = []         (in-flight message heap — empty)
        ├─ pos = 0.0, cash = 0.0
        └─ fills = [], equity = [], stats = {...}, eod = None
```

Nothing is simulated yet.

---

## PHASE 3 — `run(events, snap_groups)` — the main loop

Reads `t0, t1` from `cfg['session']`, sets `know = 0`, then **loops once over every event**.

### The five steps, executed in this exact order per event

```
FOR EACH event (ts_exch, _, _, kind, obj):

  ┌─ STEP 1 ─ _activate_until(ts_exch)
  │     Land OUR in-flight messages due strictly BEFORE this event.
  │     while pending[0].land_time < ts_exch:
  │         "ARRIVE" → _arrive(t, order)
  │                       ├─ book.bbo()  → would it cross the market NOW?
  │                       ├─ if crosses  → reject, count, discard
  │                       └─ else        → order.ahead = book.qty_at(side, price)
  │                                        work[side] = order   (now live)
  │         "CANCEL" → match (side, oid)
  │                       ├─ still the same order → work.pop(side)  ✓ cancelled
  │                       └─ already gone         → stale_cancels_ignored++
  │
  ├─ STEP 2 ─ FILL CHECKS  (against the PRE-event book — order matters!)
  │     kind == "T"  → _on_market_trade(obj)
  │                      ├─ skip AUCTION prints
  │                      ├─ passive_side = opposite of aggressor
  │                      ├─ skip if no order there, or cancel already landed
  │                      ├─ trade THROUGH our price → _fill(..., "through")
  │                      └─ trade AT our price      → drain order.ahead first,
  │                                                   remainder → _fill(..., "at_queue")
  │     kind == "U" and ORDER_ADD → _on_market_add(obj)
  │                      └─ crossing add → _fill(..., "crossing_add")
  │     kind == "U" and CANCEL   → _on_market_cancel(obj)
  │                      └─ remove that id from our ahead queues (we move up)
  │
  │     Any _fill(...) →  take = min(our qty, offered qty)
  │                       pos  += ±take
  │                       cash += ∓take × OUR price  −  fee_for(...)
  │                       fills.append({t, side, px, qty, reason})
  │
  ├─ STEP 3 ─ APPLY the event to the historical book
  │     kind == "S" → book.snapshot(snap_groups[msg_seq])
  │                     ├─ read phase + circuit limits
  │                     ├─ if no BID/OFFER rows → return (status-only; keep book)
  │                     ├─ rebuild all levels + __H_ hidden lumps
  │                     ├─ add __AGG_ deep residual (L11)
  │                     └─ self.o = tgt        ← FULL REPLACEMENT, lumps reset
  │                   then _on_snapshot_queue_reset()  → rebuild our ahead dicts
  │     kind == "U" → book.add(obj)  or  book.cancel(obj)
  │     kind == "T" → book.trade(obj)
  │
  ├─ STEP 4 ─ MARK TO MARKET
  │     bb, _, ba, _ = book.bbo()
  │     if both sides exist:
  │         mid = (bb + ba) / 2
  │         last_good_mid = mid                    ← EOD fallback reference
  │         equity.append({t, mid, equity, pos, obi_5, obi_deep})
  │     (one-sided book → row skipped, so equity is shorter than events)
  │
  └─ STEP 5 ─ STRATEGY REACTS
        if strategy has observe():
            observe(kind, obj, ts_exch, mid)       ← calibration state (σ, flow, quiet)
        know = max(know, obj.ts_cap)               ← knowledge time never rewinds

        if t0 ≤ ts_exch ≤ t1:   → _requote(know)
        elif ts_exch > t1:      → pull all quotes, then EOD FLATTEN (once)
```

### Inside `_requote(know)` — step 5's main branch

```
_requote(ts_know)
   │
   ├─ HALT / PIN GATE
   │     quotable = phase in (None, "CONTINUOUS_AUCTION") and not book.pinned()
   │     if not quotable:
   │         halted_requotes++
   │         cancel every working order (via normal latency path)
   │         RETURN — no new quotes during halts, auctions or band pins
   │
   ├─ bb, bq, ba, aq = book.bbo()
   ├─ want = strategy.quotes(bb, bq, ba, aq, pos)      ← the strategy's decision
   │
   ├─ BAND CLAMP — clip each desired price into [limit_dn, limit_up]
   │
   └─ FOR EACH side in (BUY, SELL):
         ├─ ACK GUARD (stochastic mode): if this side's cancel is unconfirmed,
         │              skip repricing it → requotes_blocked_by_ack++
         ├─ if desired == working (same px, same qty, no cancel racing) → do nothing
         ├─ if an incumbent exists → draw_out(); set cancel_at; push CANCEL(side, oid)
         │                            (order stays FILLABLE until cancel_at)
         └─ if a quote is wanted   → draw_out(); _oid++; push ARRIVE(MyOrder(...))
```

### The EOD flatten — runs exactly once

```
first event with ts_exch > t1:
   │
   ├─ pull every working quote
   └─ if eod is None:
        ├─ pending.clear()                       ← void in-flight; report becomes final
        ├─ liq_cash, unfilled, vwap = book.liquidation_value(pos, fee_for)
        │      walks the real book best-level-first, pays the spread, deducts fees
        │      EXCLUDES __AGG_ (fictional price), INCLUDES __H_ (real price)
        ├─ mid_e = closing mid, or last_good_mid if one-sided
        ├─ residual_mark = unfilled marked at ref with a 10% haircut
        └─ eod = { pos_at_close, mid_at_close, equity_mid_mark,
                   liquidation_clean, residual_marked, equity_liquidated,
                   liq_vwap, liq_slippage_per_sh, unfilled_sh }
```

### Loop ends

```
RETURN (DataFrame(fills), DataFrame(equity), stats)
        plus self.eod populated on the Backtester object
```

---

## Call-graph summary — who calls whom

```
run()
 ├── _activate_until()
 │     ├── _arrive()            → book.bbo(), book.qty_at()
 │     └── (cancel matching by side+oid)
 ├── _on_market_trade()         → _fill()
 ├── _on_market_add()           → _fill()
 ├── _on_market_cancel()
 ├── book.snapshot() ──────────► _on_snapshot_queue_reset() → book.qty_at()
 ├── book.add() / book.cancel() / book.trade()
 ├── book.bbo(), book.obi()
 ├── strategy.observe()         [optional]
 ├── _requote()
 │     ├── book.pinned()        → book.bbo()
 │     ├── book.bbo()
 │     ├── strategy.quotes()    ← THE STRATEGY DECISION POINT
 │     ├── lat.draw_out(), lat.draw_ack()
 │     └── _push()
 └── book.liquidation_value()   [once, at session end]
       └── fee_for()

_fill() → fee_for()
```

---

## The two-clock rule, in one place

| Clock | Variable | Drives | Why |
|---|---|---|---|
| **Exchange time** | `ts_exch` | Book replay, all fill decisions, order land-times | Fills are determined by exchange reality. |
| **Knowledge time** | `know` (running max of `ts_cap`) | What the strategy may see and act on | A real system can only react after data arrives. |

Our orders cross between them: a decision made at knowledge time `know` lands on the exchange timeline at `know + draw_out()`. Getting this boundary right is what makes the backtest lookahead-free.

---

## Reading the results

```python
fills, equity, stats = bt.run(events, snap_groups)

bt.eod["liquidation_clean"]     # CHECK THIS FIRST — False means PnL is an estimate
bt.eod["equity_liquidated"]     # headline PnL
bt.eod["equity_mid_mark"]       # what mid-marking would have flattered you by
fills.groupby("reason").size()  # which fill rules your PnL depends on
stats                           # halts, ack blocks, rejected crossings, stale cancels
equity[["t","pos","obi_5"]]     # inventory path and imbalance features
```
