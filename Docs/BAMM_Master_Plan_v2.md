# Basis-Aware Market Maker (BAMM) — Master Execution Plan

**Version 2.** Big Byte Insights — PSX cross-instrument market making.
Supersedes Plan 1 (roadmap), Plan 2 (17-phase gated plan), Plan 3 (feature schema), and v1 of this document.

**v2 changes:** risk-off state machine added (Phase 14); OMS gated on feed health, not just features (Phases 7, 12, 14); VECM demoted in favour of flow-lead measurement (Phase 3c); deliverable-futures expiry gate and ready-leg borrow constraint made operative (Phases 2, 3, 14); dynamic VaR-based margin with cash calls replacing the static cap (Phase 13).

---

## How to read this document

Every phase has **Work** (what to build) and a **Gate** (what must be true before the next phase starts). Do not start phase *N+1* until gate *N* passes and is recorded in `bamm/08_reports/gates.md` with a date and the actual numbers observed.

The reason for gates is not bureaucracy. In a cross-asset pipeline, an error in contract mapping, dividend timing, clock alignment, or book reconstruction does not produce an obvious crash — it produces **apparent alpha**. Every one of those errors biases the basis in the direction that looks tradeable. You will not detect them downstream; you will fund them.

**Simplifications are flagged inline as `SIMPLIFICATION:` with the production version stated alongside.** Anything not so flagged is intended to be production-grade as written.

**Code convention:** comments sit on their own line above the line they describe.

---

## What changed from the three source plans, and why

| Change | Source | Rationale |
|---|---|---|
| **Economic kill gate inserted at Phase 3**, before any pipeline build | New | All three plans build 10–17 phases of infrastructure before asking whether the trade exists. If PSX futures depth is too thin to hedge, everything after Phase 3 is wasted. This is the cheapest possible falsification and it runs on data you already have. |
| **Fitted implied carry replaces KIBOR as the anchor** | New | PSX futures are substantially a leveraged retail vehicle; embedded financing routinely diverges from interbank. Anchoring `F_theo` to KIBOR manufactures a large, persistent, seductive "arbitrage" that is simply the market's true cost of carry. KIBOR is retained as a prior and a sanity bound. |
| **Carry formula corrected to `(S − PV_D)(1 + rτ)`** | New | All three plans write `S(1+rτ) − PV_D`, which mixes conventions. The error is `PV_D · r · τ_remaining` ≈ **4.5 bps** for a 5 PKR OGDC dividend at 3 months — larger than the entire measured 3 bps gross passive edge. |
| **Cross-book staleness gated on knowledge time, not exchange time** | New | Plan 2 computes `ready_age_ms` from `ts_exch`; Plan 3 states features are "strictly based on `ts_exch`". Both differ a fresh price against one you had not yet received. Feed latency is ~80 ms median with multi-second spikes, and the bias is systematic — the stale leg lags precisely when the move happens. |
| **Feed health separated from book quiescence** | v2 | A 2-second gap is a dead feed in FFC and a completely normal quiet period in KTML (median trade gap 12.6 s). A naive age-based staleness gate would have you cancelling continuously in exactly the wide-spread names you most want to quote. |
| **Risk-off state machine: symmetric cancel-all on catastrophe, stale feed, or expiry** | v2 | v1 said "never cancel a side" and had no risk-off state at all. The no-asymmetric-cancel rule survives as a rule about *signal expression*; it was wrong as a rule about *risk*. |
| **OMS gated on `cross_state_valid`, not just the feature store** | v2 | v1 gated features on cross-book validity but left quotes resting when the hedge leg went blind. If you cannot see the hedge, you do not provide liquidity. |
| **Flow-lead measurement replaces VECM/Hasbrouck as the Phase 3 gate** | v2 | On a venue with multi-second trade gaps, 100 ms sampling produces mostly-zero returns, IS bounds widen to uninformative, and the estimate is sensitive to relative tick size and noise structure. Signed futures flow → ready mid change answers the decision question directly and is robust to step functions. |
| **Deliverable-futures expiry gate and ready-leg borrow constraint** | v2 | v1 tracked `settlement_type` and never acted on it. DFC roll week carries squeeze risk; and if ready shorting is unavailable, the future-cheap direction does not exist at all except against existing inventory. |
| **Dynamic VaR margin with cash calls replaces the static `margin_pct` cap** | v2 | NCCPL margins expand intraday with volatility — precisely when a basis dislocation makes you want maximum capital. A fixed percentage assumes free capital that will not be there. |
| **Universe screening and persistence transplanted in (Phase 11)** | Plan 1 | Plan 2 and 3 have no screening step. This discards the KTML lesson (236× ceiling swing between two dates) and the finding that symbol ranking **inverts** with fee level. |
| **Paper-trading ladder and TREC workstream appended (Phase 19)** | Plan 1 | Plan 2 ends at performance optimization with no go-live path. Regulatory access is a months-long parallel workstream, not a final checkbox. |
| **Feature schema adopted with corrections** | Plan 3 | Plan 3's concrete named columns and formulas are its best contribution. Denominators made consistent; labels re-specified (Phase 10). |
| **Self-collision (STP) logic dropped** | — | Plan 1's check — "would this order cross my resting orders on the other product" — is not possible: ready and futures are separate order books. The unified single-decision-point design already dissolves the §6 wash-trade concern, which was about MM and arb running as *separate* strategies in the *same* book. |

---

## Phase 0 — Freeze the validated single-instrument system

**Purpose.** Your existing backtester is the only artifact in this project that has been regression-tested. It must remain runnable and reproducible while BAMM is developed, because it is the only reference you have for whether the new engine is correct.

### Work

```
HFT/
├── existing_mm/
│   ├── mm_backtest.py
│   ├── micro_mm.py
│   ├── ticker_stats_core.py
│   ├── multi_sym.py
│   ├── run_all_tickers.py
│   ├── corp_actions_master.py
│   ├── corp_action_detector.py
│   └── tests/
│       └── test_mcb_regression.py
└── bamm/
    ├── 01_reference/
    ├── 02_event_stream/
    ├── 03_books/
    ├── 04_features/
    ├── 05_labels/
    ├── 06_simulator/
    ├── 07_models/
    ├── 08_reports/
    ├── config/
    └── tests/
```

```bash
cd "G:\My Drive\HFT"
git init
git add .
git commit -m "Freeze validated single-instrument backtester before BAMM"
git tag pre-bamm
```

Pin the documented MCB result as an automated test, not a remembered number:

```python
# Pins the documented MCB regression. Any BAMM change that alters these
# numbers has broken the shared Book/fill/fee code and must be reverted.
EXPECTED = {
    "events": 5353,
    "fills": 417,
    "equity_liquidated": -5610.86,
    "pos_at_close": 40,
}

def test_mcb_regression():
    fills, equity, stats, eod = run_mcb_reference_backtest()
    assert len(fills) == EXPECTED["fills"]
    assert abs(eod["equity_liquidated"] - EXPECTED["equity_liquidated"]) < 0.01
    assert eod["pos_at_close"] == EXPECTED["pos_at_close"]
    # liquidation_clean False means equity_liquidated is an estimate, not a
    # realisable number; the regression is only meaningful when True.
    assert eod["liquidation_clean"] is True
```

**Reuse, do not copy.** BAMM imports `Book`, `round_tick`, `LatencyModel`, and the fee functions from `existing_mm`. Copying them produces two divergent order-book reconstructors within a month, and you will not know which one is right.

### Gate 0

- [ ] The MCB regression test passes from a clean checkout.
- [ ] `eod["liquidation_clean"]` is `True` in that run.
- [ ] `git checkout pre-bamm` restores a working tree.
- [ ] `bamm/` imports from `existing_mm/` rather than containing copies.

---

## Phase 1 — Establish ground truth about the data

**Purpose.** Every plan so far has written `OGDC-AUG-FUT` and `OGDC-REG` as if they were facts. They are guesses. Your handoff records the actual convention as suffixed symbols such as `SYS-JUL` and `FFC-JULB`. Guessing here silently produces empty frames that look like quiet markets.

### Work

```python
import polars as pl

PARSED = r"G:\My Drive\HFT\Capital Stake - Parsed"
DATE = "2026-06-30"

files = {
    "snapshot": rf"{PARSED}\ob_snapshot\date={DATE}\{DATE}_ob_snapshot.parquet",
    "updates":  rf"{PARSED}\ob_updates\date={DATE}\{DATE}_ob_updates.parquet",
    "trades":   rf"{PARSED}\trades\date={DATE}\{DATE}_trades.parquet",
    "misc":     rf"{PARSED}\misc\date={DATE}\{DATE}_misc.parquet",
}

for name, path in files.items():
    lf = pl.scan_parquet(path)
    print(f"\n{name.upper()}")
    print(lf.collect_schema())
    print(lf.select(pl.len()).collect())
```

Enumerate the real symbol universe and its segment mapping. This is the query that replaces the guess:

```python
# segment comes from the parser's SNAPSHOT_SEGMENT_MAP / TICK_SEGMENT_MAP.
# REG = ready equities; STOCK_DEL_FUT / STOCK_CS_FUT = single-stock futures.
sym_seg = (
    pl.scan_parquet(files["trades"])
    .group_by(["symbol", "segment"])
    .agg(pl.len().alias("trades"), (pl.col("qty") * pl.col("price")).sum().alias("notional"))
    .sort("notional", descending=True)
    .collect()
)
print(sym_seg.filter(pl.col("segment") != "REG"))
```

Confirm, do not assume, on the futures rows specifically:

- Do futures trades carry initiator tags 10116/10117? Your trade-initiator classification depends on them.
- Do futures snapshots carry the same 10-level depth and the `AGG_BID`/`AGG_OFFER` aggregate rows? `Book.snapshot`'s deep-residual logic depends on them.
- Are `phase`, `prev_close`, and the circuit-breaker entry types present on the futures segment?
- Are `capture_ts` and `transact_time` both non-null on both segments?
- **Heartbeats:** does the `misc` table carry UA001 heartbeats (and `end_of_channel` / `appl_last_seq`) for *both* channels? Phase 7's feed-health logic depends on this. If heartbeats exist on only one channel, feed health on the other must be inferred from sequence continuity instead.

### Gate 1 — record these values in `bamm/config/data_contract.yaml`

```yaml
ready_symbol_pattern:
future_symbol_examples:
ready_segment_code:
future_segment_codes:
exchange_ts_snapshot:        # orig_time
exchange_ts_update:          # transact_time
exchange_ts_trade:           # transact_time
capture_ts_column:           # capture_ts
sequence_column_updates:     # appl_seq
sequence_column_snapshot:    # msg_seq
futures_has_initiator_tags:
futures_has_agg_rows:
futures_max_book_levels:
heartbeat_available_ready:
heartbeat_available_future:
heartbeat_interval_ms:
timezone:                    # UTC
```

Do not continue until every field is filled from an actual query result.

---

## Phase 2 — Contract master

**Purpose.** The reference layer that says which two instruments belong together, and on what economic and *operational* terms.

### Work

`bamm/01_reference/contract_master.parquet`, one row per `(trade_date, underlying, future_symbol)`:

| column | notes |
|---|---|
| `trade_date` | PSX trading date, not `date(ts_exch)` — a PSX day spans two UTC dates |
| `underlying` | e.g. `OGDC` |
| `ready_symbol`, `future_symbol` | from Phase 1 |
| `contract_month` | `2026-08` |
| `settlement_type` | `deliverable` / `cash_settled` — **operative in Phase 14, not decorative** |
| `expiry_date`, `last_trading_date`, `settlement_date` | from official PSX contract specs |
| `dte_calendar`, `dte_trading` | store both; act/365 uses calendar; the expiry gate uses trading days |
| `contract_multiplier`, `lot_size_ready`, `lot_size_future` | drives the hedge ratio β |
| `tick_size_ready`, `tick_size_future` | verify per segment; do not assume 0.01 on futures |
| `is_front_month`, `roll_flag`, `days_to_roll` | explicit roll marking |
| `ready_shortable` | is the ready leg short-sellable at all (regulated/deliverable-futures-eligible list)? |
| `borrow_available`, `borrow_cost_bps` | securities-lending availability and cost, where obtainable |
| `margin_scan_range`, `margin_min_pct` | NCCPL VaR margin parameters for this contract — feeds Phase 13 |

**Derive the pairing two independent ways and reconcile:** (a) suffix parsing of the symbol string, (b) the `segment` field the parser already maps. Where they disagree, raise — do not coalesce. A silent coalesce maps the wrong contract to the wrong underlying and every downstream number is fiction.

**The `ready_shortable` field is a viability constraint, not metadata.** The strategy has two arbitrage directions. Direction A (future rich → sell future, buy ready) requires only cash. Direction B (future cheap → buy future, sell ready) requires **shorting the ready leg**. If PSX short selling is unavailable, restricted, or uneconomic for a name, Direction B exists only to the extent you are already long — which means the strategy is structurally one-sided in that symbol, and every screen, ceiling, and backtest must reflect that. None of the source plans, and v1 of this one, accounted for it.

`SIMPLIFICATION:` for the Phase 3 gate you may hand-build a single-underlying master. Production requires the full daily panel across all futures underlyings for the whole 9 months, sourced from official contract specifications rather than inferred from the tape.

### Gate 2

- [ ] Every futures symbol observed maps to exactly one underlying.
- [ ] `expiry_date > trade_date` on every row.
- [ ] `dte_calendar` strictly non-increasing within a contract as `trade_date` advances.
- [ ] Roll dates explicit; front-month designation never flips back.
- [ ] `settlement_type` populated for every contract.
- [ ] `ready_shortable` populated; the fraction of the futures universe where Direction B is unavailable is recorded.
- [ ] Suffix-derived and segment-derived pairings agree on 100% of rows.

---

## Phase 3 — THE ECONOMIC KILL GATE

**Purpose.** Determine whether this business exists before building infrastructure for it. This is the highest-value phase in the document and it runs entirely on data you already have.

The thesis of BAMM is that hedging collapses the inventory penalty from price variance to *basis* variance, letting you quote tighter and carry more inventory in the high-volume names where the TREC-fee ceiling concentrates. That thesis fails if the hedge leg is too thin or too wide to use.

### Work

Sample ~20 trading days spread across the 9 months (include an expiry week, a roll, and a high-volatility day). For every underlying with a futures contract:

**3a. Futures liquidity census**

- Days the futures symbol trades at all; trades/day; notional/day.
- Median and p90 futures L1 spread in bps.
- Median touch depth (shares and PKR) each side.
- Median depth within 5 levels — what a hedge actually walks.
- Fraction of the continuous session with a two-sided futures book.

**3b. Hedged edge, in money**

For each ready trade, using the contemporaneous state of both books:

```
hedge_cost_bps = future_half_spread_bps + future_fee_bps
net_edge_bps   = ready_half_spread_bps - ready_fee_bps - hedge_cost_bps
```

Report per underlying, per fee scenario (`rt_2p00` is operative given the TREC path at ~1.6 bps round trip, but compute the full grid):

- `pct_time_hedged_viable` — fraction of session with `net_edge_bps > 0`.
- `hedged_ceiling_pkr` — qualifying volume × net edge ÷ 2.
- `hedge_absorption` — fraction of intended clip size absorbable at the futures touch, and within 5 levels.
- **Direction-split ceiling** — A and B separately, with B zeroed where `ready_shortable` is false.

**Report the hedged ceiling without the pairing adjustment, alongside the unhedged ceiling with it.** `pair_ratio` exists because an unhedged MM must round-trip in the ready leg to stay flat. A hedged MM can be persistently one-sided in ready and flat in delta. That difference *is* the economic case for BAMM, and it must be visible as a number.

**3c. Does futures flow predict ready price?**

This decides whether the strategy is viable at all, so it cannot be assumed. Plan 3 asserts "usually, the Future leads" as a parenthetical; that parenthetical is the whole ballgame.

**Primary measure — flow-lead regression.** Robust to step functions and asynchronous updating, and it answers the operational question directly:

```
1. Build event-time buckets (e.g. 50 ms) on KNOWLEDGE time.
2. For each bucket: signed aggressive futures volume (futures OFI),
   signed aggressive ready volume (ready OFI), and ready mid change.
3. Regress ready mid change at t+k on futures OFI at t, CONTROLLING for
   ready's own OFI at t and its own lagged mid changes.
   k in {10, 50, 100, 250, 500, 1000} ms.
4. Report the incremental R-squared and t-stat of the futures-OFI term.
5. Run the mirror regression (ready OFI -> futures mid change) and compare.
```

The control term matters. Raw correlation between futures flow and ready price is contaminated by common-factor response — both legs reacting to the same news produces correlation with no lead at all. What you need is the *incremental* predictive power of futures flow once ready's own flow is accounted for.

Also report signed lagged cross-correlation of mid returns at ±10/50/100/250/500 ms as a cheap cross-check that should agree in sign.

`SIMPLIFICATION:` fixed-width time buckets. Production alternative — the Hayashi–Yoshida estimator, which is built for non-synchronously observed processes and does not require choosing a sampling grid at all.

**On VECM and Hasbrouck information share — kept, but demoted, and gated on density.** These are the textbook tools for cointegrated price discovery and they are the right *concept*; the problem is fit, not validity. On sparse series, a 100 ms grid yields mostly-zero returns, bid–ask bounce dominates the innovation covariance, IS bounds widen toward uninformative, and the estimate is sensitive to the two legs' relative tick sizes and noise structure. So: **never the kill gate**, but run it as a **secondary diagnostic wherever the data is dense enough to support it**, under an explicit density criterion decided before estimation rather than by eyeballing the output:

```
Run VECM/IS on a (underlying, day) only if, in trade time:
  - both legs have >= ~1,000 mid-changes in the session, and
  - the IS upper-lower bound spread comes back < ~0.25.
Sample in trade time (per mid-change), never on a fixed calendar grid.
If the bound spread exceeds the criterion, report "IS: not identified"
rather than the midpoint of a wide interval.
```

Where both the flow-lead regression and IS are estimable, they should agree in direction; disagreement is itself a finding — usually a sign that common-factor response is contaminating one of them — and gets investigated rather than averaged away. The two thresholds above are starting defaults in `bamm/config/leadlag.yaml`, to be tightened once you see the actual bound-spread distribution on PSX data.

**Interpretation, decided before you see the numbers:**

- **Futures flow strongly predicts ready price.** The passive ready bid is adverse-selected by construction: when the basis blows out, ready is about to reprice, and you would be racing to the front of the queue to be picked off. The "arbitrage profit" is a pickoff loss wearing a costume. Your toxicity gate cannot save you, because the toxicity and the trade signal are *the same event*. Damp θ (Phase 14) by the measured lead, and expect the ready-passive leg to be marginal.
- **Roughly symmetric.** BAMM is sound. Proceed.
- **Ready leads.** BAMM is strong; the futures-passive leg carries the adverse selection instead, and the design should lean on ready-side passive fills.

### Gate 3 — the go/no-go

Record in `bamm/08_reports/kill_gate.md`:

- [ ] Underlyings with tradeable futures on ≥50% of sampled days: **____**
- [ ] Of those, with `pct_time_hedged_viable > 20%` at `rt_2p00`: **____**
- [ ] Median `hedge_absorption` at intended clip: **____**
- [ ] Incremental R² of futures OFI on ready mid change, per surviving underlying, per lag: **____**
- [ ] Underlyings where Direction B is unavailable (`ready_shortable` false): **____**

**Stop conditions.** Fewer than ~3 surviving underlyings → BAMM is a niche, not a business; revert to the strict split from handoff §6 and stop here. `hedge_absorption` below ~50% at intended clip → the hedge does not exist at size, and no amount of modelling fixes it.

---

## Phase 4 — Carry curve and corporate actions

**Purpose.** Define fair basis. Everything downstream is a deviation from this number, so an error here is indistinguishable from alpha.

### Work

**4a. Dividends**

`bamm/01_reference/corporate_actions.parquet`: `symbol, announcement_date, book_closure_start, book_closure_end, ex_date, payment_date, dividend_per_share, action_type, source, retrieved_at`.

Per your handoff doctrine: **announcements are primary; the price-based detector is a validator.** Use `corp_actions_master.py` — its four verdicts (`CONFIRMED`, `FACTOR_MISMATCH`, `SILENT_ADJUSTMENT`, `UNANNOUNCED_MOVE`) already implement this. `SILENT_ADJUSTMENT` is the dangerous case and must never be dropped.

**4b. The carry formula, correctly**

```python
def pv_dividends(divs, r, trade_date):
    """Present value of dividends with ex_date between now and expiry.

    divs: iterable of (ex_date, amount_per_share)
    r:    annual financing rate as a decimal (0.1125, not 11.25)
    """
    total = 0.0
    for ex_date, amt in divs:
        # Discount each dividend over its own time to ex-date, not to expiry.
        tau_i = (ex_date - trade_date).days / 365.0
        total += amt / (1.0 + r * tau_i)
    return total


def theoretical_future(spot, r, dte_calendar, pv_div):
    """Cost-of-carry fair value.

    CORRECT: (S - PV_D) * (1 + r*tau)
    WRONG:    S * (1 + r*tau) - PV_D   <- mixes PV and forward-value conventions.
    The two differ by PV_D * r * tau, which is ~4.5 bps for a 5 PKR OGDC
    dividend at 3 months -- larger than the entire measured passive edge.
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must be a decimal, e.g. 0.1125, not 11.25")
    if dte_calendar < 0:
        raise ValueError("dte cannot be negative")
    tau = dte_calendar / 365.0
    return (spot - pv_div) * (1.0 + r * tau)
```

**4c. The financing rate — fitted, not assumed**

```python
def implied_carry_rate(future_mid, spot_mid, pv_div, dte_calendar):
    """Financing rate embedded in observed futures prices.

    Undefined as dte -> 0: tau in the denominator makes the estimate explode
    near expiry. Exclude the final ~5 trading days from the fit.
    """
    tau = dte_calendar / 365.0
    if tau <= 5.0 / 365.0:
        return None
    return (future_mid / (spot_mid - pv_div) - 1.0) / tau
```

Fit per `(underlying, contract)`: a robust central estimate (rolling median of intraday observations, then a per-day series), and use **that** curve as the anchor for `F_theo`.

Ingest KIBOR into `bamm/01_reference/kibor_daily.parquet` (`date, tenor, bid_rate, offer_rate, mid_rate, source, retrieved_at`; decimals only, missing values raise rather than defaulting to zero) and use it two ways: a prior when a contract has too few observations to fit, and a **bound** — an implied rate outside `[0, KIBOR + 15%]` is a data error or a genuine dislocation and should alarm, not silently propagate.

**Tradeoff, stated plainly.** A fitted curve can absorb genuine mispricing into "fair", making you blind to a real dislocation. Mitigate by fitting on a long window (contract-to-date or 20 days) and trading deviations from it — never re-fit intraday, or you will define away the signal you are trying to trade.

### Gate 4

- [ ] Three unit tests pass: no dividend before expiry → `PV_D = 0`; one dividend → counted exactly once; dividend after expiry → excluded.
- [ ] `theoretical_future` matches a hand-computed spreadsheet on 10 timestamps.
- [ ] Implied carry fitted per contract; plotted against KIBOR; the spread is stable and explicable.
- [ ] Implied rate excluded inside 5 days of expiry.
- [ ] Every feature row can resolve a financing rate; missing rates raise.

---

## Phase 5 — Deterministic dual-instrument event stream

**Purpose.** One merged, reproducible, correctly ordered stream per `(date, underlying)`.

### Work

Output `bamm/02_event_stream/{date}_{underlying}_dual.parquet`:

```
event_no, ts_exch, ts_cap, instrument, symbol, source, msg_seq, appl_seq,
entry_type, action, side, price, qty, level, order_id, trade_id, initiator
```

**Ordering is the load-bearing detail.** Sorting on `ts_exch` alone is wrong in three separate ways:

1. Your existing `load_events` sorts on `(ts_exch, kind_rank, appl_seq)`. `kind_rank` places snapshots (0) before same-timestamp incrementals (1), because a snapshot stamped *T* describes the book *as of T* and same-time incrementals must build on top of it rather than be wiped by it. A plain `ts_exch` sort destroys this.
2. Ready and futures ride **different channels with independent `appl_seq` spaces**. Cross-instrument tie-breaking on `appl_seq` compares two unrelated counters.
3. Snapshot rows sharing one `msg_seq` must stay contiguous, or `Book.snapshot` receives a partial message.

Use:

```
(ts_exch, kind_rank, instrument_rank, appl_seq_or_msg_seq, row_in_message)
```

`instrument_rank` is a fixed deterministic tie-break for same-millisecond cross-leg events. It is **arbitrary but must be stable and declared**, because at millisecond resolution you cannot prove which reached the gateway first.

`SIMPLIFICATION:` a fixed instrument rank. Production alternative — treat same-ms cross-instrument ordering as genuinely unknown and run the day both ways as a robustness check; if PnL is sensitive to that ordering, the strategy is trading on a fiction.

Verify physical sort order on disk rather than trusting the writer:

```python
df = pl.read_parquet(out_path)
print(df.group_by(["instrument", "source"]).len())
print(df.select(pl.col("ts_exch").is_sorted()))
print(df.select((pl.col("ts_exch").diff() < 0).sum().alias("backward_steps")))
```

### Gate 5

- [ ] Both instruments present; all three sources present per instrument.
- [ ] `ts_exch` non-decreasing; zero backward steps.
- [ ] No duplicate full ordering keys.
- [ ] Ready and futures sessions overlap; the overlap window recorded.
- [ ] Snapshot rows contiguous within `msg_seq`.
- [ ] Zero null `ts_exch` or `ts_cap`.
- [ ] Re-running the builder produces a byte-identical file.

---

## Phase 6 — Dual book reconstruction

**Purpose.** Two independently correct books from one stream.

### Work

Two instances of your existing, validated `Book` — imported, not reimplemented. Everything it does must survive: full snapshot replacement, incremental add/cancel, trade consumption, order-ID tracking, the `__H_` hidden-residual, `__NEG_` traded-placeholder and `__AGG_` deep-residual synthetics, phase and circuit-band state.

```python
from dataclasses import dataclass
from existing_mm.mm_backtest import Book


@dataclass
class DualBookState:
    ready: Book
    future: Book
    # Exchange-time of the last event applied to each leg.
    last_ready_exch: int | None = None
    last_future_exch: int | None = None
    # Knowledge-time (running max of capture_ts) per leg. This is the clock
    # that governs what the strategy may see -- see Phase 7.
    know_ready: int = 0
    know_future: int = 0
    # Last heartbeat / sequence evidence per leg, for feed-health detection.
    last_hb_ready: int = 0
    last_hb_future: int = 0
```

**Validate each book independently against the standard the single-instrument engine already meets.** Your documented benchmark is that 97.9% of trades print inside the reconstructed pre-trade touch. Reproduce that per leg. If the futures leg does not reach a comparable figure, the futures feed differs structurally and Phase 1's answers were wrong.

### Gate 6

- [ ] Per leg: reconstructed BBO matches the next snapshot's touch within tolerance.
- [ ] Per leg: trade-inside-touch rate reported and comparable to the 97.9% benchmark.
- [ ] No crossed or negative books outside valid transition states.
- [ ] `b2_ignored` (unresolvable cancels) reported per leg as a fraction of cancels.
- [ ] Phase/halt state tracked on both legs.
- [ ] Timestamps never move backward.

---

## Phase 7 — Synchronized cross-asset state, on knowledge time

**Purpose.** This is where all three source plans go wrong, and it is the most likely single cause of a backtest that will not survive live.

### The problem

The two legs are asynchronous. When a futures event arrives, the ready book holds whatever state you last saw. If you compute the basis on **exchange time**, you are differencing a price you knew against one you did not yet know — and the bias is not random. The stale leg lags the moving leg precisely when the move happens, so the error appears as a **dislocation that was never tradeable**, systematically in the profitable-looking direction.

### Work

Maintain **one joint knowledge clock**, because a real trading system has one view of time:

```python
# Knowledge time is the running max of capture_ts across BOTH legs: a real
# system knows everything that has arrived, regardless of which feed carried it.
know = max(know, int(event.ts_cap))

# Per-leg knowledge age measures how stale each book is in OUR view, not the
# exchange's. This gates cross-asset features and, per Phase 14, the OMS.
ready_know_age_ms = know - know_ready
future_know_age_ms = know - know_future
```

### Feed health is not book quiescence — and this distinction is essential

A naive age threshold conflates two completely different situations:

- **Quiet book.** Nothing happened. KTML's median trade gap is 12.6 s; a 2-second silence there is normal and the book is perfectly valid. Cancelling on that basis would have you pulling quotes continuously in exactly the wide-spread names you most want to make markets in.
- **Dead feed.** The gateway stalled, the session dropped, or messages were lost. The book is stale and you do not know it.

They are distinguished by **heartbeats and sequence continuity**, not by time since the last book event:

```python
# Heartbeat gap: the feed itself has gone silent. UA001 heartbeats arrive on a
# known interval regardless of trading activity, so a gap here means the FEED
# is unhealthy, not that the market is quiet.
feed_healthy_ready = (know - last_hb_ready) <= HEARTBEAT_TIMEOUT_MS
feed_healthy_future = (know - last_hb_future) <= HEARTBEAT_TIMEOUT_MS

# Sequence continuity: a gap in appl_seq / appl_last_seq means dropped
# messages, so the reconstructed book is provably incomplete.
seq_continuous_ready = (expected_seq_ready == observed_seq_ready)
seq_continuous_future = (expected_seq_future == observed_seq_future)

# Book freshness is a SEPARATE and weaker condition, used only for basis
# computation -- not for the risk-off decision.
book_fresh_ready = ready_know_age_ms <= MAX_CROSS_BOOK_AGE_MS
book_fresh_future = future_know_age_ms <= MAX_CROSS_BOOK_AGE_MS
```

Emit one row per event to `bamm/03_books/{date}_{underlying}_sync.parquet`:

```
ts_exch, know, trigger_instrument,
ready_last_exch, future_last_exch,
ready_know_age_ms, future_know_age_ms,
ready_hb_age_ms, future_hb_age_ms,
seq_gap_ready, seq_gap_future,
ready_bid, ready_bid_qty, ready_ask, ready_ask_qty, ready_mid, ready_microprice,
future_bid, future_bid_qty, future_ask, future_ask_qty, future_mid, future_microprice,
ready_phase, future_phase, ready_pinned, future_pinned,
feed_healthy_ready, feed_healthy_future,
book_fresh_ready, book_fresh_future,
cross_state_valid
```

```python
# MAX_CROSS_BOOK_AGE_MS must be calibrated PER SYMBOL from the observed
# inter-event gap distribution. A single global 500 ms is wrong: it is loose
# for FFC (0.8 s median trade gap) and absurdly tight for KTML (12.6 s).
# Suggested starting rule: p95 of the symbol's own inter-book-event gap.
cross_state_valid = (
    feed_healthy_ready and feed_healthy_future
    and seq_continuous_ready and seq_continuous_future
    and book_fresh_ready and book_fresh_future
    and ready_two_sided and future_two_sided
    and not ready_pinned and not future_pinned
)
```

**Retain stale rows with their ages** rather than dropping them. You need the sensitivity curve: PnL as a function of `MAX_CROSS_BOOK_AGE_MS`. If the edge depends on trading stale-leg states, it does not exist.

### Gate 7

- [ ] Sync table produced for one day, one pair.
- [ ] Both knowledge ages and both heartbeat ages plotted; median and p99 recorded.
- [ ] Per-symbol `MAX_CROSS_BOOK_AGE_MS` calibrated from that symbol's gap distribution, not assumed.
- [ ] Fraction of events with `cross_state_valid` recorded, and **decomposed by which condition failed** — if most failures are `book_fresh` in a thin name, the threshold is wrong, not the feed.
- [ ] Basis computed on knowledge time vs exchange time, plotted together — **the gap between them is your lookahead, quantified**.
- [ ] 100 random event windows manually inspected.

---

## Phase 8 — Basis computation

**Purpose.** Turn two books plus a carry curve into the tradeable numbers.

### Work

**Denominator convention: every basis measure is in bps of `ready_mid`.** Plan 3 mixes `ready_mid`, `ready_ask` and `future_ask` across its definitions, which makes the measures non-comparable and thresholds meaningless.

```python
# Mid-to-mid, diagnostic only. Contains the carry, so it is NOT a signal.
basis_raw_bps = (future_mid - ready_mid) / ready_mid * 1e4

# Deviation from fitted carry. THE primary signal. Never feed raw prices to a
# model -- only this residual.
basis_resid_bps = (future_mid - theoretical_future(ready_mid, r_fit, dte, pv_div)) / ready_mid * 1e4
```

Mid-based residual is a *signal*, not what you can trade. Store both executable directions against theoretical value at the **executable spot**:

```python
# Direction A -- future rich: SELL future at its bid, BUY ready at its ask.
exec_edge_sell_fut_bps = (
    future_bid - theoretical_future(ready_ask, r_fit, dte, pv_div)
) / ready_mid * 1e4

# Direction B -- future cheap: BUY future at its ask, SELL ready at its bid.
# Requires ready_shortable; otherwise available only against existing inventory.
exec_edge_buy_fut_bps = (
    theoretical_future(ready_bid, r_fit, dte, pv_div) - future_ask
) / ready_mid * 1e4

# Net of BOTH legs' fees, and of borrow cost where Direction B needs a short.
net_exec_edge_a_bps = exec_edge_sell_fut_bps - ready_fee_bps - future_fee_bps
net_exec_edge_b_bps = exec_edge_buy_fut_bps - ready_fee_bps - future_fee_bps - borrow_cost_bps
```

Also store `future_ask - ready_bid` and `future_bid - ready_ask` for diagnostics, plus z-scores of `basis_resid_bps` over 1 m / 5 m / 30 m, and a longer-window `sigma_basis` for the Phase 14 risk thresholds.

**On z-scores, a warning that now carries operational weight.** A z-score is a *relative* measure. A quiet name with tiny basis volatility prints |z| > 2 constantly on moves worth a fraction of a basis point, and **near expiry basis volatility collapses so z explodes on noise**. Plan 1's `pct_time_basis_wins = % of time |basis_z| > 2` is therefore not an opportunity screen — and, critically, a pure-z catastrophe threshold in Phase 14 would fire spuriously every expiry week. **All thresholds, gates and screens are in bps net of fees, with z as a secondary condition; never z alone.**

### Gate 8

- [ ] Ten timestamps hand-validated end to end: spot, DTE, rate, PV dividends, `F_theo`, residual, both executable edges, both net of fees and borrow.
- [ ] `basis_resid_bps` distribution plotted; centred near zero by construction; tails inspected.
- [ ] Behaviour inspected around ex-dividend, expiry week, and roll.
- [ ] `sigma_basis` plotted against DTE — confirm the collapse near expiry, and record it, because Phase 14's thresholds must survive it.

---

## Phase 9 — Feature store

**Purpose.** One row per valid event, point-in-time on knowledge time, no lookahead.

Adopted from Plan 3 with corrections. Date-partitioned parquet, queried in place via a DuckDB view.

### Schema

**A. Context**

`ts_exch`, `know`, `trigger_instrument`, `ready_know_age_ms`, `future_know_age_ms`, `ready_hb_age_ms`, `future_hb_age_ms`, `dte_calendar`, `dte_trading`, `days_to_roll`, `settlement_type`, `r_fit`, `r_kibor`, `pv_div`, `cross_state_valid`, `sess_elapsed_frac`, `contract_month`

**B. Basis (Phase 8)**

`basis_raw_bps`, `basis_resid_bps`, `exec_edge_sell_fut_bps`, `exec_edge_buy_fut_bps`, `net_exec_edge_a_bps`, `net_exec_edge_b_bps`, `basis_z_1m`, `basis_z_5m`, `basis_z_30m`, `sigma_basis`, `basis_velocity_100ms`, `basis_velocity_1s`, `basis_velocity_5s`

**C. Microstructure — both legs, `ready_` and `future_` prefixed**

`spread_bps`, `microprice`, `microprice_dev_bps`, `obi_1`, `obi_5`, `obi_deep`, `bid_depth_5`, `ask_depth_5`, `book_slope_bid`, `book_slope_ask`, `trade_flow_1s`, `trade_flow_10s`, `trade_flow_60s`, `toxicity_60s`, `trade_count_60s`, `time_since_trade_ms`, `volatility_ema`, `kyle_lambda`

```python
# Microprice weights each side's price by the OPPOSITE side's size: heavy bid
# depth pulls fair value toward the ask.
microprice = (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)

# Updates only on mid CHANGES, matching micro_mm.observe(). Updating on every
# event injects zeros and biases sigma downward.
if mid != last_mid and last_mid > 0:
    ret = (mid - last_mid) / last_mid
    ema_var = alpha * ret * ret + (1 - alpha) * ema_var
    sigma = sqrt(ema_var)
```

**D. Fast-market and risk state (new in v2, consumed by Phases 13 and 14)**

`ready_ret_1s_bps`, `ready_ret_5s_bps`, `ready_ret_30s_bps`, `future_ret_1s_bps`, `spread_ratio_vs_median` (both legs), `depth_ratio_vs_median` (both legs), `trade_burst_z`, `sigma_short` (fast EMA), `sigma_long` (slow EMA), `vol_ratio = sigma_short / sigma_long`

`vol_ratio` is the input to dynamic margin in Phase 13 and to the catastrophe detector in Phase 14. It exists so that both use the same measured quantity rather than two independently invented ones.

**E. Cross-asset**

`relative_obi_1`, `relative_obi_5`, `microprice_gap_bps`, `cross_trade_flow_1s`, `relative_trade_flow_1s`, `corr_lag_{-500,-250,-100,-50,-10,0,+10,+50,+100,+250,+500}ms`, `flow_lead_beta_60m` (rolling coefficient from the Phase 3c regression)

`flow_lead_beta_60m` is the important one and is missing from all three source plans: **the lead-lag relationship is a strategy parameter, and it drifts.** A single 9-month estimate is not enough; you need it time-varying so θ can respond.

### Discipline

- Every feature declares: unit, lookback, and **which clock it uses**, in `bamm/config/feature_spec.yaml`. A feature whose clock is undeclared is a leak waiting to be found.
- Features computed in a **forward-only single pass**, in the order the simulator will see them. No `.shift(-n)`, no `rolling(center=True)`, no whole-day group operations.
- Every rolling window **backward-looking on knowledge time**, including windows on the other leg.

**The one that is secretly forward-looking if you are careless:** any feature derived from a snapshot must use the snapshot's *capture* time, not its `orig_time`. Snapshots carry second-precision `orig_time` and arrive later; keying features on `orig_time` back-dates information you did not have.

### Gate 9

- [ ] `feature_spec.yaml` complete: unit, lookback, clock, per column.
- [ ] Null and infinite rates reported per column.
- [ ] Distributions plotted; behaviour checked at open, close, halts, expiry, roll, ex-dividend.
- [ ] **Lookahead audit:** recompute features for the first *N* events using only the first *N* rows; results identical to the full-day run.
- [ ] Feature-vs-basis correlation matrix inspected; |ρ| > 0.9 pairs flagged for pruning.

---

## Phase 10 — Labels, generated separately

**Purpose.** Never construct forward-looking labels in the same function that computes real-time features. Physical separation is the cheapest leakage defence available.

### Label A — ready passive-fill markout (toxicity)

Plan 3 defines this as `Ready_Mid(t+10s) − Ready_Best_Bid(t)`, which is not a markout of a fill: it ignores whether a fill would have occurred, ignores queue position, and is unsigned by side.

```
decision_know_ts, side, hypothetical_price, queue_ahead_qty,
fill_exch_ts, fill_price, mid_at_fill,
mid_at_fill_plus_{30s,60s,300s}, markout_bps_{30s,60s,300s}
```

Sign convention: for a passive **buy** fill, `markout = (mid_h − fill_price) / fill_price`. Negative means the market moved against you after you bought. Same convention both sides.

### Label B — legging slippage

```
fill_leg, fill_exch_ts, fill_price, hedge_side, hedge_qty,
seen_touch_at_decision, hedge_arrival_ts,
hedge_vwap, hedge_slippage_bps, hedge_unfilled_qty, hedge_levels_walked
```

**Draw the latency from `LatencyModel`, never a hard-coded 10 ms.** Your priors are ~45 ms one-way with 2% tail draws averaging 400 ms. Training on 10 ms teaches an execution you cannot achieve, and it will *understate* slippage — telling you the arb is safe exactly when it is not, which is the precise failure the label exists to prevent. Generate at several latency draws so slippage sensitivity is measurable.

Walk the book with `Book.liquidation_value`, which already handles level-walking, fees, `__H_` inclusion, `__AGG_` exclusion, and unfilled residual.

**Both hedge directions must be labelled.** Plan 2 covers only futures-fill → ready-hedge. Given OGDC's ready book (1,686M notional, 4.1 bps) that is the *cheap* direction. The expensive and dangerous direction is ready fill → hedge into thin futures.

### Censoring — mark invalid, do not silently drop

- Horizon crosses session end.
- Either leg halted, in auction, or pinned at a circuit band.
- `cross_state_valid` false at decision time.
- Book cannot absorb the hedge quantity.
- Reference data missing.
- Contract rolls within the horizon.
- **Risk-off state active (Phase 14):** a sample the live system would never have taken must not train the model that governs it.

Censoring is itself informative: a pattern in *which* samples get censored is a finding about when the strategy cannot operate.

### Gate 10

- [ ] 20 label examples replayed by hand against the raw stream.
- [ ] Sign conventions verified on a known adverse and a known favourable case.
- [ ] Censoring rates reported by reason.
- [ ] Label distributions inspected for the impossible: markouts beyond the day's range, zero-slippage hedges through thin books.

---

## Phase 11 — Universe screening and persistence

**Purpose.** Choose what to trade. Transplanted from Plan 1, the only source plan that has this — and it encodes a lesson you already paid for.

Plan 2 and Plan 3 go straight from one symbol on one day to mechanical universe expansion. That discards two documented findings: KTML swung 236× in ceiling between two dates, and **symbol ranking inverts with fee level** (KTML rank 1 at 35 bps, rank 6 at 2 bps; FFC rank 5 and rank 1 respectively).

### Work

Run the full 9 months with the fee grid, splitting by `has_futures`:

**Per symbol-day:** existing `ticker_stats_core` outputs, plus `hedged_ceiling_pkr_rt_*` (Phase 3 definition, no pairing adjustment), `pct_time_hedged_viable`, `futures_depth_absorption`, `flow_lead_beta`, `direction_b_available`.

**Per symbol, across days:**

```
pct_days_top20          -- fraction of days in the daily top 20 by hedged ceiling
ceiling_median, ceiling_iqr, iqr_over_median   -- episodic names have huge ratios
rank_autocorr_t_t5      -- rank stability at 5-day lag
n_days_tradeable        -- days with a two-sided futures book and viable edge
pct_days_risk_off       -- days the Phase 14 gates would have suppressed quoting
```

`pct_days_risk_off` is new in v2 and matters: a name whose ceiling is concentrated in exactly the sessions your risk gates would have shut down has no accessible ceiling at all.

Expect three populations: stable-attractive (the watchlist), episodic (needs a trigger, not a standing quote), and dead.

**Rank on `pct_days_top20`, never on a single day.** Separate watchlist per fee scenario.

### Gate 11

- [ ] 9-month screen complete for both universes.
- [ ] Persistence metrics computed; three populations visible.
- [ ] Watchlist frozen per fee scenario, with the selection rule written down *before* looking at backtest PnL.
- [ ] The top-ranked name at `rt_2p00` checked for the KTML pathology.
- [ ] Ceiling recomputed net of `pct_days_risk_off` — the accessible ceiling, not the theoretical one.

---

## Phase 12 — BAMM simulator

**Purpose.** A new engine in `bamm/06_simulator/bamm_backtest.py` reusing validated components rather than replacing them.

### State

```
book_ready, book_future
work_ready: {side: MyOrder}, work_future: {side: MyOrder}
pos_ready, pos_future, net_delta
cash, cash_available, margin_posted, margin_required, financing_accrual
pending          -- one heap for ALL in-flight messages, both legs
know             -- ONE joint knowledge clock
risk_state       -- NORMAL | STALE | CATASTROPHE | EXPIRY_LOCK | MARGIN_LOCK
eod_ready, eod_future
```

### Event types on the heap

`MARKET_EVENT`, `ORDER_ARRIVAL`, `CANCEL_ARRIVAL`, `CANCEL_ACK`, `HEDGE_ARRIVAL`, `MARGIN_CALL`, plus a monotone sequence tiebreaker so the heap never compares payload objects.

### Cancel-all must be modelled with latency, not as an instant

This is the subtlety that makes risk-off gates real rather than decorative:

```python
# CANCEL_ALL is not instantaneous. Each cancel draws its own send latency and
# lands independently; between the decision and the landing, every order
# remains fully fillable. This is exactly the in-flight cancel risk the
# single-instrument engine already models -- reuse it, do not bypass it.
def cancel_all(self, ts_know, reason):
    for leg in ("ready", "future"):
        for side, order in list(self.work[leg].items()):
            if order.cancel_at is None:
                a_out = self.lat.draw_out()
                order.cancel_at = ts_know + a_out
                self._push(order.cancel_at, "CANCEL", (leg, side, order.oid))
    self.stats[f"cancel_all_{reason}"] += 1
```

**The window between deciding to pull and the cancels landing is where catastrophe losses actually occur**, and a simulator that treats cancel-all as instant will understate exactly the tail the gates exist to control.

### Net exposure

```python
# beta comes from contract_multiplier and lot sizes in the contract master.
# Do NOT assume 1.0, and do not add raw share counts across instruments.
net_delta = pos_ready + beta * pos_future
```

Four limits, all enforced: per-leg position, net delta, gross notional (drives margin), and available cash.

### Own-fill callback

Your handoff lists this as unbuilt, and BAMM cannot work without it: the hedge decision is triggered *by* a fill, not by the next market event.

`SIMPLIFICATION:` the existing engine reads `self.pos` in real time, so the strategy "knows" a fill instantly rather than one ack-latency later. Acceptable at a 5.6 s median trade gap for single-instrument MM. **Not acceptable for BAMM**, where the fill triggers the hedge and the ack delay *is* the legging window. Model the fill ack explicitly.

### Hedge policy as a swappable object

```python
class HedgePolicy:
    def on_fill(self, fill, dual_state, model_b) -> HedgeDecision:
        """Return: hedge aggressively now, post passively, or wait."""
```

Implement `AlwaysAggressive` (cost upper bound), `AlwaysPassive` (lower bound, maximum legging risk), and `EdgeAware` — comparing `remaining_edge − predicted_hedge_cost` against `sigma_basis * sqrt(expected_wait)`, with hard time and delta backstops escalating to aggressive.

**Report PnL under all three.** If `AlwaysAggressive` is materially worse, the passive-completion logic is where your edge lives. If it is not worse, you have saved yourself considerable complexity.

Note the asymmetry: when your ready quote fills, your futures quote may still be live. The real decision is whether to wait for the resting other leg or cross now. "Instantly fire a hedge" hardcodes the expensive branch and undoes the premise of the design.

### Gate 12 — deterministic unit scenarios

- [ ] Flat basis, no fills, no orders leak.
- [ ] Future rich → futures ask fills → hedge scheduled → ready book unchanged on arrival.
- [ ] Same, ready touch vanished on arrival.
- [ ] Same, hedge walks several levels.
- [ ] Same, hedge cannot fully fill.
- [ ] Future cheap — full symmetric case.
- [ ] Cancel and replacement arrive out of order.
- [ ] Fill occurs before the cancel lands (in-flight cancel risk).
- [ ] Cancel ack arrives after the order already filled.
- [ ] **Cancels target a specific `oid`, not a side** — this race was found and fixed once already; do not reintroduce it.
- [ ] **`cancel_all` issued, then a fill lands before the cancels do** — position and PnL correct.
- [ ] Halt on one leg while the other trades.
- [ ] Roll date inside the simulated day.
- [ ] Margin call event forces a reduction mid-session without a crash.

---

## Phase 13 — Costs: fees, financing, dynamic margin

**Purpose.** At a ~3 bps gross edge, cost modelling is not accounting detail; it is the result.

### Fees

**Per-leg schedules, separately parameterised.** Ready retail is ~17.73 bps/side, dominated by commission; the TREC own-account path is ~0.78 bps/side (~1.6 bps round trip) because commission and its SST vanish while the exchange/regulatory stack survives. Verify the futures-side equivalents independently — do not assume they match.

Add borrow cost on any Direction B position requiring a ready short.

### Financing

Long ready funded at your actual cost; short futures earns/pays the embedded rate. Use the same `r_fit` from Phase 4 — the strategy's carry P&L and its fair-value model must use one consistent number, or the backtest books a profit the pricing model says does not exist.

### Dynamic VaR margin (rewritten in v2)

NCCPL does not use a fixed percentage. It uses VaR-based margins that **expand intraday when volatility spikes** — which is precisely when a basis dislocation makes your strategy want maximum capital. A static `margin_pct` assumes free capital that will not be there; the order gets rejected for insufficient funds and you are left half-hedged, holding one naked leg in a fast market.

```python
def margin_pct(base_pct, vol_ratio, scan_range, floor_pct):
    """VaR-style margin that expands with realised volatility.

    vol_ratio = sigma_short / sigma_long, from the Phase 9 feature set.
    Deliberately uses the SAME measured quantity as the Phase 14 catastrophe
    detector, so capital and risk logic cannot disagree about the regime.
    """
    scaled = base_pct * max(1.0, vol_ratio)
    return min(max(scaled, floor_pct), scan_range)


# Capacity is recomputed on every event, not fixed at session start.
max_gross_future_notional = (cash_available * allocation_frac) / margin_pct_now
```

Three second-order effects that the notional cap alone does not capture, and that decide whether you survive the day:

1. **Margin is called in cash, when you are losing.** Model it as a scheduled `MARGIN_CALL` cash flow, not merely a constraint. Maintain an explicit liquidity buffer; record every event where the buffer is breached.
2. **The doom loop.** Basis widens → margin requirement rises → you are forced to reduce → you unwind *into* the dislocation, realising the loss at the worst available price, which widens your loss further. This is a **sequencing** property of the simulator, not a cap: the forced reduction must execute through `liquidation_value` at real book prices, not be assumed away.
3. **Negative correlation between capital and opportunity.** Available capacity shrinks exactly when measured edge is largest. Any backtest that sizes on average-day margin will overstate the trade; report deployed capital conditioned on `basis_resid_bps` decile to make this visible.

`SIMPLIFICATION:` `vol_ratio`-scaled margin is a proxy. Production version — implement the published NCCPL VaR methodology with its actual scan ranges and concentration add-ons per contract, sourced into `contract_master` (Phase 2), and reconcile the simulated requirement against real broker margin statements once live.

### Never report a single PnL number

```
gross_spread_pnl_ready, gross_spread_pnl_future,
basis_convergence_pnl, inventory_pnl, legging_slippage,
ready_fees, future_fees, borrow_cost, financing_cost, dividend_cashflows,
margin_cost, forced_reduction_cost, eod_liquidation_cost, net_pnl
```

This decomposition is how you detect the KTML pathology, where "+724 PKR" decomposed into +7,646 directional (an accidental short through a −707 bps session) minus −6,922 of actual market-making losses. Without it, that run reads as a success.

### Gate 13

- [ ] A zero-price-movement round trip loses exactly its applicable fees plus slippage — to the paisa.
- [ ] Fee schedules independently verified per leg and per aggressive/passive side.
- [ ] Financing accrual reconciles against `r_fit` × average notional × days.
- [ ] Margin requirement **rises** in a simulated volatility spike and capacity shrinks accordingly.
- [ ] At least one scenario triggers a forced reduction; its cost appears in `forced_reduction_cost`.
- [ ] Decomposition sums exactly to `net_pnl`.

---

## Phase 14 — Strategy: `BasisAwareMM`

**Purpose.** Extend `MicrostructureMM` rather than replacing it. The microstructure theory remains the skeleton; the basis enters through the reservation price; risk-off is a separate, dominating state machine.

### 14.1 The risk-off state machine (new in v2 — build this first)

v1 of this document argued against cancelling and provided no risk-off state at all. That was wrong, and the distinction that repairs it is this:

> **Never cancel *asymmetrically* as a way of expressing a signal. Always cancel *symmetrically* as a way of expressing risk.**

Pulling one side because the basis moved makes you a one-way accumulator: you fill to your limit, commit capital, cannot respond, forfeit spread income on the pulled side, and cannot profit when the basis reverts through you. That objection stands and is unchanged.

Pulling *both* sides because you cannot see the hedge, because the market has gone disorderly, or because you are three days from deliverable expiry is not signal expression — it is survival, and it dominates every optimisation below.

**Good news on implementation:** `_requote` already contains exactly this mechanism. Its halt/band-pin gate cancels every working order through the normal latency path when `phase` is not continuous or `book.pinned()` is true. The risk-off gates below are **additional conditions on that existing branch**, not new machinery.

```python
# Evaluated BEFORE any quoting logic. Any true condition pulls BOTH legs,
# BOTH sides, through the normal latency path -- and suppresses new quotes
# until the condition has been false for COOLDOWN_MS.
def risk_off_reason(self, st):
    # (1) Existing gate, retained: halt, auction, or pinned at a circuit band.
    if st.ready_phase not in (None, "CONTINUOUS_AUCTION"): return "READY_PHASE"
    if st.future_phase not in (None, "CONTINUOUS_AUCTION"): return "FUTURE_PHASE"
    if st.ready_pinned or st.future_pinned: return "PINNED"

    # (2) Cannot see the hedge -> do not provide liquidity. Feed health and
    # sequence continuity, NOT mere book quiescence (see Phase 7): a 12 s gap
    # in KTML is a quiet market, not a dead feed.
    if not st.cross_state_valid: return "STALE"

    # (3) Catastrophic dislocation. A deep passive fill in an emerging market
    # is not a gift -- it is a falling knife, a limit-down gap, or an
    # informed sweep. Threshold is a bps FLOOR combined with a sigma
    # multiple, because sigma_basis collapses near expiry and a pure
    # z-threshold would fire on noise every roll week.
    catastrophe_bps = max(K_SIGMA * st.sigma_basis_bps, CATASTROPHE_FLOOR_BPS)
    if abs(st.basis_resid_bps) > catastrophe_bps: return "DISLOCATION"

    # (4) Fast market on either leg: price velocity, spread blowout, depth
    # collapse, or trade burst.
    if abs(st.ready_ret_5s_bps) > FAST_MKT_BPS: return "FAST_MARKET_READY"
    if abs(st.future_ret_5s_bps) > FAST_MKT_BPS: return "FAST_MARKET_FUTURE"
    if st.ready_spread_ratio > SPREAD_BLOWOUT_X: return "SPREAD_BLOWOUT"
    if st.future_depth_ratio < DEPTH_COLLAPSE_X: return "DEPTH_COLLAPSE"

    # (5) Deliverable-futures expiry lock. DFC shorts must be physically
    # delivered; in roll week borrow tightens and shorts get squeezed. Do not
    # let the machine carry a DFC short into expiry -- hand roll week to a
    # human or flatten before it.
    # dfc_lock_days is a CONFIGURABLE SAFETY DEFAULT (ship at 5), resolved
    # per contract from the contract master, not a universal law. It may be
    # tightened or relaxed per (underlying, contract) based on observed
    # roll-week borrow behaviour -- but only via config with a logged
    # rationale, never inline, and never below the exchange's own
    # delivery-notice deadline for that contract.
    if st.settlement_type == "deliverable" and st.dte_trading <= st.dfc_lock_days:
        return "EXPIRY_LOCK"

    # (6) Margin: capacity exhausted or buffer breached (Phase 13).
    if st.margin_required > st.margin_available * MARGIN_HEADROOM:
        return "MARGIN_LOCK"

    return None
```

Notes that matter for calibration:

- **`EXPIRY_LOCK` should be a taper before it is a cliff.** A hard stop at `DTE_trading <= 5` is correct as v1 behaviour and is what the gate above implements. The production refinement is to scale `max_inv` down linearly from ~10 trading days out, forbid *new* futures shorts inside the lock window while allowing existing hedged pairs to be closed, and require that any DFC short be fully covered by deliverable ready inventory. A blunt cliff can strand you holding a position you are forbidden to adjust — so the lock must permit risk-reducing actions and forbid only risk-increasing ones.
- **Cash-settled contracts do not need the delivery lock**, but still need a roll-week volatility taper. Do not apply the same rule to both by accident.
- **All thresholds need hysteresis and a cooldown**, or you will thrash at the boundary — the same objection that motivates continuous skew below.
- **Message-rate and margin cost of deep resting quotes.** Even absent catastrophe, extremely deep quotes consume exchange message budget and margin allocation for near-zero fill probability. Cap the number of resting price levels per side and drop quotes beyond a configurable distance from the touch — a housekeeping rule, distinct from the risk gates.

### 14.2 Continuous skew, not a regime switch

All three source plans use a binary threshold on `|basis_z| > 2`. A hard flip causes oscillation at the boundary, cancel/replace thrash, and hysteresis patches. Collapse it into the reservation price and the "regimes" emerge for free:

```python
# D is the dislocation versus FITTED CARRY, not versus zero. Skewing toward
# zero basis when fair basis is +80 bps means systematically buying futures rich.
D = (future_mid - theoretical_future(ready_mid, r_fit, dte, pv_div))

# theta is damped by the measured flow-lead. If futures flow predicts ready
# price, a large part of D is news rather than tradeable dislocation, and
# leaning into it on the ready side is racing to be picked off.
theta_eff = theta * (1.0 - flow_lead_beta_norm)

# Joint inventory drives BOTH legs. The two books are coupled through net
# delta; their skews cannot be computed independently.
r_ready = fair_ready + theta_eff * D - gamma * sigma_resid**2 * tau * net_delta
r_future = fair_future - theta_eff * D - gamma * sigma_resid**2 * tau * net_delta
```

`sigma_resid` — not price sigma. The entire economic case for BAMM is that hedged inventory carries *basis* variance, not price variance. Using price sigma sizes you as though unhedged and discards the benefit you built the system for.

**`D ≈ 0` yields symmetric two-sided MM on both legs. Large `D` yields aggressive skew. Very large `D` hits the catastrophe gate above and pulls everything.** Three regimes, one continuous function plus one hard stop, no boundary oscillation in the normal range.

### 14.3 Viability and direction gates

- Per-leg viability: the ready-leg gate must include expected hedge cost, since an unhedgeable fill is not a market-making fill.
- Direction B suppressed entirely where `ready_shortable` is false, except to the extent of existing long ready inventory.
- **Declining to quote remains a first-class output.**

### 14.4 Reuse, do not reimplement

`round_tick` (bids floor, asks ceil, so rounding never makes a quote more aggressive), post-only clipping one tick inside the opposite touch, circuit-band clamping, and the no-churn comparison in `_requote`. Plan 1's skew snippet reimplements all of this and gets it wrong — it caps a *ready* price against a *futures* price (different books at a structural premium), and uses `min` where a floor on an ask requires `max`, so its "cap" makes the quote more aggressive rather than less.

### Gate 14

- [ ] `theta = 0` reproduces `MicrostructureMM` behaviour on the ready leg exactly.
- [ ] Every risk-off condition has a unit test that fires it and confirms both legs, both sides, are cancelled.
- [ ] Cancel-all latency is modelled; a fill during the in-flight window is handled correctly.
- [ ] Risk-off frequency per condition reported per symbol-day — **if `STALE` fires more than a few percent of the session, the Phase 7 threshold is miscalibrated for that symbol, not the feed broken**.
- [ ] Cooldown prevents flapping; requote rate per hour is sane at every `D`.
- [ ] No side is ever cancelled *asymmetrically* by the skew logic.
- [ ] `EXPIRY_LOCK` permits risk-reducing actions and forbids only risk-increasing ones.
- [ ] Quote prices always on the tick grid, inside circuit bands, never marketable against the visible book.
- [ ] `net_delta` respects its limit under adversarial fill sequences.

---

## Phase 15 — Models

**Purpose.** Two narrow predictors replacing heuristic components, per your standing ML doctrine. Not an end-to-end learner.

**Model A — passive markout / toxicity.** Predicts Label A. Outputs expected markout in bps and P(markout worse than limit).

**Model B — legging slippage.** Predicts Label B. Outputs expected slippage, **p90 slippage**, and P(incomplete hedge). For execution safety the tail matters more than the mean; a hedge policy sized on the mean is under-reserved by construction.

### Method

- Baselines in order: median label by state, then Ridge/logistic (coefficients readable as weights), then LightGBM only if it beats linear out of sample.
- Pool across symbols with per-symbol z-scored features; per-symbol data is too thin.
- **Walk-forward, expanding window, monthly roll.** Never shuffle event rows. Separate contracts and roll periods across folds.
- **Embargo ≥ 1 day**, longer than the longest label horizon plus censoring window. A 1-day embargo with 300 s labels is adequate on horizon grounds but does nothing about *overlapping* samples within a fold — use sample weighting or subsampling.
- Two mandatory leakage tests: shuffle labels → IC collapses to zero; remove the most recent 30 s of features → performance must not *improve*.
- Feature selection by permutation importance on validation folds only; drop one of any |ρ| > 0.9 pair.
- **Train only on samples the live system would have taken** — risk-off periods are censored (Phase 10). A model trained on catastrophe-period data will confidently predict regimes it will never be asked about.

### Gate 15

- [ ] Both leakage tests pass.
- [ ] OOS IC in the 0.02–0.05 range at 60 s. **0.10+ means leakage until proven otherwise.**
- [ ] A model is accepted **only** if it improves simulated out-of-sample net PnL after costs — never on RMSE, AUC or IC alone.
- [ ] Models integrated as gates, not as quote generators:

```python
# Model A gates passive quoting.
if predicted_markout_bps < -toxicity_limit:
    suppress_side = True

# Model B gates the basis trade. The correct decision is often "no quote".
expected_net_edge = (
    executable_basis_edge_bps
    - predicted_legging_slippage_p90_bps
    - total_fee_bps
    - safety_buffer_bps
)
if expected_net_edge <= 0:
    attempt_basis_trade = False
```

---

## Phase 16 — Controlled experiments

**Purpose.** Find out exactly where the profit disappears — because it will.

Run in this order, one change at a time, on held-out months:

1. No ML, no basis logic (pure dual-leg MM) — the floor.
2. Basis logic with a perfect instantaneous hedge — the ceiling.
3. Add deterministic hedge latency.
4. Add stochastic latency (`LatencyModel` with tails).
5. Add book-walk hedge slippage.
6. Add full per-leg fees and borrow.
7. Add financing and **static** margin.
8. Switch to **dynamic** margin with cash calls.
9. Add the risk-off state machine.
10. Add Model A only.
11. Add Model B only.
12. Add both.

Steps 7→8 and 8→9 are the two that v1 could not measure. Expect step 8 to remove PnL and step 9 to remove *tail* — judge them on different statistics.

### Robustness and negative controls

- `fill_on_crossing_adds` on vs off — the most assumption-heavy fill rule you have.
- `at_price_mode` in `never` / `queue` / `always` — brackets the queue assumption.
- Latency ×2 and ×5.
- Fees ×2.
- Displayed futures depth halved.
- `MAX_CROSS_BOOK_AGE_MS` swept 100 ms → 5 s.
- Catastrophe threshold swept; **plot PnL and max drawdown against it separately** — the whole point of the gate is that it costs mean and buys tail.
- **Negative control:** disable dividend adjustment. PnL must *degrade*. If not, the logic is dead code.
- **Negative control:** `theta = 0`. Basis-attributed PnL must fall to zero. If not, PnL is mis-attributed.
- **Negative control:** shift one leg's timestamps forward by 1 s. Apparent edge should collapse. If it survives, you are trading a stale-state artefact.
- **Negative control (v2):** disable all risk-off gates. Mean PnL may rise; **max drawdown and worst-day loss must worsen materially**. If they do not, either the gates are not firing or your data contains no disorderly sessions — and the second possibility means your sample is unrepresentative, not that the risk is absent.

### Gate 16

- [ ] Every step's PnL delta recorded with decomposition.
- [ ] All four negative controls behave as specified.
- [ ] The step where PnL turns negative identified by name.
- [ ] Fill provenance audited by `reason` — if most PnL comes from `crossing_add` fills, the edge is fragile and rests on a counterfactual.
- [ ] Tail statistics (max drawdown, worst day, worst hedge miss) reported alongside mean PnL at every step.

---

## Phase 17 — Scale out

Expand only after Phase 16 is clean on one pair:

1 pair × 1 day → 1 pair × 20 days → 1 pair × full contract life (including roll) → 5 pairs × 20 days → full surviving universe × 9 months.

Deliberately include: dividend days, expiry week, roll days, high-volatility days, low-volume days, halted sessions, one-sided books, feed-outage days, and days with missing reference data.

**Do not tune thresholds on 2026-06-30 and then report 2026-06-30 as evidence.** Parameter selection happens on training folds only; held-out months are looked at once.

`SIMPLIFICATION:` per-symbol parameter fitting. Plan 1 proposes sweeping gamma (21 values) × kappa (21) × threshold × size × `max_inv` × `quiet_ms` **per symbol** on nine months of thin data, selected on "the validation set". Those optima will be noise. Production approach: pooled priors across symbols, per-symbol deviation only where a walk-forward test shows stability across folds, and three well-understood parameters in preference to twelve fitted ones. The risk-off thresholds specifically should be set from **risk tolerance, not from PnL optimisation** — optimising a catastrophe threshold on nine months of data fits it to the catastrophes that happened to occur.

### Gate 17

- [ ] Results stable across contracts and months, not driven by one window.
- [ ] Roll periods produce no discontinuity in PnL attribution.
- [ ] Per-symbol results consistent with the Phase 11 ranking — if screen and backtest disagree about which names are good, one of them is wrong and you must find out which.

---

## Phase 18 — Performance

**Correctness first. Profile before optimising.**

On the widely-repeated claim that this needs Rust: per symbol per day you have roughly 3.5k snapshot messages, ~4.4k updates and ~1.1k trades ≈ 9k events; two legs ≈ 18k. A month of one pair is ~380k events — seconds to minutes in Python, not days. That estimate confuses the **parser** workload (40M snapshot rows/day across 577 symbols, genuinely heavy) with the **backtest** workload (one pair at a time). The GIL is irrelevant to a single-threaded event loop.

When scale does bite — 179 days × N pairs × parameter sweeps — the first answer is multiprocessing across days, the same 3-worker pattern that took your parser to ~17 h.

Only if profiling proves otherwise:

- Polars/DuckDB for file filtering, stream normalisation, feature and label preparation, reporting.
- Numba or Rust for the dual-book replay loop, queue simulation, pending-message heap, and book-walk hedge execution.

Polars lazy expressions do not replace an inherently stateful order-book reconstruction; use Polars to *prepare* the stream and a compiled loop for the state machine.

### Gate 18

- [ ] Profile captured before any optimisation, with the hot phase named.
- [ ] Any optimised path reproduces pre-optimisation PnL to the paisa.
- [ ] The MCB regression still passes.

---

## Phase 19 — Regulatory and paper-trading ladder

**Start the regulatory workstream at Phase 1, in parallel.** It is measured in months and gates everything.

**TREC path (verified):** proprietary trading permitted; own-account pays no commission (hence no SST), leaving ~0.78 bps/side ≈ 1.6 bps round trip — the `rt_2p00` scenario. Requires a separate brokerage entity (BBI cannot hold the TREC), Rs 2.5M certificate + Rs 100k processing, Rs 35M net worth, Rs 15M liquid capital, plus ongoing compliance opex.

Live infrastructure not covered by the backtester: FIX order entry and session management, **kill switch**, state recovery on reconnect, position and cash reconciliation against the broker, real-time risk limits, margin monitoring against actual broker statements, alerting, and end-of-day recon.

The kill switch is the live counterpart of Phase 14's risk-off machine. It must be able to flatten both legs without the strategy's cooperation, and it must be tested under a real disconnect before Stage 2.

### Ladder

**Stage 1 — pure MM, no basis logic.** 5–10 watchlist names.
**Stage 2 — basis-aware quoting, no aggressive hedging.** Passive completion only.
**Stage 3 — full BAMM with hedge execution.**

**Validate each stage on fill rate, queue position, markout, and hedge slippage — not on PnL.** At a ~3 bps edge, PnL is far too noisy to confirm "within 20% of backtest" at these sample sizes; that criterion would pass or fail essentially at random. The microstructure metrics converge orders of magnitude faster and tell you specifically *which* assumption is wrong.

Additionally validate, because they are new in v2 and cannot be confirmed offline:

- Risk-off trigger frequency, live vs simulated, per condition.
- Actual broker margin requirement vs simulated, especially on a volatile session.
- Actual heartbeat intervals and feed-gap distribution vs the Phase 7 calibration.

Use live telemetry to refit `LatencyModel` (currently priors, not measurements) and the hedge policy. This is why `HedgePolicy` is a swappable object.

### Gate 19

- [ ] Realised fill rate within tolerance of simulated, per side, per leg.
- [ ] Realised queue position distribution matches the simulator's.
- [ ] Realised markout distribution matches Model A's predictions.
- [ ] Realised hedge slippage within Model B's predicted p90.
- [ ] Realised margin requirement within tolerance of the dynamic model.
- [ ] Kill switch tested under a live disconnect.

---

## The single assumption most likely to be fatal

**That the hedge exists at size.** Not the model, not the fees, not the ML. If the futures leg is thin, every fill you hedge gives back more than the ready spread earned, and BAMM is strictly worse than pure basis arb on the same names — and possibly worse than doing nothing.

Phase 3 exists to find that out for a few days of querying instead of three months of building. Run it first, and be willing to stop there.

## The second most likely

**That the basis dislocation is tradeable rather than informational.** If futures flow predicts ready price, then at the exact moment the model says "post an aggressive ready bid", the reason is that ready is about to reprice against you. Your toxicity gate cannot catch it, because the toxicity and the signal are the same event. Phase 3c measures this; `theta_eff` is a damping, not a cure.

## The third — new in v2

**That the strategy's worst sessions are survivable.** A hedged market maker in a thin emerging market does not die from a slow bleed; it dies in one session when the feed stalls, the basis gaps, margin doubles, and the hedge cannot be completed at any price. The Phase 14 gates and the Phase 13 margin model exist for that session, and their value will not show up in mean PnL — only in the tail. Judge them accordingly, and never optimise their thresholds on backtest returns.

---

## Immediate next session

1. Phase 0 — freeze, git tag, write and pass the MCB regression test.
2. Phase 1 — run the schema and symbol queries; fill in `data_contract.yaml` from actual results, **including heartbeat availability on both channels**.
3. Phase 2 — build the contract master for the surviving underlyings, reconciled two ways, with `settlement_type` and `ready_shortable` populated.
4. Phase 3 — **run the kill gate**, including the flow-lead regression. Report the five numbers in Gate 3 before writing any further code.

Do not begin the feature store, the simulator, or the models until Gate 3 passes.
