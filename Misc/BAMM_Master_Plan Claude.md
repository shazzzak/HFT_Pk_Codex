# Basis-Aware Market Maker (BAMM) — Master Execution Plan

**Big Byte Insights — PSX cross-instrument market making**
Supersedes Plan 1 (roadmap), Plan 2 (17-phase gated plan), Plan 3 (feature schema).

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
| **Cross-book staleness gated on knowledge time, not exchange time** | New | Plan 2 computes `ready_age_ms` from `ts_exch`; Plan 3 states features are "strictly based on `ts_exch`". Both differ a fresh price against one you had not yet received. Your measured feed latency is ~80 ms median with multi-second spikes, and the bias is systematic — the stale leg lags precisely when the move happens. |
| **Universe screening and persistence transplanted in (Phase 11)** | Plan 1 | Plan 2 and 3 have no screening step at all. This discards the KTML lesson (236× ceiling swing between two dates) and the finding that symbol ranking **inverts** with fee level. |
| **Paper-trading ladder and TREC workstream appended (Phase 19)** | Plan 1 | Plan 2 ends at performance optimization with no go-live path. Regulatory access is a months-long parallel workstream, not a final checkbox. |
| **Feature schema adopted with corrections** | Plan 3 | Plan 3's concrete named columns and formulas are its best contribution. Denominators made consistent; labels re-specified (see Phase 10). |
| **Continuous skew replaces the binary regime switch; never cancel a side** | New | Cancelling one side makes you a one-way accumulator: you fill to your limit, commit capital, then cannot respond, forfeit spread income on the pulled side, and cannot profit when the basis reverts *through* you. Cancel is the degenerate infinite-skew case and is strictly worse. |
| **Margin, per-leg fees, contract multiplier, settlement type made first-class** | New | On a $3–6M base, margin on the short futures leg binds. Deliverable vs cash-settled carries real delivery/squeeze risk near expiry. |
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

Write `test_mcb_regression.py` to pin the documented MCB result as an automated test, not a remembered number:

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
    # liquidation_clean False means equity_liquidated is an estimate, not
    # a realisable number; the regression is only meaningful when True.
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
- Do futures snapshots carry the same 10-level depth and the `AGG_BID`/`AGG_OFFER` aggregate rows? Your `Book.snapshot` deep-residual logic depends on them.
- Are `phase`, `prev_close`, and the circuit-breaker entry types present on the futures segment?
- Which timestamp columns are populated, and are `capture_ts` and `transact_time` both non-null on both segments?

### Gate 1 — record these values in `bamm/config/data_contract.yaml`

```yaml
ready_symbol_pattern:      # e.g. "OGDC"
future_symbol_examples:    # e.g. ["OGDC-JUL", "OGDC-JULB"]
ready_segment_code:        # e.g. "REG"
future_segment_codes:      # e.g. ["STOCK_DEL_FUT", "STOCK_CS_FUT"]
exchange_ts_snapshot:      # orig_time
exchange_ts_update:        # transact_time
exchange_ts_trade:         # transact_time
capture_ts_column:         # capture_ts
sequence_column_updates:   # appl_seq
sequence_column_snapshot:  # msg_seq
futures_has_initiator_tags:   # true / false
futures_has_agg_rows:         # true / false
futures_max_book_levels:      # int
timezone:                     # UTC
```

Do not continue until every field is filled from an actual query result.

---

## Phase 2 — Contract master

**Purpose.** The reference layer that says which two instruments belong together, and on what economic terms.

### Work

`bamm/01_reference/contract_master.parquet`, one row per `(trade_date, underlying, future_symbol)`:

| column | notes |
|---|---|
| `trade_date` | PSX trading date, not `date(ts_exch)` — a PSX day spans two UTC dates |
| `underlying` | e.g. `OGDC` |
| `ready_symbol` | from Phase 1 |
| `future_symbol` | from Phase 1 |
| `contract_month` | `2026-08` |
| `settlement_type` | `deliverable` / `cash_settled` — **first-class, not a footnote** |
| `expiry_date`, `last_trading_date`, `settlement_date` | from official PSX contract specs |
| `dte_calendar`, `dte_trading` | store both; act/365 uses calendar |
| `contract_multiplier`, `lot_size_ready`, `lot_size_future` | drives the hedge ratio β |
| `tick_size_ready`, `tick_size_future` | verify per segment; do not assume 0.01 on futures |
| `is_front_month`, `roll_flag` | explicit roll marking |

**Derive the pairing two independent ways and reconcile:** (a) suffix parsing of the symbol string, (b) the `segment` field the parser already maps. Where they disagree, raise — do not coalesce. A silent coalesce here maps the wrong contract to the wrong underlying and every downstream number is fiction.

`SIMPLIFICATION:` for the Phase 3 gate you may hand-build a single-underlying master. Production requires the full daily panel across all futures underlyings for the whole 9 months, sourced from official contract specifications rather than inferred from the tape.

### Gate 2

- [ ] Every futures symbol observed in the data maps to exactly one underlying.
- [ ] `expiry_date > trade_date` for every row.
- [ ] `dte_calendar` is strictly non-increasing within a contract as `trade_date` advances.
- [ ] Roll dates are explicit; front-month designation never flips back.
- [ ] `settlement_type` is populated for every contract.
- [ ] Suffix-derived and segment-derived pairings agree on 100% of rows.

---

## Phase 3 — THE ECONOMIC KILL GATE

**Purpose.** Determine whether this business exists before building infrastructure for it. This is the single highest-value phase in the document and it runs entirely on data you already have.

The thesis of BAMM is that hedging collapses the inventory penalty from price variance to *basis* variance, letting you quote tighter and carry more inventory in the high-volume names where the TREC-fee ceiling actually concentrates. That thesis fails if the hedge leg is too thin or too wide to use.

### Work

Sample ~20 trading days spread across the 9 months (include an expiry week, a roll, and a high-volatility day). For every underlying with a futures contract, compute:

**3a. Futures liquidity census**

- Number of days the futures symbol trades at all; trades/day; notional/day.
- Median and p90 futures L1 spread in bps.
- Median touch depth (shares and PKR) on each side.
- Median depth within 5 levels — this is what a hedge actually walks.
- Fraction of the continuous session with a two-sided futures book.

**3b. Hedged edge, in money**

For each ready trade, using the contemporaneous state of both books:

```
hedge_cost_bps   = future_half_spread_bps + future_fee_bps
net_edge_bps     = ready_half_spread_bps - ready_fee_bps - hedge_cost_bps
```

Report, per underlying, per fee scenario (`rt_2p00` is the operative one given the TREC path at ~1.6 bps round trip, but compute the full grid):

- `pct_time_hedged_viable` — fraction of session with `net_edge_bps > 0`.
- `hedged_ceiling_pkr` — qualifying volume × net edge ÷ 2.
- `hedge_absorption` — fraction of your intended clip size absorbable at the futures touch, and within 5 levels.

**Report the hedged ceiling without the pairing adjustment, alongside the unhedged ceiling with it.** `pair_ratio` exists because an unhedged MM must round-trip in the ready leg to stay flat. A hedged MM can be persistently one-sided in ready and flat in delta. That difference *is* the economic case for BAMM, and it must be visible as a number.

**3c. Price discovery — who leads**

This decides whether the strategy is viable at all, so it cannot be assumed. Plan 3 asserts "usually, the Future leads" as a parenthetical; that parenthetical is the whole ballgame.

Ready and futures are cointegrated by construction (the basis is stationary around carry), which is exactly the setting Hasbrouck information share and Gonzalo–Granger component share were built for.

```
1. Sample both mids onto a common grid (100 ms) using KNOWLEDGE time
   (running max of capture_ts per leg), not exchange time.
2. Fit a VECM on log mids with the basis as the error-correction term.
3. Report Hasbrouck IS upper and lower bounds, plus Gonzalo-Granger CS.
4. Report per underlying, per month, and check stability across months.
```

Also report signed lagged cross-correlation of returns at ±10/50/100/250/500 ms as a cheap cross-check that should agree in sign with the VECM result.

**Interpretation, decided before you see the numbers:**

- **Futures lead strongly (IS > ~0.7).** The passive ready bid is adverse-selected by construction: when the basis blows out, ready is about to reprice, and you would be racing to the front of the queue to be picked off. The "arbitrage profit" is a pickoff loss wearing a costume. Your toxicity gate cannot save you here, because the toxicity and the trade signal are *the same event*. Damp the skew coefficient θ by the measured information share, and expect the ready-passive leg to be marginal.
- **Roughly balanced (0.3–0.7).** BAMM is sound. Proceed.
- **Ready leads.** BAMM is strong; the futures-passive leg carries the adverse selection instead, and the design should lean on ready-side passive fills.

### Gate 3 — the go/no-go

Record in `bamm/08_reports/kill_gate.md`:

- [ ] Number of underlyings with tradeable futures on ≥50% of sampled days: **____**
- [ ] Of those, number with `pct_time_hedged_viable > 20%` at `rt_2p00`: **____**
- [ ] Median `hedge_absorption` at intended clip size: **____**
- [ ] Hasbrouck IS (futures) per surviving underlying: **____**

**Stop conditions.** If fewer than ~3 underlyings survive, BAMM is a niche, not a business — revert to the strict split from handoff §6 (unhedged MM on the no-futures universe) and stop here. If `hedge_absorption` is below ~50% at your intended clip, the hedge does not exist at size and no amount of modelling fixes it.

---

## Phase 4 — Carry curve and corporate actions

**Purpose.** Define fair basis. Everything downstream is a deviation from this number, so an error here is indistinguishable from alpha.

### Work

**4a. Dividends (the trap all three plans half-catch)**

`bamm/01_reference/corporate_actions.parquet`: `symbol, announcement_date, book_closure_start, book_closure_end, ex_date, payment_date, dividend_per_share, action_type, source, retrieved_at`.

Per your handoff doctrine: **announcements are primary; the price-based detector is a validator.** Use `corp_actions_master.py` — the four verdicts (`CONFIRMED`, `FACTOR_MISMATCH`, `SILENT_ADJUSTMENT`, `UNANNOUNCED_MOVE`) already implement this. `SILENT_ADJUSTMENT` is the dangerous case and must never be dropped.

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

Invert the same relationship to back out what the market is actually charging:

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

Fit per `(underlying, contract)`: take a robust central estimate (rolling median of intraday observations, then a per-day series), and use **that** curve as the anchor for `F_theo`.

Ingest KIBOR into `bamm/01_reference/kibor_daily.parquet` (`date, tenor, bid_rate, offer_rate, mid_rate, source, retrieved_at`; decimals only, missing values raise rather than defaulting to zero) and use it two ways: as a prior when a contract has too few observations to fit, and as a **bound** — an implied rate outside, say, `[0, KIBOR + 15%]` is a data error or a genuine dislocation and should alarm, not silently propagate.

**Tradeoff, stated plainly.** A fitted curve can absorb genuine mispricing into "fair", making you blind to a real dislocation. Mitigate by fitting on a long window (contract-to-date or 20 days) and trading deviations from it — never re-fit intraday, or you will define away the signal you are trying to trade.

### Gate 4

- [ ] Three unit tests pass: no dividend before expiry → `PV_D = 0`; one dividend → counted exactly once; dividend after expiry → excluded.
- [ ] `theoretical_future` matches a hand-computed spreadsheet on 10 timestamps.
- [ ] Implied carry fitted per contract; plotted against KIBOR; the spread between them is stable and explicable.
- [ ] Implied rate excluded inside 5 days of expiry.
- [ ] Every feature row can resolve a financing rate; missing rates raise.

---

## Phase 5 — Deterministic dual-instrument event stream

**Purpose.** One merged, reproducible, correctly ordered stream per `(date, underlying)`.

### Work

Output `bamm/02_event_stream/{date}_{underlying}_dual.parquet` with:

```
event_no, ts_exch, ts_cap, instrument, symbol, source, msg_seq, appl_seq,
entry_type, action, side, price, qty, level, order_id, trade_id, initiator
```

**Ordering is the load-bearing detail.** Sorting on `ts_exch` alone is wrong in three separate ways:

1. Your existing `load_events` sorts on `(ts_exch, kind_rank, appl_seq)`. `kind_rank` places snapshots (0) before same-timestamp incrementals (1), because a snapshot stamped *T* describes the book *as of T* and same-time incrementals must build on top of it rather than be wiped by it. A plain `ts_exch` sort destroys this.
2. Ready and futures ride **different channels with independent `appl_seq` spaces**. Cross-instrument tie-breaking on `appl_seq` is meaningless — you would be comparing two unrelated counters.
3. Snapshot rows sharing one `msg_seq` must stay contiguous, or `Book.snapshot` receives a partial message.

Use this key:

```
(ts_exch, kind_rank, instrument_rank, appl_seq_or_msg_seq, row_in_message)
```

`instrument_rank` is a fixed deterministic tie-break for same-millisecond events across the two legs. It is **arbitrary but must be stable and declared**, because at millisecond resolution you cannot prove which reached the gateway first.

`SIMPLIFICATION:` a fixed instrument rank. Production alternative — treat same-ms cross-instrument ordering as genuinely unknown and run the day both ways as a robustness check; if PnL is sensitive to that ordering, the strategy is trading on a fiction.

Verify physical sort order on disk rather than trusting the writer:

```python
df = pl.read_parquet(out_path)
print(df.group_by(["instrument", "source"]).len())
print(df.select(pl.col("ts_exch").is_sorted()))
print(df.select((pl.col("ts_exch").diff() < 0).sum().alias("backward_steps")))
print(df.group_by("instrument").agg(
    pl.len().alias("events"),
    pl.col("ts_exch").min().alias("first"),
    pl.col("ts_exch").max().alias("last"),
))
```

### Gate 5

- [ ] Both instruments present; all three sources present per instrument.
- [ ] `ts_exch` non-decreasing; zero backward steps.
- [ ] No duplicate full ordering keys.
- [ ] Ready and futures sessions overlap; the overlap window is recorded.
- [ ] Snapshot rows contiguous within `msg_seq`.
- [ ] Zero null `ts_exch` or `ts_cap`.
- [ ] Re-running the builder produces a byte-identical file.

---

## Phase 6 — Dual book reconstruction

**Purpose.** Two independently correct books from one stream.

### Work

Two instances of your existing, validated `Book` — imported, not reimplemented. Everything it already does must survive: full snapshot replacement, incremental add/cancel, trade consumption, order-ID tracking, the `__H_` hidden-residual, `__NEG_` traded-placeholder and `__AGG_` deep-residual synthetics, phase and circuit-band state.

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
    # that governs what the strategy is allowed to see -- see Phase 7.
    know_ready: int = 0
    know_future: int = 0
```

**Validate each book independently against the same reconciliation standard the single-instrument engine already meets.** Your documented benchmark is that 97.9% of trades print inside the reconstructed pre-trade touch. Reproduce that per leg. If the futures leg does not reach a comparable figure, the futures feed differs structurally (fewer levels, no order IDs, no AGG rows) and Phase 1's answers were wrong.

### Gate 6

- [ ] Per leg: reconstructed BBO matches the next snapshot's touch within tolerance.
- [ ] Per leg: trade-inside-touch rate reported and comparable to the 97.9% single-instrument benchmark.
- [ ] No crossed or negative books outside valid transition states.
- [ ] `b2_ignored` (unresolvable cancels) reported per leg as a fraction of cancels; a large figure on futures means the ID space differs.
- [ ] Phase/halt state tracked on both legs.
- [ ] Timestamps never move backward.

---

## Phase 7 — Synchronized cross-asset state, on knowledge time

**Purpose.** This phase is where all three source plans go wrong, and it is the most likely single cause of a backtest that will not survive live.

### The problem

The two legs are asynchronous. When a futures event arrives, the ready book holds whatever state you last saw. If you compute the basis on **exchange time**, you are differencing a price you knew against one you did not yet know — and the bias is not random. The stale leg lags the moving leg precisely when the move happens, so the error appears as a **dislocation that was never tradeable**, systematically in the direction that looks profitable.

### Work

Extend the two-clock principle to two instruments. Maintain **one joint knowledge clock**, because a real trading system has one view of time:

```python
# Knowledge time is the running max of capture_ts across BOTH legs: a real
# system knows everything that has arrived, regardless of which feed it came on.
know = max(know, int(event.ts_cap))

# Per-leg knowledge age measures how stale each book is in OUR view, not the
# exchange's. This is the number that gates cross-asset features.
ready_know_age_ms = know - know_ready
future_know_age_ms = know - know_future
```

Emit one row per event to `bamm/03_books/{date}_{underlying}_sync.parquet`:

```
ts_exch, know, trigger_instrument,
ready_last_exch, future_last_exch,
ready_know_age_ms, future_know_age_ms,
ready_bid, ready_bid_qty, ready_ask, ready_ask_qty, ready_mid, ready_microprice,
future_bid, future_bid_qty, future_ask, future_ask_qty, future_mid, future_microprice,
ready_phase, future_phase, ready_pinned, future_pinned,
cross_state_valid
```

```python
# Tunable, not sacred: 500 ms is a starting point given ~80 ms median feed
# latency with multi-second spikes. Calibrate from the observed joint
# distribution of per-leg knowledge ages, and re-check per symbol.
MAX_CROSS_BOOK_AGE_MS = 500

cross_state_valid = (
    ready_know_age_ms <= MAX_CROSS_BOOK_AGE_MS
    and future_know_age_ms <= MAX_CROSS_BOOK_AGE_MS
    and ready_two_sided
    and future_two_sided
    and not ready_pinned
    and not future_pinned
)
```

**Retain stale rows with their ages** rather than dropping them. You need the sensitivity curve: PnL as a function of `MAX_CROSS_BOOK_AGE_MS`. If the strategy's edge depends on trading stale-leg states, it does not exist.

### Gate 7

- [ ] Sync table produced for one day, one pair.
- [ ] Distribution of both knowledge ages plotted; median and p99 recorded.
- [ ] Fraction of events with `cross_state_valid` recorded.
- [ ] Basis computed on knowledge time vs exchange time, plotted together — **the gap between them is your lookahead, quantified**.
- [ ] 100 random event windows manually inspected.

---

## Phase 8 — Basis computation

**Purpose.** Turn two books plus a carry curve into the tradeable numbers.

### Work

**Denominator convention: every basis measure is expressed in bps of `ready_mid`.** Plan 3 mixes `ready_mid`, `ready_ask` and `future_ask` across its definitions, which makes the measures non-comparable and the thresholds meaningless.

```python
# Mid-to-mid, diagnostic only. Contains the carry, so it is NOT a signal.
basis_raw_bps = (future_mid - ready_mid) / ready_mid * 1e4

# Deviation from fitted carry. THE primary signal. Never feed raw prices to a
# model -- only this residual.
basis_resid_bps = (future_mid - theoretical_future(ready_mid, r_fit, dte, pv_div)) / ready_mid * 1e4
```

Mid-based residual is a *signal*. It is not what you can trade. Store both executable directions, computed against theoretical value at the **executable spot**, not at the mid:

```python
# Direction A -- future rich: SELL future at its bid, BUY ready at its ask.
exec_edge_sell_fut_bps = (
    future_bid - theoretical_future(ready_ask, r_fit, dte, pv_div)
) / ready_mid * 1e4

# Direction B -- future cheap: BUY future at its ask, SELL ready at its bid.
exec_edge_buy_fut_bps = (
    theoretical_future(ready_bid, r_fit, dte, pv_div) - future_ask
) / ready_mid * 1e4

# Net of BOTH legs' fees. This is the only number that decides anything.
net_exec_edge_a_bps = exec_edge_sell_fut_bps - ready_fee_bps - future_fee_bps
net_exec_edge_b_bps = exec_edge_buy_fut_bps - ready_fee_bps - future_fee_bps
```

Also store the two-sided crossing cost `future_ask - ready_bid` and `future_bid - ready_ask` for diagnostics, and z-scores of `basis_resid_bps` over 1 m / 5 m / 30 m windows.

**On z-scores, a warning.** A z-score is a *relative* measure. A quiet name with tiny basis volatility prints |z| > 2 constantly on moves worth a fraction of a basis point, and near expiry basis volatility collapses so z explodes on noise. Plan 1's `pct_time_basis_wins = % of time |basis_z| > 2.0` is therefore not an opportunity screen. **All thresholds, gates and screens are in bps net of fees; z-scores are context features only.**

### Gate 8

- [ ] Ten timestamps hand-validated end to end: spot, DTE, rate, PV dividends, `F_theo`, residual, both executable edges, both net of fees.
- [ ] `basis_resid_bps` distribution plotted; centred near zero by construction of the fit; tails inspected.
- [ ] Behaviour inspected specifically around ex-dividend dates, expiry week, and roll.
- [ ] `basis_raw_bps` and `basis_resid_bps` plotted together — the difference is the carry you removed, and it should decay to zero at expiry.

---

## Phase 9 — Feature store

**Purpose.** One row per valid event, point-in-time on knowledge time, no lookahead.

Adopted from Plan 3 with corrections. Written to date-partitioned parquet exactly like your canonical store; queried in place via a DuckDB view.

### Schema

**A. Context**

`ts_exch`, `know`, `trigger_instrument`, `ready_know_age_ms`, `future_know_age_ms`, `dte_calendar`, `dte_trading`, `r_fit`, `r_kibor`, `pv_div`, `cross_state_valid`, `sess_elapsed_frac`, `contract_month`, `days_to_roll`

**B. Basis (Phase 8)**

`basis_raw_bps`, `basis_resid_bps`, `exec_edge_sell_fut_bps`, `exec_edge_buy_fut_bps`, `net_exec_edge_a_bps`, `net_exec_edge_b_bps`, `basis_z_1m`, `basis_z_5m`, `basis_z_30m`, `basis_velocity_100ms`, `basis_velocity_1s`, `basis_velocity_5s`

**C. Microstructure — computed for both legs, `ready_` and `future_` prefixed**

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

**D. Cross-asset**

`relative_obi_1`, `relative_obi_5`, `microprice_gap_bps`, `cross_trade_flow_1s`, `relative_trade_flow_1s`, `corr_lag_{-500,-250,-100,-50,-10,0,+10,+50,+100,+250,+500}ms`, `rolling_info_share_60m`

The rolling information share is the important one and is missing from all three source plans: **the lead-lag relationship is a parameter of the strategy, and it drifts.** A single 9-month estimate is not enough; you need it as a time-varying feature so the skew coefficient can respond.

### Discipline

- Every feature declares: unit, lookback window, and **which clock it uses**. Record this in `bamm/config/feature_spec.yaml`. A feature whose clock is undeclared is a leak waiting to be found.
- Features are computed in a **forward-only single pass** over the event stream, in the same order the simulator will see them. No pandas `.shift(-n)`, no `rolling(center=True)`, no group-wise operations that see the whole day.
- Every rolling window is **backward-looking on knowledge time**, including the ones on the other leg.

**The one that is secretly forward-looking if you are careless:** any feature derived from a snapshot must use the snapshot's *capture* time, not its `orig_time`. Snapshots carry second-precision `orig_time` and arrive later; keying features on `orig_time` back-dates information you did not have.

### Gate 9

- [ ] `feature_spec.yaml` complete: unit, lookback, clock, for every column.
- [ ] Null and infinite rates reported per column.
- [ ] Distributions plotted; behaviour checked at open, close, halts, expiry, roll, ex-dividend.
- [ ] **Lookahead audit:** recompute features for the first *N* events using only the first *N* rows of the stream; results must be identical to the full-day run. This catches whole-day operations mechanically rather than by inspection.
- [ ] Feature-vs-basis correlation matrix inspected; pairs with |ρ| > 0.9 flagged for later pruning.

---

## Phase 10 — Labels, generated separately

**Purpose.** Never construct forward-looking labels in the same function that computes real-time features. Physical separation is the cheapest leakage defence you have.

### Label A — ready passive-fill markout (toxicity)

Plan 3 defines this as `Ready_Mid(t+10s) − Ready_Best_Bid(t)`, which is not a markout of a fill: it ignores whether a fill would have occurred, ignores queue position, and is unsigned by side.

Correct specification. Sample only where a passive quote could genuinely rest, and record the counterfactual explicitly:

```
decision_know_ts, side, hypothetical_price, queue_ahead_qty,
fill_exch_ts, fill_price, mid_at_fill,
mid_at_fill_plus_{30s,60s,300s},
markout_bps_{30s,60s,300s}
```

Sign convention: for a passive **buy** fill, `markout = (mid_h − fill_price) / fill_price`. Negative means the market moved against you after you bought — adverse selection. Same sign convention both sides so the model learns one thing.

### Label B — legging slippage

The scenario: a passive fill on one leg at *t*, then an aggressive hedge into the other leg landing at *t + latency*.

```
fill_leg, fill_exch_ts, fill_price, hedge_side, hedge_qty,
seen_touch_at_decision, hedge_arrival_ts,
hedge_vwap, hedge_slippage_bps, hedge_unfilled_qty, hedge_levels_walked
```

**Draw the latency from `LatencyModel`, never a hard-coded 10 ms.** Your own priors are ~45 ms one-way with 2% tail draws averaging 400 ms. Training on 10 ms teaches the model an execution you cannot achieve, and it will *understate* slippage — telling you the arb is safe exactly when it is not, which is the precise failure the label exists to prevent. Generate at several latency draws so slippage sensitivity is measurable rather than assumed.

Walk the book with your existing `Book.liquidation_value`, which already handles level-walking, fees, `__H_` inclusion, `__AGG_` exclusion, and unfilled residual.

**Both hedge directions must be labelled.** Plan 2 covers only the futures-fill → ready-hedge direction. Given OGDC's ready book (1,686M notional, 4.1 bps spread) that is the *cheap* direction. The expensive and dangerous direction is ready fill → hedge into thin futures. Your hedge policy must know which leg filled, so label both.

### Censoring — mark invalid, do not silently drop

- Horizon crosses session end.
- Either leg halted, in auction, or pinned at a circuit band.
- `cross_state_valid` false at decision time.
- Book cannot absorb the hedge quantity.
- Reference data missing.
- Contract rolls within the horizon.

Censoring is itself informative: a systematic pattern in *which* samples get censored is a finding about when the strategy cannot operate.

### Gate 10

- [ ] 20 label examples replayed row by row by hand against the raw stream.
- [ ] Sign conventions verified on a known adverse case and a known favourable case.
- [ ] Censoring rates reported by reason.
- [ ] Label distributions inspected for the impossible: markouts far beyond the day's range, zero-slippage hedges through thin books.
- [ ] Labels reproduce when regenerated from a different random seed on the deterministic parts.

---

## Phase 11 — Universe screening and persistence

**Purpose.** Choose what to trade. Transplanted from Plan 1, which is the only source plan that has this — and it encodes a lesson you paid for.

Plan 2 and Plan 3 go straight from one symbol on one day to a mechanical universe expansion. That discards two documented findings: KTML swung 236× in ceiling between two dates, and **symbol ranking inverts with fee level** (KTML rank 1 at 35 bps and rank 6 at 2 bps; FFC rank 5 and rank 1 respectively).

### Work

Run the full 9 months, with the fee grid, splitting the universe by `has_futures`:

**Per symbol-day:** existing `ticker_stats_core` outputs, plus `hedged_ceiling_pkr_rt_*` (Phase 3 definition, no pairing adjustment), `pct_time_hedged_viable`, `futures_depth_absorption`, `info_share_futures`.

**Per symbol, across days:**

```
pct_days_top20          -- fraction of days in the daily top 20 by hedged ceiling
ceiling_median, ceiling_iqr, iqr_over_median   -- episodic names have huge ratios
rank_autocorr_t_t5      -- rank stability at 5-day lag
n_days_tradeable        -- days with a two-sided futures book and viable edge
```

Expect three populations: stable-attractive (the watchlist), episodic (event-driven; needs a trigger, not a standing quote), and dead.

**Rank on `pct_days_top20`, never on a single day.** Produce a separate watchlist per fee scenario; `rt_2p00` is operative under the TREC path, but keep the grid so the watchlist can be regenerated when the actual fee schedule is confirmed.

### Gate 11

- [ ] 9-month screen complete for both universes.
- [ ] Persistence metrics computed; the three populations visible in the distribution.
- [ ] Watchlist frozen per fee scenario, with the selection rule written down *before* looking at backtest PnL.
- [ ] The top-ranked name at `rt_2p00` is checked for the KTML pathology: is it top-ranked *persistently*, or on one anomalous day?

---

## Phase 12 — BAMM simulator

**Purpose.** A new engine in `bamm/06_simulator/bamm_backtest.py` that reuses the validated components rather than replacing them.

### State

```
book_ready, book_future
work_ready: {side: MyOrder}, work_future: {side: MyOrder}
pos_ready, pos_future, net_delta
cash, margin_posted, financing_accrual
pending          -- one heap for ALL in-flight messages, both legs
know             -- ONE joint knowledge clock
eod_ready, eod_future
```

### Event types on the heap

`MARKET_EVENT`, `ORDER_ARRIVAL`, `CANCEL_ARRIVAL`, `CANCEL_ACK`, `HEDGE_ARRIVAL`, plus a monotone sequence tiebreaker so the heap never compares payload objects. This mirrors what `_push`/`_activate_until` already do.

### Net exposure

```python
# beta comes from contract_multiplier and lot sizes in the contract master.
# Do NOT assume 1.0, and do not add raw share counts across instruments.
net_delta = pos_ready + beta * pos_future
```

Three limits, all enforced: per-leg position, net delta, and gross notional (which drives margin).

### Own-fill callback

Your handoff lists this as unbuilt, and BAMM cannot work without it: the hedge decision is triggered *by* a fill, not by the next market event. Build it here.

`SIMPLIFICATION:` the existing engine reads `self.pos` in real time, so the strategy "knows" a fill instantly rather than one ack-latency later. Acceptable at a 5.6 s median trade gap for single-instrument MM. **Not acceptable for BAMM**, where the hedge is triggered by the fill and the ack delay is the legging window. Model the fill ack explicitly.

### Hedge policy as a swappable object

This is where the PnL actually lives, so it must be an interface with multiple implementations, not a hardcoded branch:

```python
class HedgePolicy:
    def on_fill(self, fill, dual_state, model_b) -> HedgeDecision:
        """Return: hedge aggressively now, post passively, or wait."""
```

Implement at minimum: `AlwaysAggressive` (upper bound on cost), `AlwaysPassive` (lower bound, maximum legging risk), and `EdgeAware` — which compares `remaining_edge − predicted_hedge_cost` against `sigma_basis * sqrt(expected_wait)`, with a hard time and delta backstop that escalates to aggressive.

**Report PnL under all three.** If `AlwaysAggressive` is materially worse, the passive-completion logic is where your edge lives and deserves the engineering. If it is not worse, you have just saved yourself a great deal of complexity.

Note the asymmetry: when your ready quote fills, your futures quote may still be live, and vice versa. The real decision is whether to wait for the resting other leg or cross now. "Instantly fire a hedge" hardcodes the expensive branch and undoes the premise of the whole design.

### Gate 12 — deterministic unit scenarios

- [ ] Flat basis, no fills, no orders leak.
- [ ] Future rich → futures ask fills → hedge scheduled → ready book unchanged on arrival.
- [ ] Same, but the ready touch has vanished on arrival.
- [ ] Same, but the hedge walks several levels.
- [ ] Same, but the hedge cannot fully fill (`hedge_unfilled_qty > 0`).
- [ ] Future cheap — full symmetric case.
- [ ] Cancel and replacement arrive out of order.
- [ ] Fill occurs before the cancel lands (in-flight cancel risk).
- [ ] Cancel ack arrives after the order already filled.
- [ ] **Cancels target a specific `oid`, not a side** — this race was already found and fixed once in the single-instrument engine; do not reintroduce it.
- [ ] Halt on one leg while the other trades.
- [ ] Roll date inside the simulated day.

---

## Phase 13 — Costs: fees, financing, margin

**Purpose.** At a ~3 bps gross edge, cost modelling is not an accounting detail; it is the result.

### Work

**Per-leg fee schedules, separately parameterised.** Do not assume the futures segment carries the same stack as equities. Ready retail is ~17.73 bps/side, dominated by commission; the TREC own-account path is ~0.78 bps/side (~1.6 bps round trip) because commission and its SST vanish while the exchange/regulatory stack survives. Verify the futures-side equivalents independently.

**Financing.** Long ready funded at your actual cost; short futures earns/pays the embedded rate. This is the same `r_fit` from Phase 4 — the strategy's carry P&L and its fair-value model must use one consistent number, or the backtest books a profit that the pricing model says does not exist.

**Margin — absent from all three source plans, and it binds.** Short futures posts exchange margin (VaR-based on PSX). Model it as a hard capital constraint:

```python
# Gross futures notional is capped by posted margin, which is capped by the
# capital allocated to this strategy. On a $3-6M base this binds long before
# any position limit does.
max_gross_future_notional = capital * allocation_frac / margin_pct
```

Intraday margin calls on an adverse basis move can force liquidation at the worst moment — model the constraint, and record every event where it binds.

### Never report a single PnL number

```
gross_spread_pnl_ready, gross_spread_pnl_future,
basis_convergence_pnl, inventory_pnl, legging_slippage,
ready_fees, future_fees, financing_cost, dividend_cashflows,
margin_cost, eod_liquidation_cost, net_pnl
```

This decomposition is how you detect the KTML pathology, where "+724 PKR" decomposed into +7,646 directional (an accidental short through a −707 bps session) minus −6,922 of actual market-making losses. Without the decomposition, that run reads as a success.

### Gate 13

- [ ] A zero-price-movement round trip loses exactly its applicable fees plus slippage — to the paisa.
- [ ] Fee schedules independently verified per leg and per aggressive/passive side.
- [ ] Financing accrual reconciles against `r_fit` × average notional × days.
- [ ] Margin constraint binds in at least one test scenario and is handled without a crash.
- [ ] Decomposition sums exactly to `net_pnl`.

---

## Phase 14 — Strategy: `BasisAwareMM`

**Purpose.** Extend `MicrostructureMM` rather than replacing it. The microstructure theory remains the skeleton; the basis enters through the reservation price.

### Continuous skew, not a regime switch

All three plans use a binary threshold on `|basis_z| > 2`. A hard flip causes oscillation at the boundary, cancel/replace thrash, and hysteresis patches. Collapse it into the reservation price and the "regimes" emerge for free:

```python
# D is the dislocation versus FITTED CARRY, not versus zero. Skewing toward
# zero basis when fair basis is +80 bps means systematically buying futures rich.
D = (future_mid - theoretical_future(ready_mid, r_fit, dte, pv_div))

# theta is damped by the measured information share. If futures lead price
# discovery, a large part of D is news rather than tradeable dislocation, and
# leaning into it on the ready side is racing to be picked off.
theta_eff = theta * (1.0 - info_share_futures)

# Joint inventory drives BOTH legs. The two books are coupled through net
# delta; their skews cannot be computed independently.
r_ready = fair_ready + theta_eff * D - gamma * sigma_resid**2 * tau * net_delta
r_future = fair_future - theta_eff * D - gamma * sigma_resid**2 * tau * net_delta
```

`sigma_resid` — not price sigma. The entire economic case for BAMM is that hedged inventory carries *basis* variance, not price variance. Using price sigma sizes you as though unhedged and throws away the benefit you built the system for.

**`D ≈ 0` yields symmetric two-sided MM on both legs. Large `D` yields the violent skew. No switch, no boundary, no hysteresis needed.**

### Never cancel a side

Plans 1, 2 and 3 all "pull ready asks and futures bids" on a dislocation. Do not. Skew hard instead — a quote 40 bps off the touch essentially never fills, but when it does you are delighted, and you retain the ability to profit when the basis reverts through you. Cancelling makes you a one-way accumulator that fills to its limit and then cannot respond for the rest of the session.

### Viability gate, per leg

Your existing gate declines to quote when the spread cannot cover fees. Extend it: the ready-leg gate must include the expected hedge cost, since an unhedgeable fill is not a market-making fill. **Declining to quote remains a first-class output.**

### Reuse, do not reimplement

`round_tick` (bids floor, asks ceil, so rounding never makes a quote more aggressive), post-only clipping one tick inside the opposite touch, circuit-band clamping, and the no-churn comparison in `_requote`. Plan 1's skew snippet reimplements all of this and gets it wrong — it caps a *ready* price against a *futures* price (different books trading at a structural premium), and uses `min` where a floor on an ask requires `max`, so its "cap" makes the quote more aggressive rather than less.

### Gate 14

- [ ] `theta = 0` reproduces `MicrostructureMM` behaviour on the ready leg exactly.
- [ ] Quote prices are always on the tick grid, inside circuit bands, and never marketable against the visible book.
- [ ] No side is ever cancelled by the skew logic.
- [ ] Requote rate per hour is sane — no churn at any `D`.
- [ ] `net_delta` respects its limit under adversarial fill sequences.

---

## Phase 15 — Models

**Purpose.** Two narrow predictors replacing heuristic components, per your standing ML doctrine. Not an end-to-end learner.

**Model A — passive markout / toxicity.** Predicts Label A. Outputs expected markout in bps and P(markout worse than limit).

**Model B — legging slippage.** Predicts Label B. Outputs expected slippage, **p90 slippage**, and P(incomplete hedge). For execution safety the tail matters more than the mean; a hedge policy sized on the mean is under-reserved by construction.

### Method

- Baselines first, in order: median label by state, then Ridge/logistic (coefficients readable as weights), then LightGBM only if it beats linear out of sample.
- Pool across symbols with per-symbol z-scored features; per-symbol data is too thin.
- **Walk-forward, expanding window, monthly roll.** Never shuffle event rows. Separate contracts and roll periods across folds.
- **Embargo ≥ 1 day**, and longer than your longest label horizon plus the censoring window. A 1-day embargo with 300 s labels is adequate on horizon grounds but does nothing about *overlapping* samples within a fold — use sample weighting or subsampling for overlap.
- Two mandatory leakage tests: shuffle labels → IC must collapse to zero; remove the most recent 30 s of features → performance must not *improve*.
- Feature selection by permutation importance on validation folds only; drop one of any pair with |ρ| > 0.9.

### Gate 15

- [ ] Both leakage tests pass.
- [ ] OOS IC in the 0.02–0.05 range at 60 s. **0.10+ means leakage until proven otherwise.**
- [ ] A model is accepted **only** if it improves simulated out-of-sample net PnL after costs — never on RMSE, AUC or IC alone.
- [ ] Model integration as gates, not as quote generators:

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
6. Add full per-leg fees.
7. Add financing and margin.
8. Add Model A only.
9. Add Model B only.
10. Add both.

### Robustness and negative controls

- `fill_on_crossing_adds` on vs off — it is the most assumption-heavy fill rule you have.
- `at_price_mode` in `never` / `queue` / `always` — brackets the queue assumption.
- Latency ×2 and ×5.
- Fees ×2.
- Displayed futures depth halved.
- `MAX_CROSS_BOOK_AGE_MS` swept from 100 ms to 5 s.
- **Negative control:** disable the dividend adjustment deliberately. PnL must *degrade*. If it does not, your dividend logic is dead code and was never wired in.
- **Negative control:** set `theta = 0`. Basis-attributed PnL must fall to zero. If it does not, PnL is being mis-attributed.
- **Negative control:** shift one leg's timestamps forward by 1 s. Apparent edge should collapse. If it survives, you are trading a stale-state artefact.

### Gate 16

- [ ] Every step's PnL delta recorded with its decomposition.
- [ ] All three negative controls behave as specified.
- [ ] The step where PnL turns negative is identified by name.
- [ ] Fill provenance audited by `reason` — if most PnL comes from `crossing_add` fills, the edge is fragile and rests on a counterfactual.

---

## Phase 17 — Scale out

Expand only after Phase 16 is clean on one pair:

1 pair × 1 day → 1 pair × 20 days → 1 pair × full contract life (including roll) → 5 pairs × 20 days → full surviving universe × 9 months.

Deliberately include: dividend days, expiry week, roll days, high-volatility days, low-volume days, halted sessions, one-sided books, and days with missing reference data.

**Do not tune thresholds on 2026-06-30 and then report 2026-06-30 as evidence.** Parameter selection happens on training folds only; the held-out months are looked at once.

`SIMPLIFICATION:` per-symbol parameter fitting. Plan 1 proposes sweeping gamma (21 values) × kappa (21) × threshold × size × `max_inv` × `quiet_ms` **per symbol** on nine months of thin data, selected on "the validation set". Those optima will be noise. Production approach: fit pooled priors across symbols, allow per-symbol deviation only where a walk-forward test shows the deviation is stable across folds, and prefer three well-understood parameters to twelve fitted ones.

### Gate 17

- [ ] Results stable across contracts and months, not driven by one window.
- [ ] Roll periods produce no discontinuity in PnL attribution.
- [ ] Per-symbol results consistent with the Phase 11 ranking — if the screen and the backtest disagree about which names are good, one of them is wrong and you must find out which.

---

## Phase 18 — Performance

**Correctness first. Profile before optimising.**

A note on the widely-repeated claim that this needs Rust: per symbol per day you have roughly 3.5k snapshot messages, ~4.4k updates and ~1.1k trades ≈ 9k events; two legs ≈ 18k. A month of one pair is ~380k events — seconds to minutes in Python, not days. That estimate confuses the **parser** workload (40M snapshot rows/day across 577 symbols, genuinely heavy) with the **backtest** workload (one pair at a time). The GIL is irrelevant to a single-threaded event loop.

When scale does bite — 179 days × N pairs × parameter sweeps — the first answer is multiprocessing across days, the same 3-worker pattern that took your parser to ~17 h.

Only if profiling proves otherwise:

- Polars/DuckDB for file filtering, stream normalisation, feature and label preparation, reporting.
- Numba or Rust for the dual-book replay loop, queue simulation, pending-message heap, and book-walk hedge execution.

Polars lazy expressions do not replace an inherently stateful order-book reconstruction; use Polars to *prepare* the stream and a compiled loop for the state machine.

### Gate 18

- [ ] Profile captured before any optimisation, with the actual hot phase named.
- [ ] Any optimised path reproduces the pre-optimisation PnL to the paisa.
- [ ] The MCB regression still passes.

---

## Phase 19 — Regulatory and paper-trading ladder

**Start the regulatory workstream at Phase 1, in parallel.** It is measured in months and gates everything. Plan 1 treats "replace simulated latency with real feeds" as a development task; it is not.

**TREC path (verified):** proprietary trading permitted; own-account pays no commission (hence no SST), leaving ~0.78 bps/side ≈ 1.6 bps round trip — the `rt_2p00` scenario. Requires a separate brokerage entity (BBI cannot hold the TREC), Rs 2.5M certificate + Rs 100k processing, Rs 35M net worth, Rs 15M liquid capital, plus ongoing compliance opex.

Live infrastructure not covered by the backtester: FIX order entry and session management, kill switch, state recovery on reconnect, position and cash reconciliation against the broker, real-time risk limits, alerting, and end-of-day recon.

### Ladder

**Stage 1 — pure MM, no basis logic.** 5–10 watchlist names.
**Stage 2 — basis-aware quoting, no aggressive hedging.** Passive completion only.
**Stage 3 — full BAMM with hedge execution.**

**Validate each stage on fill rate, queue position, markout, and hedge slippage — not on PnL.** At a ~3 bps edge, PnL is far too noisy to confirm "within 20% of backtest" at these sample sizes; Plan 1's criterion would pass or fail essentially at random. The microstructure metrics converge orders of magnitude faster and tell you specifically *which* model assumption is wrong.

Use live telemetry to refit `LatencyModel` (currently priors, not measurements) and the hedge policy. This is why `HedgePolicy` is a swappable object.

### Gate 19

- [ ] Realised fill rate within tolerance of simulated, per side, per leg.
- [ ] Realised queue position distribution matches the simulator's.
- [ ] Realised markout distribution matches Model A's predictions.
- [ ] Realised hedge slippage within Model B's predicted p90.
- [ ] Kill switch tested under a live disconnect.

---

## The single assumption most likely to be fatal

**That the hedge exists at size.** Not the model, not the fees, not the ML. If the futures leg is thin, every fill you hedge gives back more than the ready spread earned, and BAMM is strictly worse than pure basis arb on the same names — and possibly worse than doing nothing.

Phase 3 exists to find that out for a few days of querying instead of three months of building. Run it first, and be willing to stop there.

## The second most likely

**That the basis dislocation is tradeable rather than informational.** If futures lead price discovery, then at the exact moment the model says "post an aggressive ready bid", the reason is that ready is about to reprice against you. Your toxicity gate cannot catch it, because the toxicity and the signal are the same event. Phase 3c measures this; Phase 14's `theta_eff` is the only defence, and it is a damping, not a cure.

## Immediate next session

1. Phase 0 — freeze, git tag, write and pass the MCB regression test.
2. Phase 1 — run the schema and symbol queries; fill in `data_contract.yaml` from actual results.
3. Phase 2 — build the contract master for the surviving underlyings, reconciled two ways.
4. Phase 3 — **run the kill gate.** Report the four numbers in Gate 3 before writing any further code.

Do not begin the feature store, the simulator, or the models until Gate 3 passes.
