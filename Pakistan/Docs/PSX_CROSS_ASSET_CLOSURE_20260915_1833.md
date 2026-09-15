# Cross-asset research: three remaining rungs closed, all negative

Date: 2026-09-15
Scripts: `leadlag_screen.py`, `leadlag_diagnose.py`, `probe_etf_hedge.py`, `probe_market_beta.py`
Saved outputs: `leadlag/leadlag_screen.parquet`, `leadlag/leadlag_diagnose.png`,
`etf_hedge_fit_20260915.csv`, `market_beta_returns_20260915.csv`, `market_beta_20260915.png`

Companion document: `PSX_HEDGE_COST_FINDING_20260915_0700.md` (futures vs share hedge cost).

---

## Why this document exists

Three items were closed on 2026-09-15 and the results existed only in a chat
session. They are recorded here because each one **removes a component from the
live engine**, and a future session that does not know they are closed will
propose building that component.

---

## 1. Sector-leader lead-lag — DEAD

**Design.** Top **2** names per sector by median **traded value** (not volume — a
leader on share count is a cheap stock, not an informed one). 157 ordered pairs.
Hayashi-Yoshida asynchronous covariance across a 39-point lag grid spanning
−30 s to +30 s.

**Result: 1 of 157 pairs clears the 1.554 bps round-trip fee** — and that is
before any of the other four acceptance conditions are applied. The median
anticipatable move across all 157 pairs is **about 0.22 bps**, roughly one
seventh of the fee. The signal is not small relative to the cost; it is
invisible relative to the cost.

Two supporting reads from `leadlag_diagnose.png`:

- **Correlation at the fitted peak lag is not higher than correlation at lag 0**
  for most pairs — the points sit on the 45° line. That is the Epps /
  non-synchronicity artifact, which is what a naive estimator produces when
  there is no lead at all.
- **Fitted peak lags spread from about −20 s to +20 s and centre on zero.** A
  real lead clusters at one sign and one horizon. A symmetric spread centred on
  zero is the shape of noise.

The pre-registered prior — "largest traded value leads its sector" — was tested
and **not supported**.

### Correction recorded against my own work

The first version of the diagnostic chart was captioned *"a microstructure lead
should be sub-second."* SZ objected that PSX is human-traded with no bots, so a
genuine lead would sit in **seconds**, not sub-seconds. That was right, and the
lag grid at the time had only three points between 3 s and 15 s — it could not
have resolved a lead where one would actually live. The grid was widened to 39
points and the screen re-run. **The conclusion did not change**, but it was only
decision-grade after the correction.

---

## 2. Broad / sector ETF hedge rung — DEAD

### The tick screen kills it before any fitting

PSX tick is a flat 0.01 PKR, so a one-tick spread is `100/P` bps. A round trip
inside the 2.67 bps edge, including the 1.554 bps spot fee, requires the
instrument to trade above **~89.6 PKR**. Any hedge instrument below that price
cannot clear the edge even with a perfect one-tick book. This is arithmetic, not
an estimate.

### The hedge quality, measured anyway

R² of each name's return on the ETF return, 465–477 names per horizon
(`etf_hedge_fit_20260915.csv`):

| horizon | median R² | 90th pct R² | best R² | median beta |
|---|---|---|---|---|
| 1 min | 0.0004 | 0.0031 | 0.0303 | 0.019 |
| 5 min | 0.0027 | 0.0176 | 0.0753 | 0.097 |
| 15 min | 0.0060 | 0.0436 | 0.1931 | 0.152 |

Seven listed ETFs appear in the fit. Only three reach any explanatory power at
all — UBLPETF (0.193), MIIETF (0.172), JSMFETF (0.137) — and only at **15
minutes**. At 1 minute the best of them is 0.024.

**A hedge that needs fifteen minutes to explain a sixth of the variance is not a
hedge for a book holding inventory for seconds to minutes.**

*Record note:* the ETF traded-activity table from `probe_etf_hedge.py` printed to
console and was not written to a CSV. The tick screen and the fits above are the
saved record.

---

## 3. Portfolio-level market-beta overlay — DEAD

**Design.** Remove the common market factor at the portfolio level with a
tolerance band, so the book is not firing hedge orders on every event.

**Result** (`market_beta_returns_20260915.csv`, 500 names × 4 horizons):

| horizon | median R² | 90th pct R² | max R² | median beta |
|---|---|---|---|---|
| 5 s | 0.0001 | 0.0005 | 0.0036 | 0.022 |
| 30 s | 0.0003 | 0.0038 | 0.0243 | 0.086 |
| 60 s | 0.0007 | 0.0111 | 0.0618 | 0.110 |
| 300 s | 0.0037 | 0.0619 | 0.2644 | 0.208 |

At the horizon a market maker actually holds inventory — **seconds** — the market
factor explains on the order of **0.01%** of a typical PSX name's return, at a
median beta of **0.02**.

There is no common factor to hedge out. The residual *is* the position. Beta
only becomes visible at five minutes, by which time the inventory is gone.

The tolerance-band design is moot as a consequence: a no-trade band exists to
stop over-trading a hedge, and there is no hedge here to over-trade.

---

## 4. What this settles for the live engine

Every cross-asset and portfolio-level component is now **closed negative**:

| rung | status | why |
|---|---|---|
| same-ticker futures hedge | inverted | shares cheaper on 99/100 names, median 3.30× |
| futures → spot lead-lag | dead | only 13 of 100 roots trade >1,000×/day |
| basis-aware futures MM | premise undermined | all four arms lose |
| sector-leader lead-lag | dead | 1 of 157 pairs clears the fee |
| ETF hedge | dead | tick screen, and R² 0.0004 at one minute |
| portfolio beta overlay | dead | R² 0.0001 at five seconds |

**Consequence:** the live engine needs **no cross-symbol state, no portfolio risk
aggregator, and no hedge execution path**. It is 113 independent quoters, each
seeing only its own book.

This is now a settled design constraint rather than an assumption, and it
removes a large amount of work from the live build. That is the useful outcome
of this block — not a new signal, but a smaller system.

---

## Caveats

- Every R² here is from a **linear, contemporaneous, unconditional** fit. A
  relationship that only appears in a regime (a market-wide selloff, an index
  rebalance) would not show in these numbers. None of the three rungs was
  rejected *because* of a marginal statistic, though — they were rejected by
  margins of one to two orders of magnitude, which no regime split recovers.
- The lead-lag screen used the sector map built from the exchange's own daily
  quotation sheet: 107 names, 19 sectors, 6 singletons. A pair not in that map
  was not tested.
- `leadlag_screen.parquet` holds the full per-pair record. Reading it needs
  `pyarrow` or `duckdb`, which the sandboxed shell used for this write-up does
  not have — the figures above were read from `leadlag_diagnose.png`, which is
  generated from that same parquet.
