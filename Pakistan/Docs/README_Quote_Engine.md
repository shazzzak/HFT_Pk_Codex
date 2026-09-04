# The Quote-Generation Engine — Every Variable, Every Statistic, Explained

This document is a complete reference for the microstructure statistics the
strategy computes before it decides where to quote — what each one is, what it
means in plain language, the academic literature behind it, and (importantly)
which ones turned out **not** to work in practice on PSX data.

There are **two files** involved, and it matters which is which:

- **`micro_mm.py`** — the **live quoting engine** (`MicrostructureMM`). It
  computes a small, fast set of statistics every event and uses them immediately
  to place quotes. This is what actually trades.
- **`build_feature_store.py`** — the **offline research feature extractor**. It
  computes a much larger set of statistics (OFI, VPIN, spread-Z, queue-depletion,
  multiple OBI depths, etc.) and writes them to disk, so signals can be *tested*
  before any are wired into the live engine. Nothing here trades; it is a
  laboratory.

A companion document, `README_Skew_and_Triggers.md`, covers how these statistics
are *turned into* quote placement (inventory skew, EOD ramp/cliff,
distance-to-lock). This document covers the **inputs** — the statistics
themselves.

---

## Part A — The live engine's inputs (`micro_mm.py`)

Every event (a trade, a book update, or a snapshot), the backtester calls
`observe(kind, obj, ts_exch, mid)`, which updates the engine's running state.
When a two-sided book is present, `quotes(bb, bq, ba, aq, pos)` reads that state
and produces quotes. Here is everything it maintains and computes.

### A.1 The raw book inputs

`bb, ba` — **best bid, best ask** (the "touch"). The highest price a buyer will
pay and the lowest a seller will accept, right now.
`bq, aq` — **bid size, ask size** — how many shares rest at the touch on each
side.
`pos` — **current inventory** (our position, signed: + long, − short).

*Layman:* the best prices and how much is available at them, plus how much stock
we're currently holding.

### A.2 Spread

```
spread = ba - bb
```

*What it is:* the gap between the best bid and ask.
*Layman:* the "toll" for trading immediately — buy at the ask, sell at the bid,
and you've paid the spread. A market maker's core business is to **earn** this
spread by posting patiently on both sides.
*In the engine:* the spread sets the floor of what we can capture, feeds the
**viability gate** (if the market spread is narrower than twice our required
half-spread, no profitable passive quote exists and we stand aside), and — via a
rolling EMA (`ema_spread`, decay `spread_alpha`) — provides the **reference
spread** that caps how far the EOD/lock triggers may widen a quote.
*Literature:* the spread as the market maker's compensation is the foundation of
the whole field — Demsetz (1968, "The Cost of Transacting"); its decomposition
into order-processing, inventory, and adverse-selection components is Glosten &
Harris (1988) and Huang & Stoll (1997).

### A.3 Order-book imbalance (OBI) — used here as the *microprice* lean

```
imb = bq / (bq + aq)                 # bid share of touch depth, in [0,1]
microprice = ba*imb + bb*(1-imb)     # depth-weighted fair value
```

*What it is:* the fraction of touch liquidity resting on the bid, and a fair
value pulled toward the heavier side.
*Layman:* if there's far more size on the bid than the ask, buyers are keener
than sellers, so the "true" price is probably a little above the mid. The
microprice leans fair value that way.
*Literature:* the microprice is Gatheral & Oomen and, most rigorously, **Stoikov
(2018, "The Micro-Price: A High-Frequency Estimator of Future Prices")**, which
shows it is a better short-horizon predictor of the future mid than the plain
mid. Order-book imbalance as a predictive signal: Cartea, Jaimungal & Penalva,
*Algorithmic and High-Frequency Trading* (2015), Ch. 3.

**⚠️ WHERE THEORY MET OUR DATA — the microprice hurt us.** This is the single
most important practical finding about the inputs. In theory the microprice is a
better fair-value estimate. In *our* backtests it was the **confirmed cause of a
systematic short drift**: because it leans toward the heavy side, it made the
strategy sell into bid-heavy (rising-pressure) books and buy into ask-heavy
(falling-pressure) books — i.e. it leaned us the wrong way for a *passive maker*,
who gets adversely selected exactly when the microprice's lean is "right" for a
*taker*. Production therefore runs with **`use_microprice = False`**, quoting
around the **plain arithmetic mid**:

```
fair = 0.5 * (bb + ba)
```

The lesson: a signal that predicts the next mid move well (good for a taker) can
be actively harmful as a fair-value center for a passive maker, because the maker
is on the losing side of that prediction at the moment of the fill. OBI still
enters the strategy — but as a *directional skew signal*, not baked into fair
value. The microprice is kept only as a toggle to test/disable it.

### A.4 Toxicity (order-flow imbalance over recent trades)

```
flow    = deque of signed trade volumes (+buy, -sell), last `flow_window` trades
signed  = sum(flow)                       # net direction
gross   = sum(|f| for f in flow)          # total volume
toxicity = |signed| / gross               # in [0,1]
```

*What it is:* how one-directional recent trading has been. 0 = perfectly
balanced two-way flow; 1 = every recent trade hit the same side (a pure sweep).
*Layman:* if the last N trades were all buys, someone is aggressively
accumulating — probably because they know something. That's "toxic" flow: if you
keep quoting normally, you'll be the one selling to them right before the price
jumps.
*In the engine:* toxicity **widens the adverse-selection half-spread** (charge
more when flow is one-sided) and **cuts our quote size** (informed traders prefer
size, so offer less of it when flow looks informed).
*Literature:* this is the empirical proxy for the **Glosten-Milgrom (1985)**
adverse-selection model — the market maker cannot tell informed from uninformed
traders and must widen against the *possibility* of information; order-flow
imbalance is the observable signature. The order-processing view of one-sided
flow is Kyle (1985, "Continuous Auctions and Insider Trading"). The
volume-synchronized refinement (VPIN) is Easley, López de Prado & O'Hara (2012,
"Flow Toxicity and Liquidity in a High-Frequency World").

### A.5 The "quiet" flag

```
quiet = (now - last_trade_ms) > quiet_ms      # e.g. > 2 seconds since last trade
```

*What it is:* has the tape gone silent?
*Layman:* a long gap with no trades is evidence *against* an active information
event — the opposite of toxic flow. So when it's quiet, we can safely **tighten**
(halve the adverse-selection charge) rather than widen.
*In the engine:* `if quiet: toxicity *= 0.5` before building the half-spread.
*Literature:* the informational content of *time between trades* is Diamond &
Verrecchia (1987) and, canonically, **Easley & O'Hara (1992, "Time and the
Process of Security Price Adjustment")** — "no trade" is itself a signal.

### A.6 Volatility (EMA of squared mid returns)

```
ret     = (mid - last_mid) / last_mid         # only on an actual mid MOVE
ema_var = vol_alpha * ret^2 + (1-vol_alpha) * ema_var   # RiskMetrics-style
sigma   = sqrt(ema_var)                        # per-event fractional-return vol
```

*What it is:* how much the price has been jumping around lately, updated cheaply
(O(1) per event) and only when the mid actually changes.
*Layman:* a rough, fast-adapting measure of "how choppy is it right now." When
volatility is high, holding inventory is riskier, so we charge a wider spread and
lean harder against inventory.
*In the engine:* `sigma` drives both the **risk half-spread** (`half_risk =
0.5·γ·σ²·τ·fair`) and, after conversion to price units, the **inventory skew**.
*Critical practical note:* `sigma` is a **fractional-return** volatility (~1e-4),
so it must be multiplied by price (`sigma_p = sigma·fair`) *before* being squared
in any variance term — an early bug used the raw fractional value, making the
inventory skew ~1/100th of a tick and effectively inert. The fix is documented in
the skew README.
*Literature:* the EMA/RiskMetrics volatility estimator is J.P. Morgan/Reuters
RiskMetrics (1996). The role of volatility in optimal spread-setting is
**Avellaneda & Stoikov (2008, "High-Frequency Trading in a Limit Order Book")** —
the paper the whole quoting model is built on.

### A.7 The horizon (tau) — time of day

```
tau = clamp( (t1 - now) / (t1 - t0), 0, 1 )    # 1.0 at open, 0.0 at close
```

*What it is:* the fraction of the trading session still remaining.
*Layman:* early in the day there's lots of time for a position to move against
you, so you're cautious (wider spread, harder inventory lean); near the close
there's little time left, so the risk term decays and you tighten. It's the
"how much longer am I exposed" dial.
*In the engine:* `tau` multiplies both the risk half-spread and the inventory
skew, so both shrink into the close. It also underlies the EOD trigger's
minutes-to-close logic.
*Literature:* the finite-horizon inventory model is **Avellaneda & Stoikov
(2008)**; the deeper treatment of horizon and terminal inventory penalties is
Cartea, Jaimungal & Penalva (2015), Ch. 6-8. The classical inventory-control
roots are Ho & Stoll (1981, "Optimal Dealer Pricing under Transactions and
Return Uncertainty").

### A.8 Inventory (the position itself)

`pos` enters the **skew**, not the width — long leans both quotes down (sell
eagerly), short leans them up. It also gates the **soft band** (stop adding past
`soft_inv`) and the **hard cap** (`max_inv`). Full detail in the skew README.
*Layman:* the more stock you're holding, the more you want to get rid of it, so
you shade your prices to encourage trades that flatten you.
*Literature:* inventory as the driver of quote placement (not width) is **Ho &
Stoll (1981)** and Amihud & Mendelson (1980, "Dealership Market: Market-Making
with Inventory"); the modern continuous-time version is Avellaneda & Stoikov
(2008).

### A.9 The reactive jump gate (a safety input, not a signal)

Keeps a short `(timestamp, mid)` history over a lookback window; if the mid moves
more than `reactive_k · σ · √n` over that window, it goes fully dark for a
cooldown. It **reacts** to a jump (it cannot predict one — PSX jumps were
validated to have *no* book precursor) to avoid the compounding second and third
toxic fill.
*Layman:* if the price just lurched, stop quoting for a bit rather than keep
getting run over.
*Literature:* the "no predictable precursor" finding is consistent with the
jump-diffusion microstructure literature (e.g. Aït-Sahalia & Jacod on jump
detection); the reactive-withdrawal response is standard risk practice rather
than a specific model.

---

## Part B — The offline research feature store (`build_feature_store.py`)

This file computes a **superset** of statistics per event and writes one row per
two-sided book state to Parquet, so candidate signals can be evaluated *before*
being trusted in the live engine. It deliberately **excludes inventory** (that
would be endogenous — the model must learn the *market*, not our past policy;
inventory belongs only in the A-S skew). The label reference is always the
**arithmetic mid**, with the microprice kept as a *feature* to be judged, not as
the center.

Everything in Part A also appears here (spread, OBI, toxicity, volatility,
time-since-trade). The additional research features are:

### B.1 Spread in basis points and its Z-score

```
spread_bps = (ba - bb) / mid * 1e4
spread_z   = (spread_ticks - mean) / std       # over a rolling history, ≥20 samples
```

*Layman:* the spread expressed as a fraction of price (comparable across names),
and how unusually wide/narrow it is right now versus its own recent norm. A high
spread-Z ("spread is abnormally wide") often flags stress.
*Literature:* spread normalization and its time-series behaviour — Chordia, Roll
& Subrahmanyam (2001, "Market Liquidity and Trading Activity").

### B.2 OBI at multiple depths

```
obi_1    = (bq - aq) / (bq + aq)     # top-of-book imbalance, in [-1, 1]
obi_5    = book.obi(5)               # top-5 levels
obi_deep = book.obi(None)           # all visible levels
```

*Layman:* the same "who's keener, buyers or sellers" imbalance, measured at the
touch, across the top 5 levels, and across the whole visible book. Deeper OBI is
a stronger but noisier signal.
*Literature:* multi-level imbalance as a predictor — Cont, Kukanov & Stoikov
(2014, "The Price Impact of Order Book Events"); Cartea-Jaimungal-Penalva Ch. 3.
*Practical note from this project:* OBI (specifically `obi_1`) was retained as
**the primary directional skew signal**. A related feature, **`micro_dev_bps`**
(how far the microprice sits from the mid), was **dropped as redundant** — it is
mechanically ≈ imbalance × half-spread, and its partial correlation with the
target after OBI was already included was ≈ 0. Two features measuring the same
thing add noise, not signal.

### B.3 Order-flow imbalance (OFI), Cont-Kukanov L1

```
ofi_l1 = dq_bid - dq_ask       # net change in touch depth, sign-aware on relevels
```

with careful guards: on a price *move* the level re-levels rather than "flows,"
so the contribution is capped at the touch quantity to avoid the 100k+ spurious
spikes that a naïve whole-queue difference produces.
*Layman:* not just how imbalanced the book *is*, but which way it's actively
*changing* — are bids being added (bullish) or pulled (bearish) faster than
asks? It captures the *flow* of liquidity, not the static picture.
*Literature:* this is the canonical **Cont, Kukanov & Stoikov (2014)** OFI, one
of the best-documented short-horizon price-impact predictors in the literature.

### B.4 Queue-depletion rate (QDR)

```
qdr_bid = (prev_bq - bq) / prev_bq     # only when the bid PRICE held and depth fell
qdr_ask = (prev_aq - aq) / prev_aq
```

*Layman:* how fast the resting queue at the touch is being eaten while the price
stays put — a fast-consumption detector. A bid queue vanishing quickly is a
warning the price is about to drop through it.
*Literature:* queue dynamics and depletion as informative — Cont, Stoikov &
Talreja (2010, "A Stochastic Model for Order Book Dynamics"); Huang, Lehalle &
Rosenbaum (2015) on queue-reactive models.

### B.5 EWMA signed trade flow

```
ewma_trade_flow = decayed running signed volume (+buy / -sell)
```

*Layman:* a smoothed version of "net buying vs selling pressure" that fades old
trades gradually rather than dropping them off a fixed window edge.
*Literature:* trade-sign persistence and its predictive value — Lillo & Farmer
(2004, "The Long Memory of the Efficient Market").

### B.6 VPIN (Volume-Synchronized Probability of Informed Trading)

```
vpin = mean of recent volume-bucket order imbalances
```

*Layman:* a more sophisticated toxicity measure that chops trading into
equal-*volume* (not equal-*time*) buckets and measures the buy/sell imbalance in
each. It was designed specifically for high-frequency markets and is meant to
spike *before* liquidity crises.
*Literature:* **Easley, López de Prado & O'Hara (2012)** — the paper that
introduced VPIN and controversially linked it to the 2010 Flash Crash. (Also
worth knowing: VPIN has been *critiqued* — Andersen & Bondarenko (2014) argue
much of its predictive power is mechanical. So it is included as a feature to
*test*, not a truth to assume — exactly the right posture for a feature store.)

### B.7 Time since last trade

```
time_since_trade_ms = now - last_trade_ms
```

The research-store version of the "quiet" signal (§A.5), kept as a continuous
feature rather than a boolean. Same literature (Easley & O'Hara 1992).

### B.8 Realized volatility in bps

```
realized_vol_bps = sigma * 1e4
```

The §A.6 volatility, expressed in basis points and matched EMA-for-EMA to the
live engine so research and production reconcile.

---

## Part C — What we learned about which inputs actually work

This is the part most references omit, and it's the most valuable. On **PSX data,
207 days, the pilot names**:

1. **Microprice / OBI-as-fair-value: HARMFUL for a passive maker.** The single
   biggest finding. The microprice is a genuinely good next-mid predictor (as the
   literature says) — but centering a *maker's* quotes on it leans you into
   adverse selection, because you are the counterparty to the move it predicts.
   Production quotes on the **plain mid**. OBI survives only as a directional
   *skew* signal, kept separate from fair value.

2. **`micro_dev_bps`: REDUNDANT.** Mechanically ≈ imbalance × half-spread;
   partial correlation ≈ 0 once OBI is in. Dropped. A caution against stacking
   collinear features.

3. **Realized-volatility gate: DID NOT PAY.** Widening/pulling on σ-spikes did
   not avoid enough adverse fills to outweigh the fills it forfeited. Rejected.
   (Details in the main project README, Phase 4.)

4. **Deep-sweep (L10+) toxicity gate: REAL BUT UNTRADEABLE.** Post-sweep markout
   was monotonic in sweep depth (the theory held) but ~a few thousandths of a bp
   against a ~6 bp edge — too small to justify going dark. Not built. (Main
   README, Phase 7.)

5. **Toxicity (order-flow imbalance): USEFUL.** Retained as the adverse-selection
   charge and size cut — the Glosten-Milgrom intuition works in practice here.

6. **Inventory skew: ESSENTIAL, once correctly scaled.** The original bug (raw
   fractional σ) made it inert; after the price-vol fix and per-symbol
   `session_scale` calibration, it is the primary inventory control.

The meta-lesson: **a statistic being well-cited and predictive in the literature
does not mean it helps *your* strategy in *your* market.** The microprice is the
cleanest example — theoretically superior, empirically harmful for a passive
maker on PSX. Every signal was tested on data before being trusted, which is the
entire reason the feature store exists as a separate laboratory from the live
engine.

---

## Part D — What may still be worth exploring (gaps)

Honest list of things the engine does **not** currently use that the literature
suggests could help, and which the feature store already computes or could:

- **OFI as a live skew input.** `ofi_l1` (Cont-Kukanov) is one of the strongest
  documented short-horizon predictors and is *computed in the feature store* but
  **not wired into the live skew** — only OBI is. Testing OFI (or an OFI/OBI
  blend) as the directional skew is the most promising unexplored lever.
- **QDR as a pre-emptive pull.** Queue-depletion on the resting side is a
  fast "about to be run over" signal; it could feed the reactive gate or a
  side-specific widen, rather than only reacting *after* the mid has moved.
- **VPIN vs. simple toxicity.** The live engine uses the simple |signed|/gross
  toxicity; whether VPIN's volume-bucketing adds anything over it on PSX was
  never A/B'd in production (and given the Andersen-Bondarenko critique, that test
  should be run before adopting it).
- **Adaptive `flow_window` / `vol_alpha`.** Both are fixed event-count windows;
  their wall-clock meaning drifts with trade rate (fast at the open, slow midday).
  A trade-rate-adaptive window is a plausible refinement.
- **κ (order-arrival intensity) calibration.** The Avellaneda-Stoikov *base*
  half-spread term is currently inert (`as_base_weight = 0`) because κ was never
  calibrated. Calibrating κ per symbol would activate the theoretically-grounded
  base spread rather than relying entirely on the risk + adverse-selection +
  floor construction.
- **Multi-level / deep OFI.** Only L1 OFI is computed; the impact literature
  (Cont-Kukanov) extends to deeper levels, which PSX's 10-level feed supports.

None of these were pursued because the project reached its economic conclusion
(spot MM real but sub-TREC-threshold; futures dead) before signal refinement
would have changed the decision. They are the natural first experiments on the
**next** exchange, where the same feature store and engine apply unchanged.

---

## File and reference summary

**Files:**
- Live quoting engine: **`micro_mm.py`** (`MicrostructureMM`) — computes spread,
  OBI/microprice, toxicity, quiet, EMA volatility, horizon; produces quotes.
- Offline feature laboratory: **`build_feature_store.py`** — computes the
  superset (adds spread_bps/-Z, obi_5/obi_deep, ofi_l1, qdr_bid/ask,
  ewma_trade_flow, VPIN, micro_dev_bps, realized_vol_bps, time_since_trade);
  writes Parquet, trades nothing.
- Companion: **`README_Skew_and_Triggers.md`** — how these inputs become quote
  placement.

**Core academic references:**
- Ho & Stoll (1981) — inventory-based dealer pricing (skew, not width).
- Amihud & Mendelson (1980) — dealership inventory model.
- Glosten & Milgrom (1985); Kyle (1985) — adverse selection / informed trading.
- Easley & O'Hara (1992) — time between trades as information.
- Glosten & Harris (1988); Huang & Stoll (1997) — spread decomposition.
- Avellaneda & Stoikov (2008) — the finite-horizon quoting model this is built on.
- Stoikov (2018) — the micro-price.
- Cont, Kukanov & Stoikov (2014) — order-flow imbalance and price impact.
- Cont, Stoikov & Talreja (2010) — stochastic order-book dynamics.
- Easley, López de Prado & O'Hara (2012) — VPIN / flow toxicity (and Andersen &
  Bondarenko 2014 for the critique).
- Cartea, Jaimungal & Penalva (2015), *Algorithmic and High-Frequency Trading* —
  the textbook synthesis; Ch. 3 (imbalance/microprice), Ch. 6-8 (horizon,
  terminal inventory), Ch. 10 (the theoretical foundation used throughout).
