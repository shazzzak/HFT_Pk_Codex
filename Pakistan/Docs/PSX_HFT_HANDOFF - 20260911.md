# PSX HFT Market-Making — Project Handoff / State-of-Play

**Purpose:** paste this into a new thread as the full context. It summarizes the
project, the confirmed edge, everything tested (and closed), the open questions,
the infrastructure, the hard-won lessons, and the exact next steps. Written after
a very long working session; the thread got unwieldy and this is the checkpoint.

---

## 0. TL;DR (read this first)

- **Confirmed production edge:** a passive market-making strategy on PSX (Pakistan
  Stock Exchange) whose only validated alpha is the **OBI (L1 order-book-imbalance)
  throttle** — a light *size cut* on the exposed side when the book leans against
  you. ~2.94 bps/day, Sharpe ~22, on a 38-name universe. Net ~$10k/yr gross — a
  pilot, not yet a business. The leverage is universe expansion + more venues.
- **NEW confirmed edge (this session), pending full-year confirmation:** **QUEUE
  SKEW** — shift *both* quotes one tick toward the OBI lean (spread width
  unchanged, center shifted). On 30 days it nearly **doubled P&L (685k→1.37M PKR),
  Sharpe 22→37, cut max drawdown to ~1/4**, and helped 35/38 names. This is the
  first thing ever to beat the OBI throttle on every axis. **Must be confirmed on
  the full 197-day sample before deploying** — that run is built and queued.
- **The single most important next action:** run `fullyear_confirm.py` (6 configs,
  197 days) — it picks the deploy config. Everything else is secondary.
- **The biggest untapped opportunity:** **universe expansion** — re-screen the full
  PSX list (~500 names) under TREC fees, not just the 38-name pilot. Queue skew on
  150+ names is where a pilot becomes a business. Untouched all session.

---

## 1. Who / What / Where

- **Operator (SZ):** CFA, Kellogg MBA, runs Big Byte Insights (Lahore) — an alt-data
  firm selling to US hedge funds. Building this PSX HFT market-maker in parallel as
  "Chapter 1" of a multi-exchange plan targeting thin, under-competed emerging
  markets (Borsa İstanbul, ADX, Tadawul, Philippines, Africa, Far East). Thesis:
  microstructure edge is more durable where competition is low.
- **Regulatory path:** a TREC (Trading Right Entitlement Certificate) for own-account
  direct market access. **Fees ≈ 1.55 bps round-trip** under TREC — a critical
  parameter; symbol attractiveness completely reverses between retail (~35.5 bps RT)
  and TREC costs, so no watchlist decision is valid until the fee regime is locked.
- **Machine:** Mac M4, **16 GB RAM** (this matters — see §8), 10 cores, Python 3.12,
  venv `.backtest`. **The data was moved this session** to:
  - Parsed store: `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Parsed`
  - Results:      `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results`
  - Code:         `/Users/shazzak/PycharmProjects/HFT/Pakistan/existing_mm_live/`
- **Paths are now centralized** in `config_pk.py` (renamed from `config.py`). The two
  foundational engine files (`run_legacy_mm.py`, `mm_harness.py`) now import paths
  from `config_pk` (with a loud-fail if it can't be imported). Most *other* scripts
  still hardcode paths and will break on a move until touched — a known cleanup debt.

---

## 2. The data & engine (the frozen foundation)

- **Data:** PSX FIX feed, parsed into a 4-table Parquet store (ob_snapshot,
  ob_updates, trades, misc), Hive-partitioned `date=YYYY-MM-DD`. ~207 trading days;
  after a 10-day trailing-median warmup, ~197 are tradeable. Symbols are tickers
  (OGDC, HBL, PPL, …), not numeric codes.
- **Key data facts:**
  - `ob_updates` has only `ORDER_ADD` and `CANCEL` (no MODIFY).
  - `trades` carries a true `aggressor_side` (no Lee-Ready needed).
  - `resting_order_id` ~99.9% populated at L1 but only ~24% resolvable overall
    (fast near-touch fills use an anonymized path) → **kappa (A-S fill intensity)
    is NOT cleanly measurable on PSX**; the base half-spread term is left off.
  - PSX is a **human/retail market**: ~0 fast (<100ms) iceberg refills, sparse/stale
    deep book levels, bursty low-frequency flow.
- **Engine (`mm_backtest.py`) — FROZEN, never reimplemented:**
  - Event-driven replay: strategy re-evaluated after every book event (S/U/T).
  - `Book` — validated order-book reconstruction (add/cancel/trade, `__NEG_`
    placeholder handling, `ranked_depth(n)`, `bbo()`, `obi(n)`).
  - `Backtester` — FIFO queue-gated fills, seeded latency model, two-clock
    anti-lookahead (`ts_exch` drives fills; `know`/`capture_ts` drives strategy
    info — never conflate), EOD liquidation (walks the real book, haircuts unfilled
    residual), `fee_for` (TREC).
  - **Two-clock, day-as-unit error bars, and reconciliation-to-engine-P&L are the
    non-negotiable disciplines.** Every trustworthy sweep asserts
    `measured_net == engine_pnl`.
  - **Timestamps are MILLISECONDS** in `build_events` output (a ns-vs-ms confusion
    caused a real bug earlier — the iceberg refill-window was 100× too wide).
  - **NEW this session:** a **gated taker path** (`allow_taker` flag on the
    Backtester, `MyOrder.taker` flag). Default OFF = byte-identical post-only engine.
    When on, a deliberately-tagged crossing order executes against the opposite book
    (walks levels, books cash/fees, reason="taker") instead of being post-only
    rejected. Fully tested. This is permanent infra for any future strategy that
    must take liquidity (hedging, arb legs, urgent risk reduction).
  - **NEW this session:** fills are tagged with `regime` ("onetick"/"normal") and
    (optionally) post-time market state for the fill-probability work.
- **Strategy (`micro_mm.py`) — the live quoting logic:**
  - Avellaneda-Stoikov inventory skew (reservation-price shift on inventory), OBI
    throttle (the edge), EOD POV unwind, lock/circuit-breaker triggers, viability
    gate. Many experimental levers added this session, all **default-off and
    byte-identical when off** (see §4/§5).
- **Harness (`mm_harness.py`):** single source of truth for calibration loaders
  (scales, volume profiles, unwind windows, session segments), the per-symbol-day
  driver (`run_symbol_day`), and **`fifo_attribution`** — the validated FIFO P&L
  attribution that books round-trips to the *opening* bucket and defines the
  residual as `engine_pnl − matched` so it **reconciles by construction**. (Use
  this; do not hand-roll FIFO — that mistake cost hours this session.)

---

## 3. THE CONFIRMED EDGE: OBI throttle (production baseline)

- **Mechanism:** when L1 OBI leans against a side past a threshold, cut that side's
  clip to 0.5× for a short hold (300ms). A *defensive size skew*. It barely touches
  spread capture, which is why it survives.
- **Params:** `obi_throttle=True, obi_throttle_thresh=0.15, throttle_frac=0.5,
  throttle_hold_ms=300`.
- **Performance:** full-year ~2.77M PKR / 2.39–2.94 bps depending on the run/universe;
  Sharpe ~22, maxDD ~−14k, ~93% winning days on the 38-name universe.
- **Why it's the only survivor:** on PSX the predictable moves are ~0.5–1.5 bps and
  the spread capture IS the edge; anything that spends capture to dodge toxicity or
  chase direction loses. The throttle is a *light* size cut, not a pull or a lean.

---

## 4. THE NEW EDGE: Queue skew (the headline result of this session)

- **Mechanism (get this right — it was mis-described twice before settling):** on a
  book leaning bid-heavy (OBI up), shift **both** quotes up by a fixed distance —
  bid moves closer to the touch (better queue position, higher fill probability),
  ask moves up too (sell higher when the anticipated up-move comes). **Spread WIDTH
  is unchanged; the CENTER is shifted toward the lean.** It is a *pure center shift*,
  not an asymmetric widening. Implemented as `half_buy`/`half_sell` differing by the
  skew, which translates the pair while preserving the gap.
- **Why it works when ~everything else failed:** it does NOT spend spread capture
  (price width constant) and does NOT add toxic size — it only changes *which side
  you get filled on*. Evidence: middle-bucket signed-OBI-at-fill flips from ~0
  (OBI) to +0.027 (QT_1t) — you start getting filled *with* the book instead of
  against it. It directly reverses the maker's curse (see §6).
- **30-day sweep results (per `queue_fine_sweep`):**
  | config | net PKR | bps | Sharpe | maxDD | win% | notes |
  |---|---|---|---|---|---|---|
  | OBI (control) | 685k | 2.93 | 22 | −14.0k | 93% | baseline |
  | QT_1t (1 tick) | 1,365k | 4.11 | 37 | −3.3k | 97% | risk-adj winner |
  | QT_2t (2 ticks) | 1,637k | 3.42 | 31 | −9.9k | 93% | more fills |
  | QBPS_2 (2 bps) | 1,637k | 3.64 | **42** | **0** | **100%** | 0 drawdown on 30d |
  | QT_1.5t | 1,758k | 3.91 | 35 | −7.1k | 97% | **DROPPED — artifact (see below)** |
- **CRITICAL: QT_1.5t was dropped.** On a price-time-priority tick grid you cannot
  quote a half tick — 1.5 ticks *rounds* to 1 or 2 depending on grid alignment, so
  "1.5t" is a stochastic 1-vs-2 mix, an accident of rounding, not a real parameter.
  Only whole-tick (QT_1t, QT_2t) or per-name-rounded price-relative (QBPS) are clean.
- **Price-relative (QBPS) — the cheap-tick fix:** `queue_skew_bps` shifts by N bps of
  mid, converted to whole ticks (min 1). Same *economic* shift on every name (2 bps =
  ~1 tick on a PKR-9 stock, ~45 ticks on a PKR-900 stock). Fixed 1 tick is ~10 bps on
  a cheap stock (too coarse) but ~0.01 bps on an expensive one. **QBPS_2 was the risk
  champion on 30 days (Sharpe 42, 0 drawdown, 100% win).**
- **Per-ticker (30d):** QT_1t beats OBI on **35/38 names** — broad, not concentrated.
  The 3 losers (KEL, PIBTL, TPL) are sub-PKR-15, ~1-tick-wide books where you can't
  skew inside a 1-tick spread. **These 3 are EXCLUDED from queue skew** (forced to
  plain OBI) in the deploy config.
- **STATUS: needs full-year confirmation.** 30 days is a tuning sample. The
  zero-drawdown/100%-win on QBPS_2 is plausibly 30-day luck. `fullyear_confirm.py`
  (below) settles it on 197 days.

---

## 5. THE DEPLOY-DECISION RUN (built, queued, NOT yet run)

**`fullyear_confirm.py`** — 6 configs × 38 names × 197 days (~44,916 cells, ~8–10h):
1. **OBI** (control)
2. **QT_1t** (queue skew, 1 tick)
3. **QT_2t** (queue skew, 2 ticks)
4. **QBPS_2** (queue skew, 2 bps price-relative)
5. **QT_1t+TAP_m.25** (QT_1t + inventory taper, aggressive)
6. **QT_1t+TAP_m.5** (QT_1t + inventory taper, gentle)

- Cheap-tick-3 (KEL/PIBTL/TPL) excluded from skew (forced to OBI).
- Risk metrics (Sharpe/Sortino/maxDD/win%) built into the summary; safe-parquet
  writes (atomic + read-back verify — see §8 for why).
- **What it settles:** (a) which queue-skew config to deploy over the full year,
  (b) whether the taper's 30-day risk improvement is real or luck (see §5a).
- **Run command:**
  ```
  cd /Users/shazzak/PycharmProjects/HFT/Pakistan
  rm -f "/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/fullyear_confirm_CKPT.jsonl"*
  caffeinate -is python existing_mm_live/fullyear_confirm.py
  ```
  Run ALONE (RAM), ideally `--workers 3` or 4. Verify first heartbeat says 6 configs
  × 38 × 197. When done: read the built-in risk table (picks the config) AND pull the
  per-ticker breakdown from the parquet.

### 5a. The inventory taper (in the full-year run, prior = weak)
- **Mechanism:** as inventory grows toward what you can still clear by the close,
  taper the *add* side's size (ramp, replacing the never-binding soft_inv cliff),
  optionally boost the *reduce* side. Anchored to `_pov_capacity()` via a SEPARATE
  multiplier `inv_taper_pov_mult` (0.25 ≈ 2.5%-equiv → makes `util` bite) that is
  **decoupled** from `unwind_pov` (kept at production 10% so the EOD trigger/last15
  is untouched). This decoupling was SZ's insight and is the right design.
- **30-day result:** taper ALONE ≈ production (t<1, does nothing). Taper ON TOP of
  queue skew: +34k PKR (+2.5%), Sharpe 37→42, drawdown→0, 100% win — BUT per-ticker
  the +34k was **top-3-names = 105% of the gain** (concentrated, 24/38 positive).
- **Verdict pending:** looked like a fragile 3-name artifact on 30 days, but SZ
  correctly insisted it go into the full-year run rather than be dismissed on a
  mixed 30-day cut. The full year decides: if the zero-drawdown holds broadly →
  deploy queue-skew+taper; if it evaporates → deploy queue skew alone.

---

## 6. Fill-probability diagnostic — the MAKER'S CURSE, measured

- Built `fill_probability.py` (+ engine fill-state capture) to log, for EVERY posted
  quote (filled or not), the market state at post time, then measure fill rates.
- **THE headline finding — maker's curse is real and quantified: you are filled
  1.51× as often when the book is AGAINST you as when it favors you** (adverse-side
  fill 2.4% vs favorable-side 1.6%, day-as-unit, tight bars). Monotonic:
  favorable < neutral < adverse.
- **It's a queue-depth effect:** the curse is widest when there's a queue ahead
  (<2 median-trade-sizes ahead: 1.1% favorable vs 3.1% adverse) and nearly gone at
  the front / deep in the queue. On the favorable side you're buried behind a thick
  queue; on the adverse side you're at the front of a thin one and get run over.
  **This is exactly what queue skew attacks** — it moves you toward the front on the
  favorable side. The 1.51× curse and the +1.17 bps queue-skew edge are the same
  fact from two directions.
- **Also measured:** fill rate collapses to 0.5% on 1-tick books vs 1.9% on 8t+
  books (why cheap-tick names can't be quoted). Curse is worst at the open (first15
  fill 2.4%). Expected-fill-wait (from a recent trade-arrival-rate estimator,
  min(3min,150 trades) window) does predict fills (2.1% at <1min wait → 1.3% at
  10-30min), though shallowly.
- **Methodology notes baked in:** queue-ahead measured in *median-trade-size* units
  (exogenous, not clip size which is endogenous to your own sizing); session buckets
  computed from each day's own event span (handles Ramadan/Friday hours); a
  recent-arrival-rate estimator that counts *trades only* (churn/cancels can't zero
  it) with a *historic per-bucket fallback* when the window has no trades.

---

## 7. Everything tested and CLOSED (do not revisit without new reason)

Format: idea — verdict — one-line why.

**Directional / toxicity signals (ALL closed negative — a strong, repeated pattern):**
- **Aggressor-flow throttle** (both fast-gate and full time-based) — predicts
  momentum, DOESN'T PAY (−0.62 bps/day vs OBI, t=−7.25) — pulling forfeits capture.
- **Run-persistence step-aside** — flow runs ARE persistent (collapsed ladder
  50%→75% at 6+ consecutive same-side orders, real cross-trader momentum) but the
  adverse markout after a run is tiny (<fee) and SHRINKS with run length — nothing
  to dodge.
- **Run-reprice** (cancel/reprice the exposed side up on a run) — NEGATIVE (−52k),
  queue-position loss on price-time priority swamps the tiny avoided move; worst on
  the liquid names (HBL) where queue position is valuable.
- **Aggression-lean** (shift fair value toward vol-weighted aggression imbalance) —
  CATASTROPHIC and monotone worse (−1.75 to −10.8 bps); leaning collapses capture
  (capture went NEGATIVE at k=1). Same failure mode as the microprice lean.
- **QDR (queue-depletion-rate) throttle** — REDUNDANT with OBI (OBI already catches
  those fills).
- **OFI-defensive widen** — DEAD at n=197.
- **Decile/dark toxicity gates** — real effect, untradeable magnitude (sub-tick).

**Order-book / signal-shape:**
- **OBI depth (equal-weight obi_1/3/5/7)** — L1 is BEST, deeper monotonically worse
  (equal-weighting lets stale deep levels drown the touch).
- **Decay-weighted OBI (obi_decay_gate)** — depth×decay sweep. **UNRESOLVED / weakly
  negative but CONTESTED — see §9.** Standalone predictive gap is 2-4× L1's, but
  partial-correlation-over-L1 ≈ 0 (no incremental info). The two metrics conflict
  and were NOT cleanly reconciled. SZ (rightly) is skeptical of the "keep L1"
  verdict. **Needs a direct P&L head-to-head, not dueling correlations.**
- **Microprice lean (Run A, earlier)** — catastrophic at every λ.
- **Iceberg / hidden-refill detection** — PSX has ~0 fast (<100ms) refills (human
  market); slow M1/M2 "absorption" is real but tiny (<fee) and concentrated in one
  name (HBL). Detector VALIDATED + PORTABLE to venues with algorithmic participants.

**Sizing / inventory / exit-timing:**
- **Clip-size sweep** — edge DECAYS with size; 2× best risk-adjusted, 3× profit-max,
  ≥4× drawdown explodes. Do not scale clips up.
- **Favorable-side SIZE boost** — fails monotonically (more size = worse; adds toxic
  fills). Contrast with queue skew which changes *position* not *size*.
- **soft_inv sweep** — SLACK; inventory rarely approaches the soft band, so the
  hard-gate cliff almost never binds.
- **POV cap (acquisition)** — REDUNDANT (soft_inv binds tighter).
- **unwind_pov sweep (10% vs 5% vs 2.5%)** — 10% CONFIRMED optimal; lower only hurts
  last15 (EOD unwind fires too early, gives away spread). Effect isolated to last15;
  first15/middle/preclose byte-identical.
- **Age-based passive exit** — BACKFIRED (crystallizes losses; adverse drift already
  sunk by ~4 min).
- **Age-CROSS exit** (taker flatten of aged inventory, using the new taker path) —
  much WORSE (−7.9 bps at 5min); paying full spread to realize a loss already
  incurred. Both exit-timing avenues dead.
- **FIFO holding-time** — profit lives in the first minute; >15min round trips lose;
  but forcing shorter holds backfires (selection, not causation).

**Latency:** speed is IRRELEVANT (markout is diffusion, not pick-off; colo doesn't
help). Do not invest in co-location for PSX.

**Locked 1-tick-book MM (this session, `onetick_regime_test`)** — one-sided
OBI-gated entry on 1-tick books, exit at touch. **CLOSED NEGATIVE on reconciled
20-day data:** the onetick regime loses on a cash basis (net PKR red at every
threshold); the 1-day preview that showed a −15→+15 bps swing was noise (evaporated
over 20 days). Keep excluding cheap-tick names. (NOTE: getting a *trustworthy*
answer here required fixing a real attribution bug — see §9.)

**Futures track (earlier):** built delivery-futures MM (carry+hedge, same-day
flatten). Futures ~8× cheaper fee (~0.19 bps RT). Not the current focus.

---

## 8. Infrastructure lessons & gotchas (IMPORTANT for the next thread)

- **16 GB RAM is a hard constraint.** Each replay worker holds a symbol-day's event
  stream; heavy diagnostics on liquid names are memory-hungry. Running TWO replay
  jobs at once (e.g. 3+6 workers) exhausts RAM → macOS swaps (seen: 70M unused, 5GB
  compressor) → jobs crawl AND risk corrupt/truncated output. **RULE: one replay job
  at a time, ≤3–4 workers.** Not-swapping at 3 workers beats thrashing at 6.
- **Parquet write safety.** A truncated parquet ("no magic bytes in footer") comes
  from an interrupted write (memory pressure, kill). Fix applied to the sweeps:
  `_safe_parquet` = write to `.tmp` → read-back verify row count → atomic
  `os.replace`, with a CSV fallback. **A recurring earlier bug: `.parquet` filenames
  written with `.to_csv()`** (CSV data, parquet extension) — reads fine as
  `read_csv`, fails as `read_parquet`. If you hit "magic bytes" errors, first try
  `pd.read_csv(path)` — the data is probably fine, just mislabeled. All four sweeps
  now write real, verified parquet. **Standing rule: outputs are Parquet, not CSV.**
- **Checkpoints are NOT timestamped** (by design, for resume). Before re-running any
  sweep, delete its `*_CKPT.jsonl` and `.jsonl.done` — a stale `.done` makes a run
  report "0 cells." A mismatched-config journal can silently resume into nonsense.
- **`discover_dates()` empties if `PARSED_ROOT` is wrong.** Standalone scripts must
  set the store path (now via `config_pk`). After the data move, several scripts
  broke until pathed correctly.
- **TEST THROUGH THE FULL PIPELINE, not just functions in isolation.** Multiple bugs
  this session lived at *integration seams* (a function consuming another's output,
  a guard referencing columns added by the caller) and passed unit tests while
  failing end-to-end. For any data-pipeline change: run the full `_one`→report path
  with a mock before handing over. "It compiled" ≠ "it works."
- **REUSE the frozen engine/harness; never hand-roll.** A per-regime P&L attribution
  was hand-rolled from scratch and reconciled at only 9% — because it re-invented
  FIFO/fees/residual instead of calling `mm_harness.fifo_attribution` (which
  reconciles by construction). Delegating to the validated function fixed it to
  100%. This is a standing rule and it was violated at cost.

---

## 9. Open questions / unfinished (for the new thread)

1. **[TOP PRIORITY] Run `fullyear_confirm.py`** — the deploy decision (§5). Picks the
   queue-skew config on 197 days and settles the taper. Not yet run.
2. **Decay-weighted OBI — UNRESOLVED (§7).** Standalone predictive gap is genuinely
   2-4× L1's (w_d5_r0.3: +0.69 vs L1 +0.16, clears error bars), but the
   partial-correlation-over-L1 test says ~0 incremental. These conflict and I could
   NOT reconcile *why* a 4×-stronger standalone signal would carry zero incremental
   info — that's suspicious, and given attribution/metric bugs elsewhere this
   session, the "keep L1" verdict is NOT trustworthy. **Resolve by a direct P&L
   head-to-head:** wire w_d5_r0.3 into the trigger, run vs L1 in the backtest, judge
   on net bps/Sharpe. If the strong standalone gap is real edge it shows as money;
   if repackaged L1 it nets flat. Do NOT close on the correlation metrics alone.
3. **Universe expansion [BIGGEST OPPORTUNITY].** Re-screen the full PSX list (~500
   names) under TREC fees with a `has_futures` split. Queue skew on 150+ names, not
   38. This is the pilot→business step and is completely untouched. Machinery exists
   (`run_daily_stats`, `persistence_metrics`, `build_watchlist`).
4. **Spread-gate universe-wide (secondary).** The one-tick work suggested "sit out
   1-tick moments" *might* lift the normal regime — but the 20-day reconciled result
   did NOT confirm it (the day-1 effect was noise). Only worth revisiting as a clean
   universe-wide "suppress quoting when spread==1t, measured on total P&L" test if
   there's appetite; prior is now weak.
5. **Audit `fifo_attribution` callers for the liq-fill double-count.** The engine now
   emits EOD liquidation fills (reason `liq`/`liq_residual`) into `dr.fills`. Feeding
   those into `fifo_attribution` double-handles the liquidation (it fees the haircut
   *mark*, ~1-2 PKR/day leak on clean-liquidation days). The one-tick tool was fixed
   by excluding `liq*` rows before attribution. Other tools
   (`analyze_bucket_attribution`, sweep decompositions) may have the same small leak
   — they reconcile only because it's tiny. Worth an audit.
6. **Cheap-tick names (KEL/PIBTL/TPL)** are excluded from queue skew (1-tick books).
   No passive MM works on them; leave excluded unless a fundamentally different
   approach emerges.

---

## 10. The core strategic picture (the "why" behind the tactics)

- **PSX is a thin, human, retail-driven book.** Measured directly this session:
  posted-limit-to-traded-volume ratio ≈ **2.5:1** vs US equities ≈ **23:1** — PSX
  has ~9× LESS resting quote competition per unit of flow. That IS the "thin,
  under-competed" thesis, quantified. It's the reason the venue is attractive despite
  small absolute P&L.
- **The recurring, hard-won truth:** on PSX, **spread capture is the edge, and
  anything that spends capture loses.** Seven+ directional/toxicity signals all
  "predicted but didn't pay" because acting on them (pulling, leaning, crossing,
  repricing, boosting size) forfeits more capture than the ~0.5–1.5 bps predictable
  move is worth. The two things that WORK — OBI throttle and queue skew — both
  preserve capture (a light size cut; a center shift at constant width). This is the
  organizing principle: **improve *where/whether* you rest without paying spread.**
- **Where the money actually scales:** NOT more signals (that avenue is exhausted and
  well-understood). It's (a) **more names** (universe expansion — the immediate
  multiplier) and (b) **more venues** (the multi-exchange plan — where the portable
  detectors like iceberg, and signals that need algorithmic counterparties, may
  finally pay). Queue skew and the OBI throttle should port to any thin,
  tick-constrained, price-time-priority venue.

---

## 11. File index (what's what in `existing_mm_live/`)

- **Engine/strategy/harness (frozen core):** `mm_backtest.py`, `micro_mm.py`,
  `mm_harness.py`, `run_legacy_mm.py`, `config_pk.py`, `snapshot_prep.py`,
  `halt_state.py`.
- **The deploy-decision run:** `fullyear_confirm.py` (built, queued).
- **Queue-skew sweeps:** `queue_fine_sweep.py` (fixed-tick + bps), `skew_sweep.py`
  (size/queue), all with `_safe_parquet`.
- **Taper/POV:** `taper_sweep.py`, `pov_sweep.py`.
- **Diagnostics (this session):** `fill_probability.py` (maker's curse, arrival-rate,
  fill model), `onetick_regime_test.py` (locked-book MM, reconciled), `obi_decay_gate.py`
  (decay-weighted OBI, unresolved), plus prior: `iceberg_detect.py`, `iceberg_sanity.py`,
  `run_persistence.py`, `run_markout.py`, `mkt_limit_profile.py`, `kyle_lambda.py`,
  `sweep_risk_score.py` (standing risk scorer), `inv_profile_diag.py`, etc.
- **A full 132-file catalog exists** (an Excel built earlier this session:
  `HFT_codebase_catalog.xlsx`) with per-file purpose/imports/outputs/category.

---

## 12. Standing rules (SZ's working preferences — honor these)

- Production-grade methodology by default; flag any simplification explicitly with
  `SIMPLIFICATION:` and give both versions.
- **Challenge incomplete approaches; don't assume simplicity is wanted.** Ask
  clarifying questions before big runs.
- Comment every code line (comment on its own line ABOVE the code).
- Code changes: give file name, then the change; snippet edits with exact anchors,
  applied bottom-up. Grep/sed in one block. Assert every `.replace()` matched.
- Day-as-unit error bars ALWAYS (never pooled row-count SEs). Per-name axis never
  dropped. Partial-ρ / Spearman over Pearson given zero-markout mass + heavy tails.
  Rolling median over mean.
- Reconciliation-to-engine-P&L on every P&L attribution.
- **Be objective; push back; the data decides, not priors.** (SZ repeatedly and
  correctly caught a drift toward prematurely closing marginal-but-live ideas —
  guard against that.)
- Give complete `.py` files with download links (not `python -c` one-liners) for
  anything needing cd/imports/logic. Parquet outputs, not CSV.

---

*End of handoff. The immediate next action is to run `fullyear_confirm.py` (alone,
3–4 workers) — it picks the queue-skew deploy config on the full year. After that,
universe expansion is the highest-leverage move.*
