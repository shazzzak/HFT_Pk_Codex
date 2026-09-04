# PSX Passive Market-Making — Executive Summary, File Manifest & Code-Flow Reference

*All figures below are transcribed from actual code files and recorded run outputs.
Where a number is a 5-day smoke or used a superseded accounting basis, that is
stated explicitly. Nothing here is estimated or filled in from assumption.*

---

## PART 1 — FILES NEEDED TO RUN THE BACKTEST

### Core engine (required)
- **`mm_backtest.py`** — the replay engine. Order-level `Book`, `Backtester`
  event loop, `LatencyModel`, fill logic, mark-to-market, OBI feature columns.
  This is the heart; everything else feeds it or wraps it.
- **`micro_mm.py`** — the market-making strategy (`MicroMM`). Ho-Stoll / Avellaneda-
  Stoikov inventory-skew quoting, OBI-defensive skew, OFI-defensive retreat,
  continuous microprice lean (`micro_lambda`), reactive toxicity gate, halt/band
  gating. This is the "brain" the engine drives.
- **`run_legacy_mm.py`** — runner that wires a strategy to the engine for a
  single-name / single-config run.

### Data build (required once, upstream of the engine)
- **`build_feature_store.py`** — builds the per-symbol/per-date Parquet feature
  store (trades, ob_updates, ob_snapshot, misc) the engine and analytics read.
- **`PSX_Parser_Mac.py`** — parses the raw PSX FIX (.gz) feed into the parsed
  Parquet tables that `build_feature_store.py` consumes.

### Sweep drivers (required to reproduce the panels)
- **`skew_sweep_2d.py`** — the multi-config sweep driver (Run A form: lambda-lean
  sweep). Writes summary + DAILY + PERNAME CSVs, day-as-unit error bars.
- **`skew_sweep_2d_RUNB_ofi.py`** — the Run B form: per-name OFI grid. Separate
  output prefix so it does not collide with Run A.

### Analytics / attribution (required for the P&L decomposition & findings)
- **`fill_attribution.py`** — capture / markout / adverse-selection decomposition
  on fills.
- **`spot_capture_markout_decomp.py`** — spot capture-vs-markout decomposition.
- **`kyle_lambda.py`** — Kyle's-lambda illiquidity estimator (1/5/15-min bins),
  uses TRUE `aggressor_side` via feature-store signed_volume.
- **`lean_vs_lambda.py`** — pre-registered Spearman test: per-name lean benefit
  vs Kyle's-lambda rank (runs after Run A lands).

### Support / preflight / tests (recommended)
- **`preflight_coverage.py`** — verifies all names are runnable + calibrated
  before a long sweep.
- **`persist_fills.py` / `persist_fills_v2.py`** — fill persistence.
- **`mm_harness.py`** — harness for multi-name orchestration.
- **`test_liq_fills.py`**, **`test_ofi_defensive.py`** — unit/sign-off tests.

### Minimum set to run one backtest end-to-end
`PSX_Parser_Mac.py` → `build_feature_store.py` → (`mm_backtest.py` + `micro_mm.py`
driven by `run_legacy_mm.py`). Add `skew_sweep_2d.py` for multi-config panels and
`fill_attribution.py` / `kyle_lambda.py` for the analysis layer.

### Data / environment dependencies (not code files)
- Raw FIX feed: 209 daily `.gz` PSX files (FIXT.1.1).
- Feature store: `.../feature_store/{SYM}/date=YYYY-MM-DD/*.parquet`.
- venv `.backtest`; Python 3.12; Mac M4 (10 cores).
- Four-table Parquet store: trades, ob_updates, ob_snapshot, misc.

---

## PART 2 — CODE FLOW & MECHANISM (the "what is modeled" prompt)

Use the following as a standing description of how the system works.

### The two-clock anti-lookahead design (the core correctness property)
The engine runs ONE pass over a merged, time-ordered event stream of three kinds:
`S` = exchange snapshot, `U` = incremental order-book update (add/cancel),
`T` = trade. Two clocks run simultaneously:
- **`ts_exch` (exchange time)** drives the historical book and ALL fill decisions.
- **`know` (knowledge time)** = running max of capture timestamps (`ts_cap`) —
  drives what the strategy is ALLOWED to see and act on. Knowledge never runs
  backwards even though receive timestamps jitter.

This separation is what prevents lookahead: the strategy can only act on
information whose wire-arrival time has already passed, while fills are judged
against the true exchange book.

### Per-event processing order (in `Backtester.run`)
For each event, in strict order:
1. **`_activate_until`** — land our own in-flight orders/cancels that are due
   before this event (our messages have latency; they arrive on the exchange
   timeline, not instantly).
2. **Fill checks against the PRE-event book** — trades test our resting orders;
   incoming adds test for crossings; cancels update our queue positions. This
   MUST precede step 3 so a fill is judged against the exact book the aggressor
   actually hit.
3. **Apply the event to the historical book** — snapshot = full-state REPLACE of
   the entire order dict; update = add/cancel mutates the dict; trade = consume
   liquidity.
4. **Mark to market** — equity = cash + position × mid, plus OBI feature columns
   (`obi_5` near-touch, `obi_deep` all visible levels). One row per event that has
   a valid TWO-SIDED book; one-sided/empty-book events are skipped.
5. **Advance knowledge time and requote** — inside the session window the strategy
   may requote; `_requote` applies the halt/band-pin gate and circuit-band clamp
   so quotes are pulled automatically outside continuous trading. After session
   end, all working orders are pulled.

### The order book model (`Book`)
- State is ONE dict: `order_id -> Order`. Price levels are NEVER stored; they are
  derived on demand (`bbo`, `qty_at`). Keeping order-level state (not level-
  aggregated) is what makes EXACT queue-position tracking possible — this is only
  feasible because PSX data carries individual order IDs inside L10.
- Reconciliation: incremental adds/cancels/trades mutate the dict between
  snapshots; each exchange snapshot (~every 5s) REPLACES the entire dict
  (full-state reconciliation), so any drift from unresolvable events is wiped at
  least every snapshot interval. Validated: 97.9% of trades print inside the
  reconstructed pre-trade touch.

### The strategy model (`MicroMM`, in `micro_mm.py`)
Grounded in Cartea-Jaimungal-Penalva Ch.10 and Ho-Stoll / Avellaneda-Stoikov:
- **Inventory skew (the core control):** inventory SHIFTS the quote pair
  (placement), not width. Long inventory → positive skew → reservation price
  below fair → leans to sell down the position. Skew is computed in PKR-variance
  units via a per-name calibrated `session_scale` (back-solved: PPL = 7.6,
  UBL = 3.9). `session_scale` is a REQUIRED keyword-only argument — no default —
  to prevent skewing on a wrong scale.
- **Fair value / microprice lean:** `fair = mid + micro_lambda·(imb−0.5)·spread`.
  `micro_lambda = 1` = classic microprice (leans INTO flow); `= 0` = plain mid;
  `< 0` = defensive (leans AWAY from flow). Proven algebraically identical to the
  old boolean `use_microprice` at lambda ∈ {0,1}.
- **OBI-defensive skew:** widens the threatened side based on order-book imbalance.
- **OFI-defensive retreat:** live Cont-Kukanov L1 order-flow imbalance over a
  min(N events, T seconds) hybrid window; normalized Σe/Σ|e| ∈ [−1,+1]; retreat-
  only (widen the threatened side). Warm-up guarded; first15 bucket hard-OFF.
- **Reactive toxicity gate:** on a large adverse move, pull/wide the side exposed
  (modes: off / symmetric / inventory-only).
- **Halt / circuit-band gating:** quotes pulled or pinned during halts and at the
  ±10% / PKR-band scrip circuit limits.

### What is modeled vs. flagged simplifications
Modeled: order-level queue position, our-order latency (`LatencyModel`), exact
fill priority, per-name tick/fee structure (spot TREC ~1.55 bps round-trip),
session buckets (first15 / middle / preclose45 / last15), circuit bands, halts,
mark-to-market equity curve, capture/markout/adverse-selection decomposition,
liquidation-fill reconciliation.
Flagged simplifications (explicit in code): end-of-session pull is instant and
latency-free; final inventory is marked at last mid — the closing auction and a
book-walk liquidation are not modeled.

### Analytics layer
- **Capture** = signed (mid_at_fill − fill_px)·qty — the spread earned at the fill.
- **Markout** = signed mid drift after the fill (adverse selection). Decomposed
  into diffusion (`dif_bps`) and jump (`jmp_bps`) components.
- **Net** = capture + markout + liquidation terms − fees.
- **Kyle's lambda** = price impact per signed notional, estimated through-origin
  `dP = lambda·Q` per name per day, day-as-unit median ± SE, using TRUE
  aggressor_side (no Lee-Ready).
- **Statistical discipline:** day-as-unit error bars (never pool row-count SEs);
  Spearman/rank over Pearson (heavy zero-markout mass, fat tails); pre-register
  tests before sweeps.

---

## PART 3 — SWEEPS RUN & FINDINGS (factual, with accounting-basis caveats)

### A. Core spot economics — full 38-name × 197-day panel (OFF config)
Exact liquidation-fill reconciliation; reconciliation residual exactly 0.
- **Portfolio: +1.70 bps, 2,545,765 PKR over 197 days (~283k PKR/month).**
- Bucket breakdown:
  - **Middle bucket IS the business:** +2.545 bps, ~2.53M PKR.
  - first15: ~breakeven-positive (+0.6 bps, ~100k PKR).
  - **preclose45 + last15: NET NEGATIVE** (−74k, −7k PKR) — end-of-day
    liquidation drag.
- Markout is deeply negative everywhere (daily t ≈ −14, p ≈ 1e-31) — a real
  adverse-selection tax. Capture (+7.5 bps middle) fights −3.2 bps diffusion
  markout.
- **Markout is diffusion-dominated, NOT jump-dominated:** `dif_bps` carries −3 to
  −5 bps; `jmp_bps` is tiny (~0.02–0.09 bps). *(This corrects an earlier
  mis-statement in-session.)*

### B. OFI-defensive quoting — full 197-day × 38-name × 9-config panel
822 min (13.7h) runtime, 67,374 cells, reconciliation exactly 0.
- **Verdict: OFI-defensive DOES NOT PAY.** Paired OFF−OFI t-stats at n=197: not one
  OFI config beats OFF. `min20ev/2s @ 0.20` was significantly WORSE (t = +2.55);
  the other 7 configs |t| < 1 (ties).
- The 2-day 38-name canary had shown OFI "winning" — that was noise, caught by the
  day-as-unit error bars (their exact purpose).
- Mechanism of failure: forfeits queue position, skips benign fills, and sweeps
  blow through the retreat. Same fate as microprice (`lambda = 1`): predicts
  (R² > 0) but does not pay.
- Machinery retained behind `ofi_defensive = False` (byte-identical when off).

### C. Kyle's lambda illiquidity — all 38 names, 1/5/15-min bins
- Median R² rises monotonically with bin size: **1min = 0.007, 5min = 0.024,
  15min = 0.047.** Denoising works as intended.
- Even at 15min, R² ≈ 0.05 → **flow explains only ~5% of PSX price variance** — a
  real structural finding: PSX price moves are weakly flow-driven (jump/news
  dominated). Consistent with the OFI/microprice failures.
- **Ranking is horizon-STABLE:** Spearman ρ = 0.989 (1v5), 0.986 (1v15),
  0.994 (5v15). Trust the ordinal rank, not the cardinal value.
- Thinnest / highest-impact: FNEL, TPL, PACE, PIAHCLA, HASCOL, NPL, TOMCL.
  Deepest / lowest-impact: OGDC, PSO, HUBC, FFC, PPL, UBL, NBP, MEBL.
- Only `lam_bps_per_notional` is cross-name comparable (raw lambda is not).

### D. OBI-defensive on/off (top-10 book, 5-day smoke, OLD pro-rata plug)
From the Aug-28 48-config sweep (stamp 20260828_1636), winning cell
exit_ticks=1 / obi+ / tol=0 / mid:
- **obi OFF = 6.452 bps / 52,915 PKR; obi ON = 7.315 bps / 53,805 PKR →
  obi-defensive added ~+0.84 bps / ~+890 PKR.**
- **Caveats:** 5-day smoke (tiny n, no error bars); used the superseded pro-rata
  liquidation plug, not exact reconciliation. A clean per-name obi on/off at
  n=197 was never saved.

### E. Grid-point optimization within the 48-config sweep (same smoke/plug)
Portfolio net-bps, top-10, obi+/mp− column:
- Winner exit1/tol0/obi+ = **7.315 bps** (54,002 PKR anchor).
- exit2/tol2/obi+ = 5.712; exit2/tol0/obi+ = 6.617; exit0/tol0/obi+ = 5.636.
- e.g. exit2/tol2 → winner = **+1.60 bps**; exit0/tol0 → winner = **+1.68 bps**.
- **Caveats:** same 5-day-smoke / old-plug basis. These are within-sweep grid
  spreads, NOT before/after of two deployed production configs.

### F. Runs launched (in flight / recent)
- **Run A** — lambda-lean sweep (`skew_sweep_2d.py`): OFI off; MICRO_LAMBDA =
  [None, −1.5, −2.0, −2.5, −3.0, −3.5, −4.0, −4.5, −5.0] = 9 configs × 38 × 197 =
  67,374 cells. Tests whether a defensive lean converts 'through' → 'at_queue' and
  pays. `lean_vs_lambda.py` runs immediately after.
- **Run B** — per-name OFI (`skew_sweep_2d_RUNB_ofi.py`): 1 OFF + 8 OFI configs,
  same universe. Re-run WITH per-name capture.
- **Run C** — Kyle's lambda: COMPLETE (Section C above).
- Note: Runs A and B each total 9 configs → both 67,374 cells (arithmetic
  coincidence; the grids are genuinely different — verified, no bug).

---

## PART 4 — HISTORICAL PARAMETER FACTS (verified from transcripts)
- Before Aug 25: **`exit_ticks` did not exist** (feature introduced Aug 28 as
  `EXIT_TICKS = [0,1,2,3]`); effective behavior was exit-at-touch.
- Before Aug 25: **`tol_ticks` default was 0.0** consistently.
- exit_ticks=2 and tol_ticks=2 existed ONLY as sweep grid points, never as a
  pinned production baseline. There was no exit=2/tol=2 production config that was
  "optimized away."

---

## PART 5 — AGGRESSOR-SIDE DATA (factual status)
- The `trades` table has `aggressor_side` (BUY/SELL) — **real and usable**
  (exchange-marked trade sign; ground truth, no Lee-Ready).
- The `initiator` (aggressor order ID) column **exists but is empty/unusable in
  practice** — per-aggressor grouping is NOT available.
- Current use of `aggressor_side`: ONLY inside `kyle_lambda.py` (signed_volume).
- Not yet used in quoting. Candidate use: realized-aggression imbalance as a
  trade-flow toxicity signal (untested; low prior given ~5% flow R², but the one
  flow variant not yet falsified). Recommended path: diagnostic first (do toxic
  fills concentrate in high-aggression windows?) before building any gate.

---

## PART 6 — KEY LEARNINGS
- Predictive signal (R² > 0) ≠ profitable quoting: microprice (λ=1) and OFI-
  defensive both predict yet LOSE money.
- Lean direction: λ=1 leans INTO flow (wrong sign for a maker); λ<0 is defensive
  (right direction) but its discrete form (obi_defensive) already captures most of
  the modest benefit.
- Kyle's λ (measured market illiquidity) ≠ micro_lambda (chosen strategy lean) —
  do not conflate.
- Day-as-unit error bars are essential (caught the 2-day OFI false positive at
  n=197).
- ALWAYS preserve the per-name axis in sweep outputs.
- Single-source formulas via helper; assert every batch `.replace()` matched.
- Profile before hypothesizing bottlenecks.
- The core PSX spot MM economics (+1.70 bps, ~283k PKR/month, exact
  reconciliation) is the encouraging result. The real next levers are
  middle-bucket scaling and EOD-drag recovery — neither involves OFI.

---

## PART 7 — OPEN ITEMS / NEXT STEPS
1. Run `lean_vs_lambda.py` the moment Run A lands (Spearman lean-benefit vs
   Kyle's-λ rank; Spearman not Pearson given R²≈0.05).
2. When Run B lands: does OFI help any specific ticker despite tying on the
   portfolio? (pre-register one test; multiple-comparisons discipline).
3. Inventory skew fix: replace the hard inventory gate with continuous A-S skew
   (inventory carry is the single largest source of strategy loss; strategy ends
   systematically short ~62–69% of days).
4. EOD-drag recovery: suppress fresh late-day position-opening (preclose45 +
   last15 lose ~80k) — candidate to lift +1.70 → ~+2.2 bps.
5. Middle-bucket concentration: can optimizing middle-hour quoting scale the
   +2.5M?
6. Wire `fill_attribution.py` onto `mm_backtest.py`'s real `at_queue` fills.
7. Aggressor-side diagnostic (Part 5) before any trade-flow gate.
8. Add incremental checkpointing to sweep drivers (they currently write only at
   the end — a crash loses all in-memory progress).
