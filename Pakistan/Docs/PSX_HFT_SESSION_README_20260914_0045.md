# PSX HFT — Session Handoff README

**Date:** 2026-09-11 (rev. 2026-09-12, 2026-09-14 — see *Corrections* at the end) · **Scope:** queue-skew deploy decision, weighted-OBI investigation, tooling fixes, lead-lag screen build, universe expansion
**Environment:** Mac M4, Python 3.12, venv `.backtest`, `config_pk` paths, DuckDB for parquet
**Results directory (all outputs go here):** `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/`

---

## TL;DR — decisions reached

1. **DEPLOY queue skew, config `QT_2t`** (fixed 2 ticks per side). Confirmed on the full year: beats plain OBI by **+1.3036 bps/day** (paired, t = +9.42, p = 1.3e-17), and earns the most money of any arm — **8,736,780 PKR** vs QBPS_2's 8,278,591 (+5.5%). Edge stable-to-growing over the year.
   *Note:* QT_2t and QBPS_2 are **not** statistically distinguishable from each other — 0.03 bps/day apart against standard errors of ~0.14. The choice is raw PKR (QT_2t) over risk metrics (QBPS_2: Sharpe 31.6 vs 30.6, maxDD −16,884 vs −19,413, win 96.4% vs 94.4%). Made on total PKR.
2. **Keep plain L1 OBI** as the imbalance signal. Weighted multi-level OBI (`wobi`) does **not** add tradable edge — flat P&L head-to-head, and the apparent win-rate gain was a magnitude-blind illusion.
3. **Add a tradeability gate** before deploy and before universe expansion: exclude names where `capture_pkr ≤ 0` **or** `|markout|/capture > 0.7`. Verified under QBPS_2: catches both structural losers (TPL, PACE), misses none, wrongly excludes none.
4. **Universe expansion IN PROGRESS** — 114 names (38 existing + 76 never run), OBI vs QT_2t, 197 days. Script: `universe_expand.py`.

---

## 1. Full-year queue-skew confirmation — THE DEPLOY DECISION

**Script:** `fullyear_confirm.py` · **Run:** 6 configs × 38 names × 197 days = 44,916 cells, ~9.5 h, 9 workers, checkpointed.
**Outputs:** `fullyear_confirm_20260911_1456.csv`, `..._DAILY_...parquet` (5,910 rows), `..._PERNAME_...parquet` (179,280 rows).
**Reconciliation:** 0.0% unexplained on all 6 configs (decomposition trustworthy).

### Results (day-as-unit, n = 197)

| config | net_bps ± SE | net PKR | Sharpe | maxDD (PKR) | win% |
|---|---|---|---|---|---|
| OBI (control) | 2.560 ± 0.205 | 2,767,978 | 14.1 | −137,390 | 81% |
| QT_1t | 4.038 ± 0.166 | 6,470,046 | 27.5 | −25,262 | 94% |
| **QT_2t** | **3.863 ± 0.143** | **8,736,780** | **30.6** | **−19,413** | **94%** |
| QBPS_2 | 3.831 ± 0.137 | 8,278,591 | 31.6 | −16,884 | 96% |
| QT_1t+TAP_m.25 | 4.165 ± 0.162 | 6,734,415 | 29.1 | −23,665 | 96% |
| QT_1t+TAP_m.5 | 4.120 ± 0.163 | 6,614,187 | 28.5 | −22,146 | 95% |

> **Estimator note.** `net_bps` above is the **day-average** (mean of the 197 daily notional-weighted figures). The run summary CSV reports the **pooled** notional-weighted figure over all fills, which is a different and smaller number (OBI 2.386, QT_2t 3.822, QBPS_2 3.740). `net_pkr` agrees exactly between the two. Always name which estimator a quoted edge refers to.
>
> **Sortino dropped from this table.** At a 94–96% win rate there are only 7–12 losing days in 197, so downside deviation is estimated from a handful of observations and the config ranking flips between standard Sortino definitions. Not decision-grade here; use Sharpe and maxDD.

### Paired vs OBI (config − OBI, day-paired, n = 197 shared days on every arm)

| config | paired edge (bps/day) | SE | t | p |
|---|---|---|---|---|
| QBPS_2 | +1.2717 | 0.1330 | +9.56 | 5.1e-18 |
| **QT_2t** | **+1.3036** | 0.1383 | **+9.42** | 1.3e-17 |
| QT_1t | +1.4779 | 0.1073 | +13.77 | 1.3e-30 |
| QT_1t+TAP_m.5 | +1.5601 | 0.1070 | +14.58 | 4.4e-33 |
| QT_1t+TAP_m.25 | +1.6048 | 0.1121 | +14.32 | 2.6e-32 |

Range **+1.27 to +1.60 bps/day**, t from **+9.42 to +14.58**. Paired equals unpaired on every arm (identical day coverage), so the table means are directly comparable. Queue skew is real and confirmed; the 30-day finding held.

**Sign convention:** t is computed as `(config − OBI)`, so **t > 0 means the config beats OBI**.

### Why QT_2t
- **Most money:** 8,736,780 PKR, +5.5% over QBPS_2, the largest of any arm.
- **Highest paired edge among the two PKR leaders:** +1.3036 vs +1.2717.
- **Simplest rule:** a fixed tick count, no price-to-tick conversion to verify or maintain.
- **What it costs:** ~1 point of Sharpe, ~2,500 PKR of max drawdown, 2 points of win rate versus QBPS_2. Small against a reduced-size rollout.
- **Alternatives by objective:** `QBPS_2` if drawdown is the binding constraint. `QT_1t+TAP.25` has the highest bps edge but is dominated on total PKR (6.73M) — **taper dropped.**

### ⚠️ Cheap-tick exclusion (undocumented until now)
`fullyear_confirm.py` hardcodes `CHEAP_EXCLUDED = {"KEL", "PIBTL", "TPL"}` and forces those three names to plain OBI — **in every config, including the control comparison**. Reason recorded in the code: their books are ~1 tick wide, so there is no room to skew inside, and the skew was measured to hurt them. Two consequences:
- Those three contribute exactly 0 PKR of the skew improvement in every arm. Verified: their per-day QBPS_2−OBI difference is **bit-identical zero** on all 197 days.
- **This is a name blacklist, and it violates the methodology principle below.** It caught three names by hand out of 38; it says nothing about the 76 new names in the expansion. Replace it with a rule on book width before the expanded universe is deployed.

### Where the edge lives (mechanism)
- By session bucket: `middle` is the workhorse; **the skew edge is concentrated in `preclose45` + `last15`** — exactly the toxic end-of-day windows where **plain OBI loses money** (−0.14, −0.05 bps). Queue skew rescues those buckets (maker's-curse reversal). Real mechanism, not a fitted number.
- The edge is **capture-driven:** skew configs have *more* negative markout than OBI (more fills that then move against you) but *much* more capture, netting far positive. **Implication:** the edge depends on backtest fill/queue realism → roll out at reduced size first and compare live fill rates to modeled before scaling.

### Time stability (decay check)
Split the 197 days in half (H1 = first ~98, H2 = last ~99). **QBPS_2** edge over OBI: **H1 +0.94, H2 +1.60** → stable-to-growing, no decay. OBI itself flat across halves (2.51 → 2.61), so it's the skew edge specifically, not a market tailwind.
**Not yet computed for QT_2t** — rerun the half-split on the deploy config before quoting a decay conclusion for it.
**Deploy expectation = full-year paired average ~+1.30 bps/day over OBI** (do NOT extrapolate a single half).

---

## 2. Per-ticker vetting + tradeability filter

**Script:** `per_ticker_stats.py` (per-ticker overall + per-bucket: net_pkr, notional-weighted net_bps, Sharpe, Sortino, maxDD, win%, capture/markout/fee/liq).
**Outputs:** `per_ticker_OVERALL_QBPS_2.parquet`, `per_ticker_BYBUCKET_QBPS_2.parquet` (in the Results dir).

⚠️ **These are QBPS_2 figures.** The deploy config is now QT_2t — regenerate as `per_ticker_OVERALL_QT_2t` before using them as the deploy record. The gate logic and the two excluded names are expected to carry over, but that is an expectation, not a measurement.

- **36/38 names net positive** under QBPS_2, spread across all sectors. Broad and healthy.
- **Two structural losers (exclude via rule, not by name):**

| name | net PKR | net_bps | capture | markout | \|mko\|/cap | pathology |
|---|---|---|---|---|---|---|
| PACE | −48,691 | −1.196 | +323,886 | −259,379 | 0.801 | **Adverse selection** — markout eats 80% of capture |
| TPL | −37,882 | −4.862 | **−34,321** | +40,692 | n/a | **Illiquidity** — negative capture (can't earn the spread) |

- **Tradeability rule (generalizes to universe expansion):** exclude if `capture_pkr ≤ 0` (catches TPL) **or** `|markout|/capture > 0.7` (catches PACE). Verified: 2 of 2 engine-negative names caught, 0 missed, 0 profitable names wrongly excluded. Removes −86,573 PKR (+1.05% of total).
- ⚠️ **The 0.70 threshold is fitted on n=1.** PACE is the only name the markout arm catches, and it sits at 0.801 — 14% of headroom above the cut. The capture arm (`capture ≤ 0`) is principled and threshold-free; the 0.70 is not. Check where the other 36 names' ratios sit before applying this to 76 unseen names.
- **Watch-but-keep (thin/low-margin, positive):** PIOC, FNEL, HASCOL, PIBTL, KEL. Also NBP/AKBL = high-volume, low-margin, high-drawdown keepers (profitable only because capture is huge).
- **Do NOT** give individual names bespoke skew params — that's overfitting to this sample. Keep skew uniform; exclude structural losers by the rule.

---

## 3. Weighted multi-level OBI (wobi) — investigated, REJECTED

**Question:** does a decay-weighted multi-level OBI add predictive info over L1?

### 3a. Correlation screen (`obi_decay_gate.py`)
First fixed a **methodology bug**: `_partial_corr_over_l1` residualized with OLS (Pearson-orthogonal) then evaluated with Spearman (rank) — a metric mismatch that made the L1-over-itself sanity check floor at −0.044 instead of 0. **Fix:** rank-transform first, residualize both sides in rank space, Pearson on rank residuals = true partial Spearman. Validated on synthetic ground truth; sanity check now reads **+0.000** exactly.
Screen result (20 days, 760 name-days): best incremental partial-ρ was `w_d2_r0.5` at **+0.023** — tiny, shrinks with depth, and the large standalone gaps (e.g. `w_d5_r0.3` +0.69) were **repackaged L1** (incremental ~0.012). Broad direction (frac>0 ≈ 0.58–0.60) but economically marginal.

### 3b. P&L head-to-head (`wobi_headtohead.py`)
L1 vs `w_d2_r0.3` vs `w_d2_r0.5`, 38 names × 20 days.

| arm | net PKR | mean bps | Sharpe | win% |
|---|---|---|---|---|
| L1 | 427,810 | 3.613 | 13.40 | 70% |
| wobi_d2r0.3 | 464,104 | 3.789 | 13.52 | 90% |
| wobi_d2r0.5 | 466,623 | 3.793 | 13.24 | 85% |

- Paired vs L1: **+0.175 / +0.180 bps/day, t = 1.02 / 0.88 — NOT significant.**
- The eye-catching win-rate jump (70%→90%) is **misleading**: Sharpe is flat/lower because the weighted arms trade many small losing days for fewer, **2.4–3.4× bigger** ones (win rate is magnitude-blind; Sortino/Sharpe see through it).
- **Confound:** weighted arms quote +2.7% more notional (smoother signal trips the throttle less) — part signal, part activity.
- **Verdict: keep L1.** Underpowered + activity-confounded. If ever revisited: activity-matched (lower the treatment threshold to match L1's firing rate) on the full year. **Low priority** — signal refinement is the avenue with the least headroom.

### 3c. `micro_mm.py` wiring for wobi
Added gated kwargs `weighted_obi` / `wobi_depth` / `wobi_decay` + `_weighted_imb()` (default off = byte-identical). **⚠️ INCIDENT:** the patched file was built from a stale project snapshot **missing `queue_skew_bps`**; installing it clobbered the real `micro_mm.py` and broke `fullyear_confirm.py`. Caught immediately by the smoke (loud `TypeError` at the constructor), recovered from PyCharm Local History. **Current `micro_mm.py` = restored real file (has `queue_skew_bps`), does NOT have `weighted_obi`.** Lesson baked in: always patch the *live* file, `git commit` before swapping, verify with `grep -c queue_skew_bps`.

---

## 4. Tooling fix — `onetick_regime_test.py`
`dr.pnl()` can return `None` on unpriceable days → `TypeError` inside `fifo_attribution` (`float(None)`). It was being silently swallowed by `_work_date`'s broad `except` (dropping days as misleading "TypeError" skips), not crashing. **Fix:** guard `if dr is None or dr.pnl() is None: return []` in `_one`, matching the house convention in `clip_size_sweep.py`.

---

## 5. Lead-lag cross-asset screen — BUILT, NOT YET RUN

**Script:** `leadlag_screen.py` · **Framing A** (defensive throttle: pull/cut the exposed side when a sector leader sweeps — NOT a directional lean).
- Sector-leader lead-lag (leader = highest median traded value per sector, data-driven).
- **Hayashi-Yoshida async estimator** (the fix for the Epps/non-synchronicity artifact). **Validated on synthetic ground truth:** recovers a known +800 ms lag (peak above zero-lag), rejects independent series (no false peak).
- 4 kill-gates: (1) leader updates faster, (2) HY lead-lag peak, (3) day-as-unit sign stability, (4) economic gate vs 1.554 bps fee.
- **TODO before running:** verify/complete the sector map (13 names left `UNCLASSIFIED` in the file). **Lowest priority** — real chance it dies at gate 1 on PSX synchronicity, and even a pass only earns a defensive-throttle engine test.

---

## 6. Universe expansion — IN PROGRESS

**Script:** `universe_expand.py` (a copy of `fullyear_confirm.py` with 7 changes; the original is untouched).
**Scope:** 114 names × 197 days × 2 configs (OBI control + QT_2t) = 44,916 cells ≈ 9.5 h on 9 workers, checkpointed.
**Outputs:** `universe_expand_{stamp}.csv`, `..._DAILY_...parquet`, `..._PERNAME_...parquet`, journal `universe_expand_CKPT.jsonl`.

### Name selection
114 names from `persistence_REG_2p00.csv` (491 screened, 207 days, TREC fees) at `days_traded ≥ 100` **and** `notional_m_median ≥ 25M PKR`. 38 are the current book and act as a **control** — the run must reproduce their known QT_2t figures. 76 have never been through the engine.

### ⚠️ The screen does not predict the engine — do not rank on it
Measured against realised per-name `net_pkr` on the 38 names where engine ground truth exists (Spearman, n=38):

| screen column | ρ vs net_pkr | ρ vs net_bps |
|---|---|---|
| `net5_trec_median` | **−0.244** | −0.157 |
| `spread_bps_median` | **−0.333** (p=0.041) | **−0.583** |
| `as_share` (adverse-selection share) | −0.190 | **−0.795** |
| `notional_m_median` | **+0.354** (p=0.029) | +0.327 |
| `pct_days_edge5_pos` | +0.270 | +0.404 |

The screen's headline metric is **anti-predictive**. Ranking 491 names by `net5_trec_median` — or by spread — would preferentially select the names the engine loses money on: wide spread is illiquidity, and the screen credits a half-spread capture that is never realised. **Selection here is on notional only**, the one column with a defensible positive sign.

Also noted: `net5_trec_median = mk5_bps_median − 0.78` on all 491 names, i.e. the screen charges **one side's fee** where `fill_attribution.fee_bps_roundtrip()` charges **1.554**. Constant +0.774 bps bias on every name. It moves the level, not the ranking (490/491 are positive either way, so `net5 > 0` is not a gate). Worth resolving whether `mk5` is a one-sided or round-trip quantity.

### Expansion headroom is smaller than assumed
Names clearing each liquidity bar, with how much of the known engine P&L each retains:

| bar | total | already run | NEW | engine PKR kept | engine losers kept |
|---|---|---|---|---|---|
| ≥200M / ≤15bps | 36 | 29 | 7 | 7,264,289 | 0 |
| ≥100M / ≤20bps | 49 | 32 | 17 | 7,911,981 | 0 |
| ≥50M / ≤30bps | 75 | 37 | 38 | 8,316,473 | 1 |
| ≥25M / ≤40bps | 110 | 38 | 72 | 8,278,591 | 2 |

The premise of "~500 names of headroom" does not survive its own screen: at a zero-loser bar the universe is ~49 names and only 17 are untested. **If the expansion confirms this, clip size / per-name capacity — not name count — is the binding lever**, and `pov_sweep` / `clip_size_sweep` are the next place to look.

---

## Current state of files

| file | status |
|---|---|
| `micro_mm.py` | **RESTORED** (real file, `queue_skew_bps` intact, floored at 1 tick — line 1607 `max(1, round(...))`). Needs `git commit`. `weighted_obi` NOT present. |
| `obi_decay_gate.py` | FIXED (rank-consistent partial-Spearman; self-test = 0). |
| `onetick_regime_test.py` | FIXED (None guard). |
| `fullyear_confirm.py` | Ran clean; deploy-decision output produced. **Unmodified.** |
| `per_ticker_stats.py` | NEW; writes to Results dir. Needs a QT_2t rerun. |
| `wobi_headtohead.py` | NEW; result = keep L1. Requires `weighted_obi` in `micro_mm.py` to run (currently absent). |
| `leadlag_screen.py` | NEW; validated, not run; sector map needs completion. |
| `stage0_audit.py` | NEW; bucket-aware audit of the full-year run + screen-vs-engine calibration. All outputs timestamped, never overwrites. |
| `universe_expand.py` | NEW; 114 names × OBI/QT_2t. Run with `SMOKE_DAYS = 3` first. |

---

## Next steps (priority order)

1. **`git commit` the restored `micro_mm.py`** (with `queue_skew_bps`) so it can't be lost again. Add a smoke test asserting the constructor signature.
2. **Smoke `universe_expand.py`** (`SMOKE_DAYS = 3`). The likely failure: `H.load_scales()` / `load_profiles()` / `load_segments()` were built for the old 38 names, and a missing `session_scale` raises `TypeError` at the `MicrostructureMM` constructor by design. If so, run the calibration scripts over the 76 new names first. Then `SMOKE_DAYS = None` for the full 9.5 h run.
3. **Apply the tradeability gate to the expansion output** and check where the surviving names' `|markout|/capture` ratios sit — that tells you whether the 0.70 threshold is robust or resting on PACE alone.
4. **Replace `CHEAP_EXCLUDED` with a rule on book width** before deploying the expanded universe. A three-name blacklist does not extend to 76 unseen names.
5. **Regenerate `per_ticker_*` under QT_2t** so the deploy record matches the deploy config.
6. **Deploy QT_2t** with the tradeability gate. Conservative rollout: reduced size first, compare live fills to modeled (the edge is capture-driven), then scale. Expectation ~+1.30 bps/day over OBI.
7. **Fill-realism harness** — `micro_mm.py` already has `log_fill_state`, which makes `dr.order_log` an unbiased fill-probability dataset. Unused so far. This is what makes the capture-driven edge trustworthy live.
8. *(Later, low priority)* wobi activity-matched full-year head-to-head — only to formally close it; expectation is flat.
9. *(Later, lowest priority)* complete the lead-lag sector map and run `leadlag_screen.py`.

---

## Methodology principles reinforced this session
- **Day-as-unit error bars** always; never pool rows for significance.
- **Name the estimator.** Day-average net_bps and pooled notional-weighted net_bps are different numbers off the same run. Quoting one and the PKR from the other makes a table that cannot be reconciled.
- **Check the key grain before computing.** The `DAILY`/`PERNAME` artifacts are keyed by `(date, config, bucket)` with an `ALL` aggregate row sitting in the same `bucket` column as its four components. Summing every bucket double-counts by exactly 2×. Reconcile any collapse against the run summary before trusting a statistic built on it.
- **Partial correlations:** residualize *and* evaluate in rank space (OLS-then-Spearman is invalid).
- **Win rate is magnitude-blind** — judge on Sharpe/Sortino. And at a 95%+ win rate, Sortino itself rests on a handful of losing days; it is not decision-grade there either.
- **A screen is only valid if it predicts the engine.** Measure the rank correlation against ground truth before ranking anything by it. A screen that passes 490 of 491 names is a ranking, not a gate.
- **Exclude structural losers by RULE, not by name** — name-blacklists overfit and don't survive universe expansion. (`CHEAP_EXCLUDED` is currently in breach of this.)
- **Validate estimator/solver logic on synthetic ground truth before real data.**
- **Patch the live file, commit before swapping** (the `micro_mm.py` clobber lesson).
- **All script outputs → the Results directory**, never relative paths. Timestamp them; never overwrite or delete a prior run's output.

---

## Corrections applied 2026-09-12

Verified by `stage0_audit.py` against `fullyear_confirm_DAILY_20260911_1456.parquet` and `..._PERNAME_...`, reconciled to the run summary CSV at ~1e-15 relative error on all six configs.

1. **Deploy config changed QBPS_2 → QT_2t** (§TL;DR, §1). Decided on total PKR: 8,736,780 vs 8,278,591. The two are near-identical on *bps* edge over OBI (+1.3036 vs +1.2717); **on money they are not** — see the 2026-09-14 correction 1. The trade-off is stated explicitly rather than presented as a clean win.
2. **Paired range corrected** from "+1.30 to +1.61" to **+1.27 to +1.60** (§1). The old lower bound excluded QBPS_2's own +1.2717 — i.e. the then-deploy config fell below the range the document claimed covered every config. Full per-config table added.
3. **t signs corrected** from "−9.4 to −14.6" to **+9.42 to +14.58** (§1). Magnitudes were right; the sign convention was inverted. Convention now stated inline.
4. **Estimator note added** (§1). The table's `net_bps` is the day-average; the run CSV's is pooled notional-weighted (OBI 2.386 vs 2.560). Both correct, different questions.
5. **Sortino removed from the main table** (§1), with the reason: 7–12 losing days in 197 makes it definition-sensitive and unstable. The "QT_2t has best Sortino" and "TAP.25 worst downside" claims rested on it and are withdrawn.
6. **`CHEAP_EXCLUDED` documented** (§1). Three names hardcoded out of the skew in every config, previously unrecorded. Flagged as a breach of the document's own by-rule-not-by-name principle.
7. **0.70 gate threshold flagged as fitted on n=1** (§2).
8. **§2 marked as QBPS_2-specific** — needs regenerating under the new deploy config.
9. **Decay check scoped to QBPS_2** (§1); the H1/H2 split has not been computed for QT_2t and must not be quoted for it.
10. **§6 added** — universe expansion, name selection rationale, the anti-predictive screen result, and the headroom table.

Unchanged and still standing: the §1 mechanism findings (bucket concentration, capture-driven edge), all of §3 (wobi rejected), §4, §5, and the portability rationale for a bps-denominated rule — `micro_mm.py` line 1607 floors the conversion at 1 tick (`max(1, round(...))`), so it does adapt across price levels as claimed. QT_2t was chosen on money, not on any defect in that argument.


---

## Corrections applied 2026-09-14

All measured against `fullyear_confirm_PERNAME_20260911_1456.parquet` (six configs,
38 names) and the two corrected-calibration runs. Every figure below reconciles to
engine P&L.

1. **"Statistically indistinguishable" was wrong as stated** (§Corrections 2026-09-12,
   item 1). True of the bps edge over OBI (+1.3036 vs +1.2717). **Not true of the
   money:** QT_2t − QBPS_2, paired by day, is **+2,326 PKR/day, SE 611, t = 3.81**.
   QT_2t trades more notional (1,459,785 fills vs 1,328,957) and converts a
   near-identical bps edge into significantly more PKR. The old phrasing invited the
   reading that the deploy choice was a coin flip. It was not.

2. **"Queue skew" is a misnomer — it is a conditional PARALLEL TRANSLATION.**
   This cost several hours of misdirected analysis and is the most important
   correction in this document. From `micro_mm.py`:
   `bid = reservation − half_buy`, `ask = reservation + half_sell`. Bid-heavy sets
   `half_buy = half − qs` and `half_sell = half + qs`, so:

   ```
   bid = reservation − half + qs    ->  UP   by qs
   ask = reservation + half + qs    ->  UP   by qs
   width = 2·half                   ->  UNCHANGED
   ```

   Both quotes translate together; the quoted width never changes. The variable
   names ("favourable side steps closer, exposed side steps back") describe the
   half-spread variables, not the resulting geometry. **QT_2t is a discrete,
   thresholded fair-value lean**, not a queue-position skew.

   Two consequences:
   - The *asymmetric* component lives in `obi_defensive` (1-tick widen of the
     exposed side), which is ON in **both** configs. QT_2t's entire contribution
     over the control is the 2-tick translation.
   - This re-opens the microprice axis. `use_microprice` was removed because the
     **continuous** lean `fair += λ·(imb−0.5)·spread` was destructive at λ=+1.
     QT_2t leans the same way, discretely, past a threshold, and makes 8.5M. The
     axis was closed on a form that is not the one that works.

3. **A parallel shift is capture-neutral to first order.** On a buy, capture
   = `half − qs`; on a sell, `half + qs`. The engine ends every day flat (EOD
   book-walk liquidation), so shares bought = shares sold by construction and the
   two cancel. Any measured capture change is therefore a **fill-selection**
   effect, not a placement effect.

4. **The capture/markout exchange rate, measured.** 38 names, aggregate bps:

   | step | Δcapture | Δmarkout | Δliq | Δnet bps | ΔPKR | fills |
   |---|---|---|---|---|---|---|
   | OBI → QT_1t | **+0.849** | +0.556 | +0.151 | **+1.555** | +3,702,068 | +38.5% |
   | QT_1t → QT_2t | **−0.314** | +0.163 | +0.031 | **−0.119** | +2,266,735 | +43.6% |

   **The first tick is free** — capture and markout both improve, no trade-off.
   **The second tick trades capture for markout at 0.52:1** — the markout gain does
   not pay for the capture given up, and net bps *falls* (3.941 → 3.822). The
   +2.27M PKR comes entirely from volume: turnover 16.4bn → 22.9bn. Day-paired
   t = 14.25.

   **So QT_1t is the efficiency winner and QT_2t is the money winner.** The marginal
   tick buys capacity, not edge. If the binding constraint is ever capital or
   inventory rather than throughput, the deploy pick is wrong.

5. **Arguments raised and WITHDRAWN on 2026-09-14** — recorded so they are not
   re-litigated:
   - *"2 ticks is 29.4 bps on KEL and 0.11 bps on SAZEW, so the tick denomination
     is broken."* Arithmetically true, economically false. On names under 200 PKR,
     QT_2t (2 ticks) beat QBPS_2 (1 tick after rounding) by **+386,831 PKR**. More
     aggression on cheap names is better, not worse. Queue priority is discrete —
     one tick is one price level regardless of share price — so the discrete
     priority gain outweighs the continuous bps cost.
   - *"The queue-skew path needs the exit path's fee clamp."* `fee_bps` is
     **−1.554 in every bucket under both configs**. The skew cannot cost anything
     in fees; it changes capture, and capture is traded for markout deliberately.
     A capture-based floor would clamp exactly the fills the signal rates highest.
     The exit path's clamp is correct *there* because its objective is to get
     flat, not to earn markout.
   - *"Replace `CHEAP_EXCLUDED` with a median-book-width rule."* The median is the
     wrong statistic: the skew only acts when `|imb−0.5| > 0.15`, so the decision
     must be conditioned on the state in which the rule acts.

6. **`CHEAP_EXCLUDED` suppresses its own evidence.** KEL, PIBTL and TPL show
   *exactly* 0.00 difference between configs because the blacklist forces both to
   plain OBI. The 197-day run contains **no information** about them; the
   justification traces to a 30-day sample under a calibration since corrected
   twice. `universe_expand_v3.py` adds `HONOUR_CHEAP_EXCLUDED` so the exclusion can
   be confirmed or retired on current data.

7. **Resume-aggregation bug found and guarded.** `universe_expand.py` skipped
   already-journaled work but never folded those results back into the
   aggregation, silently cutting a 113-name run to 75. `universe_expand_v2.py`
   adds a calibration-tagged journal name (so a calibration change cannot resume
   onto stale cells) and a hard stop that names the cohort that would vanish.

8. **38 incumbents re-run under corrected calibration.** QT_2t 8,736,780 →
   **8,543,790** (−2.2%); OBI 2,767,978 → **2,724,700** (−1.6%). Combined 113-name
   book under one calibration: **QT_2t 13,285,941 (3.316 bps), OBI 4,234,221
   (2.134 bps)**, 88 of 113 names positive.

9. **Untested axes, as of 2026-09-14.** Every measured result — QT_1t, QT_2t,
   QBPS_2, the TAP variants — sits on the parallel-shift axis. Never tested:
   **QT_3t+** (is the magnitude curve still rising?), **improve-only** (bid moves,
   ask held — narrows, first-order capture cost), **retreat-only** (ask moves, bid
   held — widens, first-order capture gain), and any **graded/staircase** response
   where the shift scales with the strength of the imbalance.
