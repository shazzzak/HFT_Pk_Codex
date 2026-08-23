# PSX Passive Market-Making Research — Project README

**Big Byte Insights — September 2025 to August 2026.**
This document is the complete chronological record of the PSX (Pakistan Stock
Exchange) passive HFT market-making research project: what was built, what was
measured, what worked, what failed, and why the project concluded the way it did.
The final verdict, reached on evidence: **futures market-making on PSX is dead
(toxic fills, not illiquidity); spot market-making is real (Sharpe 5.67,
reconciliation-exact) but earns roughly 250k PKR/month — too small to justify the
Rs 35M TREC net-worth requirement and a brokerage entity.** The engine, the
attribution machinery, and every diagnostic built here are exchange-agnostic and
carry forward to the next market. Only the parser is PSX-specific.

---

## Phase 1 — Data pipeline: from raw FIX capture to a queryable store

The raw input is the PSX FIX market-data feed (FIXT.1.1), captured daily as
gzipped message logs across all ~577 listed symbols. A custom parser
(`PSX_Parser_Mac.py`) decodes each day into **four Parquet tables**, laid out as
date-partitioned Hive directories (`<ROOT>/<table>/date=YYYY-MM-DD/*.parquet`)
and queried in place with DuckDB. Storage came in at ~43 GB/year — far below the
original 250–500 GB estimate, because book snapshots compress much harder than
expected.

**The four tables:**

**`trades`** — one row per match (aggressor fill against one resting order):
`symbol, transact_time, capture_ts, price, qty, initiator, aggressor_side,
buy_ref, sell_ref, resting_ref / resting_order_id, exec_type, appl_seq`.
A critical data property discovered late (and the source of one full rebuild of
the sweep analysis): **there is no aggressor order ID.** `initiator` is only a
three-value label (AUCTION / BUYER_INITIATED / SELLER_INITIATED), and the
aggressor-side ref column is zero on aggressor rows — only the *resting* order is
identified. A single market order sweeping several levels therefore appears as
several rows with no key linking them; any per-order reconstruction must be
heuristic.

**`ob_updates`** — order-level book events: `symbol, transact_time, capture_ts,
order_id, side, price, qty, event (add/modify/cancel), appl_seq`. This table is
what makes an honest fill model possible: it lets the backtester track the FIFO
queue at each price level, order by order.

**`ob_snapshot`** — the disseminated book state, one message (`msg_seq`) at a
time: `symbol, msg_seq, orig_time, capture_ts, entry_type, level, px, qty, phase,
order_ids, order_qtys`. PSX disseminates ten explicit levels per side
(`entry_type` BID/OFFER, `level` 1–10); everything deeper is aggregated into a
single **AGG_BID / AGG_OFFER** block with no per-level structure. Each message
also carries the trading `phase`. The parser writes snapshots sorted by
`(symbol, msg_seq, entry_type, level)` — a sort order that later became a
performance lever.

**`misc`** — everything that is neither book nor trade: session status,
closing/settlement price broadcasts, statistics (PE-ratio messages),
circuit-breaker and instrument-status notices.

**Session structure (learned the hard way, see Phase 7):** the continuous
auction runs 09:32–15:29:59 (~358 minutes) Monday–Thursday. Fridays run a
**split session** totalling ~433 minutes. Ramadan compresses the day to ~253
minutes (Mon–Thu) and ~193 (Friday), and one-off short days exist. After the
continuous close there is a **Post-Close Session** (15:35–15:50, trades only at
the official closing price), a Trade Rectification window, and a Negotiated
Deals Market — none of which are order-book trading, and all of which must be
excluded from microstructure analysis. Phases observed in the data: STARTING,
OPEN_CALL_AUCTION, TRADING_BREAK, CONTINUOUS_AUCTION, AFTER_HOUR_TRADING,
MARKET_CLOSED.

---

## Phase 2 — The backtest engine: two clocks, seeded latency, queue-gated fills

The engine (`mm_backtest.py`) is event-driven: updates, trades, and snapshot
messages are merged into a single stream and replayed in exchange order, sorted
by `(ts_exch, kind_rank, appl_seq)` so snapshots at a timestamp precede the
orders and trades that follow them, and ties break on the exchange's own
application sequence.

**The two clocks.** Every message carries two timestamps, and the engine keeps
them strictly separate. **`ts_exch`** is the exchange clock — `transact_time`
for updates and trades, `orig_time` for snapshots — and defines *when things
actually happened at the exchange*: event ordering, fill times, session bounds
all live on this clock. **`ts_cap`** is the capture clock — when our own
recorder saw the message — and defines *knowledge time*: the earliest instant
the strategy could have known about an event. The event replay runs on exchange
time, but the strategy's decisions are constrained by knowledge time, so the
backtest can never act on information before it would have arrived. This
two-clock discipline is what keeps the simulation honest about latency without
pretending the feed is instantaneous.

**The seeded latency model.** Order entry is not free: a `LatencyModel` draws
stochastic submission/acknowledgement delays for every order action. It is
constructed with a fixed seed (`LatencyModel(seed=R.LATENCY_SEED)`), so the
random latency draws are **reproducible** — two runs of the same configuration
produce identical fills to the byte. That determinism is what made the project's
verification discipline possible: after every refactor (comment insertion,
optimization, restructure), outputs were diffed against the pre-change run and
required to be byte-identical before the change was accepted.

**Queue-gated fills — the correctness-critical piece.** The first, naive fill
model ("our quote fills whenever a trade prints at our price") was discarded as
dangerously optimistic. The production model tracks our order's **FIFO queue
position** at its price level, built from the order-by-order `ob_updates`
stream: we fill only when the queue ahead of us has been consumed by trades or
cancels. Post-only semantics are enforced. Every subsequent result in the
project rests on this fill model; the standing rule became *results cannot be
trusted without the queue-position engine*.

**Halts and price bands.** PSX has three distinct halt mechanisms, all modelled
in a shared `halt_state.py` used by both backtest and (would-be) live code: a
market-wide halt via the KSE-30 index circuit breaker (±5% / ±7.5% tiers),
per-stock temporary suspensions, and per-scrip ±10% price locks. Quoting is
suppressed and orders are handled correctly through each.

**Honest end-of-day valuation.** The engine reports both `equity_mid_mark`
(position marked at the closing mid) and `equity_liquidated` (what you would
actually realize by walking the position out through the real book), along with
`unfilled_sh` — the residual the book could not absorb. The gap between the two
is the liquidation haircut, and refusing to pretend it away shaped several later
findings.

**The profiling lesson.** The engine was originally slow, and three plausible
hypotheses (equity logging, a build-once restructure, fill-context joins) were
each investigated and ruled out before cProfile identified `Book.snapshot()` as
**89% of runtime**. The fix — `snapshot_prep.py`, which pre-parses every
snapshot message once into a plain-Python `PreparsedSnapshot` so the hot path
does zero pandas work — cut per-day build time ~3× (9.3s → ~3s). The lesson
(*measure before diagnosing*) had to be relearned twice more later in the
project.

---

## Phase 3 — The strategy and its calibration

The strategy (`micro_mm.py`, `MicrostructureMM`) is a passive two-sided quoter
with Avellaneda–Stoikov-style inventory skew. Its parameters are calibrated
per symbol, walk-forward only (calibration windows strictly precede trading
days — nothing in-sample):

**Inventory skew and `session_scale`.** The A-S skew needs a scale linking
inventory to quote displacement. Rather than hand-tuning, `session_scale` is
**back-solved per symbol** so that the skew at maximum inventory equals half the
median spread — i.e., a full-inventory book leans its quotes by exactly one
half-spread (PPL ≈ 7.6, UBL ≈ 3.9). Risk aversion gamma is 0.15, with a soft
inventory band at 3 clips and a hard cap at 10.

**Clip sizing from median trade size.** The quote size ("clip") is sized as a
multiple of the symbol's **trailing 10-day median trade size** — a walk-forward
anchor that automatically adapts to each name's liquidity. A capacity sweep
found **3× median** to be the knee: at 5× the incremental P&L was small and the
overnight-inventory risk of the larger unfillable position did not justify it.
3× was locked as the production clip. (Futures traded in fixed 500-share lots,
so futures clips are in lots.)

**Time-of-day structure.** Each symbol-day is calibrated into a four-bucket
volume profile — first-15-minutes, middle, pre-close-45, last-15 — as median
shares-per-minute in each bucket. This profile drives the **POV (participation
of volume) end-of-day unwind**: as the close approaches, the strategy works its
inventory off through a ramp sized to the bucket's real volume rate, capped at
10% participation, so the exit model never assumes liquidity the tape doesn't
show. The same four buckets later became the attribution buckets, revealing
where in the day the edge lives.

**Features, in three distinct roles.** Signal research settled on a clean
division: **directional skew** (order-book imbalance, OBI, is the primary
signal; a micro-price deviation feature was dropped as redundant with OBI),
**gating / when to pull** (toxicity/VPIN-style measures), and **sizing/width**.
A per-symbol feature store (`build_feature_store.py`: mid, spread_bps, obi_1,
toxicity, realized_vol_bps per event) supports the signal lenses. A 2D sweep of
the quoting parameters located `min_edge_pct = 0.0005, improve_ticks = 0.0` as
the P&L-maximizing capture configuration.

---

## Phase 4 — Spot market-making: findings and results

**The KTML lesson: decompose direction from market-making.** An early KTML
backtest showed +724 PKR profit — which decomposition revealed to be **+7,646
of accidental directional P&L** (the strategy happened to be net short in a
falling session) **minus 6,922 of actual market-making losses**. From then on,
directional P&L was always separated from MM P&L before believing any result.

**The core early finding: the losses were inventory carry, not bad signals.**
Mid-marked P&L was roughly flat while liquidated P&L was deeply negative, and
the strategy ended short on 62–69% of days. The quoting itself was fine; the
inventory management wasn't. The A-S skew calibration above was the fix.

**Fees invert the symbol ranking.** At retail fees only KTML showed positive net
passive edge; at market-maker-programme fees FFC led. Symbol selection is
fee-regime-dependent, so all production analysis was run at the TREC own-account
fee: spot 0.777 bps/side (~1.55 bps round-trip), futures 0.0938 bps/side.

**A realized-volatility gate was tested and did not work.** The hypothesis was
that pulling or widening quotes when short-horizon realized volatility spiked
would avoid the worst fills. In testing it did not produce a net improvement
that justified deployment — the fills it avoided did not cost more than the
fills it forfeited — and it was rejected. (A second gate experiment, on deep
book sweeps, failed later for a subtler reason; see Phase 7.)

**The final spot result** (top-10 book: ENGROH, LUCK, UBL, PSO, PPL, HBL,
SAZEW, MLCF, ATRL, SYS; 3× clips; 1,965 symbol-days over ~207 trading days):
FIFO open-bucket attribution with **reconciliation exact to 0.0000 PKR on every
symbol-day**. The middle-of-session bucket is the engine of the book: **+6.36
bps on opened notional (+1.57M PKR, 158k fills)**, with the first 15 minutes at
+3.44 bps and pre-close-45 at +3.42 bps; only the last 15 minutes lose (−1.18
bps, small). Run-level: **Sharpe 5.67 annualized, Sortino 5.07**, mean ~969
PKR per symbol-day, run max drawdown −28k, worst single intraday drawdown −60k.
Execution profile: order-to-trade ratio 17.6, cancel-to-trade 7.3, fill ratio
5.7% — normal for passive quoting and inside exchange messaging norms.

The strategy is real. The problem is scale: ~250k PKR/month total. The edge is
per-share and the fees are per-share, so **trading bigger clips cannot improve
the margin** — the 5× test had already shown the capacity knee. High Sharpe on
thin absolute P&L, against a Rs 35M net-worth lock-up plus a brokerage entity,
does not clear the hurdle.

---

## Phase 5 — Futures market-making: three strategies, one verdict

PSX single-stock futures offered a tempting structure: fees ~8× lower than spot
(0.0938 bps/side) and spreads ~3× wider (~14 bps vs ~5). Contracts were mapped
to a **causal roll map** (the active contract per root per date, chosen on
trailing volume — no lookahead), and the research ran three strategy variants
against it.

**Variant 1 — carry with an EOD spot hedge.** Hold futures inventory across
days, delta-hedging the residual each close by **walking the real spot book**:
the `walk_spot_book` primitive consumes spot levels best-first for the required
quantity and returns filled size, VWAP, and slippage-versus-mid; slippage plus
fee is booked as `hedge_cost`. P&L was decomposed six ways — quoting, drift,
basis, hedge_cost, roll_cost, naked_gap — after roughly seven accounting bugs
were found and fixed. The final accounting pattern is worth recording:
**make the least-important component (`naked_gap`) the reconciling residual**,
`naked_gap = total_cash − (sum of the five measured components)`, so
reconciliation is exact *by construction* and the five components that matter
are all directly measured. Result (26-day October span, recon 0.00): the
quoting was roughly flat; **drift — the uncompensated directional move on
inventory carried between days — was the loss driver** (MLCF: −14.2k of −14.7k
total). Month-end roll cost was confirmed at zero. An inventory-cap sweep
(10/5/3 clips) trimmed losses ~11% — a second-order lever, because the cap can
reduce drift but cannot touch fill quality.

**Variant 2 — same-day flatten.** If illiquidity forces carry and carry causes
drift, force the position flat into every close via the production POV unwind,
and spot-hedge only whatever residual the book cannot absorb. Mechanically it
**worked**: flatten rate ~1.00 on BOP/MLCF/TRG, drift collapsed to zero, and
the flatten haircut — the direct, measured cost of exiting into the futures
book — was small (~40–55 PKR/day). The liquidity half of the thesis was
confirmed: liquid futures *can* be closed same-day, killing drift cheaply.

**But the fills themselves are toxic.** With drift and haircut controlled, the
capture/markout lens delivered the verdict across **ten months and all three of
the most liquid names**: capture +815,970 vs **markout −978,809 — adverse
selection is ~120% of the spread captured.** October was checked as a possible
outlier and wasn't: net (capture+markout) is negative in the large majority of
months on every name. A per-side split (who hits our bid vs lifts our ask, and
each side's markout) was added to check whether one side carried the toxicity;
the flow was roughly balanced.

**Variant 3 — basis arbitrage** was scoped (tickers with futures → basis;
without → MM), including the observation that running MM and basis on the same
symbol requires order-level self-trade prevention — but it was never reached,
because the fill toxicity killed the MM leg first.

**Verdict: futures MM is dead on PSX — not because of illiquidity (solvable, as
same-day flatten proved) but because passive futures fills are structurally
adversely selected.** The one methodological objection left open is the markout
horizon (next section); it was judged unable to flip a 120%-of-capture result
and futures was closed without running it.

---

## Phase 6 — Measuring adverse selection correctly: the markout evolution

The markout definition was corrected twice during the project, both times at
SZ's insistence, and both corrections changed conclusions. The final framework:

**Capture** = `signed(mid_at_fill − fill_px) × qty` — the half-spread earned by
being filled passively (buys fill below mid, sells above).
**Markout** = `signed(close_mid − mid_at_fill) × qty` — the adverse move after
the fill. Both are measured against the **mid**, sharing `mid_at_fill` as the
hinge, so they sum exactly to the fill's mark-to-close P&L with no overlap and
no gap.

**Correction 1: 5 seconds → same-day close.** The original 5-second markout
horizon mismeasures adverse selection when positions are held for minutes to
hours — drift scales with holding time, and a 5s window hides most of it.
Re-measured to the same-day close, the futures adverse selection roughly
doubled and the "quoting is fine" read on the carry run collapsed.

**Correction 2: equity-difference → from-fill quoting.** The decomposition's
quoting component was originally an open-of-day equity difference minus a drift
adjustment. It was refactored to the direct **from-fill** definition:
`quoting = capture + markout − fee`. Because `mid_at_fill` cancels between
capture and markout, `capture + markout = signed(close − fill_px) × qty`, which
on a flat-start/flat-end day **equals realized cash exactly** — so the
reconciliation plug absorbs nothing on clean days and the two independent views
agree to the rupee by construction rather than approximately.

**The open refinement: horizon = actual holding time.** Even same-day-close is
an arbitrary horizon: the honest per-fill markout runs from the fill to **the
moment that fill was actually offset**. The FIFO round-trip matcher (which
already produces holding-time statistics — futures median holds ~24–42 minutes)
can supply the matched exit; a cruder proxy is a fixed horizon equal to the
median hold. Marking to the close *overstates* the adverse move on positions
already flattened earlier, so the production-correct measure is
**FIFO-matched-exit markout**. It was specified but deliberately not run:
shortening the horizon could only narrow, not plausibly close, a markout gap of
120% of capture, and the futures verdict did not depend on it.

---

## Phase 7 — Sweep-depth, the L10+ gate that failed, and iceberg inference

With futures MM dead, one hypothesis remained attractive: *futures flow is more
institutional/aggressive than spot, and that's why the fills are toxic.* A
sweep-depth study was built to test it: for every aggressor order, how many
book levels did it consume (L1…L10, and **L10+** for sweeps that exhausted all
ten disseminated levels and reached the AGG block)?

**Reconstructing sweeps without an aggressor ID.** The discovery that the data
carries no aggressor order key (Phase 1) forced a heuristic: **same-timestamp +
same-aggressor-side clustering** — trades sharing one exact exchange timestamp
on one side are treated as one marketable order filling multiple resting
orders. This is a conservative *lower bound* on depth (a sweep the exchange
stamps across several instants gets split). Sanity statistics validated it:
4.97M clusters over 207 days, mean 1.79 fills per cluster, median 1, p95 = 4 —
the signature of real sweeps, after a first attempt keyed on `initiator` had
absurdly collapsed ~2,000 trades per "order" and was caught by exactly this
check. Three measures were computed: **M1** (distinct trade prices per cluster
— simplified), **M2** (book-diff: swept quantity walked against the pre-trade
L10 resting depth — authoritative), and **dist** (per-fill distance from the
touch, which needs no clustering at all).

**Performance: 97 minutes → 8 seconds.** The first implementation rescanned the
full snapshot table per cluster (O(clusters × snapshot)) and took 97 minutes
for five days. The fix built a **per-symbol-day snapshot index** once (each
message's best-first book pre-extracted; timestamps converted once) with an
O(log n) `searchsorted` lookup per cluster — and, exploiting the parser's
guaranteed sort order, built that index by slicing contiguous numpy blocks with
no groupby, guarded by a contiguity assertion. Two days then ran in 8 seconds;
the full 207 days in ~12 minutes. Fast and slow paths were proven identical at
every probe time before the fast one was trusted.

**The hypothesis was refuted.** On every name and every measure, spot sweeps
are as deep or deeper than futures (M2 L3+ volume share — BOP 54.4% spot vs
50.3% futures; MLCF 60.9 vs 50.1; TRG 61.7 vs 58.4), and 84–93% of *all* fills
print away from the touch in both markets. PSX is simply a deep-sweep market
throughout. Futures toxicity is **not** explained by bigger or more aggressive
orders — pointing instead at microstructure (tick size relative to volatility,
quote staleness against a fast-repricing book, thin two-sided flow). Two
side-findings: PPL and UBL have no futures data at all across the full history,
and the whole analysis had to learn PSX's **session-structure lessons**: an
initial "post-close leak" diagnosis was wrong twice (the trades were clean;
diagnostics proved it) — the apparent tail past the close was the pooling of
structurally different day-types onto one axis. The final version classifies
each session by length (Normal Mon–Thu ~358 min, Friday split ~433, short days
~253/~193), plots on **fraction-of-session** so all day-types align, buckets
all statistics by day-type, and excludes the Post-Close Session and NDM prints
(fixed-price and negotiated trades cannot sweep a book and are not
microstructure events).

**The L10+ dark-gate — tested before building, and correctly not built.** The
proposal: after a sweep reaches L10+, pull quotes ("go dark") for 5/15/30
seconds to dodge follow-on toxic flow. Instead of building the gate, the
justifying diagnostic was run first: time-of-day concentration of L10+ sweeps
(a clean U-shape — clustered at the open and heavily into the close, on every
day-type), inter-arrival times (spot median 41.3s between L10+ sweeps, futures
87.7s), and — decisively — **conditional markout** after deep sweeps versus
shallow ones. The spot result was scientifically tidy: signed post-sweep
markout is *monotonic in sweep depth* (L10+ > L3-L9 > L1-L2, positive, growing
from 5s to 30s) — deep sweeps genuinely predict continuation. But the
magnitude at 30s is ~0.004 price units, **a few thousandths of a basis point**
against a ~6 bps spread edge, with futures sign-flipping in a thin sample. A
gate that goes dark 15–30s after events arriving every ~41s would forfeit a
large fraction of fills to dodge an adverse drift smaller than a tick.
**Verdict: real effect, untradeable magnitude — the gate was not built.** If
anything the honest form is a tiny graduated width-skew after deeper sweeps —
noted for the next exchange, where the same clean gradient might exist at
tradeable size.

**Iceberg inference (signature, not detection).** With no order IDs, hidden
orders can only be inferred. The fingerprint used: **M2 ≫ M1** — a sweep whose
quantity should have consumed many visible levels (M2 large) but printed at one
or two prices (M1 ≤ 2) implies hidden replenishment at those prices, i.e. an
iceberg refilling. Flagged at `M1 ≤ 2 & M2 ≥ 4`, roughly **5–10% of sweeps**
show the signature (TRG spot highest at ~10%), explicitly labelled inference
since independent reposting can mimic it. The legitimate ways a professional
desk designs *around* an inferred iceberg were documented — lean on it as
support/resistance, skew quotes asymmetrically against it, queue behind it,
fade the level when the refills stop — with the bright line drawn at pinging,
spoofing, or any order activity intended to probe or manipulate it, which is
market manipulation. For this book, the natural (unbuilt) use was feeding the
refill signal into the existing OBI quote-skew machinery.

---

## Phase 8 — Finalizing the base files

Three cleanups closed the codebase:

**Attribution made exact.** The FIFO open-bucket attribution had a residual
reconciliation error (max ±5,989 PKR on 5 tail days out of 1,965, caused by the
unclean-liquidation haircut). `fifo_attribution` was fixed with the same
residual-plug pattern as the futures decomposition (residual = engine P&L −
matched realized, attributed to opening buckets), and the full panel re-run:
**reconciliation now 0.0000 PKR mean and max across all 1,965 symbol-days**,
with the headline economics unchanged (Sharpe 5.67, +6.36 bps middle) — exactly
what a pure accounting fix should do.

**Lifecycle instrumentation stitched through.** The engine's per-order audit
log (`order_log`: live time, end time, end reason per order) was wired into
`DayResult` and `run_symbol_day` as an optional, backward-compatible attribute,
so the institutional execution panel — quote uptime, time-to-fill, peak message
rate, OTR/cancel ratios — is available end-to-end if the stack ever goes to
production.

**The attribution run parallelized: 152 minutes → ~5.** The full-panel re-run
took 152 minutes serially. Profiling-honest analysis showed the engine's old
snapshot bottleneck was already fixed, and the remaining cost was simply many
independent symbol-days — `run_symbol_day` is self-contained (own reads, seeded
latency, no shared state), i.e. embarrassingly parallel. A multiprocessing
consumer (`analyze_bucket_attribution_fast.py`) runs symbol-days across all
cores; aggregation was proven order-independent so results match the serial run
byte-for-byte. A first version recomputed the trailing-median pre-pass inside
every worker (10× redundant — caught by the timing breakdown, the same
measure-first lesson again); moving it to the parent cut the pre-pass from
1m31s to 13s. Five days: 46 seconds, identical numbers.

---

## Conclusions and the decision

**Futures MM: dead**, proven three ways (carry+hedge, cap sweep, same-day
flatten) across ten months and the most liquid names, with the escape hatches —
bad month, bad name, illiquidity, institutional flow — each tested and closed.
The cause is adverse selection in the fills themselves (markout ≈ 120% of
capture), which no inventory policy, hedge, or flatten schedule can fix.

**Spot MM: real but sub-threshold.** Sharpe 5.67 on exact accounting is a
genuine edge, but ~250k PKR/month against a Rs 35M TREC net-worth lock-up plus
a brokerage entity is not a business, and because both the edge and the fees
are per-share, **scale cannot fix a per-share margin problem** (the 3× vs 5×
clip test settled that empirically). The TREC license is not justified.

**What carries forward.** Everything except `PSX_Parser_Mac.py` is
exchange-agnostic: the two-clock event engine with seeded latency and
queue-gated fills, the walk-forward calibration stack, the book-walking
execution primitives, the residual-plug decomposition pattern, the FIFO
attribution with exact reconciliation, the sweep/toxicity/iceberg diagnostics,
and the parallel runner. Pointed at a new exchange's data in the same
four-table shape, the whole stack tests for alpha immediately.

**Standing lessons, earned repeatedly:** measure before diagnosing (three wrong
engine-bottleneck guesses; two wrong session-tail diagnoses; one redundant
pre-pass — every one caught by profiling or a purpose-built diagnostic, never
by theory); verify refactors by byte-identical diff under the seeded latency
model; make the reconciliation residual explicit and structural rather than
hoping components sum; treat the fill model as correctness-critical; decompose
directional P&L from market-making P&L before believing any number; and check a
new exchange's session/phase structure — auction windows, split days, shortened
calendars, post-close mechanisms — before trusting any time-aggregated
analysis.
