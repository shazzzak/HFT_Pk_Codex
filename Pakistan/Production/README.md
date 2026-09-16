# PSX market-making engine — production

Written from scratch. Nothing here is carried over from the research tree.

## Layout

```
core/       venue-agnostic. Reusable unchanged by a second market (IDX).
  model.py    Side, OrderState, Order, Fill, QuoteIntent, DesiredQuotes,
              Actions, BookSnapshot, Trade
  venue.py    the Venue interface — the ONLY seam between shared and per-market
  strategy.py the Strategy interface — on_book in, DesiredQuotes out
  risk.py     KillSwitch, the pre-trade controls, RiskGateway
  oms.py      OrderManager — the diff, the order lifecycle, position from fills
  audit.py    AuditLog — Parquet for everything, fsynced JSONL for the crash
venues/     one file per market
  psx.py          PSXVenue: flat tick, TREC fee, Friday session break, algo tag
  psx_strategy.py MicroMMAdapter — wraps micro_mm unchanged, converts units
  psx_config.py   live_config.py moved in: three buckets, two files, preflight
sim/        the replay harness. Imports mm_backtest; nothing else may.
  replay.py   EngineReplay: mm_backtest's exchange, the production _requote
tests/
  test_risk.py
  test_oms.py
  test_audit.py
  test_strategy.py
  test_replay.py
  data/           verbatim copies of the real config files, for the loader tests
docs/
  PSX_DOC_AUDIT_20260916.md   <- read this before trusting the other three
  PSX_LIVE_ENGINE_SCOPE_20260915_1900.md
  PSX_FIX_COMPLIANCE_20260916.md
  PSX_SPEC_VERSIONS_20260916.md
```

Nothing in `core/` imports from `venues/`. That direction is the whole point of
the split — if it ever reverses, the reuse is gone.

## Two rules the design rests on

**1. Prices are integers.** Every price is a whole number of minor currency
units — paisa for PKR. Floats make `px == bb + tick` fail at random and make a
tick grid untestable. `Venue.minor_per_major` is the only place that knows how
to turn one back into a human-readable number.

**2. Nothing reaches the wire without passing `RiskGateway`.** No debug path, no
manual override. If an order can go around it, the kill switch is a suggestion
rather than a switch.

And its corollary, enforced centrally in the gateway rather than trusted to each
control: **no check may ever block a cancel.** Every limit here restricts what we
*add*. A control that can block a cancel is a control that can trap us in a
position — a worse failure than the one it prevents.

## What is built

Phase 1 of the plan in `docs/`: the domain model, the venue seam, and the risk
gateway with the kill switch. Pure logic, no I/O, no connectivity.

**None of this is compliance.** The SECP concept paper of 30-05-2025 is not being
pursued. Every control below is here because it protects capital; the paper's
section number is recorded only so that a revived framework is a filing exercise
rather than a build.

| Control | Why it exists | § if revived |
|---|---|---|
| `KillSwitchCheck` | stop everything when something is wrong and we don't yet know what | 11 |
| `PriceBandCheck` | catches a plausible price computed from a stale book | 8.1 |
| `OrderValueCheck` | bounds the cost of one bad number | 8.2 |
| `OrderQuantityCheck` | same, in shares | 8.3 |
| `PositionLimitCheck` | bounds inventory on the *worst* case, not the current one | — |
| `MessageRateCheck` | protects the session from an oscillating strategy | 7, 8.4 |
| `TradingWindowCheck` | orders outside continuous trading behave in ways the backtest never modelled | 7 |
| `OrderToTradeRatioCheck` | **queue position** — see below | 6 |

Every decision, approval as well as rejection, goes to the `on_decision` sink.

### The order-to-trade check is not what it looks like

With no regulator penalising churn, the cost is paid to the market instead:
**every cancel-and-repost surrenders queue priority and rejoins at the back.**
For a strategy whose edge is spread *capture* — getting filled while resting —
that is close to the whole game. `micro_mm` ships with `tol_ticks = 0.0`, which
means no hysteresis at all: any recomputation that moves the desired price by one
paisa cancels and reposts.

**CORRECTED 2026-09-16.** An earlier version of this file said the backtest
cannot see that cost. That was wrong, and it was asserted without reading
`mm_backtest.py`. The backtest models queue position **exactly**: `MyOrder.ahead`
is an order-id → qty dict of everything resting at our price when we arrived,
rebuilt from order-level data and maintained event by event — market cancels
remove entries, trades drain the pool, and a trade row that names its resting
victim (`rest_oid`) drains exactly that entry. Snapshots reset it conservatively,
assuming everything now at our price is ahead of us. On top of that sits a
two-leg stochastic latency model in which an order stays fillable until its
cancel actually lands.

So the queue cost of `tol_ticks = 0.0` **is already priced into every measured
result**. The check is still a health metric first — because no regulator sets a
ratio and we have not been told the broker's message limit — but not because
anything is blind to the cost.

### The order manager

The strategy says what it wants resting. The order manager works out what
messages make that true, given what is actually resting. Three things fall out
of that split:

**Quote churn is controlled in one place** — the diff, via `QuoteTolerance`.
Defaults reproduce today's behaviour exactly (`price_ticks=0`, requote on any
change, matching `tol_ticks=0.0`). Raise it deliberately and measure what it
costs; don't inherit it by accident.

**The kill switch has one job.** It sets the desired state to nothing and the
ordinary diff emits the cancels. A kill switch with its own cancel-everything
path is code never exercised until the day it matters. The switch is also
re-checked at the point of emission, so a desire set after the trip is
overridden anyway.

**Nothing acts on an order in flight.** Not to cancel it, not to replace it, not
to send another. An order manager that forgets this double-sends under latency —
twice the intended size resting, from a bug that only shows when the exchange is
slow, which is when the market is moving. Three tests cover it.

**A price change is a cancel, then a place on a later cycle** — `use_replace`
defaults to False. PSX accepts `Order Cancel/Replace Request` and amendment is
the better wire mechanic, with no window in which we have cancelled and are not
yet quoting. It is off by default anyway, because `mm_backtest._requote` sends a
cancel and a replacement as two independent messages and the old order stays
fillable until its cancel lands. Every measured result was produced under that,
including the fills taken in that window. Amendment is available behind the flag
and `sim/replay.py` raises if it is on, because `Backtester` has no amendment
path and a run using one has stopped being comparable.

And **cancels are emitted before places** within a cycle, so resting size never
briefly doubles.

The in-flight rule turned out to be mandatory rather than merely prudent: PSX
requires `OrderID` (tag 37) on both a cancel and an amendment, and the exchange
assigns it on the acknowledgement. Before the ack there is nothing to send.

Position comes from fills. Never from what we believe we sent.

### The audit log — two files, because one cannot do both jobs

**`*.parquet`** — everything, zstd-compressed, columnar. Written in batches by a
background thread. This is the file you query.

**`*.critical.jsonl`** — the kill switch, rejects, state mismatches, session
start and end. Written **synchronously and fsynced on the hot path**. This is the
file you read after something went wrong.

Why both: Parquet is columnar and batched — a single row cannot be appended, so
it can't go on a hot path. But that means whatever is still buffered when the
process dies is gone — **and the crash is precisely the event the log exists to
explain.** A compressed record of a normal morning is of limited interest; the
five records before the crash are the whole point. Critical events are rare (a
few dozen a day), so paying ~122 µs to fsync each one is cheap insurance.

**What it costs the trading path**, measured with the writer running:

| | p50 | p99 |
|---|---|---|
| ordinary record | **302 ns** | 2.1 µs |
| critical record (fsync) | ~122 µs | — |

`queue.Queue` was the first design and it cost **17.5 µs** per record: every
`put` notifies a condition variable and forces a context switch on the
*producer's* thread. A plain `deque` with a polling consumer is 55 ns — 320×
cheaper — because `append` is atomic in CPython and touches no lock. Single
producer, single consumer, no lock needed. `test_audit.py` asserts p50 stays
under a microsecond, so reintroducing a lock fails the suite instead of being
found in production.

The buffer is **bounded by an explicit length check**. A full buffer drops and
counts; it never blocks the trading path and never grows until the process runs
out of memory. Dropped records are still visible — the sequence number is
assigned *before* the drop, so the gap shows in the file.

Sequence numbers make a truncated file visibly incomplete rather than quietly
short. A log that loses its tail without saying so is worse than no log, because
it gets trusted.

Approvals are recorded as well as rejections, with every individual control's
verdict, because "why was this order *sent*" is the question that matters and
"all checks passed" doesn't answer it. The fields worth filtering on — symbol,
side, price, quantity, check, allowed — are real columns, so a query never parses
JSON. Heterogeneous payloads go in one `extra` JSON column.

**pyarrow is required and the engine refuses to start without it.** The audit
log is not optional, and discovering at shutdown that it was never written means
discovering it after the session it was meant to record.

### The strategy adapter — a translator, nothing more

`micro_mm.py` is the strategy every backtest result was produced by, and the
engine runs **that code, unchanged**. The adapter only converts:

```
integer paisa   ->  float rupees      (what micro_mm expects)
micro_mm's dict ->  DesiredQuotes     (what the order manager reads)
```

The alternative — reimplementing the signal cleanly in integer arithmetic — was
considered and rejected for now. Every measured result is evidence about one
particular implementation; run a different one and the evidence stops
describing what is trading, and the difference surfaces as a P&L gap nobody can
attribute. A rewrite is safe only once a test runs both over the same recorded
day and asserts every quote matches tick for tick.

The cost of that choice is that floats live in the trading path. **They are
converted in exactly two functions, and the conversion is the highest-risk code
in the file.** `int(px * 100)` is wrong on **32,808 of the 499,900** paisa
prices between Rs 1 and Rs 5,000 — 6.6%, and 16% in the Rs 280–290 band where
PPL trades — because the nearest double to a two-decimal value sits just below
it. Rs 280.03 × 100 is 28002.999999999996. `int(round(px * 100))` is correct;
a test walks **every** price in the range, not a sample, because the failing
prices are not evenly spaced and any fixed stride can miss them.

**Version 1 is maker-only.** `QuoteIntent` describes resting orders and nothing
else, so micro_mm's age-cross feature — which deliberately crosses the spread to
flatten aged inventory — is refused at construction *and* re-checked on every
book, because the switch is a mutable attribute. An aggressive order sent as
though it were passive is the failure this prevents.

The adapter also refuses `tol_ticks > 0`: that and the order manager's
`QuoteTolerance` are the same mechanism, and both on means requotes are
suppressed twice at an effective tolerance that is neither setting. Hysteresis
lives in the order manager, where the cost it controls — lost queue position —
is actually paid.

**A new trading day means a new strategy object.** micro_mm carries a day of
state (the volatility EMA, the flow and OFI windows, the spread EMA, the session
boundaries, the volume profile, the trigger counters) and has no reset. Clearing
some of it and not the rest would look like a fresh strategy while carrying
yesterday's volatility into this morning's quote widths, so `on_session_start`
raises rather than pretending.

### The live config — `live_config.py`, moved in

This is **your `existing_mm_live/live_config.py`**, not a second implementation
of it. Paths are injected instead of imported from `config_pk` — the live engine
must not import anything that knows where the backtest keeps its results — and
three things changed, each marked `CHANGED 2026-09-16` in the source. Everything
else is that module's design, kept because it was already right: the two-file
split, the asymmetric failure policy, the torn-read protection, the orphan rule,
the DROP semantics and the preflight.

`existing_mm_live/live_config.py` should become a thin shim that imports this and
supplies the paths from `config_pk`, so there is one implementation. Two that
drift is how a name ends up quoting a setting nobody chose.

```
config_assignment_YYYYMMDD_HHMM.csv   machine-generated, never hand-edited
live_overrides.csv                    hand-edited, sparse, WINS
```

The shipped assignment is 113 names: **68** on the lean at 0.15, **13** at 0.20,
**17** lean off, **15** dropped. The lean is QT_2t — a fixed **two ticks**, not a
basis-point distance, because that is the version that won the sweep. On a flat
one-paisa grid those are very different things: two ticks is 0.69 bps on a
Rs 289 name and 32.6 bps on a Rs 6 one.

**The failure policy is asymmetric on purpose.** At startup a bad file is fatal —
there is no last-good config to fall back to, and starting on engine defaults
would quote 113 names on a config nobody chose. On reload a bad file is ignored,
the previous config stays live, and `consecutive_failures` records it. Halting
the book because someone saved a CSV mid-edit is the worse failure.

**Absence is never an instruction.** A symbol that stops appearing keeps its last
setting and is re-warned as an ORPHAN on *every* reload — that is exactly the
state nobody notices. Stopping a name means writing DROP against it.

**`DROP` is no-add.** `soft_inv=0` stops the side that adds and keeps the side
that reduces, so inventory works off passively — micro_mm's existing band, a path
that has been in every backtest. A dropped name already flat is skipped entirely
(`skip_if_flat`), because with nothing to work off the reducing side would only
build the inventory the drop was meant to stop.

Three changes from the original:

**1. The engine numbers are read from the file and cross-checked, not
re-derived.** `live_config.LABELS` carried its own copy of (ticks, threshold)
with a comment that it "must stay identical to build_config_assignment.py's
PARAMS" — a hand-maintained duplicate of numbers that already travel in the CSV
as `skew_ticks` and `skew_thresh`. If a refit moved the lean to three ticks, the
CSV would say 3.0, the table would still say 2.0, and **the engine would quote
the old lean with no error anywhere.** Now both are read and compared, and a
disagreement refuses to start: neither source is trusted over the other, because
there is no way to tell which is stale.

**2. A pre-2026-09-15 two-bucket file is named in the error.** Those files were
already rejected, but by a message about an unknown setting — which does not tell
you that you pointed the engine at a superseded assignment decided by a different
rule (`t`, not the effect size `d`).

**3. Canonical labels resolve case-insensitively too.** `ALIASES.get(label.upper())`
handled every alias but not `OBI` and `DROP`, which are not *in* the alias table —
so an operator typing `drop` in lower case rejected the whole overrides file.
Those are the spellings the assignment file itself uses, so they are the likeliest
thing to be copied across, and it failed at 14:00 on the one edit that matters.

**The threshold when the lean is off is 0.15, not 0.0.** That is the original
module's call and it is right. The generator writes a blank cell, NaN must never
reach the engine (every comparison against NaN is False, so it works until
someone writes `<=` instead of `>`), and 0.0 is the *most*-firing value a
threshold can take — if the `queue_skew_ticks != 0` gate were ever removed, 0.0
fires on every bid-heavy book. micro_mm's own constructor default is inert while
ticks are zero and conservative if it ever stops being inert.

Preflight before the bell:

```
python -m venues.psx_config <config_assignment_*.csv> [live_overrides.csv]
```

Exit 0 means safe to start; 1 means do not.

### The replay harness — one variable, not two

Before the engine trades it has to produce the same orders and the same P&L as
the backtest every measured result came from. That test only means anything if
the **fill model is identical on both sides** — otherwise a P&L gap could be the
adapter, the order manager or the simulated exchange, and there is no way to
tell which.

So `sim/replay.py` does not simulate an exchange. `EngineReplay` subclasses
`mm_backtest.Backtester` and reuses its exchange wholesale — `Book`, `MyOrder`,
the `ahead`-dict queue, `LatencyModel`, `_arrive`, `_on_market_trade`, `_fill`,
the main loop — and replaces exactly one method: **`_requote`**.

**That one method is an order manager.** It reads the book, asks the strategy
what it wants, diffs against what is working, suppresses no-change requotes,
cancels the incumbent, sends the replacement, draws a latency per message and
gates on the trading phase. That is `core/oms.py`'s entire job. So the harness is
a controlled experiment with one variable: same market, same fills, same latency
draws, different order manager.

Three things the harness has to keep in step, and each has a test:

**Two id schemes.** Backtester identifies our orders by an integer `oid`; the
engine by a client order id string. The harness maps both ways, and the *cancel's
own* ClOrdID separately — PSX requires a new one on a cancel and the exchange's
reply quotes that, not the order's.

**Two lifecycles.** `_arrive` becomes an ack or a post-only reject. `_fill`
becomes an `on_fill` for the amount that actually executed, at *our* limit price.
A landed cancel becomes `on_cancelled` — and inside `_activate_until` an order
leaves `self.work` for exactly one reason, because fills happen on a different
path, so a disappearance there is unambiguously a cancellation.

**Backtester's halt gate**, translated into a `SecurityPhase` the adapter
understands, so the pull happens through the ordinary diff rather than a special
path that only the harness exercises.

**An amendment stops the run.** Backtester has no amendment path, so a
`ReplaceOrder` means `use_replace` was turned on and the run has silently stopped
being comparable. The harness raises rather than dropping the message.

`sim/` is the only place in this tree that imports `mm_backtest`. If that import
ever appears in `core/` or `venues/`, the split is gone.

## What is not built

The FIX session layer, the market-data handler, the strategy adapter, the
simulated exchange, and the control plane. Build order and gates are in `docs/`.

**Two numbers are deliberately `None` in `psx.py`:** the orders-per-second cap and
the order-to-trade threshold. No regulator is setting either, but the broker or
exchange may enforce a session message limit — we have not been told what it is.
`None` means *not told*, so the gateway falls back to a house limit rather than
treating silence as permission to send at an unbounded rate. A guessed default
would be indistinguishable from a measured one the moment it is in the code.

## Tests

```
cd Production
python -m pytest tests/ -q
```

126 tests, all passing. Every one asserts a rejection path or a failure mode: a
control that never rejects in a test has never been shown to work.

Thirteen of them are the harness, and they run against the REAL `Backtester` —
a harness tested against a mock of the thing it exists to integrate with proves
nothing about the integration. They need `mm_backtest` importable, and skip
cleanly when it is not:

```
PYTHONPATH=/path/to/existing_mm_live python -m pytest -q
```

## Next

`docs/PSX_LIVE_ENGINE_SCOPE_20260915_1900.md` §7 lists what is needed from
outside — the order-entry spec above all, which is the long-lead item and is not
an engineering task.
