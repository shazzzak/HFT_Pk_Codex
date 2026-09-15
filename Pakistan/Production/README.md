# PSX market-making engine — production

Written from scratch. Nothing here is carried over from the research tree.

## Layout

```
core/       venue-agnostic. Reusable unchanged by a second market (IDX).
  model.py    Side, OrderState, Order, Fill, QuoteIntent, DesiredQuotes, Actions
  venue.py    the Venue interface — the ONLY seam between shared and per-market
  risk.py     KillSwitch, the pre-trade controls, RiskGateway
venues/     one file per market
  psx.py      PSXVenue: flat tick, TREC fee, Friday session break, algo tag
tests/
  test_risk.py
docs/
  PSX_LIVE_ENGINE_SCOPE_20260915_1900.md
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

A backtest that does not simulate queue position cannot see this cost, and ours
does not yet. So the check is a **health metric first, a guard second** — a high
reading is a reason to look at the quoting cadence, not to tighten the limit.

## What is not built

The order manager, the FIX session layer, the audit log sink, the control plane,
and the market-data handler. Build order and gates are in `docs/`.

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

22 tests, all passing. Every one asserts a rejection path: a control that never
rejects in a test has never been shown to work.

## Next

`docs/PSX_LIVE_ENGINE_SCOPE_20260915_1900.md` §7 lists what is needed from
outside — the order-entry spec above all, which is the long-lead item and is not
an engineering task.
