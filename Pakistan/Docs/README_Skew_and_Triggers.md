# The Skew & Trigger Mechanism — Inventory, Time-of-Day, and Distance-to-Lock

This document explains, in detail, how the PSX market-making strategy
(`micro_mm.py`, `MicrostructureMM`) decides *where* to place its two-sided
quotes. It covers three distinct mechanisms that move or suppress the quotes:

1. **Inventory skew** — how the position shifts the quote pair (continuous,
   always on).
2. **Time-to-close (EOD) trigger** — how proximity to the closing bell widens or
   pulls quotes (a ramp then a cliff).
3. **Distance-to-lock trigger** — how proximity to the exchange circuit-breaker
   band widens or pulls quotes on the trapped side (a ramp then a cliff).

The guiding principle throughout, from Ho-Stoll and Avellaneda-Stoikov:
**inventory and urgency shift the *placement* of the quotes (skew); risk and
adverse selection set the *width* (half-spread).** Skew moves both quotes the
same direction; width moves them apart. Keeping these two separate is what makes
the behaviour interpretable.

---

## Part 0 — The anatomy of a quote

Every quoting cycle, `quotes(bb, bq, ba, aq, pos)` receives the best bid/ask and
their sizes, plus the current position, and produces a desired `{BUY: (px, size),
SELL: (px, size)}`. It is built in this order:

1. A **fair value** is computed (mid, or microprice if enabled).
2. **Inventory skew** shifts a **reservation price** away from fair.
3. A **half-spread** is built from risk + adverse-selection + a cost floor.
4. The bid is placed at `reservation − half`, the ask at `reservation + half`.
5. **Triggers** (EOD, lock) may then widen one side, kill one side, or force an
   exit lean — evaluated last, and able to override the economics.

The rest of this document walks each of the three mechanisms.

---

## Part 1 — Inventory skew (continuous, always on)

### The idea

If we are long, we want to sell more eagerly and buy less eagerly, so the
inventory drifts back toward flat on its own — without ever crossing the spread
(we stay a pure maker). We achieve this by shifting **both** quotes *down* when
long (and *up* when short). Long → both quotes lower → our ask is more likely to
be lifted (we sell), our bid is less likely to be hit (we buy less). The width
between the quotes is untouched; only their center moves. That center is the
**reservation price**.

### The computation, step by step

**Fair value.** The starting reference is the fair price:

```
if use_microprice:  fair = ba*imb + bb*(1-imb)      # imb = bq/(bq+aq)
else:               fair = 0.5*(bb + ba)            # plain mid
```

The microprice leans fair value toward the heavier side of the book. In this
project the microprice lean was identified as the *cause* of a systematic short
drift (it sells into ask-heavy books), so production runs with `use_microprice =
False` — fair is the plain mid, and directional leaning is left entirely to the
OBI signal and the inventory skew, not baked into fair value.

**Horizon (tau).** A time fraction that is 1.0 at the open and 0.0 at the
flatten time:

```
tau = clamp( (t1 - now) / (t1 - t0), 0, 1 )
```

`tau` is the "remaining fraction of the session." It appears in both the skew
and the risk half-spread, so **both the lean and the width shrink as the close
approaches** — early in the day there is a lot of time for inventory to hurt us,
so we lean and charge more; near the close there is little time left, so the
risk term decays toward zero.

**Volatility in price units.** A subtle but critical fix lives here. The
per-event volatility `self.sigma` is a *fractional-return* volatility (order
1e-4). Using it directly in a variance term (`sigma^2`) produced skews under
1/100th of a tick — inventory control was effectively inert. The fix converts to
a **price** volatility before squaring:

```
sigma_p = self.sigma * fair          # fractional vol -> PKR price vol
```

**Remaining inventory variance.** The inventory risk accumulated over the
remaining session, scaled by a calibrated dimensional bridge:

```
remaining_var = sigma_p^2 * session_scale * tau
```

`session_scale` is the per-symbol constant (units 1/PKR) that bridges PKR
variance back to a PKR price displacement. It is **not** hand-tuned — it is
back-solved per symbol so that the skew at maximum inventory equals roughly one
median half-spread (derived values: PPL ≈ 7.6, UBL ≈ 3.9). This is what makes
the skew "correctly sized": a full-inventory book leans its quotes by about half
a spread, which is meaningful but not violent.

**Inventory in lots, not shares.** The position is expressed in *clips* (base
quote size), not raw shares, so the lean scales with how many clips from flat we
are — and `session_scale` was calibrated against this same unit:

```
pos_lots = pos / size0
```

**The skew and the reservation price.**

```
skew        = gamma * remaining_var * pos_lots
reservation = fair - skew
```

`gamma` is the risk-aversion parameter (0.15, a free parameter, swept — not
derived from a formula). The sign convention: **long (pos > 0) → skew > 0 →
reservation *below* fair → we sell eagerly and buy less.** Short is the mirror.
The width is untouched — this is purely a shift of the pair's center.

### Why skew, not just a hard cap

There is a hard inventory cap (`max_inv`, 10 clips) and a soft band
(`soft_inv`, 3 clips), but those are *backstops*. Past the soft band we stop
quoting the side that would add to the position (a long past +soft_inv stops
posting bids, so fills can only reduce it); at the hard cap the adding side is
killed entirely. But the **skew is the real control** — it continuously,
proportionally leans the book back toward flat long before those discrete limits
bind. The caps just guarantee we can never blow through a limit if the skew alone
is outrun by a fast one-sided market.

---

## Part 2 — The half-spread (width) — for contrast

Skew moves the center; the **half-spread** sets how far the two quotes sit from
it. It is built from four additive parts, so you can see what inventory does
*not* touch:

```
half_risk    = 0.5 * (gamma * sigma^2 * tau) * fair      # Ho-Stoll risk term
half_adverse = toxicity * (sigma * fair) * 2.0            # adverse-selection charge
as_base      = (1/gamma)*ln(1 + gamma/kappa) * fair       # A-S base (inert until kappa calibrated)
cost_floor   = (fee_pct + min_edge_pct) * fair            # hard economic floor
half         = half_risk + half_adverse + as_base_weight*as_base + cost_floor
half         = max(half, tick)                            # never inside one tick
```

Two things adjust the width situationally: a **quiet market** halves the
adverse-selection charge (tighten when flow is benign), and a **wide market**
raises the placement so we capture the prevailing spread rather than compressing
it to our floor and donating the difference (`half = max(half, mkt_half −
improve_ticks*tick)`). Inventory does **not** enter the width — only the
placement. That separation is deliberate and is the Ho-Stoll result.

A **viability gate** sits here too: if the market's spread is narrower than twice
our required half, no profitable passive quote exists and we stand aside — except
during an active unwind, where getting flat outranks earning edge.

---

## Part 3 — Time-to-close (EOD) trigger — a ramp then a cliff

Enabled by `enable_eod_trigger`. This is the *time-of-day* urgency mechanism. It
has two zones defined by minutes remaining until the true continuous close
(`t1`, the actual bell — not wall-clock, so Friday's split session and Ramadan
hours are handled correctly):

### The two zones

**RAMP zone** — between `eod_ramp_start_min` and `eod_cliff_min` before the
close. Urgency grows on an *inverse* (exploding) shape:

```
u_t = (eod_ramp_start_min / mins_left) - 1.0
```

At the ramp start `u_t = 0` (no effect); as `mins_left` shrinks toward the cliff,
`u_t` explodes upward. This urgency **widens** the affected side(s):

```
raw = half * (1 + u_t)
cap = widen_cap_spreads * ref_spr          # ceiling: a multiple of the reference spread
half_side = max(half, min(raw, cap))       # widen, but never below base, never above cap
```

The widen is capped at a multiple of the *reference* spread (`max(current
spread, EMA spread)`, so a momentarily tight book can't collapse the cap while a
genuinely widening one still grows it). Widening a side makes it progressively
*unfillable* — a deliberate, graceful withdrawal rather than an abrupt cancel.

**CLIFF zone** — the final `eod_cliff_min` minutes. Here a flat book goes fully
**dark on both sides** (`kill_buy = kill_sell = True`): this close to the bell we
want no new position of either sign. The one-tick gap between the ramp cap and
the cliff is intentional, so a single tick of price movement cannot jump straight
from outside the ramp into the dark cliff.

### Flat vs. holding

The zones above describe the **flat** case (widen both sides in the ramp, go dark
in the cliff — any acquisition near the bell is unwanted). If we are **holding**
inventory, the behaviour changes to an **unwind**:

- The **adding side is killed** (long → stop buying; short → stop selling).
- The **exit side is leaned** to the most aggressive *post-only* placement (the
  touch: a long's sell goes to `bb + tick`, a short's buy to `ba − tick`). It is
  never crossed — we remain a pure maker — but it is placed as aggressively as a
  maker can, to get flat.

### The POV-sized engagement (the key refinement)

*When* does the holding-unwind engage? Not on a fixed clock. It engages when the
**current** inventory can no longer be cleared passively in the tradeable time
remaining, at our participation cap — SZ's POV (participation-of-volume) model:

```
tradeable minutes left  = overlap of [now, close] with the continuous segments
                          (Friday's break does NOT count as sellable time)
expected shares/min     = weighted-average of the 4-bucket volume profile over
                          the remaining minutes, allocated close-backward
                          (Last15 first, then PreClose45, then Middle, then First15)
my clearable rate       = expected shares/min * unwind_pov   (never assume > POV of the tape)
minutes needed          = |position| / my clearable rate
engage unwind  <=>  minutes needed >= tradeable minutes left
```

This means the unwind is **dormant when inventory is small** (it can always be
cleared, so no urgency) and **fires early on a thin day when genuinely loaded**
(the tape can't absorb the position, so start working it off now). The ramp start
is effectively POV-sized per name, so "stop adding, work the exit" is correct for
the whole window, not just the last minute.

---

## Part 4 — Distance-to-lock trigger — a ramp then a cliff, on the trapped side

Enabled by `enable_lock_trigger`. PSX applies a ±10% per-scrip price lock
(circuit band). If price pins to the band, one side of your book becomes a
**trap**: near the *upper* band you cannot buy back above the cap, so a fresh
**short** is the dangerous acquisition; near the *lower* band a fresh **long**
is. The trigger acts on the **trapped side only**.

### Measured as percent of price, from the mid

A hard-won design point: distance to the band is measured **from the mid, as a
percent of price** — not from the touch, and not in spread units. An earlier
spread-relative version fired spuriously whenever the spread merely widened
(median false-fire at 3-4% from the band on names that never lock). Percent-of-
price is both instantaneous and stable:

```
d_up_pct = (limit_up - mid) / mid * 100      # 0 = mid sitting on the upper band
d_dn_pct = (mid - limit_dn) / mid * 100      # 0 = mid sitting on the lower band
```

The band prices come **only** from the exchange-published circuit-breaker rows
(`limit_up` / `limit_dn`), synced by the engine — never inferred.

### Low-price tick tier

On cheap names a fixed percent can be smaller than a tick, so the thresholds take
the *larger* of the percent and a tick-based floor:

```
tick_pct  = tick / mid * 100
cliff_pct = max(lock_cliff_pct, min_cliff_ticks * tick_pct)
ramp_pct  = max(lock_ramp_pct,  cliff_pct + min_ramp_gap_ticks * tick_pct)
```

So on normal-priced names the percent thresholds bind; on cheap names the tick
floors bind, guaranteeing the zones are always at least a few ticks wide and
jump-proof.

### The two zones (upper band shown; lower is mirrored)

**RAMP zone** — `cliff_pct < d_up_pct < ramp_pct`. Exploding urgency, same shape
as the EOD ramp, applied to the **trapped (SELL) side only**:

```
u_l = (ramp_pct / d_up_pct) - 1.0
act["u_sell"] = max(act["u_sell"], u_l)      # widen the sell side (flat case)
```

**CLIFF zone** — `d_up_pct <= cliff_pct`. The trapped side goes **dark**
(`kill_sell = True` for the upper band). Note the asymmetry: near a limit-up
close, being *long* is the **safe** side (you sell into stacked bids), so the BUY
side stays live — only the SELL side is suppressed. The lower band mirrors this
exactly (trapped = long, act on BUY, LONG-into-limit-down is the danger).

### Zone-split holding rule (distinct from the EOD unwind)

A key distinction the project settled on: the **lock ramp is a price-proximity
warning, not a liquidity budget.** So while *holding* inside a lock **ramp**,
both sides stay quoted (the base inventory skew already leans appropriately) —
only the lock **cliff** actually pulls the adding side. This is the "zone-split"
rule, and it differs from the EOD trigger, where holding inside the *whole*
window (ramp and cliff) engages the unwind, because there the ramp start is
POV-sized to mean "the remaining volume can just absorb our inventory."

Whenever an unwind or cliff-pull is active, the **exit side is protected**: any
kill or widen a lock cliff might have set on the exit side is undone, because a
trapped holder must always be able to post (best-effort) on the side that gets it
flat. Exit access wins over every other rule.

---

## Part 5 — How the three combine in one cycle

Order of evaluation each quoting cycle:

1. **Reactive jump gate** (a separate safety): if a large adverse move just
   happened, go fully dark for a cooldown and return nothing.
2. **Fair value → inventory skew → reservation price** (Part 1). Always on.
3. **Half-spread** built (Part 2), viability gate checked.
4. **Trigger state** computed (Parts 3 & 4): per-side `kill`, per-side widen
   `urgency`, and a `lean_exit` flag.
5. **Placement:** bid at `reservation − half_buy`, ask at `reservation +
   half_sell`, where `half_buy`/`half_sell` include any trigger widening; sides
   are dropped if killed or if past the inventory band/cap; the exit side is
   pinned to the touch if `lean_exit` is set.
6. **Quote pegging** (optional): hold the previous desired quote until the ideal
   drifts past a tolerance, to preserve queue position on burst-flow names.

So on a normal mid-session cycle with modest inventory, only the skew is active —
the quotes lean gently toward flat and sit at the risk-and-adverse-selection
width. As the close approaches, or as price nears a band, the triggers layer on
top: first widening the relevant side (ramp), then killing it (cliff), and — if
holding — leaning the exit side to the touch to work the position off passively.
Every action is a *maker* action: the strategy never crosses the spread, even
when getting flat is urgent. The most aggressive thing it will ever do is post at
the touch.

---

## Fill tagging (why the windows are recorded)

Every quote cycle records which window it was in (`time_cliff`, `lock_cliff`,
`time_ramp`, `lock_ramp`, or `none`), cliffs taking priority over ramps. The
backtester stamps every fill with this tag, so questions like "did we actually
sell during the cliff, or only widen into it?" are directly answerable from the
fills — which is how the EOD and lock mechanics were validated rather than
assumed.

---

## Parameter summary

| Parameter | Role | Value / source |
|---|---|---|
| `gamma` | risk aversion; scales skew and risk-width | 0.15, free parameter, swept |
| `session_scale` | bridges PKR variance to price displacement in the skew | back-solved per symbol so max-inventory skew ≈ 1 median half-spread (PPL 7.6, UBL 3.9) |
| `size0` (clip) | base quote size; inventory measured in these units | 3× trailing-10-day median trade size |
| `soft_inv` | stop quoting the adding side past this | 3 clips |
| `max_inv` | hard cap; adding side killed | 10 clips |
| `use_microprice` | lean fair value toward heavier book side | False in production (caused short drift) |
| `eod_ramp_start_min` / `eod_cliff_min` | EOD ramp/cliff zone bounds | POV-sized engagement via the 4-bucket profile |
| `unwind_pov` | max participation of tape during unwind | 0.10 |
| `lock_ramp_pct` / `lock_cliff_pct` | distance-to-band zone bounds (% of price) | ramp ~2.0%, cliff ~0.5% of a 10% band, with tick-tier floors |
| `widen_cap_spreads` | ceiling on trigger widening | multiple of the reference spread |

The two triggers default **off** (`enable_eod_trigger`, `enable_lock_trigger`),
so the base PPL/UBL path is a clean skew-only quoter; they are switched on for
production and for the futures same-day-flatten runs, where the EOD unwind is the
mechanism that flattens the book into every close.
