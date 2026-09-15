# Hedge cost: the share book beats the futures book on 99 of 100 names

Date: 2026-09-15
Scripts: `probe_futures_width.py`, `probe_spot_width.py`
Data: PSX parsed store, 2026-06-01 to 2026-06-30 (20 trading dates), continuous auction only
Outputs: `futures_width_20260915.csv`, `spot_vs_futures_width_20260915.csv`

## The headline

Crossing the **share book** is cheaper than crossing the **same-ticker active-month
future** on **99 of 100 names**. Median ratio **3.30x**. The one exception, WTL, is a
tie at 1.00x.

This inverts the first rung of the delta-hedge ladder as specified
("first preference for hedge is same ticker, active month futures contract").

## The harder result

**No name can be crossed for less than the 2.67 bps gross edge in either book.**

| | best | median |
|---|---|---|
| share book round trip | EFERT 5.30 bps = 2.0x edge | 13.79 bps = 5.2x edge |
| futures round trip | BOP 11.79 bps = 4.4x edge | 48.09 bps = 18.0x edge |

Not one name's spot spread is below 2.67 bps even **before** fees (minimum 3.75 bps,
EFERT).

## Why this is structural, not a PSX quirk

A market maker **earns** the spread by posting. A hedger **pays** the spread by
crossing. The 2.67 bps edge is derived from the spread the book captures. So a hedge
executed by crossing always costs at least as much as the round trip that created the
inventory earned. A one-for-one delta hedge by crossing cannot be profitable in any
market, on any instrument.

The design survives only through **netting** — paying the toll on net residual
exposure rather than per position. That was already in the stated design
("at the portfolio level there would be cancellations"); this result makes it the
load-bearing assumption rather than a nice-to-have.

## Capacity: how much can be hedged

With a budget of 20% of the 2.67 bps edge spent on hedging:

| hedge venue | cost | max hedged notional as % of traded notional |
|---|---|---|
| EFERT shares (best) | 5.30 bps | 10.1% |
| BOP shares | 7.24 bps | 7.4% |
| median share book | 13.79 bps | 3.9% |
| median futures | 48.09 bps | 1.1% |

At a 50% budget these become 25.2% / 18.4% / 9.7% / 2.8%.

Read: for the median name, the book must net down to ~96% before hedging is
affordable.

## The tick-grid screen for the ETF rung

PSX tick is flat 0.01 PKR. A one-tick spread is `100/P` bps. Therefore:

- spread below 2.67 bps requires price > **37.45 PKR**
- round trip inside the edge including 1.554 bps spot fees requires spread
  < 1.116 bps, i.e. price > **89.6 PKR**

**Any hedge instrument trading below ~90 PKR cannot have a round-trip crossing cost
inside the 2.67 bps edge, even with a perfect one-tick book.** This is the screen to
apply to the sector-ETF and KSE100-ETF rungs before measuring anything else. ETF
spreads have not been measured — that is the open item.

## Where the futures rung is worst

Exactly on the large caps where a hedge would be most wanted: MCB 39.2x, NATF 28.4x,
POL 28.3x, MTL 25.4x, BAHL 20.2x, GLAXO 14.9x, BAFL 14.8x, GHGL 13.6x, KOHC 12.4x,
FABL 11.8x, MEBL 11.3x.

Closest to competitive (still never cheaper): WTL 1.00x, TRG 1.46x, SSGC 1.48x,
AGHA 1.48x, PTC 1.62x, PREMA 1.62x, BOP 1.63x, PACE 1.68x.

## Supporting findings

- **Futures width is deadness, not an artifact.** Spearman rho between trades/day and
  spread in bps: **-0.875**, n=100, p=1.3e-32. Dead-time inflation (all-session median
  / in-session median) is 1.01x — the widths are real all day.
- **Contract selection was never in doubt.** `vol_share` = 1.00 on 95 of 100 roots;
  the active-month pick was the root's entire daily volume. No multi-month mixing.
- **Cash-settled futures do not exist as a market.** `STOCK_CS_FUT`: 6 trades total,
  1 symbol, 1 day, across 20 trading dates. Confirms the earlier read.
- **Market codes** (last 20 dates): REG 8,609,749 trades / 501 symbols (spot);
  STOCK_DEL_FUT 1,215,248 / 215; NDM 4,204 / 281; ODD_LOT 614 / 9; STOCK_CS_FUT 6 / 1.
- **Several names are tick-floored in spot** — a 1-tick spread that is still wide in
  bps because the price is low: WTL 78.1 bps, BECO 17.8, KOSM 17.7, CNERGY 12.3,
  KEL 12.2, UNITY 8.7, FFL 5.7. For these the spread cannot be tighter anywhere.

## Corrections to earlier claims in this workstream

1. **"Futures are the cheap hedge because the fee is 0.19 bps vs 1.554 spot."** Wrong
   axis. The fee is not the cost of a hedge; crossing is. Futures cost 3.3x more than
   shares despite the 1.36 bps fee advantage.
2. **"The spot spread is probably ~3 bps."** Directionally right (shares are cheaper),
   magnitude wrong by ~4x. Median spot spread is 12.24 bps; BOP is 5.69 bps.
3. **"Dead-time weighting is inflating the wide futures names."** Wrong. Inflation is
   1.01x. The missing control was a minimum-trade floor, not a time window.

## Caveats

- Settles **cost only**. If PSX restricts cash-market shorts, the future is not the
  dearer hedge on the short side — it is the only one, and this comparison binds on
  the long side alone. Exchange-rules question, unresolved.
- Costs assume crossing at the touch with **no size impact**, so every figure is a
  **floor**. Real cost is higher for any meaningful size.
- One-leg variant (cross to put the hedge on, post to take it off) halves the figures:
  BOP futures 5.99 bps, median spot ~6.9 bps. Still nothing below 2.67 bps.
- Data window is June 2026. The four-arm futures MM run used Sep-Oct 2025 contracts,
  eight months earlier — do not cross-compare the two directly.
