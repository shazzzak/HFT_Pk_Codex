# probe_lean_price_normalisation.py -- REWRITTEN AGAIN 2026-09-15 19:25.
#
# ===========================================================================
# THE THING THE COVERAGE TABLE JUST SHOWED, WHICH MATTERS MORE THAN THIS PROBE
# ===========================================================================
#     symbol  strategy    fills    days   avg_px
#     UBL     naive     174,820     206   412.90
#     PPL     micro     166,328     207   220.50
#     PPL     naive      93,286     207   224.46
#     UBL     micro      83,968     206   409.07
#                       -------
#                       518,402
#
# TWO SYMBOLS AND TWO STRATEGIES. 268,106 of those fills (52%) are NAIVE --
# the flat baseline quoter, which has NO queue skew, NO lean, and none of the
# machinery the extreme-imbalance finding was attributed to.
#
# probe_conditional_markout.py reported "518,285 fills", which is this whole
# store. So that probe almost certainly pooled naive and micro fills, and the
# conclusion drawn from it -- "capture goes negative on the exposed side above
# obi_1 0.80, which is the signature of the lean walking the quote through the
# touch" -- was computed over a population that is half made of fills from a
# strategy WITH NO LEAN IN IT.
#
# You cannot attribute an effect to a mechanism using data where half the rows
# come from a system that does not have the mechanism. Everything built on that
# finding today -- the three competing explanations, the extreme_obi sweep, the
# queue_skew_thresh_hi patch -- inherits the problem.
#
# THIS FILE THEREFORE FILTERS TO ONE STRATEGY AND SAYS WHICH, EVERY TIME.
# It also reports the naive arm separately, because if naive shows the SAME
# negative capture at extreme imbalance, then the effect has nothing to do with
# the lean at all -- it is a property of trading at extreme imbalance, full stop,
# and that single comparison closes the question.
#
# ===========================================================================
# THE OTHER CHANGE: NO ROW-LEVEL PULLS
# ===========================================================================
# Both previous runs stopped at the same place -- the first `.df()` that
# materialises every fill into pandas. I do not know the cause; no traceback was
# captured. Rather than guess at it, this version does every aggregation in SQL
# and brings back only small summary frames, so the failing operation no longer
# happens at all. If it still fails, the traceback will now name a specific
# query rather than a 500k-row conversion.
#
# WHAT IT ASKS. On the micro arm only:
#   A. Is the loss where the SPREAD IS TIGHT IN TICKS? The favourable quote sits
#      about (spread_ticks/2 - improve_ticks - lean_ticks) from the reservation,
#      so it is pinned by the post-only clip when the spread is <= 6 ticks.
#      Crossing is a TICK question, never a bps one.
#   B. Is the loss in CAPTURE or in MARKOUT? Bad capture is placement, which the
#      lean can be blamed for. Bad markout is adverse selection, which no
#      placement change fixes.
#   C. Does NAIVE show the same thing? If yes, it is not the lean.
#
# Run: caffeinate -is python probe_lean_price_normalisation.py
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import duckdb
import matplotlib
# a non-interactive backend, so the script works over ssh
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- PATHS from config_pk; never a literal in this file ----------------------
try:
    # the project's central path module
    import config_pk
    # where every output goes
    RESULTS_ROOT = Path(config_pk.RESULTS_ROOT)
except Exception as _e:
    # a wrong store is worse than a crash
    raise SystemExit("probe_lean_price_normalisation: could not import "
                     "RESULTS_ROOT from config_pk. Original: %r" % _e)
# the fill store, named by config_pk
FILLS = None
# every attribute config_pk might use for it
for _attr in ("FILLS_DIR", "FILLS_ROOT", "FILLS", "FILL_STORE"):
    # take the first that exists
    if hasattr(config_pk, _attr):
        FILLS = Path(getattr(config_pk, _attr))
        break
# nothing named -> stop rather than invent a path
if FILLS is None:
    raise SystemExit("config_pk names no fill store.")

# run stamp on every output, so a re-run never collides with an earlier one
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
# where this probe's outputs go
OUT_DIR = RESULTS_ROOT / "lean_price"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# CONSTANTS
# ===========================================================================
# STRUCTURAL: PSX tick is a flat 0.01 PKR at every price
TICK_PKR = 0.01
# the strategy arm this probe analyses. THE LEAN ONLY EXISTS IN THIS ONE.
STRATEGY = "micro"
# the comparison arm, which has no lean and therefore no mechanism to blame
BASELINE_STRATEGY = "naive"
# the obi_1 level where the earlier probe found capture going negative
EXTREME_OBI = 0.80
# the shipped lean, in ticks
LEAN_TICKS = 2.0
# micro_mm's improve_ticks: never quote tighter than this inside the touch
IMPROVE_TICKS = 1.0
# THE CLIP THRESHOLD, derived not guessed: the favourable side is placed at
# about (spread_ticks/2 - improve_ticks - lean_ticks) ticks from the
# reservation, and at or below zero the post-only clip pins it AT the opposite
# touch. Solving for the spread: spread_ticks <= 2*(improve_ticks + lean_ticks).
CLIP_SPREAD_TICKS = 2.0 * (IMPROVE_TICKS + LEAN_TICKS)

print(f"probe_lean_price_normalisation  stamp={STAMP}")
print(f"  fills: {FILLS}")
print(f"  outputs -> {OUT_DIR}")
print(f"  post-only clip pins the leaned quote when spread <= "
      f"{CLIP_SPREAD_TICKS:.0f} ticks")


def resolve(cols, candidates, what):
    """First candidate present, or fail NAMING what was actually there."""
    # the first candidate that exists
    for c in candidates:
        # a match ends the search
        if c in cols:
            return c
    # nothing matched: say what was wanted and what is available
    raise SystemExit(f"\n  FATAL: no column for {what}.\n"
                     f"  looked for: {candidates}\n"
                     f"  the store has: {sorted(cols)}")


# ===========================================================================
# SECTION 0 -- coverage, strategy mix, and the exact vocabulary of every
# column this probe branches on. NOTHING is guessed from here on.
# ===========================================================================
print("\n=== SECTION 0: coverage and vocabulary ===")
# one connection for the whole probe
con = duckdb.connect()
# the parquet glob, hive-partitioned by date
GLOB = f"read_parquet('{FILLS}/**/*.parquet', hive_partitioning=1, union_by_name=1)"
# one row, to learn the schema without pulling anything
try:
    # LIMIT 1 is enough for column names
    head = con.execute(f"SELECT * FROM {GLOB} LIMIT 1").df()
except Exception as e:
    # a bad path is the likeliest cause
    raise SystemExit(f"\n  FATAL: could not read {FILLS}\n  {e!r}")
# the columns available
COLS = set(head.columns)
# resolve every column, failing loudly rather than guessing
C_SYM = resolve(COLS, ["symbol", "sym", "ticker"], "symbol")
C_DATE = resolve(COLS, ["date", "trade_date", "dt"], "date")
C_PX = resolve(COLS, ["price", "px", "fill_px"], "fill price")
C_QTY = resolve(COLS, ["qty", "size", "quantity", "shares"], "quantity")
C_OBI = resolve(COLS, ["obi_1", "obi1", "obi"], "book imbalance at fill")
C_CAP = resolve(COLS, ["capture", "capture_bps"], "capture in bps")
C_MKT = resolve(COLS, ["markout", "markout_bps"], "markout in bps")
C_SPR = resolve(COLS, ["spread_bps", "spread"], "spread in bps at fill")
C_STRAT = resolve(COLS, ["strategy", "strat", "config"], "strategy tag")
# prefer the TEXT side column when the store has one, since side may be numeric
C_SIDE = "side_str" if "side_str" in COLS else resolve(
    COLS, ["side", "fill_side"], "our side")

# THE VOCABULARY CHECK. Print the distinct values of everything this probe
# branches on, so the CASE expressions below are built from what is in the data
# rather than from what I assume is in it. Cheap: these are GROUP BY counts,
# not row pulls.
vocab = con.execute(f"""
    SELECT CAST({C_SIDE} AS VARCHAR) AS side_value, COUNT(*) AS n
    FROM {GLOB} GROUP BY 1 ORDER BY n DESC
""").df()
# show them
print(f"  {C_SIDE} values: "
      f"{', '.join(f'{r.side_value}={int(r.n):,}' for _, r in vocab.iterrows())}")
# the buy vocabulary this store actually uses
BUY_VALUES = [v for v in vocab.side_value.astype(str)
              if v.strip().upper().startswith("B")
              or v.strip() in ("1", "+1", "1.0")]
# nothing that looks like a buy means the CASE below would classify everything
# as a sell and silently invert the whole analysis
if not BUY_VALUES:
    raise SystemExit(f"\n  FATAL: no value in {C_SIDE} looks like a BUY.\n"
                     f"  values present: {list(vocab.side_value)}")
# the SQL literal list for the CASE
BUY_SQL = ", ".join(f"'{v}'" for v in BUY_VALUES)
# say which values were treated as buys, so a wrong call is visible
print(f"  treating as BUY: {BUY_VALUES}")

# coverage by symbol AND strategy -- the table that exposed the problem
cov = con.execute(f"""
    SELECT {C_SYM} AS symbol, CAST({C_STRAT} AS VARCHAR) AS strategy,
           COUNT(*) AS n_fills,
           COUNT(DISTINCT CAST({C_DATE} AS VARCHAR)) AS n_days,
           SUM({C_PX} * {C_QTY}) / NULLIF(SUM({C_QTY}), 0) AS avg_px
    FROM {GLOB} GROUP BY 1, 2 ORDER BY n_fills DESC
""").df()
# show it
print(f"\n  coverage -- {cov.symbol.nunique()} symbol(s), "
      f"{cov.strategy.nunique()} strategy tag(s), "
      f"{int(cov.n_fills.sum()):,} fills:")
print(f"    {'symbol':<8}{'strategy':<12}{'fills':>12}{'days':>7}{'avg_px':>10}")
# one line each
for _, r in cov.iterrows():
    print(f"    {r.symbol:<8}{r.strategy:<12}{int(r.n_fills):>12,}"
          f"{int(r.n_days):>7}{r.avg_px:>10.2f}")
# THE STRATEGY WARNING, sized to the mix and printed where it cannot be missed
mix = cov.groupby("strategy").n_fills.sum()
# how much of the store is NOT the strategy being analysed
if STRATEGY in mix.index and len(mix) > 1:
    other = int(mix.drop(STRATEGY).sum())
    print(f"\n  *** {other:,} of {int(mix.sum()):,} fills "
          f"({100.0*other/mix.sum():.0f}%) are NOT '{STRATEGY}'.")
    print(f"  *** Any analysis of the lean must exclude them: the other arms")
    print(f"  *** do not have a lean, so they cannot evidence one.")
    print(f"  *** This probe filters to strategy = '{STRATEGY}'.")
# the strategy is not in the store at all, which stops everything
elif STRATEGY not in mix.index:
    raise SystemExit(f"\n  FATAL: no fills tagged strategy='{STRATEGY}'. "
                     f"Present: {list(mix.index)}")
# the coverage caveat, sized to the symbol count
if cov.symbol.nunique() < 10:
    print(f"\n  *** {cov.symbol.nunique()} SYMBOL(S) of a 113-name book. Every")
    print(f"  *** number below is a hypothesis about the book, never a")
    print(f"  *** measurement of it.")

# the obi scale check: obi_1 spans [-1,+1]; micro_mm's |imb-0.5| spans [0,0.5]
scale = con.execute(f"SELECT MIN({C_OBI}) lo, MAX({C_OBI}) hi "
                    f"FROM {GLOB}").df().iloc[0]
# a range stopping at 0.5 means this is the engine scale, not obi_1
if scale.hi <= 0.51 and scale.lo >= -0.51:
    raise SystemExit(f"\n  FATAL: {C_OBI} spans [{scale.lo:.3f},{scale.hi:.3f}], "
                     f"the |imb-0.5| scale, NOT obi_1.")
# confirmed
print(f"\n  {C_OBI} range [{scale.lo:.4f}, {scale.hi:.4f}] -- obi_1 scale, good")

# ===========================================================================
# THE SHARED SQL -- every derived quantity, defined once
# ===========================================================================
# EXPOSURE, defined here in SQL rather than in pandas so it is computed the same
# way everywhere. micro_mm: bid-heavy (obi_1 > 0) makes the BUY side FAVOURABLE
# -- it steps CLOSER to the touch -- and the SELL side EXPOSED, stepping FURTHER
# away. So matching signs = favourable.
#
# SPREAD IN TICKS: spread_pkr = spread_bps/10000 * px, and one tick is TICK_PKR,
# so spread_ticks = spread_bps * px / (10000 * TICK_PKR).
def base_cte(strategy):
    """The common projection, filtered to ONE strategy arm."""
    # every derived column the later queries need
    return f"""
    WITH base AS (
      SELECT {C_SYM} AS symbol,
             CAST({C_DATE} AS VARCHAR) AS date,
             {C_CAP} AS capture,
             {C_MKT} AS markout,
             {C_CAP} + {C_MKT} AS gross,
             {C_PX} * {C_QTY} AS notional,
             {C_OBI} AS obi,
             ABS({C_OBI}) AS abs_obi,
             {C_SPR} * {C_PX} / (10000.0 * {TICK_PKR}) AS spread_ticks,
             CASE WHEN CAST({C_SIDE} AS VARCHAR) IN ({BUY_SQL})
                  THEN 1 ELSE -1 END AS side_sign
      FROM {GLOB}
      WHERE CAST({C_STRAT} AS VARCHAR) = '{strategy}'
        AND {C_SPR} IS NOT NULL AND {C_SPR} > 0
        AND {C_PX} > 0 AND {C_QTY} > 0
    ),
    tagged AS (
      SELECT *,
             CASE WHEN (obi > 0 AND side_sign = 1)
                    OR (obi < 0 AND side_sign = -1)
                  THEN 'favourable' ELSE 'exposed' END AS exposure,
             CASE WHEN spread_ticks < 2 THEN '1-2'
                  WHEN spread_ticks < 4 THEN '2-4'
                  WHEN spread_ticks < 6 THEN '4-6'
                  WHEN spread_ticks < 10 THEN '6-10'
                  WHEN spread_ticks < 20 THEN '10-20'
                  ELSE '20+' END AS spread_bucket,
             CASE WHEN abs_obi < 0.30 THEN '0-0.30'
                  WHEN abs_obi < 0.60 THEN '0.30-0.60'
                  WHEN abs_obi < 0.80 THEN '0.60-0.80'
                  ELSE '0.80+' END AS obi_bucket,
             spread_ticks <= {CLIP_SPREAD_TICKS} AS clipped
      FROM base
    )"""


# the order the spread buckets are reported in, so the shape reads left to right
SPREAD_ORDER = ["1-2", "2-4", "4-6", "6-10", "10-20", "20+"]
# same for the imbalance axis
OBI_ORDER = ["0-0.30", "0.30-0.60", "0.60-0.80", "0.80+"]

# ===========================================================================
# SECTION 1 -- reproduce the headline on the MICRO arm alone
# ===========================================================================
print(f"\n=== SECTION 1: exposed side above obi_1 {EXTREME_OBI}, "
      f"strategy='{STRATEGY}' only ===")
# notional-weighted components, computed in SQL so no rows come back
hl = con.execute(f"""
    {base_cte(STRATEGY)}
    SELECT COUNT(*) AS n_fills, SUM(notional) AS notional,
           SUM(capture * notional) / SUM(notional) AS capture,
           SUM(markout * notional) / SUM(notional) AS markout,
           SUM(gross   * notional) / SUM(notional) AS gross,
           MEDIAN(spread_ticks) AS med_spread_ticks
    FROM tagged WHERE exposure = 'exposed' AND abs_obi > {EXTREME_OBI}
""").df().iloc[0]
# an empty population means the finding is not in this arm at all
if not hl.n_fills:
    raise SystemExit(f"\n  no exposed fills above obi_1 {EXTREME_OBI} in the "
                     f"'{STRATEGY}' arm.")
# the headline, against what was previously reported
print(f"  n = {int(hl.n_fills):,} fills   "
      f"(probe_conditional_markout reported 86,238 on the POOLED store)")
print(f"  capture {hl.capture:+.4f}  markout {hl.markout:+.4f}  "
      f"gross {hl.gross:+.4f} bps   (notional-weighted)")
print(f"  median spread at these fills: {hl.med_spread_ticks:.1f} ticks")
# WHICH TERM CARRIES THE LOSS. Bad capture is placement, which the lean can be
# blamed for; bad markout is adverse selection, which placement cannot fix.
if hl.capture < 0 and hl.markout < 0:
    worse = "CAPTURE (placement)" if abs(hl.capture) > abs(hl.markout) \
        else "MARKOUT (adverse selection)"
    print(f"  both negative; the larger term is {worse}")
elif hl.markout < 0 <= hl.capture:
    print(f"  markout negative, capture POSITIVE -> adverse selection, not")
    print(f"  placement. No change to where the quote sits will fix that.")
elif hl.capture < 0 <= hl.markout:
    print(f"  capture negative, markout positive -> a PLACEMENT problem, which")
    print(f"  is the only reading under which the lean is implicated.")

# ===========================================================================
# SECTION 2 -- DOES THE NAIVE ARM DO THE SAME THING? The decisive comparison.
# ===========================================================================
print(f"\n=== SECTION 2: the same population in '{BASELINE_STRATEGY}' "
      f"(no lean at all) ===")
# the naive arm, same filter, same definitions
nv = con.execute(f"""
    {base_cte(BASELINE_STRATEGY)}
    SELECT COUNT(*) AS n_fills,
           SUM(capture * notional) / SUM(notional) AS capture,
           SUM(markout * notional) / SUM(notional) AS markout,
           SUM(gross   * notional) / SUM(notional) AS gross
    FROM tagged WHERE exposure = 'exposed' AND abs_obi > {EXTREME_OBI}
""").df().iloc[0]
# the comparison, if the arm exists
if nv.n_fills:
    # print it beside the micro numbers
    print(f"  n = {int(nv.n_fills):,} fills")
    print(f"  capture {nv.capture:+.4f}  markout {nv.markout:+.4f}  "
          f"gross {nv.gross:+.4f} bps")
    # THE VERDICT. naive has no queue skew, so if it shows the same shape the
    # effect cannot be caused by the lean.
    print(f"\n  micro gross {hl.gross:+.4f}   naive gross {nv.gross:+.4f}   "
          f"difference {hl.gross - nv.gross:+.4f} bps")
    # naive also negative means the lean is exonerated
    if nv.gross < 0:
        print(f"  *** NAIVE IS ALSO NEGATIVE. It has no lean, no queue skew and")
        print(f"  *** no defensive widen. So losing money on the exposed side at")
        print(f"  *** extreme imbalance is a property of TRADING THERE, not of")
        print(f"  *** the lean. The queue_skew_thresh_hi patch would be treating")
        print(f"  *** a mechanism that is not the cause.")
    # naive positive while micro is negative points back at the lean
    else:
        print(f"  *** NAIVE IS POSITIVE where micro is negative. The difference")
        print(f"  *** is attributable to what micro does and naive does not --")
        print(f"  *** the lean is back in the frame.")
else:
    # nothing to compare against
    print(f"  no '{BASELINE_STRATEGY}' fills in this population")

# ===========================================================================
# SECTION 3 -- the tick-geometry split, micro arm
# ===========================================================================
print(f"\n=== SECTION 3: by spread in TICKS (clip threshold "
      f"{CLIP_SPREAD_TICKS:.0f}) ===")
# one row per spread bucket, all weighted in SQL
sp = con.execute(f"""
    {base_cte(STRATEGY)}
    SELECT spread_bucket, COUNT(*) AS n_fills, SUM(notional) AS notional,
           SUM(capture * notional) / SUM(notional) AS capture,
           SUM(markout * notional) / SUM(notional) AS markout,
           SUM(gross   * notional) / SUM(notional) AS gross
    FROM tagged WHERE exposure = 'exposed' AND abs_obi > {EXTREME_OBI}
    GROUP BY 1
""").df()
# order the buckets so the shape reads left to right
sp["o"] = sp.spread_bucket.map({k: i for i, k in enumerate(SPREAD_ORDER)})
sp = sp.sort_values("o").drop(columns="o")
# print the table
print(f"  {'spread':<10}{'n_fills':>10}{'notional PKR':>16}{'capture':>10}"
      f"{'markout':>10}{'gross':>10}")
for _, r in sp.iterrows():
    print(f"  {r.spread_bucket:<10}{int(r.n_fills):>10,}{r.notional:>16,.0f}"
          f"{r.capture:>+10.4f}{r.markout:>+10.4f}{r.gross:>+10.4f}")
print("  ^ loss in the TIGHT buckets fading as the spread widens implicates the")
print("    post-only clip. Flat across buckets does not.")

# DAY AS UNIT for the one comparison that carries an argument
print(f"\n  clipped vs not, day-as-unit:")
# one weighted gross per day per group, aggregated in SQL
dd = con.execute(f"""
    {base_cte(STRATEGY)}
    SELECT date, clipped,
           SUM(gross * notional) / SUM(notional) AS gross
    FROM tagged WHERE exposure = 'exposed' AND abs_obi > {EXTREME_OBI}
    GROUP BY 1, 2
""").df()
# one column per group
wide = dd.pivot(index="date", columns="clipped", values="gross")
# both groups must be present on a day for it to be a paired observation
if wide.shape[1] == 2 and wide.dropna().shape[0] >= 3:
    # days where both groups traded
    both = wide.dropna()
    # tight book minus wide book
    d = both[True] - both[False]
    # paired t across days
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) \
        if d.std(ddof=1) > 0 else np.nan
    # report
    print(f"    spread <= {CLIP_SPREAD_TICKS:.0f} ticks: {both[True].mean():+.4f}"
          f" bps/day over {len(both)} days")
    print(f"    spread >  {CLIP_SPREAD_TICKS:.0f} ticks: {both[False].mean():+.4f}"
          f" bps/day")
    print(f"    difference {d.mean():+.4f}  t={t:+.2f}  "
          f"{int((d < 0).sum())}/{len(d)} days worse on the tight book")
    print(f"    |t| > 2 with most days on one side is the bar.")
else:
    # not enough overlap to make a paired claim
    print(f"    only {wide.dropna().shape[0]} days have both groups -- no "
          f"paired claim possible")

# ===========================================================================
# SECTION 4 -- the grid, and per name
# ===========================================================================
print(f"\n=== SECTION 4: imbalance x spread, exposed side, '{STRATEGY}' ===")
# every cell of the grid, weighted in SQL
gr = con.execute(f"""
    {base_cte(STRATEGY)}
    SELECT spread_bucket, obi_bucket, COUNT(*) AS n,
           SUM(gross * notional) / SUM(notional) AS gross
    FROM tagged WHERE exposure = 'exposed' GROUP BY 1, 2
""").df()
# pivot into a readable grid, in bucket order
pv = gr.pivot(index="spread_bucket", columns="obi_bucket", values="gross")
pv = pv.reindex(index=[s for s in SPREAD_ORDER if s in pv.index],
                columns=[o for o in OBI_ORDER if o in pv.columns])
# show it
print("  notional-weighted gross bps:")
print(pv.round(3).to_string())
print("  ^ read ACROSS a row: a collapse only in the 0.80+ column supports the")
print("    imbalance reading. Read DOWN a column: bad tight-spread rows at every")
print("    imbalance level means the spread was the operative variable.")

# per name, since there are only two
print(f"\n  per symbol, exposed above obi_1 {EXTREME_OBI}:")
per = con.execute(f"""
    {base_cte(STRATEGY)}
    SELECT symbol, COUNT(*) AS n_fills,
           SUM(capture * notional) / SUM(notional) AS capture,
           SUM(markout * notional) / SUM(notional) AS markout,
           SUM(gross   * notional) / SUM(notional) AS gross,
           MEDIAN(spread_ticks) AS med_spread_ticks
    FROM tagged WHERE exposure = 'exposed' AND abs_obi > {EXTREME_OBI}
    GROUP BY 1
""").df()
# one line each
print(f"  {'symbol':<10}{'n_fills':>10}{'capture':>10}{'markout':>10}"
      f"{'gross':>10}{'med spread tk':>16}")
for _, r in per.iterrows():
    print(f"  {r.symbol:<10}{int(r.n_fills):>10,}{r.capture:>+10.4f}"
          f"{r.markout:>+10.4f}{r.gross:>+10.4f}{r.med_spread_ticks:>16.1f}")
print("  ^ two names agreeing is weak evidence. Two names disagreeing is strong")
print("    evidence that there is nothing here to generalise.")

# ===========================================================================
# SECTION 5 -- charts
# ===========================================================================
# three panels
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

# --- panel 1: capture vs markout by spread ---
ax = axes[0]
if len(sp):
    # bar positions
    x = np.arange(len(sp))
    # the two terms side by side, because they mean opposite things
    ax.bar(x - 0.2, sp.capture, 0.4, label="capture (placement)", color="#4C78A8")
    ax.bar(x + 0.2, sp.markout, 0.4, label="markout (adverse selection)",
           color="#F58518")
    # break-even
    ax.axhline(0, color="k", lw=1)
    # the clip threshold sits between the 4-6 and 6-10 buckets
    if "4-6" in list(sp.spread_bucket):
        ax.axvline(list(sp.spread_bucket).index("4-6") + 0.5, color="#D62728",
                   ls="--", lw=2, label=f"clip at {CLIP_SPREAD_TICKS:.0f} ticks")
    # label the buckets
    ax.set_xticks(x)
    ax.set_xticklabels(sp.spread_bucket)
ax.set_xlabel("spread at fill, ticks")
ax.set_ylabel("bps, notional-weighted")
ax.set_title(f"'{STRATEGY}' exposed, |obi_1| > {EXTREME_OBI}\n"
             f"which term carries the loss, and where")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# --- panel 2: micro vs naive, the decisive comparison ---
ax = axes[1]
# the three components for both arms
labels = ["capture", "markout", "gross"]
# micro values
mvals = [hl.capture, hl.markout, hl.gross]
# naive values, or zeros when the arm is absent
nvals = [nv.capture, nv.markout, nv.gross] if nv.n_fills else [0, 0, 0]
# positions
x = np.arange(3)
# side by side
ax.bar(x - 0.2, mvals, 0.4, label=f"{STRATEGY} (has the lean)", color="#4C78A8")
ax.bar(x + 0.2, nvals, 0.4, label=f"{BASELINE_STRATEGY} (no lean)",
       color="#9467BD")
# break-even
ax.axhline(0, color="k", lw=1)
# labels
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("bps, notional-weighted")
ax.set_title(f"exposed side above obi_1 {EXTREME_OBI}\n"
             f"naive also negative = the lean is not the cause")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# --- panel 3: the grid ---
ax = axes[2]
# a line per spread bucket across the imbalance axis
for label in pv.index:
    row = pv.loc[label]
    # skip an all-empty row
    if row.isna().all():
        continue
    # plot it
    ax.plot(range(len(row)), row.values, marker="o", label=f"{label} ticks")
# break-even
ax.axhline(0, color="k", lw=1)
# the imbalance axis
ax.set_xticks(range(len(pv.columns)))
ax.set_xticklabels(pv.columns, rotation=20)
ax.set_xlabel("|obi_1| at fill")
ax.set_ylabel("gross bps, notional-weighted")
ax.set_title("separating the axes\n"
             "rows apart everywhere = it is the spread, not the imbalance")
ax.legend(fontsize=7, title="spread")
ax.grid(alpha=0.3)

# lay out and save with the run stamp, never overwriting an earlier run
plt.tight_layout()
PNG = OUT_DIR / f"lean_tick_geometry_{STAMP}.png"
plt.savefig(PNG, dpi=140)
print(f"\n  wrote {PNG}")
# the tables behind the panels
for name, frame in (("exposed_by_spread", sp), ("obi_spread_grid", gr),
                    ("per_symbol", per), ("coverage", cov)):
    # one timestamped CSV each
    p = OUT_DIR / f"{name}_{STAMP}.csv"
    frame.to_csv(p, index=False)
    print(f"  wrote {p}")

print("\n  READ THE OUTPUT THIS WAY:")
print(f"  1. SECTION 2 FIRST. If '{BASELINE_STRATEGY}' -- which has no lean --")
print(f"     is also negative there, the lean is exonerated and the whole")
print(f"     extreme-OBI thread closes. That single line outranks everything")
print(f"     else this probe prints.")
print(f"  2. Only if naive is CLEAN does the tick geometry in Section 3 matter.")
print(f"  3. And in every case this is {cov.symbol.nunique()} name(s) of 113.")
