# probe_conditional_markout.py -- should the lean ALSO adjust quoted size, and is
# a leader's 1-second jump worth reacting to defensively?
#
# WHY THIS EXISTS. Two questions, one measurement, and both are answerable from
# fill data already on disk -- no backtest, no sweep.
#
# QUESTION A -- SHOULD QT_2t ALSO CUT SIZE?
#   Cutting quoted size in a triggered state reduces markout AND capture in the
#   same proportion. So it is a WASH unless, conditional on the trigger firing,
#       |markout| / capture > 1
#   Above 1, every share quoted in that state loses money and quoting fewer of
#   them is pure gain. Below 1, those shares are profitable and cutting size is
#   pure loss. That single ratio, computed per OBI bucket and per side, decides
#   whether obi_throttle belongs in the shipped config -- before any sweep.
#
#   Why this is not already answered: the gate sweep established that more PRICE
#   dosage is exhausted at 2 ticks (0->2t: markout +1.08 bps for capture -0.15;
#   2->3t: markout +0.05 for capture -1.21). That is a statement about price. Size
#   is a different axis and micro_mm's own comment says why it is the cheaper
#   one: the throttle changes QUANTITY only, never price, never queue position,
#   so it does not spend spread capture the way the lean does.
#
# QUESTION B -- IS A LEADER'S JUMP A USABLE DEFENSIVE TRIGGER?
#   leadlag_screen.py found 0 of 157 pairs clear its acceptance rule, but the
#   binding gate there was `ind_bps_median > 1.554 fee`. THAT GATE IS WRONG FOR A
#   DEFENSIVE USE. It is the hurdle for INITIATING a position to capture an
#   anticipated move -- you pay the round trip, so the move must cover it. A quote
#   you never got filled on costs nothing. The defensive hurdle is only: does the
#   adverse selection avoided exceed the capture given up?
#
#   Two further reasons the screen's null does not settle this:
#     1. A correlation is an average over ALL states. A median peak correlation of
#        0.014 says the legs barely co-move on a typical tick. It says nothing
#        about whether a large jump propagates. The relationship can live entirely
#        in the tail -- which is exactly where a threshold trigger fires. The OBI
#        throttle works on the same logic.
#     2. Defence does not need the leader to LEAD. It needs the leader's move to
#        be visible before the adverse flow reaches the thin name -- flow hits the
#        liquid leg first and arrives at the follower later. That is a latency
#        race, not a forecast, and `leader_faster` held for 99 of 157 pairs.
#
#   So: for every fill, was there a large leader move in the preceding window,
#   and was it AGAINST the side we just took? Compare markout in that state to
#   markout everywhere else. Day-as-unit, never pooled across fills.
#
# Read-only. 5-15 minutes. Run: caffeinate -is python probe_conditional_markout.py
from pathlib import Path
from datetime import datetime
import duckdb, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- PATHS from config_pk; never a literal in this file ----------------------
try:
    # the project's central path module
    import config_pk
    # the parsed store, for the leader mid series
    PARSED = str(config_pk.PARSED_ROOT)
    # the results root
    RESULTS = Path(config_pk.RESULTS_ROOT)
    # where fill_attribution.py writes its per-fill output
    FILLS = Path(config_pk.FILLS_DIR)
except Exception as _e:
    # fail naming the fix
    raise SystemExit("probe_conditional_markout: could not import paths from "
                     "config_pk. Run from existing_mm_live/. Original: %r" % _e)
# fail immediately with the path named
if not FILLS.exists():
    raise SystemExit(f"FILLS_DIR not found: {FILLS} -- run fill_attribution.py first")
print(f"fills: {FILLS}")

# the sector map, so each fill can be matched to its sector's leader
try:
    # leadlag_screen keeps the exchange-derived sector map
    from leadlag_screen import SECTORS
except Exception as _e:
    # without it Question B cannot run, but Question A still can
    SECTORS = None
    print(f"  (could not import SECTORS from leadlag_screen: {_e!r} -- "
          f"Question B will be skipped)")

# RUN STAMP, YYYYMMDD_HHMM, on every output this script writes, so a re-run never
# collides with an earlier one and never has to refuse to write
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
# the round-trip fee, for reference only -- it is NOT the defensive hurdle
FEE_RT_BPS = 1.554
# leader-move thresholds to test, in bps of the leader's mid
JUMP_BPS = [5, 10, 20, 40]
# the window over which the leader's move is measured, milliseconds
JUMP_WINDOW_MS = 1000
# minimum fills in a cell before it is reported
MIN_FILLS = 200
# minimum days before a day-as-unit statistic is reported
MIN_DAYS = 10

con = duckdb.connect(); pd.set_option("display.width", 210, "display.max_columns", 40)
SNAP = f"{PARSED}/ob_snapshot/date=*/*.parquet"

# =============================================================================
# SECTION 0 -- what scale is obi_1 on? DO NOT ASSUME.
# =============================================================================
# fill_attribution.py tests `f[obi_col] > 0`, implying obi centred at ZERO.
# micro_mm's queue_skew_thresh is applied to |imb - 0.5|, implying centred at
# 0.5. Both cannot be true of the same column, and guessing which would silently
# invert the exposed/favourable split -- the single most important axis below.
q0 = f"""
SELECT COUNT(*) AS n,
       MIN(obi_1) AS lo, MAX(obi_1) AS hi,
       MEDIAN(obi_1) AS med,
       AVG(CASE WHEN obi_1 < 0 THEN 1.0 ELSE 0.0 END) AS frac_negative
FROM read_parquet('{FILLS}/**/*.parquet', hive_partitioning=1, union_by_name=1)
WHERE obi_1 IS NOT NULL
"""
o = con.execute(q0).df().iloc[0]
print("\nSECTION 0 -- the obi_1 scale, read from the data")
print(f"  n = {int(o.n):,} | min {o.lo:+.4f} | median {o.med:+.4f} | max {o.hi:+.4f}"
      f" | {100*o.frac_negative:.1f}% negative")
# a column with negatives is centred at 0; one bounded in [0,1] is centred at 0.5
if o.frac_negative > 0.01:
    # centred at zero: imbalance is already signed
    OBI_CENTRE = 0.0
    print("  --> obi_1 is CENTRED AT 0 (signed imbalance). "
          "Bid-heavy = positive.")
else:
    # bounded [0,1]: subtract 0.5 to sign it
    OBI_CENTRE = 0.5
    print("  --> obi_1 is CENTRED AT 0.5 (share of depth on the bid). "
          "Bid-heavy = above 0.5.")

# =============================================================================
# SECTION 1 -- QUESTION A: |markout| / capture by imbalance bucket and side
# =============================================================================
# THE SIDE SPLIT IS THE POINT. A bid-heavy book means price is likely to rise.
# Your resting ASK gets lifted -- you SOLD into a rise, which is the adverse
# fill. Your resting BID that trades is the favourable one. So:
#   EXPOSED    = sell fills when bid-heavy, buy fills when ask-heavy
#   FAVOURABLE = buy fills when bid-heavy, sell fills when ask-heavy
# The throttle only ever cuts the EXPOSED side, so the exposed-side ratio is the
# number that decides whether it belongs in the config.
q1 = f"""
WITH f AS (
  SELECT date, symbol, side, capture, markout,
         -- the signed imbalance, on whatever scale Section 0 detected
         obi_1 - {OBI_CENTRE} AS imb
  FROM read_parquet('{FILLS}/**/*.parquet', hive_partitioning=1, union_by_name=1)
  WHERE capture IS NOT NULL AND markout IS NOT NULL AND obi_1 IS NOT NULL
),
tagged AS (
  SELECT *,
    -- how far past balanced the book was, which is what the trigger measures
    ABS(imb) AS imb_abs,
    -- exposed when our fill side is the one the book leans AGAINST:
    -- bid-heavy (imb>0) and we SOLD (side<0), or ask-heavy and we BOUGHT
    CASE WHEN (imb > 0 AND side < 0) OR (imb < 0 AND side > 0)
         THEN 'exposed' ELSE 'favourable' END AS exposure
  FROM f
),
bucketed AS (
  SELECT *,
    -- the buckets straddle the 0.15 and 0.20 triggers already in production.
    -- SPLIT ABOVE 0.30 (2026-09-15): the first run put 82.6% of ALL fills into a
    -- single '0.30+' bucket, which made the whole interesting region one opaque
    -- number. obi_1 runs -1..+1, so |imb| reaches 1.0 and there is plenty of
    -- resolution to recover.
    CASE WHEN imb_abs < 0.05 THEN '0.00-0.05'
         WHEN imb_abs < 0.10 THEN '0.05-0.10'
         WHEN imb_abs < 0.15 THEN '0.10-0.15'
         WHEN imb_abs < 0.20 THEN '0.15-0.20'
         WHEN imb_abs < 0.30 THEN '0.20-0.30'
         WHEN imb_abs < 0.45 THEN '0.30-0.45'
         WHEN imb_abs < 0.60 THEN '0.45-0.60'
         WHEN imb_abs < 0.80 THEN '0.60-0.80'
         ELSE '0.80+' END AS bucket
  FROM tagged
)
SELECT bucket, exposure,
       COUNT(*) AS n_fills,
       COUNT(DISTINCT date) AS days,
       ROUND(AVG(capture), 4)  AS capture_bps,
       ROUND(AVG(markout), 4)  AS markout_bps,
       -- KEPT FOR REFERENCE ONLY, AND IT IS MISLEADING WHERE MARKOUT IS
       -- POSITIVE: the ABS() makes +0.57 and -0.57 read the same, so a state
       -- where the price moves IN YOUR FAVOUR is reported as though a third of
       -- capture were being lost. Read gross_bps instead.
       ROUND(ABS(AVG(markout)) / NULLIF(AVG(capture), 0), 3) AS mk_over_cap,
       -- THE HONEST NUMBER. capture + markout = what a share quoted in this
       -- state actually earns, with the sign of markout respected.
       ROUND(AVG(capture) + AVG(markout), 4) AS gross_bps
FROM bucketed
GROUP BY bucket, exposure
HAVING COUNT(*) >= {MIN_FILLS}
ORDER BY exposure, bucket
"""
a = con.execute(q1).df()
print("\nSECTION 1 -- QUESTION A: does cutting size in the triggered state help?")
print(a.to_string(index=False))
# pull out the exposed side at and above the production triggers
exp = a[a.exposure == "exposed"]
# FILL-WEIGHTED, NOT BUCKET-AVERAGED. Averaging the buckets equally gives the
# rarest state the same vote as the one carrying most of the book -- and on this
# data the top bucket holds the overwhelming majority of fills.
def _wgt(frame, col="gross_bps"):
    # weight each bucket by how many fills it actually contains
    if not len(frame) or frame.n_fills.sum() == 0:
        return float("nan")
    return float((frame[col] * frame.n_fills).sum() / frame.n_fills.sum())

print("\n  FILL DISTRIBUTION -- which states actually carry the book:")
tot = int(a.n_fills.sum())
for b in a.bucket.unique():
    nb = int(a[a.bucket == b].n_fills.sum())
    print(f"    imbalance {b:>9}: {nb:8,d} fills  ({100*nb/tot:5.1f}%)")

print("\n  THE DECIDING NUMBERS -- both sides, at and above the 0.15 trigger:")
# the favourable side too: the CUT and the BOOST are answered by the same
# ratio read in OPPOSITE directions
fav = a[a.exposure == "favourable"]
# every bucket at or above the 0.15 production trigger
HOT = ["0.15-0.20", "0.20-0.30", "0.30-0.45", "0.45-0.60", "0.60-0.80", "0.80+"]
hot = exp[exp.bucket.isin(HOT)]
hotf = fav[fav.bucket.isin(HOT)]
for lab, frame, lever in (("EXPOSED   ", hot, "cut"),
                          ("FAVOURABLE", hotf, "boost")):
    if not len(frame):
        print(f"    {lab}: no bucket cleared the fill floor")
        continue
    for _, r in frame.iterrows():
        # above 1.0 those shares lose money; below 1.0 they make money
        if lever == "cut":
            v = "LOSS-MAKING -> cutting is gain" if r.mk_over_cap > 1 \
                else "profitable -> cutting is LOSS"
        else:
            v = "profitable -> BOOSTING is gain" if r.mk_over_cap < 1 \
                else "loss-making -> boosting is LOSS"
        print(f"    {lab} {r.bucket:>9}: |markout|/capture = "
              f"{r.mk_over_cap:5.3f}  gross {r.gross_bps:+.4f} bps  ({v})")

# THE VERDICT, ON GROSS PER SHARE -- NOT on the |markout|/capture ratio.
# The first version of this block asked "do these shares LOSE money" (ratio > 1).
# That was the wrong question. On this book NOTHING loses money: every bucket has
# positive gross. The real question is ALLOCATIVE -- does a share quoted in this
# state earn LESS than the same share quoted on the other side? Quoting capacity
# and inventory headroom are the scarce things, not the sign of the P&L.
if len(hot) and len(hotf):
    # fill-weighted gross per share, per side, across the triggered region
    ge = _wgt(hot); gf = _wgt(hotf)
    # the quiet region, as the baseline the triggered region is judged against
    cool = a[~a.bucket.isin(HOT)]
    gce = _wgt(cool[cool.exposure == "exposed"])
    gcf = _wgt(cool[cool.exposure == "favourable"])
    print(f"\n  FILL-WEIGHTED GROSS PER SHARE, bps:")
    print(f"    quiet book  (imbalance < 0.15): exposed {gce:+.4f} | favourable {gcf:+.4f}")
    print(f"    lopsided    (imbalance >= 0.15): exposed {ge:+.4f} | favourable {gf:+.4f}")
    # how much each side degrades when the book goes lopsided
    print(f"    degradation moving to a lopsided book: "
          f"exposed {100*(ge/gce - 1):+.0f}% | favourable {100*(gf/gcf - 1):+.0f}%")
    # the number that decides the boost
    print(f"\n  a FAVOURABLE share in the triggered region earns "
          f"{gf/ge:.1f}x an EXPOSED one" if ge > 0 else "")

    print("\n  THE BOOST (size_boost_mult):")
    if gf > 0:
        print(f"      Favourable shares earn {gf:+.4f} bps gross in the triggered")
        print(f"      region. Adding size there ADDS P&L. The boost is justified.")
    else:
        print(f"      Favourable shares earn {gf:+.4f} bps -- boosting would add")
        print(f"      loss-making volume. Do not boost.")

    print("\n  THE CUT (obi_throttle):")
    if ge < 0:
        print(f"      Exposed shares LOSE {ge:+.4f} bps. Cutting is pure gain.")
    else:
        print(f"      Exposed shares still EARN {ge:+.4f} bps gross, so cutting")
        print(f"      them REMOVES P&L. The cut pays only if the inventory")
        print(f"      headroom it frees is redeployed at a better rate -- i.e.")
        print(f"      only if inventory capacity is the BINDING CONSTRAINT.")
        print(f"      This probe does not measure that. Check whether max_inv /")
        print(f"      soft_inv actually bind before enabling obi_throttle; if they")
        print(f"      do not, the throttle costs {abs(ge):.4f} bps per cut share")
        print(f"      for nothing.")

print("\n  Read: the throttle CUTS the exposed side, size_boost_mult BOOSTS the")
print("  favourable one, and both change QUANTITY only -- never price, never")
print("  queue position -- so neither spends the spread capture the lean spends.")
print("\n  CAVEAT ON THE BOOST. These ratios are measured at the CURRENT clip.")
print("  Per-share economics need not survive doubling it: a larger clip fills")
print("  less completely, sits longer, and is more exposed at the touch. A")
print("  positive ratio here justifies a SWEEP over size_boost_mult, not a")
print("  direct read of how much to boost. It also lengthens unwind time, so")
print("  A.4 capacity (KOIL 69.9 min, SGPL 100.2 min at 10% POV) must be redone")
print("  under any boost before it ships.")

# =============================================================================
# SECTION 2 -- QUESTION B: is a leader jump a usable defensive trigger?
# =============================================================================
if SECTORS is None:
    print("\nSECTION 2 -- skipped, no sector map")
else:
    # map every symbol to its sector's most-traded name, which is the leader
    # leadlag_screen picks leaders by traded value; reuse the same map here so
    # the two scripts cannot disagree about who leads what
    sym2sec = {s: sec for sec, names in SECTORS.items() for s in names}
    # the leader per sector, by traded value over the fill dates
    qL = f"""
    SELECT symbol, MEDIAN(v) AS med_v FROM (
      SELECT symbol, date, MAX(cum_value) AS v
      FROM read_parquet('{SNAP}', hive_partitioning=1)
      WHERE market='REG' GROUP BY symbol, date)
    GROUP BY symbol
    """
    tv = con.execute(qL).df().set_index("symbol")["med_v"].to_dict()
    # the top-traded name in each sector
    leaders = {}
    for sec, names in SECTORS.items():
        cand = {s: tv.get(s, 0.0) for s in names}
        if max(cand.values(), default=0) > 0:
            leaders[sec] = max(cand, key=cand.get)
    print(f"\nSECTION 2 -- QUESTION B: leader jump as a defensive trigger")
    print(f"  leaders: {leaders}")
    # the leader mid series on a 1-second grid, for every leader
    LEAD_SQL = ",".join(f"'{s}'" for s in set(leaders.values()))
    q2 = f"""
    WITH touch AS (
      SELECT s.date, s.symbol, s.orig_time,
             MAX(CASE WHEN s.entry_type='BID'   AND s.level=1 THEN s.px END) AS bb,
             MIN(CASE WHEN s.entry_type='OFFER' AND s.level=1 THEN s.px END) AS ba
      FROM read_parquet('{SNAP}') s
      WHERE s.market='REG' AND s.phase='CONTINUOUS_AUCTION'
        AND s.symbol IN ({LEAD_SQL})
      GROUP BY s.date, s.symbol, s.orig_time
    )
    -- one leader mid per SECOND: the last two-sided touch in that second
    SELECT date, symbol AS leader,
           CAST(epoch(orig_time) AS BIGINT) AS sec,
           LAST((bb+ba)/2 ORDER BY orig_time) AS lead_mid
    FROM touch WHERE bb>0 AND ba>0 AND ba>=bb
    GROUP BY date, symbol, sec
    """
    lm = con.execute(q2).df()
    print(f"  {len(lm):,} leader-seconds loaded")
    # the leader's move over the preceding window, in bps
    lm = lm.sort_values(["leader", "date", "sec"])
    # how many 1-second steps the window spans
    steps = max(1, JUMP_WINDOW_MS // 1000)
    # log return over the window, within a (leader, date) so no overnight gap
    lm["lead_move_bps"] = (np.log(lm.lead_mid)
                           .groupby([lm.leader, lm.date]).diff(steps) * 1e4)
    # the fills, with their own second and sector leader
    q3 = f"""
    SELECT date, symbol, side, markout, capture,
           CAST(ts / 1000 AS BIGINT) AS sec
    FROM read_parquet('{FILLS}/**/*.parquet', hive_partitioning=1, union_by_name=1)
    WHERE markout IS NOT NULL AND capture IS NOT NULL
    """
    fl = con.execute(q3).df()
    # attach each fill's sector leader
    fl["leader"] = fl.symbol.map(lambda s: leaders.get(sym2sec.get(s, ""), None))
    # a fill whose own name IS the leader has no external signal to react to
    fl = fl[fl.leader.notna() & (fl.leader != fl.symbol)]
    # dates as strings on both sides so the join cannot silently miss
    fl["date"] = fl.date.astype(str); lm["date"] = lm.date.astype(str)
    # attach the leader's preceding move to each fill
    j = fl.merge(lm[["date", "leader", "sec", "lead_move_bps"]],
                 on=["date", "leader", "sec"], how="inner").dropna(
                     subset=["lead_move_bps"])
    print(f"  {len(j):,} fills matched to a leader second "
          f"({j.date.nunique()} dates)")
    if len(j) >= MIN_FILLS and j.date.nunique() >= MIN_DAYS:
        # ADVERSE means the leader moved against the side we just took: it rose
        # while we were selling, or fell while we were buying
        j["adverse_move_bps"] = np.where(j.side > 0,
                                         -j.lead_move_bps, j.lead_move_bps)
        rows = []
        # each candidate threshold
        for X in JUMP_BPS:
            # fills taken while the leader was jumping against us
            hit = j[j.adverse_move_bps > X]
            # everything else
            rest = j[j.adverse_move_bps <= X]
            # both cells must be populated to compare
            if len(hit) < MIN_FILLS:
                continue
            # DAY AS UNIT: mean per day, then the paired difference across days.
            # Pooling fills would treat a busy day as many independent
            # observations and inflate every t-statistic.
            dh = hit.groupby("date").markout.mean()
            dr = rest.groupby("date").markout.mean()
            pair = pd.concat([dh.rename("hit"), dr.rename("rest")],
                             axis=1).dropna()
            # too few shared days for a paired statistic
            if len(pair) < MIN_DAYS:
                continue
            # the per-day difference in markout
            d = pair.hit - pair.rest
            # paired t across days
            t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) \
                if d.std(ddof=1) > 0 else np.nan
            rows.append({
                "jump_bps": X, "n_hit": len(hit),
                "pct_of_fills": round(100 * len(hit) / len(j), 2),
                "days": len(pair),
                "markout_hit": round(float(pair.hit.mean()), 4),
                "markout_rest": round(float(pair.rest.mean()), 4),
                "diff": round(float(d.mean()), 4),
                "t": round(t, 2),
                "days_worse": int((d < 0).sum())})
        res = pd.DataFrame(rows)
        print("\n  markout of fills taken WHILE the leader jumped against us, "
              "vs all other fills (day-as-unit):")
        if len(res):
            print(res.to_string(index=False))
            print("\n  Read: a NEGATIVE `diff` with |t| > 2 and most days worse means")
            print("  the leader jump really does mark bad fills, and pulling or")
            print("  shrinking the quote in that state avoids them. `pct_of_fills`")
            print("  is the dosage -- a trigger that fires on 30% of fills is not a")
            print("  tail trigger, it is a regime change, and it will cost capture")
            print("  everywhere. The lean's own tradeoff is the benchmark: it bought")
            print("  markout +1.08 bps for capture -0.15 going from 0 to 2 ticks.")
            # ---- the picture ---------------------------------------------
            INK, BAR, WARN, GRID = "#1f2933", "#2f6fb5", "#b3261e", "#dfe3e8"
            fig, (p1, p2) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor="white")
            # panel 1: Question A -- the ratio that decides the size lever
            for ex, mk, col in (("exposed", "o-", WARN),
                                ("favourable", "s--", BAR)):
                sub = a[a.exposure == ex]
                if len(sub):
                    p1.plot(range(len(sub)), sub.mk_over_cap, mk, color=col,
                            lw=2, ms=7, label=ex)
                    p1.set_xticks(range(len(sub)))
                    p1.set_xticklabels(sub.bucket, rotation=30, ha="right")
            p1.axhline(1.0, color=INK, lw=1.8, ls=":")
            p1.annotate("above 1.0 = those shares lose money", xy=(0, 1.0),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=9.5, color=INK, weight="bold")
            p1.set_ylabel("|markout| / capture", fontsize=10.5, color=INK)
            p1.set_xlabel("book imbalance at the fill", fontsize=10.5, color=INK)
            p1.set_title("Q(A): should the lean also cut size?\n"
                         "the throttle only cuts the exposed side",
                         fontsize=11.5, color=INK, loc="left")
            p1.legend(fontsize=9.5, frameon=False)
            p1.grid(True, color=GRID, lw=.8); p1.set_axisbelow(True)
            for s in ("top", "right"): p1.spines[s].set_visible(False)
            # panel 2: Question B -- markout in the triggered state
            p2.plot(res.jump_bps, res.markout_hit, "o-", color=WARN, lw=2,
                    ms=7, label="leader jumped against us")
            p2.plot(res.jump_bps, res.markout_rest, "s--", color=BAR, lw=2,
                    ms=7, label="all other fills")
            p2.set_xlabel(f"leader move over {JUMP_WINDOW_MS} ms, bps",
                          fontsize=10.5, color=INK)
            p2.set_ylabel("mean markout, bps", fontsize=10.5, color=INK)
            p2.set_title("Q(B): does a leader jump mark bad fills?\n"
                         "lower is worse", fontsize=11.5, color=INK, loc="left")
            p2.legend(fontsize=9.5, frameon=False)
            p2.grid(True, color=GRID, lw=.8); p2.set_axisbelow(True)
            for s in ("top", "right"): p2.spines[s].set_visible(False)
            plt.tight_layout()
            # timestamped: every run keeps its own figure, none is overwritten,
            # and a re-run never has to refuse to write
            out = RESULTS / f"conditional_markout_{STAMP}.png"
            plt.savefig(out, dpi=150, facecolor="white")
            print(f"\n  wrote {out}")
        else:
            print("  no threshold produced enough fills and days to compare")
    else:
        print(f"  too few matched fills or days ({len(j)} fills, "
              f"{j.date.nunique()} dates) -- need {MIN_FILLS} and {MIN_DAYS}")
