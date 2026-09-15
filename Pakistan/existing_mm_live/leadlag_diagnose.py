# leadlag_diagnose.py -- did ANY pair survive the acceptance rule, and if not,
# which condition killed it?
#
# WHY THIS EXISTS. leadlag_screen.py prints 157 rows and an acceptance rule in
# prose. Reading 157 rows against five simultaneous conditions by eye is exactly
# how a table gets mined for its best t-statistic. This applies the rule
# mechanically, reports the funnel, and draws the three pictures that say whether
# the screen found structure or noise.
#
# THE ACCEPTANCE RULE, as leadlag_screen.py states it. ALL must hold:
#   1. leader_faster            the leader updates faster than the follower
#   2. |peak_lag_mean| > peak_lag_se   the lag clears its own day-as-unit error bar
#   3. peak_lag_mean > 0        the named leader leads, rather than trails
#   4. frac_leader_leads > 0.7  the sign is stable across days, not a coin flip
#   5. peak_corr_mean > EPPS_MULT * corr0_mean   the peak is a real peak, not the
#                               contemporaneous correlation reappearing at a lag
#                               because the two legs trade at different rates
#   6. ind_bps_median > FEE_BPS the anticipatable move covers the round trip
#
# THE MULTIPLE-TESTING ARITHMETIC, which is the whole point. With N pairs tested
# over D days, a pair whose lag sign is pure coin-flip still shows a perfectly
# stable sign (D of D) with probability 2 * 0.5^D. At D=5 that is 6.25%, so out
# of 157 pairs roughly TEN will look perfectly stable having no relationship at
# all. Section 3 prints the expected count beside the observed one. If they
# match, "frac_leader_leads = 1.00" is not evidence of anything.
#
# Read-only, seconds. Run: caffeinate -is python leadlag_diagnose.py
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- PATHS from config_pk; never a literal in this file ----------------------
try:
    # the project's central path module
    from config_pk import RESULTS_ROOT
except Exception as _e:
    # fail naming the fix rather than anywhere later
    raise SystemExit("leadlag_diagnose: could not import RESULTS_ROOT from "
                     "config_pk. Run from existing_mm_live/. Original: %r" % _e)

# RUN STAMP, YYYYMMDD_HHMM. Every output this script writes carries it, so a
# re-run NEVER collides with an earlier one and never has to refuse to write.
# The earlier "do not overwrite" guard was the wrong shape: it protected old runs
# by blocking new ones. A timestamp protects both.
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
# the screen's output
SCREEN = Path(RESULTS_ROOT) / "leadlag" / "leadlag_screen.parquet"
# the round-trip fee the anticipatable move has to clear, bps
FEE_BPS = 1.554
# stability threshold on the sign of the lag
FRAC_GATE = 0.70
# how much bigger the peak correlation must be than the correlation at lag zero
# before the peak is treated as real rather than an Epps artifact
EPPS_MULT = 2.0
# the outer points of the lag grid in leadlag_screen.py -- peaks landing here are
# the search hitting its own boundary, not finding a maximum.
# NOTE: the LOCATION of the peak is not evidence either way on this venue. PSX is
# quoted by people, so a lead of several seconds is the expected scale, not a
# warning sign. Only a peak pinned to the grid EDGE is diagnostic, because that
# means the search never found an interior maximum.
GRID_EDGE_MS = 30000

# fail with the path named
if not SCREEN.exists():
    raise SystemExit(f"screen output not found: {SCREEN}\n"
                     "Run leadlag_screen.py --smoke or --run first.")
df = pd.read_parquet(SCREEN)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(f"read {SCREEN}")
print(f"  {len(df)} pairs, {df.n_days.median():.0f} median days per pair, "
      f"{df.sector.nunique()} sectors")

# =============================================================================
# SECTION 1 -- the acceptance funnel, applied mechanically
# =============================================================================
# each condition as its own boolean column, so the funnel can attribute failures
df["c1_faster"] = df.leader_faster.astype(bool)
df["c2_clears_se"] = df.peak_lag_mean.abs() > df.peak_lag_se
df["c3_positive"] = df.peak_lag_mean > 0
df["c4_stable"] = df.frac_leader_leads > FRAC_GATE
# the Epps guard: a peak that is merely the lag-zero correlation reappearing is
# not a lead-lag. Require the peak to be positive AND materially bigger.
df["c5_real_peak"] = (df.peak_corr_mean > 0) & \
                     (df.peak_corr_mean > EPPS_MULT * df.corr0_mean.abs())
df["c6_clears_fee"] = df.ind_bps_median > FEE_BPS
# the six gates in the order the rule states them
GATES = ["c1_faster", "c2_clears_se", "c3_positive", "c4_stable",
         "c5_real_peak", "c6_clears_fee"]
# human labels for the printout
LABEL = {"c1_faster": "leader updates faster",
         "c2_clears_se": "|lag| clears its own SE",
         "c3_positive": "lag is POSITIVE (leader leads)",
         "c4_stable": f"sign stable > {FRAC_GATE:.0%} of days",
         "c5_real_peak": f"peak corr > {EPPS_MULT:.0f}x |corr at lag 0|",
         "c6_clears_fee": f"anticipatable bps > {FEE_BPS} fee"}

print("\nSECTION 1 -- acceptance funnel")
print(f"  {'condition':<38} {'passes alone':>13} {'cumulative':>11}")
# cumulative mask, narrowed one gate at a time
cum = pd.Series(True, index=df.index)
for g in GATES:
    cum = cum & df[g]
    print(f"  {LABEL[g]:<38} {int(df[g].sum()):>13} {int(cum.sum()):>11}")
# the survivors, if any
surv = df[cum]
print(f"\n  SURVIVORS: {len(surv)} of {len(df)} pairs")
if len(surv):
    print(surv[["sector", "leader", "leader_rank", "follower", "n_days",
                "peak_lag_mean", "peak_lag_se", "frac_leader_leads",
                "peak_corr_mean", "corr0_mean", "ind_bps_median"]]
          .to_string(index=False))
else:
    # say which single gate is doing the killing, since that decides what to do next
    print("  Nothing survived. Single-gate pass rates say where the wall is:")
    for g in GATES:
        print(f"    {LABEL[g]:<38} {100*df[g].mean():5.1f}% of pairs")

# =============================================================================
# SECTION 2 -- is the economic magnitude there at all?
# =============================================================================
# Sample size fixes standard errors. It does NOT raise a correlation or a bps
# figure -- those are means, not error bars. So separate the two kinds of
# failure: the ones more days would cure, and the ones they would not.
print("\nSECTION 2 -- what more days would and would not fix")
print(f"  peak_corr_mean:  median {df.peak_corr_mean.median():+.4f} | "
      f"max {df.peak_corr_mean.max():+.4f} | "
      f"{100*(df.peak_corr_mean < 0).mean():.0f}% are NEGATIVE")
print(f"  ind_bps_median:  median {df.ind_bps_median.median():.3f} bps | "
      f"max {df.ind_bps_median.max():.3f} bps | fee is {FEE_BPS} bps")
print(f"  pairs clearing the fee: {int(df.c6_clears_fee.sum())} of {len(df)}")
print("\n  MORE DAYS SHRINK peak_lag_se (as 1/sqrt(days)) and would let more")
print("  pairs clear condition 2. MORE DAYS DO NOT RAISE peak_corr_mean or")
print("  ind_bps_median. If those two are the binding gates, a longer run")
print("  changes the error bars and not the conclusion.")

# =============================================================================
# SECTION 3 -- how many "stable" pairs would appear from pure chance?
# =============================================================================
print("\nSECTION 3 -- multiple testing")
# the modal number of days, which sets the coin-flip probability
D = int(df.n_days.median())
# probability a coin-flip sign comes out perfectly stable in either direction
p_perfect = 2.0 * (0.5 ** D)
# how many such pairs chance alone produces across this many tests
exp_perfect = len(df) * p_perfect
# how many were actually observed
obs_perfect = int((df.frac_leader_leads.isin([0.0, 1.0])).sum())
print(f"  {len(df)} pairs tested over {D} days each")
print(f"  P(a coin-flip sign looks perfectly stable over {D} days) = {p_perfect:.4f}")
print(f"  EXPECTED perfectly-stable pairs from chance alone : {exp_perfect:.1f}")
print(f"  OBSERVED perfectly-stable pairs                   : {obs_perfect}")
# the direct test of the size prior, #1 vs #2 within a sector
tops = set(df[df.leader_rank == 1].leader) | set(df[df.leader_rank == 2].leader)
t2 = df[(df.leader_rank == 1) & (df.follower.isin(tops))]
if len(t2):
    # how often the largest name led the second largest
    wins = int((t2.peak_lag_mean > 0).sum()); n = len(t2)
    # exact binomial tail under a fair coin, computed not quoted
    from math import comb
    pval = sum(comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    print(f"\n  SIZE PRIOR: the #1 name leads its #2 in {wins} of {n} sectors")
    print(f"  under a fair coin you would expect {n/2:.1f}; one-sided p = {pval:.3f}")
    print("  A prior is only supported if this is small. It is a PRIOR, not a finding.")

# =============================================================================
# SECTION 4 -- the pictures
# =============================================================================
INK, MUTED, BAR, WARN, GRID = "#1f2933", "#7b8794", "#2f6fb5", "#b3261e", "#dfe3e8"
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), facecolor="white")

# --- panel 1: is the peak a real peak, or the lag-zero correlation again? -----
a = axes[0]
# every pair as a point: contemporaneous correlation against peak correlation
a.scatter(df.corr0_mean, df.peak_corr_mean, s=26, color=BAR, alpha=.6,
          edgecolor="white", linewidth=.6, zorder=3)
# the 45-degree line: on it, the "peak" is just the lag-zero value
lim = float(np.nanmax(np.abs(np.r_[df.corr0_mean.values, df.peak_corr_mean.values]))) * 1.1
a.plot([-lim, lim], [-lim, lim], ls="--", lw=1.4, color=MUTED, zorder=2)
# the Epps gate
a.plot([0, lim], [0, EPPS_MULT * lim], ls=":", lw=1.6, color=WARN, zorder=2)
a.axhline(0, color=GRID, lw=1); a.axvline(0, color=GRID, lw=1)
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
a.set_xlabel("correlation at lag 0", fontsize=10, color=INK)
a.set_ylabel("correlation at the peak lag", fontsize=10, color=INK)
a.set_title("A real lead-lag sits ABOVE the dashed line\n"
            "points on it are the Epps artifact", fontsize=11.5, color=INK, loc="left")
a.grid(True, color=GRID, lw=.8); a.set_axisbelow(True)
for s in ("top", "right"): a.spines[s].set_visible(False)

# --- panel 2: where do the peak lags land? -----------------------------------
b = axes[1]
# the distribution of the fitted peak lag
b.hist(df.peak_lag_mean, bins=28, color=BAR, alpha=.8, edgecolor="white")
# the grid boundary: peaks piling up here mean the search hit its own edge
b.axvline(GRID_EDGE_MS, color=WARN, lw=2); b.axvline(-GRID_EDGE_MS, color=WARN, lw=2)
b.axvline(0, color=MUTED, lw=1.4, ls="--")
b.set_xlabel("fitted peak lag, ms  (+ = named leader leads)", fontsize=10, color=INK)
b.set_ylabel("pairs", fontsize=10, color=INK)
b.set_title("Where the fitted peak lands\n"
            "PSX is quoted by people, so SECONDS is the expected scale",
            fontsize=11.5, color=INK, loc="left")
b.grid(True, color=GRID, lw=.8, axis="y"); b.set_axisbelow(True)
for s in ("top", "right"): b.spines[s].set_visible(False)

# --- panel 3: does the move pay for the round trip? --------------------------
c = axes[2]
# sorted so the shape of the distribution is readable
v = df.ind_bps_median.sort_values().values
c.plot(np.arange(len(v)), v, lw=2, color=BAR)
# the hurdle
c.axhline(FEE_BPS, color=WARN, lw=2)
c.annotate(f"round-trip fee {FEE_BPS} bps", xy=(0, FEE_BPS), xytext=(4, 5),
           textcoords="offset points", fontsize=9.5, color=WARN, weight="bold")
c.set_xlabel("pairs, sorted", fontsize=10, color=INK)
c.set_ylabel("anticipatable move, bps", fontsize=10, color=INK)
c.set_title(f"{int(df.c6_clears_fee.sum())} of {len(df)} pairs clear the fee\n"
            "before any of the other four conditions",
            fontsize=11.5, color=INK, loc="left")
c.grid(True, color=GRID, lw=.8); c.set_axisbelow(True)
for s in ("top", "right"): c.spines[s].set_visible(False)

plt.tight_layout()
# timestamped, so every run keeps its own figure and none is ever overwritten
out = Path(RESULTS_ROOT) / "leadlag" / f"leadlag_diagnose_{STAMP}.png"
plt.savefig(out, dpi=150, facecolor="white")
print(f"\nwrote {out}")
