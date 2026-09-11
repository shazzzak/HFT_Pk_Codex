# Demonstration: reproduce the "-0.044 instead of 0.000" self-test artifact on
# synthetic data with KNOWN ground truth, then show the rank-consistent fix zeros it.
# Ground truth by construction: L1's incremental information over ITSELF is exactly 0.

# numerical arrays
import numpy as np
# rankdata, spearmanr, and a t-distribution for genuinely fat tails
from scipy import stats
# headless plotting
import matplotlib
# force a non-interactive backend so this runs without a display
matplotlib.use("Agg")
# plotting api
import matplotlib.pyplot as plt

# fixed seed so the numbers are reproducible run-to-run
rng = np.random.default_rng(20260911)
# large-ish sample so finite-sample noise is small relative to any real artifact
n = 200_000

# --- Build fat-tailed, NON-LINEARLY related data (mimics OBI vs forward markout) ---
# L1 OBI proxy: Student-t with 4 dof -> heavy tails, symmetric, mean 0
x = stats.t.rvs(df=4, size=n, random_state=rng)
# heavy-tailed noise on the target, independent of x, and dominant (weak signal, like PSX markout)
eps = stats.t.rvs(df=4, size=n, random_state=rng)
# forward markout f: MOSTLY LINEAR plus a MILD *asymmetric* monotone nonlinearity
# (softplus: near-linear for x>0, flattens for x<0 -> not symmetric about 0), swamped by
# fat-tailed noise. Asymmetry is essential: a symmetric/odd nonlinearity self-cancels in
# rank space and leaves ~0. Real OBI->markout response is asymmetric + heteroskedastic,
# which is why the surviving Spearman(x, resid) artifact is small-but-nonzero.
f = 0.15 * x + 1.4 * np.log1p(np.exp(1.3 * x)) + 1.6 * eps

# ---------------------------------------------------------------------------
# METHOD A (BUGGY, as described): OLS-residualize f on x, then Spearman(x, resid)
# For the self-test the candidate signal s IS L1 itself, i.e. s = x.
# ---------------------------------------------------------------------------
# design matrix [1, x] for an OLS fit WITH intercept (intercept is required for the
# orthogonality guarantee to hold)
A = np.column_stack([np.ones(n), x])
# least-squares solve for [intercept, slope]
coef, *_ = np.linalg.lstsq(A, f, rcond=None)
# fitted values a + b*x
fhat = A @ coef
# OLS residual of f on x
resid = f - fhat

# SANITY 1: OLS guarantees Pearson(x, resid) == 0 (up to float error)
pearson_x_resid = np.corrcoef(x, resid)[0, 1]
# THE BUGGY SELF-TEST: Spearman between L1 (x) and its own OLS residual
spearman_x_resid = stats.spearmanr(x, resid).statistic

# ---------------------------------------------------------------------------
# METHOD B (FIX, rank-consistent PART correlation): rank-transform first, then
# residualize IN RANK SPACE, and evaluate IN RANK SPACE (Pearson on ranks).
# Residualizer and evaluator now live in the same inner-product space.
# ---------------------------------------------------------------------------
# rank-transform x and f (average ranks handle ties)
xr = stats.rankdata(x)
fr = stats.rankdata(f)
# center ranks (equivalent to fitting an intercept in rank space)
xr_c = xr - xr.mean()
fr_c = fr - fr.mean()
# OLS residual of RANK(f) on RANK(x)
b_rank = (xr_c @ fr_c) / (xr_c @ xr_c)
fr_resid = fr_c - b_rank * xr_c
# rank-consistent self-test: Pearson between RANK(x) and the RANK residual of f
# (this is the correctly-specified analogue of the buggy Spearman(x, resid))
partSpearman_selftest = np.corrcoef(xr_c, fr_resid)[0, 1]

# ---------------------------------------------------------------------------
# METHOD C (FIX, FULL partial-Spearman): residualize BOTH sides in rank space.
# For the self-test the candidate w == x, so its rank residual is the ZERO vector
# and incremental info is exactly 0 BY CONSTRUCTION (no data needed).
# ---------------------------------------------------------------------------
# candidate is L1 itself for the self-test
wr = xr.copy()
wr_c = wr - wr.mean()
# rank residual of candidate on x -> identically zero when candidate == x
b_w = (xr_c @ wr_c) / (xr_c @ xr_c)
wr_resid = wr_c - b_w * xr_c
# guard the 0/0: if the candidate has no variance left after residualizing, info is 0
if np.std(wr_resid) < 1e-12:
    fullPartial_selftest = 0.0
else:
    fullPartial_selftest = np.corrcoef(fr_resid, wr_resid)[0, 1]

# ---------------------------------------------------------------------------
print("=== SELF-TEST: L1's incremental info over ITSELF (ground truth = 0.000) ===")
print(f"[sanity] Pearson(x, OLS resid)              = {pearson_x_resid:+.6f}   (OLS forces ~0)")
print(f"[BUGGY ] Spearman(x, OLS resid)             = {spearman_x_resid:+.6f}   <-- your -0.044 artifact")
print(f"[FIX B ] part-Spearman, rank-consistent     = {partSpearman_selftest:+.6f}   (~0 as it must)")
print(f"[FIX C ] full partial-Spearman (both sides) = {fullPartial_selftest:+.6f}   (0 by construction)")

# ---------------------------------------------------------------------------
# VISUAL EVIDENCE
# ---------------------------------------------------------------------------
# three-panel figure
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# Panel 1: x vs OLS residual. Pearson slope is flat (=0) but the CLOUD is not
# rank-independent of x -> that surviving monotone structure is the artifact.
# subsample for a legible scatter
idx = rng.choice(n, size=6000, replace=False)
ax[0].scatter(x[idx], resid[idx], s=4, alpha=0.25, color="#2b6cb0")
# bin x and plot the MEDIAN residual per bin to reveal the surviving monotone drift
bins = np.quantile(x, np.linspace(0, 1, 26))
centers = 0.5 * (bins[1:] + bins[:-1])
which = np.clip(np.digitize(x, bins) - 1, 0, len(centers) - 1)
med = np.array([np.median(resid[which == k]) if np.any(which == k) else np.nan
                for k in range(len(centers))])
ax[0].plot(centers, med, color="#c53030", lw=2.2, label="median resid | x")
ax[0].axhline(0, color="k", lw=0.8, ls="--")
ax[0].set_xlim(np.quantile(x, 0.01), np.quantile(x, 0.99))
ax[0].set_title(f"x vs OLS residual\nPearson={pearson_x_resid:+.4f} (flat) but Spearman={spearman_x_resid:+.4f}")
ax[0].set_xlabel("L1 OBI (x)"); ax[0].set_ylabel("OLS residual of f on x"); ax[0].legend(fontsize=8)

# Panel 2: histogram of the residual, showing the fat tails OLS leaves behind
ax[1].hist(resid, bins=200, color="#38a169", alpha=0.85,
           range=(np.quantile(resid, 0.005), np.quantile(resid, 0.995)))
ax[1].set_title("OLS residual distribution (fat-tailed)\nnon-normality is why rank != linear here")
ax[1].set_xlabel("residual"); ax[1].set_ylabel("count")

# Panel 3: the four self-test numbers side by side against the true zero
labels = ["Pearson\n(sanity)", "BUGGY\nSpearman", "FIX B\npart-Spearman", "FIX C\nfull partial"]
vals = [pearson_x_resid, spearman_x_resid, partSpearman_selftest, fullPartial_selftest]
colors = ["#a0aec0", "#c53030", "#2f855a", "#2b6cb0"]
ax[2].bar(labels, vals, color=colors)
ax[2].axhline(0, color="k", lw=1.0)
ax[2].set_title("Self-test value vs ground truth (0.000)")
ax[2].set_ylabel("self-test correlation")
for i, v in enumerate(vals):
    ax[2].text(i, v + (0.004 if v >= 0 else -0.004), f"{v:+.4f}",
               ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

# tidy and save
fig.tight_layout()
fig.savefig("/home/claude/selftest_artifact.png", dpi=130)
print("\nsaved /home/claude/selftest_artifact.png")
