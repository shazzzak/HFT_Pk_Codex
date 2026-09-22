# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# =============================================================================
# stage0_audit.py  (v2 -- bucket-aware)
# QBPS_2 pre-deploy audit + universe-expansion screen test
# =============================================================================
# Reads ONLY artifacts already in the Results dir. No engine run, no parsed
# store. Runtime: seconds. Writes a text report + 4 PNGs to the Results dir.
#
# v2 FIX -- THE BUCKET TRAP
# -------------------------
# fullyear_confirm_DAILY / _PERNAME are keyed by (date, config, bucket) and
# (date, symbol, config, bucket). v1 treated each ROW as a day, so "n_days"
# read 985 = 197 days x 5 buckets, and every statistic in PART A was wrong.
#
# Worse: summing net_pkr over ALL bucket rows gives EXACTLY 2.0000x the run
# summary total, on all six configs. That is the signature of four component
# buckets plus a fifth AGGREGATE row in the same column. This script does not
# assume that -- it TESTS it (find_aggregate_bucket) and then hard-asserts the
# collapsed total against the run summary CSV before reporting anything.
#
# WHAT IT ANSWERS
# ---------------
#   A1/A2. Day-as-unit statistics and the paired QBPS_2-vs-OBI test, on
#          correctly collapsed days, with explicit reconciliation.
#   A3.    QBPS_2 converts a 2 bps skew to round(0.02 * price) WHOLE TICKS at
#          PSX's flat 0.01 PKR tick. Below ~Rs 25 that rounds to ZERO ticks, so
#          the arm is byte-identical to OBI on those names and MUST show a
#          per-day difference of exactly 0.000.
#   A4.    The engine's own tradeability gate (capture_pkr <= 0, or
#          |markout|/capture > 0.7) recomputed from PERNAME under QBPS_2.
#   B.     Does the model-free 491-name screen PREDICT the engine? A pre-run
#          against the LEGACY universe_ranked results gave net5_trec_median
#          rho = -0.300 vs engine P&L (n=38) -- i.e. anti-predictive. This
#          re-runs it under the current QBPS_2 results to confirm or refute.
# =============================================================================

# filesystem path handling
from pathlib import Path
# glob so timestamped run artifacts are found without hardcoding a timestamp
import glob
# dataframes
import pandas as pd
# numerics
import numpy as np
# rank correlation and the t distribution
from scipy import stats
# plotting with a non-interactive backend so it never opens a window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
# the one results directory every output goes to (house rule: never relative)
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS = Path(str(_hft_paths.RESULTS_ROOT))
# per-side all-in TREC fee from mm_backtest.FEE_TOTAL_TREC
# = LAGA 0.000035 + SECP 0.0000065 + IPF 0.0000062 + clearing 0.00003
FEE_PCT_PER_SIDE = 0.0000777
# that fee in bps per side (0.777)
FEE_BPS_SIDE = FEE_PCT_PER_SIDE * 1e4
# round-trip fee in bps (1.554) -- what fill_attribution.fee_bps_roundtrip() charges
FEE_BPS_RT = 2.0 * FEE_BPS_SIDE
# PSX equity tick: flat one paisa at every price level
TICK = 0.01
# the queue skew under audit, in bps of price
SKEW_BPS = 2.0
# trading days per year for annualising a daily Sharpe
ANN = 252.0
# the control arm label, exactly as it appears in the 'throttle' column
CTRL = "OBI"
# the deploy arm label under audit
DEPLOY = "QBPS_2"
# the engine's tradeability gate: exclude when |markout|/capture exceeds this
GATE_MKO_OVER_CAP = 0.7
# candidate liquidity gates for the universe screen: (min notional M PKR, max spread bps)
GATE_GRID = [(200, 15), (150, 18), (100, 20), (100, 25), (50, 30), (25, 40)]

# -----------------------------------------------------------------------------
# WRITE SAFETY -- this script NEVER deletes or overwrites anything.
# Every output carries a run timestamp (matching the house naming convention
# already used throughout the Results dir), and safe_out() refuses to hand back
# a path that already exists rather than clobbering it.
# -----------------------------------------------------------------------------
# stamp for this run, e.g. 20260911_2145
RUN_TS = pd.Timestamp.now().strftime("%Y%m%d_%H%M")


def safe_out(stem, ext):
    # build the timestamped filename
    p = RESULTS / f"{stem}_{RUN_TS}.{ext}"
    # if that exact name somehow exists, walk a numeric suffix rather than overwrite
    n = 1
    while p.exists():
        # append a disambiguating counter
        p = RESULTS / f"{stem}_{RUN_TS}_{n}.{ext}"
        # advance the counter
        n += 1
    # hand back a path guaranteed not to exist
    return p


# every line printed is also written to the report file
_REPORT = []


def say(*parts):
    # join the parts into a single line
    line = " ".join(str(p) for p in parts)
    # keep it for the report file
    _REPORT.append(line)
    # echo it to the console
    print(line)


def newest(pattern):
    # expand the glob against the results directory
    hits = sorted(glob.glob(str(RESULTS / pattern)))
    # return the lexicographically last match (timestamps sort correctly) or None
    return Path(hits[-1]) if hits else None


def pick(df, candidates, what):
    # build a case-insensitive lookup of the frame's real column names
    lower = {c.lower(): c for c in df.columns}
    # exact case-insensitive match first
    for cand in candidates:
        if cand.lower() in lower:
            # report the choice so no column is ever silently guessed
            say(f"    [{what}] -> '{lower[cand.lower()]}'")
            return lower[cand.lower()]
    # substring fallback
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                say(f"    [{what}] -> '{c}' (fuzzy on '{cand}')")
                return c
    # nothing matched: say so rather than inventing a column
    say(f"    [{what}] -> NOT FOUND in {list(df.columns)}")
    return None


# -----------------------------------------------------------------------------
# THE BUCKET RESOLVER -- the whole point of v2
# -----------------------------------------------------------------------------
def find_aggregate_bucket(df, cfg_col, bucket_col, value_col):
    """Identify a bucket whose value equals the sum of the OTHER buckets.

    Returns (aggregate_bucket_name_or_None, diagnostic_table).

    Rationale: if one bucket is a pre-summed total sitting in the same column
    as its own components, then summing every bucket double-counts. Rather than
    assume which label that is, test every candidate arithmetically.
    """
    # per config x bucket totals of the value column
    tot = df.groupby([cfg_col, bucket_col])[value_col].sum().unstack(bucket_col)
    # every bucket label present
    buckets = list(tot.columns)
    # collect the test result for each candidate
    verdicts = {}
    # test each bucket as the possible aggregate
    for b in buckets:
        # the sum of all the OTHER buckets, per config
        others = tot.drop(columns=[b]).sum(axis=1)
        # relative difference between this bucket and that sum, per config
        rel = (tot[b] - others).abs() / others.abs().replace(0, np.nan)
        # it is the aggregate if it matches the sum of the rest on every config
        verdicts[b] = float(rel.max()) if len(rel.dropna()) else np.nan
    # the best candidate is the one with the smallest worst-case relative gap
    best = min(verdicts, key=lambda k: (np.inf if np.isnan(verdicts[k]) else verdicts[k]))
    # accept it only if it matches to within a tight tolerance
    agg = best if (np.isfinite(verdicts[best]) and verdicts[best] < 1e-6) else None
    # hand back the verdict and the diagnostic table
    return agg, tot, verdicts


def collapse_buckets(df, keys, bucket_col, component_buckets):
    """Collapse bucket rows to one row per key, correctly.

    net_pkr and opened_notional and fills SUM over component buckets.
    net_bps is RECOMPUTED notional-weighted, never averaged -- averaging bps
    across buckets of different size is the classic weighting error.
    """
    # keep only the component buckets, dropping any aggregate row
    d = df[df[bucket_col].isin(component_buckets)]
    # the additive columns present in this frame
    add_cols = [c for c in ["net_pkr", "opened_notional", "fills",
                            "capture_pkr", "markout_pkr", "liq_pkr", "fee_pkr"]
                if c in d.columns]
    # sum them per key
    g = d.groupby(keys, as_index=False)[add_cols].sum()
    # recompute net_bps from the summed PKR and summed notional
    g["net_bps"] = np.where(g["opened_notional"] > 0,
                            1e4 * g["net_pkr"] / g["opened_notional"], np.nan)
    # hand back the collapsed frame
    return g


def sharpe(x):
    # sample standard deviation, ddof=1
    sd = np.std(x, ddof=1)
    # guard a degenerate series
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    # daily mean over daily sd, scaled by root-time
    return float(np.mean(x) / sd * np.sqrt(ANN))


def sortino(x):
    # keep only the negative part of each daily observation
    downside = np.minimum(np.asarray(x, dtype=float), 0.0)
    # root-mean-square of the downside part (variants differ; this is RMS about 0)
    dd = np.sqrt(np.mean(downside ** 2))
    # guard a series with no losing days
    if dd == 0 or not np.isfinite(dd):
        return np.nan
    # daily mean over downside deviation, scaled by root-time
    return float(np.mean(x) / dd * np.sqrt(ANN))


def max_drawdown(pkr):
    # running cumulative P&L in PKR
    cum = np.cumsum(np.asarray(pkr, dtype=float))
    # running high-water mark
    peak = np.maximum.accumulate(cum)
    # drawdown at each point, always <= 0
    dd = cum - peak
    # the worst of them
    return float(np.min(dd)) if len(dd) else np.nan


# =============================================================================
# PART A -- QBPS_2 PRE-DEPLOY AUDIT
# =============================================================================
say("=" * 78)
say("PART A  --  QBPS_2 PRE-DEPLOY AUDIT  (bucket-aware)")
say("=" * 78)

# newest artifacts from the full-year confirmation run
daily_path = newest("fullyear_confirm_DAILY_*.parquet")
pername_path = newest("fullyear_confirm_PERNAME_*.parquet")
summary_path = newest("fullyear_confirm_2*.csv")
# report which files this run actually used
say(f"DAILY   : {daily_path}")
say(f"PERNAME : {pername_path}")
say(f"SUMMARY : {summary_path}")

# load the day x bucket frame
daily = pd.read_parquet(daily_path)
# load the pooled run summary -- this is the reconciliation target
summ = pd.read_csv(summary_path)
# report shape
say(f"\nDAILY raw shape={daily.shape}   (rows are date x config x BUCKET, not days)")

# -----------------------------------------------------------------------------
# A0 -- resolve the bucket structure BEFORE computing anything
# -----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("A0. BUCKET STRUCTURE  (must be resolved before any statistic is valid)")
say("-" * 78)
# the config arm column is named 'throttle' in these artifacts
c_cfg = "throttle"
# inventory the bucket labels and their row counts
say("bucket row counts:")
say(daily.groupby("bucket").size().to_string())
# distinct dates and configs, so the row arithmetic is visible
say(f"\ndistinct dates={daily['date'].nunique()}  configs={daily[c_cfg].nunique()}  "
    f"buckets={daily['bucket'].nunique()}")
say(f"rows = {daily['date'].nunique()} x {daily[c_cfg].nunique()} x "
    f"{daily['bucket'].nunique()} = {daily['date'].nunique() * daily[c_cfg].nunique() * daily['bucket'].nunique()}")

# test arithmetically whether one bucket is a pre-summed aggregate of the others
agg_bucket, bucket_tot, verdicts = find_aggregate_bucket(daily, c_cfg, "bucket", "net_pkr")
# show the per-config, per-bucket net_pkr totals
say("\nnet_pkr by config x bucket:")
say(bucket_tot.to_string(float_format=lambda v: f"{v:,.0f}"))
# show the aggregate test for each candidate bucket
say("\naggregate test -- worst relative gap between a bucket and the sum of the rest:")
for b, v in sorted(verdicts.items(), key=lambda kv: (np.inf if np.isnan(kv[1]) else kv[1])):
    say(f"    {str(b):14s} {v:.3e}" + ("   <-- AGGREGATE ROW" if b == agg_bucket else ""))

# decide the component bucket set from the test result
if agg_bucket is not None:
    # every bucket except the identified aggregate
    COMPONENTS = [b for b in bucket_tot.columns if b != agg_bucket]
    # say what was concluded and why
    say(f"\nCONCLUDED: '{agg_bucket}' is a pre-summed AGGREGATE row sitting in the")
    say(f"           same column as its components. Summing all buckets would")
    say(f"           DOUBLE-COUNT. Using components only: {COMPONENTS}")
else:
    # no aggregate found: every bucket is a genuine component
    COMPONENTS = list(bucket_tot.columns)
    say(f"\nCONCLUDED: no aggregate row detected. All buckets are components: {COMPONENTS}")

# collapse the day x bucket rows to one row per (date, config)
dayc = collapse_buckets(daily, ["date", c_cfg], "bucket", COMPONENTS)
# report the collapsed shape
say(f"\ncollapsed DAILY shape={dayc.shape}  (now one row per date x config)")

# -----------------------------------------------------------------------------
# A0b -- HARD RECONCILIATION against the run summary. Abort if it fails.
# -----------------------------------------------------------------------------
say("\nRECONCILIATION vs the run summary CSV (net_pkr must match exactly):")
# the summary's per-config net_pkr, keyed by config
sum_pkr = summ.set_index("throttle")["net_pkr"]
# the collapsed per-config net_pkr
col_pkr = dayc.groupby(c_cfg)["net_pkr"].sum()
# a flag that flips if anything fails to reconcile
recon_ok = True
# compare arm by arm
say(f"  {'config':16s} {'collapsed':>16s} {'summary':>16s} {'rel gap':>12s}")
for cfg in sorted(col_pkr.index):
    # the two totals
    a_ = float(col_pkr[cfg])
    b_ = float(sum_pkr.get(cfg, np.nan))
    # relative gap
    rel = abs(a_ - b_) / abs(b_) if b_ else np.nan
    # flag a failure
    if not (np.isfinite(rel) and rel < 1e-6):
        recon_ok = False
    # one line per arm
    say(f"  {cfg:16s} {a_:>16,.2f} {b_:>16,.2f} {rel:>12.2e}")
# refuse to report statistics built on an unreconciled decomposition
if not recon_ok:
    say("\n*** RECONCILIATION FAILED. The bucket collapse does not reproduce the")
    say("    run summary. Every statistic below would be built on a wrong")
    say("    decomposition. Stopping here rather than reporting numbers.")
    # write the partial report to a fresh timestamped file, never overwriting
    _fail = safe_out("stage0_audit_report_FAILED", "txt")
    _fail.write_text("\n".join(_REPORT), encoding="utf-8")
    print(f"\nwrote partial report -> {_fail}")
    raise SystemExit(1)
# say so plainly when it passes
say("  -> reconciled. Statistics below are built on a verified decomposition.")

# -----------------------------------------------------------------------------
# A1 -- day-as-unit table, on correctly collapsed days
# -----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("A1. DAY-AS-UNIT TABLE (collapsed days) vs the POOLED run summary")
say("-" * 78)
# one summary row per config arm
rows = []
# walk each arm
for cfg, g in dayc.groupby(c_cfg):
    # sort by date so the drawdown path is chronological
    g = g.sort_values("date")
    # that arm's daily net_bps series (notional-weighted within each day)
    x = g["net_bps"].astype(float).to_numpy()
    # that arm's daily PKR series
    p_ = g["net_pkr"].astype(float).to_numpy()
    # assemble the statistics the deploy decision rested on
    rows.append({
        "config": cfg,
        # genuine day count now
        "n_days": len(x),
        # day-average net_bps -- the README table's estimator
        "mean_bps": np.mean(x),
        # standard error of that mean
        "se_bps": np.std(x, ddof=1) / np.sqrt(len(x)),
        # total PKR, which must equal the run summary
        "net_pkr": np.sum(p_),
        # annualised Sharpe from the daily bps series
        "sharpe": sharpe(x),
        # annualised Sortino from the daily bps series
        "sortino": sortino(x),
        # worst peak-to-trough on cumulative daily PKR
        "maxDD_pkr": max_drawdown(p_),
        # share of days with a positive net_bps
        "win_pct": float(np.mean(x > 0) * 100.0),
    })
# assemble and sort
tbl = pd.DataFrame(rows).sort_values("config").reset_index(drop=True)
# print it
say(tbl.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
# print the pooled estimator alongside
say("\nPOOLED run summary (notional-weighted over the whole run):")
say(summ[["throttle", "net_bps", "net_pkr", "median_hold_s", "trades"]].to_string(index=False))
# state the expected relationship so a mismatch is not misread
say("\nNOTE: net_pkr must agree (reconciled above). net_bps need NOT: the table")
say("      is a DAY-AVERAGE, the summary is NOTIONAL-WEIGHTED over all fills.")
say("      Quote the deploy expectation against one named estimator, not both.")

# -----------------------------------------------------------------------------
# A2 -- paired test on collapsed days
# -----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("A2. PAIRED TEST vs CONTROL, on collapsed days")
say("-" * 78)
# every arm label present
labels = sorted(dayc[c_cfg].unique().tolist())
# report them
say("configs present: " + ", ".join(map(str, labels)))
# the control's daily bps indexed by date -- now a UNIQUE index
ctrl_s = dayc.loc[dayc[c_cfg] == CTRL].set_index("date")["net_bps"].astype(float)
# assert uniqueness explicitly; a duplicate index silently produced v1's wrong answer
assert ctrl_s.index.is_unique, "control index is not unique -- bucket collapse failed"
# collect for plotting
paired_rows = []
# compare every other arm to the control
for cfg in labels:
    # skip the control against itself
    if cfg == CTRL:
        continue
    # this arm's daily bps indexed by date
    arm_s = dayc.loc[dayc[c_cfg] == cfg].set_index("date")["net_bps"].astype(float)
    # guard against a non-unique index here too
    assert arm_s.index.is_unique, f"{cfg} index is not unique -- bucket collapse failed"
    # the dates BOTH arms have
    shared = ctrl_s.index.intersection(arm_s.index)
    # paired differences, signed so positive means the arm beats the control
    d = (arm_s.loc[shared] - ctrl_s.loc[shared]).to_numpy()
    # paired mean difference
    md = float(np.mean(d))
    # its standard error
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    # t statistic under the stated sign convention
    t = md / se if se > 0 else np.nan
    # two-sided p value
    pv = float(2 * stats.t.sf(abs(t), df=len(d) - 1)) if np.isfinite(t) else np.nan
    # the same difference computed unpaired, to cross-check the table
    unpaired = float(arm_s.mean() - ctrl_s.mean())
    # report
    say(f"\n  {cfg} vs {CTRL}")
    say(f"    days: arm={len(arm_s)}  control={len(ctrl_s)}  shared={len(shared)}")
    say(f"    PAIRED   mean diff = {md:+.4f} bps/day  SE={se:.4f}  t={t:+.2f}  p={pv:.3e}")
    say(f"    UNPAIRED mean diff = {unpaired:+.4f} bps/day")
    # flag any paired/unpaired gap, which can only come from unequal day coverage
    if abs(md - unpaired) > 1e-9:
        say(f"    *** PAIRED != UNPAIRED by {md - unpaired:+.4f}: unequal day coverage.")
    # restate the sign convention so the t is never misread
    say(f"    (convention: t > 0 means {cfg} BEATS {CTRL})")
    # keep for the chart
    paired_rows.append({"config": cfg, "diff": md, "se": se, "t": t})
# frame the paired results
paired = pd.DataFrame(paired_rows)

# -----------------------------------------------------------------------------
# A3 -- zero-skew consistency test
# -----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("A3. ZERO-SKEW CONSISTENCY TEST")
say("-" * 78)
# spell out the arithmetic so the output stands alone
say(f"QBPS_2 converts {SKEW_BPS} bps to whole ticks at a flat {TICK} PKR tick:")
say(f"    ticks = round({SKEW_BPS / 100:.2f} * price)")
say(f"    Rs  25 -> 0.50 ticks (rounds to 0)     Rs   50 ->  1 tick")
say(f"    Rs 200 -> 4 ticks                      Rs 1000 -> 20 ticks")
say("A name whose skew rounds to 0 ticks runs BYTE-IDENTICAL to the control, so")
say("its per-day difference must be exactly 0.000 on every single day.")

# load the per-name x bucket frame
pern = pd.read_parquet(pername_path)
# report shape
say(f"\nPERNAME raw shape={pern.shape}  (date x symbol x config x BUCKET)")
# collapse buckets to one row per (date, symbol, config)
pnc = collapse_buckets(pern, ["date", "symbol", c_cfg], "bucket", COMPONENTS)
# report collapsed shape
say(f"collapsed PERNAME shape={pnc.shape}")
# reconcile the per-name collapse against the day collapse
pn_tot = pnc.groupby(c_cfg)["net_pkr"].sum()
# compare arm by arm
say("\nPERNAME collapse reconciliation vs run summary:")
for cfg in sorted(pn_tot.index):
    # the two totals
    a_ = float(pn_tot[cfg]); b_ = float(sum_pkr.get(cfg, np.nan))
    # relative gap
    rel = abs(a_ - b_) / abs(b_) if b_ else np.nan
    # one line per arm, flagged if it fails
    say(f"  {cfg:16s} {a_:>16,.2f} vs {b_:>16,.2f}  rel={rel:.2e}"
        + ("" if (np.isfinite(rel) and rel < 1e-6) else "   *** MISMATCH"))

# control rows keyed by (symbol, date)
a = pnc.loc[pnc[c_cfg] == CTRL].set_index(["symbol", "date"])["net_pkr"].astype(float)
# deploy rows keyed identically
b = pnc.loc[pnc[c_cfg] == DEPLOY].set_index(["symbol", "date"])["net_pkr"].astype(float)
# both indexes must be unique or the join below is meaningless
assert a.index.is_unique and b.index.is_unique, "per-name index not unique after collapse"
# align on the symbol-days both arms produced
both = pd.concat([a.rename("ctrl"), b.rename("depl")], axis=1, join="inner")
# the per-symbol-day difference in PKR (PKR is exact; bps carries rounding)
both["diff"] = both["depl"] - both["ctrl"]

# collapse to one row per name
per_name = both.groupby(level=0)["diff"].agg(
    # how many symbol-days were compared
    n_days="count",
    # total PKR difference over the year
    total_diff="sum",
    # the largest single-day absolute difference: 0 proves the arms are identical
    max_abs_diff=lambda s: float(np.max(np.abs(s))),
).sort_values("total_diff")
# names where every day is bit-identical
identical = per_name.loc[per_name["max_abs_diff"] == 0.0]
# names that differ somewhere
differing = per_name.loc[per_name["max_abs_diff"] > 0.0]
# report the split -- the point of the test
say(f"\nnames compared             : {len(per_name)}")
say(f"names IDENTICAL to control : {len(identical)}   (skew rounded to 0 ticks)")
say(f"names that DIFFER          : {len(differing)}")
# name the identical ones
if len(identical):
    say("  identical: " + ", ".join(identical.index.astype(str)))
# give every reading so no outcome is ambiguous
say("\nINTERPRETATION:")
say("  0 identical  -> every name's price put the skew at >= 1 tick.")
say("  >0 identical -> those names contribute exactly 0 to the paired mean and")
say("                  DILUTE it. The edge where the skew is ACTIVE is larger")
say("                  than the headline. Re-state the deploy expectation")
say("                  stratified by realised tick count.")
say("  identical on MOST days but not all -> something OTHER than the skew")
say("                  differs between the runs. Stop and investigate.")
# the full per-name table, worst first
say("\nper-name total PKR difference (QBPS_2 - OBI), worst first:")
say(per_name.to_string(float_format=lambda v: f"{v:,.2f}"))

# -----------------------------------------------------------------------------
# A4 -- the engine's own tradeability gate, recomputed from PERNAME
# -----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("A4. TRADEABILITY GATE under QBPS_2 (capture <= 0, or |markout|/capture > 0.7)")
say("-" * 78)
# per-name totals under the deploy config
gate = pnc.loc[pnc[c_cfg] == DEPLOY].groupby("symbol")[
    ["net_pkr", "capture_pkr", "markout_pkr", "fee_pkr", "opened_notional", "fills"]].sum()
# realised net bps per name, notional-weighted
gate["net_bps"] = 1e4 * gate["net_pkr"] / gate["opened_notional"]
# the adverse-selection ratio the gate keys on. NOTE: when capture_pkr <= 0 this
# is negative or infinite and therefore meaningless -- that case is caught by
# fail_capture below, which is why the ratio is not separately guarded here.
gate["mko_over_cap"] = gate["markout_pkr"].abs() / gate["capture_pkr"]
# the two exclusion conditions
gate["fail_capture"] = gate["capture_pkr"] <= 0
gate["fail_markout"] = gate["mko_over_cap"] > GATE_MKO_OVER_CAP
# combined
gate["EXCLUDE"] = gate["fail_capture"] | gate["fail_markout"]
# report the excluded set
exc = gate[gate["EXCLUDE"]].sort_values("net_pkr")
say(f"names excluded by the gate: {len(exc)} of {len(gate)}")
say(exc[["net_pkr", "net_bps", "capture_pkr", "markout_pkr", "mko_over_cap",
         "fail_capture", "fail_markout"]].to_string(float_format=lambda v: f"{v:,.3f}"))
# does the gate actually remove losers and keep winners?
# the boolean mask of engine-negative names
neg = gate["net_pkr"] < 0
# how many names the engine actually lost money on
say(f"\nengine-negative names: {int(neg.sum())} of {len(gate)}")
# how many of those the gate catches -- parenthesised because & binds tighter than <
say(f"  of those, caught by the gate : {int((neg & gate['EXCLUDE']).sum())}")
# how many the gate MISSES, which is the number that matters for universe expansion
say(f"  of those, MISSED by the gate : {int((neg & ~gate['EXCLUDE']).sum())}")
# and how many profitable names the gate wrongly throws away
say(f"  profitable names wrongly excluded: {int((~neg & gate['EXCLUDE']).sum())}")
say(f"PKR removed by the gate: {exc['net_pkr'].sum():,.0f} "
    f"({100 * exc['net_pkr'].sum() / gate['net_pkr'].sum():+.2f}% of total)")

# =============================================================================
# PART B -- DOES THE MODEL-FREE SCREEN PREDICT THE ENGINE?
# =============================================================================
say("\n" + "=" * 78)
say("PART B  --  MODEL-FREE SCREEN vs ENGINE GROUND TRUTH")
say("=" * 78)

# the 491-name persistence screen at TREC fees
pers = pd.read_csv(RESULTS / "persistence_REG_2p00.csv")
# report its scope
say(f"persistence_REG_2p00.csv shape={pers.shape}  ({pers['symbol'].nunique()} symbols)")
# recover the fee the screen actually charges, from the data itself
implied_fee = float((pers["mk5_bps_median"] - pers["net5_trec_median"]).median())
# compare it to the house round-trip convention
say(f"\nnet5_trec_median = mk5_bps_median - {implied_fee:.3f} bps (constant on all names)")
say(f"fill_attribution.fee_bps_roundtrip() charges {FEE_BPS_RT:.3f} bps")
say(f"  -> the screen charges ONE side's fee ({FEE_BPS_SIDE:.3f}). Constant")
say(f"     {FEE_BPS_RT - implied_fee:+.3f} bps bias on every name. Confirm whether mk5 is")
say(f"     one-sided or round-trip; it shifts the LEVEL, and may not shift the RANK.")
# recompute charging the full round trip
pers["net5_rt"] = pers["mk5_bps_median"] - FEE_BPS_RT
# capturable half-spread if resting at the touch
pers["half_spread"] = pers["spread_bps_median"] / 2.0
# model-free adverse-selection share -- the model-free analog of |markout|/capture
pers["as_share"] = (pers["half_spread"] - pers["mk5_bps_median"]) / pers["half_spread"]
# how many names each fee convention passes
say(f"\npositive net5 as screened      : {int((pers['net5_trec_median'] > 0).sum())} / {len(pers)}")
say(f"positive net5 at round-trip fee: {int((pers['net5_rt'] > 0).sum())} / {len(pers)}")
say("  -> a criterion that passes ~every name is not a GATE, only a RANKING.")
say("     A ranking is worthless unless it predicts the engine. Tested below.")

# the engine's realised per-name outcome, from the gate table built in A4
eng = gate.reset_index()[["symbol", "net_pkr", "net_bps", "capture_pkr",
                          "markout_pkr", "mko_over_cap", "EXCLUDE"]]
# join the screen onto the engine results
m = pers.merge(eng, on="symbol", how="inner")
# report the overlap
say(f"\nnames in BOTH the screen and the QBPS_2 run: {len(m)}")
say(f"engine-positive names: {int((m['net_pkr'] > 0).sum())} / {len(m)}")

# the model-free predictors worth testing
CANDS = [c for c in ["net5_trec_median", "net5_rt", "mk5_bps_median", "spread_bps_median",
                     "ceiling_median", "notional_m_median", "pct_days_edge5_pos",
                     "pct_days_tradeable", "as_share", "pct_time_wide_median"]
         if c in m.columns]
# the headline test
say("\nSPEARMAN rank correlation: model-free predictor vs ENGINE realised outcome")
say(f"{'predictor':24s} {'rho(net_pkr)':>14s} {'p':>9s} {'rho(net_bps)':>14s}")
# collect for plotting
corr_rows = []
# walk each predictor
for c in CANDS:
    # rank correlation against realised PKR
    r1 = stats.spearmanr(m[c], m["net_pkr"], nan_policy="omit")
    # rank correlation against realised bps
    r2 = stats.spearmanr(m[c], m["net_bps"], nan_policy="omit")
    # one line per predictor
    say(f"{c:24s} {r1.statistic:>+14.3f} {r1.pvalue:>9.3f} {r2.statistic:>+14.3f}")
    # keep for the chart
    corr_rows.append({"predictor": c, "rho_pkr": r1.statistic, "p": r1.pvalue})
# frame it
corr = pd.DataFrame(corr_rows)
# spell out how to read the result
say("\nDECISION RULE:")
say("  rho(net_pkr) >= +0.6  -> usable as a stage-1 ranking as-is.")
say("  rho(net_pkr) near 0   -> carries no ranking information.")
say("  rho(net_pkr) NEGATIVE -> ANTI-predictive: ranking 491 names by it would")
say("                           preferentially select names the engine LOSES on.")
say("  PRIOR: a pre-run against the LEGACY universe_ranked engine results gave")
say("         net5_trec_median rho = -0.300, spread_bps_median rho = -0.508,")
say("         notional_m_median rho = +0.433, pct_days_edge5_pos rho = +0.387")
say("         (n=38). Confirm or refute under QBPS_2 here.")

# -----------------------------------------------------------------------------
# B2 -- gate sensitivity: how much expansion headroom actually exists
# -----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("B2. GATE SENSITIVITY -- how many names clear a liquidity bar")
say("-" * 78)
# the hard economic floor a passive quote must clear
say(f"hard floor: quoted spread must exceed the {FEE_BPS_RT:.3f} bps round-trip fee")
# the names already run through the engine
known = set(m["symbol"].astype(str))
# header
say(f"\n{'gate':30s} {'total':>7s} {'run':>5s} {'NEW':>5s} {'eng PKR kept':>15s} {'losers kept':>12s}")
# collect for plotting
gate_rows = []
# walk the sensitivity grid
for nt, sp in GATE_GRID:
    # apply the gate to the full universe, requiring a real trading history
    full = pers[(pers.notional_m_median >= nt) & (pers.spread_bps_median <= sp)
                & (pers.days_traded >= 100)]
    # the same gate restricted to engine-run names
    kept = m[(m.notional_m_median >= nt) & (m.spread_bps_median <= sp)]
    # engine PKR retained
    pkr_kept = float(kept["net_pkr"].sum()) if len(kept) else np.nan
    # engine-negative names the gate failed to exclude
    losers = int((kept["net_pkr"] < 0).sum()) if len(kept) else 0
    # never-run names the gate admits
    new = len(set(full.symbol.astype(str)) - known)
    # one line per gate
    say(f"{'>=%dM / <=%dbps' % (nt, sp):30s} {len(full):>7d} {len(full) - new:>5d} "
        f"{new:>5d} {pkr_kept:>15,.0f} {losers:>12d}")
    # keep for the chart
    gate_rows.append({"gate": f">={nt}M/<={sp}bps", "total": len(full), "new": new,
                      "pkr_kept": pkr_kept, "losers": losers})
# frame it
gates = pd.DataFrame(gate_rows)
# the operative question
say("\nTHE QUESTION THIS ANSWERS:")
say("  The README assumes ~500 names of expansion headroom. 'NEW' is the actual")
say("  number of never-run names each liquidity bar admits. If that number is")
say("  small at the bars where the engine makes money, the expansion is worth a")
say("  handful of names, not 462 -- and CLIP SIZE / capacity per name becomes")
say("  the binding lever instead of name count.")
say("  CAUTION: these bars are NOT calibrated. They are a sensitivity grid.")
say("  Fit the bar on the engine-run names first (PART B), then apply it.")

# write the candidate list at the tightest bar in the grid
nt0, sp0 = GATE_GRID[0]
# apply it
cand = pers[(pers.notional_m_median >= nt0) & (pers.spread_bps_median <= sp0)
            & (pers.days_traded >= 100)].copy()
# mark which are already in the book
cand["already_run"] = cand["symbol"].astype(str).isin(known)
# rank by the persistence column
cand = cand.sort_values("pct_days_edge5_pos", ascending=False)
# a fresh timestamped destination; never overwrites a previous run's list
cand_path = safe_out("stage1_candidates", "csv")
# write it
cand.to_csv(cand_path, index=False)
# report
say(f"\nwrote {len(cand)} candidates ({int((~cand.already_run).sum())} new) -> {cand_path.name}")

# =============================================================================
# PLOTS
# =============================================================================
# ---- plot 1: the calibration scatter
fig, ax = plt.subplots(figsize=(8, 6.5))
# colour by engine outcome
cols = np.where(m["net_pkr"] > 0, "steelblue", "crimson")
# scatter screen edge against engine PKR
ax.scatter(m["net5_trec_median"], m["net_pkr"], s=55, c=cols, alpha=0.85,
           edgecolor="k", linewidth=0.4)
# annotate every point
for _, r in m.iterrows():
    ax.annotate(str(r["symbol"]), (r["net5_trec_median"], r["net_pkr"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
# break-even line
ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
# the verdict in the title
rho = stats.spearmanr(m["net5_trec_median"], m["net_pkr"], nan_policy="omit").statistic
ax.set_xlabel("model-free screen: net5_trec_median (bps)")
ax.set_ylabel("engine realised net_pkr under QBPS_2")
ax.set_title(f"Does the screen predict the engine?  Spearman rho = {rho:+.3f}  (n={len(m)})\n"
             f"red = engine lost money on this name")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(safe_out("stage0_screen_vs_engine", "png"), dpi=140)
plt.close(fig)

# ---- plot 2: every predictor's rank correlation
fig, ax = plt.subplots(figsize=(8.5, 5.5))
cs = corr.sort_values("rho_pkr")
ax.barh(range(len(cs)), cs["rho_pkr"], color=np.where(cs["rho_pkr"] > 0, "seagreen", "crimson"))
ax.set_yticks(range(len(cs)))
ax.set_yticklabels(cs["predictor"], fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Spearman rho vs engine realised net_pkr")
ax.set_title("Which model-free screen columns actually predict engine P&L?")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(safe_out("stage0_predictor_ranking", "png"), dpi=140)
plt.close(fig)

# ---- plot 3: paired edge over control, with error bars
fig, ax = plt.subplots(figsize=(8, 5))
pp = paired.sort_values("diff")
ax.barh(range(len(pp)), pp["diff"], xerr=1.96 * pp["se"], color="steelblue",
        error_kw=dict(ecolor="black", lw=1.2, capsize=4))
ax.set_yticks(range(len(pp)))
ax.set_yticklabels(pp["config"], fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel(f"paired mean edge over {CTRL} (bps/day), 95% CI")
ax.set_title("Queue-skew configs vs the OBI control, day-paired")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(safe_out("stage0_paired_edge", "png"), dpi=140)
plt.close(fig)

# ---- plot 4: expansion headroom
fig, ax = plt.subplots(figsize=(9, 5))
xs = np.arange(len(gates))
ax.bar(xs, gates["total"] - gates["new"], label="already in the book", color="steelblue")
ax.bar(xs, gates["new"], bottom=gates["total"] - gates["new"],
       label="NEW (never engine-tested)", color="darkorange")
for i, r in gates.iterrows():
    ax.text(i, r["total"] + 1, f"+{int(r['new'])}", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(gates["gate"], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("names admitted")
ax.set_title("How much universe headroom actually exists at each liquidity bar?")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(safe_out("stage0_expansion_headroom", "png"), dpi=140)
plt.close(fig)

# =============================================================================
# REPORT
# =============================================================================
# report destination
rep = safe_out("stage0_audit_report", "txt")
# write every line that was printed
rep.write_text("\n".join(_REPORT), encoding="utf-8")
# tell the user where everything landed
print(f"\nwrote report -> {rep}")
print(f"wrote plots  -> stage0_*_{RUN_TS}.png  (4 files, all newly created)")
print("NOTE: this script never deletes or overwrites. Every output is timestamped.")
