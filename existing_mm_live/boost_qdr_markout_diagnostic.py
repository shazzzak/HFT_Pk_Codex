# ============================================================================
# boost_qdr_markout_diagnostic.py
# ----------------------------------------------------------------------------
# TWO pre-quoting GATES on the existing feature store, one harness:
#
#   GATE 1 (BOOST / favorable-OBI):  Does a FAVORABLE order-book imbalance
#          predict FAVORABLE side-signed markout, symmetric to the way an
#          ADVERSE imbalance predicts adverse markout (the throttle's basis)?
#          This is the OFFENSIVE mirror of the throttle. PRIOR: adverse.
#          It half-answers "will a 3x->5x favorable-side boost help"; it CANNOT
#          see fill probability (feature store has no fills), so a PASS here is
#          necessary-not-sufficient and the real judge is the backtester sweep.
#
#   GATE 2 (QDR):  When your OWN-side touch queue is being depleted fast at a
#          stable price, is a fill on that side toxic (worse side-signed
#          markout)? DEFENSIVE gate, same shape as the throttle's OBI gate.
#          PRIOR: favorable. NOTE: qdr_* conflates aggressive consumption with
#          cancellation -- it is a STATE gate, not a pure trade-consumption rate.
#
# METHOD (matches the throttle-diagnostic discipline):
#   * side-signed markout: bid fill -> +markout, ask fill -> -markout
#   * DAY-AS-UNIT error bars everywhere (never pool row-count SEs)
#   * per-name axis NEVER dropped
#   * multiple horizons (1s/5s/30s); 5s headline (flatten horizon)
#   * fee reference line at +/-1.55 bps round-trip (TREC spot)
#   * synthetic self-test with INJECTED ground truth validates the harness
#     (sign, magnitude, per-name axis, and a NULL/false-positive guard) BEFORE
#     it ever touches real data.
#
# USAGE:
#   python boost_qdr_markout_diagnostic.py --self-test
#       -> builds synthetic data in memory, asserts the harness recovers truth.
#   python boost_qdr_markout_diagnostic.py --run
#       -> walks the real feature store, writes CSVs + PNGs (heartbeat+timer).
# ============================================================================

# Standard-library argument parsing.
import argparse
# Wall-clock timing for the per-symbol-day heartbeat.
import time
# Filesystem paths.
from pathlib import Path
# Numeric arrays.
import numpy as np
# DataFrames + parquet IO.
import pandas as pd

# Non-interactive plotting backend (safe on headless / nohup runs).
import matplotlib
# Force Agg before pyplot import so no display is required.
matplotlib.use("Agg")
# Plotting API.
import matplotlib.pyplot as plt

# Optional SciPy for exact t p-values; fall back to a normal approx if absent.
try:
    # Student-t survival function for two-sided p-values.
    from scipy import stats as _scipy_stats
    # Flag that exact p-values are available.
    _HAVE_SCIPY = True
except Exception:
    # SciPy not installed -> p-values via normal approximation.
    _scipy_stats = None
    # Flag the fallback.
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Feature-store root: feature_store/{SYM}/date=YYYY-MM-DD.parquet (date in FILENAME).
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
# Where CSVs + PNGs land (SZ copies what he wants into existing_mm_live/).
OUT_DIR = Path("/Users/shazzak/Capital Stake - Results/diagnostics")
# Markout horizons present in the feature store (ms). 5s is the headline.
HORIZONS_MS = [1000, 5000, 30000]
# Headline horizon for the printed summary + plots.
HEADLINE_MS = 5000
# Round-trip TREC spot fee in bps -- the P&L hurdle the boost's edge must clear.
FEE_RT_BPS = 1.55
# Minimum rows in EACH state (active/neutral) within a symbol-day to trust its mean.
MIN_ROWS_PER_STATE = 30
# Upper quantile of |obi_1| that defines the FAVORABLE-active state (top decile).
OBI_ACTIVE_Q = 0.90
# Lower quantile of |obi_1| that defines the NEUTRAL/balanced state (bottom decile).
OBI_NEUTRAL_Q = 0.10
# Upper quantile of POSITIVE qdr that defines the high-depletion active state.
QDR_ACTIVE_Q = 0.80
# Thin-name universe (adverse selection is a thin-name phenomenon) for plot coloring.
THIN_NAMES = {"FNEL", "TPL", "PACE", "PIAHCLA", "HASCOL", "NPL", "TOMCL"}
# Deep blue-chip universe (throttle was ~neutral here) for plot coloring.
DEEP_NAMES = {"OGDC", "PSO", "HUBC", "FFC", "PPL", "UBL", "NBP", "MEBL"}


# ===========================================================================
# STATISTICS -- day-as-unit primitives
# ===========================================================================
def _tstat(x):
    # Drop NaNs so partial symbol-days don't poison the moment estimates.
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    # Need at least two units for a variance.
    n = x.size
    if n < 2:
        return np.nan, np.nan, np.nan, n
    # Sample mean of the per-unit values.
    m = float(np.mean(x))
    # Standard error = sample SD / sqrt(n) (ddof=1).
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    # Guard a degenerate zero-variance sample.
    if se == 0.0:
        return m, np.nan, np.nan, n
    # t-statistic of the mean against 0.
    t = m / se
    # Two-sided p-value: exact t if SciPy, else normal approx.
    if _HAVE_SCIPY:
        p = float(2.0 * _scipy_stats.t.sf(abs(t), df=n - 1))
    else:
        # Normal-approx survival: 0.5*erfc(|t|/sqrt(2)) via math.
        from math import erfc, sqrt
        p = float(erfc(abs(t) / sqrt(2.0)))
    # Return the quadruple used throughout.
    return m, se, t, n


# ===========================================================================
# SIDE-SIGNING -- turn unsigned mid-drift markout into per-fill adverse selection
# ===========================================================================
def _long_favorable_obi(df, horizon_ms):
    # Column holding the mid-drift markout for this horizon.
    mcol = f"markout_{horizon_ms}ms_bps"
    # Rows must have both obi_1 and the markout label; reset index so the Series
    # (symbol/date) and the positional numpy arrays below stay aligned.
    d = df[["symbol", "date", "obi_1", mcol]].dropna().reset_index(drop=True)
    # obi_1 / markout as float32 arrays (fs is already float32; this is a no-op copy).
    obi = d["obi_1"].to_numpy(dtype=np.float32)
    mk = d[mcol].to_numpy(dtype=np.float32)
    # The FAVORABLE side for a maker: bid when obi_1>0 (up-pressure), else ask.
    is_bid = obi > 0.0
    # Side-signed markout: bid fill earns +drift, ask fill earns -drift.
    signed = np.where(is_bid, mk, -mk).astype(np.float32)
    # Assemble. symbol/date come straight from the (already categorical) fs
    # columns as Series -> stay compact; no object round-trip.
    return pd.DataFrame({
        "symbol": d["symbol"],
        "date": d["date"],
        "signal": np.abs(obi),
        "signed_markout": signed,
    })


def _long_qdr(df, horizon_ms):
    # Column holding the mid-drift markout for this horizon.
    mcol = f"markout_{horizon_ms}ms_bps"
    # Need markout plus both depletion columns; reset index for clean concat.
    d = df[["symbol", "date", "qdr_bid", "qdr_ask", mcol]].dropna().reset_index(drop=True)
    # Markout as float32 (fs already float32).
    m = d[mcol].to_numpy(dtype=np.float32)
    # Stack bid+ask records. pd.concat on the (categorical) symbol/date Series
    # PRESERVES the category dtype -> only int codes are duplicated, not 314M
    # Python strings. BID: high qdr_bid makes a BID fill (+drift) toxic; ASK:
    # high qdr_ask makes an ASK fill (-drift) toxic; qdr==0 = neutral base.
    out = pd.DataFrame({
        "symbol": pd.concat([d["symbol"], d["symbol"]], ignore_index=True),
        "date": pd.concat([d["date"], d["date"]], ignore_index=True),
        "signal": np.concatenate([d["qdr_bid"].to_numpy(dtype=np.float32),
                                  d["qdr_ask"].to_numpy(dtype=np.float32)]),
        "signed_markout": np.concatenate([m, -m]),
    })
    # Free the source frame promptly (the 2x stack is the memory peak here).
    del d, m
    return out


# ===========================================================================
# GAP -- active vs neutral side-signed markout, DAY-AS-UNIT
# ===========================================================================
def _active_neutral_masks(signal, gate):
    # BOOST gate: active = strong |obi|, neutral = balanced |obi|.
    if gate == "boost":
        # Top-decile imbalance magnitude across the pooled distribution.
        hi = np.quantile(signal, OBI_ACTIVE_Q)
        # Bottom-decile imbalance magnitude.
        lo = np.quantile(signal, OBI_NEUTRAL_Q)
        # Active/neutral boolean masks.
        return (signal >= hi), (signal <= lo), hi, lo
    # QDR gate: active = high POSITIVE depletion, neutral = exactly zero.
    if gate == "qdr":
        # Positive-only distribution defines the high-depletion threshold.
        pos = signal[signal > 0.0]
        # Fall back to a tiny epsilon if no positive mass (degenerate).
        hi = np.quantile(pos, QDR_ACTIVE_Q) if pos.size else 1e9
        # Active = high depletion; neutral = no depletion this event.
        return (signal >= hi) & (signal > 0.0), (signal == 0.0), hi, 0.0
    # Unknown gate name is a programming error.
    raise ValueError(f"unknown gate {gate!r}")


def per_name_gap(long_df, gate):
    # Compute the active/neutral thresholds ONCE on the pooled signal.
    a_mask_all, n_mask_all, hi, lo = _active_neutral_masks(long_df["signal"].to_numpy(), gate)
    # Attach the state labels IN PLACE. long_df is freshly built by the caller
    # each horizon, so mutating it avoids copying a 300M-row frame.
    d = long_df
    d["_active"] = a_mask_all
    d["_neutral"] = n_mask_all
    # VECTORIZED day-as-unit means: two groupby-aggs instead of iterating 7,866
    # groups in Python over 300M rows (the old loop was the "computing gaps"
    # bottleneck the heartbeat exposed). Identical semantics: per (symbol,date)
    # mean + count of the active slice and of the neutral slice.
    gcols = ["symbol", "date"]
    # Active-state per-day mean and count (count enforces the min-rows guard).
    act = (d.loc[d["_active"], gcols + ["signed_markout"]]
           .groupby(gcols, observed=True)["signed_markout"]
           .agg(active="mean", a_n="count"))
    # Neutral-state per-day mean and count.
    neu = (d.loc[d["_neutral"], gcols + ["signed_markout"]]
           .groupby(gcols, observed=True)["signed_markout"]
           .agg(neutral="mean", n_n="count"))
    # Inner-join keeps only days present in BOTH states.
    daily = act.join(neu, how="inner").reset_index()
    # Apply the same MIN_ROWS_PER_STATE guard as the old loop, on BOTH states.
    daily = daily[(daily["a_n"] >= MIN_ROWS_PER_STATE) & (daily["n_n"] >= MIN_ROWS_PER_STATE)]
    # No usable days -> empty result.
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame(), (hi, lo)
    # Paired per-day gap (active minus neutral), same definition as before.
    daily["gap"] = daily["active"] - daily["neutral"]
    # Keep only the columns downstream expects (drop the count helpers).
    daily = daily[["symbol", "date", "active", "neutral", "gap"]].reset_index(drop=True)
    # --- PER-NAME (unit = that name's days) ---
    per_name_rows = []
    # One row per symbol; per-name axis is NEVER collapsed. observed=True so an
    # unused symbol category cannot create an empty phantom group.
    for sym, g in daily.groupby("symbol", sort=False, observed=True):
        # Day-as-unit moments of the paired gap for this name.
        m, se, t, n = _tstat(g["gap"].to_numpy())
        # Also carry the mean active/neutral levels for interpretation.
        per_name_rows.append((sym, m, se, t, n,
                              float(np.nanmean(g["active"])),
                              float(np.nanmean(g["neutral"]))))
    # Assemble and sort by effect size.
    per_name = pd.DataFrame(per_name_rows,
                            columns=["symbol", "gap_bps", "se_bps", "t", "n_days",
                                     "active_bps", "neutral_bps"]
                            ).sort_values("gap_bps").reset_index(drop=True)
    return per_name, daily, (hi, lo)


def portfolio_gap(daily):
    # Portfolio unit = DATE (names pooled within a day), matching the throttle sweep.
    per_date = daily.groupby("date", observed=True)["gap"].mean().to_numpy()
    # Day-as-unit moments across dates.
    m, se, t, n = _tstat(per_date)
    # Win rate = fraction of days with a positive portfolio gap.
    wins = int(np.sum(per_date > 0.0))
    # Return the headline tuple.
    return dict(gap_bps=m, se_bps=se, t=t, n_days=n, wins=wins, total=len(per_date))


# ===========================================================================
# BUCKET PROFILE -- side-signed markout across signal buckets, for the plots
# ===========================================================================
def bucket_profile(long_df, gate):
    # Work on the caller's freshly-built frame directly (no 300M-row copy).
    d = long_df
    # Assign each row to a signal bucket appropriate to the gate.
    if gate == "boost":
        # Deciles of |obi_1| (0..9); label by bucket centre later.
        edges = np.quantile(d["signal"].to_numpy(), np.linspace(0, 1, 11))
        # Ensure strictly increasing edges (ties -> nudge).
        edges = np.unique(edges)
        # Bucket index per row.
        d["bucket"] = np.clip(np.digitize(d["signal"].to_numpy(), edges[1:-1]), 0, len(edges) - 2)
        # Human-readable bucket labels (decile number).
        labels = {i: f"D{i}" for i in range(len(edges) - 1)}
    else:
        # QDR buckets: exactly-zero, then terciles of the positive mass.
        sig = d["signal"].to_numpy()
        # Positive-only quantiles for three nonzero bands.
        pos = sig[sig > 0.0]
        q = np.quantile(pos, [1 / 3, 2 / 3]) if pos.size else np.array([1e9, 1e9])
        # Bucket assignment: 0 zero, 1 low, 2 mid, 3 high.
        b = np.zeros(sig.size, dtype=int)
        b[(sig > 0.0) & (sig <= q[0])] = 1
        b[(sig > q[0]) & (sig <= q[1])] = 2
        b[sig > q[1]] = 3
        d["bucket"] = b
        # Labels for the four QDR bands.
        labels = {0: "0", 1: "low", 2: "mid", 3: "high"}
    # Per (name,date,bucket) mean side-signed markout (the daily unit value).
    cell = d.groupby(["symbol", "date", "bucket"], observed=True)["signed_markout"].mean().reset_index()
    # Day-as-unit across (name,date) units within each bucket: mean + SE.
    out = []
    # Walk buckets in order.
    for b in sorted(labels):
        # All daily unit means in this bucket.
        vals = cell.loc[cell["bucket"] == b, "signed_markout"].to_numpy()
        # Day-as-unit moments.
        m, se, t, n = _tstat(vals)
        # Record for plotting.
        out.append((b, labels[b], m, se, n))
    # Frame of per-bucket profile points.
    return pd.DataFrame(out, columns=["bucket", "label", "mean_bps", "se_bps", "n_units"])


# ===========================================================================
# PLOTS
# ===========================================================================
def _name_color(sym):
    # Thin names red (where adverse selection lives), deep blue-chips blue, rest grey.
    if sym in THIN_NAMES:
        return "#c0392b"
    if sym in DEEP_NAMES:
        return "#2c6fbb"
    return "#888888"


def plot_gate(per_name, profile, gate, horizon_ms, out_png):
    # Two panels: (left) signal->markout profile, (right) per-name gap bars.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    # ---- LEFT: bucket profile with day-as-unit error bars ----
    x = np.arange(len(profile))
    # Mean +/- 1 SE per bucket.
    axL.errorbar(x, profile["mean_bps"], yerr=profile["se_bps"], marker="o",
                 capsize=4, lw=1.8, color="#222222")
    # Zero reference (no edge).
    axL.axhline(0.0, color="k", lw=0.8)
    # Fee lines: the boost's edge must clear +1.55; toxicity worse than -1.55 is severe.
    axL.axhline(+FEE_RT_BPS, color="#2c6fbb", ls="--", lw=1.0, label=f"+{FEE_RT_BPS} bps fee")
    axL.axhline(-FEE_RT_BPS, color="#c0392b", ls="--", lw=1.0, label=f"-{FEE_RT_BPS} bps fee")
    # Bucket labels on x.
    axL.set_xticks(x)
    axL.set_xticklabels(profile["label"])
    # Axis + title copy depends on the gate.
    if gate == "boost":
        axL.set_xlabel("|OBI| decile (D0 balanced -> D9 strong)")
        axL.set_ylabel("favorable-side signed markout (bps)")
        axL.set_title(f"BOOST gate: does favorable OBI pay? ({horizon_ms}ms)")
    else:
        axL.set_xlabel("own-side QDR bucket")
        axL.set_ylabel("own-side signed markout (bps)")
        axL.set_title(f"QDR gate: is fast own-side depletion toxic? ({horizon_ms}ms)")
    # Legend for the fee lines.
    axL.legend(fontsize=8)
    # Light grid.
    axL.grid(alpha=0.25)
    # ---- RIGHT: per-name gap (active-neutral) with SE, colored by liquidity class ----
    xn = np.arange(len(per_name))
    # Bar colors by thin/deep/other.
    colors = [_name_color(s) for s in per_name["symbol"]]
    # Horizontal-ish bars via vertical bars with rotated labels.
    axR.bar(xn, per_name["gap_bps"], yerr=per_name["se_bps"], capsize=2, color=colors)
    # Zero reference.
    axR.axhline(0.0, color="k", lw=0.8)
    # Fee reference for scale.
    axR.axhline(+FEE_RT_BPS, color="#2c6fbb", ls="--", lw=0.8)
    axR.axhline(-FEE_RT_BPS, color="#c0392b", ls="--", lw=0.8)
    # Name labels.
    axR.set_xticks(xn)
    axR.set_xticklabels(per_name["symbol"], rotation=90, fontsize=7)
    # Axis + title.
    axR.set_ylabel("active - neutral gap (bps), day-as-unit")
    axR.set_title(f"Per-name gap (red=thin, blue=deep) ({horizon_ms}ms)")
    # Light grid.
    axR.grid(alpha=0.25, axis="y")
    # Tidy layout.
    fig.tight_layout()
    # Persist.
    fig.savefig(out_png, dpi=130)
    # Free the figure.
    plt.close(fig)


# ===========================================================================
# DRIVER over the real feature store (heartbeat + timer from v1)
# ===========================================================================
def load_feature_store(root, symbols=None):
    # Root must exist before we walk it.
    if not root.exists():
        raise FileNotFoundError(f"feature store not found: {root}")
    # Per-symbol directories.
    sym_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    # Optional symbol filter.
    if symbols:
        sym_dirs = [p for p in sym_dirs if p.name in set(symbols)]
    # Accumulate per-symbol-day frames.
    frames = []
    # Start the wall clock for the heartbeat.
    t_start = time.perf_counter()
    # Count files for progress.
    total = sum(len(list(p.glob("date=*.parquet"))) for p in sym_dirs)
    # Running file counter.
    seen = 0
    # Columns we actually need (keeps memory down).
    cols = ["symbol", "date", "obi_1", "qdr_bid", "qdr_ask"] + \
           [f"markout_{h}ms_bps" for h in HORIZONS_MS]
    # Walk symbols.
    for sp in sym_dirs:
        # Walk that symbol's day files.
        for fp in sorted(sp.glob("date=*.parquet")):
            # Read only the needed columns.
            try:
                df = pd.read_parquet(fp, columns=cols)
            except Exception as e:
                # Report and skip a corrupt/partial file rather than abort the run.
                print(f"    SKIP {fp.name} ({sp.name}): {e!r}", flush=True)
                continue
            # Keep it.
            frames.append(df)
            # Advance and heartbeat every 100 files.
            seen += 1
            if seen % 100 == 0:
                # Elapsed and rate for an ETA feel.
                el = time.perf_counter() - t_start
                print(f"    [{seen}/{total}] loaded  {el:5.1f}s  ({seen/el:4.1f} files/s)", flush=True)
    # Concatenate everything (or empty).
    if not frames:
        return pd.DataFrame()
    # One combined frame.
    fs = pd.concat(frames, ignore_index=True)
    # Free the per-file frames immediately (they duplicate all the memory).
    del frames
    # DOWNCAST to bound memory: float32 for every numeric column (bps values are
    # small; float32 precision is ample for means) and category for the two
    # grouping keys (Python-string columns are the single biggest hog at 157M
    # rows). This takes fs from ~12GB to ~4GB and speeds every groupby.
    num_cols = ["obi_1", "qdr_bid", "qdr_ask"] + [f"markout_{h}ms_bps" for h in HORIZONS_MS]
    for c in num_cols:
        # Downcast in place, ignoring any column absent from the store.
        if c in fs.columns:
            fs[c] = fs[c].astype(np.float32)
    # Grouping keys -> category (int codes + a tiny dictionary).
    fs["symbol"] = fs["symbol"].astype("category")
    fs["date"] = fs["date"].astype("category")
    return fs


def run_real(root=FS_ROOT, out_dir=OUT_DIR, symbols=None, gates=None):
    # Announce scope.
    print(f"[run] feature store: {root}", flush=True)
    # Load with heartbeat.
    t0 = time.perf_counter()
    fs = load_feature_store(root, symbols=symbols)
    print(f"[run] loaded {len(fs):,} rows across {fs['symbol'].nunique() if len(fs) else 0} names in {time.perf_counter()-t0:.1f}s", flush=True)
    # Nothing to do on an empty store.
    if len(fs) == 0:
        print("[run] EMPTY feature store -- nothing to analyze.", flush=True)
        return
    # Make sure the output directory exists.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Which gates to run (default both). Lets you isolate QDR: --gates qdr.
    all_gates = [("boost", _long_favorable_obi), ("qdr", _long_qdr)]
    run_gates = [g for g in all_gates if gates is None or g[0] in gates]
    # Run selected gates across ALL horizons; headline printed for HEADLINE_MS.
    for gate, longfn in run_gates:
        # Section banner.
        print(f"\n==================== GATE: {gate.upper()} ====================", flush=True)
        # Per horizon.
        for h in HORIZONS_MS:
            # HEARTBEAT: announce the horizon BEFORE the heavy work so a slow
            # horizon shows progress instead of looking frozen. QDR is 2x rows.
            th = time.perf_counter()
            print(f"  [{h}ms] building long frame (gate={gate}) ...", flush=True)
            # Build the side-signed long frame for this gate+horizon.
            long_df = longfn(fs, h)
            # Report the frame size + build time (the QDR 2x cost is visible here).
            print(f"  [{h}ms] frame={len(long_df):,} rows, built in {time.perf_counter()-th:.1f}s; "
                  f"computing gaps ...", flush=True)
            # Per-name + daily gaps.
            per_name, daily, (hi, lo) = per_name_gap(long_df, gate)
            # Skip degenerate horizons.
            if per_name.empty:
                print(f"  [{h}ms] insufficient data.", flush=True)
                continue
            # Portfolio day-as-unit summary.
            port = portfolio_gap(daily)
            # Bucket profile for the plot.
            prof = bucket_profile(long_df, gate)
            # Persist CSVs (per-name axis retained).
            per_name.to_csv(out_dir / f"{gate}_pername_{h}ms.csv", index=False)
            daily.to_csv(out_dir / f"{gate}_daily_{h}ms.csv", index=False)
            prof.to_csv(out_dir / f"{gate}_profile_{h}ms.csv", index=False)
            # Plot both panels.
            plot_gate(per_name, prof, gate, h, out_dir / f"{gate}_{h}ms.png")
            # Headline print for the primary horizon.
            tag = "  <== HEADLINE" if h == HEADLINE_MS else ""
            print(f"  [{h}ms] portfolio gap={port['gap_bps']:+.3f} bps  t={port['t']:+.2f}  "
                  f"wins {port['wins']}/{port['total']}  (thr hi={hi:.3f}){tag}", flush=True)
            # For the headline, print the per-name breakdown (never drop the axis).
            if h == HEADLINE_MS:
                # Count names helped/hurt significantly (|t|>2).
                helped = int(((per_name["gap_bps"] > 0) & (per_name["t"] > 2)).sum())
                hurt = int(((per_name["gap_bps"] < 0) & (per_name["t"] < -2)).sum())
                print(f"          per-name: {helped} sig-helped, {hurt} sig-hurt, {len(per_name)} names", flush=True)
            # Free the (possibly 300M-row) frame before building the next horizon.
            del long_df
            # Close the heartbeat for this horizon with total elapsed.
            print(f"  [{h}ms] done in {time.perf_counter()-th:.1f}s", flush=True)
    # Done.
    print(f"\n[run] outputs -> {out_dir}", flush=True)


# ===========================================================================
# SYNTHETIC SELF-TEST -- inject KNOWN ground truth, assert recovery
# ===========================================================================
def _make_synthetic(seed=7, n_names=8, n_days=40, n_rows=500,
                    beta_obi=4.0, gamma_qdr=5.0):
    # Deterministic RNG for a reproducible test.
    rng = np.random.default_rng(seed)
    # Use a couple of thin + deep labels so plot coloring is exercised too.
    names = ["FNEL", "TPL", "NPL", "PACE", "OGDC", "PSO", "UBL", "HBL"][:n_names]
    # Accumulate per-symbol-day frames.
    frames = []
    # Build each symbol-day independently (state resets daily, like the store).
    for sym in names:
        for d in range(n_days):
            # Balanced imbalance in [-1,1].
            obi = rng.uniform(-1.0, 1.0, n_rows)
            # qdr is 0-inflated (70% zeros) then uniform positive, per side, independent.
            qdr_bid = np.where(rng.random(n_rows) < 0.7, 0.0, rng.uniform(0, 1, n_rows))
            qdr_ask = np.where(rng.random(n_rows) < 0.7, 0.0, rng.uniform(0, 1, n_rows))
            # Base label noise (bps).
            noise = rng.normal(0.0, 3.0, n_rows)
            # INJECTED TRUTH at 5s:
            #   +beta*obi   -> bid(+)/ask(-) favorable-side markout ~ +beta*|obi|
            #   -gamma*qdr_bid (bid eaten -> mid down -> +markout toxic for bid)
            #   +gamma*qdr_ask (ask eaten -> mid up   -> -markout toxic for ask)
            m5 = beta_obi * obi - gamma_qdr * qdr_bid + gamma_qdr * qdr_ask + noise
            # 1s = half strength; 30s = 1.2x strength (exercise multi-horizon).
            m1 = 0.5 * (beta_obi * obi - gamma_qdr * qdr_bid + gamma_qdr * qdr_ask) + rng.normal(0, 3, n_rows)
            m30 = 1.2 * (beta_obi * obi - gamma_qdr * qdr_bid + gamma_qdr * qdr_ask) + rng.normal(0, 3, n_rows)
            # A pure-noise NULL column to prove no false positives.
            m_null = rng.normal(0.0, 3.0, n_rows)
            # Assemble the day frame with the real store's column names.
            frames.append(pd.DataFrame({
                "symbol": sym,
                "date": f"2026-01-{d+1:02d}",
                "obi_1": obi,
                "qdr_bid": qdr_bid,
                "qdr_ask": qdr_ask,
                "markout_1000ms_bps": m1,
                "markout_5000ms_bps": m5,
                "markout_30000ms_bps": m30,
                "markout_null_bps": m_null,
            }))
    # One big synthetic feature store.
    return pd.concat(frames, ignore_index=True), beta_obi, gamma_qdr


def self_test():
    # Build synthetic data with injected truth.
    fs, beta, gamma = _make_synthetic()
    # Announce.
    print(f"[self-test] synthetic: {len(fs):,} rows, {fs['symbol'].nunique()} names, "
          f"{fs['date'].nunique()} days; injected beta_obi={beta}, gamma_qdr={gamma}\n", flush=True)

    # ---- GATE 1: BOOST (favorable OBI) at 5s ----
    long_obi = _long_favorable_obi(fs, HEADLINE_MS)
    per_name_o, daily_o, (hi_o, lo_o) = per_name_gap(long_obi, "boost")
    port_o = portfolio_gap(daily_o)
    # Ground-truth expected gap = beta*(E|obi|_active - E|obi|_neutral) computed from arrays.
    s = long_obi["signal"].to_numpy()
    exp_gap_o = beta * (s[s >= hi_o].mean() - s[s <= lo_o].mean())
    print(f"[BOOST] portfolio gap={port_o['gap_bps']:+.3f} bps (expected ~{exp_gap_o:+.3f})  "
          f"t={port_o['t']:+.2f}  wins {port_o['wins']}/{port_o['total']}", flush=True)
    # ASSERT: sign positive, magnitude within 15% of truth, strongly significant.
    assert port_o["gap_bps"] > 0, "BOOST gap should be POSITIVE under injected favorable-OBI truth"
    assert abs(port_o["gap_bps"] - exp_gap_o) / abs(exp_gap_o) < 0.15, "BOOST gap magnitude off truth by >15%"
    assert port_o["t"] > 5, "BOOST effect should be strongly significant on synthetic"
    # ASSERT: per-name axis present with one row per name.
    assert len(per_name_o) == fs["symbol"].nunique(), "per-name axis dropped in BOOST"

    # ---- GATE 2: QDR at 5s ----
    long_q = _long_qdr(fs, HEADLINE_MS)
    per_name_q, daily_q, (hi_q, lo_q) = per_name_gap(long_q, "qdr")
    port_q = portfolio_gap(daily_q)
    # Expected QDR gap = -gamma * E[qdr | active] (neutral qdr==0 contributes 0).
    sq = long_q["signal"].to_numpy()
    exp_gap_q = -gamma * sq[(sq >= hi_q) & (sq > 0)].mean()
    print(f"[QDR]   portfolio gap={port_q['gap_bps']:+.3f} bps (expected ~{exp_gap_q:+.3f})  "
          f"t={port_q['t']:+.2f}  wins {port_q['wins']}/{port_q['total']}", flush=True)
    # ASSERT: sign negative (toxic), magnitude within 15%, strongly significant negative.
    assert port_q["gap_bps"] < 0, "QDR gap should be NEGATIVE (toxic) under injected truth"
    assert abs(port_q["gap_bps"] - exp_gap_q) / abs(exp_gap_q) < 0.15, "QDR gap magnitude off truth by >15%"
    assert port_q["t"] < -5, "QDR effect should be strongly significant negative on synthetic"
    assert len(per_name_q) == fs["symbol"].nunique(), "per-name axis dropped in QDR"

    # ---- SIGN-FLIP GUARD: deliberately mis-sign the ask side, expect the gap to move ----
    bad = long_obi.copy()
    # Corrupt: flip the sign of every signed_markout (simulates a side-sign bug).
    bad["signed_markout"] = -bad["signed_markout"]
    _, daily_bad, _ = per_name_gap(bad, "boost")
    port_bad = portfolio_gap(daily_bad)
    # A correct harness makes the flipped gap the negation of the true gap.
    assert np.sign(port_bad["gap_bps"]) == -np.sign(port_o["gap_bps"]), "sign-flip guard failed"
    print(f"[flip]  corrupted-sign gap={port_bad['gap_bps']:+.3f} (must be opposite of BOOST) OK", flush=True)

    # ---- NULL / FALSE-POSITIVE GUARD: a pure-noise label must show no effect ----
    fs_null = fs.rename(columns={"markout_null_bps": "markout_5000ms_bps_NULL"})
    # Reuse the OBI signer but point it at the NULL label by temporary rename.
    tmp = fs.copy()
    tmp["markout_5000ms_bps"] = fs["markout_null_bps"]
    long_null = _long_favorable_obi(tmp, HEADLINE_MS)
    _, daily_null, _ = per_name_gap(long_null, "boost")
    port_null = portfolio_gap(daily_null)
    print(f"[null]  pure-noise gap={port_null['gap_bps']:+.3f} bps  t={port_null['t']:+.2f}  (must be ~0)", flush=True)
    # ASSERT: no spurious signal (small |t|).
    assert abs(port_null["t"]) < 4, "NULL guard: false-positive signal on pure noise"

    # ---- Produce the plots on synthetic data so the format is visible ----
    # Write self-test plots next to THIS script (always writable, predictable).
    out = Path(__file__).resolve().parent / "selftest_out"
    out.mkdir(parents=True, exist_ok=True)
    prof_o = bucket_profile(long_obi, "boost")
    prof_q = bucket_profile(long_q, "qdr")
    plot_gate(per_name_o, prof_o, "boost", HEADLINE_MS, out / "selftest_boost_5000ms.png")
    plot_gate(per_name_q, prof_q, "qdr", HEADLINE_MS, out / "selftest_qdr_5000ms.png")
    print(f"\n[self-test] ALL ASSERTIONS PASSED. Plots -> {out}", flush=True)


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    # CLI: choose self-test or a real run.
    ap = argparse.ArgumentParser(description="Boost & QDR pre-quoting markout gates.")
    # Validate the harness on synthetic ground truth.
    ap.add_argument("--self-test", action="store_true", help="run synthetic ground-truth validation")
    # Run against the real feature store.
    ap.add_argument("--run", action="store_true", help="run against the real feature store")
    # Optional symbol subset for a quick real smoke.
    ap.add_argument("--symbols", nargs="*", default=None, help="optional symbol subset")
    # Optional gate subset: e.g. --gates qdr  (default: both).
    ap.add_argument("--gates", nargs="*", default=None, choices=["boost", "qdr"],
                    help="which gates to run (default both)")
    # Parse.
    args = ap.parse_args()
    # Default to the self-test when nothing is chosen (safest).
    if args.self_test or not args.run:
        self_test()
    # Real run only when explicitly requested.
    if args.run:
        run_real(symbols=args.symbols, gates=args.gates)
