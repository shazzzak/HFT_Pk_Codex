# ============================================================================
# throttle_sweep_analysis.py
# ----------------------------------------------------------------------------
# Consumes the per-(config, name, day) output of throttle_param_sweep.py and
# answers ONE question honestly: is there a throttle (frac, hold) setting that
# BEATS the incumbent out-of-sample, or is the grid's argmax just noise?
#
# Anti-overfit machinery (the whole point of this file):
#   * DAY-AS-UNIT paired tests everywhere (never pool row/cell counts).
#   * PER-NAME axis never dropped.
#   * PRE-REGISTERED selection rule (declared in select_config(), applied blind).
#   * OUT-OF-SAMPLE split: argmax chosen on TUNE days, judged on HOLDOUT days.
#   * PLATEAU vs SPIKE reporting (a robust neighborhood beats a lucky cell).
#
# Value column: uses per-cell 'bps' when a 'notional' column is present
# (bps = net_pnl / notional * 1e4); else falls back to PKR 'net_pnl' with a
# LOUD warning, because pooling PKR across names is dominated by the big names
# and is NOT comparable to the +0.852 bps headline.
#
# USAGE:
#   python throttle_sweep_analysis.py --self-test        # synthetic validation
#   python throttle_sweep_analysis.py --csv sweep.csv    # analyze a real sweep
# ============================================================================

# CLI parsing.
import argparse
# Paths.
from pathlib import Path
# Arrays.
import numpy as np
# DataFrames.
import pandas as pd
# Headless plotting.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional SciPy for exact t p-values; normal-approx fallback otherwise.
try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:
    _sps = None
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# The incumbent throttle point (the current deployed candidate).
INCUMBENT = dict(frac=0.5, hold=300.0)
# Label used for the throttle-OFF baseline config in the sweep output.
OFF_LABEL = "OFF"
# Minimum |t| vs incumbent required to PREFER a new config (pre-registered).
PREFER_T = 2.0
# A name is "significantly hurt" if its paired t vs incumbent < this.
HURT_T = -2.0
# OOS generalization bar: holdout gain must be at least this fraction of tune gain.
OOS_KEEP_FRAC = 0.7
# Round-trip fee reference (bps) for plot context.
FEE_RT_BPS = 1.55
# Thin/deep universes for per-name plot coloring.
THIN = {"FNEL", "TPL", "PACE", "PIAHCLA", "HASCOL", "NPL", "TOMCL"}
DEEP = {"OGDC", "PSO", "HUBC", "FFC", "PPL", "UBL", "NBP", "MEBL"}


# ===========================================================================
# STATS -- day-as-unit
# ===========================================================================
def _t(x):
    # Coerce and drop NaNs.
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    # Need >=2 units.
    n = x.size
    if n < 2:
        return np.nan, np.nan, np.nan, n
    # Mean and standard error (ddof=1).
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    # Degenerate variance guard.
    if se == 0:
        return m, np.nan, np.nan, n
    # t-stat vs 0.
    t = m / se
    return m, se, t, n


# ===========================================================================
# VALUE COLUMN -- bps if notional present, else PKR with a warning
# ===========================================================================
def value_column(df):
    # Prefer scale-free bps for cross-name pooling.
    if "notional" in df.columns and df["notional"].abs().sum() > 0:
        # Per-cell bps = pnl / notional * 1e4.
        df = df.copy()
        df["value"] = df["net_pnl"] / df["notional"].replace(0, np.nan) * 1e4
        return df, "bps"
    # Fallback: PKR. Valid for PER-NAME (same name across its days) but NOT for
    # cross-name portfolio pooling -- warn loudly.
    df = df.copy()
    df["value"] = df["net_pnl"]
    print("  WARNING: no 'notional' column -> using PKR. Per-name results are valid; "
          "portfolio pooling across names is NOT comparable to the +0.852 bps headline. "
          "Supply notional per (config,name,day) to get bps.", flush=True)
    return df, "PKR"


# ===========================================================================
# PORTFOLIO (day-as-unit; unit = DATE, names pooled within a day)
# ===========================================================================
def portfolio_by_config_date(df):
    # Mean value over names within each (config, date). Matches the throttle-sweep
    # portfolio convention (day as the unit, names averaged inside the day).
    return df.groupby(["config", "date"])["value"].mean().reset_index()


def paired_portfolio(port, cfg_a, cfg_b, date_mask=None):
    # Wide table: one column per config, indexed by date -> paired across dates.
    w = port.pivot(index="date", columns="config", values="value")
    # Restrict to a day subset (for OOS splits) if given.
    if date_mask is not None:
        w = w.loc[w.index.isin(date_mask)]
    # Both configs must exist.
    if cfg_a not in w.columns or cfg_b not in w.columns:
        return dict(gap=np.nan, se=np.nan, t=np.nan, n=0, wins=0)
    # Per-date paired difference (only dates where both ran).
    d = (w[cfg_a] - w[cfg_b]).dropna().to_numpy()
    # Day-as-unit moments.
    m, se, t, n = _t(d)
    # Win rate for the difference.
    wins = int((d > 0).sum())
    return dict(gap=m, se=se, t=t, n=n, wins=wins)


# ===========================================================================
# PER-NAME (unit = that name's days), paired A vs B -- NEVER dropped
# ===========================================================================
def per_name_paired(df, cfg_a, cfg_b, date_mask=None):
    # Optional day subset.
    d = df if date_mask is None else df[df["date"].isin(date_mask)]
    # Rows.
    rows = []
    # One row per name.
    for sym, g in d.groupby("symbol", sort=False):
        # Wide per date for this name.
        w = g.pivot_table(index="date", columns="config", values="value", aggfunc="mean")
        # Need both configs.
        if cfg_a not in w.columns or cfg_b not in w.columns:
            continue
        # Paired daily diff.
        diff = (w[cfg_a] - w[cfg_b]).dropna().to_numpy()
        # Day-as-unit moments for this name.
        m, se, t, n = _t(diff)
        rows.append((sym, m, se, t, n))
    # Assemble sorted by effect.
    return pd.DataFrame(rows, columns=["symbol", "gap", "se", "t", "n_days"]).sort_values("gap").reset_index(drop=True)


# ===========================================================================
# SURFACE over (frac, hold) on a given day split -- for plateau vs spike
# ===========================================================================
def surface(df, port, date_mask=None):
    # ON configs only (exclude OFF).
    on = df[df["config"] != OFF_LABEL][["config", "frac", "hold"]].drop_duplicates()
    # Portfolio mean+se per ON config on the chosen split, measured vs OFF.
    recs = []
    # Walk each ON config.
    for _, r in on.iterrows():
        # Paired gain over OFF on this split (day-as-unit).
        res = paired_portfolio(port, r["config"], OFF_LABEL, date_mask=date_mask)
        recs.append((r["config"], r["frac"], r["hold"], res["gap"], res["se"], res["t"], res["n"]))
    # Frame of the surface points.
    return pd.DataFrame(recs, columns=["config", "frac", "hold", "gain_vs_off", "se", "t", "n"])


# ===========================================================================
# PRE-REGISTERED SELECTION -- declared here, applied blind
# ===========================================================================
def select_config(df):
    # Value column + label.
    df, unit = value_column(df)
    # Portfolio per (config, date).
    port = portfolio_by_config_date(df)
    # Deterministic TUNE/HOLDOUT split by ODD/EVEN rank of sorted unique dates.
    dates = np.array(sorted(df["date"].unique()))
    tune = set(dates[0::2])      # odd-index dates
    holdout = set(dates[1::2])   # even-index dates
    # Incumbent config label (must exist in the sweep).
    inc = _cfg_label(INCUMBENT["frac"], INCUMBENT["hold"])
    # Surface on the TUNE split (this is what we are allowed to optimize on).
    surf_tune = surface(df, port, date_mask=tune)
    # Argmax ON config on TUNE (the candidate).
    cand_row = surf_tune.sort_values("gain_vs_off", ascending=False).iloc[0]
    cand = cand_row["config"]
    # --- pre-registered acceptance tests ---
    # (1) candidate beats incumbent on TUNE with day-as-unit t > PREFER_T.
    tune_vs_inc = paired_portfolio(port, cand, inc, date_mask=tune)
    # (2) candidate GENERALIZES: holdout gain-over-OFF >= OOS_KEEP_FRAC * tune gain-over-OFF.
    cand_tune_gain = paired_portfolio(port, cand, OFF_LABEL, date_mask=tune)["gap"]
    cand_hold_gain = paired_portfolio(port, cand, OFF_LABEL, date_mask=holdout)["gap"]
    # Ratio guarded against a non-positive tune gain.
    generalizes = (cand_tune_gain > 0) and (cand_hold_gain >= OOS_KEEP_FRAC * cand_tune_gain)
    # (3) candidate hurts no name significantly (full-sample per-name vs incumbent).
    pn = per_name_paired(df, cand, inc)
    hurts = int((pn["t"] < HURT_T).sum())
    # Verdict: prefer candidate only if ALL three hold; else keep incumbent.
    prefer = (tune_vs_inc["t"] is not np.nan and tune_vs_inc["t"] > PREFER_T
              and generalizes and hurts == 0)
    # Package the decision.
    return dict(unit=unit, inc=inc, cand=cand, cand_frac=cand_row["frac"], cand_hold=cand_row["hold"],
                tune_vs_inc_t=tune_vs_inc["t"], cand_tune_gain=cand_tune_gain,
                cand_hold_gain=cand_hold_gain, generalizes=generalizes,
                hurts=hurts, prefer=bool(prefer), tune=tune, holdout=holdout,
                surf_tune=surf_tune, port=port, per_name=pn, df=df)


def _cfg_label(frac, hold):
    # Canonical config label the driver must also emit (keep in ONE place).
    return f"F{frac:g}_H{hold:g}"


# ===========================================================================
# PLOTS
# ===========================================================================
def _color(sym):
    # Thin red, deep blue, else grey.
    return "#c0392b" if sym in THIN else ("#2c6fbb" if sym in DEEP else "#888888")


def plot_surface(sel, out_png):
    # Heatmap of gain-vs-OFF over (frac, hold) on the FULL sample + per-name bars.
    df, port = sel["df"], sel["port"]
    # Full-sample surface for the heatmap.
    surf = surface(df, port, date_mask=None)
    # Pivot to a frac x hold grid.
    grid = surf.pivot(index="frac", columns="hold", values="gain_vs_off")
    # Two panels.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    # ---- LEFT: heatmap ----
    im = axL.imshow(grid.values, origin="lower", aspect="auto", cmap="RdBu_r",
                    vmin=-abs(np.nanmax(np.abs(grid.values))), vmax=abs(np.nanmax(np.abs(grid.values))))
    # Ticks = the actual frac/hold values.
    axL.set_xticks(range(len(grid.columns)))
    axL.set_xticklabels([f"{c:g}" for c in grid.columns])
    axL.set_yticks(range(len(grid.index)))
    axL.set_yticklabels([f"{r:g}" for r in grid.index])
    axL.set_xlabel("throttle_hold_ms")
    axL.set_ylabel("throttle_frac")
    axL.set_title(f"gain vs OFF ({sel['unit']}), full sample — look for a PLATEAU")
    # Annotate each cell with its value.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                axL.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
    # Mark incumbent and selected candidate.
    def _cell(frac, hold):
        return list(grid.columns).index(hold), list(grid.index).index(frac)
    try:
        xi, yi = _cell(INCUMBENT["frac"], INCUMBENT["hold"])
        axL.scatter([xi], [yi], marker="s", s=200, facecolors="none", edgecolors="k", lw=2, label="incumbent")
    except Exception:
        pass
    try:
        xc, yc = _cell(sel["cand_frac"], sel["cand_hold"])
        axL.scatter([xc], [yc], marker="*", s=260, color="gold", edgecolors="k", label="tune argmax")
    except Exception:
        pass
    axL.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=axL, fraction=0.046)
    # ---- RIGHT: per-name candidate vs incumbent ----
    pn = sel["per_name"]
    x = np.arange(len(pn))
    axR.bar(x, pn["gap"], yerr=pn["se"], capsize=2, color=[_color(s) for s in pn["symbol"]])
    axR.axhline(0, color="k", lw=0.8)
    axR.set_xticks(x)
    axR.set_xticklabels(pn["symbol"], rotation=90, fontsize=7)
    axR.set_ylabel(f"candidate - incumbent ({sel['unit']}), day-as-unit")
    axR.set_title("per-name: candidate vs incumbent (red=thin, blue=deep)")
    axR.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ===========================================================================
# REPORT
# ===========================================================================
def report(sel):
    # Verdict banner.
    print(f"\n  value unit          : {sel['unit']}")
    print(f"  incumbent config    : {sel['inc']}")
    print(f"  tune argmax (cand)  : {sel['cand']}  (frac={sel['cand_frac']:g}, hold={sel['cand_hold']:g})")
    print(f"  cand vs inc (tune)  : t={sel['tune_vs_inc_t']:+.2f}  (need > {PREFER_T})")
    print(f"  cand gain vs OFF    : tune={sel['cand_tune_gain']:+.3f}  holdout={sel['cand_hold_gain']:+.3f}  "
          f"(generalizes: {sel['generalizes']})")
    print(f"  names sig-hurt      : {sel['hurts']}  (need 0)")
    print(f"  >>> DECISION        : {'PREFER candidate' if sel['prefer'] else 'KEEP incumbent'}")


# ===========================================================================
# SYNTHETIC SELF-TEST -- two scenarios: REAL optimum vs pure NOISE
# ===========================================================================
def _make_sweep(scenario, seed=3, n_names=38, n_days=198,
                fracs=(0.25, 0.5, 0.75), holds=(150.0, 300.0, 600.0, 1000.0)):
    # RNG.
    rng = np.random.default_rng(seed)
    # 38 names incl. some thin/deep labels.
    base = ["FNEL", "TPL", "NPL", "PACE", "HASCOL", "OGDC", "PSO", "UBL", "HBL", "PPL",
            "FFC", "NBP", "MEBL", "HUBC"]
    names = (base + [f"N{i:02d}" for i in range(n_names - len(base))])[:n_names]
    # Dates as sortable strings.
    dates = [f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}" for d in range(n_days)]
    # Per-name scale (thin names smaller, adds heterogeneity).
    name_scale = {s: (0.6 if s in THIN else (1.3 if s in DEEP else 1.0)) for s in names}
    # Shared per-date market factor -> makes paired (vs OFF) SE small, like reality.
    date_factor = {dt: rng.normal(0, 1.4) for dt in dates}
    # TRUE benefit-over-OFF surface (bps) as a function of (frac, hold).
    def true_gain(frac, hold):
        # OFF has zero gain over itself.
        # A smooth bump peaking near frac=0.5, hold=600 for the REAL scenario.
        if scenario == "real":
            fr = -((frac - 0.5) ** 2) / 0.05
            ho = -((hold - 600.0) ** 2) / (400.0 ** 2)
            return 1.30 * np.exp(fr + ho) + 0.15   # plateau ~1.0-1.3 near (0.5,600)
        # NOISE scenario: every ON config truly equals the incumbent (+0.85), no real
        # differences among ON configs -> any argmax spread is pure sampling noise.
        if scenario == "noise":
            return 0.85
        raise ValueError(scenario)
    # Build rows.
    rows = []
    # OFF config: gain 0 by definition; absolute level ~1.70 bps like the real panel.
    configs = [("OFF", None, None)] + [(_cfg_label(f, h), f, h) for f in fracs for h in holds]
    # Walk configs.
    for cfg, fr, ho in configs:
        # True absolute level for this config = OFF level + its true gain.
        g = 0.0 if cfg == "OFF" else true_gain(fr, ho)
        # Walk name-days.
        for s in names:
            for dt in dates:
                # Per-cell bps = base 1.70 + true gain + shared date factor + idiosyncratic noise.
                val = (1.70 + g) * name_scale[s] + date_factor[dt] + rng.normal(0, 2.0)
                # Emit with a synthetic notional so value_column computes bps.
                # Choose notional so net_pnl/notional*1e4 == val exactly.
                notional = 1e6 * name_scale[s]
                net_pnl = val * notional / 1e4
                rows.append((cfg, fr, ho, s, dt, net_pnl, notional))
    # Frame.
    return pd.DataFrame(rows, columns=["config", "frac", "hold", "symbol", "date", "net_pnl", "notional"])


def self_test():
    # ---- SCENARIO 1: a REAL optimum exists -> harness should PREFER & it generalizes ----
    df_real = _make_sweep("real")
    print("[self-test] scenario REAL (true bump near frac=0.5, hold=600)")
    sel_real = select_config(df_real)
    report(sel_real)
    # Argmax should land in the true plateau (hold in {600,1000}, frac 0.5).
    assert sel_real["cand_frac"] == 0.5, "REAL: argmax frac should be 0.5"
    assert sel_real["cand_hold"] in (600.0, 1000.0), "REAL: argmax hold should be in the plateau"
    # It should generalize (holdout gain ~ tune gain).
    assert sel_real["generalizes"], "REAL: true optimum must generalize OOS"
    # Per-name axis intact (all 38 names).
    assert len(sel_real["per_name"]) == 38, "REAL: per-name axis dropped"

    # ---- SCENARIO 2: pure NOISE among ON configs -> harness must NOT be fooled ----
    df_noise = _make_sweep("noise")
    print("\n[self-test] scenario NOISE (all ON configs truly equal)")
    sel_noise = select_config(df_noise)
    report(sel_noise)
    # The tune-argmax is a lucky cell; vs incumbent it must NOT clear t>2 on tune
    # AND/OR must fail to generalize -> the pre-registered rule keeps the incumbent.
    assert not sel_noise["prefer"], "NOISE: harness was FOOLED into preferring a noise-mined argmax"
    # And the candidate's edge over incumbent must be tiny (near zero true gain).
    inc = sel_noise["inc"]
    vs_inc_full = paired_portfolio(sel_noise["port"], sel_noise["cand"], inc)
    print(f"           noise cand vs incumbent (full): gap={vs_inc_full['gap']:+.3f}, t={vs_inc_full['t']:+.2f}")
    assert abs(vs_inc_full["gap"]) < 0.30, "NOISE: candidate-incumbent gap should be near zero"

    # ---- plots (real scenario) ----
    # Write self-test plots next to THIS script (always writable, predictable).
    out = Path(__file__).resolve().parent / "selftest_out"
    out.mkdir(parents=True, exist_ok=True)
    plot_surface(sel_real, out / "selftest_throttle_surface.png")
    print(f"\n[self-test] ALL ASSERTIONS PASSED. Surface plot -> {out}/selftest_throttle_surface.png")


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Throttle sweep analysis (OOS + pre-registered).")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--csv", type=str, default=None, help="per-(config,name,day) sweep CSV")
    args = ap.parse_args()
    # Default to self-test if no CSV.
    if args.self_test or not args.csv:
        self_test()
    else:
        # Real analysis.
        df = pd.read_csv(args.csv)
        sel = select_config(df)
        report(sel)
        outp = Path(args.csv).with_suffix(".surface.png")
        plot_surface(sel, outp)
        print(f"\n[analysis] surface -> {outp}")
