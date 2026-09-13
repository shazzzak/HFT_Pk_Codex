# ============================================================================
# build_config_assignment.py -- TO-DO A.1
# Emit config_assignment.csv: the per-name QT_2t / OBI / DROP decision the
# harness loads at start-up.
# ============================================================================
# READ-ONLY on every input. Writes ONE new timestamped CSV and ONE new PNG.
# Never overwrites, never deletes.
#
# WHAT THIS REPLACES
#   Today the deploy config is one global constant. Measured on 113 names over
#   197 sessions, a per-name assignment is worth +3.5% (2-way switch) to +6.8%
#   (3-way with drop) over always-QT_2t, walk-forward at a 60-day warm-up. That
#   is the number this file is built to capture -- NOT the +7.3% in-sample
#   oracle, which is unattainable by construction.
#
# THE DECISION RULE, stated before any code
#   1. CAPACITY OVERRIDE. capacity_flag != "ok" in the time-windows file means
#      the name cannot be unwound inside the policy cap at the modelled POV.
#      That is a risk limit, not a P&L opinion -> DROP regardless of P&L.
#   2. DROP. If BOTH configs lose money over the sample, no config fixes the
#      name. Assigning it the "less bad" config still loses money; not trading
#      it loses nothing.
#   3. HYSTERESIS on the QT_2t <-> OBI choice, driven by the PAIRED daily
#      t-statistic on (QT_2t - OBI):
#         t < -2.0  -> OBI      (OBI significantly better: ENTER OBI)
#         t > -1.0  -> QT_2t    (the OBI case has decayed: EXIT OBI)
#         between   -> keep whatever the previous assignment was
#      A single threshold makes a name oscillate on noise; the band means a
#      switch has to be earned twice -- once to get in, again to get out.
#   4. QT_2t IS THE DEFAULT. A name with no significant evidence stays on the
#      global winner. 23 of 113 names are statistically indistinguishable
#      between the two configs; they all default here.
#
# WHY THE PAIRED t AND NOT THE P&L GAP
#   Two names can both show "OBI ahead by 40k PKR" while one is a persistent
#   4-bps-a-day effect and the other is two good days. The paired daily
#   difference has its own standard error; the t-statistic is what separates
#   them. Point estimates on 197 days are not evidence on their own.
#
# WHAT THIS FILE IS NOT
#   A per-name flag is a PROXY for "this name's book usually looks like X". The
#   production endpoint is a state-conditional throttle keyed on book state
#   (spread regime, depth, time bucket) -- see to-do E.22. Evidence it is
#   time-varying: QT_2t's advantage is ~+4.0 bps in preclose45 and ~0 in
#   first15. This file is the simple, testable, shippable version; it is not
#   the ceiling.
#
# Run from existing_mm_live/:
#   caffeinate -is python build_config_assignment.py
# ============================================================================

# frames
import pandas as pd
# numeric
import numpy as np
# plotting, headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# for the refit-due date
from datetime import datetime, timedelta

# the ONE source of paths -- no hardcoded directory anywhere in this file
from config_pk import RESULTS_ROOT
# shared name lists, safe_out and the merge guards
import expansion_names as EX
# the harness, for newest() so the capacity file resolves exactly as the engine does
import mm_harness as H

# ---------------------------------------------------------------------------
# INPUTS -- explicit, because a glob would silently pick up the wrong run
# ---------------------------------------------------------------------------
# the 38 incumbents, re-run 2026-09-13 under the CORRECTED calibration
PERNAME_38 = RESULTS_ROOT / "universe_expand_PERNAME_20260913_1818.parquet"
# the 113-name file whose 75 NEW names are correct (its 38 are stale -> dropped)
PERNAME_75 = RESULTS_ROOT / "universe_expand_PERNAME_20260913_1253.parquet"

# RECONCILIATION ANCHORS: the engine P&L each cohort/config must reproduce.
# Taken from the two run CSVs. If a sum here moves, the wrong file was loaded
# or the grain assumption broke -- abort rather than emit a wrong assignment.
ANCHORS = {("38", "OBI"):   2724699.5757758836,
           ("38", "QT_2t"): 8543789.948386392,
           ("75", "OBI"):   1509521.6972865653,
           ("75", "QT_2t"): 4742150.606553214}
# absolute PKR tolerance on the reconciliation (float sum order only)
RECON_TOL = 1e-4

# --- the decision thresholds, all in one place ---
# paired t below this ENTERS the OBI assignment
T_ENTER_OBI = -2.0
# paired t above this EXITS the OBI assignment back to QT_2t
T_EXIT_OBI = -1.0
# a dropped name only returns if its best config is profitable AND its own
# daily t clears this -- stops a name hovering at zero from flip-flopping
T_REENTRY = 2.0
# MATERIALITY FLOOR in PKR per trading day. A live name below this earns less
# than it costs in operational surface. 0.0 = off, which reproduces the
# walk-forward-validated rule exactly. Measured on the 2026-09-13 run: 19 of 99
# live names sit under 200 PKR/day and contribute 1.96% of assigned P&L.
MIN_PKR_PER_DAY = 0.0
# how long this assignment is good for
REFIT_DAYS = 91

# OPTIONAL: the previous assignment, for hysteresis. None on the first run.
# Set to a Path to make the band actually bind on the next refit.
PRIOR_ASSIGNMENT = None


def load_cohort(path, keep, label):
    """Load one PERNAME parquet, restrict to `keep`, collapse buckets away.

    Returns a frame at the (symbol, date, throttle) grain with net_pkr,
    opened_notional and fills summed over the four time buckets.
    """
    # read the full per-name table
    df = pd.read_parquet(path)
    # GRAIN GUARD: the file must be unique on (throttle, symbol, date, bucket).
    # A duplicated key here silently doubles every downstream sum.
    key = ["throttle", "symbol", "date", "bucket"]
    # count rows vs distinct keys
    if len(df) != len(df.drop_duplicates(key)):
        # abort loudly -- this is the stage0_audit lesson
        raise SystemExit(f"{path.name}: not unique on {key}")
    # keep only this cohort's symbols
    df = df[df["symbol"].isin(keep)].copy()
    # tag the cohort so the reconciliation can be done per cohort
    df["cohort"] = label
    # collapse the bucket dimension: one row per symbol-day-config
    out = (df.groupby(["cohort", "symbol", "date", "throttle"], as_index=False)
             [["net_pkr", "opened_notional", "fills"]].sum())
    # hand it back
    return out


def main():
    # ---- load both cohorts -------------------------------------------------
    print("=" * 78)
    print("CONFIG ASSIGNMENT -- per-name QT_2t / OBI / DROP")
    print("=" * 78)
    # the 38 incumbents from the corrected re-run
    a = load_cohort(PERNAME_38, set(EX.INCUMBENT), "38")
    # the new names from the 09-13 run, explicitly EXCLUDING the stale 38
    b = load_cohort(PERNAME_75, set(EX.ALL_NAMES) - set(EX.INCUMBENT), "75")
    # one table
    df = pd.concat([a, b], ignore_index=True)
    # say what was loaded
    print(f"  {PERNAME_38.name}  -> {a.symbol.nunique()} names")
    print(f"  {PERNAME_75.name}  -> {b.symbol.nunique()} names")

    # ---- HARD RECONCILIATION before a single decision is made --------------
    print("\nRECONCILIATION vs engine P&L")
    # walk every (cohort, config) anchor
    for (coh, cfg), want in sorted(ANCHORS.items()):
        # the sum this load actually produces
        got = df[(df.cohort == coh) & (df.throttle == cfg)].net_pkr.sum()
        # the signed miss
        gap = got - want
        # report it to the cent
        print(f"  {coh}/{cfg:6s} {got:>16,.4f} vs {want:>16,.4f}   diff {gap:+.6f}")
        # a mismatch means the inputs are not what this script assumes
        if abs(gap) > RECON_TOL:
            raise SystemExit(f"RECONCILIATION FAILED for {coh}/{cfg}. "
                             f"Wrong input file or broken grain -- refusing to "
                             f"emit an assignment.")
    # every anchor held
    print("  all anchors hold.")

    # ---- per symbol-day, the two configs side by side ----------------------
    # pivot so each row is one symbol-day with an OBI and a QT_2t column
    wide = df.pivot_table(index=["cohort", "symbol", "date"],
                          columns="throttle", values="net_pkr").reset_index()
    # PAIRING GUARD: a day present for one config but not the other cannot be
    # differenced. Drop it and say how many went.
    before = len(wide)
    # keep only fully paired symbol-days
    wide = wide.dropna(subset=["OBI", "QT_2t"])
    # report any loss
    if len(wide) < before:
        print(f"  dropped {before - len(wide)} unpaired symbol-days")
    # the daily difference that the whole decision rests on
    wide["diff"] = wide["QT_2t"] - wide["OBI"]

    # ---- per-name statistics ----------------------------------------------
    # a place to build rows
    recs = []
    # walk each name
    for sym, g in wide.groupby("symbol"):
        # number of paired days
        n = len(g)
        # total P&L under each config
        tot_o, tot_q = g["OBI"].sum(), g["QT_2t"].sum()
        # the paired daily difference series
        d = g["diff"].to_numpy()
        # its mean
        dm = d.mean()
        # its sample standard deviation (ddof=1: this is a sample, not a population)
        dsd = d.std(ddof=1)
        # the paired t-statistic; undefined when the two configs are identical
        # every day (the CHEAP_EXCLUDED names, where the skew is inert)
        t_diff = dm / (dsd / np.sqrt(n)) if dsd > 0 else np.nan
        # the better config's own daily series, for the drop / re-entry test
        best_series = g["QT_2t"].to_numpy() if tot_q >= tot_o else g["OBI"].to_numpy()
        # its own standard deviation
        bsd = best_series.std(ddof=1)
        # its own t-statistic: is this name's P&L distinguishable from zero?
        t_best = (best_series.mean() / (bsd / np.sqrt(n))) if bsd > 0 else np.nan
        # traded notional under each config, for reporting
        no = df[(df.symbol == sym)].groupby("throttle").opened_notional.sum()
        # one record per name
        recs.append({"symbol": sym,
                     "cohort": g["cohort"].iloc[0],
                     "days": n,
                     "obi_pkr": tot_o, "qt2t_pkr": tot_q,
                     "obi_bps": tot_o / no.get("OBI", np.nan) * 1e4,
                     "qt2t_bps": tot_q / no.get("QT_2t", np.nan) * 1e4,
                     "t_diff": t_diff, "t_best": t_best,
                     "best_pkr": max(tot_o, tot_q)})
    # the per-name table
    P = pd.DataFrame(recs).set_index("symbol").sort_index()

    # ---- capacity flags, read the way the engine reads them ----------------
    # the newest time-windows file -- same resolution mm_harness uses
    tw_path = H.newest("time_windows_*.csv")
    # load it
    tw = pd.read_csv(tw_path).set_index("symbol")
    # attach the capacity flag, defaulting to "ok" when the column/name is absent
    P["capacity_flag"] = (tw["capacity_flag"] if "capacity_flag" in tw.columns
                          else pd.Series(dtype=object)).reindex(P.index).fillna("ok")
    # name the file in the log so the provenance is recorded
    print(f"\n  capacity from {tw_path.name}")

    # ---- the previous assignment, for hysteresis --------------------------
    # default: no prior, so the band cannot bind yet
    prior = pd.Series("", index=P.index)
    # load one if configured
    if PRIOR_ASSIGNMENT is not None:
        # read the previous decision
        pv = pd.read_csv(PRIOR_ASSIGNMENT).set_index("symbol")["assigned_config"]
        # align it to this run's names
        prior = pv.reindex(P.index).fillna("")
        # say so
        print(f"  hysteresis against {PRIOR_ASSIGNMENT.name}")
    else:
        # be explicit that the band is inert on a first run
        print("  no prior assignment -> hysteresis band inert this run "
              "(set PRIOR_ASSIGNMENT on the next refit)")
    # record what we compared against
    P["prior_config"] = prior

    # ---- apply the rule ----------------------------------------------------
    def decide(r):
        """Return (assigned_config, reason) for one name.

        ORDER MATTERS. The gate is applied to the config this function actually
        CHOOSES, never to the better of the two. Testing `best_pkr` and then
        assigning the other config lets a money-losing assignment through the
        gate -- ASL and WTL did exactly that on the 2026-09-13 run (both were
        assigned QT_2t, which loses on those names, because their paired t was
        not past the entry threshold while OBI carried the positive P&L).
        """
        # 1. capacity is a risk limit and outranks every P&L consideration
        if r.capacity_flag != "ok":
            # cannot be unwound inside the cap -> never quote it
            return "DROP", f"capacity:{r.capacity_flag}"
        # 2. BOTH configs lose -> no config fixes this name
        if max(r.obi_pkr, r.qt2t_pkr) <= 0:
            # a dropped name only returns on a significant positive
            if r.prior_config == "DROP" and not r.t_best > T_REENTRY:
                return "DROP", "loses under both (held)"
            # first-time drop
            return "DROP", "loses under both"
        # 3. CHOOSE the config. The cheap-tick names have no difference to test
        #    (QT_2t and OBI are byte-identical every day) -> the choice is
        #    cosmetic, so name that case rather than let a NaN fall through.
        if pd.isna(r.t_diff):
            # the skew is inert on a one-tick book
            cfg, why = "OBI", "skew inert (cheap tick)"
        elif r.t_diff < T_ENTER_OBI:
            # OBI significantly better -> enter/hold OBI
            cfg, why = "OBI", f"t={r.t_diff:.2f} < {T_ENTER_OBI}"
        elif r.prior_config == "OBI" and r.t_diff <= T_EXIT_OBI:
            # inside the band and already on OBI -> hold, do not churn
            cfg, why = "OBI", f"t={r.t_diff:.2f} in band, held"
        else:
            # default: the global winner
            cfg, why = "QT_2t", (f"t={r.t_diff:.2f}" if r.t_diff > T_EXIT_OBI
                                 else f"t={r.t_diff:.2f} in band, default")
        # the P&L of the config we just chose -- NOT the better of the two
        chosen = r.obi_pkr if cfg == "OBI" else r.qt2t_pkr
        # 4. GATE THE CHOICE. Quoting a config that lost money on this name over
        #    197 sessions is never correct, whatever the other config did.
        if chosen <= 0:
            # say which config was rejected, so the log explains itself
            return "DROP", f"{cfg} loses ({chosen:,.0f})"
        # 5. MATERIALITY FLOOR. A name earning a few PKR a day consumes a
        #    quoting slot, risk budget and monitoring attention for nothing.
        #    Default 0.0 keeps the validated rule unchanged; raise it to prune.
        if MIN_PKR_PER_DAY > 0 and (chosen / r.days) < MIN_PKR_PER_DAY:
            # immaterial, not unprofitable -- a distinct reason
            return "DROP", f"immaterial ({chosen / r.days:,.0f}/day)"
        # the surviving assignment
        return cfg, why
    # apply it row by row
    P[["assigned_config", "reason"]] = P.apply(
        lambda r: pd.Series(decide(r)), axis=1)
    # did this refit move the name?
    P["changed"] = (P["prior_config"] != "") & (P["prior_config"] != P["assigned_config"])
    # the P&L the assignment is expected to realise (0 for a dropped name)
    P["assigned_pkr"] = np.where(P.assigned_config == "DROP", 0.0,
                                 np.where(P.assigned_config == "OBI",
                                          P.obi_pkr, P.qt2t_pkr))
    # when this assignment should be rebuilt
    P["as_of"] = datetime.now().strftime("%Y-%m-%d")
    P["refit_due"] = (datetime.now() + timedelta(days=REFIT_DAYS)).strftime("%Y-%m-%d")

    # ---- report ------------------------------------------------------------
    # headline counts
    counts = P.assigned_config.value_counts()
    print("\n" + "=" * 78)
    for cfg in ("QT_2t", "OBI", "DROP"):
        # how many names and what they are worth
        sel = P[P.assigned_config == cfg]
        print(f"  {cfg:6s} {len(sel):>4} names   assigned P&L {sel.assigned_pkr.sum():>14,.0f} PKR")
    # the baseline this must beat
    base = P.qt2t_pkr.sum()
    # what the assignment produces
    tot = P.assigned_pkr.sum()
    print("=" * 78)
    print(f"  always-QT_2t (baseline)  {base:>14,.0f} PKR")
    print(f"  assigned                 {tot:>14,.0f} PKR   "
          f"({tot - base:+,.0f}, {(tot/base - 1)*100:+.1f}%)")
    print("  NOTE: that gap is IN-SAMPLE. Walk-forward on the same rule and")
    print("        data measured +3.5% (2-way) / +6.8% (3-way with drop) at a")
    print("        60-day warm-up. Budget the walk-forward number.")
    # the names that are not on the default
    print("\nNON-DEFAULT ASSIGNMENTS")
    nd = P[P.assigned_config != "QT_2t"].sort_values("t_diff")
    print(nd[["cohort", "assigned_config", "obi_pkr", "qt2t_pkr", "t_diff",
              "t_best", "capacity_flag", "reason"]].to_string(
        float_format=lambda v: f"{v:,.2f}"))
    # churn, once a prior exists
    if P.changed.any():
        print(f"\nCHANGED SINCE LAST REFIT: {int(P.changed.sum())}")
        print(P[P.changed][["prior_config", "assigned_config", "t_diff"]].to_string())

    # ---- write -------------------------------------------------------------
    # a fresh timestamped destination; never overwrites
    out = EX.safe_out("config_assignment", "csv")
    # the columns the harness needs plus the evidence behind each decision
    P.reset_index()[["symbol", "cohort", "assigned_config", "reason",
                     "t_diff", "t_best", "obi_pkr", "qt2t_pkr",
                     "obi_bps", "qt2t_bps", "capacity_flag", "days",
                     "prior_config", "changed", "as_of", "refit_due"]
                    ].to_csv(out, index=False)
    # say where it went
    print(f"\nwrote {out}")

    # ---- chart -------------------------------------------------------------
    # one figure: the decision map in PKR space
    fig, ax = plt.subplots(figsize=(10.5, 9), facecolor="#fcfcfb")
    # the plotting surface
    ax.set_facecolor("#fcfcfb")
    # drop the top/right frame
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # mute the remaining frame
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#e3e2df")
    # a recessive grid behind the marks
    ax.grid(color="#e3e2df", lw=0.7)
    ax.set_axisbelow(True)
    # the identity line: above it QT_2t wins, below it OBI wins
    lo = min(P.obi_pkr.min(), P.qt2t_pkr.min()) * 1.1
    hi = max(P.obi_pkr.max(), P.qt2t_pkr.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1.5, ls="--", zorder=2)
    # zero axes
    ax.axhline(0, color="#e3e2df", lw=1.2)
    ax.axvline(0, color="#e3e2df", lw=1.2)
    # one series per decision, so colour encodes the assignment
    for cfg, col, mk in (("QT_2t", "#2a78d6", "o"),
                         ("OBI", "#eb6834", "s"),
                         ("DROP", "#78858f", "x")):
        # this decision's names
        s = P[P.assigned_config == cfg]
        # plot them
        ax.scatter(s.obi_pkr, s.qt2t_pkr, s=58, c=col, marker=mk,
                   edgecolor="#fcfcfb" if mk != "x" else None, linewidth=1.4,
                   zorder=3, label=f"{cfg} ({len(s)})")
    # label only the names that are not on the default
    for sym, r in P[P.assigned_config != "QT_2t"].iterrows():
        # a small direct label beside the mark
        ax.annotate(sym, (r.obi_pkr, r.qt2t_pkr), fontsize=7, color="#0b0b0b",
                    xytext=(6, 3), textcoords="offset points")
    # square the axes so the identity line is a true 45 degrees
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # thousands, not raw PKR
    fmt = matplotlib.ticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}k")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    # axis titles
    ax.set_xlabel("OBI control — net PKR", color="#52514e", fontsize=10)
    ax.set_ylabel("QT_2t throttle — net PKR", color="#52514e", fontsize=10)
    # what the reader is looking at
    ax.set_title(f"Config assignment, {len(P)} names\n"
                 f"{counts.get('QT_2t',0)} QT_2t · {counts.get('OBI',0)} OBI · "
                 f"{counts.get('DROP',0)} drop",
                 color="#0b0b0b", fontsize=12.5, loc="left", pad=10)
    # identity is never colour-alone: the legend is always present
    lg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    # legend text takes ink colour, not the series colour
    for t in lg.get_texts():
        t.set_color("#0b0b0b")
    # tidy margins
    fig.tight_layout()
    # a fresh timestamped PNG; never overwrites
    png = EX.safe_out("config_assignment", "png")
    # write it
    fig.savefig(png, dpi=150, facecolor="#fcfcfb")
    # say where it went
    print(f"wrote {png}")


# entry point
if __name__ == "__main__":
    main()
