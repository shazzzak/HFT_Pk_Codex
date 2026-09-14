# ============================================================================
# build_config_assignment.py -- TO-DO A.1 (three-bucket version, 2026-09-14)
# Emit config_assignment.csv: for every name, which of FOUR settings to run:
#     lean off            (plain quoting, label OBI)
#     lean, trigger 0.20  (two-tick lean, fires only on a strongly lopsided book)
#     lean, trigger 0.15  (two-tick lean, the incumbent)
#     not quoted          (DROP)
# ============================================================================
# READ-ONLY on every input. Writes ONE new timestamped CSV and ONE new PNG.
# Never overwrites, never deletes.
#
# WHAT CHANGED FROM THE TWO-BUCKET VERSION
#   The 2026-09-14 gate sweep (30 names, 8 arms) and its 110-name confirmation
#   showed the lean's TRIGGER is a per-name dosage dial: raising it from 0.15 to
#   0.20 helps the names the lean hurts and hurts the names the lean helps, by
#   about the same amount, so a uniform change is a wash (+0.13 bps, flat PKR).
#   Used per name it is worth something, but LESS than a first pass suggested.
#   Rolling walk-forward, 110 names, 10 folds (60-day fit, 20-day test, step 12),
#   gain over always-0.15, with the DROP gate applied to BOTH rules (an earlier
#   comparison omitted it from both and so described neither):
#       two buckets    off / 0.15         +5.45%  se 0.94  t +5.83  10/10 folds
#       three buckets  off / 0.20 / 0.15  +6.41%  se 1.22  t +5.25  10/10 folds
#       three MINUS two                   +0.96pp se 0.58  t +1.64   8/10 folds
#   So: the DROP gate carries nearly all the value and is solid. The third bucket
#   is probably worth about +1pp and was positive in 8 of 10 windows, but it does
#   NOT clear the |t| > 2 bar this project applies to every other decision. Ship
#   it as a small positive-expectation bet, not as an established result, and
#   re-measure at the next refit.
#
# THE DECISION RULE, stated before any code
#   1. CAPACITY OVERRIDE. capacity_flag != "ok" in the time-windows file means
#      the name cannot be unwound inside the policy cap. Risk limit, not a P&L
#      opinion -> DROP regardless of P&L.
#   2. DROP if every setting loses money over the sample. No setting fixes it.
#   3. CLASSIFY on the per-day EFFECT SIZE of (lean 0.15 - lean off):
#          d = mean(daily diff) / sd(daily diff)
#      NOT the t-statistic. t = d * sqrt(n) grows with the number of days even
#      when the underlying effect is unchanged, so a band written in t means a
#      different thing on 98 days than on 197 -- the middle bucket held 27 names
#      in the walk-forward fit windows and would hold 9 on the full year. The
#      band is therefore written in d and is the same rule at any sample size.
#      D_BAND = 2.0 / sqrt(98) = 0.202 is exactly the "t = 2 on a 98-day window"
#      that the walk-forward validated.
#          d >  +D_BAND  -> lean, trigger 0.15   (the lean clearly works here)
#          d <  -D_BAND  -> lean off             (the lean clearly loses here)
#          in between    -> the better of {lean off, lean 0.20} by sample PKR
#      Hysteresis: a name already in a bucket needs d to cross the FAR edge of
#      the band (D_EXIT) before it leaves, so it does not churn on noise.
#   4. ASSIGN THEN GATE. The P&L test is applied to the setting this file
#      actually CHOOSES, never to the best of the alternatives. (ASL and WTL
#      were once assigned a money-losing setting because the gate tested the
#      wrong column.)
#
# WHAT THIS FILE IS NOT
#   A per-name flag is a PROXY for "this name's book usually looks like X". The
#   production endpoint is a state-conditional throttle keyed on book state --
#   see to-do E.22. This is the simple, testable, shippable version.
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
# the 38 incumbents, re-run 2026-09-13 under the CORRECTED calibration (lean off + lean 0.15)
PERNAME_38 = RESULTS_ROOT / "universe_expand_PERNAME_20260913_1818.parquet"
# the 113-name file whose 75 NEW names are correct (its 38 are stale -> dropped)
PERNAME_75 = RESULTS_ROOT / "universe_expand_PERNAME_20260913_1253.parquet"
# the 110-name confirmation run of lean 0.20 (all names except the 3 cheap-tick ones)
PERNAME_20 = RESULTS_ROOT / "gate20_confirm_PERNAME_20260914_1809.parquet"

# the label each file uses for the lean at 0.15 -> the label this script uses
RELABEL = {"QT_2t": "QT_2t@15"}

# RECONCILIATION ANCHORS: the engine P&L each cohort/setting must reproduce.
# Taken from the three run CSVs. If a sum here moves, the wrong file was loaded
# or the grain assumption broke -- abort rather than emit a wrong assignment.
ANCHORS = {("38",  "OBI"):      2724699.5757758836,
           ("38",  "QT_2t@15"): 8543789.948386392,
           ("75",  "OBI"):      1509521.6972865653,
           ("75",  "QT_2t@15"): 4742150.606553214,
           ("110", "QT_2t@20"): 13360331.864938661}
# absolute PKR tolerance on the reconciliation (float sum order only)
RECON_TOL = 1e-4

# --- the decision thresholds, all in one place -----------------------------
# EFFECT-SIZE band. 2.0/sqrt(98) reproduces the "t = 2 on a 98-day window" the
# walk-forward validated, and means the same thing at any sample size.
D_BAND = 2.0 / np.sqrt(98)
# a name leaves its current bucket only when d crosses the FAR edge of the
# band (half-way back toward the middle), so noise cannot flip it
D_EXIT = D_BAND / 2.0
# a dropped name only returns if its chosen setting's own daily t clears this
T_REENTRY = 2.0
# MATERIALITY FLOOR in PKR per trading day. 0.0 = off (reproduces the validated rule)
MIN_PKR_PER_DAY = 0.0
# how long this assignment is good for
REFIT_DAYS = 91

# OPTIONAL: the previous assignment, for hysteresis. None on the first run.
PRIOR_ASSIGNMENT = None

# the settings, in the order they are reported, and what each means in plain words
SETTINGS = ["QT_2t@15", "QT_2t@20", "OBI", "DROP"]
# plain-English description written into the CSV beside the machine label
PLAIN = {"QT_2t@15": "lean, trigger 0.15",
         "QT_2t@20": "lean, trigger 0.20",
         "OBI":      "lean off",
         "DROP":     "not quoted"}
# the engine parameters each label stands for, so the harness reads numbers, not labels
PARAMS = {"QT_2t@15": (2.0, 0.15),
          "QT_2t@20": (2.0, 0.20),
          "OBI":      (0.0, np.nan),
          "DROP":     (np.nan, np.nan)}


def load_cohort(path, keep, label):
    """Load one PERNAME parquet, restrict to `keep`, collapse buckets away.

    Returns a frame at the (symbol, date, throttle) grain with net_pkr,
    opened_notional and fills summed over the four time buckets.
    """
    # read the full per-name table
    df = pd.read_parquet(path)
    # GRAIN GUARD: the file must be unique on (throttle, symbol, date, bucket).
    key = ["throttle", "symbol", "date", "bucket"]
    # a duplicated key here silently doubles every downstream sum
    if len(df) != len(df.drop_duplicates(key)):
        # abort loudly -- this is the stage0_audit lesson
        raise SystemExit(f"{path.name}: not unique on {key}")
    # the 'ALL' bucket, if present, would double-count the four real buckets
    df = df[df["bucket"] != "ALL"]
    # keep only this cohort's symbols
    df = df[df["symbol"].isin(keep)].copy()
    # unify the label for the 0.15 lean across files
    df["throttle"] = df["throttle"].replace(RELABEL)
    # tag the cohort so the reconciliation can be done per cohort
    df["cohort"] = label
    # collapse the bucket dimension: one row per symbol-day-setting
    out = (df.groupby(["cohort", "symbol", "date", "throttle"], as_index=False)
             [["net_pkr", "opened_notional", "fills"]].sum())
    # hand it back
    return out


def main():
    # ---- load all three cohorts -------------------------------------------
    print("=" * 78)
    print("CONFIG ASSIGNMENT -- per-name: lean 0.15 / lean 0.20 / lean off / not quoted")
    print("=" * 78)
    # the 38 incumbents from the corrected re-run
    a = load_cohort(PERNAME_38, set(EX.INCUMBENT), "38")
    # the new names from the 09-13 run, explicitly EXCLUDING the stale 38
    b = load_cohort(PERNAME_75, set(EX.ALL_NAMES) - set(EX.INCUMBENT), "75")
    # the lean-0.20 confirmation: every name it contains (110)
    c = load_cohort(PERNAME_20, set(EX.ALL_NAMES), "110")
    # one table
    df = pd.concat([a, b, c], ignore_index=True)
    # say what was loaded
    print(f"  {PERNAME_38.name}  -> {a.symbol.nunique()} names")
    print(f"  {PERNAME_75.name}  -> {b.symbol.nunique()} names")
    print(f"  {PERNAME_20.name}  -> {c.symbol.nunique()} names")

    # ---- HARD RECONCILIATION before a single decision is made --------------
    print("\nRECONCILIATION vs engine P&L")
    # walk every (cohort, setting) anchor
    for (coh, cfg), want in sorted(ANCHORS.items()):
        # the sum this load actually produces
        got = df[(df.cohort == coh) & (df.throttle == cfg)].net_pkr.sum()
        # the signed miss
        gap = got - want
        # report it to the cent
        print(f"  {coh:>3s}/{cfg:9s} {got:>16,.4f} vs {want:>16,.4f}   diff {gap:+.6f}")
        # a mismatch means the inputs are not what this script assumes
        if abs(gap) > RECON_TOL:
            raise SystemExit(f"RECONCILIATION FAILED for {coh}/{cfg}. "
                             f"Wrong input file or broken grain -- refusing to "
                             f"emit an assignment.")
    # every anchor held
    print("  all anchors hold.")

    # ---- per symbol-day, the settings side by side -------------------------
    # the cohort tag for each symbol comes from the OBI row (present for all 113)
    coh = df[df.throttle == "OBI"].groupby("symbol").cohort.first()
    # pivot so each row is one symbol-day with one column per setting
    wide = df.pivot_table(index=["symbol", "date"],
                          columns="throttle", values="net_pkr").reset_index()
    # PAIRING GUARD: the classifier needs BOTH lean-off and lean-0.15 on a day.
    # Lean-0.20 may be absent (the 3 cheap-tick names) and that is expected.
    before = len(wide)
    # keep only symbol-days paired on the two classifier columns
    wide = wide.dropna(subset=["OBI", "QT_2t@15"])
    # report any loss
    if len(wide) < before:
        print(f"  dropped {before - len(wide)} symbol-days unpaired on OBI/QT_2t@15")
    # the daily difference the classification rests on
    wide["diff"] = wide["QT_2t@15"] - wide["OBI"]

    # ---- per-name statistics ----------------------------------------------
    # a place to build rows
    recs = []
    # walk each name
    for sym, g in wide.groupby("symbol"):
        # number of paired days
        n = len(g)
        # total P&L under each setting (lean 0.20 is NaN for the cheap-tick names)
        tot_o = g["OBI"].sum()
        tot_15 = g["QT_2t@15"].sum()
        tot_20 = g["QT_2t@20"].sum() if g["QT_2t@20"].notna().any() else np.nan
        # the paired daily difference series
        d = g["diff"].to_numpy()
        # its sample sd (ddof=1)
        dsd = d.std(ddof=1)
        # EFFECT SIZE: mean / sd of the daily difference. NaN when the two
        # settings are byte-identical every day (the cheap-tick names).
        d_eff = d.mean() / dsd if dsd > 0 else np.nan
        # the t for reference only (d * sqrt(n)); never used in the decision
        t_diff = d_eff * np.sqrt(n) if not np.isnan(d_eff) else np.nan
        # traded notional under each setting, for bps reporting
        no = df[df.symbol == sym].groupby("throttle").opened_notional.sum()
        # one record per name
        recs.append({"symbol": sym, "cohort": coh.get(sym, "?"), "days": n,
                     "obi_pkr": tot_o, "lean15_pkr": tot_15, "lean20_pkr": tot_20,
                     "obi_bps": tot_o / no.get("OBI", np.nan) * 1e4,
                     "lean15_bps": tot_15 / no.get("QT_2t@15", np.nan) * 1e4,
                     "lean20_bps": tot_20 / no.get("QT_2t@20", np.nan) * 1e4,
                     "d_eff": d_eff, "t_diff": t_diff,
                     # the daily series of each setting, for the re-entry test
                     "_ser": {"OBI": g["OBI"].to_numpy(),
                              "QT_2t@15": g["QT_2t@15"].to_numpy(),
                              "QT_2t@20": g["QT_2t@20"].to_numpy()}})
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
        # a two-bucket-era file labels the 0.15 lean "QT_2t": map it forward
        pv = pv.replace(RELABEL)
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
    def own_t(r, cfg):
        """This setting's own daily t on this name: is its P&L distinguishable from zero?"""
        # the daily series
        s = r["_ser"][cfg]
        # drop NaN days (lean 0.20 on the cheap-tick names)
        s = s[~np.isnan(s)]
        # no data -> no evidence
        if len(s) < 2:
            return np.nan
        # sample sd
        sd = s.std(ddof=1)
        # the t-statistic
        return s.mean() / (sd / np.sqrt(len(s))) if sd > 0 else np.nan

    def decide(r):
        """Return (assigned_config, reason) for one name. ORDER MATTERS."""
        # 1. capacity is a risk limit and outranks every P&L consideration
        if r.capacity_flag != "ok":
            # cannot be unwound inside the cap -> never quote it
            return "DROP", f"capacity:{r.capacity_flag}"
        # the P&L of every setting that exists for this name
        pnl = {"OBI": r.obi_pkr, "QT_2t@15": r.lean15_pkr, "QT_2t@20": r.lean20_pkr}
        # 2. EVERY setting loses -> no setting fixes this name
        if max(v for v in pnl.values() if not np.isnan(v)) <= 0:
            # first-time drop
            return "DROP", "loses under every setting"
        # 3. CLASSIFY on the effect size d of (lean 0.15 - lean off)
        d = r.d_eff
        # the cheap-tick names: the lean is inert, the two are byte-identical
        if np.isnan(d):
            cfg, why = "OBI", "lean inert (cheap tick)"
        # the lean clearly WORKS here -> full lean at 0.15
        elif d > D_BAND:
            cfg, why = "QT_2t@15", f"d={d:+.3f} > {D_BAND:.3f}"
        # the lean clearly LOSES here -> lean off
        elif d < -D_BAND:
            cfg, why = "OBI", f"d={d:+.3f} < {-D_BAND:.3f}"
        # inside the band: HYSTERESIS first -- a name already in an outer
        # bucket stays there until d crosses the near edge (D_EXIT) back
        elif r.prior_config == "QT_2t@15" and d > D_EXIT:
            cfg, why = "QT_2t@15", f"d={d:+.3f} in band, held"
        elif r.prior_config == "OBI" and d < -D_EXIT:
            cfg, why = "OBI", f"d={d:+.3f} in band, held"
        # otherwise the middle bucket: the better of lean-off and lean-0.20.
        # Walk-forward: this beats blindly assigning 0.20 (+3.99/+8.75% vs
        # +3.36/+6.22% over always-0.15).
        else:
            # lean 0.20 exists for this name and beat lean off?
            if not np.isnan(r.lean20_pkr) and r.lean20_pkr > r.obi_pkr:
                cfg, why = "QT_2t@20", f"d={d:+.3f} in band; 0.20 beats off"
            else:
                cfg, why = "OBI", f"d={d:+.3f} in band; off beats 0.20"
        # the P&L of the setting we just chose -- NOT the best of the alternatives
        chosen = pnl[cfg]
        # 4. GATE THE CHOICE. Quoting a setting that lost money on this name
        #    over the sample is never correct, whatever the others did.
        if chosen <= 0:
            # say which setting was rejected, so the log explains itself
            return "DROP", f"{PLAIN[cfg]} loses ({chosen:,.0f})"
        # 5. RE-ENTRY: a name that was dropped last time needs its chosen
        #    setting to be significantly positive before it comes back
        if r.prior_config == "DROP" and not own_t(r, cfg) > T_REENTRY:
            return "DROP", f"held out: {PLAIN[cfg]} t={own_t(r, cfg):.2f} < {T_REENTRY}"
        # 6. MATERIALITY FLOOR
        if MIN_PKR_PER_DAY > 0 and (chosen / r.days) < MIN_PKR_PER_DAY:
            # immaterial, not unprofitable -- a distinct reason
            return "DROP", f"immaterial ({chosen / r.days:,.0f}/day)"
        # the surviving assignment
        return cfg, why

    # apply it row by row
    P[["assigned_config", "reason"]] = P.apply(lambda r: pd.Series(decide(r)), axis=1)
    # the plain-English name of the setting
    P["setting"] = P.assigned_config.map(PLAIN)
    # the engine parameters, so the harness never has to parse a label
    P["skew_ticks"] = P.assigned_config.map(lambda k: PARAMS[k][0])
    P["skew_thresh"] = P.assigned_config.map(lambda k: PARAMS[k][1])
    # did this refit move the name?
    P["changed"] = (P["prior_config"] != "") & (P["prior_config"] != P["assigned_config"])
    # the P&L the assignment is expected to realise (0 for a dropped name)
    P["assigned_pkr"] = P.apply(
        lambda r: {"OBI": r.obi_pkr, "QT_2t@15": r.lean15_pkr,
                   "QT_2t@20": r.lean20_pkr, "DROP": 0.0}[r.assigned_config], axis=1)
    # when this assignment should be rebuilt
    P["as_of"] = datetime.now().strftime("%Y-%m-%d")
    P["refit_due"] = (datetime.now() + timedelta(days=REFIT_DAYS)).strftime("%Y-%m-%d")

    # ---- report ------------------------------------------------------------
    # headline counts
    counts = P.assigned_config.value_counts()
    print("\n" + "=" * 78)
    for cfg in SETTINGS:
        # how many names and what they are worth
        sel = P[P.assigned_config == cfg]
        print(f"  {PLAIN[cfg]:20s} {len(sel):>4} names   assigned P&L {sel.assigned_pkr.sum():>14,.0f} PKR")
    # the baseline this must beat
    base = P.lean15_pkr.sum()
    # what the assignment produces
    tot = P.assigned_pkr.sum()
    print("=" * 78)
    print(f"  everyone on lean 0.15    {base:>14,.0f} PKR")
    print(f"  assigned                 {tot:>14,.0f} PKR   "
          f"({tot - base:+,.0f}, {(tot/base - 1)*100:+.1f}%)")
    print("  NOTE: that gap is IN-SAMPLE. Rolling walk-forward, 110 names, 10 folds")
    print("        (60-day fit, 20-day test, step 12), gain over always-0.15:")
    print("            two buckets (off/0.15 + drop) : +5.45%  se 0.94  t +5.83  10/10 folds")
    print("            three buckets (this rule)     : +6.41%  se 1.22  t +5.25  10/10 folds")
    print("            three MINUS two               : +0.96pp se 0.58  t +1.64   8/10 folds")
    print("        Read that honestly: the DROP gate carries nearly all the value and is")
    print("        solid. The THIRD BUCKET is worth about +1pp, positive in 8 of 10 windows,")
    print("        but does NOT clear the |t| > 2 bar used elsewhere in this project.")
    print("        Budget the walk-forward number, not the in-sample one.")
    # the names that are not on the default
    print("\nNON-DEFAULT ASSIGNMENTS")
    nd = P[P.assigned_config != "QT_2t@15"].sort_values("d_eff")
    print(nd[["cohort", "setting", "obi_pkr", "lean15_pkr", "lean20_pkr", "d_eff",
              "capacity_flag", "reason"]].to_string(float_format=lambda v: f"{v:,.2f}"))
    # churn, once a prior exists
    if P.changed.any():
        print(f"\nCHANGED SINCE LAST REFIT: {int(P.changed.sum())}")
        print(P[P.changed][["prior_config", "assigned_config", "d_eff"]].to_string())

    # ---- write -------------------------------------------------------------
    # a fresh timestamped destination; never overwrites
    out = EX.safe_out("config_assignment", "csv")
    # the columns the harness needs plus the evidence behind each decision
    P.reset_index()[["symbol", "cohort", "assigned_config", "setting",
                     "skew_ticks", "skew_thresh", "reason",
                     "d_eff", "t_diff", "obi_pkr", "lean15_pkr", "lean20_pkr",
                     "obi_bps", "lean15_bps", "lean20_bps", "capacity_flag", "days",
                     "prior_config", "changed", "as_of", "refit_due"]
                    ].to_csv(out, index=False)
    # say where it went
    print(f"\nwrote {out}")

    # ---- chart -------------------------------------------------------------
    # one figure: the decision map in PKR space (lean off vs lean 0.15)
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
    # the identity line: above it the lean wins, below it lean-off wins
    lo = min(P.obi_pkr.min(), P.lean15_pkr.min()) * 1.1
    hi = max(P.obi_pkr.max(), P.lean15_pkr.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1.5, ls="--", zorder=2)
    # zero axes
    ax.axhline(0, color="#e3e2df", lw=1.2)
    ax.axvline(0, color="#e3e2df", lw=1.2)
    # one series per decision, so colour encodes the assignment (validated palette)
    for cfg, col, mk in (("QT_2t@15", "#3b6bd6", "o"),
                         ("QT_2t@20", "#1f9e78", "D"),
                         ("OBI", "#c4422e", "s"),
                         ("DROP", "#8a8781", "x")):
        # this decision's names
        s = P[P.assigned_config == cfg]
        # plot them
        ax.scatter(s.obi_pkr, s.lean15_pkr, s=58, c=col, marker=mk,
                   edgecolor="#fcfcfb" if mk != "x" else None, linewidth=1.4,
                   zorder=3, label=f"{PLAIN[cfg]} ({len(s)})")
    # label only the names that are not on the default
    for sym, r in P[P.assigned_config != "QT_2t@15"].iterrows():
        # a small direct label beside the mark
        ax.annotate(sym, (r.obi_pkr, r.lean15_pkr), fontsize=7, color="#0b0b0b",
                    xytext=(6, 3), textcoords="offset points")
    # square the axes so the identity line is a true 45 degrees
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # thousands, not raw PKR
    fmt = matplotlib.ticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}k")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    # axis titles
    ax.set_xlabel("lean off — net PKR", color="#52514e", fontsize=10)
    ax.set_ylabel("lean, trigger 0.15 — net PKR", color="#52514e", fontsize=10)
    # what the reader is looking at
    ax.set_title(f"Config assignment, {len(P)} names\n"
                 + " · ".join(f"{counts.get(c, 0)} {PLAIN[c]}" for c in SETTINGS),
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
