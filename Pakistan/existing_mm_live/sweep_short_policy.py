# ============================================================================
# sweep_short_policy.py -- what does PSX Chapter 10 cost us?
# ============================================================================
# THE QUESTION. PSX Regulations 10.15 prohibits a Blank Sale -- selling what
# you do not own, without Pre-Existing Interest and without an SLB borrow --
# for a Securities Broker's own account, except a Designated Market Maker in
# its Assigned Security. With a TREC we ARE the Securities Broker, so this
# binds our proprietary book directly.
#
# The strategy quotes two-sided from flat. It sells what it does not own. Every
# measured result was produced under that behaviour, which may not be
# permissible. This run prices the alternatives.
#
# THE ARMS
#   unrestricted   go short freely. The baseline, and what was measured.
#   no_short       never go net short: the ask is capped at the shares held and
#                  is not quoted at all when flat.
#   long_buffer    same quoting rule, but the day opens holding a buffer, BOUGHT
#                  at the first print. Its intraday price move is a real cost
#                  and is carried by the run.
#   slb_uptick     SLB-eligible names may short, but only on an Uptick or
#                  Zero-Plus Tick (10.16.1(a)). Ineligible names fall back to
#                  no_short. WITHOUT AN ELIGIBILITY LIST THIS ARM IS A DUPLICATE
#                  OF no_short, and the run says so rather than pretending.
#   cfo            unrestricted quoting, with a reprice sent as ONE Change
#                  Former Order (8.5.1(d)) instead of a cancel plus a new order.
#                  Orthogonal to the short-sale question; included because it is
#                  the other mechanism change awaiting a number.
#
# METHOD. Day-as-unit paired differences against the baseline, per this
# project's standing rule: never pool fills for significance. Each arm is run
# on the same symbol-days with the same latency seed, so the pairing is exact.
#
# READ-ONLY on every input. Writes ONE timestamped CSV and ONE PNG. Never
# overwrites, never deletes.
#
# Run from existing_mm_live/:
#   caffeinate -is python sweep_short_policy.py
#   caffeinate -is python sweep_short_policy.py --smoke      (2 names, 5 days)
# ============================================================================

# command-line flags
import argparse
# wall-clock timing for the heartbeat
import time
# numeric
import numpy as np
# frames
import pandas as pd
# plotting, headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the shared harness: calibration loaders, the single backtest driver
import mm_harness as H
# the driver module: dataset opening, date discovery, the canonical CFG
import run_legacy_mm as R
# shared name lists and the never-overwrite output helper
import expansion_names as EX

# ---------------------------------------------------------------------------
# WHAT TO RUN
# ---------------------------------------------------------------------------
# how many names in the sample. A policy difference that needs 113 names to
# show up is not a policy difference worth restructuring the book for.
N_NAMES = 12
# how many of the most recent dates to use
N_DAYS = 40
# the buffer, in clips, for the long_buffer arm. 10 clips equals max_inv at the
# production ratio, so the book opens at its inventory ceiling -- the most
# generous version of the policy, and the most capital-hungry.
BUFFER_CLIPS = 10.0
# the quote clip. Matches the production sweeps.
CLIP = 50

# Category A SLB-eligible names, per NCCPL. NOT in the PSX rulebook and not
# derivable -- it is an NCCPL publication that changes over time, so it has to
# be supplied. Put one symbol per line in this file, or leave it absent.
SLB_LIST = "slb_category_a.txt"

# the arms: label -> (strategy overrides, engine cfg overrides)
# Written as a plain table so adding an arm is one line, not a code change.
ARMS = {
    # the baseline: what every shipped number was produced under
    "unrestricted": ({"short_policy": "unrestricted"}, {}),
    # never sell what we do not hold
    "no_short": ({"short_policy": "no_short"}, {}),
    # never sell what we do not hold, but open the day holding a buffer
    "long_buffer": ({"short_policy": "long_buffer"},
                    {"opening_inventory": BUFFER_CLIPS * CLIP}),
    # short only on an uptick, and only in SLB-eligible names
    "slb_uptick": ({"short_policy": "slb_uptick"}, {}),
    # the other mechanism change: one amendment instead of two messages
    "cfo": ({"short_policy": "unrestricted"}, {"use_cfo": True}),
}
# the arm every other arm is compared against
BASELINE = "unrestricted"


def load_slb_eligible():
    """Category A SLB-eligible symbols, or None when the list is absent."""
    # the file lives beside this script
    from pathlib import Path
    # resolve relative to this file so the working directory does not matter
    path = Path(__file__).resolve().parent / SLB_LIST
    # absent is a normal state, and the run reports it rather than guessing
    if not path.exists():
        return None
    # one symbol per line, blanks and comments ignored
    return {ln.strip().upper() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")}


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # a fast shape-check before committing to the full sample
    ap.add_argument("--smoke", action="store_true",
                    help="2 names, 5 days -- checks the plumbing, not the answer")
    args = ap.parse_args()
    # smoke mode shrinks the sample to something that runs in minutes
    n_names = 2 if args.smoke else N_NAMES
    n_days = 5 if args.smoke else N_DAYS

    print("=" * 78)
    print("SHORT-SALE POLICY SWEEP -- what PSX Chapter 10 costs")
    print("=" * 78)

    # ---- the universe and the calendar --------------------------------
    # every trading date in the parsed store, oldest first
    all_dates = R.discover_dates()
    # the most recent n_days of them
    dates = all_dates[-n_days:]
    # the first n_names of the production universe, so the sample is stable
    # between runs rather than a different draw each time
    names = sorted(EX.ALL_NAMES)[:n_names]
    # say exactly what is being run
    print(f"  {len(names)} names x {len(dates)} dates x {len(ARMS)} arms")
    print(f"  names: {', '.join(names)}")
    print(f"  dates: {dates[0]} .. {dates[-1]}")

    # ---- the SLB eligibility list -------------------------------------
    # Category A, or None when the file is absent
    slb = load_slb_eligible()
    # BE LOUD ABOUT THIS. Without the list the slb_uptick arm cannot short
    # anything and is arithmetically identical to no_short -- which is a
    # correct fallback, but reading it as a measurement of the uptick rule
    # would be wrong.
    if slb is None:
        print(f"\n  !! {SLB_LIST} not found. Every name is treated as NOT")
        print(f"     SLB-eligible, so the slb_uptick arm falls back to")
        print(f"     no_short and will DUPLICATE it. That is the correct")
        print(f"     behaviour, not a measurement of the uptick rule.")
        slb = set()
    else:
        # how many of our names are actually eligible
        n_elig = len([s for s in names if s in slb])
        print(f"\n  SLB Category A: {n_elig} of {len(names)} sample names eligible")

    # ---- calibration, loaded once -------------------------------------
    # per-symbol inventory-skew scale
    scales = H.load_scales()
    # per-symbol 4-bucket volume profile
    profiles = H.load_profiles()
    # per-symbol EOD ramp/cliff minutes
    windows = H.load_windows()
    # per-date continuous-session segments
    segments = H.load_segments()

    # ---- the run -------------------------------------------------------
    # one row per (arm, symbol, date)
    rows = []
    # heartbeat timer
    t_start = time.perf_counter()
    # walk the calendar once, opening each date's datasets a single time
    for di, date in enumerate(dates, 1):
        # the date's parquet partitions
        dsets = R.open_datasets(date)
        # a missing partition is a skipped date, not a failure
        if dsets is None:
            print(f"  [{di}/{len(dates)}] {date} no datasets; skip")
            continue
        # the day's continuous-session segments
        segs = segments.get(str(date))
        # no segments means no calibrated session for this date
        if segs is None:
            print(f"  [{di}/{len(dates)}] {date} no session segments; skip")
            continue
        # each symbol in the sample
        for sym in names:
            # a name missing any calibration input cannot be run
            if sym not in scales or sym not in profiles or sym not in windows:
                continue
            # every arm, on the SAME symbol-day, so the pairing is exact
            for arm, (strat_over, cfg_over) in ARMS.items():
                # the strategy parameters, with this arm's overrides applied
                over = dict(strat_over)
                # per-symbol SLB eligibility, which only slb_uptick reads
                over["slb_eligible"] = sym in slb
                # the assembled MicrostructureMM kwargs
                params = H.build_micro_params(
                    CLIP, scales[sym], profiles[sym], windows[sym], segs,
                    overrides=over)
                # isolate a failure to one arm-symbol-day rather than the run
                try:
                    # the one backtest path every runner shares
                    dr = H.run_symbol_day(date, sym, dsets, params,
                                          cfg_overrides=cfg_over)
                except Exception as exc:                      # noqa: BLE001
                    # report and carry on
                    print(f"    {arm} {sym} {date} ERROR {exc!r}")
                    continue
                # an unrunnable symbol-day returns None
                if dr is None:
                    continue
                # the day's post-liquidation P&L
                pnl = dr.pnl()
                # a broken close has no headline number
                if pnl is None or not np.isfinite(pnl):
                    continue
                # the execution panel, for the message-rate comparison
                ex = H.execution_stats(dr)
                # the inventory panel, which is the whole point of this sweep
                inv = H.inventory_stats(dr)
                # one row
                rows.append({
                    "arm": arm, "symbol": sym, "date": str(date),
                    # the headline
                    "net_pkr": float(pnl),
                    # how much was traded, for a bps view
                    "fills": ex["n_fills"],
                    # messages sent: the CFO arm should roughly halve this
                    "orders_sent": ex["n_orders_sent"],
                    # order-to-trade, the churn headline
                    "otr": ex["otr"],
                    # the deepest position touched, signed information lost
                    "max_abs_inv": inv["max_abs_inv"],
                    # the end-of-day position before liquidation
                    "eod_inv": inv["eod_inv"],
                    # HOW OFTEN THE UPTICK RULE ACTUALLY BIT. If this is zero
                    # on the slb_uptick arm, the constraint never engaged and
                    # any difference is noise, not the rule.
                    "uptick_blocks": int(
                        dr.stats.get("short_fills_blocked_by_uptick", 0)),
                    # amendments that landed, for the CFO arm
                    "cfos": int(dr.stats.get("n_cfos", 0)),
                })
        # heartbeat with elapsed time
        print(f"  [{di}/{len(dates)}] {date}  "
              f"{H._fmt(time.perf_counter() - t_start)}", flush=True)

    # nothing ran: say so rather than emitting an empty file
    if not rows:
        raise SystemExit("no symbol-days ran; check the store and the calendar")
    # the long-form results
    df = pd.DataFrame(rows)

    # ---- day-as-unit paired statistics ---------------------------------
    # one column per arm, indexed by (symbol, date), so each row is one
    # symbol-day seen under every arm
    wide = df.pivot_table(index=["symbol", "date"], columns="arm",
                          values="net_pkr")
    # PAIRING GUARD: only symbol-days where every arm produced a number can be
    # compared. Dropping unpaired days is the difference between a paired test
    # and a misleading one.
    before = len(wide)
    wide = wide.dropna()
    # report any loss
    if len(wide) < before:
        print(f"\n  dropped {before - len(wide)} symbol-days not present in "
              f"every arm")

    print("\n" + "=" * 78)
    print(f"RESULTS -- {len(wide)} paired symbol-days")
    print("=" * 78)
    # the baseline total, which every arm is measured against
    base_total = wide[BASELINE].sum()
    # the per-arm summary
    summary = []
    # every arm, baseline first
    for arm in ARMS:
        # this arm's total P&L over the sample
        total = wide[arm].sum()
        # the paired per-day difference against the baseline
        d = (wide[arm] - wide[BASELINE]).to_numpy()
        # its mean, standard error and t, computed day-as-unit
        mean_d = d.mean()
        # sample standard deviation
        sd = d.std(ddof=1) if len(d) > 1 else np.nan
        # standard error of the mean difference
        se = sd / np.sqrt(len(d)) if len(d) > 1 else np.nan
        # the t-statistic, the project's usual |t| > 2 bar applies
        t = mean_d / se if (se and se > 0) else np.nan
        # how many days this arm beat the baseline
        wins = int((d > 0).sum())
        # one summary row
        summary.append({
            "arm": arm, "total_pkr": total,
            "vs_base_pkr": total - base_total,
            "vs_base_pct": ((total / base_total - 1) * 100
                            if base_total else np.nan),
            "mean_daily_diff": mean_d, "se": se, "t": t,
            "days_better": wins, "days": len(d),
            # the mechanism counters, summed over the sample
            "uptick_blocks": int(df[df.arm == arm]["uptick_blocks"].sum()),
            "orders_sent": int(df[df.arm == arm]["orders_sent"].sum()),
            "cfos": int(df[df.arm == arm]["cfos"].sum()),
        })
    # as a frame for printing and writing
    S = pd.DataFrame(summary)
    # print it readably
    print(S.to_string(index=False,
                      float_format=lambda v: f"{v:,.2f}"))

    print("\nREAD THIS BEFORE THE NUMBERS:")
    print("  * the |t| > 2 bar this project applies elsewhere applies here.")
    print("  * uptick_blocks == 0 on slb_uptick means the rule never engaged,")
    print("    so any difference on that arm is noise and not the constraint.")
    print("  * long_buffer carries the buffer's intraday price move as a real")
    print("    cost -- the buffer is bought at the first print, not conjured.")
    print("  * a sample this size measures whether the effect is LARGE. It")
    print("    cannot rule out a small one.")

    # ---- write ---------------------------------------------------------
    # a fresh timestamped destination; never overwrites
    out = EX.safe_out("short_policy_sweep", "csv")
    # the long-form rows, which carry everything the summary condenses
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    # the summary beside it
    out_s = EX.safe_out("short_policy_summary", "csv")
    S.to_csv(out_s, index=False)
    print(f"wrote {out_s}")

    # ---- chart ----------------------------------------------------------
    # two panels: the totals, and the paired daily difference distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                   facecolor="#fcfcfb")
    # both panels share the recessive styling used across this project
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        # drop the top/right frame
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # mute the rest
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#e3e2df")
        # a grid behind the marks
        ax.grid(color="#e3e2df", lw=0.7)
        ax.set_axisbelow(True)
    # PANEL 1: total P&L per arm, baseline highlighted
    arms = list(ARMS)
    # one bar per arm
    totals = [wide[a].sum() for a in arms]
    # the baseline is the reference and is coloured differently
    colors = ["#52514e" if a == BASELINE else "#3b6bd6" for a in arms]
    ax1.bar(arms, totals, color=colors)
    # a line at the baseline total, so the comparison is visual
    ax1.axhline(base_total, color="#c4422e", lw=1.2, ls="--")
    # zero line
    ax1.axhline(0, color="#e3e2df", lw=1.2)
    # thousands
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}k"))
    # what the reader is looking at
    ax1.set_ylabel("total net PKR over the sample", color="#52514e", fontsize=10)
    ax1.set_title("Total by policy\n(dashed = baseline)", color="#0b0b0b",
                  fontsize=11.5, loc="left", pad=8)
    # rotate the labels so they do not collide
    ax1.tick_params(axis="x", rotation=20)
    # PANEL 2: the paired daily differences, which is what the t is computed on
    # every arm except the baseline, whose difference is identically zero
    diffs = [(wide[a] - wide[BASELINE]).to_numpy() for a in arms
             if a != BASELINE]
    # their labels
    labels = [a for a in arms if a != BASELINE]
    # a box per arm shows the spread, not just the mean.
    # matplotlib renamed `labels` to `tick_labels` in 3.9 and the old name now
    # raises rather than warning, so try the new spelling first and fall back.
    try:
        # matplotlib >= 3.9
        bp = ax2.boxplot(diffs, tick_labels=labels, showfliers=False,
                         patch_artist=True)
    except TypeError:
        # matplotlib < 3.9
        bp = ax2.boxplot(diffs, labels=labels, showfliers=False,
                         patch_artist=True)
    # colour the boxes consistently with panel 1
    for patch in bp["boxes"]:
        patch.set_facecolor("#3b6bd6")
        patch.set_alpha(0.35)
    # zero is the line that matters: above it the arm beat the baseline
    ax2.axhline(0, color="#c4422e", lw=1.2)
    # thousands
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}k"))
    # what the reader is looking at
    ax2.set_ylabel("daily difference vs baseline, PKR", color="#52514e",
                   fontsize=10)
    ax2.set_title("Paired daily differences\n(day-as-unit; the t is computed "
                  "on these)", color="#0b0b0b", fontsize=11.5, loc="left",
                  pad=8)
    # rotate the labels
    ax2.tick_params(axis="x", rotation=20)
    # the run's scope, stated on the figure so a stray PNG is still readable
    fig.suptitle(f"Short-sale policy sweep -- {len(names)} names x "
                 f"{len(dates)} dates, {len(wide)} paired symbol-days",
                 color="#0b0b0b", fontsize=13, x=0.01, ha="left")
    # tidy margins, leaving room for the suptitle
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # a fresh timestamped PNG; never overwrites
    png = EX.safe_out("short_policy_sweep", "png")
    # write it
    fig.savefig(png, dpi=150, facecolor="#fcfcfb")
    print(f"wrote {png}")


# entry point
if __name__ == "__main__":
    main()
