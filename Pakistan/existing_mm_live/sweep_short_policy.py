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
#                  at the first print and sold back into the closing book.
#
#                  READ THIS ONE CAREFULLY. Because every day is an independent
#                  run, the buffer is bought and sold EVERY DAY. That has two
#                  consequences and they point in opposite directions:
#                    (a) the stock's move between the first print and the close
#                        lands in the P&L. At BUFFER_CLIPS=2 that is 100 shares,
#                        so a 1% intraday move on a Rs 300 name is 300 PKR
#                        against a baseline day of roughly 148 PKR -- the stock
#                        is the larger term, and it is not the thing we are
#                        trying to measure;
#                    (b) the overnight gap, which is where a standing buffer's
#                        real risk lives, is never seen at all.
#                  So the model counts the wrong exposure and misses the right
#                  one. This sweep therefore reports TWO columns: net_pkr (what
#                  the account would have done, buffer bet included) and
#                  net_pkr_ex_buffer (the same run with the buffer's price move
#                  subtracted out, so the arms are compared on QUOTING). The
#                  policy decision should be read off the second; the first is
#                  there so nothing is hidden.
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
# the per-side fee actually in force (TREC or retail), so the buffer's
# carry-cost arithmetic below is read from the engine and not re-typed here
from mm_backtest import FEE_TOTAL_PCT

# ---------------------------------------------------------------------------
# WHAT TO RUN
# ---------------------------------------------------------------------------
# how many names in the sample. A policy difference that needs 113 names to
# show up is not a policy difference worth restructuring the book for.
N_NAMES = 12
# how many of the most recent dates to use
N_DAYS = 40
# the buffer, in clips, for the long_buffer arm.
#
# CORRECTED 2026-09-16 after the first smoke run. This was 10.0, chosen as "the
# most generous version of the policy". It is the BROKEN version.
# build_micro_params sets max_inv = 10 clips and soft_inv = 3 clips, so a
# 10-clip buffer opens the day EXACTLY AT THE INVENTORY CEILING: micro_mm needs
# pos < eff_max AND pos < soft_inv to quote a bid, and 500 < 500 is False. The
# arm opened one-sided and spent the morning liquidating, so its P&L was the
# price path of that liquidation, not market making. max_abs_inv was 500.0 on
# every row of the first run, which is the fingerprint.
#
# A WORKABLE BUFFER SITS BELOW soft_inv so both sides still quote, and at or
# above one clip so a full ask can be shown. That window is [1, 3) clips.
BUFFER_CLIPS = 2.0
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


def paired_table(df, value_col):
    """Day-as-unit paired differences against the baseline, on one P&L column.

    Returns (wide, summary_frame). `wide` is one column per arm indexed by
    (symbol, date), so every row is one symbol-day seen under every arm --
    which is what makes the difference paired and the t meaningful.
    """
    # pivot the long-form rows into the paired shape
    wide = df.pivot_table(index=["symbol", "date"], columns="arm",
                          values=value_col)
    # PAIRING GUARD: only symbol-days present in every arm can be compared.
    # Dropping unpaired days is the difference between a paired test and a
    # misleading one.
    before = len(wide)
    wide = wide.dropna()
    # report any loss, once per column
    if len(wide) < before:
        print(f"  [{value_col}] dropped {before - len(wide)} symbol-days not "
              f"present in every arm")
    # the baseline total, which every arm is measured against
    base_total = wide[BASELINE].sum()
    # one row per arm
    summary = []
    # every arm, baseline first
    for arm in ARMS:
        # this arm's total over the sample
        total = wide[arm].sum()
        # the paired per-day difference against the baseline
        d = (wide[arm] - wide[BASELINE]).to_numpy()
        # its mean, computed day-as-unit
        mean_d = d.mean()
        # sample standard deviation
        sd = d.std(ddof=1) if len(d) > 1 else np.nan
        # standard error of the mean difference
        se = sd / np.sqrt(len(d)) if len(d) > 1 else np.nan
        # the t-statistic; the project's usual |t| > 2 bar applies
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
            # the wire-traffic comparison that actually means something
            "total_msgs": int(df[df.arm == arm]["total_msgs"].sum()),
            "orders_sent": int(df[df.arm == arm]["orders_sent"].sum()),
            "cancels": int(df[df.arm == arm]["cancels"].sum()),
            "cfos": int(df[df.arm == arm]["cfos"].sum()),
        })
    # the pair the caller needs
    return wide, pd.DataFrame(summary)


def concentration(wide, title):
    """Say whether an arm's difference is one day or the whole sample.

    A total hides concentration. An arm whose entire result is a single
    symbol-day has not been measured, it has been sampled.
    """
    # a heading so two calls are distinguishable
    print(f"\nCONCENTRATION -- {title}")
    # every arm except the baseline, whose difference is identically zero
    for arm in ARMS:
        # the baseline has nothing to concentrate
        if arm == BASELINE:
            continue
        # the paired per-day differences, keyed by symbol-day
        d = wide[arm] - wide[BASELINE]
        # the total
        tot = d.sum()
        # nothing to report on a zero total
        if abs(tot) < 1e-9:
            print(f"  {arm:14s} total is zero")
            continue
        # the single largest contributor, by absolute size
        idx = d.abs().idxmax()
        # its value
        big = d.loc[idx]
        # what the total becomes without it
        without = tot - big
        # one line per arm
        print(f"  {arm:14s} biggest day {idx[0]} {idx[1]}: {big:+,.0f} PKR "
              f"({abs(big / tot) * 100:.0f}% of the total) "
              f"-- without it: {without:+,.0f}")


def pkr_axis(ax):
    """Label an axis in PKR at a scale that is actually readable.

    CORRECTED 2026-09-16. Every axis was hardcoded to `v/1000` with no
    decimals, so a smoke run whose numbers live between 0 and 1,500 PKR
    produced an axis reading 0k, 0k, 1k, 1k, 1k -- five ticks, two distinct
    labels, no information. The scale now follows the data.
    """
    # the largest magnitude the axis has to show
    top = max(abs(v) for v in ax.get_ylim())
    # above 10,000 PKR thousands are the readable unit; below it they are not
    if top >= 10_000:
        # thousands, one decimal so adjacent ticks differ
        fmt = lambda v, p: f"{v / 1000:,.1f}k"
    else:
        # plain PKR with thousands separators
        fmt = lambda v, p: f"{v:,.0f}"
    # apply it
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))


def make_chart(wide_raw, wide_ex, n_names, n_dates, out_png):
    """The three-panel figure. Split out of main() so it can be exercised
    on synthetic frames without a parsed store behind it -- the last two
    runs each lost their chart to a matplotlib API detail after the whole
    sweep had finished computing, which is an expensive way to find out."""
    # ---- chart ----------------------------------------------------------
    # three panels: the totals on both bases, then the paired daily difference
    # distribution on each basis. The two boxes side by side are the point of
    # the figure -- they show how much of the long_buffer box is the stock.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6),
                                        facecolor="#fcfcfb")
    # every panel shares the recessive styling used across this project
    for ax in (ax1, ax2, ax3):
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
    # PANEL 1: total P&L per arm, on both bases, side by side
    arms = list(ARMS)
    # bar positions
    x = np.arange(len(arms))
    # the as-run totals
    tot_raw = [wide_raw[a].sum() for a in arms]
    # the totals with the buffer's price move removed
    tot_ex = [wide_ex[a].sum() for a in arms]
    # grouped bars: grey = as run, blue = ex-buffer
    ax1.bar(x - 0.2, tot_raw, width=0.4, color="#9a9894", label="as run")
    ax1.bar(x + 0.2, tot_ex, width=0.4, color="#3b6bd6",
            label="ex buffer price move")
    # the baseline's ex-buffer total, as the reference line
    ax1.axhline(wide_ex[BASELINE].sum(), color="#c4422e", lw=1.2, ls="--")
    # zero line
    ax1.axhline(0, color="#e3e2df", lw=1.2)
    # arm names under the groups
    ax1.set_xticks(x)
    ax1.set_xticklabels(arms)
    # PKR at a scale the numbers actually need
    pkr_axis(ax1)
    # what the reader is looking at
    ax1.set_ylabel("total net PKR over the sample", color="#52514e", fontsize=10)
    ax1.set_title("Total by policy\n(dashed = baseline, ex-buffer)",
                  color="#0b0b0b", fontsize=11.5, loc="left", pad=8)
    # rotate the labels so they do not collide
    ax1.tick_params(axis="x", rotation=20)
    # name the two bases
    ax1.legend(frameon=False, fontsize=9)

    # the arms the differences are computed for (the baseline's is always zero)
    labels = [a for a in arms if a != BASELINE]

    def draw_box(ax, wide_df, title):
        """One boxplot of paired daily differences on a given P&L basis."""
        # the per-day differences, one array per arm
        diffs = [(wide_df[a] - wide_df[BASELINE]).to_numpy() for a in labels]
        # a box per arm shows the spread, not just the mean. matplotlib
        # renamed `labels` to `tick_labels` in 3.9 and the old name now
        # raises rather than warning, so try the new spelling and fall back.
        try:
            # matplotlib >= 3.9
            bp = ax.boxplot(diffs, tick_labels=labels, showfliers=False,
                            patch_artist=True)
        except TypeError:
            # matplotlib < 3.9
            bp = ax.boxplot(diffs, labels=labels, showfliers=False,
                            patch_artist=True)
        # colour the boxes consistently with panel 1
        for patch in bp["boxes"]:
            patch.set_facecolor("#3b6bd6")
            patch.set_alpha(0.35)
        # zero is the line that matters: above it the arm beat the baseline
        ax.axhline(0, color="#c4422e", lw=1.2)
        # PKR at a scale the numbers actually need
        pkr_axis(ax)
        # what the reader is looking at
        ax.set_ylabel("daily difference vs baseline, PKR", color="#52514e",
                      fontsize=10)
        # the panel's heading
        ax.set_title(title, color="#0b0b0b", fontsize=11.5, loc="left", pad=8)
        # rotate the labels
        ax.tick_params(axis="x", rotation=20)

    # PANEL 2: as run -- the buffer's price move still in
    draw_box(ax2, wide_raw,
             "Paired daily differences, AS RUN\n(includes the buffer's price "
             "move)")
    # PANEL 3: the policy comparison -- price move removed
    draw_box(ax3, wide_ex,
             "Paired daily differences, EX BUFFER\n(the t that should decide "
             "the policy)")
    # SHARED Y RANGE, so the two boxes are comparable by eye. Without it
    # matplotlib scales each panel to its own data and the panel with the
    # smaller spread looks identical to the one with the larger.
    lo = min(ax2.get_ylim()[0], ax3.get_ylim()[0])
    hi = max(ax2.get_ylim()[1], ax3.get_ylim()[1])
    ax2.set_ylim(lo, hi)
    ax3.set_ylim(lo, hi)
    # RE-APPLY THE FORMATTER. pkr_axis picks thousands-or-units from the axis
    # range, and the range just changed, so the choice made inside draw_box
    # can now be the wrong one.
    pkr_axis(ax2)
    pkr_axis(ax3)
    # the run's scope, stated on the figure so a stray PNG is still readable
    fig.suptitle(f"Short-sale policy sweep -- {n_names} names x "
                 f"{n_dates} dates, {len(wide_ex)} paired symbol-days",
                 color="#0b0b0b", fontsize=13, x=0.01, ha="left")
    # tidy margins, leaving room for the suptitle
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # write it to the caller's destination
    fig.savefig(out_png, dpi=150, facecolor="#fcfcfb")
    # release the figure; a long run would otherwise accumulate them
    plt.close(fig)
    # say where it went
    print(f"wrote {out_png}")




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

    # THE BUFFER MUST NOT PIN THE BOOK. build_micro_params sets soft_inv at 3
    # clips, and micro_mm stops quoting the bid once the position reaches it.
    # A buffer at or above that opens one-sided, which measures a liquidation
    # rather than a policy -- and it does so silently, which is worse. Refuse.
    if BUFFER_CLIPS >= 3.0:
        raise SystemExit(
            f"BUFFER_CLIPS={BUFFER_CLIPS} is at or above the 3-clip soft_inv "
            f"that build_micro_params sets. The book would open with the bid "
            f"switched off and the arm would measure a morning liquidation, "
            f"not the buffer policy. Use a value in [1, 3).")

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
                # ---- the buffer's directional term, stripped out ----------
                # mm_backtest records what the buffer was bought at and marks
                # it at the closing MID, so this is the stock's move and
                # nothing else. Execution cost and the entry fee stay in the
                # P&L, because those are real costs of carrying a buffer.
                # the eod report, which carries the buffer fields
                _eod = dr.eod or {}
                # the price-move term (0.0 on every arm that holds no buffer)
                buf_move = _eod.get("buffer_price_move", 0.0)
                # the same day's P&L with that term removed
                pnl_ex = _eod.get("equity_ex_buffer_move")
                # OLD ENGINE GUARD: if mm_backtest predates the buffer fields
                # the key is absent. Fall back to the raw number rather than
                # silently reporting a column that does not mean what it says.
                if pnl_ex is None or not np.isfinite(pnl_ex):
                    # no isolation available on this row
                    pnl_ex = pnl
                    # and the move term is unknown, not zero
                    buf_move = np.nan
                # the execution panel, for the message-rate comparison
                ex = H.execution_stats(dr)
                # the inventory panel, which is the whole point of this sweep
                inv = H.inventory_stats(dr)
                # one row
                rows.append({
                    "arm": arm, "symbol": sym, "date": str(date),
                    # WHAT THE ACCOUNT WOULD HAVE DONE, buffer bet included
                    "net_pkr": float(pnl),
                    # THE POLICY COMPARISON COLUMN: the same day with the
                    # buffer's price move taken out, so what is left is the
                    # quoting difference the arms actually differ in
                    "net_pkr_ex_buffer": float(pnl_ex),
                    # the term that was removed, reported so it is visible
                    "buffer_price_move": float(buf_move),
                    # the buffer size in shares (0.0 on the other arms)
                    "buffer_qty": float(_eod.get("buffer_qty") or 0.0),
                    # what it was bought at; NaN when there was no buffer
                    "buffer_px": float(_eod.get("buffer_px") or np.nan),
                    # the touch at that instant, so the acquisition is auditable
                    "buffer_bid": float(_eod.get("buffer_bid") or np.nan),
                    "buffer_ask": float(_eod.get("buffer_ask") or np.nan),
                    # WHAT THE FREE ACQUISITION IS WORTH: the extra PKR a real
                    # market buy would have paid, price and fee together. NaN
                    # when the book was one-sided and there was no ask.
                    "buffer_acq_understated_pkr": float(
                        _eod.get("buffer_acq_understated_pkr") or np.nan),
                    # the closing mid it was marked back at
                    "mid_at_close": float(_eod.get("mid_at_close") or np.nan),
                    # how much was traded, for a bps view
                    "fills": ex["n_fills"],
                    # new orders (and, on the CFO arm, amendments). NOT the
                    # message total -- n_orders_sent never counted cancels.
                    "orders_sent": ex["n_orders_sent"],
                    # cancels sent, the other half of the wire traffic
                    "cancels": ex["n_cancels"],
                    # TOTAL OUTBOUND MESSAGES, which is the number a broker's
                    # session cap applies to and the one the CFO arm should
                    # roughly halve. Comparing n_orders_sent alone made the
                    # first run look as though CFO sent MORE, because on the
                    # baseline every reprice also sends a cancel that column
                    # never saw.
                    "total_msgs": ex["n_orders_sent"] + ex["n_cancels"],
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

    # ---- day-as-unit paired statistics, on BOTH P&L definitions ---------
    print()
    # AS RUN: the buffer's price move left in. This is what the trading
    # account would actually have shown, and it is the wrong number to pick a
    # policy with, because most of its variance is the stock.
    wide_raw, S_raw = paired_table(df, "net_pkr")
    # EX-BUFFER: the buffer's price move subtracted. Same fills, same fees,
    # same execution cost -- the stock's direction removed. This is the
    # quoting comparison.
    wide_ex, S_ex = paired_table(df, "net_pkr_ex_buffer")

    print("\n" + "=" * 78)
    print(f"RESULTS -- {len(wide_ex)} paired symbol-days")
    print("=" * 78)
    print("\nA) AS RUN -- net_pkr, the buffer's price move INCLUDED")
    print("   (what the account shows; dominated by the stock on long_buffer)")
    print(S_raw.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nB) POLICY COMPARISON -- net_pkr_ex_buffer, price move REMOVED")
    print("   (the buffer is still bought and still liquidated, so its fees")
    print("    and its execution cost stay in; only the direction is out)")
    print(S_ex.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\nREAD THIS BEFORE THE NUMBERS:")
    print("  * the |t| > 2 bar this project applies elsewhere applies here.")
    print("  * PICK THE POLICY OFF TABLE B. Table A's long_buffer column is a")
    print("    position in the stock plus a market-making result, added")
    print("    together, and the position is the larger of the two.")
    print("  * uptick_blocks == 0 on slb_uptick means the rule never engaged,")
    print("    so any difference on that arm is noise and not the constraint.")
    print("  * a sample this size measures whether the effect is LARGE. It")
    print("    cannot rule out a small one.")
    print("  * CHECK THE BOX PLOT, NOT THE TOTAL. On the first smoke run one")
    print("    symbol-day carried more than the whole long_buffer difference;")
    print("    the other nine were net negative. A total is one number and a")
    print("    box is ten -- the box is the one that tells you.")

    # ---- is any arm's result carried by ONE day? -------------------------
    concentration(wide_raw, "as run (price move included)")
    concentration(wide_ex, "policy comparison (price move removed)")

    # ---- what the buffer arm is NOT telling you -------------------------
    # Three things the per-day model gets wrong about a standing buffer, each
    # quantified from this run's own rows rather than asserted.
    # the long_buffer rows only
    buf = df[df.arm == "long_buffer"]
    # nothing to say if the arm did not run
    if len(buf) and buf["buffer_qty"].max() > 0:
        print("\nTHE BUFFER, SEPARATELY -- what the P&L column does not show")
        # capital tied up: shares x acquisition price, averaged over the days
        cap = (buf["buffer_qty"] * buf["buffer_px"]).mean()
        # the directional term that table B removes, day by day
        mv = buf["buffer_price_move"]
        print(f"  standing capital       {cap:>12,.0f} PKR "
              f"({buf['buffer_qty'].iloc[0]:,.0f} shares, average entry price)")
        print(f"  price move, per day    {mv.mean():>12,.2f} PKR mean, "
              f"{mv.std(ddof=1):,.2f} sd, worst {mv.min():+,.2f}")
        print(f"  ...against a typical day of quoting of "
              f"{df[df.arm == BASELINE]['net_pkr'].mean():,.2f} PKR.")
        print("  THE OVERNIGHT GAP IS NEVER MEASURED. Each date is an")
        print("  independent run: the buffer is bought at that day's first")
        print("  print and sold into that day's close, so the model sees the")
        print("  INTRADAY move and never the close-to-open gap. For a buffer")
        print("  that is actually held, the gap is the dominant exposure.")
        # the fee the model charges every day but reality pays once
        rt = 2.0 * FEE_TOTAL_PCT * cap
        # ---- what the free acquisition is worth ------------------------
        # The buffer is handed to us at the first print, whichever side that
        # print was on. A real market buy pays the ask. This is the gap.
        und = buf["buffer_acq_understated_pkr"].dropna()
        # only report it when at least one day had a two-sided book
        if len(und):
            print(f"  FREE ACQUISITION WORTH: the buffer is booked at the first")
            print(f"  print, not the ask. A market buy would have paid")
            print(f"  {und.mean():,.2f} PKR/day more on average "
                  f"(worst {und.max():+,.2f}, best {und.min():+,.2f}),")
            print(f"  totalling {und.sum():,.2f} PKR over {len(und)} "
                  f"symbol-days. That is a cost long_buffer never pays.")
            # the spread it implies, so the number is interpretable
            sp = (buf["buffer_ask"] - buf["buffer_bid"]).dropna()
            # only when both sides were present
            if len(sp):
                print(f"  (spread at acquisition: {sp.mean():.4f} PKR mean, "
                      f"{sp.max():.4f} widest -- the open is the wide part")
                print(f"   of the day, which is why this is the worst moment")
                print(f"   in the session to anchor an acquisition price to.)")
            # days where the print was AT OR WORSE than the ask
            n_neg = int((und <= 0).sum())
            # say so plainly rather than hiding it in the mean
            if n_neg:
                print(f"  On {n_neg} of {len(und)} days the value is <= 0: the")
                print(f"  print was at or above the ask, so the model paid MORE")
                print(f"  than a market order would have. Not clamped.")
        else:
            print(f"  FREE ACQUISITION WORTH: not measurable -- the book was")
            print(f"  one-sided at every acquisition, so there was no ask to")
            print(f"  compare against. Nothing is assumed in its place.")
        print(f"  ROUND-TRIP FEE OVERSTATED: the buffer is re-bought and")
        print(f"  re-sold every day, costing {rt:,.2f} PKR/day in fees. A")
        print(f"  buffer that is held pays that ONCE. Over {len(buf)} "
              f"symbol-days that is {rt * len(buf):,.2f} PKR of cost the real")
        print(f"  book would not pay -- so table B UNDERSTATES long_buffer by")
        print(f"  about that much, plus the liquidation spread each day.")

    # ---- write ---------------------------------------------------------
    # a fresh timestamped destination; never overwrites
    out = EX.safe_out("short_policy_sweep", "csv")
    # the long-form rows, which carry everything the summary condenses
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    # the summary beside it. BOTH tables go into one file with a `basis`
    # column, so a reader cannot pick up the wrong one by accident.
    out_s = EX.safe_out("short_policy_summary", "csv")
    # tag each table with the P&L definition it was computed on
    S_raw_t = S_raw.copy()
    S_raw_t.insert(0, "basis", "as_run_incl_buffer_move")
    S_ex_t = S_ex.copy()
    S_ex_t.insert(0, "basis", "ex_buffer_move")
    # one frame, both bases
    pd.concat([S_raw_t, S_ex_t], ignore_index=True).to_csv(out_s, index=False)
    print(f"wrote {out_s}")

    # ---- chart ----------------------------------------------------------
    # built by make_chart so the same code path is unit-testable
    make_chart(wide_raw, wide_ex, len(names), len(dates),
               EX.safe_out("short_policy_sweep", "png"))
# entry point
if __name__ == "__main__":
    main()
