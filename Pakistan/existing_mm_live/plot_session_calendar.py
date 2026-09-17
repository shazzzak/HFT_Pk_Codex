# ============================================================================
# plot_session_calendar.py -- pictures of the trading calendar
# ============================================================================
# WHAT THIS DRAWS, from the two CSVs session_calendar.py writes:
#
#   session_calendar_<stamp>.png   THE WHOLE HISTORY, three panels:
#       1. THE SHAPE OF EVERY DAY. One vertical mark per continuous trading
#          stretch, at its real clock time. The Ramadan block, the Friday
#          lunch break and every market halt are visible as shapes rather
#          than as numbers.
#       2. TIME IN SOME OTHER PHASE -- halts and call auctions inside the
#          session.
#       3. TIME NEVER OBSERVED -- the capture went quiet.
#
#   session_day_<date>.png         ONE DAY, in detail: each trading stretch,
#                                  each gap, with the gaps named and measured.
#
# WHY THREE PANELS RATHER THAN ONE STACKED BAR. Stacking traded, break, other
# and unobserved on one bar needs four colours side by side, and the pair that
# would have to sit adjacent fails the colour-separation floor for normal
# vision, never mind colour blindness. Small multiples cost nothing here and
# every panel is then a single series that needs no legend at all.
#
# Colours are the validated default palette: blue #2a78d6 and orange #eb6834
# pass every gate as an all-pairs set, and status red #d03b3b is reserved for
# the one thing that is a fault rather than a market state.
#
# READ-ONLY on its inputs. Writes PNGs into the results directory.
#
# Run from Production/:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/plot_session_calendar.py
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/plot_session_calendar.py --days 2026-03-02,2026-03-27
# ============================================================================

# command-line flags
import argparse
# path handling
from pathlib import Path
# newest-file selection
import re

# drawing, with no display attached
import matplotlib
matplotlib.use("Agg")
# the plotting interface
import matplotlib.pyplot as plt
# for the legend entries that are plain rectangles
from matplotlib.patches import Patch
# frames
import pandas as pd

# the results directory, from the one config every runner uses
try:
    from config_pk import RESULTS_ROOT
except Exception as _e:                                       # noqa: BLE001
    raise ImportError(
        "plot_session_calendar: could not import RESULTS_ROOT from config_pk "
        "(%r). Run with PYTHONPATH=../existing_mm_live." % _e)

# ---- THE PALETTE, from the validated default -----------------------------
# categorical slot 1: an ordinary-length session
C_REGULAR = "#2a78d6"
# categorical slot 2: a session that closed early
C_SHORT = "#eb6834"
# status critical: reserved for the fault, never for a market state
C_FAULT = "#d03b3b"
# ink
C_TEXT = "#0b0b0b"
# secondary ink, for axis labels and annotations
C_TEXT2 = "#52514e"
# the chart surface
C_SURFACE = "#fcfcfb"
# recessive grid
C_GRID = "#e3e2dd"


def newest(pattern):
    """The most recent file in the results directory matching a pattern."""
    # every match
    files = sorted(Path(RESULTS_ROOT).glob(pattern))
    # nothing to plot
    if not files:
        raise SystemExit(
            f"no {pattern} in {RESULTS_ROOT}. Run sim/session_calendar.py "
            f"first.")
    # the newest by the timestamp in its name, which sorts lexically
    return files[-1]


def hours(ts):
    """A timestamp's clock time as a decimal hour, for the y axis."""
    # hours plus the fractional part
    return ts.hour + ts.minute / 60.0 + ts.second / 3600.0


def style(ax):
    """The house style: recessive axes, no chartjunk."""
    # the surface
    ax.set_facecolor(C_SURFACE)
    # a recessive grid, behind the data
    ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    # data in front of it
    ax.set_axisbelow(True)
    # drop the box; keep only the two axes that carry scale
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # and mute the two that remain
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    # tick labels in secondary ink, never in a series colour
    ax.tick_params(colors=C_TEXT2, labelsize=9, length=0)


def plot_history(cal, spans, out):
    """The whole store, three panels."""
    # the regular market only
    C = cal[cal["is_reg_board"]].copy().sort_values("date")
    S = spans[spans["is_reg_board"]].copy()
    # dates as real dates, for a time axis that spaces months correctly
    C["d"] = pd.to_datetime(C["date"])
    S["d"] = pd.to_datetime(S["date"])
    # each stretch's start and end as a clock hour
    S["y0"] = [hours(t) for t in pd.to_datetime(S["start_pkt"])]
    S["y1"] = [hours(t) for t in pd.to_datetime(S["end_pkt"])]
    # whether that date was a shortened session, carried onto every stretch
    short = dict(zip(C["date"], C["is_short"]))
    S["is_short"] = S["date"].map(short).fillna(False)

    # three panels, sharing the date axis
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 11), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.18})
    # the surface
    fig.patch.set_facecolor(C_SURFACE)

    # ---- PANEL 1: the shape of every day --------------------------------
    ax = axes[0]
    style(ax)
    # every stretch as a vertical bar at its date, in its real clock time.
    # Drawn per group so the legend has two entries rather than 263.
    for flag, colour, label in ((False, C_REGULAR, "ordinary-length session"),
                                (True, C_SHORT, "shortened session")):
        # that group's stretches
        g = S[S["is_short"] == flag]
        # nothing in this group
        if len(g) == 0:
            continue
        # one thin vertical line per stretch
        ax.vlines(g["d"], g["y0"], g["y1"], color=colour, linewidth=2.2,
                  label=label, zorder=3)
    # MARKET HALTS ONLY, identified by the exchange's own phase code.
    # NOT simply "other_seconds is large": every Friday carries about 900
    # seconds of NORMAL_CALL_AUCTION_PM, the scheduled afternoon call
    # auction, and marking those as halts put a red triangle on all 36
    # Fridays in the first render. A halt is TEMPORARY_SUSPENSION.
    halt = C[C["other_phases"].fillna("").str.contains("TEMPORARY_SUSPENSION")]
    # marked above the day's close, so the mark cannot hide inside the bars
    if len(halt):
        ax.plot(halt["d"], [hours(t) + 0.35 for t in
                            pd.to_datetime(halt["close_pkt"])],
                marker="v", linestyle="none", markersize=7, color=C_FAULT,
                label="market halt (exchange phase TEMPORARY_SUSPENSION)",
                zorder=4)
    # the y axis is a clock
    ax.set_ylim(8.5, 18.0)
    ax.set_yticks(range(9, 18))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(9, 18)])
    ax.set_ylabel("clock time (PKT)", color=C_TEXT2, fontsize=10)
    # the title says what is being looked at, not what it is called
    ax.set_title(
        "PSX Regular Market: when the market was actually open, every "
        "trading day in the store\n"
        "One mark per continuous trading stretch, at its real clock time. "
        "A Friday shows two, split by the Jumu'ah break.",
        color=C_TEXT, fontsize=12, loc="left", pad=14)
    # two series plus a status marker, so a legend is required
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              labelcolor=C_TEXT2, ncol=3)

    # ---- PANEL 2: time in some other phase ------------------------------
    ax = axes[1]
    style(ax)
    # a single series, so no legend: the title names it
    ax.bar(C["d"], C["other_seconds"] / 60.0, width=2.0, color=C_SHORT,
           zorder=3)
    # minutes are the readable unit here
    ax.set_ylabel("minutes", color=C_TEXT2, fontsize=10)
    # THE TITLE EXPLAINS THE TWO HEIGHTS, so the reader does not have to
    # infer that the recurring ~15 minute bars are routine.
    ax.set_title(
        "Inside the session, but NOT trading and NOT a scheduled break.  "
        "The recurring ~15 min bars are Friday's afternoon call auction; "
        "the ~65 min spikes are market halts.",
        color=C_TEXT, fontsize=11, loc="left", pad=8)
    # NAME ONLY THE HALTS, and stagger the labels. The first render put five
    # dates at the same height a few days apart and they overprinted into an
    # unreadable smear.
    halts = C[C["other_phases"].fillna("").str.contains("TEMPORARY_SUSPENSION")]
    # each one, alternating the vertical offset
    for i, (_, r) in enumerate(halts.sort_values("d").iterrows()):
        # the label, with its own leader line so a staggered label is still
        # attributable to its bar
        ax.annotate(f"{r['date']}\n{r['other_seconds'] / 60:.0f} min",
                    xy=(r["d"], r["other_seconds"] / 60.0),
                    xytext=(-26 if i % 2 else 26, 14 + 16 * (i % 3)),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=C_TEXT2,
                    arrowprops=dict(arrowstyle="-", color=C_GRID,
                                    linewidth=0.9))
    # headroom for the staggered labels
    ax.set_ylim(0, max(1.0, float(C["other_seconds"].max()) / 60.0 * 1.75))

    # ---- PANEL 3: time never observed -----------------------------------
    ax = axes[2]
    style(ax)
    # the fault colour, because this one is our problem rather than PSX's
    ax.bar(C["d"], C["unobserved_seconds"] / 60.0, width=2.0, color=C_FAULT,
           zorder=3)
    ax.set_ylabel("minutes", color=C_TEXT2, fontsize=10)
    ax.set_title(
        "Inside the session, but NO message of any kind arrived "
        "— the capture went quiet",
        color=C_TEXT, fontsize=11, loc="left", pad=8)
    # the worst one, named
    for _, r in C.nlargest(3, "unobserved_seconds").iterrows():
        # only if it is actually large
        if r["unobserved_seconds"] <= 600:
            continue
        # the date, offset sideways so it cannot sit on its own bar
        ax.annotate(f"{r['date']}  {r['unobserved_seconds'] / 60:.0f} min",
                    xy=(r["d"], r["unobserved_seconds"] / 60.0),
                    xytext=(30, 2), textcoords="offset points",
                    ha="left", fontsize=8, color=C_TEXT2)
    # headroom for that label
    ax.set_ylim(0, max(1.0, float(C["unobserved_seconds"].max()) / 60.0 * 1.2))
    # only the bottom panel carries the date axis
    ax.set_xlabel("", color=C_TEXT2)

    # LAID OUT WITHOUT tight_layout: it warns and misplaces axes when a
    # panel carries offset annotations, which is every panel here.
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.06,
                        hspace=0.30)
    fig.savefig(out, dpi=120, facecolor=C_SURFACE)
    # release it
    plt.close(fig)
    # say where
    print(f"  wrote {out}")


def plot_day(cal, spans, date, out):
    """One date in detail: the stretches, and every gap named."""
    # that date's summary row for the regular market
    row = cal[(cal["is_reg_board"]) & (cal["date"] == date)]
    # a date not in the calendar cannot be drawn
    if len(row) == 0:
        print(f"  {date}: not in the calendar, skipped")
        return
    # the one row
    r = row.iloc[0]
    # its stretches, in order
    g = spans[(spans["is_reg_board"]) & (spans["date"] == date)] \
        .sort_values("span")
    # nothing to draw
    if len(g) == 0:
        print(f"  {date}: no trading stretches, skipped")
        return
    # start and end as clock hours
    y0 = [hours(t) for t in pd.to_datetime(g["start_pkt"])]
    y1 = [hours(t) for t in pd.to_datetime(g["end_pkt"])]

    # one wide, short panel: this is a timeline, not a plot
    fig, ax = plt.subplots(figsize=(15, 3.4))
    fig.patch.set_facecolor(C_SURFACE)
    style(ax)
    # the colour says whether this day was short
    colour = C_SHORT if bool(r["is_short"]) else C_REGULAR
    # each stretch as a bar on one row
    for a, b in zip(y0, y1):
        ax.barh(0, b - a, left=a, height=0.42, color=colour, zorder=3)
        # its own length, written on it
        ax.annotate(f"{(b - a) * 3600:,.0f}s", xy=((a + b) / 2, 0),
                    ha="center", va="center", fontsize=9, color="white",
                    zorder=4)
    # THE GAPS BETWEEN THEM, which are the point of this chart: a consumer
    # testing open <= t <= close would count these as trading time.
    for i in range(len(y0) - 1):
        # the gap
        a, b = y1[i], y0[i + 1]
        # its length in seconds
        secs = (b - a) * 3600
        # drawn as an open span, so it reads as absence rather than presence
        ax.barh(0, b - a, left=a, height=0.42, color="none",
                edgecolor=C_TEXT2, linewidth=1.0, linestyle=":", zorder=3)
        # NAMED WITH WHAT THE EXCHANGE CALLED IT. Written out rather than
        # using `a or b`: an empty CSV cell reads back as float NaN, which is
        # TRUTHY, so `r["break_reasons"] or r["other_phases"]` returns the
        # NaN and the chart printed the word "nan" over the gap.
        what = ""
        # the declared break reasons, if there are any
        if not pd.isna(r["break_reasons"]) and str(r["break_reasons"]).strip():
            what = str(r["break_reasons"])
        # otherwise whatever other phase the exchange reported
        elif not pd.isna(r["other_phases"]) and str(r["other_phases"]).strip():
            what = str(r["other_phases"])
        # and if it reported nothing, say that rather than inventing a reason
        else:
            what = "no phase reported"
        # the label, above the bar
        ax.annotate(f"CLOSED {secs:,.0f}s\n{what.replace(',', chr(10))}",
                    xy=((a + b) / 2, 0.30), ha="center", va="bottom",
                    fontsize=8, color=C_TEXT2)
    # the x axis is a clock
    lo = min(y0) - 0.4
    hi = max(y1) + 0.4
    ax.set_xlim(lo, hi)
    # a tick every half hour
    ticks = [t / 2 for t in range(int(lo * 2), int(hi * 2) + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t):02d}:{int(round((t % 1) * 60)):02d}"
                        for t in ticks], rotation=0)
    # the single row needs no y scale
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.75)
    ax.set_xlabel("clock time (PKT)", color=C_TEXT2, fontsize=10)
    # the headline states the facts of the day
    ax.set_title(
        f"{date} ({r['weekday']}) — {r['day_type']}   ·   "
        f"traded {r['traded_seconds']:,.0f}s of "
        f"{r['open_to_close_seconds']:,.0f}s between the bells   ·   "
        f"break {r['break_seconds']:,.0f}s   ·   "
        f"other phase {r['other_seconds']:,.0f}s   ·   "
        f"unobserved {r['unobserved_seconds']:,.0f}s",
        color=C_TEXT, fontsize=11, loc="left", pad=12)
    # tight, then written once
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # which calendar to read
    ap.add_argument("--calendar", default=None,
                    help="the session_calendar CSV; default is the newest")
    # which spans file to read
    ap.add_argument("--spans", default=None,
                    help="the session_spans CSV; default is the newest")
    # which days to draw in detail
    ap.add_argument("--days", default=None,
                    help="comma-separated dates for the per-day charts; "
                         "default picks one of each interesting kind")
    # where the PNGs go
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to the results root")
    args = ap.parse_args()

    # the inputs
    cal_path = Path(args.calendar) if args.calendar \
        else newest("session_calendar_*.csv")
    span_path = Path(args.spans) if args.spans \
        else newest("session_spans_*.csv")
    print("=" * 70)
    print("PSX SESSION CALENDAR -- CHARTS")
    print("=" * 70)
    print(f"  calendar : {cal_path.name}")
    print(f"  spans    : {span_path.name}")
    # read them
    cal = pd.read_csv(cal_path)
    spans = pd.read_csv(span_path)
    # where the output goes
    outdir = Path(args.out) if args.out else Path(RESULTS_ROOT)
    # the stamp from the calendar's own filename, so the charts and the CSVs
    # that produced them are obviously the same run
    m = re.search(r"(\d{8}_\d{4})", cal_path.name)
    stamp = m.group(1) if m else "latest"
    print()

    # ---- the whole history ----------------------------------------------
    plot_history(cal, spans, outdir / f"session_calendar_{stamp}.png")

    # ---- a few individual days -------------------------------------------
    # the regular market's rows
    C = cal[cal["is_reg_board"]]
    # which days to draw
    if args.days:
        # exactly what was asked for
        days = [d.strip() for d in args.days.split(",") if d.strip()]
    else:
        # ONE OF EACH INTERESTING KIND, chosen from the data rather than
        # hard-coded: the worst halt, the worst unobserved day, a Friday, a
        # shortened day and an ordinary one.
        days = []
        # the day with the most time in some other phase
        if (C["other_seconds"] > 600).any():
            days.append(C.nlargest(1, "other_seconds")["date"].iloc[0])
        # the day with the most unobserved time
        if (C["unobserved_seconds"] > 600).any():
            days.append(C.nlargest(1, "unobserved_seconds")["date"].iloc[0])
        # a Friday with its lunch break
        fri = C[(C["day_type"] == "REGULAR_FRIDAY")
                & (C["continuous_spans"] == 2)]
        if len(fri):
            days.append(fri["date"].iloc[len(fri) // 2])
        # a shortened day
        sh = C[C["day_type"] == "SHORT_DAY"]
        if len(sh):
            days.append(sh["date"].iloc[len(sh) // 2])
        # and an ordinary one, for the comparison
        reg = C[C["day_type"] == "REGULAR_DAY"]
        if len(reg):
            days.append(reg["date"].iloc[len(reg) // 2])
        # no duplicates, original order
        days = list(dict.fromkeys(days))

    # each one
    for d in days:
        plot_day(cal, spans, d, outdir / f"session_day_{d}.png")

    print()
    print("  The history chart is the one to look at first: the Ramadan")
    print("  block, the Friday split and every halt are shapes in it.")


# entry point
if __name__ == "__main__":
    main()
