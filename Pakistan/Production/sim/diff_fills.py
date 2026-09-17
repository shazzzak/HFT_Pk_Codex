# ============================================================================
# sim/diff_fills.py -- WHERE does one symbol-day's P&L difference come from?
# ============================================================================
# WHAT THIS IS FOR.
#
# sim/gate.py answers "do the two sides agree?" with one number per symbol-day.
# When they do not agree, that number says nothing about WHY. This takes ONE
# symbol-day, runs both sides exactly as the gate does, and takes the
# difference apart.
#
# THE DECOMPOSITION, IN THE ORDER IT IS PRINTED.
#
#   1. TRADING vs CLOSING OUT. The headline number every result in this project
#      is quoted in is `equity_liquidated` = cash + liquidation proceeds +
#      residual mark. Two runs can trade almost identically and still differ by
#      a lot, because they carried a different position into the close and the
#      liquidation walks the real book. This split says which it is, and it is
#      the first thing to read: if the trading cash agrees and only the close
#      differs, nothing about the order manager is wrong.
#
#   2. THE FILLS THEMSELVES. Counts, shares, and volume-weighted price per
#      side. Equal fill COUNTS with different volume-weighted prices means the
#      same orders filled in different places -- a queue-position question, not
#      an order-management one.
#
#   3. THE FIRST DIVERGENCE. The two fill logs walked forward together until
#      they stop matching, and the rows either side of that point printed. One
#      divergence early in the day propagates through everything after it, so
#      the LAST row that matched is the only place worth looking.
#
# WHAT IT WRITES. One timestamped CSV of both fill logs stacked with a `source`
# column, and one PNG: cumulative realised cash and position through the day,
# both sides on the same axes, with the first divergence marked.
#
# READ-ONLY on every input. Never overwrites: output names carry a timestamp.
#
# Run from Production/ with the research stack on the path:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/diff_fills.py \
#       --symbol NRL --date 2026-06-30
# ============================================================================

# command-line flags
import argparse
# path handling, so Production/ is importable when run as a script
import sys
from pathlib import Path

# make the package importable however this file was invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# frames and arrays
import pandas as pd
import numpy as np

# EVERYTHING THE GATE USES, IMPORTED FROM THE GATE. Re-implementing the loader
# or either runner here would let this tool and the gate drift apart, and then
# a difference this tool reports might be a difference between the two tools.
from sim.gate import (load_symbol_day, run_baseline, run_engine, pnl_of,
                      H, R, EX)

# THE PALETTE. The same two colours the session-calendar charts use, so every
# chart in this project reads as one set. Validated (light surface #fcfcfb):
# lightness band, chroma floor, CVD separation dE 24.7 protan / 32.7 tritan,
# normal-vision dE 33.6, contrast all PASS.
C_BACKTEST = "#2a78d6"
C_ENGINE = "#eb6834"
# a recessive ink for grid, axes and annotation
C_INK = "#6b6b6b"
# the chart surface the palette was validated against
C_SURFACE = "#fcfcfb"


def fills_frame(engine, source):
    """One engine's fill log as a frame, with the cash effect of each fill.

    The engine records fills as dicts; this adds the two derived columns the
    comparison needs and nothing else.
    """
    # no fills at all is a legitimate day, not an error
    if not engine.fills:
        # AN EMPTY FRAME WITH EVERY COLUMN THE CALLERS READ, including the two
        # running totals the chart plots. Leaving those out would turn a quiet
        # day into a KeyError three frames away from the cause.
        return pd.DataFrame({c: pd.Series(dtype="float64")
                             for c in ("t", "px", "qty", "signed_qty", "cash",
                                       "cum_pos", "cum_cash")}
                            ).assign(source="", side="", reason="", oid=0)
    # the log as given
    df = pd.DataFrame(engine.fills)
    # which run this came from
    df.insert(0, "source", source)
    # position change: a BUY adds shares, a SELL removes them
    df["signed_qty"] = np.where(df["side"] == "BUY", df["qty"], -df["qty"])
    # cash moves opposite to position, at OUR price. The fee is not included
    # here: it is a separate, small, monotonic drag and leaving it out keeps
    # this column readable as "what the trade itself did".
    df["cash"] = -df["signed_qty"] * df["px"]
    # running totals, which are what the chart plots
    df["cum_pos"] = df["signed_qty"].cumsum()
    df["cum_cash"] = df["cash"].cumsum()
    # oldest first, which the engines already guarantee, made explicit
    return df.sort_values("t").reset_index(drop=True)


def first_divergence(a, b):
    """Index of the first fill where the two logs stop agreeing, or None.

    Compared on the four things that define a fill: when, which side, at what
    price, for how much. Everything else is derived from those.
    """
    # the columns that define a fill
    keys = ["t", "side", "px", "qty"]
    # walk forward over the shorter of the two
    n = min(len(a), len(b))
    # the first row that differs
    for i in range(n):
        # compare the four fields as a tuple
        if tuple(a.loc[i, keys]) != tuple(b.loc[i, keys]):
            # this is where they part
            return i
    # no row differed, so the divergence is that one log is longer
    if len(a) != len(b):
        # the first row the shorter one does not have
        return n
    # identical logs
    return None


def decompose(bt, rep):
    """The P&L difference split into trading and closing out.

    equity_liquidated = cash + liq_cash + residual_mark. `cash` is everything
    the day's trading did; the other two are the cost of getting flat at the
    close. A difference in the second is not a difference in the first.
    """
    # both end-of-day reports
    e_bt, e_rep = bt.eod or {}, rep.eod or {}
    # the pieces, per side. cash is read off the engine because the eod dict
    # does not carry it separately.
    rows = []
    # each named component of the headline number
    for label, v_bt, v_rep in (
            ("cash from trading", bt.cash, rep.cash),
            ("position at close", e_bt.get("pos_at_close"),
             e_rep.get("pos_at_close")),
            ("mid at close", e_bt.get("mid_at_close"),
             e_rep.get("mid_at_close")),
            ("liquidation vwap", e_bt.get("liq_vwap"), e_rep.get("liq_vwap")),
            ("shares book could not absorb", e_bt.get("unfilled_sh"),
             e_rep.get("unfilled_sh")),
            ("residual marked", e_bt.get("residual_marked"),
             e_rep.get("residual_marked")),
            ("EQUITY LIQUIDATED", e_bt.get("equity_liquidated"),
             e_rep.get("equity_liquidated"))):
        # one row per component
        rows.append({"component": label, "backtest": v_bt, "engine": v_rep})
    # a frame, so it prints aligned
    out = pd.DataFrame(rows)
    # the difference, where both sides are numbers
    out["diff"] = [
        (r - b) if isinstance(b, (int, float)) and isinstance(r, (int, float))
        else None
        for b, r in zip(out["backtest"], out["engine"])]
    return out


def make_chart(a, b, sym, date, div_idx, path):
    """Cumulative cash and position through the day, both sides.

    TWO PANELS, ONE SHARED X AXIS, never two y-scales on one panel: cash and
    shares are different measures and overlaying them on twin axes is the
    single most misleading thing a chart of this kind can do.
    """
    # imported here so the tool still runs headless without a display
    import matplotlib
    # a non-interactive backend, because this writes a file
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # two stacked panels sharing the time axis; the top one is the answer and
    # gets the height
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    # the surface the palette was validated against
    fig.patch.set_facecolor(C_SURFACE)

    # both panels get the same treatment
    for ax in (ax1, ax2):
        # the same surface
        ax.set_facecolor(C_SURFACE)
        # a recessive horizontal grid only -- vertical rules add nothing here
        ax.grid(True, axis="y", color=C_INK, alpha=0.18, linewidth=0.6)
        # the grid sits behind the data
        ax.set_axisbelow(True)
        # drop the top and right spines; they enclose nothing
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # and mute the two that remain
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_INK)
            ax.spines[s].set_alpha(0.4)

    # hours since midnight, which is what a reader of a trading day wants
    ta = a["t"] / 3_600_000.0
    tb = b["t"] / 3_600_000.0
    # 2px lines, the spec for a line mark
    ax1.plot(ta, a["cum_cash"], color=C_BACKTEST, linewidth=2.0,
             label="mm_backtest", drawstyle="steps-post")
    ax1.plot(tb, b["cum_cash"], color=C_ENGINE, linewidth=2.0,
             label="production engine", drawstyle="steps-post")
    # the position panel, same encoding so the eye carries identity across
    ax2.plot(ta, a["cum_pos"], color=C_BACKTEST, linewidth=2.0,
             drawstyle="steps-post")
    ax2.plot(tb, b["cum_pos"], color=C_ENGINE, linewidth=2.0,
             drawstyle="steps-post")
    # flat is the reference line on a position chart
    ax2.axhline(0, color=C_INK, alpha=0.5, linewidth=1.0)

    # THE FIRST DIVERGENCE, marked on both panels. This is the whole point of
    # the chart: everything after this line is downstream of one event.
    if div_idx is not None:
        # the time of the last row that still matched, or the first row if
        # they never did
        src = a if div_idx < len(a) else b
        # guard the edge where one log is empty
        if len(src):
            # the divergence time, in hours
            t_div = float(src.loc[min(div_idx, len(src) - 1), "t"]) / 3_600_000.0
            # a vertical rule on each panel
            for ax in (ax1, ax2):
                ax.axvline(t_div, color=C_INK, linewidth=1.2,
                           linestyle="--", alpha=0.8)
            # labelled once, on the top panel, so it is not repeated
            ax1.annotate(f"first divergence\nfill #{div_idx + 1}",
                         xy=(t_div, ax1.get_ylim()[1]),
                         xytext=(6, -14), textcoords="offset points",
                         fontsize=9, color=C_INK, va="top")

    # a legend, because there are two series and identity must not be colour
    # alone
    ax1.legend(frameon=False, loc="upper left", fontsize=10)
    # axis titles carry the units
    ax1.set_ylabel("cumulative cash from fills (PKR)", fontsize=10,
                   color=C_INK)
    ax2.set_ylabel("position (shares)", fontsize=10, color=C_INK)
    ax2.set_xlabel("exchange time (hours)", fontsize=10, color=C_INK)
    # the title names what is plotted
    ax1.set_title(f"{sym} {date} -- backtest vs engine, fill by fill",
                  fontsize=13, loc="left", pad=12)
    # tick labels in the recessive ink
    for ax in (ax1, ax2):
        ax.tick_params(colors=C_INK, labelsize=9)
    # write it
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=C_SURFACE)
    # release the figure
    plt.close(fig)


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # which name
    ap.add_argument("--symbol", required=True)
    # which day
    ap.add_argument("--date", required=True)
    # which reprice mechanic, matching the gate's flag exactly
    ap.add_argument("--mode", choices=("replace", "cancel_new"),
                    default="replace")
    # how many rows of the fill logs to print either side of the divergence
    ap.add_argument("--context", type=int, default=6)
    args = ap.parse_args()
    # True when both sides use one amendment message per reprice
    use_g = (args.mode == "replace")

    print("=" * 78)
    print(f"FILL DIFF -- {args.symbol} {args.date}  (mode {args.mode})")
    print("=" * 78)

    # ---- the same inputs the gate builds -------------------------------
    # the date's partitions
    dsets = R.open_datasets(args.date)
    # a missing partition is fatal here: there is one day to look at
    if dsets is None:
        raise SystemExit(f"no datasets for {args.date}")
    # calibration, loaded the same way
    scales, profiles = H.load_scales(), H.load_profiles()
    windows, segments = H.load_windows(), H.load_segments()
    # the day's session segments
    segs = segments.get(str(args.date))
    # without them there is no calibrated session
    if segs is None:
        raise SystemExit(f"no session segments for {args.date}")
    # a name without calibration cannot be run
    if args.symbol not in scales:
        raise SystemExit(f"{args.symbol} has no calibration")
    # the strategy parameters, assembled exactly as every runner does
    params = H.build_micro_params(50, scales[args.symbol],
                                 profiles[args.symbol],
                                 windows[args.symbol], segs)
    # the shared inputs
    loaded = load_symbol_day(dsets, args.symbol)
    # an unrunnable symbol-day
    if loaded is None:
        raise SystemExit(f"{args.symbol} {args.date} has no usable events")
    # unpack
    events, snap_groups, t0, t1, ref_minor = loaded

    # ---- both runs, exactly as the gate runs them ----------------------
    # the thing being reproduced
    bt = run_baseline(events, snap_groups, params, t0, t1, use_g)
    # the production engine, on the policy the gate judges
    rep = run_engine(events, snap_groups, params, t0, t1, ref_minor,
                     args.symbol, args.date, use_g, quantity_policy="exact")

    # ---- 1. trading vs closing out --------------------------------------
    print("\n  WHERE THE DIFFERENCE LIVES")
    print("  The headline number is cash + liquidation + residual mark. If the")
    print("  trading cash agrees and only the close differs, the order manager")
    print("  is not what is being measured.")
    # the decomposition
    dec = decompose(bt, rep)
    print(dec.to_string(index=False,
                        float_format=lambda v: f"{v:,.2f}"))

    # ---- 2. the fills ----------------------------------------------------
    # both logs
    a = fills_frame(bt, "backtest")
    b = fills_frame(rep, "engine")
    print("\n  THE FILLS")
    # per side, for each run
    for side in ("BUY", "SELL"):
        # the rows on this side
        sa, sb = a[a["side"] == side], b[b["side"] == side]
        # volume-weighted price, which is what "filled in a different place"
        # actually means
        va = (sa["px"] * sa["qty"]).sum() / sa["qty"].sum() if len(sa) else None
        vb = (sb["px"] * sb["qty"]).sum() / sb["qty"].sum() if len(sb) else None
        # one line per side
        print(f"    {side:4}  backtest {len(sa):4} fills "
              f"{sa['qty'].sum():9,.0f} sh  vwap "
              f"{va if va is None else f'{va:,.4f}'}")
        print(f"          engine   {len(sb):4} fills "
              f"{sb['qty'].sum():9,.0f} sh  vwap "
              f"{vb if vb is None else f'{vb:,.4f}'}")
    # and by fill reason, which says WHICH rule produced the difference
    print("\n  BY FILL REASON")
    # counts per reason on each side, aligned
    ra = a["reason"].value_counts() if len(a) else pd.Series(dtype=int)
    rb = b["reason"].value_counts() if len(b) else pd.Series(dtype=int)
    # every reason either side saw
    for reason in sorted(set(ra.index) | set(rb.index)):
        print(f"    {reason:16} backtest {int(ra.get(reason, 0)):4}"
              f"   engine {int(rb.get(reason, 0)):4}")

    # ---- 3. the first divergence ----------------------------------------
    # where the two logs part company
    div = first_divergence(a, b)
    print("\n  FIRST DIVERGENCE")
    # identical logs is the answer the gate is looking for
    if div is None:
        print("    none -- the two fill logs are identical. Any P&L difference")
        print("    is therefore in the close, not in the trading.")
    else:
        print(f"    fill #{div + 1} of {len(a)} (backtest) / {len(b)} (engine)")
        print("    Everything after this point is downstream of it, so this is")
        print("    the only row worth reading.")
        # the window either side
        lo, hi = max(0, div - args.context), div + args.context + 1
        # the columns worth seeing
        cols = ["t", "side", "px", "qty", "reason", "oid"]
        print("\n    backtest:")
        print(a.loc[lo:hi, cols].to_string(index=True))
        print("\n    engine:")
        print(b.loc[lo:hi, cols].to_string(index=True))

    # ---- write ----------------------------------------------------------
    # both logs stacked, with the source column distinguishing them
    both = pd.concat([a, b], ignore_index=True)
    # a fresh timestamped destination; never overwrites
    out_csv = EX.safe_out(f"fill_diff_{args.symbol}_{args.date}", "csv")
    both.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    # the chart, beside it
    out_png = EX.safe_out(f"fill_diff_{args.symbol}_{args.date}", "png")
    make_chart(a, b, args.symbol, args.date, div, out_png)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
