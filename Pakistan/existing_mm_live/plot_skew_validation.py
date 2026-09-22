# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# plot_skew_validation.py
# Visual validation of the micro_mm.py inventory-skew fix.
# Figure 1: EOD signed-position distribution per symbol, one overlaid histogram
#           per variant -> did the mass move toward 0 / % short toward 50%?
# Figure 2: summed mid-mark vs summed liquidated P&L per variant -> did the
#           carry gap (mid-mark minus liquidated) shrink?
#
# Variants are read FROM THE DATA (naive, micro me=... ss=...x, and any pre_fix
# baseline rows you add by hand). No variant names are hard-coded.
#
# ---------------------------------------------------------------------------
# INPUT CONTRACT (emitted by confirm_micro_vs_naive.py after the wiring edits):
#   eod_positions.csv : symbol(str), variant(str), eod_pos(float, shares/day)
#   pnl_summary.csv   : symbol(str), variant(str),
#                       midmark_pnl(float, PKR summed), liquidated_pnl(float, PKR summed)
# ---------------------------------------------------------------------------

# CLI args for the two CSV paths.
import sys
# tables.
import pandas as pd
# arrays / binning.
import numpy as np
# headless rendering.
import matplotlib
# non-interactive backend BEFORE pyplot.
matplotlib.use("Agg")
# figures.
import matplotlib.pyplot as plt

# Inventory cap in shares -> the +/-50%-of-cap reference guides.
INV_CAP = 500


# Stable colour per variant, shared across both figures for cross-reading.
def _variant_colors(variants):
    # tab10/tab20 give distinct hues; extend if you sweep more than 20 variants.
    cmap = plt.get_cmap("tab10" if len(variants) <= 10 else "tab20")
    # map each variant name to a fixed colour by its sorted index.
    return {v: cmap(i % cmap.N) for i, v in enumerate(variants)}


# ------------------------------ figure 1 -----------------------------------
def plot_eod_positions(eod_df, out_path="eod_position_shift.png"):
    # symbols as subplot columns.
    symbols = sorted(eod_df["symbol"].unique())
    # one subplot per symbol, shared y for honest comparison.
    fig, axes = plt.subplots(1, len(symbols), figsize=(7 * len(symbols), 5), sharey=True)
    # normalise to an iterable even for a single symbol.
    axes = np.atleast_1d(axes)
    # iterate symbols.
    for ax, sym in zip(axes, symbols):
        # rows for this symbol.
        sub = eod_df[eod_df["symbol"] == sym]
        # variants present for this symbol, in stable order.
        variants = sorted(sub["variant"].unique())
        # fixed colour per variant.
        colors = _variant_colors(variants)
        # common bin edges across all variants so bars are comparable.
        lo, hi = sub["eod_pos"].min(), sub["eod_pos"].max()
        # 40 bins across the range (guard lo==hi).
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 41)
        # one translucent histogram per variant.
        for v in variants:
            # this variant's per-day EOD positions.
            vals = sub.loc[sub["variant"] == v, "eod_pos"].to_numpy()
            # skip empty.
            if vals.size == 0:
                continue
            # mean EOD position -> should move toward 0.
            mean_pos = vals.mean()
            # % of days ending short -> should move toward 50%.
            pct_short = 100.0 * (vals < 0).mean()
            # filled step histogram keeps many overlays legible.
            ax.hist(vals, bins=bins, histtype="stepfilled", alpha=0.35,
                    color=colors[v],
                    label=f"{v}: mean={mean_pos:+.0f}, short={pct_short:.0f}%")
            # dashed line at this variant's mean, same colour.
            ax.axvline(mean_pos, color=colors[v], linestyle="--", linewidth=1.3)
        # flat reference (well-behaved MM clusters near 0).
        ax.axvline(0.0, color="black", linewidth=1.0)
        # +/-50%-of-cap guides (the "62% of days beyond half-cap" diagnostic).
        ax.axvline(0.5 * INV_CAP, color="red", linestyle=":", linewidth=1.0)
        # short-side mirror.
        ax.axvline(-0.5 * INV_CAP, color="red", linestyle=":", linewidth=1.0)
        # title + axes.
        ax.set_title(f"{sym} -- EOD position")
        ax.set_xlabel("EOD signed position (shares)")
        ax.set_ylabel("days")
        # legend carries the mean / %-short readout.
        ax.legend(fontsize=7, loc="upper left")
    # layout, save, close.
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    # report.
    print(f"wrote {out_path}")


# ------------------------------ figure 2 -----------------------------------
def plot_pnl_gap(pnl_df, out_path="pnl_midmark_vs_liquidated.png"):
    # symbols as subplot columns.
    symbols = sorted(pnl_df["symbol"].unique())
    # one subplot per symbol (independent y; PKR scales differ by symbol).
    fig, axes = plt.subplots(1, len(symbols), figsize=(7 * len(symbols), 5))
    # normalise to iterable.
    axes = np.atleast_1d(axes)
    # iterate symbols.
    for ax, sym in zip(axes, symbols):
        # this symbol's rows, indexed by variant.
        sub = pnl_df[pnl_df["symbol"] == sym].set_index("variant")
        # variants present, stable order.
        variants = sorted(sub.index.unique())
        # x positions, one group per variant.
        x = np.arange(len(variants))
        # bar half-width.
        w = 0.4
        # mid-mark totals (left bar of each group).
        mid_vals = [float(sub.loc[v, "midmark_pnl"]) for v in variants]
        # liquidated totals (right bar); gap to mid-mark = carry cost.
        liq_vals = [float(sub.loc[v, "liquidated_pnl"]) for v in variants]
        # draw mid-mark bars.
        ax.bar(x - w / 2, mid_vals, w, label="mid-mark", color="#7fbf7f")
        # draw liquidated bars.
        ax.bar(x + w / 2, liq_vals, w, label="liquidated", color="#1f77b4")
        # zero line so sign is unambiguous.
        ax.axhline(0.0, color="black", linewidth=1.0)
        # variant labels on x, rotated (they are long: "micro me=... ss=...x").
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
        # y label + title.
        ax.set_ylabel("P&L (PKR), summed over days")
        ax.set_title(f"{sym} -- carry gap = mid-mark minus liquidated")
        # legend.
        ax.legend(fontsize=8)
    # layout, save, close.
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    # report.
    print(f"wrote {out_path}")


# ------------------------------ entrypoint ---------------------------------
if __name__ == "__main__":
    # default paths match confirm_micro_vs_naive.py's output directory.
    # Resolve this filesystem path through the canonical checkout/data configuration.
    _res = str(_hft_paths.RESULTS_ROOT)
    # override on CLI: python plot_skew_validation.py eod.csv pnl.csv
    eod_path = sys.argv[1] if len(sys.argv) > 1 else f"{_res}/eod_positions.csv"
    # second positional arg is the P&L summary CSV.
    pnl_path = sys.argv[2] if len(sys.argv) > 2 else f"{_res}/pnl_summary.csv"
    # load EOD positions.
    eod_df = pd.read_csv(eod_path)
    # fail loud if columns are mis-mapped, rather than plotting garbage.
    assert {"symbol", "variant", "eod_pos"}.issubset(eod_df.columns), \
        "eod CSV needs columns: symbol, variant, eod_pos"
    # load P&L summary.
    pnl_df = pd.read_csv(pnl_path)
    # assert its contract too.
    assert {"symbol", "variant", "midmark_pnl", "liquidated_pnl"}.issubset(pnl_df.columns), \
        "pnl CSV needs columns: symbol, variant, midmark_pnl, liquidated_pnl"
    # figure 1.
    plot_eod_positions(eod_df)
    # figure 2.
    plot_pnl_gap(pnl_df)
