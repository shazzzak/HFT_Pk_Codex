# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# onesided_close_cost.py -- STEP 1, measured not inferred: how much do the
# one-sided-close days (mid_at_close is None) actually cost, per config?
# equity_mid_mark is None exactly when the closing book is one-sided (mm_backtest
# line 1132). On those days the engine force-liquidates by walking the book and
# haircuts any unfilled residual 10%. Under MID+band150 inventory is ~flat, so the
# position walked into those closes should be small -- this script checks whether
# that's true and what it costs, instead of inferring from mismatched aggregates.
#
# Reads the enriched eod_positions.csv written by confirm_micro_vs_naive.py
# (columns: symbol, variant, date, eod_pos, liquidated, mid_is_none, liq_clean).

# tables / arrays
import pandas as pd
import numpy as np
# path
from pathlib import Path

# results dir
# Resolve this filesystem path through the canonical checkout/data configuration.
res = Path(str(_hft_paths.RESULTS_ROOT))
# per-day records
df = pd.read_csv(res / "eod_positions.csv")

# guard: this analysis needs the enriched columns; older CSVs won't have them.
need = {"symbol", "variant", "date", "liquidated", "mid_is_none"}
missing = need - set(df.columns)
if missing:
    raise SystemExit(f"eod_positions.csv missing {missing} -- re-run the updated confirm first.")

# normalise the flag to bool (CSV may load it as string/0-1)
df["mid_is_none"] = df["mid_is_none"].astype(str).str.lower().isin(["true", "1", "1.0"])

# --- per (symbol, variant): split total liquidated P&L by day-type ------------
rows = []
# group by config
for (sym, var), g in df.groupby(["symbol", "variant"]):
    # two-sided-close days (normal)
    two = g[~g["mid_is_none"]]
    # one-sided-close days (the suspects)
    one = g[g["mid_is_none"]]
    rows.append({
        "symbol": sym, "variant": var,
        # counts
        "n_days": len(g),
        "n_onesided": len(one),
        "pct_onesided": round(100.0 * len(one) / len(g), 1) if len(g) else np.nan,
        # P&L split (this is the whole point: what do the one-sided days cost?)
        "pnl_total": round(g["liquidated"].sum(), 0),
        "pnl_twosided": round(two["liquidated"].sum(), 0),
        "pnl_onesided": round(one["liquidated"].sum(), 0),
        # how big a position we carry INTO a one-sided close (should be ~0 under band)
        "mean_abs_pos_onesided": round(one["eod_pos"].abs().mean(), 1) if len(one) else np.nan,
        # mean per-day cost on those days
        "mean_pnl_per_onesided_day": round(one["liquidated"].mean(), 0) if len(one) else np.nan,
    })

# assemble + order
out = pd.DataFrame(rows).sort_values(["symbol", "variant"])
# print full width
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(out.to_string(index=False))

# --- the headline read-out ----------------------------------------------------
print("\nREAD:")
print("  pnl_onesided = total liquidated P&L booked ON the one-sided-close days.")
print("  If it's a large negative slice of pnl_total, those days are where the loss")
print("  lives -- and since equity_liquidated force-walks a one-sided book with a")
print("  10% haircut on the unfilled residual, that cost is a CLOSE-MODEL artifact")
print("  (PSX has a closing auction; walking the continuous book at the bell is the")
print("  flagged simplification), not a strategy loss. mean_abs_pos_onesided says")
print("  how much inventory we actually carry into those closes under each config.")
