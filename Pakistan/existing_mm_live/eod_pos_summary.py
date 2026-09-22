# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# eod_pos_summary.py -- the decisive check the terminal P&L can't give:
# did the skew fix move TERMINAL INVENTORY, or is it inert on position too?
# Pre-fix baseline (from the diagnosis, me=0.0005):
#   PPL: mean EOD -87, ~61-69% short, ~62% of days |pos| > 250 (half-cap)
#   UBL: mean EOD -138
# If post-fix means are still ~-87 / -138 and barely move across the ss sweep,
# the skew is NOT controlling terminal inventory -> confirms the tau->0 problem
# (or that directional flow, not quote placement, sets the drift).

# tables
import pandas as pd
# path
from pathlib import Path

# results dir
# Resolve this filesystem path through the canonical checkout/data configuration.
res = Path(str(_hft_paths.RESULTS_ROOT))
# per-day EOD positions emitted by confirm_micro_vs_naive.py
df = pd.read_csv(res / "eod_positions.csv")
# half of the 500-share inventory cap, for the "beyond half-cap" stat
HALF_CAP = 250

# group per (symbol, variant)
g = df.groupby(["symbol", "variant"])["eod_pos"]
# assemble the summary table
summary = pd.DataFrame({
    # number of days contributing
    "n_days": g.size(),
    # mean terminal position -> should move toward 0 if skew flattens
    "mean_pos": g.mean().round(1),
    # median is robust to outlier days
    "median_pos": g.median().round(1),
    # % of days ending short -> should move toward 50%
    "pct_short": (g.apply(lambda s: 100.0 * (s < 0).mean())).round(1),
    # % of days ending with |pos| beyond half the cap -> tail risk into liquidation
    "pct_beyond_half_cap": (g.apply(lambda s: 100.0 * (s.abs() > HALF_CAP).mean())).round(1),
}).reset_index()

# order variants by session_scale within each symbol for readability
# (naive first, then ss=0.25x .. 2x)
_rank = {"naive": 0, "micro me=0.0005 ss=0.25x": 1, "micro me=0.0005 ss=0.5x": 2,
         "micro me=0.0005 ss=1x": 3, "micro me=0.0005 ss=2x": 4}
# apply the rank (missing labels -> large number, sort last)
summary["_r"] = summary["variant"].map(lambda v: _rank.get(v, 99))
# sort and drop the helper
summary = summary.sort_values(["symbol", "_r"]).drop(columns="_r")

# print it
print(summary.to_string(index=False))
# explicit read-out of the decision
print("\nDECIDES: if mean_pos stays near -87 (PPL) / -138 (UBL) across the sweep,")
print("the skew is inert on terminal inventory -> stop tuning session_scale, the")
print("(T-t) term is the problem, not its magnitude.")
