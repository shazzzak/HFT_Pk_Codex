# Back-solve session_scale so max-inventory skew == 1x median spread, per symbol.
import numpy as np
# median PKR touch spread (ba - bb) for the symbol, from your ob_snapshot table
median_spread_pkr = ...          # PKR
# representative PKR price level (median fair/mid) at the same regime
fair_ref = ...                   # PKR
# per-event fractional-return EMA vol at the same regime (~1e-4)
sigma_ref = ...                  # dimensionless
gamma = 0.15                     # risk aversion actually used
size0, max_inv = 50, 500         # base clip, inventory cap
pos_lots_max = max_inv / size0   # = 10 lots at the cap
tau = 1.0                        # worst case: the open
sigma_p = sigma_ref * fair_ref   # PKR price vol
# solve median_spread = gamma * sigma_p^2 * session_scale * tau * pos_lots_max
session_scale = median_spread_pkr / (gamma * sigma_p**2 * tau * pos_lots_max)
print(round(session_scale, 3))

