# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# diag_sweep_tail.py -- find out what the 360-430 min sweep tail REALLY is.
#
# The phase diagnostic showed CONTINUOUS_AUCTION ends 15:29:59 and after-hours is
# only 15:35-15:48 (~13 min) -- too short to explain a 70-min tail. So the tail is
# NOT simply after-hours. This checks the actual clock times + session-relative
# minutes of trades for one symbol-day to locate the real cause (sess_open origin,
# auction trades, or stacking).
#
# Run from existing_mm_live/:  python3 diag_sweep_tail.py

# path handling
import sys
# make the driver importable
sys.path.insert(0, ".")
# arrays + frames
import numpy as np
import pandas as pd
# the driver
import run_legacy_mm as R
# store path
from pathlib import Path
# point at the parsed store
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))

# full trade columns
REQ_TRADES_FULL = ["symbol", "transact_time", "capture_ts", "price", "qty",
                   "initiator", "aggressor_side", "exec_type", "appl_seq"]
# snapshot columns (phase + time)
REQ_SNAP_FULL = ["symbol", "msg_seq", "orig_time", "entry_type", "px", "qty",
                 "phase"]

# one symbol-day to inspect
SYM = "TRG"
# pick the first available date
dates = [str(d) for d in R.discover_dates()]
# find a date with data
for date in dates:
    # open datasets
    dsets = R.open_datasets(date)
    # skip missing
    if dsets is None:
        continue
    # read trades
    t = R.read_symbol(dsets["trades"], REQ_TRADES_FULL, SYM)
    # read snapshots
    s = R.read_symbol(dsets["ob_snapshot"], REQ_SNAP_FULL, SYM)
    # need both
    if len(t) == 0 or len(s) == 0:
        continue
    # got a usable day
    break

# trade times in ms
t_ms = R.to_ms(t["transact_time"])
# snapshot times in ms
s_ms = R.to_ms(s["orig_time"])
# continuous snapshot times
cont_ms = s_ms[s["phase"] == "CONTINUOUS_AUCTION"]
# continuous open + close
t0 = float(cont_ms.min())
t1 = float(cont_ms.max())

# report the key anchors as clock times (convert ms epoch -> readable)
def clk(ms):
    # ms epoch -> pandas timestamp (UTC+5 PSX)
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert("Asia/Karachi")

# print the anchors
print(f"symbol {SYM}  date {date}")
print(f"continuous open  t0 = {clk(t0)}")
print(f"continuous close t1 = {clk(t1)}")
print(f"first trade time     = {clk(float(t_ms.min()))}")
print(f"last  trade time     = {clk(float(t_ms.max()))}")

# session-relative minutes under TWO possible origins
# origin A: continuous open (the CORRECT one)
sess_min_contopen = (t_ms - t0) / 60000.0
# origin B: first trade (what the buggy code may use if it precedes the gate)
sess_min_firsttrade = (t_ms - float(t_ms.min())) / 60000.0

# how many trades fall AFTER the continuous close?
after_close = t_ms > t1
# report
print(f"\ntrades after continuous close (t1): {int(after_close.sum())} "
      f"of {len(t_ms)} ({100*after_close.mean():.1f}%)")
# their clock-time span
if after_close.sum() > 0:
    # min/max clock time of post-close trades
    print(f"  post-close trade span: {clk(float(t_ms[after_close].min()))} "
          f"-> {clk(float(t_ms[after_close].max()))}")
    # their session-relative minute range under the continuous-open origin
    print(f"  post-close sess_min (cont-open origin): "
          f"{sess_min_contopen[after_close].min():.0f} -> "
          f"{sess_min_contopen[after_close].max():.0f}")

# the max session-relative minute under each origin (explains a 430-min axis)
print(f"\nmax sess_min if origin = continuous open : "
      f"{sess_min_contopen.max():.0f}")
print(f"max sess_min if origin = first trade     : "
      f"{sess_min_firsttrade.max():.0f}")
print("\nREAD: if 'max sess_min (cont-open)' ~ 358 but the chart shows ~430,")
print("the axis is being driven by trades AFTER t1 that the gate did not drop,")
print("OR sess_open was the auction/first-trade, not the continuous open.")
