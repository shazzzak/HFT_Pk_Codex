# confirm the tail is multi-day session-length heterogeneity, not a data leak:
# show, per day, the continuous session length (minutes) and the first-trade vs
# continuous-open gap, for one symbol across all days.
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import run_legacy_mm as R
from pathlib import Path
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
REQ_SNAP = ["symbol", "msg_seq", "orig_time", "phase"]
REQ_TR = ["symbol", "transact_time", "exec_type"]
SYM = "TRG"
rows = []
for date in [str(d) for d in R.discover_dates()]:
    dsets = R.open_datasets(date)
    if dsets is None: continue
    s = R.read_symbol(dsets["ob_snapshot"], REQ_SNAP, SYM)
    if len(s) == 0: continue
    cont = R.to_ms(s["orig_time"])[s["phase"] == "CONTINUOUS_AUCTION"]
    if len(cont) == 0: continue
    t0, t1 = float(cont.min()), float(cont.max())
    length_min = (t1 - t0) / 60000.0
    # weekday (Friday = 4)
    wd = pd.to_datetime(date).weekday()
    rows.append({"date": date, "weekday": wd, "cont_len_min": round(length_min, 1)})
df = pd.DataFrame(rows)
print(f"{SYM}: continuous session length across {len(df)} days")
print(df["cont_len_min"].describe().to_string())
print("\nby weekday (0=Mon..4=Fri):")
print(df.groupby("weekday")["cont_len_min"].agg(["count","min","median","max"]).to_string())
print("\ndays with UNUSUAL length (not ~358 min):")
print(df[(df.cont_len_min < 350) | (df.cont_len_min > 366)].to_string(index=False))
