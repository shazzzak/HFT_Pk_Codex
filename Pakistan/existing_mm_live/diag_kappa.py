# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# diagnose the negative-kappa result: (1) what fraction of trades have a
# RESOLVABLE resting_order_id, (2) what fraction of ORDER_ADDs match a fill,
# (3) the fill RATE (not intensity) by distance bin -- the honest picture.
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd
import run_legacy_mm as R
from pathlib import Path
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
SYM="PPL"; N_DAYS=5
DELTA_BINS=np.array([0.5,1,1.5,2,3,4,5,7,10,15,20])

dates=[str(d) for d in R.discover_dates()[:N_DAYS]]
tot_trades=0; resolved=0; tot_adds=0; matched=0
# per-bin fill counts and add counts
add_by_bin=np.zeros(len(DELTA_BINS)-1); fill_by_bin=np.zeros(len(DELTA_BINS)-1)

def mid_series(s):
    c=s[s["phase"]=="CONTINUOUS_AUCTION"]
    if len(c)==0: return None
    bids=c[c["entry_type"]=="BID"]; offs=c[c["entry_type"]=="OFFER"]
    if len(bids)==0 or len(offs)==0: return None
    bb=bids.groupby("msg_seq")["px"].max(); ba=offs.groupby("msg_seq")["px"].min()
    touch=pd.DataFrame({"bb":bb,"ba":ba}).dropna()
    mt=c.groupby("msg_seq")["orig_time"].first()
    touch["t_ms"]=R.to_ms(mt.reindex(touch.index)).to_numpy()
    touch["mid"]=0.5*(touch["bb"]+touch["ba"]); touch=touch.sort_values("t_ms")
    return touch["t_ms"].to_numpy(), touch["mid"].to_numpy()

for date in dates:
    dsets=R.open_datasets(date)
    if dsets is None: continue
    u=R.read_symbol(dsets["ob_updates"],R.REQ_UPDATES,SYM)
    t=R.read_symbol(dsets["trades"],R.REQ_TRADES,SYM)
    s=R.read_symbol(dsets["ob_snapshot"],R.REQ_SNAP,SYM)
    if len(u)==0 or len(s)==0: continue
    mids=mid_series(s)
    if mids is None: continue
    tmid,midv=mids
    # resolution rate
    if len(t)>0:
        tot_trades+=len(t)
        rid=t["resting_order_id"].map(R.parse_rest_oid)
        resolved+=rid.notna().sum()
        fillset=set(x for x in rid.dropna().tolist() if x)
    else:
        fillset=set()
    # adds
    adds=u[u["event"]=="ORDER_ADD"].copy()
    if len(adds)==0: continue
    adds["t_ms"]=R.to_ms(adds["transact_time"])
    tot_adds+=len(adds)
    def mid_at(tt):
        pos=np.searchsorted(tmid,tt,side="right")-1
        return midv[pos] if pos>=0 else None
    for k,px,tt_ in zip(adds["order_id"],adds["price"].astype(float),adds["t_ms"]):
        if k is None or not np.isfinite(px): continue
        m=mid_at(float(tt_))
        if m is None or m<=0: continue
        d=abs(px-m)/m*1e4
        b=np.searchsorted(DELTA_BINS,d)-1
        if 0<=b<len(add_by_bin):
            add_by_bin[b]+=1
            if k in fillset:
                matched+=1; fill_by_bin[b]+=1

print(f"{SYM}: trades={tot_trades:,}  resolvable resting_id={resolved:,} "
      f"({100*resolved/max(tot_trades,1):.1f}%)")
print(f"{SYM}: ORDER_ADDs={tot_adds:,}  matched to a fill={matched:,} "
      f"({100*matched/max(tot_adds,1):.1f}%)")
print("\nfill RATE by distance bin (fills/adds) -- should DECREASE with distance:")
for i,(lo,hi) in enumerate(zip(DELTA_BINS[:-1],DELTA_BINS[1:])):
    if add_by_bin[i]>0:
        print(f"  {lo:4.1f}-{hi:4.1f} bps:  adds={int(add_by_bin[i]):>7,}  "
              f"fills={int(fill_by_bin[i]):>7,}  rate={fill_by_bin[i]/add_by_bin[i]:.4f}")
