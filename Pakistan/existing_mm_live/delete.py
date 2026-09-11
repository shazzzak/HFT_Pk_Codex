import pandas as pd

d = pd.read_parquet("/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/fullyear_confirm_DAILY_20260911_1456.parquet")
CFG = "throttle"

# portfolio daily P&L + notional per (config, date): sum across buckets (+names if present)
g = d.groupby([CFG, "date"]).agg(pkr=("net_pkr","sum"), opn=("opened_notional","sum")).reset_index()
g["bps"] = g.pkr / g.opn * 1e4
piv = g.pivot(index="date", columns=CFG, values="bps").sort_index()

# split-half: is QBPS_2's edge over OBI steady across the year?
h = len(piv) // 2
print(f"{'config':>16} {'H1':>7} {'H2':>7}   edge_vs_OBI_H1/H2")
for c in piv.columns:
    h1, h2 = piv[c].iloc[:h].mean(), piv[c].iloc[h:].mean()
    e1 = h1 - piv["OBI"].iloc[:h].mean()
    e2 = h2 - piv["OBI"].iloc[h:].mean()
    print(f"{c:>16} {h1:>7.2f} {h2:>7.2f}   {e1:+.2f} / {e2:+.2f}")