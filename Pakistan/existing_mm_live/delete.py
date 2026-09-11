import pandas as pd, numpy as np
df = pd.read_parquet("/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results/diagnostics/obi_decay_gate.parquet")
cfgs = [f"w_d{d}_r{r:g}" for d in (2,3,5) for r in (0.3,0.5,0.7)]
print(f"{'config':>10} {'n':>4} {'mean_rho':>9} {'fisherz':>9} {'frac>0':>7}")
for c in cfgs:
    a = df[f"{c}_addl1"].to_numpy(); a = a[np.isfinite(a)]
    # Fisher-z mean is the unbiased way to average correlations (my earlier flag)
    fz = np.tanh(np.mean(np.arctanh(np.clip(a, -0.999, 0.999))))
    print(f"{c:>10} {len(a):>4} {a.mean():>+9.4f} {fz:>+9.4f} {(a>0).mean():>7.2f}")
s = df["l1_addl1"].to_numpy(); s = s[np.isfinite(s)]
print(f"self-test l1_addl1: mean={s.mean():+.5f}  max|.|={np.max(np.abs(s)):.5f}")