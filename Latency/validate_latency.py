#!/usr/bin/env python3
# ============================================================================
#  validate_latency.py — ground-truth tests for fit_latency.py + latency.py
#
#  Builds synthetic days where TRUE latency, TRUE clock offset (with drift),
#  TRUE congestion coupling and TRUE spike clustering are all known, then
#  checks the fitter recovers them. Without this, the fitter's numbers would
#  themselves be unvalidated — the exact failure mode we are trying to fix.
#
#  Run: python validate_latency.py
# ============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

import fit_latency as fl
import latency as lt

RNG = np.random.default_rng(11)
DAY = "2026-06-25"
OPEN = pd.Timestamp(f"{DAY} 09:30:00")
HOURS = 6.0


def make_day(offset_ms: float = 250.0, drift_ms: float = 20.0,
             floor_ms: float = 2.0, jitter_mean_ms: float = 3.0,
             congestion_ms_per_msg: float = 0.05,
             spike_ms: float = 60.0, n_spike_windows: int = 30,
             clustered: bool = True):
    """One synthetic trading day with fully known latency structure."""
    n_sec = int(HOURS * 3600)
    # bursty message rate: slow cycle x random on/off multiplier
    base = 20 + 10 * np.sin(np.linspace(0, 6 * np.pi, n_sec))
    burst = np.where(RNG.random(n_sec) < 0.05, RNG.uniform(4, 12, n_sec), 1.0)
    rate = np.maximum(base * burst, 1.0)
    counts = RNG.poisson(rate)

    sec_idx = np.repeat(np.arange(n_sec), counts)
    frac = RNG.random(sec_idx.size)
    t_s = sec_idx + frac
    order = np.argsort(t_s)
    t_s, sec_idx = t_s[order], sec_idx[order]
    transact = OPEN + pd.to_timedelta(t_s, unit="s")

    # ---- true latency components
    sd = jitter_mean_ms * 0.8
    jitter = RNG.lognormal(lt.math_log_mu(jitter_mean_ms, sd),
                           lt.math_log_sigma(jitter_mean_ms, sd), t_s.size)
    congestion = congestion_ms_per_msg * rate[sec_idx]

    spike = np.zeros(t_s.size)
    if clustered:                                   # bunched multi-second bursts
        starts = RNG.choice(n_sec - 30, n_spike_windows, replace=False)
        for s in starts:
            w = RNG.integers(5, 20)
            m = (sec_idx >= s) & (sec_idx < s + w)
            spike[m] += spike_ms
    else:                                           # scattered single events
        m = RNG.random(t_s.size) < 0.004
        spike[m] += spike_ms

    true_latency = floor_ms + jitter + congestion + spike
    true_offset = offset_ms + drift_ms * (t_s / (n_sec))     # linear drift
    capture = transact + pd.to_timedelta(true_latency + true_offset, unit="ms")

    df = pd.DataFrame({
        "transact_time": transact,
        "capture_ts": capture,
        "channel": RNG.integers(1, 4, t_s.size),
        "src": "book_updates",
    })
    truth = dict(true_latency=true_latency, true_offset=true_offset,
                 floor_ms=floor_ms, rate=rate, sec_idx=sec_idx)
    return df, truth


def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"FAILED: {name} {detail}"
    print(f"  ok  {name}{('  ' + detail) if detail else ''}")


# ============================================================ 1. main recovery test
print("=== scenario A: clustered spikes, congestion, drifting +250ms offset ===")
df, truth = make_day()
cfg = fl.FitConfig()
res = fl.fit_day(df, DAY, cfg, trailing_p99=None)
o = res["overall"]
print(f"events={res['n_events']:,}  raw p50={o['raw_diff_ms']['p50']:.1f}ms  "
      f"excess p50/p99/max={o['excess_ms']['p50']:.1f}/"
      f"{o['excess_ms']['p99']:.1f}/{o['excess_ms']['max']:.1f}ms")

# --- baseline should track (offset + floor), i.e. the UNIDENTIFIABLE constant
raw_ms = ((df["capture_ts"] - df["transact_time"]).dt.total_seconds() * 1e3).values
baseline = fl.estimate_baseline(pd.Series(raw_ms), df["transact_time"], cfg)
base_vals = baseline.reindex(pd.DatetimeIndex(df["transact_time"].values)).values
target = truth["true_offset"] + truth["floor_ms"]
mae = float(np.nanmean(np.abs(base_vals - target)))
corr = float(np.corrcoef(base_vals[np.isfinite(base_vals)],
                         target[np.isfinite(base_vals)])[0, 1])
check("baseline tracks (clock offset + latency floor)", mae < 6.0 and corr > 0.95,
      f"MAE={mae:.2f}ms corr={corr:.3f}")

# --- drift must be detected, since a single daily min would be wrong
check("intraday clock drift detected",
      "CLOCK_DRIFT" in res["quality"]["flags"],
      f"baseline swing={res['quality']['baseline_drift_ms']:.1f}ms (true 20ms)")

# --- excess should recover the identifiable part of true latency
true_excess = truth["true_latency"] - truth["floor_ms"]
est_p99, true_p99 = o["excess_ms"]["p99"], float(np.quantile(true_excess, 0.99))
rel = abs(est_p99 - true_p99) / true_p99
check("excess p99 recovers true jitter+spike tail", rel < 0.25,
      f"est={est_p99:.1f}ms true={true_p99:.1f}ms ({rel:.1%} err)")

est_p50, true_p50 = o["excess_ms"]["p50"], float(np.quantile(true_excess, 0.50))
check("excess p50 in the right range", abs(est_p50 - true_p50) < 4.0,
      f"est={est_p50:.1f}ms true={true_p50:.1f}ms")

# --- congestion coupling was injected: must be found
c = res["congestion"]
check("congestion detected (latency worsens with message rate)",
      c["verdict"] == "congestion_present",
      f"spearman={c['spearman_rate_p99']:.2f} slope={c['ols_slope_ms_per_msg']:.4f}ms/msg")

# --- spikes were bunched: dispersion must exceed Poisson
k = res["clustering"]
check("spike clustering detected", k["verdict"] == "clustered",
      f"disp_ratio={k['dispersion_ratio']:.2f} lag1={k['lag1_autocorr']:.2f}")

check("no false clock-unsync flag when offset is positive",
      "CLOCK_UNSYNC" not in res["quality"]["flags"],
      f"negative frac={res['quality']['negative_raw_frac']:.4f}")
check("per-channel breakdown present", len(res["by_channel"]) == 3)
check("absolute floor correctly declared unidentifiable",
      res["provenance"]["absolute_floor_identifiable"] is False)

# ==================================================== 2. scattered-spike control
print("\n=== scenario B: SCATTERED spikes, no congestion (must NOT fire) ===")
df_b, _ = make_day(clustered=False, congestion_ms_per_msg=0.0)
res_b = fl.fit_day(df_b, DAY, cfg)
kb = res_b["clustering"]
check("independent spikes report near-independent",
      kb["verdict"] == "near_independent",
      f"disp_ratio={kb['dispersion_ratio']:.2f} (null=1.0)")

# ==================================================== 3. negative-offset control
print("\n=== scenario C: capture clock BEHIND exchange clock (-120ms) ===")
df_c, _ = make_day(offset_ms=-120.0, drift_ms=0.0)
res_c = fl.fit_day(df_c, DAY, cfg)
check("clock-unsync flag fires on negative raw diffs",
      "CLOCK_UNSYNC" in res_c["quality"]["flags"],
      f"negative frac={res_c['quality']['negative_raw_frac']:.2%}")
check("status escalated to REVIEW", res_c["quality"]["status"] == "REVIEW")
# excess is still recoverable despite the bogus absolute level
check("excess still finite under bad clock sync",
      np.isfinite(res_c["overall"]["excess_ms"]["p99"]),
      f"p99={res_c['overall']['excess_ms']['p99']:.1f}ms")

# ==================================================== 4. degraded-capture QC
print("\n=== scenario D: p99 blowout vs trailing history ===")
res_d = fl.fit_day(df, DAY, cfg, trailing_p99=o["excess_ms"]["p99"] / 10.0)
check("degraded-capture flag fires on p99 blowout",
      "DEGRADED_CAPTURE" in res_d["quality"]["flags"])

# ==================================================== 5. latency model tiers
print("\n=== latency.py: tiering and gateway sweep ===")
t_ns = df["transact_time"].values.astype("datetime64[ns]").astype(np.int64)[:5000]
c_ns = df["capture_ts"].values.astype("datetime64[ns]").astype(np.int64)[:5000]

prod = lt.CaptureClock()
vis = prod.visible_at_ns(t_ns, c_ns)
check("CaptureClock is the production tier",
      prod.is_production and prod.tier == "production")
check("CaptureClock returns capture_ts verbatim", bool(np.array_equal(vis, c_ns)))
try:
    prod.visible_at_ns(t_ns, None)
    raise AssertionError("should have raised")
except ValueError:
    check("CaptureClock refuses to silently degrade without capture_ts", True)

emp = lt.EmpiricalFeedLatency(np.asarray(true_excess), baseline_ms=250.0, block=64)
ve = emp.visible_at_ns(t_ns, None)
check("EmpiricalFeedLatency is labelled second-best, not production",
      (not emp.is_production) and emp.tier == "second_best")
check("empirical visibility is monotone (feed cannot reorder)",
      bool(np.all(np.diff(ve) >= 0)))
check("empirical visibility strictly after exchange stamp", bool(np.all(ve > t_ns)))

fx = lt.FixedFeedLatency(mean_ms=5.0, jitter_sd_ms=1.0)
check("FixedFeedLatency is labelled simplified + carries a warning",
      fx.tier == "simplified" and "warning" in fx.describe())

sweep = lt.gateway_sweep()
check("gateway sweep spans the specified grid",
      [g.mean_ms for g in sweep] == list(lt.DEFAULT_GATEWAY_SWEEP_MS))
g = sweep[1]
eff = g.effective_ns(t_ns[:1000])
check("gateway delays order effectiveness", bool(np.all(eff > t_ns[:1000])))
cancel_lat = float(np.mean(g.draw_ms(20000, is_cancel=True)))
quote_lat = float(np.mean(g.draw_ms(20000, is_cancel=False)))
check("cancels traverse the slower path", cancel_lat > quote_lat,
      f"cancel={cancel_lat:.1f}ms quote={quote_lat:.1f}ms")

# break-even latency solver
be = lt.break_even_latency_ms([(5, 120.0), (10, 60.0), (20, -40.0), (50, -300.0)])
expected = 10 + 10 * (60.0 / 100.0)
check("break-even latency interpolates correctly", abs(be - expected) < 1e-9,
      f"{be:.2f}ms")
check("break-even returns None when edge never dies",
      lt.break_even_latency_ms([(5, 50.0), (50, 10.0)]) is None)

cfgL = lt.LatencyConfig(feed=prod, gateway=g, feed_qc_status="OK")
prov = cfgL.provenance()
check("provenance marks latency-sensitive results valid under production clock",
      prov["latency_sensitive_results_valid"] is True)
cfgS = lt.LatencyConfig(feed=fx, gateway=g)
check("provenance INVALIDATES latency-sensitive results under simplified clock",
      lt.LatencyConfig(feed=fx, gateway=g).provenance()
      ["latency_sensitive_results_valid"] is False)

print("\nALL CHECKS PASSED")
