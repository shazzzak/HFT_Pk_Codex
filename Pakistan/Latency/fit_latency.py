#!/usr/bin/env python3
# ============================================================================
#  fit_latency.py — per-day FEED latency characterisation for the PSX pipeline
#
#  Reads   parsed/date=YYYY-MM-DD/{trades,book_updates,book_snapshots}.parquet
#  Writes  calib/as_of=YYYY-MM-DD/feed_latency.json
#
#  WHAT THIS IS FOR (read before using the numbers)
#  ------------------------------------------------
#  `capture_ts` is stamped at wire arrival, so the PRODUCTION backtest uses
#  capture_ts DIRECTLY as the strategy's action clock. It does not sample from
#  a fitted distribution — the raw timestamps already contain every spike and
#  every quiet stretch exactly as they occurred. This fitter therefore exists
#  for three narrower purposes:
#
#     1. DIAGNOSTIC   — what does our feed staleness actually look like?
#     2. DATA QUALITY — flag degraded-capture days so they can be excluded or
#                       down-weighted (a p99 blowout is a capture problem, not
#                       a market signal).
#     3. STRESS       — supply parameters for "what if latency were 2x worse"
#                       runs, via block resampling that preserves clustering.
#
#  WHAT IS AND IS NOT IDENTIFIABLE
#  -------------------------------
#  raw_diff = capture_ts - transact_time = true_latency + clock_offset(t)
#
#  A CONSTANT part of raw_diff cannot be split into "clock offset" vs "minimum
#  network+processing latency" from these two columns alone. Only the component
#  ABOVE a slowly-varying floor is identifiable. So this fitter reports:
#     * baseline(t)  — rolling low quantile = offset + floor (NOT interpretable
#                      as either one alone)
#     * excess       — raw_diff - baseline(t): the identifiable jitter, which is
#                      what drives pick-off risk (F48)
#  Absolute latency level requires external evidence (PTP/NTP sync logs, or a
#  known-good reference feed). `absolute_floor_identifiable` is always false.
#
#  Note this is ALSO why using capture_ts directly as the action clock is robust:
#  a constant clock offset shifts every event equally, leaving event ordering and
#  inter-event gaps — the only things the strategy reacts to — unchanged.
#
#  Usage
#  -----
#    python fit_latency.py --parsed-root ./psx/parsed --calib-root ./psx/calib \
#                          --date 2026-06-25
#    python fit_latency.py ... --date 2026-06-25 --trailing ./psx/calib   # enables
#                                                                        # p99 blowout QC
# ============================================================================

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FITTER_VERSION = 1

# --------------------------------------------------------------------------- config

@dataclass
class FitConfig:
    baseline_window: str = "10min"   # rolling window for the clock-offset baseline
    baseline_q: float = 0.01         # low quantile taken as offset+floor
    baseline_min_periods: int = 200  # events needed before a baseline is trusted
    spike_q: float = 0.99            # excess quantile defining a "spike"
    congestion_bin: str = "1s"       # bin width for the rate-vs-latency regression
    cluster_block: str = "60s"       # block width for the index-of-dispersion test
    # QC thresholds
    max_negative_frac: float = 0.001     # >0.1% negative raw diffs => clock unsync
    # Drift matters RELATIVE to the jitter being measured: a swing several times
    # the median excess would corrupt any static-baseline estimate, even if small
    # in absolute ms. Flag when swing > max(floor, mult x median excess).
    drift_floor_ms: float = 10.0
    drift_vs_excess_mult: float = 3.0
    p99_blowout_mult: float = 3.0        # p99 excess vs trailing median => degraded


# --------------------------------------------------------------------------- io

EVENT_STEMS = ("trades", "book_updates", "book_snapshots")
NEEDED = ("transact_time", "capture_ts")
WANT = ("transact_time", "capture_ts", "channel", "symbol")


def _peek_columns(fp: Path) -> list[str]:
    if fp.suffix == ".parquet":
        import pyarrow.parquet as pq          # noqa: PLC0415
        return list(pq.ParquetFile(fp).schema.names)
    return list(pd.read_csv(fp, nrows=0).columns)


def _read_cols(fp: Path, cols: list[str]) -> pd.DataFrame:
    if fp.suffix == ".parquet":
        return pd.read_parquet(fp, columns=cols)
    return pd.read_csv(fp, usecols=cols)


def load_day_timestamps(parsed_root: Path, date: str) -> pd.DataFrame:
    """Union the (transact_time, capture_ts[, channel, symbol]) columns of every
    parsed file for one day. Only these columns are read, so this is cheap even
    on a full day of order-level updates.

    Parquet is preferred; .csv / .csv.gz are accepted as a fallback so the same
    tool works before the parsed layer is converted to parquet.
    """
    day_dir = parsed_root / f"date={date}"
    if not day_dir.is_dir():
        raise FileNotFoundError(f"no parsed dir: {day_dir}")
    frames, skipped = [], []
    for stem in EVENT_STEMS:
        fp = next((day_dir / f"{stem}{ext}" for ext in
                   (".parquet", ".csv", ".csv.gz")
                   if (day_dir / f"{stem}{ext}").exists()), None)
        if fp is None:
            continue
        have = _peek_columns(fp)
        cols = [c for c in WANT if c in have]
        missing = set(NEEDED) - set(cols)
        if missing:
            skipped.append(f"{fp.name} (missing {sorted(missing)})")
            continue
        df = _read_cols(fp, cols)
        df["src"] = stem
        frames.append(df)
    for s in skipped:
        print(f"  ! skipped {s}")
    if not frames:
        raise ValueError(f"no usable event files with {NEEDED} in {day_dir}")
    out = pd.concat(frames, ignore_index=True)
    for c in NEEDED:
        out[c] = pd.to_datetime(out[c])
    if "channel" not in out.columns:
        out["channel"] = -1
    return out.sort_values("transact_time").reset_index(drop=True)


# ------------------------------------------------------------------ core estimation

def _q(a: np.ndarray, p: float) -> float:
    return float(np.nanquantile(a, p)) if a.size else float("nan")


def estimate_baseline(raw_ms: pd.Series, ts: pd.Series, cfg: FitConfig) -> pd.Series:
    """Rolling low quantile of raw_diff = a slowly-varying estimate of
    (clock_offset + minimum latency). Rolling, not a single daily minimum,
    because unsynced clocks DRIFT over hours."""
    s = pd.Series(raw_ms.values, index=pd.DatetimeIndex(ts.values)).sort_index()
    base = s.rolling(cfg.baseline_window, min_periods=cfg.baseline_min_periods) \
            .quantile(cfg.baseline_q)
    # backfill the warm-up region with the first trusted value
    base = base.bfill()
    if base.isna().all():                      # very small day: fall back to global
        base = pd.Series(np.nanquantile(s.values, cfg.baseline_q), index=s.index)
    return base


def dist_stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in
                ("n", "mean", "sd", "p50", "p90", "p99", "p999", "max")}
    return dict(n=int(x.size), mean=float(np.mean(x)), sd=float(np.std(x, ddof=1))
                if x.size > 1 else 0.0,
                p50=_q(x, .50), p90=_q(x, .90), p99=_q(x, .99),
                p999=_q(x, .999), max=float(np.max(x)))


def congestion_test(ts: pd.Series, excess_ms: np.ndarray, cfg: FitConfig) -> dict:
    """Does staleness worsen when the feed is busy? Bin by time, then relate
    per-bin message RATE to per-bin excess percentiles.
    A positive slope means congestion is real -> heavy days matter most, and a
    single average latency number is actively misleading."""
    df = pd.DataFrame({"ts": pd.DatetimeIndex(ts.values), "e": excess_ms}) \
        .set_index("ts").sort_index()
    g = df.resample(cfg.congestion_bin)["e"]
    binned = pd.DataFrame({"rate": g.count(),
                           "p99": g.quantile(0.99),
                           "mean": g.mean()}).dropna()
    binned = binned[binned["rate"] > 0]
    if len(binned) < 30:
        return dict(n_bins=int(len(binned)), spearman_rate_p99=float("nan"),
                    ols_slope_ms_per_msg=float("nan"), verdict="insufficient_bins")
    rho = float(binned["rate"].corr(binned["p99"], method="spearman"))
    # OLS slope of p99 excess on message rate
    x = binned["rate"].values.astype(float)
    y = binned["p99"].values.astype(float)
    slope = float(np.polyfit(x, y, 1)[0])
    verdict = ("congestion_present" if (rho > 0.30 and slope > 0)
               else "no_clear_congestion")
    return dict(n_bins=int(len(binned)), spearman_rate_p99=rho,
                ols_slope_ms_per_msg=slope,
                rate_p50=float(np.median(x)), rate_p99=_q(x, .99), verdict=verdict)


def clustering_test(ts: pd.Series, excess_ms: np.ndarray, cfg: FitConfig) -> dict:
    """Are latency spikes BUNCHED or scattered?

    Matters twice over: bunched spikes are far more damaging than scattered ones
    (many quotes exposed simultaneously), and any stress-test resampling must be
    BLOCK resampling to preserve the bunching.

    Statistic: spike counts per time block, compared against the variance
    expected if spikes were INDEPENDENT across events. Note the raw index of
    dispersion (var/mean) is NOT a valid test here, because blocks contain
    different numbers of EVENTS and that alone inflates count variance. Under
    independence with per-block event counts n_b and spike probability p, the
    law of total variance gives
          Var_indep = p(1-p)*E[n] + p^2*Var[n]
    so we report  dispersion_ratio = Var_observed / Var_indep.  ~1 => scattered.

    CAVEAT: this detects bunching from ANY cause. Congestion-driven latency
    (see congestion_test) also bunches high-latency events, so a 'clustered'
    verdict does not attribute the cause — it only says spikes arrive together,
    which is what matters for exposure and for resampling design.
    """
    e = np.asarray(excess_ms, dtype=float)
    ok = np.isfinite(e)
    if ok.sum() < 1000:
        return dict(threshold_ms=float("nan"), dispersion_ratio=float("nan"),
                    lag1_autocorr=float("nan"), verdict="insufficient_events")
    thr = _q(e[ok], cfg.spike_q)
    idx = pd.DatetimeIndex(ts.values)
    spike = pd.Series((e > thr).astype(float), index=idx).sort_index()
    ones = pd.Series(np.ones(len(spike)), index=spike.index)
    counts = spike.resample(cfg.cluster_block).sum()
    nb = ones.resample(cfg.cluster_block).sum()
    m = nb.values > 0
    counts, nbv = counts.values[m], nb.values[m]
    if counts.size < 5:
        return dict(threshold_ms=thr, dispersion_ratio=float("nan"),
                    lag1_autocorr=float("nan"), verdict="insufficient_blocks")
    p = float(spike.sum() / len(spike))
    var_indep = p * (1 - p) * float(np.mean(nbv)) + (p ** 2) * float(np.var(nbv))
    var_obs = float(np.var(counts, ddof=1))
    ratio = float(var_obs / var_indep) if var_indep > 0 else float("nan")
    ac1 = float(pd.Series(counts).autocorr(lag=1)) if counts.size > 5 else float("nan")
    verdict = ("clustered" if (np.isfinite(ratio) and ratio > 1.5)
               else "near_independent")
    return dict(threshold_ms=thr, n_spikes=int(spike.sum()), spike_prob=p,
                dispersion_ratio=ratio, var_observed=var_obs,
                var_under_independence=var_indep,
                lag1_autocorr=ac1, verdict=verdict)


# ------------------------------------------------------------------------ qc

def quality_flags(raw_ms: np.ndarray, baseline: pd.Series, excess: np.ndarray,
                  cfg: FitConfig, trailing_p99: float | None) -> dict:
    neg_frac = float(np.mean(raw_ms < 0)) if raw_ms.size else float("nan")
    drift = float(np.nanmax(baseline.values) - np.nanmin(baseline.values)) \
        if len(baseline) else float("nan")
    p50 = _q(excess, .50)
    p99 = _q(excess, .99)
    flags, notes = [], []
    if neg_frac > cfg.max_negative_frac:
        flags.append("CLOCK_UNSYNC")
        notes.append(f"{neg_frac:.2%} of raw diffs negative — capture clock is "
                     f"behind exchange clock; absolute latency meaningless.")
    drift_thr = max(cfg.drift_floor_ms,
                    cfg.drift_vs_excess_mult * p50 if np.isfinite(p50) else 0.0)
    if np.isfinite(drift) and drift > drift_thr:
        flags.append("CLOCK_DRIFT")
        notes.append(f"baseline swings {drift:.1f}ms intraday (threshold "
                     f"{drift_thr:.1f}ms = {cfg.drift_vs_excess_mult}x median "
                     f"excess {p50:.1f}ms) — rolling baseline required; a single "
                     f"daily minimum would corrupt the excess estimate.")
    if trailing_p99 is not None and np.isfinite(p99) and np.isfinite(trailing_p99) \
            and trailing_p99 > 0 and p99 > cfg.p99_blowout_mult * trailing_p99:
        flags.append("DEGRADED_CAPTURE")
        notes.append(f"p99 excess {p99:.1f}ms vs trailing median {trailing_p99:.1f}ms "
                     f"— treat as a capture problem, not a market regime.")
    return dict(flags=flags, notes=notes, negative_raw_frac=neg_frac,
                baseline_drift_ms=drift, drift_threshold_ms=drift_thr,
                status=("OK" if not flags else "REVIEW"))


def trailing_median_p99(calib_root: Path, date: str, k: int = 20) -> float | None:
    """Median of p99 excess over the most recent k prior fitted days."""
    if not calib_root.is_dir():
        return None
    vals = []
    for d in sorted(calib_root.glob("as_of=*"), reverse=True):
        day = d.name.split("=", 1)[1]
        if day >= date:
            continue
        fp = d / "feed_latency.json"
        if fp.exists():
            try:
                j = json.loads(fp.read_text())
                v = j["overall"]["excess_ms"]["p99"]
                if np.isfinite(v):
                    vals.append(float(v))
            except Exception:
                pass
        if len(vals) >= k:
            break
    return float(np.median(vals)) if vals else None


# ------------------------------------------------------------------- orchestration

def fit_day(df: pd.DataFrame, date: str, cfg: FitConfig,
            trailing_p99: float | None = None) -> dict:
    raw_ms = ((df["capture_ts"] - df["transact_time"]).dt.total_seconds()
              * 1e3).astype(float)
    baseline = estimate_baseline(raw_ms, df["transact_time"], cfg)
    # align baseline back to event order
    base_vals = baseline.reindex(pd.DatetimeIndex(df["transact_time"].values)) \
                        .values
    excess = raw_ms.values - base_vals

    out = {
        "date": date,
        "fitter_version": FITTER_VERSION,
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "n_events": int(len(df)),
        # ---- provenance / interpretation guards -------------------------------
        "provenance": {
            "capture_ts_semantics": "wire_arrival (confirmed)",
            "absolute_floor_identifiable": False,
            "absolute_floor_note": (
                "baseline = clock_offset + min_latency and cannot be decomposed "
                "from these two columns alone; only `excess` is identifiable."),
            "capture_host_representative": "unknown",
            "capture_host_note": (
                "If capture ran on a non-colocated/office network path, these "
                "figures are a PESSIMISTIC BOUND on a broker-adjacent box, not a "
                "forecast. Confirm the capture host's network path before "
                "quoting these numbers as 'our latency'."),
            "action_clock": (
                "PRODUCTION: backtest uses capture_ts directly; this fit is "
                "diagnostic/QC/stress only and is NOT sampled during replay."),
        },
        "overall": {
            "raw_diff_ms": dist_stats(raw_ms.values),
            "baseline_ms": {"p50": float(np.nanmedian(base_vals)),
                            "min": float(np.nanmin(base_vals)),
                            "max": float(np.nanmax(base_vals))},
            "excess_ms": dist_stats(excess),
        },
        "congestion": congestion_test(df["transact_time"], excess, cfg),
        "clustering": clustering_test(df["transact_time"], excess, cfg),
        "quality": quality_flags(raw_ms.values, baseline, excess, cfg, trailing_p99),
        "by_channel": {},
        "by_source": {},
    }
    # per-channel: PSX FIX is multi-channel and congestion can be channel-local
    for ch, g in df.groupby("channel"):
        e = excess[g.index.values]
        out["by_channel"][str(ch)] = {"n": int(len(g)), "excess_ms": dist_stats(e)}
    for src, g in df.groupby("src"):
        e = excess[g.index.values]
        out["by_source"][str(src)] = {"n": int(len(g)), "excess_ms": dist_stats(e)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit per-day PSX feed latency")
    ap.add_argument("--parsed-root", required=True)
    ap.add_argument("--calib-root", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--no-trailing-qc", action="store_true",
                    help="skip the p99-blowout check against prior days")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = FitConfig()
    parsed_root, calib_root = Path(a.parsed_root), Path(a.calib_root)
    df = load_day_timestamps(parsed_root, a.date)
    trailing = None if a.no_trailing_qc else trailing_median_p99(calib_root, a.date)
    res = fit_day(df, a.date, cfg, trailing)

    o = res["overall"]
    print(f"{a.date}  n={res['n_events']:,}  "
          f"raw p50={o['raw_diff_ms']['p50']:.1f}ms  "
          f"excess p50/p99/max="
          f"{o['excess_ms']['p50']:.1f}/{o['excess_ms']['p99']:.1f}/"
          f"{o['excess_ms']['max']:.1f}ms  "
          f"congestion={res['congestion']['verdict']}  "
          f"spikes={res['clustering']['verdict']}  "
          f"status={res['quality']['status']}")
    for n in res["quality"]["notes"]:
        print(f"   ! {n}")

    if a.dry_run:
        return
    out_dir = calib_root / f"as_of={a.date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feed_latency.json").write_text(json.dumps(res, indent=2))
    print(f"   -> {out_dir / 'feed_latency.json'}")


if __name__ == "__main__":
    main()
