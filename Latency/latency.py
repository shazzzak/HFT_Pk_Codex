#!/usr/bin/env python3
# ============================================================================
#  latency.py — latency handling for the PSX market-making backtest
#
#  Replaces the flagged placeholder (a single assumed feed latency) with:
#     * three EXPLICITLY TIERED feed clocks, so a simplified choice can never
#       be mistaken for the production one at the call site or in the output;
#     * gateway latency as a SWEPT AXIS rather than a guessed constant;
#     * a break-even-latency solver, which is the decision-grade output.
#
#  TIERING (per the standing instruction: never build on a simplified
#  foundation unknowingly). Every clock exposes .tier and .is_production, and
#  RunConfig.provenance() surfaces them into the run manifest.
#
#     CaptureClock          PRODUCTION.  Uses the recorded wire-arrival
#                           capture_ts directly. No model, no sampling: real
#                           spikes, real quiet stretches. Also robust to a
#                           constant clock offset, since every event shifts
#                           equally and only ordering/gaps affect decisions.
#
#     EmpiricalFeedLatency  SECOND-BEST. Samples that day's fitted excess
#                           distribution (block-bootstrap to preserve spike
#                           clustering). Use ONLY where per-event capture
#                           timestamps are missing.
#
#     FixedFeedLatency      SIMPLIFIED. Constant + jitter from a prior. This is
#                           the thing that was flagged as unanchored. Retained
#                           for stress/ablation only; is_production = False.
#
#  GATEWAY latency (decision -> matching engine) is NOT measurable pre-pilot:
#  no historical file contains it. It is therefore swept, never assumed, and
#  every result carries the latency it was produced under.
# ============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- feed clocks


class FeedClock:
    """Maps an event to the time the strategy is permitted to see it."""

    tier: str = "abstract"
    is_production: bool = False

    def visible_at_ns(self, transact_ns: np.ndarray,
                      capture_ns: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"class": type(self).__name__, "tier": self.tier,
                "is_production": self.is_production}


class CaptureClock(FeedClock):
    """PRODUCTION. Visibility = the recorded wire-arrival timestamp."""

    tier = "production"
    is_production = True

    def visible_at_ns(self, transact_ns, capture_ns):
        if capture_ns is None:
            raise ValueError(
                "CaptureClock requires capture_ts. If capture timestamps are "
                "absent, choose EmpiricalFeedLatency (second-best) explicitly "
                "rather than silently degrading.")
        # Guard: capture must not precede exchange stamp by more than the
        # clock offset absorbed elsewhere; a large negative gap means unsynced
        # clocks and the run manifest should carry the QC flag from fit_latency.
        return np.asarray(capture_ns, dtype=np.int64)


class EmpiricalFeedLatency(FeedClock):
    """SECOND-BEST. Draw excess latency from the fitted empirical distribution.

    Block bootstrap (not iid sampling) because fit_latency.py's clustering test
    typically shows spikes are bunched; iid draws would understate the damage.
    """

    tier = "second_best"
    is_production = False

    def __init__(self, excess_ms_samples: np.ndarray, baseline_ms: float,
                 block: int = 64, scale: float = 1.0, seed: int = 0):
        self.samples = np.asarray(excess_ms_samples, dtype=float)
        self.samples = self.samples[np.isfinite(self.samples)]
        if self.samples.size == 0:
            raise ValueError("no finite excess samples")
        self.baseline_ms = float(baseline_ms)
        self.block = int(block)
        self.scale = float(scale)          # >1.0 for "what if 2x worse" stress
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_calib(cls, calib_json: Path, **kw) -> "EmpiricalFeedLatency":
        """Reconstruct an approximate sampler from a fit_latency.py artifact.

        SIMPLIFICATION, FLAGGED: feed_latency.json stores summary quantiles, not
        the full sample. This rebuilds a piecewise-linear quantile function from
        (p50, p90, p99, p999, max), which reproduces the tail shape only
        approximately. For serious stress work, persist the raw excess array
        (or a 1000-point quantile grid) from the fitter instead.
        """
        j = json.loads(Path(calib_json).read_text())
        e = j["overall"]["excess_ms"]
        ps = np.array([0.50, 0.90, 0.99, 0.999, 1.0])
        qs = np.array([e["p50"], e["p90"], e["p99"], e["p999"], e["max"]], float)
        u = np.random.default_rng(0).uniform(0.0, 1.0, 200_000)
        samples = np.interp(u, ps, qs, left=qs[0])
        return cls(samples, j["overall"]["baseline_ms"]["p50"], **kw)

    def _draw(self, n: int) -> np.ndarray:
        nb = int(np.ceil(n / self.block))
        starts = self.rng.integers(0, max(len(self.samples) - self.block, 1), nb)
        idx = (starts[:, None] + np.arange(self.block)[None, :]).ravel()[:n]
        return self.samples[np.mod(idx, len(self.samples))] * self.scale

    def visible_at_ns(self, transact_ns, capture_ns):
        t = np.asarray(transact_ns, dtype=np.int64)
        lat_ms = self.baseline_ms + self._draw(t.size)
        vis = t + (lat_ms * 1e6).astype(np.int64)
        return np.maximum.accumulate(vis)      # a feed cannot deliver out of order

    def describe(self) -> dict:
        d = super().describe()
        d.update(baseline_ms=self.baseline_ms, block=self.block, scale=self.scale,
                 note="samples fitted excess distribution; block bootstrap")
        return d


class FixedFeedLatency(FeedClock):
    """SIMPLIFIED / PLACEHOLDER — the flagged unanchored option.

    A constant plus lognormal jitter from a prior. Produces a fresher, smoother
    world than reality. Do not use for any latency-sensitive conclusion.
    """

    tier = "simplified"
    is_production = False

    def __init__(self, mean_ms: float = 5.0, jitter_sd_ms: float = 1.0, seed: int = 0):
        self.mean_ms, self.jitter_sd_ms = float(mean_ms), float(jitter_sd_ms)
        self.rng = np.random.default_rng(seed)

    def visible_at_ns(self, transact_ns, capture_ns):
        t = np.asarray(transact_ns, dtype=np.int64)
        sd = max(self.jitter_sd_ms, 1e-9)
        mu = math_log_mu(self.mean_ms, sd)
        lat = self.rng.lognormal(mu, math_log_sigma(self.mean_ms, sd), t.size)
        return np.maximum.accumulate(t + (lat * 1e6).astype(np.int64))

    def describe(self) -> dict:
        d = super().describe()
        d.update(mean_ms=self.mean_ms, jitter_sd_ms=self.jitter_sd_ms,
                 warning="UNANCHORED PLACEHOLDER — not measured, not swept")
        return d


def math_log_sigma(mean: float, sd: float) -> float:
    return float(np.sqrt(np.log1p((sd / mean) ** 2)))


def math_log_mu(mean: float, sd: float) -> float:
    return float(np.log(mean) - 0.5 * np.log1p((sd / mean) ** 2))


# ------------------------------------------------------------------ gateway latency


@dataclass
class GatewayLatency:
    """Decision -> matching engine. UNMEASURABLE pre-pilot: swept, not assumed.

    `mean_ms` is a SWEEP POINT, not an estimate. Every run records it so no
    result can be read without knowing the latency it assumed.
    """
    mean_ms: float
    jitter_sd_ms: float = 0.0
    cancel_extra_ms: float = 0.0    # cancels often traverse a slower path
    seed: int = 0
    tier: str = "swept_unmeasurable"

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def draw_ms(self, n: int = 1, is_cancel: bool = False) -> np.ndarray:
        base = self.mean_ms + (self.cancel_extra_ms if is_cancel else 0.0)
        if self.jitter_sd_ms <= 0:
            return np.full(n, base, dtype=float)
        sd = self.jitter_sd_ms
        return self._rng.lognormal(math_log_mu(base, sd),
                                   math_log_sigma(base, sd), n)

    def effective_ns(self, decision_ns: np.ndarray, is_cancel: bool = False):
        d = np.asarray(decision_ns, dtype=np.int64)
        return d + (self.draw_ms(d.size, is_cancel) * 1e6).astype(np.int64)


DEFAULT_GATEWAY_SWEEP_MS = (5.0, 10.0, 20.0, 50.0)


def gateway_sweep(mean_ms_grid=DEFAULT_GATEWAY_SWEEP_MS,
                  jitter_frac: float = 0.30, cancel_extra_frac: float = 0.20,
                  seed: int = 0) -> list[GatewayLatency]:
    """The run grid's latency axis. Jitter scales with the mean (congestion
    widens the distribution as it shifts it)."""
    return [GatewayLatency(mean_ms=m, jitter_sd_ms=jitter_frac * m,
                           cancel_extra_ms=cancel_extra_frac * m, seed=seed)
            for m in mean_ms_grid]


# ------------------------------------------------------------------- decision output


def break_even_latency_ms(points: list[tuple[float, float]]) -> float | None:
    """Given [(gateway_latency_ms, pnl), ...], return the latency at which PnL
    crosses zero by linear interpolation between the bracketing sweep points.

    THE decision-grade number: it answers "is Route-A DMA viable at all?" and
    "what would co-location have to buy us?" — questions a single assumed
    latency cannot address. Returns None if the curve never crosses zero
    (report ">max swept" or "<min swept" in that case).
    """
    pts = sorted((float(a), float(b)) for a, b in points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 > 0 >= y1) or (y0 >= 0 > y1):
            if y0 == y1:
                return x0
            return x0 + (x1 - x0) * (y0 / (y0 - y1))
    return None


@dataclass
class LatencyConfig:
    """Attach to every run; serialised into the run manifest."""
    feed: FeedClock = field(default_factory=CaptureClock)
    gateway: GatewayLatency = field(default_factory=lambda: GatewayLatency(10.0, 3.0))
    feed_latency_calib: str | None = None      # path to feed_latency.json used for QC
    feed_qc_status: str | None = None          # 'OK' | 'REVIEW' from the fitter

    def provenance(self) -> dict:
        return {
            "feed": self.feed.describe(),
            "gateway": {**{k: v for k, v in asdict(self.gateway).items()
                           if not k.startswith("_")},
                        "note": "swept axis; not a measurement"},
            "feed_latency_calib": self.feed_latency_calib,
            "feed_qc_status": self.feed_qc_status,
            "latency_sensitive_results_valid": bool(self.feed.is_production),
        }
