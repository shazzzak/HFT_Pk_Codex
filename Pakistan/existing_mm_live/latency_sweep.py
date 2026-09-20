# ============================================================================
# latency_sweep.py -- DOES LATENCY COST MONEY NOW THAT CROSSING IS MODELLED?
# ----------------------------------------------------------------------------
# WHY THIS EXISTS. The old sensitivity note in mm_backtest.py says "PnL is
# nearly flat 120ms -> 2000ms, so the go/no-go conclusion is latency-robust."
# That was measured on ONE day, on the CONSTANT-latency knob, and -- the part
# that matters -- on the engine that DELETED any order arriving marketable.
# In that engine, more latency meant more deletions, and a deletion was free.
# The engine now executes those orders at adverse prices, so the compensating
# error is gone and the old conclusion cannot be carried forward. This measures
# it again.
#
# WHAT THE MODEL CURRENTLY IS, so the arms are read correctly:
#   send leg = decision_ms (5) + wire_out_median_ms (40) = 45 ms for 98% of
#   messages, EXACTLY, with zero variance; the other 2% add Exp(mean 400 ms).
#   ack leg  = wire_in_median_ms (40) for 98%; the other 2% add Exp(mean 10).
#   So the body is a SPIKE, not a bell. Mean send = 53.0 ms, p99 = 321 ms.
#   `wire_out_median_ms` is a fixed value despite the name.
#
# ARMS: the latency model is the only thing that changes. Strategy params,
# clip, calibration, session and dates are identical across arms.
#
# USAGE: python latency_sweep.py --self-test | --smoke | --run [--days N]
# ============================================================================

# CLI
import argparse
# hashing for the run identity
import hashlib
# timing / ETA
import time
# log stamps
from datetime import datetime
# process pool
from multiprocessing import get_context
# Copy selected settings before sending them to another process.
from copy import deepcopy
# Read and write manifests without importing the replay stack during reporting.
from pathlib import Path
# Serialize experiment provenance and failure records.
import json
# Identify the active sweep module for source hashing.
import sys
# Import engine dependencies explicitly for their provenance hashes.
import importlib
# Print immediately, keep long stages observable, and reuse tested reporting.
from latency_sweep_support import print, Heartbeat, digest, provenance, write_report, analyse_cells, file_digest
# numerics / frames
import numpy as np
import pandas as pd

# central paths -- single source of truth
from config_pk import PARSED_ROOT, RESULTS_ROOT
# the latency model itself is what this sweep varies
from mm_backtest import LatencyModel


# log stamp
def _ts():
    # HH:MM:SS in brackets
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------- CONFIG ---
# where the outputs go
OUT_DIR = RESULTS_ROOT / "diagnostics"
# the shipped 113-name assignment, used for the name list AND each name's lean
ASSIGNMENT = RESULTS_ROOT / "config_assignment_20260915_0043.csv"
# the one-tick names the shipped book forces to plain OBI (A.2 finding)
CHEAP_EXCLUDED = {"KEL", "PIBTL", "TPL"}
# trailing window for the median trade size
TRAIL_DAYS = 10
# production clip multiple
CLIP_MULT = 3.0
# sampled dates; raise for the confirmation run
MAX_DAYS = 40
# worker processes
WORKERS = 9
# latency seeds -- changing latency changes the random PATH as well as the
# level, so more than one seed is needed before a small gap means anything
SEEDS = [0, 1, 2]

# the production defaults, spelled out so every arm is a visible delta from them
BASE = dict(decision_ms=5.0, wire_out_median_ms=40.0, wire_out_tail_ms=400.0,
            wire_in_median_ms=40.0, wire_in_tail_ms=10.0, tail_prob=0.02,
            # SHAPE: 0.0 reproduces the old point-mass body byte for byte
            wire_out_sigma=0.0, wire_in_sigma=0.0, latency_floor_ms=0.0)

# THE FLOOR: the fastest a message can ever be. Not a theory number -- it is
# the MINIMUM of a few thousand timestamped round trips to the gateway, halved.
# It is a PLACEHOLDER until that is measured, and it is set with --floor.
LAT_FLOOR_MS = 20.0

# the coefficients of variation the shape arms are named after
TARGET_CVS = (0.15, 0.30, 0.50)


# THE SIGMA TRAP, and why sigma is DERIVED rather than written down.
# `wire_out_sigma` is the lognormal sigma of the EXCESS OVER THE FLOOR, not the
# coefficient of variation of the send leg. Only (wire mean - floor) can vary,
# so the sigma that produces a given CV depends ENTIRELY on the floor:
#     floor  0 -> CV 0.30 needs sigma 0.328
#     floor 20 -> CV 0.30 needs sigma 0.613
#     floor 35 -> CV 0.30 needs sigma 1.454
# Hardcoding a sigma and then changing the floor silently changes what every
# arm means. So it is solved here, from whatever floor is actually in force.
def sigma_for_cv(cv, floor, wire=None, decision=None):
    # the mean of the whole send leg: compute time plus wire
    wire = BASE["wire_out_median_ms"] if wire is None else wire
    decision = BASE["decision_ms"] if decision is None else decision
    send = decision + wire
    # only the delay above the floor can vary
    excess = wire - floor
    # no headroom means no reachable spread; refuse rather than return nonsense
    if excess <= 0:
        raise ValueError(f"floor {floor} leaves no headroom below wire mean {wire}")
    # sd(lognormal) = excess * sqrt(exp(s^2) - 1); solve that for the target sd
    target_sd = cv * send
    # closed form, no solver needed
    return float(np.sqrt(np.log1p((target_sd / excess) ** 2)))


# build one arm's latency kwargs from the base, overriding only what it names
def arm_kwargs(**over):
    # start from production
    k = dict(BASE)
    # apply this arm's changes
    k.update(over)
    # the model takes a seed too, supplied per run
    return k


# ---- THE ARMS -------------------------------------------------------------
# 1. SYMMETRIC LEVEL SWEEP: both legs move together, which is what a change of
#    venue, colo or route actually looks like.
# 2. OUT-LEG-ONLY arms: isolate which leg binds. The docstring claims the
#    cancel (send) leg is the one that costs money; that is testable.
# 3. TAIL arms: the 2% x Exp(400) spike is a PRIOR, never measured. If P&L is
#    sensitive to it, the prior has to be replaced with telemetry before any
#    sizing decision rests on these numbers.
ARMS = {}
# symmetric sweep, both legs set to the same wire value
for _w in (10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 200.0):
    ARMS[f"lat{int(_w)}"] = arm_kwargs(wire_out_median_ms=_w, wire_in_median_ms=_w)
# send leg only -- ack held at production
for _w in (80.0, 120.0):
    ARMS[f"out{int(_w)}"] = arm_kwargs(wire_out_median_ms=_w)
# ack leg only -- send held at production
for _w in (80.0, 120.0):
    ARMS[f"ack{int(_w)}"] = arm_kwargs(wire_in_median_ms=_w)
# the tail switched off entirely: what are the 2% spikes worth?
ARMS["notail"] = arm_kwargs(tail_prob=0.0)
# a heavier tail, same body
ARMS["tail5"] = arm_kwargs(tail_prob=0.05)
# a much fatter spike, same frequency
ARMS["tail400x2"] = arm_kwargs(wire_out_tail_ms=800.0)
# SHAPE arms: the MEAN is held at production and only the spread changes, so a
# difference here is attributable to shape alone. Requires the wire_out_sigma
# support added to LatencyModel on 2026-09-19; with sigma 0 it is a no-op.
# Rebuilt whenever the floor changes, because sigma depends on it.
def add_shape_arms(floor):
    # drop any shape arms from a previous floor so they cannot linger
    for k in [k for k in ARMS if k.startswith("cv")]:
        del ARMS[k]
    # one arm per target CV, with sigma solved for THIS floor
    for cv in TARGET_CVS:
        s = sigma_for_cv(cv, floor)
        ARMS[f"cv{int(cv*100)}"] = arm_kwargs(wire_out_sigma=s, wire_in_sigma=s,
                                              latency_floor_ms=floor)


# build them once at import with the default floor
add_shape_arms(LAT_FLOOR_MS)
# the control that everything is paired against: production as shipped
CONTROL = "lat40"
# Keep the full default arm table separate from the worker's selected subset.
ALL_ARMS = deepcopy(ARMS)

# ---- STAGES ---------------------------------------------------------------
# All 17 arms x 3 seeds is ~230,000 cells, about 34 hours at the measured 111
# cells/min. That is the wrong first run. The LEVEL question is the big lever
# and answers in a few hours; shape and tail are refinements that only matter
# if level does. So the arms are grouped and a stage is chosen with --arms.
# The control is ALWAYS included, whatever the group, or nothing can be paired.
GROUPS = {
    # the first run: does average latency move P&L at all?
    "level": ["lat20", "lat40", "lat60", "lat80", "lat120"],
    # the full level ladder including the extremes
    "level_full": ["lat10", "lat20", "lat40", "lat60", "lat80", "lat120", "lat200"],
    # which leg binds -- send (cancel race) or ack?
    "legs": ["lat40", "out80", "out120", "ack80", "ack120"],
    # is the unmeasured 2% x Exp(400) prior load-bearing?
    "tail": ["lat40", "notail", "tail5", "tail400x2"],
    # does the body's SHAPE matter once the mean is held fixed?
    "shape": ["lat40", "cv15", "cv30", "cv50"],
    # everything
    "all": list(ARMS),
}


def validate_arms(arms=None, floor=None):
    # Fall back to module settings only for legacy direct calls.
    arms = ARMS if arms is None else arms
    # Display the actual shape floor selected for this invocation.
    floor = LAT_FLOOR_MS if floor is None else floor
    # the control has to be production exactly, or every paired number is
    # measured against something that does not ship
    assert arms[CONTROL] == BASE, "CONTROL arm must equal the production BASE"
    # every arm must differ from the control in at least one field, else it is
    # a duplicate spending a multiple-testing slot for nothing
    for k, v in arms.items():
        if k == CONTROL:
            continue
        assert v != BASE, f"arm {k} is identical to the control"
    # an arm that asks for spread with no headroom above the floor is a silent
    # no-op in the engine's eyes; catch it here rather than after nine hours
    for k, v in arms.items():
        if v.get("wire_out_sigma", 0.0) > 0.0:
            assert v["wire_out_median_ms"] > v.get("latency_floor_ms", 0.0), \
                f"arm {k}: wire_out_median_ms must exceed latency_floor_ms"
    # print what will run, with the EFFECTIVE one-way send latency, because
    # decision_ms is folded in and 'wire 40' is really 45 ms on the wire
    print(_ts() + f"validate_arms OK: {len(arms)} arms, control={CONTROL}")
    print(_ts() + f"  {'arm':12s} {'send body ms':>13s} {'ack body ms':>12s} "
                  f"{'tail p':>7s} {'body sd':>9s} {'body CV':>8s}")
    for k, v in arms.items():
        # the body mean of the send leg (decision + wire), before the tail
        send = v["decision_ms"] + v["wire_out_median_ms"]
        # the spread the body carries: 0 for a point mass, else the lognormal sd
        # of the excess over the floor
        sg = v.get("wire_out_sigma", 0.0)
        exc = v["wire_out_median_ms"] - v.get("latency_floor_ms", 0.0)
        sd = exc * np.sqrt(np.exp(sg * sg) - 1.0) if sg > 0 else 0.0
        print(_ts() + f"  {k:12s} {send:>13.1f} {v['wire_in_median_ms']:>12.1f} "
                      f"{v['tail_prob']:>7.3f} {sd:>8.2f}ms {sd/send:>8.3f}")
    # headroom is what makes a shape arm meaningful. When the floor is close to
    # the mean there is almost nothing left to vary, so the only way to reach a
    # given spread is an extreme sigma -- which stops being a BODY and becomes
    # a second tail. Say so rather than letting it pass silently.
    head = BASE["wire_out_median_ms"] - floor
    print(_ts() + f"  floor {floor:.1f} ms -> only {head:.1f} of the "
                  f"{BASE['decision_ms']+BASE['wire_out_median_ms']:.0f} ms send leg can vary")
    if head < 10.0:
        print(_ts() + "  WARNING: under 10 ms of headroom. The shape arms need sigma")
        print(_ts() + "  above ~1.0 to reach their target spread, which produces a")
        print(_ts() + "  spike with a very long tail, not a body. If the floor really")
        print(_ts() + "  is that close to the mean, the MEAN is the number to re-measure.")


# ------------------------------------------------- per-name lean settings ---
def per_name_setting():
    """Each name's OWN lean from the shipped assignment -- not a uniform one.

    A sweep that puts every name on one lean is not measuring the book that
    ships (the lesson extreme_obi_sweep.py records).
    """
    # the shipped assignment
    df = pd.read_csv(ASSIGNMENT)
    # the columns this needs, checked rather than assumed
    need = {"symbol", "assigned_config"}
    # fail loudly with the actual columns if the file changed shape
    if not need <= set(df.columns):
        raise SystemExit(f"{ASSIGNMENT.name}: need {sorted(need)}, "
                         f"has {sorted(df.columns)}")
    # map each name to the engine overrides its assigned config stands for
    out = {}
    # walk the assignment
    for _, r in df.iterrows():
        # the name
        s = str(r["symbol"])
        # what it ships on
        cfg = str(r["assigned_config"])
        # DROP names are not quoted at all, so they are excluded from the sweep
        if cfg == "DROP":
            continue
        # the cheap-tick three ship on plain OBI whatever the table says
        if s in CHEAP_EXCLUDED:
            out[s] = dict(queue_skew_ticks=0.0)
        # lean off
        elif cfg == "OBI":
            out[s] = dict(queue_skew_ticks=0.0)
        # lean at 0.20
        elif cfg == "QT_2t@20":
            out[s] = dict(queue_skew_ticks=2.0, queue_skew_thresh=0.20)
        # Accept the documented 0.15 label explicitly.
        elif cfg == "QT_2t@15":
            out[s] = dict(queue_skew_ticks=2.0, queue_skew_thresh=0.15)
        # Unknown labels must not silently change the shipped strategy.
        else:
            # Identify the affected symbol and assignment label.
            raise ValueError(f"Unknown assignment label for {s}: {cfg}")
    # the per-name override table
    return out


# mid at a fill, read off the engine's own equity path (the same method
# spot_capture_markout_decomp.py uses). Last equity row at or before t.
def _mid_at(eq_t, eq_mid, t):
    # an empty path cannot price anything
    if len(eq_t) == 0:
        return float("nan")
    # index of the last row with time <= t
    pos = np.searchsorted(eq_t, t, side="right") - 1
    # nothing before this fill -> no mid
    if pos < 0:
        return float("nan")
    # the mid in force when the fill happened
    return float(eq_mid[pos])


# ------------------------------------------------------------ attribution --
def _attr(dr, H):
    """Net P&L, opened notional, equity-mid capture proxy, and the CROSSING counters.

    CAPTURE IS COMPUTED HERE, NOT READ FROM fifo_attribution.
    An earlier version did `per[b].get("capture", 0.0)` -- but that function
    returns keys {realized, opened_qty, opened_notional, holds, fills} and has
    no "capture" at all, so the default silently produced a column of zeros in
    every arm. The lesson is the .get, not the key name: a default turns a
    missing input into a plausible number. Everything below either asserts or
    returns NaN.
    """
    # the day's fills
    f = dr.fills
    # the engine's own headline P&L
    net = float(dr.pnl())
    # crossing counters straight off the engine stats
    n_cross = float(dr.stats.get("crossed_on_arrival", 0))
    sh_cross = float(dr.stats.get("crossed_on_arrival_shares", 0.0))
    # no fills -> nothing more to measure
    if f is None or len(f) == 0:
        return net, 0.0, float("nan"), n_cross, sh_cross, 0.0
    # ---- LEGACY CAPTURE PROXY: the half-spread earned at the instant of each fill.
    # cap = sign * (mid_at_fill - price) * qty. A buy below the mid is positive,
    # a sell above the mid is positive. A CROSSING fill is on the WRONG side of
    # the mid, so this is the column that shows the damage.
    eq = dr.equity
    # the engine logs equity by default; if it is empty, capture is not
    # measurable and must come back NaN rather than 0.0
    if eq is None or len(eq) == 0 or "mid" not in getattr(eq, "columns", []):
        cap = float("nan")
    else:
        # the mid path, as arrays, sorted by time
        eq_t = eq["t"].to_numpy(dtype=float)
        eq_mid = eq["mid"].to_numpy(dtype=float)
        # accumulate capture over every fill
        cap = 0.0
        for r in f.itertuples(index=False):
            # the mid in force when this fill happened
            m = _mid_at(eq_t, eq_mid, float(r.t))
            # An unavailable fill mid invalidates the cell capture proxy.
            if not np.isfinite(m):
                # Preserve missingness rather than returning partial capture.
                cap = float("nan")
                # No later fill can repair this missing observation.
                break
            # +1 if we bought, -1 if we sold
            sgn = 1.0 if r.side == "BUY" else -1.0
            # the half-spread earned (or paid, when negative)
            cap += sgn * (m - float(r.px)) * float(r.qty)
    # the VALUE crossed on arrival, read from the fills' own reason tag --
    # this is the quantity that actually explains the capture damage
    if "reason" in f.columns:
        # rows the engine tagged as crossing on arrival
        cx = f[f["reason"].astype(str) == "crossed_on_arrival"]
        # their traded value
        v_cross = float((cx["px"] * cx["qty"]).sum()) if len(cx) else 0.0
        # exclude EOD liquidation fills from the FIFO input (double-counts the
        # haircut -- the documented 1-2 PKR leak)
        ff = f[~f["reason"].astype(str).str.startswith("liq")].copy()
    else:
        # no reason column -> cannot separate, use everything
        v_cross = 0.0
        ff = f.copy()
    # nothing left after the liq exclusion
    if len(ff) == 0:
        return net, 0.0, cap, n_cross, sh_cross, v_cross
    # residual lots are held to session end
    liq_t = float(ff["t"].max())
    # the validated attribution, used ONLY for the opened-notional denominator
    per = H.fifo_attribution(ff, net, liq_t)
    # assert the key exists rather than defaulting it away -- this is the exact
    # mistake that killed the capture column
    for b, v in per.items():
        assert "opened_notional" in v, f"fifo_attribution bucket {b} has no opened_notional"
    # opened notional is the bps denominator
    onot = sum(float(v["opened_notional"]) for v in per.values())
    # net, notional, capture, and the three crossing measures
    return net, onot, cap, n_cross, sh_cross, v_cross


# -------------------------------------------------------------- the worker --
# per-process globals
_R = None; _H = None; _C = None


# Install the parent's selected configuration before processing any job.
def _init(calib, options):
    # These values are private to this spawned process.
    global _R, _H, _C, ARMS, SEEDS, LAT_FLOOR_MS
    # Copy complete parameter dictionaries, not just arm names.
    ARMS = deepcopy(options['arms'])
    # Preserve the exact selected seed list.
    SEEDS = list(options['seeds'])
    # Preserve the selected shape floor as well as the derived sigmas.
    LAT_FLOOR_MS = options['floor']
    # Keep the parent-loaded calibration bundle unchanged.
    _C = calib
    # Tests can verify spawn transport without loading raw data dependencies.
    if calib is None:
        # Return only after installing the same configuration used by real jobs.
        return
    # Import replay dependencies after installing worker settings.
    import run_legacy_mm as R, mm_harness as H
    # Use the canonical parsed store for this process.
    R.PARSED_ROOT = PARSED_ROOT
    # Select the existing micro strategy.
    R.USE_MICRO = True
    # Retain the module handles for subsequent cells.
    _R, _H = R, H


# Return the settings actually visible in a worker process.
def _worker_settings(_=None):
    # Include parameter values so matching labels cannot hide a different experiment.
    return {'arms': deepcopy(ARMS), 'seeds': list(SEEDS), 'floor': LAT_FLOOR_MS}


def _one(arm, lat_kw, seed, date, sym, dsets):
    """One (arm, seed, symbol, day) cell."""
    # calibration bundle
    C = _C
    # a name without calibration cannot run
    if sym not in C["scales"] or sym not in C["profiles"]:
        return None
    # this date's session segments
    segs = C["segments"].get(str(date))
    # no segments -> skip
    if segs is None:
        return None
    # trailing median trade size
    med = _H.trailing_median(C["tstats"][sym], C["all_dates"], date, TRAIL_DAYS)
    # unusable -> skip
    if med is None or med <= 0:
        return None
    # the production clip
    clip = max(1, int(round(CLIP_MULT * med)))
    # the strategy parameters, assembled the way every runner does
    params = _H.build_micro_params(clip, C["scales"][sym], C["profiles"][sym],
                                   C["windows"].get(sym, (5.0, 1.0)), segs)
    # this NAME's own shipped lean -- not a uniform one
    params.update(C["setting"][sym])
    # THE ONLY THING THE ARM CHANGES: the latency model, with this seed.
    # log_equity is forced on because capture is computed off the equity mid
    # path; with it off the capture column would silently go NaN.
    overrides = {"latency_model": LatencyModel(seed=seed, **lat_kw),
                 "log_equity": True}
    # run the single canonical backtest path
    dr = _H.run_symbol_day(date, sym, dsets, params, cfg_overrides=overrides)
    # an unpriceable day
    if dr is None or dr.pnl() is None:
        return None
    # attribution + crossing counters
    net, onot, cap, ncx, shcx, vcx = _attr(dr, _H)
    # one tidy row
    return dict(arm=arm, seed=seed, symbol=sym, date=str(date), net_pkr=net,
                opened_notional=onot, capture_pkr=cap, crossed_orders=ncx,
                crossed_shares=shcx, crossed_value=vcx)


# Replay one complete date using the explicitly initialized experiment.
def _work(job):
    # Jobs retain the original date-level scheduling granularity.
    date = job
    # Open the historical market data once for this date.
    dsets = _R.open_datasets(date)
    # Keep successful cells and failures separate.
    rows, errors = [], []
    # Missing input data must remain visible in the final run status.
    if dsets is None:
        # Record the date-level failure rather than returning an empty success.
        errors.append({'date': str(date), 'error': 'missing datasets'})
    # Run only when all required date partitions exist.
    else:
        # Iterate the exact selected arm dictionaries received from the parent.
        for arm, lat_kw in ARMS.items():
            # Iterate only the selected seeds.
            for seed in SEEDS:
                # Evaluate every symbol in the declared universe.
                for sym in _C['universe']:
                    # Isolate failures while retaining their identities for audit.
                    try:
                        # Reuse the existing strategy and simulated exchange.
                        row = _one(arm, lat_kw, seed, date, sym, dsets)
                        # Treat an unrunnable cell as missing coverage, not zero profit.
                        if row is None:
                            # Refuse a silent exclusion from the compared portfolios.
                            raise ValueError('cell unavailable: calibration, data, or P&L missing')
                        # Retain the completed experiment cell.
                        rows.append(row)
                    # Record ordinary replay errors without hiding missing cells.
                    except Exception as exc:
                        # Preserve the complete failing cell key and reason.
                        errors.append({'arm': arm, 'seed': seed, 'symbol': sym, 'date': str(date), 'error': repr(exc)})
    # Acknowledge the actual settings used, even if the date had no usable data.
    return {'date': str(date), 'settings_sha256': digest(_worker_settings()), 'rows': rows, 'errors': errors}


# -------------------------------------------------------------- calibration -
def _calib():
    # driver + harness
    import run_legacy_mm as R, mm_harness as H
    # local store
    R.PARSED_ROOT = PARSED_ROOT
    # every tradeable date
    all_dates = R.discover_dates()
    # the shipped per-name settings, which also define the universe
    setting = per_name_setting()
    # the names actually quoted
    universe = sorted(setting)
    # the calibration bundle
    C = dict(scales=H.load_scales(), profiles=H.load_profiles(),
             windows=H.load_windows(), segments=H.load_segments(),
             all_dates=all_dates, universe=universe, setting=setting,
             tstats=H.trailing_median_trade_size(all_dates, universe, TRAIL_DAYS))
    # bundle + dates
    return C, all_dates


# ------------------------------------------------------------- the report --
# Report saved cells using market dates as the sampling units.
def _report(df):
    # Compute seed-level daily output and seed-averaged paired statistics.
    daily, dates, summary = analyse_cells(df)
    # Print corrected totals without summing simulations into fictitious P&L.
    print(summary.to_string(index=False))
    # Preserve the existing daily-table return contract.
    return daily


# ------------------------------------------------------------------- run ---
# Build a fresh selection without narrowing the module's master arm table.
def selected_options(group='level', seeds=3, floor=20.0):
    # Validate command-line settings before any expensive data preparation.
    if group not in GROUPS or seeds < 1 or not np.isfinite(floor) or floor < 0 or floor >= BASE['wire_out_median_ms']:
        # Reject floors with no headroom for the defined shape experiments.
        raise ValueError('Require a known arm group, seeds >= 1, and 0 <= shape floor < 40 ms')
    # Start from all original non-shape arms on every invocation.
    arms = {k: deepcopy(v) for k, v in ALL_ARMS.items() if not k.startswith('cv')}
    # Recompute shape parameters using the selected floor.
    for cv in TARGET_CVS:
        # Solve sigma from the desired send-body coefficient of variation.
        sigma = sigma_for_cv(cv, floor)
        # Store the complete effective latency-model arguments.
        arms[f'cv{int(cv*100)}'] = arm_kwargs(wire_out_sigma=sigma, wire_in_sigma=sigma, latency_floor_ms=floor)
    # Select only the requested stage, with the control retained.
    chosen = list(arms) if group == 'all' else GROUPS[group]
    # Return a serializable object shared by parent, workers, and manifest.
    return {'arms': {k: arms[k] for k in chosen}, 'seeds': list(range(seeds)), 'floor': float(floor)}


# Execute a new experiment with explicit spawn-safe settings and strict coverage.
def run_real(workers=WORKERS, max_days=MAX_DAYS, group='level', seeds=3, floor=20.0, output_dir=None, heartbeat_seconds=30.0):
    # Refuse nonsensical worker and sampling settings immediately.
    if workers < 1 or max_days < 0:
        # Zero days means all eligible dates; negative dates are never meaningful.
        raise ValueError('Require workers >= 1 and days >= 0')
    # Resolve the actual experiment before starting the calibration pass.
    options = selected_options(group, seeds, floor)
    # Validate and print the actual settings, including full arm parameters.
    validate_arms(options['arms'], options['floor'])
    # State the scope before a potentially slow calibration pass.
    print(_ts() + f" stage {group}: {len(options['arms'])} arms x {len(options['seeds'])} seeds")
    # Keep stdout visible during calibration and between date completions.
    with Heartbeat(heartbeat_seconds) as heartbeat:
        # Identify the longest startup stage to the operator.
        heartbeat.stage = 'calibration and trailing-size pre-pass'
        # Detect an assignment edit during the pre-pass.
        assignment_before = file_digest(ASSIGNMENT)
        # Load the same effective calibration used by the existing sweep.
        calib, all_dates = _calib()
        # Reject a run whose assignment changed during setup.
        if assignment_before != file_digest(ASSIGNMENT):
            # Avoid attributing old settings to a new assignment hash.
            raise RuntimeError('Assignment changed during calibration; restart with stable input')
        # Exclude the trailing-calibration warm-up dates.
        dates = all_dates[TRAIL_DAYS:]
        # Preserve the original evenly spaced subsampling rule for comparability.
        if max_days and len(dates) > max_days:
            # Use the same sampling stride as the existing runner.
            step = max(1, len(dates) // max_days)
            # Keep no more than the selected number of dates.
            dates = dates[::step][:max_days]
        # Refuse an empty experiment before creating output artifacts.
        if not dates or not calib['universe']:
            # Explain that no simulation work can be performed.
            raise ValueError('No eligible dates or symbols after warm-up')
        # Hash the source modules directly controlling the simulation and its inputs.
        modules = {name: importlib.import_module(name) for name in ('mm_backtest', 'micro_mm', 'mm_harness', 'run_legacy_mm', 'snapshot_prep', 'halt_state', 'config_pk', 'latency_sweep_support')}
        # Include this exact runner whether invoked as a module or a script.
        modules['latency_sweep'] = sys.modules[__name__]
        # Preserve experiment constants and effective engine/strategy defaults.
        constants = {'clip_mult': CLIP_MULT, 'trail_days': TRAIL_DAYS, 'cheap_excluded': sorted(CHEAP_EXCLUDED), 'parsed_root': str(PARSED_ROOT), 'engine_cfg': modules['run_legacy_mm'].CFG, 'micro_params': modules['run_legacy_mm'].MICRO_PARAMS}
        # Build the experiment identity from full parameters and loaded values.
        identity = provenance(options['arms'], options['seeds'], floor, dates, calib, ASSIGNMENT, modules, constants)
        # Use a parameter-sensitive identifier rather than hashing labels alone.
        tag = digest(identity)[:16]
        # Keep each attempt separate even when its experiment identity is the same.
        destination = Path(output_dir) if output_dir else OUT_DIR / f"latency_sweep_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        # Refuse to overwrite a previous attempt.
        destination.mkdir(parents=True, exist_ok=False)
        # Initialize a machine-readable run status before workers start.
        manifest = {'experiment_sha256': digest(identity), 'identity': identity, 'workers': workers, 'start_method': 'spawn', 'status': 'running'}
        # Store the initial manifest for interrupted-run diagnosis.
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
        # Count every cell that must be present for a passing comparison.
        expected = len(dates) * len(calib['universe']) * len(options['arms']) * len(options['seeds'])
        # Display the actual work count without promising a stale runtime estimate.
        print(_ts() + f" {len(calib['universe'])} names x {len(dates)} dates x {len(options['arms'])} arms x {len(options['seeds'])} seeds = {expected:,} cells")
        # State where partial evidence and the final report will be written.
        print(_ts() + f' outputs -> {destination}')
        # Accumulate completed rows and explicit failures.
        rows, errors = [], []
        # Preserve completed-date evidence if the process is interrupted later.
        try:
            # Use spawn on every platform so transport behavior is tested consistently.
            with get_context('spawn').Pool(processes=workers, initializer=_init, initargs=(calib, options)) as pool:
                # Consume dates as they complete, without blocking the heartbeat thread.
                for done, result in enumerate(pool.imap_unordered(_work, dates), 1):
                    # Verify the actual worker settings, not just the parent banner.
                    if result['settings_sha256'] != digest(options):
                        # A mismatched experiment must never enter the report.
                        raise RuntimeError('Worker settings do not match the parent experiment')
                    # Retain all successful cells from this date.
                    rows.extend(result['rows'])
                    # Retain every failed or unavailable cell.
                    errors.extend(result['errors'])
                    # Write completed-date rows without waiting for the whole experiment.
                    pd.DataFrame(result['rows']).to_csv(destination / f"cells_{done:04d}.csv", index=False)
                    # Write the current failure ledger after every date.
                    (destination / 'errors.json').write_text(json.dumps(errors, indent=2))
                    # Update the heartbeat with concrete completion counts.
                    heartbeat.stage = f"replay: {done}/{len(dates)} dates, {len(rows):,}/{expected:,} cells, {len(errors)} failures"
                    # Emit immediate progress in addition to the periodic heartbeat.
                    print(_ts() + ' ' + heartbeat.stage)
            # Retain the detailed output even when the experiment is incomplete.
            df = pd.DataFrame(rows)
            # Save completed cells before enforcing the pass condition.
            df.to_csv(destination / 'cells.csv', index=False)
            # Refuse to present incomplete portfolios as a passing comparison.
            if errors or len(rows) != expected:
                # Direct the user to the retained failure evidence.
                raise RuntimeError(f'Incomplete sweep: {len(rows)}/{expected} cells; inspect errors.json')
            # Identify the reporting phase for the heartbeat.
            heartbeat.stage = 'validating coverage and writing paired statistics'
            # Save corrected statistics without rerunning simulation or changing fills.
            write_report(df, destination / 'report', source=destination / 'cells.csv')
            # Mark the run complete only after reporting succeeds.
            manifest['status'] = 'complete'
        # Record interrupted and failed attempts, including their reason.
        except BaseException as exc:
            # Preserve a failed status instead of leaving a false success marker.
            manifest['status'] = 'failed'
            # Save the exception for post-run inspection.
            manifest['error'] = repr(exc)
            # Propagate failure to the terminal exit code.
            raise
        # Finalize run status regardless of success or failure.
        finally:
            # Record the achieved coverage alongside the planned workload.
            manifest['completed_cells'] = len(rows)
            # Record the expected coverage used by the pass condition.
            manifest['expected_cells'] = expected
            # Persist the final status without removing partial artifacts.
            (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))


# one symbol-day per arm, to check wiring on real data
def smoke():
    # arms first
    validate_arms()
    # driver + harness
    import run_legacy_mm as R, mm_harness as H
    # local store
    R.PARSED_ROOT = PARSED_ROOT
    # calibration
    calib, all_dates = _calib()
    # set the worker globals for a direct call
    global _R, _H, _C; _R, _H, _C = R, H, calib
    # a mid-sample date
    date = all_dates[len(all_dates) // 2]
    # that day's datasets
    dsets = R.open_datasets(date)
    # NOT universe[0] -- that is AGHA, and picking the alphabetically first name
    # is the "sample of one letter" mistake extreme_obi_sweep.py already
    # recorded. A latency arm can only show up on a name with enough activity
    # for the flight time to matter, so smoke on the BUSIEST calibrated names.
    liquid = [s for s in ("OGDC", "PPL", "LUCK", "HBL", "UBL", "ENGROH", "PSO")
              if s in calib["universe"]]
    # fall back to the universe's first name only if none of those are in it
    sym = liquid[0] if liquid else calib["universe"][0]
    print(_ts() + f"[smoke] {sym} {date} -- a LIQUID name on purpose")
    # collect each arm's headline so identical arms can be detected
    seen = {}
    # run every arm on that one symbol-day, seed 0
    for arm, kw in ARMS.items():
        t0 = time.perf_counter()
        r = _one(arm, kw, 0, date, sym, dsets)
        print(_ts() + f"[smoke] {arm:12s} {time.perf_counter()-t0:5.1f}s  {r}")
        # remember the P&L so duplicates across arms are visible
        if r is not None:
            seen.setdefault(round(r["net_pkr"], 6), []).append(arm)
    # ONE SYMBOL-DAY IS NOT A TEST. If most arms return the same number, the
    # cell was too quiet to exercise the mechanism -- that is information about
    # the smoke, not about latency, and it must not read as "latency does
    # nothing".
    big = max((len(v) for v in seen.values()), default=0)
    if big > len(ARMS) // 2:
        print(_ts() + f"\n  NOTE: {big} of {len(ARMS)} arms returned an IDENTICAL P&L.")
        print(_ts() + "  On one quiet symbol-day that is expected: if no price moved")
        print(_ts() + "  inside the flight window and no quote was blocked by an ack,")
        print(_ts() + "  latency changes nothing. It does NOT mean the arms are inert.")
        print(_ts() + "  The run over 113 names x N days is the test; this only checks wiring.")


# offline checks: no data, no project deps beyond the model itself
def self_test():
    # the arm table must be coherent
    validate_arms()
    # the model must actually produce what the header claims
    m = LatencyModel(seed=0)
    # ten thousand draws
    d = np.array([m.draw_out() for _ in range(20000)])
    # 98% must land on exactly decision+wire
    frac = (d == 45.0).mean()
    # allow sampling error around 0.98
    assert 0.97 < frac < 0.99, f"expected ~98% at 45.0 ms, got {frac:.3f}"
    # the mean must match 45 + tail_prob*tail_ms
    assert abs(d.mean() - 53.0) < 4.0, f"expected mean ~53 ms, got {d.mean():.1f}"
    # an arm that raises the wire must raise every draw by the same amount
    m2 = LatencyModel(seed=0, **ARMS["lat80"])
    d2 = np.array([m2.draw_out() for _ in range(20000)])
    # the body shifts by exactly 40 ms
    assert abs(np.median(d2) - np.median(d) - 40.0) < 1e-9, "body did not shift by 40 ms"
    # the notail arm must be perfectly constant
    m3 = LatencyModel(seed=0, **ARMS["notail"])
    d3 = np.array([m3.draw_out() for _ in range(5000)])
    assert d3.std() == 0.0, "notail arm still has variance"
    # a SHAPE arm must add spread WITHOUT moving the mean -- that is the whole
    # point of solving mu, and it is what makes shape separable from level
    if "cv30" in ARMS:
        m4 = LatencyModel(seed=0, **dict(ARMS["cv30"], tail_prob=0.0))
        d4 = np.array([m4.draw_out() for _ in range(200000)])
        # the mean must be unchanged to within sampling error
        assert abs(d4.mean() - 45.0) < 0.5, f"shape arm moved the mean: {d4.mean():.2f}"
        # and it must actually have spread, or the arm tests nothing
        assert 10.0 < d4.std() < 18.0, f"cv30 sd out of range: {d4.std():.2f}"
        # nothing may land below the physical floor
        assert d4.min() >= 5.0 + LAT_FLOOR_MS - 1e-9, "a draw fell below the floor"
        print(_ts() + f"[self-test] shape arm cv30: mean {d4.mean():.2f} ms, "
                      f"sd {d4.std():.2f} ms, min {d4.min():.2f} ms")
    # done
    print(_ts() + "[self-test] latency model behaves as documented.")
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


# Parse CLI options only in the parent process.
if __name__ == '__main__':
    # Build one command-line interface for replay and saved-result analysis.
    ap = argparse.ArgumentParser()
    # Prevent combining a report with a new replay accidentally.
    mode = ap.add_mutually_exclusive_group()
    # Run lightweight latency-model checks without loading historical market data.
    mode.add_argument('--self-test', action='store_true')
    # Replay one liquid symbol-day using the selected arm group.
    mode.add_argument('--smoke', action='store_true')
    # Start a new full historical sweep only when requested explicitly.
    mode.add_argument('--run', action='store_true')
    # Reanalyse an existing cell-level CSV without another exchange replay.
    mode.add_argument('--report', type=Path)
    # Preview selected settings and verify a spawned worker without market data.
    mode.add_argument('--dry-run', action='store_true')
    # Retain the existing sampled-date count option; zero means all eligible dates.
    ap.add_argument('--days', type=int, default=MAX_DAYS)
    # Retain explicit worker-count control.
    ap.add_argument('--workers', type=int, default=WORKERS)
    # Select a named family of latency experiments.
    ap.add_argument('--arms', default='level', choices=sorted(GROUPS))
    # Select the number of Monte Carlo seeds used on each date.
    ap.add_argument('--seeds', type=int, default=3)
    # Set the floor used only by the shape experiments.
    ap.add_argument('--floor', type=float, default=20.0)
    # Choose a new output directory rather than overwriting previous results.
    ap.add_argument('--output-dir', type=Path)
    # Bound the interval between parent-process progress messages.
    ap.add_argument('--heartbeat-seconds', type=float, default=30.0)
    # Generate a chart when reanalysing saved results.
    ap.add_argument('--plot', action='store_true')
    # Parse the current command line.
    a = ap.parse_args()
    # Reanalysis derives arms and seeds from the CSV, never CLI defaults.
    if a.report:
        # Generate a unique output path unless the user selected one.
        destination = a.output_dir or OUT_DIR / f"latency_report_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        # Read existing observations and produce corrected statistics only.
        write_report(pd.read_csv(a.report), destination, source=a.report, plot=a.plot)
    # Verify worker transport before committing to an expensive run.
    elif a.dry_run:
        # Resolve complete effective settings locally.
        options = selected_options(a.arms, a.seeds, a.floor)
        # Print selected arm and latency parameters.
        validate_arms(options['arms'], options['floor'])
        # Use the real initializer under the same explicit spawn context as production runs.
        with get_context('spawn').Pool(1, initializer=_init, initargs=(None, options)) as pool:
            # Ask the worker which settings it actually received.
            received = pool.apply(_worker_settings)
        # Fail if any parameter, seed, or floor changed in transit.
        if digest(received) != digest(options):
            # Prevent a parent-only configuration from appearing to pass.
            raise RuntimeError('Spawned worker settings mismatch')
        # Confirm the exact tested worker configuration.
        print(f"Spawn check passed: {len(received['arms'])} arms; seeds={received['seeds']}; shape floor={received['floor']} ms")
    # Historical replay is opt-in and receives every CLI setting explicitly.
    elif a.run:
        # Start the requested sweep with no hidden inherited globals.
        run_real(a.workers, a.days, a.arms, a.seeds, a.floor, a.output_dir, a.heartbeat_seconds)
    # Smoke uses the same selected arm parameters as a full run.
    elif a.smoke:
        # Resolve the requested subset before the one-day replay.
        options = selected_options(a.arms, a.seeds, a.floor)
        # Install the same settings on the direct-call smoke path.
        _init(None, options)
        # Reuse the existing liquid-symbol wiring check.
        smoke()
    # No run mode defaults to offline model checks.
    else:
        # Install all arms because model self-tests exercise level, tail, and shape.
        _init(None, selected_options('all', a.seeds, a.floor))
        # Run the retained offline latency-model assertions.
        self_test()
