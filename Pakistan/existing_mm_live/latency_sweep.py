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
#   So the body is a SPIKE, not a bell. Mean send = 52.9 ms, p99 = 321 ms.
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
from multiprocessing import Pool
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


def validate_arms():
    # the control has to be production exactly, or every paired number is
    # measured against something that does not ship
    assert ARMS[CONTROL] == BASE, "CONTROL arm must equal the production BASE"
    # every arm must differ from the control in at least one field, else it is
    # a duplicate spending a multiple-testing slot for nothing
    for k, v in ARMS.items():
        if k == CONTROL:
            continue
        assert v != BASE, f"arm {k} is identical to the control"
    # an arm that asks for spread with no headroom above the floor is a silent
    # no-op in the engine's eyes; catch it here rather than after nine hours
    for k, v in ARMS.items():
        if v.get("wire_out_sigma", 0.0) > 0.0:
            assert v["wire_out_median_ms"] > v.get("latency_floor_ms", 0.0), \
                f"arm {k}: wire_out_median_ms must exceed latency_floor_ms"
    # print what will run, with the EFFECTIVE one-way send latency, because
    # decision_ms is folded in and 'wire 40' is really 45 ms on the wire
    print(_ts() + f"validate_arms OK: {len(ARMS)} arms, control={CONTROL}")
    print(_ts() + f"  {'arm':12s} {'send ms mean':>13s} {'ack ms mean':>12s} "
                  f"{'tail p':>7s} {'send sd':>9s} {'send CV':>8s}")
    for k, v in ARMS.items():
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
    head = BASE["wire_out_median_ms"] - LAT_FLOOR_MS
    print(_ts() + f"  floor {LAT_FLOOR_MS:.1f} ms -> only {head:.1f} of the "
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
        # lean at 0.15 (the default bucket)
        else:
            out[s] = dict(queue_skew_ticks=2.0, queue_skew_thresh=0.15)
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
    """Net P&L, opened notional, TRUE capture, and the CROSSING counters.

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
    # ---- TRUE CAPTURE: the half-spread earned at the instant of each fill.
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
            # no mid -> this fill contributes nothing measurable
            if m != m:
                continue
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


def _init(calib):
    # module handles set once per worker
    global _R, _H, _C
    # driver + harness
    import run_legacy_mm as R, mm_harness as H
    # local parsed store
    R.PARSED_ROOT = PARSED_ROOT
    # micro strategy path
    R.USE_MICRO = True
    # stash
    _R = R; _H = H; _C = calib


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


def _work(job):
    """One date, all arms, all seeds, all symbols."""
    # unpack
    date = job
    # open the day's datasets once and reuse across every arm
    dsets = _R.open_datasets(date)
    # a missing day contributes nothing
    if dsets is None:
        return []
    # accumulated rows
    rows = []
    # each arm
    for arm, lat_kw in ARMS.items():
        # each seed
        for seed in SEEDS:
            # each name in the sweep universe
            for sym in _C["universe"]:
                #print symbol and date
                print(_ts() + f" {arm=}, lat_kw_length={len(lat_kw)}, {seed=}, {sym=}, {date=}")
                # a data error on one cell must not kill the run
                try:
                    r = _one(arm, lat_kw, seed, date, sym, dsets)
                except Exception as e:
                    print(_ts() + f"SKIP {arm}/s{seed} {date} {sym}: {e!r}")
                    continue
                # keep populated rows
                if r is not None:
                    rows.append(r)
    # this date's rows
    return rows


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
def _report(df):
    """Paired, day-as-unit, against the production control."""
    # daily portfolio totals per arm (seeds pooled into the same day)
    g = (df.groupby(["arm", "seed", "date"], as_index=False)
           .agg(pkr=("net_pkr", "sum"), opn=("opened_notional", "sum"),
                cap=("capture_pkr", "sum"), cx=("crossed_orders", "sum"),
                cxv=("crossed_value", "sum")))
    # daily net bps -- a rate, not a level
    g["bps"] = np.where(g["opn"] > 0, g["pkr"] / g["opn"] * 1e4, np.nan)
    # daily capture bps, the mechanism column
    g["cap_bps"] = np.where(g["opn"] > 0, g["cap"] / g["opn"] * 1e4, np.nan)
    # crossed value as a share of traded value, the frequency column
    g["cx_pct"] = np.where(g["opn"] > 0, g["cxv"] / g["opn"] * 100.0, np.nan)
    # A DEAD COLUMN IS WORSE THAN A MISSING ONE. If capture never varies, say
    # so instead of printing a tidy column of zeros that looks like a finding.
    cap_all = df["capture_pkr"]
    if cap_all.isna().all():
        print(_ts() + "\n  !! capture_pkr is ALL NaN -- the equity path was empty.")
        print(_ts() + "     The mechanism column is not measured. Fix before reading P&L.")
    elif float(cap_all.abs().max()) == 0.0:
        print(_ts() + "\n  !! capture_pkr is ALL ZERO -- that is a bug, not a result.")
        print(_ts() + "     Do not read the capture column in this run.")
    # the per-arm summary
    print(_ts() + "\n===== LATENCY SWEEP: level, mechanism, and frequency =====")
    print(_ts() + f"  {'arm':12s} {'net bps':>9s} {'capture bps':>12s} "
                  f"{'crossed % of traded value':>26s} {'net PKR':>14s}")
    # one line per arm, seeds pooled
    for arm in ARMS:
        a = g[g["arm"] == arm]
        if len(a) == 0:
            continue
        print(_ts() + f"  {arm:12s} {a['bps'].mean():>9.3f} {a['cap_bps'].mean():>12.3f} "
                      f"{a['cx_pct'].mean():>26.3f} {a['pkr'].sum():>14,.0f}")
    # the paired comparison: same day, same seed, arm minus control
    print(_ts() + "\n===== PAIRED vs the production control (same day, same seed) =====")
    print(_ts() + f"  {'arm':12s} {'d net bps':>10s} {'se':>7s} {'t':>8s} "
                  f"{'d capture bps':>14s} {'n pairs':>8s}")
    # the control's own series, keyed by (seed, date)
    ctl = g[g["arm"] == CONTROL].set_index(["seed", "date"])
    # every other arm
    for arm in ARMS:
        if arm == CONTROL:
            continue
        # this arm's series on the same key
        a = g[g["arm"] == arm].set_index(["seed", "date"])
        # the days both produced
        idx = a.index.intersection(ctl.index)
        # not enough to test
        if len(idx) < 3:
            continue
        # paired differences
        d = (a.loc[idx, "bps"] - ctl.loc[idx, "bps"]).dropna()
        dc = (a.loc[idx, "cap_bps"] - ctl.loc[idx, "cap_bps"]).dropna()
        # mean, standard error, t
        m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); t = m / se if se > 0 else np.nan
        print(_ts() + f"  {arm:12s} {m:>10.3f} {se:>7.3f} {t:>8.2f} "
                      f"{dc.mean():>14.3f} {len(d):>8d}")
    # the power statement, printed BESIDE the result rather than argued after
    sd = (g[g["arm"] == CONTROL]["bps"]).std(ddof=1)
    n = len(ctl)
    print(_ts() + f"\n  control daily bps sd = {sd:.3f}; with n={n} pairs the smallest")
    print(_ts() + f"  effect this run can resolve at |t|=2 is "
                  f"{2*sd/np.sqrt(max(n,1)):.3f} bps/day. A 'flat' reading below")
    print(_ts() + "  that number means 'not measured', not 'not there'.")
    # what to conclude, written before the numbers are seen
    print(_ts() + "\n  PRE-REGISTERED READING: if net bps falls monotonically as latency")
    print(_ts() + "  rises AND capture bps falls with it AND crossed % rises with it,")
    print(_ts() + "  the mechanism is confirmed and the old 'latency-robust' note is")
    print(_ts() + "  dead. If net moves but crossing does not, something else is doing")
    print(_ts() + "  the work and this sweep has not found it.")
    # the tidy daily frame, for plotting
    return g


# ------------------------------------------------------------------- run ---
def run_real(workers=WORKERS, max_days=MAX_DAYS, group="level", seeds=None,
             floor=None):
    # narrow to the chosen stage, always keeping the control so pairing works
    global ARMS, SEEDS, LAT_FLOOR_MS
    # a different floor changes what every shape arm means, so rebuild them
    if floor is not None and floor != LAT_FLOOR_MS:
        LAT_FLOOR_MS = float(floor)
        add_shape_arms(LAT_FLOOR_MS)
        print(_ts() + f"  floor set to {LAT_FLOOR_MS:.1f} ms -> shape arms rebuilt")
    if group != "all":
        keep = GROUPS[group]
        # the control is mandatory in every stage
        if CONTROL not in keep:
            keep = [CONTROL] + keep
        # rebuild the arm table in the group's order
        ARMS = {k: ARMS[k] for k in keep}
    # seeds can be cut for a first pass and raised for the confirmation
    if seeds:
        SEEDS = list(range(seeds))
    # fail before spending hours
    validate_arms()
    # say what this will cost BEFORE it starts, using the measured throughput
    # of the 113-name run (66,783 cells in 600.5 min at 9 workers)
    print(_ts() + f"  stage '{group}': {len(ARMS)} arms x {len(SEEDS)} seeds")
    # calibration pre-pass
    print(_ts() + "pre-pass: calibration")
    calib, all_dates = _calib()
    # drop the trailing-median warm-up, then subsample
    dates = all_dates[TRAIL_DAYS:]
    # even subsample
    if max_days and len(dates) > max_days:
        step = max(1, len(dates) // max_days); dates = dates[::step][:max_days]
    # the run identity: arms + seeds + names + dates + engine, so a resumed or
    # re-plotted result can never be a union of two different definitions
    import mm_backtest as _mb, micro_mm as _mm, inspect
    eng = hashlib.sha1(
        (inspect.getsource(_mb) + inspect.getsource(_mm)).encode()).hexdigest()[:12]
    tag = hashlib.sha1(repr((sorted(ARMS), SEEDS, calib["universe"],
                             dates, eng)).encode()).hexdigest()[:8]
    # announce the plan
    print(_ts() + f"  engine {eng} | tag {tag}")
    # total cells and an up-front time estimate, so a 34-hour run announces
    # itself rather than being discovered at hour three
    cells = len(calib["universe"]) * len(dates) * len(ARMS) * len(SEEDS)
    # measured throughput of the 113-name run: 66,783 cells in 600.5 minutes
    hours = cells / 111.2 / 60.0
    print(_ts() + f"  {len(calib['universe'])} names x {len(dates)} dates x "
                  f"{len(ARMS)} arms x {len(SEEDS)} seeds = {cells:,} cells")
    print(_ts() + f"  ESTIMATED {hours:.1f} hours at the measured 111 cells/min "
                  f"({workers} workers). Ctrl-C now if that is wrong.")
    # collect + clock
    rows = []; t0 = time.perf_counter()
    # pool over dates
    with Pool(processes=workers, initializer=_init, initargs=(calib,)) as pool:
        done = 0
        for res in pool.imap_unordered(_work, dates):
            rows.extend(res); done += 1
            el = (time.perf_counter() - t0) / 60.0
            print(_ts() + f"  date {done}/{len(dates)} ({el:.1f} min, "
                          f"ETA {el/done*(len(dates)-done):.1f} min)")
    # nothing produced
    if not rows:
        print(_ts() + "no rows."); return
    # the per-cell frame
    df = pd.DataFrame(rows)
    # ensure the output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # write the cell-level artifact with the tag in its name
    df.to_csv(OUT_DIR / f"latency_sweep_{tag}.csv", index=False)
    # the paired report, and the daily frame it built
    g = _report(df)
    # the daily series too, for plotting
    g.to_csv(OUT_DIR / f"latency_sweep_daily_{tag}.csv", index=False)
    # where it went
    print(_ts() + f"\n  outputs -> {OUT_DIR} (tag {tag})")


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


# entry point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    # which stage to run; see GROUPS. Default is the level question.
    ap.add_argument("--arms", default="level", choices=sorted(GROUPS))
    # number of latency seeds; 1 for a first pass, 3 to confirm
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    # the measured minimum one-way latency. Changing it re-solves every sigma.
    ap.add_argument("--floor", type=float, default=LAT_FLOOR_MS)
    a = ap.parse_args()
    # apply the floor before anything prints an arm table
    if a.floor != LAT_FLOOR_MS:
        LAT_FLOOR_MS = a.floor
        add_shape_arms(LAT_FLOOR_MS)
    if a.smoke:
        smoke()
    elif a.self_test or not a.run:
        self_test()
    if a.run:
        run_real(workers=a.workers, max_days=a.days, group=a.arms, seeds=a.seeds,
                 floor=a.floor)

# ============================================================================
# THE ENGINE CHANGE THIS SWEEP CANNOT MAKE FOR YOU
# ----------------------------------------------------------------------------
# The body of the distribution has ZERO variance: 98% of messages take exactly
# 45.0 ms. Real network latency has a body with spread -- a lognormal or gamma
# shape -- and the spread matters here specifically, because whether an order
# arrives marketable depends on how far the price moved during ITS flight, not
# during the average flight. A constant body understates the frequency of both
# the early and the late arrivals.
#
# To test that, LatencyModel needs one more parameter. In mm_backtest.py,
# LatencyModel.draw_out currently reads:
#
#     base = self.decision_ms + self.wire_out_median_ms
#     if self.rng.random() < self.tail_prob:
#         base += self.rng.exponential(self.wire_out_tail_ms)
#     return base
#
# The production version draws the body as well:
#
#     base = self.decision_ms + self.rng.lognormal(
#         mean=np.log(self.wire_out_median_ms), sigma=self.wire_out_sigma)
#     if self.rng.random() < self.tail_prob:
#         base += self.rng.exponential(self.wire_out_tail_ms)
#     return base
#
# with wire_out_sigma=0.0 reproducing today's behaviour exactly, so the change
# is byte-identical until an arm turns it on. Do NOT fit sigma from a guess --
# it comes from FIX gateway timestamps once the live feed is running.
# ============================================================================
