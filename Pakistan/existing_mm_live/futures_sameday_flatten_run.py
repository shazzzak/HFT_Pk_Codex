# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# futures_sameday_flatten_run.py -- FUTURES MM with SAME-DAY FLATTEN + residual
# spot hedge. Tests SZ's liquidity thesis: a liquid instrument lets you complete
# round trips intraday (flatten at the close) -> near-zero drift -> clean spread;
# an illiquid one forces carry -> drift dominates. This runs the top-liquidity
# futures names and, each day, tries to POV-UNWIND the whole position into the
# close (production EOD trigger enabled EVERY day, not just at the roll). Then:
#   * FLATTEN HAIRCUT = equity_liquidated - equity_mid_mark: the cost of exiting
#     into the (thin) futures book at the close. THIS IS THE LIQUIDITY COST,
#     measured directly -- exactly the number SZ's thesis is about.
#   * RESIDUAL: whatever the POV unwind could NOT absorb (engine's unfilled_sh)
#     is carried overnight and hedged with the spot mechanism we built (walk the
#     real spot book). If the futures name is liquid enough, unfilled ~ 0 and the
#     hedge is rarely used -> drift ~ 0. If it's thin, unfilled is large and we
#     fall back to carry+hedge -> drift returns. So "how often we fail to flatten"
#     is itself the illiquidity metric.
#   * MARKOUT (to same-day close): adverse selection on the fills -- toxicity
#     that flattening CANNOT fix (it happens at the fill). If markout buries the
#     capture even when drift is removed, the names are too toxic for MM.
#
# Decomposition (per span, reconciles to cash by construction, naked = residual):
#   quoting + drift + flatten_haircut + basis + naked_gap - hedge_cost = pnl_total
# with a capture/markout lens on quoting and same-day-flatten rate reported.
#
# SMOKE default: TRG + MLCF + BOP, one span each, per-day ledger.
# Run from existing_mm_live/:  caffeinate -is python3 futures_sameday_flatten_run.py

# filesystem paths
from pathlib import Path
# FIFO queue for holding-time tracking
from collections import deque
# timing + stamp
import time
# standard library datetime for the run stamp
from datetime import datetime
# frames + arrays
import pandas as pd
# numpy for vectorised price/qty math
import numpy as np
# driver + engine + strategy modules
import run_legacy_mm as R
# the backtest engine module (also holds the patchable fee constant)
import mm_backtest as MB
# the strategy module (its viability gate imports the fee by value)
import micro_mm as MM
# the engine class and the latency model
from mm_backtest import Backtester, LatencyModel
# the microstructure market-making strategy class
from micro_mm import MicrostructureMM
# heartbeat formatter
import confirm_micro_vs_naive as C
# reuse the futures runner's shared pieces + the carry runner's spot book walk
import futures_mm_run as F
# the carry runner -- reuse its spot book walk + capture/markout lens
import futures_carry_hedge_run as CH

# ---- PATHS: from config_pk, never a literal in this file -------------------
# The hardcoded paths that used to live here pointed at the OLD store location
# and every query raised IOException once the data moved under
# "~/HFT Data/Pakistan/". A path literal in a script is a defect: it is valid
# syntax, so nothing warns you, and it fails hours into a run instead of at the
# top. config_pk is the single source of truth.
try:
    # the project's central path module
    import config_pk
    # accept either spelling the module may expose for the parsed store
    _P = getattr(config_pk, "PARSED_ROOT", None) or getattr(config_pk, "PARSED", None)
    # and either spelling for the results root
    _Rr = getattr(config_pk, "RESULTS_ROOT", None) or getattr(config_pk, "RESULTS", None)
except Exception:
    # config_pk not importable from this working directory
    _P = _Rr = None
# the parsed store, as a Path (run_legacy_mm treats it as one)
# Resolve this filesystem path through the canonical checkout/data configuration.
PARSED_ROOT = Path(_P) if _P else Path(str(_hft_paths.PARSED_ROOT))
# where result CSVs are written
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS = Path(_Rr) if _Rr else Path(str(_hft_paths.RESULTS_ROOT))
# push the parsed store onto the driver module, as every other script does
R.PARSED_ROOT = PARSED_ROOT
# fail here, with the path named, rather than inside a DuckDB glob later
if not PARSED_ROOT.is_dir():
    raise SystemExit(f"PARSED store not found: {PARSED_ROOT}\n"
                     f"  set PARSED_ROOT in config_pk.py to the real location.")
# the results folder must exist before the run, not after hours of compute
if not RESULTS.is_dir():
    raise SystemExit(f"RESULTS folder not found: {RESULTS}")
# say which stores this run used, so the log is self-describing
print(f"parsed : {PARSED_ROOT}")
print(f"results: {RESULTS}")

# ---- capture the SPOT fee at IMPORT, before anything patches it ------------
# main() runs once per arm and patches MB.FEE_TOTAL_PCT to the futures fee on
# every call. Reading the spot fee inside main() would therefore pick up the
# already-patched FUTURES fee from the second arm onward and charge the spot
# hedge legs ~8x too little -- silently, with no error anywhere.
SPOT_FEE_PER_SIDE = MB.FEE_TOTAL_PCT

# ------------------------------ experiment knobs ------------------------------
# SMOKE: named roots, one span each, per-day ledger. Flip False for full run.
SMOKE = True
# ONE_MONTH: verification mode -- run only the first contract span per name so
# the per-side buy/sell markout figures can be eyeballed before the full run.
# Set False for the full multi-month run.
ONE_MONTH = True
# the top-liquidity futures names (TRG 10.5, MLCF 6.8, BOP 6.0 trades/min) --
# the fairest test of "can the MOST liquid PSX futures support MM"
SMOKE_ROOTS = ["TRG", "MLCF", "BOP"]
# full-mode universe (reuse the runner's liquid list)
ROOTS = F.ROOTS
# clip in lots (1 lot = 500 shares)
CLIP_LOTS = 1
# shares per futures lot (fixed at 500 on PSX DFC)
LOT = 500
# standing inventory cap in clips (production 10; the daily flatten means it
# rarely binds, but it caps intraday accumulation)
MAXINV_CLIPS = 10.0
# soft-inventory band in clips (skew leans back to flat beyond this)
SOFTINV_CLIPS = 3.0
# max participation-of-volume rate for the POV unwind
MAX_POV = 0.10
# walk-forward calibration window
CAL_DAYS = 20
# minimum trades for a contract-day to be quoted
MIN_TRADES_DAY = 200
# minimum days in a span to trade it
MIN_SPAN_DAYS = 5
# morning unhedge delay (only used when a residual carried overnight)
UNHEDGE_DELAY_MIN = 5.0
# engine defaults; EOD trigger ON so the position POV-unwinds into EVERY close
GAMMA = 0.15
# ---- THE ARMS -------------------------------------------------------------
# EVERY futures run to date used the PLAIN base below. The three mechanisms that
# carry the spot edge are all OFF BY DEFAULT in micro_mm (queue_skew_ticks=0.0
# line 235, obi_defensive=False line 292, obi_throttle=False line 326) and none
# of the three futures runners passed them. So the August result compared a
# plain futures quoter against a fully-tuned spot one -- never like-for-like.
#
# PLAIN: exactly what was run before. Reproduces the existing baseline.
MID_BASE_PLAIN = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                      enable_eod_trigger=True, enable_lock_trigger=True)
# 1. DEFENSIVE WIDEN: step the exposed side 1 tick back when the book leans
# against us. Frozen ON in every spot sweep (obi_defensive=True paid at exit_ticks=1).
SPOT_DEFENSIVE = dict(obi_defensive=True, obi_defensive_thresh=0.15,
                      obi_defensive_ticks=1.0)
# 2. SIZE THROTTLE: halve the clip for 300 ms past the same threshold.
# Catalogued as the confirmed spot edge (+0.66 bps/day vs no throttle).
SPOT_THROTTLE = dict(obi_throttle=True, obi_throttle_thresh=0.15,
                     throttle_frac=0.5, throttle_hold_ms=300.0)
# 3. THE LEAN: shift BOTH quotes 2 ticks toward the heavy side once the book is
# lopsided past 0.15. Worth +1.18 bps on the spot book (2.13 -> 3.32).
# SAFE HERE: probe_futures_mm.py measured the BOP futures book at a 5.0-tick
# median spread, 87.9% of the session >= 3 ticks -- ample room for a 2-tick
# shift to stay inside the opposite touch. On a 1-2 tick book this INVERTS
# capture (A.2 on KEL/PIBTL/TPL: +1.64 -> -1.79 bps), so re-check any root
# whose width probe comes back thin before trusting its number.
SPOT_LEAN = dict(queue_skew_ticks=2.0, queue_skew_thresh=0.15)
# The arms run in this order, each a strict superset of the one before, so the
# per-arm delta ATTRIBUTES the change to a single mechanism.
ARMS = [
    # the existing baseline
    ("plain", {}),
    # + defensive widen only
    ("defensive", dict(SPOT_DEFENSIVE)),
    # + size throttle on top
    ("def_throttle", {**SPOT_DEFENSIVE, **SPOT_THROTTLE}),
    # + the lean = the full spot production recipe
    ("full_spot", {**SPOT_DEFENSIVE, **SPOT_THROTTLE, **SPOT_LEAN}),
]
# the arm currently running; the entry point rebinds both of these per arm
ARM_NAME = "plain"
# the params main() actually hands the strategy
MID_BASE = dict(MID_BASE_PLAIN)
# futures fee (reuse) + spot fee captured at runtime before the patch
FUT_FEE_PER_SIDE = F.FUT_FEE_PER_SIDE
# ------------------------------------------------------------------------------


def main():
    # run stamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # SPOT fee comes from the MODULE-LEVEL constant captured at import time.
    # It must NOT be re-read from MB.FEE_TOTAL_PCT here: main() now runs once
    # per arm, and by the second call MB.FEE_TOTAL_PCT is already the patched
    # FUTURES fee -- so re-reading it would charge the spot hedge legs the
    # futures fee on every arm after the first, silently and with no error.
    # patch futures fee into both modules
    MB.FEE_TOTAL_PCT = FUT_FEE_PER_SIDE
    # and into the strategy module (imported the fee by value at import time)
    MM.FEE_TOTAL_PCT = FUT_FEE_PER_SIDE
    # announce the two-fee structure (futures quotes vs spot residual hedge)
    print(f"fees: futures {FUT_FEE_PER_SIDE*1e4:.4f} bps/side (quotes), "
          # second line of the fee banner
          f"spot {SPOT_FEE_PER_SIDE*1e4:.4f} bps/side (residual hedge)")

    # session segments + dates
    segments = F.load_segments()
    # all trading dates discovered in the parsed store
    all_dates = R.discover_dates()
    # the same dates as strings (roll-map keys are string dates)
    date_strs = [str(d) for d in all_dates]

    # ---- pre-pass 1: futures calendar + roll map ----
    print("pre-pass 1: futures calendar + roll map", flush=True)
    # one-shot DuckDB pass: per (root, contract, date) futures volume
    cal = F.load_futures_calendar()
    # the causal roll map: active contract per (root, date) by trailing volume
    roll = F.build_roll_map(cal, all_dates)

    # ---- pre-pass 2: walk-forward calibration (reuse the inline logic) ----
    print(f"pre-pass 2: calibration on first {CAL_DAYS} days", flush=True)
    # the first CAL_DAYS dates are calibration-only (never traded)
    cal_dates = date_strs[:CAL_DAYS]
    # smoke mode calibrates just the named roots; full mode the whole universe
    roots = SMOKE_ROOTS if SMOKE else ROOTS
    # per-root accumulators for spread, sigma, fair, and the 4 volume buckets
    calib = {r: {"spr": [], "sig": [], "fair": [],
                 "f": [], "m": [], "p": [], "l": []} for r in roots}
    # start the calibration timer (heartbeat)
    t0 = time.perf_counter()
    # walk each calibration date
    for i, date in enumerate(cal_dates, 1):
        dsets = R.open_datasets(date)
        # skip dates with no data or no session segments
        if dsets is None or date not in segments:
            continue
        # the day's tradeable segments (handles the Friday Jumu'ah split)
        segs = segments[date]
        # total tradeable minutes in the day
        tradeable = sum(e - s for s, e in segs) / 60000.0
        # first-15-min bucket ends 15 min after the open
        f_end = segs[0][0] + 15 * 60000
        # last-15-min bucket starts 15 min before the close
        l_start = segs[-1][1] - 15 * 60000
        # pre-close-45 bucket starts 60 min before the close
        p_start = segs[-1][1] - 60 * 60000
        # pre-close bucket minutes (bounded, mirrors build_volume_profile)
        p_min = max(min(45.0, tradeable - 30.0), 1.0)
        # middle bucket minutes (the remainder after first/pre/last)
        mid_min = max(tradeable - 30.0 - p_min, 1.0)
        # for each root being calibrated
        for root in roots:
            sym = roll.get((root, date))
            # no active contract on this date -> skip
            if sym is None:
                continue
            # read the contract's snapshot rows
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            # build the L1 mid series (None if the book is one-sided/empty)
            l1 = F.l1_mid_series(s) if len(s) else None
            # need enough L1 points for a stable spread/sigma estimate
            if l1 is not None and len(l1) > 50:
                calib[root]["spr"].append(float((l1["ba"] - l1["bb"]).median()))
                # mid-to-mid returns for the volatility estimate
                rets = l1["mid"].pct_change().dropna()
                # need enough returns to compute a std
                if len(rets) > 10:
                    calib[root]["sig"].append(float(rets.std()))
                # record the day's median mid as the fair-price proxy
                calib[root]["fair"].append(float(l1["mid"].median()))
            # read the contract's trades for the volume profile
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # only if there were trades
            if len(t):
                ts = R.to_ms(t["transact_time"]).to_numpy()
                # trade sizes
                qty = t["qty"].to_numpy()
                # mask: trades in the first-15 bucket
                in_f = ts <= f_end
                # mask: trades in the last-15 bucket
                in_l = ts >= l_start
                # mask: trades in the pre-close-45 bucket (excluding first/last)
                in_p = (ts >= p_start) & ~in_l & ~in_f
                # mask: everything else is the middle bucket
                in_m = ~(in_f | in_l | in_p)
                # first-15 shares-per-minute
                calib[root]["f"].append(qty[in_f].sum() / 15.0)
                # last-15 shares-per-minute
                calib[root]["l"].append(qty[in_l].sum() / 15.0)
                # pre-close shares-per-minute
                calib[root]["p"].append(qty[in_p].sum() / p_min)
                # middle shares-per-minute
                calib[root]["m"].append(qty[in_m].sum() / mid_min)
        # calibration heartbeat every 5 days
        if i % 5 == 0 or i == len(cal_dates):
            print(f"  calib {i}/{len(cal_dates)}  {C._fmt(time.perf_counter()-t0)}",
                  flush=True)

    # back-solve session_scale (anchor on production cap 10 for consistency)
    scales, profiles = {}, {}
    # back-solve session_scale per calibrated root
    for root in roots:
        c = calib[root]
        # skip roots with insufficient calibration data
        if not c["spr"] or not c["sig"] or not c["fair"]:
            print(f"{root}: insufficient calibration -- excluded")
            # (skip)
            continue
        # median futures spread over the calibration window (PKR)
        med_spr = float(np.median(c["spr"]))
        # median mid-return volatility
        sig = float(np.median(c["sig"]))
        # median fair price
        fair = float(np.median(c["fair"]))
        # skew denominator; anchor lots_max on the PRODUCTION cap 10 for consistency
        denom = GAMMA * (sig * fair) ** 2 * 1.0 * 10.0
        # degenerate (zero-vol) name -> skip
        if denom <= 0:
            continue
        # session_scale so skew at max inventory equals half the spread
        scales[root] = (med_spr / 2.0) / denom
        # the 4-bucket volume profile medians for the POV unwind
        profiles[root] = (float(np.median(c["f"])) if c["f"] else 0.0,
                          float(np.median(c["m"])) if c["m"] else 0.0,
                          float(np.median(c["p"])) if c["p"] else 0.0,
                          float(np.median(c["l"])) if c["l"] else 0.0)

    # trade days after calibration
    run_dates = date_strs[CAL_DAYS:]
    # roots that calibrated successfully
    live_roots = sorted(scales.keys())
    # per-span and per-day output rows
    span_rows, day_rows = [], []
    # overall run timer
    t0_all = time.perf_counter()
    # announce the run configuration
    print(f"\nsame-day-flatten run: roots={live_roots}  clip={CLIP_LOTS} lot  "
          # second line of the run banner
          f"smoke={SMOKE}\n", flush=True)

    # ---- walk each root's spans ----
    for root in live_roots:
        spans = CH.contract_spans(roll, root, run_dates)
        # ONE_MONTH verification switch: when True, only the first span per name
        # (fast check that the per-side figures populate before the full run)
        if ONE_MONTH:
            spans = spans[:1]
        # MULTI-MONTH: run ALL contract spans (every month) for these names,
        # so markout can be read per span/month (was: first span only in smoke).
        # SMOKE still controls the root list + per-day ledger verbosity.
        # walk each contract span (one contract's active life)
        for sym, span_dates in spans:
            # span state: only the RESIDUAL futures the POV couldn't flatten
            # carries; cash accumulates all flattened P&L
            resid_pos = 0.0
            # flattened-portion futures cash (accumulates all realised P&L)
            fcash = 0.0
            # spot-hedge state for the residual
            spos = 0.0
            # residual spot-hedge cash ledger
            scash = 0.0
            # decomposition accumulators
            dec = {"quoting": 0.0, "drift": 0.0, "flatten_haircut": 0.0,
                   "overnight_basis": 0.0, "overnight_naked": 0.0,
                   "hedge_cost": 0.0}
            # capture/markout lens + fees
            lens = {"capture": 0.0, "markout": 0.0, "fees": 0.0}
            # per-side flow split: how often (and how much volume) someone hit
            # our BID (we buy) vs lifted our ASK (we sell), and the markout on
            # each side separately -- shows WHICH side's flow is toxic
            side_stats = {"buy_fills": 0, "sell_fills": 0,
                          "buy_vol": 0.0, "sell_vol": 0.0,
                          "buy_markout": 0.0, "sell_markout": 0.0}
            # same-day-flatten tracking: days fully flattened vs residual carried
            n_days = 0
            # count of days the position was FULLY flattened same-day
            n_clean_flat = 0
            # running total of shares the book could not absorb
            unfilled_sh_total = 0.0
            # overnight snapshot for residual basis/naked
            prev_Fc = None
            # yesterday's spot close mark (for residual overnight P&L)
            prev_Sc = None
            # shares of the residual that were hedged overnight
            on_hedged = 0.0
            # shares of the residual left naked overnight
            on_naked = 0.0
            # FIFO holding time
            hold_lots = deque()
            # holding times (ms) of matched round trips
            hold_times = []
            # walk the span days
            for date in span_dates:
                dsets = R.open_datasets(date)
                # skip days with no data or segments (state carries through untouched)
                if dsets is None or date not in segments:
                    continue
                # the contract's order-book updates
                u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
                # the contract's snapshots
                s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
                # the contract's trades
                t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
                # skip dead/thin days (too few trades or no book)
                if len(t) < MIN_TRADES_DAY or len(s) == 0:
                    continue
                # futures L1 mid series for the day
                mids = F.l1_mid_series(s)
                # skip if the book was unusable
                if mids is None:
                    continue
                # the SPOT snapshot for the underlying (the residual hedge instrument)
                spot_snap = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, root)
                # spot L1 series for the overnight marks
                spot_l1 = F.l1_mid_series(spot_snap) if len(spot_snap) else None
                # build the engine event stream + snapshot groups
                events, snap_groups, t = R.build_events(u, s, t)
                # continuous-auction rows define the session bounds
                cont = s[s["phase"] == "CONTINUOUS_AUCTION"]
                # skip if there was no continuous session
                if len(cont) == 0:
                    continue
                # session start/end in exchange time
                t0_, t1_ = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
                # today's futures open mark
                F_open = float(mids["mid"].iloc[0])
                # today's spot open mark (NaN if no spot book)
                S_open = (float(spot_l1["mid"].iloc[0])
                          # (spot open fallback)
                          if spot_l1 is not None else np.nan)
                # today's futures close mark
                F_close = float(mids["mid"].iloc[-1])
                # today's spot close mark (NaN if no spot book)
                S_close = (float(spot_l1["mid"].iloc[-1])
                           # (spot close fallback)
                           if spot_l1 is not None else np.nan)
                # cumulative decomposition before today (for per-day P&L)
                def _dtot():
                    return (dec["quoting"] + dec["drift"]
                            # second term of the cumulative-decomposition helper
                            + dec["flatten_haircut"] + dec["overnight_basis"]
                            # third term (minus hedge cost) of the helper
                            + dec["overnight_naked"] - dec["hedge_cost"])
                # snapshot the cumulative decomposition before today's bookings
                dec_before = _dtot()

                # ==== OVERNIGHT P&L on any residual carried from yesterday ====
                if prev_Fc is not None and np.isfinite(prev_Fc) and resid_pos != 0:
                    dF = F_open - prev_Fc
                    # sign of the carried residual position
                    sgn = 1.0 if resid_pos > 0 else -1.0
                    # only attribute basis if both spot marks exist
                    if (np.isfinite(S_open) and prev_Sc is not None
                            # (spot-mark guard continued)
                            and np.isfinite(prev_Sc)):
                        dS = S_open - prev_Sc
                        # basis P&L on the hedged part of the residual: signed h*(dF - dS)
                        dec["overnight_basis"] += sgn * on_hedged * (dF - dS)
                    # naked is the residual plug at span end

                # ==== MORNING: lift any residual spot hedge ====
                if spos != 0.0 and spot_l1 is not None:
                    t_lift = spot_l1["ts"].iloc[0] + UNHEDGE_DELAY_MIN * 60000
                    # lift side: buy back a short hedge, sell out a long hedge
                    side = "BUY" if spos < 0 else "SELL"
                    # walk the spot book to lift the residual hedge
                    filled, vwap, mid, fee = CH.walk_spot_book(
                        spot_snap, t_lift, side, abs(spos), SPOT_FEE_PER_SIDE)
                    # book the leg only on a real fill (NaN-guarded)
                    if filled > 0:
                        scash += (-1.0 if side == "BUY" else 1.0) * filled * vwap
                        # pay the spot fee on the lift
                        scash -= fee
                        # slippage only meaningful with a mid benchmark
                        if np.isfinite(mid):
                            slip = (vwap - mid) if side == "BUY" else (mid - vwap)
                            # hedge cost = slippage vs mid + fee on the lift leg
                            dec["hedge_cost"] += slip * filled + fee
                        # net the lift into the standing spot position
                        spos += filled if side == "BUY" else -filled

                # ==== INTRADAY: futures MM with EOD trigger ON (flatten daily) ==
                clip = CLIP_LOTS * LOT
                # copy the locked production params for this day
                params = dict(MID_BASE)
                # clip size in shares
                params["size"] = clip
                # hard inventory cap in shares
                params["max_inv"] = int(round(MAXINV_CLIPS * clip))
                # soft inventory band in shares
                params["soft_inv"] = int(round(SOFTINV_CLIPS * clip))
                # the calibrated session_scale for this root
                params["session_scale"] = scales[root]
                # risk-aversion gamma
                params["gamma"] = GAMMA
                # the POV unwind volume profile
                params["unwind_profile"] = profiles[root]
                # the POV participation cap
                params["unwind_pov"] = MAX_POV
                # the day's session segments
                params["session_segments"] = segments[date]
                # engine config with the day's session + seeded latency
                cfg = dict(R.CFG, session=(t0_, t1_),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                # instantiate the strategy for the day
                strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
                # instantiate the engine
                bt = Backtester(strat, cfg)
                # inject any residual carried from yesterday (post-hedge-lift)
                bt.pos = resid_pos
                # inject carried residual cash (post morning-lift)
                bt.cash = fcash
                # remember the position we walked in with (its drift is carry, not edge)
                pos_in = resid_pos
                # futures marked equity at the open
                eqF_open = fcash + resid_pos * F_open
                # run the day
                fills, equity, stats = bt.run(events, snap_groups)

                # ==== capture/markout lens (to same-day close) + holding time ==
                cap_d, mko_d = CH.capture_markout_pkr(fills, mids, F_close)
                # accumulate capture (spread at fill instant)
                lens["capture"] += cap_d
                # accumulate markout (adverse selection to the same-day close)
                lens["markout"] += mko_d
                # the day's futures fee (per-side fee on the fill notional); a
                # local so it feeds the from-fill quoting number below even when
                # there are no fills (then it is zero)
                day_fee = 0.0
                # if there were fills, tally fees and feed the holding-time FIFO
                if len(fills):
                    fdf = (fills if isinstance(fills, pd.DataFrame)
                           else pd.DataFrame(fills))
                    # fee = futures per-side fee on the day's fill notional
                    day_fee = float(
                        (fdf["px"] * fdf["qty"]).sum() * FUT_FEE_PER_SIDE)
                    # accumulate the day's fee into the lens total
                    lens["fees"] += day_fee
                    # ---- per-side flow split + per-side markout (to close) ----
                    # the mid timeline (ms) and values for per-side markout
                    ts_m = mids["ts"].to_numpy()
                    # the mid values array
                    mid_m = mids["mid"].to_numpy()
                    # our BUY fills = someone hit our BID
                    fb = fdf[fdf["side"] == "BUY"]
                    # our SELL fills = someone lifted our ASK
                    fs = fdf[fdf["side"] == "SELL"]
                    # count + volume on the buy side
                    side_stats["buy_fills"] += len(fb)
                    side_stats["buy_vol"] += float(fb["qty"].sum()) if len(fb) else 0.0
                    # count + volume on the sell side
                    side_stats["sell_fills"] += len(fs)
                    side_stats["sell_vol"] += float(fs["qty"].sum()) if len(fs) else 0.0
                    # buy-side markout: signed +1, (F_close - mid_at_fill)*qty
                    if len(fb):
                        # mid index at (or just before) each buy fill
                        ib = np.clip(np.searchsorted(ts_m, fb["t"].to_numpy(),
                                     side="right") - 1, 0, len(mid_m) - 1)
                        # accumulate buy-side markout (negative = toxic bid)
                        side_stats["buy_markout"] += float(
                            np.sum((F_close - mid_m[ib]) * fb["qty"].to_numpy()))
                    # sell-side markout: signed -1, -(F_close - mid_at_fill)*qty
                    if len(fs):
                        # mid index at (or just before) each sell fill
                        iss = np.clip(np.searchsorted(ts_m, fs["t"].to_numpy(),
                                      side="right") - 1, 0, len(mid_m) - 1)
                        # accumulate sell-side markout (negative = toxic ask)
                        side_stats["sell_markout"] += float(
                            np.sum(-(F_close - mid_m[iss]) * fs["qty"].to_numpy()))
                    # walk the day's fills in time order through the FIFO
                    for _, fl in fdf.sort_values("t").iterrows():
                        f_side, f_qty, f_t = fl["side"], float(fl["qty"]), float(fl["t"])
                        # same side as the open queue (or empty) -> this fill OPENS a lot
                        if not hold_lots or hold_lots[0]["side"] == f_side:
                            hold_lots.append({"qty": f_qty, "side": f_side, "t": f_t})
                        else:
                            rem = f_qty
                            # opposite side -> CLOSE oldest opposite lots (FIFO)
                            while rem > 1e-9 and hold_lots and \
                                    hold_lots[0]["side"] != f_side:
                                lot = hold_lots[0]
                                # match size against the oldest lot
                                m = min(rem, lot["qty"])
                                # record this round trip's holding time
                                hold_times.append(f_t - lot["t"])
                                # reduce the matched lot
                                lot["qty"] -= m
                                # reduce the closing quantity
                                rem -= m
                                # lot fully closed -> drop it
                                if lot["qty"] <= 1e-9:
                                    hold_lots.popleft()
                            # any remainder flips to a new lot on this side
                            if rem > 1e-9:
                                hold_lots.append({"qty": rem, "side": f_side, "t": f_t})

                # ==== SAME-DAY FLATTEN: read the engine's EOD liquidation ====
                eod = bt.eod or {}
                # the mid-marked equity at close (before liquidation give-up)
                eqF_mid = float(eod.get("equity_mid_mark") or
                                (bt.cash + bt.pos * F_close))
                # the POV/liquidation-realised equity
                eqF_liq = float(eod.get("equity_liquidated") or eqF_mid)
                # shares the POV unwind could NOT absorb (the residual)
                unfilled = float(eod.get("unfilled_sh") or 0.0)
                # drift on any residual we walked in HOLDING (intraday move on
                # carried-in inventory). On a clean same-day-flatten start we are
                # FLAT, so pos_in = 0 and drift = 0; it is nonzero only after a
                # prior day's POV could not fully flatten and left a residual.
                drift_day = pos_in * (F_close - F_open)
                # accumulate that carried-in drift (its own component)
                dec["drift"] += drift_day
                # QUOTING = the FROM-FILL edge: capture (spread earned at the
                # fill instant) + markout (adverse move from fill to the same-day
                # close) - the day's fee. This measures fill quality from the
                # moment we traded, NOT an open-of-day equity difference -- so it
                # needs no drift subtraction and means exactly "did our fills
                # make money after adverse selection".
                dec["quoting"] += cap_d + mko_d - day_fee
                # flatten haircut = liquidation give-up vs mid mark (THE liquidity
                # cost -- what it costs to POV-exit into the thin futures book)
                dec["flatten_haircut"] += eqF_liq - eqF_mid
                # same-day-flatten bookkeeping
                n_days += 1
                # add today's unfilled shares to the running total
                unfilled_sh_total += abs(unfilled)
                # a fully-flattened day (nothing the book could not absorb)
                if abs(unfilled) < 1e-9:
                    n_clean_flat += 1

                # ==== RESIDUAL: carry + spot-hedge what couldn't be flattened ==
                if abs(unfilled) > 1e-9:
                    # the engine liquidated everything EXCEPT `unfilled`; the
                    # residual position is `unfilled` signed like the day's pos.
                    # Take fcash as the FLATTENED-portion equity, and carry the
                    # residual marked at close (its overnight move is basis/naked)
                    sgn_pos = 1.0 if bt.pos > 0 else -1.0
                    # the residual position = the unfilled shares, signed like the day's pos
                    resid_pos = sgn_pos * abs(unfilled)
                    # fcash = liquidated equity of the flattened part, minus the
                    # residual's mark (so fcash + resid_pos*F_close = eqF_liq)
                    fcash = eqF_liq - resid_pos * F_close
                    # hedge the residual on the spot book at the close
                    if spot_l1 is not None:
                        need = (-resid_pos) - spos
                        # trade only the difference vs any standing spot hedge
                        if abs(need) > 1e-9:
                            side = "BUY" if need > 0 else "SELL"
                            # walk the spot book to hedge the residual
                            filled, vwap, mid, fee = CH.walk_spot_book(
                                spot_snap, t1_, side, abs(need), SPOT_FEE_PER_SIDE)
                            # book only on a real fill
                            if filled > 0:
                                scash += (-1.0 if side == "BUY" else 1.0) * filled * vwap
                                # pay the spot fee
                                scash -= fee
                                # slippage only with a mid
                                if np.isfinite(mid):
                                    slip = (vwap - mid) if side == "BUY" else (mid - vwap)
                                    # hedge cost = slippage vs mid + fee
                                    dec["hedge_cost"] += slip * filled + fee
                                # net the fill into the standing spot position
                                spos += filled if side == "BUY" else -filled
                        # hedged part of the residual
                        on_hedged = min(abs(spos), abs(resid_pos))
                        # naked remainder of the residual
                        on_naked = abs(resid_pos) - on_hedged
                    else:
                        on_hedged = 0.0
                        # no spot book: the whole residual is naked
                        on_naked = abs(resid_pos)
                    # remember tonight's futures close for the residual overnight P&L
                    prev_Fc = F_close
                    # remember tonight's spot close
                    prev_Sc = S_close
                else:
                    # clean flatten: fully flat, no carry, no overnight
                    resid_pos = 0.0
                    # clean flatten: futures cash = the fully-liquidated equity
                    fcash = eqF_liq
                    # nothing hedged (flat)
                    on_hedged = 0.0
                    # nothing naked (flat)
                    on_naked = 0.0
                    # no residual futures mark to carry
                    prev_Fc = None
                    # no residual spot mark to carry
                    prev_Sc = None

                # per-day ledger
                day_pnl = _dtot() - dec_before
                # append the per-day ledger row
                day_rows.append({
                    # which arm produced this row
                    "arm": ARM_NAME,
                    "date": date, "root": root, "contract": sym,
                    "fills": len(fills), "day_pnl": round(day_pnl, 2),
                    "unfilled_sh": abs(unfilled),
                    "resid_pos_sh": resid_pos,
                    "quoting_cum": round(dec["quoting"], 2),
                    "drift_cum": round(dec["drift"], 2),
                    "haircut_cum": round(dec["flatten_haircut"], 2),
                    "basis_cum": round(dec["overnight_basis"], 2),
                    "hedge_cost_cum": round(dec["hedge_cost"], 2)})
                # smoke mode: print the day's ledger line live
                if SMOKE:
                    print(f"  {date} flat pos_resid={resid_pos:>6.0f}sh "
                          # (ledger: unfilled + fills)
                          f"unfill={abs(unfilled):>5.0f} fills={len(fills):>4d} "
                          # (ledger: cumulative quoting)
                          f"quote={dec['quoting']:>+9.0f} "
                          # (ledger: cumulative drift)
                          f"drift={dec['drift']:>+8.0f} "
                          # (ledger: cumulative flatten haircut)
                          f"haircut={dec['flatten_haircut']:>+8.0f} "
                          # (ledger: cumulative basis)
                          f"basis={dec['overnight_basis']:>+7.0f} "
                          # (ledger: cumulative hedge cost)
                          f"hedge=-{dec['hedge_cost']:>6.0f}", flush=True)

            # ---- forced settle any surviving residual (span end) ----
            if resid_pos != 0.0 and prev_Fc is not None:
                fcash += resid_pos * prev_Fc
                # the residual is now settled -> flat
                resid_pos = 0.0
            # close any surviving spot hedge at its last close mark
            if spos != 0.0 and prev_Sc is not None and np.isfinite(prev_Sc):
                scash += spos * prev_Sc
                # spot hedge now flat
                spos = 0.0

            # ---- reconciliation (naked = residual plug) ----
            total_cash = fcash + scash
            # the five directly-measured components
            measured = (dec["quoting"] + dec["drift"] + dec["flatten_haircut"]
                        # (measured: basis minus hedge cost)
                        + dec["overnight_basis"] - dec["hedge_cost"])
            # naked = the reconciling residual (makes recon exact by construction)
            dec["overnight_naked"] = total_cash - measured
            # decomposition total now equals cash by definition
            total_dec = measured + dec["overnight_naked"]
            # reconciliation error (zero by construction, kept for the report)
            recon = total_dec - total_cash
            # same-day-flatten rate = fraction of days fully flattened
            flat_rate = (n_clean_flat / n_days) if n_days else np.nan
            # append the per-span verdict row
            span_rows.append({
                # which arm produced this row
                "arm": ARM_NAME,
                "root": root, "contract": sym, "days": len(span_dates),
                "pnl_total": round(total_cash, 2),
                "quoting": round(dec["quoting"], 2),
                "drift": round(dec["drift"], 2),
                "flatten_haircut": round(dec["flatten_haircut"], 2),
                "basis": round(dec["overnight_basis"], 2),
                "naked_gap": round(dec["overnight_naked"], 2),
                "hedge_cost": round(dec["hedge_cost"], 2),
                "sameday_flat_rate": round(flat_rate, 3),
                "avg_unfilled_sh": round(unfilled_sh_total / n_days, 1) if n_days else np.nan,
                "recon_err": round(recon, 2),
                "q_capture": round(lens["capture"], 2),
                "q_markout": round(lens["markout"], 2),
                "q_fees": round(lens["fees"], 2),
                "hold_med_s": (round(float(np.median(hold_times)) / 1000.0, 1)
                               # (holding-time median guard)
                               if hold_times else np.nan),
                "n_round_trips": len(hold_times),
                # per-side flow: fill-count % and volume % on the buy side
                "buy_fills_pct": (round(100.0 * side_stats["buy_fills"] /
                                  (side_stats["buy_fills"] + side_stats["sell_fills"]), 1)
                                  if (side_stats["buy_fills"] + side_stats["sell_fills"]) else np.nan),
                # buy-side share of volume
                "buy_vol_pct": (round(100.0 * side_stats["buy_vol"] /
                                (side_stats["buy_vol"] + side_stats["sell_vol"]), 1)
                                if (side_stats["buy_vol"] + side_stats["sell_vol"]) else np.nan),
                # per-side markout (PKR): which side's flow is toxic
                "buy_markout": round(side_stats["buy_markout"], 2),
                "sell_markout": round(side_stats["sell_markout"], 2)})
            # full-mode per-span heartbeat
            if not SMOKE:
                el = time.perf_counter() - t0_all
                # print the span result
                print(f"  {root} {sym}: {len(span_dates)}d "
                      # (heartbeat: pnl + flatten rate)
                      f"pnl={total_cash:>+10.0f} flat_rate={flat_rate:.2f} "
                      # (heartbeat: elapsed)
                      f"{C._fmt(el)}", flush=True)

    # ---- outputs ----
    sd = pd.DataFrame(span_rows)
    # daily rows to a frame
    dd = pd.DataFrame(day_rows)
    # write the span CSV
    sd.to_csv(RESULTS / f"fut_sameday_spans_{ARM_NAME}_{stamp}.csv", index=False)
    # write the daily CSV
    dd.to_csv(RESULTS / f"fut_sameday_daily_{ARM_NAME}_{stamp}.csv", index=False)
    # header
    print("\n=== FUTURES SAME-DAY FLATTEN (+ residual spot hedge) ===")
    # column-definitions header
    print("COLUMNS (PKR over the span):")
    # (def: pnl_total)
    print("  pnl_total        = cash truth (futures + residual-hedge cash)")
    # (def: quoting)
    print("  quoting          = FROM-FILL edge = capture + markout - fee")
    print("                     (spread earned at fill + adverse move to close);")
    print("                     this IS the fill-quality number, no drift in it")
    # (def: drift)
    print("  drift            = intraday move on the carried-in RESIDUAL only")
    # (def: flatten_haircut)
    print("  flatten_haircut  = cost of POV-exiting into the futures book at")
    # (def: flatten_haircut continued)
    print("                     close = THE DIRECT LIQUIDITY COST")
    # (def: basis/naked)
    print("  basis/naked_gap  = overnight P&L on the residual (hedged/unhedged)")
    # (def: hedge_cost)
    print("  hedge_cost       = spot book-walk cost on the residual hedge")
    # (def: sameday_flat_rate)
    print("  sameday_flat_rate= fraction of days FULLY flattened (liquidity")
    # (def: sameday_flat_rate continued)
    print("                     metric: 1.0 = never forced to carry)")
    # (def: avg_unfilled_sh)
    print("  avg_unfilled_sh  = avg shares/day the book could NOT absorb")
    # (def: capture/markout)
    print("  q_capture/markout= spread earned at fill vs adverse selection to")
    # (def: capture/markout continued)
    print("                     the CLOSE (toxicity flattening can't fix)\n")
    # only print the table if there are spans
    if len(sd):
        show = ["root", "contract", "days", "pnl_total", "quoting", "drift",
                "flatten_haircut", "basis", "naked_gap", "hedge_cost",
                "sameday_flat_rate", "avg_unfilled_sh", "recon_err",
                "q_capture", "q_markout", "hold_med_s"]
        # the span table (selected columns)
        print(sd[show].to_string(index=False))
        # totals line
        print(f"\nTOTALS: pnl {sd.pnl_total.sum():>+12,.0f}  "
              # (totals: quoting)
              f"quoting {sd.quoting.sum():>+12,.0f}  "
              # (totals: drift)
              f"drift {sd.drift.sum():>+9,.0f}  "
              # (totals: haircut)
              f"haircut {sd.flatten_haircut.sum():>+9,.0f}  "
              # (totals: hedge)
              f"hedge -{sd.hedge_cost.sum():>8,.0f}")
        # the reconciliation check
        print(f"RECON: max abs span error {sd.recon_err.abs().max():.2f} PKR")
        # the quoting lens totals
        print(f"\nQUOTING LENS (PKR): capture {sd.q_capture.sum():>+12,.0f}  "
              # (lens: markout)
              f"markout {sd.q_markout.sum():>+12,.0f}  "
              # (lens: fees)
              f"fees -{sd.q_fees.sum():>9,.0f}")
        # ---- PER-TICKER summary across all that name's spans/months ----
        # aggregate each name's spans: total pnl, capture, markout, side split
        print("\n=== PER-TICKER (aggregated across all months) ===")
        print(f"{'root':>5s} {'spans':>5s} {'pnl':>10s} {'capture':>10s} "
              f"{'markout':>10s} {'buyMkt':>9s} {'sellMkt':>9s} "
              f"{'buy%vol':>8s} {'flat_rate':>9s}")
        # group the span frame by root
        for root in sorted(sd["root"].unique()):
            # this name's spans
            g = sd[sd["root"] == root]
            # volume-weighted buy% across the name's spans (weight by round trips)
            print(f"{root:>5s} {len(g):>5d} {g.pnl_total.sum():>10,.0f} "
                  f"{g.q_capture.sum():>10,.0f} {g.q_markout.sum():>10,.0f} "
                  f"{g.buy_markout.sum():>9,.0f} {g.sell_markout.sum():>9,.0f} "
                  f"{g.buy_vol_pct.mean():>8.1f} {g.sameday_flat_rate.mean():>9.2f}")
        print("  buyMkt/sellMkt = markout PKR on our BUY fills vs our SELL fills")
        print("  (negative = that side's flow is toxic). buy%vol = share of")
        print("  volume where our BID was hit (50 = balanced).")
        # ---- PER-MONTH markout: was October an outlier? ----
        # each span IS one contract-month; show markout per span chronologically
        print("\n=== PER-SPAN (MONTH) MARKOUT: is any month an outlier? ===")
        print(f"{'root':>5s} {'contract':>12s} {'pnl':>10s} {'capture':>10s} "
              f"{'markout':>10s} {'net(cap+mkt)':>13s}")
        # sort by root then contract for a chronological read
        for _, r in sd.sort_values(["root", "contract"]).iterrows():
            # net fill edge = capture + markout (both signed; fees separate)
            net = r.q_capture + r.q_markout
            print(f"{r.root:>5s} {r.contract:>12s} {r.pnl_total:>10,.0f} "
                  f"{r.q_capture:>10,.0f} {r.q_markout:>10,.0f} {net:>13,.0f}")
        print("  if net (capture+markout) is negative in EVERY month -> toxicity")
        print("  is structural, futures MM dead. If negative only in some months")
        print("  -> the single-month (October) sample misled us.")
    # the interpretation guide
    print("\nTHE TEST: if same-day flatten removes drift (drift ~ 0) AND the")
    # (guide continued)
    print("flatten_haircut is small (liquid enough to exit) AND quoting clears")
    # (guide continued)
    print("markout, futures MM works on that name. If even TRG (most liquid)")
    # (guide continued)
    print("fails, PSX futures are too illiquid/toxic for MM -- thread closed.")
    # wrote-span-file line
    print(f"\nwrote {RESULTS / f'fut_sameday_spans_{ARM_NAME}_{stamp}.csv'}")
    # wrote-daily-file line
    print(f"wrote {RESULTS / f'fut_sameday_daily_{ARM_NAME}_{stamp}.csv'}")
    # hand the span frame back so the entry point can compare arms
    return sd


# entry point
if __name__ == "__main__":
    # collected per-arm span frames, for the cross-arm comparison at the end
    _results = {}
    # run every arm in turn. Each call re-does the pre-passes (calendar, roll
    # map, calibration) -- a couple of minutes each -- but guarantees every arm
    # sees IDENTICAL calibration, dates and spans, so the only difference
    # between two arms is the mechanism being tested.
    for _name, _extra in ARMS:
        # the arm's strategy params = the plain base plus this arm's additions
        MID_BASE = dict(MID_BASE_PLAIN, **_extra)
        # tag the arm so its rows and filenames identify themselves
        ARM_NAME = _name
        # announce which arm is starting
        print("\n" + "#" * 78)
        print(f"# ARM: {_name}   extra params: {_extra or '(none -- plain)'}")
        print("#" * 78, flush=True)
        # run it, keeping the span frame for the comparison
        _results[_name] = main()

    # ---- CROSS-ARM COMPARISON: the whole point of the run ----
    print("\n" + "=" * 78)
    print("CROSS-ARM COMPARISON (same dates, same calibration, same spans)")
    print("=" * 78)
    # header
    print(f"{'arm':>14s} {'pnl_total':>12s} {'quoting':>12s} {'capture':>12s} "
          f"{'markout':>12s} {'haircut':>10s} {'flat_rate':>10s}")
    # the plain arm is the baseline every other arm is measured against
    _base = None
    # one row per arm, in the order they ran
    for _name, _ in ARMS:
        _sd = _results.get(_name)
        # skip an arm that produced no spans
        if _sd is None or not len(_sd):
            print(f"{_name:>14s}  (no spans)")
            continue
        # this arm's totals
        _p = _sd.pnl_total.sum()
        # remember the plain arm as the baseline
        if _base is None:
            _base = _p
        print(f"{_name:>14s} {_p:>12,.0f} {_sd.quoting.sum():>12,.0f} "
              f"{_sd.q_capture.sum():>12,.0f} {_sd.q_markout.sum():>12,.0f} "
              f"{_sd.flatten_haircut.sum():>10,.0f} "
              f"{_sd.sameday_flat_rate.mean():>10.2f}")
    # the deltas that answer the question
    print("\nDELTA vs the plain arm (what each mechanism is worth on futures):")
    # walk the arms after the first
    for _name, _ in ARMS[1:]:
        _sd = _results.get(_name)
        # only if this arm produced spans and we have a baseline
        if _sd is None or not len(_sd) or _base is None:
            continue
        # the money difference
        _d = _sd.pnl_total.sum() - _base
        print(f"  {_name:>14s}  {_d:>+12,.0f} PKR"
              + (f"   ({100*_d/abs(_base):+.1f}% of |plain|)" if _base else ""))
    # the reading rule, stated before the numbers are seen
    print("\nREAD: the spot book gains +1.18 bps from the lean and ~+0.66 bps/day")
    print("from the throttle. If 'full_spot' does not beat 'plain' here, the spot")
    print("recipe does NOT transfer to futures and the thread closes. If it does,")
    print("the per-arm rows say WHICH of the three mechanisms did the work.")
    print("\nCAUTION: capture going NEGATIVE under the lean is the A.2 signature")
    print("(cheap-tick spot names: capture +1.64 -> -1.79 bps). Check q_capture")
    print("per arm before believing any P&L improvement.")
