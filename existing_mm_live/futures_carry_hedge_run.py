# futures_carry_hedge_run.py -- FUTURES MM with OVERNIGHT CARRY + EOD SPOT HEDGE.
#
# Replaces the daily-flatten futures backtest (which charged an unclean-close
# haircut EVERY day -- a spot assumption, wrong for futures). Design per SZ:
#   * CARRY futures inventory across days WITHIN a contract's active span
#     (state pos/cash persists day to day; the standing max_inv cap still binds
#     because the engine passes self.pos into strategy.quotes() every event).
#   * EOD SPOT DELTA HEDGE, KISS: at each close (except the span's last day),
#     hedge the carried futures position with the OPPOSITE spot position,
#     executed by WALKING THE REAL SPOT BOOK (full fill simulation -- levels
#     consumed at their prices, spot fee paid). Lifted next morning the same way.
#     Intraday the futures MM runs naked (no continuous hedging -- fee 8x).
#   * ROLL-FLATTEN: on the contract's LAST active day (per the causal roll map),
#     the production EOD trigger + POV unwind is enabled, so the position is
#     worked off into the close and the engine's honest liquidation accounting
#     (incl. haircut if unclean) becomes the ROLL COST. No delivery -> CDC=0.
#   * 4-WAY P&L DECOMPOSITION so a "profitable" result cannot hide a directional
#     bet: quoting edge (intraday, marked at futures mid) / basis P&L (overnight
#     futures-vs-spot move on the HEDGED quantity) / naked gap (overnight move on
#     any UNHEDGED remainder -- partial hedge fills are real and reported) /
#     hedge cost (book-walk slippage vs spot mid + spot fees, both legs) /
#     roll cost (final-day mid-mark minus liquidated equity).
#   * RECONCILIATION: sum of the decomposition must equal total cash-based P&L
#     (futures cash after roll + all spot hedge cash flows). Checked per span.
#
# SMOKE MODE (default): ONE root, ONE contract span (~10-20 days) with a full
# per-day ledger printed -- verify the accounting BEFORE any full run.
# Set SMOKE=False for the full universe (runtime comparable to futures_mm_run).
#
# Run from existing_mm_live/:  caffeinate -is python3 futures_carry_hedge_run.py

# filesystem paths
from pathlib import Path
# timing + run stamp
import time
from datetime import datetime
# frames + arrays
import pandas as pd
import numpy as np
# driver + engine + strategy modules (fee patch needs module handles)
import run_legacy_mm as R
import mm_backtest as MB
import micro_mm as MM
# the engine classes
from mm_backtest import Backtester, LatencyModel
from micro_mm import MicrostructureMM
# heartbeat formatter
import confirm_micro_vs_naive as C
# REUSE the futures runner's shared pieces (no copies): roll map, calendar,
# L1 mid series, segments loader, roots list, futures fee
import futures_mm_run as F

# raw store + results locations
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ experiment knobs ------------------------------
# SMOKE: one root, one span, per-day ledger printed. Flip to False for full run.
SMOKE = True
# the root used in smoke mode (liquid, positive-edge in the flatten run)
SMOKE_ROOT = "THCCL"
# the futures universe in full mode (reuse the runner's list)
ROOTS = F.ROOTS
# clip in lots (1 lot = 500 shares) -- start at the 1x anchor only; sizing sweeps
# come after the carry+hedge mechanics are validated
CLIP_LOTS = 1
LOT = 500
# standing inventory limits in clips (NOT reset at close -- the cap carries)
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
MAX_POV = 0.10
# walk-forward calibration window (same convention as futures_mm_run)
CAL_DAYS = 20
# minimum trades for a contract-day to be quoted
MIN_TRADES_DAY = 200
# minimum days in a contract's active span to bother trading it
MIN_SPAN_DAYS = 5
# morning unhedge: minutes after the spot continuous open to lift the hedge
# (skips the open-cross chaos; flagged config, not a hidden constant)
UNHEDGE_DELAY_MIN = 5.0
# engine defaults matching the locked production config
GAMMA = 0.15
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_lock_trigger=True)
# futures fee (Laga only, proprietary) -- reuse the runner's constant
FUT_FEE_PER_SIDE = F.FUT_FEE_PER_SIDE
# ------------------------------------------------------------------------------


# ---- SPOT BOOK WALK: full fill simulation for the hedge ----------------------
# Executes a marketable spot order of `shares` at time t_ms by consuming the
# REAL spot book levels from the snapshot table: SELL hits BIDs (best px down),
# BUY hits OFFERs (best px up). Returns (filled, vwap, mid, fee_pkr).
# unfilled = shares the visible book could not absorb (partial hedge -- the
# remainder stays NAKED overnight and is reported, never silently dropped).
def walk_spot_book(snap, t_ms, side, shares, fee_pct):
    # continuous-phase rows only (no auction/halt levels)
    c = snap[snap["phase"] == "CONTINUOUS_AUCTION"]
    # nothing to walk without a book
    if len(c) == 0:
        return 0.0, np.nan, np.nan, 0.0
    # timestamps in ms for the asof cut
    ts = R.to_ms(c["orig_time"])
    # rows at or before the hedge time
    upto = c[ts <= t_ms]
    # no book yet at this time
    if len(upto) == 0:
        return 0.0, np.nan, np.nan, 0.0
    # the last snapshot message at/before t_ms = the book state we hit
    last_seq = upto["msg_seq"].iloc[-1]
    book = upto[upto["msg_seq"] == last_seq]
    # the side we consume: SELLing hits BIDs, BUYing hits OFFERs
    hit = book[book["entry_type"] == ("BID" if side == "SELL" else "OFFER")]
    # both sides' best for the mid (slippage benchmark)
    bb = book[book["entry_type"] == "BID"]["px"].max()
    ba = book[book["entry_type"] == "OFFER"]["px"].min()
    # the mid at execution time (NaN if one-sided)
    mid = 0.5 * (bb + ba) if (pd.notna(bb) and pd.notna(ba)) else np.nan
    # no depth on the side we need
    if len(hit) == 0:
        return 0.0, np.nan, mid, 0.0
    # price-priority order: best bid first (desc) for sells, best offer (asc) for buys
    lv = hit.sort_values("px", ascending=(side == "BUY"))
    # remaining shares to execute
    remaining = float(shares)
    # notional filled so far (for the vwap)
    notional = 0.0
    # shares filled so far
    filled = 0.0
    # consume levels until filled or the book is exhausted
    for _, row in lv.iterrows():
        # can't take more than the level shows
        take = min(remaining, float(row["qty"]))
        # accumulate the execution
        notional += take * float(row["px"])
        filled += take
        remaining -= take
        # done once the order is filled
        if remaining <= 1e-9:
            break
    # nothing filled -> no execution
    if filled <= 0:
        return 0.0, np.nan, mid, 0.0
    # achieved volume-weighted price
    vwap = notional / filled
    # spot fee on the executed notional (per side)
    fee = fee_pct * notional
    # (filled shares, achieved vwap, mid benchmark, fee paid)
    return filled, vwap, mid, fee


# ---- contract SPANS from the roll map: consecutive active dates per contract --
def contract_spans(roll, root, dates):
    # ordered list of (contract, [dates...]) spans for this root
    spans = []
    # the current span being built
    cur_sym, cur_dates = None, []
    # walk the calendar in order
    for d in dates:
        # the active contract on this date (None = no history yet / dead)
        sym = roll.get((root, d))
        # same contract -> extend the span
        if sym == cur_sym:
            if sym is not None:
                cur_dates.append(d)
        # contract changed -> close the old span, open a new one
        else:
            if cur_sym is not None and cur_dates:
                spans.append((cur_sym, cur_dates))
            cur_sym, cur_dates = sym, ([d] if sym is not None else [])
    # close the final span
    if cur_sym is not None and cur_dates:
        spans.append((cur_sym, cur_dates))
    # keep only spans long enough to trade
    return [(s, ds) for s, ds in spans if len(ds) >= MIN_SPAN_DAYS]


def main():
    # run stamp for outputs
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # ---- capture the SPOT fee BEFORE patching the futures fee ----
    # (the hedge legs trade SPOT and must pay the spot fee; the futures quotes
    # pay the futures fee -- two different fees live in one backtest)
    SPOT_FEE_PER_SIDE = MB.FEE_TOTAL_PCT
    # patch the futures fee into both modules (fee_for reads at call time;
    # micro_mm's viability gate imported it by value)
    MB.FEE_TOTAL_PCT = FUT_FEE_PER_SIDE
    MM.FEE_TOTAL_PCT = FUT_FEE_PER_SIDE
    # announce the two-fee structure
    print(f"fees: futures {FUT_FEE_PER_SIDE*1e4:.4f} bps/side (quotes), "
          f"spot {SPOT_FEE_PER_SIDE*1e4:.4f} bps/side (hedge legs)")

    # session segments (same trading calendar as spot)
    segments = F.load_segments()
    # all trading dates
    all_dates = R.discover_dates()
    # date strings for the roll map
    date_strs = [str(d) for d in all_dates]

    # ---- pre-pass 1: futures calendar + causal roll map (reused) ----
    print("pre-pass 1: futures calendar + roll map", flush=True)
    cal = F.load_futures_calendar()
    roll = F.build_roll_map(cal, all_dates)

    # ---- pre-pass 2: walk-forward calibration (same as futures_mm_run) ----
    # NOTE: this mirrors futures_mm_run's inline calibration; folding both into
    # a shared harness function is the follow-up refactor (flagged, not hidden).
    print(f"pre-pass 2: calibration on first {CAL_DAYS} days", flush=True)
    cal_dates = date_strs[:CAL_DAYS]
    roots = [SMOKE_ROOT] if SMOKE else ROOTS
    calib = {r: {"spr": [], "sig": [], "fair": [],
                 "f": [], "m": [], "p": [], "l": []} for r in roots}
    t0 = time.perf_counter()
    for i, date in enumerate(cal_dates, 1):
        # datasets for the calibration date
        dsets = R.open_datasets(date)
        if dsets is None or date not in segments:
            continue
        # the day's segments + bucket boundaries (mirrors the volume profile)
        segs = segments[date]
        tradeable = sum(e - s for s, e in segs) / 60000.0
        f_end = segs[0][0] + 15 * 60000
        l_start = segs[-1][1] - 15 * 60000
        p_start = segs[-1][1] - 60 * 60000
        p_min = max(min(45.0, tradeable - 30.0), 1.0)
        mid_min = max(tradeable - 30.0 - p_min, 1.0)
        for root in roots:
            # the active contract on the calibration date
            sym = roll.get((root, date))
            if sym is None:
                continue
            # L1 series for spread/sigma/fair
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            l1 = F.l1_mid_series(s) if len(s) else None
            if l1 is not None and len(l1) > 50:
                calib[root]["spr"].append(float((l1["ba"] - l1["bb"]).median()))
                rets = l1["mid"].pct_change().dropna()
                if len(rets) > 10:
                    calib[root]["sig"].append(float(rets.std()))
                calib[root]["fair"].append(float(l1["mid"].median()))
            # trades for the 4-bucket unwind profile
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t):
                ts = R.to_ms(t["transact_time"]).to_numpy()
                qty = t["qty"].to_numpy()
                in_f = ts <= f_end
                in_l = ts >= l_start
                in_p = (ts >= p_start) & ~in_l & ~in_f
                in_m = ~(in_f | in_l | in_p)
                calib[root]["f"].append(qty[in_f].sum() / 15.0)
                calib[root]["l"].append(qty[in_l].sum() / 15.0)
                calib[root]["p"].append(qty[in_p].sum() / p_min)
                calib[root]["m"].append(qty[in_m].sum() / mid_min)
        if i % 5 == 0 or i == len(cal_dates):
            print(f"  calib {i}/{len(cal_dates)}  {C._fmt(time.perf_counter()-t0)}",
                  flush=True)

    # back-solve session_scale per root (the engine's exact skew identity)
    scales, profiles = {}, {}
    for root in roots:
        c = calib[root]
        if not c["spr"] or not c["sig"] or not c["fair"]:
            print(f"{root}: insufficient calibration -- excluded")
            continue
        med_spr = float(np.median(c["spr"]))
        sig = float(np.median(c["sig"]))
        fair = float(np.median(c["fair"]))
        denom = GAMMA * (sig * fair) ** 2 * 1.0 * MAXINV_CLIPS
        if denom <= 0:
            continue
        scales[root] = (med_spr / 2.0) / denom
        profiles[root] = (float(np.median(c["f"])) if c["f"] else 0.0,
                          float(np.median(c["m"])) if c["m"] else 0.0,
                          float(np.median(c["p"])) if c["p"] else 0.0,
                          float(np.median(c["l"])) if c["l"] else 0.0)

    # ---- trade days = after the calibration window ----
    run_dates = date_strs[CAL_DAYS:]
    live_roots = sorted(scales.keys())
    # per-span result rows + per-day ledger rows
    span_rows, day_rows = [], []
    t0_all = time.perf_counter()
    print(f"\ncarry+hedge run: roots={live_roots}  clip={CLIP_LOTS} lot(s)  "
          f"smoke={SMOKE}\n", flush=True)
    # column definitions for the per-day ledger (smoke mode)
    if SMOKE:
        print("LEDGER COLUMNS (all cum* figures cumulative over the span, PKR):")
        print("  dayPnL  = today's TOTAL P&L: quoting edge + inventory drift +")
        print("            overnight basis/naked realized this morning - hedge")
        print("            legs' cost booked today + any roll cost")
        print("  pos     = futures inventory (shares) carried into tonight")
        print("  fills   = engine fills today")
        print("  quote   = cum QUOTING EDGE: mid-marked MM P&L excluding the")
        print("            intraday move on inventory carried in at the open")
        print("  drift   = cum INVENTORY DRIFT: carried-in pos x intraday futures")
        print("            move (naked by design -- EOD-only hedge)")
        print("  basis   = cum overnight (futures - spot) move on the HEDGED qty")
        print("  naked   = cum overnight futures move on the UNHEDGED remainder")
        print("  hedge   = cum hedge cost: spot book-walk slippage vs mid + spot")
        print("            fees, both legs (shown as a deduction)")
        print("  roll    = cum roll cost: final-day liquidated equity minus")
        print("            mid-marked equity (the exit's price)\n")

    # walk every root's contract spans
    for root in live_roots:
        # this root's tradeable spans within the run window
        spans = contract_spans(roll, root, run_dates)
        # smoke mode: only the FIRST span (one contract's life)
        if SMOKE:
            spans = spans[:1]
        for sym, span_dates in spans:
            # ---- span state: carried futures position + cash ----
            carry_pos = 0.0
            carry_cash = 0.0
            # spot-hedge state: shares currently held as hedge (signed) + the
            # cash ledger of every hedge leg (fills + fees)
            hedge_pos = 0.0
            hedge_cash = 0.0
            # decomposition accumulators for the span
            dec = {"quoting": 0.0, "drift": 0.0, "basis": 0.0,
                   "naked_gap": 0.0, "hedge_cost": 0.0, "roll_cost": 0.0}
            # yesterday's spot close mid (still used for forced settlement)
            prev_F_close = None
            prev_S_close = None
            # the OVERNIGHT SNAPSHOT: last night's carried futures position, how
            # much of it was actually hedged vs naked, and the futures/spot close
            # marks -- the single source of truth for overnight basis/naked P&L
            ov = {"fpos": 0.0, "hedged": 0.0, "naked": 0.0,
                  "Fc": np.nan, "Sc": np.nan}
            # hedged/unhedged quantities set at each evening hedge
            hedged_qty = 0.0
            naked_qty = 0.0
            # count unclean roll (final-day) liquidation
            roll_unclean = False
            # walk the span's days in order
            for di, date in enumerate(span_dates):
                # roll-flatten window: the last 2 span dates (the contract's
                # expiry approach is known ex-ante, so this is not lookahead;
                # 2 days covers a dead/thin final date)
                last_day = (di >= len(span_dates) - 2)
                # datasets for the day
                dsets = R.open_datasets(date)
                if dsets is None or date not in segments:
                    continue
                # the futures contract's data
                u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
                s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
                t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
                # skip dead days (but carry state through them untouched)
                if len(t) < MIN_TRADES_DAY or len(s) == 0:
                    continue
                # futures L1 mid series (open/close marks)
                mids = F.l1_mid_series(s)
                if mids is None:
                    continue
                # SPOT snapshot for the underlying (the hedge instrument)
                spot_snap = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, root)
                # spot L1 series for the overnight spot marks
                spot_l1 = F.l1_mid_series(spot_snap) if len(spot_snap) else None
                # the event stream + session bounds
                events, snap_groups, t = R.build_events(u, s, t)
                cont = s[s["phase"] == "CONTINUOUS_AUCTION"]
                if len(cont) == 0:
                    continue
                t0_, t1_ = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())

                # ---- the decomposition total BEFORE today's bookings, so the
                # day's OWN P&L (all components) can be printed per day ----
                dec_before = (dec["quoting"] + dec["drift"] + dec["basis"]
                              + dec["naked_gap"] - dec["hedge_cost"]
                              + dec["roll_cost"])
                # ---- MORNING: lift yesterday's hedge (buy back / sell out) ----
                # the unhedge time: spot continuous open + delay
                if hedge_pos != 0.0 and spot_l1 is not None:
                    # unhedge timestamp
                    t_unhedge = spot_l1["ts"].iloc[0] + UNHEDGE_DELAY_MIN * 60000
                    # lifting a SHORT hedge = BUY spot; lifting a LONG = SELL
                    side = "BUY" if hedge_pos < 0 else "SELL"
                    # walk the spot book for the full hedge size
                    filled, vwap, mid, fee = walk_spot_book(
                        spot_snap, t_unhedge, side, abs(hedge_pos),
                        SPOT_FEE_PER_SIDE)
                    # book the leg ONLY when something filled (vwap is NaN on a
                    # zero fill -- "vwap or 0" does NOT guard NaN, it's truthy)
                    if filled > 0:
                        # the cash flow: buys pay, sells receive; fee paid
                        flow = (-1.0 if side == "BUY" else 1.0) * filled * vwap
                        hedge_cash += flow - fee
                        # execution cost vs mid on this leg (slippage + fee)
                        if np.isfinite(mid):
                            slip = (vwap - mid) if side == "BUY" else (mid - vwap)
                            dec["hedge_cost"] += slip * filled + fee
                    # partial-lift warning (residual spot survives -> retried)
                    if filled < abs(hedge_pos) - 1e-9:
                        print(f"  !! {date} {root}: unhedge PARTIAL "
                              f"({filled:.0f}/{abs(hedge_pos):.0f}) -- residual "
                              f"spot carried, retried at tonight's hedge")
                    # NET the fill into the hedge position (residual survives)
                    hedge_pos += filled if side == "BUY" else -filled
                # ---- overnight P&L from the STORED overnight snapshot (computed
                # against last night's actual futures pos + hedge split + prices,
                # so a failed lift never mis-attributes basis vs naked) ----
                F_open = float(mids["mid"].iloc[0])
                S_open = (float(spot_l1["mid"].iloc[0])
                          if spot_l1 is not None else np.nan)
                if ov["fpos"] != 0.0 and np.isfinite(ov["Fc"]):
                    # sign of last night's carried futures position
                    sgn = 1.0 if ov["fpos"] > 0 else -1.0
                    # basis P&L on the qty that WAS hedged overnight:
                    # signed hedged * ((dFutures) - (dSpot))
                    if np.isfinite(S_open) and np.isfinite(ov["Sc"]):
                        dec["basis"] += sgn * ov["hedged"] * (
                            (F_open - ov["Fc"]) - (S_open - ov["Sc"]))
                    # naked gap on the qty that was NOT hedged: signed naked * dF
                    dec["naked_gap"] += sgn * ov["naked"] * (F_open - ov["Fc"])
                    # consume the snapshot (this overnight is now booked)
                    ov = {"fpos": 0.0, "hedged": 0.0, "naked": 0.0,
                          "Fc": np.nan, "Sc": np.nan}

                # ---- INTRADAY: run the futures MM with carried state ----
                # clip in shares
                clip = CLIP_LOTS * LOT
                # production params; EOD trigger ONLY on the roll-flatten day
                params = dict(MID_BASE)
                params["size"] = clip
                params["max_inv"] = int(round(MAXINV_CLIPS * clip))
                params["soft_inv"] = int(round(SOFTINV_CLIPS * clip))
                params["session_scale"] = scales[root]
                params["gamma"] = GAMMA
                params["unwind_profile"] = profiles[root]
                params["unwind_pov"] = MAX_POV
                params["session_segments"] = segments[date]
                params["enable_eod_trigger"] = bool(last_day)
                # engine config with the day's session + seeded latency
                cfg = dict(R.CFG, session=(t0_, t1_),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                # fresh strategy each day (stateless across days by design)
                strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
                # fresh engine, then INJECT the carried state (the engine passes
                # self.pos into strategy.quotes(), so the skew sees it)
                bt = Backtester(strat, cfg)
                bt.pos = carry_pos
                bt.cash = carry_cash
                # the day's futures open mid (for the quoting-P&L mark)
                F_open_mid = float(mids["mid"].iloc[0])
                # equity at the day's open (carried state marked at open)
                eq_open = carry_cash + carry_pos * F_open_mid
                # the position we WALKED IN with (its intraday drift is carry
                # P&L, not quoting edge -- decomposed separately below)
                pos_in = carry_pos
                # run the day
                fills, equity, stats = bt.run(events, snap_groups)
                # the day's futures close mid
                F_close_mid = float(mids["mid"].iloc[-1])
                # spot close mid (for the overnight basis mark)
                S_close_mid = (float(spot_l1["mid"].iloc[-1])
                               if spot_l1 is not None else np.nan)

                if last_day:
                    # ---- ROLL-FLATTEN DAY: the engine unwound via POV; take
                    # its honest liquidated equity as the span's final state ----
                    eod = bt.eod or {}
                    eq_liq = float(eod.get("equity_liquidated") or
                                   (bt.cash + bt.pos * F_close_mid))
                    eq_mid = float(eod.get("equity_mid_mark") or eq_liq)
                    # intraday drift on the carried-in position (directional
                    # carry P&L, NOT quoting edge)
                    drift_day = pos_in * (F_close_mid - F_open_mid)
                    dec["drift"] += drift_day
                    # quoting P&L on the final day: mid-marked change MINUS the
                    # carried-in drift (isolates the true MM edge)
                    dec["quoting"] += (eq_mid - eq_open) - drift_day
                    # roll cost: what the unwind/liquidation gave up vs mid-mark
                    dec["roll_cost"] += eq_liq - eq_mid
                    # unclean roll flag (couldn't fully exit even on flatten day)
                    roll_unclean = not bool(eod.get("liquidation_clean", True))
                    # the span ends flat: final futures cash is the liquidated equity
                    carry_pos = 0.0
                    carry_cash = eq_liq
                else:
                    # ---- CARRY DAY: mark at close, hedge, carry ----
                    # the day's mid-marked equity change, split into the
                    # carried-in position's intraday drift vs true quoting edge
                    eq_close = bt.cash + bt.pos * F_close_mid
                    drift_day = pos_in * (F_close_mid - F_open_mid)
                    dec["drift"] += drift_day
                    dec["quoting"] += (eq_close - eq_open) - drift_day
                    # carry the state forward
                    carry_pos = bt.pos
                    carry_cash = bt.cash
                    # ---- EOD HEDGE: short spot vs long futures (and vice versa)
                    # defaults if no spot book to hedge into (recomputed below
                    # from actual standing positions once the hedge is attempted)
                    hedged_qty = 0.0
                    naked_qty = abs(carry_pos)
                    # the spot position we WANT tonight = the exact opposite
                    # of the carried futures; trade only the DIFFERENCE vs any
                    # residual hedge still on (netting -- never overwrite)
                    target_spot = -carry_pos
                    need = target_spot - hedge_pos
                    if abs(need) > 1e-9 and spot_l1 is not None:
                        # positive need = BUY spot, negative = SELL spot
                        side = "BUY" if need > 0 else "SELL"
                        # hedge at the futures close time on the spot book
                        filled, vwap, mid, fee = walk_spot_book(
                            spot_snap, t1_, side, abs(need),
                            SPOT_FEE_PER_SIDE)
                        # book the leg ONLY when something filled (NaN guard)
                        if filled > 0:
                            # the spot cash flow of the hedge leg
                            flow = (-1.0 if side == "BUY" else 1.0) * filled * vwap
                            hedge_cash += flow - fee
                            # hedge execution cost vs spot mid (slippage + fee)
                            if np.isfinite(mid):
                                slip = (vwap - mid) if side == "BUY" else (mid - vwap)
                                dec["hedge_cost"] += slip * filled + fee
                            # NET the fill into the standing hedge position
                            hedge_pos += filled if side == "BUY" else -filled
                        pass
                        # loud flag when the spot book couldn't absorb the hedge
                        if naked_qty > 1e-9:
                            print(f"  !! {date} {root}: hedge PARTIAL "
                                  f"({filled:.0f}/{abs(carry_pos):.0f} sh) -- "
                                  f"{naked_qty:.0f} sh naked overnight")
                    # remember tonight's closes (for forced settlement)
                    prev_F_close = F_close_mid
                    prev_S_close = S_close_mid
                    # the overnight hedged/naked split from ACTUAL standing
                    # positions AFTER tonight's hedge trade: the spot hedge held
                    # (|hedge_pos|, incl. any surviving from a failed lift) caps
                    # how much of the futures position is truly hedged. This is
                    # the ONE place the split is computed -- no failed-lift leak.
                    hedged_qty = min(abs(hedge_pos), abs(carry_pos))
                    naked_qty = abs(carry_pos) - hedged_qty
                    # STORE the overnight snapshot for tomorrow's basis/naked
                    ov = {"fpos": carry_pos, "hedged": hedged_qty,
                          "naked": naked_qty, "Fc": F_close_mid,
                          "Sc": S_close_mid}

                # the decomposition total AFTER today's bookings
                dec_after = (dec["quoting"] + dec["drift"] + dec["basis"]
                             + dec["naked_gap"] - dec["hedge_cost"]
                             + dec["roll_cost"])
                # today's own P&L across all components (incl. overnight P&L
                # realized this morning + tonight's hedge leg cost)
                day_pnl = dec_after - dec_before
                # per-day ledger row (the smoke-mode audit trail)
                day_rows.append({
                    "date": date, "root": root, "contract": sym,
                    "last_day": last_day, "fills": len(fills),
                    "day_pnl": round(day_pnl, 2),
                    "eod_pos_sh": carry_pos if not last_day else 0.0,
                    "hedged_sh": hedged_qty if not last_day else 0.0,
                    "naked_sh": naked_qty if not last_day else 0.0,
                    "quoting_cum": round(dec["quoting"], 2),
                    "drift_cum": round(dec["drift"], 2),
                    "basis_cum": round(dec["basis"], 2),
                    "naked_gap_cum": round(dec["naked_gap"], 2),
                    "hedge_cost_cum": round(dec["hedge_cost"], 2),
                    "roll_cost_cum": round(dec["roll_cost"], 2)})
                # smoke mode: print the ledger line live
                if SMOKE:
                    print(f"  {date} {'ROLL' if last_day else 'carry'} "
                          f"dayPnL={day_pnl:>+8.0f} "
                          f"pos={carry_pos:>7.0f}sh fills={len(fills):>4d} "
                          f"quote={dec['quoting']:>+10.0f} "
                          f"drift={dec['drift']:>+9.0f} basis={dec['basis']:>+8.0f} "
                          f"naked={dec['naked_gap']:>+8.0f} "
                          f"hedge=-{dec['hedge_cost']:>7.0f} "
                          f"roll={dec['roll_cost']:>+8.0f}", flush=True)

            # ---- forced settlement: if the flatten days were all dead and
            # inventory survived, book it at the last futures close (marked,
            # not walked -- loud warning, the span's roll cost is understated)
            if carry_pos != 0.0 and prev_F_close is not None:
                print(f"  !! {root} {sym}: span ended with {carry_pos:.0f}sh -- "
                      f"forced settlement at last close (roll cost understated)")
                carry_cash += carry_pos * prev_F_close
                carry_pos = 0.0
            # ---- close any residual spot hedge the same way ----
            if hedge_pos != 0.0 and prev_S_close is not None:
                print(f"  !! {root} {sym}: residual spot hedge {hedge_pos:.0f}sh "
                      f"settled at last spot close")
                hedge_cash += hedge_pos * prev_S_close
                hedge_pos = 0.0
            # ---- span settlement + reconciliation ----
            # total P&L, cash-based ground truth: final futures cash (flat) +
            # every spot hedge cash flow
            total_cash = carry_cash + hedge_cash
            # the decomposition's total (quoting + basis + naked - hedge + roll)
            total_dec = (dec["quoting"] + dec["drift"] + dec["basis"]
                         + dec["naked_gap"] - dec["hedge_cost"]
                         + dec["roll_cost"])
            # reconciliation error between the two totals
            recon = total_dec - total_cash
            # the span verdict row
            span_rows.append({
                "root": root, "contract": sym, "days": len(span_dates),
                "pnl_total": round(total_cash, 2),
                "quoting": round(dec["quoting"], 2),
                "drift": round(dec["drift"], 2),
                "basis": round(dec["basis"], 2),
                "naked_gap": round(dec["naked_gap"], 2),
                "hedge_cost": round(dec["hedge_cost"], 2),
                "roll_cost": round(dec["roll_cost"], 2),
                "roll_unclean": roll_unclean,
                "recon_err": round(recon, 2)})
            # progress heartbeat in full mode
            if not SMOKE:
                el = time.perf_counter() - t0_all
                print(f"  {root} {sym}: {len(span_dates)}d "
                      f"pnl={total_cash:>+10.0f}  {C._fmt(el)}", flush=True)

    # ---- outputs ----
    sd = pd.DataFrame(span_rows)
    dd = pd.DataFrame(day_rows)
    sd.to_csv(RESULTS / f"fut_carry_spans_{stamp}.csv", index=False)
    dd.to_csv(RESULTS / f"fut_carry_daily_{stamp}.csv", index=False)
    # the span-level verdict table
    print("\n=== FUTURES CARRY + EOD SPOT HEDGE: span decomposition ===")
    # column definitions for the span table
    print("COLUMNS (PKR over the whole contract span):")
    print("  pnl_total  = cash ground truth: final futures cash (flat after the")
    print("               roll) + every spot hedge cash flow")
    print("  quoting    = the MM edge, isolated: mid-marked intraday P&L minus")
    print("               the drift on carried-in inventory")
    print("  drift      = intraday futures move on inventory carried in at each")
    print("               open (the cost/gain of being naked intraday, by design)")
    print("  basis      = overnight (futures-spot) move on hedged qty; small if")
    print("               the hedge works (converts gap risk to basis risk)")
    print("  naked_gap  = overnight futures move on any UNHEDGED remainder")
    print("               (partial hedge fills -- spot book too thin)")
    print("  hedge_cost = spot execution: book-walk slippage vs mid + spot fee,")
    print("               both legs, every overnight")
    print("  roll_cost  = final-day POV unwind: liquidated minus mid-marked")
    print("  recon_err  = (quoting+drift+basis+naked-hedge+roll) - pnl_total;")
    print("               must be ~0 or the decomposition is not trustworthy\n")
    if len(sd):
        print(sd.to_string(index=False))
        # portfolio totals per component (the honest 4-way read)
        print(f"\nTOTALS: pnl {sd.pnl_total.sum():>+12,.0f}  "
              f"quoting {sd.quoting.sum():>+12,.0f}  "
              f"drift {sd.drift.sum():>+10,.0f}  "
              f"basis {sd.basis.sum():>+10,.0f}  "
              f"naked {sd.naked_gap.sum():>+10,.0f}  "
              f"hedge -{sd.hedge_cost.sum():>10,.0f}  "
              f"roll {sd.roll_cost.sum():>+10,.0f}")
        print(f"RECON: max abs span error {sd.recon_err.abs().max():.2f} PKR "
              f"(must be ~0 for the decomposition to be trustworthy)")
    print("\nREAD: quoting is the MM edge ISOLATED from inventory drift; drift is")
    print("the intraday move on carried-in inventory (naked by design -- the KISS")
    print("EOD-only hedge accepts intraday direction); basis should be SMALL;")
    print("naked_gap is unhedged overnight risk (partial hedges); hedge_cost is")
    print("the price of neutrality; roll_cost is the monthly exit. Viable iff")
    print("quoting - hedge_cost + roll_cost > 0 with basis/naked small.")
    print(f"\nwrote {RESULTS / f'fut_carry_spans_{stamp}.csv'}")
    print(f"wrote {RESULTS / f'fut_carry_daily_{stamp}.csv'}")


# entry point
if __name__ == "__main__":
    main()
