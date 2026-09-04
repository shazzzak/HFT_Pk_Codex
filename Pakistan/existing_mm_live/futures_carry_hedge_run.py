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
# FIFO queue for holding-time tracking
from collections import deque
# timing + run stamp
import time
# stdlib datetime for the run stamp
from datetime import datetime
# frames + arrays
import pandas as pd
# numpy for vectorised math
import numpy as np
# driver + engine + strategy modules (fee patch needs module handles)
import run_legacy_mm as R
# engine module (holds the patchable fee constant)
import mm_backtest as MB
# strategy module (imports the fee by value)
import micro_mm as MM
# the engine classes
from mm_backtest import Backtester, LatencyModel
# the microstructure MM strategy class
from micro_mm import MicrostructureMM
# heartbeat formatter
import confirm_micro_vs_naive as C
# REUSE the futures runner's shared pieces (no copies): roll map, calendar,
# L1 mid series, segments loader, roots list, futures fee
import futures_mm_run as F

# raw store + results locations
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# where result CSVs are written
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ experiment knobs ------------------------------
# SMOKE: named roots, one span each, per-day ledger printed. Flip False for full.
SMOKE = True
# the roots used in smoke mode: THCCL (spot reject) + MLCF (spot winner) so we
# see whether the quoting-while-carrying problem is universal or name-specific
SMOKE_ROOTS = ["THCCL", "MLCF"]
# the futures universe in full mode (reuse the runner's list)
ROOTS = F.ROOTS
# clip in lots (1 lot = 500 shares) -- start at the 1x anchor only; sizing sweeps
# come after the carry+hedge mechanics are validated
CLIP_LOTS = 1
# shares per futures lot (fixed 500 on PSX DFC)
LOT = 500
# standing inventory cap SWEEP (clips): drift ~ position x sqrt(hold time), so
# a smaller cap should cut drift proportionally. This tests the decomposition's
# prediction directly. SOFTINV scales with it (same 3:10 ratio as production).
MAXINV_CLIPS_SWEEP = [10.0, 5.0, 3.0]
# soft-inventory band as a fraction of the hard cap (production ratio 3/10)
SOFT_FRAC = 0.30
# POV participation cap for the unwind
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
# locked production params (EOD trigger set per-day: only on the roll day)
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_lock_trigger=True)
# futures fee (Laga only, proprietary) -- reuse the runner's constant
FUT_FEE_PER_SIDE = F.FUT_FEE_PER_SIDE
# (MARKOUT_S removed -- markout is now measured to the same-day close, not a
# fixed horizon, so no horizon constant is needed)
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
    # the latest snapshot message at/before the hedge time = the book we hit
    book = upto[upto["msg_seq"] == last_seq]
    # the side we consume: SELLing hits BIDs, BUYing hits OFFERs
    hit = book[book["entry_type"] == ("BID" if side == "SELL" else "OFFER")]
    # both sides' best for the mid (slippage benchmark)
    bb = book[book["entry_type"] == "BID"]["px"].max()
    # best offer for the mid benchmark
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
        # accumulate filled shares
        filled += take
        # reduce remaining to fill
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
            # contract changed -> open a new span
            cur_sym, cur_dates = sym, ([d] if sym is not None else [])
    # close the final span
    if cur_sym is not None and cur_dates:
        spans.append((cur_sym, cur_dates))
    # keep only spans long enough to trade
    return [(s, ds) for s, ds in spans if len(ds) >= MIN_SPAN_DAYS]


# ---- split a day's fills into CAPTURE and MARKOUT in PKR, against the futures
# mid series. capture = signed (mid_at_fill - fill_px) * qty (spread earned at
# the moment of the fill); markout = signed (mid_+Ns - mid_at_fill) * qty (the
# adverse/favorable drift after -- negative = picked off). Their sum is the
# fills' gross mid-relative P&L; the engine's fees are separate. This is a LENS
# on the quoting component, not a replacement for it (the mark-difference
# quoting number stays the reconciling truth).
def capture_markout_pkr(fills, mids, f_close):
    # empty inputs -> zero split
    f = fills if isinstance(fills, pd.DataFrame) else pd.DataFrame(fills)
    # empty inputs -> zero split
    if len(f) == 0 or mids is None or len(mids) < 2:
        return 0.0, 0.0
    # the mid timeline (ms) and values
    ts = mids["ts"].to_numpy()
    # the mid values array
    mid = mids["mid"].to_numpy()
    # fill timestamps, prices, signed direction, sizes
    ft = f["t"].to_numpy()
    # fill prices
    px = f["px"].to_numpy()
    # fill sizes
    qty = f["qty"].to_numpy()
    # +1 for buys, -1 for sells
    sgn = np.where(f["side"].to_numpy() == "BUY", 1.0, -1.0)
    # mid at (or just before) each fill
    i0 = np.clip(np.searchsorted(ts, ft, side="right") - 1, 0, len(mid) - 1)
    # mid at (or just before) each fill
    m0 = mid[i0]
    # capture: bought below / sold above the mid at the fill instant (PKR)
    cap = float(np.sum(sgn * (m0 - px) * qty))
    # markout: adverse selection measured to the SAME-DAY CLOSE (not a fixed 5s
    # horizon) -- the drift from mid-at-fill to today's futures close, signed by
    # our side. This is the holding-till-close adverse selection and matches how
    # the quoting component is mid-marked at the close.
    mko = float(np.sum(sgn * (f_close - m0) * qty))
    # (capture PKR, markout PKR to close)
    return cap, mko


# the main entry point
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
    # patch the futures fee into the strategy module too
    MM.FEE_TOTAL_PCT = FUT_FEE_PER_SIDE
    # announce the two-fee structure
    print(f"fees: futures {FUT_FEE_PER_SIDE*1e4:.4f} bps/side (quotes), "
          # second line of the fee banner
          f"spot {SPOT_FEE_PER_SIDE*1e4:.4f} bps/side (hedge legs)")

    # session segments (same trading calendar as spot)
    segments = F.load_segments()
    # all trading dates
    all_dates = R.discover_dates()
    # date strings for the roll map
    date_strs = [str(d) for d in all_dates]

    # ---- pre-pass 1: futures calendar + causal roll map (reused) ----
    print("pre-pass 1: futures calendar + roll map", flush=True)
    # one-shot DuckDB futures calendar
    cal = F.load_futures_calendar()
    # the causal roll map
    roll = F.build_roll_map(cal, all_dates)

    # ---- pre-pass 2: walk-forward calibration (same as futures_mm_run) ----
    # NOTE: this mirrors futures_mm_run's inline calibration; folding both into
    # a shared harness function is the follow-up refactor (flagged, not hidden).
    print(f"pre-pass 2: calibration on first {CAL_DAYS} days", flush=True)
    # first CAL_DAYS dates are calibration-only
    cal_dates = date_strs[:CAL_DAYS]
    # which roots to calibrate
    roots = SMOKE_ROOTS if SMOKE else ROOTS
    # per-root calibration accumulators
    calib = {r: {"spr": [], "sig": [], "fair": [],
                 "f": [], "m": [], "p": [], "l": []} for r in roots}
    # calibration timer
    t0 = time.perf_counter()
    # walk each calibration date
    for i, date in enumerate(cal_dates, 1):
        # datasets for the calibration date
        dsets = R.open_datasets(date)
        # skip dates with no data/segments
        if dsets is None or date not in segments:
            continue
        # the day's segments + bucket boundaries (mirrors the volume profile)
        segs = segments[date]
        # tradeable minutes
        tradeable = sum(e - s for s, e in segs) / 60000.0
        # first-15 bucket end
        f_end = segs[0][0] + 15 * 60000
        # last-15 bucket start
        l_start = segs[-1][1] - 15 * 60000
        # pre-close-45 bucket start
        p_start = segs[-1][1] - 60 * 60000
        # pre-close bucket minutes
        p_min = max(min(45.0, tradeable - 30.0), 1.0)
        # middle bucket minutes
        mid_min = max(tradeable - 30.0 - p_min, 1.0)
        # for each root
        for root in roots:
            # the active contract on the calibration date
            sym = roll.get((root, date))
            # no active contract -> skip
            if sym is None:
                continue
            # L1 series for spread/sigma/fair
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            # L1 mid series (None if unusable)
            l1 = F.l1_mid_series(s) if len(s) else None
            # need enough points
            if l1 is not None and len(l1) > 50:
                calib[root]["spr"].append(float((l1["ba"] - l1["bb"]).median()))
                # mid returns for sigma
                rets = l1["mid"].pct_change().dropna()
                # need enough returns
                if len(rets) > 10:
                    calib[root]["sig"].append(float(rets.std()))
                # median mid as fair proxy
                calib[root]["fair"].append(float(l1["mid"].median()))
            # trades for the 4-bucket unwind profile
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # only if trades exist
            if len(t):
                ts = R.to_ms(t["transact_time"]).to_numpy()
                # trade sizes
                qty = t["qty"].to_numpy()
                # first-15 mask
                in_f = ts <= f_end
                # last-15 mask
                in_l = ts >= l_start
                # pre-close mask
                in_p = (ts >= p_start) & ~in_l & ~in_f
                # middle mask
                in_m = ~(in_f | in_l | in_p)
                # first-15 sh/min
                calib[root]["f"].append(qty[in_f].sum() / 15.0)
                # last-15 sh/min
                calib[root]["l"].append(qty[in_l].sum() / 15.0)
                # pre-close sh/min
                calib[root]["p"].append(qty[in_p].sum() / p_min)
                # middle sh/min
                calib[root]["m"].append(qty[in_m].sum() / mid_min)
        # calibration heartbeat
        if i % 5 == 0 or i == len(cal_dates):
            print(f"  calib {i}/{len(cal_dates)}  {C._fmt(time.perf_counter()-t0)}",
                  flush=True)

    # back-solve session_scale per root (the engine's exact skew identity)
    scales, profiles = {}, {}
    # back-solve scale per root
    for root in roots:
        c = calib[root]
        # skip insufficient calibration
        if not c["spr"] or not c["sig"] or not c["fair"]:
            print(f"{root}: insufficient calibration -- excluded")
            # (skip)
            continue
        # median spread
        med_spr = float(np.median(c["spr"]))
        # median sigma
        sig = float(np.median(c["sig"]))
        # median fair
        fair = float(np.median(c["fair"]))
        # calibration anchors on the PRODUCTION cap (10 clips) so session_scale
        # is identical across the runtime sweep -- we sweep the runtime cap, not
        # the calibration reference (else scale and cap would move together)
        denom = GAMMA * (sig * fair) ** 2 * 1.0 * 10.0
        # degenerate name -> skip
        if denom <= 0:
            continue
        # session_scale from the skew identity
        scales[root] = (med_spr / 2.0) / denom
        # the 4-bucket volume profile
        profiles[root] = (float(np.median(c["f"])) if c["f"] else 0.0,
                          float(np.median(c["m"])) if c["m"] else 0.0,
                          float(np.median(c["p"])) if c["p"] else 0.0,
                          float(np.median(c["l"])) if c["l"] else 0.0)

    # ---- trade days = after the calibration window ----
    run_dates = date_strs[CAL_DAYS:]
    # roots that calibrated
    live_roots = sorted(scales.keys())
    # per-span result rows + per-day ledger rows
    span_rows, day_rows = [], []
    # overall run timer
    t0_all = time.perf_counter()
    # run banner
    print(f"\ncarry+hedge run: roots={live_roots}  clip={CLIP_LOTS} lot(s)  "
          # banner line 2
          f"smoke={SMOKE}\n", flush=True)
    # column definitions for the per-day ledger (smoke mode)
    if SMOKE:
        print("LEDGER COLUMNS (all cum* figures cumulative over the span, PKR):")
        # ledger column definitions (smoke)
        print("  dayPnL  = today's TOTAL P&L: quoting edge + inventory drift +")
        # (def cont)
        print("            overnight basis/naked realized this morning - hedge")
        # (def cont)
        print("            legs' cost booked today + any roll cost")
        # (def pos)
        print("  pos     = futures inventory (shares) carried into tonight")
        # (def fills)
        print("  fills   = engine fills today")
        # (def quote)
        print("  quote   = cum QUOTING EDGE: mid-marked MM P&L excluding the")
        # (def cont)
        print("            intraday move on inventory carried in at the open")
        # (def drift)
        print("  drift   = cum INVENTORY DRIFT: carried-in pos x intraday futures")
        # (def cont)
        print("            move (naked by design -- EOD-only hedge)")
        # (def basis)
        print("  basis   = cum overnight (futures - spot) move on the HEDGED qty")
        # (def naked)
        print("  naked   = cum overnight futures move on the UNHEDGED remainder")
        # (def hedge)
        print("  hedge   = cum hedge cost: spot book-walk slippage vs mid + spot")
        # (def cont)
        print("            fees, both legs (shown as a deduction)")
        # (def roll)
        print("  roll    = cum roll cost: final-day liquidated equity minus")
        # (def cont)
        print("            mid-marked equity (the exit's price)\n")

    # SWEEP the standing inventory cap (outermost loop). Each config reruns the
    # full carry+hedge across the smoke roots; span_rows carry the config tag.
    for MAXINV_CLIPS in MAXINV_CLIPS_SWEEP:
      # soft cap tracks the hard cap at the production ratio
      SOFTINV_CLIPS = MAXINV_CLIPS * SOFT_FRAC
      # announce the sweep config
      print(f"\n########## SWEEP: MAXINV_CLIPS = {MAXINV_CLIPS:g} "
            # sweep banner line 2
            f"(soft {SOFTINV_CLIPS:g}) ##########", flush=True)
      # walk every root's contract spans
      for root in live_roots:
        # this root's tradeable spans within the run window
        spans = contract_spans(roll, root, run_dates)
        # smoke mode: only the FIRST span (one contract's life)
        if SMOKE:
            spans = spans[:1]
        # walk each contract span
        for sym, span_dates in spans:
            # ==== span state: REAL positions + cash only (no derived state) ====
            # futures position (shares, signed) carried across days
            fpos = 0.0
            # futures cash ledger (PKR): fills add/subtract, fees deducted by engine
            fcash = 0.0
            # spot-hedge position (shares, signed) carried across days
            spos = 0.0
            # spot-hedge cash ledger (PKR): every hedge leg's flow + fee
            scash = 0.0
            # ==== decomposition accumulators (each a marked-equity difference) ==
            dec = {"quoting": 0.0, "drift": 0.0, "overnight_basis": 0.0,
                   "overnight_naked": 0.0, "hedge_cost": 0.0, "roll_cost": 0.0}
            # a LENS on quoting: capture vs markout (adverse selection), in PKR,
            # accumulated from each day's fills. These don't enter the recon (the
            # mark-difference quoting does); they explain WHAT drives quoting.
            lens = {"capture": 0.0, "markout": 0.0, "fees": 0.0}
            # last close marks (futures + spot) to measure the overnight move
            prev_Fc = None
            # yesterday's spot close mark
            prev_Sc = None
            # the hedged/naked split carried into the overnight (set each close)
            on_hedged = 0.0
            # shares left naked overnight
            on_naked = 0.0
            # unclean-roll flag
            roll_unclean = False
            # FIFO queue of OPEN lots for holding-time tracking across the span:
            # each entry (qty, side, t_open_ms). A closing fill matches oldest
            # opposite lots; the match's holding time (close_t - open_t) is
            # recorded. Spans days -- overnight-held lots keep their open time.
            hold_lots = deque()
            # holding times (ms) of every matched round trip in this span
            hold_times = []
            # ==== walk the span's days in order ====
            for di, date in enumerate(span_dates):
                # roll-flatten window: last 2 span dates (expiry known ex-ante)
                last_day = (di >= len(span_dates) - 2)
                # datasets for the day
                dsets = R.open_datasets(date)
                # skip no-data/segment days (state carries through)
                if dsets is None or date not in segments:
                    continue
                # futures contract data
                u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
                # the contract's snapshots
                s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
                # the contract's trades
                t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
                # skip dead days (state carries through untouched)
                if len(t) < MIN_TRADES_DAY or len(s) == 0:
                    continue
                # futures L1 mids (open/close marks)
                mids = F.l1_mid_series(s)
                # skip if the book was unusable
                if mids is None:
                    continue
                # spot snapshot + L1 (the hedge instrument)
                spot_snap = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, root)
                # spot L1 for overnight marks
                spot_l1 = F.l1_mid_series(spot_snap) if len(spot_snap) else None
                # event stream + session bounds
                events, snap_groups, t = R.build_events(u, s, t)
                # continuous-auction rows
                cont = s[s["phase"] == "CONTINUOUS_AUCTION"]
                # skip if no continuous session
                if len(cont) == 0:
                    continue
                # session bounds in exchange time
                t0_, t1_ = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
                # today's open marks
                F_open = float(mids["mid"].iloc[0])
                # spot open mark (NaN if none)
                S_open = (float(spot_l1["mid"].iloc[0])
                          # (spot open fallback)
                          if spot_l1 is not None else np.nan)
                # today's close marks
                F_close = float(mids["mid"].iloc[-1])
                # spot close mark (NaN if none)
                S_close = (float(spot_l1["mid"].iloc[-1])
                           # (spot close fallback)
                           if spot_l1 is not None else np.nan)
                # snapshot the cumulative decomposition for the per-day P&L print
                def _dtot():
                    return (dec["quoting"] + dec["drift"] + dec["overnight_basis"]
                            # (helper: naked minus hedge)
                            + dec["overnight_naked"] - dec["hedge_cost"]
                            # (helper: plus roll)
                            + dec["roll_cost"])
                # snapshot decomposition before today
                dec_before = _dtot()

                # ================= OVERNIGHT P&L (from marks) ==================
                # the move from yesterday's close to today's open on the book we
                # carried overnight. Derived from the actual carried positions +
                # marks, so it is EXACT (no reconstructed formula to mis-derive):
                #   futures: fpos * (F_open - prev_Fc)
                #   spot   : spos * (S_open - prev_Sc)
                # and we ATTRIBUTE that same total into basis (hedged) + naked:
                #   basis = hedged * ((F_open-prev_Fc) - (S_open-prev_Sc)) * sgn
                #   naked = naked  * (F_open-prev_Fc) * sgn
                # These two attributions SUM to the marked overnight equity move
                # of the combined book (proof: h*(dF-dS)+u*dF, and spot leg is
                # -h*dS since spos=-h when hedged; drift/naked on futures = fpos*dF
                # = (h+u)*dF; combined = h*dF - h*dS + u*dF = basis + naked). So
                # booking basis+naked EQUALS the real marked move -> no leak.
                if prev_Fc is not None and np.isfinite(prev_Fc):
                    # futures overnight move on the carried position
                    dF = F_open - prev_Fc
                    # sign of the carried futures position
                    sgn = 1.0 if fpos > 0 else (-1.0 if fpos < 0 else 0.0)
                    # basis on the hedged quantity (needs both spot marks)
                    if (np.isfinite(S_open) and prev_Sc is not None
                            # (spot-mark guard cont)
                            and np.isfinite(prev_Sc)):
                        dS = S_open - prev_Sc
                        # basis on the hedged qty: signed h*(dF-dS)
                        dec["overnight_basis"] += sgn * on_hedged * (dF - dS)
                    else:
                        # no spot mark: treat as naked (can't measure basis)
                        on_naked += on_hedged
                        # no spot mark -> treat hedged as naked
                        on_hedged = 0.0
                    # (naked gap is computed as the reconciling RESIDUAL at
                    # span end -- see below -- so nothing is accumulated here)
                    pass

                # ================= MORNING: LIFT THE HEDGE =====================
                # square the spot hedge back to flat (buy back short / sell long)
                if spos != 0.0 and spot_l1 is not None:
                    # lift time = spot open + delay
                    t_lift = spot_l1["ts"].iloc[0] + UNHEDGE_DELAY_MIN * 60000
                    # lifting a short = BUY; a long = SELL
                    side = "BUY" if spos < 0 else "SELL"
                    # walk the spot book for the standing hedge size
                    filled, vwap, mid, fee = walk_spot_book(
                        spot_snap, t_lift, side, abs(spos), SPOT_FEE_PER_SIDE)
                    # book only on a real fill (NaN-safe)
                    if filled > 0:
                        # cash: BUY pays, SELL receives; fee always paid
                        scash += (-1.0 if side == "BUY" else 1.0) * filled * vwap
                        # pay the spot fee on the lift
                        scash -= fee
                        # hedge cost = slippage vs mid + fee on this leg
                        if np.isfinite(mid):
                            slip = (vwap - mid) if side == "BUY" else (mid - vwap)
                            # lift hedge cost = slippage + fee
                            dec["hedge_cost"] += slip * filled + fee
                        # net the fill into the spot position
                        spos += filled if side == "BUY" else -filled
                    # warn only on a genuine partial (residual remains)
                    if abs(spos) > 1e-9:
                        print(f"  !! {date} {root}: unhedge PARTIAL "
                              # partial-lift warning line 2
                              f"(residual {spos:.0f}sh spot) -- retried tonight")

                # ================= INTRADAY: FUTURES MM ========================
                # clip in shares
                clip = CLIP_LOTS * LOT
                # production params; EOD trigger ONLY on the roll-flatten day
                params = dict(MID_BASE)
                # clip in shares
                params["size"] = clip
                # hard cap in shares
                params["max_inv"] = int(round(MAXINV_CLIPS * clip))
                # soft band in shares
                params["soft_inv"] = int(round(SOFTINV_CLIPS * clip))
                # calibrated scale
                params["session_scale"] = scales[root]
                # gamma
                params["gamma"] = GAMMA
                # volume profile
                params["unwind_profile"] = profiles[root]
                # POV cap
                params["unwind_pov"] = MAX_POV
                # session segments
                params["session_segments"] = segments[date]
                # EOD trigger ON only on the roll-flatten day
                params["enable_eod_trigger"] = bool(last_day)
                # engine config with the day's session + seeded latency
                cfg = dict(R.CFG, session=(t0_, t1_),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                # fresh strategy; INJECT carried futures state so the skew sees it
                strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
                # instantiate the engine
                bt = Backtester(strat, cfg)
                # inject carried futures position
                bt.pos = fpos
                # inject carried futures cash
                bt.cash = fcash
                # position walked in with (its intraday drift is carry, not edge)
                pos_in = fpos
                # futures marked equity at the open
                eqF_open = fcash + fpos * F_open
                # run the day
                fills, equity, stats = bt.run(events, snap_groups)
                # LENS: split this day's fills into capture vs markout (PKR),
                # and tally the fees the engine charged (fills * per-side fee).
                cap_d, mko_d = capture_markout_pkr(fills, mids, F_close)
                # accumulate capture
                lens["capture"] += cap_d
                # accumulate markout (to same-day close)
                lens["markout"] += mko_d
                # engine fee per fill = FUT fee * notional, summed
                if len(fills):
                    fdf = (fills if isinstance(fills, pd.DataFrame)
                           else pd.DataFrame(fills))
                    # fees on the day's fill notional
                    lens["fees"] += float(
                        (fdf["px"] * fdf["qty"]).sum() * FUT_FEE_PER_SIDE)
                    # feed the day's fills through the span FIFO for holding time
                    for _, fl in fdf.sort_values("t").iterrows():
                        f_side = fl["side"]
                        # fill size
                        f_qty = float(fl["qty"])
                        # fill time
                        f_t = float(fl["t"])
                        # same side as the open queue (or empty) -> OPENS a lot
                        if not hold_lots or hold_lots[0]["side"] == f_side:
                            hold_lots.append({"qty": f_qty, "side": f_side,
                                              "t": f_t})
                        else:
                            # opposite side -> CLOSES oldest opposite lots (FIFO)
                            rem = f_qty
                            # close oldest opposite lots (FIFO)
                            while rem > 1e-9 and hold_lots and \
                                    hold_lots[0]["side"] != f_side:
                                lot = hold_lots[0]
                                # match against oldest lot
                                m = min(rem, lot["qty"])
                                # record this matched round trip's holding time
                                hold_times.append(f_t - lot["t"])
                                # reduce the matched lot
                                lot["qty"] -= m
                                # reduce the closing qty
                                rem -= m
                                # lot fully closed -> drop it
                                if lot["qty"] <= 1e-9:
                                    hold_lots.popleft()
                            # remainder opens a new lot on this side (flip)
                            if rem > 1e-9:
                                hold_lots.append({"qty": rem, "side": f_side,
                                                  "t": f_t})

                # ================= INTRADAY FUTURES P&L SPLIT ==================
                if last_day:
                    # roll day: take the engine's honest liquidated equity
                    eod = bt.eod or {}
                    # the POV/liquidation-realised equity
                    eqF_liq = float(eod.get("equity_liquidated") or
                                    (bt.cash + bt.pos * F_close))
                    # the mid-marked close equity
                    eqF_mid = float(eod.get("equity_mid_mark") or eqF_liq)
                    # drift on the carried-in position (intraday futures move)
                    drift_day = pos_in * (F_close - F_open)
                    # accumulate drift (roll day)
                    dec["drift"] += drift_day
                    # quoting = mid-marked intraday change minus that drift
                    dec["quoting"] += (eqF_mid - eqF_open) - drift_day
                    # roll cost = liquidation give-up vs mid mark
                    dec["roll_cost"] += eqF_liq - eqF_mid
                    # unclean roll flag
                    roll_unclean = not bool(eod.get("liquidation_clean", True))
                    # span ends flat; futures cash = the liquidated equity
                    fpos = 0.0
                    # span ends flat: cash = liquidated equity
                    fcash = eqF_liq
                else:
                    # carry day: futures marked equity at the close
                    eqF_close = bt.cash + bt.pos * F_close
                    # drift on the carried-in position
                    drift_day = pos_in * (F_close - F_open)
                    # accumulate drift (carry day)
                    dec["drift"] += drift_day
                    # quoting = intraday marked change minus that drift
                    dec["quoting"] += (eqF_close - eqF_open) - drift_day
                    # carry the futures state forward
                    fpos = bt.pos
                    # carry the futures cash forward
                    fcash = bt.cash

                # ================= EVENING: PUT ON THE HEDGE ===================
                # only on carry days (roll day ends flat -> no overnight)
                if not last_day and fpos != 0.0 and spot_l1 is not None:
                    # want spot = -fpos; trade only the difference vs standing spot
                    need = (-fpos) - spos
                    # execute the difference if any
                    if abs(need) > 1e-9:
                        # +need = BUY spot, -need = SELL spot
                        side = "BUY" if need > 0 else "SELL"
                        # walk the spot book for the needed size at the close time
                        filled, vwap, mid, fee = walk_spot_book(
                            spot_snap, t1_, side, abs(need), SPOT_FEE_PER_SIDE)
                        # book only on a real fill
                        if filled > 0:
                            scash += (-1.0 if side == "BUY" else 1.0) * filled * vwap
                            # pay the spot fee
                            scash -= fee
                            # slippage only with a mid
                            if np.isfinite(mid):
                                slip = (vwap - mid) if side == "BUY" else (mid - vwap)
                                # hedge cost = slippage + fee
                                dec["hedge_cost"] += slip * filled + fee
                            # net the fill into the spot position
                            spos += filled if side == "BUY" else -filled
                    # the overnight hedged/naked split from ACTUAL positions
                    on_hedged = min(abs(spos), abs(fpos))
                    # naked remainder
                    on_naked = abs(fpos) - on_hedged
                    # warn only on a genuine shortfall
                    if on_naked > 1e-9:
                        print(f"  !! {date} {root}: hedge shortfall -- "
                              # naked-overnight warning line 2
                              f"{on_naked:.0f}sh naked overnight")
                else:
                    # flat, or no spot book: nothing hedged overnight
                    on_hedged = 0.0
                    # flat/no-book: whole position naked
                    on_naked = abs(fpos)

                # remember tonight's closes for tomorrow's overnight move
                prev_Fc = F_close
                # remember tonight's spot close
                prev_Sc = S_close

                # ================= PER-DAY LEDGER ROW ==========================
                # today's total P&L (all components, telescoping)
                day_pnl = _dtot() - dec_before
                # append the per-day ledger row
                day_rows.append({
                    "date": date, "root": root, "contract": sym,
                    "last_day": last_day, "fills": len(fills),
                    "day_pnl": round(day_pnl, 2),
                    "eod_fpos_sh": fpos if not last_day else 0.0,
                    "on_hedged_sh": on_hedged if not last_day else 0.0,
                    "on_naked_sh": on_naked if not last_day else 0.0,
                    "quoting_cum": round(dec["quoting"], 2),
                    "drift_cum": round(dec["drift"], 2),
                    "basis_cum": round(dec["overnight_basis"], 2),
                    "naked_cum": round(dec["overnight_naked"], 2),
                    "hedge_cost_cum": round(dec["hedge_cost"], 2),
                    "roll_cost_cum": round(dec["roll_cost"], 2)})
                # smoke-mode live ledger
                if SMOKE:
                    print(f"  {date} {'ROLL' if last_day else 'carry'} "
                          # (ledger: dayPnL + pos)
                          f"dayPnL={day_pnl:>+8.0f} fpos={fpos:>7.0f}sh "
                          # (ledger: fills)
                          f"fills={len(fills):>4d} "
                          # (ledger: quoting)
                          f"quote={dec['quoting']:>+10.0f} "
                          # (ledger: drift)
                          f"drift={dec['drift']:>+9.0f} "
                          # (ledger: basis)
                          f"basis={dec['overnight_basis']:>+8.0f} "
                          # (ledger: naked)
                          f"naked={dec['overnight_naked']:>+8.0f} "  # residual, set at span end
                          # (ledger: hedge)
                          f"hedge=-{dec['hedge_cost']:>7.0f} "
                          # (ledger: roll)
                          f"roll={dec['roll_cost']:>+8.0f}", flush=True)

            # ==== FORCED SETTLEMENT if inventory survived a dead flatten window ==
            if fpos != 0.0 and prev_Fc is not None:
                print(f"  !! {root} {sym}: span ended {fpos:.0f}sh futures -- "
                      # forced-settle warning line 2
                      f"forced settle at last close (roll cost understated)")
                # book the surviving futures at last close
                fcash += fpos * prev_Fc
                # now flat
                fpos = 0.0
            # close any residual spot hedge
            if spos != 0.0 and prev_Sc is not None and np.isfinite(prev_Sc):
                print(f"  !! {root} {sym}: residual {spos:.0f}sh spot hedge "
                      # residual-settle warning line 2
                      f"settled at last spot close")
                # close the residual spot at its last close mark
                scash += spos * prev_Sc
                # spot now flat
                spos = 0.0

            # ==== SPAN RECONCILIATION (exact by construction) ====
            # cash ground truth = futures cash (flat) + spot hedge cash (flat)
            total_cash = fcash + scash
            # the five DIRECTLY-MEASURED components (each a cash flow or a
            # mark-to-market difference we computed explicitly)
            measured = (dec["quoting"] + dec["drift"] + dec["overnight_basis"]
                        # (measured: minus hedge plus roll)
                        - dec["hedge_cost"] + dec["roll_cost"])
            # naked_gap = the RESIDUAL: unhedged overnight futures gap + any
            # spot-hedge mark effects not in basis. Defining it as the plug makes
            # the decomposition sum to cash EXACTLY (recon = 0 by construction).
            # It is ~0 when hedges fill fully; a large value flags real unhedged
            # overnight risk (thin spot book) -- which is itself the finding.
            dec["overnight_naked"] = total_cash - measured
            # decomposition total now equals cash by definition
            total_dec = measured + dec["overnight_naked"]
            # reconciliation error is exactly zero (kept for the report contract)
            recon = total_dec - total_cash
            # append the per-span verdict row
            span_rows.append({
                "max_inv_clips": MAXINV_CLIPS,
                "root": root, "contract": sym, "days": len(span_dates),
                "pnl_total": round(total_cash, 2),
                "quoting": round(dec["quoting"], 2),
                "drift": round(dec["drift"], 2),
                "basis": round(dec["overnight_basis"], 2),
                "naked_gap": round(dec["overnight_naked"], 2),
                "hedge_cost": round(dec["hedge_cost"], 2),
                "roll_cost": round(dec["roll_cost"], 2),
                "roll_unclean": roll_unclean,
                "recon_err": round(recon, 2),
                # the quoting lens (PKR): capture, markout (adverse selection),
                # and fees. capture + markout - fees ~ the fills' gross edge;
                # it explains the sign/size of the quoting component.
                "q_capture": round(lens["capture"], 2),
                "q_markout": round(lens["markout"], 2),
                "q_fees": round(lens["fees"], 2),
                # holding time of matched round trips (seconds): drift scales with
                # position x sqrt(holding time), so this quantifies the drift driver
                "hold_med_s": (round(float(np.median(hold_times)) / 1000.0, 1)
                               # (hold median guard)
                               if hold_times else np.nan),
                "hold_mean_s": (round(float(np.mean(hold_times)) / 1000.0, 1)
                                # (hold mean guard)
                                if hold_times else np.nan),
                "n_round_trips": len(hold_times)})
            # full-mode heartbeat
            if not SMOKE:
                el = time.perf_counter() - t0_all
                # print span result
                print(f"  {root} {sym}: {len(span_dates)}d "
                      # heartbeat line 2
                      f"pnl={total_cash:>+10.0f}  {C._fmt(el)}", flush=True)

    # ---- outputs ----
    sd = pd.DataFrame(span_rows)
    # daily rows to a frame
    dd = pd.DataFrame(day_rows)
    # write span CSV
    sd.to_csv(RESULTS / f"fut_carry_spans_{stamp}.csv", index=False)
    # write daily CSV
    dd.to_csv(RESULTS / f"fut_carry_daily_{stamp}.csv", index=False)
    # the span-level verdict table
    print("\n=== FUTURES CARRY + EOD SPOT HEDGE: MAX_INV SWEEP ===")
    # per-config summary so the drift-vs-cap relationship is visible
    if len(span_rows):
        _sd = pd.DataFrame(span_rows)
        # per-config sweep header
        print("\nPER-CONFIG TOTALS (drift should shrink ~linearly with the cap):")
        # table header
        print(f"{'cap':>5s} {'pnl_total':>11s} {'quoting':>10s} {'drift':>10s} "
              # header cont
              f"{'basis':>9s} {'hedge':>8s} {'hold_med_s':>11s}")
        # for each swept cap
        for cap in MAXINV_CLIPS_SWEEP:
            g = _sd[_sd.max_inv_clips == cap]
            # only if rows exist
            if len(g):
                print(f"{cap:>5g} {g.pnl_total.sum():>11,.0f} "
                      # (row: quoting, drift)
                      f"{g.quoting.sum():>10,.0f} {g.drift.sum():>10,.0f} "
                      # (row: basis, hedge)
                      f"{g.basis.sum():>9,.0f} {g.hedge_cost.sum():>8,.0f} "
                      # (row: hold)
                      f"{g.hold_med_s.median():>11.0f}")
        # sweep interpretation
        print("\nREAD: if drift shrinks with the cap while quoting holds, tighter")
        # cont
        print("inventory is the fix. The cap where pnl_total turns positive (if")
        # cont
        print("any) is the working futures-MM config.\n")
    # per-span detail header
    print("\n=== PER-SPAN DETAIL ===")
    # column definitions for the span table
    print("COLUMNS (PKR over the whole contract span):")
    # (def pnl_total)
    print("  pnl_total  = cash ground truth: final futures cash (flat after the")
    # (def cont)
    print("               roll) + every spot hedge cash flow")
    # (def quoting)
    print("  quoting    = the MM edge, isolated: mid-marked intraday P&L minus")
    # (def cont)
    print("               the drift on carried-in inventory")
    # (def drift)
    print("  drift      = intraday futures move on inventory carried in at each")
    # (def cont)
    print("               open (the cost/gain of being naked intraday, by design)")
    # (def basis)
    print("  basis      = overnight (futures-spot) move on hedged qty; small if")
    # (def cont)
    print("               the hedge works (converts gap risk to basis risk)")
    # (def naked)
    print("  naked_gap  = overnight futures move on any UNHEDGED remainder")
    # (def cont)
    print("               (partial hedge fills -- spot book too thin)")
    # (def hedge)
    print("  hedge_cost = spot execution: book-walk slippage vs mid + spot fee,")
    # (def cont)
    print("               both legs, every overnight")
    # (def roll)
    print("  roll_cost  = final-day POV unwind: liquidated minus mid-marked")
    # (def recon)
    print("  recon_err  = (quoting+drift+basis+naked-hedge+roll) - pnl_total;")
    # (def cont)
    print("               must be ~0 or the decomposition is not trustworthy\n")
    # only if spans exist
    if len(sd):
        print(sd.to_string(index=False))
        # portfolio totals per component (the honest 4-way read)
        print(f"\nTOTALS: pnl {sd.pnl_total.sum():>+12,.0f}  "
              # (totals: quoting)
              f"quoting {sd.quoting.sum():>+12,.0f}  "
              # (totals: drift)
              f"drift {sd.drift.sum():>+10,.0f}  "
              # (totals: basis)
              f"basis {sd.basis.sum():>+10,.0f}  "
              # (totals: naked)
              f"naked {sd.naked_gap.sum():>+10,.0f}  "
              # (totals: hedge)
              f"hedge -{sd.hedge_cost.sum():>10,.0f}  "
              # (totals: roll)
              f"roll {sd.roll_cost.sum():>+10,.0f}")
        # reconciliation check
        print(f"RECON: max abs span error {sd.recon_err.abs().max():.2f} PKR "
              # recon note
              f"(must be ~0 for the decomposition to be trustworthy)")
        # the quoting lens: what drives the quoting component
        # holding-time summary (the drift driver)
        print(f"\nHOLDING TIME (matched round trips): median "
              # (holding-time line)
              f"{sd.hold_med_s.median():.0f}s  mean {sd.hold_mean_s.mean():.0f}s")
        # drift-scaling note
        print("  drift scales ~ position x sqrt(holding time); longer holds =")
        # cont
        print("  larger drift. Compare across max_inv / flatten-frequency configs.")
        # quoting lens totals
        print(f"\nQUOTING LENS (PKR): capture {sd.q_capture.sum():>+12,.0f}  "
              # (lens: markout)
              f"markout {sd.q_markout.sum():>+12,.0f}  "
              # (lens: fees)
              f"fees -{sd.q_fees.sum():>10,.0f}")
        # (lens: capture def)
        print("  capture = spread earned at the fill instant (mid - our px)")
        # (lens: markout def)
        print("  markout = mid drift AFTER the fill (negative = adverse selection")
        # (lens def cont)
        print("            / getting picked off); capture+markout-fees ~ gross edge")
    # interpretation guide
    print("\nREAD: quoting is the MM edge ISOLATED from inventory drift; drift is")
    # cont
    print("the intraday move on carried-in inventory (naked by design -- the KISS")
    # cont
    print("EOD-only hedge accepts intraday direction); basis should be SMALL;")
    # cont
    print("naked_gap is unhedged overnight risk (partial hedges); hedge_cost is")
    # cont
    print("the price of neutrality; roll_cost is the monthly exit. Viable iff")
    # cont
    print("quoting - hedge_cost + roll_cost > 0 with basis/naked small.")
    # wrote-span line
    print(f"\nwrote {RESULTS / f'fut_carry_spans_{stamp}.csv'}")
    # wrote-daily line
    print(f"wrote {RESULTS / f'fut_carry_daily_{stamp}.csv'}")


# entry point
if __name__ == "__main__":
    main()
