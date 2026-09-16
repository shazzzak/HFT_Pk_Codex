# ============================================================================
# mm_harness.py -- SHARED PRODUCTION HARNESS for all MM backtest analysis.
# ============================================================================
# Single source of truth for the scaffolding every sweep/analysis was COPYING:
#   * calibration loaders (scales, profiles, windows, segments) -- defined ONCE
#   * the per-symbol-day backtest DRIVER -- one code path, so results cannot
#     diverge between scripts (the bug class behind the futures-scorer and the
#     decomposition sign-flip: two paths that should be identical but weren't)
#   * FIFO inventory-matched fill attribution (open-bucket) -- one implementation
#   * the institutional MM STATS PANEL (OTR, cancels, drawdowns, Sharpe/Sortino,
#     inventory risk) -- computed from data the engine already emits
#
# Every runner becomes a thin consumer: import mm_harness, specify config, call
# run_symbol_day() and the analysis helpers. Nothing re-declares this scaffolding.
# ============================================================================

# filesystem paths
from pathlib import Path
# double-ended queue for the FIFO open-lot inventory queue
from collections import deque
# wall-clock timing for the pre-pass heartbeat
import time
# numeric arrays
import numpy as np
# dataframes
import pandas as pd
# the driver module: loaders, event builder, CFG, REQ specs, MICRO_PARAMS
import run_legacy_mm as R
# the frozen engine: backtester, seeded latency, naive strategy, the fee function
from mm_backtest import Backtester, LatencyModel, NaiveSymmetricMM, fee_for
# the production strategy
from micro_mm import MicrostructureMM
# the VALIDATED per-fill scorer module (score_bps + _fmt) -- reused, not rewritten
import confirm_micro_vs_naive as C
# the tested no-leak fill-context join (used by score_bps internally)
import persist_fills as PF

# ---------------------------------------------------------------------------
# PATHS -- single source of truth is config_pk (edit ONE file to move machines).
# The loader module's own PARSED_ROOT constant is stale/relative, so mm_harness
# imports the canonical roots here and pushes PARSED_ROOT onto R for every runner.
# ---------------------------------------------------------------------------
# import the canonical roots from the central config; fail LOUD if unavailable
# (a wrong/stale store that runs silently is worse than a clear crash).
try:
    # PARSED_ROOT = raw store; RESULTS_ROOT = where tools write
    from config_pk import PARSED_ROOT as _PARSED, RESULTS_ROOT as _RESULTS
# no config_pk on the path -> stop with an explicit message
except Exception as _e:
    # re-raise so the fix (run from existing_mm_live/ or add it to sys.path) is obvious
    raise ImportError("mm_harness: could not import paths from config_pk (%r)" % _e)
# push the raw-store root onto the loader module so every runner inherits it
R.PARSED_ROOT = _PARSED
# results root (from config_pk, not hardcoded)
RESULTS = _RESULTS
# spot feature store (fill-time context + the 5s net_bps lens)
FS_ROOT = RESULTS / "feature_store"

# the four session buckets, in chronological order
BUCKETS = ("first15", "middle", "preclose45", "last15")
# reuse the compact mm:ss formatter from the validated module (no re-definition)
_fmt = C._fmt


# ===========================================================================
# 1. CALIBRATION LOADERS  (defined ONCE; every runner imported copies of these)
# ===========================================================================
def newest(pattern):
    # all results files matching the glob, sorted so the newest stamp is last
    cands = sorted(RESULTS.glob(pattern))
    # a missing calibration file is a hard stop, never a silent default
    if not cands:
        raise SystemExit(f"missing {pattern}")
    # the most recent matching file
    return cands[-1]


def _load_table(pattern, cols):
    # read the newest CSV matching the pattern
    df = pd.read_csv(newest(pattern))
    # keep only rows marked ok when the file carries a note column
    ok = df[df["note"] == "ok"] if "note" in df.columns else df
    # map symbol -> tuple of the requested columns
    return {r["symbol"]: tuple(r[c] for c in cols) for _, r in ok.iterrows()}


def load_scales():
    # per-symbol back-solved session_scale, unpacked from the 1-tuple
    return {k: v[0] for k, v in _load_table("session_scales_*.csv",
                                            ["session_scale"]).items()}


def load_profiles():
    # per-symbol 4-bucket volume profile (shares/min in each bucket)
    return _load_table("volume_profile_*.csv",
                       ["vol_first15", "vol_middle", "vol_preclose45",
                        "vol_last15"])


def load_windows():
    # per-symbol EOD ramp start / cliff minutes
    return _load_table("time_windows_*.csv",
                       ["eod_ramp_start_min", "eod_cliff_min"])


def load_segments():
    # read the newest per-date session-segments CSV
    df = pd.read_csv(newest("session_segments_*.csv"))
    # output dict: date -> list of (start_ms, end_ms) continuous spans
    out = {}
    # each row carries one date's segments as "s:e;s:e" text
    for _, r in df.iterrows():
        # split the text into (start, end) integer tuples
        out[r["date"]] = [tuple(int(x) for x in p.split(":"))
                          for p in r["segments"].split(";")]
    # the full date -> segments map
    return out


def trailing_median(stats_sym, all_dates, date, ndays):
    # values strictly BEFORE the target date (walk-forward: current day excluded)
    prior = [stats_sym[str(d)] for d in all_dates
             if str(d) < str(date) and str(d) in stats_sym]
    # not enough history -> caller skips the day rather than trading a guess
    if len(prior) < ndays:
        return None
    # median of the last ndays prior values
    return float(np.median(prior[-ndays:]))


def trailing_median_trade_size(all_dates, names, ndays, heartbeat=50):
    # the standard pre-pass every sweep runs: {sym: {date: median trade qty}}
    tstats = {s: {} for s in names}
    # pre-pass timer for the heartbeat
    t0 = time.perf_counter()
    # walk every date once
    for i, date in enumerate(all_dates, 1):
        # open the date's datasets once (shared across symbols)
        dsets = R.open_datasets(date)
        # missing partition -> skip the date
        if dsets is None:
            continue
        # each symbol's median trade size that day
        for s in names:
            # the day's trades for this symbol
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, s)
            # record the median qty when the symbol traded
            if len(t):
                tstats[s][str(date)] = float(t["qty"].median())
        # heartbeat every `heartbeat` dates and on the final date
        if heartbeat and (i % heartbeat == 0 or i == len(all_dates)):
            print(f"  pre-pass {i}/{len(all_dates)}  "
                  f"{_fmt(time.perf_counter()-t0)}", flush=True)
    # the completed per-symbol per-day median map
    return tstats


# ===========================================================================
# 2. THE PER-SYMBOL-DAY BACKTEST DRIVER  (one code path for every runner)
# ===========================================================================
class DayResult:
    # a thin container for one symbol-day's backtest outputs + derived stats
    def __init__(self, symbol, date, fills, equity, stats, eod, session,
                 fs_day=None, order_log=None):
        # the symbol this result belongs to
        self.symbol = symbol
        # the trading date
        self.date = date
        # fills as a DataFrame (t, side, px, qty, reason, window, bucket)
        self.fills = (fills if isinstance(fills, pd.DataFrame)
                      else pd.DataFrame(fills))
        # per-event equity path the engine recorded (for drawdown/inventory)
        self.equity = (equity if isinstance(equity, pd.DataFrame)
                       else pd.DataFrame(equity))
        # engine + strategy counters (n_orders_sent, n_cancels, triggers, ...)
        self.stats = stats or {}
        # end-of-day liquidation report dict (equity_liquidated, liq_vwap, ...)
        self.eod = eod
        # (t0, t1) continuous-session bounds in exchange ms
        self.session = session
        # cached feature-store day for the 5s net_bps lens (optional)
        self.fs_day = fs_day
        # per-order lifecycle log from the instrumented engine (quote uptime,
        # time-to-fill, message rate); None when the engine wasn't instrumented
        self.order_log = (order_log if isinstance(order_log, pd.DataFrame)
                          else (pd.DataFrame(order_log)
                                if order_log is not None else None))

    def pnl(self):
        # no EOD report -> no headline P&L for the day
        if self.eod is None:
            return None
        # the true post-liquidation daily P&L (may be None on broken closes)
        return self.eod.get("equity_liquidated")


def build_micro_params(clip, scale, profile, window, segments,
                       overrides=None, bucket_mult=None):
    # start from the module's canonical micro parameter set
    p = dict(R.MICRO_PARAMS)
    # locked production defaults shared by every experiment in the project
    p.update(dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                  enable_eod_trigger=True, enable_lock_trigger=True))
    # the quote clip (shares per quote)
    p["size"] = clip
    # inventory ceiling at the production ratio (10 clips)
    p["max_inv"] = int(round(10.0 * clip))
    # soft-inventory band at the production ratio (3 clips)
    p["soft_inv"] = int(round(3.0 * clip))
    # per-symbol back-solved inventory-skew scale
    p["session_scale"] = scale
    # EOD ramp start (minutes before close)
    p["eod_ramp_start_min"] = window[0]
    # EOD cliff (minutes before close)
    p["eod_cliff_min"] = window[1]
    # the 4-bucket volume profile driving the POV unwind
    p["unwind_profile"] = profile
    # participation-of-volume cap for the unwind
    p["unwind_pov"] = 0.10
    # the day's continuous-session segments (break-aware)
    p["session_segments"] = segments
    # optional per-bucket clip multipliers (experiments only)
    if bucket_mult is not None:
        p["bucket_mult"] = bucket_mult
    # caller-specific overrides applied last (gates, sweeps, etc.)
    if overrides:
        p.update(overrides)
    # the assembled kwargs for MicrostructureMM
    return p


def run_symbol_day(date, sym, dsets, params, want_fs=False, cfg_overrides=None):
    # THE single backtest path every runner calls: identical loading, session
    # bounds, latency seed, and fees everywhere, so results cannot diverge.
    #
    # cfg_overrides (added 2026-09-16) are ENGINE settings, as distinct from
    # `params`, which are STRATEGY settings. Needed because the short-sale and
    # CFO work put some switches on the engine rather than the strategy:
    #   opening_inventory        shares held at the open (long_buffer policy)
    #   use_cfo                  one AMEND message instead of CANCEL + NEW
    #   cfo_*_keeps_priority     the per-venue amendment priority rules
    # Defaults to None, which reproduces R.CFG exactly, so every existing
    # caller is unaffected.
    # the day's order-book updates for this symbol
    u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
    # the day's book snapshots for this symbol
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
    # the day's trades for this symbol
    t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
    # unrunnable without both a book and trades
    if len(t) == 0 or len(s) == 0:
        return None
    # the merged event stream (adds ts_exch to s in place)
    events, snap_groups, t = R.build_events(u, s, t)
    # phase-based session window (robust to Ramadan / Friday split / post-close)
    cont = s[s["phase"] == "CONTINUOUS_AUCTION"]
    # no continuous phase -> fully halted / no-data day
    if len(cont) == 0:
        return None
    # t0 = continuous open, t1 = continuous close (the true bell)
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
    # engine config: same CFG, session bounds, seeded latency (reproducible)
    cfg = dict(R.CFG, session=(t0, t1),
               latency_model=LatencyModel(seed=R.LATENCY_SEED))
    # engine-level overrides, applied last so a sweep can change the mechanism
    # without touching the module-level R.CFG that every other runner shares
    if cfg_overrides:
        cfg.update(cfg_overrides)
    # build the production strategy with the assembled params
    strat = MicrostructureMM(session_ms=(t0, t1), **params)
    # the engine instance for this run
    bt = Backtester(strat, cfg)
    # run the backtest over the event stream
    fills, equity, stats = bt.run(events, snap_groups)
    # merged counters: engine stats authoritative, strategy-only keys added
    merged = dict(stats or {})
    # the strategy's own monitoring counters (triggers, suppression, ...)
    sstats = getattr(strat, "stats", {})
    # add strategy keys the engine doesn't already carry
    for k, v in sstats.items():
        # never overwrite an engine counter with a strategy one
        if k not in merged:
            merged[k] = v
    # feature-store day loaded only when the caller wants the net_bps lens
    fs_day = None
    # optional feature-store read
    if want_fs:
        # the symbol-day's feature-store partition path
        fs_path = FS_ROOT / sym / f"date={date}.parquet"
        # load the context columns only when the partition exists
        if fs_path.exists():
            fs_day = pd.read_parquet(
                fs_path, columns=["ts_exch", "mid", "spread_bps", "obi_1",
                                  "toxicity", "realized_vol_bps"])
    # the per-order lifecycle log if the engine was instrumented (else None)
    olog = getattr(bt, "order_log", None)
    # the packaged symbol-day result (with the lifecycle log stitched in)
    return DayResult(sym, date, fills, equity, merged, bt.eod, (t0, t1),
                     fs_day, order_log=olog)


# ===========================================================================
# 3. FIFO INVENTORY-MATCHED ATTRIBUTION  (open-bucket; one implementation)
# ===========================================================================
# Realized round-trip P&L attributed to the OPENING bucket (the bucket whose
# quoting took the risk). Measured to each fill's ACTUAL offset -- no fixed
# markout horizon. Residual EOD inventory closed at the true liquidation VWAP.
def fifo_attribution(fills, engine_pnl, liq_time, residual_hint=None):
    # FIFO inventory-matched attribution that RECONCILES to the engine's headline
    # P&L by construction, including on unclean-liquidation days.
    #
    # Method: match opposite-side fills FIFO and book each matched round trip's
    # realized P&L to the OPENING bucket. The residual (never-closed) inventory is
    # NOT priced by us -- instead the residual P&L is defined as
    #     residual_total = engine_pnl - realized_on_matched_pairs
    # and distributed to the residual lots' opening buckets pro-rata by their
    # opened notional. This makes sum(realized) == engine_pnl exactly, so the
    # 3% unclean-liquidation haircut the engine applied is attributed, not lost.
    # normalise input to records
    f = fills if isinstance(fills, list) else (
        fills.to_dict("records") if isinstance(fills, pd.DataFrame) else fills)
    # the FIFO open-lot queue (all entries share the current net side)
    open_lots = deque()
    # per-bucket accumulators
    per = {b: {"realized": 0.0, "opened_qty": 0.0, "opened_notional": 0.0,
               "holds": [], "fills": 0} for b in BUCKETS}
    # running total of realized P&L on MATCHED pairs only (residual added later)
    matched_realized = 0.0
    # process fills in stream order
    for fl in f:
        # unpack the fill
        side = fl["side"]
        px = float(fl["px"])
        qty = float(fl["qty"])
        b = fl.get("bucket", "middle")
        t = float(fl["t"])
        # count the fill in its bucket
        if b in per:
            per[b]["fills"] += 1
        # same side (or empty queue) -> OPENS inventory
        if not open_lots or open_lots[0]["side"] == side:
            # push the open lot
            open_lots.append({"qty": qty, "px": px, "bucket": b, "t": t,
                              "side": side})
            # track opened qty/notional (the residual-apportion weights)
            if b in per:
                per[b]["opened_qty"] += qty
                per[b]["opened_notional"] += qty * px
            # next fill
            continue
        # opposite side -> CLOSES open lots FIFO
        remaining = qty
        # match until consumed or the queue flips
        while remaining > 1e-9 and open_lots and open_lots[0]["side"] != side:
            # the oldest open lot
            lot = open_lots[0]
            # matched shares
            matched = min(remaining, lot["qty"])
            # sign-correct realized round trip on the matched shares
            if lot["side"] == "BUY":
                realized = (px - lot["px"]) * matched
            else:
                realized = (lot["px"] - px) * matched
            # both legs' fees on the matched shares
            realized -= fee_for(lot["px"], matched)
            realized -= fee_for(px, matched)
            # attribute the matched round trip to the OPENING bucket
            ob = lot["bucket"]
            # accumulate realized + holding time + the matched-total tally
            if ob in per:
                per[ob]["realized"] += realized
                per[ob]["holds"].append(t - lot["t"])
            matched_realized += realized
            # shrink lot + fill
            lot["qty"] -= matched
            remaining -= matched
            # drop consumed lots
            if lot["qty"] <= 1e-9:
                open_lots.popleft()
        # flip through zero: remainder opens the other side
        if remaining > 1e-9:
            # push the flipped open lot
            open_lots.append({"qty": remaining, "px": px, "bucket": b, "t": t,
                              "side": side})
            # opened tracking for the flipped remainder
            if b in per:
                per[b]["opened_qty"] += remaining
                per[b]["opened_notional"] += remaining * px
    # ---- residual (never-closed) inventory: reconcile to the engine ----
    # residual P&L is whatever the engine's headline P&L has that our matched
    # round trips do not -- i.e. the engine's liquidation cash + haircut penalty
    # on the shares that stayed open.
    residual_total = float(engine_pnl) - matched_realized
    # weight = each residual lot's opened notional (its share of the open risk)
    resid_notional = sum(lot["qty"] * lot["px"] for lot in open_lots)
    # distribute the residual P&L to the residual lots' opening buckets pro-rata
    for lot in open_lots:
        # this lot's share of the residual (by opened notional)
        w = (lot["qty"] * lot["px"] / resid_notional) if resid_notional > 0 else 0.0
        # the lot's residual P&L
        share = residual_total * w
        # attribute to the opening bucket, held to the liquidation time
        ob = lot["bucket"]
        # accumulate the residual share + the hold-to-liquidation time
        if ob in per:
            per[ob]["realized"] += share
            per[ob]["holds"].append(max(0.0, liq_time - lot["t"]))
    # the per-bucket attribution (sums to engine_pnl by construction)
    return per


def liquidation_price(dr):
    # no EOD report -> no residual to price
    if dr.eod is None:
        return 0.0
    # the engine's true liquidation VWAP (the book-walk price)
    lp = dr.eod.get("liq_vwap")
    # VWAP missing (flat day) -> fall back to the last recorded mid
    if lp is None or (isinstance(lp, float) and np.isnan(lp)):
        # last mid when the equity path has one
        if len(dr.equity) and "mid" in dr.equity.columns:
            return float(dr.equity["mid"].iloc[-1])
        # flat day with no equity path: price is irrelevant
        return 0.0
    # the true liquidation price
    return float(lp)


# ===========================================================================
# 4. INSTITUTIONAL STATS PANEL  (from data the engine already emits)
# ===========================================================================
def execution_stats(dr):
    # orders the engine sent (counted at the reconcile loop)
    n_orders = int(dr.stats.get("n_orders_sent", 0))
    # cancels that landed (counted at the cancel handler)
    n_cancels = int(dr.stats.get("n_cancels", 0))
    # fills booked
    n_fills = int(len(dr.fills))
    # the execution / exchange-compliance panel
    return {
        # raw counter: orders sent
        "n_orders_sent": n_orders,
        # raw counter: cancels landed
        "n_cancels": n_cancels,
        # raw counter: fills booked
        "n_fills": n_fills,
        # cancels that arrived after the order was already gone
        "stale_cancels": int(dr.stats.get("stale_cancels_ignored", 0)),
        # post-only rejections (would-cross orders)
        "rejected_crossing": int(dr.stats.get("rejected_crossing", 0)),
        # order-to-trade ratio (the compliance headline); inf if zero fills
        "otr": (n_orders / n_fills) if n_fills > 0 else np.inf,
        # cancel-to-trade ratio
        "cancel_to_trade": (n_cancels / n_fills) if n_fills > 0 else np.inf,
        # cancel rate: cancels per order sent (quote churn)
        "cancel_rate": (n_cancels / n_orders) if n_orders > 0 else np.nan,
        # fill ratio: fills per order sent
        "fill_ratio": (n_fills / n_orders) if n_orders > 0 else np.nan,
    }


def intraday_pnl_path(dr):
    # the engine's per-event equity path
    eq = dr.equity
    # preferred: the engine already recorded a marked equity column
    if len(eq) and "equity" in eq.columns:
        # times + the engine's own equity path
        return eq["t"].to_numpy(), eq["equity"].to_numpy()
    # else reconstruct: equity = cash + pos * mid when all three exist
    if len(eq) and {"mid"}.issubset(eq.columns) and "pos" in eq.columns \
            and "cash" in eq.columns:
        # mark-to-mid at every recorded event
        pnl = eq["cash"].to_numpy() + eq["pos"].to_numpy() * eq["mid"].to_numpy()
        # times + the reconstructed path
        return eq["t"].to_numpy(), pnl
    # last resort: cumulative signed cash from fills (coarse, no inventory mark)
    if len(dr.fills) == 0:
        # nothing at all to build a path from
        return np.array([]), np.array([])
    # fills in time order
    f = dr.fills.sort_values("t")
    # +1 buys, -1 sells
    sgn = np.where(f["side"].to_numpy() == "BUY", 1.0, -1.0)
    # cash moves opposite to position at the fill price
    cash = np.cumsum(-sgn * f["qty"].to_numpy() * f["px"].to_numpy())
    # times + the coarse cash path
    return f["t"].to_numpy(), cash


def max_drawdown(cum_pnl):
    # empty path -> no drawdown
    if len(cum_pnl) == 0:
        return 0.0
    # running peak of the cumulative P&L
    running_peak = np.maximum.accumulate(cum_pnl)
    # drawdown at every point (<= 0 everywhere)
    drawdown = cum_pnl - running_peak
    # the deepest peak-to-trough drop (most negative point)
    return float(drawdown.min())


def inventory_stats(dr):
    # the engine's equity path (carries pos when recorded)
    eq = dr.equity
    # preferred: position path straight from the equity series
    if len(eq) and "pos" in eq.columns and "t" in eq.columns:
        # times from the equity path
        t = eq["t"].to_numpy()
        # signed position path
        pos = eq["pos"].to_numpy()
    # else reconstruct the position path from fills
    elif len(dr.fills):
        # fills in time order
        f = dr.fills.sort_values("t")
        # +1 buys, -1 sells
        sgn = np.where(f["side"].to_numpy() == "BUY", 1.0, -1.0)
        # cumulative signed position
        pos = np.cumsum(sgn * f["qty"].to_numpy())
        # fill times as the path's time axis
        t = f["t"].to_numpy()
    # no fills and no path -> flat all day
    else:
        return {"max_abs_inv": 0.0, "twa_abs_inv": 0.0, "eod_inv": 0.0}
    # time-weighted average |inventory| = integral(|pos| dt) / total time
    if len(t) > 1:
        # interval lengths between consecutive path points
        dt = np.diff(t)
        # |pos| held through each interval, weighted by its length
        twa = float(np.sum(np.abs(pos[:-1]) * dt) / max(dt.sum(), 1))
    else:
        # single-point path: the only |pos| we have
        twa = float(np.abs(pos[0])) if len(pos) else 0.0
    # the inventory-risk panel
    return {
        # the largest absolute position touched
        "max_abs_inv": float(np.max(np.abs(pos))) if len(pos) else 0.0,
        # the time-weighted average absolute position
        "twa_abs_inv": twa,
        # the end-of-day signed position (pre-liquidation)
        "eod_inv": float(pos[-1]) if len(pos) else 0.0,
    }


def sharpe_sortino(daily_pnls):
    # keep only finite daily P&Ls
    a = np.asarray([x for x in daily_pnls if x is not None and np.isfinite(x)],
                   float)
    # need at least two days for a std
    if len(a) < 2:
        return {"sharpe": np.nan, "sortino": np.nan, "mean_daily": np.nan}
    # mean daily P&L
    mu = a.mean()
    # sample standard deviation
    sd = a.std(ddof=1)
    # the losing days only (downside deviation input)
    downside = a[a < 0]
    # root-mean-square of losing days (0 target)
    dd = np.sqrt(np.mean(downside ** 2)) if len(downside) else np.nan
    # the risk-adjusted panel, annualised by sqrt(252)
    return {
        # mean daily P&L (PKR)
        "mean_daily": float(mu),
        # annualised Sharpe (nan when std is zero)
        "sharpe": float(mu / sd * np.sqrt(252)) if sd > 0 else np.nan,
        # annualised Sortino (nan when no losing days)
        "sortino": float(mu / dd * np.sqrt(252)) if dd and dd > 0 else np.nan,
    }


def run_level_drawdown(daily_pnls):
    # missing days count as zero so the curve stays aligned to the calendar
    a = np.asarray([x if (x is not None and np.isfinite(x)) else 0.0
                    for x in daily_pnls], float)
    # peak-to-trough of the cumulative multi-day equity curve
    return max_drawdown(np.cumsum(a))


def net_bps_lens(dr):
    # the 5s-markout lens needs the feature-store day loaded (want_fs=True)
    if dr.fs_day is None:
        return 0, np.nan, np.nan, np.nan
    # the VALIDATED scorer: (n, capture, markout, net) at the 5s horizon
    return C.score_bps(dr.fills, dr.fs_day)


# ===========================================================================
# quick self-check when run directly (no heavy backtest -- pure-logic asserts)
# ===========================================================================
if __name__ == "__main__":
    # a simple round trip for the FIFO smoke test
    f = [{"t": 0, "side": "BUY", "px": 10, "qty": 100, "bucket": "first15"},
         {"t": 5000, "side": "SELL", "px": 11, "qty": 100, "bucket": "middle"}]
    # run the attribution (fees included via fee_for)
    per = fifo_attribution(f, 0, 0)
    # drawdown: peak 8 -> trough 2 = -6
    assert max_drawdown(np.array([0, 5, 3, 8, 2])) == -6.0
    # positive-mean daily P&L must give a positive Sharpe
    s = sharpe_sortino([10, 12, 8, 11, 9])
    assert s["sharpe"] > 0
    # confirmation banner
    print("mm_harness self-check OK "
          f"(buckets={BUCKETS}, drawdown/sharpe/fifo wired)")


# ===========================================================================
# 5. EXTENDED PANEL -- computable from existing data (added on SZ request)
# ===========================================================================
def participation_rate(dr, day_trades_qty):
    # our filled shares that day
    our_qty = float(dr.fills["qty"].sum()) if len(dr.fills) else 0.0
    # the market's total traded shares that day (caller passes trades qty sum)
    total = float(day_trades_qty) if day_trades_qty else 0.0
    # our share of the tape (capacity headroom metric); nan when no tape
    return (our_qty / total) if total > 0 else np.nan


def inventory_half_life_s(dr):
    # the signed position path (same sources as inventory_stats)
    eq = dr.equity
    # preferred: the engine's own position path
    if len(eq) and "pos" in eq.columns and "t" in eq.columns:
        # times in ms
        t = eq["t"].to_numpy()
        # signed position
        pos = eq["pos"].to_numpy()
    # fallback: reconstruct from fills
    elif len(dr.fills):
        # fills in time order
        f = dr.fills.sort_values("t")
        # +1 buys, -1 sells
        sgn = np.where(f["side"].to_numpy() == "BUY", 1.0, -1.0)
        # cumulative signed position
        pos = np.cumsum(sgn * f["qty"].to_numpy())
        # fill times as the axis
        t = f["t"].to_numpy()
    # flat all day -> no half-life to measure
    else:
        return np.nan
    # find each local |pos| peak's decay: measure time for |pos| to halve.
    # implementation: for every point where |pos| makes a new local max, scan
    # forward for the first time |pos| <= half of that max; collect those spans.
    spans = []
    # the running local max of |pos|
    peak = 0.0
    # the time the current peak was set
    peak_t = None
    # walk the path
    for i in range(len(pos)):
        # absolute inventory at this point
        a = abs(pos[i])
        # a new peak resets the measurement
        if a > peak:
            peak = a
            peak_t = t[i]
        # decayed to half the peak -> record the span and reset the peak
        elif peak > 0 and a <= 0.5 * peak and peak_t is not None:
            spans.append(t[i] - peak_t)
            peak = a
            peak_t = t[i]
    # median half-life in seconds (nan when inventory never halved)
    return float(np.median(spans) / 1000.0) if spans else np.nan


def time_at_inventory_limit(dr, max_inv):
    # the position path (same sourcing as above)
    eq = dr.equity
    # need a timed path with positions to integrate
    if len(eq) and "pos" in eq.columns and "t" in eq.columns:
        t = eq["t"].to_numpy()
        pos = eq["pos"].to_numpy()
    elif len(dr.fills):
        f = dr.fills.sort_values("t")
        sgn = np.where(f["side"].to_numpy() == "BUY", 1.0, -1.0)
        pos = np.cumsum(sgn * f["qty"].to_numpy())
        t = f["t"].to_numpy()
    else:
        return 0.0
    # nothing to integrate on a single point
    if len(t) < 2:
        return 0.0
    # interval lengths
    dt = np.diff(t)
    # "at limit" = within 5% of the ceiling (>=95% of max_inv), held per interval
    at_lim = (np.abs(pos[:-1]) >= 0.95 * max_inv)
    # fraction of session time pinned at the limit (one-sided quoting = lost spread)
    return float(np.sum(dt[at_lim]) / max(dt.sum(), 1))


def hit_rate_and_skew(per_bucket):
    # collect every round-trip realized P&L across buckets (needs the matcher's
    # per-match list -- fifo_attribution_v2 below records it as "matches")
    all_matches = []
    # each bucket contributes its match list when present
    for b, d in per_bucket.items():
        all_matches.extend(d.get("matches", []))
    # nothing matched -> no distribution
    if not all_matches:
        return {"hit_rate": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
                "win_loss_ratio": np.nan, "n_round_trips": 0}
    # the realized P&L distribution as an array
    a = np.asarray(all_matches, float)
    # winning round trips
    wins = a[a > 0]
    # losing round trips
    losses = a[a < 0]
    # the P&L-distribution panel: hit rate + skew (adverse-selection signature
    # = many small wins eaten by a few big losses -> win_loss_ratio << 1)
    return {
        # fraction of round trips that made money
        "hit_rate": float(len(wins) / len(a)),
        # the average winning round trip
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        # the average losing round trip (negative)
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        # |avg win| / |avg loss|: below 1 = losses run bigger than wins
        "win_loss_ratio": (float(wins.mean() / abs(losses.mean()))
                           if len(wins) and len(losses) else np.nan),
        # number of matched round trips in the distribution
        "n_round_trips": int(len(a)),
    }


def fifo_attribution_v2(fills, engine_pnl, liq_time):
    # same reconciling attribution as fifo_attribution, but ALSO records each
    # matched round-trip's realized P&L per bucket ("matches") for hit-rate/skew.
    # normalise input to records
    f = fills if isinstance(fills, list) else (
        fills.to_dict("records") if isinstance(fills, pd.DataFrame) else fills)
    # the FIFO open-lot queue
    open_lots = deque()
    # per-bucket accumulators including the per-match list
    per = {b: {"realized": 0.0, "opened_qty": 0.0, "opened_notional": 0.0,
               "holds": [], "fills": 0, "matches": []} for b in BUCKETS}
    # running matched-pairs realized (residual added after)
    matched_realized = 0.0
    # process fills in stream order
    for fl in f:
        # unpack
        side = fl["side"]
        px = float(fl["px"])
        qty = float(fl["qty"])
        b = fl.get("bucket", "middle")
        t = float(fl["t"])
        # count the fill
        if b in per:
            per[b]["fills"] += 1
        # same side -> opens
        if not open_lots or open_lots[0]["side"] == side:
            open_lots.append({"qty": qty, "px": px, "bucket": b, "t": t,
                              "side": side})
            if b in per:
                per[b]["opened_qty"] += qty
                per[b]["opened_notional"] += qty * px
            continue
        # opposite -> closes FIFO
        remaining = qty
        while remaining > 1e-9 and open_lots and open_lots[0]["side"] != side:
            lot = open_lots[0]
            matched = min(remaining, lot["qty"])
            if lot["side"] == "BUY":
                realized = (px - lot["px"]) * matched
            else:
                realized = (lot["px"] - px) * matched
            realized -= fee_for(lot["px"], matched)
            realized -= fee_for(px, matched)
            ob = lot["bucket"]
            if ob in per:
                per[ob]["realized"] += realized
                per[ob]["holds"].append(t - lot["t"])
                per[ob]["matches"].append(realized)
            matched_realized += realized
            lot["qty"] -= matched
            remaining -= matched
            if lot["qty"] <= 1e-9:
                open_lots.popleft()
        # flip through zero
        if remaining > 1e-9:
            open_lots.append({"qty": remaining, "px": px, "bucket": b, "t": t,
                              "side": side})
            if b in per:
                per[b]["opened_qty"] += remaining
                per[b]["opened_notional"] += remaining * px
    # residual reconciles to the engine headline
    residual_total = float(engine_pnl) - matched_realized
    # residual weights by opened notional
    resid_notional = sum(lot["qty"] * lot["px"] for lot in open_lots)
    # distribute residual to opening buckets (also recorded as a match for skew)
    for lot in open_lots:
        w = (lot["qty"] * lot["px"] / resid_notional) if resid_notional > 0 else 0.0
        share = residual_total * w
        ob = lot["bucket"]
        if ob in per:
            per[ob]["realized"] += share
            per[ob]["holds"].append(max(0.0, liq_time - lot["t"]))
            per[ob]["matches"].append(share)
    # per-bucket attribution with match distributions (sums to engine_pnl)
    return per


def markout_multi_horizon(dr):
    # fill-level markout at ALL THREE stored horizons (1s/5s/30s), side-signed,
    # read from the feature store's leak-checked forward-markout columns via a
    # backward asof join (last event row at or before each fill).
    # needs the feature-store day loaded with the markout columns
    if dr.fs_day is None or len(dr.fills) == 0:
        return {h: np.nan for h in ("1000ms", "5000ms", "30000ms")}
    # the markout columns must be present (caller loads them)
    need = [f"markout_{h}_bps" for h in ("1000ms", "5000ms", "30000ms")]
    # missing columns -> the caller loaded the short column set
    if not all(c in dr.fs_day.columns for c in need):
        return {h: np.nan for h in ("1000ms", "5000ms", "30000ms")}
    # fills sorted by time for the asof join
    f = dr.fills.sort_values("t").reset_index(drop=True)
    # feature rows sorted by event time
    fs = dr.fs_day.sort_values("ts_exch").reset_index(drop=True)
    # backward asof: the last event row at or before each fill
    j = pd.merge_asof(f[["t", "side"]], fs[["ts_exch"] + need],
                      left_on="t", right_on="ts_exch", direction="backward")
    # +1 buys, -1 sells (markout is signed by our position direction)
    sgn = np.where(j["side"].to_numpy() == "BUY", 1.0, -1.0)
    # the per-horizon fills-mean signed markout
    out = {}
    # each horizon's column, side-signed then averaged
    for h in ("1000ms", "5000ms", "30000ms"):
        # the event-level forward markout at the fill's context row
        v = j[f"markout_{h}_bps"].to_numpy()
        # side-sign and average over fills with a valid value
        ok = np.isfinite(v)
        out[h] = float(np.mean(sgn[ok] * v[ok])) if ok.any() else np.nan
    # the three-horizon markout panel
    return out


def fee_share_of_gross(dr):
    # total fees paid across the day's fills (per-side fee on every fill)
    fees = float(sum(fee_for(px, q) for px, q in
                     zip(dr.fills["px"], dr.fills["qty"]))) if len(dr.fills) else 0.0
    # the day's net P&L (post-fee, post-liquidation)
    net = dr.pnl()
    # gross = net + fees (what the quoting earned before the exchange's cut)
    gross = (float(net) + fees) if net is not None else np.nan
    # the fee-burden panel
    return {
        # PKR paid in fees
        "fees_pkr": fees,
        # PKR earned before fees
        "gross_pkr": gross,
        # the exchange's share of gross (nan when gross <= 0: ratio meaningless)
        "fee_share": (fees / gross) if (np.isfinite(gross) and gross > 0)
                     else np.nan,
    }


def attribution_by_reason(dr):
    # P&L-lens by FILL RULE (through / at_queue / at_optimistic / crossing_add):
    # measures how much of the edge depends on the more speculative fill
    # assumptions. Uses the 5s net_bps lens per reason group.
    # needs fills with the reason tag and the feature-store day
    if len(dr.fills) == 0 or "reason" not in dr.fills.columns \
            or dr.fs_day is None:
        return {}
    # one lens result per reason group
    out = {}
    # group the day's fills by their fill rule
    for reason, g in dr.fills.groupby("reason"):
        # the validated 5s scorer on this group
        n, cap, mko, net = C.score_bps(g, dr.fs_day)
        # record the group's size and per-fill economics
        out[reason] = {"fills": int(n), "capture": cap, "markout": mko,
                       "net": net}
    # the per-rule attribution
    return out


# ===========================================================================
# 6. LIFECYCLE-POWERED PANEL -- requires the engine's order_log (bt.order_log,
#    exposed on DayResult as dr.order_log when the instrumented engine runs)
# ===========================================================================
def quote_uptime(dr, order_log):
    # fraction of the continuous session with a live quote per side, and the
    # two-sided presence % (both sides live simultaneously) -- the MM-program
    # compliance metric. order_log rows: side, t_live, t_end (None = never live
    # / still live at close; treat still-live as ending at session close).
    # session bounds
    t0, t1 = dr.session
    # total session span (guard zero)
    span = max(t1 - t0, 1)
    # no log -> the instrumented engine wasn't used
    if order_log is None or len(order_log) == 0:
        return {"uptime_buy": np.nan, "uptime_sell": np.nan,
                "two_sided_pct": np.nan}
    # per-side live intervals, clipped to the session
    ivals = {"BUY": [], "SELL": []}
    # each order contributes [t_live, t_end) when it was ever live
    for _, r in order_log.iterrows():
        # never went live (rejected / in-flight at close) -> no interval
        if r["t_live"] is None or (isinstance(r["t_live"], float)
                                   and np.isnan(r["t_live"])):
            continue
        # start clipped to session open
        a = max(float(r["t_live"]), t0)
        # end = cancel/fill time, or session close when still live
        e = r["t_end"]
        # still-live orders end at the session close
        b = min(float(e) if e is not None and not (isinstance(e, float)
                                                   and np.isnan(e)) else t1, t1)
        # keep only forward intervals on a known side
        if b > a and r["side"] in ivals:
            ivals[r["side"]].append((a, b))
    # merge a side's intervals and sum coverage
    def _coverage(iv):
        # no intervals -> zero coverage
        if not iv:
            return 0.0, []
        # sort by start
        iv = sorted(iv)
        # merged list seeded with the first interval
        merged = [list(iv[0])]
        # sweep and merge overlaps
        for a, b in iv[1:]:
            # overlapping/adjacent -> extend the last merged interval
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            # disjoint -> start a new merged interval
            else:
                merged.append([a, b])
        # total covered time + the merged list (for the two-sided intersect)
        return sum(b - a for a, b in merged), merged
    # per-side coverage
    cov_b, mb_ = _coverage(ivals["BUY"])
    cov_s, ms_ = _coverage(ivals["SELL"])
    # two-sided time = the intersection of the merged interval lists
    two = 0.0
    # classic two-pointer interval intersection
    i = j = 0
    # sweep both merged lists
    while i < len(mb_) and j < len(ms_):
        # overlap of the current pair
        a = max(mb_[i][0], ms_[j][0])
        b = min(mb_[i][1], ms_[j][1])
        # accumulate positive overlaps
        if b > a:
            two += b - a
        # advance the list whose interval ends first
        if mb_[i][1] < ms_[j][1]:
            i += 1
        else:
            j += 1
    # the presence panel as fractions of the session
    return {
        # fraction of the session with a live bid
        "uptime_buy": cov_b / span,
        # fraction of the session with a live ask
        "uptime_sell": cov_s / span,
        # fraction with BOTH sides live (the MM-program requirement)
        "two_sided_pct": two / span,
    }


def time_to_fill(dr, order_log):
    # per-fill queue-wait: fill time minus the order's live time, joined on oid.
    # needs fills carrying oid (the instrumented engine adds it)
    if order_log is None or len(order_log) == 0 or "oid" not in dr.fills.columns:
        return {"median_ttf_s": np.nan, "mean_ttf_s": np.nan}
    # oid -> live time map from the lifecycle log
    live = {int(r["oid"]): r["t_live"] for _, r in order_log.iterrows()
            if r["t_live"] is not None}
    # each fill's wait = fill t minus its order's live t
    waits = []
    # walk the fills
    for _, fl in dr.fills.iterrows():
        # the fill's originating order id
        oid = int(fl["oid"]) if not pd.isna(fl.get("oid", np.nan)) else None
        # join to the live time when known
        if oid is not None and oid in live and live[oid] is not None:
            # queue-wait in ms
            waits.append(float(fl["t"]) - float(live[oid]))
    # no joinable fills -> nan panel
    if not waits:
        return {"median_ttf_s": np.nan, "mean_ttf_s": np.nan}
    # the time-to-fill panel in seconds
    return {"median_ttf_s": float(np.median(waits) / 1000.0),
            "mean_ttf_s": float(np.mean(waits) / 1000.0)}


def peak_message_rate(msg_ts, window_s=1.0):
    # peak outbound messages per second: bin all message timestamps (order sends
    # + cancel sends) into windows and take the max count. Exchanges cap RATES.
    # no messages -> zero rate
    if msg_ts is None or len(msg_ts) == 0:
        return {"peak_msgs_per_s": 0.0, "total_msgs": 0}
    # timestamps as an array (ms)
    a = np.asarray(msg_ts, float)
    # bin edges at window_s resolution across the day
    bins = np.arange(a.min(), a.max() + window_s * 1000.0, window_s * 1000.0)
    # counts per window
    counts, _ = np.histogram(a, bins=bins) if len(bins) > 1 else (np.array([len(a)]), None)
    # the rate panel
    return {
        # the busiest single window's message count (per second at window_s=1)
        "peak_msgs_per_s": float(counts.max() / window_s),
        # total outbound messages for the day
        "total_msgs": int(len(a)),
    }
