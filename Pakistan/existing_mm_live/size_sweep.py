# size_sweep.py -- HOW BIG should a quote be, per symbol? Sweeps the clip size:
#   * fixed50            : the 50-share control (reproduces the stored MID+BOTH runs)
#   * 0.5x .. 2.0x MEDIAN TRADE SIZE, where the median is computed per symbol from a
#     TRAILING 10 trading days, walk-forward (strictly before the simulated day --
#     no look-ahead). Low-priced names get big share clips, expensive names small
#     ones, automatically.
# Inventory limits scale WITH the clip so only ONE variable changes per config:
#   max_inv = 10 x clip (same 10-clip tolerance as today's 500/50)
#   soft_inv = 3 x clip (same 3-clip band as today's 150/50)
# session_scale needs NO recalibration: the skew works in LOTS (pos / clip), and
# the lot-based limits are held constant, so the skew geometry is unchanged.
#
# Alongside P&L it reports the two capacity lenses:
#   * participation: our filled notional as % of the symbol's trailing median daily
#     traded value (the real-world constraint)
#   * top_depth: median resting shares at the best bid/ask (the queue we join)
#
# HONESTY NOTE (read before trusting large multiples): the engine models FIFO queue
# position, so bigger clips DO wait longer and fill partially -- that penalty is
# real. But market impact (others reacting to a big resting order) and
# size-dependent adverse selection are NOT modeled, so results near 1x the median
# trade size are trustworthy while 2x+ is an upper bound, not a forecast.
#
# Strategy config is the LOCKED production one: MID (microprice off) + BOTH
# triggers on. Writes a per-day CSV whose name carries the run timestamp.
#
# Run from existing_mm_live/:  python size_sweep.py

# paths
from pathlib import Path
# timing + run timestamp
import time
from datetime import datetime
# frames + arrays
import pandas as pd
import numpy as np
# driver + engine + strategy
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel
from micro_mm import MicrostructureMM
# reuse confirm's helpers: Path B scorer + per-symbol session_scale table
import confirm_micro_vs_naive as C

# raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# feature store (Path B context)
FS_ROOT = Path("/Users/shazzak/Capital Stake - Results/feature_store")
# where outputs go (absolute; the CSV name carries the run timestamp)
OUT_DIR = Path("/Users/shazzak/Capital Stake - Results")

# symbols under test (PACE included deliberately: it loses at any size -- the
# sweep shows whether sizing changes that or just scales the loss)
SYMBOLS = ["PPL", "UBL"]
# trailing window (trading days) for the median trade size and the ADV
TRAIL_DAYS = 10
# clip multiples of the trailing median trade size
MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10]
# inventory limits as multiples of the clip (today's 500/50 and 150/50 ratios)
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
# the locked production strategy base: MID + BOTH triggers
BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
            enable_eod_trigger=True, enable_lock_trigger=True)
# stratified subsample size (None = all days after the warm-up)
N_DAYS = None


# build the run label list: the control plus each multiple
def labels():
    # fixed 50-share control first (comparability with every stored run)
    out = [("fixed50", None)]
    # then the median-anchored multiples
    for m in MULTS:
        out.append((f"{m:.2f}x_med", m))
    return out


# PRE-PASS: one cheap read of the TRADES table per day -> per-symbol daily median
# trade size (shares) and daily traded notional (PKR). Powers the trailing stats.
def daily_trade_stats(dates):
    # {sym: {date: (median_qty, notional_pkr)}}
    stats = {sym: {} for sym in SYMBOLS}
    # timer for the heartbeat
    t0 = time.perf_counter()
    # walk every date (the FULL history, so trailing windows exist from day 11)
    for i, date in enumerate(dates, 1):
        # open the day's datasets
        dsets = R.open_datasets(date)
        # skip missing days
        if dsets is None:
            continue
        # per symbol
        for sym in SYMBOLS:
            # trades only -- the cheap table
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # skip empty
            if len(t) == 0:
                continue
            # median shares per trade (robust to block-trade skew, unlike the mean)
            med_qty = float(t["qty"].median())
            # total traded value that day (PKR). TRADES table uses 'price' (the
            # snapshot table uses 'px' -- different schemas, easy to conflate).
            notional = float((t["price"] * t["qty"]).sum())
            # store
            stats[sym][str(date)] = (med_qty, notional)
        # heartbeat every 50 days
        if i % 50 == 0 or i == len(dates):
            print(f"  pre-pass {i}/{len(dates)} days  "
                  f"elapsed {C._fmt(time.perf_counter() - t0)}", flush=True)
    return stats


# trailing stats for one symbol strictly BEFORE a given date: median of the last
# TRAIL_DAYS daily medians, and median of the last TRAIL_DAYS daily notionals
def trailing(stats_sym, all_dates, date):
    # daily records strictly before this date, in date order
    prior = [stats_sym[str(d)] for d in all_dates
             if str(d) < str(date) and str(d) in stats_sym]
    # need a full window
    if len(prior) < TRAIL_DAYS:
        return None, None
    # last TRAIL_DAYS records
    win = prior[-TRAIL_DAYS:]
    # median of daily medians (shares) and median of daily notionals (PKR)
    return float(np.median([w[0] for w in win])), float(np.median([w[1] for w in win]))


# median top-of-book depth (shares) per side for one symbol-day, from the snapshot
# frame already loaded for the backtest (no extra IO)
def top_depth(s):
    # continuous-phase rows only
    c = s[s["phase"] == "CONTINUOUS_AUCTION"]
    # best-bid rows: per message, the BID row at the maximum price
    bids = c[c["entry_type"] == "BID"]
    asks = c[c["entry_type"] == "OFFER"]
    # empty guard
    if len(bids) == 0 or len(asks) == 0:
        return np.nan, np.nan
    # index of the best level per message, then that row's resting qty
    bb_qty = bids.loc[bids.groupby("msg_seq")["px"].idxmax(), "qty"]
    ba_qty = asks.loc[asks.groupby("msg_seq")["px"].idxmin(), "qty"]
    # medians across the day's messages
    return float(bb_qty.median()), float(ba_qty.median())


def main():
    # run timestamp -> goes into the CSV filename
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # all dates in the store
    all_dates = R.discover_dates()
    # PRE-PASS over the FULL history (cheap: trades only)
    print(f"pre-pass: daily median trade size + notional, {len(all_dates)} days", flush=True)
    tstats = daily_trade_stats(all_dates)
    # run days: skip the first TRAIL_DAYS (warm-up), then stratify to N_DAYS
    run_dates = all_dates[TRAIL_DAYS:]
    if N_DAYS is not None and N_DAYS < len(run_dates):
        idx = [round(i * (len(run_dates) - 1) / (N_DAYS - 1)) for i in range(N_DAYS)]
        run_dates = [run_dates[i] for i in idx]
    print(f"\nsize_sweep: {len(labels())} configs x {len(SYMBOLS)} symbols x "
          f"{len(run_dates)} days ({run_dates[0]} .. {run_dates[-1]})\n", flush=True)

    # accumulators keyed (sym, label)
    acc = {}
    for sym in SYMBOLS:
        for label, _ in labels():
            acc[(sym, label)] = {"pnl": 0.0, "fills": 0, "days": 0, "net": [],
                                 "part": [], "clip": [], "unclean": 0, "unfilled": 0.0}
    # per-symbol depth records (measured once per symbol-day)
    depth = {sym: [] for sym in SYMBOLS}
    # per-day CSV rows
    rows = []
    # timers + heartbeat
    t0_all = time.perf_counter()
    sd = 0
    sd_total = len(run_dates) * len(SYMBOLS)

    # OUTER: dates
    for date in run_dates:
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # MIDDLE: symbols
        for sym in SYMBOLS:
            sd += 1
            # trailing stats strictly before this day (walk-forward)
            med_qty, adv = trailing(tstats[sym], all_dates, date)
            # skip if the trailing window isn't full
            if med_qty is None or med_qty <= 0:
                continue
            # feature-store day for Path B
            fs_path = FS_ROOT / sym / f"date={date}.parquet"
            if not fs_path.exists():
                continue
            fs_day = pd.read_parquet(
                fs_path, columns=["ts_exch", "mid", "spread_bps", "obi_1",
                                  "toxicity", "realized_vol_bps"])
            # ---- build events ONCE per symbol-day ----
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0 or len(s) == 0:
                continue
            events, snap_groups, t = R.build_events(u, s, t)
            # session window from the phase field (the corrected bell)
            cont_snap = s[s["phase"] == "CONTINUOUS_AUCTION"]
            if len(cont_snap) == 0:
                continue
            t0, t1 = int(cont_snap["ts_exch"].min()), int(cont_snap["ts_exch"].max())
            # measure top-of-book depth once per symbol-day (shares each side)
            bbq, baq = top_depth(s)
            depth[sym].append((bbq, baq))
            # INNER: every clip config reuses the same event stream
            for label, mult in labels():
                # the clip: fixed 50 control, or the multiple of the trailing median
                clip = 50 if mult is None else max(1, int(round(mult * med_qty)))
                # strategy params: locked base + this clip + scaled inventory limits
                params = dict(BASE)
                params["size"] = clip
                params["max_inv"] = int(round(MAXINV_CLIPS * clip))
                params["soft_inv"] = int(round(SOFTINV_CLIPS * clip))
                params["session_scale"] = C.SESSION_SCALE_BASE[sym]
                # engine config: fresh seeded latency, corrected session window
                cfg = dict(R.CFG, session=(t0, t1),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                # build + run
                strat = MicrostructureMM(session_ms=(t0, t1), **params)
                bt = Backtester(strat, cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                # Path B per-fill economics (reused scorer)
                nf, cap_b, mk_b, net_b = C.score_bps(fills, fs_day)
                # filled notional -> participation vs the trailing median daily value.
                # Fills use 'px' (the raw-fill schema); guard for the empty/missing
                # case so a surprise degrades to NaN participation, never a crash.
                f = fills if isinstance(fills, pd.DataFrame) else pd.DataFrame(fills)
                if len(f) and "px" in f.columns and "qty" in f.columns:
                    fnot = float((f["px"] * f["qty"]).sum())
                else:
                    fnot = 0.0
                part = 100.0 * fnot / adv if adv and adv > 0 else np.nan
                # Path A: the day's liquidated P&L
                liq = None
                if bt.eod is not None:
                    liq = bt.eod["equity_liquidated"]
                # accumulate
                a = acc[(sym, label)]
                if liq is not None:
                    a["pnl"] += float(liq)
                    a["days"] += 1
                a["fills"] += nf
                # accumulate the day's mean net bps when it exists (NaN = no fills
                # or all fills near the close with no forward mid)
                if isinstance(net_b, (float, np.floating)) and not np.isnan(net_b):
                    a["net"].append(float(net_b))
                if not np.isnan(part):
                    a["part"].append(part)
                a["clip"].append(clip)
                if bt.eod is not None and bt.eod["liquidation_clean"] is False:
                    a["unclean"] += 1
                # per-day CSV row (full daily stats, per your standing request)
                rows.append({"date": str(date), "symbol": sym, "config": label,
                             "clip_sh": clip, "med_trade_sh": round(med_qty, 1),
                             "adv_pkr": round(adv, 0), "fills": nf,
                             "pnl_pkr": (round(float(liq), 2) if liq is not None else np.nan),
                             "net_bps": (round(net_b, 3) if isinstance(net_b, float) else np.nan),
                             "fill_notional_pkr": round(fnot, 0),
                             "participation_pct": (round(part, 4) if not np.isnan(part) else np.nan),
                             "top_bid_sh": bbq, "top_ask_sh": baq})
            # heartbeat with ETA
            if sd % 20 == 0 or sd == sd_total:
                el = time.perf_counter() - t0_all
                eta = el / sd * (sd_total - sd)
                print(f"  {sd}/{sd_total} symbol-days  elapsed {C._fmt(el)}  "
                      f"ETA {C._fmt(eta)}", flush=True)

    # ---- per-day CSV (timestamped filename) ----
    out_csv = OUT_DIR / f"size_sweep_daily_{stamp}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # ---- report ----
    print(f"\n{'sym':5s} {'config':10s} {'avg_clip':>8s} {'fills':>8s} {'pnl_pkr':>10s} "
          f"{'net_bps':>8s} {'part%':>7s} {'unclean':>7s}")
    for sym in SYMBOLS:
        for label, _ in labels():
            a = acc[(sym, label)]
            # skip configs that never ran
            if a["days"] == 0:
                continue
            # means across days
            avg_clip = np.mean(a["clip"]) if a["clip"] else np.nan
            net = np.nanmean(a["net"]) if a["net"] else np.nan
            part = np.nanmean(a["part"]) if a["part"] else np.nan
            print(f"{sym:5s} {label:10s} {avg_clip:>8,.0f} {a['fills']:>8,} "
                  f"{a['pnl']:>10,.0f} {net:>8.3f} {part:>7.3f} {a['unclean']:>7d}")
        # symbol-level context: the queue we join, and the anchor we sized from
        d = np.array(depth[sym], dtype=float)
        if len(d):
            print(f"{sym:5s} context: median top-of-book depth ~ "
                  f"{np.nanmedian(d[:, 0]):,.0f} bid / {np.nanmedian(d[:, 1]):,.0f} ask shares")
    print(f"\nwrote {out_csv}")
    print("\nREAD: net_bps vs clip is the capacity curve. Flat net_bps with rising")
    print("pnl = room to size up (within the model's honesty limits above).")
    print("Falling net_bps = queue effects biting; the knee is your size ceiling.")
    print("Compare each clip to the top-of-book depth: a clip >> the queue you join")
    print("is a size the model can no longer be trusted on.")


if __name__ == "__main__":
    main()
