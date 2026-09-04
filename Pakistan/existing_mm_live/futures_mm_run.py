# futures_mm_run.py -- MARKET-MAKING THE PSX DELIVERY FUTURES (DFC).
#
# The probes established: identical event vocabulary to spot (build_events/Book
# reuse directly), tick=0.01, lot=500 (always), active front month rolls monthly,
# spread ~14 bps vs spot ~5, proprietary fee 0.0938 bps/side (~8x cheaper than
# spot TREC). This runs the LOCKED production MID strategy (POV unwind) on the
# active-month future of each liquid root, at clip = 1/2/4 LOTS.
#
# Key mechanics:
#   * ACTIVE CONTRACT per (root, date): the contract with the max TRAILING
#     ROLL_TRAIL-day volume -- causal, auto-handles the monthly roll. The engine
#     flattens EOD every day (equity_liquidated), so no overnight carry, no
#     position ever crosses a roll, and the CDC delivery fee never applies.
#   * FEES: futures Laga = 0.93809 PKR / 100,000 = 0.0938 bps/side. Patched into
#     BOTH mm_backtest.FEE_TOTAL_PCT (fee_for reads it at call time) and
#     micro_mm.FEE_TOTAL_PCT (viability gate imported it by value). P&L is
#     pre-CGT (CGT is a profit tax, not a trading cost).
#   * CALIBRATION (walk-forward): per root, session_scale back-solved on the
#     FIRST CAL_DAYS days only (trade days come after), using the engine's exact
#     skew formula: scale = (med_spread/2) / (gamma*(sigma*fair)^2*tau*lots_max).
#     sigma here = std of snapshot mid-to-mid returns (proxy for observe()'s EMA
#     sigma -- flagged simplification; refine with real observe() replay later).
#     The 4-bucket volume profile is likewise computed on CAL_DAYS only.
#   * SCORING: futures have NO feature store, so net_bps is computed locally:
#     capture vs the snapshot-mid at fill time, markout vs the mid MARKOUT_S
#     seconds later, fees off. Same definition as the spot scorer, self-contained.
#
# Outputs: futures_mm_daily_<stamp>.csv + futures_mm_summary_<stamp>.csv.
# Run from existing_mm_live/:  caffeinate -is python3 futures_mm_run.py

# paths
from pathlib import Path
# timing + stamp
import time
from datetime import datetime
# frames + arrays
import pandas as pd
import numpy as np
# in-place SQL over the parquet store (roll map pre-pass)
import duckdb
# driver + engine + strategy
import run_legacy_mm as R
import mm_backtest as MB
import micro_mm as MM
from mm_backtest import Backtester, LatencyModel
from micro_mm import MicrostructureMM
# heartbeat formatter
import confirm_micro_vs_naive as C

# raw store + results
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
RESULTS = Path("/Users/shazzak/Capital Stake - Results")
# the futures feature store (built by build_feature_store_futures.py) -- scoring
# now routes through the ENGINE'S OWN mids here, not a local reimplementation
FS_FUT = RESULTS / "feature_store_fut"
PARSED = str(R.PARSED_ROOT)

# ------------------------------ experiment knobs ------------------------------
# the liquid futures roots (from futures_top20: tape >= ~3 trades/min)
ROOTS = ["TRG", "MLCF", "BOP", "SSGC", "PTC", "TPLP", "PAEL", "KOSM",
         "DGKC", "THCCL", "PIAHCLA", "PACE", "PIBTL", "TPL", "CNERGY"]
# clip sizes in LOTS (1 lot = 500 shares); the lot floor is the 1x anchor
CLIP_LOTS = [1, 2, 4]
LOT = 500
# inventory limits in clips (production ratios)
MAXINV_CLIPS = 10.0
SOFTINV_CLIPS = 3.0
MAX_POV = 0.10
# trailing window (days) for the active-contract roll decision
ROLL_TRAIL = 5
# walk-forward calibration window: first CAL_DAYS days per root are calibration
# ONLY (never traded); trading starts after
CAL_DAYS = 20
# markout horizon for the local net_bps scorer (seconds)
MARKOUT_S = 5.0
# a contract-day must have at least this many trades to be quoted (dead tails)
MIN_TRADES_DAY = 200
# engine defaults matching the locked production config
GAMMA = 0.15
MID_BASE = dict(min_edge_pct=0.0005, improve_ticks=0.0, use_microprice=False,
                enable_eod_trigger=True, enable_lock_trigger=True)
# ---- FUTURES FEE (proprietary TSC, DFC market): Laga only, per side ----------
FUT_FEE_PER_SIDE = 0.93809 / 100000.0     # 0.0938 bps per side
# ------------------------------------------------------------------------------


def newest(pattern):
    cands = sorted(RESULTS.glob(pattern))
    if not cands:
        raise SystemExit(f"missing {pattern}")
    return cands[-1]


# per-date session segments (same trading calendar as spot -- reuse spot's)
def load_segments():
    df = pd.read_csv(newest("session_segments_*.csv"))
    out = {}
    for _, r in df.iterrows():
        out[r["date"]] = [tuple(int(x) for x in p.split(":"))
                          for p in r["segments"].split(";")]
    return out


# ---- PRE-PASS 1 (DuckDB, one shot): per (root, contract, date) volume/trades --
# feeds the roll map, the median-size check, and the calibration-day selection.
def load_futures_calendar():
    con = duckdb.connect()
    q = f"""
    SELECT regexp_extract(symbol,'^([A-Z]+)-',1) AS root, symbol,
           CAST(date AS VARCHAR) AS date, COUNT(*) AS n_trades, SUM(qty) AS qty
    FROM read_parquet('{PARSED}/trades/date=*/*.parquet')
    WHERE market='STOCK_DEL_FUT'
    GROUP BY root, symbol, date
    """
    cal = con.execute(q).df()
    con.close()
    return cal[cal["root"].isin(ROOTS)].copy()


# ---- the ROLL MAP: for each (root, date), the active contract = the one with
# the max qty over the TRAILING ROLL_TRAIL days (strictly before `date`) -------
def build_roll_map(cal, all_dates):
    dates = [str(d) for d in all_dates]
    idx = {d: i for i, d in enumerate(dates)}
    roll = {}
    # per root, per contract, a date->qty series for trailing sums
    for root, g in cal.groupby("root"):
        piv = g.pivot_table(index="date", columns="symbol", values="qty",
                            aggfunc="sum").reindex(dates).fillna(0.0)
        # trailing ROLL_TRAIL-day sum, shifted 1 so `date` itself is excluded
        trail = piv.rolling(ROLL_TRAIL, min_periods=1).sum().shift(1)
        # winner per date (NaN rows -> no history yet -> no active contract)
        for d in dates:
            row = trail.loc[d]
            if row.notna().any() and row.max() > 0:
                roll[(root, d)] = row.idxmax()
    return roll


# ---- L1 mid series for one futures contract-day from the SNAPSHOT table ------
# (futures snapshots pivot exactly like the probe: BID max px, OFFER min px at L1)
def l1_mid_series(s):
    c = s[s["phase"] == "CONTINUOUS_AUCTION"].copy()
    if len(c) == 0:
        return None
    c["ts"] = R.to_ms(c["orig_time"])
    bids = c[c["entry_type"] == "BID"]
    asks = c[c["entry_type"] == "OFFER"]
    if len(bids) == 0 or len(asks) == 0:
        return None
    bb = bids.loc[bids.groupby("msg_seq")["px"].idxmax(), ["msg_seq", "ts", "px"]] \
             .rename(columns={"px": "bb"})
    ba = asks.loc[asks.groupby("msg_seq")["px"].idxmin(), ["msg_seq", "px"]] \
             .rename(columns={"px": "ba"})
    l1 = bb.merge(ba, on="msg_seq").sort_values("ts")
    l1 = l1[(l1["bb"] > 0) & (l1["ba"] >= l1["bb"])]
    if len(l1) == 0:
        return None
    l1["mid"] = 0.5 * (l1["bb"] + l1["ba"])
    return l1[["ts", "bb", "ba", "mid"]].reset_index(drop=True)


# ---- scoring now uses the VALIDATED spot scorer C.score_bps against the
# futures feature store (feature_store_fut). The local reimplementation was
# deleted: it diverged from the engine's fills and produced the impossible
# 'positive net_bps + negative P&L'. One price basis now -> consistent.


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # ---- FEE PATCH: futures Laga only, both modules (documented at top) ----
    MB.FEE_TOTAL_PCT = FUT_FEE_PER_SIDE
    MM.FEE_TOTAL_PCT = FUT_FEE_PER_SIDE
    print(f"fee patched: {FUT_FEE_PER_SIDE*1e4:.4f} bps/side "
          f"({FUT_FEE_PER_SIDE*2e4:.4f} bps round-trip)")

    segments = load_segments()
    all_dates = R.discover_dates()
    # ---- pre-pass 1: futures calendar + roll map ----
    print("pre-pass 1: futures trade calendar (DuckDB, one shot)", flush=True)
    cal = load_futures_calendar()
    roll = build_roll_map(cal, all_dates)
    print(f"  roll map built: {len(roll)} (root,date) active-contract entries")

    # ---- pre-pass 2: walk-forward calibration per root on the first CAL_DAYS --
    # collect per-root: median spread (PKR), sigma (mid-return std), fair (median
    # mid), and the 4-bucket volume profile -- all from calibration days ONLY.
    print(f"pre-pass 2: calibration on first {CAL_DAYS} days per root", flush=True)
    cal_dates = [str(d) for d in all_dates[:CAL_DAYS]]
    calib = {r: {"spr": [], "sig": [], "fair": [],
                 "f": [], "m": [], "p": [], "l": []} for r in ROOTS}
    t0 = time.perf_counter()
    for i, date in enumerate(cal_dates, 1):
        dsets = R.open_datasets(date)
        if dsets is None or str(date) not in segments:
            continue
        segs = segments[str(date)]
        tradeable = sum(e - s for s, e in segs) / 60000.0
        # bucket boundaries for this day (mirrors build_volume_profile 4-bucket)
        f_end = segs[0][0] + 15 * 60000
        l_start = segs[-1][1] - 15 * 60000
        p_start = segs[-1][1] - 60 * 60000
        p_min = max(min(45.0, tradeable - 30.0), 1.0)
        mid_min = max(tradeable - 30.0 - p_min, 1.0)
        for root in ROOTS:
            sym = roll.get((root, str(date)))
            if sym is None:
                continue
            # snapshot L1 for spread/sigma/fair
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            l1 = l1_mid_series(s) if len(s) else None
            if l1 is not None and len(l1) > 50:
                calib[root]["spr"].append(float((l1["ba"] - l1["bb"]).median()))
                rets = l1["mid"].pct_change().dropna()
                if len(rets) > 10:
                    calib[root]["sig"].append(float(rets.std()))
                calib[root]["fair"].append(float(l1["mid"].median()))
            # trades for the volume profile buckets
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
            print(f"  calib {i}/{len(cal_dates)}  "
                  f"elapsed {C._fmt(time.perf_counter() - t0)}", flush=True)

    # back-solve session_scale per root with the engine's EXACT skew formula:
    #   skew_at_max = gamma * (sigma*fair)^2 * scale * tau * lots_max = spread/2
    scales, profiles = {}, {}
    print(f"\n{'root':8s} {'med_spr':>8s} {'sigma':>10s} {'fair':>8s} "
          f"{'scale':>10s} {'prof f/m/p/l (sh/min)':>30s}")
    for root in ROOTS:
        c = calib[root]
        if not c["spr"] or not c["sig"] or not c["fair"]:
            print(f"{root:8s}  INSUFFICIENT CALIBRATION DATA -- excluded")
            continue
        med_spr = float(np.median(c["spr"]))
        sig = float(np.median(c["sig"]))
        fair = float(np.median(c["fair"]))
        # tau=1 (start of session), lots_max = MAXINV_CLIPS
        denom = GAMMA * (sig * fair) ** 2 * 1.0 * MAXINV_CLIPS
        if denom <= 0:
            print(f"{root:8s}  DEGENERATE (zero vol) -- excluded")
            continue
        scales[root] = (med_spr / 2.0) / denom
        profiles[root] = (float(np.median(c["f"])) if c["f"] else 0.0,
                          float(np.median(c["m"])) if c["m"] else 0.0,
                          float(np.median(c["p"])) if c["p"] else 0.0,
                          float(np.median(c["l"])) if c["l"] else 0.0)
        pf = profiles[root]
        print(f"{root:8s} {med_spr:>8.3f} {sig:>10.2e} {fair:>8.2f} "
              f"{scales[root]:>10.1f} {pf[0]:>7.0f}/{pf[1]:.0f}/{pf[2]:.0f}/{pf[3]:.0f}")

    # ---- MAIN RUN: trade days = after the calibration window ----
    run_dates = all_dates[CAL_DAYS:]
    live_roots = sorted(scales.keys())
    total = len(run_dates) * len(live_roots)
    print(f"\nfutures_mm_run: {len(CLIP_LOTS)} clip sizes x {len(live_roots)} roots "
          f"x {len(run_dates)} days\n", flush=True)

    acc = {}
    for root in live_roots:
        for cl in CLIP_LOTS:
            acc[(root, cl)] = {"pnl": 0.0, "days": 0, "fills": 0, "net": [],
                               "unclean": 0, "notional": []}
    rows = []
    t0_all = time.perf_counter()
    sd = 0
    for date in run_dates:
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        segs = segments.get(str(date))
        if segs is None:
            continue
        for root in live_roots:
            sd += 1
            sym = roll.get((root, str(date)))
            if sym is None:
                continue
            # the day's data for the ACTIVE contract
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # skip dead contract-days (roll edges, expiry tails)
            if len(t) < MIN_TRADES_DAY or len(s) == 0:
                continue
            # load the futures feature store for THIS active contract-day
            fs_path = FS_FUT / sym / f"date={date}.parquet"
            if not fs_path.exists():
                continue
            fs_day = pd.read_parquet(fs_path, columns=["ts_exch", "mid",
                                     "spread_bps", "obi_1", "toxicity",
                                     "realized_vol_bps"])
            events, snap_groups, t = R.build_events(u, s, t)
            cont = s[s["phase"] == "CONTINUOUS_AUCTION"]
            if len(cont) == 0:
                continue
            t0_, t1_ = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
            for cl in CLIP_LOTS:
                clip = cl * LOT
                params = dict(MID_BASE)
                params["size"] = clip
                params["max_inv"] = int(round(MAXINV_CLIPS * clip))
                params["soft_inv"] = int(round(SOFTINV_CLIPS * clip))
                params["session_scale"] = scales[root]
                params["gamma"] = GAMMA
                params["unwind_profile"] = profiles[root]
                params["unwind_pov"] = MAX_POV
                params["session_segments"] = segs
                cfg = dict(R.CFG, session=(t0_, t1_),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                strat = MicrostructureMM(session_ms=(t0_, t1_), **params)
                bt = Backtester(strat, cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                nf, cap_b, mk_b, net_b = C.score_bps(fills, fs_day)
                liq = bt.eod["equity_liquidated"] if bt.eod is not None else None
                a = acc[(root, cl)]
                if liq is not None:
                    a["pnl"] += float(liq)
                    a["days"] += 1
                a["fills"] += nf
                if np.isfinite(net_b):
                    a["net"].append(net_b)
                if bt.eod is not None and bt.eod["liquidation_clean"] is False:
                    a["unclean"] += 1
                # capital proxy: max_inv notional at the day's median mid
                a["notional"].append(params["max_inv"] * float(fs_day["mid"].median()))
                rows.append({"date": str(date), "root": root, "contract": sym,
                             "clip_lots": cl, "fills": nf,
                             "pnl_pkr": (round(float(liq), 2) if liq is not None else np.nan),
                             "net_bps": (round(net_b, 3) if np.isfinite(net_b) else np.nan)})
            if sd % 10 == 0 or sd == total:
                el = time.perf_counter() - t0_all
                print(f"  {sd}/{total} root-days  elapsed {C._fmt(el)}  "
                      f"ETA {C._fmt(el / sd * (total - sd))}", flush=True)

    daily = RESULTS / f"futures_mm_daily_{stamp}.csv"
    pd.DataFrame(rows).to_csv(daily, index=False)

    # ---- summary: per root per clip -- the futures MM verdict ----
    print(f"\n=== FUTURES MM (DFC, fee {FUT_FEE_PER_SIDE*2e4:.3f} bps rt) ===")
    print(f"{'root':8s} {'lots':>5s} {'pnl':>11s} {'pnl/day':>9s} {'net_bps':>8s} "
          f"{'fills/day':>9s} {'unclean':>7s} {'pnl/cap%':>9s}")
    summ = []
    for root in live_roots:
        for cl in CLIP_LOTS:
            a = acc[(root, cl)]
            if a["days"] == 0:
                continue
            cap = float(np.mean(a["notional"])) if a["notional"] else np.nan
            pnl_cap = 100.0 * a["pnl"] / (cap * a["days"]) if cap and cap > 0 else np.nan
            print(f"{root:8s} {cl:>5d} {a['pnl']:>11,.0f} {a['pnl']/a['days']:>9,.0f} "
                  f"{np.mean(a['net']) if a['net'] else float('nan'):>8.3f} "
                  f"{a['fills']/a['days']:>9.1f} {a['unclean']:>7d} {pnl_cap:>9.4f}")
            summ.append({"root": root, "clip_lots": cl, "pnl": a["pnl"],
                         "days": a["days"], "pnl_day": a["pnl"] / a["days"],
                         "net_bps": (np.mean(a["net"]) if a["net"] else np.nan),
                         "fills_day": a["fills"] / a["days"],
                         "unclean": a["unclean"],
                         "avg_maxinv_notional": (np.mean(a["notional"])
                                                 if a["notional"] else np.nan)})
    S = pd.DataFrame(summ)
    S.to_csv(RESULTS / f"futures_mm_summary_{stamp}.csv", index=False)
    # portfolio line at 1 lot
    for cl in CLIP_LOTS:
        x = S[S.clip_lots == cl]
        if len(x):
            print(f"\nPORTFOLIO @ {cl} lot(s): {x['pnl'].sum():>11,.0f} total, "
                  f"{x['pnl_day'].sum():>8,.0f}/day across {len(x)} roots")
    print("\nREAD: net_bps must clear ~0.2 bps (the rt fee) with margin; compare")
    print("pnl/day to the SPOT winners (~9.7k/day at 3x) and judge pnl/cap% --")
    print("the 500-lot flatters gross P&L, so per-capital is the honest metric.")
    print(f"\nwrote {RESULTS / f'futures_mm_summary_{stamp}.csv'}\nwrote {daily}")


if __name__ == "__main__":
    main()
