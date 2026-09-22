# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# diag_adverse_selection.py -- WHY does micro drift short while naive stays flat?
# Tests three code-identified suspects, per fill, against book context:
#   (1) GATE SELECTION: are micro's fills concentrated in wide-spread / toxic
#       windows (vs the session baseline)?  -> the viability gate self-selecting.
#   (2) MICROPRICE LEAN: does micro SELL when the book is ask-heavy (low obi_1)
#       and BUY when bid-heavy?  -> fair = microprice trading WITH imbalance.
#   (3) ADVERSE SELECTION: is markout negative in high-toxicity terciles?  -> the
#       imbalance signal is informed flow that keeps going, not mean-reverting.
# Runs ONE config (micro me=0.0005, session_scale = 1x per symbol). DAYS_LIMIT
# keeps the first pass fast; set to None for all 207 days.
#
# Run from existing_mm_live/:  python diag_adverse_selection.py

# paths
from pathlib import Path
# timing
import time
# frames + arrays
import pandas as pd
import numpy as np
# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# driver + engine + strategy
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel
from micro_mm import MicrostructureMM
# attribution economics
import fill_attribution as FA
# context join (same one confirm_micro_vs_naive.py trusts)
import persist_fills as PF

# raw store
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# feature store (context columns)
# Resolve this filesystem path through the canonical checkout/data configuration.
FS_ROOT = Path(str(_hft_paths.RESULTS_ROOT / 'feature_store'))
# pilot symbols
SYMBOLS = ["PPL", "UBL"]
# per-symbol session_scale (1x point); the fix is applied in micro_mm already
SESSION_SCALE_BASE = {"PPL": 7.6, "UBL": 3.9}
# cap days for a fast first pass; None = all days
DAYS_LIMIT = 40
# the single config under the microscope
OVERRIDES = {"min_edge_pct": 0.0005, "improve_ticks": 0.0}
# context columns pulled from the feature store per symbol-day
FS_COLS = ["ts_exch", "mid", "spread_bps", "obi_1", "toxicity", "realized_vol_bps"]


# compact mm:ss
def _fmt(sec):
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# collect per-fill context for micro across (a subset of) days
def collect_fills():
    # all dates, optionally truncated
    dates = R.discover_dates()
    if DAYS_LIMIT is not None:
        dates = dates[:DAYS_LIMIT]
    # per-symbol fill frames + per-symbol baseline-context samples
    fill_frames = {s: [] for s in SYMBOLS}
    base_frames = {s: [] for s in SYMBOLS}
    # timers / heartbeat
    t0_all = time.perf_counter()
    sd = 0
    sd_total = len(dates) * len(SYMBOLS)
    # announce
    print(f"diag: micro 1x, {len(SYMBOLS)} symbols x {len(dates)} days\n", flush=True)
    # OUTER: dates
    for date in dates:
        # datasets once
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # MIDDLE: symbols
        for sym in SYMBOLS:
            # feature-store day (context + baseline distribution)
            fs_path = FS_ROOT / sym / f"date={date}.parquet"
            if not fs_path.exists():
                continue
            # load context columns
            fs_day = pd.read_parquet(fs_path, columns=FS_COLS)
            # keep a subsample of the whole-session context as the BASELINE
            # (what the market looked like regardless of whether micro quoted).
            base_frames[sym].append(fs_day.sample(min(len(fs_day), 2000)))
            # tables for the engine
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0 or len(s) == 0:
                continue
            # build events once
            events, snap_groups, t = R.build_events(u, s, t)
            # continuous session window
            cont = t[t["initiator"] != "AUCTION"]
            if len(cont) == 0:
                continue
            t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
            # micro params with the 1x per-symbol session_scale injected
            params = dict(R.MICRO_PARAMS)
            params.update(OVERRIDES)
            params["session_scale"] = SESSION_SCALE_BASE[sym] * 1.0
            # config with fresh seeded latency
            cfg = dict(R.CFG, session=(t0, t1),
                       latency_model=LatencyModel(seed=R.LATENCY_SEED))
            # run
            bt = Backtester(MicrostructureMM(session_ms=(t0, t1), **params), cfg)
            fills, equity, stats = bt.run(events, snap_groups)
            # no fills -> next
            if fills is None or len(fills) == 0:
                sd += 1
                continue
            # attach context + forward mid via the trusted join
            f = PF.join_fill_context(fills, fs_day)
            # keep it
            fill_frames[sym].append(f)
            # heartbeat
            sd += 1
            if sd % 25 == 0:
                el = time.perf_counter() - t0_all
                proj = el / sd * sd_total
                print(f"  {sd}/{sd_total} symbol-days  elapsed {_fmt(el)}  "
                      f"ETA {_fmt(proj - el)}", flush=True)
    # concat per symbol
    fills = {s: (pd.concat(fill_frames[s], ignore_index=True) if fill_frames[s] else pd.DataFrame())
             for s in SYMBOLS}
    base = {s: (pd.concat(base_frames[s], ignore_index=True) if base_frames[s] else pd.DataFrame())
            for s in SYMBOLS}
    return fills, base


# run the three tests + plots for one symbol
def analyse(sym, f, b):
    # guard: need the context columns the join is supposed to carry
    need = {"side", "px", "mid0", "mid_h", "spread_bps", "obi_1", "toxicity"}
    missing = need - set(f.columns)
    if missing:
        print(f"[{sym}] MISSING columns after join: {missing}")
        print(f"[{sym}] join returned: {list(f.columns)}")
        print("  -> tell me these names and I'll adjust the join; skipping symbol.")
        return
    # signed side: +1 BUY, -1 SELL
    sgn = np.where(f["side"] == "BUY", 1.0, -1.0)
    # per-fill markout (mid move after the fill, in your favour = positive)
    f = f.copy()
    f["markout"] = FA.markout_bps(sgn, f["mid0"], f["mid_h"])

    print(f"\n=== {sym} : {len(f)} micro fills "
          f"({(f['side'] == 'BUY').mean() * 100:.0f}% BUY / "
          f"{(f['side'] == 'SELL').mean() * 100:.0f}% SELL) ===")

    # (1) GATE SELECTION: fill-time context vs session baseline
    print("  (1) selection  | fills vs baseline")
    for col in ("spread_bps", "toxicity"):
        # mean at micro's fills
        fm = f[col].mean()
        # mean across the whole session (baseline)
        bm = b[col].mean() if col in b.columns and len(b) else np.nan
        print(f"      {col:12s} fills={fm:8.3f}  baseline={bm:8.3f}  "
              f"ratio={fm / bm:5.2f}x" if bm == bm and bm != 0 else
              f"      {col:12s} fills={fm:8.3f}  baseline=n/a")

    # (2) MICROPRICE LEAN: obi_1 by side (obi_1 > 0.5 = bid-heavy)
    obi_buy = f.loc[f["side"] == "BUY", "obi_1"].mean()
    obi_sell = f.loc[f["side"] == "SELL", "obi_1"].mean()
    print("  (2) microprice | mean obi_1 by side (0.5 = balanced book)")
    print(f"      BUY  fills obi_1={obi_buy:.3f}   (expect >0.5 if buying bid-heavy)")
    print(f"      SELL fills obi_1={obi_sell:.3f}   (expect <0.5 if selling ask-heavy)")

    # (3) ADVERSE SELECTION: markout by toxicity tercile
    fv = f[f["markout"].notna()].copy()
    # tercile edges on toxicity
    fv["tox_t"] = pd.qcut(fv["toxicity"], 3, labels=["low", "mid", "high"], duplicates="drop")
    print("  (3) adverse    | mean markout_bps by toxicity tercile (neg = adverse)")
    print(fv.groupby("tox_t", observed=True)["markout"].agg(["mean", "count"]).round(3).to_string())

    # ---- plots ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # (a) spread_bps at fills vs baseline
    ax[0].hist(b["spread_bps"].dropna(), bins=40, alpha=0.5, density=True,
               color="#B0B0B0", label="baseline")
    ax[0].hist(f["spread_bps"].dropna(), bins=40, alpha=0.5, density=True,
               color="#1f77b4", label="micro fills")
    ax[0].set_title(f"{sym} (1) spread_bps: fills vs baseline")
    ax[0].set_xlabel("spread_bps"); ax[0].legend(fontsize=8)
    # (b) obi_1 distribution by side
    ax[1].hist(f.loc[f["side"] == "BUY", "obi_1"].dropna(), bins=30, alpha=0.5,
               density=True, color="#2ca02c", label="BUY fills")
    ax[1].hist(f.loc[f["side"] == "SELL", "obi_1"].dropna(), bins=30, alpha=0.5,
               density=True, color="#d62728", label="SELL fills")
    ax[1].axvline(0.5, color="black", lw=1)
    ax[1].set_title(f"{sym} (2) obi_1 by fill side")
    ax[1].set_xlabel("obi_1 (>0.5 bid-heavy)"); ax[1].legend(fontsize=8)
    # (c) markout by toxicity tercile
    m = fv.groupby("tox_t", observed=True)["markout"].mean()
    ax[2].bar(m.index.astype(str), m.values,
              color=["#7fbf7f" if v >= 0 else "#d62728" for v in m.values])
    ax[2].axhline(0, color="black", lw=1)
    ax[2].set_title(f"{sym} (3) markout by toxicity tercile")
    ax[2].set_ylabel("mean markout_bps")
    fig.tight_layout()
    out = f"diag_adverse_{sym}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# entry
if __name__ == "__main__":
    fills, base = collect_fills()
    for sym in SYMBOLS:
        if len(fills[sym]) == 0:
            print(f"\n=== {sym} : no fills collected ===")
            continue
        analyse(sym, fills[sym], base[sym])
