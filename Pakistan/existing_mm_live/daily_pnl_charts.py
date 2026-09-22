# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# daily_pnl_charts.py -- per-DAY P&L series for naive + micro configs, both symbols,
# to expose whether a few outlier days (e.g. the Iran/oil halt days) dominate the
# total P&L that the confirmation reported as an average.
#
# For each (strategy, symbol, day) it runs the REAL backtester and records the true
# EOD P&L (bt.eod["equity_liquidated"]). Writes a tidy per-day CSV, then generates,
# per (strategy, symbol): daily P&L bars, cumulative equity curve, weekly P&L, and
# a P&L distribution histogram -- with the 5 known market-halt days marked.
#
# Build-once per symbol-day (events reused across the 4 configs) so it runs ~40 min.
# Heartbeat + timer built in. Run from existing_mm_live/:  python daily_pnl_charts.py

# paths
from pathlib import Path
# timing
import time
# frames + arrays
import pandas as pd
import numpy as np
# driver + engine + strategies
import run_legacy_mm as R
from mm_backtest import Backtester, LatencyModel, NaiveSymmetricMM
from micro_mm import MicrostructureMM

# raw store
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# results dir for the CSV + charts
# Resolve this filesystem path through the canonical checkout/data configuration.
OUT_DIR = Path(str(_hft_paths.RESULTS_ROOT / 'daily_pnl'))
# make sure it exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# pilot symbols
SYMBOLS = ["PPL", "UBL"]
# run set: naive + 3 micro configs. (name, overrides, label)
RUNSET = [
    ("naive", {}, "naive"),
    ("micro", {"min_edge_pct": 0.0003, "improve_ticks": 0.0}, "micro_me0.0003"),
    ("micro", {"min_edge_pct": 0.0005, "improve_ticks": 0.0}, "micro_me0.0005"),
    ("micro", {"min_edge_pct": 0.0007, "improve_ticks": 0.0}, "micro_me0.0007"),
]
# the 5 known market-wide halt days (Iran/oil vol shock) -- prime outlier suspects
HALT_DAYS = {"2026-03-02", "2026-03-09", "2026-03-10", "2026-04-01", "2026-04-08"}


# compact mm:ss
def _fmt(sec):
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# strategy builder
def make_strategy(name, session_ms, overrides):
    # naive ignores overrides
    if name == "naive":
        return NaiveSymmetricMM(**R.STRAT)
    # micro with config overrides
    params = dict(R.MICRO_PARAMS); params.update(overrides)
    return MicrostructureMM(session_ms=session_ms, **params)


# collect per-day P&L for all (strategy, symbol, day)
def collect():
    # all dates
    dates = R.discover_dates()
    # tidy records: one row per (strategy, symbol, date)
    records = []
    # timers
    t0 = time.perf_counter()
    # symbol-day counter + total for ETA
    sd = 0
    sd_total = len(dates) * len(SYMBOLS)
    # announce
    print(f"daily P&L: {len(RUNSET)} configs x {len(SYMBOLS)} symbols x {len(dates)} days, "
          f"build-once/symbol-day\n", flush=True)
    # OUTER: dates
    for date in dates:
        # open once
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # MIDDLE: symbols
        for sym in SYMBOLS:
            # ---- build events ONCE per symbol-day ----
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            if len(t) == 0 or len(s) == 0:
                continue
            events, snap_groups, t = R.build_events(u, s, t)
            cont = t[t["initiator"] != "AUCTION"]
            if len(cont) == 0:
                continue
            t0w, t1w = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
            # ---- INNER: each config on the reused events ----
            for name, overrides, label in RUNSET:
                # fresh seeded latency per run
                cfg = dict(R.CFG, session=(t0w, t1w),
                           latency_model=LatencyModel(seed=R.LATENCY_SEED))
                # run
                bt = Backtester(make_strategy(name, (t0w, t1w), overrides), cfg)
                fills, equity, stats = bt.run(events, snap_groups)
                # read the day's true EOD P&L
                if bt.eod is not None:
                    # realizable P&L (with book-walk liquidation)
                    day_pnl = float(bt.eod["equity_liquidated"])
                    # position carried to close (context for outliers)
                    eod_pos = float(bt.eod["pos_at_close"])
                    # clean-liquidation flag
                    clean = bool(bt.eod["liquidation_clean"])
                else:
                    # no eod report -> zero, unknown
                    day_pnl, eod_pos, clean = 0.0, 0.0, None
                # record one tidy row
                records.append({
                    "date": date, "symbol": sym, "strategy": label,
                    "pnl": day_pnl, "n_fills": len(fills) if fills is not None else 0,
                    "eod_pos": eod_pos, "liq_clean": clean,
                    "is_halt_day": date in HALT_DAYS,
                })
            # heartbeat every 25 symbol-days
            sd += 1
            if sd % 25 == 0:
                el = time.perf_counter() - t0
                proj = el / sd * sd_total
                print(f"  {sd}/{sd_total} symbol-days  elapsed {_fmt(el)}  "
                      f"ETA {_fmt(proj - el)}", flush=True)
    # assemble tidy frame
    df = pd.DataFrame(records)
    # parse date for time-ordering + weekly grouping
    df["date"] = pd.to_datetime(df["date"])
    # sort for cumulative sums
    df = df.sort_values(["strategy", "symbol", "date"]).reset_index(drop=True)
    return df


# make charts per (strategy, symbol). Guarded so the CSV still lands if matplotlib absent.
def make_charts(df):
    # try to import matplotlib; if missing, skip charts but keep the CSV
    try:
        import matplotlib
        # non-interactive backend (writing files, no display)
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except Exception as e:
        print(f"matplotlib unavailable ({e}); CSV written, charts skipped.", flush=True)
        return
    # one figure per (strategy, symbol): 4 panels
    for (strat, sym), g in df.groupby(["strategy", "symbol"]):
        # order by date
        g = g.sort_values("date")
        # a 2x2 panel figure
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        # title
        fig.suptitle(f"{strat}  {sym}   total P&L = {g['pnl'].sum():,.0f} PKR "
                     f"({len(g)} days)", fontsize=13)
        # --- panel 1: daily P&L bars, halt days highlighted ---
        colors = ["crimson" if h else "steelblue" for h in g["is_halt_day"]]
        ax[0, 0].bar(g["date"], g["pnl"], color=colors, width=1.0)
        ax[0, 0].axhline(0, color="black", lw=0.6)
        ax[0, 0].set_title("daily P&L (red = market-halt day)")
        ax[0, 0].set_ylabel("PKR")
        # --- panel 2: cumulative equity curve ---
        ax[0, 1].plot(g["date"], g["pnl"].cumsum(), color="darkgreen")
        ax[0, 1].axhline(0, color="black", lw=0.6)
        ax[0, 1].set_title("cumulative P&L (equity curve)")
        ax[0, 1].set_ylabel("PKR")
        # --- panel 3: weekly P&L (sum within ISO week) ---
        wk = g.set_index("date")["pnl"].resample("W").sum()
        ax[1, 0].bar(wk.index, wk.values, color="slateblue", width=5.0)
        ax[1, 0].axhline(0, color="black", lw=0.6)
        ax[1, 0].set_title("weekly P&L")
        ax[1, 0].set_ylabel("PKR")
        # --- panel 4: daily P&L distribution histogram ---
        ax[1, 1].hist(g["pnl"], bins=40, color="gray", edgecolor="black")
        ax[1, 1].axvline(0, color="black", lw=0.6)
        ax[1, 1].axvline(g["pnl"].mean(), color="red", lw=1.2,
                         label=f"mean {g['pnl'].mean():,.0f}")
        ax[1, 1].axvline(g["pnl"].median(), color="blue", lw=1.2,
                         label=f"median {g['pnl'].median():,.0f}")
        ax[1, 1].legend()
        ax[1, 1].set_title("daily P&L distribution")
        ax[1, 1].set_xlabel("PKR/day")
        # tidy date axes
        for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
            a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        # layout + save
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fn = OUT_DIR / f"pnl_{strat}_{sym}.png"
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        print(f"  chart: {fn}", flush=True)


# outlier summary: for each (strategy, symbol), how much of total P&L is the worst
# few days? Answers "is one day destroying it?" numerically, alongside the charts.
def outlier_summary(df):
    print("\n=== outlier check: worst-day concentration ===", flush=True)
    # per cell
    for (strat, sym), g in df.groupby(["strategy", "symbol"]):
        # total
        tot = g["pnl"].sum()
        # worst single day
        worst = g.nsmallest(1, "pnl").iloc[0]
        # worst 5 days combined
        worst5 = g.nsmallest(5, "pnl")["pnl"].sum()
        # P&L excluding the 5 known halt days
        ex_halt = g[~g["is_halt_day"]]["pnl"].sum()
        # report
        print(f"  {strat:16} {sym}: total={tot:>12,.0f} | "
              f"worst day={worst['pnl']:>11,.0f} ({worst['date'].date()}) | "
              f"worst5={worst5:>12,.0f} | ex-halt-days={ex_halt:>12,.0f}", flush=True)


# main
def main():
    # collect per-day P&L
    df = collect()
    # write the tidy CSV (the durable artifact)
    csv = OUT_DIR / "daily_pnl.csv"
    df.to_csv(csv, index=False)
    print(f"\nsaved per-day CSV: {csv}  ({len(df):,} rows)", flush=True)
    # numeric outlier summary
    outlier_summary(df)
    # charts
    make_charts(df)
    print(f"\ncharts + CSV in: {OUT_DIR}", flush=True)


# entry point
if __name__ == "__main__":
    main()
