# ============================================================================
# fill_probability.py -- EMPIRICAL FILL RATES by the state a quote was posted into.
# ----------------------------------------------------------------------------
# Runs the production strategy with log_fill_state=True so the engine records,
# for EVERY posted quote (filled or not), the market at the instant it joined the
# queue: shares ahead of it, own/opposite touch depth, spread, L1 OBI, how far
# inside the touch it sat, and the session bucket. Paired with the order log's
# end_reason (filled / cancelled) this is an UNBIASED fill-probability dataset --
# the losers are in it, not just the winners.
#
# Two horizons, as agreed:
#   P(fill before replacement)  -- primary; matches how quotes actually live
#   P(fill within 5s of going live) -- secondary; fixed, comparable across configs
#
# THE HEADLINE NUMBER: fill rate when the book is FAVORABLE to the quote vs when
# it is ADVERSE. "You get filled least where you most want it" (the maker's
# curse) has been the assumed explanation for seven failed tests and has never
# been measured. This measures it.
#   favorable = (BUY and obi1>0) or (SELL and obi1<0)   [book leans our way]
#   adverse   = (BUY and obi1<0) or (SELL and obi1>0)   [book leans against us]
#
# Outputs (config_pk paths): fill_prob_orders.csv (every quote + state + outcome),
# fill_prob_table.csv (fill rate by state bucket), printed summary + PNG.
# USAGE: python fill_probability.py --self-test | --smoke | --run
# ============================================================================

# CLI parsing
import argparse
# timing / heartbeat
import time
# timestamps on log lines
from datetime import datetime
# parallel over dates
from multiprocessing import Pool
# numerics
import numpy as np
# dataframes
import pandas as pd
# headless plotting
import matplotlib
# no display
matplotlib.use("Agg")
# plot API
import matplotlib.pyplot as plt

# central paths
from config_pk import PARSED_ROOT, RESULTS_ROOT


# bracketed HH:MM:SS stamp
def _ts():
    # current local time
    return datetime.now().strftime("[%H:%M:%S]")


# output directory
OUT_DIR = RESULTS_ROOT / "diagnostics"
# the 38-name shortlist
WATCHLIST = RESULTS_ROOT / "mm_watchlist_final.csv"
# secondary fixed horizon (ms)
FIXED_H_MS = 5000
# min quotes in a cell before we print its rate
MIN_N = 30
# clip multiplier (production 3x median trade size)
CLIP_MULT = 3.0
# trailing days for the median trade size
TRAIL_DAYS = 10
# defaults
MAX_DAYS = 20
WORKERS = 6


# ---------------------------------------------------------------- labels -----
# turn the raw order log into the analysis frame: outcome flags + state buckets
def label_orders(ol, med_trade=None):
    # need the state columns (only present when log_fill_state was on)
    need = {"end_reason", "t_live", "t_end", "side", "obi1", "ahead_qty", "spread", "bucket"}
    # bail if the log lacks the state capture
    if ol is None or len(ol) == 0 or not need.issubset(ol.columns):
        # nothing usable
        return None
    # only orders that actually RESTED (t_live set); rejected crossers never posted
    d = ol[ol["t_live"].notna()].copy()
    # nothing rested
    if len(d) == 0:
        # empty
        return None
    # PRIMARY label: filled before it was replaced/cancelled
    d["filled"] = (d["end_reason"] == "filled").astype(int)
    # lifetime in ms (NaN if never terminated -- e.g. still live at close)
    d["life_ms"] = d["t_end"] - d["t_live"]
    # SECONDARY label: filled within the fixed horizon of going live
    d["filled_5s"] = ((d["end_reason"] == "filled") & (d["life_ms"] <= FIXED_H_MS)).astype(int)
    # FAVORABLE / ADVERSE / NEUTRAL from the quote's side and the book lean
    buy = d["side"] == "BUY"
    # book leans toward a BUY when obi>0; toward a SELL when obi<0
    fav = (buy & (d["obi1"] > 0.10)) | (~buy & (d["obi1"] < -0.10))
    # book leans against a BUY when obi<0; against a SELL when obi>0
    adv = (buy & (d["obi1"] < -0.10)) | (~buy & (d["obi1"] > 0.10))
    # three-way state
    d["lean"] = np.where(fav, "favorable", np.where(adv, "adverse", "neutral"))
    # SESSION BUCKET from the quote's live timestamp (the strategy never set
    # current_bucket, so the engine default "middle" was wrong for every row).
    # Derive it per name-day: first15 / preclose45 / last15 / middle off the
    # day's own [min, max] t_live span (session-relative, no calendar needed).
    # NOTE: label_orders is called on ONE symbol-day (from _one), and symbol/date
    # are added by the caller AFTER this returns -- so we must NOT depend on them
    # here (that was the bug: the guard required columns that don't exist yet, so
    # the block silently never ran and every row kept the engine default "middle").
    # A single symbol-day means the whole frame's [min,max] t_live IS the session.
    if "t_live" in d.columns and d["t_live"].notna().any():
        # session open/close from the observed quote times (this one symbol-day)
        tl = d["t_live"].to_numpy()
        # scalar session bounds (not per-group -- there is only one group)
        t0 = np.nanmin(tl); t1 = np.nanmax(tl)
        # 15/45/15-min edges in ms
        F = 15 * 60 * 1000; P = 45 * 60 * 1000; L = 15 * 60 * 1000
        # default middle
        bk = np.full(len(d), "middle", dtype=object)
        # first 15 min after the open
        bk[tl <= t0 + F] = "first15"
        # the 45->15 min window before the close
        bk[tl >= t1 - P] = "preclose45"
        # last 15 min before the close (overwrites preclose in its range)
        bk[tl >= t1 - L] = "last15"
        # overwrite the (wrong) engine-default bucket
        d["bucket"] = bk
    # queue-ahead in MEDIAN-TRADE-SIZE units (market-state scale, stable across
    # our inventory/config -- fixes the circular clip normalization). This is
    # "how many typical trades of size rest ahead of me", the true waiting unit.
    d["med_trade"] = float(med_trade) if med_trade and med_trade > 0 else np.nan
    # queue depth as a multiple of the median trade
    d["ahead_x_med"] = d["ahead_qty"] / d["med_trade"]
    # 0 = front; then <2, 2-5, 5-10, 10+ typical-trades ahead
    d["ahead_bkt"] = pd.cut(d["ahead_x_med"].fillna(0), [-0.01, 0.01, 2, 5, 10, np.inf],
                            labels=["front", "<2med", "2-5med", "5-10med", "10+med"])
    # spread in ticks (PSX tick 0.01)
    d["spread_ticks"] = (d["spread"] / 0.01).round()
    # spread buckets
    d["spread_bkt"] = pd.cut(d["spread_ticks"], [0, 1.5, 3.5, 7.5, np.inf],
                             labels=["1t", "2-3t", "4-7t", "8t+"])
    # the labelled frame
    return d


# ------------------------------------------------------------- per-day run ----
_R = None; _H = None; _CALIB = None
# pool init: driver, harness, calibration (same as the sweeps use)
def _init(calib):
    # expose globals
    global _R, _H, _CALIB
    # driver + harness
    import run_legacy_mm as R, mm_harness as H
    # point at the configured store
    R.PARSED_ROOT = PARSED_ROOT
    # use the micro strategy
    R.USE_MICRO = True
    # stash
    _R = R; _H = H; _CALIB = calib


# run one symbol-day with fill-state logging on; return the labelled order log
def _one(date, sym, dsets):
    # calibration tables
    scales = _CALIB["scales"]; profiles = _CALIB["profiles"]; windows = _CALIB["windows"]
    # session segments + trade-size stats
    segments = _CALIB["segments"]; all_dates = _CALIB["all_dates"]; tstats = _CALIB["tstats"]
    # need calibration for this name
    if sym not in scales or sym not in profiles:
        # skip
        return None
    # the day's segments
    segs = segments.get(str(date))
    # no segments -> skip
    if segs is None:
        # skip
        return None
    # trailing median trade size -> clip
    med = _H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
    # unusable
    if med is None or med <= 0:
        # skip
        return None
    # clip in shares
    clip = max(1, int(round(CLIP_MULT * med)))
    # production params + the ONLY change: turn on state logging
    params = _H.build_micro_params(clip, scales[sym], profiles[sym], windows.get(sym, (5.0, 1.0)), segs,
                                   overrides={"log_fill_state": True})
    # production OBI throttle on (the live config)
    params.update(dict(obi_throttle=True, ofi_throttle=False, obi_throttle_thresh=0.15,
                       throttle_frac=0.5, throttle_hold_ms=300.0))
    # run the day
    dr = _H.run_symbol_day(date, sym, dsets, params)
    # nothing
    if dr is None or getattr(dr, "order_log", None) is None:
        # skip
        return None
    # per-name median trade size (market-state scale, NOT our clip) for the
    # queue-depth normalization: how many TYPICAL TRADES rest ahead of us.
    med_trade = _H.trailing_median(tstats[sym], all_dates, date, TRAIL_DAYS)
    # label the order log, normalizing queue depth by median trade size
    lab = label_orders(dr.order_log, med_trade)
    # skip empties
    if lab is None:
        # nothing
        return None
    # tag
    lab["symbol"] = sym; lab["date"] = str(date)
    # ---- HISTORIC-FALLBACK expected-wait (production-realistic) --------------
    # For each swept window, when the live rate was CENSORED (no clearing trades
    # in the window), fall back to the per-bucket HISTORIC clearing rate instead
    # of calling it "won't fill". The profile is TOTAL shares/min in the bucket;
    # the clearing side (the aggressor side that fills our resting order) is ~half
    # the flow, so we halve it. This column is for the PRODUCTION estimate and to
    # measure how often the fallback fires -- it is kept SEPARATE from the pure-
    # live ewait_* columns so the window calibration stays uncontaminated.
    prof = _CALIB["profiles"].get(sym)
    if prof is not None:
        # profiles[sym] is a POSITIONAL TUPLE from mm_harness._load_table, ordered
        # (vol_first15, vol_middle, vol_preclose45, vol_last15) -- NOT a dict.
        # per-bucket TOTAL shares/min -> clearing-side ~= half the flow.
        bucket_rate = {
            "first15": float(prof[0]) * 0.5,
            "middle": float(prof[1]) * 0.5,
            "preclose45": float(prof[2]) * 0.5,
            "last15": float(prof[3]) * 0.5,
        }
        # this quote's historic clearing rate from its session bucket
        hist_rate = lab["bucket"].map(bucket_rate)
        # shares that must clear before us (queue ahead + our own size)
        to_clear = lab["ahead_qty"] + lab["qty"]
        # for each window, build the production column: live ewait if not censored,
        # else the historic-rate estimate (finite when the bucket rate is > 0)
        for wl in [c[len("ewait_"):] for c in lab.columns if c.startswith("ewait_")]:
            # historic expected-wait (minutes) = shares to clear / historic rate
            hist_ew = to_clear / hist_rate.replace(0.0, np.nan)
            # production column: live where available, historic fallback where censored
            lab[f"ewait_{wl}_prod"] = np.where(lab[f"cens_{wl}"] == 1,
                                               hist_ew, lab[f"ewait_{wl}"])
            # flag: did the historic fallback actually fire for this quote?
            lab[f"usedhist_{wl}"] = ((lab[f"cens_{wl}"] == 1) & hist_rate.gt(0)).astype(int)
    # keep the analysis columns only (the log is big)
    # base analysis columns
    keep = ["symbol", "date", "side", "bucket", "lean", "ahead_bkt", "spread_bkt",
            "obi1", "ahead_qty", "qty", "med_trade", "ahead_x_med",
            "spread_ticks", "rel_px", "life_ms", "t_live",
            "filled", "filled_5s", "end_reason"]
    # also carry every window column (live, production-fallback, censor/flags)
    keep += [c for c in lab.columns if c.startswith(("ewait_", "rate_", "cens_", "nclear_", "nevt_", "usedhist_"))]
    # subset
    return lab[[c for c in keep if c in lab.columns]]


# all names for one date
def _work_date(date):
    # datasets
    dsets = _R.open_datasets(date)
    # missing
    if dsets is None:
        # nothing
        return []
    # collected frames
    out = []
    # loop the universe
    for sym in _CALIB["names"]:
        # guard per symbol
        try:
            # run
            df = _one(date, sym, dsets)
        except Exception as e:
            # report + continue
            print(_ts() + f"SKIP {date} {sym}: {e!r}")
            # next
            continue
        # keep
        if df is not None and len(df):
            # collect
            out.append(df)
    # this date's frames
    return out


# ------------------------------------------------------------- reporting ----
# fill rate for a subset, with a wilson-ish SE and count
def _rate(sub, col):
    # count
    n = len(sub)
    # too few
    if n < MIN_N:
        # not reportable
        return np.nan, np.nan, n
    # fill rate
    p = sub[col].mean()
    # binomial standard error
    se = np.sqrt(max(p * (1 - p), 1e-12) / n)
    # rate, se, n
    return float(p), float(se), n


# day-as-unit fill rate: mean of per-name-day rates (honest error bars)
def _rate_dau(sub, col):
    # per name-day rate, requiring MIN_N quotes that day
    g = sub.groupby(["date", "symbol"])[col].agg(["mean", "count"])
    # keep solid days
    g = g[g["count"] >= MIN_N]
    # values
    v = g["mean"].to_numpy()
    # need >=2 name-days
    if v.size < 2:
        # not reportable
        return np.nan, np.nan, v.size
    # mean + se across name-days
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), v.size


# the full printed report + saved tables + chart
def _report(df, out_dir):
    # ensure dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # save every quote with its state + outcome
    df.to_parquet(out_dir / "fill_prob_orders.parquet", index=False)
    # headline counts
    n = len(df); nf = int(df["filled"].sum())
    print(_ts() + "===== FILL PROBABILITY -- empirical, from every posted quote =====")
    print(_ts() + f"  quotes posted (rested): {n:,}   filled before replacement: {nf:,} ({100*nf/max(n,1):.1f}%)")
    # ---- THE HEADLINE: favorable vs adverse (day-as-unit) ----
    print(_ts() + "\n  --- FILL RATE by BOOK LEAN (the maker's-curse test; day-as-unit) ---")
    print(_ts() + "  lean        P(fill before replace)    P(fill within 5s)   [name-days]")
    rows = []
    for lean in ["favorable", "neutral", "adverse"]:
        # this lean
        sub = df[df["lean"] == lean]
        # both horizons
        p1, s1, k1 = _rate_dau(sub, "filled"); p2, s2, k2 = _rate_dau(sub, "filled_5s")
        # record
        rows.append(dict(lean=lean, p_fill=p1, se=s1, p_fill_5s=p2, se_5s=s2, name_days=k1))
        # print
        print(_ts() + f"  {lean:10s}   {100*p1:5.1f}% +/- {100*s1:3.1f}          {100*p2:5.1f}% +/- {100*s2:3.1f}      [{k1}]")
    # the ratio that matters
    pf = rows[0]["p_fill"]; pa = rows[2]["p_fill"]
    # guard
    if not (np.isnan(pf) or np.isnan(pa)) and pf > 0:
        # adverse-to-favorable fill ratio
        print(_ts() + f"  => you are filled {pa/pf:.2f}x as often when the book is AGAINST you as when it favors you.")
        print(_ts() + "     (>1 confirms the maker's curse; ~1 refutes it.)")
    # save
    pd.DataFrame(rows).to_parquet(out_dir / "fill_prob_by_lean.parquet", index=False)
    # ---- by queue position ----
    print(_ts() + "\n  --- FILL RATE by QUEUE POSITION (shares ahead, in clips) ---")
    tab = []
    for b in ["front", "<2med", "2-5med", "5-10med", "10+med"]:
        # this bucket
        sub = df[df["ahead_bkt"] == b]
        # rate
        p, se, k = _rate_dau(sub, "filled")
        # record + print
        tab.append(dict(dim="ahead", value=b, p_fill=p, se=se, name_days=k))
        print(_ts() + f"  {b:10s}   {100*p:5.1f}% +/- {100*se:3.1f}   [{k}]")
    # ---- by spread ----
    print(_ts() + "\n  --- FILL RATE by SPREAD (ticks) ---")
    for b in ["1t", "2-3t", "4-7t", "8t+"]:
        # this bucket
        sub = df[df["spread_bkt"] == b]
        # rate
        p, se, k = _rate_dau(sub, "filled")
        # record + print
        tab.append(dict(dim="spread", value=b, p_fill=p, se=se, name_days=k))
        print(_ts() + f"  {b:10s}   {100*p:5.1f}% +/- {100*se:3.1f}   [{k}]")
    # ---- by session bucket ----
    print(_ts() + "\n  --- FILL RATE by SESSION BUCKET ---")
    for b in ["first15", "middle", "preclose45", "last15"]:
        # this bucket
        sub = df[df["bucket"] == b]
        # rate
        p, se, k = _rate_dau(sub, "filled")
        # record + print
        tab.append(dict(dim="bucket", value=b, p_fill=p, se=se, name_days=k))
        print(_ts() + f"  {b:10s}   {100*p:5.1f}% +/- {100*se:3.1f}   [{k}]")
    # ---- lean x queue: the interaction (is the curse a queue effect?) ----
    print(_ts() + "\n  --- LEAN x QUEUE (does the curse come from being buried in the queue?) ---")
    print(_ts() + "  ahead       favorable   adverse")
    for b in ["front", "<2med", "2-5med", "5-10med", "10+med"]:
        # favorable and adverse within this queue bucket
        f = df[(df["ahead_bkt"] == b) & (df["lean"] == "favorable")]
        a = df[(df["ahead_bkt"] == b) & (df["lean"] == "adverse")]
        # rates
        pf_, _, kf = _rate_dau(f, "filled"); pa_, _, ka = _rate_dau(a, "filled")
        # print
        print(_ts() + f"  {b:10s}   {100*pf_:5.1f}%      {100*pa_:5.1f}%     [nd {kf}/{ka}]")
    # save the table
    pd.DataFrame(tab).to_parquet(out_dir / "fill_prob_table.parquet", index=False)
    # ---- ARRIVAL-RATE WINDOW CALIBRATION -----------------------------------
    # Which (T, N-trade) window's expected-wait best forecasts actual fills?
    # Censoring-robust: we do NOT regress on fill TIME (censored for unfilled
    # quotes). Instead, per window we bucket quotes by predicted expected-wait and
    # check the realized fill rate is MONOTONE decreasing (short wait -> fills
    # more). A good window shows a steep, monotone gradient AND its censored
    # quotes (rate 0) fill rarely. The best window = steepest monotone spread
    # between the shortest-wait and longest-wait deciles.
    wlabs = sorted({c[len("ewait_"):] for c in df.columns if c.startswith("ewait_")})
    if wlabs:
        print(_ts() + "\n===== ARRIVAL-RATE WINDOW CALIBRATION (does expected-wait predict fills?) =====")
        print(_ts() + "  per window: fill rate by expected-wait bucket (finite waits only), then censored.")
        cal_rows = []
        for wl in wlabs:
            ew = df[f"ewait_{wl}"]; cens = df[f"cens_{wl}"]
            # finite-wait quotes (not censored)
            fin = df[(cens == 0) & np.isfinite(ew)].copy()
            # censored quotes (no clearing trades in the window)
            cen = df[cens == 1]
            # expected-wait buckets in MINUTES
            fin["ew_bkt"] = pd.cut(fin[f"ewait_{wl}"],
                                   [-0.01, 1, 3, 10, 30, np.inf],
                                   labels=["<1m", "1-3m", "3-10m", "10-30m", "30m+"])
            print(_ts() + f"  --- window {wl} ---")
            grad = {}
            for b in ["<1m", "1-3m", "3-10m", "10-30m", "30m+"]:
                sub = fin[fin["ew_bkt"] == b]
                p, se, k = _rate_dau(sub, "filled")
                grad[b] = p
                if not np.isnan(p):
                    print(_ts() + f"    ewait {b:8s}  fill {100*p:5.2f}% +/- {100*se:4.2f}   [nd {k}]")
            # censored fill rate
            pc, sec, kc = _rate_dau(cen, "filled")
            print(_ts() + f"    CENSORED(rate=0) fill {100*pc:5.2f}% +/- {100*sec:4.2f}   [nd {kc}]  (share of quotes: {100*len(cen)/max(len(df),1):.1f}%)")
            # monotonicity + spread score: shortest minus longest finite bucket
            vals = [grad.get(b) for b in ["<1m","1-3m","3-10m","10-30m","30m+"]]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            mono = all(vals[i] >= vals[i+1] - 1e-9 for i in range(len(vals)-1)) if len(vals) > 1 else False
            spread = (vals[0] - vals[-1]) if len(vals) > 1 else np.nan
            cal_rows.append(dict(window=wl, monotone=mono, spread=spread,
                                 censored_fill=pc, censored_share=len(cen)/max(len(df),1)))
            print(_ts() + f"    -> monotone={mono}  short-minus-long spread={100*spread:.2f}pp")
        # the winner: steepest monotone spread
        cal = pd.DataFrame(cal_rows)
        cal.to_parquet(out_dir / "fill_prob_window_calib.parquet", index=False)
        best = cal[cal["monotone"]].sort_values("spread", ascending=False)
        if len(best):
            print(_ts() + f"  BEST WINDOW: {best.iloc[0]['window']} (monotone, widest fill-rate spread "
                  f"{100*best.iloc[0]['spread']:.2f}pp) -> this is what 'recent' should mean.")
        else:
            print(_ts() + "  No window gave a cleanly monotone gradient -- expected-wait may not forecast fills here.")
    # chart
    _plot(df, out_dir)
    # where
    print(_ts() + f"\n[fill-prob] outputs -> {out_dir}")


# chart: fill rate by lean, and by queue position split by lean
def _plot(df, out_dir):
    # two panels
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    # panel 1: by lean
    leans = ["favorable", "neutral", "adverse"]; vals = []; errs = []
    for l in leans:
        # rate
        p, se, _ = _rate_dau(df[df["lean"] == l], "filled"); vals.append(100 * p); errs.append(100 * se)
    # bars
    a1.bar(leans, vals, yerr=errs, capsize=5, color=["#2e7d32", "#888", "#c0392b"])
    a1.set_ylabel("% of quotes filled before replacement"); a1.set_title("Fill rate by book lean (maker's-curse test)")
    a1.grid(alpha=0.25, axis="y")
    # panel 2: by queue position, favorable vs adverse
    qb = ["front", "<2med", "2-5med", "5-10med", "10+med"]; x = np.arange(len(qb))
    fv = [100 * _rate_dau(df[(df.ahead_bkt == b) & (df.lean == "favorable")], "filled")[0] for b in qb]
    av = [100 * _rate_dau(df[(df.ahead_bkt == b) & (df.lean == "adverse")], "filled")[0] for b in qb]
    # grouped bars
    a2.bar(x - 0.2, fv, 0.4, label="favorable", color="#2e7d32"); a2.bar(x + 0.2, av, 0.4, label="adverse", color="#c0392b")
    a2.set_xticks(x); a2.set_xticklabels(qb); a2.set_ylabel("% filled"); a2.set_title("Fill rate by queue position, favorable vs adverse")
    a2.legend(); a2.grid(alpha=0.25, axis="y")
    # save
    fig.tight_layout(); fig.savefig(out_dir / "fill_probability.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------- drivers ------
# build the calibration bundle (same loaders the sweeps use)
def _calib(names):
    # harness + driver
    import run_legacy_mm as R, mm_harness as H
    # point at store
    R.PARSED_ROOT = PARSED_ROOT
    # dates
    all_dates = R.discover_dates()
    # bundle
    return dict(scales=H.load_scales(), profiles=H.load_profiles(), windows=H.load_windows(),
                segments=H.load_segments(), all_dates=all_dates,
                tstats=H.trailing_median_trade_size(all_dates, names, TRAIL_DAYS), names=names), all_dates


# full run
def run_real(out_dir=OUT_DIR, symbols=None, workers=WORKERS, max_days=MAX_DAYS):
    # universe
    names = symbols or sorted(pd.read_csv(WATCHLIST)["symbol"].dropna().astype(str).unique().tolist())
    # calibration
    print(_ts() + "pre-pass: calibration + trailing median trade size")
    calib, all_dates = _calib(names)
    # sample days
    dates = all_dates
    if max_days and len(dates) > max_days:
        # stride
        step = max(1, len(dates) // max_days); dates = dates[::step][:max_days]
    # announce
    print(_ts() + f"{len(names)} names x {len(dates)} dates, {workers} workers -- fill-state logging ON")
    # collect
    frames = []; t0 = time.perf_counter()
    # pool
    with Pool(processes=workers, initializer=_init, initargs=(calib,)) as pool:
        # counter
        done = 0
        # consume
        for res in pool.imap_unordered(_work_date, dates):
            # collect
            frames.extend(res); done += 1
            # progress
            el = (time.perf_counter() - t0) / 60.0
            print(_ts() + f"  date {done}/{len(dates)} ({el:.1f} min, ETA {el/done*(len(dates)-done):.1f} min)")
    # nothing
    if not frames:
        # stop
        print(_ts() + "no quotes logged -- is log_fill_state reaching the engine?"); return
    # report
    _report(pd.concat(frames, ignore_index=True), out_dir)


# one stock-day, timed
def smoke(symbols=None):
    # universe
    names = symbols or sorted(pd.read_csv(WATCHLIST)["symbol"].dropna().astype(str).unique().tolist())
    # calibration
    calib, all_dates = _calib(names)
    # init this process
    _init(calib)
    # mid date
    date = all_dates[len(all_dates) // 2]
    # datasets
    dsets = _R.open_datasets(date)
    # time one
    t0 = time.perf_counter(); df = _one(date, names[0], dsets); dt = time.perf_counter() - t0
    # report
    print(_ts() + f"[smoke] {date} {names[0]} in {dt:.1f}s: {0 if df is None else len(df)} quotes logged")
    # preview
    if df is not None and len(df):
        # fill rate + lean split
        print(_ts() + f"[smoke] fill rate {100*df['filled'].mean():.1f}%  |  lean counts: {df['lean'].value_counts().to_dict()}")
        # confirm state columns present
        print(_ts() + f"[smoke] state cols present: {[c for c in ['obi1','ahead_qty','spread_ticks','rel_px'] if c in df.columns]}")


# validate labelling + lean mapping on a hand-built order log
def self_test():
    # a fake order log with the engine's columns
    ol = pd.DataFrame({
        "oid": [1, 2, 3, 4, 5, 6],
        "side": ["BUY", "BUY", "SELL", "SELL", "BUY", "BUY"],
        "qty": [100, 100, 100, 100, 100, 100],
        "t_live": [0, 0, 0, 0, 0, None],           # #6 never rested (rejected)
        "t_end": [2000, 8000, 3000, None, 1000, None],
        "end_reason": ["filled", "filled", "cancelled", None, "filled", None],
        "obi1": [0.5, -0.5, 0.5, -0.5, 0.0, 0.5],
        "ahead_qty": [0, 300, 150, 1500, 50, 0],
        "spread": [0.01, 0.03, 0.05, 0.10, 0.02, 0.01],
        "bucket": ["middle"] * 6,
        "symbol": ["X"] * 6,
        "date": ["2026-01-01"] * 6,
    })
    # label
    d = label_orders(ol, med_trade=100.0)
    # rejected order (#6) excluded
    assert len(d) == 5 and 6 not in d["oid"].values
    print(_ts() + "[self-test] rejected/never-rested order excluded  OK")
    # primary label
    assert d.set_index("oid")["filled"].to_dict() == {1: 1, 2: 1, 3: 0, 4: 0, 5: 1}
    # 5s label: #2 filled at 8000ms -> NOT within 5s
    assert d.set_index("oid")["filled_5s"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0, 5: 1}
    print(_ts() + "[self-test] filled / filled_5s labels correct (8s fill excluded from 5s)  OK")
    # lean: BUY+obi>0 = favorable; BUY+obi<0 = adverse; SELL+obi>0 = adverse; SELL+obi<0 = favorable; obi=0 neutral
    lean = d.set_index("oid")["lean"].to_dict()
    assert lean == {1: "favorable", 2: "adverse", 3: "adverse", 4: "favorable", 5: "neutral"}, lean
    print(_ts() + "[self-test] lean mapping (BUY/SELL x OBI sign) correct  OK")
    # queue buckets: 0 ahead -> front; 300/100=3 clips -> 2-5clip; 1500/100=15 -> 10+clip
    ab = d.set_index("oid")["ahead_bkt"].astype(str).to_dict()
    assert ab[1] == "front" and ab[2] == "2-5med" and ab[4] == "10+med", ab
    print(_ts() + "[self-test] queue-ahead buckets correct  OK")
    # spread ticks: 0.03 -> 3 ticks -> "2-3t"; 0.10 -> 10 -> "8t+"
    sb = d.set_index("oid")["spread_bkt"].astype(str).to_dict()
    assert sb[2] == "2-3t" and sb[4] == "8t+", sb
    print(_ts() + "[self-test] spread buckets correct  OK")
    print(_ts() + "[self-test] ALL ASSERTIONS PASSED.")


# entry
if __name__ == "__main__":
    # parser
    ap = argparse.ArgumentParser()
    # flags
    ap.add_argument("--self-test", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true"); ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS); ap.add_argument("--days", type=int, default=MAX_DAYS)
    # parse
    a = ap.parse_args()
    # dispatch
    if a.smoke: smoke(symbols=a.symbols)
    elif a.self_test or not a.run: self_test()
    if a.run: run_real(symbols=a.symbols, workers=a.workers, max_days=(a.days or None))
