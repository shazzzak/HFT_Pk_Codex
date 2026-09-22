# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
#!/usr/bin/env python3
# ============================================================================
# leadlag_screen.py -- PREREQUISITE SCREEN for a cross-asset (sector-leader)
# lead-lag DEFENSIVE THROTTLE (framing A). This is a DIAGNOSTIC, not a strategy:
# it answers "is there a fast sector leader whose moves precede a follower by a
# stable, async-robust, economically-meaningful lag?" -- and kills the idea early
# if PSX's synchronicity/coarseness makes the lead-lag an artifact.
#
# THE 4 STEPS (each can KILL the idea; stop at the first failure):
#   0. Leader selection: within each user-defined sector, pick the leader = name
#      with the highest median daily traded value (exchange cum_value). Data-driven,
#      not asserted.
#   1. Update-rate asymmetry: median inter-trade time, leader vs each follower.
#      If the leader does NOT update materially faster, there is no lead to exploit.
#   2. Async-robust lead-lag: Hayashi-Yoshida cross-correlation over a lag grid,
#      per (leader,follower)-day. HY is the fix for non-synchronous trading (the
#      Epps effect) -- a lead-lag "peak" at lag 0, or one that vanishes under HY,
#      was a sampling artifact, not information.
#   3. Day-as-unit stability: is the peak lag's SIGN stable across days? A lag that
#      flips sign day to day is not tradable.
#   4. Economic gate: does the anticipatable move (HY beta x leader per-event vol)
#      clear the fee/capture hurdle? A statistically real 0.2 bps lead is useless.
#
# IMPORTANT (your own hardest-won lesson): on PSX spread capture is the edge and
# directional signals have repeatedly "predicted but not paid." So even a clean
# pass here only justifies a DEFENSIVE throttle test in the engine (framing A) --
# NOT a directional lean. This screen ranks candidates; engine P&L decides.
#
# USAGE: python leadlag_screen.py --selftest | --smoke | --run
# ============================================================================

# CLI
import argparse
# timing
import time
# stamps
from datetime import datetime
# arrays / frames
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG -- EDIT (paths + the SECTOR MAP, which you MUST verify)
# ----------------------------------------------------------------------------
# central paths if available, else the handoff literals
try:
    import config_pk
    PARSED_ROOT = str(getattr(config_pk, "PARSED_ROOT", "") or getattr(config_pk, "PARSED", ""))
    RESULTS_ROOT = str(getattr(config_pk, "RESULTS_ROOT", "") or getattr(config_pk, "RESULTS", ""))
except Exception:
    PARSED_ROOT = RESULTS_ROOT = ""
if not PARSED_ROOT:
    # Resolve this filesystem path through the canonical checkout/data configuration.
    PARSED_ROOT = str(_hft_paths.PARSED_ROOT)
if not RESULTS_ROOT:
    # Resolve this filesystem path through the canonical checkout/data configuration.
    RESULTS_ROOT = str(_hft_paths.RESULTS_ROOT)

# PSX OFFICIAL SECTORS, parsed from the exchange's daily quotation sheet
# (Section 4, MARKET IN DETAIL) for 2026-09-14. All 113 production names
# matched exactly -- no guesses, no UNCLASSIFIED bucket. Only sectors with
# >= 2 names appear (leader + at least one follower); singletons are listed
# in SINGLETON_SECTORS below so nothing looks silently dropped.
SECTORS = {
    # Commercial Banks -- 12 names, 12 with a deliverable future
    "banks": ["AKBL", "BAFL", "BAHL", "BML", "BOP", "FABL", "HBL", "HMB", "MCB", "MEBL", "NBP", "UBL"],
    # Cement -- 11 names, 10 with a deliverable future
    "cement": ["CHCC", "DCL", "DGKC", "FCCL", "FECTC", "KOHC", "LUCK", "MLCF", "PIOC", "POWER", "THCCL"],
    # Technology & Communication -- 11 names, 8 with a deliverable future
    "tech": ["AIRLINK", "AVN", "HUMNL", "NETSOL", "PTC", "SYS", "TELE", "TPL", "TRG", "WTL", "ZAL"],
    # Food & Personal Care Products -- 10 names, 7 with a deliverable future
    "food": ["BBFL", "BNL", "FCEPL", "FFL", "NATF", "PREMA", "QUICE", "TOMCL", "TREET", "UNITY"],
    # Chemical -- 6 names, 4 with a deliverable future
    "chemical": ["EPCL", "GCIL", "GCWL", "GGL", "LCI", "LOTCHEM"],
    # Engineering -- 6 names, 4 with a deliverable future
    "engineering": ["AGHA", "ASL", "BECO", "CSAP", "ISL", "MUGHAL"],
    # Pharmaceuticals -- 6 names, 4 with a deliverable future
    "pharma": ["AGP", "BFBIO", "CPHL", "GLAXO", "HALEON", "SEARL"],
    # Power Generation & Distribution -- 6 names, 5 with a deliverable future
    "power": ["HUBC", "KAPCO", "KEL", "NCPL", "NPL", "SGPL"],
    # Automobile Assembler -- 5 names, 3 with a deliverable future
    "auto_assembler": ["DFML", "GAL", "GHNI", "HCAR", "SAZEW"],
    # Oil & Gas Marketing Companies -- 5 names, 3 with a deliverable future
    "omc": ["APL", "HASCOL", "PSO", "SNGP", "SSGC"],
    # Fertilizer -- 4 names, 3 with a deliverable future
    "fertilizer": ["AHCL", "EFERT", "FATIMA", "FFC"],
    # Inv. Banks / Inv. Cos. / Securities Cos. -- 4 names, 2 with a deliverable future
    "inv_banks": ["ENGROH", "FNEL", "PIAHCLA", "PSX"],
    # Oil & Gas Exploration Companies -- 4 names, 4 with a deliverable future
    "oil_gas_ep": ["MARI", "OGDC", "POL", "PPL"],
    # Refinery -- 4 names, 3 with a deliverable future
    "refinery": ["ATRL", "CNERGY", "NRL", "PRL"],
    # Cable & Electrical Goods -- 3 names, 3 with a deliverable future
    "cable_electrical": ["FCL", "PAEL", "WAVES"],
    # Property -- 3 names, 3 with a deliverable future
    "property": ["JVDC", "PACE", "TPLP"],
    # Textile Composite -- 3 names, 2 with a deliverable future
    "textile_composite": ["ILP", "KOIL", "NML"],
    # Automobile Parts & Accessories -- 2 names, 1 with a deliverable future
    "auto_parts": ["LOADS", "TBL"],
    # Transport -- 2 names, 2 with a deliverable future
    "transport": ["PIBTL", "SLGL"],
}
# Sectors holding only ONE of our names -- no follower, so not screenable.
SINGLETON_SECTORS = {
    "apparel": "IMAGE",   # Apparel
    "glass_ceramics": "TGL",   # Glass & Ceramics
    "insurance": "AICL",   # Insurance
    "leather": "SGF",   # Leather & Tanneries
    "paper": "CEPB",   # Paper, Board & Packaging
    "textile_spinning": "KOSM",   # Textile Spinning
}

# lag grid in MILLISECONDS to scan for the HY lead-lag peak. Positive = leader
# leads follower.
#
# WIDENED 2026-09-15. The previous grid ran 0, 250, 500, 1000, 2000, 3000, 5000,
# 10000, 15000, 30000 -- only THREE sample points between 3 and 15 seconds. PSX
# is quoted by people, not by machines, so the propagation scale to expect here
# is SECONDS, and the old grid sampled that band more thinly than any other.
#
# That is not cosmetic. If the true lag is, say, 7 s, each day's peak lands on
# either 5000 or 10000 depending on noise and flips between them day to day.
# That MANUFACTURES sign instability and INFLATES peak_lag_se -- which are two of
# the gates the screen failed on. A coarse grid in the band of interest can
# depress the very statistics used to declare the result noise.
#
# Now 1-second resolution out to 15 s, where a human-speed lead would live, then
# coarser to 30 s for the tail. 39 points against 19, so expect roughly double
# the runtime (the 20-day pass took about 7 minutes).
LAG_GRID_MS = [-30000, -25000, -20000, -15000, -12000, -10000, -9000, -8000,
               -7000, -6000, -5000, -4000, -3000, -2500, -2000, -1500, -1000,
               -500, -250, 0,
               250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000,
               8000, 9000, 10000, 12000, 15000, 20000, 25000, 30000]
# round-trip fee (bps) -- the economic hurdle in Step 4
FEE_BPS = 1.554
# min trades per symbol-day to attempt HY (too few -> unstable)
MIN_TRADES = 100
# sampled days for a run
MAX_DAYS = 20
# workers
WORKERS = 6
# MINIMUM DAYS a pair must survive before it is reported at all. The whole screen
# is day-as-unit -- peak_lag_se is a standard error across days -- so a pair with
# fewer than this has no error bar worth printing. Was a bare 3 inline; named here
# because --smoke must respect it or it silently produces an empty result.
MIN_PAIR_DAYS = 3
# How many names per sector take a turn as the leader, ranked by MEDIAN DAILY
# TRADED VALUE (PKR), not share volume. 2 means the largest and second-largest
# are both tested, which is the only way to find out whether the size prior
# ("largest free-float leads") actually holds -- on PSX the largest name is often
# the most index-driven, which can make it the FOLLOWER of sector news that
# surfaces first in a mid-cap with concentrated informed flow.
# COST: this roughly doubles the number of pairs, so the multiple-testing burden
# doubles with it. Judge a candidate on the full acceptance rule printed at the
# end of a run, never on the best t-statistic in the table.
N_LEADERS = 2
# days used by --smoke. MUST be >= MIN_PAIR_DAYS or every pair is skipped and the
# run ends with an empty frame. The old value was 2, which could never produce a
# single row.
SMOKE_DAYS = 5

def _ts():
    # log stamp
    return datetime.now().strftime("[%H:%M:%S]")

# ----------------------------------------------------------------------------
# THE CORE: Hayashi-Yoshida async cross-covariance + lead-lag curve
# ----------------------------------------------------------------------------
def _hy_cov(xs, xe, rx, ys, ye, ry):
    # Hayashi-Yoshida covariance for two async return series.
    # X return i lives on interval (xs[i], xe[i]]; Y return j on (ys[j], ye[j]].
    # HY sums rx[i]*ry[j] over every pair of OVERLAPPING intervals. Overlap of
    # (a0,a1] and (b0,b1] <=> a0 < b1 AND b0 < a1. Inputs sorted by start time.
    # Two-pointer sweep: amortized ~O(nX + nY + #overlaps).
    nX = len(rx); nY = len(ry)
    # running covariance sum
    cov = 0.0
    # lower Y pointer: first Y interval that could still overlap the current X
    j0 = 0
    # walk X intervals in time order
    for i in range(nX):
        # this X interval's bounds
        a0 = xs[i]; a1 = xe[i]
        # advance j0 past Y intervals that END at/before X_i starts (cannot overlap)
        while j0 < nY and ye[j0] <= a0:
            j0 += 1
        # scan Y intervals that START before X_i ends (candidates for overlap)
        j = j0
        while j < nY and ys[j] < a1:
            # confirm overlap (Y_j ends after X_i starts); j>=j0 makes this usually true
            if ye[j] > a0:
                # accumulate the cross-product of the two overlapping returns
                cov += rx[i] * ry[j]
            # next Y interval
            j += 1
    # the HY covariance at this alignment
    return cov

def hy_leadlag_curve(tX, pX, tY, pY, lags_ms):
    # Build log-return intervals for each asset, then compute the HY correlation
    # for every candidate lag (Y shifted EARLIER by lag; positive lag => X leads Y).
    # tX/tY: trade times (ms, sorted). pX/pY: trade prices (>0).
    # returns (lags, corr[lag], rv_x, rv_y) with corr = HY_cov / sqrt(RV_X*RV_Y).
    # need at least 2 prices per side to form one return
    if len(pX) < 2 or len(pY) < 2:
        return None
    # log prices
    lpX = np.log(pX); lpY = np.log(pY)
    # X return intervals: (t[k-1], t[k]] with return dlogp
    xs = tX[:-1].astype(float); xe = tX[1:].astype(float); rx = np.diff(lpX)
    # Y return intervals (unshifted)
    ys0 = tY[:-1].astype(float); ye0 = tY[1:].astype(float); ry = np.diff(lpY)
    # drop zero-duration intervals (repeated timestamps) which break overlap logic
    okx = xe > xs; xs, xe, rx = xs[okx], xe[okx], rx[okx]
    oky = ye0 > ys0; ys0, ye0, ry = ys0[oky], ye0[oky], ry[oky]
    # realized variances (HY self-variance == sum of squared returns)
    rv_x = float(np.sum(rx * rx)); rv_y = float(np.sum(ry * ry))
    # guard degenerate (flat) series
    if rv_x <= 0 or rv_y <= 0:
        return None
    # denominator for the correlation normalization
    denom = np.sqrt(rv_x * rv_y)
    # compute HY correlation at each candidate lag
    corr = np.empty(len(lags_ms), dtype=float)
    for k, lag in enumerate(lags_ms):
        # shift Y intervals EARLIER by `lag` so that if X leads Y by `lag`, they align
        ys = ys0 - lag; ye = ye0 - lag
        # HY covariance at this alignment / normalization
        corr[k] = _hy_cov(xs, xe, rx, ys, ye, ry) / denom
    # return the curve + variances (variances feed the Step-4 beta)
    return np.asarray(lags_ms, float), corr, rv_x, rv_y

def peak_leadlag(lags, corr):
    # locate the lead-lag peak = lag maximizing |HY correlation|.
    # returns (peak_lag_ms, peak_corr, corr_at_zero).
    # index of max absolute correlation
    k = int(np.argmax(np.abs(corr)))
    # correlation exactly at lag 0 (the async-artifact reference)
    z = corr[np.argmin(np.abs(lags))]
    # peak lag, peak corr, and the zero-lag value
    return float(lags[k]), float(corr[k]), float(z)

# ----------------------------------------------------------------------------
# DATA LAYER (DuckDB) -- trades for returns, ob_snapshot cum_value for leader pick
# ----------------------------------------------------------------------------
def _con():
    # lazy duckdb import so --selftest runs without it
    import duckdb
    return duckdb.connect()

def _trades_glob():
    # parquet glob for the trades table
    import os
    return os.path.join(PARSED_ROOT, "trades", "**", "*.parquet")

def _snap_glob():
    # parquet glob for the ob_snapshot table
    import os
    return os.path.join(PARSED_ROOT, "ob_snapshot", "**", "*.parquet")

def discover_dates():
    # distinct trading dates from the trades table
    con = _con()
    q = f"SELECT DISTINCT CAST(date AS VARCHAR) d FROM read_parquet('{_trades_glob()}', hive_partitioning=1) ORDER BY d"
    return con.execute(q).df()["d"].tolist()

def load_trades(sym, date):
    # trade time (ms since epoch) + price for one symbol-day, time-ordered.
    con = _con()
    q = f"""
        SELECT epoch_ms(transact_time) AS t_ms, price
        FROM read_parquet('{_trades_glob()}', hive_partitioning=1)
        WHERE symbol = ? AND CAST(date AS VARCHAR) = ?
          AND price IS NOT NULL AND price > 0 AND transact_time IS NOT NULL
        ORDER BY transact_time
    """
    df = con.execute(q, [sym, date]).df()
    return df["t_ms"].to_numpy(), df["price"].to_numpy()

def daily_traded_value(sym, date):
    # exchange cumulative traded value for the day (max of cum_value) -- the
    # data-driven leader-selection metric (no market-cap needed).
    con = _con()
    q = f"""
        SELECT MAX(cum_value) AS v
        FROM read_parquet('{_snap_glob()}', hive_partitioning=1)
        WHERE symbol = ? AND CAST(date AS VARCHAR) = ?
    """
    r = con.execute(q, [sym, date]).df()
    return float(r["v"].iloc[0]) if len(r) and pd.notna(r["v"].iloc[0]) else 0.0

# ----------------------------------------------------------------------------
# STEPS
# ----------------------------------------------------------------------------
def pick_leaders(dates):
    # Step 0: the top N_LEADERS names per sector by MEDIAN DAILY TRADED VALUE.
    # Value, not share volume: daily_traded_value() reads MAX(cum_value), the
    # exchange's own cumulative traded VALUE in PKR. Share count would rank a
    # 1.28 PKR name above a 450 PKR one on identical economic activity.
    # Returns {sector: [leader1, leader2, ...]} ordered most-traded first.
    leaders = {}; tv_table = []
    # each sector
    for sec, names in SECTORS.items():
        # need at least one leader + one follower
        if len(names) < 2:
            continue
        # median daily traded value per name
        med = {}
        for sym in names:
            vals = [daily_traded_value(sym, d) for d in dates]
            vals = [v for v in vals if v > 0]
            med[sym] = float(np.median(vals)) if vals else 0.0
            tv_table.append({"sector": sec, "symbol": sym, "median_traded_value": med[sym]})
        # rank by traded value, descending, and keep the top N that actually traded
        ranked = [s for s in sorted(med, key=med.get, reverse=True) if med[s] > 0]
        # a sector with one tradeable name cannot form a pair
        if len(ranked) < 2:
            continue
        # take the top N, but never more leaders than leaves a follower behind
        leaders[sec] = ranked[:min(N_LEADERS, len(ranked) - 1)]
    return leaders, pd.DataFrame(tv_table)

def update_rate(sym, date):
    # Step 1 metric: median inter-trade time (ms) for one symbol-day.
    t, _ = load_trades(sym, date)
    if len(t) < 2:
        return np.nan, len(t)
    return float(np.median(np.diff(t))), len(t)

def hy_beta_bps(corr_peak, rv_leader, rv_follower, t_leader, p_leader):
    # Step 4 helper: an indicative anticipatable move in bps at the peak.
    # HY regression beta of follower on leader ~ corr * sqrt(RV_follower/RV_leader).
    beta = corr_peak * np.sqrt(rv_follower / rv_leader) if rv_leader > 0 else np.nan
    # median absolute per-trade leader return in bps (the size of a typical leader tick)
    if len(p_leader) >= 2:
        lr = np.abs(np.diff(np.log(p_leader))) * 1e4
        lead_move_bps = float(np.median(lr[lr > 0])) if np.any(lr > 0) else np.nan
    else:
        lead_move_bps = np.nan
    # indicative anticipatable follower move per leader event (bps)
    return float(abs(beta) * lead_move_bps) if np.isfinite(beta) and np.isfinite(lead_move_bps) else np.nan

# ----------------------------------------------------------------------------
# SELF-TEST: prove the HY estimator on synthetic ground truth (NO data needed)
# ----------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(7)
    # ---- build a latent efficient log-price on a fine 1ms grid (10 minutes) ----
    T = 600_000                     # ms
    dt = 1.0                        # 1 ms steps
    vol = 0.02 / np.sqrt(T)         # per-step vol
    latent = np.cumsum(rng.normal(0, vol, T))   # latent efficient log-price
    base = 100.0
    price_path = base * np.exp(latent)          # latent price level
    TAU = 800                       # TRUE lead: follower lags leader by 800 ms

    def async_sample(path, rate_ms, noise_bps, lag_ms=0):
        # sample `path` at async Poisson times (mean gap rate_ms), with the sampled
        # value taken from time (t - lag_ms) [so lag_ms>0 => this asset LAGS], plus
        # independent microstructure (bid-ask-bounce) noise.
        t = 0; times = []
        while t < T:
            t += int(rng.exponential(rate_ms))
            if t < T:
                times.append(t)
        times = np.array(times, dtype=float)
        idx = np.clip((times - lag_ms).astype(int), 0, T - 1)
        px = path[idx] * np.exp(rng.normal(0, noise_bps / 1e4, len(times)))
        return times, px

    # LEADER: fast (avg 200ms between trades), some noise, no lag
    tX, pX = async_sample(price_path, rate_ms=200, noise_bps=2.0, lag_ms=0)
    # FOLLOWER: slower (avg 500ms), more noise, lags the leader by TAU ms
    tY, pY = async_sample(price_path, rate_ms=500, noise_bps=4.0, lag_ms=TAU)

    print("=== HY lead-lag SELF-TEST (synthetic ground truth) ===")
    print(f"  true lag TAU = +{TAU} ms (leader leads follower)")
    print(f"  leader trades={len(tX)} (~200ms), follower trades={len(tY)} (~500ms)")

    # CASE 1: recover the known lag
    lags, corr, rvx, rvy = hy_leadlag_curve(tX, pX, tY, pY, LAG_GRID_MS)
    pk_lag, pk_corr, at0 = peak_leadlag(lags, corr)
    print(f"[case1] peak lag = {pk_lag:+.0f} ms  peak corr = {pk_corr:+.3f}  corr@0 = {at0:+.3f}")
    # the peak must be at a POSITIVE lag near TAU (grid resolution), not at 0
    nearest = LAG_GRID_MS[int(np.argmin([abs(g - TAU) for g in LAG_GRID_MS]))]
    assert pk_lag > 0, f"peak lag not positive: {pk_lag}"
    assert abs(pk_lag - nearest) <= 500, f"peak {pk_lag} not near TAU grid point {nearest}"
    assert pk_corr > 0.3, f"peak corr too weak: {pk_corr}"
    # the async artifact check: corr@0 must be BELOW the true peak (else Epps artifact)
    assert pk_corr > abs(at0), "peak not above zero-lag -> would be an Epps artifact"
    print("  [case1] PASS: recovered a positive lag near TAU, peak > corr@0.")

    # CASE 2: two INDEPENDENT latent walks -> no stable lead-lag (false-positive guard)
    latent2 = np.cumsum(rng.normal(0, vol, T)); path2 = base * np.exp(latent2)
    tA, pA = async_sample(price_path, 200, 2.0, 0)
    tB, pB = async_sample(path2, 500, 4.0, 0)   # unrelated asset
    _, corr2, _, _ = hy_leadlag_curve(tA, pA, tB, pB, LAG_GRID_MS)
    pk2 = float(np.max(np.abs(corr2)))
    print(f"[case2] independent assets: max |corr| = {pk2:.3f} (should be small)")
    assert pk2 < 0.15, f"false peak on independent assets: {pk2}"
    print("  [case2] PASS: no spurious lead-lag on unrelated assets.")
    print("SELF-TEST PASSED.")

    # ---- save the case-1 lead-lag curve as visual proof ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(lags, corr, marker="o", ms=4, label="cointegrated pair (true lag +800ms)")
    ax.plot(lags, corr2, marker="x", ms=4, ls="--", color="gray", label="independent pair")
    ax.axvline(TAU, color="green", ls=":", label=f"true lag = +{TAU}ms")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("candidate lag (ms; +=leader leads)"); ax.set_ylabel("HY correlation")
    ax.set_title("HY lead-lag curve: peak at the true lag, flat for independent assets")
    ax.legend(fontsize=8); fig.tight_layout()
    # write beside every other result, not to a hardcoded absolute path (the
    # original pointed at /home/claude, which exists only in the sandbox the
    # file was written in -- the self-test PASSED and then crashed on save)
    import os
    # the leadlag output folder, created if this is the first run
    _out_dir = os.path.join(RESULTS_ROOT, "leadlag")
    # make sure it exists
    os.makedirs(_out_dir, exist_ok=True)
    # the self-test figure's full path
    _png = os.path.join(_out_dir, "leadlag_selftest.png")
    # save it
    fig.savefig(_png, dpi=130)
    # say where it went
    print(f"saved {_png}")

# ----------------------------------------------------------------------------
# (real-data run + smoke omitted from the offline demo path; wired for your Mac)
# ----------------------------------------------------------------------------
def run_real(max_days=MAX_DAYS, workers=WORKERS):
    # full screen: pick leaders, then Steps 1-4 per (leader,follower)-day, day-as-unit.
    dates = discover_dates()
    if max_days and len(dates) > max_days:
        step = max(1, len(dates) // max_days); dates = dates[::step][:max_days]
    print(_ts() + f"{len(dates)} dates | sectors: {list(SECTORS)}")
    # Step 0
    leaders, tv = pick_leaders(dates)
    print(_ts() + "leaders (by median traded value):")
    print(tv.sort_values(['sector','median_traded_value'], ascending=[True,False]).to_string(index=False))
    # show the top N per sector in rank order, and how many pairs that implies
    print(_ts() + f"selected top-{N_LEADERS} leaders per sector (by traded value):")
    for _sec, _ls in sorted(leaders.items()):
        print(f"    {_sec:>18}: " + ", ".join(f"#{i+1} {s}" for i, s in enumerate(_ls)))
    # the unordered pair count, which is what the multiple-testing burden scales with
    _npairs = sum(len({frozenset((l, f)) for l in _ls
                       for f in SECTORS[_sec] if f != l})
                  for _sec, _ls in leaders.items())
    print(_ts() + f"{_npairs} unordered pairs to test "
                  f"(each pair runs once -- the HY curve is antisymmetric in lag)")
    rows = []
    # pairs already measured, as unordered {A,B} sets. The HY curve is
    # ANTISYMMETRIC in lag: testing A->B across the lag grid already contains the
    # B->A answer as its mirror image. Running both adds no information and
    # doubles the multiple-testing burden, so each unordered pair runs once.
    seen_pairs = set()
    # each sector's leaders vs every other name in the sector
    for sec, sec_leaders in leaders.items():
        # each of the top N traded-value names takes a turn as the leader
        for leader in sec_leaders:
            # every other name in the sector is a candidate follower -- INCLUDING
            # the sector's other leader, because "does the largest name actually
            # lead the second largest" is the direct test of the size prior and
            # is the single most informative pair in the sector
            followers = [s for s in SECTORS[sec] if s != leader]
            for foll in followers:
                # the unordered identity of this pair
                key = (sec, frozenset((leader, foll)))
                # already measured from the other direction
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                # this pair's per-day results
                per_day = []
                # every sampled date
                for d in dates:
                    # Step 1 activity
                    lm, ln = update_rate(leader, d); fm, fn = update_rate(foll, d)
                    # both legs must have enough trades for HY to be stable
                    if ln < MIN_TRADES or fn < MIN_TRADES:
                        continue
                    tX, pX = load_trades(leader, d); tY, pY = load_trades(foll, d)
                    res = hy_leadlag_curve(tX, pX, tY, pY, LAG_GRID_MS)
                    # no overlapping quotes -> nothing to measure this day
                    if res is None:
                        continue
                    lags, corr, rvx, rvy = res
                    pk_lag, pk_corr, at0 = peak_leadlag(lags, corr)
                    ind_bps = hy_beta_bps(pk_corr, rvx, rvy, tX, pX)
                    per_day.append(dict(date=d, leader_med_ms=lm, foll_med_ms=fm,
                                        peak_lag_ms=pk_lag, peak_corr=pk_corr,
                                        corr0=at0, ind_bps=ind_bps))
                # a pair needs enough days to carry a day-as-unit error bar
                if len(per_day) < MIN_PAIR_DAYS:
                    continue
                pdd = pd.DataFrame(per_day)
                # day-as-unit aggregation (Steps 1,3,4)
                rows.append(dict(
                    sector=sec, leader=leader, follower=foll,
                    # which traded-value rank this leader holds in its sector, so
                    # the output can be read as "did the #2 name beat the #1"
                    leader_rank=sec_leaders.index(leader) + 1,
                    n_days=len(pdd),
                    # Step 1: is the leader faster? (median inter-trade time)
                    leader_ms=pdd.leader_med_ms.median(),
                    foll_ms=pdd.foll_med_ms.median(),
                    leader_faster=bool(pdd.leader_med_ms.median()
                                       < pdd.foll_med_ms.median()),
                    # Step 2/3: peak lag mean +/- SE, sign stability
                    peak_lag_mean=pdd.peak_lag_ms.mean(),
                    peak_lag_se=pdd.peak_lag_ms.std(ddof=1)/np.sqrt(len(pdd)),
                    frac_leader_leads=float((pdd.peak_lag_ms > 0).mean()),
                    peak_corr_mean=pdd.peak_corr.mean(),
                    corr0_mean=pdd.corr0.mean(),
                    # Step 4: indicative anticipatable bps vs the fee hurdle
                    ind_bps_median=pdd.ind_bps.median(),
                    clears_fee=bool(pdd.ind_bps.median() > FEE_BPS)))
    out = pd.DataFrame(rows)
    import os; os.makedirs(os.path.join(RESULTS_ROOT, "leadlag"), exist_ok=True)
    p = os.path.join(RESULTS_ROOT, "leadlag", "leadlag_screen.parquet")
    print(_ts() + "\n=== LEAD-LAG SCREEN (day-as-unit) ===")
    # EMPTY-RESULT GUARD. An empty frame has no columns, so selecting the report
    # columns from it raises a bare KeyError that says nothing about the cause.
    # Every cause is a data-sufficiency problem, so name them instead of crashing.
    if len(out) == 0:
        print("  NO PAIRS SURVIVED -- nothing to report. In order of likelihood:")
        print(f"    1. days sampled ({max_days}) < MIN_PAIR_DAYS ({MIN_PAIR_DAYS}). "
              "A pair needs that many usable days before it gets an error bar.")
        print(f"    2. too few symbol-days cleared MIN_TRADES ({MIN_TRADES} trades).")
        print("    3. the HY curve returned None on most days (no overlapping quotes).")
        print("  Nothing was written. Re-run with more days.")
        return out
    # only write a frame that has content
    out.to_parquet(p, index=False)
    cols = ["sector","leader","leader_rank","follower","n_days","leader_faster",
            "peak_lag_mean","peak_lag_se","frac_leader_leads","peak_corr_mean",
            "corr0_mean","ind_bps_median","clears_fee"]
    # sort so each sector's #1 leader is read before its #2
    print(out.sort_values(["sector","leader_rank","follower"])[cols]
             .to_string(index=False))
    # the direct test of the size prior: does the #1 name lead the #2, or trail it?
    top2 = out[out.follower.isin([l for ls in leaders.values() for l in ls])
               & out.leader.isin([l for ls in leaders.values() for l in ls])]
    if len(top2):
        print("\n  #1 vs #2 IN EACH SECTOR -- the direct test of "
              "'the largest name leads':")
        print(top2.sort_values("sector")[
            ["sector","leader","leader_rank","follower","peak_lag_mean",
             "peak_lag_se","frac_leader_leads","peak_corr_mean"]
        ].to_string(index=False))
        # a positive peak lag means the named leader genuinely leads
        _lead_wins = int((top2[top2.leader_rank == 1].peak_lag_mean > 0).sum())
        _n1 = int((top2.leader_rank == 1).sum())
        print(f"    the #1 name leads its #2 in {_lead_wins} of {_n1} sectors")
    print(_ts() + f"wrote {p}")
    print(_ts() + "READ: a candidate is only worth an engine throttle test if ALL hold: "
                  "leader_faster=True, |peak_lag_mean| clears its SE, frac_leader_leads>~0.7 "
                  "(stable sign), peak_corr_mean>>corr0_mean (not an Epps artifact), and "
                  "ind_bps_median in range of the fee. Miss any -> drop it.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--days", type=int, default=MAX_DAYS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    if a.selftest or not (a.run or a.smoke):
        selftest()
    elif a.run:
        run_real(max_days=(a.days or None), workers=a.workers)
    elif a.smoke:
        # SMOKE_DAYS, not 2: the old value was below MIN_PAIR_DAYS, so every pair
        # was skipped and the run died printing an empty frame
        run_real(max_days=SMOKE_DAYS, workers=1)
