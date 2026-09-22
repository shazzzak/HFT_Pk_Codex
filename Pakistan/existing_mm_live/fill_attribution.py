"""
fill_attribution.py

Fee-aware fill-level markout attribution for the PSX market-making backtest.

Per fill:  net_bps = capture(+half-spread) + markout(adverse selection) - round-trip fee
Then attribute across policy layers: naive symmetric -> OBI-skewed -> toxicity-gated,
split by vol regime, halt-masked to CONTINUOUS_AUCTION and outside market-halt windows.

FILL MODEL (production): fills are the ENGINE'S real queue-gated fills, persisted by
persist_fills.py (existing_mm_live/) -- same Backtester, seeded latency, and fees as the
backtests, with book-state context and +5s forward mid ASOF-joined at persist time under
hard no-leak assertions. The former fill-on-every-trade proxy is retired; fill rates and
per-fill economics now reconcile to the backtest by construction. The `reason` column
(through/at_queue/at_optimistic/crossing_add) enables P&L split by fill rule.

"""
# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths

# numeric + frames
import numpy as np
import pandas as pd

# shared PSX state classifier (same module the live trader uses)
import halt_state as HS

# =====================================================================
# FEES -- PSX schedule (from your block), per side on traded value
# =====================================================================
FEE_COMMISSION_PCT = 0.0015        # broker commission (retail only)
FEE_SST_RATE       = 0.13          # sales tax on commission
FEE_PSX_LAGA_PCT   = 0.000035      # PSX trading fee
FEE_SECP_PCT       = 0.0000065     # SECP supervisory
FEE_IPF_PCT        = 0.0000062     # PSX regulatory (IPF)
FEE_CLEARING_PCT   = 0.00003       # NCCPL + CDC (intraday low end)
FEE_MM_REBATE_PCT  = 0.0           # MM rebate (enter NEGATIVE when known)
# retail all-in per side
FEE_TOTAL_RETAIL = (FEE_COMMISSION_PCT * (1.0 + FEE_SST_RATE)
                    + FEE_PSX_LAGA_PCT + FEE_SECP_PCT + FEE_IPF_PCT
                    + FEE_CLEARING_PCT + FEE_MM_REBATE_PCT)
# TREC own-account per side (no broker commission -- you are the broker)
FEE_TOTAL_TREC = (FEE_PSX_LAGA_PCT + FEE_SECP_PCT + FEE_IPF_PCT
                  + FEE_CLEARING_PCT + FEE_MM_REBATE_PCT)
# which schedule to use
USE_TREC_FEE = True
FEE_TOTAL_PCT = FEE_TOTAL_TREC if USE_TREC_FEE else FEE_TOTAL_RETAIL

# glob to the parsed misc partitions (edit if your path differs)
# Resolve this filesystem path through the canonical checkout/data configuration.
MISC_GLOB = str(_hft_paths.PARSED_ROOT / 'misc/date=*/*.parquet')

# derive market-wide halt dates from data instead of hardcoding them
def load_market_halt_dates(con):
    # query misc for any day whose raw FIX session-status message (35=h) carries market-phase 'H'
    sql = f"""
        SELECT DISTINCT CAST(date AS VARCHAR) AS date
        FROM read_parquet('{MISC_GLOB}')
        WHERE msg_type = 'h'
          AND regexp_extract(raw, '8538=([A-Z])', 1) = 'H'
    """
    # run the query and return the dates as a set for O(1) membership tests
    return set(con.execute(sql).df()["date"])

# module-level cache; populated once by init_halts(con) before any masking runs
MARKET_HALT_DATES = set()

# call this once at startup with a duckdb connection to fill MARKET_HALT_DATES from the data
def init_halts(con):
    # declare we are writing the module-level global, not a local
    global MARKET_HALT_DATES
    # load the halt dates straight from misc and store them
    MARKET_HALT_DATES = load_market_halt_dates(con)
    # surface what was loaded so a run logs its own halt set
    print(f"loaded {len(MARKET_HALT_DATES)} market-halt dates from misc: {sorted(MARKET_HALT_DATES)}")
    # hand the set back in case the caller wants it
    return MARKET_HALT_DATES

# =====================================================================
# FEE / PER-FILL P&L MATH  (pure, unit-tested)
# =====================================================================

def fee_bps_per_side():
    # per-side fee expressed in bps of traded value
    return FEE_TOTAL_PCT * 1e4

def fee_bps_roundtrip():
    # entry + exit legs
    return 2.0 * fee_bps_per_side()

def capture_bps(side, fill_price, mid0):
    # spread captured at entry: +half-spread when you rest inside the mid
    # side=+1 you BOUGHT (filled on your bid), side=-1 you SOLD (filled on your ask)
    return 1e4 * side * (mid0 - fill_price) / mid0

def markout_bps(side, mid0, mid_h):
    # adverse selection: signed mid move over the horizon (negative = ran over)
    return 1e4 * side * (mid_h - mid0) / mid0

def gross_bps(side, fill_price, mid0, mid_h):
    # capture + markout collapses to the signed markout from your FILL PRICE to mid_h
    return 1e4 * side * (mid_h - fill_price) / mid0

def net_bps(side, fill_price, mid0, mid_h):
    # gross minus the round-trip fee (exit assumed at mid -> conservative; real MM may recapture)
    return gross_bps(side, fill_price, mid0, mid_h) - fee_bps_roundtrip()


# =====================================================================
# HALT MASK
# =====================================================================

# stamp every row with its PSX trading state using the shared classifier
def add_state(df):
    # scrip caps: prefer exchange-published xe/xf if present, else compute from prev_close
    # (expects columns upper_cap / lower_cap if you joined xe/xf; otherwise derive here)
    if "upper_cap" not in df or "lower_cap" not in df:
        # derive the +/-10%-or-PKR-1 band per row from prev_close
        band = df["prev_close"].apply(HS.scrip_band)
        # unpack the (lower, upper) tuple into columns
        df["lower_cap"] = band.apply(lambda t: t[0])
        df["upper_cap"] = band.apply(lambda t: t[1])
    # classify each row (vectorized via apply over the needed fields)
    df["state"] = df.apply(lambda r: HS.classify(
        stock_phase=r.get("phase"),                 # per-stock ob_snapshot phase
        market_phase=r.get("market_phase"),         # misc 8538 letter ('T'/'H'/...)
        suspended_all_day=bool(r.get("suspended_all_day", False)),
        best_bid=r.get("best_bid"), best_ask=r.get("best_ask"),
        upper_cap=r.get("upper_cap"), lower_cap=r.get("lower_cap")), axis=1)
    # return the frame with the new 'state' column
    return df

# boolean mask: True where a passive maker may quote normally (excludes halts, suspensions, locks)
def tradeable_mask(df):
    # ensure the state column exists (idempotent)
    if "state" not in df:
        # build it first
        df = add_state(df)
    # delegate the tradeable decision to the shared classifier
    return df["state"].map(HS.is_tradeable)

# =====================================================================
# FILL GENERATION (simplified: maker is counterparty to every trade)
# =====================================================================

def fills_from_trades(trades, mid0, mid_h):
    # RETIRED: this is the old fill-on-every-trade PROXY (maker = opposite of
    # aggressor). No longer called -- main() now computes economics on REAL
    # engine fills inline, where `side` is OURS directly (no inversion). Kept
    # for reference only; do NOT reintroduce into the pipeline.
    raise NotImplementedError("fills_from_trades is retired; real fills use inline economics in main()")
    # trades: DataFrame with aggressor_side, price, symbol, date, phase, features, in_market_halt
    # mid0/mid_h: pre-fill mid and horizon mid, aligned to each trade (as-of joined upstream)
    f = trades.copy()
    # maker side = OPPOSITE of the aggressor. aggressor SELL -> maker BOUGHT (+1)
    agg = f["aggressor_side"].astype(str).str.upper().str[0]       # 'B' or 'S'
    f["side"] = np.where(agg == "S", 1.0, np.where(agg == "B", -1.0, np.nan))
    # fill price is the trade (resting) price
    f["fill_price"] = f["price"]
    # attach the two mids
    f["mid0"], f["mid_h"] = mid0, mid_h
    # drop rows we can't sign or price
    f = f.dropna(subset=["side", "fill_price", "mid0", "mid_h"])
    # per-fill economics
    f["capture"] = capture_bps(f["side"], f["fill_price"], f["mid0"])
    f["markout"] = markout_bps(f["side"], f["mid0"], f["mid_h"])
    f["net"] = net_bps(f["side"], f["fill_price"], f["mid0"], f["mid_h"])
    return f


# =====================================================================
# ATTRIBUTION LAYERS: naive -> OBI-skew -> toxicity-gate
# =====================================================================

def attribute(fills, obi_col="obi_1", tox_col="toxicity", tox_pct=0.90):
    # apply the halt mask first
    f = fills[tradeable_mask(fills)].copy()
    # drop fills with no valid economics (near-close fills have NaN mid_h -> NaN net):
    # they count in n_fills but contribute nothing, and mismatch std()/len() in the SE.
    f = f.dropna(subset=["net", "capture", "markout"])
    # toxicity gate threshold (pull quotes in the top decile of toxicity)
    tox_thr = f[tox_col].quantile(tox_pct) if tox_col in f else np.inf

    # layer masks (each says WHICH fills the policy would accept)
    layers = {}
    # naive symmetric: accept every fill
    layers["naive"] = pd.Series(True, index=f.index)
    # OBI-skew: only BUY when book bid-heavy (obi>0); only SELL when ask-heavy (obi<0)
    layers["obi_skew"] = ((f["side"] > 0) & (f[obi_col] > 0)) | ((f["side"] < 0) & (f[obi_col] < 0))
    # OBI-skew + toxicity gate: additionally refuse fills in toxic flow
    layers["obi_skew+tox_gate"] = layers["obi_skew"] & (f[tox_col] < tox_thr)

    # summarize each layer
    rows = []
    for name, mask in layers.items():
        g = f[mask]
        rows.append(dict(layer=name, n_fills=len(g),
                         mean_net_bps=g["net"].mean(),
                         mean_capture=g["capture"].mean(),
                         mean_markout=g["markout"].mean(),
                         net_bps_se=g["net"].std() / np.sqrt(max(len(g), 1))))
    return pd.DataFrame(rows)


def attribute_by_regime(fills, n_terciles=3):
    # split by vol tercile then attribute within each
    f = fills[tradeable_mask(fills)].copy()
    # tercile on realized vol
    f["vol_t"] = pd.qcut(f["realized_vol_bps"], n_terciles, labels=["low", "mid", "high"])
    out = []
    for lvl, g in f.groupby("vol_t", observed=True):
        a = attribute(g)
        a.insert(0, "vol_regime", lvl)
        out.append(a)
    return pd.concat(out, ignore_index=True)

# =====================================================================
# FILL TABLE BUILD: trades asof-joined to the event-level mid timeline
# =====================================================================

# glob roots for the parsed trades and ob_snapshot partitions
# Resolve this filesystem path through the canonical checkout/data configuration.
TRADES_ROOT = str(_hft_paths.PARSED_ROOT / 'trades')
# Resolve this filesystem path through the canonical checkout/data configuration.
OB_ROOT     = str(_hft_paths.PARSED_ROOT / 'ob_snapshot')
# markout horizon in milliseconds (must match the label horizon: markout_5000ms_bps)
HORIZON_MS = 5000

# root of the REAL persisted fills written by persist_fills.py (engine's
# queue-gated fills, feature context pre-joined) -- the trades proxy is retired
# Resolve this filesystem path through the canonical checkout/data configuration.
FILLS_ROOT = str(_hft_paths.RESULTS_ROOT / 'fills')

# read the per-(strategy, symbol, day) REAL fill table and re-attach the
# per-stock phase timeline + prev_close/suspended flags that add_state() needs
def build_fills_for_partition(con, strategy, sym, dt):
    # the real-fill parquet for this (strategy, symbol, day)
    path = f"{FILLS_ROOT}/{strategy}/{sym}/date={dt}.parquet"
    # one SQL: real fills + phase (backward ASOF at fill time) + day scalars.
    # mid0/spread/obi/toxicity/vol/mid_h were joined AT PERSIST TIME under
    # hard no-leak assertions -- only the state columns are attached here,
    # using the SAME ASOF pattern (and syntax) the old proxy used in this file.
    sql = f"""
    WITH r AS (
        SELECT t AS ts, side, px AS price, qty, reason,
               mid0, spread_bps, obi_1, toxicity, realized_vol_bps, mid_h,
               strategy, symbol, date
        FROM read_parquet('{path}')
    ),
    ph AS (
        SELECT epoch_ms(snapshot_time) AS ts, phase
        FROM (SELECT DISTINCT snapshot_time, phase
              FROM read_parquet('{OB_ROOT}/date={dt}/*.parquet')
              WHERE symbol = '{sym}')
    ),
    pc AS (
        SELECT MAX(prev_close) AS prev_close,
               MAX(CAST(suspended_all_day AS INT)) AS susp
        FROM read_parquet('{OB_ROOT}/date={dt}/*.parquet')
        WHERE symbol = '{sym}'
    )
    SELECT r.*, p.phase, pc.prev_close, pc.susp
    FROM r
    ASOF JOIN ph p ON p.ts <= r.ts
    CROSS JOIN pc
    """
    # run the read+join; guard the missing-day case (nothing persisted)
    try:
        df = con.execute(sql).df()
    except Exception:
        return None
    # a day with zero real fills yields nothing to attribute
    if df.empty:
        return None
    # reconstruct best bid/ask from mid+spread for the lock check (as before)
    df["best_bid"] = df["mid0"] * (1.0 - df["spread_bps"] / 2e4)
    # ask is mid plus half the spread
    df["best_ask"] = df["mid0"] * (1.0 + df["spread_bps"] / 2e4)
    # hand the day's REAL fills back with full state context
    return df

# where per-day fill parquets and final results are written
# Resolve this filesystem path through the canonical checkout/data configuration.
RESULTS_DIR = str(_hft_paths.RESULTS_ROOT / 'fill_attribution')


# assemble the whole run: connect, load halts, build+checkpoint fills, attribute, save
def main():
    # stdlib for paths
    import os
    # lazy duckdb import so importing this module elsewhere doesn't require duckdb
    import duckdb
    # (feature_store_wiring no longer needed: real-fill parquets are enumerated
    # directly from fills/{strategy}/{sym}/date=*.parquet in the build phase)
    # ensure the output directories exist (fills/ holds one parquet per symbol-day)
    fills_dir = os.path.join(RESULTS_DIR, "fills")
    # create both levels if missing
    os.makedirs(fills_dir, exist_ok=True)
    # open the single connection used for the whole run
    con = duckdb.connect()
    # populate MARKET_HALT_DATES from misc (raw FIX 8538='H') BEFORE any masking runs
    init_halts(con)

    # ---- build phase: one checkpointed parquet per (strategy, symbol, day) ----
    # enumerate the REAL fill parquets persisted by persist_fills.py:
    # fills/{strategy}/{sym}/date={dt}.parquet -- self-describing, no feature_store_wiring needed
    import glob as _glob
    # every persisted real-fill file across both strategies
    real_fill_files = sorted(_glob.glob(os.path.join(FILLS_ROOT, "*", "*", "date=*.parquet")))
    # OPTIONAL symbol restriction: set to None to attribute everything on disk,
    # or a list to restrict (e.g. ["PPL","UBL"] even if fills/ holds all 38).
    SYMBOLS = ["PPL", "UBL"]
    # keep only files whose symbol folder is in the restriction (if any)
    if SYMBOLS is not None:
        real_fill_files = [f for f in real_fill_files
                           if os.path.basename(os.path.dirname(f)) in SYMBOLS]
    # decompose each path into (strategy, symbol, date) for the loop
    parts = []
    # walk the file list once
    for f in real_fill_files:
        # date from the filename: "date=YYYY-MM-DD.parquet"
        dt = os.path.basename(f)[len("date="):-len(".parquet")]
        # symbol = parent folder; strategy = grandparent folder
        sym = os.path.basename(os.path.dirname(f))
        strategy = os.path.basename(os.path.dirname(os.path.dirname(f)))
        # record the unit of work
        parts.append((strategy, sym, dt))
    # loop every (strategy, symbol, date)
    for i, (strategy, sym, dt) in enumerate(parts, 1):
        # checkpoint path for this unit (strategy-tagged so naive/micro don't collide)
        out_path = os.path.join(fills_dir, f"{strategy}_{sym}_{dt}.parquet")
        # skip work already done (crash-resume + cheap re-runs)
        if os.path.exists(out_path):
            continue
        # read this day's REAL engine fills (context pre-joined at persist time)
        day = build_fills_for_partition(con, strategy, sym, dt)
        # skip days with no usable trades
        if day is None:
            continue

        # preserve the engine's side string for auditing (BUY/SELL)
        day["side_str"] = day["side"]
        # OVERWRITE side with the +/-1 numeric convention the P&L math AND
        # attribute()'s layer masks expect: BUY (our bid filled) = +1, SELL = -1.
        # This is OUR side straight from the engine -- no aggressor inversion.
        day["side"] = np.where(day["side_str"] == "BUY", 1.0, -1.0)
        # spread captured at entry -- named 'capture' to match attribute()
        day["capture"] = capture_bps(day["side"], day["price"], day["mid0"])
        # adverse-selection markout over the 5s horizon -- named 'markout'
        day["markout"] = markout_bps(day["side"], day["mid0"], day["mid_h"])
        # gross = signed move from OUR fill price to the forward mid
        day["gross"] = gross_bps(day["side"], day["price"], day["mid0"], day["mid_h"])
        # net = gross minus round-trip fee -- named 'net' to match attribute()
        day["net"] = net_bps(day["side"], day["price"], day["mid0"], day["mid_h"])
        # SIMPLIFICATION (carried over from the proxy, unchanged): market-wide
        # phase fixed to 'T'; market halts still knock out via per-stock
        # TEMPORARY_SUSPENSION in the phase column. Production = misc 8538 ASOF.
        day["market_phase"] = "T"
        # suspended flag as boolean for the classifier
        day["suspended_all_day"] = day["susp"].astype(bool)

        # stamp the PSX trading state (halts, locks, near-limit)
        day = add_state(day)
        # persist the finished day (the checkpoint)
        day.to_parquet(out_path)
        # light progress every 25 partitions
        if i % 25 == 0:
            print(f"  fills built {i}/{len(parts)} partitions")

    # ---- load phase: read every checkpointed day back via duckdb (no pyarrow dependency) ----
    # read ONLY the strategy-prefixed real-fill checkpoints (naive_*/micro_*),
    # not any stale proxy files that may linger in the dir; union_by_name guards
    # against minor column-order differences across partitions.
    fills = con.execute(
        f"SELECT * FROM read_parquet("
        f"['{fills_dir}/naive_*.parquet', '{fills_dir}/micro_*.parquet'], "
        f"union_by_name=True)"
    ).df()
    # report the size of the assembled fill table
    print(f"loaded {len(fills):,} fills across {fills['date'].nunique()} days "
          f"({fills['symbol'].nunique()} symbols)")

    # ---- attribution phase: overall and by vol regime, both printed AND saved ----
    # attribute PER STRATEGY -- naive and micro fills must never be pooled,
    # because the whole point is comparing the two on identical days
    overall_parts, regime_parts = [], []
    # loop each strategy present in the loaded fills
    for strat_name, g in fills.groupby("strategy"):
        # split each strategy further BY SYMBOL -- PPL and UBL are opposite
        # failure modes (micro over-quotes PPL, stands aside on UBL), so pooling
        # them averages a disaster with a stand-aside and hides both stories.
        for sym_name, gs in g.groupby("symbol"):
            # overall layered attribution for this (strategy, symbol)
            a = attribute(gs)
            # tag symbol then strategy on the front (strategy leftmost)
            a.insert(0, "symbol", sym_name)
            a.insert(0, "strategy", strat_name)
            # collect
            overall_parts.append(a)
            # per-vol-regime attribution for this (strategy, symbol)
            r = attribute_by_regime(gs)
            # tag it
            r.insert(0, "symbol", sym_name)
            r.insert(0, "strategy", strat_name)
            # collect
            regime_parts.append(r)
    # assemble the strategy-split tables
    overall = pd.concat(overall_parts, ignore_index=True)
    # regime table
    by_regime = pd.concat(regime_parts, ignore_index=True)

    # print both tables
    print("\n== overall ==");   print(overall.to_string(index=False))
    print("\n== by regime =="); print(by_regime.to_string(index=False))
    # save both as CSV next to the fills
    overall.to_csv(os.path.join(RESULTS_DIR, "attribution_overall.csv"), index=False)
    # regime table
    by_regime.to_csv(os.path.join(RESULTS_DIR, "attribution_by_regime.csv"), index=False)
    # tell the user where everything landed
    print(f"\nsaved: {fills_dir}/*.parquet, attribution_overall.csv, attribution_by_regime.csv")

# run main() only when this file is executed directly, not when imported
if __name__ == "__main__":
    # call the entry point
    main()