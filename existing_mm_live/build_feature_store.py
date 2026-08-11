# ============================================================================
# build_feature_store.py -- PSX Event-Driven Feature Extractor (merged v2)
# ============================================================================
# MERGE OF TWO AGENT VERSIONS. Adopted from agent-2: micro_dev_bps, Cont-Kukanov
# L1 OFI, EWMA signed flow, bps normalization. Fixed from agent-2: (1) ts_exch
# was never set (self.now never injected) -> observe() now stores it; (2) the
# DuckDB ASOF join direction was version-fragile and unguarded -> replaced with
# the tested pandas merge_asof(direction='forward') + explicit no-leak assert;
# (3) the label mid timeline came from snapshots only (stale between snapshots,
# biasing 1s labels) -> labels join against the event-level mid timeline from
# the same Book replay as the features. Kept from agent-1: toxicity, VPIN,
# spread-Z, time-since-trade, obi_5/obi_deep from Book, loader/path wiring.
# INVENTORY IS DELIBERATELY EXCLUDED from features (endogeneity: the model must
# learn market toxicity, not our historical policy). pos stays in A-S skew only.
# ============================================================================

# Filesystem paths.
from pathlib import Path
# Per-day wall-clock timing.
import time
# Numeric arrays.
import numpy as np
# DataFrames + the tested forward ASOF label join.
import pandas as pd

# Frozen engine (Book, Backtester, LatencyModel) -- reused, never reimplemented.
import mm_backtest
# Driver module: loader, build_events, REQ_* column specs, CFG, discover_dates.
import run_legacy_mm as R
# Concrete classes.
from mm_backtest import Backtester, LatencyModel

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# Raw parsed store (moved out of CloudStorage).
PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# Override the loader's root IN THIS PROCESS ONLY (run_legacy_mm.py untouched).
R.PARSED_ROOT = PARSED_ROOT
# Results OUTSIDE the git project (git never sees multi-GB parquet).
RESULTS_ROOT = Path("/Users/shazzak/Capital Stake - Results")
# Feature-store subtree: feature_store/{symbol}/date={date}.parquet
FS_ROOT = RESULTS_ROOT / "feature_store"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Pilot names (opposite micro failure modes: PPL over-quotes, UBL stands aside).
SYMBOLS = ["PPL", "UBL"]
SYMBOLS = ["KTML", "FFC", "SYS", "BAFL", "PSO", "HBL", "KOHC", "PKGS", "ITANZ", "AKBL"]
SYMBOLS = ['AKBL', 'ATRL', 'BAFL', 'BOP', 'DGKC', 'ENGROH', 'FFC', 'FNEL', 'HASCOL', 'HBL', 'HUBC', 'KEL', 'LUCK', 'MARI', 'MEBL', 'MLCF', 'NBP', 'NCPL', 'NML', 'NPL', 'NRL', 'OGDC', 'PACE', 'PAEL', 'PIAHCLA', 'PIBTL', 'PIOC', 'PPL', 'PSO', 'PTC', 'SAZEW', 'SEARL', 'SYS', 'THCCL', 'TOMCL', 'TPL', 'TRG', 'UBL']
# Markout label horizons in ms. 5s is PRIMARY (matches the flatten horizon).
LABEL_HORIZONS_MS = [1000, 5000, 30000]
# Trailing trade window for toxicity (matches micro_mm's flow deque).
FLOW_WINDOW = 50
# EWMA decay for the FAST signed-flow average (~10-trade half-life, agent-2).
ALPHA_FAST = 0.1
# EWMA decay for realized variance (matches micro_mm.vol_alpha for reconciliation).
VOL_ALPHA = 0.05
# Rolling touch-event window for the spread Z-score.
SPREAD_Z_WINDOW = 200
# VPIN bucket size as a fraction of running daily volume (TUNING KNOB, flagged).
VPIN_BUCKET_FRACTION = 1.0 / 50.0
# Seed for cfg parity with real runs.
LATENCY_SEED = 0


# ===========================================================================
# FeatureCollector -- passive observer strategy. Quotes nothing, records
# everything, at the engine's own no-look-ahead point. One instance per
# symbol-day == all state resets daily. INVENTORY EXCLUDED BY DESIGN.
# ===========================================================================
class FeatureCollector:
    # book: THE SAME Book the Backtester mutates. session_ms: (t0, t1).
    def __init__(self, book, session_ms):
        # Live reference to the engine's book (not a copy).
        self.book = book
        # Session bounds (exchange-ms).
        self.t0, self.t1 = session_ms
        # Collected rows.
        self.rows = []
        # Current exchange time -- SET IN observe() (agent-2's bug: never set).
        self.now = None
        # --- previous-touch state for OFI / stability (agent-2 features) ---
        # Previous best bid price.
        self.prev_bb = None
        # Previous best ask price.
        self.prev_ba = None
        # Previous bid-touch quantity.
        self.prev_bq = 0.0
        # Previous ask-touch quantity.
        self.prev_aq = 0.0
        # --- trade-driven state ---
        # Trailing signed-volume window (+buy/-sell) for toxicity.
        self.flow = []
        # EWMA of signed trade flow (fast decay, agent-2).
        self.ewma_trade_flow = 0.0
        # EWMA of squared mid returns (realized variance, micro_mm-matched).
        self.ema_var = 0.0
        # Realized vol = sqrt(ema_var).
        self.sigma = 0.0
        # Last DIFFERENT mid.
        self.last_mid = None
        # Exchange-ms of the most recent trade (quiet clock).
        self.last_trade_ms = None
        # Rolling spread history (ticks) for the Z-score.
        self.spread_hist = []
        # --- VPIN state ---
        # Signed volume in the open bucket.
        self.vpin_bucket_signed = 0.0
        # Absolute volume in the open bucket.
        self.vpin_bucket_abs = 0.0
        # Closed-bucket imbalance ratios (last 50).
        self.vpin_buckets = []
        # Bucket size (lazy).
        self.vpin_bucket_vol = None
        # Running absolute volume (for lazy sizing).
        self.cum_abs_vol = 0.0

    # Called once per event by Backtester.run() -- same contract as micro_mm.
    def observe(self, kind, obj, ts_exch, mid):
        # FIX (agent-2 bug 1): store event time so rows can carry ts_exch.
        self.now = ts_exch
        # --- realized-vol EMA on genuine mid moves ---
        if mid is not None:
            if self.last_mid is not None and mid != self.last_mid and self.last_mid > 0:
                # Simple return since last different mid.
                ret = (mid - self.last_mid) / self.last_mid
                # Seed on first move; RiskMetrics EMA thereafter (micro_mm-identical).
                self.ema_var = ret * ret if self.ema_var == 0.0 \
                    else VOL_ALPHA * ret * ret + (1.0 - VOL_ALPHA) * self.ema_var
                # Vol = root variance.
                self.sigma = float(np.sqrt(self.ema_var))
            # Track for next move.
            self.last_mid = mid
        # --- trade-driven features ---
        if kind == "T":
            # TRUE aggressor side (PSX provides it -- no Lee-Ready).
            side = getattr(obj, "aggressor_side", None)
            # Only directional prints update signed features (AUCTION/None excluded).
            if side in ("BUY", "SELL"):
                # Sign and size.
                sgn = 1.0 if side == "BUY" else -1.0
                q = float(obj.qty)
                # Signed volume.
                sv = sgn * q
                # Windowed flow for toxicity.
                self.flow.append(sv)
                if len(self.flow) > FLOW_WINDOW:
                    self.flow.pop(0)
                # EWMA signed flow (agent-2 feature).
                self.ewma_trade_flow = ALPHA_FAST * sv + (1.0 - ALPHA_FAST) * self.ewma_trade_flow
                # Quiet clock.
                self.last_trade_ms = ts_exch
                # --- VPIN bucketing ---
                # Running volume for lazy bucket sizing.
                self.cum_abs_vol += q
                # Set bucket size once a volume scale exists.
                if self.vpin_bucket_vol is None and self.cum_abs_vol > 0:
                    self.vpin_bucket_vol = max(1.0, self.cum_abs_vol * VPIN_BUCKET_FRACTION)
                # Fill the open bucket.
                self.vpin_bucket_signed += sv
                self.vpin_bucket_abs += q
                # Close at target volume.
                if self.vpin_bucket_vol is not None and self.vpin_bucket_abs >= self.vpin_bucket_vol:
                    self.vpin_buckets.append(abs(self.vpin_bucket_signed) / self.vpin_bucket_abs)
                    if len(self.vpin_buckets) > 50:
                        self.vpin_buckets.pop(0)
                    self.vpin_bucket_signed = 0.0
                    self.vpin_bucket_abs = 0.0
        # --- record one feature row per TWO-SIDED book state ---
        # Read the touch from the SAME book the engine just updated.
        bb, bq, ba, aq = self.book.bbo()
        # Skip one-sided/empty books (confirmed: no quoting without both sides).
        if bb is None or ba is None or bq <= 0 or aq <= 0:
            return
        # Arithmetic mid -- the LABEL reference (microprice stays a feature).
        mid_arith = (bb + ba) / 2.0
        # Spread in bps of mid (agent-2 normalization).
        spread_bps = (ba - bb) / mid_arith * 1e4
        # Spread in ticks (flat 0.01 tick) for the Z-score history.
        spread_ticks = (ba - bb) / 0.01
        # Maintain rolling spread history.
        self.spread_hist.append(spread_ticks)
        if len(self.spread_hist) > SPREAD_Z_WINDOW:
            self.spread_hist.pop(0)
        # L1 depth-imbalance weight (bid share of touch depth).
        imb = bq / (bq + aq)
        # Microprice: depth-weighted touch price (heavier bid pulls fair UP).
        microprice = ba * imb + bb * (1.0 - imb)
        # Microprice deviation from mid, in bps (agent-2: dimensionless, cross-name).
        micro_dev_bps = (microprice - mid_arith) / mid_arith * 1e4
        # Raw L1 order-book imbalance in [-1, 1].
        obi_1 = (bq - aq) / (bq + aq)
        # Top-5 OBI from the Book (disclosed-depth signal).
        obi_5 = self.book.obi(5)
        # All-visible-levels OBI (noisier deep signal).
        obi_deep = self.book.obi(None)
        # --- Cont-Kukanov L1 OFI (agent-2, adopted) ---
        # Bid-price stability flag.
        bid_stable = 1 if bb == self.prev_bb else 0
        # Ask-price stability flag.
        ask_stable = 1 if ba == self.prev_ba else 0

        # Cont-Kukanov L1 OFI, guarded. Only defined once we have a previous
        # touch (first event -> 0, not a spurious full-queue spike). On a price
        # MOVE the level relevels rather than "flows", so we cap the contribution
        # to the touch quantity itself, preventing the 100k+ artifacts seen when
        # a whole queue is (dis)counted as flow. Same price -> true qty delta.
        if self.prev_bb is None or self.prev_ba is None:
            # No previous touch to difference against -> no flow yet.
            ofi_l1 = 0.0
        else:
            # Bid side: same price -> qty change; improve -> +new depth; worsen -> -old depth.
            if bb == self.prev_bb:
                dq_bid = bq - self.prev_bq
            elif bb > self.prev_bb:
                dq_bid = bq
            else:
                dq_bid = -self.prev_bq
            # Ask side: same price -> qty change; improve(lower) -> +new depth; worsen -> -old depth.
            if ba == self.prev_ba:
                dq_ask = aq - self.prev_aq
            elif ba < self.prev_ba:
                dq_ask = aq
            else:
                dq_ask = -self.prev_aq
            # Net L1 OFI.
            ofi_l1 = dq_bid - dq_ask

        # Queue-depletion rate on the bid (fast consumption detector, agent-2).
        qdr_bid = ((self.prev_bq - bq) / self.prev_bq) if (bid_stable and self.prev_bq > 0 and bq < self.prev_bq) else 0.0
        # Mirror QDR on the ask.
        qdr_ask = ((self.prev_aq - aq) / self.prev_aq) if (ask_stable and self.prev_aq > 0 and aq < self.prev_aq) else 0.0
        # --- windowed toxicity (micro_mm's own quantity, kept for comparability) ---
        signed = sum(self.flow)
        gross = sum(abs(f) for f in self.flow)
        toxicity = (abs(signed) / gross) if gross > 0 else 0.0
        # Spread Z-score (needs >= 20 samples; NaN before that, dropped in the fit).
        if len(self.spread_hist) >= 20:
            mu = float(np.mean(self.spread_hist)); sd = float(np.std(self.spread_hist))
            spread_z = (spread_ticks - mu) / sd if sd > 0 else 0.0
        else:
            spread_z = np.nan
        # VPIN (NaN until the first bucket closes).
        vpin = float(np.mean(self.vpin_buckets)) if self.vpin_buckets else np.nan
        # Time since last trade in ms (quiet indicator).
        tsl = (ts_exch - self.last_trade_ms) if self.last_trade_ms is not None else np.nan
        # Emit the row. NOTE: no position / inventory column -- excluded by design.
        self.rows.append({
            "ts_exch": int(ts_exch),
            "mid": mid_arith,
            "spread_bps": spread_bps,
            "obi_1": obi_1,
            "obi_5": obi_5,
            "obi_deep": obi_deep,
            "micro_dev_bps": micro_dev_bps,
            "ofi_l1": ofi_l1,
            "qdr_bid": qdr_bid,
            "qdr_ask": qdr_ask,
            "ewma_trade_flow": self.ewma_trade_flow,
            "toxicity": toxicity,
            "signed_volume": signed,
            "spread_z": spread_z,
            "realized_vol_bps": self.sigma * 1e4,
            "vpin": vpin,
            "time_since_trade_ms": tsl,
        })
        # Update previous-touch state for the next OFI computation.
        self.prev_bb, self.prev_bq = bb, bq
        self.prev_ba, self.prev_aq = ba, aq

    # Backtester calls quotes(); this collector never quotes.
    def quotes(self, bb, bq, ba, aq, pos):
        # Empty dict = no desired quotes.
        return {}


# ===========================================================================
# FeatureBacktester -- installs the collector wired to ITS OWN book.
# ===========================================================================
class FeatureBacktester(Backtester):
    # cfg carries session + latency exactly like a real run.
    def __init__(self, cfg):
        # Build the base with a placeholder object (never used before replacement).
        super().__init__(object(), cfg)
        # Install the real collector holding THIS instance's live book reference.
        self.strat = FeatureCollector(self.book, cfg["session"])


# ---------------------------------------------------------------------------
# LABELS -- tested pandas forward-ASOF (verified in synthetic tests; explicit
# no-leak assertion). The mid timeline is the feature rows' own event-level
# (ts_exch, mid) -- NOT snapshots-only (agent-2 bug 3: stale between snapshots).
# ---------------------------------------------------------------------------
def attach_labels(df):
    # ASOF requires sorted keys.
    df = df.sort_values("ts_exch").reset_index(drop=True)
    # Event-level mid timeline from the same Book replay as the features.
    mids = df[["ts_exch", "mid"]].rename(columns={"mid": "mid_future"})
    # One forward join per horizon.
    for h in LABEL_HORIZONS_MS:
        # Target label time for each row.
        target = df[["ts_exch"]].copy()
        target["ts_target"] = df["ts_exch"] + h
        # First mid at time >= t+h (forward direction -- no look-back possible).
        joined = pd.merge_asof(
            target.sort_values("ts_target"),
            mids.sort_values("ts_exch"),
            left_on="ts_target", right_on="ts_exch",
            direction="forward", suffixes=("", "_m"))
        # Restore original order.
        joined = joined.sort_values("ts_exch").reset_index(drop=True)
        # HARD GUARD: every matched mid must be dated >= t+h, else abort.
        ok = (joined["ts_exch_m"].fillna(joined["ts_target"]) >= joined["ts_target"]).all()
        assert ok, f"LOOK-AHEAD LEAK at horizon {h}ms: a label mid predates t+h"
        # Signed forward markout in bps of current mid. NaN where no future mid.
        df[f"markout_{h}ms_bps"] = (joined["mid_future"].values - df["mid"].values) / df["mid"].values * 1e4
    return df


# ---------------------------------------------------------------------------
# ONE SYMBOL-DAY
# ---------------------------------------------------------------------------
def build_one(date, sym, dsets):
    # Load the three tables with the SAME loader the backtester uses.
    u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
    t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
    # Not runnable without a book and trades.
    if len(t) == 0 or len(s) == 0:
        return None
    # Merged event stream (identical contract to the backtester).
    events, snap_groups, t = R.build_events(u, s, t)
    # Continuous session window (auctions excluded).
    cont = t[t["initiator"] != "AUCTION"]
    if len(cont) == 0:
        return None
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
    # cfg parity with real runs.
    cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=LATENCY_SEED))
    # Replay through the engine with the collector installed.
    fb = FeatureBacktester(cfg)
    fb.run(events, snap_groups)
    # Nothing two-sided happened.
    if not fb.strat.rows:
        return None
    # Assemble, label, tag.
    df = pd.DataFrame(fb.strat.rows)
    df = attach_labels(df)
    df["symbol"] = sym
    df["date"] = date
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # All trading dates in the store.
    dates = R.discover_dates()
    # Announce scope and destination.
    print(f"{len(SYMBOLS)} symbols x {len(dates)} dates -> {FS_ROOT}")
    # Walk every date.
    for di, date in enumerate(dates, 1):
        # Open the date's datasets once.
        dsets = R.open_datasets(date)
        # Missing partition -> skip.
        if dsets is None:
            print(f"  [{di}/{len(dates)}] {date} no datasets; skip"); continue
        # Time the date.
        dt0 = time.perf_counter()
        # Each pilot symbol.
        for sym in SYMBOLS:
            # Isolate failures per symbol-day.
            try:
                df = build_one(date, sym, dsets)
            except Exception as e:
                print(f"    {sym} {date} ERROR {e!r}"); continue
            # Skip empty.
            if df is None or len(df) == 0:
                continue
            # feature_store/{symbol}/date={date}.parquet
            out_dir = FS_ROOT / sym
            out_dir.mkdir(parents=True, exist_ok=True)
            # Skip already-built symbol-days so a crashed run resumes instead of restarting.
            out_path = out_dir / f"date={date}.parquet"
            if out_path.exists():
                continue
            df.to_parquet(out_dir / f"date={date}.parquet", index=False)
        # Per-date progress with timing.
        print(f"  [{di}/{len(dates)}] {date} {time.perf_counter()-dt0:.2f}s")
    # Done.
    print("done.")


# Entry point.
if __name__ == "__main__":
    main()
