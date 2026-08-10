# sweep_micro.py -- find the right micro params on TWO verified single days
# (PPL + UBL, 2026-06-30) BEFORE committing to a 5-hour full run.
# Run as a FRESH process:  python sweep_micro.py
# Overrides params per-run in memory; does NOT edit run_legacy_mm.py or micro_mm.py.

# The frozen engine.
import mm_backtest
# Fee guard -- fail loudly if a stale/wrong module loaded.
assert mm_backtest.USE_TREC_FEE is True, "wrong/stale mm_backtest (USE_TREC_FEE not True)"
assert abs(mm_backtest.FEE_TOTAL_PCT - 7.77e-05) < 1e-6, f"fee={mm_backtest.FEE_TOTAL_PCT}"

# Import the pieces run_one uses, so we can rebuild one symbol-day here.
import run_legacy_mm as R
from micro_mm import MicrostructureMM
from mm_backtest import Backtester, LatencyModel

# The one date both names are verified on.
DATE = "2026-06-30"
# The two test names -- opposite failure modes (PPL over-trades, UBL stands aside).
SYMBOLS = ["PPL", "UBL"]

# The parameter grid. min_edge_pct is the primary lever (0 = quote at cost floor,
# which is why PPL over-trades). gamma scales the risk half-spread + inventory skew.
# Each tuple is (min_edge_pct, gamma). Baseline first for reference.
GRID = [
    (0.0000, 0.15),   # baseline (what the losing full run used)
    (0.0003, 0.15),   # +3 bps edge floor
    (0.0005, 0.15),   # +5 bps edge floor
    (0.0010, 0.15),   # +10 bps edge floor
    (0.0005, 0.30),   # +5 bps floor, higher risk aversion (more skew, wider)
]

# Open the datasets once for the date (shared across all runs).
dsets = R.open_datasets(DATE)
# Guard: the date must be present in the store.
assert dsets is not None, f"no datasets for {DATE}"

# Header.
print(f"micro parameter sweep on {DATE} (fee=TREC, seeded latency)\n")
print(f"{'symbol':>6} {'min_edge':>9} {'gamma':>6} {'fills':>7} {'net_pnl':>12} {'pos_close':>10} {'unviable':>9} {'clean':>6}")

# Loop every parameter combination.
for min_edge, gamma in GRID:
    # Loop both symbols under this parameter set.
    for sym in SYMBOLS:
        # --- rebuild one symbol-day exactly as run_one does, but with a
        # --- MicrostructureMM whose params we override here. ---
        # Read the three tables for this symbol.
        u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
        s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
        t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
        # Skip if not runnable.
        if len(t) == 0 or len(s) == 0:
            print(f"{sym:>6}  (no data)")
            continue
        # Build the event stream (same helper run_one uses).
        events, snap_groups, t = R.build_events(u, s, t)
        # Continuous-session window in exchange-ms.
        cont = t[t["initiator"] != "AUCTION"]
        if len(cont) == 0:
            print(f"{sym:>6}  (no continuous session)")
            continue
        t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
        # Config: same CFG, same seeded latency, per-symbol session window.
        cfg = dict(R.CFG, session=(t0, t1),
                   latency_model=LatencyModel(seed=R.LATENCY_SEED))
        # Build micro with the swept params; everything else at MICRO_PARAMS defaults
        # EXCEPT the two we are testing. fee_pct omitted -> inherits TREC fee.
        strat = MicrostructureMM(
            session_ms=(t0, t1),
            size=50, max_inv=500,
            gamma=gamma,
            min_edge_pct=min_edge,
            tick=0.01,
            require_viable=True,
        )
        # Run it.
        bt = Backtester(strat, cfg)
        fills, equity, stats = bt.run(events, snap_groups)
        eod = bt.eod or {}
        # Pull the strategy's own quote-gate stats.
        unviable = strat.stats.get("no_quote_unviable", 0)
        # One row per (params, symbol).
        print(f"{sym:>6} {min_edge:>9.4f} {gamma:>6.2f} {len(fills):>7} "
              f"{eod.get('equity_liquidated', float('nan')):>12.2f} "
              f"{str(eod.get('pos_at_close')):>10} {unviable:>9} "
              f"{str(eod.get('liquidation_clean')):>6}")
    # Blank line between parameter sets for readability.
    print()

# Reference: naive on the same day for comparison.
print("--- naive baseline (same day) for reference ---")
for sym in SYMBOLS:
    r = R.run_one(DATE, sym, dsets)   # run_one uses NaiveSymmetricMM iff USE_MICRO=False
    if r:
        print(f"{sym:>6}  naive: fills={r['n_fills']:>6}  net_pnl={r['net_pnl']:>12.2f}  pos_close={r['pos_at_close']}")
print("\nNOTE: this reference is naive ONLY IF run_legacy_mm.USE_MICRO is False.")
print("Set USE_MICRO=False before trusting the naive reference line above.")
