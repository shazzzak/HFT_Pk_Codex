# verify_snapshot_equivalence.py -- prove the pandas->native snapshot() refactor
# gives IDENTICAL fills to the old code. Compares NEW-code backtests against the
# fills persisted by persist_fills.py (OLD-code output) across days.
# Run from existing_mm_live/:  python verify_snapshot_equivalence.py

# Path handling for the raw store and the persisted-fills reference.
from pathlib import Path
# DataFrames for reading persisted fills and comparing.
import pandas as pd
# Numeric array comparison (allclose) for price/qty equality.
import numpy as np
# Driver module: loader, event builder, CFG, MICRO_PARAMS, STRAT, dates.
import run_legacy_mm as R
# Engine classes: the backtester, seeded latency, and the naive strategy.
from mm_backtest import Backtester, LatencyModel, NaiveSymmetricMM
# The micro strategy.
from micro_mm import MicrostructureMM

# Point the loader at the raw parsed store (in-process override).
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# Root of the OLD-code fills persisted earlier (the reference to match against).
FILLS_ROOT = Path("/Users/shazzak/Capital Stake - Results/fills")

# The two strategies whose fills were persisted.
STRATEGIES = ["naive", "micro"]
# The two pilot symbols.
SYMBOLS = ["PPL", "UBL"]
# Day-sampling stride: 1 = every day (~18 min), 7 = ~1 day/week (~3 min).
STRIDE = 7


# Build a strategy exactly as persist_fills.py did, so fills are comparable.
def build_strategy(name, session_ms):
    # Micro needs the session window plus the shared MICRO_PARAMS.
    if name == "micro":
        # Construct the microstructure strategy with identical params.
        return MicrostructureMM(session_ms=session_ms, **R.MICRO_PARAMS)
    # Naive uses the shared STRAT params and ignores session_ms.
    return NaiveSymmetricMM(**R.STRAT)


# Re-run the NEW-code backtest for one (strategy, symbol, day); return fills df.
def rerun_fills(strategy, date, sym, dsets):
    # Load the updates table for this symbol-day.
    u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
    # Load the snapshot table for this symbol-day.
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
    # Load the trades table for this symbol-day.
    t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
    # Nothing to run without a book and trades.
    if len(t) == 0 or len(s) == 0:
        # Signal "no data" to the caller.
        return None
    # Build the merged event stream (new code pre-parses snapshots here).
    events, snap_groups, t = R.build_events(u, s, t)
    # Restrict to continuous-session trades (exclude auctions) for the window.
    cont = t[t["initiator"] != "AUCTION"]
    # No continuous trades -> nothing to run.
    if len(cont) == 0:
        # Signal "no data".
        return None
    # Session start/end in exchange-ms.
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
    # cfg with the SAME seeded latency persist_fills used (reproducible fills).
    cfg = dict(R.CFG, session=(t0, t1), latency_model=LatencyModel(seed=R.LATENCY_SEED))
    # Build the backtester with the requested strategy.
    bt = Backtester(build_strategy(strategy, (t0, t1)), cfg)
    # Run it; we only need the fills DataFrame.
    fills, equity, stats = bt.run(events, snap_groups)
    # Hand back the fills.
    return fills


# Compare NEW fills against the persisted OLD fills for one partition.
def compare_one(strategy, sym, date, dsets):
    # Path to the persisted OLD-code fills for this partition.
    old_path = FILLS_ROOT / strategy / sym / f"date={date}.parquet"
    # No reference on disk for this day -> cannot compare.
    if not old_path.exists():
        # Report the missing reference.
        return "no_ref", None
    # Load the reference fills (only the columns we compare).
    old = pd.read_parquet(old_path, columns=["t", "side", "px", "qty", "reason"])
    # Re-run the new-code backtest for the same partition.
    new = rerun_fills(strategy, date, sym, dsets)
    # Both empty -> trivially identical.
    if (new is None or len(new) == 0) and len(old) == 0:
        # Match with zero fills.
        return "match", 0
    # New empty but old non-empty -> mismatch.
    if new is None or len(new) == 0:
        # Report the count gap.
        return "mismatch", f"new=0 old={len(old)}"
    # Different fill counts -> immediate mismatch.
    if len(new) != len(old):
        # Report both counts.
        return "mismatch", f"count new={len(new)} old={len(old)}"
    # Sort the NEW fills identically so row order cannot cause a false mismatch.
    ncmp = new[["t", "side", "px", "qty"]].sort_values(
        ["t", "side", "px", "qty"]).reset_index(drop=True)
    # Sort the OLD fills the same way.
    ocmp = old[["t", "side", "px", "qty"]].sort_values(
        ["t", "side", "px", "qty"]).reset_index(drop=True)
    # Compare prices numerically (exact to 1e-9).
    px_ok = np.allclose(ncmp["px"].values, ocmp["px"].values, atol=1e-9)
    # Compare quantities numerically.
    qty_ok = np.allclose(ncmp["qty"].values, ocmp["qty"].values, atol=1e-9)
    # Compare sides exactly (string equality across all rows).
    side_ok = (ncmp["side"].values == ocmp["side"].values).all()
    # All three must hold for an exact match.
    if px_ok and qty_ok and side_ok:
        # Report a match with the fill count.
        return "match", len(new)
    # Otherwise report which field diverged.
    return "mismatch", f"px_ok={px_ok} qty_ok={qty_ok} side_ok={side_ok}"


# Sweep all (strategy, symbol, sampled-day) and tally matches/mismatches.
def main():
    # Sample every STRIDE-th date across the full range.
    dates = R.discover_dates()[::STRIDE]
    # Announce the scope.
    print(f"checking {len(STRATEGIES)}x{len(SYMBOLS)} sets over {len(dates)} sampled "
          f"days (STRIDE={STRIDE}); progress every 10 days per set\n", flush=True)
    # Loop each strategy.
    for strategy in STRATEGIES:
        # Loop each symbol.
        for sym in SYMBOLS:
            # Match / mismatch / no-reference counters for this set.
            n_match = n_mismatch = n_noref = 0
            # Collect the first few mismatch details for reporting.
            mismatches = []
            # Walk the sampled dates with a 1-based index for progress.
            for di, date in enumerate(dates, 1):
                # Open the date's datasets once.
                dsets = R.open_datasets(date)
                # Skip a date with a missing partition.
                if dsets is None:
                    # Move to the next date.
                    continue
                # Compare this partition (new vs persisted old).
                status, detail = compare_one(strategy, sym, date, dsets)
                # Tally a match.
                if status == "match":
                    # Increment matches.
                    n_match += 1
                # Tally a missing reference.
                elif status == "no_ref":
                    # Increment no-ref.
                    n_noref += 1
                # Otherwise it is a mismatch.
                else:
                    # Increment mismatches.
                    n_mismatch += 1
                    # Keep the first 5 mismatch details.
                    if len(mismatches) < 5:
                        # Record the date and diagnostic.
                        mismatches.append((date, detail))
                # Print a progress line every 10 sampled days.
                if di % 10 == 0:
                    # Show running counts for this set.
                    print(f"  {strategy}/{sym}: {di}/{len(dates)} days  "
                          f"({n_match} match, {n_mismatch} mismatch so far)", flush=True)
            # Print the verdict for this (strategy, symbol) set.
            print(f"{strategy:6} {sym}: {n_match} match, {n_mismatch} MISMATCH, "
                  f"{n_noref} no-ref", flush=True)
            # Print any mismatch details collected.
            for d, det in mismatches:
                # One line per recorded mismatch.
                print(f"        MISMATCH {d}: {det}", flush=True)
    # Closing guidance.
    print("\nAll '0 MISMATCH' => native snapshot() is identical to old pandas version "
          "on the sampled days. Set STRIDE=1 to check every day.")


# Standard entry-point guard.
if __name__ == "__main__":
    # Run the verification.
    main()
