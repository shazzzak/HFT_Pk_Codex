# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# ============================================================================
# throttle_param_sweep.py
# ----------------------------------------------------------------------------
# Stage-1 throttle sweep: throttle_frac x throttle_hold_ms, holding the two
# thresholds at the incumbent, stacked on the winning obi_defensive config
# (framing #2). Emits one row per (config, name, day) with net_pnl (PKR) and
# every run_one field. Checkpointed (JSONL journal, fsync, skip-on-resume),
# per-name axis retained, heartbeat+timer from v1.
#
# It REUSES run_legacy_mm.run_one and the frozen engine -- never reimplements
# the book/fill logic (redundant parallel implementations are a correctness
# risk, not a convenience).
#
# WIRING CHECK (run this BEFORE trusting any sweep number):
#   The "OFF" cell here must reproduce your existing throttle-OFF baseline
#   day-for-day. If it does not, the base config below does not match your
#   real ON/OFF definition -- fix BASE_PARAMS before reading any result.
#
# COMPUTE NOTE: this runs the SAME per-symbol-day work as a Run-A cell, so a
# 12-config Stage-1 grid ~= 12 * 38 * 197 ~= 89,832 cells. Launch it only when
# Run A / Run B have freed the cores. Detach it (Terminal.app, nohup ... &
# disown, verify PPID=1) -- children of PyCharm's terminal die with PyCharm.
#
# USAGE:
#   python throttle_param_sweep.py --workers 3
#   python throttle_param_sweep.py --workers 3 --resume   # continue a journal
# ============================================================================

# CLI.
import argparse
# JSON journal lines.
import json
# Filesystem.
import os
from pathlib import Path
# Timing / heartbeat.
import time
# Parallelism across (config, date) cells.
from multiprocessing import Pool
# DataFrames.
import pandas as pd

# The driver module: loader, run_one, MICRO_PARAMS, discover_dates, open_datasets.
import run_legacy_mm as R

# ---------------------------------------------------------------------------
# OUTPUT PATHS
# ---------------------------------------------------------------------------
# Results live OUTSIDE the git project.
# Resolve this filesystem path through the canonical checkout/data configuration.
OUT_DIR = Path(str(_hft_paths.RESULTS_ROOT / 'throttle_param_sweep'))
# JSONL checkpoint journal (one line per completed (config,date,name) cell).
JOURNAL = OUT_DIR / "throttle_param_sweep_CKPT.jsonl"
# Final labeled per-(config,name,day) CSV the analysis consumes.
FINAL_CSV = OUT_DIR / "throttle_param_sweep_PERNAME.csv"

# ---------------------------------------------------------------------------
# BASE CONFIG  <-- CONFIRM THIS MATCHES YOUR EXISTING throttle_sweep.py ON/OFF
# ---------------------------------------------------------------------------
# The "winning config" the throttle stacks on (framing #2). I have assumed it is
# obi_defensive=True at documented defaults. If your incumbent ON cell also had
# ofi_throttle on, or different obi_defensive thresholds, set them HERE so the
# sweep is comparable to the +0.852 bps result. Guessing this wrong makes every
# number below incomparable to your headline -- so verify against your real file.
BASE_PARAMS = dict(
    R.MICRO_PARAMS,          # start from the documented defaults (size=50, etc.)
    obi_defensive=True,      # the winning defensive base (framing #2)
    obi_defensive_thresh=0.15,   # default; confirm vs your winning config
    obi_defensive_ticks=1.0,     # default; confirm vs your winning config
)

# Whether the throttle-ON cells also enable the OFI throttle. The diagnostic
# found OBI the stronger signal; default to OBI-only for a clean dose sweep.
# Set True ONLY if your incumbent +0.852 cell had ofi_throttle on.
INCLUDE_OFI_THROTTLE = False

# Thresholds held FIXED at the incumbent during the frac x hold dose sweep.
OBI_THROTTLE_THRESH = 0.15
OFI_THROTTLE_THRESH = 0.30

# ---------------------------------------------------------------------------
# STAGE-1 GRID  <-- edit these two lists to change scope; no code changes needed
# ---------------------------------------------------------------------------
# Clip fraction while throttled (dose magnitude). Incumbent 0.5 is included.
GRID_FRAC = [0.25, 0.5, 0.75]
# Hold-time in ms (dose duration). Incumbent 300 is included; 0 = trigger-cycle only.
GRID_HOLD = [150.0, 300.0, 600.0, 1000.0]


def _cfg_label(frac, hold):
    # Canonical label -- MUST match throttle_sweep_analysis._cfg_label.
    return f"F{frac:g}_H{hold:g}"


def build_configs():
    # The list of (label, micro_params_dict) cells for the sweep.
    cells = []
    # OFF cell: defensive base, throttle disabled (the baseline).
    off = dict(BASE_PARAMS, obi_throttle=False, ofi_throttle=False)
    cells.append((R.__dict__.get("OFF_LABEL", "OFF"), off))
    # Correct the OFF label to the analysis constant explicitly.
    cells[-1] = ("OFF", off)
    # Throttle-ON dose grid.
    for fr in GRID_FRAC:
        for ho in GRID_HOLD:
            # ON cell: defensive base + throttle enabled + this (frac, hold).
            on = dict(
                BASE_PARAMS,
                obi_throttle=True,
                ofi_throttle=INCLUDE_OFI_THROTTLE,
                throttle_frac=fr,
                obi_throttle_thresh=OBI_THROTTLE_THRESH,
                ofi_throttle_thresh=OFI_THROTTLE_THRESH,
                throttle_hold_ms=ho,
            )
            cells.append((_cfg_label(fr, ho), on))
    return cells


# ---------------------------------------------------------------------------
# CHECKPOINT JOURNAL
# ---------------------------------------------------------------------------
def load_done():
    # Set of already-completed (config, date, symbol) keys for skip-on-resume.
    done = set()
    # Nothing to load on a fresh run.
    if not JOURNAL.exists():
        return done
    # Each line is one completed cell's result dict.
    with open(JOURNAL) as fh:
        for line in fh:
            # Tolerate a torn final line from a kill -9.
            try:
                r = json.loads(line)
            except Exception:
                continue
            # Key the cell.
            done.add((r["config"], r["date"], r["symbol"]))
    return done


def append_journal(fh, rec):
    # Serialize one cell result as a JSON line.
    fh.write(json.dumps(rec) + "\n")
    # Flush Python buffers.
    fh.flush()
    # Force the OS to write to disk so a crash cannot lose the cell.
    os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# ONE (config, date) TASK -- runs all symbols for that date under that config
# ---------------------------------------------------------------------------
def run_cell(task):
    # Unpack.
    label, params, date, symbols = task
    # Install this config as the strategy params for THIS worker process.
    R.MICRO_PARAMS = params
    # Ensure the micro strategy (not naive) is used.
    R.USE_MICRO = True
    # Open the date's datasets once.
    dsets = R.open_datasets(date)
    # Missing partition -> empty result.
    if dsets is None:
        return []
    # Accumulate per-symbol rows.
    out = []
    # Each requested symbol.
    for sym in symbols:
        # Isolate per-symbol failures.
        try:
            res = R.run_one(date, sym, dsets)
        except Exception as e:
            # Record the failure without killing the whole date.
            out.append(dict(config=label, date=date, symbol=sym, error=repr(e)))
            continue
        # Not runnable (empty book / no continuous phase).
        if res is None:
            continue
        # Tag with the config label + its params so the CSV is self-describing.
        res["config"] = label
        # Store the two swept dims explicitly for the surface plot.
        res["frac"] = params.get("throttle_frac") if label != "OFF" else None
        res["hold"] = params.get("throttle_hold_ms") if label != "OFF" else None
        # NOTE: run_one does NOT return traded notional, so bps cannot be formed
        # here. net_pnl (PKR) is exact. See the header COMPUTE/BPS note.
        out.append(res)
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # CLI.
    ap = argparse.ArgumentParser(description="Stage-1 throttle frac x hold sweep.")
    ap.add_argument("--workers", type=int, default=3, help="parallel processes")
    ap.add_argument("--resume", action="store_true", help="continue an existing journal")
    ap.add_argument("--symbols", nargs="*", default=None, help="optional symbol subset")
    args = ap.parse_args()
    # Ensure output dir.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Symbol universe: the feature-store 38 by default (single source in run_legacy).
    symbols = args.symbols or R.discover_symbols() if hasattr(R, "discover_symbols") else args.symbols
    # Fall back to the documented 38-name list if the driver exposes none.
    if not symbols:
        symbols = ['AKBL', 'ATRL', 'BAFL', 'BOP', 'DGKC', 'ENGROH', 'FFC', 'FNEL', 'HASCOL',
                   'HBL', 'HUBC', 'KEL', 'LUCK', 'MARI', 'MEBL', 'MLCF', 'NBP', 'NCPL', 'NML',
                   'NPL', 'NRL', 'OGDC', 'PACE', 'PAEL', 'PIAHCLA', 'PIBTL', 'PIOC', 'PPL',
                   'PSO', 'PTC', 'SAZEW', 'SEARL', 'SYS', 'THCCL', 'TOMCL', 'TPL', 'TRG', 'UBL']
    # All trading dates.
    dates = R.discover_dates()
    # The config cells.
    configs = build_configs()
    # Already-done cells (empty unless --resume).
    done = load_done() if args.resume else set()
    # Build the task list = (config, date) cells with their still-needed symbols.
    tasks = []
    # Walk configs x dates.
    for label, params in configs:
        for date in dates:
            # Only the symbols not already journaled for this (config,date).
            need = [s for s in symbols if (label, date, s) not in done]
            # Skip fully-done cells.
            if need:
                tasks.append((label, params, date, need))
    # Announce scope.
    total_cells = len(configs) * len(dates) * len(symbols)
    print(f"[sweep] {len(configs)} configs x {len(dates)} dates x {len(symbols)} names "
          f"= {total_cells:,} cells; {len(done):,} already done; {len(tasks):,} tasks queued",
          flush=True)
    # Open the journal in append mode (preserves prior cells on resume).
    jfh = open(JOURNAL, "a")
    # Timer.
    t0 = time.perf_counter()
    # Running cell counter for the heartbeat.
    n_cells = 0
    # Parallel pool over (config, date) tasks.
    with Pool(processes=args.workers) as pool:
        # Stream results as tasks complete.
        for rows in pool.imap_unordered(run_cell, tasks):
            # Journal each cell result (per-name row).
            for rec in rows:
                append_journal(jfh, rec)
                n_cells += 1
                # Heartbeat every 500 cells with elapsed + rate + rough ETA.
                if n_cells % 500 == 0:
                    el = time.perf_counter() - t0
                    rate = n_cells / el if el > 0 else 0.0
                    remaining = max(0, (total_cells - len(done) - n_cells))
                    eta = remaining / rate / 3600 if rate > 0 else float("nan")
                    print(f"  [{n_cells:,} cells]  {el/60:5.1f} min  {rate:4.1f} cells/s  "
                          f"ETA ~{eta:4.1f} h", flush=True)
    # Close the journal.
    jfh.close()
    # Rebuild the final CSV from the full journal (survives partial runs).
    finalize()
    # Done.
    print(f"[sweep] done in {(time.perf_counter()-t0)/3600:.2f} h -> {FINAL_CSV}", flush=True)


def finalize():
    # Read every journaled cell.
    recs = []
    with open(JOURNAL) as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    # Frame it.
    df = pd.DataFrame(recs)
    # Persist the labeled per-(config,name,day) CSV.
    df.to_csv(FINAL_CSV, index=False)
    # Mark the journal complete so a rerun does not re-journal.
    if JOURNAL.exists():
        JOURNAL.rename(JOURNAL.with_suffix(".jsonl.done"))


# Entry point.
if __name__ == "__main__":
    main()
