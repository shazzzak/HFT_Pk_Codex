# ============================================================================
# expand_scales.py -- STEP 3: back-solve session_scale for the 76 new names.
# ============================================================================
# Reuses calibrate_all_scales' ACTUAL calibration functions unchanged:
#     spread_and_price(sym)  -- median mid + median PKR spread from the store
#     sigma_median(sym, ds)  -- exact EMA sigma by replaying the real strategy
# and reapplies its exact back-solve rule:
#     session_scale = med_spread_pkr / (gamma * (sigma*fair)^2 * TAU * pos_lots_max)
#
# WHAT THIS WRAPPER CHANGES, AND WHY IT IS NOT JUST calibrate_all_scales.main()
#
#   1. PATHS. The original hardcodes the PRE-MOVE roots
#      (/Users/shazzak/Capital Stake - {Parsed,Results}) and does NOT import
#      config_pk. run_legacy_mm's own default is the EMPTY Google Drive path,
#      and the original rebinds it to the stale one at IMPORT time -- so the
#      rebind here must come AFTER the import.
#
#   2. THE NAME LIST. The original does
#          syms = pd.read_csv(WATCHLIST)["symbol"].tolist()
#      which is the 38-name watchlist. This calibrates the 76 candidates.
#
#   3. THE OUTPUT -- the reason main() cannot simply be called.
#      main() writes session_scales_{stamp}.csv containing ONLY the names it
#      calibrated. mm_harness.load_scales() takes the NEWEST matching file and
#      builds its dict from THAT FILE ALONE. A 76-name file would therefore
#      DELETE the 38 production names from load_scales(), and the next engine
#      run would silently drop the entire production book -- the same silent
#      failure shape as 2026-09-12, one layer down.
#      So this writes EXISTING UNION NEW via expansion_names.merge_with_existing:
#      every incumbent row is carried through BYTE-FOR-BYTE (never recomputed,
#      so production behaviour cannot shift), and the 76 new rows are appended.
#
#   4. CHECKPOINTING. ~1.6h of Book replay. Each completed name is appended to
#      a JSONL journal as it finishes; a re-run skips names already in it.
#      Nothing is ever deleted -- the journal is only ever appended to.
#
# COST: 76 names x 30 sampled days of full Book replay. At the 2.49 s/symbol-day
# measured on the step-2 build, roughly 1.5-2 hours.
#
# Run from existing_mm_live/:
#   caffeinate -is python expand_scales.py --smoke     # 1 name, ~2 min
#   caffeinate -is python expand_scales.py --full      # all 76
# ============================================================================

# command-line mode selection
import sys
# JSON lines for the checkpoint journal
import json
# wall-clock timing
import time
# numeric
import numpy as np
# frames
import pandas as pd

# the driver -- imported FIRST so the rebind below is the last word
import run_legacy_mm as R
# the original calibrator, untouched; sets its own stale PARSED_ROOT on import
import calibrate_all_scales as CAS
# shared expansion constants, guards and the merge helper
import expansion_names as EX

# ---------------------------------------------------------------------------
# PATH REBIND -- must run AFTER `import calibrate_all_scales`
# ---------------------------------------------------------------------------
# push the canonical raw-store root onto the driver and prove it reads
ALL_DATES = EX.bind_parsed_root(R)
# point the calibrator's feature-store root at the real one
CAS.FS_ROOT = EX.RESULTS_ROOT / "feature_store"
# and its output dir, though this wrapper does its own writing
CAS.OUT_DIR = EX.RESULTS_ROOT

# ---------------------------------------------------------------------------
# MODE
# ---------------------------------------------------------------------------
# explicit flags; anything else (including no flag) falls through to SMOKE
MODE = "full" if "--full" in sys.argv else "smoke"
# the accepted flags, printed in the banner
USAGE = "  usage: python expand_scales.py [--smoke | --full]"
# the checkpoint journal: fixed name so a re-run finds it and resumes
CKPT = EX.RESULTS_ROOT / "expand_scales_CKPT.jsonl"


def calibrate_one(sym, sample_dates, gamma, tau, pos_lots_max):
    # ONE name, using the original's functions unchanged.
    # median mid and median PKR spread, read from this name's feature store
    fair_ref, med_spread = CAS.spread_and_price(sym)
    # exact EMA sigma by replaying the real MicrostructureMM over sampled days
    sigma_ref = CAS.sigma_median(sym, sample_dates) if fair_ref is not None else None
    # any missing input means no scale -- record it, never crash the batch
    if fair_ref is None or med_spread is None or sigma_ref is None or sigma_ref <= 0:
        # the same MISSING_INPUTS shape the original writes
        return {"symbol": sym, "session_scale": np.nan, "fair_ref": fair_ref,
                "med_spread_pkr": med_spread, "sigma_ref": sigma_ref,
                "note": "MISSING_INPUTS"}
    # PKR price volatility = fractional sigma x price
    sigma_p = sigma_ref * fair_ref
    # THE BACK-SOLVE, identical to calibrate_all_scales.main()
    scale = med_spread / (gamma * (sigma_p ** 2) * tau * pos_lots_max)
    # the calibrated row, column-for-column matching the existing file
    return {"symbol": sym, "session_scale": round(scale, 4),
            "fair_ref": round(fair_ref, 4),
            "med_spread_pkr": round(med_spread, 5),
            "sigma_ref": sigma_ref, "note": "ok"}


def main():
    # announce the mode before anything expensive starts
    print(f"\nMODE = {MODE.upper()}" + ("  (default -- no flag given)"
                                        if len(sys.argv) == 1 else ""))
    print(USAGE + "\n")
    # strategy constants, read from the SAME source the original uses
    gamma = float(R.MICRO_PARAMS.get("gamma", 0.15))
    size0 = float(R.MICRO_PARAMS.get("size", 50))
    max_inv = float(R.MICRO_PARAMS.get("max_inv", 500))
    # inventory in lots at the cap -- 10 by construction (max_inv = 10 x size)
    pos_lots_max = max_inv / size0
    # worst-case horizon, the original's TAU
    tau = CAS.TAU
    # evenly-spaced calibration days spanning the whole range, exactly as the
    # original samples them
    step = max(1, len(ALL_DATES) // CAS.N_CAL_DAYS)
    sample_dates = ALL_DATES[::step][:CAS.N_CAL_DAYS]
    # the names to calibrate: the 76 candidates, never the incumbents
    syms = list(EX.NEW_NAMES)
    # smoke mode does a single name to prove the path end to end
    if MODE == "smoke":
        syms = syms[:1]

    # report the setup so the run is self-documenting
    print("=" * 74)
    print(f"SCALE CALIBRATION: {len(syms)} names x {len(sample_dates)} sampled days")
    print("=" * 74)
    print(f"  parsed store  : {R.PARSED_ROOT}")
    print(f"  feature store : {CAS.FS_ROOT}")
    print(f"  results dir   : {EX.RESULTS_ROOT}")
    print(f"  gamma={gamma}  TAU={tau}  pos_lots_max={pos_lots_max}")

    # ---- resume from the journal, if one exists ----------------------------
    # rows already calibrated in a previous run of this script
    done = {}
    # read the journal when present; it is append-only and never deleted
    if CKPT.exists():
        # each line is one completed name's row
        with open(CKPT) as fh:
            # walk the journal
            for line in fh:
                # skip blanks
                line = line.strip()
                if not line:
                    continue
                # tolerate a torn final line from a killed run
                try:
                    r = json.loads(line)
                    done[r["symbol"]] = r
                except (ValueError, KeyError):
                    continue
        # say how much work is being skipped
        print(f"  resuming: {len(done)} names already in {CKPT.name}")
    # the names still to do
    todo = [s for s in syms if s not in done]
    # report the split
    print(f"  to calibrate  : {len(todo)} of {len(syms)}\n")

    # ---- calibrate ---------------------------------------------------------
    # start the clock
    t0 = time.perf_counter()
    # append mode: the journal is only ever added to
    with open(CKPT, "a") as ck:
        # one name at a time
        for i, sym in enumerate(todo, 1):
            # do the work using the original's own functions
            row = calibrate_one(sym, sample_dates, gamma, tau, pos_lots_max)
            # journal it immediately, flushed, so a kill loses at most one name
            ck.write(json.dumps(row) + "\n")
            ck.flush()
            # keep it in memory too
            done[sym] = row
            # elapsed and ETA over the remaining names
            el = time.perf_counter() - t0
            eta = el / i * (len(todo) - i)
            # one progress line per name
            if row["note"] == "ok":
                print(f"  [{i}/{len(todo)}] {sym:9s} scale={row['session_scale']:>11,.4f}  "
                      f"(px {row['fair_ref']:.2f}, spr {row['med_spread_pkr']:.4f})  "
                      f"elapsed {CAS._fmt(el)}  ETA {CAS._fmt(eta)}", flush=True)
            else:
                print(f"  [{i}/{len(todo)}] {sym:9s} MISSING INPUTS -> NaN  "
                      f"elapsed {CAS._fmt(el)}  ETA {CAS._fmt(eta)}", flush=True)

    # the newly calibrated rows, in the requested order
    new_df = pd.DataFrame([done[s] for s in syms])

    # smoke stops here without writing a calibration file -- a 1-name file
    # must never reach the results dir, where it would become newest()
    if MODE == "smoke":
        print("\n" + "=" * 74)
        print("SMOKE RESULT (nothing written -- a 1-name file would shadow the real one)")
        print("=" * 74)
        print(new_df.to_string(index=False))
        # sanity: does the number look like the known anchors' order of magnitude?
        print("\n  known anchors for scale: PPL ~7.6, UBL ~3.9, PACE ~46.15")
        print("  a scale many orders of magnitude from these means sigma or")
        print("  spread was read wrong -- check fair_ref and med_spread_pkr above.")
        print("\n  -> if it looks sane, run the full calibration:")
        print("     caffeinate -is python expand_scales.py --full")
        return

    # ---- merge with the existing calibration, then write -------------------
    print("\n" + "=" * 74)
    print("MERGE: existing UNION new")
    print("=" * 74)
    # carry every incumbent row through untouched; append the new ones
    merged, src, n_carried, n_new = EX.merge_with_existing(
        new_df, "session_scales_*.csv", key="symbol")
    # name the file being extended
    print(f"  existing source : {src.name if src else '(none -- first ever)'}")
    # how many rows were preserved unchanged
    print(f"  carried through : {n_carried}")
    # how many are being added
    print(f"  newly calibrated: {n_new}")
    # the resulting total
    print(f"  merged total    : {len(merged)}")

    # GUARD: the merged table must still contain every production name, or the
    # next engine run silently loses them
    missing = sorted(set(EX.INCUMBENT) - set(merged["symbol"].astype(str)))
    # refuse to write a file that would drop the production book
    if missing:
        print(f"\n*** ABORT: {len(missing)} production names absent from the merged")
        print(f"    table: {missing}")
        print("    Writing this would delete them from load_scales(). Nothing written.")
        raise SystemExit(2)
    # the incumbents survived
    print(f"  production names present: {len(EX.INCUMBENT)}/{len(EX.INCUMBENT)} OK")

    # GUARD: the incumbents' values must be IDENTICAL to the source file --
    # this wrapper must never silently re-calibrate production
    if src is not None:
        # the file being extended
        old = pd.read_csv(src).set_index("symbol")
        # the merged view of the same names
        new_view = merged.set_index("symbol")
        # compare each incumbent's scale
        drifted = []
        # walk the production names
        for s in EX.INCUMBENT:
            # only compare names present in both
            if s in old.index and s in new_view.index:
                # the two scale values
                a, b = old.loc[s, "session_scale"], new_view.loc[s, "session_scale"]
                # NaN == NaN counts as unchanged
                same = (pd.isna(a) and pd.isna(b)) or (a == b)
                # record any change
                if not same:
                    drifted.append((s, a, b))
        # a changed incumbent means the merge logic is wrong
        if drifted:
            print(f"\n*** ABORT: {len(drifted)} production scales CHANGED: {drifted}")
            print("    Incumbent rows must be carried through byte-for-byte.")
            raise SystemExit(2)
        # they are untouched
        print(f"  production scales unchanged: OK")

    # a fresh timestamped destination; never overwrites a prior calibration
    out = EX.safe_out("session_scales", "csv")
    # write the merged table
    merged.to_csv(out, index=False)
    # say where it landed
    print(f"\nwrote {out}")

    # ---- report ------------------------------------------------------------
    # the new names that failed to calibrate cannot be run by the engine
    bad = new_df[new_df.note != "ok"]
    # name them explicitly
    if len(bad):
        print(f"\n!!! {len(bad)} new names have no scale (will be SKIPPED by the "
              f"engine): {list(bad.symbol)}")
    # the smell test the original does, now against the carried-through rows
    print("\n--- smell test vs known anchors (PPL~7.6, UBL~3.9, PACE~46.15) ---")
    # look each anchor up in the merged table
    for anchor in ("PPL", "UBL", "PACE"):
        # that anchor's row
        r = merged[merged.symbol == anchor]
        # print it when present
        if len(r):
            print(f"  {anchor}: {r.iloc[0]['session_scale']}")
    # the new names' scale distribution, so an implausible spread is visible
    ok_new = new_df[new_df.note == "ok"]
    # describe it only when there is something to describe
    if len(ok_new):
        print(f"\nnew-name session_scale distribution (n={len(ok_new)}):")
        print(ok_new["session_scale"].describe().to_string())
    # the next command
    print("\n  -> next: step 4, volume profile")


# entry point
if __name__ == "__main__":
    main()
