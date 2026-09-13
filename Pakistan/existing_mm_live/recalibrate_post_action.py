# ============================================================================
# recalibrate_post_action.py -- recompute session_scale for the names whose
# price series has a corporate-action step in it, using the POST-action
# window ONLY.
# ============================================================================
# WHY ONLY FOUR NAMES
#   detect_splits.py confirmed six corporate actions against the exchange's own
#   published prev_close. But a split only CORRUPTS the calibration when the
#   207-day median lands in the wrong regime. Measured, calibrated price vs the
#   two regime prices:
#
#     BECO  calib 6.83   pre 71.29  post 7.14   -> landed POST, already correct
#     BNL   calib 12.44  pre 128.50 post 12.86  -> landed POST, already correct
#     BML   calib 6.25   pre 4.76   post 92.29  -> landed PRE,  WRONG
#     MTL   calib 532.75 pre 605.00 post 303.63 -> landed PRE,  WRONG
#     FNEL  calib 16.27  pre 17.78  post 1.77   -> landed PRE,  WRONG  (LIVE)
#     BAFL  calib 105.78 pre 126.00 post 62.67  -> landed PRE,  WRONG  (LIVE)
#
#   BECO and BNL are majority-post-split (73% and 66% of days), so their
#   medians fell in the current regime by weight of days. They are NOT
#   recalibrated here -- touching a correct value adds risk for no gain.
#
# WHAT CHANGES vs expand_scales.py
#   Only the data window. Same feature store, same sigma measurement through
#   the real MicrostructureMM, same back-solve:
#       scale = med_spread_pkr / (gamma * (sigma*fair)^2 * TAU * pos_lots_max)
#   Both inputs are restricted to dates >= the action date:
#     * the feature-store median mid and median PKR spread (date-filtered query)
#     * the sigma sample days (drawn only from post-action dates)
#
# EXPECTED RESULTS -- printed and checked against, so a silent no-op is caught.
#   scale ~ spread/price^2, and a k:1 action scales both by the same factor, so
#   the post-action scale should come out at (1/adj_ratio) times the blended one.
#     FNEL 18.75 -> ~188   BAFL 14.00 -> ~28   BML 92.68 -> ~4.8   MTL 4.36 -> ~8.7
#
# Run from existing_mm_live/:
#   caffeinate -is python recalibrate_post_action.py --smoke    # 1 name, no write
#   caffeinate -is python recalibrate_post_action.py --full     # all 4, merges
# ============================================================================

# command-line mode
import sys
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
# shared constants, guards and the merge helper
import expansion_names as EX

# ---------------------------------------------------------------------------
# PATH REBIND -- must run AFTER `import calibrate_all_scales`
# ---------------------------------------------------------------------------
# push the canonical raw-store root on and prove it reads
ALL_DATES = EX.bind_parsed_root(R)
# the feature store
CAS.FS_ROOT = EX.RESULTS_ROOT / "feature_store"
# results root
CAS.OUT_DIR = EX.RESULTS_ROOT

# ---------------------------------------------------------------------------
# THE AFFECTED NAMES
# ---------------------------------------------------------------------------
# symbol -> (first post-action date, adj_ratio from the exchange's prev_close,
#            human-readable factor, the blended scale this replaces)
# adj_ratio is the EXCHANGE's own adjustment, not the median-mid ratio -- it is
# computed from actual closes and is far cleaner (BNL: 0.1052 by median mid
# vs 0.100078 by exchange adjustment).
AFFECTED = {
    # 1:10 split. LIVE IN PRODUCTION. One tick went 5.6 -> 56.5 bps of price,
    # so this is a regime change, not just a wrong constant.
    "FNEL": ("2026-02-02", 0.099550, "1:10 split", 18.7502),
    # 1:2 split. LIVE IN PRODUCTION.
    "BAFL": ("2026-04-20", 0.497381, "1:2 split", 13.9976),
    # 19.4:1 consolidation -- matches no standard factor, flagged for manual
    # confirmation against PSX announcements before this value is trusted.
    "BML": ("2026-02-02", 19.388655, "19.4:1 reverse", 92.6775),
    # 1:2 split with only 5 post-action days in the window -- see the guard below
    "MTL": ("2026-06-22", 0.501868, "1:2 split", 4.3597),
}
# below this many post-action days, a median is not a median. MTL trips it.
MIN_POST_DAYS = 30

# explicit flags; anything else falls through to SMOKE
MODE = "full" if "--full" in sys.argv else "smoke"
# accepted flags
USAGE = "  usage: python recalibrate_post_action.py [--smoke | --full]"


def spread_and_price_from(sym, start_date):
    # CAS.spread_and_price restricted to dates >= start_date. Identical query
    # and identical filters otherwise -- the only change is the date floor.
    # ISO date strings sort lexicographically, so a string compare is correct.
    glob = f"{CAS.FS_ROOT}/{sym}/date=*.parquet"
    # DuckDB path, querying the parquet in place
    try:
        # in-place parquet query
        import duckdb
        # same medians as the original, plus the date floor and a day count
        q = f"""
            SELECT median(mid) AS fair_ref,
                   median(spread_bps * mid / 10000.0) AS med_spread_pkr,
                   count(DISTINCT regexp_extract(filename,'date=([0-9-]+)',1)) AS n_days,
                   count(*) AS n_rows
            FROM read_parquet('{glob}', filename=true)
            WHERE mid > 0 AND spread_bps > 0
              AND regexp_extract(filename, 'date=([0-9-]+)', 1) >= '{start_date}'
        """
        # one row back
        row = duckdb.sql(q).df().iloc[0]
        # the three numbers the caller needs
        return (float(row["fair_ref"]), float(row["med_spread_pkr"]),
                int(row["n_days"]), int(row["n_rows"]))
    except Exception:
        # pandas fallback: read each post-action partition
        import glob as _g
        # every partition for this symbol
        files = sorted(_g.glob(glob))
        # keep only those on or after the action date
        files = [f for f in files
                 if f.rsplit("date=", 1)[1].replace(".parquet", "") >= start_date]
        # nothing post-action
        if not files:
            return None, None, 0, 0
        # concatenate the two columns we need
        d = pd.concat([pd.read_parquet(f, columns=["mid", "spread_bps"])
                       for f in files], ignore_index=True)
        # same cleanliness filter as the original
        d = d[(d["mid"] > 0) & (d["spread_bps"] > 0)]
        # nothing usable
        if len(d) == 0:
            return None, None, len(files), 0
        # the two medians, plus counts
        return (float(d["mid"].median()),
                float((d["spread_bps"] * d["mid"] / 1e4).median()),
                len(files), len(d))


def recalibrate(sym, start_date, gamma, tau, pos_lots_max):
    # post-action median mid and median PKR spread
    fair_ref, med_spread, n_days, n_rows = spread_and_price_from(sym, start_date)
    # every trading date on or after the action
    post_dates = [d for d in ALL_DATES if str(d) >= start_date]
    # sample evenly across the post-action window, same cadence as the original
    step = max(1, len(post_dates) // CAS.N_CAL_DAYS)
    sample_dates = post_dates[::step][:CAS.N_CAL_DAYS]
    # exact EMA sigma, measured by replaying the real strategy on those days only
    sigma_ref = (CAS.sigma_median(sym, sample_dates)
                 if fair_ref is not None else None)
    # any missing input means no scale -- record it rather than crash
    if fair_ref is None or med_spread is None or sigma_ref is None or sigma_ref <= 0:
        # the same MISSING_INPUTS shape the original writes
        return {"symbol": sym, "session_scale": np.nan, "fair_ref": fair_ref,
                "med_spread_pkr": med_spread, "sigma_ref": sigma_ref,
                "note": "MISSING_INPUTS", "calib_from": start_date,
                "post_days": n_days, "sample_days": len(sample_dates)}
    # PKR price volatility
    sigma_p = sigma_ref * fair_ref
    # the back-solve, identical to calibrate_all_scales
    scale = med_spread / (gamma * (sigma_p ** 2) * tau * pos_lots_max)
    # the row, with the extra provenance columns. mm_harness._load_table pulls
    # only the columns it asks for, so these are safe to carry.
    return {"symbol": sym, "session_scale": round(scale, 4),
            "fair_ref": round(fair_ref, 4),
            "med_spread_pkr": round(med_spread, 5),
            "sigma_ref": sigma_ref, "note": "ok",
            "calib_from": start_date, "post_days": n_days,
            "sample_days": len(sample_dates), "n_rows": n_rows}


def main():
    # announce the mode before anything runs
    print(f"\nMODE = {MODE.upper()}" + ("  (default -- no flag given)"
                                        if len(sys.argv) == 1 else ""))
    print(USAGE + "\n")
    # strategy constants, from the same source the original uses
    gamma = float(R.MICRO_PARAMS.get("gamma", 0.15))
    size0 = float(R.MICRO_PARAMS.get("size", 50))
    max_inv = float(R.MICRO_PARAMS.get("max_inv", 500))
    # inventory in lots at the cap
    pos_lots_max = max_inv / size0
    # worst-case horizon
    tau = CAS.TAU
    # the names to redo
    syms = list(AFFECTED)
    # smoke does the production name with the largest error first
    if MODE == "smoke":
        syms = ["FNEL"]

    # report the setup
    print("=" * 78)
    print(f"POST-ACTION RECALIBRATION: {len(syms)} name(s)")
    print("=" * 78)
    print(f"  feature store : {CAS.FS_ROOT}")
    print(f"  gamma={gamma}  TAU={tau}  pos_lots_max={pos_lots_max}\n")

    # collected rows
    rows = []
    # start the clock
    t0 = time.perf_counter()
    # one name at a time
    for i, sym in enumerate(syms, 1):
        # its action date, factor and the blended value being replaced
        start, adj, factor, old_scale = AFFECTED[sym]
        # what the arithmetic says the answer should be: scale ~ spread/price^2,
        # and both scale by the action factor, so post/blended = 1/adj
        expected = old_scale / adj
        # say what is being attempted
        print(f"  [{i}/{len(syms)}] {sym}  {factor} on {start}")
        print(f"      blended scale {old_scale:>10.4f}  -> expected ~{expected:>10.2f}",
              flush=True)
        # do the work on the post-action window only
        row = recalibrate(sym, start, gamma, tau, pos_lots_max)
        # carry the action metadata onto the row
        row["corp_action"] = f"{factor} {start}"
        # keep it
        rows.append(row)
        # report what came back
        if row["note"] == "ok":
            # the ratio of measured to expected -- 1.0 means the arithmetic holds
            got = row["session_scale"]
            print(f"      measured      {got:>10.4f}  "
                  f"(px {row['fair_ref']:.2f}, spr {row['med_spread_pkr']:.4f}, "
                  f"{row['post_days']} post-action days, "
                  f"{row['sample_days']} sampled)")
            print(f"      measured/expected = {got/expected:.3f}  "
                  f"(1.0 = the k-fold arithmetic holds exactly)")
            # too few days for a stable median
            if row["post_days"] < MIN_POST_DAYS:
                print(f"      *** ONLY {row['post_days']} POST-ACTION DAYS "
                      f"(< {MIN_POST_DAYS}). This median is not reliable.")
                print(f"          Recommend EXCLUDING {sym} from the expansion")
                print(f"          rather than trading a calibration built on it.")
        else:
            print(f"      MISSING INPUTS -> NaN")
        # elapsed
        print(f"      elapsed {CAS._fmt(time.perf_counter()-t0)}\n", flush=True)

    # the recalibrated rows
    new_df = pd.DataFrame(rows)

    # smoke stops without writing -- a partial file must never become newest()
    if MODE == "smoke":
        print("=" * 78)
        print("SMOKE RESULT (nothing written)")
        print("=" * 78)
        print(new_df.to_string(index=False))
        print("\n  -> if measured/expected is near 1.0, run the full pass:")
        print("     caffeinate -is python recalibrate_post_action.py --full")
        return

    # ---- merge into the current calibration --------------------------------
    print("=" * 78)
    print("MERGE: existing UNION recalibrated")
    print("=" * 78)
    # replace only these names; carry every other row through untouched
    merged, src, n_carried, n_new = EX.merge_with_existing(
        new_df, "session_scales_*.csv", key="symbol")
    # name the file being extended
    print(f"  existing source : {src.name if src else '(none)'}")
    print(f"  carried through : {n_carried}")
    print(f"  recalibrated    : {n_new}")
    print(f"  merged total    : {len(merged)}")

    # GUARD 1: every name that existed before must still be there
    if src is not None:
        # the file being extended
        old = pd.read_csv(src)
        # names that vanished
        lost = sorted(set(old.symbol.astype(str)) - set(merged.symbol.astype(str)))
        # refuse to write a file that drops names
        if lost:
            print(f"\n*** ABORT: {len(lost)} names lost in the merge: {lost}")
            raise SystemExit(2)
        # all present
        print(f"  no names lost   : {len(old)} -> {len(merged)} OK")

        # GUARD 2: every name NOT being recalibrated must be byte-identical
        untouched = [s for s in old.symbol.astype(str) if s not in AFFECTED]
        # index both for comparison
        o = old.set_index("symbol")
        m = merged.set_index("symbol")
        # any that drifted
        drift = []
        # walk the untouched names
        for s in untouched:
            # the two values
            a, b = o.loc[s, "session_scale"], m.loc[s, "session_scale"]
            # NaN == NaN counts as unchanged
            if not ((pd.isna(a) and pd.isna(b)) or a == b):
                drift.append((s, a, b))
        # a drifted untouched name means the merge is wrong
        if drift:
            print(f"\n*** ABORT: {len(drift)} untouched scales CHANGED: {drift[:5]}")
            raise SystemExit(2)
        # they held
        print(f"  untouched scales unchanged: {len(untouched)} OK")

        # GUARD 3 (the inverse): every recalibrated name MUST have changed. If
        # one did not, the date filter silently failed and this run did nothing.
        same = []
        # walk the names that were supposed to change
        for s in AFFECTED:
            # only check names present in both
            if s in o.index and s in m.index:
                # unchanged means the filter did not apply
                if o.loc[s, "session_scale"] == m.loc[s, "session_scale"]:
                    same.append(s)
        # refuse a silent no-op
        if same:
            print(f"\n*** ABORT: {len(same)} names were supposed to be")
            print(f"    recalibrated but are UNCHANGED: {same}")
            print("    The post-action date filter did not take effect.")
            raise SystemExit(2)
        # they all moved
        print(f"  all {len(AFFECTED)} target names changed: OK")

    # a fresh timestamped destination; never overwrites
    out = EX.safe_out("session_scales", "csv")
    # write it
    merged.to_csv(out, index=False)
    # say where it went
    print(f"\nwrote {out}")

    # ---- before / after ----------------------------------------------------
    print("\nBEFORE -> AFTER")
    print(f"  {'name':7} {'blended':>10} {'post-action':>12} {'ratio':>8} {'action':>22}")
    # one line per recalibrated name
    for r in new_df.itertuples():
        # the value being replaced
        old_scale = AFFECTED[r.symbol][3]
        # the change factor
        ratio = r.session_scale / old_scale if old_scale else np.nan
        # one row
        print(f"  {r.symbol:7} {old_scale:>10.4f} {r.session_scale:>12.4f} "
              f"{ratio:>8.2f}x {r.corp_action:>22}")
    # the production reminder
    print("\nFNEL and BAFL are LIVE. Their scales have now changed materially,")
    print("so any production run using the previous file will behave differently.")
    print("FNEL is additionally a ~1-tick book post-split (one tick = 56.5 bps of")
    print("price), which is the regime CHEAP_EXCLUDED exists for -- a corrected")
    print("scale does not by itself make it tradeable.")
    # next step
    print("\n  -> next: step 4, volume profile (same post-action treatment needed")
    print("     for these names' shares/min, which blend the two regimes too)")


# entry point
if __name__ == "__main__":
    main()
