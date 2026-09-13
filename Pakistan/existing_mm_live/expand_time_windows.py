# ============================================================================
# expand_time_windows.py -- STEP 5: per-name EOD unwind windows, with two
# corrections to calibrate_time_windows' methodology.
# ============================================================================
# THE RULE (unchanged, from the original):
#     minutes_needed = max_inv / (closing_vol_per_min * MAX_POV)
#     ramp  = clip(minutes_needed, RAMP_MIN, RAMP_MAX)
#     cliff = max(CLIFF_MIN, ramp / CLIFF_RATIO)
# Start the unwind exactly early enough that clearing max inventory never
# exceeds MAX_POV of the market's natural closing volume.
#
# CORRECTION 1 -- THE SESSION DEFINITION (the inconsistency SZ flagged)
#   The original measures the closing window against the LAST TRADE of the day
#   with no market filter:
#       t_close = ts.max()
#       in_win  = qty where ts >= t_close - CLOSING_WIN_MIN*60_000
#   build_volume_profile, calibrating the SAME market from the SAME data, does
#   the opposite: market == "REG" only, clipped to the continuous segments,
#   with the explicit note that "post-close prints are market=REG too, so the
#   phase window (not the market flag) is what excludes them".
#   Anchoring on the last print pulls the closing auction and post-close prints
#   into the window, inflating vol_per_min, which SHORTENS the ramp. This
#   version uses the continuous close (segs[-1][1]) and the same REG +
#   in-segments filters, so both calibrations now describe one session.
#
# CORRECTION 2 -- max_inv WAS 3x TOO SMALL
#   The original:  max_inv = MAXINV_CLIPS(10.0) * median_trade_size
#   commented "matches the universe runner: max_inv = 10 x clip".
#   But the engine does clip = CLIP_MULT(3.0) * median_trade_size, then
#   mm_harness.build_micro_params sets max_inv = 10 * clip -- i.e.
#       max_inv = 10 * 3.0 * median_trade_size = 30x, not 10x.
#   Verified against the live file: max_inv_sh / median_trade_qty = 10.0 on
#   every one of the 38 names. So minutes_needed is understated threefold.
#
#   Both errors shorten the window, and both are currently INVISIBLE because
#   37 of 38 names clip at the RAMP_MIN = 5.0 floor. Only TPL shows through at
#   6.6 min. The 76 candidates are thinner, so they will not hide it.
#
# CORRECTION 3 -- post-action windows for the four corporate-action names,
#   same dates and same reasoning as steps 3 and 4.
#
# SCOPE: all 114 names. The 38 are recomputed too, because leaving them on a
#   basis that is inconsistent with both the engine and the volume profile
#   would put two standards in one book. Every production window that moves is
#   reported explicitly with its before and after.
#
# Run from existing_mm_live/:
#   caffeinate -is python expand_time_windows.py --smoke   # 3 names, no write
#   caffeinate -is python expand_time_windows.py --full    # all 114, merges
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
# the harness, for load_segments (the calendar the engine itself uses)
import mm_harness as H
# the original window calibrator, for its policy constants
import calibrate_time_windows as CTW
# the profile builder, for in_segments
import build_volume_profile as BVP
# shared constants, guards and the merge helper
import expansion_names as EX

# ---------------------------------------------------------------------------
# PATH REBIND -- must run AFTER the imports above
# ---------------------------------------------------------------------------
# push the canonical raw-store root on and prove it reads
ALL_DATES = EX.bind_parsed_root(R)

# ---------------------------------------------------------------------------
# POLICY KNOBS -- taken from the original so the policy is unchanged
# ---------------------------------------------------------------------------
# never more than this share of the closing tape while unwinding
MAX_POV = CTW.MAX_POV
# the closing window over which natural volume is measured
CLOSING_WIN_MIN = CTW.CLOSING_WIN_MIN
# ramp floor and cap, in minutes
RAMP_MIN, RAMP_MAX = CTW.RAMP_MIN, CTW.RAMP_MAX
# the cliff preserves the 5:1 shape, floored at the 1-minute hard stop
CLIFF_RATIO, CLIFF_MIN = CTW.CLIFF_RATIO, CTW.CLIFF_MIN
# inventory cap in CLIPS -- the original's intent
MAXINV_CLIPS = 10.0
# the engine's clip multiplier. THIS is what the original omitted:
# clip = CLIP_MULT * median_trade_size, so max_inv = MAXINV_CLIPS * CLIP_MULT
# * median_trade_size. Set CLIP_MULT = 1.0 to reproduce the legacy behaviour.
CLIP_MULT = 3.0

# first post-action date for the four corporate-action names
POST_ACTION = {
    "FNEL": "2026-02-02",   # 1:10 split -- LIVE
    "BAFL": "2026-04-20",   # 1:2 split  -- LIVE
    "BML":  "2026-02-02",   # 19.4:1 reverse
    "MTL":  "2026-06-22",   # 1:2 split -- only 5 post-action days
}
# below this many contributing days a median is not a median
MIN_DAYS = 30
# explicit flags; anything else falls through to SMOKE
MODE = "full" if "--full" in sys.argv else "smoke"
# accepted flags
USAGE = "  usage: python expand_time_windows.py [--smoke | --full]"


def main():
    # announce the mode
    print(f"\nMODE = {MODE.upper()}" + ("  (default -- no flag given)"
                                        if len(sys.argv) == 1 else ""))
    print(USAGE + "\n")
    # every name in the post-expansion book
    syms = list(EX.ALL_NAMES)
    # smoke does one production name, one candidate, one post-action name
    if MODE == "smoke":
        syms = ["PPL", EX.NEW_NAMES[0], "FNEL"]

    # the trading calendar, as the engine sees it
    segments = H.load_segments()
    # report the setup
    print("=" * 78)
    print(f"TIME WINDOWS: {len(syms)} name(s) x {len(ALL_DATES)} dates")
    print("=" * 78)
    print(f"  parsed store : {R.PARSED_ROOT}")
    print(f"  segments     : {H.newest('session_segments_*.csv').name}")
    print(f"  MAX_POV={MAX_POV:.0%}  closing window={CLOSING_WIN_MIN}min  "
          f"ramp bounds=[{RAMP_MIN},{RAMP_MAX}]")
    print(f"  max_inv = {MAXINV_CLIPS} clips x {CLIP_MULT} x median trade size "
          f"= {MAXINV_CLIPS*CLIP_MULT:.0f}x  (legacy file used 10x)")
    # GUARD: the calendar must cover every date
    uncovered = [d for d in ALL_DATES if str(d) not in segments]
    # stop if it does not
    if uncovered:
        print(f"\n*** {len(uncovered)} dates have no session segments. Stopping.")
        raise SystemExit(2)
    print(f"  calendar covers all {len(ALL_DATES)} dates: OK\n")

    # per-name accumulators: daily closing-window shares, and daily median size
    close_vol = {s: [] for s in syms}
    med_trade = {s: [] for s in syms}
    # start the clock
    t0 = time.perf_counter()

    # ---- one pass over the dates ------------------------------------------
    # walk every trading date
    for i, date in enumerate(ALL_DATES, 1):
        # that date's datasets
        dsets = R.open_datasets(date)
        # missing partition
        if dsets is None:
            continue
        # this day's continuous segments
        segs = segments[str(date)]
        # CORRECTION 1: the close is the end of the last CONTINUOUS segment,
        # not the last print of the day
        t_close = segs[-1][1]
        # the start of the closing measurement window
        win_start = t_close - CLOSING_WIN_MIN * 60000
        # each name
        for sym in syms:
            # skip days before this name's corporate action
            if sym in POST_ACTION and str(date) < POST_ACTION[sym]:
                continue
            # read WITH the market column
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES + ["market"], sym)
            # no trades
            if len(t) == 0:
                continue
            # CORRECTION 1: REGULAR market prints only
            t = t[t["market"] == "REG"]
            # nothing left
            if len(t) == 0:
                continue
            # exchange-ms timestamps
            ts = R.to_ms(t["transact_time"]).to_numpy()
            # quantities
            qty = t["qty"].to_numpy()
            # CORRECTION 1: clip to the continuous segments, which is what
            # excludes pre-open and post-close prints
            m = BVP.in_segments(ts, segs)
            # nothing in session
            if not m.any():
                continue
            # the in-session trades
            ts, qty = ts[m], qty[m]
            # the day's median trade size, over in-session REG prints
            med_trade[sym].append(float(np.median(qty)))
            # shares traded in the closing window, measured to the real close
            close_vol[sym].append(float(qty[(ts >= win_start) & (ts <= t_close)].sum()))
        # heartbeat
        if i % 25 == 0 or i == len(ALL_DATES):
            print(f"  {i}/{len(ALL_DATES)} dates  "
                  f"{time.perf_counter()-t0:,.0f}s", flush=True)

    # ---- derive the windows -----------------------------------------------
    rows = []
    # walk the names
    for sym in syms:
        # no data at all
        if not close_vol[sym]:
            rows.append({"symbol": sym, "note": "NO_TRADES"})
            continue
        # median closing-window shares across days
        med_close_vol = float(np.median(close_vol[sym]))
        # per-minute rate over the closing window
        vol_per_min = med_close_vol / CLOSING_WIN_MIN
        # median daily median trade size
        med_sz = float(np.median(med_trade[sym]))
        # CORRECTION 2: the engine's real inventory cap
        max_inv = MAXINV_CLIPS * CLIP_MULT * med_sz
        # what the legacy file computed, kept for the comparison report
        legacy_max_inv = MAXINV_CLIPS * med_sz
        # a dead closing window means no passive clearing is possible
        if vol_per_min <= 0:
            minutes = RAMP_MAX
        else:
            # THE RULE
            minutes = max_inv / (vol_per_min * MAX_POV)
        # clip into the policy bounds
        ramp = float(np.clip(minutes, RAMP_MIN, RAMP_MAX))
        # the cliff preserves the 5:1 shape
        cliff = float(max(CLIFF_MIN, ramp / CLIFF_RATIO))
        # one row, columns matching the original plus provenance
        rows.append({"symbol": sym,
                     "eod_ramp_start_min": round(ramp, 1),
                     "eod_cliff_min": round(cliff, 1),
                     "minutes_needed_raw": round(minutes, 1),
                     "med_close_vol_sh": round(med_close_vol, 0),
                     "vol_per_min_sh": round(vol_per_min, 0),
                     "max_inv_sh": round(max_inv, 0),
                     "med_trade_sz": round(med_sz, 1),
                     "days": len(close_vol[sym]),
                     # a name that cannot clear max_inv inside the cap is a
                     # capacity problem, not a window problem
                     "capacity_flag": ("EXCEEDS_CAP" if minutes > RAMP_MAX else "ok"),
                     "note": "ok",
                     "windows_from": POST_ACTION.get(sym, ""),
                     "legacy_max_inv_sh": round(legacy_max_inv, 0)})
    # the new table
    new_df = pd.DataFrame(rows)

    # ---- report ------------------------------------------------------------
    # only the rows that produced numbers
    ok = new_df[new_df.note == "ok"] if "note" in new_df.columns else new_df
    # how many escape the floor now that max_inv is right
    print(f"\nramp distribution ({len(ok)} names):")
    print(f"  at the {RAMP_MIN:.0f}-min floor : {(ok.eod_ramp_start_min == RAMP_MIN).sum()}")
    print(f"  above the floor       : {(ok.eod_ramp_start_min > RAMP_MIN).sum()}")
    print(f"  EXCEEDS_CAP           : {(ok.capacity_flag == 'EXCEEDS_CAP').sum()}")
    # capacity problems first -- these are names to size down, not widen
    bad = ok[ok.capacity_flag != "ok"]
    # name them
    if len(bad):
        print(f"\n*** {len(bad)} names cannot clear max_inv within {RAMP_MAX:.0f}min "
              f"at {MAX_POV:.0%} POV:")
        print(bad[["symbol", "minutes_needed_raw", "vol_per_min_sh",
                   "max_inv_sh"]].to_string(index=False))
        print("    -> REDUCE max_inv on these (fewer clips) rather than widening")
        print("       the window further.")
    # thin samples
    thin = ok[ok["days"] < MIN_DAYS]
    # flag them
    if len(thin):
        print(f"\n*** {len(thin)} names have < {MIN_DAYS} contributing days:")
        print(thin[["symbol", "days", "windows_from"]].to_string(index=False))

    # smoke stops without writing
    if MODE == "smoke":
        print("\n" + "=" * 78)
        print("SMOKE RESULT (nothing written)")
        print("=" * 78)
        print(new_df.to_string(index=False))
        print("\n  -> run the full pass:")
        print("     caffeinate -is python expand_time_windows.py --full")
        return

    # ---- what changes for the production names -----------------------------
    # the file being replaced, for the before/after
    src_old = H.newest("time_windows_*.csv")
    # read it
    old = pd.read_csv(src_old).set_index("symbol")
    # the new view
    nv = new_df.set_index("symbol")
    # every production name whose window moved
    moved = []
    # walk them
    for s in EX.INCUMBENT:
        # only comparable when present in both
        if s in old.index and s in nv.index:
            # the two ramps
            a, b = old.loc[s, "eod_ramp_start_min"], nv.loc[s, "eod_ramp_start_min"]
            # record a change
            if a != b:
                moved.append({"symbol": s, "ramp_was": a, "ramp_now": b,
                              "cliff_was": old.loc[s, "eod_cliff_min"],
                              "cliff_now": nv.loc[s, "eod_cliff_min"],
                              "raw_was": old.loc[s, "minutes_needed_raw"],
                              "raw_now": nv.loc[s, "minutes_needed_raw"]})
    # the production impact, stated plainly
    print("\n" + "=" * 78)
    print(f"PRODUCTION IMPACT: {len(moved)} of {len(EX.INCUMBENT)} windows change")
    print("=" * 78)
    # show them
    if moved:
        print(pd.DataFrame(moved).to_string(index=False))
        print("\n  These move because max_inv is now the engine's real 30x rather")
        print("  than the legacy 10x, and because the closing window is measured")
        print("  to the continuous close on REG prints only. Both corrections")
        print("  LENGTHEN the ramp -- the old windows opened too late.")
    else:
        print("  none -- every production name still clips at the floor")

    # ---- merge and write ---------------------------------------------------
    print("\n" + "=" * 78)
    print("MERGE: existing UNION new")
    print("=" * 78)
    # carry any row not recomputed; append these
    merged, src, n_carried, n_new = EX.merge_with_existing(
        new_df, "time_windows_*.csv", key="symbol")
    # report
    print(f"  existing source : {src.name if src else '(none)'}")
    print(f"  carried through : {n_carried}")
    print(f"  recomputed      : {n_new}")
    print(f"  merged total    : {len(merged)}")
    # GUARD: every production name must survive
    missing = sorted(set(EX.INCUMBENT) - set(merged["symbol"].astype(str)))
    # refuse to drop the book
    if missing:
        print(f"\n*** ABORT: {len(missing)} production names absent: {missing}")
        raise SystemExit(2)
    # they are all there
    print(f"  production names present: {len(EX.INCUMBENT)}/{len(EX.INCUMBENT)} OK")
    # GUARD: load_windows filters note == "ok", so every name needs one
    if (merged["note"] != "ok").any():
        # the names that would be dropped by the loader
        dropped = merged.loc[merged["note"] != "ok", "symbol"].tolist()
        print(f"  note: {len(dropped)} names are not 'ok' and will fall back to "
              f"the (5.0, 1.0) default: {dropped}")

    # a fresh timestamped destination; never overwrites
    out = EX.safe_out("time_windows", "csv")
    # write it
    merged.to_csv(out, index=False)
    # say where it went
    print(f"\nwrote {out}")
    # the next step
    print("\n  -> next: step 6, preflight, then the backtest")


# entry point
if __name__ == "__main__":
    main()
