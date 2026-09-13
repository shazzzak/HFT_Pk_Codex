# ============================================================================
# expand_volume_profile.py -- STEP 4: per-name bucket volume rates for the 76
# new names, with post-action windows for the split-affected ones.
# ============================================================================
# BLOCKING for the backtest. fullyear_confirm._one() does
#     if sym not in scales or sym not in profiles: return None
# so without this every one of the 76 silently produces zero cells, exactly as
# on 2026-09-12.
#
# WHAT IT REUSES AND WHAT IT CHANGES
#   The bucket arithmetic is build_volume_profile's, reproduced exactly:
#     First15    = first 15 min of the FIRST continuous segment
#     Last15     = final 15 min of the LAST continuous segment
#     PreClose45 = minutes 60 -> 15 before the close, clamped on short days
#     Middle     = whatever remains
#     trades filtered to market == "REG" and clipped to the continuous segments
#     (post-close prints are market=REG too, so the phase window is what
#      excludes them -- not the market flag)
#     per-name output = the MEDIAN across days of each bucket's shares/min
#
#   Three deliberate differences:
#
#   1. SEGMENTS ARE READ, NOT RE-DETECTED. The original derives each day's
#      continuous segments from anchor snapshots and writes a new
#      session_segments_{stamp}.csv. That file is keyed by DATE, and
#      mm_harness.load_segments() takes the NEWEST one wholesale -- so writing
#      a fresh one risks silently changing the tradeable calendar for all 114
#      names. The existing file is complete and is what the engine already
#      uses, so this loads it via mm_harness.load_segments() and writes no
#      segments file at all. Fewer moving parts and nothing to clobber.
#
#   2. POST-ACTION WINDOWS. Four names have a corporate action inside the
#      window, confirmed against the exchange's own prev_close. shares/min
#      blends both regimes across a split, so for these the profile is built
#      from post-action days ONLY. BECO and BNL are NOT here: they are
#      majority-post-split, so their medians already sit in the current regime.
#
#   3. MERGED OUTPUT. The original writes only the names it processed.
#      load_profiles() reads the newest file alone, so a 76-name output would
#      delete the 38 production names. This writes EXISTING UNION NEW.
#
# Run from existing_mm_live/:
#   caffeinate -is python expand_volume_profile.py --smoke   # 2 names, no write
#   caffeinate -is python expand_volume_profile.py --full    # all 76, merges
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
# the original profile builder, untouched; sets its own stale PARSED_ROOT
import build_volume_profile as BVP
# shared constants, guards and the merge helper
import expansion_names as EX

# ---------------------------------------------------------------------------
# PATH REBIND -- must run AFTER the imports above, both of which set
# R.PARSED_ROOT at import time (mm_harness to the right one, BVP to the stale)
# ---------------------------------------------------------------------------
# push the canonical raw-store root on and prove it reads
ALL_DATES = EX.bind_parsed_root(R)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# the open/close bucket width, from the original
BUCKET_MIN = BVP.BUCKET_MIN
# first post-action date for the names whose profile would otherwise blend two
# regimes. Same four as the scale recalibration, same dates, same reasoning:
# BECO and BNL are majority-post-split so their medians are already correct.
POST_ACTION = {
    "FNEL": "2026-02-02",   # 1:10 split -- LIVE
    "BAFL": "2026-04-20",   # 1:2 split  -- LIVE
    "BML":  "2026-02-02",   # 19.4:1 reverse
    "MTL":  "2026-06-22",   # 1:2 split -- only 5 post-action days
}
# below this many contributing days a median is not a median
MIN_DAYS = 30
# explicit flags; anything else falls through to SMOKE
MODE = ("full" if "--full" in sys.argv
        else "fix-production" if "--fix-production" in sys.argv
        else "smoke")
# accepted flags
USAGE = ("  usage: python expand_volume_profile.py "
         "[--smoke | --full | --fix-production]")


def main():
    # announce the mode before anything expensive
    print(f"\nMODE = {MODE.upper()}" + ("  (default -- no flag given)"
                                        if len(sys.argv) == 1 else ""))
    print(USAGE + "\n")
    # the names to profile
    syms = list(EX.NEW_NAMES)
    # smoke does one ordinary name and one post-action name
    if MODE == "smoke":
        syms = [EX.NEW_NAMES[0], "FNEL"]
    # --fix-production repairs a BUG in the --full pass. POST_ACTION lists four
    # names, but syms was NEW_NAMES only, so the two that are PRODUCTION names
    # (FNEL, BAFL) were never reprocessed -- their POST_ACTION entries were
    # dead code, and the merge then carried their OLD blended 207-day rows
    # through exactly as designed. BML and MTL worked because they are in the
    # 76. This mode reprocesses just the production ones, on their post-action
    # windows, and is the ONLY mode permitted to change a production row.
    elif MODE == "fix-production":
        # exactly the POST_ACTION names that live in the production book
        syms = [x for x in POST_ACTION if x in EX.INCUMBENT]

    # ---- the trading calendar, as the ENGINE sees it -----------------------
    # load the existing session_segments rather than re-detecting them
    segments = H.load_segments()
    # report the source so a stale calendar is visible
    print("=" * 78)
    print(f"VOLUME PROFILE: {len(syms)} name(s) x {len(ALL_DATES)} dates")
    print("=" * 78)
    print(f"  parsed store : {R.PARSED_ROOT}")
    print(f"  segments     : {H.newest('session_segments_*.csv').name} "
          f"({len(segments)} dates)")
    # GUARD: the calendar must cover the dates being profiled, or the buckets
    # for the uncovered days would silently vanish
    uncovered = [d for d in ALL_DATES if str(d) not in segments]
    # report and stop if the calendar is short
    if uncovered:
        print(f"\n*** {len(uncovered)} dates have no session segments: "
              f"{uncovered[:5]}{' ...' if len(uncovered) > 5 else ''}")
        print("    Those days would be dropped from every name's profile.")
        raise SystemExit(2)
    # the calendar is complete
    print(f"  calendar covers all {len(ALL_DATES)} dates: OK")
    # name any post-action restrictions in force
    active = {s: d for s, d in POST_ACTION.items() if s in syms}
    # report them
    if active:
        print(f"  post-action windows: {active}")
    print()

    # per-name per-day bucket rates, in shares/min
    prof = {s: {"f": [], "m": [], "p": [], "l": []} for s in syms}
    # start the clock
    t0 = time.perf_counter()

    # ---- one pass over the dates ------------------------------------------
    # walk every trading date
    for i, date in enumerate(ALL_DATES, 1):
        # that date's datasets
        dsets = R.open_datasets(date)
        # a missing partition is a dead date
        if dsets is None:
            continue
        # this day's continuous segments, from the engine's own calendar
        segs = segments[str(date)]
        # tradeable minutes = the summed segment lengths (Jumu'ah-aware)
        tradeable = sum((e - s) for s, e in segs) / 60000.0
        # ---- bucket boundaries for this day, identical to the original -----
        # First15 = the first BUCKET_MIN of the FIRST segment
        f_end = segs[0][0] + BUCKET_MIN * 60000
        # Last15 = the final BUCKET_MIN of the LAST segment
        l_start = segs[-1][1] - BUCKET_MIN * 60000
        # PreClose45 = minutes 60 -> 15 before the final close
        p_start = segs[-1][1] - 60 * 60000
        # PreClose45 minutes actually available; short sessions clamp it
        p_min = max(min(45.0, tradeable - 2 * BUCKET_MIN), 1.0)
        # middle minutes = what remains after the three caps
        mid_min = max(tradeable - 2 * BUCKET_MIN - p_min, 1.0)
        # ---- each name's trades for this day -------------------------------
        # walk the names being profiled
        for sym in syms:
            # skip days before this name's corporate action
            if sym in POST_ACTION and str(date) < POST_ACTION[sym]:
                continue
            # read WITH the market column, as the original does
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES + ["market"], sym)
            # no trades that day
            if len(t) == 0:
                continue
            # REGULAR market prints only (drops negotiated / odd-lot / futures)
            t = t[t["market"] == "REG"]
            # nothing left after the market filter
            if len(t) == 0:
                continue
            # exchange-ms timestamps
            ts = R.to_ms(t["transact_time"]).to_numpy()
            # share quantities
            qty = t["qty"].to_numpy()
            # clip to the continuous segments -- this is what excludes the
            # pre-open and post-close prints, which are market=REG too
            m = BVP.in_segments(ts, segs)
            # nothing inside the continuous session
            if not m.any():
                continue
            # the in-session trades
            ts, qty = ts[m], qty[m]
            # bucket masks, precedence First15 -> Last15 -> PreClose45 -> Middle
            in_f = ts <= f_end
            in_l = ts >= l_start
            in_p = (ts >= p_start) & ~in_l & ~in_f
            in_m = ~(in_f | in_l | in_p)
            # shares per minute in each bucket for this day
            prof[sym]["f"].append(qty[in_f].sum() / BUCKET_MIN)
            prof[sym]["l"].append(qty[in_l].sum() / BUCKET_MIN)
            prof[sym]["p"].append(qty[in_p].sum() / p_min)
            prof[sym]["m"].append(qty[in_m].sum() / mid_min)
        # heartbeat
        if i % 25 == 0 or i == len(ALL_DATES):
            print(f"  {i}/{len(ALL_DATES)} dates  "
                  f"{time.perf_counter()-t0:,.0f}s", flush=True)

    # ---- collapse to one row per name --------------------------------------
    rows = []
    # walk the names
    for sym in syms:
        # that name's per-day rates
        p = prof[sym]
        # no contributing days at all
        if not p["f"]:
            rows.append({"symbol": sym, "note": "NO_TRADES"})
            continue
        # the median across days for each bucket, same as the original
        rows.append({"symbol": sym,
                     "vol_first15": round(float(np.median(p["f"])), 1),
                     "vol_middle": round(float(np.median(p["m"])), 1),
                     "vol_preclose45": round(float(np.median(p["p"])), 1),
                     "vol_last15": round(float(np.median(p["l"])), 1),
                     "days": len(p["f"]), "note": "ok",
                     # provenance: blank unless the window was restricted
                     "profile_from": POST_ACTION.get(sym, "")})
    # the new rows
    new_df = pd.DataFrame(rows)

    # ---- sanity: the U-shape the original checks ---------------------------
    # only the rows that produced numbers
    ok = new_df[new_df.note == "ok"] if "note" in new_df.columns else new_df
    # the measured close ramp
    if len(ok):
        # Last15 should generally exceed Middle
        u = (ok["vol_last15"] > ok["vol_middle"]).mean()
        # and PreClose45 should too
        ramp = (ok["vol_preclose45"] > ok["vol_middle"]).mean()
        # report both
        print(f"\nU-shape check: Last15 > Middle on {100*u:.0f}% of names; "
              f"PreClose45 > Middle on {100*ramp:.0f}%")
        print("  (the 38-name run measured the same ramp; a very different")
        print("   number here would mean the bucketing is wrong)")
    # names with too few contributing days for a stable median
    thin = ok[ok["days"] < MIN_DAYS] if "days" in ok.columns else ok.iloc[0:0]
    # flag them
    if len(thin):
        print(f"\n*** {len(thin)} names have < {MIN_DAYS} contributing days:")
        print(thin[["symbol", "days", "profile_from"]].to_string(index=False))

    # smoke stops without writing -- a partial file must never become newest()
    if MODE == "smoke":
        print("\n" + "=" * 78)
        print("SMOKE RESULT (nothing written)")
        print("=" * 78)
        print(new_df.to_string(index=False))
        print("\n  -> run the full pass:")
        print("     caffeinate -is python expand_volume_profile.py --full")
        return

    # ---- merge and write ---------------------------------------------------
    print("\n" + "=" * 78)
    print("MERGE: existing UNION new")
    print("=" * 78)
    # carry every existing row through; append the new ones
    merged, src, n_carried, n_new = EX.merge_with_existing(
        new_df, "volume_profile_*.csv", key="symbol")
    # name the file being extended
    print(f"  existing source : {src.name if src else '(none)'}")
    print(f"  carried through : {n_carried}")
    print(f"  newly profiled  : {n_new}")
    print(f"  merged total    : {len(merged)}")

    # GUARD 1: every production name must survive, or the engine drops them
    missing = sorted(set(EX.INCUMBENT) - set(merged["symbol"].astype(str)))
    # refuse to write a file that loses the production book
    if missing:
        print(f"\n*** ABORT: {len(missing)} production names absent: {missing}")
        raise SystemExit(2)
    # they are all there
    print(f"  production names present: {len(EX.INCUMBENT)}/{len(EX.INCUMBENT)} OK")

    # GUARD 2: production rows must be carried through unchanged
    if src is not None:
        # the file being extended
        old = pd.read_csv(src).set_index("symbol")
        # the merged view
        new_view = merged.set_index("symbol")
        # any production row whose numbers moved
        drift = []
        # the four bucket columns
        cols = ["vol_first15", "vol_middle", "vol_preclose45", "vol_last15"]
        # walk the production names
        for s in EX.INCUMBENT:
            # a production name being deliberately reprocessed is EXPECTED to
            # change; the inverse guard below checks it actually did
            if s in syms:
                continue
            # only compare names present in both
            if s in old.index and s in new_view.index:
                # compare each bucket
                for c in cols:
                    # pull the two values
                    a, b = old.loc[s, c], new_view.loc[s, c]
                    # NaN == NaN counts as unchanged
                    if not ((pd.isna(a) and pd.isna(b)) or a == b):
                        drift.append((s, c, a, b))
        # a changed production row means the merge is wrong
        if drift:
            print(f"\n*** ABORT: {len(drift)} production values CHANGED: "
                  f"{drift[:5]}")
            raise SystemExit(2)
        # they held
        n_checked = len([x for x in EX.INCUMBENT if x not in syms])
        print(f"  production profiles unchanged: {n_checked} OK")
        # INVERSE GUARD: a production name that was reprocessed but came back
        # identical means the post-action window silently failed to apply --
        # which is precisely the bug this mode exists to fix.
        redone = [x for x in syms if x in EX.INCUMBENT]
        # names that should have moved but did not
        stuck = []
        # walk them
        for s in redone:
            # only checkable when present in both
            if s in old.index and s in new_view.index:
                # identical on every bucket means nothing happened
                if all(old.loc[s, c] == new_view.loc[s, c] for c in cols):
                    stuck.append(s)
        # refuse a silent no-op
        if stuck:
            print(f"\n*** ABORT: {len(stuck)} production names were reprocessed")
            print(f"    but came back UNCHANGED: {stuck}")
            print("    The post-action window did not take effect.")
            raise SystemExit(2)
        # they moved
        if redone:
            print(f"  production profiles deliberately changed: {redone} OK")

    # a fresh timestamped destination; never overwrites
    out = EX.safe_out("volume_profile", "csv")
    # write it
    merged.to_csv(out, index=False)
    # say where it went
    print(f"\nwrote {out}")
    # no segments file was written, and that is deliberate
    print("  (no session_segments file written -- the existing one is complete")
    print("   and is what the engine already uses)")

    # ---- report ------------------------------------------------------------
    # names that produced nothing cannot be run by the engine
    bad = new_df[new_df.note != "ok"] if "note" in new_df.columns else new_df.iloc[0:0]
    # name them
    if len(bad):
        print(f"\n!!! {len(bad)} names have NO profile (the engine will SKIP "
              f"them): {list(bad.symbol)}")
    # the next step
    print("\n  -> next: step 5, time windows")


# entry point
if __name__ == "__main__":
    main()
