# build_volume_profile.py -- the two inputs SZ's unwind model needs:
#
#   (A) per-NAME bucket volume rates: First15 / Middle / Last15 shares-per-minute,
#       where First15 = first 15 min of CONTINUOUS trading, Last15 = final 15 min
#       before the close, Middle = everything in between (Friday: spans BOTH
#       continuous segments minus the Jumu'ah break).
#   (B) per-DATE session segments: the continuous-trading intervals of each day,
#       detected from the data (a gap > GAP_MIN minutes in market-wide trading =
#       a break). Handles regular days (one segment), split Fridays (two
#       segments), and Ramadan days (one short segment) WITHOUT a calendar.
#
# Each date is also CLASSIFIED into {reg, reg-f, ram-reg, ram-f} from its
# weekday + measured tradeable length, and cross-checked against the published
# PSX schedule lengths; mismatches are flagged for eyeballing rather than
# silently trusted.
#
# Outputs (timestamped):
#   volume_profile_<stamp>.csv   symbol -> vol_first15 / vol_middle / vol_last15
#   session_segments_<stamp>.csv date -> segments (ms), tradeable_min, type, flag
#
# Run from existing_mm_live/:  python build_volume_profile.py

# paths
from pathlib import Path
# timing + stamp
import time
from datetime import datetime
# frames + arrays
import pandas as pd
import numpy as np
# driver (datasets, reads, ms conversion)
import run_legacy_mm as R
# mm:ss formatter
import confirm_micro_vs_naive as C

# raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# shortlist + output dir
WATCHLIST = Path("/Users/shazzak/Capital Stake - Results/mm_watchlist_final.csv")
OUT_DIR = Path("/Users/shazzak/Capital Stake - Results")

# ------------------------------ knobs ----------------------------------------
# a gap in MARKET-WIDE trading longer than this = a session break (Jumu'ah ~152min)
GAP_MIN = 45.0
# the open/close bucket widths (fixed 15 minutes, per SZ's model)
BUCKET_MIN = 15.0
# published PSX tradeable lengths (minutes) for the classification cross-check:
#   reg: Mon-Thu 09:32-15:30 = 358 | reg-f: 09:17-12:00 + 14:32-16:30 = 163+118 = 281
#   ram-reg: 09:17-13:30 = 253     | ram-f: 09:17-12:30 = 193
EXPECTED = {"reg": 358.0, "reg-f": 281.0, "ram-reg": 253.0, "ram-f": 193.0}
# tolerance (min) when matching measured length to a published type
TOL = 20.0
# liquid anchor symbols whose SNAPSHOT phase defines the day's continuous
# segments (market-wide bells; several anchors so one halted name cannot
# mislead). Trades are then filtered to market=REG AND clipped to these
# segments -- post-close/pre-open prints are market=REG too, so the phase
# window (not the market flag) is what excludes them.
ANCHORS = ["UBL", "OGDC", "HBL"]
# ------------------------------------------------------------------------------


# detect the day's CONTINUOUS-phase segments from anchor snapshots (ms array of
# ts where phase == CONTINUOUS_AUCTION, pooled over anchors, sorted)
def detect_segments(ts_sorted):
    # gaps between consecutive continuous-phase observations (minutes)
    gaps = np.diff(ts_sorted) / 60000.0
    # indices where a break occurs (Jumu'ah ~152min >> GAP_MIN)
    brk = np.where(gaps > GAP_MIN)[0]
    # build [(start, end)] segments split at the breaks
    segs = []
    start = ts_sorted[0]
    for i in brk:
        segs.append((int(start), int(ts_sorted[i])))
        start = ts_sorted[i + 1]
    segs.append((int(start), int(ts_sorted[-1])))
    return segs


# is each timestamp inside ANY segment? (vectorised mask)
def in_segments(ts, segs):
    # start false, OR in each segment's interval
    m = np.zeros(len(ts), dtype=bool)
    for s, e in segs:
        m |= (ts >= s) & (ts <= e)
    return m


# classify a date into {reg, reg-f, ram-reg, ram-f} from weekday + measured length
def classify(date_str, tradeable_min, n_segs):
    # Friday?
    is_fri = pd.Timestamp(date_str).dayofweek == 4
    # candidate types for this weekday
    cands = ["reg-f", "ram-f"] if is_fri else ["reg", "ram-reg"]
    # nearest published length
    best = min(cands, key=lambda k: abs(EXPECTED[k] - tradeable_min))
    # flag if the measured length is far from ANY published type, or a split
    # session appears on a non-Friday (both deserve eyeballs, not silence)
    off = abs(EXPECTED[best] - tradeable_min) > TOL
    odd_split = (n_segs > 1) and not is_fri
    flag = "CHECK" if (off or odd_split) else "ok"
    return best, flag


def main():
    # run stamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # shortlist
    syms = pd.read_csv(WATCHLIST)["symbol"].tolist()
    # dates
    dates = R.discover_dates()
    print(f"build_volume_profile: {len(syms)} names x {len(dates)} days\n", flush=True)

    # per-date segment rows
    seg_rows = []
    # per-name per-day bucket volumes: {sym: {"f": [], "m": [], "l": []}} in sh/min
    prof = {s: {"f": [], "m": [], "p": [], "l": []} for s in syms}
    # timer
    t0_all = time.perf_counter()

    # one pass over days
    for i, date in enumerate(dates, 1):
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # ---- the day's CONTINUOUS segments, from anchor snapshots' phase ----
        anchor_ts = []
        for a in ANCHORS:
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, a)
            if len(s) == 0:
                continue
            # continuous-phase rows only (the exchange's own session state)
            c = s[s["phase"] == "CONTINUOUS_AUCTION"]
            if len(c):
                anchor_ts.append(R.to_ms(c["orig_time"]).to_numpy())
        # no anchor data -> cannot bound the session; skip the date
        if not anchor_ts:
            continue
        # pooled + sorted continuous-phase timeline
        tape = np.sort(np.concatenate(anchor_ts))
        # detect this day's continuous segments (Friday -> two)
        segs = detect_segments(tape)
        # ---- per-name trades: market=REG only, clipped to the segments ----
        per_sym = {}
        for sym in syms:
            # read WITH the market column (REQ_TRADES + market)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES + ["market"], sym)
            if len(t) == 0:
                continue
            # REGULAR market prints only (drops negotiated/odd-lot/futures)
            t = t[t["market"] == "REG"]
            if len(t) == 0:
                continue
            # exchange ms
            ts = R.to_ms(t["transact_time"]).to_numpy()
            qty = t["qty"].to_numpy()
            # clip to the CONTINUOUS segments (drops pre-open/post-close prints,
            # which are market=REG but outside the continuous phase)
            m = in_segments(ts, segs)
            if not m.any():
                continue
            per_sym[sym] = (ts[m], qty[m])
        # tradeable minutes = sum of segment lengths
        tradeable = sum((e - s) for s, e in segs) / 60000.0
        # classify + flag
        stype, flag = classify(str(date), tradeable, len(segs))
        # record (segments serialised "s1:e1;s2:e2")
        seg_rows.append({"date": str(date),
                         "segments": ";".join(f"{s}:{e}" for s, e in segs),
                         "n_segments": len(segs),
                         "tradeable_min": round(tradeable, 1),
                         "session_type": stype, "flag": flag})
        # ---- bucket boundaries for this day (4 buckets, 2026-08-20 upgrade) ----
        # First15 = first BUCKET_MIN of the FIRST segment
        f_end = segs[0][0] + BUCKET_MIN * 60000
        # Last15 = final BUCKET_MIN of the LAST segment
        l_start = segs[-1][1] - BUCKET_MIN * 60000
        # PreClose45 = minutes 60 -> 15 before the final close (the measured ramp:
        # 1.3x/1.4x/1.8x the midday rate across those three 15-min buckets)
        p_start = segs[-1][1] - 60 * 60000
        # PreClose45 minutes actually available (short sessions clamp it)
        p_min = max(min(45.0, tradeable - 2 * BUCKET_MIN), 1.0)
        # middle tradeable minutes (what remains after the three caps)
        mid_min = max(tradeable - 2 * BUCKET_MIN - p_min, 1.0)
        # ---- per-name bucket volumes ----
        for sym, (ts, qty) in per_sym.items():
            # masks per bucket; precedence: First15, Last15, PreClose45, Middle
            in_f = ts <= f_end
            in_l = ts >= l_start
            in_p = (ts >= p_start) & ~in_l & ~in_f
            in_m = ~(in_f | in_l | in_p)
            # shares per minute for each bucket this day
            prof[sym]["f"].append(qty[in_f].sum() / BUCKET_MIN)
            prof[sym]["l"].append(qty[in_l].sum() / BUCKET_MIN)
            prof[sym]["p"].append(qty[in_p].sum() / p_min)
            prof[sym]["m"].append(qty[in_m].sum() / mid_min)
        # heartbeat
        if i % 50 == 0 or i == len(dates):
            print(f"  {i}/{len(dates)} days  elapsed {C._fmt(time.perf_counter() - t0_all)}",
                  flush=True)

    # ---- write the per-date segments table ----
    seg_df = pd.DataFrame(seg_rows)
    seg_csv = OUT_DIR / f"session_segments_{stamp}.csv"
    seg_df.to_csv(seg_csv, index=False)

    # ---- write the per-name profile (median across days per bucket) ----
    rows = []
    for sym in syms:
        p = prof[sym]
        # need data
        if not p["f"]:
            rows.append({"symbol": sym, "note": "NO_TRADES"})
            continue
        rows.append({"symbol": sym,
                     "vol_first15": round(float(np.median(p["f"])), 1),
                     "vol_middle": round(float(np.median(p["m"])), 1),
                     "vol_preclose45": round(float(np.median(p["p"])), 1),
                     "vol_last15": round(float(np.median(p["l"])), 1),
                     "days": len(p["f"]), "note": "ok"})
    prof_df = pd.DataFrame(rows)
    prof_csv = OUT_DIR / f"volume_profile_{stamp}.csv"
    prof_df.to_csv(prof_csv, index=False)

    # ---- report ----
    # session-type census + any flagged days
    print("\nsession-type census:")
    print(seg_df.groupby(["session_type", "flag"]).size().to_string())
    bad = seg_df[seg_df.flag != "ok"]
    if len(bad):
        print(f"\n!!! {len(bad)} dates flagged CHECK (measured length far from published "
              f"schedule, or odd split):")
        print(bad[["date", "n_segments", "tradeable_min", "session_type"]]
              .to_string(index=False))
    # the U-shape sanity: Last15 should generally exceed Middle
    ok = prof_df[prof_df.note == "ok"]
    u = (ok["vol_last15"] > ok["vol_middle"]).mean()
    ramp = (ok["vol_preclose45"] > ok["vol_middle"]).mean()
    print(f"\nU-shape check: Last15 > Middle on {100*u:.0f}% of names; "
          f"PreClose45 > Middle on {100*ramp:.0f}% (the measured ramp)")
    print(f"\n{'symbol':8s} {'first15':>9s} {'middle':>9s} {'preclose45':>10s} {'last15':>9s}")
    for _, r in ok.sort_values("vol_last15", ascending=False).iterrows():
        print(f"{r.symbol:8s} {r.vol_first15:>9,.0f} {r.vol_middle:>9,.0f} "
              f"{r.vol_preclose45:>10,.0f} {r.vol_last15:>9,.0f}")
    print(f"\nwrote {prof_csv}\nwrote {seg_csv}")


if __name__ == "__main__":
    main()
