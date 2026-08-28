# diag_spread_by_bucket.py -- MEASURE, don't assume: what are PSX touch spreads
# on the top-10 spot names, by session bucket? Answers SZ's question directly:
# "PSX spreads are typically not that narrow, so the tight-book viability bypass
# won't impact us much -- what do the data say?"
#
# For each (name, bucket) it reports:
#   * median / p25 / p75 spread in BPS  (spread / mid)
#   * median spread in TICKS  (spread / tick)
#   * frac_below_gate: fraction of snapshots whose spread < 2 * cost_floor_half,
#     i.e. how often the viability gate would block (and the holding_exit bypass
#     would fire). cost_floor_half = (fee_pct + min_edge_pct) * mid.
#
# If frac_below_gate is ~0 across buckets, the tight-book bypass rarely fires and
# SZ's intuition holds: on PSX spot the spread almost always covers fee+min_edge,
# so Option 1 barely changes behavior. If it's material in some bucket (likely
# first15 or last15 where spreads gap), the bypass matters there.
#
# Reconstructs L1 touch from the LEVEL-BASED ob_snapshot (entry_type BID/OFFER,
# min-ask / max-bid per snapshot msg_seq) exactly as diag_kappa.py does.
#
# Run:  python3 existing_mm_live/diag_spread_by_bucket.py
# Smoke-defaulted to SMOKE_DAYS days; set None for the full panel.

from pathlib import Path
import numpy as np
import pandas as pd
import run_legacy_mm as R
import mm_harness as H

R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")

# ------------------------------ config ---------------------------------------
# top-10 spot names (the working watchlist)
NAMES = ["PPL", "OGDC", "LUCK", "ENGRO", "HUBC", "MARI", "FFC", "PSO", "UBL", "MCB"]
# session buckets
BUCKETS = ("first15", "middle", "preclose45", "last15")
# tick size (PSX spot)
TICK = 0.01
# fee + min_edge policy (from build_micro_params): the viability half uses these
FEE_PCT = 0.777e-4       # per side
MIN_EDGE_PCT = 0.0005    # 5 bps policy floor
# smoke: first N days. Set None for the full panel.
SMOKE_DAYS = 5
# -----------------------------------------------------------------------------


# canonical session bucket for a timestamp (ms), keyed off segments
def bucket_of(t, segs):
    open_ms = segs[0][0]
    close_ms = segs[-1][1]
    if t < open_ms + 15 * 60000:
        return "first15"
    if t >= close_ms - 15 * 60000:
        return "last15"
    if t >= close_ms - 60 * 60000:
        return "preclose45"
    return "middle"


# reconstruct L1 touch (bb, ba, ts) per snapshot from the level-based table,
# exactly as diag_kappa.py: bb = max BID px, ba = min OFFER px, per msg_seq.
def l1_touch(snap):
    # keep only continuous-auction snapshots (exclude auctions/halts)
    c = snap[snap["phase"] == "CONTINUOUS_AUCTION"].copy()
    if len(c) == 0:
        return None
    # raw snapshots carry NO ts_exch -- it is added downstream by R.build_events
    # (run_legacy_mm.py:132: ts_exch = to_ms(orig_time)). Derive it identically.
    c["ts_exch"] = R.to_ms(c["orig_time"])
    # bid and offer level rows
    bids = c[c["entry_type"] == "BID"]
    offs = c[c["entry_type"] == "OFFER"]
    if len(bids) == 0 or len(offs) == 0:
        return None
    # best bid = highest bid px per snapshot; best ask = lowest offer px
    bb = bids.groupby("msg_seq")["px"].max()
    ba = offs.groupby("msg_seq")["px"].min()
    # snapshot timestamp per msg_seq (exchange ms)
    ts = c.groupby("msg_seq")["ts_exch"].first()
    # join into one frame
    t = pd.DataFrame({"bb": bb, "ba": ba, "ts": ts}).dropna()
    # keep only two-sided, positive, non-crossed books
    t = t[(t["bb"] > 0) & (t["ba"] > 0) & (t["ba"] > t["bb"])]
    return t if len(t) else None


def main():
    # trading calendar + session segments
    all_dates = R.discover_dates()
    if SMOKE_DAYS is not None:
        all_dates = all_dates[:SMOKE_DAYS]
    segments = H.load_segments()
    # accumulator: per (name, bucket) -> list of spreads (bps) and (ticks) and
    # a count of snapshots below the viability-gate spread
    acc = {(n, b): {"bps": [], "ticks": [], "below": 0, "n": 0}
           for n in NAMES for b in BUCKETS}
    # walk days
    for di, date in enumerate(all_dates, 1):
        segs = segments.get(str(date))
        if segs is None:
            continue
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # heartbeat
        print(f"  day {di}/{len(all_dates)} {date}", flush=True)
        for sym in NAMES:
            # this symbol's snapshots
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            if len(s) == 0:
                continue
            # L1 touch series
            t = l1_touch(s)
            if t is None:
                continue
            # per-row mid, spread, bucket
            mid = 0.5 * (t["bb"] + t["ba"])
            spread = t["ba"] - t["bb"]
            # viability half in PKR at this mid (fee + min_edge)
            cost_floor_half = (FEE_PCT + MIN_EDGE_PCT) * mid
            # the gate blocks when spread < 2 * cost_floor_half
            below = spread < (2.0 * cost_floor_half)
            # assign each row to a bucket
            for ts_v, sp_v, md_v, bl_v in zip(t["ts"].to_numpy(),
                                              spread.to_numpy(),
                                              mid.to_numpy(),
                                              below.to_numpy()):
                b = bucket_of(float(ts_v), segs)
                key = (sym, b)
                if key not in acc:
                    continue
                # spread in bps and ticks
                acc[key]["bps"].append(1e4 * sp_v / md_v)
                acc[key]["ticks"].append(sp_v / TICK)
                acc[key]["below"] += int(bl_v)
                acc[key]["n"] += 1
    # ---- report ----
    print("\n" + "=" * 92)
    print("PSX TOUCH SPREAD by name x bucket  (continuous-auction snapshots)")
    print(f"viability gate blocks when spread < 2*(fee+min_edge)*mid  "
          f"[fee={FEE_PCT*1e4:.3f}bps min_edge={MIN_EDGE_PCT*1e4:.1f}bps]")
    print("=" * 92)
    hdr = (f"{'name':6s} {'bucket':11s} {'n':>8s} {'sprd_bps_p25':>12s} "
           f"{'sprd_bps_med':>12s} {'sprd_bps_p75':>12s} {'sprd_ticks_med':>14s} "
           f"{'frac_below_gate':>15s}")
    print(hdr)
    for n in NAMES:
        for b in BUCKETS:
            d = acc[(n, b)]
            if d["n"] == 0:
                print(f"{n:6s} {b:11s} {0:>8d} {'nan':>12s} {'nan':>12s} "
                      f"{'nan':>12s} {'nan':>14s} {'nan':>15s}")
                continue
            bps = np.array(d["bps"])
            tks = np.array(d["ticks"])
            frac_below = d["below"] / d["n"]
            print(f"{n:6s} {b:11s} {d['n']:>8,d} "
                  f"{np.percentile(bps,25):>12.2f} {np.median(bps):>12.2f} "
                  f"{np.percentile(bps,75):>12.2f} {np.median(tks):>14.1f} "
                  f"{frac_below:>15.3f}")
    # portfolio-level rollup per bucket (all names pooled)
    print("\n" + "-" * 92)
    print("POOLED across the 10 names, per bucket:")
    print(f"{'bucket':11s} {'n':>10s} {'sprd_bps_med':>12s} {'sprd_ticks_med':>14s} "
          f"{'frac_below_gate':>15s}")
    for b in BUCKETS:
        allbps = np.concatenate([np.array(acc[(n, b)]["bps"])
                                 for n in NAMES if acc[(n, b)]["n"] > 0]) \
            if any(acc[(n, b)]["n"] > 0 for n in NAMES) else np.array([])
        alltks = np.concatenate([np.array(acc[(n, b)]["ticks"])
                                 for n in NAMES if acc[(n, b)]["n"] > 0]) \
            if any(acc[(n, b)]["n"] > 0 for n in NAMES) else np.array([])
        tot_below = sum(acc[(n, b)]["below"] for n in NAMES)
        tot_n = sum(acc[(n, b)]["n"] for n in NAMES)
        if tot_n == 0:
            print(f"{b:11s} {0:>10d} {'nan':>12s} {'nan':>14s} {'nan':>15s}")
            continue
        print(f"{b:11s} {tot_n:>10,d} {np.median(allbps):>12.2f} "
              f"{np.median(alltks):>14.1f} {tot_below/tot_n:>15.3f}")
    print("-" * 92)
    print("READ: frac_below_gate ~0 -> spreads almost always cover fee+min_edge,")
    print("so the tight-book viability bypass rarely fires (SZ's intuition holds).")
    print("A material frac in first15/last15 -> the bypass matters in those buckets.")


if __name__ == "__main__":
    main()
