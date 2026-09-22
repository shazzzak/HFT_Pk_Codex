# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# lock_approach.py -- two measurements that set the cliff design from data, not guess:
#   (1) BAND-IN-SPREADS per symbol: (limit_up - limit_dn) / (2 * median spread).
#       This is the "runway" from mid-band to a lock. cliff = clip(0.10*runway, 3, 10).
#   (2) APPROACH SPEED on lock days: spreads-to-band at T-60/30/10/1s BEFORE the lock
#       first fires. If price is still many spreads out at T-1s and only snaps at the
#       end -> the graded widen ramp has no time to act and only the CLIFF matters.
#       If it crawls in over tens of seconds -> the ramp is useful.
#
# Vectorized from raw snapshots (no book reconstruction), like lock_screener.
# Run from existing_mm_live/:  python lock_approach.py

# filesystem paths
from pathlib import Path
# timing
import time
# dataframes
import pandas as pd
# numeric
import numpy as np
# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# driver (datasets, reads)
import run_legacy_mm as R

# raw store
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# the three run symbols: PACE (locky) + PPL/UBL (reference)
SYMBOLS = ["PACE", "PPL", "UBL"]
# tolerance (price units) for calling the touch "at" the published limit
LOCK_TOL = 0.01
# time offsets (seconds) before the lock to sample the approach
OFFSETS = [60, 30, 10, 1]
# cliff design params to overlay (pct of runway, floor, cap) -- the thing we're testing
CLIFF_PCT, CLIFF_FLOOR, CLIFF_CAP = 0.10, 3.0, 10.0
# cap days for a fast pass; None = all 207
DAYS_LIMIT = None


# compact mm:ss
def _fmt(sec):
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# per-message best bid/ask/ts for the continuous phase of one symbol-day
def per_message(s):
    # this script does NOT call build_events, so ts_exch isn't present yet --
    # derive it from orig_time exactly as build_events does (line 132 there).
    s = s.copy()
    # exchange-ms timestamp from the snapshot's orig_time
    s["ts_exch"] = R.to_ms(s["orig_time"])
    # continuous-trading snapshots only
    c = s[s["phase"] == "CONTINUOUS_AUCTION"]
    # nothing continuous -> None
    if len(c) == 0:
        return None, None, None
    # day-constant published limits (first non-null circuit-breaker prices)
    up = c.loc[c["entry_type"] == "UPPER_CIRCUIT_BREAKER", "px"].dropna()
    dn = c.loc[c["entry_type"] == "LOWER_CIRCUIT_BREAKER", "px"].dropna()
    # limits or None if the feed didn't publish them
    lim_up = float(up.iloc[0]) if len(up) else None
    lim_dn = float(dn.iloc[0]) if len(dn) else None
    # best bid per message (max BID px)
    bid = c[c["entry_type"] == "BID"].groupby("msg_seq")["px"].max()
    # best ask per message (min OFFER px)
    ask = c[c["entry_type"] == "OFFER"].groupby("msg_seq")["px"].min()
    # message timestamp (first ts_exch in the message)
    ts = c.groupby("msg_seq")["ts_exch"].first()
    # assemble a per-message frame, sorted by time
    m = pd.DataFrame({"ts": ts, "bid": bid, "ask": ask}).sort_values("ts")
    # return the frame + the two limits
    return m, lim_up, lim_dn


# analyse one symbol-day: band-in-spreads, and (if it locks) the approach samples
def analyse_day(m, lim_up, lim_dn):
    # need limits to do anything lock-related
    if m is None or lim_up is None or lim_dn is None:
        return None
    # per-message spread (NaN when one-sided)
    spr = (m["ask"] - m["bid"])
    # median spread over two-sided messages (the yardstick)
    med_spr = float(spr[spr > 0].median()) if (spr > 0).any() else np.nan
    # can't express anything in spreads without a spread
    if not (med_spr > 0):
        return None
    # runway = half-band width in median spreads
    band_in_spreads = (lim_up - lim_dn) / (2.0 * med_spr)
    # lock detection per message: bid at/above upper, or ask at/below lower
    up_lock = m["bid"] >= (lim_up - LOCK_TOL)
    dn_lock = m["ask"] <= (lim_dn + LOCK_TOL)
    # any lock at all?
    locked = (up_lock.fillna(False) | dn_lock.fillna(False))
    # base result (band width always; approach only if it locks)
    res = {"band_in_spreads": band_in_spreads, "approach": None}
    # no lock -> just the band width
    if not locked.any():
        return res
    # timestamp the lock FIRST fires
    first_idx = locked.idxmax() if locked.any() else None
    # the row where it first locks
    first_row = m.loc[first_idx]
    # lock timestamp
    t_lock = float(first_row["ts"])
    # which side locked (prefer whichever condition is true at first lock)
    side = "upper" if bool(up_lock.get(first_idx, False)) else "lower"
    # sample spreads-to-band at each offset BEFORE the lock
    approach = {}
    for off in OFFSETS:
        # target time = off seconds before the lock
        target = t_lock - off * 1000.0
        # candidate messages at or before the lock and at/after the target
        pre = m[(m["ts"] <= t_lock) & (m["ts"] >= target)]
        # need a two-sided quote to measure distance
        pre = pre[(pre["bid"].notna()) & (pre["ask"].notna())]
        # skip if nothing in that window
        if len(pre) == 0:
            approach[off] = np.nan
            continue
        # take the EARLIEST such message (closest to 'off' seconds before)
        row = pre.iloc[0]
        # distance to the relevant band, in median spreads
        if side == "upper":
            # spreads from best bid up to the upper limit
            approach[off] = (lim_up - row["bid"]) / med_spr
        else:
            # spreads from best ask down to the lower limit
            approach[off] = (row["ask"] - lim_dn) / med_spr
    # attach approach + side
    res["approach"] = approach
    res["side"] = side
    return res


def main():
    # all dates, optionally truncated
    dates = R.discover_dates()
    if DAYS_LIMIT is not None:
        dates = dates[:DAYS_LIMIT]
    # per-symbol accumulators
    band = {s: [] for s in SYMBOLS}
    # approach records: per symbol, list of {offset: spreads_to_band}
    appr = {s: [] for s in SYMBOLS}
    # lock-day counter
    lock_days = {s: 0 for s in SYMBOLS}
    # timers
    t0_all = time.perf_counter()
    sd = 0
    sd_total = len(dates) * len(SYMBOLS)
    # announce
    print(f"lock_approach: {len(SYMBOLS)} symbols x {len(dates)} days\n", flush=True)
    # OUTER over dates
    for date in dates:
        # open datasets
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        # MIDDLE over symbols
        for sym in SYMBOLS:
            # tick counter
            sd += 1
            # read snapshots (entry_type/px/phase/msg_seq)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            # skip empties
            if len(s) == 0:
                continue
            # per-message frame + limits
            m, lim_up, lim_dn = per_message(s)
            # analyse the day
            r = analyse_day(m, lim_up, lim_dn)
            # skip if unmeasurable
            if r is None:
                continue
            # record band width
            band[sym].append(r["band_in_spreads"])
            # record approach if the day locked
            if r["approach"] is not None:
                appr[sym].append(r["approach"])
                lock_days[sym] += 1
            # heartbeat
            if sd % 100 == 0:
                el = time.perf_counter() - t0_all
                proj = el / sd * sd_total
                print(f"  {sd}/{sd_total} symbol-days  elapsed {_fmt(el)}  "
                      f"ETA {_fmt(proj - el)}", flush=True)

    # ---- report ----
    print("\n--- band-in-spreads (runway) + derived cliff ---")
    # per symbol summary
    for sym in SYMBOLS:
        # skip symbols with no data
        if not band[sym]:
            print(f"{sym}: no data"); continue
        # median runway in spreads
        med_band = float(np.median(band[sym]))
        # derived cliff via the clip rule we're testing
        cliff = float(np.clip(CLIFF_PCT * med_band, CLIFF_FLOOR, CLIFF_CAP))
        # print
        print(f"  {sym:5s}: median band = {med_band:6.1f} spreads | "
              f"cliff = clip(10% , {CLIFF_FLOOR:.0f}, {CLIFF_CAP:.0f}) = {cliff:.1f} spreads | "
              f"lock-days = {lock_days[sym]}")

    print("\n--- approach speed: median spreads-to-band BEFORE the lock ---")
    print("  (if still far at T-1s and only near 0 at the lock -> SNAP: ramp useless, "
          "cliff is what matters)")
    # header
    print(f"  {'sym':5s} " + " ".join(f"T-{o}s" for o in OFFSETS))
    # per symbol approach medians
    approach_med = {}
    for sym in SYMBOLS:
        # skip if no lock days
        if not appr[sym]:
            print(f"  {sym:5s} (no lock days)"); continue
        # median spreads-to-band at each offset
        meds = []
        for off in OFFSETS:
            vals = [a[off] for a in appr[sym] if off in a and not np.isnan(a[off])]
            meds.append(float(np.median(vals)) if vals else np.nan)
        # store for the chart
        approach_med[sym] = meds
        # print row
        print(f"  {sym:5s} " + " ".join(f"{v:5.1f}" for v in meds))

    # ---- chart: approach curves + cliff line (focus on PACE, the locky one) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # x-axis: time before lock (reverse so lock/0 is at the right)
    xs = OFFSETS + [0]
    # plot each symbol's approach curve (append 0 at the lock = 0 spreads to band)
    for sym, meds in approach_med.items():
        # add the endpoint: at the lock, distance is ~0 spreads
        ys = meds + [0.0]
        ax.plot(xs, ys, "o-", label=f"{sym} approach")
    # overlay PACE's derived cliff as a horizontal line (if PACE present)
    if "PACE" in band and band["PACE"]:
        pace_cliff = float(np.clip(CLIFF_PCT * np.median(band["PACE"]), CLIFF_FLOOR, CLIFF_CAP))
        ax.axhline(pace_cliff, color="red", ls="--", lw=1.2,
                   label=f"PACE cliff = {pace_cliff:.1f} spreads (go dark here)")
    # time runs toward the lock -> invert x so left=far(60s), right=lock(0s)
    ax.invert_xaxis()
    # labels
    ax.set_xlabel("seconds before the lock fires (0 = lock)")
    ax.set_ylabel("spreads-to-band (distance price still has to cover)")
    ax.set_title("Lock approach speed: does price crawl in (ramp has time)\n"
                 "or snap at the end (only the cliff matters)?")
    # legend + grid
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # save
    fig.tight_layout()
    fig.savefig("lock_approach.png", dpi=140)
    plt.close(fig)
    print("\nwrote lock_approach.png")
    print("\nDECIDES: read PACE's row/curve. If spreads-to-band at T-1s is still well")
    print("above the cliff line, price snaps into the lock in the final tick(s) -> the")
    print("graded widen ramp cannot act in time; keep the CLIFF (go dark) + time trigger")
    print("and drop the elaborate ramp. If it descends gradually through the cliff line")
    print("over tens of seconds, the ramp is doing real work and we keep it.")


if __name__ == "__main__":
    main()
