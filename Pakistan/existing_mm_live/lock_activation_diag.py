# lock_activation_diag.py -- WHY does the lock trigger fire so often on names that
# rarely lock? Replays the EXACT lock-zone math from micro_mm._trigger_state over
# each continuous two-sided book snapshot and records, at every ramp/cliff
# activation, the TRUE distance to the band -- in three units:
#   d_spreads : distance in instantaneous spreads (what the trigger uses)
#   d_pct     : distance as % of price (the ground truth: 10% = the actual lock)
#   spr_ratio : the spread at activation / the day's median spread
#
# Hypothesis under test: the trigger uses spread-units, so when the spread WIDENS
# (thin book), the spread-distance shrinks and the ramp fires even though price is
# far from the band in % terms. If activations cluster at large d_pct (e.g. 4-9%,
# nowhere near the 10% lock) AND high spr_ratio, that CONFIRMS the miscalibration is
# the spread yardstick -- a parameter/formula fix, not a dead idea.
#
# Vectorized from raw snapshots (no book reconstruction). Uses the SAME constants
# as the live strategy (imported), so it reproduces production behaviour exactly.
#
# Run in PyCharm:  python lock_activation_diag.py

# filesystem paths
from pathlib import Path
# timing
import time
# dataframes
import pandas as pd
# numeric
import numpy as np
# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# driver (datasets, reads, to_ms)
import run_legacy_mm as R
# the ACTUAL live trigger constants -- so the diagnostic == production logic
from micro_mm import (LOCK_START_FRAC, LOCK_START_MIN_SPR, LOCK_START_MAX_SPR,
                      LOCK_CLIFF_FRAC, LOCK_CLIFF_MIN_SPR, LOCK_CLIFF_MAX_SPR)

# raw store
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# symbols: PACE (locky) + PPL/UBL (rarely lock -- where spurious fires would show)
SYMBOLS = ["PPL", "UBL", "PACE"]
# tolerance (price) for calling the touch pinned at the limit
LOCK_TOL = 0.01
# a d_pct below this is "genuinely near the lock" (reference line on the plot)
NEAR_LOCK_PCT = 1.0
# cap days for speed; None = all 207
DAYS_LIMIT = None


# compact mm:ss
def _fmt(sec):
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# per-message best bid/ask for the continuous phase of one symbol-day
def per_message(s):
    # derive ts_exch (build_events isn't called here)
    s = s.copy()
    s["ts_exch"] = R.to_ms(s["orig_time"])
    # continuous only
    c = s[s["phase"] == "CONTINUOUS_AUCTION"]
    # limits (day-constant): first non-null circuit-breaker prices
    up = c.loc[c["entry_type"] == "UPPER_CIRCUIT_BREAKER", "px"].dropna()
    dn = c.loc[c["entry_type"] == "LOWER_CIRCUIT_BREAKER", "px"].dropna()
    lim_up = float(up.iloc[0]) if len(up) else None
    lim_dn = float(dn.iloc[0]) if len(dn) else None
    # best bid / ask per message
    bid = c[c["entry_type"] == "BID"].groupby("msg_seq")["px"].max()
    ask = c[c["entry_type"] == "OFFER"].groupby("msg_seq")["px"].min()
    # two-sided messages only
    m = pd.DataFrame({"bid": bid, "ask": ask}).dropna()
    # spread > 0
    m = m[(m["ask"] - m["bid"]) > 0]
    return m, lim_up, lim_dn


# replay the exact lock-zone classification for one day; return activation records
def classify_day(m, lim_up, lim_dn):
    # need limits and rows
    if m is None or lim_up is None or lim_dn is None or len(m) == 0:
        return None
    # arrays
    bid = m["bid"].to_numpy()
    ask = m["ask"].to_numpy()
    # instantaneous spread (the trigger's yardstick)
    spr = ask - bid
    # day median spread for the ratio context
    med_spr = float(np.median(spr))
    # band width in spreads (EXACT _trigger_state formula)
    band_spr = (lim_up - lim_dn) / (2.0 * spr)
    # cliff / start distances in spreads, clipped (EXACT formulas + constants)
    cliff_spr = np.clip(LOCK_CLIFF_FRAC * band_spr, LOCK_CLIFF_MIN_SPR, LOCK_CLIFF_MAX_SPR)
    start_spr = np.clip(LOCK_START_FRAC * band_spr, LOCK_START_MIN_SPR, LOCK_START_MAX_SPR)
    # ramp must sit outside the cliff (same guard as the live code)
    start_spr = np.maximum(start_spr, cliff_spr + 1.0)
    # distance of each touch to its band, in spreads
    d_up = np.maximum(0.0, lim_up - bid) / spr
    d_dn = np.maximum(0.0, ask - lim_dn) / spr
    # a side "fires" (ramp or cliff) when its spread-distance is inside start_spr
    fire_up = d_up < start_spr
    fire_dn = d_dn < start_spr
    # records: for each firing side, capture the truth in multiple units
    recs = []
    # upper-band fires
    for k in np.where(fire_up)[0]:
        recs.append({
            # distance in spreads (what the trigger saw)
            "d_spreads": float(d_up[k]),
            # distance as % of price (ground truth; ~10% = the real lock)
            "d_pct": float((lim_up - bid[k]) / bid[k] * 100.0),
            # is this a cliff (<=cliff_spr) or a ramp fire?
            "zone": "cliff" if d_up[k] <= cliff_spr[k] else "ramp",
            # spread relative to the day median (thin-book indicator)
            "spr_ratio": float(spr[k] / med_spr) if med_spr > 0 else np.nan,
            # which band
            "band": "upper",
        })
    # lower-band fires
    for k in np.where(fire_dn)[0]:
        recs.append({
            "d_spreads": float(d_dn[k]),
            "d_pct": float((ask[k] - lim_dn) / ask[k] * 100.0),
            "zone": "cliff" if d_dn[k] <= cliff_spr[k] else "ramp",
            "spr_ratio": float(spr[k] / med_spr) if med_spr > 0 else np.nan,
            "band": "lower",
        })
    # also return the total continuous message count for a firing RATE
    return {"n_msgs": len(m), "recs": recs}


def main():
    # dates
    dates = R.discover_dates()
    if DAYS_LIMIT is not None:
        dates = dates[:DAYS_LIMIT]
    # per-symbol: total messages, and all activation records
    tot_msgs = {s: 0 for s in SYMBOLS}
    fires = {s: [] for s in SYMBOLS}
    # timing
    t0 = time.perf_counter()
    sd = 0
    sd_total = len(dates) * len(SYMBOLS)
    print(f"lock_activation_diag: {len(SYMBOLS)} symbols x {len(dates)} days\n", flush=True)
    # walk days x symbols
    for date in dates:
        dsets = R.open_datasets(date)
        if dsets is None:
            continue
        for sym in SYMBOLS:
            sd += 1
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            if len(s) == 0:
                continue
            m, lim_up, lim_dn = per_message(s)
            r = classify_day(m, lim_up, lim_dn)
            if r is None:
                continue
            # accumulate
            tot_msgs[sym] += r["n_msgs"]
            fires[sym].extend(r["recs"])
            # heartbeat
            if sd % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  {sd}/{sd_total} symbol-days  elapsed {_fmt(el)}", flush=True)

    # ---- report ----
    print("\n--- lock-trigger firing rate + WHERE it fires (price truth) ---")
    for sym in SYMBOLS:
        # total continuous messages seen
        n = tot_msgs[sym]
        # activation records
        rr = fires[sym]
        # skip empty
        if n == 0:
            print(f"{sym}: no data"); continue
        # firing rate = activations / messages (can exceed 100% if both bands fire)
        rate = 100.0 * len(rr) / n
        # the ground-truth distance distribution at activation
        if rr:
            dpct = np.array([x["d_pct"] for x in rr])
            sprr = np.array([x["spr_ratio"] for x in rr])
            n_ramp = sum(1 for x in rr if x["zone"] == "ramp")
            n_cliff = sum(1 for x in rr if x["zone"] == "cliff")
            # the key numbers: how far from the band (in %) were fires, and how
            # wide was the spread then
            print(f"\n{sym}: {len(rr):,} activations over {n:,} messages ({rate:.1f}%)")
            print(f"  ramp={n_ramp:,}  cliff={n_cliff:,}")
            print(f"  d_pct at activation (dist to band, % of price):")
            print(f"    median {np.median(dpct):.2f}%  |  10th-90th [{np.percentile(dpct,10):.2f}, "
                  f"{np.percentile(dpct,90):.2f}]  |  near-lock (<{NEAR_LOCK_PCT}%): "
                  f"{100*np.mean(dpct<NEAR_LOCK_PCT):.0f}% of fires")
            print(f"  spr_ratio at activation (spread / day-median): "
                  f"median {np.median(sprr):.1f}x  |  90th {np.percentile(sprr,90):.1f}x")
        else:
            print(f"\n{sym}: 0 activations over {n:,} messages")

    # ---- plots: d_pct histogram + spr_ratio-vs-d_pct scatter, per symbol ----
    fig, axes = plt.subplots(2, len(SYMBOLS), figsize=(6 * len(SYMBOLS), 9))
    for j, sym in enumerate(SYMBOLS):
        rr = fires[sym]
        # TOP: histogram of d_pct at activation
        axT = axes[0, j]
        if rr:
            dpct = np.array([x["d_pct"] for x in rr])
            axT.hist(dpct, bins=40, color="#1f77b4", alpha=0.8)
            # the "genuinely near the lock" reference
            axT.axvline(NEAR_LOCK_PCT, color="green", ls="--", lw=1.2,
                        label=f"near-lock ({NEAR_LOCK_PCT}%)")
            # the band itself (10% region -- the actual lock)
            axT.axvline(10.0, color="red", ls=":", lw=1.2, label="band (~10%)")
            axT.legend(fontsize=8)
        axT.set_title(f"{sym}: dist-to-band at lock-trigger fire")
        axT.set_xlabel("distance to band (% of price)")
        axT.set_ylabel("activations")
        axT.grid(alpha=0.3)
        # BOTTOM: spr_ratio vs d_pct scatter (tests the wide-spread mechanism)
        axB = axes[1, j]
        if rr:
            dpct = np.array([x["d_pct"] for x in rr])
            sprr = np.array([x["spr_ratio"] for x in rr])
            axB.scatter(sprr, dpct, s=4, alpha=0.3, color="#d62728")
            axB.axhline(NEAR_LOCK_PCT, color="green", ls="--", lw=1)
        axB.set_title(f"{sym}: wide-spread -> spurious far-from-band fire?")
        axB.set_xlabel("spread / day-median (thin-book indicator)")
        axB.set_ylabel("distance to band (% of price)")
        axB.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("lock_activation_diag.png", dpi=140)
    plt.close(fig)
    print("\nwrote lock_activation_diag.png")
    print("\nDECIDES:")
    print("  If PPL/UBL fires cluster at LARGE d_pct (far from the 10% band) and high")
    print("  spr_ratio -> the spread-unit distance is the bug: wide-spread moments look")
    print("  'near the band' when price isn't. Fix = re-anchor the distance metric (use")
    print("  a reference spread, or measure distance in % of price), NOT scrap the idea.")
    print("  If fires cluster near the band (small d_pct) -> targeting is fine and the")
    print("  cost is the idea itself; rethink whether the lock trigger earns its keep.")


if __name__ == "__main__":
    main()
