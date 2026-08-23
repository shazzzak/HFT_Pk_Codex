# sweep_depth_analysis.py -- how deep do aggressor orders cut into the book?
#
# QUESTION: is futures flow more aggressive/institutional than spot? A trade that
# sweeps many book levels is a large, aggressive order (size/urgency = the
# footprint of informed/institutional flow). If futures show systematically
# DEEPER sweeps than spot, that supports "futures MM fails because the flow is
# more toxic", and explains the markout > capture result.
#
# METHOD. A single aggressor order that sweeps L1->L2->L3 appears as MULTIPLE
# trade rows (one per resting order hit), all sharing the SAME aggressor order
# reference. We group by that reference to reconstruct each aggressor's full
# sweep, then measure how many BOOK LEVELS it consumed against the pre-trade
# L10 snapshot. PSX disseminates 10 explicit levels; everything beyond L10 is a
# single aggregated AGG_BID / AGG_OFFER block, so a sweep that exhausts all 10
# visible levels and reaches the AGG block is bucketed L10+.
#
# TWO METHODS (both reported; divergence itself is informative -> hidden depth):
#   M1 trade-distinct : # of DISTINCT trade prices in the aggressor's sweep.
#                       Simplified -- conflates "distinct prices" with "levels"
#                       and cannot see resting depth; fast, trades-only.
#   M2 book-diff      : walk the aggressor's total swept qty against the
#                       pre-trade snapshot's resting depth per level -> the true
#                       number of book levels consumed. PRODUCTION version.
#
# TWO WEIGHTINGS (both reported):
#   count  : each aggressor order = 1 (how OFTEN deep sweeps happen)
#   volume : each aggressor order weighted by its shares (what FRACTION of
#            traded volume arrives via deep sweeps -- the institutional-flow read)
#
# SELF-CHECK: before trusting anything, the script verifies that the aggressor
# reference actually groups multi-level sweeps (prints rows-per-aggressor stats).
#
# Run from existing_mm_live/:  python3 sweep_depth_analysis.py
# Edit SPOT_SYMS / FUT_ROOTS + DATES below.

# filesystem paths
from pathlib import Path
# timing + stamp
import time
# stdlib datetime for the run stamp
from datetime import datetime
# frames + arrays
import pandas as pd
# numpy for vectorised depth math
import numpy as np
# plotting (Agg backend -> file output, no display needed)
import matplotlib
# headless backend -> write PNGs, no display
matplotlib.use("Agg")
# pyplot for the histograms
import matplotlib.pyplot as plt
# the driver (dataset discovery, readers, roll map for futures)
import run_legacy_mm as R
# futures roll map + calendar
import futures_mm_run as F
# heartbeat
import confirm_micro_vs_naive as C

# raw store + results
R.PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# where PNGs + CSV are written
RESULTS = Path("/Users/shazzak/Capital Stake - Results")

# full trade columns needed for sweep grouping: aggressor side + BOTH order refs
# + initiator (the aggressor order id) + exec_type to filter to real trades
REQ_TRADES_FULL = ["symbol", "transact_time", "capture_ts", "price", "qty",
                   "initiator", "aggressor_side", "buy_ref", "sell_ref",
                   "resting_ref", "exec_type", "appl_seq"]
# snapshot columns needed for the book-diff: level + qty per level + entry_type
REQ_SNAP_FULL = ["symbol", "msg_seq", "orig_time", "capture_ts", "entry_type",
                 "level", "px", "qty", "phase"]

# ------------------------------ config ------------------------------
# spot symbols to analyse (edit as needed)
SPOT_SYMS = ["PPL", "UBL", "MLCF", "TRG", "BOP"]
# futures roots to analyse (mapped to active contract per date via the roll map)
FUT_ROOTS = ["MLCF", "TRG", "BOP"]
# dates to analyse; None = a sample of the first N_SAMPLE trading days
DATES = None
# how many days to sample when DATES is None (sweep depth is stable; a sample
# is enough and keeps runtime down -- flagged: widen for a final figure)
N_SAMPLE = 500
# terminal bucket: sweeps reaching AGG_BID/AGG_OFFER (beyond L10) -> "L10+"
MAX_LEVEL = 10
# ---------------------------------------------------------------------


# ---- resolve the aggressor reference for each trade row ----
# the aggressor is whichever side crossed the spread; its order ref is buy_ref
# when aggressor_side is BUY, else sell_ref. All rows of one sweep share it.
# NOTE on grouping: the data has NO aggressor order id. `initiator` is only a
# label (AUCTION/BUYER_INITIATED/SELLER_INITIATED), and the aggressor-side ref
# column is 0 on aggressor rows (only resting_ref/the passive order is recorded).
# So a sweep is reconstructed HEURISTICALLY by SAME-TIMESTAMP + SAME-SIDE
# clustering: trades sharing one exact exchange transact_time on the same
# aggressor side are one marketable order filling against multiple resting
# orders. This is a conservative LOWER BOUND on sweep depth (a sweep the
# exchange spread across timestamps is split into separate groups). Done inline
# in analyse_symbol_day; no per-order key function is possible.


# ---- M2 book-diff: how many snapshot levels does `swept_qty` consume? ----
# given the pre-trade book on the RESTING side (levels sorted best-first, with an
# AGG block last), walk the swept quantity down the levels and count how many
# levels are (partially or fully) consumed. Returns min(levels, MAX_LEVEL+1)
# where MAX_LEVEL+1 encodes the L10+ (touched AGG) bucket.
def levels_consumed(swept_qty, level_qtys, has_agg):
    # remaining quantity to account for
    rem = float(swept_qty)
    # levels touched so far
    touched = 0
    # walk explicit levels best-first
    for q in level_qtys:
        # nothing left -> done
        if rem <= 1e-9:
            break
        # this level is (partially) consumed
        touched += 1
        # subtract this level's resting size
        rem -= float(q)
    # if quantity remains after all explicit levels, it reached the AGG block
    if rem > 1e-9 and has_agg:
        # encode L10+ as MAX_LEVEL+1 (touched beyond the disseminated depth)
        return MAX_LEVEL + 1
    # if it consumed more explicit levels than MAX_LEVEL (shouldn't exceed 10 on
    # PSX, but guard), cap at L10+
    if touched > MAX_LEVEL:
        return MAX_LEVEL + 1
    # otherwise the number of explicit levels touched (>=1 if any qty)
    return max(touched, 1) if swept_qty > 0 else 0


# ---- pre-trade resting-side book from the snapshot, at/just before a time ----
# returns (level_qtys_best_first, has_agg) for the side the aggressor HITS:
# a BUY aggressor hits OFFERs, a SELL aggressor hits BIDs.
# ---- build a per-symbol-day snapshot INDEX once (the speed fix) ----
# Instead of rescanning the whole snapshot table per cluster (O(n) each, the
# 97-min bottleneck), we pre-extract every snapshot message ONCE into a dict
# keyed by msg_seq, and a SORTED array of each message's timestamp. A cluster
# then finds its book by binary-searching the sorted times (O(log n)).
def build_snapshot_index(snap):
    # continuous-phase rows only
    c = snap[snap["phase"] == "CONTINUOUS_AUCTION"]
    # empty -> a null index
    if len(c) == 0:
        return None
    # ---- FAST PATH: snapshots arrive pre-sorted by the parser on
    # (symbol, msg_seq, entry_type, level); within this single-symbol read that
    # means msg_seq order = exchange sequence = time order. So we do NOT sort. ----
    # pull the columns we need as numpy arrays once (no per-row pandas access)
    seq = c["msg_seq"].to_numpy()
    et = c["entry_type"].to_numpy()
    px = c["px"].to_numpy(dtype=float)
    qty = c["qty"].to_numpy(dtype=float)
    # message time in ms, converted ONCE for the whole table
    t_ms = R.to_ms(c["orig_time"]).to_numpy()
    # message boundaries: indices where msg_seq changes (data is grouped by seq)
    # (np.unique with return_index gives the first row of each message, in order)
    uniq_seq, first_idx = np.unique(seq, return_index=True)
    # CONTIGUITY GUARD: the fast array-slicing path REQUIRES rows contiguous by
    # msg_seq (the parser sorts by symbol,msg_seq,... so a single-symbol read is
    # contiguous). If the count of unique seqs != count of seq-change points, the
    # data is not grouped and the slicing would be wrong -> fall back to a sort.
    changes = 1 + int(np.count_nonzero(np.diff(seq)))
    if changes != len(uniq_seq):
        # not contiguous: sort a copy by msg_seq and rebuild the arrays in order
        order = np.argsort(seq, kind="stable")
        seq = seq[order]; et = et[order]; px = px[order]
        qty = qty[order]; t_ms = t_ms[order]
        # recompute boundaries on the now-sorted arrays
        uniq_seq, first_idx = np.unique(seq, return_index=True)
    # np.unique sorts ascending; msg_seq ascending IS time ascending, so the
    # returned order is already the time order we need for searchsorted
    # append a sentinel end index so each message spans [first_idx[i], next)
    bounds = np.append(first_idx, len(seq))
    # per-message book dict + the sorted time array (one time per message)
    books = {}
    # the message time = the first row's time in that message block
    time_order = t_ms[first_idx]
    # walk each message block by slicing the pre-sorted arrays (no groupby)
    for i in range(len(uniq_seq)):
        # this message's row span
        a, b = bounds[i], bounds[i + 1]
        # entry types / prices / qtys for this message
        et_i = et[a:b]
        px_i = px[a:b]
        q_i = qty[a:b]
        # bid mask + offer mask within the message
        is_bid = et_i == "BID"
        is_off = et_i == "OFFER"
        # bid prices/qtys, sorted best-first (highest price first)
        bpx = px_i[is_bid]
        bq = q_i[is_bid]
        border = np.argsort(-bpx) if bpx.size else np.array([], dtype=int)
        # offer prices/qtys, sorted best-first (lowest price first)
        opx = px_i[is_off]
        oq = q_i[is_off]
        oorder = np.argsort(opx) if opx.size else np.array([], dtype=int)
        # store the pre-extracted, best-first book for this message
        books[uniq_seq[i]] = {
            "bid_q": bq[border], "bid_px": bpx[border],
            "off_q": oq[oorder], "off_px": opx[oorder],
            "has_agg_bid": bool((et_i == "AGG_BID").any()),
            "has_agg_off": bool((et_i == "AGG_OFFER").any()),
            "bb": bpx[border][0] if bpx.size else np.nan,
            "ba": opx[oorder][0] if opx.size else np.nan}
    # the index: time-ordered times + seqs, and the per-seq book dict
    return {"times": time_order, "seqs": uniq_seq, "books": books}


# ---- O(log n) book lookup at/just before t_ms on the hit side ----
def book_at(index, t_ms, hit_side):
    # null index -> empty book
    if index is None or len(index["times"]) == 0:
        return [], [], False, np.nan, np.nan
    # binary-search the sorted message times for the last message <= t_ms
    pos = np.searchsorted(index["times"], t_ms, side="right") - 1
    # no message at/before this time -> empty book
    if pos < 0:
        return [], [], False, np.nan, np.nan
    # the message sequence at that position, and its pre-extracted book
    b = index["books"][index["seqs"][pos]]
    # the aggressor HITS offers (BUY) or bids (SELL)
    if hit_side == "BUY":
        # BUY hits the OFFER side
        return (list(b["off_q"]), list(b["off_px"]), b["has_agg_off"],
                b["bb"], b["ba"])
    # SELL hits the BID side
    return (list(b["bid_q"]), list(b["bid_px"]), b["has_agg_bid"],
            b["bb"], b["ba"])


# ---- map a fill price to its book LEVEL index (distance from touch) ----
# given the hit-side level prices best-first, the level whose price equals the
# fill price is its depth. A fill at the touch = L1; one level in = L2; beyond
# the 10 explicit levels (into AGG) = L10+. Uses a small tick tolerance.
def price_level_bucket(fill_px, level_pxs, has_agg, tick=1e-6):
    # no book -> unknown, treat as L1 (conservative)
    if not level_pxs:
        return 1
    # find the level whose price matches the fill (within tolerance)
    for i, lp in enumerate(level_pxs):
        # matched this explicit level -> L(i+1), capped at L10+
        if abs(float(lp) - float(fill_px)) <= tick:
            return min(i + 1, MAX_LEVEL + 1)
    # fill price is past all explicit levels -> it reached the AGG block (L10+)
    # (only if the book actually has an aggregate block; else cap at last level)
    return (MAX_LEVEL + 1) if has_agg else min(len(level_pxs), MAX_LEVEL + 1)


# ---- analyse one symbol-day: return per-aggressor sweep-depth records ----
def analyse_symbol_day(date, sym, dsets):
    # read this symbol's trades + snapshots
    t = R.read_symbol(dsets["trades"], REQ_TRADES_FULL, sym)
    # read this symbol's L10 snapshots
    s = R.read_symbol(dsets["ob_snapshot"], REQ_SNAP_FULL, sym)
    # nothing to do without trades or a book
    if len(t) == 0 or len(s) == 0:
        return []
    # continuous-phase trades only (ignore auction prints)
    t = t[t["exec_type"].astype(str).str.upper().str.contains("TRADE", na=False)]
    # no trades -> nothing to analyse
    if len(t) == 0:
        return []
    # work on a copy
    t = t.copy()
    # normalise the aggressor side to BUY/SELL (drop AUCTION rows)
    t["hit_side"] = np.where(
        t["aggressor_side"].astype(str).str.upper().str.startswith("B"),
        "BUY",
        np.where(t["aggressor_side"].astype(str).str.upper().str.startswith("S"),
                 "SELL", "OTHER"))
    # keep only real buy/sell-initiated trades (no auction prints)
    t = t[t["hit_side"].isin(["BUY", "SELL"])]
    # nothing left -> done
    if len(t) == 0:
        return []
    # trade time in ms (the clustering key together with the side)
    t["t_ms"] = R.to_ms(t["transact_time"])
    # ---- build the snapshot index ONCE for this symbol-day (the speed fix) ----
    index = build_snapshot_index(s)
    # records for this symbol-day
    recs = []
    # ---- SAME-TIMESTAMP + SAME-SIDE clustering: one marketable order filling
    # against multiple resting orders is stamped at one exchange instant ----
    for (t_ms, hit_side), g in t.groupby(["t_ms", "hit_side"], sort=False):
        # per-fill prices and quantities as arrays (avoid per-row iterrows)
        pxs = g["price"].to_numpy(dtype=float)
        qtys = g["qty"].to_numpy(dtype=float)
        # total swept quantity across this instant's same-side fills
        swept_qty = float(qtys.sum())
        # skip empty/zero
        if swept_qty <= 0:
            continue
        # M1 (trade-distinct): number of DISTINCT trade prices in the cluster
        m1_bucket = min(int(np.unique(pxs).size), MAX_LEVEL + 1)
        # the resting book on the hit side, O(log n) via the pre-built index
        level_qtys, level_pxs, has_agg, best_bid, best_ask = book_at(
            index, float(t_ms), hit_side)
        # M2 (book-diff): how many resting levels the swept quantity consumed
        m2_bucket = levels_consumed(swept_qty, level_qtys, has_agg)
        # the touch on the side being hit: BUY hits the ASK, SELL hits the BID
        touch = best_ask if hit_side == "BUY" else best_bid
        # touch as a float or NaN, computed once for the cluster
        touch_f = float(touch) if pd.notna(touch) else np.nan
        # per-fill distance-from-touch bucket, vectorised over the cluster's fills
        for k in range(len(pxs)):
            # the level index of this fill's price = its distance-from-touch bucket
            dist_bucket = price_level_bucket(float(pxs[k]), level_pxs, has_agg)
            # record this fill (cluster-level M1/M2 + per-fill distance bucket)
            recs.append({"date": date, "symbol": sym, "hit_side": hit_side,
                         "t_ms": float(t_ms), "cluster_qty": swept_qty,
                         "fill_qty": float(qtys[k]), "fill_px": float(pxs[k]),
                         "touch": touch_f,
                         "m1_bucket": m1_bucket, "m2_bucket": m2_bucket,
                         "dist_bucket": dist_bucket})
    # per-symbol-day records
    return recs


# ---- turn records into a level histogram (count + volume weighted) ----
def histogram(recs, method):
    # the bucket column for this method (m1_bucket / m2_bucket / dist_bucket)
    col = f"{method}_bucket"
    # frame of per-fill records
    df = pd.DataFrame(recs)
    # empty guard
    if len(df) == 0:
        return None
    # bucket labels L1..L10, L10+
    labels = [f"L{i}" for i in range(1, MAX_LEVEL + 1)] + [f"L{MAX_LEVEL}+"]
    # COUNT weighting differs by method:
    #  - dist_bucket is per-FILL -> count individual fills
    #  - m1/m2 are per-CLUSTER (repeated across the cluster's fill rows) -> dedupe
    #    to one row per cluster (t_ms + hit_side) so clusters aren't over-counted
    if method == "dist":
        # per-fill: each fill is one observation
        cnt = df.groupby(col).size()
        # volume: sum the fill quantities
        vol = df.groupby(col)["fill_qty"].sum()
    else:
        # per-cluster: dedupe to one row per (t_ms, hit_side) cluster
        clusters = df.drop_duplicates(subset=["t_ms", "hit_side"])
        # count clusters per bucket
        cnt = clusters.groupby(col).size()
        # volume: the cluster's total swept quantity, once per cluster
        vol = clusters.groupby(col)["cluster_qty"].sum()
    # assemble a tidy table indexed by bucket 1..MAX_LEVEL+1
    out = pd.DataFrame(index=range(1, MAX_LEVEL + 2))
    # counts (0 where absent)
    out["count"] = cnt.reindex(out.index, fill_value=0).values
    # volume (0 where absent)
    out["volume"] = vol.reindex(out.index, fill_value=0.0).values
    # % of orders/fills for the count histogram
    out["pct_count"] = 100.0 * out["count"] / out["count"].sum()
    # % of volume for the volume histogram
    out["pct_volume"] = 100.0 * out["volume"] / out["volume"].sum()
    # human-readable labels
    out["level"] = labels
    # (tidy per-bucket table)
    return out


# ---- plot spot-vs-futures histograms for one ticker + method + weighting ----
def plot_hist(tbl_spot, tbl_fut, sym, method, weight, stamp):
    # y column for the chosen weighting
    ycol = "pct_count" if weight == "count" else "pct_volume"
    # x positions for the buckets
    x = np.arange(MAX_LEVEL + 1)
    # bar width (two series side by side)
    w = 0.4
    # a new figure
    fig, ax = plt.subplots(figsize=(10, 5))
    # spot bars (left offset) if present
    if tbl_spot is not None:
        ax.bar(x - w / 2, tbl_spot[ycol].values, width=w, label="spot",
               color="#3b6")
    # futures bars (right offset) if present
    if tbl_fut is not None:
        ax.bar(x + w / 2, tbl_fut[ycol].values, width=w, label="futures",
               color="#c53")
    # x tick labels are the level buckets
    ax.set_xticks(x)
    # x labels L1..L10 then L10+
    ax.set_xticklabels([f"L{i}" for i in range(1, MAX_LEVEL + 1)] +
                       [f"L{MAX_LEVEL}+"])
    # axis labels + title
    ax.set_xlabel("book levels consumed by the aggressor order")
    # y label depends on weighting
    ax.set_ylabel(f"% of {'orders' if weight == 'count' else 'volume'}")
    # chart title
    ax.set_title(f"{sym}: sweep depth ({method}, {weight}-weighted)")
    # legend + grid
    ax.legend()
    # faint horizontal gridlines
    ax.grid(axis="y", alpha=0.3)
    # tidy layout
    fig.tight_layout()
    # output path
    p = RESULTS / f"sweepFAST_{sym}_{method}_{weight}_{stamp}.png"
    # save + close
    fig.savefig(p, dpi=120)
    # free the figure
    plt.close(fig)
    # (path for the caller to report)
    return p


# the main entry point
def main():
    # run stamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # all trading dates
    all_dates = R.discover_dates()
    # the dates to analyse
    dates = DATES if DATES else [str(d) for d in all_dates[:N_SAMPLE]]
    # roll map for futures (active contract per root/date)
    print("building futures roll map...", flush=True)
    # futures expiry/calendar
    cal = F.load_futures_calendar()
    # active contract per root/date
    roll = F.build_roll_map(cal, all_dates)
    # accumulators: per-symbol records for spot and futures
    spot_recs = {sym: [] for sym in SPOT_SYMS}
    # per-futures-root sweep records
    fut_recs = {root: [] for root in FUT_ROOTS}
    # self-check accumulator: rows per aggressor ref (to confirm grouping works)
    rows_per_agg = []
    # timer
    t0 = time.perf_counter()
    # walk the sampled dates
    for i, date in enumerate(dates, 1):
        # open the day's datasets
        dsets = R.open_datasets(date)
        # skip missing days
        if dsets is None:
            continue
        # spot symbols
        for sym in SPOT_SYMS:
            # per-aggressor sweep records
            recs = analyse_symbol_day(date, sym, dsets)
            # accumulate
            spot_recs[sym].extend(recs)
            # self-check: mean fills per aggressor (sweeps should sometimes >1)
            if recs:
                # cluster size = # of per-fill records sharing (t_ms,hit_side)
                _cl = {}
                for r in recs:
                    _cl[(r["t_ms"], r["hit_side"])] = _cl.get(
                        (r["t_ms"], r["hit_side"]), 0) + 1
                rows_per_agg.extend(list(_cl.values()))
        # futures roots (map to the active contract)
        for root in FUT_ROOTS:
            # the active contract on this date
            fsym = roll.get((root, date))
            # skip if no active contract
            if fsym is None:
                continue
            # per-aggressor sweep records for the futures contract
            recs = analyse_symbol_day(date, fsym, dsets)
            # relabel the symbol to the ROOT for aggregation across the roll
            for r in recs:
                r["symbol"] = root
            # accumulate this contract's records under the root
            fut_recs[root].extend(recs)
            # only tally the self-check if there were records
            if recs:
                # cluster size = # of per-fill records sharing (t_ms,hit_side)
                _cl = {}
                for r in recs:
                    _cl[(r["t_ms"], r["hit_side"])] = _cl.get(
                        (r["t_ms"], r["hit_side"]), 0) + 1
                rows_per_agg.extend(list(_cl.values()))
        # heartbeat
        if i % 5 == 0 or i == len(dates):
            print(f"  {i}/{len(dates)} days  {C._fmt(time.perf_counter()-t0)}",
                  flush=True)

    # ---- SELF-CHECK: same-timestamp clustering (fills per cluster) ----
    rpa = np.array(rows_per_agg) if rows_per_agg else np.array([0])
    # self-check header
    print("\n=== SELF-CHECK: same-timestamp sweep clustering ===")
    # how many clusters (candidate aggressor orders)
    print(f"  same-timestamp clusters analysed: {len(rpa):,}")
    # fills per cluster: mean should be modest (a few), NOT thousands
    print(f"  fills per cluster: mean {rpa.mean():.2f}  median {np.median(rpa):.0f}"
          f"  max {rpa.max():.0f}  p95 {np.quantile(rpa,0.95):.0f}")
    # % of clusters that touched >1 resting order (real multi-level sweeps)
    print(f"  multi-fill clusters (>1 fill): {100.0*(rpa>1).mean():.1f}%")
    # sanity bounds: mean fills per cluster should be small (say < 50). If it is
    # thousands, clustering is over-merging; if all are 1, no sweeps are captured.
    if rpa.mean() > 50:
        print("  !! WARNING: mean fills/cluster is very high -- same-timestamp")
        print("     merging may be lumping distinct orders. Inspect the data.")
    elif (rpa > 1).mean() < 0.005:
        print("  !! WARNING: almost no multi-fill clusters -- sweeps rarely share")
        print("     a timestamp here; M2 book-diff still valid, M1 understates.")
    else:
        print("  -> clustering looks sane (modest fills/cluster, sweeps present).")

    # ---- build histograms + plots per ticker ----
    all_rows = []
    # section header
    print("\n=== SWEEP DEPTH by ticker (M2 book-diff, the authoritative method) ===")
    # the union of tickers (spot and/or futures)
    tickers = sorted(set(SPOT_SYMS) | set(FUT_ROOTS))
    # each ticker (spot and/or futures root)
    for sym in tickers:
        # spot + futures histograms for all three methods (m1, m2, dist)
        for method in ("m1", "m2", "dist"):
            ts = histogram(spot_recs.get(sym, []), method) if sym in spot_recs else None
            # futures histogram
            tf = histogram(fut_recs.get(sym, []), method) if sym in fut_recs else None
            # plot both weightings
            for weight in ("count", "volume"):
                if ts is not None or tf is not None:
                    plot_hist(ts, tf, sym, method, weight, stamp)
            # collect tidy rows for the CSV (M2 only in the printed summary)
            for label, tbl, mkt in (("spot", ts, "spot"), ("fut", tf, "futures")):
                if tbl is not None:
                    for _, row in tbl.iterrows():
                        all_rows.append({"ticker": sym, "market": mkt,
                                         "method": method, "level": row["level"],
                                         "count": int(row["count"]),
                                         "volume": row["volume"],
                                         "pct_count": round(row["pct_count"], 2),
                                         "pct_volume": round(row["pct_volume"], 2)})
        # print the M2 volume-weighted comparison for this ticker
        ts2 = histogram(spot_recs.get(sym, []), "m2") if sym in spot_recs else None
        # futures M2 histogram for the summary
        tf2 = histogram(fut_recs.get(sym, []), "m2") if sym in fut_recs else None
        # deep-sweep share = % of volume at L3+ (a compact institutional-flow read)
        def deep_share(tbl):
            # None -> n/a
            if tbl is None:
                return np.nan
            # % of volume in buckets L3..L10+
            return float(tbl[tbl.index >= 3]["pct_volume"].sum())
        # M2 cluster-depth: % of VOLUME via clusters consuming L3+ levels
        print(f"  {sym:>6s}: [M2 sweep] L3+ vol share  spot {deep_share(ts2):>5.1f}%"
              f"   futures {deep_share(tf2):>5.1f}%")
        # dist-from-touch: % of FILLS that printed AWAY from the touch (L2+),
        # i.e. did not get the best price -> a per-fill aggression read
        ds_spot = histogram(spot_recs.get(sym, []), "dist") if sym in spot_recs else None
        ds_fut = histogram(fut_recs.get(sym, []), "dist") if sym in fut_recs else None
        # % of fills at L2+ (away from touch) = 100 - L1 share
        def away_share(tbl):
            # None -> n/a
            if tbl is None:
                return np.nan
            # % of fills (count-weighted) in buckets L2..L10+
            return float(tbl[tbl.index >= 2]["pct_count"].sum())
        # print the away-from-touch share, spot vs futures
        print(f"         [dist] fills away from touch (L2+)  "
              f"spot {away_share(ds_spot):>5.1f}%   futures {away_share(ds_fut):>5.1f}%")
    # interpretation line
    print("\n  higher L3+ sweep share OR more fills away-from-touch in futures =")
    print("  deeper/more aggressive flow -> supports 'futures flow more toxic'.")

    # ---- write the tidy CSV ----
    out = pd.DataFrame(all_rows)
    # write the tidy CSV
    out.to_csv(RESULTS / f"sweepFAST_{stamp}.csv", index=False)
    print(f"\nwrote {RESULTS / f'sweepFAST_{stamp}.csv'}")
    print(f"wrote PNGs: sweepFAST_<ticker>_<m1|m2|dist>_<count|volume>_{stamp}.png")
    # explicit total runtime for the parallel timing test
    print(f"\n### FAST-VERSION TOTAL RUNTIME: {C._fmt(time.perf_counter()-t0)} "
          f"for {len(dates)} days ###")


# entry point
if __name__ == "__main__":
    main()
