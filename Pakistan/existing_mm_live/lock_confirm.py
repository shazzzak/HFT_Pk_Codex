# Share the configured data and current checkout roots; never fall back to a legacy tree.
import config_pk as _hft_paths
# lock_confirm.py -- are the one-sided-close days actually LOCKS (price pinned at
# the +/-10% band) and, if so, is the loss concentrated in the TRAPPED-sign subset
# (short into limit-up / long into limit-down, where there's no counterparty)?
#
# Method: from the last run's eod_positions.csv, take the one-sided-close days
# (mid_is_none == True) for one config, with their pos_at_close. For each such
# (symbol, date), replay snapshots to the bell (like close_thinning) and read the
# closing book's present side, its price vs limit_up/limit_dn, and the phase string.
# Join with position sign and that day's liquidated P&L. No strategy run needed.
#
# Run from existing_mm_live/:  python lock_confirm.py

# filesystem paths
from pathlib import Path
# wall-clock timing for the heartbeat
import time
# dataframes
import pandas as pd
# numeric helpers
import numpy as np
# driver module (dataset discovery, per-symbol reads, event building)
import run_legacy_mm as R
# the order-book object we drive with snapshots
from mm_backtest import Book

# point the driver at the parsed data root on this machine
# Resolve this filesystem path through the canonical checkout/data configuration.
R.PARSED_ROOT = Path(str(_hft_paths.PARSED_ROOT))
# results directory holding the per-day eod_positions.csv
# Resolve this filesystem path through the canonical checkout/data configuration.
RES = Path(str(_hft_paths.RESULTS_ROOT))
# which config's pos_at_close to use (mid_is_none is a market fact, same across configs)
CONFIG_LABEL = "MID+band150 me=0.0005"
# how close (in ticks) the present side must sit to a limit to call it "at the lock"
LOCK_TOL_TICKS = 2
# the price tick
TICK = 0.01


# format seconds as compact mm:ss
def _fmt(sec):
    # minutes and zero-padded seconds
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


# replay snapshots up to the bell; return the closing book's key state
def close_state(events, snap_groups, t1):
    # fresh empty book
    book = Book()
    # last book state captured at/under the bell
    state = None
    # walk events in time order
    for ts_exch, _, _, kind, obj in events:
        # only snapshots change book state
        if kind == "S":
            # apply the snapshot (full replacement)
            book.snapshot(snap_groups[obj.msg_seq])
            # only capture state up to and including the bell
            if ts_exch <= t1:
                # best bid/ask + depths
                bb, bq, ba, aq = book.bbo()
                # read the daily band + phase defensively (feed may not populate them)
                state = {
                    # best bid price (None if bid side empty)
                    "bb": bb,
                    # best ask price (None if ask side empty)
                    "ba": ba,
                    # upper price limit for the day (+10% band), if present
                    "limit_up": getattr(book, "limit_up", None),
                    # lower price limit for the day (-10% band), if present
                    "limit_dn": getattr(book, "limit_dn", None),
                    # market phase string, if present
                    "phase": getattr(book, "phase", None),
                }
    # the last captured state = the book at the close
    return state


def main():
    # load the per-day records from the last run
    eod = pd.read_csv(RES / "eod_positions.csv")
    # guard: this needs the enriched columns
    need = {"symbol", "variant", "date", "eod_pos", "liquidated", "mid_is_none"}
    missing = need - set(eod.columns)
    # bail clearly if the CSV predates the enrichment
    if missing:
        raise SystemExit(f"eod_positions.csv missing {missing} -- re-run the updated confirm first.")
    # normalise the one-sided flag to bool
    eod["mid_is_none"] = eod["mid_is_none"].astype(str).str.lower().isin(["true", "1", "1.0"])
    # keep the chosen config's one-sided-close days only
    tgt = eod[(eod["variant"] == CONFIG_LABEL) & (eod["mid_is_none"])].copy()
    # index them by (symbol, date) for a fast membership test and pos/pnl lookup
    tgt["date"] = tgt["date"].astype(str)
    # a dict (symbol, date) -> (pos_at_close, liquidated)
    want = {(r.symbol, r.date): (r.eod_pos, r.liquidated) for r in tgt.itertuples()}
    # announce the target set size
    print(f"lock_confirm: {len(want)} one-sided-close days for '{CONFIG_LABEL}'\n", flush=True)

    # the set of dates we actually need to open (a subset of all dates)
    want_dates = {d for (_, d) in want}
    # all available dates
    dates = R.discover_dates()
    # collected per-day classification rows
    rows = []
    # distinct entry_type codes seen (so if limit_up/dn are None we can see which
    # entry_type carries the lock, without a second run)
    entry_types_seen = set()
    # timer + counter
    t0_all = time.perf_counter()
    done = 0
    # OUTER loop over dates, skipping any we don't need
    for date in dates:
        # skip dates with no target day
        if str(date) not in want_dates:
            continue
        # open datasets once
        dsets = R.open_datasets(date)
        # skip if missing
        if dsets is None:
            continue
        # MIDDLE loop over the two symbols
        for sym in ("PPL", "UBL"):
            # only process the exact (symbol, date) targets
            if (sym, str(date)) not in want:
                continue
            # read the three tables
            u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym)
            s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym)
            t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym)
            # skip empties
            if len(t) == 0 or len(s) == 0:
                continue
            # record the distinct entry_type codes present in this day's snapshot
            if "entry_type" in s.columns:
                entry_types_seen.update(s["entry_type"].dropna().unique().tolist())
            # build the event stream + snapshot groups
            events, snap_groups, t = R.build_events(u, s, t)
            # continuous-session end = the bell
            cont = t[t["initiator"] != "AUCTION"]
            # skip days with no continuous session
            if len(cont) == 0:
                continue
            # bell timestamp
            t1 = int(cont["ts_exch"].max())
            # reconstruct the closing book state
            st = close_state(events, snap_groups, t1)
            # skip if nothing captured
            if st is None:
                continue
            # look up this day's position + P&L
            pos, pnl = want[(sym, str(date))]
            # which side is present at the close?
            bid_only = (st["bb"] is not None and st["ba"] is None)
            ask_only = (st["ba"] is not None and st["bb"] is None)
            # present-side label
            side = "bid_only" if bid_only else ("ask_only" if ask_only else "other")
            # distance (in ticks) of the present side from the RELEVANT limit
            gap_ticks = np.nan
            # for bid-only, compare best bid to the upper limit (limit-up lock)
            if bid_only and st["limit_up"] is not None and st["bb"] is not None:
                gap_ticks = round((st["limit_up"] - st["bb"]) / TICK, 1)
            # for ask-only, compare best ask to the lower limit (limit-down lock)
            if ask_only and st["limit_dn"] is not None and st["ba"] is not None:
                gap_ticks = round((st["ba"] - st["limit_dn"]) / TICK, 1)
            # is the present side sitting at the lock (within tolerance)?
            at_lock = (not np.isnan(gap_ticks)) and abs(gap_ticks) <= LOCK_TOL_TICKS
            # position sign (nan if pos was None)
            pos_sign = (np.sign(pos) if pd.notna(pos) else np.nan)
            # TRAPPED = short into a limit-up (bid_only) or long into a limit-down (ask_only)
            trapped = ((bid_only and pos_sign < 0) or (ask_only and pos_sign > 0))
            # record the classification
            rows.append({
                "symbol": sym, "date": str(date), "side": side,
                "gap_ticks": gap_ticks, "at_lock": at_lock,
                "phase": st["phase"], "pos": pos, "trapped": bool(trapped),
                "liquidated": pnl,
            })
            # progress tick
            done += 1
            if done % 10 == 0:
                # elapsed time
                el = time.perf_counter() - t0_all
                # print progress
                print(f"  {done}/{len(want)} days  elapsed {_fmt(el)}", flush=True)

    # assemble
    df = pd.DataFrame(rows)
    # nothing found guard
    if len(df) == 0:
        print("no days processed -- check dates match eod_positions.csv"); return

    # ---- report ----
    # full-width printing
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    # per-day detail
    print("\n--- per one-sided-close day ---")
    print(df.to_string(index=False))

    # summary per symbol
    print("\n--- summary ---")
    for sym, g in df.groupby("symbol"):
        # counts
        n = len(g)
        n_lock = int(g["at_lock"].sum())
        n_bid = int((g["side"] == "bid_only").sum())
        n_ask = int((g["side"] == "ask_only").sum())
        n_trap = int(g["trapped"].sum())
        # P&L split by trapped vs safe
        pnl_trap = g.loc[g["trapped"], "liquidated"].sum()
        pnl_safe = g.loc[~g["trapped"], "liquidated"].sum()
        # phases seen
        phases = sorted(set(str(p) for p in g["phase"].dropna().unique()))
        # print the block
        print(f"\n{sym}: {n} one-sided-close days")
        print(f"  at the lock (|gap|<= {LOCK_TOL_TICKS} ticks): {n_lock}/{n}")
        print(f"  bid_only (limit-up cand.): {n_bid}   ask_only (limit-dn cand.): {n_ask}")
        print(f"  TRAPPED sign (short-into-up / long-into-dn): {n_trap}/{n}")
        print(f"  liquidated P&L  trapped={pnl_trap:,.0f}   safe={pnl_safe:,.0f}")
        print(f"  phase strings seen: {phases if phases else 'none/None'}")

    # show which entry_type codes exist (the lock rides on one of these if
    # limit_up/limit_dn came back None above)
    print(f"\nentry_type codes seen in snapshots: {sorted(str(e) for e in entry_types_seen)}")

    # the decision text
    print("\nDECIDES:")
    print("  If most days are 'at the lock' -> the one-sided closes ARE the +/-10%")
    print("  band, confirming the lock hypothesis. If the loss (liquidated) is")
    print("  concentrated in the TRAPPED subset, the fix is an intraday distance-to-")
    print("  limit guard that stops us reaching the trapped side. If 'at_lock' is low")
    print("  or limit_up/limit_dn came back None, it's NOT locks (or the feed lacks")
    print("  the fields) and we rethink before building the guard.")


if __name__ == "__main__":
    main()
