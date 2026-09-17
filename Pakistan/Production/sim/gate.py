# ============================================================================
# sim/gate.py -- does the production engine reproduce the backtest?
# ============================================================================
# THE QUESTION THIS ANSWERS, AND WHY NOTHING SHIPS BEFORE IT PASSES.
#
# Every number this project has produced came out of mm_backtest. The
# production engine is a different pile of code: a strategy adapter, an order
# manager, a risk gateway, an audit log. If it does not put the same orders in
# the same places at the same times, then the measured edge belongs to a
# program we are not going to run, and every decision taken on those numbers is
# a decision about something else.
#
# So: same symbol-days, same strategy parameters, same latency seed, same
# exchange. One side driven by mm_backtest's own _requote, the other by the
# production order manager through sim/replay.py. Compare the P&L.
#
# THE TARGET IS ZERO DIFFERENCE, NOT A TOLERANCE. Both sides run against the
# SAME Backtester exchange with the SAME seeded latency model, so if the
# message sequence matches, every latency draw matches and every fill matches.
# A difference is therefore a named bug, not noise to be averaged away. If a
# tolerance ever appears in this file, something has been given up.
#
# TWO MECHANICS, BOTH RUNNABLE (--mode).
#
#   replace     A reprice is ONE Order Cancel/Replace Request (MsgType 'G').
#               It carries the exchange's id for the order already resting, so
#               it goes out immediately -- there is no cancel to wait for and
#               no window in which two orders are live. Both sides use it:
#               the engine through use_replace=True, mm_backtest through
#               use_cfo=True.
#
#               WHAT 'G' DOES TO QUEUE POSITION is the venue's rule, not a
#               choice, and both sides read it from the same place --
#               PSXVenue's three replace_*_keeps_priority answers, which
#               sim/replay.py wires onto the engine. PSX Regulations 8.5.2: a
#               price change or a size INCREASE goes to the back of the queue
#               at the new level; a size REDUCTION is amended in place and
#               keeps its position.
#
# TWO REQUOTE POLICIES, BOTH RUN, EVERY TIME. Each symbol-day is run three
# times: mm_backtest once, and the production engine twice.
#
#   top up   Restore the full clip whenever the size resting is not the size
#            wanted -- including after a partial fill. This is mm_backtest's
#            rule, so it is what every measured number was produced under, and
#            it is the run the PASS/FAIL gate judges.
#
#   hold     Leave the remainder resting and keep the place in the queue it
#            has already earned.
#
# PSX Regulations 8.5.2 applies identically to both: an amendment that raises
# the size or changes the price goes to the BACK of the queue at that price;
# an amendment that REDUCES the size is applied in place and keeps its
# position. So topping up after a partial fill is not free -- the exchange
# charges for it in priority every single time -- while reducing is. Neither
# policy is a simulation artefact and neither is more "real" than the other;
# the difference is a choice, and this run puts a number on it.
#
#   cancel_new  A reprice is a cancel plus a new order.
#
#               CORRECTED 2026-09-17. This note used to say the two sides
#               differ by design here, because mm_backtest fired both messages
#               in the same cycle while the order manager waited for the
#               cancel to be acknowledged. mm_backtest firing both at once was
#               a DEFECT, not a design: the cancel and the replacement drew
#               independent latencies, so whenever the replacement won it
#               landed on top of an order that was still resting and dropped
#               it on the floor -- never cancelled, never filled, never closed
#               out. 5,205 of them over four symbol-days.
#
#               Both sides now wait for the cancel to land, so this mode is
#               expected to reconcile like the other one. If it does not, that
#               is a finding rather than an explanation.
#
# READ-ONLY on every input. Writes ONE timestamped CSV. Never overwrites.
#
# Run from Production/ with existing_mm_live on the path:
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/gate.py --smoke
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/gate.py
#   PYTHONPATH=../existing_mm_live caffeinate -is python sim/gate.py --mode cancel_new
# ============================================================================

# command-line flags
import argparse
# path handling, so Production/ is importable when run as a script
import sys
from pathlib import Path

# make the package importable however this file was invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# frames
import pandas as pd

# ---- the research stack, which must be on PYTHONPATH ----------------------
# These live in existing_mm_live and are the thing being reproduced. Failing
# here with a clear message beats an ImportError three frames deep.
try:
    # the shared harness: calibration loaders and the canonical run path
    import mm_harness as H
    # the driver module: dataset opening, date discovery, the canonical config
    import run_legacy_mm as R
    # the shared name list and the never-overwrite output helper
    import expansion_names as EX
    # the exchange, and the seeded latency model both sides share
    from mm_backtest import Backtester, LatencyModel
    # the strategy itself
    from micro_mm import MicrostructureMM
except Exception as _e:                                       # noqa: BLE001
    raise SystemExit(
        "sim/gate.py needs the research stack on PYTHONPATH. Run it as:\n"
        "  PYTHONPATH=../existing_mm_live python sim/gate.py --smoke\n"
        f"Original error: {_e!r}")

# ---- the production engine -------------------------------------------------
# the order manager under test
from core.oms import OrderManager
# the kill switch, the gateway, and the one check this run applies
from core.risk import KillSwitch, OrderQuantityCheck, RiskGateway
# the quote tolerance, which is how the engine is told to match mm_backtest
from core.oms import QuoteTolerance

# THE GATE SAMPLE. Ranked by measured net P&L from the 113-name run
# (universe_expand_PERSYMBOL_113_CORRECTED_20260913.csv), best first.
#
# NOT THE ALPHABET. The first version took sorted(ALL_NAMES)[:n], which gave
# AGHA and AGP -- AGHA makes 19,709 PKR over 196 days on 20 fills a day. A
# gate that exercises the order manager twenty times a day proves nothing
# about a system that has to survive three hundred.
#
# These names are both PROFITABLE and ACTIVE: NRL is first by P&L (556,638)
# and fourth by fill rate (281 a day); MLCF is third by P&L (446,410) at 254
# fills a day with a t-statistic of 13.4. If the engine reproduces the
# backtest on these, it reproduces it where it matters.
GATE_NAMES = ["NRL", "MLCF", "ENGROH", "NBP", "LUCK", "NPL",
              "SEARL", "NML", "SYS", "PPL", "THCCL", "NCPL"]
# the session segment the venue hands out
from core.venue import SessionSegment
# the venue, which owns the tick grid and the three amendment rules
from venues.psx import PSXVenue
# the bridge from micro_mm to the order manager
from venues.psx_strategy import MicroMMAdapter
# the simulated exchange: Backtester with the production engine driving it
from sim.replay import EngineReplay


def load_symbol_day(dsets, sym):
    """The day's events for one symbol, built ONCE and shared by both runs.

    Built once on purpose. R.build_events adds columns to the snapshot frame in
    place, and two runs that each built their own event stream would be two
    runs against two subtly different days -- which is exactly the kind of
    difference this file exists to rule out.
    """
    # THE REGULAR MARKET ONLY, added 2026-09-17.
    #
    # A symbol can be listed in more than one PSX market under the same
    # ticker: MLCF carries 69 EQ_SQUARE_UP rows beside 272,894 REG rows, and
    # a square-up SNAPSHOT replaces that symbol's whole book -- square-up best
    # bid 122.49 against the regular market's 95.86 ask -- until the next
    # regular snapshot arrives. mm_harness.run_symbol_day was fixed for this;
    # this file was written before that fix and never inherited it, so the
    # gate was comparing two runs over a day the backtest would not have seen.
    #
    # read_symbol applies the filter only where the column list carries
    # `market`, which today is REQ_SNAP alone, so passing it to all three is
    # safe and keeps this identical to what mm_harness does. Identical is the
    # requirement: the gate's job is to reproduce the backtest, so it must
    # read the same day the backtest reads, not a better one.
    #
    # the day's order-book updates for this symbol
    u = R.read_symbol(dsets["ob_updates"], R.REQ_UPDATES, sym, market="REG")
    # the day's book snapshots
    s = R.read_symbol(dsets["ob_snapshot"], R.REQ_SNAP, sym, market="REG")
    # the day's trades
    t = R.read_symbol(dsets["trades"], R.REQ_TRADES, sym, market="REG")
    # unrunnable without both a book and trades
    if len(t) == 0 or len(s) == 0:
        return None
    # the merged event stream
    events, snap_groups, t = R.build_events(u, s, t)
    # the continuous-trading snapshots define the session window
    cont = s[s["phase"] == "CONTINUOUS_AUCTION"]
    # no continuous phase means a fully halted or no-data day
    if len(cont) == 0:
        return None
    # t0 = continuous open, t1 = continuous close
    t0, t1 = int(cont["ts_exch"].min()), int(cont["ts_exch"].max())
    # a representative price for this symbol. The adapter uses it ONLY to check
    # that the venue's tick grid and the strategy's tick agree and that the
    # grid is flat -- it is a sanity probe at construction, never a trading
    # input -- so the day's first traded price is ample.
    ref_minor = int(round(float(t["price"].iloc[0]) * 100))
    # everything both runs need
    return events, snap_groups, t0, t1, ref_minor


def run_baseline(events, snap_groups, params, t0, t1, use_g):
    """mm_backtest driving itself. This is the thing being reproduced."""
    # the canonical config, with a FRESH seeded latency model so the draw
    # sequence starts from the same place as the engine run's
    cfg = dict(R.CFG, session=(t0, t1),
               latency_model=LatencyModel(seed=R.LATENCY_SEED),
               # the mechanic under test
               use_cfo=use_g)
    # a fresh strategy: it carries state, so the two runs cannot share one
    strat = MicrostructureMM(session_ms=(t0, t1), **params)
    # the engine
    bt = Backtester(strat, cfg)
    # run it
    bt.run(events, snap_groups)
    # the whole engine, so the caller can read stats as well as P&L
    return bt


def run_engine(events, snap_groups, params, t0, t1, ref_minor, sym, date,
               use_g, quantity_policy):
    """The production order manager driving the same exchange.

    `quantity_policy` picks the REQUOTE RULE, and it is the whole point of
    running this twice:

      "exact"        TOP UP. Requote whenever the size resting is not the size
                     wanted. This is mm_backtest's rule, so it is what every
                     measured number was produced under, and it is the run the
                     gate judges. After a partial fill it restores the full
                     clip -- and under PSX 8.5.2 raising the size sends the
                     order to the BACK of the queue at that price, so the
                     exchange charges for every top-up.

      "queue_preserving"
                     NEVER GIVE UP QUEUE POSITION. After a partial fill the
                     remainder is neither cancelled nor amended upward -- the
                     shortfall goes out as a SECOND ORDER, so the shares
                     already resting keep the place they earned and only the
                     increment joins the back. When the strategy wants LESS,
                     the YOUNGEST order is shrunk first: a reduction is the one
                     amendment 8.5.2 applies in place, so it is free, and the
                     oldest order has the best position and is the last thing
                     to give up.

    Both are legal, both are things a real desk does, and the exchange applies
    the identical rule to each. The difference is ours, and this is what puts
    a number on it.
    """
    # the venue. Its session provider hands back the same window the backtest
    # uses, so the trading-window check cannot be the thing that differs.
    venue = PSXVenue(session_provider=lambda d: [
        SessionSegment(start_ms=t0, end_ms=t1)])
    # a fresh strategy, identical parameters
    strat = MicrostructureMM(session_ms=(t0, t1), **params)
    # the bridge
    adapter = MicroMMAdapter(sym, venue, strat, reference_price_minor=ref_minor)
    # A DELIBERATELY PERMISSIVE GATEWAY. mm_backtest has no risk layer at all,
    # so any check that refused an action would make this run differ for a
    # reason that has nothing to do with the order manager. The gate measures
    # reproduction; the risk checks are tested in tests/test_risk.py. The
    # rejection counter is reported anyway, so a non-zero value is visible
    # rather than silently absorbed.
    gateway = RiskGateway([OrderQuantityCheck(max_quantity=1_000_000)])
    # THE REQUOTE POLICY, per the docstring above. price_ticks=0 means any
    # price change at all triggers a requote, which is what micro_mm does and
    # what mm_backtest does; only the quantity rule differs between the two
    # runs.
    tol = QuoteTolerance(price_ticks=0, qty_ratio=0.0,
                         quantity_policy=quantity_policy)
    # the order manager under test, with the mechanic matched to the baseline
    oms = OrderManager(venue=venue, gateway=gateway, kill_switch=KillSwitch(),
                       session_id="GATE", account="GATE001",
                       tolerance=tol, use_replace=use_g)
    # the same config as the baseline, and its own fresh seeded latency model
    cfg = dict(R.CFG, session=(t0, t1),
               latency_model=LatencyModel(seed=R.LATENCY_SEED))
    # the simulated exchange with the production engine driving it
    rep = EngineReplay(strategy=strat, adapter=adapter, oms=oms, symbol=sym,
                       cfg=cfg)
    # the date the trading-window check needs
    rep.session_date = str(date)
    # run it
    rep.run(events, snap_groups)
    # the whole harness, for stats as well as P&L
    return rep


def pnl_of(engine):
    """The post-liquidation P&L, or None when the close was unusable."""
    # the end-of-day report, which may not exist on a broken day
    eod = getattr(engine, "eod", None)
    # no report means no headline number
    if not eod:
        return None
    # the number every result in this project is quoted in
    return eod.get("equity_liquidated")


def main():
    # the command line
    ap = argparse.ArgumentParser()
    # a fast shape-check before committing to the full sample
    ap.add_argument("--smoke", action="store_true",
                    help="2 names, 3 days -- checks the plumbing, not the answer")
    # which reprice mechanic both sides use
    ap.add_argument("--mode", choices=("replace", "cancel_new"),
                    default="replace",
                    help="replace = one 'G' per reprice (the gate); "
                         "cancel_new = cancel plus new order (expected to "
                         "differ; see the header)")
    # how many names and days in a full run
    ap.add_argument("--names", type=int, default=12)
    ap.add_argument("--days", type=int, default=20)
    args = ap.parse_args()
    # True when both sides use one amendment message per reprice
    use_g = (args.mode == "replace")
    # the sample
    n_names = 2 if args.smoke else args.names
    n_days = 3 if args.smoke else args.days

    print("=" * 78)
    print("RECONCILE GATE -- does the production engine reproduce the backtest?")
    print("=" * 78)
    print(f"  mode      : {args.mode}")
    # say plainly what this mode is for, every run, so nobody reads the wrong
    # one as a pass
    if use_g:
        print("              one Order Cancel/Replace per reprice, both sides.")
        print("              THE TARGET IS ZERO DIFFERENCE.")
    else:
        print("              cancel plus new order. THE TWO SIDES DIFFER BY")
        print("              DESIGN here -- mm_backtest sends both messages at")
        print("              once, the order manager waits for the cancel to be")
        print("              acknowledged. This mode MEASURES that gap. It is")
        print("              not a pass/fail gate.")

    # ---- the universe and the calendar ---------------------------------
    # every trading date in the parsed store, oldest first
    all_dates = R.discover_dates()
    # the most recent n_days
    dates = all_dates[-n_days:]
    # the curated sample, in P&L order, so --names 2 gives the two best
    names = GATE_NAMES[:n_names]
    # a name in the list that is not in the store is worth knowing about
    missing = [n for n in names if n not in EX.ALL_NAMES]
    # say so rather than silently running a shorter sample
    if missing:
        print(f"  !! not in the universe: {', '.join(missing)}")
    print(f"  sample    : {len(names)} names x {len(dates)} dates")
    print(f"  names     : {', '.join(names)}")
    print(f"  dates     : {dates[0]} .. {dates[-1]}")

    # ---- calibration, loaded once --------------------------------------
    # per-symbol inventory-skew scale
    scales = H.load_scales()
    # per-symbol volume profile
    profiles = H.load_profiles()
    # per-symbol end-of-day window
    windows = H.load_windows()
    # per-date session segments
    segments = H.load_segments()

    # one row per symbol-day
    rows = []
    # walk the calendar, opening each date's data once
    for di, date in enumerate(dates, 1):
        # the date's partitions
        dsets = R.open_datasets(date)
        # a missing partition is a skipped date, not a failure
        if dsets is None:
            print(f"  [{di}/{len(dates)}] {date} no datasets; skip")
            continue
        # the day's session segments
        segs = segments.get(str(date))
        # no segments means no calibrated session
        if segs is None:
            print(f"  [{di}/{len(dates)}] {date} no session segments; skip")
            continue
        # every name in the sample
        for sym in names:
            # a name missing calibration cannot be run
            if sym not in scales or sym not in profiles or sym not in windows:
                continue
            # the strategy parameters, assembled exactly as every runner does
            params = H.build_micro_params(50, scales[sym], profiles[sym],
                                          windows[sym], segs)
            # isolate a failure to one symbol-day rather than the whole run
            try:
                # the shared inputs
                loaded = load_symbol_day(dsets, sym)
                # an unrunnable symbol-day
                if loaded is None:
                    continue
                # unpack
                events, snap_groups, t0, t1, ref_minor = loaded
                # the thing being reproduced
                bt = run_baseline(events, snap_groups, params, t0, t1, use_g)
                # THE SAME ENGINE, TWICE, under the two requote policies.
                # topup is the one the gate judges; hold is the one that says
                # what keeping queue position is worth.
                rep = run_engine(events, snap_groups, params, t0, t1,
                                 ref_minor, sym, date, use_g,
                                 quantity_policy="exact")
                # the realistic alternative, run on the identical day
                hold = run_engine(events, snap_groups, params, t0, t1,
                                  ref_minor, sym, date, use_g,
                                  quantity_policy="queue_preserving")
            except Exception as exc:                          # noqa: BLE001
                # report and carry on: one broken day must not lose the rest
                print(f"    {sym} {date} ERROR {exc!r}")
                continue
            # the three headline numbers
            base_pnl, eng_pnl, hold_pnl = pnl_of(bt), pnl_of(rep), pnl_of(hold)
            # a broken close on any side has nothing to compare
            if base_pnl is None or eng_pnl is None or hold_pnl is None:
                continue
            # one row, with enough detail that a mismatch has a diagnosis
            rows.append({
                "symbol": sym, "date": str(date),
                # the comparison itself
                "backtest_pnl": float(base_pnl),
                "engine_pnl": float(eng_pnl),
                "diff": float(eng_pnl) - float(base_pnl),
                # THE MEASUREMENT, as distinct from the gate. Same engine,
                # same day, same exchange -- only the requote policy differs.
                "engine_hold_pnl": float(hold_pnl),
                "hold_minus_topup": float(hold_pnl) - float(eng_pnl),
                # how much churn each policy generated, which is the mechanism
                "hold_orders": hold.stats.get("n_orders_sent", 0),
                "hold_cancels": hold.stats.get("n_cancels", 0),
                "hold_fills": len(hold.fills),
                # amendments that landed, and how many kept their place. On
                # the hold policy the kept count should be HIGHER: it reduces
                # size where the top-up policy raises it.
                "hold_cfos": hold.stats.get("n_cfos", 0),
                "hold_cfos_kept": hold.stats.get("n_cfos_kept_priority", 0),
                # fills are the first place a divergence shows
                "backtest_fills": len(bt.fills),
                "engine_fills": len(rep.fills),
                # then message counts
                "backtest_orders": bt.stats.get("n_orders_sent", 0),
                "engine_orders": rep.stats.get("n_orders_sent", 0),
                "backtest_cancels": bt.stats.get("n_cancels", 0),
                "engine_cancels": rep.stats.get("n_cancels", 0),
                # amendments that landed, and how many kept their place
                "backtest_cfos": bt.stats.get("n_cfos", 0),
                "engine_cfos": rep.stats.get("n_cfos", 0),
                "backtest_cfos_kept": bt.stats.get("n_cfos_kept_priority", 0),
                "engine_cfos_kept": rep.stats.get("n_cfos_kept_priority", 0),
                # THINGS THAT MUST BE ZERO. A rejection means the risk gateway,
                # not the order manager, changed the run.
                "gateway_rejections": rep.engine_stats.get(
                    "gateway_rejections", 0),
                # books the production stack refused to quote against
                "skipped_crossed_book": rep.engine_stats.get(
                    "skipped_crossed_book", 0),
                # ---- THE STAND-DOWNS, SIDE BY SIDE -----------------------
                # ADDED after the first failing run, which showed the two
                # sides sending 1,822 and 359 new orders against near-equal
                # cancel counts -- they disagree about when a side is EMPTY,
                # and these are the counters that say why. Every one of them
                # is the same key on the same Backtester stats dict, so like
                # is compared with like.
                #
                # A halt, a crossed book or a stale feed all CANCEL
                # everything, which empties the side, which makes the next
                # quotable cycle send a NEW order rather than an amendment.
                # If one side stands down far more often than the other, that
                # is the whole explanation and it is visible here.
                "backtest_halted": bt.stats.get("halted_requotes", 0),
                "engine_halted": rep.stats.get("halted_requotes", 0),
                "backtest_crossed": bt.stats.get("crossed_book_requotes", 0),
                "engine_crossed": rep.stats.get("crossed_book_requotes", 0),
                "backtest_stale": bt.stats.get("stale_feed_requotes", 0),
                "engine_stale": rep.stats.get("stale_feed_requotes", 0),
                # one-sided books, which the engine counts and the backtest
                # does not -- there the strategy declines instead
                "engine_one_sided": rep.engine_stats.get(
                    "skipped_one_sided", 0),
                # amendments the exchange refused because the order had
                # already gone. A real cost, and a source of divergence if
                # only one side incurs it.
                "engine_replace_rejected": rep.engine_stats.get(
                    "replace_rejected_stale", 0),
                # and how often each side skipped a reprice because a cancel
                # was still unacknowledged
                "backtest_ack_blocked": bt.stats.get(
                    "requotes_blocked_by_ack", 0),
                "engine_ack_blocked": rep.stats.get(
                    "requotes_blocked_by_ack", 0),
                # ---- THE DUPLICATE-SEND COUNT ----------------------------
                # THE ARITHMETIC THAT PUT THESE HERE. On MLCF 2026-06-24 the
                # backtester sent 1,822 messages, of which 66 were amendments,
                # leaving 1,756 NEW orders. An order can only leave a side by
                # being cancelled or fully filled: 231 cancels and at most 32
                # fills is at most 263 departures, plus the two orders that
                # open the day. The side cannot have been legitimately empty
                # 1,756 times when only about 265 orders ever left it. The
                # engine's own figures reconcile: 359 - 87 = 272 new orders
                # against 234 cancels and at most 24 fills.
                #
                # The one way to place a new order on a side that is not empty
                # is for the side to READ as empty when it is not, and that is
                # exactly what self.work does in mm_backtest: an order is
                # recorded there when it LANDS, not when it is SENT. These two
                # counters measure that directly rather than inferring it.
                "backtest_dup_sends": bt.stats.get(
                    "orders_sent_while_new_in_flight", 0),
                "engine_dup_sends": rep.stats.get(
                    "orders_sent_while_new_in_flight", 0),
                # and what those duplicates cost when they land: a resting
                # order replaced in place, never cancelled, never closed out
                "backtest_orphaned": bt.stats.get(
                    "orders_orphaned_by_overwrite", 0),
                "engine_orphaned": rep.stats.get(
                    "orders_orphaned_by_overwrite", 0),
                # ---- WHAT THE FIX PUT IN THEIR PLACE ---------------------
                # Every duplicate send is now a cycle that does nothing,
                # because the side is held while its order is on the wire.
                # This number should be LARGE and should roughly replace the
                # duplicate count; the duplicate count itself should be zero.
                "backtest_blocked_in_flight": bt.stats.get(
                    "requotes_blocked_in_flight", 0),
                "engine_blocked_in_flight": rep.stats.get(
                    "requotes_blocked_in_flight", 0),
                # an arrival for an order the side is no longer holding. The
                # old code applied these by overwriting whatever was there.
                "backtest_stale_arrivals": bt.stats.get(
                    "stale_arrivals_ignored", 0),
                "engine_stale_arrivals": rep.stats.get(
                    "stale_arrivals_ignored", 0),
                # ---- WHY EACH POLICY'S SIDES WENT QUIET ------------------
                # The order manager counts the four -- and only four -- ways a
                # side can produce no actions. Added after the two policies
                # came back with a threefold difference in orders sent and the
                # reason was GUESSED at twice: first as "more messages to wait
                # on", which was wrong because a partially filled order is
                # acknowledged and holds nothing up. Guessing stops here.
                "engine_acted": rep._oms.plan_counts["acted"],
                "engine_held_inflight":
                    rep._oms.plan_counts["held_message_in_flight"],
                "engine_held_suspended": rep._oms.plan_counts["held_suspended"],
                "engine_no_change": rep._oms.plan_counts["no_change"],
                "hold_acted": hold._oms.plan_counts["acted"],
                "hold_held_inflight":
                    hold._oms.plan_counts["held_message_in_flight"],
                "hold_held_suspended": hold._oms.plan_counts["held_suspended"],
                "hold_no_change": hold._oms.plan_counts["no_change"],
            })
        # progress
        print(f"  [{di}/{len(dates)}] {date} done", flush=True)

    # nothing ran: say so rather than emitting an empty file
    if not rows:
        raise SystemExit("no symbol-days ran; check the store and PYTHONPATH")
    # the results
    df = pd.DataFrame(rows)

    # ---- the verdict ----------------------------------------------------
    print("\n" + "=" * 78)
    print(f"RESULT -- {len(df)} symbol-days")
    print("=" * 78)
    # how many matched to the paisa
    exact = int((df["diff"].abs() < 0.005).sum())
    # the worst single disagreement
    worst = df.loc[df["diff"].abs().idxmax()]
    print(f"  exact matches        : {exact} of {len(df)}")
    print(f"  total backtest P&L   : {df['backtest_pnl'].sum():,.2f} PKR")
    print(f"  total engine P&L     : {df['engine_pnl'].sum():,.2f} PKR")
    print(f"  total difference     : {df['diff'].sum():,.2f} PKR")
    print(f"  worst symbol-day     : {worst['symbol']} {worst['date']} "
          f"{worst['diff']:+,.2f} PKR")
    # THE ONE THING THAT MUST BE ZERO. A rejection means the risk gateway, not
    # the order manager, changed the run, and the difference below cannot be
    # read as a statement about the order manager at all.
    n = int(df["gateway_rejections"].sum())
    # only worth a line when non-zero, and then it is the explanation
    if n:
        print(f"  !! gateway_rejections = {n}  "
              f"(actions the risk gateway refused)")
        print(f"     This changed the run for a reason that is NOT the")
        print(f"     order manager. Fix it before reading the difference.")
    # ---- CROSSED BOOKS: A CROSS-CHECK, NOT A WARNING ---------------------
    # CORRECTED 2026-09-17. This block used to announce that "mm_backtest has
    # no such check and quotes anyway". That was WRONG, and the gate's own
    # output disproved it: backtest_crossed and engine_crossed came back
    # identical on every symbol-day (87/87, 90/90, 5/5, 26/26). mm_backtest
    # has had the check since the skip_crossed_book flag was added, and the
    # two programs stand down together.
    #
    # What skipped_crossed_book actually counts is the PRODUCTION ADAPTER
    # refusing a crossed book, one layer above the replay's own check, so it
    # very nearly duplicates engine_crossed. Printed here as a cross-check
    # between the two layers rather than as a defect.
    _adapter = int(df["skipped_crossed_book"].sum())
    # the replay-level counter on each side of the comparison
    _bt_x = int(df["backtest_crossed"].sum())
    _en_x = int(df["engine_crossed"].sum())
    # a line only when a crossed book was actually seen
    if _adapter or _bt_x or _en_x:
        print(f"  crossed books        : backtest {_bt_x}, engine {_en_x}, "
              f"adapter {_adapter}")
        # the two REPLAY counters are the ones that must agree; the adapter's
        # sits at a different layer and is expected to be close, not equal
        if _bt_x != _en_x:
            print(f"     !! the two sides did NOT stand down together. That is")
            print(f"        a divergence in its own right -- read it before")
            print(f"        reading the P&L difference.")
        else:
            print(f"     both sides stood down on the same events, so crossed")
            print(f"     books contribute nothing to the difference below.")

    # EVERY DAY THAT DID NOT MATCH, listed individually. An average would hide
    # the one day that is wrong among nineteen that are right, and the one day
    # is the bug.
    bad = df[df["diff"].abs() >= 0.005]
    # the pass/fail line
    if bad.empty:
        print("\n  PASS -- every symbol-day matched to the paisa.")
    else:
        print(f"\n  FAIL -- {len(bad)} symbol-day(s) did not match:")
        print(bad[["symbol", "date", "backtest_pnl", "engine_pnl", "diff",
                   "backtest_fills", "engine_fills",
                   "backtest_orders", "engine_orders"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        print("\n  WHERE TO LOOK. A fill-count difference means the two sides")
        print("  put orders in different places or at different times -- start")
        print("  there, not at the P&L. Equal fills with different P&L means")
        print("  the same orders filled at different prices, which points at")
        print("  the queue model rather than the order manager.")
        # ---- THE STAND-DOWNS, PRINTED SIDE BY SIDE ----------------------
        # A message-count gap this size has a cause, and it is one of these.
        # Printed as a second table rather than added to the first, because a
        # nineteen-column table is unreadable.
        print("\n  WHY THE TWO SIDES SENT DIFFERENT NUMBERS OF MESSAGES")
        print("  Each pair is the SAME counter on the SAME kind of object, so")
        print("  like is compared with like. Read the pair, not the number.")
        print(bad[["symbol", "date",
                   # duplicate new orders sent into the latency window
                   "backtest_dup_sends", "engine_dup_sends",
                   # resting orders silently replaced when those land
                   "backtest_orphaned", "engine_orphaned",
                   # and the cycles that now correctly do nothing instead
                   "backtest_blocked_in_flight", "engine_blocked_in_flight",
                   # and the three reasons either side pulls its quote
                   "backtest_halted", "engine_halted",
                   "backtest_crossed", "engine_crossed",
                   "backtest_stale", "engine_stale"]]
              .to_string(index=False))
        # the two columns that only exist on one side, so they cannot be
        # paired and get a line of their own
        print(f"\n  engine-only: one-sided books skipped "
              f"{int(bad['engine_one_sided'].sum())}, amendments the exchange "
              f"refused {int(bad['engine_replace_rejected'].sum())}")
        # AND THE READING. Say what a large dup_sends figure MEANS, here, where
        # the number is, rather than leaving it as a column to be puzzled over.
        if int(bad["backtest_dup_sends"].sum()) > 0:
            print("\n  !! backtest_dup_sends is NOT zero.")
            print("     mm_backtest records an order in self.work when it")
            print("     LANDS at the exchange, not when it is SENT. For one")
            print("     network latency after every send, that side reads as")
            print("     empty, and the next requote -- which is event-driven,")
            print("     so it can fire many times in that window -- sends")
            print("     ANOTHER new order. backtest_orphaned is what those")
            print("     duplicates cost: each one that lands overwrites the")
            print("     order already resting there, which is then never")
            print("     cancelled, never filled and never closed out.")
            print("     The production order manager marks an order")
            print("     PENDING_NEW at SEND time and holds the side until the")
            print("     exchange answers, which is why its figure is lower.")
            print("     THE ENGINE IS RIGHT AND THE BACKTEST IS WRONG. Fixing")
            print("     it changes every measured number this project has")
            print("     produced, so it is a decision, not a patch.")

    # ---- THE MEASUREMENT: what is queue position worth? ------------------
    # This is NOT part of the pass/fail above. The gate asks whether the engine
    # reproduces the backtest; this asks which of two legal requote policies
    # makes more money, with the exchange's own priority rule applied to both.
    print("\n" + "=" * 78)
    print("WHAT KEEPING QUEUE POSITION IS WORTH")
    print("=" * 78)
    print("  THE WORKED EXAMPLE, because the names alone are not enough.")
    print("  The clip is 500. A partial fill takes 400 and leaves 100 resting")
    print("  with the queue position it has already earned. Both policies want")
    print("  500 showing again; they differ ONLY in how they get there.")
    print()
    print("  TOP UP BY AMENDMENT   (column: engine_*, quantity_policy=exact)")
    print("          ONE order. Amend the resting 100 back up to 500. Under PSX")
    print("          8.5.2 raising the size sends THE WHOLE ORDER to the back")
    print("          of the queue -- the 100 loses the place it earned along")
    print("          with the 400 that is new. This is mm_backtest's rule and")
    print("          what every measured number in this project was produced")
    print("          under, which is why it is the run the gate judges.")
    print()
    print("  TOP UP BY SECOND ORDER (column: hold_*, quantity_policy=")
    print("                          queue_preserving)")
    print("          TWO orders. The 100 is LEFT ALONE and keeps its place; a")
    print("          SEPARATE order for the 400 joins the back of the queue")
    print("          under its own id. Total size showing is the same 500. You")
    print("          pay in priority only on the increment, never on the")
    print("          remainder. When the strategy wants LESS than is resting it")
    print("          shrinks the YOUNGEST order first, and 8.5.2 applies a size")
    print("          reduction in place, so that costs nothing at all.")
    print()
    print("  CORRECTED 2026-09-17. This block used to describe the second")
    print("  policy as one that 'shows less size until it is hit'. That was")
    print("  wrong and it misdescribed every number below it. The policy has")
    print("  always topped the size back up with a second order -- see")
    print("  core/oms.py, _plan_side_multi, the SHORT OF WHAT WE WANT branch.")
    print("  A policy that genuinely declines to top up, and shows the smaller")
    print("  size until it is hit, is a THIRD thing and is not implemented.")
    print()
    print("  Both are legal and both are things a real desk does. The exchange")
    print("  applies the identical rule to each; the difference is ours.\n")
    # the paired per-day difference, which is how this project measures
    d = df["hold_minus_topup"].to_numpy()
    # the totals under each policy
    print(f"  total, amendment     : {df['engine_pnl'].sum():,.2f} PKR")
    print(f"  total, second order  : {df['engine_hold_pnl'].sum():,.2f} PKR")
    print(f"  difference           : {d.sum():+,.2f} PKR "
          f"({'second order' if d.sum() > 0 else 'amendment'} ahead)")
    # DAY AS UNIT, never pooled fills -- the standing rule on this project
    if len(d) > 1:
        # the mean paired difference
        mean_d = d.mean()
        # its standard error
        se = d.std(ddof=1) / (len(d) ** 0.5)
        # and the t-statistic the |t| > 2 bar applies to
        t = mean_d / se if se > 0 else float("nan")
        print(f"  per symbol-day       : {mean_d:+,.2f} PKR mean, "
              f"se {se:,.2f}, t {t:,.2f}  ({len(d)} days)")
        print(f"  days 2nd order better: {int((d > 0).sum())} of {len(d)}")
        # a sample this size measures a LARGE effect and nothing smaller
        print("  A sample this size can only show a large effect. It cannot")
        print("  rule out a small one, and |t| > 2 is the bar here as elsewhere.")
    # the churn each policy generated, which is the mechanism behind any gap
    print(f"\n  orders sent          : amendment {df['engine_orders'].sum():,}"
          f"   second order {df['hold_orders'].sum():,}")
    print(f"  fills                : amendment {df['engine_fills'].sum():,}"
          f"   second order {df['hold_fills'].sum():,}")
    print(f"  amendments kept place: amendment {df['engine_cfos_kept'].sum():,}"
          f" of {df['engine_cfos'].sum():,}"
          f"   second order {df['hold_cfos_kept'].sum():,}"
          f" of {df['hold_cfos'].sum():,}")
    print("  The last line is 8.5.2 doing its work: under the second-order")
    print("  policy more amendments are size REDUCTIONS, which keep their")
    print("  place, because topping UP no longer needs an amendment at all.")
    print()
    print("  READ THE ORDER COUNT BEFORE THE P&L. If the second-order policy")
    print("  is sending far FEWER orders and taking far fewer fills, it is not")
    print("  losing a fair contest -- it is barely competing, and the reason")
    print("  has to be found before the P&L comparison means anything.")
    print()
    print("  WHY EACH POLICY'S SIDES WERE QUIET, counted rather than guessed:")
    # the four causes, per policy, as the order manager counted them
    for _label, _col in (("amendment   ", "engine"), ("second order", "hold")):
        # one line per policy, all four causes on it
        print(f"    {_label}  acted {df[_col + '_acted'].sum():>7,}"
              f"   held: message in flight "
              f"{df[_col + '_held_inflight'].sum():>7,}"
              f"   suspended {df[_col + '_held_suspended'].sum():>5,}"
              f"   no change {df[_col + '_no_change'].sum():>7,}")
    print("  A quiet side has exactly these four causes. Whichever one differs")
    print("  between the two rows is the explanation; nothing else can be.")

    # ---- write ----------------------------------------------------------
    # a fresh timestamped destination; never overwrites
    out = EX.safe_out(f"gate_{args.mode}", "csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    # a non-zero exit on failure, so this can gate a pipeline
    raise SystemExit(0 if bad.empty else 1)


# entry point
if __name__ == "__main__":
    main()
