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
# wall-clock, so a long run can say how far along it is
import time
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
# the kill switch, the gateway, and the checks this run applies
from core.risk import (KillSwitch, OrderQuantityCheck, PriceBandCheck,
                       RiskGateway)
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
# HOW FAR FROM OUR OWN MID THE GATEWAY TOLERATES A QUOTE, in percent. Wide on
# purpose here so the EXCHANGE band is what binds -- see run_engine. Not a
# live-trading value.
HOUSE_BAND_PCT = 25.0
# the session segment the venue hands out, and the published price band
from core.venue import PriceBand, SessionSegment
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


def _end_reasons(run):
    """Count how every order record on this run was closed.

    ONE RECORD PER ORDER GENERATION. An amendment CLOSES the record it
    amends and OPENS a new one, so this counts quote versions rather than
    distinct orders -- `records` is not the number of orders that needed
    disposing of and must not be read as one.

    Every record carries exactly one reason once mm_backtest labels the
    crossings and the two at-close cases, so the buckets sum to `records`
    with no remainder. `unlabelled` exists to catch a record that slipped
    through, and must be zero.
    """
    # every reason a record can close with, in the order the table prints
    keys = ("cancelled", "amended", "filled", "taker", "crossed_filled",
            "rejected_crossing", "open_at_close", "in_flight_at_close")
    # start every bucket at zero so a run that never hits one still reports
    # the column -- a missing key would change the CSV's shape between days
    out = {k: 0 for k in keys}
    # records carrying a reason this function does not know about
    out["unlabelled"] = 0
    # the total, which the buckets above must add up to
    out["records"] = 0
    # walk the lifecycle log this run produced
    for rec in run._olog.values():
        # one more record, counted before anything can skip it
        out["records"] += 1
        # the reason it closed with, or None if nothing set one
        r = rec.get("end_reason")
        # into its own bucket, or into the defect bucket
        if r in keys:
            out[r] += 1
        else:
            out["unlabelled"] += 1
    # the caller prefixes these with the run's own column stem
    return out


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
    # ---- THE PUBLISHED PRICE BAND, AND WHY THIS EXISTS -------------------
    # THE BUG THIS FIXES. Until 2026-09-18 this venue was built with NO band
    # provider, so PSXVenue.price_band() returned None for every symbol. Two
    # things downstream then went quiet rather than wrong:
    #
    #   * MicroMMAdapter._clamp_to_band clamped to nothing, so a quote outside
    #     the daily circuit limit went out as-is. mm_backtest clamps (see its
    #     _requote, "BAND CLAMP"), so the two sides sent different prices.
    #   * The adapter sets micro_mm.limit_up/limit_dn from the same call, so
    #     the strategy's lock trigger was fed None in the engine run.
    #
    # MEASURED, not inferred: on THCCL 2026-06-05 the strategy asked for an
    # ask of 66.89 in BOTH runs. The backtest clamped it to the published
    # 66.14 and the engine sent 66.89. That is the whole of the one symbol-day
    # in 240 whose order counts disagreed. Live, PSX rejects 66.89 and that
    # side stops quoting.
    #
    # READ THE BAND, NEVER COMPUTE IT. PriceBand's own docstring is explicit:
    # on a split the exchange bands off the ADJUSTED close, so a +/-10%
    # reconstruction is wrong by the split ratio on exactly the day it
    # matters. mm_backtest reads it off the UPPER/LOWER_CIRCUIT_BREAKER rows
    # in the snapshot feed and keeps it on its Book, so that is the source.
    #
    # LATE BINDING, because the venue has to exist before the exchange that
    # publishes the band. This one-element list holds the EngineReplay once it
    # is built; the closure reads through it on every call, so the band is
    # always the one being published right now rather than a snapshot.
    _exchange = []

    def _band_provider(symbol):
        # nothing to read until the exchange has been constructed below
        if not _exchange:
            return None
        # mm_backtest's own Book, which the snapshot feed keeps current
        book = _exchange[0].book
        # the two published bounds, in major units (rupees)
        up, dn = book.limit_up, book.limit_dn
        # NO BAND MEANS NO BAND. The sentinels are already handled upstream --
        # the parser nulls the upper one and flags a suspicious lower one --
        # so None here is an absent bound, not a wide one, and inventing a
        # band would be worse than having none.
        if up is None and dn is None:
            return None
        # the venue speaks in paisa. Each bound converts independently,
        # because one side can be published without the other.
        return PriceBand(
            upper_minor=(int(round(up * 100)) if up is not None else None),
            lower_minor=(int(round(dn * 100)) if dn is not None else None))

    # the venue. Its session provider hands back the same window the backtest
    # uses, so the trading-window check cannot be the thing that differs, and
    # its band provider reads the same circuit limits mm_backtest clamps to.
    venue = PSXVenue(session_provider=lambda d: [
        SessionSegment(start_ms=t0, end_ms=t1)],
        band_provider=_band_provider)
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
    #
    # PRICE BAND ADDED 2026-09-18, AND IT IS THE ONE EXCEPTION TO "PERMISSIVE".
    # THCCL 2026-06-05 was the engine quoting 66.89 against a published ceiling
    # of 66.14 -- an order PSX rejects on arrival -- and nothing in this run
    # noticed. MicroMMAdapter._clamp_to_band is what prevents it; this is the
    # backstop that catches the day the clamp is fed a band it did not expect.
    #
    # IT MUST NEVER FIRE. The clamp runs first and pulls every price inside the
    # published band, so a rejection here means the band moved between the
    # clamp and the gateway, or the clamp was bypassed. Either is worth failing
    # the run for, and `gateway_rejections` is already a must-be-zero column.
    #
    # THE HOUSE BAND IS WIDE ON PURPOSE. PriceBandCheck enforces the tighter of
    # the exchange band and a percentage around our own mid. Here the EXCHANGE
    # band is the one we want binding, so the house figure is set well outside
    # anything micro_mm can legitimately produce -- it quotes inside the touch,
    # so its distance from the mid is half a spread. A live deployment should
    # tighten this considerably: the house band is the control that catches a
    # plausible-looking price computed from a stale book, which the exchange
    # band at +/-10% is far too wide to catch.
    gateway = RiskGateway([OrderQuantityCheck(max_quantity=1_000_000),
                           PriceBandCheck(venue, house_band_pct=HOUSE_BAND_PCT)])
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
    # ARM THE BAND PROVIDER. Before this line it returns None, which is
    # correct: nothing has been published yet because nothing has run. From
    # here the venue reads the exchange's live circuit limits.
    _exchange.append(rep)
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
    # ---- THE HEARTBEAT ---------------------------------------------------
    # A full run is 12 names x 20 days x FOUR engine runs each, and until now
    # it printed one line per DAY -- 48 engine runs apart. On a run that takes
    # an hour that is indistinguishable from a hang. It now prints a line per
    # symbol-day with how long it has taken and how long is left.
    #
    # NOTE `tail` HOLDS EVERYTHING UNTIL THE PROCESS ENDS, so piping this into
    # `tail -60` hides the heartbeat completely. Run it without the pipe, or
    # send it to a file and `tail -f` that.
    t_started = time.time()
    # how many symbol-days this run will attempt, for the percentage
    planned = len(dates) * len([n_ for n_ in names
                                if n_ in scales and n_ in profiles
                                and n_ in windows])
    # how many have finished
    done_n = 0
    # say the size of the job up front
    print(f"  to do     : {planned} symbol-days, 4 engine runs each\n",
          flush=True)
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
                # THE SAME ENGINE, THREE TIMES, under the three requote
                # policies. Identical day, identical exchange, identical
                # latency seed -- the ONLY thing that differs is what each one
                # does when the size resting is not the size wanted. That is
                # what makes the comparison a measurement rather than three
                # runs that happen to disagree.
                #
                # 1. AMEND UP. One order; amend it back to the full clip after
                #    a partial fill. PSX 8.5.2 sends the whole order to the
                #    back of the queue for that, so the shares that were
                #    already resting lose their place along with the new ones.
                #    This is mm_backtest's rule and what the PASS/FAIL gate
                #    judges.
                rep = run_engine(events, snap_groups, params, t0, t1,
                                 ref_minor, sym, date, use_g,
                                 quantity_policy="exact")
                # 2. SECOND ORDER. Leave the remainder alone and send a
                #    separate order for the shortfall under its own id. Full
                #    size showing; priority paid only on the increment.
                hold = run_engine(events, snap_groups, params, t0, t1,
                                  ref_minor, sym, date, use_g,
                                  quantity_policy="queue_preserving")
                # 3. DON'T TOP UP. Leave the remainder resting, show the
                #    SMALLER size, and wait to be hit. No new order, no
                #    amendment, nothing given up at all -- and less size
                #    working, which is the price of it. Takes the one
                #    amendment 8.5.2 makes free (a REDUCTION, when the
                #    strategy wants less than is resting) and pays for none.
                reduce = run_engine(events, snap_groups, params, t0, t1,
                                    ref_minor, sym, date, use_g,
                                    quantity_policy="reduce_only")
            except Exception as exc:                          # noqa: BLE001
                # report and carry on: one broken day must not lose the rest
                print(f"    {sym} {date} ERROR {exc!r}")
                continue
            # the four headline numbers
            base_pnl, eng_pnl = pnl_of(bt), pnl_of(rep)
            hold_pnl, reduce_pnl = pnl_of(hold), pnl_of(reduce)
            # a broken close on any side has nothing to compare
            if (base_pnl is None or eng_pnl is None or hold_pnl is None
                    or reduce_pnl is None):
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
                # ---- POLICY 3: DON'T TOP UP -----------------------------
                # Every column the other two carry, so the three are read the
                # same way. `reduce_orders` should be the LOWEST of the three
                # by construction: this policy never sends a message to grow.
                "engine_reduce_pnl": float(reduce_pnl),
                "reduce_minus_topup": float(reduce_pnl) - float(eng_pnl),
                "reduce_orders": reduce.stats.get("n_orders_sent", 0),
                "reduce_cancels": reduce.stats.get("n_cancels", 0),
                "reduce_fills": len(reduce.fills),
                # EVERY amendment this policy sends is a reduction, and 8.5.2
                # applies those in place -- so reduce_cfos_kept should equal
                # reduce_cfos. If it does not, the venue rule is not being
                # read the way this policy assumes.
                "reduce_cfos": reduce.stats.get("n_cfos", 0),
                "reduce_cfos_kept": reduce.stats.get("n_cfos_kept_priority", 0),
                "reduce_acted": reduce._oms.plan_counts["acted"],
                "reduce_held_inflight":
                    reduce._oms.plan_counts["held_message_in_flight"],
                "reduce_held_suspended":
                    reduce._oms.plan_counts["held_suspended"],
                "reduce_no_change": reduce._oms.plan_counts["no_change"],
                # ---- CAN THIS HARNESS EVEN RUN THAT POLICY? --------------
                # EngineReplay inherits Backtester's `work`, which is
                # dict[side] -> ONE MyOrder. A policy that wants SEVERAL
                # orders on a side gets the second one landing ON TOP of the
                # first -- including the very order whose queue position the
                # policy exists to protect. Counted per run so the column can
                # be refused rather than read.
                "engine_occupied": rep.engine_stats.get(
                    "placed_onto_occupied_side", 0),
                "hold_occupied": hold.engine_stats.get(
                    "placed_onto_occupied_side", 0),
                "reduce_occupied": reduce.engine_stats.get(
                    "placed_onto_occupied_side", 0),
                # ---- WHAT KIND OF AMENDMENT, PER POLICY ------------------
                # Under 8.5.2 only a pure size REDUCTION keeps its place. So
                # "how many kept their place?" is the same question as "how
                # many were reductions?", and this decomposition answers it
                # rather than leaving it to be predicted from the policy's
                # description -- which was done twice, wrongly, on 2026-09-17.
                "engine_cfo_px": rep.stats.get("n_cfos_price_change", 0),
                "engine_cfo_up": rep.stats.get("n_cfos_qty_up", 0),
                "engine_cfo_dn": rep.stats.get("n_cfos_qty_down", 0),
                "hold_cfo_px": hold.stats.get("n_cfos_price_change", 0),
                "hold_cfo_up": hold.stats.get("n_cfos_qty_up", 0),
                "hold_cfo_dn": hold.stats.get("n_cfos_qty_down", 0),
                "reduce_cfo_px": reduce.stats.get("n_cfos_price_change", 0),
                "reduce_cfo_up": reduce.stats.get("n_cfos_qty_up", 0),
                "reduce_cfo_dn": reduce.stats.get("n_cfos_qty_down", 0),
                # ---- HOW EVERY ORDER RECORD ENDED, PER RUN ---------------
                # The order log already recorded this per order, but nothing
                # carried it up to the CSV -- so the disposal of ~12,000
                # records per policy could only be seen by loading a run by
                # hand. Every key is prefixed with the run's own column stem,
                # so the three policies and the backtest read the same way.
                **{f"engine_end_{_k}": _v
                   for _k, _v in _end_reasons(rep).items()},
                **{f"hold_end_{_k}": _v
                   for _k, _v in _end_reasons(hold).items()},
                **{f"reduce_end_{_k}": _v
                   for _k, _v in _end_reasons(reduce).items()},
                **{f"backtest_end_{_k}": _v
                   for _k, _v in _end_reasons(bt).items()},
                # ---- ORDERS THAT ARRIVED MARKETABLE ----------------------
                # A quote that left passive and landed through the touch.
                # PSX matches these (Regulation 8.4.2); until 2026-09-18 both
                # engines threw them away. Counted on both sides so the two
                # can be checked against each other rather than assumed equal.
                "backtest_crossed_arr": bt.stats.get("crossed_on_arrival", 0),
                "engine_crossed_arr": rep.stats.get("crossed_on_arrival", 0),
                "backtest_crossed_sh":
                    bt.stats.get("crossed_on_arrival_shares", 0.0),
                "engine_crossed_sh":
                    rep.stats.get("crossed_on_arrival_shares", 0.0),
                # ---- MESSAGES, SPLIT BY WHAT THEY ARE --------------------
                # n_orders_sent is the SUM of new orders and amendments, which
                # made the old table's "orders" column silently include its
                # "amendments" column. These two are disjoint and add to it.
                "engine_new_sent": rep.stats.get("n_new_orders_sent", 0),
                "engine_amend_sent": rep.stats.get("n_amends_sent", 0),
                "hold_new_sent": hold.stats.get("n_new_orders_sent", 0),
                "hold_amend_sent": hold.stats.get("n_amends_sent", 0),
                "reduce_new_sent": reduce.stats.get("n_new_orders_sent", 0),
                "reduce_amend_sent": reduce.stats.get("n_amends_sent", 0),
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
            # ---- one line per symbol-day, as it finishes -----------------
            # how many are complete
            done_n += 1
            # seconds since the run started
            elapsed = time.time() - t_started
            # seconds each symbol-day has taken on average so far
            per = elapsed / done_n
            # and the projection for what is left
            left = per * (planned - done_n)
            # the day's headline number, so a run that has gone wrong shows it
            # here rather than an hour later
            print(f"  [{done_n:>4}/{planned}] {sym:<7} {date}"
                  f"  backtest {float(base_pnl):>9,.2f}"
                  f"  engine {float(eng_pnl):>9,.2f}"
                  f"  |  {elapsed/60:.1f}m elapsed,"
                  f" ~{left/60:.0f}m left", flush=True)
        # the date is finished
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
    print("  ONE WORKED EXAMPLE, because the names alone are not enough.")
    print("  The clip is 500. A partial fill takes 400 and leaves 100 resting")
    print("  with the queue position it has already earned. What now?")
    print()
    print("  1. AMEND UP      (engine_*,  quantity_policy = exact)")
    print("     ONE order. Amend the resting 100 back up to 500. PSX 8.5.2")
    print("     sends THE WHOLE ORDER to the back of the queue for a size")
    print("     increase, so the 100 loses the place it earned along with the")
    print("     400 that is new. Full size showing, priority paid on all of")
    print("     it. This is mm_backtest's rule and what every published number")
    print("     in this project was produced under -- which is why it is the")
    print("     run the PASS/FAIL gate judges and the baseline for the rest.")
    print()
    print("  2. SECOND ORDER  (hold_*,    quantity_policy = queue_preserving)")
    print("     TWO orders. The 100 is left alone and keeps its place; a")
    print("     SEPARATE order for the 400 joins the back under its own id.")
    print("     Full size showing, priority paid only on the increment.")
    print("     Shrinking works youngest-first, and 8.5.2 applies a REDUCTION")
    print("     in place, so giving size back costs nothing.")
    print()
    print("  3. DON\'T TOP UP  (reduce_*,  quantity_policy = reduce_only)")
    print("     NO message at all. The 100 stays exactly where it is and the")
    print("     quote shows 100, not 500, until someone hits it. Nothing is")
    print("     given up, and nothing is added -- so the cost is not priority,")
    print("     it is the 400 shares of working size you are not showing.")
    print("     Takes the free reduction when the strategy wants LESS, pays")
    print("     for nothing. This is what a desk does when it thinks the flow")
    print("     at that price is toxic: keep the place, stop feeding it size.")
    print()
    print("  All three are legal and all three are things a real desk does.")
    print("  The exchange applies the identical rule to each; the difference")
    print("  is ours, and this is what it is worth.\n")
    # the paired per-day difference, which is how this project measures
    # THE THREE POLICIES, in one table. Column key -> label, and the P&L
    # column each one reports under. "amend up" is the baseline every
    # difference is measured against, because it is what produced every number
    # this project has published.
    POLICIES = (("amend up    ", "engine", "engine_pnl"),
                ("second order", "hold", "engine_hold_pnl"),
                ("don't top up", "reduce", "engine_reduce_pnl"))
    # the totals
    print("  TOTAL P&L, SAME DAYS, SAME EXCHANGE, SAME LATENCY SEED")
    for _label, _col, _pnl in POLICIES:
        # the policy's own total
        print(f"    {_label}  {df[_pnl].sum():>12,.2f} PKR")
    print()
    # ---- the paired per-day comparison, each against the baseline --------
    # DAY AS UNIT, never pooled fills -- the standing rule on this project
    print("  AGAINST 'AMEND UP', PAIRED BY SYMBOL-DAY")
    print("  Paired because the two runs are the SAME day: the day's own")
    print("  volatility cancels, which an unpaired comparison would leave in.")
    for _label, _diff_col in (("second order", "hold_minus_topup"),
                              ("don't top up", "reduce_minus_topup")):
        # the paired differences
        d = df[_diff_col].to_numpy()
        # the total, and which way it points
        line = (f"    {_label}  total {d.sum():+10,.2f} PKR"
                f"   better on {int((d > 0).sum())} of {len(d)} days")
        # the t-statistic, where there is more than one day to compute it on
        if len(d) > 1:
            # mean paired difference
            mean_d = d.mean()
            # its standard error
            se = d.std(ddof=1) / (len(d) ** 0.5)
            # and the statistic the |t| > 2 bar applies to
            t = mean_d / se if se > 0 else float("nan")
            line += f"   mean {mean_d:+8,.2f}  se {se:7,.2f}  t {t:5,.2f}"
        print(line)
    # a sample this size measures a LARGE effect and nothing smaller
    print("  A sample this size can only show a large effect. It cannot rule")
    print("  out a small one, and |t| > 2 is the bar here as elsewhere.")

    # ---- the churn, which is the mechanism behind any gap ----------------
    print("\n  WHAT EACH POLICY DID")
    print("  Every column counts a DIFFERENT thing, and they do not overlap.")
    print("    new orders  a brand new order sent to the exchange")
    print("    amends      a Change Former Order sent: one message that")
    print("                changes the price or size of an order already")
    print("                resting, instead of cancelling and re-sending")
    print("    landed      of those amendments, how many the exchange")
    print("                actually applied. The rest arrived to find their")
    print("                target already filled or already gone")
    print("    kept place  of the ones that landed, how many kept their")
    print("                position in the queue at that price. PSX 8.5.2:")
    print("                only a pure SIZE REDUCTION is applied in place. A")
    print("                price change, or asking for MORE size, sends the")
    print("                order to the BACK of the line -- behind everyone")
    print("                who was already there, so it fills later or not")
    print("                at all. This column is what churn costs you.")
    print("    cancels     cancels that reached the exchange and removed an")
    print("                order that was still resting")
    print("    fills       fill EVENTS, not shares and not round trips: one")
    print("                partial fill is one event")
    print(f"\n    {'policy':14} {'new orders':>11} {'amends':>8} {'landed':>8}"
          f" {'kept place':>11} {'cancels':>9} {'fills':>7}")
    for _label, _col, _pnl in POLICIES:
        # one line per policy, every column disjoint from the others
        print(f"    {_label} {df[_col + '_new_sent'].sum():>11,}"
              f" {df[_col + '_amend_sent'].sum():>8,}"
              f" {df[_col + '_cfos'].sum():>8,}"
              f" {df[_col + '_cfos_kept'].sum():>11,}"
              f" {df[_col + '_cancels'].sum():>9,}"
              f" {df[_col + '_fills'].sum():>7,}")
    # ---- WHAT THE AMENDMENTS ACTUALLY WERE -------------------------
    # THE KEPT-PLACE COUNT IS NOT A THING TO PREDICT. Under 8.5.2 only a pure
    # size REDUCTION is applied in place; a price change re-queues, and so
    # does a size increase. So "how many kept their place" is the same
    # question as "how many were reductions", and the honest way to answer it
    # is to count the kinds rather than reason from the policy's description.
    print("\n  WHAT THOSE AMENDMENTS WERE")
    print(f"    {'policy':14} {'price move':>11} {'size up':>9}"
          f" {'size down':>11} {'kept place':>11}")
    for _label, _col, _pnl in POLICIES:
        # the three kinds, and the kept count they should explain
        print(f"    {_label} {df[_col + '_cfo_px'].sum():>11,}"
              f" {df[_col + '_cfo_up'].sum():>9,}"
              f" {df[_col + '_cfo_dn'].sum():>11,}"
              f" {df[_col + '_cfos_kept'].sum():>11,}")
    print("  KEPT PLACE SHOULD EQUAL SIZE DOWN on PSX, because the venue's")
    print("  three rules say a reduction keeps its position and nothing else")
    print("  does. If those two columns disagree, the engine is not reading")
    print("  the venue the way this project believes it is.")
    print("  A PRICE MOVE DOMINATES EVERY POLICY here, which is the real")
    print("  finding: this strategy reprices on any tick, so a partial fill")
    print("  rarely survives long enough for the size policy to matter. That")
    print("  is why the three policies churn so similarly, and it bounds how")
    print("  much any of them can be worth.")

    # ---- why each policy's sides were quiet ------------------------------
    print("\n  WHY EACH POLICY'S SIDES WERE QUIET, counted rather than guessed")
    print("  A side produces no actions for exactly four reasons. Whichever")
    print("  one differs between these rows is the explanation for the order")
    print("  counts above; nothing else can be.")
    print(f"    {'policy':14} {'acted':>9} {'held: in flight':>16}"
          f" {'suspended':>10} {'no change':>10}")
    for _label, _col, _pnl in POLICIES:
        # all four causes, per policy
        print(f"    {_label} {df[_col + '_acted'].sum():>9,}"
              f" {df[_col + '_held_inflight'].sum():>16,}"
              f" {df[_col + '_held_suspended'].sum():>10,}"
              f" {df[_col + '_no_change'].sum():>10,}")
    print("  READ THE ORDER COUNT BEFORE THE P&L. A policy sending a third of")
    print("  the messages and taking a third of the fills is not losing a")
    print("  fair contest -- it is barely competing, and that has to be")
    print("  explained before its P&L means anything.")

    # ---- HOW EVERY ORDER RECORD ENDED, WITH NOTHING LEFT OVER ------------
    # This used to be a "remainder" column of about 12,000 per policy that
    # pooled four different outcomes, two of which carried no label at all.
    # Every record now closes with exactly one reason, and the arithmetic is
    # CHECKED below rather than asserted in prose.
    print("\n  HOW EVERY ORDER RECORD ENDED")
    print("  One record per order GENERATION, not per distinct order: an")
    print("  amendment CLOSES the record it amends and OPENS a new one, so")
    print("  these are versions of quotes, not orders needing disposal.")
    print("    crossed     arrived marketable and was fully consumed on")
    print("                contact (PSX 8.4.2). A partial crossing does NOT")
    print("                appear here: the rest rested, so the record is")
    print("                still open and ends some other way.")
    print("    refused     the OLD post-only behaviour. Must be 0 unless")
    print("                cfg[\"cross_on_arrival\"] was set False.")
    print("    open        still resting at the exchange when the day closed")
    print("    in flight   sent, but the day ended before it landed")
    print("    unlabelled  MUST BE 0: a reason mm_backtest failed to set")
    print(f"    {'policy':12} {'records':>9} {'cancelled':>10} {'amended':>9}"
          f" {'filled':>8} {'taker':>7} {'crossed':>9} {'refused':>9}"
          f" {'open':>7} {'in flight':>10} {'unlabelled':>11}")
    # the backtest reads the same way, so it is printed in the same loop
    for _label, _col in (list((l, c) for l, c, _ in POLICIES)
                         + [("backtest    ", "backtest")]):
        # the record total this run produced
        _n = int(df[_col + "_end_records"].sum())
        # the eight disjoint outcomes, in the order of the header
        _v = [int(df[_col + "_end_" + _k].sum()) for _k in
              ("cancelled", "amended", "filled", "taker", "crossed_filled",
               "rejected_crossing", "open_at_close", "in_flight_at_close",
               "unlabelled")]
        # one line per run
        print(f"    {_label} {_n:>9,} {_v[0]:>10,} {_v[1]:>9,} {_v[2]:>8,}"
              f" {_v[3]:>7,} {_v[4]:>9,} {_v[5]:>9,} {_v[6]:>7,}"
              f" {_v[7]:>10,} {_v[8]:>11,}")
        # THE POINT OF THE TABLE: the parts must equal the whole, said as a
        # failure line rather than left for the reader to add up by eye
        if sum(_v) != _n:
            print(f"      MISMATCH: parts sum to {sum(_v):,}, "
                  f"records is {_n:,}")

    # ---- ORDERS THAT ARRIVED MARKETABLE ---------------------------------
    # How often the book moved far enough during the latency window that a
    # quote which left passive landed through the touch. Both sides must
    # agree: they run the same exchange off the same seeded latency.
    _bx = int(df["backtest_crossed_arr"].sum())
    _ex = int(df["engine_crossed_arr"].sum())
    print("\n  ORDERS THAT ARRIVED MARKETABLE (PSX matches these; 8.4.2)")
    print(f"    backtest {_bx:,} orders, "
          f"{float(df['backtest_crossed_sh'].sum()):,.0f} shares traded")
    print(f"    engine   {_ex:,} orders, "
          f"{float(df['engine_crossed_sh'].sum()):,.0f} shares traded")
    # a disagreement here is a divergence in the exchange, not the policy
    if _bx != _ex:
        print("    THE TWO SIDES DISAGREE. Same exchange, same latency seed:")
        print("    they cannot legitimately differ. Diagnose before reading")
        print("    anything else in this run.")

    # ---- THE SIDE-SHARING COUNT, once a defect and now a measurement ----
    # Backtester held ONE order per side until 2026-09-17, so a policy that
    # rests two had its second order overwrite its first -- the one carrying
    # the queue position it exists to protect. `work` is a LIST per side now,
    # so this counts something real: how often a policy put a second order
    # alongside one already resting, which is the manoeuvre the whole
    # second-order design is built on.
    print("\n  ORDERS RESTED ALONGSIDE ANOTHER ON THE SAME SIDE")
    for _label, _col, _pnl in POLICIES:
        # how many of that policy's orders joined an occupied side
        print(f"    {_label} {int(df[_col + '_occupied'].sum()):>9,}")
    print("  The single-order policies should read ZERO here: each holds one")
    print("  order per side by construction, so a non-zero figure for 'amend")
    print("  up' or \'don\'t top up\' means one of them is resting size it does")
    print("  not know about. The second-order policy should be the only one")
    print("  with a count, and that count is the design working.")

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
