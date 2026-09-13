# ============================================================================
# test_micro_mm_contract.py -- TO-DO A.8
# The guard that would have caught the micro_mm.py clobber.
# ============================================================================
# Run it before every commit and before every sweep:
#     python test_micro_mm_contract.py
# Exit code 0 = contract intact. Non-zero = something the runners depend on is
# missing, and no sweep should start.
#
# WHY THIS FILE EXISTS
#   micro_mm.py was overwritten once and lost `queue_skew_bps`. Nothing noticed
#   until a sweep driver tried to pass it, hours later. Every parameter the
#   runners hand to MicrostructureMM is a contract; this asserts the contract
#   without needing data, a store, or a backtest.
#
# WHAT IT CHECKS
#   1. SIGNATURE  -- every kwarg the drivers pass exists on __init__.
#   2. QUOTES     -- quotes() still accepts `depth` (mm_backtest passes it
#                    unconditionally; build_feature_store's collector did NOT,
#                    which is why that build needed a monkey-patch).
#   3. CONSTRUCT  -- the full sweep kwarg set actually instantiates.
#   4. GEOMETRY   -- when the queue skew fires, BOTH quotes translate by the
#                    same number of ticks and the quoted width is unchanged.
#                    This pins the property established 2026-09-14: QT_2t is a
#                    conditional parallel shift, not an asymmetric skew. Any
#                    future edit that makes it asymmetric will fail here loudly
#                    instead of silently changing what the config means.
#   5. NO-CROSS   -- the post-only clip holds under an extreme skew.
#
#   Checks 1-3 are the hard gate and need nothing but an import. Checks 4-5 call
#   quotes() directly; if the strategy needs engine state the constructor does
#   not set, they report exactly what was missing rather than failing obscurely.
# ============================================================================

# signature introspection -- the whole point of checks 1 and 2
import inspect
# exit codes
import sys

# the strategy under contract
from micro_mm import MicrostructureMM

# ---------------------------------------------------------------------------
# THE CONTRACT: every kwarg any runner passes. Add to this list whenever a
# driver starts passing a new one -- that is what keeps the guard honest.
# ---------------------------------------------------------------------------
# the synthetic session window every behavioural check runs inside:
# midnight to +6h in exchange-ms. One definition, used everywhere.
SESSION = (0, 6 * 3600 * 1000)

REQUIRED_KWARGS = [
    # core sizing / inventory
    "size", "max_inv", "soft_inv", "session_ms", "tick",
    # fair value and risk
    "use_microprice", "micro_lambda", "gamma", "session_scale", "kappa",
    # costs and gating
    "fee_pct", "min_edge_pct", "quiet_ms", "require_viable",
    # exit skew (sweep axis 1)
    "exit_ticks_inside", "exit_inv_threshold",
    "enable_age_exit", "age_exit_ms", "enable_age_cross", "age_cross_ms",
    # OBI defensive (sweep axis 2)
    "obi_defensive", "obi_defensive_thresh", "obi_defensive_ticks",
    # OFI defensive (stage 3)
    "ofi_defensive", "ofi_window_ev", "ofi_window_s",
    "ofi_defensive_thresh", "ofi_defensive_ticks", "ofi_depth_levels",
    # size throttle (stage 4)
    "obi_throttle", "ofi_throttle", "throttle_frac",
    "obi_throttle_thresh", "ofi_throttle_thresh", "throttle_hold_ms",
    "qdr_throttle", "qdr_throttle_thresh",
    "flow_throttle", "flow_hl_s", "flow_thresh_n", "flow_std_hl_s",
    # THE ONE THAT WAS LOST IN THE CLOBBER -- never remove either of these
    "queue_skew_ticks", "queue_skew_bps", "queue_skew_thresh",
    # the graded staircase -- REQUIRED as of 2026-09-14; the sweep cannot run
    # without it, and a clobber that drops it must fail loudly, not silently
    "queue_skew_stairs",
    # size skew / taper
    "size_boost_mult", "size_boost_thresh",
    "enable_inv_taper", "inv_taper_k", "inv_taper_pov_mult",
    "inv_taper_floor", "inv_taper_both",
    # capacity cap and repricing
    "enable_pov_cap", "pov_cap_mult",
    "enable_run_reprice", "run_reprice_n", "run_reprice_ticks",
    "enable_aggr_lean", "aggr_lean_k", "aggr_lean_win_ms", "aggr_lean_obi_calm",
    # instrumentation
    "log_fill_state", "tol_ticks",
]

# KWARGS THE RUNNERS WILL PASS BUT micro_mm.py MAY NOT HAVE YET. Reported as
# PENDING, never as a failure, so an un-applied edit cannot block a run whose
# real contract is intact. Move a name up into REQUIRED_KWARGS once it lands.
PENDING_KWARGS = [
    # (empty) -- queue_skew_stairs landed 2026-09-14 and moved up to REQUIRED.
]

# a running tally so every failure is reported, not just the first
FAILURES = []


def check(name, ok, detail=""):
    # one line per check, so the output reads as a report
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    # remember failures for the exit code
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 74)
    print("micro_mm contract test")
    print("=" * 74)

    # ---- 1. SIGNATURE -------------------------------------------------------
    # the constructor's declared parameters
    sig = inspect.signature(MicrostructureMM.__init__)
    # the names it accepts
    have = set(sig.parameters)
    # anything the runners pass that the constructor no longer has
    missing = [k for k in REQUIRED_KWARGS if k not in have]
    # this is the clobber guard
    check("constructor accepts every runner kwarg",
          not missing,
          f"MISSING: {missing}" if missing else f"{len(REQUIRED_KWARGS)} checked")
    # pending kwargs are REPORTED, never failed -- they gate a sweep, not a run
    pend_have = [k for k in PENDING_KWARGS if k in have]
    # anything still absent from micro_mm.py
    pend_miss = [k for k in PENDING_KWARGS if k not in have]
    # one informational line so the state is never a surprise
    print(f"  [INFO] pending kwargs present: {pend_have or 'none'}"
          + (f"   NOT YET IN micro_mm.py: {pend_miss}" if pend_miss else ""))

    # ---- 2. QUOTES SIGNATURE ------------------------------------------------
    # mm_backtest calls strat.quotes(bb, bq, ba, aq, pos, depth=_depth)
    qsig = inspect.signature(MicrostructureMM.quotes)
    # the parameter names quotes() declares
    qhave = list(qsig.parameters)
    # positional contract
    check("quotes() positional args are (self, bb, bq, ba, aq, pos)",
          qhave[:6] == ["self", "bb", "bq", "ba", "aq", "pos"],
          f"got {qhave[:6]}")
    # the depth kwarg the engine passes unconditionally
    check("quotes() accepts depth=", "depth" in qhave,
          "mm_backtest passes depth on every call")

    # ---- 3. CONSTRUCT -------------------------------------------------------
    # build a kwarg set from the signature's own defaults, so this test does not
    # have to know values it was never told
    kw = {}
    # anything without a default must be supplied explicitly
    needs_value = [n for n, p in sig.parameters.items()
                   if n != "self" and p.default is inspect.Parameter.empty]
    # the few we know how to supply
    KNOWN = {"session_ms": SESSION, "size": 100, "max_inv": 3000,
             "session_scale": 1.0, "tick": 0.01}
    # fill them
    for n in needs_value:
        # a required parameter this test does not know about is itself a finding
        if n not in KNOWN:
            check(f"constructor requires unknown parameter '{n}'", False,
                  "add it to KNOWN in this file, then re-run")
            return
        kw[n] = KNOWN[n]
    # PIN the session window whether or not the signature makes it required.
    # (2026-09-14: the geometry block read kw["session_ms"], which only existed
    # when the parameter had no default -- KeyError otherwise. The test must not
    # depend on whether a parameter carries a default.)
    kw["session_ms"] = SESSION
    # the full sweep configuration, exactly as universe_expand builds it
    kw.update(dict(obi_defensive=True, use_microprice=False, tol_ticks=0.0,
                   obi_throttle=True, obi_throttle_thresh=0.15,
                   throttle_frac=0.5, throttle_hold_ms=300.0,
                   queue_skew_ticks=2.0, queue_skew_thresh=0.15,
                   exit_ticks_inside=1))
    # instantiate
    try:
        strat = MicrostructureMM(**kw)
        check("instantiates with the full sweep kwarg set", True)
    except Exception as e:
        check("instantiates with the full sweep kwarg set", False, repr(e))
        return
    # the skew must actually be stored, not silently swallowed
    check("queue_skew_ticks is retained on the instance",
          getattr(strat, "queue_skew_ticks", None) == 2.0,
          f"got {getattr(strat, 'queue_skew_ticks', 'ABSENT')}")

    # ---- 4 & 5. GEOMETRY + NO-CROSS ----------------------------------------
    # a wide, clean, deliberately imbalanced book: 10 ticks between the touches
    bb, ba = 100.00, 100.10
    # bid-heavy: imb = 900/1000 = 0.90, well past the 0.15 threshold
    bq, aq = 900.0, 100.0
    # Everything below is BEST EFFORT. It calls quotes() directly, with no engine
    # behind it, so it can fail for reasons that say nothing about the contract.
    # Checks 1-5 are the hard gate; this section must therefore NEVER crash the
    # run. (2026-09-14: an earlier version left three lines outside the guard and
    # took the whole test down with a traceback, hiding the passes above it.)
    def geometry():
        # a wide, clean, deliberately imbalanced book: 10 ticks between the touches
        bb, ba = 100.00, 100.10
        # bid-heavy: imb = 900/1000 = 0.90, well past the 0.15 threshold
        bq, aq = 900.0, 100.0
        # the engine sets `now` from the event stream; supply it so triggers resolve
        strat.now = SESSION[0] + 60_000
        # a symmetric reference with the skew OFF, same everything else
        kw_off = dict(kw)
        # switch the skew off only
        kw_off["queue_skew_ticks"] = 0.0
        # build the reference strategy
        ref = MicrostructureMM(**kw_off)
        # same clock
        ref.now = strat.now
        # call both
        q_on = strat.quotes(bb, bq, ba, aq, 0.0, depth=None)
        q_off = ref.quotes(bb, bq, ba, aq, 0.0, depth=None)
        # it returned without raising
        check("quotes() callable without the engine", True)
        # both sides present in each, or the comparison is meaningless
        if not ({"BUY", "SELL"} <= set(q_on)) or not ({"BUY", "SELL"} <= set(q_off)):
            # say which side was missing rather than indexing into a KeyError
            check("both sides quoted in the test book", False,
                  f"skew-on={sorted(q_on)} skew-off={sorted(q_off)}")
            return
        # both sides quoted
        check("both sides quoted in the test book", True)
        # the displacement each side moved when the skew was switched on
        d_bid = q_on["BUY"][0] - q_off["BUY"][0]
        # and the same for the ask
        d_ask = q_on["SELL"][0] - q_off["SELL"][0]
        # PARALLEL SHIFT: both sides move the same way by the same amount
        check("queue skew translates BOTH quotes equally (parallel shift)",
              abs(d_bid - d_ask) < 1e-9,
              f"bid moved {d_bid:+.4f}, ask moved {d_ask:+.4f}")
        # the quoted width with the skew on
        w_on = q_on["SELL"][0] - q_on["BUY"][0]
        # and with it off
        w_off = q_off["SELL"][0] - q_off["BUY"][0]
        # a translation cannot change the width
        check("quoted width unchanged by the skew",
              abs(w_on - w_off) < 1e-9,
              f"width {w_off:.4f} -> {w_on:.4f}")
        # bid-heavy must translate UPWARD (toward the side the book leans)
        check("bid-heavy book translates the pair upward", d_bid > 0,
              f"bid moved {d_bid:+.4f}")
        # POST-ONLY: neither quote may cross, however hard the skew pushes
        check("post-only holds: bid stays below the ask",
              q_on["BUY"][0] <= ba - strat.tick + 1e-9,
              f"bid {q_on['BUY'][0]:.2f} vs ask {ba:.2f}")
        # the mirror on the sell side
        check("post-only holds: ask stays above the bid",
              q_on["SELL"][0] >= bb + strat.tick - 1e-9,
              f"ask {q_on['SELL'][0]:.2f} vs bid {bb:.2f}")


    # ---- 6. STAIRCASE: the rung actually selected, and the ladder guard -------
    # The stair arm is the only config where the skew DISTANCE is a function of
    # the imbalance. A silent fall-through to rung 0 would still produce a
    # plausible-looking run -- it would just be QT_2t wearing a STAIR label and
    # the sweep's whole conclusion would be wrong. So assert the selection.
    def staircase():
        # the ladder the sweep actually runs
        ladder = [(0.15, 2.0), (0.20, 3.0), (0.25, 4.0)]
        # the stair config: gate MUST equal rungs[0][0] or __init__ raises
        kw_st = dict(kw)
        # the fixed-tick path must be off, or the test proves nothing
        kw_st["queue_skew_ticks"] = 0.0
        # the ladder
        kw_st["queue_skew_stairs"] = ladder
        # the gate, matching rung 0
        kw_st["queue_skew_thresh"] = 0.15
        # build it
        st = MicrostructureMM(**kw_st)
        # same clock as the rest of the block
        st.now = SESSION[0] + 60_000
        # a reference with no skew at all, to measure displacement against
        kw_ref = dict(kw)
        # skew fully off
        kw_ref["queue_skew_ticks"] = 0.0
        # build the reference
        rf = MicrostructureMM(**kw_ref)
        # same clock
        rf.now = st.now
        # one tick, for converting displacement back into rungs
        tk = st.tick
        # A WIDE book on purpose. quotes() clamps the favorable half at
        # max(0.0, half - _qs), so on a narrow book a 4-tick rung can bind the
        # clamp and the measured displacement would be smaller than the rung --
        # a false failure that says nothing about the ladder. 40 ticks between
        # the touches puts `half` near 20 ticks, well clear of the top rung.
        wb, wa = 100.00, 100.40
        # (imb, expected_ticks). The gate is on |imb - 0.5|, NOT on imb, so the
        # rung boundaries in imb terms are 0.65 / 0.70 / 0.75, and the test is
        # STRICTLY greater -- an imbalance sitting exactly on a rung does not
        # climb it. 0.60 -> |0.10|, below every rung: no skew at all.
        cases = [(0.60, 0.0), (0.68, 2.0), (0.73, 3.0), (0.80, 4.0)]
        # walk each case
        for imb, want in cases:
            # total book size is arbitrary; only the ratio matters
            tot = 1000.0
            # bid quantity produces the target imbalance
            bqi = imb * tot
            # ask quantity is the remainder
            aqi = tot - bqi
            # quote with the staircase on
            q_s = st.quotes(wb, bqi, wa, aqi, 0.0, depth=None)
            # and with it off
            q_r = rf.quotes(wb, bqi, wa, aqi, 0.0, depth=None)
            # if either side is missing the comparison is meaningless -- say so
            if not ({"BUY", "SELL"} <= set(q_s)) or not ({"BUY", "SELL"} <= set(q_r)):
                check(f"stair rung at imb={imb:.2f}", False,
                      f"one side unquoted: stair={sorted(q_s)} ref={sorted(q_r)}")
                continue
            # displacement of the bid, in ticks
            got = (q_s["BUY"][0] - q_r["BUY"][0]) / tk
            # the rung the ladder should have selected
            check(f"stair selects {want:g}t at imb={imb:.2f}",
                  abs(got - want) < 1e-6,
                  f"expected {want:g}t, got {got:.4f}t")
        # THE LADDER GUARD. A descending ladder silently inverts "last rung wins",
        # so __init__ must refuse it. If this check fails, a typo in a sweep
        # config would run for hours and produce a wrong answer instead of an error.
        kw_bad = dict(kw_st)
        # descending: 0.25 before 0.15
        kw_bad["queue_skew_stairs"] = [(0.25, 4.0), (0.15, 2.0)]
        # and the gate matches the FIRST entry, so only the ordering is wrong
        kw_bad["queue_skew_thresh"] = 0.25
        # it must raise
        try:
            MicrostructureMM(**kw_bad)
            check("descending ladder is rejected", False, "constructor accepted it")
        except ValueError:
            check("descending ladder is rejected", True)
        # THE GATE GUARD. rungs[0][0] must equal queue_skew_thresh, or the fire
        # test in quotes() and the ladder's bottom step disagree silently.
        kw_gap = dict(kw_st)
        # ladder starts at 0.15 ...
        kw_gap["queue_skew_stairs"] = ladder
        # ... but the gate is 0.20: rung 0 could never fire
        kw_gap["queue_skew_thresh"] = 0.20
        # it must raise
        try:
            MicrostructureMM(**kw_gap)
            check("rung0 != queue_skew_thresh is rejected", False,
                  "constructor accepted it")
        except ValueError:
            check("rung0 != queue_skew_thresh is rejected", True)

    # run it under a blanket guard: nothing in here may take the process down
    try:
        geometry()
    # a bare Exception is deliberate -- this block's job is to report, not to judge
    except Exception as e:
        # name the failure exactly
        check("geometry checks ran", False, repr(e))
        # and make clear the contract itself is unaffected
        print("\n  NOTE: the contract checks above are the gate and they stand.")
        print("        The geometry block needs engine state the constructor does")
        print("        not set. Paste the error above if you want it diagnosed.")

    # same blanket guard: the staircase block must never take the process down
    try:
        staircase()
    # deliberate bare Exception -- report, do not judge
    except Exception as e:
        check("staircase checks ran", False, repr(e))


# entry point
if __name__ == "__main__":
    main()
    print("=" * 74)
    # a single decisive line, and a non-zero exit so CI or a shell && can gate on it
    if FAILURES:
        print(f"CONTRACT BROKEN -- {len(FAILURES)} failed: {FAILURES}")
        sys.exit(1)
    print("CONTRACT INTACT")
    sys.exit(0)
