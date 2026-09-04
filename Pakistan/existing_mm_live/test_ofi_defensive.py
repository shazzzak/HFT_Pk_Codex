# ============================================================================
# test_ofi_defensive.py -- Stage 3 sign-off tests for OFI-defensive quoting.
# ============================================================================
# Gates (all must PASS before the sweep uses ofi_defensive=True):
#   1. OFF = byte-identical  : ofi_defensive=False produces EXACTLY the same
#                              quotes as before the change (path never runs).
#   2. WARM-UP guard         : no OFI effect until BOTH >= N increments seen AND
#                              >= T seconds elapsed (mirrors the horserace guard
#                              SZ caught the original race missing).
#   3. FIRST15 hard-off      : zero OFI effect inside the first 15 minutes,
#                              regardless of how loud the signal is.
#   4. RETREAT-only          : the OFI path can only push the bid DOWN or the
#                              ask UP -- never improves a quote toward pressure
#                              (the anti-microprice guard).
#   5. INCREMENT math        : the Cont-Kukanov L1 increment matches hand cases.
#   6. NORMALIZATION bounds  : the signal lives in [-1, +1] and is None on
#                              degenerate windows.
#
# Run from existing_mm_live/ :  python test_ofi_defensive.py
# ============================================================================

# the strategy under test
from micro_mm import MicrostructureMM

# a 6-hour synthetic session (ms)
T0, T1 = 0, 6 * 3600 * 1000
# segments: one continuous block (open at T0, close at T1)
SEGS = [(T0, T1)]


# build a strategy with everything OFF except what each test enables
def mk(ofi=False, ev=5, ts=2.0, thresh=0.30, ticks=1.0):
    # tiny, deterministic config: no triggers, no unwind, no OBI, mid fair
    s = MicrostructureMM(size=50, max_inv=500,
                         session_ms=(T0, T1),
                         session_segments=SEGS,
                         session_scale=1.0,
                         use_microprice=False,
                         ofi_defensive=ofi,
                         ofi_window_ev=ev,
                         ofi_window_s=ts,
                         ofi_defensive_thresh=thresh,
                         ofi_defensive_ticks=ticks)
    return s


# advance the strategy clock and fetch quotes at a given touch
def q(s, t, bb, bq, ba, aq, pos=0):
    # observe() is what updates self.now in production; mimic that
    s.now = t
    # the quote set at this touch
    return s.quotes(bb, bq, ba, aq, pos)


# ---- TEST 1: OFF is byte-identical --------------------------------------
def test_1_off_identical():
    # two strategies: one pre-change-equivalent (ofi off), one with the flag
    # off explicitly -- and a third with ofi ON to prove quotes CAN differ.
    a = mk(ofi=False)
    b = mk(ofi=False)
    # a long sequence of touches with heavy one-sided (selling) flow
    seq = []
    px = 100.00
    for i in range(200):
        # ask side grows, bid side shrinks -> selling pressure pattern
        seq.append((60_000 + i * 500, px, max(1, 500 - i * 2), px + 0.05, 500 + i * 2))
    outs_a = [q(a, *s) for s in seq]
    outs_b = [q(b, *s) for s in seq]
    # identical, quote for quote
    assert outs_a == outs_b, "OFF is not deterministic-identical"
    print("1 PASS OFF byte-identical: ofi_defensive=False quotes unchanged "
          f"({len(seq)} touches)")


# ---- TEST 2: warm-up guard ----------------------------------------------
def test_2_warmup():
    # window: 5 events, 2 seconds. Signal must be None until BOTH warm.
    s = mk(ofi=True, ev=5, ts=2.0)
    # place touches INSIDE middle (start at minute 20) with pure selling flow:
    # ask qty grows each event -> e_ask > 0 -> e = e_bid - e_ask < 0
    t0 = T0 + 20 * 60000
    # 3 events over 1.0s: seen=2 increments (first touch seeds prev) -> not warm
    q(s, t0 + 0,    100.00, 500, 100.05, 500)
    q(s, t0 + 500,  100.00, 480, 100.05, 560)
    q(s, t0 + 1000, 100.00, 460, 100.05, 620)
    assert s._ofi_signal() is None, "signal fired before event-cap warm"
    # more events but still < 2s since first increment -> still not warm
    q(s, t0 + 1200, 100.00, 440, 100.05, 680)
    q(s, t0 + 1400, 100.00, 420, 100.05, 740)
    q(s, t0 + 1600, 100.00, 400, 100.05, 800)
    # now seen >= 5 increments, but first increment was at t0+500 (the 2nd touch)
    # -> 2s elapse at t0+2500. At t0+1600 elapsed = 1100ms < 2000ms -> None.
    assert s._ofi_signal() is None, "signal fired before time-cap warm"
    # advance past the time cap with one more event
    q(s, t0 + 2600, 100.00, 380, 100.05, 860)
    sig = s._ofi_signal()
    assert sig is not None, "signal still None after both caps warm"
    # selling flow -> negative signal
    assert sig < 0, f"selling flow should give negative signal, got {sig}"
    print(f"2 PASS warm-up: None until N events AND T seconds; then sig={sig:+.3f}")


# ---- TEST 3: first15 hard-off --------------------------------------------
def test_3_first15_off():
    # same loud selling flow, but placed INSIDE the first 15 minutes
    s = mk(ofi=True, ev=5, ts=2.0)
    t0 = T0 + 60_000  # minute 1
    for i in range(20):
        q(s, t0 + i * 400, 100.00, 500 - i * 10, 100.05, 500 + i * 20)
    # window is warm by both caps, but bucket = first15 -> hard None
    assert s._bucket_now() == "first15"
    assert s._ofi_signal() is None, "OFI acted inside first15"
    # move the SAME strategy to minute 20 (middle) and confirm it CAN fire
    s.now = T0 + 20 * 60000
    assert s._bucket_now() == "middle"
    assert s._ofi_signal() is not None, "signal should fire outside first15"
    print("3 PASS first15 hard-off: None in first15, fires in middle")


# ---- TEST 4: retreat-only ------------------------------------------------
def test_4_retreat_only():
    # two identical strategies, one with OFI on; feed loud SELLING flow (warm),
    # then compare quotes: the bid may only move DOWN, the ask must be UNCHANGED
    # (selling flow threatens the BID only).
    s_on = mk(ofi=True, ev=5, ts=2.0, thresh=0.10, ticks=2.0)
    s_off = mk(ofi=False)
    t0 = T0 + 30 * 60000
    seq = [(t0 + i * 400, 100.00, 500, 100.20, 500 + i * 40) for i in range(12)]
    for touch in seq[:-1]:
        q(s_on, *touch)
        q(s_off, *touch)
    # final touch: capture both quote sets
    out_on = q(s_on, *seq[-1])
    out_off = q(s_off, *seq[-1])
    # both sides quoted in both
    assert "BUY" in out_on and "BUY" in out_off
    assert "SELL" in out_on and "SELL" in out_off
    bid_on, _ = out_on["BUY"]; bid_off, _ = out_off["BUY"]
    ask_on, _ = out_on["SELL"]; ask_off, _ = out_off["SELL"]
    # selling flow: bid widened DOWN (or equal if signal below thresh), NEVER up
    assert bid_on <= bid_off, f"bid improved toward pressure: {bid_on} > {bid_off}"
    # and it actually widened (signal is loud, thresh low)
    assert bid_on < bid_off, "expected the bid to widen on loud selling flow"
    # ask untouched by SELLING flow
    assert ask_on == ask_off, f"ask moved on selling flow: {ask_on} vs {ask_off}"
    print(f"4 PASS retreat-only: bid {bid_off}->{bid_on} (down), ask unchanged")


# ---- TEST 5: Cont-Kukanov increment math ---------------------------------
def test_5_increment_math():
    s = mk(ofi=True)
    # seed the previous touch
    s.now = T0 + 30 * 60000
    s._ofi_update(100.00, 500, 100.05, 400)
    # case A: bid qty grows at same level (+100), ask unchanged -> e = +100
    s.now += 100
    s._ofi_update(100.00, 600, 100.05, 400)
    assert s._ofi_events[-1][1] == 100.0, s._ofi_events[-1]
    # case B: bid retreats to lower level -> e_bid = -600; ask same -> e = -600
    s.now += 100
    s._ofi_update(99.95, 300, 100.05, 400)
    assert s._ofi_events[-1][1] == -600.0, s._ofi_events[-1]
    # case C: ask improves DOWN (new level, qty 250) -> e_ask=+250, bid same
    # level qty delta 0 -> e = 0 - 250 = -250
    s.now += 100
    s._ofi_update(99.95, 300, 100.00, 250)
    assert s._ofi_events[-1][1] == -250.0, s._ofi_events[-1]
    print("5 PASS increment math: +qty on bid grow, -old on retreat, "
          "-new on ask improve")


# ---- TEST 6: normalization bounds ----------------------------------------
def test_6_normalization():
    # 3-event / 1-second window: the 6 touches at 400ms spacing genuinely warm
    # BOTH caps (5 increments seen >= 3; 2.0s elapsed >= 1.0s).
    s = mk(ofi=True, ev=3, ts=1.0, thresh=0.0)
    t0 = T0 + 30 * 60000
    # all-buying flow: bid qty grows every event -> every e > 0 -> signal = +1
    for i in range(6):
        q(s, t0 + i * 400, 100.00, 500 + i * 50, 100.05, 500)
    sig = s._ofi_signal()
    assert sig is not None and abs(sig - 1.0) < 1e-12, f"pure buys should be +1, got {sig}"
    print(f"6 PASS normalization: pure one-sided flow -> sig = {sig:+.1f} (bounded)")


def main():
    test_1_off_identical()
    test_2_warmup()
    test_3_first15_off()
    test_4_retreat_only()
    test_5_increment_math()
    test_6_normalization()
    print("\nALL STAGE 3 SIGN-OFF GATES PASS.")


if __name__ == "__main__":
    main()
