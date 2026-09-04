"""Focused sign-off tests for the experimental multi-level OFI path."""

from micro_mm import MicrostructureMM
from mm_backtest import Book, Order


T0, T1 = 0, 6 * 3600 * 1000
SEGS = [(T0, T1)]


def strategy(depth_levels=3, enabled=True, threshold=0.10):
    return MicrostructureMM(
        size=50,
        max_inv=500,
        session_ms=(T0, T1),
        session_segments=SEGS,
        session_scale=1.0,
        use_microprice=False,
        ofi_defensive=enabled,
        ofi_depth_levels=depth_levels,
        ofi_window_ev=2,
        ofi_window_s=0.5,
        ofi_defensive_thresh=threshold,
        ofi_defensive_ticks=2.0,
    )


def quote(s, t, depth):
    s.now = t
    bids, asks = depth
    bb, bq = bids[0]
    ba, aq = asks[0]
    return s.quotes(bb, bq, ba, aq, 0, depth=depth)


def test_rank_increment_signs():
    inc = MicrostructureMM._ofi_rank_increment
    assert inc((99.95, 400), (99.95, 250), "BUY") == -150.0
    assert inc((100.10, 400), (100.10, 250), "SELL") == 150.0
    assert inc((99.95, 400), (100.00, 300), "BUY") == 300.0
    assert inc((100.10, 400), (100.15, 300), "SELL") == 400.0
    print("1 PASS rank increments: bid/ask signs mirror correctly")


def test_deep_flow_visible_before_l1():
    s = strategy(depth_levels=3)
    t = T0 + 20 * 60_000
    # The touch never changes. Only level-2 ask depth is consumed, which should
    # produce positive (buying) deep OFI while an L1 signal would remain zero.
    for i, q2 in enumerate((500, 420, 340, 260)):
        depth = (
            ((100.00, 500), (99.95, 500), (99.90, 500)),
            ((100.20, 500), (100.25, q2), (100.30, 500)),
        )
        out = quote(s, t + i * 300, depth)
    sig = s._ofi_signal()
    assert sig is not None and sig > 0, sig
    assert "SELL" in out
    print(f"2 PASS deep precursor: stable L1, depleted L2 ask -> OFI={sig:+.3f}")


def test_deep_retreat_is_one_sided():
    s_on = strategy(depth_levels=3)
    s_off = strategy(depth_levels=3, enabled=False)
    t = T0 + 20 * 60_000
    for i, q2 in enumerate((500, 420, 340, 260)):
        depth = (
            ((100.00, 500), (99.95, 500), (99.90, 500)),
            ((100.20, 500), (100.25, q2), (100.30, 500)),
        )
        on = quote(s_on, t + i * 300, depth)
        off = quote(s_off, t + i * 300, depth)
    assert on["BUY"] == off["BUY"], (on, off)
    assert on["SELL"][0] > off["SELL"][0], (on, off)
    print("3 PASS retreat-only: deep buying widens ask and leaves bid unchanged")


def test_deep_requires_payload():
    s = strategy(depth_levels=5)
    s.now = T0 + 20 * 60_000
    try:
        s.quotes(100.00, 500, 100.20, 500, 0)
    except ValueError as exc:
        assert "deep OFI requires" in str(exc)
    else:
        raise AssertionError("deep OFI silently ran without depth")
    print("4 PASS integration guard: missing depth fails loudly")


def test_book_visible_depth_view():
    book = Book()
    book.o = {
        "b1": Order("BUY", 100.00, 300),
        "b1_hidden": Order("BUY", 100.00, 100),
        "__NEG_BUY_100.0": Order("BUY", 100.00, -50),
        "b2": Order("BUY", 99.95, 200),
        "a1": Order("SELL", 100.10, 250),
        "a2": Order("SELL", 100.15, 350),
        "__AGG_BUY": Order("BUY", 99.90, 10_000),
    }
    bids, asks = book.depth_levels(2)
    assert bids == ((100.00, 350.0), (99.95, 200.0)), bids
    assert asks == ((100.10, 250.0), (100.15, 350.0)), asks
    print("5 PASS book view: levels ranked/netted; beyond-L10 aggregate excluded")


def main():
    test_rank_increment_signs()
    test_deep_flow_visible_before_l1()
    test_deep_retreat_is_one_sided()
    test_deep_requires_payload()
    test_book_visible_depth_view()
    print("\nALL DEEP OFI SIGN-OFF GATES PASS.")


if __name__ == "__main__":
    main()
