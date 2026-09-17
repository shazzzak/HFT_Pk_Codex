# ============================================================================
# test_stale_feed.py -- does the engine stand down when the feed goes dark?
# ============================================================================
# WHAT THIS IS TESTING, and why it exists.
#
# The capture has stretches during continuous trading where NO message of any
# kind arrives -- no trade, no book update, no heartbeat -- for 7 to 33
# seconds. Measured on NRL and MLCF over three days: 21 such stretches, 444
# seconds in total, recurring on a ~47-minute cycle. The PSX feed heartbeats
# every 3 seconds per channel, so a 10-second silence is our receiver having
# stopped, not the market having gone quiet.
#
# Before 2026-09-17 the engine did NOTHING about this, and "nothing" is the
# problem. run() is event-driven, so no events means no requotes and no fill
# checks: our orders simply rested through the blackout and could not be hit.
# That is the single most flattering assumption available -- free queue
# position through exactly the window in which a real book moved without us.
#
# Two separate mechanisms are under test here:
#   THE GUARD     -- on the first event after a silence, pull everything and
#                    refuse to quote until a SNAPSHOT restores the book.
#   THE LEDGER    -- record each window, clip it to the session, and size the
#                    adverse selection the backtest was spared.
#
# Run from existing_mm_live/:
#   caffeinate -is python test_stale_feed.py
# ============================================================================

# exit codes only
import sys
# the engine, the latency model, and the Order record the Book holds
from mm_backtest import Backtester, LatencyModel, Order


class Stub:
    """A strategy that always wants to quote, so any refusal is the engine's."""
    # PSX is a flat one paisa
    tick = 0.01

    def observe(self, *args):
        # the engine calls this on every event; nothing here needs it
        return None

    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # always wants a bid at 289.00 for 50 shares, whatever the book says
        return {"BUY": (289.00, 50)}


class Upd:
    """One historical ORDER_ADD, with the fields the engine reads off it."""

    def __init__(self, ts, oid, side, px, qty):
        # exchange time and capture time are the same in these tests: the
        # silences being tested are measured on the CAPTURE clock, and keeping
        # the two equal means each event's ts is the only number to reason about
        self.ts_exch = ts
        self.ts_cap = ts
        # the engine dispatches on this string
        self.event = "ORDER_ADD"
        # Book.add keys on this
        self.order_id = oid
        # and reads these three
        self.side = side
        self.price = px
        self.qty = qty


class Snap:
    """One pre-parsed exchange snapshot: a FULL book replacement."""

    def __init__(self, ts, key, bid, ask, bid_qty=500, ask_qty=300):
        # the two clocks, kept equal as above
        self.ts_exch = ts
        self.ts_cap = ts
        # run() looks the snapshot's levels up by this key
        self.snap_key = key
        # continuous trading, no published circuit band
        self.phase = "CONTINUOUS_AUCTION"
        self.limit_up = None
        self.limit_dn = None
        # (side, price, qty, order_ids, order_qtys) -- no disclosed ids, so the
        # whole level parks as one hidden lump, which is all the touch needs
        self.levels = [("BUY", bid, bid_qty, "", ""),
                       ("SELL", ask, ask_qty, "", "")]
        # no whole-side aggregate rows, so no deep residual lump is built
        self.agg = {}
        # this message carries visible levels, so it replaces the book
        self.has_visible = True


def build(bid=289.00, ask=289.02, **over):
    """An engine with a hand-built two-sided book at the given touch."""
    # every source of randomness pinned: 100ms each way, no tail
    cfg = dict(latency_model=LatencyModel(decision_ms=0.0,
                                          wire_out_median_ms=100.0,
                                          wire_out_tail_ms=0.0,
                                          wire_in_median_ms=100.0,
                                          wire_in_tail_ms=0.0, tail_prob=0.0),
               at_price_mode="queue", fill_on_crossing_adds=False,
               log_equity=False, session=(0, 10 ** 9))
    # whatever this case changes
    cfg.update(over)
    # the engine
    bt = Backtester(Stub(), cfg)
    # the book, built the way Book itself stores it
    bt.book.o.clear()
    bt.book.o["H1"] = Order("BUY", bid, 500)
    bt.book.o["H2"] = Order("SELL", ask, 300)
    # continuous trading, no published band
    bt.book.phase = "CONTINUOUS_AUCTION"
    bt.book.limit_up = None
    bt.book.limit_dn = None
    # ready to drive
    return bt


def drive(bt, objs, snaps=None):
    """Run the engine over a list of event objects, in the order given."""
    # run() takes (ts_exch, _, _, kind, obj) tuples; the two middle fields are
    # tie-breakers the merge uses and this loop never reads
    events = []
    # one tuple per object, with the kind derived from its type
    for o in objs:
        # a Snap is kind "S", an Upd is kind "U"
        kind = "S" if isinstance(o, Snap) else "U"
        # the tuple, with zeroes in the two unread slots
        events.append((o.ts_exch, 0, 0, kind, o))
    # the snapshot lookup table, keyed exactly as run() looks it up
    return bt.run(events, snaps or {})


# every (name, passed?) pair
results = []


def check(name, condition, detail=""):
    # keep it for the summary, and show it as it happens
    results.append((name, bool(condition)))
    print(("PASS  " if condition else "FAIL  ") + name)
    # a failure explains itself on the next line
    if not condition and detail:
        print("        " + detail)


# ==========================================================================
# PART 1 -- THE GUARD. Does a stale flag actually stop the quoting?
# ==========================================================================
print("PART 1 -- THE GUARD")

# ---- a clean engine quotes normally --------------------------------------
bt = build()
bt._requote(1000)
check("a live feed quotes", len(bt.pending) == 1)
check("and counts no stale requote",
      bt.stats.get("stale_feed_requotes", 0) == 0)

# ---- a stale flag sends nothing ------------------------------------------
bt = build()
# the flag the detector sets; set here directly so this case tests the GUARD
# alone and does not depend on the detector being right
bt._feed_stale = True
bt._requote(1000)
check("a stale feed sends NOTHING", len(bt.pending) == 0)
check("and is counted", bt.stats.get("stale_feed_requotes") == 1)

# ---- an existing quote is PULLED, not left resting -----------------------
bt = build()
bt._requote(1000)
# let the order land
bt._activate_until(2000)
check("an order is resting to begin with", "BUY" in bt.work)
# the feed goes dark under us
bt._feed_stale = True
bt._requote(3000)
check("the resting quote is PULLED, not left exposed through the blackout",
      len(bt.pending) == 1 and bt.pending[0][2] == "CANCEL")

# ==========================================================================
# PART 2 -- THE DETECTOR. Does a silence in the event stream set the flag?
# ==========================================================================
print("\nPART 2 -- THE DETECTOR")

# ---- events one second apart are not a silence ---------------------------
bt = build()
drive(bt, [Upd(1000, "A1", "BUY", 288.90, 10),
           Upd(2000, "A2", "BUY", 288.90, 10),
           Upd(3000, "A3", "BUY", 288.90, 10)])
check("one-second spacing detects nothing",
      bt.stats.get("stale_feed_windows", 0) == 0)

# ---- a five-second gap is under the seven-second threshold ---------------
bt = build()
drive(bt, [Upd(1000, "B1", "BUY", 288.90, 10),
           Upd(6000, "B2", "BUY", 288.90, 10)])
check("a 5s gap is under the 7s threshold and is NOT a window",
      bt.stats.get("stale_feed_windows", 0) == 0,
      f"got {bt.stats.get('stale_feed_windows', 0)}")

# ---- a ten-second gap is --------------------------------------------------
bt = build()
drive(bt, [Upd(1000, "C1", "BUY", 288.90, 10),
           Upd(11000, "C2", "BUY", 288.90, 10)])
check("a 10s gap IS a window", bt.stats.get("stale_feed_windows", 0) == 1)
check("and its length is recorded as 10 seconds",
      abs(bt.stats.get("stale_feed_seconds", 0.0) - 10.0) < 1e-9,
      f"got {bt.stats.get('stale_feed_seconds')}")

# ---- the threshold is configurable ---------------------------------------
bt = build(stale_feed_seconds=3.0)
drive(bt, [Upd(1000, "D1", "BUY", 288.90, 10),
           Upd(6000, "D2", "BUY", 288.90, 10)])
check("a 5s gap IS a window at a 3s threshold",
      bt.stats.get("stale_feed_windows", 0) == 1)

# ---- zero disables it entirely, reproducing every pre-2026-09-17 run -----
bt = build(stale_feed_seconds=0.0)
drive(bt, [Upd(1000, "E1", "BUY", 288.90, 10),
           Upd(61000, "E2", "BUY", 288.90, 10)])
check("stale_feed_seconds=0 detects nothing, however long the gap",
      bt.stats.get("stale_feed_windows", 0) == 0)
check("and never refuses a quote",
      bt.stats.get("stale_feed_requotes", 0) == 0)

# ==========================================================================
# PART 3 -- WHAT CLEARS IT. Only a snapshot may, because only a snapshot
# replaces the book wholesale.
# ==========================================================================
print("\nPART 3 -- WHAT CLEARS THE FLAG")

# ---- an incremental update does NOT clear it -----------------------------
bt = build()
drive(bt, [Upd(1000, "F1", "BUY", 288.90, 10),
           Upd(11000, "F2", "BUY", 288.90, 10),
           Upd(12000, "F3", "BUY", 288.90, 10)])
check("an update after the silence does NOT clear the flag",
      bt._feed_stale is True,
      "an incremental update applied to a stale book leaves it stale")
check("so every requote after the silence is refused",
      bt.stats.get("stale_feed_requotes", 0) == 2,
      f"got {bt.stats.get('stale_feed_requotes', 0)} for the 2 post-silence "
      f"events")

# ---- a snapshot DOES clear it --------------------------------------------
bt = build()
# the snapshot the run loop will look up, returning the book unchanged
snaps = {"K1": Snap(12000, "K1", 289.00, 289.02)}
drive(bt, [Upd(1000, "G1", "BUY", 288.90, 10),
           Upd(11000, "G2", "BUY", 288.90, 10),
           snaps["K1"],
           Upd(13000, "G3", "BUY", 288.90, 10)], snaps)
check("a snapshot clears the flag", bt._feed_stale is False)
check("and quoting resumes after it",
      bt.stats.get("stale_feed_requotes", 0) == 1,
      f"only the one event between the silence and the snapshot should have "
      f"been refused; got {bt.stats.get('stale_feed_requotes', 0)}")

# ==========================================================================
# PART 4 -- THE LEDGER. What the window cost, and what it was handed.
# ==========================================================================
print("\nPART 4 -- THE LEDGER")

# An order resting at 289.00 when the feed dies; the market comes back at
# 288.50 / 288.52, a mid of 288.51. Anyone could have sold into our bid at
# 289.00 during the blackout, leaving us long at 289.00 against a market
# at 288.51. The gift is 50 shares x 0.49 = 24.50 PKR.
bt = build()
# the snapshot that brings the market back 49 paisa lower
snaps = {"K2": Snap(12000, "K2", 288.50, 288.52)}
# NOTE the order ids: "H1" and "H2" are the ids build() gave the historical
# book's own two orders, and Book.add overwrites by id. Reusing them here
# would silently replace the offer side with a bid and leave the book
# one-sided -- which is exactly what this test did on its first run.
drive(bt, [Upd(1000, "P1", "BUY", 288.90, 10),
           Upd(2000, "P2", "BUY", 288.90, 10),
           snaps["K2"]], snaps)
# exactly one window, and it is closed
check("one window was ledgered", len(bt.blind_windows) == 1)
# the record
w = bt.blind_windows[0] if bt.blind_windows else {}
check("the pre-silence mid is the market we left",
      w.get("mid_before") is not None
      and abs(w["mid_before"] - 289.01) < 1e-9,
      f"got {w.get('mid_before')}, expected 289.01")
check("the post-silence mid is the market we came back to",
      w.get("mid_after") is not None
      and abs(w["mid_after"] - 288.51) < 1e-9,
      f"got {w.get('mid_after')}, expected 288.51")
check("our resting bid was captured",
      w.get("resting") == [("BUY", 289.00, 50.0)],
      f"got {w.get('resting')}")
check("the spared adverse selection is 50 x 0.49 = 24.50 PKR",
      w.get("spared_adverse_pkr") is not None
      and abs(w["spared_adverse_pkr"] - 24.50) < 1e-6,
      f"got {w.get('spared_adverse_pkr')}")
check("and it is accumulated onto the stats",
      abs(bt.stats.get("blind_spared_adverse_pkr", 0.0) - 24.50) < 1e-6)
# no pending record left open
check("no record is left waiting for its exit mid",
      bt._pending_blind is None)

# ---- a move IN OUR FAVOUR is not a gift ----------------------------------
# Same setup, but the market comes back HIGHER. Our bid at 289.00 could only
# have been hit by someone selling to us at 289.00, and the market is now
# above that, so resting through the blackout cost us nothing.
bt = build()
snaps = {"K3": Snap(12000, "K3", 289.50, 289.52)}
drive(bt, [Upd(1000, "I1", "BUY", 288.90, 10),
           Upd(2000, "I2", "BUY", 288.90, 10),
           snaps["K3"]], snaps)
w = bt.blind_windows[0] if bt.blind_windows else {}
check("a favourable move books ZERO spared adverse selection",
      w.get("spared_adverse_pkr") == 0.0,
      f"got {w.get('spared_adverse_pkr')}")

# ---- an empty book on the far side is recorded as unmeasurable -----------
# The post-silence snapshot has a bid but no offer, so there is no mid to
# compare against. The record must say so rather than invent one.
bt = build()
# a one-sided snapshot: bid only
one_sided = Snap(12000, "K4", 288.50, 288.52)
one_sided.levels = [("BUY", 288.50, 500, "", "")]
snaps = {"K4": one_sided}
drive(bt, [Upd(1000, "J1", "BUY", 288.90, 10),
           Upd(2000, "J2", "BUY", 288.90, 10),
           one_sided], snaps)
w = bt.blind_windows[0] if bt.blind_windows else {}
check("a one-sided return leaves the exposure UNMEASURED, not zero",
      w.get("mid_after") is None and w.get("spared_adverse_pkr") is None,
      f"got mid_after={w.get('mid_after')}, "
      f"spared={w.get('spared_adverse_pkr')}")
check("and the record is still closed, so the next window is not lost",
      bt._pending_blind is None)

# ==========================================================================
# PART 5 -- SESSION CLIPPING. Only the in-session part of a silence may be
# taken out of a session denominator.
# ==========================================================================
print("\nPART 5 -- SESSION CLIPPING")

# The session opens at t=10000. A 10-second silence ending at t=12000 spans
# [2000, 12000], of which only [10000, 12000] -- two seconds -- is inside it.
bt = build(session=(10000, 10 ** 9))
drive(bt, [Upd(2000, "L1", "BUY", 288.90, 10),
           Upd(12000, "L2", "BUY", 288.90, 10)])
check("the FULL silence is still recorded as 10 seconds",
      abs(bt.stats.get("stale_feed_seconds", 0.0) - 10.0) < 1e-9,
      f"got {bt.stats.get('stale_feed_seconds')}")
check("but only the 2 seconds inside the session count against the clock",
      abs(bt.stats.get("stale_feed_seconds_in_session", 0.0) - 2.0) < 1e-9,
      f"got {bt.stats.get('stale_feed_seconds_in_session')}")

# A silence wholly before the open contributes nothing at all.
bt = build(session=(60000, 10 ** 9))
drive(bt, [Upd(2000, "M1", "BUY", 288.90, 10),
           Upd(12000, "M2", "BUY", 288.90, 10)])
check("a silence wholly outside the session contributes ZERO in-session "
      "seconds",
      bt.stats.get("stale_feed_seconds_in_session", 0.0) == 0.0,
      f"got {bt.stats.get('stale_feed_seconds_in_session')}")

# ==========================================================================
# PART 6 -- THE EOD FIELDS. The numbers a day's result is actually judged on.
# ==========================================================================
print("\nPART 6 -- THE EOD FIELDS")

# A session of exactly 20 seconds [0, 20000] containing ONE 10-second
# silence, then an event past the close to trigger the EOD report.
#
# The spacing after the silence matters: every gap is measured, so the run
# out to the close is one second at a time. A single jump straight to the
# post-close event would be a SECOND silence and would be counted as one --
# correctly, but it would stop this case testing what it says it tests.
bt = build(session=(0, 20000))
# 1s, then the 10s silence to 11s, then one event a second to 21s
_evs = [Upd(1000, "N0", "BUY", 288.90, 10)]
# the far side of the silence, and every second after it
for _t in range(11000, 21001, 1000):
    # a distinct id per event, so none overwrites the historical book
    _evs.append(Upd(_t, f"N{_t}", "BUY", 288.90, 10))
drive(bt, _evs)
# the report
eod = bt.eod or {}
check("an EOD report was produced", bool(eod))
check("the session is 20 seconds",
      abs(eod.get("session_seconds", 0) - 20.0) < 1e-9,
      f"got {eod.get('session_seconds')}")
check("10 of them were blind",
      abs(eod.get("blind_seconds", 0) - 10.0) < 1e-9,
      f"got {eod.get('blind_seconds')}")
check("so the effective session is 10 seconds",
      abs(eod.get("effective_seconds", 0) - 10.0) < 1e-9,
      f"got {eod.get('effective_seconds')}")
check("which is 50% of the session",
      abs(eod.get("blind_pct_of_session", 0) - 50.0) < 1e-9,
      f"got {eod.get('blind_pct_of_session')}")
check("and one window made it up", eod.get("blind_windows") == 1)
# THE POINT OF THE WHOLE EXERCISE: the per-hour number is computed on the
# clock the engine could see, not on the wall clock.
#
# HONEST NOTE ON WHAT THIS CHECK PROVES. This synthetic day trades nothing,
# so its P&L is zero and the two rates are both zero. The check therefore
# verifies the FORMULA -- that each rate divides by the denominator it
# claims to -- and not that the numbers differ. The denominators themselves
# are checked above, where they are 20s and 10s and demonstrably not equal.
_pe = eod.get("pnl_per_effective_hour")
_ps = eod.get("pnl_per_session_hour")
# what each should be, computed here from the day's own numbers
_exp_e = (eod.get("equity_liquidated", 0.0)
          / (eod.get("effective_seconds", 1.0) / 3600.0))
_exp_s = (eod.get("equity_liquidated", 0.0)
          / (eod.get("session_seconds", 1.0) / 3600.0))
check("P&L per EFFECTIVE hour divides by the effective clock (formula check; "
      "this synthetic day's P&L is zero)",
      _pe is not None and abs(_pe - _exp_e) < 1e-9,
      f"got {_pe}, expected {_exp_e}")
check("and P&L per SESSION hour divides by the full session clock",
      _ps is not None and abs(_ps - _exp_s) < 1e-9,
      f"got {_ps}, expected {_exp_s}")

# ---- summary -------------------------------------------------------------
print()
# anything that failed
failed = [n for n, ok in results if not ok]
# the headline
print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
# named, so a failure says what broke
for n in failed:
    print("  FAILED:", n)
# non-zero exit on any failure, so this can gate a script
sys.exit(1 if failed else 0)
