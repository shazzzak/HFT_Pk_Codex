"""Tests for the strategy adapter and the live config.

Every test here asserts a rejection or a conversion. The adapter has no logic
of its own worth testing -- it is a translator, and a translator is judged on
exactly two things: does it convert correctly, and does it refuse what it
cannot convert.
"""
# for the temp config files
import csv
# filesystem
import tempfile
from pathlib import Path

# the test runner
import pytest

# the domain objects
from core.model import BookLevel, BookSnapshot, DesiredQuotes, Side, Trade
# the venue value objects
from core.venue import PriceBand, SecurityPhase, SessionSegment, MarketPhase
# what is under test
from venues.psx import PSXVenue
from venues.psx_config import (DEFAULT_THRESH, FILE_PARAMS, ConfigError,
                               LiveConfig, preflight)
from venues.psx_strategy import MicroMMAdapter, TakerNotSupported


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def a_venue(band=None):
    """A PSX venue with one ordinary session and an optional price band."""
    # a single continuous segment, the shape of a non-Friday
    segments = [SessionSegment(start_ms=0, end_ms=1)]
    # the venue, with a band provider only when a band was asked for
    return PSXVenue(session_provider=lambda d: segments,
                    band_provider=(lambda s: band) if band is not None
                    else None)


class StubMM:
    """Stands in for MicrostructureMM.

    A stub rather than the real strategy on purpose: these tests are about the
    translation, and using the real object would mean any failure could be
    either the adapter or nine hundred lines of quoting logic.
    """
    # micro_mm advertises this so the engine knows to sync the circuit limits
    wants_limits = True

    def __init__(self, returns=None, tick=0.01):
        # what quotes() should hand back
        self._returns = returns if returns is not None else {}
        # the grid micro_mm believes it is on
        self.tick = tick
        # the switches the adapter inspects
        self.enable_age_cross = False
        self.allow_taker = False
        self.tol_ticks = 0.0
        self.ofi_depth_levels = 1
        self.want_taker_side = None
        # the circuit limits the adapter writes into
        self.limit_up = None
        self.limit_dn = None
        # every observe() call, so tests can assert the clock advanced
        self.observed = []
        # every quotes() call
        self.quoted = []

    def observe(self, kind, obj, ts_exch, mid):
        # record the call rather than doing anything with it
        self.observed.append((kind, obj, ts_exch, mid))

    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # record the inputs so unit conversion can be asserted
        self.quoted.append((bb, bq, ba, aq, pos, depth))
        # whatever the test configured
        return self._returns


def an_adapter(mm=None, band=None, reference_price_minor=28900):
    """An adapter over a stub, on a real PSX venue."""
    # default stub if the test did not supply one
    return MicroMMAdapter("PPL", a_venue(band), mm or StubMM(),
                          reference_price_minor=reference_price_minor)


def a_book(bid=28900, bid_qty=500, ask=28902, ask_qty=300, ts=1_000):
    """A two-sided book at a realistic PPL price."""
    # one level each side is enough for every test here
    return BookSnapshot(symbol="PPL", timestamp_ms=ts,
                        bids=(BookLevel(bid, bid_qty),),
                        asks=(BookLevel(ask, ask_qty),))


# ---------------------------------------------------------------------------
# the configurations the adapter must refuse
# ---------------------------------------------------------------------------
def test_age_cross_is_refused_at_construction():
    """Crossing the spread has no representation in QuoteIntent."""
    # a strategy configured to flatten aged inventory by crossing
    mm = StubMM()
    mm.enable_age_cross = True
    # the adapter must not be constructible at all
    with pytest.raises(TakerNotSupported, match="maker-only"):
        an_adapter(mm)


def test_allow_taker_is_refused_at_construction():
    """The same switch seen from the engine's side."""
    # the engine-facing flag, set without the feature flag
    mm = StubMM()
    mm.allow_taker = True
    # still refused
    with pytest.raises(TakerNotSupported):
        an_adapter(mm)


def test_taker_requested_after_construction_raises_rather_than_posting():
    """A switch flipped at runtime must not become a resting order."""
    # a legal adapter
    mm = StubMM(returns={"BUY": (289.00, 50)})
    adapter = an_adapter(mm)
    # something turns the taker path on mid-session
    mm.want_taker_side = "SELL"
    # the quote must not be translated into a passive intent
    with pytest.raises(TakerNotSupported, match="maker-only"):
        adapter.on_book(a_book(), position=0)


def test_strategy_side_hysteresis_is_refused():
    """tol_ticks and QuoteTolerance are the same mechanism; only one may run."""
    # a strategy with its own pegging tolerance
    mm = StubMM()
    mm.tol_ticks = 1.0
    # the adapter refuses rather than letting both suppress requotes
    with pytest.raises(ValueError, match="QuoteTolerance"):
        an_adapter(mm)


def test_disagreeing_tick_is_refused():
    """A strategy on a different grid produces off-grid prices."""
    # ten paisa where PSX ticks at one
    mm = StubMM(tick=0.10)
    # the mismatch must be caught before any price is produced
    with pytest.raises(ValueError, match="ticks at"):
        an_adapter(mm)


# ---------------------------------------------------------------------------
# unit conversion -- the highest-risk code in the adapter
# ---------------------------------------------------------------------------
def test_the_rounding_trap_that_costs_a_paisa():
    """int(px * 100) is wrong on 6.6% of PSX prices; int(round(...)) is not.

    Rs 280.03 times 100 is 28002.999999999996 as a double, so truncating gives
    28002 -- one tick below what the strategy asked for, on this price and not
    on the one next to it.
    """
    # demonstrate the trap exists, so this test fails loudly if it ever stops
    # being a trap rather than silently passing for the wrong reason
    assert int(280.03 * 100) == 28002
    # the adapter must produce the price the strategy actually asked for
    mm = StubMM(returns={"BUY": (280.03, 50)})
    quotes = an_adapter(mm).on_book(a_book(bid=28003, ask=28010), position=0)
    # one paisa matters: this is the whole reason the conversion is isolated
    assert quotes.bid.price_minor == 28003


def test_prices_round_trip_across_the_whole_psx_range():
    """EVERY paisa price from Rs 1 to Rs 5,000 survives the round trip.

    Every one, not a sample. A sample that skips by any fixed stride can miss
    the failing prices entirely -- they are not evenly spaced -- and a
    conversion test that passes because of its stride is worse than none.
    """
    # the adapter's two conversion helpers
    adapter = an_adapter()
    # how many prices would have been wrong under truncation, as a check that
    # this test is actually exercising the trap rather than a benign range
    would_truncate = 0
    # step through every paisa
    for price_minor in range(100, 500_000):
        # out to the strategy's units and back
        as_major = adapter._to_major(price_minor)
        # must be exactly what we started with, not close to it
        assert adapter._to_minor(as_major) == price_minor
        # count the ones a truncating conversion would have got wrong
        if int(as_major * 100) != price_minor:
            would_truncate += 1
    # the measured figure. If this ever changes, the platform's floating point
    # has changed and the conversion deserves re-examining rather than trust.
    assert would_truncate == 32_808


def test_a_price_off_the_minor_grid_is_refused():
    """Half a paisa is not a price; snapping it would hide a strategy bug."""
    # a price that is not a whole number of paisa
    mm = StubMM(returns={"BUY": (289.455, 50)})
    # the adapter refuses rather than choosing a direction to round
    with pytest.raises(ValueError, match="not on the minor-unit grid"):
        an_adapter(mm).on_book(a_book(bid=28945, ask=28950), position=0)


def test_the_book_reaches_the_strategy_in_major_units():
    """Paisa in, rupees out -- on every field, not just the price."""
    # a stub that records what it was given
    mm = StubMM()
    # a book at a known price
    an_adapter(mm).on_book(a_book(bid=28900, bid_qty=500,
                                  ask=28902, ask_qty=300), position=-40)
    # bid, bid size, ask, ask size, position, depth
    bb, bq, ba, aq, pos, depth = mm.quoted[0]
    # prices converted, quantities left as shares
    assert (bb, bq, ba, aq) == (289.00, 500, 289.02, 300)
    # the position passes through untouched, sign and all
    assert pos == -40
    # depth is not built when the strategy only uses one level
    assert depth is None


# ---------------------------------------------------------------------------
# the clock
# ---------------------------------------------------------------------------
def test_the_clock_advances_on_book_events_not_only_on_trades():
    """micro_mm's end-of-day triggers count against a clock only observe sets.

    Advancing it only on trades means that on a quiet name the strategy still
    believes it is mid-session when the bell has gone.
    """
    # a stub that records observe calls
    mm = StubMM()
    # a book event with no trade anywhere near it
    an_adapter(mm).on_book(a_book(ts=12_345_678), position=0)
    # observe must have been called with the book's exchange timestamp
    assert mm.observed[0][0] != "T"
    assert mm.observed[0][2] == 12_345_678


def test_a_trade_with_no_published_aggressor_still_advances_the_clock():
    """A feed that omits the side must not cost us the timestamp."""
    # a print with no aggressor
    mm = StubMM()
    an_adapter(mm).on_trade(Trade(symbol="PPL", timestamp_ms=99,
                                  price_minor=28901, quantity=10,
                                  aggressor=None))
    # the trade was still observed, at the right time
    kind, view, ts, _ = mm.observed[0]
    assert kind == "T" and ts == 99
    # and with a side micro_mm will ignore rather than one we invented
    assert view.aggressor_side not in ("BUY", "SELL")


def test_a_trade_with_an_aggressor_passes_the_side_through():
    """The flow signals are built on this field; it must arrive intact."""
    # a buyer-initiated print
    mm = StubMM()
    an_adapter(mm).on_trade(Trade(symbol="PPL", timestamp_ms=99,
                                  price_minor=28902, quantity=10,
                                  aggressor=Side.BUY))
    # the literal string micro_mm compares against
    assert mm.observed[0][1].aggressor_side == "BUY"
    assert mm.observed[0][1].qty == 10


# ---------------------------------------------------------------------------
# when not to quote
# ---------------------------------------------------------------------------
def test_a_one_sided_book_produces_no_quotes():
    """Normal at the open and on thin names -- not an error."""
    # a book with no offer at all
    book = BookSnapshot(symbol="PPL", timestamp_ms=1,
                        bids=(BookLevel(28900, 500),), asks=())
    # nothing desired on either side
    quotes = an_adapter().on_book(book, position=0)
    assert quotes == DesiredQuotes.flat("PPL")


def test_a_halt_makes_the_strategy_want_nothing_resting():
    """Desiring nothing emits cancels; the gateway merely refuses new orders."""
    # an adapter told the security is halted
    adapter = an_adapter(StubMM(returns={"BUY": (289.00, 50)}))
    adapter.on_phase(SecurityPhase(phase=MarketPhase.HALTED))
    # even though the strategy would have quoted, we want nothing resting
    assert adapter.on_book(a_book(), position=0) == DesiredQuotes.flat("PPL")


def test_an_all_day_suspension_also_stops_quoting():
    """Open market, suspended security: still nothing."""
    # continuous trading, but this name is suspended
    adapter = an_adapter(StubMM(returns={"BUY": (289.00, 50)}))
    adapter.on_phase(SecurityPhase(phase=MarketPhase.CONTINUOUS,
                                   suspended_all_day=True))
    # nothing resting
    assert adapter.on_book(a_book(), position=0) == DesiredQuotes.flat("PPL")


# ---------------------------------------------------------------------------
# quotes that must never reach the wire
# ---------------------------------------------------------------------------
def test_a_bid_that_would_cross_the_offer_is_refused():
    """micro_mm clips to stay inside the touch; this catches it if it stops."""
    # a bid at the offer
    mm = StubMM(returns={"BUY": (289.02, 50)})
    # the adapter refuses rather than sending a marketable order as passive
    with pytest.raises(ValueError, match="cross"):
        an_adapter(mm).on_book(a_book(bid=28900, ask=28902), position=0)


def test_an_offer_that_would_cross_the_bid_is_refused():
    """The mirror of the above."""
    # an offer at the bid
    mm = StubMM(returns={"SELL": (289.00, 50)})
    # refused
    with pytest.raises(ValueError, match="cross"):
        an_adapter(mm).on_book(a_book(bid=28900, ask=28902), position=0)


def test_a_zero_size_quote_is_refused():
    """Nothing resting is expressed by omitting the side, never by a zero."""
    # a quote for no shares at all
    mm = StubMM(returns={"BUY": (289.00, 0)})
    # refused, and the message names the symbol
    with pytest.raises(ValueError, match="PPL"):
        an_adapter(mm).on_book(a_book(), position=0)


def test_an_omitted_side_means_nothing_resting_there():
    """The protocol: absence is 'cancel', not 'leave it alone'."""
    # the strategy wants only a bid
    mm = StubMM(returns={"BUY": (289.00, 50)})
    quotes = an_adapter(mm).on_book(a_book(), position=0)
    # the bid is there
    assert quotes.bid is not None and quotes.bid.quantity == 50
    # and the offer is genuinely absent
    assert quotes.ask is None


# ---------------------------------------------------------------------------
# the published circuit limits
# ---------------------------------------------------------------------------
def test_published_bands_reach_the_strategy_in_its_own_units():
    """The lock trigger reads limit_up / limit_dn as rupees."""
    # a band a little either side of the touch
    band = PriceBand(upper_minor=31790, lower_minor=26010)
    mm = StubMM()
    an_adapter(mm, band=band).on_book(a_book(), position=0)
    # converted, not passed through as paisa
    assert mm.limit_up == 317.90
    assert mm.limit_dn == 260.10


def test_an_absent_band_leaves_the_limits_as_none():
    """PSX's 'no limit' sentinels are already None by the time they get here.

    micro_mm's lock trigger tests both limits for None before it fires, so
    None correctly disables the trigger. A number invented in their place
    would arm it against a boundary that does not exist.
    """
    # no band provider wired up at all
    mm = StubMM()
    an_adapter(mm).on_book(a_book(), position=0)
    # both must stay None
    assert mm.limit_up is None and mm.limit_dn is None


def test_one_sided_band_converts_only_the_bound_that_exists():
    """An up-limit with no down-limit is a real state, not a broken band."""
    # only the upper bound published
    band = PriceBand(upper_minor=31790, lower_minor=None)
    mm = StubMM()
    an_adapter(mm, band=band).on_book(a_book(), position=0)
    # the one that exists converts; the one that does not stays None
    assert mm.limit_up == 317.90 and mm.limit_dn is None


# ---------------------------------------------------------------------------
# the circuit band: clamped, not rejected
# ---------------------------------------------------------------------------
def test_a_price_outside_the_band_is_clamped_to_the_edge_not_dropped():
    """mm_backtest._requote clamps. Rejecting would leave that side empty.

    On the days micro_mm's lock trigger fires -- which is exactly when the
    price is near a circuit limit -- a rejected quote and a clamped one are
    different books, and the measured results are the clamped one.
    """
    # a band whose upper edge sits below what the strategy wants to offer
    band = PriceBand(upper_minor=28950, lower_minor=26010)
    # the strategy asks for an offer above the cap
    mm = StubMM(returns={"SELL": (290.00, 50)})
    quotes = an_adapter(mm, band=band).on_book(a_book(), position=0)
    # it rests at the edge rather than not resting at all
    assert quotes.ask is not None
    assert quotes.ask.price_minor == 28950


def test_a_price_below_the_lower_band_is_clamped_up():
    """The mirror."""
    # a band whose lower edge sits above what the strategy wants to bid
    band = PriceBand(upper_minor=31790, lower_minor=28890)
    # the strategy asks for a bid below the floor
    mm = StubMM(returns={"BUY": (288.00, 50)})
    quotes = an_adapter(mm, band=band).on_book(a_book(), position=0)
    # clamped up to the floor
    assert quotes.bid is not None
    assert quotes.bid.price_minor == 28890


def test_an_absent_bound_clamps_nothing():
    """A missing limit is not a limit of zero, and not one of infinity."""
    # only an upper bound published
    band = PriceBand(upper_minor=31790, lower_minor=None)
    # a bid well below where a lower bound would have been
    mm = StubMM(returns={"BUY": (100.00, 50)})
    quotes = an_adapter(mm, band=band).on_book(a_book(), position=0)
    # untouched
    assert quotes.bid.price_minor == 10000


def test_no_band_at_all_clamps_nothing():
    """Before the feed publishes limits there is nothing to clamp against."""
    # no band provider wired up
    mm = StubMM(returns={"BUY": (289.00, 50)})
    quotes = an_adapter(mm).on_book(a_book(), position=0)
    # the strategy's price, unmodified
    assert quotes.bid.price_minor == 28900


# ---------------------------------------------------------------------------
# multi-level depth
# ---------------------------------------------------------------------------
def test_depth_is_built_only_when_the_strategy_uses_more_than_one_level():
    """Two list allocations per book update for an off-by-default feature."""
    # a strategy configured for five-level OFI
    mm = StubMM()
    mm.ofi_depth_levels = 5
    # a book with two levels a side
    book = BookSnapshot(symbol="PPL", timestamp_ms=1,
                        bids=(BookLevel(28900, 500), BookLevel(28899, 100)),
                        asks=(BookLevel(28902, 300), BookLevel(28903, 200)))
    an_adapter(mm).on_book(book, position=0)
    # depth arrives as (bids, asks) of (price, qty), best first, in rupees
    _, _, _, _, _, depth = mm.quoted[0]
    assert depth == ([(289.00, 500), (288.99, 100)],
                     [(289.02, 300), (289.03, 200)])


# ---------------------------------------------------------------------------
# the day boundary
# ---------------------------------------------------------------------------
def test_a_new_day_needs_a_new_strategy_object():
    """Half-resetting micro_mm's state would be worse than not resetting it."""
    # the method exists so that someone looking for it finds the reason
    with pytest.raises(NotImplementedError, match="new strategy"):
        an_adapter().on_session_start("2026-09-17")




# ---------------------------------------------------------------------------
# the live config
#
# THESE RUN AGAINST THE REAL FILES. tests/data/ holds verbatim copies of
# config_assignment_20260915_0043.csv (the shipped three-bucket assignment),
# live_overrides.csv, and one pre-2026-09-15 two-bucket file. A loader tested
# only against fixtures the test wrote itself proves the test's idea of the
# schema, not the generator's.
# ---------------------------------------------------------------------------
# where the real files live
DATA = Path(__file__).resolve().parent / "data"
# the shipped assignment
REAL_ASSIGNMENT = DATA / "config_assignment_20260915_0043.csv"
# the real hand-edited overrides file, which normally has zero data rows
REAL_OVERRIDES = DATA / "live_overrides.csv"
# a file from the superseded two-bucket era
TWO_BUCKET = DATA / "config_assignment_20260913_2322_two_bucket.csv"

ASSIGNMENT_HEADER = ("symbol", "assigned_config", "skew_ticks", "skew_thresh")


@pytest.fixture
def config_dir():
    """A temporary directory for config files."""
    # cleaned up automatically
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def write_csv(path, header, rows):
    """Write a CSV from an explicit header and a list of row dicts."""
    # plain CSV, the way the generator writes it
    with Path(path).open("w", newline="") as handle:
        # the header decides the columns
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        # then the rows
        for row in rows:
            writer.writerow(row)


def assignment_row(symbol, label, **over):
    """One assignment row with the numbers the generator would have written."""
    # what the file should contain for this label
    ticks, thresh = FILE_PARAMS[label]
    # a blank cell is how the generator writes 'not applicable'
    row = {"symbol": symbol, "assigned_config": label,
           "skew_ticks": "" if ticks is None else ticks,
           "skew_thresh": "" if thresh is None else thresh}
    # anything the test wants to change
    row.update(over)
    return row


def a_store(config_dir, rows, overrides=None):
    """A LiveConfig over freshly written files."""
    # the assignment
    assignment = config_dir / "config_assignment_20260916_0900.csv"
    write_csv(assignment, ASSIGNMENT_HEADER, rows)
    # the optional overrides
    path = None
    if overrides is not None:
        path = config_dir / "live_overrides.csv"
        write_csv(path, ("symbol", "config"), overrides)
    # constructed, which loads
    return LiveConfig(assignment, path)


# ---- against the real files -----------------------------------------------
def test_the_shipped_assignment_loads_and_matches_its_published_counts():
    """68 / 13 / 17 / 15 across 113 names, as the run's own chart reports."""
    # the real file, no overrides
    cfg = LiveConfig(REAL_ASSIGNMENT, None)
    # every name present
    assert len(cfg.symbols(quoting_only=False)) == 113
    # the distribution the assignment run published
    assert cfg.counts() == {"QT_2t@15": 68, "QT_2t@20": 13,
                            "OBI": 17, "DROP": 15}
    # the 15 dropped names are excluded from the quoting set
    assert len(cfg.symbols()) == 98


def test_the_real_files_give_the_engine_the_right_numbers():
    """Spot-check one name from each of the four buckets."""
    # the real assignment
    cfg = LiveConfig(REAL_ASSIGNMENT, None)
    # PPL is on the incumbent lean
    assert cfg.strategy_kwargs("PPL") == {"queue_skew_ticks": 2.0,
                                          "queue_skew_thresh": 0.15}
    # BAFL sits in the band and took the higher trigger
    assert cfg.strategy_kwargs("BAFL") == {"queue_skew_ticks": 2.0,
                                           "queue_skew_thresh": 0.20}
    # TBL is a name the lean loses money on -- lean off, zero ticks
    assert cfg.strategy_kwargs("TBL") == {"queue_skew_ticks": 0.0,
                                          "queue_skew_thresh": DEFAULT_THRESH}
    # SGPL was dropped on capacity -- no_add via soft_inv
    assert cfg.strategy_kwargs("SGPL") == {"queue_skew_ticks": 0.0,
                                           "queue_skew_thresh": DEFAULT_THRESH,
                                           "soft_inv": 0}
    # and the control-plane fields travel separately
    assert cfg.params_for("SGPL")["drop_semantics"] == "no_add"
    assert cfg.params_for("SGPL")["skip_if_flat"] is True
    assert cfg.params_for("PPL")["drop_semantics"] is None


def test_lean_off_gets_micro_mms_own_default_threshold_not_zero():
    """Zero is the MOST-firing value a threshold can take.

    The threshold is inert while queue_skew_ticks == 0, so any value works
    today. But if that gate is ever removed or the threshold read somewhere
    else, 0.0 fires on every bid-heavy book and 0.15 -- micro_mm's own
    constructor default -- fires the way the leaning names do.
    """
    # micro_mm's default, restated here so a change to either is visible
    assert DEFAULT_THRESH == 0.15
    # a lean-off name
    cfg = LiveConfig(REAL_ASSIGNMENT, None)
    # gets the default, not zero, and not NaN
    assert cfg.params_for("TBL")["queue_skew_thresh"] == 0.15


def test_the_real_overrides_file_is_all_comments_and_applies_nothing():
    """Its normal state: a header, a documentation block, zero data rows."""
    # both real files
    cfg = LiveConfig(REAL_ASSIGNMENT, REAL_OVERRIDES)
    # the comment rows must not have become symbols
    assert len(cfg.symbols(quoting_only=False)) == 113
    # the summary says no overrides are in force
    assert "0 override(s)" in cfg.summary()
    # and the commented-out example rows did not apply
    assert cfg.params_for("TRG")["label"] == "QT_2t@15"


def test_a_two_bucket_file_is_refused_at_startup_and_says_why():
    """It carries no skew columns and labels the lean 'QT_2t'.

    Pointing the engine at one would run a superseded assignment. Silently
    mapping it forward would be worse -- that file's buckets were decided by a
    different rule (t, not the effect size d).
    """
    # a real pre-2026-09-15 file. Startup is FATAL: there is no last-good
    # config to fall back to.
    with pytest.raises(ConfigError) as exc:
        LiveConfig(TWO_BUCKET, None)
    # the error names the missing column and the likely cause
    assert "skew_ticks" in str(exc.value)
    assert "TWO-BUCKET" in str(exc.value)


# ---- the failure policy, which is asymmetric on purpose --------------------
def test_a_bad_file_at_startup_is_fatal(config_dir):
    """There is no last-good config to fall back to."""
    # a file with a typo
    path = config_dir / "a.csv"
    write_csv(path, ASSIGNMENT_HEADER,
              [{"symbol": "PPL", "assigned_config": "PAUSE",
                "skew_ticks": 2.0, "skew_thresh": 0.15}])
    # refuse to start rather than quote on a config nobody chose
    with pytest.raises(ConfigError, match="unknown setting"):
        LiveConfig(path, None)


def test_a_bad_file_on_reload_keeps_the_last_good_config(config_dir):
    """Dropping the book because someone saved a CSV mid-edit is worse."""
    # a good file first
    cfg = a_store(config_dir, [assignment_row("PPL", "QT_2t@15")])
    # then a broken one
    cfg.assignment_path.write_text("symbol,assigned_config\nPPL,QT_2t@15\n")
    # refresh NEVER raises
    assert cfg.refresh() == {}
    # the old setting is still live
    assert cfg.params_for("PPL")["label"] == "QT_2t@15"
    # and the failure is visible to a health check rather than swallowed
    assert cfg.consecutive_failures == 1
    # a later good file clears the counter and applies
    write_csv(cfg.assignment_path, ASSIGNMENT_HEADER,
              [assignment_row("PPL", "QT_2t@20")])
    assert cfg.refresh() == {"PPL": ("QT_2t@15", "QT_2t@20")}
    assert cfg.consecutive_failures == 0


def test_an_unchanged_file_is_a_no_op(config_dir):
    """Content hashes, not mtime: a copy can bump mtime without changing bytes."""
    # a config
    cfg = a_store(config_dir, [assignment_row("PPL", "QT_2t@15")])
    # touching the file without changing its bytes must not re-apply anything
    cfg.assignment_path.touch()
    assert cfg.refresh() == {}
    # changing the bytes does
    write_csv(cfg.assignment_path, ASSIGNMENT_HEADER,
              [assignment_row("PPL", "OBI")])
    assert cfg.refresh() == {"PPL": ("QT_2t@15", "OBI")}


def test_a_file_that_is_not_utf8_is_refused(config_dir):
    """A mojibake file is corrupt, not something to coerce."""
    # a valid config first, so this is a reload rather than a startup
    cfg = a_store(config_dir, [assignment_row("PPL", "QT_2t@15")])
    # bytes that are not valid UTF-8
    cfg.assignment_path.write_bytes(
        b"symbol,assigned_config,skew_ticks,skew_thresh\n\xff\xfePPL,OBI,0.0,\n")
    # rejected, previous config still live
    assert cfg.refresh() == {}
    assert cfg.params_for("PPL")["label"] == "QT_2t@15"


# ---- the rules ------------------------------------------------------------
def test_a_symbol_missing_from_a_later_file_keeps_its_setting_and_is_an_orphan(
        config_dir):
    """Absence is a partial write, not an instruction to stop trading."""
    # both names present at first
    cfg = a_store(config_dir, [assignment_row("PPL", "QT_2t@15"),
                               assignment_row("UBL", "QT_2t@20")])
    # nothing orphaned yet
    assert cfg.orphans() == []
    # the next write drops UBL's ROW -- not the same as flagging it DROP
    write_csv(cfg.assignment_path, ASSIGNMENT_HEADER,
              [assignment_row("PPL", "QT_2t@20")])
    cfg.refresh()
    # PPL moved bucket
    assert cfg.params_for("PPL")["queue_skew_thresh"] == 0.20
    # UBL kept its last setting and is still quoting
    assert cfg.params_for("UBL")["label"] == "QT_2t@20"
    assert "UBL" in cfg.symbols()
    # AND it is reported, every cycle, so it cannot quietly outlive the
    # analysis that chose its setting
    assert cfg.orphans() == ["UBL"]
    assert "1 ORPHAN(S)" in cfg.summary()


def test_drop_stops_adding_but_keeps_reducing(config_dir):
    """DROP is no_add, not cancel-everything."""
    # one dropped name
    cfg = a_store(config_dir, [assignment_row("PPL", "DROP")])
    # soft_inv=0 makes micro_mm quote only the side that reduces the position
    assert cfg.strategy_kwargs("PPL")["soft_inv"] == 0
    # excluded from the actively quoted set, but still a known name
    assert cfg.symbols() == []
    assert cfg.symbols(quoting_only=False) == ["PPL"]
    # and the caller is told the flat case needs handling separately
    assert cfg.params_for("PPL")["skip_if_flat"] is True


def test_a_name_never_configured_has_no_setting(config_dir):
    """Never-seen is different from DROP: nobody chose to trade it."""
    # a config with one name in it
    cfg = a_store(config_dir, [assignment_row("PPL", "QT_2t@15")])
    # a name that has never appeared returns None, not a default
    assert cfg.params_for("HBL") is None
    assert cfg.strategy_kwargs("HBL") is None


def test_control_plane_fields_never_reach_the_strategy(config_dir):
    """quote, label, drop_semantics and skip_if_flat are not constructor args.

    Passing params_for()'s whole dict into MicrostructureMM would raise
    TypeError on the first symbol, so the split lives here rather than at every
    call site.
    """
    # a dropped name, which carries the most extra fields
    cfg = a_store(config_dir, [assignment_row("PPL", "DROP")])
    # the strategy gets exactly the arguments it has parameters for
    assert set(cfg.strategy_kwargs("PPL")) == {"queue_skew_ticks",
                                               "queue_skew_thresh", "soft_inv"}
    # while the control plane keeps what it needs
    assert set(cfg.params_for("PPL")) >= {"quote", "label", "drop_semantics",
                                          "skip_if_flat"}


# ---- the overrides file ---------------------------------------------------
def test_an_override_wins_and_is_marked_as_hand_set(config_dir):
    """The hand-edited file is the one someone touches under time pressure."""
    # the override pulls one name, using an alias the file documents
    cfg = a_store(config_dir,
                  [assignment_row("PPL", "QT_2t@15"),
                   assignment_row("UBL", "QT_2t@15")],
                  overrides=[{"symbol": "PPL", "config": "PULL"}])
    # the override wins for the name it lists
    assert cfg.params_for("PPL")["label"] == "DROP"
    # the summary says one is in force
    assert "1 override(s)" in cfg.summary()
    # every other name is left alone
    assert cfg.params_for("UBL")["label"] == "QT_2t@15"


def test_every_documented_override_alias_is_accepted(config_dir):
    """The file documents these spellings, so the loader must honour them."""
    # alias -> what it must resolve to
    aliases = {"LEAN": "QT_2t@15", "LEAN15": "QT_2t@15", "QT_2t@15": "QT_2t@15",
               "LEAN20": "QT_2t@20", "QT_2t@20": "QT_2t@20",
               "OFF": "OBI", "LEANOFF": "OBI", "LEAN_OFF": "OBI", "OBI": "OBI",
               "STOP": "DROP", "PULL": "DROP", "HALT": "DROP", "DROP": "DROP"}
    # one name per alias
    names = [f"SYM{i}" for i in range(len(aliases))]
    # write each alias in lower case, since the file says case-insensitive
    cfg = a_store(config_dir,
                  [assignment_row(n, "QT_2t@15") for n in names],
                  overrides=[{"symbol": n, "config": a.lower()}
                             for n, a in zip(names, aliases)])
    # each one resolved to the setting its documentation promises
    for name, alias in zip(names, aliases):
        assert cfg.params_for(name)["label"] == aliases[alias], alias


def test_comment_rows_in_the_overrides_file_are_ignored(config_dir):
    """The file keeps its own instructions in rows beginning with '#'."""
    # a real name
    assignment = config_dir / "a.csv"
    write_csv(assignment, ASSIGNMENT_HEADER, [assignment_row("PPL", "OBI")])
    # a file whose rows are all comments, including ones that look like rows
    overrides = config_dir / "live_overrides.csv"
    overrides.write_text(
        "symbol,config,reason,set_by,set_at\n"
        "# ------------------------------------------------\n"
        "# TRG,DROP,earnings after close,SZ,2026-09-15 13:40\n"
        "# LUCK,OFF,index rebal, book is one-way,SZ,2026-09-15 11:05\n")
    # the commented-out rows must not apply, and the extra comma in the third
    # one must not break the read
    cfg = LiveConfig(assignment, overrides)
    assert cfg.params_for("PPL")["label"] == "OBI"
    assert cfg.params_for("TRG") is None


def test_one_bad_value_rejects_the_whole_overrides_file(config_dir):
    """A typo cannot half-apply: two names on the new setting, three on the old."""
    # the first override is valid, the second is a typo
    with pytest.raises(ConfigError, match="PAUSE"):
        a_store(config_dir,
                [assignment_row("PPL", "QT_2t@15"),
                 assignment_row("UBL", "QT_2t@15")],
                overrides=[{"symbol": "PPL", "config": "OFF"},
                           {"symbol": "UBL", "config": "PAUSE"}])


def test_an_override_for_a_name_not_in_the_assignment_is_refused(config_dir):
    """The universe is decided by the refit, not at 14:00."""
    # an override naming a symbol the assignment does not contain
    with pytest.raises(ConfigError, match="not in the assignment"):
        a_store(config_dir, [assignment_row("PPL", "OBI")],
                overrides=[{"symbol": "HBL", "config": "LEAN"}])


# ---- malformed assignments ------------------------------------------------
def test_numbers_that_disagree_with_their_label_are_refused(config_dir):
    """The one failure live_config.py could not see.

    Its LABELS table carried its own copy of (ticks, threshold) and re-derived
    them from the label, so a refit that moved the lean to three ticks would
    write 3.0 into the CSV, the table would still say 2.0, and the engine would
    quote the old lean with no error anywhere.
    """
    # labelled 0.15 but carrying 0.20
    path = config_dir / "a.csv"
    write_csv(path, ASSIGNMENT_HEADER,
              [{"symbol": "PPL", "assigned_config": "QT_2t@15",
                "skew_ticks": 2.0, "skew_thresh": 0.20}])
    # there is no way to tell which of the two is the stale one
    with pytest.raises(ConfigError, match="the file says"):
        LiveConfig(path, None)


def test_a_lean_moved_to_three_ticks_is_refused_rather_than_quoted_at_two(
        config_dir):
    """The concrete version of the above, which is the case that matters."""
    # what the CSV would look like after a refit that changed the lean
    path = config_dir / "a.csv"
    write_csv(path, ASSIGNMENT_HEADER,
              [{"symbol": "PPL", "assigned_config": "QT_2t@15",
                "skew_ticks": 3.0, "skew_thresh": 0.15}])
    # refuse, and say that this code's table may be the stale one
    with pytest.raises(ConfigError, match="were not updated"):
        LiveConfig(path, None)


def test_a_duplicated_symbol_is_refused(config_dir):
    """Two conflicting instructions, no defensible way to pick one."""
    # the same name twice
    path = config_dir / "a.csv"
    write_csv(path, ASSIGNMENT_HEADER, [assignment_row("PPL", "QT_2t@15"),
                                        assignment_row("PPL", "DROP")])
    with pytest.raises(ConfigError, match="more than once"):
        LiveConfig(path, None)


def test_a_header_only_assignment_is_refused(config_dir):
    """Zero rows would orphan the entire universe in one reload."""
    # a header and nothing else
    path = config_dir / "a.csv"
    write_csv(path, ASSIGNMENT_HEADER, [])
    with pytest.raises(ConfigError, match="zero rows"):
        LiveConfig(path, None)


def test_an_empty_label_is_a_half_finished_edit(config_dir):
    """Refuse the file rather than guess what someone was about to type."""
    # a row with a blank setting
    path = config_dir / "a.csv"
    write_csv(path, ASSIGNMENT_HEADER,
              [{"symbol": "PPL", "assigned_config": "",
                "skew_ticks": "", "skew_thresh": ""}])
    with pytest.raises(ConfigError, match="empty assigned_config"):
        LiveConfig(path, None)


def test_a_missing_assignment_file_is_refused(config_dir):
    """Starting with no config at all would quote nothing, silently."""
    # a path that does not exist
    with pytest.raises(ConfigError, match="assignment not found"):
        LiveConfig(config_dir / "nope.csv", None)


def test_symbols_are_normalised_so_a_hand_edit_in_lower_case_works(config_dir):
    """Someone types 'ppl' at 14:00. That must not silently do nothing."""
    # a lower-case override against an upper-case assignment
    cfg = a_store(config_dir, [assignment_row("PPL", "QT_2t@15")],
                  overrides=[{"symbol": "ppl", "config": "off"}])
    # it applied
    assert cfg.params_for("PPL")["label"] == "OBI"
    # and the lookup is normalised on the way in too
    assert cfg.params_for("ppl")["label"] == "OBI"


def test_the_preflight_exit_code_is_the_go_no_go():
    """0 = safe to start the book, 1 = do not."""
    # the real files pass
    assert preflight(REAL_ASSIGNMENT, REAL_OVERRIDES) == 0
    # a superseded file does not
    assert preflight(TWO_BUCKET, None) == 1


# ---------------------------------------------------------------------------
# the fee, where the backtest and the venue have to agree
# ---------------------------------------------------------------------------
def test_the_venue_fee_matches_the_backtest_schedule():
    """PSXVenue and mm_backtest must charge the same thing.

    mm_backtest's TREC schedule is LAGA 0.0035% + SECP 0.00065% + IPF 0.00062%
    + clearing 0.003%, with no broker commission because we are our own broker.
    That is 0.0000777 per side = 0.777 bps, or 1.554 bps round trip. If these
    two ever drift, the engine's viability gate and the backtest's P&L are
    computed against different costs and the comparison is meaningless.
    """
    # the venue's number
    venue = a_venue()
    # mm_backtest's components, spelled out rather than imported, because the
    # live engine must never import the backtest
    trec_per_side = 0.000035 + 0.0000065 + 0.0000062 + 0.00003
    # the venue carries it in basis points
    assert abs(venue.fee_bps_per_side - trec_per_side * 1e4) < 1e-9
    # and the round trip is the 1.554 bps every measured edge is quoted against
    assert abs(venue.fee_bps_per_side * 2 - 1.554) < 1e-9


if __name__ == "__main__":
    # A pytest file is not a script: there is no runner, and the project root is
    # not on the import path. Say so rather than failing with ModuleNotFoundError.
    raise SystemExit(
        "This is a pytest file, not a script.\n"
        "Run the suite from the Production directory:\n"
        "    python -m pytest -q\n"
        "or one file:\n"
        "    python -m pytest tests/test_strategy.py -q")
