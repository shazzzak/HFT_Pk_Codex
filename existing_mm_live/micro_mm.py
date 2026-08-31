import math
from collections import deque
from mm_backtest import FEE_TOTAL_PCT


# ============================ TRIGGER DEFAULTS (cfg-ready) ====================
# These are the EOD / distance-to-lock trigger knobs, declared at module top so
# they are easy to toggle now and easy to move into a periodically-read .cfg file
# in the production system. Each is consumed as the DEFAULT of an __init__ param,
# so any instance can still override them per-run (e.g. in a sweep).

# minutes before the continuous close where the FLAT-case exploding widen starts
EOD_RAMP_START_MIN = 5.0
# minutes before the continuous close where the FLAT-case hard cliff fires
# (go dark: no new positions in the final minute)
EOD_CLIFF_MIN = 1.0
# ---- LOCK trigger thresholds: PERCENT OF PRICE, measured FROM THE MID to the
# EXCHANGE-PUBLISHED band (UPPER/LOWER_CIRCUIT_BREAKER rows -> limit_up/limit_dn).
# Spread-free by design: the 2026-08 diagnostic proved spread-unit distances fire
# spuriously whenever the spread widens (median fire at 3-4% from the band on
# names that never lock). Percent-of-price is instantaneous AND stable.
# LOCK_RAMP_PCT: start the flat-case widen when the mid is within this % of a band
# (2.0 = price has moved ~+8% of a 10% band).
LOCK_RAMP_PCT = 2.0
# LOCK_CLIFF_PCT: go dark (flat) on the trapped side within this % of a band
# (0.5 = ~+9.5% moved on a 10% band).
LOCK_CLIFF_PCT = 0.5
# LOW-PRICE TIER (production guard for cheap names, e.g. a stock falling under
# Rs 10): one tick is a big % move there, so a fixed % cliff could be jumped in a
# single tick. The effective thresholds are "X% of price OR N ticks, whichever is
# LARGER", so the cliff/ramp zones are always at least N ticks wide. For PACE at
# ~16 PKR the tick floor never binds (tick = 0.06% of price); for a Rs 1 stock
# (tick = 1% of price) the cliff auto-widens to 2% and the ramp to 4%.
# MIN_CLIFF_TICKS: the cliff zone must span at least this many ticks.
MIN_CLIFF_TICKS = 2.0
# MIN_RAMP_GAP_TICKS: the ramp zone must extend at least this many ticks beyond
# the cliff (so a single tick cannot jump from outside-the-ramp into the lock).
MIN_RAMP_GAP_TICKS = 2.0
# cap on the exploding widen: the widened half-spread never exceeds this many
# REFERENCE market spreads from the reservation ("~10 spreads out = unfillable").
# The reference is max(current spread, EMA spread) -- see SPREAD_ALPHA below.
WIDEN_CAP_SPREADS = 10.0
# EMA decay for the rolling spread estimate (0.05 ~ 13.5-update half-life, same
# family as vol_alpha). The cap uses max(instantaneous, EMA) so a momentarily
# tight book cannot collapse the cap to nothing exactly when we want to be wide;
# a genuinely widening book still grows it. Distances-to-band stay on the
# INSTANTANEOUS spread (spread cancels in the unclipped region -> pure price
# rule; and a blown-out spread makes the floor-clipped cliff MORE conservative).
SPREAD_ALPHA = 0.05
# ==============================================================================


class MicrostructureMM:
    """Market maker built from O'Hara, Market Microstructure Theory (1998).

    Implements, with chapter references:
      Ch 3.3 Glosten-Milgrom : fair value = flow-conditional expectation
                               (imbalance-weighted microprice, not the mid)
      Ch 2.2 Stoll /
      Ch 2.3 Ho-Stoll        : inventory shifts PLACEMENT (skew), not width;
                               width carries the risk term gamma*sigma^2*tau
      Ch 2.3 Ho-Stoll        : spread decays as the horizon tau -> 0
      Ch 3.4 Easley-O'Hara   : adverse selection rises in size -> cut SIZE, not
                               just widen, when flow looks informed
      Ch 6.3 Easley-O'Hara   : absence of trade => less informed trading =>
                               TIGHTEN (opposite of Diamond-Verrecchia)
      Ch 7.1 viability       : if the required spread exceeds the market spread,
                               no viable quote exists -> quote nothing

    EOD / LOCK TRIGGERS (production guards, OFF by default):
      Two urgency sources -- time-to-close and distance-to-the-circuit-band --
      each with a graded RAMP and a hard CLIFF, and both holding-aware:
        FLAT   (|pos| < one clip): ramp = exploding widen of the acquiring
               side(s), capped at WIDEN_CAP_SPREADS x market spread; cliff =
               go dark (time: both sides in the last EOD_CLIFF_MIN minutes;
               lock: the TRAPPED side within the cliff distance of the band).
        HOLDING: the adding side is pulled inside ANY trigger window; the EXIT
               side is ALWAYS quoted, never widened or pulled, and is leaned to
               the most aggressive post-only placement to get flat.
      MEASURED CONTEXT (2026-08, PPL/UBL/PACE, 207 days): lock approaches SNAP --
      price sits ~17-42 spreads out until T-10s and covers the distance in the
      final seconds. So on these names the RAMP will almost never engage and
      "the ramp made no difference" in a backtest is the EXPECTED result, not
      evidence it is broken. It is kept deliberately (defense-in-depth) for
      names/regimes that crawl instead of snap. Do not remove it for inactivity;
      check stats["*_ramp_widen"] to see when it actually engaged.

    FLAGGED SIMPLIFICATIONS (production versions noted):
      * sigma is a rolling realised vol of the reconstructed mid. Production:
        two-scale estimator with microstructure-noise correction.
      * toxicity is a signed-volume imbalance ratio over a trade window as a
        cheap stand-in for GM's mu / PIN. Production: fit PIN or VPIN, or the
        OFI-conditioned toxicity model on your own markout labels.
      * Kyle's lambda is NOT estimated here. Production: regress mid changes on
        signed order flow per symbol-day and use lambda to size the repricing
        response and the toxicity term.
      * the Avellaneda-Stoikov base half-spread (1/gamma)*ln(1+gamma/kappa) is
        computed but weighted 0 by default -- kappa must be calibrated from your
        real resting-order fill outcomes before it means anything.
      * gamma is a free risk-aversion parameter, NOT calibrated. Sweep it.
    """
    """
    quiet_ms=2000 — this default is wrong for both your symbols, in opposite directions. 
    It should scale with the symbol's median inter-trade gap: MCB's is 5.6 s (so 2 s
     calls it "active" mid-gap when nothing is wrong), OGDC's is 0.4 s (so 2 s almost 
     never triggers). Something like 2–3× the symbol's median gap, computed from the 
     data, not a constant.    
     
     sigma_window=200, flow_window=50 — event-count windows, so their wall-clock meaning 
     differs wildly per symbol (200 events ≈ most of an hour on MCB, ≈ 2 minutes on OGDC). 
     Per-symbol they should be chosen to represent a comparable time horizon, or replaced 
     by time-based windows.
     
     gamma — a genuinely free risk-aversion parameter with no formula; sweep it per symbol 
     and pick by out-of-sample PnL/drawdown. kappa — calibrate from real resting-order fill 
     outcomes (already flagged; inert until then via as_base_weight=0). min_edge_pct — 
     business policy, not calibration: how much you insist on earning above cost. tick=0.01 — 
     the one true constant (PSX-wide). session_ms — per day, already passed in
    
    So the parameter set splits into four classes: universal constants (tick), 
    per-symbol calibrated from data (size, max_inv, quiet_ms, windows, later κ and λ), 
    free parameters to sweep (gamma, and the weightings), and policy (min_edge_pct, 
    fee scenario). The natural next artifact when you scale beyond two symbols is a 
    small per-symbol config table — symbol → calibrated parameters — generated by a 
    calibration script from each symbol's own history, which the ticker-stats work 
    you've already built is most of the way toward producing.
     
    """

    # Duck-typed flag the backtester checks each event: when True it syncs the
    # book's published circuit-breaker prices onto self.limit_up / self.limit_dn.
    # Naive lacks this attribute entirely, so the engine skips it there.
    wants_limits = True

    def __init__(self, size=50, max_inv=500, gamma=0.15, kappa=1.5,
                 session_ms=(0, 1), fee_pct=None, min_edge_pct=0.0,
                 flow_window=50, tick=0.01,
                 quiet_ms=2000, require_viable=True, as_base_weight=0.0,
                 improve_ticks=1.0, size_notional=None, tol_ticks=0.0, vol_alpha=0.05,
                 # session_scale is REQUIRED, keyword-only: no default -> a missing
                 # value raises TypeError at construction instead of silently
                 # skewing on a wrong scale.
                 # use_microprice=False -> quote around plain mid (neutral, like
                 # naive); it exists to test/disable the microprice, the confirmed
                 # driver of the short drift.
                 # soft_inv=N -> once |pos|>N, stop quoting the side that ADDS to the
                 # position (pull that quote) so fills can only reduce it. None
                 # disables the band.
                 *, session_scale, use_microprice=True,
                 # continuous microprice lean coefficient. None -> derive from the
                 # use_microprice boolean (back-compat). Set explicitly to sweep the
                 # lean: +1 = classic microprice, 0 = mid, <0 = DEFENSIVE (lean away
                 # from imbalance to convert 'through' pick-offs into 'at_queue').
                 micro_lambda=None,
                 soft_inv=None,
                 # --- EOD / LOCK triggers (OFF by default: PPL/UBL runs are
                 # unaffected unless a config explicitly enables them) ---
                 enable_eod_trigger=False, enable_lock_trigger=False,
                 # time thresholds (minutes), defaults from the module-top constants
                 eod_ramp_start_min=EOD_RAMP_START_MIN, eod_cliff_min=EOD_CLIFF_MIN,
                 unwind_profile=None, unwind_pov=0.10, session_segments=None,
                 reactive_mode="off", reactive_k=3.0, reactive_cooldown_s=60.0,
                 reactive_lookback_s=10.0,
                 # lock thresholds in PERCENT OF PRICE (spread-free), plus the
                 # low-price tier floors in ticks (see module constants)
                 lock_ramp_pct=LOCK_RAMP_PCT,
                 lock_cliff_pct=LOCK_CLIFF_PCT,
                 min_cliff_ticks=MIN_CLIFF_TICKS,
                 min_ramp_gap_ticks=MIN_RAMP_GAP_TICKS,
                 # cap on the exploding widen, in reference market spreads
                 widen_cap_spreads=WIDEN_CAP_SPREADS,
                 # EMA decay for the rolling spread reference (cap robustness)
                 spread_alpha=SPREAD_ALPHA,
                 # --- INVENTORY-EXIT SKEW SWEEP (default OFF = current behavior) ---
                 # When |pos| exceeds exit_inv_threshold lots, place the EXIT side
                 # this many ticks INSIDE the touch to leave faster (trading
                 # capture for shorter hold -> less diffusive markout). 0 = current
                 # behavior (never post inside the touch from this mechanism).
                 exit_ticks_inside=0,
                 # inventory threshold (in LOTS) beyond which the tick-exit engages
                 exit_inv_threshold=1.0,
                 # --- OBI-DEFENSIVE SKEW SWEEP (default OFF = current behavior) ---
                 # When True, suppress (widen) the side the book leans AGAINST, so
                 # we stop resting in front of predictable flow. imb>0.5 = bid-heavy
                 # -> suppress SELL adds less / widen BUY; imb<0.5 = ask-heavy ->
                 # widen BUY so we don't buy into selling pressure.
                 obi_defensive=False,
                 # imbalance distance from 0.5 beyond which the defensive widen
                 # engages (0.15 -> engages when imb <0.35 or >0.65)
                 obi_defensive_thresh=0.15,
                 # how many ticks to widen the disadvantaged side when engaged
                 obi_defensive_ticks=1.0,
                 # ---- OFI-DEFENSIVE (Stage 3): trailing-FLOW retreat ----
                 # master switch: widen the side the trailing order FLOW is
                 # running against. RETREAT-ONLY (never a fair-value lean; the
                 # sweep proved leaning destructive). Hard-OFF in first15 (the
                 # horserace measured zero signal there on every window).
                 ofi_defensive=False,
                 # min-window EVENT cap: trailing window holds at most N book
                 # events (the horserace-winning min(N ev, T s) hybrid form)
                 ofi_window_ev=50,
                 # min-window TIME cap (seconds): entries older are evicted;
                 # whichever cap holds FEWER events binds
                 ofi_window_s=5.0,
                 # engage threshold on the NORMALIZED trailing OFI in [-1,+1]
                 # (window sum of signed flow / window sum of |flow|; +1 = all
                 # buying). Bounded like OBI's imbalance -> same interpretability.
                 ofi_defensive_thresh=0.30,
                 # ticks to widen the threatened side when OFI-defensive fires
                 ofi_defensive_ticks=1.0,
                 # number of price levels used by OFI. 1 preserves the validated
                 # L1 touch path byte-for-byte; 5/10 enable the multi-level
                 # extension. Deep mode requires the caller to pass ranked depth
                 # into quotes(depth=...).
                 ofi_depth_levels=1,
                 # ---- SIZE THROTTLE (Stage 4): cut CLIP SIZE on the exposed
                 # side when the book/flow is adverse, instead of (or on top of)
                 # widening. The "0.5x clip instead of base" idea, time-boxed.
                 # All flags default OFF -> byte-identical to the pre-throttle
                 # strategy (the before/after identity test asserts this).
                 obi_throttle=False,
                 ofi_throttle=False,
                 # fraction of base clip to quote while throttled (0.5 = half)
                 throttle_frac=0.5,
                 # OBI engage threshold (mirrors obi_defensive_thresh form)
                 obi_throttle_thresh=0.15,
                 # OFI engage threshold on the normalized [-1,+1] trailing OFI
                 ofi_throttle_thresh=0.30,
                 # QDR SIZE THROTTLE: on/off + the fraction of the own-side touch
                 # queue eaten in ONE event (at a stable price) that trips the
                 # cut. 0.40 is where the diagnostic's toxic depletion mass sits.
                 qdr_throttle=False,
                 qdr_throttle_thresh=0.40,
                 # time-box (ms of EXCHANGE time): a triggered side stays
                 # throttled this long, then restores. Brackets the 100-500ms
                 # window from the throttle diagnostic. 0 => only the trigger
                 # cycle is throttled (no hold).
                 throttle_hold_ms=300.0):
        # Baseline quote size in shares (Ch 3.4: this gets cut when flow is toxic).
        self.size0 = size
        # Hard inventory cap in shares (a backstop; the skew is the real control).
        self.max_inv = max_inv
        # Soft inventory band in shares: once |pos| exceeds it we stop quoting the
        # side that ADDS to the position, so fills can only reduce it. None disables
        # the band. Sweep {100, 150, 200}.
        self.soft_inv = soft_inv
        # Directional-pricing toggle: True = imbalance microprice (Ch 3.3); False =
        # plain mid (neutral, like naive). The microprice is the confirmed cause of
        # the short drift, so this exists to test/disable it.
        self.use_microprice = use_microprice
        # continuous lean coefficient (None -> derived from the boolean at quote time)
        self.micro_lambda = micro_lambda
        # Risk aversion. Scales BOTH the inventory skew and the risk half-spread.
        self.gamma = gamma
        # Dimensional bridge (units 1/PKR) scaling the inventory skew from PKR
        # variance to PKR price; gamma is treated as dimensionless in the skew, so
        # THIS carries the dimension. CALIBRATED, not free: back-solved per symbol so
        # skew at max inventory ~ 1x median spread (derived PPL~7.6, UBL~3.9,
        # PACE~46.15). SWEEP it {0.25, 0.5, 1, 2}x the derived value.
        self.session_scale = session_scale
        # Order-arrival intensity decay for the A-S base term (needs calibration).
        self.kappa = kappa
        # Session start/end in exchange-ms; defines the Ho-Stoll horizon tau.
        # NOTE: with the phase-based session fix, t1 is the CONTINUOUS close, so the
        # time trigger below counts down to the true bell automatically.
        self.t0, self.t1 = session_ms
        # All-in per-side fee as a fraction of traded value (the spread floor).
        # Fee floor for quoting decisions. Defaults to the SAME schedule the
        # backtester charges on fills -- one source of truth. Pass explicitly
        # only to run what-if scenarios (e.g. MM-programme fee relief).
        self.fee_pct = FEE_TOTAL_PCT if fee_pct is None else fee_pct
        # Extra edge demanded above fees before quoting at all.
        self.min_edge_pct = min_edge_pct
        # Price grid.
        self.tick = tick
        # Gap (ms) with no trade after which we treat the market as quiet.
        self.quiet_ms = quiet_ms
        # If True, refuse to quote when the market spread cannot cover our cost.
        self.require_viable = require_viable
        # Weight on the A-S base half-spread; 0 until kappa is calibrated.
        self.as_base_weight = as_base_weight

        # --- trigger config (stored per-instance so a sweep can override) ---
        # master switch for the time-to-close trigger
        self.enable_eod_trigger = enable_eod_trigger
        # master switch for the distance-to-lock trigger
        self.enable_lock_trigger = enable_lock_trigger
        # minutes before the close where the flat-case widen ramp starts
        self.eod_ramp_start_min = eod_ramp_start_min
        # minutes before the close where the flat-case hard cliff fires
        self.eod_cliff_min = eod_cliff_min
        # ---- POV unwind model (SZ's weighted-bucket design, 2026-08) ----
        # unwind_profile: (vol_first15, vol_middle, vol_last15) shares-per-minute
        # for THIS name, from build_volume_profile.py. None -> the OLD fixed-window
        # behaviour (holding inside the time window engages the unwind), keeping
        # the old strategy available for A/B comparison.
        self.unwind_profile = unwind_profile
        # participation cap: we never assume more than this share of market volume
        self.unwind_pov = unwind_pov
        # the day's continuous-trading segments [(start_ms, end_ms), ...]; REQUIRED
        # when unwind_profile is set (Friday's Jumu'ah break makes wall-clock time
        # WRONG -- tradeable minutes must be summed over segments)
        self.session_segments = session_segments
        # the active trigger window, updated every _trigger_state evaluation and
        # read by the backtester to tag fills. "none" before the first evaluation
        # and permanently when both triggers are disabled.
        self.current_window = "none"
        # ---- REACTIVE JUMP GATE (2026-08-21) -----------------------------------
        # PSX jumps have NO book precursor (validated), so we cannot PREDICT them.
        # This REACTS instead: after a large move over the last reactive_lookback_s
        # seconds, go dark for reactive_cooldown_s to avoid the compounding 2nd/3rd
        # toxic fill. mode: "off" | "symmetric" (any big move) | "inventory"
        # (only moves AGAINST current inventory -- preserves profitable reversion).
        self.reactive_mode = reactive_mode
        # trigger size in trailing-sigma multiples over the lookback window
        self.reactive_k = reactive_k
        # stay-dark duration once tripped (ms)
        self.reactive_cooldown_ms = reactive_cooldown_s * 1000.0
        # trailing window over which the move is measured (ms)
        self.reactive_lookback_ms = reactive_lookback_s * 1000.0
        # (ts_ms, mid) history for the lookback move; trimmed each observe
        self._mid_hist = deque()
        # timestamp until which we stay dark (0 = not gated)
        self._dark_until = 0.0
        # lock thresholds (% of price) + the low-price tier tick floors
        self.lock_ramp_pct = lock_ramp_pct
        self.lock_cliff_pct = lock_cliff_pct
        self.min_cliff_ticks = min_cliff_ticks
        self.min_ramp_gap_ticks = min_ramp_gap_ticks
        # widen cap in reference market spreads
        self.widen_cap_spreads = widen_cap_spreads
        # EMA decay for the rolling spread reference
        self.spread_alpha = spread_alpha
        # --- inventory-exit tick skew (sweep axis 1; 0 = current behavior) ---
        # ticks to post inside the touch on the EXIT side when loaded
        self.exit_ticks_inside = exit_ticks_inside
        # inventory threshold (lots) beyond which the tick-exit engages
        self.exit_inv_threshold = exit_inv_threshold
        # --- OBI-defensive skew (sweep axis 2; False = current behavior) ---
        # master toggle
        self.obi_defensive = obi_defensive
        # |imb-0.5| beyond which the defensive widen engages
        self.obi_defensive_thresh = obi_defensive_thresh
        # ticks to widen the disadvantaged side
        self.obi_defensive_ticks = obi_defensive_ticks
        # --- OFI-defensive (Stage 3; False = current behavior) ---
        # master toggle
        self.ofi_defensive = ofi_defensive
        # min-window caps: at most N events AND at most T seconds old
        self.ofi_window_ev = int(ofi_window_ev)
        self.ofi_window_s = float(ofi_window_s)
        # engage threshold on the normalized [-1,+1] trailing OFI
        self.ofi_defensive_thresh = ofi_defensive_thresh
        # ticks to widen the threatened side
        self.ofi_defensive_ticks = ofi_defensive_ticks
        # OFI depth is a separate axis from window/threshold: lets L1, L5, L10 be
        # compared without changing any other knob. 1 = validated L1 path.
        self.ofi_depth_levels = int(ofi_depth_levels)
        if not 1 <= self.ofi_depth_levels <= 10:
            raise ValueError("ofi_depth_levels must be between 1 and 10")
        # --- SIZE THROTTLE (Stage 4; all default False -> byte-identical off) ---
        self.obi_throttle = obi_throttle
        self.ofi_throttle = ofi_throttle
        self.throttle_frac = float(throttle_frac)
        self.obi_throttle_thresh = float(obi_throttle_thresh)
        self.ofi_throttle_thresh = float(ofi_throttle_thresh)
        self.throttle_hold_ms = float(throttle_hold_ms)
        # per-side hold-until timestamps (exchange-ms); None = not throttled. A
        # trigger sets these to now+hold_ms; the cut persists across quote cycles
        # until self.now passes them (this implements the time-box).
        self._throttle_buy_until = None
        self._throttle_sell_until = None
        # --- QDR SIZE THROTTLE state (default off -> byte-identical) ---
        # on/off switch and the depletion fraction that trips the cut
        self.qdr_throttle = qdr_throttle
        self.qdr_throttle_thresh = float(qdr_throttle_thresh)
        # previous-event touch, needed for the event-to-event depletion math
        self._qdr_prev_bb = None
        self._qdr_prev_bq = 0.0
        self._qdr_prev_ba = None
        self._qdr_prev_aq = 0.0
        # trailing (ts_ms, ofi_increment) events; evicted by the time cap on
        # update, capped to the last N events at READ time (min-window semantics)
        self._ofi_events = deque()
        # previous touch (bb, bq, ba, aq) for the Cont-Kukanov L1 increment
        self._ofi_prev_touch = None
        # previous ranked book for multi-level OFI: (bids, asks), each a tuple of
        # (price, aggregate_qty) best-first. L1 mode never reads it.
        self._ofi_prev_depth = None
        # total increments seen today (event-cap warm-up: need >= N seen)
        self._ofi_seen = 0
        # exchange-ms of the first increment today (time-cap warm-up)
        self._ofi_first_ts = None
        # rolling EMA of the market spread (0 until the first quote cycle seeds it);
        # updated in quotes() because that is where the live book is visible.
        self.ema_spread = 0.0
        # published circuit-breaker prices, synced each event by the backtester
        # (see wants_limits above); None until the feed publishes them.
        self.limit_up = None
        self.limit_dn = None

        # Rolling signed trade volume for the toxicity estimate.
        self.flow = deque(maxlen=flow_window)
        # Current exchange time, updated by observe().
        self.now = self.t0
        # Exchange time of the most recent trade (for the quiet test).
        self.last_trade_ms = None
        # Rolling per-event return volatility.
        self.sigma = 0.0

        # Placement: quote at most this many ticks inside the prevailing touch.
        self.improve_ticks = improve_ticks
        # PKR per quote (overrides share-count sizing when set).
        self.size_notional = size_notional
        # Pegging tolerance in ticks (0 = off).
        self.tol_ticks = tol_ticks
        # EMA decay for the volatility estimator (0.05 ~ 13.5-move half-life).
        self.vol_alpha = vol_alpha
        # Last observed mid (EMA vol updates only on actual mid moves).
        self.last_mid = None
        # Running EMA of squared mid returns.
        self.ema_var = 0.0
        # Our last DESIRED quotes, for the pegging hysteresis.
        self.last_desired = {}

        # Diagnostics. The trigger counters are the MONITORING SYSTEM: they record
        # every activation so "did the ramp/cliff ever engage, and how hard" is a
        # measured fact per run, not an inference from P&L.
        self.stats = {
            # how often the reactive jump gate went dark (events)
            "reactive_darkened": 0,
            # how often the viability gate refused to quote
            "no_quote_unviable": 0,
            # how often we produced at least one quote
            "quotes_made": 0,
            # FLAT + time ramp widened the quotes (events)
            "eod_ramp_widen": 0,
            # FLAT + time cliff went dark, both sides (events)
            "eod_cliff_dark": 0,
            # FLAT + lock ramp widened the trapped side (events)
            "lock_ramp_widen": 0,
            # FLAT + lock cliff went dark on the trapped side (events)
            "lock_cliff_dark": 0,
            # HOLDING inside a window: adding side pulled (events, ANY cause)
            "hold_add_pulled": 0,
            # cause split: hold-pull while the TIME window was active
            "hold_add_pulled_time": 0,
            # cause split: hold-pull while the LOCK window was active (these two can
            # sum to more than hold_add_pulled when both windows coincide)
            "hold_add_pulled_lock": 0,
            # HOLDING inside a window: exit side leaned to max-aggressive (events)
            "hold_exit_leaned": 0,
            # moments where the LOW-PRICE TIER (tick floor) set a lock threshold
            # (0 on normal-priced names; non-zero means the tier is doing work)
            "lock_tier_active": 0,
            # events where the exploding widen HIT THE CAP (cap won the min ->
            # the ramp wanted to go wider but was clamped). This is the "cap binds"
            # signal: if it stays ~0 the dual-window spread upgrade is unnecessary.
            "widen_capped": 0,
            # maximum time-urgency reached this run (0 = ramp never engaged)
            "u_time_max": 0.0,
            # maximum lock-urgency reached this run (0 = ramp never engaged)
            "u_lock_max": 0.0,
        }

    # ---- calibration state: called once per event by the backtester ---------
    def observe(self, kind, obj, ts_exch, mid):
        # Advance our clock so the Ho-Stoll horizon tau is current.
        self.now = ts_exch

        # EMA volatility: O(1) per event, updated only when the mid actually
        # moves (unchanged mids carry no volatility information). Replaces the
        # O(window) recompute over the mids deque.
        if mid is not None:
            if self.last_mid is not None and mid != self.last_mid and self.last_mid > 0:
                # Simple return since the last DIFFERENT mid.
                ret = (mid - self.last_mid) / self.last_mid
                # Seed on first move; RiskMetrics-style EMA thereafter.
                if self.ema_var == 0.0:
                    self.ema_var = ret * ret
                else:
                    self.ema_var = self.vol_alpha * ret * ret + (1.0 - self.vol_alpha) * self.ema_var
                self.sigma = math.sqrt(self.ema_var)
            self.last_mid = mid
        # ---- reactive-gate mid history: keep (ts, mid) over the lookback window ----
        # only when the gate is active, to avoid overhead in the off case
        if self.reactive_mode != "off" and mid is not None:
            # append the current observation
            self._mid_hist.append((ts_exch, mid))
            # drop points older than the lookback window
            cutoff = ts_exch - self.reactive_lookback_ms
            while self._mid_hist and self._mid_hist[0][0] < cutoff:
                self._mid_hist.popleft()

        # Trades carry the direction signal Glosten-Milgrom conditions on.
        if kind == "T":
            side = getattr(obj, "aggressor_side", None)
            if side in ("BUY", "SELL"):
                # Signed volume: +buy, -sell.
                self.flow.append(float(obj.qty) * (1.0 if side == "BUY" else -1.0))
                # Reset the quiet clock.
                self.last_trade_ms = ts_exch

    # ---- derived microstructure quantities ---------------------------------
    def _toxicity(self):
        # No trades observed yet -> assume benign.
        if not self.flow:
            return 0.0
        # Net direction of recent flow.
        signed = sum(self.flow)
        # Total recent volume regardless of direction.
        gross = sum(abs(f) for f in self.flow)
        # |net| / gross in [0,1]: 0 = balanced two-way flow (uninformed),
        # 1 = one-directional sweep (the GM signature of informed trading).
        return abs(signed) / gross if gross > 0 else 0.0

    def _quiet(self):
        # No trade seen at all -> treat as quiet.
        if self.last_trade_ms is None:
            return True
        # Ch 6.3: a long gap since the last trade is evidence AGAINST an
        # information event, so it should tighten us, not widen us.
        return (self.now - self.last_trade_ms) > self.quiet_ms

    def _horizon(self):
        # Session length in ms, guarded against zero.
        span = max(self.t1 - self.t0, 1)
        # Ho-Stoll tau: 1.0 at the open, 0.0 at the flatten time. The risk
        # component decays with it, so the spread narrows into the close.
        return max(0.0, min(1.0, (self.t1 - self.now) / span))

    # ---- SZ's POV unwind model (the Weight_Avg_Shares_per_Min.xlsx logic) ----
    # Decide whether the CURRENT inventory can still be cleared passively in the
    # tradeable time remaining, at our participation cap. Mirrors the sheet:
    #   Mins allocation : fills the remaining time from the CLOSE BACKWARD --
    #                     Last15 first (=MIN(15, left)), then Middle, then First15
    #                     (the MIN(I,H) column mechanic).
    #   Shares/Min      : SUMPRODUCT(bucket_rates, bucket_mins)/SUM(bucket_mins)
    #   My Trades/Min   : Shares/Min x POV      (never assume >POV of the tape)
    #   Time Needed     : |inventory| / My Trades/Min
    #   engage unwind  <=> Time Needed >= tradeable minutes left (the "Ramp" cell)
    def _unwind_needed(self, pos):
        # tradeable minutes remaining: sum of the overlap of [now, end] with each
        # continuous segment -- NOT wall clock (Friday's Jumu'ah break must not
        # count as sellable time)
        left_ms = 0
        # total tradeable ms this day (for the Middle bucket's total length)
        total_ms = 0
        for s, e in self.session_segments:
            # this segment's full length
            total_ms += (e - s)
            # the part of it still ahead of us
            if self.now < e:
                left_ms += (e - max(self.now, s))
        # minutes remaining
        left_min = left_ms / 60000.0
        # nothing tradeable left -> cannot clear passively; engage if holding
        if left_min <= 0.0:
            return True
        # bucket TOTALS for this day. 4-bucket profile (2026-08-20): the measured
        # close ramp starts ~60min out (1.3x/1.4x/1.8x midday), so a PreClose45
        # zone (minutes 60->15) sits between Middle and Last15. A 3-tuple profile
        # still works (PreClose45 collapses into Middle -- the old model).
        total_min = total_ms / 60000.0
        if len(self.unwind_profile) == 4:
            vf, vm, vp, vl = self.unwind_profile
            p_total = max(min(45.0, total_min - 30.0), 0.0)
        else:
            vf, vm, vl = self.unwind_profile
            vp, p_total = vm, 0.0
        mid_total = max(total_min - 30.0 - p_total, 1.0)
        # ---- the MIN(I,H) back-fill: allocate remaining minutes close-backward ----
        # Last15 takes the final minutes first
        a_last = min(15.0, left_min)
        # PreClose45 takes the next block (zero-length under a 3-tuple profile)
        a_pre = min(p_total, left_min - a_last)
        # Middle takes what remains, up to its day total
        a_mid = min(mid_total, left_min - a_last - a_pre)
        # First15 takes any residue (nonzero only inside the opening cap)
        a_first = max(0.0, left_min - a_last - a_pre - a_mid)
        # ---- weighted-average expected shares/min over the remaining time ----
        exp_vol = (a_first * vf + a_mid * vm + a_pre * vp + a_last * vl) / left_min
        # a dead tape -> cannot clear passively; engage
        if exp_vol <= 0.0:
            return True
        # our clearable rate at the participation cap
        my_rate = exp_vol * self.unwind_pov
        # minutes needed to clear the CURRENT position (live state, not max_inv)
        mins_needed = abs(pos) / my_rate
        # the sheet's Normal/Ramp decision
        return mins_needed >= left_min

    # ---- EOD / LOCK trigger state (one call per quote cycle) ----------------
    def _trigger_state(self, bb, ba, pos):
        """Evaluate both triggers and return the per-side actions.

        Returns a dict:
          kill_buy / kill_sell : hard suppression of that side (cliffs / holding rule)
          u_buy / u_sell       : widen urgency for that side (0 = no widening)
          lean_exit            : holding inside a window -> lean the exit side
        Direction logic (sign-critical, verified against pinned()):
          UPPER band: pinned when bb >= limit_up; the TRAPPED position is SHORT
          (you cannot buy back above the cap), so the dangerous acquisition is a
          SELL fill -> near the upper band we suppress/widen the SELL side.
          LOWER band: mirrored; trapped is LONG; suppress/widen the BUY side.
        """
        # default: no action on either side
        act = {"kill_buy": False, "kill_sell": False,
               "u_buy": 0.0, "u_sell": 0.0, "lean_exit": False}
        # nothing enabled -> zero-cost early exit (PPL/UBL default path)
        if not (self.enable_eod_trigger or self.enable_lock_trigger):
            return act
        # flat = less than one clip of inventory (sub-clip residue treated as flat)
        is_flat = abs(pos) < self.size0
        # current market spread in PKR (book is two-sided when quotes() runs)
        spr = ba - bb
        # window flags: is ANY trigger currently in its ramp-or-cliff zone?
        in_time_window = False
        in_lock_window = False

        # ---------------- time-to-close trigger ----------------
        if self.enable_eod_trigger:
            # minutes remaining until the continuous close (t1 = the true bell)
            mins_left = (self.t1 - self.now) / 60000.0
            # CLIFF: final eod_cliff_min minutes -> flat goes dark on BOTH sides
            if mins_left <= self.eod_cliff_min:
                # inside the cliff zone regardless of holding state
                in_time_window = True
                # flat: no new position of either sign this close to the bell
                if is_flat:
                    act["kill_buy"] = True
                    act["kill_sell"] = True
                    self.stats["eod_cliff_dark"] += 1
            # RAMP: exploding widen between ramp-start and the cliff
            elif mins_left < self.eod_ramp_start_min:
                # inside the ramp zone
                in_time_window = True
                # inverse shape: 0 at ramp start, explodes toward the cliff
                u_t = (self.eod_ramp_start_min / mins_left) - 1.0
                # monitoring: record the deepest urgency reached
                self.stats["u_time_max"] = max(self.stats["u_time_max"], u_t)
                # flat: widen BOTH sides (any acquisition is unwanted near the bell)
                if is_flat:
                    act["u_buy"] = max(act["u_buy"], u_t)
                    act["u_sell"] = max(act["u_sell"], u_t)
                    self.stats["eod_ramp_widen"] += 1

        # ---------------- distance-to-lock trigger (PERCENT OF PRICE) ----------
        # Uses ONLY the exchange-published band prices (UPPER/LOWER_CIRCUIT_BREAKER
        # rows -> limit_up/limit_dn, synced by the engine). Distance is measured
        # FROM THE MID (not the touch) as a % of price -- fully spread-free, so a
        # wide-spread flicker cannot fire it (the 2026-08 diagnostic bug).
        # cliff flags per cause (the zone-split holding rule needs cliff-vs-ramp)
        in_time_cliff = False
        in_lock_cliff = False
        # the time branch above set in_time_window; recompute its cliff flag here
        if self.enable_eod_trigger:
            # cliff = the final eod_cliff_min minutes
            in_time_cliff = ((self.t1 - self.now) / 60000.0) <= self.eod_cliff_min
        if self.enable_lock_trigger and self.limit_up is not None \
                and self.limit_dn is not None:
            # the mid: the price reference for the % distance (spread-invariant)
            mid_px = 0.5 * (bb + ba)
            # one tick expressed as a % of price (drives the low-price tier floors)
            tick_pct = (self.tick / mid_px) * 100.0 if mid_px > 0 else 0.0
            # EFFECTIVE thresholds: "% of price OR N ticks, whichever is larger" --
            # the low-price tier. On normal-priced names the % values bind; on cheap
            # names the tick floors bind so the zones are always jump-proof.
            cliff_pct = max(self.lock_cliff_pct, self.min_cliff_ticks * tick_pct)
            ramp_pct = max(self.lock_ramp_pct,
                           cliff_pct + self.min_ramp_gap_ticks * tick_pct)
            # monitoring: count moments where the tick floor (tier) set a threshold
            if cliff_pct > self.lock_cliff_pct or ramp_pct > self.lock_ramp_pct:
                self.stats["lock_tier_active"] += 1
            # distance from the MID to each band, as % of price (0 = at the band)
            d_up_pct = max(0.0, self.limit_up - mid_px) / mid_px * 100.0
            d_dn_pct = max(0.0, mid_px - self.limit_dn) / mid_px * 100.0

            # --- upper band: trapped position is SHORT -> act on the SELL side ---
            if d_up_pct <= cliff_pct:
                # inside the upper CLIFF zone
                in_lock_window = True
                in_lock_cliff = True
                # flat: go dark on the trapped side only (BUY stays -- being long
                # into a limit-up close is the SAFE side: sell into stacked bids)
                if is_flat:
                    act["kill_sell"] = True
                    self.stats["lock_cliff_dark"] += 1
            elif d_up_pct < ramp_pct:
                # inside the upper RAMP zone
                in_lock_window = True
                # inverse shape: 0 at ramp start, explodes toward the cliff
                u_l = (ramp_pct / max(d_up_pct, 1e-9)) - 1.0
                # monitoring: deepest lock urgency reached
                self.stats["u_lock_max"] = max(self.stats["u_lock_max"], u_l)
                # flat: widen the trapped (SELL) side only
                if is_flat:
                    act["u_sell"] = max(act["u_sell"], u_l)
                    self.stats["lock_ramp_widen"] += 1

            # --- lower band: trapped position is LONG -> act on the BUY side ---
            if d_dn_pct <= cliff_pct:
                # inside the lower CLIFF zone
                in_lock_window = True
                in_lock_cliff = True
                # flat: go dark on the trapped side only
                if is_flat:
                    act["kill_buy"] = True
                    self.stats["lock_cliff_dark"] += 1
            elif d_dn_pct < ramp_pct:
                # inside the lower RAMP zone
                in_lock_window = True
                # inverse shape toward the lower cliff
                u_l = (ramp_pct / max(d_dn_pct, 1e-9)) - 1.0
                # monitoring
                self.stats["u_lock_max"] = max(self.stats["u_lock_max"], u_l)
                # flat: widen the trapped (BUY) side only
                if is_flat:
                    act["u_buy"] = max(act["u_buy"], u_l)
                    self.stats["lock_ramp_widen"] += 1

        # ---------------- window state (for fill tagging) -----------------------
        # record WHICH window this quote cycle is in, cliffs taking priority over
        # ramps (a fill during an overlap is attributed to the harder zone). The
        # backtester reads this attribute at fill time and stamps every fill with
        # it -- making "did we actually sell during the cliff?" directly answerable.
        if in_time_cliff:
            self.current_window = "time_cliff"
        elif in_lock_cliff:
            self.current_window = "lock_cliff"
        elif in_time_window:
            self.current_window = "time_ramp"
        elif in_lock_window:
            self.current_window = "lock_ramp"
        else:
            self.current_window = "none"
        # ---------------- holding rule: UNWIND (time) + ZONE-SPLIT (lock) -------
        # TIME trigger while holding: pull the adding side + lean the exit across the
        # WHOLE window (ramp AND cliff). Justified because the ramp start is now
        # POV-SIZED per name (Minutes = max_inv / (vol_per_min x MAX_POV)): the window
        # opens exactly when the remaining closing volume can still absorb our
        # inventory at our participation cap -- so "stop adding, work the exit" is the
        # correct behaviour for the entire window, not just the last minute.
        # LOCK trigger while holding: ZONE-SPLIT kept (the 2026-08 fix) -- the lock
        # ramp is a price-proximity warning, not a liquidity budget; holding inside it
        # keeps BOTH sides quoted (base skew leans) and only the lock CLIFF pulls.
        # engagement rule for the TIME-driven unwind:
        #   NEW (unwind_profile set): SZ's POV model -- engage only when the
        #   CURRENT inventory cannot clear in the tradeable time left at our
        #   participation cap. Dormant whenever inventory is comfortably small;
        #   fires early when genuinely loaded on a thin day.
        #   OLD (no profile): the fixed clock window (holding inside ramp/cliff).
        if self.unwind_profile is not None and self.session_segments is not None:
            time_unwind = (not is_flat) and self.enable_eod_trigger \
                and self._unwind_needed(pos)
        else:
            time_unwind = in_time_window and not is_flat
        if (time_unwind or in_lock_cliff) and not is_flat:
            # long: the adding side is BUY; short: the adding side is SELL
            if pos > 0:
                act["kill_buy"] = True
            else:
                act["kill_sell"] = True
            # count the pull (overall)
            self.stats["hold_add_pulled"] += 1
            # the exit side gets the bounded lean (max-aggressive post-only: the
            # touch). PURE MAKER: never crosses the spread -- the lean is the most
            # aggressive PASSIVE placement, full stop.
            act["lean_exit"] = True
            # count the lean (overall)
            self.stats["hold_exit_leaned"] += 1
            # CAUSE SPLIT: attribute to whichever cause was active; both count
            # when both are (their sum can exceed the overall -- the overlap signal)
            if time_unwind:
                self.stats["hold_add_pulled_time"] += 1
            if in_lock_cliff:
                self.stats["hold_add_pulled_lock"] += 1
            # the exit side is never widened or killed: undo anything a lock cliff
            # set on it. Exit access wins -- a trapped holder still posts (best
            # effort) on the exit side.
            if pos > 0:
                act["kill_sell"] = False
                act["u_sell"] = 0.0
            else:
                act["kill_buy"] = False
                act["u_buy"] = 0.0
        # return the per-side action set
        return act

    # ---- OFI-defensive machinery (Stage 3) ----------------------------------
    def _ofi_append(self, e):
        # append one signed OFI increment and maintain the trailing window's
        # warm-up state. Shared by the L1 and deep paths so both feed ONE window.
        self._ofi_events.append((self.now, float(e)))
        # warm-up counter: total increments seen today
        self._ofi_seen += 1
        # first-increment timestamp for the time-cap warm-up
        if self._ofi_first_ts is None:
            self._ofi_first_ts = self.now
        # evict entries older than the TIME cap (event cap applied at read)
        cutoff = self.now - self.ofi_window_s * 1000.0
        while self._ofi_events and self._ofi_events[0][0] < cutoff:
            self._ofi_events.popleft()

    def _ofi_update(self, bb, bq, ba, aq):
        # Cont-Kukanov L1 order-flow increment between the PREVIOUS touch state
        # and this one. Positive = net buying pressure. quotes() sees every event
        # inside the session, so consecutive touches here = event-level OFI (the
        # same definition as the feature store's ofi_l1 that won the horserace).
        if self._ofi_prev_touch is not None:
            # unpack the previous touch
            pbb, pbq, pba, paq = self._ofi_prev_touch
            # bid-side flow: improve -> +new qty; same level -> qty delta;
            # retreat -> -old qty (bids pulled/consumed)
            if bb > pbb:
                e_bid = bq
            elif bb == pbb:
                e_bid = bq - pbq
            else:
                e_bid = -pbq
            # ask-side flow (mirrored): improve (down) -> +new qty; same -> delta;
            # retreat (up) -> -old qty
            if ba < pba:
                e_ask = aq
            elif ba == pba:
                e_ask = aq - paq
            else:
                e_ask = -paq
            # the signed increment: buy pressure minus sell pressure
            e = float(e_bid - e_ask)
            # append to the shared trailing window
            self._ofi_append(e)
        # remember this touch for the next increment
        self._ofi_prev_touch = (bb, bq, ba, aq)

    @staticmethod
    def _ofi_rank_increment(previous, current, side):
        # Cont-Kukanov increment for ONE ranked price level. previous/current are
        # (price, qty) or None. Positive always = buy pressure: bid additions/
        # improvements and ask removals/retreats are positive; mirrors negative.
        if previous is None and current is None:
            return 0.0
        if previous is None:
            _, qty = current
            return float(qty if side == "BUY" else -qty)
        if current is None:
            _, qty = previous
            return float(-qty if side == "BUY" else qty)
        old_px, old_qty = previous
        new_px, new_qty = current
        if side == "BUY":
            if new_px > old_px:
                return float(new_qty)
            if new_px < old_px:
                return float(-old_qty)
            return float(new_qty - old_qty)
        if new_px < old_px:
            return float(-new_qty)
        if new_px > old_px:
            return float(old_qty)
        return float(old_qty - new_qty)

    def _ofi_update_depth(self, depth):
        # Equal-weight multi-level OFI from a ranked depth snapshot. The scalar
        # increment is the SUM of the canonical increment at ranks 1..N. Leaves
        # depth N as an experiment axis rather than baking in a decay curve.
        # depth = (bids, asks), each a list of (price, qty) best-first.
        bids, asks = depth
        # cap to the configured number of levels
        bids = tuple(bids[:self.ofi_depth_levels])
        asks = tuple(asks[:self.ofi_depth_levels])
        # this cycle's ranked book
        current = (bids, asks)
        # need a previous ranked book to difference against
        if self._ofi_prev_depth is not None:
            # unpack previous ranked book
            old_bids, old_asks = self._ofi_prev_depth
            # accumulate the per-rank increments
            e = 0.0
            for rank in range(self.ofi_depth_levels):
                # previous/current (price,qty) at this rank, or None if absent
                old_bid = old_bids[rank] if rank < len(old_bids) else None
                new_bid = bids[rank] if rank < len(bids) else None
                old_ask = old_asks[rank] if rank < len(old_asks) else None
                new_ask = asks[rank] if rank < len(asks) else None
                # bid-side + ask-side increments at this rank
                e += self._ofi_rank_increment(old_bid, new_bid, "BUY")
                e += self._ofi_rank_increment(old_ask, new_ask, "SELL")
            # one scalar increment for the whole depth snapshot
            self._ofi_append(e)
        # remember this ranked book for the next increment
        self._ofi_prev_depth = current

    def _ofi_signal(self):
        # The NORMALIZED trailing OFI in [-1, +1], or None when not warm / not
        # applicable. Warm-up guard (mirrors the horserace exactly): the signal
        # is OFF until (a) >= N increments have been seen today (event cap warm),
        # (b) >= T seconds have elapsed since the first increment (time cap warm),
        # and (c) the current window holds >= 2 events. NaN-equivalent = None.
        if not (self.ofi_defensive or self.ofi_throttle):
            return None
        # hard-OFF in first15: the horserace measured ZERO signal there on every
        # window (thin frenetic book: state = flow); acting on noise only hurts.
        if self._bucket_now() == "first15":
            return None
        # event-cap warm-up: need to have SEEN at least N increments
        if self._ofi_seen < self.ofi_window_ev:
            return None
        # time-cap warm-up: need T seconds of history since the first increment
        if self._ofi_first_ts is None \
                or (self.now - self._ofi_first_ts) < self.ofi_window_s * 1000.0:
            return None
        # the deque already holds only the last T seconds; the MIN window binds
        # by whichever holds fewer events -> cap to the last N entries
        ev = list(self._ofi_events)
        if len(ev) > self.ofi_window_ev:
            ev = ev[-self.ofi_window_ev:]
        # need at least 2 events in the bound window
        if len(ev) < 2:
            return None
        # normalized signed flow: sum(e) / sum(|e|) in [-1, +1]
        s = sum(x for _, x in ev)
        a = sum(abs(x) for _, x in ev)
        # degenerate (all-zero) window -> no signal
        if a <= 0:
            return None
        # the bounded, threshold-comparable signal
        return s / a

    def _bucket_now(self):
        # canonical session bucket of self.now, keyed off the session segments
        # (first15/middle/preclose45/last15). Falls back to t0/t1 when segments
        # are absent.
        if self.session_segments:
            open_ms = self.session_segments[0][0]
            close_ms = self.session_segments[-1][1]
        else:
            open_ms, close_ms = self.t0, self.t1
        # first 15 minutes after the open
        if self.now < open_ms + 15 * 60000:
            return "first15"
        # last 15 minutes before the close
        if self.now >= close_ms - 15 * 60000:
            return "last15"
        # the 45 minutes before last15
        if self.now >= close_ms - 60 * 60000:
            return "preclose45"
        # everything else
        return "middle"

    # ---- quoting -----------------------------------------------------------
    def quotes(self, bb, bq, ba, aq, pos, depth=None):
        # No two-sided book with depth on both sides -> nothing to quote against.
        if bb is None or ba is None or bq <= 0 or aq <= 0:
            return {}
        # ---- REACTIVE JUMP GATE: react to a large recent move by going dark ----
        # Checked first: if we are inside an active cooldown, quote nothing at all.
        # Then test for a fresh trigger over the lookback window. "Adverse" (for
        # the inventory mode) = move against the current position: long + price
        # DOWN, or short + price UP. sigma here is the per-event EMA vol scaled to
        # the lookback horizon is complex; we use a simpler, robust test: the raw
        # move over the window vs reactive_k * (trailing sigma * sqrt(n_moves)).
        if self.reactive_mode != "off":
            # still inside a cooldown -> stay dark
            if self.now < self._dark_until:
                self.stats["reactive_darkened"] += 1
                return {}
            # enough history to measure a move?
            if len(self._mid_hist) >= 2:
                m0 = self._mid_hist[0][1]
                m1 = self._mid_hist[-1][1]
                if m0 > 0:
                    # signed move over the window (fractional return)
                    move = (m1 - m0) / m0
                    # trigger size: k * per-event sigma, scaled to the window by
                    # the number of observations in it (random-walk sqrt scaling).
                    n = max(1, len(self._mid_hist) - 1)
                    thresh = self.reactive_k * self.sigma * math.sqrt(n)
                    # is the move large enough?
                    big = abs(move) >= thresh and thresh > 0
                    # adverse to inventory? long hurt by a drop, short by a rise
                    adverse = (pos > 0 and move < 0) or (pos < 0 and move > 0)
                    # fire per mode
                    fire = big and (self.reactive_mode == "symmetric"
                                    or (self.reactive_mode == "inventory" and adverse))
                    if fire:
                        # open a cooldown and go dark now
                        self._dark_until = self.now + self.reactive_cooldown_ms
                        self.stats["reactive_darkened"] += 1
                        return {}
        # Bid share of top-of-book depth.
        imb = bq / (bq + aq)
        # --- OFI-defensive: fold this touch into the trailing flow window and
        # compute the normalized signal ONCE for both sides. Runs ONLY when the
        # master switch is on -- with it off this path never executes, so
        # ofi_defensive=False is byte-identical to the pre-Stage-3 strategy
        # (the before/after test asserts this).
        if self.ofi_defensive:
            # L1 retains the validated touch path byte-for-byte; deep mode
            # consumes ranked market depth supplied by the engine.
            if self.ofi_depth_levels == 1:
                self._ofi_update(bb, bq, ba, aq)
            else:
                if depth is None:
                    raise ValueError(
                        "deep OFI requires ranked depth passed to quotes()")
                self._ofi_update_depth(depth)
            # the normalized [-1,+1] signal, or None (warm-up / first15 / off)
            ofi_sig = self._ofi_signal()
        else:
            ofi_sig = None
        # --- SIZE THROTTLE (Stage 4): feed the OFI window when the throttle
        # needs it but ofi_defensive is off, then decide per-side throttling. ---
        if self.ofi_throttle and not self.ofi_defensive:
            # keep the trailing OFI window fed so ofi_sig is valid for throttling
            if self.ofi_depth_levels == 1:
                self._ofi_update(bb, bq, ba, aq)
            else:
                if depth is None:
                    raise ValueError(
                        "deep OFI requires ranked depth passed to quotes()")
                self._ofi_update_depth(depth)
            # the normalized signal for the throttle path
            ofi_sig = self._ofi_signal()
        # --- QDR SIZE THROTTLE: own-side touch queue being eaten fast at an
        # UNCHANGED best price => imminent adverse move on that side. Computed
        # event-to-event (quotes() runs every event), the SAME definition as the
        # feature builder. Default off (qdr_throttle=False) => qdr_*_trig stay
        # False and prev-touch is never updated => byte-identical to before.
        qdr_buy_trig = False
        qdr_sell_trig = False
        if self.qdr_throttle:
            # fraction of the BID queue that vanished since last event at a stable best-bid
            qdr_bid = ((self._qdr_prev_bq - bq) / self._qdr_prev_bq) \
                if (self._qdr_prev_bb is not None and bb == self._qdr_prev_bb
                    and self._qdr_prev_bq > 0 and bq < self._qdr_prev_bq) else 0.0
            # fraction of the ASK queue that vanished since last event at a stable best-ask
            qdr_ask = ((self._qdr_prev_aq - aq) / self._qdr_prev_aq) \
                if (self._qdr_prev_ba is not None and ba == self._qdr_prev_ba
                    and self._qdr_prev_aq > 0 and aq < self._qdr_prev_aq) else 0.0
            # bid eaten fast => protect the BUY side; ask eaten fast => protect the SELL side
            qdr_buy_trig = qdr_bid >= self.qdr_throttle_thresh
            qdr_sell_trig = qdr_ask >= self.qdr_throttle_thresh
            # remember THIS touch for the next event's depletion computation
            self._qdr_prev_bb, self._qdr_prev_bq = bb, bq
            self._qdr_prev_ba, self._qdr_prev_aq = ba, aq
        # BUY-side trigger: adverse OBI (ask-heavy), adverse OFI (selling flow), or fast bid depletion
        buy_trig = ((self.obi_throttle
                     and (0.5 - imb) > self.obi_throttle_thresh)
                    or (self.ofi_throttle and ofi_sig is not None
                        and ofi_sig < -self.ofi_throttle_thresh)
                    or qdr_buy_trig)
        # SELL-side trigger: adverse OBI (bid-heavy), adverse OFI (buying flow), or fast ask depletion
        sell_trig = ((self.obi_throttle
                      and (imb - 0.5) > self.obi_throttle_thresh)
                     or (self.ofi_throttle and ofi_sig is not None
                         and ofi_sig > self.ofi_throttle_thresh)
                     or qdr_sell_trig)
        # time-box: a fresh trigger (re)arms the hold to now+hold_ms; with
        # hold_ms=0 the hold expires immediately (only the trigger cycle cut)
        if buy_trig:
            self._throttle_buy_until = self.now + self.throttle_hold_ms
        if sell_trig:
            self._throttle_sell_until = self.now + self.throttle_hold_ms
        # a side is throttled NOW if its hold is set and not yet expired
        buy_throttled = (self._throttle_buy_until is not None
                         and self.now <= self._throttle_buy_until)
        sell_throttled = (self._throttle_sell_until is not None
                          and self.now <= self._throttle_sell_until)
        # Ch 3.3 microprice, GENERALIZED to a continuous lean coefficient lambda:
        #   fair = mid + lambda * (imb - 0.5) * spread
        # lambda=+1 == the classic microprice (heavy bid -> fair UP, the confirmed
        #   destructive lean); lambda=0 == plain mid; lambda<0 == DEFENSIVE lean
        #   (heavy bid -> fair DOWN -> our bid pulls BACK from the buying pressure,
        #   converting 'through' pick-offs toward 'at_queue' fills). Proven equal
        #   to the old boolean code at lambda in {0,1} (100k-case identity test).
        # Resolve lambda: explicit micro_lambda wins; else fall back to the boolean
        # (True->1.0, False->0.0) so existing callers are byte-identical.
        if self.micro_lambda is not None:
            _lam = self.micro_lambda
        else:
            _lam = 1.0 if self.use_microprice else 0.0
        # mid and spread
        _mid = 0.5 * (bb + ba)
        _spr = ba - bb
        # the continuous-lambda fair value
        fair = _mid + _lam * (imb - 0.5) * _spr
        # Ho-Stoll horizon.
        tau = self._horizon()
        # Per-unit inventory risk: risk aversion x variance x remaining horizon.
        inv_risk = self.gamma * (self.sigma ** 2) * tau
        # --- inventory skew, corrected to PKR variance (Ho-Stoll / A-S) --------
        # BUG FIX: self.sigma is a per-event FRACTIONAL-return vol (~1e-4). The old
        # skew inv_risk*pos*fair therefore used sigma^2 ~ 1e-8 and produced ~7.5e-5
        # PKR at full inventory -- under 1/100th of a tick, so inventory control was
        # inert. Convert sigma to a PKR PRICE vol before squaring.
        sigma_p = self.sigma * fair
        # Remaining inventory variance over the session: PKR-variance x a calibrated
        # dimensional bridge (units 1/PKR) x the remaining-time fraction. LINEAR in
        # tau -- A-S is linear in time-to-horizon and tau already encodes (T-now)/span.
        remaining_var = (sigma_p ** 2) * self.session_scale * tau
        # Inventory in LOTS (pos / base clip), not raw shares, so the lean scales with
        # how many clips we are from flat. session_scale was calibrated against this.
        pos_lots = pos / self.size0
        # Ch 2.2/2.3: inventory SHIFTS the quote pair. Long (pos>0) -> skew>0 ->
        # reservation below fair -> sell eagerly / buy less. Width left untouched.
        skew = self.gamma * remaining_var * pos_lots
        # The reservation price: our own indifference value, fair value less skew.
        reservation = fair - skew
        # Avellaneda-Stoikov base half-spread; inert until kappa is calibrated.
        as_base = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa) \
            if self.gamma > 0 else 0.0
        # Half-spread component compensating inventory/price risk over tau.
        half_risk = 0.5 * inv_risk * fair
        # Current adverse-selection estimate from flow one-sidedness.
        tox = self._toxicity()
        # Ch 6.3: quiet market -> halve the adverse-selection charge (tighten).
        if self._quiet():
            tox *= 0.5
        # Adverse-selection half-spread: scales with toxicity and volatility.
        half_adverse = tox * (self.sigma * fair) * 2.0
        # Ch 7.1: the hard economic floor -- fees plus the edge we insist on.
        cost_floor = (self.fee_pct + self.min_edge_pct) * fair
        # Total half-spread: risk + adverse selection + (optional A-S base) + floor.
        half = half_risk + half_adverse + self.as_base_weight * as_base * fair + cost_floor
        # Never quote inside one tick.
        half = max(half, self.tick)
        # ---- EOD / LOCK TRIGGERS: evaluated BEFORE the viability gate, because the
        # UNWIND (lean_exit) must be able to post the exit side even when the market
        # spread is too tight for profitable quoting -- during the unwind we are
        # deliberately paying edge to get flat (min_edge is waived on the exit side).
        trig = self._trigger_state(bb, ba, pos)
        # INVENTORY-EXIT bypass: when we are holding past the exit threshold and
        # the exit skew is active, the exit side is trying to GET FLAT -- the same
        # "getting flat > earning edge" logic as the EOD unwind. So it may post on
        # a tight book too, bounded by the EXACT fee floor in the skew placement
        # (never below fee-solvency). When FLAT (skew inactive) the gate applies
        # normally: no inventory to exit -> quotes must be fully viable.
        holding_exit = (self.exit_ticks_inside > 0
                        and abs(pos_lots) >= self.exit_inv_threshold)
        # If holding_exit is bypassing the gate on a tight book, we must post ONLY
        # the exit side -- never ADD inventory on an unviable book. Suppress the
        # adding side (long -> BUY adds; short -> SELL adds). This mirrors the
        # holding-rule pull, but applies mid-session too, which is exactly the
        # case the inventory-exit bypass opens up. Only bites when the gate would
        # otherwise have blocked (tight book); on a viable book this is harmless
        # because the adding side quotes normally below.
        gate_would_block = self.require_viable and (ba - bb) < 2.0 * half
        if holding_exit and gate_would_block and not trig["lean_exit"]:
            # long: adding side is BUY; short: adding side is SELL
            if pos > 0:
                trig["kill_buy"] = True
            else:
                trig["kill_sell"] = True
            self.stats["hold_add_pulled"] += 1
        # Ch 7.1 VIABILITY GATE: if the market's spread is narrower than the
        # spread we require, no profitable passive quote exists -> stand aside.
        # NOTE: evaluated on the BASE half, before any trigger widening -- the
        # widening is deliberate unfillable-ness, not an economics test.
        # BYPASS when lean_exit (EOD unwind) OR holding_exit (inventory skew): both
        # are deliberately paying edge to get flat, fee-floored in placement, and
        # in both cases the ADDING side is suppressed so only the exit posts.
        if self.require_viable and (ba - bb) < 2.0 * half \
                and not trig["lean_exit"] and not holding_exit:
            self.stats["no_quote_unviable"] += 1
            return {}

        # Ch 3.4: informed flow prefers size, so cut OUR size when flow is toxic.

        # WIDE-MARKET PLACEMENT (the key change for KTML-class names): the gate
        # above checked economic viability with the COST-based half. For placement,
        # never quote tighter than one improve_ticks inside the prevailing touch --
        # on a 53bps market, capture ~the full spread instead of compressing it
        # to our 35bps floor and donating the difference.
        mkt_half = (ba - bb) / 2.0
        half = max(half, mkt_half - self.improve_ticks * self.tick)
        # NOTIONAL SIZING: 50 shares is 2.8k PKR on KTML but 20k on MCB. Fix the
        # PKR-at-risk per quote instead; fall back to share count if unset.
        base_size = (self.size_notional / fair) if self.size_notional else self.size0
        size = base_size * (0.5 if tox > 0.5 else 1.0)

        size = max(1.0, round(size))

        # ---- EOD / LOCK TRIGGERS: per-side actions (trig computed above, before
        # the viability gate -- see the unwind exception there) ----
        # current market spread (distance units use THIS -- see SPREAD_ALPHA note)
        spr = ba - bb
        # update the rolling spread EMA (seed on first cycle)
        if self.ema_spread == 0.0:
            self.ema_spread = spr
        else:
            self.ema_spread = self.spread_alpha * spr + (1.0 - self.spread_alpha) * self.ema_spread
        # cap REFERENCE: a momentarily tight book cannot collapse the cap, a
        # genuinely widening book still grows it.
        ref_spr = max(spr, self.ema_spread)
        # per-side halves: apply the exploding widen, capped at k x reference
        # spread, and never BELOW the base half (the cap is a ceiling, not a target)
        half_buy = half
        # widen the BUY side if a trigger set urgency on it
        if trig["u_buy"] > 0.0:
            # the two candidates: the raw exploded half, and the cap ceiling
            raw = half * (1.0 + trig["u_buy"])
            cap = self.widen_cap_spreads * ref_spr
            # apply cap, keep at/above base half
            half_buy = max(half, min(raw, cap))
            # the cap BOUND if it was the smaller of the two (it clamped the ramp)
            if cap < raw:
                self.stats["widen_capped"] += 1
        # widen the SELL side if a trigger set urgency on it
        half_sell = half
        if trig["u_sell"] > 0.0:
            # same two candidates for the sell side
            raw = half * (1.0 + trig["u_sell"])
            cap = self.widen_cap_spreads * ref_spr
            # apply cap, keep at/above base half
            half_sell = max(half, min(raw, cap))
            # count a bind (note: both sides binding in one cycle counts twice,
            # which is the intended "how many side-widenings got capped" measure)
            if cap < raw:
                self.stats["widen_capped"] += 1

        out = {}
        # Bid unless: trigger-killed, long at the hard cap, or long past the soft
        # band (past +soft_inv we stop buying so fills can only reduce a long).
        if (not trig["kill_buy"]) and pos < self.max_inv \
                and (self.soft_inv is None or pos < self.soft_inv):
            # Floor onto the tick grid so rounding never makes us more aggressive.
            px = math.floor((reservation - half_buy) / self.tick) * self.tick
            # Bounded exit lean: SHORT + inside a trigger window -> BUY is the exit;
            # place it at the most aggressive post-only price (the bound is
            # structural: post-only cannot cross, so ba - tick is the ceiling).
            if trig["lean_exit"] and pos < 0:
                px = ba - self.tick
            # --- SWEEP AXIS 1: inventory-exit tick skew (bb + N*tick, venue-agnostic) ---
            # BUY is the EXIT side when we are SHORT (pos < 0). Improve our own bid
            # by exit_ticks_inside TICKS above the best bid (N*self.tick scales to
            # ANY venue's tick automatically). This sets the DESIRED price; the
            # engine's post-only clip below then governs it exactly as it governs
            # the base quote -- one gate, applied last. N=0 -> no change.
            if (self.exit_ticks_inside > 0 and pos < 0
                    and -pos_lots >= self.exit_inv_threshold):
                # improve our bid N ticks above the best bid
                improved = bb + self.exit_ticks_inside * self.tick
                # OPTION B fee floor: the exit skew may spend the min_edge cushion
                # (that IS the idea -- waive edge to get flat) but must NEVER buy
                # above the fee-covering price, or the round trip loses on fees.
                true_mid = 0.5 * (bb + ba)
                # EXACT fee ceiling. The exchange charges the fee on the ACTUAL
                # fill price -- and for a limit BUY the fill price IS px (the level
                # we post at). Capture on a BUY at px is (mid - px). Fee-covering
                # requires capture >= fee charged on px:
                #     mid - px >= fee_pct * px   ->   px <= mid / (1 + fee_pct)
                # So px is its OWN fee base (the truth), not fee_pct*fair or
                # fee_pct*mid. N-independent, venue-safe. Ceiling for a BUY:
                fee_ceiling_buy = true_mid / (1.0 + self.fee_pct)
                # take the improvement, but never above the fee-covering ceiling
                improved = min(improved, fee_ceiling_buy)
                # only ever MORE aggressive than the base quote (never less)
                px = max(px, improved)
            # --- SWEEP AXIS 2: OBI-defensive widen ---
            # imb = bq/(bq+aq); imb < 0.5 = ask-heavy (sellers stacked) -> buying
            # here is adverse (price likely to fall). Widen our BUY (lower px) so we
            # stop resting in front of that selling pressure. Engages only when the
            # imbalance against us exceeds the threshold.
            if self.obi_defensive and (0.5 - imb) > self.obi_defensive_thresh:
                # push the bid DOWN by the defensive ticks (less likely to fill)
                px = px - self.obi_defensive_ticks * self.tick
            # --- STAGE 3: OFI-defensive widen (RETREAT-ONLY, mirrors OBI) ---
            # ofi_sig < -thresh = sustained SELLING flow -> price likely to fall
            # -> buying here is adverse -> push the bid DOWN. Never improves a
            # quote toward the pressure (the anti-microprice guard); the post-only
            # clip below still governs last. None = off/warm-up/first15.
            if ofi_sig is not None and ofi_sig < -self.ofi_defensive_thresh:
                # widen the threatened bid by the OFI defensive ticks
                px = px - self.ofi_defensive_ticks * self.tick
            # Post-only clip (the engine's cross guard): stay at least one tick
            # inside the ask. Applied LAST so it governs the base quote, the exit
            # skew, and the OBI widen identically -- nothing can cross the ask.
            px = min(px, ba - self.tick)
            # SIZE THROTTLE: cut BUY clip while throttled (>=1 share). When no
            # throttle switch is on, buy_throttled is False -> size unchanged.
            size_buy = max(1.0, round(size * self.throttle_frac)) \
                if buy_throttled else size
            out["BUY"] = (round(px, 2), size_buy)
        # Offer unless: trigger-killed, short at the hard cap, or short past the
        # soft band (past -soft_inv we stop selling so only the bid remains).
        if (not trig["kill_sell"]) and pos > -self.max_inv \
                and (self.soft_inv is None or pos > -self.soft_inv):
            # Ceil onto the tick grid (again, never more aggressive).
            px = math.ceil((reservation + half_sell) / self.tick) * self.tick
            # Bounded exit lean: LONG + inside a trigger window -> SELL is the exit;
            # most aggressive post-only placement is bb + tick.
            if trig["lean_exit"] and pos > 0:
                px = bb + self.tick
            # --- SWEEP AXIS 1: inventory-exit tick skew (ba - N*tick, venue-agnostic) ---
            # SELL is the EXIT side when we are LONG (pos > 0). Improve our own ask
            # by exit_ticks_inside TICKS below the best ask (N*self.tick scales to
            # any venue's tick). Sets the DESIRED price; the engine's post-only
            # clip below governs it exactly as it governs the base quote. N=0 -> no change.
            if (self.exit_ticks_inside > 0 and pos > 0
                    and pos_lots >= self.exit_inv_threshold):
                # improve our ask N ticks below the best ask
                improved = ba - self.exit_ticks_inside * self.tick
                # OPTION B raw-fee floor: the exit skew may spend the min_edge
                # cushion but must NEVER sell below the fee-covering price. Floor
                # for a SELL = mid + fee (selling any lower earns less than the
                # fee). True mid, venue-safe (scales with fee_pct/fair, any N).
                true_mid = 0.5 * (bb + ba)
                # EXACT fee floor (mirror of BUY). Fee is charged on the fill
                # price; capture on a SELL at px is (px - mid). Fee-covering:
                #     px - mid >= fee_pct * px   ->   px >= mid / (1 - fee_pct)
                # Uses the transaction price as its own fee base. Floor for a SELL:
                fee_floor_sell = true_mid / (1.0 - self.fee_pct)
                # take the improvement, but never below the fee-covering floor
                improved = max(improved, fee_floor_sell)
                # only ever MORE aggressive than the base quote (never less)
                px = min(px, improved)
            # --- SWEEP AXIS 2: OBI-defensive widen ---
            # imb > 0.5 = bid-heavy (buyers stacked) -> selling here is adverse
            # (price likely to rise). Widen our SELL (raise px) so we stop resting
            # in front of that buying pressure.
            if self.obi_defensive and (imb - 0.5) > self.obi_defensive_thresh:
                # push the ask UP by the defensive ticks (less likely to fill)
                px = px + self.obi_defensive_ticks * self.tick
            # --- STAGE 3: OFI-defensive widen (RETREAT-ONLY, mirrors OBI) ---
            # ofi_sig > +thresh = sustained BUYING flow -> price likely to rise
            # -> selling here is adverse -> push the ask UP. Retreat only; the
            # post-only clip below still governs last. None = off/warm-up/first15.
            if ofi_sig is not None and ofi_sig > self.ofi_defensive_thresh:
                # widen the threatened ask by the OFI defensive ticks
                px = px + self.ofi_defensive_ticks * self.tick
            # Post-only clip (the engine's cross guard): stay at least one tick
            # outside the bid. Applied LAST so it governs the base quote, the exit
            # skew, and the OBI widen identically -- nothing can cross the bid.
            px = max(px, bb + self.tick)
            # SIZE THROTTLE: cut SELL clip while throttled (>=1 share).
            size_sell = max(1.0, round(size * self.throttle_frac)) \
                if sell_throttled else size
            out["SELL"] = (round(px, 2), size_sell)

        # QUOTE PEGGING (burst-flow names): hold the previous desired quote until
        # the ideal drifts >= tol_ticks. Returns the FULL desired state (our
        # protocol: an omitted side means cancel) and tracks only our own last
        # DESIRE -- never a claim about what rests on the exchange.
        if self.tol_ticks > 0:
            tol = self.tol_ticks * self.tick
            pegged = {}
            for s_, w_ in out.items():
                prev = self.last_desired.get(s_)
                # Close enough to the previous desire -> hold it (keep queue position).
                if prev is not None and abs(w_[0] - prev[0]) < tol and w_[1] == prev[1]:
                    pegged[s_] = prev
                else:
                    pegged[s_] = w_
            out = pegged
            self.last_desired = dict(out)
        if out:
            self.stats["quotes_made"] += 1
        return out
