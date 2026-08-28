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
                 *, session_scale, use_microprice=True, soft_inv=None,
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
                 obi_defensive_ticks=1.0):
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

    # ---- quoting -----------------------------------------------------------
    def quotes(self, bb, bq, ba, aq, pos):
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
        # Ch 3.3 microprice: heavier bid depth pulls fair value UP toward the ask.
        # This is DIRECTIONAL -- the confirmed cause of the short drift (micro sells
        # into ask-heavy books, buys into bid-heavy ones). use_microprice=False
        # quotes around the plain mid, like naive, to remove the lean.
        if self.use_microprice:
            fair = ba * imb + bb * (1.0 - imb)
        else:
            fair = 0.5 * (bb + ba)
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
        # Ch 7.1 VIABILITY GATE: if the market's spread is narrower than the
        # spread we require, no profitable passive quote exists -> stand aside.
        # NOTE: evaluated on the BASE half, before any trigger widening -- the
        # widening is deliberate unfillable-ness, not an economics test.
        # UNWIND EXCEPTION: when lean_exit is active the gate is bypassed; the
        # adding side is killed by the trigger anyway, and the exit side posts at
        # the touch regardless of edge economics (getting flat > earning edge).
        if self.require_viable and (ba - bb) < 2.0 * half and not trig["lean_exit"]:
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
            # Post-only clip: stay at least one tick inside the ask.
            px = min(px, ba - self.tick)
            # Bounded exit lean: SHORT + inside a trigger window -> BUY is the exit;
            # place it at the most aggressive post-only price (the bound is
            # structural: post-only cannot cross, so ba - tick is the ceiling).
            if trig["lean_exit"] and pos < 0:
                px = ba - self.tick
            # --- SWEEP AXIS 1: inventory-exit tick skew ---
            # BUY is the EXIT side when we are SHORT (pos < 0). When loaded beyond
            # the threshold, post exit_ticks_inside ticks inside the touch to leave
            # faster (post-only clip ba - tick is the most aggressive bound, so we
            # move UP toward it by N ticks from our computed px). N=0 -> no change.
            if (self.exit_ticks_inside > 0 and pos < 0
                    and -pos_lots >= self.exit_inv_threshold):
                # target = N ticks inside the touch, but never cross post-only
                target = ba - self.tick - (self.exit_ticks_inside - 1) * self.tick
                # only ever make the exit MORE aggressive (raise the bid), never less
                px = max(px, min(ba - self.tick, target))
            # --- SWEEP AXIS 2: OBI-defensive widen ---
            # imb = bq/(bq+aq); imb < 0.5 = ask-heavy (sellers stacked) -> buying
            # here is adverse (price likely to fall). Widen our BUY (lower px) so we
            # stop resting in front of that selling pressure. Engages only when the
            # imbalance against us exceeds the threshold.
            if self.obi_defensive and (0.5 - imb) > self.obi_defensive_thresh:
                # push the bid DOWN by the defensive ticks (less likely to fill)
                px = px - self.obi_defensive_ticks * self.tick
                # keep on the grid + post-only
                px = min(math.floor(px / self.tick) * self.tick, ba - self.tick)
            out["BUY"] = (round(px, 2), size)
        # Offer unless: trigger-killed, short at the hard cap, or short past the
        # soft band (past -soft_inv we stop selling so only the bid remains).
        if (not trig["kill_sell"]) and pos > -self.max_inv \
                and (self.soft_inv is None or pos > -self.soft_inv):
            # Ceil onto the tick grid (again, never more aggressive).
            px = math.ceil((reservation + half_sell) / self.tick) * self.tick
            # Post-only clip: stay at least one tick outside the bid.
            px = max(px, bb + self.tick)
            # Bounded exit lean: LONG + inside a trigger window -> SELL is the exit;
            # most aggressive post-only placement is bb + tick.
            if trig["lean_exit"] and pos > 0:
                px = bb + self.tick
            # --- SWEEP AXIS 1: inventory-exit tick skew ---
            # SELL is the EXIT side when we are LONG (pos > 0). When loaded beyond
            # the threshold, post exit_ticks_inside ticks inside the touch (move
            # DOWN toward bb + tick, the post-only floor) to leave faster. N=0 ->
            # no change.
            if (self.exit_ticks_inside > 0 and pos > 0
                    and pos_lots >= self.exit_inv_threshold):
                # target = N ticks inside the touch from the bid side
                target = bb + self.tick + (self.exit_ticks_inside - 1) * self.tick
                # only ever make the exit MORE aggressive (lower the ask), never less
                px = min(px, max(bb + self.tick, target))
            # --- SWEEP AXIS 2: OBI-defensive widen ---
            # imb > 0.5 = bid-heavy (buyers stacked) -> selling here is adverse
            # (price likely to rise). Widen our SELL (raise px) so we stop resting
            # in front of that buying pressure.
            if self.obi_defensive and (imb - 0.5) > self.obi_defensive_thresh:
                # push the ask UP by the defensive ticks (less likely to fill)
                px = px + self.obi_defensive_ticks * self.tick
                # keep on the grid + post-only
                px = max(math.ceil(px / self.tick) * self.tick, bb + self.tick)
            out["SELL"] = (round(px, 2), size)

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
