# Reject nonfinite calibration.
import math

# Freeze the user's requested common settings in all twelve combinations.
COMMON = dict(obi_throttle=True, obi_throttle_thresh=0.15, throttle_frac=0.5,
              throttle_hold_ms=300.0, ofi_throttle=False, qdr_throttle=False,
              flow_throttle=False, enable_pov_cap=True, pov_cap_mult=1.0,
              enable_run_reprice=False, enable_aggr_lean=False, enable_age_cross=False,
              enable_inv_taper=False, size_boost_mult=1.0, queue_skew_bps=0.0,
              exit_ticks_inside=1, exit_inv_threshold=1.0, obi_defensive=True,
              obi_defensive_thresh=0.15, obi_defensive_ticks=1.0,
              use_microprice=False, tol_ticks=0.0, ofi_defensive=False,
              min_edge_pct=0.0005, improve_ticks=0.0, gamma=0.15, kappa=1.5,
              tick=0.01, require_viable=True, unwind_pov=0.10,
              enable_eod_trigger=True, enable_lock_trigger=True)
# Preserve the explicit zero-skew exceptions, including revived TPL.
CHEAP = frozenset(('KEL', 'PIBTL', 'TPL'))
# Check actual constructed strategy attributes before it receives any events.
def validate_strategy(strategy, job):
    # Check all user-specified shared settings.
    for key, expected in COMMON.items():
        # Fail if a constructor ignores or changes a requested switch.
        if getattr(strategy, key) != expected:
            # Identify the actual runtime mismatch.
            raise ValueError('Effective strategy mismatch: ' + key)
    # Verify stock-specific queue knobs as well as shared controls.
    for key in ('queue_skew_ticks', 'queue_skew_thresh'):
        # Compare actual attributes to the frozen job.
        if getattr(strategy, key) != job['params'][key]:
            # Stop before an incorrectly assigned replay.
            raise ValueError('Effective assignment mismatch: ' + key)
    # Verify the actual clip and static inventory ratios.
    if strategy.size0 != job['clip'] or strategy.max_inv != 10 * job['clip'] or strategy.soft_inv != 3 * job['clip']:
        # Do not allow hidden sizing overrides.
        raise ValueError('Effective size or inventory ratio mismatch')
    # Require both inputs needed by the enabled acquisition cap.
    if strategy.unwind_profile is None or strategy.session_segments is None:
        # A cap silently bypassed for missing calibration is unacceptable.
        raise ValueError('Missing capacity calibration')
    # Reject corrupt volume inputs rather than silently bypassing the cap.
    if len(strategy.unwind_profile) != 4 or any(not math.isfinite(v) or v < 0 for v in strategy.unwind_profile):
        # Zero expected volume remains a legitimate no-capacity condition.
        raise ValueError('Invalid four-bucket volume profile')
