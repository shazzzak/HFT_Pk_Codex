# live_config.py -- the entry point the live quoter and the preflight already
# use. The IMPLEMENTATION moved to Production/venues/psx_config.py on
# 2026-09-16; this file now only supplies the paths, which is the one thing that
# genuinely belongs in the research tree beside config_pk.
#
# WHY IT MOVED. There were two loaders: this one, and one written for the
# production engine. Two implementations of "which names quote and how hard they
# lean" is exactly how a name ends up running a setting nobody chose -- the two
# drift, each looks correct on its own, and nothing compares them. One file, one
# behaviour, one test suite.
#
# WHAT CHANGED in the move (all three marked CHANGED 2026-09-16 in psx_config.py):
#   1. skew_ticks / skew_thresh are read from the CSV and cross-checked against
#      the label table, rather than re-derived from the label. The old LABELS
#      table was a hand-maintained duplicate of numbers the file already carries.
#   2. A pre-2026-09-15 two-bucket file is named as such in the error.
#   3. 'drop' and 'obi' in lower case now resolve. They did not before, because
#      the canonical labels were not in the alias table.
# Everything else -- the two-file split, startup-fatal/reload-tolerant, the
# single-read torn-file protection, content hashing, the orphan re-warning, the
# lock, DROP_SEMANTICS and the preflight -- is unchanged.

# where the production engine lives, relative to this file. Adjust the number of
# .parent hops if the checkout is laid out differently.
import sys
from pathlib import Path

# existing_mm_live/ -> Pakistan/ -> Pakistan/Production
_PRODUCTION = Path(__file__).resolve().parent.parent / "Production"
# put it on the path so `venues.psx_config` imports
if str(_PRODUCTION) not in sys.path:
    sys.path.insert(0, str(_PRODUCTION))

# the single implementation, re-exported so existing callers need no change
from venues.psx_config import (              # noqa: E402
    ALIASES, DROP_SEMANTICS, ENGINE_PARAMS, FILE_PARAMS, REFRESH_SECONDS,
    ConfigError, LiveConfig as _LiveConfig, preflight as _preflight)

# ---------------------------------------------------------------------------
# PATHS -- from config_pk, and on the live path a missing config_pk is FATAL.
# There is deliberately no fallback literal here. A backtest that silently reads
# the wrong store wastes an afternoon; a live book that does is a real loss.
# ---------------------------------------------------------------------------
try:
    # the project's central path module -- the only place a path is written down
    from config_pk import RESULTS_ROOT
except Exception as _e:
    # fail at import, naming the fix, rather than anywhere later
    raise ImportError(
        "live_config: could not import RESULTS_ROOT from config_pk. Run from "
        "the existing_mm_live/ directory, or put it on sys.path. "
        "Original: %r" % _e)

# the machine-generated assignment currently shipped
ASSIGNMENT_PATH = RESULTS_ROOT / "config_assignment_20260915_0043.csv"
# The sparse hand-edited intervention file, kept in existing_mm_live next to the
# code rather than among the run outputs -- it is opened under time pressure and
# must be one keystroke away. Resolved RELATIVE TO THIS FILE, so moving the
# checkout cannot make it stale. May legitimately not exist.
OVERRIDES_PATH = Path(__file__).resolve().parent / "live_overrides.csv"


class LiveConfig(_LiveConfig):
    """The same class, with this tree's paths as the defaults."""

    def __init__(self, assignment_path=ASSIGNMENT_PATH,
                 overrides_path=OVERRIDES_PATH):
        # everything else is inherited
        super().__init__(assignment_path, overrides_path)


# ---------------------------------------------------------------------------
# Run this file directly to validate the current files before a session opens.
# Exit code 0 = safe to start, 1 = do not start.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # the shared preflight, with this tree's paths
    raise SystemExit(_preflight(ASSIGNMENT_PATH, OVERRIDES_PATH))
