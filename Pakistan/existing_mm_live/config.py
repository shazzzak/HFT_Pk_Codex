# config.py -- single source of truth for every filesystem path the HFT scripts
# use. Import from here instead of hardcoding strings at the top of each script:
#     from config import PARSED_ROOT, RESULTS_ROOT, FEATURE_STORE, FILLS_DIR
# When you move machines or reorganise, change paths HERE only.
#
# NOTE on the two fills folders: FILLS_DIR points at the queue-position attribution
# fills (the source of truth). FILLS_RAW_DIR is the optimistic "counterparty to
# every trade" set -- kept for reference but NOT for net-P&L analysis. Confirm which
# is which from the schema probe before trusting either (see fills_to_csv.py).

# pathlib for OS-independent path building
from pathlib import Path

# ---------------------------------------------------------------- roots ------
# raw parsed FIX data (date-partitioned parquet: trades / ob_updates / ob_snapshot)
PARSED_ROOT = Path("/Users/shazzak/Capital Stake - Parsed")
# everything the backtests/tools write
RESULTS_ROOT = Path("/Users/shazzak/Capital Stake - Results")
# project code root
PROJECT_ROOT = Path("/Users/shazzak/PycharmProjects/HFT")

# ---------------------------------------------------- derived data paths -----
# per-symbol feature store (feature_store/{SYM}/date=YYYY-MM-DD.parquet)
FEATURE_STORE = RESULTS_ROOT / "feature_store"
# queue-position attribution fills -- the SOURCE OF TRUTH for net P&L
FILLS_DIR = RESULTS_ROOT / "fill_attribution" / "fills"
# optimistic raw fills (counterparty-to-every-trade); reference only, NOT net P&L
FILLS_RAW_DIR = RESULTS_ROOT / "fills"
# per-day EOD inventory + liquidation records (written by confirm)
EOD_POSITIONS = RESULTS_ROOT / "eod_positions.csv"
# per-config P&L summary (written by confirm)
PNL_SUMMARY = RESULTS_ROOT / "pnl_summary.csv"
# the confirm run's main output table
CONFIRM_CSV = RESULTS_ROOT / "confirm_micro_vs_naive.csv"
# watchlist of shortlist symbols
WATCHLIST = RESULTS_ROOT / "mm_watchlist_final.csv"

# ------------------------------------------------------------- outputs -------
# where combined/export artifacts (e.g. the Tableau CSV) get written
EXPORT_DIR = RESULTS_ROOT / "exports"

# ------------------------------------------------------------- symbols -------
# development symbols
DEV_SYMBOLS = ["PPL", "UBL"]
# thin/locky third name (measured; exercises the EOD/lock triggers)
LOCK_SYMBOL = "PACE"
# per-symbol back-solved session_scale (skew at max inventory ~ 1x median spread)
SESSION_SCALE = {"PPL": 7.6, "UBL": 3.9, "PACE": 46.15}
