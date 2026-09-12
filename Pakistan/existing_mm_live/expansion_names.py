# ============================================================================
# expansion_names.py -- shared constants + guards for the universe expansion
# ============================================================================
# One place for the things every expansion wrapper needs, so the name list
# cannot drift between four scripts and the two landmines below cannot be
# re-stepped-on in each of them.
#
# LANDMINE 1 -- the empty store.
#   run_legacy_mm's default PARSED_ROOT points at the (empty) Google Drive
#   path. mm_harness overrides it on import; the calibration scripts do NOT
#   import mm_harness, so a wrapper that forgets to set it reads an empty
#   store and produces a clean-looking, completely empty result.
#
# LANDMINE 2 -- mm_harness.newest().
#   load_scales/profiles/windows each take the NEWEST file matching their
#   glob and build their dict from THAT FILE ALONE. Writing a new calibration
#   file containing only the 76 new names would make the 38 production names
#   vanish from the loaders. Every calibration output must therefore be
#   written as EXISTING UNION NEW. merge_with_existing() below does that.
# ============================================================================

# filesystem paths
from pathlib import Path
# run timestamps for output filenames
from datetime import datetime
# frames for the merge
import pandas as pd

# the canonical roots -- imported, never hardcoded. Fails loud if absent.
from config_pk import PARSED_ROOT, RESULTS_ROOT

# ---------------------------------------------------------------------------
# THE EXPANSION UNIVERSE
# ---------------------------------------------------------------------------
# the 76 never-run candidates: persistence_REG_2p00.csv at days_traded >= 100
# and notional_m_median >= 25M PKR. Notional is the ONLY selector: on the 38
# production names it was the strongest measured predictor of realised
# per-name net_pkr (Spearman +0.354, p=0.029), while the screen's own
# net5_trec_median was ANTI-predictive (-0.244) and spread_bps_median -0.333.
NEW_NAMES = ['AGHA', 'AGP', 'AHCL', 'AICL', 'AIRLINK', 'APL', 'ASL', 'AVN',
             'BAHL', 'BBFL', 'BECO', 'BFBIO', 'BML', 'BNL', 'CEPB', 'CHCC',
             'CNERGY', 'CPHL', 'CSAP', 'DCL', 'DFML', 'EFERT', 'EPCL',
             'FABL', 'FATIMA', 'FCCL', 'FCEPL', 'FCL', 'FECTC', 'FFL',
             'GAL', 'GCIL', 'GCWL', 'GGL', 'GHNI', 'GLAXO', 'HALEON',
             'HCAR', 'HMB', 'HUMNL', 'ILP', 'IMAGE', 'ISL', 'JVDC',
             'KAPCO', 'KOHC', 'KOIL', 'KOSM', 'LCI', 'LOADS', 'LOTCHEM',
             'MCB', 'MTL', 'MUGHAL', 'NATF', 'NETSOL', 'POL', 'POWER',
             'PREMA', 'PRL', 'PSX', 'QUICE', 'SGF', 'SGPL', 'SLGL',
             'SNGP', 'SSGC', 'TBL', 'TELE', 'TGL', 'TPLP', 'TREET',
             'UNITY', 'WAVES', 'WTL', 'ZAL']

# the 38 names already in production. Their calibration already exists and
# must be PRESERVED through every merge -- never recomputed, never dropped.
INCUMBENT = ['AKBL', 'ATRL', 'BAFL', 'BOP', 'DGKC', 'ENGROH', 'FFC', 'FNEL',
             'HASCOL', 'HBL', 'HUBC', 'KEL', 'LUCK', 'MARI', 'MEBL', 'MLCF',
             'NBP', 'NCPL', 'NML', 'NPL', 'NRL', 'OGDC', 'PACE', 'PAEL',
             'PIAHCLA', 'PIBTL', 'PIOC', 'PPL', 'PSO', 'PTC', 'SAZEW',
             'SEARL', 'SYS', 'THCCL', 'TOMCL', 'TPL', 'TRG', 'UBL']

# the full post-expansion book
ALL_NAMES = sorted(set(NEW_NAMES) | set(INCUMBENT))

# sanity: the two lists must not overlap, and must sum to the expected total
assert not (set(NEW_NAMES) & set(INCUMBENT)), "NEW_NAMES overlaps INCUMBENT"
assert len(NEW_NAMES) == 76, f"expected 76 new names, got {len(NEW_NAMES)}"
assert len(INCUMBENT) == 38, f"expected 38 incumbents, got {len(INCUMBENT)}"
assert len(ALL_NAMES) == 114, f"expected 114 total, got {len(ALL_NAMES)}"


def run_stamp():
    # the house filename convention: YYYYMMDD_HHMM
    return datetime.now().strftime("%Y%m%d_%H%M")


def safe_out(stem, ext, root=None):
    # default destination is the canonical results root
    root = Path(root) if root is not None else RESULTS_ROOT
    # this run's stamp
    ts = run_stamp()
    # the candidate path
    p = root / f"{stem}_{ts}.{ext}"
    # disambiguating counter, used only on a same-minute collision
    n = 1
    # walk until the name is free -- this function NEVER returns an existing path
    while p.exists():
        # append the counter
        p = root / f"{stem}_{ts}_{n}.{ext}"
        # advance it
        n += 1
    # a path guaranteed not to exist
    return p


def bind_parsed_root(R):
    # LANDMINE 1: push the canonical raw-store root onto the driver module.
    # Call this immediately after importing run_legacy_mm (and after importing
    # any script that sets its own stale PARSED_ROOT at import time).
    R.PARSED_ROOT = PARSED_ROOT
    # prove the store is actually readable before the caller does any work
    dates = R.discover_dates()
    # an empty list means a wrong path, not a market that never traded
    if not dates:
        # name the path that was read so the fix is obvious
        raise SystemExit(f"discover_dates() returned 0 dates from "
                         f"{R.PARSED_ROOT!r}. Wrong or empty parsed store -- "
                         f"check config_pk.PARSED_ROOT.")
    # hand the caller the verified date list
    return dates


def newest_existing(pattern, root=None):
    # mirror mm_harness.newest(): the lexicographically last match wins
    root = Path(root) if root is not None else RESULTS_ROOT
    # every file matching the glob
    cands = sorted(root.glob(pattern))
    # None when nothing exists yet (a first-ever calibration)
    return cands[-1] if cands else None


def merge_with_existing(new_df, pattern, key="symbol", root=None):
    """LANDMINE 2: return EXISTING UNION NEW, so the loaders keep every name.

    mm_harness builds its calibration dicts from the newest matching file
    ALONE. A file containing only the new names would therefore delete the
    production names from load_scales()/load_profiles()/load_windows().

    Rows for a key present in BOTH are taken from new_df (an explicit
    recalibration wins); every other existing row is carried through
    untouched. Returns (merged_df, source_path_or_None, n_carried, n_new).
    """
    # locate the current newest file for this calibration family
    src = newest_existing(pattern, root)
    # nothing to merge with -> the new frame is the whole table
    if src is None:
        return new_df.copy(), None, 0, len(new_df)
    # read the existing calibration exactly as the loaders would
    old = pd.read_csv(src)
    # keys being (re)written by this run
    incoming = set(new_df[key].astype(str))
    # every existing row whose key is NOT being rewritten
    carried = old[~old[key].astype(str).isin(incoming)]
    # union, new rows last, index reset for a clean CSV
    merged = pd.concat([carried, new_df], ignore_index=True)
    # sort by key so the file is diffable between runs
    merged = merged.sort_values(key).reset_index(drop=True)
    # hand back the table plus what happened, for the caller to print
    return merged, src, len(carried), len(new_df)
