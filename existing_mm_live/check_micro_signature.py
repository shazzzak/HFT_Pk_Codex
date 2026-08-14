# check_micro_signature.py -- confirm the microprice/band edits landed in
# micro_mm.MictrostructureMM.__init__ BEFORE burning compute on the full run.
# Run from the project root (HFT/):  python check_micro_signature.py

# stdlib: path handling
import sys
# stdlib: read a callable's parameter list
import inspect
# stdlib: filesystem paths
from pathlib import Path

# make the strategy package importable regardless of where we launch from:
# add existing_mm_live/ (which holds micro_mm.py) to the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent / "existing_mm_live"))

# import the strategy class under test
from micro_mm import MicrostructureMM

# pull the __init__ parameter names
params = inspect.signature(MicrostructureMM.__init__).parameters

# the two new keyword-only args the edits were supposed to add
has_mp = "use_microprice" in params
has_band = "soft_inv" in params

# report each explicitly
print(f"use_microprice present: {has_mp}")
print(f"soft_inv present:       {has_band}")

# single-line verdict
if has_mp and has_band:
    # both landed -> safe to run confirm_micro_vs_naive.py
    print("OK -- both args present; safe to run the full confirm.")
else:
    # at least one missing -> the confirm 2x2 configs would TypeError
    print("FAIL -- edit(s) missing; fix micro_mm.py before the run.")
    # non-zero exit so a wrapper/CI would catch it too
    sys.exit(1)
