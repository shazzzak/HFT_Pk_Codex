"""Makes `core` and `venues` importable no matter where pytest is started from.

WHY THIS FILE EXISTS. `python tests/test_audit.py` and `pytest` from a
subdirectory both fail with `ModuleNotFoundError: No module named 'core'`,
because neither puts the project root on the import path. pytest imports
conftest.py before collecting anything, so one line here fixes it for every
invocation instead of requiring a PYTHONPATH the caller has to remember.
"""
# the import path
import sys
# to locate this file's own directory
from pathlib import Path

# the project root is wherever this file sits
ROOT = str(Path(__file__).resolve().parent)
# put it first, so `core` and `venues` resolve before anything installed
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
