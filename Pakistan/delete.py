# show_snapshot_internals.py -- gather what's needed to optimize snapshot()
# run from existing_mm_live/:  python show_snapshot_internals.py
import subprocess
# tail of snapshot() (the AGG loop end + book assignment)
print("=== snapshot() tail (325-345) ===")
print(subprocess.run(["sed","-n","325,345p","mm_backtest.py"],capture_output=True,text=True).stdout)
# how snap_groups is constructed and passed to run()
print("=== snap_groups / build_events grouping in run_legacy_mm.py ===")
print(subprocess.run(["grep","-n","snap_groups","run_legacy_mm.py"],capture_output=True,text=True).stdout)