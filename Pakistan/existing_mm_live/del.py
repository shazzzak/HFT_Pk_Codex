
import sys
import runpy

# 1. Override sys.argv with the arguments your script expects
sys.argv = [
    "latency_sweep.py",
    "--run",
    "--arms", "level",
    "--seeds", "1",
    "--workers", "1",
    "--days", "20"
]

# Optional: If you are using the terminal and want to use the built-in debugger, uncomment the next line.
# If you are using an IDE like VS Code or PyCharm, just place a breakpoint in latency_sweep.py instead.
# import pdb; pdb.set_trace()

# 2. Run the target script as if it were executed from the command line
runpy.run_path("latency_sweep.py", run_name="__main__")


