"""Run a script as __main__ and print the process' PEAK working set (MB) at exit: ``python peak_rss.py <script.py> [args...]``.

The WP-C rule is 'keep memory < 2 GB' on the dev box; this makes the claim a measurement instead of an intention."""
import runpy
import sys

import psutil

script = sys.argv[1]
sys.argv = sys.argv[1:]
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:                         # the script's own exit status is passed through
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
finally:
    print("PEAK_WSET_MB", round(psutil.Process().memory_info().peak_wset / 2 ** 20, 1), flush=True)
sys.exit(code)
