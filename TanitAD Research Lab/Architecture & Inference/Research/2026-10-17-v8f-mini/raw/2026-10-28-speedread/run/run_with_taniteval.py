"""Run-mechanics wrapper for PREREG_V8F_SPEEDREAD s8 (stdlib only; the pinned instrument is run unchanged): put <tree>/taniteval on sys.path
(the data loader imports taniteval.floors_kin, and `python -I` ignores PYTHONPATH), then run the instrument as __main__ with its own argv.
python -I run_with_taniteval.py <tree> <instrument.py> <instrument args...>"""
import os
import runpy
import sys

tree, script = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(tree, "taniteval"))
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
