"""run_isolated.py -- run a script / module under ``python -I`` with a given stack on sys.path FIRST.

    python -I -X utf8 run_isolated.py --stack <dir containing tanitad/> script.py [args]
    python -I -X utf8 run_isolated.py --stack <dir containing tanitad/> -m pytest [args]

Why: ``-I`` (required for scripts that read downloaded data) ignores PYTHONPATH, and this venv's editable ``tanitad`` install points at the
G: mount, which cannot be imported from reliably (Errno 22).  A sliced stack (here: tanitad/{__init__,refs/{refcv7_heads,refcv7_oracle},
train/config_hygiene}.py at 701922c) put first on sys.path wins over the editable finder.  Prints the peak working set at exit.
"""
import atexit
import os
import runpy
import sys


def _peak():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        print("ZZPEAKWS %.2f GB (peak working set), ZZPEAKCOMMIT %.2f GB" % (mi.peak_wset / 2 ** 30, mi.peak_pagefile / 2 ** 30), flush=True)
    except Exception as e:  # noqa: BLE001
        print("ZZPEAKWS unavailable", type(e).__name__)


if len(sys.argv) < 4 or sys.argv[1] != "--stack":
    raise SystemExit(__doc__)
atexit.register(_peak)
sys.path.insert(0, sys.argv[2])
rest = sys.argv[3:]
if rest[0] == "-m":
    sys.argv = [rest[1]] + rest[2:]
    runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
else:
    sys.path.insert(1, os.path.dirname(os.path.abspath(rest[0])))
    sys.argv = rest
    runpy.run_path(rest[0], run_name="__main__")
