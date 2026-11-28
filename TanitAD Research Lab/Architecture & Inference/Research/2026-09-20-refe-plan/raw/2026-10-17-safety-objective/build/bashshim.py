#!/usr/bin/env python3
"""bashshim.py <test.py> [args] : run one test as __main__ with a BARE `bash` in a subprocess argument list resolved to Git's POSIX bash.
Why: on this dev box CreateProcess finds C:\Windows\System32\bash.exe (the WSL launcher) BEFORE anything on PATH, and that launcher drops the argument after `-c '<script>'`
(`bash -c 'ARGS="$1"; printf "%s\n" $ARGS' _ "--a 1 --b 2"` prints an empty line), so refe/test_launch_gate.py:339 (the only bare "bash" in the tests of this sweep) fails in 11 s on the
UNPATCHED tree too, before any of its 700 checks run.  Nothing else changes: same interpreter flags, same cwd, same sys.path, same __main__.
Used by build/run_tests_shim.sh; the table says which rows ran under it."""
import os
import runpy
import subprocess
import sys

GIT_BASH = r"C:\Program Files\Git\usr\bin\bash.exe"
_orig = subprocess.Popen.__init__


def _init(self, args, *a, **kw):
    if isinstance(args, (list, tuple)) and args and args[0] == "bash":
        args = [GIT_BASH, *args[1:]]
    _orig(self, args, *a, **kw)


subprocess.Popen.__init__ = _init
test = os.path.abspath(sys.argv[1])
sys.argv = [test] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(test))
runpy.run_path(test, run_name="__main__")
