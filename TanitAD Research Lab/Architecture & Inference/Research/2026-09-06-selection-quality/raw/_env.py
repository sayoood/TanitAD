"""Import shim: force the OFF-DRIVE clone ahead of the G: editable install.

The venv carries an EDITABLE `tanitad` install pointing at the G: mount, which
dies with Errno 22 mid-import when the mount flaps (CLAUDE.md dev-box trap).
This tree was copied from the repo and verified FILE BY FILE: 498 of 499 .py
files md5-identical, the one miss (`stack/scripts/pod_pull_b1_epcache.py`) a
mount flap on a file nothing here imports.
"""
import sys
import os

CLONE = r"C:\Users\Admin\tanitad-selq-20260906"
STACK = os.path.join(CLONE, "stack")
TEVAL = os.path.join(CLONE, "taniteval")

# Kill any editable-install finder that would resolve `tanitad` to G:.
for _m in [m for m in sys.meta_path
           if type(m).__name__.startswith("_EditableFinder")
           or "editable" in type(m).__module__.lower()]:
    sys.meta_path.remove(_m)

for p in (TEVAL, STACK):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def check():
    import tanitad
    import taniteval
    assert tanitad.__file__.startswith(STACK), tanitad.__file__
    assert taniteval.__file__.startswith(TEVAL), taniteval.__file__
    return tanitad.__file__, taniteval.__file__
