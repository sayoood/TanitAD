"""pytest path bootstrap for TanitEval.

Makes ``python -m pytest`` work from ANY machine (dev box or pod) with no preset
PYTHONPATH. Historically each test hand-inserted the pod paths
``/root/taniteval`` + ``/root/TanitAD/stack`` and, later, machine-relative
equivalents — but *inconsistently*, so ``test_bench_diagnostic.py`` broke
collection off-pod: ``bench.py`` imports ``driving_diagnostic`` from
``stack/scripts`` and that dir was never on the path unless the test remembered
to add it.

Centralising the setup here means (a) collection no longer depends on which
test file you happen to import first, and (b) the paths are derived from THIS
file's location, so the same checkout runs green on the dev box and on the pod.
Non-existent candidates are skipped, so adding the pod's real paths remains the
job of each module's own ``sys.path.insert`` (harmless no-ops here off-pod).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))   # <repo>/taniteval (package parent)
_REPO = os.path.dirname(_HERE)                        # <repo>

for _p in (
    os.path.join(_REPO, "stack"),                    # the `tanitad` package
    os.path.join(_REPO, "stack", "scripts"),         # `driving_diagnostic`
    _HERE,                                            # the `taniteval` package parent
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
