"""Make the package standalone-testable (`pytest <package>/tests`): put both the
package dir (for `import ci_i2_tripwire`) and the repo stack/ dir (for `import
tanitad`) on sys.path, wherever the package currently lives."""

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent          # the intake package dir
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def _find_stack(start: Path):
    d = start
    while d != d.parent:
        cand = d / "stack"
        if (cand / "tanitad").is_dir():
            return cand
        if (d / "tanitad").is_dir():
            return d
        d = d.parent
    return None


_STACK = _find_stack(_PKG)
if _STACK and str(_STACK) not in sys.path:
    sys.path.insert(0, str(_STACK))
