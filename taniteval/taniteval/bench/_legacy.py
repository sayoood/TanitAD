"""The PRE-EXISTING ``taniteval/taniteval/bench.py`` (the diagnostic panel), loaded under an
importable name.

⛔ WHY THIS FILE EXISTS. Until 2026-09-19 ``taniteval.bench`` WAS that module. The one-command
suite (W1, ``BUILD_PLAN.md`` §1: ``python -m taniteval.bench <benchmark> …``) is a PACKAGE of the
same name, and CPython's path finder prefers a package directory over a same-named ``.py`` — so
the package SHADOWS the module. MEASURED importers that would break: ``runner.py:39``,
``refc_rerank.py:78``, ``generalization.py:901``, six test files, five stack scripts, and
``rerun_all.sh:7`` (``python3 -m taniteval.bench --model …``).

This module executes the legacy file's source, UNCHANGED, in its own namespace, so every legacy
name keeps its behaviour and its functions carry an importable ``__module__``
(``taniteval.bench._legacy`` — a child process can re-import it; a private ``sys.modules`` alias
could not). ``bench/__init__.py`` delegates every attribute it does not define to this module
(PEP 562), and ``bench/__main__.py`` routes the legacy CLI flags here. The legacy file is NOT
edited, moved or copied: it remains the single source (owner: its existing maintainers).
"""
import os as _os

LEGACY_SOURCE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                              "bench.py")

with open(LEGACY_SOURCE, encoding="utf-8") as _fh:
    _SRC = _fh.read()

# compile() with the REAL path, so tracebacks and coverage point at bench.py's own lines.
exec(compile(_SRC, LEGACY_SOURCE, "exec"), globals())  # noqa: S102 — the legacy module, verbatim
del _fh, _SRC
