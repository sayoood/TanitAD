"""TanitEval per-run bench reports (EvalFlyWheel W5, 2026-09-19).

    python -m taniteval.benchreport <run_dir>     # -> <run_dir>/report/index.html + report/fig/*

A run directory is the BUILD_PLAN §1 contract (W1 schema ``taniteval.bench.summary/1``). The report
prints the SAME numbers as ``summary.json`` — verified by parsing its own HTML (``verify.py``) — plus a
failure gallery drawn with the NavSim devkit's own ``navsim.visualization`` (NavSim venv, py3.9).

Named ``benchreport`` (not ``report``) so it never shadows the legacy ``taniteval/taniteval/report.py``
(orchestrator ruling 2026-09-19). W4 imports ``taniteval.benchreport.leaderboard_charts``.

Submodules are imported lazily so ``import taniteval.benchreport`` stays cheap (no numpy/torch).
"""
from __future__ import annotations

import importlib

REPORT_VERSION = "w5-1.0"

__all__ = ["REPORT_VERSION", "render_report", "verify_report"]


def __getattr__(name: str):
    if name == "render_report":
        return importlib.import_module(f"{__name__}.render").render_report
    if name == "verify_report":
        return importlib.import_module(f"{__name__}.verify").verify_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
