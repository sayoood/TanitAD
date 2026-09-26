"""taniteval.bench — the ONE-COMMAND benchmark suite (EvalFlyWheel W1, 2026-09-19).

    python -m taniteval.bench <benchmark> --ckpt <path|none> --split <split> [--arms A1,STOP,CV,...] [--device auto|cpu|cuda]
        benchmark in {navsim_v2, navsim_v1, nuscenes_ol, internal_t1}
    python -m taniteval.bench submit <run_dir> --target <name> [--pi-approval <decision-id>]   # REFUSED without a PI decision
    python -m taniteval.bench validate <run_dir>                                               # schema + contract check
    python -m taniteval.bench gpu-gap                                                          # the gap launcher's current verdict

Contract: ``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-eval-suite-build/BUILD_PLAN.md`` §1.
Schemas: ``taniteval/taniteval/bench/schema/{bench_run,summary}.schema.json``.

⛔ BACKWARD COMPATIBILITY — READ BEFORE EDITING THIS FILE
--------------------------------------------------------
Before this package existed, ``taniteval.bench`` was the MODULE ``taniteval/taniteval/bench.py``
(the diagnostic panel: ``run``, ``diagnostic``, ``_agg``, ``_suite``, …). This package shadows it.
Every attribute NOT defined by the suite is therefore delegated (PEP 562 ``__getattr__``) to
``taniteval.bench._legacy``, which executes that file unchanged. Suite submodule names raise
``AttributeError`` here WITHOUT loading the legacy module, so ``from taniteval.bench import
contract`` never pays for the legacy panel's torch import. Pinned by
``taniteval/tests/test_bench_suite_legacy_compat.py`` (incl. a deliberate-regression arm).
"""
from __future__ import annotations

import os as _os

SUITE_VERSION = "1.0.0"

_HERE = _os.path.dirname(_os.path.abspath(__file__))


def _suite_submodules() -> frozenset:
    names = set()
    for entry in _os.listdir(_HERE):
        if entry.endswith(".py") and entry != "__init__.py":
            names.add(entry[:-3])
        elif _os.path.isdir(_os.path.join(_HERE, entry)) and not entry.startswith("__"):
            names.add(entry)
    return frozenset(names)


_SUITE_SUBMODULES = _suite_submodules()


def _load_legacy():
    """⛔ NOT named ``_legacy``: a package-level function of the same name as the submodule makes
    ``from . import _legacy`` return the FUNCTION (the attribute exists, so no import happens) —
    MEASURED 2026-09-19, it turned every legacy lookup into an AttributeError."""
    import importlib
    return importlib.import_module(__name__ + "._legacy")   # executes ../bench.py on first use only


def __getattr__(name: str):
    # dunders and our own submodules: let the import machinery handle them (never load legacy)
    if (name.startswith("__") and name.endswith("__")) or name in _SUITE_SUBMODULES:
        raise AttributeError(name)
    mod = _load_legacy()
    try:
        return getattr(mod, name)
    except AttributeError:
        raise AttributeError(
            f"module 'taniteval.bench' has no attribute {name!r} — neither the suite nor the legacy "
            f"diagnostic panel (taniteval/taniteval/bench.py) defines it") from None


def __dir__():
    import sys
    names = set(globals()) | set(_SUITE_SUBMODULES)
    leg = sys.modules.get(__name__ + "._legacy")
    if leg is not None:
        names |= set(dir(leg))
    return sorted(names)
