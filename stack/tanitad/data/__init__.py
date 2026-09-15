"""Dataset adapters — imported LAZILY so a consumer pays only for what it uses.

⛔ WHY THIS IS LAZY. These submodules were imported eagerly at package import, so
``import tanitad.data`` pulled in **torch** (via :mod:`tanitad.data.toy_driving`)
for every consumer, including ones that touch none of it. `trajrecon` is an
optional extra that needs opencv and ffmpeg but NOT torch, and it became
unimportable in any environment without the training stack:

    >>> import tanitad.data.trajrecon.pipeline
    ModuleNotFoundError: No module named 'torch'

MEASURED 2026-09-15: this blocked a dashcam re-render outright in a container with
no torch installed. ``tests/test_trajrecon_integrity.py::
test_import_is_lazy_and_dependency_free`` has been asserting the opposite
property — *"importing the package must not drag in opencv/scipy/matplotlib"* —
and was failing, so the suite had already flagged it.

PEP 562 ``__getattr__`` keeps the public API byte-identical: ``from tanitad.data
import ToyDrivingDataset`` still works and still returns the same object, it is
merely resolved on first access instead of at import. ``__all__`` is unchanged, so
``from tanitad.data import *`` is unchanged too.

⚠️ The cost of laziness is that a broken submodule now fails at first USE rather
than at import, which can surface far from the cause. :func:`_eager_import_all` is
provided for exactly that: it forces every binding and is what the integrity test
calls, so the "does it all still import" check keeps happening in CI even though
it no longer happens for every consumer.
"""
from __future__ import annotations

import importlib
from typing import Any

# public name -> submodule that defines it
_EXPORTS = {
    "ToyDrivingDataset": "tanitad.data.toy_driving",
    "ToyEpisode": "tanitad.data.toy_driving",
    "generate_episode": "tanitad.data.toy_driving",
    "frame_change_fraction": "tanitad.data.toy_driving",
    # MetaDrive adapter helpers import cleanly (MetaDrive itself is imported
    # lazily, only inside generate_metadrive_episode), so this is safe with no
    # sim deps.
    "MetaDriveDataset": "tanitad.data.metadrive_env",
    "generate_metadrive_episode": "tanitad.data.metadrive_env",
    # comma2k19 (D-009 primary corpus). Video decode (`av`) is imported lazily
    # inside the decode path only, so this is safe with no codec/data present.
    "Comma2k19Dataset": "tanitad.data.comma2k19",
    "build_episode": "tanitad.data.comma2k19",
    "discover_segments": "tanitad.data.comma2k19",
    "split_by_route": "tanitad.data.comma2k19",
    # Consequence-dominance (A8) statistics harness — measures per-corpus change
    # fraction for change-weighting (H3/A8) and the D-010 real-vs-sim mix decision.
    "consequence_dominance_stats": "tanitad.data.stats",
    "stats_by_label": "tanitad.data.stats",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (PEP 562)."""
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value          # cache, so this runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


def _eager_import_all() -> None:
    """Force every lazy binding — the check that laziness would otherwise skip.

    Call this where "do all the adapters still import?" is the question, rather
    than relying on package import to answer it as a side effect.
    """
    for name in _EXPORTS:
        getattr(__import__(__name__, fromlist=[name]), name)
