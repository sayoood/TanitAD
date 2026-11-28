"""Thor (linux-aarch64) compatibility layer for metric caches BUILT ON WINDOWS.  Import-time only.

Two Windows artefacts are baked into every dev-box metric-cache pickle (MEASURED 2026-10-04 on a
navhard cache file: ``type(mc.file_path) == WindowsPath`` and
``mc.map_parameters.map_root == 'C:/Users/Admin/navsim-crun/data/maps'``):

1. ``MetricCache.file_path`` is a pickled ``pathlib.WindowsPath``. On POSIX ``WindowsPath(...)``
   raises ``NotImplementedError`` at unpickle time, so NO cache file can be read. Fix: while
   unpickling, ``pathlib.WindowsPath`` resolves to ``_UnpickledWindowsPath`` (a ``PureWindowsPath``):
   same parts, same ``str()``, never touches the filesystem. ``file_path`` is metadata -- the scorer
   never opens it (the loader opens the path from the metadata CSV, remapped by the wrapper).

2. ``MapParameters.map_root`` is a Windows directory. The two-stage reactive traffic policy
   (``NavsimIDMTrafficAgents.simulate_traffic_agents``) calls ``get_maps_api(map_root, ...)``.
   Fix: ``get_maps_api`` / ``get_maps_db`` receive ``NUPLAN_MAPS_ROOT`` in place of any
   ``X:/...`` or ``X:\\...`` root. The map FILES are byte-identical copies (sha256 recorded in
   ``raw/THOR_BACKEND_VALIDATED.json``), so the map API is the same object built from the same bytes.

Every substitution actually made is counted in ``REMAPS`` so the manifest shows it happened (or not).
``install()`` must run BEFORE any navsim / nuplan module is imported, because the devkit binds
``get_maps_api`` with ``from ... import`` at import time.
"""
from __future__ import annotations

import os
import pathlib
import re

REMAPS: dict = {"windows_map_root": {}, "windows_path_unpickled": 0}
_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")
_INSTALLED = False


class _UnpickledWindowsPath(pathlib.PureWindowsPath):
    """What a pickled ``WindowsPath`` becomes on POSIX: a pure path with identical parts/str."""

    def __new__(cls, *args):
        REMAPS["windows_path_unpickled"] += 1
        return super().__new__(cls, *args)


def _local_map_root(map_root):
    if isinstance(map_root, str) and _WIN_ABS.match(map_root):
        dst = os.environ["NUPLAN_MAPS_ROOT"]
        REMAPS["windows_map_root"][map_root] = REMAPS["windows_map_root"].get(map_root, 0) + 1
        return dst
    return map_root


def install() -> dict:
    """Idempotent. Returns a description for the run manifest."""
    global _INSTALLED
    if os.name == "nt":
        return {"thor_compat": "not installed (running on Windows)"}
    if _INSTALLED:
        return describe()
    pathlib.WindowsPath = _UnpickledWindowsPath          # only reached by unpickling on POSIX

    from nuplan.common.maps.nuplan_map import map_factory as _mf
    _orig_api = _mf.get_maps_api
    _orig_db = _mf.get_maps_db

    def get_maps_api(map_root, map_version, map_name):
        return _orig_api(_local_map_root(map_root), map_version, map_name)

    def get_maps_db(map_root, map_version):
        return _orig_db(_local_map_root(map_root), map_version)

    get_maps_api.__wrapped__ = _orig_api
    get_maps_db.__wrapped__ = _orig_db
    _mf.get_maps_api = get_maps_api
    _mf.get_maps_db = get_maps_db
    _INSTALLED = True
    return describe()


def describe() -> dict:
    return {"thor_compat": "installed",
            "patches": ["pathlib.WindowsPath -> PureWindowsPath subclass (unpickle only)",
                        "nuplan map_factory.get_maps_api/get_maps_db: Windows map_root -> $NUPLAN_MAPS_ROOT"],
            "NUPLAN_MAPS_ROOT": os.environ.get("NUPLAN_MAPS_ROOT"),
            "remaps_so_far": {"windows_path_unpickled": REMAPS["windows_path_unpickled"],
                              "windows_map_root": dict(REMAPS["windows_map_root"])}}


def token_of(p) -> str:
    """Separator-agnostic parent-directory name (the cache token) of a cache file path."""
    return re.split(r"[\\/]", str(p).strip())[-2]


def remap_cache_path(p: str, cache_root: pathlib.Path) -> str:
    """Map a metadata-CSV row (absolute path written on the dev box, either separator) onto the
    local copy of the SAME cache directory: everything after the cache directory's own name is
    kept, so ``<root>/<log>/unknown/<token>/metric_cache.pkl`` keeps its relative layout."""
    parts = re.split(r"[\\/]", p.strip())
    name = cache_root.name
    idx = max(i for i, x in enumerate(parts) if x == name)
    return str(cache_root.joinpath(*parts[idx + 1:]))
