"""IMPORT-ONLY stub of ``rasterio`` for the Thor (linux-aarch64, CPython 3.9) NavSim CPU backend.

WHY IT EXISTS (MEASURED 2026-10-04): PyPI carries NO ``rasterio`` wheel for cp39 / linux-aarch64
(uv: "we only found wheels for rasterio (v1.2.10.dev0) with ABI tag cp38"); building it needs
system GDAL headers, which we may not install on Thor.

WHY A STUB IS ADMISSIBLE: ``nuplan/database/maps_db/gpkg_mapsdb.py`` imports rasterio at module
level, but only the RASTER-layer helpers call it (``_get_map_dataset`` / ``get_layer_dataset`` /
``get_raster_layer_names``), reached only via ``NuPlanMap.get_raster_map_layer``. The NavSim
scoring path (v2 two-stage and v1.1) reads VECTOR layers only (``load_vector_layer`` -> pyogrio).

FAIL LOUD: any call into this stub RAISES, so a code path that did need a raster would turn a
token into an invalid row and the count guards (log successful == expected, CSV valid rows,
seam call count) refuse the run. Nothing can silently fall back. ``CALLS`` is kept for audits.
"""
from __future__ import annotations

__version__ = "0.0.0+thor-import-stub"
STUB = True
CALLS: list = []


class _StubError(RuntimeError):
    pass


class errors:  # noqa: N801  (mirrors the rasterio.errors namespace used at import time)
    class NotGeoreferencedWarning(UserWarning):
        pass

    class RasterioIOError(OSError):
        pass


class DatasetReader:  # used only as a return annotation in gpkg_mapsdb.py
    def __init__(self, *a, **k):
        CALLS.append(("DatasetReader", a))
        raise _StubError("rasterio stub (Thor NavSim backend): raster datasets are not available")


def open(*a, **k):  # noqa: A001
    CALLS.append(("open", a))
    raise _StubError("rasterio stub (Thor NavSim backend): rasterio.open() called -- a RASTER "
                     "map layer was requested; this backend supports the vector-only scoring path")


def __getattr__(name):
    CALLS.append(("getattr", name))
    raise AttributeError(f"rasterio stub (Thor NavSim backend): attribute {name!r} not provided")
