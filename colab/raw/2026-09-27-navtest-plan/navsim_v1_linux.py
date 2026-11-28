#!/usr/bin/env python3
"""Per-process wrapper shim for Linux: W3's `navsim_v1_win.py` with its path constants re-pointed, then its OWN
main(). Launched by run_v1.run_wrapped (via linux_score_v1.py, which sets `WRAP` to this file) with the NAVSIM venv's
python and exactly the argv run_v1 builds for the Windows wrapper.

Reads from the environment (set by linux_score_v1.py): NAVPLAN_W3_CODE, NAVPLAN_E1, NAVPLAN_V11, NAVPLAN_NUPLAN,
NAVPLAN_WINPATH_SHIM. Everything else -- the RAM guard, the observation hooks, the C8 import-provenance assertions,
the one loader patch -- is navsim_v1_win.py's unchanged code.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

W3_CODE = os.environ["NAVPLAN_W3_CODE"]
if os.environ.get("NAVPLAN_WINPATH_SHIM") == "1":
    # a metric cache written on Windows pickles MetricCache.file_path as pathlib.WindowsPath, which cannot be
    # instantiated on Linux (NotImplementedError); unpickle it as the pure (platform-free) class instead
    pathlib.WindowsPath = pathlib.PureWindowsPath

spec = importlib.util.spec_from_file_location("navsim_v1_win", os.path.join(W3_CODE, "navsim_v1_win.py"))
w = importlib.util.module_from_spec(spec)
sys.modules["navsim_v1_win"] = w
spec.loader.exec_module(w)
w.V11_TREE = os.environ["NAVPLAN_V11"]
w.NUPLAN_TREE = os.environ["NAVPLAN_NUPLAN"]
w.NUPLAN_SOURCE = os.environ["NAVPLAN_NUPLAN"]
w.E1_WRAPPER = os.environ["NAVPLAN_E1"]
w.ENV_DEFAULTS = {k: os.environ[k] for k in ("NUPLAN_MAP_VERSION", "NUPLAN_MAPS_ROOT", "NAVSIM_EXP_ROOT",
                                             "NAVSIM_DEVKIT_ROOT", "OPENSCENE_DATA_ROOT") if k in os.environ}
sys.argv[0] = os.path.join(W3_CODE, "navsim_v1_win.py")
sys.exit(w.main())
