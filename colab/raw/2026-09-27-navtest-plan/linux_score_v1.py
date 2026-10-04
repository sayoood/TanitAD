#!/usr/bin/env python3
"""Linux launcher for W3's NAVSIM v1.1 harness -- `eval/score_navtest_refe.py`'s main(), with ONLY the Windows path
constants re-pointed. PROTOTYPE for the Colab plan (colab/NAVTEST_ON_COLAB_PLAN.md); first exercised by the
2026-09-27 CPU smoke.

Why it exists. `score_navtest_refe.py` loads W3's `run_v1.py` by path and calls its `cmd_score` UNCHANGED, re-pointing
only `RAW` and `PKG`. On Linux the same is true, but run_v1.py and its per-process wrapper `navsim_v1_win.py` also
carry hard-coded dev-box paths (NV_PY = the C: NAVSIM venv, V11_TREE / EXP / DATA on D:, NUPLAN_TREE on C:, E1's
wrapper on D:). This launcher sets those module globals, and points run_v1's `WRAP` at `navsim_v1_linux.py`, which
does the same for the wrapper process and then calls the wrapper's own `main()`. The devkit, W3's agent, the guards
C1-C3/C8 and the scoring code are the unchanged files.

    python linux_score_v1.py --label refe_x --seam seam.npz --tokens toks.json --out <dir> \
        --w3-code <dir with run_v1.py, navsim_v1_win.py, w3_agents_v1.py> --e1-wrapper <navsim_win.py> \
        --nv-py <navsim venv python> --v11-tree <navsim @3e8291b> --nuplan-tree <nuplan-devkit @ce3c323> \
        --exp <NAVSIM_EXP_ROOT, holds metric_cache_navtest/> --openscene <OPENSCENE_DATA_ROOT> [--maps <root>] \
        [--windowspath-shim]

`--windowspath-shim` is needed ONLY when the metric-cache pickles were written on Windows: navsim v1.1's
`MetricCache.file_path` is a `pathlib.Path`, so a dev-box pickle carries `pathlib.WindowsPath`, which Python refuses
to instantiate on Linux. The shim maps that class to `PureWindowsPath` at unpickle time; nothing the scorer computes
reads `file_path` (W3's observation hook derives a token from it for its JSONL rows only).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seam", required=True)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w3-code", required=True)
    ap.add_argument("--e1-wrapper", required=True)
    ap.add_argument("--nv-py", required=True)
    ap.add_argument("--v11-tree", required=True)
    ap.add_argument("--nuplan-tree", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--openscene", required=True)
    ap.add_argument("--maps", default="")
    ap.add_argument("--cache-name", default="navtest")
    ap.add_argument("--windowspath-shim", action="store_true")
    a = ap.parse_args(argv)
    if not a.label.startswith("refe"):
        sys.exit("labels must start with 'refe' (score_navtest_refe.py's rule)")
    spec = importlib.util.spec_from_file_location("w3_run_v1", os.path.join(a.w3_code, "run_v1.py"))
    w3 = importlib.util.module_from_spec(spec)
    sys.modules["w3_run_v1"] = w3
    spec.loader.exec_module(w3)
    # ---- the re-pointing: path constants only ----------------------------------------------------------------
    w3.NV_PY = a.nv_py
    w3.V11_TREE = a.v11_tree
    w3.EXP = a.exp
    w3.DATA = a.openscene
    w3.NAVTEST_YAML = (f"{a.v11_tree}/navsim/planning/script/config/common/train_test_split/"
                       f"scene_filter/navtest.yaml")
    w3.WRAP = os.path.join(HERE, "navsim_v1_linux.py")
    w3.ENV.update({"NUPLAN_MAPS_ROOT": a.maps or os.path.join(a.exp, "_no_maps_needed_for_scoring"),
                   "NAVSIM_EXP_ROOT": a.exp, "NAVSIM_DEVKIT_ROOT": a.v11_tree, "OPENSCENE_DATA_ROOT": a.openscene,
                   # read by navsim_v1_linux.py in the wrapper process
                   "NAVPLAN_W3_CODE": a.w3_code, "NAVPLAN_E1": a.e1_wrapper, "NAVPLAN_V11": a.v11_tree,
                   "NAVPLAN_NUPLAN": a.nuplan_tree, "NAVPLAN_WINPATH_SHIM": "1" if a.windowspath_shim else "0"})
    os.makedirs(a.out, exist_ok=True)
    w3.RAW = os.path.abspath(a.out)
    w3.PKG = os.path.abspath(a.out)
    # ---- exactly score_navtest_refe.py's namespace ----------------------------------------------------------------
    ns = types.SimpleNamespace(label=a.label, arm=f"SEAM:{os.path.abspath(a.seam)}", tokens=a.tokens,
                               cache_name=a.cache_name, worker="sequential", hashseed="1", record_poses=True,
                               patch_loader=True)
    return w3.cmd_score(ns)


if __name__ == "__main__":
    sys.exit(main())
