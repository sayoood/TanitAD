#!/usr/bin/env python3
"""Linux launcher for W3's NAVSIM v1.1 harness (`run_v1.py`), for Colab/any Linux VM: `score` and `cache`.

Promoted from the plan's banked prototype (`colab/raw/2026-09-27-navtest-plan/linux_score_v1.py`, exercised by the
2026-09-27 CPU smoke). W3's `run_v1.py` is loaded by path THROUGH `run_v1_fix1.load()` (NT-FIX1: byte-pins it and adds
three report-path fixes) and its `cmd_score` / `cmd_cache` run otherwise UNCHANGED; only the
Windows path constants are re-pointed (NV_PY, V11_TREE, EXP, DATA, NAVTEST_YAML, WRAP, RAW, PKG and the ENV paths),
and `WRAP` points at `navsim_v1_linux.py`, which does the same for W3's per-process wrapper and then calls ITS main().
The devkit, W3's agents, guards C1-C3/C8/C9 and the scoring code are the unchanged files.

    python linux_run_v1.py score --label refe_x --arm SEAM:<seam.npz> --tokens <toks.json> --out <dir> <paths...>
    python linux_run_v1.py score --label refe_cv --arm CV   ...            # W3's floors: CV | STOP | HUMAN
    python linux_run_v1.py cache --label refe_cache --cache-name navtest_vm --tokens <toks.json> --out <dir> <paths...>
  <paths>: --paths paths.json, or --w3-code --e1-wrapper --nv-py --v11-tree --nuplan-tree --exp --openscene --maps;
  [--windowspath-shim]

`score` passes exactly `score_navtest_refe.py`'s namespace for a SEAM arm (worker=sequential, hashseed 1,
record_poses, patch_loader), so a Linux run differs from a dev-box run only in paths. `--windowspath-shim` is needed
ONLY for metric-cache pickles written on Windows (navsim v1.1 pickles MetricCache.file_path as a pathlib.Path; see the
wrapper). ⛔ Labels must start with 'refe' (score_navtest_refe.py's rule), so devkit scratch never collides.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def load_w3(a):
    # NT-FIX1 (2026-10-19): W3's run_v1.py is loaded THROUGH run_v1_fix1.load() (it sits beside run_v1.py in the
    # bundle's w3code/), which byte-verifies run_v1.py against its pinned sha256 and adds three fixes measured on
    # RL-1's V1 navtest: (A) a cross-drive os.path.relpath no longer crashes the report, (B) stale *.csv from an
    # earlier attempt are moved to quarantine_* before scoring, (C) a count-mismatched CSV prints and stores NO
    # summary. A missing run_v1_fix1.py is a hard stop, never a fallback to the unfixed module.
    # FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-navtest-harness-fix/RESULT.md
    fix1_path = os.path.join(a.w3_code, "run_v1_fix1.py")
    if not os.path.isfile(fix1_path):
        sys.exit(f"{fix1_path} is missing: pack_code.py must ship run_v1_fix1.py beside run_v1.py in w3code/")
    spec = importlib.util.spec_from_file_location("w3_run_v1_fix1", fix1_path)
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    w3 = fx.load()
    w3.NV_PY = a.nv_py
    w3.V11_TREE = a.v11_tree
    w3.EXP = a.exp
    w3.DATA = a.openscene
    w3.NAVTEST_YAML = f"{a.v11_tree}/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    w3.WRAP = os.path.join(HERE, "navsim_v1_linux.py")
    w3.ENV.update({"NUPLAN_MAPS_ROOT": a.maps, "NAVSIM_EXP_ROOT": a.exp, "NAVSIM_DEVKIT_ROOT": a.v11_tree,
                   "OPENSCENE_DATA_ROOT": a.openscene, "MPLBACKEND": "Agg",
                   # read by navsim_v1_linux.py in the wrapper process
                   "NAVPLAN_W3_CODE": a.w3_code, "NAVPLAN_E1": a.e1_wrapper, "NAVPLAN_V11": a.v11_tree,
                   "NAVPLAN_NUPLAN": a.nuplan_tree, "NAVPLAN_WINPATH_SHIM": "1" if a.windowspath_shim else "0"})
    os.makedirs(a.out, exist_ok=True)
    w3.RAW = os.path.abspath(a.out)
    w3.PKG = os.path.abspath(a.out)
    return w3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score", "cache"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", default=None, help="score: SEAM:<npz> | CV | STOP | HUMAN")
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-name", default="navtest")
    ap.add_argument("--paths", default=None, help="a JSON file carrying the eight path keys below (flags win)")
    keys = ("w3_code", "e1_wrapper", "nv_py", "v11_tree", "nuplan_tree", "exp", "openscene", "maps")
    for k in keys:
        ap.add_argument("--" + k.replace("_", "-"), default=None)
    ap.add_argument("--windowspath-shim", action="store_true")
    a = ap.parse_args(argv)
    if a.paths:
        import json
        pj = json.load(open(a.paths))
        for k in keys:
            if getattr(a, k) is None:
                setattr(a, k, pj[k])
    missing = [k for k in keys if not getattr(a, k)]
    if missing:
        sys.exit(f"missing paths: {missing}")
    if not a.label.startswith("refe"):
        sys.exit("labels must start with 'refe' (score_navtest_refe.py's rule)")
    w3 = load_w3(a)
    if a.cmd == "score":
        if not a.arm:
            sys.exit("score needs --arm")
        arm = a.arm
        if arm.startswith("SEAM:"):
            arm = "SEAM:" + os.path.abspath(arm.split(":", 1)[1])
        ns = types.SimpleNamespace(label=a.label, arm=arm, tokens=a.tokens, cache_name=a.cache_name,
                                   worker="sequential", hashseed="1", record_poses=True, patch_loader=True)
        return w3.cmd_score(ns)
    ns = types.SimpleNamespace(label=a.label, tokens=a.tokens, cache_name=a.cache_name, worker="sequential",
                               per_log=False, logs_per_process=1, min_avail_mb=3500.0, max_wait_min=30.0,
                               log_tries=1)
    return w3.cmd_cache(ns)


if __name__ == "__main__":
    sys.exit(main())
