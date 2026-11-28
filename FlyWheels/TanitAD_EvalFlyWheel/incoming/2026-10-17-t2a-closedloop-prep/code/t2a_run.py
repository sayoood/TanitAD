"""Entry point: register the T2-A arms, then run the in-house closed-loop harness (``closedloop_drive.main``).

    python t2a_run.py --scene-dir <NuRec scene> --arm refcv7|cl_p10|cl_ha0_ext_live|cl_ha0|cl_ha0_ext|cl_ha ...
                      --ckpt <ckpt_50400.pt> --starts 0,17,34,51,68,85,102,119,136 --steps 50 --out <dir>
                      [--t2a-vmax-ms X] [--t2a-infer-seed N] [--t2a-p10 p10_S2.pt]

Layout (the kit):   <kit>/t2a/   this directory
                    <kit>/tree/  {stack, taniteval} at the LAUNCH commit fec3a0dc + the overlay (patched
                                 closedloop_drive.py, taniteval/floors_kin.py)         -> env T2A_TREE overrides
Environment:
    T2A_TREE=<dir>     the code tree (default <this dir>/../tree)
    T2A_STUB=1         inject the CPU stub renderer (dry path). FORCES CPU-ONLY.
    T2A_CPU_ONLY=1     hide every GPU (CUDA_VISIBLE_DEVICES=-1; an EMPTY string does not hide it on this box) and assert.
    REFCV6_KIT=<dir>   the eval kit whose data/ holds the anchors / tau / class-weights files the refcv7 loader reads
                       (default <kit>/refcv6_eval_kit)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUB = os.environ.get("T2A_STUB") == "1"
if STUB or os.environ.get("T2A_CPU_ONLY") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
TREE = Path(os.environ.get("T2A_TREE") or (HERE.parent / "tree")).resolve()
if not (TREE / "stack").is_dir():
    raise SystemExit(f"[t2a_run] code tree {TREE} has no stack/ (set T2A_TREE or build the kit with make_kit.py)")
if (HERE.parent / "refcv6_eval_kit").is_dir():
    os.environ.setdefault("REFCV6_KIT", str(HERE.parent / "refcv6_eval_kit"))
for p in (TREE / "taniteval" / "tools", TREE / "taniteval", TREE / "stack" / "scripts", TREE / "stack",
          TREE / "stack" / "experiments" / "nurec-gsplat", TREE / "stack" / "experiments" / "alpasim-gsplat", HERE):
    sys.path.insert(0, str(p))

if STUB or os.environ.get("T2A_CPU_ONLY") == "1":
    import torch
    assert not torch.cuda.is_available(), "a GPU is visible but this run is CPU-ONLY"

import closedloop_drive as cd  # noqa: E402
import t2a_policies  # noqa: E402

if not hasattr(cd, "EXTRA_ARMS"):
    raise SystemExit("[t2a_run] closedloop_drive.py in the tree lacks EXTRA_ARMS: the overlay was not applied")
t2a_policies.register()
if STUB:
    import stub_gsplat_renderer
    stub_gsplat_renderer.install()

if __name__ == "__main__":
    cd.main()
