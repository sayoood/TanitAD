"""l1c_panels.py -- build the train-like S2 / S6 panels of a rolled dump tree (PREREG_L1C; the retime_trainfit_r8.cmd_panels logic UNCHANGED, re-pointed).

    python -X utf8 l1c_panels.py --root F:/l1c/tree --tree step48852ema_fresh

Reads  <root>/<tree>/dump_s0   (the dump the l1c roll wrote)   -> writes <root>/<tree>/panel_s0 and <root>/<tree>/s6_s0   (baselines={}: no refcv7 'R' arm).
The environment is the chain's: R8_LAUNCH_TREE, PYTHONPATH=<tree>/stack;<tree>/taniteval.  CPU only.
"""
import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
R8 = os.environ.get("L1_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code")
sys.path.insert(0, R8)

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--tree", required=True)
a = ap.parse_args()
import retime_trainfit_r8 as TF  # noqa: E402

TF.TRAIN_ROOT, TF.TRAIN_TREE, TF.SEEDS = a.root, a.tree, (0,)
raise SystemExit(TF.cmd_panels())
