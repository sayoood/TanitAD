"""Dev-box CPU smoke of ONE ladder arm through the REAL `train()` (3 steps + an in-run eval), BEFORE Thor sees it.

Thor paths -> the dev-box kit (`refcv7_loader.PATH_REMAP`), the eval139 cache standing in as the TRAIN cache (the kit
holds no train cache), the eval139 v9 release (and its md5) for both splits, the X10 sidecar from its package. NOT a
measurement: it proves the arm's argv builds, joins, trains, logs (`r8_spd_crc`, `gs_*`, `cd_*`) and evaluates.
Usage:  python local_smoke.py --argv <W>/argv/<arm>.json --tree <tree> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

KIT = Path(os.environ.get("REFCV6_KIT", "D:/refcv6_eval_kit"))
V9 = "D:/Projects/TanitAD-artifacts/v9labels"
POSE = ("D:/Projects/TanitAD/FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing/raw/"
        "refcv6_pose_sync_sidecar.jsonl")
EVAL_MD5 = "6b5c7f207cffc3b7eb3cd527fd433599"
SPLIT = Path(os.environ.get("LADDER_SMOKE_SPLIT", "C:/Users/Admin/ladder_smoke_cache"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--argv", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lite", action="store_true", help="no grad-share / conflict probe (both fire EVERY step at "
                    "log-every 1 and a full-size CPU step then takes > 45 min); their rows are read on Thor")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.tree) / "stack"))
    os.environ.setdefault("REFCV6_REPO", a.tree)
    from tanitad.eval import refcv7_loader as L
    import ladder_arms as LA
    rec = json.loads(Path(a.argv).read_text(encoding="utf-8"))
    ps = LA.pairs(rec["argv"])
    loc = dict(L.PATH_REMAP)
    # a DISJOINT 8 / 4 split of the eval139 cache (the in-run eval refuses an overlapping eval cache, no override)
    loc.update({"--v2-cache": str(SPLIT / "train"), "--eval-cache": str(SPLIT / "eval"),
                "--v7-labels": str(KIT / "data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"),
                "--r8-v9-labels": f"{V9}/v9_labels_eval139.npz", "--r8-v9-labels-eval": f"{V9}/v9_labels_eval139.npz",
                "--pose-sync-sidecar": POSE,
                "--join-defect-masks": str(Path(a.tree) / "stack/tanitad/configs/refcv8_join_label_defects.json"),
                "--out": a.out})
    out = []
    for f, v in ps:
        if f in loc and v:
            v = [loc[f]]
        out.append((f, v))
    for f, v in (("--r8-v9-md5", [EVAL_MD5]), ("--steps", [str(a.steps)]), ("--batch", ["2"]), ("--workers", ["0"]),
                 ("--device", ["cpu"]), ("--log-every", ["1"]), ("--grad-share-every", ["1"]),
                 ("--eval-every", [str(a.steps)]), ("--eval-batches", ["1"]), ("--save-every", ["1000"]),
                 ("--allow-eval-clips-in-train", []),    # a SMOKE on the eval cache, never scored
                 ("--label-clock-max-unverified", ["0.05"])):   # 3 / 139 eval clips lack a clock
        out = LA.setf(out, f, v)
    if a.lite:
        out = LA.setf(LA.setf(out, "--grad-share-every", ["0"]), "--conflict-detector", ["off"])
    argv = LA.flat(out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "local_argv.json").write_text(json.dumps(argv, indent=1), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=f"{a.tree}/stack;{a.tree}/taniteval", OMP_NUM_THREADS="6",
               CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, f"{a.tree}/stack/scripts/refc_v3_train.py", *argv], env=env,
                       cwd=f"{a.tree}/stack")
    print("TRAIN_RC", r.returncode)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
