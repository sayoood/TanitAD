#!/usr/bin/env python3
"""align_check.py -- the KNOWN ANSWER of PREREG_FROZEN.md section 3 (dev box, CPU).  Reads box 2's out/align_*.npy (pulled to SCR/align_out/) and the cache's own pool4x10 rows.
GATE: median relative L2 <= 0.05 and NO row > 0.5.  Writes raw/align_check.json; raw/ALIGN_GATE_PASS only on pass.  Also reports (not gated) the JPEG-vs-lossless crop-feature difference."""
from __future__ import annotations

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = f"{HERE}/raw"
SCR = "F:/TanitAD-scratch/trigger_probe5_scratch"
TRAIN_CACHE = "F:/TanitAD-scratch/speedread/cache/train/clips"
EVAL_CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips"
GATE_MEDIAN, GATE_MAX = 0.05, 0.5


def rel_l2(a, b):
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def verdict(rl):
    """the gate as a pure function (pytested)."""
    rl = np.asarray(rl)
    return bool(np.median(rl) <= GATE_MEDIAN and rl.max() <= GATE_MAX)


def main():
    meta = json.load(open(f"{RAW}/align_meta.json"))
    out = f"{SCR}/align_out"
    ours = np.load(f"{out}/align_full_pool4x10.npy")
    cached = []
    for m in meta:
        d = TRAIN_CACHE if m["src"] == "train" else EVAL_CACHE
        o = torch.load(f"{d}/{m['clip']}.pt", map_location="cpu", weights_only=True, mmap=True)
        cached.append(o["pool4x10"][m["t"]].float().numpy())
    cached = np.stack(cached)
    rl = rel_l2(ours, cached)
    png, jpg = np.load(f"{out}/align_crops_png.npy"), np.load(f"{out}/align_crops_jpg.npy")
    jl = rel_l2(jpg.reshape(len(jpg) * 2, -1), png.reshape(len(png) * 2, -1))
    info = json.load(open(f"{out}/align_info.json"))
    res = {"n": len(rl), "rel_l2_median": float(np.median(rl)), "rel_l2_mean": float(rl.mean()), "rel_l2_max": float(rl.max()), "rel_l2_min": float(rl.min()),
           "per_row": [round(float(x), 4) for x in rl], "gate_median": GATE_MEDIAN, "gate_max": GATE_MAX, "pass": verdict(rl) and bool(info["weights_ok"]),
           "weights_ok": info["weights_ok"], "weights_sha256": info["weights_sha256"], "jpeg_vs_png_crop_rel_l2_median": float(np.median(jl)), "jpeg_vs_png_crop_rel_l2_max": float(jl.max()),
           "crop_feature_norm_mean": float(np.linalg.norm(png.reshape(len(png) * 2, -1), axis=1).mean())}
    json.dump(res, open(f"{RAW}/align_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_row"}))
    if res["pass"]:
        open(f"{RAW}/ALIGN_GATE_PASS", "w").write("pass")
    else:
        print("ALIGN GATE FAILED: package refused")
        sys.exit(2)


if __name__ == "__main__":
    main()
