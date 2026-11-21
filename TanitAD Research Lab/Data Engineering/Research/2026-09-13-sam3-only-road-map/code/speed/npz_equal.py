"""Bit-level comparison of two front-only raw frame directories (the files refine reads).
Keys compared: tok, T_world_rig, cls_CAM_FW, evid_CAM_FW, xstripe_CAM_FW, pts_1..7, rng_1..7 -- exact array equality (shape, dtype,
values). Also, for rasters that differ: share of pixels that differ over the pixels labelled in either (cls > 0), per frame.
Usage: npz_equal.py <ref dir> <test dir> [max frames]  ->  prints ZZNPZEQ-<n_frames>-<n_identical>-<n_diff>ZZ"""
import json, sys
from pathlib import Path
import numpy as np

ref, test = Path(sys.argv[1]), Path(sys.argv[2])
lim = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
KEYS = ["tok", "T_world_rig", "cls_CAM_FW", "evid_CAM_FW", "xstripe_CAM_FW"] + [f"pts_{k}" for k in range(1, 8)] + [f"rng_{k}" for k in range(1, 8)]
files = sorted(test.glob("[0-9][0-9][0-9].npz"))[:lim]
n_id = n_diff = 0; worst = []; agree = []
for f in files:
    a = np.load(ref / f.name, allow_pickle=True); b = np.load(f, allow_pickle=True)
    bad = []
    for k in KEYS:
        if k not in a.files or k not in b.files:
            bad.append(f"{k}:missing({k in a.files},{k in b.files})"); continue
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y):
            bad.append(f"{k}:{x.shape}{x.dtype}/{y.shape}{y.dtype}")
    ca, cb = a["cls_CAM_FW"], b["cls_CAM_FW"]
    lab = (ca > 0) | (cb > 0)
    agree.append(1.0 - float(((ca != cb) & lab).sum()) / max(int(lab.sum()), 1))
    if bad:
        n_diff += 1; worst.append((f.name, bad[:6]))
    else:
        n_id += 1
print(f"frames {len(files)} identical {n_id} differing {n_diff}; cls agreement over labelled px: mean {np.mean(agree):.6f} min {np.min(agree):.6f}")
for w in worst[:8]:
    print("  ", w)
print(f"ZZNPZEQ-{len(files)}-{n_id}-{n_diff}ZZ")
