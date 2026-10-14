#!/usr/bin/env python3
"""CPU cost of ONE training sample's image path, as train.py's Dataset.__getitem__ does it (refe/train.py:793-810): per camera
cv2.imdecode -> cv2.resize((960, 512)) -> BGR->RGB -> /255 float32 -> CHW, 4 cameras, then torch.stack.
The JPEG is SYNTHETIC (1920x1080, blurred noise + gradient, q=90): a stand-in for nuPlan's 1920x1080 camera JPEGs, UNVERIFIED against navtrain_pixels' real bytes.
    python bench_loader.py --n 40 [--threads N] [--pin 0-3] --out bench_loader.json
Reports process CPU-s and wall-s per sample at cv2's default thread count and at cv2.setNumThreads(1): the difference is what a DataLoader worker on a CFS-quota
cgroup would burn on cv2's own parallel_for threads (cv2 sizes its pool from the VISIBLE cores: 96 on the pod, 11 allowed).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import psutil
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--pin", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
P = psutil.Process()
if a.pin:
    cpus = []
    for part in a.pin.split(","):
        x, _, y = part.partition("-")
        cpus += list(range(int(x), int(y or x) + 1))
    P.cpu_affinity(cpus)
rng = np.random.default_rng(0)
base = cv2.GaussianBlur((rng.random((1080, 1920, 3)) * 255).astype(np.uint8), (0, 0), 6)
base = (0.6 * base + 0.4 * np.linspace(0, 255, 1920, dtype=np.float32)[None, :, None]).astype(np.uint8)
ok, blob = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 90])
blob = np.frombuffer(blob.tobytes(), np.uint8)


def sample():
    fr = []
    for _ in range(4):
        im = cv2.imdecode(blob, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (960, 512))[:, :, ::-1].astype(np.float32) / 255.0
        fr.append(torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))))
    return torch.stack(fr)


res = {"jpeg_bytes": int(blob.size), "cv2_default_threads": cv2.getNumThreads(), "torch_threads": torch.get_num_threads(), "affinity": P.cpu_affinity(), "arms": {}}
for name, nt in (("cv2_default", cv2.getNumThreads()), ("cv2_1thread", 1), ("cv2_default_again", cv2.getNumThreads()), ("cv2_1thread_again", 1)):
    cv2.setNumThreads(nt)
    sample()
    t0 = P.cpu_times()
    w0 = time.perf_counter()
    for _ in range(a.n):
        sample()
    t1 = P.cpu_times()
    w = time.perf_counter() - w0
    c = (t1.user + t1.system) - (t0.user + t0.system)
    res["arms"][name] = {"threads": nt, "cpu_per_sample_s": round(c / a.n, 4), "wall_per_sample_s": round(w / a.n, 4), "cpu_over_wall": round(c / w, 2)}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res["arms"]))
