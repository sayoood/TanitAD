"""Yardstick for the stage-1 render gap (descriptive, decides nothing).
Stage 1 measured the flattened, uncentered cosine of the [40, 1024] pooled tokens between a NuRec frame rendered at the logged pose and the
real frame: median 0.8366 (scene 00040136) and 0.7404 (scene 7c72937c) (`stage1/pull_box2/pull/STAGE1_RESULT.json`, steps.gap).
This script measures the SAME statistic (t2a_v8f_encoder.cosine) between REAL frames of the eval-141 cache: two frames of different clips,
and two frames of one clip k ticks apart. Seeded (numpy default_rng 0). usage: python -I gapscale.py <cache clips dir> <out.json>"""
import glob
import json
import os
import sys

import numpy as np
import torch


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def q(x):
    x = np.asarray(x)
    return {"n": int(x.size), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max())}


def main(clips_dir, out):
    files = sorted(glob.glob(os.path.join(clips_dir, "*.pt")))
    toks = [torch.load(f, map_location="cpu", weights_only=False)["pool4x10"].to(torch.float32) for f in files]
    mean = torch.cat(toks).double().mean(0)
    rng = np.random.default_rng(0)
    across = []
    for _ in range(2000):
        i, j = rng.choice(len(toks), 2, replace=False)
        across.append(cos(toks[i][rng.integers(len(toks[i]))], toks[j][rng.integers(len(toks[j]))]))
    lag = {}
    for k in (1, 10, 30, 50):
        v = []
        for _ in range(2000):
            t = toks[rng.integers(len(toks))]
            if len(t) <= k:
                continue
            a = rng.integers(len(t) - k)
            v.append(cos(t[a], t[a + k]))
        lag[f"k{k}"] = q(v)
    to_mean = [cos(toks[i][f], mean) for i in range(len(toks)) for f in range(0, len(toks[i]), 10)]
    res = {"statistic": "t2a_v8f_encoder.cosine: flattened, uncentered cosine of pool4x10 [40,1024] tokens", "cache": clips_dir,
           "n_clips": len(toks), "n_frames": int(sum(len(t) for t in toks)),
           "stage1_render_vs_real_median": {"00040136": 0.836640459240057, "7c72937c": 0.7404358159002863},
           "real_across_clips": q(across), "real_within_clip_lag_ticks": lag, "real_frame_vs_eval_mean": q(to_mean)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cache",)}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
