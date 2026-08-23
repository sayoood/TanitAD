#!/usr/bin/env python3
"""Two open format questions, answered from the file:
  Q1  where do dynamic_deformables' per-track time ranges live?
  Q2  is the sky cubemap's [6,H,W,3] reshape the RIGHT memory layout?
      (all six face means within 0.315-0.327 is suspicious: either a scrambled
       layout or values that need an activation. A correct cubemap of a night
       street has structure INSIDE each face; a scrambled one looks like noise.)
"""
import json
import sys
from pathlib import Path

import numpy as np

SCENE = Path(sys.argv[1]).expanduser()
sys.path.insert(0, "/home/nvidia/nurec-gsplat")
from nurec_loader import NuRecScene, read_volume_nurec  # noqa: E402

sc = NuRecScene(read_volume_nurec(SCENE / "volume.nurec"), quat_layout="wxyz")
out = {}

# ---- Q1 ---------------------------------------------------------------------
q1 = {}
for L in ("dynamic_rigids", "dynamic_deformables"):
    ks = [k for k in sc.sd if k.startswith(f".gaussians_nodes.{L}.")
          and "timestamps_us_ranges" in k and not k.endswith(".shape")]
    q1[L] = {}
    for k in ks:
        arr = np.frombuffer(sc.sd[k], np.int64)
        shp = sc.sd.get(k + ".shape")
        q1[L][k[len(f".gaussians_nodes.{L}."):]] = {
            "declared_shape": list(shp) if shp is not None else None,
            "n_int64": int(arr.size),
            "first_rows": arr.reshape(-1, 2)[:3].tolist() if arr.size % 2 == 0 else None,
            "span_s": (arr.reshape(-1, 2)[:, 1] - arr.reshape(-1, 2)[:, 0]).astype(float).tolist()[:6]
            if arr.size % 2 == 0 else None,
        }
    cid = np.frombuffer(sc.sd[f".gaussians_nodes.{L}.gaussian_cuboid_ids"], np.int32)
    q1[L]["cuboid_ids_unique"] = np.unique(cid).tolist()[:40]
    q1[L]["n_gauss_per_cuboid"] = {int(c): int((cid == c).sum())
                                   for c in np.unique(cid)[:40]}
out["Q1_deformable_time_ranges"] = q1

# ---- Q2 ---------------------------------------------------------------------
k = ".background.textures"
raw = np.frombuffer(sc.sd[k], np.float16).astype(np.float32)
bg = sc.cfg["background"]
H, W = bg["height"], bg["width"]
shp = sc.sd.get(k + ".shape")
q2 = {"declared_shape": list(shp) if shp is not None else None,
      "n_elems": int(raw.size), "expect_6HW3": int(6 * H * W * 3),
      "min": float(raw.min()), "max": float(raw.max()), "mean": float(raw.mean()),
      "n_negative": int((raw < 0).sum()), "n_gt1": int((raw > 1).sum())}


def smoothness(a):
    """Mean |Laplacian| relative to std — an IMAGE is smooth, noise is not."""
    a = a.astype(np.float32)
    lap = np.abs(4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1]
                 - a[1:-1, :-2] - a[1:-1, 2:])
    return float(lap.mean() / max(a.std(), 1e-9))


for name, arr in (("6_H_W_3", raw.reshape(6, H, W, 3)),
                  ("6_3_H_W", raw.reshape(6, 3, H, W).transpose(0, 2, 3, 1)),
                  ("3_6_H_W", raw.reshape(3, 6, H, W).transpose(1, 2, 3, 0)),
                  ("H_W_6_3", raw.reshape(H, W, 6, 3).transpose(2, 0, 1, 3))):
    q2[name] = {
        "face_means": [round(float(arr[i].mean()), 4) for i in range(6)],
        "face_stds": [round(float(arr[i].std()), 4) for i in range(6)],
        "smoothness_luma": [round(smoothness(arr[i].mean(-1)), 3) for i in range(6)],
    }
out["Q2_cubemap_layout"] = q2
print(json.dumps(out, indent=1, default=float))
