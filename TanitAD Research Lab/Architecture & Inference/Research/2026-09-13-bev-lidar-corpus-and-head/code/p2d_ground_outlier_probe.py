#!/usr/bin/env python3
"""P2d - two clips quarantined on the ground-plane re-derivation (peak z +1.175 / +1.475 m
against a declared band of +/-0.20 m). Wrong extrinsics, or a scene with no visible ground?

Discriminators, each with a control clip that passed:
  1. LiDAR + camera `sensor_extrinsics` of the outliers against the corpus distribution
     (z, quaternion) -- a mount-geometry defect shows up here and nowhere else;
  2. the near-range (3-20 m) z histogram: top-3 modes and the mass below 0.3 m -- a scene
     without ground (tunnel, garage, traffic jam) keeps ground-like returns somewhere; a
     vertical offset moves the WHOLE distribution;
  3. the LOWEST dense z level over the full 360 deg (5th percentile of z within 3-40 m) -- the
     road is the lowest large surface; an offset moves it, an occluding scene does not.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_build_corpus as B  # noqa: E402
from lidar_bev import lidar_to_rig  # noqa: E402
from lidar_fetch import LIDAR_CACHE, sha12  # noqa: E402

OUTLIERS = {"e902aba7fb9a", "ff9d2a66d2a2", "693335b810c3", "7be92e6870d8", "570b47ad7408"}
CONTROLS = {"c64f0103aece", "d0c6db065fa0", "d4c144f52747"}   # quarantine-free, parquets kept (rule-A run)


def main() -> int:
    import pyarrow.parquet as pq
    import DracoPy
    join = B.join_clips()
    by = {sha12(c): c for c in join}
    # corpus distribution of the lidar extrinsics
    ext_all = []
    for c in join:
        try:
            ext_all.append(B.extrinsics(c, "lidar_top_360fov"))
        except KeyError:
            pass
    zs = np.array([e["z"] for e in ext_all])
    res = {"schema": "tanitad.ground_outlier_probe/1", "evidence_class": "MEASURED (ours)",
           "corpus_lidar_ext_z": {"n": int(len(zs)), "min": float(zs.min()), "median": float(np.median(zs)),
                                  "max": float(zs.max())},
           "clips": []}
    avail = {sha12(Path(p).name.split(".")[0]) for p in glob.glob(str(Path(LIDAR_CACHE) / "*.parquet"))}
    for s in sorted(OUTLIERS | CONTROLS):
        if s not in avail or s not in by:
            res["clips"].append({"clip": s, "note": "parquet not available"})
            continue
        cid = by[s]
        le = B.extrinsics(cid, "lidar_top_360fov")
        ce = B.extrinsics(cid, "camera_front_wide_120fov")
        lp = Path(LIDAR_CACHE) / f"{cid}.lidar_top_360fov.parquet"
        pf = pq.ParquetFile(lp)
        peaks, p5, low_mass, modes = [], [], [], []
        for k in np.linspace(5, pf.metadata.num_row_groups - 6, 8).round().astype(int):
            blob = pf.read_row_group(int(k), columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
            pts = np.asarray(DracoPy.decode(blob).points)
            rig = lidar_to_rig(pts, le)
            r = np.hypot(rig[:, 0], rig[:, 1])
            m = (r > 3) & (r < 20)
            h, e = np.histogram(rig[m, 2], bins=np.arange(-3, 4.001, 0.05))
            top = np.argsort(h)[::-1][:3]
            modes.append([round(float(e[t] + 0.025), 3) for t in top])
            peaks.append(float(e[int(np.argmax(h))] + 0.025))
            m40 = (r > 3) & (r < 40)
            p5.append(float(np.percentile(rig[m40, 2], 5)))
            low_mass.append(float((rig[m, 2] < 0.3).mean()))
        res["clips"].append({
            "clip": s, "role": "OUTLIER" if s in OUTLIERS else "control",
            "lidar_ext": {k: round(v, 4) for k, v in le.items()},
            "camera_ext_z": round(ce["z"], 4),
            "ground_peak_z_per_spin": [round(x, 3) for x in peaks],
            "top3_modes_per_spin": modes,
            "z_p5_3_40m_per_spin": [round(x, 3) for x in p5],
            "frac_near_points_below_0.3m": [round(x, 3) for x in low_mass],
        })
        print(json.dumps(res["clips"][-1]), flush=True)
    print(json.dumps(res["corpus_lidar_ext_z"]))
    (HERE.parent / "raw" / "p2d_ground_outlier_probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
