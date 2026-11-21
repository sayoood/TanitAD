#!/usr/bin/env python3
"""pai_extrinsics_table.py — clip_id -> the MEASURED front-wide camera extrinsic.

Emits the JSON that ``render_refcv3_video.py --extrinsics`` consumes, read from
PhysicalAI-AV's OWN ``calibration/sensor_extrinsics`` parquet (via
``clip_index.parquet`` for the clip -> chunk map). Nothing here is fitted,
assumed or inherited.

⛔ WHY THIS EXISTS AND WHY IT IS NOT A CONSTANT. Three camera-height constants
circulate in this repo — 1.22 m (``tanitad/replay/rr_log.py``), 1.43 m and 1.5 m
(``taniteval/cam_overlay.py``) — and MEASURED over 40 PhysicalAI clips
(``…/pod-rescue-20260802/pod3/workspace/idm3_geom.py``) **all three are wrong as
a constant**: the observed range is **1.245-1.607 m**, median **1.306 m**, 37
distinct values in 40 clips, CV 7.4 %. 1.22 is BELOW the observed minimum.
⚠️ And the rig label is not a proxy: rig-A and rig-B medians differ by **1.5 %**
while the WITHIN-rig spread is **29 %**.

⚠️ The camera also sits **~2.0-2.1 m FORWARD** of the vehicle origin and carries
a per-clip pitch of **-1.15 .. +2.34 deg**. Both terms move the near-field
overlay by metres, which is exactly where the trajectory is.

PhysicalAI-AV is GATED, so the parquet is not in this repo — this reads whatever
local copy the machine holds and writes a small JSON the renderer can consume
without the dataset.

    python taniteval/tools/pai_extrinsics_table.py \
        --root <physicalai root, the parent of calibration/> \
        --clips <cid,cid,...>  --out extrinsics.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

SENSOR = "camera_front_wide_120fov"


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="the PhysicalAI root holding calibration/ and "
                         "clip_index.parquet")
    ap.add_argument("--clips", required=True, help="comma-separated clip_ids")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import pandas as pd

    idx_p = os.path.join(a.root, "clip_index.parquet")
    if not os.path.exists(idx_p):
        sys.exit(f"[pai_extrinsics] no clip_index.parquet at {idx_p}")
    ci = pd.read_parquet(idx_p)
    cids = [c.strip() for c in a.clips.split(",") if c.strip()]
    out, missing = {}, []
    for cid in cids:
        if cid not in ci.index:
            missing.append((cid, "not in clip_index.parquet"))
            continue
        ch = int(ci.loc[cid, "chunk"])
        p = os.path.join(a.root, "calibration", "sensor_extrinsics",
                         f"sensor_extrinsics.chunk_{ch:04d}.parquet")
        if not os.path.exists(p):
            missing.append((cid, f"chunk {ch} parquet absent locally"))
            continue
        e = pd.read_parquet(p)
        try:
            r = e.loc[(cid, SENSOR)]
        except KeyError:
            missing.append((cid, f"no {SENSOR} row in chunk {ch}"))
            continue
        R = quat_to_R(float(r.qx), float(r.qy), float(r.qz), float(r.qw))
        axis = [R[i][2] for i in range(3)]        # camera +z (boresight)
        out[cid] = {
            "qx": float(r.qx), "qy": float(r.qy), "qz": float(r.qz),
            "qw": float(r.qw), "x": float(r.x), "y": float(r.y),
            "z": float(r.z), "chunk": ch, "sensor": SENSOR,
            "source": f"{os.path.basename(p)} (PhysicalAI-AV sensor_extrinsics)",
            "_read": {
                "camera_height_m": round(float(r.z), 4),
                "camera_forward_offset_m": round(float(r.x), 4),
                "pitch_down_deg": round(math.degrees(math.atan2(
                    -axis[2], math.hypot(axis[0], axis[1]))), 4),
                "convention": "t = camera position in the VEHICLE frame; R = "
                              "camera->vehicle. Camera axes +x right, +y DOWN, "
                              "+z boresight (calib.ftheta_project_ray). A ground "
                              "point p_veh maps to p_cam = R^T (p_veh - t).",
            },
        }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    for cid, why in missing:
        print(f"[pai_extrinsics] MISSING {cid}: {why}  -> the renderer will "
              f"DISABLE the camera overlay for it and say so on the frame",
              flush=True)
    print(f"[pai_extrinsics] {len(out)}/{len(cids)} clips -> {a.out}", flush=True)
    for cid, v in out.items():
        r = v["_read"]
        print(f"  {cid}  h={r['camera_height_m']:.3f} m  "
              f"fwd={r['camera_forward_offset_m']:.3f} m  "
              f"pitch_down={r['pitch_down_deg']:+.3f} deg", flush=True)


if __name__ == "__main__":
    main()
