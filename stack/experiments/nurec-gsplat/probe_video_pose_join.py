#!/usr/bin/env python3
"""Probe: how does the clipgt POSE index relate to the reference-video FRAME index?

⛔ This exists because an off-by-stride here would score every decision against the
wrong observation and still produce a plausible-looking number. The join is MEASURED
(timestamps on both sides), never assumed, and the residual is printed so a silent
mismatch is impossible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def probe(scene_dir: Path, members_dir: Path, cam="camera_front_wide_120fov"):
    import pyarrow.parquet as pq
    from nurec_loader import RigTrajectories

    out = {"scene": scene_dir.name}

    t = pq.read_table(members_dir / "clipgt" / "egomotion_estimate.parquet")
    keys = t.column("key").to_pylist()
    ego = t.column("egomotion_estimate").to_pylist()
    ts_pose = np.array([k["timestamp_micros"] for k in keys], np.int64)
    P = np.array([[d["location"]["x"], d["location"]["y"]] for d in ego], float)
    out["n_poses"] = len(ts_pose)
    out["pose_dt_us_median"] = int(np.median(np.diff(ts_pose)))

    rig = RigTrajectories(members_dir / "rig_trajectories.json")
    out["cameras"] = rig.camera_names()
    n_cf = rig.n_frames(cam)
    ts_cam = np.array([rig.frame_timestamps_us(cam, i) for i in range(n_cf)], np.int64)
    out["n_camera_frames"] = n_cf
    out["cam_dt_us_median"] = int(np.median(np.diff(ts_cam[:, 0])))
    c = rig.camera(cam)
    out["camera"] = {"w": c.width, "h": c.height, "cx": round(c.cx, 2),
                     "cy": round(c.cy, 2), "poly1": round(c.angle_to_pixeldist_poly[1], 2),
                     "shutter": c.shutter_type, "max_angle": round(c.max_angle, 4)}

    # the join: nearest camera frame (shutter START) to each pose timestamp
    j = np.abs(ts_cam[:, 0][None, :] - ts_pose[:, None]).argmin(1)
    resid_us = np.abs(ts_cam[j, 0] - ts_pose)
    out["pose_to_camframe"] = {
        "first5": j[:5].tolist(), "last5": j[-5:].tolist(),
        "stride_median": float(np.median(np.diff(j))),
        "residual_us_max": int(resid_us.max()),
        "residual_us_median": int(np.median(resid_us)),
    }

    mp4 = scene_dir / f"{cam}.mp4"
    if mp4.exists():
        import cv2
        cap = cv2.VideoCapture(str(mp4))
        out["mp4"] = {"n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                      "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 4),
                      "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        ok, fr = cap.read()
        out["mp4"]["first_frame_shape"] = None if not ok else list(fr.shape)
        cap.release()
    else:
        out["mp4"] = f"absent ({mp4})"

    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    dt = np.diff(ts_pose) / 1e6
    out["speed_mps"] = {"mean": round(float((seg / dt).mean()), 3),
                        "max": round(float((seg / dt).max()), 3)}
    return out


if __name__ == "__main__":
    root = Path("/home/nvidia/nurec_scenes/sample_set/26.04_release")
    mem = HERE / "scene_members"
    for sid in sys.argv[1:] or [p.name for p in sorted(root.iterdir()) if p.is_dir()]:
        print(json.dumps(probe(root / sid, mem / sid), indent=1))
