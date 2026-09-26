#!/usr/bin/env python3
"""refcv6's BEV-lift geometry for a NavSim rig (TANITAD VENV) — rig calibration, not a label.

⭐ WHY THIS EXISTS. refcv6 is built with a perception branch (``--w-map 1 --w-box3d 1``) whose
BEV lift runs INSIDE the forward and feeds the behaviour decoder (``--bev-coupling``). Its
forward therefore needs ``perception_grid`` / ``perception_valid`` — the per-clip lift geometry
the trainer builds from a PER-CLIP mount pose (``LiftGeometryBank``) and ⛔ REFUSES to default
(the corpus' mount height spans 1.2131-1.6672 m; one pose biases every BEV cell). On NavSim the
"camera" is the virtual cylindrical camera of the 3-camera stitch (``frames416.py``), so its
mount pose is DERIVED from the scene's own calibration — admissible (calibration, nothing from
the future), exactly as the training rig's extrinsics are.

THE FRAMES (both stated, both tested):
* refcv6's RIG frame (``tanitad.data.rig_projection``): +x fwd, +y LEFT, +z UP, origin on the rear
  axle AT THE ROAD PLANE (z = 0 is the road, MEASURED on PhysicalAI from cuboid bottoms);
* NavSim / OpenScene's ego frame: the same axes, origin at the rear axle, and ``lidar2ego`` is the
  IDENTITY on every frame read (asserted), so ``sensor2lidar`` IS ``sensor2ego``. ⚠️ Its z = 0 is
  NOT the road: vehicle cuboid bottoms read ~-0.3 m (measured per log below), i.e. the origin sits
  at axle height. The road plane is therefore MEASURED — with the programme's own anchoring method
  (median bottom face ``z - h/2`` of ``vehicle`` boxes near the ego, the method that fixed
  ``rig z = 0`` on PhysicalAI) — and subtracted: camera height above the road = ``F0.z - road_z``.

THE VIRTUAL CAMERA (the stitch's own, ``frames416.virtual_rotation``): rotation cam->ego =
``M`` = [right, down, fwd] (CAM_F0's boresight, rolled level); centre = CAM_F0's
``sensor2lidar_translation`` (the stitch is a pure rotation about one centre; F0's is the one
whose pixels dominate the forward field). -> ``RigCamera(R_cam_to_rig=M, t_cam_in_rig=(x, y,
z - road_z), frame=FRAME_416x1024)`` -> ``bev_lift.build_lift_geometry`` with EXACTLY the
trainer's parameters (``refc_v3_train.py:6827``): ``stride = PerceptionBranchConfig.stride``
(16), ``heights_m = HEIGHTS_M`` (0, 0.5, 1.5, 2.5), ``grid = GRID_DEFAULT``, and ``observed`` =
the bottom ``--equalize-bottom-rows`` (43) rows UNOBSERVED — the trainer passes that mask to its
bank (⚠️ ``taniteval/tools/refcv3_arm.rebuild_perception_branch`` does NOT; see RESULT.md).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frames416 as F4  # noqa: E402  (repo bootstrap + E2's stitch)

import torch  # noqa: E402
from tanitad.data.bev_raster import GRID_DEFAULT  # noqa: E402
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.models.bev_lift import HEIGHTS_M, build_lift_geometry  # noqa: E402
from tanitad.models.trunk_shapes import FRAME_416x1024  # noqa: E402

#: vehicle cuboids used for the road plane: 'vehicle' class, within this box around the ego
ROAD_BOX_X_M = (3.0, 30.0)
ROAD_BOX_Y_M = 10.0
ROAD_MIN_N = 20


def road_plane_from_log(frames: list) -> dict:
    """Median bottom face (``z - h/2``) of nearby ``vehicle`` cuboids over a log, in the NavSim
    ego frame. Returns the estimate with its n and spread; asserts lidar2ego is the identity."""
    zs = []
    n_frames = 0
    for f in frames:
        t = np.asarray(f.get("lidar2ego_translation", [0, 0, 0]), dtype=float)
        q = np.asarray(f.get("lidar2ego_rotation", [1, 0, 0, 0]), dtype=float)
        if np.abs(t).max() > 1e-9 or np.abs(q - np.array([1, 0, 0, 0])).max() > 1e-9:
            raise SystemExit(f"⛔ lidar2ego is not the identity on a frame ({t}, {q}) — the "
                             f"sensor2lidar == sensor2ego premise fails; refusing")
        a = f.get("anns") or {}
        if "gt_boxes" not in a:
            continue
        n_frames += 1
        gb = np.asarray(a["gt_boxes"], dtype=float)
        names = np.asarray(a["gt_names"])
        if gb.size == 0:
            continue
        m = ((names == "vehicle") & (gb[:, 0] >= ROAD_BOX_X_M[0]) & (gb[:, 0] <= ROAD_BOX_X_M[1])
             & (np.abs(gb[:, 1]) <= ROAD_BOX_Y_M))
        zs.extend((gb[m, 2] - gb[m, 5] / 2.0).tolist())
    zs = np.asarray(zs)
    if zs.size < ROAD_MIN_N:
        return {"road_z_m": None, "n_boxes": int(zs.size), "n_frames": n_frames,
                "status": "INSUFFICIENT"}
    return {"road_z_m": float(np.median(zs)), "n_boxes": int(zs.size), "n_frames": n_frames,
            "p10": float(np.percentile(zs, 10)), "p90": float(np.percentile(zs, 90)),
            "status": "OK"}


def rig_camera(rig: dict, road_z_m: float, frame=FRAME_416x1024) -> RigCamera:
    """``rig`` = a ``frames416.rig_record``. The stitch's virtual camera in refcv6's RIG frame."""
    m = torch.tensor(rig["virtual_R_cam_to_lidar"], dtype=torch.float64)
    t = np.asarray(rig["f0_sensor2lidar_translation"], dtype=float)
    tt = torch.tensor([t[0], t[1], t[2] - float(road_z_m)], dtype=torch.float64)
    return RigCamera(R_cam_to_rig=m, t_cam_in_rig=tt, frame=frame)


def observed_mask(frame=FRAME_416x1024, equalize_bottom_rows: int = 43) -> torch.Tensor:
    """The trainer's lift mask: every pixel observed except the bottom ``equalize_bottom_rows``
    rows (``LiftGeometryBank.__init__`` with ``equalize_bottom_rows``)."""
    o = torch.ones(int(frame.height), int(frame.width), dtype=torch.bool)
    if int(equalize_bottom_rows) > 0:
        o[-int(equalize_bottom_rows):, :] = False
    return o


def lift_geometry(rig: dict, road_z_m: float, *, stride: int, heights_m=HEIGHTS_M,
                  equalize_bottom_rows: int = 43, frame=FRAME_416x1024):
    """``(grid [Z,X,Y,2] f32, valid [Z,X,Y] bool, camera summary)`` for one NavSim rig."""
    cam = rig_camera(rig, road_z_m, frame)
    g = build_lift_geometry(cam, frame=frame, stride=int(stride), heights_m=tuple(heights_m),
                            grid=GRID_DEFAULT,
                            observed=observed_mask(frame, equalize_bottom_rows))
    fwd = cam.R_cam_to_rig[:, 2].numpy()
    summ = {"cam_height_m": float(cam.t_cam_in_rig[2]), "cam_x_m": float(cam.t_cam_in_rig[0]),
            "cam_y_m": float(cam.t_cam_in_rig[1]),
            "pitch_down_deg": float(math.degrees(math.asin(-fwd[2]))),
            "yaw_deg": float(math.degrees(math.atan2(fwd[1], fwd[0]))),
            "valid_frac": float(g.valid.float().mean())}
    return g.grid, g.valid, summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="measure the NavSim road plane per log")
    ap.add_argument("--logs", nargs="+", required=True, help="log pickles")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {"method": "median(z - h/2) of 'vehicle' gt_boxes with x in "
                     f"{ROAD_BOX_X_M}, |y| <= {ROAD_BOX_Y_M} m, NavSim ego frame", "logs": {}}
    for p in a.logs:
        fr = F4.E2BF.load(p)
        out["logs"][os.path.basename(p)[:-4]] = road_plane_from_log(fr)
    vals = [v["road_z_m"] for v in out["logs"].values() if v["road_z_m"] is not None]
    out["summary"] = {"n_logs": len(out["logs"]), "n_ok": len(vals),
                      "median": float(np.median(vals)) if vals else None,
                      "min": float(min(vals)) if vals else None,
                      "max": float(max(vals)) if vals else None}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
