#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate-1 fine-tune DATA EXTRACTOR.

Replays each junction rollout.asl and reconstructs, per planned drive step, the
EXACT REF-C input the driver saw (refc_driver.RefCPolicy.plan preprocessing:
ftheta_crop_resize(center='principal') -> stack_frames(3) -> [8,9,256,256]),
pairs it with:
  - v0   = preds.jsonl `speed` (the exact v0 the driver used; t-matched)
  - nav  = _nav_from_route on the .asl route_request (driver's own logic)
  - tgt  = gate1_rollouts.jsonl `ref_lookahead_rig` [4,2] (expert recovery path,
           rig frame = REF-C's decoder output space) -- the DAgger/CAT-K label

Stores a COMPACT per-rollout bundle: the canonical single-frame sequence
canon_u8 [T,3,256,256] + per-step {k, v0, nav, tgt, pose}. A window is
reconstructed at train time as stack_frames(canon_u8[k-10:k])[-8:], byte-exact
to the driver's stack_frames(deque)[-8:] (crop is per-frame independent).

Built-in checks: f_eff ~= 266 (canonicalization correct) and reconstructed pose
== preds pose (alignment). WITHIN-SIM / ~3.2x OOD (see GATE1_ROLLOUTS_NOTE.md).

Run: cd /workspace/alpa-invest/alpasim &&
  PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python /workspace/gate1_extract.py --out /workspace/gate1_ft_data
"""
from __future__ import annotations
import argparse, asyncio, glob, json, os
from collections import deque
from io import BytesIO
import numpy as np
import torch
from PIL import Image
from alpasim_utils.logs import async_read_pb_log
from alpasim_utils.geometry import quat_to_yaw
from tanitad.data.calib import FThetaIntrinsics, ftheta_crop_resize
from tanitad.data.comma2k19 import stack_frames

LOGDIR = "/workspace/gate1_junc"
PREDS = f"{LOGDIR}/preds.jsonl"
ROLLOUTS = f"{LOGDIR}/gate1_rollouts.jsonl"
CAM = "camera_front_wide_120fov"
NAV = {"follow": 0, "left": 1, "right": 2, "straight": 3}
WINDOW = 8
NEED = 10                      # 3-frame stack (2) + window (8)
SIZE = 256


def intr_from_cameras(cams):
    for cam in cams:
        if not (CAM in cam.logical_id or "front:wide" in cam.logical_id):
            continue
        spec = cam.intrinsics
        if spec.WhichOneof("camera_param") != "ftheta_param":
            continue
        ft = spec.ftheta_param
        fwd = list(ft.angle_to_pixeldist_poly)
        if len(fwd) < 2:
            continue
        return FThetaIntrinsics(poly=tuple(fwd),
                                cx=float(ft.principal_point_x),
                                cy=float(ft.principal_point_y),
                                width=int(spec.resolution_w),
                                height=int(spec.resolution_h),
                                per_clip=True)
    return None


def nav_from_route(wps):
    if not wps:
        return NAV["follow"]
    far = wps[min(len(wps) - 1, 3)]
    x, y = far[0], far[1]
    if x is None or abs(x) < 1e-3:
        return NAV["straight"]
    return NAV["left"] if y > 2.0 else (NAV["right"] if y < -2.0 else NAV["straight"])


def fields_of(msg):
    return {f.name for f, _ in msg.ListFields()}


async def extract_one(asl, preds_rows, tgt_rows):
    """Returns (canon_u8 [T,3,256,256] uint8, steps list, diag dict)."""
    all_raw = []                       # every front-wide raw frame, in order
    route = []
    intr = None
    intr_src = None
    n_recv_at_drive = []               # per driver_request: len(all_raw) so far
    drive_ts = []                      # per driver_request: time_now_us
    drive_nav = []                     # per driver_request: nav at that time
    ego_last = {"x": None, "y": None, "yaw": None}
    drive_pose = []                    # per driver_request: (x,y,yaw)

    async for e in async_read_pb_log(asl):
        f = fields_of(e)
        if "driver_session_request" in f and intr is None:
            sr = e.driver_session_request
            try:
                cams = sr.rollout_spec.vehicle.available_cameras
                got = intr_from_cameras(cams)
                if got is not None:
                    intr, intr_src = got, "driver_session_request"
            except Exception:
                pass
        if "available_cameras_return" in f and intr is None:
            got = intr_from_cameras(e.available_cameras_return.available_cameras)
            if got is not None:
                intr, intr_src = got, "available_cameras_return"
        if "driver_camera_image" in f:
            ci = e.driver_camera_image.camera_image
            if CAM in ci.logical_id or "front:wide" in ci.logical_id:
                arr = np.array(Image.open(BytesIO(ci.image_bytes)).convert("RGB"))
                all_raw.append(arr)
        if "route_request" in f:
            route = [(w.x, w.y, w.z) for w in e.route_request.route.waypoints]
        if "driver_ego_trajectory" in f:
            poses = e.driver_ego_trajectory.trajectory.poses
            if poses:
                p = poses[-1].pose
                ego_last = {"x": float(p.vec.x), "y": float(p.vec.y),
                            "yaw": float(quat_to_yaw(p.quat))}
        if "driver_request" in f:
            n_recv_at_drive.append(len(all_raw))
            drive_ts.append(int(e.driver_request.time_now_us))
            drive_nav.append(nav_from_route(route))
            drive_pose.append((ego_last["x"], ego_last["y"], ego_last["yaw"]))

    if intr is None:
        raise RuntimeError("no ftheta intrinsics found in asl")

    # Canonicalize the FULL received sequence once (crop is per-frame independent,
    # so canon_u8[k-10:k] == the driver's canonicalized last-10-of-deque).
    vid = torch.from_numpy(np.stack(all_raw)).permute(0, 3, 1, 2)   # [T,3,480,854]
    canon = ftheta_crop_resize(vid, intr, SIZE, center="principal") # [T,3,256,256] u8
    f_eff = float(ftheta_crop_resize.last_f_eff)

    # Planned drives = those with >= NEED frames received. Preds/targets are the
    # same set, t-sorted. Pair by planned-step index i.
    planned = [i for i, k in enumerate(n_recv_at_drive) if k >= NEED]
    preds_sorted = sorted(preds_rows, key=lambda r: r["t"])
    if len(planned) != len(preds_sorted):
        # tolerate off-by: match planned drive t to preds t
        pass
    steps = []
    pose_err = []
    used_targets = min(len(planned), len(tgt_rows), len(preds_sorted))
    for i in range(used_targets):
        di = planned[i]
        k = n_recv_at_drive[di]
        v0 = float(preds_sorted[i]["speed"])
        tgt = tgt_rows[i]["ref_lookahead_rig"]        # [4,2] rig frame (fwd,left)
        # validate pose alignment: reconstructed ego vs preds (both world frame)
        px, py = preds_sorted[i]["x"], preds_sorted[i]["y"]
        dp = drive_pose[di]
        if dp[0] is not None:
            pose_err.append(float(np.hypot(dp[0] - px, dp[1] - py)))
        steps.append({"k": int(k), "v0": v0, "nav": int(drive_nav[di]),
                      "tgt": [[float(a), float(b)] for a, b in tgt],
                      "t": int(drive_ts[di]),
                      "pose": [float(px), float(py), float(preds_sorted[i]["yaw"])]})
    diag = {"intr_src": intr_src, "f_eff": round(f_eff, 2),
            "n_raw": len(all_raw), "n_drives": len(n_recv_at_drive),
            "n_planned": len(planned), "n_preds": len(preds_sorted),
            "n_targets": len(tgt_rows), "n_steps": len(steps),
            "pose_err_max_m": round(max(pose_err), 4) if pose_err else None,
            "pose_err_mean_m": round(float(np.mean(pose_err)), 4) if pose_err else None,
            "cx": intr.cx, "cy": intr.cy, "wh": [intr.width, intr.height]}
    return canon.to(torch.uint8), steps, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/gate1_ft_data")
    ap.add_argument("--only", default=None, help="single scene8 for a quick test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    preds = [json.loads(l) for l in open(PREDS) if l.strip()]
    by_sess = {}
    for r in preds:
        by_sess.setdefault(r["session"], []).append(r)
    tgts = [json.loads(l) for l in open(ROLLOUTS) if l.strip()]
    tgt_by_scene = {}
    for r in tgts:
        tgt_by_scene.setdefault(r["scene8"], []).append(r)
    for s in tgt_by_scene:
        tgt_by_scene[s].sort(key=lambda r: r["step"])

    roll_dirs = sorted(glob.glob(f"{LOGDIR}/rollouts/clipgt-*/*/"))
    manifest = {"bundles": [], "total_steps": 0}
    for d in roll_dirs:
        d = d.rstrip("/")
        sess = d.split("/")[-1]
        scene_full = d.split("/")[-2].replace("clipgt-", "")
        scene8 = scene_full[:8]
        if args.only and scene8 != args.only:
            continue
        asl = os.path.join(d, "rollout.asl")
        if not os.path.exists(asl) or sess not in by_sess:
            print(f"  SKIP {scene8}: asl/preds missing")
            continue
        canon, steps, diag = asyncio.run(
            extract_one(asl, by_sess[sess], tgt_by_scene.get(scene8, [])))
        out = os.path.join(args.out, f"{scene8}.pt")
        torch.save({"scene8": scene8, "scene_full": scene_full, "session": sess,
                    "canon_u8": canon, "steps": steps, "diag": diag}, out)
        manifest["bundles"].append({"scene8": scene8, "file": os.path.basename(out),
                                    **diag})
        manifest["total_steps"] += len(steps)
        print(f"  {scene8}: steps={len(steps):2d} f_eff={diag['f_eff']} "
              f"pose_err(max/mean)={diag['pose_err_max_m']}/{diag['pose_err_mean_m']} "
              f"intr={diag['intr_src']} canon={tuple(canon.shape)}")
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"\nTOTAL steps={manifest['total_steps']} bundles={len(manifest['bundles'])}")
    print("wrote", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()
