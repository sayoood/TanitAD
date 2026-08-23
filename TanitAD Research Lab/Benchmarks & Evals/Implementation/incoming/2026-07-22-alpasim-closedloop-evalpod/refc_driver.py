#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M3/M4: REF-C as a TRUSTWORTHY AlpaSim external driver (EgodriverService).

Option A build (Sayed-approved 2026-07-22). Wraps `tanitad.refs.refc.RefCModel`
(base 104.2M / XL 251.9M) behind the gRPC EgodriverService, drops into the M2
closed-loop topology (`driver_source=external_static`, localhost:6789).

MAKE-OR-BREAK RESOLVED (asl_camera_probe.py): NuRec renders NATIVE f-theta
`camera_front_wide_120fov`. The renderer hands the driver the ftheta CameraSpec
in `DriveSessionRequest.rollout_spec.vehicle.available_cameras[*].intrinsics`
(forward poly `angle_to_pixeldist`, cx, cy, native res) — so we build the
per-scene `FThetaIntrinsics` from the SESSION (no USDZ parse, no poly inversion).

Preprocessing = REF-C's TRAINING canonicalization (validated in B2, f_eff self-
check): per-frame `ftheta_crop_resize(center="principal")` -> 3-frame stack
(`comma2k19.stack_frames`) -> [W=8, 9, 256, 256] -> REF-C. NOT the naive resize
(B2: naive shifts the plan 0.747 m > REF-C's 0.4728 ADE budget).

Gate-2 waypoint timing: REF-C emits 4 ego-frame waypoints at 0.5/1/1.5/2.0 s
(horizons 5/10/15/20 @ dt=0.1); we stamp them at those exact times (NOT uniform hz).

WARNING: label every closed-loop number "REF-C on NuRec reconstructions" (sim2real
caveat) — reconstructions are not REF-C's real-footage training distribution.

Run (alpasim venv + tanitad stack on PYTHONPATH):
  PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    /workspace/alpa-invest/alpasim/.venv/bin/python /workspace/refc_driver.py \
      --port 6789 --ckpt /root/models/refc-base-30k/ckpt.pt --preset base
"""
from __future__ import annotations

import argparse
import logging
import threading
from collections import deque
from concurrent import futures
from io import BytesIO

import numpy as np
import torch
from PIL import Image

import grpc
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import (Empty, Pose, PoseAtTime, Quat,
                                        SessionRequestStatus, Trajectory, Vec3,
                                        VersionId)
from alpasim_grpc.v0.egodriver_pb2 import DriveResponse
from alpasim_grpc.v0.egodriver_pb2_grpc import (
    EgodriverServiceServicer, add_EgodriverServiceServicer_to_server)
from alpasim_utils.geometry import quat_to_yaw, yaw_to_quat_components

from tanitad.data.calib import F_REF, FThetaIntrinsics, ftheta_crop_resize
from tanitad.data.comma2k19 import stack_frames
from refc_v12_cache import load_frozen

logger = logging.getLogger("refc_driver")

CAM = "camera_front_wide_120fov"
HORIZON_S = (0.5, 1.0, 1.5, 2.0)          # REF-C waypoint times
NAV = {"follow": 0, "left": 1, "right": 2, "straight": 3}
WINDOW = 8                                 # REF-C state window
NEED_FRAMES = WINDOW + 2                    # 3-frame stack costs 2, need >=10


def _intrinsics_from_camera(cam) -> FThetaIntrinsics | None:
    """Build TanitAD FThetaIntrinsics from an AvailableCamera proto (ftheta only)."""
    spec = cam.intrinsics
    if spec.WhichOneof("camera_param") != "ftheta_param":
        return None
    ft = spec.ftheta_param
    fwd = list(ft.angle_to_pixeldist_poly)     # forward: angle -> pixel (TanitAD poly)
    if not fwd or len(fwd) < 2:
        return None
    return FThetaIntrinsics(poly=tuple(fwd),
                            cx=float(ft.principal_point_x),
                            cy=float(ft.principal_point_y),
                            width=int(spec.resolution_w),
                            height=int(spec.resolution_h),
                            per_clip=True)


class RefCPolicy:
    """Loads REF-C and turns a frame history + intrinsics into a rig-frame plan."""

    def __init__(self, ckpt: str, preset: str = "base", device: str = "cuda"):
        dev = device if torch.cuda.is_available() else "cpu"
        self.model, self.cfg, self.step = load_frozen(ckpt, preset, None, dev)
        self.device = dev
        self.window = int(self.cfg.window)
        self.n_steps = len(self.cfg.trajectory.horizons)
        self._feff_checked = False
        logger.info("REF-C %s loaded (step=%s, window=%d, anchors=%d) on %s",
                    preset, self.step, self.window, self.cfg.anchors.n_anchors, self.device)

    @torch.no_grad()
    def plan(self, raw_frames: list, intr: FThetaIntrinsics, v0: float, nav_cmd: int):
        """raw_frames: list of HxWx3 uint8 (oldest..newest, >= NEED_FRAMES).
        Returns (xy[4,2], headings[4]) in the rig frame (x-fwd, y-left)."""
        vid = torch.from_numpy(np.stack(raw_frames)).permute(0, 3, 1, 2)   # [T,3,H,W] u8
        canon = ftheta_crop_resize(vid, intr, 256, center="principal")     # [T,3,256,256] u8
        if not self._feff_checked:
            fe = float(ftheta_crop_resize.last_f_eff)
            ok = abs(fe - F_REF) < 8.0
            logger.info("CANON f_eff=%.1f (F_REF=%.1f) %s", fe, F_REF, "OK" if ok else "FAIL")
            if not ok:
                logger.error("f_eff self-check FAILED — canonicalization wrong")
            self._feff_checked = True
        stacked = stack_frames(canon, 3)                                   # [T-2,9,256,256]
        fw = stacked[-self.window:][None].to(self.device).float().div_(255.0)  # [1,W,9,256,256]
        v0t = torch.tensor([v0], dtype=torch.float32, device=self.device)
        navt = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
        out = self.model(fw, nav_cmd=navt, v0=v0t, steps=2)                # eval() deterministic
        traj = out["traj"][0].cpu().numpy()                                # [4,2] rig frame
        d = np.diff(np.vstack([[0.0, 0.0], traj]), axis=0)
        headings = np.arctan2(d[:, 1], np.maximum(d[:, 0], 1e-6))
        return traj, headings


class RefCDriver(EgodriverServiceServicer):
    def __init__(self, policy: RefCPolicy, log_preds: str | None = None):
        self._p = policy
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._log_preds = log_preds     # JSONL path for open-loop diagnostic (predictions + pose)

    def start_session(self, request, context):
        intr = None
        for cam in request.rollout_spec.vehicle.available_cameras:
            if CAM in cam.logical_id or "front:wide" in cam.logical_id:
                intr = _intrinsics_from_camera(cam)
                if intr is not None:
                    logger.info("start_session %s: ftheta cx=%.1f cy=%.1f %dx%d poly1=%.1f",
                                request.session_uuid, intr.cx, intr.cy, intr.width,
                                intr.height, intr.poly[1])
                break
        if intr is None:
            logger.error("start_session %s: NO ftheta intrinsics for %s", request.session_uuid, CAM)
        with self._lock:
            self._sessions[request.session_uuid] = {
                "intr": intr, "frames": deque(maxlen=24), "speed": 0.0,
                "route": [], "poses": []}
        return SessionRequestStatus()

    def close_session(self, request, context):
        with self._lock:
            self._sessions.pop(request.session_uuid, None)
        return Empty()

    def get_version(self, request, context):
        return VersionId(version_id=f"tanitad-refc-{self._p.step}", git_hash="tanitad",
                         grpc_api_version=API_VERSION_MESSAGE)

    def submit_image_observation(self, request, context):
        s = self._sessions.get(request.session_uuid)
        img = request.camera_image
        if s is not None and (CAM in img.logical_id or "front:wide" in img.logical_id):
            arr = np.array(Image.open(BytesIO(img.image_bytes)).convert("RGB"))
            s["frames"].append(arr)
        return Empty()

    def submit_egomotion_observation(self, request, context):
        s = self._sessions.get(request.session_uuid)
        if s is not None:
            s["poses"].extend(request.trajectory.poses)
            s["poses"].sort(key=lambda p: p.timestamp_us)
            if request.dynamic_states:
                ds = request.dynamic_states[-1]
                s["speed"] = float((ds.linear_velocity.x ** 2 + ds.linear_velocity.y ** 2) ** 0.5)
        return Empty()

    def submit_route(self, request, context):
        s = self._sessions.get(request.session_uuid)
        if s is not None:
            s["route"] = [(w.x, w.y, w.z) for w in request.route.waypoints]
        return Empty()

    def submit_recording_ground_truth(self, request, context):
        return Empty()

    def _nav_from_route(self, s) -> int:
        wps = s["route"]
        if not wps:
            return NAV["follow"]
        far = wps[min(len(wps) - 1, 3)]
        x, y = far[0], far[1]
        if x is None or abs(x) < 1e-3:
            return NAV["straight"]
        return NAV["left"] if y > 2.0 else (NAV["right"] if y < -2.0 else NAV["straight"])

    def drive(self, request, context):
        s = self._sessions.get(request.session_uuid)
        if s is None or not s["poses"]:
            return DriveResponse(trajectory=Trajectory())
        current = s["poses"][-1]
        if s["intr"] is None or len(s["frames"]) < NEED_FRAMES:
            return DriveResponse(trajectory=Trajectory())     # warmup: hold
        # v0: prefer the dynamic-state speed; fall back to pose finite-difference
        # (force-GT bypasses the controller and sends no dynamic state -> speed=0).
        v0 = s["speed"]
        if v0 < 0.1 and len(s["poses"]) >= 2:
            p1, p0 = s["poses"][-1], s["poses"][-2]
            dt = (p1.timestamp_us - p0.timestamp_us) / 1e6
            if dt > 1e-3:
                v0 = ((p1.pose.vec.x - p0.pose.vec.x) ** 2
                      + (p1.pose.vec.y - p0.pose.vec.y) ** 2) ** 0.5 / dt
        xy_rig, head_rig = self._p.plan(list(s["frames"]), s["intr"], v0,
                                        self._nav_from_route(s))
        cx, cy = current.pose.vec.x, current.pose.vec.y
        yaw = quat_to_yaw(current.pose.quat)
        c, sn = np.cos(yaw), np.sin(yaw)
        if self._log_preds:  # OPEN-LOOP diagnostic: log the rig-frame prediction + world pose
            import json
            with open(self._log_preds, "a") as f:
                f.write(json.dumps({
                    "session": request.session_uuid, "t": int(request.time_now_us),
                    "x": float(cx), "y": float(cy), "z": float(current.pose.vec.z),
                    "yaw": float(yaw), "speed": float(v0),
                    "pred_rig": [[float(xy_rig[i, 0]), float(xy_rig[i, 1])]
                                 for i in range(len(xy_rig))]}) + "\n")
        traj = Trajectory()
        traj.poses.append(current)
        for i in range(len(xy_rig)):
            dx, dy = float(xy_rig[i, 0]), float(xy_rig[i, 1])
            lx = cx + c * dx - sn * dy
            ly = cy + sn * dx + c * dy
            w, x, y, z = yaw_to_quat_components(float(head_rig[i]) + yaw)
            traj.poses.append(PoseAtTime(
                pose=Pose(vec=Vec3(x=lx, y=ly, z=current.pose.vec.z),
                          quat=Quat(w=w, x=x, y=y, z=z)),
                timestamp_us=int(request.time_now_us + HORIZON_S[i] * 1_000_000)))
        return DriveResponse(trajectory=traj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6789)
    ap.add_argument("--ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--preset", default="base", choices=["base", "small", "xl", "smoke"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-preds", default=None,
                    help="JSONL path: log rig-frame predictions + world pose per drive "
                         "(open-loop diagnostic; use with a force-GT rollout)")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    policy = RefCPolicy(ckpt=args.ckpt, preset=args.preset, device=args.device)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    add_EgodriverServiceServicer_to_server(RefCDriver(policy, log_preds=args.log_preds), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logger.info("RefCDriver serving on %s:%d (preset=%s ckpt=%s)",
                args.host, args.port, args.preset, args.ckpt)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
