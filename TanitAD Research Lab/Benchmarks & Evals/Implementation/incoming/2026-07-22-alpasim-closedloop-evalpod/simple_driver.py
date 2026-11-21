#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal external AlpaSim driver (EgodriverService) for TanitAD closed-loop on the eval pod.

Why this exists: the stock `alpasim_driver` package hard-imports `vam`+`alpamayo_r1`+
`alpamayo1_5` (heavy 10B-model git repos) via `models/__init__.py`, and its `manual`
model needs a pygame display. This standalone driver depends only on `alpasim_grpc`
(protos) + `alpasim_utils.geometry` + numpy + grpcio, runs headless & CPU-only, and
implements the same gRPC EgodriverService interface. Use with
`driver=manual driver_source=external_static` (the wizard then does NOT launch a driver;
it points the runtime at this process at localhost:PORT).

M2: `ConstantForwardPolicy` — straight forward at a fixed speed (stock stand-in driver).
M3: subclass `TrajectoryPolicy.compute_rig_trajectory` to plug in REF-C (camera+nav in,
    rig-frame (x-forward, y-left) trajectory out); the gRPC plumbing here is unchanged.

Coordinate frames (AlpaSim CONTRIBUTING.md): the driver receives ego poses as the active
transform local->rig_est and returns a Trajectory of PoseAtTime in the LOCAL frame. We
generate offsets in the rig frame (x forward, y left) and rotate+translate into local,
mirroring alpasim_driver.main._convert_prediction_to_alpasim_trajectory.
"""
from __future__ import annotations

import argparse
import logging
import threading
from concurrent import futures
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np

import grpc
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import (
    Empty,
    Pose,
    PoseAtTime,
    Quat,
    SessionRequestStatus,
    Trajectory,
    Vec3,
    VersionId,
)
from alpasim_grpc.v0.egodriver_pb2 import DriveResponse
from alpasim_grpc.v0.egodriver_pb2_grpc import (
    EgodriverServiceServicer,
    add_EgodriverServiceServicer_to_server,
)
from alpasim_utils.geometry import quat_to_yaw, yaw_to_quat_components

logger = logging.getLogger("simple_driver")


@dataclass
class Session:
    uuid: str
    seed: int = 0
    poses: list = field(default_factory=list)          # list[PoseAtTime], sorted by ts
    images: dict = field(default_factory=dict)          # logical_id -> (ts_us, np.ndarray)
    route_waypoints: list = field(default_factory=list)  # list[(x,y,z)] rig frame
    n_images: int = 0


class TrajectoryPolicy:
    """Base policy. Override compute_rig_trajectory for a real model (REF-C in M3)."""

    def __init__(self, hz: float = 10.0, horizon_s: float = 5.0):
        self.hz = hz
        self.horizon_s = horizon_s
        self.n = max(1, int(round(hz * horizon_s)))

    def times(self) -> np.ndarray:
        return np.arange(1, self.n + 1) / self.hz  # seconds, starting at dt

    def compute_rig_trajectory(self, session: Session, time_now_us: int):
        """Return (xy[N,2], headings[N]) in the rig frame (x forward, y left)."""
        raise NotImplementedError


class ConstantForwardPolicy(TrajectoryPolicy):
    """M2 stock driver: drive straight forward at a constant speed."""

    def __init__(self, speed_mps: float = 5.0, hz: float = 10.0, horizon_s: float = 5.0):
        super().__init__(hz=hz, horizon_s=horizon_s)
        self.speed = speed_mps

    def compute_rig_trajectory(self, session: Session, time_now_us: int):
        t = self.times()
        xy = np.column_stack([self.speed * t, np.zeros_like(t)])  # x forward, y=0
        headings = np.zeros(self.n)
        return xy, headings


def _rig_offsets_to_local(current_pose: PoseAtTime, offsets_rig: np.ndarray) -> np.ndarray:
    cx = current_pose.pose.vec.x
    cy = current_pose.pose.vec.y
    yaw = quat_to_yaw(current_pose.pose.quat)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    off = np.asarray(offsets_rig, dtype=float).reshape(-1, 2)
    return off @ rot.T + np.array([cx, cy])


class SimpleDriver(EgodriverServiceServicer):
    def __init__(self, policy: TrajectoryPolicy):
        self._policy = policy
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def start_session(self, request, context):
        with self._lock:
            self._sessions[request.session_uuid] = Session(
                uuid=request.session_uuid, seed=request.random_seed
            )
        logger.info("start_session %s", request.session_uuid)
        return SessionRequestStatus()

    def close_session(self, request, context):
        with self._lock:
            self._sessions.pop(request.session_uuid, None)
        logger.info("close_session %s", request.session_uuid)
        return Empty()

    def get_version(self, request, context):
        return VersionId(
            version_id="tanitad-simple-forward-0.1",
            git_hash="tanitad",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    def submit_image_observation(self, request, context):
        img = request.camera_image
        sess = self._sessions.get(request.session_uuid)
        if sess is not None:
            try:
                from PIL import Image

                arr = np.array(Image.open(BytesIO(img.image_bytes)))
                sess.images[img.logical_id] = (img.frame_end_us, arr)
            except Exception:  # noqa: BLE001 - images are optional for the forward policy
                pass
            sess.n_images += 1
        return Empty()

    def submit_egomotion_observation(self, request, context):
        sess = self._sessions.get(request.session_uuid)
        if sess is not None:
            sess.poses.extend(request.trajectory.poses)
            sess.poses.sort(key=lambda p: p.timestamp_us)
        return Empty()

    def submit_route(self, request, context):
        sess = self._sessions.get(request.session_uuid)
        if sess is not None:
            sess.route_waypoints = [(w.x, w.y, w.z) for w in request.route.waypoints]
        return Empty()

    def submit_recording_ground_truth(self, request, context):
        return Empty()

    def drive(self, request, context):
        sess = self._sessions.get(request.session_uuid)
        if sess is None or not sess.poses:
            return DriveResponse(trajectory=Trajectory())

        current = sess.poses[-1]
        xy_rig, headings_rig = self._policy.compute_rig_trajectory(
            sess, request.time_now_us
        )
        local_xy = _rig_offsets_to_local(current, xy_rig)
        curr_yaw = quat_to_yaw(current.pose.quat)
        curr_z = current.pose.vec.z
        dt_us = int(1_000_000 / self._policy.hz)

        traj = Trajectory()
        traj.poses.append(current)  # anchor
        for i, (lx, ly) in enumerate(local_xy, start=1):
            yaw = float(headings_rig[i - 1]) + curr_yaw
            w, x, y, z = yaw_to_quat_components(yaw)
            traj.poses.append(
                PoseAtTime(
                    pose=Pose(
                        vec=Vec3(x=float(lx), y=float(ly), z=float(curr_z)),
                        quat=Quat(w=w, x=x, y=y, z=z),
                    ),
                    timestamp_us=int(request.time_now_us + i * dt_us),
                )
            )
        return DriveResponse(trajectory=traj)


def serve(host: str, port: int, policy: TrajectoryPolicy) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    add_EgodriverServiceServicer_to_server(SimpleDriver(policy), server)
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("SimpleDriver serving on %s (policy=%s)", addr, type(policy).__name__)
    server.wait_for_termination()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6789)
    ap.add_argument("--speed", type=float, default=5.0, help="constant forward speed m/s")
    ap.add_argument("--hz", type=float, default=10.0, help="trajectory output frequency")
    ap.add_argument("--horizon-s", type=float, default=5.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    policy = ConstantForwardPolicy(
        speed_mps=args.speed, hz=args.hz, horizon_s=args.horizon_s
    )
    serve(args.host, args.port, policy)


if __name__ == "__main__":
    main()
