#!/usr/bin/env python3
"""AlpaSim sensorsim renderer, backed by gsplat — front camera only, aarch64/Thor.

WHAT CONTRACT THIS SATISFIES
----------------------------
AlpaSim's `RendererService` protocol (`alpasim_runtime/services/renderer.py`) is the
runtime-side *client* wrapper; the wire contract it speaks is
`nre.grpc.protos.sensorsim.SensorsimService` (`alpasim_grpc/v0/sensorsim.proto`), which
`base_config.yaml` points at through `services.renderer` / `renderer: null # Optional
override`. This module is a drop-in server for that wire contract, so AlpaSim's stock
`SensorsimService` client can drive it unchanged — NVIDIA's closed `nre-ga:26.04`
container is amd64-only and cannot run on the Jetson Thor.

Implemented RPCs: get_version, get_available_scenes, get_loaded_scenes,
get_available_cameras, get_available_trajectories, get_available_ego_masks,
render_rgb, batch_render_rgb, render_aggregated (rgb part).
render_lidar returns UNIMPLEMENTED **explicitly** rather than an empty cloud — a silent
empty return is exactly the kind of thing that gets mistaken for "the lidar sees nothing".

⛔ FRONT CAMERA ONLY (the PI's explicit steer). `get_available_cameras` advertises exactly
one camera; a request for any other logical_id is refused loudly.

POSE CONVENTION (stated, because getting it wrong renders a plausible wrong picture):
`RGBRenderRequest.sensor_pose.end_pose` is the **camera-optical -> recording-world**
pose, in the same `world` frame as `rig_trajectories.json`'s `T_rig_worlds`. The server
applies the scene's `world_to_nre` itself. `get_available_trajectories` publishes the GT
rig trajectory in that same frame so a client cannot be inconsistent with the server.

Run (Thor):
  export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
  export OMP_NUM_THREADS=6
  PYTHONPATH=$HOME/alpasim/src/grpc:$HOME/nurec_work \
    python sensorsim_gsplat_server.py --scene-dir <dir> --port 6011
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent import futures
from pathlib import Path

import numpy as np

logger = logging.getLogger("sensorsim_gsplat")

CAM_FRONT = "camera_front_wide_120fov"


def _np_to_bytes(img: np.ndarray, fmt: int, quality: float, pb) -> bytes:
    """Encode HxWx3 uint8 RGB in the format the request asked for."""
    IF = pb.ImageFormat
    if fmt == IF.PNG:
        import cv2
        ok, buf = cv2.imencode(".png", img[:, :, ::-1])
        if not ok:
            raise RuntimeError("PNG encode failed")
        return buf.tobytes()
    if fmt == IF.JPEG:
        import cv2
        q = int(np.clip(quality if quality > 0 else 92, 1, 100))
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()
    if fmt == IF.RGB_UINT8_PLANAR:
        return np.ascontiguousarray(img.transpose(2, 0, 1)).tobytes()
    if fmt == IF.UNDEFINED:                     # default to PNG, like NRE
        import cv2
        ok, buf = cv2.imencode(".png", img[:, :, ::-1])
        return buf.tobytes()
    raise NotImplementedError(f"image_format {fmt} not supported by the gsplat renderer")


def _pose_to_T(pose) -> np.ndarray:
    """common.Pose -> 4x4. Convention per the proto: translate, then rotate."""
    from gsplat_renderer import quat_wxyz_to_R
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R([pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z])
    T[:3, 3] = (pose.vec.x, pose.vec.y, pose.vec.z)
    return T


def _T_to_pose(T: np.ndarray, pb_common):
    import math
    R = T[:3, :3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return pb_common.Pose(vec=pb_common.Vec3(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
                          quat=pb_common.Quat(w=w, x=x, y=y, z=z))


def build_servicer(renderer, scene_id: str):
    from alpasim_grpc.v0 import common_pb2 as cpb
    from alpasim_grpc.v0 import sensorsim_pb2 as pb
    from alpasim_grpc.v0 import sensorsim_pb2_grpc as pbg
    import grpc

    class GsplatSensorsim(pbg.SensorsimServiceServicer):
        def __init__(self):
            self.scene_id = scene_id
            self.r = renderer
            self.n_render = 0
            self.render_ms = []

        # -- metadata ----------------------------------------------------------
        def get_version(self, request, context):
            return cpb.VersionId(version_id="tanitad-gsplat-sensorsim-1.0",
                                 git_hash="tanitad",
                                 grpc_api_version=cpb.VersionId.APIVersion(major=0, minor=1, patch=0))

        def get_available_scenes(self, request, context):
            return cpb.AvailableScenesReturn(scene_ids=[self.scene_id])

        def get_loaded_scenes(self, request, context):
            return pb.LoadedScenesReturn(
                scenes=[pb.LoadedSceneEntry(scene_id=self.scene_id,
                                            loaded_instance_count=1,
                                            reusable_instance_count=1)],
                loaded_instance_capacity=1)

        def get_available_ego_masks(self, request, context):
            # We render no ego hood; say so by advertising none rather than
            # advertising one we do not apply.
            return pb.AvailableEgoMasksReturn()

        def _camera_spec(self):
            c = self.r.cam
            ft = pb.FthetaCameraParam(
                principal_point_x=float(c.cx), principal_point_y=float(c.cy),
                reference_poly=(pb.FthetaCameraParam.PIXELDIST_TO_ANGLE
                                if str(c.reference_poly).upper().startswith("PIXELDIST")
                                else pb.FthetaCameraParam.ANGLE_TO_PIXELDIST),
                pixeldist_to_angle_poly=list(c.pixeldist_to_angle_poly),
                angle_to_pixeldist_poly=list(c.angle_to_pixeldist_poly),
                max_angle=float(c.max_angle),
                linear_cde=pb.LinearCde(linear_c=float(c.linear_cde[0]),
                                        linear_d=float(c.linear_cde[1]),
                                        linear_e=float(c.linear_cde[2])))
            return pb.CameraSpec(ftheta_param=ft, logical_id=self.r.cam_name,
                                 resolution_h=self.r.height, resolution_w=self.r.width,
                                 shutter_type=pb.ShutterType.ROLLING_TOP_TO_BOTTOM)

        def get_available_cameras(self, request, context):
            T = self.r.cam.T_sensor_rig
            return pb.AvailableCamerasReturn(available_cameras=[
                pb.AvailableCamerasReturn.AvailableCamera(
                    intrinsics=self._camera_spec(),
                    rig_to_camera=_T_to_pose(np.asarray(T, np.float64), cpb),
                    logical_id=self.r.cam_name)])

        def get_available_trajectories(self, request, context):
            poses = []
            for f in range(self.r.n_frames()):
                ts0, ts1 = self.r.frame_timestamps_us(f)
                poses.append(cpb.PoseAtTime(pose=_T_to_pose(self.r.gt_rig_to_world(f), cpb),
                                            timestamp_us=int(ts1)))
            return pb.AvailableTrajectoriesReturn(available_trajectories=[
                pb.AvailableTrajectoriesReturn.AvailableTrajectory(
                    trajectory=cpb.Trajectory(poses=poses))])

        # -- rendering ---------------------------------------------------------
        def _render_one(self, req, context):
            if req.scene_id and req.scene_id not in ("", "*", self.scene_id):
                context.abort(grpc.StatusCode.NOT_FOUND,
                              f"scene {req.scene_id!r} not loaded (this server serves "
                              f"exactly {self.scene_id!r})")
            lid = req.camera_intrinsics.logical_id or self.r.cam_name
            if lid != self.r.cam_name:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                              f"FRONT CAMERA ONLY: this renderer serves {self.r.cam_name!r}, "
                              f"got {lid!r}. Multi-camera rigs are deliberately out of scope.")
            H = int(req.resolution_h or self.r.height)
            W = int(req.resolution_w or self.r.width)
            cam_to_world = _pose_to_T(req.sensor_pose.end_pose)
            cam_to_nre = self.r.rig.world_to_nre @ cam_to_world
            tau = self.r.tau_of_us(req.frame_end_us) if req.frame_end_us else 0.0
            img, _alpha, ms = self.r.render(cam_to_nre, self.r.width, self.r.height,
                                            tau=tau,
                                            actor_time_us=(float(req.frame_end_us) or None))
            if (H, W) != (self.r.height, self.r.width):
                import cv2
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            self.n_render += 1
            self.render_ms.append(ms)
            return pb.RGBRenderReturn(
                image_bytes=_np_to_bytes(img, req.image_format, req.image_quality, pb))

        def render_rgb(self, request, context):
            return self._render_one(request, context)

        def batch_render_rgb(self, request, context):
            items = []
            for it in request.items:
                try:
                    res = self._render_one(it.request, context)
                    items.append(pb.BatchRGBRenderReturnItem(
                        camera_name=it.camera_name, result=res, success=True))
                except Exception as e:                      # noqa: BLE001
                    items.append(pb.BatchRGBRenderReturnItem(
                        camera_name=it.camera_name, success=False, error_message=repr(e)))
            return pb.BatchRGBRenderReturn(items=items)

        def render_aggregated(self, request, context):
            rgb = [self._render_one(r, context) for r in request.rgb_requests]
            if request.lidar_requests:
                context.abort(grpc.StatusCode.UNIMPLEMENTED,
                              "the gsplat renderer has no lidar model")
            return pb.AggregatedRenderReturn(rgb_returns=rgb)

        def render_lidar(self, request, context):
            context.abort(grpc.StatusCode.UNIMPLEMENTED,
                          "the gsplat renderer has no lidar model — refusing rather "
                          "than returning an empty point cloud")

    return GsplatSensorsim()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6011)
    ap.add_argument("--layers", default="background,road")
    ap.add_argument("--actors", action="store_true", help="attach dynamic_rigids actors")
    ap.add_argument("--loader-dir", default=None)
    ap.add_argument("--ready-file", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import grpc
    from alpasim_grpc.v0 import sensorsim_pb2_grpc as pbg
    from gsplat_renderer import NuRecGsplatRenderer

    sd = Path(args.scene_dir).expanduser()
    t0 = time.time()
    r = NuRecGsplatRenderer(sd, layers=[x for x in args.layers.split(",") if x],
                            loader_dir=args.loader_dir)
    if args.actors:
        from actor_map import attach_actors_verified
        info = attach_actors_verified(r, sd)
        logger.info("actors: %s", {k: v for k, v in info.items() if k != "per_track"})

    scene_id = args.scene_id or f"clipgt-{sd.name}"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                         options=[("grpc.max_receive_message_length", 64 * 1024 * 1024),
                                  ("grpc.max_send_message_length", 64 * 1024 * 1024)])
    svc = build_servicer(r, scene_id)
    pbg.add_SensorsimServiceServicer_to_server(svc, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logger.info("gsplat sensorsim serving on %s:%d  scene=%s  cam=%s %dx%d  "
                "gaussians=%d  boot=%.1fs", args.host, args.port, scene_id,
                r.cam_name, r.width, r.height, r.n_gauss, time.time() - t0)
    if args.ready_file:
        Path(args.ready_file).write_text(f"ready {scene_id} {r.n_gauss}\n")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
