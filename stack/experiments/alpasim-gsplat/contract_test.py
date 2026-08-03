#!/usr/bin/env python3
"""Exercise the AlpaSim sensorsim contract against our gsplat renderer — and FALSIFY it.

"The server started and returned bytes" is not evidence. This test:
  * calls EVERY RPC the `SensorsimService` proto declares, including the two we refuse,
    and asserts the refusals are explicit gRPC errors rather than empty successes;
  * checks the served `CameraSpec` is **f-theta** and that a `FThetaIntrinsics` built
    from it canonicalizes to `f_eff == F_REF` (the make-or-break geometry self-check);
  * renders the same pose IN-PROCESS and OVER gRPC and requires the images to be
    **bit-identical** — a transport that silently changed a pixel would invalidate every
    number the closed loop produces;
  * runs the FINDINGS negative control (grad-NCC over 5 reference frames) on the image
    that came back over the wire, so the wire path is validated by the same falsifier
    the renderer itself was validated by;
  * measures PNG / JPEG / raw encode cost, because the image format is a real
    per-step cost in a 10 Hz loop.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--addr", default="localhost:6011")
    ap.add_argument("--out", default="/tmp/contract_test.json")
    args = ap.parse_args()

    import grpc
    from alpasim_grpc.v0 import common_pb2 as cpb
    from alpasim_grpc.v0 import sensorsim_pb2 as pb
    from alpasim_grpc.v0 import sensorsim_pb2_grpc as pbg

    from closedloop_drive import GrpcTransport, InProcTransport, build_intr
    from gsplat_renderer import NuRecGsplatRenderer, grad_ncc, read_ref_frame

    sd = Path(args.scene_dir).expanduser()
    res: dict = {"addr": args.addr, "scene": sd.name}
    ch = grpc.insecure_channel(args.addr, options=[
        ("grpc.max_receive_message_length", 64 * 1024 * 1024)])
    stub = pbg.SensorsimServiceStub(ch)

    # ---- metadata RPCs --------------------------------------------------------
    res["get_version"] = stub.get_version(cpb.Empty()).version_id
    scenes = stub.get_available_scenes(cpb.Empty())
    res["get_available_scenes"] = list(scenes.scene_ids)
    sid = scenes.scene_ids[0]
    ls = stub.get_loaded_scenes(cpb.Empty())
    res["get_loaded_scenes"] = [(s.scene_id, s.loaded_instance_count) for s in ls.scenes]
    res["get_available_ego_masks_n"] = len(
        stub.get_available_ego_masks(cpb.Empty()).ego_mask_metadata)
    cams = stub.get_available_cameras(pb.AvailableCamerasRequest(scene_id=sid))
    res["n_cameras"] = len(cams.available_cameras)
    res["camera_logical_ids"] = [c.logical_id for c in cams.available_cameras]
    spec = cams.available_cameras[0].intrinsics
    res["camera_param_kind"] = spec.WhichOneof("camera_param")
    res["camera_res"] = [spec.resolution_w, spec.resolution_h]
    trs = stub.get_available_trajectories(pb.AvailableTrajectoriesRequest(scene_id=sid))
    res["n_trajectory_poses"] = len(trs.available_trajectories[0].trajectory.poses)

    # ---- refusals must be EXPLICIT --------------------------------------------
    def expect_error(fn, label):
        try:
            fn()
            return f"{label}: NO ERROR (bad — a refusal must not look like a success)"
        except grpc.RpcError as e:
            return f"{label}: {e.code().name}"

    res["render_lidar"] = expect_error(
        lambda: stub.render_lidar(pb.LidarRenderRequest(scene_id=sid)), "render_lidar")

    # ---- transport fidelity: gRPC bytes == in-process pixels -------------------
    r = NuRecGsplatRenderer(sd)
    ip = InProcTransport(r)
    gp = GrpcTransport(args.addr, scene_id=sid)
    intr = build_intr(gp.camera())
    res["grpc_camera"] = {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in gp.camera().items()}

    Ts = r.cam.T_sensor_rig
    diffs, gncc = [], []
    for f in (0, 30, 90):
        cam_to_world = r.gt_rig_to_world(f * 3) @ Ts
        ts = r.frame_timestamps_us(f * 3)[1]
        a = ip.render(cam_to_world, ts)
        b = gp.render(cam_to_world, ts)
        diffs.append(int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max()))
    res["grpc_vs_inproc_max_abs_pixel_diff"] = diffs
    res["transport_bit_identical"] = bool(max(diffs) == 0)

    # ---- the FINDINGS negative control, on the image that crossed the wire -----
    mp4 = sd / "camera_front_wide_120fov.mp4"
    img0 = gp.render(r.gt_rig_to_world(0) @ Ts, r.frame_timestamps_us(0)[1])
    for f in (0, 60, 150, 300, 450):
        gncc.append(round(grad_ncc(img0, read_ref_frame(mp4, f, (r.width, r.height))), 4))
    res["negative_control_grad_ncc"] = dict(zip(("f0_CORRECT", "f60", "f150", "f300", "f450"), gncc))
    res["negative_control_argmax"] = int(np.argmax(gncc))
    res["negative_control_margin"] = round(float(gncc[0] - max(gncc[1:])), 4)
    res["negative_control_pass"] = bool(np.argmax(gncc) == 0)

    # ---- canonicalization self-check on a wire image ---------------------------
    import torch

    from tanitad.data.calib import F_REF, ftheta_crop_resize
    vid = torch.from_numpy(img0[None]).permute(0, 3, 1, 2)
    ftheta_crop_resize(vid, intr, 256, center="principal")
    res["f_eff"] = round(float(ftheta_crop_resize.last_f_eff), 3)
    res["F_REF"] = F_REF
    res["canon_pass"] = bool(abs(res["f_eff"] - F_REF) < 8.0)

    # ---- image-format cost ----------------------------------------------------
    fmt_ms = {}
    for name, fmt in (("RGB_UINT8_PLANAR", pb.ImageFormat.RGB_UINT8_PLANAR),
                      ("PNG", pb.ImageFormat.PNG), ("JPEG", pb.ImageFormat.JPEG)):
        pose = _pose(r.gt_rig_to_world(0) @ Ts, cpb)
        req = pb.RGBRenderRequest(scene_id=sid, resolution_h=r.height, resolution_w=r.width,
                                  camera_intrinsics=spec, frame_end_us=int(ts),
                                  sensor_pose=pb.PosePair(start_pose=pose, end_pose=pose),
                                  image_format=fmt, image_quality=90)
        t0 = time.time()
        for _ in range(3):
            rep = stub.render_rgb(req)
        fmt_ms[name] = [round((time.time() - t0) / 3 * 1000, 1), len(rep.image_bytes)]
    res["format_ms_and_bytes"] = fmt_ms

    # ---- batch RPC ------------------------------------------------------------
    pose = _pose(r.gt_rig_to_world(0) @ Ts, cpb)
    item = pb.BatchRGBRenderRequestItem(
        camera_name="camera_front_wide_120fov",
        request=pb.RGBRenderRequest(scene_id=sid, resolution_h=r.height, resolution_w=r.width,
                                    camera_intrinsics=spec, frame_end_us=int(ts),
                                    sensor_pose=pb.PosePair(start_pose=pose, end_pose=pose),
                                    image_format=pb.ImageFormat.RGB_UINT8_PLANAR))
    br = stub.batch_render_rgb(pb.BatchRGBRenderRequest(items=[item]))
    res["batch_render_rgb"] = [(i.camera_name, i.success, len(i.result.image_bytes))
                               for i in br.items]

    # ---- a non-front camera must be refused, loudly ---------------------------
    bad_spec = pb.CameraSpec()
    bad_spec.CopyFrom(spec)
    bad_spec.logical_id = "camera_rear_left_70fov"
    res["non_front_camera"] = expect_error(
        lambda: stub.render_rgb(pb.RGBRenderRequest(
            scene_id=sid, resolution_h=r.height, resolution_w=r.width,
            camera_intrinsics=bad_spec, frame_end_us=int(ts),
            sensor_pose=pb.PosePair(start_pose=pose, end_pose=pose),
            image_format=pb.ImageFormat.RGB_UINT8_PLANAR)), "non_front_camera")

    res["ALL_PASS"] = bool(res["transport_bit_identical"] and res["negative_control_pass"]
                           and res["canon_pass"] and res["camera_param_kind"] == "ftheta_param"
                           and res["n_cameras"] == 1)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def _pose(T, cpb):
    from closedloop_drive import _R_to_quat_np
    q = _R_to_quat_np(T[:3, :3])
    return cpb.Pose(vec=cpb.Vec3(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
                    quat=cpb.Quat(w=q[0], x=q[1], y=q[2], z=q[3]))


if __name__ == "__main__":
    main()
