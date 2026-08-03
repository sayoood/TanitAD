#!/usr/bin/env python3
"""Closed-loop driving of TanitAD policies inside a NuRec scene, on the Jetson Thor.

THE LOOP (the model's own actions determine its next observation):
  render(ego pose) -> f-theta canonicalize -> policy -> waypoints
       -> pure-pursuit control -> kinematic bicycle -> new ego pose -> render ...

The renderer is reached over **gRPC, through AlpaSim's own generated
`SensorsimServiceStub`** (`sensorsim_gsplat_server.py`), so the loop exercises the same
wire contract AlpaSim's runtime speaks. `--inproc` bypasses gRPC for speed once the
contract has been demonstrated; the two are compared head-to-head by `--verify-transport`.

WHAT IS REUSED (never reinvented)
  * `tanitad.data.calib.ftheta_crop_resize(center="principal")` + `comma2k19.stack_frames`
    — the exact training canonicalization, with the `f_eff == F_REF` self-check.
  * `taniteval.closedloop.wp_to_control` — the pure-pursuit + P-speed harness controller.
  * `scripts/refb_labels.classify_maneuver_v2` / `nav_command_v21` / `route_from_future_v21`
    — the programme's OWN manoeuvre and route labelling, for the tactical/strategic
    metric families. A metric family must not invent its own class boundaries.

⚠️ EVERY number produced here is **WITHIN-SIM RELATIVE**. REF-C's open-loop ADE is 1.5157
on these reconstructions vs 0.4728 on real footage (3.21x OOD). Orderings survive;
absolute rates do not.

GROUND FOLLOWING: the bicycle is planar, the road is not. After each step the rig's
z / roll / pitch are taken from the nearest logged rig pose while x / y / yaw come from
the model's own driving. This is a HARNESS choice, stated so it is never mistaken for
a physics engine.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("closedloop")

CAM = "camera_front_wide_120fov"
DT = 0.1
WINDOW = 8
STACK = 3
NEED_FRAMES = WINDOW + STACK - 1          # 10 native frames -> [8,9,256,256]
HORIZON_S = (0.5, 1.0, 1.5, 2.0)
WP_STEPS = (5, 10, 15, 20)
LOOKAHEAD_IDX = 0                          # the 0.5 s waypoint
MAN_NAMES = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")
NAV_NAMES = ("follow", "left", "right", "straight")


# ------------------------------------------------------------------------------- #
# transports                                                                        #
# ------------------------------------------------------------------------------- #
class InProcTransport:
    """Direct renderer calls — same maths, no serialisation."""

    def __init__(self, renderer):
        self.r = renderer
        self.render_ms = []

    def camera(self):
        c = self.r.cam
        return dict(cx=float(c.cx), cy=float(c.cy), width=int(c.width), height=int(c.height),
                    poly=tuple(float(x) for x in c.angle_to_pixeldist_poly))

    def render(self, cam_to_world: np.ndarray, ts_us: float) -> np.ndarray:
        cam_to_nre = self.r.rig.world_to_nre @ cam_to_world
        tau = self.r.tau_of_us(ts_us)
        img, _a, ms = self.r.render(cam_to_nre, tau=tau, actor_time_us=float(ts_us))
        self.render_ms.append(ms)
        return img


class GrpcTransport:
    """AlpaSim `SensorsimService` client — the contract path."""

    def __init__(self, addr: str, scene_id: str = "*"):
        import grpc
        from alpasim_grpc.v0 import sensorsim_pb2 as pb
        from alpasim_grpc.v0 import sensorsim_pb2_grpc as pbg
        self.pb = pb
        self.ch = grpc.insecure_channel(
            addr, options=[("grpc.max_receive_message_length", 64 * 1024 * 1024),
                           ("grpc.max_send_message_length", 64 * 1024 * 1024)])
        self.stub = pbg.SensorsimServiceStub(self.ch)
        self.scene_id = scene_id
        self.spec = None
        self.render_ms = []
        self._cam = self._fetch_camera()

    def _fetch_camera(self):
        from alpasim_grpc.v0.sensorsim_pb2 import AvailableCamerasRequest
        rep = self.stub.get_available_cameras(AvailableCamerasRequest(scene_id=self.scene_id))
        if not rep.available_cameras:
            raise RuntimeError("renderer advertises no cameras")
        c = rep.available_cameras[0]
        if c.intrinsics.WhichOneof("camera_param") != "ftheta_param":
            raise RuntimeError("renderer did not serve an f-theta CameraSpec — the "
                               "canonicalization would be wrong. Refusing.")
        self.spec = c.intrinsics
        ft = c.intrinsics.ftheta_param
        return dict(cx=float(ft.principal_point_x), cy=float(ft.principal_point_y),
                    width=int(c.intrinsics.resolution_w), height=int(c.intrinsics.resolution_h),
                    poly=tuple(float(x) for x in ft.angle_to_pixeldist_poly))

    def camera(self):
        return self._cam

    def gt_trajectory(self):
        from alpasim_grpc.v0.sensorsim_pb2 import AvailableTrajectoriesRequest
        rep = self.stub.get_available_trajectories(
            AvailableTrajectoriesRequest(scene_id=self.scene_id))
        return rep.available_trajectories[0].trajectory.poses

    def render(self, cam_to_world: np.ndarray, ts_us: float) -> np.ndarray:
        from alpasim_grpc.v0 import common_pb2 as cpb
        pb = self.pb
        q = _R_to_quat_np(cam_to_world[:3, :3])
        pose = cpb.Pose(vec=cpb.Vec3(x=cam_to_world[0, 3], y=cam_to_world[1, 3],
                                     z=cam_to_world[2, 3]),
                        quat=cpb.Quat(w=q[0], x=q[1], y=q[2], z=q[3]))
        req = pb.RGBRenderRequest(
            scene_id=self.scene_id, resolution_h=self._cam["height"],
            resolution_w=self._cam["width"], camera_intrinsics=self.spec,
            frame_start_us=int(ts_us), frame_end_us=int(ts_us),
            sensor_pose=pb.PosePair(start_pose=pose, end_pose=pose),
            image_format=pb.ImageFormat.RGB_UINT8_PLANAR)
        t0 = time.time()
        rep = self.stub.render_rgb(req)
        self.render_ms.append((time.time() - t0) * 1000.0)
        h, w = self._cam["height"], self._cam["width"]
        return np.frombuffer(rep.image_bytes, np.uint8).reshape(3, h, w).transpose(1, 2, 0)


def _R_to_quat_np(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                         (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


# ------------------------------------------------------------------------------- #
# policies                                                                          #
# ------------------------------------------------------------------------------- #
class _BasePolicy:
    name = "?"

    def __init__(self, device="cuda"):
        import torch
        from tanitad.data.calib import F_REF, ftheta_crop_resize
        from tanitad.data.comma2k19 import stack_frames
        self.torch, self.F_REF = torch, F_REF
        self._crop, self._stack = ftheta_crop_resize, stack_frames
        self.device = device if torch.cuda.is_available() else "cpu"
        self.f_eff = None

    def canon(self, frames, intr):
        torch = self.torch
        vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        canon = self._crop(vid, intr, 256, center="principal")
        if self.f_eff is None:
            self.f_eff = float(self._crop.last_f_eff)
            ok = abs(self.f_eff - self.F_REF) < 8.0
            logger.info("CANON f_eff=%.2f (F_REF=%.1f) %s", self.f_eff, self.F_REF,
                        "OK" if ok else "FAIL")
            if not ok:
                raise RuntimeError(f"f_eff self-check FAILED ({self.f_eff}) — the model "
                                   "would see a raster it was never trained at. Refusing.")
        st = self._stack(canon, STACK)                       # [T-2, 9, 256, 256]
        fw = st[-WINDOW:][None].to(self.device).float().div_(255.0)
        if tuple(fw.shape[-3:]) != (9, 256, 256) or fw.shape[1] != WINDOW:
            raise RuntimeError(f"raster assertion failed: {tuple(fw.shape)} != "
                               f"(1,{WINDOW},9,256,256)")
        return fw


class FlagshipV1Policy(_BasePolicy):
    name = "flagship-v1"

    def __init__(self, ckpt, device="cuda"):
        super().__init__(device)
        torch = self.torch
        from tanitad.config import flagship4b_config
        from tanitad.models.fourbrain import WorldModel
        cfg = flagship4b_config()
        object.__setattr__(cfg.predictor, "action_dim", 3)
        if getattr(cfg, "tactical_pred", None) is not None:
            object.__setattr__(cfg.tactical_pred, "action_dim", 3)
        self.model = WorldModel(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ck["model"])
        self.model = self.model.to(self.device).eval()
        self.step = ck.get("step")
        self.horizons = list(cfg.tactical_policy.waypoint_horizons)
        logger.info("flagship v1 loaded step=%s horizons=%s", self.step, self.horizons)

    def plan(self, frames, intr, v0, nav_cmd):
        torch = self.torch
        with torch.no_grad():
            fw = self.canon(frames, intr)
            nav = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
            states = self.model.encode_window(fw)
            sout = self.model.strategic_policy(states, nav)
            tout = self.model.tactical_policy(states, sout["ctx"])
            wpd = tout["waypoints"]
            traj = np.stack([wpd[h][0].float().cpu().numpy() for h in self.horizons])
        extra = {}
        for k, v in list(tout.items()) + [("s_" + k, v) for k, v in sout.items()]:
            if hasattr(v, "shape") and v.ndim == 2 and v.shape[0] == 1 and v.shape[1] <= 16:
                extra[k] = v[0].float().cpu().numpy().tolist()
        return traj.astype(np.float64), extra


class RefCPolicy(_BasePolicy):
    def __init__(self, ckpt, preset="base", device="cuda"):
        super().__init__(device)
        self.name = f"refc-{preset}"
        from refc_v12_cache import load_frozen
        self.model, self.cfg, self.step = load_frozen(ckpt, preset, None, self.device)
        logger.info("REF-C %s loaded step=%s anchors=%d", preset, self.step,
                    self.cfg.anchors.n_anchors)

    def plan(self, frames, intr, v0, nav_cmd):
        torch = self.torch
        with torch.no_grad():
            fw = self.canon(frames, intr)
            v0t = torch.tensor([float(v0)], dtype=torch.float32, device=self.device)
            navt = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
            out = self.model(fw, nav_cmd=navt, v0=v0t, steps=2)
            traj = out["traj"][0].float().cpu().numpy()
        extra = {}
        for k, v in out.items():
            if hasattr(v, "shape") and v.ndim == 2 and v.shape[0] == 1 and v.shape[1] <= 16:
                extra[k] = v[0].float().cpu().numpy().tolist()
        return traj.astype(np.float64), extra


# ------------------------------------------------------------------------------- #
# the loop                                                                          #
# ------------------------------------------------------------------------------- #
def _yaw(T):
    return math.atan2(T[1, 0], T[0, 0])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    R = np.eye(3)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def _ego_xy(p_world, T_ego):
    """World point -> ego frame (x fwd, y left)."""
    d = np.asarray(p_world, np.float64)[:2] - T_ego[:2, 3]
    y = _yaw(T_ego)
    c, s = math.cos(-y), math.sin(-y)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


# ⛔ These are `taniteval.closedloop`'s OWN constants, copied deliberately rather than
# re-chosen. STEER_CLAMP=0.05 rad looks tiny until you read its comment: the corpus has
# |steer| <= 0.016, so 0.05 is already 3x head-room. Loosening it here would make this
# harness incomparable with every published TanitAD closed-loop number.
STEER_CLAMP, ACCEL_CLAMP, SPEED_TC, LD2_FLOOR, WHEELBASE = 0.05, 3.0, 0.5, 0.25, 2.7


def wp_to_control(w_look, v, wheelbase=WHEELBASE, steer_clamp=STEER_CLAMP,
                  accel_clamp=ACCEL_CLAMP, speed_tc=SPEED_TC, ld2_floor=LD2_FLOOR):
    """taniteval.closedloop.wp_to_control, scalar form (same formulae + clamps)."""
    x, y = float(w_look[0]), float(w_look[1])
    ld2 = max(x * x + y * y, ld2_floor)
    kappa = 2.0 * y / ld2
    steer = float(np.clip(math.atan(wheelbase * kappa), -steer_clamp, steer_clamp))
    v_target = x / (WP_STEPS[LOOKAHEAD_IDX] * DT)
    accel = float(np.clip((v_target - v) / speed_tc, -accel_clamp, accel_clamp))
    return steer, accel, v_target, kappa


class GroundFollower:
    """Holds the ego on the logged road surface: x/y/yaw are the model's, z/roll/pitch
    are the nearest logged rig pose's. Stated as a harness choice, not physics."""

    def __init__(self, gt_T):
        self.T = gt_T
        self.xy = np.stack([T[:2, 3] for T in gt_T])

    def nearest(self, xy):
        return int(np.argmin(np.linalg.norm(self.xy - np.asarray(xy)[None, :2], axis=1)))

    def correct(self, T):
        i = self.nearest(T[:3, 3])
        G = self.T[i]
        R_rp = _rz(-_yaw(G)) @ G[:3, :3]                   # roll/pitch of the log
        out = np.eye(4)
        out[:3, :3] = _rz(_yaw(T)) @ R_rp
        out[:3, 3] = (T[0, 3], T[1, 3], G[2, 3])
        return out, i


def gt_poses_xyv(gt_T, dt=DT):
    """[T,4] (x, y, yaw, v) from the logged rig poses — the format refb_labels wants."""
    xy = np.stack([T[:2, 3] for T in gt_T])
    yaw = np.array([_yaw(T) for T in gt_T])
    v = np.zeros(len(gt_T))
    if len(gt_T) > 1:
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
        v[:-1] = d
        v[-1] = d[-1]
    return np.stack([xy[:, 0], xy[:, 1], yaw, v], 1)


def nav_from_route(gtp, i, horizon_steps=None):
    """Programme-native strategic command from the logged route ahead of index i.

    ⚠️ `nav_command_v21`'s default lookahead is 25 s with a 15 s minimum, but this
    scene is only 20 s long — so from most timesteps the canonical call is
    *structurally* invalid and returns (follow, valid=False). That is reported, not
    hidden: `valid` travels with every nav we emit, and a scene-length-adapted
    short-horizon variant is computed alongside it for the strategic family.
    """
    import torch
    from refb_labels import nav_command_v21
    t = torch.from_numpy(gtp).float()
    out = {}
    try:
        nav, valid = nav_command_v21(t, int(i))
        out["nav_canonical"], out["nav_valid"] = int(nav), bool(valid)
    except Exception as e:                                   # noqa: BLE001
        out["nav_canonical"], out["nav_valid"], out["nav_err"] = 0, False, repr(e)[:80]
    try:
        h = int(horizon_steps or min(60, max(10, gtp.shape[0] - int(i) - 2)))
        nav_s, valid_s = nav_command_v21(t, int(i), horizon_steps=h, min_steps=10)
        out["nav_short"], out["nav_short_valid"], out["nav_short_h"] = int(nav_s), bool(valid_s), h
    except Exception as e:                                   # noqa: BLE001
        out["nav_short"], out["nav_short_valid"] = 0, False
        out["nav_short_err"] = repr(e)[:80]
    nav = out["nav_canonical"] if out["nav_valid"] else out.get("nav_short", 0)
    return int(nav), out


def maneuver_of(poses_xyv):
    """[H+1,4] sub-path -> class index, via the programme's own v2 classifier."""
    import torch
    from refb_labels import classify_maneuver_v2
    t = torch.from_numpy(np.asarray(poses_xyv, np.float32))[None]
    return int(classify_maneuver_v2(t)[0])


def plan_to_poses(traj, v0):
    """Model plan [4,2] ego-frame -> [H+1,4] (x,y,yaw,v) sub-path at 10 Hz.

    The 4 knots at 0.5/1/1.5/2 s are linearly densified to 20 steps (the same
    densification `taniteval.closedloop.densify_plan` uses), yaw from the local
    tangent, v from the along-path speed. This is what makes a PLANNED manoeuvre
    comparable with an EXECUTED one under one classifier."""
    knots = np.vstack([[0.0, 0.0], traj])                    # [5,2]
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    tq = np.arange(0, 21) * DT
    x = np.interp(tq, ts, knots[:, 0])
    y = np.interp(tq, ts, knots[:, 1])
    d = np.diff(np.stack([x, y], 1), axis=0, prepend=np.zeros((1, 2)))
    yaw = np.arctan2(d[:, 1], np.maximum(d[:, 0], 1e-6))
    v = np.linalg.norm(d, axis=1) / DT
    v[0] = v0
    return np.stack([x, y, yaw, v], 1)


def run_rollout(transport, renderer, policy, intr, start_frame, n_steps, gt_T,
                gt_ts_us, warm=NEED_FRAMES, save_frames=None, log=None):
    """One closed-loop rollout. Returns a record dict."""
    from collections import deque
    gf = GroundFollower(gt_T)
    gtp = gt_poses_xyv(gt_T)
    frames = deque(maxlen=NEED_FRAMES)
    rec = {"start_frame": start_frame, "n_steps": n_steps, "arm": policy.name,
           "steps": []}

    # ---- force-GT warm-up: the observation window comes from the logged path ----
    for k in range(warm):
        f = min(start_frame + k, len(gt_T) - 1)
        T_rig = gt_T[f]
        img = transport.render(T_rig @ renderer.cam.T_sensor_rig, gt_ts_us[f])
        frames.append(img)
    f0 = min(start_frame + warm - 1, len(gt_T) - 1)
    T_ego = gt_T[f0].copy()
    v = float(gtp[f0, 3])
    t_us = gt_ts_us[f0]

    for k in range(n_steps):
        i_gt = gf.nearest(T_ego[:3, 3])
        nav, navd = nav_from_route(gtp, i_gt)
        t_plan = time.time()
        traj, extra = policy.plan(list(frames), intr, v, nav)
        plan_ms = (time.time() - t_plan) * 1000.0
        steer, accel, v_target, kappa = wp_to_control(traj[LOOKAHEAD_IDX], v)

        # --- record BEFORE stepping (state at decision time) ---------------------
        st = {
            "k": k, "t_us": float(t_us), "i_gt": i_gt, "nav": nav,
            "ego": [float(T_ego[0, 3]), float(T_ego[1, 3]), float(T_ego[2, 3]), _yaw(T_ego)],
            "v": v, "plan": traj.tolist(), "steer": steer, "accel": accel,
            "v_target": v_target, "kappa_plan": kappa, "plan_ms": plan_ms,
            "extra": extra,
        }
        rec["steps"].append(st)
        if save_frames is not None:
            save_frames(k, frames[-1], st)

        # --- step the bicycle one tick ------------------------------------------
        dyaw = v / WHEELBASE * math.tan(steer) * DT
        D = np.eye(4)
        D[:3, :3] = _rz(dyaw)
        D[0, 3] = v * DT
        T_new = T_ego @ D
        T_ego, _ = gf.correct(T_new)
        v = max(0.0, v + accel * DT)
        t_us = t_us + DT * 1e6

        img = transport.render(T_ego @ renderer.cam.T_sensor_rig, t_us)
        frames.append(img)
        if log and k % 20 == 0:
            logger.info("%s  k=%3d  v=%.2f  steer=%+.3f  nav=%s", policy.name, k, v,
                        steer, NAV_NAMES[nav])
    rec["render_ms"] = float(np.mean(transport.render_ms[-(n_steps + warm):]))
    return rec


# ------------------------------------------------------------------------------- #
def build_intr(cam_dict):
    from tanitad.data.calib import FThetaIntrinsics
    return FThetaIntrinsics(poly=tuple(cam_dict["poly"]), cx=cam_dict["cx"],
                            cy=cam_dict["cy"], width=cam_dict["width"],
                            height=cam_dict["height"], per_clip=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["flagship-v1", "refc-base", "refc-xl"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--condition", default="empty", choices=["empty", "objects"])
    ap.add_argument("--layers", default=None)
    ap.add_argument("--addr", default=None, help="gRPC renderer addr; omit for in-process")
    ap.add_argument("--starts", default="0")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-video-frames", action="store_true")
    ap.add_argument("--loader-dir", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from gsplat_renderer import ActorTracks, NuRecGsplatRenderer
    layers = args.layers or ("background,road" if args.condition == "empty"
                             else "background,road")
    sd = Path(args.scene_dir).expanduser()
    r = NuRecGsplatRenderer(sd, layers=[x for x in layers.split(",") if x],
                            loader_dir=args.loader_dir)
    if args.condition == "objects":
        from actor_map import attach_actors_verified
        info = attach_actors_verified(r, sd)
        if info["verdict"] != "ACCEPTED":
            raise SystemExit("actor placement REFUSED by its own falsifier: "
                             + json.dumps({k: v for k, v in info.items() if k != "per_track"}))
        (out / "actor_attach.json").write_text(json.dumps(info, indent=2))
        logger.info("actors: %s", {k: v for k, v in info.items() if k != "per_track"})

    transport = (GrpcTransport(args.addr) if args.addr else InProcTransport(r))
    intr = build_intr(transport.camera())
    logger.info("camera: %dx%d cx=%.1f cy=%.1f poly1=%.1f (transport=%s)",
                intr.width, intr.height, intr.cx, intr.cy, intr.poly[1],
                "grpc" if args.addr else "inproc")

    # The rig logs the camera at 30 Hz (MEASURED: dt = 33_333 us); the control loop
    # and every TanitAD model run at 10 Hz. Sub-sample by 3 so one loop index is one
    # 0.1 s tick — feeding a 30 Hz stack to a 10 Hz-trained encoder would be the same
    # class of train/serve skew as a wrong raster.
    stride = int(round(1e5 / (r.frame_timestamps_us(1)[1] - r.frame_timestamps_us(0)[1])))
    n = r.n_frames() // stride
    gt_T = [r.gt_rig_to_world(f * stride) for f in range(n)]
    gt_ts = [r.frame_timestamps_us(f * stride)[1] for f in range(n)]
    logger.info("GT: %d camera frames -> %d ticks at 10 Hz (stride %d)",
                r.n_frames(), n, stride)

    if args.arm == "flagship-v1":
        pol = FlagshipV1Policy(args.ckpt)
    else:
        pol = RefCPolicy(args.ckpt, preset=args.arm.split("-", 1)[1])

    recs = []
    for s in [int(x) for x in args.starts.split(",")]:
        saver = None
        vdir = None
        if args.save_video_frames:
            vdir = out / f"frames_s{s}"
            vdir.mkdir(exist_ok=True)

            def saver(k, img, st, _d=vdir):
                import cv2
                cv2.imwrite(str(_d / f"{k:05d}.jpg"), img[:, :, ::-1],
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        t0 = time.time()
        rec = run_rollout(transport, r, pol, intr, s, args.steps, gt_T, gt_ts,
                          save_frames=saver, log=True)
        rec["wall_s"] = time.time() - t0
        rec["condition"] = args.condition
        rec["transport"] = "grpc" if args.addr else "inproc"
        rec["f_eff"] = pol.f_eff
        rec["frames_dir"] = str(vdir) if vdir else None
        recs.append(rec)
        logger.info("start %d done: %d steps in %.1fs (%.2f s/step, render %.0f ms)",
                    s, args.steps, rec["wall_s"], rec["wall_s"] / args.steps,
                    rec["render_ms"])

    gt_dump = [{"f": f, "x": float(gt_T[f][0, 3]), "y": float(gt_T[f][1, 3]),
                "z": float(gt_T[f][2, 3]), "yaw": _yaw(gt_T[f]), "ts_us": float(gt_ts[f])}
               for f in range(n)]
    payload = {"arm": pol.name, "ckpt": args.ckpt, "condition": args.condition,
               "scene": sd.name, "layers": layers, "steps": args.steps,
               "f_eff": pol.f_eff, "gt": gt_dump, "rollouts": recs}
    p = out / f"rollouts_{args.arm}_{args.condition}.json"
    p.write_text(json.dumps(payload))
    logger.info("wrote %s (%d rollouts)", p, len(recs))


if __name__ == "__main__":
    main()
