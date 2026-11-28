#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate-0 FREE INFERENCE-TIME FLOOR on REF-C-base's anchored-diffusion planner.

Extends the validated M4 REF-C driver (refc_driver.py) with a training-free
drivable-area floor over the SAME 128 denoised anchor trajectories the model
already returns (out["anchor_traj"], out["anchor_logits"]) — no model change:

  (1) cost-guided SELECTION: pick argmax(conf - lam*offroad_cost - mu*collision_cost)
      over the 128 anchors, instead of the model's argmax(conf).
  (2) road-boundary SAFETY CLAMP: if the selected plan still exits the drivable
      lane union, override to the most-on-road anchor (argmin offroad_cost).

The off-road cost is distance-to-lane-union in the MAP frame (0 inside), where the
lane union is built with AlpaSim's OWN _get_lane_polygon (offroad.py) so it aligns
with the scored metric. FRAME VALIDATED (gate0_cost_validation.json): the driver's
received current.pose is in the map frame under IDENTITY (force-GT poses score
cost~0 on-road; clean off-road points score 90-202 m).

--floor off  reproduces the baseline argmax(conf) (paired control).
--floor on   applies (1)+(2).
Per-drive diagnostics (base vs floor selection + costs + clamp) -> --floor-log JSONL.

Run (alpasim venv + tanitad stack on PYTHONPATH), inside vs_suite_run.sh:
  PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    .venv/bin/python refc_floor_driver.py --port 6789 \
      --ckpt /root/models/refc-base-30k/ckpt.pt --preset base \
      --floor on --lam 5.0 --mu 0.0 --clamp-m 0.75 --floor-log /workspace/gate0_floor_ON.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
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

# AlpaSim scene/map access (drivable-area geometry) — same pod, same USDZs.
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
import shapely
import shapely.ops
import shapely.geometry
import shapely.prepared
from alpasim_runtime.scene_loader import ArtifactSceneProvider

logger = logging.getLogger("refc_floor_driver")

CAM = "camera_front_wide_120fov"
HORIZON_S = (0.5, 1.0, 1.5, 2.0)
NAV = {"follow": 0, "left": 1, "right": 2, "straight": 3}
WINDOW = 8
NEED_FRAMES = WINDOW + 2
SCENESET_GLOB = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/*"


# ============================================================================
# Drivable-area cost map (per scene) — AlpaSim-aligned geometry
# ============================================================================

def _lane_polygon(lane, road_width_m: float = 3.7):
    """Verbatim port of AlpaSim eval/scorers/offroad.py:_get_lane_polygon so the
    floor's drivable surface == the scored metric's drivable surface."""
    if lane.left_edge is not None:
        pts = np.concatenate([lane.left_edge.points[..., :2],
                              np.flip(lane.right_edge.points[..., :2], axis=0)],
                             axis=0)
        poly = shapely.Polygon(pts)
    else:
        poly = shapely.LineString(lane.center.points[..., :2]).buffer(road_width_m / 2)
    if not poly.is_valid:
        poly = shapely.make_valid(poly).buffer(0)
    return poly


class SceneCostMap:
    """Lane-union drivable surface + logged agent trajectories for one scene.
    Off-road cost of a MAP-frame point = 0 inside the union, else distance to it."""

    def __init__(self, ds):
        polys = []
        for lane in ds.map.lanes:
            try:
                polys.append(_lane_polygon(lane))
            except Exception:
                pass
        self.lane_union = shapely.ops.unary_union(polys) if polys else None
        self.prep = shapely.prepared.prep(self.lane_union) if self.lane_union is not None else None
        self.n_lanes = len(polys)
        # agents (map frame): (positions [T,2], timestamps_us [T], is_static)
        self.agents = []
        try:
            objs = ds.traffic_objects
            it = objs.values() if hasattr(objs, "values") else list(objs)
            for o in it:
                try:
                    tr = o.trajectory
                    pos = np.asarray(tr.positions)[:, :2]
                    ts = np.asarray(tr.timestamps_us, dtype=np.float64)
                    self.agents.append((pos, ts, bool(getattr(o, "is_static", False))))
                except Exception:
                    pass
        except Exception:
            pass

    def offroad_cost(self, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
        """wx, wy: [K] map-frame coords -> [K] distance-outside-lane (0 inside)."""
        if self.lane_union is None:
            return np.zeros(len(wx))
        out = np.zeros(len(wx))
        for i in range(len(wx)):
            p = shapely.geometry.Point(float(wx[i]), float(wy[i]))
            if not self.prep.contains(p):
                out[i] = p.distance(self.lane_union)
        return out

    def nudge_world(self, wx: np.ndarray, wy: np.ndarray):
        """Descent direction of the off-road energy E=0.5*dist(p, lane_union)^2:
        -dE/dp = (q - p) where q = nearest point ON the drivable union (points TOWARD
        the road). 0 inside. Returns (ndx, ndy) map-frame. Only off-road points queried."""
        ndx = np.zeros(len(wx)); ndy = np.zeros(len(wy))
        if self.lane_union is None:
            return ndx, ndy
        for i in range(len(wx)):
            p = shapely.geometry.Point(float(wx[i]), float(wy[i]))
            if self.prep.contains(p):
                continue
            q = shapely.ops.nearest_points(self.lane_union, p)[0]
            ndx[i] = q.x - wx[i]; ndy[i] = q.y - wy[i]
        return ndx, ndy

    def agent_positions_at(self, t_us: float) -> np.ndarray:
        """Interpolate every agent's map-frame position at absolute time t_us.
        Returns [M,2]; agents outside their logged time window are clamped to the
        nearest endpoint (conservative — keep clear of the whole corridor)."""
        pts = []
        for pos, ts, _static in self.agents:
            if len(pos) == 0:
                continue
            x = np.interp(t_us, ts, pos[:, 0], left=pos[0, 0], right=pos[-1, 0])
            y = np.interp(t_us, ts, pos[:, 1], left=pos[0, 1], right=pos[-1, 1])
            pts.append((x, y))
        return np.asarray(pts) if pts else np.zeros((0, 2))


# per-scene cost-map registry (built lazily on first sight of a scene_id)
_providers: dict = {}
_costmaps: dict = {}
_costmap_lock = threading.Lock()


def get_costmap(scene_id: str, sceneset_dir: str | None):
    if not scene_id:
        return None
    with _costmap_lock:
        if scene_id in _costmaps:
            return _costmaps[scene_id]
        candidates = [sceneset_dir] if sceneset_dir else []
        candidates += [s for s in sorted(glob.glob(SCENESET_GLOB)) if not s.endswith(".lock")]
        for ss in candidates:
            if not ss:
                continue
            try:
                prov = _providers.get(ss)
                if prov is None:
                    prov = ArtifactSceneProvider.from_path(ss, smooth_trajectories=True)
                    _providers[ss] = prov
                if scene_id in set(prov.scene_ids):
                    cm = SceneCostMap(prov.get_data_source(scene_id))
                    _costmaps[scene_id] = cm
                    logger.info("costmap built scene=%s lanes=%d agents=%d union=%s",
                                scene_id[:20], cm.n_lanes, len(cm.agents),
                                "yes" if cm.lane_union is not None else "NONE")
                    return cm
            except Exception as e:
                logger.warning("costmap load fail on %s: %r", ss, e)
        logger.error("NO sceneset contains scene_id=%s -> floor DISABLED for it", scene_id)
        _costmaps[scene_id] = None
        return None


def _intrinsics_from_camera(cam) -> FThetaIntrinsics | None:
    spec = cam.intrinsics
    if spec.WhichOneof("camera_param") != "ftheta_param":
        return None
    ft = spec.ftheta_param
    fwd = list(ft.angle_to_pixeldist_poly)
    if not fwd or len(fwd) < 2:
        return None
    return FThetaIntrinsics(poly=tuple(fwd), cx=float(ft.principal_point_x),
                            cy=float(ft.principal_point_y), width=int(spec.resolution_w),
                            height=int(spec.resolution_h), per_clip=True)


class RefCPolicy:
    """Loads REF-C; returns the FULL denoised anchor set + confidences (not just
    the argmax) so the floor can re-select."""

    def __init__(self, ckpt: str, preset: str = "base", device: str = "cuda"):
        dev = device if torch.cuda.is_available() else "cpu"
        self.model, self.cfg, self.step = load_frozen(ckpt, preset, None, dev)
        self.device = dev
        self.window = int(self.cfg.window)
        self.n_steps = len(self.cfg.trajectory.horizons)
        self._feff_checked = False
        logger.info("REF-C %s loaded (step=%s, window=%d, anchors=%d) on %s",
                    preset, self.step, self.window, self.cfg.anchors.n_anchors, self.device)

    def _prep_fw(self, raw_frames: list, intr: FThetaIntrinsics):
        vid = torch.from_numpy(np.stack(raw_frames)).permute(0, 3, 1, 2)
        canon = ftheta_crop_resize(vid, intr, 256, center="principal")
        if not self._feff_checked:
            fe = float(ftheta_crop_resize.last_f_eff)
            ok = abs(fe - F_REF) < 8.0
            logger.info("CANON f_eff=%.1f (F_REF=%.1f) %s", fe, F_REF, "OK" if ok else "FAIL")
            self._feff_checked = True
        stacked = stack_frames(canon, 3)
        return stacked[-self.window:][None].to(self.device).float().div_(255.0)

    @torch.no_grad()
    def plan_anchors(self, raw_frames: list, intr: FThetaIntrinsics, v0: float, nav_cmd: int):
        """-> (anchor_traj [N,4,2] rig frame, conf [N]). Model's own decode (steps=2)."""
        fw = self._prep_fw(raw_frames, intr)
        v0t = torch.tensor([v0], dtype=torch.float32, device=self.device)
        navt = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
        out = self.model(fw, nav_cmd=navt, v0=v0t, steps=2)
        return out["anchor_traj"][0].cpu().numpy(), out["anchor_logits"][0].cpu().numpy()

    def _nudge(self, x, cx, cy, c, sn, costmap, eta):
        """One gradient-descent step of the off-road energy on x [1,N,H,2] (rig frame).
        rig -> map (via ego pose) -> map-frame descent (q - p) -> rig -> x += eta*rig_nudge.
        Escapes the fixed anchor set: x is NOT constrained to the vocabulary."""
        xn = x[0].detach().cpu().numpy()                    # [N,H,2]
        N, H = xn.shape[:2]
        dx = xn[..., 0]; dy = xn[..., 1]
        wx = (cx + c * dx - sn * dy).ravel()                # map frame
        wy = (cy + sn * dx + c * dy).ravel()
        ndx_w, ndy_w = costmap.nudge_world(wx, wy)          # map-frame descent [N*H]
        # map -> rig: rig = R^T @ map, R = [[c,-sn],[sn,c]] -> R^T = [[c,sn],[-sn,c]]
        rdx = (c * ndx_w + sn * ndy_w).reshape(N, H)
        rdy = (-sn * ndx_w + c * ndy_w).reshape(N, H)
        xn2 = xn.copy()
        xn2[..., 0] += eta * rdx; xn2[..., 1] += eta * rdy
        return torch.from_numpy(xn2).to(x.dtype).to(x.device)[None]

    @torch.no_grad()
    def plan_nudged(self, raw_frames: list, intr: FThetaIntrinsics, v0: float, nav_cmd: int,
                    cx: float, cy: float, c: float, sn: float, costmap, eta: float, iters: int):
        """Cost-guided DIFFUSION: replicate REF-C's decode (byte-identical to model.forward
        when eta=0) and inject the off-road gradient nudge AFTER each denoise refinement, plus
        `iters` final pure-projection steps -> a trajectory OUTSIDE the fixed anchor set.
        -> (anchor_traj [N,4,2] rig frame, conf [N])."""
        fw = self._prep_fw(raw_frames, intr)
        return self._decode_nudged(fw, v0, nav_cmd, cx, cy, c, sn, costmap, eta, iters)

    @torch.no_grad()
    def _decode_nudged(self, fw, v0: float, nav_cmd: int, cx: float, cy: float, c: float,
                       sn: float, costmap, eta: float, iters: int):
        """The decode replication + nudge (eta=0, costmap=None -> byte-identical to
        model.forward(fw, steps=2))."""
        mdl = self.model; dec = mdl.decoder
        b, w = fw.shape[:2]
        fmap_all, pooled_all = mdl.encoder(fw.reshape(b * w, *fw.shape[2:]))
        pooled_seq = pooled_all.reshape(b, w, -1); pooled = pooled_seq[:, -1]
        fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
        ctx = mdl.strategic(pooled_seq) if mdl.cfg.hierarchy else None
        navt = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
        nav_oh = torch.nn.functional.one_hot(navt, 4).to(pooled.dtype)
        v = (torch.tensor([v0], dtype=torch.float32, device=self.device) / 10.0).reshape(b, 1).to(pooled.dtype)
        m = mdl.measurement(torch.cat([v, nav_oh], dim=-1))
        man_logits = mdl.maneuver_head(pooled)
        kv = dec.feat_proj(fmap.flatten(2).transpose(1, 2))
        cond = dec.cond_proj(m)
        if dec.ctx_to_cond is not None and ctx is not None:
            cond = cond + dec.ctx_to_cond(ctx)
        anchors = dec.anchors.to(fmap.dtype)
        n = anchors.shape[0]
        x0 = anchors[None].expand(b, n, dec.n_steps, 2)
        conf, offset = dec._decode(kv, cond, x0, 0)
        x = anchors[None] + offset
        if dec.maneuver_to_anchor is not None:
            conf = conf + dec.maneuver_to_anchor(torch.log_softmax(man_logits, dim=-1))
        for i in range(2):                                  # the 2 truncated-diffusion steps
            t_idx = min(i + 1, dec.cfg.diffusion_steps)
            _, off = dec._decode(kv, cond, x, t_idx)
            x = x + off
            if eta > 0 and costmap is not None:
                x = self._nudge(x, cx, cy, c, sn, costmap, eta)
        for _ in range(iters):                              # final pure-projection descent
            if eta > 0 and costmap is not None:
                x = self._nudge(x, cx, cy, c, sn, costmap, eta)
        return x[0].cpu().numpy(), conf[0].cpu().numpy()


class RefCFloorDriver(EgodriverServiceServicer):
    def __init__(self, policy: RefCPolicy, mode: str, lam: float, mu: float,
                 clamp_m: float, coll_r: float, eta: float, iters: int,
                 sceneset_dir: str | None, floor_log: str | None):
        self._p = policy
        self._mode = mode                       # off | on (select) | grad (nudge+select)
        self._floor = mode in ("on", "grad")
        self._lam, self._mu, self._clamp_m, self._coll_r = lam, mu, clamp_m, coll_r
        self._eta, self._iters = eta, iters
        self._sceneset_dir = sceneset_dir
        self._floor_log = floor_log
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start_session(self, request, context):
        intr = None
        for cam in request.rollout_spec.vehicle.available_cameras:
            if CAM in cam.logical_id or "front:wide" in cam.logical_id:
                intr = _intrinsics_from_camera(cam)
                if intr is not None:
                    break
        scene_id = ""
        try:
            scene_id = request.debug_info.scene_id
        except Exception:
            pass
        costmap = get_costmap(scene_id, self._sceneset_dir) if self._floor else None
        if intr is None:
            logger.error("start_session %s: NO ftheta intrinsics", request.session_uuid)
        logger.info("start_session %s scene=%s mode=%s costmap=%s", request.session_uuid,
                    scene_id[:20], self._mode,
                    "yes" if costmap is not None else "none")
        with self._lock:
            self._sessions[request.session_uuid] = {
                "intr": intr, "frames": deque(maxlen=24), "speed": 0.0,
                "route": [], "poses": [], "scene_id": scene_id, "costmap": costmap,
                "n_drive": 0}
        return SessionRequestStatus()

    def close_session(self, request, context):
        with self._lock:
            self._sessions.pop(request.session_uuid, None)
        return Empty()

    def get_version(self, request, context):
        tag = f"tanitad-refc-floor-{self._mode}-{self._p.step}"
        return VersionId(version_id=tag, git_hash="tanitad",
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

    def _select(self, s, anchors, conf, cx, cy, yaw, t_now_us):
        """Return (sel_idx, diag). Floor off -> argmax(conf). Floor on -> cost-guided
        selection + safety clamp over the 128 world-frame anchor trajectories."""
        base_idx = int(np.argmax(conf))
        cm = s["costmap"]
        c, sn = np.cos(yaw), np.sin(yaw)
        dx = anchors[..., 0]; dy = anchors[..., 1]            # [N,4]
        wx = cx + c * dx - sn * dy                            # [N,4] map frame
        wy = cy + sn * dx + c * dy
        diag = {"base_idx": base_idx}
        if not self._floor or cm is None or cm.lane_union is None:
            diag["floor_idx"] = base_idx
            return base_idx, diag
        n, h = wx.shape
        off = cm.offroad_cost(wx.reshape(-1), wy.reshape(-1)).reshape(n, h)  # [N,4]
        # emphasise later waypoints (drift accumulates): weights 1,2,3,4 normalised
        w = np.array([1., 2., 3., 4.]); w = w / w.sum()
        off_w = (off * w).sum(axis=1)                         # [N] weighted mean
        off_max = off.max(axis=1)                             # [N]
        coll = np.zeros(n)
        if self._mu > 0 and cm.agents:
            for k, th in enumerate(HORIZON_S):
                ap = cm.agent_positions_at(t_now_us + th * 1e6)   # [M,2]
                if len(ap):
                    d = np.sqrt((wx[:, k][:, None] - ap[:, 0][None]) ** 2
                                + (wy[:, k][:, None] - ap[:, 1][None]) ** 2)  # [N,M]
                    coll += np.maximum(0.0, self._coll_r - d.min(axis=1))
        score = conf - self._lam * off_w - self._mu * coll
        sel = int(np.argmax(score))
        clamp = False
        if off_max[sel] > self._clamp_m:                      # selected plan exits road
            sel = int(np.argmin(off_w))                       # most-on-road anchor
            clamp = (sel != base_idx)
        diag.update({"floor_idx": sel, "base_off": float(off_w[base_idx]),
                     "sel_off": float(off_w[sel]), "base_offmax": float(off_max[base_idx]),
                     "sel_offmax": float(off_max[sel]), "clamp": bool(clamp),
                     "coll_sel": float(coll[sel]) if self._mu > 0 else 0.0})
        return sel, diag

    def drive(self, request, context):
        s = self._sessions.get(request.session_uuid)
        if s is None or not s["poses"]:
            return DriveResponse(trajectory=Trajectory())
        current = s["poses"][-1]
        if s["intr"] is None or len(s["frames"]) < NEED_FRAMES:
            return DriveResponse(trajectory=Trajectory())
        v0 = s["speed"]
        if v0 < 0.1 and len(s["poses"]) >= 2:
            p1, p0 = s["poses"][-1], s["poses"][-2]
            dt = (p1.timestamp_us - p0.timestamp_us) / 1e6
            if dt > 1e-3:
                v0 = ((p1.pose.vec.x - p0.pose.vec.x) ** 2
                      + (p1.pose.vec.y - p0.pose.vec.y) ** 2) ** 0.5 / dt
        nav = self._nav_from_route(s)
        cx, cy = current.pose.vec.x, current.pose.vec.y
        yaw = quat_to_yaw(current.pose.quat)
        if self._mode == "grad" and s["costmap"] is not None:
            c, sn = float(np.cos(yaw)), float(np.sin(yaw))
            anchors, conf = self._p.plan_nudged(list(s["frames"]), s["intr"], v0, nav,
                                                cx, cy, c, sn, s["costmap"], self._eta, self._iters)
        else:
            anchors, conf = self._p.plan_anchors(list(s["frames"]), s["intr"], v0, nav)
        sel, diag = self._select(s, anchors, conf, cx, cy, yaw, float(request.time_now_us))
        diag["mode"] = self._mode
        xy_rig = anchors[sel]                                 # [4,2] rig frame
        d = np.diff(np.vstack([[0.0, 0.0], xy_rig]), axis=0)
        head_rig = np.arctan2(d[:, 1], np.maximum(d[:, 0], 1e-6))
        s["n_drive"] += 1
        if self._floor_log:
            rec = {"session": request.session_uuid, "scene": s["scene_id"],
                   "t": int(request.time_now_us), "n": s["n_drive"],
                   "x": float(cx), "y": float(cy), "yaw": float(yaw), "speed": float(v0)}
            rec.update(diag)
            with open(self._floor_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
        c, sn = np.cos(yaw), np.sin(yaw)
        traj = Trajectory()
        traj.poses.append(current)
        for i in range(len(xy_rig)):
            dxi, dyi = float(xy_rig[i, 0]), float(xy_rig[i, 1])
            lx = cx + c * dxi - sn * dyi
            ly = cy + sn * dxi + c * dyi
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
    ap.add_argument("--floor", default="on", choices=["off", "on", "grad"],
                    help="off=argmax(conf); on=cost-guided selection+clamp; grad=+per-denoise gradient nudge")
    ap.add_argument("--lam", type=float, default=5.0, help="offroad weight (logits/m)")
    ap.add_argument("--mu", type=float, default=0.0, help="collision weight")
    ap.add_argument("--clamp-m", type=float, default=0.75, help="safety-clamp trigger (m outside lane)")
    ap.add_argument("--coll-r", type=float, default=2.5, help="collision radius (m)")
    ap.add_argument("--grad-eta", type=float, default=0.5, help="gradient-nudge step (fraction toward drivable)")
    ap.add_argument("--grad-iters", type=int, default=4, help="extra pure-projection descent iters after denoise")
    ap.add_argument("--sceneset-dir", default=None)
    ap.add_argument("--floor-log", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    policy = RefCPolicy(ckpt=args.ckpt, preset=args.preset, device=args.device)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    add_EgodriverServiceServicer_to_server(
        RefCFloorDriver(policy, mode=args.floor, lam=args.lam, mu=args.mu,
                        clamp_m=args.clamp_m, coll_r=args.coll_r, eta=args.grad_eta,
                        iters=args.grad_iters, sceneset_dir=args.sceneset_dir,
                        floor_log=args.floor_log), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logger.info("RefCFloorDriver serving on %s:%d (floor=%s lam=%.1f mu=%.1f clamp=%.2f eta=%.2f iters=%d ckpt=%s)",
                args.host, args.port, args.floor, args.lam, args.mu, args.clamp_m,
                args.grad_eta, args.grad_iters, args.ckpt)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
