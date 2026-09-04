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

    def render(self, cam_to_world: np.ndarray, ts_us: float,
               cam_to_world_start: np.ndarray | None = None) -> np.ndarray:
        """`cam_to_world` is the shutter-END pose. Supplying `cam_to_world_start`
        switches on ROLLING-shutter rendering (the rig declares
        `shutter_type = ROLLING_TOP_TO_BOTTOM`)."""
        w2n = self.r.rig.world_to_nre
        cam_to_nre = w2n @ cam_to_world
        tau = self.r.tau_of_us(ts_us)
        end = None
        if cam_to_world_start is not None:
            # gsplat's convention: `viewmats` = shutter START, `viewmats_rs` = END
            cam_to_nre, end = w2n @ cam_to_world_start, cam_to_nre
        img, _a, ms = self.r.render(cam_to_nre, tau=tau, actor_time_us=float(ts_us),
                                    cam_to_nre_end=end)
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

    def render(self, cam_to_world: np.ndarray, ts_us: float,
               cam_to_world_start: np.ndarray | None = None) -> np.ndarray:
        from alpasim_grpc.v0 import common_pb2 as cpb
        pb = self.pb

        def _pose(T):
            q = _R_to_quat_np(T[:3, :3])
            return cpb.Pose(vec=cpb.Vec3(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
                            quat=cpb.Quat(w=q[0], x=q[1], y=q[2], z=q[3]))

        # The wire protocol already carries a START/END PosePair — i.e. it was designed
        # for a rolling shutter. Sending the same pose twice (what this did until
        # 2026-08-03) throws that away and asks for a global-shutter frame.
        end = _pose(cam_to_world)
        start = _pose(cam_to_world_start) if cam_to_world_start is not None else end
        req = pb.RGBRenderRequest(
            scene_id=self.scene_id, resolution_h=self._cam["height"],
            resolution_w=self._cam["width"], camera_intrinsics=self.spec,
            frame_start_us=int(ts_us), frame_end_us=int(ts_us),
            sensor_pose=pb.PosePair(start_pose=start, end_pose=end),
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
    # E1: does this arm's INPUT actually carry the nav command's companion bit?
    # The driver reads this before passing `nav_known`, so a bit that cannot be
    # consumed is recorded as DROPPED in the payload instead of vanishing. An
    # arm that says False is not broken — it is un-wired, and the rollout says so.
    consumes_nav_known = False
    #: ⭐ G1 (2026-09-04). WHICH CANONICALISATION THIS ARM WAS TRAINED AT.
    #: `"ftheta256"` is the historical path — `ftheta_crop_resize(..., 256,
    #: center="principal")` onto a 256x256 f-theta raster at `F_REF = 266`, the
    #: geometry flagship-v1 / refc-base / refc-xl were built on. It is UNCHANGED.
    #: `"cyl256x640"` is the 256x640 EQUIDISTANT-AZIMUTH CYLINDRICAL frame the v7
    #: corpus (and therefore refcv3) was built at.
    #: ⛔ An arm must be fed the raster it was trained at; this attribute is the
    #: only place that choice is made, and it is per-ARM, never a CLI flag — a
    #: flag would let the wrong geometry be selected for an arm by accident.
    canon_mode = "ftheta256"
    #: observation-window length in 10 Hz ticks. Read from the loaded config for
    #: refcv3 rather than assumed; `WINDOW` is the historical arms' value.
    window = WINDOW

    def __init__(self, device="cuda"):
        import torch
        from tanitad.data.calib import F_REF, ftheta_crop_resize
        from tanitad.data.comma2k19 import stack_frames
        self.torch, self.F_REF = torch, F_REF
        self._crop, self._stack = ftheta_crop_resize, stack_frames
        self.device = device if torch.cuda.is_available() else "cpu"
        self.f_eff = None
        self.canon_provenance = None

    # ------------------------------------------------------------------ #
    def _canon_cyl(self, frames, intr):
        """G1: the 256x640 cylindrical canonicalisation (refcv3's own).

        ⚠️ **STATE THE PROJECTION BEFORE USING ANY CAMERA FORMULA.** This frame
        is CYLINDRICAL: the column is LINEAR IN AZIMUTH, so the field is
        ``2 * (W/2) / f_ref`` = 2 * 320 / 305.5774907364391 = **2.0944 rad =
        120.0 deg**, which is exactly the rig's own name
        (`camera_front_wide_120fov`). The pinhole formula ``2*atan((W/2)/f)``
        returns a plausible-looking **92.6 deg** and is WRONG here.
        """
        torch = self.torch
        from tanitad.data.calib import (PHYSICALAI_WIDE120_256x640,
                                        cylindrical_rectify)
        frame = PHYSICALAI_WIDE120_256x640
        vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        # `require_per_clip=True` (the default) REFUSES a corpus-median
        # intrinsic. The renderer serves a real per-clip principal point over
        # the wire, so `build_intr` sets `per_clip=True` and this is wired, not
        # defaulted — see calib.py:981 for why that matters (~215 px rig error).
        canon = cylindrical_rectify(vid, intr, frame)          # [T,3,256,640] u8
        if self.f_eff is None:
            self.f_eff = float(cylindrical_rectify.last_f_eff)
            obs = float(cylindrical_rectify.last_observed_frac)
            # `last_f_eff` IS `frame.f_ref` by construction, so this asserts the
            # frame we ran with, not a resampling accident.
            ok_f = abs(self.f_eff - frame.f_ref) < 1e-9
            # ⚠️ The threshold is NOT 1.0. `PHYSICALAI_WIDE120_256x640`'s observed
            # mask is 8.897 % on rig B and 0.0017 % on rig A (calib.py, MEASURED
            # over 3,000 clips), so a rig-B clip legitimately reads ~0.911 — that
            # is what refcv3's own training frames look like. A WRONG crop box
            # reads far lower, which is what this refusal is for.
            ok_o = obs > 0.85
            rig = "B (cy~755)" if intr.cy > 650 else "A (cy~543)"
            logger.info("CANON[cyl] f_eff=%.7f (f_ref=%.7f) observed_frac=%.6f "
                        "hfov=%.3f deg rig=%s %s", self.f_eff, frame.f_ref, obs,
                        math.degrees(2.0 * (frame.width / 2.0) / frame.f_ref), rig,
                        "OK" if (ok_f and ok_o) else "FAIL")
            self.canon_provenance = {
                "mode": "cyl256x640", "f_eff": self.f_eff, "f_ref": frame.f_ref,
                "observed_frac": obs, "projection": frame.projection,
                "height": frame.height, "width": frame.width,
                "hfov_deg_cylindrical": math.degrees(
                    2.0 * (frame.width / 2.0) / frame.f_ref),
                "hfov_deg_pinhole_formula_WRONG_HERE": math.degrees(
                    2.0 * math.atan((frame.width / 2.0) / frame.f_ref)),
                "intr_cx": float(intr.cx), "intr_cy": float(intr.cy),
                "intr_per_clip": bool(intr.per_clip), "rig": rig}
            if not (ok_f and ok_o):
                raise RuntimeError(
                    f"cylindrical canon self-check FAILED (f_eff={self.f_eff}, "
                    f"observed_frac={obs}) — the model would see a raster it was "
                    f"never trained at. Refusing.")
        st = self._stack(canon, STACK)                       # [T-2, 9, 256, 640]
        fw = st[-self.window:][None].to(self.device)
        # ⛔ A 0-dim DEVICE-TENSOR divisor, not the Python scalar 255.0. The run
        # trained with `--u8-batches`, whose ingest is
        # `refc_v3_train.frames_to_device` — and MEASURED 2026-09-02, a Python
        # scalar makes CUDA take the multiply-by-reciprocal path and land 1 ulp
        # off the CPU contract on 126/256 uint8 values. Same map, same bits.
        fw = fw.float().div_(self.torch.tensor(255.0, device=fw.device))
        if tuple(fw.shape[-3:]) != (9, 256, 640) or fw.shape[1] != self.window:
            raise RuntimeError(f"raster assertion failed: {tuple(fw.shape)} != "
                               f"(1,{self.window},9,256,640)")
        return fw

    def canon(self, frames, intr):
        if self.canon_mode == "cyl256x640":
            return self._canon_cyl(frames, intr)
        torch = self.torch
        vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        canon = self._crop(vid, intr, 256, center="principal")
        if self.f_eff is None:
            self.f_eff = float(self._crop.last_f_eff)
            ok = abs(self.f_eff - self.F_REF) < 8.0
            logger.info("CANON f_eff=%.2f (F_REF=%.1f) %s", self.f_eff, self.F_REF,
                        "OK" if ok else "FAIL")
            self.canon_provenance = {"mode": "ftheta256", "f_eff": self.f_eff,
                                     "f_ref": float(self.F_REF),
                                     "projection": "ftheta_crop_resize",
                                     "height": 256, "width": 256}
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

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        # ⛔ `nav_known` is ACCEPTED AND NOT CONSUMED here: the flagship's
        # `StrategicPolicy` FiLM-conditions on `nav_emb(nav_cmd)` alone and has no
        # companion-bit seam (`consumes_nav_known` is False, so the driver never
        # passes one). Named, not hidden — see the E1 note in `refs/refc.py`.
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
        self.consumes_nav_known = bool(getattr(self.cfg, "nav_known_channel", False))
        logger.info("REF-C %s loaded step=%s anchors=%d nav_known_channel=%s", preset,
                    self.step, self.cfg.anchors.n_anchors, self.consumes_nav_known)

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        torch = self.torch
        with torch.no_grad():
            fw = self.canon(frames, intr)
            v0t = torch.tensor([float(v0)], dtype=torch.float32, device=self.device)
            navt = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
            # E1: only fed when the loaded ckpt was BUILT with the channel. The
            # model raises if it is handed a bit it cannot read, so this branch is
            # the whole guard — there is no silent-drop path.
            nk = (torch.tensor([float(nav_known)], dtype=torch.float32,
                               device=self.device)
                  if (self.consumes_nav_known and nav_known is not None) else None)
            out = self.model(fw, nav_cmd=navt, v0=v0t, steps=2, nav_known=nk)
            traj = out["traj"][0].float().cpu().numpy()
        extra = {}
        for k, v in out.items():
            if hasattr(v, "shape") and v.ndim == 2 and v.shape[0] == 1 and v.shape[1] <= 16:
                extra[k] = v[0].float().cpu().numpy().tolist()
        return traj.astype(np.float64), extra


class RefCV3Policy(_BasePolicy):
    """G2 (2026-09-04). refcv3 in the closed loop.

    ⭐ **The loader is IMPORTED, never forked.** `taniteval/tools/refcv3_arm.py`
    already owns a strict, cross-checked `load_model` that rebuilds the config
    through the trainer's OWN `build_parser` + `_pin_trainer_cfg` on the recorded
    argv, cross-checks arm / image_hw / tac_vocab_version / horizons /
    param_breakdown against `config.json`, and REFUSES a non-strict state-dict
    load ("fix the config, never the weights"). Re-implementing any of that here
    would be a second, drifting copy of the one contract that matters.
    """
    name = "refcv3"
    canon_mode = "cyl256x640"
    consumes_nav_known = False       # refc.py:2042-2045 RAISES if fed with the
                                     # gate off, and nothing in v3 turns it on.

    def __init__(self, ckpt, device="cuda", config=None):
        super().__init__(device)
        import refcv3_arm
        self.model, self.cfg, self.targs, self.prov = refcv3_arm.load_model(
            ckpt, config, self.device, allow_nonstrict=False)
        self.step = self.prov.get("step")
        self.horizons = [int(h) for h in self.prov["horizons"]]
        self.window = int(self.prov["window"])
        self.decoder_steps = int(self.prov["decoder_steps"])
        # ---- G3: ASSERT the grid, never assume it -------------------------- #
        # The harness steers to `traj[LOOKAHEAD_IDX]` and scores at
        # HORIZON_S = (0.5, 1.0, 1.5, 2.0) s. A checkpoint built on a different
        # horizon grid would silently steer to the wrong lookahead.
        if tuple(self.horizons[:len(WP_STEPS)]) != tuple(WP_STEPS):
            raise RuntimeError(
                f"⛔ HORIZON GRID MISMATCH: the checkpoint's first "
                f"{len(WP_STEPS)} horizons are {self.horizons[:len(WP_STEPS)]} "
                f"but the harness scores/steers on WP_STEPS={list(WP_STEPS)} "
                f"(= {list(HORIZON_S)} s at {DT} s/tick). Refusing — "
                f"LOOKAHEAD_IDX={LOOKAHEAD_IDX} would be the wrong waypoint.")
        # image geometry must be the one `_canon_cyl` produces
        ihw = list(self.cfg.core.encoder.image_hw())
        if ihw != [256, 640]:
            raise RuntimeError(f"⛔ checkpoint image_hw={ihw}, but this policy "
                               f"canonicalises to [256, 640]. Refusing.")
        if int(self.cfg.core.encoder.in_channels) != 3 * STACK:
            raise RuntimeError(f"⛔ encoder in_channels="
                               f"{self.cfg.core.encoder.in_channels} != 3*STACK="
                               f"{3 * STACK}; the D-015 frame stack disagrees.")
        logger.info("refcv3 loaded step=%s window=%d horizons=%s decoder=%s/%d "
                    "image_hw=%s hier=%s anchors=%d params=%s",
                    self.step, self.window, self.horizons,
                    self.prov["decoder_mode"], self.decoder_steps, ihw,
                    self.prov["hier"], self.prov["n_anchors"],
                    self.prov["param_breakdown"].get("total"))
        logger.info("refcv3 strict load: missing=%s unexpected=%s",
                    self.prov["state_dict_load"]["missing_keys"],
                    self.prov["state_dict_load"]["unexpected_keys"])

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        torch = self.torch
        with torch.no_grad():
            fw = self.canon(frames, intr)                    # [1,W,9,256,640]
            v0t = torch.tensor([float(v0)], dtype=torch.float32, device=self.device)
            navt = torch.tensor([nav_cmd], dtype=torch.long, device=self.device)
            # `lan` and `ego_state` are deliberately NOT passed: the run's config
            # carries neither (`graft_lan`/`goal_str` absent from argv,
            # `ego_state_inject` off), and `forward` RAISES on an ego block it
            # would have to drop. `steps` is the run's own decoder step count.
            out = self.model(fw, nav_cmd=navt, v0=v0t, steps=self.decoder_steps)
            # ⭐ THE DEPLOYED SELECTION IS out["traj"] AND NOTHING ELSE
            # (refc_v3.py:520-525 on hier; refc.py:1531-1534 on flat).
            full = out["traj"][0].float().cpu().numpy()      # [S, 2]
            traj = full[:len(WP_STEPS)]                      # index-select, never interp
        extra = {}
        for k, v in out.items():
            if hasattr(v, "shape") and v.ndim == 2 and v.shape[0] == 1 and v.shape[1] <= 16:
                extra[k] = v[0].float().cpu().numpy().tolist()
        # the 3-6 s tail this arm can serve but the harness does not score, banked
        # verbatim so a longer-horizon rescore never needs the GPU again
        extra["traj_full_6s"] = full.tolist()
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
    from refb_labels import nav_command_v21, nav_input_v22
    t = torch.from_numpy(gtp).float()
    out = {}
    try:
        nav, valid = nav_command_v21(t, int(i))
        out["nav_canonical"], out["nav_valid"] = int(nav), bool(valid)
    except Exception as e:                                   # noqa: BLE001
        out["nav_canonical"], out["nav_valid"], out["nav_err"] = 0, False, repr(e)[:80]
    # E1: the COMPANION BIT. `nav_command_v21` collapses ROUTE_UNKNOWN and
    # ROUTE_STRAIGHT onto the same NAV_FOLLOW token, so `nav_valid` is NOT the
    # same statement — a window can be `valid` and still be a road-following
    # judgement rather than a route one. `nav_known` is 0.0 exactly when the
    # command is the UNKNOWN sentinel. It is RECORDED on every tick whether or
    # not any arm consumes it: measuring how often the input is a confession is
    # the point, and it costs one label call.
    try:
        _nv, known = nav_input_v22(t, int(i))
        out["nav_known"] = float(known)
        out["nav_known_agrees_canonical"] = bool(_nv == out["nav_canonical"])
    except Exception as e:                                   # noqa: BLE001
        out["nav_known"], out["nav_known_err"] = None, repr(e)[:80]
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


# ------------------------------------------------------------------------------- #
# TRIVIAL FLOOR POLICIES — the `ha0`-equivalent bar the closed loop never had       #
# ------------------------------------------------------------------------------- #
#: the three trivial arms. Names are prefixed `cl_` so a closed-loop floor can never
#: be confused with the open-loop suite's `ha`/`ha0`, which are computed by a
#: DIFFERENT instrument on a DIFFERENT tier and are not comparable as levels.
FLOOR_ARMS = ("cl_ha0", "cl_ha", "cl_ha0_ext")

#: below this speed the unicycle cannot turn at all (`yaw_rate = v * kappa`, so a
#: stopped vehicle has no yaw rate — `kinematic.py:228-231`). A constant-yaw-rate
#: extrapolation therefore has to DROP the yaw rate rather than divide by ~0, and
#: this is where that happens. Stated, not silently clamped inside a formula.
FLOOR_V_EPS = 1e-3


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class KinematicFloorPolicy:
    """⭐ THE TRIVIAL CONTROL EVERY CLOSED-LOOP NUMBER IN THIS PROGRAMME LACKED.

    **Why this class exists.** In OPEN loop, MEASURED 2026-09-04 on 4,823 windows /
    141 episodes, the trivial hold-action control (`ha`, 0.2996 m) **beats** the
    deployed model (`os`, 0.4419 m) by +0.1423 m [+0.1187, +0.1658] — and oracle
    selection (0.0751) plus oracle nav (0.0239) together, 0.099 m, do not close that
    gap. If a trivial control beats the model where prediction is *easiest*, then
    **no closed-loop level is interpretable without the same control**, and until
    2026-09-04 the closed-loop harness had none: `refcv3 2.8755` vs `refc-base
    2.6554` is a difference between two arms with no bar under either.

    **It is a POLICY, not a second harness.** It plugs into `run_rollout` through
    the same `plan()` contract as `RefCV3Policy`, so the floor is produced by the
    *identical* render → canonicalise-slot → plan → `wp_to_control` → kinematic
    bicycle → `GroundFollower` → `cl_metrics.py` path as every model arm. The
    frames are still rendered (`window = 8`, so `f0 = start + 9`, bit-identically to
    every arm this harness has run) — they are simply not read. Anything else would
    make the floor an incomparable number, which is exactly the disease.

    ⛔ **INPUTS: MEASURED STATE AT t0 ONLY.** `a0` and `omega0` come from the logged
    rig poses at indices `f0-2, f0-1, f0` through BACKWARD differences, so every
    index used is `<= t0`. Nothing after the window origin enters any floor arm.
    This is the PI's 2026-09-02 ruling (velocity at cycle time is a legal initial
    state) applied exactly as `refcv3_arm.hold_controls` applies it open-loop.
    ⚠️ One inherited asymmetry, named rather than hidden: the harness's own initial
    speed `v` is `gt_poses_xyv`'s FORWARD difference (it uses pose `f0+1`). That is
    the harness's convention, fed identically to every model arm, and the floor
    consumes it through `plan(v0=...)` unchanged — changing it for the floor alone
    would have broken the pairing it exists to support.

    **The three arms.**

    * ``cl_ha0``     — a = 0, kappa = 0. Constant velocity, straight, forever. The
      direct analogue of the open-loop `ha0` and the arm every margin is taken over.
    * ``cl_ha``      — hold the action that CLOSES at t0: constant `a0` and constant
      CURVATURE `kappa0` (a fixed steering angle). The closed-loop analogue of the
      open-loop `ha`.
    * ``cl_ha0_ext`` — constant `a0` and constant YAW RATE `omega0` (CTRA). The
      strongest trivial baseline, and the one a model with ego inputs could echo.
      It differs from `cl_ha` exactly when the speed changes: `cl_ha` holds
      `kappa`, so its yaw rate `v*kappa` tracks the speed; `cl_ha0_ext` holds the
      yaw rate, so its curvature does.

    ⚠️ **THE CONTROLLER IS NOT A PASS-THROUGH, AND THAT IS DELIBERATE.**
    `wp_to_control` reads ONLY `traj[LOOKAHEAD_IDX]` (the 0.5 s waypoint) and sets
    `v_target = x / (L*dt)`, which is the AVERAGE speed over the lookahead, not the
    terminal one — and `rollout_unicycle` advances position on the speed at the
    START of each step. So under a constant intended acceleration `a`,
    `x_L = L*v*dt + a*dt^2*L(L-1)/2`, hence `v_target = v + a*dt*(L-1)/2` and

        accel_executed = a * dt*(L-1)/2 / speed_tc = a * 0.1*2/0.5 = **0.4 * a**

    — the harness executes **40 %** of any constant intended acceleration.
    (MEASURED, `test_closedloop_floor.py`; the continuous-time answer is `0.5a` and
    writing that from the integral was this docstring's first draft.) Every model
    arm is distorted by the same controller in the same way, which is why the floor
    must suffer it too rather than inject controls behind it. Reported, never
    corrected — correcting it would make this harness incomparable with every
    published TanitAD closed-loop number.
    """

    #: 8 ticks — the observation window of EVERY arm this harness has run. It is
    #: what fixes `f0 = start + WINDOW + STACK - 2 = start + 9`, i.e. WHERE the
    #: rollout begins. A floor arm declaring a different window would start from a
    #: different pose and could not be paired with anything. Stated, not defaulted.
    window = WINDOW
    consumes_nav_known = False
    #: no canonicalisation happens at all — see `canon_provenance` below.
    canon_mode = None

    def __init__(self, mode: str):
        if mode not in FLOOR_ARMS:
            raise ValueError(f"unknown floor arm {mode!r}; expected one of {FLOOR_ARMS}")
        self.name = mode
        self.mode = mode
        # `f_eff` is the raster self-check every MODEL arm must pass. A floor arm
        # never looks at a pixel, so there is nothing to check and `None` is the
        # honest value — not a copied constant that would imply a check happened.
        self.f_eff = None
        self.canon_provenance = {
            "mode": "none",
            "why": ("a trivial kinematic floor consumes no image. The frames are "
                    "still RENDERED (identical rollout/render path, identical "
                    "`f0`), they are simply never canonicalised or read."),
            "reads_pixels": False}
        self.n_plan = int(WP_STEPS[-1])
        self.t0 = None

    # ------------------------------------------------------------------ #
    def set_t0(self, gt_T, f0):
        """Measure the t0 state from LOGGED poses at indices ``<= f0`` ONLY.

        Called once per rollout by `run_rollout`, immediately after `f0` is fixed.
        `a0` is a second difference of position and therefore needs `f0-2`; that is
        asserted rather than silently clamped, because a floor arm that quietly
        started from a different state than it claims is worse than no floor.
        """
        f0 = int(f0)
        if f0 < 2:
            raise RuntimeError(
                f"floor arm {self.name!r} needs f0 >= 2 (a0 is a second difference "
                f"of position: it reads poses f0-2, f0-1, f0) but f0={f0}. Refusing "
                f"— the alternative is a fabricated t0 state.")
        p = [np.asarray(gt_T[i][:2, 3], np.float64) for i in (f0 - 2, f0 - 1, f0)]
        yaw = [_yaw(gt_T[i]) for i in (f0 - 1, f0)]
        # BACKWARD differences: v_b[i] uses poses i-1 and i, so `a0` reads f0-2..f0
        # and `omega0` reads f0-1..f0. Every index is <= t0.
        vb_prev = float(np.linalg.norm(p[1] - p[0]) / DT)
        vb_now = float(np.linalg.norm(p[2] - p[1]) / DT)
        a0 = (vb_now - vb_prev) / DT
        omega0 = float(_wrap(yaw[1] - yaw[0]) / DT)
        kappa0 = omega0 / max(vb_now, FLOOR_V_EPS)
        self.t0 = {
            "f0": f0, "a0_mps2": a0, "omega0_rads": omega0, "kappa0_1pm": kappa0,
            "v_backward_at_t0_ms": vb_now, "v_backward_at_t0m1_ms": vb_prev,
            "pose_indices_read": [f0 - 2, f0 - 1, f0],
            "differencing": "backward — every index used is <= t0, no future pose",
        }
        logger.info("%s t0: f0=%d a0=%+.4f m/s^2 omega0=%+.5f rad/s kappa0=%+.6f 1/m "
                    "(v_bwd=%.4f m/s)", self.name, f0, a0, omega0, kappa0, vb_now)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _path(accel, kappa, v, n):
        """``[n, 2]`` ego-frame path under ``(accel, kappa[k])`` from speed ``v``.

        ⭐ The integrator is the PROGRAMME'S OWN — `kinematic.rollout_unicycle`,
        the same one `refav1_arm.paths_from_controls` uses to build the open-loop
        `ha`/`ha0` floors — called at **float64** through `state0`'s dtype.
        (`refa_v1_plan.unicycle_paths` builds a float32 `state0` internally, so it
        is bypassed here and only here; `test_closedloop_floor.py` pins the two
        against each other.)

        ⭐ **THE ZERO-CONTROL BRANCH IS EXACT ON PURPOSE.** For `a = 0, kappa = 0`
        the recurrence reduces to `x_k = (k+1) v dt, y_k = 0`, and evaluating that
        closed form is what makes `cl_ha0`'s control read **exactly** 0.0. Summing
        `fl(v*0.1)` five times instead misses `v/2` by ~6e-16, `wp_to_control`
        turns that into `accel = 2*(2*x5 - v)` ~ 2e-15 m/s^2, and a floor whose
        defining property ("it does nothing") holds only to 1e-15 cannot serve as a
        control that must read a known value. The branch is on the CONTROLS, never
        on the arm name, so `cl_ha` and `cl_ha0_ext` collapse onto it
        BIT-IDENTICALLY whenever their measured `a0` and `omega0` are both zero —
        which is the degeneracy control.
        """
        kap = np.asarray(kappa, np.float64)
        if float(accel) == 0.0 and not np.any(kap):
            xs = float(v) * (np.arange(1, n + 1, dtype=np.float64) * DT)
            return np.stack([xs, np.zeros(n, np.float64)], 1)
        import torch
        from tanitad.models.kinematic import rollout_unicycle
        ctrl = torch.from_numpy(np.stack(
            [np.full(n, float(accel), np.float64), kap], 1))[None]
        st0 = torch.zeros(1, 4, dtype=torch.float64)
        st0[0, 3] = float(v)
        return rollout_unicycle(st0, ctrl, dt=DT)[0, :, :2].numpy()

    # ------------------------------------------------------------------ #
    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        """`frames`, `intr`, `nav_cmd` and `nav_known` are ACCEPTED AND NOT READ.

        Named rather than dropped from the signature: the floor must satisfy the
        same contract as a model policy, and the fact that it ignores the route
        command is a PROPERTY OF THE ARM worth being able to point at — a trivial
        control that consulted the nav would not be trivial.
        """
        if self.t0 is None:
            raise RuntimeError(
                f"{self.name}.set_t0() was never called. The floor's t0 state would "
                f"be invented. Refusing — see run_rollout's set_t0 hook.")
        v = float(v0)
        n = self.n_plan
        if self.mode == "cl_ha0":
            a, kap = 0.0, np.zeros(n, np.float64)
        elif self.mode == "cl_ha":
            a = float(self.t0["a0_mps2"])
            kap = np.full(n, float(self.t0["kappa0_1pm"]), np.float64)
        else:                                            # cl_ha0_ext — CTRA
            a = float(self.t0["a0_mps2"])
            w = float(self.t0["omega0_rads"])
            # The speed the integrator will USE at the start of each step, in closed
            # form under a constant `a` with the integrator's own clamp-at-zero.
            # `kappa_k = omega0 / v_k` is then exactly what holds the yaw rate
            # constant, because the model DEFINES `yaw_rate = v * kappa`.
            vk = np.empty(n, np.float64)
            s = v
            for k in range(n):
                vk[k] = s
                s = max(0.0, s + a * DT)
            kap = np.where(vk > FLOOR_V_EPS, w / np.maximum(vk, FLOOR_V_EPS), 0.0)
        path = self._path(a, kap, v, n)
        traj = np.ascontiguousarray(path[[h - 1 for h in WP_STEPS]], np.float64)
        # ⛔ Every value here is a SCALAR. `cl_metrics.resolve_stamp` RAISES on an
        # unrecognised list of width 4 or 5 in `extra` (it is how a renamed head was
        # caught), so a floor arm must never bank a bare 4- or 5-vector.
        extra = {
            "floor_mode": self.mode,
            "floor_a0_mps2": float(self.t0["a0_mps2"]),
            "floor_omega0_rads": float(self.t0["omega0_rads"]),
            "floor_kappa0_1pm": float(self.t0["kappa0_1pm"]),
            "floor_v_plan_ms": v,
            "floor_accel_cmd_mps2": float(a),
            "floor_kappa_cmd_head_1pm": float(kap[0]),
            "floor_reads_pixels": 0.0,
            "floor_reads_nav": 0.0,
        }
        return traj, extra


def run_rollout(transport, renderer, policy, intr, start_frame, n_steps, gt_T,
                gt_ts_us, warm=None, save_frames=None, log=None, leadgeom=None,
                shutter_s=0.0, gt_stride=0):
    """One closed-loop rollout. Returns a record dict."""
    from collections import deque
    gf = GroundFollower(gt_T)
    gtp = gt_poses_xyv(gt_T)
    # The native-frame need is a property of THE ARM's observation window, read
    # from its loaded config — not a module constant. For every historical arm
    # `policy.window` IS `WINDOW`, so this is `NEED_FRAMES` unchanged.
    need = int(getattr(policy, "window", WINDOW)) + STACK - 1
    if warm is None:
        warm = need
    frames = deque(maxlen=need)
    rec = {"start_frame": start_frame, "n_steps": n_steps, "arm": policy.name,
           "steps": []}

    # Rolling shutter: the rig declares ROLLING_TOP_TO_BOTTOM with a 30.559 ms readout,
    # over which the ego moves up to 0.63 m. `shutter_s` > 0 renders the frame from the
    # pose it had `shutter_s` earlier at the top of the image to the current pose at the
    # bottom. The roll-back uses the SAME bicycle model this loop steps with, so the two
    # can never disagree about the vehicle's motion.
    Ts_cam = renderer.cam.T_sensor_rig

    def _rollback(T, v_now, steer_now, dt):
        if dt <= 0:
            return None
        D = np.eye(4)
        D[:3, :3] = _rz(v_now / WHEELBASE * math.tan(steer_now) * dt)
        D[0, 3] = v_now * dt
        return T @ np.linalg.inv(D)

    # ---- force-GT warm-up: the observation window comes from the logged path ----
    for k in range(warm):
        f = min(start_frame + k, len(gt_T) - 1)
        T_rig = gt_T[f]
        # on the logged path the true shutter-start rig pose is in the file
        T_start = (renderer.rig.T_rig_world(renderer.cam_name, f * gt_stride, shutter=0)
                   if (shutter_s > 0 and gt_stride) else None)
        img = transport.render(T_rig @ Ts_cam, gt_ts_us[f],
                               cam_to_world_start=(T_start @ Ts_cam
                                                   if T_start is not None else None))
        frames.append(img)
    f0 = min(start_frame + warm - 1, len(gt_T) - 1)
    T_ego = gt_T[f0].copy()
    v = float(gtp[f0, 3])
    t_us = gt_ts_us[f0]

    # ⭐ FLOOR ARMS ONLY: measure the trivial control's t0 state from the LOGGED
    # poses at indices <= f0, now that f0 is fixed. Model policies carry no
    # `set_t0`, so this hook is INERT for them and every historical rollout is
    # byte-identical with or without it (asserted by the patch-neutrality control).
    if hasattr(policy, "set_t0"):
        policy.set_t0(gt_T, f0)
        rec["floor_t0"] = dict(policy.t0)

    # ⚠️ Anchor any lateral (cut-in) profile to THIS ROLLOUT's first decision, not to
    # the clip start. Anchored to the clip, the 2.0-3.5 s ramp would have finished long
    # before a rollout that starts at tick 120 ever looked — 8 of the 9 clusters would
    # have silently been a plain lead vehicle while the panel called them cut-ins.
    act = getattr(renderer, "_actor", None) or {}
    if hasattr(act.get("tracks"), "t0_us"):
        act["tracks"].t0_us = float(t_us)
    if leadgeom is not None:
        leadgeom.set_t0(float(t_us))

    for k in range(n_steps):
        i_gt = gf.nearest(T_ego[:3, 3])
        nav, navd = nav_from_route(gtp, i_gt)
        nk = navd.get("nav_known")
        t_plan = time.time()
        traj, extra = policy.plan(
            list(frames), intr, v, nav,
            nav_known=(nk if getattr(policy, "consumes_nav_known", False) else None))
        plan_ms = (time.time() - t_plan) * 1000.0
        steer, accel, v_target, kappa = wp_to_control(traj[LOOKAHEAD_IDX], v)

        # --- record BEFORE stepping (state at decision time) ---------------------
        st = {
            "k": k, "t_us": float(t_us), "i_gt": i_gt, "nav": nav,
            # E1 provenance: WAS the bit fed, not merely computed. A rollout whose
            # `nav_known_fed` is False is one where the model could not tell an
            # UNKNOWN sentinel from a real "go straight".
            "nav_known": nk,
            "nav_known_fed": bool(getattr(policy, "consumes_nav_known", False)
                                  and nk is not None),
            "ego": [float(T_ego[0, 3]), float(T_ego[1, 3]), float(T_ego[2, 3]), _yaw(T_ego)],
            "v": v, "plan": traj.tolist(), "steer": steer, "accel": accel,
            "v_target": v_target, "kappa_plan": kappa, "plan_ms": plan_ms,
            "extra": extra,
        }
        # Lead geometry for EVERY synthetic condition, on EVERY run — including
        # `empty`, where it is the matched counterfactual the pairing needs.
        if leadgeom is not None:
            st["lead"] = leadgeom.at(T_ego, t_us, v)
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

        T_start = _rollback(T_ego, v, steer, shutter_s)
        img = transport.render(T_ego @ Ts_cam, t_us,
                               cam_to_world_start=(T_start @ Ts_cam
                                                   if T_start is not None else None))
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
                    choices=["flagship-v1", "refc-base", "refc-xl", "refcv3"]
                            + list(FLOOR_ARMS),
                    help="a model arm, or one of the TRIVIAL FLOOR arms "
                         f"{FLOOR_ARMS} — the `ha0`-equivalent bar every "
                         "closed-loop number needs and none had before "
                         "2026-09-04. A floor arm takes no --ckpt.")
    ap.add_argument("--ckpt", default=None,
                    help="required for a model arm; REFUSED for a floor arm, which "
                         "has no weights and must not appear to have any.")
    ap.add_argument("--ckpt-config", default=None,
                    help="refcv3 only: the run's config.json. Defaults to the "
                         "one beside --ckpt, which is where the trainer writes it.")
    ap.add_argument("--condition", default="empty",
                    choices=["empty", "objects", "lead25", "lead15", "lead8",
                             "cutin", "behind"],
                    help="empty/objects use the scene's own annotation; the rest are "
                         "CONSTRUCTED lead vehicles (see synth_actor.py) because two "
                         "probes agree the scene has no close-following geometry")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--addr", default=None, help="gRPC renderer addr; omit for in-process")
    ap.add_argument("--starts", default="0")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-video-frames", action="store_true")
    ap.add_argument("--loader-dir", default=None)
    # ---- render quality (2026-08-03; every default keeps the previous behaviour) ----
    ap.add_argument("--sky-gain", type=float, default=0.0,
                    help="gated sky-env-map gain; 0 = off. MEASURED on scene 00040136: "
                         "gain 0.3 cuts full-frame MAE 0.1007 -> 0.0811 and raises "
                         "grad-NCC 0.2773 -> 0.2894. gain 1.0 over-brightens (MAE 0.1431).")
    ap.add_argument("--sky-lo-deg", type=float, default=0.0)
    ap.add_argument("--sky-hi-deg", type=float, default=6.0)
    ap.add_argument("--rolling-shutter", action="store_true",
                    help="render the declared ROLLING_TOP_TO_BOTTOM shutter. Biggest "
                         "measured quality lever (grad-NCC 0.2774 -> 0.3170, MAE -9.4%%) "
                         "and by far the most expensive (23 ms -> ~3700 ms/frame).")
    ap.add_argument("--cull-scale-quantile", type=float, default=None,
                    help="drop static splats above this quantile of max-axis scale")
    ap.add_argument("--all-dynamic-layers", action="store_true",
                    help="with --condition objects, also render dynamic_deformables "
                         "(the scene ships 2 tracks / 1039 gaussians there)")
    args = ap.parse_args()
    # ⛔ An arm/ckpt mismatch must STOP, never degrade. A floor arm silently
    # carrying a --ckpt would bank a payload whose `ckpt` field names weights that
    # were never loaded — the exact shape of provenance error this programme's
    # registry rule exists to prevent.
    if args.arm in FLOOR_ARMS:
        if args.ckpt:
            raise SystemExit(
                f"--ckpt={args.ckpt!r} was given for the TRIVIAL FLOOR arm "
                f"{args.arm!r}, which loads no weights. Refusing: the payload would "
                f"claim a checkpoint it never read.")
    elif not args.ckpt:
        raise SystemExit(f"--ckpt is required for the model arm {args.arm!r}.")

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
    if args.cull_scale_quantile:
        logger.info("scale cull: %s", r.cull_by_scale(args.cull_scale_quantile))
    if args.sky_gain > 0:
        sky = r.attach_sky(lo_deg=args.sky_lo_deg, hi_deg=args.sky_hi_deg,
                           gain=args.sky_gain)
        if sky is None:
            raise SystemExit("--sky-gain given but the scene ships no sky-env-map")
        logger.info("gated sky ON: gain=%.2f ramp %.1f-%.1f deg above horizon",
                    args.sky_gain, args.sky_lo_deg, args.sky_hi_deg)
    attach_info = None
    if args.condition == "objects":
        if args.all_dynamic_layers:
            from actor_map import attach_all_dynamic_layers
            info = attach_all_dynamic_layers(r, sd)
        else:
            from actor_map import attach_actors_verified
            info = attach_actors_verified(r, sd)
        if info["verdict"] != "ACCEPTED":
            raise SystemExit("actor placement REFUSED by its own falsifier: "
                             + json.dumps({k: v for k, v in info.items() if k != "per_track"}))
        (out / "actor_attach.json").write_text(json.dumps(info, indent=2))
        logger.info("actors: %s", {k: v for k, v in info.items() if k != "per_track"})
    elif args.condition != "empty":
        from synth_actor import SYNTH_CONDITIONS, attach_synth_lead
        assert args.condition in SYNTH_CONDITIONS, args.condition
        attach_info = attach_synth_lead(r, args.condition)
        (out / "synth_attach.json").write_text(json.dumps(attach_info, indent=2))
        logger.info("CONSTRUCTED lead attached: %s", attach_info)

    transport = (GrpcTransport(args.addr) if args.addr else InProcTransport(r))
    intr = build_intr(transport.camera())
    logger.info("camera: %dx%d cx=%.1f cy=%.1f poly1=%.1f (transport=%s)",
                intr.width, intr.height, intr.cx, intr.cy, intr.poly[1],
                "grpc" if args.addr else "inproc")

    # The rig logs the camera at 30 Hz (MEASURED: dt = 33_333 us); the control loop
    # and every TanitAD model run at 10 Hz. Sub-sample by 3 so one loop index is one
    # 0.1 s tick — feeding a 30 Hz stack to a 10 Hz-trained encoder would be the same
    # class of train/serve skew as a wrong raster.
    # shutter READOUT duration, read from the rig's own per-frame timestamps rather
    # than assumed (MEASURED: 30_559 us, constant across frames on this clip)
    _t0, _t1 = r.frame_timestamps_us(0)
    shutter_s = ((_t1 - _t0) / 1e6) if args.rolling_shutter else 0.0
    if args.rolling_shutter:
        logger.info("ROLLING SHUTTER ON: type=%s readout=%.1f ms — expect ~3.7 s/frame "
                    "(vs ~23 ms global). Simulated time is unchanged; only wall clock.",
                    r.rolling_shutter_type, shutter_s * 1e3)
    stride = int(round(1e5 / (r.frame_timestamps_us(1)[1] - r.frame_timestamps_us(0)[1])))
    n = r.n_frames() // stride
    gt_T = [r.gt_rig_to_world(f * stride) for f in range(n)]
    gt_ts = [r.frame_timestamps_us(f * stride)[1] for f in range(n)]
    logger.info("GT: %d camera frames -> %d ticks at 10 Hz (stride %d)",
                r.n_frames(), n, stride)

    # ⚠️ half_len is derived INSIDE LeadGeometry from the renderer, never from the
    # attach info — otherwise the `empty` control (which attaches nothing) would use a
    # different half-length from the lead conditions and the counterfactual headway
    # would carry a constant bias. See synth_actor.LeadGeometry.
    from synth_actor import LeadGeometry
    leadgeom = LeadGeometry(r)
    logger.info("lead geometry probe active (half_len=%.3f m, %d conditions)",
                leadgeom.half_len, len(leadgeom.tracks))

    if args.arm in FLOOR_ARMS:
        pol = KinematicFloorPolicy(args.arm)
    elif args.arm == "flagship-v1":
        pol = FlagshipV1Policy(args.ckpt)
    elif args.arm == "refcv3":
        pol = RefCV3Policy(args.ckpt, config=args.ckpt_config)
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
                          save_frames=saver, log=True, leadgeom=leadgeom,
                          shutter_s=shutter_s, gt_stride=stride)
        rec["wall_s"] = time.time() - t0
        rec["condition"] = args.condition
        rec["render_quality"] = {
            "layers": layers, "all_dynamic_layers": bool(args.all_dynamic_layers),
            "sky_gain": args.sky_gain, "sky_ramp_deg": [args.sky_lo_deg, args.sky_hi_deg],
            "rolling_shutter": bool(args.rolling_shutter),
            "shutter_s": shutter_s, "shutter_type": r.rolling_shutter_type,
            "cull_scale_quantile": args.cull_scale_quantile,
            "cull": getattr(r, "cull_info", None)}
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
               "f_eff": pol.f_eff, "gt": gt_dump, "rollouts": recs,
               "synth_attach": attach_info,
               # G1 provenance: WHICH raster the model actually saw, banked in
               # the artifact so a geometry question never needs the GPU again.
               "canon": pol.canon_provenance,
               "window": int(getattr(pol, "window", WINDOW)),
               "model_provenance": getattr(pol, "prov", None),
               "lead_path_extrapolated_calls": int(leadgeom.path.n_extrap)}
    p = out / f"rollouts_{args.arm}_{args.condition}.json"
    # `default=str` only ever fires on the provenance block (a rebuilt dataclass
    # can carry a non-JSON leaf); every metric-bearing field is already plain.
    p.write_text(json.dumps(payload, default=str))
    logger.info("wrote %s (%d rollouts)", p, len(recs))


if __name__ == "__main__":
    main()
