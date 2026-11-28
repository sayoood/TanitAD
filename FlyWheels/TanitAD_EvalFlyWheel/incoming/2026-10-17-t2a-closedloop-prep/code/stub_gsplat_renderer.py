"""STUB renderer for the CPU dry path of the T2-A closed-loop harness.

It stands in for ``gsplat_renderer.NuRecGsplatRenderer`` (which needs CUDA + gsplat + a 640 MB scene) so
that EVERYTHING in ``closedloop_drive.py`` up to the first real render -- argparse, policy construction,
GT / camera setup, the control loop, the bicycle, the ground-follower, the record writer -- runs on a CPU
with no scene.

What it is NOT: it renders no NuRec scene. Its frames are a cheap ground-plane lane texture projected through
the REAL rig's f-theta intrinsics (1920x1080, cx 958.4379, cy 751.1790, poly from
``results/contract_test.json``). The only property that matters to the dry path is the one the real
renderer has: **the image is a function of the camera pose**, so the loop is closed on perception even
here (a test asserts that the pose handed to ``render`` is the EXECUTED pose, not the logged one).

Scene (a known-value world): the logged rig drives STRAIGHT along +x at a constant ``STUB_V`` m/s for 20 s
at 30 Hz (600 camera frames), so a hold-speed controller has an analytic answer.

Injection: ``install()`` puts this module in ``sys.modules['gsplat_renderer']``; ``closedloop_drive.main``
does ``from gsplat_renderer import ActorTracks, NuRecGsplatRenderer`` at call time and so receives the stub.
"""
from __future__ import annotations

import math
import sys
import types

import numpy as np

STUB_V = 15.0                       # m/s, the logged (and known-answer) speed
STUB_HZ = 30                        # camera frames per second (the real rig's rate)
STUB_FRAMES = 20 * STUB_HZ          # 20 s clip, like the real scenes
# the REAL rig (contract_test.json: grpc_camera)
CX, CY, W, H = 958.4378662109375, 751.178955078125, 1920, 1080
POLY = (0.0, 945.0581665039062, -14.906691551208496, 35.57021713256836,
        -77.3123550415039, 32.0064697265625)
# camera optical (x right, y down, z fwd) -> rig (x fwd, y left, z up), mounted 1.0 m ahead / 1.4 m up
_T_SENSOR_RIG = np.array([[0, 0, 1, 1.0],
                          [-1, 0, 0, 0.0],
                          [0, -1, 0, 1.4],
                          [0, 0, 0, 1.0]], np.float64)


class _Cam:
    cx, cy, width, height = CX, CY, W, H
    angle_to_pixeldist_poly = POLY
    shutter_type = "ROLLING_TOP_TO_BOTTOM"
    T_sensor_rig = _T_SENSOR_RIG


class _Rig:
    world_to_nre = np.eye(4)

    def T_rig_world(self, cam_name, frame, shutter=1):
        T = np.eye(4)
        T[0, 3] = STUB_V * (float(frame) / STUB_HZ)
        return T


class ActorTracks:                    # imported by closedloop_drive.main / cl_metrics.main; never used on `empty`
    def __init__(self, *a, **k):
        self.t0_us = 0.0


def _landmarks() -> np.ndarray:
    """Lane-marking points on the ground plane (world z = 0): dashed lines at y = -5.25, -1.75, 1.75, 5.25."""
    xs = np.arange(-20.0, 400.0, 2.0)
    pts = [(x, y, 0.0) for y in (-5.25, -1.75, 1.75, 5.25) for x in xs]
    return np.asarray(pts, np.float64)


_LM = _landmarks()


class NuRecGsplatRenderer:
    """Same public surface the harness touches; CPU; no scene."""

    cam_name = "camera_front_wide_120fov"
    instances: list = []                              # every renderer main() built (test hook)

    def __init__(self, scene_dir, layers=("background", "road"), loader_dir=None, **_):
        self.scene_dir = scene_dir
        self.cam = _Cam()
        self.rig = _Rig()
        self._actor = None
        self.layers = list(layers)
        self.render_calls: list[np.ndarray] = []      # every cam_to_nre handed to render() (test hook)
        self.cull_calls: list[float] = []             # --cull-scale-quantile values that REACHED the renderer (test hook)
        self.sky_calls: list[dict] = []               # attach_sky kwargs that REACHED the renderer (test hook)
        NuRecGsplatRenderer.instances.append(self)

    # --- geometry the harness reads ------------------------------------------------------------
    def n_frames(self):
        return STUB_FRAMES

    def frame_timestamps_us(self, frame):
        t = int(round(frame * 1e6 / STUB_HZ))
        return (t, t + 30_559)

    def gt_rig_to_world(self, frame):
        return self.rig.T_rig_world(self.cam_name, frame)

    @property
    def rolling_shutter_type(self):
        return self.cam.shutter_type

    def tau_of_us(self, ts_us):
        return float(ts_us) / 20e6

    def cull_by_scale(self, q):
        self.cull_calls.append(float(q))
        return {"stub": True, "q": q}

    def attach_sky(self, **kw):
        self.sky_calls.append(dict(kw))
        return object()                                # truthy: the stub "has" a sky and ignores it (the real run needs one:
                                                       # the harness aborts if --sky-gain meets a scene with no env map)

    # --- the render: pose -> image ---------------------------------------------------------------
    def render(self, cam_to_nre, tau=0.0, actor_time_us=None, cam_to_nre_end=None):
        self.render_calls.append(np.array(cam_to_nre, copy=True))
        Tinv = np.linalg.inv(cam_to_nre)
        pc = (Tinv[:3, :3] @ _LM.T).T + Tinv[:3, 3]
        keep = pc[:, 2] > 0.5
        pc = pc[keep]
        rho = np.hypot(pc[:, 0], pc[:, 1])
        theta = np.arctan2(rho, pc[:, 2])
        r = sum(c * theta ** i for i, c in enumerate(POLY))
        u = np.round(CX + r * pc[:, 0] / np.maximum(rho, 1e-9)).astype(int)
        v = np.round(CY + r * pc[:, 1] / np.maximum(rho, 1e-9)).astype(int)
        img = np.full((H, W, 3), 40, np.uint8)
        ok = (u > 1) & (u < W - 2) & (v > 1) & (v < H - 2)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                img[v[ok] + dv, u[ok] + du] = 255
        alpha = np.ones((H, W), np.float32)
        return img, alpha, 0.0


class _NullLeadGeometry:
    """``synth_actor.LeadGeometry`` needs a real renderer scene (cuboid pick). The dry path has none."""
    half_len = 0.0

    def __init__(self, renderer, *a, **k):
        self.tracks = {}
        self.path = types.SimpleNamespace(n_extrap=0)

    def set_t0(self, t_us):
        pass

    def at(self, T_ego, t_us, v_ego):
        return {}


def install():
    """Make ``from gsplat_renderer import ...`` and ``synth_actor.LeadGeometry`` resolve to the stubs."""
    mod = types.ModuleType("gsplat_renderer")
    mod.NuRecGsplatRenderer = NuRecGsplatRenderer
    mod.ActorTracks = ActorTracks
    sys.modules["gsplat_renderer"] = mod
    import synth_actor
    synth_actor.LeadGeometry = _NullLeadGeometry
    return mod
