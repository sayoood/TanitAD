"""T2-A policies for ``closedloop_drive.py`` -- refcv7-50,400 and the per-tick kinematic floors.

Registered through ``closedloop_drive.EXTRA_ARMS`` (``register()``), so ``main()`` needs no arm-specific code.
Everything here is importable and runnable on CPU; the ONLY thing that needs a GPU on the real run is the
renderer (gsplat) and, for speed, the model.

ARMS
  refcv7           refcv7-r101-s0 @ step 50,400, built by the launch gate's OWN G-EVAL loader
                   (``tanitad.eval.refcv7_loader.build_model``: strict load, G-DVB, stamp checks).
  cl_p10           the learned kinematic floor P10 (S2 head: 4 waypoints @ 0.5..2.0 s), RE-EVALUATED EVERY TICK
                   from the vehicle's OWN executed (v, a, steer).
  cl_p10_t0        P10 evaluated ONCE at t0 (logged poses f0-2..f0) and held -- the like-for-like companion of
                   the harness's cl_ha* floors. Always read beside the live version.
  cl_ha0_ext_live  constant-(a0, steer0) extrapolation, re-estimated every tick (the per-tick echo).
  (the held-at-t0 echo ``cl_ha0_ext`` and the zero control ``cl_ha0`` already live in the harness.)

THE UNIT TRAP (RETRACTION_LOG R46 class: "a control inherits the unit of the nearest default")
  The recorded action channel 1 is a road-wheel STEER ANGLE ``arctan(2.9 * kappa)`` (``physicalai.py:620``),
  not a curvature. ``floors_kin.const_control_paths`` integrates it correctly ONLY with ``action_units="steer"``
  (kappa = tan(steer)/2.9); its default ``"kappa"`` over-rotates ~2.9x. Both live floors below pass
  ``action_units="steer"`` explicitly, and ``test_t2a_dry.py`` pins it with a circle whose radius is known and a
  deliberate-regression arm (``"kappa"``) that must read ~2.9x wrong.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

import closedloop_drive as cd

STEER_L_ENC = 2.9           # the ENCODER's wheelbase (kinematic.STEER_WHEELBASE_M) -- not a vehicle property
V_EPS = 1e-3
IDX_S2 = [h - 1 for h in cd.WP_STEPS]          # fine-tick indices of the 0.5/1.0/1.5/2.0 s waypoints
N_FINE_S2 = int(cd.WP_STEPS[-1])                # 20 fine ticks of 0.1 s


# ============================================================================================ #
# kinematic state from the EXECUTED history                                                      #
# ============================================================================================ #
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def kin_from_history(h: np.ndarray, dt: float = cd.DT) -> dict:
    """``(v0, a0, steer0)`` at the last row of an ``[N>=2, 4]`` (x, y, yaw, v) history. Backward differences only.

    * ``v0``    = the last row's speed (the harness's ``v``);
    * ``a0``    = ``(v[-1] - v[-2]) / dt``  (``floors_kin.t0_kinematics`` with stride 1);
    * ``kappa`` = ``dyaw / (v_prev * dt)`` -- the curvature the vehicle ACTUALLY executed over the last step. The
      step ``k-1 -> k`` advances yaw on the speed at its START (the bicycle / ``rollout_unicycle`` convention), so
      the denominator is the PREVIOUS row's speed;
    * ``steer0`` = ``arctan(2.9 * kappa)`` -- the recorded channel-1 ENCODING (``physicalai.py:620``), because
      ``P10``'s features were fit on that channel. Converting back happens inside ``floors_kin`` with
      ``action_units="steer"``.
    """
    h = np.asarray(h, np.float64)
    if h.ndim != 2 or h.shape[0] < 2 or h.shape[1] < 4:
        raise ValueError(f"history must be [N>=2, >=4] (x, y, yaw, v), got {h.shape}")
    v_prev, v_now = float(h[-2, 3]), float(h[-1, 3])
    dyaw = _wrap(float(h[-1, 2]) - float(h[-2, 2]))
    kappa = dyaw / (v_prev * dt) if v_prev > V_EPS else 0.0
    return {"v0": v_now, "a0": (v_now - v_prev) / dt, "kappa": kappa,
            "steer0": math.atan(STEER_L_ENC * kappa)}


class _P10Core:
    """P10 = ``kd_x`` + MLP residual on the 8 kinematic features, as ``run_refcv7_floors.py`` builds it."""

    def __init__(self, weights_path: str, action_units: str = "steer"):
        import torch
        from taniteval import floors_kin as fk
        self.fk, self.action_units = fk, action_units
        ck = torch.load(weights_path, map_location="cpu", weights_only=False)
        n_out = len(cd.WP_STEPS) * 2
        self.net = fk._make_mlp(8, n_out)
        self.net.load_state_dict(ck["state"])
        self.stats = (ck["xm"], ck["xs"], ck["ym"])
        self.best = ck["best"]
        self.meta = ck.get("meta", {})
        if int(self.meta.get("n_fine_ticks", N_FINE_S2)) != N_FINE_S2 or \
                list(self.meta.get("instants_frames", list(cd.WP_STEPS))) != list(cd.WP_STEPS):
            raise RuntimeError(f"P10 weights {weights_path} were fit on a different grid "
                               f"({self.meta.get('instants_frames')}), not the harness's {list(cd.WP_STEPS)}")

    def paths(self, kin: np.ndarray):
        """-> dict of [n,4,2] float64: ha0, ha0_ext, kd_x, P10 for ``kin`` [n,3] = (v0, a0, steer0)."""
        fk = self.fk
        kin = np.asarray(kin, np.float64)
        h0 = fk.ha0_paths(kin[:, 0], cd.DT, N_FINE_S2)
        he = fk.ha0_ext_paths(kin, cd.DT, N_FINE_S2, action_units=self.action_units)
        kd = fk.kd_x(h0, he).astype(np.float64)
        base = kd[:, IDX_S2]
        kf = fk.kin_features(kin)
        p10 = fk.predict_p10((self.net, self.stats, self.best), kf, base.astype(np.float32))
        return {"ha0": h0[:, IDX_S2], "ha0_ext": he[:, IDX_S2], "kd_x": base, "P10": np.asarray(p10, np.float64)}


LIVE_ARMS = ("cl_p10", "cl_ha0_ext_live")


class LiveKinematicPolicy:
    """Per-tick kinematic floors. Reads NO pixel and NO nav; reads only the executed (x, y, yaw, v) history.

    ⛔ The history is supplied by ``run_rollout``'s ``set_history`` hook (rows <= NOW only). Nothing after NOW
    can reach this class: it never sees the logged future, only the logged PAST up to f0 and then its own
    executed track.
    """
    window = cd.WINDOW
    consumes_nav_known = False
    canon_mode = None

    def __init__(self, mode: str, p10_weights: str | None = None, action_units: str = "steer"):
        if mode not in LIVE_ARMS:
            raise ValueError(f"unknown live floor {mode!r}; expected one of {LIVE_ARMS}")
        self.name, self.mode = mode, mode
        self.f_eff = None
        self.canon_provenance = {"mode": "none", "reads_pixels": False,
                                 "why": "a kinematic floor consumes no image; frames are rendered, never read"}
        self.prov = {"floor": mode, "action_units": action_units,
                     "state": "executed history, backward differences, every index <= NOW"}
        self._hist = None
        self.core = None
        if mode == "cl_p10":
            if not p10_weights:
                raise ValueError("cl_p10 needs --t2a-p10 <p10_S2.pt>")
            self.core = _P10Core(p10_weights, action_units)
            self.prov["p10_weights"] = str(p10_weights)
            self.prov["p10_fit"] = {k: self.core.meta.get(k) for k in
                                    ("train_manifest_md5", "n_train_windows", "n_train_episodes", "selection",
                                     "training_seed", "ade_p10_train_m")}
        else:
            from taniteval import floors_kin as fk
            self.fk, self.action_units = fk, action_units

    def set_history(self, h):
        self._hist = np.asarray(h, np.float64)

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        if self._hist is None:
            raise RuntimeError(f"{self.name}.set_history() was never called -- run_rollout's hook is missing; the "
                               f"floor's kinematic state would be invented. Refusing.")
        k = kin_from_history(self._hist)
        if abs(k["v0"] - float(v0)) > 1e-6:
            raise RuntimeError(f"history v={k['v0']} != harness v0={float(v0)}: the history and the loop disagree")
        kin = np.array([[k["v0"], k["a0"], k["steer0"]]], np.float64)
        if self.mode == "cl_p10":
            traj = self.core.paths(kin)["P10"][0]
        else:
            traj = self.fk.ha0_ext_paths(kin, cd.DT, N_FINE_S2, action_units=self.action_units)[0, IDX_S2]
        extra = {"floor_mode": self.mode, "floor_v0_ms": k["v0"], "floor_a0_mps2": k["a0"],
                 "floor_kappa_exec_1pm": k["kappa"], "floor_steer0_enc_rad": k["steer0"],
                 "floor_reads_pixels": 0.0, "floor_reads_nav": 0.0}
        return np.ascontiguousarray(traj, np.float64), extra


class HeldKinematicPolicy(LiveKinematicPolicy):
    """``cl_p10_t0``: P10 evaluated ONCE at the rollout's t0 and that plan re-issued every tick -- the SAME convention
    as the harness's ``cl_ha*`` floors (state measured at t0 from LOGGED poses f0-2..f0, held for the whole rollout).

    Why both exist: a per-tick re-estimating echo (``cl_p10``) is a feedback controller with loop gain ~1 (the pure-
    pursuit waypoint at 0.5 s maps a held curvature back onto itself), so ANY bias in the floor integrates tick after
    tick (MEASURED in the dry path: P10's lateral residual at the zero state ramps a straight log into a 2.35 m lateral
    error over 5 s). The held version is the like-for-like companion of ``cl_ha0_ext``; the live one is the floor
    played as a driver. Read them side by side; never quote one without the other.
    """
    HELD_MODE = "cl_p10_t0"

    def __init__(self, p10_weights: str, action_units: str = "steer"):
        super().__init__("cl_p10", p10_weights=p10_weights, action_units=action_units)
        self.name = self.mode = self.HELD_MODE
        self.prov["floor"] = self.HELD_MODE
        self.prov["state"] = "measured ONCE at t0 from logged poses f0-2..f0 (backward differences), held"
        self._t0 = None
        self._held = None

    def set_t0(self, gt_T, f0):
        f0 = int(f0)
        if f0 < 2:
            raise RuntimeError(f"{self.name} needs f0 >= 2 (a0 is a second difference), got {f0}")
        p = [np.asarray(gt_T[i][:2, 3], np.float64) for i in (f0 - 2, f0 - 1, f0)]
        yaw = [cd._yaw(gt_T[i]) for i in (f0 - 1, f0)]
        vb_prev = float(np.linalg.norm(p[1] - p[0]) / cd.DT)
        vb_now = float(np.linalg.norm(p[2] - p[1]) / cd.DT)
        kappa = _wrap(yaw[1] - yaw[0]) / (max(vb_now, V_EPS) * cd.DT) if vb_now > V_EPS else 0.0
        self._t0 = {"f0": f0, "a0_mps2": (vb_now - vb_prev) / cd.DT, "kappa0_1pm": kappa,
                    "steer0_enc_rad": math.atan(STEER_L_ENC * kappa), "v_backward_at_t0_ms": vb_now,
                    "pose_indices_read": [f0 - 2, f0 - 1, f0]}
        self.t0 = dict(self._t0)
        self._held = None

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        if self._t0 is None:
            raise RuntimeError(f"{self.name}.set_t0() was never called -- the t0 state would be invented. Refusing.")
        if self._held is None:
            kin = np.array([[float(v0), self._t0["a0_mps2"], self._t0["steer0_enc_rad"]]], np.float64)
            self._held = np.ascontiguousarray(self.core.paths(kin)["P10"][0], np.float64)
        extra = {"floor_mode": self.name, "floor_a0_mps2": self._t0["a0_mps2"],
                 "floor_kappa0_1pm": self._t0["kappa0_1pm"], "floor_steer0_enc_rad": self._t0["steer0_enc_rad"],
                 "floor_reads_pixels": 0.0, "floor_reads_nav": 0.0}
        return self._held.copy(), extra


# ============================================================================================ #
# rig extrinsics for the lift bank                                                              #
# ============================================================================================ #
def rot_to_quat_xyzw(R: np.ndarray):
    """Shepperd's method. Returns (qx, qy, qz, qw), qw >= 0."""
    R = np.asarray(R, np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return qx, qy, qz, qw


def quat_xyzw_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def extrinsics_from_T_sensor_rig(T_sensor_rig: np.ndarray):
    """``FrontWideExtrinsics`` from a NuRec ``T_sensor_rig`` (camera/optical -> rig).

    CONVENTION CHECK (MEASURED 2026-10-17, ``raw/scene_info.json`` -> ``extrinsics_rot_frob_dist_to_nearest_physicalai_row``): the two renderable scenes'
    ``T_sensor_rig`` rotations sit 0.0033 / 0.0052 (Frobenius) from the nearest of 4,508 PhysicalAI-AV
    ``sensor_extrinsics`` rows, with a relative rotation of identity -- i.e. the SAME convention (camera axes
    +x right, +y down, +z boresight; R = camera -> vehicle). No axis permutation is applied here.
    """
    from tanitad.data.physicalai import FrontWideExtrinsics
    T = np.asarray(T_sensor_rig, np.float64)
    qx, qy, qz, qw = rot_to_quat_xyzw(T[:3, :3])
    return FrontWideExtrinsics(qx=qx, qy=qy, qz=qz, qw=qw, x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3]))


def read_rig_json(scene_or_rig: str) -> dict:
    """Locate and read ``rig_trajectories.json`` for a NuRec scene dir (or take the file itself)."""
    p = Path(scene_or_rig)
    cands = [p] if p.is_file() else [p / "rig_trajectories.json", p / "extracted" / "rig_trajectories.json"]
    for c in cands:
        if c.is_file():
            return {"path": str(c), "json": json.loads(c.read_text(encoding="utf-8"))}
    raise FileNotFoundError(f"no rig_trajectories.json under {scene_or_rig!r} (tried {[str(c) for c in cands]})")


def front_wide_T_sensor_rig(rig_json: dict) -> np.ndarray:
    cams = rig_json["camera_calibrations"]
    items = cams.items() if isinstance(cams, dict) else enumerate(cams)
    for k, v in items:
        if cd.CAM in str(k) or (isinstance(v, dict) and cd.CAM in json.dumps(v)[:400]):
            return np.array(v["T_sensor_rig"], np.float64)
    raise KeyError(f"{cd.CAM} not in camera_calibrations")


# ============================================================================================ #
# refcv7                                                                                        #
# ============================================================================================ #
class RefCV7Policy(cd._BasePolicy):
    """refcv7-50,400 in the closed loop.

    INPUT CONTRACT -- MEASURED against one REAL eval item (``raw/cpu_model_smoke.json`` -> ``eval_item_contract``):
      frames  [W=8, 9, 416, 1024] uint8   -> here: ``canon`` -> ``[1, 8, 9, 416, 1024]`` float via a 0-dim TENSOR divisor
      pose_hist [8, 4] (x, y, yaw, v)     -> here: the executed history (``set_history``); ``pose_last == pose_hist[-1]``
      nav_cmd int64 scalar                -> ``nav_from_route`` (same labeller family; NAV_STRAIGHT is never emitted)
      v_max_ms f32 / v_max_valid f32      -> ORACLE ceiling, see ``vmax`` below
      map_ep (episode id)                 -> a ONE-clip lift bank built from the scene's own camera mount

    ⛔ CEILING (``v_max_ms``). The T1 sidecar value is an ORACLE: the max of the ego's own REALISED speed over
    [t0+2 s, +6 s], quantised to {30, 50, 100, 120} km/h (PI-authorised 2026-09-16,
    ``refcv6_speed_max_v8_eval.jsonl.meta.json``). There is no sidecar for a NuRec scene. Three modes:
      * ``withheld``    ``v_max_valid = 0`` -- the all-zero row; the honest, DEPLOYABLE setting (the default);
      * ``logged_2_6s`` the T1 derivation re-applied to the LOGGED path of THIS scene at EACH rollout's t0 (``set_t0``
                        hook): max logged speed over [f0+20, f0+60] ticks, ``v_max_valid = 1`` (T1-parity, ORACLE);
      * ``fixed``       a constant ``vmax_ms`` for the whole run (ORACLE if derived from the log).
    MEASURED on the two scenes: scene 2 starts at standstill, so the [2 s, 6 s] max (4.90 m/s -> the 30 km/h bin) and
    the whole-clip max (15.2 m/s -> the 100 km/h bin) differ -- a constant is NOT interchangeable with the per-start
    value there. Which mode ran is stamped in the payload (``model_provenance.vmax_mode``).
    """
    name = "refcv7"
    canon_mode = "cyl416x1024"
    consumes_nav_known = False

    VMAX_MODES = ("withheld", "fixed", "logged_2_6s")

    def __init__(self, ckpt: str, config_path: str, rig: dict, vmax_ms: float | None = None,
                 infer_seed: int = 0, device: str = "cuda", scene_id: str = "scene",
                 vmax_mode: str | None = None):
        super().__init__(device)
        import copy
        from collections import OrderedDict
        torch = self.torch
        vmax_mode = vmax_mode or ("withheld" if vmax_ms is None else "fixed")
        if vmax_mode not in self.VMAX_MODES:
            raise ValueError(f"vmax_mode {vmax_mode!r} not in {self.VMAX_MODES}")
        if (vmax_mode == "fixed") != (vmax_ms is not None):
            raise ValueError("vmax_mode 'fixed' needs vmax_ms, and vmax_ms is only valid with 'fixed'")
        self.vmax_mode = vmax_mode
        self.t0 = None
        from tanitad.eval import refcv7_loader as L
        from tanitad.data.v2_dataset import stable_episode_id
        from tanitad.models.trunk_shapes import FRAME_416x1024
        self.frame = FRAME_416x1024
        config = L.load_config(config_path)
        self.model, self.cfg, self.targs, rec = L.build_model(config, ckpt, device=self.device)
        self.prov = {
            "loader": rec["loader"], "repo": rec["repo"], "step": rec["state_dict"]["step"],
            "state_dict_strict": rec["state_dict"]["strict"], "missing": rec["state_dict"]["missing"],
            "unexpected": rec["state_dict"]["unexpected"], "param_total": rec["param_breakdown"]["rebuilt"].get("total"),
            "param_breakdown_equal_config": rec["param_breakdown"]["equal"],
            "declared_vs_built_mismatches": rec["declared_vs_built"]["mismatches"],
            "decoder_steps": rec["decoder_steps"], "sampler": rec["sampler"], "departures": rec["departures"],
            "horizons": [int(h) for h in self.cfg.core.trajectory.horizons], "window": int(self.cfg.core.window),
            "image_hw": list(self.cfg.core.encoder.image_hw()), "vmax_ms": vmax_ms,
            "vmax_mode": {"withheld": "withheld (v_max_valid=0) -- deployable",
                          "fixed": "ORACLE fixed constant (v_max_valid=1)",
                          "logged_2_6s": "ORACLE per-rollout max logged speed over [t0+2 s, t0+6 s] (v_max_valid=1)"}[vmax_mode],
            "infer_seed": int(infer_seed), "ckpt": str(ckpt), "config": str(config_path)}
        self.horizons = self.prov["horizons"]
        self.window = self.prov["window"]
        self.decoder_steps = int(rec["decoder_steps"])
        self.infer_seed = int(infer_seed)
        self.vmax_ms = vmax_ms
        # ---- refuse-on-mismatch, as RefCV3Policy does --------------------------------------------------- #
        if tuple(self.horizons[:len(cd.WP_STEPS)]) != tuple(cd.WP_STEPS):
            raise RuntimeError(f"HORIZON GRID MISMATCH: ckpt {self.horizons[:4]} vs harness {list(cd.WP_STEPS)}")
        if self.prov["image_hw"] != [416, 1024]:
            raise RuntimeError(f"ckpt image_hw {self.prov['image_hw']} != [416, 1024]")
        if int(self.cfg.core.encoder.in_channels) != 3 * cd.STACK:
            raise RuntimeError("encoder in_channels != 3*STACK")
        if getattr(self.model.core, "ego_hist", None) is None:
            raise RuntimeError("this build has no ego-history encoder: not a refcv7 build")
        if not bool(getattr(self.cfg, "max_speed_onehot_v6", False)):
            raise RuntimeError("this build has no refcv6 4-way ceiling seam: not a refcv7 build")
        # ---- the ONE-clip lift bank (shares the trained frame/stride/heights/grid/observed-mask) --------- #
        bank0 = self.model._lift_bank_hires
        if bank0 is None:
            raise RuntimeError("model has no 10 cm lift bank")
        T = front_wide_T_sensor_rig(rig["json"])
        self.extr = extrinsics_from_T_sensor_rig(T)
        sid = int(stable_episode_id(str(scene_id)))
        bank = copy.copy(bank0)
        bank._extr = {sid: self.extr}
        bank.clip_of = {sid: str(scene_id)}
        bank._cache = OrderedDict()
        self.bank, self.map_ep = bank, sid
        self.prov["scene_id"], self.prov["rig_json"] = str(scene_id), rig["path"]
        self.prov["extrinsics_xyz_m"] = [self.extr.x, self.extr.y, self.extr.z]
        self._hist = None
        self._tick = 0
        self._canon_cache = {}

    # ------------------------------------------------------------------------------------------ #
    def set_history(self, h):
        self._hist = np.asarray(h, np.float64)

    def set_t0(self, gt_T, f0):
        """Called once per rollout by ``run_rollout`` (LOGGED poses; f0 = the window's NOW). Only the ceiling reads it."""
        f0 = int(f0)
        self._tick = 0
        info = {"f0": f0, "vmax_mode": self.vmax_mode}
        if self.vmax_mode == "logged_2_6s":
            v = cd.gt_poses_xyv(gt_T)[:, 3]
            lo, hi = f0 + int(round(2.0 / cd.DT)), min(f0 + int(round(6.0 / cd.DT)), len(v) - 1)
            if hi < lo:
                self.vmax_ms = None
                info.update(vmax_ms=None, why="logged path ends before t0+2 s: ceiling WITHHELD for this rollout")
            else:
                self.vmax_ms = float(v[lo:hi + 1].max())
                info.update(vmax_ms=self.vmax_ms, window_ticks=[lo, hi], window_truncated=bool(hi < f0 + 60))
        else:
            info["vmax_ms"] = self.vmax_ms
        self.t0 = info

    def canon(self, frames, intr):
        """416x1024 cylindrical canonicalisation -> ``[1, W, 9, 416, 1024]`` float32 in [0, 1].

        Each NATIVE frame is resampled ONCE and cached (the harness re-passes the last 10 frames every tick; the resample is
        per-frame and independent of its neighbours, so caching changes no value). The cache holds a REFERENCE to the frame
        it was computed from, so a recycled ``id()`` can never produce a stale hit.
        """
        torch = self.torch
        from tanitad.data.calib import cylindrical_rectify
        canon = []
        for f in frames:
            hit = self._canon_cache.get(id(f))
            if hit is None or hit[0] is not f:
                vid = torch.from_numpy(np.ascontiguousarray(f))[None].permute(0, 3, 1, 2).to(self.device)
                c = cylindrical_rectify(vid, intr, self.frame)[0]                  # [3,416,1024] u8 on self.device
                if self.f_eff is None:
                    self.f_eff = float(cylindrical_rectify.last_f_eff)
                    obs = float(cylindrical_rectify.last_observed_frac)
                    if abs(self.f_eff - self.frame.f_ref) > 1e-9 or not (0.5 < obs <= 1.0):
                        raise RuntimeError(f"416x1024 canon self-check FAILED (f_eff={self.f_eff}, observed_frac={obs})")
                    self.canon_provenance = {
                        "mode": "cyl416x1024", "f_eff": self.f_eff, "f_ref": self.frame.f_ref, "observed_frac": obs,
                        "projection": self.frame.projection, "height": self.frame.height, "width": self.frame.width,
                        "hfov_deg_cylindrical": math.degrees(2.0 * (self.frame.width / 2.0) / self.frame.f_ref),
                        "intr_cx": float(intr.cx), "intr_cy": float(intr.cy), "intr_per_clip": bool(intr.per_clip),
                        "resample_device": str(self.device)}
                hit = (f, c)
                self._canon_cache[id(f)] = hit
                while len(self._canon_cache) > 24:
                    self._canon_cache.pop(next(iter(self._canon_cache)))
            canon.append(hit[1])
        st = self._stack(torch.stack(canon), cd.STACK)                              # [T-2, 9, 416, 1024]
        fw = st[-self.window:][None].to(self.device)
        fw = fw.float().div_(torch.tensor(255.0, device=fw.device))            # TENSOR divisor (CUDA 1-ulp trap)
        if tuple(fw.shape) != (1, self.window, 9, 416, 1024):
            raise RuntimeError(f"raster assertion failed: {tuple(fw.shape)}")
        return fw

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        torch = self.torch
        if self._hist is None:
            raise RuntimeError("refcv7.set_history() was never called: the ego-history encoder would read nothing")
        if self._hist.shape[0] < self.window:
            raise RuntimeError(f"history has {self._hist.shape[0]} rows < window {self.window}")
        with torch.no_grad():
            fw = self.canon(frames, intr)
            ph = torch.from_numpy(self._hist[-self.window:].astype(np.float32))[None].to(self.device)
            if abs(float(ph[0, -1, 3]) - float(v0)) > 1e-4:
                raise RuntimeError(f"pose_hist[-1].v={float(ph[0, -1, 3])} != v0={float(v0)}: pose_last must be the "
                                   f"last history row (the dataset's contract)")
            v0t = ph[:, -1, 3].clone()
            navt = torch.tensor([int(nav_cmd)], dtype=torch.long, device=self.device)
            if self.vmax_ms is None:
                vm = torch.zeros(1, dtype=torch.float32, device=self.device)
                vv = torch.zeros(1, dtype=torch.float32, device=self.device)
            else:
                vm = torch.tensor([float(self.vmax_ms)], dtype=torch.float32, device=self.device)
                vv = torch.ones(1, dtype=torch.float32, device=self.device)
            pg, pv = self.bank.for_episodes([self.map_ep], device=self.device)
            self.model.core.set_ego_window(ph, int(ph.shape[1]), actions=None)
            torch.manual_seed(self.infer_seed + 1000 * self._tick)      # ONE DDIM draw, reproducible per (seed, tick)
            self._tick += 1
            out = self.model(fw, nav_cmd=navt, v0=v0t, steps=self.decoder_steps, v_max_ms=vm, v_max_valid=vv,
                             perception_grid=pg, perception_valid=pv)
            full = out["traj"][0].float().cpu().numpy()                 # [8, 2]
        traj = full[:len(cd.WP_STEPS)]
        seg = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], full]), axis=0), axis=1)
        dts = np.diff(np.array([0] + self.horizons, np.float64)) * cd.DT
        extra = {"traj_full_6s": full.tolist(), "vmax_fed_ms": float(vm[0]), "vmax_valid": float(vv[0]),
                 "plan_peak_speed_ms": float((seg / dts).max())}
        return traj.astype(np.float64), extra


# ============================================================================================ #
# registration                                                                                  #
# ============================================================================================ #
def _add_args(ap):
    g = ap.add_argument_group("T2-A arms (t2a_policies)")
    g.add_argument("--t2a-config", default=None, help="refcv7 run config.json (default: <ckpt dir>/config.json)")
    g.add_argument("--t2a-vmax-mode", default="withheld", choices=list(RefCV7Policy.VMAX_MODES),
                   help="refcv7 speed-ceiling input. withheld = v_max_valid 0 (DEPLOYABLE, default); logged_2_6s = the "
                        "T1 oracle re-derived from the LOGGED path at each rollout's t0; fixed = --t2a-vmax-ms.")
    g.add_argument("--t2a-vmax-ms", type=float, default=None, help="ceiling for --t2a-vmax-mode fixed (m/s)")
    g.add_argument("--t2a-infer-seed", type=int, default=0, help="refcv7 DDIM inference seed")
    g.add_argument("--t2a-p10", default=None, help="P10 weights (p10_S2.pt) for --arm cl_p10")
    g.add_argument("--t2a-action-units", default="steer", choices=["steer", "kappa"],
                   help="⛔ unit of the recorded channel-1 the live floors integrate. 'steer' is CORRECT; 'kappa' "
                        "exists ONLY for the deliberate-regression control (RETRACTION_LOG R46 class).")


def _build_refcv7(args):
    cfg = args.t2a_config or str(Path(args.ckpt).with_name("config.json"))
    rig = read_rig_json(args.scene_dir)
    # CPU only when the caller EXPLICITLY hid the GPU (CUDA_VISIBLE_DEVICES=-1, the dry path). Otherwise "cuda", which fails LOUD on a
    # box with no GPU instead of silently running a 165-200 s/tick CPU forward.
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda"
    return RefCV7Policy(args.ckpt, cfg, rig, vmax_ms=args.t2a_vmax_ms, vmax_mode=args.t2a_vmax_mode,
                        infer_seed=args.t2a_infer_seed, device=dev,
                        scene_id=Path(args.scene_dir).name if Path(args.scene_dir).is_dir() else "scene")


def _build_live(mode):
    def b(args):
        return LiveKinematicPolicy(mode, p10_weights=args.t2a_p10, action_units=args.t2a_action_units)
    return b


def register():
    """Idempotent. Call BEFORE ``closedloop_drive.main()`` so the parser sees the arms and their flags."""
    cd.EXTRA_ARMS["refcv7"] = {"build": _build_refcv7, "needs_ckpt": True, "add_args": _add_args}
    for m in LIVE_ARMS:
        cd.EXTRA_ARMS[m] = {"build": _build_live(m), "needs_ckpt": False, "add_args": _add_args}
    cd.EXTRA_ARMS[HeldKinematicPolicy.HELD_MODE] = {
        "build": lambda args: HeldKinematicPolicy(args.t2a_p10, action_units=args.t2a_action_units),
        "needs_ckpt": False, "add_args": _add_args}
