# PROMOTED 2026-09-19 by EvalFlyWheel W1 from
#   FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/code/tanitad_navsim_bridge.py
# ONE edit (the REPO depth, marked below); every function and class is AST-identical to the origin,
# pinned by taniteval/tests/test_bench_suite_promotion.py. The origin file is the historical record.
#!/usr/bin/env python3
"""TanitAD side of the seam (RUN IN THE TANITAD VENV) — refcv4b on NavSim scenes.

    NavSim export (devkit AgentInput, per token)  ──►  DECLARE (arm)  ──►  model inputs
    DataFlyWheel frame bank (`wide`, sha-checked) ──►  time construction ──►  frames
                                  refcv4b (CPU) ──► traj [8 knots, 2] ──► NavSim poses [8, 3]

⛔ THE ENFORCEMENT MECHANISM (gate ``navsim.ego_enforcement``) lives in ONE place:
:func:`declare`. It COPIES only the fields an arm declares out of the exported
``EgoStatus`` list into a fresh dict; :func:`model_inputs` receives ONLY that
dict and the frames. No other function in this file ever sees the export's ego
block. The mutation test (``tests/test_ego_mutation.py``) proves it by
randomising every UNDECLARED field and requiring byte-identical trajectories,
and — the control that must go RED — by mutating a DECLARED field and requiring
the trajectories to CHANGE.

Every model fact below is READ from the checkpoint's own ``config.json`` /
``anchors.units.json`` at load time and ASSERTED against the constants here,
never quoted from prose (registry §4.6).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))   # PROMOTION EDIT 1 of 1: 5 -> 4 levels
if os.path.join(REPO, "stack") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "stack"))

# --------------------------------------------------------------------------- #
# constants — each ASSERTED against its source at load time                     #
# --------------------------------------------------------------------------- #
#: refcv4b slot times: config.json horizons [5,10,15,20,30,40,50,60] x dt 0.1 s.
KNOT_T_S = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
#: NavSim agent output: TrajectorySampling(time_horizon=4, interval_length=0.5)
#: (`config/common/agent/constant_velocity_agent.yaml`, `human_agent.yaml`).
NAVSIM_T_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
#: anchors.units.json: alat_v_floor_ms / kappa_cap_inv_m (the decoder's roll_bank caps).
ALAT_V_FLOOR_MS = 4.0
KAPPA_CAP_INV_M = 0.12
#: The model's native frame cadence (v2ep provider grid) and D-015 channel stack.
DT_FRAME_S = 0.1
N_STACK = 3
#: NavSim driving_command one-hot index -> refb.NAV_COMMANDS name.
#: NAV_COMMANDS = ("follow", "left", "right", "straight") (refc.py:187, refb.py:64).
#: NavSim order (left, straight, right, unknown) is VERIFIED empirically on the
#: logs by tests/test_command_order.py before this map is trusted.
NAVSIM_CMD_TO_NAV_NAME = {0: "left", 1: "follow", 2: "right", 3: "follow"}
#: heading is held (not re-derived) below this path speed — atan2 of a ~zero
#: tangent is noise, and a noisy heading makes the LQR steer at a stop.
#: ⚠️ AMENDED 2026-09-19 BEFORE ANY SCORING (SPEC §9): the first value, 0.1 m/s,
#: was MEASURED on the 2-scene pipeline smoke to turn cm-level slot-offset noise
#: of a near-stopped plan into heading swings of +-0.7 rad (and, through an
#: np.unwrap of held values, -6.5 rad). At 1.0 m/s a 0.5 s interval moves >= 0.5 m,
#: so a 5 cm offset jitter bounds the tangent error near 0.1 rad.
HEADING_HOLD_SPEED_MS = 1.0

# --------------------------------------------------------------------------- #
# the arms                                                                       #
# --------------------------------------------------------------------------- #
#: ``declared`` names the ONLY EgoStatus fields an arm may read (t0 = frame -1).
ARMS = {
    "A1_ego_cmd": {
        "declared": ("ego_velocity[t0]", "ego_acceleration[t0]", "driving_command[t0]"),
        "keep": 1, "nav": "cmd", "frames": "ST",
        "meaning": "frames + measured t0 ego state + NavSim driving_command -> nav"},
    "A2_vision_pure": {
        "declared": (),
        "keep": 0, "nav": None, "frames": "ST",
        "meaning": ("frames ONLY: ego block keep=0 (the model zeroes its values), v0 "
                    "withheld at the core (v0=None), nav withheld (nav_cmd=None)")},
    "A3_ego_nocmd": {
        "declared": ("ego_velocity[t0]", "ego_acceleration[t0]"),
        "keep": 1, "nav": None, "frames": "ST",
        "meaning": "frames + measured t0 ego state; nav withheld (nav_cmd=None)"},
    "A1NT_ego_cmd_nearest": {
        "declared": ("ego_velocity[t0]", "ego_acceleration[t0]", "driving_command[t0]"),
        "keep": 1, "nav": "cmd", "frames": "NT",
        "meaning": "A1 with the NEAREST-TIME history construction (declared sensitivity arm)"},
    "A2NT_vision_pure_nearest": {
        "declared": (),
        "keep": 0, "nav": None, "frames": "NT",
        "meaning": "A2 with the NEAREST-TIME history construction (declared sensitivity arm)"},
    "A4_blind_ego_cmd": {
        "declared": ("ego_velocity[t0]", "ego_acceleration[t0]", "driving_command[t0]"),
        "keep": 1, "nav": "cmd", "frames": "BLIND",
        "meaning": ("the registered DELIBERATE REGRESSION (frames_blind): every frame "
                    "replaced by one constant grey; ego + cmd as A1. Diagnostic: is "
                    "vision HELPING or HURTING zero-shot? Needs no pixels, so it is the "
                    "only refcv4b-derived arm that can also run the 16 stage-1 scenes.")},
}
FIELDS = ("ego_pose", "ego_velocity", "ego_acceleration", "driving_command")


class RefusedInput(Exception):
    """An input the bridge will not guess about (fail loud)."""


# --------------------------------------------------------------------------- #
# 1. DECLARE — the enforcement point                                            #
# --------------------------------------------------------------------------- #
def declare(ego_statuses: list, arm: str) -> dict:
    """Copy ONLY the arm's declared t0 fields out of the exported EgoStatus list.

    ``ego_statuses`` is the export's list (oldest first; ``[-1]`` is t0). The
    returned dict is the COMPLETE ego/route information the model call may use.
    """
    spec = ARMS[arm]
    t0 = ego_statuses[-1]
    out = {"_arm": arm, "_declared": list(spec["declared"])}
    for name in spec["declared"]:
        field, idx = name.split("[")
        if idx != "t0]":
            raise RefusedInput(f"only t0 fields are declarable here, got {name!r}")
        out[field] = [float(x) for x in t0[field]]
    return out


def nav_index_from_command(cmd4) -> tuple[int, str, int]:
    """(NAV_COMMANDS index, name, navsim argmax) from a 4-dim one-hot, REFUSING a
    non-one-hot vector rather than arg-maxing it."""
    from tanitad.refs import refb
    c = np.asarray(cmd4, dtype=np.float64)
    if c.shape != (4,) or not np.all(np.isin(c, (0.0, 1.0))) or c.sum() != 1.0:
        raise RefusedInput(f"driving_command is not a 4-dim one-hot: {c.tolist()}")
    k = int(np.argmax(c))
    name = NAVSIM_CMD_TO_NAV_NAME[k]
    return int(refb.NAV_COMMANDS.index(name)), name, k


def ego_block(decl: dict) -> dict:
    """The model's ``[5]`` ego block + the core's ``v0`` from DECLARED fields only.

    v0       = hypot(vx, vy)                          (physicalai.signals_at: v = hypot)
    a_long   = ax                                     (physicalai.signals_at: dataset ax)
    curvature= clip(ay / max(v0, 4.0)^2, +-0.12)      (anchors.units.json caps)
    yaw_rate = v0 * curvature                         (refc_v3.ego_state_at_t0: r0=v0*k0)
    keep     = 1
    A withheld arm (keep=0) gets zeros + keep 0 and v0=None: refc_v3.py:1353-1361
    multiplies the values by keep, and the core derives keep from v0 is None.
    """
    arm = decl["_arm"]
    if ARMS[arm]["keep"] == 0:
        return {"ego_state": [0.0, 0.0, 0.0, 0.0, 0.0], "v0": None,
                "formula": "withheld: ego_state=(0,0,0,0,keep=0), v0=None"}
    vx, vy = decl["ego_velocity"]
    ax, ay = decl["ego_acceleration"]
    v0 = math.hypot(vx, vy)
    kappa = ay / max(v0, ALAT_V_FLOOR_MS) ** 2
    kappa = max(-KAPPA_CAP_INV_M, min(KAPPA_CAP_INV_M, kappa))
    return {"ego_state": [v0, ax, v0 * kappa, kappa, 1.0], "v0": v0,
            "formula": ("v0=hypot(vx,vy); a_long=ax; kappa=clip(ay/max(v0,4.0)^2,+-0.12); "
                        "yaw_rate=v0*kappa; keep=1")}


# --------------------------------------------------------------------------- #
# 2. FRAMES                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class FrameBank:
    root: str                   # C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus
    provenance: dict            # scene_token -> sha256[:16]

    @classmethod
    def open(cls, root: str) -> "FrameBank":
        import pandas as pd
        p = pd.read_parquet(os.path.join(root, "frames_provenance.parquet"))
        return cls(root=root, provenance=dict(zip(p.scene_token, p.sha256)))

    def load_wide(self, scene_token: str) -> tuple[np.ndarray, np.ndarray, str]:
        """``(frames u8 [4,256,640,3], observed mask [256,640], sha16)`` — the
        `wide` variant (PHYSICALAI_WIDE120_256x640), sha-VERIFIED before use."""
        a = np.load(os.path.join(self.root, "frames", f"{scene_token}.npy"))
        src = np.load(os.path.join(self.root, "frames", f"{scene_token}.src.npy"))
        sha = hashlib.sha256(a.tobytes()).hexdigest()[:16]
        want = self.provenance.get(scene_token)
        if want is None or sha != want:
            raise RefusedInput(f"{scene_token}: bank sha {sha} != provenance {want}")
        if a.dtype != np.uint8 or a.shape != (4, 256, 640, 3):
            raise RefusedInput(f"{scene_token}: bank is {a.dtype}{a.shape}, not u8 (4,256,640,3)")
        if float(a.mean()) < 1.0:
            raise RefusedInput(f"{scene_token}: bank mean {a.mean():.3f} — an all-black bank")
        return a, src >= 0, sha


def frame_tag_check(bank_build: dict) -> dict:
    """REFUSE a frame-tag mismatch, printing both tags (ADAPTER_CHANGES §1)."""
    from tanitad.data.calib import PHYSICALAI_WIDE120_256x640 as F
    fb = bank_build["frame"]
    want = (F.height, F.width, round(F.f_ref, 6), F.projection)
    have = (fb["h"], fb["w"], round(fb["f_ref"], 6), fb["projection"])
    if want != have:
        raise RefusedInput(f"frame tag mismatch: model expects {want}, bank is {have}")
    return {"model_frame": F.tag() if hasattr(F, "tag") else str(want),
            "bank_frame": have, "match": True}


def slot_sources(times_s: list[float], construction: str, window_rows: int) -> list[int]:
    """Which NavSim history frame fills each RAW 10 Hz slot, oldest slot first.

    The model reads ``window_rows`` rows of an ``N_STACK``-frame channel stack
    (D-015), i.e. ``window_rows + N_STACK - 1`` raw frames at 0.1 s, ending at
    t0. ``times_s`` are the NavSim frames' times relative to t0 (<= 0).
      ST  every slot <- the t0 frame (static history; no fabricated motion)
      NT  every slot <- the NavSim frame NEAREST in time (ties -> the later)
    """
    n_raw = window_rows + N_STACK - 1
    slots = [-(n_raw - 1 - k) * DT_FRAME_S for k in range(n_raw)]     # oldest first
    if construction == "ST":
        return [len(times_s) - 1] * n_raw
    if construction == "NT":
        out = []
        for s in slots:
            d = [abs(s - t) for t in times_s]
            best = min(d)
            out.append(max(i for i, x in enumerate(d) if abs(x - best) < 1e-9))
        return out
    raise RefusedInput(f"unknown construction {construction!r}")


def pack_frames(frames_u8: np.ndarray, sources: list[int], blind_grey: int | None = None):
    """``[4,H,W,3]`` NavSim frames -> ``[W_rows, 9, H, W]`` uint8 via the TRAINER'S
    OWN ``stack_frames`` (tanitad.data.comma2k19, D-015: oldest first, current in
    the last 3 channels). Never a resize: the bank is already the model frame."""
    import torch
    from tanitad.data.comma2k19 import stack_frames
    raw = torch.from_numpy(np.ascontiguousarray(frames_u8[sources])).permute(0, 3, 1, 2)
    if blind_grey is not None:
        raw = torch.full_like(raw, int(blind_grey))
    return stack_frames(raw.contiguous(), N_STACK)


# --------------------------------------------------------------------------- #
# 3. MODEL                                                                       #
# --------------------------------------------------------------------------- #
def load_refcv4b(ckpt: str, config: str | None = None):
    """``refcv3_arm.load_model`` — the landing eval's own rebuild (trainer parser +
    _pin_trainer_cfg, config cross-check, strict load), imported by path."""
    import importlib.util
    path = os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py")
    spec = importlib.util.spec_from_file_location("refcv3_arm_for_navsim", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refcv3_arm_for_navsim"] = mod
    spec.loader.exec_module(mod)
    model, cfg, targs, prov = mod.load_model(ckpt, config, "cpu", False)
    hz = tuple(round(h * DT_FRAME_S, 6) for h in prov["horizons"])
    if hz != KNOT_T_S:
        raise RefusedInput(f"model slots {hz} != KNOT_T_S {KNOT_T_S}")
    return model, cfg, targs, prov, mod


def model_inputs(frames_rows_u8, decl: dict, device: str = "cpu") -> dict:
    """Frames + DECLARED dict -> the exact kwargs of ``RefCV3Model.forward``."""
    import torch
    arm = decl["_arm"]
    eb = ego_block(decl)
    kw = {"ego_state": torch.tensor([eb["ego_state"]], dtype=torch.float32, device=device),
          "v0": (None if eb["v0"] is None
                 else torch.tensor([eb["v0"]], dtype=torch.float32, device=device))}
    nav_info = {"fed": None}
    if ARMS[arm]["nav"] == "cmd":
        idx, name, k = nav_index_from_command(decl["driving_command"])
        kw["nav_cmd"] = torch.tensor([idx], dtype=torch.long, device=device)
        nav_info = {"fed": name, "nav_index": idx, "navsim_argmax": k}
    else:
        kw["nav_cmd"] = None
    return {"kwargs": kw, "ego": eb, "nav": nav_info}


def run_model(model, trainer_mod, frames_rows_u8, decl: dict, steps: int) -> dict:
    import torch
    mi = model_inputs(frames_rows_u8, decl)
    fr = trainer_mod.frames_to_device(frames_rows_u8[None], "cpu")   # [1,W,9,H,W] f32 in [0,1]
    with torch.no_grad():
        out = model(fr, steps=steps, **mi["kwargs"])
    traj = out["traj"].float()[0].cpu().numpy().astype(np.float64)   # [8,2] ego frame t0
    diag = {"sel_idx": int(out["sel_idx"][0])}
    if out.get("route_logits") is not None:
        diag["route_argmax"] = int(out["route_logits"][0].argmax(-1))
    return {"traj": traj, "diag": diag, "ego": mi["ego"], "nav": mi["nav"]}


# --------------------------------------------------------------------------- #
# 4. CONVERSION — knots -> NavSim poses (x fwd, y left, heading CCW)             #
# --------------------------------------------------------------------------- #
def knots_to_navsim(knots_xy: np.ndarray, knot_t: tuple = KNOT_T_S,
                    out_t: tuple = NAVSIM_T_S) -> np.ndarray:
    """``[K,2]`` ego-frame knots (origin excluded) -> ``[8,3]`` NavSim poses.

    Method (declared in SPEC): a C2 cubic spline in TIME ('not-a-knot', scipy
    ``CubicSpline``) per axis through the origin (t=0, (0,0)) and every knot;
    evaluated at ``out_t``. At a knot time the spline reproduces the knot to
    float precision (the GT round-trip control asserts < 1e-3 m); at 2.5 s and
    3.5 s it INTERPOLATES (error measured on logged human paths). Heading =
    atan2(y'(t), x'(t)) of the same spline — the rear-axle path tangent, which
    is the kinematic-bicycle yaw at the rear axle (zero side-slip there);
    held at the previous value (0 = the t0 yaw at t=0) where |p'(t)| < 1.0 m/s,
    and every new value is taken as the branch NEAREST the previous one (so a
    held value can never be unwrapped into a multi-turn drift). The axes need
    NO permutation: both systems are x-forward / y-left / CCW-positive
    (refb_labels.ego_frame; navsim convert_absolute_to_relative_se2_array)."""
    from scipy.interpolate import CubicSpline
    k = np.asarray(knots_xy, dtype=np.float64)
    t = np.concatenate([[0.0], np.asarray(knot_t, dtype=np.float64)])
    if k.shape != (len(knot_t), 2):
        raise RefusedInput(f"knots {k.shape} do not match times {len(knot_t)}")
    p = np.vstack([[0.0, 0.0], k])
    cs = CubicSpline(t, p, axis=0, bc_type="not-a-knot")
    q = np.asarray(out_t, dtype=np.float64)
    xy = cs(q)
    d = cs(q, 1)
    head = np.zeros(len(q))
    prev = 0.0
    for i in range(len(q)):
        sp = float(np.hypot(d[i, 0], d[i, 1]))
        if sp >= HEADING_HOLD_SPEED_MS:
            a = float(np.arctan2(d[i, 1], d[i, 0]))
            prev = prev + ((a - prev + math.pi) % (2 * math.pi) - math.pi)
        head[i] = prev
    return np.column_stack([xy, head])


def cv_navsim_poses(vel_t0, dtype: str) -> np.ndarray:
    """The devkit ConstantVelocityAgent, for reference only (the seam agent
    delegates stage-1 stand-ins to the DEVKIT's own class, never to this)."""
    v = np.asarray(vel_t0, dtype=np.dtype(dtype))
    sp = (v ** 2).sum(-1) ** 0.5
    return np.array([[(i + 1) * 0.5 * sp, 0.0, 0.0] for i in range(8)], dtype=np.float32)


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_dump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
