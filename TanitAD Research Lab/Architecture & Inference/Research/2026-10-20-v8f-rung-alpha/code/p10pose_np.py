"""v8F alpha-1 (pose-only variant, mode `p10_pose`): the numpy reference side -- features, windows, floors.

The pose kinematics are the SAME derivation `ha0_ext_pose` uses (`tanitad.models.ego_history.ego_channels_from_poses`
-> `kinematic_prior.prior_controls`): v0 = v[t0]; a0 = (v[t0] - v[t0-1]) / 0.1 (ONE frame); omega0 = wrap(yaw[t0] -
yaw[t0-1]) / 0.1; kappa0 = clip(omega0 / max(v0, POSE_KAPPA_V_FLOOR), +-POSE_KAPPA_CAP). Nothing recorded after t0 and no
steer / action channel enters. The third feature of P10's eight ("k") is this kappa0 (1/m) instead of the steer angle, and
the base path integrates it as a CURVATURE (`action_units="kappa"`).
This file is the independent reference the torch port is pinned to; the S1 apples-to-apples pre-check uses the S1 windows.
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys

import numpy as np

ROOT = os.environ.get("V8FA1_TREE", "C:/Users/Admin/v8fa1/tree")
for p in (f"{ROOT}/stack", f"{ROOT}/taniteval"):
    if p not in sys.path:
        sys.path.insert(0, p)
from taniteval import floors_kin as fk                                       # noqa: E402
from tanitad.models.kinematic_prior import POSE_KAPPA_CAP, POSE_KAPPA_V_FLOOR  # noqa: E402

S1_SCRATCH = "C:/Users/Admin/s1_floors_scratch"
TRAIN_MAN = f"{S1_SCRATCH}/train4572_v2manifest.pt"
EVAL_MAN = f"{S1_SCRATCH}/eval139_v2manifest.pt"
BATT = "E:/refcv7_eval_kit/battery/step50400_a6text"      # banked refcv7-50,400 T1 dumps (DATA, read-only)
H_S2 = [5, 10, 15, 20]
H_S6 = [10, 20, 30, 40, 50, 60]
DT_FRAME = 0.1


def wrap_pi(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def pose_kin(v, yaw, t0, *, dt: float = DT_FRAME, stride: int = 1) -> np.ndarray:
    """``[n, 3]`` = (v0, a0, kappa0) from the POSE track at frames ``t0`` -- every index <= t0."""
    t0 = np.asarray(t0, np.int64)
    if t0.size and int(t0.min()) < stride:
        raise ValueError(f"pose_kin needs frame t0-{stride} inside the window (min t0={int(t0.min())})")
    v = np.asarray(v, np.float64)
    yaw = np.asarray(yaw, np.float64)
    v0 = v[t0]
    a0 = (v0 - v[t0 - stride]) / (stride * dt)
    om = wrap_pi(yaw[t0] - yaw[t0 - stride]) / (stride * dt)
    kap = np.clip(om / np.maximum(v0, POSE_KAPPA_V_FLOOR), -POSE_KAPPA_CAP, POSE_KAPPA_CAP)
    return np.stack([v0, a0, kap], axis=-1)


def load_manifest(path: str) -> dict:
    import torch
    m = torch.load(path, map_location="cpu", weights_only=False)
    return {"clip_id": [str(c) for c in m["clip_id"]],
            "poses": [p.numpy().astype(np.float64) for p in m["poses"]],
            "actions": [a.numpy().astype(np.float64) for a in m["actions"]],
            "md5": hashlib.md5(open(path, "rb").read()).hexdigest()}


def pose_floors(kin: np.ndarray, n_fine: int, sel: list[int]) -> dict:
    """ha0, ha0_ext_pose, kd_x_pose at the grid instants ``sel`` (frame offsets), the S1 way: integrate on the NATIVE
    0.1 s polyline (``n_fine`` ticks), re-time on it, then index-select."""
    idx = [h - 1 for h in sel]
    h0 = fk.ha0_paths(kin[:, 0], DT_FRAME, n_fine)
    he = fk.const_control_paths(kin[:, 0], kin[:, 1], kin[:, 2], DT_FRAME, n_fine, action_units="kappa")
    kdx = fk.kd_x(h0, he).astype(np.float64)
    return {"ha0": h0[:, idx], "ha0_ext_pose": he[:, idx], "kd_x_pose": kdx[:, idx]}


def eval_windows(man: dict, tag: str, sel: list[int]) -> dict:
    """The banked dump's own windows (``ws`` = provider t0) with POSE kinematics rebuilt from the manifest."""
    files = sorted(glob.glob(f"{BATT}/{tag}/ep*.npz"))
    cols: dict[str, list] = {}
    worst = {"v0": 0.0, "g": 0.0}
    for f in files:
        z = np.load(f)
        ci = int(z["clip_index"][0])
        P = man["poses"][ci]
        t0 = z["ws"].astype(np.int64)
        kin = pose_kin(P[:, 3], P[:, 2], t0)
        g = fk.gt_waypoints(P, t0, sel)
        worst["v0"] = max(worst["v0"], float(np.abs(kin[:, 0] - z["v0"]).max()))
        worst["g"] = max(worst["g"], float(np.abs(g - z["g"]).max()))
        cols.setdefault("kin", []).append(kin)
        cols.setdefault("t0", []).append(t0)
        cols.setdefault("clip", []).append(np.full(len(t0), ci))
        for k in z.files:
            if k not in ("eid", "clip_index", "ws"):
                cols.setdefault("d_" + k, []).append(z[k])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["ep"] = np.concatenate([np.full(len(c), i) for i, c in enumerate(cols["t0"])])
    out["files"] = files
    out["check"] = worst
    return out


def train_windows(man: dict, sel: list[int], *, stride: int = 5, first: int = 7) -> dict:
    """TRAIN windows, the S1 rule: per episode e, ``t0 = first + (e % stride) + stride*j`` for every t0 with
    ``t0 + max(sel) <= T-1``. POSE kinematics + futures only (no frame, no label, no steer)."""
    kins, gs, eps, t0s = [], [], [], []
    last = max(sel)
    for e, P in enumerate(man["poses"]):
        T = P.shape[0]
        t0 = np.arange(first + (e % stride), T - 1 - last + 1, stride)
        if t0.size == 0:
            continue
        kins.append(pose_kin(P[:, 3], P[:, 2], t0))
        gs.append(fk.gt_waypoints(P, t0, sel))
        eps.append(np.full(len(t0), e))
        t0s.append(t0)
    return {"kin": np.concatenate(kins), "g": np.concatenate(gs), "ep": np.concatenate(eps), "t0": np.concatenate(t0s)}
