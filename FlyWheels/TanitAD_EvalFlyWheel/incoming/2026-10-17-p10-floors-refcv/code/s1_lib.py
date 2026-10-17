"""S1 (H-V8F-FLOOR-1) shared builders: v2 manifests -> per-window kinematics, GT and floors, on two grids.

* refcv grid  (refcv3_arm.py): native 0.1 s tick, ``t0`` is a provider index into the manifest arrays,
  a0 = (v[t0]-v[t0-1])/0.1, k0 = actions[t0,0] (STEER), floors integrated with action_units="steer".
  S2 = 4 steps x 0.5 s (frame offsets 5,10,15,20), S6 = 6 steps x 1.0 s (offsets 10..60).
* refav1 grid (extract.py, the P10 package): operative tick 0.2 s, ``t`` an operative index, origin raw frame
  i0 = 2t, a0 = (v[i0]-v[i0-2])/0.2, k0 = actions[i0,0] (STEER), floors integrated with action_units="kappa"
  (extract.py's default -- the over-rotation defect is part of that package's banked arrays and is reproduced,
  not repaired, because this grid exists only to reproduce the known value 2.9257 m).
  The manifest arrays are raw[2:] (n_stack=3), so raw index j == manifest index j-2.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

ROOT = "E:/Projects/TanitAD"
for p in (f"{ROOT}/stack", f"{ROOT}/taniteval"):
    if p not in sys.path:
        sys.path.insert(0, p)
from taniteval import floors_kin as fk                                       # noqa: E402

SCRATCH = "C:/Users/Admin/s1_floors_scratch"
TRAIN_MAN = f"{SCRATCH}/train4572_v2manifest.pt"
EVAL_MAN = f"{SCRATCH}/eval139_v2manifest.pt"
PICK = (f"{ROOT}/TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refav1-trunk-probe/raw/train/"
        "train_pick_sha12.json")
BATT = "E:/refcv7_eval_kit/battery/step50400_a6text"

H_S2 = [5, 10, 15, 20]                 # frame offsets of the S2 grid (0.5 s steps)
H_S6 = [10, 20, 30, 40, 50, 60]        # frame offsets of the S6 grid (1.0 s steps)
DT_FRAME = 0.1


def load_manifest(path: str) -> dict:
    import torch
    m = torch.load(path, map_location="cpu", weights_only=False)
    return {"clip_id": list(m["clip_id"]),
            "poses": [p.numpy().astype(np.float64) for p in m["poses"]],
            "actions": [a.numpy().astype(np.float64) for a in m["actions"]],
            "n_stack": list(m["n_stack"]), "md5": hashlib.md5(open(path, "rb").read()).hexdigest()}


# ----------------------------------------------------------------------------------------------- refcv grid ---
def refcv_floors(kin: np.ndarray, n_fine: int, sel: list[int]) -> dict:
    """ha0, ha0_ext, damp50, kd_x at the grid instants ``sel`` (frame offsets), from the NATIVE 0.1 s polyline.

    All four are integrated on ``n_fine`` ticks (60 for S6, 20 for S2), ``kd_x`` is re-timed on that fine
    polyline, then everything is index-selected -- the programme's rule (``integrate_select``): no path is resampled.
    """
    idx = [h - 1 for h in sel]
    h0 = fk.ha0_paths(kin[:, 0], DT_FRAME, n_fine)
    he = fk.ha0_ext_paths(kin, DT_FRAME, n_fine, action_units="steer")
    kdx = fk.kd_x(h0, he).astype(np.float64)
    d50 = fk.damp50(h0, he)
    return {"ha0": h0[:, idx], "ha0_ext": he[:, idx], "damp50": d50[:, idx], "kd_x": kdx[:, idx]}


def refcv_eval_windows(man: dict, batt_dir: str, tag: str, sel: list[int]) -> dict:
    """The dump's own windows (``ws`` = provider t0), with kinematics rebuilt from the manifest.

    ``tag``: ``dump_s0`` / ``dump_s1`` (S2) or ``s6_s0`` / ``s6_s1`` (S6). Returns per-window arrays plus the
    dump's arms; asserts v0/g reproduce the dump's exactly (to float32 tolerance).
    """
    import glob
    files = sorted(glob.glob(f"{batt_dir}/{tag}/ep*.npz"))
    cols: dict[str, list] = {}
    worst = {"v0": 0.0, "g": 0.0}
    for f in files:
        z = np.load(f)
        ci = int(z["clip_index"][0])
        P, A = man["poses"][ci], man["actions"][ci]
        t0 = z["ws"].astype(np.int64)
        kin = fk.t0_kinematics(P[:, 3], A[:, 0], t0, stride=1, dt=DT_FRAME)
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
    # episode id = ordinal of the dump file (the same ordering the battery's eid_w uses: ep000, ep001, ...)
    out["ep"] = np.concatenate([np.full(len(c), i) for i, c in enumerate(cols["t0"])])
    out["files"] = files
    out["check"] = worst
    return out


def refcv_train_windows(man: dict, sel: list[int], *, stride: int = 5, first: int = 7,
                        need_future: int | None = None) -> dict:
    """TRAIN windows: per episode e, ``t0 = first + (e % stride) + stride*j`` for every t0 with
    ``t0 + max(sel) <= T-1`` (the S6 rule; the eval windows obey the same bound). Kinematics + 6 s futures only."""
    kins, gs, eps, t0s = [], [], [], []
    last = max(sel)
    for e, (P, A) in enumerate(zip(man["poses"], man["actions"])):
        T = P.shape[0]
        t0 = np.arange(first + (e % stride), T - 1 - last + 1, stride)
        if t0.size == 0:
            continue
        kins.append(fk.t0_kinematics(P[:, 3], A[:, 0], t0, stride=1, dt=DT_FRAME))
        gs.append(fk.gt_waypoints(P, t0, sel))
        eps.append(np.full(len(t0), e))
        t0s.append(t0)
    return {"kin": np.concatenate(kins), "g": np.concatenate(gs), "ep": np.concatenate(eps), "t0": np.concatenate(t0s)}


# ---------------------------------------------------------------------------------------------- refav1 grid ---
def refav1_episode(P_man: np.ndarray, A_man: np.ndarray, stride: int, W: int = 4, K: int = 30, dt: float = 0.2,
                   k2: int = 10) -> dict | None:
    """One episode on the refav1 operative grid (extract.py's definitions). ``P_man`` = raw[2:]."""
    pad = np.full((2, P_man.shape[1]), np.nan)
    P = np.concatenate([pad, P_man])
    A = np.concatenate([np.full((2, A_man.shape[1]), np.nan), A_man])
    T = P.shape[0]
    t_c = (T + 1) // 2
    ts = np.array([t for t in range(W - 1, t_c - K - 1) if (t - (W - 1)) % stride == 0], dtype=np.int64)
    if ts.size == 0:
        return None
    i0 = 2 * ts
    kin = fk.t0_kinematics(P[:, 3], A[:, 0], i0, stride=2, dt=dt)
    g = fk.gt_waypoints(P, i0, list(range(2, 2 * k2 + 1, 2)))
    ok = i0 + 2 * K <= T - 1
    g6 = np.full((len(ts), K, 2), np.nan)
    if ok.any():
        g6[ok] = fk.gt_waypoints(P, i0[ok], list(range(2, 2 * K + 1, 2)))
    h0_2, h0_6 = fk.ha0_paths(kin[:, 0], dt, k2), fk.ha0_paths(kin[:, 0], dt, K)
    he_2 = fk.ha0_ext_paths(kin, dt, k2, action_units="kappa")
    he_6 = fk.ha0_ext_paths(kin, dt, K, action_units="kappa")
    return {"kin": kin, "g": g, "g6": g6, "g6_ok": ok, "ha0": h0_2, "ha0_ext": he_2, "ha0_6": h0_6,
            "ha0_ext_6": he_6, "ws": ts}


def pick_train_ids(man: dict) -> list[str]:
    """The 600 clips of the P10 package: first N of (B1 listing - eval141) by sha256(clip)[:12] (pick JSON)."""
    want = set(json.load(open(PICK))["sha12"])
    sh = {hashlib.sha256(c.encode()).hexdigest()[:12]: c for c in man["clip_id"]}
    return sorted(sh[s] for s in want)
