"""The replan STATE the T1 tool rebuilds, computed with the tool's OWN functions (not re-implemented): ``v8f_t1_eval.hermite_frames`` fills the 10 Hz
frames from the plan's 0.2 s knots, ``v8f_mini_data.yaw_kin_from_track`` turns them into kin3 = (v0, a0, kappa0) (one-frame backward differences at
frame 2j, kappa0 = clamp(omega0 / max(v0, 2), +-0.3)). At a replan after m = 10 executed knots the state reads frames 2j-1 and 2j only, i.e. plan
knots 8, 9, 10, 11 (knot 11 is NOT executed: it sets the Catmull-Rom tangent at knot 10)."""
import importlib.util
import math
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/t1k/stack")
from tanitad.data import v8f_mini_data as D  # noqa: E402

_spec = importlib.util.spec_from_file_location("t1tool", "C:/Users/Admin/t1k/stack/scripts/v8f_t1_eval.py")
T1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T1)


def replan_state(plan, pose=(0.0, 0.0, 0.0, 0.0), k_prev=None, m=10):
    """kin3 (v0, a0, kappa0) and the frame-2j (psi, v) the tool would hold after executing the first m knots of ``plan`` [30, 2] (ego frame at ``pose``),
    exactly as WindowState.commit_plan + _finish compute them. ``k_prev`` affects interval 1 only (never frames 2m-1, 2m for m >= 3)."""
    pose = np.asarray(pose, np.float64)
    W = T1.plan_to_world(np.asarray(plan, np.float64), pose)
    knots = np.vstack([pose[None, :2], W])
    if k_prev is None:
        c, s = math.cos(pose[2]), math.sin(pose[2])
        k_prev = pose[:2] - np.array([c, s]) * max(np.linalg.norm(W[0]), 1e-3)     # a straight-back stand-in (irrelevant for m >= 3)
    xy, psi, v = T1.hermite_frames(k_prev, knots, m, float(pose[2]))
    tr = np.zeros((2 * m + 1, 4))
    tr[0] = [pose[0], pose[1], pose[2], pose[3]]
    tr[1:, :2], tr[1:, 2], tr[1:, 3] = xy, psi, v
    kin, om = D.yaw_kin_from_track(tr)
    return kin[m], tr[2 * m], tr


def state_from_knots(Kfull, start, m=10):
    """kin3 at the replan that follows executing knots start+1 .. start+m of the WORLD knot sequence ``Kfull`` [N, 2] from the pose at ``Kfull[start]``
    (any rigid frame: v, a, kappa are frame-invariant). Needs Kfull[start + m + 1]. Returns (kin3, frame-2j row (x, y, psi, v))."""
    K = np.asarray(Kfull, np.float64)
    k_prev = K[start - 1] if start >= 1 else K[start] - (K[start + 1] - K[start])
    d = K[start + 1] - K[start]
    psi0 = math.atan2(d[1], d[0])
    xy, psi, v = T1.hermite_frames(k_prev, K[start:], m, psi0)
    tr = np.zeros((2 * m + 1, 4))
    tr[0] = [K[start, 0], K[start, 1], psi0, 0.0]
    tr[1:, :2], tr[1:, 2], tr[1:, 3] = xy, psi, v
    kin, _ = D.yaw_kin_from_track(tr)
    return kin[m], tr[2 * m]
