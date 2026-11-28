"""AGENT F audit, process A (executor side; the tip tree on sys.path).  CPU only.

For each (scene, f0): the LOGGED future is fed to the real T2 playback executor as the 'plan' (a ReplayPolicy, the T2g control), three replans of 10 knots.
Writes, per case, the arrays the scorer process needs:
  cl        the executed knots in the t0 ego frame (`tool.exec_in_ego(ex.track, t0)`), the T2 record's `cl`
  g         the T2 record's logged future (`t2v8f_exec._ego_knots`)
  g_t1      the SAME logged future by the T1 reader's own path: `tool.exec_in_ego(<logged native rows>, t0)` = `floors_kin.gt_waypoints` on the 0.2 s cache steps
  cl_skew1  the executor with its deliberate-regression knob `skew=1` (the plan's knots rolled by one tick)
  g_shift1  `g` rebuilt one RAW row (0.1 s) late  (a logged-future alignment error of half a knot)
  g_shift2  `g` rebuilt one KNOT (0.2 s) late
  g_yawflip `g` rebuilt with the yaw sign flipped (rotation by +yaw instead of -yaw)
Usage: python -I selfscore_run.py OUT.npz  scene_npz:f0 [scene_npz:f0 ...]
"""
from __future__ import annotations

import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE + "/../code")
import numpy as np  # noqa: E402

import t2v8f_exec as X  # noqa: E402


class ReplayPolicy:
    """Returns the LOGGED plan at NOW (the T2g control): the car then follows the human."""
    window = 17

    def __init__(self, gt_T):
        self.rows = X.logged_rows(gt_T)

    def set_t0(self, gt_T, f0, **k):
        r = self.rows[:f0 + 1]
        self.raw_off = (len(r) - 1) % 2
        self.track = r[self.raw_off:].copy()
        self.log = self.rows[self.raw_off:]

    def set_nav(self, **k):
        pass

    def set_history(self, h, n_new=None):
        self.track = np.vstack([self.track, np.asarray(h)[len(h) - n_new:]])

    def plan(self, frames, intr, v0, nav_cmd):
        assert (len(self.track) - 1) % 2 == 0
        return X.logged_plan(self.log, len(self.track) - 1, pose=self.track[-1]), {}


def ego_knots(rows, j_raw, first_raw, yaw_sign=-1.0):
    """The logged future as 30 knots 0.2 s apart starting at raw row `first_raw`, in the frame of raw row `j_raw` (yaw_sign -1 = the correct rotation by -yaw)."""
    xy = rows[first_raw:first_raw + 2 * 30:2, :2]
    p0 = rows[j_raw]
    c, s = math.cos(yaw_sign * p0[2]), math.sin(yaw_sign * p0[2])
    d = xy - p0[None, :2]
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1)


def run_case(gt_T, ts, f0):
    tool = X.load_tool()
    pol = ReplayPolicy(gt_T)
    out = {}
    rec = X.run_t2_rollout(pol, lambda T, t: None, gt_T, ts, f0, None, nav_cmd=0, condition="T2", tool=tool)
    off, t0 = rec["raw_off"], rec["t0_step"]
    rows = X.logged_rows(gt_T)[off:]
    out["cl"], out["g"] = np.asarray(rec["cl"]), np.asarray(rec["g"])
    out["g_t1"] = np.asarray(tool.exec_in_ego(rows, t0))              # the T1 reader's path on the same logged track
    out["track"] = np.asarray(rec["track"])
    out["rows"] = rows
    out["t0"], out["off"] = np.array([t0]), np.array([off])
    rec2 = X.run_t2_rollout(ReplayPolicy(gt_T), lambda T, t: None, gt_T, ts, f0, None, nav_cmd=0, condition="T2", skew=1, tool=tool)
    out["cl_skew1"] = np.asarray(rec2["cl"])
    j = 2 * t0
    out["g_shift1"] = ego_knots(rows, j, j + 3)
    out["g_shift2"] = ego_knots(rows, j, j + 4)
    out["g_yawflip"] = ego_knots(rows, j, j + 2, yaw_sign=+1.0)
    out["g_rebuilt"] = ego_knots(rows, j, j + 2)
    return out


def main(argv):
    out_path, cases = argv[0], argv[1:]
    res = {}
    for i, c in enumerate(cases):
        p, f0 = c.rsplit(":", 1)
        z = np.load(p)
        o = run_case(z["gt_T"], z["gt_ts_us"], int(f0))
        for k, v in o.items():
            res[f"c{i}_{k}"] = v
        res[f"c{i}_name"] = np.array([c])
    res["n_cases"] = np.array([len(cases)])
    np.savez(out_path, **res)
    print("wrote", out_path, len(cases), "cases")


if __name__ == "__main__":
    main(sys.argv[1:])
