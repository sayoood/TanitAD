"""E3 -- the playback executor and the T2 re-perception rollout loop for v8F-mini (conditions T2 and RR).

PLAYBACK: the car's pose follows the plan's knots exactly. The executor CALLS the T1-kin tool's own functions (`envelope`, `plan_to_world`,
`hermite_frames`, `exec_in_ego`, `reindex_rows`), loaded by pinned path, so the executed track is the one `WindowState.commit_plan` writes.
Each start: the logged history (rows <= f0), then `n_replans` x `m` executed knots at 0.2 s (3 x 10 = 6 s). The harness' bicycle actuator is untouched.

Conditions (render at each 0.2 s grid tick; the odd 0.1 s ticks are never read by the policy, so they carry `None`):
  T2      the camera at the EXECUTED pose and time (z / roll / pitch from the logged road surface, the harness' `GroundFollower`).
  RR      the camera at the LOGGED pose whose along-track arc equals the car's own executed arc, at that logged time: the tool's `reindex_rows`
          on the executed grid positions, anchored at t0, tau(i) = max(tau(i-1), min(i, tau*(i))). `logged` (control) renders the logged frame i.
The warm-up frames (the logged past) are rendered at the logged poses in every condition. Every `set_history` call is made on EVERY raw tick.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from collections import deque

import numpy as np

TOOL = "E:/Projects/TanitAD-artifacts/dagland/motion_c/lf6_build/tool/v8f_t1_eval_ri_lf6.py"
TOOL_SHA = "949c4b7a384943015a4b520987bc9c4267dee4a2160f566aa2225208ed3f7f01"
CONDITIONS = ("T2", "RR", "logged")
_TOOL = {}


def load_tool(tree=None):
    """The reindexed T1-kin tool, by path, sha256 checked, after the tip tree is on sys.path (assert `tanitad` is not on G:)."""
    if "m" in _TOOL:
        return _TOOL["m"]
    import t2a_v8f_policy as P
    P.bootstrap(tree)
    h = hashlib.sha256(open(TOOL, "rb").read()).hexdigest()
    if h != TOOL_SHA:
        raise RuntimeError(f"REFUSED: the T1 tool is {h}, not the pinned {TOOL_SHA}")
    spec = importlib.util.spec_from_file_location("t1tool_exec", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _TOOL["m"] = m
    return m


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


def rz3(a):
    """The 3x3 yaw rotation (the harness' `_rz`). `rz` above is the 4x4 pose form: the two must not be mixed in a product."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def ground_correct(gt_T, xy, yaw):
    """The harness' `GroundFollower.correct`, line for line: x / y / yaw are the car's, z / roll / pitch are the nearest logged rig pose's.
    The rotation product is 3x3 x 3x3 (the first version multiplied the 4x4 `rz` with a 3x3 block and raised ValueError on the first T2 step)."""
    P = np.stack([T[:2, 3] for T in gt_T])
    G = gt_T[int(np.argmin(np.linalg.norm(P - np.asarray(xy)[None, :2], axis=1)))]
    out = np.eye(4)
    out[:3, :3] = rz3(yaw) @ (rz3(-yaw_of(G)) @ G[:3, :3])
    out[:3, 3] = (xy[0], xy[1], G[2, 3])
    return out


class PlaybackExecutor:
    """Owns the raw 10 Hz executed track (x, y, yaw, v); NOW is the last row and an even index. `commit` = `WindowState.commit_plan`."""

    def __init__(self, track_rows, tool=None, skew: int = 0):
        self.tool = tool or load_tool()
        self.track = np.asarray(track_rows, np.float64).copy()
        self.skew = int(skew)                                   # test hook only: a deliberate-regression arm shifts the knots

    @property
    def j(self):
        return (len(self.track) - 1) // 2

    def commit(self, plan_ego, m: int = 10):
        T, j = self.tool, self.j
        pose = self.track[2 * j, :3]
        plan_ego, self.hit = T.envelope(plan_ego)
        W = T.plan_to_world(plan_ego, pose)
        if self.skew:
            W = np.roll(W, self.skew, axis=0)
        knots = np.vstack([pose[None, :2], W])
        xy, psi, v = T.hermite_frames(self.track[2 * (j - 1), :2], knots, m, float(pose[2]))
        new = np.column_stack([xy, psi, v])
        self.track = np.vstack([self.track, new])
        return new


def logged_rows(gt_T):
    """(x, y, yaw, v) per logged 10 Hz row, v by forward difference (the policy's `set_t0` spelling)."""
    r = np.stack([[T[0, 3], T[1, 3], yaw_of(T), 0.0] for T in gt_T])
    d = np.linalg.norm(np.diff(r[:, :2], axis=0), axis=1) / 0.1
    r[:-1, 3] = d
    r[-1, 3] = d[-1] if len(d) else 0.0
    return r


def logged_plan(rows, j_raw, n=30, pose=None):
    """The logged future as a plan: knots j+1..j+n (0.2 s apart) in the ego frame at raw row `j_raw` (the control used by gate T2g)."""
    p0 = rows[j_raw] if pose is None else pose            # `pose`: the frame of the policy's OWN pose (the executed yaw), not the logged one
    c, s = math.cos(p0[2]), math.sin(p0[2])
    d = rows[j_raw + 2:j_raw + 2 * n + 1:2, :2] - p0[None, :2]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def run_t2_rollout(policy, render, gt_T, gt_ts_us, f0, intr, nav_row=None, nav_cmd=0, condition="T2", n_replans=3, m=10,
                   skew=0, rr_mode="reindex", tool=None):
    """One start. `render(T_rig_world, t_us)` -> frame (the caller binds the camera mount: `transport.render(T @ Ts_cam, t)`).
    Returns a record with the plans, the executed track, every render call (grid tick, pose, time, tau) and the T1-window arrays (`cl`, `g`, `v0`)."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    tool = tool or load_tool()
    need = int(policy.window) + 2
    if f0 - (need - 1) < 0 or f0 + 2 * n_replans * m >= len(gt_T) + 10 ** 6:
        raise ValueError("start too early for the observation window")
    policy.set_t0(gt_T, f0)
    if nav_row is not None:
        policy.set_nav(row=nav_row)
    else:
        policy.set_nav(nav_cmd=nav_cmd)
    off = policy.raw_off                                         # raw row r of the policy track = logged row r + off
    frames, calls = deque(maxlen=need), []
    for k in range(need):                                        # force-GT warm-up: the logged past at the logged poses
        f = f0 - (need - 1) + k
        if (need - 1 - k) % 2:
            frames.append(None)
            continue
        frames.append(render(gt_T[f], gt_ts_us[f]))
        calls.append({"phase": "warm", "f": f, "pose": gt_T[f].copy(), "t_us": float(gt_ts_us[f])})
    ex = PlaybackExecutor(policy.track, tool, skew)
    t0 = ex.j
    xy_log = np.stack([[gt_T[2 * s + off][0, 3], gt_T[2 * s + off][1, 3]] for s in range((len(gt_T) - 1 - off) // 2 + 1)])
    rec = {"f0": f0, "t0_step": t0, "raw_off": off, "plans": [], "extra": [], "condition": condition}
    for r in range(n_replans):
        v0 = float(ex.track[-1, 3])
        traj, extra = policy.plan(list(frames), intr, v0, nav_cmd)
        rec["plans"].append(np.asarray(traj, np.float64).copy())
        rec["extra"].append(extra)
        n_old = len(ex.track)
        new = ex.commit(traj, m)
        for q in range(len(new)):
            n_now = n_old + q                                    # raw index of this executed row
            policy.set_history(ex.track[:n_now + 1][-16:], n_new=1)          # on EVERY raw tick
            if n_now % 2:
                frames.append(None)
                continue
            i = n_now // 2
            if condition == "T2":
                T = ground_correct(gt_T, ex.track[n_now, :2], float(ex.track[n_now, 2]))
                t_us, tau = float(gt_ts_us[f0] + 1e5 * (n_now - 2 * t0)), None
            elif condition == "logged":
                fl = 2 * i + off
                T, t_us, tau = gt_T[fl], float(gt_ts_us[fl]), i
            else:
                own = ex.track[0::2, :2][t0:i + 1]
                if rr_mode == "time":                            # deliberate-regression arm: the index is the clock, not the arc
                    tau = i
                else:
                    tau = int(tool.reindex_rows(np.arange(t0, i + 1), t0, own, xy_log)["tau"][-1])
                fl = 2 * tau + off
                T, t_us = gt_T[fl], float(gt_ts_us[fl])
            frames.append(render(T, t_us))
            calls.append({"phase": "exec", "i": i, "tau": tau, "pose": np.array(T), "t_us": t_us})
    rec.update({"track": ex.track.copy(), "calls": calls, "policy_track_equals_executor": bool(np.array_equal(policy.track, ex.track))})
    rows = logged_rows(gt_T)[off:]
    rec["g"] = _ego_knots(rows, 2 * t0, rows[2 * t0 + 2: 2 * t0 + 2 * 30 + 1: 2, :2])
    rec["cl"] = tool.exec_in_ego(ex.track, t0)
    rec["v0"] = float(ex.track[2 * t0, 3])
    return rec


def _ego_knots(rows, j_raw, xy):
    p0 = rows[j_raw]
    c, s = math.cos(p0[2]), math.sin(p0[2])
    d = xy - p0[None, :2]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
