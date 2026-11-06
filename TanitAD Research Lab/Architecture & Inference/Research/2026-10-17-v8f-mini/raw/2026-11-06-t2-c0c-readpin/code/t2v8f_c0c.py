"""C0C at T2 (AGENT L, 2026-11-06): C0's checkpoint, then the T1-kin COMPOSITE on every replan, using the T1-kin tool's OWN two functions.

The composite (BRIEF_T2V8F_C0C_L.md; PREREG_T2_DOMINANCE.md section 3): plan_c = feasible_plan(smooth_plan(raw, prior, "sg72"), "kbox"), where
  raw   = the model's `traj30` of the forward (float32),
  prior = `out["p10"]` of the SAME forward (float32),
exactly the order `model_planner(..., smooth="sg72", feasible="kbox")` uses in `motion_c/ri/code/v8f_t1_eval_ri.py` lines 626-660 (S first, the projection AFTER it).
The two functions are NOT re-implemented: `load_ri_tool` loads that file BY PATH, sha256 checked (0417c838...), and `composite_plan` calls its `smooth_plan` and `feasible_plan`.
`kbox` = `tanitad.refs.feasible_decode.project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, clamp_entry=False)` (the curvature clamp only), imported by the tool from the tip tree
that the shipped package already carries (`pkg/tree/stack/tanitad/refs/feasible_decode.py`, stage 1's MANIFEST).

At T2 the next frame is rendered at the EXECUTED pose, so the composite changes the video: it is not a post hoc filter of C0's rollouts and has its own cells (C0C_s0, C0C_s1).

Wiring without touching a frozen file: `install(R)` extends `t2v8f_run`'s ARMS / CKPT_FILE and wraps `R.build_arm` BEFORE `R.setup()` registers the arms. Stage 2 and 3 spawn `t2v8f_run.py`
itself, which never imports this module, so their behaviour is unchanged by construction (a test pins the sha256 of every frozen file).
`extra` of every tick keeps all of the inner policy's keys (incl. `s_ckpt`, `s_pose`, which the scorer's S-POS rows read) and adds `plan_raw`, `plan_s`, `composite`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RI_TOOL_SHA = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"
RI_TOOL_DEV = "E:/Projects/TanitAD-artifacts/dagland/motion_c/ri/code/v8f_t1_eval_ri.py"
ARMS_C0C = ("C0C_s0", "C0C_s1")
CKPT_FILE_C0C = {"C0C_s0": "C0_s0.pt", "C0C_s1": "C0_s1.pt"}
SMOOTH, FEASIBLE = "sg72", "kbox"
_TOOL = {}


def sha256_file(path, n: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def ri_tool_path():
    return (os.environ.get("T2V8F_RI_TOOL") or RI_TOOL_DEV).replace("\\", "/")


def load_ri_tool(path=None, tree=None, want=RI_TOOL_SHA):
    """The T1-kin RI tool, by file path, sha256 checked, after the tip tree is on sys.path (assert `tanitad` is inside it). Cached per path."""
    path = (path or ri_tool_path()).replace("\\", "/")
    if path in _TOOL:
        return _TOOL[path]
    import t2a_v8f_policy as P
    P.bootstrap(tree)
    got = sha256_file(path)
    if got != want:
        raise RuntimeError(f"REFUSED: the T1-kin RI tool at {path} is {got}, not the pinned {want}")
    spec = importlib.util.spec_from_file_location("t1tool_ri_c0c", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _TOOL[path] = m
    return m


def composite_plan(raw, prior, smooth=SMOOTH, feasible=FEASIBLE, tool=None):
    """raw, prior: float32 [30, 2] (ego frame at NOW). -> float32 [30, 2], the tool's `feasible_plan(smooth_plan(raw, prior, smooth), feasible)` on a batch of one.
    `smooth` / `feasible` 'off' are the tool's own identity modes (the same object comes back), so both off returns `raw` itself."""
    T = tool or load_ri_tool()
    raw32, prior32 = np.asarray(raw, np.float32), np.asarray(prior, np.float32)
    if raw32.shape != (30, 2) or prior32.shape != (30, 2):
        raise ValueError(f"composite_plan: raw {raw32.shape} and prior {prior32.shape} must be [30, 2]")
    s = T.smooth_plan(raw32[None], prior32[None], smooth)
    return np.asarray(T.feasible_plan(s, feasible)[0], np.float32), np.asarray(s[0], np.float32)


class CompositePolicy:
    """Delegates the kit interface to a MODEL-arm `V8FMiniT2Policy` and returns the composite of its plan over its own prior (like `ZeroPolicy`, a wrapper: no model code is touched)."""
    name = "v8f_c0c"
    ARM_ID = "C0C"

    def __init__(self, inner, smooth=SMOOTH, feasible=FEASIBLE, tool=None, prior_hook=None):
        if inner.arm != "model":
            raise ValueError("CompositePolicy wraps the MODEL arm (its prior is read from the same forward)")
        self.inner, self.smooth, self.feasible, self._tool = inner, smooth, feasible, tool
        self._prior_hook = prior_hook                  # test hook only: the deliberate-regression arm 'prior swapped for zeros'

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def from_forward(self, plan, extra):
        """(plan float64 [30,2] as `forward_batch` returns it, its `extra`) -> (composite float64 [30,2], extra')."""
        raw64 = np.asarray(plan, np.float64)
        prior64 = np.asarray(extra["p10_prior"], np.float64)
        raw32, prior32 = raw64.astype(np.float32), prior64.astype(np.float32)
        if not (np.array_equal(raw32.astype(np.float64), raw64) and np.array_equal(prior32.astype(np.float64), prior64)):
            raise RuntimeError("CompositePolicy: the inner plan / prior are not exact float32 widenings; the T1-kin arithmetic (float32 in, float32 out) would not be reproduced")
        if self._prior_hook is not None:
            prior32 = self._prior_hook(prior32)
        out32, s32 = composite_plan(raw32, prior32, self.smooth, self.feasible, self._tool)
        tool = self._tool or load_ri_tool()
        extra = dict(extra, arm="C0C", plan_raw=raw64.tolist(), plan_s=s32.astype(np.float64).tolist(),
                     composite={"smooth": self.smooth, "feasible": self.feasible, "ri_tool_sha256": RI_TOOL_SHA, "ri_tool_loaded_sha256": sha256_file(tool.__file__)})
        return np.ascontiguousarray(out32, np.float64), extra

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        traj, extra = self.inner.plan(frames, intr, v0, nav_cmd)
        return self.from_forward(traj, extra)


def install(R):
    """Register C0C_s0 / C0C_s1 in `t2v8f_run` (module `R`) before `R.setup()`. Idempotent. Never changes how any existing arm is built."""
    if getattr(R, "_C0C_INSTALLED", False):
        return R
    R.ARMS = tuple(R.ARMS) + tuple(a for a in ARMS_C0C if a not in R.ARMS)
    R.CKPT_FILE.update(CKPT_FILE_C0C)
    orig = R.build_arm

    def build_arm(args):
        pol = orig(args)
        if args.arm in ARMS_C0C:
            pol.inner = CompositePolicy(pol.inner)
        return pol
    R.build_arm = build_arm
    R._C0C_INSTALLED = True
    return R
