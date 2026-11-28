"""E4 (part) -- the arms of the T2 re-perception rollout as kit policies. Registration into `closedloop_drive.EXTRA_ARMS` is `register()` (entry
script `t2v8f_run.py`, open item in HANDOFF_E4E6.md); this file holds what is already built and gated:
  ZeroPolicy   the ZERO arm: wraps the model arm and returns the prior the same forward emits. T3 requires ZERO == P10 (policy arm 'p10') exactly.
No kit file is edited; the historical arms are untouched.
"""
from __future__ import annotations

import numpy as np

import t2a_v8f_policy as P


class ZeroPolicy:
    """Delegates the whole kit interface to a model-arm policy and returns `extra['p10_prior']` as the plan (a different code path from arm='p10')."""
    name = "v8f_zero"
    ARM_ID = "ZERO"

    def __init__(self, inner: "P.V8FMiniT2Policy", skew: float = 0.0):
        if inner.arm != "model":
            raise ValueError("ZeroPolicy wraps the MODEL arm (its prior is read from the same forward)")
        self.inner, self._skew = inner, float(skew)            # `skew`: test hook for the deliberate-regression arm only

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        _, extra = self.inner.plan(frames, intr, v0, nav_cmd)
        out = np.ascontiguousarray(np.asarray(extra["p10_prior"], np.float64)) + self._skew
        extra = dict(extra, arm="ZERO")
        return out, extra


# ================================================================================================================================
# E4 (nav) -- the T1 tool's route input, built from a logged route with the v9 BUILDER's own functions
# ================================================================================================================================
import hashlib
import importlib.util
import os
import sys

V9_DIR = os.environ.get("T2V8F_V9_DIR", "F:/TanitAD-scratch/t2v8f/agentA/v9")
V9_PINS = {"build_v9_labels.py": "3a96dfd6a30c74d67f52f04f838c6b14108eadc7dee97ed040fc0b6bfd1e750a",    # git 24b97ba9:stack/scripts/
           "s2_geom_emit_v7.py": "6b7b8af41330407c0343c1e98a80c0b0fef67e7d2ad771c05f47c02e65c31400"}
_V9 = {}


def load_v9(v9_dir=None, tree=None):
    """The v9 label builder (`build_v9_labels`, which imports `s2_geom_emit_v7`), loaded by path, sha256 checked, after the tip tree is on sys.path."""
    if "m" in _V9:
        return _V9["m"]
    P.bootstrap(tree)
    d = v9_dir or V9_DIR
    for fn, want in V9_PINS.items():
        got = hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest()
        if got != want:
            raise RuntimeError(f"REFUSED: {fn} is {got}, not the pinned {want}")
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("build_v9_labels", os.path.join(d, "build_v9_labels.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["build_v9_labels"] = m
    spec.loader.exec_module(m)
    _V9["m"] = m
    return m


def join_row(token, d_next_m, valid):
    """`v8f_mini_data.build_label_join`'s spelling of the join column `nav` (lines 207-214): one-hot(clip(token, 0, 2)), d/100 clipped to [0, 3], valid."""
    ok = bool(valid == 1) and bool(np.isfinite(d_next_m))
    r = np.zeros(5, np.float32)
    r[int(np.clip(int(token), 0, 2))] = 1.0
    r[3] = np.clip(d_next_m / 100.0, 0.0, 3.0) if ok else 0.0
    r[4] = ok
    return r


def route_geometry(ts_s, x, y, yaw, v, suppression=None, v9=None):
    """The builder's whole-recording geometry for a logged route: the `EgoLog`, the 10 Hz track and the turn table (what `clip_geo` builds, minus the smoothed paths)."""
    B = v9 or load_v9()
    log = B.log_from_arrays(np.asarray(ts_s, np.float64), np.asarray(x, np.float64), np.asarray(y, np.float64), np.unwrap(np.asarray(yaw, np.float64)),
                            np.asarray(v, np.float64))
    tr = B.track10(log)
    return log, tr, B.turn_table(log, tr, suppression), float(tr.s[-1])


def nav_at(geo, now_s, v9=None):
    """The 5-vector nav row at raw time `now_s`, exactly the builder's `nav_fields` at (s_now, now, v_now, s_max), then `join_row`."""
    B = v9 or load_v9()
    log, tr, turns, s_max = geo
    v_now = float(B.sample(log, now_s)[3])
    f = B.nav_fields(turns, float(B.s_at(tr, now_s)), float(now_s), v_now, s_max)
    return join_row(f["nav_token"], f["nav_d_next_m"], f["nav_args_valid"])


def scene_track(gt_T, gt_ts_us):
    """A scene's logged route as the arrays the builder's `log_from_arrays` takes. x, y, yaw from the rig poses; speed = central difference of the 10 Hz track
    (the rig poses carry no velocity channel); time = seconds from the first pose."""
    gt_T = list(gt_T)
    x = np.array([T[0, 3] for T in gt_T])
    y = np.array([T[1, 3] for T in gt_T])
    yaw = np.unwrap([math.atan2(T[1, 0], T[0, 0]) for T in gt_T])
    ts = (np.asarray(gt_ts_us, np.float64) - float(gt_ts_us[0])) / 1e6
    v = np.gradient(np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])) .cumsum(), ts)
    return ts, x, y, yaw, v


def nav_table(gt_T, gt_ts_us, suppression=None, v9=None):
    """[n_ticks, 5] float32: the nav row at every logged 10 Hz tick of a scene (NO v9 release row exists for a scene outside the release; this is the
    construction that makes one: the builder's turn table on the logged route, `nav_fields` per tick, `join_row`)."""
    ts, x, y, yaw, v = scene_track(gt_T, gt_ts_us)
    geo = route_geometry(ts, x, y, yaw, v, suppression, v9)
    return np.stack([nav_at(geo, float(t), v9) for t in ts]), geo


# ================================================================================================================================
# E4 (arms) -- BLIND, the nav-injecting wrapper, the registry
# ================================================================================================================================
import math  # noqa: E402


class NavPolicy:
    """Wraps any kit policy so `set_t0` also hands it the scene's logged nav rows for the steps <= t0 (the tool reads the join row of each step), and `set_nav`
    is fed the row at t0 by the caller. Everything else is delegated."""

    def __init__(self, inner, table):
        self.inner, self.table = inner, np.asarray(table, np.float32)

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def set_t0(self, gt_T, f0, nav_rows=None, nav_cmd=None):
        off = int(f0) % 2                                    # the policy trims one leading row when len(rows)-1 is odd
        t0 = (int(f0) - off) // 2
        rows = self.table[[2 * s + off for s in range(t0 + 1)]]
        self.inner.set_t0(gt_T, f0, nav_rows=rows)
        self.nav_row_t0 = rows[-1].copy()


class BlindPolicy:
    """BLIND arm: the model arm with every scene-token row replaced by the banked mean tokens (`mean` [40, 1024] fp16). The camera is never read."""
    name, ARM_ID = "v8f_blind", "BLIND"

    def __init__(self, inner, mean):
        self.inner, self.mean = inner, mean
        if inner.encoder is not None:
            raise ValueError("BLIND wraps a policy WITHOUT an encoder")

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        p = self.inner
        j = (len(p.track) - 1) // 2
        for k in range(p.hist + p.cfg.max_age_s):
            if j - k >= 0 and (j - k) not in p.tok:
                p.push_tokens(j - k, self.mean)
        traj, extra = p.plan(None, intr, v0, nav_cmd, nav_known)
        return traj, dict(extra, arm="BLIND")


def blind_mean(cache_clips_dir, grid="pool4x10"):
    """Per-position mean of the scene tokens over every frame of every clip in a cache split -> (float16 [40, 1024], n_frames, n_clips). float64 accumulation."""
    import glob
    import torch
    tot, n, nc = None, 0, 0
    for f in sorted(glob.glob(os.path.join(cache_clips_dir, "*.pt"))):
        o = torch.load(f, map_location="cpu", weights_only=False)
        t = o[grid].to(torch.float32).double()
        tot = t.sum(0) if tot is None else tot + t.sum(0)
        n, nc = n + t.shape[0], nc + 1
    return (tot / n).to(torch.float16), n, nc


# ================================================================================================================================
# E4 (registry and starts)
# ================================================================================================================================
import json  # noqa: E402

ART = os.environ.get("T2V8F_ART", "E:/Projects/TanitAD-artifacts")
C0_DIR = {0: "v8f-recover/M2b_s0_recft_C0", 1: "v8f-recover/M2b_s1_recft_C0"}
SCREEN_TABLE = os.environ.get("T2V8F_SCREEN_TABLE", f"{ART}/dagland/motion_c/t2_scenes/raw/SCREEN_TABLE.json")


def init_ckpt_from_argv(argv_json, root=None):
    """C0's `--init-ckpt` (DAG-OACC) as a path under `root`, matched on its last two components. None when the named checkpoint is absent -> the arm is 'beside, not built'."""
    a = json.load(open(argv_json))["argv"]
    named = a[a.index("--init-ckpt") + 1].replace("\\", "/")
    tail = "/".join(named.split("/")[-2:])                   # M2b_s0_dagoacc/ckpt_last.pt
    for base in ([root] if root else [f"{ART}/v8f-dagoacc", f"{ART}/v8f-recover"]):
        p = os.path.join(base, tail)
        if os.path.isfile(p):
            return p.replace("\\", "/")
    return None


def arm_ckpts(root=None):
    """{arm: ckpt path or None}. P10, ZERO and BLIND use the C0_s0 model (the prior and the token path are read from the same forward)."""
    out = {}
    for s in (0, 1):
        d = f"{ART}/{C0_DIR[s]}"
        out[f"C0_s{s}"] = f"{d}/ckpt_last.pt" if os.path.isfile(f"{d}/ckpt_last.pt") else None
        out[f"DAGOACC_s{s}"] = init_ckpt_from_argv(f"{d}/argv.json", root)
    for a in ("P10", "ZERO", "BLIND"):
        out[a] = out["C0_s0"]
    return out


F_MIN = 18                          # a start needs 18 logged frames of history (card 6b)
START_DELTAS_S = (4.0, 2.0)         # launch scenes: t_c - 4 s and t_c - 2 s
GUARD_N, GUARD_SEED = 2, 1          # guard scenes: two starts from ONE numpy default_rng(1) stream
TAIL_TICKS = 60                     # the 6 s rollout must fit: f0 + 60 < n_ticks


def _collapse(c):
    """Two starts on the same frame are one start (order kept)."""
    out = []
    for f in c:
        if f not in out:
            out.append(f)
    return out


def starts_for_scene(entry, rng=None):
    """f0 ticks for one SCREEN_TABLE scene, card 6b. Launch: max(round((t_c - d) * 10), F_MIN) for d in (4, 2) s, same-frame starts collapsed, dropped only if f0 + 60 >= n_ticks.
    Guard: GUARD_N draws uniform over the renderable span [F_MIN, n - 61] from `rng` (the single default_rng(1) stream of `all_starts`; a guard without one is refused)."""
    n = int(entry["n_ticks"])
    lo, hi = F_MIN, n - TAIL_TICKS - 1
    if entry.get("slot", "").startswith("launch"):
        c = [max(int(round((float(entry["launch_events"][0]["t_c_s"]) - d) * 10)), lo) for d in START_DELTAS_S]
    else:
        if rng is None:
            raise ValueError("a guard scene's starts come from the shared default_rng(1) stream: use all_starts()")
        c = [int(rng.integers(lo, hi + 1)) for _ in range(GUARD_N)]
    return _collapse([f for f in c if lo <= f <= hi])


def all_starts(scenes=None):
    """{scene id: [f0, ...]} for the selected scenes of SCREEN_TABLE in order_index order; ONE default_rng(1) is consumed by the guard scenes in that order (card 6b)."""
    if scenes is None:
        scenes = json.load(open(SCREEN_TABLE))["scenes"]
    sel = sorted((e for e in scenes if e.get("selected")), key=lambda e: int(e["order_index"]))
    rng = np.random.default_rng(GUARD_SEED)
    return {e["scene"]: starts_for_scene(e, rng) for e in sel}


# ================================================================================================================================
# `ground_correct` lives in `t2v8f_exec` (one implementation, equal to the harness' `GroundFollower.correct`); this name is kept for callers of the kit module
# ================================================================================================================================
def ground_correct(gt_T, xy, yaw):
    import t2v8f_exec as X
    return X.ground_correct(gt_T, xy, yaw)
