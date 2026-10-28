#!/usr/bin/env python3
"""mlaunch.py -- M-LAUNCH, the registered routing variable of PREREG_V8F_LAUNCH_INTENT section 7 / 11.  ZERO TRAINING, dev-box CPU only, read-only on every input.

REGISTERED DEFINITION (verbatim, Project Steering/PREREG_V8F_LAUNCH_INTENT.md @ d7dc9f11, line 222):
  "M-LAUNCH, the mechanism row. X - Y of the mean (v_own - v_GT) at ticks 1-2 in the GT-launch windows, from the tick dumps, in each pair.
   The lever predicts it moves toward 0. It routes NEXT-LAUNCH (section 11) and decides no outcome."
  section 11: "moved" = separated toward 0 in BOTH pairs at BOTH seeds (the replication standard of section 10; section 0 item 8 (b)).
  pairs: main  X = A_s,     Y = CTRL_s = DAG-OACC_s      replicate  X = A_rep,s, Y = REP_s      s in {0, 1}      lane: the causal T1-kin lane (*_m10_causal_heads).

EVIDENCE CLASS of every number this writes: MEASURED (ours; the artifact is MLAUNCH.json next to this file).  TIER: T1-kin = the ego state is the model's OWN executed motion, the camera stream
is logged; it is OPEN LOOP at the scene level and is never called "closed loop".  ESTIMATOR: taniteval.ci paired episode-cluster bootstrap, B = 2000, seed 0, over the EPISODES of the launch windows.
It answers "would another draw of EPISODES say this?"; it is BLIND to training variance (H-ESTIM-SEED-1) and the two seeds + the replicate pair are the only replicate; no inference seed is varied
(the planner is deterministic).

WHAT IS READ AND WHERE (file:line hops are in NOTES.md):
  tick, v_own    the T1 tool's TickRecorder (v8f_t1_eval.py) stores one row per (window, tick): tick n is the executed state at step j = t0 + 10 n; v_own = `v0` = batch["p10f"][:, -1, 0], the measured speed
                 of the OWN executed state (one Hermite frame at 2j).  Loaded by the DAG-ORC rig's own loader (t1hd_read.load_ticks, AST-exec'd from the pinned source, not re-derived).
  v_GT           the GT's speed at the same tick = kin3 v0 of the GT knots under the SAME state rule (lcdiag statekin.state_from_knots, which calls the tool's own hermite_frames and
                 yaw_kin_from_track), the definition DIAGNOSIS.md's "O's own v0 - GT" used.  Sensitivity: the executed-chord reading (beside, decides nothing).
  launch windows hmc_read.py:384-388 (@ d7dc9f11), exactly: v0 < 6.0 m/s and |g[29] - g[28]| / 0.2 >= v0 + 1.0.

    CUDA_VISIBLE_DEVICES=-1 python mlaunch.py [--out MLAUNCH.json] [--n-boot 2000] [--manifest runs.json]
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
ORC = f"{ART}/dagorc_rebase/read"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
MANIFEST = "F:/TanitAD-scratch/launch_intent_t1_out/runs.json"
REG_COMMIT = "d7dc9f11"

# ---------------------------------------------------------------------------------------------------------------- constants (registered)
DT = 0.2                    # one cache step = 0.2 s (hmc_lib.py:23, v8f_mini.py LAUNCH_DT)
M_STEPS = 10                # replan every 10 cache steps (m = 10): ticks 0, 1, 2 at steps t0, t0+10, t0+20
TICKS = 3
V0_MAX, DV = 6.0, 1.0       # hmc_read.py:387, v8f_mini.py:396
TICK_SET = (1, 2)           # "ticks 1-2": 0-based tick index, tick 0 = the logged state
INJECT = 0.5                # m/s: the known effect of control b5
N_BOOT, SEED = 2000, 0
N_WINDOWS, N_EPISODES = 2399, 141
INHERITED_LAUNCH = (327, 49)   # PREREG_V8F_LAUNCH_INTENT section 7/0: "~327 windows / 49 episodes" (INHERITED, to be recounted)
SEEDS = (0, 1)
PAIRS = {"main": {"X": "A", "Y": "CTRL"}, "replicate": {"X": "Arep", "Y": "REP"}}
BETWEEN_ROWS = ("along_mae_m", "speed_mae_mps", "L2_curvature_mae_fast_1pm")

PINS = {      # sha256 of every source this reads for logic (AST-exec'd or imported); a mismatch REFUSES
    "t1hd_read.py": (f"{TREE}/{PK}/raw/2026-10-26-t1hdread/t1hd_read.py", "3c6d9f790f85511553c607a9d2102a7a425ac7c9a674be69ac23ea3a6cb1e9e2"),
    "dag_read.py": (f"{ORC}/ref/dag_read.py", "1cf56c7c30195ac47718bb07feef11531a6309cb01e0b28a34f926c6a696084c"),
    "ci.py": (f"{TREE}/taniteval/taniteval/ci.py", "38db571e368f02dc4c0d9df7b47333ebbb082e43695617ba0dfba881668cd73e"),
    "t1k_read.py": (f"{TREE}/{PK}/raw/2026-10-26-t1kin-hacc/t1k_read.py", "5610114d3e09501347ed0ac65e44790eb2954eb5d5b703af089a64e845dbf5e4"),
    "t1klc_read.py": (f"{HERE}/ref/t1klc_read.py", "9a5623e5f737677e6fe0cf6df41c3588b11282423e27008830a7914253f7a7ca"),
    "statekin.py": (f"{HERE}/ref/statekin.py", "8360556d0bf11624c62f9bca2e49035e5bee8021d16f2f4e98d122ea81ceb034"),
    "v8f_t1_eval.py": (f"{TREE}/stack/scripts/v8f_t1_eval.py", "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"),
    "v8f_mini_data.py": (f"{TREE}/stack/tanitad/data/v8f_mini_data.py", "0e4a8cea438532424addb14d3c28a8f0b34f6c1ca33562384c3e389141f49588"),
    "launch_mask_excerpt.py": (f"{HERE}/ref/launch_mask_excerpt.py", "2fc48fe8d17c27dad5e3f596ae42cdc142bf308eab9a7448a18285bc27447cd9"),
}


# ================================================================================================================ PURE LAYER (numpy only; unit-tested)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def launch_subset(v0, g, dt=DT, v0_max=V0_MAX, dv=DV):
    """The GT-launch windows, exactly hmc_read.py:386-387: v0 (ego speed at t0) < 6 m/s and the GT chord speed over the LAST 0.2 s of the 6 s horizon
    (|g[:, 29] - g[:, 28]| / 0.2) >= v0 + 1 m/s.  ``v0`` [N], ``g`` [N, 30, 2] (the GT in the t0 ego frame).  A NaN v0 gives False."""
    v0 = np.asarray(v0, np.float64)
    g = np.asarray(g, np.float64)
    vg6 = np.linalg.norm(g[:, 29] - g[:, 28], axis=-1) / dt
    with np.errstate(invalid="ignore"):
        return (v0 < v0_max) & (vg6 >= v0 + dv)


def launch_subset_loop(v0, g):
    """The same definition as a plain Python loop (an INDEPENDENT recount for control (c): no vectorisation, no numpy norm)."""
    out = []
    for i in range(len(v0)):
        dx, dy = float(g[i][29][0]) - float(g[i][28][0]), float(g[i][29][1]) - float(g[i][28][1])
        v6 = math.sqrt(dx * dx + dy * dy) / 0.2
        v = float(v0[i])
        out.append(bool(v == v and v < 6.0 and v6 >= v + 1.0))
    return np.array(out, bool)


def central_speed_cf(Kminus, Kplus, dt=DT):
    """Closed form of the state speed at a knot: |K[i+1] - K[i-1]| / (2 dt) -- the Catmull-Rom tangent norm / 0.2 that ``hermite_frames`` writes into the even frame (v8f_t1_eval.py:85-116)."""
    return np.linalg.norm(np.asarray(Kplus, np.float64) - np.asarray(Kminus, np.float64), axis=-1) / (2.0 * dt)


def gt_state_speed_cf(g, n, m=M_STEPS):
    """v_GT at tick n (n >= 1) in closed form: knots K = [origin; g] so K[i] = g[i - 1]; tick n is at K[m n]; speed = |K[m n + 1] - K[m n - 1]| / 0.4 = |g[m n] - g[m n - 2]| / 0.4."""
    g = np.asarray(g, np.float64)
    return central_speed_cf(g[:, m * n - 2], g[:, m * n])


def own_state_speed_cf(plan_prev, m=M_STEPS):
    """v_own at tick n in closed form from the PLAN issued at tick n-1 (ego frame, knots K[0] = origin, K[i] = plan[i - 1]): the executed state is at K[m], so the speed is
    |K[m + 1] - K[m - 1]| / 0.4 = |plan[m] - plan[m - 2]| / 0.4 (valid when the plan was inside the speed envelope)."""
    plan_prev = np.asarray(plan_prev, np.float64)
    return central_speed_cf(plan_prev[:, m - 2], plan_prev[:, m])


def chord_speed(x, n, m=M_STEPS):
    """Executed-chord speed at tick n: the speed over the last 0.2 s before the tick = |x[m n - 1] - x[m n - 2]| / 0.2 on any 30-knot track ``x`` [N, 30, 2] in the t0 frame."""
    x = np.asarray(x, np.float64)
    return np.linalg.norm(x[:, m * n - 1] - x[:, m * n - 2], axis=-1) / DT


def window_value(own, gt, mask, ticks=TICK_SET):
    """The per-window M-LAUNCH value: the mean over ``ticks`` of (v_own - v_GT), for the windows in ``mask``.  ``own``, ``gt`` [N, 3] indexed by tick."""
    own, gt = np.asarray(own, np.float64), np.asarray(gt, np.float64)
    return (own[mask][:, list(ticks)] - gt[mask][:, list(ticks)]).mean(axis=1)


def abs_mean(v):
    """The reducer of the 'toward 0' contrast: |mean(v)|.  With the paired bootstrap's  red(a) - red(b)  it gives  |mean X| - |mean Y|  on the SAME resampled episodes."""
    return abs(float(np.nanmean(v)))


def is_moved(cell):
    """A (pair, seed) cell is 'moved' = SEPARATED TOWARD 0:  X - Y separated (the paired CI excludes 0)  AND  |X| - |Y| separated below 0 (the paired CI of the absolute means is entirely negative)."""
    return bool(cell["x_minus_y"]["separated"] and cell["abs_x_minus_abs_y"]["separated"] and cell["abs_x_minus_abs_y"]["delta"] < 0)


def is_moved_sign_reading(cell):
    """The ALTERNATIVE reading (beside, decides nothing): X - Y separated and its sign is toward 0 given the sign of Y's mean (Y negative => X - Y > 0; Y positive => X - Y < 0)."""
    d, ym = cell["x_minus_y"]["delta"], cell["y"]["mean"]
    toward = (d > 0) if ym < 0 else (d < 0)
    return bool(cell["x_minus_y"]["separated"] and toward)


def route(moved_by_cell):
    """PREREG section 11, total over its 2 outcomes: M-LAUNCH 'moved' only when EVERY (pair, seed) cell is moved (both pairs, both seeds).  -> (word, committed next)."""
    word = "moved" if len(moved_by_cell) == 4 and all(moved_by_cell.values()) else "not moved"
    nxt = {"moved": "LOOP-STATE (zero training, dev-box CPU): decompose the crawl by plan index in X's tick dumps, as DIAGNOSIS.md did for HACC; its read names the loop term; the next PREREG intervenes on it",
           "not moved": "LAUNCH-INTENT-HI: the same lever at s = 1/2 (W_HI = 4.17852), under section 10's table, with its own PI go. If it also leaves M-LAUNCH in place, label weighting is not a lever on O's launch"}[word]
    return word, nxt


# ================================================================================================================ HEAVY LAYER
class Rig:
    """The pinned instruments: ci (numpy only), the DAG-ORC rig's tick / dump loaders, comps / lat_rows / paired, the tool's state functions.  Built lazily; ``heavy=False`` loads only ci."""

    def __init__(self, heavy=True):
        self.pin_report = verify_pins()
        import importlib.util
        spec = importlib.util.spec_from_file_location("mlaunch_ci", PINS["ci.py"][0])
        self.ci = importlib.util.module_from_spec(spec)
        sys.modules["mlaunch_ci"] = self.ci
        spec.loader.exec_module(self.ci)
        self.PB = self.ci.paired_episode_cluster_bootstrap
        self.EB = self.ci.episode_cluster_bootstrap
        ns = {"np": np, "os": os, "glob": glob, "TICKS": TICKS, "M_STEPS": M_STEPS}
        exec_from(PINS["dag_read.py"][0], ns, funcs=("need", "load_dump"), classes=("Refusal",))
        exec_from(PINS["t1hd_read.py"][0], ns, funcs=("load_ticks",))
        self.load_dump, self.load_ticks = ns["load_dump"], ns["load_ticks"]
        self.comps = self.paired_rows = self.lat_rows = self.tool = self.D = self.state_from_knots = None
        if heavy:
            self._heavy()

    def _heavy(self):
        os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", os.path.join(TREE, "stack").replace("/", os.sep))
        sys.path.insert(0, os.path.join(TREE, "taniteval"))
        import torch
        from taniteval import four_families as FF       # noqa: E402  (pins `tanitad` to the tool tree's stack)
        import importlib.util
        ns = {"np": np, "torch": torch, "FF": FF, "PB": self.PB, "math": math, "DT": DT, "VMIN": 5.2}
        exec_from(PINS["t1k_read.py"][0], ns, funcs=("comps", "paired"))
        exec_from(PINS["t1klc_read.py"][0], ns, funcs=("lat_rows",))
        self.comps, self.paired_rows, self.lat_rows, self.torch = ns["comps"], ns["paired"], ns["lat_rows"], torch
        from tanitad.data import v8f_mini_data as D
        spec = importlib.util.spec_from_file_location("t1tool", PINS["v8f_t1_eval.py"][0])
        T1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(T1)
        ns2 = {"np": np, "math": math, "T1": T1, "D": D}
        exec_from(PINS["statekin.py"][0], ns2, funcs=("state_from_knots",))
        self.tool, self.D, self.state_from_knots = T1, D, ns2["state_from_knots"]
        mns = {}
        exec(compile(open(PINS["launch_mask_excerpt.py"][0], encoding="utf-8").read(), PINS["launch_mask_excerpt.py"][0], "exec"), mns)
        self.launch_mask_ref = mns["launch_mask"]


def exec_from(src, ns, funcs=(), assigns=(), classes=()):
    """Execute only the named top-level definitions of ``src`` into ``ns`` (the lcdiag / rr_lib reuse pattern: the reader's own code, not a copy)."""
    tree = ast.parse(open(src, encoding="utf-8").read())
    for node in tree.body:
        hit = ((isinstance(node, ast.FunctionDef) and node.name in funcs) or (isinstance(node, ast.ClassDef) and node.name in classes)
               or (isinstance(node, ast.Assign) and any(getattr(t, "id", "") in assigns for t in node.targets)))
        if hit:
            exec(compile(ast.Module([node], []), src, "exec"), ns)


def verify_pins():
    out, bad = {}, []
    for k, (p, want) in PINS.items():
        got = sha256_file(p) if os.path.exists(p) else None
        out[k] = {"path": p, "sha256": got, "pinned": want, "ok": got == want}
        if got != want:
            bad.append(k)
    if bad:
        raise SystemExit(f"REFUSED: pinned sources changed or missing: {bad}")
    return out


def dir_sha(d):
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
        h.update(os.path.basename(f).encode() + b":" + sha256_file(f).encode() + b"\n")
    return h.hexdigest()


def load_manifest(path):
    m = json.load(open(path, encoding="utf-8"))
    runs = {}
    for arm, per in m["arms"].items():
        for s, e in per.items():
            runs[(arm, int(s))] = e["t1"]
    return m, runs


def load_run(rig, t1):
    """One run: the dump (g, cl, fl, eid, clip_index, ws) by the rig's load_dump, the ticks by the rig's load_ticks, the per-window v0, and the DUMP-WINDOWS alignment control."""
    d = rig.load_dump(t1["dump_dir"])
    zs = [np.load(f) for f in sorted(glob.glob(os.path.join(t1["dump_dir"], "ep*.npz")))]
    d["v0"] = np.concatenate([z["v0"] for z in zs])          # the dump's per-window ego speed at t0 (hmc_read.py's d["v0"]); load_dump does not return it
    tk = rig.load_ticks(t1["ticks"])
    ok = bool(tk["N"] == len(d["ws"]) and np.array_equal(tk["ci"], d["clip_index"]) and np.array_equal(tk["t0"], d["ws"]))
    row = tk["row"]
    return {"d": d, "tk": tk, "aligned": ok, "tag": t1["tag"],
            "v_own": tk["v0"].astype(np.float64),               # [N, 3] indexed by tick
            "plan": tk["plan"], "ticks_sha256": sha256_file(t1["ticks"]), "dump_sha256": dir_sha(t1["dump_dir"]),
            "v0_tick0_equals_dump_v0": float(np.abs(tk["v0"][:, 0] - d["v0"]).max()), "row": row}


def gt_speeds(rig, g):
    """v_GT at ticks 0..2 for every window.  R1 (primary): kin3 v0 of the GT knots under the tool's own state rule (statekin.state_from_knots on [origin; g], start = 10 (n - 1)); tick 0 is NaN
    (the logged state has no 'rebuilt' counterpart; every arm starts there).  R2 (sensitivity): the executed-chord speed |x[10n-1] - x[10n-2]| / 0.2.  Returns (R1 [N,3], R2 [N,3])."""
    N = len(g)
    r1 = np.full((N, TICKS), np.nan)
    for i in range(N):
        K = np.vstack([np.zeros((1, 2)), g[i]])
        for n in (1, 2):
            kin, _ = rig.state_from_knots(K, M_STEPS * (n - 1))
            r1[i, n] = kin[0]
    r2 = np.full((N, TICKS), np.nan)
    for n in (1, 2):
        r2[:, n] = chord_speed(g, n)
    return r1, r2


def own_chord(cl):
    out = np.full((len(cl), TICKS), np.nan)
    for n in (1, 2):
        out[:, n] = chord_speed(cl, n)
    return out


# ---------------------------------------------------------------------------------------------------------------- the cell statistic
def cell_stats(rig, own_x, own_y, gt, mask, eid, ticks=TICK_SET, n_boot=N_BOOT):
    """One (pair, seed) cell.  ``own_x``, ``own_y``, ``gt`` [N, 3] by tick; ``mask`` the launch windows; the unit is the WINDOW, clustered by EPISODE."""
    dx, dy = window_value(own_x, gt, mask, ticks), window_value(own_y, gt, mask, ticks)
    e = np.asarray(eid)[mask]
    diff = rig.PB(dx, dy, e, n_boot=n_boot, seed=SEED)
    absd = rig.PB(dx, dy, e, n_boot=n_boot, seed=SEED, reduce=abs_mean)
    mx, my = rig.EB(dx, e, n_boot=n_boot, seed=SEED), rig.EB(dy, e, n_boot=n_boot, seed=SEED)
    cell = {"n_windows": int(mask.sum()), "n_episodes": int(len(np.unique(e))), "ticks": list(ticks),
            "x": {"mean": float(np.mean(dx)), "ci95": [mx["lo"], mx["hi"]]}, "y": {"mean": float(np.mean(dy)), "ci95": [my["lo"], my["hi"]]},
            "x_minus_y": {k: diff[k] for k in ("delta", "lo", "hi", "separated")}, "x_minus_y_unrounded": float(np.mean(dx) - np.mean(dy)),
            "abs_x_minus_abs_y": {k: absd[k] for k in ("delta", "lo", "hi", "separated")},
            "toward_zero_point": bool(abs(float(np.mean(dx))) < abs(float(np.mean(dy))))}
    for k in ("degenerate", "degenerate_note"):
        if k in diff:
            cell["x_minus_y"][k] = diff[k]
    cell["moved"] = is_moved(cell)
    cell["moved_sign_reading"] = is_moved_sign_reading(cell)
    return cell


def per_tick_cells(rig, own_x, own_y, gt, mask, eid, n_boot=N_BOOT):
    """BESIDE (decides nothing): the same statistic at tick 1 and tick 2 separately."""
    out = {}
    for n in TICK_SET:
        c = cell_stats(rig, own_x, own_y, gt, mask, eid, ticks=(n,), n_boot=n_boot)
        out[f"tick{n}"] = {k: c[k] for k in ("x", "y", "x_minus_y", "abs_x_minus_abs_y", "toward_zero_point", "moved")}
    return out


def read_reading(rig, name, runs, gt, mask, eid, n_boot=N_BOOT):
    """All four (pair, seed) cells for one reading of v_own / v_GT.  ``runs[(arm, s)]`` = own speeds [N, 3]."""
    cells = {}
    for pair, xy in PAIRS.items():
        for s in SEEDS:
            ox, oy = runs[(xy["X"], s)], runs[(xy["Y"], s)]
            c = cell_stats(rig, ox, oy, gt, mask, eid, n_boot=n_boot)
            c["X"], c["Y"] = f"{xy['X']}_s{s}", f"{xy['Y']}_s{s}"
            c["per_tick"] = per_tick_cells(rig, ox, oy, gt, mask, eid, n_boot=n_boot)
            cells[f"{pair}_s{s}"] = c
    word, nxt = route({k: v["moved"] for k, v in cells.items()})
    word_sign, _ = route({k: v["moved_sign_reading"] for k, v in cells.items()})
    return {"reading": name, "cells": cells, "M_LAUNCH": word, "committed_next": nxt, "M_LAUNCH_under_sign_reading": word_sign,
            "moved_by_cell": {k: v["moved"] for k, v in cells.items()}}


def floors_beside(rig, runs, gt, mask, eid, n_boot=N_BOOT):
    """BESIDE, decides nothing -- the only look this programme has at TRAINING variance on M-LAUNCH: the same statistic between two runs that differ ONLY in the training seed (REP vs DAG-OACC: DAG-OACC's recipe at
    --seed 10 / 11; Arep vs A: the lever at the replicate seed).  The episode bootstrap cannot see this variance; if |REP - CTRL| is large next to |A - CTRL|, a separated A - CTRL would have needed to beat it."""
    out = {}
    for s in SEEDS:
        for xa, ya, why in (("REP", "CTRL", "same recipe (no lever), other training seed"), ("Arep", "A", "the lever, other training seed")):
            c = cell_stats(rig, runs[(xa, s)], runs[(ya, s)], gt, mask, eid, n_boot=n_boot)
            out[f"{xa}_minus_{ya}_s{s}"] = {"what": why, "x_minus_y": c["x_minus_y"], "x_mean": c["x"]["mean"], "y_mean": c["y"]["mean"], "n_windows": c["n_windows"], "n_episodes": c["n_episodes"]}
    return out


# ---------------------------------------------------------------------------------------------------------------- controls
def pb_zero(r):
    return bool(r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and r["separated"] is False)


def controls(rig, R, runs1, gt1, runs2, gt2, mask, eid, cells1, n_boot=N_BOOT):
    """(a) known zeros, (b) deliberate regressions that MUST move the number, (c) independent recounts of the window set.  -> dict; ['PASS'] is the conjunction."""
    C = {}
    # ---- (a) X = Y gives exactly 0.0 with CI [0, 0]: every run against itself, both readings
    a = {}
    for (arm, s), ox in sorted(runs1.items()):
        r = rig.PB(window_value(ox, gt1, mask), window_value(ox, gt1, mask), eid[mask], n_boot=n_boot, seed=SEED)
        a[f"R1:{arm}_s{s}"] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "PASS": pb_zero(r)}
    for pair, xy in PAIRS.items():
        for s in SEEDS:
            ox = runs2[(xy["Y"], s)]
            r = rig.PB(window_value(ox, gt2, mask), window_value(ox, gt2, mask), eid[mask], n_boot=n_boot, seed=SEED)
            a[f"R2:{xy['Y']}_s{s}"] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "PASS": pb_zero(r)}
    # (a2) tick 0 is the logged state: every arm starts from it, so X - Y at tick 0 is exactly 0 in every pair (this is what proves the two runs' windows are aligned)
    z = {}
    for pair, xy in PAIRS.items():
        for s in SEEDS:
            dx = R[(xy["X"], s)]["v_own"][mask][:, 0]
            dy = R[(xy["Y"], s)]["v_own"][mask][:, 0]
            r = rig.PB(dx, dy, eid[mask], n_boot=n_boot, seed=SEED)
            z[f"{pair}_s{s}"] = {"max_abs_v_own_tick0_X_minus_Y": float(np.abs(dx - dy).max()), "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "PASS": bool(np.abs(dx - dy).max() == 0.0 and pb_zero(r))}
    # (a3) v_own := v_GT (a perfect tracker) reads exactly 0 per window
    gg = {}
    for n in TICK_SET:
        gg[f"tick{n}"] = float(np.abs(gt1[mask][:, n] - gt1[mask][:, n]).max())
    perfect = window_value(gt1, gt1, mask)
    C["a_X_equals_Y"] = {"PASS": bool(all(v["PASS"] for v in a.values())), "runs": a}
    C["a2_tick0_known_zero_X_minus_Y"] = {"PASS": bool(all(v["PASS"] for v in z.values())), "cells": z}
    C["a3_perfect_tracker_reads_zero"] = {"PASS": bool(np.all(perfect == 0.0)), "max_abs_window_value": float(np.abs(perfect).max())}
    # ---- (b) deliberate regressions
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(gt1))
    b1, b2, b4, b5 = {}, {}, {}, {}
    for key, c in cells1.items():
        pair, s = key.rsplit("_s", 1)
        xy, s = PAIRS[pair], int(s)
        ox, oy = runs1[(xy["X"], s)], runs1[(xy["Y"], s)]
        lo, hi = c["x_minus_y"]["lo"], c["x_minus_y"]["hi"]
        # (b1) v_GT shuffled across ALL windows: each arm's own mean must move; X - Y must NOT (v_GT cancels in a paired difference)
        gs = gt1[perm]
        cs = cell_stats(rig, ox, oy, gs, mask, eid, n_boot=n_boot)
        b1[key] = {"x_mean": {"actual": c["x"]["mean"], "regressed": cs["x"]["mean"], "changed": bool(abs(cs["x"]["mean"] - c["x"]["mean"]) > 1e-6)},
                   "y_mean": {"actual": c["y"]["mean"], "regressed": cs["y"]["mean"], "changed": bool(abs(cs["y"]["mean"] - c["y"]["mean"]) > 1e-6)},
                   "x_minus_y_unchanged_max_abs_diff": abs(cs["x_minus_y_unrounded"] - c["x_minus_y_unrounded"]),
                   "abs_x_minus_abs_y": {"actual": c["abs_x_minus_abs_y"]["delta"], "regressed": cs["abs_x_minus_abs_y"]["delta"]}}
        b1[key]["PASS"] = bool(b1[key]["x_mean"]["changed"] and b1[key]["y_mean"]["changed"] and b1[key]["x_minus_y_unchanged_max_abs_diff"] <= 1e-12)
        # (b2) the launch mask swapped with its COMPLEMENT: each arm's mean (v_own - v_GT) must leave its launch CI (the ordinary windows do not have a slow launch); the count must be the complement.
        #      (X - Y itself is NOT judged here: the two arms are near-identical everywhere, so X - Y is ~0 on both sets and cannot show the swap; see b5 for the sensitivity control.)
        cc = cell_stats(rig, ox, oy, gt1, ~mask, eid, n_boot=n_boot)
        b2[key] = {"n_windows": cc["n_windows"], "count_is_complement": bool(cc["n_windows"] == len(mask) - int(mask.sum())),
                   "x_mean": {"launch": c["x"]["mean"], "launch_ci": c["x"]["ci95"], "complement": cc["x"]["mean"], "outside_launch_ci": bool(cc["x"]["mean"] < c["x"]["ci95"][0] or cc["x"]["mean"] > c["x"]["ci95"][1])},
                   "y_mean": {"launch": c["y"]["mean"], "launch_ci": c["y"]["ci95"], "complement": cc["y"]["mean"], "outside_launch_ci": bool(cc["y"]["mean"] < c["y"]["ci95"][0] or cc["y"]["mean"] > c["y"]["ci95"][1])},
                   "x_minus_y": {"launch": c["x_minus_y"]["delta"], "complement": cc["x_minus_y"]["delta"], "differs": bool(cc["x_minus_y_unrounded"] != c["x_minus_y_unrounded"])}}
        b2[key]["PASS"] = bool(b2[key]["count_is_complement"] and b2[key]["x_mean"]["outside_launch_ci"] and b2[key]["y_mean"]["outside_launch_ci"] and b2[key]["x_minus_y"]["differs"])
        # (b4) Y's windows PERMUTED across all windows (the pairing destroyed): X - Y becomes X_launch - Y_population and must leave the registered CI
        cm = cell_stats(rig, ox, oy[perm], gt1, mask, eid, n_boot=n_boot)
        b4[key] = {"actual": c["x_minus_y"]["delta"], "actual_ci": [lo, hi], "regressed": cm["x_minus_y"]["delta"], "outside_actual_ci": bool(cm["x_minus_y"]["delta"] < lo or cm["x_minus_y"]["delta"] > hi)}
        b4[key]["PASS"] = b4[key]["outside_actual_ci"]
        # (b5) a KNOWN EFFECT injected into X: +/- 0.5 m/s on X's v_own at ticks 1-2 (a synthetic lever).  The readout must move by exactly that, the toward-0 injection must read MOVED,
        #      the away-from-0 injection must read separated but NOT moved.  This is the control that proves the section-11 verdict CAN read 'moved'.
        sgn = -1.0 if c["y"]["mean"] > 0 else 1.0                       # toward 0 for Y's sign
        inj = {}
        for nm, shift in (("toward_0", sgn * INJECT), ("away_from_0", -sgn * INJECT)):
            ox2 = np.array(ox, copy=True)
            ox2[:, list(TICK_SET)] += shift
            ci_ = cell_stats(rig, ox2, oy, gt1, mask, eid, n_boot=n_boot)
            inj[nm] = {"shift": shift, "x_minus_y_unrounded": ci_["x_minus_y_unrounded"], "reads_actual_plus_shift_max_abs_err": abs(ci_["x_minus_y_unrounded"] - (c["x_minus_y_unrounded"] + shift)),
                       "x_minus_y": ci_["x_minus_y"], "abs_x_minus_abs_y": ci_["abs_x_minus_abs_y"], "moved": ci_["moved"]}
        inj["toward_0"]["PASS"] = bool(inj["toward_0"]["reads_actual_plus_shift_max_abs_err"] <= 1e-12 and inj["toward_0"]["moved"])
        inj["away_from_0"]["PASS"] = bool(inj["away_from_0"]["reads_actual_plus_shift_max_abs_err"] <= 1e-12 and inj["away_from_0"]["x_minus_y"]["separated"] and not inj["away_from_0"]["moved"])
        b5[key] = {**inj, "PASS": bool(inj["toward_0"]["PASS"] and inj["away_from_0"]["PASS"])}
    C["b1_v_gt_shuffled_across_windows"] = {"PASS": bool(all(v["PASS"] for v in b1.values())), "seed": SEED, "cells": b1}
    C["b2_launch_mask_swapped_with_complement"] = {"PASS": bool(all(v["PASS"] for v in b2.values())), "cells": b2}
    C["b4_windows_of_Y_permuted_across_windows"] = {"PASS": bool(all(v["PASS"] for v in b4.values())), "seed": SEED, "cells": b4}
    C["b5_known_effect_injected_into_X_reads_exactly_and_flips_the_verdict"] = {"PASS": bool(all(v["PASS"] for v in b5.values())), "injected_m_per_s": INJECT, "cells": b5}
    # ---- (c) the window count against independent recounts
    ref = R[("CTRL", 0)]
    g0, v00 = ref["d"]["g"], ref["d"]["v0"]
    n_v, ep_v = int(mask.sum()), int(len(np.unique(eid[mask])))
    cnt = {"c1_dump_vectorised(hmc_read.py:386-387)": (n_v, ep_v)}
    tk0 = ref["tk"]
    gt_t0, v0_t0 = tk0["gt"][:, 0], tk0["v0"][:, 0]                      # tick 0 of the TICKS file: GT in the t0 frame and the logged v0, a different file from the dump
    m2 = launch_subset(v0_t0, gt_t0)
    cnt["c2_tick0_rows_of_the_ticks_file"] = (int(m2.sum()), int(len(np.unique(eid[m2]))))
    m3 = rig.launch_mask_ref(rig.torch.as_tensor(v00), rig.torch.as_tensor(g0), rig.torch.ones(len(v00)))
    m3 = m3.numpy().astype(bool)
    cnt["c3_v8f_mini.launch_mask@d7dc9f11:401(torch, the TRAIN-side definition)"] = (int(m3.sum()), int(len(np.unique(eid[m3]))))
    m4 = launch_subset_loop(v00, g0)
    cnt["c4_plain_python_loop"] = (int(m4.sum()), int(len(np.unique(eid[m4]))))
    sets_equal = bool(np.array_equal(m2, mask) and np.array_equal(m3, mask) and np.array_equal(m4, mask))
    per_run = {f"{a_}_s{s}": (int(launch_subset(R[(a_, s)]["d"]["v0"], R[(a_, s)]["d"]["g"]).sum())) for (a_, s) in sorted(R)}
    same_windows = {f"{a_}_s{s}": bool(np.array_equal(R[(a_, s)]["d"]["eid"], eid) and np.array_equal(R[(a_, s)]["d"]["clip_index"], ref["d"]["clip_index"])
                                          and np.array_equal(R[(a_, s)]["d"]["ws"], ref["d"]["ws"]) and np.array_equal(R[(a_, s)]["d"]["g"], g0) and np.array_equal(R[(a_, s)]["d"]["v0"], v00)) for (a_, s) in sorted(R)}
    vg6 = np.linalg.norm(g0[:, 29] - g0[:, 28], axis=-1) / DT
    margin = {"min_abs_v0_minus_6": float(np.min(np.abs(v00 - V0_MAX))), "min_abs_vG6_minus_(v0+1)": float(np.min(np.abs(vg6 - (v00 + DV))))}
    C["c_window_count_independent_recount"] = {"PASS": bool(len(set(cnt.values())) == 1 and sets_equal and all(per_run[k] == n_v for k in per_run) and all(same_windows.values())),
                                               "counts_n_windows_n_episodes": {k: list(v) for k, v in cnt.items()}, "mask_sets_identical_across_recounts": sets_equal,
                                               "per_run_launch_count": per_run, "all_8_runs_share_windows_g_v0_eid": same_windows, "threshold_margin": margin,
                                               "inherited_PREREG_327_49": list(INHERITED_LAUNCH), "matches_inherited": bool((n_v, ep_v) == INHERITED_LAUNCH)}
    # ---- (d) the closed forms reproduce the tool's own rebuild (v_GT) and the stored v_own
    d1 = {}
    for n in TICK_SET:
        d1[f"v_GT_tick{n}_tool_minus_closed_form_max_abs"] = float(np.abs(gt1[:, n] - gt_state_speed_cf(g0, n)).max())
    for (arm, s), r in sorted(R.items()):
        for n in TICK_SET:
            cf = own_state_speed_cf(r["plan"][:, n - 1])
            diff = np.abs(r["v_own"][:, n] - cf)
            d1[f"v_own_{arm}_s{s}_tick{n}"] = {"max_abs_stored_minus_closed_form": float(diff.max()), "n_windows_over_1e-4": int((diff > 1e-4).sum())}
    ok_d = bool(all(d1[f"v_GT_tick{n}_tool_minus_closed_form_max_abs"] < 1e-5 for n in TICK_SET))
    C["d_closed_forms_reproduce_the_tool"] = {"PASS": ok_d, "detail": d1,
                                               "note": "v_GT: the tool's hermite/yaw_kin rebuild equals |g[10n] - g[10n-2]|/0.4.  v_own: stored v0 (float32) vs |plan_(n-1)[10] - plan_(n-1)[8]|/0.4 -- the envelope clip is the only legitimate exception; reported, not gating"}
    C["PASS"] = bool(all(C[k]["PASS"] for k in C))
    return C


# ---------------------------------------------------------------------------------------------------------------- the beside rows (M2): along-track MAE, speed MAE, L2 on the launch subset
def beside_m2(rig, R, mask, eid, n_boot=N_BOOT):
    """X - Y per pair per seed on the SAME launch subset: along-track MAE (m), speed MAE (m/s) and L2 curvature MAE (1/m, GT >= 5.2 only) of the executed track `cl` against the GT `g`,
    by the reader's own `comps` / `lat_rows` / `paired` (t1k_read.py, t1klc_read.py; AST-exec'd from the pinned sources).  DECIDES NOTHING."""
    rows = {}
    for pair, xy in PAIRS.items():
        for s in SEEDS:
            X, Y = R[(xy["X"], s)]["d"], R[(xy["Y"], s)]["d"]
            Cx, Cy = rig.comps(X["cl"], X["g"]), rig.comps(Y["cl"], Y["g"])
            Cx.update({k: v for k, v in rig.lat_rows(X["cl"], X["g"]).items() if not k.startswith("_")})
            Cy.update({k: v for k, v in rig.lat_rows(Y["cl"], Y["g"]).items() if not k.startswith("_")})
            out = {}
            for r in BETWEEN_ROWS:
                a, b = Cx[r][mask], Cy[r][mask]
                m = np.isfinite(a) & np.isfinite(b)
                p = rig.paired_rows(Cx[r][mask], Cy[r][mask], eid[mask])
                out[r] = {"x_mean": float(np.mean(a[m])), "y_mean": float(np.mean(b[m])), "x_minus_y": p, "lower_is_better": True}
            rows[f"{pair}_s{s}"] = {"X": f"{xy['X']}_s{s}", "Y": f"{xy['Y']}_s{s}", "rows": out}
    return {"what": "along-track MAE, speed MAE and L2 on the GT-launch subset, X - Y; PREREG section 7 'M2' (decides nothing)", "tier": "T1-kin", "estimator": "paired episode-cluster bootstrap B=2000 seed 0 (windows with a finite value in both arms)",
            "pairs": rows}


# ---------------------------------------------------------------------------------------------------------------- main
def fmt_cell(k, c):
    return (f"  {k:14s} X {c['x']['mean']:+.4f} [{c['x']['ci95'][0]:+.4f},{c['x']['ci95'][1]:+.4f}]  Y {c['y']['mean']:+.4f} [{c['y']['ci95'][0]:+.4f},{c['y']['ci95'][1]:+.4f}]  "
            f"X-Y {c['x_minus_y']['delta']:+.4f} [{c['x_minus_y']['lo']:+.4f},{c['x_minus_y']['hi']:+.4f}] sep {c['x_minus_y']['separated']}  "
            f"|X|-|Y| {c['abs_x_minus_abs_y']['delta']:+.4f} [{c['abs_x_minus_abs_y']['lo']:+.4f},{c['abs_x_minus_abs_y']['hi']:+.4f}] sep {c['abs_x_minus_abs_y']['separated']}  MOVED {c['moved']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out", default=f"{HERE}/MLAUNCH.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--no-beside", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    rig = Rig(heavy=True)
    man, t1s = load_manifest(a.manifest)
    need = [(arm, s) for pair in PAIRS.values() for arm in pair.values() for s in SEEDS]
    R = {}
    for key in sorted(set(need)):
        R[key] = load_run(rig, t1s[key])
        print(f"loaded {key[0]}_s{key[1]} {R[key]['tag']}: aligned {R[key]['aligned']} N {R[key]['tk']['N']}", flush=True)
    if not all(r["aligned"] for r in R.values()):
        raise SystemExit("REFUSED: a ticks file is not window-aligned with its dump (DUMP-WINDOWS)")
    ref = R[("CTRL", 0)]["d"]
    eid, g, v0 = np.asarray(ref["eid"]), ref["g"], ref["v0"]
    mask = launch_subset(v0, g)
    print(f"launch windows: {int(mask.sum())} of {len(mask)}, {len(np.unique(eid[mask]))} episodes", flush=True)
    gt1, gt2 = gt_speeds(rig, g)                                        # R1 (primary), R2 (executed-chord sensitivity)
    runs1 = {k: r["v_own"] for k, r in R.items()}
    runs2 = {k: own_chord(r["d"]["cl"]) for k, r in R.items()}
    primary = read_reading(rig, "R1: v_own = ticks v0 (own state speed); v_GT = kin3 v0 of the GT knots under the tool's state rule", runs1, gt1, mask, eid, a.n_boot)
    for k, c in primary["cells"].items():
        print(fmt_cell(k, c), flush=True)
    print(f"M-LAUNCH (R1) = {primary['M_LAUNCH']}  (sign reading: {primary['M_LAUNCH_under_sign_reading']})", flush=True)
    sens = read_reading(rig, "R2: executed-chord speeds |x[10n-1]-x[10n-2]|/0.2 on cl (own) and g (GT)", runs2, gt2, mask, eid, a.n_boot)
    for k, c in sens["cells"].items():
        print("R2" + fmt_cell(k, c), flush=True)
    print(f"M-LAUNCH (R2, sensitivity) = {sens['M_LAUNCH']}", flush=True)
    floors = floors_beside(rig, runs1, gt1, mask, eid, a.n_boot)
    for k, v in floors.items():
        p = v["x_minus_y"]
        print(f"  FLOOR {k:22s} {v['what']:48s} {p['delta']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}] sep {p['separated']}", flush=True)
    ctl = controls(rig, R, runs1, gt1, runs2, gt2, mask, eid, primary["cells"], n_boot=a.n_boot)
    for k, v in ctl.items():
        if k != "PASS":
            print(f"CONTROL {k}: {'PASS' if v['PASS'] else 'FAIL'}", flush=True)
    print(f"CONTROLS {'ALL PASS' if ctl['PASS'] else 'FAILED'}", flush=True)
    out = {"what": "M-LAUNCH (PREREG_V8F_LAUNCH_INTENT section 7 / 11); zero training; decides no outcome",
           "registered_definition": "X - Y of the mean (v_own - v_GT) at ticks 1-2 in the GT-launch windows, from the tick dumps, in each pair. The lever predicts it moves toward 0. It routes NEXT-LAUNCH (section 11) and decides no outcome.",
           "prereg": f"Project Steering/PREREG_V8F_LAUNCH_INTENT.md @ {REG_COMMIT}", "evidence_class": "MEASURED (ours; artifact = this file)", "tier": "T1-kin (own executed state, logged camera; never 'closed loop')",
           "estimator": f"taniteval.ci paired episode-cluster bootstrap, B={a.n_boot}, seed {SEED}, over the {len(np.unique(eid[mask]))} episodes of the launch windows; blind to training variance (H-ESTIM-SEED-1)",
           "unit": "per-window value = mean over ticks (1, 2) of (v_own - v_GT); sign: negative = the car is slower than the human", "tick_set_0based": list(TICK_SET),
           "launch_subset": {"definition": "v0 < 6.0 m/s and |g[29]-g[28]|/0.2 >= v0 + 1.0 (hmc_read.py:386-387 @ d7dc9f11)", "n_windows": int(mask.sum()), "n_episodes": int(len(np.unique(eid[mask]))),
                             "of_windows": int(len(mask)), "inherited_PREREG": list(INHERITED_LAUNCH)},
           "verdict_rule": "a cell is MOVED = X-Y separated AND |X|-|Y| separated below 0; M-LAUNCH is 'moved' only if all four cells (main/replicate x s0/s1) are moved (PREREG section 11, section 0 item 8 (b))",
           "primary": primary, "sensitivity": sens, "beside_training_seed_floor": floors, "controls": ctl,
           "M_LAUNCH": primary["M_LAUNCH"] if ctl["PASS"] else "REFUSED (a control failed; the rows are not an outcome)", "committed_next": primary["committed_next"] if ctl["PASS"] else None,
           "runs": {f"{k[0]}_s{k[1]}": {"tag": r["tag"], "ticks_sha256": r["ticks_sha256"], "dump_sha256": r["dump_sha256"], "ticks": t1s[k]["ticks"], "dump_dir": t1s[k]["dump_dir"]} for k, r in R.items()},
           "manifest": {"path": a.manifest, "sha256": sha256_file(a.manifest)}, "pins": rig.pin_report}
    if not a.no_beside:
        out["beside_M2"] = beside_m2(rig, R, mask, eid, a.n_boot)
        for k, v in out["beside_M2"]["pairs"].items():
            for r, x in v["rows"].items():
                p = x["x_minus_y"]
                print(f"  M2 {k:14s} {r:28s} X {x['x_mean']:.4f} Y {x['y_mean']:.4f} X-Y {p['delta']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}] sep {p['separated']} n {p['n_windows']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (o.item() if hasattr(o, "item") else str(o)))
    wp = os.path.join(os.path.dirname(os.path.abspath(a.out)), "raw", "mlaunch_windows.npz")
    os.makedirs(os.path.dirname(wp), exist_ok=True)
    np.savez_compressed(wp, eid=eid, clip_index=ref["clip_index"], ws=ref["ws"], v0=v0, launch=mask, v_gt_R1=gt1, v_gt_R2=gt2,
                        **{f"v_own_R1_{k[0]}_s{k[1]}": v for k, v in runs1.items()}, **{f"v_own_R2_{k[0]}_s{k[1]}": v for k, v in runs2.items()})
    print(f"M-LAUNCH = {out['M_LAUNCH']}  -> {a.out}  ({time.time() - t0:.0f}s)", flush=True)
    return 0 if ctl["PASS"] else 2


if __name__ == "__main__":
    sys.exit(main())
