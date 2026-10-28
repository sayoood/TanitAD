#!/usr/bin/env python3
"""crawl_diag.py -- H-CRAWL-01 / -02 / -03, ZERO TRAINING, dev-box CPU only, read-only on every input.

The committed readings are in PREREG_FROZEN.md (the cards, verbatim) and READINGS.md (the choices fixed before any result). This file only implements them.

EVIDENCE CLASS of every number written: MEASURED (ours; artifacts CRAWL_DIAG.json + raw/).  TIER: T1-kin = the ego state is the model's OWN executed motion, the camera is logged
(open loop at the scene level, never "closed loop"); the first-tick counterfactuals of H-CRAWL-02 are T0-class (teacher-forced, one forward call, no rollout) and are never
"driving performance".  ESTIMATOR: taniteval.ci paired / plain episode-cluster bootstrap, B = 2000, seed 0.  It answers "would another draw of EPISODES say this?" and is BLIND to
training variance (H-ESTIM-SEED-1); the two runs are two training seeds of one recipe and no inference seed exists (deterministic planner).

    CUDA_VISIBLE_DEVICES=-1 python -I crawl_diag.py --task 01|02|03|merge|all
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
MLAUNCH_DIR = f"{ART}/dagland/launch_intent_mlaunch"
RUNS_JSON = "F:/TanitAD-scratch/launch_intent_t1_out/runs.json"
MANIFEST_TRAIN = "C:/Users/Admin/box2_ship/logs/v8f_dscale/manifest_train.jsonl"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
EVAL_CACHE = f"{ART}/sfresh_inputs/cache"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
EVAL_JOIN = f"{TREE}/{PK}/raw/join_eval141.npz"
WINDOW_LIST_SHA = "e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168"
CKPT_PREFIX = {"CTRL": "67265ec1", "REP": "b03fd6ef"}
RUN_NAME = {"CTRL": "DAG-OACC s0", "REP": "REP s0"}
T1_TOOL_SHA = "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"
DAGOACC_ARGV = ["--arm", "M2b", "--seed", "0", "--grid", "pool4x10", "--out", "x", "--kin-mode", "yaw", "--exclude", "540:600", "--batch", "64", "--steps-mult", "4", "--steps", "3000",
                "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0", "--dagger-p", "0.1", "--w-acc", "49.6969"]

DT = 0.2
N_BOOT, SEED = 2000, 0
TICKS = (1, 2)                      # 0-based ticks of the registered M-LAUNCH reading
V_EDGES = (0.5, 1.5, 3.0, 6.0)      # READINGS R1.2: speed buckets [0,.5) [.5,1.5) [1.5,3) [3,6); >= 6 is out
D_EDGES = (-0.25, 0.25)             # READINGS R1.2: speed change over the last 0.6 s
N_V, N_D, MIN_CELL = 4, 3, 50
D_LAG = 3                           # cache steps (0.6 s)
STOP_V = 0.5
HS = (10, 20)                       # output-speed knots (2 s, 4 s)
DELTAS = (-0.5, 0.5, 1.0)


# ================================================================================================================ PURE LAYER (numpy only; unit-tested)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def v_bucket(v):
    """Speed bucket 0..3 for [0,.5) [.5,1.5) [1.5,3) [3,6); -1 for >= 6, negative or non-finite."""
    v = np.asarray(v, np.float64)
    b = np.digitize(v, V_EDGES)
    ok = np.isfinite(v) & (v >= 0.0) & (b < N_V)
    return np.where(ok, b, -1).astype(np.int64)


def d_class(d):
    """History class 0 (d <= -0.25), 1 (between), 2 (d >= +0.25) of the speed change over the last 0.6 s."""
    d = np.asarray(d, np.float64)
    return np.where(d <= D_EDGES[0], 0, np.where(d >= D_EDGES[1], 2, 1)).astype(np.int64)


def cell_of(vb, dc):
    vb, dc = np.asarray(vb), np.asarray(dc)
    return np.where(vb >= 0, vb * N_D + dc, -1).astype(np.int64)


def build_table(vb, dc, vgt, min_cell=MIN_CELL):
    """The outcome-blind conditional table from TRAIN windows: per (speed bucket x history class) cell and per tick the median and the mean of v_GT.
    ``vgt`` [N, 2] = v_GT at ticks 1, 2.  A cell with < ``min_cell`` windows falls back to its speed-bucket marginal (listed in ``fallback_cells``)."""
    vb, dc, vgt = np.asarray(vb), np.asarray(dc), np.asarray(vgt, np.float64)
    nc = N_V * N_D
    cell = cell_of(vb, dc)
    cnt = np.zeros(nc, np.int64)
    med, mean = np.full((nc, vgt.shape[1]), np.nan), np.full((nc, vgt.shape[1]), np.nan)
    mmed, mmean = np.full((N_V, vgt.shape[1]), np.nan), np.full((N_V, vgt.shape[1]), np.nan)
    mcnt = np.zeros(N_V, np.int64)
    for c in range(nc):
        m = cell == c
        cnt[c] = int(m.sum())
        if cnt[c]:
            med[c], mean[c] = np.median(vgt[m], axis=0), vgt[m].mean(axis=0)
    for b in range(N_V):
        m = vb == b
        mcnt[b] = int(m.sum())
        if mcnt[b]:
            mmed[b], mmean[b] = np.median(vgt[m], axis=0), vgt[m].mean(axis=0)
    fell = []
    for c in range(nc):
        if cnt[c] < min_cell:
            med[c], mean[c] = mmed[c // N_D], mmean[c // N_D]
            fell.append(int(c))
    return {"count": cnt, "median": med, "mean": mean, "marg_count": mcnt, "marg_median": mmed, "marg_mean": mmean, "fallback_cells": fell}


def predict(table, vb, dc, stat="median", vonly=False):
    """Table entry [N, 2] for each window; NaN where the window has no bucket."""
    vb, dc = np.asarray(vb), np.asarray(dc)
    if vonly:
        t = table["marg_" + stat]
        out = t[np.clip(vb, 0, N_V - 1)]
    else:
        t = table[stat]
        out = t[np.clip(cell_of(vb, dc), 0, t.shape[0] - 1)]
    return np.where((vb >= 0)[:, None], out, np.nan)


def window_gap(own, gt, ticks=TICKS):
    """Per-window mean over ``ticks`` of (own - gt); ``own``, ``gt`` [N, 3] indexed by tick."""
    own, gt = np.asarray(own, np.float64), np.asarray(gt, np.float64)
    return (own[:, list(ticks)] - gt[:, list(ticks)]).mean(axis=1)


def blind_gap(pred, vgt):
    """Per-window outcome-blind gap: mean over the two ticks of (table prediction - v_GT).  ``pred``, ``vgt`` [N, 2]."""
    return (np.asarray(pred, np.float64) - np.asarray(vgt, np.float64)).mean(axis=1)


def chord_speeds(g, dt=DT):
    """[N, 30] chord speeds |g[i]-g[i-1]|/dt of a GT track in the t0 frame, the first from the origin."""
    g = np.asarray(g, np.float64)
    K = np.concatenate([np.zeros((len(g), 1, 2)), g], axis=1)
    return np.linalg.norm(np.diff(K, axis=1), axis=-1) / dt


def stay_mask(v0, g, v_stop=STOP_V, dv=1.0):
    """GT-stay window (READINGS R1.6): stopped (v0 < 0.5) and the GT never exceeds v0 + 1 m/s at any 0.2 s chord of the 6 s horizon."""
    v0 = np.asarray(v0, np.float64)
    top = chord_speeds(g).max(axis=1)
    with np.errstate(invalid="ignore"):
        return (v0 < v_stop) & (top < v0 + dv)


def out_speed(plan, h):
    """State speed at knot K[h] of a plan [N, 30, 2] (knot i = plan[i-1]): |plan[h]-plan[h-2]|/0.4 -- ``mlaunch.own_state_speed_cf`` at m = h."""
    plan = np.asarray(plan, np.float64)
    return np.linalg.norm(plan[:, h] - plan[:, h - 2], axis=-1) / (2.0 * DT)


def out_chord_end(plan):
    plan = np.asarray(plan, np.float64)
    return np.linalg.norm(plan[:, 29] - plan[:, 28], axis=-1) / DT


def shift_speed(v, delta):
    """v' = max(v + delta, 0); NaN stays NaN."""
    return np.maximum(np.asarray(v, np.float64) + delta, 0.0)


def flat_speed(v, v_now):
    """The history replaced by the current speed (finite rows only)."""
    v = np.asarray(v, np.float64)
    return np.where(np.isfinite(v), v_now, np.nan)


def zero_speed(v):
    v = np.asarray(v, np.float64)
    return np.where(np.isfinite(v), 0.0, np.nan)


def donors_other_episode(eid, pool, seed=0):
    """For every window i a donor index drawn from ``pool`` (indices into ``eid``) whose episode differs from window i's.  Deterministic in ``seed``."""
    eid = np.asarray(eid)
    pool = np.asarray(pool, np.int64)
    rng = np.random.default_rng(seed)
    out = np.empty(len(eid), np.int64)
    for i in range(len(eid)):
        cand = pool[eid[pool] != eid[i]]
        if len(cand) == 0:
            raise ValueError(f"no donor from another episode for window {i}")
        out[i] = cand[rng.integers(len(cand))]
    return out


def ratio_of_sums(dvo, dvi, idx):
    den = float(np.sum(dvi[idx]))
    return float(np.sum(dvo[idx]) / den) if den != 0.0 else float("nan")


def ratio_ci(ci, dvo, dvi, eid, n_boot=N_BOOT, seed=SEED):
    """S_v = sum(dv_out) / sum(dv_in) over a window subset, with an episode-cluster bootstrap CI of the ratio (``taniteval.ci`` with an index-valued reducer)."""
    dvo, dvi = np.asarray(dvo, np.float64), np.asarray(dvi, np.float64)
    if len(dvo) == 0 or float(np.sum(dvi)) == 0.0:
        return {"S_v": None, "lo": None, "hi": None, "n_windows": int(len(dvo)), "note": "undefined: no realised change of the input speed on this subset"}
    idx = np.arange(len(dvo), dtype=np.float64)
    r = ci.episode_cluster_bootstrap(idx, eid, reduce=lambda v: ratio_of_sums(dvo, dvi, v.astype(np.int64)), n_boot=n_boot, seed=seed, dp=6)
    return {"S_v": ratio_of_sums(dvo, dvi, np.arange(len(dvo))), "lo": r["lo"], "hi": r["hi"], "se": r["se"], "n_windows": r["n_windows"], "n_episodes": r["n_episodes"]}


def paired_ratio_ci(ci, dvo_x, dvi_x, dvo_y, dvi_y, eid, n_boot=N_BOOT, seed=SEED):
    """S_v(x) - S_v(y) on the SAME resampled episodes (``paired_episode_cluster_bootstrap`` on index-encoded values: x -> 0..n-1, y -> n..2n-1)."""
    n = len(dvo_x)
    a = np.arange(n, dtype=np.float64)
    b = a + n
    X = (np.asarray(dvo_x, np.float64), np.asarray(dvi_x, np.float64))
    Y = (np.asarray(dvo_y, np.float64), np.asarray(dvi_y, np.float64))

    def red(v):
        i = v.astype(np.int64)
        if i.min() >= n:
            return ratio_of_sums(Y[0], Y[1], i - n)
        return ratio_of_sums(X[0], X[1], i)
    r = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=n_boot, seed=seed, reduce=red)
    return {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "contains_zero": bool(r["lo"] <= 0.0 <= r["hi"]),
            "n_windows": r["n_windows"], "n_episodes": r["n_episodes"]}


def verdict_h01(d_point, d_lo, d_hi, stay_gap):
    """READINGS R1.8, total over its outcomes.  D = model gap - outcome-blind gap (median table)."""
    a = bool(d_point <= -0.4 and d_hi < 0.0 and stay_gap <= 0.0)
    b_ci = bool(d_lo > -0.2 and d_hi < 0.2)
    b_pt = bool(-0.2 < d_point < 0.2)
    if a:
        return {"outcome": "a", "text": "model crawls beyond what its inputs allow: shortcut/copycat explanations (02, 04-06) stay alive"}
    if b_ci:
        return {"outcome": "b", "text": "crawl is information-limited: copycat fixes will not close it; lever is trigger perception (06 probe) or output representation (09)"}
    if b_pt:
        return {"outcome": "b_point_only", "text": "point estimate inside +/-0.2 but the CI is not: (b) on the point only"}
    why = []
    if d_point > -0.4:
        why.append(f"D {d_point:+.4f} is above -0.4")
    if not d_hi < 0.0:
        why.append("the CI does not exclude 0")
    if not stay_gap <= 0.0:
        why.append(f"the GT-stay gap {stay_gap:+.4f} is positive")
    return {"outcome": "neither", "text": "neither committed outcome: " + "; ".join(why) if why else "neither committed outcome"}


def verdict_h02(sv_ctrl, sv_rep):
    """READINGS R2.6, falsifier 1 and the supported reading.  Each sv = {"S_v", "lo", "hi"} (pooled, primary subset, h = 10)."""
    s = [x for x in (sv_ctrl, sv_rep)]
    if any(x["S_v"] is None for x in s):
        return {"outcome": "undefined", "text": "S_v undefined on the primary subset"}
    fires = all(x["S_v"] <= 0.3 and x["hi"] < 0.5 for x in s)
    supported = all(x["S_v"] >= 0.6 for x in s)
    if fires:
        return {"outcome": "falsifier_1_fires", "text": "output speed largely independent of input speed on both runs: 04 and 05 lose their mechanism"}
    if supported:
        return {"outcome": "supported", "text": "S_v >= 0.6 on both runs (point); lower bounds: " + ", ".join(f"{x['lo']:+.3f}" for x in s)}
    return {"outcome": "between", "text": "between the committed outcomes: " + ", ".join(f"S_v {x['S_v']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for x in s)}


# ================================================================================================================ HEAVY LAYER
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mean_ci(rig, v, eid):
    r = rig.ci.episode_cluster_bootstrap(np.asarray(v, np.float64), eid, reduce="mean", n_boot=N_BOOT, seed=SEED, dp=6)
    return {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "n_windows": r["n_windows"], "n_episodes": r["n_episodes"]}


def diff_ci(rig, a, b, eid):
    r = rig.PB(np.asarray(a, np.float64), np.asarray(b, np.float64), eid, n_boot=N_BOOT, seed=SEED)
    return {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n_windows": r["n_windows"], "n_episodes": r["n_episodes"]}


class Env:
    """The pinned instruments: mlaunch (its Rig, loaders, pure layer), the T1 tool, the two runs."""

    def __init__(self):
        self.ml = load_module("mlaunch", f"{MLAUNCH_DIR}/mlaunch.py")
        self.rig = self.ml.Rig(heavy=False)                      # verifies every mlaunch pin; REFUSES on a changed source
        self.man = json.load(open(RUNS_JSON, encoding="utf-8"))
        self.runs = {}
        for k in ("CTRL", "REP"):
            e = self.man["arms"][k]["0"]
            ck = e["train_dir"] + "/ckpt_last.pt"
            self.runs[k] = {"t1": e["t1"], "ckpt": ck, "train_dir": e["train_dir"]}
        self.T1 = None
        self.R = None

    def tool(self):
        if self.T1 is None:
            got = sha256_file(f"{TREE}/stack/scripts/v8f_t1_eval.py")
            if got != T1_TOOL_SHA:
                raise SystemExit(f"REFUSED: v8f_t1_eval.py changed ({got})")
            self.T1 = load_module("t1tool", f"{TREE}/stack/scripts/v8f_t1_eval.py")
        return self.T1

    def load_runs(self):
        if self.R is None:
            self.R = {k: self.ml.load_run(self.rig, v["t1"]) for k, v in self.runs.items()}
            for k, r in self.R.items():
                if not r["aligned"]:
                    raise SystemExit(f"REFUSED: {k} ticks not window-aligned with its dump")
        return self.R

    def window_set(self):
        """The mlaunch window set and the R1 arrays, with the recount controls."""
        R = self.load_runs()
        d = R["CTRL"]["d"]
        eid, g, v0 = np.asarray(d["eid"]), d["g"], d["v0"]
        mask = self.ml.launch_subset(v0, g)
        z = np.load(f"{MLAUNCH_DIR}/raw/mlaunch_windows.npz")
        out = {"eid": eid, "g": np.asarray(g, np.float64), "v0": np.asarray(v0, np.float64), "mask": mask, "clip_index": np.asarray(d["clip_index"]), "ws": np.asarray(d["ws"]),
               "gt1": z["v_gt_R1"], "own": {k: R[k]["v_own"] for k in R}, "fl": d["fl"], "cl": {k: R[k]["d"]["cl"] for k in R}}
        out["checks"] = {"launch_n_windows": int(mask.sum()), "launch_n_episodes": int(len(np.unique(eid[mask]))),
                         "npz_launch_equals_recomputed": bool(np.array_equal(z["launch"], mask)),
                         "npz_v_own_equals_ticks_CTRL": float(np.abs(z["v_own_R1_CTRL_s0"] - R["CTRL"]["v_own"]).max()),
                         "npz_v_own_equals_ticks_REP": float(np.abs(z["v_own_R1_REP_s0"] - R["REP"]["v_own"]).max()),
                         "gt_closed_form_minus_npz_max_abs": {f"tick{n}": float(np.abs(self.ml.gt_state_speed_cf(out["g"], n) - out["gt1"][:, n]).max()) for n in TICKS},
                         "npz_sha256": sha256_file(f"{MLAUNCH_DIR}/raw/mlaunch_windows.npz")}
        return out


# ----------------------------------------------------------------------------------------------------------------- H-CRAWL-01
def train_inputs(env):
    """The 306,566 TRAIN windows (digest-checked against the banked run), with v0, the 0.6 s speed change and v_GT at ticks 1, 2."""
    T1 = env.tool()
    D = T1.D
    sys.path[:0] = [f"{TREE}/stack", f"{TREE}/stack/scripts"]
    import v8f_mini_train as TR  # noqa: E402  (the trainer's own parser / make_config, as measure_f.py does)
    pa = TR.build_parser().parse_args(DAGOACC_ARGV)
    cfg = TR.make_config(pa)
    first = D.effective_window_start(cfg.hist, cfg.window_start)
    ents = {}
    for ln in open(MANIFEST_TRAIN, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            ents[r["clip"]] = int(r["n_frames"])
    clips_all = list(ents)
    ex = set(sorted(clips_all, key=D.sha12)[540:600])
    clips = D.select_train_clips([c for c in clips_all if c not in ex], pa.data_frac)
    wins = [(c, t) for c in clips for t in range(first, ents[c] - D.HORIZON)]
    digest = hashlib.sha256("\n".join(f"{c}:{t}" for c, t in wins).encode("utf-8")).hexdigest()
    if digest != WINDOW_LIST_SHA:
        raise SystemExit(f"REFUSED: the rebuilt train window list is not the banked one ({digest})")
    tracks = D.load_native_tracks(TRACKS)
    V0, DV, VG, CL = [], [], [], []
    for c in clips:
        tr = tracks[c]
        pg = tr[0::2]
        T = len(pg)
        assert T == ents[c], (c, T, ents[c])
        kin3, _ = D.yaw_kin_from_track(tr)
        gt, gok = D.gt_future(pg)
        ts = np.arange(first, T - D.HORIZON)
        assert gok[ts].all()
        v0 = kin3[ts, 0]
        V0.append(v0)
        DV.append(v0 - kin3[ts - D_LAG, 0])
        g = gt[ts]
        VG.append(np.stack([env.ml.gt_state_speed_cf(g, 1), env.ml.gt_state_speed_cf(g, 2)], 1))
        CL.append(np.full(len(ts), len(CL)))
    out = {"v0": np.concatenate(V0), "dv": np.concatenate(DV), "vgt": np.concatenate(VG), "clip_i": np.concatenate(CL), "n_clips": len(clips), "n_windows": len(wins),
           "digest": digest, "first": first, "clips": clips}
    return out, tracks


def spot_check_train_gt(env, tracks, tr, n=300):
    """c5: the closed-form train v_GT against the tool's own state rule (mlaunch's pinned ``state_from_knots``) on ``n`` random train windows."""
    T1 = env.tool()
    D = T1.D
    ns2 = {"np": np, "math": math, "T1": T1, "D": D}
    env.ml.exec_from(env.ml.PINS["statekin.py"][0], ns2, funcs=("state_from_knots",))
    rng = np.random.default_rng(SEED)
    pick = rng.choice(tr["n_windows"], size=n, replace=False)
    clips, first = tr["clips"], tr["first"]
    lens = None
    worst = 0.0
    # map a flat window index to (clip, t)
    ts_per_clip = []
    for c in clips:
        ts_per_clip.append(len(tracks[c][0::2]) - D.HORIZON - first)
    cum = np.concatenate([[0], np.cumsum(ts_per_clip)])
    for w in pick:
        ci = int(np.searchsorted(cum, w, side="right") - 1)
        t = first + int(w - cum[ci])
        pg = tracks[clips[ci]][0::2]
        gt, _ = D.gt_future(pg)
        g = gt[t]
        K = np.vstack([np.zeros((1, 2)), g])
        for k, nn in enumerate((1, 2)):
            kin, _ = ns2["state_from_knots"](K, 10 * (nn - 1))
            worst = max(worst, abs(float(kin[0]) - float(tr["vgt"][w, k])))
    return {"n": int(n), "max_abs_tool_state_rule_minus_closed_form": worst, "PASS": bool(worst < 1e-5)}


def eval_history(env, ws_out):
    """d = kin3[t0,0] - kin3[t0-3,0] for every eval window, from the clip's own native track, rows <= t0 only."""
    T1 = env.tool()
    D = T1.D
    tracks = D.load_native_tracks(TRACKS)
    J = D.load_join(EVAL_JOIN)
    cache = {}
    dv = np.full(len(ws_out["ws"]), np.nan)
    for i, (ci, t0) in enumerate(zip(ws_out["clip_index"], ws_out["ws"])):
        clip = str(J["clip"][int(ci)])
        if clip not in cache:
            cache[clip] = D.yaw_kin_from_track(tracks[clip])[0]
        k = cache[clip]
        assert t0 - D_LAG >= 0
        dv[i] = k[int(t0), 0] - k[int(t0) - D_LAG, 0]
        assert abs(k[int(t0), 0] - ws_out["v0"][i]) < 1e-6, ("eval v0 mismatch", i)
    return dv


def run_floor(env, ws, path):
    """P10 floor re-run through the tool's own ``floor_planner`` + ``rollout`` on every dump window; records p10f[:, -1, 0] at ticks 0..2 (what TickRecorder stores as v0)."""
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    import torch
    T1 = env.tool()
    D, M = T1.D, T1.M
    torch.set_num_threads(4)
    ck = torch.load(env.runs["CTRL"]["ckpt"], map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    tracks = D.load_native_tracks(TRACKS)
    J = D.load_join(EVAL_JOIN)
    store = D.ClipStore(EVAL_CACHE, "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    proxy = T1._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    dev = torch.device("cpu")
    wins = [(int(c), int(t), 0, 0) for c, t in zip(ws["clip_index"], ws["ws"])]
    v0t = np.full((len(wins), 3), np.nan)
    fl = []
    t_start = time.time()
    for b0 in range(0, len(wins), 32):
        chunk = wins[b0:b0 + 32]
        rec = {}

        def hook(n, batch, plan, states, rec=rec):
            rec[n] = batch["p10f"][:, -1, 0].double().cpu().numpy()
        st = T1.make_states(store, tracks, J, chunk, "deadreckon")
        _, exe = T1.rollout(T1.floor_planner(model, dev), st, ds, proxy, m=10, schedule="periodic", scene="logged", limit="causal", device=dev, hook=hook)
        for n in range(3):
            v0t[b0:b0 + len(chunk), n] = rec[n]
        fl += exe
        if (b0 // 32) % 10 == 0:
            print(f"  floor re-run [{b0 + len(chunk)}/{len(wins)}] {time.time() - t_start:.0f}s", flush=True)
    out = {"v_own_p10": v0t, "fl": np.asarray(fl)}
    np.savez(path, **out)
    return out


def p10_same_in_both_ckpts(env):
    import torch
    sd = {}
    for k in ("CTRL", "REP"):
        ck = torch.load(env.runs[k]["ckpt"], map_location="cpu", weights_only=False)
        sd[k] = {n: t for n, t in ck["model"].items() if n.startswith("prior.")}
    same = set(sd["CTRL"]) == set(sd["REP"]) and all(torch.equal(sd["CTRL"][n], sd["REP"][n]) for n in sd["CTRL"])
    return {"n_prior_tensors": len(sd["CTRL"]), "identical_in_DAG_OACC_and_REP": bool(same)}


def task_01(env, out_path):
    t_start = time.time()
    ml, rig = env.ml, env.rig
    ws = env.window_set()
    eid, mask, v0 = ws["eid"], ws["mask"], ws["v0"]
    gt1 = ws["gt1"]
    controls = {}
    # ---- c4: the stored gaps are reproduced
    gap = {k: window_gap(ws["own"][k], gt1) for k in ("CTRL", "REP")}
    controls["c4_stored_gaps_and_counts"] = {"DAG-OACC_mean_gap_327": float(gap["CTRL"][mask].mean()), "REP_mean_gap_327": float(gap["REP"][mask].mean()),
                                              "n_windows": int(mask.sum()), "n_episodes": int(len(np.unique(eid[mask]))), "recount": ws["checks"],
                                              "PASS": bool(abs(gap["CTRL"][mask].mean() + 0.8747) < 5e-5 and abs(gap["REP"][mask].mean() + 1.4307) < 5e-5 and mask.sum() == 327
                                                           and len(np.unique(eid[mask])) == 49 and ws["checks"]["npz_launch_equals_recomputed"] and ws["checks"]["npz_v_own_equals_ticks_CTRL"] == 0.0
                                                           and ws["checks"]["npz_v_own_equals_ticks_REP"] == 0.0)}
    # ---- the table, from TRAIN windows only
    print("building the train table ...", flush=True)
    tr, tracks = train_inputs(env)
    vb_tr, dc_tr = v_bucket(tr["v0"]), d_class(tr["dv"])
    ok = vb_tr >= 0
    tab = build_table(vb_tr[ok], dc_tr[ok], tr["vgt"][ok])
    controls["c5_train_closed_form_vs_tool_state_rule"] = spot_check_train_gt(env, tracks, tr)
    # c1 / c2: the table on its own cell reads exactly 0 median / mean bias (train windows)
    cell_tr = cell_of(vb_tr, dc_tr)
    c1 = c2 = 0.0
    for c in range(N_V * N_D):
        if c in tab["fallback_cells"]:
            continue
        m = cell_tr == c
        if m.any():
            c1 = max(c1, float(np.abs(np.median(tab["median"][c][None] - tr["vgt"][m], axis=0)).max()))
            c2 = max(c2, float(np.abs((tab["mean"][c][None] - tr["vgt"][m]).mean(axis=0)).max()))
    controls["c1_median_table_own_cell_median_bias"] = {"max_abs": c1, "PASS": bool(c1 < 1e-12)}
    controls["c2_mean_table_own_cell_mean_bias"] = {"max_abs": c2, "PASS": bool(c2 < 1e-9)}
    # c3: deliberate regression, the table rebuilt from v_GT shuffled across the train windows
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(int(ok.sum()))
    tab_sh = build_table(vb_tr[ok], dc_tr[ok], tr["vgt"][ok][perm])
    spread = lambda t: float(np.nanstd(np.where(t["count"][:, None] >= MIN_CELL, t["median"], np.nan)))   # noqa: E731
    # ---- eval side: the history class and the table prediction
    dv_ev = eval_history(env, ws)
    vb_ev, dc_ev = v_bucket(v0), d_class(dv_ev)
    vgt_ev = np.stack([gt1[:, 1], gt1[:, 2]], 1)
    pred = {"median": predict(tab, vb_ev, dc_ev, "median"), "mean": predict(tab, vb_ev, dc_ev, "mean"),
            "median_vonly": predict(tab, vb_ev, dc_ev, "median", vonly=True), "mean_vonly": predict(tab, vb_ev, dc_ev, "mean", vonly=True)}
    pred_sh = predict(tab_sh, vb_ev, dc_ev, "median")
    g_med_real = float(blind_gap(pred["median"], vgt_ev)[mask].mean())
    g_med_shuf = float(blind_gap(pred_sh, vgt_ev)[mask].mean())
    controls["c3_deliberate_regression_shuffled_vgt_table"] = {"outcome_blind_gap_real": g_med_real, "outcome_blind_gap_shuffled": g_med_shuf, "between_cell_spread_real": spread(tab),
                                                                "between_cell_spread_shuffled": spread(tab_sh),
                                                                "PASS": bool(abs(g_med_real - g_med_shuf) > 1e-3 and spread(tab) > spread(tab_sh))}
    controls["c7_perfect_predictor_reads_zero"] = {"max_abs": float(np.abs(blind_gap(vgt_ev, vgt_ev)).max()), "PASS": bool(np.all(blind_gap(vgt_ev, vgt_ev) == 0.0))}
    # ---- P10
    print("P10 floor re-run ...", flush=True)
    p10 = run_floor(env, ws, f"{HERE}/raw/p10_rerun.npz")
    gap["P10"] = window_gap(p10["v_own_p10"], gt1)
    controls["c6_p10_rerun_equals_dump_fl"] = {"max_abs_m": float(np.abs(p10["fl"] - ws["fl"]).max()), "n_windows": int(len(ws["fl"])), "PASS": bool(np.abs(p10["fl"] - ws["fl"]).max() < 1e-4),
                                               "prior_tensors_same_in_both_ckpts": p10_same_in_both_ckpts(env)}
    # R2 (executed-chord) reading of P10 beside R1, for the consistency of the two routes
    # c6 failed strictly if the float-level re-run differs from the dump: BOUND the effect on the quantity that is read (chord speeds at ticks 1, 2), numbers only, no threshold (post hoc, not registered)
    def _chords(fl_):
        return np.stack([np.zeros(len(fl_))] + [ml.chord_speed(fl_, n) for n in TICKS], 1)
    ch_dump, ch_rerun = _chords(ws["fl"]), _chords(p10["fl"])
    dch = (ch_rerun - ch_dump)[:, 1:]
    controls["c6_p10_rerun_equals_dump_fl"]["impact_bound_post_hoc"] = {
        "rerun_minus_dump_chord_speed_ticks_1_2_all_2399": {"max_abs_mps": float(np.abs(dch).max()), "mean_signed_mps": float(dch.mean()), "n_windows_differing_over_1e-4_m": int((np.abs(p10["fl"] - ws["fl"]).max(axis=(1, 2)) > 1e-4).sum())},
        "rerun_minus_dump_chord_speed_ticks_1_2_launch_327": {"max_abs_mps": float(np.abs(dch[mask]).max()), "mean_signed_mps": float(dch[mask].mean())},
        "windows_bit_identical": int((np.abs(p10["fl"] - ws["fl"]).max(axis=(1, 2)) == 0).sum()),
        "diagnosis_files": ["raw/c6_thread_diagnosis_chunk_matched.json (4 vs 16 torch threads at the original 32-window chunk alignment: neither reproduces the dump; with 4 threads 144 of 192 windows are bit-identical, with 16 none)",
                            "raw/c6_thread_diagnosis_first_FLAWED_default_arm.json (its 'threads=default' arm was invalid: set_num_threads(None) left 4 threads; its bsz=1 arm shows batch composition changes the float result)"],
        "reading": "float-level nondeterminism (thread partition / batch composition) amplified by the closed-loop rollout: the discrepancy grows with the rollout step; the original run's thread setting is unknown and not recoverable"}
    gt2 = np.load(f"{MLAUNCH_DIR}/raw/mlaunch_windows.npz")["v_gt_R2"]
    r2 = {"P10_R2_from_dump_fl": window_gap(ch_dump, gt2), "P10_R2_from_rerun_fl": window_gap(ch_rerun, gt2)}
    for k in ("CTRL", "REP"):
        r2[k] = window_gap(np.load(f"{MLAUNCH_DIR}/raw/mlaunch_windows.npz")[f"v_own_R2_{k}_s0"], gt2)
    # ---- subsets and rows
    stay = stay_mask(v0, ws["g"])
    subsets = {"launch_327": mask, "launch_stopped_v0_lt_0.5": mask & (v0 < STOP_V), "launch_moving_v0_ge_0.5": mask & (v0 >= STOP_V), "gt_stay_stopped": stay}
    rows = {}
    for sn, m in subsets.items():
        e = eid[m]
        r = {"n_windows": int(m.sum()), "n_episodes": int(len(np.unique(e)))}
        if m.sum() == 0:
            rows[sn] = r
            continue
        r["model_gap"] = {RUN_NAME[k]: mean_ci(rig, gap[k][m], e) for k in ("CTRL", "REP")}
        r["P10_gap"] = mean_ci(rig, gap["P10"][m], e)
        r["outcome_blind_gap"] = {s: mean_ci(rig, blind_gap(pred[s], vgt_ev)[m], e) for s in pred}
        r["model_minus_outcome_blind"] = {RUN_NAME[k]: {s: diff_ci(rig, gap[k][m], blind_gap(pred[s], vgt_ev)[m], e) for s in pred} for k in ("CTRL", "REP")}
        r["P10_minus_outcome_blind_median"] = diff_ci(rig, gap["P10"][m], blind_gap(pred["median"], vgt_ev)[m], e)
        r["model_minus_P10"] = {RUN_NAME[k]: diff_ci(rig, gap[k][m], gap["P10"][m], e) for k in ("CTRL", "REP")}
        r["REP_minus_DAG-OACC"] = diff_ci(rig, gap["REP"][m], gap["CTRL"][m], e)
        r["R2_executed_chord_reading_beside"] = {"DAG-OACC": mean_ci(rig, r2["CTRL"][m], e), "REP": mean_ci(rig, r2["REP"][m], e), "P10_from_dump_fl_no_rerun": mean_ci(rig, r2["P10_R2_from_dump_fl"][m], e),
                                                 "P10_from_rerun_fl": mean_ci(rig, r2["P10_R2_from_rerun_fl"][m], e),
                                                 "DAG-OACC_minus_P10_dump": diff_ci(rig, r2["CTRL"][m], r2["P10_R2_from_dump_fl"][m], e), "REP_minus_P10_dump": diff_ci(rig, r2["REP"][m], r2["P10_R2_from_dump_fl"][m], e)}
        r["mean_v0"] = float(v0[m].mean())
        r["mean_vgt_ticks12"] = float(vgt_ev[m].mean())
        rows[sn] = r
    # the table cells hit by the 327 windows (coverage; the table is not changed by it)
    cells_hit = np.bincount(cell_of(vb_ev, dc_ev)[mask], minlength=N_V * N_D).tolist()
    # ---- verdicts (READINGS R1.8), per run, for the median and the mean table
    stay_gap = {k: float(gap[k][stay].mean()) for k in ("CTRL", "REP")}
    verdicts = {}
    for k in ("CTRL", "REP"):
        dd = rows["launch_327"]["model_minus_outcome_blind"][RUN_NAME[k]]
        verdicts[RUN_NAME[k]] = {s: verdict_h01(dd[s]["delta"], dd[s]["lo"], dd[s]["hi"], stay_gap[k]) for s in ("median", "mean", "median_vonly")}
    res = {"task": "H-CRAWL-01",
           "evidence": "MEASURED (ours; crawl_diag.py -> this file). Tier T1-kin for every model / P10 gap (own executed state, logged camera), never 'closed loop'. Estimator: taniteval.ci episode-cluster "
                       "bootstrap (paired for differences), B=2000, seed 0 over the episodes of each subset: it answers 'another draw of EPISODES?' only, blind to training variance (H-ESTIM-SEED-1).",
           "train": {"n_windows": tr["n_windows"], "n_clips": tr["n_clips"], "window_list_sha256": tr["digest"], "n_in_table": int(ok.sum()), "n_v0_ge_6_or_invalid": int((~ok).sum()),
                     "cell_counts": tab["count"].tolist(), "speed_bucket_counts": tab["marg_count"].tolist(), "fallback_cells": tab["fallback_cells"],
                     "median_v_gt_tick1_by_cell": tab["median"][:, 0].tolist(), "median_v_gt_tick2_by_cell": tab["median"][:, 1].tolist(),
                     "mean_v_gt_tick1_by_cell": tab["mean"][:, 0].tolist(), "mean_v_gt_tick2_by_cell": tab["mean"][:, 1].tolist(),
                     "edges": {"v": list(V_EDGES), "d": list(D_EDGES), "d_lag_cache_steps": D_LAG, "min_cell": MIN_CELL}},
           "eval_cells_hit_by_the_327": cells_hit, "n_eval_windows_without_bucket": int((vb_ev[mask] < 0).sum()),
           "rows": rows, "model_gap_on_gt_stay_point": stay_gap, "verdicts": verdicts, "controls": controls,
           "seconds": time.time() - t_start}
    controls["failed"] = [k for k, v in controls.items() if isinstance(v, dict) and "PASS" in v and not v["PASS"]]
    controls["PASS"] = bool(not controls["failed"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w", encoding="utf-8", newline="\n"), indent=1, default=_jd)
    np.savez_compressed(f"{HERE}/raw/crawl01_arrays.npz", launch=mask, stay=stay, v0=v0, dv=dv_ev, vb=vb_ev, dc=dc_ev, vgt=vgt_ev, eid=eid,
                        gap_CTRL=gap["CTRL"], gap_REP=gap["REP"], gap_P10=gap["P10"], **{f"pred_{s}": pred[s] for s in pred})
    print(f"H-CRAWL-01 done in {time.time() - t_start:.0f}s -> {out_path}; controls {'ALL PASS' if controls['PASS'] else 'FAILED: ' + ', '.join(controls['failed'])}", flush=True)
    return res


# ----------------------------------------------------------------------------------------------------------------- H-CRAWL-02
def apply_cf(T1, st, spec):
    """Counterfactual speed history on a fresh WindowState, in place (READINGS R2.2): the native-track speed column (rows <= 2 t0) and the cache speed (rows <= t0) are changed, then every derived
    input is rebuilt with the tool's own functions.  kind: shift (v' = max(v + delta, 0)), flat (history := current speed), zero (speed := 0)."""
    D, fk = T1.D, T1.fk
    t0, d, tr = st.t0, st.d, st.track
    n = 2 * t0 + 1
    v_tr = tr[:n, 3].copy()
    v_ca = np.asarray(d["v"][:t0 + 1], np.float64).copy()
    kind = spec[0]
    if kind == "shift":
        nv_tr, nv_ca = shift_speed(v_tr, spec[1]), shift_speed(v_ca, spec[1])
    elif kind == "flat":
        nv_tr, nv_ca = flat_speed(v_tr, v_tr[-1]), flat_speed(v_ca, v_ca[-1])
    elif kind == "zero":
        nv_tr, nv_ca = zero_speed(v_tr), zero_speed(v_ca)
    else:
        raise ValueError(kind)
    tr[:n, 3] = nv_tr
    d["v"][:t0 + 1] = nv_ca.astype(d["v"].dtype)
    kin, om = D.yaw_kin_from_track(tr[:n])
    d["kin3"][1:t0 + 1] = kin[1:]
    d["ch1"][1:t0 + 1] = np.nan_to_num(om[1:], nan=0.0).astype(d["ch1"].dtype)
    d["p10f"][1:t0 + 1] = fk.kin_features(kin[1:]).astype(np.float32)
    d["base"][1:t0 + 1] = D.kdx_base(kin[1:], D.KIN_UNITS["yaw"])


def _same(a, b):
    """Bit-identity of two tensors, NaN == NaN."""
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
    return bool(torch.equal(a, b))


def _maxdiff(a, b):
    import torch
    return float(torch.nan_to_num((a.double() - b.double()).abs(), nan=0.0).max())


def _donor_from_pool(eid_w, eid_all, pool, seed):
    """Donor (global dump index) for every window, drawn from ``pool`` (global indices) with a different episode."""
    rs = np.random.default_rng(seed)
    out = np.empty(len(eid_w), np.int64)
    for i in range(len(eid_w)):
        cand = pool[eid_all[pool] != eid_w[i]]
        out[i] = cand[rs.integers(len(cand))]
    return out


ARMS = [("base", None), ("delta0", ("shift", 0.0)), ("dm05", ("shift", -0.5)), ("dp05", ("shift", 0.5)), ("dp10", ("shift", 1.0)), ("hist_flat", ("flat",)), ("v_zero", ("zero",)),
        ("img_launch", ("swap", "launch")), ("img_stay", ("swap", "stay")), ("img_self", ("swap", "self")), ("scramble_old_p10f", ("scramble",))]
DELTA_ARMS = {"dm05": -0.5, "dp05": 0.5, "dp10": 1.0}


def task_02(env, out_path):
    import torch
    t_start = time.time()
    rig = env.rig
    ws = env.window_set()
    T1 = env.tool()
    D, M = T1.D, T1.M
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    eid_all, mask, v0_all = ws["eid"], ws["mask"], ws["v0"]
    L = np.flatnonzero(mask)                                                  # launch windows, in dump order
    eid, v0 = eid_all[L], v0_all[L]
    stay_idx = np.flatnonzero(stay_mask(v0_all, ws["g"]))
    tracks = D.load_native_tracks(TRACKS)
    J = D.load_join(EVAL_JOIN)
    models, cfgs = {}, {}
    ck_sha = {}
    for k in ("CTRL", "REP"):
        ck = torch.load(env.runs[k]["ckpt"], map_location="cpu", weights_only=False)
        cfg = M.V8FMiniConfig(**ck["cfg"])
        assert cfg.kin_mode == "yaw"
        m = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        m.load_state_dict(ck["model"])
        m.eval()
        assert m.hacc_w is None
        models[k], cfgs[k] = m, cfg
        ck_sha[k] = sha256_file(env.runs[k]["ckpt"])
        assert ck_sha[k].startswith(CKPT_PREFIX[k]), (k, ck_sha[k])
    cfg = cfgs["CTRL"]
    store = D.ClipStore(EVAL_CACHE, "eval", lru=48, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    proxy = T1._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    don_launch = L[donors_other_episode(eid, np.arange(len(L)), SEED)]          # global indices
    don_stay = _donor_from_pool(eid, eid_all, stay_idx, SEED + 1)
    N = len(L)
    keys = sorted(T1.MODEL_KEYS)

    def item_of(gi, spec=None):
        st = T1.make_states(store, tracks, J, [(int(ws["clip_index"][gi]), int(ws["ws"][gi]), 0, 0)], "deadreckon")[0]
        if spec is not None and spec[0] in ("shift", "flat", "zero"):
            apply_cf(T1, st, spec)
        return T1.build_item(ds, proxy, st, 0, 0, 0, "logged", "causal", None)

    res = {a: {"v_in": np.full(N, np.nan), **{k: {"plan": np.full((N, 30, 2), np.nan), "prior": np.full((N, 30, 2), np.nan)} for k in ("CTRL", "REP")}} for a, _ in ARMS}
    ctl = {"delta0_batch_max_abs_diff": 0.0, "delta0_batch_keys_not_equal": [], "scramble_p10f_rows_changed": True, "swap_self_batch_equal": True}
    rng = np.random.default_rng(SEED)
    print(f"H-CRAWL-02: {N} launch windows, {len(stay_idx)} GT-stay donors", flush=True)
    for b0 in range(0, N, 32):
        idx = np.arange(b0, min(b0 + 32, N))
        batches = {}
        base_items = [item_of(int(L[i])) for i in idx]
        for arm, spec in ARMS:
            if arm == "base":
                items = base_items
            elif spec[0] in ("shift", "flat", "zero"):
                items = [item_of(int(L[i]), spec) for i in idx]
            elif spec[0] == "swap":
                items = []
                for j, i in enumerate(idx):
                    it = dict(base_items[j])
                    if spec[1] == "launch":
                        dn = item_of(int(don_launch[i]))
                    elif spec[1] == "stay":
                        dn = item_of(int(don_stay[i]))
                    else:
                        dn = base_items[j]
                    it["feats"], it["frame_ok"] = dn["feats"], dn["frame_ok"]
                    items.append(it)
            elif spec[0] == "scramble":
                items = []
                for j in range(len(idx)):
                    it = dict(base_items[j])
                    p = np.array(it["p10f"], copy=True)
                    p[:-1] = rng.normal(size=p[:-1].shape).astype(np.float32) * 10.0
                    it["p10f"] = p
                    items.append(it)
            full = D.collate(items)
            batch = {k: full[k] for k in T1.MODEL_KEYS}
            batches[arm] = batch
            res[arm]["v_in"][idx] = batch["p10f"][:, -1, 0].double().numpy()
            for rk in ("CTRL", "REP"):
                with torch.no_grad():
                    out = models[rk].forward_infer(batch)
                res[arm][rk]["plan"][idx] = out["traj30"].double().numpy()
                res[arm][rk]["prior"][idx] = M.subsample_02(M.upsample_prior(out["p10"])).double().numpy()
        # controls that compare tensors
        for k in keys:
            a, b = batches["base"][k], batches["delta0"][k]
            if not _same(a, b):
                ctl["delta0_batch_keys_not_equal"].append(k)
                ctl["delta0_batch_max_abs_diff"] = max(ctl["delta0_batch_max_abs_diff"], _maxdiff(a, b))
            if not _same(batches["base"][k], batches["img_self"][k]):
                ctl["swap_self_batch_equal"] = False
        ctl["scramble_p10f_rows_changed"] &= bool((batches["scramble_old_p10f"]["p10f"][:, :-1] != batches["base"]["p10f"][:, :-1]).any())
        print(f"  [{idx[-1] + 1}/{N}] {time.time() - t_start:.0f}s", flush=True)
    ctl["delta0_batch_keys_not_equal"] = sorted(set(ctl["delta0_batch_keys_not_equal"]))
    # ---- the stored tick-0 plan of the ticks file
    Rr = env.load_runs()
    stored = {k: float(np.abs(Rr[k]["tk"]["plan"][L, 0] - res["base"][k]["plan"]).max()) for k in ("CTRL", "REP")}
    # ---- analysis
    sub = {"stopped_start_v0_lt_0.5": v0 < STOP_V, "all_327": np.ones(N, bool), "moving_v0_ge_0.5": v0 >= STOP_V}
    ref_arm = "base" if not ctl["delta0_batch_keys_not_equal"] else "delta0"

    def vo(arm, rk, which, h):
        p = res[arm][rk][which]
        return out_chord_end(p) if h == "chord" else out_speed(p, h)

    analysis = {}
    raw_dv = {}
    for rk in ("CTRL", "REP"):
        A = {}
        for which in ("plan", "prior", "resid"):
            for h in (10, 20, "chord"):
                def vfun(arm, rk=rk, which=which, h=h):
                    if which == "resid":
                        return vo(arm, rk, "plan", h) - vo(arm, rk, "prior", h)
                    return vo(arm, rk, which, h)
                dvo = {a: vfun(a) - vfun(ref_arm) for a, _ in ARMS}
                dvi = {a: res[a]["v_in"] - res[ref_arm]["v_in"] for a, _ in ARMS}
                pool_o = sum(dvo[a] for a in DELTA_ARMS)
                pool_i = sum(dvi[a] for a in DELTA_ARMS)
                key = f"{which}_h{h}"
                raw_dv[(rk, key)] = (dvo, dvi, pool_o, pool_i)
                per = {}
                for sn, sm in sub.items():
                    e = eid[sm]
                    r = {"n_windows": int(sm.sum()), "n_episodes": int(len(np.unique(e)))}
                    if sm.sum() == 0:
                        per[sn] = r
                        continue
                    r["S_v_pooled"] = ratio_ci(rig.ci, pool_o[sm], pool_i[sm], e)
                    r["S_v_by_delta"] = {f"{DELTA_ARMS[a]:+.1f}": ratio_ci(rig.ci, dvo[a][sm], dvi[a][sm], e) for a in DELTA_ARMS}
                    r["mean_dv_in_by_delta"] = {f"{DELTA_ARMS[a]:+.1f}": float(dvi[a][sm].mean()) for a in DELTA_ARMS}
                    r["mean_dv_out_by_delta"] = {f"{DELTA_ARMS[a]:+.1f}": float(dvo[a][sm].mean()) for a in DELTA_ARMS}
                    for a in ("hist_flat", "v_zero", "img_launch", "img_stay"):
                        r[a] = {"mean_dv_out": mean_ci(rig, dvo[a][sm], e), "mean_abs_dv_out": float(np.abs(dvo[a][sm]).mean()), "mean_dv_in": float(dvi[a][sm].mean())}
                    if sn == "moving_v0_ge_0.5":
                        r["S_v_vzero_ratio"] = ratio_ci(rig.ci, dvo["v_zero"][sm], dvi["v_zero"][sm], e)
                    per[sn] = r
                A[key] = per
        analysis[RUN_NAME[rk]] = A
    # run difference of the pooled S_v (paired), per subset, for plan h=10
    diff = {}
    for which in ("plan", "prior", "resid"):
        for h in (10, 20, "chord"):
            key = f"{which}_h{h}"
            xo, xi = raw_dv[("CTRL", key)][2], raw_dv[("CTRL", key)][3]
            yo, yi = raw_dv[("REP", key)][2], raw_dv[("REP", key)][3]
            diff[key] = {}
            for sn, sm in sub.items():
                if sm.sum() == 0 or float(np.sum(xi[sm])) == 0.0:
                    continue
                diff[key][sn] = paired_ratio_ci(rig.ci, xo[sm], xi[sm], yo[sm], yi[sm], eid[sm])
    # ---- controls
    d0 = {rk: float(np.abs(res["delta0"][rk]["plan"] - res["base"][rk]["plan"]).max()) for rk in ("CTRL", "REP")}
    scr = {rk: float(np.abs(res["scramble_old_p10f"][rk]["plan"] - res["base"][rk]["plan"]).max()) for rk in ("CTRL", "REP")}
    swf = {rk: float(np.abs(res["img_self"][rk]["plan"] - res["base"][rk]["plan"]).max()) for rk in ("CTRL", "REP")}
    din = {a: {"max_abs_realised_minus_requested_where_unclipped": float(np.abs((res[a]["v_in"] - res[ref_arm]["v_in"] - dlt)[(v0 + dlt) >= 0]).max())} for a, dlt in DELTA_ARMS.items()}
    mv = sub["moving_v0_ge_0.5"]
    pos = {RUN_NAME[rk]: mean_ci(rig, (vo("v_zero", rk, "plan", 10) - vo(ref_arm, rk, "plan", 10))[mv], eid[mv]) for rk in ("CTRL", "REP")}
    controls = {"delta0_equals_untouched_tool_path": {"batch_keys_not_equal": ctl["delta0_batch_keys_not_equal"], "batch_max_abs_diff": ctl["delta0_batch_max_abs_diff"], "plan_max_abs_diff": d0,
                                                       "PASS": bool(not ctl["delta0_batch_keys_not_equal"] and max(d0.values()) == 0.0), "reference_arm_used": ref_arm},
                "delta0_reads_exactly_zero": {"max_abs_dv_out": {rk: float(np.abs(out_speed(res["delta0"][rk]["plan"], 10) - out_speed(res[ref_arm][rk]["plan"], 10)).max()) for rk in ("CTRL", "REP")},
                                              "PASS": bool(all(float(np.abs(out_speed(res["delta0"][rk]["plan"], 10) - out_speed(res[ref_arm][rk]["plan"], 10)).max()) == 0.0 for rk in ("CTRL", "REP"))
                                                           if ref_arm == "base" else True)},
                "positive_control_v_in_zero_lowers_v_out_on_moving_windows": {"mean_dv_out_h10": pos, "PASS": bool(all(p["hi"] < 0.0 for p in pos.values()))},
                "older_p10f_rows_scramble_changes_output_by_exactly_zero": {"batch_rows_changed": ctl["scramble_p10f_rows_changed"], "plan_max_abs_diff": scr,
                                                                            "PASS": bool(ctl["scramble_p10f_rows_changed"] and max(scr.values()) == 0.0)},
                "image_swap_with_own_image_is_identity": {"batch_equal": ctl["swap_self_batch_equal"], "plan_max_abs_diff": swf, "PASS": bool(ctl["swap_self_batch_equal"] and max(swf.values()) == 0.0)},
                "my_tick0_plan_reproduces_stored_ticks_file_plan": {"max_abs_m": stored, "PASS": bool(max(stored.values()) < 1e-3)},
                "realised_input_shift_equals_requested": {"detail": din, "PASS": bool(all(v["max_abs_realised_minus_requested_where_unclipped"] < 1e-5 for v in din.values()))},
                "image_swap_changes_the_batch": {"launch_donor_feats_differ": bool(not np.array_equal(res["img_launch"]["CTRL"]["plan"], res["base"]["CTRL"]["plan"])), "PASS": bool(not np.array_equal(res["img_launch"]["CTRL"]["plan"], res["base"]["CTRL"]["plan"]))}}
    controls["PASS"] = bool(all(v["PASS"] for v in controls.values()))
    # ---- verdicts (READINGS R2.6): pooled, primary subset, h = 10, plan
    prim = {rk: analysis[RUN_NAME[rk]]["plan_h10"]["stopped_start_v0_lt_0.5"]["S_v_pooled"] for rk in ("CTRL", "REP")}
    verdict = verdict_h02(prim["CTRL"], prim["REP"])
    dd = diff["plan_h10"].get("stopped_start_v0_lt_0.5")
    verdict2 = None
    if dd is not None:
        verdict2 = {"S_v_DAG-OACC_minus_REP": dd, "equal_in_the_weak_sense_CI_contains_0": bool(dd["contains_zero"]), "CI_inside_pm_0.15": bool(dd["lo"] > -0.15 and dd["hi"] < 0.15),
                    "launch_gap_REP_minus_DAG-OACC_stored": -0.5560, "REP_has_larger_S_v": bool(prim["REP"]["S_v"] > prim["CTRL"]["S_v"])}
    out = {"task": "H-CRAWL-02",
           "evidence": "MEASURED (ours; crawl_diag.py -> this file). Tier: first-tick inference on the T1-kin windows' tick 0 = T0-class (teacher-forced, one forward_infer call, no rollout), never 'driving performance'. "
                       "Counterfactual inputs are measurement tools and are never deployed. Estimator: taniteval.ci episode-cluster bootstrap B=2000 seed 0 (ratio of sums for S_v; paired on the same episodes for the "
                       "run difference): 'another draw of EPISODES?' only.",
           "forward_call": "stack/tanitad/models/v8f_mini.py:773 V8FMini.forward_infer -> _forward :726 -> run_s :672 / run_t :683 / run_o :698 (prior(p10f[NOW], base) :710); batch built by v8f_t1_eval.py build_item :237 + collate, "
                           "keys MODEL_KEYS :62; ages (0,0) at tick 0 (ages_for :136)",
           "ckpt_sha256": ck_sha, "n_launch": N, "n_episodes": int(len(np.unique(eid))), "n_stopped_start": int((v0 < STOP_V).sum()), "n_episodes_stopped_start": int(len(np.unique(eid[v0 < STOP_V]))),
           "v_in_def": "the realised current speed input p10f[:, -1, 0] after the counterfactual rebuild", "v_out_def": "|plan[h]-plan[h-2]|/0.4 at h=10 (2 s, primary), h=20 (4 s), and the final chord |plan[29]-plan[28]|/0.2",
           "analysis": analysis, "run_difference_S_v": diff, "verdict_falsifier_1": verdict, "verdict_falsifier_2": verdict2, "controls": controls, "seconds": time.time() - t_start}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8", newline="\n"), indent=1, default=_jd)
    np.savez_compressed(f"{HERE}/raw/crawl02_arrays.npz", launch_idx=L, v0=v0, eid=eid, **{f"{a}_{rk}_plan": res[a][rk]["plan"] for a, _ in ARMS for rk in ("CTRL", "REP")},
                        **{f"{a}_v_in": res[a]["v_in"] for a, _ in ARMS})
    print(f"H-CRAWL-02 done in {time.time() - t_start:.0f}s -> {out_path}; controls {'ALL PASS' if controls['PASS'] else 'FAILED'}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------------------------- H-CRAWL-03
def _walk_depth(root, maxdepth, skip=()):
    root = root.replace("\\", "/").rstrip("/")
    base = root.count("/")
    for dp, dn, fn in os.walk(root):
        dpn = dp.replace("\\", "/")
        if any(sk in dpn for sk in skip):
            dn[:] = []
            continue
        if dpn.count("/") - base >= maxdepth:
            dn[:] = []
        for f in fn:
            yield dpn + "/" + f


WEIGHT_EXT = (".pt", ".pth", ".safetensors", ".ckpt")


def task_03(env, out_path):
    """Feasibility only (READINGS R3): two independent probes, nothing evaluated, nothing needs box 2."""
    import torch
    t_start = time.time()
    own = {}
    for k in ("CTRL", "REP"):
        td = env.runs[k]["train_dir"]
        files = sorted(f.replace("\\", "/") for f in glob.glob(td + "/**/*", recursive=True) if os.path.isfile(f))
        own[k] = {"train_dir": td, "files": [os.path.basename(f) for f in files],
                  "weight_files": [{"path": f, "bytes": os.path.getsize(f), "sha256": sha256_file(f)} for f in files if f.lower().endswith(WEIGHT_EXT)]}
    own_paths = {w["path"] for k in own for w in own[k]["weight_files"]}
    # probe 1 (by LOCATION): every run directory of the two families on E: and the F: asides
    fam = []
    for pat in (f"{ART}/v8f-dagoacc*", f"{ART}/v8f-dag-orc-oacc*", "F:/TanitAD-scratch/aside_from_E/v8f-dag-orc-oacc*"):
        fam += [d.replace("\\", "/") for d in sorted(glob.glob(pat)) if os.path.isdir(d)]
    p1 = {}
    for d in fam:
        ws_ = [f for f in _walk_depth(d, 4) if f.lower().endswith(WEIGHT_EXT)]
        p1[d] = [{"path": f, "bytes": os.path.getsize(f)} for f in ws_]
    # probe 2 (by NAME): any weight file anywhere (depth <= 5) whose path names either run, skipping source-tree copies
    names = ("m2b_s0_dagoacc/", "m2b_s0_dagoaccorcrep/", "dagoaccorcrep", "m2b_s0_dagoacc.", "m2b_s0_dagoacc_")
    p2 = []
    for root in (ART, "F:/TanitAD-scratch"):
        for f in _walk_depth(root, 5, skip=("dagt1_tree_", "dagorc_combined", "/.git/")):
            fl = f.lower()
            if fl.endswith(WEIGHT_EXT) and any(n in fl for n in names):
                p2.append({"path": f, "bytes": os.path.getsize(f)})
    # the trainer's save rule (second independent evidence: the code, not the disk)
    src = open(f"{TREE}/stack/scripts/v8f_mini_train.py", encoding="utf-8").read().split("\n")
    rule = [(i + 1, ln.strip()[:170]) for i, ln in enumerate(src) if any(x in ln for x in ('"ckpt_last.pt"', "--ckpt-every", "torch.save(blob", "os.replace(ck_path", "step % a.ckpt_every"))]
    inner = {}
    for k in ("CTRL", "REP"):
        ck = torch.load(env.runs[k]["ckpt"], map_location="cpu", weights_only=False)
        a = json.load(open(env.runs[k]["train_dir"] + "/argv.json", encoding="utf-8"))
        ic = ck.get("init_ckpt")
        inner[k] = {"step": ck.get("step"), "seed": ck.get("seed"), "has_optimizer_state": "opt" in ck, "has_ema": any("ema" in str(x).lower() for x in ck), "init_ckpt_record": ic if not hasattr(ic, "keys") else {kk: ic[kk] for kk in ic if kk != "tensors"},
                    "argv_init_ckpt": a["argv"][a["argv"].index("--init-ckpt") + 1] if "--init-ckpt" in a["argv"] else None, "argv_has_ckpt_every": "--ckpt-every" in a["argv"], "steps_flag": a["argv"][a["argv"].index("--steps") + 1]}
    # the init checkpoint candidate: any ckpt_last.pt whose sha equals the recorded init sha (second name probe by CONTENT)
    init_sha = None
    for k in inner:
        ic = inner[k]["init_ckpt_record"]
        if isinstance(ic, dict):
            for kk, vv in ic.items():
                if isinstance(vv, str) and len(vv) == 64:
                    init_sha = vv
    cand = []
    for f in _walk_depth(f"{ART}/sfresh_inputs/inputs/M2b_s0", 2):
        if f.endswith("ckpt_last.pt"):
            cand.append({"path": f, "bytes": os.path.getsize(f), "sha256": sha256_file(f)})
    others = {}
    for d, v in p1.items():
        v2 = [w for w in v if w["path"] not in own_paths]
        if v2:
            others[d] = v2
    p2_extra = [w for w in p2 if w["path"] not in own_paths]
    big = [w for w in p2_extra if w["bytes"] >= (1 << 20)]
    stubs = [w for w in p2_extra if w["bytes"] < (1 << 20)]
    same_run = [w for w in big if ("/m2b_s0_dagoacc/" in w["path"].lower() or "/m2b_s0_dagoaccorcrep/" in w["path"].lower())]
    p2_extra = {"real_size_weight_files_naming_a_run_of_the_family": big, "n_tiny_placeholder_stubs_under_1MB": len(stubs),
                "stub_size_bytes_distribution": {str(b): int(sum(1 for w in stubs if w["bytes"] == b)) for b in sorted({w["bytes"] for w in stubs})},
                "stub_locations": sorted({"/".join(w["path"].split("/")[:4]) for w in stubs}), "real_size_files_that_are_THESE_two_runs_other_than_ckpt_last": same_run}
    present = bool(same_run)
    res = {"task": "H-CRAWL-03", "evidence": "MEASURED (file-system probes and a read of the trainer source on the dev box, this run). Feasibility only: no evaluation, nothing needs box 2.",
           "these_two_runs": own, "probe1_by_location_weight_files_in_run_dirs_of_both_families_not_these_two_runs": others,
           "probe2_by_name_weight_files_naming_either_run_other_than_their_ckpt_last": p2_extra,
           "trainer_save_rule_lines": rule, "inside_the_two_checkpoints": inner, "recorded_init_sha256": init_sha, "init_checkpoint_candidates_found_locally": cand,
           "init_matches_recorded_sha": [c["sha256"] == init_sha for c in cand] if init_sha else None,
           "finding": ("PRESENT: see probe2" if present else "ABSENT: no intermediate checkpoint of DAG-OACC s0 or REP s0 exists on this box (probe 1 by location, probe 2 by name, the trainer's save rule overwrites one file). Present: each run's final step-3000 file and the shared fine-tune init (matched by sha256)"),
           "seconds": time.time() - t_start}
    json.dump(res, open(out_path, "w", encoding="utf-8", newline="\n"), indent=1, default=_jd)
    print(f"H-CRAWL-03 done -> {out_path}: {res['finding']}", flush=True)
    return res


def _jd(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=["01", "02", "03", "merge", "all"])
    a = ap.parse_args(argv)
    os.makedirs(f"{HERE}/raw", exist_ok=True)
    env = Env()
    if a.task in ("01", "all"):
        task_01(env, f"{HERE}/raw/part_01.json")
    if a.task in ("02", "all"):
        task_02(env, f"{HERE}/raw/part_02.json")
    if a.task in ("03", "all"):
        task_03(env, f"{HERE}/raw/part_03.json")
    if a.task in ("merge", "all"):
        merged = {"what": "H-CRAWL-01/-02/-03 zero-training diagnostics (PREREG_FROZEN.md, READINGS.md)", "frozen": open(f"{HERE}/FROZEN_SHA.txt", encoding="utf-8").read(),
                  "readings": open(f"{HERE}/READINGS_SHA.txt", encoding="utf-8").read(), "crawl_diag_py_sha256": sha256_file(f"{HERE}/crawl_diag.py")}
        for t in ("01", "02", "03"):
            p = f"{HERE}/raw/part_{t}.json"
            if os.path.exists(p):
                merged[f"H-CRAWL-{t}"] = json.load(open(p, encoding="utf-8"))
        json.dump(merged, open(f"{HERE}/CRAWL_DIAG.json", "w", encoding="utf-8", newline="\n"), indent=1, default=_jd)
        print(f"merged -> {HERE}/CRAWL_DIAG.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
