#!/usr/bin/env python3
"""Shared loaders + the offline REFe pick rule for the YSEL selection probe (CPU only, zero GPU).

The pick rule reproduced here is REFe's `navsim_v1_lane` aggregate, read from source:
  `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/planner.py`
    :551  p = score.sigmoid()
    :552-556  rule navsim_v1_lane: p0 * p1 * (5 p2 + 5 p3 + 2 p4) / 12 * p6     (V1_W = (5,5,2) at :520)
    :583  _pick (select="best", the default at :322): int(self.aggregate(score)[0].argmax())
    :475  infer() -> k = self._pick(score)   (the SAME function the navtest seam calls, eval/refe_navtest_seam.py:250)
Component order (planner.py:547-549 + C3 OPERATIONALISATION): 0 NC, 1 DAC, 2 EP, 3 TTC, 4 C, 5 DDC, 6 lane.
"""
import json
import os

import numpy as np

ROOT = "E:/Projects/TanitAD"
C3 = ROOT + "/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-11-c3-selection-headroom"
EGO = "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106"
OUT = "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107"
DT = 0.2
SLOW_FRAC = 0.15
SIM_TOKEN = "41d7b533797c5209"
V1_W = (5.0, 5.0, 2.0)
HEADS = ("NC", "DAC", "EP", "TTC", "C", "DDC", "lane")


def agg_np(logits, dtype=np.float32, lane=True):
    """navsim_v1_lane aggregate, numpy, in `dtype` (planner.py:551-556)."""
    z = np.asarray(logits).astype(dtype)
    p = (1.0 / (1.0 + np.exp(-z))).astype(dtype)
    w = np.asarray(V1_W, dtype)
    a = p[..., 0] * p[..., 1] * (p[..., 2] * w[0] + p[..., 3] * w[1] + p[..., 4] * w[2]) / w.sum()
    if lane:
        a = a * p[..., 6]
    return a


def agg_torch(logits, lane=True):
    """navsim_v1_lane aggregate via torch float32 on CPU -- the same ops as planner.py:551-556."""
    import torch
    p = torch.as_tensor(np.asarray(logits, np.float32)).sigmoid()
    w1 = torch.tensor(V1_W, dtype=p.dtype)
    a = p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()
    if lane:
        a = a * p[..., 6]
    return a.numpy()


def pick_from_logits(logits, impl="torch", lane=True):
    a = agg_torch(logits, lane) if impl == "torch" else agg_np(logits, np.float32 if impl == "np32" else np.float64, lane)
    return np.argmax(a, axis=-1)      # first index of the max, as torch.argmax


def step_speed(P, idx=19):
    """The ego probe's one-step chord speed (analyze_ego_probe.py:22-26) at step idx (19 -> t=4 s), 0.2 s grid."""
    xy = np.asarray(P, np.float64)[..., :2]
    prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], axis=-2)
    v = np.linalg.norm(xy - prev, axis=-1) / DT
    return v[..., idx]


def gt_speed4(pose20, t20):
    i = 19
    dq = np.linalg.norm(pose20[:, i, :2] - pose20[:, i - 1, :2], axis=-1)
    return dq / (t20[:, i] - t20[:, i - 1])


def load():
    """Everything aligned to the C3 table's token order. Returns a dict."""
    T = np.load(C3 + "/table.npz", allow_pickle=False)
    Z = np.load(EGO + "/full/results.npz", allow_pickle=False)
    tok = np.array([str(x) for x in T["token"]])
    zt = {str(t): i for i, t in enumerate(Z["token"])}
    zi = np.array([zt[t] for t in tok])
    assert (np.array([str(x) for x in Z["token"]])[zi] == tok).all()
    allj = json.load(open(EGO + "/tokens_all.json"))
    cj = json.load(open(EGO + "/tokens_cruise.json"))
    log = np.array([allj["token_log"][t] for t in tok])
    assert (log == np.array([str(x) for x in Z["log"]])[zi]).all()
    cruise_set = set(cj["tokens"])
    cruise = np.array([t in cruise_set for t in tok])
    arms = [str(x) for x in Z["arms"]]
    a0 = arms.index("A0")
    ego = Z["ego_real"][zi]
    v0 = ego[:, 6]
    traj = Z["traj_all_A0"][zi].astype(np.float64)        # [N, 64, 20, 3] raw REFe proposals, 0.2 s grid
    v4_all = step_speed(traj)                              # [N, 64]
    D = dict(token=tok, log=log, stratum=np.array([allj["stratum"][t] for t in tok]), cruise=cruise,
             pdms=T["pdms"].astype(np.float64), sub=T["sub"].astype(np.float64), sub_names=[str(x) for x in T["sub_names"]],
             valid=T["valid"].astype(bool), logits=T["logits"], proposals=T["proposals"].astype(np.float64),
             pick=T["pick"].astype(np.int64), pdms_at_pick_banked=T["pdms_at_pick_banked"],
             logits_A0=Z["logits_all_A0"][zi], pick_A0_probe=Z["pick"][zi, a0].astype(np.int64),
             traj=traj, v0=v0, ax0=ego[:, 2], v4_all=v4_all, gt_valid=Z["gt_valid"][zi].astype(bool),
             v4_gt=gt_speed4(Z["gt_pose20"][zi], Z["gt_t20"][zi]), speed_all_A0=Z["speed_all"][zi, a0].astype(np.float64))
    return D


def log_boot_idx(log, B=10000, seed=20260927):
    """Paired log-cluster bootstrap: one shared [B, n_logs] resample of log ids (the programme's draws)."""
    ul, li = np.unique(log, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(ul), size=(B, len(ul)), replace=True)
    return ul, li, draws


def boot_mean(vals, mask, li, n_logs, draws):
    """Ratio estimator sum/count over resampled logs, restricted to mask. Returns point, lo, hi (95 % percentile)."""
    vals = np.asarray(vals, np.float64)
    m = np.asarray(mask, bool) & np.isfinite(vals)
    s = np.bincount(li[m], weights=vals[m], minlength=n_logs)
    c = np.bincount(li[m], minlength=n_logs).astype(np.float64)
    S = s[draws].sum(1)
    C = c[draws].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = S / C
    r = r[np.isfinite(r)]
    pt = s.sum() / max(c.sum(), 1e-12)
    return float(pt), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))
