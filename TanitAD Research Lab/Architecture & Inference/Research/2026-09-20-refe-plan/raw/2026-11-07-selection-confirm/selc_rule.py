#!/usr/bin/env python3
"""The FROZEN re-pick rule gBG(0.20, 0.05) of PREREG_SEL.md section 1 (sha256 e00fa28a...), as pure functions of arrays.

This is a COPY of `refe_selection_probe_20261107/p2_score.py::rule_vec` and `::rule_loop`, with the module-level globals turned
into arguments and nothing else changed. `selc_gates.py` G-RULE asserts that this copy and p2_score's code give the same pick on
1,058/1,058 C3 tokens (it re-executes p2_score's functions from source), so the copy cannot drift silently.

Inputs per token i, over the 64 proposals k:
  L   [N,64,7] raw u400 scorer logits (order NC, DAC, EP, TTC, C, DDC, lane)
  v4  [N,64]   one-step 0.2 s chord speed at t = 4 s of the RAW proposal (xy[19]-xy[18]) / 0.2
  v0  [N]      measured ego speed now (ego_real[:, 6]);  ax0 [N] measured longitudinal acceleration now (ego_real[:, 2])
"""
import numpy as np

DELTA, EPS = 0.20, 0.05
V1_W = (5.0, 5.0, 2.0)
DT = 0.2


def probs(L):
    return 1.0 / (1.0 + np.exp(-np.asarray(L, np.float64)))


def agg64(L):
    """navsim_v1_lane aggregate in float64 (sel_common.agg_np(L, np.float64))."""
    z = np.asarray(L).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-z))
    w = np.asarray(V1_W, np.float64)
    return p[..., 0] * p[..., 1] * (p[..., 2] * w[0] + p[..., 3] * w[1] + p[..., 4] * w[2]) / w.sum() * p[..., 6]


def banked_pick(L):
    """The banked pick: torch float32 argmax of the aggregate (planner.py:551-556, :583) -- sel_common.pick_from_logits(L, 'torch')."""
    import torch
    p = torch.as_tensor(np.asarray(L, np.float32)).sigmoid()
    w1 = torch.tensor(V1_W, dtype=p.dtype)
    a = p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()
    a = a * p[..., 6]
    return np.argmax(a.numpy(), axis=-1)


def step_speed4(traj):
    """sel_common.step_speed(P, 19): |xy[19] - xy[18]| / 0.2 on the raw 20 x 0.2 s proposals."""
    xy = np.asarray(traj, np.float64)[..., :2]
    prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], axis=-2)
    v = np.linalg.norm(xy - prev, axis=-1) / DT
    return v[..., 19]


def speed_ratio(v4, v0):
    return np.minimum(1.0, (np.asarray(v4, np.float64) + 1.0) / (np.asarray(v0, np.float64)[:, None] + 1.0))


def rule_vec(L, v4, v0, ax0, delta=DELTA, eps=EPS, flip_guard=False, gated=True):
    L = np.asarray(L)
    N = L.shape[0]
    r = np.arange(N)
    s = speed_ratio(v4, v0)
    P = probs(L)
    agg = agg64(L)
    GATE = (np.asarray(v0) > 5.0) & (np.abs(np.asarray(ax0)) < 0.3)
    base = banked_pick(L)
    if delta == 0:
        return base
    band = agg >= (1.0 - delta) * agg[r, base][:, None]
    pb = P[r, base][:, None, :]
    if flip_guard:
        guard = np.all(P[..., [0, 1, 3]] >= pb[..., [0, 1, 3]] + eps, axis=-1)
    else:
        guard = np.all(P[..., [0, 1, 3]] >= pb[..., [0, 1, 3]] - eps, axis=-1)
    adm = band & guard
    pk = np.where(adm.any(-1), np.argmax(np.where(adm, s + 1e-6 * agg, -1.0), -1), base)
    return np.where(GATE, pk, base) if gated else pk


def rule_loop(L, v4, v0, ax0, i, delta=DELTA, eps=EPS):
    """PREREG_SEL section 1, implemented literally for ONE token."""
    agg = agg64(L[i:i + 1])[0]
    P = probs(L[i])
    s = speed_ratio(v4[i:i + 1], v0[i:i + 1])[0]
    b = int(np.argmax(agg))
    if not (v0[i] > 5.0 and abs(ax0[i]) < 0.3):
        return b
    best, bestkey = b, -np.inf
    for k in range(agg.shape[0]):
        if not agg[k] >= (1 - delta) * agg[b]:
            continue
        if any(P[k, j] < P[b, j] - eps for j in (0, 1, 3)):
            continue
        key = s[k] + 1e-6 * agg[k]
        if key > bestkey:
            best, bestkey = k, key
    return best
