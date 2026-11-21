#!/usr/bin/env python3
"""Candidate re-pick rules (selection only; REFe u400's scorer and proposals are FIXED). Each returns pick indices [N].

Notation: agg = navsim_v1_lane aggregate (planner.py:552-556) over sigmoid of the 7 logits.
v4 = one-step 0.2 s chord speed at t = 4 s of each raw proposal; v0 = measured ego speed (ego_real[:, 6]).
s = min(1, (v4 + 1) / (v0 + 1))  -- the speed-consistency ratio, 1 m/s softening so a standstill start is not divided by 0.

  S(lam)   agg * s**lam                                    speed-consistency factor (no reward above v0)
  L(lam)   NC*DAC*avg * p_lane**lam                        lane-head temper (lam = 1 is the banked rule, 0 is navsim_v1)
  E(lam)   NC*DAC*(5(1+lam)EP + 5TTC + 2C)/(12+5lam)*lane  progress-weight up-weighting
  V(tau)   argmax agg over proposals with s >= tau, fallback to the banked pick when none qualifies
lam = 0 (or tau = 0) reproduces the banked pick exactly (asserted by p2 gate G0b).
"""
import numpy as np

import sel_common as sc


def probs(logits):
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64)))


def speed_ratio(v4_all, v0):
    return np.minimum(1.0, (np.asarray(v4_all, np.float64) + 1.0) / (np.asarray(v0, np.float64)[:, None] + 1.0))


def score_S(logits, s, lam):
    return sc.agg_np(logits, np.float64) * s ** lam


def score_L(logits, lam):
    p = probs(logits)
    return p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0 * p[..., 6] ** lam


def score_E(logits, lam):
    p = probs(logits)
    return p[..., 0] * p[..., 1] * (5 * (1 + lam) * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / (12.0 + 5 * lam) * p[..., 6]


def pick_S(logits, s, lam):
    if lam == 0:
        return sc.pick_from_logits(logits, "torch")
    return np.argmax(score_S(logits, s, lam), -1)


def pick_L(logits, lam):
    if lam == 1:
        return sc.pick_from_logits(logits, "torch")
    return np.argmax(score_L(logits, lam), -1)


def pick_E(logits, lam):
    if lam == 0:
        return sc.pick_from_logits(logits, "torch")
    return np.argmax(score_E(logits, lam), -1)


def pick_V(logits, s, tau):
    base = sc.pick_from_logits(logits, "torch")
    if tau == 0:
        return base
    a = sc.agg_np(logits, np.float64)
    ok = s >= tau
    alt = np.argmax(np.where(ok, a, -1.0), -1)
    return np.where(ok.any(-1), alt, base)


def pick_B(logits, s, delta):
    """B(delta): among proposals whose aggregate is within a factor (1 - delta) of the best, take the one with the
    highest speed-consistency s; ties in s (e.g. every proposal at s = 1) are broken by the higher aggregate.
    delta = 0 reproduces the banked pick (only the max-aggregate proposal is admitted)."""
    base = sc.pick_from_logits(logits, "torch")
    if delta == 0:
        return base
    a = sc.agg_np(logits, np.float64)
    band = a >= (1.0 - delta) * a.max(-1, keepdims=True)
    key = np.where(band, s + 1e-6 * a, -1.0)      # lexicographic (s, agg): agg <= 1 so 1e-6*agg only breaks ties in s
    return np.argmax(key, -1)


def pick_BG(logits, s, delta, eps):
    """BG(delta, eps): B(delta) with a SAFETY GUARD from the scorer's own heads -- a proposal is admitted to the band only
    if its predicted NC, DAC and TTC probabilities are each >= the banked pick's minus eps. eps = inf is plain B(delta)."""
    base = sc.pick_from_logits(logits, "torch")
    if delta == 0:
        return base
    p = probs(logits)
    a = sc.agg_np(logits, np.float64)
    band = a >= (1.0 - delta) * a.max(-1, keepdims=True)
    pb = np.take_along_axis(p, base[:, None, None], axis=1)          # [N, 1, 7]
    guard = np.all(p[..., [0, 1, 3]] >= pb[..., [0, 1, 3]] - eps, axis=-1)
    adm = band & guard
    key = np.where(adm, s + 1e-6 * a, -1.0)
    out = np.argmax(key, -1)
    return np.where(adm.any(-1), out, base)
