#!/usr/bin/env python3
"""EP-1 candidate rules, exactly as fixed in PREREG_EP1.md section 1 (sha256 of the prereg in PREREG_EP1.sha256). Pure functions of the six head
outputs and the proposals' own 8-pose plans.  No labels."""
import numpy as np
from ep_common import NC, DAC, EP, TTC, C, sigmoid, score_terms, geometry

DELTA_GRID = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
TAU_GRID = [1.0, 0.99, 0.98, 0.97, 0.95, 0.93, 0.90, 0.85, 0.80, 0.70]


def heads(t):
    p = sigmoid(t["logits"].astype(np.float64))
    g, w = score_terms(p)
    L, _ = geometry(t["proposals"])
    return p, g, w, L


def pick_base(p, g, w):
    return np.argmax(g * w, axis=1)


def pick_c1(p, g, w):
    s1 = (5 * p[..., EP] + g * (5 * p[..., TTC] + 2 * p[..., C])) / 12.0
    return np.argmax(s1, axis=1)


def pick_c1_identity(p, g, w):
    """(5 pEP g + g (5 pTTC + 2 pC)) / 12 == S0 : the identity control for C1."""
    s = (5 * p[..., EP] * g + g * (5 * p[..., TTC] + 2 * p[..., C])) / 12.0
    return np.argmax(s, axis=1)


def pick_c2(p, g, w, L, delta):
    base = pick_base(p, g, w)
    if delta == 0.0:
        return base
    ls = np.log(np.maximum(g * w, 1e-300))
    band = ls >= ls.max(1, keepdims=True) - delta
    return np.argmax(np.where(band, L, -np.inf), axis=1)


def pick_c3(p, g, w, L, tau):
    base = pick_base(p, g, w)
    q = g * p[..., TTC]
    ok = q >= tau
    has = ok.any(1)
    alt = np.argmax(np.where(ok, L, -np.inf), axis=1)
    return np.where(has, alt, base)
