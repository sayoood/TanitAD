"""EP-GEOM (L3), PREREG_EPGEOM.md section 2: a geometric progress (EP) surrogate, applied only inside the confidently-safe set.

Inputs are C1's own per-proposal head logits [n,64,6] (order NC, DAC, EP, TTC, C, DDC), the 8-pose executed plans props8 [n,64,8,3]
(x fwd, y left, heading; NAVSIM 0.5 s grid) and C1's own pick [n]. Nothing else: no map, agents, future or metric cache (vision-only rule:
model outputs + nav command only; the geometry comes from the proposal itself). Pure numpy, CPU.
"""
from __future__ import annotations

import numpy as np

SAFE_ROWS = (0, 1, 3, 5)                 # NC, DAC, TTC, DDC
SURROGATES = ("arc", "x8", "pep")        # (a) arc length incl. origin segment, (b) final forward displacement, (c) C1's own sig(EP)
TAU_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)
NBOOT, SEED = 10000, 20260927            # as ENS-SCORE / C1r read (p5c_read.Boot)


def sig(x):
    x = np.asarray(x, dtype=np.float32)
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


def safe_q(logits, rows=SAFE_ROWS):
    """q_k = sig(NC) * sig(DAC) * sig(TTC) * sig(DDC), float32, [n,64], from logits [n,64,6]."""
    p = sig(logits)
    q = p[..., rows[0]]
    for r in rows[1:]:
        q = (q * p[..., r]).astype(np.float32)
    return q


def safe_q_from_probs(probs):
    """L3-SH hook: probs [n,64,4] already in (NC, DAC, TTC, DDC) order (the L2 SAFE-HEAD probabilities)."""
    probs = np.asarray(probs, dtype=np.float32)
    q = probs[..., 0]
    for r in range(1, probs.shape[-1]):
        q = (q * probs[..., r]).astype(np.float32)
    return q


def aggregate_v1(logits):
    """navsim_v1 rule NC x DAC x (5 EP + 5 TTC + 2 C) / 12 on sigmoid probabilities (refe/planner.py:517-575), float32 numpy."""
    p = sig(logits)
    return (p[..., 0] * p[..., 1] * (np.float32(5) * p[..., 2] + np.float32(5) * p[..., 3] + np.float32(2) * p[..., 4]) / np.float32(12)).astype(np.float32)


def surrogate(name, logits, props8):
    """EP surrogate g [n,64], float32."""
    if name == "arc":
        xy = np.asarray(props8, dtype=np.float32)[..., :2]
        pts = np.concatenate([np.zeros(xy.shape[:-2] + (1, 2), np.float32), xy], axis=-2)      # the segment origin -> pose 1 is included
        return np.linalg.norm(np.diff(pts, axis=-2), axis=-1).sum(-1).astype(np.float32)
    if name == "x8":
        return np.asarray(props8, dtype=np.float32)[..., 7, 0].copy()
    if name == "pep":
        return sig(logits)[..., 2]
    raise ValueError(f"unknown surrogate {name!r}")


def select(logits, props8, pick, surr, tau, q=None, g=None, roll=0):
    """The rule. Per token: S_tau = {k : q_k >= tau}; empty -> C1's own pick; else argmax_{k in S_tau} g_k, ties (exact float32 equality of g) ->
    C1's pick if it is among the tied, else the argmax of C1's aggregate over the tied. Returns int64 [n].
    `q` / `g` override the defaults (L3-SH passes q from the L2 head). `roll` misaligns g by that many proposal indices (CTRL-RED only)."""
    logits = np.asarray(logits)
    pick = np.asarray(pick, dtype=np.int64)
    n = logits.shape[0]
    if q is None:
        q = safe_q(logits)
    if g is None:
        g = surrogate(surr, logits, props8)
    if roll:
        g = np.roll(g, roll, axis=1)
    mask = q >= np.float32(tau)
    empty = ~mask.any(1)
    gm = np.where(mask, g, -np.inf)
    gmax = gm.max(1, keepdims=True)
    tied = mask & (gm == gmax)
    ar = np.arange(n)
    agg = aggregate_v1(logits)
    tie_best = np.where(tied, agg, -np.inf).argmax(1)
    out = np.where(tied[ar, pick], pick, tie_best)
    return np.where(empty, pick, out).astype(np.int64)


def pdms_at(tr_pdms, picks):
    return np.asarray(tr_pdms)[np.arange(len(picks)), picks]


class Boot:
    """ONE draw matrix, shared: verbatim the construction of eval/p5c/p5c_read.py::Boot (paired log-cluster bootstrap)."""

    def __init__(self, logs, n_boot: int = NBOOT, seed: int = SEED):
        ul = sorted(set(logs))
        pos = {l: i for i, l in enumerate(ul)}
        self.li = np.array([pos[l] for l in logs])
        self.L = len(ul)
        self.n = np.bincount(self.li, minlength=self.L).astype(float)
        self.D = np.random.default_rng(seed).integers(0, self.L, (n_boot, self.L))
        self.nD = self.n[self.D].sum(1)
        self.n_boot, self.seed = n_boot, seed

    def draws(self, v) -> np.ndarray:
        s = np.bincount(self.li, weights=np.asarray(v, float), minlength=self.L)
        return s[self.D].sum(1) / self.nD

    def ci(self, v) -> dict:
        m = self.draws(v)
        return {"point": float(np.mean(v)), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def fit(logits, props8, pick, tr_pdms, surrogates=SURROGATES, taus=TAU_GRID):
    """Grid fit on the given (TRAIN) tokens: maximise the mean tracked tr_pdms of the pick. Candidate order = surrogate order, then ascending tau.
    Exact ties in the mean -> earlier surrogate, then HIGHER tau (more conservative). Returns (best dict, table list)."""
    table = []
    q = safe_q(logits)
    for s in surrogates:
        g = surrogate(s, logits, props8)
        for t in taus:
            pk = select(logits, props8, pick, s, t, q=q, g=g)
            table.append({"surrogate": s, "tau": float(t), "mean_pdms": float(pdms_at(tr_pdms, pk).mean()), "switch_rate": float((pk != pick).mean())})
    best = None
    for r in table:
        if best is None or r["mean_pdms"] > best["mean_pdms"] or (r["mean_pdms"] == best["mean_pdms"] and r["surrogate"] == best["surrogate"] and r["tau"] > best["tau"]):
            best = r
    return best, table
