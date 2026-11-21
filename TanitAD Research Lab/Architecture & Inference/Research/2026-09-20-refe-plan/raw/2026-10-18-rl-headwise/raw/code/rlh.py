#!/usr/bin/env python3
"""RL-1 helper: RL-val loaders (zero-training, CPU numpy) on the banked LIVE score dumps; fast paired log-cluster bootstrap.

Channels of a scorer dump [N, 64, 7]: 0 NC, 1 DAC, 2 EP, 3 TTC, 4 C, 5 DDC (not in the rule), 6 lane (teacher-lane output).
Selection rule navsim_v1_lane: NC * DAC * (5 EP + 5 TTC + 2 C) / 12 * lane  (rl_val_eval.select), ties -> lowest index (np.argmax).
Estimator: paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (== proxy_eval.boot; identity-checked in ctl_boot).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.join(HERE, "pk")
sys.path.insert(0, os.path.join(PK, "eval"))
sys.path.insert(0, os.path.join(PK, "refe"))
sys.path.insert(0, os.path.join(PK, "raw", "2026-10-14-rls1b-attribution"))
import attrib_rls1b as AT  # noqa: E402
import proxy_eval as PE  # noqa: E402

DL = "C:/Users/Admin/rls1b_dl"
X = DL + "/data/x"
SEED, NBOOT = 20260927, 10000
SAFETY = (0, 1, 3)            # NC, DAC, TTC
HNAME = {0: "NC", 1: "DAC", 2: "EP", 3: "TTC", 4: "C", 5: "DDC", 6: "lane"}


def load_labels():
    kv, tgv, dv, ckv, st = AT.load_side(X + "/pdm_val/pdm_val", X + "/terms_val/terms_val")
    s = AT.comps(tgv)
    logs = np.asarray([k[0] for k in kv]).astype(str)
    return kv, ckv, s, logs, st


def load_dump(path, kv, ckv):
    return AT.load_scores_aligned(path, kv, ckv)


def sel(p):
    """p: probabilities [..., 7] -> rule score [...]"""
    return p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0 * p[..., 6]


def sigm(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))


def pick(logits):
    return sel(sigm(logits)).argmax(-1)


def compose(rl, base, rl_heads):
    """logit composite: channels in rl_heads from `rl`, every other channel from `base`."""
    x = base.copy()
    for h in rl_heads:
        x[..., h] = rl[..., h]
    return x


class Boot:
    """paired log-cluster bootstrap, groups sorted by log name, rng.integers(0, G, G) per resample, seed fixed (== PE.boot)."""

    def __init__(self, logs, n=NBOOT, seed=SEED):
        u = sorted(set(logs.tolist()))
        self.G = len(u)
        gid = np.searchsorted(np.asarray(u), logs)
        self.gid = gid
        self.cnt = np.bincount(gid, minlength=self.G).astype(np.float64)
        rng = np.random.default_rng(seed)
        self.idx = np.stack([rng.integers(0, self.G, self.G) for _ in range(n)])     # [n, G], the very stream PE.boot draws
        self.den = self.cnt[self.idx].sum(1)

    def ci(self, d):
        """d [N] per-set values -> (mean, lo, hi) of the resampled mean."""
        s = np.bincount(self.gid, weights=np.asarray(d, np.float64), minlength=self.G)
        m = s[self.idx].sum(1) / self.den
        return float(np.mean(d)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    def ci_ratio(self, num, den):
        """ratio of sums (a ratio estimator) with the same resamples"""
        a = np.bincount(self.gid, weights=np.asarray(num, np.float64), minlength=self.G)
        b = np.bincount(self.gid, weights=np.asarray(den, np.float64), minlength=self.G)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = a[self.idx].sum(1) / b[self.idx].sum(1)          # a resample with an empty denominator is dropped (nan), as for any subset ratio
        return float(a.sum() / b.sum()), float(np.nanpercentile(m, 2.5)), float(np.nanpercentile(m, 97.5))


def ctl_boot(kv, logs, d):
    """identity control: Boot.ci == proxy_eval.boot on the same per-set vector"""
    dd = {AT.RV.kstr(k): float(d[i]) for i, k in enumerate(kv)}
    tl = {AT.RV.kstr(k): k[0] for k in kv}
    ref = PE.boot(dd, tl)
    mine = Boot(logs).ci(d)
    return ref, mine, max(abs(a - b) for a, b in zip(ref, mine))


# ---------------------------------------------------------------------------------------------------------------- RL-1 additions
def _ratio_diff(self, nB, dB, nA, dA):
    a1 = np.bincount(self.gid, weights=nB, minlength=self.G); b1 = np.bincount(self.gid, weights=dB, minlength=self.G)
    a0 = np.bincount(self.gid, weights=nA, minlength=self.G); b0 = np.bincount(self.gid, weights=dA, minlength=self.G)
    m = a1[self.idx].sum(1) / b1[self.idx].sum(1) - a0[self.idx].sum(1) / b0[self.idx].sum(1)
    mu = a1.sum() / b1.sum() - a0.sum() / b0.sum()
    return float(mu), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


Boot.ci_ratio_diff = _ratio_diff


def auc_parts(score, pos):
    """within-scene AUC numerator / denominator per set. score [N, M], pos [N, M] bool (label 1 = pass). ties 0.5."""
    d = score[:, :, None] - score[:, None, :]
    pr = pos[:, :, None] & ~pos[:, None, :]
    num = (((d > 0) + 0.5 * (d == 0)) * pr).sum((1, 2)).astype(np.float64)
    den = pr.sum((1, 2)).astype(np.float64)
    return num, den


NCH = 7
_SUBSETS = np.arange(1 << NCH)
_BITS = ((_SUBSETS[:, None] >> np.arange(NCH)[None, :]) & 1).astype(bool)        # [128, 7]
_SIZE = _BITS.sum(1)
import math as _m
_W = np.array([_m.factorial(k) * _m.factorial(NCH - k - 1) / _m.factorial(NCH) for k in range(NCH)])


def shapley_pairs(LB, LR, iB, iR, idx):
    """exact Shapley decomposition of the pick-pair margin change m_B - m_R over the 7 channels.
    LB, LR [N, 64, 7] logits (B = RL arm, R = ref); iB, iR [N] picks; idx = set indices.  Returns phi [n, 7], mB [n], mR [n], mAll_check [n], mNone_check [n]."""
    n = len(idx)
    xb_B, xr_B = sigm(LB[idx, iB[idx]]), sigm(LR[idx, iB[idx]])           # [n, 7] probabilities at B's pick: from B / from ref
    xb_R, xr_R = sigm(LB[idx, iR[idx]]), sigm(LR[idx, iR[idx]])           # at ref's pick
    # m(coalition) for all 128 coalitions: [n, 128]
    cb = _BITS[None, :, :]                                                 # [1, 128, 7]
    pB = np.where(cb, xb_B[:, None, :], xr_B[:, None, :])
    pR = np.where(cb, xb_R[:, None, :], xr_R[:, None, :])
    m = sel(pB) - sel(pR)                                                  # [n, 128]
    phi = np.zeros((n, NCH))
    for h in range(NCH):
        has = _BITS[:, h]
        for c in np.flatnonzero(~has):
            c2 = c | (1 << h)
            phi[:, h] += _W[_SIZE[c]] * (m[:, c2] - m[:, c])
    return phi, m[:, -1], m[:, 0]


def margin_direct(L, iB, iR, idx):
    return sel(sigm(L[idx, iB[idx]])) - sel(sigm(L[idx, iR[idx]]))
