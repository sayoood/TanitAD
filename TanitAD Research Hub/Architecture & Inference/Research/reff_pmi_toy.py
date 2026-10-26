#!/usr/bin/env python3
"""REF-F V0 toy: does CLM-style InfoNCE selection under-choose the dominant action?

A dual encoder (state table, action table, dot-product score) is trained three ways
on samples from a synthetic driving-like world whose action prior is dominated by
"cruise" (the parity corpus is ~74 % straight cruise, `train_flagship_v4.py:267`):

  nce      bidirectional in-batch InfoNCE with CLM's false-negative mask
           (pairs sharing the positive's action are masked out)
  nce+pri  the same critic, scored with + log p_hat(a) at selection time
  voc      softmax cross-entropy over the whole action vocabulary (CLM "Choice")

Theory (van den Oord et al. 2018; Poole et al. 2019): InfoNCE's optimal critic is
log p(a|s) - log p(a) + c(s), i.e. pointwise mutual information, so its argmax is
not the most likely action.  This script measures, on held-out draws, how often each
rule picks the action a fresh sample from p(a|s) would take ("hit rate"), and how its
selected-action frequencies compare with the true marginal.  numpy only; CPU; seconds.
"""
import json
import sys

import numpy as np

S, A, D = 60, 16, 24          # states, actions, embedding width
CRUISE = 0
STEPS, B, LR = 4000, 256, 0.5


def make_world(rng):
    p_s = np.full(S, 1.0 / S)
    p_a_s = np.zeros((S, A))
    for s in range(S):
        c = rng.uniform(0.15, 0.85)            # cruise probability in this state
        alts = rng.choice(np.arange(1, A), size=2, replace=False)
        w = rng.dirichlet([2.0, 1.0])
        p_a_s[s, CRUISE] = c
        p_a_s[s, alts] = (1 - c) * w
    return p_s, p_a_s


def sample(rng, p_s, p_a_s, n):
    s = rng.choice(S, size=n, p=p_s)
    u = rng.random(n)[:, None]
    a = (u > np.cumsum(p_a_s[s], axis=1)).sum(axis=1)
    return s, np.minimum(a, A - 1)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


SCALE = 1.0 / 0.07      # CLM: exp(logit_scale) with logit_scale0 = log(1/0.07)


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n, n


def _back_norm(g_hat, x_hat, n):
    """gradient through x_hat = x/|x|: (I - x_hat x_hat^T) g / |x|"""
    return (g_hat - (g_hat * x_hat).sum(1, keepdims=True) * x_hat) / n


def train(rng, p_s, p_a_s, mode):
    """CLM parameterisation: L2-normalised state/action embeddings, score = SCALE * cos."""
    Es = rng.standard_normal((S, D))
    Ea = rng.standard_normal((A, D))
    vEs, vEa = np.zeros_like(Es), np.zeros_like(Ea)
    for _ in range(STEPS):
        s, a = sample(rng, p_s, p_a_s, B)
        us, ns = _norm(Es)
        ua, na = _norm(Ea)
        xs = us[s]
        gUs = np.zeros_like(Es)
        gUa = np.zeros_like(Ea)
        if mode == "voc":                                        # CLM "Choice": softmax over the vocabulary
            logits = SCALE * xs @ ua.T                           # B x A
            g = softmax(logits)
            g[np.arange(B), a] -= 1.0
            g *= SCALE / B
            np.add.at(gUs, s, g @ ua)
            gUa += g.T @ xs
        else:                                                    # CLM bidirectional InfoNCE, in-batch
            xa = ua[a]
            logits = SCALE * xs @ xa.T                           # B x B
            same = (a[:, None] == a[None, :]) & ~np.eye(B, dtype=bool)
            logits = np.where(same, -1e9, logits)                # CLM false-negative (same-task) mask
            gr = softmax(logits, 1)
            gr[np.arange(B), np.arange(B)] -= 1.0
            gc = softmax(logits.T, 1)
            gc[np.arange(B), np.arange(B)] -= 1.0
            gL = SCALE * (gr + gc.T) / (2 * B)
            np.add.at(gUs, s, gL @ xa)
            np.add.at(gUa, a, gL.T @ xs)
        gEs = _back_norm(gUs, us, ns)
        gEa = _back_norm(gUa, ua, na)
        vEs = 0.9 * vEs + gEs
        vEa = 0.9 * vEa + gEa
        Es -= LR * vEs
        Ea -= LR * vEa
    us, _ = _norm(Es)
    ua, _ = _norm(Ea)
    return us * SCALE, ua                                        # scores = returned_s @ ua.T


def evaluate(scores, p_a_s, p_s):
    pick = scores.argmax(axis=1)
    hit = float(np.sum(p_s * p_a_s[np.arange(S), pick]))         # P(pick == fresh sample)
    sel_freq = np.bincount(pick, weights=p_s, minlength=A)
    true_marg = p_s @ p_a_s
    return {"hit_rate": round(hit, 4),
            "cruise_selected_share": round(float(sel_freq[CRUISE]), 4),
            "cruise_true_share": round(float(true_marg[CRUISE]), 4),
            "map_agreement": round(float(np.mean(pick == p_a_s.argmax(1))), 4)}


def main(seed=0):
    rng = np.random.default_rng(seed)
    p_s, p_a_s = make_world(rng)
    p_a = p_s @ p_a_s
    out = {"seed": seed, "S": S, "A": A, "D": D, "steps": STEPS, "batch": B}
    oracle = np.log(p_a_s + 1e-12)
    out["oracle_MAP"] = evaluate(oracle, p_a_s, p_s)
    out["oracle_PMI"] = evaluate(oracle - np.log(p_a + 1e-12), p_a_s, p_s)
    out["prior_only"] = evaluate(np.tile(np.log(p_a + 1e-12), (S, 1)), p_a_s, p_s)
    Es, Ea = train(rng, p_s, p_a_s, "nce")
    f = Es @ Ea.T
    out["nce_score_row_std"] = round(float(f.std(axis=1).mean()), 3)
    out["nce"] = evaluate(f, p_a_s, p_s)
    out["nce+pri(beta=1)"] = evaluate(f + np.log(p_a + 1e-12), p_a_s, p_s)
    grid = [0.5, 1, 2, 4, 8, 16, 32]
    best = max(grid, key=lambda b: evaluate(f + b * np.log(p_a + 1e-12), p_a_s, p_s)["hit_rate"])
    out["nce+pri(beta_fit)"] = dict(evaluate(f + best * np.log(p_a + 1e-12), p_a_s, p_s), beta=best)
    Es, Ea = train(rng, p_s, p_a_s, "voc")
    out["voc"] = evaluate(Es @ Ea.T, p_a_s, p_s)
    return out


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2]
    res = [main(s) for s in seeds]
    print(json.dumps(res, indent=1))
