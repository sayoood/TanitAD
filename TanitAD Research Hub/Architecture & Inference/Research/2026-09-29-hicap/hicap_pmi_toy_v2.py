#!/usr/bin/env python3
"""HiCAP PMI toy v2 (numpy only): corrects the two protocol defects the red team found in reff_pmi_toy.py.

Defects of v1: (a) the log-prior weight beta was chosen by maximising the EVALUATION hit rate computed with the TRUE
p(a|s) and the TRUE marginal, on a grid whose lower edge (0.5) was selected in all five seeds; (b) the loss trained was
only the masked one, so the row labelled "the InfoNCE optimum" was never trained.
v2: trains BOTH the unmasked and the masked (CLM same-action mask) bidirectional InfoNCE critic in the same synthetic
world; estimates the prior q_hat from the TRAINING draws; fits beta on a HELD-OUT sample of expert actions by empirical
top-1 accuracy over a grid that brackets zero on both sides; evaluates hit rate against the true p(a|s) on fresh states.
Evidence class: MEASURED (toy). Mechanism only: a low-rank critic in a synthetic world, not driving data.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "TanitAD Research Hub/Architecture & Inference/Research")
import reff_pmi_toy as T  # noqa: E402

GRID = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def train_critic(rng, p_s, p_a_s, masked):
    """Same recipe as reff_pmi_toy.train(mode='nce'), with the false-negative mask switchable."""
    S, A, D, B, STEPS, LR, SCALE = T.S, T.A, T.D, T.B, T.STEPS, T.LR, T.SCALE
    Es = rng.standard_normal((S, D)); Ea = rng.standard_normal((A, D))
    vEs, vEa = np.zeros_like(Es), np.zeros_like(Ea)
    for _ in range(STEPS):
        s, a = T.sample(rng, p_s, p_a_s, B)
        us, ns = T._norm(Es); ua, na = T._norm(Ea)
        xs = us[s]; xa = ua[a]
        gUs = np.zeros_like(Es); gUa = np.zeros_like(Ea)
        logits = SCALE * xs @ xa.T
        if masked:
            same = (a[:, None] == a[None, :]) & ~np.eye(B, dtype=bool)
            logits = np.where(same, -1e9, logits)
        gr = T.softmax(logits, 1); gr[np.arange(B), np.arange(B)] -= 1.0
        gc = T.softmax(logits.T, 1); gc[np.arange(B), np.arange(B)] -= 1.0
        gL = SCALE * (gr + gc.T) / (2 * B)
        np.add.at(gUs, s, gL @ xa); np.add.at(gUa, a, gL.T @ xs)
        gEs = T._back_norm(gUs, us, ns); gEa = T._back_norm(gUa, ua, na)
        vEs = 0.9 * vEs + gEs; vEa = 0.9 * vEa + gEa
        Es -= LR * vEs; Ea -= LR * vEa
    us, _ = T._norm(Es); ua, _ = T._norm(Ea)
    return (us * SCALE) @ ua.T                                        # [S, A] scores


def hit(score, p_a_s, p_s):
    pick = score.argmax(1)
    return float(np.sum(p_s * p_a_s[np.arange(T.S), pick]))


def main(seed):
    rng = np.random.default_rng(seed)
    p_s, p_a_s = T.make_world(rng)
    p_a = p_s @ p_a_s
    # estimate the prior from TRAINING draws (not the true marginal)
    s_tr, a_tr = T.sample(rng, p_s, p_a_s, 20000)
    q_hat = np.bincount(a_tr, minlength=T.A) + 0.5
    q_hat = q_hat / q_hat.sum()
    # held-out expert actions for beta selection
    s_ho, a_ho = T.sample(rng, p_s, p_a_s, 20000)
    out = {"seed": seed, "true_cruise_share": round(float(p_a[T.CRUISE]), 4),
           "oracle_MAP_hit": round(hit(np.log(p_a_s + 1e-12), p_a_s, p_s), 4),
           "oracle_PMI_hit": round(hit(np.log(p_a_s + 1e-12) - np.log(p_a + 1e-12), p_a_s, p_s), 4),
           "prior_only_hit": round(hit(np.tile(np.log(q_hat), (T.S, 1)), p_a_s, p_s), 4)}
    for name, masked in (("unmasked", False), ("masked", True)):
        f = train_critic(rng, p_s, p_a_s, masked)
        acc = {b: float((np.argmax(f[s_ho] + b * np.log(q_hat)[None, :], axis=1) == a_ho).mean()) for b in GRID}
        b_star = max(GRID, key=lambda b: acc[b])
        row = {"raw_hit": round(hit(f, p_a_s, p_s), 4),
               "beta1_hit": round(hit(f + np.log(q_hat), p_a_s, p_s), 4),
               "beta_fit": b_star,
               "beta_fit_hit": round(hit(f + b_star * np.log(q_hat), p_a_s, p_s), 4),
               "raw_cruise_share": round(float(np.mean(f.argmax(1) == T.CRUISE)), 3),
               "beta_fit_cruise_share": round(float(np.mean((f + b_star * np.log(q_hat)).argmax(1) == T.CRUISE)), 3)}
        out[name] = row
    # soft-target classifier (vocabulary softmax) for reference
    Es_, Ea_ = T.train(rng, p_s, p_a_s, "voc")
    out["softmax_classifier_hit"] = round(hit(Es_ @ Ea_.T, p_a_s, p_s), 4)
    return out


if __name__ == "__main__":
    res = [main(s) for s in (0, 1, 2, 3, 4)]
    agg = {}
    for key in ("oracle_MAP_hit", "oracle_PMI_hit", "prior_only_hit", "softmax_classifier_hit"):
        v = np.array([r[key] for r in res]); agg[key] = [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)]
    for name in ("unmasked", "masked"):
        for key in ("raw_hit", "beta1_hit", "beta_fit_hit", "raw_cruise_share", "beta_fit_cruise_share"):
            v = np.array([r[name][key] for r in res]); agg[f"{name}.{key}"] = [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)]
        agg[f"{name}.beta_fit_values"] = [r[name]["beta_fit"] for r in res]
    print(json.dumps({"per_seed": res, "mean_sd_over_5_seeds": agg}, indent=1))
