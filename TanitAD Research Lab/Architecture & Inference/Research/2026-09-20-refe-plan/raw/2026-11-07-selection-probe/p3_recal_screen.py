#!/usr/bin/env python3
"""P3 (Rule Zero continuation, SCREEN ONLY -- TRAIN logs only, never the held-out logs).

P1 measured that the scorer's safety heads FALL with proposal speed while their truths are flat (within-token Spearman vs v4/v0, ALL
CRUISE: pred TTC -0.48 / NC -0.44 / DAC -0.29, true TTC -0.05 / NC -0.04 / DAC -0.04), and on the sim token the four fast proposals
(true PDMS 1.000) lose on TTC (-0.16..-0.66) and NC. Lever: recalibrate NC, DAC, EP, TTC with a speed covariate,
    logit'_j = a_j * logit_j + b_j * d + c_j,   d = clip(v4/v0, 0, 2) - 1     (current ego speed + the proposal's own speed only)
fit by logistic regression (IRLS; soft label for EP) on proposals of the fit folds, then pick = argmax navsim_v1_lane(p').
Screened by 5-fold LOG cross-validation INSIDE the TRAIN logs (folds by sorted log, seed 20261108). Variants: ungated, ego-gated.
This is a screen to decide whether the arm earns a fresh-token confirmation; it is NOT decision evidence (TRAIN was explored).
"""
import json

import numpy as np

import sel_common as sc
import sel_rules as sr

D = sc.load()
split = json.load(open(sc.OUT + "/split.json"))
tr = np.isin(D["log"], split["train_logs"])
assert not np.isin(D["log"][tr], split["heldout_logs"]).any()
W = {"zero": 529 / 12146, "random": 11617 / 12146}
v0 = np.maximum(D["v0"], 1e-6)
dcov = np.clip(D["v4_all"] / v0[:, None], 0, 2) - 1.0
GATE = (D["v0"] > 5.0) & (np.abs(D["ax0"]) < 0.3)
L = D["logits"].astype(np.float64)
HEADS_RECAL = (0, 1, 2, 3)          # NC, DAC, EP, TTC (truth exists); C, DDC, lane untouched


def irls(X, y, w=None, iters=30, ridge=1e-3):
    beta = np.zeros(X.shape[1]); beta[0] = 1.0
    w = np.ones(len(y)) if w is None else w
    for _ in range(iters):
        z = X @ beta
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (w * (p - y)) + ridge * beta
        H = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def fit(tok_mask, use_speed=True):
    m = tok_mask[:, None] & D["valid"]
    B = {}
    for j in HEADS_RECAL:
        cols = [L[..., j][m], np.ones(m.sum())] + ([dcov[m]] if use_speed else [])
        X = np.stack(cols, 1)
        B[j] = irls(X, D["sub"][..., j][m])
    return B


def recal_logits(B, use_speed=True):
    Lr = L.copy()
    for j, b in B.items():
        Lr[..., j] = b[0] * L[..., j] + b[1] + (b[2] * dcov if use_speed else 0.0)
    return Lr


def evaluate(pk, mask):
    r = np.arange(len(pk))
    base = D["pick"]
    st = D["stratum"]
    dp = D["pdms"][r, pk] - D["pdms"][r, base]
    mix = 100 * (W["zero"] * dp[mask & (st == "zero")].mean() + W["random"] * dp[mask & (st == "random")].mean())
    ratio = D["v4_all"][r, pk] / v0
    cm = mask & D["cruise"]
    return {"dmix_x100": float(mix), "slow_base": float((D["v4_all"][r, base] / v0 < 0.85)[cm].mean()), "slow_new": float((ratio < 0.85)[cm].mean()),
            "d_ttc_fail": float(((D["sub"][r, pk, 3] < 1).astype(float) - (D["sub"][r, base, 3] < 1))[mask].mean()),
            "changed": float((pk != base)[mask].mean()), "n_tok": int(mask.sum()), "n_cruise": int(cm.sum())}


tl = np.array(sorted(split["train_logs"]))
perm = np.random.default_rng(20261108).permutation(len(tl))
folds = [set(tl[perm[k::5]]) for k in range(5)]
out = {"folds": [sorted(f) for f in folds], "arms": {}}
arms = {"R_speed_ungated": (True, False), "R_speed_gated": (True, True), "R_nospeed_ungated": (False, False)}
for name, (use_speed, gated) in arms.items():
    pk_cv = D["pick"].copy()
    betas = []
    for f in folds:
        te = np.isin(D["log"], list(f))
        B = fit(tr & ~te, use_speed)
        betas.append({str(j): [float(x) for x in b] for j, b in B.items()})
        pk = np.argmax(sc.agg_np(recal_logits(B, use_speed), np.float64), -1)
        if gated:
            pk = np.where(GATE, pk, D["pick"])
        pk_cv[te] = pk[te]
    out["arms"][name] = {"cv_over_train": evaluate(pk_cv, tr), "betas_per_fold": betas}
    print(name, out["arms"][name]["cv_over_train"])
# full-TRAIN fit (the spec a confirmation would freeze) + the sim token
B = fit(tr, True)
out["full_train_fit_speed"] = {sc.HEADS[j]: [float(x) for x in b] for j, b in B.items()}
Lr = recal_logits(B, True)
i = int(np.where(D["token"] == sc.SIM_TOKEN)[0][0])
k = int(np.argmax(sc.agg_np(Lr[i], np.float64)))
out["sim_token_full_train_fit"] = {"banked_pick": int(D["pick"][i]), "recal_pick": k, "v4_over_v0": float(D["v4_all"][i, k] / v0[i]),
                                   "pdms": float(D["pdms"][i, k]), "note": "TRAIN token: spec check only"}
print("full-train betas (a, c, b_speed)", out["full_train_fit_speed"])
print("sim", out["sim_token_full_train_fit"])
json.dump(out, open(sc.OUT + "/p3_recal_screen.json", "w"), indent=1)
