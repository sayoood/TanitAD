#!/usr/bin/env python3
"""FIT-only (in-sample for P6), CPU, chunked float32 version of why_miss/neighbour_rule.py plus safety-only neighbour variants.
Descriptive: informs a card's candidate list; reads NO navtest token (E1/E2/E3 untouched); decides nothing."""
import json, sys
import numpy as np
NC, DAC, EP, TTC, C, DDC = range(6)
def rule(x): return x[..., NC] * x[..., DAC] * (5 * x[..., EP] + 5 * x[..., TTC] + 2 * x[..., C]) / 12
Z = np.load("F:/tanitad_scratch/p6fit/fit_pack8.npz", allow_pickle=False)
ok = Z["has_pdm"].astype(bool)
pdm = Z["pdm"].astype(np.float32)[ok]
PD = pdm[..., 6]; V = ~np.isnan(PD); PD = np.nan_to_num(PD)
L = Z["logits"][ok].astype(np.float32); XY = Z["traj_xy"][ok].astype(np.float32)
N = len(L); print("sets", N, "traj", XY.shape, file=sys.stderr)
P = 1 / (1 + np.exp(-L)); R = rule(P); R[~V] = 0; SAFE = P[..., NC] * P[..., DAC]; SAFE[~V] = 0
names = ["base"] + [f"smooth_mean_s{s}" for s in (0.25, 0.5, 1.0)] + [f"worst_nbr_r{r}" for r in (0.5, 1.0)] + [f"safe_worst_nbr_r{r}" for r in (0.25, 0.5, 1.0)] + [f"medoid_top{k}" for k in (3, 8)]
picks = {n: np.zeros(N, np.int64) for n in names}
CH = 256
for a in range(0, N, CH):
    b = min(N, a + CH); xy = XY[a:b]; r_ = R[a:b]; v = V[a:b]; sf = SAFE[a:b]
    d = np.sqrt(((xy[:, :, None] - xy[:, None]) ** 2).sum(-1)).mean(-1)  # [n,64,64] mean pointwise distance
    picks["base"][a:b] = r_.argmax(1)
    for s in (0.25, 0.5, 1.0):
        K = np.exp(-0.5 * (d / s) ** 2) * v[:, None]
        S = (K * r_[:, None]).sum(-1) / np.maximum(K.sum(-1), 1e-9); S[~v] = -1
        picks[f"smooth_mean_s{s}"][a:b] = S.argmax(1)
    for r in (0.5, 1.0):
        M = np.where((d <= r) & v[:, None], r_[:, None, :].repeat(64, 1), np.inf).min(-1)
        S = r_ * np.minimum(M / np.maximum(r_, 1e-9), 1.0); S[~v] = -1
        picks[f"worst_nbr_r{r}"][a:b] = S.argmax(1)
    for r in (0.25, 0.5, 1.0):
        M = np.where((d <= r) & v[:, None], sf[:, None, :].repeat(64, 1), np.inf).min(-1)  # worst neighbour SAFETY (incl. self)
        S = r_ * np.minimum(M / np.maximum(sf, 1e-9), 1.0); S[~v] = -1
        picks[f"safe_worst_nbr_r{r}"][a:b] = S.argmax(1)
    ix = np.arange(b - a)
    for tk in (3, 8):
        o = r_.argsort(1)[:, ::-1][:, :tk]
        sub = np.take_along_axis(np.take_along_axis(d, o[:, :, None], 1), o[:, None, :].repeat(tk, 1), 2)
        picks[f"medoid_top{tk}"][a:b] = o[ix, sub.mean(-1).argmin(1)]
ix = np.arange(N)
best = np.where(V, PD, -1).argmax(1)
base_v = PD[ix, picks["base"]]
out = {"n_sets": int(N), "best": float(100 * PD[ix, best].mean()), "note": "FIT, in-sample for P6, unweighted, no interval, descriptive only"}
for n in names:
    r = PD[ix, picks[n]]
    out[n] = {"pick": round(float(100 * r.mean()), 3), "minus_base": round(float(100 * (r - base_v).mean()), 3),
              "zero_picks": int((r == 0).sum()), "changed": int((picks[n] != picks["base"]).sum()),
              "better": int((r > base_v).sum()), "worse": int((r < base_v).sum())}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
