#!/usr/bin/env python3
"""Descriptive rule-family probe: does judging a plan by its neighbours avoid near-twin grazes? FIT (in-sample for P6) and E1 (already read). E2/E3 NOT read. Decides nothing; nominating needs the FIT split and a card."""
import sys, json
import numpy as np
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read")
NC, DAC, EP, TTC, C, DDC = range(6)
def rule(x): return x[..., NC] * x[..., DAC] * (5 * x[..., EP] + 5 * x[..., TTC] + 2 * x[..., C]) / 12
def run(L, PD, V, w, xy, name):
    N = len(L); ix = np.arange(N); P = 1 / (1 + np.exp(-L.astype(np.float64))); R = rule(P).copy(); R[~V] = 0
    d = np.sqrt(((xy[:, :, None] - xy[:, None]) ** 2).sum(-1)).mean(-1)  # [N,64,64]
    base = (PD[ix, R.argmax(1)] * w).sum() / w.sum() * 100
    out = {"base": base}
    for sig in (0.25, 0.5, 1.0):
        K = np.exp(-0.5 * (d / sig) ** 2) * V[:, None]
        S = (K * R[:, None]).sum(-1) / np.maximum(K.sum(-1), 1e-9); S[~V] = -1
        out[f"smooth_mean_sigma{sig}"] = (PD[ix, S.argmax(1)] * w).sum() / w.sum() * 100 - base
    for r in (0.5, 1.0):
        M = np.where((d <= r) & V[:, None], R[:, None].repeat(64, 1), np.inf).min(-1); S = R * np.minimum(M / np.maximum(R, 1e-9), 1.0) ** 1.0; S[~V] = -1
        out[f"worst_neighbour_r{r}"] = (PD[ix, S.argmax(1)] * w).sum() / w.sum() * 100 - base
    for tk in (3, 8):
        o = R.argsort(1)[:, ::-1][:, :tk]
        sub = np.take_along_axis(np.take_along_axis(d, o[:, :, None], 1), o[:, None, :].repeat(tk, 1), 2)
        med = sub.mean(-1).argmin(1)
        out[f"medoid_of_top{tk}"] = (PD[ix, o[ix, med]] * w).sum() / w.sum() * 100 - base
    print(name, json.dumps({k: round(v, 3) for k, v in out.items()}))
Z = np.load("F:/tanitad_scratch/p6fit/fit_pack8.npz", allow_pickle=False); ok = Z["has_pdm"].astype(bool); pdm = Z["pdm"].astype(np.float64)[ok]
run(Z["logits"][ok], pdm[..., 6], ~np.isnan(pdm[..., 6]), np.ones(int(ok.sum())), Z["traj_xy"][ok].astype(np.float64), "FIT")
import selc6_analyze as SA
D0 = np.load("F:/TanitAD_p6read_work/run/dump/dump_p6.npz", allow_pickle=False); T = np.load("F:/TanitAD_p6read_work/run/score/table_p6.npz", allow_pickle=False)
dtok = [str(t) for t in D0["token"]]
arrs = {k: D0[k] for k in ("logits", "pick", "traj", "props8", "ego", "gt_valid", "gt_pose20", "gt_t20", "human8", "frame_ctrl_m", "fingerprint")}; arrs.update(pdms=T["pdms"], sub=T["sub"], valid=T["valid"])
D, A = SA.align_to_c3(arrs, dtok); zs = np.asarray(D["stratum"]) == "zero"; w = np.where(zs, SA.W_ZERO / zs.sum(), SA.W_RANDOM / (~zs).sum())
run(A["logits"], A["pdms"].astype(np.float64), A["valid"].astype(bool), w, A["props8"][..., :2].astype(np.float64), "E1")
