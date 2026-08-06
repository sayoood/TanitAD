"""Does v1.6 read VISION, or does it extrapolate ego dynamics? — the discriminating probe.

The head's input surface is latents-only (net.1.weight [512, 4096] = 2*state_dim exactly;
head_speed_input=False in the artifact). But two ego-dynamics channels remain upstream and
this probe ablates each SEPARATELY so their shares are attributable:

  arm `real`        real context latents, predictor rolled under TRUE future actions
                    (= the banked eval condition, the baseline).
  arm `hold_act`    SAME context latents, future actions HELD at the last context action
                    -> the roll keeps vision but loses future ego intent. If v1.6 tracks
                    `real`, the action channel is not what it reads; if it collapses, the
                    "WM rollout" was an action echo.
  arm `mean_lat`    batch-mean context latents, TRUE future actions -> the roll keeps ego
                    intent but loses vision (the wm_reliance ablation arm).
  arm `cv`          hold v0, straight — the no-information floor.

Attribution: degradation(real->mean_lat) is the VISION share; degradation(real->hold_act)
is the FUTURE-ACTION share. Both are reported against the cv floor.

PLUS the interpretability instrument: ridge probes (closed form, 5-fold episode split)
from z_ctx (last context state) and z_hat_k (predicted latent at k=5,10,20) to physical
quantities (v, accel, yaw rate at t0+k). R^2 says what the WM state and its PREDICTION
linearly encode — the first quantitative read on the latent content.
"""
import glob
import json
import time

import numpy as np
import torch

DT, K, W = 0.1, 20, 8
DEV = "cuda"

from taniteval import loaders
from taniteval.data import load_frames
from taniteval.rollout import SPEED_SCALE
from tanitad.models.metric_dynamics import (UnicycleStepReadout, accumulate_se2,
                                            gt_ego_waypoints, rollout_transitions)

L = loaders.load({"arch": "flagship-worldmodel-v2",
                  "ckpt": "/workspace/experiments/flagship-v1arch-v2bal-30k/ckpt.pt",
                  "run_config": "/workspace/experiments/flagship-v1arch-v2bal-30k/config.json",
                  "speed_input": True}, device=DEV)
model = L["model"].eval()
for p in model.parameters():
    p.requires_grad_(False)
ck = torch.load("/workspace/experiments/unicycle-readout-v2-latentsonly/unicycle_readout.pt",
                map_location="cpu", weights_only=False)
head = UnicycleStepReadout(2048, hidden=512, speed_input=False,
                           predict_delta=False).to(DEV)
head.load_state_dict(ck["head"])
head.eval()
print("[loaded] trunk + v1.6 head", flush=True)


def decode_unicycle(trans, v0):
    v = v0.clone()
    ap = torch.zeros_like(v)
    yp = torch.zeros_like(v)
    rows = []
    for zp, zh in trans:
        aj, yj = head(zp, zh, v, ap, yp)
        rows.append(torch.stack([v * DT, torch.zeros_like(v), yj * DT], -1))
        v = (v + aj * DT).clamp_min(0.0)
        ap, yp = aj, yj
    return accumulate_se2(torch.stack(rows, 1))


files = sorted(glob.glob("/workspace/pai_epcache/physicalai-oodval-6f4b94e4c7ce-q90/ep_*.pt"))[:40]
arms = ("real", "hold_act", "mean_lat", "cv")
per_ep = {a: [] for a in arms}
gt_by_ep = []
Z_ctx, Z_hat = [], {5: [], 10: [], 20: []}
TGT = {k: {"v": [], "accel": [], "yawrate": []} for k in (0, 5, 10, 20)}
EP_ID = []
t0 = time.time()
for fi, f in enumerate(files):
    ep = load_frames([f])[0]
    poses = ep.poses.float()
    T = min(ep.feats.shape[0], poses.shape[0], ep.actions.shape[0])
    starts = list(range(0, T - W - K, 8))               # the stride-8 grid
    outs = {a: [] for a in arms}
    gts = []
    with torch.no_grad():
        for i0 in range(0, len(starts), 16):
            ch = starts[i0:i0 + 16]
            last = torch.tensor([s + W - 1 for s in ch])
            fw = torch.stack([torch.as_tensor(ep.feats[s:s + W]) for s in ch]).to(DEV).float().div_(255.0)
            aw = torch.stack([ep.actions[s:s + W] for s in ch]).to(DEV).float()
            fa = torch.stack([ep.actions[s + W:s + W + K] for s in ch]).to(DEV).float()
            pl = poses[last].to(DEV)
            ego = pl[:, 3:4] / SPEED_SCALE
            awE = torch.cat([aw, ego[:, None].expand(-1, aw.shape[1], -1)], -1)
            faE = torch.cat([fa, ego[:, None].expand(-1, fa.shape[1], -1)], -1)
            states = model.encode_window(fw)
            v0 = pl[:, 3]

            # -- real: vision + true future actions --------------------------------
            tr = rollout_transitions(model.predictor, states, awE, faE, K)
            outs["real"].append(decode_unicycle(tr, v0).float().cpu().numpy())

            # bank latents for the probes (from the REAL roll only)
            Z_ctx.append(states[:, -1].float().cpu().numpy())
            for kk in (5, 10, 20):
                Z_hat[kk].append(tr[kk - 1][1].float().cpu().numpy())
            fp = torch.stack([poses[s + W:s + W + K] for s in ch]).to(DEV)
            for kk in (0, 5, 10, 20):
                pk = pl if kk == 0 else fp[:, kk - 1]
                TGT[kk]["v"].append(pk[:, 3].float().cpu().numpy())
            # accel/yawrate at t0+k from pose speed/yaw finite differences
            allp = torch.cat([pl[:, None], fp], 1)      # [B, K+1, 4]
            vv = allp[:, :, 3]
            yy = allp[:, :, 2]
            acc = (vv[:, 1:] - vv[:, :-1]) / DT
            dyaw = (yy[:, 1:] - yy[:, :-1] + np.pi) % (2 * np.pi) - np.pi
            yr = dyaw / DT
            for kk in (0, 5, 10, 20):
                j = min(kk, K - 1)
                TGT[kk]["accel"].append(acc[:, j].float().cpu().numpy())
                TGT[kk]["yawrate"].append(yr[:, j].float().cpu().numpy())
            EP_ID.append(np.full(len(ch), fi))

            # -- hold_act: vision kept, future ego intent removed ------------------
            faH = awE[:, -1:, :].expand(-1, K, -1)
            trH = rollout_transitions(model.predictor, states, awE, faH, K)
            outs["hold_act"].append(decode_unicycle(trH, v0).float().cpu().numpy())

            # -- mean_lat: vision removed, ego intent kept -------------------------
            sM = states.mean(0, keepdim=True).expand_as(states).contiguous()
            trM = rollout_transitions(model.predictor, sM, awE, faE, K)
            outs["mean_lat"].append(decode_unicycle(trM, v0).float().cpu().numpy())

            # -- cv floor ----------------------------------------------------------
            cv = torch.zeros(len(ch), K, 3, device=DEV)
            cv[:, :, 0] = (v0[:, None] * DT).cumsum(1)
            outs["cv"].append(cv.float().cpu().numpy())

            gts.append(gt_ego_waypoints(pl, fp, list(range(1, K + 1))).cpu().numpy())
    for a in arms:
        per_ep[a].append(np.concatenate(outs[a]))
    gt_by_ep.append(np.concatenate(gts))
    if (fi + 1) % 10 == 0:
        print(f"  [{fi+1}/40] {time.time()-t0:.0f}s", flush=True)

def fam(P, G):
    ade = float(np.linalg.norm(P[..., :2] - G[..., :2], axis=-1).mean())
    d = np.diff(np.concatenate([np.zeros((P.shape[0], 1, 2)), P[..., :2]], 1), axis=1)
    dg = np.diff(np.concatenate([np.zeros((G.shape[0], 1, 2)), G[..., :2]], 1), axis=1)
    sp, sg = np.linalg.norm(d, axis=-1) / DT, np.linalg.norm(dg, axis=-1) / DT
    return ade, float(np.abs(sp - sg).mean())


out = {"_grid": "stride-8, 40 OOD-val q90 episodes",
       "_arms": {"real": "vision + true future actions (banked eval condition)",
                 "hold_act": "vision kept, future actions held at last context action",
                 "mean_lat": "batch-mean latents, true actions (vision removed)",
                 "cv": "hold v0 straight"}}
G = np.concatenate(gt_by_ep)
ades = {}
for a in arms:
    P = np.concatenate(per_ep[a])
    ade, smae = fam(P, G)
    ades[a] = ade
    out[a] = {"ade_m": ade, "speed_mae_mps": smae, "n_windows": int(P.shape[0])}
    print(f"[{a}] ade={ade:.4f}  speed_mae={smae:.4f}", flush=True)
denom = ades["cv"] - ades["real"]
out["attribution"] = {
    "vision_share_of_cv_gap": float((ades["mean_lat"] - ades["real"]) / denom),
    "future_action_share_of_cv_gap": float((ades["hold_act"] - ades["real"]) / denom),
    "_note": "shares can sum !=1 (interaction); each is degradation/(cv-real)."}
print("[attribution]", out["attribution"], flush=True)

# ---- ridge probes: what do z_ctx and z_hat_k linearly encode? -----------------
Zc = np.concatenate(Z_ctx)
EP = np.concatenate(EP_ID)


def ridge_r2(Z, y, ep, lam=10.0):
    r2 = []
    for f5 in range(5):
        tr = (ep % 5) != f5                      # episode-disjoint folds
        te = ~tr
        if te.sum() < 10:
            continue
        Zt = Z[tr] - Z[tr].mean(0)
        yt = y[tr] - y[tr].mean()
        A = Zt.T @ Zt + lam * np.eye(Z.shape[1])
        wgt = np.linalg.solve(A, Zt.T @ yt)
        pred = (Z[te] - Z[tr].mean(0)) @ wgt + y[tr].mean()
        ss = ((y[te] - y[te].mean()) ** 2).sum()
        r2.append(1.0 - ((y[te] - pred) ** 2).sum() / max(ss, 1e-9))
    return float(np.mean(r2))


probes = {}
for kk in (0, 5, 10, 20):
    Zk = Zc if kk == 0 else np.concatenate(Z_hat[kk])
    src = "z_ctx" if kk == 0 else f"z_hat_{kk}"
    for tgt in ("v", "accel", "yawrate"):
        y = np.concatenate(TGT[kk][tgt])
        m = np.isfinite(y)
        probes[f"{src}->{tgt}@t+{kk*DT:.1f}s"] = ridge_r2(Zk[m], y[m], EP[m])
# cross-check: can the CONTEXT state alone predict the future? (ego-extrapolation ceiling)
for kk in (10, 20):
    y = np.concatenate(TGT[kk]["v"])
    m = np.isfinite(y)
    probes[f"z_ctx->v@t+{kk*DT:.1f}s"] = ridge_r2(Zc[m], y[m], EP[m])
out["latent_probes_r2"] = {k: round(v, 4) for k, v in probes.items()}
for k, v in out["latent_probes_r2"].items():
    print(f"[probe] {k:28s} R2={v:.4f}", flush=True)

json.dump(out, open("/workspace/v16_eval/vision_vs_ego.json", "w"), indent=1)
print("VVE_DONE", flush=True)
