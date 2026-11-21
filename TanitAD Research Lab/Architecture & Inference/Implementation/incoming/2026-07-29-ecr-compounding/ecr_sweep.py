"""E-CR — CR_k = e_rollout / e_teacher-forced. Resolves C61.

Mirrors train_flagship_v4.canary_rollout EXACTLY (same loader, same ds, same actions,
same step_readout, same rollout_decode primitive) and adds ONE arm: a teacher-forced
roll whose window advances with the TRUE latent instead of the prediction.
Pre-registered: Project Steering/PREREG_deep_research_2026-07-29.md
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import torch, numpy as np
import eval_flagship_v4 as E
from tanitad.models.flagship_v15 import SPEED_SCALE
from tanitad.models.metric_dynamics import gt_ego_waypoints, accumulate_se2

CK = sys.argv[1]
VAL = "/workspace/val40cache"
OUT = sys.argv[2]
EPISODES = int(sys.argv[3]) if len(sys.argv) > 3 else 40
K_MAX = 20
REPORT_K = (4, 8, 16, 20)
STRIDE, BATCH = 8, 8

def tf_transitions(predictor, states, actions, future_actions, z_true, k):
    win_s, win_a = states, actions
    trans = []
    for j in range(k):
        z_hat = predictor(win_s, win_a)[1]
        trans.append((win_s[:, -1], z_hat))
        if j < k - 1:
            a_next = future_actions[:, j]
            win_s = torch.cat([win_s[:, 1:], z_true[:, j].unsqueeze(1)], dim=1)   # TRUE, not z_hat
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], dim=1)
    return trans

def roll_transitions(predictor, states, actions, future_actions, k):
    win_s, win_a = states, actions
    trans = []
    for j in range(k):
        z_hat = predictor(win_s, win_a)[1]
        trans.append((win_s[:, -1], z_hat))
        if j < k - 1:
            a_next = future_actions[:, j]
            win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], dim=1)
    return trans

def decode_wp(step_readout, trans, k):
    # ⚠️ MUST run INSIDE autocast, exactly as canary_rollout does — the step_readout
    # is bf16 under autocast and raises on fp32 input. Error is taken in fp32 AFTER.
    dp = torch.stack([step_readout(trans[j][0], trans[j][1]) for j in range(k)], 1)
    return accumulate_se2(dp)

dev = torch.device("cuda")
ck = torch.load(CK, map_location="cpu", weights_only=False)
world, grounding, step = E.load_v1_from_ck(ck, dev)[:3]
sr = grounding.step["op"]
import glob
from tanitad.data.mixing import load_episode
eps = [load_episode(f, mmap=True) for f in sorted(glob.glob(VAL + "/ep_*.pt"))]
cfg = E._eval_cfg(None); plan = E._plan(cfg)
ds = E.build_val_dataset_base(eps, cfg, plan)
print(f"[ecr] ckpt step={step} ds={len(ds)} episodes<={EPISODES}", flush=True)
pos = {}
for i, (e, t) in enumerate(ds.index):
    pos[(e, t)] = i
sel = [i for i, (e, t) in enumerate(ds.index) if e < EPISODES and t % STRIDE == 0]
print(f"[ecr] windows={len(sel)}", flush=True)
RA, TA, EP = [], [], []
t0 = time.time()
for b0 in range(0, len(sel), BATCH):
    idx = sel[b0:b0 + BATCH]
    items = [ds[i] for i in idx]
    ets = [ds.index[i] for i in idx]
    fr = torch.stack([x["frames"] for x in items]).to(dev)
    aw2 = torch.stack([x["actions"] for x in items]).to(dev).float()
    fa2 = torch.stack([x["future_actions"] for x in items]).to(dev).float()
    fp = torch.stack([x["future_poses"] for x in items]).to(dev).float()
    pl = torch.stack([x["pose_last"] for x in items]).to(dev).float()
    vch = (pl[:, 3] / SPEED_SCALE)[:, None, None]
    aw = torch.cat([aw2, vch.expand(-1, aw2.shape[1], -1)], -1)
    fa = torch.cat([fa2, vch.expand(-1, fa2.shape[1], -1)], -1)
    gt = gt_ego_waypoints(pl, fp, list(range(1, K_MAX + 1)))
    # TRUE future latents: the last frame of the window starting j+1 later
    futf = []
    ok = True
    for (e, t) in ets:
        seq = []
        for j in range(K_MAX):
            k = pos.get((e, t + j + 1))
            if k is None:
                ok = False; break
            seq.append(ds[k]["frames"][-1])
        if not ok: break
        futf.append(torch.stack(seq))
    if not ok:
        continue
    fut = torch.stack(futf).to(dev)                       # [B,K,C,H,W]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        states = world.encode_window(fr)
        B, Kk = fut.shape[0], fut.shape[1]
        z_true = world.encode(fut.reshape(B * Kk, *fut.shape[2:])).reshape(B, Kk, -1)
        tr_r = roll_transitions(world.predictor, states, aw, fa, K_MAX)
        tr_t = tf_transitions(world.predictor, states, aw, fa, z_true, K_MAX)
        wp_r = decode_wp(sr, tr_r, K_MAX)
        wp_t = decode_wp(sr, tr_t, K_MAX)
    RA.append((wp_r.float() - gt).norm(dim=-1).cpu())
    TA.append((wp_t.float() - gt).norm(dim=-1).cpu())
    EP.append(torch.tensor([e for (e, t) in ets]))
    if b0 % (BATCH * 20) == 0:
        print(f"[ecr] {b0}/{len(sel)} ({time.time()-t0:.0f}s)", flush=True)
RA = torch.cat(RA); TA = torch.cat(TA); EP = torch.cat(EP)
out = {"ckpt": CK, "ckpt_step": int(step), "val": VAL, "k_max": K_MAX,
       "n_windows": int(RA.shape[0]), "n_episodes": int(EP.unique().numel()), "CR": {}}
for k in REPORT_K:
    er, et = float(RA[:, k-1].mean()), float(TA[:, k-1].mean())
    out["CR"][f"k{k}"] = {"e_rollout": round(er, 5), "e_teacher_forced": round(et, 5),
                          "CR": round(er / max(et, 1e-9), 4),
                          "ER": round(float((RA[:, k-1] - RA[:, k-2]).mean()), 5)}
np.savez("/workspace/ecr_arrays.npz", rollout=RA.numpy(), tf=TA.numpy(), episode=EP.numpy())
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out["CR"], indent=1), flush=True)
print(f"-> {OUT} (+/workspace/ecr_arrays.npz for the PAIRED bootstrap)", flush=True)
