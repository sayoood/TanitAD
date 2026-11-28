"""CLOSED-LOOP v1.6 (+ run-6) — the predictor conditioned on the DECODER'S OWN actions.

PI request 2026-08-06: "let v1.6 drive in closed loop by conditioning the predictor by
actions from the decoder." This closes the standing 'WM rollout under TRUE future
actions' caveat *at the action channel*: the roll no longer sees any recorded future.
Perception context (8 frames to t0) is unchanged — this is closed-loop ACTION feedback
inside the world model's imagination, not re-perception simulation (that needs AlpaSim).

Action construction per step (the corpus contract, physicalai.signals_at: actions =
[steer, accel], steer = atan(wheelbase*curvature), legacy wheelbase 2.9):
    head emits (a_j, yr_j) with carried speed v_j -> kappa = yr_j / max(v_j, 0.3)
    steer_j = atan(2.9 * kappa), accel_j = a_j -> appended action [steer_j, accel_j, v0/SS]
The appended ego channel stays v0 (t0 speed), matching the open-loop eval convention
exactly, so the ONLY change vs the banked eval is where future actions come from.

Emits per-episode npz (stride-1, same grid as /workspace/v16_eval/dump):
    g   GT waypoints             o16 open-loop v1.6 (true actions; byte-check vs banked)
    o6  open-loop run-6 head     c16 closed-loop v1.6      c6  closed-loop run-6
    h16 hold-action v1.6 (for the queued S-curve decisive arm)   ws  window origins
All metrics (families, lag, S-curve, gates) are computed downstream on the dump — this
script only rolls and dumps.
"""
import glob
import os
import time

import numpy as np
import torch

DT, K, W = 0.1, 20, 8
DEV = "cuda"
WHEELBASE = 2.9

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


def load_head(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    h = UnicycleStepReadout(2048, hidden=512, speed_input=False,
                            predict_delta=False).to(DEV)
    h.load_state_dict(ck["head"])
    h.eval()
    return h


head16 = load_head("/workspace/experiments/unicycle-readout-v2-latentsonly/unicycle_readout.pt")
head6 = load_head("/workspace/experiments/unicycle-readout-v3-speedloss/unicycle_readout.pt")
print("[loaded] trunk + v1.6 + run6 heads", flush=True)


def decode_open(head, trans, v0):
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


def roll_closed(head, states, awE, v0, ego):
    """Closed loop: each predictor step is conditioned on the action the head just chose."""
    win_s, win_a = states, awE
    v = v0.clone()
    ap = torch.zeros_like(v)
    yp = torch.zeros_like(v)
    rows = []
    for j in range(K):
        z_hat = model.predictor(win_s, win_a)[1]
        aj, yj = head(win_s[:, -1], z_hat, v, ap, yp)
        rows.append(torch.stack([v * DT, torch.zeros_like(v), yj * DT], -1))
        v = (v + aj * DT).clamp_min(0.0)
        ap, yp = aj, yj
        if j < K - 1:
            kappa = yj / v.clamp_min(0.3)
            steer = torch.atan(WHEELBASE * kappa)
            a_next = torch.stack([steer, aj], -1)
            a_next = torch.cat([a_next, ego], -1)          # ego channel: v0 held (eval convention)
            win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], 1)
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], 1)
    return accumulate_se2(torch.stack(rows, 1))


files = sorted(glob.glob("/workspace/pai_epcache/physicalai-oodval-6f4b94e4c7ce-q90/ep_*.pt"))[:40]
os.makedirs("/workspace/v16_eval/dump_cl", exist_ok=True)
t0 = time.time()
for fi, f in enumerate(files):
    ep = load_frames([f])[0]
    poses = ep.poses.float()
    T = min(ep.feats.shape[0], poses.shape[0], ep.actions.shape[0])
    starts = list(range(0, T - W - K))                     # stride-1, same grid as the dump
    acc = {k: [] for k in ("g", "o16", "o6", "c16", "c6", "h16")}
    lastl = []
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

            tr = rollout_transitions(model.predictor, states, awE, faE, K)
            acc["o16"].append(decode_open(head16, tr, v0).float().cpu().numpy())
            acc["o6"].append(decode_open(head6, tr, v0).float().cpu().numpy())
            acc["c16"].append(roll_closed(head16, states, awE, v0, ego).float().cpu().numpy())
            acc["c6"].append(roll_closed(head6, states, awE, v0, ego).float().cpu().numpy())
            faH = awE[:, -1:, :].expand(-1, K, -1)
            trH = rollout_transitions(model.predictor, states, awE, faH, K)
            acc["h16"].append(decode_open(head16, trH, v0).float().cpu().numpy())
            fp = torch.stack([poses[s + W:s + W + K] for s in ch]).to(DEV)
            acc["g"].append(gt_ego_waypoints(pl, fp, list(range(1, K + 1))).cpu().numpy())
            lastl += [int(x) for x in last]
    np.savez_compressed(f"/workspace/v16_eval/dump_cl/ep{fi:03d}.npz",
                        **{k: np.concatenate(v).astype(np.float32) for k, v in acc.items()},
                        ws=np.array(lastl))
    print(f"  [{fi+1}/40] {time.time()-t0:.0f}s", flush=True)
print("CLDUMP_DONE", flush=True)
