"""Smoke: frozen v1 WM as a differentiable simulator.
Validate (1) rollout under GT actions reproduces ~0.45 ADE, (2) gradient of ADE
flows back into the fed future-action tensor (the analytic-gradient path)."""
import sys, torch
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
from taniteval.loaders import load
from taniteval.data import load_raw, list_val_episodes
from tanitad.models.metric_dynamics import rollout_decode, gt_ego_waypoints
from driving_diagnostic import WP_STEPS

DEV = "cuda"; SPEED_SCALE = 10.0; WINDOW = 8; K = 20
entry = dict(key="flagship-30k", arch="flagship-worldmodel",
             ckpt="/root/models/flagship-30k/ckpt.pt", speed_input=True)
h = load(entry, device=DEV)
model, step_readout = h["model"], h["step_readout"]
for p in model.parameters(): p.requires_grad_(False)
for p in h["grounding"].parameters(): p.requires_grad_(False)
model.eval()
print("loaded flagship-30k; state_dim", h["state_dim"], "step", h["step"])

eps = load_raw(list_val_episodes("/root/valdata/physicalai-val-0c5f7dac3b11", n=2))
ep = eps[0]
T = min(ep.frames.shape[0], ep.actions.shape[0], ep.poses.shape[0])
starts = list(range(0, T - WINDOW - K, 8))[:16]
last = torch.tensor([t + WINDOW - 1 for t in starts])
fw = torch.stack([ep.frames[t:t+WINDOW] for t in starts]).to(DEV).float().div_(255.)
aw = torch.stack([ep.actions[t:t+WINDOW] for t in starts]).to(DEV)
fa = torch.stack([ep.actions[t+WINDOW:t+WINDOW+K] for t in starts]).to(DEV)
poses = ep.poses.to(DEV)
v0 = poses[last, 3:4].float() / SPEED_SCALE
aw = torch.cat([aw, v0[:, None].expand(-1, aw.shape[1], -1)], -1)
fa = torch.cat([fa, v0[:, None].expand(-1, fa.shape[1], -1)], -1)

with torch.no_grad():
    states = model.encode_window(fw)
    wp_full, _ = rollout_decode(model.predictor, states, aw, fa, step_readout, K)
fut_poses = torch.stack([poses[s+WINDOW:s+WINDOW+K] for s in starts])
gt_wp = gt_ego_waypoints(poses[last], fut_poses, range(1, K+1))
wp_idx = torch.tensor([k-1 for k in WP_STEPS], device=DEV)
pred4 = wp_full.index_select(1, wp_idx); gt4 = gt_wp.index_select(1, wp_idx)
ade2s = (pred4 - gt4).norm(dim=-1).mean().item()
fde2s = (pred4[:, -1] - gt4[:, -1]).norm(dim=-1).mean().item()
print(f"GT-action rollout ADE@2s(mean4wp)={ade2s:.4f} FDE@2s={fde2s:.4f} (n={len(starts)} win, ep0)")

fa_g = fa.detach().clone().requires_grad_(True)
states2 = model.encode_window(fw)
wp_g, _ = rollout_decode(model.predictor, states2, aw, fa_g, step_readout, K)
loss = (wp_g.index_select(1, wp_idx) - gt4).norm(dim=-1).mean()
loss.backward()
g = fa_g.grad
print(f"grad->future_actions norm={g.norm().item():.4e} nonzero={float((g.abs()>0).float().mean()):.3f} "
      f"steer_accel_gradnorm={g[...,:2].norm().item():.4e}")
print("SMOKE_OK")
