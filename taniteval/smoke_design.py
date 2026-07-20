"""Design smoke: validate the floor/ceiling/skill collector on real data (2 eps)."""
import sys, math
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
import torch
from driving_diagnostic import (WP_STEPS, baseline_waypoints, gt_ego_waypoints,
                                net_heading_change_deg, scalar_metrics, _ego, _wrap)
from tanitad.models.readout import RidgeProbe
from tanitad.models.metric_dynamics import rollout_decode
from taniteval import loaders, data
from taniteval.registry import MODELS

K_MAX = max(WP_STEPS)
SPEED_SCALE = 10.0
EGO_HIST = 4

def ego_status_feats(poses, last):
    """Perception-free ego-status vector per window (AD-MLP repro input)."""
    feats = []
    for j in range(EGO_HIST):
        d = poses[last - j, :2] - poses[last - j - 1, :2]
        feats.append(_ego(d, poses[last, 2]))          # [b,2] ego-frame disp
    disp = torch.cat(feats, dim=-1)                    # [b, 2*EGO_HIST]
    v0 = (poses[last, :2] - poses[last - 1, :2]).norm(dim=-1, keepdim=True)
    vm1 = (poses[last - 1, :2] - poses[last - 2, :2]).norm(dim=-1, keepdim=True)
    a0 = v0 - vm1
    om0 = _wrap(poses[last, 2] - poses[last - 1, 2]).unsqueeze(-1)
    om1 = _wrap(poses[last - 1, 2] - poses[last - 2, 2]).unsqueeze(-1)
    al0 = om0 - om1
    return torch.cat([disp, v0, a0, om0, al0], dim=-1)  # [b, 2*EGO_HIST+4]

e = [m for m in MODELS if m["key"] == "flagship-speed"][0]
L = loaders.load(e, "cuda")
print("loaded", e["key"], "feed", L["feed"], "traj", L["traj_capable"], "step", L["step"])
files = data.list_val_episodes("/root/valdata/physicalai-val-0c5f7dac3b11", 3)
eps = data.load_frames(files)
print("eps", len(eps), "T", eps[0].feats.shape, "poses", eps[0].poses.shape)

# combined single-pass collect (mirror rollout.collect + aux)
window, stride, batch = 8, 8, 8
wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
device = "cuda"
model, step_readout = L["model"], L["step_readout"]
PRED, GT, BASE, EGO, ST, SPD, HDG, EID = [], [], {n: [] for n in ("constant_velocity","go_straight","constant_yaw_rate")}, [], [], [], [], []
with torch.no_grad():
    for ep in eps:
        feats = ep.feats; T = feats.shape[0]
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i+batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(feats[t:t+window]) for t in ch]).to(device)
            if fw.dtype == torch.uint8: fw = fw.float().div_(255.0)
            elif fw.dtype == torch.float16: fw = fw.float()
            aw = torch.stack([ep.actions[t:t+window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t+window:t+window+K_MAX] for t in ch]).to(device)
            if e.get("speed_input"):
                v0 = (ep.poses[last, 3:4] / SPEED_SCALE).to(device)
                aw = torch.cat([aw, v0.unsqueeze(1).expand(-1, aw.shape[1], -1)], dim=-1)
                fa = torch.cat([fa, v0.unsqueeze(1).expand(-1, fa.shape[1], -1)], dim=-1)
            states = model.encode_window(fw)
            wp_full, _ = rollout_decode(model.predictor, states, aw, fa, step_readout, K_MAX)
            PRED.append(wp_full.index_select(1, wp_idx.to(device)).cpu().float())
            GT.append(gt_ego_waypoints(ep.poses, last))
            bp = baseline_waypoints(ep.poses, last)
            for n in BASE: BASE[n].append(bp[n])
            EGO.append(ego_status_feats(ep.poses, last))
            ST.append(states[:, -1].cpu().float())
            SPD.append(ep.poses[last, 3]); HDG.append(net_heading_change_deg(ep.poses, last))
            EID.extend([ep.episode_id]*len(ch))

pred = torch.cat(PRED); gt = torch.cat(GT).float()
base = {n: torch.cat(v).float() for n, v in BASE.items()}
ego = torch.cat(EGO).float(); st = torch.cat(ST).float()
print("N windows", pred.shape[0], "state_dim", st.shape[1], "ego_dim", ego.shape[1])

def de(a, b): return torch.linalg.norm(a - b, dim=-1)   # [N,4]
model_ade = float(de(pred, gt).mean())
print("model ade_0_2s", round(model_ade, 4))
for n in base:
    print(f"  baseline {n:20s} ade", round(float(de(base[n], gt).mean()), 4))
# best-of-3 floor: per-window min across baselines
stack_de = torch.stack([de(base[n], gt) for n in base])   # [3,N,4]
floor_de = stack_de.min(0).values                          # [N,4]
print("floor(best-of-3) ade", round(float(floor_de.mean()), 4))
# ridge ego-status ceiling (in-sample sanity)
gt_flat = gt.reshape(gt.shape[0], -1)
pr_e = RidgeProbe(alpha=10.0).fit(ego, gt_flat)
ceil_e = pr_e.predict(ego).reshape(-1, 4, 2)
print("ego-status ridge ceiling ade", round(float(de(ceil_e, gt).mean()), 4), "r2", round(pr_e.r2(ego, gt_flat), 3))
pr_l = RidgeProbe(alpha=10.0).fit(st, gt_flat)
ceil_l = pr_l.predict(st).reshape(-1, 4, 2)
print("latent ridge ceiling ade", round(float(de(ceil_l, gt).mean()), 4), "r2", round(pr_l.r2(st, gt_flat), 3))
print("SMOKE_OK")
