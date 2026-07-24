"""Probe: confirm flagship-30k runtime specifics for the DAgger fine-tune design.
Prints state_dim, tactical_policy param names/count, predictor output type, and a
recovery-expert geometry self-check. No training; read-only."""
import sys, time
LOCAL = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
sys.path.insert(0, LOCAL + r"\stack")
sys.path.insert(0, LOCAL + r"\stack\scripts")
sys.path.insert(0, LOCAL + r"\taniteval")
import torch
from taniteval import closedloop as cl
from taniteval import data, loaders
from taniteval.registry import MODELS

SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
CKPT = SP + r"\ckpt\ckpt.pt"
VALDIR = SP + r"\valsub"
dev = "cuda"

entry = dict([m for m in MODELS if m["key"] == "flagship-30k"][0])
entry["ckpt"] = CKPT
t0 = time.time()
L = loaders.load(entry, dev)
model = L["model"]
print(f"[probe] loaded step={L['step']} state_dim={L['state_dim']} "
      f"traj_capable={L['traj_capable']} ({time.time()-t0:.1f}s)")
tp = model.tactical_policy
sp = model.strategic_policy
n_tac = sum(p.numel() for p in tp.parameters())
n_str = sum(p.numel() for p in sp.parameters())
print(f"[probe] tactical_policy params={n_tac:,}  strategic params={n_str:,}")
print(f"[probe] tactical anchor_tactical={tp.anchor_tactical} "
      f"wp_heads={list(tp.wp_heads.keys()) if tp.wp_heads else None}")
print(f"[probe] waypoint_horizons={tp.cfg.waypoint_horizons} window={tp.window}")

# one real window through the plan path
files = data.list_val_episodes(VALDIR, 2)
eps = data.load_frames(files)
ep = eps[0]
W = cl.WINDOW
fw = torch.as_tensor(ep.feats[0:W]).float().div(255.0)[None].to(dev)  # [1,W,9,256,256]
states0 = model.encode_window(fw)
print(f"[probe] states0 {tuple(states0.shape)} dtype={states0.dtype}")
nav = torch.zeros(1, dtype=torch.long, device=dev)
strat = sp(states0, nav)
tac = tp(states0, strat["ctx"])
print(f"[probe] ctx {tuple(strat['ctx'].shape)} keys(tac)={list(tac.keys())}")
wp = tac["waypoints"]
print(f"[probe] waypoints keys={list(wp.keys())} wp[5]={tuple(wp[5].shape)}")
# predictor output
spd = bool(entry.get("speed_input"))
aw = ep.actions[0:W].float()[None].to(dev)
v0 = ep.poses[W-1, 3].float().reshape(1).to(dev)
if spd:
    aw = torch.cat([aw, (v0/cl.SPEED_SCALE)[:, None, None].expand(-1, W, -1)], -1)
pr = model.predictor(states0, aw)
print(f"[probe] predictor return type={type(pr).__name__} "
      f"keys/len={list(pr.keys()) if isinstance(pr, dict) else len(pr)}")
z1 = pr[1]
print(f"[probe] predictor[1] {tuple(z1.shape)}")
sr = L["step_readout"](states0[:, -1], z1)
print(f"[probe] step_readout out {tuple(sr.shape)}")

# recovery-expert geometry self-check: at Q=origin, recovery target == gt_ego_waypoints
from driving_diagnostic import gt_ego_waypoints, _ego, WP_STEPS
last = torch.tensor([W-1])
gtwp = gt_ego_waypoints(ep.poses.float(), last)  # [1,4,2] frame-0 ego
p0 = ep.poses[W-1, :2].float(); yaw0 = ep.poses[W-1, 2].float()
# frame-0 ego positions of the logged path at abs ticks
G = torch.stack([_ego(ep.poses[W-1+k, :2].float()-p0, yaw0) for k in range(0, 2*cl.K_MAX+1)])
# recovery at Q=origin (qx,qy,qyaw=0): expert_wp[h]=_ego(G[h]-0, 0)=G[h]
rec0 = torch.stack([_ego(G[k]-torch.zeros(2), torch.tensor(0.0)) for k in WP_STEPS])
err = (rec0 - gtwp[0]).abs().max().item()
print(f"[probe] recovery-expert@origin vs gt_ego_waypoints max|err|={err:.6f} (should ~0)")
print("[probe] G shape", tuple(G.shape), "-> abs ticks 0..", 2*cl.K_MAX)
print("PROBE_DONE")
