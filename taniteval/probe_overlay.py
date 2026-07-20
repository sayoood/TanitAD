import sys
sys.path.insert(0, "/root/taniteval"); sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
import torch
from pathlib import Path
from taniteval import loaders, data
from taniteval.registry import MODELS
from taniteval.rollout import append_ego
from tanitad.data.mixing import load_episode
from tanitad.models.metric_dynamics import rollout_decode
from tanitad.refs.refb import MANEUVER_CLASSES, ROUTE_CLASSES
from taniteval.cam_overlay import ego_future_path
device="cuda"; WINDOW=8; K=20; SPEED_SCALE=10.0
print("MAN:",MANEUVER_CLASSES); print("ROUTE:",ROUTE_CLASSES)

def probe(key, corpus_root, use_feats, feat_kind, speed_input, dyn_input):
    entry=[m for m in MODELS if m["key"]==key][0]
    L=loaders.load(entry, device); model, sr = L["model"], L["step_readout"]
    print(f"\n== {key} step={L['step']} feed={L['feed']} "
          f"strat={model.strategic_policy is not None} tac={model.tactical_policy is not None}")
    files=sorted(Path(corpus_root).glob("ep_*.pt"))
    ep=load_episode(str(files[0]), mmap=True)
    poses, actions = ep.poses.float(), ep.actions.float()
    if use_feats:
        feats = data.load_features([files[0]], feat_kind, device, verbose=False)[0].feats
        enc = feats
    else:
        enc = ep.frames
    # one window at t0
    s=0; last=torch.tensor([s+WINDOW-1])
    if use_feats:
        fw=torch.as_tensor(enc[s:s+WINDOW])[None].to(device).float()
    else:
        fw=torch.as_tensor(enc[s:s+WINDOW])[None].to(device).float().div_(255.0)
    aw=actions[s:s+WINDOW][None].to(device)
    fa=actions[s+WINDOW:s+WINDOW+K][None].to(device)
    aw,fa=append_ego(aw,fa,poses,last,speed_input,False,dyn_input,device)
    states=model.encode_window(fw)
    print("  states:", tuple(states.shape))
    wp_full,_=rollout_decode(model.predictor, states, aw, fa, sr, K)
    print("  wp_full:", tuple(wp_full.shape))
    gt=ego_future_path(poses, int(last), K)
    idx=torch.tensor([4,9,14,19])
    ade=float(torch.linalg.norm(wp_full[0].cpu()[idx]-gt[idx],dim=-1).mean())
    print("  ADE@window:", round(ade,3), "v0:", round(float(poses[last,3]),2))
    follow=torch.zeros(1,dtype=torch.long,device=device)
    sf=model.strategic_policy(states, follow)
    route=int(sf["route_logits"].argmax(-1)); 
    tacf=model.tactical_policy(states, sf["ctx"])
    man=int(tacf["maneuver_logits"].argmax(-1))
    print("  route:", ROUTE_CLASSES[route], "maneuver:", MANEUVER_CLASSES[man])

probe("flagship-30k", "/root/valdata/physicalai-val-0c5f7dac3b11", False, None, True, False)
probe("refa-dynin-30k", "/root/valdata/physicalai-val-0c5f7dac3b11", True, "dinov2", True, True)
print("\nPROBE_OK")
