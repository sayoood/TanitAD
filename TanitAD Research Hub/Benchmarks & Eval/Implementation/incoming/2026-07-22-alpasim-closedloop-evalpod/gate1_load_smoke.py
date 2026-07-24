#!/usr/bin/env python3
"""Verify base + FT ckpts load STRICT via the driver's load_frozen, and that the
FT planner's selected traj moved toward the recovery target vs baseline. Mirrors
the driver's model(win, nav, v0, steps=2) eval path."""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/root/TanitAD/stack/scripts")
from refc_v12_cache import load_frozen
from tanitad.data.comma2k19 import stack_frames

dev = "cuda"
base, cfg, bs = load_frozen("/root/models/refc-base-30k/ckpt.pt", "base", None, dev)
ft, _, fs = load_frozen("/root/models/refc-gate1-ft/ckpt.pt", "base", None, dev)
print(f"LOAD OK: base step={bs}  ft step={fs} (both STRICT)")

b = torch.load("/workspace/gate1_ft_data/c3d4065e.pt", weights_only=False)  # a clean roundabout departure
canon = b["canon_u8"]
errs_b, errs_f = [], []
for st in b["steps"]:
    win = stack_frames(canon[st["k"]-10:st["k"]], 3)[None].to(dev).float().div_(255.0)
    v0 = torch.tensor([st["v0"]], device=dev); nav = torch.tensor([st["nav"]], device=dev, dtype=torch.long)
    tgt = torch.tensor(st["tgt"], device=dev)
    with torch.no_grad():
        tb = base(win, nav_cmd=nav, v0=v0, steps=2)["traj"][0]
        tf = ft(win,   nav_cmd=nav, v0=v0, steps=2)["traj"][0]
    errs_b.append(float((tb-tgt).abs().mean())); errs_f.append(float((tf-tgt).abs().mean()))
print(f"scene c3d4065e: base plan->recovery L1 = {np.mean(errs_b):.3f}  ft = {np.mean(errs_f):.3f}")
# show one late step (where baseline departs far)
st = b["steps"][-6]
win = stack_frames(canon[st["k"]-10:st["k"]], 3)[None].to(dev).float().div_(255.0)
with torch.no_grad():
    tb = base(win, nav_cmd=torch.tensor([st["nav"]],device=dev,dtype=torch.long), v0=torch.tensor([st["v0"]],device=dev), steps=2)["traj"][0].cpu().numpy()
    tf = ft(win,   nav_cmd=torch.tensor([st["nav"]],device=dev,dtype=torch.long), v0=torch.tensor([st["v0"]],device=dev), steps=2)["traj"][0].cpu().numpy()
print("late step recovery tgt (fwd,left):", [[round(x,2) for x in p] for p in st["tgt"]])
print("  base plan:", [[round(float(x),2) for x in p] for p in tb])
print("  ft   plan:", [[round(float(x),2) for x in p] for p in tf])
