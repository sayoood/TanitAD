"""Why is REF-B bad in curves? Four discriminating tests:
 1 UNDER-ROTATION: pred vs GT net-heading-change slope (per curvature bucket)
 2 YAW-RATE DECODABILITY: ridge probe yawrate from REF-B encoder states
 3 NAV-COMMAND SENSITIVITY: sharp-turn windows follow vs correct turn command
 4 DATA BALANCE: turn-window fractions
Run on the eval pod. Uses saved windows + fresh forwards on turn windows."""
import sys
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

import numpy as np
import torch
from pathlib import Path

from taniteval import loaders, rollout as ro
from taniteval.registry import MODELS
from tanitad.data.mixing import load_episode
from driving_diagnostic import curvature_bucket

RES = Path("/root/taniteval/results")
device = "cuda"

# ---------- 1. UNDER-ROTATION from saved windows ----------
w = ro.load_windows(RES / "windows_refb.pt")
pred, gt = w["pred"], w["gt"]                     # [N,4,2]
def net_heading(way):                              # heading of last segment vs first
    d1 = way[:, 0]                                 # first waypoint dir from origin
    dl = way[:, -1] - way[:, -2]
    a1 = torch.atan2(d1[:, 1], d1[:, 0])
    al = torch.atan2(dl[:, 1], dl[:, 0])
    return torch.rad2deg(torch.atan2(torch.sin(al - a1), torch.cos(al - a1)))
h_pred, h_gt = net_heading(pred), net_heading(gt)
curv = [curvature_bucket(float(x)) for x in w["head_deg"]]
print("=" * 70)
print("1. UNDER-ROTATION (pred vs GT path heading-change, deg)")
print("=" * 70)
for lab in ["straight", "gentle", "sharp"]:
    idx = torch.tensor([i for i, l in enumerate(curv) if l == lab])
    if not len(idx):
        continue
    hg, hp = h_gt[idx], h_pred[idx]
    mask = hg.abs() > 1.0
    slope = float((hp[mask] * hg[mask]).sum() / (hg[mask] ** 2).sum()) if mask.any() else float("nan")
    print(f"  {lab:9} n={len(idx):4d}  |GT dh| median {hg.abs().median():5.1f}  "
          f"|pred dh| median {hp.abs().median():5.1f}  rotation-gain {slope:.2f} "
          f"({'UNDER-rotates' if slope < 0.85 else 'ok'})")

# ---------- 2+3 need the model + episodes ----------
e = [m for m in MODELS if m["key"] == "refb"][0]
model = loaders.load(e, device)["model"]
files = sorted(Path("/root/valdata/physicalai-val-0c5f7dac3b11").glob("ep_*.pt"))[:40]
eps = [load_episode(str(f), mmap=True) for f in files]

print("=" * 70)
print("2. YAW-RATE DECODABILITY from REF-B encoder states (ridge probe)")
print("=" * 70)
X, Y_yaw, Y_spd = [], [], []
with torch.no_grad():
    for ep in eps[:20]:
        fr, T = ep.frames, ep.frames.shape[0]
        starts = list(range(8, T - 30, 8))
        for i in range(0, len(starts), 16):
            ch = starts[i:i + 16]
            fw = torch.stack([torch.as_tensor(fr[t - 8 + 1:t + 1]) for t in ch]
                             ).to(device).float().div_(255.0)
            st = model.encode_window(fw)[:, -1]            # last state [b,S]
            X.append(st.cpu())
            lastt = torch.tensor(ch)
            yr = (ep.poses[lastt, 2] - ep.poses[lastt - 1, 2])
            yr = torch.atan2(torch.sin(yr), torch.cos(yr)) / 0.1
            Y_yaw.append(yr)
            Y_spd.append(ep.poses[lastt, 3])
X = torch.cat(X).float(); Y_yaw = torch.cat(Y_yaw).float(); Y_spd = torch.cat(Y_spd).float()
n = X.shape[0]; ntr = int(0.7 * n)
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
tr, te = perm[:ntr], perm[ntr:]
def ridge_r2(X, y, lam=10.0):
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    A = Xtr.T @ Xtr + lam * torch.eye(X.shape[1])
    wgt = torch.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
    pr = Xte @ wgt + ytr.mean()
    return 1 - ((pr - yte) ** 2).mean() / ((yte - yte.mean()) ** 2).mean()
print(f"  states->yaw-rate  R2 = {float(ridge_r2(X, Y_yaw)):.3f}   (n={n})")
print(f"  states->speed     R2 = {float(ridge_r2(X, Y_spd)):.3f}")

print("=" * 70)
print("3. NAV-COMMAND SENSITIVITY on TURN windows (follow vs correct cmd)")
print("=" * 70)
# nav ids follow refb_labels convention: 0=follow, 1=left, 2=right (3=straight?)
res = {}
with torch.no_grad():
    ades = {"follow": [], "cmd": []}
    for ep in eps:
        fr, T = ep.frames, ep.frames.shape[0]
        starts = list(range(0, T - 8 - 20, 8))
        for i in range(0, len(starts), 16):
            ch = starts[i:i + 16]
            last = torch.tensor([t + 7 for t in ch])
            hd = []
            for t in ch:
                yawd = ep.poses[t + 7 + 20, 2] - ep.poses[t + 7, 2]
                hd.append(float(torch.rad2deg(torch.atan2(
                    torch.sin(yawd), torch.cos(yawd)))))
            sel = [j for j, h in enumerate(hd) if abs(h) > 10.0]   # turning
            if not sel:
                continue
            fw = torch.stack([torch.as_tensor(fr[t:t + 8]) for t in ch]
                             ).to(device).float().div_(255.0)[sel]
            v0 = ep.poses[last, 3].to(device)[sel]
            gtw = []
            from driving_diagnostic import gt_ego_waypoints
            g = gt_ego_waypoints(ep.poses, last)[sel]
            cmds = torch.tensor([1 if hd[j] > 0 else 2 for j in sel],
                                device=device)      # left if heading grows
            for tag, nav in (("follow", None), ("cmd", cmds)):
                out = model(fw, nav_cmd=nav, v0=v0)
                wp = torch.stack([out["waypoints"][k] for k in (5, 10, 15, 20)],
                                 dim=1).cpu()
                ades[tag].append(torch.linalg.norm(wp - g, dim=-1).mean(dim=1))
    for tag in ("follow", "cmd"):
        v = torch.cat(ades[tag])
        print(f"  turn windows (|dh|>10deg, n={len(v)}): ade[{tag:6}] = {float(v.mean()):.3f} m")

print("=" * 70)
print("4. DATA BALANCE (val-window mix as proxy for train mix)")
print("=" * 70)
import collections
print("  ", dict(collections.Counter(curv)))
