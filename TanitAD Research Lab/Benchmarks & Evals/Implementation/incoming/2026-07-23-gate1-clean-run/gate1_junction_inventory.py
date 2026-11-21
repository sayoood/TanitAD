"""P0 REAL-FOOTAGE JUNCTION INVENTORY — CPU only, no GPU.
Counts junction windows per episode in the 40-ep clean val, using the SAME
net_heading_change definition the low-OOD harness uses (driving_diagnostic).
Junction = |net_heading_change over the 2s window| >= threshold.
"""
import sys, json, glob
sys.path.insert(0, "/root/TanitAD/stack/scripts")
sys.path.insert(0, "/root/TanitAD/stack")
import numpy as np
import torch
from tanitad.data.mixing import load_episode
from driving_diagnostic import net_heading_change_deg

W = 8; K = 20; STRIDE = 8
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
eps = sorted(glob.glob(VAL + "/ep_*.pt"))
print(f"n_episodes={len(eps)}")

thresholds = [10.0, 20.0, 30.0]
per_ep = []
tot_windows = 0
tot_junc = {t: 0 for t in thresholds}
for p in eps:
    ep = load_episode(p, mmap=True)
    poses = ep.poses.float()
    T = poses.shape[0]
    starts = list(range(0, T - W - K, STRIDE))
    if not starts:
        per_ep.append({"ep": p.split("/")[-1], "T": int(T), "n_win": 0}); continue
    last = torch.tensor([s + W - 1 for s in starts])
    hd = net_heading_change_deg(poses, last)
    hd = hd.numpy() if torch.is_tensor(hd) else np.asarray(hd)
    ahd = np.abs(hd)
    n_win = len(starts); tot_windows += n_win
    row = {"ep": p.split("/")[-1], "T": int(T), "n_win": n_win,
           "max_abs_hd_deg": round(float(ahd.max()), 1)}
    for t in thresholds:
        c = int((ahd >= t).sum()); row[f"junc_ge{int(t)}"] = c; tot_junc[t] += c
    per_ep.append(row)

# episode-level inventory: an episode "is a junction episode" if it has >=1 junction window
summary = {"n_episodes": len(eps), "total_windows": tot_windows}
for t in thresholds:
    ti = int(t)
    ep_with = [r for r in per_ep if r.get(f"junc_ge{ti}", 0) > 0]
    ep_with5 = [r for r in per_ep if r.get(f"junc_ge{ti}", 0) >= 5]
    summary[f"ge{ti}"] = {
        "total_junc_windows": tot_junc[t],
        "n_episodes_with_any": len(ep_with),
        "n_episodes_with_ge5": len(ep_with5),
        "junc_windows_per_ep_median": float(np.median([r[f"junc_ge{ti}"] for r in ep_with])) if ep_with else 0,
    }
print(json.dumps(summary, indent=2))
print("=== per-episode junction-window counts (ge10 / ge20 / ge30), sorted by ge10 desc ===")
for r in sorted(per_ep, key=lambda x: -x.get("junc_ge10", 0)):
    print(f"{r['ep']:>16}  T={r['T']:>4}  win={r['n_win']:>3}  "
          f"ge10={r.get('junc_ge10',0):>3} ge20={r.get('junc_ge20',0):>3} "
          f"ge30={r.get('junc_ge30',0):>3}  maxHD={r.get('max_abs_hd_deg',0)}")
json.dump({"summary": summary, "per_ep": per_ep},
          open("/workspace/gate1_junction_inventory.json", "w"), indent=2)
print("WROTE /workspace/gate1_junction_inventory.json")
print("INVENTORY_DONE")
