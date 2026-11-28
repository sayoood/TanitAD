"""How well does Alpamayo's `lane` field separate TURNS from curve-following?

CONTROL: the `lane == "Turn Left"/"Turn Right"` rows are Alpamayo's OWN turn
declaration. If |dyaw| does not separate them from `Lane Keep`, the geometry is
not measuring what we think and no threshold is admissible.
"""
import json, math
from pathlib import Path
import torch

ROOT = Path("/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl")
tax = json.load(open("/home/nvidia/gtac/tax201.json"))
rows = []
for cid, t in tax.items():
    f = ROOT / f"{cid}.v2ep.pt"
    if not f.exists():
        continue
    p = torch.load(f, map_location="cpu", weights_only=False)["poses"].double()
    T = p.shape[0]; t0 = T // 2; t1 = min(t0 + 60, T - 1)
    dy = math.degrees(math.atan2(math.sin(float(p[t1,2]) - float(p[t0,2])),
                                 math.cos(float(p[t1,2]) - float(p[t0,2]))))
    arc = float((p[t0+1:t1+1,:2] - p[t0:t1,:2]).norm(dim=-1).sum())
    rows.append({"clip": cid, "lane": t.get("lane"), "lateral": t.get("lateral"),
                 "dyaw_deg": dy, "arc_m": arc, "v0": float(p[t0,3])})

def stat(sel, name):
    d = sorted(abs(r["dyaw_deg"]) for r in sel)
    if not d: return
    print(f"  {name:34s} n={len(d):4d}  |dyaw| p50={d[len(d)//2]:6.1f}  "
          f"p90={d[int(.9*len(d))]:6.1f}  max={d[-1]:6.1f}")

print(f"n with poses = {len(rows)}")
print("\n=== |dyaw| over the 2-6s band, by Alpamayo `lane` ===")
for ln in ("Turn Left", "Turn Right", "Lane Keep", "Left Lane Change",
           "Right Lane Change", "Slightly Shift Left", "Slightly Shift Right"):
    stat([r for r in rows if r["lane"] == ln], f"lane={ln}")
print("\n=== the 41% gap population: lane=Lane Keep AND lateral=Steer L/R ===")
steer = [r for r in rows if r["lane"] == "Lane Keep"
         and r["lateral"] in ("Steer Left", "Steer Right")]
stat(steer, "Lane Keep + Steer L/R")
turns = [r for r in rows if r["lane"] in ("Turn Left", "Turn Right")]
for thr in (20, 30, 45, 60):
    n_steer_over = sum(1 for r in steer if abs(r["dyaw_deg"]) >= thr)
    n_turn_over = sum(1 for r in turns if abs(r["dyaw_deg"]) >= thr)
    print(f"  |dyaw| >= {thr:3d} deg :  Steer pop {n_steer_over:3d}/{len(steer)} "
          f"({n_steer_over/max(len(steer),1)*100:5.1f}%)   |   Alpamayo-declared "
          f"turns {n_turn_over:3d}/{len(turns)} ({n_turn_over/max(len(turns),1)*100:5.1f}%)")
print("\n=== CONTROL: does |dyaw| separate declared turns from Lane Keep at all? ===")
lk = sorted(abs(r["dyaw_deg"]) for r in rows if r["lane"] == "Lane Keep")
tn = sorted(abs(r["dyaw_deg"]) for r in rows if r["lane"] in ("Turn Left","Turn Right"))
if lk and tn:
    print(f"  Lane Keep p50={lk[len(lk)//2]:.1f}  vs  declared-Turn p50={tn[len(tn)//2]:.1f}"
          f"   ratio={tn[len(tn)//2]/max(lk[len(lk)//2],1e-6):.1f}x")
    print(f"  PASS={tn[len(tn)//2] > 3*lk[len(lk)//2]}  (separation must be large "
          f"or no threshold is admissible)")
json.dump(rows, open("/home/nvidia/gtac/turn_split.json","w"), indent=1)
