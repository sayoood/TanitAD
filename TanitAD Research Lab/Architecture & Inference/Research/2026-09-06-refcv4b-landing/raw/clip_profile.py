"""Per-clip manoeuvre profile for the refcv4b reel's clip SELECTION RULE.

⛔ WHY A RULE AND NOT A PICK. The banked refcv3 reel says of itself: *"This is a
hand-picked reel and it must never be quoted as a representative one."* The only
way to lose that caveat is to state a criterion that is MEASURED, reproducible
and fixed BEFORE the clips are looked at. This file computes the criterion.

Reads ONLY poses out of each `*.v2ep.pt` (no frames, no model, no GPU) and the
v7.2 nav token out of the label blob.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/taniteval")

EPS = sys.argv[1]
LABELS = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


nav_by_clip = {}
try:
    from tanitad.data import v7_labels as v7l
    from tanitad.data.v2_dataset import stable_episode_id       # noqa: F401
    lab, man = v7l.load_v7_labels(LABELS, allow_oracle_nav=True)
    for rec in lab:
        n = getattr(rec, "nav_command", None)
        nav_by_clip[str(rec.clip_id)] = n
except Exception as e:                                          # noqa: BLE001
    print("[warn] nav tokens unavailable: %r" % (e,))

rows = []
for p in sorted(glob.glob(os.path.join(EPS, "*.v2ep.pt"))):
    cid = os.path.basename(p).replace(".v2ep.pt", "")
    d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    poses = d.get("poses")
    if poses is None:
        continue
    P = np.asarray(poses, dtype=np.float64)          # [T, 4] = x, y, yaw, v
    yaw, v = P[:, 2], P[:, 3]
    dy = np.abs(np.sum(wrap(np.diff(yaw))))          # total turned angle, rad
    net = abs(wrap(yaw[-1] - yaw[0]))
    rows.append({
        "clip_id": cid, "n_frames": int(P.shape[0]),
        "abs_turn_deg": round(float(np.degrees(dy)), 2),
        "net_yaw_deg": round(float(np.degrees(net)), 2),
        "v_min": round(float(v.min()), 3), "v_max": round(float(v.max()), 3),
        "v_mean": round(float(v.mean()), 3),
        "v_span": round(float(v.max() - v.min()), 3),
        "signed_net_yaw_deg": round(
            float(np.degrees(wrap(yaw[-1] - yaw[0]))), 2),
        "nav": nav_by_clip.get(cid),
    })

assert rows, "no clips read"                        # CONTROL
print("n_clips=%d   (CONTROL: nav tokens resolved for %d)"
      % (len(rows), sum(1 for r in rows if r["nav"] is not None)))
navs = {}
for r in rows:
    navs[str(r["nav"])] = navs.get(str(r["nav"]), 0) + 1
print("nav distribution:", navs)
print("abs_turn_deg: min %.1f  median %.1f  max %.1f"
      % (min(r["abs_turn_deg"] for r in rows),
         sorted(r["abs_turn_deg"] for r in rows)[len(rows) // 2],
         max(r["abs_turn_deg"] for r in rows)))
print("v_span m/s : min %.1f  median %.1f  max %.1f"
      % (min(r["v_span"] for r in rows),
         sorted(r["v_span"] for r in rows)[len(rows) // 2],
         max(r["v_span"] for r in rows)))
print()
print("TOP 12 by abs_turn_deg:")
for r in sorted(rows, key=lambda r: -r["abs_turn_deg"])[:12]:
    print("  %s  turn=%7.1f deg  net=%+8.1f  v %.1f..%.1f (span %.1f)  nav=%s"
          % (r["clip_id"][:8], r["abs_turn_deg"], r["signed_net_yaw_deg"],
             r["v_min"], r["v_max"], r["v_span"], r["nav"]))
print()
print("TOP 12 by v_span:")
for r in sorted(rows, key=lambda r: -r["v_span"])[:12]:
    print("  %s  turn=%7.1f deg  net=%+8.1f  v %.1f..%.1f (span %.1f)  nav=%s"
          % (r["clip_id"][:8], r["abs_turn_deg"], r["signed_net_yaw_deg"],
             r["v_min"], r["v_max"], r["v_span"], r["nav"]))

if OUT:
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"n_clips": len(rows), "episodes_dir": EPS,
                   "labels": LABELS, "clips": rows}, fh, indent=1)
    print("\nwrote %s" % OUT)
