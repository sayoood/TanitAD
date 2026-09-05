"""GT-ONLY census of the refav1 eval slice: how many turn windows exist, of
which direction, in how many EPISODE CLUSTERS.

PRE-REGISTRATION FIREWALL: this script reads ONLY ground-truth ego poses.
It never opens an arm record, a dump's `cl` array, or any model output, so
running it before the SPEC is committed cannot leak the outcome.
"""
import os, sys, json, math
import numpy as np
import torch

WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.models.metric_dynamics import gt_ego_waypoints
from tanitad.refs.refc_tactical import factor_from_kinematics
from taniteval import four_families as ff

EPS = "C:/Users/Admin/refav1_margin/p4/eps"
CACHE = "C:/Users/Admin/refav1_margin/p4/fp8"
W, K, K_WM, DT = 4, 10, 30, 0.2
REACH = K_WM  # loader: str_ext_steps=0 -> reach = K_loader = 30

names = sorted(os.path.splitext(os.path.basename(p))[0]
               for p in os.listdir(CACHE) if p.endswith(".pt") and p != "index.pt")
print("episodes:", len(names))

rows = []
for ei, nm in enumerate(names):
    o = torch.load(os.path.join(EPS, nm + ".v2ep.pt"), map_location="cpu",
                   weights_only=False)
    poses = o["poses"].float()                      # [T_ep, 4] x,y,yaw,v
    t_c = math.ceil(poses.shape[0] / 2)
    ts = list(range(W - 1, t_c - REACH - 1))
    G = []
    for t in ts:
        f0 = 2 * t
        fut = poses[f0 + 2: f0 + 2 * K + 1: 2]
        assert fut.shape[0] == K, (nm, t, fut.shape)
        G.append(gt_ego_waypoints(poses[f0][None], fut[None], list(range(1, K + 1))))
    G = torch.cat(G, 0)                              # [n, K, 2]
    dy, dv, v0, v1, _ = ff.maneuver_kinematics(G, DT)
    lat, lon = factor_from_kinematics(dy, dv, v0, v1)
    # GT signed curvature at t0, the crossover the programme uses (4e-2)
    Gg = ff._seq_geometry(G, DT)
    # ff's OWN signed curvature [n, K-1], never a re-derivation; invalid pairs -> 0
    kap = Gg["curvature"] * Gg["pair_valid"].float()
    k0 = kap[:, 0]
    kmax = kap.gather(1, kap.abs().argmax(1, keepdim=True)).squeeze(1)
    for j, t in enumerate(ts):
        rows.append(dict(ei=ei, name=nm, t=int(t),
                         lat=int(lat[j]), lon=int(lon[j]),
                         dyaw=float(dy[j]), dv=float(dv[j]),
                         v0=float(v0[j]),
                         k0=float(k0[j]), kmax_signed=float(kmax[j]),
                         lat_ext=float(G[j, :, 1].abs().max())))
    print("  %-14s T_ep=%4d t_c=%4d windows=%4d" % (nm[:13], poses.shape[0], t_c, len(ts)))

json.dump(rows, open(sys.argv[1], "w"))
print("TOTAL windows:", len(rows))

LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}
import collections
c = collections.Counter(LAT[r["lat"]] for r in rows)
print("\n=== GT lateral census (v1 dyaw gate, |dyaw| > 0.15 rad), ALL windows ===")
for k in ("lane_keep", "turn_left", "turn_right"):
    eps_with = sorted({r["ei"] for r in rows if LAT[r["lat"]] == k})
    print("  %-11s n=%4d  episodes=%d %s" % (k, c[k], len(eps_with), eps_with))

print("\n=== per-episode breakdown ===")
print("  ei  name           n_win  LK   TL   TR")
for ei, nm in enumerate(names):
    sub = [r for r in rows if r["ei"] == ei]
    cc = collections.Counter(LAT[r["lat"]] for r in sub)
    print("  %2d  %-13s %5d %4d %4d %4d" % (ei, nm[:13], len(sub),
          cc["lane_keep"], cc["turn_left"], cc["turn_right"]))

print("\n=== the SHIPPED stride-16 panel (the 40 windows already banked) ===")
sel = [r for r in rows if (r["t"] - (W - 1)) % 16 == 0]
cc = collections.Counter(LAT[r["lat"]] for r in sel)
print("  n=%d  LK=%d TL=%d TR=%d" % (len(sel), cc["lane_keep"], cc["turn_left"], cc["turn_right"]))
for k in ("turn_left", "turn_right"):
    e = sorted({r["ei"] for r in sel if LAT[r["lat"]] == k})
    print("    %-11s episodes=%d %s" % (k, len(e), e))
