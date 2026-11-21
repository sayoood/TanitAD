#!/usr/bin/env python3
"""Augment the S1 decision points with per-option CENTERLINE geometry in the ego frame,
so the blind_conditioning_baseline can be made as STRONG as it honestly can be.

A weak blind baseline fails UNSAFE: it under-states acc_blind and admits a circular label.
The decisive blind feature is 'how close does the goal/route run to this option's centreline',
which needs the centreline, not a bearing summary."""
import sys, os, glob, json, collections
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
import numpy as np
from alpasim_runtime.scene_loader import ArtifactSceneProvider
from trajdata.maps.vec_map_elements import MapElementType

DP = json.load(open("/workspace/s1_decision_points.json"))
SSROOT = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets"
by_scene = collections.defaultdict(list)
for d in DP:
    by_scene[d["scene_id"]].append(d)
print("scenes to revisit:", len(by_scene))

prov_by_scene = {}
for sd in sorted(glob.glob(SSROOT + "/*")):
    if not os.path.isdir(sd):
        continue
    try:
        p = ArtifactSceneProvider.from_path(sd, smooth_trajectories=True)
    except Exception:
        continue
    for sid in p.scene_ids:
        prov_by_scene.setdefault(sid, p)

out = []
for i, (sid, ds_list) in enumerate(sorted(by_scene.items()), 1):
    prov = prov_by_scene.get(sid)
    if prov is None:
        print("NO PROVIDER", sid); continue
    ds = prov.get_data_source(sid)
    lanes = ds.map.elements[MapElementType.ROAD_LANE]
    traj = ds.rig.trajectory
    xyz = np.asarray(traj.positions, dtype=np.float64)
    q = xyz[:, :2]
    try:
        yaws = np.asarray(traj.yaws, dtype=np.float64).ravel()
    except Exception:
        yaws = np.zeros(len(q))
    for d in ds_list:
        t = d["t_dp"]
        c0, s0 = np.cos(-yaws[t]), np.sin(-yaws[t])
        def to_ego(P):
            P = np.asarray(P, dtype=np.float64) - q[t]
            return np.stack([c0 * P[:, 0] - s0 * P[:, 1], s0 * P[:, 0] + c0 * P[:, 1]], axis=1)
        cl = []
        for o in d["options"]:
            L = lanes.get(o)
            if L is None:
                cl.append([]); continue
            pts = to_ego(L.center.points[:, :2])
            # resample to <=24 points for compactness
            if len(pts) > 24:
                idx = np.linspace(0, len(pts) - 1, 24).astype(int)
                pts = pts[idx]
            cl.append([[round(float(x), 2), round(float(y), 2)] for x, y in pts])
        d["option_centerlines_egoframe"] = cl
        # DENSE realised future in the ego frame (every frame, not every 5th)
        fut = to_ego(q[t:])
        d["future_dense_egoframe"] = [[round(float(x), 2), round(float(y), 2)] for x, y in fut[:400]]
        out.append(d)
    try:
        ds.clear_cache()
    except Exception:
        pass
    print("[%2d/%2d] %s  dps=%d" % (i, len(by_scene), sid[8:20], len(ds_list)), flush=True)

json.dump(out, open("/workspace/s1_decision_points_aug.json", "w"), indent=1)
print("WROTE /workspace/s1_decision_points_aug.json  n=", len(out))
