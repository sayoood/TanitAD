#!/usr/bin/env python3
"""CONFIRM: the driver-received poses are in the map (lane) frame under IDENTITY.
Uses the REAL force-GT driver-received poses (refc_openloop_preds.jsonl) vs each
scene's lane union. Force-GT => ego follows the recorded on-road path => cost ~0
proves frame+cost are correct. Plus a clean off-road point must score high."""
import sys, json, glob, os, numpy as np
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
from alpasim_runtime.scene_loader import ArtifactSceneProvider
import shapely, shapely.ops

SESS2SCENE = {
    "0a7361a0": "clipgt-000525f6-3999-4812-9924-8adff40ca514",
    "09e980ac": "clipgt-00040136-e651-4abd-991d-0655ccda9430",
    "0299069c": "clipgt-000548db-e266-49e5-a832-6674ab53a615",
    "02ddd5d8": "clipgt-00064c58-7047-4a53-8a36-b033baaaa5fb",
}

def lane_polygon(lane, road_width_m=3.7):
    if lane.left_edge is not None:
        pts = np.concatenate([lane.left_edge.points[..., :2],
                              np.flip(lane.right_edge.points[..., :2], axis=0)], axis=0)
        poly = shapely.Polygon(pts)
    else:
        poly = shapely.LineString(lane.center.points[..., :2]).buffer(road_width_m / 2)
    if not poly.is_valid:
        poly = shapely.make_valid(poly).buffer(0)
    return poly

def build_lane_union(vm):
    polys = []
    for lane in vm.lanes:
        try: polys.append(lane_polygon(lane))
        except Exception: pass
    return shapely.ops.unary_union(polys)

def cost_xy(xy, lu):
    p = shapely.Point(float(xy[0]), float(xy[1]))
    return 0.0 if lu.contains(p) else float(p.distance(lu))

# find a sceneset containing these openloop scenes
scenesets = sorted(glob.glob("/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/*"))
print("scenesets on pod:", [os.path.basename(s) for s in scenesets])

# load driver-received poses grouped by session
poses = {k: [] for k in SESS2SCENE}
for l in open("/workspace/refc_openloop_preds.jsonl"):
    d = json.loads(l); s8 = d["session"][:8]
    if s8 in poses:
        poses[s8].append((d["x"], d["y"]))

# locate each scene in whatever sceneset has it
prov_cache = {}
def find_scene(scene_id):
    for ss in scenesets:
        key = ss
        if key not in prov_cache:
            try: prov_cache[key] = ArtifactSceneProvider.from_path(ss, smooth_trajectories=True)
            except Exception: prov_cache[key] = None
        p = prov_cache[key]
        if p is not None and scene_id in set(p.scene_ids):
            return p
    return None

print("\n=== DRIVER-RECEIVED FORCE-GT POSES vs LANES (identity frame) ===")
all_onroad = []
for s8, scene_id in SESS2SCENE.items():
    prov = find_scene(scene_id)
    if prov is None:
        print(f"{s8} {scene_id[:16]}: sceneset NOT on pod (skip)"); continue
    ds = prov.get_data_source(scene_id); lu = build_lane_union(ds.map)
    pts = np.array(poses[s8])
    if not len(pts): print(f"{s8}: no poses"); continue
    costs = np.array([cost_xy(p, lu) for p in pts])
    onroad = (costs < 0.5).mean()
    all_onroad.append(onroad)
    print(f"{s8} {scene_id[:16]} n={len(pts)}: driver-pose cost mean=%.2f max=%.2f frac_onroad(<0.5m)=%.2f  lanes_bounds=%s"
          % (costs.mean(), costs.max(), onroad, [round(b) for b in lu.bounds]))
    # clean off-road point: 60 m outside the lane bounds
    off = np.array([lu.bounds[2] + 60.0, lu.bounds[3] + 60.0])
    print(f"      clean OFF-ROAD point {off.round(0)} cost=%.1f (expect >>0)" % cost_xy(off, lu))

if all_onroad:
    print("\nMEAN frac_onroad across force-GT sessions: %.2f" % np.mean(all_onroad))
    print("FRAME+COST VERDICT:", "IDENTITY-CONFIRMED" if np.mean(all_onroad) > 0.8 else "REVIEW")
print("PROBE_DONE")
