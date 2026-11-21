#!/usr/bin/env python3
"""Gate-0 PREREQUISITE probe: can the driver read drivable-area/road-boundary geometry at inference?
Loads suite USDZs via AlpaSim's own ArtifactSceneProvider (the driver runs on the same pod with the
same USDZs) and inspects data_source.map (vec_map) for road edges + lanes, plus the ego trajectory
frame (needed to align the map to the rig-frame plan)."""
import sys, glob, json
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
from alpasim_runtime.scene_loader import ArtifactSceneProvider
import numpy as np

SSDIR = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/986fec83193b1baf3d5121f09462e248"
# intersection + roundabout suite clips (the junction-failure focus)
WANT = {"00169207": "intersection", "41c06176": "intersection", "3cc29c99": "roundabout",
        "6dcd2117": "roundabout", "adb72a39": "roundabout"}

prov = ArtifactSceneProvider.from_path(SSDIR, smooth_trajectories=True)
ids = sorted(prov.scene_ids)
print("discovered", len(ids), "scenes")
out = {}
for sid in ids:
    c8 = sid.replace("clipgt-", "")[:8]
    if c8 not in WANT:
        continue
    rec = {"scene_id": sid, "category": WANT[c8]}
    try:
        ds = prov.get_data_source(sid)
        vm = ds.map
        rec["map_loaded"] = vm is not None
        if vm is not None:
            from trajdata.maps.vec_map_elements import MapElementType, RoadLane
            # elements: {MapElementType: {id: element}}
            try:
                rec["element_counts"] = {str(k): len(v) for k, v in vm.elements.items()}
            except Exception as e:
                rec["elem_err"] = repr(e)[:150]
            # ego trajectory frame (map + ego share the USDZ frame)
            traj = ds.rig.trajectory
            try:
                xy = np.asarray(traj.xy) if hasattr(traj, "xy") else np.asarray([[p.x, p.y] for p in traj.poses])
                rec["ego_traj_type"] = type(traj).__name__
                rec["ego_start_xy"] = [round(float(v), 2) for v in xy[0]]
                rec["ego_end_xy"] = [round(float(v), 2) for v in xy[-1]]
                rec["ego_path_extent_m"] = round(float(np.linalg.norm(xy[-1] - xy[0])), 1)
                q = np.array([float(xy[len(xy)//2][0]), float(xy[len(xy)//2][1]), 0.0])
            except Exception as e:
                rec["traj_err"] = repr(e)[:150]; q = np.zeros(3)
            # closest road edge to a mid-path ego point (3D query)
            try:
                edge = vm.get_closest_road_edge(q)
                arr = np.asarray(edge)
                rec["closest_road_edge_shape"] = list(arr.shape)
                d = float(np.linalg.norm(arr[..., :2].reshape(-1, 2) - q[:2], axis=-1).min())
                rec["closest_road_edge_dist_m"] = round(d, 2)
                rec["road_edge_query_OK"] = True
            except Exception as e:
                rec["road_edge_err"] = repr(e)[:200]
            # road-edge polylines directly (for the safety clamp)
            try:
                re_dict = vm.elements.get(MapElementType.ROAD_EDGE, {})
                rec["n_road_edges"] = len(re_dict)
                lane_dict = vm.elements.get(MapElementType.ROAD_LANE, {})
                rec["n_lanes"] = len(lane_dict)
            except Exception as e:
                rec["edge_dict_err"] = repr(e)[:150]
    except Exception as e:
        rec["load_err"] = repr(e)[:200]
    out[c8] = rec
    print(json.dumps(rec))
json.dump(out, open("/workspace/gate0_prereq_probe.json", "w"), indent=2)
ok = any(r.get("map_loaded") and r.get("n_road_edges", 0) > 0 and r.get("road_edge_query_OK") for r in out.values())
print("PREREQ:", "MET" if ok else "NOT-MET/UNCLEAR")
