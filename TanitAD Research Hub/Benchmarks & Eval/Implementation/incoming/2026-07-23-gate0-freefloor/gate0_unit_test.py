#!/usr/bin/env python3
"""Unit-test the PRODUCTION floor selection (refc_floor_driver._select + SceneCostMap)
on known on-road/off-road anchors for an intersection scene. Banks gate0_cost_validation.json.
CPU-only: _select never touches the model, so policy=None."""
import sys, json, numpy as np
sys.path.insert(0, "/workspace")                                   # refc_floor_driver
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
import importlib.util
spec = importlib.util.spec_from_file_location("rfd", "/workspace/refc_floor_driver.py")
rfd = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(rfd)
except Exception as e:
    print("IMPORT of refc_floor_driver failed:", repr(e)); raise

from alpasim_runtime.scene_loader import ArtifactSceneProvider
SS = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/986fec83193b1baf3d5121f09462e248"
prov = ArtifactSceneProvider.from_path(SS, smooth_trajectories=True)
sid = [s for s in prov.scene_ids if s.replace("clipgt-", "")[:8] == "00169207"][0]
ds = prov.get_data_source(sid)
cm = rfd.SceneCostMap(ds)
print("costmap: lanes=%d agents=%d union_area=%.0f" % (cm.n_lanes, len(cm.agents), cm.lane_union.area))

# a real on-road ego pose (GT start is on-road, cost 0, validated)
gt = np.asarray(ds.rig.trajectory.positions)
i0 = 5
cx, cy = float(gt[i0, 0]), float(gt[i0, 1])
d = gt[i0 + 1, :2] - gt[i0, :2]
yaw = float(np.arctan2(d[1], d[0]))
print("ego on-road pose: (%.1f,%.1f) yaw=%.2f" % (cx, cy, yaw))

# build 2 anchors in RIG frame: [0]=on-road straight-ahead, [1]=veer 25 m left (off-road)
onroad = np.array([[3, 0], [6, 0], [9, 0], [12, 0]], float)
offroad = np.array([[3, 8], [6, 16], [9, 24], [12, 30]], float)     # hard left -> off the road
anchors = np.stack([onroad, offroad])                                # [2,4,2]
# give the OFF-ROAD anchor the HIGHER confidence, so only the floor can reject it
conf = np.array([0.0, 12.0])

def run(floor, lam=5.0, clamp=0.75, mu=0.0):
    drv = rfd.RefCFloorDriver(policy=None, floor=floor, lam=lam, mu=mu, clamp_m=clamp,
                              coll_r=2.5, sceneset_dir=SS, floor_log=None)
    s = {"costmap": cm, "scene_id": sid}
    sel, diag = drv._select(s, anchors, conf, cx, cy, yaw, 0.0)
    return sel, diag

sel_off, d_off = run(floor=False)
sel_on, d_on = run(floor=True)
print("\nfloor OFF -> sel=%d (expect 1, the high-conf off-road)  diag=%s" % (sel_off, d_off))
print("floor ON  -> sel=%d (expect 0, the on-road)             diag=%s" % (sel_on, d_on))

# off-road cost of each anchor's world waypoints (for the record)
c, sn = np.cos(yaw), np.sin(yaw)
def wcost(a):
    wx = cx + c * a[:, 0] - sn * a[:, 1]; wy = cy + sn * a[:, 0] + c * a[:, 1]
    return cm.offroad_cost(wx, wy)
print("on-road anchor per-wp cost:", wcost(onroad).round(2))
print("off-road anchor per-wp cost:", wcost(offroad).round(2))

ok_off = (sel_off == 1)
ok_on = (sel_on == 0)
ok_clamp_or_soft = ok_on   # floor rejected the higher-conf off-road plan
verdict = "PASS" if (ok_off and ok_on) else "FAIL"
print("\nUNIT-TEST VERDICT:", verdict)

# ---- assemble gate0_cost_validation.json (all MEASURED evidence) ----
val = {
  "purpose": "Gate-0 off-road cost geometry validation (frame + sign) before the suite run",
  "evidence_class": "MEASURED (tanitad-eval, offline CPU)",
  "frame_resolution": {
    "finding": "driver-received current.pose is in the MAP (lane) frame under IDENTITY; ds.rig.world_to_nre (translation only) must NOT be applied",
    "gt_raw_ego_local_onroad_cost_mean_m": 0.00,
    "gt_after_world_to_nre_offroad_cost_mean_m": 21.53,
    "note": "the lane union spans the ego path region; first-50-lane sampling earlier gave a false 192 m gap"
  },
  "force_gt_driver_pose_check": {
    "method": "real force-GT driver-received poses (refc_openloop_preds.jsonl) vs each scene lane-union, identity",
    "sessions": {
      "0a7361a0/clipgt-000525f6": {"n": 75, "cost_mean_m": 0.00, "cost_max_m": 0.07, "frac_onroad_lt0p5m": 1.00, "clean_offroad_pt_cost_m": 202.3},
      "09e980ac/clipgt-00040136": {"n": 75, "cost_mean_m": 0.00, "cost_max_m": 0.00, "frac_onroad_lt0p5m": 1.00, "clean_offroad_pt_cost_m": 132.7},
      "0299069c/clipgt-000548db": {"n": 75, "cost_mean_m": 0.00, "cost_max_m": 0.12, "frac_onroad_lt0p5m": 1.00, "clean_offroad_pt_cost_m": 100.8},
      "02ddd5d8/clipgt-00064c58": {"n": 75, "cost_mean_m": 0.00, "cost_max_m": 0.00, "frac_onroad_lt0p5m": 1.00, "clean_offroad_pt_cost_m": 89.6}
    },
    "mean_frac_onroad": 1.00,
    "verdict": "IDENTITY-CONFIRMED"
  },
  "alpasim_metric_alignment": {
    "lane_polygon": "verbatim port of eval/scorers/offroad.py:_get_lane_polygon (left_edge ++ flip(right_edge))",
    "drivable_surface": "shapely.ops.unary_union over all lanes; off-road cost = distance-to-union (0 inside)"
  },
  "production_select_unit_test": {
    "scene": sid, "ego_pose": [round(cx,1), round(cy,1), round(yaw,2)],
    "onroad_anchor_cost_m": [round(float(x),2) for x in wcost(onroad)],
    "offroad_anchor_cost_m": [round(float(x),2) for x in wcost(offroad)],
    "conf": [0.0, 12.0],
    "floor_off_selected": int(sel_off), "floor_off_expected": 1,
    "floor_on_selected": int(sel_on), "floor_on_expected": 0,
    "floor_on_diag": {k: (round(v,3) if isinstance(v,float) else v) for k,v in d_on.items()},
    "verdict": verdict
  },
  "overall_verdict": ("COST TRUSTWORTHY - PROCEED TO SUITE" if verdict == "PASS"
                      else "COST NOT TRUSTWORTHY - DO NOT RUN SUITE")
}
json.dump(val, open("/workspace/gate0_cost_validation.json", "w"), indent=2)
print("\nwrote /workspace/gate0_cost_validation.json ; overall:", val["overall_verdict"])
