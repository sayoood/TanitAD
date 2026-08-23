#!/usr/bin/env python3
"""Resolve the driver-pose frame <-> map frame transform, and validate the off-road
cost's SIGN. Offline, CPU-only. This is THE de-risking step: a frame/sign error here
gives a misleading Gate-0 verdict."""
import sys, json, glob, numpy as np
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
from alpasim_runtime.scene_loader import ArtifactSceneProvider
import shapely, shapely.ops, shapely.geometry
from trajdata.maps import vec_map_elements

SS = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/986fec83193b1baf3d5121f09462e248"
prov = ArtifactSceneProvider.from_path(SS, smooth_trajectories=True)

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
        try:
            polys.append(lane_polygon(lane))
        except Exception:
            pass
    return shapely.ops.unary_union(polys)

def offroad_cost_xy(xy, lane_union):
    """0 if inside drivable union, else distance to nearest lane boundary (>0)."""
    p = shapely.Point(float(xy[0]), float(xy[1]))
    if lane_union.contains(p):
        return 0.0
    return float(p.distance(lane_union))

for c8 in ["00169207"]:
    sid = [s for s in prov.scene_ids if s.replace("clipgt-", "")[:8] == c8][0]
    ds = prov.get_data_source(sid)
    vm = ds.map
    rig = ds.rig
    w2n = rig.world_to_nre
    print(f"\n===== scene {c8} =====")
    print("world_to_nre type:", type(w2n).__name__)
    M = np.asarray(w2n, dtype=float)
    print("world_to_nre shape:", M.shape)
    print(M)
    assert M.shape == (4, 4), f"unexpected world_to_nre shape {M.shape}"
    lane_union = build_lane_union(vm)
    print("lane_union area=%.0f bounds=%s" % (lane_union.area, [round(b,1) for b in lane_union.bounds]))

    gt_nre = np.asarray(ds.rig.trajectory.positions)  # ego-local (starts 0,0,0)
    # nre -> world = inverse(world_to_nre)
    N2W = np.linalg.inv(M)
    gt_h = np.concatenate([gt_nre, np.ones((len(gt_nre), 1))], axis=1)  # (T,4)
    gt_world = (N2W @ gt_h.T).T[:, :3]
    print("GT nre  start %s end %s" % (gt_nre[0, :2].round(1), gt_nre[-1, :2].round(1)))
    print("GT world(via N2W) start %s end %s" % (gt_world[0, :2].round(1), gt_world[-1, :2].round(1)))
    print("lane_union bounds x[%.0f,%.0f] y[%.0f,%.0f]" % (lane_union.bounds[0], lane_union.bounds[2], lane_union.bounds[1], lane_union.bounds[3]))

    # VALIDATION: GT transformed to world must be ~on-road (cost ~0); raw nre must be off; shifted must be high
    cost_world = np.array([offroad_cost_xy(p, lane_union) for p in gt_world[:, :2]])
    cost_nre = np.array([offroad_cost_xy(p, lane_union) for p in gt_nre[:, :2]])
    # lateral shift 15 m in world frame (perpendicular to heading via consecutive diff)
    d = np.diff(gt_world[:, :2], axis=0); head = np.arctan2(d[:, 1], d[:, 0]); head = np.append(head, head[-1])
    perp = np.stack([-np.sin(head), np.cos(head)], 1)
    gt_shift = gt_world[:, :2] + 15.0 * perp
    cost_shift = np.array([offroad_cost_xy(p, lane_union) for p in gt_shift])
    print("COST GT->world  : mean=%.2f max=%.2f frac_onroad(<0.5m)=%.2f" % (cost_world.mean(), cost_world.max(), (cost_world < 0.5).mean()))
    print("COST GT raw-nre : mean=%.2f max=%.2f  (expect HIGH if nre!=world)" % (cost_nre.mean(), cost_nre.max()))
    print("COST GT+15m lat : mean=%.2f min=%.2f  (expect HIGH)" % (cost_shift.mean(), cost_shift.min()))
    verdict = "N2W-ALIGNS" if cost_world.mean() < 1.0 and cost_shift.mean() > 3.0 else "NEEDS-REVIEW"
    print("FRAME VERDICT:", verdict)

# ---- session -> scene mapping for the openloop jsonl (to test driver frame) ----
print("\n===== openloop rollout session->scene =====")
import os
for root in ["/workspace/refcopenloop"]:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f.endswith("rollout.asl") or "metadata" in f.lower():
                print(os.path.join(dp, f))
print("PROBE_DONE")
