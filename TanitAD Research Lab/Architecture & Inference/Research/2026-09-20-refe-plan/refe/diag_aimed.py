"""ANALYTIC TARGET: drive straight through the nearest curb. The answer is known by construction.

A deliberate-regression arm that does not actually violate the thing being detected proves nothing.
MEASURED: the '25 m sideways veer' moves AWAY from the curb in this scene (10.66 m -> 19.06 m), so
OffRoad staying silent was CORRECT and my control was the defect."""
import sys, math, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP, augment_routes as AR
from driverl.datatypes.data_enums import LaneType
from driverl.utils.geometry import project_points_to_segments
cfg, _ = G.build_engine_config(); calcs = SP.build_calculators(cfg)
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
sd, _ = SP.enrich_lane_graph(sd, sc.map_api, AR._anchor_from_ego(smps[step].ego_state),
                             init.route_roadblock_ids)
gxy, gyw = G.teacher_future(smps, step)
lp, lm = sd.lanes_points, sd.lanes_points_mask
t = lp[..., 0, 2]; curb = (lm & (t == LaneType.CURB.code))[0]
s0, s1 = lp[0, curb, 0, :2], lp[0, curb, 1, :2]
ego0 = torch.zeros(2)
_, _, d = project_points_to_segments(ego0.view(1, 1, 1, 2), s0.view(1, 1, -1, 2),
                                     s1.view(1, 1, -1, 2), return_dist=True)
j = int(d.reshape(-1).argmin()); mid = (s0[j] + s1[j]) / 2
print(f"nearest curb segment #{j}: {s0[j].tolist()} -> {s1[j].tolist()}")
print(f"  midpoint {mid.tolist()}  distance from ego origin {float(d.reshape(-1)[j]):.2f} m")
W = ("off_road.OffRoad.info", "off_road.OffRoad.reward", "off_road.CrossLane.info",
     "center_line.CenterLine.info")
print(f"\n{'candidate':44s} " + " ".join(f"{w.split('.')[-1][:12]:>13s}" for w in W))
def show(tag, xy):
    e = SP.score_proposal(sd, log_sd, calcs, xy, gyw)
    r = SP.score_proposal_rollout(sd, log_sd, calcs, xy, gyw)
    print(f"{tag:44s} " + " ".join(f"{e.get(w, float('nan')):13.4f}" for w in W)
          + "  |  " + " ".join(f"{r.get(w, float('nan')):13.4f}" for w in W))
show("teacher path (must be CLEAN)", gxy)
for over in (0.0, 3.0, 8.0, 15.0):
    tgt = mid * (1 + over / max(float(mid.norm()), 1e-6))
    xy = torch.stack([tgt * f for f in torch.linspace(1.0 / 20, 1.0, 20)], 0)
    yaw = torch.full((20,), math.atan2(float(tgt[1]), float(tgt[0])))
    _, _, dd = project_points_to_segments(xy.view(1, -1, 1, 2), s0.view(1, 1, -1, 2),
                                          s1.view(1, 1, -1, 2), return_dist=True)
    show(f"AIMED through curb, {over:g} m past it (min d {float(dd.min()):.2f})", xy)
