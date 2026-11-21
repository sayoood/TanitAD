"""Does the ego EVER reach a curb in this scene? Distance is the control the event flag needs."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP, augment_routes as AR
from driverl.datatypes.data_enums import LaneType
from driverl.utils.geometry import project_points_to_segments
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
sd, _ = SP.enrich_lane_graph(sd, sc.map_api, AR._anchor_from_ego(smps[step].ego_state),
                             init.route_roadblock_ids)
gxy, gyw = G.teacher_future(smps, step)
lp, lm = sd.lanes_points, sd.lanes_points_mask
t = lp[..., 0, 2]; curb = (lm & (t == LaneType.CURB.code))[0]
s0, s1 = lp[0, curb, 0, :2], lp[0, curb, 1, :2]
print(f"curb segments: {int(curb.sum())}")
print(f"{'lat m':>7s} {'min dist ego-corner to CURB':>30s} {'centre-to-curb':>16s}")
for d in (0, 2, 4, 6, 8, 10, 12, 13, 14, 16, 18, 20, 30, 40, 60, 80):
    xy = gxy.clone(); xy[:, 1] += torch.linspace(0, float(d), xy.shape[0])
    cand = SP.inject_proposal(sd, xy, gyw)
    pn, pl, _ = cand.get_recent_agent_polygons()
    corners = pn[0, 0]                                    # [4,2]
    _, _, dc = project_points_to_segments(corners.unsqueeze(0).unsqueeze(2),
                                          s0.unsqueeze(0).unsqueeze(0),
                                          s1.unsqueeze(0).unsqueeze(0), return_dist=True)
    ctr = pn[0:1, 0:1].mean(dim=2)
    _, _, dcc = project_points_to_segments(ctr.unsqueeze(2), s0.unsqueeze(0).unsqueeze(0),
                                           s1.unsqueeze(0).unsqueeze(0), return_dist=True)
    print(f"{d:7.1f} {float(dc.min()):30.2f} {float(dcc.min()):16.2f}")
