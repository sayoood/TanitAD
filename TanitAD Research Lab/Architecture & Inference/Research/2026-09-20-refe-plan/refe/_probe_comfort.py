import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP, augment_routes as AR, build_scorer_targets as BST
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
sd, _ = SP.enrich_lane_graph(sd, sc.map_api, AR._anchor_from_ego(smps[step].ego_state),
                             init.route_roadblock_ids)
gxy, gyw = BST.teacher_future(smps, step)
stop = torch.zeros_like(gxy)
for tag, xy in (("teacher", gxy), ("stopped", stop)):
    c = SP.inject_proposal(sd, xy, gyw)
    v = c.agent_velocity_all[0, 0, -20:, :2]
    print(f"{tag:9s} ego velocity over the injected horizon: "
          f"first {v[0].tolist()}  last {v[-1].tolist()}  |v| mean {float(v.norm(dim=-1).mean()):.3f}")
