"""Is the scorer target DETERMINISTIC? The reward moved on identical inputs between runs."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP, augment_routes as AR
cfg, _ = G.build_engine_config(); calcs = SP.build_calculators(cfg)
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
sd, _ = SP.enrich_lane_graph(sd, sc.map_api, AR._anchor_from_ego(smps[step].ego_state),
                             init.route_roadblock_ids)
gxy, gyw = G.teacher_future(smps, step)
xy = gxy.clone(); xy[:, 1] += torch.linspace(0, 6.0, 20)
rf = sd.randomized_features
print("randomized_features type:", type(rf).__name__)
print("  collision_reward_weight x5:",
      [round(float(torch.as_tensor(rf.get('collision_reward_weight', calculate=True)).reshape(-1)[0]), 4)
       for _ in range(5)])
print("\nsame candidate scored 5x in ONE process:")
for i in range(5):
    r = SP.score_proposal(sd, log_sd, calcs, xy, gyw)
    print(f"  {i}: OffRoad.reward {r['off_road.OffRoad.reward']:+.6f}  "
          f"CenterLine.info {r['center_line.CenterLine.info']:.6f}  "
          f"Comfort.reward {r['comfort.Comfort.reward']:.6f}")
