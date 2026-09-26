"""Is `goal_reaching` BROKEN, or correctly silent? The distinction decides whether to fix it."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
gp = sd.goal_positions
print(f"goal_positions {tuple(gp.shape)}  nonzero {int((gp!=0).sum())}")
ego_goals = gp[0, 0, -1, :].reshape(-1, 2)
print(f"  ego goal points (ego frame): {[[round(float(x),2) for x in p] for p in ego_goals]}")
print(f"  distance from ego origin   : {[round(float(p.norm()),2) for p in ego_goals]} m")
rf = sd.randomized_features
for k in ("goal_reaching_threshold", "goal_reaching_weight", "goal_reaching_distance_weight",
          "survival_reward_numerator"):
    v = rf.get(k, calculate=True)
    print(f"  {k:32s} {float(torch.as_tensor(v).reshape(-1)[0]):.4f}")
gxy, gyw = G.teacher_future(smps, step)
print(f"\n  teacher 4 s endpoint      : {[round(float(x),2) for x in gxy[-1]]}  "
      f"|{float(gxy[-1].norm()):.2f}| m")
print("  => a 0.2 s STEP covers ~1-2 m; the nearest goal is the distance above.")
print("     GoalReaching fires only when the step's swept segment passes within the threshold,")
print("     so 0.0000 on a 4 s proposal is EXPECTED, not a broken calculator.")
# the discriminating control: put a candidate DIRECTLY on the goal point
cfg, _ = G.build_engine_config(); calcs = SP.build_calculators(cfg)
tgt = ego_goals[0]
xy = torch.stack([tgt * f for f in torch.linspace(1.0/20, 1.0, 20)], 0)
r = SP.score_proposal_rollout(sd, log_sd, calcs, xy, torch.zeros(20))
rt = SP.score_proposal_rollout(sd, log_sd, calcs, gxy, gyw)
print(f"\n  CONTROL -- a candidate driven ONTO the nearest goal point {[round(float(x),1) for x in tgt]}:")
for k in ("goal_reaching.GoalReaching.info", "goal_reaching.GoalReaching.reward",
          "goal_reaching.GoalReaching.stage_reached"):
    print(f"    {k:44s} teacher {rt.get(k, float('nan')):8.4f}   onto-goal {r.get(k, float('nan')):8.4f}")
