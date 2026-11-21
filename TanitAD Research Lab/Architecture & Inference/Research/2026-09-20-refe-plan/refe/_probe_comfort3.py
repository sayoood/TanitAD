import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP, augment_routes as AR, build_scorer_targets as BST
cfg, _ = G.build_engine_config(); calcs = SP.build_calculators(cfg)
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
gxy, gyw = BST.teacher_future(smps, step)
names = ["a_long", "a_lat", "j_long", "j_lat", "jerk_long_rw", "a_long_rw",
         "jerk_lat_rw", "a_lat_rw", "steer_rate_rw", "steer_acc_rw"]
for tag, xy in (("teacher", gxy), ("stopped", torch.zeros_like(gxy))):
    c = SP.inject_proposal(sd, xy, gyw)
    for n in ("agent_velocity_all", "agent_acceleration_state_all", "agent_steering_state_all",
              "agent_yaw_rate_all", "agent_jerk_long_all", "agent_jerk_lat_all"):
        t = getattr(c, n, None)
        if torch.is_tensor(t): print(f"  {tag:8s} {n:32s} len {t.shape[2]}")
    shared = {}
    calcs["comfort"].forward(c, log_sd, shared)
    d = shared["Comfort"]
    print(f"  {tag:8s} reward {float(d['reward'][0,0]):.4f}")
    inf = d["info"][0, 0]
    print(f"  {tag:8s} gates  " + "  ".join(f"{n}={float(v):.2f}" for n, v in zip(names, inf)))
