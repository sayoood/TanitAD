import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP, augment_routes as AR, build_scorer_targets as BST
cfg, _ = G.build_engine_config()
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
gxy, gyw = BST.teacher_future(smps, step)
stop = torch.zeros_like(gxy)
TH = dict(lon_jerk=cfg.nuplan_comfort_max_abs_lon_jerk, mag_jerk=cfg.nuplan_comfort_max_abs_mag_jerk,
          lat_acc=cfg.nuplan_comfort_max_abs_lat_accel, lon_acc_max=cfg.nuplan_comfort_max_lon_accel,
          lon_acc_min=cfg.nuplan_comfort_min_lon_accel, yaw_rate=cfg.nuplan_comfort_max_abs_yaw_rate)
print("thresholds:", {k: round(float(v), 3) for k, v in TH.items()})
for tag, xy in (("teacher", gxy), ("stopped", stop)):
    c = SP.inject_proposal(sd, xy, gyw)
    T = xy.shape[0]
    al = c.agent_acceleration_state_all[0, 0, -T:]
    jl = c.agent_jerk_long_all[0, 0, -T:]
    jt = c.agent_jerk_lat_all[0, 0, -T:]
    yr = c.agent_yaw_rate_all[0, 0, -T:]
    print(f"\n{tag}:")
    print(f"  a_long  max|.| {float(al.abs().max()):8.2f}  (limits {TH['lon_acc_min']:.2f}..{TH['lon_acc_max']:.2f})")
    print(f"  j_long  max|.| {float(jl.abs().max()):8.2f}  (limit {TH['lon_jerk']:.2f})")
    print(f"  j_lat   max|.| {float(jt.abs().max()):8.2f}  (limit {TH['lat_acc']:.2f} is accel; mag jerk {TH['mag_jerk']:.2f})")
    print(f"  yaw_rate max|.| {float(yr.abs().max()):8.3f} (limit {TH['yaw_rate']:.2f})")
    print(f"  first 5 a_long: {[round(float(x),2) for x in al[:5]]}")
