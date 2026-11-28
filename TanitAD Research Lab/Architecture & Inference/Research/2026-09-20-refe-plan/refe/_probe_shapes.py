import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
for n in ("agent_positions_all", "agent_velocity_all", "agent_acceleration_state_all",
          "agent_jerk_lat_all", "agent_jerk_long_all", "agent_steering_state_all",
          "agent_yaw_rate_all", "agent_orientation_all"):
    v = getattr(sd, n, None)
    print(f"  {n:32s} {str(tuple(v.shape)) if torch.is_tensor(v) else type(v).__name__:20s} "
          f"{'numel '+str(v.numel()) if torch.is_tensor(v) else ''}")
