"""UNBIASED steady-state split of a CPU-teacher rollout: plain perf_counter wrappers, no cProfile.

cProfile inflates call-heavy Python (the map-step lever looked like ~1.4x under it and measured
1.11x on the loop), so levers are ranked here from wall-clock timers around a handful of stages.
argv: n_warm n_timed [fast 0|1]
"""
import json
import os
import sys
import time
from collections import defaultdict

SP = os.path.dirname(os.path.abspath(__file__))
PKG = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
N_WARM, N_TIME = int(sys.argv[1]), int(sys.argv[2])
FAST = sys.argv[3] if len(sys.argv) > 3 else "1"
env = json.load(open("D:/Projects/TanitAD/data/refe_navtrain10/devbox_logs/env10.json"))
os.environ.update(env)
os.environ["REFE_MAP_STEP_FAST"] = FAST
sys.path.insert(0, os.path.join(PKG, "refe"))
sys.path.insert(0, os.path.join(PKG, "code"))
os.chdir(os.path.join(PKG, "refe"))
import torch  # noqa: E402
torch.set_num_threads(1)
import build_teacher_rollouts as BR  # noqa: E402
import navtrain_scenarios as NS  # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization  # noqa: E402

T = defaultdict(float)
C = defaultdict(int)


def wrap(obj, name, tag):
    orig = getattr(obj, name)

    def w(*a, **k):
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            T[tag] += time.perf_counter() - t
            C[tag] += 1
    setattr(obj, name, w)


need = N_WARM + N_TIME
lg, toks = None, []
with open("D:/Projects/TanitAD/data/refe_navtrain10/r0_s3/targets_rank0.jsonl", encoding="utf-8") as f:
    by = {}
    for line in f:
        if not line.endswith("\n"):
            continue
        r = json.loads(line)
        by.setdefault(r["log_name"], []).append(r["token"])
        if len(by[r["log_name"]]) >= need:
            lg, toks = r["log_name"], by[r["log_name"]][:need]
            break
dbs = NS.index_dbs()
BR.TEACHER_DEVICE = "cpu"
planner = BR.build_planner("cpu")
cams, cam_ts = BR.camera_arrays(dbs[lg])
scs = list(NS.build_scenarios_for_log(dbs[lg], toks))
# stage timers (installed AFTER build_planner so they wrap whatever the levers installed)
import driverl.nuplan.feature_builder as FB  # noqa: E402
import driverl.nuplan.planner as DP  # noqa: E402
from nuplan.planning.script import driverl_runtime_map_features as M  # noqa: E402
wrap(FB.DriveRLNuPlanFeatureBuilder, "build", "features.build")
wrap(FB.DriveRLNuPlanFeatureBuilder, "_build_map_features", "features.map")
wrap(FB, "_build_map_arrays", "map.arrays")                     # FB binds these BY NAME
wrap(FB, "_sample_route_points_from_map", "map.route_points")
wrap(FB, "stitch_lane_segments", "map.stitch")
wrap(M, "_drivable_area_boundaries", "map.arrays.drivable")
wrap(M, "_merge_proximal_map_objects", "map.arrays.proximal")
wrap(M, "_write_segment_candidates", "map.arrays.write_segments")
wrap(FB.DriveRLNuPlanFeatureBuilder, "_write_agent_slot", "agents.write_slot")
wrap(type(planner), "compute_planner_trajectory", "planner.step")


def one(sc):
    controller = BR.build_controller(sc, "cpu")
    planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                             mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
    return BR.make_row(sc, planner, controller, 0, cams, cam_ts, 0)


for sc in scs[:N_WARM]:
    one(sc)
agent = getattr(planner, "_agent", None)        # loaded lazily by the first initialize()
if agent is not None:
    for nm, tag in (("forward_with_features", "policy.forward"), ("preprocess", "policy.preprocess")):
        if hasattr(agent, nm):
            wrap(agent, nm, tag)
print("agent wrapped:", agent is not None)
T.clear(); C.clear()
t0 = time.perf_counter()
for sc in scs[N_WARM:]:
    one(sc)
wall = time.perf_counter() - t0
print(f"log {lg}  FAST={FAST}  timed {N_TIME} rollouts: {wall:.2f} s  ({wall / N_TIME:.3f} s/rollout)")
for k in sorted(T, key=lambda k: -T[k]):
    print(f"  {k:22s} {T[k]:8.2f} s  {100 * T[k] / wall:5.1f} %  calls {C[k]:6d}  "
          f"{1000 * T[k] / max(C[k], 1):7.2f} ms/call")
