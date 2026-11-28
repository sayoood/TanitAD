"""D6 -- MEASURE the CPU cost of scoring a 117-candidate fan per token (DAC/DDC only, map-only metrics), so the
'score the fan on the DAC-zero tokens' probe is priced from a measurement, not an estimate.

The candidates here are SYNTHETIC (the banked plan shifted laterally / scaled in speed; 117 of them) -- only the
timing and the batch behaviour are measured, NOT any property of refcv7's real fan (which is not banked).
    PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit" OMP_NUM_THREADS=2 \
    C:/Users/Admin/navsim-crun/venv/Scripts/python.exe d6_fanbench.py <hooks.json> <tokens.txt> <out.json>
"""
import json
import lzma
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from d6_rescore import build_objects, cache_path                     # noqa: E402
from navsim.common.dataclasses import Trajectory                      # noqa: E402,F401
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array   # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex   # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling   # noqa: E402


def main():
    hooks = {c["token"]: c for c in json.load(open(sys.argv[1], encoding="utf-8"))["pdm_score_calls"]}
    toks = [l.strip() for l in open(sys.argv[2]) if l.strip()]
    ps, simulator, scorer, policy = build_objects()
    res = []
    for tok in toks:
        h = hooks[tok]
        with lzma.open(cache_path(h["log_name"], tok), "rb") as f:
            mc = pickle.load(f)
        plan = np.asarray(h["agent_poses"], dtype=np.float64)
        ini = mc.ego_state
        rng = np.random.default_rng(0)
        N = 117
        # synthetic fan: lateral shift in [-3, 3] m ramping with time x speed scale in [0.5, 1.3]
        arrs = []
        t0 = time.time()
        for i in range(N):
            sh = rng.uniform(-3, 3)
            sc = rng.uniform(0.5, 1.3)
            p = plan.copy()
            p[:, 0] *= sc
            p[:, 1] = p[:, 1] * sc + sh * np.linspace(0.1, 1.0, 8)
            tr = Trajectory(poses=p.astype(np.float32), trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5))
            itr = transform_trajectory(tr, ini)
            arrs.append(get_trajectory_as_array(itr, ps, ini.time_point))
        t_build = time.time() - t0
        t0 = time.time()
        sim = simulator.simulate_proposals(np.stack(arrs, 0), ini)
        t_sim = time.time() - t0
        t0 = time.time()
        r = scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                   mc.map_parameters, None, mc.past_human_trajectory)
        t_score = time.time() - t0
        dac = np.array([float(x["drivable_area_compliance"].iloc[0]) for x in r])
        res.append({"build_s": round(t_build, 3), "simulate_117_s": round(t_sim, 3), "score_117_incl_all_metrics_s": round(t_score, 3),
                    "n_candidates": N, "n_dac_pass": int(dac.sum())})
        print(res[-1], flush=True)
    json.dump({"note": "synthetic candidates; timing only", "rows": res,
               "median_total_s_per_token": float(np.median([r["build_s"] + r["simulate_117_s"] + r["score_117_incl_all_metrics_s"] for r in res]))},
              open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
