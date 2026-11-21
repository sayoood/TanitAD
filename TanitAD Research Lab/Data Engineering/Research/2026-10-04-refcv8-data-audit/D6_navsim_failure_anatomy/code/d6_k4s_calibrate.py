"""D6 -- CALIBRATION of the state-level negative control K4s (SPEC_P1P2_A2_P1PRIME.md) on a CONTROL population with NO candidate evaluation.   NAVSIM VENV, CPU only.

    PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit" OMP_NUM_THREADS=2 \
    C:/Users/Admin/navsim-crun/venv/Scripts/python.exe d6_k4s_calibrate.py <A1 hooks.json> <tokens file> <out.json> [--n 1000] [--seed 20261006]

Population: n random navhard scenes drawn from <tokens file> (the caller passes the scenes OUTSIDE every P1 / P1' set), ONE plan per scene (the banked emitted
pick, and separately the STOP plan) -- no fan, no candidate search.  Mutation: simulate the plan with the devkit tracker, then translate the SIMULATED ego
states laterally by +30 m / +100 m (so the tracker cannot damp it) and score DAC / DDC map-only.  The share that stays DAC- and DDC-clean is the single-plan
residual of the mutation (scenes whose drivable polygon is so large that the shifted footprint is still inside it).  The per-scene any-of-117 rate is >= this.
"""
import json
import lzma
import math
import pickle
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import d6_rescore as R                                              # noqa: E402
from navsim.common.dataclasses import Trajectory                     # noqa: E402
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array   # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex, StateIndex   # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling   # noqa: E402

hooks_path, tokens_path, out_path = sys.argv[1:4]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1000
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 20261006
hooks = {c["token"]: c for c in json.load(open(hooks_path, encoding="utf-8"))["pdm_score_calls"]}
toks = sorted(l.strip() for l in open(tokens_path) if l.strip())
rng = np.random.default_rng(seed)
pick = sorted(rng.choice(toks, min(n, len(toks)), replace=False))
ps, simulator, scorer, policy = R.build_objects()
SAMP = TrajectorySampling(time_horizon=4, interval_length=0.5)
res = {"n": len(pick), "seed": seed, "plans": {}}
acc = {(pl, sh): [] for pl in ("pick", "stop") for sh in (0.0, 30.0, 100.0)}
for t in pick:
    h = hooks[t]
    with lzma.open(R.cache_path(h["log_name"], t), "rb") as f:
        mc = pickle.load(f)
    ini = mc.ego_state
    h0 = ini.rear_axle.heading
    for name, plan in (("pick", np.asarray(h["agent_poses"], np.float32)), ("stop", np.zeros((8, 3), np.float32))):
        tr = Trajectory(poses=plan, trajectory_sampling=SAMP)
        arr = get_trajectory_as_array(transform_trajectory(tr, ini), ps, ini.time_point)
        sim0 = simulator.simulate_proposals(arr[None], ini)
        for sh in (0.0, 30.0, 100.0):
            sim = sim0.copy()
            sim[:, :, StateIndex.X] += -math.sin(h0) * sh
            sim[:, :, StateIndex.Y] += math.cos(h0) * sh
            r = scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map, mc.map_parameters, None, mc.past_human_trajectory)[0]
            acc[(name, sh)].append((float(r["drivable_area_compliance"].iloc[0]) == 1.0) and (float(r["driving_direction_compliance"].iloc[0]) == 1.0))
for (name, sh), v in acc.items():
    res["plans"][f"{name}_shift{int(sh)}m"] = {"share_DAC_and_DDC_clean": float(np.mean(v)), "n_clean": int(np.sum(v)), "n": len(v)}
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res["plans"], indent=1))
