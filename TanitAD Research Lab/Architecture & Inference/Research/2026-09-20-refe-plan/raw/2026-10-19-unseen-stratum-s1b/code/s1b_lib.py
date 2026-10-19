"""UNSEEN-STRATUM S1b library: navtrain's OWN token filter, rebuilt with OUR PDM scorer (DB + map only; no frame, no model, no head).

    navtrain keeps a token iff   PDMS(constant-velocity agent) <= 0.8   AND   PDMS(human / logged trajectory) >= 0.8
    (Dauner et al. 2024 sec. 3.1; products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md:386-391).  Inclusivity is NOT stated there; the
    two variants are measured (CV_STRICT: "> 0.8 removed" == keep CV <= 0.8; HUMAN: "< 0.8 removed" == keep human >= 0.8) and reported.

THE SCORER (every piece quotes its source; all from the tip origin/agent/arch-inf-20260803 @ 5ba3f3f, byte-identical copy in tree/refe):
    refe/navsim_pdm_targets.py:87-96   metric_cache_parts(scenario): NAVSIM's OWN MetricCacheProcessor steps (PDM-Closed + GT observation)
    refe/navsim_pdm_targets.py:99-129  score_plans(parts, plans8[M,8,3]): NAVSIM's OWN PDMSimulator + PDMScorer, EP normalised pairwise vs PDM-Closed
    refe/navtrain_scenarios.py:189-276 build_scenarios_for_log: one NuPlanScenario per token (route keyed on the token)
    CV agent: devkit navsim/agents/constant_velocity_agent.py:31-46 (pinned 3e8291b copy: identical logic, :31-47):
              speed = |ego_statuses[-1].ego_velocity|; poses[i] = [(i+1)*dt*speed, 0, 0] for i in 0..num_poses-1, dt 0.5, 8 poses (time_horizon 4)
    human   : navsim/agents/human_agent.py:35-41 -> Scene.get_future_trajectory(8) (navsim/common/dataclasses.py:288-313): the 8 logged 2 Hz poses
              in the ego rear-axle frame; the DB route is scenario.get_ego_future_trajectory(0, 4.0, 8) + rear-axle local frame, MEASURED 0.0000 m
              against W3's human_future_poses (eval/refe_navtest_seam.py:44-47, :296-301).
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

V11 = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
MAPS = "D:/Archive/devbox-C/navsim/data/maps"
HERE = os.path.dirname(os.path.abspath(__file__))
REFE = os.path.join(os.path.dirname(HERE), "tree", "refe")
S1_CODE = "C:/Users/Admin/par18/unseen_s1/code"
DB_TRAINVAL = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"

from s1b_rules import cv_plan  # noqa: E402,F401  (pure, tested)

_ENV = {"NUPLAN_MAPS_ROOT": MAPS, "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0", "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}


def setup_env():
    for k, v in _ENV.items():
        os.environ[k] = v
    for p in (S1_CODE, REFE, V11):          # V11 first: the pinned NAVSIM v1.1 shadows any installed devkit
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def human_plan(sc) -> np.ndarray | None:
    """The logged 8 future poses at 0.5 s, in the ego rear-axle frame (== Scene.get_future_trajectory, refe_navtest_seam.py:288-301)."""
    ego = sc.initial_ego_state
    px, py, pyaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
    c, s = math.cos(-pyaw), math.sin(-pyaw)
    f8 = list(sc.get_ego_future_trajectory(0, 4.0, 8))[:8]
    if len(f8) != 8:
        return None
    return np.array([[(f.rear_axle.x - px) * c - (f.rear_axle.y - py) * s,
                      (f.rear_axle.x - px) * s + (f.rear_axle.y - py) * c,
                      wrap(f.rear_axle.heading - pyaw)] for f in f8], dtype=np.float64)


def ego_speed(sc) -> float:
    """|ego_velocity| of the current frame: OpenScene's `ego_dynamic_state[:2]` == nuPlan's rear_axle_velocity_2d (ego frame)."""
    v = sc.initial_ego_state.dynamic_car_state.rear_axle_velocity_2d
    return float((v.x ** 2 + v.y ** 2) ** 0.5)


def score_token(PT, sc, extra_plans: np.ndarray | None = None) -> dict:
    """One token: PDMS + sub-scores of the CV agent and the human, plus extra [M,8,3] plans (RED / control arms) in the SAME call.
    Returns {'cv': {...}, 'human': {...}|None, 'speed': v, 'extra': {...}|None, 't_*': seconds}."""
    t0 = time.perf_counter()
    parts = PT.metric_cache_parts(sc)
    t1 = time.perf_counter()
    v = ego_speed(sc)
    cv = cv_plan(v)
    hp = human_plan(sc)
    plans = [cv] + ([hp] if hp is not None else [])
    n_fix = len(plans)
    if extra_plans is not None and len(extra_plans):
        plans += list(np.asarray(extra_plans, np.float64))
    res = PT.score_plans(parts, np.stack(plans).astype(np.float32))
    t2 = time.perf_counter()

    def pick(j):
        return {k: float(res[k][j]) for k in ("pdms", "nc", "dac", "ep", "ttc", "c", "ddc")}
    out = {"speed": v, "cv": pick(0), "human": pick(1) if hp is not None else None, "t_cache": t1 - t0, "t_score": t2 - t1,
           "pdm_closed_progress_ok": True}
    if len(plans) > n_fix:
        out["extra"] = [pick(j) for j in range(n_fix, len(plans))]
    return out
