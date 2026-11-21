"""Where does the seam's time go OUTSIDE the GPU forward? MEASURED on the dev box's CPU with the model call STUBBED.

The loop is refe_navtest_seam.py's own per-token loop (same scenario builder, same planner input construction, same
route/goal, same frame reads, same frame control), timed per phase; only `self.model(...)` returns zeros of the
right shapes, so no GPU is touched (CUDA_VISIBLE_DEVICES=-1) and no checkpoint is needed. Two regimes, because the
per-LOG fixed costs (DB index, map layers, calibration) amortise very differently:
  * 'full-like'   : every navtest token of 2 logs (full navtest averages 89 tokens/log)
  * 'sub200-like' : 1 token from each of 10 other logs (W3's sub200 averages 2.15 tokens/log)
Writes seam_cpu_side_timing.json. The per-token GPU forward must be ADDED to these numbers.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
os.environ.setdefault("REFE_BACKBONE_ROOT", "D:/Projects/TanitAD/data/backbones")
PKG = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, f"{PKG}/eval")
sys.path.insert(0, f"{PKG}/refe")
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import navtrain_scenarios as NS  # noqa: E402
import refe_navtest_seam as S  # noqa: E402
from planner import REFePlanner  # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization  # noqa: E402


class ZeroModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, img, ego_vec, goal, calib=None):
        return torch.zeros(1, 64, 20, 3), torch.zeros(1, 64, 6)


def main() -> int:
    exp = json.load(gzip.open(S.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    sub = json.load(open("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/"
                         "raw/A1_sub200_tokens.json", encoding="utf-8"))
    by_log: dict = {}
    for t in exp:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    logs = sorted(by_log)
    sub_logs = sorted({sub["token_log"][t] for t in sub["tokens"]})
    full_like = [lg for lg in logs if 60 <= len(by_log[lg]) <= 110][:2]
    one_each = [lg for lg in sub_logs if lg not in full_like][:10]
    plan = [("full-like", lg, by_log[lg]) for lg in full_like] + [("sub200-like", lg, by_log[lg][:1]) for lg in one_each]
    t_init = time.perf_counter()
    planner = REFePlanner(checkpoint=None, images_root="D:/Projects/TanitAD/data/refe_navtest/frames",
                          db_dir=S.TEST_DB_DIR, backbone="vitl16", device="cpu", select="best")
    planner.model = ZeroModel(planner.model.cfg)
    planner.per_sample_calib = True            # the trained snapshots read each log's rig (R22): keep that cost
    init_s = time.perf_counter() - t_init
    ph = {}
    per_regime = {}

    def add(k, dt):
        ph[k] = ph.get(k, 0.0) + dt

    for regime, log, toks in plan:
        t_log = time.perf_counter()
        db = os.path.join(S.TEST_DB_DIR, f"{log}.db")
        n = 0
        t = time.perf_counter()
        it = NS.build_scenarios_for_log(db, toks, history_rows=1, future_rows=80)
        while True:
            try:
                sc = next(it)
            except StopIteration:
                break
            add("scenario_build", time.perf_counter() - t)
            planner._scenario = sc
            t = time.perf_counter()
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                     mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            add("initialize_route_ids_goal_ego", time.perf_counter() - t)
            t = time.perf_counter()
            img = planner._image_for(ego)
            add("frames_decode_resize", time.perf_counter() - t)
            t = time.perf_counter()
            traj, score, k = planner.infer(ego, img)          # model stubbed: this is ego vec + route goal + calib
            add("infer_minus_model(route_goal,calib)", time.perf_counter() - t)
            t = time.perf_counter()
            S.to_navsim(planner.executed(traj, k).float().cpu().numpy())
            np.stack([S.to_navsim(planner.executed(traj, j).float().cpu().numpy()) for j in range(traj.shape[0])])
            add("to_navsim_x65", time.perf_counter() - t)
            t = time.perf_counter()
            list(sc.get_ego_future_trajectory(0, 4.0, 8))
            list(sc.get_ego_future_trajectory(0, 4.0, 20))
            add("frame_control_futures", time.perf_counter() - t)
            n += 1
            t = time.perf_counter()
        r = per_regime.setdefault(regime, {"logs": 0, "tokens": 0, "seconds": 0.0})
        r["logs"] += 1
        r["tokens"] += n
        r["seconds"] += time.perf_counter() - t_log
    for r in per_regime.values():
        r["s_per_token"] = round(r["seconds"] / max(r["tokens"], 1), 3)
        r["seconds"] = round(r["seconds"], 1)
    tot = sum(r["tokens"] for r in per_regime.values())
    out = {"host": "dev box CPU (i9-12900F), CUDA hidden, model call stubbed", "planner_build_s": round(init_s, 1),
           "regimes": per_regime, "phase_seconds_total": {k: round(v, 2) for k, v in ph.items()},
           "phase_s_per_token_mean": {k: round(v / max(tot, 1), 4) for k, v in ph.items()},
           "logs": {"full-like": full_like, "sub200-like": one_each}, "tokens_total": tot,
           "torch_threads": torch.get_num_threads()}
    json.dump(out, open(os.path.join(HERE, "seam_cpu_side_timing.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
