"""D6 P4 -- EXACT devkit scoring of one arm's plans, one token at a time, keeping everything the two-stage aggregation needs.   NAVSIM VENV.

    PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit" OMP_NUM_THREADS=2 \
    C:/Users/Admin/navsim-crun/venv/Scripts/python.exe d6_p4_score.py --plans raw/p4_plans_<arm>.npz --out raw/p4_scored/<arm>_s<k>.jsonl --shard k/n

Per token this is the BODY of the devkit's ``run_pdm_score.run_pdm_score`` loop (run_pdm_score.py:84-117 / :136-169 @0a380a9), verbatim in what
it computes: ``pdm_score(metric_cache, Trajectory(poses, TrajectorySampling(4, 0.5)), simulator.proposal_sampling, simulator, scorer, policy)``
-> the PDMResults row, ``valid``, ``log_name``, ``frame_type``, ``start_time``, the absolute endpoint of the plan's last pose, the start point,
and ``ego_simulated_states`` (the two-frame extended comfort reads it).  The objects are ``d6_rescore.build_objects`` (the ones that reproduced
2,678 banked rows exactly); the plan is built as the seam agent builds it (``Trajectory(poses_f32.copy(), sampling)``).
Output: one JSON line per token (floats by repr = exact; ``ego_simulated_states`` as base64 float64).  Resumable: tokens already in --out
(status OK) are skipped.
"""
from __future__ import annotations

import argparse
import base64
import json
import lzma
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_rescore as R                                              # noqa: E402  (env defaults + navsim imports + cache_path)
from navsim.common.dataclasses import Trajectory                     # noqa: E402
from navsim.evaluate.pdm_score import pdm_score                      # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2  # noqa: E402
from nuplan.common.geometry.convert import relative_to_absolute_poses  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

AGENT_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)      # TanitADSeamAgent default
COLS = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
        "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "multiplicative_metrics_prod",
        "pdm_score")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    z = np.load(a.plans, allow_pickle=False)
    toks = [str(x) for x in z["token"]]
    logs = [str(x) for x in z["log_name"]]
    poses_all = np.asarray(z["poses"], np.float32)
    if poses_all.shape != (len(toks), 8, 3) or not np.isfinite(poses_all).all():
        raise SystemExit(f"plans shape {poses_all.shape} / non-finite")
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "OK":
                done.add(r["token"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ps, simulator, scorer, policy = R.build_objects()
    cnt = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for i, tok in enumerate(toks):
            if i % n != k or tok in done:
                continue
            if a.limit and cnt >= a.limit:
                break
            t0 = time.time()
            rec = {"token": tok}
            try:
                poses = poses_all[i]
                with lzma.open(R.cache_path(logs[i], tok), "rb") as f:
                    mc = pickle.load(f)
                traj = Trajectory(poses.copy(), AGENT_SAMPLING)
                row, ess = pdm_score(mc, traj, simulator.proposal_sampling, simulator, scorer, policy)
                for c in COLS:
                    rec[c] = float(row[c].iloc[0])
                rec["weighted_metrics"] = [float(x) for x in np.asarray(row["weighted_metrics"].iloc[0], np.float64)]
                rec["weighted_metrics_array"] = [float(x) for x in np.asarray(row["weighted_metrics_array"].iloc[0], np.float64)]
                rec["log_name"] = mc.log_name
                rec["frame_type"] = int(mc.scene_type)
                rec["start_time"] = float(mc.timepoint.time_s)
                end_pose = StateSE2(x=traj.poses[-1, 0], y=traj.poses[-1, 1], heading=traj.poses[-1, 2])
                ae = relative_to_absolute_poses(mc.ego_state.rear_axle, [end_pose])[0]
                rec["endpoint_x"], rec["endpoint_y"] = float(ae.x), float(ae.y)
                rec["start_point_x"], rec["start_point_y"] = float(mc.ego_state.rear_axle.x), float(mc.ego_state.rear_axle.y)
                e = np.ascontiguousarray(np.asarray(ess, np.float64))
                rec["ess_shape"] = list(e.shape)
                rec["ess_b64"] = base64.b64encode(e.tobytes()).decode("ascii")
                rec["plan_sha16"] = __import__("hashlib").sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()[:16]
                rec["valid"] = True
                rec["status"] = "OK"
            except Exception as ex:                                 # noqa: BLE001
                rec["status"] = "ERR"
                rec["err"] = repr(ex)[:300]
            rec["wall_s"] = round(time.time() - t0, 3)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            cnt += 1
    print("DONE", cnt, "->", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
