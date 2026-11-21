"""D6 P1 -- CPU scoring of the exported candidate universe (SPEC_P1P2 s3).   NAVSIM VENV (C:/Users/Admin/navsim-crun/venv).

    PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit" OMP_NUM_THREADS=2 \
    C:/Users/Admin/navsim-crun/venv/Scripts/python.exe d6_fan_score.py --fan <fan_R7_A1.jsonl> --hooks <A1 30k hooks.json> \
        --out <scored.jsonl> [--shard i/n] [--limit N] [--no-tierc] [--lateral-shift 8.0] [--k5 100] [--exact-ids <file>]

Per scene (resumable): the 181 exported candidates (117 fan + 64 WTA) and the STOP plan (candidate 181, all-zero poses) are simulated
in ONE batch and scored in ONE ``score_proposals`` call against the reactive environment simulated for the BANKED PICK
(tiers A+B of the SPEC: DAC / DDC / TLC exact, NC / TTC approximate because the IDM agents would react to each candidate differently).
Tier C (``--no-tierc`` to skip): the highest-``r7_score`` candidate that is clean at A+B is re-scored through the exact devkit path
(``pdm_score``: its own environment, human-penalty filter) and its eight sub-scores are banked.  ``--exact-ids`` additionally runs the
exact path on the emitted pick and on STOP for the listed tokens (controls K1 / K3).
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
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array   # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling   # noqa: E402

SAMP = TrajectorySampling(time_horizon=4, interval_length=0.5)
N_FAN = 117


def dec(b64, shape):
    if b64 is None:
        return None
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape).copy()


def traj_arr(poses, ini, ps):
    tr = Trajectory(poses=np.asarray(poses, dtype=np.float32), trajectory_sampling=SAMP)
    return get_trajectory_as_array(transform_trajectory(tr, ini), ps, ini.time_point)


def codes(res_list, key):
    return [float(r[key].iloc[0]) for r in res_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan", required=True)
    ap.add_argument("--hooks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-tierc", action="store_true")
    ap.add_argument("--lateral-shift", type=float, default=0.0, help="K4 mutation: add this to every candidate's y (m)")
    ap.add_argument("--k5", type=int, default=0)
    ap.add_argument("--exact-ids", default="")
    ap.add_argument("--only-tokens", default="", help="score only the scenes listed in this file")
    ap.add_argument("--shift-states", type=float, default=0.0, help="POST-HOC diagnostic K4b: translate the SIMULATED ego states laterally (m) after the tracker, before scoring")
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    exact_ids = set(l.strip() for l in open(a.exact_ids) if l.strip()) if a.exact_ids else set()
    only = set(l.strip() for l in open(a.only_tokens) if l.strip()) if a.only_tokens else None
    hooks = {c["token"]: c for c in json.load(open(a.hooks, encoding="utf-8"))["pdm_score_calls"]}
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["token"])
            except Exception:                                       # noqa: BLE001
                pass
    ps, simulator, scorer, policy = R.build_objects()
    rng = np.random.default_rng(0)
    cnt = 0
    with open(a.fan, encoding="utf-8") as ff, open(a.out, "a", encoding="utf-8") as fo:
        for li, line in enumerate(ff):
            if li % n != k:
                continue
            try:
                fr = json.loads(line)
            except Exception:                                       # noqa: BLE001
                continue                                            # torn last line of a killed export
            tok = fr["token"]
            if tok in done or (only is not None and tok not in only):
                continue
            if a.limit and cnt >= a.limit:
                break
            t0 = time.time()
            h = hooks[tok]
            rec = {"token": tok, "stage": fr["stage"], "sel_idx": fr["sel_idx"], "sel_idx_base": fr.get("sel_idx_base"), "r7_sel_idx": fr.get("r7_sel_idx"), "n_cands": fr["n_cands"]}
            try:
                nc_ = fr["n_cands"]
                poses = dec(fr["cands_poses_b64"], (nc_, 8, 3))
                if a.lateral_shift:
                    poses = poses.copy()
                    poses[:, :, 1] += a.lateral_shift
                emitted = np.asarray(fr["poses_emitted"], dtype=np.float32)
                rec["pick_member_vs_emitted_max_abs"] = float(np.abs(dec(fr["cands_poses_b64"], (nc_, 8, 3))[fr["sel_idx"]] - emitted).max())
                with lzma.open(R.cache_path(h["log_name"], tok), "rb") as f:
                    mc = pickle.load(f)
                ini = mc.ego_state
                allp = np.concatenate([poses, np.zeros((1, 8, 3), np.float32)], axis=0)      # + STOP as the last candidate
                arrs = np.stack([traj_arr(p, ini, ps) for p in allp], axis=0)
                sim = simulator.simulate_proposals(arrs, ini)
                if a.shift_states:                      # K4b (post-hoc): a mutation the LQR tracker cannot damp -- shift the simulated states themselves
                    import math
                    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex as _SI
                    h0 = ini.rear_axle.heading
                    sim = sim.copy()
                    sim[:, :, _SI.X] += -math.sin(h0) * a.shift_states
                    sim[:, :, _SI.Y] += math.cos(h0) * a.shift_states
                pick_idx = fr["sel_idx"]
                pick_sim = sim[pick_idx] if not a.lateral_shift else sim[pick_idx]
                env = policy.simulate_environment(pick_sim, mc)
                res = scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                             mc.map_parameters, env, mc.past_human_trajectory)
                rec["nc"] = codes(res, "no_at_fault_collisions")
                rec["dac"] = codes(res, "drivable_area_compliance")
                rec["ddc"] = codes(res, "driving_direction_compliance")
                rec["tlc"] = codes(res, "traffic_light_compliance")
                rec["ttc"] = codes(res, "time_to_collision_within_bound")
                rec["prog_raw"] = [round(float(x), 3) for x in scorer._progress_raw]
                rec["end_dist"] = [round(float(np.hypot(*p[-1, :2])), 3) for p in allp]
                clean = [(rec["nc"][i] == 1.0 and rec["dac"][i] == 1.0 and rec["ddc"][i] == 1.0 and rec["tlc"][i] == 1.0) for i in range(nc_)]
                rec["n_clean_full_approx"] = int(sum(clean))
                # Tier C: best-ranked clean candidate under r7_score
                if not a.no_tierc and any(clean):
                    sc = dec(fr.get("r7_score_b64"), (nc_,))
                    if sc is None:          # as-launched refcv7 has no r7 scorer (amendment A1): rank by the E9 selection score the model actually uses
                        s9 = dec(fr["sel_score_v3_b64"], (N_FAN,))
                        rk9 = dec(fr["reach_keep_b64"], (N_FAN,))
                        sc = np.where(rk9 > 0, s9, -np.inf)[:nc_]
                    order = [i for i in np.argsort(-sc) if clean[i]]
                    best = int(order[0])
                    row, _ = pdm_score(mc, Trajectory(poses=poses[best].astype(np.float32), trajectory_sampling=SAMP), ps, simulator, scorer, policy)
                    rec["tierc"] = {"cand": best, **{kk: float(row[kk].iloc[0]) for kk in R.SUBS}}
                    rec["tierc"]["confirmed_clean"] = bool(all(rec["tierc"][kk] == 1.0 for kk in ("no_at_fault_collisions", "drivable_area_compliance",
                                                                                                   "driving_direction_compliance", "traffic_light_compliance")))
                if tok in exact_ids:
                    ex = {}
                    for name, pl in (("pick", emitted), ("stop", np.zeros((8, 3), np.float32))):
                        row, _ = pdm_score(mc, Trajectory(poses=pl, trajectory_sampling=SAMP), ps, simulator, scorer, policy)
                        ex[name] = {kk: float(row[kk].iloc[0]) for kk in R.SUBS}
                    ex["banked_pick"] = {kk: float(h["row"][kk]) for kk in R.SUBS}
                    rec["exact"] = ex
                if a.k5 and cnt < a.k5:
                    j = int(rng.integers(0, nc_))
                    s1 = simulator.simulate_proposals(arrs[j][None], ini)
                    r1 = scorer.score_proposals(s1, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                                mc.map_parameters, None, mc.past_human_trajectory)
                    rec["k5"] = {"cand": j, "single_dac": float(r1[0]["drivable_area_compliance"].iloc[0]),
                                 "single_ddc": float(r1[0]["driving_direction_compliance"].iloc[0]),
                                 "batch_dac": rec["dac"][j], "batch_ddc": rec["ddc"][j]}
                rec["status"] = "OK"
            except Exception as e:                                  # noqa: BLE001
                rec["status"] = "ERR"
                rec["err"] = repr(e)[:300]
            rec["wall_s"] = round(time.time() - t0, 3)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            cnt += 1
    print("DONE", cnt, "->", a.out)


if __name__ == "__main__":
    main()
