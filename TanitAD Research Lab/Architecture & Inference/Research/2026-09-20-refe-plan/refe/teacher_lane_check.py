#!/usr/bin/env python3
"""Does the TRAJECTORY TEACHER (DriveRL, whose rollouts are REFe's proposal targets) respect drivable area, driving
direction and lanes? PI 2026-10-04: "check and validate the teacher itself, if it is considering drivable areas etc."

For every held-out sample in the on-policy QUEUE (props_*.jsonl*, which carry the teacher's rollout `teacher` [T, 3] beside
the model's 64 `props`), three trajectories are judged by the SAME NAVSIM-faithful rules, each on NAVSIM's own
simulation of the plan from the sample's ego state:
  * teacher   the DriveRL teacher's rollout (the supervision target)
  * human     the logged expert future at that step (the floor: legal manoeuvres trip these rules too)
  * model     the model's 64 proposals (summarised: share violating)
Rules: DAC (navsim_dac, 100.00 % agreement with NAVSIM), DDC (navsim_lane, 100.00 % on 13,000 navtest trajectories),
lane keeping lk10 (navsim_lane; > 1.0 m off the containing lane's centreline for >= 1 s outside junctions).
CPU only.  python refe/teacher_lane_check.py --queue <dir> --out <file.jsonl> [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
_S: dict = {}


def _init():
    import navtrain_scenarios as NS
    _S["NS"] = NS
    _S["dbs"] = NS.index_dbs()


def human_p8(sc, step, ego):
    """the logged future at 2 Hz over 4 s, as NAVSIM-frame poses [8, 3] relative to the rear axle at `step`"""
    from nuplan.common.geometry.convert import absolute_to_relative_poses
    fut = list(sc.get_ego_future_trajectory(iteration=step, time_horizon=4.0, num_samples=8))
    rel = absolute_to_relative_poses([ego.rear_axle] + [s.rear_axle for s in fut])[1:]
    return np.array([[p.x, p.y, p.heading] for p in rel], np.float64)


def check_log(job):
    import navsim_dac as ND
    import navsim_lane as NL
    log, rows = job
    out = []
    try:
        if not _S:
            _init()
        scs = {sc.scenario_name: sc for sc in _S["NS"].build_scenarios_for_log(_S["dbs"][log], sorted({r["token"] for r in rows}))}
    except Exception as e:                                                   # noqa: BLE001
        return [json.dumps({"log": log, "error": f"scenario build {type(e).__name__}: {str(e)[:120]}"}) + "\n"]
    for r in rows:
        try:
            sc = scs[r["token"]]
            step = int(r["step"])
            ego = sc.get_ego_state_at_iteration(step)
            tr = np.asarray(r["teacher"], np.float64)
            P = np.asarray(r["props"], np.float64)
            assert tr.shape == (20, 3) and P.shape[1:] == (20, 3), (tr.shape, P.shape)
            p8 = np.concatenate([np.stack([ND.to_navsim(tr)]), human_p8(sc, step, ego)[None],
                                 np.stack([ND.to_navsim(p) for p in P])], 0)       # 0 teacher, 1 human, 2.. model
            sim = ND.simulated_states(p8, ego)
            ids, diag = NL.corrected_route_ids(sc.map_api, ego.rear_axle, list(sc.get_route_roadblock_ids()))
            ddc, lk = NL.navsim_ddc_and_lane(sim, sc.map_api, ego, ids)
            from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
            from navsim_pdm.pdm_array_representation import state_array_to_coords_array
            from navsim_pdm.pdm_enums import BBCoordsIndex
            coords = state_array_to_coords_array(sim, get_pacifica_parameters())
            ci = [i for i in range(coords.shape[2]) if i != int(BBCoordsIndex.CENTER)]
            dac_v = ND.violations(ND.drivable_polygons(sc.map_api, ego), coords[:, :, ci]).astype(float)
            out.append(json.dumps({"log": log, "token": r["token"], "step": step, "rank": int(r.get("rank", 0)),
                                   "speed": float(np.hypot(ego.dynamic_car_state.rear_axle_velocity_2d.x,
                                                           ego.dynamic_car_state.rear_axle_velocity_2d.y)),
                                   "route_changed": bool(diag.get("changed")),
                                   "teacher": {"dac_viol": dac_v[0], "ddc": ddc[0], "lk10": lk[0]},
                                   "human": {"dac_viol": dac_v[1], "ddc": ddc[1], "lk10": lk[1]},
                                   "model": {"dac_viol_share": float(dac_v[2:].mean()), "ddc_lt1_share": float((ddc[2:] < 1).mean()),
                                             "lk10_share": float(lk[2:].mean())}}) + "\n")
        except Exception:                                                    # noqa: BLE001
            out.append(json.dumps({"log": log, "token": r.get("token"), "error": traceback.format_exc()[-300:]}) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    by: dict = {}
    n = 0
    for fp in sorted(glob.glob(os.path.join(a.queue, "props_*.jsonl*"))):
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "teacher" not in r or "props" not in r:
                continue
            by.setdefault(r["log_name"], []).append(r)
            n += 1
            if a.limit and n >= a.limit:
                break
        if a.limit and n >= a.limit:
            break
    print(f"{n} samples over {len(by)} logs", flush=True)
    t0 = time.time()
    k = 0
    with open(a.out, "w", encoding="utf-8") as f, mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for lines in pool.imap_unordered(check_log, list(by.items())):
            f.writelines(lines)
            f.flush()
            k += len(lines)
    print(f"ZZTEACHER_CHECK_DONE {k} {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
