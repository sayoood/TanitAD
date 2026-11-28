#!/usr/bin/env python3
"""zt5_smoke.py -- PLUMBING ONLY (declared in PREREG_ZT5 section 0): can the dev box rebuild the held-out sets' scenarios and
reproduce the banked PDM labels, and how long does one set take?  Prints identity booleans, timings and the ego
state's steering / yaw-rate fields.  Reads no outcome beyond those identities.

    PYTHONPATH=<navsim v1.1>;<refe> navsim-crun python zt5_smoke.py --n 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

SETS = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets"
LABELS = "D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--file", default="onpolicy_r0_hw0.jsonl")
    a = ap.parse_args()
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    dbs = NS.index_dbs()
    rows = []
    with open(f"{SETS}/{a.file}", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            if r["log_name"] in dbs:
                rows.append(r)
            if len(rows) >= a.n:
                break
    lab = {}
    with open(f"{LABELS}/pdm_{a.file}", encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            lab[tuple(d["key"])] = d
    for r in rows:
        t0 = time.time()
        sc = {s.scenario_name: s for s in NS.build_scenarios_for_log(dbs[r["log_name"]], [r["token"]], history_rows=40,
                                                                      future_rows=100)}[r["token"]]
        t1 = time.time()
        parts = PT.metric_cache_parts(sc)
        t2 = time.time()
        props = np.concatenate([np.asarray(r["traj"], np.float64), np.asarray(r["yaw"], np.float64)[..., None]], -1)
        p8 = PT.executed_plans8(props, repair=True)
        t = PT.score_plans(parts, p8)
        t3 = time.time()
        L = lab[(r["log_name"], r["token"], int(r["step"]), int(r["rank"]))]
        ident = {k: float(np.abs(np.round(t[k], 5) - np.asarray(L[k])).max()) for k in ("nc", "dac", "ep", "ttc", "c", "pdms")}
        ego = parts[1]
        print(json.dumps({"token": r["token"], "M": int(props.shape[0]), "T": int(props.shape[1]), "ident_maxabs": ident,
                          "t_scenario": round(t1 - t0, 2), "t_parts": round(t2 - t1, 2), "t_score64": round(t3 - t2, 2),
                          "ego_steer": float(ego.tire_steering_angle), "ego_yawrate": float(ego.dynamic_car_state.angular_velocity),
                          "ego_v": [float(ego.dynamic_car_state.rear_axle_velocity_2d.x), float(ego.dynamic_car_state.rear_axle_velocity_2d.y)],
                          "ego_a": [float(ego.dynamic_car_state.rear_axle_acceleration_2d.x),
                                    float(ego.dynamic_car_state.rear_axle_acceleration_2d.y)]}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
