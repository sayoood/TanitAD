#!/usr/bin/env python3
"""(NAVSIM VENV, v2 tree) The discriminating experiment for PREREG §2: is the NC / DAC FUNCTION unchanged,
with the smoke's one C-NC difference caused by the INPUTS (the observation window) alone?

    <py> code/counterfactual_v2fn_v1inputs.py --run <suite run dir> --arm CV \
        --v1-cache D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest --v2-cache <v2 cache> \
        --out raw/counterfactual_<name>_<arm>.json

W8, 2026-09-26 (PREREG AMENDMENT A2). For every token of the run it scores the arm's OWN plan (the poses the
official scorer received, from the run's pdm_score hook) with the v2 devkit's simulator + PDMScorer, twice:

  CONTROL  v2 function on the v2 cache WITH the non-reactive log-replay detections (exactly the call chain of
           evaluate/pdm_score.py:137-169, minus the human filter which never fires on navtest NC/DAC) —
           must reproduce the run's own CSV (NC, DAC) EXACTLY, or this re-scoring path is not the scorer's;
  CF       v2 function on the v1.1 cache's inputs (its cached observation, centreline, route, drivable map,
           ego state, PDM-Closed trajectory; no detection update) — if the function is unchanged, NC and DAC
           must equal W3's v1.1 CSV EXACTLY on every token; TTC (whose loop domain DID change, v2 L536-538)
           may only be >= v1's.

The v1 pickle is read in the v2 tree (e_obs_classify.py's control: 418/418 identical digests to a v1-tree
read). Nothing here edits the devkit.
"""
from __future__ import annotations

import argparse
import csv
import json
import lzma
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
W3 = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--v1-cache", required=True, type=Path)
    ap.add_argument("--v2-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokens-json", default=None, type=Path,
                    help="restrict to A2_counterfactual_tokens.tokens of a cross_protocol_check output (A2)")
    a = ap.parse_args(argv)
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents

    t0 = time.time()
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    sim, scorer, tap = PDMSimulator(ps), PDMScorer(ps, PDMScorerConfig()), LogReplayTrafficAgents(ps)
    hooks = json.loads((a.run / "raw" / a.arm / f"{a.arm}_hooks.json").read_text(encoding="utf-8"))["pdm_score_calls"]
    poses = {c["token"]: np.asarray(c["agent_poses"], dtype=np.float64) for c in hooks if c.get("token")}
    with open(a.run / "scores" / f"{a.arm}.csv", encoding="utf-8", newline="") as fh:
        v2csv = {r["token"]: r for r in csv.DictReader(fh) if r["token"] != "average_all_frames"}
    with open(W3 / f"{a.arm}_navtest" / f"{a.arm}_navtest.csv", encoding="utf-8", newline="") as fh:
        v1csv = {r["token"]: r for r in csv.DictReader(fh) if r["token"] != "average"}
    v2p = {p.parts[-2]: p for p in a.v2_cache.glob("*/*/*/metric_cache.pkl")}

    def score(mc, plan, detections):
        traj = Trajectory(plan, TrajectorySampling(time_horizon=4, interval_length=0.5))
        pred = transform_trajectory(traj, mc.ego_state)
        st = np.concatenate([get_trajectory_as_array(mc.trajectory, ps, mc.ego_state.time_point)[None],
                             get_trajectory_as_array(pred, ps, mc.ego_state.time_point)[None]], axis=0)
        sims = sim.simulate_proposals(st, mc.ego_state)
        dets = tap.simulate_environment(sims[1], mc) if detections else None
        r = scorer.score_proposals(sims, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                   None, dets)[1]
        return {k: float(r[k].iloc[0]) for k in ("no_at_fault_collisions", "drivable_area_compliance",
                                                   "time_to_collision_within_bound")}

    only = None
    if a.tokens_json:
        only = set(json.loads(a.tokens_json.read_text(encoding="utf-8"))["A2_counterfactual_tokens"]["tokens"])
    rows, fails = {}, []
    for tok in sorted(v2csv):
        if only is not None and tok not in only:
            continue
        try:
            p2 = v2p[tok]
            p1 = a.v1_cache / p2.parts[-4] / p2.parts[-3] / tok / "metric_cache.pkl"
            m2 = pickle.loads(lzma.decompress(p2.read_bytes()))
            m1 = pickle.loads(lzma.decompress(p1.read_bytes()))
            rows[tok] = {"control_v2fn_v2in": score(m2, poses[tok], True), "cf_v2fn_v1in": score(m1, poses[tok], False)}
        except Exception as e:                                           # noqa: BLE001
            fails.append({"token": tok, "exception": f"{type(e).__name__}: {e}"[:300]})

    def f(x):
        return float(x)
    res = {"control_NC": [], "control_DAC": [], "cf_NC": [], "cf_DAC": [], "cf_TTC_lt_v1": []}
    for tok, r in rows.items():
        c, cf = r["control_v2fn_v2in"], r["cf_v2fn_v1in"]
        if c["no_at_fault_collisions"] != f(v2csv[tok]["no_at_fault_collisions"]):
            res["control_NC"].append(tok)
        if c["drivable_area_compliance"] != f(v2csv[tok]["drivable_area_compliance"]):
            res["control_DAC"].append(tok)
        if cf["no_at_fault_collisions"] != f(v1csv[tok]["no_at_fault_collisions"]):
            res["cf_NC"].append({"token": tok, "cf": cf["no_at_fault_collisions"], "v1": f(v1csv[tok]["no_at_fault_collisions"])})
        if cf["drivable_area_compliance"] != f(v1csv[tok]["drivable_area_compliance"]):
            res["cf_DAC"].append(tok)
        if cf["time_to_collision_within_bound"] < f(v1csv[tok]["time_to_collision_within_bound"]):
            res["cf_TTC_lt_v1"].append(tok)
    rec = {"schema": "w8-counterfactual/1", "arm": a.arm, "run": str(a.run).replace("\\", "/"),
           "restricted_to": (str(a.tokens_json).replace("\\", "/") if a.tokens_json else None),
           "n_tokens": len(rows), "n_failures": len(fails), "failures": fails[:10],
           "CONTROL_reproduces_run_csv": {"NC_mismatch": len(res["control_NC"]), "DAC_mismatch": len(res["control_DAC"]),
                                          "must_read": 0},
           "CF_v2_function_on_v1_inputs_vs_v1_csv": {"NC_mismatch": len(res["cf_NC"]), "DAC_mismatch": len(res["cf_DAC"]),
                                                     "TTC_below_v1": len(res["cf_TTC_lt_v1"]), "want": "0 / 0 / 0",
                                                     "first_NC": res["cf_NC"][:10]},
           "the_token_that_failed_C_NC": {k: rows.get("937ca624cc2658a6", {}).get(k) for k in ("control_v2fn_v2in", "cf_v2fn_v1in")},
           "wall_s": round(time.time() - t0, 1)}
    a.out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("failures",)}, indent=1))
    ok = (rows and not fails and not res["control_NC"] and not res["control_DAC"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
