#!/usr/bin/env python3
"""Gate for the SFT-4 labeller: does refe/navsim_pdm_targets.py (NAVSIM's metric cache REBUILT from the nuPlan DB through
our scenario builder) reproduce NAVSIM's scoring on its REAL navtest metric cache?

Reference: raw/2026-10-04-lane-discipline/lane_census.jsonl -- NAVSIM v1.1's scorer on the official navtest metric cache for
the final model's 64 hypotheses + the human (reproduced pick == the harness CSV on 12,146 / 12,146). Here the same 65 plans
(proposals.npz, NAVSIM grid) are scored with the rebuilt cache. Reported per component: agreement within the census rounding (|d| < 1e-4) and
the mean |d|; for the PDMS of the 64: the mean |d| and the within-set rank agreement of the argmax.
    python eval/validate_pdm_targets.py [--n 150]      (re-launches itself in the navsim v1.1 venv)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DRV = os.environ.get("REFE_DRIVE", "E:")
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11 = f"{DRV}/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
MAPS = f"{DRV}/Archive/devbox-C/navsim/data/maps"
DB = f"{DRV}/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
PROPS = f"{DRV}/Projects/TanitAD/data/refe_navtest/proptable/navtest_final/proposals.npz"
CENSUS = os.path.join(PKG, "raw", "2026-10-04-lane-discipline", "lane_census.jsonl")
OUT = os.path.join(PKG, "raw", "2026-10-04-lane-discipline", "validate_pdm_targets.json")


def main() -> int:
    if os.environ.get("REFE_PDMV_CHILD") != "1":
        env = dict(os.environ, REFE_PDMV_CHILD="1", PYTHONPATH=f"{V11};{os.path.join(PKG, 'refe')}", PYTHONIOENCODING="utf-8",
                   NUPLAN_MAPS_ROOT=MAPS, NUPLAN_MAP_VERSION="nuplan-maps-v1.0", REFE_SIM_HZ="10", OMP_NUM_THREADS="2")
        return subprocess.call([NV_PY, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    P = np.load(PROPS)
    idx = {str(t): i for i, t in enumerate(P["token"])}
    rows = []
    for line in open(CENSUS, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:
            rows.append(r)
        if len(rows) >= a.n:
            break
    by: dict = {}
    for r in rows:
        by.setdefault(r["log"], []).append(r)
    comps = ("nc", "dac", "ttc", "c", "ddc", "pdms")          # the census stores no separate EP; it enters PDMS
    agree = {c: 0 for c in comps}
    absd = {c: [] for c in comps}
    total, argmax_same, skipped, t0 = 0, 0, [], time.time()
    per_token = []
    for log, rs in sorted(by.items()):
        want = {r["token"]: r for r in rs}
        got = set()
        for sc in NS.build_scenarios_for_log(f"{DB}/{log}.db", list(want), map_root=MAPS, history_rows=40, future_rows=100):
            tok = sc._initial_lidar_token
            got.add(tok)
            r = want[tok]
            plans = np.concatenate([P["proposals"][idx[tok]].astype(np.float64),
                                    np.asarray(E[tok]["human_future_poses"], np.float64)[None]], 0)
            out = PT.score_plans(PT.metric_cache_parts(sc), plans)
            for c in comps:
                ref = np.array(r[c][1:66], dtype=np.float64)
                d = np.abs(out[c] - ref)
                agree[c] += int((d < 1e-4).sum())          # the census rounds to 4 dp
                absd[c].append(float(d.mean()))
            total += 65
            same = int(np.argmax(out["pdms"][:64])) == int(np.argmax(np.array(r["pdms"][1:65])))
            argmax_same += same
            per_token.append({"token": tok, "max_abs_d": {c: round(float(np.abs(out[c] - np.array(r[c][1:66])).max()), 4) for c in comps}})
        skipped += [t for t in want if t not in got]
        print(f"  {len(per_token)} tokens, {time.time() - t0:.0f} s", flush=True)
    res = {"tokens": len(per_token), "trajectories": total, "skipped_by_scenario_guard": len(skipped),
           "exact_agreement_pct": {c: round(100 * agree[c] / max(total, 1), 3) for c in comps},
           "mean_abs_diff": {c: round(float(np.mean(absd[c])), 5) for c in comps},
           "argmax_pdms_same_pct": round(100 * argmax_same / max(len(per_token), 1), 2),
           "worst_tokens": sorted(per_token, key=lambda x: -x["max_abs_d"]["pdms"])[:15]}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "worst_tokens"}, indent=1))
    print("ZZPDMT_AGREE", json.dumps(res["exact_agreement_pct"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
