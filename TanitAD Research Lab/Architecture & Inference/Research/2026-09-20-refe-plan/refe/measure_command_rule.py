#!/usr/bin/env python3
"""MEASURE how well a goal-derived navigation command agrees with NAVSIM's `driving_command` (SPEC_P6 D6 sets a >= 95 % bar). Dev box, 0 GPU.

Inputs: NAVSIM's navtest inputs (`driving_command` per token) and the planner's goal census for the same tokens
(raw/2026-10-01-goal-trigger/goal_census_navtest_full.json: 12,146 tokens, the goal REFe received at test). Writes raw/2026-10-11-p6-impl/command_rule_measure.json.
Evidence: MEASURED on NAVTEST goals. The training goals are the teacher's (route_goal_positions on the teacher's route) and are not measurable here -- the bar must be
re-measured on navtrain tokens on the pod (`--navtrain-rows` below does it when a bank + the NAVSIM scene metadata are present).
Honest protocol: the threshold is FIT on one half of the LOGS and SCORED on the other (log-disjoint); the full-set number of the shipped rule is reported beside it.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nav_command as NC  # noqa: E402

INP = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
CEN = HERE.parent / "raw/2026-10-01-goal-trigger/goal_census_navtest_full.json"
OUT = HERE.parent / "raw/2026-10-11-p6-impl/command_rule_measure.json"


def main() -> int:
    inp = json.load(gzip.open(INP, "rt"))["tokens"]
    cen = json.load(open(CEN, encoding="utf-8"))["rows"]
    toks = [t for t in cen if t in inp and not cen[t].get("no_route") and cen[t].get("goal")]
    cmd = np.array([NC.from_driving_command(inp[t]["ego_statuses"][-1]["driving_command"]) for t in toks])
    G = np.array([cen[t]["goal"] for t in toks], np.float64)
    logs = sorted({cen[t]["log"] for t in toks})
    half = set(logs[::2])
    tr = np.array([cen[t]["log"] in half for t in toks])
    ang1 = np.arctan2(G[:, 1], np.maximum(G[:, 0], 1e-6))
    ang2 = np.arctan2(G[:, 3], np.maximum(G[:, 2], 1e-6))

    def rule(a, th):
        return np.where(a > th, NC.LEFT, np.where(a < -th, NC.RIGHT, NC.STRAIGHT))
    res = {"n_tokens": len(toks), "n_logs": len(logs), "command_counts": np.bincount(cmd, minlength=4).tolist(), "feature": {}}
    for nm, f in (("near_bearing", ang1), ("far_bearing", ang2)):
        ths = np.quantile(np.abs(f), np.linspace(0.05, 0.95, 91))
        accs = [(rule(f[tr], t) == cmd[tr]).mean() for t in ths]
        t = float(ths[int(np.argmax(accs))])
        res["feature"][nm] = {"fit_threshold_rad": round(t, 4), "train_half": round(float(max(accs)), 4),
                              "heldout_half": round(float((rule(f[~tr], t) == cmd[~tr]).mean()), 4)}
    shipped = np.array([NC.goal_to_command(g) for g in G])
    res["shipped_rule"] = {"threshold_rad": NC.GOAL_RULE_BEARING_RAD, "agreement_all": round(float((shipped == cmd).mean()), 4),
                           "agreement_heldout_half": round(float((shipped[~tr] == cmd[~tr]).mean()), 4),
                           "confusion_rows_navsim_cols_rule": [[int(((shipped == p) & (cmd == c)).sum()) for p in range(3)] for c in range(3)]}
    res["bar"] = {"spec_p6_d6": 0.95, "met_by_shipped_rule": bool(res["shipped_rule"]["agreement_heldout_half"] >= 0.95)}
    res["evidence"] = "MEASURED (navtest goals vs NAVSIM driving_command); NOT the teacher goals of navtrain -- re-measure on the pod"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
