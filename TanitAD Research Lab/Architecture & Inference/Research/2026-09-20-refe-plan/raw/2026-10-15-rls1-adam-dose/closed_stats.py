#!/usr/bin/env python3
"""Combine the three closed-loop runs (adam_closed.py) into the same statistics as adam_sim.py, next to the open-loop numbers of the SAME stream (stream 0, same draws).
    python closed_stats.py --closed-dir <scratch/closed> --sim adam_sim.json --out closed_stats.json"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adam_sim import stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-dir", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    q = ap.parse_args()
    sim = json.load(open(q.sim))
    res = {"tool": "closed_stats/1", "class": "ESTIMATED (closed loop: the trainer's own step on moving theta; substituted sctx; stream 0 draws; K=50)", "regimes": {}}
    for regime in ("T", "P"):
        runs = {}
        for lam in ("0.0", "0.25", "1.0"):
            p = os.path.join(q.closed_dir, f"closed_{regime}_{lam}.pt")
            if not os.path.exists(p):
                break
            runs[lam] = torch.load(p, weights_only=False)
        if len(runs) < 3:
            continue
        names = runs["0.0"]["names"]
        R = {"at": {}, "log_summary": {}}
        for k in (25, 50):
            st = stats(runs["0.0"]["D"][k], runs["0.25"]["D"][k], runs["1.0"]["D"][k], names)
            ol = sim["streams"]["0"]["modes"][regime]["at"][str(k)]
            R["at"][str(k)] = {"CL_S_C80": st["S_C80"], "OL_S_C80_same_stream": ol["S_C80"], "CL_S_all": st["S_all"], "CL_cos_C80": st["cos_C80"], "CL_U": st["U_full_norm_ratio"], "CL_cos_full": st["cos_full"],
                               "CL_elasticity_e": st["elasticity_e"], "CL_groups": st["groups"]}
        for lam, r in runs.items():
            lg = r["log"]
            R["log_summary"][lam] = {"grad_norm_pre_clip_first": lg[0]["grad_norm_pre_clip"], "grad_norm_pre_clip_last5_mean": sum(x["grad_norm_pre_clip"] for x in lg[-5:]) / 5,
                                     "bce_last5_mean": sum(x["bce"] for x in lg[-5:]) / 5, "epo_last5_mean": sum(x["epo"] for x in lg[-5:]) / 5,
                                     "clip_engaged_steps": sum(x["grad_norm_pre_clip"] > 1.0 for x in lg)}
        res["regimes"][regime] = R
        print(regime, {k: (round(v["CL_S_C80"], 3), round(v["OL_S_C80_same_stream"], 3), round(v["CL_cos_C80"], 3), round(v["CL_U"], 3)) for k, v in R["at"].items()}, flush=True)
    json.dump(res, open(q.out, "w"), indent=1)
    print("ZZCLOSEDSTATS_OK")


if __name__ == "__main__":
    main()
