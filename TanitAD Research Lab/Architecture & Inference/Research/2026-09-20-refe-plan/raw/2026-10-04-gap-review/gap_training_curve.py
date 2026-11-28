#!/usr/bin/env python3
"""REVIEW 7 -- per-epoch training curve of the final REFe run (metrics.jsonl, read-only). CPU, stdlib only.
NOTE: these are TRAINING-side quantities (trainer curves, not eval results)."""
import json, math, collections
SRC = r"D:/Projects/TanitAD-artifacts/refe-final-2026-10-01/metrics.jsonl"
OUT = r"E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-gap-review/gap_training_curve.json"
ep = collections.defaultdict(lambda: {"n": 0, "traj": 0.0, "score": 0.0, "cov": 0.0, "lr": 0.0, "op_n": 0, "op_skill": 0.0, "op_pick": 0.0, "op_best": 0.0, "op_lag": 0.0})
last_epoch = 0
for line in open(SRC):
    r = json.loads(line)
    if "traj_L1" in r and isinstance(r.get("traj_L1"), (int, float)) and not math.isnan(r["traj_L1"]):
        e = r["epoch"]; last_epoch = e; d = ep[e]
        d["n"] += 1; d["traj"] += r["traj_L1"]; d["score"] += r.get("score", 0.0) or 0.0
        d["cov"] += r.get("scorer_cov", 0.0) or 0.0; d["lr"] += r.get("lr", 0.0)
    elif r.get("event") == "onpolicy":
        d = ep[last_epoch]; d["op_n"] += 1
        d["op_skill"] += r.get("skill", 0.0); d["op_pick"] += r.get("pick", 0.0); d["op_best"] += r.get("best", 0.0); d["op_lag"] += r.get("lag_steps", 0.0)
out = {}
for e in sorted(ep):
    d = ep[e]
    if d["n"] == 0:
        continue
    out[str(e)] = {"steps": d["n"], "traj_L1": round(d["traj"] / d["n"], 4), "score_loss": round(d["score"] / d["n"], 4),
                   "scorer_cov": round(d["cov"] / d["n"], 4), "lr_mean": d["lr"] / d["n"]}
    if d["op_n"]:
        out[str(e)].update({"train_side_onpolicy_pick": round(d["op_pick"] / d["op_n"], 4), "best": round(d["op_best"] / d["op_n"], 4),
                            "skill": round(d["op_skill"] / d["op_n"], 4), "lag_steps": round(d["op_lag"] / d["op_n"], 1)})
json.dump(out, open(OUT, "w"), indent=1)
for k, v in out.items():
    print(k, v)
