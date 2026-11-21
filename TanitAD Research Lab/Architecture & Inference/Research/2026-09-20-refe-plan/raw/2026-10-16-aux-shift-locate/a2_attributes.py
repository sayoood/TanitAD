#!/usr/bin/env python3
"""Per-frame scene attributes for BOTH domains (CPU, read-only):  ego speed, driving command, map, scenario_type (train bank only).
train  : D:/Projects/TanitAD-artifacts/_rlfan_work/bank_full/targets_rank0.jsonl   (the rows the label store was built from; ego[6] = |v|)
navtest: D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz (ego_statuses[-1] = the current frame)
Output: scratch attributes.json = {"train": {"log|token": [speed, scenario_type, source]}, "navtest": {"token": [speed, cmd_idx, map]}}"""
import gzip, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C
t0 = time.time()
tr = {}
with open("D:/Projects/TanitAD-artifacts/_rlfan_work/bank_full/targets_rank0.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        tr[r["log_name"] + "|" + r["token"]] = [r["ego"][6], r.get("scenario_type"), r.get("source"), r.get("rank"), r.get("step")]
print("train rows", len(tr), time.time() - t0, flush=True)
d = json.load(gzip.open("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz", "rt", encoding="utf-8"))["tokens"]
nt = {}
for tok, v in d.items():
    es = v["ego_statuses"][-1]
    vel = es["ego_velocity"]
    cmd = es["driving_command"]
    nt[tok] = [math.hypot(vel[0], vel[1]), int(max(range(len(cmd)), key=lambda i: cmd[i])), v["map_name"], v["log_name"]]
print("navtest rows", len(nt), flush=True)
C.jdump({"train": tr, "navtest": nt}, os.path.join(C.SCRATCH, "attributes.json"))
from collections import Counter
print("train scenario_type top:", Counter(v[1] for v in tr.values()).most_common(8))
print("train source:", Counter(v[2] for v in tr.values()).most_common(5))
print("DONE", time.time() - t0)
