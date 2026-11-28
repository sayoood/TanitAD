#!/usr/bin/env python3
"""Build onpolicy_relabel_teacher_lane queue rows (kind onpolicy_props) for the C3 tokens from u400's raw proposals (traj_all_A0).
teacher = PLACEHOLDER (the banked pick's trajectory): it only sets the EP denominator, never the centre-line leaf (gate G-L2 tests this).
Also builds the control queue: the 24 real queue rows (smoke/oldq) with the teacher replaced by the placeholder (prop 0)."""
import json, sys, os
import numpy as np
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107")
import sel_common as sc
sc.C3 = "F:/scratch/lane_truth_work/run/c3tab"
W = "F:/scratch/lane_truth_work/run"
D = sc.load()
tok, log, cru, pick, traj = D["token"], D["log"], D["cruise"], D["pick"], D["traj"]
def r5(a): return [[[round(float(v), 5) for v in p] for p in prop] for prop in a]
def rows(idx, d):
    os.makedirs(d, exist_ok=True)
    with open(d + "/props_r0_000000_0000000001.jsonl", "w", encoding="utf-8") as f:
        for i in idx:
            f.write(json.dumps({"kind": "onpolicy_props", "log_name": str(log[i]), "token": str(tok[i]), "step": 0, "rank": 0, "ckpt_step": 0,
                                "teacher": r5([traj[i, pick[i]]])[0], "aug": None, "props": r5(traj[i])}) + "\n")
ci = np.where(cru)[0]; oi = np.where(~cru)[0]
rows(ci, W + "/q_cruise"); rows(oi, W + "/q_other")
print("cruise", len(ci), "other", len(oi), "logs cruise", len(set(log[ci])))
# control
src = "E:/Projects/TanitAD-artifacts/_tlane_work/smoke/oldq/props_r0_010075_0000000001.jsonl"
os.makedirs(W + "/q_ctrl_ph", exist_ok=True)
n = 0
with open(W + "/q_ctrl_ph/props_r0_010075_0000000001.jsonl", "w", encoding="utf-8") as f:
    for ln in open(src, encoding="utf-8"):
        r = json.loads(ln); r["teacher"] = r["props"][0]; f.write(json.dumps(r) + "\n"); n += 1
print("control rows", n)
