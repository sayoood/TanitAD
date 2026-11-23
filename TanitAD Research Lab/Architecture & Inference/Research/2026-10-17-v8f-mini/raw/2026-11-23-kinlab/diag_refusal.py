"""KINLAB diagnostic (NOT a tuning step; no constant is read from it): where does the checker's accel term of a refused_infeasible kin_v1 record sit? On the first N ok records of shard 0 (the same data and tree as census_kinlab2.py).
For each record whose status is refused_infeasible with the accel term set: recompute the label unchecked, take the step of max |chord accel| and compare it with the merge step. usage: python diag_refusal.py <shard> <tracks> <out.json> <tree>"""
import json
import sys
import zipfile

import numpy as np

shard, tpath, outp, tree = sys.argv[1:5]
sys.path.insert(0, tree + "/stack")
import torch  # noqa: E402
from tanitad.data import v8f_recovery as RV  # noqa: E402
sys.path.insert(0, tree + "/../")
from census_kinlab2 import read_member  # noqa: E402,F401  (same reader)

N = 30000
zf = zipfile.ZipFile(shard)
hdr = json.loads(bytes(read_member(zf, "header_json")).decode("utf-8"))
ci, rr, reason = (read_member(zf, k) for k in ("ci", "r", "reason"))
p10 = read_member(zf, "p10f", last_row_only=True)
names = {int(k): v for k, v in hdr["clips"].items()}
tracks = {c: t.double().numpy() for c, t in torch.load(tpath, map_location="cpu", weights_only=True)["tracks"].items()}
ok = np.flatnonzero(reason == 0)[::12][:N]                                 # a stride over the shard: both ticks, many clips
n = {"records": 0, "refused_infeasible": 0, "accel_term": 0}
at_merge = []
for i in ok:
    g = tracks[names[int(ci[i])]][0::2][:, :3]
    own = np.asarray(p10[i, :3], np.float64)
    res = RV.kin_label(g, int(rr[i]), own)
    n["records"] += 1
    if res["status"] != "refused_infeasible":
        continue
    n["refused_infeasible"] += 1
    if not res["viol"]["accel"]:
        continue
    n["accel_term"] += 1
    un = RV.kin_label(g, int(rr[i]), own, check=False)
    km = int(round(un["t_merge"] / RV.DT))
    pts = np.concatenate([[[-own[0] * RV.DT, 0.0], [0.0, 0.0]], un["label"]])
    ac = np.abs(RV.controls(pts)[1])                                          # ac[j] = the accel between chord j and chord j + 1 (chord index 0 = the virtual predecessor step)
    j = int(np.argmax(ac))
    at_merge.append(int(j) - km)                                              # label chord j+1 .. : j + 1 - km = 0 or 1 is the step across the merge point
n["accel_violation_step_minus_merge_step_hist"] = {str(k): int(v) for k, v in zip(*np.unique(at_merge, return_counts=True))}
n["share_max_accel_within_1_step_of_merge"] = float(np.mean(np.abs(np.asarray(at_merge)) <= 1)) if at_merge else None
json.dump(n, open(outp, "w"), indent=1)
print(json.dumps(n, indent=1))
