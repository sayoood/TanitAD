#!/usr/bin/env python3
"""G-L1b per AMENDMENT_2 section 3. No model output; no correlation with the head. Sets = the intersection of both control arms (34).
rho = Spearman (average ranks) of soft(d) vs v4 = sel_common.step_speed(raw proposal, idx 19); valid = proposal with finite d in rebuilt and banked."""
import json, sys, lzma, numpy as np
from scipy.stats import rankdata
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107")
import sel_common as sc
W = "F:/scratch/lane_truth_work/run"; OUT = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/lane_truth"
LEAF = "center_line.CenterLine.info"
BP = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-lane-discipline/heldout_teacher_lane_labels.jsonl.xz"
soft = lambda d: np.clip(1.0 - (np.asarray(d, np.float64) - 0.5) / 1.5, 0, 1)
dvec = lambda d: np.array([np.nan if p.get(LEAF) is None else float(p[LEAF]) for p in d["props"]])
fk = lambda d: (tuple(d["key"]), d["ckpt_step"])
banked = {fk(d): dvec(d) for d in map(json.loads, lzma.open(BP, "rt"))}
mine = {fk(d): dvec(d) for d in map(json.loads, open(W + "/out_ctrl2_ph/teacher_lane_r0.jsonl", encoding="utf-8"))}
real = {fk(d) for d in map(json.loads, open(W + "/out_ctrl2_real/teacher_lane_r0.jsonl", encoding="utf-8"))}
rows = []
for r in map(json.loads, open(W + "/q_ctrl2_ph/props_r0_010075_0000000001.jsonl", encoding="utf-8")):
    k = ((r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))), int(r["ckpt_step"]))
    if k in mine and k in real:
        rows.append((k, np.asarray(r["props"], np.float64)))
def rho(t, v):
    a, b = rankdata(t), rankdata(v)
    return None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
per = []
for k, P in rows:
    v4 = sc.step_speed(P, 19); m = np.isfinite(mine[k]) & np.isfinite(banked[k])
    if m.sum() < 5: per.append({"set": [k[0][0], k[0][1]], "entered": False, "reason": "<5 valid"}); continue
    a, b = rho(soft(mine[k][m]), v4[m]), rho(soft(banked[k][m]), v4[m])
    per.append({"set": [k[0][0], k[0][1]], "n_valid": int(m.sum()), "entered": a is not None and b is not None, "rho_rebuilt": a, "rho_banked": b})
e = [p for p in per if p["entered"]]
diff = np.array([p["rho_rebuilt"] - p["rho_banked"] for p in e])
mabs, mdiff = float(np.abs(diff).mean()), float(diff.mean())
res = {"n_sets": len(rows), "n_entered": len(e), "mean_abs_diff": mabs, "abs_mean_diff": abs(mdiff), "mean_diff": mdiff, "max_abs_diff": float(np.abs(diff).max()),
       "mean_rho_rebuilt": float(np.mean([p["rho_rebuilt"] for p in e])), "mean_rho_banked": float(np.mean([p["rho_banked"] for p in e])),
       "bars": {"mean_abs<=0.05": mabs <= 0.05, "abs_mean<=0.03": abs(mdiff) <= 0.03, "n_entered>=20": len(e) >= 20},
       "PASS": bool(mabs <= 0.05 and abs(mdiff) <= 0.03 and len(e) >= 20), "per_set": per}
json.dump(res, open(OUT + "/gl1b.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "per_set"})
