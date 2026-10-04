"""Do the TEACHER SIMULATOR's own lane signals (paper-pure candidates) pick out the plans NAVSIM's geometry flags?
Joins, per held-out set (key + ckpt_step), the teacher leaves (refe/onpolicy_relabel_teacher_lane.py) with the
NAVSIM-faithful labels (refe/onpolicy_relabel_lane.py): navsim_ddc (oncoming) and lane_keep (lk10). For each teacher
signal: within-set AUC of the teacher signal ranking NAVSIM-clean hypotheses above NAVSIM-violating ones (sets with both
classes), plus the teacher's own human-relative sanity (the teacher's rollout values). CPU, seconds."""
import glob, json, os, sys
import numpy as np
SP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
nav = {}
for f in glob.glob(f"{SP}/heldout_lane/lane_*.jsonl"):
    for l in open(f):
        d = json.loads(l); nav[(tuple(d["key"]), d["ckpt_step"])] = d
tl = {}
for f in glob.glob(f"{SP}/heldout_teacher_lane/teacher_lane_r*.jsonl"):
    for l in open(f):
        d = json.loads(l); tl[(tuple(d["key"]), d["ckpt_step"])] = d
keys = sorted(set(nav) & set(tl))
SIG = {"CenterLine.reward (higher = better)": ("center_line.CenterLine.reward", +1),
       "CenterLine.info distance (lower = better)": ("center_line.CenterLine.info", -1),
       "CrossLane.reward (higher = better)": ("off_road.CrossLane.reward", +1),
       "lane_change event (absent = better)": ("off_road.CrossLane.lane_change_info", -1),
       "wrong-way flag ddc.violation (absent = better)": ("ddc.violation", -1)}
def auc(s, bad):
    ok, b = s[~bad], s[bad]
    if len(ok) == 0 or len(b) == 0: return None
    return float((ok[:, None] > b[None, :]).mean() + 0.5 * (ok[:, None] == b[None, :]).mean())
out = {"sets_joined": len(keys), "signals": {}}
for name, (leaf, sign) in SIG.items():
    res = {}
    for tgt in ("navsim_oncoming", "lane_keep_lk10"):
        a = []
        frac_vary = 0
        for k in keys:
            v = np.array([np.nan if p.get(leaf) is None else p[leaf] for p in tl[k]["props"]], float)
            if np.isnan(v).any(): continue
            s = sign * v
            if v.max() - v.min() > 1e-9: frac_vary += 1
            bad = (np.array(nav[k]["navsim_ddc"]) < 1) if tgt == "navsim_oncoming" else (np.array(nav[k]["lane_keep"]) > 0)
            x = auc(s, bad)
            if x is not None: a.append(x)
        res[tgt] = {"mixed_sets": len(a), "mean_auc": round(float(np.mean(a)), 4) if a else None}
        res["sets_where_signal_varies"] = frac_vary
    vals = np.array([p.get(leaf) for k in keys for p in tl[k]["props"] if p.get(leaf) is not None], float)
    res["distribution"] = {"min": round(float(vals.min()), 3), "p50": round(float(np.median(vals)), 3), "max": round(float(vals.max()), 3)}
    out["signals"][name] = res
tv = np.array([tl[k]["teacher"]["center_line.CenterLine.reward"] for k in keys], float)
out["teacher_rollout_centerline_reward_p50"] = round(float(np.median(tv)), 3)
json.dump(out, open(os.path.join(HERE, "teacher_lane_validation.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
