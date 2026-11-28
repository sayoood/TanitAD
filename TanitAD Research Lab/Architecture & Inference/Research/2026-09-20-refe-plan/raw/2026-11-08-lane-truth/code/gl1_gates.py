#!/usr/bin/env python3
"""Step 1+2: G-L1 (as written) / G-L2 on sets labelled in BOTH control arms (intersection); CHAIN3 variants. No model output, no correlation."""
import json, lzma, numpy as np
W = "F:/scratch/lane_truth_work/run"; OUT = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/lane_truth"
LEAF = "center_line.CenterLine.info"
BP = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-lane-discipline/heldout_teacher_lane_labels.jsonl.xz"
soft = lambda d: np.clip(1.0 - (np.asarray(d, np.float64) - 0.5) / 1.5, 0, 1)
dvec = lambda d: np.array([np.nan if p.get(LEAF) is None else float(p[LEAF]) for p in d["props"]])
fk = lambda d: (tuple(d["key"]), d["ckpt_step"])
def rd(p): return {fk(d): dvec(d) for d in map(json.loads, open(p, encoding="utf-8"))}
banked = {fk(d): dvec(d) for d in map(json.loads, lzma.open(BP, "rt"))}
q = [json.loads(l) for l in open(W + "/q_ctrl2_real/props_r0_010075_0000000001.jsonl", encoding="utf-8")]
qkeys = [fk({"key": [r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))], "ckpt_step": int(r["ckpt_step"])}) for r in q]
def cmp(a, keys, label):
    dd = np.concatenate([np.abs(a[k] - banked[k]) for k in keys]); ds = np.concatenate([np.abs(soft(a[k]) - soft(banked[k])) for k in keys])
    unequal = [[k[0][0], k[0][1], int(j), float(a[k][j]), float(banked[k][j])] for k in keys for j in np.where(~(np.abs(a[k] - banked[k]) <= 1e-3))[0]]
    f1, f2 = float((dd <= 1e-3).mean()), float((ds <= 0.01).mean())
    return {"label": label, "n_sets": len(keys), "n_props": int(dd.size), "frac_abs_d_le_1e-3": f1, "frac_soft_le_0.01": f2, "max_abs_d": float(np.nanmax(dd)),
            "n_unequal": len(unequal), "n_nan_props": int(sum(np.isnan(a[k]).sum() for k in keys)), "unequal_props": unequal, "PASS_as_written": bool(f1 >= 0.99 and f2 >= 0.99)}
ph, real = rd(W + "/out_ctrl2_ph/teacher_lane_r0.jsonl"), rd(W + "/out_ctrl2_real/teacher_lane_r0.jsonl")
inter = [k for k in qkeys if k in ph and k in real]
failed_ph = [k for k in qkeys if k not in ph]; failed_real = [k for k in qkeys if k not in real]
dr = np.concatenate([np.abs(ph[k] - real[k]) for k in inter])
res = {"queue_rows": len(q), "written_ph": len(ph), "written_real": len(real), "intersection_n": len(inter), "failed_keys_ph": [[k[0][0], k[0][1], k[0][2]] for k in failed_ph],
       "failed_keys_real": [[k[0][0], k[0][1], k[0][2]] for k in failed_real], "failed_same_set": failed_ph == failed_real,
       "G_L1_ph": cmp(ph, inter, "placeholder arm"), "G_L1_real": cmp(real, inter, "real-teacher arm"),
       "G_L2": {"n_props": int(dr.size), "frac_abs_d_le_1e-6": float((dr <= 1e-6).mean()), "max_abs_d": float(np.nanmax(dr)), "PASS": bool((dr <= 1e-6).all())}}
res["variants"] = {}
for nm in ("nocache", "norepair"):
    v = rd(f"{W}/out_ctrl3_{nm}/teacher_lane_r0.jsonl"); ks = [k for k in qkeys[:8] if k in v]
    res["variants"][nm] = dict(cmp(v, ks, nm), rows_attempted=8, failed=[[k[0][0], k[0][1]] for k in qkeys[:8] if k not in v])
    kk = [k for k in ks if k in ph]; res["variants"][nm]["baseline_ph_on_same_sets"] = cmp(ph, kk, "baseline ph, same sets") if kk else None
json.dump(res, open(OUT + "/gl1_gates.json", "w"), indent=1, default=float)
for k in ("queue_rows", "written_ph", "written_real", "intersection_n", "failed_keys_ph", "failed_same_set", "G_L2"): print(k, res[k])
for k in ("G_L1_ph", "G_L1_real"): print(k, {a: b for a, b in res[k].items() if a != "unequal_props"})
for nm, v in res["variants"].items(): print(nm, {a: b for a, b in v.items() if a not in ("unequal_props", "baseline_ph_on_same_sets")}, "\n  baseline:", {a: b for a, b in (v["baseline_ph_on_same_sets"] or {}).items() if a != "unequal_props"})
