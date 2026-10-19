"""summarise_diag2.py -- raw/diag2_arms.json -> raw/diag2_summary.json (replay agreement + redraw spread)."""
import json, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent
A = json.load(open(H / "raw" / "diag2_arms.json", encoding="utf-8"))["arms"]
g0 = json.load(open("E:/refcv8_eval_kit/battery/step5000/g0.json", encoding="utf-8"))
gp = g0["wrapper_control"]["conditions"]["P3_fp32_det"]["analysis"]["terms"]
rel = lambda a, b: abs(a - b) / max(abs(a), 1e-12)
ref = A["rec_plain"]["scalars"]
out = {"replay_vs_rec_plain": {}, "redraw": {}}
for n in ("rp_micro13", "rp_micro31", "rp_perm"):
    s = A[n]["scalars"]
    terms = [k for k in ref if k in s and abs(ref[k]) > 1e-6 and k not in ("goal_gate_grad", "r8_spd_crc")]
    d = {k: rel(ref[k], s[k]) for k in terms}
    out["replay_vs_rec_plain"][n] = {"n_terms": len(terms), "max_rel": max(d.values()),
                                     "worst3": sorted(((v, k) for k, v in d.items()), reverse=True)[:3],
                                     "cascade": d.get("cascade"), "r8_subscore": d.get("r8_subscore"), "traj": d.get("traj"),
                                     "per_row_outputs_max_rel": max(r for _, r in A[n]["vs_ref_top"]),
                                     "served": A[n]["served"]}
for n in ("rp_row0", "rp_row1", "rp_row2", "rp_row3"):
    out["replay_vs_rec_plain"][n] = {"per_row_outputs_max_rel": max(r for _, r in A[n]["vs_ref_top"]), "served": A[n]["served"]}
seeds = ["rec_plain"] + [f"live_plain_s{s}" for s in range(1, 8)]
for k in ("cascade", "r8_subscore", "traj", "r8_alloc_l1", "sel_v3", "loss"):
    v = [A[n]["scalars"][k] for n in seeds]
    devs = [rel(v[0], x) for x in v[1:]]
    out["redraw"][k] = {"values_seed0to7": v, "mean": st.mean(v), "sd": st.stdev(v), "rel_sd": st.stdev(v) / abs(st.mean(v)),
                        "rel_dev_from_seed0_max": max(devs), "rel_dev_from_seed0_median": st.median(devs),
                        "g0_probe_P3_cuda": {x: gp[k][x] for x in ("wrapper", "floor", "bar")} if k in gp else None}
json.dump(out, open(H / "raw" / "diag2_summary.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=0)[:5000])
