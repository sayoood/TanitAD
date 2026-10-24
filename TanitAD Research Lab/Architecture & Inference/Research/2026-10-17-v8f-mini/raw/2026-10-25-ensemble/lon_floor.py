"""Zero-GPU: are GRID's and HIST's separated longitudinal gains larger than M2b's own training-seed gap on the same metrics?
F_m = |M2b_s0 - M2b_s1| per metric (the construction of the PRIMARY's F). EXPLORATORY: decides nothing."""
import json
L = "C:/Users/Admin/box2_ship/logs/"
P = {"B0": "v8f_mini/eval/eval_M2b_s0.json", "B1": "v8f_mini/eval/eval_M2b_s1.json",
     "G0": "v8f_grid/eval/eval_M2b-8x20_s0.json", "H0": "v8f_hist/eval/eval_M2b-h10_s0.json"}
J = {k: json.load(open(L + v, encoding="utf-8"))["raw"]["families"]["longitudinal"] for k, v in P.items()}
out = {}
for m in ("speed_mae_mps", "along_mae_m", "accel_mae_mps2"):
    v = {k: J[k][m] for k in J}
    F = abs(v["B0"] - v["B1"])
    out[m] = {**v, "F": F, "G0-B0": v["G0"] - v["B0"], "H0-B0": v["H0"] - v["B0"], "G0-B1": v["G0"] - v["B1"], "H0-B1": v["H0"] - v["B1"]}
    print(f"{m:16s} B0 {v['B0']:.4f} B1 {v['B1']:.4f} F {F:.4f} | G0-B0 {v['G0']-v['B0']:+.4f} H0-B0 {v['H0']-v['B0']:+.4f} | G0-B1 {v['G0']-v['B1']:+.4f} H0-B1 {v['H0']-v['B1']:+.4f}")
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
