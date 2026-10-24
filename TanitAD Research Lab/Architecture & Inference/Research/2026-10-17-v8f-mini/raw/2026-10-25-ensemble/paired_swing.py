"""Zero-GPU: the floor of a SEED-PAIRED contrast. If seed luck is common to arms that share a seed (H-V8F-SEED-1), a delta
such as G_0 - M2b_0 cancels it, and |M2b_0 - M2b_1| overstates its noise. The robust floor is how much one seed-paired
contrast swings between seeds: |c_0 - c_1| with c_s = M2b_s - M1_s (the only two-arm, two-seed pair banked on this rig).
EXPLORATORY: decides nothing."""
import json
import numpy as np
L = "C:/Users/Admin/box2_ship/logs/v8f_mini/eval/"
R = {k: json.load(open(L + f"eval_{k}.json", encoding="utf-8"))["raw"] for k in ("M2b_s0", "M2b_s1", "M1_s0", "M1_s1")}
A = {k: np.load(L + f"eval_{k}.npz")["ade6"].mean() for k in R}
out = {}
rows = [("ade6", {k: float(A[k]) for k in R})] + [(m, {k: R[k]["families"]["longitudinal"][m] for k in R})
                                                 for m in ("speed_mae_mps", "along_mae_m", "accel_mae_mps2")]
for m, v in rows:
    c0, c1 = v["M2b_s0"] - v["M1_s0"], v["M2b_s1"] - v["M1_s1"]
    out[m] = {"c0": c0, "c1": c1, "swing": abs(c0 - c1), "seed_gap_M2b": abs(v["M2b_s0"] - v["M2b_s1"]), "seed_gap_M1": abs(v["M1_s0"] - v["M1_s1"])}
    print(f"{m:16s} c0 {c0:+.4f} c1 {c1:+.4f} swing {abs(c0-c1):.4f} | seed gap M2b {out[m]['seed_gap_M2b']:.4f} M1 {out[m]['seed_gap_M1']:.4f}")
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
