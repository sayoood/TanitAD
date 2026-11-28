"""MM 2026-11-20: two checks of the OPTB3 decide against the Amendment 2 decide (same windows, same FT-NULL arms, same planner).
(1) STRATEGIC identity control: rows that do not depend on the new arms must be equal (B = FTNULL_s0, HOLD, the logged path, the text rows).
    Row A is the lever arm of each read (FTRSP_s0 in Amendment 2, FTT15_s0 here), so it is expected to differ.
(2) Floors: F_T, F_I and Floor_k per chassis and quantity in both reads.
usage: python identity_floors_vs_amend2.py <DECIDE_OPTB.json> <DECIDE_OPTB3.json> <out.json>"""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
J = lambda x: json.dumps(x, sort_keys=True)
sa, sb = a["strategic"], b["strategic"]
ident = {
    "B_FTNULL_s0_equal": J(sa["arms"]["B"]) == J(sb["arms"]["B"]),
    "HOLD_equal": J(sa["arms"]["HOLD"]) == J(sb["arms"]["HOLD"]),
    "logged_path_equal": J(sa["logged_path"]) == J(sb["logged_path"]),
    "text_rows_equal": J(sa["text_rows"]) == J(sb["text_rows"]),
    "A_differs_as_expected": J(sa["arms"]["A"]) != J(sb["arms"]["A"]),
    "A_amend2_FTRSP_s0": sa["arms"]["A"],
    "A_optb3_FTT15_s0": sb["arms"]["A"],
}
ident["pass"] = all(ident[k] for k in ("B_FTNULL_s0_equal", "HOLD_equal", "logged_path_equal", "text_rows_equal"))
fl = {}
for c in ("0", "1"):
    for m in ("acc_mae", "yaw_mae"):
        fa, fb = a["floors"][c][m], b["floors"][c][m]
        fl[f"chassis{c}_{m}"] = {
            "amend2": {"F_T": fa["F_T"]["value"], "F_I": fa["F_I"]["value"], "F_I_n_arms": fa["F_I"].get("n_arms_with_both_seeds"), "Floor": fa["Floor"]},
            "optb3": {"F_T": fb["F_T"]["value"], "F_I": fb["F_I"]["value"], "F_I_n_arms": fb["F_I"].get("n_arms_with_both_seeds"), "Floor": fb["Floor"]},
            "Floor_equal": fa["Floor"] == fb["Floor"],
        }
out = {"strategic_identity_control": ident, "floors": fl,
       "note": "F_I takes the max over every arm with both inference seeds, so adding the four FTT arms can only raise it; Floor_k = max(F_T,k, F_I) and F_T dominates at both chassis"}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("identity pass", ident["pass"], "| floors equal", all(v["Floor_equal"] for v in fl.values()))
