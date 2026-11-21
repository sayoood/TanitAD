#!/usr/bin/env python3
"""closed-loop halves of the identity controls: (a3) FULL K=25 vs banked closed_P_1.0.pt D[25]; (b) EMPTY closed loop, K=3; (c) mutation closed loop.  -> controls_closed.json"""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda n: pickle.load(open(os.path.join(HERE, "closed_out", n), "rb"))
FR = pickle.load(open(os.path.join(HERE, "full_ref.pkl"), "rb"))
res = {}
z = L("closed_s0_P_FULL_R_lr1_lam1_ident.pkl")["identity_vs_banked"][25]
res["a3_closed_FULL_K25_vs_banked"] = {**z, "tolerance_rel_l2": 1e-3, "pass": bool(z["rel_l2_diff"] <= 1e-3)}
e1, e0 = L("closed_s0_P_EMPTY_R_lr1_lam1_b3.pkl")["snaps"][3], L("closed_s0_P_EMPTY_R_lr1_lam0_b3.pkl")["snaps"][3]
dj = {p: float((e1["eval"][p]["J"] - e0["eval"][p]["J"]).mean()) for p in ("train", "val")}
eq = {p: float(max(np.abs(e1["eval"][p]["J"] - FR["theta0"][p]["J"]).max(), np.abs(e0["eval"][p]["J"] - FR["theta0"][p]["J"]).max())) for p in ("train", "val")}   # informational: J(theta_0) was computed in another process (3 vs 6 threads) => ~1e-7 float noise, not a restriction defect
res["b_closed_EMPTY_K3"] = {"disp_norm_lam1": e1["disp_norm"], "disp_norm_lam0": e0["disp_norm"], "delta_J": dj, "info_max_abs_J_diff_vs_theta0_other_process": eq,
                            "same_process_J_lam1_equals_J_lam0_bitwise": {p: bool(np.array_equal(e1["eval"][p]["J"], e0["eval"][p]["J"])) for p in ("train", "val")},
                            "pass_rule": "PREREG s.7(b): max|theta-theta0| == 0.0 and Delta J == 0.0 with ==",
                            "pass": bool(e1["disp_norm"] == 0.0 and e0["disp_norm"] == 0.0 and all(v == 0.0 for v in dj.values()))}
m = L("closed_s0_P_PH_C_HEAD_R_lr1_lam1_mut.pkl")["snaps"][3]
res["c_closed_mutation_K3"] = {"outside_group_max_abs_displacement_mutated": m["outside"], "check_outside_eq_0.0_mutated": bool(m["outside"] == 0.0), "reads_RED": bool(m["outside"] > 0.0), "pass": bool(m["outside"] > 0.0)}
json.dump(res, open(os.path.join(HERE, "controls_closed.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
