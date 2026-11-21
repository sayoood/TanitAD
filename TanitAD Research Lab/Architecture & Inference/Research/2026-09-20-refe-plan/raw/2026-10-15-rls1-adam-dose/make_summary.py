#!/usr/bin/env python3
"""Assemble RESULTS_SUMMARY.json from the measurement JSONs of this folder (no number is retyped).  python make_summary.py"""
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
L = lambda n: json.load(open(os.path.join(H, n), encoding="utf-8"))  # noqa: E731
fv, fs, dr, sim, cl, b0, lp, ab, ct = (L(n) for n in ("face_validity.json", "face_validity_swap.json", "disp_real.json", "adam_sim.json", "closed_stats.json",
                                                      "closed_beta0_stats.json", "late_probe.json", "ab_vs_base.json", "controls_trainer.json"))
S = sim["streams"]


def agg(regime, k, key="S_C80"):
    v = np.array([S[s]["modes"][regime]["at"][k][key] for s in S if k in S[s]["modes"][regime]["at"]])
    return {"mean": float(v.mean()), "sd": float(v.std()), "per_stream": [float(x) for x in v]}


out = {
    "tool": "make_summary/1", "prereg": "PREREG_H_ADAM.md (+ PREREG_H_ADAM.sha256)",
    "verdict": "H-ADAM FAILS as specified (S_P LOW; realised rho MID is a closed-loop effect). The (LOW, MID) cell was not enumerated in the committed table; both enumerated LOW-mechanism rows read FAILS-as-mechanism (Amendment 2).",
    "face_validity": {"V": fv["V_median_over_heads_of_mean_within_scene_r"], "status": fv["status"], "n_val_frames": fv["n_val_scenes"], "swapped_ctx_V": fs["swapped_ctx_shift1"]["V_median_over_heads"],
                      "mean_abs_logit_diff_true_vs_swapped": [fs["true_ctx"]["mean_abs_logit_diff_all"], fs["swapped_ctx_shift1"]["mean_abs_logit_diff_all"]],
                      "D1_BCE_pool_vs_logged": [fv["D1_mean_BCE_base_on_pool"], fv["D1_logged_R0_bceA_u25"]], "D2_J_pool_vs_logged": [fv["D2_mean_J_base_on_pool"], fv["D2_logged_R0_epoJ_u25"]],
                      "D3_pool_vs_partition": [fv["D3_pool"], fv["D3_RL_train_partition"]], "pool_n": fv["train_pool"]["n_pool"]},
    "open_loop_S": {"P25": agg("P", "25"), "P50": agg("P", "50"), "P100": agg("P", "100"), "T50": agg("T", "50"), "T250": agg("T", "250"),
                    "U_P50": agg("P", "50", "U_full_norm_ratio"), "U_T50": agg("T", "50", "U_full_norm_ratio"), "cos_C80_P50": agg("P", "50", "cos_C80")},
    "gradient_ratio_r_stream0": S["0"]["r"]["total"],
    "clip": {rg: S["0"]["modes"][rg]["clip"] for rg in ("P", "T")},
    "controls": {"adam_sim": sim["controls"], "trainer": {k: v for k, v in ct.items() if k.startswith("C")}},
    "closed_loop": {rg: {k: {x: v[x] for x in ("CL_S_C80", "OL_S_C80_same_stream", "CL_cos_C80", "CL_U")} for k, v in R["at"].items()} for rg, R in cl["regimes"].items()},
    "closed_loop_beta0_regimeP": b0["at"],
    "realised_displacement": {u: {k: v for k, v in dr["updates"][u].items() if k in ("rho_floor_corrected_C80", "rho_uncorrected_C80", "rho_floor_corrected_all_tensors", "cos_R0b_vs_R0_EPOdisp_C80",
                                                                                         "projection_of_R0b_on_R0_norm_units", "n_C80_tensors", "global_norms")} for u in dr["updates"]},
    "realised_displacement_groups_u500": {g: {"share_of_R0_EPO_energy": v["share_of_R0_EPO_energy"], "rho_group": v["rho_group"]} for g, v in dr["updates"]["500"]["groups"].items()},
    "late_probe": {k: {x: v[x] for x in ("r_total_lam1", "mean_J", "mean_BCE", "mean_KL_to_base", "V_vs_banked_RLval")} for k, v in lp["states"].items()},
    "g5_gap_decomposition_x100": {k: {c: {q: [row[c][q]["mean"], row[c][q]["ci95"][0], row[c][q]["ci95"][1]] for q in ("TTC", "PDMS")} for c in row} for k, row in ab["rows"].items()},
}
json.dump(out, open(os.path.join(H, "RESULTS_SUMMARY.json"), "w"), indent=1)
print("ZZSUMMARY_OK", {k: (round(v["mean"], 4) if isinstance(v, dict) and "mean" in v else None) for k, v in out["open_loop_S"].items()})
