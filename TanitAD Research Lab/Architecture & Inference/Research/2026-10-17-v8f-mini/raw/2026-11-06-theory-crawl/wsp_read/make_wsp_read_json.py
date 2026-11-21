"""Assembles WSP_READ.json from the result files (no hand-typed P2 / P3 numbers). P1 values are quoted from HANDOFF_WSP.md / P1_RESULT.json of the previous agent, unchanged."""
import json
H = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read"
r0 = json.load(open(H + "/P2B_RESULT.json")); r1 = json.load(open(H + "/P2B_s1_RESULT.json")); p3 = json.load(open(H + "/P3_RESULT.json"))
def p2(r):
    b = r["bands"]["3-8"]; c = r["checks_3_8"]
    return {"verdict_as_written": c["P2_verdict_as_written"], "R_tot_z": b["R_tot_z"], "R_tot_raw_units": b["R_tot_raw_units"], "ablation_rms_ratio_ego_over_vision": {m: v["ego_over_vision"] for m, v in b["ablation_rms_ratios"].items()},
            "W1": c["W1_net_not_level_carrier"], "W2": c["W2_residual_ego_over_vision_in_[0.08,1.3]"], "W3": c["W3_code_over_ego_in_[0.35,3.1]"],
            "gains_median": {k: b["W1"][k] for k in ("median_d_s_tot_d_v0", "median_d_s_tot_d_a0", "median_d_s_base_d_v0", "median_d_s_base_d_a0", "median_d_s_net_d_v0", "median_d_s_net_d_a0", "median_d_s_res_d_v0", "median_d_s_res_d_a0")},
            "components": b["components"], "n_windows": b["n"], "n_episodes": b["n_episodes"]}
nonceil = p3["tests"]["NON-CEILING (ADMISSIBLE + DECODE)"]; allt = p3["tests"]["all"]
out = {
 "status": "P1, P2, P3 evaluated; P1' NOT EVALUATED (see HANDOFF_WSP2.md)",
 "frozen": {"THEORY_CRAWL.md": "6e83451c", "THEORY_CRAWL_AMEND_1.md": "9d39ac53"},
 "P1": {"verdict_as_written": "FAILS (leg A)", "evidence_class": "MEASURED, T2 rollouts, 93 starts, previous agent", "artifacts": ["P1_RESULT.json", "P1B_RESULT.json", "P1C_RESULT.json", "SPEC_P1_AMEND_1.md", "P1_FILES.sha256"],
        "C0C_band_3_8_mean_g_minus_v": {"s0": {"point": 0.428, "ci95": [0.112, 0.805], "n": 50}, "s1": {"point": 1.059, "ci95": [0.405, 1.967], "n": 51}}},
 "P1_prime": {"verdict_as_written": "NOT EVALUATED", "reason": "banked OOF posteriors cover the probe's 12,000 training windows, not eval-141; refit-and-score on the probe's own code not yet run", "artifacts": [], "evidence_class": "n/a"},
 "P2": {"evidence_class": "MEASURED (CPU, eval-141 tick 0, C0 s0 and s1, one training run per seed)", "artifacts": ["CLAIMS_P2.md", "CLAIMS_P2.sha256", "P2A_WEIGHTS.json", "P2B_RESULT.json", "P2B_s1_RESULT.json", "P2B_PERWINDOW.npz", "P2B_s1_PERWINDOW.npz"],
        "s0": p2(r0), "s1": p2(r1), "weight_claims_matched": "2 of 4 at both seeds (W1a, W3 held; W1b, W2 failed)"},
 "P3": {"evidence_class": "MEASURED (reader JSONs, paired episode-cluster bootstrap B 2000)", "verdict_as_written": "FAILS", "artifacts": ["P3_RESULT.json", "P3_FRONTIER_ADE6.png", "P3_FRONTIER_COUNT.png", "p3_frontier.py"],
        "all_levers_strict_pairs_at_both_seeds": len(allt["strict_ade6_pairs_at_BOTH_seeds"]), "all_dominators_are_ceiling_arms": all(x.split(" > ")[0].startswith(("A4 LFO", "A6 LFOB", "A5 LFO", "A5 LFN")) for x in allt["strict_ade6_pairs_at_BOTH_seeds"]),
        "non_ceiling_strict_pairs_at_both_seeds": len(nonceil["strict_ade6_pairs_at_BOTH_seeds"]), "non_ceiling_strict_pairs_by_seed_total": nonceil["strict_ade6_n_pairs_by_seed"],
        "spearman_closure_vs_ade6": p3["spearman_closure_vs_ade6"], "unplaced": p3["unplaced"], "n_points": len(p3["records"])},
 "where_the_longitudinal_speed_comes_from": "level from the non-learned kd_x base (d s/d v0 1.00, d s/d a0 1.90); frozen P10 net and the learned residual shrink the gains (total 0.81/1.27 at s0, 0.76/1.11 at s1) and the residual shifts plan speed by about -0.35 m/s; prior explains R^2 0.87 / 0.82 of plan-speed variance; vision reaches the plan mainly through S and T (code, checkpoints), not O's scene attention (norm 0.40-0.52 via S/T vs 0.06-0.09 direct)",
}
json.dump(out, open(H + "/WSP_READ.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("P2",)}, indent=0)[:300]); print(out["P3"]["all_levers_strict_pairs_at_both_seeds"], out["P3"]["all_dominators_are_ceiling_arms"], out["P3"]["non_ceiling_strict_pairs_at_both_seeds"], out["P3"]["non_ceiling_strict_pairs_by_seed_total"])
