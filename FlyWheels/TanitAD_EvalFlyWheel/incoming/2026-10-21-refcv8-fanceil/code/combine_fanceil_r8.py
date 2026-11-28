#!/usr/bin/env python
"""combine_fanceil_r8.py -- assemble FANCEIL_R8.json (the deliverable) from the per-roll artifacts, by READING them (no new
numbers are computed here except the s0-vs-s1 inference-seed spread of the headline quantities, which is a difference of two
banked full-set means and is labelled as such)."""
import json
import os
import sys

D = "C:/Users/Admin/refcv8_pf/fanceil"
R7 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/raw/fanctl_x8/"
      "ctl_refcv7_50400_fan_s{s}.json")


def j(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def head(f):
    P = f["panel_2s"]
    g = lambda d: {k: d.get(k) for k in ("mean", "delta", "lo", "hi", "separated") if k in d}
    A = f["refcv8_additions_2s"]
    S = f["panel_s6"]
    return {"selected": g(P["selected"]), "oracle_in_fan_T0": g(P["oracle_in_fan"]),
            "random_pick_expectation_T0": g(P["random_expectation"]),
            "random_pick_12_draws_mean_T0": P["draws_summary"]["random_pick_mean_over_draws"],
            "random_pick_12_draws_min_max_T0": P["draws_summary"]["random_pick_min_max"],
            "oracle_minus_random_T0": g(P["best_minus_random_exp"]),
            "random_minus_selected_T0": g(P["random_exp_minus_selected"]),
            "selected_minus_oracle_T0": g(P["sel_gap__selected_minus_best"]),
            "skill_fraction_T0": P["skill_fraction"]["value"],
            "oracle_in_reach_kept_T0": g(A["reach_kept"]["oracle_in_reach_kept"]),
            "n_reach_kept_per_window": A["reach_kept"]["n_kept_per_window"],
            "selected_rank_by_ade": A["selected_rank_by_ade"],
            "base117_vs_extra32": {k: (g(v) if isinstance(v, dict) else v) for k, v in A["base_vs_extra"].items()},
            "lon_lat_split_T0": {"selected_along": g(A["lon_lat_split_2s"]["selected"]["along_m"]),
                                 "selected_cross": g(A["lon_lat_split_2s"]["selected"]["cross_m"]),
                                 "oracle_along": g(A["lon_lat_split_2s"]["oracle_in_fan"]["along_m"]),
                                 "oracle_cross": g(A["lon_lat_split_2s"]["oracle_in_fan"]["cross_m"]),
                                 "selected_minus_oracle_along": g(A["lon_lat_split_2s"]["selected_minus_oracle_along"]),
                                 "selected_minus_oracle_cross": g(A["lon_lat_split_2s"]["selected_minus_oracle_cross"])},
            "reference_plans_2s": {k: {"ade": g(v["ade"]), "minus_selected": g(v["minus_selected"])}
                                   for k, v in P.get("reference_plans", {}).items()},
            "alt_pickers_2s": {k: {"ade": g(v["ade"]), "minus_selected": g(v["minus_selected"]),
                                   "agree_with_sel_frac": v["agree_with_sel_frac"]} for k, v in P.get("extra_pickers", {}).items()},
            "S6_1to6s": {"n_windows": S["n_windows"], "selected": g(S["selected"]), "oracle_in_fan_T0": g(S["oracle_in_fan"]),
                         "random_expectation_T0": g(S["random_expectation"]),
                         "selected_minus_oracle_T0": g(S["sel_gap__selected_minus_best"]),
                         "skill_fraction_T0": S["skill_fraction"]["value"],
                         "oracle_in_reach_kept_T0": g(S["oracle_in_reach_kept"]),
                         "control_vs_banked_s6_os": S["control_selected_vs_banked_s6_os"]}}


def main():
    out = {"what": "D-REFCV8-FANCEIL: the fan ceiling of refcv8-r101-s0 EMA-48,852 (149-candidate diffusion fan) on PhysicalAI "
                   "eval139 (4,754 windows, 139 episodes), per-frame nav feed (perframe_full), the battery's code path",
           "tier": "T0 DIAGNOSTIC for oracle / random / every difference containing them (GT used); `selected` = the os pick "
                   "(T1 stamp, ruling open)",
           "estimator": "FULL-SET mean; episode-cluster bootstrap (single) / PAIRED episode-cluster bootstrap (differences); "
                        "taniteval/taniteval/ci.py n_boot 2000 seed 0 (fan_control_x8.fan_panel imported unchanged)",
           "ci_answers": "ONLY another draw of EPISODES. One TRAINING seed. The INFERENCE-seed variance is read by the s0 vs s1 "
                         "rolls (two fresh DDIM draws), not by the CI.",
           "evidence_class": "MEASURED (artifacts listed under runs)", "runs": {}}
    for tag, lab in (("s1", "EMA-48,852 inference seed 1 (alignment-controlled vs the banked battery s1)"),
                     ("s0", "EMA-48,852 inference seed 0, os-only (a FRESH draw: the banked s0 was a 3-row full-arm roll)"),
                     ("dec_s1", "theta_DEC (WiSE mod2 DEC, md5 dad4cb5b) inference seed 1, os-only (no per-frame bank of its own)")):
        f = j(f"{D}/FANCEIL_R8_{tag}.json")
        if f is None:
            out["runs"][tag] = {"label": lab, "status": "NOT RUN / not finished"}
            continue
        out["runs"][tag] = {"label": lab, "artifact": f"{D}/FANCEIL_R8_{tag}.json", "controls": f["controls"],
                            "headline_2s": head(f), "fan_roll": {k: f.get("fan_roll", {}).get(k) for k in
                                                                 ("wall_s", "cuda_max_memory_allocated_gib", "host", "torch",
                                                                  "ckpt", "inference_seed", "os_only", "traj_equals_sidecar_nav_true")}}
        pk = j(f"{D}/PICKERS_R8_{tag}.json")
        if pk:
            out["runs"][tag]["gt_free_pickers_POSTHOC"] = {"artifact": f"{D}/PICKERS_R8_{tag}.json",
                                                           "mbr_cell": pk["mbr_chosen_on_A"],
                                                           "pickers": {k: {kk: (vv if not isinstance(vv, dict) else
                                                                                {x: vv.get(x) for x in ("mean", "delta", "lo", "hi", "separated") if x in vv})
                                                                           for kk, vv in v.items()} for k, v in pk["pickers"].items()}}
        ts = j(f"{D}/TACSPLIT_R8_{tag}.json")
        if ts:
            out["runs"][tag]["tactical_proxy_S6_T0"] = {"artifact": f"{D}/TACSPLIT_R8_{tag}.json",
                                                        **{k: ts[k] for k in ("n_s6_windows", "gt_class_counts", "on_GT_turns_only",
                                                                              "strategic_family", "defn")},
                                                        **{k: {x: ts[k].get(x) for x in ("mean", "lo", "hi")} for k in
                                                           ("selected_class_matches_gt", "s6_oracle_class_matches_gt_T0",
                                                            "fan_covers_gt_class_T0")}}
        rp = j(f"{D}/R7R8_PAIRED_{tag}.json")
        if rp:
            out["runs"][tag]["paired_vs_refcv7_50400_same_seed"] = {"artifact": f"{D}/R7R8_PAIRED_{tag}.json",
                                                                     "gt_control_max_abs_m": rp["control_gt_max_abs_diff_m"],
                                                                     "per_model": rp["per_model"],
                                                                     "paired_r8_minus_r7": {k: {x: v.get(x) for x in ("delta", "lo", "hi", "separated")}
                                                                                            for k, v in rp["paired_r8_minus_r7"].items()}}
    dv = j(f"{D}/DEC_VS_EMA_PAIRED_s1.json")
    if dv and "dec_s1" in out["runs"]:
        out["runs"]["dec_s1"]["paired_dec_minus_ema_same_seed1"] = {
            "artifact": f"{D}/DEC_VS_EMA_PAIRED_s1.json", "gt_control_max_abs_m": dv["control_gt_max_abs_diff_m"],
            "per_model": dv["per_model"],
            "paired_B_minus_A": {k: {x: v.get(x) for x in ("delta", "lo", "hi", "separated")}
                                 for k, v in dv["paired_r8_minus_r7"].items()},
            "A": "refcv8 EMA-48,852 s1", "B": "theta_DEC s1"}
    r7 = {}
    for s in (0, 1):
        f = j(R7.format(s=s))
        if f:
            P = f["panel"]
            r7[f"s{s}"] = {"selected": P["selected"]["mean"], "oracle_in_fan": P["oracle_in_fan"]["mean"],
                           "random_expectation": P["random_expectation"]["mean"], "skill_fraction": P["skill_fraction"]["value"],
                           "selected_minus_oracle": P["sel_gap__selected_minus_best"]["delta"], "n_candidates": P["n_candidates"]}
    out["refcv7_50400_reference"] = {"artifact": R7.format(s="{0,1}"), "values": r7}
    a, b = out["runs"].get("s0", {}), out["runs"].get("s1", {})
    if "headline_2s" in a and "headline_2s" in b:
        out["inference_seed_spread_s0_minus_s1"] = {
            k: round(a["headline_2s"][k]["mean"] - b["headline_2s"][k]["mean"], 4)
            for k in ("selected", "oracle_in_fan_T0", "random_pick_expectation_T0")}
        out["inference_seed_spread_s0_minus_s1"]["note"] = ("difference of two full-set means from two fresh DDIM draws of the "
                                                            "same checkpoint; NOT a CI")
    json.dump(out, open(f"{D}/FANCEIL_R8.json", "w", encoding="utf-8"), indent=1)
    print("ZZCOMBINEDZZ", sorted(out["runs"]))


if __name__ == "__main__":
    sys.exit(main())
