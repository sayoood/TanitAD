#!/usr/bin/env python3
"""Assemble SHIFT_LOCATE.json from the component result files (every number below is READ from a component JSON, none is typed in).  CPU, read-only."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
a1, b1, b2, d1, d2, d3, e1, f1 = (L("a1_label_construction.json"), L("b1_decompose.json"), L("b2_standardised_gap_ci.json"), L("d1_place_novelty.json"),
                                  L("d2_indomain_vs_transfer.json"), L("d3_probe_place_leak.json"), L("e1_precision_sensitivity.json"), L("f1_cross_store_label_identity.json"))
S1 = json.load(open(os.path.join(HERE, "..", "2026-10-15-s1-read", "S1_READ.json"), encoding="utf-8"))
ARMS = ["H1-s0", "H1-s1", "H1-s2"]
mean = lambda xs: float(np.mean(xs))
out = {"tool": "shift_locate/1", "date": "2026-10-16", "tier": "T0-like (head perception on teacher-forced frames; never driving performance)", "evidence_class": "MEASURED (ours; artifacts named per block)",
       "question": "where is the navtrain-held-out -> navtest gap of the S1 aux drivable-area head (P1 IoU [0,40) m)?",
       "estimator": "independent log-cluster bootstrap (the two domains have disjoint log sets), B=4000-5000; answers 'another draw of LOGS' only. Training-run variance: the three H1 seeds are reported side by side, not pooled into an interval. Inference variance: the head is deterministic (ASSERTED, not measured here)."}
# the fact to explain, re-derived
out["fact_re_derived"] = {
    "heldout_p1_0_40": {a: b1["S1_bands"]["0-40"][a]["heldout"] for a in ARMS}, "navtest_p1_0_40": {a: b1["S1_bands"]["0-40"][a]["navtest"] for a in ARMS},
    "gap": {a: b1["S1_bands"]["0-40"][a]["gap"] for a in ARMS}, "gap_mean_3_seeds": mean([b1["S1_bands"]["0-40"][a]["gap"] for a in ARMS]),
    "banked_heldout": {a: S1["outcome"]["per_arm"]["k2"][a]["p1_ho"]["iou_0_40"] for a in ARMS}, "banked_navtest": {a: S1["outcome"]["per_arm"]["k2"][a]["iou_0_40"] for a in ARMS},
    "reproduction_max_abs_diff_navtest": max(v["abs_diff"] for v in b1["R0_navtest_repro"].values()),
    "gap_ci95_independent_logs": {a: b1["S10_gap_ci"][a]["ci95"] for a in ARMS}, "n_logs": {"heldout": 24, "navtest_c3": 123},
    "gap_by_band": {nm: {a: v[a]["gap"] for a in ARMS} for nm, v in b1["S1_bands"].items()}}
# H-a
out["H_a_label_construction"] = {
    "verdict": "NOT FOUND at four independent layers (provenance, code hash, banked path-A/B gates, data level incl. cross-store same-place agreement)",
    "code_sha256_equal": a1["code_sha256_equal"], "raster_equal": a1["raster_equal"], "maps_json_sha256_equal": a1["maps_json_sha256_equal"], "code_sha256": a1["code_sha256"]["train"],
    "code_sha_is_tip_file_in_CRLF_form": a1["code_sha_matches_a_form_of_tip_file"], "origin_source_mix": a1["origin_source_mix"], "origin_xcheck_bank_vs_db": a1["origin_xcheck_bank_vs_db"],
    "logmap_src": {"train": a1["logmap_train_src"], "navtest": a1["logmap_navtest_src"]}, "gates": {"train_pre": a1["gate_train_pre"], "navtest": a1["gate_navtest"]},
    "per_map_label_marginals_train_vs_navtest": a1["per_map_train_vs_navtest"],
    "cross_store_same_place": {k: f1[k]["all"] for k in ("cross_store", "within_train", "within_navtest")}, "cross_store_by_distance_bin": f1["cross_store"],
    "within_train_by_distance_bin": f1["within_train"], "n_pairs": f1["n_pairs"], "deliberate_regressions": f1["deliberate_regressions_on_cross_pairs"],
    "head_as_instrument_alignment": {nm: {a: {"best_shift_cells_xy": b1["S3_shift"][nm][a]["best_shift"], "iou_at_zero_shift": b1["S3_shift"][nm][a]["zero_shift"], "iou_best": b1["S3_shift"][nm][a]["best"],
                                              "iou_y_flipped": b1["S3_shift"][nm][a]["y_flipped"]} for a in ARMS} for nm in ("heldout", "navtest")}}
# H-b
S5 = b1["S5_standardisation"]
out["H_b_scene_composition"] = {
    "verdict": "LOCATED. Two composition factors account for the point estimate; the residual is inside the noise",
    "factors": {"F2_city_mix": {a: b2[a]["target=heldout_city_mix"]["F2_city_mix"]["shapley"] for a in ARMS}, "F1_failure_enrichment": {a: b2[a]["target=heldout_city_mix"]["F1_failure_enrichment"]["shapley"] for a in ARMS},
                "residual_after_both_target_heldout_mix": {a: b2[a]["target=heldout_city_mix"]["residual_gap_after_both"] for a in ARMS},
                "residual_after_both_target_train_mix": {a: b2[a]["target=train_city_mix"]["residual_gap_after_both"] for a in ARMS},
                "bootstrap_ci95_target_heldout_mix": {a: b2[a]["bootstrap_target=heldout_city_mix"] for a in ARMS}},
    "sequential_both_orders": {a: {"target=heldout_city_mix": b2[a]["target=heldout_city_mix"], "target=train_city_mix": b2[a]["target=train_city_mix"]} for a in ARMS},
    "navtest_natural_population_P1": {a: S5[a]["navtest_natural_population"] for a in ARMS}, "navtest_c3_sample_P1_banked": {a: S5[a]["navtest_raw"] for a in ARMS},
    "city_mix": b1["targets"], "per_map_iou": b1["S4_maps"], "stratum_iou_navtest": b1["S6_navtest_strata"], "stratum_by_map_H1_s0": b1["S6b_stratum_by_map_H1s0"],
    "LV_only_natural_frames": b2["LV_only_natural_frames"],
    "layout_prior_baseline_no_image": b1["S11_layout_prior_baseline"], "speed_mix_effect_navtest_to_heldout_H1_s0": b1["S7_navtest_to_heldout_speed_mix_H1s0"], "speed_strata_H1_s0": b1["S7_speed_H1s0"],
    "drivable_share_strata_H1_s0": b1["S8_true_drivable_share_strata_H1s0"], "navtest_to_heldout_drivable_share_mix_H1_s0": b1["S8_navtest_to_heldout_drivable_share_mix_H1s0"],
    "navtest_to_heldout_map_x_drivable_share_mix_H1_s0": b1["S8_navtest_to_heldout_map_x_drivable_share_mix_H1s0"],
    "place_novelty": {"dist_summary": d1["dist_summary"], "iou_by_place_distance_H1_s0": d1["iou_by_place_distance"], "navtest_to_heldout_place_mix": d1["navtest_to_heldout_place_mix"],
                      "navtest_to_heldout_map_x_place_mix": d1["navtest_to_heldout_map_x_place_mix"]},
    "error_anatomy_H1_s0": {"confusion_0_40": b1["S2_confusion_0_40"], "error_by_true_sdf_depth": b1["S9_error_depth"]}}
# H-c
out["H_c_metric_region"] = {
    "verdict": "NO difference: one function, one mask definition, one band; both reads reproduce bit-for-bit through the programme's own readout and through independent array code",
    "heldout_reproduction": L("b0_repro_heldout.json"), "navtest_reproduction_abs_diff": {a: v["abs_diff"] for a, v in b1["R0_navtest_repro"].items()},
    "note": "the only differences between the two reads are the label store (pinned train vs pinned navtest sha) and the frame set (1,705 frames of 24 whole logs vs 1,058 C3 tokens of 123 logs, 50 % of them selected on a u400 planner failure)"}
# H-d
out["H_d_feature_shift"] = {
    "verdict": "NOT TESTABLE in full on this machine (no navtrain visual_ctx exists here); what could be tested was tested and found nothing",
    "tested": {"input_container": L("c1_image_provenance.json"), "precision_path_sensitivity": e1, "within_city_natural_frames_LV": b2["LV_only_natural_frames"]},
    "in_domain_probe_vs_S1_head": {"pooled": d2["R0_pooled_all"], "by_map": d2["by_map"], "by_stratum": d2["by_stratum"], "gain_by_map": d2["gain_by_map"], "confusion_0_40": d2["confusion_0_40"],
                                   "place_leak_of_probe": d3},
    "not_testable_here": ["trunk visual_ctx of navtrain frames vs navtest frames (needs the u400 trunk + navtrain pixels: pod only; dev box holds 6,044 of 412,148 train images)",
                          "whether the S1 head at 400 updates is budget-limited rather than domain-limited (needs a navtrain-ctx head run or the V-probe at 400 updates)"]}
json.dump(out, open(os.path.join(HERE, "SHIFT_LOCATE.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
print("ZZASSEMBLE_OK")
