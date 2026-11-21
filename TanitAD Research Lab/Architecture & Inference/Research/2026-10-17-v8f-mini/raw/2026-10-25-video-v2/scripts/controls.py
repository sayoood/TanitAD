"""controls.py -- merges the control results into OUT/CONTROLS_V2.json (read-only on every input; tiny write)."""
import hashlib, json, os, sys
import numpy as np
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
OUT = f"{SC}/viz2"
p1 = json.load(open(f"{OUT}/CONTROLS_V2_part1.json"))["per_window"]
p2 = json.load(open(f"{OUT}/CONTROLS_V2_part2.json"))["per_window"]
sel = json.load(open(f"{SC}/viz/out/selection.json"))
al = json.load(open(f"{SC}/viz/out/align_control.json"))["clips"]
KEYS = ["JUNCTION", "CURVE", "NO-TURN", "FAILURE(worst)"]
mx = lambda d, k: max(v[k] for v in d.values())
C = {"spec": "SELECTION_V2.md controls; CPU float32 re-run of the banked M2b checkpoints; windows = the 4 of viz/out/selection.json, seeds M2b_s0 / M2b_s1"}
# ---- control 1: the window's own nav at ALL layers reproduces B's own-code plan exactly
c1 = {k: mx(p1, k) for k in ("chain_vs_forward_traj30", "chain_vs_forward_ckpt", "chain_vs_forward_lat", "own_nav_all_layers_vs_forward_traj30",
                              "own_nav_all_layers_vs_forward_ckpt", "own_nav_all_layers_vs_forward_lat", "own_nav_all_layers_vs_forward_lon")}
c1["tolerance"] = 0.0
c1["PASS"] = all(v == 0.0 for k, v in c1.items() if k != "tolerance")
c1["n_window_seed_pairs"] = len(p1)
c1["note"] = ("B's own-code plan = model.forward_infer on the window. The chain (run_s -> run_t -> run_o with nav replaced) with the window's own nav at S, T and O, and with no override, "
              "is bit-identical to it at all 8 window-seed pairs: the injection harness changes nothing by itself.")
C["control_1_own_nav_all_layers_reproduces_B_own_code_plan"] = c1
# ---- informational: this CPU re-run vs the banked GPU box-2 arrays (float32 round-off)
C["info_cpu_rerun_vs_banked_gpu"] = {k: mx(p1, k) for k in ("own_traj30_vs_banked_pred", "own_ckpt_Sframe_vs_banked_ckpt", "own_ckpt_now_vs_banked_ckpt_reexpressed", "own_ade6_vs_banked_ade6",
                                                          "fresh_ckpt_vs_sfresh_banked", "fresh_traj30_vs_sfresh_banked_pred", "held_ckpt_Sframe_vs_sfresh_cycle_banked")}
C["info_cpu_rerun_vs_banked_gpu"]["unit"] = "metres (max abs over the 8 window-seed pairs); note: banked ckpt npz is in S's TICK frame, compared in that frame"
C["info_cpu_rerun_vs_banked_gpu"]["tolerance_used"] = 1e-4
C["info_cpu_rerun_vs_banked_gpu"]["PASS"] = all(v <= 1e-4 for k, v in C["info_cpu_rerun_vs_banked_gpu"].items() if isinstance(v, float))
# ---- control 2: drawn plans equal the banked per-window arrays
c2 = {k: mx(p2, k) for k in ("drawn_B_vs_source_tau", "drawn_H_vs_source", "drawn_P10_vs_source", "drawn_HS_vs_source", "tau_vs_kind_banked_pred",
                              "ade6_recomputed_vs_banked_B", "ade6_recomputed_vs_banked_H", "ade6_recomputed_vs_banked_P10", "ade6_recomputed_vs_banked_HS", "ade6_B_vs_selection_json")}
c2["tolerance_m"] = 1e-5
c2["PASS"] = all(v <= 1e-5 for k, v in c2.items() if k != "tolerance_m")
c2["n_window_seed_pairs"] = len(p2)
c2["note"] = ("H/HS/B/P10 as drawn are the arrays read from hblend/hspec score_*_perwindow.npz (B = kind-own `tau`, P10 = `P`), at window indices taken from selection.json; "
              "ADE6 recomputed from the drawn arrays against GT equals the banked per-window ade6 (<= 3.5e-7 m).")
C["control_2_drawn_plans_equal_banked_perwindow_arrays"] = c2
# ---- control 3: frame alignment (v1 `align_control`, re-used: it covers all 101 cache steps of each of the 4 clips)
clips = [sel[k]["clip"] for k in KEYS]
c3 = {"clips": {}}
for c in clips:
    a = al[c]
    c3["clips"][c] = {"steps_checked": a["steps_checked"], "n_argmax_is_self": a["n_argmax_is_self"], "psnr_min_db": a["psnr_overlay_vs_banked_min"], "maxabs": a["maxabs_overlay_vs_banked_max"]}
c3["total_argmax_is_self"] = sum(v["n_argmax_is_self"] for v in c3["clips"].values())
c3["total_steps"] = sum(v["steps_checked"] for v in c3["clips"].values())
c3["psnr_min_db"] = min(v["psnr_min_db"] for v in c3["clips"].values())
c3["PASS"] = c3["total_argmax_is_self"] == c3["total_steps"] and c3["psnr_min_db"] >= 48.0
c3["note"] = ("Cache step j = native frame 2j; each of the 101 steps of the 4 clips was re-encoded and its DINOv3 feature argmax-matched against the banked cache (v1, `viz/out/align_control.json`); "
              "v2 draws frames with the SAME accessor (v2_lib.v2ep_frame) and the SAME step index. Not re-run in v2 (identical inputs, identical accessor).")
C["control_3_frame_alignment_as_v1"] = c3
# ---- the structural zero: a model whose nav embedding is zeroed cannot move with the command
z = np.load(f"{OUT}/data/cf_windows.npz")
zk = [k for k in z.files if "/cfzero/" in k and k.endswith(("/D6", "/D6_TURN_L", "/D6_TURN_R", "/lat_sep_LR", "/LR_dist"))]
C["control_4_zeroed_e_nav_structural_zero"] = {"n_cells": len(zk), "max_abs": float(max(abs(float(z[k])) for k in zk)), "PASS": all(float(z[k]) == 0.0 for k in zk),
                                              "note": "e_nav.weight zeroed => the nav embedding is a constant; D6, D6_TURN_L/R, lat_sep and LR_dist must read EXACTLY 0.0 at every injection point (identity, not an estimate)."}
# ---- pre-declaration precedence
sp = f"{OUT}/SELECTION_V2.md"
mt = os.path.getmtime(sp)
later = {}
for f in ("data/cf_windows.npz", "data/cf_evalwide.npz", "data/plans_pack.npz", "data/tables.json", "CONTROLS_V2_part1.json", "CONTROLS_V2_part2.json", "v8f_evidence_v2.mp4"):
    later[f] = os.path.getmtime(f"{OUT}/{f}") > mt
C["pre_declaration"] = {"file": "SELECTION_V2.md", "sha256": hashlib.sha256(open(sp, "rb").read()).hexdigest(), "mtime_utc_epoch": mt,
                        "every_output_written_after_it": later, "PASS": all(later.values()),
                        "note": "SELECTION_V2.md was written before any run and has not been edited since (mtime precedes every output; sha256 recorded at the time of this merge)."}
C["ALL_PASS"] = all(v.get("PASS", True) for v in C.values() if isinstance(v, dict))
json.dump(C, open(f"{OUT}/CONTROLS_V2.json", "w"), indent=1)
print(json.dumps({k: (v.get("PASS") if isinstance(v, dict) else None) for k, v in C.items()}, indent=1)); print("ALL_PASS", C["ALL_PASS"])
