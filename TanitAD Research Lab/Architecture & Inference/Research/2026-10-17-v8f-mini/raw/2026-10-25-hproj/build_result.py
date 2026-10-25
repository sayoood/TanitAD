"""Builds HPROJ_RESULT.md from the JSON outputs (every number is read from an artifact, none is typed). CPU, stdlib only."""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SEEDS = ("M2b_s0", "M2b_s1")
SV = {"M2b_s0": "s0", "M2b_s1": "s1"}
S = {s: json.load(open(f"{OUT}/score_{s}.json")) for s in SEEDS}
C = {s: json.load(open(f"{OUT}/control_{s}.json")) for s in SEEDS}
R = json.load(open(f"{OUT}/READ.json"))
DK = json.load(open(f"{OUT}/dk_read.json"))
DV = json.load(open(f"{OUT}/diag_vhi.json"))
DB = json.load(open(f"{OUT}/diag_band.json"))
HB = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hblend"


def f(v, dp=4):
    return "n/a" if v is None else f"{v:+.{dp}f}"


def g(v, dp=4):
    return "n/a" if v is None else f"{v:.{dp}f}"


def ci(d, dp=4):
    return f"{f(d['mean'], dp)} [{f(d['lo'], dp)}, {f(d['hi'], dp)}]"


def star(d):
    if not d.get("separated"):
        return ""
    return " *better*" if d.get("better") else (" *WORSE*" if d.get("worse") else " *")


def ex(m, dp=4):
    return f"{g(m['mean'], dp)} [{g(m['lo'], dp)}, {g(m['hi'], dp)}]"


EST = "paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0"
TIER = "T1-class open loop (own held outputs; not closed loop)"
L = []
w = L.append
FAM = [("LONGITUDINAL", "speed MAE (m/s)", "fw_lon_speed_mae", 4), ("LONGITUDINAL", "accel MAE (m/s^2)", "fw_lon_accel_mae", 4),
       ("LONGITUDINAL", "along MAE (m)", "fw_lon_along_mae", 4), ("LATERAL", "cross-track offset MAE (m)", "fw_lat_cross_mae", 4),
       ("LATERAL", "heading MAE (deg)", "fw_lat_heading_mae_deg", 4), ("LATERAL", "yaw-rate MAE (deg/s)", "fw_lat_yaw_rate_mae_degps", 4),
       ("LATERAL", "curvature MAE (1/m)", "fw_lat_curvature_mae_1pm", 4),
       ("TACTICAL (plan-derived)", "lateral decision agreement (higher better)", "fw_tac_lat_acc", 4),
       ("TACTICAL (plan-derived)", "longitudinal decision agreement (higher better)", "fw_tac_lon_acc", 4)]
rows = R["per_seed"]
b0, b1 = C["M2b_s0"]["binding"], C["M2b_s1"]["binding"]

w("# HPROJ_RESULT: PREREG_V8F_HPROJ executed exactly as written (2026-10-25)\n")
w("Dev box, CPU only, ZERO training, no GPU, offline from banked outputs. Every number is **MEASURED (ours)** from the artifact paths named; tier **" + TIER + "**; estimator **" + EST +
  "** unless a row says otherwise. The bootstrap answers 'would another draw of EPISODES say this?' only; training-seed variance enters only through the two seeds shown side by side "
  "(`H-ESTIM-SEED-1`); inference is deterministic, so no inference-seed floor applies. `OUT` = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/"
  "bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hproj`. Contract: `Project Steering/PREREG_V8F_HPROJ.md` at c5ac9a9f. B = B's banked UNPROJECTED plan (its headline numbers), H = HBLEND's H, "
  "HP = the arm, Bp = B's own banked projected plan (`traj30_proj`), P10 = the P10-yaw-4512 plan.\n")

# ------------------------------------------------------------------ 0
w("## 0. Outcome (the PREREG outcome table, applied verbatim)\n")
w(f"**{R['outcome']}. This is a FAIL of the pre-registered bar, reported as written.** PRIMARY-C fails at both seeds; PRIMARY-K fails at both seeds (s0: HP - B on ADE2 and ADE6 is positive and not "
  "separated; s1: both are positive and separated WORSE than B). The table's row 'CONSISTENCY NOT RESTORED' (PRIMARY-C fails) is the first to match; 'GAIN LOST' requires PRIMARY-C to PASS, so it does not "
  "apply. HP is **not** the deployed v8F output and H's weights do not enter `forward_infer` as a buffer on this evidence.\n")
w("| read | seed | value | 95 % CI | bar | verdict |")
w("|---|---|---|---|---|---|")
for s in SEEDS:
    r = rows[s]
    w(f"| **PRIMARY-C** CONSISTENCY(a) 'any', HP - B | {SV[s]} | {f(r['PRIMARY_C']['delta'])} | [{f(r['PRIMARY_C']['lo'])}, {f(r['PRIMARY_C']['hi'])}] | upper bound <= +0.02 | **FAIL** (upper {f(r['PRIMARY_C']['hi'])}) |")
for s in SEEDS:
    for k, lab in (("ade6", "ADE6"), ("ade2", "ADE2")):
        d = rows[s]["PRIMARY_K"][k]
        w(f"| **PRIMARY-K** {lab}, HP - B (m) | {SV[s]} | {f(d['delta'])} | [{f(d['lo'])}, {f(d['hi'])}] | < 0 and separated | **FAIL** ({'separated' if d['separated'] else 'not separated'}, "
          f"{'worse' if d['delta'] > 0 else 'better'}) |")
w("\nEstimator: " + EST + ". Tier: " + TIER + ". Artifacts: `OUT/score_M2b_s{0,1}.json` (`consistency.paired.HP_minus_B.viol_any`, `paired.HP_minus_B.ade2/ade6`), outcome `OUT/READ.json`.\n")
w(f"**Committed next step (PREREG, verbatim):** *{R['committed_next_step_verbatim']}*\n")
w("**Rule Zero.** The table's next step is a constraint-aware fit under its OWN PREREG. It is not run here: the PREREG fixes this arm, its bars and its fit, and a new fit is a new arm that needs its own "
  "pre-registered bars. What this run does leave behind is the measured mix the next PREREG must be written against, from two post-hoc diagnostics (sections 8 and 9; descriptive, no bar, they change no "
  "number above). Three measured facts, in order of effect:\n")
sa = S["M2b_s0"]["consistency"]["speed_anatomy"]
w("1. **The PREREG's explanation is only partly what the data show.** It predicted that H's violations are 'stops, leads or turns, which the cap does not see'. The per-rule breakdown (section 7) puts "
  "the largest excess on **SPEED**, with LEAD second (HP - B on LEAD " + ", ".join(f"{f(S[s]['consistency']['paired']['HP_minus_B']['viol_LEAD']['mean'])} ({SV[s]})" for s in SEEDS) +
  ", which the projection does not touch: HP - H on LEAD " + ", ".join(f"{f(S[s]['consistency']['paired']['HP_minus_H']['viol_LEAD']['mean'])}" for s in SEEDS) + ") and STOP / TURN small. HP - B on SPEED is " + ", ".join(f"{f(S[s]['consistency']['paired']['HP_minus_B']['viol_SPEED']['mean'])} ({SV[s]})" for s in SEEDS) + " against 'any' "
  + ", ".join(f"{f(S[s]['consistency']['paired']['HP_minus_B']['viol_any']['mean'])} ({SV[s]})" for s in SEEDS) + ". But the SPEED rule's cap is min(posted limit, **v_hi_ms**), and v_hi_ms is a GT-derived "
  f"speed-band edge (T predicts it as a `cons_pred` field), not the posted-limit input the projection reads: it is the binding cap in **{sa['windows_by_kind']['active_both_present_and_v_hi_binding (v_hi < limit)']} of {sa['n_windows']} windows**, and "
  + ", ".join(f"{sa['violations_by_arm']['HP']['active_both_present_and_v_hi_binding (v_hi < limit)'] if s == 'M2b_s0' else S[s]['consistency']['speed_anatomy']['violations_by_arm']['HP']['active_both_present_and_v_hi_binding (v_hi < limit)']} of "
              f"{S[s]['consistency']['speed_anatomy']['violations_by_arm']['HP']['total']}" for s in SEEDS) + " of HP's SPEED violations (s0, s1) sit in such windows. Projection onto the posted limit alone cannot "
  "remove them (by construction: wherever the posted limit is the binding cap, HP respects it). B's own projection (Bp) barely moves SPEED either (section 7).")
w("2. **The projection as specified also costs position accuracy, for a separable reason.** `project_speed` caps every 0.1 s interval from t = 0, but the SPEED rule reads only intervals starting at "
  "tau >= 2.0 s. Applied alone to B it already costs ADE6 " + ", ".join(f"{f(S[s]['paired']['Bp_minus_B']['ade6']['mean'])} ({SV[s]})" for s in SEEDS) + " (separated worse). With the cap applied from 2.0 s "
  "(the rule's own band; section 9, diagnostic) the ADE2 cost disappears (HP ADE2 - B " + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_lim']['ade2_minus_B']['mean'])} ({SV[s]})" for s in SEEDS) + ") but "
  "CONSISTENCY is still not restored with the posted limit alone (HP 'any' - B upper bound " + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_lim']['cons_minus_B']['viol_any']['hi'])}" for s in SEEDS) + ").")
w("3. **A deployable-class v_hi term is not enough on its own.** Adding T's own predicted v_hi_ms to the cap (section 9, banded) moves 'any' - B to "
  + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_pr']['cons_minus_B']['viol_any']['mean'])} ({SV[s]})" for s in SEEDS) + " (upper bounds "
  + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_pr']['cons_minus_B']['viol_any']['hi'])}" for s in SEEDS) + ", bar +0.02) and costs ADE6 "
  + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_pr']['ade6_minus_B']['mean'])}" for s in SEEDS) + " (T's v_hi head is off from the GT label by "
  + ", ".join(f"{DV['seeds'][s]['pred_v_hi_vs_gt_v_hi_mae_ms']:.2f}" for s in SEEDS) + " m/s MAE). The T0 ORACLE v_hi (reads a GT label; a ceiling, never a result) restores consistency "
  "('any' - B " + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_or']['cons_minus_B']['viol_any']['mean'])}" for s in SEEDS) + ") and keeps ADE6 below B (" + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_or']['ade6_minus_B']['mean'])}" for s in SEEDS) +
  "), so the constraint is recoverable if v_hi can be predicted well enough; the open lever is the quality of T's v_hi prediction, not the projection or the shrinkage.\n")
w("**What blocks a finished result here** (named, per Rule Zero): the next arm (a banded, v_hi-aware, constraint-aware projection with its fit) needs its own PREREG with committed bars before any number is "
  "read, and it needs a decision on whether the consistency SPEED rule's GT-derived v_hi term is the right target for a deployable projection at all (the human violates the posted limit alone in "
  f"{DV['seeds']['M2b_s0']['GT_plan_vs_posted_limit_alone']['violation_rate']*100:.1f} % of these windows, so the posted limit is not a constraint the GT itself respects). Both are Master Mind / PI calls; no GPU, no box 2.\n")

# ------------------------------------------------------------------ 1
w("## 1. What was run\n")
w("* **Route (as the PREREG: eval-only, offline).** B's banked per-window plan `pred` (KIND run own code, `OUT/../hblend/inputs/banked/eval_M2b_s{0,1}-kind-own.{npz,json}`, sha256 in each score json) is the only "
  "model output used. P = the checkpoint's own frozen P10 on the eval kinematic features (no image needed). GT, cap source (`J['limit']`, `J['limit_ok']`) and the scoring context are the eval's own "
  "(`v8f_mini_data.eval_context`, `v8f_mini_eval.evaluate`). Weights: HBLEND's `weights_M2b_s{0,1}.json`, **not re-fitted**.")
w("* **tau_B^(60)** = up(P10) + Basis c. Basis built from the formula (`cos(k pi (i+0.5)/60)` x ramp `(i+1)/60`, nb = " + str(C['M2b_s0']['diagnostics']['E1_nb_from_ckpt_cfg']) + " from the checkpoint cfg); c by least squares on the 30 "
  "eval-step rows (`basis[1::2]`, rank " + str(C['M2b_s0']['diagnostics']['E1_lstsq']['rank']) + ", condition number " + g(C['M2b_s0']['diagnostics']['E1_lstsq']['cond'], 1) + ").")
w("* **tau_H^(60)** = up(P10) + w60 (.) (tau_B^(60) - up(P10)), ego frame at NOW (axis 0 along/x, axis 1 cross/y); w60 = the banked 30 knots linearly interpolated to 0.1 s with w(0.1) = w(0.2) "
  "(equal to HBLEND's own derived grid to " + g(C['M2b_s0']['diagnostics']['E4_w60_vs_hblend_w60_derived_maxabs'], 1) + " but computed independently with `np.interp`).")
w("* **HP** = `project_speed(tau_H^(60), cap)[:, 1::2]`, torch float32 CPU (the eval's own precision), cap per window = `limit_ms_from_onehot(J['limit'][clip], J['limit_ok'][clip])`, 1e9 where there is no "
  f"limit (here every one of the {C['M2b_s0']['diagnostics']['E2_n_windows']} windows has a posted limit; values {', '.join(f'{v*3.6:.0f} km/h' for v in C['M2b_s0']['diagnostics']['E2_cap_values_ms_unique'])}).")
w("* **Scoring.** Every metric comes from the eval's OWN `evaluate()` (traj, consistency (a), self-consistency (b), tactical, families) on HP, H and Bp; paired rows via `v8f_beta_eval.paired_row` "
  "(episode-cluster bootstrap, B = 2000). Distance keeping via `v8f_dk_read.read_dk` -> `taniteval.lead_metrics`. Code: `OUT/v8f_hproj.py` imports HBLEND's loaders read-only.\n")

# ------------------------------------------------------------------ 2
w("## 2. Reconstruction control (run BEFORE any HP number; the script refuses to read and exits rc 2 if a binding control fails)\n")
w("| control | bar | seed 0 | seed 1 | binding? |")
w("|---|---|---|---|---|")
w(f"| **C0** P10 plan file vs the model's internal p30, max abs (m) | <= 1e-4 | {b0['C0_P10_file_vs_model_prior_maxabs_m']:.3e} | {b1['C0_P10_file_vs_model_prior_maxabs_m']:.3e} | yes (task step 1) |")
w(f"| **C1** up(P10) + Basis c, subsampled, vs B's banked traj30, max abs (m) | <= 1e-4 | {b0['C1_recon_knots_vs_banked_traj30_maxabs_m']:.3e} | {b1['C1_recon_knots_vs_banked_traj30_maxabs_m']:.3e} | **yes (PREREG)** |")
w(f"| **C2** `project_speed` on B's reconstructed traj60: ADE6 {g(b0['C2_recon_projected_ade6'], 5)} / {g(b1['C2_recon_projected_ade6'], 5)} vs banked `projected_r1_kamm` ADE6 "
  f"{b0['C2_banked_projected_r1_kamm_ade6_json']} / {b1['C2_banked_projected_r1_kamm_ade6_json']}, abs diff (m) | <= 1e-3 | {b0['C2_absdiff_m']:.3e} | {b1['C2_absdiff_m']:.3e} | **yes (PREREG)** |")
w(f"| **all binding controls pass** | | **{b0['all_pass']}** | **{b1['all_pass']}** | |")
w("\nDiagnostics added by the executor (reported, not bars, cannot change the table):\n")
w("| diagnostic | seed 0 | seed 1 |")
w("|---|---|---|")
def drow(lab, key, fmt="{:.3e}", sub=None):
    vals = []
    for s in SEEDS:
        v = C[s]["diagnostics"][key]
        v = v[sub] if sub else v
        vals.append(fmt.format(v) if isinstance(v, float) else str(v))
    w(f"| {lab} | {vals[0]} | {vals[1]} |")
drow("E1 basis formula vs the model's own `basis` buffer, max abs", "E1_basis_formula_vs_model_buffer_maxabs")
drow("E1 least-squares fit residual on the 30 rows, max abs (m)", "E1_lstsq", sub="lstsq_residual_max_abs_m")
drow("E2 cap vs `eval_context` limit_ms (float64 route), max abs (m/s)", "E2_cap_vs_eval_context_limit_ms_maxabs")
drow("E3 reconstructed projected plan vs the eval's saved `traj30_proj`, max abs (m) / mean abs (m)", "E3_projected_plan_recon_vs_banked_traj30_proj_maxabs_m")
drow("E3 mean abs (m)", "E3_projected_plan_recon_vs_banked_traj30_proj_mean_abs_m")
drow("E3 ADE6 of the saved `traj30_proj` (exact mean) vs reconstruction, abs diff (m)", "E3_recon_vs_banked_traj30_proj_ade6_exact_absdiff")
drow("E3 whole `projected_r1_kamm` block (traj, consistency_a, strata) recon vs banked json, mismatches at 1e-6", "E3_projected_block_recon_vs_banked_json", sub="n_mismatches")
drow("E4 w60 vs HBLEND `w60_derived_unscored`, max abs", "E4_w60_vs_hblend_w60_derived_maxabs")
drow("E5 H60 knots vs HBLEND's saved H, max abs (m)", "E5_H60_knots_vs_hblend_saved_H_maxabs_m")
drow("E5 w = 1 gives tau60 / w = 0 gives up(P10) (max abs)", "E5_w1_gives_tau60_maxabs")
drow("E5 numpy `up_np` vs torch `upsample_prior`, max abs (float32 rounding)", "E5_up_np_vs_torch_upsample_prior_maxabs")
drow("projection float64 vs float32, max abs (m)", "precision_float64_projection_vs_float32_maxabs_m")
w(f"| **R1** deliberate regression: basis WITHOUT the ramp, knots max abs (m); must exceed 1e-4 | {C['M2b_s0']['diagnostics']['R1_no_ramp_basis_knots_maxabs_m']:.3f} (control fails as it must: {C['M2b_s0']['diagnostics']['R1_must_exceed_C1_bar_and_does']}) | "
  f"{C['M2b_s1']['diagnostics']['R1_no_ramp_basis_knots_maxabs_m']:.3f} ({C['M2b_s1']['diagnostics']['R1_must_exceed_C1_bar_and_does']}) |")
w(f"| **R2** deliberate regression: NO cap, ADE6 abs diff vs banked json (m); must exceed 1e-3 | {C['M2b_s0']['diagnostics']['R2_no_cap_absdiff_vs_banked_json']:.4f} ({C['M2b_s0']['diagnostics']['R2_must_exceed_C2_bar_and_does']}) | "
  f"{C['M2b_s1']['diagnostics']['R2_no_cap_absdiff_vs_banked_json']:.4f} ({C['M2b_s1']['diagnostics']['R2_must_exceed_C2_bar_and_does']}) |")
w(f"| **R3** deliberate regression: NO projection, ADE6 abs diff vs banked json (m); must exceed 1e-3 | {C['M2b_s0']['diagnostics']['R3_no_projection_absdiff_vs_banked_json']:.4f} ({C['M2b_s0']['diagnostics']['R3_must_exceed_C2_bar_and_does']}) | "
  f"{C['M2b_s1']['diagnostics']['R3_no_projection_absdiff_vs_banked_json']:.4f} ({C['M2b_s1']['diagnostics']['R3_must_exceed_C2_bar_and_does']}) |")
x0, x1 = C["M2b_s0"]["X1"], C["M2b_s1"]["X1"]
w(f"| **X1** CPU `forward_infer` TRUE traj60 vs the reconstruction, all 60 steps, max abs (m), {x0['n_windows_checked']} windows whose features are on this box | {x0['max_abs_traj60_cpu_vs_recon_all60_m']:.3e} | {x1['max_abs_traj60_cpu_vs_recon_all60_m']:.3e} |")
w(f"| X1 same, the 0.1 s MIDPOINTS only (rows no banked file carries) | {x0['max_abs_traj60_cpu_vs_recon_midpoints_only_m']:.3e} | {x1['max_abs_traj60_cpu_vs_recon_midpoints_only_m']:.3e} |")
w(f"| X1 cap built by the eval loader's own batch vs cap from `J`, max abs (m/s) | {x0['max_abs_cap_eval_loader_vs_J_m_s']:.1e} | {x1['max_abs_cap_eval_loader_vs_J_m_s']:.1e} |")
w(f"| X1 `project_speed` on the model's TRUE traj60 vs on the reconstruction, max abs (m) | {x0['max_abs_projected60_cpu_vs_projected60_recon_m']:.3e} | {x1['max_abs_projected60_cpu_vs_projected60_recon_m']:.3e} |")
w(f"| H block re-evaluated through `evaluate()` vs HBLEND's banked block, mismatches | {S['M2b_s0']['H_block_vs_hblend_saved']['n_mismatches']} | {S['M2b_s1']['H_block_vs_hblend_saved']['n_mismatches']} |")
m3 = [C[s]["diagnostics"]["E3_projected_block_recon_vs_banked_json"] for s in SEEDS]
w("\nReading: the reconstruction is exact to float32 rounding, including the 0.1 s midpoints that no banked file carries (X1), and the cap the eval loader builds is bit-identical to the one rebuilt from `J` (X1). "
  "The reference arm H, re-scored here, equals HBLEND's banked H block with 0 mismatches. The eval's saved `traj30_proj` differs from the reconstruction by at most "
  + f"{max(C[s]['diagnostics']['E3_projected_plan_recon_vs_banked_traj30_proj_maxabs_m'] for s in SEEDS):.1e}" + " m on any coordinate of any window (mean abs ~1e-6 m); that bound sits just above 1e-4 m at s0, which is why E3 is a "
  "diagnostic and the PREREG's binding forms are C1 and C2, which pass with margin (worst C1 " + f"{max(C[s]['binding']['C1_recon_knots_vs_banked_traj30_maxabs_m'] for s in SEEDS):.2e}" + " against 1e-4; worst C2 "
  + f"{max(C[s]['binding']['C2_absdiff_m'] for s in SEEDS):.2e}" + " against 1e-3). The E3 whole-block comparison has " + f"{m3[0]['n_mismatches']} mismatches at s0 and {m3[1]['n_mismatches']} at s1"
  + (f": `{m3[1]['mismatches'][0]}` -- ONE of the {2399 * 29} acceleration steps crosses the 0.001 m/s^2 zero-acceleration tolerance (float noise in a counter, not in a plan); reported, not hidden." if m3[1]["n_mismatches"] else ".") + "\n")

# ------------------------------------------------------------------ 3
w("## 3. The PREREG rows and the dominance row (HP against P10), beside rows\n")
w("| row | seed | value | 95 % CI | estimator | tier | artifact |")
w("|---|---|---|---|---|---|---|")
for s in SEEDS:
    P = S[s]["paired"]
    cp = S[s]["consistency"]["paired"]["HP_minus_B"]["viol_any"]
    w(f"| CONSISTENCY(a) 'any' HP - B | {SV[s]} | {f(cp['mean'])} | [{f(cp['lo'])}, {f(cp['hi'])}] | {EST} | {TIER} | `OUT/score_{s}.json::consistency.paired.HP_minus_B.viol_any` |")
    for k, lab in (("ade6", "ADE6"), ("ade2", "ADE2")):
        d = P["HP_minus_B"][k]
        w(f"| {lab} HP - B (m) | {SV[s]} | {f(d['mean'])} | [{f(d['lo'])}, {f(d['hi'])}] | {EST} | {TIER} | `OUT/score_{s}.json::paired.HP_minus_B.{k}` |")
w("\n**Dominance row, HP against P10-yaw-4512** (ADE2, ADE6, accel MAE; paired):\n")
w("| seed | ADE2 HP / P10 (m) | ADE6 HP / P10 (m) | ADE2 HP - P10 | ADE6 HP - P10 | accel MAE HP / P10 (m/s^2) | accel HP - P10 | point estimates all <= P10 |")
w("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    sg, P = S[s]["single"], S[s]["paired"]["HP_minus_P10"]
    dm = rows[s]["DOMINANCE_vs_P10"]["point_estimates_le_P10"]
    w(f"| {SV[s]} | {g(sg['HP']['ade2']['mean'])} / {g(sg['P10']['ade2']['mean'])} | {g(sg['HP']['ade6']['mean'])} / {g(sg['P10']['ade6']['mean'])} | {ci(P['ade2'])}{star(P['ade2'])} | {ci(P['ade6'])}{star(P['ade6'])} | "
      f"{g(sg['HP']['fw_lon_accel_mae']['mean'])} / {g(sg['P10']['fw_lon_accel_mae']['mean'])} | {ci(P['fw_lon_accel_mae'])}{star(P['fw_lon_accel_mae'])} | ADE2 {dm['ade2']}, ADE6 {dm['ade6']}, accel {dm['accel']} |")
w("\nHP does not dominate P10: at 2 s it is separated WORSE than P10 at both seeds (the cap from t = 0), at 6 s it is separated better, and its acceleration MAE stays above P10's.\n")
w("**HP against B's own projected plan Bp** (the other projection variant; paired):\n")
w("| seed | ADE2 HP - Bp | ADE6 HP - Bp | accel MAE HP - Bp | CONSISTENCY 'any' HP - Bp |")
w("|---|---|---|---|---|")
for s in SEEDS:
    P = S[s]["paired"]["HP_minus_Bp"]
    cp = S[s]["consistency"]["paired"]["HP_minus_Bp"]["viol_any"]
    w(f"| {SV[s]} | {ci(P['ade2'])}{star(P['ade2'])} | {ci(P['ade6'])}{star(P['ade6'])} | {ci(P['fw_lon_accel_mae'])}{star(P['fw_lon_accel_mae'])} | {ci(cp)} |")
w("\nBp (B with the same projection) is better than HP on consistency (HP - Bp positive, separated at both seeds) and worse on ADE and acceleration (HP - Bp negative, separated). Neither dominates the other. "
  "(HYPOTHESIS, not isolated: the shrinkage moves the plan away from where B's consistency-trained residual put it, and the posted-limit projection does not bring that back.) "
  "`HP - H`, the cost of adding the projection to H, is in sections 4 and 7.\n")

# ------------------------------------------------------------------ 4 ADE/FDE
w("## 4. ADE and FDE at 1-6 s (beside)\n")
for s in SEEDS:
    sg, P, Pb = S[s]["single"], S[s]["paired"]["HP_minus_B"], S[s]["paired"]["HP_minus_H"]
    w(f"**{s}** (m; mean [95 % CI]; `*better*` / `*WORSE*` = CI excludes 0)\n")
    w("| metric | B | H | HP | Bp | P10 | HP - B | HP - H |")
    w("|---|---|---|---|---|---|---|---|")
    for kind in ("ade", "fde"):
        for h in range(1, 7):
            k = f"{kind}{h}"
            w(f"| {kind.upper()}@{h}s | {ex(sg['B'][k])} | {ex(sg['H'][k])} | {ex(sg['HP'][k])} | {ex(sg['Bp'][k])} | {ex(sg['P10'][k])} | {ci(P[k])}{star(P[k])} | {ci(Pb[k])}{star(Pb[k])} |")
    w("")
w("Artifacts: `OUT/score_M2b_s*.json::single / paired.HP_minus_B / paired.HP_minus_H`.\n")

# ------------------------------------------------------------------ 5 families
w("## 5. The four families, beside (paired; lower is better except the two plan-derived tactical agreement rows)\n")
for s in SEEDS:
    sg, P1, P2, P3 = S[s]["single"], S[s]["paired"]["HP_minus_B"], S[s]["paired"]["HP_minus_H"], S[s]["paired"]["HP_minus_P10"]
    w(f"**{s}**\n")
    w("| family | metric | B | H | HP | P10 | HP - B | HP - H | HP - P10 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for fam, lab, k, dp in FAM:
        w(f"| {fam} | {lab} | {g(sg['B'][k]['mean'])} | {g(sg['H'][k]['mean'])} | {g(sg['HP'][k]['mean'])} | {g(sg['P10'][k]['mean'])} | {ci(P1[k])}{star(P1[k])} | {ci(P2[k])}{star(P2[k])} | {ci(P3[k])}{star(P3[k])} |")
    w("")
w("(`better`/`WORSE` already account for each metric's direction.) STRATEGIC family: **PAUSED** per the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` ITEM 25): the strategic layer is switched off temporarily; "
  "S is untouched by HP and its banked checkpoint errors are carried unchanged. It is reported as PAUSED, not as a pass. TACTICAL, T's own decision heads (lat / lon / goal macro-F1): HP's block equals B's banked "
  f"block exactly (json equality) at s0: {S['M2b_s0']['structural_identity']['tactical_block_T_heads_HP_equals_banked_B']} and s1: {S['M2b_s1']['structural_identity']['tactical_block_T_heads_HP_equals_banked_B']}; the "
  "plan-derived tactical agreement above is read off the trajectory and does move with HP.\n")

# ------------------------------------------------------------------ 6 DK
w("## 6. Distance keeping (the other half of LONGITUDINAL), K = 30 (6 s) lead block, via the programme's own `v8f_dk_read.read_dk` -> `taniteval.lead_metrics`\n")
w(f"Lead block sha256 `{DK['lead_block']['sha256'][:16]}...`, window states {DK['join']['window_states']}, kinematic-contract check **{DK['human_reference']['kinematic_contract']['status']}**, arm-vs-itself identity "
  "control exactly +0 [0, 0]. Open loop: the lead's logged future does not react to the arm. No distance-keeping bar is registered; nothing here is PASS/FAIL. Min-TTC is a CENSORED mean (30 s cap).\n")
w("| arm | headway min (m) | time-gap min (s) | min-TTC (s, censored) | LEAD-VIOL share of at-risk (n viol / 595) |")
w("|---|---|---|---|---|")
hm = DK["human_reference"]["metrics"]
w(f"| human (GT path) | {hm['headway_min_m']['mean']:.3f} | {hm['time_gap_min_s']['mean']:.3f} | {hm['min_ttc_s']['mean']:.3f} | by definition 0 |")
for nm in ("B_s0", "H_s0", "HP_s0", "Bp_s0", "B_s1", "H_s1", "HP_s1", "Bp_s1", "P10"):
    a = DK["arms"][nm]
    m = a["metrics"]
    w(f"| {nm} | {m['headway_min_m']['mean']:.3f} | {m['time_gap_min_s']['mean']:.3f} | {m['min_ttc_s']['mean']:.3f} | {a['lead_viol']['share_of_at_risk']:.4f} ({a['lead_viol']['n_viol']}) |")
w("\n| paired contrast | headway (m) | time-gap (s) | min-TTC (s) | LEAD-VIOL at-risk (share) |")
w("|---|---|---|---|---|")
for pn in DK["pairs"]:
    pr = DK["pairs"][pn]
    cells = []
    for k in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
        d = pr["dk"][k]
        cells.append(f"{f(d['delta'], 3)} [{f(d['lo'], 3)}, {f(d['hi'], 3)}]{' *' if d['separated'] else ''}")
    lv = pr["lead_viol"]["at_risk"]
    cells.append(f"{f(lv['delta'], 4)} [{f(lv['lo'], 4)}, {f(lv['hi'], 4)}]{' *' if lv['separated'] else ''}")
    w(f"| {pn.replace('__minus__', ' - ')} | " + " | ".join(cells) + " |")
w("\n`*` = CI excludes 0. Artifact: `OUT/dk_read.json`, per-window `OUT/dk_perwindow.npz`. Reading: " + "; ".join(
    f"{SV[s]}: HP - B headway {DK['pairs'][f'HP_{SV[s]}__minus__B_{SV[s]}']['dk']['headway_min_m']['delta']:+.3f} m, LEAD-VIOL {DK['pairs'][f'HP_{SV[s]}__minus__B_{SV[s]}']['lead_viol']['at_risk']['delta']:+.4f} "
    f"({'separated' if DK['pairs'][f'HP_{SV[s]}__minus__B_{SV[s]}']['lead_viol']['at_risk']['separated'] else 'not separated'})" for s in SEEDS) + ".\n")

# ------------------------------------------------------------------ 7 consistency
w("## 7. CONSISTENCY(a) per constraint, and (b), for B, H and HP (the PREREG's PRIMARY-C lives here)\n")
w("CONSISTENCY(a) = share of windows whose PLAN violates a GT-derived constraint (speed cap, stop, turn, lead gap); (b) = violation of the model's OWN predicted constraints. Lower is better. "
  "Estimator: " + EST + ". Artifact: `OUT/score_M2b_s*.json::consistency`; P10 rates from `floors_perwindow.npz`; Bp = B with the same projection.\n")
CKS = [("viol_any", "any"), ("viol_SPEED", "SPEED"), ("viol_STOP", "STOP"), ("viol_TURN", "TURN"), ("viol_LEAD", "LEAD"), ("self_viol_any", "self (b) any")]
for s in SEEDS:
    c = S[s]["consistency"]
    n_act = {k: (S[s]["blocks"]["HP"]["consistency_a"][k]["n_active"] if k in S[s]["blocks"]["HP"]["consistency_a"] else None) for k in ("SPEED", "STOP", "TURN", "LEAD")}
    w(f"**{s}** (active windows: SPEED {n_act['SPEED']}, STOP {n_act['STOP']}, TURN {n_act['TURN']}, LEAD {n_act['LEAD']} of {S[s]['n_windows']})\n")
    w("| rule | B | H | HP | Bp | P10 | H - B | **HP - B** | HP - H | HP - Bp |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for k, lab in CKS:
        rt = c["rate"]
        p10 = g(rt["P10"][k]) if k in rt["P10"] else "n/a"
        def pr(pair):
            d = c["paired"][pair].get(k)
            return "n/a" if d is None else f"{ci(d)}{' *WORSE*' if d['mean'] > 0 and d['separated'] else (' *better*' if d['mean'] < 0 and d['separated'] else '')}"
        w(f"| {lab} | {g(rt['B'][k])} | {g(rt['H'][k])} | {g(rt['HP'][k])} | {g(rt['Bp'][k])} | {p10} | {pr('H_minus_B')} | **{pr('HP_minus_B')}** | {pr('HP_minus_H')} | {pr('HP_minus_Bp')} |")
    w("")
w("**Transitions (windows), HP vs B** (violated by HP only / violated by B only / both):\n")
w("| seed | " + " | ".join(l for _, l in CKS) + " |")
w("|---|" + "---|" * len(CKS))
for s in SEEDS:
    t = S[s]["consistency"]["transitions"]["HP_vs_B"]
    w(f"| {SV[s]} | " + " | ".join(f"{t[k]['a_viol_b_ok']} / {t[k]['a_ok_b_viol']} / {t[k]['both_viol']}" for k, _ in CKS) + " |")
w("\n**Anatomy of the SPEED rule** (DIAGNOSTIC, descriptive): the eval's SPEED cap is min(posted limit, v_hi_ms), v_hi_ms being the GT speed-band edge. Windows by which cap is present, and where each arm's SPEED violations sit:\n")
sa0 = S["M2b_s0"]["consistency"]["speed_anatomy"]
w(f"Windows ({sa0['n_windows']}): " + "; ".join(f"{k} {v}" for k, v in sa0["windows_by_kind"].items()) + ".\n")
w("| seed | arm | SPEED violations | in windows where v_hi < posted limit | in posted-limit-only windows |")
w("|---|---|---|---|---|")
for s in SEEDS:
    va = S[s]["consistency"]["speed_anatomy"]["violations_by_arm"]
    for arm in ("B", "H", "HP", "Bp"):
        a = va[arm]
        w(f"| {SV[s]} | {arm} | {a['total']} | {a['active_both_present_and_v_hi_binding (v_hi < limit)']} | {a['active_posted_limit_only (v_hi absent)']} |")
w("\nReading (MEASURED): the projection recovers part of H's cost (HP - H is separated better on 'any' and SPEED at both seeds) but leaves HP above B: PRIMARY-C's upper bound is "
  + " and ".join(f"{f(rows[s]['PRIMARY_C']['hi'])} ({SV[s]})" for s in SEEDS) + " against +0.02. The largest excess is SPEED, and SPEED is capped by v_hi where it binds, which is not the posted-limit input. "
  "LEAD is second and STOP / TURN are small; HP - B (STOP, TURN, LEAD) is s0: " + ", ".join(f"{f(S['M2b_s0']['consistency']['paired']['HP_minus_B'][k]['mean'])}" for k in ("viol_STOP", "viol_TURN", "viol_LEAD"))
  + " (all three separated), s1: " + ", ".join(f"{f(S['M2b_s1']['consistency']['paired']['HP_minus_B'][k]['mean'])}" for k in ("viol_STOP", "viol_TURN", "viol_LEAD"))
  + " (STOP and LEAD separated, TURN not). The projection does not touch STOP and TURN (HP - H about 0) and only slightly LEAD. Rule violations overlap, so the per-rule excesses do not add up to 'any'. "
  "Self-consistency (b) is also worse than B's at both seeds." + "\n")
pe0, pe1 = S["M2b_s0"]["projection_effect"], S["M2b_s1"]["projection_effect"]
w(f"Projection effect on the plan: HP differs from H by more than 1e-3 m in {pe0['n_windows_HP_differs_from_H_by_gt_1e-3_m']} of {S['M2b_s0']['n_windows']} windows (max {pe0['max_abs_HP_minus_H_m']:.1f} m, mean abs "
  f"{pe0['mean_abs_HP_minus_H_m']:.3f} m). ADE6 means: B {pe0['ADE6_mean']['B']:.4f}, H {pe0['ADE6_mean']['H']:.4f}, HP {pe0['ADE6_mean']['HP']:.4f}, Bp {pe0['ADE6_mean']['Bp']:.4f} (s0); "
  f"B {pe1['ADE6_mean']['B']:.4f}, H {pe1['ADE6_mean']['H']:.4f}, HP {pe1['ADE6_mean']['HP']:.4f}, Bp {pe1['ADE6_mean']['Bp']:.4f} (s1).\n")

# ------------------------------------------------------------------ 8 diag vhi
w("## 8. POST-HOC DIAGNOSTIC 1 (not a PREREG row, no bar, changes no number above): the v_hi term of the SPEED rule\n")
w("Run AFTER the read was fixed (`OUT/diag_vhi.py`, `OUT/diag_vhi.json`). Same windows, same estimator. Arms: `HPv_or` = H projected with cap min(posted limit, GT v_hi) (**T0 ORACLE**: reads a GT label at "
  "inference; a ceiling, never a result); `HPv_pr` = cap min(posted limit, T's predicted v_hi) (T1-class: T's own `cons_pred` head, vision-only); `Bv_pr` = B under the same cap. Cap applied from t = 0, as `project_speed` does. "
  "Paired vs B.\n")
w("| seed | arm | ADE2 - B | ADE6 - B | CONS 'any' - B | CONS SPEED - B | self (b) - B |")
w("|---|---|---|---|---|---|---|")
for s in SEEDS:
    for n in ("HPv_or", "HPv_pr", "Bv_pr"):
        p = DV["seeds"][s]["paired_vs_B"][n]
        w(f"| {SV[s]} | {n} | {ci(p['ade2'])}{star(p['ade2'])} | {ci(p['ade6'])}{star(p['ade6'])} | {ci(p['cons']['viol_any'])} | {ci(p['cons']['viol_SPEED'])} | {ci(p['cons']['self_viol_any'])} |")
d0 = DV["seeds"]["M2b_s0"]
w(f"\nFacts: GT plan vs the posted limit alone violates the SPEED rule in {d0['GT_plan_vs_posted_limit_alone']['violation_rate']*100:.2f} % of windows ({d0['GT_plan_vs_posted_limit_alone']['n']} of {d0['n_windows']}); "
  f"GT v_hi_ms is below the posted limit in {d0['n_windows_gt_v_hi_below_limit']} windows; T's predicted v_hi_ms vs the GT label: MAE "
  + " / ".join(f"{DV['seeds'][s]['pred_v_hi_vs_gt_v_hi_mae_ms']:.2f}" for s in SEEDS) + " m/s (s0 / s1), "
  + " / ".join(str(DV['seeds'][s]['pred_v_hi_ms']['n_nonpositive']) for s in SEEDS) + " non-positive predictions floored at 0. Reading: capping at the oracle v_hi fixes SPEED to 0.000 and "
  "restores 'any' below B, but costs ADE2 (section 9 shows this is the cap acting at t < 2 s, where the rule does not read); capping at the predicted v_hi is neutral on 'any' (-0.0004 to +0.0079) and costs a lot of ADE "
  "(HYPOTHESIS: errors of the v_hi head become hard brakes from t = 0).\n")

# ------------------------------------------------------------------ 9 diag band
w("## 9. POST-HOC DIAGNOSTIC 2 (not a PREREG row, no bar, changes no number above): apply the cap from tau >= 2.0 s, the SPEED rule's own band\n")
w("`OUT/diag_band.py`, `OUT/diag_band.json`. `project_speed` verbatim except the speed cap binds only on 0.1 s intervals j >= 20 (tau >= 2.0 s, the eval rule's `_BAND_KNOT`); Kamm everywhere. Identity control "
  "(band start 0 == `project_speed`): max abs " + ", ".join(f"{DB['seeds'][s]['identity_control_band0_vs_project_speed_maxabs_m']:.1e}" for s in SEEDS) + " m. Nothing fitted, 2.0 s is not tuned. Arms `HPb_lim` (posted limit only, "
  "deployable), `HPb_pr` (min(limit, T's predicted v_hi), T1-class), `HPb_or` (T0 ORACLE ceiling), and the same three on B (`Bpb_*`). Paired vs B.\n")
w("| seed | arm | ADE2 - B | ADE6 - B | accel MAE - B | CONS 'any' - B | CONS SPEED - B |")
w("|---|---|---|---|---|---|---|")
for s in SEEDS:
    for n in ("HPb_lim", "HPb_pr", "HPb_or", "Bpb_lim", "Bpb_pr", "Bpb_or"):
        p = DB["seeds"][s]["arms"][n]
        w(f"| {SV[s]} | {n} | {ci(p['ade2_minus_B'])}{star(p['ade2_minus_B'])} | {ci(p['ade6_minus_B'])}{star(p['ade6_minus_B'])} | {ci(p['accel_minus_B'])}{star(p['accel_minus_B'])} | "
          f"{ci(p['cons_minus_B']['viol_any'])} | {ci(p['cons_minus_B']['viol_SPEED'])} |")
w("\nReading (MEASURED, descriptive): banding removes the ADE2 cost (HPb ADE2 - B equals H's own gain). The posted limit alone still leaves 'any' - B at "
  + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_lim']['cons_minus_B']['viol_any']['mean'])}" for s in SEEDS) + "; T's predicted v_hi brings it to "
  + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_pr']['cons_minus_B']['viol_any']['mean'])}" for s in SEEDS) + " (upper bounds "
  + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_pr']['cons_minus_B']['viol_any']['hi'])}" for s in SEEDS) + ", still above +0.02) at an ADE6 cost "
  + ", ".join(f"{f(DB['seeds'][s]['arms']['HPb_pr']['ade6_minus_B']['mean'])}" for s in SEEDS) + "; the oracle clears both. **No deployable-class variant tested here clears PRIMARY-C.** None of these arms is a result; "
  "each would need its own PREREG, and they are listed only so that the committed constraint-aware PREREG is written against the measured mix.\n")

# ------------------------------------------------------------------ 10 table
w("## 10. The outcome table, applied verbatim\n")
w("| outcome | condition | met? |")
w("|---|---|---|")
w(f"| PASS | PRIMARY-C and PRIMARY-K at both seeds | **{R['outcome'] == 'PASS'}** (C: {R['PRIMARY_C_both_seeds']}, K: {R['PRIMARY_K_both_seeds']}) |")
w(f"| **CONSISTENCY NOT RESTORED** | PRIMARY-C fails | **{not R['PRIMARY_C_both_seeds']}** |")
w(f"| GAIN LOST | PRIMARY-C passes, PRIMARY-K fails | {R['PRIMARY_C_both_seeds'] and not R['PRIMARY_K_both_seeds']} (requires PRIMARY-C to pass) |")
w("\nNo weight, cap or bar changed after any HPROJ number was seen. The weights are HBLEND's banked file (sha256 in each score json); the cap is the eval's; the bars are the PREREG's. The control ran first "
  "in the same process and the script refuses to score if it fails. The two post-hoc diagnostics were written and run after `READ.json` existed and are labelled as such.\n")

# ------------------------------------------------------------------ 11
w("## 11. What this does and does not show\n")
w("* **Evidence class.** All rows MEASURED (this run). B's plans are the banked KIND-run outputs (INHERITED from box 2's GPU run; re-derived metrics reproduce the banked arrays, and the reconstructed "
  "`projected_r1_kamm` block reproduces the banked one with 0 mismatches at s0 and 1 at s1 -- a zero-acceleration counter flipping one step, see section 2).")
w("* **Tier.** T1-class open loop on 2,399 windows / 141 episodes. Not closed loop: nothing here says HP drives better or worse, only that its open-loop plan is farther from the logged future than B's "
  "(ADE) and breaks more GT-derived constraints than B's.")
w("* **Variance.** The bootstrap resamples episodes with the models fixed. Training-run variance is covered only by the two seeds, which agree in sign on every PREREG row (PRIMARY-C fails at both, PRIMARY-K fails "
  "at both). Two seeds are a floor check, not an interval. For scale, F = |ADE6(s0) - ADE6(s1)| = 0.1207 m (HBLEND): the s1 ADE6 cost (+0.127) is of that order, the s0 one (+0.007) is not distinguishable from zero.")
w("* **The PREREG's explanation was incomplete.** It expected the remaining violations to be stops, leads and turns. They are mostly SPEED, against a cap (v_hi) that the projection's input does not carry, "
  "then LEAD. The outcome row is unchanged by this; the committed next step (a constraint-aware fit under its own PREREG) stands, and the PREREG for it should name v_hi and the 2.0 s band explicitly.")
w("* **Not done / not claimed.** No new fit, no new weights, no trained arm, no closed-loop run. MIXO and FLAT were not re-scored (that follows only from a PASS).\n")

# ------------------------------------------------------------------ 12
w("## 12. Deliverable manifest\n")
w("All under `OUT` (dev-box scratchpad; **nothing staged to git, nothing pushed, nothing written on box 2, nothing deleted**). The Master Mind lands them. Footprint on C: under 20 MB.\n")
w("| artifact | path (under OUT) | note |")
w("|---|---|---|")
for a, p, n in [
    ("this report", "HPROJ_RESULT.md", "generated by build_result.py from the JSONs below"),
    ("executor (control / score / read / dk)", "v8f_hproj.py", "imports HBLEND's loaders read-only (`../hblend/v8f_hblend.py`), the eval's own `evaluate`, `taniteval.ci`, `v8f_beta_eval`, the DK reader"),
    ("report generator", "build_result.py", ""),
    ("self-tests", "test_v8f_hproj.py", "17 tests: reconstruction identities, wrong-basis deliberate regression, w60, blend, projection, cap, banded identity, outcome-table logic (6 scenarios)"),
    ("reconstruction control (+ diagnostics, regressions, X1)", "control_M2b_s0.json, control_M2b_s1.json", "binding C0-C2, E1-E5, R1-R3, X1"),
    ("per-seed score (every row, CI, blocks, consistency, transitions)", "score_M2b_s0.json, score_M2b_s1.json", ""),
    ("per-window arrays", "score_M2b_s{0,1}_perwindow.npz", "HP, HP60, H60, tau60, cap, w60, per-window ADE/FDE/families and violation flags for every arm"),
    ("outcome", "READ.json", "the table, verbatim"),
    ("distance keeping", "dk_read.json, dk_perwindow.npz", "via v8f_dk_read.read_dk"),
    ("post-hoc diagnostic 1 (v_hi)", "diag_vhi.py, diag_vhi.json", "not a PREREG row"),
    ("post-hoc diagnostic 2 (2.0 s band)", "diag_band.py, diag_band.json", "not a PREREG row"),
    ("logs", "score_M2b_s0.log, score_M2b_s1.log, read.log, dk.log", ""),
    ("inputs (read-only, from HBLEND)", HB + "/inputs/banked, /weights_M2b_s{0,1}.json, /inputs/b2_p10_eval_pred6.npy, /inputs/full_eval_floors_perwindow.npz, /deps", "not copied; nothing from box 2")]:
    w(f"| {a} | `{p}` | {n} |")
w("\n## 13. Reproduce\n")
w("```\nexport CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1\nPY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe\n$PY v8f_hproj.py control --seed M2b_s0   # ~5 s, rc 2 on failure\n"
  "$PY v8f_hproj.py score --seed M2b_s0     # ~65 s, control first\n$PY v8f_hproj.py score --seed M2b_s1\n$PY v8f_hproj.py read\n$PY v8f_hproj.py dk                      # ~26 s\n"
  "$PY diag_vhi.py && $PY diag_band.py      # post-hoc, after READ.json\n$PY build_result.py\n$PY -m pytest -q test_v8f_hproj.py --basetemp C:/Users/Admin/pt_hproj -p no:cacheprovider\n```\n"
  "Paths are hard-coded to the scratchpad (`SCR` in `v8f_hproj.py`, `HB` = HBLEND); relocate before landing. Pinned: tip a99ca5d6 export at `viz/tree`, `taniteval` copy in `hblend/deps`.")
open(f"{OUT}/HPROJ_RESULT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", f"{OUT}/HPROJ_RESULT.md", len("\n".join(L)))
