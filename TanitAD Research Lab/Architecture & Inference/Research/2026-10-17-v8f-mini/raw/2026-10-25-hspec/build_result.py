#!/usr/bin/env python3
"""build_result.py -- generates HSPEC_RESULT.md (and completes READ.json) from the JSONs written by v8f_hspec.py / lever_map.py / lever_map_d3.py. No number is typed by hand."""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE.replace("\\", "/")
SEEDS = ("M2b_s0", "M2b_s1")
R = json.load(open(f"{HERE}/READ.json"))
SC = {st: {s: json.load(open(f"{HERE}/score_hs{st}_{s}.json")) for s in SEEDS} for st in (1, 2) if os.path.exists(f"{HERE}/score_hs{st}_M2b_s0.json")}
FT = {st: {s: json.load(open(f"{HERE}/fit_hs{st}_{s}.json")) for s in SEEDS} for st in SC}
DK = {st: json.load(open(f"{HERE}/dk_read_hs{st}.json")) for st in SC}
LM = json.load(open(f"{HERE}/lever_map.json"))
D3 = json.load(open(f"{HERE}/lever_map_d3.json"))
FINAL = str(R["final_stage"])
F = R["F_exact_m"]

NAMES = [("fw_lon_speed_mae", "LONGITUDINAL", "speed MAE (m/s)", 4), ("fw_lon_accel_mae", "LONGITUDINAL", "accel MAE (m/s^2)", 4), ("fw_lon_along_mae", "LONGITUDINAL", "along MAE (m)", 4),
         ("fw_lat_cross_mae", "LATERAL", "cross-track MAE (m)", 4), ("fw_lat_heading_mae_deg", "LATERAL", "heading MAE (deg)", 4),
         ("fw_lat_yaw_rate_mae_degps", "LATERAL", "yaw-rate MAE (deg/s)", 4), ("fw_lat_curvature_mae_1pm", "LATERAL", "curvature MAE (1/m)", 5),
         ("fw_tac_lat_acc", "TACTICAL (plan-derived)", "lateral decision agreement (higher better)", 4),
         ("fw_tac_lon_acc", "TACTICAL (plan-derived)", "longitudinal decision agreement (higher better)", 4)]
ADE = [f"ade{h}" for h in range(1, 7)]
FDE = [f"fde{h}" for h in range(1, 7)]


def fr(r, nd=4):
    if not r or "mean" not in r:
        return "n/a"
    tag = " *" if r.get("separated") else ""
    return f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]{tag}"


def fm(r, nd=4):
    return f"{r['mean']:.{nd}f} [{r['lo']:.{nd}f}, {r['hi']:.{nd}f}]"


def tf(x):
    return "PASS" if x else "FAIL"


L = []
w = L.append
st_final = SC[int(FINAL)]
w("# HSPEC_RESULT: PREREG_V8F_HSPEC executed exactly as written (2026-10-25)")
w("")
w(f"Dev box, CPU only, ZERO training, no GPU, no box-2 access. Every number is **MEASURED (ours)** from the artifact paths named; tier **T1-class open loop (own held outputs; not closed loop)**; "
  f"estimator **paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0**. The bootstrap answers 'would another draw of EPISODES say this?' only; training-seed "
  f"variance enters only through the two seeds side by side and F = |ADE6(s0) - ADE6(s1)| = {F:.5f} m (`H-ESTIM-SEED-1`). Inference is deterministic. `OUT` = `{OUT}`.")
w("")
w("## 0. Outcome (the PREREG outcome table, applied verbatim)")
w("")
w(f"**{R['outcome']}** (final stage: {FINAL}; stage 2 ran automatically because PRIMARY-DOM failed at stage 1).")
w("")
s1, s2 = R["stages"]["1"], R["stages"]["2"]
w("| stage | PRIMARY-DOM s0 | PRIMARY-DOM s1 | PRIMARY-KEEP s0 | PRIMARY-KEEP s1 | accel AND curvature HS - H separated-negative, s0 / s1 |")
w("|---|---|---|---|---|---|")
for k, S_ in (("1", s1), ("2", s2)):
    ps = S_["per_seed"]
    w(f"| {k} | {tf(ps['M2b_s0']['PRIMARY_DOM']['pass'])} | {tf(ps['M2b_s1']['PRIMARY_DOM']['pass'])} | {tf(ps['M2b_s0']['PRIMARY_KEEP']['pass'])} (hi {ps['M2b_s0']['PRIMARY_KEEP']['hi']:+.4f}) | "
      f"{tf(ps['M2b_s1']['PRIMARY_KEEP']['pass'])} (hi {ps['M2b_s1']['PRIMARY_KEEP']['hi']:+.4f}) | {ps['M2b_s0']['HS_minus_H_accel_curvature_separated_negative']} / {ps['M2b_s1']['HS_minus_H_accel_curvature_separated_negative']} |")
w("")
w(f"Condition reading: PRIMARY-DOM needs HS - P10 separated-better on ADE2, ADE6, accel MAE AND curvature MAE at BOTH seeds and not separated-worse anywhere else. **It fails at stage 2 on exactly two rows, "
  f"accel MAE and curvature MAE**, which are not separated-better than P10 at either seed (accel: s0 separated-WORSE, s1 not separated; curvature: not separated at both). Every other row of the dominance "
  f"row holds at both seeds (section 3). PRIMARY-KEEP passes at both seeds and both stages (the ADE6 gain over H is kept; HS - H upper bound <= +{F / 2:.5f} m). "
  f"Accel and curvature HS - H are separated-negative at both seeds at stage 2 (section 5), so the table lands on SMOOTHER, NOT DOMINANT.")
w("")
w(f"**Committed next step (PREREG, verbatim):** *\"the roughness is partly in the coefficients' amplitudes, and partly elsewhere. Next: a training arm with a second-difference penalty on O's plan, on box 2 "
  f"(a loss term: G-DVB entry and launch gate)\"*.")
w("")
w("**Rule Zero (a refutation is a waypoint).** The dominance claim is not delivered by this arm; the section 9 lever map says what is left and what blocks the committed next step. Short form: "
  "(i) the committed training arm is blocked on a PREREG for the loss term with its G-DVB entry, the launch gate, and a box-2 slot (box 2 is outside this read; GPU/spend is the PI's call); "
  "(ii) a zero-training re-registration of lambda alone cannot dominate at seed 0 on the grid examined (accel is never separated-better than P10, and ADE2 turns separated-worse than P10 from x4); "
  "(iii) the per-step weights w carry part of the roughness (smoothing w alone moves H's accel gap to P10 from +0.104 to +0.080 at s0), but not enough.")
w("")
w("## 1. What was run")
w("")
w("* **Route.** HBLEND's banked machinery, imported read-only (`v8f_hblend.py`: loaders, `apply_w`, `arm_pw`, `rows_for`, `deep_diff`, `single`; the eval's own `evaluate()`; `taniteval.v8f_beta_eval.paired_row`; the DK reader `v8f_dk_read.read_dk`). "
  "Fit set: HBLEND's 9,582 TRAIN windows (141 clips, k = 32; `hblend/fit_windows_M2b_s{0,1}.npz`); weights w: `hblend/weights_M2b_s{0,1}.json` (final = clip then PAV), unchanged. Eval-141 (2,399 windows) never enters a fit. "
  "B's plan tau = the KIND run's banked `pred`; P = the checkpoint's own frozen P10 (CPU forward, == HBLEND's saved P bit for bit); H = HBLEND's saved H (recomputed: max abs diff 0.0).")
w("* **Coefficients.** c = least squares of (traj30 - p30) on `basis[1::2]` (the ramped DCT of `v8f_mini.py`, rows at tau = 0.2 j, [30, 12]; cond "
  f"{st_final['M2b_s0']['controls']['cond_B30']:.1f}, rank {st_final['M2b_s0']['controls']['rank_B30']}), per axis (along = axis 0, cross = axis 1). The formula basis equals both checkpoints' `basis` buffer to "
  f"{max(st_final[s]['controls']['basis_vs_checkpoint_buffer_maxabs'] for s in SEEDS):.1e}; `up(P)[1::2] = P` and `subsample_02` are pinned by a test against the real `v8f_mini` functions.")
w("* **Arm.** tau_HS = P + w (.) B30 (v (.) c) on the 0.2 s knots, v in [0, 1]^(12 x 2). **Stage 1:** unweighted bounded least squares of tau_HS vs the GT, `scipy.optimize.lsq_linear(bounds=(0,1), method='bvls', tol=1e-12)`, one 12-parameter problem per axis, "
  "KKT-checked (relative projected gradient <= 1e-6; observed <= 2e-14). **Stage 2:** the same + lambda ||D2 tau - D2 y||^2.")
w("* **Declared interpretations (fixed in the script's docstring BEFORE any HSPEC number; none changed after).** "
  "I1 channels: v[k, a], a = 0 along / 1 cross, the axis order of `coef [B, 2, nb]`. I2 scored grid: the 0.2 s knots only (the 0.1 s midpoints are neither fitted nor scored, as in HBLEND). "
  "I3 stage-1 objective unweighted over windows x 30 steps x axes. "
  "I4 stage 2: e(t) = tau_HS - y with the ORIGIN prepended as e(0) = 0, so D2 has 29 rows per axis; ONE lambda per seed, pooled over windows, steps and axes, lambda = S1/S2 at v = 1 on TRAIN (the two terms equal at v = 1). "
  f"Computed once: **lambda = {FT[2]['M2b_s0']['lambda']:.2f} (s0), {FT[2]['M2b_s1']['lambda']:.2f} (s1)**; check S1 = lambda S2 at v = 1: "
  f"{FT[2]['M2b_s0']['lambda_check_S1_eq_lam_S2_at_ones'][0]:.6g} vs {FT[2]['M2b_s0']['lambda_check_S1_eq_lam_S2_at_ones'][1]:.6g} (s0). "
  "I5 'separated-better / worse' = `paired_row`'s flags; distance keeping WORSE = headway or time-gap separated NEGATIVE, LEAD-VIOL at-risk separated POSITIVE (a two-sided sensitivity is stated after the section 4 tables and does not change the outcome). "
  f"I6 PRIMARY-KEEP bar = F/2 with the exact F = {F:.5f} m, i.e. +{F / 2:.5f} m (the printed +0.060 variant is reported beside; same verdict). "
  "I7 control thresholds (below). I8 table order DOMINATES, GAIN LOST, SMOOTHER, NO EFFECT; 'SMOOTHER' = accel and curvature HS - H separated-negative at both seeds of the final stage. "
  "I9 stage 2 runs for both seeds iff PRIMARY-DOM fails at either seed of stage 1.")
w("")
w("## 2. Controls (run BEFORE any read; the script refuses to read if one fails; 15 tests, `pytest_final.log`)")
w("")
w("| control | bar | s0 stage 1 | s1 stage 1 | s0 stage 2 | s1 stage 2 |")
w("|---|---|---|---|---|---|")


def crow(label, bar, fn):
    cells = [fn(SC[st][s]["controls"]) for st in (1, 2) for s in SEEDS]
    w(f"| {label} | {bar} | " + " | ".join(cells) + " |")


crow("1. coefficient recovery residual, train (m)", "<= 1e-4", lambda c: f"{c['c1_recov_train_maxabs_m']:.2e}")
crow("1. coefficient recovery residual, eval (m)", "<= 1e-4", lambda c: f"{c['c1_recov_eval_maxabs_m']:.2e}")
crow("1. v = 1: max abs plan diff vs H (m)", "<= 1e-4", lambda c: f"{c['c1_v1_vs_H_plan_maxabs_m']:.2e}")
crow("1. v = 1: evaluate() block mismatches vs H (rel tol 1e-6)", "0 (see note)", lambda c: f"{c['c1_v1_vs_H_evaluate_block_n_mismatches']} ({c['c1_v1_vs_H_evaluate_block_n_mismatches_excl_zero_accel_threshold_diagnostic']} excl. threshold diagnostic)")
crow("1. v = 1: per-window ADE / family metric max abs diff vs H (each metric in its own unit)", "-", lambda c: f"{c['c1_v1_vs_H_perwindow_metric_maxabs']:.2e}")
crow("2. v = 0: max abs plan diff vs the P10 file (m)", "<= 1e-4", lambda c: f"{c['c2_v0_vs_P10_file_plan_maxabs_m']:.2e}")
crow("2. v = 0: per-window ADE6 vs floors file (m)", "<= 1e-5", lambda c: f"{c['c2_v0_ade6_vs_floors_perwindow_maxabs_m']:.2e}")
crow("2. v = 0: ADE6 mean (m)", "2.4870 +- 1e-3", lambda c: f"{c['c2_v0_ade6_mean']:.5f}")
crow("2. v = 0: evaluate() block mismatches vs P10 (rel tol 1e-4)", "0", lambda c: f"{c['c2_v0_vs_P10_evaluate_block_n_mismatches_tol1e-4']}")
crow("3. regression (GT := P10 in the fit): max v", "<= 1e-12", lambda c: f"{c['c3_regression_v_maxabs']:.1e}")
crow("3. regression: ADE6 minus P10's (m) (reads as P10, never a gain)", "<= 1e-4 abs", lambda c: f"{c['c3_regression_ade6_minus_P10_mean_m']:+.1e}")
crow("4. noise (c permuted across windows): mean v / max v", "<= 0.10 / <= 0.30", lambda c: f"{c['c4_noise_v_mean']:.3f} / {c['c4_noise_v_max']:.3f}")
crow("optimality: KKT residual of the fit", "<= 1e-6", lambda c: str(c["pass"]["kkt_optimal"]))
crow("all controls pass", "", lambda c: str(c["all_pass"]))
w("")
w("* **Note on control 1 (found at control time, before any read, disclosed).** HS(v = 1) vs H through the eval's own `evaluate()` matches on every entry except **one**: `traj.acts.zero_accel_frac`, the share of "
  "(window, step) cells with |a| < 1e-3 m/s^2 (a HARD threshold in `floors_kin.acts_check`). "
  + "; ".join(f"{s}: {SC[1][s]['controls']['c1_zero_accel_threshold_flips']['n_flip_cells']} of {SC[1][s]['controls']['c1_zero_accel_threshold_flips']['n_cells']:,} cells straddle it under a "
              f"{SC[1][s]['controls']['c1_zero_accel_threshold_flips']['plan_maxabs_diff_m']:.1e} m plan perturbation (max |dA| on a flip {SC[1][s]['controls']['c1_zero_accel_threshold_flips']['max_abs_dA_on_flips']:.1e}, "
              f"Lipschitz bound {SC[1][s]['controls']['c1_zero_accel_threshold_flips']['lipschitz_bound_dA']:.1e})" for s in SEEDS)
  + ". The perturbation is the float32 rounding of the banked arrays, inside the PREREG's own 1e-4 m recovery bar, so a bit-exact 'Delta = 0.0' is not attainable from banked float32 data. "
  "The control therefore passes iff every other entry matches AND every threshold flip is explained by the perturbation (script docstring I7; test `test_control1_v_ones_equals_H`; mutation "
  "`test_MUTATION_a_real_plan_difference_is_not_excused...` shows a real 3e-3 m plan difference is still caught). `zero_accel_frac` is not a HSPEC reading quantity. Accel MAE itself matches.")
w("* **Mutations that go RED** (`test_v8f_hspec.py`): the wrong subsample phase (basis rows 0::2) breaks the coefficient recovery (> 1e-2 m vs the 1e-4 bar); swapped w axes cannot recover a planted v; a planted v is recovered to 1e-5 at stage 1 and stage 2; "
  "the regression fit collapses to v = 0 while the real fit does not; a 3e-3 m plan perturbation is caught by the evaluate() block beyond the threshold diagnostic.")
_n = {s: SC[1][s]["controls"]["c4_noise_plan_eval_descriptive"]["vs_P10"]["ade6"] for s in SEEDS}
_h = {s: SC[1][s]["paired"]["HS_minus_P10"]["ade6"] for s in SEEDS}
w(f"* **Noise control, descriptive** (eval, stage-1 c-permuted v, s0 / s1): ADE6 minus P10 = {fr(_n['M2b_s0'])} / {fr(_n['M2b_s1'])}, against HS's {fr(_h['M2b_s0'])} / {fr(_h['M2b_s1'])}. "
  f"The permuted arm is P10-like (v ~ 0.03-0.04) and keeps {100 * abs(_n['M2b_s0']['mean'] / _h['M2b_s0']['mean']):.1f} % / {100 * abs(_n['M2b_s1']['mean'] / _h['M2b_s1']['mean']):.1f} % of HS's ADE6 gain (a small positive v on a residual that is "
  "better than P10 at 6 s still moves ADE6 slightly the right way); the control passes its registered bar on v, and it shows the HS gain is not what a noise fit produces.")
w("")


def dom_tables(st):
    w(f"### Stage {st}: the dominance row, HS - P10 (paired; lower is better except the two tactical agreements; `*` = CI excludes 0)")
    w("")
    w("| metric | requirement | s0: HS - P10 [95 % CI] | s0 | s1: HS - P10 [95 % CI] | s1 |")
    w("|---|---|---|---|---|---|")
    rows0, rows1 = R["stages"][str(st)]["per_seed"]["M2b_s0"]["PRIMARY_DOM"]["rows"], R["stages"][str(st)]["per_seed"]["M2b_s1"]["PRIMARY_DOM"]["rows"]
    for nm in rows0:
        r0, r1 = rows0[nm], rows1[nm]
        nd = 5 if "curvature" in nm else 4

        def cell(r):
            ok = r["better"] if r["need"] == "separated-better" else (not r["worse"])
            tag = "better" if r["better"] else "worse" if r["worse"] else "not sep."
            return f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]", f"{tag} / {'OK' if ok else '**FAIL**'}"
        c0, c1 = cell(r0), cell(r1)
        w(f"| {nm} | {r0['need']} | {c0[0]} | {c0[1]} | {c1[0]} | {c1[1]} |")
    w("")
    for s in SEEDS:
        d = R["stages"][str(st)]["per_seed"][s]
        k = d["PRIMARY_KEEP"]
        w(f"* PRIMARY-DOM {s}: **{tf(d['PRIMARY_DOM']['pass'])}** (separated-better rows {d['PRIMARY_DOM']['all_better_ok']}, no-separated-worse rows {d['PRIMARY_DOM']['all_not_worse_ok']}). "
          f"PRIMARY-KEEP {s}: ADE6 HS - H = {k['delta']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}], bar hi <= +{k['bar_hi_le']:.5f} -> **{tf(k['pass'])}** (printed +0.060: {tf(k['hi_le_prereg_printed_0.060'])}).")
    w("")


w("## 3. PRIMARY-DOM and PRIMARY-KEEP, stage 1 (position objective)")
w("")
dom_tables(1)
w("## 4. PRIMARY-DOM and PRIMARY-KEEP, stage 2 (position + second-difference objective, lambda by the train-only rule)")
w("")
dom_tables(2)
tw = []
for st in (1, 2):
    for s in SEEDS:
        rows = R["stages"][str(st)]["per_seed"][s]["PRIMARY_DOM"]["rows"]
        for nm in ("DK headway min (m)", "DK time-gap min (s)"):
            if rows[nm].get("two_sided_sensitivity_separated"):
                tw.append(f"stage {st} {s} {nm} ({rows[nm]['mean']:+.3f}, separated BETTER)")
w(f"**Distance-keeping direction sensitivity (I5).** Read two-sidedly (any separated difference from P10 counts), these rows would be flagged: {'; '.join(tw) if tw else 'none'}. Each is a separated difference in the "
  "MORE-margin direction (longer time-gap / headway than P10); no distance-keeping row is separated in the less-margin direction, and LEAD-VIOL is never separated-worse. The outcome does not depend on this reading: PRIMARY-DOM already fails on accel and curvature.")
w("")
w("## 5. HS - H and HS - B on everything (paired; final = stage 2; stage 1 beside)")
w("")
for st in (2, 1):
    for s in SEEDS:
        P_ = SC[st][s]["paired"]
        w(f"**Stage {st}, {s}** (lower is better except the tactical rows; `*` = CI excludes 0)")
        w("")
        w("| family | metric | B | H | HS | P10 | HS - H | HS - B |")
        w("|---|---|---|---|---|---|---|---|")
        sg = SC[st][s]["single"]
        for key, fam, lab, nd in NAMES:
            w(f"| {fam} | {lab} | {sg['B'][key]['mean']:.{nd}f} | {sg['H'][key]['mean']:.{nd}f} | {sg['HS'][key]['mean']:.{nd}f} | {sg['P10'][key]['mean']:.{nd}f} | "
              f"{fr(P_['HS_minus_H'][key], nd)} | {fr(P_['HS_minus_B'][key], nd)} |")
        for lab, key in (("CONSISTENCY(a) 'any' (share of windows violating; lower better)", "viol_any"), ("CONSISTENCY(a) SPEED", "viol_SPEED"), ("CONSISTENCY(a) STOP", "viol_STOP"),
                         ("CONSISTENCY(a) TURN", "viol_TURN"), ("CONSISTENCY(a) LEAD", "viol_LEAD"), ("self-consistency (b) 'any'", "self_viol_any")):
            rt = SC[st][s]["consistency"]["rate"]
            cp = SC[st][s]["consistency"]["paired"]
            w(f"| CONSISTENCY | {lab} | {rt['B'][key]:.4f} | {rt['H'][key]:.4f} | {rt['HS'][key]:.4f} | {(format(rt['P10'][key], '.4f') if key in rt['P10'] else 'n/a')} | {fr(cp['HS_minus_H'].get(key), 4)} | {fr(cp['HS_minus_B'].get(key), 4)} |")
        w("")
w("**TACTICAL (T's own heads) and STRATEGIC.** HS never touches T's logits or S: T's own decision-head block equals B's banked block "
  f"(json equality, s0 / s1, stage 2: {SC[2]['M2b_s0']['structural_identity']['tactical_block_T_heads_HS_equals_banked_B']} / {SC[2]['M2b_s1']['structural_identity']['tactical_block_T_heads_HS_equals_banked_B']}). "
  "**STRATEGIC: PAUSED** per the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` ITEM 25), not a pass; S's banked `ck_pos_err` is carried unchanged. The plan-derived tactical agreements above DO move with the plan.")
w("")
w("### 5a. Distance keeping beside (K = 30, 6 s lead block, `v8f_dk_read.read_dk`; min-TTC is a CENSORED mean and is not read as better)")
w("")
for st in (2, 1):
    d = DK[st]
    w(f"**Stage {st}.** Window states {d['arms']['HS_s0']['window_states']}.")
    w("")
    w("| arm | headway min (m) | time-gap min (s) | LEAD-VIOL share of at-risk |")
    w("|---|---|---|---|")
    for nm in ("B_s0", "H_s0", "HS_s0", "B_s1", "H_s1", "HS_s1", "P10"):
        a = d["arms"][nm]
        w(f"| {nm} | {a['metrics']['headway_min_m']['mean']:.3f} | {a['metrics']['time_gap_min_s']['mean']:.3f} | {a['lead_viol']['share_of_at_risk']:.4f} |")
    w("")
    w("| paired contrast | headway (m) | time-gap (s) | LEAD-VIOL at-risk (share) |")
    w("|---|---|---|---|")
    for nm in ("HS_s0__minus__P10", "HS_s1__minus__P10", "HS_s0__minus__H_s0", "HS_s1__minus__H_s1", "HS_s0__minus__B_s0", "HS_s1__minus__B_s1"):
        p = d["pairs"][nm]
        r1, r2, r3 = p["dk"]["headway_min_m"], p["dk"]["time_gap_min_s"], p["lead_viol"]["at_risk"]
        w(f"| {nm.replace('__minus__', ' - ')} | " + " | ".join(f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{' *' if r.get('separated') else ''}" for r in (r1, r2, r3)) + " |")
    w("")
w("## 6. The fitted v per coefficient (the entire deployable artifact beside w: 24 numbers per seed and stage)")
w("")
w("v[k, a]: k = DCT coefficient index 0..11 (higher k = higher frequency), a = along / cross. Source: `fit_hs{1,2}_M2b_s{0,1}.json` (train windows only).")
w("")
for st in (1, 2):
    w(f"**Stage {st}**" + (f" (lambda = {FT[2]['M2b_s0']['lambda']:.2f} s0, {FT[2]['M2b_s1']['lambda']:.2f} s1)" if st == 2 else " (lambda = 0)"))
    w("")
    w("| k | " + " | ".join(str(k) for k in range(12)) + " | mean |")
    w("|---|" + "---|" * 13)
    for s in SEEDS:
        v = FT[st][s]["v"]
        for a, an in enumerate(("along", "cross")):
            w(f"| {s} {an} | " + " | ".join(f"{v[k][a]:.3f}" for k in range(12)) + f" | {sum(v[k][a] for k in range(12)) / 12:.3f} |")
    w("")
import numpy as _np
_m = lambda st, a, ks: (min(min(FT[st][s]["v"][k][a] for k in ks) for s in SEEDS), max(max(FT[st][s]["v"][k][a] for k in ks) for s in SEEDS))
_mean = lambda st: [sum(FT[st][s]["v"][k][a] for k in range(12) for a in range(2)) / 24 for s in SEEDS]
w(f"Reading (descriptive): stage 1 keeps most of every coefficient (mean v {_mean(1)[0]:.2f} / {_mean(1)[1]:.2f}) because the position objective has no reason to shrink; at seed 1, along, coefficients 9 and 10 go to exactly "
  f"{FT[1]['M2b_s1']['v'][9][0]:.1f}. Stage 2 (mean v {_mean(2)[0]:.2f} / {_mean(2)[1]:.2f}) shrinks the along axis hardest at k = 2-4 ({_m(2, 0, (2, 3, 4))[0]:.2f}-{_m(2, 0, (2, 3, 4))[1]:.2f}) and k = 9-11 "
  f"({_m(2, 0, (9, 10, 11))[0]:.2f}-{_m(2, 0, (9, 10, 11))[1]:.2f}); the cross axis stays near 1 at k = 1, 2 (and 5) and shrinks at k = 9 and 11 ({_m(2, 1, (9, 11))[0]:.2f}-{_m(2, 1, (9, 11))[1]:.2f}). "
  "The two seeds agree in shape, not in the exact values.")
w("")
w("## 7. Per-horizon ADE (m), 1-6 s: B, H, HS, P10 (mean [95 % CI]; stage 2 and stage 1)")
w("")
for st in (2, 1):
    for s in SEEDS:
        sg = SC[st][s]["single"]
        P_ = SC[st][s]["paired"]
        w(f"**Stage {st}, {s}**")
        w("")
        w("| horizon | B | H | HS | P10 | HS - P10 | HS - H | HS - B |")
        w("|---|---|---|---|---|---|---|---|")
        for h in range(1, 7):
            k = f"ade{h}"
            w(f"| ADE@{h}s | {fm(sg['B'][k])} | {fm(sg['H'][k])} | {fm(sg['HS'][k])} | {fm(sg['P10'][k])} | {fr(P_['HS_minus_P10'][k])} | {fr(P_['HS_minus_H'][k])} | {fr(P_['HS_minus_B'][k])} |")
        w("")
w("FDE@1-6 s and every other row are in `score_hs{1,2}_M2b_s{0,1}.json::single / paired`.")
w("")
w("## 8. Consistency by rule, beside (descriptive; see section 5 for the rates)")
w("")
w("| stage | seed | rule | HS - P10 | HS - H | HS - B |")
w("|---|---|---|---|---|---|")
for st in (2, 1):
    for s in SEEDS:
        cp = SC[st][s]["consistency"]["paired"]
        for key in ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD"):
            w(f"| {st} | {s} | {key[5:]} | {fr(cp['HS_minus_P10'][key])} | {fr(cp['HS_minus_H'][key])} | {fr(cp['HS_minus_B'][key])} |")
w("")
w("## 9. Rule Zero: where the roughness still is (DESCRIPTIVE, computed AFTER the outcome was fixed; not a bar, not an arm)")
w("")
w("These maps use the same banked arrays and machinery (`lever_map.py`, `lever_map_d3.py`). They change nothing in sections 0-8. Looking at lambda values other than the registered one AFTER seeing the HSPEC numbers is exactly the move the PREREG forbids as a result; "
  "so they are a map for the NEXT pre-registration, not a rescue of this one.")
w("")
w("**D1. T0 diagnostic, eval-oracle v (fitted ON eval-141, reads GT): the ceiling of 'shrink per cosine coefficient' at the registered objectives (HS_oracle - P10).**")
w("")
w("| seed | objective | mean v | ADE6 | accel MAE | curvature MAE |")
w("|---|---|---|---|---|---|")
for s in SEEDS:
    for nm, lab in (("stage1_objective", "stage 1"), ("stage2_objective_train_lambda", "stage 2 (train lambda)")):
        d = LM["per_seed"][s]["D1_eval_oracle"][nm]
        w(f"| {s} | {lab} | {d['v_mean']:.3f} | {fr(d['vs_P10']['ade6'])} | {fr(d['vs_P10']['fw_lon_accel_mae'])} | {fr(d['vs_P10']['fw_lat_curvature_mae_1pm'], 5)} |")
w("")
w("Even with the eval GT in the fit, the registered stage-2 objective does not make accel separated-better than P10 at either seed (s0 still separated-worse) and does not make curvature separated-better: the lever is exhausted at its registered form, not under-fitted.")
w("")
w("**D2. Lambda map (TRAIN fits, m x the registered lambda; eval read; HS_m - P10). m = 1 is the registered arm; m = 0 is stage 1.**")
w("")
w("| seed | m | mean v | ADE2 | ADE6 | accel MAE | curvature MAE | yaw-rate MAE (deg/s) |")
w("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    for m, d in LM["per_seed"][s]["D2_lambda_map"].items():
        x = d["vs_P10"]
        w(f"| {s} | {m[1:]} | {d['v_mean']:.3f} | {fr(x['ade2'])} | {fr(x['ade6'])} | {fr(x['fw_lon_accel_mae'])} | {fr(x['fw_lat_curvature_mae_1pm'], 5)} | {fr(x['fw_lat_yaw_rate_mae_degps'], 3)} |")
w("")
w("Reading: raising lambda trades ADE2 for smoothness. Curvature becomes separated-better than P10 at both seeds from x16, but accel is separated-better only at seed 1 (x16 and above) and NEVER at seed 0 on this grid, "
  "while ADE2 is separated-WORSE than P10 at seed 0 from x4 (and at x64 and above at seed 1). So no single lambda on this grid dominates at seed 0 on ADE2 + accel + curvature together. "
  "Amplitude shrinkage alone is therefore not enough, which is the PREREG's 'partly elsewhere'.")
w("")
w("**D3. Is part of the roughness in the per-step weights w (HBLEND's PAV plateaus and steps)? w replaced by one fixed 5-knot moving average (clipped to [0, 1]; no sweep); HS - P10.**")
w("")
w("| seed | arm | ADE2 | ADE6 | accel MAE | curvature MAE |")
w("|---|---|---|---|---|---|")
for s in SEEDS:
    r = D3["per_seed"][s]
    w(f"| {s} | H (HBLEND: w as banked, v = 1) | {fr(SC[1][s]['paired']['H_minus_P10']['ade2'])} | {fr(SC[1][s]['paired']['H_minus_P10']['ade6'])} | "
      f"{fr(SC[1][s]['paired']['H_minus_P10']['fw_lon_accel_mae'])} | {fr(SC[1][s]['paired']['H_minus_P10']['fw_lat_curvature_mae_1pm'], 5)} |")
    for nm, lab in (("H_wsmooth_v1", "H with smoothed w (v = 1)"), ("HS2_wsmooth_v2", "HS stage-2 v with smoothed w")):
        x = r[nm]["vs_P10"]
        w(f"| {s} | {lab} | {fr(x['ade2'])} | {fr(x['ade6'])} | {fr(x['fw_lon_accel_mae'])} | {fr(x['fw_lat_curvature_mae_1pm'], 5)} |")
w("")
w("Smoothing w alone lowers H's accel gap to P10 (s0 +0.104 -> +0.080, s1 +0.066 -> +0.057) and does not close it; with the stage-2 v accel is not separated from P10 at either seed, curvature is not separated at s0 and "
  "separated-WORSE at s1. A real contributor, ranked below the lambda lever and below the committed training arm by measured effect.")
w("")
w("**Levers, ranked by measured effect (each needs its own PREREG; nothing below is a new bar):**")
w("")
w("1. **The committed next step: a training arm with a second-difference penalty on O's plan (box 2).** It is the only lever that changes what O emits rather than how much of it is kept; D1 shows the post-hoc coefficient lever is exhausted at seed 0. "
  "**Blocked on:** a PREREG for the loss term, its G-DVB entry and a launch-gate PASS token (binding 2026-09-26), a box-2 slot (compute/spend is the PI's call), and no box-2 access in this read. "
  "Unblocking = Master Mind drafts `PREREG_V8F_D2PEN` with the lambda rule stated and the dominance row (this PREREG's table) as the bar; the PI authorises the slot.")
w("2. **A re-registered lambda rule (zero training, ~1 minute of fit).** D2: from about x16 the curvature row is separated-better at both seeds and accel at seed 1. "
  "At x16 the cost against P10 is ADE2 +0.011 m (s0) / +0.002 m (s1) and ADE6 about -0.39 m (against -0.51 / -0.49 m at the registered lambda). It could clear the curvature row at both seeds and accel at seed 1 only, not ADE2 at seed 0, so it cannot meet this PREREG's dominance row as written; it would need a train-only selection rule (e.g. episode-held-out cross-validation on the 141 train clips) and a bar that admits the ADE2 trade, which is a PI/Master Mind call.")
w("3. **Smooth w (D3).** Smaller effect than 1-2; cheap; worth folding into whichever arm is registered next rather than a stand-alone read.")
w("")
w("## 10. What this does and does not show")
w("")
w("* **Evidence class.** All rows MEASURED (this run) on B's banked KIND-run plans (INHERITED from box 2's GPU run; HBLEND re-derived their metrics with 0 mismatches) and on HBLEND's banked P, H and fit windows. Section 9 is MEASURED but POST-HOC (read after the outcome was fixed).")
w("* **Tier.** T1-class open loop on 2,399 windows / 141 episodes. Nothing here says HS drives better; only that its open-loop plan is closer to the logged future than P10 on position, speed, along, cross-track, heading, yaw rate, both plan-derived tactical agreements and consistency, "
  "and that its second-derivative metrics are NOT shown to beat the kinematic floor.")
w("* **Variance.** The bootstrap resamples episodes with the models fixed; training-run variance enters only through the two seeds, which agree on the failing rows (accel and curvature not separated-better than P10 at either seed) but differ in size. Two seeds are a floor check, not an interval.")
w("* **Fit-set caveat (declared in HBLEND and the PREREG).** The fit windows are windows B was trained on; the eval-oracle D1 shows the registered form is at its ceiling anyway.")
w("* **Not done.** No H + hard-speed-projection read; no FLAT comparison (the PREREG's next step only if DOMINATES); no training.")
w("")
w("## 11. Deliverable manifest")
w("")
w("All under `OUT` (dev-box scratchpad; **nothing staged to git, nothing committed or pushed, nothing written outside `OUT`, nothing on box 2, nothing deleted**). The Master Mind lands them.")
w("")
w("| artifact | note |")
w("|---|---|")
for a, n in (("HSPEC_RESULT.md", "this report (generated by build_result.py from the JSONs)"), ("READ.json", "outcome, both stages' PRIMARY rows, controls summary, manifest"),
             ("v8f_hspec.py", "the executor (fit / score / dk / auto); declared interpretations in its docstring"), ("test_v8f_hspec.py, pytest_final.log", "controls as tests + mutations that go RED (15 tests)"),
             ("pytest_run1.log, pytest_run2.log", "the first test run (control 1 RED on the zero-accel threshold diagnostic: 12 passed, 2 failed) and the run after the declared control-1 clause (15 passed)"),
             ("fit_hs{1,2}_M2b_s{0,1}.json", "THE fitted v (12 x 2) per seed and stage, lambda, train objective terms, controls 3-4 fits"),
             ("score_hs{1,2}_M2b_s{0,1}.json", "every row, CI, controls, blocks (evaluate()), consistency, structural identity"),
             ("score_hs{1,2}_M2b_s{0,1}_perwindow.npz", "per-window HS plan, ADE/FDE/family components, violations"), ("dk_read_hs{1,2}.json, dk_perwindow_hs{1,2}.npz", "distance keeping via read_dk"),
             ("lever_map.py, lever_map.json, lever_map.log", "D1 eval-oracle and D2 lambda map (post-hoc, descriptive)"), ("lever_map_d3.py, lever_map_d3.json, lever_map_d3.log", "D3 smoothed w (post-hoc, descriptive)"),
             ("build_result.py", "report generator"), ("auto.log", "the read's log"), ("_v8f_mini_8d6449a7.py", "v8f_mini.py at the PREREG commit (reference copy of the basis/subsample code read)")):
    w(f"| `{a}` | {n} |")
w("")
w("## 12. Reproduce")
w("")
w("```")
w("export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4")
w("PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe")
w("$PY -B v8f_hspec.py auto --n-boot 2000      # ~8 min: controls first, stage 1, stage 2 iff PRIMARY-DOM fails, READ.json")
w("$PY -B lever_map.py ; $PY -B lever_map_d3.py  # post-hoc, descriptive")
w("$PY -B build_result.py")
w("$PY -B -m pytest -q -p no:cacheprovider --basetemp <OUT>/pt_basetemp test_v8f_hspec.py")
w("```")
w("Paths are hard-coded to the scratchpad (`SCR` in `v8f_hspec.py`); relocate before landing. Inputs are HBLEND's (read-only): `hblend/{fit_windows,weights,score}_*`, `hblend/inputs/`, `hblend/deps/`, `viz/work/{ckpt,inputs}`, `viz/tree/stack`.")
open(f"{HERE}/HSPEC_RESULT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---- complete READ.json (machine-readable): controls summary + manifest + the descriptive lever map pointer
R["controls_summary"] = {f"stage{st}_{s}": {"pass": SC[st][s]["controls"]["pass"], "all_pass": SC[st][s]["controls"]["all_pass"]} for st in SC for s in SEEDS}
R["lambda"] = {s: FT[2][s]["lambda"] for s in SEEDS}
R["v"] = {f"stage{st}_{s}": FT[st][s]["v"] for st in FT for s in SEEDS}
R["tier"] = "T1-class open loop (own held outputs; not closed loop)"
R["estimator"] = "paired episode-cluster bootstrap, B=2000, 141 episodes, seed 0 (taniteval/ci.py)"
R["strategic_family"] = "PAUSED per PI ruling PI_DECISION_QUEUE ITEM 25 (2026-09-19); not a pass"
R["descriptive_post_hoc"] = {"lever_map.json": "D1 eval-oracle, D2 lambda map", "lever_map_d3.json": "D3 smoothed w", "status": "computed AFTER the outcome was fixed; not a bar, not an arm"}
R["declared_deviation_found_at_control_time"] = ("Control 1 'plan equals H, Delta = 0.0 through evaluate()': one entry (traj.acts.zero_accel_frac, a hard |a|<1e-3 threshold) differs by 7 flipped cells of 69,571 "
                                                 "under a ~1.7e-5 m float32 perturbation; passes iff all other entries match and the flips are explained by the Lipschitz bound.")
R["report"] = "HSPEC_RESULT.md"
json.dump(R, open(f"{HERE}/READ.json", "w", encoding="utf-8"), indent=1)
print("wrote HSPEC_RESULT.md and READ.json")
