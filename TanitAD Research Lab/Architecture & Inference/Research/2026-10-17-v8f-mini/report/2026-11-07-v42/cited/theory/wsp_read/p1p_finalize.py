#!/usr/bin/env python3
"""p1p_finalize.py -- SPEC_P1P_AMEND_1.md: amended N2/N3 flags from the stored numbers, then P1P_RESULT.md.  Reads P1P_RESULT.json and P1P_REFIT_R1.json only (no new data).  python -I p1p_finalize.py"""
import hashlib
import json
import os

N_MIN = 30

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
R = json.load(open(f"{HERE}/P1P_RESULT.json"))
R1 = json.load(open(f"{HERE}/P1P_REFIT_R1.json"))
P = R["PRIMARY"]
ps = P["per_seed"]


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


# ---- amended N2 / N3 (statistic level)
n2 = R["N2_identity_rho_equals_1"]["per_seed"]
n3 = R["N3_planted_rho_equals_posterior"]["per_seed"]
am = {"N2": bool(all(r["median_conf"] == 1.0 and r["median_unsure"] == 1.0 and r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and not r["separated"] and not r["positive"] for r in n2.values())),
      "N3": bool(all(r["delta"] > 0 and r["separated"] for r in n3.values()))}
R["controls_amended"] = {"rule": "SPEC_P1P_AMEND_1.md: N2 and N3 read on the statistic, the group-size rule set aside", **am}
R["verdict"] = P["verdict"]
R["verdict_reason"] = (f"CONFIDENT holds {ps['s0']['n_conf']} windows at s0 and {ps['s1']['n_conf']} at s1 (floor 30); UNSURE holds {ps['s0']['n_unsure']} and {ps['s1']['n_unsure']}. "
                       "The criterion names this NOT EVALUABLE, and the groups are not padded.")
json.dump(R, open(f"{HERE}/P1P_RESULT.json", "w"), indent=1)

ctrl_ok = {"C4": R["C4_unit_tests"]["all_pass"], "R1": R["R1_refit_reproduces_banked"]["pass"], "R2": R["R2_all_train_vs_banked_eval141"]["pass"], "R3": R["R3_eval_feature_construction"]["pass"],
           "N1": R["N1_posterior_shuffle"]["pass"], "N2 (amended)": am["N2"], "N3 (amended)": am["N3"]}
S = R["SENSITIVITIES_not_the_verdict"]
k1, k2, k3, k124 = R["K1_probe_auc_eval_pmove"], R["K2_probe_auc_eval_band_3_6"], R["K3_probe_auc_for_a_human_gt_0p5_band_3_8"], R["K1_go24_auc_eval_pmove"]
r2, r3 = R["R2_all_train_vs_banked_eval141"], R["R3_eval_feature_construction"]
steps = P["window_filter_steps"]
bins = P["posterior_bins"]
s1_steps = S["S1_in_domain_v0_3_6"]["window_filter_steps"]

L = []
w = L.append
w("# P1P_RESULT: P1′ (THEORY_CRAWL_AMEND_1 §4) on the T1-kin eval-141 tick-0 windows")
w("")
w(f"**VERDICT, as the criterion is written: {P['verdict']}.** The CONFIDENT group holds {ps['s0']['n_conf']} windows (floor 30) at both seeds; the UNSURE group holds {ps['s0']['n_unsure']}. "
  "The criterion names this case and says it is never padded, so P1′ is neither held nor failed, and T1′ is neither supported nor contradicted by this read.")
w("")
w(f"The probe can test T1′ in principle: its AUC on the eval windows is {f(k2['auc'])} [{f(k2['lo'])}, {f(k2['hi'])}] in the P1′ band where its label exists (v0 in [3, 6), n {k2['n']}, {k2['n_clips']} episodes) "
  f"and {f(k1['auc'])} [{f(k1['lo'])}, {f(k1['hi'])}] on the whole P-MOVE range; neither interval covers 0.5.")
w("")
w("Evidence class MEASURED (ours, dev-box CPU, zero planner training, no weight edit). Tier: a tick-0 plan-level read of the T1-kin dumps (the plan is made from the logged history); not a driving result. "
  "Estimator: episode-cluster bootstrap over the eval-141 episodes, B 2,000, seed 0 (the tp_lib construction); it answers \"another draw of EPISODES\" and is blind to training and inference variance (H-ESTIM-SEED-1). The two planner seeds are read separately.")
w("")
w("## 1. Spec and order of work")
w("* `SPEC_P1P.md` sha256 `" + R["spec_sha256"] + "` (frozen file `SPEC_P1P.sha256`: 2026-11-06T23:16:43Z, i.e. 00:16:44 +0100 on 2026-11-07), written before any eval-141 posterior or ρ existed. The probe refit (train rows only) ran after it; the eval read ran after that.")
w("* `SPEC_P1P_AMEND_1.md` reports one defect in the wording of controls N2 and N3 (verdict-string wording that the group-size gate makes unmeetable); no criterion changes. `THEORY_CRAWL_AMEND_1.md` sha256 `" + R["amend_sha256"][:16] + "...` unchanged.")
w("* Probe: P-MOVE (the only population that overlaps v0 in 3–8; P-STOP is defined on v0 < 0.5), label GO (the registered launch mask, the probe's decision label), V+E linear. Five fold models refit by the committed `tp_lib.py` (byte-identical copy `p1p_tp_lib.py`) on the probe's own 12,000 windows and outer folds; posterior = mean of the five `predict_proba`. Chosen C: 0.1 at all five folds and for the all-train model.")
w("* Run history, for the record: `p1p_probe_refit.py` was started three times (two stray duplicate processes were killed by PID; the outputs are from the third, single run). `p1p_read.py` stopped twice on code errors (a string-array comparison with `equal_nan`; the missing `taniteval` import path) before any feature, posterior or ρ was computed; the third launch is the one reported, and the code fixes touched no statistic.")
w("")
w("## 2. Window set and group counts")
w("")
w(f"| step (primary, v0 in [3, 8), `plan`) | windows |\n|---|---|\n| tick-0 rows of the dump | {steps['tick0']} |\n| v0 in [3, 8) | {steps['v0_band']} |\n| and gt_ok | {steps['gt_ok']} |\n| and a_human > 0.5 m/s² | {steps['a_human_gt_0p5']} |\n| and all of a_human, â_plan, ρ, posterior finite | {steps['finite']} ({ps['s0']['n_episodes']} episodes) |")
w("")
w(f"Posterior bins over the {steps['finite']} windows: below 0.35: {bins['lt_0.35']}; UNSURE [0.35, 0.65]: {bins['0.35_0.65_UNSURE']}; (0.65, 0.8): {bins['0.65_0.8']}; CONFIDENT ≥ 0.8: {bins['ge_0.8_CONFIDENT']}. "
  f"The windows, a_human and the posterior do not depend on the planner seed, so both seeds hold the same groups. CONFIDENT spans {ps['s0']['n_episodes_conf']} episodes, UNSURE {ps['s0']['n_episodes_unsure']}.")
w("")
w(f"Domain note: the probe was trained on v0 below 6. {steps['finite'] - s1_steps['finite']} of the {steps['finite']} filtered windows have v0 in [6, 8) and were scored by extrapolation; "
  f"they place {bins['ge_0.8_CONFIDENT'] - S['S1_in_domain_v0_3_6']['posterior_bins']['ge_0.8_CONFIDENT']} windows in CONFIDENT and {bins['0.35_0.65_UNSURE'] - S['S1_in_domain_v0_3_6']['posterior_bins']['0.35_0.65_UNSURE']} in UNSURE (S1 below restricts to the in-domain band).")
w("")
w("## 3. The statistic, shown as DESCRIPTIVE because the groups are below the floor (it is not the verdict)")
w("")
w("| seed | n CONF / UNSURE | median ρ CONFIDENT [95 %] | median ρ UNSURE [95 %] | Δ [95 %] | Δ > 0 and separated |\n|---|---|---|---|---|---|")
for s in ("s0", "s1"):
    r = ps[s]
    w(f"| {s} | {r['n_conf']} / {r['n_unsure']} | {f(r['median_conf'])} [{f(r['median_conf_ci'][0])}, {f(r['median_conf_ci'][1])}] | {f(r['median_unsure'])} [{f(r['median_unsure_ci'][0])}, {f(r['median_unsure_ci'][1])}] | "
      f"{f(r['delta'])} [{f(r['lo'])}, {f(r['hi'])}] | {'yes' if r['positive'] and r['separated'] else 'no'} |")
w("")
w(f"Median a_human (m/s²) is {f(ps['s0']['median_a_human_conf'])} in CONFIDENT and {f(ps['s0']['median_a_human_unsure'])} in UNSURE; median â_plan is {f(ps['s0']['median_a_plan_conf'])} / {f(ps['s0']['median_a_plan_unsure'])} at s0 and "
  f"{f(ps['s1']['median_a_plan_conf'])} / {f(ps['s1']['median_a_plan_unsure'])} at s1. The direction is the one P1′ predicts and the interval excludes 0, but it rests on {ps['s0']['n_conf']} windows from {ps['s0']['n_episodes_conf']} episodes in one group. "
  "The rule treats that as too few to read, and this report does the same: no verdict is drawn from this table.")
w("")
w("## 4. Controls")
w("")
w("| control | what it checks | result |\n|---|---|---|")
w(f"| C4 unit tests | weighted median = `np.median` of the replicated array; bootstrap counts = tp_lib construction; group edges at 0.8 / 0.35 / 0.65 | {'pass' if ctrl_ok['C4'] else 'FAIL'} |")
w(f"| R1 refit reproduces the banked artifacts (train only) | five fold PCAs vs `pca_V_pmove.npz` S; held-out scores vs `oof_pmove_V_linear.npz` (`V+E`, `GO24__V+E`) | {'pass' if ctrl_ok['R1'] else 'FAIL'}: PCA relative difference ≤ {max(v for k, v in R['R1_refit_reproduces_banked']['pca_max_abs_diff_rel'].items() if k != '5'):.1e}; held-out score difference ≤ {max(list(R['R1_refit_reproduces_banked']['heldout_vs_banked_oof_max_abs_go'].values()) + list(R['R1_refit_reproduces_banked']['heldout_vs_banked_oof_max_abs_go24'].values())):.1e} (tolerance 1e-6; the same fit on the banked PCA scores differs by ≤ 3e-13, so the gap is PCA float noise); per-fold AUC equals the banked value to 4 places |")
w(f"| R2 all-train model vs the banked eval-141 confirmation | AUC of linear V+E on the 507 windows with 0.5 ≤ v0 < 6 | {'pass' if ctrl_ok['R2'] else 'FAIL'}: {r2['auc']:.4f} vs banked {r2['banked_auc']:.4f}; C {r2['refit_C']} vs {r2['banked_hp_C']} |")
w(f"| R3 eval feature construction | E, V, GO label, v0 on the {r3['overlap']} windows `eval141_features.npz` covers | {'pass' if ctrl_ok['R3'] else 'FAIL'}: E max abs {r3['E_max_abs']}, V bit-equal {r3['V_bit_equal']}, GO equal {r3['go_equal']}, v0 vs kin3 max abs {r3['v0_dump_vs_kin3_max_abs']} |")
w(f"| K1 probe AUC on eval, P-MOVE range | posterior vs GO, 0.5 ≤ v0 < 6 | {f(k1['auc'])} [{f(k1['lo'])}, {f(k1['hi'])}], n {k1['n']} ({k1['n_pos']} GO, {k1['n_clips']} episodes); above chance |")
w(f"| K2 probe AUC on eval, P1′ band | posterior vs GO, 3 ≤ v0 < 6 | {f(k2['auc'])} [{f(k2['lo'])}, {f(k2['hi'])}], n {k2['n']}; above chance |")
w(f"| K3 two-second go indicator | posterior vs a_human > 0.5, 3 ≤ v0 < 8 (descriptive) | {f(k3['auc'])} [{f(k3['lo'])}, {f(k3['hi'])}], n {k3['n']} ({k3['n_pos']} positive) |")
w(f"| (K1 on GO24) | GO24 posterior vs GO24, 0.5 ≤ v0 < 6 (descriptive) | {f(k124['auc'])} [{f(k124['lo'])}, {f(k124['hi'])}] |")
w(f"| N1 posterior shuffle | 20 permutations of the posterior, real ρ | {'pass' if ctrl_ok['N1'] else 'FAIL'}: separated {R['N1_posterior_shuffle']['separated_of_20']['s0']} of 20 at s0, {R['N1_posterior_shuffle']['separated_of_20']['s1']} of 20 at s1 (limit 3) |")
w(f"| N2 identity (ρ ≡ 1) | exact values | {'pass' if ctrl_ok['N2 (amended)'] else 'FAIL'} on the statistic (medians 1.0, Δ 0.0, interval [0, 0], not separated); as run, the original verdict-string flag reads false (defect, AMEND_1) |")
w(f"| N3 planted (ρ = posterior) | the pipeline sees a planted mechanism | {'pass' if ctrl_ok['N3 (amended)'] else 'FAIL'} on the statistic (Δ {f(n3['s0']['delta'])} [{f(n3['s0']['lo'])}, {f(n3['s0']['hi'])}], separated); as run, the verdict-string flag reads false (defect, AMEND_1) |")
w("")
w("## 5. Sensitivities (declared in SPEC_P1P.md §4; none is the verdict, each through the identical verdict logic)")
w("")
w("| variant | n CONF / UNSURE | Δ s0 [95 %] | Δ s1 [95 %] | verdict logic returns |\n|---|---|---|---|---|")
for k, v in S.items():
    a, b = v["per_seed"]["s0"], v["per_seed"]["s1"]
    w(f"| {v['name']} | {a['n_conf']} / {a['n_unsure']} | {f(a['delta'])} [{f(a['lo'])}, {f(a['hi'])}] | {f(b['delta'])} [{f(b['lo'])}, {f(b['hi'])}] | {v['verdict']} |")
w("")
s4 = S["S4_go24_label"]
w(f"S1, S2 and S3 are {'also below the floor' if all(S[k]['verdict'] == 'NOT EVALUABLE' for k in ('S1_in_domain_v0_3_6', 'S2_raw_plan', 'S3_all_train_probe')) else 'NOT all below the floor (see the table)'}. "
  f"S4 (the probe's secondary label GO24, a different probe from the one AMEND_1 names) reaches {s4['per_seed']['s0']['n_conf']} / {s4['per_seed']['s0']['n_unsure']} windows and the logic returns {s4['verdict']}; "
  "that is a reading of a different go/stay probe, declared in advance as non-decisive, and it is not promoted here.")
w("")
w("## 6. What blocks, and the cheapest next steps (the Master Mind's to pre-register; none was run)")
w(f"* **The blocker is group size, not direction.** The primary band holds {steps['finite']} windows with a_human > 0.5 from {ps['s0']['n_episodes']} episodes; the CONFIDENT bin takes {ps['s0']['n_conf']} of them, {N_MIN - ps['s0']['n_conf']} short of the floor. The {bins['0.65_0.8']} windows in (0.65, 0.8) are the neighbouring bin and are not CONFIDENT under the criterion; no threshold is moved here.")
w(f"* **Next step 1 (zero compute):** a dated pre-registration, before it is read, that names the evaluable variant it wants: either the GO24 probe as the banked go/stay probe (S4 already shows {s4['per_seed']['s0']['n_conf']} / {s4['per_seed']['s0']['n_unsure']}, "
  f"Δ {f(s4['per_seed']['s0']['delta'])} / {f(s4['per_seed']['s1']['delta'])} with lower bounds {f(s4['per_seed']['s0']['lo'])} / {f(s4['per_seed']['s1']['lo'])}), or a lower CONFIDENT threshold with its own committed floor. Either changes AMEND_1 and needs its own sha before any reading is quoted.")
w("* **Next step 2 (CPU, no training):** more windows. The tick-0 plan of C0C needs only a forward pass; a second held-out window set with cached `pool4x10`, logged history and logged future would add windows to both groups. Blocker: no such second set is in the artifacts store that this read has touched (UNVERIFIED that none exists elsewhere); the training windows are not admissible because the planner was trained on them.")
w("* Ticks 1 and 2 of the dump cannot supply windows: their history is the arm's own and no probe feature exists for them.")
w("")
w("## 7. Provenance and manifest (all under `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read/`, nothing staged, committed or pushed; each exists in this one place only)")
w("")
w("| artifact | role | sha256 (first 16) |\n|---|---|---|")
for fn, role in (("SPEC_P1P.md", "pre-registered spec"), ("SPEC_P1P.sha256", "spec hash and time"), ("SPEC_P1P_AMEND_1.md", "defect note on N2/N3 wording"), ("p1p_tp_lib.py", "byte copy of trigger_probe/tp_lib.py"),
                 ("p1p_probe_refit.py", "train-only refit and R1"), ("p1p_read.py", "eval read, controls, sensitivities"), ("p1p_finalize.py", "amended N2/N3 flags and this report"),
                 ("P1P_PROBE_MODELS.npz", "refit PCA bases, scalers, coefficients (6 models, GO and GO24)"), ("P1P_REFIT_R1.json", "R1 numbers"), ("P1P_PERWINDOW.npz", "per-window posteriors, a_human, a_plan, ρ"),
                 ("P1P_RESULT.json", "machine-readable result"), ("p1p_probe_refit.log", "log"), ("p1p_read.log", "log")):
    p = f"{HERE}/{fn}"
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else "(missing)"
    w(f"| `{fn}` | {role} | `{h}` |")
w("")
w("Inputs read, unmodified: the two T1-kin dumps `C:/Users/Admin/rr_scratch/out/C0C_s{0,1}/ticks/ticks_C0C_s{0,1}.npz`; `trigger_probe/raw/{windows_train,sample_pmove,oof_pmove_V_linear,pca_V_pmove,eval141_features}.npz`, `results_eval141.json`; "
  "`F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch/{V_pmove,V_eval141}.npy`; the eval cache `sfresh_inputs/cache/eval/clips`; `sfresh_inputs/inputs/native_tracks.pt`. "
  "`WSP_READ.md` / `WSP_READ.json` are not edited; their P1′ row is the Master Mind's to update from this file.")
open(f"{HERE}/P1P_RESULT.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote P1P_RESULT.md; amended controls", am, "ctrl_ok", ctrl_ok)
