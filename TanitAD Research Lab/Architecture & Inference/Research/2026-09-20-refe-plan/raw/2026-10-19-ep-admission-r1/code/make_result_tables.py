"""Markdown tables of sim_results.json (so RESULT.md quotes numbers by machine, not by hand).  python make_result_tables.py <sim_results.json> <precompute.json> [<repro.json>]"""
import json
import sys

res = json.load(open(sys.argv[1], encoding="utf-8"))
pre = json.load(open(sys.argv[2], encoding="utf-8"))
C = res["conditions"]


def pc(r, d=1):
    return f"{100 * r['rate']:.{d}f} % [{100 * r['wilson95'][0]:.{d}f}, {100 * r['wilson95'][1]:.{d}f}] ({r['k']}/{r['n']})"


out = []
out.append("### T1. FALSE-REFUSAL on CALIBRATED replicates (generator A = log-resample; N = %d per condition; Wilson 95 %%)\n" % res["n_reps"])
out.append("| geometry (changed picks, seed 0 / 1) | A1-r1, 2 seeds (any-seed) | A1-r1 seed 0 | A1-r1 seed 1 | both seeds | A1 DRAFT (any-seed) | A2-r1, 3 heads x 2 seeds | A2 DRAFT | A1 or A2 (r1) |")
out.append("|---|---|---|---|---|---|---|---|---|")
for g in ("G78", "G25", "G05"):
    c = C[f"{g}|A|CAL"]
    m0, m1 = pre["meta"][f"{g}|s0"]["n_changed"], pre["meta"][f"{g}|s1"]["n_changed"]
    a1, a2 = c["A1"], c["A2"]
    out.append(f"| {g} ({m0} / {m1}) | **{pc(a1['r1_any_seed'])}** | {pc(a1['r1_seed0'])} | {pc(a1['r1_seed1'])} | {pc(a1['r1_both_seeds'])} | {pc(a1['draft_any_seed'])} | **{pc(a2['r1_any_head_any_seed'])}** | {pc(a2['draft_any_head_any_seed'])} | {pc(c['gate_r1_A1_or_A2'])} |")
b = C["G78|B|CAL"]
out.append(f"| G78, generator B (wild sign-flip, A1 only) | **{pc(b['A1']['r1_any_seed'])}** | {pc(b['A1']['r1_seed0'])} | {pc(b['A1']['r1_seed1'])} | {pc(b['A1']['r1_both_seeds'])} | {pc(b['A1']['draft_any_seed'])} | n/a | n/a | n/a |")
out.append("")
out.append("Independence reference for A1-r1 on G78 (1 - (1 - p0)(1 - p1), the rate the two seeds would give if they failed independently): %.1f %%; the observed any-seed rate is the measured one above." % (100 * C["G78|A|CAL"]["A1"]["r1_indep_expected_any"]))
out.append("")
out.append("A1-r1 seed-level failures by conjunct on the calibrated G78 replicates (2 x %d seed evaluations): %s" % (res["n_reps"], json.dumps(C["G78|A|CAL"]["A1"]["r1_seed_level_by_conjunct"])))
out.append("")
out.append("### T2. A2-r1 false refusal per head and seed (G78, calibrated)\n")
out.append("| head | seed 0 | seed 1 | any seed |")
out.append("|---|---|---|---|")
a2 = C["G78|A|CAL"]["A2"]
for h in a2["heads"]:
    out.append(f"| {h} | {pc(a2['r1_per_head_per_seed'][h]['seed0'])} | {pc(a2['r1_per_head_per_seed'][h]['seed1'])} | {pc(a2['r1_per_head_any_seed'][h])} |")
out.append("")
out.append("### T3. POWER of A1-r1: refusal rate of a PLANTED over-prediction `delta` in D_M units (any-seed; Wilson 95 %)\n")
out.append("| geometry | delta | A1-r1 refuses (2 seeds) | seed 0 | seed 1 | A1 DRAFT refuses | lower-CI conjunct (any seed) | point conjunct (any seed) | mean D_M point | net sigma scale at changed picks (a_e x f, seed 0) |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for g in ("G78", "G25", "G05"):
    for name in sorted([n for n in C if n.startswith(f"{g}|A|PLANT_EP_")], key=lambda n: float(n.rsplit("_", 1)[1])):
        d = name.rsplit("_", 1)[1]
        c = C[name]["A1"]
        pm = pre["meta"][f"{g}|s0"]
        ns = pm["scalars"]["a_e"] * pm["plant"][d]["f"] if d in pm["plant"] else float("nan")
        out.append(f"| {g} | +{d} | **{pc(c['r1_any_seed'])}** | {pc(c['r1_seed0'])} | {pc(c['r1_seed1'])} | {pc(c['draft_any_seed'])} | {pc(c['r1_any_seed_by_conjunct']['lower_ci_gt_0_any_seed'])} | {pc(c['r1_any_seed_by_conjunct']['point_gt_0.05_any_seed'])} | {c['D_M_point_mean'][0]:+.4f} / {c['D_M_point_mean'][1]:+.4f} | {ns:.2f} |")
for name in ("G78|B|PLANT_EP_0.05", "G78|B|PLANT_EP_0.135"):
    c = C[name]["A1"]
    out.append(f"| G78 gen B | +{name.rsplit('_', 1)[1]} | **{pc(c['r1_any_seed'])}** | {pc(c['r1_seed0'])} | {pc(c['r1_seed1'])} | {pc(c['draft_any_seed'])} | {pc(c['r1_any_seed_by_conjunct']['lower_ci_gt_0_any_seed'])} | {pc(c['r1_any_seed_by_conjunct']['point_gt_0.05_any_seed'])} | {c['D_M_point_mean'][0]:+.4f} / {c['D_M_point_mean'][1]:+.4f} | n/a |")
out.append("")
out.append("### T4. D_M interval half-width and spread on the calibrated replicates\n")
out.append("| condition | mean half-width seed 0 / 1 | SD of the D_M point seed 0 / 1 | seed-to-seed correlation of the D_M point |")
out.append("|---|---|---|---|")
for name in ("G78|A|CAL", "G25|A|CAL", "G05|A|CAL", "G78|B|CAL"):
    a = C[name]["A1"]
    out.append(f"| {name} | {a['D_M_half_width_mean'][0]:.4f} / {a['D_M_half_width_mean'][1]:.4f} | {a['D_M_point_sd'][0]:.4f} / {a['D_M_point_sd'][1]:.4f} | {a['seed_corr_of_D_M_point']:.3f} |")
out.append("")
out.append("### T5. POWER of A2-r1: refusal of rows that UNDER-predict risk by rho at the picks (G78; any seed; Wilson 95 %)\n")
out.append("| head planted | rho | A2-r1 refuses | A2 DRAFT refuses |")
out.append("|---|---|---|---|")
for h in ("NC", "DAC", "TTC"):
    for rho in (1.25, 1.5):
        c = C[f"G78|A|PLANT_A2_{h}_{rho}"]["A2"]
        out.append(f"| {h} | {rho} | **{pc(c['r1_any_head_any_seed'])}** | {pc(c['draft_any_head_any_seed'])} |")
out.append("")
out.append("### T6. Controls\n")
ctl = res["controls"]
out.append(f"* PC1 (calibrated by construction): " + "; ".join(f"{n}: {'PASS' if v['pass'] else 'FAIL'} (M_p {v['M_p'][0]:.4f}/{v['M_p'][1]:.4f}, M_e {v['M_e'][0]:.4f}/{v['M_e'][1]:.4f}, mean D_M {v['D_M_point_mean'][0]:+.5f}/{v['D_M_point_mean'][1]:+.5f}" + (f", A2 ratio means {json.dumps({h: [round(x, 3) for x in r] for h, r in v['A2_ratio_point_mean'].items()})}" if v.get("A2_ratio_point_mean") else "") + ")" for n, v in ctl["PC1_calibrated_by_construction"].items()))
t = ctl["PC2_check_has_teeth"]
out.append(f"* PC2 (the check has teeth): {'PASS' if t['pass'] else 'FAIL'}; uncalibrated real replicates fail the calibrated check as required: {t['uncalibrated_real_replicates_FAIL_the_calibrated_check']['pass']} (mean D_M {t['uncalibrated_real_replicates_FAIL_the_calibrated_check']['D_M_point_mean']}); plant +0.135 passes the planted form: {t['plant_0.135_PASSes_the_planted_form']['pass']}; plant +0.05: {t['plant_0.05_PASSes_the_planted_form']['pass']}; generator B plant +0.135: {t['generator_B_plant_0.135_PASSes_the_planted_form']['pass']}")
p3 = ctl["PC3_shortcut_is_the_estimator"]
out.append(f"* PC3 (shortcut = estimator): {'PASS' if p3['pass'] else 'FAIL'}; identity draw equals the library pipeline for all 6 geometry-seeds; 18 materialised set-level resamples equal the shortcut to 1e-12; `analyze_candidate` equal; landed EP-RELIAB D_M reproduced to {max(v['max_abs_diff'] for v in p3['landed'].values())}")
out.append(f"* PC4 (decision code is the module's): {'PASS' if ctl['PC4_decision_code_is_the_modules']['pass'] else 'FAIL'}; total inline-vs-module disagreements {sum(ctl['PC4_decision_code_is_the_modules']['disagreements_by_condition'].values())}; ep_admission.py sha256 {ctl['PC4_decision_code_is_the_modules']['ep_admission_sha256']}")
out.append(f"* PC6 (identity candidate): {'PASS' if ctl['PC6_identity_candidate']['pass'] else 'FAIL'}")
if len(sys.argv) > 3:
    rp = json.load(open(sys.argv[3], encoding="utf-8"))
    out.append(f"* PC5 (determinism, separate process): {'PASS' if rp['bit_identical_first_20_replicates_of_3_conditions'] else 'FAIL'}")
out.append("")
out.append("### T7. Flags committed in advance (SIM_DESIGN.md section 8)\n")
for k, v in res["flags"].items():
    out.append(f"* {k}: **{v}**")
print("\n".join(out))
