#!/usr/bin/env python3
"""Render RESULT_A2_ACTIVATION.md from a2_real_results.json (+ job_identity_result.json).  usage: make_report.py <a2_real_results.json> <job_identity_result.json> <out.md>"""
import json
import sys

R = json.load(open(sys.argv[1]))
J = json.load(open(sys.argv[2]))
n = R["n_scenes"]
agg = R["aggregate"]
cc = R["comfort_detector_comparison"]
ch = R["comfort_channel_violation_counts"]
order = R["comfort_channel_order"]
L = []
a = L.append
a("# A2 behaviour labels: real-data measurement (MEASURED, dev box, CPU)\n")
a(f"Scenes: **{n}** nuPlan **trainval** states, seeded (seed 20261109), navtest-disjoint by BOTH gate keys. navtrain identity UNVERIFIED (the navtrain scene-filter file is not on this box). "
  "Fan: 64 seeded proposals per scene = 8 speed profiles x 8 lateral profiles around the logged future (NOT REFe's fan). Rates below are therefore term-SENSITIVITY checks and false-positive floors, not training-label rates. "
  "Constants: `behaviour_terms.C`, version " + R["version"] + ".\n")
a("## Identity (the brief's bit-identical requirement)\n")
a(f"* `score_plans`: pre-A2 tip vs overlay OFF vs overlay ON (PDM keys), byte-equal on every scene: **{R['identity_all_pass']}** (includes a deliberate-regression arm that must break equality, and a repeat run).")
a(f"* through the real worker path `online_labeller._job`, separate processes, tip vs OFF vs ON labels, n={J['n_scenes']} scenes: tip==OFF **{J['tip_vs_overlay_off_bytes_equal']}**, OFF==ON labels **{J['overlay_off_vs_on_labels_bytes_equal']}**, "
  f"ON repeated in a 2nd process **{J['on_vs_on_two_processes_bytes_equal']}**, OFF returns no behaviour arrays **{J['off_has_no_behaviour_arrays']}**.")
a(f"* `q_comf == 1  <=>  NAVSIM c == 1` for every proposal on every scene: **{R['c_identity_all']}**.\n")
a("## Per-term activation (over all proposals; `applicable` = term evaluated; `fired` = q < 1)\n")
a("| term | applicable (prop. rate) | fired (prop. rate) | scenes where q varies across the 64 | PDM-Closed row: applicable / fired | logged-human row: applicable / fired |")
a("|---|---|---|---|---|---|")
for t, v in agg.items():
    a(f"| {t} | {v['applicable_prop_rate']:.3f} | {v['fired_prop_rate']:.3f} | {v['scenes_discriminating']}/{v['n_scenes']} | {v['ref_row_applicable']}/{n} / {v['ref_row_fired']}/{n} | {v['gt_row_applicable']}/{n} / {v['gt_row_fired']}/{n} |")
a("\nThe last two columns are the FALSE-POSITIVE FLOOR: a term that fires on PDM-Closed or on the logged human drive is measuring something the reference does too. "
  "`discriminating` is the D-RL-COLL-SPARSE-1 check: a term that is constant across the 64 proposals of a scene gives the listwise loss no gradient in that scene.\n")
a("## Comfort: the pilot's problem, checked\n")
a(f"Same simulated states, two detectors (`pilot_style_comfort` = raw 10 Hz differences as in the pilot's gym_env.py:331-348, reimplemented; `comfort_violations` = NAVSIM's own filtered channels). Proposals with >= 1 violated channel:\n")
a("| set | NAVSIM-filtered | pilot-style raw |\n|---|---|---|")
a(f"| all proposals ({cc['n_all']}) | {cc['navsim_all']} | {cc['pilot_all']} |")
a(f"| PDM-Closed row ({cc['n_ref']} scenes) | {cc['navsim_ref']} | {cc['pilot_ref']} |")
a(f"| logged-human row ({cc['n_gt']} scenes) | {cc['navsim_gt']} | {cc['pilot_gt']} |")
a("\nViolations per channel, summed over proposals (" + ", ".join(order) + "):  NAVSIM " + str(ch["navsim"]) + "   pilot-style " + str(ch["pilot"]) + "\n")
a("## Cost\n")
import statistics as st
bs = [c["base_s"] for c in R["cost"][2:]]
ex = [c["extra_s"] for c in R["cost"][2:]]      # first two scenes carry one-off warm-up
cm = dict(base_s=st.median(bs), extra_s=st.median(ex))
a(f"MEDIAN per scene (first two scenes dropped as warm-up): PDM scoring {cm['base_s']:.2f} s (one process), behaviour terms +{cm['extra_s']:.2f} s ({100 * cm['extra_s'] / cm['base_s']:.0f} % on top). "
  "The labeller budget (M-LAB) was 0.34-0.83 CPU-s per set on the pod: the extension adds the same order again, so `--label-workers` must be re-sized with `behaviour` ON (INHERITED figure, re-measure on the pod).\n")
a("## Per scene\n")
a("| # | loc | v0 | free steps | lead steps | q_free app/fired | q_gap app/fired | q_osc fired | q_stall fired | q_comf fired |\n|---|---|---|---|---|---|---|---|---|---|")
for e in R["per_scene"]:
    a(f"| {e['scene']} | {e['loc'][:8]} | {e['v0']:.1f} | {e['n_free_steps_mean']:.1f} | {e['n_lead_steps_mean']:.1f} | {e['q_free']['applicable']:.2f}/{e['q_free']['fired']:.2f} | "
      f"{e['q_gap']['applicable']:.2f}/{e['q_gap']['fired']:.2f} | {e['q_osc']['fired']:.2f} | {e['q_stall']['fired']:.2f} | {e['q_comf']['fired']:.2f} |")
open(sys.argv[3], "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", sys.argv[3])
