#!/usr/bin/env python3
"""Write N1_PINS.json (PREREG_TF_N1 section 7 C-PIN) from a run dir made by n1_prep.sh: the sha256 of every pinned file,
keyed by its path relative to the run dir (absolute for the vendored NAVSIM devkit, outside the run dir).
  python n1_make_pins.py <run-dir> <out N1_PINS.json>"""
import hashlib, json, os, sys
DEVKIT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
REL = ["w/pk/refe/navsim_dac.py", "w/pk/refe/navsim_pdm_targets.py", "w/pk/refe/navtrain_scenarios.py",
       "w/pk/eval/p5_target_labels.py", "w/pk/eval/p4_tracker_comp.py", "w/pk/eval/proxy_eval.py",
       "w/pk/eval/instruments/transp_score.py", "w/pk/eval/instruments/families_run.py", "w/tf2_score.py",
       "repo/taniteval/adapters/navsim.py", "repo/taniteval/taniteval/four_families.py",
       "n1/tf_n1_build.py", "n1/tf_n1_analyze.py", "n1/tf_n1_chain.py",
       *[f"inputs/rollouts/rows_s{i}of4.jsonl.xz" for i in range(4)], "inputs/refe_p4_teacher_navtest.npz",
       "inputs/refe_p4_teacher_navtest.csv", "inputs/refe_p4_human_navtest.csv", "inputs/frame_control_navtest_full.json",
       "inputs/p5s_cubic.npz"]
ABS = [f"{DEVKIT}/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
       f"{DEVKIT}/navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
       f"{DEVKIT}/navsim/evaluate/pdm_score.py", f"{DEVKIT}/navsim/planning/metric_caching/metric_cache_processor.py"]
run, out = sys.argv[1], sys.argv[2]
files = {}
for k in REL + ABS:
    p = os.path.join(run, k)
    files[k] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"card": "PREREG_TF_N1.md section 7 C-PIN", "made_from": os.path.abspath(run), "files": files}, open(out, "w", encoding="utf-8"), indent=1)
print("ZZN1_PINS", len(files), "files ->", out)
