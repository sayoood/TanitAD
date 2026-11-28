# HJ-UNITS-2 -- FROZEN callers that now fail (NOT edited; they are the record of what ran)

Scanned at mirror tip 5fb509a (`code/frozen_scan2.py`, raw `raw/frozen_callers_scan2.json`). A frozen script that is re-run against the live tree fails LOUDLY.

## A. CLI: rollout invocations of `taniteval/tools/refav1_arm.py` without `--action-units` -- 37 invocations in 37 files

They now exit 2 (`--action-units is REQUIRED ...`, naming `kappa` and `steer`). To reproduce a banked run pass `--action-units kappa` (the old default, which every pre-2026-09-03 dump used). Lower bound: a command assembled from a variable the scan cannot resolve is not counted.

- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-ccos-eval/tools/devbox_chain.sh:28`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-ccos-eval/tools/thor_chain_ccos.sh:40`  (UNVERIFIED: the file mentions action-units elsewhere, the flag may ride in a variable)
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-ccos-eval/tools/thor_launch_arm.sh:17`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-close-the-gaps/raw/queueCTG.sh:42`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-close-the-gaps/raw/queueCTG2.sh:68`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/L1_wk_cos.sh:20`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueB2.sh:11`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueD.sh:40`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueDv2.sh:40`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueE.sh:40`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueEv2.sh:40`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueF.sh:41`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueFv3.sh:41`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueG.sh:27`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueH.sh:36`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueJ.sh:51`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/queueK.sh:48`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-cost-geometry/raw/wk_lib.sh:26`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-goal-margin/tools/run_p4.sh:46`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-goal-margin/tools/run_p4c.sh:64`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-goal-margin/tools/run_p4wk.sh:55`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-longitudinal/raw/queueLON.sh:62`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-longitudinal/raw/queueLON2.sh:62`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-longitudinal/raw/queueLON3.sh:62`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-longitudinal/raw/queueLON4.sh:62`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-longitudinal/raw/queueLON5.sh:62`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-lonshift-t1/raw/queueGRS.sh:39`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-lonshift-t1/raw/queueTHOR.sh:34`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-make-it-drive/tools/run_ab.sh:46`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-make-it-drive/tools/run_ab_targeted.sh:55`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-make-it-drive/tools/run_replicate.sh:37`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-vocab-l3/tools/run_oracle.sh:49`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refav1-decoded-gap/code/run_bar3_ab.sh:40`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refav1-distance-keeping/RESULT.md:288`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refav1-distance-keeping/code/run_ab_smoke.sh:23`
- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refav1-fullgrid-loncomb3/raw/run/run_fullgrid.sh:57`
- `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/RESULT.md:379`

## B. API: calls that omit the unit -- 7 call sites in 5 files (all raise TypeError against the live module; 0 define their own copy)

- `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-tacdec-L0-L1-E0/tools/seed_goal_agreement_report.py:134`  plan(...)
- `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/refav1_r1r6/code/fix/stack/tanitad/refs/refav1_traj.py:98`  unicycle_paths(...)
- `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/refav1_r1r6/code/fix/stack/tests/test_refav1_r1_vmax_input.py:130`  plan(...)
- `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/refav1_r1r6/code/fix/stack/tests/test_refav1_r1_vmax_input.py:134`  plan(...)
- `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/refav1_r1r6/code/fix/stack/tests/test_refav1_r1_vmax_input.py:137`  plan(...)
- `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/refav1_r1r6/code/fix/stack/tests/test_refav1_r6_strategic_off.py:151`  plan(...)
- `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/refav1_r1r6/raw/bitident/bitident_dump.py:140`  plan(...)
