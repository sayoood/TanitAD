# PREREG_V8F_PROOF_ADD11: HIERREP, replicate arms of the matched hierarchy / flat pair (pod, training)

Frozen before any arm of this card is trained (file time and sha256 in `TIMES_NOTE.md`). Written 2026-11-13 after reading H-V8F-PROOF-7 (HIERWHERE), H-V8F-PROOF-9 (GATEMIX) and H-V8F-PROOF-12 (SOUP); no pod training output exists at freeze time.

## Question
Does the hierarchical v8F-mini (arm M2b) beat its flat control (M2b with `--o-interface cut`, same parameter count) on selection-critical windows, and on all windows, when the comparison is repeated over training seeds? Earlier reads had one run per arm per seed (two seeds): H-V8F-FLAT-1 and H-V8F-FLATDAG-1 read NOT ESTABLISHED; HIERWHERE found the difference concentrated on selection-critical windows in all four cells. An episode interval cannot answer the run-to-run question (H-ESTIM-SEED-1). This card answers it with seeds as the unit.

## Arms (one code tree, one pod-built token cache, pod A40)
* Stage 1, from scratch, 19,164 steps, batch 64, argv of `raw/2026-10-26-flat/runs/M2b_s0/argv.json` with pod paths: HIER_s = `--arm M2b --seed s`; FLAT_s = the same with `--o-interface cut`. Seeds s = 0, 1, 2, 3, 4, 5. Twelve arms.
* Stage 2, own-state fine-tune, 3,000 steps, argv of `raw/2026-10-27-dagger-run/M2b_s0_dag/argv.json` with pod paths (`--init-ckpt` = the Stage 1 checkpoint of the same arm and seed, `--lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0 --dagger-p 0.1`, bank built from the frozen parent on TRAIN clips only, `--dagger-seed s`): DAG_s and FLATDAG_s. Twelve arms. The no-swap fine-tunes (`--dagger-p 0`) are not trained: they are a control of the swap, not of the hierarchy.
* Token cache: rebuilt on the pod from the HF frames with the programme's pod route (`cache_pod.py`); it differs from box 2's cache by about 2.4e-2 relative L2, so these arms are a new, internally matched set and not a reproduction of box 2's checkpoints. Excluded clips `540:600` as in the source argv.
* Gate: every argv passes the trainer's own `--selfcheck-only` run (config against build, live loss terms, frozen-prior control exactly 0.0) before its training line starts, and the training argv equals the gated argv by hash. `launch_gate.py` has no profile for this trainer; this is the gate the programme has used for every v8F-mini arm.

## Read
* Tier T1-kin strict: `v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal`, eval-141, 2,399 first-tick windows. A diagnostic on logged frames, never driving performance.
* Window classes from the LOGGED joint mode with the thresholds of addendum 6 (STOP 0.5 m/s, +/- 2.0 m/s, +/- 10 deg): 1,305 selection-critical, 1,094 bulk. They do not depend on any arm.
* Per seed s and per protocol: d_s = mean over the class of ADE6(hierarchical arm) - ADE6(flat arm), paired per window. D = mean over the six seeds.
* Intervals: (A) two-level bootstrap, seeds then episodes, B 2,000; (B) exact sign-flip test over the six d_s (64 assignments); (C) the t interval over the six d_s, reported, deciding nothing.
* Noise floor, reported beside every difference: for each arm type the 15 seed-pair differences of the SAME arm on the same windows (median absolute value, and how many of the 15 the paired episode bootstrap calls separated).
* Primary: Stage 2 (DAG_s against FLATDAG_s), selection-critical windows, ADE6. Secondary, each with the same statistics: Stage 2 on all windows and on bulk; Stage 1 on the three classes; tactical-band ADE.
* Four families on all windows for both protocols (longitudinal incl. headway and time gap, lateral incl. heading, curvature and yaw rate, tactical decisions and goal), seed-level statistics as above. STRATEGIC: PAUSED (PI item 25).

## Bars and outcomes (written before the data)
A class reads HIER when all six d_s are negative (exact two-sided p = 0.031) and interval (A) excludes 0; FLAT when all six are positive and (A) excludes 0; OPEN otherwise.
* **HIER-DOMINATES**: selection-critical HIER and all-windows HIER (primary protocol).
* **HIER-ON-CRITICAL**: selection-critical HIER, all-windows not HIER. The bulk difference is the named cost.
* **NOT-ESTABLISHED**: selection-critical OPEN.
* **FLAT-AHEAD**: selection-critical FLAT.
The Stage 1 protocol gets the same four labels as a secondary result. No bar moves after the data. A missed bar is reported as written.

## Controls that must read known values
* r1 harness: the pod dump of an uploaded checkpoint (`stopbal_BAL-NULL_s0`) reproduces the dev-box dump of the same checkpoint within 0.005 m on ADE6 over all windows. Else STOP before any arm is read.
* r2 gate: 24 selfcheck passes, each with the frozen-prior control at exactly 0.0; a selfcheck with a deliberately wrong lever value (the hierarchical argv checked against a model built with `cut`) must FAIL.
* r3 parameter count identical in the two arm types (7,214,495 trainable in the source runs).
* r4 label swap: with the arm labels swapped within seed at random (20 draws), the primary must read OPEN in at least 18 of 20.
* r5 completeness: an arm that did not reach its last step is not read; a seed enters only with both arm types of both stages.

## What it cannot show
Six seeds, one corpus (4,512 training clips), one evaluation set of 141 episodes, a kinematic rollout on logged frames. It says nothing about another corpus, a re-perception simulator, or a larger model.

## Committed next levers (run in the same chain or the next one, no stop at a miss)
* HIER-DOMINATES: the data-efficiency pair (both arm types at 1/4 and 1/2 of the training clips, three seeds) and the finer-interface pair if FULLRES (addendum 9) reads INTERFACE-LIMITED.
* HIER-ON-CRITICAL: a critical-window flag head trained on the logged joint mode (target AUROC 0.9 from H-V8F-PROOF-10), then the commit rule on these rollouts.
* NOT-ESTABLISHED: the same six seeds with training weighted toward selection-critical windows (theorem T8), both arm types.
* FLAT-AHEAD: the interface ablation (which term of the operative interface costs), three seeds.
* In every case: within-arm soups are not possible across different inits; the third same-init replicate for the three-member soup (H-V8F-PROOF-12) is trained after the last arm of this card.
