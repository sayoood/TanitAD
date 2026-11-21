# PREREAD_V8F_STOPCUE: is the stop cue in T's features (mis-thresholded) or absent? (Master Mind, 2026-11-10, written before the AUROC exists)

Status: FROZEN at the sha256 in `PREREAD_V8F_STOPCUE.sha256` before `stopcue_read.py` runs. Zero training, zero model passes: it reads the per-window files of H-V8F-STOPDEC-1 (`theory/stopdec/out/stopdec_PARITY_DAGOACC_s{0,1}.npz`, their sha256 recorded by the script). Tier: T1-kin first tick, eval-141, a diagnostic of T's decision head, never driving performance. STRATEGIC: PAUSED (PI item 25).

## Why
H-V8F-STOPDEC-1 read **DECISION-MISSES** at both seeds (MEASURED): in GT STOP_POINT windows T's goal head says STOP_POINT (p >= 0.5) in 0.103 / 0.128 of them, and 0.940 / 0.938 of the group's speed-error mass sits in the misses; where T's decision does see a slowdown its speed cells follow it. The committed trigger is a stop-cue perception card. Its first question: does T's head RANK stop windows above the rest (the cue is in its features, the 0.5 cut is wrong for a ~5 % class), or not (the cue is absent: perception)?
**Already seen before this card (disclosed):** the STOPDEC table, including the mean p(STOP_POINT) in STOP_POINT windows 0.154 / 0.185 and the false-positive rate at 0.5 of 0.006 / 0.009. No ranking statistic has been computed.

## Measure
* Positives: windows with `has` whose GT goal includes STOP_POINT (`stop_hit`, with or without a traffic-light goal). Negatives: windows with `has` and no GT STOP_POINT goal.
* Score: T's own `goal_prob[:, 6]` (the per-goal sigmoid, `v8f_mini.py:968`).
* **PRIMARY: AUROC** (Mann-Whitney, ties counted one half), per seed.
* Beside, no bar: AUROC of `lon_prob[:, STOP]` for GT regime STOP (`v_end_gt < 0.5`) against the rest; AUROC of `lon_prob[HOLD] + lon_prob[STOP] + lon_prob[DECELERATE]` for GT regime STOP or DECEL (`v_end_gt < 0.5` or `v_end_gt - v2_gt < -1`) against the rest; average precision of the primary score; recall at the score threshold whose false-positive rate on the negatives equals 0.05 (an eval-chosen threshold: a diagnostic, never a deployable setting).
* Estimator: `taniteval.ci.stat_cluster_bootstrap`, B 2,000, seed 0, by episode (`eid`); "another draw of episodes" only. Both seeds are the training-run look.

## Committed outcomes (PRIMARY, both seeds must agree, else SEEDS-DISAGREE)
* **CUE-PRESENT:** AUROC >= 0.85 with its interval lower bound >= 0.80. T's features rank stops well; the defect is the decision's operating point for a rare class. Trigger: a training card that rebalances the STOP_POINT goal (positive weight in its BCE, and the lon STOP class weight), fitted and checked on training windows only, read on the same T1-kin first-tick eval with this read's AUROC as a non-regression control.
* **CUE-ABSENT:** AUROC < 0.70. T's features do not carry a usable stop cue. Trigger: a perception card (input resolution of the stop region / the LoRA arm L's read adds this AUROC / a stop-line auxiliary label), ranked by cost.
* **PARTIAL:** otherwise. Both cards, the rebalancing first (cheaper).

## Controls (INVALID if one fails)
1. The two npz files' sha256 are recorded; their `has` counts equal STOPDEC's (2,365) and the positive count is >= 50 per seed.
2. `goal_prob` lies in [0, 1]; the AUROC routine returns exactly 1.0 on a planted perfect score (score = label) and exactly 0.5 on a constant score, before it touches the data.

## Log
* 2026-11-10 written and frozen by the Master Mind before the AUROC was computed.
