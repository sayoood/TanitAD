# DSCALE10K DATA-HURTS diagnostics: frozen reading rules (written before any diagnostic number was computed)

Date 2026-11-18. Card: `card/CARD_DATASCALE_TRAIN.md` section 6, DATA-HURTS row ("per-goal-class F1 (VLM vs geometry bits)", "eval-141 per country and platform"). Evidence class MEASURED (ours; banked T1 tick dumps, zero training, CPU).
Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, with the SAME episode resampler as `taniteval.ci` (`ci.episode_index`, `ci._draws` from the read tree `F:/TanitAD-scratch/otband_read/tree`, the tree the block reads use). It answers "another draw of EPISODES" only. Training-run variance is not in it; the replicate-pair floor is printed beside and decides nothing here.
Tier: T1-kin tick 0 of each window. At tick 0 the executed pose equals the logged pose and `age_t` is 0 (MEASURED on the seed-1 stage-B ticks file), so T's head output is the head read at the window's own frame. The goal read is a tactical-head diagnostic, never driving performance.

## Inputs
* Heads: `read/t1/ticks/ticks_DS-<arm>-<stage>_m10_causal_heads.npz` (seed 0) and `read2/t1/ticks/...` (seed 1), field `t_goal_prob` [N, 22] (sigmoid, `GOAL22` order), rows with `tick == 0`; 2,399 windows, 141 episodes.
* Goal labels: `strat1/read/inputs/join_eval141.npz`, `goal_y` and `goal_w` (bitmasks, bit i = `GOAL22[i]`), at join row `off[ci] + t0`. Clip order of the join must equal the ticks file's `clip_names` (asserted).
* Trajectory tactical cells TAC-LAT / TAC-LON: recomputed with the block reads' own loader (`ds_read.load_arm_cells`, `obd_geom`), per window, stage B.
* Country: `strat_labels_table.npz` eval labels (`eval_country_id`, valid and `eval_flag == 0`), clip-level = the modal country over the clip's eval frames. Layer: the STRAT labels table of the STRAT1 read (a per-frame label release layer, not a published PhysicalAI-AV feature). Platform: `strat_platform_by_clip.npz` (`eval_platform_id`, vocab hyperion_8 / hyperion_8.1).

## F1 read (rules)
1. Prediction for class c = `t_goal_prob[c] > 0.5` (fixed, not tuned). A window is scored for class c only when bit c of `goal_w` is set at its join row (the label exists there, the training-mask semantic). Truth = bit c of `goal_y`.
2. Per class: TP, FP, FN over the scored windows; F1 = 2TP / (2TP + FP + FN), defined 0 when TP = 0 and (FP + FN) > 0, and left undefined (class out of the macro) when TP + FP + FN = 0. A class with zero truth-positives among the scored windows (n_pos = 0) is listed with n and left out of the macro. The set of classes in each macro is fixed by the B_s0 arm's support (n_pos > 0), identical for every arm and seed (support is label-side, so arm-free).
3. Source split, from the release's own `GOAL_PROV` (`build_v9_labels.py`): geometry-sourced = FOLLOW_LANE, TURN_L, TURN_R, YIELD_FOR_TURN_L, YIELD_FOR_TURN_R, STOP_POINT, SPEED_BAND. VLM-sourced = every other class (the card's masked set), including LANE_CHANGE_L / LANE_CHANGE_R (`sam3|vlm`).
4. Group macro-F1 = the unweighted mean of per-class F1 over the group's classes in the macro. Difference D = macro(U) - macro(B), at stage B (deciding), seed 0 (U_s0 - B_s0) and seed 1 (U_s1 - B_s1). CI by the paired episode-cluster bootstrap over the same windows. "Negative and separated" at a seed = D < 0 and the CI excludes 0 (hi < 0).
5. Group "loses" = negative and separated at BOTH seeds.
6. Labels (pre-committed):
   * DILUTION-CONSISTENT: VLM macro D is negative and separated at both seeds, AND the geometry macro D is not negative-and-separated at either seed.
   * NOT-DILUTION: both groups lose, or neither group loses.
   * MIXED: anything else.
   Per-class rows, the floor pair (B_rep seed 2 against B_s0, stage B) and stage A are printed beside and decide nothing.
7. Controls that must read known values: an oracle arm (prediction = truth) reads F1 1.0 on every class with support; U against itself reads D exactly 0.0 with CI [0, 0]; the recomputed TAC-LAT / TAC-LON stage-B deltas equal the banked `RESULT_DSCALE10K_B2.json` pairs (K-RE).

## Country / platform read (rules)
1. Cells TAC-LAT and TAC-LON per window (1 if the executed trajectory's decision class equals the label class), the same per-window arrays as the deciding read, stage B.
2. For each stratum (each country, each platform class), D = U - B accuracy at seed 0 and at seed 1, paired episode-cluster bootstrap within the stratum's episodes, with n windows and n episodes.
3. Descriptive only: no label decides from it. A stratum is FLAGGED only if D > 0 and separated (the direction opposite to the overall cost) at both seeds on the same cell. A stratum with fewer than 5 episodes is listed with its n and cannot be flagged. Strata with D < 0 separated at both seeds are listed as "same direction" (descriptive, not a flag).
4. The number of strata read is stated beside the flags (multiplicity: 25 countries x 2 cells x 2 seeds), and no stratum result is promoted without its own pre-registration.
