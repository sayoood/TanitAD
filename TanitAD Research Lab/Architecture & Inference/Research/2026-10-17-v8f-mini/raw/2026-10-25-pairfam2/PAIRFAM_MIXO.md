# PAIRFAM_MIXO - paired X - B intervals for the TACTICAL and STRATEGIC families, MIXO seed 0

**Status: descriptive. No outcome table exists for these cells, so nothing here decides a bar or a next step.** It closes the gap `RESULT_V8F_MIXO.md` section 5.3 / 5.4 lists as a work item (tactical and strategic cells had only each arm's own unpaired interval), using the new tool `stack/scripts/v8f_paired_families.py`.

**Tier:** T1-class open loop (own held outputs; no recorded future action).  **Estimator:** paired episode-cluster bootstrap, B=2000, 141 episodes. X - B on the same resampled episodes in every draw; every non-additive statistic (kappa, macro-F1, the goal head's 22-token macro-F1) is re-computed on each resample.

**Which variance this answers.** "Separated" means the 95 % interval excludes 0, and it answers *another draw of episodes* only. X (`M2b-mixo_s0`) and B (`M2b_s0`) are each ONE training seed, so a separated cell would be necessary and not sufficient for a lever effect (`H-ESTIM-SEED-1`). STRATEGIC is active for v8F-mini (PI-V8F-2026-10-17; R72/R73) and is reported in full below.

## 1. Result in one paragraph

**No tactical or strategic cell is separated.** 37 cells carry a paired interval (31 tactical, 6 strategic); 0 of them exclude 0. A further 8 goal-token F1 cells carry a point difference only (the token's F1 is undefined on some resamples) and 5 are UNAVAILABLE with their reason (section 5). The 16 point differences that `RESULT_V8F_MIXO.md` section 5.3 / 5.4 print equal the paired read's (asserted by the generator of this file, to 1.5e-4); what is new is the interval. With 37 intervals at 95 %, about 1.9 would be expected to exclude 0 by chance alone; none did. That is not equivalence: the intervals are wide (for example the lateral decision kappa difference is +0.0033 [-0.0396, +0.0392], and the 80 m checkpoint position error difference is +0.1623 [-0.2843, +0.6321] m). The data do not rule out a tactical or strategic difference of that size in either direction; they show none.

## 2. Inputs and controls

* **X** = `eval_M2b-mixo_s0.npz` (sha256 `081cb82c66622296fd6813f1f09b264c16091c85eaf887eb01e9e6b8653def03`), **B** = `eval_M2b_s0.npz` (sha256 `8ddcd42c119f7ca9cc6a434103f40d017d61f809161c517310f2fb778df20501`), from `C:/Users/Admin/box2_ship/logs/v8f_mixo/eval/` (read only). B's own-code ADE6 is 2.1753 m (the required 2.1753); X's is 2.1977 m (`RESULT_V8F_MIXO.md`: +0.022 m).
* Grid and labels: `eval_grid_windows.json, eval141_kin.npz, join_eval141.npz` from `raw/` of this package (sha256 in the json, `inputs_sha256`), the eval's own `eval_context` (2,399 windows, 141 episodes, kin mode `yaw`, ages `cycle`).

| control | condition | result |
|---|---|---|
| same windows | identical `eid` sequence, and equal to the grid's | **True** (2,399 windows) |
| window order inside episodes (the thing `eid` cannot see) | each dump's stored per-window `ade6` equals ADE6 recomputed from its `pred` and the grid GT | max deviation **0.0 m** (X), **0.0 m** (B) |
| checkpoint-label mask | each dump's `ck_pos_err` NaN mask equals the grid's `rc_ok` | X **True**, B **True** |
| identity | the whole pipeline on X against X reads exactly 0.0 [0.0, 0.0], not separated, on every cell | **True**: 45 cells checked, 0 not zero |
| reproduction, X | each point value, and each per-arm interval where the eval JSON has one, equals `eval_M2b-mixo_s0.json` to 1e-6 | **True**: 119 checks, 0 failed, max abs difference 0 |
| reproduction, B | the same against `eval_M2b_s0.json` | **True**: 119 checks, 0 failed, max abs difference 5.55e-17 |
| heading rebuild | the held-age rebuild of S's NOW-frame checkpoints reproduces the stored `ck_pos_err` on every window (the gate for the heading cells; the dump does not store the heading error) | max deviation X **0.0 m**, B **0.0 m** |
| refusal on misaligned windows | different episode ids, different length, another grid, a within-episode reordering: rc 2, nothing written | pinned in `stack/tests/test_v8f_paired_families.py` (`test_misaligned_windows_are_refused`, `test_the_cli_refuses_rc2_on_misaligned_dumps_and_writes_nothing`) |
| mutation proof | every mutant turns the suite RED; the unmutated tool is GREEN before and after, bytes identical | **31 of 31** mutants RED (31 on an assertion); before: GREEN (24 passed in 31.92s); after: GREEN (24 passed in 22.55s), bytes identical True (`mutate_pairfam.out`) |

Reproduction compares at the eval JSON's own precision (its intervals are rounded to 4 dp); the per-arm intervals match because the tool uses the eval's seed 0 and B = 2,000.

## 3. TACTICAL

Two reads, kept apart. **Plan-derived**: the manoeuvre read off the arm's PLAN by the programme's own labeller (`maneuver_kinematics` -> `factor_from_kinematics`) against the one read off the human path; both are EXECUTED manoeuvres (the eval's own caveat: this is not "selected vs executed"). **Heads**: the lat / lon decision heads and the goal-token head against the v9 labels. Accuracy and kappa of the heads, the goal-token accuracy and the per-token F1 are cells the eval JSON does not carry; they use the same definitions as the plan-derived rows (accuracy, Cohen's kappa) and the eval's own goal threshold (`goal_prob > 0.5`). The goal-token accuracy is a micro accuracy over the labelled (window, token) cells and is dominated by the all-negative majority; read it beside the macro-F1.

### TACTICAL, plan-derived decisions and goal point

| cell | X | B | X - B [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.plan_derived.lateral_decision.accuracy` | 0.8441 | 0.8458 | -0.0017 [-0.0225, +0.0167] | 2,399 / 141 | no |
| `tactical.plan_derived.lateral_decision.kappa` | 0.6685 | 0.6652 | +0.0033 [-0.0396, +0.0392] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.accuracy` | 0.5661 | 0.5790 | -0.0129 [-0.0358, +0.0079] | 2,399 / 141 | no |
| `tactical.plan_derived.longitudinal_decision.kappa` | 0.3398 | 0.3566 | -0.0168 [-0.0494, +0.0135] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.accuracy` | 0.5740 | 0.5811 | -0.0071 [-0.0346, +0.0179] | 2,399 / 141 | no |
| `tactical.plan_derived.maneuver_5way_collapsed.kappa` | 0.4529 | 0.4586 | -0.0057 [-0.0389, +0.0254] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_point_error_m` | 6.4306 | 6.2810 | +0.1497 [-0.1038, +0.4086] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_long_bias_m` | -2.4016 | -2.2474 | -0.1542 [-0.4820, +0.1693] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_lat_bias_m` | -0.0068 | -0.1872 | +0.1804 [-0.0643, +0.4006] | 2,399 / 141 | no |
| `tactical.goal_setting.goal_bearing_mae_deg` | 4.1132 | 4.0999 | +0.0132 [-1.1455, +0.9031] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_bearing_bias_deg` | -0.6974 | -0.5903 | -0.1072 [-1.2535, +0.8257] | 2,367 / 141 | no |
| `tactical.goal_setting.goal_range_ratio` | 0.9535 | 0.9537 | -0.0002 [-0.0160, +0.0164] | 2,367 / 141 | no |

### TACTICAL, heads (lat / lon / goal)

| cell | X | B | X - B [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `tactical.heads.lateral.macro_f1` | 0.8269 | 0.8157 | +0.0112 [-0.0319, +0.0687] | 2,107 / 139 | no |
| `tactical.heads.lateral.accuracy` | 0.9544 | 0.9549 | -0.0005 [-0.0076, +0.0071] | 2,107 / 139 | no |
| `tactical.heads.lateral.kappa` | 0.7288 | 0.7240 | +0.0047 [-0.0429, +0.0573] | 2,107 / 139 | no |
| `tactical.heads.longitudinal.macro_f1` | 0.5015 | 0.4992 | +0.0024 [-0.0378, +0.0348] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.accuracy` | 0.6308 | 0.6085 | +0.0223 [-0.0090, +0.0554] | 2,240 / 136 | no |
| `tactical.heads.longitudinal.kappa` | 0.5247 | 0.5012 | +0.0235 [-0.0153, +0.0639] | 2,240 / 136 | no |
| `tactical.heads.goal.macro_f1` | 0.2712 | 0.2618 | +0.0095 [-0.0181, +0.0723] | 2,365 / 139 | no |
| `tactical.heads.goal.token_accuracy` | 0.9562 | 0.9571 | -0.0009 [-0.0034, +0.0015] | 2,365 / 139 | no |
| `tactical.heads.goal.per_token.FOLLOW_LANE` | 0.9762 | 0.9752 | +0.0010 [-0.0032, +0.0054] | 2,107 labelled, 1910 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_L` | 0.7475 | 0.7045 | +0.0429 [-0.0700, +0.2038] | 2,320 labelled, 47 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TURN_R` | 0.7211 | 0.7619 | -0.0408 [-0.0972, +0.0166] | 2,320 labelled, 150 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_L` | 0.6087 | 0.6364 | -0.0277 (no interval) | 2,107 labelled, 11 pos | no interval: the F1 is undefined on some resamples (point difference only) |
| `tactical.heads.goal.per_token.YIELD_FOR_TURN_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 2,107 labelled, 10 pos | no interval: the F1 is undefined on some resamples (point difference only) |
| `tactical.heads.goal.per_token.YIELD` | 0.1053 | 0.1010 | +0.0043 [+0.0000, +0.0170] | 695 labelled, 85 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.STOP_POINT` | 0.3137 | 0.3498 | -0.0361 [-0.1092, +0.0300] | 2,246 labelled, 131 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.SPEED_BAND` | UNAVAILABLE | | | 0 labelled, 0 pos | no positive label on this grid (the eval reports None) |
| `tactical.heads.goal.per_token.CORRIDOR_OFFSET` | 0.2135 | 0.2331 | -0.0196 [-0.1104, +0.0663] | 695 labelled, 120 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.EVADE_IN_CORRIDOR` | 0.4000 | 0.4699 | -0.0699 [-0.2133, +0.0622] | 695 labelled, 110 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.OVERTAKE_VEHICLE` | 0.2609 | 0.0952 | +0.1656 (no interval) | 695 labelled, 20 pos | no interval: the F1 is undefined on some resamples (point difference only) |
| `tactical.heads.goal.per_token.MERGE` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 25 pos | no interval: the F1 is undefined on some resamples (point difference only) |
| `tactical.heads.goal.per_token.GAP_TARGET` | 0.0328 | 0.0000 | +0.0328 [+0.0000, +0.1081] | 695 labelled, 60 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.REACT_ON_ONCOMING` | 0.1132 | 0.0385 | +0.0747 [+0.0000, +0.2157] | 695 labelled, 50 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TAKE_EXIT_L` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label on this grid (the eval reports None) |
| `tactical.heads.goal.per_token.TAKE_EXIT_R` | 0.0588 | 0.0556 | +0.0033 (no interval) | 695 labelled, 30 pos | no interval: the F1 is undefined on some resamples (point difference only) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT` | UNAVAILABLE | | | 695 labelled, 0 pos | no positive label on this grid (the eval reports None) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_RED` | 0.6016 | 0.5273 | +0.0744 [-0.0575, +0.3282] | 720 labelled, 65 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_YELLOW` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | no interval: the F1 is undefined on some resamples (point difference only) |
| `tactical.heads.goal.per_token.TRAFFIC_LIGHT_REACT_GREEN` | 0.0000 | 0.0250 | -0.0250 [-0.0870, +0.0000] | 695 labelled, 75 pos; 2,365 / 139 | no |
| `tactical.heads.goal.per_token.LANE_CHANGE_L` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 15 pos | no interval: the F1 is undefined on some resamples (point difference only) |
| `tactical.heads.goal.per_token.LANE_CHANGE_R` | 0.0000 | 0.0000 | +0.0000 (no interval) | 695 labelled, 10 pos | no interval: the F1 is undefined on some resamples (point difference only) |

### Where n is below 2,399, and why

Each reason below is counted from the mask the cell uses (printed in the json as `n_reason`), not guessed. For the strategic cells the mask is the `ck_pos_err` NaN pattern in the dumps themselves (identical in X and B, and equal to the grid's checkpoint-label mask). The lateral, longitudinal and goal-head masks are the v9 label join's (`has`, `lat_cls`, `lon_cls`, `goal_w`): the dumps carry no label arrays, so those masks cannot come from the npz.

* 2,367 of 2,399 windows: the 32 whose human goal is within 0.5 m of the origin carry no bearing (mask: |gt[-1]| > 0.5 m). Cells: `goal_setting.goal_bearing_mae_deg`, `goal_setting.goal_bearing_bias_deg`, `goal_setting.goal_range_ratio`.
* 2,107 of 2,399 windows are labelled (mask: has & lat_cls >= 0): 34 have no v9 label row (has False) and 258 have a row with no lateral class (IGNORE). Cells: `heads.lateral.macro_f1`, `heads.lateral.accuracy`, `heads.lateral.kappa`.
* 2,240 of 2,399 windows are labelled (mask: has & lon_cls >= 0): 34 have no v9 label row (has False) and 125 have a row with no longitudinal class (IGNORE). Cells: `heads.longitudinal.macro_f1`, `heads.longitudinal.accuracy`, `heads.longitudinal.kappa`.
* 2,365 of 2,399 windows have a v9 label row (34 do not: has False); the cells scored are the 23,657 (window, token) pairs with goal_w set, threshold goal_prob > 0.5. Cells: `heads.goal.macro_f1`.
* 2,365 of 2,399 windows have a v9 label row (34 do not: has False); the cells scored are the 23,657 (window, token) pairs with goal_w set, threshold goal_prob > 0.5 (micro accuracy over those cells: dominated by the all-negative majority, so read it beside the macro-F1). Cells: `heads.goal.token_accuracy`.
* 2,322 of 2,399 windows: `ck_pos_err` is NaN in both dumps at exactly the 77 windows without a valid 30 m checkpoint label (34 have no v9 label row, has False; 43 have a row but no valid checkpoint at 30 m). Cells: `strategic.position_error_m.30`, `strategic.heading_error_deg.30`.
* 2,319 of 2,399 windows: `ck_pos_err` is NaN in both dumps at exactly the 80 windows without a valid 50 m checkpoint label (34 have no v9 label row, has False; 46 have a row but no valid checkpoint at 50 m). Cells: `strategic.position_error_m.50`, `strategic.heading_error_deg.50`.
* 2,294 of 2,399 windows: `ck_pos_err` is NaN in both dumps at exactly the 105 windows without a valid 80 m checkpoint label (34 have no v9 label row, has False; 71 have a row but no valid checkpoint at 80 m). Cells: `strategic.position_error_m.80`, `strategic.heading_error_deg.80`.

**Goal tokens.** 3 of 22 tokens have no positive label on this grid and are UNAVAILABLE (`SPEED_BAND`, `TAKE_EXIT_L`, `TRAFFIC_LIGHT_REACT`; the eval reports them as None for both arms). 8 more (`YIELD_FOR_TURN_L`, `YIELD_FOR_TURN_R`, `OVERTAKE_VEHICLE`, `MERGE`, `TAKE_EXIT_R`, `TRAFFIC_LIGHT_REACT_YELLOW`, `LANE_CHANGE_L`, `LANE_CHANGE_R`) have positives in so few episodes that the token's F1 is undefined on some resamples; no percentile interval exists for them, so they carry the point difference only. The goal macro-F1 handles this the way the eval's definition does: on each resample it averages over the tokens that have a positive label in the drawn episodes.

**Unavailable by design (both arms, the eval's own reason):** anchor selection ("each dump's arm commits to ONE path per window: there is no candidate fan and no selection to score") and selected-vs-executed (both label streams are executed manoeuvres). Both stay work items, not passes.

## 4. STRATEGIC (S's checkpoint error at 30 / 50 / 80 m; active for v8F-mini)


| cell | X | B | X - B [95 % CI] | n (windows / episodes) | separated |
|---|---|---|---|---|---|
| `strategic.position_error_m.30` | 7.0141 | 7.0231 | -0.0090 [-0.0721, +0.0590] | 2,322 / 137 | no |
| `strategic.position_error_m.50` | 8.0368 | 8.0122 | +0.0245 [-0.1547, +0.2015] | 2,319 / 137 | no |
| `strategic.position_error_m.80` | 11.3682 | 11.2059 | +0.1623 [-0.2843, +0.6321] | 2,294 / 136 | no |
| `strategic.heading_error_deg.30` | 4.1894 | 4.2002 | -0.0108 [-0.3329, +0.3380] | 2,322 / 137 | no |
| `strategic.heading_error_deg.50` | 6.7680 | 6.6864 | +0.0815 [-0.4630, +0.5949] | 2,319 / 137 | no |
| `strategic.heading_error_deg.80` | 12.9855 | 13.0540 | -0.0685 [-0.9740, +0.9320] | 2,294 / 136 | no |

Position error is the dump's own `ck_pos_err`. Heading error is rebuilt from the raw tick-frame `ckpt` and the held-age poses and is used because the rebuild reproduces the stored position error exactly (section 2). The n of each strategic cell is below 2,399 for the reason in section 3 (the checkpoint label is absent at that arc).

## 5. Separated cells

**None.** Of 37 intervals, 0 exclude 0. UNAVAILABLE: 5 (3 goal tokens with no positive label, plus anchor selection and selected-vs-executed). Point-difference-only: 8 goal tokens.

## 6. What this does not say

* **One training seed per arm.** The intervals contain no training-seed variance, so they understate the uncertainty of any lever claim; a separated interval would not have been enough, and a non-separated one is not a no-effect claim. The seed floor F = 0.1207 m (B's two seeds) is an ADE6 figure and is not transferable to these cells.
* **Tier.** Open-loop scoring of each arm's own held outputs (T1-class by the PREREG). The eval JSONs stamp `raw.tier` T0 for this eval; that labelling mismatch is the open item `RESULT_V8F_MIXO.md` section 9 already lists. The plan-derived manoeuvre read is an executed-manoeuvre agreement, not a selected-vs-executed score.
* **Multiplicity.** 37 intervals are listed; they are descriptive, not a family-wise test.
* **Added cells.** The heads' accuracy and kappa, the goal-token accuracy and the per-token F1 are not in the eval JSON, so they have no eval value to reproduce; they are checked by the brute-force bootstrap test and the mutation proof, not by the JSON control.

## 7. For the next arm (FLAT, and every later v8F eval)

The same tool takes any two `run` dumps on the eval-141 grid (it refuses otherwise). For FLAT:

```
python stack/scripts/v8f_paired_families.py --a <FLAT eval_*.npz> --b <B eval_*.npz> --out pairfam_<arm>.json --md pairfam_<arm>_tables.md --label-a <arm> --label-b B
```

with `PYTHONPATH=stack:taniteval` and `CUDA_VISIBLE_DEVICES=-1`. About 4 minutes on CPU (the identity control doubles the run; `--no-identity-control` skips it). rc 0 = controls passed, rc 2 = refusal, rc 3 = a control failed (the json is still written). `compute_pairs(A, B, ctx)` is importable for a reader that wants the blocks inline. The held-age options (`--ages`, `--hist`, `--max-age-t`, `--max-age-s`, `--window-start`) default to the M2b family's; the heading cells fall to UNAVAILABLE, with the deviation, if the rebuild does not reproduce a dump's stored `ck_pos_err`.

## 8. Manifest (nothing staged, committed or pushed)

| artifact | path (relative to the repo root) |
|---|---|
| the tool | `stack/scripts/v8f_paired_families.py` |
| its tests | `stack/tests/test_v8f_paired_families.py` |
| mutation script and output | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-pairfam2/mutate_pairfam.py`, `mutate_pairfam.out` |
| the banked read | `.../raw/2026-10-25-pairfam2/pairfam_mixo_s0.json` (schema `v8f-paired-families-v1`), `pairfam_mixo_s0_tables.md`, `pairfam_mixo_s0.out` |
| this write-up and its generator | `.../raw/2026-10-25-pairfam2/PAIRFAM_MIXO.md`, `make_pairfam_mixo_md.py` |
