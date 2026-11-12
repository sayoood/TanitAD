# FAM_PERTOKEN: per-token arrays for every family metric, and the complete paired family CI

Base: tip `d4f0a0ff` (mirror `C:/Users/Admin/tanitad-push`). Hold dir: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/instruments/fam_pertoken/` (repo-relative mirror; the MM lands it).
Nothing staged, committed or pushed. No navtest value opened. Evidence class for every number below: MEASURED (ours), artifacts named.

## 1. What the work item turned out to be
Mostly "pass the existing flag". `families_run.py --per-window` already existed and already forwards `per_window=True` to
`taniteval/adapters/navsim.py::four_families_block`, which attaches `block["_per_window"]` (module `taniteval/taniteval/per_window.py`, schema
`taniteval.per_window/1`). The banked reroll simply never passed it. Import reachability verified: `load_adapter()` appends `<repo>/taniteval` to `sys.path`,
`from taniteval import per_window` resolves there (tip export; 16 runs OK). It needs torch (the adapter does too).
The export carries every LONGITUDINAL / LATERAL / TACTICAL component as mean / rms / ratio (num, den, scale) / median records, with tokens and eid inside.
Not in the export: `tactical.goal_setting.goal_bearing_bias_deg` (stays NO-PER-TOKEN-DATA; a gap in `per_window.py`, not edited here).

## 2. What changed (diffs: `families_run.diff` 21 lines, `fam_ci.diff`)
* `eval/instruments/families_run.py`: with `--per-window` the JSON also gets top-level `tokens`, `logs` (array order = token-sorted, `read_rows`) and
  `per_window_schema`. **The flag is DEFAULT OFF: callers must pass `--per-window`** (section 5). Default output is byte-identical to the tip (section 3).
* `raw/2026-11-12-instruments/fam_ci/fam_ci.py`: new `paired_components()`. Pairing by token id: token sets equal, token-to-log map equal, `_per_window.tokens` equal to the
  top-level `tokens`, human manoeuvre ids equal; any failure raises `TokenMismatch` (CLI exit 2); B is realigned to A's order when only the order differs.
  Statistic per record kind (clusters = logs, B 10,000, seed 20260927, percentile 2.5/97.5, one shared draw matrix):
  mean `sum(v)/n` | rms `sqrt(sum(v)/n)` | **ratio `scale*sum(num)/sum(den)` (micro mean: LATERAL errors over valid steps, subset means; each arm uses its own num/den)** |
  median | decisions: accuracy = mean of per-token 0/1 correctness, per-class recall and precision as ratios of sums, kappa from the drawn confusion counts.
  The 6 legacy distance-keeping rows are unchanged (intersection pairing). The existing 13 tests are untouched and green.
* New: `test_fam_ci_pertoken.py` (10 tests), `regress_pertoken.py` (regression control).
The question answered is still only "would another draw of LOGS say this?".

## 3. Regression control (bar 1e-9): PASS
`regress_pertoken_result.json`: 8 banked files (`run/fam/fam_{R0,RJ5,RG8,R5HZ}_{C,R}.json`), interpreter `C:/Users/Admin/navsim-crun/venv/Scripts/python.exe`
(the reroll's), PYTHONPATH = the pinned devkit, rows from `rolls_scores.tar.xz`, tip `taniteval` + `stack/tanitad` export, n-boot 2000.
1. Default run equals the banked JSON byte for byte after removing CR: 8/8. (families_run writes CRLF on Windows; the banked files are LF.)
2. `--per-window` run minus {tokens, logs, per_window_schema, `_per_window`} equals the banked JSON, every number: 8/8.
3. Per-component aggregate recomputed from the per-token arrays, rounded to the block's decimals, vs the block's printed value: **1,416 comparisons, max |diff| = 0.0**
   (3 frames x 8 files; includes accuracy, kappa, per-class recall/precision, goal metrics). 24 components have no printed value (`last_pose_heading_err_deg`,
   "extra to the four-family block"): reported, not checked.
Control sanity: corrupting one lateral `num` entry and one tactical `pred` entry makes the check flag 6 components (RED).

## 4. Tests (`CUDA_VISIBLE_DEVICES=-1 python -m unittest test_fam_ci_pertoken test_fam_ci`: 23 pass = 13 old + 10 new)
Identity [0,0] on all three families and all frames | ratio-of-sums known value (steps 1/9, error per step 10/2: A 2.8, B 3.8, diff +1, CI [1,1]) |
**mutation arm: ratio swapped for a mean of per-token means reads 6.0 instead of 2.8, so the known-value test goes RED** | constant shifts (mean, ratio, median) read c |
only the changed metric moves | accuracy/kappa/per-class on a perfect arm | human-only never compared | token-id realignment plus 3 refusals (token, log map, block tokens) |
mean path agrees with legacy `cluster_boot` to 1e-12 (not bit-identical: sum(B)-sum(A) vs sum(B-A) differ at 1e-16) | no `_per_window` gives the old behaviour.

## 5. Callers that must pass `--per-window` to get a paired CI (mirror refe-plan tree; NOT changed here)
* `raw/2026-11-11-teacher-fidelity/n2/reroll/tfn2_reroll_chain.py:181` (the reroll; its family stage would need a re-run, ~10 s per file on CPU)
* `raw/2026-11-11-teacher-fidelity/n1/tf_n1_chain.py:242` (N1 families)
* **Frozen / pinned:** `n2/navtest/tfn2_nt_chain.py:511` and `n2/navtest_a1/tfn2_nt_chain.py:532` (navtest chains; `n2/navtest*/N2_PINS.json` and `n1/N1_PINS.json` pin
  the sha256 of `families_run.py`, `n1_make_pins.py` lists it). Landing this file changes that hash: a chain started from its pinned tree is unaffected, a new pin set must be
  made for any new chain. A chain is running under `F:/tanitad_scratch/tfn2_nt/`; untouched.
* `eval/instruments/validate_controls.py:108` imports `families_run` and keeps working (`run(..., per_window=False)` default).
* `tf_n1_analyze*.py` only mention it in text.

## 6. Demo (EXPLORATORY, navtrain held-out, 469 tokens, 13 logs; frame `executed_tracked`, form R; STRATEGIC PAUSED)
Inputs: `raw_fam/fam_{R0,RJ5,R5HZ}_R_per_window.json` (made by the new families_run); outputs `demo/fam_ci_{RJ5,R5HZ}R_vs_R0R.json` (all three frames inside). Diff = B - A, A = R0.
"sep" = CI excludes 0: necessary, not sufficient for a lever claim (log-draw question only; the R-arms are deterministic CPU rollouts of one teacher).
Levels are errors against the logged human future, not a quality verdict; these are family descriptors, not a PDMS ruling. 61 paired metrics; separated: RJ5 29, R5HZ 27
(the metrics are strongly correlated). The 6 legacy rows reproduce FAM_CI.md's values exactly.

| metric | n tok/logs | R0 level | RJ5 - R0 [95% CI] | sep | R5HZ - R0 [95% CI] | sep |
|---|---|---|---|---|---|---|
| **LONGITUDINAL** | | | | | | |
| headway_min_m | 244/12 | 16.89 | -0.2625 [-0.4621, -0.1155] | yes | -0.05159 [-0.1747, +0.1395] | no |
| time_gap_min_s | 216/12 | 5.862 | -0.05996 [-0.08503, -0.01926] | yes | -0.006886 [-0.0272, +0.03799] | no |
| min_ttc_s | 244/12 | 14.29 | -0.8602 [-1.416, -0.439] | yes | -0.2124 [-0.7207, +0.3281] | no |
| lead_viol | 252/13 | 0 | +0 [+0, +0] | no | +0 [+0, +0] | no |
| lead_at_risk | 252/13 | 0.8175 | +0 [+0, +0] | no | +0 [+0, +0] | no |
| lead_progress_ratio | 204/12 | 1.156 | +0.06072 [+0.04116, +0.08395] | yes | +0.02971 [+0.01262, +0.05187] | yes |
| speed_mae_mps | 469/13 | 0.6956 | +0.18 [+0.1008, +0.2657] | yes | +0.07298 [+0.03749, +0.1083] | yes |
| speed_bias_mps | 469/13 | 0.3503 | +0.2902 [+0.2127, +0.3744] | yes | +0.102 [+0.06378, +0.1366] | yes |
| speed_rmse_mps | 469/13 | 1.005 | +0.271 [+0.1577, +0.3631] | yes | +0.1059 [+0.04485, +0.155] | yes |
| target_speed_acc.within_0.5_mps | 469/13 | 0.5408 | -0.05704 [-0.06933, -0.04003] | yes | -0.02905 [-0.04562, -0.01163] | yes |
| target_speed_acc.within_1.0_mps | 469/13 | 0.7692 | -0.07249 [-0.09843, -0.04717] | yes | -0.02559 [-0.04196, -0.01031] | yes |
| target_speed_acc.within_2.0_mps | 469/13 | 0.9283 | -0.05117 [-0.08417, -0.02088] | yes | -0.01786 [-0.03579, +0] | no |
| along_mae_m | 469/13 | 1.034 | +0.2635 [+0.1419, +0.3912] | yes | +0.1214 [+0.06252, +0.1831] | yes |
| along_bias_m | 469/13 | 0.4911 | +0.4473 [+0.3356, +0.5602] | yes | +0.1255 [+0.06044, +0.1833] | yes |
| along_final_bias_m | 469/13 | 1.248 | +1.02 [+0.7601, +1.275] | yes | +0.3542 [+0.2086, +0.488] | yes |
| accel_mae_mps2 | 469/13 | 0.4435 | +0.07623 [+0.04683, +0.108] | yes | +0.05071 [+0.03249, +0.06744] | yes |
| ego_progress.progress_ratio_mean | 465/13 | 1.098 | +0.08648 [+0.06496, +0.1117] | yes | +0.03934 [+0.02458, +0.05828] | yes |
| ego_progress.progress_ratio_median | 465/13 | 1.09 | +0.05652 [+0.04482, +0.07949] | yes | +0.01186 [+0.00373, +0.03274] | yes |
| ego_progress.progress_error_mean | 465/13 | 0.1867 | +0.04616 [+0.02673, +0.06812] | yes | +0.01731 [+0.004936, +0.03469] | yes |
| ego_progress.under_progress_rate | 465/13 | 0.3097 | -0.1032 [-0.1326, -0.07636] | yes | -0.0172 [-0.02975, -0.001934] | yes |
| ego_progress.gt_progress_mean_m | | | HUMAN-ONLY | | HUMAN-ONLY | |
| ego_progress.t0_axis_gt_self_ratio | | | HUMAN-ONLY | | HUMAN-ONLY | |
| **LATERAL** | | | | | | |
| heading_mae_deg | 467/13 | 2.515 | +0.1773 [+0.05647, +0.3017] | yes | +0.1414 [+0.09206, +0.1843] | yes |
| yaw_rate_mae_degps | 463/13 | 2.75 | +0.1491 [+0.05546, +0.2461] | yes | +0.1503 [+0.07891, +0.2149] | yes |
| curvature_mae_1pm | 463/13 | 0.01492 | -0.0003132 [-0.0004345, -0.0001671] | yes | +0.0003826 [-4.455e-06, +0.0006817] | no |
| curvature_bias_1pm | 463/13 | 0.002859 | -0.0004432 [-0.0007254, -0.0001154] | yes | -0.0003029 [-0.0007178, -2.213e-05] | yes |
| cross_mae_m | 469/13 | 0.2987 | +0.06887 [+0.02395, +0.1213] | yes | +0.03187 [+0.01826, +0.04507] | yes |
| cross_bias_m | 469/13 | -0.06427 | +0.008311 [-0.03572, +0.04045] | no | +0.000948 [-0.02402, +0.02002] | no |
| cross_final_mae_m | 469/13 | 0.7003 | +0.2266 [+0.07915, +0.4047] | yes | +0.08545 [+0.03324, +0.133] | yes |
| last_pose_heading_err_deg | 469/13 | 4.173 | +0.3291 [+0.02639, +0.5833] | yes | +0.2176 [+0.03607, +0.3425] | yes |
| **TACTICAL** | | | | | | |
| lateral_decision.accuracy | 469/13 | 0.9339 | -0.008529 [-0.02765, +0] | no | -0.01279 [-0.03125, -0.002212] | yes |
| lateral_decision.kappa | 469/13 | 0.8825 | -0.01387 [-0.0497, +0.0003462] | no | -0.02172 [-0.05496, -0.004353] | yes |
| lateral_decision.per_class.lane_keep.recall | 283/13 | 0.9081 | -0.01767 [-0.04615, +0] | no | -0.01767 [-0.04615, +0] | no |
| lateral_decision.per_class.lane_keep.precision | 262/13 | 0.9809 | +0.003459 [-0.0008774, +0.01336] | no | -0.004172 [-0.01585, -6.588e-05] | yes |
| lateral_decision.per_class.turn_left.recall | 131/11 | 0.9771 | +0 [+0, +0] | no | -0.01527 [-0.02721, +0] | no |
| lateral_decision.per_class.turn_left.precision | 143/12 | 0.9078 | -0.0127 [-0.03051, +0] | no | -0.01418 [-0.03273, +0] | no |
| lateral_decision.per_class.turn_right.recall | 55/10 | 0.9636 | +0.01818 [+0, +0.03659] | no | +0.01818 [+0, +0.03659] | no |
| lateral_decision.per_class.turn_right.precision | 70/10 | 0.803 | -0.0316 [-0.1265, +0.00488] | no | -0.0316 [-0.1265, +0.00488] | no |
| longitudinal_decision.accuracy | 469/13 | 0.6695 | -0.01279 [-0.02635, +0.01111] | no | +0.004264 [-0.01034, +0.01828] | no |
| longitudinal_decision.kappa | 469/13 | 0.5096 | -0.0166 [-0.03458, +0.01787] | no | +0.006102 [-0.01499, +0.02588] | no |
| longitudinal_decision.per_class.brake_stop.recall | 158/12 | 0.6456 | -0.01899 [-0.03759, +0] | no | -0.01266 [-0.02618, +0] | no |
| longitudinal_decision.per_class.brake_stop.precision | 130/12 | 0.7907 | -0.004983 [-0.01596, +0] | no | +0.002953 [-0.0157, +0.02493] | no |
| longitudinal_decision.per_class.steady.recall | 181/12 | 0.4972 | -0.03315 [-0.0718, +0.008852] | no | +0.01657 [-0.02083, +0.05023] | no |
| longitudinal_decision.per_class.steady.precision | 145/13 | 0.6429 | +0.01339 [-0.02379, +0.06554] | no | +0.007493 [-0.01659, +0.03341] | no |
| longitudinal_decision.per_class.accelerate.recall | 130/11 | 0.9385 | +0.02308 [+0, +0.07143] | no | +0.007692 [+0, +0.03191] | no |
| longitudinal_decision.per_class.accelerate.precision | 216/12 | 0.61 | -0.0286 [-0.05515, +0.01471] | no | +0.005 [-0.009889, +0.02421] | no |
| maneuver_5way_collapsed.accuracy | 469/13 | 0.7932 | -0.002132 [-0.0113, +0.007394] | no | -0.006397 [-0.01636, +0.002421] | no |
| maneuver_5way_collapsed.kappa | 469/13 | 0.7367 | -0.002518 [-0.01401, +0.009345] | no | -0.007978 [-0.02102, +0.00335] | no |
| maneuver_5way_collapsed.per_class.lane_keep.recall | 83/9 | 0.4096 | +0 [-0.04225, +0.07273] | no | +0 [-0.02941, +0.04918] | no |
| maneuver_5way_collapsed.per_class.lane_keep.precision | 56/8 | 0.6415 | +0.01234 [-0.03272, +0.1335] | no | -0.02333 [-0.07671, +0.01323] | no |
| maneuver_5way_collapsed.per_class.turn_left.recall | 131/11 | 0.9771 | +0 [+0, +0] | no | -0.01527 [-0.02721, +0] | no |
| maneuver_5way_collapsed.per_class.turn_left.precision | 143/12 | 0.9078 | -0.0127 [-0.03051, +0] | no | -0.01418 [-0.03273, +0] | no |
| maneuver_5way_collapsed.per_class.turn_right.recall | 55/10 | 0.9636 | +0.01818 [+0, +0.03659] | no | +0.01818 [+0, +0.03659] | no |
| maneuver_5way_collapsed.per_class.turn_right.precision | 70/10 | 0.803 | -0.0316 [-0.1265, +0.00488] | no | -0.0316 [-0.1265, +0.00488] | no |
| maneuver_5way_collapsed.per_class.accelerate.recall | 73/10 | 0.9589 | +0 [+0, +0] | no | +0 [+0, +0] | no |
| maneuver_5way_collapsed.per_class.accelerate.precision | 112/11 | 0.6481 | +0.01223 [-0.02214, +0.0656] | no | +0.03146 [+0, +0.07333] | no |
| maneuver_5way_collapsed.per_class.brake_stop.recall | 127/11 | 0.685 | -0.01575 [-0.04545, +0] | no | -0.01575 [-0.04, +0] | no |
| maneuver_5way_collapsed.per_class.brake_stop.precision | 101/12 | 0.8614 | +0.005961 [-0.008929, +0.03027] | no | -0.01139 [-0.04728, +0.01933] | no |
| goal_setting.goal_point_error_m | 469/13 | 2.681 | +0.7169 [+0.3942, +1.059] | yes | +0.2434 [+0.09348, +0.3904] | yes |
| goal_setting.goal_long_bias_m | 469/13 | 1.248 | +1.02 [+0.7601, +1.275] | yes | +0.3542 [+0.2086, +0.488] | yes |
| goal_setting.goal_lat_bias_m | 469/13 | -0.1441 | +0.05861 [-0.07675, +0.1673] | no | +0.01724 [-0.04533, +0.07115] | no |
| goal_setting.goal_bearing_mae_deg | 465/13 | 2.233 | +0.3271 [+0.08114, +0.5801] | yes | +0.1704 [+0.06391, +0.2557] | yes |
| goal_setting.goal_range_ratio | 465/13 | 1.102 | +0.08816 [+0.06628, +0.1137] | yes | +0.0404 [+0.02527, +0.05963] | yes |
| goal_setting.goal_bearing_bias_deg | | | NO-PER-TOKEN-DATA | | NO-PER-TOKEN-DATA | |
| **STRATEGIC** | PAUSED (PI 2026-09-19) | | | | | |

Reading (descriptive): both knobs move LONGITUDINAL. Speed MAE and bias, along-track bias and goal long-bias rise (RJ5 +0.18 m/s MAE, +1.02 m final along bias; R5HZ roughly 0.35 to 0.4 times
of that), and progress ratio rises, so both arms drive further and faster than the logged human, consistent with a closer lead gap. LATERAL: heading, yaw-rate and cross MAE rise
slightly (RJ5 heading +0.18 deg, CI excludes 0); signed cross-track bias is unmoved. TACTICAL decisions: nothing separates for RJ5; R5HZ lateral accuracy -0.013 and kappa -0.022 separate
narrowly (sparse classes: CIs touching exactly 0 are discrete-count artefacts). Goal point error +0.72 m (RJ5), +0.24 m (R5HZ). Not decision-grade: 13 logs.

## 7. Open items
* `goal_bearing_bias_deg`: no export in `per_window.py` (UNVERIFIED which lines to add; the block prints it from `tactical_goal`).
* `lateral.last_pose_heading_err_deg` has no printed block value to regress against.
* Default-off flag: nothing changes for existing chains until a caller passes it.
