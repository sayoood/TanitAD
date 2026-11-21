# PREREG_LANE_SPEED: does u400's lane head read speed as danger? (written 2026-11-08, BEFORE any lane-truth correlation was computed)

Tier: **T0** (teacher-forced / table read; a scorer-calibration diagnostic, NOT driving performance). Families: LONGITUDINAL is the only one touched (speed); LATERAL, TACTICAL n/a; STRATEGIC PAUSED (PI 2026-09-19, ITEM 25).
Nothing in this file was run before it was hashed. What existed before: the u400 reference values (`refe_behaviour_20261107/u400_reference.json`) for the NC/DAC/EP/TTC heads, and the predicted lane Spearman -0.329 (P1_DIAG, legacy estimator, no truth).

## 1. Question and estimand
Does the lane output (7th logit) of u400 (`model_sft4_B_u400.pt`) rank faster proposals as worse than the lane truth does?
Estimand, per token i of the population: `gap_i = rho_pred_i - rho_true_i`, where both are within-token Spearman against `v4/v0`.
- `v4/v0`: `sel_common.step_speed(traj_all_A0, idx=19) / max(v0, 1e-6)` (the 0.2 s chord at 4 s on the raw proposal), as in `u400_reference.py`.
- `rho_pred_i`: Spearman of the head output (sigmoid of logit 6; rank-equivalent to the logit) vs `v4/v0`.
- `rho_true_i`: Spearman of the TRUE lane target vs `v4/v0`.
- Valid proposals only (`table.npz` `valid`), at least 5 per token (as `u400_reference.py`).
- Ranks: **tie-correct average ranks** (`scipy.stats.rankdata`), the estimator of `u400_reference.py`/`PREREG_P6_SCORER_SPEED.md`.

## 2. Population and token entry
The 154 CRUISE tokens of the C3 table (`refe_ego_probe_20261106/tokens_cruise.json`, sha256 `57db811e644218c00020cd6d25b417b604ffc9af7b81612976d0d52f02263f78`).
A token ENTERS the paired estimate only if its true lane target varies across its valid proposals (rank std > 0) and the prediction varies too. **n_entered is reported with n = 154.** Tokens that do not enter are not imputed. If a token cannot be labelled (scenario build fails), it is dropped and counted as `n_unlabelled`.

## 3. The truth: definition from source (verified before the read)
The lane head was trained on `clip(1 - (d - 0.5) / (2.0 - 0.5), 0, 1)`, with `d = center_line.CenterLine.info` of the teacher simulator (DriveRL calculators, stride 2), max over the horizon, per proposal.
- Label construction: `refe/train.py:535` (`LANE_D0, LANE_D1 = 0.5, 2.0`) and `:563-568` (E: working tree; `:537`, `:563-568` on the mirror tip e112911d). Producer: `refe/onpolicy_relabel_teacher_lane.py:35-36` (`LEAVES`), `:41-57` (`label_log`), run with `--repair-last-heading` (the banked labels carry `repair: last_heading_hold`).
- Loss: `refe/scorer_finetune.py` `lane_bce` = BCE-with-logits of logit 6 against `tg[...,6]`, masked where the target is below 0 (E: working tree `:707-713`).
- Primary truth variable in this study = the SOFT TARGET `t = clip(1-(d-0.5)/1.5, 0, 1)` (what the head was trained to output). Secondary (labelled sensitivity only, never decides): `-d`.
- INHERITED, not re-verified: that the u400 training run used these side files with `--lane-head` (PREREG_SCORER1 section 2 lists `--lane-head --lane-weight`).

**Truth is rebuilt on CPU** by feeding the u400 raw proposals (`results.npz` `traj_all_A0`, 64 x 20 x 3, rounded to 5 decimals as `onpolicy_dump.py`) as queue rows (step 0, rank 0) to `onpolicy_relabel_teacher_lane.label_log` UNMODIFIED, `--repair-last-heading`. The teacher trajectory of a queue row only sets the EP denominator; it cannot reach the centre-line leaf. A placeholder (the banked pick's trajectory) is used and this is TESTED, not assumed (G-L2).

## 4. Gates (the read is INVALID if any fails; outcomes in section 6 apply only on a pass)
- **G-L1 (definition control against banked training-side labels).** 24 held-out sets whose banked label exists (`raw/2026-10-04-lane-discipline/heldout_teacher_lane_labels.jsonl.xz`, 3,137 sets, ckpt 10075) are relabelled by the pipeline of section 3 and compared proposal by proposal. PASS = at least 99 % of the 24 x 64 proposals have |delta d| <= 1e-3 m, and the soft target differs by at most 0.01 on at least 99 %. (Known cross-machine float noise is ~1e-7, with one banked near-tie flip of 0.08 m; the 1 % allowance covers that. Every unequal proposal is listed.)
- **G-L2 (placeholder-teacher invariance).** On the same control rows relabelled with the REAL teacher, `center_line.CenterLine.info` equals the placeholder-teacher value on 100 % of proposals (|delta| <= 1e-6).
- **G-L3 (join and coverage).** Of the 154 CRUISE tokens at least 150 are labelled with all 64 proposals finite; the token order and ids equal `tokens_cruise.json`.
- **G-L4 (regression of the instrument).** The predicted side reproduces `p1_diag_all.json` lane Spearman (legacy estimator) -0.329 to within 0.001, and the slow-pick margin block reproduces "lane is the largest term on 31 of 55" (P1_DIAG Q2) exactly.

## 5. Estimator
Paired log-cluster bootstrap over logs, B = 10,000, seed 20260927, 95 % percentile (`sel_common.log_boot_idx`, `boot_mean`, the programme's draws). Point estimate and interval of the mean of `gap_i` over entered tokens; also `rho_pred` and `rho_true` on the same entered tokens. This interval answers "another draw of logs" only (one checkpoint, deterministic planner; not training or inference variance, `H-ESTIM-SEED-1`). Estimator named in every table.

## 6. The bar and both outcomes (committed)
Let `G` = mean paired `gap` (point) and `[lo, hi]` its interval.
- **O1, the lane head READS SPEED AS DANGER:** `G < -0.15` AND `hi < 0`. Meaning: the lane head ranks faster proposals as worse than the teacher-lane truth does, as the NC/TTC heads do. ACTION: **add the lane head to the pairwise loss**, as a fourth head in L_pair (heads NC, DAC, TTC, lane), balanced F/S sampling unchanged, with the lane truth as the label (O2 of PREREG_SCORER1 realised). That needs the lane truth for the navtrain tokens (the SFT-4 side files already exist for the training sets), so it costs no new label run on the train side; the E2 set would need the lane table built here for its 2,000 tokens (CPU).
- **O2, the lane head is CALIBRATED on speed:** `|G| <= 0.15` (point). Meaning: the lane head does not read speed as danger beyond the bar; its large share of the slow-pick margin comes from where it is truly low (or from a level offset), not from a speed slope. ACTION: **do not add a lane arm for the speed defect**; keep SCORER1 as written (NC, DAC, TTC).
- **O3, the lane head reads speed as SAFER than truth:** `G > +0.15` AND `lo > 0`. Meaning: it is not a source of the slow-down. ACTION: no lane arm for this defect; note it as a separate calibration finding.
- **O4, INCONCLUSIVE:** every other case (including `|G| > 0.15` with an interval that contains 0, and `n_entered < 30`). ACTION: the Master Mind decides; the default is NOT to add the lane head on this evidence, and the missing power is stated (n_entered, interval width).
A committed bar missed is reported as written. The bar is not moved after the data, and no estimator is swapped.

## 7. Margin-share block (descriptive; no decision rule beyond G-L4)
Population = the P1 definition: CRUISE tokens whose pick has `v4/v0 < 0.85` and which have a faster valid proposal (`v4/v0 >= 0.9`), comparator `f_sc` (the scorer's own best fast proposal) and `f_or` (oracle), the `attrib` code of `p1_diag.py`. Reported: the lane term, the other five, their sum, and the margin `log agg(pick) - log agg(f)`; **identity check: max over tokens of |sum of terms - margin| <= 1e-9**; the dominant-term counts. Additionally, where lane truth exists: the TRUE lane target of pick vs f, and the share of tokens where the lane head ranks f below pick although the truth ranks f at least as high.

## 8. Regression arm (labelled)
The same within-token Spearman under the LEGACY `argsort(argsort(.))` estimator (`p1_diag.py::spearman_rows`), reported beside the tie-correct one with the difference. It does not decide.

## 9. Inputs (pins)
| input | sha256 (prefix) |
|---|---|
| `table.npz` (C3, mirror tip e112911d, blob 7e89d02c) | recorded in RESULT |
| `refe_ego_probe_20261106/full/results.npz` | recorded in RESULT |
| `tokens_cruise.json` | `57db811e644218c0` |
| `heldout_teacher_lane_labels.jsonl.xz` | `570e4cb7629f136c` |
| `onpolicy_relabel_teacher_lane.py` (E: = tip) | `6f30d2e0168feaf4` |
| `onpolicy_label.py` (E: = tip) | `caf4c66c48108fa4` |
| `sel_common.py`, `p1_diag.py`, `u400_reference.py` | recorded in RESULT |
Environment: `CUDA_VISIBLE_DEVICES=-1`, DriveRL teacher code `C:/Users/Admin/dz/DriveZero/DriveRL`, venv `C:/Users/Admin/venvs/driverl-eval`, nuPlan DBs and maps at the E: paths (drive-letter flip noted in LANE_TRUTH.md), at most 4 CPU workers.
