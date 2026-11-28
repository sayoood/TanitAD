# FAM_CI: paired log-cluster bootstrap CI for the four metric families

Files: `fam_ci.py` (instrument), `test_fam_ci.py` (13 unittest, CPU, all pass), `demo/fam_ci_RJ5R_vs_R0R.json`, `demo/fam_ci_R5HZR_vs_R0R.json`.
Run: `python fam_ci.py --a fam_R0_R.json --b fam_RJ5_R.json --tokens run/tokens.txt --out out.json` (exit 2 = refused on token mismatch).

## What it computes
For two `families_run.py` JSONs over the same tokens: per metric, arm A level, arm B level and the paired difference B - A, each with a
log-cluster bootstrap 95 % CI, grouped per family (LONGITUDINAL, LATERAL, TACTICAL; STRATEGIC = PAUSED, PI ruling 2026-09-19), never pooled,
for all three frames (`executed_tracked` PRIMARY, `executed_transparent` and `reference` diagnostic).
Estimator = `tfn2_analyze.boot` convention (clusters = logs, resample logs with replacement, statistic = mean over the drawn tokens, B 10,000,
seed 20260927, percentile 2.5/97.5); a test re-implements that function verbatim and requires bit-identity.
Subset metrics pair on the intersection of tokens defined in both arms, with n tokens and n logs per metric; levels are on that same intersection.
Logs without a valid token in the intersection are not resampled.

**Question answered: "would another draw of LOGS say this?"** It does NOT answer "another training run" or "another inference run".
The R-arms here are deterministic CPU rollouts of one teacher with one knob changed, so there is no inference variance to speak of
(HANDOFF.md of the re-roll); a separated CI is still only the log-draw statement.

## Findings that matter more than the numbers (MEASURED, artifacts below)
1. **The banked JSONs carry per-token values for only 6 metrics**, all under `longitudinal.distance_keeping._per_window`
   (`headway_min_m, time_gap_min_s, min_ttc_s, lead_viol, lead_at_risk, lead_progress_ratio`; `fam_R0_R.json`).
   Everything else is reported `NO-PER-TOKEN-DATA` with its JSON-level point value: LONGITUDINAL 16 (speed/along/accel/progress metrics),
   LATERAL 7 (heading, yaw rate, curvature, cross-track), TACTICAL 12 scalars (decision accuracy/kappa, goal error/bearing/range/bias).
   The paired family CI is therefore **complete only for the distance-keeping half of LONGITUDINAL**; LATERAL and TACTICAL have no paired CI yet.
2. **The JSONs store no token ids and no log names.** `tokens.txt` lines are `log:token`; the JSON arrays are in token-sorted order
   (`families_run.read_rows` sorts by token; rows hold the bare token). I verified this mapping on the banked data: with the order from
   `tokens.txt` sorted by token, finite `headway_min_m` appears on 0 of the 217 NO_LEAD tokens (shuffled order: 107 violations)
   (`run/rolls_scores.tar.xz` rows, `lead.state`). Token-set refusal for these files therefore rests on count checks
   (`n`, `n_logs`, `lead_states`, array length), not on id equality.
3. **Work item for `families_run.py` (not done, not mine to edit):** (a) emit `tokens` and `logs` (array order) at top level so the pair check
   is by identity; (b) emit per-window arrays for LONGITUDINAL (per-token speed MAE, along MAE, accel MAE, progress ratio), LATERAL (per-token
   heading/yaw/curvature/cross error sums AND valid-step counts, because those are MICRO means: sum(err)/sum(valid steps); the cluster
   statistic must then be a ratio of sums, not a mean of per-token means) and TACTICAL (per-token true/pred class, per-token goal error).
   (c) ~~`--per-window` is currently unreachable~~ **RETRACTED by the MM (2026-11-12):** the agent reported `taniteval.per_window` absent
   from the tree. It is present at tip e1b3a762 as `taniteval/taniteval/per_window.py`, with `taniteval/tests/test_per_window_export.py`
   (MM `git ls-tree` probe). The agent looked at one location (`taniteval/per_window.py`) only. Whether `families_run.py`'s own run-tree
   export puts that package on `sys.path` is UNVERIFIED; check it when the work item is done.
   `fam_ci.py` needs a small extension for ratio-of-sums metrics once those arrays exist.

## Demo (EXPLORATORY, navtrain held-out, 469 tokens, 13 logs; frame `executed_tracked`, form R)
Inputs: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2/reroll/run/fam/fam_{R0,RJ5,R5HZ}_R.json`, `run/tokens.txt`.
Evidence class: MEASURED (ours; outputs `demo/*.json`). Levels reproduce the JSON's own aggregates (headway 16.8888, time gap 5.862, TTC 14.2933).
Diff = B - A (A = R0). "sep" = CI excludes 0 (necessary, not sufficient for a lever claim; this is the log-draw question only).

| metric | n tok / logs | R0 level | RJ5 - R0 [95 % CI] | sep | R5HZ - R0 [95 % CI] | sep |
|---|---|---|---|---|---|---|
| headway_min_m | 244 / 12 | 16.889 | -0.263 [-0.462, -0.116] | yes | -0.052 [-0.175, +0.140] | no |
| time_gap_min_s | 216 / 12 | 5.862 | -0.060 [-0.085, -0.019] | yes | -0.007 [-0.027, +0.038] | no |
| min_ttc_s (censored at 30 s) | 244 / 12 | 14.293 | -0.860 [-1.416, -0.439] | yes | -0.212 [-0.721, +0.328] | no |
| lead_viol | 252 / 13 | 0.000 | 0 [0, 0] | no | 0 [0, 0] | no |
| lead_at_risk | 252 / 13 | 0.818 | 0 [0, 0] | no | 0 [0, 0] | no |
| lead_progress_ratio | 204 / 12 | 1.156 | +0.061 [+0.041, +0.084] | yes | +0.030 [+0.013, +0.052] | yes |
| LATERAL (7 metrics), TACTICAL (12 scalars), 16 other LONGITUDINAL | | point only | NO-PER-TOKEN-DATA | | NO-PER-TOKEN-DATA | |
| STRATEGIC | | PAUSED | | | | |

Reading: RJ5 closes on the lead a little more (smaller headway/time gap/TTC) while progressing more relative to the lead; R5HZ is within the log-draw
noise except progress ratio. These are family descriptors, not a PDMS verdict; they do not say which arm is better.
The other two frames are in the JSON files. TTC means are over censored data (63 of 244 windows never close; `n_closing` 181 in the source JSON).

## Tests (`CUDA_VISIBLE_DEVICES=-1 python -m unittest test_fam_ci -v`, 13 pass)
Self vs self: every difference exactly [0, 0]. Constant shift c: point and both bounds equal c (1e-12). Shift on one metric moves only that metric.
Token mismatch (set, count, `n` vs tokens, `lead_states`, missing tokens): refused with counts. NaN/None intersection counts exact.
Order realignment. Seed fixes the result bit for bit (and a different seed changes it). NO-PER-TOKEN-DATA and PAUSED present.
**Mutation arm:** on a fixture with perfectly within-log-correlated values (8 logs x 25 tokens) the cluster CI is 4 to 6.5x the per-token CI (design
effect 25 -> SE ratio 5); `compare()` must keep it wider than 4x the per-token width, and `test_mutation_detected` monkeypatches the cluster resample
to a per-token one and requires that check to FAIL (it does).
