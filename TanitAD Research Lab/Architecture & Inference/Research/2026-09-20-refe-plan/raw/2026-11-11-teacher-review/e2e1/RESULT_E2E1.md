<title>Teacher review E2 and E1 result</title>

# RESULT: teacher review experiments E2 (tracker-transparent scoring) and E1 (rate sensitivity)

Run 2026-11-11, CPU only, banked data only, no navtest, no GPU. Outcome rules are those of `TEACHER_REVIEW.md` s6 (landed at c57ebb47, before any number); no threshold moved. All numbers EXPLORATORY, MEASURED (ours, artifacts in `json/`). Population: held-out NAVTRAIN, **11 logs**, not navtest, so absolute levels differ from the registry's navtest 89.79.
Estimator everywhere: **log-cluster bootstrap** (resample the held-out logs with replacement, ratio mean), **10,000 draws, seed 20260927**, 95 % percentile interval. With 11 logs (E2) and 5 logs (E1) the intervals are wide and the cluster count is the limiting fact.

## E2: tracker-transparent scoring of the 1,018 banked targets

Method (`e2_score.py`): NAVSIM v1.1's own PDMScorer (vendored simulator, `navsim_pdm_targets._setup`), one scorer call per sample against PDM-Closed. Four arms on the same 1,018 samples: teacher tracked (20 poses -> 8 -> LQR, no A7 repair, as in D1), teacher transparent (the 20 poses as executed: cubic spline to 41 states at 0.1 s, analytic velocity/acceleration/yaw terms, no LQR), human tracked (8 logged poses -> LQR, NAVSIM's standard row), human transparent (logged future at 0.2 s through the identical builder). 1,018 of 1,018 scored, 0 errors; a full rerun reproduced every value bit-for-bit.

| arm (percent, [95 % CI]) | NC | DAC | TTC | EP | C | PDMS-like |
|---|---|---|---|---|---|---|
| teacher tracked | 99.26 [98.81, 99.66] | 91.95 [85.36, 97.70] | 96.46 [94.11, 98.41] | 83.04 [75.91, 89.47] | 100.0 | 86.73 [79.87, 92.83] |
| **teacher transparent** | 100.0 | **99.41 [98.67, 100.0]** | 98.63 [97.10, 99.54] | 90.08 [88.09, 92.54] | 100.0 | **94.95 [93.78, 96.15]** |
| human tracked | 100.0 | 100.0 | 99.80 [99.46, 100.0] | 87.06 [83.55, 91.00] | 100.0 | 94.53 [93.07, 96.15] |
| human transparent (as run) | 100.0 | 99.80 [99.51, 100.0] | 100.0 | 86.61 [83.21, 90.49] | 90.47 [84.14, 96.31] | 92.72 [90.48, 94.99] |
| human transparent, C neutralised (post hoc) | | | | | 100 | 94.31 [92.80, 96.04] |

Paired contrasts (points, [CI]): teacher transparent minus tracked PDMS **+8.23 [2.66, 14.48]**, DAC +7.47 [2.00, 13.91]; teacher transparent minus human tracked PDMS **+0.43 [-1.13, 1.60]**; teacher transparent minus human transparent as run **+2.23 [0.25, 3.89]**, with C neutralised (post hoc) **+0.65 [-1.01, 1.83]**.

Controls (all pass):
* **Identity:** the rerun tracked DAC=0 count is 82, equal to the banked sim8 failures (d1_navtrain, 82 = 8.055 %), agreeing on **1,018 of 1,018** samples.
* **Liveness:** transparent differs from tracked on **80** samples in DAC (78 tracked-fail/transparent-ok, 2 the reverse) and on 713 in PDMS.
* **Human row:** run through the same transparent builder and the standard tracked path (table).
* Transparent DAC failures = 6 = the banked raw20 failures (6 of 6 the same samples).

**Outcome label: A on the DAC clause; the PDMS clause is met or not depending on which human row, so the label is "A, qualified".** Transparent DAC 99.41 >= 99.3 (interval [98.67, 100] straddles the 98.5 line of B, so B is not excluded by the interval alone, but the point reading is A, not B). PDMS-like vs the human: within 1 point against the standard tracked human (+0.43) and against the comfort-neutral same-path human (+0.65); NOT within 1 against the same-path human as run (+2.23). That last reading is an instrument artifact: the same human trajectory is comfortable 100 % when tracked and 90.5 % through my finite-difference builder (noisy logged positions give noisy second derivatives), so the as-run human-transparent PDMS is depressed by comfort, not by behaviour. The neutralisation is post hoc and is reported only as a sensitivity; the transparent-comfort values (teacher 100 %) are derivation-dependent and should not be quoted.

What it implies per the review's own Outcome A text: the teacher's own rollout is essentially paper-grade once the tracker is bypassed (PDMS-like 94.95, human-level; DAC 99.41; paper 95.8 / 99.9 differs by -0.85 / -0.49, inside the interval for PDMS), so the REFe-side action is to **rescore rather than retrain**, and P5-C's expected value falls toward 0. The remaining DAC gap to 99.9 is the 6 raw rollout failures (0.59 %), as the review's class-A figure said. Caveat: 11 logs; the tracked teacher DAC interval alone spans [85, 98].

## E1: rate sensitivity

Step 1 (`e1_overlap.py`, probed E:/Projects/TanitAD/data and F:/Archive, names `targets_rank0*.jsonl`): the 20 Hz rows are the legacy rows without a `sim_hz` field in `E:/Projects/TanitAD/data/refe_navtrain/targets_rank0.jsonl` (18,122 rows = the dev-box count in D-REFE-SIMRATE-1; shards in `r0_shard0..3` hold the same). Overlap with the 1,018 by (log, token), rank 0, step 0, 20 poses: **490 (>= 300, so step 2 ran)**, in 5 logs. The `_INVALID_2s_horizon_20Hz_stride` rows were excluded (0 overlap). The pod's 4,265 rows were not probed (unreachable here). Control: sim8 on the 10 Hz rows agrees with banked d1 on 490 of 490.

Step 2 (`e1_rate.py`), same 490 samples, paired:

| | 10 Hz | 20 Hz | 20 Hz minus 10 Hz paired |
|---|---|---|---|
| sim8 DAC failure | 13.27 % [3.83, 19.65] (65 of 490) | 15.10 % [5.46, 20.89] (74 of 490) | **+1.84 pts [0.58, 3.13]**; discordant 4 (10 Hz only) vs 13 (20 Hz only) |
| class-C margin, median m (own set, n) | 0.377 [0.362, 0.492] (61) | 0.301 [0.291, 0.337] (64) | |
| same 10 Hz class-C set evaluated at 20 Hz | 0.377 | 0.297 [0.291, 0.475] | |
| all-sample margin mean change | | | -0.013 m [-0.018, -0.006] |

Median end-pose difference between rates 0.55 m. Raw-rollout off-road (class A by raw20, whether or not sim8 fails): 6 at 10 Hz, 12 at 20 Hz; class C (sim8 fail only) 61 and 64.

**Outcome label: not A; B in substance, with its stated numeric band not matched.** A required 20 Hz failure <= 3 % and margin >= 0.55 m: 15.1 % and 0.30 m, so rate/hold is not the cure. B says both rates fail 6 to 10 % with margin about 0.4 m: here both fail more (13.3 % and 15.1 %, because these 5 logs run hotter than the 1,018-sample 8.06 %), and the margin at 20 Hz is lower (0.30), not equal. The B conclusion holds a fortiori: **the thin margin is the policy, rate is eliminated as the explanation, and 20 Hz is slightly worse, not better** (the 20 Hz minus 10 Hz difference excludes 0 over 5 logs). Caveats: the only rates tested are the two banked ones, no 5 Hz-native rows exist; 5 logs.

## Combined reading

E2 says the tracker, not the rollout, carries about 7.5 of the 8 DAC-failure points on navtrain held-out, and with it removed the teacher is at human level on the PDMS formula. E1 says a rate change does not fix the tracked margin. Together: the cheap remedy is on the scoring/plan-to-tracker interface side (the 72 % protocol attribution of the review), not a teacher re-roll. Not shown here: that the paper used a transparent protocol; only that our teacher is paper-grade under one.

## Manifest (all in `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/teacher_review/e2e1/`, ONE place only; the Master Mind lands)

`RESULT_E2E1.md`; scripts `e2_score.py`, `e2_analyze.py`, `e1_overlap.py`, `e1_rate.py`, `e1_analyze.py`; launchers `launch_e2_shard_example.cmd`, `launch_e1_shard_example.cmd`; `json/e2_result.json`, `json/e2_rows_1018.jsonl`, `json/e1_result.json`, `json/e1_rows_490.jsonl`, `json/e1_overlap.json`. The scripts import `refe/navsim_dac.py`, `navtrain_scenarios.py`, `navsim_pdm_targets.py`, `navsim_pdm/` exported from tip c57ebb47 (pass `--refe`).

## Deletion log
Scratch of this run, all re-creatable (tip c57ebb47 exports, rerunnable shards; results banked above): `F:/tanitad_scratch/teacher_e2e1/w` (size below) then the empty dir `F:/tanitad_scratch/teacher_e2e1`.
