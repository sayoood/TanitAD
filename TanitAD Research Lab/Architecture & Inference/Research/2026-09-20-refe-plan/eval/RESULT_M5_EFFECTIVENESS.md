# RESULT -- Measure 5 effectiveness (label_version 4 slow-plan labels; scorer-only dev-box fine-tune)

*Pre-registration: `eval/PREREG_MEASURE5.md`, blob `c668b5b6` (SPEC_PREREG_HASH, 2026-09-27 13:22 Berlin), criteria, arms, readout and decision rule applied AS WRITTEN. Numbers in the tables below are written by `eval/m5_write_result.py` from the banked artifacts; none is typed.*

## Verdict (as registered): **ADOPT**

| criterion (§5) | bar | measured | met |
|---|---|---|---|
| E1 V4 - V3 (masked slow-plan pair concordance) | >= +0.03 | +0.1534 | yes |
| E1 CI lower bound | > 0 | +0.1037 (CI [+0.1037, +0.2104]) | yes |
| E1 effect > the seed floor | > 0.0403 | +0.1534 | yes |
| E3a V4 - V3 CI lower bound (pair concordance among the 64) | > -0.01 | -0.0081 (diff -0.0018) | yes |
| E3b V4 - V3 CI lower bound (PDMS of the pick among the 64) | > -2.0 | -0.22 (diff +1.25) | yes |
| REFUTED iff E1 CI upper bound < 0 | -- | +0.2104 | not refuted |

Estimator: paired log-cluster bootstrap over the logs, 10,000 resamples, 95 % percentile, seed 20260927 (eval/snapshot_pair_under_rule.py's draws). Variance answered: EPISODES for the CI; TRAINING (batch order) by the seed floor; inference is deterministic. n = 200 tokens over 93 logs; seeds [0, 1, 2]. Estimator self-test (reproduces snapshot_pair_under_rule's published ep014 -> ep015 interval): {'reference_ci': [-2.43, 6.56], 'ours': [-2.43, 6.56], 'identical': True}.

## Gates (all must pass before the statistic is read)

| gate | value | bar |
|---|---|---|
| the 8 copies scored by the model == the harness-scored copies | max abs 0.0 | 0.0 |
| harness rows missing / invalid for the extras | 0 | 0 |
| A0 (re-run on the new cache) vs the E-6 table's logits | max abs 3.337860107421875e-06 | <= 1e-3 |
| A0's pick == the table's pick | 200 / 200 | all |
| masked route leaves the 64's scores unchanged | max abs 1.0013580322265625e-05 | <= 1e-4 |
| ADDED control: A0 vs the INDEPENDENT reference (slow-copy probe GPU dump, loop concordance; `a0_reference.json`) | E1 0.64998 vs 0.64998; E3a 0.64694 vs 0.64694; E3b 61.18301 vs 61.183; E2 70.07764 vs 70.078 | equal to rounding |

## Arm token means (every model)

| model | E1 masked (PRIMARY) | E1 plain | E3a | E3b PDMS | E2 PDMS |
|---|---|---|---|---|---|
| A0 | 0.6500 | 0.6612 | 0.6469 | 61.18 | 70.08 |
| V3_s0 | 0.5774 | 0.5933 | 0.6372 | 62.64 | 68.48 |
| V3_s1 | 0.5901 | 0.6064 | 0.6530 | 60.84 | 67.58 |
| V3_s2 | 0.6177 | 0.6434 | 0.6430 | 62.75 | 70.77 |
| V4_s0 | 0.7348 | 0.7575 | 0.6370 | 64.51 | 76.55 |
| V4_s1 | 0.7484 | 0.7863 | 0.6488 | 62.81 | 76.90 |
| V4_s2 | 0.7622 | 0.7779 | 0.6420 | 62.66 | 77.93 |
| V4all_s0 | 0.7649 | 0.7696 | 0.6407 | 62.81 | 79.55 |
| V4all_s1 | 0.7746 | 0.7897 | 0.6445 | 62.62 | 78.20 |
| V4all_s2 | 0.7830 | 0.7789 | 0.6409 | 61.12 | 80.09 |

## Paired comparisons (arm = seed mean per token; paired log-cluster bootstrap)

| metric | pair | diff [95 %] | separated |
|---|---|---|---|
| E1_masked | V4_minus_V3 | +0.1534 [+0.1037, +0.2104] | yes |
| E1_masked | V4all_minus_V4 | +0.0257 [+0.0158, +0.0368] | yes |
| E1_masked | V3_minus_A0 | -0.0549 [-0.0912, -0.0248] | yes |
| E1_plain | V4_minus_V3 | +0.1596 [+0.1176, +0.2057] | yes |
| E1_plain | V4all_minus_V4 | +0.0055 [+0.0020, +0.0093] | yes |
| E1_plain | V3_minus_A0 | -0.0469 [-0.0695, -0.0236] | yes |
| E3a | V4_minus_V3 | -0.0018 [-0.0081, +0.0048] | no |
| E3a | V4all_minus_V4 | -0.0006 [-0.0112, +0.0094] | no |
| E3a | V3_minus_A0 | -0.0025 [-0.0212, +0.0160] | no |
| E3b | V4_minus_V3 | +1.25 [-0.22, +2.84] | no |
| E3b | V4all_minus_V4 | -1.14 [-3.36, +0.75] | no |
| E3b | V3_minus_A0 | +0.89 [-2.64, +4.30] | no |
| E2 | V4_minus_V3 | +8.19 [+5.09, +11.81] | yes |
| E2 | V4all_minus_V4 | +2.16 [-0.09, +4.30] | no |
| E2 | V3_minus_A0 | -1.14 [-4.52, +1.91] | no |

Seed floor (E1 masked, largest |seed_i - seed_j| of the token mean within an arm): **0.04027** -- {"V3": {"seed_token_means": [0.57738, 0.59006, 0.61765], "max_pair_abs": 0.04027}, "V4": {"seed_token_means": [0.73481, 0.74844, 0.76216], "max_pair_abs": 0.02735}}.

## Ego progress (EP) and the other harness sub-scores of the PICK

Harness sub-scores x100 of each arm's pick (arm = seed mean per token). E3b = the argmax among the 64 (the shipped route); E2 = the argmax among the 64 + 9 extras (masked route). `eval/raw/m5_effectiveness/families.json`.

**E3b** arm means: A0 EP 58.24 / PDMS 61.18; V3 EP 58.62 / PDMS 62.08; V4 EP 59.52 / PDMS 63.33; V4all EP 60.67 / PDMS 62.18

| pair | NC | DAC | EP | TTC | C | DDC | PDMS |
|---|---|---|---|---|---|---|---|
| E3b V4_minus_V3 | +0.42 [-0.71, +1.51] | +0.17 [-1.47, +1.84] | +0.90 [-0.38, +2.25] | +1.67 [-0.18, +3.65] | +0.83 [-0.76, +2.53] | +0.17 [-0.49, +0.86] | +1.25 [-0.22, +2.84] |
| E3b V3_minus_A0 | +1.67 [-1.50, +4.92] | -0.83 [-4.69, +2.73] | +0.38 [-3.06, +3.69] | +0.33 [-4.19, +4.76] | +2.83 [-0.54, +6.49] | +1.33 [-0.56, +3.33] | +0.89 [-2.64, +4.30] |
| E3b V4_minus_A0 | +2.08 [-0.72, +5.10] | -0.67 [-4.06, +2.52] | +1.28 [-1.89, +4.35] | +2.00 [-2.06, +5.99] | +3.67 [+0.00, +7.72] | +1.50 [-0.66, +3.70] | +2.14 [-0.98, +5.15] |
| E3b V4all_minus_V4 | -1.50 [-3.46, +0.40] | +0.33 [-1.59, +2.08] | +1.15 [-0.83, +3.02] | -2.83 [-5.41, -0.49] | -4.00 [-7.31, -1.16] | +0.50 [-0.75, +1.85] | -1.15 [-3.36, +0.75] |

Seed floor per column (E3b): {"NC": 3.0, "DAC": 3.0, "EP": 2.179, "TTC": 2.5, "C": 3.0, "DDC": 2.5, "PDMS": 1.908}

**E2** arm means: A0 EP 54.69 / PDMS 70.08; V3 EP 52.42 / PDMS 68.94; V4 EP 61.27 / PDMS 77.12; V4all EP 67.98 / PDMS 79.28

| pair | NC | DAC | EP | TTC | C | DDC | PDMS |
|---|---|---|---|---|---|---|---|
| E2 V4_minus_V3 | +3.33 [+0.95, +6.30] | +4.00 [+0.46, +8.03] | +8.84 [+5.63, +12.27] | +4.50 [+1.76, +7.62] | +13.00 [+7.45, +19.07] | +0.33 [-0.65, +1.31] | +8.19 [+5.09, +11.81] |
| E2 V3_minus_A0 | +0.17 [-2.22, +2.49] | -1.17 [-5.06, +2.84] | -2.26 [-5.52, +0.67] | +0.17 [-3.07, +3.40] | -1.83 [-5.94, +2.37] | +2.42 [+0.59, +4.45] | -1.14 [-4.52, +1.91] |
| E2 V4_minus_A0 | +3.50 [+0.57, +6.84] | +2.83 [+0.19, +5.64] | +6.58 [+2.40, +10.83] | +4.67 [+1.22, +8.53] | +11.17 [+5.58, +17.66] | +2.75 [+1.05, +4.69] | +7.05 [+3.39, +10.86] |
| E2 V4all_minus_V4 | -1.33 [-3.60, +0.51] | -0.50 [-2.02, +0.91] | +6.71 [+4.46, +9.07] | -3.83 [-6.33, -1.86] | +8.67 [+4.25, +13.47] | -2.25 [-4.15, -0.83] | +2.16 [-0.09, +4.30] |

Seed floor per column (E2): {"NC": 1.5, "DAC": 4.5, "EP": 2.913, "TTC": 2.0, "C": 5.5, "DDC": 1.0, "PDMS": 3.192}

E2 selection mix (original / 0.75x copy / STOP) per model: {"A0": {"original": 73, "copy_075": 48, "stop": 79}, "V3_s0": {"original": 78, "copy_075": 37, "stop": 85}, "V3_s1": {"original": 93, "copy_075": 29, "stop": 78}, "V3_s2": {"original": 80, "copy_075": 41, "stop": 79}, "V4_s0": {"original": 53, "copy_075": 72, "stop": 75}, "V4_s1": {"original": 64, "copy_075": 71, "stop": 65}, "V4_s2": {"original": 47, "copy_075": 98, "stop": 55}, "V4all_s0": {"original": 53, "copy_075": 116, "stop": 31}, "V4all_s1": {"original": 61, "copy_075": 95, "stop": 44}, "V4all_s2": {"original": 49, "copy_075": 121, "stop": 30}}

## The four metric families (families6.py, per-model blocks vs the logged human future; seed means)

| route | model | long: speed MAE m/s | long: progress ratio | long: along MAE m | lat: heading MAE deg | lat: cross MAE m | lat: curvature MAE 1/m | tac: goal-point err m | tac: long. kappa | tac: lat. kappa | strategic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E3b | A0 | 1.1749 | 1.0867 | 1.7077 | 3.5372 | 0.61 | 0.0258 | 4.7473 | 0.3848 | 0.81 | UNAVAILABLE |
| E3b | V3 | 1.0367 | 1.0334 | 1.46 | 3.376 | 0.5669 | 0.0263 | 4.1555 | 0.391 | 0.8334 | UNAVAILABLE |
| E3b | V4 | 1.0533 | 1.0374 | 1.4943 | 3.4407 | 0.5723 | 0.0276 | 4.2256 | 0.3966 | 0.8391 | UNAVAILABLE |
| E3b | V4all | 0.9543 | 1.0716 | 1.3515 | 3.3737 | 0.538 | 0.0224 | 3.8586 | 0.4416 | 0.8269 | UNAVAILABLE |
| E2 | A0 | 3.1369 | 0.623 | 6.7515 | 4.6531 | 0.886 | 0.0282 | 12.5107 | 0.4337 | 0.6092 | UNAVAILABLE |
| E2 | V3 | 3.1339 | 0.5672 | 6.7548 | 4.082 | 0.8016 | 0.0339 | 12.4798 | 0.4487 | 0.5962 | UNAVAILABLE |
| E2 | V4 | 2.703 | 0.623 | 5.8601 | 3.9116 | 0.7982 | 0.0254 | 10.6533 | 0.435 | 0.5338 | UNAVAILABLE |
| E2 | V4all | 1.8147 | 0.7837 | 3.8121 | 3.7399 | 0.7519 | 0.0216 | 6.9745 | 0.3737 | 0.5525 | UNAVAILABLE |

Amendment 6 family guard, each arm vs the shipped pick (A0, E3b): {"A0_E2": {"pass": false, "fails": ["progress>=0.90", "speedMAE<=+0.30", "cross<=+0.10", "heading<=+0.5", "goal<=+1.0"]}, "V3_E2": {"pass": false, "fails": ["progress>=0.90", "speedMAE<=+0.30", "cross<=+0.10", "heading<=+0.5", "goal<=+1.0"]}, "V4_E2": {"pass": false, "fails": ["progress>=0.90", "speedMAE<=+0.30", "cross<=+0.10", "goal<=+1.0"]}, "V4all_E2": {"pass": false, "fails": ["progress>=0.90", "speedMAE<=+0.30", "cross<=+0.10", "goal<=+1.0"]}, "V3_E3b": {"pass": true, "fails": []}, "V4_E3b": {"pass": true, "fails": []}, "V4all_E3b": {"pass": true, "fails": []}}. Control: A0's rebuilt pick vs the landed seam max |d| = 0.0 m.

## E4 (reported): predicted NC / TTC / EP probability vs the harness means

Harness: STOP [0.97, 0.965, 0.3059], copies [0.9663, 0.9287, 0.6741].

| model | STOP predicted NC / TTC / EP | copies predicted NC / TTC / EP |
|---|---|---|
| A0 | [0.7759, 0.7576, 0.2638] | [0.8687, 0.6423, 0.7779] |
| V3_s0 | [0.8603, 0.7513, 0.3571] | [0.8065, 0.5364, 0.8883] |
| V3_s1 | [0.86, 0.7789, 0.2894] | [0.8302, 0.5818, 0.8501] |
| V3_s2 | [0.8669, 0.8065, 0.2868] | [0.8578, 0.6339, 0.8263] |
| V4_s0 | [0.9488, 0.942, 0.0989] | [0.9085, 0.63, 0.827] |
| V4_s1 | [0.9634, 0.9578, 0.0469] | [0.9237, 0.7073, 0.7492] |
| V4_s2 | [0.9651, 0.9677, 0.047] | [0.9339, 0.748, 0.7348] |

## Fine-tune record and navtrain-val BCE per component (reported)

* seed 0: device cuda, bf16 autocast True, lr 0.00010322678309765115, wd 0.01, score weight 0.1, batch 16, epochs 4, N 1600, steps 400, 298.6 s; batch-order sha ['b38eafa4eb532cdf', '6faca4c63c9a6bb1', 'd090d59ea8bb32f9', '3f34c363b8871241']
  * epoch 0: V3 pure [0.37559, 0.46327, 0.11541, 0.5154, 0.42284, 0.08241] copies [0.17752, 0.43385, 0.69861, 0.54, 0.53467, 0.05975]; V4 pure [0.37617, 0.46853, 0.11572, 0.51374, 0.42519, 0.08479] copies [0.0964, 0.21175, 0.51332, 0.36679, 0.09941, 0.05741]; V4all pure [0.3876, 0.4714, 0.12284, 0.52774, 0.41752, 0.08564] copies [0.09267, 0.21167, 0.50379, 0.36187, 0.06715, 0.05724]
  * epoch 1: V3 pure [0.3792, 0.4654, 0.11554, 0.5111, 0.40849, 0.08628] copies [0.21073, 0.52074, 0.69427, 0.62415, 0.57196, 0.06583]; V4 pure [0.37765, 0.46801, 0.11534, 0.51092, 0.40947, 0.09088] copies [0.08795, 0.20727, 0.51208, 0.35851, 0.08396, 0.05701]; V4all pure [0.37672, 0.46765, 0.12468, 0.52019, 0.41322, 0.08976] copies [0.0862, 0.20581, 0.50144, 0.34037, 0.06324, 0.06129]
  * epoch 2: V3 pure [0.37472, 0.44964, 0.11485, 0.51567, 0.40769, 0.07964] copies [0.17535, 0.53801, 0.67862, 0.56632, 0.53221, 0.06312]; V4 pure [0.37414, 0.45412, 0.11485, 0.51591, 0.42102, 0.08041] copies [0.09961, 0.20172, 0.50607, 0.35347, 0.07155, 0.05405]; V4all pure [0.38742, 0.45518, 0.12431, 0.52172, 0.4078, 0.08243] copies [0.09298, 0.19915, 0.49713, 0.34067, 0.0453, 0.0549]
  * epoch 3: V3 pure [0.38003, 0.45146, 0.12033, 0.51758, 0.40876, 0.08187] copies [0.19993, 0.49411, 0.73268, 0.70357, 0.36947, 0.06788]; V4 pure [0.38632, 0.45739, 0.12111, 0.52623, 0.40794, 0.08463] copies [0.10039, 0.19369, 0.54217, 0.40513, 0.04723, 0.05887]; V4all pure [0.3791, 0.46425, 0.12199, 0.51106, 0.42814, 0.08211] copies [0.09083, 0.19072, 0.51917, 0.35405, 0.03521, 0.05831]
* seed 1: device cuda, bf16 autocast True, lr 0.00010322678309765115, wd 0.01, score weight 0.1, batch 16, epochs 4, N 1600, steps 400, 267.1 s; batch-order sha ['560457a58f4ff403', '2c71926d388c7f77', '3562883fb2489688', '88f2511397b3d046']
  * epoch 0: V3 pure [0.3773, 0.46251, 0.11538, 0.5213, 0.42489, 0.08627] copies [0.18373, 0.38761, 0.71093, 0.5675, 0.46164, 0.0614]; V4 pure [0.37488, 0.46867, 0.11537, 0.5214, 0.42764, 0.08965] copies [0.10255, 0.21596, 0.52255, 0.38745, 0.10653, 0.05872]; V4all pure [0.38574, 0.48177, 0.1228, 0.54214, 0.42682, 0.08742] copies [0.08764, 0.20864, 0.51039, 0.36304, 0.06109, 0.05576]
  * epoch 1: V3 pure [0.36996, 0.45758, 0.11764, 0.50863, 0.41432, 0.08735] copies [0.20413, 0.44157, 0.76731, 0.65199, 0.44111, 0.06495]; V4 pure [0.36771, 0.46189, 0.11593, 0.50791, 0.41433, 0.09213] copies [0.09989, 0.21747, 0.5215, 0.39485, 0.06647, 0.05682]; V4all pure [0.37614, 0.46904, 0.12028, 0.51592, 0.4183, 0.09055] copies [0.09344, 0.20479, 0.50853, 0.36936, 0.04775, 0.05893]
  * epoch 2: V3 pure [0.37432, 0.45576, 0.11597, 0.51528, 0.40229, 0.07786] copies [0.17717, 0.50221, 0.72015, 0.5639, 0.47002, 0.0623]; V4 pure [0.37541, 0.46063, 0.11547, 0.51306, 0.41051, 0.0791] copies [0.08518, 0.20305, 0.51051, 0.36056, 0.05672, 0.05186]; V4all pure [0.38048, 0.45727, 0.12029, 0.51881, 0.40815, 0.08142] copies [0.08216, 0.20058, 0.50373, 0.347, 0.03998, 0.05305]
  * epoch 3: V3 pure [0.3802, 0.45972, 0.11604, 0.51646, 0.39268, 0.07924] copies [0.16928, 0.44344, 0.68918, 0.58243, 0.50329, 0.06271]; V4 pure [0.38085, 0.47043, 0.11467, 0.51479, 0.40867, 0.08351] copies [0.0902, 0.19777, 0.50579, 0.35973, 0.05311, 0.05515]; V4all pure [0.39813, 0.47745, 0.12389, 0.52228, 0.39865, 0.08192] copies [0.08875, 0.19247, 0.49793, 0.34365, 0.03966, 0.05267]
* seed 2: device cuda, bf16 autocast True, lr 0.00010322678309765115, wd 0.01, score weight 0.1, batch 16, epochs 4, N 1600, steps 400, 264.2 s; batch-order sha ['3880f1683c5d3941', 'ee01e111752bb1e3', '6afa23adfca6cbcc', 'a184a458222d7a5c']
  * epoch 0: V3 pure [0.3795, 0.46091, 0.11488, 0.51832, 0.416, 0.08293] copies [0.16317, 0.41954, 0.70554, 0.52834, 0.54685, 0.06007]; V4 pure [0.38016, 0.466, 0.11531, 0.51745, 0.42191, 0.08481] copies [0.09671, 0.21659, 0.51552, 0.36316, 0.11708, 0.05564]; V4all pure [0.39451, 0.47185, 0.11974, 0.52585, 0.41696, 0.08528] copies [0.09319, 0.21508, 0.50841, 0.35692, 0.06506, 0.05584]
  * epoch 1: V3 pure [0.37933, 0.47356, 0.11773, 0.51744, 0.41163, 0.07903] copies [0.21418, 0.5046, 0.75381, 0.67476, 0.53982, 0.06173]; V4 pure [0.37554, 0.46986, 0.11591, 0.51363, 0.41136, 0.08269] copies [0.0898, 0.22409, 0.51622, 0.35515, 0.09456, 0.05409]; V4all pure [0.37911, 0.47276, 0.1213, 0.5145, 0.41647, 0.08204] copies [0.08033, 0.21191, 0.50682, 0.34138, 0.06361, 0.05385]
  * epoch 2: V3 pure [0.38102, 0.46789, 0.11906, 0.51263, 0.41806, 0.07962] copies [0.18787, 0.56372, 0.73023, 0.56741, 0.61529, 0.06159]; V4 pure [0.3782, 0.46557, 0.11827, 0.5127, 0.42924, 0.08229] copies [0.08406, 0.2175, 0.52098, 0.35042, 0.08107, 0.0521]; V4all pure [0.38614, 0.46359, 0.12299, 0.51466, 0.41018, 0.0818] copies [0.08396, 0.21801, 0.51196, 0.33824, 0.06339, 0.05343]
  * epoch 3: V3 pure [0.37736, 0.45157, 0.11607, 0.50884, 0.40206, 0.07955] copies [0.15756, 0.479, 0.63747, 0.52481, 0.56985, 0.06387]; V4 pure [0.37783, 0.45323, 0.11524, 0.5062, 0.41278, 0.08261] copies [0.08149, 0.20194, 0.5104, 0.33939, 0.0604, 0.05531]; V4all pure [0.38418, 0.45612, 0.12784, 0.52126, 0.41273, 0.08417] copies [0.08047, 0.20219, 0.49915, 0.32975, 0.04307, 0.0565]

## Data

Labels (`D:/Projects/TanitAD/data/refe_m5/label_summary.json`): {"driver": "eval/m5_label.py (16 workers) then eval/m5_label_finish.py (kept m5w0,m5w1,m5w2,m5w3,m5w4,m5w5,m5w6,m5w7,m5w8,m5w9)", "retired": ["m5w10", "m5w11", "m5w12", "m5w13", "m5w14", "m5w15"], "seconds_finisher": 1030.0, "failed": 0, "selfcheck_failed": 0, "skipped_ndiff": 0, "status_note": "status files count per worker PROCESS lifetime (a restarted worker restarts its counters); coverage below is from the files themselves", "keys_in_chunks": 1900, "keys_labelled": 1900, "set_lines": 1983, "keys_unlabelled": 0}. Fine-tune data (`ft_data.json`): {"train": {"cached": 1600, "usable": 1600, "carrying_at_frac_0.5": 803, "excluded": {"no_set": 0, "no_aux": 0, "no_props": 0, "bad_spec": 0, "kept_mismatch": 0}}, "val": {"cached": 300, "usable": 300, "carrying_at_frac_0.5": 156, "excluded": {"no_set": 0, "no_aux": 0, "no_props": 0, "bad_spec": 0, "kept_mismatch": 0}}}.

## POST HOC, NOT GATING: the same metrics against the REPAIRED harness truth (Amendment 7)

| metric | A0 | V3 | V4 | V4all | V4 - V3 [95 %] | V3 - A0 [95 %] |
|---|---|---|---|---|---|---|
| E1_masked | 0.59719 | 0.60245 | 0.56369 | 0.565 | -0.0388 [-0.0791, -0.0009] (separated) | +0.0053 [-0.0191, +0.0304] |
| E3a | 0.63092 | 0.62958 | 0.63117 | 0.6329 | +0.0016 [-0.0043, +0.0083] | -0.0013 [-0.0216, +0.0179] |
| E3b | 77.93812 | 81.74953 | 82.5531 | 80.21939 | +0.8036 [-0.6564, +2.3633] | +3.8114 [+0.5034, +7.3596] (separated) |
| E2 | 72.56026 | 74.82041 | 77.90858 | 79.85711 | +3.0882 [+1.0369, +5.4244] (separated) | +2.2601 [-0.7270, +5.0793] |

## GPU stage record (`gpu_chain_times.json`)

```
{
 "m6_gpu_done_seen": "2026-09-27T23:56:29",
 "G1_timing_vram_at_start": {
  "free": 5845,
  "used": 2113,
  "cap": 4737,
  "need": 2600
 },
 "G1_timing_start": "2026-09-27T23:56:29",
 "G1_timing_end": "2026-09-27T23:57:35",
 "G1_timing_minutes": 1.09,
 "G1_timing_rc": 0,
 "G1_timing_peak_vram_used_total_mib": 5423,
 "rate_s_per_sample": 0.527,
 "gpu_peak_gb": 2.29,
 "load_min": 0.91,
 "G2_navtrain_train_vram_at_start": {
  "free": 5845,
  "used": 2113,
  "cap": 4737,
  "need": 3146
 },
 "G2_navtrain_train_start": "2026-09-27T23:57:35",
 "G2_navtrain_train_end": "2026-09-28T00:22:25",
 "G2_navtrain_train_minutes": 24.84,
 "G2_navtrain_train_rc": 0,
 "G2_navtrain_train_peak_vram_used_total_mib": 5637,
 "rate_s_per_sample_train_median_shard": 0.51,
 "G3_navtest_vram_at_start": {
  "free": 5659,
  "used": 2299,
  "cap": 4551,
  "need": 3146
 },
 "G3_navtest_start": "2026-09-28T00:22:26",
 "G3_navtest_end": "2026-09-28T00:34:56",
 "G3_navtest_minutes": 12.51,
 "G3_navtest_rc": 0,
 "G3_navtest_peak_vram_used_total_mib": 4341,
 "G4_navtrain_val_vram_at_start": {
  "free": 5663,
  "used": 2295,
  "cap": 4555,
  "need": 3146
 },
 "G4_navtrain_val_start": "2026-09-28T00:34:57",
 "G4_navtrain_val_end": "2026-09-28T00:41:51",
 "G4_navtrain_val_minutes": 6.91,
 "G4_navtrain_val_rc": 0,
 "G4_navtrain_val_peak_vram_used_total_mib": 5606,
 "label_wait_start": "2026-09-28T00:41:51",
 "label_wait_end": "2026-09-28T00:57:52",
 "C_data_cpu_start": "2026-09-28T00:57:52",
 "C_data_cpu_end": "2026-09-28T00:58:22",
 "C_data_cpu_minutes": 0.5,
 "C_data_cpu_rc": 0,
 "C_data_cpu_peak_vram_used_total_mib": 5862,
 "G5_finetune_vram_at_start": {
  "free": 6012,
  "used": 1946,
  "cap": 4904,
  "need": 1800
 },
 "G5_finetune_start": "2026-09-28T01:27:39",
 "G5_finetune_end": "2026-09-28T01:45:48",
 "G5_finetune_minutes": 18.14,
 "G5_finetune_rc": 0,
 "G5_finetune_peak_vram_used_total_mib": 4117,
 "marker_created": "2026-09-28T01:08:38",
 "sequenced_start": "2026-09-28T01:27:39",
 "G6_eval_vram_at_start": {
  "free": 6014,
  "used": 1944,
  "cap": 4906,
  "need": 1200
 },
 "G6_eval_start": "2026-09-28T01:45:48",
 "G6_eval_end": "2026-09-28T01:47:48",
 "G6_eval_minutes": 2.01,
 "G6_eval_rc": 0,
 "G6_eval_peak_vram_used_total_mib": 3429,
 "marker_created:M5_GPU_DONE_2": "2026-09-28T01:47:48"
}
```


<!-- HAND-WRITTEN BELOW THIS LINE (kept on re-run) -->
## Interpretation (written by hand; every number cites its artifact under `eval/raw/m5_effectiveness/`)

### 1. The registered verdict stands: ADOPT, on the registered truth
All three criteria of §5 are met on the registered truth: the E-6 table of snapshot 015 plus the harness-scored copies and
STOP (`readout.json`).
* E1 V4 - V3 = +0.1534 [+0.1037, +0.2104]. That is 3.8x the seed floor of 0.0403.
* E3a is non-inferior: -0.0018 [-0.0081, +0.0048].
* E3b is non-inferior: +1.25 [-0.22, +2.84] PDMS.

Every gate passed. A0 re-run on the new cache also reproduces an independent reference built from a different code path
(`a0_reference.json`: the slow-copy probe's GPU dump and a loop re-implementation of the concordance). The values agree
to rounding: E1 0.6500, E3a 0.6469, E3b 61.18, E2 70.08.

The interval answers EPISODES. The seed floor answers TRAINING (batch order, 3 seeds per arm). Inference is
deterministic.

### 2. EP and the four families: what M5 does to the plan the planner actually ships (E3b = the 64 alone)
**EP (ego progress) on the shipped route does not move.**
* V4 - V3 = +0.90 [-0.38, +2.25] x100, against a seed floor of 2.18.
* V4 - A0 = +1.28 [-1.89, +4.35].

M5 therefore neither causes nor repairs the -5.74 EP of the 015 -> 016 like-for-like. That EP loss lives in the proposal
set and trunk, which this scorer-only proxy holds fixed at 015.

On the E2 route (the 64 + extras at test time, NOT what M5 deploys), V4 - V3 EP = +8.84 [+5.63, +12.27]. That gain comes
from V4 picking STOP less often (V3 78-85 of 200 tokens, V4 55-75) and 0.75x copies more (V3 29-41, V4 71-98)
(`families.json` E2_selection_mix).

Four families (`families6.py` blocks vs the logged human future, seed means, E3b route), V4 vs V3:

| family | measure | V4 | V3 |
|---|---|---|---|
| longitudinal | speed MAE | 1.053 m/s | 1.037 m/s |
| longitudinal | progress ratio | 1.037 | 1.033 |
| lateral | heading MAE | 3.44 deg | 3.38 deg |
| lateral | cross-track MAE | 0.572 m | 0.567 m |
| tactical | goal-point error | 4.23 m | 4.16 m |
| tactical | longitudinal / lateral decision kappa | 0.397 / 0.839 | 0.391 / 0.833 |
| strategic | -- | UNAVAILABLE | UNAVAILABLE |

Strategic is UNAVAILABLE by design: NAVSIM has no route-decision label, and a scorer fine-tune does not set the route.

* Amendment 6's family guard vs the shipped pick passes for V3, V4 and V4all on E3b.
* Every arm fails it on E2, as the stop-candidate probe already found for test-time slow candidates.
* These family blocks are per-model with families6's own intervals. They are NOT paired intervals.

### 3. POST HOC, NOT GATING: against the DEPLOYED harness truth, the E1 effect is gone
The registered truth predates SPEC Amendment 7 (ADOPTED 2026-09-27). Under Amendment 7 the executed plan takes
heading[18] at the last native pose. In that truth, every original proposal carries the corrupted t = 4.0 s heading,
while a 0.75x copy never reaches that pose.

To re-read against the deployed truth, the repaired E-6 table of the 64 (like_for_like_016, all gates pass) was combined
with the 8 repaired copies, which were scored tonight by the unchanged harness, 8/8 PASS (`repaired_extras.json`). Its
controls:
* the unrepaired rebuild of every copy equals build.npz bit for bit;
* the repair changes only the 4.0 s heading;
* the repair is the identity on STOP.

**Result (`families.json` posthoc_repaired_truth; same draws, same pairs):**
* **E1 V4 - V3 = -0.0388 [-0.0791, -0.0009]**, against a seed floor of 0.0517 on this truth.
  * The +0.1534 of the registered truth does not survive.
  * The sign reverses, but the magnitude is inside the rig's own training noise, so by H-ESTIM-SEED-1 it is no effect
    in either direction.
* E3a: +0.0016 [-0.0043, +0.0083].
* E3b: +0.80 [-0.66, +2.36] PDMS (floor 3.79).
* E2: +3.09 [+1.04, +5.42] (floor 1.26). This is the only survivor. It comes from preferring copies over STOP (STOP
  scores 62.58 repaired, copies 78.58).

**Why (`a0_reference.json`, `label_mechanism.json`):**
* The truth itself. Under the registered truth, a 0.75x copy beats its own source original by +21.01 PDMS (better on
  52.7 % of 1,600 pairs). Under the repaired truth the margin is +1.43, and the copy is WORSE than its source on 60.8 %.
* The labels. Over all 15,200 (copy, source) pairs in the 1,900 registered v4 sets, the trainer's comfort target is
  +0.446 higher for the copy overall.
  * It is never lower.
  * Where the source's last heading is > 0.5 rad off its path tangent (64.8 % of sources, median defect 0.99 rad), the
    copy's comfort target is +0.686 higher.
  * Where the source's heading is sound, it is +0.006 higher.
  * The other components move as a slower plan should: EP -0.199 (the copy is lower on 93.4 %), DAC +0.232,
    TTC +0.229, NC +0.064.
* ⇒ The v4 labels teach "a slowed copy is more comfortable" almost entirely through the heading defect, which the
  deployed planner now repairs after selection (Amendment 7) and which M6 targets at the source.

**Classification: a REFUTATION-IN-WAITING, not a refutation.** The registered test PASSED. The post-hoc re-read is not
pre-registered and uses the same 200 tokens, so it cannot overturn the verdict. It does remove the case for deploying.

### 4. Recommendation and next lever
**HOLD the §8 live deployment (the PI's go).** Effectiveness is proven against the truth the pre-registration fixed. It
is NOT established against the evaluation pipeline as it now runs, with the repair ON. The coordinator made the same
recommendation to the PI.

* **Re-test (drafted, NOT registered, NOT run):** `eval/PREREG_MEASURE5_AMENDMENT1_DRAFT.md`. It re-tests the SAME
  trained V4 vs V3 models with the REPAIRED truth as the reference, on Amendment 5's 923 fresh tokens (43 logs,
  disjoint from W3's 200), with both outcomes committed.
* **Next lever (named, NOT run): label the REPAIRED plans.** The v4 labeller would apply `refe/planner.py
  repair_last_heading` to every original and every copy before the teacher and NAVSIM-comfort labelling. That labels
  the executed plan, the pattern that made DAC and comfort agree 100 % with the harness. It is cheap once decided:
  * the trunk cache and the 1,900 props chunks are reusable as they are;
  * relabelling is about 45-60 min of CPU at 10 workers;
  * the fine-tune is 18 min of GPU (MEASURED tonight) and the eval 2 min.
  It needs its own validity run (comfort on repaired plans vs the harness on repaired seams, bar 100 %) and a
  pre-registration.
* **Blocked on:** a PI decision (a labeller change inside measure 5), then a GPU slot of about 20 min.

## Declared departures and incidents
1. **Learning rate 1.0323e-4, not the 6.9e-5 printed in §3d.** §3d says the exact value is read from the pod's
   metrics.jsonl before the run and recorded. The recorded value is 1.0322678e-4 at step 4933, epoch 15
   (`pod_status_1332.json`). The 6.9e-5 was the prereg's own formula estimate.
2. **Numerics.**
   * Forward: bf16 autocast + TF32, batch 4, as registered. The declared batch-2 fallback was NOT triggered.
   * Fine-tune: GPU bf16 autocast, the trainer's numerics.
   * Eval: fp32, the planner's numerics.
   * Memory controls (the allocator cap via `set_per_process_memory_fraction`, and a VRAM reservation at process start)
     change no arithmetic. No cap was reached: peak total VRAM was 4,117 MiB in the fine-tune and 5,637 MiB in the forwards.
3. **Incident 1 (01:08:38): G5 attempt 1 killed by the VRAM watchdog.**
   * An M6b arm passed its gate during G5's 301-s CPU preload, when G5 held almost no VRAM.
   * When G5's steps began, total VRAM stayed above the 7,300 MiB ceiling for 20 s (peak 7,370 MiB).
   * The chain's finally wrote `M5_GPU_DONE` as specified.
4. **Incident 2 (01:14:50): G5 attempt 2 stopped by hand** during its preload, before any training step. M6b's Wt2 and
   decode_T0 had passed their gates while still ramping, 38 s before G5's gate read 2,557 MiB used.
5. **Nothing from either attempt is reused.** `ft/` did not exist when attempt 3 ran, and attempt 3 ran with `--force`.
   Attempt 3 was SEQUENCED behind M6b's `ZZ_M6B_GPU_DONE_ZZ` token (coordinator, 01:18). It reserved 1,800 MiB at
   process start and completed. `M5_GPU_DONE_2` was written at 01:47:48.
6. **Labelling.**
   * The coordinator's RAM rule applied: 16 workers held 14.6 GB and left 3.1 GB free.
   * At 00:40, 6 workers were retired by explicit PID and their 6 claimed chunks returned to the queue. A finisher
     (`m5_label_finish.py`) supervised the remaining 10.
   * One worker (m5w12) had exited rc 1 once and was restarted. The driver's restart overwrote its log, so the cause is
     UNVERIFIED.
   * Coverage: 1,900 / 1,900 keys labelled, 0 failed, 0 selfcheck_failed. There are 1,983 set lines, because re-claimed
     chunks produced duplicates; the data stage keeps one line per key.
   * `m5_finetune_eval.py data` found 1,600 / 1,600 train and 300 / 300 val sets usable, with 0 kept-slot mismatches.
7. **The window.** The PI withdrew the 23:10 cap at ~20:42 ("fixes first", 04:30 limit).
   * M6's chain held its GPU stages behind its own old 23:15 pause, deadlocking against M5 from 23:12 to 23:36. The
     coordinator resolved it.
   * M5's window opened at 23:56:25.

## GPU stage timings (MEASURED, `gpu_chain.log`)

| stage | start -> end | minutes | result |
|---|---|---|---|
| G1 timing, 20 samples | 23:56:29 -> 23:57:35 | 1.1 | 0.527 s/sample, torch peak 2.29 GB |
| G2 navtrain train, 1,600 | 23:57:35 -> 00:22:25 | 24.8 | 25/25 shard gates PASS (re-run max abs 0.0; swapped-context control min 2.398) |
| G3 navtest, 200 | 00:22:26 -> 00:34:56 | 12.5 | props == STOP dump and logits == E-6 table, both 0.0 |
| G4 navtrain val, 300 | 00:34:57 -> 00:41:51 | 6.9 | 5/5 gates PASS |
| (label wait, GPU idle; coordinator's decision) | 00:41:51 -> 00:57:52 | 16.0 | -- |
| G5 attempt 1 | 01:02:22 -> 01:08:38 | 6.3 | killed (VRAM watchdog) |
| G5 attempt 2 | 01:13:32 -> 01:14:50 | 1.3 | stopped by hand in preload |
| G5 fine-tune (3 arms x 3 seeds) | 01:27:39 -> 01:45:48 | 18.1 | 400 steps/seed, 264-299 s/seed |
| G6 eval, 10 models | 01:45:48 -> 01:47:48 | 2.0 | all gates PASS |

**Nothing remains for another GPU window.** The drafted amendment and the next lever each need their own decision first.
