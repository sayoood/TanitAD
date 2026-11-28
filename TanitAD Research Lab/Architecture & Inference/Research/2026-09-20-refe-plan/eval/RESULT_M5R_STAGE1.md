# RESULT -- Measure 5r (v4r: the M5 labels on the Amendment-7-repaired plans), STAGE 1

*Pre-registration `eval/PREREG_MEASURE5R.md`, blob `a8241136` (registered by the coordinator 2026-09-28, before any v4r label). Applied as written. Numbers are written by `eval/m5r_write_result.py` from `eval/raw/m5r/`; none is typed.*

## Stage-1 verdict (as registered): **STAGE-1 PASS**

Registered consequence: Stage 2 runs next: the same four bars on Amendment 5's 923 fresh tokens (43 logs), repaired truth (73 new harness runs), M6's eval cache for the scorer context after its admissibility gate. ADOPT is decided ONLY there; an ADOPT then leads to a DRAFTED pod kit, shipped only on the PI's go.

| criterion (§6, stage 1) | bar | measured (V4r - V3) | met |
|---|---|---|---|
| E1 slow-plan pair concordance (masked) | >= +0.03, CI lo > 0, > seed floor 0.02619 | +0.0591 [+0.0112, +0.1040] (separated) | yes |
| E3a pair concordance among the 64 | CI lo > -0.01 | +0.0246 [+0.0055, +0.0445] (separated) | yes |
| E3b PDMS of the pick among the 64 | CI lo > -2.0 | +1.98 [-1.81, +6.61] | yes |
| E2 PDMS of the pick, 64 + extras | CI lo > -2.0 | +7.98 [+4.16, +11.75] (separated) | yes |
| REFUTED iff E1 CI hi < 0 | -- | +0.1040 | no |

Estimator: paired log-cluster bootstrap over the logs, 10,000 resamples, 95 % percentile, seed 20260927; answers EPISODES for the CI; TRAINING (batch order) by the seed floor; inference deterministic; n = 200 tokens, 93 logs; estimator self-test {'reference_ci': [-2.43, 6.56], 'ours': [-2.43, 6.56], 'identical': True}.

## Gates

| gate | pass | artifact |
|---|---|---|
| G-R1 | True | `eval/raw/m5r/selftest_repair.json` |
| G-R2 | True | `eval/raw/m5r/gate_gr2.json` |
| G-R3 | True | `eval/raw/m5r/gate_gr34.json` |
| G-R4 | True | `eval/raw/m5r/gate_gr34.json` |
| G-R5 | True | `eval/raw/m5r/gate_gr5.json` |
| G-R6 | True | `eval/raw/m5r/gate_gr6_eval.json` |

G-R3 NAVSIM agreement on repaired plans: DAC 1.0, comfort 1.0 over 14600 slots. G-R4 collision (copies vs originals): {"copies_agree": 0.971875, "originals_agree": 0.967656, "copies_label_collision_harness_no": 19, "copies_label_no_harness_collision": 26, "one_directional": false, "bar": "copies agreement >= originals' AND >= 0.90 AND neither direction > 3x the other", "pass": true}. Label-ranking ceiling on the extras: {"tokens": 200, "mean": 0.8576393538983789}.

## Arm token means

| model | E1 masked | E1 plain | E3a | E3b | E2 |
|---|---|---|---|---|---|
| A0 | 0.5972 | 0.5646 | 0.6309 | 77.94 | 72.56 |
| V3_s0 | 0.5862 | 0.5538 | 0.6230 | 82.12 | 74.81 |
| V3_s1 | 0.6124 | 0.5775 | 0.6400 | 80.68 | 74.76 |
| V3_s2 | 0.6087 | 0.5595 | 0.6257 | 82.45 | 74.90 |
| V3r_s0 | 0.6756 | 0.6175 | 0.6497 | 84.66 | 84.66 |
| V3r_s1 | 0.6829 | 0.6417 | 0.6415 | 83.52 | 81.96 |
| V3r_s2 | 0.6824 | 0.6313 | 0.6488 | 83.92 | 81.57 |
| V4_s0 | 0.5400 | 0.5122 | 0.6240 | 84.43 | 77.10 |
| V4_s1 | 0.5918 | 0.5479 | 0.6442 | 80.64 | 78.26 |
| V4_s2 | 0.5593 | 0.5328 | 0.6253 | 82.59 | 78.36 |
| V4r_s0 | 0.6522 | 0.6488 | 0.6544 | 83.09 | 81.50 |
| V4r_s1 | 0.6645 | 0.6533 | 0.6522 | 83.83 | 82.88 |
| V4r_s2 | 0.6680 | 0.6475 | 0.6560 | 84.27 | 84.02 |

## All paired contrasts (reported; only V4r - V3 gates)

| metric | pair | diff [95 %] |
|---|---|---|
| E1_masked | V4r_minus_V3 | +0.0591 [+0.0112, +0.1040] (separated) |
| E1_masked | V3r_minus_V3 | +0.0778 [+0.0488, +0.1069] (separated) |
| E1_masked | V4r_minus_V3r | -0.0188 [-0.0508, +0.0093] |
| E1_masked | V4r_minus_V4 | +0.0979 [+0.0699, +0.1265] (separated) |
| E1_masked | V3_minus_A0 | +0.0053 [-0.0191, +0.0304] |
| E1_masked | V4r_minus_A0 | +0.0644 [+0.0192, +0.1075] (separated) |
| E1_plain | V4r_minus_V3 | +0.0863 [+0.0387, +0.1308] (separated) |
| E1_plain | V3r_minus_V3 | +0.0665 [+0.0414, +0.0920] (separated) |
| E1_plain | V4r_minus_V3r | +0.0197 [-0.0153, +0.0521] |
| E1_plain | V4r_minus_V4 | +0.1189 [+0.0875, +0.1522] (separated) |
| E1_plain | V3_minus_A0 | -0.0010 [-0.0239, +0.0216] |
| E1_plain | V4r_minus_A0 | +0.0853 [+0.0401, +0.1295] (separated) |
| E3a | V4r_minus_V3 | +0.0246 [+0.0055, +0.0445] (separated) |
| E3a | V3r_minus_V3 | +0.0171 [-0.0032, +0.0379] |
| E3a | V4r_minus_V3r | +0.0075 [+0.0023, +0.0132] (separated) |
| E3a | V4r_minus_V4 | +0.0231 [+0.0022, +0.0438] (separated) |
| E3a | V3_minus_A0 | -0.0013 [-0.0216, +0.0179] |
| E3a | V4r_minus_A0 | +0.0233 [-0.0079, +0.0537] |
| E3b | V4r_minus_V3 | +1.98 [-1.81, +6.61] |
| E3b | V3r_minus_V3 | +2.28 [-1.68, +6.88] |
| E3b | V4r_minus_V3r | -0.30 [-1.98, +1.26] |
| E3b | V4r_minus_V4 | +1.18 [-3.13, +5.92] |
| E3b | V3_minus_A0 | +3.81 [+0.50, +7.36] (separated) |
| E3b | V4r_minus_A0 | +5.79 [+1.03, +10.60] (separated) |
| E2 | V4r_minus_V3 | +7.98 [+4.16, +11.75] (separated) |
| E2 | V3r_minus_V3 | +7.91 [+4.74, +11.10] (separated) |
| E2 | V4r_minus_V3r | +0.07 [-2.24, +2.31] |
| E2 | V4r_minus_V4 | +4.89 [+0.77, +8.74] (separated) |
| E2 | V3_minus_A0 | +2.26 [-0.73, +5.08] |
| E2 | V4r_minus_A0 | +10.24 [+5.61, +14.73] (separated) |

Seed floor (E1): {"floor": 0.02619, "by_arm": {"V3": {"seed_token_means": [0.58621, 0.61241, 0.60874], "max_pair_abs": 0.02619}, "V4r": {"seed_token_means": [0.65216, 0.66446, 0.66799], "max_pair_abs": 0.01582}}}

## EP and the NAVSIM sub-scores of the picks (x100; order NC, DAC, EP, TTC, C, DDC)

* A0: E3b [92.25, 92.0, 72.051, 88.0, 100.0, 91.75]; E2 [95.5, 95.0, 57.245, 92.5, 78.5, 96.25]
* V3: E3b [95.0, 93.5, 74.266, 92.0, 100.0, 94.25]; E2 [97.083, 96.833, 57.502, 95.667, 78.0, 99.333]
* V4: E3b [95.667, 94.0, 75.727, 91.5, 100.0, 93.917]; E2 [98.0, 97.0, 62.114, 96.0, 83.667, 99.0]
* V3r: E3b [97.833, 94.0, 78.442, 92.167, 100.0, 95.833]; E2 [98.167, 93.333, 75.786, 93.167, 98.333, 96.75]
* V4r: E3b [97.667, 93.167, 78.05, 93.0, 100.0, 95.75]; E2 [97.333, 92.667, 76.112, 93.5, 99.667, 95.833]

| route | pair | EP [95 %] | NC | DAC | TTC | C |
|---|---|---|---|---|---|---|
| E3b | V4r_minus_V3 | +3.79 [-0.09, +8.38] | +2.67 [+0.87, +5.11] | -0.33 [-3.87, +4.08] | +1.00 [-1.21, +3.32] | +0.00 [+0.00, +0.00] |
| E3b | V3r_minus_V3 | +4.18 [+0.33, +8.52] | +2.83 [+0.98, +5.16] | +0.50 [-3.10, +4.95] | +0.17 [-2.58, +2.94] | +0.00 [+0.00, +0.00] |
| E3b | V4r_minus_V3r | -0.39 [-1.93, +1.05] | -0.17 [-1.03, +0.71] | -0.83 [-2.48, +0.72] | +0.83 [-0.91, +2.61] | +0.00 [+0.00, +0.00] |
| E3b | V4r_minus_V4 | +2.32 [-1.90, +6.86] | +2.00 [-0.30, +4.92] | -0.83 [-4.55, +3.50] | +1.50 [-1.32, +4.59] | +0.00 [+0.00, +0.00] |
| E3b | V3_minus_A0 | +2.21 [-0.88, +5.52] | +2.75 [+0.19, +5.66] | +1.50 [-0.96, +4.31] | +4.00 [+0.63, +7.93] | +0.00 [+0.00, +0.00] |
| E3b | V4r_minus_A0 | +6.00 [+1.20, +10.83] | +5.42 [+2.22, +9.10] | +1.17 [-2.89, +5.36] | +5.00 [+0.91, +9.52] | +0.00 [+0.00, +0.00] |
| E2 | V4r_minus_V3 | +18.61 [+14.27, +22.93] | +0.25 [-1.87, +2.24] | -4.17 [-7.62, -0.91] | -2.17 [-5.50, +0.70] | +21.67 [+15.12, +28.57] |
| E2 | V3r_minus_V3 | +18.28 [+14.72, +21.97] | +1.08 [-0.55, +2.86] | -3.50 [-6.25, -0.89] | -2.50 [-5.68, +0.33] | +20.33 [+14.06, +26.91] |
| E2 | V4r_minus_V3r | +0.33 [-1.83, +2.44] | -0.83 [-1.96, +0.17] | -0.67 [-3.23, +1.69] | +0.33 [-1.51, +2.34] | +1.33 [+0.16, +2.98] |
| E2 | V4r_minus_V4 | +14.00 [+9.78, +18.13] | -0.67 [-2.91, +1.28] | -4.33 [-8.48, -0.76] | -2.50 [-6.12, +0.78] | +16.00 [+10.34, +22.46] |
| E2 | V3_minus_A0 | +0.26 [-2.92, +3.33] | +1.58 [-0.50, +3.83] | +1.83 [-1.21, +5.07] | +3.17 [+0.43, +6.08] | -0.50 [-3.99, +3.16] |
| E2 | V4r_minus_A0 | +18.87 [+13.95, +23.79] | +1.83 [-0.68, +4.46] | -2.33 [-6.74, +1.65] | +1.00 [-2.67, +4.66] | +21.17 [+14.86, +28.14] |

E2 selection mix: {"A0": {"original": 73, "copy_075": 48, "stop": 79}, "V3_s0": {"original": 78, "copy_075": 37, "stop": 85}, "V3_s1": {"original": 93, "copy_075": 29, "stop": 78}, "V3_s2": {"original": 80, "copy_075": 41, "stop": 79}, "V3r_s0": {"original": 170, "copy_075": 27, "stop": 3}, "V3r_s1": {"original": 166, "copy_075": 32, "stop": 2}, "V3r_s2": {"original": 156, "copy_075": 39, "stop": 5}, "V4_s0": {"original": 53, "copy_075": 72, "stop": 75}, "V4_s1": {"original": 64, "copy_075": 71, "stop": 65}, "V4_s2": {"original": 47, "copy_075": 98, "stop": 55}, "V4r_s0": {"original": 160, "copy_075": 38, "stop": 2}, "V4r_s1": {"original": 152, "copy_075": 48, "stop": 0}, "V4r_s2": {"original": 159, "copy_075": 39, "stop": 2}}

## Four families (families6.py on the REPAIRED executed picks; per-model blocks, not paired)

| pick | speed MAE | progress ratio | heading MAE | cross MAE | goal-point err | long. kappa | lat. kappa | strategic |
|---|---|---|---|---|---|---|---|---|
| A0_E3b | 1.1749 | 1.0867 | 3.5372 | 0.61 | 4.7473 | 0.3848 | 0.81 | UNAVAILABLE |
| A0_E2 | 3.1369 | 0.623 | 4.6531 | 0.886 | 12.5107 | 0.4337 | 0.6092 | UNAVAILABLE |
| V3_s0_E3b | 1.0308 | 0.9918 | 3.3453 | 0.5578 | 4.1151 | 0.3867 | 0.8273 | UNAVAILABLE |
| V3_s0_E2 | 3.2465 | 0.5101 | 4.1953 | 0.8155 | 12.9241 | 0.4398 | 0.5749 | UNAVAILABLE |
| V3_s1_E3b | 1.0653 | 1.0678 | 3.4178 | 0.5862 | 4.2728 | 0.3928 | 0.8284 | UNAVAILABLE |
| V3_s1_E2 | 3.0456 | 0.6167 | 4.0565 | 0.8033 | 12.1264 | 0.4486 | 0.6312 | UNAVAILABLE |
| V3_s2_E3b | 1.0139 | 1.0405 | 3.3648 | 0.5567 | 4.0785 | 0.3934 | 0.8445 | UNAVAILABLE |
| V3_s2_E2 | 3.1095 | 0.5747 | 3.9941 | 0.786 | 12.3889 | 0.4577 | 0.5825 | UNAVAILABLE |
| V3r_s0_E3b | 0.9467 | 1.027 | 3.0555 | 0.4674 | 3.7396 | 0.4112 | 0.8435 | UNAVAILABLE |
| V3r_s0_E2 | 1.0562 | 0.9764 | 3.2237 | 0.4913 | 4.1275 | 0.4244 | 0.7994 | UNAVAILABLE |
| V3r_s1_E3b | 0.8693 | 1.1083 | 3.022 | 0.4526 | 3.4594 | 0.4117 | 0.8366 | UNAVAILABLE |
| V3r_s1_E2 | 0.9763 | 1.0557 | 3.1641 | 0.4634 | 3.8524 | 0.4396 | 0.7928 | UNAVAILABLE |
| V3r_s2_E3b | 0.86 | 1.0763 | 2.9765 | 0.4684 | 3.3905 | 0.4027 | 0.8609 | UNAVAILABLE |
| V3r_s2_E2 | 1.0864 | 1.0041 | 3.1626 | 0.4889 | 4.287 | 0.4307 | 0.7981 | UNAVAILABLE |
| V4_s0_E3b | 1.0442 | 0.9877 | 3.335 | 0.5523 | 4.1471 | 0.4097 | 0.8445 | UNAVAILABLE |
| V4_s0_E2 | 2.9515 | 0.5486 | 4.0195 | 0.8259 | 11.6483 | 0.4206 | 0.487 | UNAVAILABLE |
| V4_s1_E3b | 1.0994 | 1.085 | 3.5052 | 0.5968 | 4.4295 | 0.393 | 0.8364 | UNAVAILABLE |
| V4_s1_E2 | 2.5885 | 0.6534 | 3.946 | 0.7649 | 10.2142 | 0.376 | 0.555 | UNAVAILABLE |
| V4_s2_E3b | 1.0162 | 1.0396 | 3.4819 | 0.5677 | 4.1003 | 0.3872 | 0.8364 | UNAVAILABLE |
| V4_s2_E2 | 2.5689 | 0.6671 | 3.7692 | 0.8037 | 10.0974 | 0.5085 | 0.5593 | UNAVAILABLE |
| V4r_s0_E3b | 0.9713 | 1.0394 | 3.0915 | 0.4582 | 3.8458 | 0.4185 | 0.8353 | UNAVAILABLE |
| V4r_s0_E2 | 1.0998 | 0.9934 | 3.3209 | 0.5151 | 4.2595 | 0.4327 | 0.7724 | UNAVAILABLE |
| V4r_s1_E3b | 0.8535 | 1.1108 | 3.0692 | 0.4571 | 3.4062 | 0.4537 | 0.8537 | UNAVAILABLE |
| V4r_s1_E2 | 0.9611 | 1.0714 | 3.2088 | 0.4756 | 3.6784 | 0.4609 | 0.8008 | UNAVAILABLE |
| V4r_s2_E3b | 0.8418 | 1.088 | 2.89 | 0.4409 | 3.3217 | 0.4414 | 0.8609 | UNAVAILABLE |
| V4r_s2_E2 | 1.0311 | 1.0387 | 3.0181 | 0.4617 | 4.0103 | 0.4986 | 0.8239 | UNAVAILABLE |

<!-- HAND-WRITTEN BELOW THIS LINE (kept on re-run) -->
## Interpretation, declared departures, what remains (hand-written; every number from `eval/raw/m5r/readout_stage1.json` / `families_stage1.json`)

### 1. What Stage 1 decides, as registered
Stage 1 PASSES on all four bars (V4r - V3, W3's 200 tokens / 93 logs, repaired truth):

| bar | measured | margin to the bar |
|---|---|---|
| E1 | +0.0591 [+0.0112, +0.1040]; seed floor 0.0262 | +0.029 over the +0.03 point bar |
| E3a | +0.0247 [+0.0055, +0.0445] | lower bound 0.015 above −0.01 |
| E3b | +1.98 [−1.81, +6.61] | **lower bound only 0.19 PDMS above the −2.0 bar: the thinnest margin** |
| E2 | +7.98 [+4.16, +11.75] | lower bound well clear of −2.0 |

- **Estimator:** paired log-cluster bootstrap over 93 logs (10,000 draws, seed 20260927). Its self-test reproduces the published ep014→ep015 CI exactly.
- **Variance:** the CI answers "another draw of episodes". The seed floor answers "another training run" (3 seeds per arm). Inference is deterministic.
- **Gates:** G-R1..G-R6 all pass.
- **Exposure (declared in the prereg):** V3 and V4 had been read post hoc on these tokens.

**Under PREREG_MEASURE5R §6 a Stage-1 PASS licenses exactly one thing: running Stage 2.** It does NOT license adoption or deployment. ADOPT is decided only on Amendment 5's 923 fresh tokens, with the same four bars.

### 2. The reported contrasts: the effect comes from repairing the labels, not from the slowed copies (reported, NOT gating)

| contrast | E1 | E3a | E2 |
|---|---|---|---|
| V3r − V3 (repaired labels, no copies) | **+0.0778 [+0.0488, +0.1069]** | +0.0171 [−0.0032, +0.0379] | **+7.91 [+4.74, +11.10]** |
| V4r − V3r (copies, given repaired labels) | −0.0188 [−0.0508, +0.0093] | +0.0075 [+0.0023, +0.0132] | +0.07 [−2.24, +2.31] |
| V4r − V4 (repair, given copies) | +0.0979 [+0.0699, +0.1265] | — | — |

- In words: labelling the executed plan is the lever. Adding the 0.75x copies on top adds nothing measurable to slow-plan ranking or to the pick. It slightly helps ranking among the 64 (E3a).
- **E2 pick mix:** stopping collapses from 78–85 of 200 picks (V3) to 0–5 (V3r, V4r), and those picks go back to originals (152–170).
- **EP of the E2 pick:** V4r − V3 = **+18.61 [+14.27, +22.93] ×100**, with comfort +21.67 [+15.12, +28.57] and DAC −4.17 [−7.62, −0.91].
- **EP on the shipped route (E3b):** +3.79 [−0.09, +8.38], not separated. NC is +2.67 [+0.87, +5.11], separated.
- ⚠️ These contrasts were not pre-registered as gating, and they are on the exposed tokens. They motivate a registration; they decide nothing.

### 3. The four families (families6.py, per-model blocks vs the human future, seed means; E3b = the shipped route)

| arm | long: speed MAE | long: progress ratio | long: along MAE | lat: heading MAE | lat: cross MAE | tac: goal-point error | tac: long. kappa | tac: lat. kappa |
|---|---|---|---|---|---|---|---|---|
| V3 | 1.037 m/s | 1.033 | 1.460 m | 3.38° | 0.567 m | 4.16 m | 0.391 | 0.833 |
| V4r | **0.889** m/s | 1.079 | **1.243** m | **3.02°** | **0.452** m | **3.52** m | 0.438 | 0.850 |
| V3r | 0.892 m/s | 1.071 | 1.243 m | 3.02° | 0.463 m | 3.53 m | 0.409 | 0.847 |

- **Strategic:** UNAVAILABLE in NAVSIM (no route label).
- **E2 route:** V3's pick is far from the human (speed MAE 3.13 m/s, progress 0.57, goal error 12.5 m). V4r's is close (1.03 m/s, 1.03, 3.98 m). The difference is STOP picks being replaced.
- ⚠️ **The heading is the path tangent, so this block does not see the repair.** The families6 NAVSIM adapter drops the heading column (its heading MAE is the path tangent). A0's E3b block is therefore identical to the unrepaired one from last night.
- These are per-model blocks with families6's own intervals, NOT paired intervals.

**`proxy_eval.fam_adverse` / `fam_direction`: NOT USED.** My families step reads families6.py through `stop_candidate_probe.summarize_families` only. Two probes confirmed this:
- a grep of m5r_*.py, m5_*.py, m5r_write_result.py, stop_candidate_probe.py and families6.py finds no fam_adverse, fam_direction or `"acc" in` pattern;
- no direction or adverse classification is applied anywhere in the M5r readout.

The 13:45 fix therefore changes nothing here, and the families step was not re-run.

### 4. Declared departures
1. The M5 V3 and V4 model files were reused. This is gated in the prereg: the key lists are identical and G-R6 passed.
2. The fine-tune ran with `--preload`: 7 GB or more of RAM was free at 09:34. The numerics are the registered GPU bf16. It ran in window 1 under the shared lock, 09:34:24–09:46:57 (12.6 min, peak total VRAM 3,022 MiB); the eval ran 09:46:57–09:48:38.
3. Labelling ran under the coordinator's RAM rules (cap 3, 6.3/4.7 GB; queue-A exception).
   - 2 workers were retired, costing one re-labelled chunk each. Duplicates carry identical labels (G-R2 shows the labeller is deterministic).
   - At 07:02 the queue-A worker was retired instead of the newest one. Adopted workers had been stamped with the same launch time; this was fixed at 07:04.
4. Queue C was held until G-R3/G-R4 passed on all 200 tokens (approved by the coordinator).

### 5. Next registered step, and what blocks it
**Stage 2:** `eval/m5r_stage2.py` (staged, not run; it refuses to run without a Stage-1 PASS, and that condition is now met):
- `gate`: M6's eval cache must be admissible as the scorer context for the 923 tokens, checked bit for bit against `a7confirm_ep015/proposals.npz` and A0's logits;
- `seams`: 73 repaired seams;
- `score`: 73 unchanged-harness runs, about 11 CPU-h, RAM-bound;
- `eval`: on the GPU through the shared lock, about 5 min;
- `readout`.

Blockers:
- **Resources, not a PI decision:** about 11 CPU-h of harness scoring at roughly 1 GB per worker, plus a short GPU slot. Today both are held by the Master Mind's step-5,000 chain (~13:55 → ~04:00).
- **Stage 2 needs no pod.** An ADOPT there would lead to a DRAFTED pod kit (`onpolicy_label_v4.py --slow-copies --slow-factors 0.75 --slow-frac 0.5 --repair-last-heading`, label_version 5). That kit ships only on the PI's go, and nothing has been sent to the pod.
- **Rule Zero lever (not registered, not run):** V3r, repaired labels without the copies, carries the effect. It is the cheaper candidate for the live change, and its own pre-registration on the 923 fresh tokens could ride on the same Stage-2 harness runs at no extra CPU. The scoring truth is identical; only one more arm would be read. This needs the coordinator or PI to register it before Stage 2's readout.
