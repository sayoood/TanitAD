# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 14, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFT6RL:GFT6", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRT6RL:GRT6", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRT6:GFT6RL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRT6RL:GFT6RL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GFT6RL minus GFT6 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRT6RL minus GRT6 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRT6 minus GFT6RL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRT6RL minus GFT6RL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GFT6RL minus GFT6

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0783 | -0.0685 | [-0.1016, -0.0380] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0483 / -0.0498 | -0.0491 | [-0.0733, -0.0275] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0685 / +0.0607 | +0.0646 | [+0.0342, +0.1000] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4438 / +0.4387 | +0.4413 | [+0.1876, +0.7647] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.2634 / +1.2125 | +1.2379 | [+0.6461, +1.9387] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.7340 / +2.7329 | +2.7334 | [+1.6028, +4.1807] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3074 / +0.3187 | +0.3131 | [+0.0854, +0.6077] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7715 / +0.5187 | +0.6451 | [+0.1352, +1.2715] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.5212 / +1.0019 | +1.2615 | [+0.1358, +2.5194] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0357 / +0.0421 | +0.0389 | [+0.0159, +0.0666] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0976 / +0.0746 | +0.0861 | [+0.0103, +0.1684] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GRT6RL minus GRT6

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0658 / -0.0712 | -0.0685 | [-0.0972, -0.0427] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0483 / -0.0498 | -0.0491 | [-0.0737, -0.0277] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0421 / +0.0389 | +0.0405 | [+0.0244, +0.0602] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3811 / +0.3846 | +0.3828 | [+0.1629, +0.6598] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.9804 / +0.8707 | +0.9255 | [+0.4878, +1.4502] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.8650 / +1.9215 | +1.8932 | [+1.1935, +2.7167] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3070 / +0.3214 | +0.3142 | [+0.0955, +0.5946] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6437 / +0.5111 | +0.5774 | [+0.1451, +1.1107] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.9344 / +0.5592 | +0.7468 | [-0.0065, +1.6413] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0401 / +0.0409 | +0.0405 | [+0.0168, +0.0680] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1438 / +0.0858 | +0.1148 | [+0.0282, +0.2177] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GRT6 minus GFT6RL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0623 / +0.0641 | +0.0632 | [+0.0369, +0.0921] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0467 / +0.0498 | +0.0483 | [+0.0270, +0.0730] | 0.0421 (0.0171) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0607 / -0.0280 | -0.0444 | [-0.0753, -0.0150] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2932 / -0.2938 | -0.2935 | [-0.5689, -0.0849] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -0.8928 / -0.4761 | -0.6844 | [-1.1821, -0.2038] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.4525 / -0.9063 | -1.6794 | [-2.9469, -0.3184] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.2887 / -0.2971 | -0.2929 | [-0.5644, -0.0869] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.4505 / -0.2654 | -0.3580 | [-0.8152, +0.0280] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -1.3283 / +0.2650 | -0.5317 | [-1.8531, +0.7825] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0371 / -0.0073 | -0.0222 | [-0.0567, +0.0174] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.2045 / -0.1033 | -0.1539 | [-0.3272, +0.0167] | 1.7849 (0.8089) | OPEN |

## GRT6RL minus GFT6RL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / -0.0071 | -0.0053 | [-0.0187, +0.0049] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0016 / +0.0000 | -0.0008 | [-0.0036, +0.0023] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0187 / +0.0109 | -0.0039 | [-0.0320, +0.0231] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0879 / +0.0908 | +0.0894 | [+0.0622, +0.1205] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.0877 / +0.3946 | +0.2411 | [-0.0878, +0.5422] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.5876 / +1.0151 | +0.2138 | [-0.9923, +1.4529] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0183 / +0.0243 | +0.0213 | [-0.0039, +0.0508] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.1932 / +0.2456 | +0.2194 | [+0.0262, +0.4183] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.3939 / +0.8242 | +0.2151 | [-0.8153, +1.2598] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0029 / +0.0336 | +0.0183 | [-0.0093, +0.0527] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0607 / -0.0175 | -0.0391 | [-0.1760, +0.0839] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT6 (0,1) | 0.1201 [0.0734, 0.1731] | 0.0506 [0.0281, 0.0755] | 0.1900 [0.1253, 0.2584] | -0.2167 [-0.2618, -0.1654] | 0.9637 [0.4299, 1.4763] | 5.1432 [2.9081, 7.4572] | 0.4852 [0.4418, 0.5347] | 2.5879 [2.2987, 2.9080] | 7.9793 [6.5034, 9.6963] | 1.0540 [0.7300, 1.4381] | 2.1003 [1.4546, 2.8105] |
| GFT6RL (0,1) | 0.0516 [0.0252, 0.0847] | 0.0016 [0.0000, 0.0040] | 0.2547 [0.1786, 0.3317] | 0.2245 [-0.0320, 0.5406] | 2.2017 [1.4112, 3.0245] | 7.8766 [5.3921, 10.4152] | 0.7982 [0.5726, 1.1002] | 3.2331 [2.6311, 3.9487] | 9.2408 [7.2165, 11.4317] | 1.0929 [0.7677, 1.4801] | 2.1865 [1.5505, 2.9055] |
| GRT6 (0,1) | 0.1148 [0.0742, 0.1608] | 0.0498 [0.0283, 0.0749] | 0.2103 [0.1531, 0.2694] | -0.0689 [-0.1395, 0.0075] | 1.5173 [1.0001, 2.1083] | 6.1972 [4.3909, 7.9833] | 0.5053 [0.4487, 0.5685] | 2.8751 [2.4766, 3.3472] | 8.7091 [7.3765, 10.1580] | 1.0707 [0.7392, 1.4555] | 2.0325 [1.4331, 2.7233] |
| GRT6RL (0,1) | 0.0463 [0.0233, 0.0723] | 0.0008 [0.0000, 0.0032] | 0.2508 [0.1868, 0.3159] | 0.3139 [0.0417, 0.6365] | 2.4428 [1.6390, 3.3403] | 8.0904 [6.0396, 10.2526] | 0.8195 [0.5790, 1.1322] | 3.4525 [2.7577, 4.2892] | 9.4559 [7.6624, 11.3899] | 1.1112 [0.7822, 1.5022] | 2.1473 [1.5339, 2.8498] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
