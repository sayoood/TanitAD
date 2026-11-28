# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 20, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LFTRL:LFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LCORL:LCO", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LORL:LO", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LCO:LFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LO:LFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LC:LFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LCORL:LFTRL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LFTRL minus LFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LCORL minus LCO over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LORL minus LO over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LCO minus LFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LO minus LFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LC minus LFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LCORL minus LFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LFTRL minus LFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0907 / -0.0801 | -0.0854 | [-0.1240, -0.0504] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0592 / -0.0639 | -0.0615 | [-0.0900, -0.0363] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0654 / +0.0794 | +0.0724 | [+0.0370, +0.1174] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4357 / +0.4399 | +0.4378 | [+0.1828, +0.7589] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.3874 / +1.5441 | +1.4657 | [+0.7389, +2.3847] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +3.5043 / +3.9700 | +3.7371 | [+2.3164, +5.5960] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3175 / +0.3225 | +0.3200 | [+0.0908, +0.6167] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6057 / +0.5842 | +0.5949 | [+0.0757, +1.2642] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.1749 / +1.4327 | +1.3038 | [+0.0790, +2.7971] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0398 / +0.0428 | +0.0413 | [+0.0172, +0.0699] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0153 / +0.0704 | +0.0428 | [-0.0481, +0.1312] | 1.7849 (0.8089) | OPEN |

## LCORL minus LCO

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0463 / -0.0658 | -0.0560 | [-0.0868, -0.0302] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0234 / -0.0218 | -0.0226 | [-0.0354, -0.0105] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0093 / +0.0062 | -0.0016 | [-0.0141, +0.0099] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0170 / +0.0242 | +0.0206 | [+0.0102, +0.0354] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.0643 / +0.0634 | +0.0639 | [+0.0024, +0.1319] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.2668 / +0.3578 | +0.3123 | [+0.1019, +0.5230] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0054 / +0.0138 | +0.0096 | [-0.0005, +0.0243] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0312 / -0.0661 | -0.0486 | [-0.1206, +0.0125] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0399 / +0.0712 | +0.0156 | [-0.1639, +0.1644] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0165 / +0.0144 | +0.0154 | [+0.0031, +0.0323] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.0318 / -0.0041 | -0.0179 | [-0.0683, +0.0361] | 1.7849 (0.8089) | OPEN |

## LORL minus LO

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0569 | -0.0578 | [-0.0852, -0.0329] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0202 / -0.0093 | -0.0148 | [-0.0288, -0.0032] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0031 / +0.0031 | +0.0031 | [-0.0029, +0.0098] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0180 / +0.0258 | +0.0219 | [+0.0105, +0.0367] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.0401 / -0.0295 | +0.0053 | [-0.0761, +0.0673] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.3117 / -0.0835 | +0.1141 | [-0.2665, +0.5102] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0053 / +0.0184 | +0.0118 | [+0.0006, +0.0277] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.0245 / -0.0705 | -0.0475 | [-0.1196, +0.0015] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.2046 / -0.3888 | -0.2967 | [-0.6205, -0.0527] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0182 / +0.0150 | +0.0166 | [+0.0038, +0.0347] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0142 / -0.0097 | +0.0023 | [-0.0461, +0.0486] | 1.7849 (0.8089) | OPEN |

## LCO minus LFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0196 / +0.0249 | +0.0222 | [-0.0076, +0.0539] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0202 / +0.0218 | +0.0210 | [+0.0091, +0.0340] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1900 / +0.2336 | +0.2118 | [+0.1621, +0.2645] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.7491 / +1.9263 | +1.8377 | [+1.2573, +2.5106] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +5.3259 / +6.0301 | +5.6780 | [+3.9219, +7.7122] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +9.5935 / +10.7887 | +10.1911 | [+7.0564, +13.8268] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.4197 / +1.6124 | +1.5161 | [+0.9330, +2.2054] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +4.9340 / +5.4527 | +5.1933 | [+3.4099, +7.1903] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +9.5220 / +10.2425 | +9.8822 | [+6.7545, +13.3934] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.8328 / +1.3075 | +1.0702 | [+0.6634, +1.6402] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +1.4001 / +1.5565 | +1.4783 | [+0.9651, +2.1716] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LO minus LFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0374 / +0.0231 | +0.0302 | [-0.0001, +0.0625] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0187 / +0.0093 | +0.0140 | [+0.0032, +0.0265] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1604 / +0.2555 | +0.2079 | [+0.1321, +0.2849] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.6896 / +1.9321 | +1.8108 | [+1.2326, +2.4942] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.1629 / +6.7137 | +4.9383 | [+2.6399, +8.2165] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.2983 / +12.5724 | +9.4353 | [+5.2196, +15.2953] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.3660 / +1.6002 | +1.4831 | [+0.8982, +2.1611] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.8523 / +5.9996 | +4.4259 | [+2.3223, +7.4660] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +6.4787 / +11.9445 | +9.2116 | [+5.3852, +14.6586] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.8787 / +1.4476 | +1.1631 | [+0.7072, +1.8128] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +0.7382 / +2.3863 | +1.5622 | [+0.6212, +3.1650] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LC minus LFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0730 / +0.0641 | +0.0685 | [+0.0369, +0.1041] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0545 / +0.0483 | +0.0514 | [+0.0287, +0.0770] | 0.0421 (0.0171) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0358 / -0.0389 | -0.0374 | [-0.0847, +0.0042] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.4222 / -0.4036 | -0.4129 | [-0.7350, -0.1552] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -1.0473 / -1.2235 | -1.1354 | [-2.0322, -0.4302] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.1720 / -2.6829 | -2.4275 | [-4.3994, -0.8190] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.3194 / -0.3360 | -0.3277 | [-0.6246, -0.0967] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.4745 / -0.4669 | -0.4707 | [-1.1536, +0.0718] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.2954 / -0.4869 | -0.3912 | [-1.8387, +0.8603] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0301 / +0.0088 | -0.0106 | [-0.0658, +0.0521] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.2236 / +0.3209 | +0.2723 | [+0.1026, +0.4893] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LCORL minus LFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0267 / -0.0409 | -0.0338 | [-0.0630, -0.0104] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0031 / +0.0000 | -0.0016 | [-0.0062, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.1807 / +0.2399 | +0.2103 | [+0.1561, +0.2692] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.7661 / +1.9506 | +1.8583 | [+1.2765, +2.5335] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +5.3902 / +6.0935 | +5.7419 | [+3.9571, +7.7932] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +9.8604 / +11.1464 | +10.5034 | [+7.3391, +14.1482] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.4252 / +1.6262 | +1.5257 | [+0.9459, +2.2164] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +4.9028 / +5.3866 | +5.1447 | [+3.3831, +7.1456] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +9.4821 / +10.3136 | +9.8978 | [+6.7211, +13.5190] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.8493 / +1.3219 | +1.0856 | [+0.6829, +1.6574] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +1.3683 / +1.5524 | +1.4603 | [+0.9419, +2.1655] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LC (0,1) | 0.1246 [0.0809, 0.1708] | 0.0530 [0.0295, 0.0793] | 0.1807 [0.1295, 0.2356] | -0.1975 [-0.2479, -0.1477] | 0.8489 [0.3548, 1.3336] | 4.5461 [2.8669, 6.2820] | 0.4755 [0.4254, 0.5289] | 2.6780 [2.3775, 3.0159] | 7.9414 [6.9331, 9.0713] | 0.9822 [0.6653, 1.3655] | 2.3237 [1.6840, 3.0560] |
| LCO (0,1) | 0.0783 [0.0490, 0.1114] | 0.0226 [0.0105, 0.0354] | 0.4299 [0.3645, 0.4948] | 2.0531 [1.2651, 2.9565] | 7.6624 [5.3794, 10.2552] | 17.1647 [12.9244, 21.7622] | 2.3193 [1.5291, 3.1990] | 8.3420 [6.0987, 10.9234] | 18.2148 [14.1357, 22.7485] | 2.0630 [1.4692, 2.8140] | 3.5297 [2.8312, 4.4338] |
| LCORL (0,1) | 0.0222 [0.0114, 0.0356] | 0.0000 [0.0000, 0.0000] | 0.4283 [0.3624, 0.4939] | 2.0737 [1.2847, 2.9749] | 7.7262 [5.4493, 10.3072] | 17.4770 [13.2353, 22.1040] | 2.3289 [1.5399, 3.2133] | 8.2934 [6.0515, 10.8654] | 18.2304 [14.1467, 22.7706] | 2.0784 [1.4860, 2.8291] | 3.5117 [2.8091, 4.4498] |
| LFT (0,1) | 0.1415 [0.0965, 0.1878] | 0.0631 [0.0370, 0.0918] | 0.1456 [0.0958, 0.1980] | -0.2224 [-0.2672, -0.1763] | 0.5186 [0.0504, 0.9670] | 3.2365 [1.7611, 4.7609] | 0.4832 [0.4338, 0.5358] | 2.5537 [2.2424, 2.8619] | 7.0288 [6.1826, 7.9892] | 0.9515 [0.6423, 1.3322] | 2.0086 [1.4447, 2.7032] |
| LFTRL (0,1) | 0.0560 [0.0301, 0.0870] | 0.0016 [0.0000, 0.0062] | 0.2181 [0.1638, 0.2758] | 0.2154 [-0.0402, 0.5332] | 1.9843 [1.2798, 2.7899] | 6.9736 [5.2239, 8.7284] | 0.8032 [0.5710, 1.1043] | 3.1487 [2.5815, 3.8749] | 8.3326 [6.9243, 9.8570] | 0.9928 [0.6759, 1.3704] | 2.0514 [1.4786, 2.7324] |
| LO (0,1) | 0.0863 [0.0542, 0.1203] | 0.0156 [0.0033, 0.0300] | 0.4260 [0.3440, 0.5045] | 2.0263 [1.2562, 2.9294] | 6.9227 [4.2823, 10.4875] | 16.4089 [11.7976, 22.6707] | 2.2863 [1.5438, 3.1573] | 7.5746 [5.0730, 11.0222] | 17.5442 [13.1896, 23.5785] | 2.1560 [1.5184, 2.9748] | 3.6136 [2.3252, 5.4031] |
| LORL (0,1) | 0.0285 [0.0146, 0.0445] | 0.0008 [0.0000, 0.0031] | 0.4291 [0.3473, 0.5064] | 2.0481 [1.2776, 2.9450] | 6.9280 [4.3295, 10.4803] | 16.5230 [12.0283, 22.6181] | 2.2981 [1.5467, 3.1702] | 7.5271 [5.0264, 10.9843] | 17.2475 [12.9039, 23.2118] | 2.1726 [1.5345, 2.9899] | 3.6159 [2.3406, 5.3812] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
