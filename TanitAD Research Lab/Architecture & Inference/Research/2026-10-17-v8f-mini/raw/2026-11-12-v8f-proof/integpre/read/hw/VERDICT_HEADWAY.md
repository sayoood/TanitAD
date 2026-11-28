# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 38, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.10563170027848558, "A_hi": 0.15899261888352556, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IRQ_RLAm:IR0_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IRQ_RLAm:IR0_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IRQ_RLAm:IRL_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IRQ_RLAm:IRQ_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IR0_RLAm:IR0_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IRL_RLAm:IR0_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IRL_RLFT:IR0_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IR0_RLAm:IPox_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IR0_RLFT:IPox_RLFT", "seeds": [0, 1, 2, 3]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IRQ_RLAm minus IR0_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IRQ_RLAm minus IR0_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IRQ_RLAm minus IRL_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IRQ_RLAm minus IRQ_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IR0_RLAm minus IR0_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IRL_RLAm minus IR0_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IRL_RLFT minus IR0_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IR0_RLAm minus IPox_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IR0_RLFT minus IPox_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IRQ RLAm minus IR0 RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0356 / -0.0783 / -0.0676 / -0.0747 | -0.0641 | [-0.0954, -0.0361] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0187 / -0.0265 / -0.0436 / -0.0545 | -0.0358 | [-0.0540, -0.0197] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1324 / +0.1636 / +0.1760 / +0.1231 | +0.1488 | [+0.1054, +0.1942] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7300 / +0.8804 / +0.9062 / +0.8802 | +0.8492 | [+0.6371, +1.0927] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6790 / +3.3625 / +3.4357 / +3.0041 | +3.1203 | [+2.2586, +4.0833] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.5133 / +6.0633 / +6.9223 / +4.5291 | +5.7570 | [+4.1591, +7.5639] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.5626 / +0.6932 / +0.6493 / +0.5944 | +0.6249 | [+0.4445, +0.8311] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.3014 / +2.2462 / +2.4687 / +2.0009 | +2.2543 | [+1.6107, +2.9942] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.5134 / +4.5008 / +4.4870 / +2.9525 | +4.1134 | [+2.7397, +5.7522] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1992 / +0.2203 / +0.3041 / +0.1551 | +0.2197 | [+0.1420, +0.3307] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.7601 / +0.5690 / +0.6881 / +0.3789 | +0.5990 | [+0.3853, +0.8415] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IRQ RLAm minus IR0 RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0463 / -0.0819 / -0.0765 / -0.0783 | -0.0707 | [-0.1014, -0.0427] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0234 / -0.0202 / -0.0374 / -0.0498 | -0.0327 | [-0.0491, -0.0184] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1371 / +0.1620 / +0.1807 / +0.1464 | +0.1565 | [+0.1159, +0.2003] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7419 / +0.8881 / +0.8980 / +0.8674 | +0.8489 | [+0.6400, +1.0876] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6701 / +3.4108 / +3.3464 / +3.0672 | +3.1236 | [+2.2572, +4.0857] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.5992 / +5.8822 / +6.6668 / +5.1381 | +5.8216 | [+4.2705, +7.5704] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.5594 / +0.6864 / +0.6588 / +0.6024 | +0.6267 | [+0.4463, +0.8335] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.2127 / +2.3342 / +2.4889 / +2.0706 | +2.2766 | [+1.6437, +2.9959] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.3643 / +4.4340 / +4.5758 / +3.3383 | +4.1781 | [+2.8307, +5.7361] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1291 / +0.2233 / +0.2114 / +0.1560 | +0.1800 | [+0.1169, +0.2697] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.6694 / +0.7240 / +0.6129 / +0.5552 | +0.6404 | [+0.4714, +0.8297] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IRQ RLAm minus IRL RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0214 / +0.0142 / +0.0125 / +0.0196 | +0.0169 | [+0.0060, +0.0290] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0218 / +0.0156 / +0.0156 / +0.0078 | +0.0152 | [+0.0063, +0.0272] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0670 / +0.1106 / +0.1059 / +0.0810 | +0.0911 | [+0.0616, +0.1231] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3043 / +0.4468 / +0.4537 / +0.4205 | +0.4063 | [+0.2809, +0.5363] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.3364 / +1.7840 / +1.9538 / +1.6563 | +1.6826 | [+1.1736, +2.2742] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +2.6018 / +2.9674 / +3.1019 / +2.1932 | +2.7161 | [+1.7874, +3.8177] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2482 / +0.3672 / +0.3412 / +0.2894 | +0.3115 | [+0.1978, +0.4269] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.5413 / +1.7134 / +1.8368 / +1.5189 | +1.6526 | [+1.2274, +2.1484] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.9062 / +3.0357 / +2.9935 / +2.2697 | +2.8013 | [+1.9482, +3.8016] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0861 / +0.1826 / +0.1667 / +0.1189 | +0.1386 | [+0.0740, +0.2290] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.5348 / +0.6156 / +0.5422 / +0.5162 | +0.5522 | [+0.4059, +0.7227] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IRQ RLAm minus IRQ RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / -0.0036 / -0.0053 / +0.0196 | +0.0022 | [-0.0078, +0.0153] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / -0.0016 / -0.0078 / -0.0031 | -0.0031 | [-0.0083, +0.0011] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0078 / +0.0062 / -0.0125 / -0.0436 | -0.0105 | [-0.0352, +0.0113] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0478 / -0.0087 / +0.0026 / -0.0833 | -0.0343 | [-0.0837, +0.0123] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.1780 / -0.1065 / +0.0347 / -0.4057 | -0.0749 | [-0.4012, +0.2444] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.5215 / -0.0193 / -0.2212 / -1.0149 | -0.1835 | [-0.8549, +0.4668] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0199 / -0.0110 / -0.0235 / -0.0703 | -0.0312 | [-0.0710, +0.0037] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.1261 / -0.1428 / -0.0994 / -0.2867 | -0.1007 | [-0.3594, +0.1522] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.2681 / -0.3759 / -0.4066 / -0.8675 | -0.3455 | [-0.8885, +0.1849] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0596 / -0.0182 / +0.0966 / -0.0434 | +0.0236 | [-0.0439, +0.0908] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.2688 / -0.1645 / -0.1180 / -0.3988 | -0.1031 | [-0.3479, +0.1586] | 1.7849 (0.8089) | OPEN |

## IR0 RLAm minus IR0 RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0107 / +0.0036 / +0.0089 / +0.0036 | +0.0067 | [-0.0022, +0.0156] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0047 / -0.0062 / -0.0062 / -0.0047 | -0.0031 | [-0.0090, +0.0029] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0047 / +0.0016 / -0.0047 / -0.0234 | -0.0078 | [-0.0213, +0.0046] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0118 / -0.0077 / +0.0082 / +0.0128 | +0.0004 | [-0.0105, +0.0112] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0088 / -0.0482 / +0.0893 / -0.0631 | -0.0033 | [-0.0861, +0.0832] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.0859 / +0.1811 / +0.2555 / -0.6090 | -0.0646 | [-0.4799, +0.3239] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0032 / +0.0067 / -0.0095 / -0.0080 | -0.0019 | [-0.0097, +0.0060] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0887 / -0.0880 / -0.0202 / -0.0697 | -0.0223 | [-0.0931, +0.0686] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.1491 / +0.0668 / -0.0888 / -0.3858 | -0.0647 | [-0.3307, +0.2415] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0701 / -0.0031 / +0.0927 / -0.0009 | +0.0397 | [-0.0136, +0.0964] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0907 / -0.1550 / +0.0752 / -0.1763 | -0.0413 | [-0.1783, +0.0852] | 1.7849 (0.8089) | OPEN |

## IRL RLAm minus IR0 RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0676 / -0.0961 / -0.0890 / -0.0979 | -0.0876 | [-0.1189, -0.0615] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0452 / -0.0358 / -0.0530 / -0.0576 | -0.0479 | [-0.0670, -0.0297] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0701 / +0.0514 / +0.0748 / +0.0654 | +0.0654 | [+0.0355, +0.0980] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4376 / +0.4413 / +0.4443 / +0.4470 | +0.4425 | [+0.2443, +0.6642] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.3337 / +1.6267 / +1.3925 / +1.4108 | +1.4410 | [+0.8719, +2.0842] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.9974 / +2.9148 / +3.5648 / +2.9449 | +3.1055 | [+2.0337, +4.4202] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3112 / +0.3192 / +0.3176 / +0.3130 | +0.3152 | [+0.1436, +0.5098] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6714 / +0.6208 / +0.6521 / +0.5518 | +0.6240 | [+0.2323, +1.0908] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.4581 / +1.3983 / +1.5823 / +1.0686 | +1.3768 | [+0.5137, +2.3447] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0430 / +0.0407 / +0.0447 / +0.0371 | +0.0414 | [+0.0241, +0.0616] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1345 / +0.1084 / +0.0707 / +0.0390 | +0.0882 | [+0.0247, +0.1605] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IRL RLFT minus IR0 RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0605 / -0.1014 / -0.0765 / -0.0925 | -0.0827 | [-0.1150, -0.0562] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0421 / -0.0421 / -0.0576 / -0.0623 | -0.0510 | [-0.0703, -0.0320] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0732 / +0.0701 / +0.0779 / +0.0561 | +0.0693 | [+0.0400, +0.1010] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4382 / +0.4416 / +0.4449 / +0.4511 | +0.4439 | [+0.2464, +0.6666] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.1509 / +1.6272 / +1.4398 / +1.4665 | +1.4211 | [+0.8580, +2.0686] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.5756 / +3.4931 / +3.7527 / +2.6896 | +3.1277 | [+2.0453, +4.6246] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3105 / +0.3192 / +0.3162 / +0.3111 | +0.3142 | [+0.1420, +0.5079] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6280 / +0.6278 / +0.6673 / +0.5844 | +0.6269 | [+0.2375, +1.0755] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.3630 / +2.0392 / +1.4879 / +0.8472 | +1.4343 | [+0.5438, +2.5550] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0418 / +0.0431 / +0.0416 / +0.0430 | +0.0424 | [+0.0247, +0.0628] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0992 / +0.1150 / +0.0859 / -0.0124 | +0.0719 | [-0.0042, +0.1482] | 1.7849 (0.8089) | OPEN |

## IR0 RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.7849 (0.8089) | OPEN |

## IR0 RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPox_RLAm (0,1,2,3) | 0.1343 [0.0993, 0.1727] | 0.0483 [0.0300, 0.0674] | 0.1671 [0.1317, 0.2030] | -0.2256 [-0.2652, -0.1892] | 0.7543 [0.3440, 1.1418] | 4.3759 [3.2451, 5.5775] | 0.4875 [0.4494, 0.5255] | 2.6551 [2.4528, 2.8628] | 7.3060 [6.5324, 8.2349] | 1.1495 [0.8658, 1.4616] | 2.0556 [1.6548, 2.5183] |
| IPox_RLFT (0,1,2,3) | 0.1277 [0.0931, 0.1663] | 0.0514 [0.0323, 0.0707] | 0.1748 [0.1399, 0.2101] | -0.2260 [-0.2668, -0.1880] | 0.7576 [0.3585, 1.1267] | 4.4404 [3.2460, 5.7031] | 0.4894 [0.4522, 0.5277] | 2.6774 [2.4554, 2.8968] | 7.3706 [6.6689, 8.1312] | 1.1098 [0.8400, 1.4028] | 2.0969 [1.6720, 2.5587] |
| IR0_RLAm (0,1,2,3) | 0.1343 [0.0993, 0.1727] | 0.0483 [0.0300, 0.0674] | 0.1671 [0.1317, 0.2030] | -0.2256 [-0.2652, -0.1892] | 0.7543 [0.3440, 1.1418] | 4.3759 [3.2451, 5.5775] | 0.4875 [0.4494, 0.5255] | 2.6551 [2.4528, 2.8628] | 7.3060 [6.5324, 8.2349] | 1.1495 [0.8658, 1.4616] | 2.0556 [1.6548, 2.5183] |
| IR0_RLFT (0,1,2,3) | 0.1277 [0.0931, 0.1663] | 0.0514 [0.0323, 0.0707] | 0.1748 [0.1399, 0.2101] | -0.2260 [-0.2668, -0.1880] | 0.7576 [0.3585, 1.1267] | 4.4404 [3.2460, 5.7031] | 0.4894 [0.4522, 0.5277] | 2.6774 [2.4554, 2.8968] | 7.3706 [6.6689, 8.1312] | 1.1098 [0.8400, 1.4028] | 2.0969 [1.6720, 2.5587] |
| IRL_RLAm (0,1,2,3) | 0.0467 [0.0306, 0.0658] | 0.0004 [0.0000, 0.0016] | 0.2325 [0.1919, 0.2732] | 0.2169 [0.0243, 0.4305] | 2.1953 [1.6415, 2.7710] | 7.4814 [6.2155, 8.8409] | 0.8027 [0.6293, 0.9971] | 3.2791 [2.8365, 3.7702] | 8.6828 [7.6037, 9.9495] | 1.1908 [0.9092, 1.5083] | 2.1437 [1.7444, 2.6076] |
| IRL_RLFT (0,1,2,3) | 0.0449 [0.0291, 0.0636] | 0.0004 [0.0000, 0.0017] | 0.2442 [0.2020, 0.2847] | 0.2180 [0.0238, 0.4309] | 2.1787 [1.6636, 2.7309] | 7.5682 [6.2815, 8.9573] | 0.8036 [0.6283, 0.9980] | 3.3043 [2.8612, 3.7781] | 8.8049 [7.7032, 10.1042] | 1.1521 [0.8855, 1.4480] | 2.1689 [1.7554, 2.6218] |
| IRQ_RLAm (0,1,2,3) | 0.0636 [0.0458, 0.0846] | 0.0156 [0.0065, 0.0280] | 0.3236 [0.2783, 0.3676] | 0.6232 [0.4150, 0.8502] | 3.8779 [3.1411, 4.6809] | 10.1975 [8.6362, 11.9326] | 1.1142 [0.9323, 1.3236] | 4.9317 [4.2572, 5.6891] | 11.4840 [10.0483, 13.1182] | 1.3294 [1.0464, 1.6554] | 2.6960 [2.2585, 3.1778] |
| IRQ_RLFT (0,1,2,3) | 0.0614 [0.0399, 0.0855] | 0.0187 [0.0083, 0.0307] | 0.3341 [0.2902, 0.3780] | 0.6575 [0.4545, 0.8829] | 3.9528 [3.1868, 4.7703] | 10.3809 [8.8829, 12.0140] | 1.1454 [0.9614, 1.3535] | 5.0324 [4.3338, 5.8126] | 11.8295 [10.4836, 13.3915] | 1.3058 [1.0272, 1.6218] | 2.7991 [2.3067, 3.3062] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
