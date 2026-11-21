# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 18, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPlq_LS:IPoq_LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcq_LS:IPoq_LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcq_LS:IPlq_LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcqq_LS:IPcq_LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcqq_LS:IPlq_LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcqq_LS:IPoq_LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcq0q_LS:IPcq_LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcqidq_LS:IPcq_LS", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IPlq_LS minus IPoq_LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcq_LS minus IPoq_LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcq_LS minus IPlq_LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcqq_LS minus IPcq_LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcqq_LS minus IPlq_LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcqq_LS minus IPoq_LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcq0q_LS minus IPcq_LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcqidq_LS minus IPcq_LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IPlq LS minus IPoq LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0819 / -0.0979 | -0.0899 | [-0.1289, -0.0583] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0312 / -0.0561 | -0.0436 | [-0.0742, -0.0189] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1012 / +0.0826 | +0.0919 | [+0.0505, +0.1402] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8760 / +0.9087 | +0.8923 | [+0.3654, +1.5803] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.5249 / +3.6576 | +3.5913 | [+1.5274, +6.2694] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.3433 / +6.6985 | +6.5209 | [+3.0242, +11.2862] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7786 / +0.7497 | +0.7641 | [+0.2688, +1.3946] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.7957 / +2.9536 | +2.8746 | [+0.9259, +5.4873] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.1331 / +4.2555 | +4.6943 | [+1.1797, +9.0961] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3055 / +0.3128 | +0.3092 | [+0.1407, +0.5189] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0707 / +1.3239 | +1.1973 | [+0.4891, +2.1805] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPcq LS minus IPoq LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0641 | -0.0578 | [-0.0888, -0.0323] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0140 / -0.0312 | -0.0226 | [-0.0430, -0.0076] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0935 / +0.1511 | +0.1223 | [+0.0631, +0.1897] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7180 / +1.2597 | +0.9888 | [+0.3926, +1.6551] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.9820 / +4.8137 | +3.8978 | [+1.6369, +6.4980] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.2714 / +8.7344 | +7.0029 | [+2.9826, +11.6298] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6504 / +1.0989 | +0.8747 | [+0.3398, +1.4612] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.3677 / +4.4727 | +3.4202 | [+1.2038, +5.9605] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.2720 / +6.8589 | +5.5654 | [+2.1208, +9.7396] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1081 / +1.3109 | +0.7095 | [+0.0580, +1.6365] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.5504 / +2.7364 | +1.6434 | [+0.2593, +3.3325] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPcq LS minus IPlq LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0302 / +0.0338 | +0.0320 | [+0.0160, +0.0536] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0171 / +0.0249 | +0.0210 | [+0.0072, +0.0386] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0078 / +0.0685 | +0.0304 | [-0.0184, +0.0869] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1580 / +0.3510 | +0.0965 | [-0.2459, +0.5697] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.5429 / +1.1560 | +0.3066 | [-0.8844, +2.0077] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.0719 / +2.0359 | +0.4820 | [-1.6105, +3.4786] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.1282 / +0.3493 | +0.1105 | [-0.2181, +0.5543] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.4280 / +1.5191 | +0.5456 | [-0.7567, +2.3539] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.8611 / +2.6034 | +0.8712 | [-1.4111, +4.0236] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1974 / +0.9982 | +0.4004 | [-0.2848, +1.3153] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.5202 / +1.4125 | +0.4461 | [-0.7308, +2.0299] | 1.7849 (0.8089) | OPEN |

## IPcqq LS minus IPcq LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0374 / -0.0338 | -0.0356 | [-0.0518, -0.0208] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0171 / -0.0140 | -0.0156 | [-0.0296, -0.0042] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1044 / +0.0950 | +0.0997 | [+0.0654, +0.1375] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.1683 / +0.8879 | +1.0281 | [+0.6711, +1.4091] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +4.0852 / +3.2534 | +3.6693 | [+2.3090, +5.0796] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.9591 / +5.5651 | +6.2621 | [+3.7674, +8.9861] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.0177 / +0.7653 | +0.8915 | [+0.5549, +1.2518] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.8325 / +2.8573 | +3.3449 | [+1.9592, +4.7583] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +6.6171 / +4.8038 | +5.7104 | [+3.0365, +8.5850] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +1.9058 / +1.5607 | +1.7332 | [+1.0206, +2.4863] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +3.4052 / +2.5637 | +2.9845 | [+1.7914, +4.2053] | 1.7849 (0.8089) | WORSE |

## IPcqq LS minus IPlq LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / +0.0000 | -0.0036 | [-0.0172, +0.0156] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0109 | +0.0055 | [+0.0000, +0.0169] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0966 / +0.1636 | +0.1301 | [+0.0740, +0.1927] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0103 / +1.2389 | +1.1246 | [+0.7276, +1.5582] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.5423 / +4.4094 | +3.9758 | [+2.4996, +5.6839] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.8872 / +7.6011 | +6.7441 | [+3.8332, +9.9398] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.8895 / +1.1145 | +1.0020 | [+0.6287, +1.4221] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.4045 / +4.3764 | +3.8904 | [+2.4213, +5.6347] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.7560 / +7.4072 | +6.5816 | [+3.7780, +9.7520] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +1.7084 / +2.5589 | +2.1336 | [+1.2415, +3.3149] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +2.8850 / +3.9763 | +3.4306 | [+2.1434, +4.9641] | 1.7849 (0.8089) | WORSE |

## IPcqq LS minus IPoq LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0890 / -0.0979 | -0.0934 | [-0.1313, -0.0595] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0312 / -0.0452 | -0.0382 | [-0.0614, -0.0181] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1978 / +0.2461 | +0.2220 | [+0.1552, +0.2944] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.8863 / +2.1476 | +2.0169 | [+1.3891, +2.7005] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +7.0672 / +8.0670 | +7.5671 | [+5.0591, +10.2877] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +12.2305 / +14.2995 | +13.2650 | [+8.7880, +18.2369] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.6681 / +1.8642 | +1.7662 | [+1.1803, +2.4112] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +6.2002 / +7.3300 | +6.7651 | [+4.3850, +9.4423] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +10.8890 / +11.6627 | +11.2759 | [+7.1621, +15.8638] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +2.0139 / +2.8716 | +2.4428 | [+1.4974, +3.6120] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +3.9556 / +5.3002 | +4.6279 | [+3.0722, +6.3528] | 1.7849 (0.8089) | WORSE |

## IPcq0q LS minus IPcq LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0605 / -0.0676 | -0.0641 | [-0.0959, -0.0385] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0171 / -0.0249 | -0.0210 | [-0.0386, -0.0072] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4502 / +0.5202 | +0.4852 | [+0.4024, +0.5633] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +7.5164 / +6.9340 | +7.2252 | [+6.4175, +8.0652] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +19.9170 / +18.6375 | +19.2773 | [+16.7071, +21.9258] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +29.4496 / +27.6494 | +28.5495 | [+24.0817, +33.3171] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +7.0122 / +6.4413 | +6.7268 | [+5.9549, +7.5296] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +19.1175 / +17.4743 | +18.2959 | [+15.6972, +21.0302] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +28.3007 / +26.0615 | +27.1811 | [+22.5538, +32.0977] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +31.1147 / +31.3390 | +31.2268 | [+26.5901, +36.0737] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +26.6448 / +27.5122 | +27.0785 | [+23.9947, +30.2345] | 1.7849 (0.8089) | WORSE |

## IPcqidq LS minus IPcq LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPcq0q_LS (0,1) | 0.0027 [0.0000, 0.0076] | 0.0008 [0.0000, 0.0032] | 0.8185 [0.7679, 0.8609] | 7.9791 [7.2745, 8.6636] | 24.3618 [21.7539, 27.0788] | 40.9918 [35.9533, 46.2980] | 8.1177 [7.4330, 8.7846] | 24.5016 [21.9191, 27.1820] | 41.0764 [36.0405, 46.3752] | 33.1728 [28.4708, 38.0689] | 31.1821 [28.1127, 34.3441] |
| IPcq_LS (0,1) | 0.0667 [0.0406, 0.0991] | 0.0218 [0.0072, 0.0402] | 0.3333 [0.2577, 0.4108] | 0.7539 [0.1875, 1.3875] | 5.0845 [3.1106, 7.5347] | 12.4422 [9.0261, 16.5555] | 1.3909 [0.8617, 1.9878] | 6.2058 [4.2480, 8.6130] | 13.8952 [10.7147, 17.9140] | 1.9460 [1.3404, 2.6999] | 4.1036 [3.0513, 5.3147] |
| IPcqidq_LS (0,1) | 0.0667 [0.0406, 0.0991] | 0.0218 [0.0072, 0.0402] | 0.3333 [0.2577, 0.4108] | 0.7539 [0.1875, 1.3875] | 5.0845 [3.1106, 7.5347] | 12.4422 [9.0261, 16.5555] | 1.3909 [0.8617, 1.9878] | 6.2058 [4.2480, 8.6130] | 13.8952 [10.7147, 17.9140] | 1.9460 [1.3404, 2.6999] | 4.1036 [3.0513, 5.3147] |
| IPcqq_LS (0,1) | 0.0311 [0.0141, 0.0566] | 0.0062 [0.0000, 0.0191] | 0.4330 [0.3442, 0.5192] | 1.7820 [1.1674, 2.4628] | 8.7538 [6.3979, 11.2949] | 18.7043 [13.9715, 23.4441] | 2.2824 [1.7001, 2.9206] | 9.5507 [7.2600, 12.0585] | 19.6057 [15.1279, 24.1498] | 3.6792 [2.7531, 4.7823] | 7.0881 [5.6825, 8.5718] |
| IPlq_LS (0,1) | 0.0347 [0.0182, 0.0529] | 0.0008 [0.0000, 0.0032] | 0.3030 [0.1967, 0.4152] | 0.6574 [0.1291, 1.3161] | 4.7780 [2.6629, 7.3615] | 11.9602 [7.8899, 16.6314] | 1.2804 [0.7847, 1.9100] | 5.6602 [3.6461, 8.1930] | 13.0241 [9.0007, 17.6474] | 1.5456 [1.0082, 2.1097] | 3.6575 [2.4266, 4.9034] |
| IPoq_LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
