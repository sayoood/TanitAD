# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 26, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.10563170027848558, "A_hi": 0.15899261888352556, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPlx_RLAm:IPox_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPox_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPcx_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPlx_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcx_RLAm:IPox_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPix_RLAm", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPix_RLAm:IPox_RLAm", "seeds": [0, 1, 2, 3]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IPlx_RLAm minus IPox_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPox_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPcx_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPlx_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcx_RLAm minus IPox_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPix_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPix_RLAm minus IPox_RLAm over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IPlx RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0890 / -0.0996 / -0.0979 / -0.0996 | -0.0965 | [-0.1250, -0.0713] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0467 / -0.0358 / -0.0530 / -0.0561 | -0.0479 | [-0.0667, -0.0299] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0950 / +0.0779 / +0.1028 / +0.1090 | +0.0962 | [+0.0606, +0.1346] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8836 / +0.8864 / +0.8893 / +0.8938 | +0.8883 | [+0.4899, +1.3355] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.4885 / +3.4990 / +3.4049 / +3.5937 | +3.4965 | [+1.9794, +5.2436] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.2675 / +5.5460 / +6.2343 / +6.6207 | +6.1671 | [+3.6204, +9.3129] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7618 / +0.7646 / +0.7643 / +0.7663 | +0.7643 | [+0.3862, +1.1830] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.7349 / +2.3751 / +2.5461 / +2.6213 | +2.5694 | [+1.1882, +4.1510] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.6477 / +3.9711 / +4.1734 / +4.6934 | +4.3714 | [+2.0507, +7.1799] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3105 / +0.2775 / +0.2673 / +0.3049 | +0.2901 | [+0.1787, +0.4347] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.3058 / +0.7856 / +0.7055 / +1.0530 | +0.9625 | [+0.4803, +1.6606] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0605 / -0.0427 / -0.0463 | -0.0436 | [-0.0674, -0.0233] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0171 / -0.0062 / -0.0202 / -0.0140 | -0.0144 | [-0.0232, -0.0073] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0903 / +0.1090 / +0.1028 / +0.1184 | +0.1051 | [+0.0673, +0.1445] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7526 / +0.9954 / +0.6964 / +0.7744 | +0.8047 | [+0.4512, +1.1978] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.8397 / +4.0483 / +2.6306 / +3.0974 | +3.1540 | [+1.7405, +4.7327] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.8492 / +6.8658 / +4.7628 / +5.2333 | +5.4278 | [+3.1304, +8.2599] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6831 / +0.8976 / +0.6123 / +0.7028 | +0.7239 | [+0.3935, +1.0888] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.5922 / +3.1876 / +2.1640 / +2.4798 | +2.6059 | [+1.3884, +4.0170] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.0697 / +5.8419 / +3.3177 / +4.2345 | +4.3660 | [+2.2575, +6.9216] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1707 / +0.2482 / +0.2915 / +0.1968 | +0.2268 | [+0.1356, +0.3442] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.9708 / +1.1822 / +0.6724 / +0.6697 | +0.8738 | [+0.4611, +1.4053] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLAm minus IPcx RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / +0.0107 / +0.0196 / +0.0089 | +0.0142 | [+0.0044, +0.0252] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0062 / +0.0078 / +0.0078 / +0.0156 | +0.0093 | [+0.0035, +0.0163] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0031 / -0.0156 / -0.0156 / -0.0016 | -0.0074 | [-0.0222, +0.0075] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0463 / -0.0692 / -0.0627 / +0.0593 | -0.0066 | [-0.0912, +0.0823] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.1562 / -0.1646 / -0.2303 / +0.2694 | +0.0077 | [-0.3029, +0.3428] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.1744 / -0.5443 / -0.9862 / +0.2786 | -0.3566 | [-1.0063, +0.3752] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0721 / -0.0658 / -0.0494 / +0.0611 | +0.0045 | [-0.0788, +0.0889] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.2605 / -0.1311 / -0.0896 / +0.2824 | +0.0805 | [-0.2055, +0.3938] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.1738 / -0.4148 / -0.6576 / +0.7182 | -0.0451 | [-0.7467, +0.7198] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1757 / -0.2968 / -0.2334 / -0.0896 | -0.1989 | [-0.3550, -0.0785] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.1714 / -0.3571 / -0.3568 / +0.0550 | -0.2076 | [-0.4543, +0.0180] | 1.7849 (0.8089) | OPEN |

## IPkx RLAm minus IPlx RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0641 / +0.0391 / +0.0552 / +0.0534 | +0.0529 | [+0.0347, +0.0735] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0296 / +0.0296 / +0.0327 / +0.0421 | +0.0335 | [+0.0187, +0.0499] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0047 / +0.0312 / +0.0000 / +0.0093 | +0.0090 | [-0.0120, +0.0311] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1310 / +0.1090 / -0.1929 / -0.1194 | -0.0836 | [-0.2602, +0.1019] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.6488 / +0.5493 / -0.7743 / -0.4964 | -0.3426 | [-1.0436, +0.4664] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.4183 / +1.3198 / -1.4716 / -1.3875 | -0.7394 | [-2.1270, +0.9983] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0788 / +0.1330 / -0.1520 / -0.0635 | -0.0403 | [-0.2085, +0.1356] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.1427 / +0.8125 / -0.3821 / -0.1415 | +0.0365 | [-0.6277, +0.7857] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.5779 / +1.8708 / -0.8557 / -0.4589 | -0.0054 | [-1.3339, +1.6032] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1398 / -0.0293 / +0.0242 / -0.1081 | -0.0632 | [-0.2111, +0.0784] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.3350 / +0.3966 / -0.0331 / -0.3833 | -0.0887 | [-0.5330, +0.3614] | 1.7849 (0.8089) | OPEN |

## IPcx RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0427 / -0.0712 / -0.0623 / -0.0552 | -0.0578 | [-0.0840, -0.0360] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0234 / -0.0140 / -0.0280 / -0.0296 | -0.0238 | [-0.0347, -0.0141] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0872 / +0.1246 / +0.1184 / +0.1199 | +0.1125 | [+0.0744, +0.1525] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7063 / +1.0646 / +0.7591 / +0.7151 | +0.8113 | [+0.4731, +1.1983] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6835 / +4.2128 / +2.8609 / +2.8280 | +3.1463 | [+1.8107, +4.6932] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.0237 / +7.4102 / +5.7490 / +4.9546 | +5.7844 | [+3.5608, +8.3998] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6110 / +0.9634 / +0.6617 / +0.6417 | +0.7194 | [+0.3957, +1.0856] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.3317 / +3.3188 / +2.2535 / +2.1975 | +2.5254 | [+1.3845, +3.8880] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.8960 / +6.2567 / +3.9753 / +3.5163 | +4.4111 | [+2.3085, +6.9166] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3464 / +0.5450 / +0.5250 / +0.2864 | +0.4257 | [+0.2357, +0.6622] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.1422 / +1.5393 / +1.0292 / +0.6147 | +1.0814 | [+0.5966, +1.6354] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLAm minus IPix RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / +0.0000 / +0.0053 / +0.0018 | +0.0027 | [-0.0003, +0.0064] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0016 / +0.0016 / +0.0016 | +0.0016 | [+0.0003, +0.0033] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0031 / -0.0031 / -0.0062 / -0.0093 | -0.0055 | [-0.0105, -0.0009] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0140 / -0.0043 / +0.0070 / +0.0013 | +0.0045 | [-0.0191, +0.0291] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0318 / -0.0047 / +0.0492 / -0.0100 | +0.0166 | [-0.0753, +0.1111] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.1057 / +0.0198 / -0.1803 / -0.2200 | -0.1216 | [-0.2933, +0.0891] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0140 / -0.0043 / +0.0138 / +0.0013 | +0.0062 | [-0.0182, +0.0310] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0318 / -0.0016 / +0.0514 / -0.0044 | +0.0193 | [-0.0696, +0.1125] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0411 / +0.0212 / -0.1258 / -0.1923 | -0.0845 | [-0.2555, +0.1126] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0067 / -0.0325 / -0.0079 / -0.0999 | -0.0367 | [-0.1167, -0.0024] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.0356 / -0.1183 / -0.0123 / -0.0988 | -0.0662 | [-0.1381, -0.0134] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPix RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0605 / -0.0480 / -0.0480 | -0.0463 | [-0.0704, -0.0259] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0187 / -0.0078 / -0.0218 / -0.0156 | -0.0160 | [-0.0251, -0.0084] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0935 / +0.1121 / +0.1090 / +0.1277 | +0.1106 | [+0.0723, +0.1504] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7386 / +0.9996 / +0.6895 / +0.7731 | +0.8002 | [+0.4556, +1.1874] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.8078 / +4.0530 / +2.5814 / +3.1074 | +3.1374 | [+1.7667, +4.6927] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.9550 / +6.8460 / +4.9431 / +5.4533 | +5.5493 | [+3.2882, +8.2992] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6691 / +0.9019 / +0.5985 / +0.7015 | +0.7177 | [+0.3914, +1.0794] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.5604 / +3.1892 / +2.1125 / +2.4842 | +2.5866 | [+1.3987, +3.9853] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.1108 / +5.8207 / +3.4435 / +4.4268 | +4.4505 | [+2.3758, +6.9924] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1774 / +0.2807 / +0.2994 / +0.2967 | +0.2636 | [+0.1509, +0.3938] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0064 / +1.3005 / +0.6847 / +0.7685 | +0.9400 | [+0.5094, +1.4976] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPcx_RLAm (0,1,2,3) | 0.0765 [0.0552, 0.1028] | 0.0245 [0.0126, 0.0375] | 0.2796 [0.2331, 0.3233] | 0.5856 [0.2434, 0.9721] | 3.9006 [2.6892, 5.3434] | 10.1602 [8.0083, 12.8472] | 1.2069 [0.8899, 1.5804] | 5.1804 [4.0159, 6.5595] | 11.7170 [9.5773, 14.3154] | 1.5751 [1.2325, 1.9940] | 3.1370 [2.5027, 3.8076] |
| IPix_RLAm (0,1,2,3) | 0.0881 [0.0656, 0.1136] | 0.0323 [0.0177, 0.0490] | 0.2776 [0.2339, 0.3225] | 0.5746 [0.2347, 0.9556] | 3.8917 [2.6402, 5.3294] | 9.9252 [7.6368, 12.6048] | 1.2052 [0.8816, 1.5700] | 5.2417 [4.0548, 6.6419] | 11.7564 [9.5861, 14.4164] | 1.4130 [1.1200, 1.7543] | 2.9956 [2.4352, 3.6308] |
| IPkx_RLAm (0,1,2,3) | 0.0907 [0.0678, 0.1164] | 0.0339 [0.0187, 0.0507] | 0.2722 [0.2279, 0.3171] | 0.5791 [0.2352, 0.9604] | 3.9083 [2.6598, 5.3582] | 9.8036 [7.5001, 12.5873] | 1.2114 [0.8828, 1.5761] | 5.2610 [4.0452, 6.6461] | 11.6719 [9.4751, 14.3547] | 1.3763 [1.0875, 1.7157] | 2.9294 [2.3856, 3.5555] |
| IPlx_RLAm (0,1,2,3) | 0.0378 [0.0212, 0.0576] | 0.0004 [0.0000, 0.0018] | 0.2632 [0.2186, 0.3074] | 0.6627 [0.2661, 1.0962] | 4.2508 [2.8000, 5.9189] | 10.5430 [7.9523, 13.4618] | 1.2518 [0.8740, 1.6663] | 5.2244 [3.8318, 6.8247] | 11.6774 [9.2969, 14.4884] | 1.4395 [1.1543, 1.7753] | 3.0181 [2.4287, 3.7726] |
| IPox_RLAm (0,1,2,3) | 0.1343 [0.0993, 0.1727] | 0.0483 [0.0300, 0.0674] | 0.1671 [0.1317, 0.2030] | -0.2256 [-0.2652, -0.1892] | 0.7543 [0.3440, 1.1418] | 4.3759 [3.2451, 5.5775] | 0.4875 [0.4494, 0.5255] | 2.6551 [2.4528, 2.8628] | 7.3060 [6.5324, 8.2349] | 1.1495 [0.8658, 1.4616] | 2.0556 [1.6548, 2.5183] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
