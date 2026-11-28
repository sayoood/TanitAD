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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPlx_RLFT:IPox_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPox_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPcx_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPlx_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcx_RLFT:IPox_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPix_RLFT", "seeds": [0, 1, 2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPix_RLFT:IPox_RLFT", "seeds": [0, 1, 2, 3]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IPlx_RLFT minus IPox_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPox_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPcx_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPlx_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcx_RLFT minus IPox_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPix_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPix_RLFT minus IPox_RLFT over seeds [0, 1, 2, 3] of [0, 1, 2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IPlx RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0694 / -0.1068 / -0.0907 / -0.1032 | -0.0925 | [-0.1234, -0.0652] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0421 / -0.0421 / -0.0592 / -0.0623 | -0.0514 | [-0.0707, -0.0323] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0919 / +0.0997 / +0.0966 / +0.0919 | +0.0950 | [+0.0629, +0.1310] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8777 / +0.8873 / +0.8914 / +0.8991 | +0.8889 | [+0.4914, +1.3366] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.4035 / +3.5187 / +3.4475 / +3.6327 | +3.5006 | [+1.9820, +5.2634] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.7205 / +5.9319 / +6.4302 / +6.5054 | +6.1470 | [+3.6592, +9.2469] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7531 / +0.7649 / +0.7652 / +0.7654 | +0.7621 | [+0.3848, +1.1814] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.7589 / +2.3898 / +2.5441 / +2.6107 | +2.5759 | [+1.1941, +4.1443] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.4360 / +4.4319 / +4.0609 / +4.5796 | +4.3771 | [+2.0523, +7.1561] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3193 / +0.2827 / +0.2740 / +0.2947 | +0.2927 | [+0.1790, +0.4425] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0867 / +0.8058 / +0.7429 / +1.0586 | +0.9235 | [+0.4823, +1.5057] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0231 / -0.0623 / -0.0356 / -0.0338 | -0.0387 | [-0.0634, -0.0196] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0125 / -0.0109 / -0.0296 / -0.0218 | -0.0187 | [-0.0296, -0.0097] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0841 / +0.1184 / +0.1059 / +0.1090 | +0.1044 | [+0.0678, +0.1425] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7402 / +0.9881 / +0.6784 / +0.7893 | +0.7990 | [+0.4531, +1.1846] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.7394 / +3.9265 / +2.5450 / +3.1456 | +3.0891 | [+1.7280, +4.6504] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.4964 / +6.8334 / +4.9587 / +5.3257 | +5.4035 | [+3.1789, +8.1513] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6640 / +0.9020 / +0.5914 / +0.7044 | +0.7154 | [+0.3922, +1.0812] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.4898 / +3.1203 / +2.0221 / +2.4981 | +2.5326 | [+1.3452, +3.9296] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.9481 / +5.8209 / +3.2752 / +4.4046 | +4.3622 | [+2.2758, +6.8875] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2004 / +0.2442 / +0.2193 / +0.2486 | +0.2281 | [+0.1409, +0.3288] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8252 / +1.1935 / +0.5958 / +0.7436 | +0.8395 | [+0.4611, +1.2953] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLFT minus IPcx RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / +0.0125 / +0.0160 / +0.0178 | +0.0129 | [+0.0038, +0.0236] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0078 / +0.0062 / +0.0016 / +0.0125 | +0.0070 | [+0.0013, +0.0138] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0016 / -0.0187 / -0.0202 / -0.0031 | -0.0101 | [-0.0248, +0.0037] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0237 / -0.0583 / -0.0751 / +0.0506 | -0.0148 | [-0.0895, +0.0719] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0547 / -0.2661 / -0.3295 / +0.2291 | -0.0779 | [-0.3705, +0.2700] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.2090 / -0.7946 / -0.9345 / +0.3409 | -0.3993 | [-1.0489, +0.3714] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0504 / -0.0476 / -0.0618 / +0.0397 | -0.0048 | [-0.0736, +0.0734] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.1166 / -0.1565 / -0.2461 / +0.2377 | -0.0121 | [-0.2733, +0.3035] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0460 / -0.5978 / -0.6354 / +0.7653 | -0.1055 | [-0.7851, +0.7152] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1318 / -0.2702 / -0.3102 / -0.0611 | -0.1933 | [-0.3782, -0.0501] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.1881 / -0.3045 / -0.4275 / +0.0444 | -0.2189 | [-0.4743, +0.0143] | 1.7849 (0.8089) | OPEN |

## IPkx RLFT minus IPlx RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0463 / +0.0445 / +0.0552 / +0.0694 | +0.0538 | [+0.0349, +0.0746] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0296 / +0.0312 / +0.0296 / +0.0405 | +0.0327 | [+0.0186, +0.0474] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0078 / +0.0187 / +0.0093 / +0.0171 | +0.0093 | [-0.0090, +0.0265] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1375 / +0.1008 / -0.2130 / -0.1098 | -0.0899 | [-0.2703, +0.1000] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.6641 / +0.4078 / -0.9025 / -0.4871 | -0.4115 | [-1.1171, +0.3593] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.2241 / +0.9015 / -1.4715 / -1.1797 | -0.7435 | [-2.0516, +0.7428] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0891 / +0.1371 / -0.1738 / -0.0610 | -0.0467 | [-0.2226, +0.1360] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.2691 / +0.7305 / -0.5219 / -0.1126 | -0.0433 | [-0.7273, +0.7025] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.4880 / +1.3891 / -0.7857 / -0.1749 | -0.0149 | [-1.2411, +1.3408] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1189 / -0.0386 / -0.0547 / -0.0461 | -0.0646 | [-0.2006, +0.0543] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.2615 / +0.3877 / -0.1471 / -0.3151 | -0.0840 | [-0.5189, +0.3728] | 1.7849 (0.8089) | OPEN |

## IPcx RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0747 / -0.0516 / -0.0516 | -0.0516 | [-0.0793, -0.0286] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0202 / -0.0171 / -0.0312 / -0.0343 | -0.0257 | [-0.0372, -0.0152] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0826 / +0.1371 / +0.1262 / +0.1121 | +0.1145 | [+0.0746, +0.1566] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7165 / +1.0464 / +0.7535 / +0.7387 | +0.8138 | [+0.4809, +1.1946] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6847 / +4.1925 / +2.8745 / +2.9165 | +3.1670 | [+1.8600, +4.7076] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.7054 / +7.6280 / +5.8931 / +4.9848 | +5.8028 | [+3.5746, +8.5243] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6136 / +0.9496 / +0.6531 / +0.6647 | +0.7203 | [+0.4064, +1.0813] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.3732 / +3.2768 / +2.2683 / +2.2604 | +2.5447 | [+1.4232, +3.8836] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.9021 / +6.4187 / +3.9105 / +3.6394 | +4.4677 | [+2.3724, +6.9464] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3322 / +0.5144 / +0.5295 / +0.3097 | +0.4214 | [+0.2388, +0.6537] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0132 / +1.4980 / +1.0233 / +0.6991 | +1.0584 | [+0.6232, +1.5449] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLFT minus IPix RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0071 / +0.0071 / +0.0036 | +0.0044 | [+0.0008, +0.0091] | 0.0463 (0.0214) | MIXED-SIGN |
| C contact rate | +0.0047 / +0.0047 / +0.0031 / +0.0062 | +0.0047 | [+0.0016, +0.0086] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0140 / -0.0047 / -0.0140 / -0.0078 | -0.0101 | [-0.0170, -0.0046] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2720 / -0.2640 / -0.3220 / -0.3675 | -0.3064 | [-0.4281, -0.2030] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -1.0547 / -0.9921 / -1.1755 / -1.3355 | -1.1395 | [-1.6092, -0.7380] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.7048 / -1.7383 / -1.9428 / -2.6063 | -1.9981 | [-2.9689, -1.2222] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.2675 / -0.2640 / -0.3186 / -0.3663 | -0.3041 | [-0.4272, -0.2005] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -1.0076 / -0.9651 / -1.1499 / -1.2985 | -1.1053 | [-1.5682, -0.7077] | 1.0764 (0.6913) | BETTER |
| gap error 6.0 s (m) | -1.5908 / -1.6676 / -1.7580 / -2.4400 | -1.8641 | [-2.8171, -1.0902] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.3014 / -0.3188 / -0.4950 / -0.4432 | -0.3896 | [-0.6440, -0.1967] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.7456 / -0.6782 / -0.7880 / -0.9634 | -0.7938 | [-1.1682, -0.4752] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPix RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0231 / -0.0694 / -0.0427 / -0.0374 | -0.0431 | [-0.0696, -0.0222] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0171 / -0.0156 / -0.0327 / -0.0280 | -0.0234 | [-0.0356, -0.0123] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0981 / +0.1231 / +0.1199 / +0.1168 | +0.1145 | [+0.0783, +0.1521] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0122 / +1.2521 / +1.0003 / +1.1569 | +1.1054 | [+0.6890, +1.5472] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.7942 / +4.9185 / +3.7205 / +4.4811 | +4.2286 | [+2.5948, +6.0189] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.2012 / +8.5717 / +6.9015 / +7.9320 | +7.4016 | [+4.6545, +10.7462] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.9315 / +1.1660 / +0.9100 / +1.0706 | +1.0195 | [+0.6266, +1.4334] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.4975 / +4.0854 / +3.1720 / +3.7966 | +3.6379 | [+2.1766, +5.2206] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.5388 / +7.4885 / +5.0331 / +6.8446 | +6.2263 | [+3.5821, +9.2010] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.5018 / +0.5630 / +0.7143 / +0.6918 | +0.6177 | [+0.3685, +0.9356] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.5708 / +1.8717 / +1.3838 / +1.7069 | +1.6333 | [+1.0456, +2.2837] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPcx_RLFT (0,1,2,3) | 0.0761 [0.0533, 0.1026] | 0.0257 [0.0134, 0.0380] | 0.2893 [0.2433, 0.3353] | 0.5878 [0.2482, 0.9750] | 3.9246 [2.7082, 5.3371] | 10.2433 [8.0574, 12.8871] | 1.2096 [0.9029, 1.5783] | 5.2221 [4.0586, 6.5915] | 11.8383 [9.7481, 14.3783] | 1.5312 [1.2014, 1.9220] | 3.1554 [2.5516, 3.7893] |
| IPix_RLFT (0,1,2,3) | 0.0845 [0.0610, 0.1113] | 0.0280 [0.0159, 0.0412] | 0.2893 [0.2451, 0.3336] | 0.8794 [0.4702, 1.3179] | 4.9862 [3.5091, 6.6243] | 11.8420 [9.0874, 15.0370] | 1.5089 [1.1176, 1.9226] | 6.3153 [4.8255, 7.9053] | 13.5969 [10.9659, 16.6796] | 1.7275 [1.3673, 2.1724] | 3.7302 [3.0295, 4.4624] |
| IPkx_RLFT (0,1,2,3) | 0.0890 [0.0645, 0.1169] | 0.0327 [0.0186, 0.0474] | 0.2792 [0.2344, 0.3242] | 0.5730 [0.2382, 0.9553] | 3.8467 [2.6111, 5.2736] | 9.8440 [7.5207, 12.5502] | 1.2048 [0.8865, 1.5712] | 5.2100 [4.0216, 6.5906] | 11.7328 [9.5541, 14.3937] | 1.3379 [1.0620, 1.6424] | 2.9365 [2.3678, 3.5321] |
| IPlx_RLFT (0,1,2,3) | 0.0351 [0.0221, 0.0500] | 0.0000 [0.0000, 0.0000] | 0.2699 [0.2244, 0.3160] | 0.6629 [0.2684, 1.0958] | 4.2582 [2.8087, 5.9291] | 10.5874 [7.9955, 13.5718] | 1.2515 [0.8740, 1.6681] | 5.2533 [3.8487, 6.8437] | 11.7477 [9.3299, 14.5879] | 1.4024 [1.1280, 1.7185] | 3.0204 [2.4317, 3.7015] |
| IPox_RLFT (0,1,2,3) | 0.1277 [0.0931, 0.1663] | 0.0514 [0.0323, 0.0707] | 0.1748 [0.1399, 0.2101] | -0.2260 [-0.2668, -0.1880] | 0.7576 [0.3585, 1.1267] | 4.4404 [3.2460, 5.7031] | 0.4894 [0.4522, 0.5277] | 2.6774 [2.4554, 2.8968] | 7.3706 [6.6689, 8.1312] | 1.1098 [0.8400, 1.4028] | 2.0969 [1.6720, 2.5587] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
