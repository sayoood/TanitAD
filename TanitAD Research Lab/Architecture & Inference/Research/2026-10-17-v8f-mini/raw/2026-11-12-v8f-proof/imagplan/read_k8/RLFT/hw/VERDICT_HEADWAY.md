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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPlx_RLFT:IPox_RLFT", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPox_RLFT", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPcx_RLFT", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPlx_RLFT", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcx_RLFT:IPox_RLFT", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPksx_RLFT:IPkx_RLFT", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPix_RLFT", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPix_RLFT:IPox_RLFT", "seeds": [2, 3]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IPlx_RLFT minus IPox_RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPox_RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPcx_RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPlx_RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcx_RLFT minus IPox_RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPksx_RLFT minus IPkx_RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPix_RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPix_RLFT minus IPox_RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IPlx RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0907 / -0.1032 | -0.0970 | [-0.1350, -0.0607] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0592 / -0.0623 | -0.0607 | [-0.0870, -0.0359] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0966 / +0.0919 | +0.0942 | [+0.0504, +0.1478] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8914 / +0.8991 | +0.8953 | [+0.3687, +1.5677] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.4475 / +3.6327 | +3.5401 | [+1.5144, +6.1293] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.4302 / +6.5054 | +6.4678 | [+3.1185, +11.0871] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7652 / +0.7654 | +0.7653 | [+0.2685, +1.3954] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.5441 / +2.6107 | +2.5774 | [+0.7579, +4.9404] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.0609 / +4.5796 | +4.3202 | [+1.3121, +8.5705] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2740 / +0.2947 | +0.2844 | [+0.1341, +0.4774] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.7429 / +1.0586 | +0.9008 | [+0.3297, +1.7364] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0356 / -0.0338 | -0.0347 | [-0.0582, -0.0146] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0296 / -0.0218 | -0.0257 | [-0.0400, -0.0130] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1059 / +0.1090 | +0.1075 | [+0.0607, +0.1605] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6784 / +0.7893 | +0.7339 | [+0.3241, +1.2271] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.5450 / +3.1456 | +2.8453 | [+1.2581, +4.8471] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.9587 / +5.3257 | +5.1422 | [+2.5603, +8.4421] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.5914 / +0.7044 | +0.6479 | [+0.2647, +1.1028] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.0221 / +2.4981 | +2.2601 | [+0.9250, +3.9165] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.2752 / +4.4046 | +3.8399 | [+1.5409, +6.8340] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2193 / +0.2486 | +0.2339 | [+0.1089, +0.3842] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.5958 / +0.7436 | +0.6697 | [+0.3327, +1.1432] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLFT minus IPcx RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0160 / +0.0178 | +0.0169 | [+0.0035, +0.0316] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0016 / +0.0125 | +0.0070 | [-0.0008, +0.0183] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0202 / -0.0031 | -0.0117 | [-0.0344, +0.0051] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0751 / +0.0506 | -0.0122 | [-0.1219, +0.1299] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.3295 / +0.2291 | -0.0502 | [-0.5193, +0.5207] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.9345 / +0.3409 | -0.2968 | [-1.3775, +1.0208] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0618 / +0.0397 | -0.0110 | [-0.1070, +0.1162] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.2461 / +0.2377 | -0.0042 | [-0.4274, +0.5423] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.6354 / +0.7653 | +0.0649 | [-1.0822, +1.4488] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.3102 / -0.0611 | -0.1856 | [-0.4642, +0.0404] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.4275 / +0.0444 | -0.1915 | [-0.5866, +0.2036] | 1.7849 (0.8089) | OPEN |

## IPkx RLFT minus IPlx RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0552 / +0.0694 | +0.0623 | [+0.0352, +0.0906] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0296 / +0.0405 | +0.0350 | [+0.0155, +0.0562] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0093 / +0.0171 | +0.0132 | [-0.0073, +0.0336] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2130 / -0.1098 | -0.1614 | [-0.4093, +0.0752] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.9025 / -0.4871 | -0.6948 | [-1.6230, +0.1944] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.4715 / -1.1797 | -1.3256 | [-2.9823, +0.2014] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.1738 / -0.0610 | -0.1174 | [-0.3603, +0.1163] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.5219 / -0.1126 | -0.3172 | [-1.2389, +0.5546] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.7857 / -0.1749 | -0.4803 | [-2.1386, +1.1240] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0547 / -0.0461 | -0.0504 | [-0.2321, +0.1272] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1471 / -0.3151 | -0.2311 | [-0.7962, +0.2163] | 1.7849 (0.8089) | OPEN |

## IPcx RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0516 | -0.0516 | [-0.0809, -0.0253] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0312 / -0.0343 | -0.0327 | [-0.0485, -0.0191] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1262 / +0.1121 | +0.1192 | [+0.0678, +0.1763] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7535 / +0.7387 | +0.7461 | [+0.3626, +1.1979] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.8745 / +2.9165 | +2.8955 | [+1.4561, +4.6357] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.8931 / +4.9848 | +5.4390 | [+3.0582, +8.4202] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6531 / +0.6647 | +0.6589 | [+0.3007, +1.0777] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.2683 / +2.2604 | +2.2643 | [+1.0239, +3.7506] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.9105 / +3.6394 | +3.7749 | [+1.7474, +6.2952] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.5295 / +0.3097 | +0.4196 | [+0.1673, +0.7791] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0233 / +0.6991 | +0.8612 | [+0.4402, +1.3798] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPksx RLFT minus IPkx RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0854 / -0.1032 | -0.0943 | [-0.1334, -0.0591] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0296 / -0.0405 | -0.0350 | [-0.0562, -0.0155] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4704 / +0.4657 | +0.4681 | [+0.4084, +0.5280] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +5.7344 / +5.8960 | +5.8152 | [+5.1200, +6.4547] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +15.8176 / +16.1077 | +15.9626 | [+13.7892, +18.1011] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +24.4660 / +24.9761 | +24.7211 | [+21.0457, +28.5278] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +5.3064 / +5.3727 | +5.3396 | [+4.6770, +5.9352] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +14.7385 / +15.0461 | +14.8923 | [+12.7868, +16.9434] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +22.7579 / +23.3278 | +23.0429 | [+19.2472, +27.0386] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +20.1551 / +23.0033 | +21.5792 | [+17.5745, +25.6982] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +21.3839 / +24.0958 | +22.7399 | [+19.1824, +26.0262] | 1.7849 (0.8089) | WORSE |

## IPkx RLFT minus IPix RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / +0.0036 | +0.0053 | [+0.0009, +0.0113] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0031 / +0.0062 | +0.0047 | [+0.0009, +0.0093] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0140 / -0.0078 | -0.0109 | [-0.0215, -0.0036] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.3220 / -0.3675 | -0.3448 | [-0.5374, -0.1937] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -1.1755 / -1.3355 | -1.2555 | [-2.0349, -0.6699] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.9428 / -2.6063 | -2.2745 | [-3.9120, -1.1051] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.3186 / -0.3663 | -0.3425 | [-0.5352, -0.1910] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -1.1499 / -1.2985 | -1.2242 | [-1.9785, -0.6441] | 1.0764 (0.6913) | BETTER |
| gap error 6.0 s (m) | -1.7580 / -2.4400 | -2.0990 | [-3.7494, -0.9111] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.4950 / -0.4432 | -0.4691 | [-0.8525, -0.1614] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.7880 / -0.9634 | -0.8757 | [-1.4595, -0.3871] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPix RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0427 / -0.0374 | -0.0400 | [-0.0643, -0.0193] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0327 / -0.0280 | -0.0304 | [-0.0453, -0.0165] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1199 / +0.1168 | +0.1184 | [+0.0704, +0.1721] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0003 / +1.1569 | +1.0786 | [+0.5500, +1.6855] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.7205 / +4.4811 | +4.1008 | [+2.0266, +6.6085] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.9015 / +7.9320 | +7.4167 | [+3.7995, +12.0209] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.9100 / +1.0706 | +0.9903 | [+0.4875, +1.5676] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.1720 / +3.7966 | +3.4843 | [+1.6408, +5.7060] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.0331 / +6.8446 | +5.9388 | [+2.5655, +10.3154] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.7143 / +0.6918 | +0.7030 | [+0.3143, +1.1897] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.3838 / +1.7069 | +1.5454 | [+0.7664, +2.5121] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPcx_RLFT (2,3) | 0.0792 [0.0475, 0.1184] | 0.0280 [0.0123, 0.0463] | 0.2858 [0.2221, 0.3474] | 0.5062 [0.1099, 0.9547] | 3.6545 [2.2755, 5.2348] | 9.4020 [6.9403, 12.0736] | 1.1539 [0.7892, 1.5758] | 4.9609 [3.6712, 6.5100] | 10.9376 [8.7515, 13.3459] | 1.5836 [1.0960, 2.1310] | 3.0063 [2.3025, 3.7856] |
| IPix_RLFT (2,3) | 0.0907 [0.0564, 0.1300] | 0.0304 [0.0135, 0.0495] | 0.2850 [0.2194, 0.3472] | 0.8387 [0.3179, 1.4249] | 4.8598 [2.8781, 7.2072] | 11.3797 [7.5977, 16.1294] | 1.4854 [0.9797, 2.0807] | 6.1809 [4.3283, 8.4396] | 13.1015 [9.5281, 17.6019] | 1.8670 [1.3237, 2.4382] | 3.6905 [2.7440, 4.7827] |
| IPksx_RLFT (2,3) | 0.0018 [0.0000, 0.0070] | 0.0000 [0.0000, 0.0000] | 0.7422 [0.6755, 0.7956] | 6.3092 [5.5300, 7.0279] | 19.5670 [16.9388, 22.3307] | 33.8262 [28.8850, 39.2944] | 6.4825 [5.7387, 7.1821] | 19.8490 [17.3184, 22.5307] | 34.0454 [29.2340, 39.4449] | 22.9771 [18.8880, 27.0695] | 25.5547 [21.8412, 29.0164] |
| IPkx_RLFT (2,3) | 0.0961 [0.0607, 0.1348] | 0.0350 [0.0155, 0.0562] | 0.2741 [0.2078, 0.3360] | 0.4940 [0.0876, 0.9810] | 3.6043 [2.0869, 5.4613] | 9.1052 [6.2669, 12.4680] | 1.1429 [0.7517, 1.6095] | 4.9567 [3.5743, 6.6847] | 11.0026 [8.4091, 14.0415] | 1.3979 [0.9902, 1.8202] | 2.8148 [2.1306, 3.6018] |
| IPlx_RLFT (2,3) | 0.0338 [0.0156, 0.0546] | 0.0000 [0.0000, 0.0000] | 0.2609 [0.1971, 0.3239] | 0.6554 [0.1279, 1.3082] | 4.2992 [2.4239, 6.7757] | 10.4308 [7.0350, 15.0376] | 1.2603 [0.7648, 1.9028] | 5.2740 [3.4754, 7.6539] | 11.4829 [8.3508, 15.7445] | 1.4484 [1.0460, 1.8866] | 3.0459 [2.2230, 4.0960] |
| IPox_RLFT (2,3) | 0.1308 [0.0864, 0.1776] | 0.0607 [0.0359, 0.0870] | 0.1667 [0.1191, 0.2158] | -0.2399 [-0.3006, -0.1816] | 0.7591 [0.3016, 1.2441] | 3.9630 [2.3927, 5.5712] | 0.4950 [0.4454, 0.5507] | 2.6966 [2.4072, 3.0124] | 7.1627 [6.2266, 8.0914] | 1.1640 [0.7953, 1.5597] | 2.1451 [1.5350, 2.8354] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
