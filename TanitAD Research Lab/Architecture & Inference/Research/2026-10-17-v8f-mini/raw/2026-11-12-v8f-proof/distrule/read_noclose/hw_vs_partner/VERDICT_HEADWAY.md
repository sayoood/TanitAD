# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 22, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSRZ:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDRZ:LSDRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQZ:LSQC", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQZ:LSDQC", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSRZ minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDRZ minus LSDRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQZ minus LSQC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQZ minus LSDQC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSRZ minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0018 | +0.0018 | [-0.0116, +0.0147] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0047 | +0.0023 | [+0.0000, +0.0082] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0187 / -0.0031 | -0.0109 | [-0.0296, +0.0095] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0237 / -0.0455 | -0.0346 | [-0.0753, -0.0024] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.3244 / -0.1417 | -0.2330 | [-0.5793, +0.1263] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.4972 / -0.3837 | -0.4404 | [-0.9905, +0.0961] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0206 / -0.0440 | -0.0323 | [-0.0733, -0.0014] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.2342 / -0.1418 | -0.1880 | [-0.5199, +0.1408] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.4164 / -0.2896 | -0.3530 | [-0.8705, +0.1656] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0025 / -0.0162 | -0.0094 | [-0.0559, +0.0245] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0470 / -0.0127 | -0.0298 | [-0.1208, +0.0681] | 1.7849 (0.8089) | OPEN |

## LSDRZ minus LSDRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / -0.0053 | -0.0036 | [-0.0108, +0.0010] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0016 / -0.0093 | -0.0055 | [-0.0154, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0016 / -0.0062 | -0.0039 | [-0.0155, +0.0078] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0162 / -0.0404 | -0.0283 | [-0.0689, +0.0009] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.1017 / -0.4129 | -0.2573 | [-0.6292, +0.0808] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.1104 / -0.6160 | -0.3632 | [-1.1014, +0.2415] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0056 / -0.0419 | -0.0238 | [-0.0659, +0.0068] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0961 / -0.3561 | -0.2261 | [-0.5508, +0.0672] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0563 / -0.5518 | -0.3040 | [-0.9963, +0.2826] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0014 / +0.0071 | +0.0028 | [-0.0042, +0.0157] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0076 / -0.0184 | -0.0130 | [-0.0611, +0.0350] | 1.7849 (0.8089) | OPEN |

## LSQZ minus LSQC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / +0.0053 | +0.0009 | [-0.0124, +0.0142] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0016 / +0.0062 | +0.0023 | [-0.0052, +0.0115] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0078 / -0.0171 | -0.0125 | [-0.0329, +0.0084] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0470 / -0.0954 | -0.0712 | [-0.1314, -0.0181] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -0.3812 / -0.4208 | -0.4010 | [-0.7075, -0.0998] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.1112 / -0.9011 | -1.0062 | [-1.6233, -0.3998] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0367 / -0.0723 | -0.0545 | [-0.1083, -0.0066] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.4166 / -0.3853 | -0.4010 | [-0.7394, -0.0950] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -1.1450 / -0.8787 | -1.0119 | [-1.6810, -0.3597] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.1601 / -0.0794 | +0.0403 | [-0.1151, +0.2311] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.2687 / -0.1103 | +0.0792 | [-0.2130, +0.4248] | 1.7849 (0.8089) | OPEN |

## LSDQZ minus LSDQC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / +0.0071 | -0.0044 | [-0.0256, +0.0134] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0016 / +0.0047 | +0.0016 | [-0.0053, +0.0081] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0062 / -0.0078 | -0.0070 | [-0.0246, +0.0105] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0417 / -0.0458 | -0.0437 | [-0.1034, +0.0157] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0824 / -0.6422 | -0.3623 | [-0.9096, +0.1697] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.2419 / -1.5965 | -0.6773 | [-2.1308, +0.7242] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0192 / -0.0503 | -0.0348 | [-0.1004, +0.0206] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.1824 / -0.5896 | -0.3860 | [-0.8599, +0.0769] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.1414 / -1.4265 | -0.6425 | [-1.9831, +0.5571] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0138 / +0.0231 | +0.0184 | [-0.0386, +0.0822] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0308 / -0.0042 | +0.0133 | [-0.1830, +0.2230] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LSDQC (0,1) | 0.0658 [0.0389, 0.0988] | 0.0179 [0.0047, 0.0379] | 0.4104 [0.3222, 0.4975] | 0.9297 [0.5356, 1.3644] | 5.2680 [3.7314, 6.8447] | 13.2754 [10.2952, 16.3070] | 1.3863 [1.0227, 1.7990] | 6.2113 [4.8507, 7.6690] | 14.4393 [11.7734, 17.3270] | 1.6424 [0.9985, 2.3604] | 3.3313 [2.3127, 4.4132] |
| LSDQZ (0,1) | 0.0614 [0.0318, 0.1007] | 0.0195 [0.0048, 0.0418] | 0.4034 [0.3173, 0.4888] | 0.8860 [0.5179, 1.2950] | 4.9057 [3.3476, 6.5154] | 12.5981 [9.3383, 15.5273] | 1.3516 [1.0134, 1.7377] | 5.8253 [4.4966, 7.2675] | 13.7967 [11.1344, 16.3311] | 1.6608 [1.0197, 2.3732] | 3.3446 [2.3064, 4.4521] |
| LSDRH (0,1) | 0.1174 [0.0734, 0.1670] | 0.0335 [0.0072, 0.0708] | 0.3255 [0.2493, 0.4022] | 0.2189 [-0.0074, 0.5021] | 3.0880 [1.9867, 4.2275] | 9.9390 [7.5229, 12.1905] | 0.8106 [0.6022, 1.0655] | 4.2859 [3.4176, 5.2197] | 11.4028 [9.5253, 13.4774] | 1.4992 [0.8581, 2.2308] | 2.7695 [1.8682, 3.7196] |
| LSDRZ (0,1) | 0.1139 [0.0709, 0.1612] | 0.0280 [0.0065, 0.0608] | 0.3217 [0.2441, 0.4007] | 0.1906 [-0.0228, 0.4479] | 2.8307 [1.6862, 4.0351] | 9.5758 [7.1212, 11.8803] | 0.7868 [0.5898, 1.0280] | 4.0598 [3.1581, 5.0082] | 11.0988 [9.1744, 13.0523] | 1.5020 [0.8594, 2.2291] | 2.7564 [1.8403, 3.7299] |
| LSQC (0,1) | 0.0649 [0.0384, 0.0985] | 0.0195 [0.0078, 0.0364] | 0.3777 [0.2840, 0.4724] | 0.8362 [0.4309, 1.3170] | 4.8035 [3.4585, 6.2442] | 12.7183 [9.6831, 15.6759] | 1.3408 [0.9562, 1.7937] | 5.8570 [4.6629, 7.2501] | 14.0894 [11.3615, 16.9036] | 1.5291 [1.0824, 1.9879] | 3.4098 [2.5781, 4.2949] |
| LSQZ (0,1) | 0.0658 [0.0378, 0.1040] | 0.0218 [0.0067, 0.0447] | 0.3653 [0.2687, 0.4650] | 0.7650 [0.3845, 1.2240] | 4.4025 [3.1204, 5.8402] | 11.7121 [8.9593, 14.4143] | 1.2863 [0.9192, 1.7256] | 5.4560 [4.3283, 6.7354] | 13.0775 [10.7193, 15.4900] | 1.5695 [1.0548, 2.1342] | 3.4890 [2.5209, 4.5788] |
| LSRH (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| LSRZ (0,1) | 0.0863 [0.0532, 0.1267] | 0.0257 [0.0089, 0.0493] | 0.2991 [0.2188, 0.3845] | 0.2720 [-0.0115, 0.6056] | 2.9674 [2.0333, 4.0448] | 9.0675 [6.8195, 11.3895] | 0.9095 [0.6568, 1.2228] | 4.2088 [3.3643, 5.1750] | 10.7634 [8.9089, 12.6988] | 1.4222 [0.9760, 1.8918] | 2.9447 [2.1333, 3.8503] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
