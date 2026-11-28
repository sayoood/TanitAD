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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQE:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQE:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQE:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQE:LSMC", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQE:LSQC", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSQE minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQE minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQE minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQE minus LSMC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQE minus LSQC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSQE minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0747 / -0.0569 | -0.0658 | [-0.1055, -0.0269] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0202 / -0.0452 | -0.0327 | [-0.0584, -0.0118] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1542 / +0.2165 | +0.1854 | [+0.1147, +0.2541] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0920 / +1.2697 | +1.1808 | [+0.7869, +1.6030] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.7153 / +4.3552 | +4.0352 | [+2.6379, +5.4558] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.9959 / +8.7744 | +7.8851 | [+5.0584, +10.8955] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.8763 / +0.9365 | +0.9064 | [+0.5670, +1.2628] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.9595 / +3.7273 | +3.3434 | [+2.0606, +4.7592] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.7937 / +6.4255 | +6.1096 | [+3.6896, +8.7329] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2491 / +0.4807 | +0.3649 | [+0.1857, +0.6242] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.9427 / +1.4746 | +1.2086 | [+0.7678, +1.7678] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQE minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0338 / -0.0178 | -0.0258 | [-0.0462, -0.0081] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0078 / -0.0156 | -0.0117 | [-0.0235, -0.0024] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0670 / +0.1059 | +0.0864 | [+0.0466, +0.1260] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7224 / +0.5560 | +0.6392 | [+0.4303, +0.8637] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.0642 / +1.9789 | +2.0216 | [+1.3310, +2.7873] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.5894 / +4.0437 | +3.8165 | [+2.3760, +5.3810] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.5738 / +0.3879 | +0.4808 | [+0.2709, +0.7033] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.8590 / +1.6054 | +1.7322 | [+1.0665, +2.4844] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.2837 / +3.3621 | +3.3229 | [+1.9231, +4.8605] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.2270 / +0.1124 | +0.1697 | [+0.0809, +0.2837] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8056 / +0.5831 | +0.6943 | [+0.4196, +0.9664] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQE minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0267 | +0.0133 | [-0.0103, +0.0411] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0109 / +0.0125 | +0.0117 | [+0.0029, +0.0230] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0810 / +0.1636 | +0.1223 | [+0.0572, +0.1882] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6561 / +0.8229 | +0.7395 | [+0.5125, +0.9783] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.3664 / +3.0278 | +2.6971 | [+1.7303, +3.7180] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.9094 / +5.3393 | +4.6244 | [+2.6513, +6.6312] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.5501 / +0.6430 | +0.5966 | [+0.4000, +0.7949] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.2496 / +3.0100 | +2.6298 | [+1.6519, +3.6765] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.8844 / +5.3901 | +4.6372 | [+2.7310, +6.6273] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2079 / +0.4409 | +0.3244 | [+0.1460, +0.5854] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8046 / +1.3844 | +1.0945 | [+0.6490, +1.6689] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQE minus LSMC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / +0.0267 | +0.0053 | [-0.0259, +0.0401] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0078 / +0.0016 | +0.0047 | [-0.0040, +0.0142] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0031 / -0.0156 | -0.0093 | [-0.0368, +0.0171] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0614 / -0.5135 | -0.2261 | [-0.6319, +0.1570] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0543 / -0.8354 | -0.3905 | [-1.2194, +0.4485] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.4810 / -0.8916 | -0.2053 | [-1.6199, +1.0246] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0745 / -0.5072 | -0.2164 | [-0.6173, +0.1611] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0710 / -0.7825 | -0.3557 | [-1.1334, +0.4585] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.4258 / -0.8634 | -0.2188 | [-1.5450, +0.9807] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0102 / -0.3727 | -0.1813 | [-0.4602, +0.0702] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1949 / -0.8075 | -0.5012 | [-1.0398, -0.0645] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQE minus LSQC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / +0.0036 | -0.0062 | [-0.0251, +0.0167] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0047 / -0.0109 | -0.0078 | [-0.0180, -0.0001] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0047 / +0.0421 | +0.0187 | [-0.0153, +0.0564] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1895 / +0.0298 | +0.1096 | [-0.0726, +0.2869] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.4145 / +0.4223 | +0.4184 | [-0.0863, +0.9071] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.3631 / +0.8491 | +0.6061 | [-0.2943, +1.5370] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.1806 / -0.0168 | +0.0819 | [-0.1095, +0.2668] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.3431 / +0.2008 | +0.2719 | [-0.2381, +0.7673] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.1725 / +0.5275 | +0.3500 | [-0.5869, +1.2827] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.1565 / -0.0121 | +0.0722 | [-0.0493, +0.2177] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.3691 / +0.1490 | +0.2590 | [+0.0344, +0.5231] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| LSMC (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4058 [0.3276, 0.4795] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5297 [10.7494, 16.4204] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6582 [12.0491, 17.4352] | 1.7826 [1.3123, 2.2538] | 4.1701 [3.3443, 5.0752] |
| LSQC (0,1) | 0.0649 [0.0384, 0.0985] | 0.0195 [0.0078, 0.0364] | 0.3777 [0.2840, 0.4724] | 0.8362 [0.4309, 1.3170] | 4.8035 [3.4585, 6.2442] | 12.7183 [9.6831, 15.6759] | 1.3408 [0.9562, 1.7937] | 5.8570 [4.6629, 7.2501] | 14.0894 [11.3615, 16.9036] | 1.5291 [1.0824, 1.9879] | 3.4098 [2.5781, 4.2949] |
| LSQE (0,1) | 0.0587 [0.0279, 0.0979] | 0.0117 [0.0029, 0.0230] | 0.3964 [0.3181, 0.4722] | 0.9459 [0.5697, 1.3420] | 5.2219 [3.9272, 6.5447] | 13.3244 [10.3415, 16.3360] | 1.4227 [1.0820, 1.7778] | 6.1289 [4.9571, 7.3833] | 14.4394 [11.7160, 17.2551] | 1.6013 [1.1031, 2.1354] | 3.6688 [2.7423, 4.6699] |
| LSRH (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| LSRL (0,1) | 0.0454 [0.0238, 0.0728] | 0.0000 [0.0000, 0.0000] | 0.2741 [0.1700, 0.3895] | 0.2064 [-0.0545, 0.5267] | 2.5248 [1.6354, 3.5112] | 8.7001 [6.1813, 11.2690] | 0.8261 [0.5991, 1.1154] | 3.4991 [2.7134, 4.3601] | 9.8022 [7.5079, 12.2263] | 1.2769 [0.7638, 1.8367] | 2.5743 [1.5172, 3.8082] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
