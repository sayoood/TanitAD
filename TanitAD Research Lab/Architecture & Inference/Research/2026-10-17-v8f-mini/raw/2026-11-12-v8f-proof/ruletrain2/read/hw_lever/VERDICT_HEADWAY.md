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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOW:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOW4:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFTb:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOW4:GOW", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GOW minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOW4 minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GFTb minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOW4 minus GOW over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GOW minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0267 | -0.0276 | [-0.0647, +0.0089] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0249 / -0.0343 | -0.0296 | [-0.0534, -0.0104] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.3520 / +0.3364 | +0.3442 | [+0.2719, +0.4132] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.3001 / +2.9978 | +2.6490 | [+1.6060, +3.9451] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +10.3329 / +10.6875 | +10.5102 | [+7.1965, +14.2886] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +19.8409 / +19.8957 | +19.8683 | [+13.5833, +26.4681] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.9619 / +2.5582 | +2.2600 | [+1.2427, +3.5007] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +9.1058 / +9.5038 | +9.3048 | [+6.2102, +12.8880] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +18.2430 / +17.8699 | +18.0564 | [+12.1608, +24.1581] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +2.5259 / +3.2287 | +2.8773 | [+1.9454, +3.9634] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +3.8702 / +3.2787 | +3.5745 | [+2.3126, +4.9555] | 1.7849 (0.8089) | WORSE |

## GOW4 minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0356 / -0.0694 | -0.0525 | [-0.0973, -0.0116] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0234 / -0.0343 | -0.0288 | [-0.0533, -0.0105] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4829 / +0.3879 | +0.4354 | [+0.3446, +0.5260] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +4.2222 / +3.7534 | +3.9878 | [+2.6450, +5.4360] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +15.5927 / +11.2236 | +13.4081 | [+8.7964, +18.1640] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +27.5906 / +20.2460 | +23.9183 | [+16.3264, +32.2407] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +3.8307 / +3.2878 | +3.5592 | [+2.2254, +5.0241] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +14.1974 / +10.1310 | +12.1642 | [+7.8874, +16.7066] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +25.6352 / +17.9412 | +21.7882 | [+14.3563, +30.0215] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +5.7104 / +5.1463 | +5.4284 | [+4.0157, +6.8638] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +8.1631 / +4.5884 | +6.3758 | [+3.8310, +9.0083] | 1.7849 (0.8089) | WORSE |

## GFTb minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0036 | +0.0018 | [-0.0147, +0.0185] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0125 / +0.0031 | -0.0047 | [-0.0199, +0.0081] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0483 / +0.0047 | +0.0265 | [-0.0054, +0.0723] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1090 / +0.0367 | -0.0361 | [-0.1300, +0.0453] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.6828 / +0.1919 | +0.4373 | [+0.0599, +0.9951] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.5168 / +0.2247 | +1.3708 | [-0.1896, +3.5977] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0348 / -0.0050 | +0.0149 | [-0.0116, +0.0578] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.2566 / +0.1823 | +0.2195 | [-0.0462, +0.6192] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +1.4834 / +0.0912 | +0.7873 | [-0.2242, +2.5768] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1727 / +0.1407 | -0.0160 | [-0.2274, +0.2639] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.2648 / +0.0949 | +0.1799 | [+0.0126, +0.3770] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GOW4 minus GOW

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / -0.0427 | -0.0249 | [-0.0584, +0.0026] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0000 | +0.0008 | [-0.0053, +0.0069] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.1308 / +0.0514 | +0.0911 | [+0.0286, +0.1502] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.9221 / +0.7556 | +1.3388 | [+0.5722, +2.2686] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +5.2599 / +0.5361 | +2.8980 | [-0.0603, +5.8981] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +7.7497 / +0.3503 | +4.0500 | [-0.8978, +8.8262] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +1.8688 / +0.7296 | +1.2992 | [+0.5483, +2.2114] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +5.0916 / +0.6272 | +2.8594 | [+0.0511, +5.7345] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +7.3922 / +0.0713 | +3.7318 | [-1.1214, +8.4856] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +3.1845 / +1.9177 | +2.5511 | [+1.3422, +3.7163] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +4.2929 / +1.3097 | +2.8013 | [+0.7430, +4.7403] | 1.7849 (0.8089) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT (0,1) | 0.1121 [0.0711, 0.1570] | 0.0405 [0.0212, 0.0648] | 0.1931 [0.1119, 0.2846] | -0.1611 [-0.2359, -0.0881] | 1.0823 [0.4765, 1.7433] | 5.2278 [2.7695, 7.6887] | 0.4680 [0.4239, 0.5183] | 2.7143 [2.3027, 3.2118] | 7.9076 [6.3657, 9.6000] | 1.2414 [0.7961, 1.7429] | 2.3332 [1.5451, 3.2287] |
| GFTb (0,1) | 0.1139 [0.0746, 0.1575] | 0.0358 [0.0148, 0.0639] | 0.2196 [0.1162, 0.3360] | -0.1972 [-0.2667, -0.1386] | 1.5197 [0.6401, 2.4813] | 6.5986 [3.0172, 10.6094] | 0.4829 [0.4326, 0.5378] | 2.9338 [2.4471, 3.5291] | 8.6949 [6.3608, 11.6601] | 1.2253 [0.8514, 1.6538] | 2.5131 [1.6507, 3.4819] |
| GOW (0,1) | 0.0845 [0.0506, 0.1200] | 0.0109 [0.0034, 0.0196] | 0.5374 [0.4218, 0.6388] | 2.4879 [1.4553, 3.7317] | 11.5925 [8.4754, 14.9974] | 25.0961 [18.6553, 31.3656] | 2.7280 [1.7111, 3.9766] | 12.0191 [8.9328, 15.4730] | 25.9640 [19.7118, 32.0403] | 4.1187 [3.1580, 5.1582] | 5.9076 [4.3993, 7.3870] |
| GOW4 (0,1) | 0.0596 [0.0352, 0.0893] | 0.0117 [0.0041, 0.0212] | 0.6285 [0.4821, 0.7630] | 3.8267 [2.4614, 5.3003] | 14.4905 [9.7170, 19.3796] | 29.1462 [20.2390, 38.4956] | 4.0272 [2.6922, 5.4804] | 14.8785 [10.3871, 19.5509] | 29.6958 [21.4025, 38.5684] | 6.6697 [5.0271, 8.2798] | 8.7089 [5.7377, 11.6355] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
