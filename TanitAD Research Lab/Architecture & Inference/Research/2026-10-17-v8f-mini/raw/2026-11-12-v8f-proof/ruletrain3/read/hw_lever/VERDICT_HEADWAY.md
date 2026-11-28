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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOL:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOL4:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFTb:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOL:GOW", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOL4:GOW4", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GOL minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOL4 minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GFTb minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOL minus GOW over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOL4 minus GOW4 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GOL minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / -0.0658 | -0.0409 | [-0.0919, +0.0011] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0187 / -0.0374 | -0.0280 | [-0.0509, -0.0102] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.2617 / +0.3069 | +0.2843 | [+0.2098, +0.3586] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.2247 / +2.5978 | +2.4113 | [+1.4207, +3.6421] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +9.1380 / +8.7287 | +8.9333 | [+5.7670, +12.4567] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +16.1046 / +15.1228 | +15.6137 | [+10.5735, +20.9692] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.9255 / +2.2166 | +2.0710 | [+1.1071, +3.2699] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +8.0539 / +7.7279 | +7.8909 | [+4.9578, +11.1816] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +14.8778 / +13.0922 | +13.9850 | [+9.1904, +19.0604] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +2.4096 / +1.9680 | +2.1888 | [+1.3963, +3.0758] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +2.5795 / +2.3203 | +2.4499 | [+1.8651, +3.1039] | 1.7849 (0.8089) | WORSE |

## GOL4 minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0391 / -0.0765 | -0.0578 | [-0.1045, -0.0152] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0343 / -0.0389 | -0.0366 | [-0.0599, -0.0177] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4174 / +0.4455 | +0.4315 | [+0.3634, +0.4959] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +3.3899 / +3.6258 | +3.5078 | [+2.3794, +4.8193] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +12.5682 / +12.0905 | +12.3293 | [+9.0438, +15.8393] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +22.8888 / +21.8983 | +22.3936 | [+17.0596, +28.0563] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +3.0197 / +3.1618 | +3.0907 | [+1.9905, +4.3675] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +11.2112 / +10.8606 | +11.0359 | [+7.9525, +14.3460] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +20.9624 / +19.2249 | +20.0936 | [+14.9484, +25.5084] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +4.9613 / +4.7030 | +4.8322 | [+3.4916, +6.4055] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +5.5379 / +4.0062 | +4.7721 | [+3.5026, +6.1607] | 1.7849 (0.8089) | WORSE |

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

## GOL minus GOW

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0125 / -0.0391 | -0.0133 | [-0.0539, +0.0184] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0062 / -0.0031 | +0.0016 | [-0.0089, +0.0104] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0903 / -0.0296 | -0.0600 | [-0.1126, -0.0069] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0754 / -0.4000 | -0.2377 | [-0.6019, +0.0571] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -1.1949 / -1.9588 | -1.5768 | [-2.6273, -0.7079] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -3.7362 / -4.7729 | -4.2546 | [-6.9943, -1.9814] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0364 / -0.3416 | -0.1890 | [-0.5269, +0.0962] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -1.0519 / -1.7759 | -1.4139 | [-2.4457, -0.5587] | 1.0764 (0.6913) | BETTER |
| gap error 6.0 s (m) | -3.3652 / -4.7776 | -4.0714 | [-6.9706, -1.7338] | 3.3507 (1.4831) | BETTER |
| time-gap error 2.0 s (s) | -0.1164 / -1.2606 | -0.6885 | [-1.5973, +0.1408] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -1.2908 / -0.9584 | -1.1246 | [-2.2359, -0.1772] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GOL4 minus GOW4

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / -0.0071 | -0.0053 | [-0.0192, +0.0086] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0109 / -0.0047 | -0.0078 | [-0.0170, -0.0022] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0654 / +0.0576 | -0.0039 | [-0.0812, +0.0780] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.8323 / -0.1276 | -0.4800 | [-1.2622, +0.0612] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -3.0245 / +0.8669 | -1.0788 | [-3.9692, +1.5176] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -4.7018 / +1.6523 | -1.5248 | [-6.2562, +3.3186] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.8110 / -0.1260 | -0.4685 | [-1.2387, +0.0588] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -2.9862 / +0.7296 | -1.1283 | [-3.9297, +1.3714] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -4.6728 / +1.2837 | -1.6946 | [-6.2166, +2.8775] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.7491 / -0.4433 | -0.5962 | [-1.3819, +0.1606] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -2.6252 / -0.5821 | -1.6037 | [-3.1216, -0.0393] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT (0,1) | 0.1121 [0.0711, 0.1570] | 0.0405 [0.0212, 0.0648] | 0.1931 [0.1119, 0.2846] | -0.1611 [-0.2359, -0.0881] | 1.0823 [0.4765, 1.7433] | 5.2278 [2.7695, 7.6887] | 0.4680 [0.4239, 0.5183] | 2.7143 [2.3027, 3.2118] | 7.9076 [6.3657, 9.6000] | 1.2414 [0.7961, 1.7429] | 2.3332 [1.5451, 3.2287] |
| GFTb (0,1) | 0.1139 [0.0746, 0.1575] | 0.0358 [0.0148, 0.0639] | 0.2196 [0.1162, 0.3360] | -0.1972 [-0.2667, -0.1386] | 1.5197 [0.6401, 2.4813] | 6.5986 [3.0172, 10.6094] | 0.4829 [0.4326, 0.5378] | 2.9338 [2.4471, 3.5291] | 8.6949 [6.3608, 11.6601] | 1.2253 [0.8514, 1.6538] | 2.5131 [1.6507, 3.4819] |
| GOL (0,1) | 0.0712 [0.0422, 0.1067] | 0.0125 [0.0040, 0.0240] | 0.4774 [0.3931, 0.5564] | 2.2502 [1.2646, 3.4589] | 10.0156 [6.9124, 13.3702] | 20.8415 [15.4135, 26.6666] | 2.5390 [1.5762, 3.7429] | 10.6053 [7.6094, 13.9370] | 21.8926 [16.7320, 27.4750] | 3.4302 [2.4293, 4.5261] | 4.7831 [3.7688, 5.8412] |
| GOL4 (0,1) | 0.0543 [0.0311, 0.0824] | 0.0039 [0.0000, 0.0112] | 0.6246 [0.5394, 0.6974] | 3.3468 [2.2200, 4.6308] | 13.4117 [10.1503, 16.7908] | 27.6214 [21.8552, 33.5413] | 3.5587 [2.4493, 4.8387] | 13.7502 [10.5978, 17.0573] | 28.0012 [22.5196, 33.7936] | 6.0735 [4.5921, 7.7298] | 7.1053 [5.3575, 8.9175] |
| GOW (0,1) | 0.0845 [0.0506, 0.1200] | 0.0109 [0.0034, 0.0196] | 0.5374 [0.4218, 0.6388] | 2.4879 [1.4553, 3.7317] | 11.5925 [8.4754, 14.9974] | 25.0961 [18.6553, 31.3656] | 2.7280 [1.7111, 3.9766] | 12.0191 [8.9328, 15.4730] | 25.9640 [19.7118, 32.0403] | 4.1187 [3.1580, 5.1582] | 5.9076 [4.3993, 7.3870] |
| GOW4 (0,1) | 0.0596 [0.0352, 0.0893] | 0.0117 [0.0041, 0.0212] | 0.6285 [0.4821, 0.7630] | 3.8267 [2.4614, 5.3003] | 14.4905 [9.7170, 19.3796] | 29.1462 [20.2390, 38.4956] | 4.0272 [2.6922, 5.4804] | 14.8785 [10.3871, 19.5509] | 29.6958 [21.4025, 38.5684] | 6.6697 [5.0271, 8.2798] | 8.7089 [5.7377, 11.6355] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
