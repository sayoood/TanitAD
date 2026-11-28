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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQF:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQF:LSDRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQF:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQF:LSDMC", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQF:LSDQC", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSDQF minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQF minus LSDRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQF minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQF minus LSDMC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQF minus LSDQC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSDQF minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0694 / -0.0872 | -0.0783 | [-0.1237, -0.0388] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0156 / -0.0483 | -0.0319 | [-0.0608, -0.0085] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1464 / +0.1433 | +0.1449 | [+0.0990, +0.1909] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.9008 / +1.0374 | +0.9691 | [+0.6155, +1.3927] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.3622 / +3.5924 | +3.4773 | [+2.2960, +4.7951] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.3743 / +6.4610 | +5.9177 | [+3.9763, +8.0394] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6496 / +0.7720 | +0.7108 | [+0.4185, +1.0642] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.7774 / +3.0897 | +2.9335 | [+1.8405, +4.1686] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.4611 / +4.4832 | +4.4722 | [+2.5409, +6.5355] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1422 / +0.1795 | +0.1608 | [+0.0981, +0.2501] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.6913 / +0.6271 | +0.6592 | [+0.3638, +0.9686] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQF minus LSDRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0498 / -0.0730 | -0.0614 | [-0.1001, -0.0291] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0109 / -0.0296 | -0.0202 | [-0.0399, -0.0053] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0561 / +0.0888 | +0.0724 | [+0.0383, +0.1084] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4935 / +0.6006 | +0.5470 | [+0.3785, +0.7630] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.7594 / +1.9826 | +1.8710 | [+1.2455, +2.6263] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +2.3911 / +3.0858 | +2.7384 | [+1.6274, +4.0153] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3410 / +0.4795 | +0.4102 | [+0.2521, +0.6212] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.4998 / +1.5892 | +1.5445 | [+0.9678, +2.2261] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.0722 / +2.7408 | +2.4065 | [+1.2702, +3.7728] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.1160 / +0.1353 | +0.1256 | [+0.0658, +0.2080] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.5147 / +0.3430 | +0.4288 | [+0.2186, +0.7051] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQF minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0196 / -0.0018 | +0.0089 | [-0.0153, +0.0282] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0031 / +0.0234 | +0.0132 | [+0.0013, +0.0368] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0670 / +0.0903 | +0.0787 | [+0.0452, +0.1144] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4643 / +0.5948 | +0.5295 | [+0.3569, +0.7436] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.0097 / +2.1729 | +2.0913 | [+1.4016, +2.8276] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +2.6756 / +2.8586 | +2.7671 | [+1.5758, +4.0301] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3349 / +0.4973 | +0.4161 | [+0.2437, +0.6352] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.9937 / +2.2516 | +2.1226 | [+1.4276, +2.8761] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.6268 / +3.3720 | +2.9994 | [+1.6735, +4.3494] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0981 / +0.1401 | +0.1191 | [+0.0608, +0.1981] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.5190 / +0.5503 | +0.5346 | [+0.2730, +0.8316] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQF minus LSDMC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / +0.0018 | +0.0044 | [-0.0157, +0.0232] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0062 | +0.0039 | [-0.0041, +0.0168] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0218 / -0.0016 | -0.0117 | [-0.0397, +0.0224] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.4466 / -0.2771 | -0.3619 | [-0.5867, -0.1169] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -0.6370 / -0.4154 | -0.5262 | [-1.2645, +0.2282] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.0468 / -0.8789 | -0.9629 | [-2.2751, +0.2919] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.4375 / -0.2836 | -0.3606 | [-0.5649, -0.1317] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.6873 / -0.4075 | -0.5474 | [-1.2429, +0.1729] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -1.1046 / -0.5865 | -0.8456 | [-2.0834, +0.4663] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.2302 / -0.1994 | -0.2148 | [-0.3347, -0.1092] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.4641 / -0.4853 | -0.4747 | [-0.7303, -0.2336] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQF minus LSDQC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0107 / -0.0089 | -0.0098 | [-0.0302, +0.0103] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0078 / -0.0016 | -0.0047 | [-0.0141, +0.0054] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0374 / +0.0125 | -0.0125 | [-0.0511, +0.0323] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2472 / -0.0803 | -0.1637 | [-0.3598, +0.0401] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.4170 / -0.2011 | -0.3090 | [-1.0115, +0.3765] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.7199 / -0.4761 | -0.5980 | [-1.8855, +0.7711] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.2364 / -0.0946 | -0.1655 | [-0.3441, +0.0265] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.4951 / -0.2669 | -0.3810 | [-1.0710, +0.2789] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.8835 / -0.3764 | -0.6300 | [-1.8937, +0.8036] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0170 / -0.0181 | -0.0176 | [-0.0739, +0.0386] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1644 / -0.1016 | -0.1330 | [-0.3396, +0.0506] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LSD (0,1) | 0.1343 [0.0873, 0.1879] | 0.0452 [0.0109, 0.0906] | 0.2531 [0.1958, 0.3138] | -0.2031 [-0.2748, -0.1325] | 1.4817 [0.8172, 2.2957] | 6.7598 [4.6651, 8.6599] | 0.5100 [0.4630, 0.5586] | 2.8968 [2.3956, 3.5375] | 9.3372 [8.1968, 10.6886] | 1.4640 [0.8170, 2.2047] | 2.5391 [1.6003, 3.5518] |
| LSDMC (0,1) | 0.0516 [0.0277, 0.0823] | 0.0093 [0.0000, 0.0261] | 0.4097 [0.3379, 0.4816] | 1.1278 [0.7191, 1.5873] | 5.4852 [3.9211, 7.1340] | 13.6403 [10.7057, 16.7188] | 1.5814 [1.2109, 2.0053] | 6.3777 [4.9718, 7.8509] | 14.6549 [12.0072, 17.5735] | 1.8397 [1.1472, 2.5956] | 3.6730 [2.6322, 4.7572] |
| LSDQC (0,1) | 0.0658 [0.0389, 0.0988] | 0.0179 [0.0047, 0.0379] | 0.4104 [0.3222, 0.4975] | 0.9297 [0.5356, 1.3644] | 5.2680 [3.7314, 6.8447] | 13.2754 [10.2952, 16.3070] | 1.3863 [1.0227, 1.7990] | 6.2113 [4.8507, 7.6690] | 14.4393 [11.7734, 17.3270] | 1.6424 [0.9985, 2.3604] | 3.3313 [2.3127, 4.4132] |
| LSDQF (0,1) | 0.0560 [0.0347, 0.0836] | 0.0132 [0.0013, 0.0368] | 0.3980 [0.3258, 0.4644] | 0.7660 [0.4417, 1.1437] | 4.9590 [3.5968, 6.2755] | 12.6774 [9.9580, 15.3705] | 1.2208 [0.9186, 1.5932] | 5.8303 [4.6295, 7.0566] | 13.8093 [11.4509, 16.4562] | 1.6249 [0.9888, 2.3517] | 3.1983 [2.1883, 4.3081] |
| LSDRH (0,1) | 0.1174 [0.0734, 0.1670] | 0.0335 [0.0072, 0.0708] | 0.3255 [0.2493, 0.4022] | 0.2189 [-0.0074, 0.5021] | 3.0880 [1.9867, 4.2275] | 9.9390 [7.5229, 12.1905] | 0.8106 [0.6022, 1.0655] | 4.2859 [3.4176, 5.2197] | 11.4028 [9.5253, 13.4774] | 1.4992 [0.8581, 2.2308] | 2.7695 [1.8682, 3.7196] |
| LSDRL (0,1) | 0.0472 [0.0227, 0.0829] | 0.0000 [0.0000, 0.0000] | 0.3193 [0.2469, 0.3910] | 0.2364 [-0.0076, 0.5420] | 2.8677 [1.8890, 3.9087] | 9.9103 [7.7746, 11.9708] | 0.8047 [0.5833, 1.0843] | 3.7077 [2.8459, 4.6228] | 10.8099 [8.9499, 12.7615] | 1.5058 [0.8571, 2.2451] | 2.6637 [1.6935, 3.7219] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
