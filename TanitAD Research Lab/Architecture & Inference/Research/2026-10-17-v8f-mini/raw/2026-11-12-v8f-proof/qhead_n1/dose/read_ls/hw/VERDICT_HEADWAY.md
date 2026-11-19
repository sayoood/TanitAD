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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQF:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQF:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQF:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQF:LSMC", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQF:LSQC", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSQF minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQF minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQF minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQF minus LSMC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQF minus LSQC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSQF minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0730 / -0.0765 | -0.0747 | [-0.1128, -0.0393] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0265 / -0.0389 | -0.0327 | [-0.0518, -0.0156] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1745 / +0.1713 | +0.1729 | [+0.1222, +0.2242] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0019 / +1.1500 | +1.0759 | [+0.7246, +1.4617] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.6546 / +3.4877 | +3.5712 | [+2.4836, +4.7395] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.7886 / +6.6940 | +6.7413 | [+4.7218, +8.8577] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7717 / +0.8683 | +0.8200 | [+0.5151, +1.1544] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.8740 / +2.9546 | +2.9143 | [+1.9542, +3.9681] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.5747 / +4.4833 | +5.0290 | [+3.0543, +7.0445] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2767 / +0.4684 | +0.3726 | [+0.1330, +0.6472] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8149 / +1.2085 | +1.0117 | [+0.5984, +1.5017] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQF minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0320 / -0.0374 | -0.0347 | [-0.0523, -0.0189] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0140 / -0.0093 | -0.0117 | [-0.0221, -0.0037] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0872 / +0.0607 | +0.0740 | [+0.0438, +0.1052] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6323 / +0.4364 | +0.5344 | [+0.3430, +0.7398] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.0035 / +1.1114 | +1.5575 | [+0.7892, +2.3821] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +3.3822 / +1.9633 | +2.6727 | [+1.4022, +4.1518] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4691 / +0.3196 | +0.3944 | [+0.2247, +0.5657] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.7735 / +0.8327 | +1.3031 | [+0.5610, +2.1259] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.0648 / +1.4199 | +2.2423 | [+0.8009, +3.8317] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.2547 / +0.1002 | +0.1774 | [+0.0695, +0.4219] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.6778 / +0.3169 | +0.4974 | [+0.2070, +0.8575] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQF minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0071 | +0.0044 | [-0.0133, +0.0252] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0047 / +0.0187 | +0.0117 | [+0.0021, +0.0293] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1012 / +0.1184 | +0.1098 | [+0.0730, +0.1500] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5660 / +0.7033 | +0.6346 | [+0.4200, +0.8811] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.3058 / +2.1603 | +2.2330 | [+1.4821, +3.0359] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.7022 / +3.2590 | +3.4806 | [+2.0591, +4.9128] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4455 / +0.5748 | +0.5101 | [+0.3151, +0.7289] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.1641 / +2.2373 | +2.2007 | [+1.5026, +2.9511] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.6654 / +3.4479 | +3.5567 | [+2.2112, +4.9413] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2355 / +0.4287 | +0.3321 | [+0.0924, +0.6097] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.6768 / +1.1183 | +0.8975 | [+0.4770, +1.4009] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQF minus LSMC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0142 / +0.0071 | -0.0036 | [-0.0246, +0.0195] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0078 | +0.0047 | [-0.0025, +0.0167] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0171 / -0.0607 | -0.0218 | [-0.0770, +0.0327] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0287 / -0.6331 | -0.3309 | [-0.7582, +0.0564] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0064 / -1.7029 | -0.8546 | [-2.0759, +0.4609] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.2738 / -2.9720 | -1.3491 | [-3.7794, +1.0084] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0301 / -0.5755 | -0.3028 | [-0.6969, +0.0485] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0145 / -1.5552 | -0.7848 | [-1.9317, +0.4316] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.2069 / -2.8056 | -1.2994 | [-3.5754, +0.9032] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0378 / -0.3849 | -0.1736 | [-0.4731, +0.1946] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.3227 / -1.0736 | -0.6982 | [-1.2905, -0.1182] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQF minus LSQC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0142 / -0.0160 | -0.0151 | [-0.0297, -0.0006] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0109 / -0.0047 | -0.0078 | [-0.0158, -0.0000] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0156 / -0.0031 | +0.0062 | [-0.0167, +0.0287] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0994 / -0.0899 | +0.0048 | [-0.1959, +0.1790] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.3539 / -0.4452 | -0.0457 | [-0.8766, +0.7307] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.1559 / -1.2312 | -0.5377 | [-2.0292, +0.8379] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0760 / -0.0850 | -0.0045 | [-0.1823, +0.1524] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.2576 / -0.5719 | -0.1572 | [-0.9842, +0.6148] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0465 / -1.4146 | -0.7306 | [-2.1974, +0.5711] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.1841 / -0.0244 | +0.0799 | [-0.0708, +0.3505] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.2413 / -0.1172 | +0.0621 | [-0.2167, +0.4086] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| LSMC (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4058 [0.3276, 0.4795] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5297 [10.7494, 16.4204] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6582 [12.0491, 17.4352] | 1.7826 [1.3123, 2.2538] | 4.1701 [3.3443, 5.0752] |
| LSQC (0,1) | 0.0649 [0.0384, 0.0985] | 0.0195 [0.0078, 0.0364] | 0.3777 [0.2840, 0.4724] | 0.8362 [0.4309, 1.3170] | 4.8035 [3.4585, 6.2442] | 12.7183 [9.6831, 15.6759] | 1.3408 [0.9562, 1.7937] | 5.8570 [4.6629, 7.2501] | 14.0894 [11.3615, 16.9036] | 1.5291 [1.0824, 1.9879] | 3.4098 [2.5781, 4.2949] |
| LSQF (0,1) | 0.0498 [0.0271, 0.0820] | 0.0117 [0.0021, 0.0293] | 0.3840 [0.2797, 0.4904] | 0.8410 [0.5095, 1.2031] | 4.7578 [3.4920, 6.0822] | 12.1807 [9.0386, 15.2610] | 1.3362 [1.0303, 1.6757] | 5.6998 [4.6302, 6.8898] | 13.3588 [10.5585, 16.2260] | 1.6090 [1.0888, 2.1764] | 3.4719 [2.5044, 4.5203] |
| LSRH (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| LSRL (0,1) | 0.0454 [0.0238, 0.0728] | 0.0000 [0.0000, 0.0000] | 0.2741 [0.1700, 0.3895] | 0.2064 [-0.0545, 0.5267] | 2.5248 [1.6354, 3.5112] | 8.7001 [6.1813, 11.2690] | 0.8261 [0.5991, 1.1154] | 3.4991 [2.7134, 4.3601] | 9.8022 [7.5079, 12.2263] | 1.2769 [0.7638, 1.8367] | 2.5743 [1.5172, 3.8082] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
