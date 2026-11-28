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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LCS:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LFTRL:LFT", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LCS minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LFTRL minus LFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LCS minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / +0.0000 | -0.0036 | [-0.0194, +0.0129] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0125 / +0.0078 | +0.0101 | [-0.0050, +0.0258] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0561 / +0.0125 | -0.0218 | [-0.0850, +0.0291] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0212 / +0.0364 | +0.0076 | [-0.0384, +0.0594] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.3115 / +0.1359 | -0.0878 | [-0.4701, +0.2867] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.3207 / +0.8538 | -0.2335 | [-1.9765, +1.4750] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0126 / +0.0095 | -0.0015 | [-0.0247, +0.0315] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.3706 / +0.1561 | -0.1073 | [-0.4619, +0.2541] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -1.0825 / +0.4586 | -0.3119 | [-1.6001, +0.8960] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1606 / +0.1046 | -0.0280 | [-0.2337, +0.1513] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.5922 / +0.0066 | -0.2928 | [-0.7725, +0.0934] | 1.7849 (0.8089) | OPEN |

## LFTRL minus LFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0907 / -0.0801 | -0.0854 | [-0.1240, -0.0504] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0592 / -0.0639 | -0.0615 | [-0.0900, -0.0363] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0654 / +0.0794 | +0.0724 | [+0.0370, +0.1174] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4357 / +0.4399 | +0.4378 | [+0.1828, +0.7589] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.3874 / +1.5441 | +1.4657 | [+0.7389, +2.3847] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +3.5043 / +3.9700 | +3.7371 | [+2.3164, +5.5960] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3175 / +0.3225 | +0.3200 | [+0.0908, +0.6167] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6057 / +0.5842 | +0.5949 | [+0.0757, +1.2642] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.1749 / +1.4327 | +1.3038 | [+0.0790, +2.7971] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0398 / +0.0428 | +0.0413 | [+0.0172, +0.0699] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0153 / +0.0704 | +0.0428 | [-0.0481, +0.1312] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LCS (0,1) | 0.1210 [0.0777, 0.1708] | 0.0545 [0.0288, 0.0863] | 0.1893 [0.1297, 0.2491] | -0.2273 [-0.2956, -0.1736] | 1.0989 [0.6024, 1.6114] | 5.2059 [3.6945, 6.6401] | 0.5147 [0.4615, 0.5802] | 2.6783 [2.3307, 3.0695] | 8.0179 [6.9642, 9.2438] | 1.2085 [0.7878, 1.6529] | 2.1674 [1.4069, 2.9766] |
| LFT (0,1) | 0.1415 [0.0965, 0.1878] | 0.0631 [0.0370, 0.0918] | 0.1456 [0.0958, 0.1980] | -0.2224 [-0.2672, -0.1763] | 0.5186 [0.0504, 0.9670] | 3.2365 [1.7611, 4.7609] | 0.4832 [0.4338, 0.5358] | 2.5537 [2.2424, 2.8619] | 7.0288 [6.1826, 7.9892] | 0.9515 [0.6423, 1.3322] | 2.0086 [1.4447, 2.7032] |
| LFTRL (0,1) | 0.0560 [0.0301, 0.0870] | 0.0016 [0.0000, 0.0062] | 0.2181 [0.1638, 0.2758] | 0.2154 [-0.0402, 0.5332] | 1.9843 [1.2798, 2.7899] | 6.9736 [5.2239, 8.7284] | 0.8032 [0.5710, 1.1043] | 3.1487 [2.5815, 3.8749] | 8.3326 [6.9243, 9.8570] | 0.9928 [0.6759, 1.3704] | 2.0514 [1.4786, 2.7324] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
