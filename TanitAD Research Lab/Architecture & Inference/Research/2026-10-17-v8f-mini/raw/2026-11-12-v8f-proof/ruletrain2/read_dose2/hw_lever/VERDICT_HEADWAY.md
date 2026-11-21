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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRW6:GFT6", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRW6:GRT6", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFT6b:GFT6", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GRW6 minus GFT6 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRW6 minus GRT6 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GFT6b minus GFT6 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GRW6 minus GFT6

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / -0.0231 | -0.0125 | [-0.0417, +0.0067] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0062 / -0.0047 | -0.0055 | [-0.0136, +0.0023] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0125 / +0.0452 | +0.0288 | [-0.0088, +0.0776] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2913 / +0.4294 | +0.3604 | [+0.2050, +0.5434] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.4190 / +1.0298 | +0.7244 | [+0.1759, +1.4065] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.2291 / +1.6567 | +0.7138 | [-0.7029, +2.5942] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.1064 / +0.1504 | +0.1284 | [+0.0267, +0.2543] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.3024 / +0.4916 | +0.3970 | [+0.0939, +0.7621] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.1175 / +1.2424 | +0.5625 | [-0.5460, +1.9422] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0168 / -0.1890 | -0.0861 | [-0.3124, +0.0395] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0563 / -0.2318 | -0.1440 | [-0.4001, +0.0395] | 1.7849 (0.8089) | OPEN |

## GRW6 minus GRT6

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0053 / -0.0089 | -0.0071 | [-0.0232, +0.0071] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0047 / -0.0047 | -0.0047 | [-0.0131, +0.0035] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0047 / +0.0125 | +0.0086 | [-0.0144, +0.0335] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1407 / +0.2844 | +0.2126 | [+0.0989, +0.3623] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.0484 / +0.2934 | +0.1709 | [-0.1530, +0.5640] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.5105 / -0.1699 | -0.3402 | [-1.1085, +0.4228] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0877 / +0.1288 | +0.1082 | [+0.0291, +0.2189] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | -0.0186 / +0.2383 | +0.1098 | [-0.1660, +0.4871] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.3104 / -0.0244 | -0.1674 | [-0.8618, +0.5386] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0183 / -0.2239 | -0.1028 | [-0.3334, +0.0400] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0506 / -0.2030 | -0.0762 | [-0.2666, +0.0981] | 1.7849 (0.8089) | OPEN |

## GFT6b minus GFT6

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / +0.0071 | +0.0018 | [-0.0187, +0.0206] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0187 / +0.0000 | -0.0093 | [-0.0258, +0.0086] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0498 / -0.0062 | +0.0218 | [-0.0242, +0.0655] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0093 / -0.0085 | +0.0004 | [-0.0229, +0.0258] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.4370 / -0.1303 | +0.1534 | [-0.2775, +0.6014] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +1.6375 / -0.6725 | +0.4825 | [-1.1930, +2.1357] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0023 / -0.0045 | -0.0011 | [-0.0169, +0.0170] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.4491 / -0.1052 | +0.1719 | [-0.2405, +0.6014] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +1.3046 / -0.5300 | +0.3873 | [-0.9864, +1.7925] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.1954 / -0.0917 | +0.0518 | [-0.1421, +0.2568] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.4358 / -0.1288 | +0.1535 | [-0.2809, +0.5835] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT6 (0,1) | 0.1201 [0.0734, 0.1731] | 0.0506 [0.0281, 0.0755] | 0.1900 [0.1253, 0.2584] | -0.2167 [-0.2618, -0.1654] | 0.9637 [0.4299, 1.4763] | 5.1432 [2.9081, 7.4572] | 0.4852 [0.4418, 0.5347] | 2.5879 [2.2987, 2.9080] | 7.9793 [6.5034, 9.6963] | 1.0540 [0.7300, 1.4381] | 2.1003 [1.4546, 2.8105] |
| GFT6b (0,1) | 0.1219 [0.0735, 0.1794] | 0.0413 [0.0191, 0.0704] | 0.2118 [0.1254, 0.3098] | -0.2164 [-0.2703, -0.1648] | 1.1171 [0.3236, 2.0043] | 5.6257 [2.4175, 9.2734] | 0.4841 [0.4393, 0.5340] | 2.7599 [2.2936, 3.3274] | 8.3665 [6.1832, 11.2276] | 1.1059 [0.7431, 1.5106] | 2.2538 [1.4543, 3.1418] |
| GRT6 (0,1) | 0.1148 [0.0742, 0.1608] | 0.0498 [0.0283, 0.0749] | 0.2103 [0.1531, 0.2694] | -0.0689 [-0.1395, 0.0075] | 1.5173 [1.0001, 2.1083] | 6.1972 [4.3909, 7.9833] | 0.5053 [0.4487, 0.5685] | 2.8751 [2.4766, 3.3472] | 8.7091 [7.3765, 10.1580] | 1.0707 [0.7392, 1.4555] | 2.0325 [1.4331, 2.7233] |
| GRW6 (0,1) | 0.1077 [0.0687, 0.1547] | 0.0452 [0.0232, 0.0702] | 0.2188 [0.1568, 0.2788] | 0.1436 [-0.0093, 0.3172] | 1.6882 [1.1773, 2.2185] | 5.8570 [4.1293, 7.5196] | 0.6136 [0.5051, 0.7452] | 2.9849 [2.6130, 3.4041] | 8.5417 [7.2976, 9.9050] | 0.9679 [0.6851, 1.2877] | 1.9563 [1.2813, 2.6723] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
