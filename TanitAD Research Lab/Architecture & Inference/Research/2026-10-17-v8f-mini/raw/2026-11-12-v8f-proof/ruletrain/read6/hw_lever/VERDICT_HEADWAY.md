# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 16, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRT6:GFT6", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFT6b:GFT6", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFT6:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRT6:GRT", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GRT6 minus GFT6 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GFT6b minus GFT6 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GFT6 minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRT6 minus GRT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GRT6 minus GFT6

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / -0.0142 | -0.0053 | [-0.0281, +0.0119] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0016 / +0.0000 | -0.0008 | [-0.0063, +0.0050] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0078 / +0.0327 | +0.0202 | [-0.0108, +0.0551] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1506 / +0.1450 | +0.1478 | [+0.0912, +0.2164] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.3707 / +0.7364 | +0.5535 | [+0.0631, +1.0600] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.2815 / +1.8265 | +1.0540 | [-0.3376, +2.6995] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0187 / +0.0216 | +0.0202 | [-0.0139, +0.0615] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.3210 / +0.2533 | +0.2871 | [+0.0213, +0.6255] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.1929 / +1.2669 | +0.7299 | [-0.4271, +1.9143] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0015 / +0.0349 | +0.0167 | [-0.0122, +0.0532] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1069 / -0.0287 | -0.0678 | [-0.2167, +0.0714] | 1.7849 (0.8089) | OPEN |

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

## GFT6 minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0160 | +0.0080 | [-0.0199, +0.0316] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0140 / +0.0062 | +0.0101 | [-0.0024, +0.0244] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0265 / +0.0202 | -0.0031 | [-0.0562, +0.0472] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0918 / -0.0196 | -0.0557 | [-0.1046, -0.0104] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.1955 / -0.0417 | -0.1186 | [-0.4695, +0.2121] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.4454 / +0.2760 | -0.0847 | [-1.5327, +1.4456] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0236 / +0.0107 | +0.0172 | [+0.0029, +0.0343] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.3424 / +0.0896 | -0.1264 | [-0.5109, +0.2494] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.1691 / +0.3124 | +0.0717 | [-1.0318, +1.2828] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.4667 / +0.0920 | -0.1873 | [-0.5668, +0.1909] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.4361 / -0.0296 | -0.2329 | [-0.5837, +0.0437] | 1.7849 (0.8089) | OPEN |

## GRT6 minus GRT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0053 | +0.0027 | [-0.0135, +0.0220] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0109 / +0.0093 | +0.0101 | [-0.0013, +0.0220] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0919 / -0.0062 | -0.0491 | [-0.1135, +0.0159] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0156 / +0.0684 | +0.0264 | [-0.0446, +0.0980] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.4623 / +0.1729 | -0.1447 | [-0.6564, +0.3959] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -2.0559 / +0.1846 | -0.9356 | [-2.6691, +0.8992] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0456 / +0.0318 | +0.0387 | [+0.0120, +0.0722] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.1882 / +0.1416 | -0.0233 | [-0.3477, +0.3052] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -1.6155 / +0.5428 | -0.5364 | [-2.1616, +1.2048] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.4667 / +0.0648 | -0.2009 | [-0.5729, +0.1483] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1978 / -0.2112 | -0.2045 | [-0.3954, -0.0198] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT (0,1) | 0.1121 [0.0711, 0.1570] | 0.0405 [0.0212, 0.0648] | 0.1931 [0.1119, 0.2846] | -0.1611 [-0.2359, -0.0881] | 1.0823 [0.4765, 1.7433] | 5.2278 [2.7695, 7.6887] | 0.4680 [0.4239, 0.5183] | 2.7143 [2.3027, 3.2118] | 7.9076 [6.3657, 9.6000] | 1.2414 [0.7961, 1.7429] | 2.3332 [1.5451, 3.2287] |
| GFT6 (0,1) | 0.1201 [0.0734, 0.1731] | 0.0506 [0.0281, 0.0755] | 0.1900 [0.1253, 0.2584] | -0.2167 [-0.2618, -0.1654] | 0.9637 [0.4299, 1.4763] | 5.1432 [2.9081, 7.4572] | 0.4852 [0.4418, 0.5347] | 2.5879 [2.2987, 2.9080] | 7.9793 [6.5034, 9.6963] | 1.0540 [0.7300, 1.4381] | 2.1003 [1.4546, 2.8105] |
| GFT6b (0,1) | 0.1219 [0.0735, 0.1794] | 0.0413 [0.0191, 0.0704] | 0.2118 [0.1254, 0.3098] | -0.2164 [-0.2703, -0.1648] | 1.1171 [0.3236, 2.0043] | 5.6257 [2.4175, 9.2734] | 0.4841 [0.4393, 0.5340] | 2.7599 [2.2936, 3.3274] | 8.3665 [6.1832, 11.2276] | 1.1059 [0.7431, 1.5106] | 2.2538 [1.4543, 3.1418] |
| GRT (0,1) | 0.1121 [0.0751, 0.1520] | 0.0397 [0.0224, 0.0590] | 0.2593 [0.1678, 0.3553] | -0.0953 [-0.1798, -0.0122] | 1.6620 [0.9858, 2.3766] | 7.1328 [4.7012, 9.7188] | 0.4666 [0.4233, 0.5201] | 2.8984 [2.4723, 3.3656] | 9.2455 [7.2804, 11.4222] | 1.2717 [0.8459, 1.7530] | 2.2371 [1.6207, 2.9295] |
| GRT6 (0,1) | 0.1148 [0.0742, 0.1608] | 0.0498 [0.0283, 0.0749] | 0.2103 [0.1531, 0.2694] | -0.0689 [-0.1395, 0.0075] | 1.5173 [1.0001, 2.1083] | 6.1972 [4.3909, 7.9833] | 0.5053 [0.4487, 0.5685] | 2.8751 [2.4766, 3.3472] | 8.7091 [7.3765, 10.1580] | 1.0707 [0.7392, 1.4555] | 2.0325 [1.4331, 2.7233] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
