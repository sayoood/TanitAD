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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLVa:RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLVa:RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLVb:RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLVb:RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLAh:RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLAh:RLAm", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, RLVa minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLVa minus RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLVb minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLVb minus RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLAh minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLAh minus RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## RLVa minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0142 / -0.0053 | +0.0044 | [-0.0133, +0.0212] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0047 / -0.0062 | -0.0008 | [-0.0093, +0.0090] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0093 / -0.0109 | -0.0101 | [-0.0290, +0.0074] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0089 / -0.0053 | +0.0018 | [-0.0109, +0.0151] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.1511 / -0.0290 | -0.0900 | [-0.2046, +0.0280] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.5582 / +0.0803 | -0.2390 | [-0.8083, +0.3525] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0137 / +0.0064 | -0.0037 | [-0.0187, +0.0101] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0843 / -0.1608 | -0.1226 | [-0.2071, -0.0476] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.3072 / -0.2303 | -0.2687 | [-0.6201, +0.0546] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0328 / +0.0082 | +0.0205 | [-0.0188, +0.0639] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0097 / -0.1893 | -0.0995 | [-0.2642, +0.0186] | 1.7849 (0.8089) | OPEN |

## RLVa minus RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / -0.0089 | -0.0027 | [-0.0143, +0.0094] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [-0.0054, +0.0052] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0047 / -0.0125 | -0.0086 | [-0.0307, +0.0123] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0207 / +0.0023 | +0.0115 | [-0.0024, +0.0272] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.1599 / +0.0193 | -0.0703 | [-0.2682, +0.0697] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.4723 / -0.1009 | -0.2866 | [-0.9966, +0.2726] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0169 / -0.0004 | -0.0086 | [-0.0236, +0.0030] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.1730 / -0.0728 | -0.1229 | [-0.2683, -0.0303] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.4563 / -0.2971 | -0.3767 | [-0.9731, +0.0956] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0373 / +0.0113 | -0.0130 | [-0.0567, +0.0239] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1004 / -0.0343 | -0.0673 | [-0.1623, +0.0181] | 1.7849 (0.8089) | OPEN |

## RLVb minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / -0.0036 | +0.0000 | [-0.0086, +0.0098] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0016 / +0.0016 | +0.0000 | [-0.0081, +0.0069] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0031 / -0.0218 | -0.0093 | [-0.0358, +0.0103] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0028 / -0.0033 | -0.0003 | [-0.0114, +0.0139] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.1087 / -0.0605 | -0.0846 | [-0.1799, +0.0015] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.0876 / -0.2883 | -0.1880 | [-0.5706, +0.1963] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0114 / +0.0066 | -0.0024 | [-0.0199, +0.0095] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0012 / -0.0936 | -0.0474 | [-0.1411, +0.0337] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.1291 / -0.1594 | -0.1442 | [-0.4896, +0.1502] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0218 / +0.0598 | +0.0408 | [-0.0038, +0.0997] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0113 / -0.2503 | -0.1308 | [-0.3515, +0.0580] | 1.7849 (0.8089) | OPEN |

## RLVb minus RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / -0.0071 | -0.0071 | [-0.0155, +0.0009] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0062 / +0.0078 | +0.0008 | [-0.0122, +0.0108] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0078 / -0.0234 | -0.0078 | [-0.0323, +0.0170] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0146 / +0.0044 | +0.0095 | [-0.0007, +0.0254] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.1175 / -0.0123 | -0.0649 | [-0.2218, +0.0463] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.0017 / -0.4695 | -0.2356 | [-0.6733, +0.4036] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0146 / -0.0001 | -0.0074 | [-0.0244, +0.0025] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0899 / -0.0056 | -0.0477 | [-0.1880, +0.0493] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.2782 / -0.2262 | -0.2522 | [-0.6999, +0.1069] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0482 / +0.0628 | +0.0073 | [-0.0742, +0.0966] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1020 / -0.0953 | -0.0987 | [-0.2337, +0.0322] | 1.7849 (0.8089) | OPEN |

## RLAh minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / -0.0036 | +0.0027 | [-0.0126, +0.0172] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0016 / -0.0016 | -0.0016 | [-0.0062, +0.0031] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0016 / -0.0031 | -0.0008 | [-0.0161, +0.0142] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0022 / -0.0052 | -0.0015 | [-0.0096, +0.0064] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0079 / +0.0407 | +0.0164 | [-0.0535, +0.0754] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0120 / +0.3271 | +0.1696 | [-0.2247, +0.5363] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0077 / +0.0090 | +0.0006 | [-0.0112, +0.0125] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0337 / -0.0021 | +0.0158 | [-0.0308, +0.0698] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0124 / +0.1706 | +0.0791 | [-0.2247, +0.3623] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0540 / +0.1411 | +0.0975 | [+0.0149, +0.1881] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0801 / -0.1062 | -0.0130 | [-0.1732, +0.1110] | 1.7849 (0.8089) | OPEN |

## RLAh minus RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / -0.0071 | -0.0044 | [-0.0152, +0.0059] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0062 / +0.0047 | -0.0008 | [-0.0104, +0.0080] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0062 / -0.0047 | +0.0008 | [-0.0135, +0.0161] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0140 / +0.0025 | +0.0083 | [-0.0013, +0.0199] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0167 / +0.0890 | +0.0361 | [-0.1046, +0.1314] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0979 / +0.1460 | +0.1220 | [-0.3481, +0.5482] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0109 / +0.0022 | -0.0043 | [-0.0160, +0.0055] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0550 / +0.0859 | +0.0155 | [-0.1286, +0.1183] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.1615 / +0.1037 | -0.0289 | [-0.5253, +0.3349] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0161 / +0.1442 | +0.0640 | [-0.0429, +0.1833] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0106 / +0.0488 | +0.0191 | [-0.0673, +0.1099] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| RLAh (0,1) | 0.1272 [0.0813, 0.1840] | 0.0405 [0.0194, 0.0656] | 0.1822 [0.1363, 0.2299] | -0.2136 [-0.2603, -0.1676] | 0.7725 [0.1138, 1.3498] | 5.0875 [3.6486, 6.4778] | 0.4844 [0.4366, 0.5355] | 2.6740 [2.3763, 3.0077] | 7.6576 [6.7749, 8.6562] | 1.1531 [0.7613, 1.5985] | 2.0357 [1.5149, 2.6296] |
| RLAm (0,1) | 0.1317 [0.0831, 0.1892] | 0.0413 [0.0194, 0.0678] | 0.1815 [0.1372, 0.2287] | -0.2218 [-0.2701, -0.1740] | 0.7364 [0.0292, 1.3799] | 4.9655 [3.4659, 6.5884] | 0.4887 [0.4391, 0.5413] | 2.6585 [2.3764, 2.9619] | 7.6865 [6.6276, 8.9726] | 1.0890 [0.7424, 1.4765] | 2.0166 [1.4838, 2.6347] |
| RLFT (0,1) | 0.1246 [0.0752, 0.1873] | 0.0421 [0.0206, 0.0660] | 0.1830 [0.1376, 0.2315] | -0.2121 [-0.2594, -0.1651] | 0.7561 [0.0790, 1.3428] | 4.9179 [3.3324, 6.5198] | 0.4837 [0.4329, 0.5361] | 2.6582 [2.3576, 2.9983] | 7.5785 [6.6350, 8.6730] | 1.0555 [0.7232, 1.4345] | 2.0488 [1.4877, 2.6843] |
| RLVa (0,1) | 0.1290 [0.0841, 0.1799] | 0.0413 [0.0189, 0.0672] | 0.1729 [0.1315, 0.2160] | -0.2103 [-0.2580, -0.1627] | 0.6661 [0.0802, 1.1764] | 4.6789 [3.4173, 5.8926] | 0.4801 [0.4344, 0.5293] | 2.5356 [2.2547, 2.8422] | 7.3098 [6.4500, 8.2031] | 1.0760 [0.7317, 1.4777] | 1.9493 [1.4292, 2.5616] |
| RLVb (0,1) | 0.1246 [0.0783, 0.1812] | 0.0421 [0.0201, 0.0674] | 0.1737 [0.1250, 0.2270] | -0.2123 [-0.2565, -0.1665] | 0.6715 [0.0245, 1.2481] | 4.7299 [3.1122, 6.2666] | 0.4813 [0.4348, 0.5321] | 2.6108 [2.3322, 2.9074] | 7.4343 [6.5385, 8.4116] | 1.0963 [0.7379, 1.5172] | 1.9179 [1.4039, 2.5142] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
