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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLVc:RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLVc:RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLVd:RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLVd:RLAm", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, RLVc minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLVc minus RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLVd minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLVd minus RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## RLVc minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0107 / -0.0053 | +0.0027 | [-0.0118, +0.0156] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0016 / +0.0031 | +0.0008 | [-0.0050, +0.0084] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0109 / -0.0296 | -0.0093 | [-0.0446, +0.0202] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0149 / +0.0071 | -0.0039 | [-0.0184, +0.0119] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0339 / -0.0507 | -0.0084 | [-0.1235, +0.0861] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0739 / -0.6495 | -0.2878 | [-0.9310, +0.2798] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0059 / +0.0038 | +0.0048 | [+0.0003, +0.0097] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.0724 / -0.1478 | -0.0377 | [-0.2037, +0.1043] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0887 / -0.3524 | -0.1319 | [-0.5877, +0.2445] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0194 / +0.0257 | +0.0226 | [-0.0196, +0.0657] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.1098 / -0.1551 | -0.0226 | [-0.2587, +0.1807] | 1.7849 (0.8089) | OPEN |

## RLVc minus RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / -0.0089 | -0.0044 | [-0.0144, +0.0038] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0062 / +0.0093 | +0.0016 | [-0.0097, +0.0128] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0156 / -0.0312 | -0.0078 | [-0.0438, +0.0251] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0031 / +0.0148 | +0.0058 | [-0.0073, +0.0182] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0251 / -0.0024 | +0.0113 | [-0.1060, +0.1304] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.1599 / -0.8306 | -0.3354 | [-1.1460, +0.4658] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0027 / -0.0029 | -0.0001 | [-0.0057, +0.0067] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0163 / -0.0598 | -0.0381 | [-0.1374, +0.0697] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0605 / -0.4192 | -0.2399 | [-0.7188, +0.2587] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0507 / +0.0288 | -0.0109 | [-0.0773, +0.0469] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0191 / -0.0001 | +0.0095 | [-0.1093, +0.1274] | 1.7849 (0.8089) | OPEN |

## RLVd minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / -0.0178 | -0.0071 | [-0.0258, +0.0101] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0016 / -0.0062 | -0.0039 | [-0.0092, +0.0016] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0109 / +0.0031 | +0.0070 | [-0.0089, +0.0236] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0105 / -0.0015 | -0.0060 | [-0.0153, +0.0034] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0356 / +0.0679 | +0.0518 | [-0.0354, +0.1342] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.3999 / +0.5134 | +0.4566 | [+0.0816, +0.7999] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0020 / +0.0023 | +0.0021 | [-0.0040, +0.0086] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0753 / -0.0898 | -0.0072 | [-0.1394, +0.1094] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.3193 / +0.0919 | +0.2056 | [-0.1413, +0.5295] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0229 / -0.0241 | -0.0006 | [-0.0443, +0.0473] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.2011 / -0.1089 | +0.0461 | [-0.1971, +0.2580] | 1.7849 (0.8089) | OPEN |

## RLVd minus RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / -0.0214 | -0.0142 | [-0.0277, +0.0000] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0062 / +0.0000 | -0.0031 | [-0.0107, +0.0041] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0156 / +0.0016 | +0.0086 | [-0.0078, +0.0257] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0014 / +0.0062 | +0.0038 | [-0.0043, +0.0109] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0268 / +0.1162 | +0.0715 | [-0.0641, +0.1775] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.4858 / +0.3322 | +0.4090 | [+0.0439, +0.8150] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0013 / -0.0044 | -0.0028 | [-0.0092, +0.0042] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0133 / -0.0018 | -0.0076 | [-0.1076, +0.0779] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.1702 / +0.0251 | +0.0976 | [-0.2486, +0.4417] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0472 / -0.0210 | -0.0341 | [-0.0723, -0.0002] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1104 / +0.0461 | +0.0782 | [-0.0238, +0.1775] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| RLAm (0,1) | 0.1317 [0.0831, 0.1892] | 0.0413 [0.0194, 0.0678] | 0.1815 [0.1372, 0.2287] | -0.2218 [-0.2701, -0.1740] | 0.7364 [0.0292, 1.3799] | 4.9655 [3.4659, 6.5884] | 0.4887 [0.4391, 0.5413] | 2.6585 [2.3764, 2.9619] | 7.6865 [6.6276, 8.9726] | 1.0890 [0.7424, 1.4765] | 2.0166 [1.4838, 2.6347] |
| RLFT (0,1) | 0.1246 [0.0752, 0.1873] | 0.0421 [0.0206, 0.0660] | 0.1830 [0.1376, 0.2315] | -0.2121 [-0.2594, -0.1651] | 0.7561 [0.0790, 1.3428] | 4.9179 [3.3324, 6.5198] | 0.4837 [0.4329, 0.5361] | 2.6582 [2.3576, 2.9983] | 7.5785 [6.6350, 8.6730] | 1.0555 [0.7232, 1.4345] | 2.0488 [1.4877, 2.6843] |
| RLVc (0,1) | 0.1272 [0.0831, 0.1790] | 0.0428 [0.0211, 0.0687] | 0.1737 [0.1189, 0.2320] | -0.2160 [-0.2662, -0.1667] | 0.7477 [0.0497, 1.3922] | 4.6301 [2.7778, 6.4518] | 0.4886 [0.4382, 0.5413] | 2.6205 [2.3585, 2.9046] | 7.4467 [6.4346, 8.6076] | 1.0781 [0.7306, 1.4747] | 2.0261 [1.4831, 2.6768] |
| RLVd (0,1) | 0.1174 [0.0747, 0.1680] | 0.0382 [0.0179, 0.0617] | 0.1900 [0.1425, 0.2408] | -0.2181 [-0.2668, -0.1695] | 0.8079 [0.1631, 1.3935] | 5.3745 [3.8890, 6.9079] | 0.4859 [0.4369, 0.5375] | 2.6510 [2.3847, 2.9424] | 7.7842 [6.7588, 8.9295] | 1.0549 [0.7192, 1.4417] | 2.0948 [1.5460, 2.7341] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
