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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLA:RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLAm:RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLFTb:RLFT", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, RLA minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLAm minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLFTb minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## RLA minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0427 / -0.0480 | -0.0027 | [-0.0721, +0.0579] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0109 / +0.0140 | +0.0125 | [-0.0008, +0.0285] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0187 / +0.0421 | +0.0117 | [-0.0416, +0.0782] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0014 / +0.0828 | +0.0407 | [-0.0179, +0.0975] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.2949 / +0.7716 | +0.2383 | [-0.4608, +1.0878] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.3028 / +1.1081 | -0.0973 | [-1.9685, +2.0461] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0068 / -0.0350 | -0.0209 | [-0.0451, +0.0069] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0048 / -0.1171 | -0.0610 | [-0.2584, +0.1619] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.5332 / +0.6913 | +0.0791 | [-1.1421, +1.4991] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0507 / -0.1266 | -0.0380 | [-0.2031, +0.1048] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0082 / -0.3084 | -0.1583 | [-0.4426, +0.1093] | 1.7849 (0.8089) | OPEN |

## RLAm minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0107 / +0.0036 | +0.0071 | [-0.0021, +0.0164] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0047 / -0.0062 | -0.0008 | [-0.0107, +0.0082] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0047 / +0.0016 | -0.0016 | [-0.0158, +0.0128] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0118 / -0.0077 | -0.0098 | [-0.0174, -0.0033] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.0088 / -0.0482 | -0.0197 | [-0.1216, +0.1073] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.0859 / +0.1811 | +0.0476 | [-0.4239, +0.4633] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0032 / +0.0067 | +0.0050 | [-0.0015, +0.0101] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0887 / -0.0880 | +0.0003 | [-0.1242, +0.1617] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.1491 / +0.0668 | +0.1080 | [-0.2452, +0.5049] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0701 / -0.0031 | +0.0335 | [-0.0221, +0.0999] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0907 / -0.1550 | -0.0321 | [-0.2251, +0.1370] | 1.7849 (0.8089) | OPEN |

## RLFTb minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0125 / -0.0178 | -0.0027 | [-0.0253, +0.0217] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0047 / -0.0047 | +0.0000 | [-0.0086, +0.0091] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0078 / -0.0031 | -0.0055 | [-0.0207, +0.0102] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0028 / +0.0059 | +0.0043 | [-0.0037, +0.0125] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0817 / +0.1227 | +0.0205 | [-0.1245, +0.1572] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.3829 / +0.4715 | +0.0443 | [-0.5818, +0.6466] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0063 / +0.0046 | -0.0009 | [-0.0111, +0.0072] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0146 / -0.0512 | -0.0183 | [-0.0849, +0.0460] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0682 / +0.0920 | +0.0119 | [-0.2366, +0.2716] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0045 / +0.0244 | +0.0145 | [-0.0213, +0.0578] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0342 / -0.0283 | -0.0313 | [-0.1006, +0.0418] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| RLA (0,1) | 0.1219 [0.0741, 0.1788] | 0.0545 [0.0316, 0.0794] | 0.1947 [0.1433, 0.2516] | -0.1714 [-0.2448, -0.1025] | 0.9945 [0.5086, 1.4651] | 4.8206 [3.4328, 6.4270] | 0.4628 [0.4078, 0.5256] | 2.5972 [2.2867, 2.9369] | 7.6576 [6.7027, 8.8512] | 1.0176 [0.6878, 1.3781] | 1.8904 [1.3301, 2.5688] |
| RLAm (0,1) | 0.1317 [0.0831, 0.1892] | 0.0413 [0.0194, 0.0678] | 0.1815 [0.1372, 0.2287] | -0.2218 [-0.2701, -0.1740] | 0.7364 [0.0292, 1.3799] | 4.9655 [3.4659, 6.5884] | 0.4887 [0.4391, 0.5413] | 2.6585 [2.3764, 2.9619] | 7.6865 [6.6276, 8.9726] | 1.0890 [0.7424, 1.4765] | 2.0166 [1.4838, 2.6347] |
| RLFT (0,1) | 0.1246 [0.0752, 0.1873] | 0.0421 [0.0206, 0.0660] | 0.1830 [0.1376, 0.2315] | -0.2121 [-0.2594, -0.1651] | 0.7561 [0.0790, 1.3428] | 4.9179 [3.3324, 6.5198] | 0.4837 [0.4329, 0.5361] | 2.6582 [2.3576, 2.9983] | 7.5785 [6.6350, 8.6730] | 1.0555 [0.7232, 1.4345] | 2.0488 [1.4877, 2.6843] |
| RLFTb (0,1) | 0.1219 [0.0806, 0.1692] | 0.0421 [0.0201, 0.0682] | 0.1776 [0.1339, 0.2232] | -0.2078 [-0.2544, -0.1605] | 0.7766 [0.2075, 1.2970] | 4.9622 [3.6614, 6.2299] | 0.4829 [0.4347, 0.5333] | 2.6399 [2.3524, 2.9458] | 7.5904 [6.7085, 8.5996] | 1.0700 [0.7272, 1.4666] | 2.0175 [1.4511, 2.6805] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
