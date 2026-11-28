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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LC:LCZ", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LCO:LCOZ", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LC minus LCZ over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LCO minus LCOZ over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LC minus LCZ

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / -0.0018 | +0.0018 | [-0.0036, +0.0094] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0031 / +0.0016 | +0.0023 | [-0.0019, +0.0073] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0016 | +0.0008 | [-0.0022, +0.0036] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0057 / -0.0026 | -0.0041 | [-0.0065, -0.0021] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.0143 / +0.0121 | +0.0132 | [+0.0019, +0.0251] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.0616 / +0.0567 | +0.0592 | [-0.0367, +0.1438] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0016 / +0.0004 | +0.0010 | [+0.0001, +0.0024] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.0051 / -0.0152 | -0.0051 | [-0.0216, +0.0099] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0599 / +0.0195 | +0.0397 | [-0.0206, +0.1067] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0082 / -0.0004 | +0.0039 | [-0.0026, +0.0130] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0004 / -0.0061 | -0.0032 | [-0.0172, +0.0110] | 1.7849 (0.8089) | OPEN |

## LCO minus LCOZ

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / -0.0053 | -0.0036 | [-0.0109, +0.0020] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0156 / +0.0140 | +0.0148 | [+0.0074, +0.0230] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0218 / +0.0329 | +0.0273 | [+0.0179, +0.0379] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.1226 / +0.1394 | +0.1310 | [+0.0971, +0.1652] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3665 / +0.2362 | +0.3013 | [+0.1700, +0.4241] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0123 / +0.0234 | +0.0178 | [+0.0082, +0.0287] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.1057 / +0.1020 | +0.1039 | [+0.0712, +0.1379] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.2434 / +0.2231 | +0.2332 | [+0.1448, +0.3274] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0347 / +0.0712 | +0.0530 | [+0.0240, +0.0913] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0833 / +0.0595 | +0.0714 | [+0.0414, +0.1022] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LC (0,1) | 0.1246 [0.0809, 0.1708] | 0.0530 [0.0295, 0.0793] | 0.1807 [0.1295, 0.2356] | -0.1975 [-0.2479, -0.1477] | 0.8489 [0.3548, 1.3336] | 4.5461 [2.8669, 6.2820] | 0.4755 [0.4254, 0.5289] | 2.6780 [2.3775, 3.0159] | 7.9414 [6.9331, 9.0713] | 0.9822 [0.6653, 1.3655] | 2.3237 [1.6840, 3.0560] |
| LCO (0,1) | 0.0783 [0.0490, 0.1114] | 0.0226 [0.0105, 0.0354] | 0.4299 [0.3645, 0.4948] | 2.0531 [1.2651, 2.9565] | 7.6624 [5.3794, 10.2552] | 17.1647 [12.9244, 21.7622] | 2.3193 [1.5291, 3.1990] | 8.3420 [6.0987, 10.9234] | 18.2148 [14.1357, 22.7485] | 2.0630 [1.4692, 2.8140] | 3.5297 [2.8312, 4.4338] |
| LCOZ (0,1) | 0.0819 [0.0515, 0.1165] | 0.0226 [0.0105, 0.0354] | 0.4151 [0.3516, 0.4805] | 2.0258 [1.2391, 2.9263] | 7.5314 [5.2661, 10.1157] | 16.8633 [12.6653, 21.4461] | 2.3015 [1.5166, 3.1733] | 8.2382 [6.0091, 10.8104] | 17.9816 [13.9454, 22.4963] | 2.0100 [1.4410, 2.7364] | 3.4583 [2.7585, 4.3814] |
| LCZ (0,1) | 0.1228 [0.0801, 0.1688] | 0.0506 [0.0286, 0.0764] | 0.1799 [0.1282, 0.2352] | -0.1933 [-0.2429, -0.1443] | 0.8357 [0.3363, 1.3196] | 4.4870 [2.8313, 6.1927] | 0.4745 [0.4245, 0.5274] | 2.6831 [2.3761, 3.0274] | 7.9017 [6.9041, 8.9966] | 0.9783 [0.6616, 1.3629] | 2.3269 [1.6855, 3.0649] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
