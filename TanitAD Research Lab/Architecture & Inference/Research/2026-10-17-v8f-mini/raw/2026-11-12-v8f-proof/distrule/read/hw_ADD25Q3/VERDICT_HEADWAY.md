# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 12, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LS:G", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSH:G", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSD:G", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* Q3, cell U, LS minus G over seeds [0, 1] of [0, 1]: **SUP-DISTANCE-EQUAL**
* Q3, cell U, LSH minus G over seeds [0, 1] of [0, 1]: **SUP-DISTANCE-EQUAL**
* Q3, cell U, LSD minus G over seeds [0, 1] of [0, 1]: **SUP-DISTANCE-EQUAL**

## LS minus G

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / -0.0053 | -0.0027 | [-0.0193, +0.0135] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0031 / -0.0109 | -0.0039 | [-0.0208, +0.0126] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0327 / -0.0109 | -0.0218 | [-0.0514, +0.0089] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0147 / -0.0989 | -0.0421 | [-0.1131, +0.0295] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.5373 / -0.0578 | -0.2976 | [-0.6855, +0.0292] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.3649 / +0.1304 | -0.6172 | [-1.8364, +0.6527] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0190 / +0.0462 | +0.0326 | [+0.0067, +0.0571] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.2558 / -0.0070 | -0.1314 | [-0.3818, +0.0787] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.8538 / -0.5122 | -0.6830 | [-1.3038, -0.0506] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.1309 / -0.0071 | -0.0690 | [-0.1960, +0.1147] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1763 / +0.1137 | -0.0313 | [-0.3142, +0.2396] | 1.7849 (0.8089) | OPEN |

## LSH minus G

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / +0.0089 | +0.0062 | [-0.0103, +0.0234] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0078 / -0.0140 | -0.0031 | [-0.0200, +0.0146] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0530 / +0.0000 | -0.0265 | [-0.0721, +0.0214] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0712 / -0.0076 | -0.0394 | [-0.0872, +0.0124] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.7621 / -0.1163 | -0.4392 | [-0.9040, +0.0725] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.6202 / +0.3801 | -0.6200 | [-2.1692, +1.1174] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0194 / -0.0064 | +0.0065 | [-0.0220, +0.0290] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.5287 / +0.0480 | -0.2404 | [-0.6506, +0.1375] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -1.4646 / -0.9946 | -1.2296 | [-2.0949, -0.3128] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.4370 / +0.0178 | -0.2096 | [-0.5547, +0.1085] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.4987 / +0.1250 | -0.1869 | [-0.6479, +0.2032] | 1.7849 (0.8089) | OPEN |

## LSD minus G

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / +0.0071 | +0.0071 | [-0.0099, +0.0266] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0093 / +0.0031 | -0.0031 | [-0.0182, +0.0092] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0296 / +0.0701 | +0.0202 | [-0.0482, +0.0920] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0260 / -0.0467 | -0.0103 | [-0.0658, +0.0473] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.2039 / +0.1988 | -0.0025 | [-0.3537, +0.3090] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.6373 / +2.0437 | +0.7032 | [-1.1540, +2.4861] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0200 / +0.0326 | +0.0263 | [+0.0058, +0.0512] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.1777 / +0.1374 | -0.0202 | [-0.3047, +0.2200] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.7492 / +1.3979 | +0.3244 | [-1.1261, +1.8059] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.2030 / +0.1142 | +0.1586 | [+0.0008, +0.3378] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.2081 / +0.3033 | +0.0476 | [-0.3435, +0.3789] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| LSD (0,1) | 0.1343 [0.0873, 0.1879] | 0.0452 [0.0109, 0.0906] | 0.2531 [0.1958, 0.3138] | -0.2031 [-0.2748, -0.1325] | 1.4817 [0.8172, 2.2957] | 6.7598 [4.6651, 8.6599] | 0.5100 [0.4630, 0.5586] | 2.8968 [2.3956, 3.5375] | 9.3372 [8.1968, 10.6886] | 1.4640 [0.8170, 2.2047] | 2.5391 [1.6003, 3.5518] |
| LSH (0,1) | 0.1335 [0.0883, 0.1849] | 0.0452 [0.0222, 0.0736] | 0.2064 [0.1380, 0.2897] | -0.2322 [-0.2989, -0.1671] | 1.0451 [0.4824, 1.6813] | 5.4365 [3.5098, 7.5196] | 0.4902 [0.4365, 0.5462] | 2.6766 [2.3186, 3.1134] | 7.7832 [6.5384, 9.2696] | 1.0958 [0.7200, 1.4891] | 2.3047 [1.4352, 3.2622] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
