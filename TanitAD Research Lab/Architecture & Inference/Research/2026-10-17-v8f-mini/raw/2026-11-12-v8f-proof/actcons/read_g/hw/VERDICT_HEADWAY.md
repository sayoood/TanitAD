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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLAm:RLFT", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLA3:RLFT", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLA2:RLFT", "seeds": [2, 3]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, RLAm minus RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLA3 minus RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLA2 minus RLFT over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## RLAm minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / +0.0036 | +0.0062 | [-0.0100, +0.0211] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0062 / -0.0047 | -0.0055 | [-0.0116, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0047 / -0.0234 | -0.0140 | [-0.0329, +0.0014] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0082 / +0.0128 | +0.0105 | [+0.0026, +0.0188] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.0893 / -0.0631 | +0.0131 | [-0.1124, +0.1203] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.2555 / -0.6090 | -0.1767 | [-0.7838, +0.3753] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0095 / -0.0080 | -0.0088 | [-0.0146, -0.0023] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.0202 / -0.0697 | -0.0450 | [-0.1058, +0.0137] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0888 / -0.3858 | -0.2373 | [-0.5257, +0.0295] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0927 / -0.0009 | +0.0459 | [-0.0463, +0.1276] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0752 / -0.1763 | -0.0505 | [-0.2340, +0.1201] | 1.7849 (0.8089) | OPEN |

## RLA3 minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0196 / -0.0018 | +0.0089 | [-0.0097, +0.0317] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0093 / -0.0093 | -0.0093 | [-0.0172, -0.0023] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0171 / +0.0000 | -0.0086 | [-0.0241, +0.0091] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0052 / +0.0053 | +0.0052 | [-0.0044, +0.0136] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0481 / +0.0909 | +0.0214 | [-0.0960, +0.1449] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.4588 / -0.2947 | -0.3767 | [-0.7079, -0.0659] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0058 / -0.0026 | -0.0042 | [-0.0105, +0.0036] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0596 / +0.0450 | +0.0523 | [-0.0057, +0.1113] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0889 / -0.0549 | +0.0170 | [-0.2260, +0.2534] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0816 / +0.0816 | +0.0816 | [+0.0341, +0.1387] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1133 / -0.0708 | +0.0212 | [-0.1139, +0.1743] | 1.7849 (0.8089) | OPEN |

## RLA2 minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / -0.0071 | -0.0018 | [-0.0140, +0.0098] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0031 / -0.0093 | -0.0062 | [-0.0144, +0.0027] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0249 / -0.0016 | -0.0132 | [-0.0314, +0.0052] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0046 / +0.0064 | +0.0055 | [-0.0030, +0.0133] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0773 / +0.0304 | -0.0235 | [-0.1263, +0.0882] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.5655 / -0.0626 | -0.3140 | [-0.7723, +0.1418] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0041 / -0.0034 | -0.0038 | [-0.0099, +0.0035] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.1011 / -0.0518 | -0.0764 | [-0.1442, -0.0104] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.3527 / -0.1542 | -0.2535 | [-0.5319, +0.0204] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0342 / +0.0065 | +0.0204 | [-0.0184, +0.0575] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0523 / -0.1456 | -0.0989 | [-0.2195, +0.0345] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| RLA2 (2,3) | 0.1290 [0.0861, 0.1741] | 0.0545 [0.0305, 0.0807] | 0.1534 [0.1030, 0.2076] | -0.2344 [-0.2952, -0.1765] | 0.7356 [0.2770, 1.2126] | 3.6489 [1.8969, 5.5096] | 0.4913 [0.4402, 0.5510] | 2.6202 [2.3385, 2.9242] | 6.9092 [5.9511, 7.9223] | 1.1844 [0.8110, 1.5917] | 2.0462 [1.4789, 2.6945] |
| RLA3 (2,3) | 0.1397 [0.0905, 0.1926] | 0.0514 [0.0283, 0.0772] | 0.1581 [0.1062, 0.2101] | -0.2347 [-0.2969, -0.1759] | 0.7805 [0.2855, 1.2685] | 3.5862 [1.8796, 5.3102] | 0.4908 [0.4398, 0.5502] | 2.7489 [2.4439, 3.0801] | 7.1797 [6.2842, 8.1074] | 1.2456 [0.8516, 1.6690] | 2.1663 [1.5681, 2.8594] |
| RLAm (2,3) | 0.1370 [0.0897, 0.1855] | 0.0553 [0.0309, 0.0821] | 0.1526 [0.1067, 0.1995] | -0.2294 [-0.2878, -0.1750] | 0.7722 [0.3178, 1.2553] | 3.7863 [2.4486, 5.1684] | 0.4863 [0.4369, 0.5437] | 2.6516 [2.3632, 2.9619] | 6.9254 [6.0739, 7.8562] | 1.2099 [0.8219, 1.6524] | 2.0946 [1.4948, 2.7981] |
| RLFT (2,3) | 0.1308 [0.0864, 0.1776] | 0.0607 [0.0359, 0.0870] | 0.1667 [0.1191, 0.2158] | -0.2399 [-0.3006, -0.1816] | 0.7591 [0.3016, 1.2441] | 3.9630 [2.3927, 5.5712] | 0.4950 [0.4454, 0.5507] | 2.6966 [2.4072, 3.0124] | 7.1627 [6.2266, 8.0914] | 1.1640 [0.7953, 1.5597] | 2.1451 [1.5350, 2.8354] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
