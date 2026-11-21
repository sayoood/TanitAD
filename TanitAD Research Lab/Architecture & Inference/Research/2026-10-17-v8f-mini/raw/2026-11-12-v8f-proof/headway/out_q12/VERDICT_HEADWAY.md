# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 18, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.1052699410116895, "A_hi": 0.1509396295606466, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "HIER:FLAT", "seeds": [0, 1, 2, 3, 4, 5]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "G:HIER", "seeds": [0, 1, 2, 3, 4, 5]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* Q1, cell U, HIER minus FLAT over seeds [0, 1, 2, 3, 4, 5] of [0, 1, 2, 3, 4, 5]: **DISTANCE-EQUAL** Sensitivity (names nothing): with the spread of FLAT (max 0.0374) the rule on U reads OPEN.
* Q2, cell U, G minus HIER over seeds [0, 1, 2, 3, 4, 5] of [0, 1, 2, 3, 4, 5]: **GRID-DISTANCE-EQUAL**

## HIER minus FLAT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0267 / +0.0160 / +0.0214 / -0.0214 / -0.0320 / +0.0089 | -0.0056 | [-0.0269, +0.0140] | 0.0463 (0.0338) | OPEN |
| C contact rate | -0.0156 / +0.0156 / -0.0047 / +0.0078 / +0.0047 / +0.0047 | +0.0021 | [-0.0097, +0.0132] | 0.0374 (0.0140) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.1044 / -0.0296 / +0.0607 / +0.1028 / +0.0389 / +0.0452 | +0.0537 | [+0.0038, +0.0992] | 0.1604 (0.0623) | MIXED-SIGN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1313 / -0.0737 / -0.0710 / -0.0644 / -0.1423 / -0.0446 | -0.0879 | [-0.1260, -0.0547] | 0.0695 (0.0312) | BETTER |
| gap bias 4.0 s (m) | +1.0125 / -0.1320 / +0.5866 / +0.4986 / +0.1433 / -0.2096 | +0.3166 | [-0.0413, +0.6770] | 1.2254 (0.4997) | OPEN |
| gap bias 6.0 s (m) | +3.4792 / +0.1530 / +1.9372 / +3.0528 / +2.9897 / +1.8425 | +2.2424 | [+0.9672, +3.5456] | 3.4187 (1.3133) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0618 / +0.0342 / +0.0373 / +0.0278 / +0.0670 / +0.0066 | +0.0391 | [+0.0173, +0.0606] | 0.0320 (0.0143) | WORSE |
| gap error 4.0 s (m) | +0.5016 / -0.3749 / +0.2993 / +0.2038 / +0.1743 / +0.0636 | +0.1446 | [-0.1100, +0.3687] | 0.7686 (0.2688) | OPEN |
| gap error 6.0 s (m) | +1.6091 / +0.0963 / +0.7825 / +0.9663 / +0.6753 / +0.7300 | +0.8099 | [+0.0288, +1.5856] | 1.9287 (0.9057) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.4593 / -0.0659 / +0.2972 / +0.2364 / -0.0306 / +0.1703 | +0.1778 | [+0.0119, +0.3703] | 0.6521 (0.2607) | MIXED-SIGN |
| time-gap error over the scored steps (s) | +1.0392 / -0.3923 / +0.3981 / +0.2504 / -0.3071 / +0.3550 | +0.2239 | [-0.1775, +0.6285] | 1.4163 (0.6721) | OPEN |

## G minus HIER

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / -0.0125 / +0.0000 / -0.0107 / +0.0267 / +0.0018 | +0.0021 | [-0.0136, +0.0175] | 0.0463 (0.0338) | OPEN |
| C contact rate | +0.0031 / +0.0062 / -0.0031 / -0.0047 / +0.0156 / -0.0218 | -0.0008 | [-0.0145, +0.0103] | 0.0374 (0.0140) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0031 / +0.0156 / +0.0125 / -0.0327 / -0.0452 / -0.0607 | -0.0190 | [-0.0510, +0.0119] | 0.1604 (0.0623) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0149 / +0.0782 / +0.0223 / -0.0059 / +0.0355 / -0.0098 | +0.0225 | [-0.0033, +0.0504] | 0.0695 (0.0312) | OPEN |
| gap bias 4.0 s (m) | +0.2693 / +0.2709 / +0.4061 / +0.0749 / -0.3563 / -0.0494 | +0.1026 | [-0.1411, +0.3494] | 1.2254 (0.4997) | OPEN |
| gap bias 6.0 s (m) | +0.8364 / -0.2996 / +0.2359 / -0.1489 / -2.2949 / -1.4704 | -0.5236 | [-1.5115, +0.4858] | 3.4187 (1.3133) | OPEN |
| gap error 2.0 s (m) | -0.0302 / -0.0520 / -0.0031 / -0.0053 / -0.0352 / -0.0345 | -0.0267 | [-0.0439, -0.0090] | 0.0320 (0.0143) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.1654 / -0.0233 / +0.4361 / -0.0484 / -0.3634 / -0.2218 | -0.0092 | [-0.2216, +0.2187] | 0.7686 (0.2688) | OPEN |
| gap error 6.0 s (m) | +0.3697 / +0.3361 / +0.6874 / +0.0896 / -1.3062 / -1.0410 | -0.1441 | [-0.8828, +0.5609] | 1.9287 (0.9057) | OPEN |
| time-gap error 2.0 s (s) | +0.1345 / +0.0502 / +0.2835 / -0.0646 / -0.1973 / -0.1749 | +0.0052 | [-0.1384, +0.1623] | 0.6521 (0.2607) | OPEN |
| time-gap error over the scored steps (s) | +0.2117 / -0.0829 / +0.6255 / -0.0203 / -0.1570 / -0.2350 | +0.0570 | [-0.1701, +0.3556] | 1.4163 (0.6721) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FLAT (0,1,2,3,4,5) | 0.1332 [0.1040, 0.1641] | 0.0439 [0.0306, 0.0571] | 0.1812 [0.1452, 0.2195] | -0.1570 [-0.1948, -0.1158] | 0.8220 [0.4529, 1.2295] | 3.9099 [2.6848, 5.1774] | 0.4766 [0.4493, 0.5050] | 2.6841 [2.4928, 2.8907] | 7.8406 [7.1873, 8.5532] | 1.0065 [0.8125, 1.2114] | 2.1010 [1.7433, 2.4878] |
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| HIER (0,1,2,3,4,5) | 0.1275 [0.0970, 0.1581] | 0.0460 [0.0304, 0.0618] | 0.2349 [0.1823, 0.2870] | -0.2448 [-0.2793, -0.2098] | 1.1386 [0.7628, 1.5582] | 6.1523 [4.8920, 7.4125] | 0.5157 [0.4875, 0.5450] | 2.8288 [2.5932, 3.1023] | 8.6505 [7.8085, 9.4642] | 1.1843 [0.9174, 1.4851] | 2.3249 [1.8077, 2.8695] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
