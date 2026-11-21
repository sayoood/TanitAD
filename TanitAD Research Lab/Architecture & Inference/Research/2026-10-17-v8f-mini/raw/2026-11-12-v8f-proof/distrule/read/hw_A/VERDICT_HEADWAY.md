# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 24, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.1052699410116895, "A_hi": 0.1509396295606466, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "HIERRL:HIER", "seeds": [0, 1, 2, 3, 4, 5]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRL:G", "seeds": [0, 1, 2, 3, 4, 5]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, HIERRL minus HIER over seeds [0, 1, 2, 3, 4, 5] of [0, 1, 2, 3, 4, 5]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRL minus G over seeds [0, 1, 2, 3, 4, 5] of [0, 1, 2, 3, 4, 5]: **NOT NAMED (no question of the card for this pair)** (decides nothing) Sensitivity (names nothing): with the spread of G (max 0.0463) the rule on U reads BETTER.

## HIERRL minus HIER

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0623 / -0.0890 / -0.1050 / -0.0747 / -0.0641 / -0.0765 | -0.0786 | [-0.1015, -0.0557] | 0.0463 (0.0338) | BETTER |
| C contact rate | -0.0234 / -0.0607 / -0.0452 / -0.0483 / -0.0421 / -0.0483 | -0.0447 | [-0.0601, -0.0293] | 0.0374 (0.0140) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0639 / +0.0607 / +0.0732 / +0.0592 / +0.0592 / +0.0483 | +0.0607 | [+0.0390, +0.0841] | 0.1604 (0.0623) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4385 / +0.4475 / +0.4455 / +0.4430 / +0.4495 / +0.4363 | +0.4434 | [+0.2935, +0.6199] | 0.0695 (0.0312) | WORSE |
| gap bias 4.0 s (m) | +1.3935 / +1.2867 / +1.4315 / +1.3960 / +1.4045 / +1.3825 | +1.3824 | [+0.9470, +1.8617] | 1.2254 (0.4997) | WORSE |
| gap bias 6.0 s (m) | +2.6423 / +3.3435 / +3.0950 / +2.7448 / +2.6387 / +2.1289 | +2.7655 | [+1.8349, +3.7789] | 3.4187 (1.3133) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3256 / +0.2920 / +0.3226 / +0.3194 / +0.3012 / +0.3137 | +0.3124 | [+0.1823, +0.4722] | 0.0320 (0.0143) | WORSE |
| gap error 4.0 s (m) | +0.7281 / +0.5477 / +0.5973 / +0.6231 / +0.6361 / +0.7050 | +0.6396 | [+0.3113, +1.0319] | 0.7686 (0.2688) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.3945 / +1.1504 / +1.2291 / +1.4405 / +1.0335 / +1.0655 | +1.2189 | [+0.5628, +1.9336] | 1.9287 (0.9057) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0507 / +0.0387 / +0.0438 / +0.0463 / +0.0402 / +0.0418 | +0.0436 | [+0.0292, +0.0595] | 0.6521 (0.2607) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1939 / +0.0679 / +0.0275 / +0.1083 / +0.1226 / +0.1004 | +0.1035 | [+0.0400, +0.1756] | 1.4163 (0.6721) | INSIDE-SPREAD |

## GRL minus G

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0765 / -0.0730 / -0.0996 / -0.0658 / -0.0747 / -0.0890 | -0.0798 | [-0.1024, -0.0596] | 0.0463 (0.0338) | BETTER |
| C contact rate | -0.0280 / -0.0670 / -0.0421 / -0.0436 / -0.0592 / -0.0265 | -0.0444 | [-0.0612, -0.0285] | 0.0374 (0.0140) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0841 / +0.0623 / +0.0763 / +0.0639 / +0.0623 / +0.0592 | +0.0680 | [+0.0461, +0.0932] | 0.1604 (0.0623) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4359 / +0.4418 / +0.4390 / +0.4439 / +0.4403 / +0.4385 | +0.4399 | [+0.2906, +0.6160] | 0.0695 (0.0312) | WORSE |
| gap bias 4.0 s (m) | +1.3721 / +1.2025 / +1.3792 / +1.1924 / +1.2474 / +1.2615 | +1.2758 | [+0.8760, +1.7459] | 1.2254 (0.4997) | WORSE |
| gap bias 6.0 s (m) | +3.1209 / +3.6189 / +2.9974 / +2.6435 / +3.3050 / +2.4222 | +3.0180 | [+2.1331, +4.0859] | 3.4187 (1.3133) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3366 / +0.3087 / +0.3175 / +0.3093 / +0.3124 / +0.3242 | +0.3181 | [+0.1888, +0.4757] | 0.0320 (0.0143) | WORSE |
| gap error 4.0 s (m) | +0.7863 / +0.6249 / +0.6234 / +0.6220 / +0.6097 / +0.7592 | +0.6709 | [+0.3507, +1.0668] | 0.7686 (0.2688) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +2.1574 / +0.8260 / +1.0819 / +1.1047 / +1.3204 / +1.2367 | +1.2879 | [+0.5460, +2.1575] | 1.9287 (0.9057) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0508 / +0.0426 / +0.0438 / +0.0435 / +0.0410 / +0.0466 | +0.0447 | [+0.0309, +0.0598] | 0.6521 (0.2607) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.2006 / +0.0598 / +0.1088 / +0.1126 / +0.0638 / +0.0982 | +0.1073 | [+0.0502, +0.1715] | 1.4163 (0.6721) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GRL (0,1,2,3,4,5) | 0.0498 [0.0331, 0.0686] | 0.0008 [0.0000, 0.0020] | 0.2840 [0.2286, 0.3420] | 0.2176 [0.0686, 0.3899] | 2.5170 [1.9197, 3.1655] | 8.6467 [7.1111, 10.2343] | 0.8071 [0.6746, 0.9648] | 3.4905 [3.0022, 4.0515] | 9.7943 [8.4630, 11.1687] | 1.2343 [0.9316, 1.5712] | 2.4892 [1.8531, 3.1744] |
| HIER (0,1,2,3,4,5) | 0.1275 [0.0970, 0.1581] | 0.0460 [0.0304, 0.0618] | 0.2349 [0.1823, 0.2870] | -0.2448 [-0.2793, -0.2098] | 1.1386 [0.7628, 1.5582] | 6.1523 [4.8920, 7.4125] | 0.5157 [0.4875, 0.5450] | 2.8288 [2.5932, 3.1023] | 8.6505 [7.8085, 9.4642] | 1.1843 [0.9174, 1.4851] | 2.3249 [1.8077, 2.8695] |
| HIERRL (0,1,2,3,4,5) | 0.0489 [0.0343, 0.0649] | 0.0013 [0.0002, 0.0026] | 0.2957 [0.2399, 0.3500] | 0.1985 [0.0509, 0.3703] | 2.5210 [1.9961, 3.0934] | 8.9178 [7.7097, 10.1120] | 0.8281 [0.6945, 0.9861] | 3.4683 [3.0648, 3.9369] | 9.8694 [8.8302, 10.9518] | 1.2279 [0.9606, 1.5258] | 2.4284 [1.9108, 2.9911] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
