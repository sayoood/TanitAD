# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 7, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.0752218477307355, "A_hi": 0.19026440814805612, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "HIERRP:HIER", "seeds": [0]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm HIER, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, HIERRP minus HIER over seeds [0] of [0]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## HIERRP minus HIER

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0569 | +0.0569 | [+0.0222, +0.0927] | 0.0463 (0.0338) | WORSE |
| C contact rate | +0.0498 | +0.0498 | [+0.0216, +0.0847] | 0.0374 (0.0140) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.1106 | -0.1106 | [-0.1530, -0.0722] | 0.1604 (0.0623) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.9535 | -0.9535 | [-0.9755, -0.9274] | 0.0695 (0.0312) | BETTER |
| gap bias 4.0 s (m) | -2.7518 | -2.7518 | [-2.8706, -2.6239] | 1.2254 (0.4997) | BETTER |
| gap bias 6.0 s (m) | -4.6353 | -4.6353 | [-5.2898, -3.9385] | 3.4187 (1.3133) | BETTER |
| gap error 2.0 s (m) | +0.7149 | +0.7149 | [+0.6491, +0.7759] | 0.0320 (0.0143) | WORSE |
| gap error 4.0 s (m) | +0.1363 | +0.1363 | [-0.3932, +0.6299] | 0.7686 (0.2688) | OPEN |
| gap error 6.0 s (m) | -0.9025 | -0.9025 | [-1.9666, +0.1822] | 1.9287 (0.9057) | OPEN |
| time-gap error 2.0 s (s) | -0.4041 | -0.4041 | [-0.7423, -0.1390] | 0.6521 (0.2607) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.6141 | -0.6141 | [-1.0325, -0.2670] | 1.4163 (0.6721) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HIER (0,1,2,3,4,5) | 0.1275 [0.0970, 0.1581] | 0.0460 [0.0304, 0.0618] | 0.2349 [0.1823, 0.2870] | -0.2448 [-0.2793, -0.2098] | 1.1386 [0.7628, 1.5582] | 6.1523 [4.8920, 7.4125] | 0.5157 [0.4875, 0.5450] | 2.8288 [2.5932, 3.1023] | 8.6505 [7.8085, 9.4642] | 1.1843 [0.9174, 1.4851] | 2.3249 [1.8077, 2.8695] |
| HIERRP (0) | 0.1637 [0.1002, 0.2314] | 0.0748 [0.0352, 0.1202] | 0.1963 [0.1258, 0.2710] | -1.1696 [-1.2430, -1.0978] | -0.9249 [-1.8501, 0.0248] | 2.7853 [0.1384, 5.4575] | 1.2382 [1.1704, 1.3076] | 3.3665 [2.9197, 3.8558] | 8.7217 [7.4309, 10.1626] | 1.1350 [0.6959, 1.6491] | 2.4884 [1.6613, 3.5337] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
