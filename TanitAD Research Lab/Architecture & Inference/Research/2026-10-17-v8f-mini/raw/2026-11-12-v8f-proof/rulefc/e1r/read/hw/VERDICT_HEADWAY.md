# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:\TanitAD-scratch\v8proof\rhacc10k/pins/join_eval141.npz. Alignment reference (ego speed): F:\TanitAD-scratch\v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 6, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09891171799189782, "A_hi": 0.16379738797148666, "spread_max": 0.005338078291814943, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "H_RQ50KE1:H_RQ50K", "seeds": [0, 1, 2]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm H_RQ50K, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, H_RQ50KE1 minus H_RQ50K over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## H RQ50KE1 minus H RQ50K

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0178 / -0.0089 / -0.0160 | -0.0142 | [-0.0249, -0.0059] | 0.0053 (0.0053) | BETTER |
| C contact rate | -0.0280 / -0.0187 / -0.0265 | -0.0244 | [-0.0404, -0.0099] | 0.0031 (0.0031) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0109 / +0.0078 / +0.0125 | +0.0104 | [+0.0043, +0.0175] | 0.0498 (0.0249) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1519 / +0.0914 / +0.1457 | +0.1296 | [+0.0815, +0.1811] | 0.2647 (0.2418) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.2551 / +0.2334 / +0.2493 | +0.2459 | [+0.1754, +0.3174] | 0.6949 (0.4278) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3845 / +0.4171 / +0.3804 | +0.3940 | [+0.2750, +0.5146] | 1.3246 (0.6677) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1096 / +0.0632 / +0.1100 | +0.0943 | [+0.0492, +0.1426] | 0.2309 (0.2303) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.0951 / +0.0797 / +0.0841 | +0.0863 | [+0.0137, +0.1557] | 0.5582 (0.3880) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.0357 / -0.0685 / -0.0147 | -0.0396 | [-0.1749, +0.0906] | 0.8319 (0.4581) | OPEN |
| time-gap error 2.0 s (s) | +0.0451 / +0.0167 / +0.0371 | +0.0329 | [+0.0135, +0.0612] | 0.0965 (0.0494) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0422 / -0.0048 / +0.0361 | +0.0245 | [-0.0124, +0.0703] | 0.2763 (0.1579) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H_RQ50K (0,1,2) | 0.0747 [0.0464, 0.1060] | 0.0363 [0.0171, 0.0575] | 0.1636 [0.1222, 0.2056] | 0.6197 [0.3730, 0.8805] | 1.9001 [1.3104, 2.5168] | 3.8743 [2.7267, 5.0904] | 0.9244 [0.7043, 1.1591] | 3.0874 [2.6423, 3.5753] | 6.3648 [5.6273, 7.2000] | 1.0783 [0.8082, 1.3770] | 2.3566 [1.7622, 3.0473] |
| H_RQ50KE1 (0,1,2) | 0.0605 [0.0355, 0.0863] | 0.0119 [0.0025, 0.0251] | 0.1739 [0.1300, 0.2174] | 0.7493 [0.4586, 1.0556] | 2.1461 [1.5141, 2.7781] | 4.2684 [3.1364, 5.4539] | 1.0187 [0.7554, 1.2984] | 3.1737 [2.6895, 3.6906] | 6.3251 [5.5607, 7.1888] | 1.1113 [0.8347, 1.4190] | 2.3811 [1.7742, 3.0801] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
