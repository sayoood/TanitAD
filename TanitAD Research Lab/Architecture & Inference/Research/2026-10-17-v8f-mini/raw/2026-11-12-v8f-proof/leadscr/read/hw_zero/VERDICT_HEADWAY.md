# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 10, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LCS:LCSZ", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LCS minus LCSZ over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LCS minus LCSZ

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / +0.0000 | -0.0009 | [-0.0069, +0.0051] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0031 | +0.0016 | [+0.0000, +0.0052] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0047 / +0.0093 | +0.0023 | [-0.0109, +0.0147] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0021 / -0.0118 | -0.0049 | [-0.0172, +0.0042] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.1218 / +0.0827 | +0.1022 | [+0.0539, +0.1503] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.1366 / +0.1921 | +0.1644 | [-0.0318, +0.3459] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0025 / +0.0101 | +0.0063 | [+0.0004, +0.0156] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.0505 / +0.0454 | +0.0480 | [+0.0193, +0.0800] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.0017 / +0.2046 | +0.1015 | [-0.1109, +0.2946] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0401 / +0.0306 | +0.0354 | [+0.0196, +0.0549] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0595 / +0.0377 | +0.0486 | [-0.0070, +0.1045] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LCS (0,1) | 0.1210 [0.0777, 0.1708] | 0.0545 [0.0288, 0.0863] | 0.1893 [0.1297, 0.2491] | -0.2273 [-0.2956, -0.1736] | 1.0989 [0.6024, 1.6114] | 5.2059 [3.6945, 6.6401] | 0.5147 [0.4615, 0.5802] | 2.6783 [2.3307, 3.0695] | 8.0179 [6.9642, 9.2438] | 1.2085 [0.7878, 1.6529] | 2.1674 [1.4069, 2.9766] |
| LCSZ (0,1) | 0.1219 [0.0790, 0.1724] | 0.0530 [0.0281, 0.0835] | 0.1869 [0.1226, 0.2516] | -0.2225 [-0.2828, -0.1731] | 0.9967 [0.5256, 1.5074] | 5.0415 [3.5683, 6.4845] | 0.5084 [0.4579, 0.5709] | 2.6303 [2.2944, 3.0091] | 7.9164 [6.8842, 9.0384] | 1.1731 [0.7634, 1.6101] | 2.1188 [1.3706, 2.9283] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
