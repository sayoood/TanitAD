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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFTRL:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRTRL:GRT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRT:GFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRTRL:GFTRL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GFTRL minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRTRL minus GRT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRT minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRTRL minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GFTRL minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0569 / -0.0783 | -0.0676 | [-0.1008, -0.0362] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0358 / -0.0452 | -0.0405 | [-0.0648, -0.0212] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0670 / +0.0685 | +0.0678 | [+0.0281, +0.1134] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4346 / +0.4358 | +0.4352 | [+0.1818, +0.7601] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.5137 / +1.1397 | +1.3267 | [+0.5822, +2.2604] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.8306 / +2.8614 | +2.8460 | [+1.3194, +4.6864] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3354 / +0.3216 | +0.3285 | [+0.0964, +0.6274] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7415 / +0.5542 | +0.6479 | [+0.0752, +1.3464] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.4310 / +1.1152 | +1.2731 | [+0.1971, +2.6265] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0448 / +0.0424 | +0.0436 | [+0.0203, +0.0708] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0738 / +0.0992 | +0.0865 | [+0.0093, +0.1686] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GRTRL minus GRT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0605 / -0.0641 | -0.0623 | [-0.0897, -0.0375] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0374 / -0.0421 | -0.0397 | [-0.0590, -0.0224] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0187 / +0.0483 | +0.0335 | [+0.0052, +0.0678] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3960 / +0.4117 | +0.4038 | [+0.1693, +0.7056] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.8440 / +1.0154 | +0.9297 | [+0.4899, +1.5660] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.3877 / +2.2114 | +1.7996 | [+0.9543, +2.9783] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3311 / +0.3245 | +0.3278 | [+0.1007, +0.6166] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.4805 / +0.6495 | +0.5650 | [+0.1027, +1.1748] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.2712 / +1.2947 | +0.7830 | [-0.1055, +2.0872] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0407 / +0.0399 | +0.0403 | [+0.0189, +0.0656] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0288 / +0.1446 | +0.0867 | [-0.0255, +0.2119] | 1.7849 (0.8089) | OPEN |

## GRT minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0605 / +0.0747 | +0.0676 | [+0.0411, +0.0956] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0374 / +0.0421 | +0.0397 | [+0.0224, +0.0590] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0062 / -0.0093 | -0.0016 | [-0.0377, +0.0301] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.3602 / -0.3787 | -0.3695 | [-0.6729, -0.1347] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -0.8762 / -0.6179 | -0.7471 | [-1.3233, -0.2427] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.9386 / -0.9435 | -0.9411 | [-2.0076, +0.0374] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.3386 / -0.3211 | -0.3298 | [-0.6148, -0.1012] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.5747 / -0.3529 | -0.4638 | [-1.0346, +0.0171] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.2083 / -0.0787 | +0.0648 | [-0.9418, +1.0157] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0463 / +0.0197 | -0.0133 | [-0.0837, +0.0540] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.4190 / +0.0538 | -0.1826 | [-0.5543, +0.1116] | 1.7849 (0.8089) | OPEN |

## GRTRL minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0107 | +0.0053 | [-0.0062, +0.0172] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0249 / +0.0389 | +0.0319 | [-0.0008, +0.0602] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0358 / +0.0329 | +0.0344 | [+0.0205, +0.0489] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.0323 / +0.3974 | +0.1826 | [-0.1545, +0.5033] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.4491 / +1.2679 | +0.8585 | [-0.1454, +1.7066] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0075 / +0.0034 | -0.0021 | [-0.0164, +0.0113] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0942 / +0.2966 | +0.1012 | [-0.1965, +0.3943] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.4795 / +1.2160 | +0.8477 | [-0.0868, +1.6783] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0056 / +0.0595 | +0.0269 | [-0.0414, +0.0869] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.3902 / +0.1984 | -0.0959 | [-0.5200, +0.2575] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT (0,1) | 0.1121 [0.0711, 0.1570] | 0.0405 [0.0212, 0.0648] | 0.1931 [0.1119, 0.2846] | -0.1611 [-0.2359, -0.0881] | 1.0823 [0.4765, 1.7433] | 5.2278 [2.7695, 7.6887] | 0.4680 [0.4239, 0.5183] | 2.7143 [2.3027, 3.2118] | 7.9076 [6.3657, 9.6000] | 1.2414 [0.7961, 1.7429] | 2.3332 [1.5451, 3.2287] |
| GFTRL (0,1) | 0.0445 [0.0255, 0.0664] | 0.0000 [0.0000, 0.0000] | 0.2609 [0.1758, 0.3522] | 0.2741 [0.0113, 0.5893] | 2.4091 [1.4418, 3.4539] | 8.0738 [5.4192, 10.5852] | 0.7965 [0.5692, 1.0945] | 3.3622 [2.6218, 4.1937] | 9.1807 [7.1953, 11.2347] | 1.2850 [0.8478, 1.7875] | 2.4197 [1.6515, 3.2980] |
| GRT (0,1) | 0.1121 [0.0751, 0.1520] | 0.0397 [0.0224, 0.0590] | 0.2593 [0.1678, 0.3553] | -0.0953 [-0.1798, -0.0122] | 1.6620 [0.9858, 2.3766] | 7.1328 [4.7012, 9.7188] | 0.4666 [0.4233, 0.5201] | 2.8984 [2.4723, 3.3656] | 9.2455 [7.2804, 11.4222] | 1.2717 [0.8459, 1.7530] | 2.2371 [1.6207, 2.9295] |
| GRTRL (0,1) | 0.0498 [0.0265, 0.0768] | 0.0000 [0.0000, 0.0000] | 0.2928 [0.2117, 0.3806] | 0.3085 [0.0411, 0.6250] | 2.5916 [1.7489, 3.4862] | 8.9323 [6.5521, 11.3041] | 0.7944 [0.5683, 1.1000] | 3.4634 [2.8113, 4.2598] | 10.0285 [8.1470, 11.9586] | 1.3119 [0.8873, 1.7925] | 2.3238 [1.7222, 2.9986] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
