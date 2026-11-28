# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 16, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSXC:LSXZ", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDXC:LSDXZ", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.
Dumps not present: LSXM2_s0, LSXM2_s1.

## Names of section 3 of the card

* None, cell U, LSXC minus LSXZ over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDXC minus LSDXZ over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* LSXM2_minus_LSXG: no seed with both arms; nothing named.

## LSXC minus LSXZ

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0089 / +0.0071 | -0.0009 | [-0.0180, +0.0139] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / -0.0047 | -0.0023 | [-0.0082, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0171 | +0.0086 | [-0.0098, +0.0299] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0481 / +0.0544 | +0.0512 | [+0.0040, +0.1084] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.0798 / +0.4121 | +0.1661 | [-0.2704, +0.6194] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.3138 / +0.8499 | +0.2680 | [-0.7377, +1.4852] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0406 / +0.0335 | +0.0371 | [-0.0067, +0.0869] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.1463 / +0.3098 | +0.0817 | [-0.3062, +0.5006] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.2398 / +0.7895 | +0.2748 | [-0.6608, +1.4225] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0305 / +0.0258 | -0.0023 | [-0.0557, +0.0523] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0257 / +0.0282 | +0.0269 | [-0.0705, +0.1227] | 1.7849 (0.8089) | OPEN |

## LSDXC minus LSDXZ

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0196 / +0.0125 | +0.0160 | [+0.0043, +0.0299] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0031 / -0.0078 | -0.0023 | [-0.0112, +0.0049] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0062 / -0.0062 | -0.0062 | [-0.0181, +0.0058] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0172 / +0.0597 | +0.0385 | [+0.0025, +0.0948] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.1266 / +0.0992 | +0.1129 | [-0.0701, +0.3218] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.3325 / +0.3386 | +0.3355 | [-0.0476, +0.8092] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0082 / +0.0552 | +0.0317 | [-0.0019, +0.0878] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.1293 / +0.1240 | +0.1267 | [-0.0472, +0.3351] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.3503 / +0.3519 | +0.3511 | [-0.0296, +0.8221] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0235 / -0.0226 | -0.0230 | [-0.0545, +0.0009] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0832 / -0.0996 | -0.0914 | [-0.2267, +0.0107] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LSDXC (0,1) | 0.0774 [0.0473, 0.1158] | 0.0171 [0.0042, 0.0385] | 0.3255 [0.2582, 0.3932] | 0.2838 [0.0126, 0.6204] | 2.9890 [1.8870, 4.1690] | 10.0100 [7.6793, 12.3139] | 0.8679 [0.6258, 1.1778] | 3.9884 [3.0501, 5.0176] | 11.1839 [9.2636, 13.2363] | 1.4949 [0.8513, 2.2420] | 2.6935 [1.7438, 3.7231] |
| LSDXZ (0,1) | 0.0614 [0.0323, 0.1030] | 0.0195 [0.0032, 0.0467] | 0.3318 [0.2650, 0.3969] | 0.2453 [-0.0049, 0.5551] | 2.8761 [1.8262, 4.0054] | 9.6745 [7.5776, 11.6781] | 0.8362 [0.6095, 1.1196] | 3.8617 [2.9711, 4.8228] | 10.8328 [9.1522, 12.5669] | 1.5179 [0.8692, 2.2734] | 2.7849 [1.8421, 3.8356] |
| LSXC (0,1) | 0.0685 [0.0363, 0.1122] | 0.0218 [0.0072, 0.0433] | 0.2975 [0.2024, 0.3978] | 0.2637 [-0.0212, 0.6042] | 2.9305 [1.9758, 3.9639] | 9.2626 [6.7534, 11.7512] | 0.8931 [0.6464, 1.2029] | 3.9922 [3.2155, 4.8727] | 10.7310 [8.6213, 12.9089] | 1.2963 [0.7856, 1.8377] | 2.7557 [1.7348, 3.9155] |
| LSXG (0,1) | 0.0774 [0.0481, 0.1123] | 0.0156 [0.0053, 0.0288] | 0.2936 [0.2166, 0.3781] | 0.2605 [-0.0090, 0.5814] | 2.7694 [1.9052, 3.7896] | 8.8331 [6.7092, 10.9397] | 0.8811 [0.6359, 1.1904] | 3.9000 [3.1509, 4.7730] | 10.2834 [8.4180, 12.1821] | 1.3624 [0.9040, 1.8612] | 2.8131 [1.9603, 3.7515] |
| LSXZ (0,1) | 0.0694 [0.0419, 0.1063] | 0.0241 [0.0072, 0.0500] | 0.2889 [0.1850, 0.3985] | 0.2125 [-0.0567, 0.5367] | 2.7644 [1.6694, 3.9942] | 8.9946 [6.2010, 12.0337] | 0.8560 [0.6312, 1.1440] | 3.9105 [2.9618, 4.9515] | 10.4562 [8.0463, 13.2088] | 1.2986 [0.7759, 1.8737] | 2.7288 [1.6905, 3.9145] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
