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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "PF:P", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, PF minus P over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## PF minus P

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0071 | -0.0160 | [-0.0455, +0.0101] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0093 | +0.0055 | [-0.0054, +0.0173] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0047 / -0.0125 | -0.0086 | [-0.0370, +0.0190] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1158 / -0.0162 | -0.0660 | [-0.1427, +0.0071] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.1826 / -0.6713 | -0.4269 | [-0.8931, -0.0002] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.1585 / -1.6386 | -0.8985 | [-2.2235, +0.3656] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0123 / -0.0195 | -0.0036 | [-0.0394, +0.0333] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0743 / -0.2587 | -0.0922 | [-0.4247, +0.2370] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0957 / -0.5831 | -0.3394 | [-1.2039, +0.4513] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.2816 / -0.1018 | -0.1917 | [-0.4233, -0.0436] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.1279 / -0.1308 | -0.1293 | [-0.3353, +0.0756] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| P (0,1) | 0.1352 [0.0890, 0.1798] | 0.0382 [0.0171, 0.0671] | 0.2477 [0.1946, 0.3078] | -0.1558 [-0.2789, -0.0360] | 1.6545 [1.1152, 2.2688] | 6.9048 [5.2846, 8.8537] | 0.5118 [0.4568, 0.5720] | 3.1075 [2.7086, 3.5814] | 9.3152 [8.0718, 10.9599] | 1.4488 [0.8795, 2.0939] | 2.7848 [1.9362, 3.7401] |
| PF (0,1) | 0.1192 [0.0765, 0.1627] | 0.0436 [0.0197, 0.0772] | 0.2391 [0.1816, 0.2982] | -0.2219 [-0.2841, -0.1596] | 1.2275 [0.6119, 1.9506] | 6.0063 [4.3811, 7.7250] | 0.5081 [0.4615, 0.5615] | 3.0154 [2.5671, 3.5929] | 8.9758 [7.7768, 10.3408] | 1.2571 [0.7845, 1.7751] | 2.6555 [1.8448, 3.5618] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
