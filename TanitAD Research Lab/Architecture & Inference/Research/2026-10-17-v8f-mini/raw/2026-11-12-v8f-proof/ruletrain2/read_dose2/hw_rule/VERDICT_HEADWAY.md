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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFT6RL:GFT6", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRW6RL:GRW6", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRW6:GFT6RL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRW6RL:GFT6RL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GFT6RL minus GFT6 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRW6RL minus GRW6 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRW6 minus GFT6RL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRW6RL minus GFT6RL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GFT6RL minus GFT6

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0783 | -0.0685 | [-0.1016, -0.0380] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0483 / -0.0498 | -0.0491 | [-0.0733, -0.0275] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0685 / +0.0607 | +0.0646 | [+0.0342, +0.1000] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4438 / +0.4387 | +0.4413 | [+0.1876, +0.7647] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.2634 / +1.2125 | +1.2379 | [+0.6461, +1.9387] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.7340 / +2.7329 | +2.7334 | [+1.6028, +4.1807] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3074 / +0.3187 | +0.3131 | [+0.0854, +0.6077] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7715 / +0.5187 | +0.6451 | [+0.1352, +1.2715] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.5212 / +1.0019 | +1.2615 | [+0.1358, +2.5194] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0357 / +0.0421 | +0.0389 | [+0.0159, +0.0666] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0976 / +0.0746 | +0.0861 | [+0.0103, +0.1684] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GRW6RL minus GRW6

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0480 / -0.0641 | -0.0560 | [-0.0877, -0.0286] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0389 / -0.0436 | -0.0413 | [-0.0656, -0.0210] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0343 / +0.0202 | +0.0273 | [+0.0113, +0.0496] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3273 / +0.3162 | +0.3217 | [+0.1372, +0.5506] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.2590 / +0.7162 | +0.9876 | [+0.3454, +1.9297] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.2303 / +1.9812 | +2.1057 | [+1.2283, +3.3188] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2674 / +0.2689 | +0.2682 | [+0.0834, +0.5044] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.8878 / +0.3839 | +0.6358 | [+0.0061, +1.5661] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.0719 / +0.5366 | +0.8043 | [-0.1601, +2.0889] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0359 / +0.0320 | +0.0340 | [+0.0131, +0.0587] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1350 / +0.0205 | +0.0778 | [-0.0224, +0.2277] | 1.7849 (0.8089) | OPEN |

## GRW6 minus GFT6RL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0569 / +0.0552 | +0.0560 | [+0.0290, +0.0862] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0421 / +0.0452 | +0.0436 | [+0.0224, +0.0683] | 0.0421 (0.0171) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0561 / -0.0156 | -0.0358 | [-0.0742, +0.0004] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1525 / -0.0093 | -0.0809 | [-0.2951, +0.1192] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.8444 / -0.1827 | -0.5135 | [-1.1307, +0.1296] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -2.9631 / -1.0762 | -2.0196 | [-3.5266, -0.5733] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.2010 / -0.1684 | -0.1847 | [-0.3843, -0.0235] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.4691 / -0.0271 | -0.2481 | [-0.7542, +0.2440] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -1.6387 / +0.2406 | -0.6991 | [-2.2017, +0.7209] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0189 / -0.2312 | -0.1250 | [-0.3484, +0.0051] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1539 / -0.3064 | -0.2302 | [-0.4843, -0.0204] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GRW6RL minus GFT6RL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / -0.0089 | +0.0000 | [-0.0152, +0.0153] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0031 / +0.0016 | +0.0023 | [-0.0013, +0.0070] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0218 / +0.0047 | -0.0086 | [-0.0403, +0.0188] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1748 / +0.3068 | +0.2408 | [+0.1387, +0.3581] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.4146 / +0.5335 | +0.4741 | [-0.0035, +1.0337] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.7328 / +0.9050 | +0.0861 | [-1.2957, +1.3974] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0664 / +0.1006 | +0.0835 | [+0.0208, +0.1581] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.4186 / +0.3567 | +0.3877 | [-0.0273, +0.9690] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.5668 / +0.7772 | +0.1052 | [-1.0737, +1.2796] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0171 / -0.1991 | -0.0910 | [-0.3167, +0.0389] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0189 / -0.2859 | -0.1524 | [-0.4522, +0.0877] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT6 (0,1) | 0.1201 [0.0734, 0.1731] | 0.0506 [0.0281, 0.0755] | 0.1900 [0.1253, 0.2584] | -0.2167 [-0.2618, -0.1654] | 0.9637 [0.4299, 1.4763] | 5.1432 [2.9081, 7.4572] | 0.4852 [0.4418, 0.5347] | 2.5879 [2.2987, 2.9080] | 7.9793 [6.5034, 9.6963] | 1.0540 [0.7300, 1.4381] | 2.1003 [1.4546, 2.8105] |
| GFT6RL (0,1) | 0.0516 [0.0252, 0.0847] | 0.0016 [0.0000, 0.0040] | 0.2547 [0.1786, 0.3317] | 0.2245 [-0.0320, 0.5406] | 2.2017 [1.4112, 3.0245] | 7.8766 [5.3921, 10.4152] | 0.7982 [0.5726, 1.1002] | 3.2331 [2.6311, 3.9487] | 9.2408 [7.2165, 11.4317] | 1.0929 [0.7677, 1.4801] | 2.1865 [1.5505, 2.9055] |
| GRW6 (0,1) | 0.1077 [0.0687, 0.1547] | 0.0452 [0.0232, 0.0702] | 0.2188 [0.1568, 0.2788] | 0.1436 [-0.0093, 0.3172] | 1.6882 [1.1773, 2.2185] | 5.8570 [4.1293, 7.5196] | 0.6136 [0.5051, 0.7452] | 2.9849 [2.6130, 3.4041] | 8.5417 [7.2976, 9.9050] | 0.9679 [0.6851, 1.2877] | 1.9563 [1.2813, 2.6723] |
| GRW6RL (0,1) | 0.0516 [0.0277, 0.0802] | 0.0039 [0.0007, 0.0083] | 0.2461 [0.1767, 0.3143] | 0.4653 [0.1593, 0.8376] | 2.6758 [1.7211, 3.8460] | 7.9627 [5.8289, 10.2345] | 0.8817 [0.6111, 1.2300] | 3.6207 [2.7735, 4.7184] | 9.3460 [7.5519, 11.4240] | 1.0019 [0.7204, 1.3234] | 2.0341 [1.3513, 2.7821] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
