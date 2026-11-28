# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:\TanitAD-scratch\v8proof\rhacc10k/pins/join_eval141.npz. Alignment reference (ego speed): F:\TanitAD-scratch\v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 12, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09891171799189782, "A_hi": 0.16379738797148666, "spread_max": 0.00533807829181495, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "H_RLcE1:H_RLcK", "seeds": [0, 1, 2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "H_RLcFC:H_RLcK", "seeds": [0, 1, 2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "H_RLcE1FC:H_RLcK", "seeds": [0, 1, 2]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm H_RLcK, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, H_RLcE1 minus H_RLcK over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, H_RLcFC minus H_RLcK over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, H_RLcE1FC minus H_RLcK over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## H RLcE1 minus H RLcK

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0142 / -0.0160 / -0.0142 | -0.0148 | [-0.0248, -0.0069] | 0.0053 (0.0053) | BETTER |
| C contact rate | +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0000 (0.0000) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0016 / +0.0031 / +0.0031 | +0.0026 | [+0.0004, +0.0057] | 0.0280 (0.0140) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0313 / +0.0310 / +0.0264 | +0.0296 | [+0.0181, +0.0430] | 0.0211 (0.0163) | WORSE |
| gap bias 4.0 s (m) | +0.1049 / +0.0893 / +0.0984 | +0.0975 | [+0.0602, +0.1417] | 0.2531 (0.1416) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.1294 / +0.1179 / +0.1510 | +0.1327 | [+0.0687, +0.2093] | 0.6842 (0.3428) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0241 / +0.0217 / +0.0219 | +0.0226 | [+0.0123, +0.0353] | 0.0061 (0.0042) | WORSE |
| gap error 4.0 s (m) | +0.0646 / +0.0538 / +0.0621 | +0.0602 | [+0.0262, +0.1019] | 0.1803 (0.1015) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.0680 / +0.0730 / +0.0926 | +0.0779 | [+0.0165, +0.1545] | 0.4058 (0.3232) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0079 / +0.0071 / +0.0061 | +0.0070 | [+0.0038, +0.0105] | 0.0431 (0.0392) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0097 / +0.0134 / +0.0157 | +0.0129 | [+0.0063, +0.0203] | 0.2057 (0.1322) | INSIDE-SPREAD |

## H RLcFC minus H RLcK

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0089 / -0.0214 / -0.0107 | -0.0136 | [-0.0247, -0.0035] | 0.0053 (0.0053) | BETTER |
| C contact rate | +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0000 (0.0000) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0062 / +0.0109 / +0.0187 | +0.0119 | [+0.0043, +0.0217] | 0.0280 (0.0140) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0549 / +0.0467 / +0.0497 | +0.0504 | [+0.0344, +0.0684] | 0.0211 (0.0163) | WORSE |
| gap bias 4.0 s (m) | +0.2115 / +0.1804 / +0.2087 | +0.2002 | [+0.1356, +0.2779] | 0.2531 (0.1416) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3486 / +0.3148 / +0.3940 | +0.3525 | [+0.2163, +0.5145] | 0.6842 (0.3428) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0391 / +0.0329 / +0.0389 | +0.0369 | [+0.0235, +0.0515] | 0.0061 (0.0042) | WORSE |
| gap error 4.0 s (m) | +0.1544 / +0.1312 / +0.1509 | +0.1455 | [+0.0852, +0.2148] | 0.1803 (0.1015) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.2710 / +0.2319 / +0.3340 | +0.2790 | [+0.1437, +0.4457] | 0.4058 (0.3232) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0620 / +0.0469 / +0.0480 | +0.0523 | [+0.0277, +0.0811] | 0.0431 (0.0392) | WORSE |
| time-gap error over the scored steps (s) | +0.0808 / +0.0762 / +0.0729 | +0.0766 | [+0.0348, +0.1281] | 0.2057 (0.1322) | INSIDE-SPREAD |

## H RLcE1FC minus H RLcK

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0178 / -0.0285 / -0.0267 | -0.0243 | [-0.0399, -0.0109] | 0.0053 (0.0053) | BETTER |
| C contact rate | +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0000 (0.0000) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0093 / +0.0125 / +0.0187 | +0.0135 | [+0.0065, +0.0224] | 0.0280 (0.0140) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0957 / +0.0879 / +0.0930 | +0.0922 | [+0.0626, +0.1243] | 0.0211 (0.0163) | WORSE |
| gap bias 4.0 s (m) | +0.2818 / +0.2656 / +0.2942 | +0.2805 | [+0.1982, +0.3748] | 0.2531 (0.1416) | WORSE |
| gap bias 6.0 s (m) | +0.4669 / +0.4913 / +0.5441 | +0.5008 | [+0.3388, +0.6944] | 0.6842 (0.3428) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0744 / +0.0673 / +0.0730 | +0.0716 | [+0.0457, +0.1024] | 0.0061 (0.0042) | WORSE |
| gap error 4.0 s (m) | +0.2037 / +0.1885 / +0.2085 | +0.2002 | [+0.1238, +0.2873] | 0.1803 (0.1015) | WORSE |
| gap error 6.0 s (m) | +0.3464 / +0.3561 / +0.4348 | +0.3791 | [+0.2121, +0.5748] | 0.4058 (0.3232) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0702 / +0.0578 / +0.0618 | +0.0633 | [+0.0352, +0.0961] | 0.0431 (0.0392) | WORSE |
| time-gap error over the scored steps (s) | +0.0911 / +0.0973 / +0.0922 | +0.0935 | [+0.0436, +0.1567] | 0.2057 (0.1322) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H_RLcE1 (0,1,2) | 0.0261 [0.0126, 0.0416] | 0.0000 [0.0000, 0.0000] | 0.1537 [0.1119, 0.2014] | 0.5440 [0.3108, 0.8090] | 1.8758 [1.3140, 2.5045] | 3.8695 [2.8218, 4.9927] | 0.8243 [0.6040, 1.0767] | 2.8193 [2.3492, 3.3677] | 5.7622 [5.0243, 6.6242] | 0.9762 [0.7334, 1.2644] | 2.1170 [1.5724, 2.7374] |
| H_RLcE1FC (0,1,2) | 0.0166 [0.0072, 0.0287] | 0.0000 [0.0000, 0.0000] | 0.1646 [0.1234, 0.2101] | 0.6066 [0.3596, 0.8890] | 2.0588 [1.5124, 2.6983] | 4.2376 [3.1866, 5.3530] | 0.8733 [0.6373, 1.1406] | 2.9593 [2.4931, 3.5108] | 6.0634 [5.3257, 6.9487] | 1.0324 [0.7817, 1.3129] | 2.1976 [1.6439, 2.8267] |
| H_RLcFC (0,1,2) | 0.0273 [0.0132, 0.0442] | 0.0000 [0.0000, 0.0000] | 0.1630 [0.1223, 0.2083] | 0.5648 [0.3314, 0.8267] | 1.9785 [1.4440, 2.6041] | 4.0893 [3.0586, 5.1726] | 0.8387 [0.6193, 1.0885] | 2.9046 [2.4502, 3.4362] | 5.9633 [5.2611, 6.8166] | 1.0215 [0.7744, 1.3030] | 2.1807 [1.6241, 2.8058] |
| H_RLcK (0,1,2) | 0.0409 [0.0212, 0.0642] | 0.0000 [0.0000, 0.0000] | 0.1511 [0.1099, 0.1977] | 0.5144 [0.2920, 0.7679] | 1.7782 [1.2417, 2.3763] | 3.7368 [2.7085, 4.7992] | 0.8017 [0.5912, 1.0434] | 2.7591 [2.3148, 3.2701] | 5.6843 [4.9906, 6.4977] | 0.9692 [0.7264, 1.2570] | 2.1041 [1.5574, 2.7253] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
