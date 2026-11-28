# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 10, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50h:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50h:I10RQ50c", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50h:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50h:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50:I10R0", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RQ50h minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50h minus I10RQ50c over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50h minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50h minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RQ50h minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / -0.0036 | +0.0009 | [-0.0107, +0.0122] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0062 / +0.0000 | -0.0031 | [-0.0098, +0.0030] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0047 / -0.0187 | -0.0070 | [-0.0293, +0.0143] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0257 / -0.0085 | -0.0171 | [-0.0433, +0.0097] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.0429 / -0.0470 | -0.0020 | [-0.2429, +0.2517] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.2188 / -0.0752 | +0.0718 | [-0.5163, +0.7735] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.0179 / -0.0108 | -0.0143 | [-0.0359, +0.0067] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0896 / -0.0819 | +0.0038 | [-0.2248, +0.2796] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.1549 / -0.3482 | -0.0966 | [-0.6411, +0.6597] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | -0.0951 / -0.0390 | -0.0670 | [-0.1442, +0.0009] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0949 / -0.1879 | -0.1414 | [-0.3280, +0.0359] | 0.5140 (0.5140) | OPEN |

## I10RQ50h minus I10RQ50c

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0160 / +0.0107 | +0.0133 | [+0.0003, +0.0288] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | +0.0218 / +0.0047 | +0.0132 | [+0.0010, +0.0331] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0202 / +0.0171 | +0.0187 | [-0.0006, +0.0387] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0057 / +0.0416 | +0.0236 | [-0.0304, +0.0736] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.0616 / +0.4369 | +0.2493 | [-0.2348, +0.6945] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.1715 / +0.0849 | +0.1282 | [-0.7104, +0.9648] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0276 / +0.0507 | +0.0392 | [-0.0044, +0.0777] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0824 / +0.4751 | +0.2788 | [-0.1789, +0.6695] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.3480 / +0.2325 | +0.2902 | [-0.6053, +1.1411] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | -0.0197 / +0.0167 | -0.0015 | [-0.0382, +0.0275] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0594 / +0.0875 | +0.0735 | [-0.0332, +0.1759] | 0.5140 (0.5140) | OPEN |

## I10RQ50h minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0534 / -0.0445 | -0.0489 | [-0.0821, -0.0209] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0514 / -0.0202 | -0.0358 | [-0.0655, -0.0128] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1371 / +0.0872 | +0.1121 | [+0.0596, +0.1693] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5763 / +0.3760 | +0.4762 | [+0.2326, +0.7519] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.4561 / +1.5724 | +2.0142 | [+1.0123, +3.1558] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.2910 / +3.5685 | +4.9298 | [+2.4659, +7.7024] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4228 / +0.2511 | +0.3369 | [+0.1331, +0.5742] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.5217 / +1.0737 | +1.2977 | [+0.6376, +2.0310] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +3.2096 / +1.6714 | +2.4405 | [+0.8841, +4.2684] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0832 / +0.0589 | +0.0711 | [+0.0305, +0.1250] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.4350 / +0.2469 | +0.3409 | [+0.1786, +0.5575] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50h minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0463 / +0.0391 | +0.0427 | [+0.0200, +0.0680] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0249 / +0.0187 | +0.0218 | [+0.0082, +0.0382] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0483 / +0.0202 | +0.0343 | [+0.0024, +0.0705] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1300 / -0.0707 | +0.0297 | [-0.1593, +0.2428] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.5464 / +0.1910 | +0.3687 | [-0.0576, +0.9639] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +1.5537 / +0.3992 | +0.9764 | [-0.1589, +2.4668] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.1077 / -0.0650 | +0.0213 | [-0.1519, +0.2111] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.6625 / +0.3750 | +0.5188 | [+0.1447, +1.0497] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.7548 / +0.6904 | +1.2226 | [+0.2006, +2.5043] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0406 / +0.0133 | +0.0269 | [-0.0224, +0.0852] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.3315 / +0.1270 | +0.2293 | [+0.0664, +0.4268] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0409 | -0.0498 | [-0.0868, -0.0174] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0452 / -0.0202 | -0.0327 | [-0.0577, -0.0128] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1324 / +0.1059 | +0.1192 | [+0.0710, +0.1710] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6020 / +0.3845 | +0.4933 | [+0.2312, +0.7833] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.4132 / +1.6194 | +2.0163 | [+1.0353, +3.1714] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.0722 / +3.6437 | +4.8579 | [+2.5270, +7.5458] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4406 / +0.2619 | +0.3513 | [+0.1357, +0.5963] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.4321 / +1.1556 | +1.2939 | [+0.6447, +2.0721] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +3.0547 / +2.0196 | +2.5372 | [+1.1229, +4.2045] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1783 / +0.0979 | +0.1381 | [+0.0529, +0.2535] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.5299 / +0.4347 | +0.4823 | [+0.2771, +0.7058] | 0.5140 (0.5140) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50c (0,1) | 0.0676 [0.0425, 0.0969] | 0.0101 [0.0013, 0.0254] | 0.2336 [0.1715, 0.2965] | 0.2290 [-0.0037, 0.4971] | 2.2132 [1.4459, 3.1186] | 7.5956 [5.6614, 9.6262] | 0.7707 [0.5511, 1.0157] | 3.5110 [2.7391, 4.4854] | 9.0281 [7.3886, 10.8566] | 1.1161 [0.7670, 1.4997] | 2.3093 [1.6943, 3.0099] |
| I10RQ50h (0,1) | 0.0810 [0.0478, 0.1190] | 0.0234 [0.0096, 0.0397] | 0.2523 [0.1911, 0.3124] | 0.2526 [0.0301, 0.5181] | 2.4625 [1.6646, 3.3220] | 7.7238 [5.9036, 9.5807] | 0.8098 [0.5971, 1.0465] | 3.7897 [3.1218, 4.5743] | 9.3184 [7.8974, 10.8716] | 1.1146 [0.7685, 1.4959] | 2.3827 [1.7661, 3.0930] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
