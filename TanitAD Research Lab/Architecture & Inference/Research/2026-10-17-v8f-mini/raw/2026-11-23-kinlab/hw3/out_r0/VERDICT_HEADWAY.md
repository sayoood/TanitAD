# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 10, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.003558718861209967, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0KHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10UEA_R0C10XHb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RH_R0KHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RH R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / -0.0018 | +0.0027 | [-0.0035, +0.0106] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0016 / +0.0031 | +0.0008 | [-0.0048, +0.0064] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0109 / -0.0062 | -0.0086 | [-0.0162, -0.0006] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 | -0.0350 | [-0.0569, -0.0141] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.0565 / -0.1507 | -0.1036 | [-0.2004, -0.0095] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.1106 / -0.3363 | -0.2234 | [-0.4549, -0.0298] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0316 / +0.0219 | +0.0268 | [+0.0107, +0.0452] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0904 / +0.0547 | +0.0725 | [+0.0193, +0.1348] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.0379 / +0.0351 | +0.0365 | [-0.0700, +0.1447] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 | +0.0374 | [-0.0087, +0.0888] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0146 / -0.0607 | -0.0231 | [-0.1343, +0.0585] | 0.2075 (0.2075) | OPEN |

## I10RH R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / +0.0178 | +0.0178 | [+0.0064, +0.0313] | 0.0036 (0.0036) | WORSE |
| C contact rate | +0.0171 / +0.0125 | +0.0148 | [+0.0040, +0.0276] | 0.0062 (0.0062) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0685 / -0.0639 | -0.0662 | [-0.1025, -0.0380] | 0.0561 (0.0561) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 | -0.0350 | [-0.0569, -0.0141] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.2982 / -0.5005 | -0.3993 | [-0.5954, -0.2054] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.8277 / -2.1552 | -1.9914 | [-2.6824, -1.3897] | 1.8616 (1.8616) | BETTER |
| gap error 2.0 s (m) | +0.0316 / +0.0219 | +0.0268 | [+0.0107, +0.0452] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0711 / -0.1617 | -0.0453 | [-0.2437, +0.1545] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | -0.8786 / -1.3742 | -1.1264 | [-1.8312, -0.4986] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 | +0.0374 | [-0.0087, +0.0888] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.7011 / -0.7134 | -0.7072 | [-1.0008, -0.4429] | 0.2075 (0.2075) | BETTER |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10B0_R0C10XHb20 (0,1) | 0.0294 [0.0104, 0.0534] | 0.0210 [0.0073, 0.0374] | 0.2796 [0.2236, 0.3374] | 0.0891 [0.0465, 0.1353] | 2.1882 [1.7296, 2.6381] | 7.4618 [6.0483, 8.9499] | 0.4317 [0.3899, 0.4802] | 2.9698 [2.6130, 3.3670] | 8.6467 [7.5117, 9.8733] | 0.8992 [0.6263, 1.2159] | 3.8517 [2.8854, 4.8691] |
| I10B0_R0KHb20 (0,1) | 0.0925 [0.0563, 0.1324] | 0.0600 [0.0297, 0.0927] | 0.1145 [0.0783, 0.1567] | 0.0891 [0.0465, 0.1353] | 0.7204 [0.3444, 1.1110] | 1.7458 [0.5279, 3.0407] | 0.4317 [0.3899, 0.4802] | 2.3660 [2.0834, 2.6885] | 6.0590 [5.2991, 6.9059] | 0.8992 [0.6263, 1.2159] | 2.1774 [1.5055, 2.9188] |
| I10RH_R0C10XHb20 (0,1) | 0.0472 [0.0221, 0.0795] | 0.0358 [0.0137, 0.0613] | 0.2134 [0.1621, 0.2687] | 0.0541 [-0.0003, 0.1137] | 1.7888 [1.3557, 2.2578] | 5.4704 [4.1477, 6.8065] | 0.4584 [0.4066, 0.5113] | 2.9245 [2.5330, 3.3539] | 7.5203 [6.6391, 8.5576] | 0.9366 [0.6450, 1.2757] | 3.1445 [2.3022, 4.0131] |
| I10RH_R0KHb20 (0,1) | 0.0952 [0.0569, 0.1374] | 0.0607 [0.0306, 0.0944] | 0.1059 [0.0714, 0.1459] | 0.0541 [-0.0003, 0.1137] | 0.6168 [0.2197, 1.0551] | 1.5224 [0.2789, 2.8409] | 0.4584 [0.4066, 0.5113] | 2.4385 [2.1253, 2.7781] | 6.0955 [5.3116, 6.9523] | 0.9366 [0.6450, 1.2757] | 2.1543 [1.4999, 2.8825] |
| I10UEA_R0C10XHb20 (0,1) | 0.0623 [0.0388, 0.0910] | 0.0280 [0.0123, 0.0462] | 0.3910 [0.3191, 0.4592] | -0.2236 [-0.2757, -0.1682] | 3.0145 [2.3455, 3.6779] | 10.3479 [8.3469, 12.6031] | 0.4729 [0.4299, 0.5219] | 3.5797 [3.0625, 4.1116] | 11.7414 [10.0248, 13.7045] | 1.0435 [0.7059, 1.4212] | 4.8571 [3.8136, 5.9361] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
